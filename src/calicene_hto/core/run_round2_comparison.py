"""Round-2 prespecified nested comparison on the FROZEN round-1 outer folds.

Every family predicts the same two continuous gaps and is scored after applying the
same prespecified thresholds. Tuning budget is identical across families (fixed-seed
random search, --configs draws each). Outer and inner folds are loaded from the
round-1 splits.json and are never recomputed, so no split shopping is possible.

Round 1 remains valid and must be reported alongside this run.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             matthews_corrcoef, brier_score_loss)
from xgboost import XGBRegressor
from label_definitions import TASKS, TARGETS, THRESHOLDS, STRICT, FORBIDDEN, positive_mask
from optimized_architectures import (GapNet, standardize_targets, class_weights,
                                     masked_huber, weighted_bce, parent_batches, mixup)

GAPS = [TARGETS[t] for t in TASKS]
THR = np.array([THRESHOLDS[t] for t in TASKS], float)
FAMILIES = ['MT', 'ST', 'XGB', 'RF', 'Ridge']
NETS = ('MT', 'ST')
ALPHAS = [0., .25, .5, .75, 1.]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_configs(family, n, rng):
    out = []
    for _ in range(n):
        if family in NETS:
            out.append(dict(width=int(rng.choice([32, 64, 128])),
                            dropout=float(rng.choice([.1, .2, .3])),
                            wd=float(10 ** rng.uniform(-4, -2)),
                            lr=float(10 ** rng.uniform(-3.2, -2.2)),
                            private=int(rng.choice([0, 8, 16])),
                            context=int(rng.choice([0, 16])),
                            mixup=float(rng.choice([0., .2, .4])),
                            batch_parents=int(rng.choice([8, 16, 32])),
                            cls_weight=float(rng.choice([0., .3, 1.]))))
        elif family == 'XGB':
            out.append(dict(max_depth=int(rng.choice([2, 3, 4])),
                            learning_rate=float(10 ** rng.uniform(-1.8, -0.8)),
                            n_estimators=int(rng.choice([200, 400, 800])),
                            subsample=float(rng.choice([.7, .85, 1.])),
                            colsample_bytree=float(rng.choice([.5, .7, .9])),
                            min_child_weight=float(rng.choice([1., 3., 5.])),
                            reg_lambda=float(10 ** rng.uniform(-1, 1.5))))
        elif family == 'RF':
            out.append(dict(n_estimators=500,
                            max_features=float(rng.choice([.2, .3, .5, .8])),
                            min_samples_leaf=int(rng.choice([1, 2, 3, 5, 8]))))
        else:
            out.append(dict(alpha=float(10 ** rng.uniform(-2, 3))))
    return out


def preprocess(x, train, test):
    # keep_empty_features matches round 1 so the feature count cannot drift between folds.
    p = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                  ('scale', StandardScaler())])
    return p.fit_transform(x[train]), p.transform(x[test]), p


def fit_net(cfg, xtr, gtr, ytr, groups_tr, xte, groups_te, seed, max_epochs, patience, thr):
    """Early stopping on a parent-disjoint hold-out carved from the training rows only.

    thr holds the prespecified threshold of EACH selected task, in the same column order
    as gtr/ytr. It must never be indexed positionally from the global THR array: the
    single-task HTO group is column 0 locally but threshold 0.0, not 0.4.
    """
    thr = np.asarray(thr, float)
    n_task = len(thr)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    parents = np.unique(groups_tr)
    n_es = max(1, int(round(.15 * len(parents))))
    es_par = rng.choice(parents, n_es, replace=False)
    es = np.isin(groups_tr, es_par)
    fit = ~es
    if fit.sum() < 8 or es.sum() < 2:
        fit = np.ones(len(xtr), bool)
        es = fit
    mu, sd = standardize_targets(gtr[fit])
    t_std = torch.tensor((thr - mu) / sd, dtype=torch.float32)
    mask = np.isfinite(gtr).astype(float)
    G = torch.tensor(np.nan_to_num((gtr - mu) / sd), dtype=torch.float32)
    M = torch.tensor(mask, dtype=torch.float32)
    Y = torch.tensor(ytr, dtype=torch.float32)
    X = torch.tensor(xtr, dtype=torch.float32)
    gid = torch.tensor(pd.factorize(groups_tr)[0], dtype=torch.long)
    pw = class_weights(Y[fit])
    use_cls = cfg['cls_weight'] > 0
    m = GapNet(xtr.shape[1], width=cfg['width'], dropout=cfg['dropout'], n_task=n_task,
               private=cfg['private'], context=cfg['context'], cls_head=use_cls)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    fit_idx = np.flatnonzero(fit)
    es_idx = np.flatnonzero(es)
    best, best_state, bad, epoch = np.inf, None, 0, 0
    for epoch in range(max_epochs):
        m.train()
        for b in parent_batches(groups_tr[fit_idx], cfg['batch_parents'], rng):
            idx = fit_idx[b]
            gsel = gid[idx] if cfg['context'] else None
            xb, gb, yb, mb, _ = mixup(X[idx], G[idx], Y[idx], M[idx], cfg['mixup'],
                                      rng, X.device, groups=gsel)
            reg, cls = m(xb, gsel)
            loss = masked_huber(reg, gb, mb).mean()
            if cls is not None:
                loss = loss + cfg['cls_weight'] * weighted_bce(cls, yb, pw).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            reg, _ = m(X[es_idx], gid[es_idx] if cfg['context'] else None)
            v = float(masked_huber(reg, G[es_idx], M[es_idx]).mean())
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        ctx = torch.tensor(pd.factorize(groups_te)[0], dtype=torch.long) if cfg['context'] else None
        reg, cls = m(torch.tensor(xte, dtype=torch.float32), ctx)
        alphas = ALPHAS if cls is not None else [1.]
        probs = {a: m.probabilities(reg, cls, t_std, a).numpy() for a in alphas}
    meta = dict(parameters=int(sum(p.numel() for p in m.parameters())), epochs=epoch + 1,
                es_parents=int(n_es), best_val=best)
    return probs, meta


def fit_flat(family, cfg, xtr, gtr, xte, seed, thr):
    """Baselines receive the SAME continuous targets and the SAME thresholds.

    thr is per selected task, in gtr's column order -- see the note in fit_net.
    Returns a ranking score (threshold minus predicted gap); larger means more likely
    to be a positive. Probabilities come from inner-OOF Platt scaling.
    """
    thr = np.asarray(thr, float)
    score = np.empty((len(xte), len(thr)))
    for j in range(len(thr)):
        ok = np.isfinite(gtr[:, j])
        if family == 'XGB':
            m = XGBRegressor(n_jobs=2, random_state=seed, tree_method='hist', **cfg)
        elif family == 'RF':
            m = RandomForestRegressor(n_jobs=2, random_state=seed, **cfg)
        else:
            m = Ridge(**cfg)
        m.fit(xtr[ok], gtr[ok, j])
        score[:, j] = thr[j] - m.predict(xte)
    return score, dict(parameters=0)


def calibrate(score_inner, y_inner, score_outer):
    """Platt scaling fitted on inner OOF scores only."""
    out = np.empty_like(score_outer)
    for j in range(score_outer.shape[1]):
        y = y_inner[:, j].astype(int)
        if len(np.unique(y)) < 2:
            out[:, j] = 1. / (1. + np.exp(-score_outer[:, j]))
            continue
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(score_inner[:, j:j + 1], y)
        out[:, j] = lr.predict_proba(score_outer[:, j:j + 1])[:, 1]
    return out


def ap(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None


def macro_ap(y, p):
    v = [ap(y[:, j], p[:, j]) for j in range(y.shape[1])]
    if any(t is None for t in v):
        return None, v
    return float(np.mean(v)), v


def threshold_from(y, p):
    if len(np.unique(y)) < 2:
        return .5
    ts = np.unique(np.r_[.5, p])
    return float(max(ts, key=lambda t: (f1_score(y, p >= t, zero_division=0), -abs(t - .5), -t)))


def select_candidate(family, configs, x, g, y, groups, inner, seeds, args, thr):
    """Model selection on pooled inner OOF only. Returns config, alpha, inner OOF."""
    n = len(y)
    n_task = len(thr)
    best = None
    records, scores = [], []
    for k, cfg in enumerate(configs):
        # A net without a classifier head yields only the regression-derived probability.
        alphas = ALPHAS if (family in NETS and cfg['cls_weight'] > 0) else [1.]
        acc = {a: np.full((len(seeds), n, n_task), np.nan) for a in alphas}
        for fold, (tr, va) in enumerate(inner):
            a_tr, a_va, _ = preprocess(x, tr, va)
            for s, seed in enumerate(seeds):
                if family in NETS:
                    probs, meta = fit_net(cfg, a_tr, g[tr], y[tr], groups[tr], a_va, groups[va],
                                          seed, args.max_epochs, args.patience, thr)
                    for a, p in probs.items():
                        acc[a][s, va] = p
                else:
                    sc, meta = fit_flat(family, cfg, a_tr, g[tr], a_va, seed, thr)
                    acc[1.][s, va] = sc
                records.append(dict(candidate=k, fold=fold, seed=seed, **meta))
        entry = dict(candidate=k, config=cfg, by_alpha={})
        for a, arr in acc.items():
            if not np.isfinite(arr).all():
                raise ValueError('Inner OOF incomplete')
            m_ap, per = macro_ap(y, arr.mean(0))
            if m_ap is None:
                raise ValueError('Pooled inner labels single-class; cannot select by AP')
            entry['by_alpha'][str(a)] = dict(macro_AP=m_ap, task_AP=per)
            if best is None or m_ap > best[0]:
                best = (m_ap, k, a, arr.mean(0))
        scores.append(entry)
    _, k, alpha, inner_oof = best
    return configs[k], alpha, scores, records, inner_oof


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--blocks', required=True)
    p.add_argument('--round1', required=True, help='round-1 run dir supplying the frozen splits')
    p.add_argument('--out', required=True)
    p.add_argument('--configs', type=int, default=24, help='identical tuning budget per family')
    p.add_argument('--members', type=int, default=5, help='ensemble members / seeds')
    p.add_argument('--max-epochs', dest='max_epochs', type=int, default=400)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    r1 = Path(args.round1)

    data = pd.read_csv(args.data)
    df = data[data.intended_species == 1].reset_index(drop=True)
    blocks = json.loads(Path(args.blocks).read_text(encoding='utf-8-sig'))
    features = blocks['inherited_94'] + blocks['conformer_block']
    assert len(features) == len(set(features)) == 117
    assert not set(features) & FORBIDDEN
    assert not df.duplicated(['Molecule', 'conformer']).any()

    x = df[features].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    g = df[GAPS].to_numpy(float)
    y = df[TASKS].to_numpy(float)
    groups = df.Molecule.to_numpy()
    assert np.isin(y, [0, 1]).all()
    # The labels must remain exact thresholdings of the continuous targets.
    for j, task in enumerate(TASKS):
        assert (positive_mask(g[:, j], task).astype(int) == y[:, j].astype(int)).all(), task

    # Frozen splits from round 1. Row order is identical (same filter, same reset_index).
    r1_rows = pd.read_csv(r1 / 'rows.csv')
    assert (r1_rows.Molecule.to_numpy() == groups).all()
    assert (r1_rows.conformer.to_numpy() == df.conformer.to_numpy()).all()
    splits = json.loads((r1 / 'splits.json').read_text(encoding='utf-8'))
    outer = [(np.array(s['train_rows']), np.array(s['test_rows'])) for s in splits]
    inner_all = [[(np.array(i['train_rows']), np.array(i['val_rows'])) for i in s['inner']] for s in splits]

    seeds = [42, 123, 456, 789, 1011][:args.members]
    if args.smoke:
        seeds = seeds[:1]
        args.configs, args.max_epochs, args.patience = 2, 6, 3
    rng = np.random.default_rng(20260913)
    configs = {f: sample_configs(f, args.configs, rng) for f in FAMILIES}

    protocol = dict(
        scope='ROUND 2 prespecified nested CV on frozen round-1 outer folds. Round 1 remains reported.',
        formulation='Continuous-gap regression for ALL families; class probabilities derived at the '
                    'prespecified label thresholds (small_gap 0.4 eV, HTO 0.0 eV). Labels unchanged.',
        round1_dir=str(r1), round1_protocol_sha256=digest(r1 / 'protocol.json'),
        round1_splits_sha256=digest(r1 / 'splits.json'),
        data_sha256=digest(args.data), blocks_sha256=digest(args.blocks),
        script_sha256=digest(__file__),
        architectures_sha256=digest(Path(__file__).with_name('optimized_architectures.py')),
        features=features, thresholds=THRESHOLDS, targets=TARGETS,
        tuning_budget=dict(configs_per_family=args.configs, members=args.members,
                           max_epochs=args.max_epochs, patience=args.patience,
                           note='identical draw count for every family; fixed-seed random search'),
        configs=configs, seeds=seeds, n=len(df), parents=int(df.Molecule.nunique()),
        smoke=args.smoke,
        selection='pooled inner OOF macro AP; blend alpha selected on inner OOF only',
        guardrail='Outer predictions are never used for any choice. Report regardless of outcome.')
    write_json(out / 'protocol.json', protocol)
    df[['Molecule', 'conformer'] + TASKS + GAPS].to_csv(out / 'rows.csv', index_label='row')

    started = time.time()
    try:
        for k, (tr, te) in enumerate(outer):
            fold_out = out / ('fold_%d' % k)
            fold_out.mkdir()
            # The frozen round-1 splits store GLOBAL row indices (0..n-1), but select_candidate is
            # handed arrays already restricted to this fold's outer-training rows (x[tr], g[tr], ...)
            # and sizes its inner-OOF accumulator as len(y[tr]). The inner indices therefore have to
            # be remapped to positions within tr; passing them unremapped raised
            # "IndexError: index 138 is out of bounds for axis 0 with size 130" on the first fold.
            # Checked for all 5 folds: every inner fold is an exact partition of that fold's outer
            # training rows, so the remapping is well defined and loses nothing.
            pos = {int(r): i for i, r in enumerate(tr)}
            inner = []
            for a, b in inner_all[k]:
                assert set(a.tolist()) | set(b.tolist()) == set(int(r) for r in tr), \
                    'inner fold is not a partition of the outer training rows (fold %d)' % k
                inner.append((np.array([pos[int(r)] for r in a]),
                              np.array([pos[int(r)] for r in b])))
            xt, xe, _ = preprocess(x, tr, te)
            rows, fits = [], []
            for family in FAMILIES:
                print('fold=%d family=%s elapsed=%.1fs' % (k, family, time.time() - started), flush=True)
                write_json(out / 'status.json', dict(status='running', fold=k, family=family,
                                                     elapsed_seconds=time.time() - started))
                groups_of = [[0, 1]] if family == 'MT' else [[0], [1]]
                for js in groups_of:
                    nt = len(js)
                    thr_js = THR[js]          # per selected task; never positional on THR
                    cfg, alpha, scores, recs, inner_oof = select_candidate(
                        family, configs[family], x[tr], g[tr][:, js], y[tr][:, js], groups[tr],
                        inner, seeds, args, thr_js)
                    tag = family + '_' + '-'.join(map(str, js))
                    write_json(fold_out / (tag + '_selection.json'),
                               dict(selected=cfg, alpha=alpha, scores=scores, fits=recs))
                    np.savez_compressed(fold_out / (tag + '_inner_oof.npz'),
                                        inner_oof=inner_oof, rows=tr)
                    outer_raw = []
                    for seed in seeds:
                        if family in NETS:
                            probs, meta = fit_net(cfg, xt, g[tr][:, js], y[tr][:, js], groups[tr],
                                                  xe, groups[te], seed, args.max_epochs,
                                                  args.patience, thr_js)
                            outer_raw.append(probs[alpha])
                        else:
                            sc, meta = fit_flat(family, cfg, xt, g[tr][:, js], xe, seed, thr_js)
                            outer_raw.append(sc)
                        fits.append(dict(family=family, tasks=js, seed=seed, selected=cfg,
                                         alpha=alpha, **meta))
                    pred = np.mean(outer_raw, 0)
                    ref = inner_oof
                    if family not in NETS:
                        # Platt scaling and the operating threshold both come from inner OOF only.
                        pred = calibrate(inner_oof, y[tr][:, js], pred)
                        ref = calibrate(inner_oof, y[tr][:, js], inner_oof)
                    thr = [threshold_from(y[tr][:, js][:, c], ref[:, c]) for c in range(nt)]
                    if not np.isfinite(pred).all():
                        raise ValueError('Nonfinite outer prediction')
                    for c, j in enumerate(js):
                        for n_i, idx in enumerate(te):
                            rows.append(dict(fold=k, family=family, row=int(idx),
                                             Molecule=groups[idx], conformer=df.conformer.iloc[idx],
                                             task=TASKS[j], y=int(y[idx, j]),
                                             probability=float(pred[n_i, c]), threshold=thr[c]))
            pd.DataFrame(rows).to_csv(fold_out / 'predictions.csv', index=False)
            write_json(fold_out / 'fit_metadata.json', fits)
        summarize(out)
        write_json(out / 'COMPLETE.json', dict(scope=protocol['scope'], smoke=args.smoke,
                                               elapsed_seconds=time.time() - started))
        write_json(out / 'status.json', dict(status='completed', elapsed_seconds=time.time() - started))
    except Exception as e:
        write_json(out / 'FAILED.json', dict(error=repr(e), elapsed_seconds=time.time() - started))
        raise


def summarize(out, bootstrap=2000):
    preds = pd.concat([pd.read_csv(f) for f in sorted(out.glob('fold_*/predictions.csv'))],
                      ignore_index=True)
    preds.to_csv(out / 'all_outer_predictions.csv', index=False)
    metrics = []
    for (family, task), s in preds.groupby(['family', 'task']):
        both = s.y.nunique() == 2
        for rule in ['fixed_0.5', 'inner_OOF_F1']:
            lab = s.probability >= (.5 if rule == 'fixed_0.5' else s.threshold)
            metrics.append(dict(family=family, task=task, threshold_rule=rule,
                                AP=ap(s.y, s.probability),
                                AUC=float(roc_auc_score(s.y, s.probability)) if both else None,
                                F1=float(f1_score(s.y, lab, zero_division=0)),
                                MCC=float(matthews_corrcoef(s.y, lab)),
                                Brier=float(brier_score_loss(s.y, s.probability)),
                                prevalence=float(s.y.mean()), n=len(s)))
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)

    # Full pairwise parent-cluster bootstrap, not only MT versus the rest.
    fams = sorted(preds.family.unique())
    ci = []
    for i, a in enumerate(fams):
        for b in fams[i + 1:]:
            j = preds[preds.family == a].merge(
                preds[preds.family == b],
                on=['row', 'task', 'y', 'Molecule', 'conformer', 'fold'],
                suffixes=('_a', '_b'), validate='one_to_one')
            parents = j.Molecule.unique()
            index = {p: np.flatnonzero(j.Molecule.to_numpy() == p) for p in parents}
            rng = np.random.default_rng(20260913)
            draws = []
            for _ in range(bootstrap):
                ids = np.concatenate([index[p] for p in rng.choice(parents, len(parents), replace=True)])
                sample = j.iloc[ids]
                vals = []
                for task in TASKS:
                    t = sample[sample.task == task]
                    pa, pb = ap(t.y, t.probability_a), ap(t.y, t.probability_b)
                    vals.append(None if pa is None or pb is None else pa - pb)
                if all(v is not None for v in vals):
                    draws.append(vals + [float(np.mean(vals))])
            arr = np.asarray(draws)
            for c, task in enumerate(TASKS + ['macro_AP']):
                v = arr[:, c] if len(arr) else np.array([])
                ci.append(dict(comparison=a + '-' + b, task=task,
                               low=float(np.quantile(v, .025)) if len(v) else None,
                               high=float(np.quantile(v, .975)) if len(v) else None,
                               p_rope_0p02=float(np.mean(np.abs(v) < .02)) if len(v) else None,
                               valid=len(v), attempts=bootstrap,
                               limitation='Conditional on fitted OOF predictions; not a full '
                                          'retraining bootstrap; development dataset already examined'))
    pd.DataFrame(ci).to_csv(out / 'paired_parent_bootstrap.csv', index=False)


if __name__ == '__main__':
    main()
