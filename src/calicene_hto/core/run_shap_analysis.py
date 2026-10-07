"""Attribution analysis for Reviewer 1 Comments 16/17 and Reviewer 2 Comment 2.

Four things the historical SHAP analysis did not do:

  1. Attributions are computed OUT OF FOLD on the frozen round-1 splits, so every
     attribution comes from a model that never saw that parent molecule. Explaining
     an in-sample fit would describe memorisation, not transferable behaviour.
  2. Each task is attributed SEPARATELY, from its own model (R1-16).
  3. The 117 descriptors are heavily correlated, and Shapley values split credit
     arbitrarily among correlated inputs. Attributions are therefore aggregated to
     descriptor CLUSTERS obtained from |Spearman| distance on the feature matrix
     alone (no labels used, so the grouping cannot leak).
  4. Cross-fold rank stability is quantified (Kendall tau between folds). An
     unstable ranking cannot support a mechanistic claim (R1-17), and reliability is
     reported against an applicability-domain distance (R2-2).

Exact TreeSHAP comes from xgboost's native pred_contribs, so no extra dependency.
This describes MODEL BEHAVIOUR. It is not evidence of mechanism.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from label_definitions import TASKS, TARGETS, THRESHOLDS

GAPS = [TARGETS[t] for t in TASKS]
THR = np.array([THRESHOLDS[t] for t in TASKS], float)
# Documented default; replaced by the round-2 selection when that run exists.
DEFAULT_CFG = dict(max_depth=3, learning_rate=0.05, n_estimators=400, subsample=0.85,
                   colsample_bytree=0.7, min_child_weight=3.0, reg_lambda=1.0)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def load(args):
    data = pd.read_csv(args.data)
    df = data[data.intended_species == 1].reset_index(drop=True)
    blocks = json.loads(Path(args.blocks).read_text(encoding='utf-8-sig'))
    feats = blocks['inherited_94'] + blocks['conformer_block']
    x = df[feats].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    g = df[GAPS].to_numpy(float)
    y = df[TASKS].to_numpy(float)
    groups = df.Molecule.to_numpy()
    splits = json.loads((Path(args.round1) / 'splits.json').read_text(encoding='utf-8'))
    outer = [(np.array(s['train_rows']), np.array(s['test_rows'])) for s in splits]
    return df, feats, x, g, y, groups, outer


def prep(x, train, test):
    p = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    return p.fit_transform(x[train]), p.transform(x[test])


def cluster_features(x, feats, cut):
    """Correlation clusters from the feature matrix only. No labels involved."""
    z = SimpleImputer(strategy='median').fit_transform(x)
    const = [f for f, s in zip(feats, z.std(0)) if s == 0]
    if const:
        # Zero-variance descriptors carry no attribution and cluster arbitrarily.
        print('WARNING %d zero-variance descriptors: %s' % (len(const), ', '.join(const)))
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = spearmanr(z).correlation
    if np.ndim(rho) == 0:
        rho = np.array([[1.]])
    rho = np.nan_to_num(rho, nan=0.)
    d = 1 - np.abs(rho)
    np.fill_diagonal(d, 0.)
    d = (d + d.T) / 2
    link = linkage(squareform(d, checks=False), method='average')
    labels = fcluster(link, t=cut, criterion='distance')
    table = pd.DataFrame(dict(feature=feats, cluster=labels))
    # Name each cluster by its most central member (highest mean |rho| inside).
    names = {}
    for c, part in table.groupby('cluster'):
        idx = part.index.to_numpy()
        sub = np.abs(rho[np.ix_(idx, idx)])
        names[c] = feats[idx[int(np.argmax(sub.mean(1)))]]
    table['cluster_name'] = table.cluster.map(names)
    return table


def applicability(xtr, xte, k=5):
    """Mean distance to the k nearest training rows in standardized descriptor space."""
    d = np.sqrt(((xte[:, None, :] - xtr[None, :, :]) ** 2).sum(-1))
    k = min(k, d.shape[1])
    return np.sort(d, 1)[:, :k].mean(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--blocks', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--round2', default=None, help='use this run\'s selected XGB config')
    p.add_argument('--cluster-cut', dest='cut', type=float, default=0.3,
                   help='1-|rho| distance at which descriptor clusters are cut')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, feats, x, g, y, groups, outer = load(a)

    cfg = dict(DEFAULT_CFG)
    cfg_source = 'documented default (round 2 not available)'
    if a.round2:
        picks = sorted(Path(a.round2).glob('fold_*/XGB_*_selection.json'))
        if picks:
            cfg = json.loads(picks[0].read_text(encoding='utf-8'))['selected']
            cfg_source = str(picks[0])

    clusters = cluster_features(x, feats, a.cut)
    clusters.to_csv(out / 'descriptor_clusters.csv', index=False)
    n_cl = clusters.cluster.nunique()
    print('%d descriptors -> %d correlation clusters at 1-|rho| <= %.2f'
          % (len(feats), n_cl, a.cut))

    shap = {t: np.full((len(df), len(feats)), np.nan) for t in TASKS}
    ad = np.full(len(df), np.nan)
    oof = np.full((len(df), len(TASKS)), np.nan)
    per_fold = []
    for k, (tr, te) in enumerate(outer):
        xt, xe = prep(x, tr, te)
        ad[te] = applicability(xt, xe)
        for j, task in enumerate(TASKS):
            m = xgb.XGBRegressor(n_jobs=2, random_state=42, tree_method='hist', **cfg)
            m.fit(xt, g[tr, j])
            oof[te, j] = THR[j] - m.predict(xe)
            contrib = m.get_booster().predict(xgb.DMatrix(xe), pred_contribs=True)
            assert contrib.shape[1] == len(feats) + 1
            shap[task][te] = contrib[:, :-1]
            per_fold.append(dict(fold=k, task=task,
                                 mean_abs=dict(zip(feats, np.abs(contrib[:, :-1]).mean(0)))))
    assert np.isfinite(ad).all()
    for t in TASKS:
        assert np.isfinite(shap[t]).all()

    # Out-of-fold ranking quality of the very model being explained (R2-2 context).
    quality = []
    for j, task in enumerate(TASKS):
        quality.append(dict(task=task, OOF_AP=float(average_precision_score(y[:, j], oof[:, j])),
                            prevalence=float(y[:, j].mean())))
    print('\nOOF ranking quality of the explained model:')
    print(pd.DataFrame(quality).to_string(index=False))

    # Cluster-level attributions, per task (R1-16).
    rows = []
    cmap = clusters.set_index('feature')
    for task in TASKS:
        mabs = pd.Series(np.abs(shap[task]).mean(0), index=feats)
        signed = pd.Series(shap[task].mean(0), index=feats)
        for c, part in clusters.groupby('cluster'):
            f = part.feature.tolist()
            rows.append(dict(task=task, cluster=int(c), cluster_name=part.cluster_name.iloc[0],
                             n_features=len(f), mean_abs_shap=float(mabs[f].sum()),
                             mean_signed_shap=float(signed[f].sum()),
                             top_feature=str(mabs[f].idxmax())))
    imp = pd.DataFrame(rows)
    for task in TASKS:
        s = imp[imp.task == task].mean_abs_shap.sum()
        imp.loc[imp.task == task, 'share'] = imp.loc[imp.task == task, 'mean_abs_shap'] / s
    imp = imp.sort_values(['task', 'mean_abs_shap'], ascending=[True, False])
    imp.to_csv(out / 'cluster_attributions.csv', index=False)
    print('\ntop 8 descriptor clusters per task:')
    for task in TASKS:
        print(' ' + task)
        print(imp[imp.task == task].head(8)[
            ['cluster_name', 'n_features', 'mean_abs_shap', 'share']].to_string(index=False))

    # Cross-fold rank stability (R1-17).
    stab = []
    pf = pd.DataFrame(per_fold)
    for task in TASKS:
        mats = [pd.Series(r) for r in pf[pf.task == task].mean_abs.tolist()]
        taus = []
        for i in range(len(mats)):
            for j2 in range(i + 1, len(mats)):
                taus.append(kendalltau(mats[i][feats].to_numpy(),
                                       mats[j2][feats].to_numpy()).correlation)
        # Same at cluster level, which is the level any claim would be made at.
        cl = [s.groupby(cmap.loc[feats, 'cluster'].to_numpy()).sum() for s in
              [m[feats] for m in mats]]
        ctaus = []
        for i in range(len(cl)):
            for j2 in range(i + 1, len(cl)):
                ctaus.append(kendalltau(cl[i].to_numpy(), cl[j2].to_numpy()).correlation)
        stab.append(dict(task=task, pairs=len(taus),
                         feature_kendall_mean=float(np.mean(taus)),
                         feature_kendall_min=float(np.min(taus)),
                         cluster_kendall_mean=float(np.mean(ctaus)),
                         cluster_kendall_min=float(np.min(ctaus))))
    st = pd.DataFrame(stab)
    st.to_csv(out / 'attribution_stability.csv', index=False)
    print('\ncross-fold rank stability (Kendall tau between the 5 outer folds):')
    print(st.to_string(index=False))

    # Do the two tasks rely on the same descriptors? (R1-16)
    a_imp = imp[imp.task == TASKS[0]].set_index('cluster').mean_abs_shap
    b_imp = imp[imp.task == TASKS[1]].set_index('cluster').mean_abs_shap
    common = a_imp.index.intersection(b_imp.index)
    tau_tasks = kendalltau(a_imp[common].to_numpy(), b_imp[common].to_numpy()).correlation
    print('\ncluster-importance agreement BETWEEN the two tasks: Kendall tau = %.3f' % tau_tasks)

    # Attribution reliability against the applicability domain (R2-2).
    band = np.asarray(pd.qcut(ad, 3, labels=['near', 'mid', 'far']))
    dom = []
    for task in TASKS:
        j = TASKS.index(task)
        tot = np.abs(shap[task]).sum(1)
        for b in ['near', 'mid', 'far']:
            m = band == b
            ok = len(np.unique(y[m, j])) == 2
            prev = float(y[m, j].mean())
            v = float(average_precision_score(y[m, j], oof[m, j])) if ok else None
            # AP's baseline IS the prevalence, and prevalence differs sharply between
            # bands, so only the lift over baseline is comparable across bands -- and
            # even that is fragile at these positive counts.
            dom.append(dict(task=task, domain=b, n=int(m.sum()),
                            positives=int(y[m, j].sum()),
                            mean_AD_distance=float(ad[m].mean()),
                            total_abs_attribution=float(tot[m].mean()),
                            AP_within_band=v, prevalence=prev,
                            lift_over_prevalence=(v / prev) if (v and prev > 0) else None))
    dm = pd.DataFrame(dom)
    dm.to_csv(out / 'applicability_domain.csv', index=False)
    print('\nperformance and attribution mass by applicability-domain band:')
    print(dm.to_string(index=False))
    thin = dm[(dm.positives < 6) | dm.AP_within_band.isna()]
    if len(thin):
        print('UNDERPOWERED bands (<6 positives) -- do not interpret: %s'
              % ', '.join('%s/%s' % (r.task, r.domain) for r in thin.itertuples()))

    # What is inside the dominant cluster? Needed before any chemical reading.
    top = imp.sort_values('mean_abs_shap', ascending=False).iloc[0]
    members = clusters[clusters.cluster == top.cluster].feature.tolist()
    print('\nlargest attribution cluster "%s" (%.0f%% of %s attribution) contains: %s'
          % (top.cluster_name, 100 * top.share, top.task, ', '.join(members)))

    np.savez_compressed(out / 'oof_shap.npz', features=np.array(feats),
                        applicability=ad, oof_score=oof,
                        **{('shap_' + t): shap[t] for t in TASKS})
    write_json(out / 'shap_protocol.json', dict(
        config=cfg, config_source=cfg_source, cluster_cut=a.cut, n_clusters=int(n_cl),
        round1=str(a.round1), tasks=TASKS, targets=TARGETS, thresholds=THRESHOLDS,
        quality=quality, task_agreement_kendall=float(tau_tasks),
        note='Out-of-fold exact TreeSHAP on the frozen round-1 splits. Attributions '
             'describe model behaviour on unseen parent molecules. They are not '
             'evidence of mechanism and are reported at descriptor-cluster level '
             'because the inputs are strongly correlated.'))
    print('\nwrote %s' % out)


if __name__ == '__main__':
    main()
