"""Capacity diagnostics for Reviewer 2 Comment 1 and Reviewer 1 Comment 15.

These are DIAGNOSTICS, not model search: no family is added, dropped or reselected,
and nothing here can change which model the paper reports.

  capacity     effective sample size (ICC, design effect) and parameter counts
  scramble     parent-level y-randomisation; AP must collapse to prevalence
  curve        learning curve over parent subsets, with parent-bootstrap bands

Run on the frozen round-1 splits, same as run_round2_comparison.py.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from label_definitions import TASKS, TARGETS, THRESHOLDS, positive_mask
from run_round2_comparison import (GAPS, THR, FAMILIES, NETS, preprocess, fit_net,
                                   fit_flat, calibrate, ap, write_json, digest,
                                   sample_configs)


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
    return df, x, g, y, groups, outer


def icc_oneway(values, groups):
    """One-way random-effects ICC(1) for a clustered continuous outcome."""
    parents = np.unique(groups)
    k = len(parents)
    n = len(values)
    sizes = np.array([(groups == p).sum() for p in parents], float)
    grand = values.mean()
    means = np.array([values[groups == p].mean() for p in parents])
    msb = float((sizes * (means - grand) ** 2).sum() / max(k - 1, 1))
    msw = float(sum(((values[groups == p] - means[i]) ** 2).sum()
                    for i, p in enumerate(parents)) / max(n - k, 1))
    m0 = (n - (sizes ** 2).sum() / n) / max(k - 1, 1)
    denom = msb + (m0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else 0.0, msb, msw, float(m0)


def capacity(df, g, y, groups, out, round1):
    rows = []
    sizes = pd.Series(groups).value_counts()
    mbar = float(sizes.mean())
    for j, task in enumerate(TASKS):
        val, msb, msw, m0 = icc_oneway(g[:, j], groups)
        deff = 1 + (mbar - 1) * max(val, 0.)
        pos_parents = int(pd.Series(groups[y[:, j] > 0]).nunique())
        rows.append(dict(task=task, target=TARGETS[task], threshold=THRESHOLDS[task],
                         n_rows=len(df), n_parents=int(sizes.size),
                         conformers_per_parent=mbar, ICC1=val, MSB=msb, MSW=msw, m0=m0,
                         design_effect=deff, n_effective=len(df) / deff,
                         positives=int(y[:, j].sum()), positive_parents=pos_parents,
                         prevalence=float(y[:, j].mean())))
    cap = pd.DataFrame(rows)
    cap.to_csv(out / 'capacity.csv', index=False)
    print(cap.to_string(index=False))

    # Round-1 parameter counts against the effective sample size.
    meta = Path(round1) / 'audited_fit_metadata.csv'
    if meta.exists():
        m = pd.read_csv(meta)
        if 'parameters' in m and 'family' in m:
            s = (m[m.parameters > 0].groupby(['family', 'tasks']).parameters
                 .agg(['min', 'max', 'mean']).reset_index())
            s.to_csv(out / 'round1_parameter_counts.csv', index=False)
            print('\nround-1 parameter counts by family:')
            print(s.to_string(index=False))
    return cap


def one_pass(family, cfg, x, g, y, groups, outer, seeds, args, thr, js):
    """One nested-free pass on the frozen outer folds with a FIXED config."""
    oof = np.full((len(y), len(js)), np.nan)
    params = []
    for tr, te in outer:
        xt, xe, _ = preprocess(x, tr, te)
        acc = []
        for seed in seeds:
            if family in NETS:
                probs, meta = fit_net(cfg, xt, g[tr][:, js], y[tr][:, js], groups[tr], xe,
                                      groups[te], seed, args.max_epochs, args.patience, thr)
                acc.append(probs[max(probs)])
                params.append(meta['parameters'])
            else:
                sc, _ = fit_flat(family, cfg, xt, g[tr][:, js], xe, seed, thr)
                acc.append(sc)
        oof[te] = np.mean(acc, 0)
    return oof, (float(np.mean(params)) if params else 0.)


def scramble(x, g, y, groups, outer, args, out):
    """Parent-level y-randomisation: permute whole parents so clustering is preserved,
    then re-derive the labels from the permuted gaps."""
    rng = np.random.default_rng(20260913)
    cfgs = {f: sample_configs(f, 1, np.random.default_rng(7))[0] for f in args.families}
    rows = []
    for rep in range(args.reps):
        parents = np.unique(groups)
        perm = rng.permutation(len(parents))
        mapping = dict(zip(parents, parents[perm]))
        # Move each parent's whole block of gap values onto another parent's rows.
        blocks = {p: g[groups == p] for p in parents}
        gp = np.empty_like(g)
        for p in parents:
            src = blocks[mapping[p]]
            tgt = np.flatnonzero(groups == p)
            take = np.resize(np.arange(len(src)), len(tgt))
            gp[tgt] = src[take]
        yp = np.column_stack([positive_mask(gp[:, j], t).astype(float)
                              for j, t in enumerate(TASKS)])
        for family in args.families:
            js = [0, 1] if family == 'MT' else [0]
            if family != 'MT' and args.single_task_index is not None:
                js = [args.single_task_index]
            oof, _ = one_pass(family, cfgs[family], x, gp, yp, groups, outer,
                              [42], args, THR[js], js)
            for c, j in enumerate(js):
                v = ap(yp[:, j], oof[:, c])
                rows.append(dict(rep=rep, family=family, task=TASKS[j], AP=v,
                                 prevalence=float(yp[:, j].mean()),
                                 ratio=(v / float(yp[:, j].mean())) if v else None))
        print('scramble rep %d done' % rep, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / 'y_scramble.csv', index=False)
    print('\ny-scramble (AP should sit near prevalence, ratio near 1):')
    print(d.groupby(['family', 'task'])[['AP', 'prevalence', 'ratio']].mean().to_string())
    return d


def curve(x, g, y, groups, outer, args, out):
    rng = np.random.default_rng(20260913)
    cfgs = {f: sample_configs(f, 1, np.random.default_rng(7))[0] for f in args.families}
    parents = np.unique(groups)
    rows = []
    for frac in args.fractions:
        k = max(4, int(round(frac * len(parents))))
        for rep in range(args.reps):
            keep = set(rng.choice(parents, k, replace=False).tolist())
            sub_outer = []
            for tr, te in outer:
                t = tr[np.isin(groups[tr], list(keep))]
                if len(np.unique(groups[t])) < 3:
                    continue
                sub_outer.append((t, te))
            if len(sub_outer) < len(outer):
                continue
            for family in args.families:
                js = [0, 1] if family == 'MT' else [0]
                oof, params = one_pass(family, cfgs[family], x, g, y, groups, sub_outer,
                                       [42], args, THR[js], js)
                ok = np.isfinite(oof).all(1)
                for c, j in enumerate(js):
                    rows.append(dict(parents=k, rep=rep, family=family, task=TASKS[j],
                                     AP=ap(y[ok, j], oof[ok, c]), parameters=params))
            print('curve parents=%d rep=%d done' % (k, rep), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / 'learning_curve.csv', index=False)
    print('\nlearning curve (mean AP by parent count):')
    print(d.groupby(['family', 'task', 'parents']).AP.mean().to_string())
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--blocks', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--what', nargs='+', default=['capacity'],
                   choices=['capacity', 'scramble', 'curve'])
    p.add_argument('--families', nargs='+', default=['MT', 'XGB'])
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--fractions', nargs='+', type=float, default=[.2, .4, .6, .8, 1.])
    p.add_argument('--single-task-index', dest='single_task_index', type=int, default=None)
    p.add_argument('--max-epochs', dest='max_epochs', type=int, default=400)
    p.add_argument('--patience', type=int, default=40)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, x, g, y, groups, outer = load(a)
    write_json(out / 'diagnostics_protocol.json',
               dict(what=a.what, families=a.families, reps=a.reps, fractions=a.fractions,
                    round1=str(a.round1), data_sha256=digest(a.data),
                    script_sha256=digest(__file__),
                    note='Diagnostics only. No family is added, dropped or reselected; '
                         'nothing here changes which model the paper reports.'))
    started = time.time()
    if 'capacity' in a.what:
        capacity(df, g, y, groups, out, a.round1)
    if 'scramble' in a.what:
        scramble(x, g, y, groups, outer, a, out)
    if 'curve' in a.what:
        curve(x, g, y, groups, outer, a, out)
    write_json(out / 'DIAGNOSTICS_COMPLETE.json',
               dict(what=a.what, elapsed_seconds=time.time() - started))
    print('\nelapsed %.1fs' % (time.time() - started))


if __name__ == '__main__':
    main()
