"""Torsion / HTO analysis for Reviewer 2 Comment 4 and Reviewer 1 Comments 22, 23, 24.

The reviewers' objection is that a dataset-wide association between torsional freedom
and HTO is confounded by chemical substitution, and that a molecular rotatable-bond
count is not a conformer's actual torsion. Both are answered by splitting every
torsion variable into the part that varies BETWEEN parent molecules (confounded with
substitution) and the part that varies WITHIN a parent (pure conformational change,
substituents held fixed) -- the Mundlak within-between decomposition.

  y_ij = a + b_within * (t_ij - mean_i(t)) + b_between * mean_i(t) + e_ij

b_within is the conformational effect the paper needs. b_between carries the
substitution confound. Uncertainty resamples PARENT MOLECULES as clusters.

Primary outcomes are the continuous signed gaps; HTO switching is secondary (R1-23),
and Boltzmann populations of HTO-positive conformers are reported for R1-24.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from label_definitions import TASKS, TARGETS, THRESHOLDS

GAPS = [TARGETS[t] for t in TASKS]
# Per-conformer geometric coordinates (vary within a parent) plus the molecular
# rotatable-bond count the reviewers single out (cannot vary within a parent).
TORSIONS = ['interring_twist_max_deg', 'interring_twist_mean_deg',
            'rot_torsion_max_dev_deg', 'rot_torsion_mean_dev_deg',
            'substituent_twist_max_deg', 'substituent_twist_mean_deg',
            'heteroatom_substituent_twist_max_deg', 'amine_pyramidalization_max_deg',
            'planarity_rmsd_heavy_A', 'planarity_maxdev_heavy_A', 'planarity_ratio',
            'num_rotatable_bonds']
R_KCAL = 0.0019872041
T_K = 298.15


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def within_between(df, t, outcome, groups):
    """OLS on [1, within, between]. Returns slopes in outcome-units per t-unit."""
    tv = df[t].to_numpy(float)
    yv = df[outcome].to_numpy(float)
    m = pd.Series(tv).groupby(groups).transform('mean').to_numpy()
    w = tv - m
    X = np.column_stack([np.ones(len(tv)), w, m])
    if np.std(w) < 1e-12:                      # molecular variable: no within part
        X = np.column_stack([np.ones(len(tv)), m])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        return dict(b_within=None, b_between=float(beta[1]))
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return dict(b_within=float(beta[1]), b_between=float(beta[2]))


def cluster_bootstrap(df, t, outcome, groups, n=2000, seed=20260913):
    """Resampling takes WHOLE parents, so each parent's within-deviation and its
    between-mean are invariant under resampling. Precompute them once per parent and
    a draw becomes block concatenation plus one 3x3 solve."""
    rng = np.random.default_rng(seed)
    parents = np.unique(groups)
    tv = df[t].to_numpy(float)
    yv = df[outcome].to_numpy(float)
    blocks = []
    for p in parents:
        idx = np.flatnonzero(groups == p)
        tp = tv[idx]
        mp = tp.mean()
        blocks.append((tp - mp, np.full(len(idx), mp), yv[idx]))
    molecular = np.std(np.concatenate([b[0] for b in blocks])) < 1e-12
    wi, be = [], []
    for _ in range(n):
        pick = rng.integers(0, len(parents), len(parents))
        w = np.concatenate([blocks[k][0] for k in pick])
        m = np.concatenate([blocks[k][1] for k in pick])
        y = np.concatenate([blocks[k][2] for k in pick])
        X = np.column_stack([np.ones(len(y)), m]) if molecular \
            else np.column_stack([np.ones(len(y)), w, m])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if molecular:
            be.append(float(beta[1]))
        else:
            wi.append(float(beta[1]))
            be.append(float(beta[2]))
    q = lambda v: (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if len(v) else (None, None)
    return q(wi), q(be), len(wi), len(be)


def variance_split(df, t, groups):
    g = df.groupby(groups)[t]
    return float(g.std().fillna(0.).mean()), float(g.mean().std())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.data)
    df = d[d.intended_species == 1].reset_index(drop=True)
    groups = df.Molecule.to_numpy()
    present = [t for t in TORSIONS if t in df.columns]
    missing = [t for t in TORSIONS if t not in df.columns]
    if missing:
        print('missing torsion columns: %s' % ', '.join(missing))

    # ---- 1. How much torsional variation is conformational at all? (R1-22, R2-4)
    rows = []
    for t in present:
        w, b = variance_split(df, t, groups)
        rows.append(dict(variable=t, within_parent_sd=w, between_parent_sd=b,
                         ratio_between_over_within=(b / w) if w > 1e-12 else np.inf,
                         conformational=bool(w > 1e-12)))
    var = pd.DataFrame(rows).sort_values('ratio_between_over_within', ascending=False)
    var.to_csv(out / 'torsion_variance_split.csv', index=False)
    print('\nvariation of each torsion variable, within vs between parent molecules:')
    print(var.to_string(index=False))

    # ---- 2. Within vs between association with the continuous gaps (R2-4)
    rows = []
    for t in present:
        for outcome in GAPS:
            est = within_between(df, t, outcome, groups)
            (wl, wh), (bl, bh), nw, nb = cluster_bootstrap(df, t, outcome, groups, a.bootstrap)
            rows.append(dict(variable=t, outcome=outcome,
                             b_within=est['b_within'], within_lo=wl, within_hi=wh,
                             within_excludes_zero=(None if est['b_within'] is None
                                                   else bool(wl is not None and (wl > 0) == (wh > 0))),
                             b_between=est['b_between'], between_lo=bl, between_hi=bh,
                             between_excludes_zero=bool(bl is not None and (bl > 0) == (bh > 0)),
                             draws_within=nw, draws_between=nb))
    assoc = pd.DataFrame(rows)
    assoc.to_csv(out / 'torsion_gap_association.csv', index=False)
    print('\nwithin-parent (conformational) vs between-parent (substitution-confounded) slopes:')
    show = assoc[['variable', 'outcome', 'b_within', 'within_excludes_zero',
                  'b_between', 'between_excludes_zero']]
    for outcome in GAPS:
        print(' outcome = %s' % outcome)
        print(show[show.outcome == outcome].drop(columns='outcome').to_string(index=False))

    # ---- 3. Same-parent HTO switching (R1-23)
    sw = []
    for parent, part in df.groupby('Molecule'):
        for task in TASKS:
            if part[task].nunique() > 1:
                pos = part[part[task] == 1]
                neg = part[part[task] == 0]
                col = TARGETS[task]
                sw.append(dict(Molecule=parent, task=task, n_conformers=len(part),
                               n_positive=int(part[task].sum()),
                               gap_min=float(part[col].min()), gap_max=float(part[col].max()),
                               gap_range=float(part[col].max() - part[col].min()),
                               threshold=THRESHOLDS[task],
                               twist_pos=float(pos.interring_twist_max_deg.mean()),
                               twist_neg=float(neg.interring_twist_max_deg.mean())))
    switch = pd.DataFrame(sw)
    switch.to_csv(out / 'same_parent_switching.csv', index=False)
    print('\nsame-parent label switching:')
    if len(switch):
        for task in TASKS:
            s = switch[switch.task == task]
            print('  %s: %d of %d parents switch' % (task, len(s), df.Molecule.nunique()))
        print(switch.to_string(index=False))
    else:
        print('  none')

    # ---- 4. Boltzmann populations of HTO-positive conformers (R1-24)
    boltz = []
    has_w = 'w_G' in df.columns and 'dG_kcal' in df.columns
    for parent, part in df.groupby('Molecule'):
        rec = dict(Molecule=parent, n_conformers=len(part))
        if has_w:
            wsum = float(part.w_G.sum())
            rec['retained_population_G'] = wsum
            for task in TASKS:
                pos = float(part.loc[part[task] == 1, 'w_G'].sum())
                rec['pop_' + task] = pos
                rec['pop_' + task + '_renorm'] = (pos / wsum) if wsum > 1e-9 else None
        # Recomputed from dG within the retained conformers only, as a cross-check.
        if 'dG_kcal' in part:
            e = part.dG_kcal.to_numpy(float)
            w = np.exp(-(e - e.min()) / (R_KCAL * T_K))
            w = w / w.sum()
            for task in TASKS:
                rec['recomputed_' + task] = float(w[part[task].to_numpy() == 1].sum())
        boltz.append(rec)
    bz = pd.DataFrame(boltz)
    bz.to_csv(out / 'boltzmann_populations.csv', index=False)
    if has_w:
        valid = bz[bz.retained_population_G > 1e-6]
        print('\nBoltzmann populations (Gibbs, %.2f K):' % T_K)
        print('  parents with usable stored weights: %d of %d' % (len(valid), len(bz)))
        print('  retained population per parent: median %.3f, min %.3f, max %.3f'
              % (valid.retained_population_G.median(), valid.retained_population_G.min(),
                 valid.retained_population_G.max()))
        for task in TASKS:
            col = 'recomputed_' + task
            n_any = int((bz[col] > 0.01).sum())
            n_maj = int((bz[col] > 0.5).sum())
            print('  %s: %d parents with >1%% population positive, %d with >50%%'
                  % (task, n_any, n_maj))

    write_json(out / 'torsion_protocol.json', dict(
        variables=present, missing=missing, outcomes=GAPS, tasks=TASKS,
        thresholds=THRESHOLDS, bootstrap=a.bootstrap, temperature_K=T_K,
        method='Mundlak within-between OLS; parent-cluster bootstrap for uncertainty',
        note='Primary outcomes are the continuous signed gaps. Within-parent slopes '
             'hold substitution fixed; between-parent slopes carry the substitution '
             'confound the reviewers raise. Stored Boltzmann weights were computed over '
             'the full conformer search, so retained rows need not sum to 1; the '
             'retained fraction is reported and a within-retained recomputation is '
             'given alongside.'))
    print('\nwrote %s' % out)


if __name__ == '__main__':
    main()
