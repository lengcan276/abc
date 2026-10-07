"""Within-parent (conformer-level) association between ground-state descriptors and excited-state
ordering, controlling for chemical substitution (R1-22, R1-23, R2-4).

For every descriptor that varies within parents:
  * parent fixed-effect correlation: Pearson r between parent-demeaned x and parent-demeaned y
    over all conformers of multi-conformer parents; two-sided p from within-parent permutation
    (y shuffled inside each parent, 5000 draws)
  * consistency: number of parents (>=3 conformers) in which within-parent Spearman rho has the
    pooled sign
Targets: continuous gap_S1_T2 (primary), gap_S1_T1, binary HTO.
Also: switching-parent table (parents with both HTO and non-HTO conformers), paired HTO-minus-nonHTO
descriptor differences per parent with a sign test across parents.
Outputs phase2/data/model_v2/within_parent_effects_v2.csv, switching_parents_v2.csv, reports/within_parent_v2_summary.json
"""
import json, argparse, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

P2 = Path(os.environ.get('CALICENE_HTO_ROOT', '.')).resolve() / 'revision_CAJ_20260910/phase2'
ap = argparse.ArgumentParser(); ap.add_argument('--intended_only', action='store_true'); args = ap.parse_args()
d = pd.read_csv(P2 / 'data/model_v2/dataset_v2.csv')
SUF = ''
if args.intended_only:
    d = d[d.intended_species == 1].copy(); SUF = '_intended'
blocks = json.loads((P2 / 'data/model_v2/feature_blocks.json').read_text())
feats = [c for c in blocks['conformer_varying_within_parent'] if c not in ('w_E', 'w_G')]
multi = d.groupby('Molecule').filter(lambda g: len(g) >= 2).copy()
rng = np.random.default_rng(20260910)
NPERM = 5000


def demean(df, cols):
    return df[cols] - df.groupby('Molecule')[cols].transform('mean')


def fe_corr(x, y, groups, nperm):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[ok], y[ok], groups[ok]
    if x.std() == 0 or y.std() == 0: return np.nan, np.nan, int(ok.sum())
    r = np.corrcoef(x, y)[0, 1]
    idx = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    cnt = 0
    for _ in range(nperm):
        yp = y.copy()
        for ii in idx.values():
            if len(ii) > 1: yp[ii] = y[rng.permutation(ii)]
        if abs(np.corrcoef(x, yp)[0, 1]) >= abs(r) - 1e-12: cnt += 1
    return float(r), (cnt + 1) / (nperm + 1), int(ok.sum())


rows = []
for target in ['gap_S1_T2_eV', 'gap_S1_T1_eV', 'HTO_label']:
    ydm = demean(multi, [target])[target].to_numpy()
    for f in feats:
        xdm = demean(multi, [f])[f].to_numpy()
        r, p, n = fe_corr(xdm, ydm, multi.Molecule.to_numpy(), NPERM)
        signs = []
        for mol, g in multi.groupby('Molecule'):
            if len(g) >= 3 and g[f].std() > 0 and g[target].std() > 0:
                signs.append(np.sign(stats.spearmanr(g[f], g[target])[0]))
        rows.append(dict(target=target, descriptor=f, fixed_effect_r=r, perm_p=p, n_conformers=n, n_parents_ge3=len(signs),
                         parents_same_sign=int(sum(1 for s in signs if s == np.sign(r))) if signs and np.isfinite(r) else None))
eff = pd.DataFrame(rows).sort_values(['target', 'perm_p'])
eff['perm_p_BH'] = eff.groupby('target')['perm_p'].transform(lambda p: stats.false_discovery_control(p.fillna(1)))
eff.to_csv(P2 / f'data/model_v2/within_parent_effects_v2{SUF}.csv', index=False)

# switching parents
sw = d.groupby('Molecule').filter(lambda g: 0 < g.HTO_label.sum() < len(g)).copy()
show = ['Molecule', 'conformer', 'HTO_label', 'small_gap_label', 'gap_S1_T1_eV', 'gap_S1_T2_eV', 'dE_kcal', 'w_G', 'interring_twist_max_deg',
        'substituent_twist_max_deg', 'heteroatom_substituent_twist_max_deg', 'rot_torsion_max_dev_deg', 'planarity_rmsd_heavy_A', 'amine_pyramidalization_max_deg', 'dipole_debye', 'homo', 'lumo']
sw[show].sort_values(['Molecule', 'gap_S1_T2_eV']).to_csv(P2 / f'data/model_v2/switching_parents_v2{SUF}.csv', index=False)
paired = []
for mol, g in sw.groupby('Molecule'):
    a, b = g[g.HTO_label == 1], g[g.HTO_label == 0]
    paired.append(dict(Molecule=mol, n_HTO=len(a), n_nonHTO=len(b), **{f'd_{f}': a[f].mean() - b[f].mean() for f in feats if f in g}))
paired = pd.DataFrame(paired)
sign = {}
for f in feats:
    col = f'd_{f}'
    if col in paired:
        v = paired[col].dropna(); v = v[v != 0]
        if len(v):
            k = int((v > 0).sum()); n = len(v)
            sign[f] = dict(n_parents=n, positive=k, sign_test_p=float(stats.binomtest(k, n).pvalue), mean_diff=float(v.mean()))
paired.to_csv(P2 / f'data/model_v2/switching_parents_paired_diff_v2{SUF}.csv', index=False)
top = eff[eff.target == 'gap_S1_T2_eV'].head(12)
summary = dict(subset='intended_species_only' if args.intended_only else 'all_structures', n_structures=len(d), n_multi_conformer_parents=int(multi.Molecule.nunique()), n_conformers_in_them=len(multi), n_switching_parents=int(sw.Molecule.nunique()),
               descriptors_tested=len(feats), permutations=NPERM,
               top_gap_S1_T2=top[['descriptor', 'fixed_effect_r', 'perm_p', 'perm_p_BH', 'n_parents_ge3', 'parents_same_sign']].to_dict('records'),
               significant_BH_005={t: eff[(eff.target == t) & (eff.perm_p_BH < 0.05)].descriptor.tolist() for t in eff.target.unique()},
               switching_sign_tests=sign)
(P2 / f'reports/within_parent_v2{SUF}_summary.json').write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps(summary, indent=1, default=float))
