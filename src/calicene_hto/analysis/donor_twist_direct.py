"""Direct donor-twist measurement from the optimised geometries (audit of 2026-09-19).

Why: substituent_twist_mean_deg is 0.0 BY DEFAULT whenever no substituent plane is measurable
(conformer_geometry_descriptors.py l.143) and amine_pyramidalization_max_deg only sees 3-coordinate exocyclic N
(l.150). The N of N-PMe3 has two heavy neighbours, so both routines skip it: every NPMe3 conformer reads 0.0 as a
default, not a measurement, and 88 of the 159 intended-species rows are such defaults (16 of the 22 HTO positives).
Any statement built on those zeros ("NPMe3 planar", "HTO 13.3 deg vs 29.2 deg") is therefore unsupported.

Geometries: data/canonical_v2/geoms_v2.json, extracted on 101 (read-only) from the first orientation block of each
singlet TD log listed in excited_states_v2.csv (the TD run is a single point on the optimised S0 geometry), with a
standard-library snippet; 184 structures, 14-42 atoms.

Measurement, never defaulted: for every exocyclic X in {N, O, S, P} bonded to exactly one ring atom Cr, with non-ring
neighbours Y (heavy or H):
    >= 2 Y (NMe2, NH2, NO2)   angle between the plane (X, Y...) and the ring plane; for N also 360 - sum of angles
    == 1 Y (N=PMe3, OMe, OH)  folded dihedral Cr'-Cr-X-Y (0 = Y in the ring plane) and the Cr-X-Y angle
    == 0 Y                    terminal (=O), not a group; skipped
Nitro-type N (all Y are O) is flagged and excluded from the donor statistics.
Outputs data/canonical_v2/donor_twist_direct.csv (one row per substituent) and
reports/donor_twist_direct_summary.json (exemplars, coverage, between-molecule and within-parent statistics).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolTransforms
from scipy import stats

P2 = Path(r'F:/2025/H-CAAN/1125/1204_recomputer/manuscripts/JMCC投稿/0702修改/revised/revision_review_20260910/phase2')
PT = Chem.GetPeriodicTable()
geoms = json.loads((P2 / 'data/canonical_v2/geoms_v2.json').read_text())


def fold(d):
    d = abs(d) % 180.0
    return min(d, 180.0 - d)


def plane_normal(pts):
    return np.linalg.svd(pts - pts.mean(0))[2][-1]


def plane_angle(n1, n2):
    return float(np.degrees(np.arccos(abs(np.clip(n1 @ n2, -1, 1)))))


recs = []
for key, g in geoms.items():
    geom = np.array(g['geom'], dtype=float)
    xyz = f'{len(geom)}\n\n' + '\n'.join(f'{PT.GetElementSymbol(int(z))} {x:.8f} {y:.8f} {w:.8f}' for z, x, y, w in geom)
    m = Chem.MolFromXYZBlock(xyz)
    try:
        rdDetermineBonds.DetermineBonds(m, charge=0)
    except Exception:
        rdDetermineBonds.DetermineConnectivity(m)
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)                      # ring info is not populated by bond perception alone
    conf = m.GetConformer()
    rings = [list(x) for x in m.GetRingInfo().AtomRings()]
    in_ring = {i for ring in rings for i in ring}
    assert rings, f'no ring perceived for {key}'
    normals = [plane_normal(geom[ring, 1:]) for ring in rings]
    for a in m.GetAtoms():
        if a.GetIdx() in in_ring or a.GetSymbol() not in ('N', 'O', 'S', 'P'):
            continue
        ring_nb = [n for n in a.GetNeighbors() if n.GetIdx() in in_ring]
        if len(ring_nb) != 1:
            continue
        cr = ring_nb[0]
        ys = [n for n in a.GetNeighbors() if n.GetIdx() != cr.GetIdx()]
        rn = normals[[i for i, ring in enumerate(rings) if cr.GetIdx() in ring][0]]
        rec = dict(Molecule=g['Molecule'], conformer=g['conformer'], X=a.GetSymbol(), X_idx=a.GetIdx(),
                   Y_elements=''.join(sorted(n.GetSymbol() for n in ys)), n_Y=len(ys),
                   nitro_like=int(a.GetSymbol() == 'N' and len(ys) >= 2 and all(n.GetSymbol() == 'O' for n in ys)))
        if len(ys) >= 2:
            pts = geom[[a.GetIdx()] + [n.GetIdx() for n in ys], 1:]
            rec.update(kind='plane', twist_deg=plane_angle(plane_normal(pts), rn))
            if a.GetSymbol() == 'N' and len(ys) == 2:
                nb = [cr.GetIdx()] + [n.GetIdx() for n in ys]
                angs = [rdMolTransforms.GetAngleDeg(conf, nb[i], a.GetIdx(), nb[j]) for i in range(3) for j in range(i + 1, 3)]
                rec['pyramidalisation_deg'] = 360.0 - sum(angs)
        elif len(ys) == 1:
            y = ys[0]
            crp = [n for n in cr.GetNeighbors() if n.GetIdx() != a.GetIdx() and n.GetIdx() in in_ring][0]
            d = rdMolTransforms.GetDihedralDeg(conf, crp.GetIdx(), cr.GetIdx(), a.GetIdx(), y.GetIdx())
            rec.update(kind='dihedral', twist_deg=fold(d), raw_dihedral_deg=d,
                       Cr_X_Y_angle_deg=rdMolTransforms.GetAngleDeg(conf, cr.GetIdx(), a.GetIdx(), y.GetIdx()))
        else:
            rec.update(kind='terminal', twist_deg=np.nan)
        recs.append(rec)
sub = pd.DataFrame(recs)
sub.to_csv(P2 / 'data/canonical_v2/donor_twist_direct.csv', index=False)

# ---- per-conformer donor twist: mean over donor groups (N/O/S/P with >= 1 Y, not nitro-like)
don = sub[(sub.kind != 'terminal') & (sub.nitro_like == 0)]
per = don.groupby(['Molecule', 'conformer']).agg(donor_twist_deg=('twist_deg', 'mean'), n_donor=('twist_deg', 'size'),
                                                 donor_groups=('Y_elements', lambda s: ';'.join(f'{x}({y})' for x, y in zip(don.loc[s.index, 'X'], s)))).reset_index()
d = pd.read_csv(P2 / 'data/model_v2/dataset_v2.csv')
d = d[d.intended_species == 1].merge(per, on=['Molecule', 'conformer'], how='left')
meas = d[d.donor_twist_deg.notna()].copy()
old_meas = d[d.substituent_twist_mean_deg > 0]
cov = dict(intended_rows=len(d), rows_with_measured_donor=len(meas), HTO_rows=int(d.HTO_label.sum()),
           HTO_rows_measured=int(meas.HTO_label.sum()), parents=int(d.Molecule.nunique()), parents_measured=int(meas.Molecule.nunique()),
           old_descriptor_nonzero_rows=len(old_meas),
           corr_new_vs_old_on_old_measured=float(np.corrcoef(old_meas.substituent_twist_mean_deg, d.loc[old_meas.index, 'donor_twist_deg'].fillna(-1))[0, 1]))

# ---- exemplars
ex = {}
for mol in ['5ring_npme3_3ring_CN', '5ring_npme3', '5ring_nme2_3ring_cn_in_con2', '5ring_cn_in']:
    g = d[d.Molecule == mol]
    s = don[don.Molecule == mol]
    ex[mol] = dict(n_conformers=int(len(g)), HTO=int(g.HTO_label.sum()), n_donor_groups=int(s.groupby('conformer').size().max()) if len(s) else 0,
                   groups=sorted(set(f'{x}({y})' for x, y in zip(s.X, s.Y_elements))),
                   twist_min=float(s.twist_deg.min()) if len(s) else None, twist_max=float(s.twist_deg.max()) if len(s) else None,
                   twist_mean=float(g.donor_twist_deg.mean()) if len(s) else None,
                   pyramidalisation_mean=float(s.pyramidalisation_deg.mean()) if 'pyramidalisation_deg' in s and s.pyramidalisation_deg.notna().any() else None,
                   Cr_X_Y_angle_mean=float(s.Cr_X_Y_angle_deg.mean()) if 'Cr_X_Y_angle_deg' in s and s.Cr_X_Y_angle_deg.notna().any() else None)

# ---- between molecules, measured rows only
a, b = meas.loc[meas.HTO_label == 1, 'donor_twist_deg'], meas.loc[meas.HTO_label == 0, 'donor_twist_deg']
u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
between = dict(n_HTO=len(a), n_nonHTO=len(b), HTO_mean=float(a.mean()), nonHTO_mean=float(b.mean()), HTO_median=float(a.median()),
               nonHTO_median=float(b.median()), AUC_HTO_more_twisted=float(u / (len(a) * len(b))), mannwhitney_p=float(p))

# ---- within parents, measured rows only
multi = meas.groupby('Molecule').filter(lambda g: len(g) >= 2)
rng = np.random.default_rng(20260910)


def fe(x, y, groups, nperm=5000):
    r = np.corrcoef(x, y)[0, 1]
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    cnt = 0
    for _ in range(nperm):
        yp = y.copy()
        for ii in idx:
            if len(ii) > 1:
                yp[ii] = y[rng.permutation(ii)]
        if abs(np.corrcoef(x, yp)[0, 1]) >= abs(r) - 1e-12:
            cnt += 1
    return float(r), (cnt + 1) / (nperm + 1)


within = {}
for target in ['gap_S1_T2_eV', 'gap_S1_T1_eV']:
    x = (multi.donor_twist_deg - multi.groupby('Molecule').donor_twist_deg.transform('mean')).to_numpy()
    y = (multi[target] - multi.groupby('Molecule')[target].transform('mean')).to_numpy()
    r, pp = fe(x, y, multi.Molecule.to_numpy())
    signs = [np.sign(stats.spearmanr(g.donor_twist_deg, g[target])[0]) for _, g in multi.groupby('Molecule')
             if len(g) >= 3 and g.donor_twist_deg.std() > 0 and g[target].std() > 0]
    within[target] = dict(fixed_effect_r=r, perm_p=pp, n_conformers=len(multi), n_parents=int(multi.Molecule.nunique()),
                          parents_ge3=len(signs), parents_same_sign=int(sum(1 for s in signs if s == np.sign(r))))

# ---- switching parents
sw = d.groupby('Molecule').filter(lambda g: 0 < g.HTO_label.sum() < len(g))
rows = []
for mol, g in sw.groupby('Molecule'):
    h, nh = g.loc[g.HTO_label == 1, 'donor_twist_deg'], g.loc[g.HTO_label == 0, 'donor_twist_deg']
    measured = bool(h.notna().all() and nh.notna().all())
    rows.append(dict(parent=mol, n=len(g), donor_groups=g.donor_groups.dropna().iloc[0] if g.donor_groups.notna().any() else 'none measurable',
                     HTO_twist=float(h.mean()) if measured else None, nonHTO_twist=float(nh.mean()) if measured else None,
                     measured=measured, HTO_less_twisted=bool(h.mean() < nh.mean()) if measured else None))
switch = dict(n_switching=len(rows), n_measured=sum(r['measured'] for r in rows),
              HTO_less_twisted_in=sum(1 for r in rows if r['HTO_less_twisted']), per_parent=rows)

out = dict(coverage=cov, exemplars=ex, between_molecules_measured_only=between, within_parents_measured_only=within, switching_parents=switch)
(P2 / 'reports/donor_twist_direct_summary.json').write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
