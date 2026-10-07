"""Three label-level analyses on canonical v2 (read-only on logs).

1. boltzmann_v2.csv / boltzmann_parent_v2.csv  (R1-24)
   Within-parent Boltzmann weights at 298.15 K from (a) electronic energy at the TD geometry
   and (b) G(298) from the frequency job; parent-level P(HTO), P(small-gap), energy spread.
2. threshold_sensitivity_v2.csv  (R1-3, R1-10)
   Small-gap counts vs threshold; HTO boundary population and label flips under +/- shifts.
3. state_character_v2.csv  (R1-8, R1-20, R1-21)
   Orbital composition of S1, T1, T2 from the accepted TD logs: weights of HOMO->LUMO,
   HOMO-1->LUMO, HOMO->LUMO+1, dominant pair, and cosine similarity of the |coefficient|
   vectors S1~T1, S1~T2 (descriptive, not an exchange integral).
"""
import csv, re, json, math, argparse, os
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd

ROOT = Path(os.environ.get('CALICENE_HTO_ROOT', '.')).resolve()
V2 = ROOT / 'revision_CAJ_20260910/phase2/data/canonical_v2'
OUTR = ROOT / 'revision_CAJ_20260910/phase2/reports'
KT = 0.0019872041 * 298.15          # kcal/mol
H2KCAL = 627.509474
FLOAT = r'[-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?'

ap = argparse.ArgumentParser(); ap.add_argument('--intended_only', action='store_true'); args = ap.parse_args()
d = pd.read_csv(V2 / 'excited_states_v2.csv')
SUF = ''
if args.intended_only:
    sp = pd.read_csv(V2 / 'species_v2.csv')[['Molecule', 'conformer', 'intended_species']]
    d = d.merge(sp, on=['Molecule', 'conformer']); d = d[d.intended_species == 1].reset_index(drop=True); SUF = '_intended'

# ---------------- 1. Boltzmann ----------------
rows, prows = [], []
for parent, g in d.groupby('Molecule'):
    e = g['td_scf_hartree'].to_numpy() * H2KCAL
    de = e - e.min()
    w_e = np.exp(-de / KT); w_e /= w_e.sum()
    gg = g['G_298_hartree'].to_numpy(float)
    if np.isfinite(gg).all():
        dg = (gg - gg.min()) * H2KCAL; w_g = np.exp(-dg / KT); w_g /= w_g.sum()
    else:
        dg = np.full(len(g), np.nan); w_g = np.full(len(g), np.nan)
    for i, (_, r) in enumerate(g.iterrows()):
        rows.append(dict(Molecule=parent, conformer=r['conformer'], n_conformers=len(g), dE_kcal=round(de[i], 3), w_E=round(w_e[i], 4),
                         dG_kcal=round(dg[i], 3) if np.isfinite(dg[i]) else None, w_G=round(w_g[i], 4) if np.isfinite(w_g[i]) else None,
                         HTO=int(r['HTO_label']), small_gap=int(r['small_gap_label']), gap_S1_T2_eV=r['gap_S1_T2_eV'], gap_S1_T1_eV=r['gap_S1_T1_eV']))
    hto = g['HTO_label'].to_numpy(); sg = g['small_gap_label'].to_numpy()
    prows.append(dict(Molecule=parent, n_conformers=len(g), n_HTO=int(hto.sum()), n_small_gap=int(sg.sum()),
                      dE_span_kcal=round(de.max(), 3), dG_span_kcal=round(np.nanmax(dg), 3) if np.isfinite(dg).any() else None,
                      P_HTO_E=round(float((w_e * hto).sum()), 4), P_HTO_G=round(float((w_g * hto).sum()), 4) if np.isfinite(w_g).all() else None,
                      P_small_gap_E=round(float((w_e * sg).sum()), 4), P_small_gap_G=round(float((w_g * sg).sum()), 4) if np.isfinite(w_g).all() else None,
                      lowest_E_conformer_is_HTO=int(hto[np.argmin(de)]), lowest_G_conformer_is_HTO=int(hto[np.nanargmin(dg)]) if np.isfinite(dg).all() else None,
                      HTO_switching_parent=int(0 < hto.sum() < len(g)),
                      conformers_within_3kcal_E=int((de <= 3.0).sum())))
pd.DataFrame(rows).to_csv(V2 / f'boltzmann_v2{SUF}.csv', index=False)
pp = pd.DataFrame(prows); pp.to_csv(V2 / f'boltzmann_parent_v2{SUF}.csv', index=False)

# ---------------- 2. Threshold sensitivity ----------------
ts = []
for thr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    m = d['gap_S1_T1_eV'] <= thr
    ts.append(dict(criterion='small_gap', threshold_eV=thr, positive_conformers=int(m.sum()), positive_parents=int(d.loc[m, 'Molecule'].nunique()),
                   dual_positive=int((m & (d['HTO_label'] == 1)).sum())))
base = d['HTO_label'] == 1
for shift in [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]:
    m = (d['gap_S1_T1_eV'] > 0) & (d['gap_S1_T2_eV'] <= shift)
    ts.append(dict(criterion='HTO', threshold_eV=shift, positive_conformers=int(m.sum()), positive_parents=int(d.loc[m, 'Molecule'].nunique()),
                   flips_vs_base=int((m != base).sum())))
for band in [0.02, 0.05, 0.10]:
    ts.append(dict(criterion='HTO_boundary_band', threshold_eV=band, positive_conformers=int((d['gap_S1_T2_eV'].abs() <= band).sum()),
                   positive_parents=int(d.loc[d['gap_S1_T2_eV'].abs() <= band, 'Molecule'].nunique())))
    ts.append(dict(criterion='small_gap_boundary_band_at_0.4', threshold_eV=band, positive_conformers=int(((d['gap_S1_T1_eV'] - 0.4).abs() <= band).sum()),
                   positive_parents=int(d.loc[(d['gap_S1_T1_eV'] - 0.4).abs() <= band, 'Molecule'].nunique())))
pd.DataFrame(ts).to_csv(V2 / f'threshold_sensitivity_v2{SUF}.csv', index=False)

# ---------------- 3. State character ----------------
STATE = re.compile(r'Excited State\s+(\d+):\s+(Singlet|Triplet)-\S+\s+(' + FLOAT + r')\s+eV[^\n]*\n((?:\s*\d+\s*(?:->|<-)\s*\d+\s+' + FLOAT + r'\s*\n)*)')
TR = re.compile(r'(\d+)\s*(->|<-)\s*(\d+)\s+(' + FLOAT + ')')


def parse_states(path, spin):
    text = path.read_text(errors='replace')
    nocc = len(re.findall(r'Alpha\s+occ\. eigenvalues --([^\n]+)', text))
    nocc = sum(len(re.findall(FLOAT, l)) for l in re.findall(r'Alpha\s+occ\. eigenvalues --([^\n]+)', text))
    nocc_first = None
    # occ eigenvalue blocks may be printed more than once; count per block set using the first "Alpha virt" occurrence
    m = re.search(r'((?:\s*Alpha\s+occ\. eigenvalues --[^\n]*\n)+)', text)
    if m:
        nocc_first = sum(len(re.findall(FLOAT, l)) for l in re.findall(r'Alpha\s+occ\. eigenvalues --([^\n]+)', m.group(1)))
    homo = nocc_first
    states = {}
    for n, sp, e, block in STATE.findall(text):
        if sp != spin: continue
        vec = {}
        for i, arrow, a, c in TR.findall(block):
            if arrow == '->':
                vec[(int(i), int(a))] = float(c)
        states[int(n)] = (float(e), vec)
    return homo, states


def orb_label(idx, ref, name):
    off = idx - ref
    return name if off == 0 else f'{name}{off:+d}'


def weights(vec):
    tot = sum(c * c for c in vec.values())
    return {k: (c * c) / tot for k, c in vec.items()} if tot > 0 else {}, tot


def cos(u, v):
    keys = set(u) | set(v)
    a = np.array([abs(u.get(k, 0.0)) for k in keys]); b = np.array([abs(v.get(k, 0.0)) for k in keys])
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) if a.any() and b.any() else None


sc = []
for _, r in d.iterrows():
    homo, S = parse_states(ROOT / r['singlet_log'], 'Singlet')
    homo_t, T = parse_states(ROOT / r['triplet_log'], 'Triplet')
    rec = dict(Molecule=r['Molecule'], conformer=r['conformer'], HOMO_index=homo, HOMO_index_triplet_log=homo_t,
               HTO=int(r['HTO_label']), small_gap=int(r['small_gap_label']))
    if not homo or 1 not in S or 1 not in T or 2 not in T:
        rec['note'] = 'parse_incomplete'; sc.append(rec); continue
    L = homo + 1
    for tag, (e, vec) in [('S1', S[1]), ('T1', T[1]), ('T2', T[2])]:
        w, tot = weights(vec)
        rec[f'{tag}_eV'] = e; rec[f'{tag}_sumsq_raw'] = round(tot, 3)
        rec[f'{tag}_w_H_L'] = round(w.get((homo, L), 0.0), 3)
        rec[f'{tag}_w_Hm1_L'] = round(w.get((homo - 1, L), 0.0), 3)
        rec[f'{tag}_w_H_Lp1'] = round(w.get((homo, L + 1), 0.0), 3)
        rec[f'{tag}_w_Hm1_Lp1'] = round(w.get((homo - 1, L + 1), 0.0), 3)
        rec[f'{tag}_w_Hm2_any'] = round(sum(v for (i, a), v in w.items() if i == homo - 2), 3)
        rec[f'{tag}_w_from_Hm1_total'] = round(sum(v for (i, a), v in w.items() if i == homo - 1), 3)
        rec[f'{tag}_w_from_H_total'] = round(sum(v for (i, a), v in w.items() if i == homo), 3)
        if w:
            (i, a), v = max(w.items(), key=lambda kv: kv[1])
            rec[f'{tag}_dominant'] = orb_label(i, homo, 'H') + '->' + orb_label(a, L, 'L')
            rec[f'{tag}_dominant_weight'] = round(v, 3)
            rec[f'{tag}_n_pairs_ge_0.1'] = int(sum(1 for v_ in w.values() if v_ >= 0.1))
    rec['cos_S1_T1'] = cos(S[1][1], T[1][1]); rec['cos_S1_T2'] = cos(S[1][1], T[2][1]); rec['cos_T1_T2'] = cos(T[1][1], T[2][1])
    rec['S1_closer_to'] = 'T2' if (rec['cos_S1_T2'] or 0) > (rec['cos_S1_T1'] or 0) else 'T1'
    sc.append(rec)
sc = pd.DataFrame(sc); sc.to_csv(V2 / f'state_character_v2{SUF}.csv', index=False)

summary = dict(
    subset='intended_species_only' if args.intended_only else 'all', n=len(d),
    boltzmann=dict(parents=len(pp), switching_parents=int(pp['HTO_switching_parent'].sum()),
                   parents_with_any_HTO=int((pp['n_HTO'] > 0).sum()),
                   P_HTO_G_ge_0_5=int((pp['P_HTO_G'] >= 0.5).sum()), P_HTO_G_gt_0=int((pp['P_HTO_G'] > 0).sum()),
                   lowest_G_is_HTO=int(pp['lowest_G_conformer_is_HTO'].fillna(0).sum()),
                   max_dE_span_kcal=float(pp['dE_span_kcal'].max()), parents_span_gt_3kcal=int((pp['dE_span_kcal'] > 3).sum()),
                   conformers_total=int(pp['n_conformers'].sum()), conformers_within_3kcal=int(pp['conformers_within_3kcal_E'].sum()),
                   G_missing_parents=int(pp['P_HTO_G'].isna().sum())),
    threshold=ts,
    state_character=dict(n=len(sc), parse_incomplete=int(sc['note'].notna().sum()) if 'note' in sc else 0,
                         S1_dominant=sc['S1_dominant'].value_counts().to_dict() if 'S1_dominant' in sc else {},
                         T1_dominant=sc['T1_dominant'].value_counts().to_dict() if 'T1_dominant' in sc else {},
                         T2_dominant=sc['T2_dominant'].value_counts().to_dict() if 'T2_dominant' in sc else {},
                         HTO_S1_closer_to_T2=int(((sc['HTO'] == 1) & (sc['S1_closer_to'] == 'T2')).sum()) if 'S1_closer_to' in sc else None,
                         HTO_n=int((sc['HTO'] == 1).sum()),
                         nonHTO_S1_closer_to_T2=int(((sc['HTO'] == 0) & (sc['S1_closer_to'] == 'T2')).sum()) if 'S1_closer_to' in sc else None,
                         mean_T2_w_from_Hm1_HTO=float(sc.loc[sc['HTO'] == 1, 'T2_w_from_Hm1_total'].mean()) if 'T2_w_from_Hm1_total' in sc else None,
                         mean_T2_w_from_Hm1_nonHTO=float(sc.loc[sc['HTO'] == 0, 'T2_w_from_Hm1_total'].mean()) if 'T2_w_from_Hm1_total' in sc else None,
                         mean_S1_w_H_L_HTO=float(sc.loc[sc['HTO'] == 1, 'S1_w_H_L'].mean()), mean_S1_w_H_L_nonHTO=float(sc.loc[sc['HTO'] == 0, 'S1_w_H_L'].mean())))
(OUTR / f'excited_state_analyses_v2{SUF}_summary.json').write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
