"""Is there ANY single orbital-character descriptor that defines HTO at the conformer level?

The revised manuscript wants to state that no single orbital descriptor is sufficient. That is a
universal claim, so it has to be measured rather than asserted from the two or three descriptors we
happened to inspect. This script tests every numeric orbital-character quantity in the canonical v2
state-character table on two levels:

  between-molecule : Mann-Whitney AUC over all 159 intended conformers (the level at which the
                     originally proposed HOMO-1 criterion was argued)
  within-parent    : for the 6 parent molecules that contain both HTO and non-HTO conformers, does
                     the descriptor separate them in the same direction? A conformer-level criterion
                     must work here; a descriptor that only works between molecules is a scaffold
                     feature, not a conformational one.

A descriptor would have to score well on BOTH to support a conformer-level orbital criterion.
Writes reports/orbital_criterion_search.csv and a summary used verbatim in the response letter.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

P2 = Path(r'F:/2025/H-CAAN/1125/1204_recomputer/manuscripts/JMCC投稿/0702修改/revised/revision_review_20260910/phase2')
d = pd.read_csv(P2 / 'data/canonical_v2/state_character_v2_intended.csv')

# candidate descriptors: every numeric orbital-character quantity, plus the two composite contrasts
d['contrast_S1T1_vs_T2'] = d.cos_S1_T1 - d[['cos_S1_T2', 'cos_T1_T2']].max(axis=1)
d['T2_minus_T1_Hm1'] = d.T2_w_from_Hm1_total - d.T1_w_from_Hm1_total
SKIP = {'HTO', 'small_gap', 'HOMO_index', 'HOMO_index_triplet_log', 'S1_eV', 'T1_eV', 'T2_eV',
        'S1_sumsq_raw', 'T1_sumsq_raw', 'T2_sumsq_raw'}
cands = [c for c in d.columns
         if d[c].dtype != object and c not in SKIP and d[c].notna().sum() > 100 and d[c].nunique() > 3]

sw_counts = d.groupby('Molecule').HTO.nunique()
switching = sorted(sw_counts[sw_counts > 1].index)
rows = []
for c in cands:
    a, b = d.loc[d.HTO == 1, c].dropna(), d.loc[d.HTO == 0, c].dropna()
    if len(a) < 5 or len(b) < 5:
        continue
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    auc = u / (len(a) * len(b))
    # direction implied by the between-molecule comparison
    sign = 1 if auc >= 0.5 else -1
    agree, tested, deltas = 0, 0, []
    for m in switching:
        g = d[d.Molecule == m]
        h, nh = g.loc[g.HTO == 1, c].dropna(), g.loc[g.HTO == 0, c].dropna()
        if not len(h) or not len(nh):
            continue
        tested += 1
        delta = (h.mean() - nh.mean()) * sign
        deltas.append(delta)
        if delta > 0:
            agree += 1
    rows.append(dict(descriptor=c, between_AUC=round(max(auc, 1 - auc), 3), between_p=p,
                     within_parents_tested=tested, within_parents_agreeing=agree,
                     within_agree_frac=round(agree / tested, 2) if tested else None,
                     median_within_delta=round(float(np.median(deltas)), 4) if deltas else None))

r = pd.DataFrame(rows).sort_values(['within_parents_agreeing', 'between_AUC'], ascending=False)
r.to_csv(P2 / 'reports/orbital_criterion_search.csv', index=False)

n_sw = len(switching)
strong_between = r[(r.between_AUC >= 0.7) & (r.between_p < 0.05)]
conformer_level = r[(r.between_AUC >= 0.7) & (r.between_p < 0.05) & (r.within_parents_agreeing >= n_sw - 1)]
summary = dict(
    n_conformers=int(len(d)), n_HTO=int(d.HTO.sum()),
    n_switching_parents=n_sw, switching_parents=switching,
    n_descriptors_tested=len(r),
    n_with_between_molecule_AUC_ge_070=int(len(strong_between)),
    best_between_molecule=[dict(descriptor=x.descriptor, AUC=x.between_AUC,
                                within=f'{x.within_parents_agreeing}/{x.within_parents_tested}')
                           for x in strong_between.head(6).itertuples()],
    n_that_also_work_within_parents=int(len(conformer_level)),
    conformer_level_descriptors=list(conformer_level.descriptor),
    best_within_parent_agreement=int(r.within_parents_agreeing.max()),
    note=('A conformer-level orbital criterion would need a high between-molecule AUC AND consistent '
          'direction inside the switching parents. Descriptors are scored in the direction implied by '
          'the between-molecule comparison.'))
(P2 / 'reports/orbital_criterion_search_summary.json').write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print()
print('=== top 14 by within-parent agreement, then between-molecule AUC ===')
print(r.head(14).to_string(index=False))
