"""Draft figures from phase2 v2 results (PNG 300 dpi + PDF). Skips panels whose inputs are missing."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P2 = Path(os.environ.get('CALICENE_HTO_ROOT', '.')).resolve() / 'revision_CAJ_20260910/phase2'
F = P2 / 'figures_v2'; F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'Logistic': '#4C72B0', 'MT-DNN': '#C44E52', 'ST-DNN': '#DD8452', 'RF': '#55A868', 'XGBoost': '#8172B2'}


def save(fig, name):
    fig.tight_layout(); fig.savefig(F / f'{name}.png', dpi=300, bbox_inches='tight'); fig.savefig(F / f'{name}.pdf', bbox_inches='tight'); plt.close(fig)
    print('saved', name)


# ---- Fig: model performance (LOMO + GROUPED) with parent-bootstrap CI, per feature set ----
for suf, title in [('_intended', 'intended species (n=159)'), ('', 'all structures (n=184)')]:
    sets = [fs for fs in ['inherited_94', 'inherited_plus_conformer', 'conformer_only'] if (P2 / f'reports/models_v2/{fs}{suf}/summary_LOMO.csv').exists()]
    if not sets: continue
    fig, axes = plt.subplots(2, len(sets), figsize=(4.2 * len(sets), 6.2), squeeze=False)
    for ci_, fs in enumerate(sets):
        for ri, proto in enumerate(['LOMO', 'GROUPED']):
            ax = axes[ri, ci_]
            f = P2 / f'reports/models_v2/{fs}{suf}/summary_{proto}.csv'
            if not f.exists() or not (P2 / f'reports/models_v2/{fs}{suf}/parent_bootstrap_ci_{proto}.csv').exists(): ax.set_visible(False); continue
            s = pd.read_csv(f); s = s[s.threshold_rule == 'fixed_0.5']
            ci = pd.read_csv(P2 / f'reports/models_v2/{fs}{suf}/parent_bootstrap_ci_{proto}.csv')
            s = s.merge(ci, on=['model', 'task'])
            models = ['Logistic', 'ST-DNN', 'MT-DNN', 'RF', 'XGBoost']; x = np.arange(len(models))
            for k, task in enumerate(['small_gap', 'HTO']):
                d = s[s.task == task].set_index('model').loc[models]
                ax.errorbar(x + (k - 0.5) * 0.25, d.roc_auc_seedavg, yerr=[d.roc_auc_seedavg - d.roc_auc_ci_low, d.roc_auc_ci_high - d.roc_auc_seedavg],
                            fmt='o' if task == 'HTO' else 's', color='#C44E52' if task == 'HTO' else '#4C72B0', capsize=3, label=task.replace('small_gap', 'small-gap'))
            ax.axhline(0.5, ls=':', c='grey'); ax.set_ylim(0.3, 1.02); ax.set_xticks(x); ax.set_xticklabels(models, rotation=30)
            ax.set_ylabel('ROC-AUC (pooled OOF, seed-avg; 95% parent bootstrap)'); ax.set_title(f'{proto} | {fs}', fontsize=9)
            if ri == 0 and ci_ == 0: ax.legend(frameon=False)
    fig.suptitle(f'Leakage-free evaluation, {title}', fontsize=10)
    save(fig, f'Fig_model_performance{suf or "_all"}')

# ---- Fig: within-parent associations (intended) ----
d = pd.read_csv(P2 / 'data/model_v2/dataset_v2.csv'); d = d[d.intended_species == 1]
multi = d.groupby('Molecule').filter(lambda g: len(g) >= 2)
eff = pd.read_csv(P2 / 'data/model_v2/within_parent_effects_v2_intended.csv')
top = eff[eff.target == 'gap_S1_T2_eV'].sort_values('perm_p').head(4).descriptor.tolist()
fig, axes = plt.subplots(1, len(top), figsize=(3.6 * len(top), 3.4))
for ax, f in zip(np.atleast_1d(axes), top):
    x = multi[f] - multi.groupby('Molecule')[f].transform('mean'); y = multi.gap_S1_T2_eV - multi.groupby('Molecule').gap_S1_T2_eV.transform('mean')
    ax.scatter(x, y, c=np.where(multi.HTO_label == 1, '#C44E52', '#4C72B0'), s=18, alpha=.8)
    r = eff[(eff.target == 'gap_S1_T2_eV') & (eff.descriptor == f)].iloc[0]
    ax.set_title(f'{f}\nfixed-effect r={r.fixed_effect_r:.2f}, perm p={r.perm_p:.3g}', fontsize=8)
    ax.set_xlabel('parent-demeaned descriptor'); ax.set_ylabel('parent-demeaned S1-T2 (eV)'); ax.axhline(0, c='grey', lw=.5); ax.axvline(0, c='grey', lw=.5)
save(fig, 'Fig_within_parent_effects_intended')

# ---- Fig: switching parents ----
sw = pd.read_csv(P2 / 'data/model_v2/switching_parents_v2_intended.csv')
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, (mol, g) in enumerate(sw.groupby('Molecule')):
    ax.scatter(g.gap_S1_T2_eV, [i] * len(g), c=np.where(g.HTO_label == 1, '#C44E52', '#4C72B0'), s=30 + 40 * g.w_G.fillna(0), edgecolor='k', lw=.4)
    for _, r in g.iterrows(): ax.annotate(f"{r.conformer}\n{r.dE_kcal:.1f} kcal", (r.gap_S1_T2_eV, i), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=6)
ax.axvline(0, c='grey', ls='--'); ax.set_yticks(range(sw.Molecule.nunique())); ax.set_yticklabels(sorted(sw.Molecule.unique()), fontsize=8)
ax.set_xlabel('E(S1) - E(T2) (eV); marker size ~ Boltzmann weight (G, 298 K)'); ax.set_title('Same-parent HTO switching (intended species)', fontsize=9)
save(fig, 'Fig_switching_parents_intended')

# ---- Fig: threshold sensitivity + boundary ----
ts = pd.read_csv(P2 / 'data/canonical_v2/threshold_sensitivity_v2_intended.csv')
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
a = ts[ts.criterion == 'small_gap']; axes[0].plot(a.threshold_eV, a.positive_conformers, 'o-', c='#4C72B0'); axes[0].axvline(0.4, ls='--', c='grey')
axes[0].set_xlabel('small-gap threshold on E(S1)-E(T1) (eV)'); axes[0].set_ylabel('positive conformers'); axes[0].set_title('(A) TADF-like threshold', fontsize=9)
b = ts[ts.criterion == 'HTO']; axes[1].plot(b.threshold_eV, b.positive_conformers, 'o-', c='#C44E52'); axes[1].axvline(0, ls='--', c='grey')
axes[1].set_xlabel('HTO criterion shift on E(S1)-E(T2) (eV)'); axes[1].set_ylabel('positive conformers'); axes[1].set_title('(B) HTO boundary sensitivity', fontsize=9)
save(fig, 'Fig_threshold_sensitivity_intended')

# ---- Fig: state character ----
sc = pd.read_csv(P2 / 'data/canonical_v2/state_character_v2_intended.csv')
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
for ax, col, lab in zip(axes, ['S1_w_H_L', 'T2_w_from_Hm1_total', 'cos_S1_T2'], ['S1: weight of HOMO->LUMO', 'T2: total weight from HOMO-1', 'cos similarity S1 ~ T2']):
    data = [sc[sc.HTO == 0][col].dropna(), sc[sc.HTO == 1][col].dropna()]
    ax.boxplot(data, labels=[f'non-HTO (n={len(data[0])})', f'HTO (n={len(data[1])})'], widths=.5)
    for i, v in enumerate(data): ax.scatter(np.random.normal(i + 1, .05, len(v)), v, s=8, alpha=.5, c='#C44E52' if i else '#4C72B0')
    ax.set_title(lab, fontsize=9)
save(fig, 'Fig_state_character_intended')

# ---- Fig: t-SNE with quantitative overlap ----
ov = pd.read_csv(P2 / 'data/model_v2/class_overlap_v2.csv'); co = pd.read_csv(P2 / 'data/model_v2/tsne_coordinates_v2.csv')
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, fs in zip(axes, ['inherited_94', 'inherited_plus_conformer']):
    cls = co.small_gap_label + 2 * co.HTO_label
    for k, (lab, c) in enumerate([('negative', '#BBBBBB'), ('small-gap only', '#4C72B0'), ('HTO only', '#C44E52'), ('dual', '#8172B2')]):
        m = cls == k; ax.scatter(co.loc[m, f'tsne1_{fs}'], co.loc[m, f'tsne2_{fs}'], s=14, c=c, label=lab, alpha=.85)
    o = ov[(ov.feature_set == fs) & (ov.labelling == 'HTO')].iloc[0]
    ax.set_title(f'{fs}\nHTO silhouette {o.silhouette_original:.2f} (orig) / {o.silhouette_tsne:.2f} (t-SNE); kNN5 purity {o.knn5_purity:.2f}, parent-null p={o.knn5_p_parent_preserving:.2f}', fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(frameon=False, fontsize=7)
save(fig, 'Fig_tsne_quantified')

# ---- Fig: Boltzmann parent-level ----
bp = pd.read_csv(P2 / 'data/canonical_v2/boltzmann_parent_v2_intended.csv'); bp = bp[bp.n_HTO > 0].sort_values('P_HTO_G')
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.barh(bp.Molecule, bp.P_HTO_G, color='#C44E52', alpha=.8, label='P(HTO), G(298 K)'); ax.barh(bp.Molecule, bp.P_HTO_E, color='none', edgecolor='k', label='P(HTO), electronic E')
ax.set_xlabel('Boltzmann-weighted HTO probability within parent'); ax.legend(frameon=False, fontsize=7); ax.tick_params(axis='y', labelsize=7)
save(fig, 'Fig_boltzmann_parent_intended')
print('done')
