"""Assemble the v2 modelling dataset: canonical-v2 labels + inherited ground-state
descriptors (94, reconstructed from raw values with the phase1 lineage-checked builder)
+ a new per-conformer geometry/electrostatic block computed from the accepted TD geometry.

No excited-state quantity enters any feature. Raw values only; imputation/scaling are
done inside CV folds by the model script. Outputs phase2/data/model_v2/dataset_v2.csv,
feature_blocks.json, manifest.json.
"""
import csv, json, hashlib, sys, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ.get('CALICENE_HTO_ROOT', '.')).resolve()
P = ROOT / 'reverse_TADF_system_deepreseach_0617'
PH1 = ROOT / 'revision_CAJ_20260910/phase1'
V2 = ROOT / 'revision_CAJ_20260910/phase2/data/canonical_v2'
OUT = ROOT / 'revision_CAJ_20260910/phase2/data/model_v2'
sys.path.insert(0, str(PH1 / 'scripts'))
from build_diagnostic_dataset import construct, FORBIDDEN   # lineage-checked, raises on excited-state sources

H2KCAL = 627.509474
raw = pd.read_csv(P / 'data/extracted/all_conformers_data.csv')
lab = pd.read_csv(V2 / 'excited_states_v2.csv')
geo = pd.read_csv(V2 / 'conformer_geometry_descriptors.csv')
bol = pd.read_csv(V2 / 'boltzmann_v2.csv')
keys = ['Molecule', 'conformer']
hist = json.loads((P / 'data/paper_figures/1204_me_write/1/pipeline_results.json').read_text())['all_results'][0]['step4_dl']['feature_names']
inherited = [n for n in hist[:97] if n not in FORBIDDEN]
assert len(inherited) == 94

m = lab[keys + ['small_gap_label', 'HTO_label', 'gap_S1_T1_eV', 'gap_S1_T2_eV', 'natoms', 'td_scf_hartree', 'G_298_hartree']].merge(raw, on=keys, validate='one_to_one')
X_inh, lineage = construct(m, inherited)
# substituent presence/count flags are stored sparsely in the raw extractor (absent = NaN); absent means 0, as in the
# original pipeline's fillna(0). Only these indicator columns are zero-filled; everything else keeps NaN for fold-wise imputation.
flag_cols = [c for c in inherited if c.startswith('has_') or c.startswith('count_')]
X_inh[flag_cols] = X_inh[flag_cols].fillna(0.0)

# ---- per-conformer block (ground state only) ----
conf = m[keys].copy()
conf = conf.merge(geo.drop(columns=['natoms', 'atom_order_consistent', 'n_rings_perceived', 'n_rotatable_heavy_bonds', 'n_substituent_planes'], errors='ignore'), on=keys, how='left')
conf = conf.merge(bol[keys + ['dE_kcal', 'w_E', 'dG_kcal', 'w_G']], on=keys, how='left')
conf['xtb_rel_energy_kcal'] = pd.to_numeric(m['crest_energy'], errors='coerce') / H2KCAL * 1.0   # raw column is kcal/mol*627.5 (unit bug in extractor); rescaled to kcal/mol
conf['ground_homo_lumo_gap_eV'] = pd.to_numeric(m['homo_lumo_gap'], errors='coerce')
conf['gaussian_radius_of_gyration_raw'] = pd.to_numeric(m['gaussian_radius_of_gyration'], errors='coerce')
conf_cols = [c for c in conf.columns if c not in keys]
# sanity: none of the conformer block columns may be excited-state derived
banned = {'s1', 't1', 't2', 'gap_S1', 'osc', 'wavelength', 'inver'}
for c in conf_cols:
    assert not any(b in c for b in ['s1_', 't1_', 't2_', 'oscill', 'wavelength', 'inver', 'S1_', 'T1_', 'T2_']), c

data = pd.concat([m[keys + ['small_gap_label', 'HTO_label', 'gap_S1_T1_eV', 'gap_S1_T2_eV']].reset_index(drop=True),
                  X_inh.reset_index(drop=True), conf[conf_cols].reset_index(drop=True)], axis=1)
OUT.mkdir(parents=True, exist_ok=True)
data.to_csv(OUT / 'dataset_v2.csv', index=False)
blocks = dict(inherited_94=inherited, conformer_block=conf_cols,
              conformer_varying_within_parent=[c for c in inherited + conf_cols if (data.groupby('Molecule')[c].std().fillna(0) > 1e-9).sum() > 0])
(OUT / 'feature_blocks.json').write_text(json.dumps(blocks, indent=2))
g = data.groupby('Molecule')
manifest = dict(n=len(data), parents=data.Molecule.nunique(), HTO=int(data.HTO_label.sum()), small_gap=int(data.small_gap_label.sum()),
                both=int((data.HTO_label & data.small_gap_label).sum()), n_inherited=len(inherited), n_conformer_block=len(conf_cols),
                inherited_varying_within_parent=int(sum((g[c].std().fillna(0) > 1e-9).sum() > 0 for c in inherited)),
                conformer_block_varying_within_parent=int(sum((g[c].std().fillna(0) > 1e-9).sum() > 0 for c in conf_cols)),
                missing_values={c: int(data[c].isna().sum()) for c in inherited + conf_cols if data[c].isna().sum()},
                inputs_sha256={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in [V2 / 'excited_states_v2.csv', V2 / 'conformer_geometry_descriptors.csv', V2 / 'boltzmann_v2.csv', P / 'data/extracted/all_conformers_data.csv', Path(__file__)]},
                zero_filled_indicator_columns=flag_cols, note='labels: small_gap = S1-T1<=0.4 eV; HTO = T1<S1<=T2 (wB97XD/def2TZVP TD, ground final geometry). Features: ground state / CREST / geometry only.')
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
print('conformer block:', conf_cols)
