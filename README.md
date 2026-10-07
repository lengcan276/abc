# Calicene HTO ML

Reproducible core code for the revised conformer-resolved study of higher-triplet ordering (HTO) in calicene-dominated donor–acceptor systems.

The repository contains the final Round-2 analysis used by the revised manuscript:

- continuous regression of `E(S1)-E(T1)` and `E(S1)-E(T2)`;
- fixed published thresholds (`small-gap <= 0.40 eV`, `HTO < 0.00 eV`);
- multi-task and single-task neural networks with `LayerNorm + GELU`;
- matched XGBoost, random-forest and ridge baselines;
- parent-grouped nested cross-validation and frozen-fold diagnostics;
- out-of-fold SHAP, donor-twist, orbital-character and within-parent analyses.

Raw quantum-chemical outputs, private server paths, credentials, manuscript files and frozen result archives are intentionally excluded. Supply the audited modelling CSV, feature-block JSON and frozen split directory to the command-line scripts.

The auxiliary archive-derived analysis scripts use the `CALICENE_HTO_ROOT` environment variable for the local analysis root; no machine-specific server path is embedded.

## Layout

`src/calicene_hto/core/` contains the final model, label definitions, Round-2 comparison, diagnostics, SHAP and excitation-state runners.

`src/calicene_hto/analysis/` contains the provenance-controlled dataset, Boltzmann, orbital-character, donor-twist and within-parent analysis scripts.

`docs/` records the evaluation protocol and data contract used for the revised analysis.

## Run the model comparison

```bash
python src/calicene_hto/core/run_round2_comparison.py \
  --data /path/to/dataset_v2.csv \
  --blocks /path/to/feature_blocks.json \
  --round1 /path/to/frozen_round1_split \
  --out /path/to/results/round2
```

The runner applies the same continuous targets and thresholds to every model family. Outer test labels are not used for tuning, calibration or threshold selection.

## Verify the implementation

```bash
python src/calicene_hto/core/test_round2_pipeline.py \
  --data /path/to/dataset_v2.csv \
  --blocks /path/to/feature_blocks.json \
  --round1 /path/to/frozen_round1_split
```

The expected primary analysis set is 159 intended-species conformers from 52 parent molecules with 117 audited descriptors. The accepted library contains 184 records; the four rejected records and 25 accepted rearranged-species records are tracked separately in the provenance audit.

## Environment

Python 3.10+ is recommended. Install the packages in `requirements.txt`; GPU execution is optional and controlled by PyTorch.

## Provenance

The scripts were extracted from the frozen revised-analysis bundle `FROZEN_CAJ_REVISION_20260919`. The manuscript reports associations and method sensitivity; the code does not treat feature attribution as mechanistic proof.
