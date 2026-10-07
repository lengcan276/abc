# Final evaluation protocol

The primary analysis uses 159 intended-species conformers from 52 parent molecules and 117 audited descriptors. Every conformer from a parent remains in one outer fold. The five outer folds and three inner folds are supplied as frozen split metadata.

The two continuous targets are `gap_S1_T1_eV` and `gap_S1_T2_eV`. Labels are deterministic thresholdings of those targets: small-gap is `<= 0.40 eV`; HTO is `< 0.00 eV`. These thresholds are fixed before evaluation and are never fitted from outer test predictions.

The neural architecture used by the revised Round-2 protocol is a shared multi-task trunk with linear layers, layer normalization, GELU activation and dropout, followed by task-specific regression heads. The same descriptor matrix, fold protocol and tuning budget are used for MT-DNN, ST-DNN, XGBoost, random forest and ridge regression.
