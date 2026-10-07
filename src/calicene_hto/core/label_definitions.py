"""Label definitions for the round-2 formulation. No heavy imports, so these can be
audited without torch.

Verified against snapshot/dataset_v2.csv (159 intended-species rows, all values present):

    small_gap_label == (gap_S1_T1_eV <= 0.4)    range 0.006 .. 1.565
    HTO_label       == (gap_S1_T2_eV <  0.0)    range -0.328 .. 1.313

Both labels are exact thresholdings of fully observed continuous quantities. Round 2
regresses those quantities and applies these thresholds unchanged. The thresholds are
PRESPECIFIED (they are the published label definitions) and are never fitted.
"""

TASKS = ['small_gap_label', 'HTO_label']

# Task -> continuous quantity that defines it.
TARGETS = {'small_gap_label': 'gap_S1_T1_eV', 'HTO_label': 'gap_S1_T2_eV'}

# Task -> threshold. The positive class is "gap below threshold" for both tasks.
THRESHOLDS = {'small_gap_label': 0.4, 'HTO_label': 0.0}

# Strictness of the comparison, kept explicit so the assertions in the runner and the
# tests cannot drift from the published definitions.
STRICT = {'small_gap_label': False, 'HTO_label': True}   # False -> <=, True -> <

# Columns that must never be used as model inputs (carried over from round 1).
FORBIDDEN = {'small_gap_label', 'HTO_label', 'gap_S1_T1_eV', 'gap_S1_T2_eV',
             'DA_st_gap_effect', 'st_average_energy', 'aromatic_gap_product',
             'conformer_energy'}


def positive_mask(gap_values, task):
    """Reproduce a task's label from its continuous quantity."""
    thr = THRESHOLDS[task]
    return gap_values < thr if STRICT[task] else gap_values <= thr
