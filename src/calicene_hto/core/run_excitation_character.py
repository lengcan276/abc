"""Dataset-wide excited-state character analysis for Reviewer 1 Comment 21.

Also supplies evidence for R1-8 (roots and state tracking), R1-2 (ordering alone does
not establish a pathway) and R1-3 (what "TADF-like" should mean).

Source: evidence/all_conformers_data.csv carries S1..S10 and T1..T10 energies plus
S1..S10 oscillator strengths. Its S1/T1/T2 energies are first reconciled against the
gaps stored in the modelling dataset; rows that disagree are excluded from the
character analysis and reported separately, because a disagreement means the two
tables do not describe the same calculation.

Nothing here feeds the models. It characterises the dataset.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from label_definitions import TASKS, TARGETS, THRESHOLDS

N_ROOT = 10
S_E = ['s%d_energy_ev' % i for i in range(1, N_ROOT + 1)]
T_E = ['t%d_energy_ev' % i for i in range(1, N_ROOT + 1)]
S_F = ['s%d_oscillator' % i for i in range(1, N_ROOT + 1)]
DARK = 0.01          # oscillator strength below which a singlet is treated as dark
BRIGHT = 0.05        # above which it is treated as clearly bright


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def boot_diff(values, flag, groups, n=2000, seed=20260913):
    """Difference in mean between flag==1 and flag==0, resampling parent molecules."""
    rng = np.random.default_rng(seed)
    parents = np.unique(groups)
    index = {p: np.flatnonzero(groups == p) for p in parents}
    out = []
    for _ in range(n):
        ids = np.concatenate([index[p] for p in rng.choice(parents, len(parents), replace=True)])
        v, f = values[ids], flag[ids]
        if f.sum() == 0 or (1 - f).sum() == 0:
            continue
        out.append(float(v[f == 1].mean() - v[f == 0].mean()))
    if not out:
        return None, None, 0
    return float(np.quantile(out, .025)), float(np.quantile(out, .975)), len(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='modelling dataset (dataset_v2.csv)')
    p.add_argument('--states', required=True, help='evidence/all_conformers_data.csv')
    p.add_argument('--out', required=True)
    p.add_argument('--tol', type=float, default=0.01, help='eV tolerance for reconciliation')
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = pd.read_csv(a.data)
    ds = ds[ds.intended_species == 1].reset_index(drop=True)
    ev = pd.read_csv(a.states)
    cols = ['Molecule', 'conformer'] + S_E + T_E + S_F
    have = [c for c in dict.fromkeys(cols) if c in ev.columns]
    miss = [c for c in cols if c not in ev.columns]
    if miss:
        print('missing state columns: %s' % ', '.join(miss))
    df = ds[['Molecule', 'conformer'] + TASKS + [TARGETS[t] for t in TASKS]].merge(
        ev[have], on=['Molecule', 'conformer'], how='left', validate='one_to_one')
    print('merged %d rows, %d with state data' % (len(df), df.s1_energy_ev.notna().sum()))

    # ---- 0. Reconcile the two energy sources before using either (R1-5, R1-9).
    d1 = (df[TARGETS['small_gap_label']] - (df.s1_energy_ev - df.t1_energy_ev)).abs()
    d2 = (df[TARGETS['HTO_label']] - (df.s1_energy_ev - df.t2_energy_ev)).abs()
    df['reconciled'] = (d1 <= a.tol) & (d2 <= a.tol)
    df['recon_dev_S1T1'] = d1
    df['recon_dev_S1T2'] = d2
    bad = df[~df.reconciled]
    bad[['Molecule', 'conformer', 'recon_dev_S1T1', 'recon_dev_S1T2',
         TARGETS['small_gap_label'], 's1_energy_ev', 't1_energy_ev', 't2_energy_ev']] \
        .sort_values('recon_dev_S1T1', ascending=False).to_csv(out / 'unreconciled_rows.csv', index=False)
    print('\nreconciliation at %.3f eV: %d of %d rows agree; %d excluded'
          % (a.tol, int(df.reconciled.sum()), len(df), len(bad)))
    if len(bad):
        print('excluded molecules: %s' % ', '.join(sorted(bad.Molecule.unique())))

    w = df[df.reconciled].reset_index(drop=True)
    groups = w.Molecule.to_numpy()
    se = w[S_E].to_numpy(float)
    te = w[T_E].to_numpy(float)
    sf = w[S_F].to_numpy(float)

    # ---- 1. Is S1 bright? (R1-21, bears on R1-3's definition of TADF-like)
    w['f_S1'] = sf[:, 0]
    bright_idx = np.where((sf >= BRIGHT).any(1), (sf >= BRIGHT).argmax(1) + 1, np.nan)
    w['lowest_bright_state'] = bright_idx
    offset = np.full(len(w), np.nan)
    ok = np.isfinite(bright_idx)
    offset[ok] = se[ok, (bright_idx[ok] - 1).astype(int)] - se[ok, 0]
    w['bright_offset_eV'] = offset
    br = dict(n=len(w), median_f_S1=float(np.median(sf[:, 0])),
              frac_f_S1_below_0p01=float((sf[:, 0] < DARK).mean()),
              frac_f_S1_below_0p001=float((sf[:, 0] < 0.001).mean()),
              frac_no_bright_state_in_10=float((~ok).mean()),
              median_bright_offset_eV=float(np.nanmedian(offset)),
              frac_bright_above_S1_by_0p3eV=float(np.nanmean(offset > 0.3)))
    print('\nS1 brightness across the dataset:')
    for k, v in br.items():
        print('  %-32s %s' % (k, ('%.4f' % v) if isinstance(v, float) else v))

    # ---- 2. Triplet manifold below S1 (R1-2: ordering alone is not a pathway)
    n_below = (te < se[:, [0]]).sum(1)
    w['n_triplets_below_S1'] = n_below
    print('\ntriplets below S1: median %d, range %d-%d' % (np.median(n_below), n_below.min(), n_below.max()))
    chk = w.groupby(TASKS[1]).n_triplets_below_S1.agg(['mean', 'min', 'max', 'size'])
    print(chk.to_string())
    consistent = bool((w.loc[w[TASKS[1]] == 1, 'n_triplets_below_S1'] == 1).all())
    print('  HTO-positive rows have exactly one triplet below S1: %s '
          '(required by the label definition; a state-ordering check)' % consistent)

    # ---- 3. Is "T2" a well-defined state? (R1-8, qualifies R2-3)
    w['gap_T2_T1'] = te[:, 1] - te[:, 0]
    w['gap_T3_T2'] = te[:, 2] - te[:, 1]
    deg = dict(median_T3_T2_eV=float(w.gap_T3_T2.median()),
               frac_T3_T2_below_0p05=float((w.gap_T3_T2 < 0.05).mean()),
               frac_T3_T2_below_0p10=float((w.gap_T3_T2 < 0.10).mean()),
               median_T2_T1_eV=float(w.gap_T2_T1.median()),
               frac_T2_T1_below_0p05=float((w.gap_T2_T1 < 0.05).mean()))
    print('\nrobustness of the T2 assignment:')
    for k, v in deg.items():
        print('  %-28s %.4f' % (k, v))

    # ---- 4. Character contrast between HTO-positive and HTO-negative (R1-21)
    rows = []
    for name in ['f_S1', 'n_triplets_below_S1', 'gap_T2_T1', 'gap_T3_T2', 'bright_offset_eV']:
        v = w[name].to_numpy(float)
        keep = np.isfinite(v)
        for task in TASKS:
            flag = w[task].to_numpy(float)
            lo, hi, n = boot_diff(v[keep], flag[keep], groups[keep], a.bootstrap)
            m1 = float(v[keep][flag[keep] == 1].mean()) if (flag[keep] == 1).any() else None
            m0 = float(v[keep][flag[keep] == 0].mean()) if (flag[keep] == 0).any() else None
            rows.append(dict(variable=name, task=task, mean_positive=m1, mean_negative=m0,
                             difference=(None if m1 is None or m0 is None else m1 - m0),
                             ci_low=lo, ci_high=hi, draws=n,
                             excludes_zero=bool(lo is not None and (lo > 0) == (hi > 0))))
    con = pd.DataFrame(rows)
    con.to_csv(out / 'character_contrast.csv', index=False)
    print('\nexcited-state character, positive vs negative (parent-cluster bootstrap):')
    print(con[['variable', 'task', 'mean_positive', 'mean_negative', 'difference',
               'ci_low', 'ci_high', 'excludes_zero']].to_string(index=False))

    w.to_csv(out / 'excitation_character_rows.csv', index=False)
    write_json(out / 'excitation_protocol.json', dict(
        n_rows_total=len(df), n_reconciled=int(df.reconciled.sum()),
        n_excluded=int((~df.reconciled).sum()), tolerance_eV=a.tol,
        excluded_molecules=sorted(bad.Molecule.unique().tolist()),
        roots=N_ROOT, dark_threshold=DARK, bright_threshold=BRIGHT,
        brightness=br, degeneracy=deg,
        hto_one_triplet_below_S1=consistent,
        note='Character analysis only. Nothing here is used as a model input or to '
             'select anything. Rows whose stored gaps disagree with the state table '
             'by more than the tolerance are excluded and listed, because the two '
             'sources then do not describe the same calculation.'))
    print('\nwrote %s' % out)


if __name__ == '__main__':
    main()
