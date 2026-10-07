"""Round-2 guards. Torch-free tests run anywhere; net tests skip without torch.

The single-task threshold test exists because an earlier draft indexed the global
THR array positionally, so the single-task HTO group (local column 0, true threshold
0.0 eV) was thresholded at 0.4 eV -- silently wrong, never raised.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TASKS = ['small_gap_label', 'HTO_label']
GAPS = ['gap_S1_T1_eV', 'gap_S1_T2_eV']
THR = np.array([0.4, 0.0])
FAIL = []


def check(name, cond, detail=''):
    print(('PASS  ' if cond else 'FAIL  ') + name + (('  ' + detail) if detail else ''))
    if not cond:
        FAIL.append(name)


def load(data_csv, blocks_json, round1):
    data = pd.read_csv(data_csv)
    df = data[data.intended_species == 1].reset_index(drop=True)
    blocks = json.loads(Path(blocks_json).read_text(encoding='utf-8-sig'))
    feats = blocks['inherited_94'] + blocks['conformer_block']
    return df, feats, Path(round1)


def test_threshold_mapping():
    """Every task must carry its OWN threshold, whatever its local column index."""
    from label_definitions import THRESHOLDS, TARGETS, positive_mask
    check('THRESHOLDS names match TASKS', sorted(THRESHOLDS) == sorted(TASKS))
    check('positive_mask small_gap is inclusive',
          bool(positive_mask(np.array([0.4]), 'small_gap_label')[0]))
    check('positive_mask HTO is strict',
          not bool(positive_mask(np.array([0.0]), 'HTO_label')[0]))
    check('small_gap threshold is 0.4', THRESHOLDS['small_gap_label'] == 0.4)
    check('HTO threshold is 0.0', THRESHOLDS['HTO_label'] == 0.0)
    check('TARGETS map to the continuous gaps',
          [TARGETS[t] for t in TASKS] == GAPS)
    for js, want in ([[0, 1], [0.4, 0.0]], [[0], [0.4]], [[1], [0.0]]):
        got = THR[js].tolist()
        check('THR[%s] == %s' % (js, want), got == want, 'got %s' % got)


def test_single_task_hto_uses_zero(df, feats):
    """fit_flat on the HTO group alone must threshold at 0.0, not 0.4."""
    try:
        from run_round2_comparison import fit_flat
    except ImportError as e:
        print('SKIP  single-task threshold test (%s; run on the server)' % e)
        return
    x = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy(float)
    g = df[GAPS].to_numpy(float)
    js = [1]
    score, _ = fit_flat('Ridge', dict(alpha=1.0), x[:120], g[:120][:, js], x[120:], 42, THR[js])
    # score = thr - prediction; recover the implied threshold from a known prediction.
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1.0).fit(x[:120], g[:120, 1])
    implied = score[:, 0] + m.predict(x[120:])
    check('single-task HTO implied threshold == 0.0',
          np.allclose(implied, 0.0, atol=1e-8), 'mean %.6f' % implied.mean())
    score2, _ = fit_flat('Ridge', dict(alpha=1.0), x[:120], g[:120][:, [0]], x[120:], 42, THR[[0]])
    m2 = Ridge(alpha=1.0).fit(x[:120], g[:120, 0])
    implied2 = score2[:, 0] + m2.predict(x[120:])
    check('single-task small_gap implied threshold == 0.4',
          np.allclose(implied2, 0.4, atol=1e-8), 'mean %.6f' % implied2.mean())


def test_labels_and_splits(df, feats, r1):
    g = df[GAPS].to_numpy(float)
    y = df[TASKS].to_numpy(float)
    groups = df.Molecule.to_numpy()
    check('small_gap_label == (gap_S1_T1 <= 0.4)',
          ((g[:, 0] <= 0.4).astype(int) == y[:, 0].astype(int)).all())
    check('HTO_label == (gap_S1_T2 < 0.0)',
          ((g[:, 1] < 0.0).astype(int) == y[:, 1].astype(int)).all())
    check('117 unique features', len(feats) == len(set(feats)) == 117)
    forbidden = {'small_gap_label', 'HTO_label', 'gap_S1_T1_eV', 'gap_S1_T2_eV',
                 'DA_st_gap_effect', 'st_average_energy', 'aromatic_gap_product',
                 'conformer_energy'}
    check('no forbidden column used as a feature', not set(feats) & forbidden)
    rows = pd.read_csv(r1 / 'rows.csv')
    check('round-1 row order preserved',
          (rows.Molecule.to_numpy() == groups).all()
          and (rows.conformer.to_numpy() == df.conformer.to_numpy()).all())
    splits = json.loads((r1 / 'splits.json').read_text(encoding='utf-8'))
    check('5 frozen outer folds', len(splits) == 5)
    ok_outer = ok_inner = ok_cover = True
    seen = set()
    for s in splits:
        tr, te = np.array(s['train_rows']), np.array(s['test_rows'])
        ok_outer &= not (set(groups[tr]) & set(groups[te]))
        seen |= set(te.tolist())
        for i in s['inner']:
            a, b = np.array(i['train_rows']), np.array(i['val_rows'])
            ok_inner &= not (set(groups[a]) & set(groups[b]))
            ok_cover &= (set(a) | set(b)) == set(tr.tolist())
    check('outer folds parent-disjoint', ok_outer)
    check('inner folds parent-disjoint', ok_inner)
    check('inner folds partition the outer train rows', ok_cover)
    check('outer test folds cover every row', seen == set(range(len(df))),
          '%d of %d' % (len(seen), len(df)))


def test_net_shapes():
    try:
        import torch
    except ImportError:
        print('SKIP  net shape tests (torch not installed here; run on the server)')
        return
    from optimized_architectures import GapNet
    x = torch.randn(9, 117)
    gid = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    for ctx, priv, cls in [(0, 0, True), (16, 8, True), (16, 0, False), (0, 16, False)]:
        m = GapNet(117, width=64, n_task=2, private=priv, context=ctx, cls_head=cls)
        reg, c = m(x, gid if ctx else None)
        check('GapNet ctx=%d priv=%d cls=%s forward' % (ctx, priv, cls),
              reg.shape == (9, 2) and (c is None if not cls else c.shape == (9, 2)))
        p = m.probabilities(reg, c, torch.tensor([0.5, -0.3]), 0.5)
        check('  probabilities in [0,1]', bool((p >= 0).all() and (p <= 1).all()))
    m1 = GapNet(117, width=32, n_task=1, private=0, context=0, cls_head=True)
    reg, c = m1(x)
    check('single-task GapNet forward', reg.shape == (9, 1) and c.shape == (9, 1))
    # A lower predicted gap must give a higher positive probability.
    mm = GapNet(4, width=8, n_task=1, cls_head=False)
    with torch.no_grad():
        lo = mm.probabilities(torch.tensor([[-2.]]), None, torch.tensor([0.]), 1.)
        hi = mm.probabilities(torch.tensor([[2.]]), None, torch.tensor([0.]), 1.)
    check('probability decreases with predicted gap', float(lo) > float(hi),
          '%.3f vs %.3f' % (float(lo), float(hi)))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(HERE / 'snapshot/dataset_v2.csv'))
    p.add_argument('--blocks', default=str(HERE / 'snapshot/feature_blocks.json'))
    p.add_argument('--round1', default=str(HERE / 'nested_20260912'))
    a = p.parse_args()
    df, feats, r1 = load(a.data, a.blocks, a.round1)
    test_threshold_mapping()
    test_labels_and_splits(df, feats, r1)
    test_single_task_hto_uses_zero(df, feats)
    test_net_shapes()
    print('\n%d failed' % len(FAIL))
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
