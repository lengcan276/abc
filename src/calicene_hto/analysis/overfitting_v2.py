"""Overfitting diagnostics for the DNN under the leakage-free protocol (R2-1, R1-15, R1-11).

Intended-species subset, feature set inherited_94 + conformer block (117), LOMO unless stated.
 1. train-vs-OOF gap : pooled ROC-AUC on the inner-training rows vs pooled OOF, all five models, 3 seeds
 2. capacity ablation: MT-DNN with hidden (256,128,64) | (64,32,16) | (16,8,4) | linear head (0 hidden), 3 seeds
 3. label permutation: parent-preserving label shuffle x10, MT-DNN and Logistic, seed 42 (OOF AUC should be ~0.5)
 4. learning curve   : GROUPED (10 splits x 3 seeds), training parents subsampled 40/60/80/100 %, MT-DNN and Logistic
Each stage writes its own CSV so a timeout leaves partial but usable output.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json, copy, time, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
P2 = Path(os.environ.get('CALICENE_HTO_ROOT', '.')).resolve() / 'revision_CAJ_20260910/phase2'
sys.path.insert(0, str(P2 / 'scripts'))
from run_models_v2 import fit_predict, preprocess, inner_split, nn_predict, CFG, TASKS
torch.set_num_threads(2)
OUT = P2 / 'reports/overfitting_v2'; OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(P2 / 'data/model_v2/dataset_v2.csv'); d = d[d.intended_species == 1].reset_index(drop=True)
blocks = json.loads((P2 / 'data/model_v2/feature_blocks.json').read_text())
feats = blocks['inherited_94'] + blocks['conformer_block']
X = d[feats].to_numpy(float); Y = d[TASKS].to_numpy(float); groups = d.Molecule.to_numpy()
TASKN = CFG['tasks']; t0 = time.time()


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


class NetCfg(nn.Module):
    def __init__(self, dim, hidden, heads=2):
        super().__init__()
        layers, last = [], dim
        for i, h in enumerate(hidden[:-1] if hidden else []):
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(.3)]; last = h
        self.shared = nn.Sequential(*layers) if layers else nn.Identity()
        hh = hidden[-1] if hidden else None
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(last, hh), nn.ReLU(), nn.Dropout(.2), nn.Linear(hh, 1)) if hh else nn.Linear(last, 1) for _ in range(heads)])

    def forward(self, x):
        z = self.shared(x); return torch.cat([h(z) for h in self.heads], 1)


def train_cfg(Xtr, Ytr, Xva, Yva, seed, hidden):
    torch.manual_seed(seed); np.random.seed(seed)
    m = NetCfg(Xtr.shape[1], hidden); opt = torch.optim.Adam(m.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    pos = Ytr.sum(0); pw = np.divide(len(Ytr) - pos, pos, out=np.ones_like(pos, dtype=float), where=pos > 0)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32))
    x, y, v, w = [torch.tensor(a, dtype=torch.float32) for a in (Xtr, Ytr, Xva, Yva)]
    best, saved, bad = np.inf, copy.deepcopy(m.state_dict()), 0
    for ep in range(CFG['epochs']):
        m.train(); opt.zero_grad(); l = loss(m(x), y); l.backward(); opt.step(); m.eval()
        with torch.no_grad(): vl = float(loss(m(v), w))
        if vl < best - 1e-6: best, saved, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= CFG['patience']: break
    m.load_state_dict(saved); m.eval(); return m, ep + 1, sum(p.numel() for p in m.parameters())


def lomo(seed):
    return list(LeaveOneGroupOut().split(X, groups=groups))


# ---------- 1. train vs OOF ----------
rows = []
for seed in CFG['seeds']:
    oof = {m: np.zeros((len(d), 2)) for m in CFG['models']}; trn = {m: [] for m in CFG['models']}
    for outer, te in lomo(seed):
        tr, va = inner_split(groups, outer, seed); xt, xv, xe, _ = preprocess(X, tr, va, te)
        for model in CFG['models']:
            _, pt, _, _ = fit_predict(model, xt, Y[tr], xv, Y[va], np.vstack([xe, xt]), seed)
            oof[model][te] = pt[:len(te)]
            trn[model].append([auc(Y[tr, j], pt[len(te):, j]) for j in range(2)])
    for model in CFG['models']:
        ta = np.nanmean(np.array(trn[model]), 0)
        for j, task in enumerate(TASKN):
            rows.append(dict(seed=seed, model=model, task=task, train_auc_mean_over_folds=ta[j], oof_auc=auc(Y[:, j], oof[model][:, j]), gap=ta[j] - auc(Y[:, j], oof[model][:, j])))
    print(f'[1] seed {seed} done {time.time() - t0:.0f}s', flush=True)
r1 = pd.DataFrame(rows); r1.to_csv(OUT / 'train_vs_oof_gap.csv', index=False)
print(r1.groupby(['model', 'task'])[['train_auc_mean_over_folds', 'oof_auc', 'gap']].mean().round(3).to_string(), flush=True)

# ---------- 2. capacity ablation ----------
rows = []
for name, hidden in [('256-128-64 (paper)', (256, 128, 64)), ('64-32-16', (64, 32, 16)), ('16-8-4', (16, 8, 4)), ('linear head', ())]:
    for seed in CFG['seeds']:
        oof = np.zeros((len(d), 2)); eps, npar = [], 0
        for outer, te in lomo(seed):
            tr, va = inner_split(groups, outer, seed); xt, xv, xe, _ = preprocess(X, tr, va, te)
            m, ep, npar = train_cfg(xt, Y[tr], xv, Y[va], seed, hidden); oof[te] = nn_predict(m, xe); eps.append(ep)
        for j, task in enumerate(TASKN):
            rows.append(dict(architecture=name, n_params=npar, seed=seed, task=task, oof_auc=auc(Y[:, j], oof[:, j]), oof_ap=average_precision_score(Y[:, j], oof[:, j]), mean_epochs=np.mean(eps)))
    print(f'[2] {name} done {time.time() - t0:.0f}s', flush=True)
r2 = pd.DataFrame(rows); r2.to_csv(OUT / 'capacity_ablation.csv', index=False)
print(r2.groupby(['architecture', 'task'])[['n_params', 'oof_auc', 'oof_ap', 'mean_epochs']].mean().round(3).to_string(), flush=True)

# ---------- 3. parent-preserving label permutation ----------
rng = np.random.default_rng(20260911)


def parent_shuffle(Y):
    out = Y.copy(); by = {}
    for p in np.unique(groups): by.setdefault(int((groups == p).sum()), []).append(p)
    for size, ps in by.items():
        perm = rng.permutation(len(ps))
        for a, b in zip(ps, [ps[i] for i in perm]): out[groups == a] = Y[groups == b]
    return out


rows = []
for k in range(10):
    Yp = parent_shuffle(Y)
    oof = {m: np.zeros((len(d), 2)) for m in ['MT-DNN', 'Logistic']}
    for outer, te in lomo(42):
        tr, va = inner_split(groups, outer, 42); xt, xv, xe, _ = preprocess(X, tr, va, te)
        for model in oof:
            _, pt, _, _ = fit_predict(model, xt, Yp[tr], xv, Yp[va], xe, 42); oof[model][te] = pt
    for model in oof:
        for j, task in enumerate(TASKN):
            rows.append(dict(permutation=k, model=model, task=task, oof_auc=auc(Yp[:, j], oof[model][:, j])))
    print(f'[3] perm {k} done {time.time() - t0:.0f}s', flush=True)
r3 = pd.DataFrame(rows); r3.to_csv(OUT / 'label_permutation.csv', index=False)
print(r3.groupby(['model', 'task']).oof_auc.agg(['mean', 'std', 'max']).round(3).to_string(), flush=True)

# ---------- 4. learning curve ----------
rows = []
for seed in CFG['seeds']:
    for k, (outer, te) in enumerate(GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=seed).split(X, groups=groups)):
        tr, va = inner_split(groups, outer, seed)
        pars = np.array(sorted(set(groups[tr]))); r_ = np.random.default_rng(seed * 100 + k)
        for frac in [0.4, 0.6, 0.8, 1.0]:
            sel = set(r_.choice(pars, max(3, int(round(frac * len(pars)))), replace=False)) if frac < 1 else set(pars)
            sub = np.array([i for i in tr if groups[i] in sel])
            if Y[sub].sum(0).min() < 2: continue
            xt, xv, xe, _ = preprocess(X, sub, va, te)
            for model in ['MT-DNN', 'Logistic']:
                _, pt, _, _ = fit_predict(model, xt, Y[sub], xv, Y[va], xe, seed)
                for j, task in enumerate(TASKN):
                    rows.append(dict(seed=seed, split=k, fraction=frac, n_train_parents=len(sel), n_train=len(sub), model=model, task=task, test_auc=auc(Y[te, j], pt[:, j])))
    print(f'[4] seed {seed} done {time.time() - t0:.0f}s', flush=True)
r4 = pd.DataFrame(rows); r4.to_csv(OUT / 'learning_curve.csv', index=False)
print(r4.groupby(['model', 'task', 'fraction']).test_auc.agg(['mean', 'std', 'count']).round(3).to_string(), flush=True)
(OUT / 'COMPLETE.json').write_text(json.dumps(dict(elapsed_seconds=time.time() - t0, n=len(d), features=len(feats))))
print('ALL DONE', flush=True)
