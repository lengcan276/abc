"""Round-2 architectures: continuous-gap multi-task net for clustered small data.

Rationale (prespecified, not outcome-driven):
  small_gap_label == (gap_S1_T1_eV <= 0.4) and HTO_label == (gap_S1_T2_eV < 0) are
  deterministic thresholdings of two fully observed continuous quantities. Round 1
  trained on the thresholded bits only, discarding 159 real-valued observations and
  leaving 10 positive parent clusters per task as the effective signal. Round 2
  regresses the continuous gaps and derives class probabilities at the PRESPECIFIED
  thresholds. No label is altered; no threshold is chosen from outer scores.

Also fixes three round-1 specification defects: full-batch updates (150 optimiser
steps total), no early stopping, and no epoch selection.
"""
import numpy as np
import torch
from torch import nn

from label_definitions import TASKS, TARGETS, THRESHOLDS, STRICT, FORBIDDEN, positive_mask


class GapNet(nn.Module):
    """Shared trunk -> per-task regression on standardized gaps (+ optional classifier head).

    Optional conformer-set context: attention pooling over the conformers of the same
    parent molecule. Uses only feature vectors, never labels and never conformer_energy
    (which is on the round-1 FORBIDDEN list), so it introduces no target leakage.
    """

    def __init__(self, d, width=64, dropout=.2, n_task=2, private=0, context=0, cls_head=True):
        super().__init__()
        h = max(width//2, 8)
        self.trunk = nn.Sequential(
            nn.Linear(d, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout))
        self.private = nn.ModuleList([
            nn.Sequential(nn.Linear(d, private), nn.GELU(), nn.Dropout(dropout))
            for _ in range(n_task)]) if private else None
        self.context = context
        if context:
            self.attn = nn.Sequential(nn.Linear(h, context), nn.Tanh(), nn.Linear(context, 1))
            self.mix = nn.Linear(2*h, h)
        feat = h + (private or 0)
        self.reg = nn.ModuleList([nn.Linear(feat, 1) for _ in range(n_task)])
        self.cls = nn.ModuleList([nn.Linear(feat, 1) for _ in range(n_task)]) if cls_head else None
        # Per-task softness of the regression-derived probability; learned, then the
        # blend weight between the two probability sources is chosen on inner OOF.
        self.log_tau = nn.Parameter(torch.zeros(n_task))

    def _pool(self, z, groups):
        if not self.context or groups is None:
            return z
        w = self.attn(z).squeeze(-1)
        out = torch.empty_like(z)
        for g in torch.unique(groups):
            m = groups == g
            a = torch.softmax(w[m], 0).unsqueeze(-1)
            out[m] = (a*z[m]).sum(0, keepdim=True).expand(int(m.sum()), -1)
        return self.mix(torch.cat([z, out], 1))

    def forward(self, x, groups=None):
        z = self._pool(self.trunk(x), groups)
        f = [torch.cat([z, self.private[j](x)], 1) if self.private else z
             for j in range(len(self.reg))]
        reg = torch.cat([self.reg[j](f[j]) for j in range(len(self.reg))], 1)
        cls = torch.cat([self.cls[j](f[j]) for j in range(len(self.reg))], 1) if self.cls else None
        return reg, cls

    def probabilities(self, reg, cls, thr_std, alpha):
        """Blend regression-derived and direct-classifier probabilities.

        thr_std: thresholds expressed in the SAME standardization as the regression
        targets (computed on training rows only). alpha in [0,1] is an inner-OOF
        hyperparameter, not fitted on outer data.
        """
        tau = self.log_tau.exp().clamp(1e-3, 10.)
        p_reg = torch.sigmoid((thr_std - reg)/tau)
        if cls is None:
            return p_reg
        return alpha*p_reg + (1-alpha)*torch.sigmoid(cls)


def standardize_targets(g_train):
    mu = np.nanmean(g_train, 0)
    sd = np.nanstd(g_train, 0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def class_weights(y):
    pos = y.sum(0)
    neg = len(y) - pos
    return torch.where((pos > 0) & (neg > 0), neg/pos.clamp_min(1), torch.ones_like(pos))


def masked_huber(pred, target, mask, delta=1.0):
    d = (pred-target)*mask
    a = d.abs()
    loss = torch.where(a <= delta, .5*d.square(), delta*(a-.5*delta))
    return loss.sum(0)/mask.sum(0).clamp_min(1.)


def weighted_bce(logits, labels, pos_weight):
    """Soft-label safe. `labels > 0` would hand the full positive weight to every
    mixup-blended row, so the weight is interpolated by the label value instead.
    Reduces exactly to the hard-label rule when labels are 0/1."""
    w = labels*pos_weight + (1-labels)
    raw = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    return (raw*w).sum(0)/w.sum(0).clamp_min(1e-8)


def parent_batches(groups, batch_parents, rng):
    """Batch by parent molecule so clustered rows stay together (required for the
    conformer-set context and for cluster-aware mixup)."""
    parents = np.unique(groups)
    order = rng.permutation(len(parents))
    for i in range(0, len(parents), batch_parents):
        sel = parents[order[i:i+batch_parents]]
        yield np.flatnonzero(np.isin(groups, sel))


def mixup(x, reg, cls, mask, lam_alpha, rng, device, groups=None):
    """Tabular mixup. Applied to inputs and to BOTH target kinds; for the regression
    target the interpolation is exact, for the class target it yields a soft label.

    When the conformer-set context is active the rows keep their parent ids, so the
    permutation is constrained WITHIN each parent: a row blended across parents would
    then be pooled under a grouping it no longer belongs to.
    """
    if lam_alpha <= 0:
        return x, reg, cls, mask, None
    lam = float(rng.beta(lam_alpha, lam_alpha))
    if groups is None:
        perm = torch.randperm(len(x), device=device)
    else:
        g = groups.detach().cpu().numpy()
        order = np.arange(len(g))
        for v in np.unique(g):
            idx = np.flatnonzero(g == v)
            order[idx] = idx[rng.permutation(len(idx))]
        perm = torch.as_tensor(order, device=device, dtype=torch.long)
    xm = lam*x + (1-lam)*x[perm]
    rm = lam*reg + (1-lam)*reg[perm]
    cm = lam*cls + (1-lam)*cls[perm]
    mm = mask*mask[perm]
    return xm, rm, cm, mm, perm
