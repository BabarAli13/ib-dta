"""
Deletion-based faithfulness audit for drug-target interaction models.

This is the measurement instrument for the paper's central claim. It is
model-agnostic: anything that can (a) produce a prediction and (b) emit a
per-pair or per-token importance map can be audited, including DrugBAN,
GCA, attention-based DTA models, and IB-DTA itself.

The question
------------
An explanation says "these contacts matter". Does the model agree? Delete
the contacts the explanation ranks highest and watch the prediction move.
If it moves no faster than deleting random contacts, the explanation is
describing something other than the model's computation.

Three curves are produced per batch:

    most    delete highest-ranked pairs first
    random  delete pairs in random order (the null)
    least   delete lowest-ranked pairs first

A faithful explanation gives most > random > least in area under the
prediction-change curve. The headline scalar is

    AFS = AUC(most) - AUC(random)          "above-random faithfulness"

normalised by the total achievable change, so it is comparable across
models and datasets. AFS <= 0 means the explanation is no better than
chance -- and a negative AFS means it is actively anti-correlated with
importance, which is the failure mode randomised baselines exist to catch.

The exactness check
-------------------
For an additive bottleneck the predicted change is *analytically* known:
removing a set T changes the output by exactly sum of contributions in T.
`check_additive_exactness` verifies this numerically. For a pooled model
(DrugBAN) the same check fails by construction, and the size of the failure
is itself a reportable quantity -- it measures how much of the prediction
flows around the interaction map.
"""

import numpy as np
import torch


# --------------------------------------------------------------------------
# Core curve machinery
# --------------------------------------------------------------------------
def default_fractions(lo=0.001, hi=0.5, n=13):
    """Log-spaced deletion fractions, plus zero.

    Linear spacing is the obvious choice and the wrong one. A sparse model
    puts all its mass on a handful of pairs, so a linear grid starting at
    5% deletes the entire support in its first step and the curve is a
    step function -- it shows that the explanation matters but says nothing
    about *how concentrated* it is. Log spacing resolves the first
    percent, which is where a sparse model's whole story lives.
    """
    return np.concatenate([[0.0], np.geomspace(lo, hi, n)])


@torch.no_grad()
def deletion_curve(predict_fn, importance, fractions=None, order="most",
                   generator=None):
    """Prediction change as pairs are progressively deleted.

    Parameters
    ----------
    predict_fn : callable(drop_mask) -> Tensor (B,)
        Runs the model with `drop_mask` applied. drop_mask is a bool tensor
        shaped like `importance`, True = delete this pair. The caller owns
        batching and device placement.
    importance : Tensor (B, ...) float
        Per-pair importance. Ranked by magnitude: sign is not importance.
    fractions : sequence of float in [0, 1]
        Deletion fractions to evaluate. Default: 0 to 50% in 11 steps.
        Deleting beyond ~50% pushes the input so far off-distribution that
        the curve stops measuring explanation quality and starts measuring
        robustness to garbage.
    order : {'most', 'least', 'random'}
    generator : torch.Generator, optional
        For reproducible random ordering.

    Returns
    -------
    fractions : ndarray (F,)
    deltas    : ndarray (F,)   mean |y_full - y_deleted| at each fraction
    """
    if fractions is None:
        fractions = default_fractions()
    fractions = np.asarray(fractions, dtype=float)

    B = importance.shape[0]
    flat = importance.reshape(B, -1)
    n_pairs = flat.shape[1]

    if order == "random":
        scores = torch.rand(flat.shape, device=flat.device, generator=generator)
    elif order == "most":
        scores = flat.abs()
    elif order == "least":
        # Pairs with exactly zero importance are *outside* the support. Ranking
        # them as "least important" is correct but makes the least-curve
        # trivially flat for sparse models; that is a true property, not a bug.
        scores = -flat.abs()
    else:
        raise ValueError(f"unknown order: {order}")

    ranking = scores.argsort(dim=-1, descending=True)

    zero_mask = torch.zeros_like(flat, dtype=torch.bool)
    y_full = predict_fn(zero_mask.reshape(importance.shape)).reshape(-1).float()

    deltas = []
    for f in fractions:
        k = int(round(f * n_pairs))
        drop = torch.zeros_like(flat, dtype=torch.bool)
        if k > 0:
            drop.scatter_(1, ranking[:, :k], True)
        y_del = predict_fn(drop.reshape(importance.shape)).reshape(-1).float()
        deltas.append((y_full - y_del).abs().mean().item())

    return fractions, np.asarray(deltas)


def audit(predict_fn, importance, fractions=None, seed=0):
    """Run all three orderings and compute the faithfulness scalars.

    Returns a dict with the curves and:
        afs       AUC(most) - AUC(random), normalised  -- the headline number
        polarity  AUC(most) - AUC(least), normalised   -- does ranking order
                  track importance at all?
    """
    g = torch.Generator(device=importance.device)
    g.manual_seed(seed)

    fr, d_most = deletion_curve(predict_fn, importance, fractions, "most")
    _, d_rand = deletion_curve(predict_fn, importance, fractions, "random", generator=g)
    _, d_least = deletion_curve(predict_fn, importance, fractions, "least")

    auc_most = float(np.trapezoid(d_most, fr))
    auc_rand = float(np.trapezoid(d_rand, fr))
    auc_least = float(np.trapezoid(d_least, fr))

    # Normalise by the largest area seen so the score is scale-free across
    # models whose outputs live on different ranges (pKd vs logit).
    denom = max(auc_most, auc_rand, auc_least, 1e-12)

    return {
        "fractions": fr,
        "most": d_most,
        "random": d_rand,
        "least": d_least,
        "auc_most": auc_most,
        "auc_random": auc_rand,
        "auc_least": auc_least,
        "afs": (auc_most - auc_rand) / denom,
        "polarity": (auc_most - auc_least) / denom,
    }


# --------------------------------------------------------------------------
# The exactness property that only an additive bottleneck has
# --------------------------------------------------------------------------
@torch.no_grad()
def check_additive_exactness(predict_fn, contrib, drop_mask, atol=1e-4):
    """Verify |y - y_dropped| == |sum of dropped contributions|.

    Holds by construction for IB-DTA. For a pooled model the residual is
    large, and reporting that residual quantifies how much prediction mass
    bypasses the interaction map.

    Returns (max_abs_residual, passed, detail_dict).
    """
    zero = torch.zeros_like(drop_mask, dtype=torch.bool)
    y_full = predict_fn(zero).reshape(-1).float()
    y_drop = predict_fn(drop_mask).reshape(-1).float()

    observed = (y_full - y_drop)
    dims = tuple(range(1, contrib.dim()))
    predicted = (contrib * drop_mask.float()).sum(dim=dims)

    residual = (observed - predicted).abs()
    max_res = residual.max().item()
    scale = observed.abs().mean().clamp_min(1e-12).item()

    return max_res, bool(max_res <= atol), {
        "max_residual": max_res,
        "mean_residual": residual.mean().item(),
        "relative_residual": residual.mean().item() / scale,
        "mean_observed_delta": observed.abs().mean().item(),
    }


# --------------------------------------------------------------------------
# Adapters: make a baseline auditable
# --------------------------------------------------------------------------
class DrugBANAdapter:
    """Wraps DrugBAN so its bilinear attention map can be audited.

    DrugBAN has no mechanism for deleting an interaction pair -- which is
    the point. We delete by zeroing the attention map entries before
    `attention_pooling`, which is the most generous possible reading of
    "remove this contact from the explanation". If the prediction barely
    moves even under that intervention, the explanation is not driving it.

    Requires a small hook into ban.py: BANLayer.forward must accept an
    optional `drop_pairs` tensor. See patch_ban.py.
    """

    def __init__(self, model, bg_d, v_p, head=0):
        self.model = model
        self.bg_d = bg_d
        self.v_p = v_p
        self.head = head
        self.model.eval()

    @torch.no_grad()
    def importance(self):
        _, _, _, att = self.model(self.bg_d, self.v_p, mode="eval")
        return att[:, self.head]            # B,V,Q

    def predict_fn(self):
        def fn(drop_mask):
            # drop_mask: B,V,Q -> expand to all heads
            dm = drop_mask.unsqueeze(1).expand(-1, self.model.bcn.h_out, -1, -1)
            _, _, score, _ = self.model(self.bg_d, self.v_p, mode="eval", drop_pairs=dm)
            return score
        return fn


class IBDTAAdapter:
    """Wraps IB-DTA. Importance is the signed contribution, not alpha."""

    def __init__(self, model, bg_d, v_p):
        self.model = model
        self.bg_d = bg_d
        self.v_p = v_p
        self.model.eval()

    @torch.no_grad()
    def importance(self):
        _, _, _, _, contrib = self.model(self.bg_d, self.v_p, mode="eval")
        return contrib                      # B,H,V,Q

    def predict_fn(self):
        def fn(drop_mask):
            _, _, y, _, _ = self.model(
                self.bg_d, self.v_p, mode="eval", drop_pairs=drop_mask)
            return y
        return fn


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def format_report(name, result, exactness=None):
    lines = [
        f"{name}",
        f"  AFS (most - random, normalised) : {result['afs']:+.4f}",
        f"  Polarity (most - least)         : {result['polarity']:+.4f}",
        f"  AUC most / random / least       : "
        f"{result['auc_most']:.4f} / {result['auc_random']:.4f} / {result['auc_least']:.4f}",
    ]
    if result["afs"] <= 0:
        lines.append("  WARNING: explanation is not better than random deletion.")
    if exactness is not None:
        ok = "exact" if exactness[1] else "NOT exact"
        lines.append(f"  Additive deletion bound         : {ok} "
                     f"(max residual {exactness[2]['max_residual']:.3e}, "
                     f"relative {exactness[2]['relative_residual']:.3e})")
    return "\n".join(lines)
