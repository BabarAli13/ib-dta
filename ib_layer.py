"""
Sparse additive interaction bottleneck for drug-target prediction.

This module replaces DrugBAN's BANLayer. The difference is one line of
architecture with large consequences.

DrugBAN (ban.py, attention_pooling):

    fusion_logits = einsum('bvk,bvq,bqk->bk', v, att_map, q)

The v x q interaction map is contracted away into a single k-dimensional
joint vector, which is then fed to an MLP. Information reaches the output
through that vector, so no individual (atom, residue) pair is on the
critical path. The attention map is a *correlate* of the prediction.

IB-DTA:

    y_hat = b + sum_{(i,j) in S} w_ij * g(h_i, h_j)

There is no joint vector. The prediction is literally a sum of signed
contributions over a sparse selected support S. Removing a pair from S
changes the output by exactly that pair's contribution -- the attribution
and the causal effect are the same number, by construction.

Selection modes
---------------
entmax   alpha-entmax (alpha=1.5 default). Smooth, differentiable, and
         produces *exact* zeros. Recommended starting point: it trains like
         softmax but sparsifies on its own.
topk     Hard top-k with a straight-through estimator. Gives an exact,
         controllable budget |S| = k. Harder to train from scratch; anneal
         into it from entmax.
softmax  Dense baseline for ablation. Keeps the additive decoder but
         removes sparsity, isolating the contribution of each.

Budget
------
With ADAPTIVE=True the per-complex budget is learned: a concave penalty
Omega(|S|) = sum_ij sqrt(a_ij) (applied to the normalised selection mass)
pushes mass onto few pairs while letting a large ligand against a deep
pocket keep more contacts than a small fragment. Fixed-k sparse attention
is common; instance-adaptive budgets in interaction modelling are not.

Masking
-------
Padded atoms and residues must not be selectable. Pass boolean masks
(True = real) or the model will happily spend its budget on padding and
the explanations will be meaningless.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# --------------------------------------------------------------------------
# alpha-entmax (alpha = 1.5), exact bisection form.
# Vendored rather than imported so the repo has no new hard dependency.
# Reference: Peters, Niculae & Martins (2019), "Sparse Sequence-to-Sequence
# Models". The forward pass is a projection onto the simplex under the
# Tsallis alpha-entropy; unlike softmax it yields exact zeros.
# --------------------------------------------------------------------------
class _Entmax15Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, dim=-1, n_iter=50):
        ctx.dim = dim
        max_val, _ = inputs.max(dim=dim, keepdim=True)
        inputs = inputs - max_val          # shift for numerical stability
        inputs = inputs / 2                # alpha = 1.5 reparameterisation

        lo = inputs.max(dim=dim, keepdim=True)[0] - 1.0
        hi = inputs.max(dim=dim, keepdim=True)[0] - (1.0 / inputs.size(dim)) ** 0.5

        # Bisect on tau. s(tau) = sum(clamp(x - tau, 0)^2) is monotonically
        # DECREASING in tau, so s < 1 means tau is too large and the upper
        # bound must come down. Getting this backwards still converges --
        # to the wrong root -- and shows up as entmax keeping a single
        # entry regardless of the input scale.
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            p = torch.clamp(inputs - mid, min=0) ** 2
            s = p.sum(dim)
            too_high = (s < 1.0).unsqueeze(dim)
            hi = torch.where(too_high, mid, hi)
            lo = torch.where(too_high, lo, mid)

        tau = (lo + hi) / 2
        p = torch.clamp(inputs - tau, min=0) ** 2
        p = p / p.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        (p,) = ctx.saved_tensors
        dim = ctx.dim
        gppr = torch.where(p > 0, p.sqrt(), torch.zeros_like(p))
        d = grad_output * gppr
        q = d.sum(dim=dim, keepdim=True) / gppr.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        return d - q * gppr, None, None


def entmax15(x, dim=-1):
    """alpha-entmax with alpha=1.5. Returns a distribution with exact zeros."""
    return _Entmax15Function.apply(x, dim)


# --------------------------------------------------------------------------
# The bottleneck
# --------------------------------------------------------------------------
class InteractionBottleneck(nn.Module):
    """Sparse additive interaction bottleneck.

    Args
    ----
    v_dim, q_dim : int
        Atom and residue embedding widths (drug side, protein side).
    h_dim : int
        Shared latent width for the compatibility form and the pair decoder.
    heads : int
        Interaction heads. Each head selects its own support; contributions
        are summed. Keep small (1-2) -- more heads means a larger effective
        support and a weaker bottleneck.
    selection : {'entmax', 'topk', 'softmax'}
    budget_k : int
        Pairs kept per head when selection='topk'.
    adaptive_budget : bool
        Learn a per-complex budget via the concave sparsity penalty.
    out_dim : int
        1 for scalar affinity (pKd) or a binary logit.

    Shapes
    ------
    v : (B, V, v_dim)    atom embeddings
    q : (B, Q, q_dim)    residue embeddings
    v_mask : (B, V) bool, True = real atom       (optional but strongly advised)
    q_mask : (B, Q) bool, True = real residue
    returns
      y        : (B, out_dim)         prediction
      contrib  : (B, heads, V, Q)     signed per-pair contribution in output units
      alpha    : (B, heads, V, Q)     selection weights (exact zeros off-support)
    """

    NEG_INF = -1e9

    def __init__(self, v_dim, q_dim, h_dim=128, heads=1,
                 selection="entmax", budget_k=32, adaptive_budget=True,
                 out_dim=1, dropout=0.1, pair_hidden=None,
                 normalize_logits=True, init_temp=0.5413, chunk_q=0,
                 checkpoint_chunks=True):
        super().__init__()
        if selection not in ("entmax", "topk", "softmax"):
            raise ValueError(f"unknown selection mode: {selection}")
        self.heads = heads
        self.h_dim = h_dim
        self.selection = selection
        self.budget_k = budget_k
        self.adaptive_budget = adaptive_budget
        self.out_dim = out_dim
        self.chunk_q = int(chunk_q)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        if heads < 1:
            raise ValueError("heads must be >= 1")
        if budget_k < 1:
            raise ValueError("budget_k must be >= 1")

        # Compatibility form M_ij = (W_a h_i)^T diag(head) (W_r h_j)
        self.v_proj = nn.Linear(v_dim, h_dim * heads)
        self.q_proj = nn.Linear(q_dim, h_dim * heads)
        self.scale = h_dim ** -0.5

        # Pair decoder g_theta. Deliberately small: it scores ONE contact, and
        # a high-capacity decoder here would start doing the work the
        # bottleneck is meant to expose.
        pair_hidden = pair_hidden or h_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * h_dim, pair_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, out_dim),
        )
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Temperature on the compatibility logits; learned, kept positive.
        #
        # IMPORTANT: entmax support size is extremely sensitive to logit
        # scale. Measured on random inputs with V*Q=117 pairs, support
        # collapses to a SINGLE pair once the logit standard deviation
        # exceeds roughly 2, and only reaches ~70% of pairs when it falls
        # below 0.5. The whole usable range lives in std in [0.5, 2].
        #
        # Raw bilinear logits land far outside that band (std ~15 in the
        # same test), so without normalisation entmax degenerates to
        # argmax and the "bottleneck" keeps exactly one contact. That is
        # not sparsity, it is collapse, and it would show up as a model
        # that trains but explains nothing.
        #
        # Standardising the logits per sample before the temperature puts
        # us in the band by construction and makes `temp` interpretable.
        # Measured support after standardisation (117 pairs, any embedding
        # width -- normalisation makes this dimension-independent, which it
        # is NOT without it):
        #
        #     temp 0.5 ->  1%     temp 1.5 -> 66%
        #     temp 0.7 ->  3%     temp 2.0 -> 83%
        #     temp 1.0 -> 17%     temp 3.0 -> 100%
        #
        # init_temp defaults to softplus^-1(1.0) = 0.5413, i.e. temp = 1.0,
        # giving roughly 15-20% support at initialisation: sparse enough to
        # be a bottleneck, dense enough that gradients reach many pairs
        # early in training. The parameter is learned from there.
        self.normalize_logits = normalize_logits
        self.log_temp = nn.Parameter(torch.full((heads,), float(init_temp)))

    # ---- selection -------------------------------------------------------
    def _select(self, logits, pair_mask):
        """logits, pair_mask: (B, H, V, Q) -> alpha (B, H, V, Q), zeros off-support."""
        B, H, V, Q = logits.shape
        logits = logits.masked_fill(~pair_mask, self.NEG_INF)
        flat = logits.reshape(B, H, V * Q)

        if self.selection == "softmax":
            alpha = F.softmax(flat, dim=-1)
        elif self.selection == "entmax":
            alpha = entmax15(flat, dim=-1)
        else:  # topk, straight-through
            k = min(self.budget_k, V * Q)
            soft = F.softmax(flat, dim=-1)
            _, idx = soft.topk(k, dim=-1)
            hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
            # keep the soft mass on selected entries, renormalised
            kept = (soft * hard)
            kept = kept / kept.sum(-1, keepdim=True).clamp_min(1e-12)
            # straight-through: hard support in forward, soft grads in backward
            alpha = hard * kept + (soft - soft.detach()) * hard

        alpha = alpha.reshape(B, H, V, Q)
        # padding can still carry mass if a whole row was masked; zero it hard
        return alpha * pair_mask.float()

    # ---- forward ---------------------------------------------------------
    def forward(self, v, q, v_mask=None, q_mask=None, drop_pairs=None):
        """
        drop_pairs : optional (B, H, V, Q) bool, True = remove this pair.
            Used by the faithfulness audit to delete contacts at inference.
            Because the decoder is additive, deleting a pair removes exactly
            its contribution and nothing else.
        """
        B, V, _ = v.shape
        Q = q.shape[1]
        H, D = self.heads, self.h_dim

        if v_mask is None:
            v_mask = torch.ones(B, V, dtype=torch.bool, device=v.device)
        if q_mask is None:
            q_mask = torch.ones(B, Q, dtype=torch.bool, device=q.device)
        pair_mask = (v_mask[:, None, :, None] & q_mask[:, None, None, :]).expand(B, H, V, Q)

        vh = self.v_proj(v).view(B, V, H, D).permute(0, 2, 1, 3)   # B,H,V,D
        qh = self.q_proj(q).view(B, Q, H, D).permute(0, 2, 1, 3)   # B,H,Q,D

        logits = torch.einsum("bhvd,bhqd->bhvq", vh, qh) * self.scale

        if self.normalize_logits:
            # Standardise over real pairs only, per (sample, head). Padding
            # must not drag the mean, or the statistics shift with ligand
            # size and the effective sparsity varies for the wrong reason.
            pm = pair_mask.float()
            n = pm.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            mean = (logits * pm).sum(dim=(2, 3), keepdim=True) / n
            var = (((logits - mean) * pm) ** 2).sum(dim=(2, 3), keepdim=True) / n
            logits = (logits - mean) / var.clamp_min(1e-8).sqrt()

        temp = F.softplus(self.log_temp).clamp_min(1e-3).view(1, H, 1, 1)
        logits = logits / temp

        alpha = self._select(logits, pair_mask)
        if drop_pairs is not None:
            alpha = alpha * (~drop_pairs).float()

        contrib = self._pair_contributions(vh, qh, alpha)   # B,H,V,Q,out_dim
        y = contrib.sum(dim=(1, 2, 3)) + self.bias          # B,out_dim

        return y, contrib.squeeze(-1), alpha

    def _pair_contributions(self, vh, qh, alpha):
        """alpha * g(h_i, h_j) for every pair, computed in chunks over Q.

        The naive form materialises B x H x V x Q x 2D floats. At V=290,
        Q=1000, D=128, B=64, H=1 that is ~19 TB -- it does not fit anywhere.
        Chunking over the residue axis bounds peak memory at
        B x H x V x chunk x 2D while leaving the result bit-identical,
        because every pair is independent in an additive decoder. This is
        one of the few places where the additive form is also an
        engineering advantage.

        That bound holds unconditionally only at inference. Under autograd
        every chunk's intermediate activations stay alive until backward,
        so the saved tensors add back up to the unchunked total and peak
        training memory is roughly what it would have been without
        chunking. Measured at B=4, V=290, Q=1200, D=128: 0.77 GB with
        no_grad against 4.14 GB with grad. checkpoint_chunks=True (the
        default) recomputes each chunk during backward instead of storing
        it, which restores the bound during training at the cost of one
        extra forward pass over the pair MLP. Set it to False to trade
        memory back for speed when the model fits comfortably.

        chunk_q=0 disables chunking (single pass), which is marginally
        faster for small proteins.
        """
        B, H, V, D = vh.shape
        Q = qh.shape[2]
        chunk = self.chunk_q if self.chunk_q and self.chunk_q > 0 else Q

        if chunk >= Q:
            vq = torch.cat([
                vh.unsqueeze(3).expand(B, H, V, Q, D),
                qh.unsqueeze(2).expand(B, H, V, Q, D),
            ], dim=-1)
            return alpha.unsqueeze(-1) * self.pair_mlp(vq)

        def _one_chunk(vh_, qh_s, alpha_s):
            w = qh_s.shape[2]
            vq = torch.cat([
                vh_.unsqueeze(3).expand(B, H, V, w, D),
                qh_s.unsqueeze(2).expand(B, H, V, w, D),
            ], dim=-1)
            return alpha_s.unsqueeze(-1) * self.pair_mlp(vq)

        use_ckpt = (self.checkpoint_chunks and self.training
                    and torch.is_grad_enabled())
        out = []
        for s in range(0, Q, chunk):
            e = min(s + chunk, Q)
            args = (vh, qh[:, :, s:e], alpha[:, :, :, s:e])
            if use_ckpt:
                out.append(torch.utils.checkpoint.checkpoint(
                    _one_chunk, *args, use_reentrant=False))
            else:
                out.append(_one_chunk(*args))
        return torch.cat(out, dim=3)

    # ---- regularisation --------------------------------------------------
    def sparsity_penalty(self, alpha, eps=1e-12):
        """Concave penalty Omega(|S|) = sum sqrt(alpha).

        sqrt is concave, so mass is cheaper when concentrated: spreading
        0.5/0.5 costs 1.41, putting 1.0 on one pair costs 1.0. With
        adaptive_budget this lets the effective support size vary per
        complex instead of being pinned to a global k.
        """
        if not self.adaptive_budget:
            return alpha.new_zeros(())
        return torch.sqrt(alpha.clamp_min(eps)).sum(dim=(1, 2, 3)).mean()

    @staticmethod
    def support_size(alpha, thresh=0.0):
        """Mean number of pairs with non-zero selection mass, per sample."""
        return (alpha > thresh).float().sum(dim=(1, 2, 3))


class TopPairs:
    """Helper: pull the k most influential contacts out of a contribution map.

    Ranks by |contribution| rather than by alpha, because the thing that
    moves the prediction is alpha * g, not alpha alone. Ranking explanations
    by attention weight is precisely the habit this architecture is meant
    to retire.
    """

    @staticmethod
    def topk(contrib, k):
        """contrib: (B,H,V,Q) -> (values, (head, atom, residue)) index tuples."""
        B = contrib.shape[0]
        flat = contrib.reshape(B, -1)
        k = min(k, flat.shape[1])
        vals, idx = flat.abs().topk(k, dim=-1)
        H, V, Q = contrib.shape[1:]
        head = idx // (V * Q)
        rem = idx % (V * Q)
        atom = rem // Q
        res = rem % Q
        signed = torch.gather(flat, 1, idx)
        return signed, torch.stack([head, atom, res], dim=-1)
