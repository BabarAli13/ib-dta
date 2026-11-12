"""
DrugBAN's pooled bilinear decoder, as the control for the additive one.

The paper's claim is about what the contraction in~\\eqref{eq:decode} costs
and what it buys. Quoting DrugBAN's published number does not test that:
their result comes from a different split protocol, a different training
budget and a different codebase, so any difference between it and ours is
confounded with all three. The only comparison that isolates the decoder
is one where everything else is literally the same objects.

So this module reimplements BANLayer and MLPDecoder from the DrugBAN
source vendored in drugban_upstream/, to be dropped into IBDTA in place of
InteractionBottleneck. The encoders, the data loader, the splits, the
optimiser, the schedule and the early-stopping rule are then not merely
matched but shared, and `IB.DECODER` is the one thing that differs.

Two deliberate departures from the upstream code, both of which make the
control stronger rather than weaker:

Masking. Upstream sums the attention map over every (atom, residue) pair
including padding, because its loader pads to a fixed length and the model
learns to ignore it. Our bottleneck masks padding out. Leaving that
difference in place would hand the additive decoder an advantage that has
nothing to do with the contraction, so the mask is applied here too.

Deletion. The pooled model has no additive decomposition, so there is no
`contrib` to audit. What it does have is the attention map, which is the
object a reader is invited to interpret when a paper prints a heatmap.
This module therefore returns the attention map in the `contrib` position
and implements `drop_pairs` by zeroing those entries before pooling. That
makes the deletion audit run unchanged against the pooled model, and what
it measures is exactly the question the paper asks: when you delete the
contacts the heatmap says matter, does the prediction move the way the
heatmap implies? For the additive decoder it must, by algebra. For this
one it is an empirical question with, we expect, an unflattering answer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """Upstream drugban_upstream/ban.py FCNet, ReLU and dropout only."""

    def __init__(self, dims, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers += [weight_norm(nn.Linear(dims[i], dims[i + 1]), dim=None),
                       nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class PooledBilinearDecoder(nn.Module):
    """BANLayer + MLPDecoder, masked, with deletion support.

    Signature matches InteractionBottleneck so IBDTA can switch between
    them without touching its forward pass.
    """

    def __init__(self, v_dim, q_dim, h_dim=256, heads=2, out_dim=1,
                 dropout=0.2, k=3, mlp_hidden=512):
        super().__init__()
        self.h_dim, self.heads, self.k = h_dim, heads, k
        self.v_net = FCNet([v_dim, h_dim * k], dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * k], dropout=dropout)
        self.p_net = nn.AvgPool1d(k, stride=k) if k > 1 else None
        # The h_out <= 32 branch upstream; heads is 1 or 2 here.
        self.h_mat = nn.Parameter(torch.empty(1, heads, 1, h_dim * k).normal_())
        self.h_bias = nn.Parameter(torch.empty(1, heads, 1, 1).normal_())
        self.bn = nn.BatchNorm1d(h_dim)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, mlp_hidden), nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden),
            nn.Linear(mlp_hidden, 128), nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, out_dim))

    def _pool(self, v_, q_, att):
        logits = torch.einsum("bvk,bvq,bqk->bk", v_, att, q_)
        if self.p_net is not None:
            logits = self.p_net(logits.unsqueeze(1)).squeeze(1) * self.k
        return logits

    def forward(self, v, q, v_mask=None, q_mask=None, drop_pairs=None):
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        att = torch.einsum("xhyk,bvk,bqk->bhvq", self.h_mat, v_, q_) \
            + self.h_bias

        if v_mask is not None or q_mask is not None:
            m = torch.ones(att.shape[0], 1, att.shape[2], att.shape[3],
                           dtype=torch.bool, device=att.device)
            if v_mask is not None:
                m = m & v_mask[:, None, :, None]
            if q_mask is not None:
                m = m & q_mask[:, None, None, :]
            att = att.masked_fill(~m, 0.0)

        # Deleting a contact means removing its term from the pooled sum,
        # which is the closest analogue the contraction admits.
        if drop_pairs is not None:
            d = drop_pairs if drop_pairs.dim() == 4 else drop_pairs.unsqueeze(1)
            att = att.masked_fill(d, 0.0)

        pooled = self._pool(v_, q_, att[:, 0])
        for h in range(1, self.heads):
            pooled = pooled + self._pool(v_, q_, att[:, h])
        y = self.mlp(self.bn(pooled))

        # The attention map stands in for the attribution, because it is
        # what a reader of a published heatmap is given to interpret.
        return y, att, att

    # ---- parity with InteractionBottleneck -------------------------------
    def sparsity_penalty(self, alpha):
        """No bottleneck, so nothing to penalise. Returns a real zero on
        the right device rather than a Python 0, so the training loop can
        add it unconditionally."""
        return alpha.sum() * 0.0

    @staticmethod
    def support_size(alpha):
        """Every pair participates in the contraction, so the support is
        the whole map. Reporting it this way rather than raising keeps the
        logged columns comparable: the pooled row shows 21,012 where the
        sparse rows show tens."""
        dims = tuple(range(1, alpha.dim()))
        return (alpha != 0).sum(dim=dims).float()
