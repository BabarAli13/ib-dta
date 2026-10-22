"""
Unit tests for the interaction bottleneck and the faithfulness audit.

These run on synthetic tensors -- no dgl, no rdkit, no dataset. The point is
to verify the properties the paper's claims rest on:

  1. entmax produces EXACT zeros (not small values)
  2. the selection respects padding masks
  3. gradients flow through hard top-k via the straight-through estimator
  4. the additive deletion bound holds NUMERICALLY, not just on paper
  5. a pooled (DrugBAN-style) decoder VIOLATES that bound -- this is the
     control that makes test 4 meaningful
  6. the faithfulness audit separates a faithful from an unfaithful model

Run:  python -m pytest test_ib.py -v     (or just: python test_ib.py)
"""

import math

import torch
import torch.nn as nn

from ib_layer import InteractionBottleneck, entmax15, TopPairs
import faithfulness as F_


torch.manual_seed(0)
B, V, Q, VD, QD = 4, 9, 13, 32, 24


def _mk(selection="entmax", heads=1, **kw):
    return InteractionBottleneck(
        v_dim=VD, q_dim=QD, h_dim=16, heads=heads,
        selection=selection, out_dim=1, **kw)


def _inputs(pad_v=2, pad_q=3):
    v = torch.randn(B, V, VD)
    q = torch.randn(B, Q, QD)
    v_mask = torch.ones(B, V, dtype=torch.bool)
    q_mask = torch.ones(B, Q, dtype=torch.bool)
    if pad_v:
        v_mask[:, -pad_v:] = False
    if pad_q:
        q_mask[:, -pad_q:] = False
    return v, q, v_mask, q_mask


# --------------------------------------------------------------------------
def test_entmax_exact_zeros():
    x = torch.randn(5, 200) * 3
    p = entmax15(x, dim=-1)
    assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-4), "not normalised"
    n_zero = (p == 0).float().mean().item()
    assert n_zero > 0.3, f"entmax should zero most entries, got {n_zero:.2%}"
    # softmax, by contrast, zeros nothing
    s = torch.softmax(x, dim=-1)
    assert (s == 0).float().mean().item() == 0.0
    print(f"  entmax zeroed {n_zero:.1%} of entries; softmax zeroed 0.0%")


def test_sparsity_and_masking():
    for mode in ("entmax", "topk"):
        m = _mk(mode, budget_k=10)
        v, q, vm, qm = _inputs()
        y, contrib, alpha = m(v, q, vm, qm)
        assert y.shape == (B, 1)
        assert alpha.shape == (B, 1, V, Q)
        # padding never selected
        pad_mass = alpha[:, :, -2:, :].abs().sum() + alpha[:, :, :, -3:].abs().sum()
        assert pad_mass.item() == 0.0, f"{mode}: padding carries mass {pad_mass}"
        supp = InteractionBottleneck.support_size(alpha).mean().item()
        real = (V - 2) * (Q - 3)
        assert supp <= real, f"{mode}: support {supp} exceeds {real} real pairs"
        print(f"  {mode:8s} support {supp:5.1f} / {real} real pairs")


def test_topk_budget_exact():
    k = 7
    m = _mk("topk", budget_k=k)
    v, q, vm, qm = _inputs()
    _, _, alpha = m(v, q, vm, qm)
    supp = InteractionBottleneck.support_size(alpha)
    assert torch.all(supp <= k), f"budget exceeded: {supp}"
    print(f"  topk support sizes {supp.tolist()} (budget {k})")


def test_gradients_flow():
    for mode in ("entmax", "topk", "softmax"):
        m = _mk(mode, budget_k=8)
        v, q, vm, qm = _inputs()
        y, _, _ = m(v, q, vm, qm)
        y.sum().backward()
        gnorm = sum(p.grad.abs().sum().item()
                    for p in m.parameters() if p.grad is not None)
        assert math.isfinite(gnorm) and gnorm > 0, f"{mode}: no gradient ({gnorm})"
        # the projection layers specifically must receive gradient, otherwise
        # selection is detached and the encoders cannot learn
        assert m.v_proj.weight.grad.abs().sum().item() > 0, f"{mode}: v_proj detached"
        print(f"  {mode:8s} grad norm {gnorm:.4f}, v_proj reached")


def test_additive_deletion_is_exact():
    """THE key property. Deleting pairs must change the output by exactly
    the sum of their contributions."""
    m = _mk("entmax").eval()
    v, q, vm, qm = _inputs()

    def predict(drop):
        with torch.no_grad():
            y, _, _ = m(v, q, vm, qm, drop_pairs=drop)
        return y

    with torch.no_grad():
        _, contrib, alpha = m(v, q, vm, qm)

    # delete the 5 largest-magnitude contacts
    drop = torch.zeros(B, 1, V, Q, dtype=torch.bool)
    flat = contrib.reshape(B, -1).abs()
    idx = flat.topk(5, dim=-1).indices
    drop.reshape(B, -1).scatter_(1, idx, True)

    max_res, ok, detail = F_.check_additive_exactness(predict, contrib, drop)
    assert ok, f"additive bound violated: {detail}"
    assert detail["mean_observed_delta"] > 1e-6, "deletion had no effect; test is vacuous"
    print(f"  exact: max residual {max_res:.3e}, "
          f"mean |delta| {detail['mean_observed_delta']:.4f}")


def test_pooled_decoder_violates_the_bound():
    """Control for the test above.

    A DrugBAN-style pooled decoder is given the SAME attention map and the
    same deletion. If the bound held for it too, the architecture would be
    pointless. It does not hold -- and the size of the violation is the
    quantity the paper reports as 'prediction mass bypassing the map'.
    """
    class Pooled(nn.Module):
        def __init__(self):
            super().__init__()
            self.vp = nn.Linear(VD, 16)
            self.qp = nn.Linear(QD, 16)
            self.head = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, v, q, drop=None):
            vh, qh = self.vp(v), self.qp(q)
            att = torch.einsum("bvd,bqd->bvq", vh, qh)
            att = torch.softmax(att.reshape(B, -1), -1).reshape(B, V, Q)
            if drop is not None:
                att = att * (~drop.squeeze(1)).float()
            fused = torch.einsum("bvd,bvq,bqd->bd", vh, att, qh)   # the bypass
            return self.head(fused), att

    pm = Pooled().eval()
    v, q, _, _ = _inputs(0, 0)
    with torch.no_grad():
        y_full, att = pm(v, q)
    drop = torch.zeros(B, 1, V, Q, dtype=torch.bool)
    idx = att.reshape(B, -1).topk(5, -1).indices
    drop.reshape(B, -1).scatter_(1, idx, True)
    with torch.no_grad():
        y_drop, _ = pm(v, q, drop)

    observed = (y_full - y_drop).abs().mean().item()
    # "attributed" importance under the usual attention-weight reading
    attributed = (att.unsqueeze(1) * drop.float()).sum(dim=(1, 2, 3)).mean().item()
    gap = abs(observed - attributed)
    assert gap > 1e-3, "pooled model unexpectedly satisfied the bound"
    print(f"  pooled: observed delta {observed:.4f} vs attributed {attributed:.4f} "
          f"-> mismatch {gap:.4f} (bound does NOT hold)")


def test_faithfulness_audit_discriminates():
    """A model whose explanation is its mechanism should score AFS > 0.
    A random importance map should score about 0."""
    m = _mk("entmax").eval()
    v, q, vm, qm = _inputs()

    def predict(drop):
        with torch.no_grad():
            y, _, _ = m(v, q, vm, qm, drop_pairs=drop)
        return y

    with torch.no_grad():
        _, contrib, _ = m(v, q, vm, qm)

    good = F_.audit(predict, contrib, seed=0)
    noise = torch.rand_like(contrib)
    bad = F_.audit(predict, noise, seed=0)

    print(F_.format_report("  IB-DTA (true contributions)", good))
    print(F_.format_report("  IB-DTA (random importance)", bad))
    assert good["afs"] > bad["afs"], (
        f"true importance ({good['afs']:.4f}) did not beat random "
        f"({bad['afs']:.4f})")
    assert good["polarity"] > 0, "ranking does not track importance"


def test_toppairs_indices_are_consistent():
    m = _mk("entmax").eval()
    v, q, vm, qm = _inputs()
    with torch.no_grad():
        _, contrib, _ = m(v, q, vm, qm)
    vals, idx = TopPairs.topk(contrib, 3)
    assert vals.shape == (B, 3) and idx.shape == (B, 3, 3)
    for b in range(B):
        for r in range(3):
            h, a, res = idx[b, r].tolist()
            assert abs(contrib[b, h, a, res].item() - vals[b, r].item()) < 1e-5, \
                "index does not point at the reported value"
    print(f"  top-3 |contrib| per sample recovered correctly")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
            print("  PASS")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
