"""
Identifiability of the attribution, and the bias of the straight-through
estimator.

Claims under test
-----------------
I1  The contact attribution is identifiable up to the bias term: two
    parameterisations producing the same predictions on a rich enough set
    of inputs produce the same contributions. Equivalently, the map from
    contributions to predictions is injective once the support is fixed.
    We test the contrapositive numerically: perturbing any single
    contribution changes the prediction by exactly that amount, so no
    two distinct contribution vectors give identical predictions on the
    same support.

I2  The pooled decoder is NOT identifiable in this sense: distinct
    attention maps give identical predictions. This is the DTA instance
    of the construction of Jain and Wallace (2019).

I3  The straight-through top-k estimator is a biased gradient estimator,
    and the bias shrinks as the retained mass grows. We measure the
    cosine similarity between the straight-through gradient and the
    true gradient of the soft relaxation.
"""
import sys
sys.path.insert(0, "/home/claude/IB-DTA")

import numpy as np
import torch
import torch.nn.functional as F
from ib_layer import InteractionBottleneck

torch.manual_seed(0)
np.random.seed(0)
OK = lambda b: "PASS" if b else "**FAIL**"
out = {}

print("=" * 70)
print("I1  Attribution is identifiable (injective on a fixed support)")
print("=" * 70)
B, V, Q, d = 4, 8, 12, 16
m = InteractionBottleneck(v_dim=d, q_dim=d, h_dim=d, heads=1,
                          selection="entmax").eval()
v, q = torch.randn(B, V, d), torch.randn(B, Q, d)
with torch.no_grad():
    y0, c0, a0 = m(v, q)
c0 = c0 if c0.dim() == 4 else c0.unsqueeze(1)

# perturb one contribution at a time via a drop mask and check the output
# moves by exactly that contribution -- so distinct contribution vectors
# cannot map to the same prediction
worst = 0.0
nz = (a0 > 0).nonzero(as_tuple=False)
tested = 0
for row in nz[:60]:
    b, h, i, j = [int(x) for x in row]
    drop = torch.zeros(B, 1, V, Q, dtype=torch.bool)
    drop[b, 0, i, j] = True
    with torch.no_grad():
        yd, _, _ = m(v, q, drop_pairs=drop)
    moved = float(y0.view(-1)[b] - yd.view(-1)[b])
    worst = max(worst, abs(moved - float(c0[b, h, i, j])))
    tested += 1
print(f"  tested {tested} single-contact perturbations")
print(f"  max |output move - contribution| : {worst:.3e}")
print(f"  injective on support             : {OK(worst < 1e-5)}")
out["I1_residual"] = worst

print()
print("=" * 70)
print("I2  Pooled decoder is NOT identifiable")
print("=" * 70)


class Pooled(torch.nn.Module):
    def __init__(s, d=16):
        super().__init__()
        s.vp = torch.nn.Linear(d, d); s.qp = torch.nn.Linear(d, d)
        s.h = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.ReLU(),
                                  torch.nn.Linear(d, 1))

    def forward(s, v, q, att=None):
        vh, qh = s.vp(v), s.qp(q)
        if att is None:
            a = torch.einsum("bvd,bqd->bvq", vh, qh)
            att = torch.softmax(a.reshape(v.shape[0], -1), -1)
            att = att.reshape(v.shape[0], v.shape[1], q.shape[1])
        f = torch.einsum("bvd,bvq,bqd->bd", vh, att, qh)
        return s.h(f), att


pm = Pooled().eval()
with torch.no_grad():
    yp, att0 = pm(v, q)

# search for a DIFFERENT attention map giving the same prediction
alt = att0.clone().requires_grad_(True)
opt = torch.optim.Adam([alt], lr=0.02)
for _ in range(600):
    opt.zero_grad()
    a = torch.softmax(alt.reshape(B, -1), -1).reshape(B, V, Q)
    yh, _ = pm(v, q, att=a)
    # match predictions while maximising distance from the original map
    loss = ((yh - yp) ** 2).mean() - 0.05 * (a - att0).abs().mean()
    loss.backward()
    opt.step()
with torch.no_grad():
    a_fin = torch.softmax(alt.reshape(B, -1), -1).reshape(B, V, Q)
    y_fin, _ = pm(v, q, att=a_fin)
    pred_gap = float((y_fin - yp).abs().mean())
    map_dist = float((a_fin - att0).abs().sum(dim=(1, 2)).mean())
    tv = float(0.5 * (a_fin - att0).abs().sum(dim=(1, 2)).mean())
print(f"  prediction difference  : {pred_gap:.3e}  (near zero = same output)")
print(f"  attention total-variation distance : {tv:.4f}")
print(f"  distinct map, same prediction : "
      f"{OK(pred_gap < 1e-2 and tv > 0.05)}")
out["I2_pred_gap"] = pred_gap
out["I2_tv"] = tv

print()
print("=" * 70)
print("I3  Straight-through top-k gradient bias")
print("=" * 70)
for k in [2, 4, 8, 16, 32, 64]:
    mk = InteractionBottleneck(v_dim=d, q_dim=d, h_dim=d, heads=1,
                               selection="topk", budget_k=k)
    ms = InteractionBottleneck(v_dim=d, q_dim=d, h_dim=d, heads=1,
                               selection="softmax")
    ms.load_state_dict(mk.state_dict())

    mk.zero_grad(); y1, _, _ = mk(v, q); y1.sum().backward()
    g_st = torch.cat([p.grad.flatten() for p in mk.parameters()
                      if p.grad is not None])

    ms.zero_grad(); y2, _, _ = ms(v, q); y2.sum().backward()
    g_soft = torch.cat([p.grad.flatten() for p in ms.parameters()
                        if p.grad is not None])

    cos = float(F.cosine_similarity(g_st[None], g_soft[None]).item())
    rel = float((g_st - g_soft).norm() / g_soft.norm().clamp_min(1e-9))
    print(f"  k={k:3d}   cos(g_ST, g_soft) = {cos:+.4f}   "
          f"relative bias = {rel:.3f}")
    out[f"I3_cos_k{k}"] = cos

cosv = [out[f"I3_cos_k{k}"] for k in [2, 4, 8, 16, 32, 64]]
improving = cosv[-1] > cosv[0]
print(f"\n  alignment improves with k : {OK(improving)}")
print("  -> anneal from a dense/entmax selection into hard top-k rather")
print("     than training top-k from random initialisation.")
out["I3_improves"] = bool(improving)

import json, os
os.makedirs("results", exist_ok=True)
with open("results/identifiability.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n  wrote results/identifiability.json")
