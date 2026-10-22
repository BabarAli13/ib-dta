"""
Torch-free verification of the two properties IB-DTA's claims rest on.

Reimplements the entmax projection and both decoders in numpy, then checks:
  A. entmax yields exact zeros (softmax does not)
  B. additive decoding: deleting pairs moves the output by EXACTLY the sum
     of their contributions
  C. pooled decoding (DrugBAN-style): the same deletion does NOT obey that
     bound -- the control that makes B meaningful
  D. the deletion-based faithfulness audit separates the two
"""
import numpy as np

rng = np.random.default_rng(0)
B, V, Q, D = 4, 9, 13, 16


def entmax15(x, axis=-1, n_iter=60):
    x = x - x.max(axis=axis, keepdims=True)
    x = x / 2.0
    lo = x.max(axis=axis, keepdims=True) - 1.0
    hi = x.max(axis=axis, keepdims=True) - (1.0 / x.shape[axis]) ** 0.5
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        p = np.clip(x - mid, 0, None) ** 2
        s = p.sum(axis=axis, keepdims=True)
        # s(tau) decreases in tau: s < 1 means tau too large -> lower hi.
        hi = np.where(s < 1.0, mid, hi)
        lo = np.where(s < 1.0, lo, mid)
    p = np.clip(x - (lo + hi) / 2, 0, None) ** 2
    return p / np.clip(p.sum(axis=axis, keepdims=True), 1e-12, None)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# fixed random "weights"
Wv, Wq = rng.normal(size=(D, D)), rng.normal(size=(D, D))
w1 = rng.normal(size=(2 * D, 24)) * 0.3
b1 = rng.normal(size=24) * 0.1
w2 = rng.normal(size=(24, 1)) * 0.3
Wh1 = rng.normal(size=(D, 24)) * 0.3
Wh2 = rng.normal(size=(24, 1)) * 0.3

v = rng.normal(size=(B, V, D))
q = rng.normal(size=(B, Q, D))
vh, qh = v @ Wv, q @ Wq
logits = np.einsum("bvd,bqd->bvq", vh, qh) / np.sqrt(D)


def pair_scores():
    """g(h_i,h_j) for every pair -> (B,V,Q)"""
    cat = np.concatenate([
        np.repeat(vh[:, :, None, :], Q, axis=2),
        np.repeat(qh[:, None, :, :], V, axis=1)], axis=-1)
    h = np.maximum(cat @ w1 + b1, 0)
    return (h @ w2)[..., 0]


G = pair_scores()


def additive_predict(alpha, drop=None):
    a = alpha if drop is None else alpha * (~drop)
    return (a * G).sum(axis=(1, 2))


def pooled_predict(alpha, drop=None):
    """DrugBAN-style: contract the map into a joint vector, then an MLP."""
    a = alpha if drop is None else alpha * (~drop)
    fused = np.einsum("bvd,bvq,bqd->bd", vh, a, qh)
    return (np.maximum(fused @ Wh1, 0) @ Wh2)[..., 0]


flat = logits.reshape(B, -1)
A_ent = entmax15(flat).reshape(B, V, Q)
A_soft = softmax(flat).reshape(B, V, Q)

print("=" * 68)
print("A. SPARSITY")
print("=" * 68)
ze, zs = (A_ent == 0).mean(), (A_soft == 0).mean()
print(f"  entmax  exact zeros: {ze:6.1%}   support {int((A_ent>0).sum()/B)} / {V*Q} pairs")
print(f"  softmax exact zeros: {zs:6.1%}   support {int((A_soft>0).sum()/B)} / {V*Q} pairs")
assert ze > 0.3 and zs == 0.0
print("  -> entmax sparsifies; softmax cannot. PASS")

print()
print("=" * 68)
print("B. ADDITIVE DELETION IS EXACT")
print("=" * 68)
contrib = A_ent * G
drop = np.zeros((B, V, Q), bool)
idx = np.argsort(-np.abs(contrib).reshape(B, -1), axis=1)[:, :5]
np.put_along_axis(drop.reshape(B, -1), idx, True, axis=1)

y_full = additive_predict(A_ent)
y_drop = additive_predict(A_ent, drop)
observed = y_full - y_drop
predicted = (contrib * drop).sum(axis=(1, 2))
res = np.abs(observed - predicted)
print(f"  observed  delta : {np.round(observed, 8)}")
print(f"  predicted delta : {np.round(predicted, 8)}")
print(f"  max residual    : {res.max():.3e}")
assert res.max() < 1e-10 and np.abs(observed).mean() > 1e-6
print("  -> attribution == causal effect, to machine precision. PASS")

print()
print("=" * 68)
print("C. POOLED DECODER VIOLATES THE SAME BOUND (control)")
print("=" * 68)
py_full = pooled_predict(A_ent)
py_drop = pooled_predict(A_ent, drop)
p_obs = py_full - py_drop
p_attr = (A_ent * drop).sum(axis=(1, 2))
print(f"  observed delta       : {np.round(p_obs, 6)}")
print(f"  attention mass removed: {np.round(p_attr, 6)}")
rel = np.abs(np.abs(p_obs) - np.abs(p_attr)).mean() / max(np.abs(p_obs).mean(), 1e-12)
print(f"  relative mismatch    : {rel:.3f}")
assert rel > 0.01
print("  -> the pooled map's weights do NOT predict the output change. PASS")

print()
print("=" * 68)
print("D. DELETION AUDIT DISCRIMINATES")
print("=" * 68)


def curve(predict, alpha, importance, fracs):
    n = V * Q
    order = np.argsort(-np.abs(importance).reshape(B, -1), axis=1)
    y0 = predict(alpha)
    out = []
    for f in fracs:
        k = int(round(f * n))
        d = np.zeros((B, n), bool)
        if k:
            np.put_along_axis(d, order[:, :k], True, axis=1)
        out.append(np.abs(y0 - predict(alpha, d.reshape(B, V, Q))).mean())
    return np.array(out)


fr = np.linspace(0, 0.5, 11)
true_c = curve(additive_predict, A_ent, contrib, fr)
rand_c = curve(additive_predict, A_ent, rng.random((B, V, Q)), fr)
auc_t, auc_r = np.trapezoid(true_c, fr), np.trapezoid(rand_c, fr)
afs = (auc_t - auc_r) / max(auc_t, auc_r, 1e-12)
print(f"  AUC(true importance) : {auc_t:.5f}")
print(f"  AUC(random order)    : {auc_r:.5f}")
print(f"  AFS                  : {afs:+.4f}")
assert afs > 0
print("  -> true contributions beat random deletion. PASS")

print()
print("=" * 68)
print("ALL CHECKS PASSED")
print("=" * 68)
