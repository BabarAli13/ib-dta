"""
Numerically verify every theoretical claim made in the paper.
Any proposition that fails here does not go in the paper.
"""
import sys
sys.path.insert(0, "/home/claude/IB-DTA")
import itertools, math
import numpy as np, torch
from ib_layer import InteractionBottleneck, entmax15

torch.manual_seed(0); np.random.seed(0)
OK = lambda b: "PASS" if b else "**FAIL**"
res = {}

# ---------------------------------------------------------------- Prop 1
# Exact deletion: |y - y_\T| = |sum_{T} c_ij|  for ANY subset T
print("=" * 70); print("PROP 1  Exact deletion for arbitrary subsets")
B,V,Q = 6,7,11
m = InteractionBottleneck(v_dim=24,q_dim=20,h_dim=16,heads=1,selection="entmax").eval()
v,q = torch.randn(B,V,24), torch.randn(B,Q,20)
with torch.no_grad(): y0, contrib, alpha = m(v,q)
worst = 0.0
for trial in range(40):
    drop = torch.rand(B,1,V,Q) < np.random.uniform(0.05,0.9)
    with torch.no_grad(): yd,_,_ = m(v,q,drop_pairs=drop)
    pred = (contrib.unsqueeze(1) if contrib.dim()==3 else contrib)
    pred = (pred*drop.float()).sum((1,2,3))
    worst = max(worst, float(((y0.view(-1)-yd.view(-1)) - pred).abs().max()))
print(f"  worst residual over 40 random subsets: {worst:.3e}   {OK(worst<1e-5)}")
res["prop1"] = worst

# ---------------------------------------------------------------- Prop 2
# Additivity over disjoint subsets: effect(A u B) = effect(A)+effect(B)
print("=" * 70); print("PROP 2  Effects are additive over disjoint subsets")
worst2 = 0.0
for trial in range(25):
    mask = torch.rand(B,1,V,Q)
    A = mask < 0.25
    Bs = (mask >= 0.25) & (mask < 0.5)
    with torch.no_grad():
        yA,_,_ = m(v,q,drop_pairs=A); yB,_,_ = m(v,q,drop_pairs=Bs)
        yAB,_,_ = m(v,q,drop_pairs=A|Bs)
    lhs = (y0.view(-1)-yAB.view(-1))
    rhs = (y0.view(-1)-yA.view(-1)) + (y0.view(-1)-yB.view(-1))
    worst2 = max(worst2, float((lhs-rhs).abs().max()))
print(f"  worst |effect(AuB) - effect(A) - effect(B)|: {worst2:.3e}   {OK(worst2<1e-5)}")
res["prop2"] = worst2

# ---------------------------------------------------------------- Prop 3
# Shapley equivalence: phi_ij = c_ij exactly, for the deletion game.
print("=" * 70); print("PROP 3  Shapley value of a contact equals its contribution")
# small instance so we can enumerate all permutations
Bs_,Vs,Qs = 1,2,3
ms = InteractionBottleneck(v_dim=8,q_dim=8,h_dim=6,heads=1,selection="softmax").eval()
vs,qs = torch.randn(Bs_,Vs,8), torch.randn(Bs_,Qs,8)
with torch.no_grad(): ys0, cs, _ = ms(vs,qs)
cs = cs.reshape(-1)            # Vs*Qs contributions
n = cs.numel()
def val(S):                    # characteristic function: value of coalition S
    drop = torch.ones(Bs_,1,Vs,Qs,dtype=torch.bool)
    flat = drop.reshape(-1)
    for i in S: flat[i] = False          # keep members of S
    with torch.no_grad(): yv,_,_ = ms(vs,qs,drop_pairs=drop)
    return float(yv.view(-1)[0])
shap = np.zeros(n)
players = list(range(n))
for perm in itertools.permutations(players):
    seen = []
    for p in perm:
        before = val(seen); seen.append(p); after = val(seen)
        shap[p] += (after-before)
shap /= math.factorial(n)
err = float(np.abs(shap - cs.numpy()).max())
print(f"  n={n} contacts, {math.factorial(n)} permutations enumerated")
print(f"  max |Shapley - contribution|: {err:.3e}   {OK(err<1e-5)}")
res["prop3"] = err

# ---------------------------------------------------------------- Prop 4
# Completeness: sum of all contributions + bias = prediction
print("=" * 70); print("PROP 4  Completeness (efficiency axiom)")
with torch.no_grad(): y1, c1, _ = m(v,q)
tot = c1.sum(dim=tuple(range(1,c1.dim()))) + m.bias
err4 = float((y1.view(-1)-tot.view(-1)).abs().max())
print(f"  max |y - (b + sum c)|: {err4:.3e}   {OK(err4<1e-5)}")
res["prop4"] = err4

# ---------------------------------------------------------------- Prop 5
# A pooled bilinear decoder violates Prop 1; measure the violation.
print("=" * 70); print("PROP 5  Pooled decoder: bypass is non-zero")
class Pooled(torch.nn.Module):
    def __init__(s,d=16):
        super().__init__(); s.vp=torch.nn.Linear(24,d); s.qp=torch.nn.Linear(20,d)
        s.h=torch.nn.Sequential(torch.nn.Linear(d,d),torch.nn.ReLU(),torch.nn.Linear(d,1))
    def forward(s,v,q,drop=None):
        vh,qh=s.vp(v),s.qp(q)
        a=torch.einsum("bvd,bqd->bvq",vh,qh)
        a=torch.softmax(a.reshape(v.shape[0],-1),-1).reshape(v.shape[0],v.shape[1],q.shape[1])
        if drop is not None: a=a*(~drop.squeeze(1)).float()
        f=torch.einsum("bvd,bvq,bqd->bd",vh,a,qh)
        return s.h(f), a
pm = Pooled().eval()
with torch.no_grad(): yp0, att = pm(v,q)
viol=[]
for trial in range(25):
    drop = torch.rand(B,1,V,Q) < 0.2
    with torch.no_grad(): ypd,_ = pm(v,q,drop)
    observed = (yp0.view(-1)-ypd.view(-1))
    attributed = (att.unsqueeze(1)*drop.float()).sum((1,2,3))
    viol.append(float((observed.abs()-attributed.abs()).abs().mean()))
print(f"  mean |observed - attributed| over 25 subsets: {np.mean(viol):.4f}")
print(f"  violation is non-trivial: {OK(np.mean(viol)>1e-3)}")
res["prop5"] = float(np.mean(viol))

# ---------------------------------------------------------------- Prop 6
# entmax KKT: sqrt(p_i) = x_i/2 - tau for all i in support
print("=" * 70); print("PROP 6  entmax KKT stationarity on the support")
x = torch.randn(64,200)*1.3
p = entmax15(x,dim=-1)
errs=[]
for b in range(x.shape[0]):
    s = p[b]>0
    if s.sum()<2: continue
    r = x[b][s]/2 - p[b][s].sqrt()
    errs.append(float(r.std()))
print(f"  mean std of tau across support: {np.mean(errs):.3e}   {OK(np.mean(errs)<1e-5)}")
print(f"  simplex: sum={float(p.sum(-1).mean()):.6f}, min={float(p.min()):.1e}")
res["prop6"] = float(np.mean(errs))

# ---------------------------------------------------------------- Prop 7
# Complexity: chunked decode is exact, memory scales with chunk
print("=" * 70); print("PROP 7  Chunked decoding is exact")
mc = InteractionBottleneck(v_dim=24,q_dim=20,h_dim=16,heads=2,selection="entmax",chunk_q=0).eval()
vv,qq = torch.randn(3,9,24), torch.randn(3,40,20)
with torch.no_grad(): yA,cA,aA = mc(vv,qq)
d=[]
for ck in (4,7,16,40,100):
    mc.chunk_q=ck
    with torch.no_grad(): yB,cB,_ = mc(vv,qq)
    d.append(float((yA-yB).abs().max()))
print(f"  max |y_dense - y_chunked| over chunk sizes {{4,7,16,40,100}}: {max(d):.3e}   {OK(max(d)==0.0)}")
res["prop7"] = max(d)

print("=" * 70)
print("SUMMARY")
for k,v in res.items(): print(f"  {k:8s} {v:.3e}")

# ---------------------------------------------------------------- Prop 8
# Marginal attribution: atom-level and residue-level attributions are
# EXACT marginals of the contact attribution, and removing an atom changes
# the output by exactly its marginal.
print("=" * 70); print("PROP 8  Marginal (atom / residue) attribution is exact")
m8 = InteractionBottleneck(v_dim=24, q_dim=20, h_dim=16, heads=1,
                           selection="entmax").eval()
v8, q8 = torch.randn(5, 8, 24), torch.randn(5, 12, 20)
with torch.no_grad():
    y8, c8, a8 = m8(v8, q8)
c8 = c8 if c8.dim() == 4 else c8.unsqueeze(1)

atom_attr = c8.sum(dim=(1, 3))        # B,V   marginal over residues
res_attr  = c8.sum(dim=(1, 2))        # B,Q   marginal over atoms

# completeness of the marginals
err_a = float((atom_attr.sum(1) - c8.sum((1, 2, 3))).abs().max())
err_r = float((res_attr.sum(1) - c8.sum((1, 2, 3))).abs().max())
print(f"  atom marginals sum to total:    {err_a:.3e}")
print(f"  residue marginals sum to total: {err_r:.3e}")

# counterfactual: delete ALL contacts of one atom -> output moves by its marginal
worst8 = 0.0
for trial in range(30):
    b = np.random.randint(0, 5); i = np.random.randint(0, 8)
    drop = torch.zeros(5, 1, 8, 12, dtype=torch.bool)
    drop[b, :, i, :] = True
    with torch.no_grad():
        yd8, _, _ = m8(v8, q8, drop_pairs=drop)
    observed = float((y8.view(-1)[b] - yd8.view(-1)[b]))
    predicted = float(atom_attr[b, i])
    worst8 = max(worst8, abs(observed - predicted))
print(f"  atom ablation: max |observed - marginal| over 30 trials: {worst8:.3e}")
print(f"  {OK(max(err_a, err_r, worst8) < 1e-5)}")
res["prop8"] = max(err_a, err_r, worst8)

# ---------------------------------------------------------------- Prop 9
# Counterfactual composition: the predicted effect of swapping a SET of
# atoms equals the sum of their marginals (no interaction correction).
print("=" * 70); print("PROP 9  Multi-atom counterfactuals compose additively")
worst9 = 0.0
for trial in range(25):
    b = np.random.randint(0, 5)
    k = np.random.randint(2, 5)
    atoms = np.random.choice(8, k, replace=False)
    drop = torch.zeros(5, 1, 8, 12, dtype=torch.bool)
    for i in atoms:
        drop[b, :, i, :] = True
    with torch.no_grad():
        yd9, _, _ = m8(v8, q8, drop_pairs=drop)
    observed = float(y8.view(-1)[b] - yd9.view(-1)[b])
    predicted = float(sum(atom_attr[b, i] for i in atoms))
    worst9 = max(worst9, abs(observed - predicted))
print(f"  max |effect(set) - sum of marginals| over 25 trials: {worst9:.3e}")
print(f"  {OK(worst9 < 1e-5)}")
res["prop9"] = worst9

print("=" * 70)
print("FULL SUMMARY")
for k, v in res.items():
    print(f"  {k:8s} {v:.3e}")
