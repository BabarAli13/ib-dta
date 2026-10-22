import numpy as np
rng=np.random.default_rng(0); B,V,Q,D=8,9,13,16
def entmax15(x,axis=-1,n=60):
    x=x-x.max(axis=axis,keepdims=True); x=x/2.0
    lo=x.max(axis=axis,keepdims=True)-1.0; hi=x.max(axis=axis,keepdims=True)-(1.0/x.shape[axis])**0.5
    for _ in range(n):
        m=(lo+hi)/2; p=np.clip(x-m,0,None)**2; s=p.sum(axis=axis,keepdims=True)
        lo=np.where(s<1.0,m,lo); hi=np.where(s>=1.0,m,hi)
    p=np.clip(x-(lo+hi)/2,0,None)**2
    return p/np.clip(p.sum(axis=axis,keepdims=True),1e-12,None)
Wv,Wq=rng.normal(size=(D,D)),rng.normal(size=(D,D))
v,q=rng.normal(size=(B,V,D)),rng.normal(size=(B,Q,D))
lg=np.einsum("bvd,bqd->bvq",v@Wv,q@Wq)/np.sqrt(D)
print(f"raw logit std = {lg.std():.2f}   (n_pairs={V*Q})")
print(f"{'temperature':>12} {'logit std':>10} {'mean support':>13} {'% of pairs':>11}")
for t in [0.25,0.5,1,2,4,8,16,32]:
    A=entmax15((lg/t).reshape(B,-1)).reshape(B,V,Q)
    s=(A>0).sum(axis=(1,2)).mean()
    print(f"{t:>12} {lg.std()/t:>10.2f} {s:>13.1f} {100*s/(V*Q):>10.1f}%")
