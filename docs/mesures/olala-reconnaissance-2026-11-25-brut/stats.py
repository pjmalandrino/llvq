import json,numpy as np,os
idx=json.load(open('w/index.json'))
def load(k):
    v=idx[k]; raw=np.fromfile(v['file'],dtype=np.uint16 if v['dtype']=='BF16' else np.float32)
    if v['dtype']=='BF16': raw=(raw.astype(np.uint32)<<16).view(np.float32)
    return raw.reshape(v['shape']).astype(np.float32)
def kurt(x):
    x=x.ravel(); x=x-x.mean(); s=x.std(); return float((x**4).mean()/s**4-3)
def fwht(a):  # along last axis, power of 2
    n=a.shape[-1]; h=1; a=a.copy()
    while h<n:
        a=a.reshape(*a.shape[:-1],n//(2*h),2,h); x=a[...,0,:]+a[...,1,:]; y=a[...,0,:]-a[...,1,:]
        a=np.stack([x,y],-2).reshape(*a.shape[:-3],n); h*=2
    return a/np.sqrt(n)
def rot(W,rng):  # random signs + block Hadamard on columns (blocks of the largest power of two dividing n)
    n=W.shape[-1]; b=n & -n; s=rng.choice([-1.,1.],n).astype(np.float32); W=W*s
    return fwht(W.reshape(*W.shape[:-1],n//b,b)).reshape(W.shape)
def int4(W,g):  # asymmetric min-max int4 per group of g along rows
    sh=W.shape; X=W.reshape(-1,g); lo=X.min(1,keepdims=True); hi=X.max(1,keepdims=True); sc=(hi-lo)/15; sc[sc==0]=1
    Q=np.round((X-lo)/sc).clip(0,15)*sc+lo; return float(((Q-X)**2).sum()/(X**2).sum())
rng=np.random.default_rng(0)
print(f"{'tensor':58s} {'shape':>16s} {'std':>8s} {'max/std':>7s} {'kurt':>6s} {'kurt_rot':>8s} {'rowCV':>6s} {'int4g128':>9s} {'rot+int4':>9s}")
for k in idx:
    if 'geodesic' in k or 'expert_bias' in k: continue
    W=load(k); W2=W.reshape(-1,W.shape[-1])
    sub=W2[:4096] if W2.shape[0]>4096 else W2
    R=rot(sub,rng); rn=np.linalg.norm(W2,axis=1)
    g=128 if W.shape[-1]%128==0 else 64
    print(f"{k[6:]:58s} {str(tuple(W.shape)):>16s} {W.std():8.5f} {np.abs(W).max()/W.std():7.1f} {kurt(sub):6.2f} {kurt(R):8.2f} {rn.std()/rn.mean():6.3f} {int4(sub,g):9.5f} {int4(R,g):9.5f}",flush=True)
print('gaussian ref int4 g128', int4(rng.standard_normal((2048,1536)).astype(np.float32),128))
# per-expert norms and usage bias
for L in (0,1,5,17,19,35):
    A=load(f'model.layers.{L}.mlp.experts.experts.weight'); B=load(f'model.layers.{L}.mlp.experts.output_experts.weight')
    na=np.linalg.norm(A.reshape(256,-1),axis=1); nb=np.linalg.norm(B.reshape(256,-1),axis=1); eb=load(f'model.layers.{L}.mlp.expert_bias')
    # expert similarity: cosine between flattened experts (sample)
    F=A.reshape(256,-1); F=F/np.linalg.norm(F,axis=1,keepdims=True); C=F@F.T; np.fill_diagonal(C,0)
    sv=np.linalg.svd(A[0],compute_uv=False); er=float((sv.sum()**2)/(sv**2).sum())
    print(f"L{L:2d} |W_in| min/med/max {na.min():.2f}/{np.median(na):.2f}/{na.max():.2f}  |W_out| {nb.min():.2f}/{np.median(nb):.2f}/{nb.max():.2f}  bias min/med/max {eb.min():+.4f}/{np.median(eb):+.4f}/{eb.max():+.4f}  maxcos {np.abs(C).max():.3f}  stable-ish rank e0 {er:.0f}/384")
gs=json.load(open('dl/header.json'))
