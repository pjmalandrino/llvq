import torch,json,numpy as np,os
def cmp(a,b):
    A=torch.load(f'res_{a}.pt',weights_only=False); B=torch.load(f'res_{b}.pt',weights_only=False)
    agree=(A['top1']==B['top1']).float().mean().item()
    la,ia=A['lp_top']; lb,ib=B['lp_top']
    m=(ia.unsqueeze(-1)==ib.unsqueeze(-2)); has=m.any(-1); idx=m.float().argmax(-1); lbm=torch.gather(lb,-1,idx)
    lbm=torch.where(has,lbm,lb[...,-1:].expand_as(lbm))
    kl=(la.exp()*(la-lbm)).sum(-1).mean().item(); mass=la.exp().sum(-1).mean().item()
    d=(B['nlls']-A['nlls']); return agree,kl,mass,d.mean().item(),d.std().item()/d.numel()**0.5,d.numel(),d
pairs=[('base',t) for t in ('exp085','exp085_s77','exp085_dense010','Cnew_o0','exp010','exp040','all085','d_inproj','d_mixerproj','d_shared','d_attn','d_latent','d_router','d_router010')]
pairs+=[(f'base_o{o}',f'{t}_o{o}') for o in (2048,4096,6144) for t in ('exp085','C')]+[('base_o2048',t) for t in ('Cbig_o2048','router010_o2048','latent010_o2048','C_s77_o2048')]
pool={}
for a,b in pairs:
    if not os.path.exists(f'res_{b}.pt'): print(b,'missing'); continue
    r=cmp(a,b); print(f'{b:18s} top1 agree {r[0]*100:5.1f}%  KL~{r[1]:.4f} (top-64 mass {r[2]:.3f})  dNLL {r[3]:+.4f} ± {r[4]:.4f}  n={r[5]}')
    key='exp085' if b.startswith('exp085') and 's77' not in b and 'dense' not in b else ('C' if b[:2] in('C_','Cn') and 's77' not in b else None)
    if key: pool.setdefault(key,[]).append((r[6],r[0],r[1]))
for k,v in pool.items():
    d=torch.cat([x[0].reshape(-1) for x in v]); print(f'POOLED {k}: windows {len(v)} n={d.numel()} dNLL {d.mean():+.4f} ± {d.std()/d.numel()**0.5:.4f}  ppl x{float(torch.exp(d.mean())):.3f}  top1 {np.mean([x[1] for x in v])*100:.1f}%  KL~{np.mean([x[2] for x in v]):.4f}')
N=24*512
for tag in ('route2','route3'):
    d=json.load(open(f'fwd_{tag}.json'))['st']; allc=[np.array(d[str(i)]['cnt']) for i in range(36)]; A=np.array(allc)
    print(tag,'mean tokens/expert',A.mean(),'zero',int((A==0).sum()),'/',A.size,' under 72 (quarter of mean):',int((A<72).sum()),' over 1152 (4x mean):',int((A>1152).sum()))
    print(' per-layer zeros',[int((c==0).sum()) for c in allc]); print(' per-layer max share of tokens for one expert',[round(float(c.max()/N),2) for c in allc]); print(' per-layer experts under 72 tokens',[int((c<72).sum()) for c in allc])
    print(' margin p1/p10/p50 at layers 0,1,17,35',[['%.1e'%x for x in d[str(i)]['margin']] for i in (0,1,17,35)])
a=np.array([np.array(json.load(open('fwd_route2.json'))['st'][str(i)]['cnt']) for i in range(36)]); b=np.array([np.array(json.load(open('fwd_route3.json'))['st'][str(i)]['cnt']) for i in range(36)])
s=a+b; print('pooled 24,576 tokens: zero',int((s==0).sum()),' corr of counts between halves',round(float(np.corrcoef(a.ravel(),b.ravel())[0,1]),3))
for thr in (384,768,1536): print(f' share of (layer, expert) cells that would see under {thr} tokens at 131,072 tokens, scaling counts x5.33: {float((s*5.333<thr).mean()):.4f}; layers 2-35 only: {float((s[2:]*5.333<thr).mean()):.4f}')
print(' same, layers 0 and 1 under 768:',[round(float((s[i]*5.333<768).mean()),3) for i in (0,1)])
