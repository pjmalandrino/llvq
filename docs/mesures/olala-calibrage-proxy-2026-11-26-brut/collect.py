import torch,os
OFFS=(0,2048,4096,6144); ARMS=[a for a in ('t1','n1','t2','n2','q4','n4') if all(os.path.exists(f'cal/res_{a}_o{o}.pt') for o in OFFS)]
D={a:[torch.load(f'cal/res_{a}_o{o}.pt',weights_only=False) for o in OFFS] for a in ARMS}
def se(x): return float(x.std()/x.numel()**0.5)
print('arm   KL mean +- se (8,192 positions)   top-1     dNLL +- se          KL by window (offsets 0, 2048, 4096, 6144)        rel error of the six tensors')
for a in ARMS:
    kl=torch.cat([d['kl'].reshape(-1) for d in D[a]]); t1=torch.cat([d['top1'].reshape(-1) for d in D[a]]).float(); dn=torch.cat([d['dnll'].reshape(-1) for d in D[a]])
    re=D[a][0]['relerr']
    print(f"{a:4s}  {float(kl.mean()):.5f} +- {se(kl):.5f}           {float(t1.mean())*100:6.2f} %  {float(dn.mean()):+.5f} +- {se(dn):.5f}  "+'  '.join(f"{float(d['kl'].mean()):.5f}" for d in D[a])+'    '+' '.join(f'{v:.5f}' for v in re.values()))
def ratio(a,b):
    if a in D and b in D:
        ka=torch.cat([d['kl'].reshape(-1) for d in D[a]]); kb=torch.cat([d['kl'].reshape(-1) for d in D[b]])
        print(f'ratio KL({a})/KL({b}) = {float(ka.mean()/kb.mean()):.3f}   by window: '+'  '.join(f"{float(x['kl'].mean()/y['kl'].mean()):.3f}" for x,y in zip(D[a],D[b])))
def paired(a,b):
    if a in D and b in D:
        d=torch.cat([(x['kl']-y['kl']).reshape(-1) for x,y in zip(D[a],D[b])]); m=float(d.mean()); s=se(d)
        print(f'paired KL({a}) - KL({b}) = {m:+.6f} +- {s:.6f}  [{m-1.96*s:+.6f}, {m+1.96*s:+.6f}]   by window: '+'  '.join(f"{float((x['kl']-y['kl']).mean()):+.6f}" for x,y in zip(D[a],D[b])))
ratio('t1','n1'); ratio('t2','n2'); ratio('q4','n4'); paired('t2','q4'); paired('t1','n1'); paired('t2','n2'); paired('n2','n4')
