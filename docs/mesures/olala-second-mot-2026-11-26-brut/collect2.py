import torch
def cmp(a,b):
    A=torch.load(f'res_{a}.pt',weights_only=False); B=torch.load(f'res_{b}.pt',weights_only=False)
    agree=(A['top1']==B['top1']).float().mean().item()
    la,ia=A['lp_top']; lb,ib=B['lp_top']
    m=(ia.unsqueeze(-1)==ib.unsqueeze(-2)); has=m.any(-1); idx=m.float().argmax(-1); lbm=torch.gather(lb,-1,idx)
    lbm=torch.where(has,lbm,lb[...,-1:].expand_as(lbm))
    kl=(la.exp()*(la-lbm)).sum(-1).mean().item(); d=(B['nlls']-A['nlls']); return agree,kl,d
for t in ('exp085','hot20','hot50','two'):
    rows=[]
    for o,base,name in ((0,'base',t+'_o0' if t!='exp085' else 'exp085'),(2048,'base_o2048',t+'_o2048')):
        a,k,d=cmp(base,name); rows.append((a,k,d)); print(f'{name:14s} top1 {a*100:5.1f}%  KL~ {k:.4f}  dNLL {d.mean():+.4f} ± {d.std()/d.numel()**0.5:.4f}')
    d=torch.cat([r[2].reshape(-1) for r in rows]); print(f'  POOLED {t}: n={d.numel()} dNLL {d.mean():+.4f} ± {d.std()/d.numel()**0.5:.4f}  x{float(torch.exp(d.mean())):.3f}  top1 {sum(r[0] for r in rows)/2*100:.1f}%  KL~ {sum(r[1] for r in rows)/2:.4f}')
