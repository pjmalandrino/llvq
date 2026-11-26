# Added after the reads, from the saved per-position results: nothing here was in the plan.
# Clusters by sequence (16 sequences of 512 positions), pairs the top-1, and reads how KL scales with the error.
import torch,math
OFFS=(0,2048,4096,6144); ARMS=('t1','n1','t2','n2','q4','n4')
D={a:[torch.load(f'cal/res_{a}_o{o}.pt',weights_only=False) for o in OFFS] for a in ARMS}
kl={a:torch.cat([d['kl'] for d in D[a]]).double() for a in ARMS}          # 16 x 512
t1={a:torch.cat([d['top1'] for d in D[a]]) for a in ARMS}
re={a:sum(D[a][0]['relerr'].values())/6 for a in ARMS}
def byseq(a,b):
    d=(kl[a]-kl[b]).mean(1); m=float(d.mean()); s=float(d.std()/4)
    print(f'by sequence, KL({a}) - KL({b}): mean {m:+.6f} +- {s:.6f} over 16 sequences  [{m-2.131*s:+.6f}, {m+2.131*s:+.6f}] (t, 15 df)   negative in {int((d<0).sum())} of 16')
for a,b in (('t2','q4'),('t1','n1'),('t2','n2'),('q4','n4'),('n2','n4')): byseq(a,b)
def rseq(a,b):
    r=kl[a].mean(1)/kl[b].mean(1)
    print(f'by sequence, KL({a})/KL({b}): min {float(r.min()):.3f}  median {float(r.median()):.3f}  max {float(r.max()):.3f}')
for a,b in (('t1','n1'),('t2','n2'),('q4','n4')): rseq(a,b)
def top(a,b):
    x=t1[a].reshape(-1); y=t1[b].reshape(-1); n10=int((x&~y).sum()); n01=int((~x&y).sum())
    z=(n10-n01)/math.sqrt(max(n10+n01,1))
    print(f'top-1, {a} against {b}: {float(x.float().mean())*100:.2f} % and {float(y.float().mean())*100:.2f} %; agree with bf16 where the other does not: {n10} and {n01} of 8,192; z = {z:+.2f}')
for a,b in (('t2','q4'),('t1','n1'),('t2','n2'),('q4','n4')): top(a,b)
def expo(a,b):
    ka=float(kl[a].mean()); kb=float(kl[b].mean())
    print(f'KL({a})/KL({b}) = {ka/kb:.3f} for an error ratio of {re[a]/re[b]:.3f}: exponent {math.log(ka/kb)/math.log(re[a]/re[b]):.3f}')
for a,b in (('n1','n2'),('t1','t2'),('n4','n2'),('n1','n4'),('t1','q4')): expo(a,b)
print('mean rel error:', {a:round(re[a],5) for a in ARMS})
