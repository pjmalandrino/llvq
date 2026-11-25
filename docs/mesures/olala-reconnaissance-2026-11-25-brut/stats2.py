import json,numpy as np
exec(open('stats.py').read().split("rng=np.random")[0])
rng=np.random.default_rng(0)
for k in ['model.layers.0.mlp.experts.experts.weight','model.layers.0.mlp.experts.output_experts.weight','model.layers.0.mlp.shared_experts.fc_1.weight','model.layers.1.mlp.experts.experts.weight','model.layers.1.mlp.experts.output_experts.weight','model.layers.1.mixer.in_proj.weight','model.layers.7.mixer.in_proj.weight','model.layers.1.mixer_proj.weight','model.layers.17.mlp.experts.experts.weight']:
    W=load(k); W2=W.reshape(-1,W.shape[-1]); rn=np.linalg.norm(W2,axis=1); 
    q=np.quantile(rn,[0.01,0.5,0.99,0.999,1.0]); Wn=W2/np.maximum(rn[:,None],1e-30)
    e=(rn**2); top=np.sort(e)[::-1]; share=top[:max(1,len(top)//100)].sum()/e.sum()
    R=rot(Wn[:8192],rng); rnr=np.linalg.norm(R.reshape(R.shape[0],-1,24),axis=2).ravel()*np.sqrt(W.shape[-1]/24)
    print(f"{k[6:]:48s} rownorm q1/50/99/99.9/max {q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g}/{q[3]:.3g}/{q[4]:.3g}  top1%rows energy {share:.2f}  kurt rownormed {kurt(Wn[:8192]):.2f} rot {kurt(R):.2f}  int4 rownormed+rot {int4(R,128):.5f}  block-gain CV {rnr.std()/rnr.mean():.3f}")
A=load('model.layers.1.mlp.experts.experts.weight'); n=np.linalg.norm(A.reshape(256,-1),axis=1); print('L1 expert norms sorted', np.round(np.sort(n)[[0,25,64,128,192,230,255]],2))
