import json,numpy as np
exec(open('stats.py').read().split("rng=np.random")[0])
rng=np.random.default_rng(1); out=[]
ks=['layers.17.mlp.experts.experts','layers.17.mlp.experts.output_experts','layers.35.mlp.experts.experts','layers.5.mlp.experts.experts','layers.0.mlp.experts.experts','layers.0.mlp.experts.output_experts','layers.1.mlp.experts.experts','layers.1.mlp.experts.output_experts','layers.17.mlp.shared_experts.fc_1','layers.17.mlp.shared_experts.fc_2','layers.17.mlp.down_proj','layers.17.mlp.up_proj','layers.17.mlp.moe_gate','layers.17.mixer_proj','layers.17.mixer.in_proj','layers.1.mixer.in_proj','layers.7.mixer.in_proj','layers.1.mixer_proj','layers.4.mixer.c_q','layers.4.gate_proj']
for k in ks:
    W=load('model.'+k+'.weight'); W2=W.reshape(-1,W.shape[-1])
    if W2.shape[0]>4608: W2=W2[rng.choice(W2.shape[0],4608,replace=False)]
    W2.astype('<f4').tofile('f32/'+k+'.f32'); out.append((k,W2.shape[0],W2.shape[1]))
for n in (384,768,1536):
    rng.standard_normal((2304,n)).astype('<f4').tofile(f'f32/gauss{n}.f32'); out.append((f'gauss{n}',2304,n))
open('f32/list.txt','w').write('\n'.join(f'{k} {a} {b}' for k,a,b in out))
