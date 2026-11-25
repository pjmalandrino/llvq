import json,struct,subprocess,sys,os
D='dl/'; h=json.load(open(D+'header.json')); h.pop('__metadata__',None)
N=struct.unpack('<Q',open(D+'hdrlen.bin','rb').read())[0]; base=8+N
url='https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/resolve/main/model.safetensors'
want=[]
for L in (0,1,5,17,19,35):
    for s in ('mlp.experts.experts.weight','mlp.experts.output_experts.weight','mlp.shared_experts.fc_1.weight','mlp.shared_experts.fc_2.weight','mlp.down_proj.weight','mlp.up_proj.weight','mlp.moe_gate.weight','mlp.expert_bias','mixer_proj.weight'):
        want.append(f'model.layers.{L}.{s}')
for L in (0,1,7,17): want.append(f'model.layers.{L}.mixer.in_proj.weight')
for L in (4,19): want+= [f'model.layers.{L}.mixer.c_q.weight',f'model.layers.{L}.gate_proj.weight']
for k in h:
    if 'geodesic' in k: want.append(k)
idx={}
for k in want:
    v=h[k]; a,b=v['data_offsets']; fn='w/'+k+'.bin'; idx[k]=dict(shape=v['shape'],dtype=v['dtype'],file=fn)
    if os.path.exists(fn) and os.path.getsize(fn)==b-a: continue
    r=subprocess.run(['curl','-sSL','--retry','3','-m','900','-r',f'{base+a}-{base+b-1}',url,'-o',fn])
    print(k,b-a,os.path.getsize(fn),flush=True)
json.dump(idx,open('w/index.json','w'))
