# The routed experts of layers 6, 17 and 28 as raw little-endian f32, from the local safetensors file.
import json,struct,numpy as np
h=json.load(open('dl/header.json')); h.pop('__metadata__',None)
N=struct.unpack('<Q',open('dl/hdrlen.bin','rb').read())[0]; base=8+N
f=open('model/model.safetensors','rb')
for L in (6,17,28):
    for g,s in ((0,'experts'),(1,'output_experts')):
        v=h[f'model.layers.{L}.mlp.experts.{s}.weight']; a,b=v['data_offsets']; assert v['dtype']=='BF16'
        f.seek(base+a); raw=np.frombuffer(f.read(b-a),dtype='<u2')
        w=(raw.astype(np.uint32)<<16).view(np.float32); w.tofile(f'cal/L{L}.g{g}.f32'); print(L,s,v['shape'],float(w.std()),flush=True)
