# One arm on one window: the routed experts of LAYERS replaced, everything else the bf16 model.
# env: ARM = base | t1 | t2 | q4 | noise ; R (noise) ; OFF ; SEED ; THREADS
import torch,json,sys,time,os,numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
ARM=os.environ['ARM']; R=float(os.environ.get('R','0')); OFF=int(os.environ.get('OFF','0')); TAG=os.environ.get('TAG',ARM)
LAYERS=[int(x) for x in os.environ.get('LAYERS','6,17,28').split(',')]
torch.set_num_threads(int(os.environ.get('THREADS','4'))); torch.set_grad_enabled(False); torch.manual_seed(int(os.environ.get('SEED','1234')))
NSEQ,SL=4,512
tok=AutoTokenizer.from_pretrained('model',trust_remote_code=True)
txt=open('dl/report.txt').read()+open('/home/user/llvq-tetra/docs/how-it-works.md').read()
ids=tok(txt,return_tensors='pt').input_ids[0][OFF:OFF+NSEQ*SL].view(NSEQ,SL)
m=AutoModelForCausalLM.from_pretrained('model',trust_remote_code=True,dtype=torch.bfloat16,low_cpu_mem_usage=True); m.eval()
def noise(W,r):
    sh=W.shape; X=W.data.reshape(-1,sh[-1]); c=1-r/2; s_=(1-c*c)**0.5
    n=torch.randn_like(X); n-=(n*X).sum(1,keepdim=True)/X.square().sum(1,keepdim=True).clamp_min(1e-30)*X
    n*=X.norm(dim=1,keepdim=True)/n.norm(dim=1,keepdim=True); W.data=(c*X+s_*n).reshape(sh)
def int4_g128(W):
    # embedquant::quantize_affine at bits = 4, group = 128: the f16 value, min and max of the group, scale and bias rounded to f16
    sh=W.shape; X=W.data.half().float().reshape(-1,128); mn=X.min(1,keepdim=True).values; mx=X.max(1,keepdim=True).values
    s=((mx-mn)/15).half().float(); graded=s>0; s=torch.where(graded,s,torch.ones_like(s)); b=mn.half().float()
    q=torch.where(graded,torch.floor((X-b)/s+0.5).clamp(0,15),torch.zeros_like(X)); W.data=(q*s+b).reshape(sh)
relerr={}
def pre(mod,args,kwargs=None):
    mod.float(); L=mod._idx
    if ARM=='base' or L not in LAYERS: return
    for g,name in ((0,'experts'),(1,'output_experts')):
        p=getattr(mod.mlp.experts,name).weight; w0=p.data.clone()
        if ARM in('t1','t2'):
            rec=np.fromfile(f"cal/L{L}.g{g}.w{ARM[1]}.f32",dtype='<f4'); assert rec.size==p.numel(),(rec.size,p.numel())
            p.data=torch.from_numpy(rec.copy()).reshape(p.shape)
        elif ARM=='q4': int4_g128(p)
        elif ARM=='noise': noise(p,R)
        relerr[f'L{L}.g{g}']=float((p.data-w0).square().sum()/w0.square().sum())
def post(mod,args,out):
    for p in mod.parameters(): p.data=torch.empty(0)
for i,blk in enumerate(m.model.layers):
    blk._idx=i; blk.register_forward_pre_hook(pre); blk.register_forward_hook(post)
m.model.embedding.float(); m.lm_head.float()
t=time.time(); lg=m(input_ids=ids,use_cache=False).logits.float(); lp=torch.log_softmax(lg,-1); del lg
if ARM=='base':
    np.save(f'cal/lp_base_o{OFF}.npy',lp.numpy()); print('base saved',OFF,round(time.time()-t),'s NLL',float(-lp[:,:-1].gather(-1,ids[:,1:,None]).mean()))
else:
    lb=torch.from_numpy(np.load(f'cal/lp_base_o{OFF}.npy',mmap_mode='r')[:])
    kl=(lb.exp()*(lb-lp)).sum(-1)                      # (4, 512), every position, whole vocabulary
    top1=(lb.argmax(-1)==lp.argmax(-1))
    dnll=(-lp[:,:-1].gather(-1,ids[:,1:,None])[...,0])-(-lb[:,:-1].gather(-1,ids[:,1:,None])[...,0])
    torch.save(dict(kl=kl,top1=top1,dnll=dnll,relerr=relerr),f'cal/res_{TAG}_o{OFF}.pt')
    print(f'{TAG} o{OFF}: KL {float(kl.mean()):.5f} top1 {float(top1.float().mean())*100:.2f}% dNLL {float(dnll.mean()):+.5f} relerr {relerr} {round(time.time()-t)} s',flush=True)
