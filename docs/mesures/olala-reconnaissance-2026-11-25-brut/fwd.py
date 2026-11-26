import torch,json,sys,time,resource,numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(4); torch.set_grad_enabled(False)
NSEQ,SL=int(sys.argv[1]),int(sys.argv[2]); TAG=sys.argv[3]; RE,RD,HEAD,OFF=float(sys.argv[4]),float(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]) if len(sys.argv)>7 else 0
import os,re
torch.manual_seed(int(os.environ.get('SEED','1234'))); DRE=re.compile(os.environ.get('DENSE_RE','.'))
tok=AutoTokenizer.from_pretrained('model',trust_remote_code=True)
txt=open('dl/report.txt').read()+open('/home/user/llvq-tetra/docs/how-it-works.md').read()
ids=tok(txt,return_tensors='pt').input_ids[0]
print('tokens available',len(ids),flush=True)
ids=ids[OFF:OFF+NSEQ*SL].view(NSEQ,SL)
t=time.time()
m=AutoModelForCausalLM.from_pretrained('model',trust_remote_code=True,dtype=torch.bfloat16,low_cpu_mem_usage=True)
m.eval(); print('loaded',time.time()-t,'rss GB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,flush=True)
st={}
def noise(W,r):
    # angular error of relative MSE r per row, norm kept: the shape of a retracted lattice code's error
    if r<=0: return
    sh=W.shape; X=W.data.reshape(-1,sh[-1]); c=1-r/2; s_=(1-c*c)**0.5
    n=torch.randn_like(X); n-= (n*X).sum(1,keepdim=True)/X.square().sum(1,keepdim=True).clamp_min(1e-30)*X
    n*= X.norm(dim=1,keepdim=True)/n.norm(dim=1,keepdim=True)
    W.data=(c*X+s_*n).reshape(sh)
def pre(mod,args,kwargs=None):
    mod.float()
    for nme,p in mod.named_parameters():
        if p.dim()<2 or p.numel()<3e5: continue
        isexp='experts.experts' in nme or 'output_experts' in nme
        if isexp: noise(p,RE)
        elif DRE.search(nme): noise(p,RD)
def mk(i,blk):
    def gate_hook(mod,inp,out):
        x=inp[0].reshape(-1,inp[0].shape[-1]).float()
        sc=torch.sigmoid(x@mod.weight.float().t())+blk.mlp.expert_bias.float()
        top=sc.topk(6,dim=1); cnt=torch.bincount(top.indices.reshape(-1),minlength=256)
        srt=sc.sort(dim=1,descending=True).values; st.setdefault(i,{}).update(dict(cnt=cnt.tolist(),margin=(srt[:,5]-srt[:,6]).quantile(torch.tensor([0.01,0.1,0.5])).tolist(),xmax=float(x.abs().max())))
    def up_hook(mod,inp,out): st[i]['expout_max']=float(inp[0].abs().max()); st[i]['up_out_max']=float(out.abs().max())
    def mp_hook(mod,inp,out): st.setdefault(i,{})['mixer_out_max']=float(inp[0].abs().max())
    def fc2_hook(mod,inp,out): st[i]['shared_act_max']=float(inp[0].abs().max())
    def post(mod,args,out):
        for p in mod.parameters(): p.data=torch.empty(0)
        print('layer',i,round(time.time()-t),'s rss',round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,1),{k:(v if k!='cnt' else None) for k,v in st[i].items()},flush=True)
    blk.register_forward_pre_hook(pre); blk.mlp.moe_gate.register_forward_hook(gate_hook); blk.mlp.up_proj.register_forward_hook(up_hook)
    blk.mixer_proj.register_forward_hook(mp_hook); blk.mlp.shared_experts.fc_2.register_forward_hook(fc2_hook); blk.register_forward_hook(post)
for i,blk in enumerate(m.model.layers): mk(i,blk)
m.model.embedding.float(); m.lm_head.float()
if HEAD:
    out=m(input_ids=ids,use_cache=False); lg=out.logits.float()
    lp=torch.log_softmax(lg[:,:-1],-1); tgt=ids[:,1:]
    nlls=-lp.gather(-1,tgt[...,None])[...,0]; nll=nlls.mean(); print('NLL',float(nll),'ppl',float(nll.exp()))
    top1=lp.argmax(-1); torch.save(dict(nlls=nlls,top1=top1,lp_top=lp.topk(64,-1)),f'res_{TAG}.pt')
else:
    m.model(input_ids=ids,use_cache=False); nll=torch.tensor(float('nan'))
json.dump(dict(nseq=NSEQ,sl=SL,nll=float(nll),st=st),open(f'fwd_{TAG}.json','w'))
