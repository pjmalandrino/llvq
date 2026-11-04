# Verifier's own check (read-only): recompute the plan micro/SE and pairing claims of bigmodel.md §8.9-11.
import csv, math, sys
from collections import defaultdict
def load(p):
    with open(p) as f:
        lines=[l for l in f if not l.startswith('#')]
    return {(r['subject'],int(r['index'])):r for r in csv.DictReader(lines)}
def micro(rows, keys):
    by=defaultdict(list); pop={}
    for k in keys:
        by[k[0]].append(int(rows[k]['correct'])); pop[k[0]]=int(rows[k]['population'])
    n=sum(pop.values())
    acc=sum(pop[s]*sum(v)/len(v) for s,v in by.items())/n
    var=sum((pop[s]/n)**2*(sum(v)/len(v))*(1-sum(v)/len(v))/len(v) for s,v in by.items())
    return 100*acc,100*math.sqrt(var)
def paired(a,b,keys):
    by=defaultdict(list); pop={}
    for k in keys:
        by[k[0]].append(int(a[k]['correct'])-int(b[k]['correct'])); pop[k[0]]=int(a[k]['population'])
    n=sum(pop.values())
    d=sum(pop[s]*sum(v)/len(v) for s,v in by.items())/n
    var=0
    for s,v in by.items():
        mu=sum(v)/len(v); var+=(pop[s]/n)**2*sum((x-mu)**2 for x in v)/(len(v)-1)/len(v)
    disc=sum(1 for k in keys if a[k]['correct']!=b[k]['correct'])
    return 100*d,100*math.sqrt(var),disc
R=sys.argv[1]
plan4=load(R+'/docs/data/mmlu-dumps/mmlu-4b-f16.csv')
plan14=load(R+'/docs/data/mmlu-dumps/mmlu-14b-f16.csv')
awqplan=load(R+'/docs/data/mmlu-dumps/mmlu-14b-awq.csv')
llvqplan=load(R+'/docs/data/mmlu-dumps/mmlu-14b-llvq.csv')
sealed=load(R+'/docs/mesures/sealed-8b-14b-2026-09-23-brut/mmlu-14b/mmlu-14b-sealed-FULL.csv')
awqF=load(R+'/docs/mesures/census-14b-ref-2026-09-22-brut/mmlu-14b-awq-FULL.csv')
f16F=load(R+'/docs/mesures/census-14b-ref-2026-09-22-brut/mmlu-14b-f16-FULL.csv')
keys=list(plan4.keys())
print('plan size',len(keys),'same keys 4b/14b',set(keys)==set(plan14.keys()))
print('qhash 4b==14b plan', all(plan4[k]['qhash']==plan14[k]['qhash'] for k in keys))
for nm,d in [('sealedFULL',sealed),('awqFULL',awqF),('f16FULL',f16F),('awqPLAN',awqplan),('llvqPLAN',llvqplan)]:
    miss=[k for k in keys if k not in d]; bad=[k for k in keys if k in d and d[k]['qhash']!=plan4[k]['qhash']]
    print(nm,'missing',len(miss),'qhash mismatch',len(bad))
print('f16 plan vs f16 FULL same picks', sum(plan14[k]['pick']==f16F[k]['pick'] for k in keys))
print('awq plan vs awq FULL same picks', sum(awqplan[k]['pick']==awqF[k]['pick'] for k in keys))
for nm,d in [('f16 plan',plan14),('awq FULL->plan',awqF),('awq PLAN',awqplan),('sealed FULL->plan',sealed),('llvq PLAN (old)',llvqplan)]:
    print(nm,'micro %.2f SE %.2f'%micro(d,keys))
print('sealed-awq paired %.2f SE %.2f disc %d'%paired(sealed,awqF,keys))
print('f16-sealed paired %.2f SE %.2f disc %d'%paired(plan14,sealed,keys))
allk=list(sealed.keys())
print('full split sealed %.2f awq %.2f f16 %.2f'%(micro(sealed,allk)[0],micro(awqF,allk)[0],micro(f16F,allk)[0]))
g16=load(R+'/docs/data/mmlu-dumps/mmlu-4b-gguf-f16.csv')
print('4B gguf f16 %.2f ours f16 %.2f same picks %d'%(micro(g16,keys)[0],micro(plan4,keys)[0],sum(g16[k]['pick']==plan4[k]['pick'] for k in keys)))
print('4B gguf-f16 vs ours f16 paired %.2f SE %.2f disc %d'%paired(g16,plan4,keys))
