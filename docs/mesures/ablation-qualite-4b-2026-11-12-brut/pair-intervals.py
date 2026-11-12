import sys, random, math, collections
D = sys.argv[1]
def load(p):
    r = {}
    for line in open(p):
        if line.startswith('#') or line.startswith('subject,'): continue
        f = line.rstrip('\n').split(',')
        r[(f[0], f[1], f[3])] = (int(f[5]), int(f[6]))
    return r
arms = ['control','qkv','o','gateup','down','attn','mlp','all']
X = {a: load(f'{D}/mmlu-{a}-FULL.csv') for a in arms}
X['f16'] = load(f'{D}/mmlu-4b-f16-FULL.csv')
ref = load(f'{D}/ref/mmlu-4b-embed-q4-swap-FULL.csv')
keys = sorted(X['control'])
assert all(set(X[a]) == set(keys) for a in X), 'key sets differ'
print('questions', len(keys))
same = sum(1 for k in keys if k in ref and ref[k][0] == X['control'][k][0])
print('control picks equal to embed-q4-swap ref:', same, 'of', sum(1 for k in keys if k in ref), '; ref correct', sum(v[1] for v in ref.values()), 'control correct', sum(X['control'][k][1] for k in keys))
n = len(keys)
c = {a: [X[a][k][1] for k in keys] for a in X}
for a in X: print(f'{a:8s} correct {sum(c[a]):5d} micro {100*sum(c[a])/n:.4f}')
# normal interval on paired differences (no fpc), as the caller did
def normal(v):
    m = sum(v)/n; s = math.sqrt(sum((x-m)**2 for x in v)/(n-1)/n)
    return 100*m, 100*(m-1.96*s), 100*(m+1.96*s)
lin = {
 'gain qkv': {'qkv':1,'control':-1}, 'gain o': {'o':1,'control':-1}, 'gain gateup': {'gateup':1,'control':-1},
 'gain down': {'down':1,'control':-1}, 'gain attn': {'attn':1,'control':-1}, 'gain mlp': {'mlp':1,'control':-1},
 'gain all': {'all':1,'control':-1}, 'f16 - all': {'f16':1,'all':-1},
 'excess all-(mlp+attn)': {'all':1,'mlp':-1,'attn':-1,'control':1},
 'excess all-(4 singles)': {'all':1,'qkv':-1,'o':-1,'gateup':-1,'down':-1,'control':3},
 'excess attn-(qkv+o)': {'attn':1,'qkv':-1,'o':-1,'control':1},
 'excess mlp-(gateup+down)': {'mlp':1,'gateup':-1,'down':-1,'control':1},
}
vec = {name: [sum(w*c[a][i] for a, w in co.items()) for i in range(n)] for name, co in lin.items()}
print('\nnormal interval on paired per-question differences, no fpc')
for name, v in vec.items():
    m, lo, hi = normal(v); print(f'  {name:26s} {m:+.4f} [{lo:+.2f}, {hi:+.2f}]')
# stratified bootstrap by subject, 10000 draws; census so stratified micro = pooled mean
subj = collections.defaultdict(list)
for i, k in enumerate(keys): subj[k[0]].append(i)
rng = random.Random(0xb0075eed)
B = 10000
gains = {a: [] for a in ['qkv','o','gateup','down','attn','mlp','all']}
lins = {name: [] for name in ['excess all-(mlp+attn)','excess all-(4 singles)','excess attn-(qkv+o)','excess mlp-(gateup+down)']}
import array
cols = {a: c[a] for a in arms}
for b in range(B):
    idx = []
    for s, ii in subj.items():
        m = len(ii); idx.extend(ii[int(rng.random()*m)] for _ in range(m))
    tot = {a: sum(cols[a][i] for i in idx) for a in arms}
    g = {a: 100*(tot[a]-tot['control'])/n for a in gains}
    for a in gains: gains[a].append(g[a])
    lins['excess all-(mlp+attn)'].append(g['all']-g['mlp']-g['attn'])
    lins['excess all-(4 singles)'].append(g['all']-g['qkv']-g['o']-g['gateup']-g['down'])
    lins['excess attn-(qkv+o)'].append(g['attn']-g['qkv']-g['o'])
    lins['excess mlp-(gateup+down)'].append(g['mlp']-g['gateup']-g['down'])
def pct(v):
    v = sorted(v); return v[int(0.025*len(v))], v[int(0.975*len(v))-1]
print('\nstratified bootstrap (10000 draws, python Random seed 0xb0075eed, not mmlupair stream), no fpc')
for a, v in gains.items(): lo, hi = pct(v); print(f'  gain {a:8s} [{lo:+.2f}, {hi:+.2f}]')
for name, v in lins.items(): lo, hi = pct(v); print(f'  {name:26s} [{lo:+.2f}, {hi:+.2f}]')
G = {a: 100*(sum(c[a])-sum(c['control']))/n for a in gains}
S = G['all']/(G['qkv']+G['o']+G['gateup']+G['down'])
Aa = G['attn']/(G['qkv']+G['o']); Am = G['mlp']/(G['gateup']+G['down'])
Sv = [gains['all'][i]/(gains['qkv'][i]+gains['o'][i]+gains['gateup'][i]+gains['down'][i]) for i in range(B)]
Amv = [gains['mlp'][i]/(gains['gateup'][i]+gains['down'][i]) for i in range(B)]
den_attn = [gains['qkv'][i]+gains['o'][i] for i in range(B)]
print(f'\nS {S:.4f} bootstrap [{pct(Sv)[0]:.3f}, {pct(Sv)[1]:.3f}]')
print(f'A_mlp {Am:.4f} bootstrap [{pct(Amv)[0]:.3f}, {pct(Amv)[1]:.3f}]')
print(f'A_attn {Aa:.4f}; denominator qkv+o {G["qkv"]+G["o"]:+.4f} bootstrap [{pct(den_attn)[0]:+.2f}, {pct(den_attn)[1]:+.2f}], share of draws with denominator <= 0: {sum(1 for d in den_attn if d<=0)/B:.4f}')
print(f'sum of four singles {G["qkv"]+G["o"]+G["gateup"]+G["down"]:+.4f}; mlp+attn {G["mlp"]+G["attn"]:+.4f}')
