# Verifier's arithmetic check of wave5-deep/brick.md numbers (bounds, swizzle enumeration,
# synthetic int16/int24/int8/f16 relative errors). Pure Python, no file or network I/O, no repo access.
# Not draft repository code: run once by the verifier as a tiny `python3 -I` arithmetic script
# (2026-11-02), the only execution. Random seed 1, 6 vectors per case, as the architect's sample size.
import math, random, struct

print("== bounds ==")
s = math.sqrt(24*416); P = s*32767
print("sum|y| <= %.4f, |P| < 2^%.3f" % (s, math.log2(P)))
print("after shift<=3: 2^%.3f" % math.log2(P*8))
span = (1.375/0.625)*math.sqrt(26); print("S span ratio %.3f = %.3f binades" % (span, math.log2(span)))
Smax = 2**24 * span; print("S_int max ~ 2^%.3f (<2^28?)" % math.log2(Smax))
for nb in (1066, 1067):
    print("|I| < 2^%.3f at %d blocks (S_int<2^28)" % (math.log2(P*8*2**28*nb), nb))
# int24 three planes
P24 = s*(2**23-1); print("int24 |P| < 2^%.3f ; P*S < 2^%.3f" % (math.log2(P24), math.log2(P24*2**28)))

print("== swizzle enumeration (16-byte chunks, 8 per 128B wavefront) ==")
def chunk(l, h, swz=True): return 2*l + ((h ^ ((l >> 2) & 1)) if swz else h)
def wavefronts(addrs):   # addrs: list of 16B chunk indices in one quarter-warp phase
    banks = {}
    for a in addrs: banks.setdefault(a % 8, set()).add(a)
    return max(len(v) for v in banks.values())
for swz in (False, True):
    # decode stores: 2 STS.128 per lane, quarter-warp phases of 8 lanes
    st = sum(wavefronts([chunk(l, h, swz) for l in range(q*8, q*8+8)]) for h in (0,1) for q in range(4))
    # ldmatrix.x4 per block jb: matrix i uses lanes 8i..8i+7 row addresses
    ld = 0
    for jb in (0,1):
        for i in range(4):
            rows = [ (i & 1)*8 + r for r in range(8)]; h = i >> 1
            ld += wavefronts([chunk(r + 16*jb, h, swz) for r in rows])
    print("swizzle=%s: decode-store wavefronts/round=%d, ldmatrix wavefronts/round=%d" % (swz, st, ld))

print("== synthetic activations ==")
def f16r(x): return struct.unpack('e', struct.pack('e', x))[0]
def exp2_of(a): return math.frexp(a)[1]-1
def q_slices(x, bits, slice_n=384, cap=3):
    ns = (len(x)+slice_n-1)//slice_n; e=[None]*ns
    for s_ in range(ns):
        seg = x[s_*slice_n:(s_+1)*slice_n]; am = max(abs(v) for v in seg)
        if am > 0: e[s_] = exp2_of(am) - (bits-2)
    ehi = max(v for v in e if v is not None); elo = ehi - cap; qmax = 2**(bits-1)-1
    out=[]; spread = ehi - min(v for v in e if v is not None)
    for s_ in range(ns):
        es = max(e[s_], elo) if e[s_] is not None else elo
        for v in x[s_*slice_n:(s_+1)*slice_n]:
            q = round(v*2.0**-es)          # Python round = half-even
            q = max(-qmax, min(qmax, q)); out.append(q*2.0**es)
    return out, spread
def q_vec(x, bits):
    return q_slices(x, bits, slice_n=len(x), cap=0)[0]
def q_int8_amax127(x):
    am = max(abs(v) for v in x); st = am/127
    return [max(-127, min(127, round(v/st)))*st for v in x]
def rel(x, y): return math.sqrt(sum((a-b)**2 for a,b in zip(x,y))/sum(a*a for a in x))
random.seed(1)
def student_t(nu): 
    z = random.gauss(0,1); c = sum(random.gauss(0,1)**2 for _ in range(nu)); return z/math.sqrt(c/nu)
for name, gen in (("gauss", lambda: random.gauss(0,1)), ("t4", lambda: student_t(4))):
    for n in (2544, 9720):
        res = {k: [] for k in ("f16","i16v","i16s","i16s_nocap","i8v","i24s","amax_rms_vec","amax_rms_slice_max","spread")}
        for _ in range(6):
            x = [gen() for _ in range(n)]
            rms = math.sqrt(sum(v*v for v in x)/n)
            res["f16"].append(rel(x,[f16r(v) for v in x]))
            res["i16v"].append(rel(x,q_vec(x,16)))
            qs, sp = q_slices(x,16); res["i16s"].append(rel(x,qs)); res["spread"].append(sp)
            res["i16s_nocap"].append(rel(x,q_slices(x,16,cap=99)[0]))
            res["i8v"].append(rel(x,q_int8_amax127(x)))
            res["i24s"].append(rel(x,q_slices(x,24)[0]))
            res["amax_rms_vec"].append(max(abs(v) for v in x)/rms)
            sl = [x[i:i+384] for i in range(0,n,384)]
            res["amax_rms_slice_max"].append(max(max(abs(v) for v in s_)/math.sqrt(sum(v*v for v in s_)/len(s_)) for s_ in sl))
        print(name, n, {k: ("%.2e..%.2e" % (min(v), max(v))) for k,v in res.items()})
