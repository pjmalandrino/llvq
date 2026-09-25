# Stage classification of the tv_tetra48_h inner loop (one iteration = one
# 24-weight block per lane), sm_89 SASS from NVRTC 12.4. Address sets were
# assigned by reading the data dependencies of each line.
import re, collections, sys
src = sys.argv[1]
ins = []
for l in open(src).read().splitlines():
    m = re.match(r'\s+/\*([0-9a-f]{4})\*/\s+(.*?);', l)
    if m: ins.append((int(m.group(1), 16), m.group(2).strip()))
body = [(a, t) for a, t in ins if 0x270 <= a <= 0x1170]
NORM = {0x8f0, 0x950, 0x980, 0xa80, 0xae0, 0xb00, 0x930, 0x970, 0xac0, 0xaf0, 0xb10, 0xb20}
def op(t): return re.sub(r'^@!?U?P[0-9T]\s+', '', t).split()[0]
def stage(a, t):
    o = op(t)
    if 0x270 <= a <= 0x320 or a in (0x3c0, 0x3f0): return "A fetch the 6-byte word"
    if 0x330 <= a <= 0x650: return "B Golay path and rank rows"
    if 0x660 <= a <= 0x6f0: return "C values as bytes (tables by parity)"
    if a in NORM or 0xb30 <= a <= 0xba0: return "F magnitude and gain"
    if a == 0x990: return "D byte to float"
    if 0x700 <= a <= 0xb20: return "C values as bytes (per section)"
    if a in (0xbb0, 0xbc0, 0xbd0) or o == "LDS.128": return "E activation from shared"
    if a in (0x10c0, 0x1100, 0x1170): return "H loop control"
    if a in (0x1150, 0x1160): return "G scale and accumulate"
    if o == "FFMA": return "E dot (FFMA)"
    if o in ("PRMT", "FADD", "IMAD.U32", "MOV", "IMAD.MOV.U32"): return "D byte to float"
    return "?"
cnt = collections.Counter()
out = []
for a, t in body:
    s = stage(a, t); cnt[s] += 1
    out.append(f"{a:04x}  {s:40s}  {t}")
open("loop-annotated.txt", "w").write("\n".join(out) + "\n")
tot = sum(cnt.values())
for k in sorted(cnt): print(f"{cnt[k]:4d}  {k}")
print(f"{tot:4d}  total per block, {tot/24:.2f} per weight")
