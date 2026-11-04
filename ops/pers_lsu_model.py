"""What bounds the Tetra decode stream: three floors per arm, from the card's SASS and the file's words.

    python3 -I ops/pers_lsu_model.py

Every output is *computed* from the inputs below, and *estimated* as a time, since each floor rests on a
model of the SM (docs/mesures/pers-sass-2026-11-05.txt):

  LSU   the L1/shared data pipe passes one wavefront a cycle a SM. A global load costs one wavefront per
        distinct 128-byte line its active lanes touch (the L1TEX tag rate), a shared load the largest number of
        distinct 4-byte words one bank is asked for, a 128-bit shared load that over each quarter-warp.
  issue one warp instruction a cycle a scheduler, four schedulers a SM.
  DRAM  the bytes at 864 GB/s, the L40S's peak.

Per warp-iteration wavefronts come from `tetragather` on the sealed 4B's words (every 4th row, tile 64, the
168 Tetra matrices it holds, 2026-11-05): d_in 2,560 and 9,728 are read off the file, d_in 4,096 (o_proj,
int4 in the sealed file) is interpolated between them by active lanes (28.33). Instruction counts are the
inner loops of the card's cubin (tv_tetra48_persall, driver ptxas) and of the devel ptxas for the L2 arms.
"""

F_MAX, F_LOW = 2.52e9, 2.20e9  # Hz: gpu.txt's clocks.max.sm, and a sustained clock under load (assumed)
SMS = 142
DRAM = 864e9

# Warp-iterations of one pass over the bare 4B (36 layers), lane l on blocks l, l + 32, ...: per row
# ceil over 64-block tiles, which equals ceil(nblocks / 32) for these widths.
ITERS = {2560: 36 * 25600 * 4, 4096: 36 * 2560 * 6, 9728: 36 * 2560 * 13}
ROWS = {2560: 36 * 25600, 4096: 36 * 2560, 9728: 36 * 2560}
TAIL_W = {2560: 16, 4096: 16, 9728: 8}

# tetragather, per warp-iteration: words (two loads), tables in L1 (lines), tables in shared (bank
# wavefronts), activation at stride 24 (six LDS.128), trio stride 28 (six LDS.128), stride 25 (24 LDS.32).
LANES = {2560: 26.50, 9728: 31.15}
G = {
    2560: dict(words=3.50, t_l1=83.48, t_sh=16.25, x24=40.50, x28=21.00, x25=24.00),
    9728: dict(words=3.85, t_l1=95.95, t_sh=17.59, x24=47.08, x28=23.54, x25=24.00),
}
_k = (28.33 - LANES[2560]) / (LANES[9728] - LANES[2560])
G[4096] = {key: G[2560][key] + _k * (G[9728][key] - G[2560][key]) for key in G[2560]}

BYTES = 0.9467e9  # the f16-tail family the L2 arms read (pers.rs::bare_file_bytes)
WORD_BYTES = 0.9083e9


def wavefronts(tables, x, decode=True):
    w = 0.0
    for d, n in ITERS.items():
        g = G[d]
        w += n * (g["words"] + (g["t_l1"] if tables == "l1" else g["t_sh"] if decode else 0.0) + (g[x] if x else 0.0))
        if x:
            w += ROWS[d] * 2 * TAIL_W[d]  # lane 0's tail: one load of the tail, one of x, per column
    return w


def floors(name, tables, x, instr, nbytes=BYTES, decode=True, extra_wf=0.0):
    wf = wavefronts(tables, x, decode) + extra_wf
    it = sum(ITERS.values())
    lsu = (wf / (SMS * F_MAX), wf / (SMS * F_LOW))
    iss = (it * instr / (SMS * 4 * F_MAX), it * instr / (SMS * 4 * F_LOW))
    dram = nbytes / DRAM
    lo = max(lsu[0], iss[0], dram)
    hi = max(lsu[1], iss[1], dram)
    bind = max((lsu[1], "LSU"), (iss[1], "issue"), (dram, "DRAM"))[1]
    print(f"  {name:<14} {wf / it:6.1f} wf/it  LSU {lsu[0] * 1e3:5.2f}-{lsu[1] * 1e3:5.2f}  issue {iss[0] * 1e3:4.2f}-"
          f"{iss[1] * 1e3:4.2f}  DRAM {dram * 1e3:4.2f}  floor {lo * 1e3:4.2f}-{hi * 1e3:4.2f} ms ({bind}), "
          f"{nbytes / hi / 1e9:3.0f}-{nbytes / lo / 1e9:3.0f} GB/s at most")


def main():
    it = sum(ITERS.values())
    print(f"{it:,} warp-iterations a pass; clocks {F_LOW / 1e9:.2f}-{F_MAX / 1e9:.2f} GHz; floors in ms")
    # tpersall: the card's measured point. Its 64-block tiles add a staging per CTA-tile (LDG lines + STS,
    # 96 wavefronts) for 9,600 CTA-tiles a layer.
    staging = 36 * 9600 * 96
    print("tpersall, measured 2.394 ms [2.390-2.403] (pers-l1-2026-11-04.txt):")
    floors("tpersall", "l1", "x24", 243, nbytes=0.9806e9, extra_wf=staging)
    print("L2 arms (one CTA a SM, 768 threads; the staging once a site, under 0.02 ms, left out):")
    floors("pword", None, None, 15, nbytes=WORD_BYTES, decode=False)  # 58 a loop unrolled four blocks deep
    floors("pdec_l1", "l1", None, 204, nbytes=WORD_BYTES)
    floors("pdec_smem", "smem", None, 200, nbytes=WORD_BYTES)
    floors("pl1", "l1", "x24", 235)
    floors("pl1_t28", "l1", "x28", 228)
    floors("psmem", "smem", "x24", 231)
    floors("psmem_s25", "smem", "x25", 249)
    floors("psmem_t28", "smem", "x28", 224)
    floors("psmem_t28_u2", "smem", "x28", 235)

if __name__ == "__main__":
    main()
