//! Step L0 of the persistent-kernel lead: the persistent kernels' text,
//! executed on the Mac, equals the served kernels' text bit for bit.
//!
//! `tests/host_pers.cpp` is built twice, both without contraction. The
//! SERVED side runs `tv_tetra48_h`, `tv_q4_hh`, `rot_apply` and a replay of
//! candle's rmsnorm tree as written, plus a replay of `tv_tetra48_h` whose
//! three contractible sites can be fused. The unfused replay must equal the
//! served text bit for bit; the fused one, with the fusions NVVM made in the
//! served unit's PTX (`docs/mesures/ptx-probe-2026-11-04.txt` §2), is the
//! reference. The PERSISTENT side compiles `llvq_pers.cuh`, `tetra48_pers.cu`
//! and `pers_segment.cu`, whose arithmetic is spelled with explicit
//! intrinsics. Host clang cannot stand in for NVVM: with `-ffp-contract=fast`
//! it fuses the epilogue and leaves `acc += t * invnorm` alone (found while
//! writing this test).
//! Every comparison below is byte for byte.
//!
//! The kernels run through `tests/host_grid.h`: every CTA of a cooperative
//! grid at once, 32-lane warps, real barriers and spin-waits, and a seeded
//! scheduler that lets warps drift apart.
//!
//! What this establishes: the persistent text computes the served values on
//! a correct compiler, for the operations as NVVM emitted them. What it does
//! not: the card's SASS, timing, memory ordering between real SMs, candle's
//! real SiLU table or rsqrt instruction (both sides call the same stand-in).
//! The card checks those at load (`prereg-persistent-1.md`, gate 5).
//!
//! Each guard of the persistent text has a mutant here, applied to a copy of
//! the sources with an exactly-once count (repo rule 10), and each is killed
//! by an assertion.

use llvq_cuda::pers;
use llvq_quant::rotation::Rotation;
use llvq_search::tetra::Tetra;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

const KMAX: usize = 32;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("workspace root").to_path_buf()
}

/// Build one side from the sources under `root` (the repository, or a
/// mutated copy of the files it needs).
fn build_side(root: &Path, side: u32, contract: &str, tag: &str) -> PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_pers_{tag}_{side}_{contract}_{}", std::process::id()));
    let res = Command::new("clang++")
        .args(["-std=c++17", "-O2", &format!("-ffp-contract={contract}"), "-Wall", "-Wextra", "-Werror"])
        .arg(format!("-DPERS_SIDE={side}"))
        .arg(root.join("llvq-cuda/tests/host_pers.cpp"))
        .arg("-o")
        .arg(&out)
        .output()
        .expect("clang++ is on PATH (hard rule 10: this test fails, it does not skip)");
    assert!(res.status.success(), "clang++ refused host_pers.cpp ({tag}, side {side}):\n{}", String::from_utf8_lossy(&res.stderr));
    out
}

fn served() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side(&repo(), 0, "off", "served"))
}

fn persistent() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side(&repo(), 1, "off", "pers"))
}

/// A copy of the files the harness reads, with `from` replaced by `to`
/// exactly once in `file` (a path relative to the repository).
fn mutant(tag: &str, file: &str, from: &str, to: &str) -> PathBuf {
    mutant_many(tag, &[(file, from, to)])
}

/// The same with several edits, each applied exactly once.
fn mutant_many(tag: &str, edits: &[(&str, &str, &str)]) -> PathBuf {
    let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for dir in ["llvq-cuda/kernels", "llvq-cuda/tests", "llvq-llm/kernels"] {
        let src = repo().join(dir);
        let dst = root.join(dir);
        std::fs::create_dir_all(&dst).unwrap();
        for e in std::fs::read_dir(&src).unwrap() {
            let p = e.unwrap().path();
            let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("");
            if ["cu", "cuh", "h", "cpp", "inc"].contains(&ext) {
                std::fs::copy(&p, dst.join(p.file_name().unwrap())).unwrap();
            }
        }
    }
    for &(file, from, to) in edits {
        let path = root.join(file);
        let text = std::fs::read_to_string(&path).unwrap();
        let n = text.matches(from).count();
        assert_eq!(n, 1, "mutant {tag}: {n} sites for {from:?} in {file}, expected exactly one");
        std::fs::write(&path, text.replacen(from, to, 1)).unwrap();
    }
    build_side(&root, 1, "off", tag)
}

/// The decoder tables and invnorm, as every fixture's header.
fn header() -> &'static Vec<u8> {
    static H: OnceLock<Vec<u8>> = OnceLock::new();
    H.get_or_init(|| {
        let t = Tetra::new();
        let mut h = Vec::new();
        for v in t.rows() {
            h.extend(v.to_le_bytes());
        }
        for s in 0..64usize {
            h.extend_from_slice(&t.prefixes()[s]);
        }
        for s in 0..64usize {
            for b in 0..16usize {
                let (byte, s16) = t.branches()[s][b];
                h.extend((byte as u16 | (s16 as u16) << 8).to_le_bytes());
            }
        }
        for s in 0..64usize {
            h.extend_from_slice(&t.suffixes()[s]);
        }
        for m in 0..32usize {
            let v = if m == 0 { 0.0f32 } else { (1.0f64 / ((16 * m) as f64).sqrt()) as f32 };
            h.extend(v.to_le_bytes());
        }
        h
    })
}

fn run(bin: &Path, cmd: &[u8]) -> Vec<u8> {
    let mut child = Command::new(bin).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = header().clone();
    input.extend_from_slice(cmd);
    child.stdin.take().unwrap().write_all(&input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{} failed: {}", bin.display(), String::from_utf8_lossy(&out.stderr));
    out.stdout
}

/// A mutant's output, or `None` when the process failed: a crash or an abort
/// kills the mutant as surely as a wrong byte does.
fn run_mutant(bin: &Path, cmd: &[u8]) -> Option<Vec<u8>> {
    let mut child = Command::new(bin).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = header().clone();
    input.extend_from_slice(cmd);
    let _ = child.stdin.take().unwrap().write_all(&input);
    let out = child.wait_with_output().unwrap();
    out.status.success().then_some(out.stdout)
}

struct Cmd(Vec<u8>);
impl Cmd {
    fn new(tag: u8) -> Self {
        Cmd((tag as u32).to_le_bytes().to_vec())
    }
    fn u32(mut self, v: u32) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }
    fn u64(mut self, v: u64) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }
    fn f32(mut self, v: f32) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }
    fn u32s(mut self, v: &[u32]) -> Self {
        for x in v {
            self.0.extend(x.to_le_bytes());
        }
        self
    }
    fn f32s(mut self, v: &[f32]) -> Self {
        for x in v {
            self.0.extend(x.to_le_bytes());
        }
        self
    }
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}
fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Sign words, padded odd block and 1/sqrt(m) of the rotation at width `n`.
struct Rot {
    m: u32,
    k: u32,
    inv: f32,
    signs: Vec<u32>,
    small: Vec<f32>,
}
fn rot(n: usize, seed: u64) -> Rot {
    let r = Rotation::new(n, seed);
    let (m, k) = (r.pow2(), r.odd());
    assert_eq!(pers::rot_mk(n as u32).unwrap(), (m as u32, k as u32), "pers.rs plans the rotation as llvq-quant does");
    let mut signs = vec![0u32; n.div_ceil(32)];
    for (i, &s) in r.signs().iter().enumerate() {
        if s < 0.0 {
            signs[i >> 5] |= 1 << (i & 31);
        }
    }
    let mut small = vec![0.0f32; KMAX * KMAX];
    for g in 0..k {
        for t in 0..k {
            small[g * KMAX + t] = r.small()[g * k + t] as f32;
        }
    }
    Rot { m: m as u32, k: k as u32, inv: (1.0f64 / (m as f64).sqrt()) as f32, signs, small }
}

fn rot_cmd(n: u32, threads: u32, seed: u64) -> Vec<u8> {
    let r = rot(n as usize, seed);
    Cmd::new(b'R').u32(n).u32(r.m).u32(r.k).f32(r.inv).u32(threads).u64(seed).u32s(&r.signs).f32s(&r.small).0
}

/// Every width the three served models rotate, the register WHT and the
/// shared-memory stages above 512 included (m = 1,024 at the 14B, 4,096 at
/// the 8B and on o_proj).
const ROT_WIDTHS: [u32; 6] = [2560, 9728, 4096, 12288, 5120, 17408];

#[test]
fn the_rotation_prologue_is_rot_apply() {
    for &n in &ROT_WIDTHS {
        for threads in [64u32, 256] {
            let cmd = rot_cmd(n, threads, 0x5eed_0000 + n as u64);
            let a = run(served(), &cmd);
            let b = run(persistent(), &cmd);
            assert_eq!(a.len(), 4 * n as usize);
            assert!(a == b, "n = {n}, {threads} threads: the persistent rotation differs from rot_apply");
        }
    }
}

/// A column whose last real term underflows to -0: rot_mix's padded terms
/// turn it into +0, and so must the persistent mix's single `+ 0.0f`.
#[test]
fn the_mix_turns_minus_zero_into_plus_zero_as_the_padding_does() {
    let (m, k) = (32u32, 3u32);
    let mut s = vec![0.0f32; (k * m) as usize];
    let mut small = vec![0.0f32; KMAX * KMAX];
    for j in 0..m as usize {
        s[2 * m as usize + j] = 1e-30;
        s[j] = if j % 2 == 0 { 0.0 } else { 0.5 };
    }
    for g in 0..k as usize {
        small[g * KMAX + 2] = -1e-20; // -1e-50 underflows to -0 under round to nearest
        small[g * KMAX] = if g == 0 { 1.0 } else { 0.0 };
    }
    let cmd = Cmd::new(b'M').u32(m).u32(k).f32(1.0).f32s(&s).f32s(&small).0;
    let a = run(served(), &cmd);
    let b = run(persistent(), &cmd);
    assert_eq!((-1e-20f32).mul_add(1e-30, 0.0).to_bits(), 0x8000_0000, "the fixture's last term rounds to -0");
    assert!(a == b, "the persistent mix differs from rot_mix");
    assert!(u32s(&a).contains(&0), "some output is +0 (the column the padding fixed)");
    let mutant = mutant("mix_no_fadd", "llvq-cuda/kernels/llvq_pers.cuh", "if (k < LLVQ_ROT_KMAX) acc = __fadd_rn(acc, 0.0f);", "");
    let c = run(&mutant, &cmd);
    assert!(a != c, "dropping the + 0.0f survived the -0 column");
}

#[test]
fn rotation_mutants_are_killed() {
    let cmd = rot_cmd(4096, 128, 7);
    let cmd2 = rot_cmd(2560, 128, 9);
    let want = run(served(), &cmd);
    let want2 = run(served(), &cmd2);
    let cases = [
        ("wht_lane_stage_order", "for (u32 len = 1u; len < 16u; len <<= 1) {", "for (u32 len = 8u; len >= 1u; len >>= 1) {"),
        ("wht_upper_sign", "v[i] = upper ? (p - v[i]) : (v[i] + p);", "v[i] = upper ? (v[i] - p) : (v[i] + p);"),
        ("wht_shfl_order", "for (u32 sh = 1u; sh < 32u; sh <<= 1) {", "for (u32 sh = 16u; sh >= 1u; sh >>= 1) {"),
        ("smem_stage_start", "for (u32 len = 512u; len < m; len <<= 1) {", "for (u32 len = 1024u; len < m; len <<= 1) {"),
        ("scale_k1", "for (u32 i = threadIdx.x; i < n; i += blockDim.x) s[i] = s[i] * inv;", "for (u32 i = threadIdx.x; i < n; i += blockDim.x) s[i] = s[i];"),
        ("rot_sign", "s[i] = ((signbits[i >> 5] >> (i & 31u)) & 1u) ? -v : v;\n    }\n}\n\n// Phases", "s[i] = ((signbits[i >> 5] >> (i & 30u)) & 1u) ? -v : v;\n    }\n}\n\n// Phases"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, "llvq-cuda/kernels/llvq_pers.cuh", from, to);
        let killed = run(&b, &cmd) != want || run(&b, &cmd2) != want2;
        assert!(killed, "rotation mutant {tag} survived");
    }
}

fn norm_cmd(ncols: u32, variant: u32, threads: u32, seed: u64) -> Vec<u8> {
    Cmd::new(b'N').u32(ncols).f32(1e-6).u32(variant).u32(threads).u64(seed).0
}

/// candle's rmsnorm: block 1,024 from 1,024 columns, 32 below (q/k-norm at
/// 128), both contraction variants, on CTAs of 64 to 768 threads.
#[test]
fn the_norm_is_candles_tree() {
    let mut variants_differ = false;
    for ncols in [2560u32, 4096, 5120, 1024, 1000, 128] {
        for threads in [64u32, 256, 768] {
            for seed in [1u64, 2] {
                let a0 = run(served(), &norm_cmd(ncols, 0, threads, seed));
                let a1 = run(served(), &norm_cmd(ncols, 1, threads, seed));
                variants_differ |= a0 != a1;
                assert!(a0 == run(persistent(), &norm_cmd(ncols, 0, threads, seed)), "ncols {ncols}, {threads} threads, variant 0");
                assert!(a1 == run(persistent(), &norm_cmd(ncols, 1, threads, seed)), "ncols {ncols}, {threads} threads, variant 1");
            }
        }
    }
    // xi is a widened binary16 (11 significant bits), so xi * xi is exact in
    // binary32 and the contracted and split accumulations round the same
    // sums: the variant nvcc picked for candle cannot matter.
    assert!(!variants_differ, "the contracted and split norms differ on binary16 inputs");
}

#[test]
fn norm_mutants_are_killed() {
    let mut cmds: Vec<Vec<u8>> = Vec::new();
    for seed in 3..9u64 {
        for (n, v) in [(2560u32, 0u32), (128, 0), (5120, 1), (4096, 0)] {
            cmds.push(norm_cmd(n, v, 256, seed));
        }
    }
    let want: Vec<Vec<u8>> = cmds.iter().map(|c| run(served(), c)).collect();
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    let cases = [
        ("norm_butterfly_up", "for (int mask = 16; mask > 0; mask >>= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));\n        if (lane == 0u) s_sum[vw] = tmp;", "for (int mask = 1; mask < 32; mask <<= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));\n        if (lane == 0u) s_sum[vw] = tmp;"),
        ("norm_block_768", "const u32 block = ncols < 1024u ? 32u : 1024u;", "const u32 block = ncols < 1024u ? 32u : 768u;"),
        ("norm_block_switch", "const u32 block = ncols < 1024u ? 32u : 1024u;", "const u32 block = ncols < 128u ? 32u : 1024u;"),
        ("norm_mean_mul", "const float mean = __fdiv_rn(tot, (float)ncols);", "const float mean = __fmul_rn(tot, 1.0f / (float)ncols);"),
        ("norm_apply_order", "return f2h(__fmul_rn(__fmul_rn(scale, h2f(x)), h2f(alpha)));", "return f2h(__fmul_rn(scale, __fmul_rn(h2f(x), h2f(alpha))));"),
        ("norm_square_rounded", "const float xi = h2f(__ldcg(x + col));", "const float xi = __fmul_rn(h2f(__ldcg(x + col)), 1.0009765625f);"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = cmds.iter().zip(&want).any(|(c, w)| &run(&b, c) != w);
        assert!(killed, "norm mutant {tag} survived");
    }
}

fn rows_cmd(d_out: u32, d_in: u32, variant: u32, grid: u32, threads: u32, seed: u64) -> Vec<u8> {
    Cmd::new(b'T').u32(d_out).u32(d_in).u32(variant).u32(grid).u32(threads).u64(seed).0
}

/// The replay of `tv_tetra48_h` with nothing fused IS the served text: the
/// same lane order, tiling, butterfly and epilogue, bit for bit on y and on
/// the f32 value before `f2h`. With the three PTX fusions it differs on some
/// rows: the sites are exercised.
#[test]
fn the_served_replay_is_the_served_text() {
    let mut fused_differs = false;
    for &(d_out, d_in) in &[(64u32, 2560u32), (48, 4096), (40, 9728), (16, 120)] {
        let out = run(served(), &Cmd::new(b'U').u32(d_out).u32(d_in).u64(0x7e7a_0000 + d_in as u64).0);
        let k = 6 * d_out as usize;
        assert_eq!(out.len(), 3 * k);
        assert!(out[..k] == out[k..2 * k], "{d_out}x{d_in}: the unfused replay is not the served text");
        fused_differs |= out[k..2 * k] != out[2 * k..];
    }
    assert!(fused_differs, "the fused and unfused replays never differ: the three sites are not exercised");
}

/// A persistent Tetra row equals the served row, as NVVM compiled it, in
/// every staging layout and grid: y and the f32 value before `f2h`.
#[test]
fn tetra_rows_are_the_served_rows() {
    for &(d_out, d_in) in &[(64u32, 2560u32), (48, 4096), (40, 9728), (16, 120)] {
        let seed = 0x7e7a_0000 + d_in as u64;
        let want = run(served(), &rows_cmd(d_out, d_in, 0, 1, 64, seed));
        assert_eq!(want.len(), 6 * d_out as usize, "y in binary16, then the f32 value before f2h");
        for variant in 0..4u32 {
            for &(grid, threads) in &[(3u32, 96u32), (2, 64)] {
                let got = run(persistent(), &rows_cmd(d_out, d_in, variant, grid, threads, seed));
                assert!(got == want, "{d_out}x{d_in}, variant {variant}, grid {grid}x{threads}: rows or their f32 values differ");
            }
        }
    }
}

#[test]
fn tetra_row_mutants_are_killed() {
    let cmds: Vec<Vec<u8>> = [(64u32, 2560u32), (40, 9728)].iter().map(|&(o, i)| rows_cmd(o, i, 2, 3, 96, 11)).collect();
    let want: Vec<Vec<u8>> = cmds.iter().map(|c| run(served(), c)).collect();
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    let cases = [
        ("site1_unfused", "return __fmaf_rn(__fmul_rn(acc_b, gscale[g]), invnorm[m], acc);", "return __fadd_rn(acc, __fmul_rn(__fmul_rn(acc_b, gscale[g]), invnorm[m]));"),
        ("site1_assoc", "return __fmaf_rn(__fmul_rn(acc_b, gscale[g]), invnorm[m], acc);", "return __fmaf_rn(acc_b, __fmul_rn(gscale[g], invnorm[m]), acc);"),
        ("site2_unfused", "for (u32 i = 0; i < tail_w; ++i) tv = __fmaf_rn(h2f(tail_row[i]), xt[i], tv);", "for (u32 i = 0; i < tail_w; ++i) tv = __fadd_rn(tv, __fmul_rn(h2f(tail_row[i]), xt[i]));"),
        ("site3_unfused", "return __fmaf_rn(acc, rscale, tv);", "return __fadd_rn(__fmul_rn(acc, rscale), tv);"),
        ("trio_component", "a = __fmaf_rn(f1r_v3_float(q[i], 1u), v.y, a);", "a = __fmaf_rn(f1r_v3_float(q[i], 1u), v.z, a);"),
        ("inv_order", "0, 1, 2, 3, 4, 16, 8, 5, 17, 18,", "0, 1, 2, 3, 4, 8, 16, 5, 17, 18,"),
        ("lane_order", "for (u32 j = lane; j < nblocks; j += 32u) {\n            u32 lo, hi16;", "for (u32 j = nblocks - 1u - lane; j < nblocks; j -= 32u) {\n            u32 lo, hi16;"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = cmds.iter().zip(&want).any(|(c, w)| &run(&b, c) != w);
        assert!(killed, "Tetra row mutant {tag} survived");
    }
}

#[test]
fn int4_rows_are_the_served_rows() {
    for &(d_out, d_in) in &[(64u32, 9728u32), (40, 2560), (16, 4096)] {
        let cmd = Cmd::new(b'Q').u32(d_out).u32(d_in).u32(3).u32(96).u64(0x9400 + d_in as u64).0;
        assert!(run(served(), &cmd) == run(persistent(), &cmd), "{d_out}x{d_in}");
    }
    let cmd = Cmd::new(b'Q').u32(64).u32(2560).u32(3).u32(96).u64(5).0;
    let want = run(served(), &cmd);
    // Fusing the dequantization is an EQUIVALENT mutant: an 11-bit scale
    // times a 4-bit code is exact in binary32. Checked as such.
    let b = mutant("q4_deq_fused", "llvq-cuda/kernels/llvq_pers.cuh", "return __fadd_rn(__fmul_rn(h2f((unsigned short)sbits), (float)q), h2f((unsigned short)bbits));", "return __fmaf_rn(h2f((unsigned short)sbits), (float)q, h2f((unsigned short)bbits));");
    assert!(run(&b, &cmd) == want, "the fused dequantization differs: the exactness argument is wrong");
    let b = mutant("q4_group", "llvq-cuda/kernels/llvq_pers.cuh", "const u32 g = s0 + (c >> 7);", "const u32 g = s0 + (c >> 6) / 2u + (wi & 1u);");
    assert!(run(&b, &cmd) != want, "a wrong group index survived");
    let b = mutant("q4_lane_order", "llvq-cuda/kernels/llvq_pers.cuh", "for (u32 wi = lane; wi < nwords; wi += 32u) {\n        const u32 p = wq[w0 + wi];\n        const u32 c = wi << 3;\n        const u32 g", "for (u32 wi = nwords - 1u - lane; wi < nwords; wi -= 32u) {\n        const u32 p = wq[w0 + wi];\n        const u32 c = wi << 3;\n        const u32 g");
    assert!(run(&b, &cmd) != want, "a reversed int4 lane order survived");
}

#[test]
fn the_argmax_key_is_tok_argmax_key() {
    let cmd = Cmd::new(b'K').0;
    assert_eq!(u32s(&run(persistent(), &cmd)), vec![0], "the persistent key differs from tok_argmax.cu's");
    let b = mutant("key_negzero", "llvq-cuda/kernels/llvq_pers.cuh", "if (b == 0x8000u) b = 0u;\n    return (b & 0x8000u)", "return (b & 0x8000u)");
    assert_ne!(u32s(&run(&b, &cmd)), vec![0], "the -0 fold of the key survived");
    let b = mutant("key_brev", "llvq-cuda/kernels/llvq_pers.cuh", "(1023u - (__brev(i & 1023u) >> 22))", "(1023u - (i & 1023u))");
    assert_ne!(u32s(&run(&b, &cmd)), vec![0], "the bit reversal of the key survived");
}

/// The host mirrors of the device structs (`pers.rs`) against the kernels'
/// own definitions: a field one word off would read a wrong pointer on the
/// card, with no error.
#[test]
fn the_struct_mirrors_match_the_kernel_layouts() {
    let o = u32s(&run(persistent(), &Cmd::new(b'L').0));
    let mut want = vec![pers::PERS_ARGS_BYTES as u32];
    want.extend((0..pers::PERS_ARGS_PTRS.len() as u32).map(|i| 8 * i));
    // nsites, hidden, inter, m_h, k_h, m_i, k_i, then (3 pads) inv_h, inv_i, eps
    want.extend([176, 180, 184, 188, 192, 196, 200, 216, 220, 224]);
    want.push(pers::PERS_LAUNCH_BYTES as u32);
    want.extend([0, 4, 8, 16, 24, 32]);
    want.push(4 * pers::SITE_U32 as u32);
    want.push(48);
    want.extend([0, 8, 16, 24, 32, 36, 40, 44]);
    want.push(pers::SEG_ARGS_BYTES as u32);
    want.extend([0, 8, 16, 24, 32, 40, 88, 136, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288]);
    want.extend([296, 300, 304, 308, 312, 316, 320, 324, 328, 332]);
    want.push(pers::SEG_LAUNCH_BYTES as u32);
    want.extend([0, 8, 16]);
    assert_eq!(o, want);
    assert_eq!(std::mem::size_of::<pers::SegLaunch>(), pers::SEG_LAUNCH_BYTES);
    let seg = pers::SegArgsHost { act: 5, gpr: 76, eps: 1e-6, down: pers::SegTetraHost { d_out: 2560, ..Default::default() }, ..Default::default() }.bytes();
    assert_eq!(seg.len(), pers::SEG_ARGS_BYTES);
    assert_eq!(u64::from_le_bytes(seg[32..40].try_into().unwrap()), 5);
    assert_eq!(u32::from_le_bytes(seg[136 + 44..136 + 48].try_into().unwrap()), 2560);
    assert_eq!(f32::from_le_bytes(seg[328..332].try_into().unwrap()), 1e-6);
    assert_eq!(u32::from_le_bytes(seg[332..336].try_into().unwrap()), 76);
    // the Rust image puts the same values at the same offsets
    let img = pers::PersArgsHost { ptrs: std::array::from_fn(|i| 0x1000 + i as u64), nsites: 7, inv_h: 0.5, eps: 1e-6, ..Default::default() }.bytes();
    assert_eq!(u64::from_le_bytes(img[8 * 21..8 * 22].try_into().unwrap()), 0x1000 + 21);
    assert_eq!(u32::from_le_bytes(img[176..180].try_into().unwrap()), 7);
    assert_eq!(f32::from_le_bytes(img[216..220].try_into().unwrap()), 0.5);
    assert_eq!(f32::from_le_bytes(img[224..228].try_into().unwrap()), 1e-6);
}

/// The L1 one-launch pass walks its work list onto the right rows: a fused
/// group (two parts, two centroid pairs) and a single matrix, on grids
/// smaller and larger than the 7 work groups, equal to `tv_tetra48_seg` on
/// each site, bit for bit.
#[test]
fn persall_is_the_fused_kernel_site_by_site() {
    for grid in [1u32, 3, 7, 12] {
        let cmd = Cmd::new(b'A').u32(grid).u64(0xa11).0;
        assert!(run(served(), &cmd) == run(persistent(), &cmd), "grid {grid}");
    }
    let cmd = Cmd::new(b'A').u32(3).u64(0xa11).0;
    let want = run(served(), &cmd);
    let b = mutant("persall_scan", "llvq-cuda/kernels/tetra48_persall.cu", "w >= (u32)(sites[(si + 1u) * LLVQ_TOCC_SITE_WORDS + 7u] & 0xffffffffull)) ++si;", "w > (u32)(sites[(si + 1u) * LLVQ_TOCC_SITE_WORDS + 7u] & 0xffffffffull)) ++si;");
    assert!(run_mutant(&b, &cmd) != Some(want), "an off-by-one site scan survived");
}

/// 4E's head phase: the persistent head rows are the served tv_emb_q4_h
/// logits bit for bit, and the token every CTA reads back after its W4
/// barrier is the served argmax (candle's tie order), planted ties included,
/// on grids that split the vocabulary differently.
#[test]
fn the_4e_head_phase_is_the_served_argmax() {
    for (vocab, plant) in [(3072u32, 0u32), (3072, 1), (4104, 1)] {
        for (grid, threads) in [(3u32, 64u32), (5, 96)] {
            let cmd = Cmd::new(b'H').u32(vocab).u32(256).u32(grid).u32(threads).u32(plant).u64(0x4e + vocab as u64).0;
            let (want, got) = (run(served(), &cmd), run(persistent(), &cmd));
            assert!(u32s(&want[..4])[0] < vocab);
            assert!(want == got, "vocab {vocab}, plant {plant}, grid {grid}x{threads}: token or logits differ");
        }
    }
    let cmd = Cmd::new(b'H').u32(3072).u32(256).u32(3).u32(64).u32(1).u64(0x4e + 3072).0;
    let want = run(served(), &cmd);
    let b = mutant("head_no_w4", "llvq-cuda/kernels/llvq_pers.cuh", "    if (!pers_grid_wait(bar, bar_target, abort_flag)) return 0xffffffffu;\n    if (pers_ld_acquire(nan_flag)", "    (void)bar; (void)bar_target; (void)abort_flag;\n    if (pers_ld_acquire(nan_flag)");
    assert!(run_mutant(&b, &cmd) != Some(want.clone()), "the W4 barrier is not load-bearing");
    let b = mutant("head_key_index", "llvq-cuda/kernels/llvq_pers.cuh", "return ((0x3FFFFFu - (u32)(k & 0x3FFFFFu)) << 10) | (__brev(br) >> 22);", "return ((0x3FFFFFu - (u32)(k & 0x3FFFFFu)) << 10) | br;");
    assert!(run_mutant(&b, &cmd) != Some(want), "the key's bit reversal on the way back survived");
}

/// 4E's embedding, q/k/v and o phases equal the served chain bit for bit:
/// emb_q4_gather; ln1, rot_apply, tv_tetra48_h (as NVVM compiled it) and
/// tv_q4_hh on the unrotated ln1 output; tv_q4_hh then the residual add.
#[test]
fn the_4e_layer_phases_are_the_served_chain() {
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    for (tok, grid) in [(0u32, 3u32), (517, 4), (1023, 5)] {
        let cmd = Cmd::new(b'E').u32(1024).u32(2560).u32(tok).u32(grid).u64(0xe3b).0;
        assert!(run(served(), &cmd) == run(persistent(), &cmd), "embedding, token {tok}, grid {grid}");
    }
    let r = rot(2560, 41);
    let qkv = |grid: u32, threads: u32, seed: u64| {
        Cmd::new(b'V').u32(grid).u32(threads).u64(seed).u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small).0
    };
    for (grid, threads, seed) in [(3u32, 96u32, 1u64), (4, 64, 2)] {
        assert!(run(served(), &qkv(grid, threads, seed)) == run(persistent(), &qkv(grid, threads, seed)), "q/k/v, grid {grid}");
    }
    for (grid, threads) in [(3u32, 96u32), (5, 64)] {
        let cmd = Cmd::new(b'W').u32(grid).u32(threads).u64(7).0;
        assert!(run(served(), &cmd) == run(persistent(), &cmd), "o, grid {grid}");
    }
    // mutants
    let e = Cmd::new(b'E').u32(1024).u32(2560).u32(517).u32(3).u64(0xe3b).0;
    let b = mutant("embed_group", f, "const u32 g = s0 + (c >> 6);\n        dst[c]", "const u32 g = s0 + (c >> 7);\n        dst[c]");
    assert!(run_mutant(&b, &e) != Some(run(served(), &e)), "the embedding's g64 group survived as g128");
    let v = qkv(3, 96, 1);
    let want = run(served(), &v);
    let b = mutant("v_reads_rotated", f, "pers_q4_row(P.vw, P.vsc, P.vbi, xn,", "pers_q4_row(P.vw, P.vsc, P.vbi, ((void)xn, xr),");
    assert!(run_mutant(&b, &v) != Some(want.clone()), "v reading the rotated input survived");
    let b = mutant("k_offset", f, "const u32 rk = r - P.q_dout;", "const u32 rk = r - P.q_dout + 1u;");
    assert!(run_mutant(&b, &v) != Some(want), "a k row offset survived");
    let w = Cmd::new(b'W').u32(3).u32(96).u64(7).0;
    let b = mutant("o_residual", f, "out[r] = exact_hadd(__ldcg(x + r), f2h(acc));", "out[r] = f2h(__fadd_rn(h2f(__ldcg(x + r)), acc));");
    assert!(run_mutant(&b, &w) != Some(run(served(), &w)), "the o residual's binary16 rounding of o survived");
}

fn attn_cmd(n_kv: u32, groups: u32, cap: u32, l: u32, grid: u32, threads: u32, seed: u64) -> (Vec<u8>, Vec<u16>, Vec<u32>) {
    use llvq_cuda::attn_ref;
    let nh = (n_kv * groups) as usize;
    let q = attn_ref::fill(nh * 128, seed, 2.0);
    let mut k = attn_ref::fill(n_kv as usize * cap as usize * 128, seed + 1, 1.0);
    let v = attn_ref::fill(n_kv as usize * cap as usize * 128, seed + 2, 1.0);
    // a planted run of equal keys, so ties of the tile maximum are exercised
    for p in 0..3usize {
        let (a, b) = (p * 128, (p + 9) * 128);
        if b + 128 <= k.len() / n_kv as usize {
            let row: Vec<u16> = k[a..a + 128].to_vec();
            k[b..b + 128].copy_from_slice(&row);
        }
    }
    let c = attn_ref::score_scale(128);
    let kv = attn_ref::Kv { k: &k, v: &v, cap: cap as usize };
    let want = attn_ref::attend(&q, &kv, n_kv as usize, groups as usize, l as usize - 1, 1, c, attn_ref::UNIT_SERVED);
    // the f32 state of each head before the division, built the reference's
    // way: units into splits, splits into the row
    let mut state = Vec::with_capacity(nh * 130);
    for hq in 0..nh {
        let mut qf = [0f32; 128];
        for (i, x) in qf.iter_mut().enumerate() {
            *x = attn_ref::h2f(q[hq * 128 + i]);
        }
        let mut total = attn_ref::St::empty();
        for s in 0..(l as usize).div_ceil(attn_ref::SPLIT) {
            attn_ref::st_merge(&mut total, &attn_ref::split_state(&qf, &kv, hq / groups as usize, s, l as usize, c, attn_ref::UNIT_SERVED));
        }
        state.extend(total.a.iter().map(|x| x.to_bits()));
        state.push(total.m.to_bits());
        state.push(total.d.to_bits());
    }
    let mut cmd = Cmd::new(b'X').u32(n_kv).u32(groups).u32(cap).u32(l).u32(grid).u32(threads).f32(c).0;
    for x in q.iter().chain(&k).chain(&v) {
        cmd.extend(x.to_le_bytes());
    }
    (cmd, want, state)
}

/// 4E's attention phase on the persistent grid equals attn_ref.rs (and so
/// attn_dec.cu) bit for bit: lengths inside one tile, at unit and split
/// boundaries, across several splits; groups of 4 (4B, 8B) and 5 (14B);
/// grids of 2 to 5 CTAs; one barrier between its two phases.
#[test]
fn the_4e_attention_phase_is_attn_ref() {
    for &(n_kv, groups, cap, l) in &[(2u32, 4u32, 64u32, 1u32), (2, 4, 64, 5), (2, 4, 128, 64), (2, 4, 128, 65), (2, 5, 512, 256), (2, 5, 512, 257), (1, 4, 1408, 1300)] {
        for &(grid, threads) in &[(3u32, 64u32), (5, 96)] {
            let (cmd, want, state) = attn_cmd(n_kv, groups, cap, l, grid, threads, 0xa77 + l as u64);
            let got = run(persistent(), &cmd);
            let n = want.len() * 2;
            assert_eq!(u16s(&got[..n]), want, "n_kv {n_kv}, groups {groups}, l {l}, grid {grid}x{threads}");
            let t = u32s(&got[n..n + 8]);
            assert_eq!((t[0], t[1]), (grid, 0), "one barrier, no abort");
            assert_eq!(u32s(&got[n + 8..]), state, "the f32 state before the division, l {l}");
        }
    }
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    let (cmd, want, state) = attn_cmd(1, 4, 1408, 1300, 3, 64, 0xa77);
    let mut want: Vec<u8> = want.iter().flat_map(|x| x.to_le_bytes()).collect();
    want.extend([3u32, 0].iter().flat_map(|x| x.to_le_bytes()));
    want.extend(state.iter().flat_map(|x| x.to_le_bytes()));
    let cases = [
        ("attn_flat_merge", "pers_attn_merge(sm, sd, sa, __ldcg(st + PERS_HD), __ldcg(st + PERS_HD + 1u), ba);", "pers_attn_merge(tm, td, ta, __ldcg(st + PERS_HD), __ldcg(st + PERS_HD + 1u), ba);"),
        ("attn_carry_fused", "d = __fadd_rn(__fmul_rn(d, carry), ts);", "d = __fmaf_rn(d, carry, ts);"),
        ("attn_tile", "#define PERS_TT 8u", "#define PERS_TT 16u"),
        ("attn_no_barrier", "    if (!pers_grid_wait(bar, bar_target, abort_flag)) return false;\n    for (u32 hq = gwarp;", "    (void)bar; (void)bar_target; (void)abort_flag;\n    for (u32 hq = gwarp;"),
        ("attn_scale", "sc[j] = __fmul_rn(t, c);", "sc[j] = __fmul_rn(t, __fmul_rn(c, 0.5f));"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &cmd) != Some(want.clone()), "attention mutant {tag} survived");
    }
}

/// The fused binary16 FMA of RoPE's candidate forms rounds once from the exact
/// value: against an exact integer reference on 20,000 random finite triples
/// and on planted ties decided by the third term's sign or by evenness.
#[test]
fn the_fused_f16_fma_rounds_once() {
    let cmd = Cmd::new(b'F').u32(20_000).u64(0xf16).0;
    let (want, got) = (run(served(), &cmd), run(persistent(), &cmd));
    assert!(want == got, "the f64 + TwoSum emulation differs from the exact reference");
    let tail: Vec<u16> = u16s(&want)[20_000..20_005].to_vec();
    assert_eq!(tail, vec![0x3e02, 0x3e02, 0x3e01, 0x3e02, 0x3e01], "the planted ties: even, up, down, up, down");
    // An EQUIVALENT mutant, checked as such: for binary16 inputs the f64 sum is
    // exact (an in-range product and the third term span at most 41 bits), so
    // TwoSum's error is always zero and the tie rule never reads it.
    let b = mutant("tie_ignores_err", "llvq-cuda/kernels/llvq_pers.cuh", "(rem == half && (beyond || (err == 0.0 && (q & 1ull))))", "(rem == half && ((void)beyond, (q & 1ull)))");
    assert!(run_mutant(&b, &cmd) == Some(want.clone()), "dropping the TwoSum error changed a result: the exactness argument is wrong");
    let b = mutant("f16_round_down", "llvq-cuda/kernels/llvq_pers.cuh", "if (rem > half || (rem == half && (beyond", "if (rem > half + half || (rem == half && (beyond");
    assert!(run_mutant(&b, &cmd) != Some(want), "a wrong rounding of the f64 value survived");
}

/// 4E's q/k-norm, RoPE and KV append, the three candidate forms, against the
/// served replay (candle's 32-thread norm tree, rope's pairing at i and i +
/// 64, the exact reference for the fused forms): q out, the K and V caches
/// written at `pos` only.
#[test]
fn the_4e_norm_rope_append_phase() {
    let z = |form: u32, pos: u32, grid: u32| Cmd::new(b'Z').u32(form).u32(8).u32(2).u32(16).u32(pos).u32(grid).u32(64).u64(0x20 + pos as u64).0;
    for form in 0..3u32 {
        for (pos, grid) in [(0u32, 2u32), (7, 3), (15, 4)] {
            assert!(run(served(), &z(form, pos, grid)) == run(persistent(), &z(form, pos, grid)), "form {form}, pos {pos}");
        }
    }
    let differ = run(served(), &z(0, 7, 3)) != run(served(), &z(1, 7, 3)) || run(served(), &z(0, 7, 3)) != run(served(), &z(2, 7, 3));
    assert!(differ, "the three forms never differ: the load-time choice would read nothing");
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    let cmd = z(1, 7, 3);
    let want = run(served(), &cmd);
    for (tag, from, to) in [
        ("rope_pair", "const unsigned short b = exact_rms_apply(scale, __ldcg(src + i + 64u), alpha[i + 64u]);", "const unsigned short b = exact_rms_apply(scale, __ldcg(src + i + 63u), alpha[i + 64u]);"),
        ("rope_form_swap", "        *o1 = pers_f16_fma_exact(a, c, (unsigned short)(bs ^ 0x8000u), false);", "        *o1 = pers_f16_fma_exact(b, s, ac, true);"),
        ("kv_pos", "kc + ((unsigned long long)h * cap + pos) * 128u, lane);", "kc + ((unsigned long long)h * cap + pos + 1u) * 128u, lane);"),
        ("qk_norm_tree", "for (int mask = 16; mask > 0; mask >>= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));\n    const float scale = rsqrtf", "for (int mask = 1; mask < 32; mask <<= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));\n    const float scale = rsqrtf"),
    ] {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &cmd) != Some(want.clone()), "norm/rope mutant {tag} survived");
    }
}

// ---- the L2 bench arms ------------------------------------------------------

/// The served pattern on two layers (a Tetra down, then an int4-pattern one)
/// at the real widths, rows cut to `rows` a site so the host run stays short.
fn short_schedule(rows: u32) -> Vec<pers::Site> {
    let mut s = pers::schedule_for(2, 2560, 9728, |l| l == 0);
    for site in &mut s {
        site.d_out = rows;
    }
    pers::validate(&s).expect("valid");
    s
}

struct BenchOut {
    y: Vec<u16>,
    f32s: Vec<u8>,
    norm: Vec<u16>,
    rot: Vec<u8>,
    bar: u32,
    abort: u32,
}

fn bench_cmd(arm: usize, grid: u32, threads: u32, launches: u32, seed: u64, sites: &[pers::Site]) -> (Vec<u8>, u32) {
    let (h, i) = (2560usize, 9728usize);
    let (rh, ri) = (rot(h, 21), rot(i, 22));
    let mut c = Cmd::new(b'B').u32(arm as u32).u32(grid).u32(threads).u32(launches).u64(seed).u32(sites.len() as u32);
    for st in sites {
        c = c.u32s(&pers::site_record(st, 0, 0, 0));
    }
    let (lo, hi) = pers::slice_tables(sites, grid);
    c = c.u32s(&lo).u32s(&hi);
    c = c.u32(h as u32).u32(i as u32).u32(rh.m).u32(rh.k).u32(ri.m).u32(ri.k).f32(rh.inv).f32(ri.inv).f32(1e-6);
    c = c.u32s(&rh.signs).u32s(&ri.signs).f32s(&rh.small).f32s(&ri.small);
    // The ledger the bench keeps, launch by launch.
    let mut led = pers::Ledger::new(pers::N_PERS_ARMS);
    let a = &pers::PERS_ARMS[arm];
    for _ in 0..launches {
        let lc = led.before(arm, sites.len()).unwrap();
        c = c.u32(lc.bar_base).u32(lc.epoch).u32(lc.cnt_offset);
        led.after(arm, pers::n_bar(sites, a.waits), grid, a.waits > 0);
    }
    let final_bar = led.before(arm, sites.len()).unwrap().bar_base;
    (c.0, final_bar)
}

fn parse_bench(b: &[u8], sites: &[pers::Site]) -> BenchOut {
    let total: usize = sites.iter().map(|s| s.d_out as usize).sum();
    let (y, rest) = b.split_at(2 * total);
    let (f32s, rest) = rest.split_at(4 * total);
    // norm and rot sizes follow from the restage rule of host_pers_bench.inc
    let mut n_norm = 0usize;
    let mut n_prol = 0usize;
    for (k, s) in sites.iter().enumerate() {
        let restage = k == 0 || sites[k - 1].d_in != s.d_in || sites[k - 1].layer != s.layer;
        if restage && s.prologue != pers::Prologue::None {
            n_prol += 1;
            n_norm += usize::from(s.prologue == pers::Prologue::NormRot);
        }
    }
    let (norm, rest) = rest.split_at(2 * n_norm * 2560);
    let (rot, tail) = rest.split_at(4 * n_prol * 9728);
    let t = u32s(tail);
    BenchOut { y: u16s(y), f32s: f32s.to_vec(), norm: u16s(norm), rot: rot.to_vec(), bar: t[0], abort: t[1] }
}

/// Every arm on the short schedule, three launches, a grid of 3: the owed
/// arms give the served rows bit for bit, no arm leaves a row unwritten, the
/// chained arms' barrier counter ends where the ledger says, nothing aborts,
/// and pchain_wp's prologue vectors are candle's norm and rot_apply's
/// rotation.
#[test]
fn every_bench_arm_runs_the_served_rows() {
    let sites = short_schedule(24);
    let (cmd0, _) = bench_cmd(0, 3, 64, 1, 0xbe7c, &sites);
    let want = parse_bench(&run(served(), &cmd0), &sites);
    assert!(want.y.iter().all(|&v| v != pers::POISON_H));
    for (k, a) in pers::PERS_ARMS.iter().enumerate() {
        let (cmd, final_bar) = bench_cmd(k, 3, 64, 3, 0xbe7c, &sites);
        let got = parse_bench(&run(persistent(), &cmd), &sites);
        assert_eq!(got.abort, 0, "{}: a launch aborted", a.name);
        assert!(got.y.iter().all(|&v| v != pers::POISON_H), "{}: a row was never written", a.name);
        if a.owes_identity() {
            assert!(got.y == want.y, "{}: rows differ from tv_tetra48_h", a.name);
            assert!(got.f32s == want.f32s, "{}: the f32 values before f2h differ", a.name);
        }
        if a.waits > 0 {
            assert_eq!(got.bar, final_bar, "{}: the barrier counter left the ledger", a.name);
        }
        if a.prologue {
            assert!(got.norm == want.norm, "{}: the norm prologue is not candle's", a.name);
            assert!(got.rot == want.rot, "{}: the rotation prologue is not rot_apply", a.name);
        }
    }
}

/// The work-claiming race of the draft: without the barrier before the
/// cursor is set again, a warp still claiming site s draws indices of site
/// s + 1 (same staging, no wait between q and k). The scheduler's drift finds
/// it within a few seeds; the fixed text never does.
#[test]
fn the_claiming_race_is_closed() {
    let sites = short_schedule(24);
    let arm = pers::PERS_ARMS.iter().position(|a| a.name == "psmem_t28").unwrap();
    // The draft's text: the cursor set again with no barrier before it, and
    // no barrier at the end of the site either.
    let b = mutant_many(
        "claim_race",
        &[
            ("llvq-cuda/kernels/llvq_pers.cuh", "    __syncthreads();  // every warp has left the previous claim loop\n", ""),
            ("llvq-cuda/kernels/tetra48_pers.cu", "        __syncthreads();  // the CTA's rows of site s precede its signal and the next reset\n", ""),
        ],
    );
    let mut caught = false;
    for seed in 1..=6u64 {
        // the seed drives both the fixture and the scheduler
        let (cmd, _) = bench_cmd(arm, 3, 96, 1, seed, &sites);
        let want = parse_bench(&run(served(), &cmd), &sites).y;
        let got = parse_bench(&run(persistent(), &cmd), &sites).y;
        let bad: Vec<(usize, u16, u16)> = got.iter().zip(&want).enumerate().filter(|(_, (a, b))| a != b).map(|(i, (a, b))| (i, *a, *b)).take(8).collect();
        assert!(bad.is_empty(), "seed {seed}: the fixed text differs at (row, got, want) {bad:?}");
        caught |= parse_bench(&run(&b, &cmd), &sites).y != want;
    }
    assert!(caught, "the claiming race never showed: the test has no teeth");
}

/// One arm at its real block size and a full layer's rows: the template at
/// 768 threads, every row of every site of one layer.
#[test]
fn one_arm_at_full_size() {
    let mut sites = pers::schedule_for(1, 2560, 9728, |_| true);
    // o reads 4,096 columns; keep it, cut nothing
    for s in &mut sites {
        s.d_out = s.d_out.min(4096);
    }
    let arm = pers::PERS_ARMS.iter().position(|a| a.name == "pchain_wp").unwrap();
    let (cmd, final_bar) = bench_cmd(arm, 2, 768, 1, 77, &sites);
    let want = parse_bench(&run(served(), &cmd), &sites);
    let got = parse_bench(&run(persistent(), &cmd), &sites);
    assert_eq!(got.abort, 0);
    assert_eq!(got.bar, final_bar);
    assert!(got.y == want.y && got.f32s == want.f32s, "rows differ at full size");
    assert!(got.norm == want.norm && got.rot == want.rot, "prologues differ at full size");
}

// ---- route P, step A ------------------------------------------------------

fn seg_cmd(grid: u32, threads: u32, normv: u32, tetra_down: bool, seed: u64, bar_base: u32) -> Vec<u8> {
    let (h, i) = (2560u32, 9728u32);
    let (rh, ri) = (rot(h as usize, 31), rot(i as usize, 32));
    let gu: Vec<u32> = pers::slices(i, grid, 1).iter().flat_map(|&(a, b)| [a, b]).collect();
    let dn: Vec<u32> = pers::slices(h, grid, 1).iter().flat_map(|&(a, b)| [a, b]).collect();
    Cmd::new(b'S')
        .u32(grid)
        .u32(threads)
        .u32(normv)
        .u32(u32::from(tetra_down))
        .u64(seed)
        .u32(h)
        .u32(i)
        .u32(rh.m)
        .u32(rh.k)
        .u32(ri.m)
        .u32(ri.k)
        .f32(rh.inv)
        .f32(ri.inv)
        .f32(1e-6)
        .u32(bar_base)
        .u32s(&rh.signs)
        .u32s(&ri.signs)
        .f32s(&rh.small)
        .f32s(&ri.small)
        .u32s(&gu)
        .u32s(&dn)
        .0
}

/// The MLP segment in one launch equals the eager chain of eight (Tetra down)
/// or six (int4 down) launches, for both norm variants, and its one barrier
/// moves the counter by one grid.
#[test]
fn the_step_a_segment_is_the_eager_chain() {
    for tetra_down in [true, false] {
        for normv in [0u32, 1] {
            let bar_base = 0xffff_fff0u32; // across the wrap
            let cmd = seg_cmd(3, 128, normv, tetra_down, 0x5e9 + normv as u64, bar_base);
            let want = run(served(), &cmd);
            let got = run(persistent(), &cmd);
            let n = 2 * (2560 + 9728);
            assert!(want[..n] == got[..n], "segment differs (tetra_down {tetra_down}, variant {normv})");
            let t = u32s(&got[n..]);
            assert_eq!(t[1], 0, "the segment aborted");
            assert_eq!(t[0], bar_base.wrapping_add(3), "one barrier, one arrival a CTA");
        }
    }
}

#[test]
fn step_a_mutants_are_killed() {
    let cmd = seg_cmd(3, 96, 0, true, 0x5ea, 0);
    let cmd_q4 = seg_cmd(3, 96, 0, false, 0x5eb, 0);
    let want = run(served(), &cmd);
    let want_q4 = run(served(), &cmd_q4);
    let f = "llvq-cuda/kernels/pers_segment.cu";
    let cases = [
        ("seg_no_barrier", "if (!pers_grid_wait(A->bar, L.bar_base + gridDim.x, A->abort_flag)) return;", "(void)0;"),
        ("seg_barrier_target", "if (!pers_grid_wait(A->bar, L.bar_base + gridDim.x, A->abort_flag)) return;", "if (!pers_grid_wait(A->bar, L.bar_base + gridDim.x + 1u, A->abort_flag)) return;"),
        ("seg_residual_order", "L.out[r] = exact_hadd(__ldcg(L.x + r), f2h(y));", "L.out[r] = f2h(__fadd_rn(h2f(__ldcg(L.x + r)), y));"),
        ("seg_silu_operand", "A->act[r] = exact_silu_mul(A->silu, g, u);", "A->act[r] = exact_silu_mul(A->silu, u, g);"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = run(&b, &cmd) != want || run(&b, &cmd_q4) != want_q4;
        assert!(killed, "segment mutant {tag} survived");
    }
}

/// Grids of 2 to 5 CTAs give the same bits: every interleaving the harness
/// tries is the served chain.
#[test]
fn the_segment_does_not_depend_on_the_grid() {
    let want = run(served(), &seg_cmd(4, 64, 0, false, 1, 0));
    let n = 2 * (2560 + 9728);
    for grid in 2..=5u32 {
        assert!(run(persistent(), &seg_cmd(grid, 64, 0, false, 1, 0))[..n] == want[..n], "grid {grid}");
    }
}
