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
        ("norm_butterfly_up", "for (int mask = 16; mask > 0; mask >>= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));", "for (int mask = 1; mask < 32; mask <<= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));"),
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
    let b = mutant("q4_lane_order", "llvq-cuda/kernels/llvq_pers.cuh", "for (u32 wi = lane; wi < nwords; wi += 32u) {\n        const u32 p = wq[w0 + wi];", "for (u32 wi = nwords - 1u - lane; wi < nwords; wi -= 32u) {\n        const u32 p = wq[w0 + wi];");
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
        ("seg_no_barrier", "if (!pers_grid_wait(A->bar, A->bar_base + gridDim.x, A->abort_flag)) return;", "(void)0;"),
        ("seg_barrier_target", "if (!pers_grid_wait(A->bar, A->bar_base + gridDim.x, A->abort_flag)) return;", "if (!pers_grid_wait(A->bar, A->bar_base + gridDim.x + 1u, A->abort_flag)) return;"),
        ("seg_residual_order", "A->out[r] = exact_hadd(A->x[r], f2h(y));", "A->out[r] = f2h(__fadd_rn(h2f(A->x[r]), y));"),
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
