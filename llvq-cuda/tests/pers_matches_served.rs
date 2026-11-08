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
    build_side_with(root, side, contract, tag, &[])
}

/// [`build_side`] with extra `-D` defines (the 4E row form of the module).
fn build_side_with(root: &Path, side: u32, contract: &str, tag: &str, defines: &[&str]) -> PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_pers_{tag}_{side}_{contract}_{}", std::process::id()));
    let res = Command::new("clang++")
        .args(["-std=c++17", "-O2", &format!("-ffp-contract={contract}"), "-Wall", "-Wextra", "-Werror"])
        .args(defines.iter().map(|d| format!("-D{d}")))
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

/// The persistent side with 4E's row form 3 (4E v5: form 2 with each CTA's
/// last rows split over its warps).
fn persistent_v5() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side_with(&repo(), 1, "off", "pers_v5", &["PERS_4E_ROWS=3u"]))
}

/// The persistent side with 4E's row form 2 (4E v4: v2's row loop on the
/// stream's static rows).
fn persistent_v4() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side_with(&repo(), 1, "off", "pers_v4", &["PERS_4E_ROWS=2u"]))
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
        ("norm_square_rounded", "[&](u32 col) { return h2f(__ldcg(x + col)); }", "[&](u32 col) { return __fmul_rn(h2f(__ldcg(x + col)), 1.0009765625f); }"),
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
    let b = mutant("q4_lane_order", "llvq-cuda/kernels/llvq_pers.cuh", "for (u32 wi = lane; wi < nwords; wi += 32u) {\n            const u32 p = wq[w0 + wi];\n            const u32 c = wi << 3;\n            const u32 g", "for (u32 wi = nwords - 1u - lane; wi < nwords; wi -= 32u) {\n            const u32 p = wq[w0 + wi];\n            const u32 c = wi << 3;\n            const u32 g");
    assert!(run(&b, &cmd) != want, "a reversed int4 lane order survived");
    // the PREFETCH form (odd rows of the command): its next word's group, and
    // the hand-over of the word in flight
    let b = mutant("q4_pf_group", "llvq-cuda/kernels/llvq_pers.cuh", "sb2 = scales[s0 + ((wi + 32u) >> 4)];", "sb2 = scales[s0 + ((wi + 32u) >> 5)];");
    assert!(run(&b, &cmd) != want, "a wrong group of the prefetched word survived");
    let b = mutant("q4_pf_stale", "llvq-cuda/kernels/llvq_pers.cuh", "            p = p2;\n            sb = sb2;", "            (void)p2;\n            sb = sb2;");
    assert!(run(&b, &cmd) != want, "a prefetched word never handed over survived");
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
    want.push(pers::P4E_QKV_BYTES as u32);
    want.extend([40, 72, 80, 104, 112, 120, 128, 144]);
    want.push(pers::P4E_LAYER_BYTES as u32);
    want.extend([24, 32, pers::P4E_LAYER_QKV as u32, 192, 240, 248, 252, pers::P4E_LAYER_MLP as u32, pers::P4E_LAYER_OT as u32]);
    want.push(pers::P4E_ARGS_BYTES as u32);
    // invnorm, attn_ticket, n_layers, cap, eps, c, head_w, head_b
    want.extend([160, 168, 176, 204, 208, 212, pers::P4E_ARGS_HEAD as u32, pers::P4E_ARGS_HEAD as u32 + 16]);
    want.push(pers::P4E_OT_BYTES as u32);
    // t, signs, small, m, k, inv
    want.extend([0, 48, 56, 64, 68, 72]);
    want.push(pers::P4E_LAUNCH_BYTES as u32);
    want.extend([16, 24, 32]);
    assert_eq!(o, want);
    assert_eq!(std::mem::size_of::<pers::P4eLaunch>(), pers::P4E_LAUNCH_BYTES);
    let lay = pers::P4eLayerHost { o_gpr: 32, tetra_down: true, qkv: pers::P4eQkvHost { d_in: 2560, out_v: 9, ..Default::default() }, ..Default::default() }.bytes();
    assert_eq!(lay.len(), pers::P4E_LAYER_BYTES);
    assert_eq!(u32::from_le_bytes(lay[248..252].try_into().unwrap()), 32);
    assert_eq!(u32::from_le_bytes(lay[252..256].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(lay[40 + 120..40 + 124].try_into().unwrap()), 2560);
    assert_eq!(u64::from_le_bytes(lay[40 + 144..40 + 152].try_into().unwrap()), 9);
    let lay = pers::P4eLayerHost { ot: 0xa7, ..Default::default() }.bytes();
    assert_eq!(u64::from_le_bytes(lay[pers::P4E_LAYER_OT..pers::P4E_LAYER_OT + 8].try_into().unwrap()), 0xa7);
    let ot = pers::P4eOTHost { t: pers::SegTetraHost { d_out: 4096, ..Default::default() }, small: 3, k: 19, inv: 0.5, ..Default::default() }.bytes();
    assert_eq!(ot.len(), pers::P4E_OT_BYTES);
    assert_eq!(u32::from_le_bytes(ot[44..48].try_into().unwrap()), 4096);
    assert_eq!(u64::from_le_bytes(ot[56..64].try_into().unwrap()), 3);
    assert_eq!(u32::from_le_bytes(ot[68..72].try_into().unwrap()), 19);
    assert_eq!(f32::from_le_bytes(ot[72..76].try_into().unwrap()), 0.5);
    let mut ptrs = [0u64; 22];
    ptrs[21] = 0x7e;
    let args = pers::P4eArgsHost { ptrs, cap: 8448, c: 0.25, head: [0x11, 0x22, 0x33], ..Default::default() }.bytes();
    assert_eq!(u64::from_le_bytes(args[216..224].try_into().unwrap()), 0x11);
    assert_eq!(u64::from_le_bytes(args[232..240].try_into().unwrap()), 0x33);
    assert_eq!(args.len(), pers::P4E_ARGS_BYTES);
    assert_eq!(u64::from_le_bytes(args[168..176].try_into().unwrap()), 0x7e);
    assert_eq!(u32::from_le_bytes(args[204..208].try_into().unwrap()), 8448);
    assert_eq!(f32::from_le_bytes(args[212..216].try_into().unwrap()), 0.25);
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
    // L1's kernel (variant 0) and its four variants (pers::PERSALL_VARIANTS)
    for variant in 0..pers::PERSALL_VARIANTS.len() as u32 {
        for grid in [1u32, 3, 7, 12] {
            let cmd = Cmd::new(b'A').u32(grid).u64(0xa11).u32(variant).0;
            assert!(run(served(), &cmd) == run(persistent(), &cmd), "variant {variant}, grid {grid}");
        }
    }
    let cmd = Cmd::new(b'A').u32(3).u64(0xa11).u32(0).0;
    let want = run(served(), &cmd);
    let b = mutant("persall_scan", "llvq-cuda/kernels/tetra48_persall.cu", "uniform too.\n        while (si + 1u < nsites && w >= (u32)(sites[(si + 1u) * LLVQ_TOCC_SITE_WORDS + 7u] & 0xffffffffull)) ++si;", "uniform too.\n        while (si + 1u < nsites && w > (u32)(sites[(si + 1u) * LLVQ_TOCC_SITE_WORDS + 7u] & 0xffffffffull)) ++si;");
    assert!(run_mutant(&b, &cmd) != Some(want), "an off-by-one site scan survived");
    // the variants' own loop: the prefetched words handed over, the tables
    // read from where they were staged
    let cmd = Cmd::new(b'A').u32(3).u64(0xa11).u32(3).0;
    let want = run(served(), &cmd);
    let f = "llvq-cuda/kernels/tetra48_persall.cu";
    for (tag, from, to) in [
        ("persall_pf_stale", "                    lo = lo2;\n                    hi16 = hi2;", "                    lo = lo2;\n                    (void)hi2;"),
        ("persall_ts_unstaged", "for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) s_br[i] = branches[i];", "for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) s_br[i] = branches[i ^ 1u];"),
        ("persall_pf_tile_restart", "        for (u32 t = 0; t < ntiles; ++t) {\n            const u32 jlo = t * TILE_BLOCKS;\n            const u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;\n            const u32 n = (jhi - jlo) * LLVQ_DIM;\n            __syncthreads();\n            for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];\n            __syncthreads();\n            for (; j < jhi;", "        for (u32 t = 0; t < ntiles; ++t) {\n            const u32 jlo = t * TILE_BLOCKS;\n            const u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;\n            const u32 n = (jhi - jlo) * LLVQ_DIM;\n            __syncthreads();\n            for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];\n            __syncthreads();\n            for (j = jlo + lane + (t > 0u ? 32u : 0u); j < jhi;"),
    ] {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &cmd) != Some(want.clone()), "persall mutant {tag} survived");
    }
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
    // the 512-thread path (the row stream) is the one these commands run
    for (tag, from, to) in [
        ("v_reads_rotated", "pers_q4_row_4e<true>(P.vw, P.vsc, P.vbi, xn, rv,", "pers_q4_row_4e<true>(P.vw, P.vsc, P.vbi, ((void)xn, xr), rv,"),
        ("k_offset", "PersTRow{P.kw + (r - P.q_dout) * P.k_stride,", "PersTRow{P.kw + (r - P.q_dout + 1u) * P.k_stride,"),
        ("stream_tail_half", "                    if (tail_w > 8u) t1 = tp[1];\n", ""),
        ("stream_lane_order", "                const u32 j = lane + 32u * i;\n                if (j < nblocks) acc = pers_block(", "                const u32 j = lane + 32u * (nit - 1u - i);\n                if (j < nblocks) acc = pers_block("),
        ("stream_slot", "            if (j2 < nblocks) f1r_load(w2, j2, lo[d], hi[d]);\n            if (++i2 == nit) {", "            if (j2 < nblocks) f1r_load(w2, j2, lo[(d + 1u) % D], hi[(d + 1u) % D]);\n            if (++i2 == nit) {"),
    ] {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &v) != Some(want.clone()), "q/k/v mutant {tag} survived");
    }
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
/// grids of 1 to 5 CTAs, of 2 to 16 warps (the 16-warp CTA holds every task
/// at once, smaller ones walk them); the split fold through the tickets,
/// which it leaves at zero.
#[test]
fn the_4e_attention_phase_is_attn_ref() {
    let lens = [(2u32, 4u32, 64u32, 1u32), (2, 4, 64, 5), (2, 4, 128, 64), (2, 4, 128, 65), (2, 4, 192, 150), (2, 5, 512, 256), (2, 5, 512, 257), (1, 4, 1408, 1300), (3, 5, 2304, 2049)];
    for &(n_kv, groups, cap, l) in &lens {
        for &(grid, threads) in &[(3u32, 64u32), (5, 96), (1, 512)] {
            let (cmd, want, state) = attn_cmd(n_kv, groups, cap, l, grid, threads, 0xa77 + l as u64);
            let got = run(persistent(), &cmd);
            let n = want.len() * 2;
            assert_eq!(u16s(&got[..n]), want, "n_kv {n_kv}, groups {groups}, l {l}, grid {grid}x{threads}");
            let t = u32s(&got[n..n + 8]);
            assert_eq!((t[0], t[1]), (0, 0), "every ticket back at zero");
            assert_eq!(u32s(&got[n + 8..]), state, "the f32 state before the division, l {l}");
            let port = run(served(), &cmd);
            assert_eq!(u16s(&port[..n]), want, "the C++ port of attn_ref (the 4E test's reference), l {l}");
        }
    }
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    let wanted = |n_kv: u32, groups: u32, cap: u32, l: u32| {
        let (cmd, want, state) = attn_cmd(n_kv, groups, cap, l, 3, 64, 0xa77);
        let mut w: Vec<u8> = want.iter().flat_map(|x| x.to_le_bytes()).collect();
        w.extend([0u32, 0].iter().flat_map(|x| x.to_le_bytes()));
        w.extend(state.iter().flat_map(|x| x.to_le_bytes()));
        (cmd, w)
    };
    let (cmd, want) = wanted(1, 4, 1408, 1300);
    let cases = [
        ("attn_unit_order", "for (u32 uu = 0; uu < PERS_UPS; ++uu) {\n                const float* row", "for (u32 uu = PERS_UPS; uu-- > 0u;) {\n                const float* row"),
        ("attn_carry_fused", "const float dn = __fadd_rn(__fmul_rn(d, carry), ts);", "const float dn = __fmaf_rn(d, carry, ts);"),
        ("attn_tile", "#define PERS_TT 8u", "#define PERS_TT 16u"),
        ("attn_no_fold", "atomicAdd(ticket + h, 1u) == ns - 1u ? 1u : 0u;", "atomicAdd(ticket + h, 1u) == ns ? 1u : 0u;"),
        ("attn_no_reset", "if (threadIdx.x == 0u) ticket[h] = 0u;", "(void)0;"),
        ("attn_stage_pos", "const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + (pl % PERS_TT);", "const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + ((pl + 1u) % PERS_TT);"),
        ("attn_step_tile_order", "const u32 sl = (u * tpu + t) * PERS_TT;", "const u32 sl = (u * tpu + (tpu - 1u - t)) * PERS_TT;"),
        ("attn_scale", "sc[j] = __fmul_rn(t, c);", "sc[j] = __fmul_rn(t, __fmul_rn(c, 0.5f));"),
        ("attn_no_syncwarp", "    __syncwarp();  // memory order: every lane's reads of the row precede lane 0's rewrite of m and d\n", ""),
        ("attn_split_chunk", "pers_attn_merge1(am, ad, aa, cmd[(j * groups + g) * 2u], cmd[(j * groups + g) * 2u + 1u], ba[j]);", "pers_attn_merge1(am, ad, aa, cmd[(j * groups + g) * 2u], cmd[(j * groups + g) * 2u + 1u], ba[(j + 1u) % PERS_ATTN_CH]);"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &cmd) != Some(want.clone()), "attention mutant {tag} survived");
    }
    // groups of 5: the half-warp past the last head must keep nothing
    let (cmd, want) = wanted(2, 5, 512, 257);
    let b = mutant("attn_spare_half_writes", f, "    if (own) {\n#pragma unroll\n        for (u32 i = 0; i < 8u; ++i) row[dl + i] = a[i];", "    if (true) {\n#pragma unroll\n        for (u32 i = 0; i < 8u; ++i) row[dl + i] = a[i];");
    assert!(run_mutant(&b, &cmd) != Some(want), "attention mutant attn_spare_half_writes survived");
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
        ("rope_pair", "const unsigned short b = exact_rms_apply(scale, hx[j + 2u], alpha[i + 64u]);", "const unsigned short b = exact_rms_apply(scale, hx[j ^ 1u], alpha[i + 64u]);"),
        ("rope_form_swap", "        *o1 = pers_f16_fma_exact(a, c, (unsigned short)(bs ^ 0x8000u), false);", "        *o1 = pers_f16_fma_exact(b, s, ac, true);"),
        ("kv_pos", "kc + ((unsigned long long)h * cap + pos) * 128u, lane);", "kc + ((unsigned long long)h * cap + pos + 1u) * 128u, lane);"),
        ("qk_norm_tree", "for (int mask = 16; mask > 0; mask >>= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));\n    const float scale = rsqrtf", "for (int mask = 1; mask < 32; mask <<= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));\n    const float scale = rsqrtf"),
    ] {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &cmd) != Some(want.clone()), "norm/rope mutant {tag} survived");
    }
}

#[allow(clippy::too_many_arguments)]
fn g_cmd(form: u32, grid: u32, threads: u32, n: u32, pos0: u32, cap: u32, tok0: u32, eos: u32, seed: u64) -> Vec<u8> {
    g_cmd_runs(form, grid, threads, &[(pos0, n, tok0)], cap, eos, seed)
}

/// Several launches through the same state, each `(pos0, n, tok0)`, the way
/// `Runtime4e::decode` is called launch after launch.
fn g_cmd_runs(form: u32, grid: u32, threads: u32, runs: &[(u32, u32, u32)], cap: u32, eos: u32, seed: u64) -> Vec<u8> {
    g_cmd_variant(form, grid, threads, runs, cap, eos, seed, 0)
}

/// [`g_cmd_runs`] on one variant of the persistent side: 0 the 768-thread
/// token symbols (rows without prefetch), 1 the 512-thread stamp symbols,
/// 2 the 512-thread token symbols (rows with prefetch), 3 the 768-thread
/// stamp symbols. A stamp variant's output carries the stamps after the tail.
#[allow(clippy::too_many_arguments)]
fn g_cmd_variant(form: u32, grid: u32, threads: u32, runs: &[(u32, u32, u32)], cap: u32, eos: u32, seed: u64, variant: u32) -> Vec<u8> {
    let (pos0, n, tok0) = runs[0];
    let mut c = Cmd::new(b'G').u32(form).u32(grid).u32(threads).u32(n).u32(pos0).u32(cap).u32(tok0).u32(eos).u32(1024).u64(seed);
    for (w, s) in [(2560usize, 51u64), (2560, 52), (9728, 53)] {
        let r = rot(w, s);
        c = c.u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small);
    }
    c = c.u32(runs.len() as u32 - 1);
    for &(p, k, t) in &runs[1..] {
        c = c.u32(p).u32(k).u32(t);
    }
    c = c.u32(variant);
    // + 16: the 8B's Tetra o (its rotation at the context's width) and an
    // untied head table
    if variant & 16 != 0 {
        let r = rot(4096, 54);
        c = c.u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small);
    }
    c.0
}

/// 4E's token loop: N tokens in one cooperative launch equal the served chain
/// run token by token (embedding, two layers of q/k/v, norm/RoPE/append,
/// attention, o, MLP with a Tetra then an int4 down, final norm, head,
/// argmax): the tokens, both layers' KV caches, the final residual. Each
/// RoPE form, both row variants (768 threads without the rows' prefetch, 512
/// with it); an end-of-sequence stops the loop on every CTA; the barrier
/// counter ends at (tokens made) x (2 + 6 x layers) x grid; positions past
/// 256 take attention's split fold, whose tickets end at zero.
#[test]
fn the_4e_token_loop_is_the_served_chain() {
    let n = 3u32;
    let split = |b: &[u8]| -> (Vec<u32>, Vec<u8>, Vec<u32>) {
        let ring = u32s(&b[..4 * n as usize]);
        let tail = u32s(&b[b.len() - 8..]);
        (ring, b[4 * n as usize..b.len() - 8].to_vec(), tail)
    };
    for (form, grid, threads, variant) in [(0u32, 3u32, 128u32, 0u32), (2, 4, 96, 2)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, n, 7)], 16, u32::MAX, 0x4e00 + form as u64, variant);
        let (want_ring, want_state, _) = split(&run(served(), &cmd));
        let (ring, state, tail) = split(&run(persistent(), &cmd));
        assert!(want_ring.iter().all(|&t| t < 1024), "the served run made {n} tokens: {want_ring:?}");
        assert_eq!(ring, want_ring, "form {form}, grid {grid}: tokens");
        assert!(state == want_state, "form {form}, grid {grid}: residual or KV caches differ");
        assert_eq!(tail, vec![n * llvq_cuda::pers::n_bar_4e(2) * grid, 0], "barrier count and abort");
        // an end-of-sequence at the second token stops every CTA after it
        let cmd = g_cmd_variant(form, grid, threads, &[(5, n, 7)], 16, want_ring[1], 0x4e00 + form as u64, variant);
        let (ring, _, tail) = split(&run(persistent(), &cmd));
        assert_eq!(ring, vec![want_ring[0], want_ring[1], u32::MAX], "EOS");
        assert_eq!(tail[0], 2 * llvq_cuda::pers::n_bar_4e(2) * grid, "two tokens' barriers");
    }
    // Past 256 positions: two splits, the tickets' fold, both variants.
    for (variant, threads) in [(2u32, 128u32), (0, 64)] {
        let cmd = g_cmd_variant(1, 3, threads, &[(270, 2, 7)], 288, u32::MAX, 0x4e30, variant);
        let want = run(served(), &cmd);
        let got = run(persistent(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "positions 270-271, variant {variant}: tokens, residual or caches");
        assert_eq!(u32s(&got[got.len() - 8..]), vec![2 * llvq_cuda::pers::n_bar_4e(2) * 3, 0], "barriers, abort and tickets");
    }
    let f = "llvq-cuda/kernels/pers_4e.cu";
    let cmd = g_cmd(0, 3, 128, n, 5, 16, 7, u32::MAX, 0x4e00);
    let want = run(served(), &cmd)[..4 * n as usize].to_vec();
    // the segment's row stream (the 512-thread symbols): the deferred SiLU
    // lookup and the residual loaded at a down row's start
    let cmd512 = g_cmd_variant(2, 4, 96, &[(5, n, 7)], 16, u32::MAX, 0x4e02, 2);
    let want512 = run(served(), &cmd512)[..4 * n as usize].to_vec();
    let fs = "llvq-cuda/kernels/pers_segment.cu";
    for (tag, from, to) in [
        ("silu_deferred", "pend_sv = A->silu[g];", "pend_sv = A->silu[g ^ 1u];"),
        ("down_residual_row", "xr = __ldcg(L.x + dlo + warp + k * nw);", "xr = __ldcg(L.x + dlo + warp + k * nw + 1u);"),
        ("silu_last_pair", "        if (lane == 0u && pend_r != 0xffffffffu) A->act[pend_r] = exact_hmul(pend_sv, pend_u);\n", ""),
    ] {
        let b = mutant(tag, fs, from, to);
        let got = run_mutant(&b, &cmd512).map(|o| o[..4 * n as usize].to_vec());
        assert!(got != Some(want512.clone()), "segment stream mutant {tag} survived");
    }
    for (tag, from, to) in [
        ("4e_pos", "const u32 pos = L.pos0 + t;", "const u32 pos = L.pos0 + t + (t > 0u ? 1u : 0u);"),
        ("4e_ping_pong", "const SegLaunch S{A->x_b, A->x_a,", "const SegLaunch S{A->x_a, A->x_a,"),
        ("4e_attn_len", "A->cap, pos + 1u, A->c,", "A->cap, pos, A->c,"),
    ] {
        let b = mutant(tag, f, from, to);
        let got = run_mutant(&b, &cmd).map(|o| o[..4 * n as usize].to_vec());
        assert!(got != Some(want.clone()), "4E mutant {tag} survived");
    }
}

/// Consecutive decodes through one state, as `Runtime4e::decode` runs them
/// and as fusedrun's 4E mode times them: the same launch again (rewound to
/// the prompt), a launch that continues, then a rewind again. Every launch's
/// tokens, the caches, the residual and the barrier count equal the served
/// chain run the same way. The first card reading (job 6aebcb97) decoded its
/// first launch right and a rewound one wrong.
#[test]
fn consecutive_4e_decodes_keep_their_tokens() {
    let runs = [(5u32, 3u32, 7u32), (5, 3, 7), (8, 2, 9), (5, 3, 7)];
    let made: u32 = runs.iter().map(|r| r.1).sum();
    for (form, grid, threads, seed) in [(0u32, 3u32, 128u32, 0x4e10u64), (1, 4, 96, 0x4e11), (2, 3, 64, 0x4e12)] {
        let cmd = g_cmd_runs(form, grid, threads, &runs, 16, u32::MAX, seed);
        let want = run(served(), &cmd);
        let got = run(persistent(), &cmd);
        let rings = |b: &[u8]| u32s(&b[..4 * made as usize]);
        let (wr, gr) = (rings(&want), rings(&got));
        assert!(wr.iter().all(|&t| t < 1024), "the served runs made every token: {wr:?}");
        assert_eq!(&wr[..3], &wr[3..6], "the served chain rewound gives the same tokens");
        assert_eq!(gr, wr, "form {form}, grid {grid}: the tokens of consecutive launches");
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "form {form}, grid {grid}: residual or caches");
        let tail = u32s(&got[got.len() - 8..]);
        assert_eq!(tail, vec![made * llvq_cuda::pers::n_bar_4e(2) * grid, 0], "barrier count and abort");
    }
    // Teeth: without the clear at launch start, a rewound launch reads the
    // previous launch's key (form 1, grid 4 caught it before the fix).
    let b = mutant_many(
        "4e_slots_kept",
        &[("llvq-cuda/kernels/pers_4e.cu", "        A->slots[0] = 0ull;\n        A->slots[1] = 0ull;\n", "")],
    );
    let cmd = g_cmd_runs(1, 4, 96, &runs, 16, u32::MAX, 0x4e11);
    let want = u32s(&run(served(), &cmd)[..4 * made as usize]);
    let got = run_mutant(&b, &cmd).map(|o| u32s(&o[..4 * made as usize]));
    assert!(got != Some(want), "the slots-kept mutant survived");
}

/// 4E's phase-stamp symbols (`LLVQ_PERS_4E`'s stamped launch) make the
/// tokens, caches and residual of the token symbols, over two launches
/// through one state, and their stamps read as the barriers they bracket:
/// every CTA's stamps in event order, every release after the wait's last
/// arrival, the phases' walls summing to the token. The harness clock counts
/// calls, so this proves the order of the stamps, never a duration.
#[test]
fn the_4e_phase_stamps_keep_the_tokens_and_bracket_the_waits() {
    use llvq_cuda::pers::{p4e_events, p4e_phase_report};
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let made: usize = runs.iter().map(|r| r.1 as usize).sum();
    let ev = p4e_events(2) as usize;
    for (form, grid, threads, seed, variant) in [(0u32, 3u32, 128u32, 0x4e20u64, 1u32), (1, 4, 96, 0x4e21, 3)] {
        let plain = run(persistent(), &g_cmd_runs(form, grid, threads, &runs, 16, u32::MAX, seed));
        let cmd = g_cmd_variant(form, grid, threads, &runs, 16, u32::MAX, seed, variant);
        let want = run(served(), &cmd);
        assert!(want[..want.len() - 8] == plain[..plain.len() - 8], "form {form}, grid {grid}: the token symbols are the served chain");
        let got = run(persistent(), &cmd);
        let n_st = made * ev * grid as usize;
        assert_eq!(got.len(), plain.len() + 8 * n_st, "the stamps follow the tail");
        assert!(got[..plain.len()] == plain[..], "form {form}, grid {grid}: the stamped launch moved an output");
        let st: Vec<u64> = got[plain.len()..].chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
        let (first, second) = st.split_at(3 * ev * grid as usize);
        let r = p4e_phase_report(first, 3, 2, grid).expect("the first launch's stamps");
        assert_eq!(r.tokens_read, 2);
        assert_eq!(r.grain, 1, "the harness clock steps by one");
        let r = p4e_phase_report(second, 2, 2, grid).expect("the second launch's stamps");
        let walls: f64 = r.phases.iter().map(|p| p.wall.0).sum();
        assert_eq!(walls, r.token.0, "one token read: the phases' walls are the token");
        assert!(r.phases.iter().all(|p| p.crit.0 >= p.mean.0 && p.mean.0 > 0.0), "{r:?}");
        // the inner stamps: every part worked, and the parts are their phase
        assert!(r.subs.iter().all(|p| p.crit.0 >= p.mean.0 && p.mean.0 > 0.0), "{r:?}");
        let parts = |lo: usize, hi: usize| r.subs[lo..hi].iter().map(|p| p.mean.0).sum::<f64>();
        assert!((parts(0, 2) - r.phases[1].mean.0).abs() < 1e-9, "q/k/v parts");
        assert!((parts(2, 7) - r.phases[5].mean.0).abs() < 1e-9, "MLP parts");
    }
    // Teeth. A release left unstamped, and an arrival stamped after the wait
    // instead of before it: the report refuses both.
    let cmd = g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x4e21, 1);
    let plain_len = run(persistent(), &g_cmd_runs(1, 4, 96, &runs, 16, u32::MAX, 0x4e21)).len();
    let first_launch = |o: &[u8]| -> Vec<u64> {
        o[plain_len..plain_len + 8 * 3 * ev * 4].chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect()
    };
    for (tag, edits) in [
        ("4e_release_unstamped", vec![("llvq-cuda/kernels/pers_4e.cu", "        PERS_STAMP_AT((e) + 1u);", "        PERS_STAMP_AT((e) + 0u);")]),
        (
            "4e_arrival_after_wait",
            vec![
                ("llvq-cuda/kernels/pers_4e.cu", "            PERS_STAMP_AT(e);", "            (void)0;"),
                ("llvq-cuda/kernels/pers_4e.cu", "        PERS_STAMP_AT((e) + 1u);", "        PERS_STAMP_AT(e); PERS_STAMP_AT((e) + 1u);"),
            ],
        ),
    ] {
        let b = mutant_many(tag, &edits);
        let o = run_mutant(&b, &cmd).unwrap_or_else(|| panic!("{tag}: the mutant must still run"));
        let r = p4e_phase_report(&first_launch(&o), 3, 2, 4);
        assert!(r.is_err(), "the {tag} mutant survived: {r:?}");
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

/// The two decode arms fold the same values, one reading the tables through
/// L1, the other from its shared copy: their rows are equal, and a copy that
/// swaps two branch entries makes them differ (the card-side check of
/// persbench.rs rests on this).
#[test]
fn the_decode_arms_read_one_table() {
    let sites = short_schedule(24);
    let pos = |n: &str| pers::PERS_ARMS.iter().position(|a| a.name == n).unwrap();
    let (l1, sm, word) = (pos("pdec_l1"), pos("pdec_smem"), pos("pword"));
    let rows = |bin: &Path, arm: usize| parse_bench(&run(bin, &bench_cmd(arm, 3, 64, 1, 0xdec0, &sites).0), &sites).y;
    let a = rows(persistent(), l1);
    assert!(a.iter().all(|&v| v != pers::POISON_H), "pdec_l1 left a row unwritten");
    assert!(a == rows(persistent(), sm), "pdec_smem's rows differ from pdec_l1's");
    assert!(a != rows(persistent(), word), "the decode arms fold nothing beyond the words");
    let b = mutant("dec_tables", "llvq-cuda/kernels/tetra48_pers.cu", "t_br[i] = A->br[i];", "t_br[i] = A->br[i ^ 1u];");
    assert!(a != rows(&b, sm), "a swapped branch table survived: the test has no teeth");
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

// ---- the row-loop microbenchmark (pers_trows.cu) ---------------------------

// Every tail is 8 or 16 wide, as on the 4B: the stream reads the tail as
// 16-byte chunks (pers_tetra_stream).
const TROWS_SITES: [(u32, u32, u64); 5] = [(64, 2560, 0x7101), (48, 4096, 0x7102), (40, 9728, 0x7103), (16, 128, 0x7104), (23, 2560, 0x7105)];

fn trows_cmd(arm: u32, grid: u32, threads: u32, stamps: u32, sites: &[(u32, u32, u64)]) -> Vec<u8> {
    let mut c = Cmd::new(b'J').u32(arm).u32(grid).u32(threads).u32(stamps).u64(0x7e70).u32(sites.len() as u32);
    for &(o, i, s) in sites {
        c = c.u32(o).u32(i).u64(s);
    }
    c.0
}

/// The four arms that owe the bits (V2, V3, ILP2, V2U) give the served rows on
/// every site, grid and warp count, with and without stamps; the three
/// diagnostics run and move rows, so each changes what it says it changes.
#[test]
fn the_row_loop_arms_are_the_served_rows() {
    let want = run(served(), &trows_cmd(0, 1, 64, 0, &TROWS_SITES));
    let rows: u32 = TROWS_SITES.iter().map(|s| s.0).sum();
    assert_eq!(want.len(), 6 * rows as usize, "binary16 rows, then their f32 values");
    for arm in 0..4u32 {
        for &(grid, threads, stamps) in &[(3u32, 96u32, 0u32), (2, 64, 1), (4, 128, 1), (5, 32, 0)] {
            let got = run(persistent(), &trows_cmd(arm, grid, threads, stamps, &TROWS_SITES));
            assert!(got == want, "row-loop arm {arm}, grid {grid}x{threads}, stamps {stamps}: rows or their f32 values differ from tv_tetra48_h");
        }
    }
    for arm in 4..7u32 {
        let got = run(persistent(), &trows_cmd(arm, 3, 96, 1, &TROWS_SITES));
        assert_eq!(got.len(), want.len());
        assert!(got != want, "diagnostic arm {arm} gives the served rows: it changes nothing");
    }
}

/// K2 (one decode, two activations): its rows on each activation are the
/// served rows on that activation, and its rows on the first are the k = 1
/// arm's (V2) bit for bit, binary16 and f32 before f2h, at every grid.
#[test]
fn the_k2_arm_is_the_k1_arm_on_each_activation() {
    let want = run(served(), &trows_cmd(7, 1, 64, 0, &TROWS_SITES));
    let rows: usize = TROWS_SITES.iter().map(|s| s.0 as usize).sum();
    assert_eq!(want.len(), 12 * rows, "two activations: binary16 rows, then their f32 values");
    let k1 = run(persistent(), &trows_cmd(0, 3, 96, 0, &TROWS_SITES));
    let (h, f) = (2 * rows, 4 * rows);
    // served layout: [y(x), y(x2)] then [f(x), f(x2)]
    assert_eq!(&want[..h], &k1[..h], "the served rows on the first activation are V2's");
    assert_eq!(&want[2 * h..2 * h + f], &k1[h..], "and so are their f32 values");
    assert!(want[..h] != want[h..2 * h], "the two activations must give different rows");
    for &(grid, threads, stamps) in &[(3u32, 96u32, 0u32), (2, 64, 1), (4, 128, 1), (5, 32, 0)] {
        let got = run(persistent(), &trows_cmd(7, grid, threads, stamps, &TROWS_SITES));
        assert!(got == want, "K2, grid {grid}x{threads}, stamps {stamps}: rows differ from the k = 1 rows on their activation");
    }
}

#[test]
fn row_loop_mutants_are_killed() {
    let cmds: Vec<Vec<u8>> = (0..4u32).map(|arm| trows_cmd(arm, 3, 96, 0, &TROWS_SITES)).collect();
    let want = run(served(), &cmds[0]);
    let f = "llvq-cuda/kernels/pers_trows.cu";
    let cases = [
        ("ilp2_fold_order", "        acc = pers_fold(acc, a0, q0, h0, gscale, invnorm);\n        if (two) acc = pers_fold(acc, a1, q1, h1, gscale, invnorm);",
         "        if (two) acc = pers_fold(acc, a1, q1, h1, gscale, invnorm);\n        acc = pers_fold(acc, a0, q0, h0, gscale, invnorm);"),
        ("ilp2_second_x", "const float* x1 = two ? x0 + 32u * 24u : x0;", "const float* x1 = x0;"),
        // Removing the guard is an equivalent mutant: zero words decode to the
        // zero vector, whose fold leaves acc as it is. Tightening it is not.
        ("ilp2_guard", "if (two) acc = pers_fold(acc, a1, q1, h1, gscale, invnorm);", "if (j + 64u < nblocks) acc = pers_fold(acc, a1, q1, h1, gscale, invnorm);"),
        ("ilp2_prefetch", "if (j + 96u < nblocks) f1r_load(wrow, j + 96u, m1, g1);", "if (j + 64u < nblocks) f1r_load(wrow, j + 64u, m1, g1);"),
        // V2U's guard keeps reads inside the staged activation; for the bits it
        // is equivalent (zero words past the row), so the trip count is mutated.
        ("uniform_trip", "const u32 nit = (nblocks + 31u) >> 5;\n    float acc = 0.0f;\n    u32 lo = 0u, hi16 = 0u;", "const u32 nit = nblocks >> 5;\n    float acc = 0.0f;\n    u32 lo = 0u, hi16 = 0u;"),
        ("cta_slice", "hi = (u32)((unsigned long long)d_out * (c + 1u) / g);", "hi = (u32)((unsigned long long)d_out * (c + 1u) / g) - (c == 0u);"),
        ("warp_stride", "for (u32 r = lo + warp; r < hi; r += nw) {\n                const u32* wrow", "for (u32 r = lo + warp; r < hi; r += nw + (nw > 1u)) {\n                const u32* wrow"),
        ("v3_rows", "            pers_tetra_stream<PERS_STREAM_D>(pers_warp_rows(lo, hi, warp, nw), nb, tw, tab, t_inv, xs, lane, row_of,",
         "            pers_tetra_stream<PERS_STREAM_D>(pers_warp_rows(lo, hi, warp, nw) - (warp == 0u), nb, tw, tab, t_inv, xs, lane, row_of,"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = cmds.iter().any(|c| run_mutant(&b, c).is_none_or(|got| got != want));
        assert!(killed, "row-loop mutant {tag} survived");
    }
    // K2, against its own served output (two activations).
    let k2cmd = trows_cmd(7, 3, 96, 0, &TROWS_SITES);
    let k2want = run(served(), &k2cmd);
    let k2cases = [
        ("k2_second_fold", "acc2 = pers_fold(acc2, b, q, hi16, gscale, invnorm);", "acc2 = pers_fold(acc2, a, q, hi16, gscale, invnorm);"),
        ("k2_second_x", "const float* xb = xs2 + j * 24u;", "const float* xb = (xs2 == xs ? xs2 : xs) + j * 24u;"),
        ("k2_staging", "if (ARM == TROWS_K2) pers_stage_f32<24u, false>(x2, xs2, nb, tw);", "if (ARM == TROWS_K2) pers_stage_f32<24u, false>(x2 == x ? x2 : x, xs2, nb, tw);"),
        ("k2_second_tail", "const float v2 = pers_epilogue(acc2, rs[r], tail + (size_t)r * tw, xs2 + nb * 24u, tw);", "const float v2 = pers_epilogue(acc2, rs[r], tail + (size_t)r * tw, xs + nb * 24u, tw);"),
    ];
    for (tag, from, to) in k2cases {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &k2cmd).is_none_or(|got| got != k2want), "row-loop mutant {tag} survived");
    }
}


// ---- plan-300 step 1: K4's row-loop arms, K11's int4 rows, K3b's counters ----

/// K4's five arms (NC, NC256, CPA, TILE, BAL: arms 8 to 12) give the served
/// rows on every site, grid and warp count, with and without stamps. The
/// 9,728 site holds three full tiles and 21 blocks left, the 4,096 one tile
/// and 42, the 2,560 none, so TILE runs both of its planes.
#[test]
fn the_step1_row_loop_arms_are_the_served_rows() {
    let want = run(served(), &trows_cmd(0, 1, 64, 0, &TROWS_SITES));
    for arm in 8..13u32 {
        for &(grid, threads, stamps) in &[(3u32, 96u32, 0u32), (2, 64, 1), (4, 128, 1), (5, 32, 0)] {
            let got = run(persistent(), &trows_cmd(arm, grid, threads, stamps, &TROWS_SITES));
            assert!(got == want, "K4 arm {arm}, grid {grid}x{threads}, stamps {stamps}: rows or their f32 values differ from tv_tetra48_h");
        }
    }
}

#[test]
fn step1_row_loop_mutants_are_killed() {
    let f = "llvq-cuda/kernels/pers_trows.cu";
    let want = run(served(), &trows_cmd(0, 1, 64, 0, &TROWS_SITES));
    let cases: [(&str, u32, &str, &str); 11] = [
        ("nc_word", 8, "const u32 w1 = trows_ldw<LD>(row + w + 1u);", "const u32 w1 = trows_ldw<LD>(row + w);"),
        ("nc_prefetch", 9, "if (j + 32u < nblocks) trows_load<LD>(wrow, j + 32u, lo2, hi2);", "if (j + 64u < nblocks) trows_load<LD>(wrow, j + 32u, lo2, hi2);"),
        ("cpa_slot", 10, "f1r_load(ring + TROWS_CPA_SLOT * (i % 3u), lane, lo, hi16);", "f1r_load(ring + TROWS_CPA_SLOT * ((i + 1u) % 3u), lane, lo, hi16);"),
        ("cpa_chunk", 10, "if (c + 4u <= row_u32) trows_cp16(", "if (c + 8u <= row_u32) trows_cp16("),
        ("cpa_ahead", 10, "        trows_cpa_issue(ring, wrow, row_u32, i + 2u, nit, lane);\n", "        trows_cpa_issue(ring, wrow, row_u32, i + 3u, nit, lane);\n"),
        ("tile_hi_half", 11, "acc = pers_block(acc, lo[s], (u32)(H >> (16u * s)) & 0xffffu,", "acc = pers_block(acc, lo[s], (u32)(H >> (16u * (s ^ 1u))) & 0xffffu,"),
        ("tile_put_lane", 11, "drow[192u * t + 4u * l + s] = lo;", "drow[192u * t + 4u * s + l] = lo;"),
        ("tile_next", 11, "if (t + 1u < nt) {\n            const u32* nx", "if (t + 2u < nt) {\n            const u32* nx"),
        ("tile_rem_hi", 11, "const unsigned short* rhi = reinterpret_cast<const unsigned short*>(rlo + rem);", "const unsigned short* rhi = reinterpret_cast<const unsigned short*>(rlo + rem + 1u);"),
        // A change of the stride alone is equivalent here (relayout and read
        // share it); pers.rs pins it to the served stride at the 4B's widths.
        ("tile_put_rem_hi", 11, "h[2u * (192u * nt + rem) + rr] = (unsigned short)hi16;", "h[2u * (192u * nt + rem) + (rr ^ 1u)] = (unsigned short)hi16;"),
        ("bal_last_row", 12, "if (r >= hi) break;", "if (r + 1u >= hi) break;"),
    ];
    for (tag, arm, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = [(3u32, 96u32), (2, 64)].iter().any(|&(g, th)| run_mutant(&b, &trows_cmd(arm, g, th, 0, &TROWS_SITES)).is_none_or(|got| got != want));
        assert!(killed, "K4 mutant {tag} survived");
    }
}

/// The row lead's arms (2026-11-07): GB (v2 with a grid barrier after every
/// site) and GBC (the rows claimed across the grid) give the served rows on
/// every site, grid and warp count, with and without stamps; the harness
/// checks that every CTA arrived at the barrier after every site.
#[test]
fn the_rows_grid_arms_are_the_served_rows() {
    let want = run(served(), &trows_cmd(0, 1, 64, 0, &TROWS_SITES));
    for arm in [13u32, 14] {
        for &(grid, threads, stamps) in &[(3u32, 96u32, 0u32), (2, 64, 1), (4, 128, 1), (5, 32, 0), (1, 64, 0)] {
            let got = run(persistent(), &trows_cmd(arm, grid, threads, stamps, &TROWS_SITES));
            assert!(got == want, "row-lead arm {arm}, grid {grid}x{threads}, stamps {stamps}: rows or their f32 values differ from tv_tetra48_h");
        }
    }
}

#[test]
fn rows_grid_mutants_are_killed() {
    let f = "llvq-cuda/kernels/pers_trows.cu";
    let want = run(served(), &trows_cmd(0, 1, 64, 0, &TROWS_SITES));
    // GB's row choice has no mutant: without its branch it falls to the
    // uniform loop, which owes the same bits (an equivalent mutant).
    let cases: [(&str, u32, &str, &str); 5] = [
        ("gbc_first", 14, "if (lane == 0u) r = atomicAdd(cur, 1u);", "if (lane == 0u) r = atomicAdd(cur, 1u) + 1u;"),
        ("gbc_next", 14, "if (lane == 0u) rn = atomicAdd(cur, 1u);", "if (lane == 0u) rn = atomicAdd(cur, 2u);"),
        ("gbc_bound", 14, "while (r < d_out) {", "while (r + 1u < d_out) {"),
        ("gbc_site_cursor", 14, "reinterpret_cast<u32*>(const_cast<float*>(x2)) + 2u + s;", "reinterpret_cast<u32*>(const_cast<float*>(x2)) + 2u;"),
        ("gb_barrier", 13, "if (!pers_grid_wait(ctr, (s + 1u) * g, ctr + 1)) return;", "(void)ctr;"),
    ];
    for (tag, arm, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = [(3u32, 96u32), (2, 64)].iter().any(|&(g, th)| run_mutant(&b, &trows_cmd(arm, g, th, 0, &TROWS_SITES)).is_none_or(|got| got != want));
        assert!(killed, "row-lead mutant {tag} survived");
    }
}

// Int4 sites of the 4B's three shapes, cut to a few rows; d_in in whole
// steps of 32 words, d_out a multiple of 8 (tv_q4_hh's blocks). 9,728: 9
// full tiles of 128 words and 2 steps left.
const Q4_SITES: [(u32, u32, u64); 4] = [(16, 2560, 0x7401), (24, 4096, 0x7402), (8, 9728, 0x7403), (40, 2560, 0x7404)];

fn q4rows_cmd(arm: u32, grid: u32, threads: u32, stamps: u32, sites: &[(u32, u32, u64)]) -> Vec<u8> {
    let mut c = Cmd::new(b'y').u32(arm).u32(grid).u32(threads).u32(stamps).u64(0x7e74).u32(sites.len() as u32);
    for &(o, i, s) in sites {
        c = c.u32(o).u32(i).u64(s);
    }
    c.0
}

/// K11: 4E's int4 row (Q4), its read-only loads (NC) and its tile layout
/// (TILE) give tv_q4_hh's rows bit for bit, binary16 and f32 before f2h, on
/// every grid; READ, the diagnostic, owes nothing and moves the rows.
#[test]
fn the_int4_rows_alone_are_the_served_rows() {
    let want = run(served(), &q4rows_cmd(0, 1, 64, 0, &Q4_SITES));
    let rows: usize = Q4_SITES.iter().map(|s| s.0 as usize).sum();
    assert_eq!(want.len(), 6 * rows);
    for arm in 0..3u32 {
        for &(grid, threads, stamps) in &[(3u32, 96u32, 0u32), (2, 64, 1), (5, 128, 1), (4, 32, 0)] {
            let got = run(persistent(), &q4rows_cmd(arm, grid, threads, stamps, &Q4_SITES));
            assert!(got == want, "int4 arm {arm}, grid {grid}x{threads}, stamps {stamps}: rows differ from tv_q4_hh");
        }
    }
    let got = run(persistent(), &q4rows_cmd(3, 3, 96, 1, &Q4_SITES));
    assert_eq!(got.len(), want.len());
    assert!(got != want, "READ gives the served rows: it computes something");
}

#[test]
fn int4_rows_mutants_are_killed() {
    let f = "llvq-cuda/kernels/pers_q4rows.cu";
    let want = run(served(), &q4rows_cmd(0, 1, 64, 0, &Q4_SITES));
    let cases: [(&str, u32, &str, &str); 6] = [
        ("q4nc_next", 1, "p2 = trows_ldw<LD>(wq + w0 + wi + 32u);", "p2 = trows_ldw<LD>(wq + w0 + wi + 64u);"),
        ("q4nc_scale", 1, "sb2 = scales[s0 + ((wi + 32u) >> 4)];", "sb2 = scales[s0 + (wi >> 4)];"),
        ("q4t_index", 2, "return (t << 7) + 4u * (k & 31u) + (k >> 5);", "return (t << 7) + 4u * (k >> 5) + (k & 31u);"),
        ("q4t_next", 2, "if (t + 1u < nt) P2 = ", "if (t + 2u < nt) P2 = "),
        ("q4t_group", 2, "const u32 c = ((t << 7) + 32u * s + lane) << 3;", "const u32 c = ((t << 7) + 32u * s + (lane ^ 16u)) << 3;"),
        ("q4t_rest", 2, "for (u32 wi = (nt << 7) + lane; wi < nwords; wi += 32u) {", "for (u32 wi = (nt << 7) + lane + 32u * (nt > 0u); wi < nwords; wi += 32u) {"),
    ];
    for (tag, arm, from, to) in cases {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &q4rows_cmd(arm, 3, 96, 0, &Q4_SITES)).is_none_or(|got| got != want), "int4 mutant {tag} survived");
    }
}

fn ctr_cmd(arm: u32, grid: u32, threads: u32, n: u32, work: u32) -> Vec<u8> {
    Cmd::new(b'c').u32(arm).u32(grid).u32(threads).u32(n).u32(work).0
}

/// K3b: every counter kernel runs its hops to the end on the host grid, at
/// grids with and without a short last group, over two launches whose second
/// starts from the first's counters (the epoch). A hop that moved a target
/// would hang and raise the abort flag; a lost arrival would leave a count.
#[test]
fn the_counter_kernels_count_every_hop() {
    for (k, &(name, _)) in pers::CTR_ARMS.iter().enumerate() {
        for &(grid, threads) in &[(12u32, 64u32), (5, 32), (17, 96)] {
            let n = 4u32;
            let got = u32s(&run(persistent(), &ctr_cmd(k as u32, grid, threads, n, 6)));
            let want: Vec<u32> = pers::ctr_expected(name, grid, n).iter().map(|v| 2 * v).collect();
            assert_eq!(got[..got.len() - 1], want[..], "{name}, grid {grid}x{threads}: counts after two launches");
            assert_eq!(got[got.len() - 1], 0, "{name}, grid {grid}x{threads}: abort flag");
        }
    }
}

#[test]
fn counter_mutants_are_killed() {
    let f = "llvq-cuda/kernels/pers_ctr.cu";
    let cases: [(&str, u32, &str, &str); 4] = [
        ("ctr8_signal", 3, "ctr_red_release(ctr + CTR_LINE * p);", "ctr_red_release(ctr + CTR_LINE * q);"),
        ("rel_target", 2, "            ctr_red_release(ctr);\n            ok = ctr_spin_tight(ctr, (base + i + 1u) * gridDim.x, abort_flag);", "            ctr_red_release(ctr);\n            ok = ctr_spin_tight(ctr, (base + i + 2u) * gridDim.x, abort_flag);"),
        ("tight_arrival", 1, "            __threadfence();\n            atomicAdd(ctr, 1u);\n            ok = ctr_spin_tight", "            __threadfence();\n            ok = ctr_spin_tight"),
        ("split_target", 4, "if (threadIdx.x == 0u) ok = ctr_spin_tight(ctr, (base + i + 1u) * gridDim.x, abort_flag);", "if (threadIdx.x == 0u) ok = ctr_spin_tight(ctr, (base + i + 2u) * gridDim.x, abort_flag);"),
    ];
    for (tag, arm, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let name = pers::CTR_ARMS[arm as usize].0;
        let want: Vec<u32> = pers::ctr_expected(name, 12, 4).iter().map(|v| 2 * v).chain([0]).collect();
        let killed = run_mutant(&b, &ctr_cmd(arm, 12, 64, 4, 6)).is_none_or(|o| u32s(&o) != want);
        assert!(killed, "counter mutant {tag} survived");
    }
}


// ---- 4E v4: v2's row loop inside v3's structure (PERS_4E_ROWS = 2) ---------

/// The module built with row form 2 runs the served chain bit for bit: the
/// q/k/v phase, the token loop (both RoPE forms read, positions past 256,
/// EOS), and the stamped symbol's tokens and inner stamps.
#[test]
fn the_4e_v4_rows_are_the_served_chain() {
    let r = rot(2560, 41);
    let qkv = |grid: u32, threads: u32, seed: u64| {
        Cmd::new(b'V').u32(grid).u32(threads).u64(seed).u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small).0
    };
    for (grid, threads, seed) in [(3u32, 96u32, 1u64), (4, 64, 2)] {
        assert!(run(served(), &qkv(grid, threads, seed)) == run(persistent_v4(), &qkv(grid, threads, seed)), "v4 q/k/v, grid {grid}");
    }
    let n = 3u32;
    for (form, grid, threads) in [(0u32, 3u32, 128u32), (2, 4, 96)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, n, 7)], 16, u32::MAX, 0x4e40 + form as u64, 2);
        let want = run(served(), &cmd);
        let got = run(persistent_v4(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v4, form {form}, grid {grid}: tokens, residual or caches");
    }
    let cmd = g_cmd_variant(1, 3, 128, &[(270, 2, 7)], 288, u32::MAX, 0x4e50, 2);
    let (want, got) = (run(served(), &cmd), run(persistent_v4(), &cmd));
    assert!(got[..got.len() - 8] == want[..want.len() - 8], "v4, positions 270-271");
    // the stamp symbol of v4: the same outputs as the token symbol, stamps that read
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let plain = run(persistent_v4(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x4e21, 2));
    let stamped = run(persistent_v4(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x4e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v4: the stamped launch moved an output");
    let ev = pers::p4e_events(2) as usize;
    let st: Vec<u64> = stamped[plain.len()..].chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
    let r = pers::p4e_phase_report(&st[..3 * ev * 4], 3, 2, 4).expect("v4's stamps");
    assert!(r.subs.iter().all(|p| p.mean.0 > 0.0), "{r:?}");
}

/// v4's driver: each mutant moves the q/k/v rows or the token loop.
#[test]
fn the_4e_v4_mutants_are_killed() {
    let r = rot(2560, 41);
    let v = Cmd::new(b'V').u32(3).u32(96).u64(1).u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small).0;
    let g = g_cmd_variant(0, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x4e40, 2);
    let (want_v, want_g) = (run(served(), &v), run(served(), &g));
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    for (tag, from, to) in [
        ("v4_rscale", "            rs = R.rscale[R.r];\n            start(k);", "            rs = R.rscale[R.r ^ 1u];\n            start(k);"),
        ("v4_tail", "epi(k, pers_epilogue(a, rs, R.tail + (unsigned long long)R.r * tail_w,", "epi(k, pers_epilogue(a, rs, R.tail + (unsigned long long)(R.r + 1u) * tail_w,"),
        ("v4_start", "            rs = R.rscale[R.r];\n            start(k);", "            rs = R.rscale[R.r];\n            (void)start;"),
        ("v4_row_order", "    for (u32 k = 0; k < nrows; ++k) {\n        const PersTRow R = row_of(k);", "    for (u32 k = nrows; k-- > 0;) {\n        const PersTRow R = row_of(k);"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to); // copies the tree and applies the edit exactly once
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v4"), &["PERS_4E_ROWS=2u"]);
        let killed = run_mutant(&b, &v).is_none_or(|o| o != want_v) || run_mutant(&b, &g).is_none_or(|o| o != want_g);
        assert!(killed, "v4 mutant {tag} survived");
    }
}


// ---- 4E v5: the last rows of a CTA split over its warps (PERS_4E_ROWS = 3) --

/// The module built with row form 3 runs the served chain bit for bit, on
/// grids whose CTAs hold a row count that is not a multiple of their warps.
#[test]
fn the_4e_v5_rows_are_the_served_chain() {
    let r = rot(2560, 41);
    let qkv = |grid: u32, threads: u32, seed: u64| {
        Cmd::new(b'V').u32(grid).u32(threads).u64(seed).u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small).0
    };
    for (grid, threads, seed) in [(3u32, 96u32, 1u64), (4, 64, 2), (7, 128, 3)] {
        assert!(run(served(), &qkv(grid, threads, seed)) == run(persistent_v5(), &qkv(grid, threads, seed)), "v5 q/k/v, grid {grid}x{threads}");
    }
    // grids 3 to 5: gate/up's pairs a CTA exceed form 3's 512 and take form 2's
    // path; 20 x 96: 486 or 487 pairs a CTA, 2 rows left over on 3 warps, the
    // split taken (and the down's and q|k's at every grid)
    for (form, grid, threads) in [(0u32, 3u32, 128u32), (2, 4, 96), (1, 5, 64), (1, 20, 96)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x4e60 + form as u64, 2);
        let want = run(served(), &cmd);
        let got = run(persistent_v5(), &cmd);
        let first = got.iter().zip(&want).position(|(a, b)| a != b);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v5, form {form}, grid {grid}x{threads}: tokens, residual or caches; lengths {} {}, first byte differing {first:?}", got.len(), want.len());
    }
    let cmd = g_cmd_variant(1, 3, 128, &[(270, 2, 7)], 288, u32::MAX, 0x4e70, 2);
    let (want, got) = (run(served(), &cmd), run(persistent_v5(), &cmd));
    assert!(got[..got.len() - 8] == want[..want.len() - 8], "v5, positions 270-271");
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let plain = run(persistent_v5(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x4e21, 2));
    let stamped = run(persistent_v5(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x4e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v5: the stamped launch moved an output");
}

/// v5's split: each mutant moves the q/k/v rows or the token loop.
#[test]
fn the_4e_v5_mutants_are_killed() {
    let r = rot(2560, 41);
    let v = |grid: u32, threads: u32| Cmd::new(b'V').u32(grid).u32(threads).u64(1).u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small).0;
    let cmds = [v(3, 96), v(7, 128), g_cmd_variant(0, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x4e60, 2), g_cmd_variant(1, 20, 96, &[(5, 3, 7)], 16, u32::MAX, 0x4e61, 2)];
    let want: Vec<Vec<u8>> = cmds.iter().map(|c| run(served(), c)).collect();
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    let s = "llvq-cuda/kernels/pers_segment.cu";
    for (tag, file, from, to) in [
        ("v5_fold_order", f, "        for (u32 i = 0; i < nit; ++i) {\n            const u32 u = rr * nit + i;", "        for (u32 i = nit; i-- > 0;) {\n            const u32 u = rr * nit + i;"),
        ("v5_gscale", f, "tv = __fmul_rn(a, R.gscale[(hi16 >> 15) & 1u]);", "tv = __fmul_rn(a, R.gscale[0]);"),
        ("v5_unit_step", f, "const u32 rr = u / nit, i = u - rr * nit;", "const u32 rr = u / nit, i = (u - rr * nit + 1u) % nit;"),
        ("v5_owner_row", f, "            epi(full + rr, pers_epilogue(acc, R.rscale[R.r],", "            epi(full + (rr ^ 1u) % rem, pers_epilogue(acc, R.rscale[R.r],"),
        ("v5_gu_product", s, "exact_hmul(A->silu[sgu[2u * p]], sgu[2u * p + 1u]);", "exact_hmul(A->silu[sgu[2u * p + 1u]], sgu[2u * p]);"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, file, from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v5"), &["PERS_4E_ROWS=3u"]);
        let killed = cmds.iter().zip(&want).any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "v5 mutant {tag} survived");
    }
}

// ---- 4E v6: ln1, ln2 and the down's input in one round trip (PERS_4E_ROWS = 4) --

/// The persistent side with 4E's row form 4 (4E v6: v4's rows, the
/// redundant prologues loaded as vectors, `small` staged in shared memory).
fn persistent_v6() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side_with(&repo(), 1, "off", "pers_v6", &["PERS_4E_ROWS=4u"]))
}

/// The -0 column of [`the_mix_turns_minus_zero_into_plus_zero_as_the_padding_does`]
/// at odd factor `k`: the last real term of every g >= 1 underflows to -0.
fn mix_minus_zero_cmd(k: u32) -> Vec<u8> {
    let m = 32u32;
    let mut s = vec![0.0f32; (k * m) as usize];
    let mut small = vec![0.0f32; KMAX * KMAX];
    let last = (k - 1) as usize;
    for j in 0..m as usize {
        s[last * m as usize + j] = 1e-30;
        s[j] = if j % 2 == 0 { 0.0 } else { 0.5 };
    }
    for g in 0..k as usize {
        small[g * KMAX + last] = -1e-20;
        small[g * KMAX] = if g == 0 { 1.0 } else { 0.0 };
    }
    Cmd::new(b'M').u32(m).u32(k).f32(1.0).f32s(&s).f32s(&small).0
}

/// The module built with row form 4 runs the served chain bit for bit:
/// - the rotation at every served width through the vector load and the
///   staged mix (k = 5 and 19 unrolled, the other k on the generic column),
///   at 64 threads (several batches, `small` staged in a loop) and 512 (the
///   card's geometry, one batch);
/// - the -0 column through the unrolled mixes;
/// - the token loop (ln1, ln2, a Tetra and an int4 down input), at four
///   geometries, past 256 positions, and through the stamp symbol.
#[test]
fn the_4e_v6_prologues_are_the_served_chain() {
    for &n in &ROT_WIDTHS {
        for threads in [64u32, 512] {
            let cmd = rot_cmd(n, threads, 0x6e00 + n as u64);
            assert!(run(served(), &cmd) == run(persistent_v6(), &cmd), "v6 rotation, n = {n}, {threads} threads");
        }
    }
    for k in [3u32, 5, 19] {
        let cmd = mix_minus_zero_cmd(k);
        let (a, b) = (run(served(), &cmd), run(persistent_v6(), &cmd));
        assert!(a == b, "v6 mix, k = {k}");
        assert!(u32s(&a).contains(&0), "k = {k}: some output is +0 (the column the padding fixed)");
    }
    for (form, grid, threads) in [(0u32, 3u32, 128u32), (2, 4, 96), (1, 5, 64), (1, 3, 512)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x6e40 + form as u64, 2);
        let want = run(served(), &cmd);
        let got = run(persistent_v6(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v6, form {form}, grid {grid}x{threads}: tokens, residual or caches");
    }
    let cmd = g_cmd_variant(1, 3, 128, &[(270, 2, 7)], 288, u32::MAX, 0x6e50, 2);
    let (want, got) = (run(served(), &cmd), run(persistent_v6(), &cmd));
    assert!(got[..got.len() - 8] == want[..want.len() - 8], "v6, positions 270-271");
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let plain = run(persistent_v6(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x6e21, 2));
    let stamped = run(persistent_v6(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x6e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v6: the stamped launch moved an output");
    let ev = pers::p4e_events(2) as usize;
    let st: Vec<u64> = stamped[plain.len()..].chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
    let r = pers::p4e_phase_report(&st[..3 * ev * 4], 3, 2, 4).expect("v6's stamps");
    assert!(r.subs.iter().all(|p| p.mean.0 > 0.0), "{r:?}");
}

/// v6's one-trip code: each mutant moves a rotation, the -0 column or the
/// token loop. A mutant that survived would mean the vector path is not the
/// one the commands run.
#[test]
fn the_4e_v6_mutants_are_killed() {
    let mut cmds = vec![rot_cmd(2560, 64, 0x6e01), rot_cmd(9728, 64, 0x6e02), rot_cmd(9728, 512, 0x6e03)];
    cmds.extend([5u32, 19].map(mix_minus_zero_cmd));
    cmds.push(g_cmd_variant(1, 3, 96, &[(5, 3, 7)], 16, u32::MAX, 0x6e60, 2));
    let want: Vec<Vec<u8>> = cmds.iter().map(|c| run(served(), c)).collect();
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    for (tag, from, to) in [
        ("v6_half_order", "for (u32 e = 0; e < 8u; ++e) s[8u * q + e] = ((bits >> e) & 1u) ? -f[e] : f[e];", "for (u32 e = 0; e < 8u; ++e) s[8u * q + (e ^ 1u)] = ((bits >> e) & 1u) ? -f[e] : f[e];"),
        ("v6_sign_word", "if (signbits != nullptr) sw[u] = signbits[q >> 2];", "if (signbits != nullptr) sw[u] = signbits[q >> 3];"),
        ("v6_ln_sign_shift", "const u32 bits = sw[u] >> ((q & 3u) * 8u);\n#pragma unroll\n                for (u32 e = 0; e < 8u; ++e) {", "const u32 bits = sw[u] >> ((q & 3u) * 4u);\n#pragma unroll\n                for (u32 e = 0; e < 8u; ++e) {"),
        ("v6_small_store", "if (i0 < ns) ssm[i0] = r.a0;", "if (i0 < ns) ssm[i0 ^ 1u] = r.a0;"),
        ("v6_small_rest", "for (u32 i = threadIdx.x + 2u * blockDim.x; i < ns; i += blockDim.x) ssm[i] = small[i];", ""),
        ("v6_mix_order", "for (u32 t = 0; t < K; ++t) acc = __fmaf_rn(ssm[g * LLVQ_ROT_KMAX + t], col[t], acc);", "for (u32 t = K; t-- > 0;) acc = __fmaf_rn(ssm[g * LLVQ_ROT_KMAX + t], col[t], acc);"),
        ("v6_mix_fadd", "if (K < LLVQ_ROT_KMAX) acc = __fadd_rn(acc, 0.0f);", ""),
        ("v6_apply_order", "return f2h(__fmul_rn(__fmul_rn(scale, xf), af));", "return f2h(__fmul_rn(scale, __fmul_rn(xf, af)));"),
        ("v6_alpha_vector", "a[u] = av[q];", "a[u] = av[q ^ 1u];"),
        ("v6_nat_copy", "if (nat) xs[i] = hv;", "if (nat) xs[i] = -hv;"),
        ("v6_rms_loader", "[&](u32 col) { return xs[col]; }", "[&](u32 col) { return __fmul_rn(xs[col], 1.0009765625f); }"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to); // copies the tree and applies the edit exactly once
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v6"), &["PERS_4E_ROWS=4u"]);
        let killed = cmds.iter().zip(&want).any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "v6 mutant {tag} survived");
    }
    // the call sites in the MLP segment: the token loop runs them (ln2, a Tetra
    // and an int4 down input), so each mutant moves its tokens or state
    let (g, want_g) = (cmds.last().unwrap(), want.last().unwrap());
    for (tag, from, to) in [
        ("v6_seg_ln2_signs", "pers_ln_1rt<NORMV>(L.x, A->alpha, A->signs_h,", "pers_ln_1rt<NORMV>(L.x, A->alpha, A->signs_i,"),
        ("v6_seg_down_k", "pers_load_1rt(A->act, A->signs_i, v, I, A->small_i, A->k_i, ssm);", "pers_load_1rt(A->act, A->signs_i, v, I, A->small_i, A->k_i - 1u, ssm);"),
        ("v6_seg_widen_signs", "pers_load_1rt(A->act, nullptr, v, I, nullptr, 0u, nullptr);", "pers_load_1rt(A->act, A->signs_i, v, I, nullptr, 0u, nullptr);"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, "llvq-cuda/kernels/pers_segment.cu", from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v6"), &["PERS_4E_ROWS=4u"]);
        assert!(run_mutant(&b, g).is_none_or(|o| &o != want_g), "v6 segment mutant {tag} survived");
    }
}

// ---- 4E v7: the launch's constants read during the barriers (PERS_4E_ROWS = 5) --

/// The persistent side with 4E's row form 5 (4E v7: v6, with the argument
/// structs, the CTA's slices, the rotations' sign words and `small` copied
/// into shared memory during the barrier before they are read, the o phase's
/// input widened from vectors, and `small` read as float4 in the mix).
fn persistent_v7() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side_with(&repo(), 1, "off", "pers_v7", &["PERS_4E_ROWS=5u"]))
}

/// The module built with row form 5 runs the served chain bit for bit:
/// - the rotation at every served width with its signs and `small` read
///   from the v7 area, and the float4 mix (k = 5 and 19), at 64 and 512
///   threads;
/// - the -0 column through the float4 mixes;
/// - the q/k/v phase and the o phase (its vector widening, x read at the
///   row's start) at several grids;
/// - the token loop (ln1 and ln2 from the staged area, a Tetra and an int4
///   down input, the layer struct staged during the barriers) at four
///   geometries, past 256 positions, over consecutive launches, and through
///   the stamp symbol.
#[test]
fn the_4e_v7_staged_constants_are_the_served_chain() {
    for &n in &ROT_WIDTHS {
        for threads in [64u32, 512] {
            let cmd = rot_cmd(n, threads, 0x7e00 + n as u64);
            assert!(run(served(), &cmd) == run(persistent_v7(), &cmd), "v7 rotation, n = {n}, {threads} threads");
        }
    }
    for k in [3u32, 5, 19] {
        let cmd = mix_minus_zero_cmd(k);
        let (a, b) = (run(served(), &cmd), run(persistent_v7(), &cmd));
        assert!(a == b, "v7 mix, k = {k}");
        assert!(u32s(&a).contains(&0), "k = {k}: some output is +0 (the column the padding fixed)");
    }
    for (grid, threads) in [(3u32, 96u32), (4, 64), (7, 128)] {
        let w = Cmd::new(b'W').u32(grid).u32(threads).u64(7).0;
        assert!(run(served(), &w) == run(persistent_v7(), &w), "v7 o phase, grid {grid}x{threads}");
    }
    let r = rot(2560, 41);
    for (grid, threads, seed) in [(3u32, 96u32, 1u64), (4, 64, 2)] {
        let v = Cmd::new(b'V').u32(grid).u32(threads).u64(seed).u32(r.m).u32(r.k).f32(r.inv).u32s(&r.signs).f32s(&r.small).0;
        assert!(run(served(), &v) == run(persistent_v7(), &v), "v7 q/k/v, grid {grid}");
    }
    for (form, grid, threads) in [(0u32, 3u32, 128u32), (2, 4, 96), (1, 5, 64), (1, 3, 512)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x7e40 + form as u64, 2);
        let want = run(served(), &cmd);
        let got = run(persistent_v7(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v7, form {form}, grid {grid}x{threads}: tokens, residual or caches");
    }
    let cmd = g_cmd_variant(1, 3, 128, &[(270, 2, 7)], 288, u32::MAX, 0x7e50, 2);
    let (want, got) = (run(served(), &cmd), run(persistent_v7(), &cmd));
    assert!(got[..got.len() - 8] == want[..want.len() - 8], "v7, positions 270-271");
    // consecutive launches through the same state, one rewinding: each
    // launch stages layer 0 again in its embedding barrier
    let cmd = g_cmd_variant(2, 4, 96, &[(5, 3, 7), (8, 2, 9), (6, 2, 3)], 16, u32::MAX, 0x7e60, 2);
    let (want, got) = (run(served(), &cmd), run(persistent_v7(), &cmd));
    assert!(got[..got.len() - 8] == want[..want.len() - 8], "v7, three consecutive launches");
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let plain = run(persistent_v7(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x7e21, 2));
    let stamped = run(persistent_v7(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x7e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v7: the stamped launch moved an output");
    let ev = pers::p4e_events(2) as usize;
    let st: Vec<u64> = stamped[plain.len()..].chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
    let rep = pers::p4e_phase_report(&st[..3 * ev * 4], 3, 2, 4).expect("v7's stamps");
    assert!(rep.subs.iter().all(|p| p.mean.0 > 0.0), "{rep:?}");
}

/// v7's staging: each mutant moves a rotation, the o phase or the token loop.
/// A survivor would mean the staged copy is not the one the commands read.
#[test]
fn the_4e_v7_mutants_are_killed() {
    let mut cmds = vec![rot_cmd(2560, 64, 0x7e01), rot_cmd(9728, 512, 0x7e03)];
    cmds.extend([5u32, 19].map(mix_minus_zero_cmd));
    cmds.push(Cmd::new(b'W').u32(3).u32(96).u64(7).0);
    cmds.push(g_cmd_variant(1, 3, 96, &[(5, 3, 7)], 16, u32::MAX, 0x7e70, 2));
    let want: Vec<Vec<u8>> = cmds.iter().map(|c| run(served(), c)).collect();
    let cuh = "llvq-cuda/kernels/llvq_pers.cuh";
    let e4 = "llvq-cuda/kernels/pers_4e.cu";
    let seg = "llvq-cuda/kernels/pers_segment.cu";
    for (tag, file, from, to) in [
        // the float4 mix: a component, and the padding's + 0 that turns -0 into +0
        ("v7_mix_lane", cuh, "acc = __fmaf_rn(w.x, col[4u * t4], acc);", "acc = __fmaf_rn(w.y, col[4u * t4], acc);"),
        ("v7_mix_fadd", cuh, "acc = K < LLVQ_ROT_KMAX ? __fadd_rn(acc, 0.0f) : acc;", "acc = K < LLVQ_ROT_KMAX ? __fadd_rn(acc, -0.0f) : acc;"),
        // the staged rows of `small` and sign words
        ("v7_stage_small", cuh, "                  k * LLVQ_ROT_KMAX);\n}", "                  (k - 1u) * LLVQ_ROT_KMAX);\n}"),
        ("v7_stage_signs", cuh, "pers_copy_u32(reinterpret_cast<u32*>(area + PERS_V7_SIGNS), signbits, width >> 5);",
         "pers_copy_u32(reinterpret_cast<u32*>(area + PERS_V7_SIGNS), signbits, (width >> 5) - 1u);"),
        // the gap wait's closing barrier: without it a thread reads the copies before they are made
        ("v7_gap_close", cuh, "    __syncthreads();\n    return hdr[PERS_HDR_OK] != 0u;\n}\n\n// n words", "    return hdr[PERS_HDR_OK] != 0u;\n}\n\n// n words"),
        // the o phase: the vector widening's width, and x at the row's start
        ("v7_o_width", cuh, "pers_load_1rt(ctx, nullptr, cs, d_ctx, nullptr, 0u, nullptr);", "pers_load_1rt(ctx, nullptr, cs, d_ctx - 8u, nullptr, 0u, nullptr);"),
        ("v7_o_x", cuh, "const unsigned short xr = lane == 0u ? __ldcg(x + r) : (unsigned short)0u;", "const unsigned short xr = lane == 0u ? __ldcg(x + (r ^ 1u)) : (unsigned short)0u;"),
        // 4E's staging: the args copy, the area's offset, the next layer, its slices, ln1 and ln2 inputs
        ("v7_args_words", e4, "(u32)(sizeof(Pers4eArgs) / 4u));", "(u32)(sizeof(Pers4eArgs) / 8u));"),
        ("v7_area_offset", e4, "if (threadIdx.x == 0u) pers_hdr()[PERS_HDR_V7] = v7_off;", "if (threadIdx.x == 0u) pers_hdr()[PERS_HDR_V7] = v7_off + 4u;"),
        ("v7_next_layer", e4, "if (l + 1u < A->n_layers) stage_layer(l + 1u);", "if (l + 1u < A->n_layers) stage_layer(l);"),
        ("v7_slice_hi", e4, "sl[2u * blockIdx.x + (threadIdx.x & 1u)];", "sl[2u * blockIdx.x];"),
        ("v7_ln1_stage", e4, "pers_v7_stage_rot(area, g->signs_a, Ag->hidden, g->small_a, g->k_a);", "pers_v7_stage_rot(area, g->mlp.signs_h, Ag->hidden, g->small_a, g->k_a);"),
        ("v7_ln2_stage", e4, "pers_v7_stage_rot(area, Y.mlp.signs_h, H, Y.mlp.small_h, Y.mlp.k_h);", "pers_v7_stage_rot(area, Y.signs_a, H, Y.mlp.small_h, Y.mlp.k_h);"),
        // the segment: the down's staging, and the staged signs it reads
        ("v7_down_stage", seg, "if (TETRA_DOWN) pers_v7_stage_rot(area, A->signs_i, I, A->small_i, A->k_i);", "if (TETRA_DOWN) pers_v7_stage_rot(area, A->signs_i, I, A->small_h, A->k_i);"),
        ("v7_down_signs", seg, "pers_load_1rt(A->act, v7g, v, I, nullptr, A->k_i, ssm);", "pers_load_1rt(A->act, A->signs_h, v, I, nullptr, A->k_i, ssm);"),
        ("v7_seg_slice", seg, "#define PERS_SEG_DN(i) (IN4E ? v7s[2u + (i)] :", "#define PERS_SEG_DN(i) (IN4E ? v7s[(i)] :"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, file, from, to); // copies the tree and applies the edit exactly once
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v7"), &["PERS_4E_ROWS=5u"]);
        let killed = cmds.iter().zip(&want).any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "v7 mutant {tag} survived");
    }
}

// ---- 4E v8: v7 with the attention steps in a cp.async ring (PERS_4E_ROWS = 6) --

/// The persistent side with 4E's row form 6 (4E v8: v7, and the attention
/// phase's steps staged into a ring of three stages, K3a of plan-300). The
/// host's cp.async copies at once, so these tests hold the ring's indexing
/// and barriers; the card's load-time identity holds its asynchrony.
fn persistent_v8() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side_with(&repo(), 1, "off", "pers_v8", &["PERS_4E_ROWS=6u"]))
}

/// The attention phase of form 6 is attn_ref's on every row length the
/// form-1 test reads (one step, two, four and eight a job; one split and
/// several), at three geometries; and the token loop is the served chain.
#[test]
fn the_4e_v8_attention_ring_is_the_served_chain() {
    let lens = [(2u32, 4u32, 64u32, 1u32), (2, 4, 64, 5), (2, 4, 128, 64), (2, 4, 128, 65), (2, 4, 192, 150), (2, 5, 512, 256), (2, 5, 512, 257), (1, 4, 1408, 1300), (3, 5, 2304, 2049)];
    for &(n_kv, groups, cap, l) in &lens {
        for &(grid, threads) in &[(3u32, 64u32), (5, 96), (1, 512)] {
            let (cmd, want, state) = attn_cmd(n_kv, groups, cap, l, grid, threads, 0xa77 + l as u64);
            let got = run(persistent_v8(), &cmd);
            let n = want.len() * 2;
            assert_eq!(u16s(&got[..n]), want, "v8: n_kv {n_kv}, groups {groups}, l {l}, grid {grid}x{threads}");
            let t = u32s(&got[n..n + 8]);
            assert_eq!((t[0], t[1]), (0, 0), "v8: every ticket back at zero");
            assert_eq!(u32s(&got[n + 8..]), state, "v8: the f32 state before the division, l {l}");
        }
    }
    for (form, grid, threads) in [(0u32, 3u32, 128u32), (2, 4, 96), (1, 5, 64), (1, 3, 512)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x8e40 + form as u64, 2);
        let want = run(served(), &cmd);
        let got = run(persistent_v8(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v8, form {form}, grid {grid}x{threads}: tokens, residual or caches");
    }
    // rows of 66 to 130 positions (four steps a job) and past 256 (two splits)
    for (pos0, cap, seed) in [(64u32, 160u32, 0x8e50u64), (125, 160, 0x8e51), (270, 288, 0x8e52)] {
        let cmd = g_cmd_variant(1, 3, 128, &[(pos0, 3, 7)], cap, u32::MAX, seed, 2);
        let (want, got) = (run(served(), &cmd), run(persistent_v8(), &cmd));
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v8, positions from {pos0}");
    }
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let plain = run(persistent_v8(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x8e21, 2));
    let stamped = run(persistent_v8(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x8e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v8: the stamped launch moved an output");
}

/// The ring's indexing and its reader barrier: each mutant moves a row of
/// four or eight steps. (Waits and commits are no-ops on the host; the card's
/// identity gate holds them.)
#[test]
fn the_4e_v8_mutants_are_killed() {
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [(2u32, 4u32, 192u32, 150u32), (1, 4, 1408, 1300)]
        .iter()
        .map(|&(n_kv, groups, cap, l)| {
            let (cmd, want, state) = attn_cmd(n_kv, groups, cap, l, 3, 64, 0xa78);
            let mut w: Vec<u8> = want.iter().flat_map(|x| x.to_le_bytes()).collect();
            w.extend([0u32, 0].iter().flat_map(|x| x.to_le_bytes()));
            w.extend(state.iter().flat_map(|x| x.to_le_bytes()));
            (cmd, w)
        })
        .collect();
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    for (tag, from, to) in [
        ("v8_issue_step", "pers_attn_issue(sbuf, kh, vh, base, st + PERS_ATTN_RING, nsteps, tpu, l);", "pers_attn_issue(sbuf, kh, vh, base, st + PERS_ATTN_RING - 1u, nsteps, tpu, l);"),
        ("v8_slot", "uint4* const sbuf = stage + (st % PERS_ATTN_RING) * PERS_STAGE_U4;", "uint4* const sbuf = stage + (st % 2u) * PERS_STAGE_U4;"),
        ("v8_issue_pos", "const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + pl % PERS_TT;", "const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + (pl + 1u) % PERS_TT;"),
        ("v8_first_issues", "for (u32 st = 0; st < PERS_ATTN_RING; ++st)\n            pers_attn_issue(", "for (u32 st = 0; st + 1u < PERS_ATTN_RING; ++st)\n            pers_attn_issue("),
        ("v8_reader_barrier", "            __syncthreads();  // the slot's readers are done: it takes step st + RING\n", ""),
        ("v8_kv_half", "const unsigned short* src = (idx < PERS_STAGE * 16u ? kh : vh);", "const unsigned short* src = (idx <= PERS_STAGE * 16u ? kh : vh);"),
        // the unit states and q after the whole ring, not after one stage
        ("v8_states_after_ring", "smem + 16u * PERS_STAGE_U4 * PERS_ATTN_RING);", "smem + 16u * PERS_STAGE_U4);"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v8"), &["PERS_4E_ROWS=6u"]);
        let killed = cmds.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "v8 mutant {tag} survived");
    }
}

// ---- 4E v9: v8 with the int4 rows on tiled words (PERS_4E_ROWS = 7) -------

/// The persistent side with 4E's row form 7 (4E v9: v8, and every int4 row of
/// the token loop, v, o and the int4 down, read from words relaid by
/// `pers_q4_tile_index`, K11 of plan-300). The harness relays the 4E side's
/// records as `persist4e_cuda.rs` does on the card; the served side keeps
/// the served words.
fn persistent_v9() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side_with(&repo(), 1, "off", "pers_v9", &["PERS_4E_ROWS=7u"]))
}

/// The token loop of form 7 is the served chain: both row variants, each
/// RoPE form, positions past 256, consecutive launches, and the stamp symbol.
/// The v rows (d_in 2,560: 2 tiles and a 64-word tail), the o rows (4,096: 4
/// tiles, no tail) and layer 1's int4 down (9,728: 9 tiles and a tail) all
/// run the tiled reader.
#[test]
fn the_4e_v9_int4_tiles_are_the_served_chain() {
    for (form, grid, threads, variant) in [(0u32, 3u32, 128u32, 0u32), (2, 4, 96, 2), (1, 5, 64, 2), (1, 3, 512, 2), (1, 2, 128, 0)] {
        let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x9e40 + form as u64, variant);
        let want = run(served(), &cmd);
        let got = run(persistent_v9(), &cmd);
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v9, form {form}, grid {grid}x{threads}, variant {variant}: tokens, residual or caches");
    }
    for (pos0, cap, seed) in [(64u32, 160u32, 0x9e50u64), (270, 288, 0x9e52)] {
        let cmd = g_cmd_variant(1, 3, 128, &[(pos0, 3, 7)], cap, u32::MAX, seed, 2);
        let (want, got) = (run(served(), &cmd), run(persistent_v9(), &cmd));
        assert!(got[..got.len() - 8] == want[..want.len() - 8], "v9, positions from {pos0}");
    }
    let runs = [(5u32, 3u32, 7u32), (8, 2, 9)];
    let cmd = g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x9e21, 2);
    let (want, plain) = (run(served(), &cmd), run(persistent_v9(), &cmd));
    assert!(plain[..plain.len() - 8] == want[..want.len() - 8], "v9: two consecutive launches");
    let stamped = run(persistent_v9(), &g_cmd_variant(1, 4, 96, &runs, 16, u32::MAX, 0x9e21, 1));
    assert!(stamped[..plain.len()] == plain[..], "v9: the stamped launch moved an output");
}

/// The tiled reader, its layout, and each int4 site of the token loop: each
/// mutant moves the tokens, the residual or a cache. A site mutant puts that
/// site back on the served reader, which then reads tiled words: it proves
/// the site runs the tiled reader under form 7.
#[test]
fn the_4e_v9_mutants_are_killed() {
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [(1u32, 3u32, 128u32, 2u32, 0x9e60u64), (1, 4, 96, 0, 0x9e61)]
        .iter()
        .map(|&(form, grid, threads, variant, seed)| {
            let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, seed, variant);
            let want = run(persistent_v9(), &cmd);
            (cmd, want)
        })
        .collect();
    let cuh = "llvq-cuda/kernels/llvq_pers.cuh";
    let seg = "llvq-cuda/kernels/pers_segment.cu";
    for (tag, f, from, to) in [
        ("v9_index", cuh, "return (t << 7) + 4u * (k & 31u) + (k >> 5);", "return (t << 7) + 4u * (k & 31u) + ((k >> 5) ^ 1u);"),
        ("v9_first_tile", cuh, "if (nt > 0u) P = *reinterpret_cast<const uint4*>(wrow + 4u * lane);", "if (nt > 0u) P = *reinterpret_cast<const uint4*>(wrow + 4u * (lane ^ 1u));"),
        ("v9_next_tile", cuh, "if (t + 1u < nt) P2 = *reinterpret_cast<const uint4*>(wrow + ((t + 1u) << 7) + 4u * lane);", "if (t + 1u < nt) P2 = *reinterpret_cast<const uint4*>(wrow + (t << 7) + 4u * lane);"),
        ("v9_column", cuh, "const u32 c = ((t << 7) + 32u * s + lane) << 3;", "const u32 c = ((t << 7) + 32u * (3u - s) + lane) << 3;"),
        ("v9_tail", cuh, "for (u32 wi = (nt << 7) + lane; wi < nwords; wi += 32u) {\n        const u32 p = wrow[wi];", "for (u32 wi = (nt << 7) + lane + 32u; wi < nwords; wi += 32u) {\n        const u32 p = wrow[wi];"),
        ("v9_site_v", cuh, "const float acc = pers_q4_row_4e<true>(P.vw,", "const float acc = pers_q4_row<true>(P.vw,"),
        ("v9_site_o", cuh, ": (unsigned short)0u;\n        const float acc = pers_q4_row_4e<PF>(ow,", ": (unsigned short)0u;\n        const float acc = pers_q4_row<PF>(ow,"),
        ("v9_site_down", seg, "const float acc = pers_q4_row_4e<true>(A->q4w,", "const float acc = pers_q4_row<true>(A->q4w,"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v9"), &["PERS_4E_ROWS=7u"]);
        let killed = cmds.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "v9 mutant {tag} survived");
    }
}

// ---- One copy of the int4 bytes: the eager readers on 4E v9's tiled words ---

/// The served side built from a mutated copy of the sources, each edit
/// applied exactly once (`mutant_many`'s rule, for the served kernels).
fn mutant_served(tag: &str, edits: &[(&str, &str, &str)]) -> PathBuf {
    let root = std::env::temp_dir().join(format!("llvq_pers_mutant_served_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for dir in ["llvq-cuda/kernels", "llvq-cuda/tests", "llvq-llm/kernels"] {
        let (src, dst) = (repo().join(dir), root.join(dir));
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
    build_side(&root, 0, "off", &format!("served_{tag}"))
}

/// Random served-order words for `d_out` rows of `d_in` columns.
fn q4_words(d_out: u32, d_in: u32, seed: u64) -> Vec<u32> {
    let mut r = llvq_core::SplitMix64::new(seed);
    (0..d_out as usize * d_in as usize / 8).map(|_| r.next() as u32).collect()
}

/// The eager int4 entry point `entry` (0 tv_q4_hh, 1 tv_q4_h, 2 tv_q4_hht,
/// 3 tv_q4_ht) on `words`: y then each row's f32, as bytes.
fn q4_cmd(d_out: u32, d_in: u32, entry: u32, seed: u64, words: &[u32]) -> Vec<u8> {
    Cmd::new(b'q').u32(d_out).u32(d_in).u32(entry).u64(seed).u32s(words).0
}

/// The widths of every int4 record of the three sealed files (v, o and down
/// d_in at 4B, 8B and 14B), and two that are not: one full tile, and a width
/// below one tile, all tail.
const Q4_WIDTHS: [u32; 8] = [2560, 4096, 5120, 9728, 12288, 17408, 1024, 512];

/// `pers::q4_tile_index` (the loader's permutation, on the host) is
/// `pers_q4_tile_index` (the device's, which 4E v9's relayout and reader use)
/// on every word of every width.
#[test]
fn the_host_tile_index_is_the_device_one() {
    let widths: Vec<u32> = Q4_WIDTHS.iter().map(|d| d / 8).collect();
    let got = u32s(&run(persistent(), &Cmd::new(b'i').u32(widths.len() as u32).u32s(&widths).0));
    let want: Vec<u32> = widths.iter().flat_map(|&nw| (0..nw).map(move |wi| pers::q4_tile_index(wi, nw))).collect();
    assert_eq!(got.len(), want.len());
    assert!(got == want, "the host permutation differs from pers_q4_tile_index");
}

/// The permutation keeps every word of a row, in that row, and moves no word
/// past the last full tile.
#[test]
fn the_tile_permutation_keeps_each_row_s_words() {
    for &d_in in &Q4_WIDTHS {
        let (d_out, nw) = (8usize, d_in as usize / 8);
        let words: Vec<u32> = (0..(d_out * nw) as u32).collect();
        let t = pers::q4_tile_words(&words, d_out, nw).unwrap();
        for r in 0..d_out {
            let mut row = t[r * nw..(r + 1) * nw].to_vec();
            let tail = nw / 128 * 128;
            assert!(row[tail..] == words[r * nw + tail..(r + 1) * nw], "d_in {d_in}: the tail moved");
            row.sort_unstable();
            assert!(row == words[r * nw..(r + 1) * nw], "d_in {d_in}: row {r} lost or gained a word");
        }
    }
    assert!(pers::q4_tile_words(&[0; 6], 1, 6).is_err(), "a row of 6 words is not whole 16-byte loads");
    assert!(pers::q4_tile_words(&[0; 8], 1, 4).is_err(), "8 words for one row of 4");
}

/// The single copy's claim: `tv_q4_hht` / `tv_q4_ht` on the words the loader
/// tiles return `tv_q4_hh` / `tv_q4_h`'s bytes on the file's words, every
/// row, f32 before the narrowing included, at every sealed width.
#[test]
fn the_tiled_eager_readers_are_the_served_readers() {
    for (i, &d_in) in Q4_WIDTHS.iter().enumerate() {
        let d_out = 24u32;
        let seed = 0x7a10 + i as u64;
        let words = q4_words(d_out, d_in, seed);
        let tiled = pers::q4_tile_words(&words, d_out as usize, d_in as usize / 8).unwrap();
        for (served_entry, tiled_entry) in [(0u32, 2u32), (1, 3)] {
            let want = run(served(), &q4_cmd(d_out, d_in, served_entry, seed, &words));
            let got = run(served(), &q4_cmd(d_out, d_in, tiled_entry, seed, &tiled));
            assert!(got == want, "d_in {d_in}: entry {tiled_entry} on tiled words against entry {served_entry}");
            if d_in >= 1024 {
                // the file's words through a tiled reader: wrong numbers, so the
                // comparison above is not vacuous
                let wrong = run(served(), &q4_cmd(d_out, d_in, tiled_entry, seed, &words));
                assert!(wrong != want, "d_in {d_in}: the tiled reader reads the file's order too");
            }
        }
    }
}

/// Each guard of `tv_q4_tile.cu` has a mutant, killed by a byte.
#[test]
fn the_tiled_eager_reader_mutants_are_killed() {
    let cases: Vec<(Vec<u8>, Vec<u8>)> = [2560u32, 9728, 4096]
        .iter()
        .flat_map(|&d_in| {
            let words = q4_words(16, d_in, 0x7a20 + d_in as u64);
            let tiled = pers::q4_tile_words(&words, 16, d_in as usize / 8).unwrap();
            [2u32, 3].map(|e| {
                let cmd = q4_cmd(16, d_in, e, 0x7a20 + d_in as u64, &tiled);
                let want = run(served(), &cmd);
                (cmd, want)
            })
        })
        .collect();
    let f = "llvq-llm/kernels/tv_q4_tile.cu";
    for (tag, from, to) in [
        ("tile_lane", "wrow + (t << 7) + 4u * lane);", "wrow + (t << 7) + 4u * (lane ^ 1u));"),
        ("tile_base", "wrow + (t << 7) + 4u * lane);", "wrow + 4u * lane);"),
        ("tile_word_order", "const u32 w[4] = {P.x, P.y, P.z, P.w};", "const u32 w[4] = {P.y, P.x, P.z, P.w};"),
        ("tile_column", "u32 c = ((t << 7) + 32u * s + lane) << 3;", "u32 c = ((t << 7) + 8u * s + lane) << 3;"),
        ("tile_group", "u32 g  = s0 + (c >> 7);\n            u32 sb", "u32 g  = s0 + (c >> 8);\n            u32 sb"),
        ("tile_count", "u32 nt = nwords >> 7;", "u32 nt = nwords >> 8;"),
        ("tile_tail_start", "for (u32 wi = (nt << 7) + lane; wi < nwords; wi += 32u) {", "for (u32 wi = (nt << 7) + lane + 32u; wi < nwords; wi += 32u) {"),
        ("tile_row_stride", "const u32* wrow = wq + (unsigned long long)row * nwords;", "const u32* wrow = wq + (unsigned long long)row * (nwords - 4u);"),
        ("tile_hh_widen", "for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xs[i] = h2f(x[i]);\n    __syncthreads();\n    q4t_staged_dot", "for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xs[i] = h2f(x[i]) * 0.5f;\n    __syncthreads();\n    q4t_staged_dot"),
    ] {
        let b = mutant_served(tag, &[(f, from, to)]);
        let killed = cases.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "tiled reader mutant {tag} survived");
    }
}

// ---- 4E at the 8B and the 14B: a Tetra o, an untied head --------------------

/// The 8B's two differences in the token loop (variant + 16): layer 1's
/// o_proj in Tetra under its own rotation at width 4,096, and a head table of
/// its own. Every form (v4, v6, v8, v9), both row variants, consecutive
/// launches, positions past 256, and the stamp symbol, against the served
/// chain (rot_apply then tv_tetra48_h for that o, tv_emb_q4_h on that table).
#[test]
fn the_4e_tetra_o_and_untied_head_are_the_served_chain() {
    type Side = fn() -> &'static Path;
    let forms: [(&str, Side); 4] =
        [("v4", persistent_v4), ("v6", persistent_v6), ("v8", persistent_v8), ("v9", persistent_v9)];
    for (label, side) in forms {
        for (form, grid, threads, variant) in [(1u32, 3u32, 128u32, 2u32), (0, 4, 96, 0), (2, 2, 512, 2)] {
            let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, 0x8b00 + form as u64, variant | 16);
            let (want, got) = (run(served(), &cmd), run(side(), &cmd));
            assert!(got[..got.len() - 8] == want[..want.len() - 8], "{label}, form {form}, {grid}x{threads}, variant {variant}: Tetra o, untied head");
        }
    }
    let runs = [(270u32, 3u32, 7u32), (5, 2, 9)];
    let cmd = g_cmd_variant(1, 3, 128, &runs, 288, u32::MAX, 0x8b10, 2 | 16);
    let (want, plain) = (run(served(), &cmd), run(persistent_v9(), &cmd));
    assert!(plain[..plain.len() - 8] == want[..want.len() - 8], "v9: past 256 and a rewound second launch");
    let stamped = run(persistent_v9(), &g_cmd_variant(1, 3, 128, &runs, 288, u32::MAX, 0x8b10, 1 | 16));
    assert!(stamped[..plain.len()] == plain[..], "v9: the stamped launch moved an output");
    // the extension is not vacuous: its tokens or state differ from the 4B
    // shape's on the same seed
    let base = run(served(), &g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x8b01, 2));
    let ext = run(served(), &g_cmd_variant(1, 3, 128, &[(5, 3, 7)], 16, u32::MAX, 0x8b01, 2 | 16));
    assert!(base != ext, "the Tetra o and the untied head change nothing in the served chain");
}

/// Each new guard has a mutant, killed by the 8B commands: the head reading
/// the embedding's table, the Tetra o phase skipped, its rotation dropped,
/// its residual read from the wrong row, its rows starting one too far.
#[test]
fn the_4e_tetra_o_and_untied_head_mutants_are_killed() {
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [(1u32, 3u32, 128u32, 2u32, 0x8b20u64), (1, 4, 96, 0, 0x8b21)]
        .iter()
        .map(|&(form, grid, threads, variant, seed)| {
            let cmd = g_cmd_variant(form, grid, threads, &[(5, 3, 7)], 16, u32::MAX, seed, variant | 16);
            let want = run(persistent_v9(), &cmd);
            (cmd, want)
        })
        .collect();
    let f = "llvq-cuda/kernels/pers_4e.cu";
    for (tag, from, to) in [
        ("head_tied", "pers_head_argmax<PF>(A->head_w, A->head_s, A->head_b,", "pers_head_argmax<PF>(A->emb_w, A->emb_s, A->emb_b,"),
        ("o_tetra_skipped", "            if (Y.ot != nullptr)\n                pers_4e_o_tetra(", "            if (false)\n                pers_4e_o_tetra("),
        ("o_tetra_no_rot", "    pers_rot_finish(cs, O.small, d_ctx, O.m, O.k, O.inv);\n    const u32 lane", "    __syncthreads();\n    const u32 lane"),
        ("o_tetra_resid", "const auto start = [&](u32 k) { xr = __ldcg(x + lo + warp + k * nw); };\n    const auto epi = [&](u32 k, float y) { out[lo", "const auto start = [&](u32 k) { xr = __ldcg(x + lo + warp + k * nw + 1u); };\n    const auto epi = [&](u32 k, float y) { out[lo"),
        ("o_tetra_rows", "    pers_4e_rows(pers_warp_rows(lo, hi, warp, nw), O.t.nblocks", "    pers_4e_rows(pers_warp_rows(lo + 1u, hi, warp, nw), O.t.nblocks"),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v9"), &["PERS_4E_ROWS=7u"]);
        let killed = cmds.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "8B mutant {tag} survived");
    }
}


// ---- The verify pass (pers_4ev.cu): n rows of the decode in one launch -----

/// `PERS_ATTN_SMEM(4)` of form 7 in floats: the smallest staging area a
/// verify launch takes at the 4B's groups. At it the o rows run in groups of
/// 3 and 1 and the down rows one at a time.
const V_ATTN4: u32 = 14_880;
/// The area that takes the down's 9,728 two rows at a time, and every row
/// phase four at a time (`P4V_SSM` and the head's maxima, then the rows).
const V_DOWN2: u32 = 2 * 9728 + 648;
const V_ALL4: u32 = 4 * 9728 + 648;

/// A `'G'` command whose last run is a verify run (variant bit 32): its rows
/// are that run's tok0 then `forced`, on a staging area of `act_floats`.
#[allow(clippy::too_many_arguments)]
fn v_cmd(form: u32, grid: u32, threads: u32, runs: &[(u32, u32, u32)], cap: u32, seed: u64, variant: u32, act_floats: u32, forced: &[u32]) -> Vec<u8> {
    assert_eq!(forced.len() + 1, runs.last().expect("a run").1 as usize, "forced tokens: the verify run's rows less one");
    let mut c = g_cmd_variant(form, grid, threads, runs, cap, u32::MAX, seed, variant | 32);
    c.extend(act_floats.to_le_bytes());
    for &t in forced {
        c.extend(t.to_le_bytes());
    }
    c
}

/// The verify launch is the served chain run token by token on the same
/// tokens: each row's argmax, both layers' KV caches (every row appends its
/// position), the last row's residual, and the barrier count of one 4E token
/// whatever the rows. Every RoPE form; a staging area that shares every row
/// phase across the four rows, one that takes the down two rows at a time,
/// one that takes it a row at a time and the o rows 3 + 1; the 8B's Tetra o
/// and untied head; after a 4E launch through the same state; rows across
/// the first split (lengths 255 to 258, the fold of row 2 and 3); one row.
#[test]
fn the_4e_verify_rows_are_the_served_chain() {
    let nb = llvq_cuda::pers::n_bar_4e(2);
    let check = |cmd: &[u8], grid: u32, bars: u32, what: &str| {
        let want = run(served(), cmd);
        let got = run(persistent_v9(), cmd);
        assert_eq!(got.len(), want.len(), "{what}: output length");
        let n = got.len() - 8;
        assert!(got[..n] == want[..n], "{what}: tokens, residual or KV caches differ from the served chain");
        assert_eq!(u32s(&got[n..]), vec![bars * grid, 0], "{what}: barrier count, abort and tickets");
        want
    };
    let forced = [11u32, 523, 7];
    for (form, grid, threads, act, variant) in
        [(0u32, 3u32, 128u32, V_ALL4, 0u32), (1, 4, 96, V_DOWN2, 0), (2, 5, 64, V_ATTN4, 0), (1, 3, 128, V_ATTN4, 16), (1, 4, 512, V_DOWN2, 16)]
    {
        let cmd = v_cmd(form, grid, threads, &[(5, 4, 7)], 16, 0xf400 + form as u64, variant, act, &forced);
        let want = check(&cmd, grid, nb, &format!("form {form}, grid {grid}x{threads}, area {act}, variant {variant}"));
        let ring = u32s(&want[..16]);
        assert!(ring.iter().all(|&t| t < 1024), "four argmaxes: {ring:?}");
    }
    // after a 4E launch of 3 tokens, through the same state and counter
    let cmd = v_cmd(1, 4, 96, &[(5, 3, 7), (8, 4, 9)], 16, 0xf410, 2, V_DOWN2, &forced);
    check(&cmd, 4, 4 * nb, "a 4E launch, then a verify launch");
    // rows across the first split: lengths 255, 256, 257, 258
    let cmd = v_cmd(1, 3, 128, &[(254, 4, 7)], 288, 0xf420, 0, V_ATTN4, &forced);
    check(&cmd, 3, nb, "rows across 256");
    let cmd = v_cmd(2, 4, 96, &[(270, 3, 7)], 288, 0xf421, 16, V_ALL4, &forced[..2]);
    check(&cmd, 4, nb, "three rows past 256, the 8B's parts");
    // one row: the verify launch is a 4E token
    let cmd = v_cmd(0, 3, 128, &[(5, 1, 7)], 16, 0xf430, 0, V_ATTN4, &[]);
    check(&cmd, 3, nb, "one row");
}

/// Each guard of the verify pass has a mutant, killed by a verify command:
/// a row's attention length, its RoPE position, site 1 of the Tetra rows, an
/// int4 row's vector stride, a group's v rows written to its first row, the
/// o residual, the down's rotation, the v rows read on signed values, the
/// vector ln that stages no `small`, the head key's row, the tickets left as
/// they were (rows past 256 fold on a stale count).
#[test]
fn the_4e_verify_mutants_are_killed() {
    let forced = [11u32, 523, 7];
    let cmds: Vec<(Vec<u8>, Vec<u8>)> = [
        v_cmd(1, 3, 128, &[(5, 4, 7)], 16, 0xf440, 0, V_ALL4, &forced),
        v_cmd(2, 4, 96, &[(254, 4, 7)], 288, 0xf441, 16, V_ATTN4, &forced),
    ]
    .into_iter()
    .map(|c| {
        let w = run(persistent_v9(), &c);
        (c, w)
    })
    .collect();
    let f = "llvq-cuda/kernels/pers_4ev.cu";
    for (tag, from, to) in [
        ("v_len", "const u32 l = L.pos0 + r + 1u;", "const u32 l = L.pos0 + 1u;"),
        ("v_rope_pos", "hh = w - r * per, pos = L.pos0 + r;", "hh = w - r * per, pos = L.pos0;"),
        ("v_site1", "acc[r] = __fmaf_rn(__fmul_rn(a, gs), iv, acc[r]);", "acc[r] = __fmaf_rn(a, __fmul_rn(gs, iv), acc[r]);"),
        ("v_q4_stride", "const float d = pers_q4_deq((wv[s] >> (4u * k)) & 0xfu, sb, bb);\n#pragma unroll\n                for (u32 r = 0; r < P4V_KMAX; ++r)\n                    if (r < n) acc[r] = __fmaf_rn(d, xs[r * xst + c + k], acc[r]);", "const float d = pers_q4_deq((wv[s] >> (4u * k)) & 0xfu, sb, bb);\n#pragma unroll\n                for (u32 r = 0; r < P4V_KMAX; ++r)\n                    if (r < n) acc[r] = __fmaf_rn(d, xs[c + k], acc[r]);"),
        ("v_v_row", "L.v_raw[(size_t)(r0 + r) * dkv + rv] = f2h(o[r]);", "L.v_raw[(size_t)r0 * dkv + rv] = f2h(o[r]);"),
        ("v_o_resid", "L.x_b[at] = exact_hadd(__ldcg(L.x_a + at), f2h(o[r]));", "L.x_b[at] = f2h(o[r]);"),
        ("v_down_rot", "pers_rot_finish<true, true>(R + r * I, ssm, I, M.m_i, M.k_i, M.inv_i);", "__syncthreads();"),
        ("v_v_natural", "                flip();\n                for (u32 rv = vlo + warp;", "                for (u32 rv = vlo + warp;"),
        ("v_ln_small", "pers_ln_1rt<0>(x, alpha, signs, w, eps, small, k, ssm, s, scratch, false);", "pers_ln_1rt<0>(x, alpha, signs, w, eps, nullptr, k, ssm, s, scratch, false);"),
        ("v_head_key", "else atomicMax(cmax + r, pers_tie_key(hb, row));", "else atomicMax(cmax + r, pers_tie_key(hb, row ^ 1u));"),
        ("v_tickets", "for (u32 i = threadIdx.x; i < P4V_KMAX * A->n_kv; i += blockDim.x) L.tickets[i] = 0u;", ""),
    ] {
        let root = std::env::temp_dir().join(format!("llvq_pers_mutant_{tag}_{}", std::process::id()));
        let _ = mutant(tag, f, from, to);
        let b = build_side_with(&root, 1, "off", &format!("{tag}_v9"), &["PERS_4E_ROWS=7u"]);
        let killed = cmds.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "verify mutant {tag} survived");
    }
}

/// `pers.rs::P4vLaunch` against `Pers4vLaunch` of the verify kernel: its
/// size, its fields' offsets, and the row bound.
#[test]
fn the_verify_launch_mirror_matches_the_kernel() {
    let o = u32s(&run(persistent_v9(), &Cmd::new(b'P').0));
    assert_eq!(o, vec![pers::P4V_LAUNCH_BYTES as u32, 12, 16, 24, 32, 88, 96, 104, 112, 120, pers::P4V_KMAX]);
    assert_eq!(std::mem::size_of::<pers::P4vLaunch>(), pers::P4V_LAUNCH_BYTES);
    // 4B: every phase of 4 rows fits under the L40S's 101,376 B; the 14B's
    // down does not, and takes the most that fits
    assert_eq!(pers::p4v_act_floats(2560, 9728, 4096, 4, 101_376), Ok(20_576));
    assert_eq!(pers::p4v_act_floats(2560, 9728, 4096, 4, 200_000), Ok(4 * 9728 + 648));
    assert_eq!(pers::p4v_act_floats(1024, 3072, 2048, 2, 101_376), Ok(13_584));
    assert!(pers::p4v_act_floats(5120, 17408, 5120, 5, 60_000).is_err());
}
