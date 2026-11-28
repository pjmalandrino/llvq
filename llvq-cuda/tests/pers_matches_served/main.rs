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
use llvq_search::rotation::Rotation;
use llvq_search::tetra::Tetra;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

mod served_text;
mod phases_4e;
mod route_p;
mod row_forms_4e;
mod rc;
mod tiled;
mod wide;
mod verify;
mod k8;

use phases_4e::{attn_cmd, g_cmd_variant};
use row_forms_4e::{persistent_e1, persistent_e1_twists, persistent_v6, persistent_v8, persistent_v9};

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
    let root = mutant_root(tag, edits);
    build_side(&root, 1, "off", tag)
}

/// The copy [`mutant_many`] builds from, not built: the edited sources only.
fn mutant_root(tag: &str, edits: &[(&str, &str, &str)]) -> PathBuf {
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
    root
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
