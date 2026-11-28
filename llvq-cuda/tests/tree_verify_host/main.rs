//! Stage S0 of the tree verify (`docs/plan-verify-tree.md` section 5): the
//! tree verify pass `kernels/pers_4et.cu`, executed on the Mac through
//! `tests/host_grid.h` (every CTA of the cooperative grid at once, a seeded
//! scheduler that lets warps drift apart), its tensor-core MMA modelled from
//! the PTX fragment layout (`tests/host_mma.h`), cp.async a synchronous copy.
//! The model is command 'G''s: two layers at the 4B's widths, a Tetra then an
//! int4 down, random weights (`tests/host_pers_tree.inc`).
//!
//! The gates of S0:
//!
//! 1. Identity. A row's argmax, logits, residual, K and V are bit for bit the
//!    same at R = 1 to 32, under slot permutations, scheduler seeds and grid
//!    shapes; and a row of a tree equals the same token decoded later as the
//!    root of an R = 1 pass, once its ancestors' K and V were moved into the
//!    cache by that pass's first phase.
//! 2. Closeness. R = 1 within 1e-2 relative RMS of 4E's logits for the same
//!    token at the same position, on the same fixture.
//! 3. Mutants. Each guard of the pass (row order, scale association, the
//!    tree mask, the fold order) has a mutant, applied to a copy of the
//!    source with an exactly-once count, and each is killed by gate 1 or 2.
//!
//! Times on the dev Mac, one core: closeness 46 s, the rest about 28 minutes
//! (R = 1 to 32 nine of them, the mutants thirteen). Those four are ignored in
//! debug: `cargo test --release -p llvq-cuda --test tree_verify_host --
//! --include-ignored --test-threads=1` runs them.
//!
//! What this does not establish: the card's tensor-core summation, which this
//! model replaces with a sequential f32 sum (identity on the card is S1's and
//! S3's), NVRTC compiling the device branches (the host build compiles none
//! of the `asm`), timing, memory ordering between real SMs.

use llvq_cuda::pers;
use llvq_search::rotation::Rotation;
use llvq_search::tetra::Tetra;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

mod gates;
mod stream_arms;
mod fold_arms;
mod spec_loop;
mod chain_prefill;
mod prefill_groups;
mod full_tile;

use chain_prefill::{gate_chain_rows, pass_of};
use gates::{gate_close, FIX};
use prefill_groups::{chain_at, chain_passes_at, g32_prompts, FIX_LONG};
use stream_arms::{gate_pair, SPLIT_N2, SPLIT_N2B};

const KMAX: usize = 32;
const RMAX: usize = 32;
const H: usize = 2560;
const DKV: usize = 1024;
const NL: usize = 2;
const VOCAB: u32 = 1024;
/// d_in slices of q/k/v, o, gate/up, down, head: three or more slices on the
/// down and q/k/v so a fold order that moved would change bits (two
/// partials commute), one on gate/up for the direct store.
const SPLIT: [u32; 5] = [3, 2, 1, 8, 2];
/// A second plan for the tiled gate (job N1): a1 changes the slices, so the
/// tiled pass is held to the untiled one at the same plan. Every phase folds.
const SPLIT_N1: [u32; 5] = [5, 3, 4, 6, 3];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("workspace root").to_path_buf()
}

fn build(root: &Path, tag: &str) -> PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_tree_{tag}_{}", std::process::id()));
    let res = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-DPERS_4E_ROWS=7u", "-DPERS_SIDE=1"])
        .arg(root.join("llvq-cuda/tests/host_pers.cpp"))
        .arg("-o")
        .arg(&out)
        .output()
        .expect("clang++ is on PATH (hard rule 10: this test fails, it does not skip)");
    assert!(res.status.success(), "clang++ refused host_pers.cpp ({tag}):\n{}", String::from_utf8_lossy(&res.stderr));
    out
}

fn host() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build(&repo(), "v9"))
}

/// One edit of a mutant: the file, the text, its replacement.
type Edit<'a> = (&'a str, &'a str, &'a str);

/// A copy of the files the harness reads with each edit applied exactly once,
/// built.
fn mutant(tag: &str, edits: &[Edit]) -> PathBuf {
    let root = std::env::temp_dir().join(format!("llvq_tree_mutant_{tag}_{}", std::process::id()));
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
    let b = build(&root, tag);
    let _ = std::fs::remove_dir_all(&root);
    b
}

/// The decoder tables and invnorm, every command's header.
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

/// The command's output, or the process's stderr when it failed: a crash or
/// an abort kills a mutant as surely as a wrong byte.
fn run(bin: &Path, cmd: &[u8]) -> Result<Vec<u8>, String> {
    let mut child = Command::new(bin).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = header().clone();
    input.extend_from_slice(cmd);
    let _ = child.stdin.take().unwrap().write_all(&input);
    let out = child.wait_with_output().unwrap();
    if out.status.success() {
        Ok(out.stdout)
    } else {
        Err(format!("{} failed: {}", bin.display(), String::from_utf8_lossy(&out.stderr)))
    }
}

fn put(v: &mut Vec<u8>, x: u32) {
    v.extend(x.to_le_bytes());
}

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

fn put_rot(v: &mut Vec<u8>, r: &Rot) {
    put(v, r.m);
    put(v, r.k);
    v.extend(r.inv.to_le_bytes());
    for &s in &r.signs {
        put(v, s);
    }
    for &s in &r.small {
        v.extend(s.to_le_bytes());
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A drafted tree: node 0 the root, every other node's parent an earlier
/// node, at most `max_depth` levels under the root.
struct Tree {
    parent: Vec<usize>,
    tok: Vec<u32>,
    depth: Vec<usize>,
}
fn tree(n: usize, max_depth: usize, seed: u64) -> Tree {
    let mut g = Lcg(seed | 1);
    let mut t = Tree { parent: vec![usize::MAX], tok: vec![(g.next() % VOCAB as u64) as u32], depth: vec![0] };
    while t.tok.len() < n {
        let p = g.below(t.tok.len());
        if t.depth[p] < max_depth {
            t.parent.push(p);
            t.depth.push(t.depth[p] + 1);
            t.tok.push((g.next() % VOCAB as u64) as u32);
        }
    }
    t
}
impl Tree {
    /// The root first, `i` last.
    fn path(&self, i: usize) -> Vec<usize> {
        let mut p = vec![i];
        while self.parent[*p.last().unwrap()] != usize::MAX {
            p.push(self.parent[*p.last().unwrap()]);
        }
        p.reverse();
        p
    }
    /// An ancestor-closed subset of `n` nodes (the root included), in a
    /// random slot order.
    fn subset(&self, n: usize, seed: u64) -> Vec<usize> {
        let mut g = Lcg(seed | 1);
        let mut inside = vec![false; self.tok.len()];
        inside[0] = true;
        let mut nodes = vec![0usize];
        while nodes.len() < n {
            let i = 1 + g.below(self.tok.len() - 1);
            if !inside[i] && inside[self.parent[i]] {
                inside[i] = true;
                nodes.push(i);
            }
        }
        for i in (1..nodes.len()).rev() {
            nodes.swap(i, g.below(i + 1));
        }
        nodes
    }
}

/// One launch: each slot's token, depth and path (slots, root first).
struct Pass {
    pos0: u32,
    seed: u64,
    toks: Vec<u32>,
    depth: Vec<u32>,
    path: Vec<Vec<u8>>,
    move_pos0: u32,
    mv: Vec<u32>,
}
impl Pass {
    /// Tree nodes `nodes` in that slot order, the root at pos0.
    fn of(t: &Tree, nodes: &[usize], pos0: u32, seed: u64) -> Pass {
        let slot = |n: usize| nodes.iter().position(|&m| m == n).expect("an ancestor-closed subset") as u8;
        Pass {
            pos0,
            seed,
            toks: nodes.iter().map(|&n| t.tok[n]).collect(),
            depth: nodes.iter().map(|&n| t.depth[n] as u32).collect(),
            path: nodes.iter().map(|&n| t.path(n).into_iter().map(slot).collect()).collect(),
            move_pos0: 0,
            mv: vec![],
        }
    }
    /// One row, a root: token `tok` at `pos0`, after moving the previous
    /// launch's slots `mv` to positions move_pos0, move_pos0 + 1, ...
    fn root(tok: u32, pos0: u32, seed: u64, move_pos0: u32, mv: Vec<u32>) -> Pass {
        Pass { pos0, seed, toks: vec![tok], depth: vec![0], path: vec![vec![0]], move_pos0, mv }
    }
}

#[derive(Clone, Copy)]
struct Fix {
    form: u32,
    grid: u32,
    threads: u32,
    cap: u32,
    variant: u32,
    seed: u64,
    split: [u32; 5],
}

fn y_cmd(f: &Fix, reference: Option<(u32, u32)>, passes: &[Pass]) -> Vec<u8> {
    let mut c = Vec::new();
    put(&mut c, b'Y' as u32);
    for v in [f.form, f.grid, f.threads, f.cap, VOCAB] {
        put(&mut c, v);
    }
    c.extend(f.seed.to_le_bytes());
    put(&mut c, f.variant);
    for (w, s) in [(2560usize, 51u64), (2560, 52), (9728, 53)] {
        put_rot(&mut c, &rot(w, s));
    }
    if f.variant & 16 != 0 {
        put_rot(&mut c, &rot(4096, 54));
    }
    for s in f.split {
        put(&mut c, s);
    }
    let (rp, rt) = reference.unwrap_or((u32::MAX, 0));
    put(&mut c, rp);
    put(&mut c, rt);
    put(&mut c, passes.len() as u32);
    for p in passes {
        put(&mut c, p.pos0);
        put(&mut c, p.toks.len() as u32);
        c.extend(p.seed.to_le_bytes());
        for &t in &p.toks {
            put(&mut c, t);
        }
        for &d in &p.depth {
            put(&mut c, d);
        }
        for pr in &p.path {
            let mut row = [0u8; RMAX];
            row[..pr.len()].copy_from_slice(pr);
            c.extend_from_slice(&row);
        }
        put(&mut c, p.mv.len() as u32);
        put(&mut c, p.move_pos0);
        for &m in &p.mv {
            put(&mut c, m);
        }
    }
    c
}

/// What a row of a pass leaves: its argmax, binary16 logits, final residual,
/// and its K then V at both layers.
#[derive(PartialEq)]
struct RowOut {
    tok: u32,
    logits: Vec<u16>,
    x: Vec<u16>,
    kv: Vec<u16>,
    /// a12 only: the row's five tap slots, `[5][H]` (empty otherwise)
    taps: Vec<u16>,
}

/// The variant bit of arm a12 (the taps).
const A12: u32 = 268435456;
/// The variant bit of arm a12ip: a12 reading the words in place, no copy
/// (step M1 of the memory axis).
const A12IP: u32 = 536870912;
/// The variant bit of arm a12ipg: a12ip whose attention reads each split
/// that ends at or below pos0 once for a group of rows (D1 of the
/// large-prefill study, `kernels/pers_4et_a12g.cu`).
const A12IPG: u32 = 2147483648;
/// The variant bit of arm a12ipg32: a12ipg's pass at RC = 4, 32 rows a
/// launch (design A of the large-prefill rows study, `kernels/pers_4et_a12g32.cu`).
const A12IPG32: u32 = 262144;

struct Out {
    reference: Option<(u32, Vec<u16>)>,
    passes: Vec<Vec<RowOut>>,
    tail: [u32; 2],
}

struct Rd<'a>(&'a [u8]);
impl Rd<'_> {
    fn u32s(&mut self, n: usize) -> Vec<u32> {
        let (a, b) = self.0.split_at(4 * n);
        self.0 = b;
        a.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn u16s(&mut self, n: usize) -> Vec<u16> {
        let (a, b) = self.0.split_at(2 * n);
        self.0 = b;
        a.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }
}

fn parse(bytes: &[u8], reference: bool, passes: &[Pass], taps: bool) -> Result<Out, String> {
    let v = VOCAB as usize;
    let tw = if taps { 5 * H } else { 0 };
    let mut need = if reference { 4 + 2 * v } else { 0 } + 8;
    for p in passes {
        let r = p.toks.len();
        need += r * (4 + 2 * v + 2 * H + 2 * tw + 2 * 2 * NL * DKV);
    }
    if bytes.len() != need {
        return Err(format!("output of {} bytes, expected {need}", bytes.len()));
    }
    let mut rd = Rd(bytes);
    let reference = if reference { Some((rd.u32s(1)[0], rd.u16s(v))) } else { None };
    let mut out = Vec::new();
    for p in passes {
        let r = p.toks.len();
        let ring = rd.u32s(r);
        let lg = rd.u16s(r * v);
        let x = rd.u16s(r * H);
        let tp = rd.u16s(r * tw);
        let kv: Vec<Vec<u16>> = (0..2 * NL).map(|_| rd.u16s(r * DKV)).collect();
        out.push(
            (0..r)
                .map(|s| RowOut {
                    tok: ring[s],
                    logits: lg[s * v..(s + 1) * v].to_vec(),
                    x: x[s * H..(s + 1) * H].to_vec(),
                    kv: kv.iter().flat_map(|b| b[s * DKV..(s + 1) * DKV].iter().copied()).collect(),
                    taps: tp[s * tw..(s + 1) * tw].to_vec(),
                })
                .collect(),
        );
    }
    let t = rd.u32s(2);
    Ok(Out { reference, passes: out, tail: [t[0], t[1]] })
}

fn exec(bin: &Path, f: &Fix, reference: Option<(u32, u32)>, passes: &[Pass]) -> Result<Out, String> {
    let out = parse(&run(bin, &y_cmd(f, reference, passes))?, reference.is_some(), passes, f.variant & (A12 | A12IP | A12IPG | A12IPG32) != 0)?;
    if out.tail[1] != 0 {
        return Err(format!("abort, NaN flag or tickets left: {}", out.tail[1]));
    }
    Ok(out)
}

/// Where two rows that owe each other their bits differ.
fn diff(a: &RowOut, b: &RowOut) -> Option<&'static str> {
    if a.tok != b.tok {
        Some("argmax")
    } else if a.logits != b.logits {
        Some("logits")
    } else if a.x != b.x {
        Some("residual")
    } else if a.kv != b.kv {
        Some("K or V")
    } else {
        None
    }
}
