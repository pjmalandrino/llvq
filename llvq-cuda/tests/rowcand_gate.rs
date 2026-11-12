//! Stage 1 of the decode-speed research phase's verifier (MODE RECHERCHE, R2,
//! `docs/mesures/recherche-debit-r1-cadrage-2026-11-12.txt`): the Mac gate,
//! $0, seconds.
//!
//! Every arm registered in `kernels/rowcand.cu` (`ROWC_ARMS`, read by
//! `llvq_cuda::rowcand::arms`) runs on the host through `tests/host_pers.cpp`
//! (command 'C', the persistent side) on RECORDED words, and its rows are held
//! against the served reference on the same words and activation (the served
//! side's `served_ref`: `tv_tetra48_h` with NVVM's three fusions, which
//! `tests/pers_matches_served.rs::the_served_replay_is_the_served_text` pins to
//! the served text). An arm that owes the bits must give y and its f32 value
//! bit for bit; an arm that declares a bound must stay under it, the relative
//! error of each row against the rms of its site's served rows, reported as
//! the max and the 0.99 quantile; a diagnostic owes nothing and is reported.
//!
//! The recorded words (`recorded()`):
//!   - a coverage set built field by field (`coverage_words`): every parity,
//!     section-1 class, mixed split and gain bit together (16 combinations),
//!     every prefix (s8, b1), every branch (s8, b2) with both suffix bits,
//!     hence every (s16, b3), every table row of each section in each class
//!     (section 1: (r, i1), section 2: i2 over the whole mixed order, section
//!     3: (p ^ r ^ delta, i3)), the origin, and words that reach every shell
//!     a random search of 2^18 labels reaches;
//!   - the real blocks of the six lattice records of
//!     `llvq-tetra/tests/fixtures/mini` (a one-layer Qwen3 the encoder wrote),
//!     at their own shapes with their own centroids, row scales and tails, and
//!     again relaid into rows of the 4B's widths;
//!   - seeded random words.
//!
//! `coverage_of_the_recorded_words` prints what is reached, and asserts it.
//!
//! The gate is mutated (`the_gate_kills_the_served_row_mutants`): edits of the
//! served row's own text, each applied to a copy of the sources with an
//! exactly-once count, and the gate must refuse every one.

use llvq_core::SplitMix64;
use llvq_cuda::rowcand::{self, Owes, RowErrors};
use llvq_search::tetra::{Fields, Tetra, LABEL_MASK, N0_MIXED};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("workspace root").to_path_buf()
}

fn build_side(root: &Path, side: u32, tag: &str) -> PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_rowcand_gate_{tag}_{side}_{}", std::process::id()));
    let res = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"])
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
    B.get_or_init(|| build_side(&repo(), 0, "served"))
}

fn candidate() -> &'static Path {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| build_side(&repo(), 1, "cand"))
}

/// A copy of the kernel and harness sources with each edit applied exactly
/// once, built as the persistent side. Returns the binary and, for each edit,
/// the number of sites it matched (always 1, or this panics): a mutant whose
/// text did not apply cannot be read as a survivor.
fn mutant(tag: &str, edits: &[(&str, &str, &str)]) -> (PathBuf, Vec<usize>) {
    let root = std::env::temp_dir().join(format!("llvq_rowcand_mutant_{tag}_{}", std::process::id()));
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
    let mut sites = Vec::new();
    for &(file, from, to) in edits {
        let path = root.join(file);
        let text = std::fs::read_to_string(&path).unwrap();
        let n = text.matches(from).count();
        assert_eq!(n, 1, "mutant {tag}: {n} sites for {from:?} in {file}, expected exactly one");
        let edited = text.replacen(from, to, 1);
        assert_ne!(edited, text, "mutant {tag}: the edit changes nothing");
        std::fs::write(&path, edited).unwrap();
        sites.push(n);
    }
    (build_side(&root, 1, tag), sites)
}

/// The decoder tables and invnorm, the harness's stdin header.
fn header() -> &'static Vec<u8> {
    static H: OnceLock<Vec<u8>> = OnceLock::new();
    H.get_or_init(|| {
        let t = tetra();
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

fn tetra() -> &'static Tetra {
    static T: OnceLock<Tetra> = OnceLock::new();
    T.get_or_init(Tetra::new)
}

/// The process's output, or `None` when it failed (a crash kills a mutant as
/// surely as a wrong byte).
fn run(bin: &Path, cmd: &[u8]) -> Option<Vec<u8>> {
    let mut child = Command::new(bin).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = header().clone();
    input.extend_from_slice(cmd);
    let _ = child.stdin.take().unwrap().write_all(&input);
    let out = child.wait_with_output().unwrap();
    if !out.status.success() {
        eprintln!("{} failed: {}", bin.display(), String::from_utf8_lossy(&out.stderr));
    }
    out.status.success().then_some(out.stdout)
}

// ---- recorded words ------------------------------------------------------------------------------------------

/// One Tetra matrix as the served stream holds it.
#[derive(Clone)]
struct Site {
    tag: &'static str,
    d_out: u32,
    d_in: u32,
    /// u32, row-strided, `stride_u32(nblocks)` a row.
    words: Vec<u32>,
    gscale: [f32; 2],
    rscale: Vec<f32>,
    tail: Vec<u16>,
}

fn stride(nblocks: usize) -> usize {
    (6 * nblocks).div_ceil(8) * 2
}

/// 48-bit words, row-major `d_out x nblocks`, into the served stream.
fn pack(w48: &[u64], d_out: usize, nblocks: usize) -> Vec<u32> {
    assert_eq!(w48.len(), d_out * nblocks);
    let st = stride(nblocks);
    let mut bytes = vec![0u8; d_out * st * 4];
    for r in 0..d_out {
        for j in 0..nblocks {
            let at = r * st * 4 + 6 * j;
            bytes[at..at + 6].copy_from_slice(&w48[r * nblocks + j].to_le_bytes()[..6]);
        }
    }
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn f16(v: f32) -> u16 {
    llvq_cuda::f16_bits(v)
}

/// A synthetic site of `d_in` filled with `w48` in order, padded with seeded
/// random words to whole rows and to a multiple of 8 rows (the served launch).
fn synth_site(tag: &'static str, w48: &[u64], d_in: u32, seed: u64) -> Site {
    let nb = (d_in / 24) as usize;
    let tw = (d_in % 24) as usize;
    let mut rng = SplitMix64::new(seed);
    let d_out = w48.len().div_ceil(nb).div_ceil(8) * 8;
    let mut all = w48.to_vec();
    while all.len() < d_out * nb {
        all.push(rng.next() & 0xffff_ffff_ffff);
    }
    Site {
        tag,
        d_out: d_out as u32,
        d_in,
        words: pack(&all, d_out, nb),
        gscale: [0.6 + 0.1 * rng.next_gaussian().abs() as f32, 1.3 + 0.1 * rng.next_gaussian().abs() as f32],
        rscale: (0..d_out).map(|_| (0.01 + 0.01 * rng.next_gaussian().abs()) as f32).collect(),
        tail: (0..d_out * tw).map(|_| f16((0.05 * rng.next_gaussian()) as f32)).collect(),
    }
}

/// The coverage set, field by field (module doc).
fn coverage_words() -> Vec<u64> {
    let t = tetra();
    let mut v = vec![0u64, 1u64 << 47, 0xffff_ffff_ffff];
    for pass in 0..4u64 {
        let (p, r) = (pass & 1, pass >> 1);
        for k in 0..2048u64 {
            let f = Fields {
                p: p as u8,
                r: r as u8,
                s8: (k & 63) as u8,
                b1: ((k >> 10) & 1 ^ (k >> 6) & 1) as u8,
                i1: k as u16,
                b2: ((k >> 6) & 15) as u8,
                i2: ((k + 1024 * (pass & 1)) % 2048) as u16,
                b3: ((k >> 10) & 1) as u8,
                i3: k as u16,
                gain: ((k >> 1 ^ pass) & 1) as u8,
            };
            v.push(f.join());
        }
    }
    // shells 2 and 3 by enumeration (llvq_core's Shell(2) and Shell(3), class by class): the first four points of
    // each class pattern that are Tetra codewords, with both gain bits
    let leech = llvq_core::Leech::new();
    for m in [2u32, 3] {
        let mut per: std::collections::BTreeMap<Vec<i32>, usize> = std::collections::BTreeMap::new();
        leech.for_each_shell_point(m, |pt| {
            let mut key: Vec<i32> = pt.iter().map(|a| a.abs()).collect();
            key.sort_unstable();
            let n = per.entry(key).or_insert(0);
            if *n < 4 {
                if let Some(w) = t.encode(pt) {
                    v.push(w | ((*n as u64 & 1) << 47));
                    *n += 1;
                }
            }
        });
    }
    // the higher shells: seeded random labels, then labels whose three rows are drawn from the last 128 of their
    // class (the table is in cost order, so these are the largest points); one word a shell not yet held
    let mut seen = [false; 32];
    for &w in &v {
        seen[shell(t, w)] = true;
    }
    let mut rng = SplitMix64::new(0x5e11);
    for k in 0..(1u32 << 19) {
        let w = if k % 2 == 0 {
            rng.next() & 0xffff_ffff_ffff
        } else {
            let f = Fields::split(rng.next());
            let top = |r: u64| (2047 - (r % 128)) as u16;
            let i2 = if rng.next() & 1 == 0 { (N0_MIXED - 1 - (rng.next() % 128) as usize) as u16 } else { top(rng.next()) };
            Fields { i1: top(rng.next()), i2, i3: top(rng.next()), ..f }.join()
        };
        let m = shell(t, w);
        if !seen[m] {
            seen[m] = true;
            v.push(w);
        }
    }
    v
}

fn shell(t: &Tetra, w: u64) -> usize {
    let y = t.decode(w & LABEL_MASK);
    let n2: i64 = y.iter().map(|&a| i64::from(a) * i64::from(a)).sum();
    assert_eq!(n2 % 16, 0, "a decoded point off the lattice");
    (n2 / 16) as usize
}

/// A safetensors tensor of the mini fixture: (dtype, shape, bytes).
fn st_tensor<'a>(file: &'a [u8], key: &str) -> (&'a str, Vec<usize>, &'a [u8]) {
    let n = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    let hdr = std::str::from_utf8(&file[8..8 + n]).unwrap();
    let at = hdr.find(&format!("\"{key}\":{{")).unwrap_or_else(|| panic!("no {key} in the fixture"));
    let rest = &hdr[at..];
    let field = |name: &str| -> &str {
        let s = rest.find(&format!("\"{name}\":")).unwrap() + name.len() + 3;
        let e = s + rest[s..].find([']', ',', '}']).unwrap() + 1;
        &rest[s..e]
    };
    let dtype = field("dtype").trim_matches(|c| c == '"' || c == ',' || c == '}');
    let nums = |s: &str| -> Vec<usize> {
        let s0 = rest.find(&format!("\"{s}\":[")).unwrap() + s.len() + 4;
        let e = s0 + rest[s0..].find(']').unwrap();
        rest[s0..e].split(',').filter(|x| !x.is_empty()).map(|x| x.trim().parse().unwrap()).collect()
    };
    let shape = nums("shape");
    let off = nums("data_offsets");
    (dtype, shape, &file[8 + n + off[0]..8 + n + off[1]])
}

fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

const MINI: &str = "llvq-tetra/tests/fixtures/mini/model.safetensors";
const MINI_RECORDS: [&str; 6] = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj"];

/// The fixture's lattice records: (name, d_out, d_in, 48-bit words in the
/// served convention, centroids, row scales, tail).
#[allow(clippy::type_complexity)]
fn fixture_records() -> Vec<(String, usize, usize, Vec<u64>, [f32; 2], Vec<f32>, Vec<f32>)> {
    let file = std::fs::read(repo().join(MINI)).expect("the mini fixture (hard rule 10: it fails, it does not skip)");
    let mut out = Vec::new();
    for rec in MINI_RECORDS {
        let p = format!("model.layers.0.{rec}");
        let (dt, _, codes) = st_tensor(&file, &format!("{p}.codes"));
        assert_eq!(dt, "U8");
        let (_, rs_shape, rs) = st_tensor(&file, &format!("{p}.row_scales"));
        let (_, tail_shape, tail) = st_tensor(&file, &format!("{p}.tail"));
        let (_, _, cen) = st_tensor(&file, &format!("{p}.centroids"));
        let d_out = rs_shape[0];
        let tw = tail_shape[1];
        let nblocks = codes.len() / 6 / d_out;
        assert_eq!(codes.len(), 6 * d_out * nblocks, "{p}: whole blocks");
        // hfpack's codes: one 48-bit value a block, MSB first, index then gain
        // bit (llvq_artifact::code_stream); the served word is index | gain << 47.
        let w48: Vec<u64> = codes
            .chunks_exact(6)
            .map(|c| {
                let v = c.iter().fold(0u64, |a, &b| a << 8 | u64::from(b));
                (v >> 1) | ((v & 1) << 47)
            })
            .collect();
        let c = f64s(cen);
        out.push((p, d_out, nblocks * 24 + tw, w48, [c[0] as f32, c[1] as f32], f64s(rs).iter().map(|&v| v as f32).collect(), f32s(tail)));
    }
    out
}

/// Every recorded site, and the activation.
fn recorded() -> &'static (Vec<Site>, Vec<f32>) {
    static R: OnceLock<(Vec<Site>, Vec<f32>)> = OnceLock::new();
    R.get_or_init(|| {
        let cov = coverage_words();
        let mut sites = vec![
            synth_site("coverage@2560", &cov, 2560, 0xc0_2560),
            synth_site("coverage@4096", &cov, 4096, 0xc0_4096),
            synth_site("coverage@9728", &cov, 9728, 0xc0_9728),
        ];
        let mut real = Vec::new();
        for (_, d_out, d_in, w48, gs, rs, tail) in fixture_records() {
            real.extend_from_slice(&w48);
            sites.push(Site {
                tag: "fixture",
                d_out: d_out as u32,
                d_in: d_in as u32,
                words: pack(&w48, d_out, d_in / 24),
                gscale: gs,
                rscale: rs,
                tail: tail.iter().map(|&v| f16(v)).collect(),
            });
        }
        sites.push(synth_site("fixture@2560", &real, 2560, 0xf1_2560));
        sites.push(synth_site("fixture@9728", &real, 9728, 0xf1_9728));
        let mut rng = SplitMix64::new(0x7a_7d);
        let rand: Vec<u64> = (0..4096).map(|_| rng.next() & 0xffff_ffff_ffff).collect();
        sites.push(synth_site("random@2560", &rand, 2560, 0x7a_2560));
        let mut rng = SplitMix64::new(0x7e_70);
        let x: Vec<f32> = (0..9728).map(|_| rng.next_gaussian() as f32).collect();
        (sites, x)
    })
}

// ---- the gate ------------------------------------------------------------------------------------------------

fn cmd(arm: usize, mode: u32, grid: u32, threads: u32, sites: &[&Site], x: &[f32]) -> Vec<u8> {
    let mut c: Vec<u8> = Vec::new();
    for v in [u32::from(b'C'), arm as u32, mode, grid, threads, sites.len() as u32] {
        c.extend(v.to_le_bytes());
    }
    for s in sites {
        c.extend(s.d_out.to_le_bytes());
        c.extend(s.d_in.to_le_bytes());
        for w in &s.words {
            c.extend(w.to_le_bytes());
        }
        for g in s.gscale {
            c.extend(g.to_le_bytes());
        }
        for r in &s.rscale {
            c.extend(r.to_le_bytes());
        }
        for t in &s.tail {
            c.extend(t.to_le_bytes());
        }
    }
    c.extend((x.len() as u32).to_le_bytes());
    for v in x {
        c.extend(v.to_le_bytes());
    }
    c
}

fn split(out: &[u8], rows: usize) -> (Vec<u16>, Vec<f32>) {
    assert_eq!(out.len(), 6 * rows, "binary16 rows, then their f32 values");
    let y = out[..2 * rows].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    (y, f32s(&out[2 * rows..]))
}

/// The gate on one binary: every site group, both bodies (mode 0 over every
/// site; mode 1 over gate/up pairs), several grids. Returns the errors, folded.
fn gate(bin: &Path, arm: usize) -> Option<RowErrors> {
    let (sites, x) = recorded();
    let all: Vec<&Site> = sites.iter().collect();
    // mode 1: the fixture's gate/up pair, then the 2,560-wide coverage site cut to the random site's rows and
    // the random site, as a pair (a gate/up pair has one shape)
    let fx_gate = sites.iter().position(|s| s.tag == "fixture" && s.d_out == 256).unwrap();
    let rand = sites.last().unwrap();
    assert_eq!((sites[0].d_in, rand.d_in, rand.tag), (2560, 2560, "random@2560"));
    let mut cut = sites[0].clone();
    let n = rand.d_out as usize;
    cut.words.truncate(n * stride(106));
    cut.rscale.truncate(n);
    cut.tail.truncate(n * 16);
    cut.d_out = rand.d_out;
    let pref: Vec<&Site> = vec![&sites[fx_gate], &sites[fx_gate + 1], &cut, rand];
    let mut folded: Option<RowErrors> = None;
    for (mode, set) in [(0u32, &all), (1u32, &pref)] {
        let rows: Vec<usize> = set.iter().map(|s| s.d_out as usize).collect();
        let total: usize = rows.iter().sum();
        let want = run(served(), &cmd(arm, mode, 1, 64, set, x)).expect("the served side runs");
        let (wy, wf) = split(&want, total);
        for &(grid, threads) in &[(3u32, 96u32), (2, 64), (5, 32)] {
            let got = run(bin, &cmd(arm, mode, grid, threads, set, x))?;
            if got.len() != want.len() {
                return None;
            }
            let (gy, gf) = split(&got, total);
            let e = rowcand::row_errors(&gy, &gf, &wy, &wf, &rows);
            folded = Some(match folded {
                None => e,
                Some(f) => RowErrors {
                    rows: f.rows + e.rows,
                    y_equal: f.y_equal + e.y_equal,
                    f_equal: f.f_equal + e.f_equal,
                    max: f.max.max(e.max),
                    q99: f.q99.max(e.q99),
                    nonfinite: f.nonfinite + e.nonfinite,
                },
            });
        }
    }
    folded
}

fn line(name: &str, owes: Owes, e: &RowErrors) -> String {
    format!(
        "{name:<10} owes {:<14} rows {:>6}: y equal {:>6}, f32 equal {:>6}, max rel err {:.3e}, q99 {:.3e}: {}",
        match owes {
            Owes::Bits => "the bits".to_string(),
            Owes::Bound(b) => format!("bound {b:.0e}"),
            Owes::Nothing => "nothing".to_string(),
        },
        e.rows,
        e.y_equal,
        e.f_equal,
        e.max,
        e.q99,
        match owes {
            Owes::Nothing => "diagnostic",
            o if e.passes(o) => "PASS",
            _ => "FAIL",
        }
    )
}

/// Every registered arm keeps what it owes; the served arm is the served row.
#[test]
fn every_registered_arm_keeps_what_it_owes() {
    let arms = rowcand::arms();
    for (i, a) in arms.iter().enumerate() {
        let e = gate(candidate(), i).unwrap_or_else(|| panic!("{}: the candidate side failed", a.name));
        println!("{}", line(&a.name, a.owes, &e));
        assert!(e.passes(a.owes), "{} does not keep what it owes: {e:?}", a.name);
        if a.is_served() {
            assert!(e.bit_equal(), "the served arm is not the served row");
        }
        if a.owes == Owes::Nothing {
            assert!(!e.bit_equal(), "{}: a diagnostic that gives the served rows computes them", a.name);
        }
    }
}

/// What the recorded words reach, printed and asserted.
#[test]
fn coverage_of_the_recorded_words() {
    let t = tetra();
    let cov = coverage_words();
    let mut combos = std::collections::BTreeSet::new();
    let (mut pre, mut br, mut suf) = (vec![false; 128], vec![false; 1024], vec![false; 128]);
    let (mut s1, mut s2, mut s3) = (vec![false; 4096], vec![false; 2048], vec![false; 4096]);
    for &w in &cov {
        let f = Fields::split(w);
        let delta = u8::from(usize::from(f.i2) >= N0_MIXED);
        combos.insert((f.p, f.r, delta, f.gain));
        pre[2 * f.s8 as usize + f.b1 as usize] = true;
        br[16 * f.s8 as usize + f.b2 as usize] = true;
        let s16 = t.branches()[f.s8 as usize][f.b2 as usize].1;
        suf[2 * s16 as usize + f.b3 as usize] = true;
        s1[2048 * f.r as usize + f.i1 as usize] = true;
        s2[f.i2 as usize] = true;
        let r3 = (f.p ^ f.r ^ delta) & 1;
        s3[2048 * r3 as usize + f.i3 as usize] = true;
    }
    let count = |v: &[bool]| v.iter().filter(|&&b| b).count();
    let shells = |ws: &[u64]| -> std::collections::BTreeMap<usize, usize> {
        let mut h = std::collections::BTreeMap::new();
        for &w in ws {
            *h.entry(shell(t, w)).or_insert(0) += 1;
        }
        h
    };
    let cov_sh = shells(&cov);
    // the shells a wider random search reaches, against the coverage set's
    let mut rng = SplitMix64::new(0x5e_a7c4);
    let wide: Vec<u64> = (0..(1u32 << 20)).map(|_| rng.next() & 0xffff_ffff_ffff).collect();
    let wide_sh = shells(&wide);
    let real: Vec<u64> = fixture_records().into_iter().flat_map(|r| r.3).collect();
    let real_sh = shells(&real);
    let real_gain = real.iter().filter(|&&w| (w >> 47) & 1 == 1).count();
    println!("coverage set: {} words", cov.len());
    println!("  (p, r, delta, gain): {} of 16", combos.len());
    println!("  prefixes (s8, b1): {} of 128; branches (s8, b2): {} of 1024; suffixes (s16, b3): {} of 128", count(&pre), count(&br), count(&suf));
    println!("  table rows read: section 1 (r, i1) {} of 4096; section 2 i2 {} of 2048; section 3 (p^r^delta, i3) {} of 4096", count(&s1), count(&s2), count(&s3));
    println!("  shells m (count): {cov_sh:?}");
    println!("  shells in 0..=26 not reached: {:?}", (0..=26usize).filter(|m| !cov_sh.contains_key(m)).collect::<Vec<_>>());
    println!("  shells a random search of 2^20 labels reaches: {:?}", wide_sh.keys().collect::<Vec<_>>());
    println!("fixture: {} real blocks of {} records, gain bit set on {real_gain}; shells (count): {real_sh:?}", real.len(), MINI_RECORDS.len());
    assert_eq!(combos.len(), 16);
    assert_eq!((count(&pre), count(&br), count(&suf)), (128, 1024, 128));
    assert_eq!((count(&s1), count(&s2), count(&s3)), (4096, 2048, 4096));
    assert!(cov_sh.contains_key(&0), "the origin");
    for m in 2..=25usize {
        assert!(cov_sh.contains_key(&m), "shell {m}: reached by the seeds of this file, and no longer");
    }
    for m in wide_sh.keys() {
        assert!(cov_sh.contains_key(m), "shell {m} is reached at random and not by the coverage set");
    }
    assert!(real.len() > 4000 && real_gain > 0 && real_gain < real.len(), "the fixture's blocks carry both gain levels");
}

/// The served row's lane loop, the site of the two lane mutants.
const LANE_SITE: &str = "        u32 lo = 0u, hi16 = 0u;\n        if (lane < nblocks) f1r_load(wrow, lane, lo, hi16);\n        for (u32 j = lane; j < nblocks; j += 32u) {";

/// Mutants of the served row: each must be refused by the gate on the served
/// arm. The edits touch the text the served arm compiles (the shared
/// decoder headers), in a copy built as the candidate side only.
#[test]
fn the_gate_kills_the_served_row_mutants() {
    let v3 = "llvq-cuda/kernels/llvq_f1rank_v3.cuh";
    let pers = "llvq-cuda/kernels/llvq_pers.cuh";
    let cases: [(&str, &str, &str, &str); 5] = [
        // a wrong table index: section 3's row read in the other class when p = 1
        ("table_index", v3, "b.row3 = t.rows[2048u * r3 + i3];", "b.row3 = t.rows[2048u * (r3 ^ p) + i3];"),
        // a dropped sign: every decoded coordinate's magnitude
        (
            "dropped_sign",
            v3,
            "return __uint_as_float(__byte_perm(quad, 0x4b000000u, 0x7440u | j)) - F1R_V3_BIAS;",
            "return fabsf(__uint_as_float(__byte_perm(quad, 0x4b000000u, 0x7440u | j)) - F1R_V3_BIAS);",
        ),
        // a swapped lane: lanes 0 and 1 trade their blocks (every block still folded once,
        // on its own activation: a pure reassociation of the butterfly)
        (
            "swapped_lane",
            pers,
            LANE_SITE,
            "        const u32 sl = lane < 2u ? lane ^ 1u : lane;\n        u32 lo = 0u, hi16 = 0u;\n        if (sl < nblocks) f1r_load(wrow, sl, lo, hi16);\n        for (u32 j = sl; j < nblocks; j += 32u) {",
        ),
        // the gain level dropped: every block at centroid 0
        (
            "gain_dropped",
            pers,
            "    const u32 g = (hi16 >> 15) & 1u;\n    return __fmaf_rn(__fmul_rn(acc_b, gscale[g]), invnorm[m], acc);",
            "    const u32 g = 0u * (hi16 >> 15);\n    return __fmaf_rn(__fmul_rn(acc_b, gscale[g]), invnorm[m], acc);",
        ),
        // the mixed split moved by one: only a word at i2 = 1,240 sees it
        ("mixed_split", v3, "bool mid  = i2 < F1R_N0_MIXED;", "bool mid  = i2 <= F1R_N0_MIXED;"),
    ];
    let served_arm = 0usize;
    assert!(rowcand::arms()[served_arm].is_served());
    for (tag, file, from, to) in cases {
        let (bin, sites) = mutant(tag, &[(file, from, to)]);
        let verdict = match gate(&bin, served_arm) {
            None => "killed: the process failed".to_string(),
            Some(e) => {
                assert!(!e.passes(Owes::Bits), "mutant {tag} survived the gate: {e:?}");
                format!("killed: {} of {} rows moved, max rel err {:.3e}, q99 {:.3e}", e.rows - e.f_equal, e.rows, e.max, e.q99)
            }
        };
        println!("mutant {tag:<13} {} site(s) in {file}: {verdict}", sites.iter().sum::<usize>());
    }
    // An EQUIVALENT mutant, checked as such: every pair of lanes 2k, 2k + 1 swapped is an automorphism of
    // warp_sum's xor butterfly, and each of its additions is commutative, so the bits cannot move. The gate
    // must report it equal (a gate that refused it would refuse the served row too).
    let (bin, sites) = mutant(
        "xor_pairs",
        &[(
            pers,
            LANE_SITE,
            "        const u32 sl = lane ^ 1u;\n        u32 lo = 0u, hi16 = 0u;\n        if (sl < nblocks) f1r_load(wrow, sl, lo, hi16);\n        for (u32 j = sl; j < nblocks; j += 32u) {",
        )],
    );
    let e = gate(&bin, served_arm).expect("the equivalent mutant runs");
    assert!(e.bit_equal(), "the xor-pair lane swap moved bits: the butterfly is not what this test assumes: {e:?}");
    println!("mutant xor_pairs     {} site(s) in {pers}: equivalent, {} of {} rows equal bit for bit", sites[0], e.f_equal, e.rows);
}

/// The bandwidth probe on the host: every kernel's sum is the host's own sum of
/// the hash, with and without the prefetch, on lengths that leave a remainder;
/// and the probe's remainder loop is load-bearing (its mutant is caught).
#[test]
fn the_probe_sums_what_the_host_sums() {
    let probe = |bin: &Path, k: u32, n: u64, seed: u32, grid: u32, threads: u32, pf: u32| -> Option<(u64, u64)> {
        let mut c = Vec::new();
        c.extend(u32::from(b'b').to_le_bytes());
        c.extend(k.to_le_bytes());
        c.extend(n.to_le_bytes());
        for v in [seed, grid, threads, pf] {
            c.extend(v.to_le_bytes());
        }
        let o = run(bin, &c)?;
        Some((u64::from_le_bytes(o[..8].try_into().unwrap()), u64::from_le_bytes(o[8..16].try_into().unwrap())))
    };
    for (k, &(name, _, vec, unr)) in rowcand::BW_KERNELS.iter().enumerate() {
        for &(n, grid, threads, pf) in &[(65_536u64, 3u32, 64u32, 0u32), (65_540, 2, 96, 2), (4 * 1001, 5, 32, 1)] {
            let (sum, parts) = probe(candidate(), k as u32, n, 0xb00 + n as u32, grid, threads, pf).expect("the probe runs");
            assert_eq!(parts, u64::from(grid));
            assert_eq!(sum, rowcand::bw_expected_sum(n, 0xb00 + n as u32), "{name} (vec {vec}, unroll {unr}), n {n}, grid {grid}x{threads}, prefetch {pf}");
        }
    }
    let (bin, sites) = mutant("probe_tail", &[("llvq-cuda/kernels/bwprobe.cu", "    for (; i < ne; i += S) sum += bw_load<VEC>(buf + i * VEC);\n", "")]);
    assert_eq!(sites, [1]);
    let (sum, _) = probe(&bin, 2, 65_540, 3, 2, 96, 0).expect("the mutant runs");
    assert_ne!(sum, rowcand::bw_expected_sum(65_540, 3), "the probe's remainder loop is not load-bearing");
}
