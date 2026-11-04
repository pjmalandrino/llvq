//! The split-KV decode-attention kernel, executed on the development Mac and
//! diffed against its Rust replay bit for bit.
//!
//! `tests/host_attn_dec.cpp` compiles `kernels/attn_dec.cu` with clang++ and
//! runs it on `tests/host_cta.h`: 128 OS threads a CTA, real barriers,
//! 16-lane shuffles, real tickets, CTAs in a permuted order. So what is
//! executed here is the kernel's whole text, its indexing, staging, unit fold,
//! ticket and split combine, not only its shuffle-free arithmetic. Every
//! output is compared with `llvq_cuda::attn_ref` bit for bit.
//!
//! What is NOT proved here: memory ordering between CTAs that run at the same
//! time on a card, the inline PTX conversions (`cvt.f32.f16`,
//! `cvt.rn.f16.f32`, replaced by `_Float16` on the host), and NVRTC's reading
//! of the text. The card's load-time check (`llvq_cuda::attn`) compares the
//! same replay against the card before any token, and stops the load on one
//! differing bit.
//!
//! A missing clang++ fails and names the tool (hard rule 10).

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use llvq_cuda::attn_ref::{self, Kv, HD, SPLIT};

fn build(tag: &str, extra: &[&str]) -> std::path::PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_attn_dec_{tag}_{}", std::process::id()));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-pthread"])
        .args(extra)
        .arg(dir.join("host_attn_dec.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ must be on PATH: the attention kernel text is checked by it (rule 10)");
    assert!(st.success(), "attn_dec.cu does not compile as host C++ through host_cta.h ({tag})");
    out
}

/// Build the probe once for the whole test binary: cargo runs these tests as
/// threads of one process, and each build writes the same file otherwise.
fn probe() -> &'static std::path::Path {
    static EXE: OnceLock<std::path::PathBuf> = OnceLock::new();
    EXE.get_or_init(|| build("o2", &[]))
}

fn run(input: &[u8]) -> Vec<u8> {
    run_with(probe(), input)
}

fn run_with(exe: &std::path::Path, input: &[u8]) -> Vec<u8> {
    let mut ch = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the probe runs");
    let mut stdin = ch.stdin.take().expect("piped");
    let data = input.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&data).expect("fixture written"));
    let out = ch.wait_with_output().expect("the probe exits");
    writer.join().expect("writer");
    assert!(
        out.status.success(),
        "the probe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn u32s(v: &mut Vec<u8>, xs: &[u32]) {
    for x in xs {
        v.extend_from_slice(&x.to_le_bytes());
    }
}

fn u16s(v: &mut Vec<u8>, xs: &[u16]) {
    for x in xs {
        v.extend_from_slice(&x.to_le_bytes());
    }
}

fn read_u16(b: &[u8], n: usize, at: &mut usize) -> Vec<u16> {
    let v = b[*at..*at + 2 * n].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    *at += 2 * n;
    v
}

/// One kernel launch on the host, as the card would see it.
struct Launch {
    g: usize,
    unit: usize,
    n_kv: usize,
    cap: usize,
    pos0: usize,
    k: usize,
    grid_x: usize,
    s_stride: usize,
    with_new: bool,
    seed: u64,
}

struct Result {
    out: Vec<u16>,
    kc: Vec<u16>,
    vc: Vec<u16>,
    tickets: Vec<u32>,
}

fn encode(l: &Launch, c: f32, q: &[u16], kc: &[u16], vc: &[u16], new: Option<(&[u16], &[u16])>) -> Vec<u8> {
    let mut v = Vec::new();
    u32s(
        &mut v,
        &[
            2,
            l.g as u32,
            l.unit as u32,
            l.n_kv as u32,
            l.cap as u32,
            l.pos0 as u32,
            l.k as u32,
            l.grid_x as u32,
            l.s_stride as u32,
        ],
    );
    v.extend_from_slice(&c.to_le_bytes());
    u32s(&mut v, &[l.with_new as u32]);
    v.extend_from_slice(&l.seed.to_le_bytes());
    u16s(&mut v, q);
    u16s(&mut v, kc);
    u16s(&mut v, vc);
    if let Some((kn, vn)) = new {
        u16s(&mut v, kn);
        u16s(&mut v, vn);
    }
    v
}

fn decode(l: &Launch, b: &[u8]) -> Result {
    let nq = l.n_kv * l.g * l.k * HD;
    let ncache = l.n_kv * l.cap * HD;
    let mut at = 0;
    let out = read_u16(b, nq, &mut at);
    let kc = read_u16(b, ncache, &mut at);
    let vc = read_u16(b, ncache, &mut at);
    let tickets = b[at..at + 4 * l.n_kv]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    at += 4 * l.n_kv;
    assert_eq!(at, b.len(), "the probe wrote {} bytes, {at} expected", b.len());
    Result { out, kc, vc, tickets }
}

/// A fixture for one launch: q, the caches before the call, and the fresh
/// rows when the kernel is to write them. Positions the rows do not read hold
/// a value no finite arithmetic survives (inf), so reading one shows.
struct Fixture {
    q: Vec<u16>,
    kc: Vec<u16>,
    vc: Vec<u16>,
    kn: Vec<u16>,
    vn: Vec<u16>,
}

const F16_INF: u16 = 0x7c00;

fn fixture(l: &Launch, seed: u64) -> Fixture {
    let nq = l.n_kv * l.g * l.k * HD;
    let ncache = l.n_kv * l.cap * HD;
    let mut kc = attn_ref::fill(ncache, seed ^ 0x11, 1.0);
    let mut vc = attn_ref::fill(ncache, seed ^ 0x22, 1.0);
    // Past the last position read: a poison value.
    for h in 0..l.n_kv {
        for p in l.pos0 + l.k..l.cap {
            for i in 0..HD {
                kc[(h * l.cap + p) * HD + i] = F16_INF;
                vc[(h * l.cap + p) * HD + i] = F16_INF;
            }
        }
    }
    let kn = attn_ref::fill(l.n_kv * l.k * HD, seed ^ 0x33, 1.0);
    let vn = attn_ref::fill(l.n_kv * l.k * HD, seed ^ 0x44, 1.0);
    if l.with_new {
        // The fresh positions are poisoned in the cache: the kernel must
        // write them before it reads them.
        for h in 0..l.n_kv {
            for r in 0..l.k {
                for i in 0..HD {
                    kc[(h * l.cap + l.pos0 + r) * HD + i] = F16_INF;
                    vc[(h * l.cap + l.pos0 + r) * HD + i] = F16_INF;
                }
            }
        }
    }
    // Queries of a realistic scale: scores of a few units after the scale.
    let q = attn_ref::fill(nq, seed ^ 0x55, 4.0);
    Fixture { q, kc, vc, kn, vn }
}

/// The cache as the kernel must leave it: the fresh rows written, if any.
fn expected_cache(l: &Launch, f: &Fixture) -> (Vec<u16>, Vec<u16>) {
    let (mut kc, mut vc) = (f.kc.clone(), f.vc.clone());
    if l.with_new {
        for h in 0..l.n_kv {
            for r in 0..l.k {
                let dst = (h * l.cap + l.pos0 + r) * HD;
                let src = (h * l.k + r) * HD;
                kc[dst..dst + HD].copy_from_slice(&f.kn[src..src + HD]);
                vc[dst..dst + HD].copy_from_slice(&f.vn[src..src + HD]);
            }
        }
    }
    (kc, vc)
}

fn first_diff(a: &[u16], b: &[u16]) -> Option<(usize, u16, u16)> {
    a.iter().zip(b).enumerate().find(|(_, (x, y))| x != y).map(|(i, (x, y))| (i, *x, *y))
}

/// Run a batch of launches in ONE probe process, and check every one.
fn check_all(launches: &[Launch]) {
    let c = attn_ref::score_scale(HD);
    let mut input = Vec::new();
    let mut fixtures = Vec::new();
    for (i, l) in launches.iter().enumerate() {
        let f = fixture(l, 0x9e37_79b9 ^ (i as u64 * 0x1000_0001));
        let new = l.with_new.then_some((f.kn.as_slice(), f.vn.as_slice()));
        input.extend(encode(l, c, &f.q, &f.kc, &f.vc, new));
        fixtures.push(f);
    }
    let bytes = run(&input);
    let mut at = 0;
    for (l, f) in launches.iter().zip(&fixtures) {
        let len = 2 * (l.n_kv * l.g * l.k * HD + 2 * l.n_kv * l.cap * HD) + 4 * l.n_kv;
        let got = decode(l, &bytes[at..at + len]);
        at += len;
        let (kc, vc) = expected_cache(l, f);
        let tag = format!(
            "G={} unit={} L={}..{} k={} grid={} s_stride={} new={} seed={}",
            l.g,
            l.unit,
            l.pos0 + 1,
            l.pos0 + l.k,
            l.k,
            l.grid_x,
            l.s_stride,
            l.with_new,
            l.seed
        );
        assert_eq!(first_diff(&got.kc, &kc), None, "{tag}: K cache after the call");
        assert_eq!(first_diff(&got.vc, &vc), None, "{tag}: V cache after the call");
        assert!(got.tickets.iter().all(|&t| t == 0), "{tag}: tickets not reset: {:?}", got.tickets);
        let want = attn_ref::attend(
            &f.q,
            &Kv { k: &kc, v: &vc, cap: l.cap },
            l.n_kv,
            l.g,
            l.pos0,
            l.k,
            c,
            l.unit,
        );
        if let Some((i, x, y)) = first_diff(&got.out, &want) {
            let (hq, r, d) = (i / (l.k * HD), (i / HD) % l.k, i % HD);
            panic!("{tag}: output differs at query head {hq}, row {r}, dim {d}: kernel {x:#06x}, replay {y:#06x}");
        }
    }
    assert_eq!(at, bytes.len(), "trailing probe output");
}

fn launch(g: usize, unit: usize, pos0: usize, k: usize, frozen: bool, with_new: bool, seed: u64) -> Launch {
    let lmax = pos0 + k;
    let cap = lmax.div_ceil(SPLIT) * SPLIT + if frozen { SPLIT } else { 0 };
    let eager = lmax.div_ceil(SPLIT);
    let s_stride = cap / SPLIT;
    Launch {
        g,
        unit,
        n_kv: 2,
        cap,
        pos0,
        k,
        grid_x: if frozen { s_stride } else { eager },
        s_stride,
        with_new,
        seed,
    }
}

/// Every row length the lead names as an edge, under both unit sizes and
/// both group sizes, decode and verify, eager grid. Rows of a verify call
/// straddle the 256 boundary at 255..262 and 253..260.
#[test]
fn the_kernel_text_returns_the_replay_bits() {
    let mut ls = Vec::new();
    for g in [4, 5] {
        for unit in [64, 32] {
            for &(pos0, k) in &[
                (0, 1),
                (6, 1),
                (7, 1),
                (8, 1),
                (62, 1),
                (63, 1),
                (64, 1),
                (254, 1),
                (255, 1),
                (256, 1),
                (519, 1),
                (0, 8),
                (0, 3),
                (252, 8),
                (254, 3),
                (517, 8),
            ] {
                ls.push(launch(g, unit, pos0, k, false, false, 0));
            }
        }
    }
    check_all(&ls);
}

/// The two things a frozen grid and a permuted finishing order exercise: CTAs
/// with no work still draw their ticket, and whichever CTA arrives last folds
/// the same splits in the same order. Plus the fresh-row write.
#[test]
fn a_frozen_grid_any_order_and_the_fresh_rows_give_the_same_bits() {
    let mut ls = Vec::new();
    for g in [4, 5] {
        for &(pos0, k) in &[(0, 1), (255, 1), (300, 8), (510, 3)] {
            for seed in [0u64, 7, 0xdead_beef] {
                ls.push(launch(g, 64, pos0, k, true, false, seed));
            }
            ls.push(launch(g, 64, pos0, k, false, true, 3));
            ls.push(launch(g, 32, pos0, k, true, true, 11));
        }
    }
    check_all(&ls);
}

/// The k rows of one verify call equal k decode calls, on the kernel itself
/// rather than through the replay.
#[test]
fn verify_rows_equal_decode_calls_on_the_kernel() {
    let c = attn_ref::score_scale(HD);
    for g in [4, 5] {
        let (pos0, k) = (250, 8);
        let lv = launch(g, 64, pos0, k, false, false, 5);
        let f = fixture(&lv, 42);
        let mut input = encode(&lv, c, &f.q, &f.kc, &f.vc, None);
        let mut singles = Vec::new();
        for r in 0..k {
            let l1 = Launch { pos0: pos0 + r, k: 1, grid_x: (pos0 + r + 1).div_ceil(SPLIT), ..lv };
            let mut q1 = Vec::new();
            for hq in 0..lv.n_kv * g {
                let o = (hq * k + r) * HD;
                q1.extend_from_slice(&f.q[o..o + HD]);
            }
            input.extend(encode(&l1, c, &q1, &f.kc, &f.vc, None));
            singles.push(l1);
        }
        let bytes = run(&input);
        let lenof = |l: &Launch| 2 * (l.n_kv * l.g * l.k * HD + 2 * l.n_kv * l.cap * HD) + 4 * l.n_kv;
        let mut at = lenof(&lv);
        let verify = decode(&lv, &bytes[..at]);
        for (r, l1) in singles.iter().enumerate() {
            let one = decode(l1, &bytes[at..at + lenof(l1)]);
            at += lenof(l1);
            for hq in 0..lv.n_kv * g {
                let a = &verify.out[(hq * k + r) * HD..(hq * k + r + 1) * HD];
                let b = &one.out[hq * HD..(hq + 1) * HD];
                assert_eq!(a, b, "G={g}: verify row {r}, head {hq}, differs from its decode call");
            }
        }
    }
}

fn ex2_kernel(xs: &[f32]) -> Vec<f32> {
    let mut v = Vec::new();
    u32s(&mut v, &[1, xs.len() as u32]);
    for x in xs {
        v.extend_from_slice(&x.to_le_bytes());
    }
    run(&v).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// `attn_ex2` as compiled from the kernel text, against `ex2_fixed`, on the
/// edges and on a dense sweep of its domain.
#[test]
fn the_kernel_exp2_is_the_replay_exp2() {
    let mut xs = vec![
        f32::NEG_INFINITY,
        -126.0,
        f32::from_bits((-126.0f32).to_bits() - 1), // just above -126
        -125.5,
        -1.0e-45,
        -0.0,
        0.0,
        -1.0,
        -0.5,
        -f32::EPSILON,
        f32::NAN,
        -f32::NAN,
    ];
    let mut s = 0x1234_5678u64;
    for i in 0..200_000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 40) as f32 / (1u64 << 24) as f32;
        xs.push(if i % 2 == 0 { -130.0 * u } else { -4.0 * u });
    }
    let got = ex2_kernel(&xs);
    for (x, y) in xs.iter().zip(&got) {
        let want = attn_ref::ex2_fixed(*x);
        if x.is_nan() {
            assert!(y.is_nan(), "ex2(NaN) must stay NaN");
        } else {
            assert_eq!(y.to_bits(), want.to_bits(), "ex2({x:e}) kernel {y:e} replay {want:e}");
        }
    }
}

/// The same text under UBSan, aborting on the first report: a signed left
/// shift of a negative exponent in `attn_ex2` computes the right bits on
/// every machine this repository uses and is still undefined in C++17, so no
/// value test can see it. One sweep of exp2 and launches of both unit sizes.
#[test]
fn the_kernel_text_has_no_undefined_behaviour() {
    let exe = build("ubsan", &["-fsanitize=undefined", "-fno-sanitize-recover=all"]);
    let xs: Vec<f32> = (0..20_000).map(|i| -130.0 * i as f32 / 20_000.0).collect();
    let mut v = Vec::new();
    u32s(&mut v, &[1, xs.len() as u32]);
    for x in &xs {
        v.extend_from_slice(&x.to_le_bytes());
    }
    let got = run_with(&exe, &v);
    assert_eq!(got.len(), 4 * xs.len());
    let c = attn_ref::score_scale(HD);
    for (g, unit) in [(4, 64), (5, 32)] {
        let l = launch(g, unit, 300, 8, true, true, 9);
        let f = fixture(&l, 1);
        let input = encode(&l, c, &f.q, &f.kc, &f.vc, Some((&f.kn, &f.vn)));
        let out = decode(&l, &run_with(&exe, &input));
        let (kc, vc) = expected_cache(&l, &f);
        let want = attn_ref::attend(&f.q, &Kv { k: &kc, v: &vc, cap: l.cap }, l.n_kv, g, l.pos0, l.k, c, unit);
        assert_eq!(out.out, want, "G={g} unit={unit} under UBSan");
    }
}
