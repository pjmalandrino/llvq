//! `tv_q4_hs` returns `tv_q4_hh`'s bits, at every width a Qwen3-32B needs.
//!
//! `tv_q4_hs` (`kernels/tv_q4_wide.cu`) stages the int4 matvec's activation as
//! f16 bits instead of f32: 51,200 B at d_in 25,600, where `tv_q4_hh` asks
//! 102,400 and a 99 KiB card offers 101,376. The claim is that this changes
//! no bit. `host_tv_q4_wide.cpp` executes both kernels on the CPU, entry
//! points included, and returns each lane's partial sum before `warp_sum`;
//! both kernels call the same `warp_sum` on the card, so equal partials mean
//! equal outputs. The host has no shared-memory ceiling, so `tv_q4_hh` runs
//! here at 25,600 and 31,744 too.
//!
//! The same harness writes `tests/fixtures/q4-wide-hashes.txt`: the sha256 of
//! `tv_q4_hs`'s output on synthetic records the card draws identically
//! (`fused::int4_synthetic`), which `fusedrun`'s `LLVQ_INT4_WIDE_CHECK`
//! compares the card's bytes against.

// `llvq-cuda` is a dev-dependency on macOS only (see `llvq-llm/Cargo.toml`): on a
// Linux CPU build it would pull `cudarc`, whose build script needs `nvcc`.
// Under `--features cuda` it is a normal dependency, so the file runs there.
#![cfg(any(target_os = "macos", feature = "cuda"))]

use llvq_cuda::sha256::sha256_hex;
use llvq_llm::fused::{
    int4_synthetic, parse_int4_wide_fixture, FusedInt4, INT4_WIDE_CARD_CASES, INT4_WIDE_KERNEL_NAME,
    Q4_WIDE_CU_EMBED,
};
use std::io::Write;
use std::process::{Command, Stdio};

fn harness() -> std::path::PathBuf {
    static EXE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    EXE.get_or_init(|| {
        let out = std::env::temp_dir().join("llvq_host_tv_q4_wide");
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
        let st = Command::new("clang++")
            .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"])
            .arg(dir.join("host_tv_q4_wide.cpp"))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("clang++ is on PATH");
        assert!(st.success(), "tv_q4_wide.cu does not compile as host C++");
        out
    })
    .clone()
}

/// What the harness returns for one record.
struct Run {
    hh: Vec<f32>,
    hs: Vec<f32>,
    y: Vec<u16>,
}

/// Run both kernels on `rec`, the activation `x` placed `x_off` elements into
/// a buffer whose first `x_off` values are 1.0, so a kernel that ignored the
/// offset reads ones instead of the activation.
fn run(rec: &FusedInt4, x: &[u16], x_off: usize) -> Run {
    let gpr = rec.d_in.div_ceil(rec.group);
    let mut fx: Vec<u8> = Vec::new();
    for v in [rec.d_out as u32, rec.d_in as u32, gpr as u32, x_off as u32] {
        fx.extend(v.to_le_bytes());
    }
    fx.extend(&rec.packed);
    for v in rec.scales.iter().chain(&rec.biases) {
        fx.extend(v.to_le_bytes());
    }
    for _ in 0..x_off {
        fx.extend(0x3c00u16.to_le_bytes());
    }
    for v in x {
        fx.extend(v.to_le_bytes());
    }
    let mut ch = Command::new(harness())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("the harness runs");
    ch.stdin.take().expect("piped").write_all(&fx).expect("fixture written");
    let out = ch.wait_with_output().expect("the harness exits");
    assert!(out.status.success(), "the harness failed on {}", rec.name);
    let parts = rec.d_out * 32;
    assert_eq!(out.stdout.len(), 2 * parts * 4 + rec.d_out * 2, "{}", rec.name);
    let f = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let hh = f(&out.stdout[..parts * 4]);
    let hs = f(&out.stdout[parts * 4..2 * parts * 4]);
    let y = out.stdout[2 * parts * 4..]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Run { hh, hs, y }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every int4 activation width of the served 4B, 8B and 14B, the 32B's three
/// (5,120, 8,192, 25,600), 25,344 (the widest whose f32 staging fits a 99 KiB
/// card) and 31,744.
const WIDTHS: [usize; 11] =
    [2_560, 4_096, 5_120, 8_192, 9_728, 12_288, 17_408, 25_344, 25_352, 25_600, 31_744];

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and runs eleven widths, run in release")]
fn the_f16_staged_kernel_is_tv_q4_hh_lane_for_lane() {
    for (i, &d_in) in WIDTHS.iter().enumerate() {
        let (rec, x) = int4_synthetic(16, d_in, 0x100 + i as u64);
        for x_off in [0usize, 37] {
            let r = run(&rec, &x, x_off);
            assert_eq!(
                bits(&r.hs),
                bits(&r.hh),
                "d_in {d_in}, x_off {x_off}: a lane of tv_q4_hs differs from tv_q4_hh"
            );
        }
    }
}

/// The identity above would hold for two kernels that agreed on garbage. So
/// the output is also held against an f64 dot of the reader's dequantization,
/// `f16(scale) * q + f16(bias)`, within what f32 accumulation and the f16
/// store allow.
#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++, run in release")]
fn the_f16_staged_kernel_computes_the_dot_product() {
    for &d_in in &[5_120usize, 25_600] {
        let (rec, x) = int4_synthetic(16, d_in, 0x77);
        let r = run(&rec, &x, 0);
        let gpr = d_in.div_ceil(rec.group);
        for row in 0..rec.d_out {
            let (mut dot, mut mag) = (0.0f64, 0.0f64);
            for (c, &xc) in x.iter().enumerate() {
                let i = row * d_in + c;
                let q = (rec.packed[i / 2] >> (4 * (i % 2))) & 0xf;
                let g = row * gpr + c / rec.group;
                let s = half::f16::from_bits(rec.scales[g]).to_f32();
                let b = half::f16::from_bits(rec.biases[g]).to_f32();
                let w = (s * q as f32 + b) as f64;
                let xv = half::f16::from_bits(xc).to_f64();
                dot += w * xv;
                mag += (w * xv).abs();
            }
            let got = half::f16::from_bits(r.y[row]).to_f64();
            let tol = 1e-5 * mag + dot.abs() * 1e-3 + 1e-3;
            assert!(
                (got - dot).abs() <= tol,
                "d_in {d_in} row {row}: {got} against {dot} (tolerance {tol:.3e})"
            );
        }
    }
}

const FIXTURE: &str = "tests/fixtures/q4-wide-hashes.txt";

/// The committed fixture is the host's `tv_q4_hs` output on the card cases,
/// and on those cases `tv_q4_hh` agrees lane for lane.
/// `LLVQ_Q4_WIDE_FIXTURE_WRITE=1` rewrites it instead of checking.
#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and runs 256-row records, run in release")]
fn the_card_fixture_is_the_host_output() {
    let mut text = String::from(
        "# tv_q4_hs on the host, one line per card case: d_out, d_in, seed, x_off, sha256\n\
         # of the f16 output (little-endian), on fused::int4_synthetic(d_out, d_in, seed).\n\
         # Written by tests/proj_q4_wide.rs (LLVQ_Q4_WIDE_FIXTURE_WRITE=1), compiled into\n\
         # llvq-llm and read by fusedrun's LLVQ_INT4_WIDE_CHECK.\n",
    );
    for (d_out, d_in, seed, x_off) in INT4_WIDE_CARD_CASES {
        let (rec, x) = int4_synthetic(d_out, d_in, seed);
        let r = run(&rec, &x, x_off);
        assert_eq!(bits(&r.hs), bits(&r.hh), "{d_out}x{d_in} {seed:#x} at {x_off}");
        let ybytes: Vec<u8> = r.y.iter().flat_map(|v| v.to_le_bytes()).collect();
        text.push_str(&format!("{d_out} {d_in} {seed:#x} {x_off} {}\n", sha256_hex(&ybytes)));
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    if std::env::var("LLVQ_Q4_WIDE_FIXTURE_WRITE").ok().as_deref() == Some("1") {
        std::fs::write(&path, &text).expect("fixture written");
        eprintln!("wrote {}", path.display());
        return;
    }
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{FIXTURE} is missing ({e}); llvq-llm compiles it in"));
    assert_eq!(
        parse_int4_wide_fixture(&committed).expect("the committed fixture parses"),
        parse_int4_wide_fixture(&text).expect("the host's own lines parse"),
        "{FIXTURE} is not what tv_q4_hs computes on the host"
    );
}

/// The compiled-in fixture covers every card case and nothing else.
#[test]
fn the_compiled_in_fixture_covers_every_card_case() {
    let m = parse_int4_wide_fixture(llvq_llm::fused::INT4_WIDE_FIXTURE).expect("it parses");
    for c in INT4_WIDE_CARD_CASES {
        let (d_out, d_in, seed, x_off) = c;
        assert!(m.contains_key(&(d_out, d_in, seed, x_off)), "no fixture line for {c:?}");
    }
    assert_eq!(m.len(), INT4_WIDE_CARD_CASES.len());
}

/// The name the runtime looks up is the one the unit defines.
#[test]
fn the_wide_kernel_name_matches_its_source() {
    assert!(Q4_WIDE_CU_EMBED.contains(&format!("__global__ void {INT4_WIDE_KERNEL_NAME}(")));
    assert!(Q4_WIDE_CU_EMBED.contains("#ifndef LLVQ_Q4_GROUP"), "the composition guard moved");
}

/// Two draws from the same seed are the same bytes, and two seeds differ: the
/// card and the host must draw the very record the fixture hashed.
#[test]
fn the_synthetic_record_is_a_function_of_its_seed() {
    let (a, xa) = int4_synthetic(16, 5_120, 0x42);
    let (b, xb) = int4_synthetic(16, 5_120, 0x42);
    let (c, _) = int4_synthetic(16, 5_120, 0x43);
    assert_eq!((&a.packed, &a.scales, &a.biases, &xa), (&b.packed, &b.scales, &b.biases, &xb));
    assert_ne!(a.packed, c.packed);
    assert_eq!(a.packed.len(), 16 * 5_120 / 2);
    assert_eq!(a.scales.len(), 16 * 40);
}
