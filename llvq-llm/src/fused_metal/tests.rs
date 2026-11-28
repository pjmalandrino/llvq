//! The unit tests of the fused Metal module: the served tuning, the `LLVQ_METAL_KERNEL` names and `h2f`. One
//! file because they test the module through its own paths, as the old inline `mod tests` did.

use super::{h2f, kernel_named, MetalTuning, Q4_TILE_COLS, SERVED_TETRA, TETRA_PREFILL_MAX};
use crate::served::MetalKernels;

/// The default tuning is the served path of every config before
/// 2026-11-26, and the Metal configs' `lutd+h+pq` is the bundle arm of
/// `metal-nuit-2026-11-06` (M2, M8, M10) and nothing else: the matvec
/// `lutd`, the half-io twins, the Tetra and int4 prefill kernels at 8
/// rows, the served int4 tile, no prefetch, no threadgroup rotation.
#[test]
fn the_served_metal_kernels_are_the_measured_arm() {
    let d = MetalTuning::default();
    assert_eq!(
        d,
        MetalTuning {
            q4_tile_cols: Q4_TILE_COLS,
            tetra_kernel: SERVED_TETRA,
            q4_prefetch: false,
            half_io: false,
            prefill_rows: 1,
            q4_rows: false,
            rot_tg: false,
        }
    );
    assert_eq!(MetalTuning::of_served(MetalKernels::Default).unwrap(), d);
    assert_eq!(MetalTuning::parse_arm("lut").unwrap(), d);
    let t = MetalTuning::of_served(MetalKernels::LutdHPq).unwrap();
    assert_eq!(
        t,
        MetalTuning {
            tetra_kernel: "tv_tetra48_metal_lutd",
            half_io: true,
            prefill_rows: TETRA_PREFILL_MAX,
            q4_rows: true,
            ..d
        }
    );
    for name in MetalKernels::NAMES {
        let k = MetalKernels::parse(Some(name)).unwrap();
        assert_eq!(MetalTuning::of_served(k).unwrap(), MetalTuning::parse_arm(name).unwrap(), "{name}");
    }
    // The bench's other spellings still read, and a form is not a kernel.
    assert!(MetalTuning::parse_arm("lutd+h+pq+r").unwrap().rot_tg);
    assert_eq!(MetalTuning::parse_arm("lut+t8192").unwrap().q4_tile_cols, 8192);
    for bad in ["lutd+grp", "lutd+ker", "lutx", "lutd+zz", "lutd+t", "lutd+t100", "lutp+h"] {
        assert!(MetalTuning::parse_arm(bad).is_err(), "{bad:?} must be refused");
    }
}

/// Unset and `lut` serve the served matvec, each A/B name maps to its own
/// kernel, and an unknown name is refused by name instead of falling back
/// to the served kernel under a label it did not run.
#[test]
fn an_unknown_metal_kernel_is_refused() {
    assert_eq!(kernel_named(None).unwrap(), "tv_tetra48_metal_lut");
    assert_eq!(kernel_named(Some("lut")).unwrap(), "tv_tetra48_metal_lut");
    assert_eq!(kernel_named(Some("plain")).unwrap(), "tv_tetra48_metal");
    assert_eq!(kernel_named(Some("c3")).unwrap(), "tv_tetra48_metal_c3");
    // `ilp` names an 8-lane kernel, which the 32-lane grid would run past
    // the output.
    for bad in ["lutd", "", "LUT", "ilp"] {
        let e = kernel_named(Some(bad)).expect_err(bad).to_string();
        assert!(e.contains("LLVQ_METAL_KERNEL"), "{bad:?}: {e}");
    }
}

/// Every name the variable accepts, `validate` accepts too: the variable
/// does not pass through it.
#[test]
fn every_kernel_the_variable_names_passes_validate() {
    for v in [None, Some("lut"), Some("c3"), Some("ar"), Some("u2"), Some("plain")] {
        let tetra_kernel = kernel_named(v).unwrap();
        super::MetalTuning { tetra_kernel, ..Default::default() }
            .validate()
            .unwrap_or_else(|e| panic!("{v:?} names {tetra_kernel}, which validate refuses: {e}"));
    }
}

/// The IEEE definition of binary16, in f64, from the fields.
///
/// Not a second copy of `h2f`: this computes the VALUE from the exponent
/// and the significand, where `h2f` assembles a bit pattern. A shared
/// mistake would have to be a mistake about IEEE itself.
fn ieee_half(h: u16) -> Option<f64> {
    let s = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    match e {
        // Subnormal: no hidden bit, fixed exponent of -14.
        0 => Some(s * (m / 1024.0) * 2f64.powi(-14)),
        // inf / NaN, which the caller checks separately.
        31 => None,
        _ => Some(s * (1.0 + m / 1024.0) * 2f64.powi(e - 15)),
    }
}

/// All 65,536 patterns. The subnormal and inf arms were both wrong until
/// an audit of 2026-09-21, and the gate's Gaussian fixtures never drew one.
#[test]
fn h2f_widens_every_binary16_pattern() {
    let mut checked = 0u32;
    for b in 0u32..=0xffff {
        let h = b as u16;
        let got = h2f(h);
        match ieee_half(h) {
            Some(want) => {
                assert_eq!(
                    got,
                    want as f32,
                    "0x{b:04x}: h2f {got:e} against the IEEE value {want:e}"
                );
                checked += 1;
            }
            None => {
                // exp == 31: infinity when the significand is zero, NaN otherwise.
                if h & 0x3ff == 0 {
                    assert!(got.is_infinite(), "0x{b:04x} is an infinity");
                    assert_eq!(got.is_sign_negative(), h & 0x8000 != 0, "0x{b:04x} sign");
                } else {
                    assert!(got.is_nan(), "0x{b:04x} is a NaN");
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 65_536, "every pattern is judged");
}

/// The three patterns the earlier version got wrong, named.
#[test]
fn the_patterns_an_earlier_h2f_got_wrong() {
    // The smallest positive subnormal: 2^-24.
    assert_eq!(h2f(0x0001), 2f32.powi(-24));
    // The largest subnormal: 1023 * 2^-24.
    assert_eq!(h2f(0x03ff), 1023.0 * 2f32.powi(-24));
    // Positive infinity, which an earlier version returned as 65536.
    assert!(h2f(0x7c00).is_infinite() && h2f(0x7c00) > 0.0);
}
