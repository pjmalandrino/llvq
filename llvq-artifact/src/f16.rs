//! IEEE 754 binary16 on the host, with no `half` crate.
//!
//! `llvq-cuda` and `llvq-metal` each carried an identical copy until
//! 2026-11-08 (audit batch 10); both re-export these under their old paths.
//! `crate::sealed::f16_to_f32` is the narrower reader the format uses.

/// f32 → IEEE 754 binary16 bits, round to nearest even.
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        // Inf / NaN.
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e16 = exp - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00; // overflow to inf
    }
    if e16 <= 0 {
        // Subnormal (or zero): shift the implicit-1 mantissa down.
        if e16 < -10 {
            return sign;
        }
        let man = man | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = 1u32 << (shift - 1);
        let mut v = man >> shift;
        // Round to nearest even.
        let rem = man & ((1 << shift) - 1);
        if rem > half || (rem == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = (sign as u32) | ((e16 as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && v & 1 == 1) {
        v += 1; // may carry into the exponent — that is correct rounding
    }
    v as u16
}

/// binary16 bits → f64. Exact: every binary16 is a binary64.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x3ff) as f64;
    sign * match exp {
        0 => man * (-24f64).exp2(),
        31 => {
            if man == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        e => (1.0 + man / 1024.0) * ((e as f64) - 15.0).exp2(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every binary16 pattern that is a number round-trips through f64 and
    /// back, NaNs stay NaNs, and ties round to even.
    #[test]
    fn every_pattern_round_trips() {
        for h in 0..=u16::MAX {
            let x = f16_to_f64(h);
            if x.is_nan() {
                assert!(f16_to_f64(f16_bits(x as f32)).is_nan(), "{h:#06x}");
            } else {
                assert_eq!(f16_bits(x as f32), h, "{h:#06x}");
            }
        }
        // 1 + 2^-11 is halfway between 1 and the next binary16: even wins.
        assert_eq!(f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f16_bits(65520.0), 0x7c00, "past the largest finite value");
    }
}
