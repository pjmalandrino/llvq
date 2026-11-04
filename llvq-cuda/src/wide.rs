//! Which kernel a width gets once it no longer fits one block's shared memory.
//!
//! Two kernels of the served path stage a whole activation in f32 shared
//! memory, `d_in * 4` bytes: `rot_apply` (the incoherence rotation) and
//! `tv_q4_h` / `tv_q4_hh` (the int4 projections). At the Qwen3-32B `down_proj`
//! width, 25,600, both ask 102,400 B, and the cards with 99 KiB of opt-in
//! (sm_86, sm_89, sm_120) offer 101,376. Each has an exact way out:
//!
//!   * `rot_apply_dev` (`kernels/rotate_wide.cu`) runs the same helpers on a
//!     device scratch: no shared memory at all;
//!   * `tv_q4_hs` (`llvq-llm/kernels/tv_q4_wide.cu`) stages the activation as
//!     its f16 bits and widens at use: `d_in * 2` bytes, 51,200 at 25,600.
//!
//! Both return the bits of the kernel they replace, and both are chosen here
//! **only where that kernel is refused**. That rule is the whole guarantee
//! that the served 4B, 8B and 14B keep today's kernels, and the tests below
//! pin it on L40S bounds.
//!
//! Portable for the reason [`crate::shared`] is: the callers are
//! `cfg(target_os = "linux")`, and the development machine has no card.

use crate::shared::{ceiling, plan, rot_bytes, Fit};
use crate::ROT_KMAX;

/// Which rotation kernel a width gets on this card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotKernel {
    /// `rot_apply`: the whole vector in shared memory, under the default
    /// allowance or after the opt-in. Today's kernel, kept wherever it runs.
    Shared(Fit),
    /// `rot_apply_dev`: staged in device memory. Chosen only when `Shared`
    /// is refused, that is when `n * 4` passes the card's ceiling.
    Device,
}

/// The rotation kernel for a width of `n`: [`RotKernel::Shared`] whenever
/// [`crate::shared::rot_plan`] accepts the width, [`RotKernel::Device`] only
/// where it refuses.
pub fn rot_kernel(n: usize, default_limit: usize, optin_limit: usize) -> RotKernel {
    match plan(rot_bytes(n), default_limit, optin_limit) {
        Some(fit) => RotKernel::Shared(fit),
        None => RotKernel::Device,
    }
}

/// Floats of device scratch `rot_apply_dev` / `rot_apply_rows_dev` need for
/// `n_rows` rows of a width `n = k * m`: one staged row each, plus
/// `(KMAX - k) * m` so a speculated read of `rot_mix`'s unrolled column loop
/// past `k` stays inside the allocation.
pub fn rot_scratch_floats(n_rows: usize, n: usize, m: usize, k: usize) -> usize {
    n_rows * n + ROT_KMAX.saturating_sub(k) * m
}

/// How the int4 matvec stages its activation for a record of `d_in`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int4Staging {
    /// `tv_q4_h` / `tv_q4_hh`: `d_in` f32. Today's kernels.
    F32,
    /// `tv_q4_hs`: `d_in` f16 bits, widened at use. Same operands in every
    /// FMA, so the same bits; half the bytes. Takes an **f16** activation
    /// only: narrowing an f32 one would change the arithmetic, so the runtime
    /// refuses it by name at such a record.
    F16,
}

impl Int4Staging {
    /// Dynamic shared bytes one launch asks for at `d_in`.
    pub fn bytes(self, d_in: usize) -> usize {
        match self {
            Int4Staging::F32 => d_in * 4,
            Int4Staging::F16 => d_in * 2,
        }
    }
}

/// The staging a record of `d_in` gets: F32 whenever its f32 staging fits the
/// card's ceiling, F16 only past it, `None` when neither fits.
pub fn int4_staging(d_in: usize, default_limit: usize, optin_limit: usize) -> Option<Int4Staging> {
    let top = ceiling(default_limit, optin_limit);
    if Int4Staging::F32.bytes(d_in) <= top {
        Some(Int4Staging::F32)
    } else if Int4Staging::F16.bytes(d_in) <= top {
        Some(Int4Staging::F16)
    } else {
        None
    }
}

/// The dynamic shared size each int4 entry point is loaded with.
///
/// `f32` is posed on `tv_q4_h` and `tv_q4_hh`, the widest F32-staged record;
/// `f16` on `tv_q4_hs`, the widest F16-staged one. `None` means that entry
/// point is not needed. Each goes through `Cuda::func_dynamic_shared`, which
/// poses the opt-in only past the default: `tv_q4_hs` at 51,200 B needs its
/// own, since that is past 49,152.
///
/// Before this, ONE size, the widest record in f32, was posed on both entry
/// points. A 32B with an int4 `down_proj` would ask 102,400 B there and fail
/// at load, new kernel or not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Int4EntryBytes {
    pub f32: Option<usize>,
    pub f16: Option<usize>,
}

/// [`Int4EntryBytes`] over a model's int4 records; refuses, naming the width,
/// a record no staging fits.
pub fn int4_entry_bytes(
    d_ins: impl IntoIterator<Item = usize>,
    default_limit: usize,
    optin_limit: usize,
) -> Result<Int4EntryBytes, String> {
    let mut e = Int4EntryBytes { f32: None, f16: None };
    for d in d_ins {
        let slot = match int4_staging(d, default_limit, optin_limit) {
            Some(Int4Staging::F32) => (&mut e.f32, Int4Staging::F32),
            Some(Int4Staging::F16) => (&mut e.f16, Int4Staging::F16),
            None => {
                return Err(format!(
                    "an int4 record of d_in {d} fits neither staging: {} B in f32, {} B in \
                     f16, against {} B on this card",
                    Int4Staging::F32.bytes(d),
                    Int4Staging::F16.bytes(d),
                    ceiling(default_limit, optin_limit)
                ))
            }
        };
        let b = slot.1.bytes(d);
        *slot.0 = Some(slot.0.map_or(b, |a| a.max(b)));
    }
    Ok(e)
}

/// The widths `bin/rotbench` checks on a card, with the seed of each draw.
///
/// Here rather than in the binary because the host test writes the fixture
/// the card compares against (`tests/fixtures/rot-hashes.txt`) and must draw
/// the same `(n, seed)` pairs. 12 widths of the identity prereg, plus 96 and
/// 24 that `rotbench` has always checked.
pub const CARD_WIDTHS: [(usize, u64, &str); 14] = [
    (2_560, 0x5, "Qwen3-4B hidden, k=5"),
    (4_096, 0x6, "Qwen3-4B o_proj, k=1, no mix"),
    (5_120, 0xa, "Qwen3-14B/32B hidden, k=5, m=1,024"),
    (6_144, 0xc, "Qwen3.8-27B o_proj input, k=3, m=2,048"),
    (8_192, 0xd, "Qwen3-32B o_proj input, k=1"),
    (9_728, 0x7, "Qwen3-4B down_proj, k=19"),
    (12_288, 0x8, "Qwen3-8B down_proj, k=3, on the default bound"),
    (17_408, 0x9, "Qwen3-14B down_proj, k=17, opt-in"),
    (25_600, 0xb, "Qwen3-32B down_proj, k=25: past the 99 KiB opt-in"),
    (31_744, 0xe, "widest odd factor, k=31"),
    (3_072, 0x4, "Qwen3-0.6B intermediate"),
    (1_024, 0x3, "narrow, k=1"),
    (96, 0x2, "small"),
    (24, 0x1, "one Leech block"),
];

/// The bytes the fixture's input hash covers, in this order, little-endian:
/// the `n` f16 bits of the activation, the sign words, the `KMAX * KMAX`
/// padded `small`, and `inv`. Hashing them first separates "the card drew
/// another input" from "the kernel computed other bits".
pub fn rot_input_bytes(bits: &[u16], words: &[u32], small: &[f32], inv: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(bits.len() * 2 + words.len() * 4 + small.len() * 4 + 4);
    for b in bits {
        v.extend(b.to_le_bytes());
    }
    for w in words {
        v.extend(w.to_le_bytes());
    }
    for s in small {
        v.extend(s.to_le_bytes());
    }
    v.extend(inv.to_le_bytes());
    v
}

/// The bytes of a rotation's f32 output, little-endian, as the fixture hashes them.
pub fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The parsed fixture: `(n, seed)` to `(input sha256, output sha256)`.
pub type RotFixture = std::collections::BTreeMap<(usize, u64), (String, String)>;

/// One fixture line per `(n, seed)`: `<n> <seed hex> <input sha256> <output sha256>`.
///
/// Refuses a malformed line or a duplicate pair by line number, so a fixture
/// that drifted fails before any launch.
pub fn parse_rot_fixture(text: &str) -> Result<RotFixture, String> {
    let mut m = std::collections::BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 4 {
            return Err(format!("fixture line {}: {} fields, expected 4", i + 1, f.len()));
        }
        let n: usize = f[0].parse().map_err(|e| format!("fixture line {}: n: {e}", i + 1))?;
        let seed = u64::from_str_radix(f[1].trim_start_matches("0x"), 16)
            .map_err(|e| format!("fixture line {}: seed: {e}", i + 1))?;
        for h in &f[2..] {
            if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("fixture line {}: {h:?} is not a sha256", i + 1));
            }
        }
        if m.insert((n, seed), (f[2].to_string(), f[3].to_string())).is_some() {
            return Err(format!("fixture line {}: ({n}, {seed:#x}) twice", i + 1));
        }
    }
    Ok(m)
}

/// The committed fixture, compiled in.
///
/// The runtime image holds binaries, `fetch-qtip.sh` and `configs/` only, so a
/// path read at run time would not exist on the card. The build stage copies
/// the whole uploaded tree, so the file is there at compile time, and a
/// missing one fails the free image build instead of a billed job.
pub const ROT_FIXTURE: &str = include_str!("../tests/fixtures/rot-hashes.txt");

#[cfg(test)]
mod tests {
    use super::*;

    // L40S measured (`fusedrun-14b-2026-08-17.txt`); A100 measured
    // (`f4-a100-2026-08-18.txt:95`); H200 from the CUDA guide's 227 KB for
    // compute capability 9.0, never read off a card here.
    const L40S: (usize, usize) = (49_152, 101_376);
    const A100: (usize, usize) = (49_152, 166_912);
    const H200: (usize, usize) = (49_152, 232_448);

    #[test]
    fn served_widths_keep_todays_rotation_and_only_the_32b_moves() {
        let (d, o) = L40S;
        assert_eq!(rot_kernel(2_560, d, o), RotKernel::Shared(Fit::Default));
        assert_eq!(rot_kernel(4_096, d, o), RotKernel::Shared(Fit::Default));
        assert_eq!(rot_kernel(5_120, d, o), RotKernel::Shared(Fit::Default));
        assert_eq!(rot_kernel(8_192, d, o), RotKernel::Shared(Fit::Default));
        assert_eq!(rot_kernel(9_728, d, o), RotKernel::Shared(Fit::Default));
        // On the default bound: `<=`, or the 8B would move.
        assert_eq!(rot_kernel(12_288, d, o), RotKernel::Shared(Fit::Default));
        // The 14B's down rotation: opt-in, never Device.
        assert_eq!(rot_kernel(17_408, d, o), RotKernel::Shared(Fit::OptIn));
        // 101,376 / 4 = 25,344: the widest the opt-in holds, on the bound.
        assert_eq!(rot_kernel(25_344, d, o), RotKernel::Shared(Fit::OptIn));
        assert_eq!(rot_kernel(25_600, d, o), RotKernel::Device);
        assert_eq!(rot_kernel(31_744, d, o), RotKernel::Device);
    }

    /// The selection is exactly `rot_plan`'s verdict, at every width: Shared
    /// where it accepts, Device where it refuses. No width a served object
    /// has can change kernel, whatever the card.
    #[test]
    fn device_is_chosen_exactly_where_rot_plan_refuses() {
        for (d, o) in [L40S, A100, H200, (49_152, 0)] {
            for n in (1..=40).map(|i| i * 1_024).chain([9_728, 12_288, 17_408, 25_344, 25_600]) {
                let shared = crate::shared::rot_plan(n, d, o);
                match (rot_kernel(n, d, o), shared) {
                    (RotKernel::Shared(f), Ok(g)) => assert_eq!(f, g, "{n}"),
                    (RotKernel::Device, Err(_)) => {}
                    (k, s) => panic!("{n} on ({d}, {o}): {k:?} against rot_plan {s:?}"),
                }
            }
        }
    }

    #[test]
    fn the_32b_needs_no_lift_on_cards_with_a_larger_opt_in() {
        for (d, o) in [A100, H200] {
            assert_eq!(rot_kernel(25_600, d, o), RotKernel::Shared(Fit::OptIn));
            assert_eq!(int4_staging(25_600, d, o), Some(Int4Staging::F32));
        }
    }

    #[test]
    fn the_scratch_covers_every_row_and_the_mix_tail() {
        // 32B down: k = 25, m = 1,024.
        assert_eq!(rot_scratch_floats(1, 25_600, 1_024, 25), 25_600 + 7 * 1_024);
        assert_eq!(rot_scratch_floats(4, 25_600, 1_024, 25), 4 * 25_600 + 7 * 1_024);
        assert_eq!(rot_scratch_floats(1, 32_768, 32_768, 1), 32_768 + 31 * 32_768);
        assert_eq!(rot_scratch_floats(1, 31 * 1_024, 1_024, 31), 31 * 1_024 + 1_024);
    }

    #[test]
    fn int4_moves_to_f16_staging_only_past_the_ceiling() {
        let (d, o) = L40S;
        for d_in in [2_560, 4_096, 5_120, 8_192, 9_728, 12_288, 17_408, 25_344] {
            assert_eq!(int4_staging(d_in, d, o), Some(Int4Staging::F32), "{d_in}");
        }
        assert_eq!(int4_staging(25_352, d, o), Some(Int4Staging::F16));
        assert_eq!(int4_staging(25_600, d, o), Some(Int4Staging::F16)); // 51,200 B
        assert_eq!(int4_staging(50_688, d, o), Some(Int4Staging::F16)); // on the bound
        assert_eq!(int4_staging(50_696, d, o), None);
        assert_eq!(int4_staging(60_000, d, o), None);
    }

    /// The pattern of `the_int4_staging_fits_the_card_at_every_sealed_size`
    /// (`llvq-llm/src/fused.rs`), per entry point, with a 32B shape added.
    #[test]
    fn each_int4_entry_point_gets_its_own_size_at_every_shape() {
        let (d, o) = L40S;
        // (hidden, o_proj d_in, intermediate) of Qwen3-4B, 8B and 14B: one
        // F32 size, today's, and nothing staged in f16.
        let sealed = [
            ((2_560, 4_096, 9_728), 38_912),
            ((4_096, 4_096, 12_288), 49_152),
            ((5_120, 5_120, 17_408), 69_632),
        ];
        for ((h, op, i), want) in sealed {
            let e = int4_entry_bytes([h, op, i], d, o).unwrap();
            assert_eq!(e, Int4EntryBytes { f32: Some(want), f16: None }, "{h}/{op}/{i}");
        }
        // Qwen3-32B on the L40S: v at 5,120 and o at 8,192 stay F32 (32,768 B,
        // under the default); down at 25,600 goes F16, 51,200 B, past the
        // default, so `tv_q4_hs` takes its own opt-in.
        let e = int4_entry_bytes([5_120, 8_192, 25_600], d, o).unwrap();
        assert_eq!(e, Int4EntryBytes { f32: Some(32_768), f16: Some(51_200) });
        assert_eq!(plan(51_200, d, o), Some(Fit::OptIn));
        assert_eq!(plan(32_768, d, o), Some(Fit::Default));
        // Order does not matter, and the f32 size is over the F32 records only.
        let e = int4_entry_bytes([25_600, 8_192, 5_120, 25_600], d, o).unwrap();
        assert_eq!(e, Int4EntryBytes { f32: Some(32_768), f16: Some(51_200) });
        // Qwen3-32B on the A100: everything F32, one opt-in at 102,400 B.
        let (da, oa) = A100;
        let e = int4_entry_bytes([5_120, 8_192, 25_600], da, oa).unwrap();
        assert_eq!(e, Int4EntryBytes { f32: Some(102_400), f16: None });
        assert_eq!(int4_entry_bytes([], d, o).unwrap(), Int4EntryBytes { f32: None, f16: None });
        let err = int4_entry_bytes([5_120, 60_000], d, o).unwrap_err();
        assert!(err.contains("60000") && err.contains("101376"), "{err}");
    }

    #[test]
    fn the_staged_bytes_are_two_or_four_a_column() {
        assert_eq!(Int4Staging::F32.bytes(25_600), 102_400);
        assert_eq!(Int4Staging::F16.bytes(25_600), 51_200);
    }

    #[test]
    fn a_malformed_fixture_is_refused_by_line() {
        let h = "a".repeat(64);
        assert!(parse_rot_fixture(&format!("2560 0x5 {h}\n")).is_err());
        assert!(parse_rot_fixture(&format!("2560 0x5 {h} {h}\n2560 0x5 {h} {h}\n")).is_err());
        assert!(parse_rot_fixture(&format!("2560 0x5 {h} xyz\n")).is_err());
        let m = parse_rot_fixture(&format!("# comment\n\n2560 0x5 {h} {h}\n")).unwrap();
        assert_eq!(m[&(2560, 5)], (h.clone(), h));
    }

    /// The fixture rotbench compiles in carries one line for every width it
    /// checks, and nothing else.
    #[test]
    fn the_compiled_in_fixture_covers_every_card_width() {
        let m = parse_rot_fixture(ROT_FIXTURE).expect("the committed fixture parses");
        for (n, seed, what) in CARD_WIDTHS {
            assert!(m.contains_key(&(n, seed)), "no fixture line for {what} ({n}, {seed:#x})");
        }
        assert_eq!(m.len(), CARD_WIDTHS.len(), "the fixture has lines no card width reads");
    }
}
