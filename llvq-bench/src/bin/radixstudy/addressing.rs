//! How a lane finds its block, priced two ways, and the kernel accounting
//! that turns a block width into the b/weight the CUDA bench streams.

use super::*;

// ---------------------------------------------------------------------------
// Addressing: the two ways a lane finds its block, priced side by side
// ---------------------------------------------------------------------------

/// One variant's group accumulator, carrying **both** addressing modes at
/// once: they see the same block sequence and differ only in what a closed
/// group costs, so pricing them separately would be two sweeps of the file for
/// no reason.
///
/// A partial group — one closed early by a matrix boundary — is charged the
/// way each mode really behaves, and the two disagree:
///
/// * `Grp32Max` pays **32 full lanes** regardless. The stride is uniform and
///   the padding lanes are addressable whether or not they hold a block; this
///   is also what `E1c` does, padding its trailing group with origin blocks
///   and charging them (`bin/rtbits:245-252`).
/// * `WarpScan` pays only the widths it actually holds, rounded to a word.
///   Nothing is bit-sliced across lanes here, so an absent block occupies no
///   bits.
///
/// The distinction is inert on both sealed artifacts as they stand — **measured**
/// by a header scan of each: every matrix's block count is `d_out · (d_in/24)`
/// with `d_out` a multiple of 32 (1024, 2560, 4096, 9728, 12288), so no group
/// is ever partial. That is a property of those two files, not of the format,
/// which is exactly why the sweep *counts* partial groups instead of assuming
/// there are none. Pinned for the 4B by
/// [`tests::the_sealed_4b_reproduces_the_published_class_major_verdict`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct Group {
    /// Lanes filled in the group being built.
    pub(super) filled: u64,
    /// Widest block of the group being built.
    gmax: u64,
    /// Sum of widths of the group being built.
    gsum: u64,
    /// Total bits over closed groups, `Grp32Max`.
    pub(super) max_bits: u64,
    /// Total bits over closed groups, `WarpScan`.
    pub(super) scan_bits: u64,
    /// Groups closed, partial ones included.
    pub(super) groups: u64,
    /// Groups closed with fewer than 32 lanes.
    pub(super) partial: u64,
}

impl Group {
    pub(super) fn push(&mut self, w: u64) {
        self.gmax = self.gmax.max(w);
        self.gsum += w;
        self.filled += 1;
        if self.filled == GROUP {
            self.close();
        }
    }

    /// Close the group being built, if any. Idempotent on an empty one, so a
    /// matrix boundary that happens to fall on a group boundary costs nothing
    /// and counts nothing.
    pub(super) fn close(&mut self) {
        if self.filled == 0 {
            return;
        }
        self.max_bits += GROUP * self.gmax.div_ceil(8) * 8 + BASE_OFFSET_BITS;
        self.scan_bits += self.gsum.div_ceil(WORD_BITS) * WORD_BITS + BASE_OFFSET_BITS;
        self.groups += 1;
        self.partial += u64::from(self.filled < GROUP);
        self.filled = 0;
        self.gmax = 0;
        self.gsum = 0;
    }

    pub(super) fn bits(&self, mode: Mode) -> u64 {
        match mode {
            Mode::Grp32Max => self.max_bits,
            Mode::WarpScan => self.scan_bits,
        }
    }
}

/// How a lane finds its own block inside a group of 32.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    /// Uniform byte stride at the group's widest block.
    Grp32Max,
    /// Exact widths, offsets by warp prefix sum.
    WarpScan,
}

impl Mode {
    pub(super) const ALL: [Mode; 2] = [Mode::Grp32Max, Mode::WarpScan];
    pub(super) fn tag(self) -> &'static str {
        match self {
            Mode::Grp32Max => "max",
            Mode::WarpScan => "scan",
        }
    }
}

/// In which order groups of 32 are formed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Order {
    /// Class histogram, ascending class id — the 2026-08-12 column. Groups are
    /// class-homogeneous almost everywhere, so the stride is as tight as it
    /// can ever be. **Optimistic by construction**; kept as the control.
    ClassMajor,
    /// 32 consecutive blocks of the file, groups closed at every matrix
    /// boundary — what a kernel reads.
    FileOrder,
}

impl Order {
    fn tag(self) -> &'static str {
        match self {
            Order::ClassMajor => "CM",
            Order::FileOrder => "FO",
        }
    }
}

/// Column label of one (order, mode) pair — the four combinations, named the
/// same way everywhere they are printed.
pub(super) fn col(o: Order, m: Mode) -> String {
    format!("{}/{}", o.tag(), m.tag())
}

// ---------------------------------------------------------------------------
// The kernel accounting
// ---------------------------------------------------------------------------

/// The shapes the kernel accounting divides by. Mirrors `bin/rtbits`' `Shapes`
/// and `planesbench`' `Mat`: `weights` is **every** weight including the tail
/// columns, `tail_weights` the `KeepExact` columns kept as `f32` on the
/// device, `rows` the `f32` row scales.
///
/// Read from the file when there is one, so the 8B arm of T2bis is priced on
/// the 8B rather than on remembered 4B constants — which is a real trap: the
/// same stream is 3.05 b/weight on one and something else on the other.
/// [`tests::the_sealed_4b_reproduces_the_published_class_major_verdict`] pins
/// that the 4B file reproduces the constants below exactly.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct Shapes {
    pub(super) weights: u64,
    pub(super) tail_weights: u64,
    pub(super) rows: u64,
    pub(super) blocks: u64,
}

impl Shapes {
    /// Qwen3-4B, the published `leech1c12` file. Same constants as `rtbits`'
    /// acceptance test.
    pub(super) const fn qwen3_4b() -> Self {
        Self {
            weights: 3_633_315_840,
            tail_weights: 16_957_440,
            rows: 1_105_920,
            blocks: 150_681_600,
        }
    }

    pub(super) fn push(&mut self, d_out: usize, d_in: usize) {
        self.weights += (d_out * d_in) as u64;
        self.tail_weights += (d_out * (d_in % DIM)) as u64;
        self.rows += d_out as u64;
        self.blocks += (d_out * (d_in / DIM)) as u64;
    }

    /// Kernel-accounting b/weight of a stream costing `bits_per_block`: the
    /// `f32` tail and row scales join the numerator, and the denominator is
    /// every weight. The one grandeur the 2.6 criterion is stated in.
    pub(super) fn kernel_bpw(&self, bits_per_block: f64) -> f64 {
        let side = (self.tail_weights + self.rows) * 32;
        (bits_per_block * self.blocks as f64 + side as f64) / self.weights as f64
    }

    /// The inverse: what a b/weight threshold is worth in bits per block on
    /// **these** shapes. Printed next to the pre-registration's own figures so
    /// a rounding cannot pass for a disagreement.
    pub(super) fn bits_for(&self, bpw: f64) -> f64 {
        let side = (self.tail_weights + self.rows) * 32;
        (bpw * self.weights as f64 - side as f64) / self.blocks as f64
    }
}

/// Kernel b/weight on the 4B shapes — the fixed conversion the criterion was
/// written against, kept as a free function because the tests pin it.
pub(super) fn kernel_bpw(bits_per_block: f64) -> f64 {
    Shapes::qwen3_4b().kernel_bpw(bits_per_block)
}
