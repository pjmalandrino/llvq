//! The host structures the kernels read: one matrix's stream ([`HostStream`],
//! [`FusedMatrix`], [`FusedInt4`]), the byte accounting of its side arrays, and
//! the rotation tables shared by the projections that share an activation.

use llvq_quant::rotation::Rotation;

use super::FusedGroup;

/// Mirrors `LLVQ_ROT_KMAX` in `llvq-cuda/kernels/llvq_rot.cuh`.
///
/// Duplicated across a language boundary on purpose — the kernel is in another
/// crate that does not build on this machine — and pinned by
/// `the_kmax_constant_matches_the_kernel` on the CUDA side.
pub(super) const ROT_KMAX: usize = 32;

/// One matrix's payload in one runtime layout, as the words a kernel indexes.
///
/// An enum rather than optional fields, and that is the guarantee: a
/// `Planes14` stream **has no bases** — not an empty vector, no field at all —
/// so no code path can upload or launch with a bases array on the planes path.
/// The compiler enforces what a review would otherwise have to.
pub enum HostStream {
    Slot32 {
        /// `Slot32` payload as `u32` words, padded for the kernel's five-word
        /// read window past the last block.
        words: Vec<u32>,
        /// Byte offset of each group of 32 blocks, plus a final sentinel.
        bases: Vec<u32>,
    },
    Planes14 {
        /// The uniform 14-byte records as `u32` words, padded for the
        /// kernel's four-word read window past the last block.
        words: Vec<u32>,
    },
    Planes12x {
        /// The uniform 12-byte main-stream records as `u32` words. `12·b` is
        /// always word-aligned, so the three-word window needs no padding at
        /// all; the four spare bytes are packing symmetry with the other
        /// streams, not a correctness need.
        words: Vec<u32>,
        /// Matrix-wide block index of each exception, strictly ascending.
        exc_idx: Vec<u32>,
        /// One exact 14-byte `Planes14` record per exception, as `u32` words,
        /// padded for the four-word window `planes_fields` reads on it.
        exc_words: Vec<u32>,
        /// `d_out + 1` prefix offsets into `exc_idx`: output row `r` owns
        /// exceptions `row_exc[r] .. row_exc[r+1]`.
        ///
        /// This table is what lets the inference kernel keep one warp per
        /// output row and drop the bench kernel's memset-and-atomicAdd
        /// protocol — see `kernels/tv_planes12x_h.cu`. It exists because
        /// `exc_idx` is ascending and blocks are row-major, so a row's
        /// exceptions are contiguous; [`row_offsets`] asserts that rather
        /// than assuming it, since the kernel derives an exception's column
        /// as `b − row·nblocks` and a wrong row would read the wrong
        /// activation slice and produce finite, plausible, wrong numbers.
        row_exc: Vec<u32>,
    },
    Golay70 {
        /// The 9-byte main-stream records as `u32` words. The transcoder
        /// itself word-aligns the stream and appends one zero word so the
        /// kernel's three-word window exists for the last block — no extra
        /// packing pad here, unlike the plane streams.
        words: Vec<u32>,
        /// Matrix-wide block index of each exception, strictly ascending.
        exc_idx: Vec<u32>,
        /// One exact 14-byte `Planes14` record per exception, as `u32` words,
        /// padded for the four-word window `planes_fields` reads on it —
        /// the exception machinery is Planes12x's, verbatim.
        exc_words: Vec<u32>,
        /// `d_out + 1` prefix offsets into `exc_idx` — same contract, same
        /// [`row_offsets`] check, same reason as the `Planes12x` field: one
        /// warp per row, no memset, no atomic (`kernels/tv_golay70_h.cu`).
        row_exc: Vec<u32>,
    },
    Tetra48 {
        /// The 48-bit words as `u32`, **row-strided**: `stride_u32` words per
        /// output row, the row's six-byte records at its start and the rest
        /// zero. Every other stream here is flat over the matrix's blocks;
        /// this one is not, and the reason is alignment rather than taste —
        /// a six-byte record at `6·b` is u32-aligned only every second block,
        /// so `f1r_load`'s two-word window would read at a shifted phase on
        /// every row after the first. `llvq_artifact::tetra48` builds the
        /// stride, rounds it to eight bytes so the last block's window fits,
        /// and asserts that rather than assuming it.
        words: Vec<u32>,
        /// Words per row. The kernel's second argument; a stream whose stride
        /// disagrees with it returns finite, plausible, wrong numbers.
        stride_u32: u32,
    },
}

/// One projection, transcoded and ready to upload.
pub struct FusedMatrix {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub nblocks: usize,
    pub tail_w: usize,
    /// The payload, in the layout [`FusedModel::layout`](crate::fused::FusedModel::layout) names.
    pub stream: HostStream,
    /// The two gain centroids. One bit of gain is hard-coded in every decoder.
    pub gscale: [f32; 2],
    pub rscale: Vec<f32>,
    /// `d_out × tail_w`, row-major, in the rotated basis, as **binary16
    /// bits** — see [`tail_f16_bits`] for why the width changed and what it
    /// is aligned to.
    pub tail: Vec<u16>,
    /// Key into [`FusedModel::rotations`](crate::fused::FusedModel::rotations), or `None` in the natural basis.
    pub rotation: Option<RotKey>,
    /// Bytes this matrix costs at runtime — every device array the kernel
    /// reads: the payload, whatever addressing the layout carries beside it
    /// (`Slot32`'s bases, `Planes12x`'s exception table and row offsets), and
    /// [`matrix_side_bytes`] for the tail and the row scales.
    pub bytes: u64,
}

/// One `Int4G128` projection, uploaded as it is stored.
///
/// Deliberately **not** a variant of [`FusedMatrix`]. That struct carries a
/// `gscale`, a `rscale`, a `tail` and a `rotation`, and an int4 record has
/// none of the four: it is stored group-affine in the natural basis, never
/// GPTQ, never the rotation, never the lattice (`calib.rs:793-812`). Folding
/// it in would mean four `Option`s on the hot struct and a reader who cannot
/// tell which combinations are legal. A separate list says it once.
///
/// The weights are the file's own bytes. Nothing here decodes, rescales or
/// reorders them — `tv_q4_h` reads **`scale · q + bias`** off exactly these
/// three arrays, and `tests/proj_q4.rs` pins that arithmetic against
/// [`llvq_artifact::Int4Matrix::to_f32`] on the development machine.
///
/// 🕳️ This said `scale · (q − zero)` until 2026-09-10, which is the OTHER
/// affine convention — the one where the offset is subtracted from the
/// quantum before scaling. `embedquant` stores `bias = min` and the kernel
/// spells the two roundings out (`__fmul_rn` then `__fadd_rn`, never
/// contracted) precisely so a served row equals the row the file decodes to.
/// A doc naming the wrong convention is what a second implementation would
/// have been written against.
#[derive(Clone)]
pub struct FusedInt4 {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    /// Groups along `d_in`; `d_in / group` per output row.
    pub group: usize,
    /// `d_out · d_in / 2` bytes, row-major, **low nibble first**. A lattice
    /// index in the same file is packed MSB-first: two orders in one file, and
    /// a reader that assumed one convention for both would produce plausible,
    /// wrong weights.
    pub packed: Vec<u8>,
    /// One binary16 per group, `d_out × (d_in / group)`, row-major.
    pub scales: Vec<u16>,
    /// Same shape, same order.
    pub biases: Vec<u16>,
    /// Bytes this matrix costs at runtime: the three arrays, and nothing else.
    /// There is no tail and no row scale to add.
    pub bytes: u64,
}

/// Bytes one `TailPolicy::KeepExact` weight costs on the device.
///
/// Two since 2026-08-09 (lot A7a), four before it. Named rather than spelled
/// so [`matrix_side_bytes`] and the tail upload cannot drift apart, and so a
/// mutation of the width has exactly one site.
pub const TAIL_BYTES: u64 = 2;

/// Bytes one row scale costs on the device. Still `f32`, and deliberately:
/// `rscale[row]` multiplies the whole accumulated block sum, so its rounding
/// is a *relative* error on every quantized weight of the row at once, where
/// the tail's is an absolute error on `d_in mod 24` of them. Different
/// exposure, different decision — and 1,105,920 rows on the 4B is 4.4 MB, a
/// tenth of what the tail was costing.
pub const ROW_SCALE_BYTES: u64 = 4;

/// Narrow a `KeepExact` tail to the precision the **reference** model holds it
/// at, and return its binary16 bits.
///
/// ## The only question that mattered here, and its answer in the code
///
/// The tail is full precision *by design*: `TailPolicy::KeepExact` leaves the
/// `d_in mod 24` unquantized columns alone so that they never contribute an
/// error of their own. Rounding them looks, at first sight, like spending
/// quality for bytes — which this dossier has refused for much better trades.
/// It is not, and the reason is that the arm the fused path is *compared to*
/// already rounds them:
///
/// * `bin/fusedrun` runs both arms at `DType::F16` (`bin/fusedrun/main.rs`, the
///   `let dtype` beside the device);
/// * its dense arm is `crate::sealed::load(path, dtype, device)`, which takes
///   `llvq_artifact::decode_matrix`'s `Vec<f32>` — tail columns restored into
///   it, then un-rotated — and calls `.to_dtype(dtype)` on the whole matrix
///   (`crate::sealed`). Nothing exempts the tail: it is a column range of a
///   tensor that gets narrowed to binary16 in one move;
/// * `bin/mmlu` defaults to F16 and `bin/ppl` prints the dtype it used, and
///   **both score through `sealed::load`** — the fused path is a matvec and
///   `bin/ppl` keeps the dense path for scoring. So no published perplexity
///   or MMLU figure reads this array at all; what it can move is
///   `bin/fusedrun`'s token agreement, and nothing else.
///
/// The `f32` that was here before was never a precision decision for the
/// card. It is the *file's* storage width — `calib.rs` narrows the tail to
/// `f32` before writing it, precisely so the file and the evaluated model are
/// one object, and `llvq_artifact::format` reads it back from 32 bits — and
/// the loader carried that width forward because it had no reason not to.
/// Keeping 24 mantissa bits resident bought precision the reference does not
/// have, at 2 bytes a weight.
///
/// ## Size of the rounding, against the rounding already there
///
/// Half an ulp of binary16 is `2⁻¹¹ ≈ 4.9e-4` relative. If tail and block
/// columns carry comparable energy, the tail's share of a row's dot product
/// scales as `√(tail_w / d_in)`, so the perturbation on `y` is about
/// `4.9e-4 · √(tail_w/d_in)`: **7.9 %** of it on the 4B's `d_in = 2560`
/// (tail 16), **2.9 %** on its `down_proj` (`d_in = 9728`, tail 8). The
/// kernel then narrows `y` itself to binary16 — a `4.9e-4` relative rounding,
/// i.e. 12× to 34× *larger*. ⚠️ Labelled honestly: that is an **estimate**
/// under an energy assumption, not a measurement, and the only thing that
/// settles it is a card diffing `fusedrun`'s two arms.
///
/// ## Why the double rounding is not one
///
/// `v` comes from the file, where the tail is stored as 32 bits and widened
/// to `f64` on read, so `v as f32` is exact and `f64 → f32 → f16` rounds
/// once. That is asserted by `narrowing_the_tail_rounds_once`.
pub fn tail_f16_bits(tail: &[f64]) -> Vec<u16> {
    tail.iter()
        .map(|&v| half::f16::from_f32(v as f32).to_bits())
        .collect()
}

/// Device bytes the quantized projections occupy — every list the model holds.
///
/// A function with a name and a test rather than an expression inside `load`,
/// for the reason [`matrix_side_bytes`] gives: this is the number `fusedrun`
/// prints as "GB on the card" and the numerator
/// [`FusedModel::runtime_bits_per_weight`] divides, so a term missing here is
/// published rather than caught.
///
/// 🕳️ The `int4` term WAS missing, and on one side of that division only:
/// `quantized_weights` counts an int4 record's weights while this sum did not
/// count its bytes. On the served mixed file that is 2.030 b/weight printed
/// against a true 2.141, and a footprint 50 MB light.
pub(super) fn runtime_bytes_of(
    matrices: &[FusedMatrix],
    groups: &[FusedGroup],
    int4: &[FusedInt4],
) -> u64 {
    matrices.iter().map(|m| m.bytes).sum::<u64>()
        + groups.iter().map(|g| g.bytes).sum::<u64>()
        + int4.iter().map(|q| q.bytes).sum::<u64>()
}

/// Device bytes a matrix costs **beside** its block stream: the narrowed tail
/// and the row scales.
///
/// Extracted from `load` so the accounting is a function with a name and a
/// test rather than an expression inside a loop — `runtime_bytes` is the
/// number `fusedrun` prints as "GB on the card" and that
/// [`FusedModel::runtime_bits_per_weight`](crate::fused::FusedModel::runtime_bits_per_weight) divides, so an arithmetic slip
/// here is published, not caught.
pub fn matrix_side_bytes(d_out: usize, tail_w: usize) -> u64 {
    (d_out * tail_w) as u64 * TAIL_BYTES + d_out as u64 * ROW_SCALE_BYTES
}

/// A rotation is determined by its width and its seed, and shared by every
/// matrix consuming the same activation.
pub type RotKey = (usize, u64);

/// `Q`, flattened into what a kernel can read.
pub struct RotationTables {
    pub n: usize,
    /// Power-of-two factor: the width of one Walsh–Hadamard group.
    pub m: usize,
    /// Odd factor: the side of the dense block, and the number of groups.
    pub k: usize,
    /// One bit per coordinate, set when the sign is negative.
    pub signbits: Vec<u32>,
    /// `Q_odd`, zero-padded to `ROT_KMAX × ROT_KMAX` row-major. The padding is
    /// load-bearing: the kernel's inner loops run to a compile-time bound.
    pub small: Vec<f32>,
    /// `1/√m`, narrowed once here rather than recomputed on the device where
    /// `sqrtf` and `rsqrtf` would each land somewhere else.
    pub inv: f32,
}

impl RotationTables {
    pub fn build(n: usize, seed: u64) -> Result<Self, String> {
        let rot = Rotation::new(n, seed);
        let (m, k) = (rot.pow2(), rot.odd());
        if k > ROT_KMAX {
            return Err(format!(
                "width {n}: odd factor {k} above KMAX={ROT_KMAX}. The rotation kernel does \
                 not handle it, see llvq_rot.cuh."
            ));
        }
        let mut signbits = vec![0u32; n.div_ceil(32)];
        for (i, &s) in rot.signs().iter().enumerate() {
            if s < 0.0 {
                signbits[i >> 5] |= 1 << (i & 31);
            }
        }
        let mut small = vec![0.0f32; ROT_KMAX * ROT_KMAX];
        for g in 0..k {
            for t in 0..k {
                small[g * ROT_KMAX + t] = rot.small()[g * k + t] as f32;
            }
        }
        Ok(Self {
            n,
            m,
            k,
            signbits,
            small,
            inv: (1.0f64 / (m as f64).sqrt()) as f32,
        })
    }
}
