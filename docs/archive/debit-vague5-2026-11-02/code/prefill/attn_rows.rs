// DRAFT, never compiled or run. A sketch for `llvq-llm/src/model.rs` and `bin/fusedrun.rs`,
// written read-only against HEAD 06e1602 (debit-trims). Signatures follow candle 0.9.2.
// Purpose: lead "prefill" (a), the attention of a prefill chunk with a lower score peak, and the
// measurement mode of prereg-prefill-1.md. Nothing here is served by default: `AttnRows::Off`
// is today's code path, call for call.
//
// Revision 2 (2026-11-02), after the adversarial review:
//   * Two EXACT variants added. `InPlace(r)` keeps both cuBLAS calls at today's operands and
//     shapes and runs scale, mask and softmax per row block, written back into the score buffer
//     with `slice_set`. `InPlaceAll` runs the same three candle kernels in place on the whole
//     buffer. Neither changes a GEMM, so neither is a T8 event.
//   * `Rows(r)` and `RowsTrim(r)` change the cuBLAS shapes. On the L40S only 2 of 6 measured
//     prefill reshapes kept their bits (docs/mesures/gqa-ab-2026-11-01.txt:62-72), so they are
//     T8 candidates whatever a finite grid of shapes says.
//   * A host replica of candle's CUDA gemm stride rules, asserted on every matmul operand the
//     new variants build: the CPU gemm accepts layouts cuBLAS refuses (gqa-ppl deviation E1).

use candle_core::{DType, Layout, Result, Tensor};

/// How the prefill attention walks its query rows. `Off` is every published path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnRows {
    /// Today's code, call for call. Peak 3S (S = heads x l x L x 2 bytes).
    Off,
    /// Same calls, same shapes; the scaled scores die before the softmax and the masked
    /// scores before the PV product. Peak 2S. Exact by construction.
    Lifetimes,
    /// Both GEMMs at today's operands; scale, mask add and softmax per block of `r` query rows,
    /// written back into the raw score buffer with `slice_set`. Peak S + 2 S_b(r). Exact by
    /// construction (see `inplace_rows`). Costs one extra pass over S (estimated 0.2-0.3 s at 8k).
    InPlace(usize),
    /// Both GEMMs at today's operands; candle's own affine, binary-add and softmax kernels run
    /// in place on the whole buffer through `Tensor::inplace_op1`. Peak S, no extra pass.
    /// Exact by construction. CUDA only; elsewhere it falls back to `Lifetimes`.
    InPlaceAll,
    /// Query rows in blocks of `r`, each block through the existing form; keys untouched.
    /// Changes the cuBLAS M dimension: a T8 candidate.
    Rows(usize),
    /// Blocks of `r` with the keys narrowed to the block's causal limit. Changes M, and N or K:
    /// a T8 candidate. Cuts the rectangles to 0.75 (r = 1,024) of today's at 8k.
    RowsTrim(usize),
}

impl AttnRows {
    pub fn label(&self) -> String {
        match self {
            AttnRows::Off => "off".into(),
            AttnRows::Lifetimes => "lifetimes".into(),
            AttnRows::InPlace(r) => format!("in-place({r})"),
            AttnRows::InPlaceAll => "in-place-all".into(),
            AttnRows::Rows(r) => format!("rows({r})"),
            AttnRows::RowsTrim(r) => format!("rows-trim({r})"),
        }
    }
    /// Whether the variant keeps every cuBLAS call of `Off`. Only these may be proposed without
    /// a T8 (prereg-prefill-1 R1); the others are T8 candidates (R2).
    pub fn keeps_gemm_shapes(&self) -> bool {
        matches!(self, AttnRows::Off | AttnRows::Lifetimes | AttnRows::InPlace(_) | AttnRows::InPlaceAll)
    }
}

/// The attention form, as `forward_cached` dispatches it (model.rs:1648-1657).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// `repeat_kv(..).contiguous()` then [b, h, l, hd] x [b, h, hd, L] (model.rs:1658-1688).
    Expanded,
    /// q as [b, n_kv, groups*l, hd] against the stored key heads (model.rs:571-593).
    Grouped,
}

/// `grouped_attention` (model.rs:571-593) with the two early drops. Every operation, operand
/// and shape is the original's; only lifetimes change.
fn grouped_attention_lean(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>, scale: f64) -> Result<Tensor> {
    let (b, n_heads, l, hd) = q.dims4()?;
    let kv_len = k.dim(2)?;
    let (q5, n_kv, groups) = group_view(q, k)?; // model.rs:509-517, unchanged
    let q = q5.reshape((b, n_kv, groups * l, hd))?;
    let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    let scores = match mask {
        Some(m) => {
            let masked = scores
                .reshape((b, n_kv, groups, l, kv_len))?
                .broadcast_add(&m.unsqueeze(2)?)?
                .reshape((b, n_kv, groups * l, kv_len))?;
            drop(scores); // the original keeps this binding alive to the end of the function
            masked
        }
        None => scores,
    };
    let p = candle_nn::ops::softmax_last_dim(&scores)?;
    drop(scores);
    p.matmul(v)?.reshape((b, n_heads, l, hd))
}

/// `InPlace(r)` for either form. `q` is `[b, h, l, hd]`; `k`, `v` are what the form reads today
/// (expanded: the `repeat_kv(..).contiguous()` copies; grouped: the stored heads after
/// `rows_contiguous`); `mask` is `[b, 1, l, L]` (model.rs:2454-2467).
///
/// Why the bits cannot move, operation by operation:
///   1. QK^T is today's call: same operands, shapes and strides, so cuBLAS picks what it picks
///      today and the raw f16 buffer `s` holds today's values.
///   2. `* scale` is candle's affine kernel, `x * mul + add` with add = +0 (affine.cu). On the
///      narrowed view it takes the strided branch of the same kernel and the same expression.
///      The only possible difference between the branches is a contraction into an fma, and with
///      add = +0 that can change nothing but the sign of a zero result.
///   3. The mask add, x + m with m in {+0, -inf}, maps -0 and +0 to +0 and anything to -inf, so
///      after it every element equals today's bit for bit, sign of zero included.
///   4. The softmax runs one warp per row on a contiguous block (reduce.cu:233-273, launch at
///      candle-nn ops.rs:355-360); a row's arithmetic depends only on that row.
///   5. `slice_set` is a strided copy (tensor_cat.rs:246-300, copy2d_f16), exact.
///   6. PV is today's call on a buffer of today's shape and strides holding today's values.
/// The strict alternative to step 2 is `sb.contiguous()?` before the scale: the contiguous
/// branch exactly as today, for one more copy of each block. The card check decides whether it
/// is ever needed (prereg-prefill-1 R3).
pub fn inplace_rows(form: Form, r: usize, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, scale: f64) -> Result<Tensor> {
    let (b, n_heads, l, hd) = q.dims4()?;
    let kv_len = k.dim(2)?;
    if r == 0 {
        candle_core::bail!("in-place rows: a block of zero rows");
    }
    match form {
        Form::Expanded => {
            let s = q.matmul(&k.transpose(2, 3)?)?; // [b, h, l, L], contiguous, unscaled: today's call
            let mut r0 = 0usize;
            while r0 < l {
                let len = r.min(l - r0);
                let scaled = (s.narrow(2, r0, len)? * scale)?;               // new, contiguous
                let masked = scaled.broadcast_add(&mask.narrow(2, r0, len)?)?; // new, contiguous
                drop(scaled);
                let p = candle_nn::ops::softmax_last_dim(&masked)?;           // new, contiguous
                drop(masked);
                s.slice_set(&p, 2, r0)?; // s is contiguous and p is not its storage: accepted
                r0 += len;
            }
            s.matmul(v) // today's PV call, today's operand shape and strides
        }
        Form::Grouped => {
            let (q5, n_kv, groups) = group_view(q, k)?;
            let q = q5.reshape((b, n_kv, groups * l, hd))?;
            let s = q.matmul(&k.transpose(2, 3)?)?;                 // [b, n_kv, groups*l, L]: today's call
            let s5 = s.reshape((b, n_kv, groups, l, kv_len))?;     // a view: reshape of a contiguous tensor shares storage
            let mut r0 = 0usize;
            while r0 < l {
                let len = r.min(l - r0);
                let scaled = (s5.narrow(3, r0, len)? * scale)?;
                let masked = scaled.broadcast_add(&mask.narrow(2, r0, len)?.unsqueeze(2)?)?;
                drop(scaled);
                let p = candle_nn::ops::softmax_last_dim(&masked)?; // rows of L, as today
                drop(masked);
                s5.slice_set(&p, 3, r0)?;
                r0 += len;
            }
            drop(s5);
            s.matmul(v)?.reshape((b, n_heads, l, hd))
        }
    }
}

// ---- `InPlaceAll`, outline only -----------------------------------------------------------
//
// Three `candle_core::InplaceOp1` implementations whose `cuda_fwd` launch candle's OWN kernels
// on the score buffer, with the launch configurations candle uses:
//   * scale: `affine_f16` (candle_kernels AFFINE) with `inp = null`, which the kernel reads as
//     "in place" (`x = inp ? inp[i] : out[i]`, affine.cu); the buffer is contiguous, so the
//     contiguous branch runs, exactly today's.
//   * mask: `badd_f16` (BINARY) with `out = lhs = s` and `rhs = mask` broadcast; with lhs
//     contiguous each thread reads lhs[i] and writes out[i] at the same index (binary_op_macros.cuh).
//   * softmax: `softmax_f16` (REDUCE) with `src = dst = s`, grid (n_rows, 1, 1), block (1, 32, 1)
//     as candle-nn launches it (ops.rs:340-365); each lane reads then writes only its own columns.
// None of the three kernels declares `__restrict__`, so the aliasing is defined. The functions are
// reached through `CudaDevice::get_or_load_func` (cuda_backend/device.rs:227), as candle-nn does.
// Peak S. A CPU `cpu_fwd` does the same arithmetic in place for the host tests, but only the card
// check proves the CUDA path (prereg-prefill-1 step 3a).

/// The row-block driver for `Rows`/`RowsTrim`, for either form (unchanged from revision 1).
/// `attend` is the form's function or its lean twin. Under `Cat`/`Grow` the chunk's queries are
/// the LAST `l` of the `L` positions, so row `i` sits at absolute position `L - l + i` and needs
/// keys `[0, L - l + i]`.
pub fn attention_rows(
    attend: fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    mode: AttnRows,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let l = q.dim(2)?;
    let kv_len = k.dim(2)?;
    let (r, trim) = match mode {
        AttnRows::Rows(r) => (r, false),
        AttnRows::RowsTrim(r) => (r, true),
        _ => return attend(q, k, v, mask, scale), // the exact variants have their own entry points
    };
    if r == 0 {
        candle_core::bail!("attention rows: a block of zero rows");
    }
    // A chunk no longer than a block is the original call verbatim: decode steps and short
    // prompts cannot move, by construction rather than by test.
    if l <= r {
        return attend(q, k, v, mask, scale);
    }
    let mask = match mask {
        Some(m) => m,
        None => candle_core::bail!("attention rows: {l} query rows without a causal mask"),
    };
    let mut parts: Vec<Tensor> = Vec::with_capacity(l.div_ceil(r));
    let mut r0 = 0usize;
    while r0 < l {
        let len = r.min(l - r0);
        let qb = q.narrow(2, r0, len)?; // one batch stride: gemm_config's stride_b rule accepts it
        let mb = mask.narrow(2, r0, len)?;
        let part = if trim {
            let lim = kv_len - l + r0 + len; // keys [0, lim); the columns past it are -inf for this block
            attend(&qb, &k.narrow(2, 0, lim)?, &v.narrow(2, 0, lim)?, Some(&mb.narrow(3, 0, lim)?), scale)?
        } else {
            attend(&qb, k, v, Some(&mb), scale)?
        };
        parts.push(part);
        r0 += len;
    }
    Tensor::cat(&parts, 2) // [b, h, l, hd], rows in order
}

/// Host replica of candle 0.9.2's CUDA `gemm_config` layout rules
/// (candle-core-0.9.2/src/cuda_backend/mod.rs:1341-1432), for `lhs [.., m, k] x rhs [.., k, n]`.
/// The CPU gemm accepts layouts this refuses, which is how gqa-ppl attempt 1 passed on the Mac
/// and died on the card ($0.09, deviation E1). Every matmul the new variants build goes through
/// this check in the host tests. Mutant: the E1 layout (a value projection returned as a
/// transposed view by a fresh `Cat` append, `[1, 8, 4096, 128]` with stride
/// `[4194304, 128, 1024, 1]`, mnk (16384, 128, 4096)) must be refused.
pub fn cuda_gemm_layout_check(lhs: &Layout, rhs: &Layout) -> Result<()> {
    let (ld, rd) = (lhs.dims(), rhs.dims());
    let (ls, rs) = (lhs.stride(), rhs.stride());
    let (nl, nr) = (ld.len(), rd.len());
    if nl < 2 || nr < 2 || nl != nr {
        candle_core::bail!("gemm check: ranks {nl} and {nr}");
    }
    let (m, k, n) = (ld[nl - 2], ld[nl - 1], rd[nr - 1]);
    let (rhs_m1, rhs_m2, lhs_m1, lhs_m2) = (rs[nr - 1], rs[nr - 2], ls[nl - 1], ls[nl - 2]);
    let a_ok = ((rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1)) || ((rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1));
    let b_ok = ((lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1)) || ((lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1));
    let batch_ok = |s: &[usize], d: &[usize]| -> bool {
        match s[..s.len() - 2] {
            [s1, st] => s1 == st * d[1] || d[0] == 1 || d[1] == 1,
            [_] | [] => true,
            _ => false,
        }
    };
    if !(a_ok && b_ok && batch_ok(ls, ld) && batch_ok(rs, rd)) {
        candle_core::bail!("matmul refused by cuBLAS's layout rules: lhs {lhs:?} rhs {rhs:?} mnk ({m}, {n}, {k})");
    }
    Ok(())
}

/// The (l, L) pairs the served prompt driver meets for a prompt of `p` tokens in chunks of
/// `chunk` (model.rs:2190-2203: `tokens.chunks(chunk)`), i.e. the shapes a grid must cover.
pub fn served_chunk_pairs(p: usize, chunk: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut done = 0usize;
    while done < p {
        let l = chunk.min(p - done);
        done += l;
        out.push((l, done));
    }
    out
}

/// The grid of prereg-prefill-1 step 3a: P every 256 from 256 to 8,192, plus 16 pseudo-random P
/// in [2, 8,192] from a printed seed, mapped to their chunk pairs and deduplicated. About 48
/// pairs. Identity is certified on this grid only, never "for every shape".
pub fn identity_grid(seed: u64) -> Vec<(usize, usize)> {
    let mut ps: Vec<usize> = (1..=32).map(|k| 256 * k).collect();
    let mut x = seed | 1;
    for _ in 0..16 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17; // xorshift64: reproducible from the printed seed
        ps.push(2 + (x % 8191) as usize);
    }
    let mut pairs: Vec<(usize, usize)> = ps.iter().flat_map(|&p| served_chunk_pairs(p, 4096)).collect();
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Bitwise comparison of two f16 tensors of one shape, on the host: (differing elements,
/// largest |difference|). Bits, not values: -0 against +0 counts as a difference, and so does
/// a NaN payload. The comparator must flag a planted one-ulp change before its zero is believed
/// (prereg gate 6).
pub fn bitwise_diff(a: &Tensor, b: &Tensor) -> Result<(u64, f32)> {
    if a.dims() != b.dims() || a.dtype() != DType::F16 || b.dtype() != DType::F16 {
        candle_core::bail!("bitwise_diff: {:?} {:?} against {:?} {:?}", a.dims(), a.dtype(), b.dims(), b.dtype());
    }
    let x = a.flatten_all()?.to_vec1::<half::f16>()?;
    let y = b.flatten_all()?.to_vec1::<half::f16>()?;
    let mut n = 0u64;
    let mut worst = 0f32;
    for (p, q) in x.iter().zip(y.iter()) {
        if p.to_bits() != q.to_bits() {
            n += 1;
            worst = worst.max((p.to_f32() - q.to_f32()).abs());
        }
    }
    Ok((n, worst))
}

/// The default pool's used-memory high-water mark, reset and read around a prefill.
/// cudarc 0.19.8 allocates with cuMemAllocAsync (core.rs:1535-1536), so candle's tensors and
/// the fused runtime's buffers both come from this pool. One `unsafe` block, a hardware boundary.
/// Home: `llvq-cuda`, not `llvq-llm`. The Mac type-checks llvq-cuda's Linux half
/// (`CUDARC_CUDA_VERSION=12040 cargo clippy --target x86_64-unknown-linux-gnu -p llvq-cuda --all-targets`),
/// while llvq-llm's cuda feature does not build on the Mac (served-grouped-2026-11-02: the image was the first
/// compiler of fusedrun.rs's changes). Shown here beside its callers for readability only.
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod pool {
    use cudarc::driver::sys;

    unsafe fn default_pool(dev: sys::CUdevice) -> Result<sys::CUmemoryPool, String> {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let r = sys::cuDeviceGetDefaultMemPool(&mut pool, dev);
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuDeviceGetDefaultMemPool: {r:?}"));
        }
        Ok(pool)
    }

    /// Reset the high-water mark (CUDA allows a reset to zero only). Call before the section.
    pub fn reset_high(dev: sys::CUdevice) -> Result<(), String> {
        unsafe {
            let pool = default_pool(dev)?;
            let mut zero: u64 = 0;
            let attr = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH;
            let r = sys::cuMemPoolSetAttribute(pool, attr, &mut zero as *mut u64 as *mut std::ffi::c_void);
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuMemPoolSetAttribute(USED_MEM_HIGH): {r:?}"));
            }
        }
        Ok(())
    }

    /// Bytes: the largest use since the last reset. Read after a synchronising call.
    pub fn read_high(dev: sys::CUdevice) -> Result<u64, String> {
        unsafe {
            let pool = default_pool(dev)?;
            let mut v: u64 = 0;
            let attr = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH;
            let r = sys::cuMemPoolGetAttribute(pool, attr, &mut v as *mut u64 as *mut std::ffi::c_void);
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuMemPoolGetAttribute(USED_MEM_HIGH): {r:?}"));
            }
            Ok(v)
        }
    }
}

// ---- The host tests (Mac CPU), skeleton --------------------------------------------------
//
// #[test] fn exact_variants_equal_off_bit_for_bit()
//     both forms, f32 and f16, l not a multiple of r, r in {1, 7, 64}, Cat and Grow-shaped k/v:
//     Lifetimes, InPlace(r) and InPlaceAll (its cpu_fwd) against Off, bitwise.
//     Mutants killed: wrong mask rows; a block written back at the wrong offset; the scale
//     applied after the mask; slice_set on a non-contiguous view (must be refused, not silent).
// #[test] fn row_variants_equal_off_on_the_cpu()
//     Rows(r), RowsTrim(r) against Off on the CPU (bitwise there: the CPU gemm's order does not
//     depend on M). Mutants: wrong trim offset, reversed concatenation, a block skipped.
//     This proves the indexing only; the card is what decides identity.
// #[test] fn every_matmul_operand_passes_the_cuda_rules()
//     wrap each matmul of the variants in cuda_gemm_layout_check, both forms, Cat and Grow
//     views, all r; plus the E1 layout, which must be refused (the mutant of the check itself).
// #[test] fn grid_covers_the_remainder_blocks()
//     identity_grid(seed) holds (1904, 6000)-type pairs whose l mod 512 and l mod 1024 are not 0.

// ---- The measurement mode, skeleton (bin/fusedrun.rs, beside LLVQ_GQA_AB) ----------------
//
// if let Ok(v) = std::env::var("LLVQ_PREFILL_AB") {
//     let lengths = parse_lengths("LLVQ_PREFILL_AB", &v)?;          // 2048,4096,8192
//     refuse beside LLVQ_CONFIG and beside every other measurement mode (one mode at a time);
//     load the fused model (load_with), set Grow(GROW_CHUNK) + AttnHeads::Grouped, as the
//         served grouped config does; events off on the single stream (as LLVQ_GQA_PPL);
//
//     // 3a: identity grid. Planted one-ulp check of bitwise_diff first (gate 6). For each pair of
//     //     identity_grid(seed), both forms, seeded random f16 q/k/v laid out as the store lays
//     //     them out (Cat for expanded, Grow(256) views for grouped), the driver's mask
//     //     (build_causal_mask): Off once, then every variant, bitwise against Off. Print one
//     //     line per (pair, form, variant): identical, or count and max |d|.
//     //     Timing at (2048,2048), (4096,4096), (4096,8192) only: 1 discarded + 5 timed rounds,
//     //     each a burst of 10 calls between two fences, all variants per round in a fixed order;
//     //     pool high-water per variant at (4096,8192).
//     // 3b: end to end. Timed at 8,192: Off, InPlace(512), RowsTrim(1024), 1 discarded + 3 rounds,
//     //     interleaved, ratio t_Off / t_arm round by round (rate_stats). Identity-only, round 0:
//     //     every variant at 2,048, 4,096 and 8,192, last-position logits (f32) bitwise against
//     //     Off and 256 greedy tokens (first_divergence). Pool high-water of every prefill.
//     // 3c: f16 GEMM probe through candle matmul at the 4B prefill shapes and the grouped score
//     //     shape; bursts of 20 calls between fences; TFLOPS = 2mnk / t; 1 discarded + 5 rounds.
//     // 3d: tv_tetra48_rows_h on 4 pseudo-random rows against 4 launches of tv_tetra48_h, one real
//     //     record per Tetra width (the check_int4_half_parity pattern, fused_cuda.rs:2224).
//     // 3e: FP64 probe (code/prefill/fp64_probe.cu): DMUL/DADD instructions per second, bursts
//     //     of 10 launches between fences.
//     print every table in full; return Ok(()).
// }
