//! The attention forms that read the cached heads unexpanded, the masked softmax in its two memory
//! forms, and the `Kernel` form's dispatch of rows and prefill chunks. One file because they are
//! free functions over the same `q`, `k`, `v` and mask, held to return the same bits.

use candle_core::{IndexOp, Result, Tensor};

use super::cache::rows_contiguous;

/// An attention form that reads the cached heads unexpanded: `q`, `k`, `v`,
/// the mask (or `None` where it would be all zeros), the score scale.
pub(super) type AttnFn = fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>;

/// `q` as `[b, n_kv, groups, l, hd]`, with the shapes it is read against.
///
/// Query head `h` reads key head `h / groups`, the convention of `repeat_kv`,
/// so the `groups` query heads of one key head are contiguous in `q`.
fn group_view(q: &Tensor, k: &Tensor) -> Result<(Tensor, usize, usize)> {
    let (b, n_heads, l, hd) = q.dims4()?;
    let n_kv = k.dim(1)?;
    if n_kv == 0 || n_heads % n_kv != 0 {
        candle_core::bail!("attention: {n_heads} query heads over {n_kv} key heads");
    }
    let groups = n_heads / n_kv;
    Ok((q.contiguous()?.reshape((b, n_kv, groups, l, hd))?, n_kv, groups))
}

/// Attention of `q`, `[b, n_heads, l, hd]`, over `k` and `v` holding the
/// `n_kv` heads the cache stores, `[b, n_kv, L, hd]`, one matmul per position
/// in the group, without expanding them.
///
/// Position `r` of every group is the `[b, n_kv, l, hd]` view `q5[.., .., r]`,
/// whose batch stride skips the other positions: the matmul multiplies it by
/// the stored key head with the shape the expanded form uses for one query
/// head, and the CPU gemm and cuBLAS read the stride rather than copy. The
/// scores are stacked back in query-head order before the scale, the mask and
/// the softmax, each of which works row by row. On the CPU this is bit for
/// bit the expanded form (`attention_forms_agree`); on another backend it is
/// as long as its matmul rounds a batch element the same way whatever the
/// batch count.
pub(super) fn shared_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let (b, n_heads, l, hd) = q.dims4()?;
    let (q5, _, groups) = group_view(q, k)?;
    let kt = k.transpose(2, 3)?;
    let scores = (0..groups)
        .map(|r| q5.i((.., .., r))?.matmul(&kt))
        .collect::<Result<Vec<_>>>()?;
    let scores = (Tensor::stack(&scores, 2)? * scale)?;
    let scores = match mask {
        Some(m) => scores.broadcast_add(&m.unsqueeze(2)?)?,
        None => scores,
    };
    let p = candle_nn::ops::softmax_last_dim(&scores)?;
    let ctx = (0..groups)
        .map(|r| p.i((.., .., r))?.matmul(v))
        .collect::<Result<Vec<_>>>()?;
    Tensor::stack(&ctx, 2)?.reshape((b, n_heads, l, hd))
}

/// Attention of `q`, `[b, n_heads, l, hd]`, over `k` and `v` holding the
/// `n_kv` heads the cache stores, `[b, n_kv, L, hd]`, one matmul per key head,
/// without expanding them.
///
/// Query head `h` reads key head `h / groups`, the convention of `repeat_kv`,
/// so the `groups` query heads of one key head are contiguous in `q`. Viewed
/// as `[b, n_kv, groups·l, hd]`, they become the rows of one matrix that
/// multiplies its key head once: no copy of the history, and on Qwen3-4B a
/// quarter of the bytes `repeat_kv` makes the expanded matmuls read. The rows
/// are ordered (query head in the group, position), so the `[l, L]` causal
/// mask applies to each of the `groups` blocks of rows, through a
/// 5-dimensional view. Each score is the same dot product as in the expanded
/// form; whether a backend rounds it the same way at the new matmul shape is
/// a measurement, not a property of this function.
pub(super) fn grouped_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let (b, n_heads, l, hd) = q.dims4()?;
    let kv_len = k.dim(2)?;
    let (q5, n_kv, groups) = group_view(q, k)?;
    let q = q5.reshape((b, n_kv, groups * l, hd))?;
    let raw = q.matmul(&k.transpose(2, 3)?)?;
    let p = match mask {
        Some(m) if in_place(&raw) => {
            softmax_in_place(&raw, m, scale, l)?;
            raw
        }
        // Each step drops its input before the next allocates: two score
        // tensors alive at most, where three shadowed bindings kept three.
        Some(m) => {
            let scores = {
                let scaled = (raw * scale)?;
                scaled
                    .reshape((b, n_kv, groups, l, kv_len))?
                    .broadcast_add(&m.unsqueeze(2)?)?
                    .reshape((b, n_kv, groups * l, kv_len))?
            };
            candle_nn::ops::softmax_last_dim(&scores)?
        }
        None => candle_nn::ops::softmax_last_dim(&(raw * scale)?)?,
    };
    p.matmul(v)?.reshape((b, n_heads, l, hd))
}

/// Score tensors of at least this many bytes take [`softmax_in_place`]. Below
/// it the three element-wise kernels run once on the whole tensor: fewer
/// launches, for a transient that does not matter. Both paths return the same
/// bits, so the threshold moves memory and launches, never an output.
const IN_PLACE_BYTES: usize = 64 << 20;

/// Rows one block of [`softmax_in_place`] carries: its two transients are
/// `SOFTMAX_ROWS × L` scores each, 16.8 MB in f16 at L = 8,192.
pub(super) const SOFTMAX_ROWS: usize = 1024;

/// MEASUREMENT ONLY, `false` everywhere else: every masked softmax runs on
/// the whole score tensor, the memory form of every run before 2026-11-08.
/// `bin/fusedrun` sets it from `LLVQ_SOFTMAX_WHOLE=1` so a second process can
/// check on the card that the in-place form returns the same logits; it moves
/// memory and nothing else, which is what that check confirms. Process-wide
/// because the attention bodies are free functions; nothing served sets it.
static SOFTMAX_WHOLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_softmax_whole_for_measurement(on: bool) {
    SOFTMAX_WHOLE.store(on, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(test))]
fn in_place_from() -> usize {
    match SOFTMAX_WHOLE.load(std::sync::atomic::Ordering::Relaxed) {
        true => usize::MAX,
        false => IN_PLACE_BYTES,
    }
}

/// What a run line prints for the softmax's memory form.
pub fn softmax_form() -> &'static str {
    match in_place_from() {
        usize::MAX => "whole tensor (LLVQ_SOFTMAX_WHOLE=1)",
        _ => "in place",
    }
}

#[cfg(not(test))]
fn softmax_rows() -> usize {
    SOFTMAX_ROWS
}

// The host tests move both knobs per thread (each test runs on its own), so
// a tiny model reaches the in-place path and its remainder blocks.
#[cfg(test)]
thread_local! {
    pub(super) static IN_PLACE_FROM: std::cell::Cell<usize> = const { std::cell::Cell::new(IN_PLACE_BYTES) };
    pub(super) static SOFTMAX_ROWS_T: std::cell::Cell<usize> = const { std::cell::Cell::new(SOFTMAX_ROWS) };
}

#[cfg(test)]
fn in_place_from() -> usize {
    IN_PLACE_FROM.with(|c| c.get())
}

#[cfg(test)]
fn softmax_rows() -> usize {
    SOFTMAX_ROWS_T.with(|c| c.get())
}

pub(super) fn in_place(raw: &Tensor) -> bool {
    raw.elem_count() * raw.dtype().size_in_bytes() >= in_place_from()
}

/// `softmax(raw · scale + mask)` over the last axis, written back into `raw`.
///
/// `raw` is the contiguous output of the score matmul, `[b, slabs, rows, L]`
/// with `rows` a whole number of blocks of `l` query rows: the grouped form's
/// `[b, n_kv, groups·l, L]` and the expanded form's `[b, n_heads, l, L]` are
/// both `b·slabs·rows / l` slabs of `l` rows, row `i` of a slab under row `i`
/// of `mask`, `[mb, 1, l, L]` with `mb` 1 or `b`.
///
/// Why: the whole-tensor form keeps the scaled, the masked and the softmaxed
/// scores alive, three tensors of `heads × l × L` each, 6.44 GB on the 4B at
/// the second chunk of an 8,192-token prompt. Here `raw` is the only one, plus
/// two blocks of [`SOFTMAX_ROWS`] rows.
///
/// Why it keeps every bit: the matmuls on both sides of it see the operands
/// they saw before, shape and strides included, so cuBLAS (and the CPU and
/// Metal gemms) take the same algorithm. Between them each element meets the
/// same three operations in the same order: the affine `x · scale + 0` (what
/// `Tensor * f64` is), then `x + m` with `m` either +0 or -inf, then a softmax
/// whose kernel reduces one row at a time and reads nothing outside it (one
/// warp a row on CUDA, `candle-kernels/src/reduce.cu`). A block is a
/// contiguous run of whole rows, so it takes each kernel's contiguous branch,
/// the branch the whole tensor took; the mask add, which ran broadcast,
/// becomes an add of equal shapes, the same `x + y` per element. The copy back
/// is a copy.
pub(super) fn softmax_in_place(raw: &Tensor, mask: &Tensor, scale: f64, l: usize) -> Result<()> {
    let dims = raw.dims();
    let (b, kv_len) = (dims[0], dims[dims.len() - 1]);
    let rows_total = raw.elem_count() / kv_len;
    if !raw.is_contiguous() || dims.len() != 4 || l == 0 || !dims[2].is_multiple_of(l) {
        candle_core::bail!("softmax_in_place: scores {dims:?}, not contiguous slabs of {l} rows");
    }
    let (mb, m_rows, m_cols) = (mask.dim(0)?, mask.dim(2)?, mask.dim(3)?);
    if mask.dims4()?.1 != 1 || (mb != 1 && mb != b) || m_rows != l || m_cols != kv_len {
        candle_core::bail!("softmax_in_place: mask {:?} against scores {dims:?} in slabs of {l}", mask.dims());
    }
    // A view of the same storage: `slice_set` on it writes into `raw`.
    let flat = raw.reshape((rows_total, kv_len))?;
    let mask = mask.reshape((mb, l, kv_len))?;
    let per_batch = rows_total / (b * l);
    let step = softmax_rows();
    for s in 0..rows_total / l {
        let m = mask.get(if mb == 1 { 0 } else { s / per_batch })?;
        for r0 in (0..l).step_by(step) {
            let n = step.min(l - r0);
            let at = s * l + r0;
            let p = {
                let scaled = (flat.narrow(0, at, n)? * scale)?;
                let masked = scaled.add(&m.narrow(0, r0, n)?)?;
                drop(scaled);
                candle_nn::ops::softmax_last_dim(&masked)?
            };
            flat.slice_set(&p, 0, at)?;
        }
    }
    Ok(())
}

/// The `Kernel` form's dispatch of one call of `l` rows, batch 1, over a
/// cache of `n` positions that already holds them (rows at `n - l .. n`).
/// Rows at position `from` or later go to `kern`; the rows below take
/// [`grouped_attention`] over the history they see, with their rows of
/// `mask`. A call on one side of `from` is one launch of one path; a call
/// that straddles it is split, so a row's path depends on its position and
/// on nothing else.
pub(super) fn kernel_rows(
    kern: &dyn crate::device::AttnKernel,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    from: usize,
) -> Result<Tensor> {
    let (l, n) = (q.dim(2)?, k.dim(2)?);
    // Rows at `from` or later: positions max(from, n - l) .. n.
    let tail = n.saturating_sub(from.max(n - l));
    let head = l - tail;
    if tail == 0 {
        return grouped_attention(q, &rows_contiguous(k)?, &rows_contiguous(v)?, mask, scale);
    }
    if head == 0 {
        return kern.attend(&q.contiguous()?, k, v);
    }
    let seen = n - tail;
    let mask = mask.map(|m| m.narrow(2, 0, head)?.narrow(3, 0, seen)).transpose()?;
    let low = grouped_attention(
        &q.narrow(2, 0, head)?,
        &rows_contiguous(&k.narrow(2, 0, seen)?)?,
        &rows_contiguous(&v.narrow(2, 0, seen)?)?,
        mask.as_ref(),
        scale,
    )?;
    let high = kern.attend(&q.narrow(2, head, tail)?.contiguous()?, k, v)?;
    Tensor::cat(&[&low, &high], 2)
}

/// [`crate::kvq::PrefillAttn::Kernel`]: a call of `l` rows, more than the
/// kernel carries, as consecutive calls of at most `kern.max_rows()` rows,
/// each through [`kernel_rows`] over the cache prefix its last row sees.
///
/// Every row at `from` or later gets the kernel's bits for its position,
/// which are a function of that row's query and the keys and values up to it
/// and of nothing else (`llvq-cuda/src/attn_ref.rs`): the result does not
/// depend on how the prompt was chunked, and it is what a decode step at the
/// same position returns. The memory is `l` output rows; no `l × L` score
/// tensor is ever made. Rows below `from` (none under a persistent config,
/// whose threshold is 0) take the grouped path block by block, which is the
/// grouped form at other matmul shapes.
pub(super) fn kernel_prefill(
    kern: &dyn crate::device::AttnKernel,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    from: usize,
) -> Result<Tensor> {
    let (l, n) = (q.dim(2)?, k.dim(2)?);
    let step = kern.max_rows().max(1);
    let pos0 = n - l;
    let mut out = Vec::with_capacity(l.div_ceil(step));
    for r0 in (0..l).step_by(step) {
        let rows = step.min(l - r0);
        let seen = pos0 + r0 + rows;
        let m = mask.map(|m| m.narrow(2, r0, rows)?.narrow(3, 0, seen)).transpose()?;
        out.push(kernel_rows(
            kern,
            &q.narrow(2, r0, rows)?,
            &k.narrow(2, 0, seen)?,
            &v.narrow(2, 0, seen)?,
            m.as_ref(),
            scale,
            from,
        )?);
    }
    Tensor::cat(&out, 2)
}
