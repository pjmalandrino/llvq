//! A prompt's rows through the resident experts of a layer in a few launches, the Metal twin of
//! `resident_cuda.rs`'s `forward_rows_batched`. One file because it is one path, built on the slab of
//! `moe.rs`, with its own kernel text.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};

use super::moe::{moe_off_msl, moe_shape};
use super::launch::{GROUP, LANES};
use super::types::MetalRuntime;
use super::MetalMoeSlab;

// ---------------------------------------------------------------------------
// A prompt's rows through the resident experts of a layer in a few launches
// (the Metal twin of `resident_cuda.rs`'s `forward_rows_batched`).
//
// `ResidentExperts::forward_general` runs, for each expert that received
// tokens, a gather, an upload of its ids and weights, a rotation, gate and up
// in chunks of 8 rows through `tv_tetra48_metal_lutdrf`, candle's activation,
// a rotation, down in chunks, a product by the weights and an `index_add`:
// about 35 launches an expert, 4,500 a layer visit for a piece of 512 tokens.
// Here every (token, expert) pair is a row, sorted by expert then token, and
// a visit is:
//   a. `moe_rot_metal` (the one-row path's kernel) over all the pairs, each
//      row taking the input rotation of its own expert;
//   b. `moe_rows_lutdrf`: gate and up of every chunk of at most 8 rows of one
//      expert, the body of `tv_tetra48_metal_lutdrf` with its pointers taken
//      from the slab;
//   c. candle's activation on all the rows, then `moe_rot_metal` again with
//      the down rotations;
//   d. `moe_rows_lutdrf` for down;
//   e. `moe_pairs_sum`: each token's rows times their weights, added in
//      ascending expert order from +0, the `mul` and the `add` rounded apart.
// A row's value does not depend on the other rows of its chunk or launch, the
// chunks are the general path's (consecutive rows of one expert, at most 8),
// and the sum runs the general path's additions in its order. So the values
// are meant to be its own, bit for bit; `tests/inkling_moe_rows.rs` demands
// it. The text is appended to the served unit (its `fp contract(off)`
// covers it) and compiled as a library of its own.
// ---------------------------------------------------------------------------

macro_rules! moe_rows_msl {
    () => {
        r#"
// moe_rows_lutdrf: simdgroup g computes output row `row` of matrix `mat0 + mi`
// for the rows of chunk `ci`: g = (mi * nchunks + ci) * d_out + row. A chunk
// is (expert, first pair row, rows), rows 1..=TETRA48_PREFILL_MAX. d_out is a
// multiple of 8, so the 8 simdgroups of a threadgroup share one chunk and one
// row count. The loop and the epilogue are `tv_tetra48_metal_lutdrf`'s text.
kernel void moe_rows_lutdrf(const device uint*         slab           [[buffer(0)]],
                            const device float*        slabf          [[buffer(1)]],
                            const device half*         slabh          [[buffer(2)]],
                            const device MoeExpertOff* off            [[buffer(3)]],
                            const device uint*         chunks         [[buffer(4)]],
                            const device uint*         rows           [[buffer(5)]],
                            const device uchar*        prefixes       [[buffer(6)]],
                            const device ushort*       branches       [[buffer(7)]],
                            const device uchar*        suffixes       [[buffer(8)]],
                            const device float*        invnorm        [[buffer(9)]],
                            const device float*        xall           [[buffer(10)]],
                            device float*              yall           [[buffer(11)]],
                            constant uint&             row_stride_u32 [[buffer(12)]],
                            constant uint&             nblocks        [[buffer(13)]],
                            constant uint&             tail_w         [[buffer(14)]],
                            constant uint&             d_out          [[buffer(15)]],
                            constant uint&             nchunks        [[buffer(16)]],
                            constant uint&             mat0           [[buffer(17)]],
                            constant uint&             npairs         [[buffer(18)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]])
{
    uint gg = gid >> 5;
    uint per_mat = nchunks * d_out;
    uint mi = gg / per_mat;
    uint rem = gg - mi * per_mat;
    uint ci = rem / d_out;
    uint row = rem - ci * d_out;
    uint e = chunks[3u * ci];
    uint first = chunks[3u * ci + 1u];
    uint nrows = chunks[3u * ci + 2u];
    uint mat = mat0 + mi;
    MoeExpertOff o = off[e];
    const device uint* words = slab + o.words[mat];
    const device float* gscale = slabf + o.gscale[mat];
    const device float* rscale = slabf + o.rscale[mat];
    const device half* tail = slabh + o.tail[mat];
    uint d_in = nblocks * 24u + tail_w;
    const device float* x = xall + (ulong)first * d_in;
    device float* y = yall + ((ulong)mi * npairs + first) * d_out;

    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc[TETRA48_PREFILL_MAX];
    for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
        acc[r] = 0.0f;
    }
    for (uint j = lane; j < nblocks; j += 32u) {
        uint lo, hi16;
        f1r_load(wrow, j, lo, hi16);
        F1rV3Block b = f1r_v3_fetch(lo, hi16, tab);
        uint cs[3] = { b.c1, b.c2, b.c3 };
        uint rs[3] = { b.row1, b.row2, b.row3 };
        float v[24];
        float n2 = 0.0f;
        for (uint sec = 0u; sec < 3u; ++sec) {
            uint c = cs[sec];
            uint rw = rs[sec];
            for (uint k = 0u; k < 8u; ++k) {
                uint o8 = b.p + 2u * ((c >> k) & 1u);
                uint rho = (rw >> (4u * k)) & 15u;
                float vv = F1R_VAL[(o8 << 4) | rho];
                v[sec * 8u + k] = vv;
                n2 = fma(vv, vv, n2);
            }
        }
        uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
        uint g = (hi16 >> 15) & 1u;
        float gs = gscale[g];
        float inv = invnorm[m];
        for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
            if (r < nrows) {
                const device float* xb = x + r * d_in + j * 24u;
                float a = 0.0f;
                for (uint i = 0u; i < 24u; ++i) {
                    a = fma(v[i], xb[TETRA48_ORDER[i]], a);
                }
                acc[r] += a * gs * inv;
            }
        }
    }
    for (uint r = 0u; r < TETRA48_PREFILL_MAX; ++r) {
        if (r < nrows) {
            tetra48_rows_epilogue(acc[r], row, lane, rscale, tail, x + r * d_in, y + r * d_out,
                                  nblocks, tail_w);
        }
    }
}

/// `+0.0f + p` with the IEEE sign rule spelled in integers (see
/// `moe_zero_plus` of the one-row unit).
inline float moe_rows_zero_plus(float p)
{
    return as_type<uint>(p) == 0x80000000u ? 0.0f : p;
}

// moe_pairs_sum: out[t][c] = sum over the token's k rows, in ascending expert
// order, of y[pair][c] * w, from +0. `tp` holds, a token, its k pair rows in
// that order; `tw` their weights.
kernel void moe_pairs_sum(const device float* y    [[buffer(0)]],
                          const device uint*  tp   [[buffer(1)]],
                          const device float* tw   [[buffer(2)]],
                          device float*       out  [[buffer(3)]],
                          constant uint&      k    [[buffer(4)]],
                          constant uint&      h    [[buffer(5)]],
                          constant uint&      t    [[buffer(6)]],
                          uint2 gp [[thread_position_in_grid]])
{
    uint c = gp.x;
    uint tok = gp.y;
    if (c >= h || tok >= t) { return; }
    float acc = 0.0f;
    for (uint kk = 0u; kk < k; ++kk) {
        float p = y[(ulong)tp[tok * k + kk] * h + c] * tw[tok * k + kk];
        acc = kk == 0u ? moe_rows_zero_plus(p) : acc + p;
    }
    out[(ulong)tok * h + c] = acc;
}
"#
    };
}

pub(super) const SRC_MOE_ROWS: &str = concat!(
    include_str!("../../kernels/llvq_tetra48.metal"),
    include_str!("../../kernels/llvq_tetra48_rows.metal"),
    moe_off_msl!(),
    moe_rows_msl!()
);

/// Which matrices one launch of `moe_rows_lutdrf` computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeRowsMats {
    /// Gate and up: `[2, P, inter]` from rows `[P, hidden]`.
    GateUp,
    /// Down: `[1, P, hidden]` from rows `[P, inter]`.
    Down,
}

impl MetalRuntime {
    /// Whether [`Self::moe_rows`] and [`Self::moe_rotate_pairs`] reproduce the
    /// general path of `ResidentExperts`: the one-row fused path's conditions,
    /// and the rows kernel in use (`prefill_rows` at least 2).
    pub fn moe_rows_eligible(&self) -> bool {
        self.moe_fused_eligible() && self.prefill_rows >= 2
    }

    /// Rows `x` f16 `[P, n]` into the basis of their own expert's input
    /// rotation (`down == false`, n = hidden) or down rotation (`down ==
    /// true`, n = inter); `pair_expert` u32 `[P]`, the expert of each row,
    /// checked by the caller against the slab. f32 `[P, n]`; row r is what
    /// `rotate` returns for that row and expert alone.
    pub fn moe_rotate_pairs(&self, slab: &MetalMoeSlab, down: bool, x: &Tensor, pair_expert: &Tensor) -> Result<Tensor> {
        x.apply_op2_no_bwd(pair_expert, &MoeRotPairsOp { rt: self, slab, down })
    }

    /// Gate and up, or down, of every chunk of `chunks` (u32 `[3 C]`: expert,
    /// first row, rows; rows 1..=`prefill_rows`) on the rotated rows `x` f32
    /// `[P, d_in]`: f32 `[2, P, inter]` or `[1, P, hidden]`. Each row is what
    /// `matvec_rows_f32` returns for it in its chunk on that expert alone.
    pub fn moe_rows(&self, slab: &MetalMoeSlab, mats: MoeRowsMats, x: &Tensor, chunks: &Tensor) -> Result<Tensor> {
        x.apply_op2_no_bwd(chunks, &MoeRowsOp { rt: self, slab, mats })
    }

    /// Each token's `k` rows of `y` f32 `[P, h]` (pair rows `tp` u32 `[T k]`,
    /// ascending expert order) times their weights `tw` f32 `[T k]`, added from
    /// +0: f32 `[T, h]`.
    pub fn moe_pairs_sum(&self, y: &Tensor, tp: &Tensor, tw: &Tensor, k: usize) -> Result<Tensor> {
        let (p, h) = y.dims2()?;
        let tk = tp.elem_count();
        if k == 0 || !tk.is_multiple_of(k) || tw.elem_count() != tk || p == 0 {
            candle_core::bail!("moe pairs sum: {tk} pair rows, {} weights, k = {k}, {p} rows", tw.elem_count());
        }
        let op = MoePairsSumOp { rt: self, k, h, t: tk / k };
        let yt = y.contiguous()?;
        let tp = tp.contiguous()?;
        let tw = tw.contiguous()?;
        yt.apply_op3_no_bwd(&tp, &tw, &op)
    }
}

struct MoeRotPairsOp<'a> {
    rt: &'a MetalRuntime,
    slab: &'a MetalMoeSlab,
    down: bool,
}

impl CustomOp2 for MoeRotPairsOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-moe-rot-pairs"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("moe pair rotation: Metal only; its reference is `rotate`, one expert at a time")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, pe: &MetalStorage, lp: &Layout) -> Result<(MetalStorage, Shape)> {
        let s = moe_shape(self.slab)?;
        if x.dtype() != DType::F16 || pe.dtype() != DType::U32 {
            candle_core::bail!("moe pair rotation: f16 rows and u32 experts, got {:?} and {:?}", x.dtype(), pe.dtype());
        }
        let (start, end) =
            lx.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe pair rotation: a strided activation"))?;
        let (pstart, pend) =
            lp.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe pair rotation: a strided expert list"))?;
        let npairs = pend - pstart;
        let (n, m, k, inv, threads) = if self.down { s.rot_down } else { s.rot_in };
        let n = n as usize;
        if npairs == 0 || end - start != npairs * n {
            candle_core::bail!("moe pair rotation: {} values for {npairs} rows of {n}", end - start);
        }
        let dev = x.device().clone();
        let pipe = self.rt.pipeline("moe_rot_metal")?;
        let out = dev.new_buffer(npairs * n, DType::F32, "llvq-moe-rot-pairs")?;
        let scratch = dev.new_buffer(npairs * n, DType::F32, "llvq-moe-rot-pairs-scratch")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (x.buffer(), 0usize));
        set_param(&enc, 1, (&*self.slab.slab, 0usize));
        set_param(&enc, 2, (&*self.slab.slab, 0usize));
        set_param(&enc, 3, (&*self.slab.offsets, 0usize));
        set_param(&enc, 4, (pe.buffer(), pstart * 4));
        set_param(&enc, 5, (&*out, 0usize));
        set_param(&enc, 6, (&*scratch, 0usize));
        set_param(&enc, 7, n as u32);
        set_param(&enc, 8, m);
        set_param(&enc, 9, k);
        set_param(&enc, 10, f32::from_bits(inv));
        set_param(&enc, 11, start as u32);
        set_param(&enc, 12, n as u32);
        set_param(&enc, 13, u32::from(self.down));
        for b in [x.buffer(), &*self.slab.slab, &*self.slab.offsets, pe.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        for b in [&*out, &*scratch] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Write);
        }
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: threads * npairs, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev, npairs * n, DType::F32), Shape::from((npairs, n))))
    }
}

struct MoeRowsOp<'a> {
    rt: &'a MetalRuntime,
    slab: &'a MetalMoeSlab,
    mats: MoeRowsMats,
}

impl CustomOp2 for MoeRowsOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-moe-rows"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("moe rows: Metal only; their reference is `matvec_rows_f32`, one expert at a time")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, ch: &MetalStorage, lc: &Layout) -> Result<(MetalStorage, Shape)> {
        let s = moe_shape(self.slab)?;
        if x.dtype() != DType::F32 || ch.dtype() != DType::U32 {
            candle_core::bail!("moe rows: f32 rows and u32 chunks, got {:?} and {:?}", x.dtype(), ch.dtype());
        }
        if !lx.is_contiguous() {
            candle_core::bail!("moe rows: the rows must be contiguous");
        }
        let (cstart, cend) =
            lc.contiguous_offsets().ok_or_else(|| candle_core::Error::msg("moe rows: a strided chunk table"))?;
        if cend == cstart || !(cend - cstart).is_multiple_of(3) {
            candle_core::bail!("moe rows: a chunk table of {} words", cend - cstart);
        }
        let nchunks = (cend - cstart) / 3;
        let (d_in, d_out, nb, tw, stride, mat0, nmat) = match self.mats {
            MoeRowsMats::GateUp => (s.hidden, s.inter, s.nb_in, s.tail_in, s.stride_in, 0u32, 2usize),
            MoeRowsMats::Down => (s.inter, s.hidden, s.nb_down, s.tail_down, s.stride_down, 2u32, 1usize),
        };
        let dims = lx.dims();
        if dims.len() != 2 || dims[1] != d_in || dims[0] == 0 {
            candle_core::bail!("moe rows: rows {dims:?} for [P, {d_in}]");
        }
        let npairs = dims[0];
        if !d_out.is_multiple_of(GROUP / LANES) {
            candle_core::bail!("moe rows: d_out {d_out} is not a multiple of {} rows", GROUP / LANES);
        }
        let dev = x.device().clone();
        let pipe = self.rt.pipeline("moe_rows_lutdrf")?;
        let out_len = nmat * npairs * d_out;
        let out = dev.new_buffer(out_len, DType::F32, "llvq-moe-rows")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        let t = &self.rt.tables;
        set_param(&enc, 0, (&*self.slab.slab, 0usize));
        set_param(&enc, 1, (&*self.slab.slab, 0usize));
        set_param(&enc, 2, (&*self.slab.slab, 0usize));
        set_param(&enc, 3, (&*self.slab.offsets, 0usize));
        set_param(&enc, 4, (ch.buffer(), cstart * 4));
        set_param(&enc, 5, (&*t.rows, 0usize));
        set_param(&enc, 6, (&*t.prefixes, 0usize));
        set_param(&enc, 7, (&*t.branches, 0usize));
        set_param(&enc, 8, (&*t.suffixes, 0usize));
        set_param(&enc, 9, (&*t.invnorm, 0usize));
        set_param(&enc, 10, (x.buffer(), lx.start_offset() * DType::F32.size_in_bytes()));
        set_param(&enc, 11, (&*out, 0usize));
        set_param(&enc, 12, stride as u32);
        set_param(&enc, 13, nb as u32);
        set_param(&enc, 14, tw as u32);
        set_param(&enc, 15, d_out as u32);
        set_param(&enc, 16, nchunks as u32);
        set_param(&enc, 17, mat0);
        set_param(&enc, 18, npairs as u32);
        for b in [&*self.slab.slab, &*self.slab.offsets, ch.buffer(), &*t.rows, &*t.prefixes, &*t.branches, &*t.suffixes, &*t.invnorm, x.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: nmat * nchunks * d_out * LANES, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev, out_len, DType::F32), Shape::from((nmat, npairs, d_out))))
    }
}

struct MoePairsSumOp<'a> {
    rt: &'a MetalRuntime,
    k: usize,
    h: usize,
    t: usize,
}

impl candle_core::CustomOp3 for MoePairsSumOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-moe-pairs-sum"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("moe pairs sum: Metal only; its reference is the general path's index_add")
    }

    fn metal_fwd(
        &self,
        y: &MetalStorage,
        ly: &Layout,
        tp: &MetalStorage,
        lp: &Layout,
        tw: &MetalStorage,
        lw: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if y.dtype() != DType::F32 || tp.dtype() != DType::U32 || tw.dtype() != DType::F32 {
            candle_core::bail!("moe pairs sum: f32 rows, u32 pairs, f32 weights");
        }
        let off = |l: &Layout| l.start_offset() * 4;
        let dev = y.device().clone();
        let pipe = self.rt.pipeline("moe_pairs_sum")?;
        let n = self.t * self.h;
        let out = dev.new_buffer(n, DType::F32, "llvq-moe-pairs-sum")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (y.buffer(), off(ly)));
        set_param(&enc, 1, (tp.buffer(), off(lp)));
        set_param(&enc, 2, (tw.buffer(), off(lw)));
        set_param(&enc, 3, (&*out, 0usize));
        set_param(&enc, 4, self.k as u32);
        set_param(&enc, 5, self.h as u32);
        set_param(&enc, 6, self.t as u32);
        for b in [y.buffer(), tp.buffer(), tw.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: self.h, height: self.t, depth: 1 },
            objc2_metal::MTLSize { width: 32, height: 8, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev, n, DType::F32), Shape::from((self.t, self.h))))
    }
}
