//! Resident experts, the additive entry points: one matrix and one rotation
//! uploaded after the runtime exists, and the rotation by a key outside its
//! map. One file because they are one additive block, reached through new
//! entry points only.

use super::*;

// ---------------------------------------------------------------------------
// Resident experts: additive entry points (Inkling 2 Mini, 2026-11-22).
//
// `FusedRuntime::new` uploads a whole `FusedModel` at once, with every
// rotation in its map. A mixture of experts is loaded a layer at a time
// (`inkling::resident_cuda`), so it needs to upload one matrix and one
// rotation after the runtime exists, and to rotate by a rotation that is not
// in the map. Nothing above this line calls what follows, and nothing above
// this line changed: the served paths and their kernels are untouched. The
// launches below are the served ones (`launch_rotation`,
// `launch_rotation_rows`, `upload_matrix`, `upload_rotation`), reached
// through new entry points only.
// ---------------------------------------------------------------------------

/// One rotation uploaded after the runtime was built, outside its map.
///
/// It carries its key so that [`FusedRuntime::rotate_ext`] can refuse a
/// projection quantized under another rotation: the wrong basis gives finite,
/// plausible, wrong numbers and no error at all.
pub struct ExtRotation {
    pub(super) key: RotKey,
    pub(super) buf: RotBuffers,
    bytes: u64,
}

impl ExtRotation {
    /// `(width, seed)`, the key a projection's [`FusedProj::rotation`] names.
    pub fn key(&self) -> RotKey {
        self.key
    }

    /// Device bytes of the two tables (sign bits and the padded odd block).
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl FusedRuntime {
    /// Upload one `Tetra48` matrix to a runtime built for `Tetra48`.
    ///
    /// Refused on a runtime of another layout: `tetra_tabs` is present exactly
    /// when the layout is `Tetra48`, and the kernel this runtime launches is
    /// the one that layout named. The upload itself is [`upload_matrix`],
    /// which refuses a stream of another layout and a `d_out` that is not a
    /// multiple of 8.
    pub fn upload_matrix_ext(&self, m: &FusedMatrix) -> candle_core::Result<FusedProj> {
        if self.tetra_tabs.is_none() {
            candle_core::bail!(
                "{}: upload_matrix_ext needs a runtime built for tetra48, and this one carries no \
                 Tetra tables",
                m.name
            );
        }
        upload_matrix(&self.cuda, m, FusedLayout::Tetra48)
    }

    /// Build and upload the rotation `(n, seed)`.
    ///
    /// Refused unless the runtime was built with a rotation of the same width:
    /// `new` posed the shared-memory opt-in of `rot_apply` for the widest
    /// width it saw, and compiled `rot_apply_dev` only for the device-staged
    /// widths it saw. A width it never saw could need either and get neither.
    pub fn upload_rotation_ext(&self, n: usize, seed: u64) -> candle_core::Result<ExtRotation> {
        if !self.rotations.keys().any(|k| k.0 == n) {
            let mut widths: Vec<usize> = self.rotations.keys().map(|k| k.0).collect();
            widths.sort_unstable();
            widths.dedup();
            candle_core::bail!(
                "rotation of width {n}: the runtime was built for widths {widths:?}, so the opt-in \
                 of rot_apply and the wide unit were sized without it"
            );
        }
        let t = RotationTables::build(n, seed).map_err(candle_core::Error::msg)?;
        let dev = self.cuda.device().map_err(candle_core::Error::msg)?;
        let kernel = llvq_cuda::wide::rot_kernel(
            n,
            dev.shared_per_block as usize,
            dev.shared_per_block_optin as usize,
        );
        let bytes = 4 * t.signbits.len() as u64 + 4 * t.small.len() as u64;
        Ok(ExtRotation { key: (n, seed), buf: upload_rotation(&self.cuda, &t, kernel)?, bytes })
    }

    /// [`Self::rotate`] by an [`ExtRotation`]: `x` `[1, d_in]` into the basis
    /// `proj` was quantized in, f32 `[1, d_in]`, one `rot_apply`.
    pub fn rotate_ext(
        &self,
        rot: &ExtRotation,
        proj: &FusedProj,
        x: &Tensor,
    ) -> candle_core::Result<Tensor> {
        self.rotate_ext_rows(rot, proj, x, 1)
    }

    /// [`Self::rotate_rows`] by an [`ExtRotation`]: `x` `[n_rows, d_in]` into
    /// the basis of `proj`, f32 `[n_rows, d_in]`, contiguous at offset zero.
    /// One row takes `rot_apply`, as [`Self::rotate`] does; more take
    /// `rot_apply_rows`, as [`Self::rotate_rows`] does.
    pub fn rotate_ext_rows(
        &self,
        rot: &ExtRotation,
        proj: &FusedProj,
        x: &Tensor,
        n_rows: usize,
    ) -> candle_core::Result<Tensor> {
        if proj.rotation != Some(rot.key) {
            candle_core::bail!(
                "{}: quantized under rotation {:?}, asked to rotate by {:?}",
                proj.name,
                proj.rotation,
                rot.key
            );
        }
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != proj.d_in {
            candle_core::bail!("{} expects d_in={}, got {d_in}", proj.name, proj.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != n_rows || n_rows == 0 {
            candle_core::bail!("{}: {rows} rows of activation for {n_rows}", proj.name);
        }
        let x = x.to_dtype(DType::F16)?;
        let op = RotExtOp { rt: self, rot: &rot.buf, name: &proj.name, d_in, n_rows };
        x.apply_op1_no_bwd(&op)
    }
}

/// `rot_apply` (one row) or `rot_apply_rows` (several) as a candle op, by an
/// [`ExtRotation`]. The two served ops ([`RotOp`], [`RotRowsOp`]) find their
/// tables in the runtime's map through the projection's key; this one is
/// handed them. The launches are the same two functions.
struct RotExtOp<'a> {
    rt: &'a FusedRuntime,
    rot: &'a RotBuffers,
    name: &'a str,
    d_in: usize,
    n_rows: usize,
}

impl candle_core::CustomOp1 for RotExtOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-rot-apply-ext"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the LLVQ rotation has no CPU path")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let all = storage.as_cuda_slice::<f16>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous activation", self.name)))?;
        let want = self.d_in * self.n_rows;
        if end - start != want {
            candle_core::bail!(
                "{}: activation of {} values for {} rows of d_in={}",
                self.name,
                end - start,
                self.n_rows,
                self.d_in
            );
        }
        if self.rot.n as usize != self.d_in {
            candle_core::bail!("{}: rotation of width {} for d_in={}", self.name, self.rot.n, self.d_in);
        }
        let mut xr = unsafe { self.rt.device.cuda_stream().alloc::<f32>(want) }
            .map_err(|e| candle_core::Error::msg(format!("alloc rot ext: {e}")))?;
        if self.n_rows == 1 {
            self.rt.launch_rotation(self.rot, all, &mut xr, start as u32)?;
        } else {
            self.rt.launch_rotation_rows(
                self.rot,
                all,
                &mut xr,
                start as u32,
                self.d_in as u32,
                self.n_rows as u32,
            )?;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(xr, self.rt.device.clone()),
            Shape::from(vec![self.n_rows, self.d_in]),
        ))
    }
}
