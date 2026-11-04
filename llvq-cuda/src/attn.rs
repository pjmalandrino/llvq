//! The split-KV decode-attention kernel on the card: compile, launch, check.
//!
//! `kernels/attn_dec.cu` is its own NVRTC translation unit, compiled once per
//! process onto a given stream (candle's, in the model) and never concatenated
//! with the projections' unit, so the served unit keeps its bytes and its
//! printed sha256. The arithmetic is `crate::attn_ref`'s, bit for bit; that is
//! checked on the Mac by executing the kernel text (`tests/
//! attn_dec_matches_rust.rs`) and on the card by [`AttnDec::self_check`], which
//! a caller runs at load and which refuses on one differing bit.
//!
//! The launcher keeps a partials buffer and one ticket counter per KV head.
//! The kernel resets each counter itself; the buffer grows with the longest
//! cache it has seen, under a lock, since the model holds the kernel behind
//! `&self`.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};

use crate::attn_ref::{self, Kv, HD, KMAX, PSTRIDE, SPLIT};
use crate::gpu::{Cuda, FnReport, KernelSource};

/// Threads per CTA, the kernel's `ATTN_THREADS`.
pub const THREADS: u32 = 128;

/// The entry point for a group size and a unit size, or a refusal naming what
/// exists. Qwen3-4B and 8B have 4 query heads per KV head, 14B has 5.
pub fn kernel_name(groups: usize, unit: usize) -> Result<&'static str, String> {
    attn_ref::check_unit(unit)?;
    match (groups, unit) {
        (4, 64) => Ok("attn_dec_g4_u64"),
        (5, 64) => Ok("attn_dec_g5_u64"),
        (4, 32) => Ok("attn_dec_g4_u32"),
        (5, 32) => Ok("attn_dec_g5_u32"),
        (g, _) => Err(format!(
            "attention kernel: {g} query heads per KV head; the kernel compiles 4 (Qwen3-4B, 8B) \
             and 5 (Qwen3-14B)"
        )),
    }
}

/// The compiled translation unit: one module, four entry points.
pub struct AttnModule {
    cuda: Cuda,
    /// sha256 of the exact text NVRTC compiled.
    pub sha256: String,
    /// `Some(dir)` when `LLVQ_KERNEL_DIR` replaced the embedded text.
    pub overridden_from: Option<String>,
}

fn source() -> Result<(KernelSource, Option<String>), String> {
    let set = crate::load_sources_many(&["attn_dec.cu"])?;
    let parts: Vec<&str> = set.parts.iter().map(String::as_str).collect();
    Ok((KernelSource::new(&parts), set.overridden_from))
}

impl AttnModule {
    /// Compile onto an existing stream, candle's in the model.
    pub fn on_stream(stream: Arc<CudaStream>) -> Result<Arc<Self>, String> {
        let (src, overridden_from) = source()?;
        let cuda = Cuda::on_stream(stream, &src)?;
        Ok(Arc::new(Self { cuda, sha256: src.sha256, overridden_from }))
    }

    /// Compile on a fresh stream of device 0, with cudarc's event tracking
    /// off (one stream, ordered by issue): what a bench times.
    pub fn fresh() -> Result<Arc<Self>, String> {
        let (src, overridden_from) = source()?;
        let cuda = Cuda::new_on_fresh_stream(&src)?;
        Ok(Arc::new(Self { cuda, sha256: src.sha256, overridden_from }))
    }

    pub fn cuda(&self) -> &Cuda {
        &self.cuda
    }

    /// The line a journal copies: what was compiled, from where.
    pub fn provenance(&self) -> String {
        match &self.overridden_from {
            None => format!("attn_dec.cu: embedded, sha256 {}, target {}", self.sha256, crate::gpu::arch()),
            Some(d) => format!(
                "attn_dec.cu: OVERRIDDEN from {d} (LLVQ_KERNEL_DIR), sha256 {}, target {}",
                self.sha256,
                crate::gpu::arch()
            ),
        }
    }
}

struct Scratch {
    /// `[n_kv][s_stride][KMAX][G][PSTRIDE]` f32.
    part: CudaSlice<f32>,
    s_stride: usize,
    /// `[n_kv]`, zero at rest; the kernel resets what it draws.
    ticket: CudaSlice<u32>,
}

/// One entry point, sized for one model's geometry.
pub struct AttnDec {
    module: Arc<AttnModule>,
    f: CudaFunction,
    pub name: &'static str,
    pub n_kv: usize,
    pub groups: usize,
    pub unit: usize,
    /// What the driver made of the entry point.
    pub report: FnReport,
    scratch: Mutex<Scratch>,
}

impl AttnDec {
    pub fn new(module: Arc<AttnModule>, n_kv: usize, groups: usize, unit: usize) -> Result<Self, String> {
        let name = kernel_name(groups, unit)?;
        if n_kv == 0 {
            return Err("attention kernel: zero KV heads".into());
        }
        let report = module.cuda.report(name)?;
        let f = module.cuda.func(name)?;
        let s_stride = 1;
        let part = module.cuda.zeros_f32(n_kv * s_stride * KMAX * groups * PSTRIDE)?;
        let ticket = module.cuda.zeros_u32(n_kv)?;
        Ok(Self {
            module,
            f,
            name,
            n_kv,
            groups,
            unit,
            report,
            scratch: Mutex::new(Scratch { part, s_stride, ticket }),
        })
    }

    pub fn module(&self) -> &AttnModule {
        &self.module
    }

    /// The register line every load prints.
    pub fn report_line(&self) -> String {
        format!(
            "{}: {} registers, {} B local (spill), {} B static shared, sm_{}",
            self.name,
            self.report.num_regs,
            self.report.local_bytes,
            self.report.shared_bytes,
            self.report.binary_version
        )
    }

    /// The splits a call at `pos0` of `k` rows reads: the eager grid.
    pub fn eager_grid(pos0: usize, k: usize) -> usize {
        (pos0 + k).div_ceil(SPLIT)
    }

    fn check_call(&self, pos0: usize, k: usize, cap: usize, grid_x: usize) -> Result<(), String> {
        if !(1..=KMAX).contains(&k) {
            return Err(format!("attention kernel: {k} rows in one call, 1..={KMAX} allowed"));
        }
        if !cap.is_multiple_of(SPLIT) {
            return Err(format!(
                "attention kernel: cache capacity {cap} is not a multiple of {SPLIT}; the kernel \
                 stages whole splits"
            ));
        }
        if pos0 + k > cap {
            return Err(format!("attention kernel: rows up to {} past capacity {cap}", pos0 + k));
        }
        if grid_x < Self::eager_grid(pos0, k) {
            return Err(format!(
                "attention kernel: a grid of {grid_x} splits for rows up to {}",
                pos0 + k
            ));
        }
        Ok(())
    }

    /// Grow the partials to hold `splits` splits, and hand back the lock.
    fn scratch_for(&self, splits: usize) -> Result<std::sync::MutexGuard<'_, Scratch>, String> {
        let mut s = self.scratch.lock().map_err(|_| "attention scratch poisoned".to_string())?;
        if splits > s.s_stride {
            let s_stride = splits.next_power_of_two();
            s.part = self
                .module
                .cuda
                .zeros_f32(self.n_kv * s_stride * KMAX * self.groups * PSTRIDE)?;
            s.s_stride = s_stride;
        }
        Ok(s)
    }

    fn config(&self, grid_x: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (grid_x as u32, self.n_kv as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Attention of `k` rows at `pos0..pos0 + k` over caches that already hold
    /// them. `q` is `[n_kv*G][k][HD]`, the caches `[n_kv][cap][HD]`, `out`
    /// `[n_kv*G][k][HD]`, all f16. `grid_x` `None` is the eager grid.
    #[allow(clippy::too_many_arguments)]
    pub fn attend<T>(
        &self,
        q: &CudaView<'_, T>,
        kc: &CudaView<'_, T>,
        vc: &CudaView<'_, T>,
        out: &mut CudaSlice<T>,
        pos0: usize,
        k: usize,
        cap: usize,
        c: f32,
        grid_x: Option<usize>,
    ) -> Result<(), String> {
        let grid_x = grid_x.unwrap_or_else(|| Self::eager_grid(pos0, k));
        self.check_call(pos0, k, cap, grid_x)?;
        self.check_lens(q.len(), kc.len(), vc.len(), out.len(), k, cap)?;
        let mut s = self.scratch_for(Self::eager_grid(pos0, k))?;
        let null = 0u64;
        let (pos0, k, cap, ss) = (pos0 as i32, k as i32, cap as i32, s.s_stride as i32);
        let stream = self.module.cuda.stream();
        let mut b = stream.launch_builder(&self.f);
        let Scratch { part, ticket, .. } = &mut *s;
        b.arg(q).arg(&null).arg(&null).arg(kc).arg(vc).arg(&pos0).arg(&k).arg(&cap).arg(&c)
            .arg(part).arg(ticket).arg(out).arg(&ss);
        unsafe { b.launch(self.config(grid_x)) }.map_err(|e| format!("{}: {e}", self.name))?;
        Ok(())
    }

    /// [`Self::attend`], with the kernel writing the `k` fresh rows `k_new`,
    /// `v_new` (`[n_kv][k][HD]`) into the caches first. The caches are the
    /// caller's own buffers here, hence `&mut`.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_writing<T>(
        &self,
        q: &CudaSlice<T>,
        k_new: &CudaSlice<T>,
        v_new: &CudaSlice<T>,
        kc: &mut CudaSlice<T>,
        vc: &mut CudaSlice<T>,
        out: &mut CudaSlice<T>,
        pos0: usize,
        k: usize,
        cap: usize,
        c: f32,
        grid_x: Option<usize>,
    ) -> Result<(), String> {
        let grid_x = grid_x.unwrap_or_else(|| Self::eager_grid(pos0, k));
        self.check_call(pos0, k, cap, grid_x)?;
        self.check_lens(q.len(), kc.len(), vc.len(), out.len(), k, cap)?;
        let want_new = self.n_kv * k * HD;
        if k_new.len() < want_new || v_new.len() < want_new {
            return Err(format!("attention kernel: fresh rows shorter than {want_new} values"));
        }
        let mut s = self.scratch_for(Self::eager_grid(pos0, k))?;
        let (pos0, k, cap, ss) = (pos0 as i32, k as i32, cap as i32, s.s_stride as i32);
        let stream = self.module.cuda.stream();
        let mut b = stream.launch_builder(&self.f);
        let Scratch { part, ticket, .. } = &mut *s;
        b.arg(q).arg(k_new).arg(v_new).arg(kc).arg(vc).arg(&pos0).arg(&k).arg(&cap).arg(&c)
            .arg(part).arg(ticket).arg(out).arg(&ss);
        unsafe { b.launch(self.config(grid_x)) }.map_err(|e| format!("{}: {e}", self.name))?;
        Ok(())
    }

    fn check_lens(&self, q: usize, kc: usize, vc: usize, out: usize, k: usize, cap: usize) -> Result<(), String> {
        let nq = self.n_kv * self.groups * k * HD;
        let nc = self.n_kv * cap * HD;
        if q < nq || out < nq || kc < nc || vc < nc {
            return Err(format!(
                "attention kernel: buffers q {q}, out {out}, K {kc}, V {vc} values for {nq} query \
                 values and {nc} cache values"
            ));
        }
        Ok(())
    }

    /// The load-time check: the card against the CPU replay, bit for bit.
    ///
    /// Synthetic data of this geometry, rows ending at L in {1, 255, 256,
    /// 257, 1000, 4096} with k in {1, 3, 8}; for each, the eager grid and a
    /// grid of every split the capacity holds plus one (CTAs with no work
    /// still draw their ticket); the k rows of one verify call against k
    /// decode calls on the card; and one call where the kernel writes the
    /// fresh rows itself. One differing bit is an error naming the call, the
    /// head, the row and the dim; the caller stops the load on it.
    pub fn self_check(&self) -> Result<String, String> {
        let cuda = &self.module.cuda;
        let c = attn_ref::score_scale(HD);
        let (n_kv, g) = (self.n_kv, self.groups);
        let mut calls = 0usize;
        let cases: Vec<(usize, usize)> = [1usize, 255, 256, 257, 1000, 4096]
            .iter()
            .flat_map(|&l| [1usize, 3, 8].map(move |k| (l.max(k) - k, k)))
            .collect();
        // Replays first, in parallel: they are the slow half.
        let datas: Vec<_> = std::thread::scope(|sc| {
            let hs: Vec<_> = cases
                .iter()
                .enumerate()
                .map(|(i, &(pos0, k))| {
                    sc.spawn(move || {
                        let cap = (pos0 + k).div_ceil(SPLIT) * SPLIT;
                        let seed = 0x5eed_0000 + i as u64;
                        let kc = attn_ref::fill(n_kv * cap * HD, seed ^ 1, 1.0);
                        let vc = attn_ref::fill(n_kv * cap * HD, seed ^ 2, 1.0);
                        let q = attn_ref::fill(n_kv * g * k * HD, seed ^ 3, 4.0);
                        let want =
                            attn_ref::attend(&q, &Kv { k: &kc, v: &vc, cap }, n_kv, g, pos0, k, c, self.unit);
                        (pos0, k, cap, q, kc, vc, want)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("replay thread")).collect()
        });
        for (pos0, k, cap, q, kc, vc, want) in &datas {
            let (pos0, k, cap) = (*pos0, *k, *cap);
            let dq = cuda.up_u16(q)?;
            let dk = cuda.up_u16(kc)?;
            let dv = cuda.up_u16(vc)?;
            let mut out = cuda.zeros_u16(want.len())?;
            for grid in [None, Some(cap / SPLIT + 1)] {
                self.attend(&dq.slice(..), &dk.slice(..), &dv.slice(..), &mut out, pos0, k, cap, c, grid)?;
                calls += 1;
                let got = cuda.down_u16(&out)?;
                diff(&got, want, k, &format!("{} rows {}..={} grid {:?}", self.name, pos0 + 1, pos0 + k, grid))?;
            }
            if k > 1 {
                // Each row of the verify call, as a decode call of its own.
                for r in 0..k {
                    let mut q1 = Vec::with_capacity(n_kv * g * HD);
                    for hq in 0..n_kv * g {
                        q1.extend_from_slice(&q[(hq * k + r) * HD..(hq * k + r + 1) * HD]);
                    }
                    let dq1 = cuda.up_u16(&q1)?;
                    let mut out1 = cuda.zeros_u16(q1.len())?;
                    self.attend(&dq1.slice(..), &dk.slice(..), &dv.slice(..), &mut out1, pos0 + r, 1, cap, c, None)?;
                    calls += 1;
                    let got = cuda.down_u16(&out1)?;
                    let mut row = Vec::with_capacity(q1.len());
                    for hq in 0..n_kv * g {
                        row.extend_from_slice(&want[(hq * k + r) * HD..(hq * k + r + 1) * HD]);
                    }
                    diff(&got, &row, 1, &format!("{} row {} of a verify call, alone", self.name, pos0 + r + 1))?;
                }
            }
        }
        // The fresh-row write: the cache holds garbage at the new positions
        // until the kernel writes them.
        let (pos0, k) = (300usize, 8usize);
        let cap = (pos0 + k).div_ceil(SPLIT) * SPLIT;
        let mut kc = attn_ref::fill(n_kv * cap * HD, 77, 1.0);
        let mut vc = attn_ref::fill(n_kv * cap * HD, 78, 1.0);
        let kn = attn_ref::fill(n_kv * k * HD, 79, 1.0);
        let vn = attn_ref::fill(n_kv * k * HD, 80, 1.0);
        let q = attn_ref::fill(n_kv * g * k * HD, 81, 4.0);
        let (mut dk, mut dv) = (cuda.up_u16(&kc)?, cuda.up_u16(&vc)?);
        let (dkn, dvn, dq) = (cuda.up_u16(&kn)?, cuda.up_u16(&vn)?, cuda.up_u16(&q)?);
        let mut out = cuda.zeros_u16(q.len())?;
        self.attend_writing(&dq, &dkn, &dvn, &mut dk, &mut dv, &mut out, pos0, k, cap, c, None)?;
        calls += 1;
        for h in 0..n_kv {
            for r in 0..k {
                let (dst, src) = ((h * cap + pos0 + r) * HD, (h * k + r) * HD);
                kc[dst..dst + HD].copy_from_slice(&kn[src..src + HD]);
                vc[dst..dst + HD].copy_from_slice(&vn[src..src + HD]);
            }
        }
        if cuda.down_u16(&dk)? != kc || cuda.down_u16(&dv)? != vc {
            return Err(format!("{}: the fresh rows were not written where they belong", self.name));
        }
        let want = attn_ref::attend(&q, &Kv { k: &kc, v: &vc, cap }, n_kv, g, pos0, k, c, self.unit);
        diff(&cuda.down_u16(&out)?, &want, k, &format!("{} with fresh rows written by the kernel", self.name))?;
        Ok(format!(
            "{}: the card equals the CPU replay bit for bit on {calls} calls (rows ending at 1, 255, \
             256, 257, 1000, 4096; k 1, 3, 8; eager and wide grids; verify rows alone; fresh rows)",
            self.name
        ))
    }
}

fn diff(got: &[u16], want: &[u16], k: usize, what: &str) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("{what}: {} values back, {} expected", got.len(), want.len()));
    }
    match got.iter().zip(want).position(|(a, b)| a != b) {
        None => Ok(()),
        Some(i) => Err(format!(
            "{what}: query head {}, row {}, dim {}: card {:#06x}, CPU replay {:#06x}",
            i / (k * HD),
            (i / HD) % k,
            i % HD,
            got[i],
            want[i]
        )),
    }
}
