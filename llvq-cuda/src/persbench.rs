//! The persistent sections of `planesbench` (Linux, CUDA): step L1 (the
//! one-launch Tetra pass `tpersall`, the barrier and launch microbenchmarks,
//! the driver cubins) and step L2 (the served `tv_tetra48_h` reference and
//! the twelve persistent arms). Preregs: `proofs/preregistration-pers-l1-*`
//! and `-l2-*`. Portable half and its tests: [`crate::pers`].
//!
//! Everything here is off unless a variable asks for it, and nothing here
//! touches the bench unit or the table: the persistent kernels live in their
//! own NVRTC module on the bench's stream, and the served reference in a
//! third. Every arm is checked before it is timed; an arm that fails its
//! check is reported and never timed, and the job goes on (A3's rule).

use crate::gpu::{Cuda, KernelSource};
use crate::pers::{self, PersLaunch, RowVerdict};
use cudarc::driver::{sys, CudaFunction, CudaSlice, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use std::path::PathBuf;
use std::time::Instant;

// SAFETY: `PersLaunch` is `#[repr(C)]`, plain integers, and its layout is the
// kernel's (`tests/pers_matches_served.rs::the_struct_mirrors_match_the_kernel_layouts`).
unsafe impl DeviceRepr for PersLaunch {}
// SAFETY: the same for route P's per-call struct (`kernels/pers_segment.cu`).
unsafe impl DeviceRepr for pers::SegLaunch {}
// SAFETY: the same for 4E's per-launch struct (`kernels/pers_4e.cu`).
unsafe impl DeviceRepr for pers::P4eLaunch {}

/// What the environment asks of the persistent sections.
#[derive(Clone, Debug, Default)]
pub struct Flags {
    /// `LLVQ_TETRA_OCC=1`: the one-launch Tetra pass of step L1.
    pub tetra_occ: bool,
    /// `LLVQ_PERS_MICRO=1`: the barrier, ring and launch microbenchmarks.
    pub micro: bool,
    /// `LLVQ_PERS_ARMS`: the L2 arms, in registration order.
    pub arms: Vec<usize>,
    /// `LLVQ_DUMP_CUBIN=<dir>`: the driver-JIT cubin of every module loaded.
    pub dump_cubin: Option<PathBuf>,
}

fn flag01(name: &str) -> Result<bool, String> {
    match std::env::var(name).as_deref() {
        Err(_) => Ok(false),
        Ok("1") => Ok(true),
        Ok("0") => Ok(false),
        Ok(v) => Err(format!("{name}={v:?}: the values are 0 and 1, and unset is 0")),
    }
}

impl Flags {
    /// Read and check the four variables. The Tetra sections are read against
    /// the fused geometry (the A3 grammar), so they need `LLVQ_SEG_TETRA=1`
    /// and the tetra48 arm.
    pub fn from_env(seg_tetra: bool, tetra48: bool) -> Result<Self, String> {
        let f = Flags {
            tetra_occ: flag01("LLVQ_TETRA_OCC")?,
            micro: flag01("LLVQ_PERS_MICRO")?,
            arms: pers::parse_pers_arms(std::env::var("LLVQ_PERS_ARMS").ok().as_deref())?,
            dump_cubin: std::env::var("LLVQ_DUMP_CUBIN").ok().map(PathBuf::from),
        };
        if (f.tetra_occ || !f.arms.is_empty()) && !(seg_tetra && tetra48) {
            return Err("LLVQ_TETRA_OCC and LLVQ_PERS_ARMS are read against the fused Tetra geometry: they \
                        need LLVQ_SEG_TETRA=1 and tetra48 in LLVQ_BENCH_ARMS"
                .into());
        }
        Ok(f)
    }

    pub fn any(&self) -> bool {
        self.tetra_occ || self.micro || !self.arms.is_empty()
    }
}

/// The driver's own JIT of `ptx`, as a cubin: `cuLinkCreate`, `cuLinkAddData`,
/// `cuLinkComplete`. That is the code the card runs, compiled by the driver's
/// ptxas, which a devel-image `cuobjdump` does not show.
pub fn driver_cubin(ptx: &str, name: &str) -> Result<Vec<u8>, String> {
    let ptx_c = std::ffi::CString::new(ptx).map_err(|e| e.to_string())?;
    let name_c = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
    let mut state: sys::CUlinkState = std::ptr::null_mut();
    // SAFETY: plain driver calls on the current context; every pointer
    // outlives its call; the link state is destroyed on every path, and the
    // image is copied out before it is.
    unsafe {
        let r = sys::cuLinkCreate_v2(0, std::ptr::null_mut(), std::ptr::null_mut(), &mut state);
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuLinkCreate: {r:?}"));
        }
        let r = sys::cuLinkAddData_v2(
            state,
            sys::CUjitInputType::CU_JIT_INPUT_PTX,
            ptx_c.as_ptr() as *mut std::ffi::c_void,
            ptx_c.as_bytes_with_nul().len(),
            name_c.as_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        if r != sys::CUresult::CUDA_SUCCESS {
            sys::cuLinkDestroy(state);
            return Err(format!("cuLinkAddData: {r:?}"));
        }
        let mut cubin: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut size: usize = 0;
        let r = sys::cuLinkComplete(state, &mut cubin, &mut size);
        if r != sys::CUresult::CUDA_SUCCESS {
            sys::cuLinkDestroy(state);
            return Err(format!("cuLinkComplete: {r:?}"));
        }
        let bytes = std::slice::from_raw_parts(cubin as *const u8, size).to_vec();
        sys::cuLinkDestroy(state);
        Ok(bytes)
    }
}

/// Compile `text` as the loader does and write the driver's cubin of it to
/// `dir/<name>.cubin`, printing its size and sha256.
pub fn dump_cubin(dir: &std::path::Path, name: &str, text: &str) -> Result<(), String> {
    let opts = cudarc::nvrtc::CompileOptions { arch: Some(crate::gpu::arch()), ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(text, opts).map_err(|e| format!("{name}: NVRTC: {e}"))?;
    let cubin = driver_cubin(&ptx.to_src(), name)?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{name}.cubin"));
    std::fs::write(&path, &cubin).map_err(|e| format!("{}: {e}", path.display()))?;
    std::fs::write(dir.join(format!("{name}.ptx")), ptx.to_src()).map_err(|e| e.to_string())?;
    println!(
        "  cubin {name}: {} bytes, sha256 {} (driver JIT of a {}-byte unit, sha256 {})",
        cubin.len(),
        crate::sha256::sha256_hex(&cubin),
        text.len(),
        crate::sha256::sha256_hex(text.as_bytes())
    );
    Ok(())
}

fn ptr<T>(s: &CudaSlice<T>, cuda: &Cuda) -> u64 {
    s.device_ptr(cuda.stream()).0
}

/// One site of the one-launch pass, with its reference rows (the A4 kernel's
/// own output on the same site) and an f64 reference for a row that misses.
pub struct PersallSite<'a> {
    pub name: String,
    pub words: u64,
    pub gscale: u64,
    pub gs_off: u64,
    pub rscale: u64,
    pub tail: u64,
    pub d_out: u32,
    pub nblocks: u32,
    pub tail_w: u32,
    pub stride_u32: u32,
    pub reference: Vec<f32>,
    pub f64_row: Box<dyn Fn(usize) -> (f64, f64) + 'a>,
}

/// One matrix of the bare file for the L2 arms, in served order.
pub struct L2Site<'a> {
    pub name: String,
    pub words: u64,
    pub gscale: u64,
    pub rscale: u64,
    pub tail_f16: Vec<u16>,
    pub d_out: u32,
    pub d_in: u32,
    pub stride_u32: u32,
    pub f64_row: Box<dyn Fn(usize) -> (f64, f64) + 'a>,
}

/// The decoder tables, device pointers.
#[derive(Clone, Copy)]
pub struct Tables {
    pub rows: u64,
    pub pref: u64,
    pub br: u64,
    pub suf: u64,
    pub invnorm: u64,
}

struct Persall<'a> {
    f: CudaFunction,
    table: CudaSlice<u64>,
    y: CudaSlice<f32>,
    grid: u32,
    shared: u32,
    total_groups: u32,
    nsites: u32,
    valid: bool,
    series: Vec<f64>,
    _sites: std::marker::PhantomData<&'a ()>,
}

struct L2Arm {
    idx: usize,
    f: CudaFunction,
    shared: u32,
    valid: bool,
    series: Vec<f64>,
}

struct L2<'a> {
    sites: Vec<L2Site<'a>>,
    schedule: Vec<pers::Site>,
    served: Cuda,
    f_ref: CudaFunction,
    y_ref: CudaSlice<u16>,
    ref_rows: Vec<u16>,
    ref_series: Vec<f64>,
    tails: Vec<CudaSlice<u16>>,
    args: CudaSlice<u8>,
    y_all: CudaSlice<u16>,
    dbg: CudaSlice<f32>,
    dbg_norm: CudaSlice<u16>,
    dbg_rot: CudaSlice<f32>,
    bar: CudaSlice<u32>,
    cnts: CudaSlice<u32>,
    abort: CudaSlice<u32>,
    _keep: Keep,
    arms: Vec<L2Arm>,
    ledger: pers::Ledger,
    grid: u32,
    row0: Vec<usize>,
    x: u64,
    tables: Tables,
    tile_bytes: u32,
}

/// The persistent sections of one bench run.
pub struct Section<'a> {
    flags: Flags,
    module: Cuda,
    sm: u32,
    persall: Option<Persall<'a>>,
    l2: Option<L2<'a>>,
}

impl<'a> Section<'a> {
    /// Compile the persistent module on the bench's stream, print its digest,
    /// dump its cubin (and the bench unit's) when asked.
    pub fn new(flags: Flags, bench: &Cuda, bench_text: &str, tile_define: &str) -> Result<Self, String> {
        let text = pers::pers_unit_text(tile_define);
        let src = KernelSource::new(&[text.as_str()]);
        println!("\n  Persistent module: {} bytes, sha256 {}", src.text.len(), src.sha256);
        let module = Cuda::on_stream(bench.stream().clone(), &src)?;
        let sm = bench.device()?.sm_count as u32;
        if let Some(dir) = &flags.dump_cubin {
            dump_cubin(dir, "bench", bench_text)?;
            dump_cubin(dir, "persistent", &src.text)?;
        }
        Ok(Section { flags, module, sm, persall: None, l2: None })
    }

    /// Step L1: the barrier, ring and launch microbenchmarks. Each is one
    /// cooperative launch of `n` hops, 7 launches with 2 discarded, timed on
    /// the host around launch and sync.
    pub fn micro(&self) -> Result<(), String> {
        if !self.flags.micro {
            return Ok(());
        }
        const N: u32 = 10_000;
        const REPS: usize = 7;
        const DISCARD: usize = 2;
        let hdr = pers::SMEM_HEADER;
        let fb = self.module.func("tv_pers_bar")?;
        let fr = self.module.func("tv_pers_ring")?;
        let fe = self.module.func("tv_pers_empty")?;
        let mut abort = self.module.up_u32(&[0])?;
        println!("\n  Microbenchmarks (L1): {N} hops a launch, {REPS} launches, {DISCARD} discarded, median [range]");
        for (k, threads) in [(1u32, 768u32), (2, 512), (6, 256)] {
            let occ = fb
                .occupancy_max_active_blocks_per_multiprocessor(threads, hdr as usize, None)
                .map_err(|e| format!("occupancy: {e}"))?;
            if occ < k {
                println!("  barrier {}x{threads}: NOT RUN, {occ} resident a SM, {k} needed", self.sm * k);
                continue;
            }
            let grid = self.sm * k;
            let mut bar = self.module.up_u32(&[0])?;
            let mut t = Vec::new();
            for rep in 0..REPS {
                let base = (rep as u32).wrapping_mul(N).wrapping_mul(grid);
                let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: hdr };
                let t0 = Instant::now();
                let mut b = self.module.stream().launch_builder(&fb);
                b.arg(&mut bar).arg(&base).arg(&N).arg(&mut abort);
                // SAFETY: the grid passed the occupancy check, the arguments
                // are the kernel's, and a cooperative launch refuses a grid
                // that cannot be co-resident.
                unsafe { b.launch_cooperative(cfg) }.map_err(|e| format!("tv_pers_bar: {e}"))?;
                self.module.sync()?;
                t.push(t0.elapsed().as_secs_f64() / N as f64 * 1e6);
            }
            if self.module.down_u32(&abort)?[0] != 0 {
                println!("  barrier {grid}x{threads}: ABORTED on the watchdog, not reported");
                abort = self.module.up_u32(&[0])?;
                continue;
            }
            let (m, lo, hi) = pers::med_range(&t[DISCARD..]);
            println!("  barrier {grid:>4} x {threads:<4}       {m:7.3} us a barrier [{lo:.3}-{hi:.3}]");
        }
        {
            let ring = 16u32;
            let mut flag = self.module.up_u32(&[0])?;
            let mut t = Vec::new();
            for rep in 0..REPS {
                let base = (rep as u32).wrapping_mul(N).wrapping_mul(ring);
                let cfg = LaunchConfig { grid_dim: (ring, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: hdr };
                let t0 = Instant::now();
                let mut b = self.module.stream().launch_builder(&fr);
                b.arg(&mut flag).arg(&base).arg(&N).arg(&mut abort);
                // SAFETY: as above; 16 CTAs of 32 threads are co-resident on any card here.
                unsafe { b.launch_cooperative(cfg) }.map_err(|e| format!("tv_pers_ring: {e}"))?;
                self.module.sync()?;
                t.push(t0.elapsed().as_secs_f64() / (N as f64 * ring as f64) * 1e6);
            }
            let (m, lo, hi) = pers::med_range(&t[DISCARD..]);
            println!("  ring of {ring} CTAs           {m:7.3} us a hop [{lo:.3}-{hi:.3}]");
        }
        for coop in [false, true] {
            let mut t = Vec::new();
            let launches = 10_000u32;
            for _ in 0..REPS {
                let cfg = LaunchConfig { grid_dim: (self.sm, 1, 1), block_dim: (768, 1, 1), shared_mem_bytes: 0 };
                let t0 = Instant::now();
                for i in 0..launches {
                    let mut b = self.module.stream().launch_builder(&fe);
                    b.arg(&i);
                    // SAFETY: an empty kernel of one u32 argument.
                    unsafe {
                        if coop {
                            b.launch_cooperative(cfg).map(|_| ())
                        } else {
                            b.launch(cfg).map(|_| ())
                        }
                    }
                    .map_err(|e| format!("tv_pers_empty: {e}"))?;
                }
                self.module.sync()?;
                t.push(t0.elapsed().as_secs_f64() / launches as f64 * 1e6);
            }
            let (m, lo, hi) = pers::med_range(&t[DISCARD..]);
            println!(
                "  empty {}x768, {}  {m:7.3} us a launch [{lo:.3}-{hi:.3}]",
                self.sm,
                if coop { "cooperative" } else { "normal     " }
            );
        }
        Ok(())
    }

    /// Step L1: build `tv_tetra48_persall` over the fused sites, then check
    /// every row against the A4 kernel's row on the same site.
    pub fn persall_setup(&mut self, sites: Vec<PersallSite<'a>>, tables: Tables, x: u64, tile_bytes: u32) -> Result<(), String> {
        if !self.flags.tetra_occ {
            return Ok(());
        }
        let d_outs: Vec<u32> = sites.iter().map(|s| s.d_out).collect();
        let (g0, total_groups) = pers::persall_groups(&d_outs);
        let rows: usize = d_outs.iter().map(|&d| d as usize).sum();
        let y = self.module.zeros_f32(rows)?;
        let ybase = ptr(&y, &self.module);
        let mut words = Vec::with_capacity(sites.len() * 8);
        let mut at = 0u64;
        for (k, s) in sites.iter().enumerate() {
            words.extend(pers::persall_site_words(
                s.words, s.gscale, s.gs_off, s.rscale, s.tail, ybase + 4 * at, s.nblocks, s.tail_w, s.stride_u32,
                g0[k], s.d_out / pers::PERSALL_GROUP_ROWS,
            ));
            at += s.d_out as u64;
        }
        let table = self.module.up_u64(&words)?;
        let f = self.module.func_dynamic_shared("tv_tetra48_persall", tile_bytes)?;
        let rep = self.module.report("tv_tetra48_persall")?;
        let occ = f
            .occupancy_max_active_blocks_per_multiprocessor(256, tile_bytes as usize, None)
            .map_err(|e| format!("occupancy: {e}"))?;
        let grid = total_groups.min(occ * self.sm).max(1);
        println!(
            "\n  One-launch Tetra pass (L1, tpersall): {} sites, {rows} rows, {total_groups} groups of 8, grid {grid} \
             ({occ} a SM x {}), {} registers, {} B local",
            sites.len(),
            self.sm,
            rep.num_regs,
            rep.local_bytes
        );
        let mut p = Persall {
            f,
            table,
            y,
            grid,
            shared: tile_bytes,
            total_groups,
            nsites: sites.len() as u32,
            valid: rep.local_bytes == 0,
            series: Vec::new(),
            _sites: std::marker::PhantomData,
        };
        if !p.valid {
            println!("  tpersall: SPILLS {} B, not timed (gate 8)", rep.local_bytes);
        } else {
            self.persall_launch(&mut p, tables, x)?;
            self.module.sync()?;
            let all = self.module.down_f32(&p.y)?;
            let (mut same, mut moved, mut worst, mut at) = (0usize, 0usize, 0.0f64, 0usize);
            for s in &sites {
                for r in 0..s.d_out as usize {
                    let (got, want) = (all[at + r], s.reference[r]);
                    if got.to_bits() == want.to_bits() {
                        same += 1;
                        continue;
                    }
                    let (v, mag) = (s.f64_row)(r);
                    let err = (got as f64 - v).abs() / mag.max(1.0);
                    worst = worst.max(err);
                    if err <= pers::F32_TOL {
                        moved += 1;
                        if moved == 1 {
                            println!("  first moved row: {} row {r}: {got:e} against {want:e}, f64 error {err:.1e}", s.name);
                        }
                    } else {
                        p.valid = false;
                        println!("  tpersall INVALIDATED: {} row {r}: {got:e} against {want:e}, f64 error {err:.1e}", s.name);
                        break;
                    }
                }
                at += s.d_out as usize;
                if !p.valid {
                    break;
                }
            }
            println!(
                "  tpersall: {same} rows identical BIT FOR BIT to the A4 kernels{}{}",
                if moved > 0 { format!(", {moved} with a moved association (worst {worst:.1e})") } else { String::new() },
                if p.valid { "" } else { "; INVALIDATED, not timed" }
            );
        }
        self.persall = Some(p);
        Ok(())
    }

    fn persall_launch(&self, p: &mut Persall<'a>, t: Tables, x: u64) -> Result<(), String> {
        let cfg = LaunchConfig { grid_dim: (p.grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: p.shared };
        let mut b = self.module.stream().launch_builder(&p.f);
        b.arg(&p.table).arg(&p.nsites).arg(&t.rows).arg(&t.pref).arg(&t.br).arg(&t.suf).arg(&t.invnorm).arg(&x).arg(&p.total_groups);
        // SAFETY: the arguments are the kernel's; every pointer in the site
        // table points into a live allocation of the bench.
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_tetra48_persall: {e}"))?;
        Ok(())
    }

    /// Step L2: the served unit as the reference, then the arms. `sites` are
    /// the 252 matrices of the bare file in served order.
    pub fn l2_setup(&mut self, sites: Vec<L2Site<'a>>, tables: Tables, x: u64, x_len: usize, tile_define: &str) -> Result<(), String> {
        if self.flags.arms.is_empty() {
            return Ok(());
        }
        let schedule = pers::served_4b_schedule();
        if sites.len() != schedule.len() {
            return Err(format!("LLVQ_PERS_ARMS: {} Tetra matrices, the served pattern has {}", sites.len(), schedule.len()));
        }
        for (s, st) in sites.iter().zip(&schedule) {
            if s.d_out != st.d_out || s.d_in != st.d_in {
                return Err(format!("{}: {}x{} where the served pattern puts {}x{}", s.name, s.d_out, s.d_in, st.d_out, st.d_in));
            }
        }
        if x_len < 9728 {
            return Err(format!("the bench activation has {x_len} values, the persistent arms read 9,728"));
        }
        let m = &self.module;
        // The reference: the served unit, assembled and compiled as the
        // served door does, its digest printed for the job's gate.
        let served_text = pers::served_unit_text(&format!("{tile_define}{}", crate::tile::Prefill::resolve()?.defines()));
        let ssrc = KernelSource::new(&[served_text.as_str()]);
        println!("\n  Persistent arms (L2). served unit: {} bytes, sha256 {}", ssrc.text.len(), ssrc.sha256);
        let served = Cuda::on_stream(m.stream().clone(), &ssrc)?;
        if let Some(dir) = &self.flags.dump_cubin {
            dump_cubin(dir, "served", &ssrc.text)?;
        }
        let tile_blocks: u32 = tile_define
            .trim()
            .trim_start_matches("#define TILE_BLOCKS ")
            .trim_end_matches('u')
            .parse()
            .map_err(|e| format!("tile define {tile_define:?}: {e}"))?;
        let tile_bytes = tile_blocks * 24 * 4;
        let f_ref = served.func("tv_tetra48_h")?;
        let total: usize = sites.iter().map(|s| s.d_out as usize).sum();
        let row0: Vec<usize> = sites.iter().scan(0usize, |a, s| {
            let r = *a;
            *a += s.d_out as usize;
            Some(r)
        }).collect();
        let tails = sites.iter().map(|s| m.up_u16(&s.tail_f16)).collect::<Result<Vec<_>, _>>()?;
        let y_ref = served.zeros_u16(total)?;
        let grid = self.sm;
        // Per-site pointer tables, the sites, the slices, the prologue inputs.
        let (lo, hi) = pers::slice_tables(&schedule, grid);
        let recs: Vec<u32> = schedule
            .iter()
            .zip(&sites)
            .zip(&row0)
            .flat_map(|((st, s), &r0)| pers::site_record(st, s.words, r0 as u32, s.stride_u32))
            .collect();
        let gs: Vec<u64> = sites.iter().map(|s| s.gscale).collect();
        let rs: Vec<u64> = sites.iter().map(|s| s.rscale).collect();
        let tl: Vec<u64> = tails.iter().map(|t| ptr(t, m)).collect();
        let (h, i) = (2560usize, 9728usize);
        let (rh, ri) = (llvq_quant::rotation::Rotation::new(h, 0x9e55), llvq_quant::rotation::Rotation::new(i, 0x9e56));
        let pack = |r: &llvq_quant::rotation::Rotation| {
            let n = r.dim();
            let mut signs = vec![0u32; n.div_ceil(32)];
            for (j, &v) in r.signs().iter().enumerate() {
                if v < 0.0 {
                    signs[j >> 5] |= 1 << (j & 31);
                }
            }
            let (mm, k) = (r.pow2(), r.odd());
            let mut small = vec![0.0f32; 1024];
            for g in 0..k {
                for t in 0..k {
                    small[g * 32 + t] = r.small()[g * k + t] as f32;
                }
            }
            (signs, small, mm as u32, k as u32, (1.0f64 / (mm as f64).sqrt()) as f32)
        };
        let (sg_h, sm_h, m_h, k_h, inv_h) = pack(&rh);
        let (sg_i, sm_i, m_i, k_i, inv_i) = pack(&ri);
        let mut rng = llvq_core::SplitMix64::new(0x9e57);
        let resid: Vec<u16> = (0..i).map(|_| crate::f16_bits(rng.next_gaussian() as f32)).collect();
        let alpha: Vec<u16> = (0..h).map(|_| crate::f16_bits(1.0 + 0.1 * rng.next_gaussian() as f32)).collect();
        let d_recs = m.up_u32(&recs)?;
        let d_gs = m.up_u64(&gs)?;
        let d_rs = m.up_u64(&rs)?;
        let d_tl = m.up_u64(&tl)?;
        let d_lo = m.up_u32(&lo)?;
        let d_hi = m.up_u32(&hi)?;
        let d_resid = m.up_u16(&resid)?;
        let d_alpha = m.up_u16(&alpha)?;
        let d_sgh = m.up_u32(&sg_h)?;
        let d_sgi = m.up_u32(&sg_i)?;
        let d_smh = m.up_f32(&sm_h)?;
        let d_smi = m.up_f32(&sm_i)?;
        let y_all = m.zeros_u16(total)?;
        let dbg = m.zeros_f32(total)?;
        let n_norm = 72usize;
        let n_prol = 96usize;
        let dbg_norm = m.zeros_u16(n_norm * h)?;
        let dbg_rot = m.zeros_f32(n_prol * i)?;
        let bar = m.up_u32(&[0])?;
        let cnts = m.up_u32(&vec![0u32; pers::N_PERS_ARMS * pers::counter_region(schedule.len())])?;
        let abort = m.up_u32(&[0])?;
        let ptrs = [
            ptr(&d_recs, m), tables.rows, tables.pref, tables.br, tables.suf, tables.invnorm, ptr(&d_gs, m), ptr(&d_rs, m),
            ptr(&d_tl, m), x, ptr(&y_all, m), ptr(&d_lo, m), ptr(&d_hi, m), ptr(&bar, m), ptr(&cnts, m), ptr(&abort, m),
            ptr(&d_resid, m), ptr(&d_alpha, m), ptr(&d_sgh, m), ptr(&d_sgi, m), ptr(&d_smh, m), ptr(&d_smi, m),
        ];
        let img = pers::PersArgsHost {
            ptrs,
            nsites: schedule.len() as u32,
            hidden: h as u32,
            inter: i as u32,
            m_h,
            k_h,
            m_i,
            k_i,
            inv_h,
            inv_i,
            eps: 1e-6,
        }
        .bytes();
        let args = m.stream().clone_htod(&img).map_err(|e| format!("H2D args: {e}"))?;
        // The device buffers the pointers name must outlive every launch.
        let keep = Keep {
            u32s: vec![d_recs, d_lo, d_hi, d_sgh, d_sgi],
            u64s: vec![d_gs, d_rs, d_tl],
            u16s: vec![d_resid, d_alpha],
            f32s: vec![d_smh, d_smi],
        };
        let mut l2 = L2 {
            sites,
            schedule,
            served,
            f_ref,
            y_ref,
            ref_rows: Vec::new(),
            ref_series: Vec::new(),
            tails,
            args,
            y_all,
            dbg,
            dbg_norm,
            dbg_rot,
            bar,
            cnts,
            abort,
            _keep: keep,
            arms: Vec::new(),
            ledger: pers::Ledger::new(pers::N_PERS_ARMS),
            grid,
            row0,
            x,
            tables,
            tile_bytes,
        };
        // The reference rows, once.
        l2_ref_pass(&l2, m)?;
        m.sync()?;
        l2.ref_rows = l2.served.down_u16(&l2.y_ref)?;

        // The prologue references: candle's tree transcribed (pers_ref.cu)
        // and rot_apply from the served unit.
        let want_norm = {
            let f = m.func("pers_ref_rmsnorm")?;
            let d_out = m.zeros_u16(h)?;
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
            let mut b = m.stream().launch_builder(&f);
            let (ncols, block, eps) = (h as i32, 1024i32, 1e-6f32);
            b.arg(&ptrs[16]).arg(&d_out).arg(&ptrs[17]).arg(&ncols).arg(&block).arg(&eps);
            // SAFETY: one CTA of 1,024 threads on live buffers of hidden width.
            unsafe { b.launch(cfg) }.map_err(|e| format!("pers_ref_rmsnorm: {e}"))?;
            m.sync()?;
            m.down_u16(&d_out)?
        };
        let rot_of = |input: &[u16], n: usize, signs: &[u32], small: &[f32], mm: u32, k: u32, inv: f32| -> Result<Vec<f32>, String> {
            let f = l2.served.func("rot_apply")?;
            let d_in = m.up_u16(input)?;
            let d_s = m.up_u32(signs)?;
            let d_sm = m.up_f32(small)?;
            let d_o = m.zeros_f32(n)?;
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 4 * n as u32 };
            let mut b = m.stream().launch_builder(&f);
            let (nn, zero) = (n as u32, 0u32);
            b.arg(&d_in).arg(&d_s).arg(&d_sm).arg(&d_o).arg(&nn).arg(&mm).arg(&k).arg(&inv).arg(&zero);
            // SAFETY: rot_apply's own arguments, one CTA, n floats of shared.
            unsafe { b.launch(cfg) }.map_err(|e| format!("rot_apply: {e}"))?;
            m.sync()?;
            m.down_f32(&d_o)
        };
        let want_rot_h = rot_of(&want_norm, h, &sg_h, &sm_h, m_h, k_h, inv_h)?;
        let want_rot_i = rot_of(&resid, i, &sg_i, &sm_i, m_i, k_i, inv_i)?;

        for &a in &self.flags.arms.clone() {
            let arm = &pers::PERS_ARMS[a];
            let shared = pers::shared_bytes(arm);
            let f = m.func_dynamic_shared(arm.symbol, shared)?;
            let rep = m.report(arm.symbol)?;
            let occ = f
                .occupancy_max_active_blocks_per_multiprocessor(arm.threads, shared as usize, None)
                .map_err(|e| format!("occupancy: {e}"))?;
            let mut valid = rep.local_bytes == 0 && occ >= 1;
            print!("  {:<13} {} threads, {} B shared, {} registers, {} B local, {} resident a SM", arm.name, arm.threads, shared, rep.num_regs, rep.local_bytes, occ);
            if !valid {
                println!(": NOT TIMED (gate 8)");
                l2.arms.push(L2Arm { idx: a, f, shared, valid, series: Vec::new() });
                continue;
            }
            println!();
            let mut l2arm = L2Arm { idx: a, f, shared, valid, series: Vec::new() };
            // Identity launch: poison, debug buffers on.
            let poison = vec![pers::POISON_H; l2.row0.last().copied().unwrap_or(0) + l2.sites.last().map_or(0, |s| s.d_out as usize)];
            m.stream().memcpy_htod(&poison, &mut l2.y_all).map_err(|e| format!("poison: {e}"))?;
            let ok = l2_launch(&mut l2, &l2arm, m, true)?;
            if !ok {
                println!("  {}: ABORTED on the watchdog: NOT TIMED (gate 8)", arm.name);
                valid = false;
            } else {
                let got = m.down_u16(&l2.y_all)?;
                let f32s = m.down_f32(&l2.dbg)?;
                let (mut same, mut moved, mut unwritten, mut worst) = (0usize, 0usize, 0usize, 0.0f64);
                for (k, s) in l2.sites.iter().enumerate() {
                    for r in 0..s.d_out as usize {
                        let at = l2.row0[k] + r;
                        if !arm.owes_identity() {
                            if got[at] == pers::POISON_H {
                                unwritten += 1;
                            }
                            continue;
                        }
                        let (want_h, got_h) = (l2.ref_rows[at], got[at]);
                        if want_h == got_h {
                            same += 1;
                            continue;
                        }
                        let (v, mag) = (s.f64_row)(r);
                        match pers::judge_row(want_h, got_h, f32s[at], v, mag) {
                            RowVerdict::Identical => same += 1,
                            RowVerdict::Moved { f32_err, .. } => {
                                moved += 1;
                                worst = worst.max(f32_err);
                                if moved == 1 {
                                    println!("    first moved row: {} row {r}: {got_h:#06x} against {want_h:#06x}, f32 error {f32_err:.1e}", s.name);
                                }
                            }
                            RowVerdict::Unwritten => unwritten += 1,
                            RowVerdict::Defect { f32_err, h_err } => {
                                if valid {
                                    println!("    INVALIDATED: {} row {r}: {got_h:#06x} against {want_h:#06x}, f32 error {f32_err:.1e}, binary16 error {h_err:.1e}", s.name);
                                }
                                valid = false;
                            }
                        }
                    }
                }
                if unwritten > 0 {
                    println!("    INVALIDATED: {unwritten} rows never written (gate 6)");
                    valid = false;
                }
                if arm.owes_identity() {
                    println!(
                        "    {same} rows identical BIT FOR BIT to tv_tetra48_h{}",
                        if moved > 0 { format!(", {moved} moved (worst f32 error {worst:.1e})") } else { String::new() }
                    );
                }
                if arm.prologue {
                    let norm = m.down_u16(&l2.dbg_norm)?;
                    let rot = m.down_f32(&l2.dbg_rot)?;
                    let kinds = prologue_kinds(&l2.schedule);
                    let mut bad = 0usize;
                    for (slot, &kind) in kinds.iter().enumerate() {
                        let got = &rot[slot * i..slot * i + if kind == 1 { h } else { i }];
                        let want: &[f32] = if kind == 1 { &want_rot_h } else { &want_rot_i };
                        bad += got.iter().zip(want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    }
                    let bad_norm = (0..n_norm).map(|s| norm[s * h..(s + 1) * h].iter().zip(&want_norm).filter(|(a, b)| a != b).count()).sum::<usize>();
                    println!("    prologues: {n_norm} norms with {bad_norm} values off candle's tree, {} rotations with {bad} values off rot_apply", kinds.len());
                    if bad + bad_norm > 0 {
                        println!("    INVALIDATED (gate 7)");
                        valid = false;
                    }
                }
            }
            l2arm.valid = valid;
            l2.arms.push(l2arm);
        }
        self.l2 = Some(l2);
        Ok(())
    }

    /// One round: every valid section arm once, timed around launch and sync.
    pub fn time_round(&mut self, timed: bool, tables: Tables, x: u64) -> Result<(), String> {
        if let Some(mut p) = self.persall.take() {
            if p.valid {
                let t0 = Instant::now();
                self.persall_launch(&mut p, tables, x)?;
                self.module.sync()?;
                if timed {
                    p.series.push(t0.elapsed().as_secs_f64());
                }
            }
            self.persall = Some(p);
        }
        if let Some(mut l2) = self.l2.take() {
            let t0 = Instant::now();
            l2_ref_pass(&l2, &self.module)?;
            self.module.sync()?;
            if timed {
                l2.ref_series.push(t0.elapsed().as_secs_f64());
            }
            let mut arms = std::mem::take(&mut l2.arms);
            for a in arms.iter_mut().filter(|a| a.valid) {
                let t0 = Instant::now();
                let ok = l2_launch(&mut l2, a, &self.module, false)?;
                let s = t0.elapsed().as_secs_f64();
                if !ok {
                    println!("  {}: ABORTED during the rounds: its series is void", pers::PERS_ARMS[a.idx].name);
                    a.valid = false;
                    a.series.clear();
                } else if timed {
                    a.series.push(s);
                }
            }
            l2.arms = arms;
            self.l2 = Some(l2);
        }
        Ok(())
    }

    /// The report, after the rounds. `t48` and `tseg` are A4's `tetra48`
    /// (252 launches) and `tetra48seg` (144) series from the same rounds.
    pub fn report(&self, t48: &[f64], tseg: &[f64]) {
        let ms = |v: &[f64]| {
            let (m, lo, hi) = pers::med_range(v);
            format!("{:.3} ms [{:.3}-{:.3}]", 1e3 * m, 1e3 * lo, 1e3 * hi)
        };
        let ratio = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x / y).collect() };
        if let Some(p) = &self.persall {
            println!("\n  One-launch Tetra pass (L1), rounds interleaved with A4:");
            if !p.valid || p.series.is_empty() {
                println!("  tpersall: not timed");
            } else {
                println!("  tetra48 (252 launches)   {}", ms(t48));
                println!("  tetra48seg (144)         {}", ms(tseg));
                println!("  tpersall (1)             {}", ms(&p.series));
                let gain: Vec<f64> = tseg.iter().zip(&p.series).map(|(s, t)| 100.0 * (s - t) / s).collect();
                let (g, gl, gh) = pers::med_range(&gain);
                let bpa: Vec<f64> = p.series.iter().map(|t| pers::bare_file_bytes().2 as f64 / t / 1e9).collect();
                let (b, bl, bh) = pers::med_range(&bpa);
                println!("  one-launch gain over tetra48seg, round by round: {g:.1} % [{gl:.1}-{gh:.1}]");
                println!("  B_pa = 0.9806 GB / t: {b:.0} GB/s [{bl:.0}-{bh:.0}]");
            }
        }
        if let Some(l2) = &self.l2 {
            println!("\n  Persistent arms (L2), rounds interleaved with A4 and tetra48h:");
            println!("  tetra48h (252 launches, served unit)  {}", ms(&l2.ref_series));
            let series = |name: &str| l2.arms.iter().find(|a| pers::PERS_ARMS[a.idx].name == name && a.valid && !a.series.is_empty()).map(|a| a.series.clone());
            for a in &l2.arms {
                let arm = &pers::PERS_ARMS[a.idx];
                if !a.valid || a.series.is_empty() {
                    println!("  {:<13} not timed", arm.name);
                    continue;
                }
                let (r, rl, rh) = pers::med_range(&ratio(&a.series, &l2.ref_series));
                let bytes = if arm.mode == 1 { pers::bare_file_bytes().0 } else { pers::bare_file_bytes().1 };
                let bt: Vec<f64> = a.series.iter().map(|t| bytes as f64 / t / 1e9).collect();
                let (b, bl, bh) = pers::med_range(&bt);
                println!("  {:<13} {}  x{r:.3} [{rl:.3}-{rh:.3}] of tetra48h  {b:.0} GB/s [{bl:.0}-{bh:.0}]", arm.name, ms(&a.series));
            }
            let n = pers::n_grid_syncs(&l2.schedule) as f64;
            if let (Some(w), Some(b)) = (series("pchain_w"), series(pers::CHAIN_BASE)) {
                let h: Vec<f64> = w.iter().zip(&b).map(|(x, y)| (x - y) / n * 1e6).collect();
                let (m, lo, hi) = pers::med_range(&h);
                println!("  h_exp = (pchain_w - psmem_t28) / {n}: {m:.2} us [{lo:.2}-{hi:.2}]");
                if let Some(w2) = series("pchain_w2") {
                    let hm: Vec<f64> = w2.iter().zip(&w).map(|(x, y)| (x - y) / n * 1e6).collect();
                    let (m, lo, hi) = pers::med_range(&hm);
                    println!("  h_marg = (pchain_w2 - pchain_w) / {n}: {m:.2} us [{lo:.2}-{hi:.2}]");
                }
                if let Some(wp) = series("pchain_wp") {
                    let pc: Vec<f64> = wp.iter().zip(&w).map(|(x, y)| (x - y) * 1e3).collect();
                    let (m, lo, hi) = pers::med_range(&pc);
                    println!("  prologue cost = pchain_wp - pchain_w: {m:.3} ms [{lo:.3}-{hi:.3}]");
                    if !t48.is_empty() && t48.len() == wp.len() && tseg.len() == wp.len() {
                        let q: Vec<f64> = (0..wp.len()).map(|k| (wp[k] / l2.ref_series[k]) / (tseg[k] / t48[k])).collect();
                        let (m, lo, hi) = pers::med_range(&q);
                        println!("  Q = [pchain_wp / tetra48h] / [tetra48seg / tetra48], round by round: {m:.3} [{lo:.3}-{hi:.3}]");
                    }
                }
            }
        }
    }
}

/// The prologue kinds in launch order, by the restage rule of the kernel.
fn prologue_kinds(s: &[pers::Site]) -> Vec<u32> {
    let mut v = Vec::new();
    for (k, site) in s.iter().enumerate() {
        let restage = k == 0 || s[k - 1].d_in != site.d_in || s[k - 1].layer != site.layer;
        match site.prologue {
            pers::Prologue::NormRot if restage => v.push(1),
            pers::Prologue::Rot if restage => v.push(2),
            _ => {}
        }
    }
    v
}

/// One pass of the served kernel over the 252 sites, into `y_ref`.
fn l2_ref_pass(l2: &L2, m: &Cuda) -> Result<(), String> {
    let t = l2.tables;
    for (k, s) in l2.sites.iter().enumerate() {
        let (nb, tw) = (s.d_in / 24, s.d_in % 24);
        let cfg = LaunchConfig { grid_dim: (s.d_out / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: l2.tile_bytes };
        let y = ptr(&l2.y_ref, &l2.served) + 2 * l2.row0[k] as u64;
        let tail = ptr(&l2.tails[k], m);
        let mut b = m.stream().launch_builder(&l2.f_ref);
        b.arg(&s.words).arg(&s.stride_u32).arg(&t.rows).arg(&t.pref).arg(&t.br).arg(&t.suf).arg(&s.gscale).arg(&t.invnorm)
            .arg(&s.rscale).arg(&tail).arg(&l2.x).arg(&y).arg(&nb).arg(&tw);
        // SAFETY: tv_tetra48_h's own arguments; every pointer names a live
        // allocation sized for this site.
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_tetra48_h: {e}"))?;
    }
    Ok(())
}

/// One cooperative launch of an L2 arm, ledger kept. Returns false when the
/// launch aborted on the watchdog (counters zeroed, ledger reset).
fn l2_launch(l2: &mut L2, a: &L2Arm, m: &Cuda, identity: bool) -> Result<bool, String> {
    let arm = &pers::PERS_ARMS[a.idx];
    let lc = l2.ledger.before(a.idx, l2.schedule.len())?;
    let launch = PersLaunch {
        bar_base: lc.bar_base,
        epoch: lc.epoch,
        cnt_offset: lc.cnt_offset,
        pad: 0,
        dbg: if identity { ptr(&l2.dbg, m) } else { 0 },
        dbg_norm: if identity && arm.prologue { ptr(&l2.dbg_norm, m) } else { 0 },
        dbg_rot: if identity && arm.prologue { ptr(&l2.dbg_rot, m) } else { 0 },
    };
    let cfg = LaunchConfig { grid_dim: (l2.grid, 1, 1), block_dim: (arm.threads, 1, 1), shared_mem_bytes: a.shared };
    let mut b = m.stream().launch_builder(&a.f);
    b.arg(&l2.args).arg(&launch);
    // SAFETY: the arm's own signature (PersArgs*, PersLaunch); the grid is one
    // CTA a SM, which passed the occupancy check, and a cooperative launch
    // refuses a grid that cannot be co-resident.
    unsafe { b.launch_cooperative(cfg) }.map_err(|e| format!("{}: {e}", arm.symbol))?;
    m.sync()?;
    l2.ledger.after(a.idx, pers::n_bar(&l2.schedule, arm.waits), l2.grid, arm.waits > 0);
    if arm.waits > 0 && m.down_u32(&l2.abort)?[0] != 0 {
        l2.ledger.aborted();
        m.stream().memset_zeros(&mut l2.bar).map_err(|e| e.to_string())?;
        m.stream().memset_zeros(&mut l2.cnts).map_err(|e| e.to_string())?;
        m.stream().memset_zeros(&mut l2.abort).map_err(|e| e.to_string())?;
        l2.ledger.reset();
        return Ok(false);
    }
    Ok(true)
}

/// The device buffers whose raw pointers live in the uploaded `PersArgs`
/// image: kept alive for as long as the arms can launch.
#[derive(Default)]
#[allow(dead_code)] // held, never read: dropping them would free what PersArgs points to
struct Keep {
    u32s: Vec<CudaSlice<u32>>,
    u64s: Vec<CudaSlice<u64>>,
    u16s: Vec<CudaSlice<u16>>,
    f32s: Vec<CudaSlice<f32>>,
}
