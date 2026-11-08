//! The persistent sections of `planesbench` (Linux, CUDA): step L1's
//! one-launch Tetra pass `tpersall` and its variants, the row-loop
//! microbenchmark of plan-300 (K4's Tetra rows, K11's int4 rows, K3b's
//! counters), and the driver cubins. Preregs:
//! `proofs/preregistration-pers-l1-*`, `-step1-micro-*`, `-rows-grid-*`.
//! Portable half and its tests: [`crate::pers`].
//!
//! L2's arms and L1's barrier, ring and launch microbenchmarks had their
//! verdicts, and their host side is gone; their kernels stay in
//! `tetra48_pers.cu`, which sits in the pinned 4E unit
//! ([`crate::pers::PERS_UNIT_PARTS`]).
//!
//! Everything here is off unless a variable asks for it, and nothing here
//! touches the bench unit or the table: the persistent kernels live in their
//! own NVRTC module on the bench's stream, and the served reference in a
//! third. Every arm is checked before it is timed; an arm that fails its
//! check is reported and never timed, and the job goes on (A3's rule).

use crate::gpu::{Cuda, KernelSource};
use crate::pers;
use cudarc::driver::{sys, CudaFunction, CudaSlice, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use std::path::PathBuf;
use std::time::Instant;

// SAFETY: route P's per-call struct (`kernels/pers_segment.cu`) is
// `#[repr(C)]`, plain integers, and its layout is the kernel's
// (`tests/pers_matches_served.rs::the_struct_mirrors_match_the_kernel_layouts`).
unsafe impl DeviceRepr for pers::SegLaunch {}
// SAFETY: the same for 4E's per-launch struct (`kernels/pers_4e.cu`).
unsafe impl DeviceRepr for pers::P4eLaunch {}
// SAFETY: the same for the verify pass's per-launch struct (`kernels/pers_4ev.cu`).
unsafe impl DeviceRepr for pers::P4vLaunch {}

/// What the environment asks of the persistent sections.
#[derive(Clone, Debug, Default)]
pub struct Flags {
    /// `LLVQ_TETRA_OCC=1`: the one-launch Tetra pass of step L1.
    pub tetra_occ: bool,
    /// `LLVQ_DUMP_CUBIN=<dir>`: the driver-JIT cubin of every module loaded.
    pub dump_cubin: Option<PathBuf>,
    /// `LLVQ_TROWS=1`: 4E's Tetra row loops alone (`pers_trows.cu`).
    pub trows: bool,
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
    /// Read and check the three variables. The Tetra sections are read
    /// against the fused geometry (the A3 grammar), so they need
    /// `LLVQ_SEG_TETRA=1` and the tetra48 arm.
    pub fn from_env(seg_tetra: bool, tetra48: bool) -> Result<Self, String> {
        let f = Flags {
            tetra_occ: flag01("LLVQ_TETRA_OCC")?,
            dump_cubin: std::env::var("LLVQ_DUMP_CUBIN").ok().map(PathBuf::from),
            trows: flag01("LLVQ_TROWS")?,
        };
        if (f.tetra_occ || f.trows) && !(seg_tetra && tetra48) {
            return Err("LLVQ_TETRA_OCC and LLVQ_TROWS are read against the fused Tetra geometry: \
                        they need LLVQ_SEG_TETRA=1 and tetra48 in LLVQ_BENCH_ARMS"
                .into());
        }
        Ok(f)
    }

    pub fn any(&self) -> bool {
        self.tetra_occ || self.trows
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

/// One matrix of the bare file for the row-loop microbenchmark, in served
/// order.
pub struct RowSite {
    pub name: String,
    pub words: u64,
    pub gscale: u64,
    pub rscale: u64,
    pub tail_f16: Vec<u16>,
    pub d_out: u32,
    pub d_in: u32,
    pub stride_u32: u32,
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
    name: &'static str,
    sym: &'static str,
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

/// The persistent sections of one bench run.
pub struct Section<'a> {
    flags: Flags,
    module: Cuda,
    sm: u32,
    persall: Vec<Persall<'a>>,
    trows: Option<Trows>,
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
        Ok(Section { flags, module, sm, persall: Vec::new(), trows: None })
    }

    /// Step L1: build `tv_tetra48_persall` and its variants
    /// ([`pers::PERSALL_VARIANTS`]) over the fused sites, then check every
    /// row of each against the A4 kernel's row on the same site. A variant
    /// that spills or moves a row is reported and never timed.
    pub fn persall_setup(&mut self, sites: Vec<PersallSite<'a>>, tables: Tables, x: u64, tile_bytes: u32) -> Result<(), String> {
        if !self.flags.tetra_occ {
            return Ok(());
        }
        let d_outs: Vec<u32> = sites.iter().map(|s| s.d_out).collect();
        let (g0, total_groups) = pers::persall_groups(&d_outs);
        let rows: usize = d_outs.iter().map(|&d| d as usize).sum();
        for &(name, sym, ts) in pers::PERSALL_VARIANTS.iter() {
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
            let shared = tile_bytes + if ts { pers::TABLE_BYTES } else { 0 };
            let f = self.module.func_dynamic_shared(sym, shared)?;
            let rep = self.module.report(sym)?;
            let occ = f
                .occupancy_max_active_blocks_per_multiprocessor(256, shared as usize, None)
                .map_err(|e| format!("occupancy: {e}"))?;
            let grid = total_groups.min(occ * self.sm).max(1);
            println!(
                "\n  One-launch Tetra pass (L1, {name}): {} sites, {rows} rows, {total_groups} groups of 8, grid {grid} \
                 ({occ} a SM x {}), {} registers, {} B local, {shared} B shared",
                sites.len(),
                self.sm,
                rep.num_regs,
                rep.local_bytes
            );
            let mut p = Persall {
                name,
                sym,
                f,
                table,
                y,
                grid,
                shared,
                total_groups,
                nsites: sites.len() as u32,
                valid: rep.local_bytes == 0 && occ >= 1,
                series: Vec::new(),
                _sites: std::marker::PhantomData,
            };
            if !p.valid {
                println!("  {name}: SPILLS {} B or does not fit ({occ} a SM), not timed (gate 8)", rep.local_bytes);
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
                                println!("  {name}: first moved row: {} row {r}: {got:e} against {want:e}, f64 error {err:.1e}", s.name);
                            }
                        } else {
                            p.valid = false;
                            println!("  {name} INVALIDATED: {} row {r}: {got:e} against {want:e}, f64 error {err:.1e}", s.name);
                            break;
                        }
                    }
                    at += s.d_out as usize;
                    if !p.valid {
                        break;
                    }
                }
                // A variant owes the bits: a moved association is not timed
                // either (the base kernel keeps L1's reading, which allowed it).
                if moved > 0 && name != "tpersall" {
                    p.valid = false;
                }
                println!(
                    "  {name}: {same} rows identical BIT FOR BIT to the A4 kernels{}{}",
                    if moved > 0 { format!(", {moved} with a moved association (worst {worst:.1e})") } else { String::new() },
                    if p.valid { "" } else { "; INVALIDATED, not timed" }
                );
            }
            self.persall.push(p);
        }
        Ok(())
    }

    fn persall_launch(&self, p: &mut Persall<'a>, t: Tables, x: u64) -> Result<(), String> {
        let cfg = LaunchConfig { grid_dim: (p.grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: p.shared };
        let mut b = self.module.stream().launch_builder(&p.f);
        b.arg(&p.table).arg(&p.nsites).arg(&t.rows).arg(&t.pref).arg(&t.br).arg(&t.suf).arg(&t.invnorm).arg(&x).arg(&p.total_groups);
        // SAFETY: the arguments are the kernel's; every pointer in the site
        // table points into a live allocation of the bench.
        unsafe { b.launch(cfg) }.map_err(|e| format!("{}: {e}", p.sym))?;
        Ok(())
    }

    /// The row-loop microbenchmark (`LLVQ_TROWS=1`, `pers_trows.cu`): 4E's
    /// Tetra row loops alone over the 252 matrices of the bare file (`sites`,
    /// served order), at 4E's geometry. The served `tv_tetra48_h` gives the
    /// reference rows; an arm that owes the bits and moves one row is reported
    /// and never timed. Each valid arm then runs one stamped launch, outside
    /// the rounds: cycles a step, the warps' balance, the SM clock.
    pub fn trows_setup(&mut self, sites: &[RowSite], tables: Tables, x: u64, x_len: usize, tile_define: &str) -> Result<(), String> {
        if !self.flags.trows {
            return Ok(());
        }
        if x_len < 9728 {
            return Err(format!("the bench activation has {x_len} values, the row loops read 9,728"));
        }
        let text = pers::trows_unit_text(tile_define);
        let src = KernelSource::new(&[text.as_str()]);
        println!("\n  Row-loop microbenchmark (LLVQ_TROWS): module {} bytes, sha256 {}", src.text.len(), src.sha256);
        let m = Cuda::on_stream(self.module.stream().clone(), &src)?;
        let served_text = pers::served_unit_text(&format!("{tile_define}{}", crate::tile::Prefill::resolve()?.defines()));
        let ssrc = KernelSource::new(&[served_text.as_str()]);
        println!("  reference: the served unit, {} bytes, sha256 {}", ssrc.text.len(), ssrc.sha256);
        let served = Cuda::on_stream(m.stream().clone(), &ssrc)?;
        if let Some(dir) = &self.flags.dump_cubin {
            dump_cubin(dir, "trows", &src.text)?;
        }
        let tile_blocks: u32 = tile_define
            .trim()
            .trim_start_matches("#define TILE_BLOCKS ")
            .trim_end_matches('u')
            .parse()
            .map_err(|e| format!("tile define {tile_define:?}: {e}"))?;
        let f_ref = served.func("tv_tetra48_h")?;
        let tails = sites.iter().map(|s| m.up_u16(&s.tail_f16)).collect::<Result<Vec<_>, _>>()?;
        // K2's second activation: 9,728 values of a seeded gaussian, its own
        // reference rows below.
        let mut rng = llvq_core::SplitMix64::new(0x7e72);
        let x2_host: Vec<f32> = (0..9728).map(|_| rng.next_gaussian() as f32).collect();
        let x2_buf = m.up_f32(&x2_host)?;
        let x2 = ptr(&x2_buf, &m);
        // The reference rows, site by site, one tv_tetra48_h launch each, on
        // the bench activation and on the second one.
        let mut refs: Vec<Vec<u16>> = Vec::with_capacity(sites.len());
        let mut refs2: Vec<Vec<u16>> = Vec::with_capacity(sites.len());
        for (k, s) in sites.iter().enumerate() {
          for (xa, out) in [(x, &mut refs), (x2, &mut refs2)] {
            let (nb, tw) = (s.d_in / 24, s.d_in % 24);
            let y = served.zeros_u16(s.d_out as usize)?;
            let cfg = LaunchConfig { grid_dim: (s.d_out / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: tile_blocks * 96 };
            let tail = ptr(&tails[k], &m);
            let mut b = m.stream().launch_builder(&f_ref);
            b.arg(&s.words).arg(&s.stride_u32).arg(&tables.rows).arg(&tables.pref).arg(&tables.br).arg(&tables.suf)
                .arg(&s.gscale).arg(&tables.invnorm).arg(&s.rscale).arg(&tail).arg(&xa).arg(&y).arg(&nb).arg(&tw);
            // SAFETY: tv_tetra48_h's own arguments on live allocations of this site.
            unsafe { b.launch(cfg) }.map_err(|e| format!("tv_tetra48_h: {e}"))?;
            m.sync()?;
            out.push(served.down_u16(&y)?);
          }
        }
        let mut arms = Vec::new();
        let relayout = m.func("tv_trows_tile_relayout")?;
        let mut tiled: Vec<CudaSlice<u32>> = Vec::new();
        for &(name, sym, owed, minb) in pers::TROWS_ARMS.iter() {
            let pick: Vec<usize> = (0..sites.len()).filter(|&k| !pers::trows_small_only(name) || sites[k].d_in <= 4096).collect();
            let rows: usize = pick.iter().map(|&k| sites[k].d_out as usize).sum();
            let k2 = name == "k2";
            let (tile, cpa) = (name == "tile", name == "cpa");
            let y = m.zeros_u16(rows)?;
            let y2 = m.zeros_u16(if k2 { rows } else { 1 })?;
            let mut words = Vec::with_capacity(pick.len() * 10);
            let mut at = 0u64;
            let mut bytes = 0u64;
            let (mut served_words, mut tile_words) = (0u64, 0u64);
            for &k in &pick {
                let s = &sites[k];
                let (nb, tw) = (s.d_in / 24, s.d_in % 24);
                let y2p = if k2 { ptr(&y2, &m) + 2 * at } else { 0 };
                // TILE reads its own copy of the words, relaid once here by a
                // kernel that only moves bits (tv_trows_tile_relayout).
                let (wp, stride) = if tile {
                    let ts = pers::trows_tile_stride(nb);
                    let buf = m.zeros_u32(s.d_out as usize * ts as usize)?;
                    let cfg = LaunchConfig { grid_dim: (1024, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    let mut b = m.stream().launch_builder(&relayout);
                    b.arg(&s.words).arg(&s.stride_u32).arg(&buf).arg(&ts).arg(&s.d_out).arg(&nb);
                    // SAFETY: the relayout's own arguments: the site's live word
                    // stream and a zeroed buffer of d_out x ts words.
                    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_trows_tile_relayout: {e}"))?;
                    let wp = ptr(&buf, &m);
                    tiled.push(buf);
                    served_words += u64::from(s.d_out) * 4 * u64::from(s.stride_u32);
                    tile_words += u64::from(s.d_out) * 4 * u64::from(ts);
                    (wp, ts)
                } else {
                    (s.words, s.stride_u32)
                };
                words.extend(pers::trows_site_words(
                    wp, s.gscale, s.rscale, ptr(&tails[k], &m), ptr(&y, &m) + 2 * at, y2p, s.d_out, nb, tw, stride,
                ));
                at += u64::from(s.d_out);
                bytes += u64::from(s.d_out) * (4 * u64::from(stride) + 2 * u64::from(tw) + 4);
            }
            if tile {
                m.sync()?;
                println!(
                    "  tile: words relaid on card, {:.4} GB against the served stream's {:.4} GB ({:+} B), a second copy for the bench only",
                    tile_words as f64 / 1e9,
                    served_words as f64 / 1e9,
                    tile_words as i64 - served_words as i64
                );
            }
            let table = m.up_u64(&words)?;
            // One CTA a SM: 4E's own shared bytes. Two: the small sites' staging.
            let shared = match (k2, cpa, minb) {
                (true, _, _) => pers::trows_k2_smem_bytes(),
                (false, true, _) => pers::trows_cpa_smem_bytes(16),
                (false, false, 1) => pers::trows_smem_bytes(9728),
                _ => pers::trows_smem_bytes(4096),
            };
            let f = m.func_dynamic_shared(sym, shared)?;
            let rep = m.report(sym)?;
            let occ = f
                .occupancy_max_active_blocks_per_multiprocessor(512, shared as usize, None)
                .map_err(|e| format!("occupancy: {e}"))?;
            let grid = self.sm * minb;
            println!(
                "  {name:<7} {sym}: {} sites, {rows} rows, grid {grid} ({minb} a SM, {occ} fit), 512 threads, {} registers, {} B local, {shared} B shared",
                pick.len(),
                rep.num_regs,
                rep.local_bytes
            );
            let mut a = TrowsArm {
                name,
                f,
                table,
                y,
                y2,
                nsites: pick.len() as u32,
                shapes: pick.iter().map(|&k| (sites[k].d_out, sites[k].d_in)).collect(),
                grid,
                shared,
                bytes,
                valid: rep.local_bytes == 0 && occ >= minb,
                series: Vec::new(),
                ctrs: if pers::trows_counters(name) { Some(m.zeros_u32(2 + pick.len())?) } else { None },
            };
            if !a.valid {
                println!("  {name}: SPILLS {} B or does not fit ({occ} a SM, {minb} needed), not timed", rep.local_bytes);
                arms.push(a);
                continue;
            }
            if cpa && !pick.iter().all(|&k| pers::trows_cpa_ok(sites[k].stride_u32)) {
                println!("  cpa: REFUSED, a row stride is not a multiple of 16 B; not timed");
                a.valid = false;
                arms.push(a);
                continue;
            }
            trows_launch(&m, &mut a, &tables, x, x2, None)?;
            m.sync()?;
            // Each output against its reference: y on the bench activation,
            // and for K2 also y2 on the second one.
            let mut checks = vec![(m.down_u16(&a.y)?, &refs, "")];
            if k2 {
                checks.push((m.down_u16(&a.y2)?, &refs2, " on the second activation"));
            }
            for (got, want_of, what) in &checks {
                let (mut same, mut at) = (0usize, 0usize);
                let mut first = None;
                for &k in &pick {
                    for (r, &want) in want_of[k].iter().enumerate() {
                        if got[at + r] == want {
                            same += 1;
                        } else if first.is_none() {
                            first = Some(format!("{} row {r}: {:#06x} against {want:#06x}", sites[k].name, got[at + r]));
                        }
                    }
                    at += sites[k].d_out as usize;
                }
                if owed {
                    if same == rows {
                        println!("  {name}: {rows} rows identical BIT FOR BIT to tv_tetra48_h{what}");
                    } else {
                        a.valid = false;
                        println!("  {name} INVALIDATED{what}: {same} of {rows} rows equal; first: {}", first.unwrap_or_default());
                    }
                }
                if !owed {
                    println!("  {name}: diagnostic, rows not owed ({same} of {rows} happen to equal tv_tetra48_h)");
                }
            }
            if a.valid {
                // The stamped launch, outside the rounds.
                let nw = 16u32;
                let stamps = m.zeros_u32(a.shapes.len() * grid as usize * (nw as usize + 1))?;
                let ends: CudaSlice<u64> = m.stream().alloc_zeros(4 * grid as usize).map_err(|e| format!("alloc u64: {e}"))?;
                trows_launch(&m, &mut a, &tables, x, x2, Some((&stamps, &ends)))?;
                m.sync()?;
                let st = m.down_u32(&stamps)?;
                let en: Vec<u64> = m.stream().clone_dtoh(&ends).map_err(|e| format!("D2H u64: {e}"))?;
                let mhz = pers::trows_clock_mhz(&en);
                for cl in pers::trows_classes(&st, &a.shapes, grid, nw) {
                    println!(
                        "  {name:<7} stamps d_in {:>5}: {:>3} sites, {:>7.0} cycles a step (32 blocks of a row), {:.3} us at {mhz:.0} MHz; warp balance {:.3}",
                        cl.d_in,
                        cl.sites,
                        cl.cycles_per_step(),
                        cl.cycles_per_step() / mhz,
                        cl.balance(nw)
                    );
                }
                println!(
                    "  {name:<7} CTA spread over {} sites (sum of the slowest CTA's busy cycles over the mean's): {:.3}",
                    a.shapes.len(),
                    pers::trows_cta_spread(&st, a.shapes.len(), grid, nw)
                );
            }
            arms.push(a);
        }
        let q4 = q4rows_setup(&m, &served, self.sm)?;
        ctr_micro(&m, self.sm)?;
        self.trows = Some(Trows { module: m, _served: served, _tails: tails, x2, _x2: x2_buf, arms, _tiled: tiled, q4 });
        Ok(())
    }

    /// One round: every valid section arm once, timed around launch and sync.
    pub fn time_round(&mut self, timed: bool, tables: Tables, x: u64) -> Result<(), String> {
        let mut all = std::mem::take(&mut self.persall);
        for p in all.iter_mut().filter(|p| p.valid) {
            let t0 = Instant::now();
            self.persall_launch(p, tables, x)?;
            self.module.sync()?;
            if timed {
                p.series.push(t0.elapsed().as_secs_f64());
            }
        }
        self.persall = all;
        if let Some(mut tr) = self.trows.take() {
            for a in tr.arms.iter_mut().filter(|a| a.valid) {
                let t0 = Instant::now();
                trows_launch(&tr.module, a, &tables, x, tr.x2, None)?;
                tr.module.sync()?;
                if timed {
                    a.series.push(t0.elapsed().as_secs_f64());
                }
            }
            q4rows_round(&tr.module, &mut tr.q4, timed)?;
            self.trows = Some(tr);
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
        if !self.persall.is_empty() {
            println!("\n  One-launch Tetra pass (L1) and its variants, rounds interleaved with A4:");
            println!("  tetra48 (252 launches)   {}", ms(t48));
            println!("  tetra48seg (144)         {}", ms(tseg));
            let base = self.persall.first().filter(|p| p.valid && !p.series.is_empty()).map(|p| p.series.clone());
            for p in &self.persall {
                if !p.valid || p.series.is_empty() {
                    println!("  {:<18} not timed", p.name);
                    continue;
                }
                let bpa: Vec<f64> = p.series.iter().map(|t| pers::bare_file_bytes().2 as f64 / t / 1e9).collect();
                let (b, bl, bh) = pers::med_range(&bpa);
                println!("  {:<18} {}  B_pa = 0.9806 GB / t: {b:.0} GB/s [{bl:.0}-{bh:.0}]", p.name, ms(&p.series));
                if p.name == "tpersall" {
                    let gain: Vec<f64> = tseg.iter().zip(&p.series).map(|(s, t)| 100.0 * (s - t) / s).collect();
                    let (g, gl, gh) = pers::med_range(&gain);
                    println!("  {:<18} one-launch gain over tetra48seg, round by round: {g:.1} % [{gl:.1}-{gh:.1}]", "");
                } else if let Some(bs) = &base {
                    let r: Vec<f64> = bs.iter().zip(&p.series).map(|(b, t)| b / t).collect();
                    let (m, lo, hi) = pers::med_range(&r);
                    println!("  {:<18} speed over tpersall, round by round: x{m:.3} [{lo:.3}-{hi:.3}]", "");
                }
            }
        }
        if let Some(tr) = &self.trows {
            trows_report(tr);
        }
    }
}

/// One arm of the row-loop microbenchmark.
struct TrowsArm {
    name: &'static str,
    f: CudaFunction,
    table: CudaSlice<u64>,
    y: CudaSlice<u16>,
    /// K2's rows on the second activation (one element for the other arms).
    y2: CudaSlice<u16>,
    nsites: u32,
    shapes: Vec<(u32, u32)>,
    grid: u32,
    shared: u32,
    bytes: u64,
    valid: bool,
    series: Vec<f64>,
    /// gb and gbc's counters (barrier, abort, a cursor a site), zeroed
    /// before each launch and passed as `x2`.
    ctrs: Option<CudaSlice<u32>>,
}

/// The row-loop section: its module, the served reference's module and the
/// binary16 tails its site tables point to (kept alive), and the arms.
struct Trows {
    module: Cuda,
    _served: Cuda,
    _tails: Vec<CudaSlice<u16>>,
    /// K2's second activation, its address and its buffer (kept alive).
    x2: u64,
    _x2: CudaSlice<f32>,
    arms: Vec<TrowsArm>,
    /// TILE's relaid words (kept alive).
    _tiled: Vec<CudaSlice<u32>>,
    /// K11's int4 rows.
    q4: Q4Rows,
}

fn trows_launch(m: &Cuda, a: &mut TrowsArm, t: &Tables, x: u64, x2: u64, stamps: Option<(&CudaSlice<u32>, &CudaSlice<u64>)>) -> Result<(), String> {
    let cfg = LaunchConfig { grid_dim: (a.grid, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: a.shared };
    let x2 = match a.ctrs.as_mut() {
        Some(c) => {
            m.stream().memset_zeros(c).map_err(|e| format!("{}: counters: {e}", a.name))?;
            ptr(c, m)
        }
        None => x2,
    };
    let (sp, ep) = match stamps {
        Some((s, e)) => (ptr(s, m), ptr(e, m)),
        None => (0u64, 0u64),
    };
    let mut b = m.stream().launch_builder(&a.f);
    b.arg(&a.table).arg(&a.nsites).arg(&t.rows).arg(&t.pref).arg(&t.br).arg(&t.suf).arg(&t.invnorm).arg(&x).arg(&x2).arg(&sp).arg(&ep);
    // SAFETY: the kernel's own arguments; every pointer in the site table and
    // the stamp buffers (sized sites x grid x 17 and grid x 4) is live.
    unsafe { b.launch(cfg) }.map_err(|e| format!("{}: {e}", a.name))?;
    Ok(())
}

fn trows_report(tr: &Trows) {
    println!("\n  Row-loop microbenchmark, rounds interleaved with A4; bytes = words (row stride) + binary16 tails + row scales:");
    let series = |n: &str| tr.arms.iter().find(|a| a.name == n && a.valid && !a.series.is_empty()).map(|a| a.series.clone());
    for a in &tr.arms {
        if !a.valid || a.series.is_empty() {
            println!("  {:<7} not timed", a.name);
            continue;
        }
        let (m, lo, hi) = pers::med_range(&a.series);
        let gb: Vec<f64> = a.series.iter().map(|t| a.bytes as f64 / t / 1e9).collect();
        let (b, bl, bh) = pers::med_range(&gb);
        let base = match a.name {
            "v2" | "v2_s" => None,
            "v2_o2" => Some("v2_s"),
            "l1w" | "bcast" | "xbcast" => Some("v2u"),
            "gbc" => Some("gb"),
            "k2" => None,
            _ => Some("v2"),
        };
        let vs = base.and_then(|n| series(n).map(|s| (n, s))).map(|(n, s)| {
            let r: Vec<f64> = s.iter().zip(&a.series).map(|(x, y)| x / y).collect();
            let (rm, rl, rh) = pers::med_range(&r);
            format!("  speed over {n}: x{rm:.3} [{rl:.3}-{rh:.3}]")
        });
        println!(
            "  {:<7} {:.3} ms [{:.3}-{:.3}]  {:.4} GB: {b:.0} GB/s [{bl:.0}-{bh:.0}]{}",
            a.name,
            1e3 * m,
            1e3 * lo,
            1e3 * hi,
            a.bytes as f64 / 1e9,
            vs.unwrap_or_default()
        );
        if let Some(v) = k4_verdict(a.name, base, &series, &a.series) {
            println!("  {:<7} {v}", "");
        }
        // The row lead's rule: gbc over gb, the median of the round-by-round speeds.
        if a.name == "gbc" {
            if let Some(gb) = series("gb") {
                let r: Vec<f64> = gb.iter().zip(&a.series).map(|(b, t)| b / t).collect();
                let (rm, _, _) = pers::med_range(&r);
                let verdict = if rm >= pers::ROWS_GRID_LEVER {
                    "LEVER: at or over x1.05, licenses v10 (4E with rows claimed across the grid)"
                } else if rm >= pers::ROWS_GRID_DEAD {
                    "reported, not a lever: between x1.02 and x1.05"
                } else {
                    "DEAD: under x1.02"
                };
                println!("  {:<7} row-lead rule: x{rm:.3} over gb, {verdict}", "");
            }
        }
        // K2: the cost of a second token as a fraction of a pass, round by
        // round against V2 (same rows, same bytes, one decode for two).
        if a.name == "k2" {
            if let Some(v2) = series("v2") {
                let f: Vec<f64> = a.series.iter().zip(&v2).map(|(k, v)| k / v - 1.0).collect();
                let (fm, fl, fh) = pers::med_range(&f);
                println!(
                    "  {:<7} time over v2, round by round, less 1: the second token costs {fm:.3} [{fl:.3}-{fh:.3}] of a pass",
                    ""
                );
            }
        }
    }
    q4rows_report(&tr.q4);
}

/// K4's rule (plan-300 step 1) for an arm of the five: the median of its
/// round-by-round speed over v2 against the bar.
fn k4_verdict(name: &str, base: Option<&str>, series: &dyn Fn(&str) -> Option<Vec<f64>>, s: &[f64]) -> Option<String> {
    if !["nc", "nc256", "cpa", "tile", "bal"].contains(&name) || base != Some("v2") {
        return None;
    }
    let v2 = series("v2")?;
    let r: Vec<f64> = v2.iter().zip(s).map(|(b, t)| b / t).collect();
    let (m, _, _) = pers::med_range(&r);
    let verdict = if m >= pers::K4_LEVER {
        "LEVER: at or over x1.15, licenses the 4E port (step 4)"
    } else if m >= pers::K4_DEAD {
        "reported, not a lever: between x1.10 and x1.15"
    } else {
        "DEAD: under x1.10"
    };
    Some(format!("K4 rule: x{m:.3} over v2, {verdict}"))
}

// ---- plan-300 step 1: K11, the int4 rows alone; K3b, the counters ---------

/// One int4 record of K11's synthetic set: its buffers and shape.
struct Q4Rec {
    wq: CudaSlice<u32>,
    sc: CudaSlice<u16>,
    bi: CudaSlice<u16>,
    d_out: u32,
    d_in: u32,
}

/// One arm of K11 (`pers_q4rows.cu`).
struct Q4Arm {
    name: &'static str,
    f: CudaFunction,
    table: CudaSlice<u64>,
    _y: CudaSlice<u16>,
    grid: u32,
    shared: u32,
    valid: bool,
    series: Vec<f64>,
}

/// K11's section: the served `tv_q4_hh` over the 84 records (the reference of
/// rows and time), then the arms, timed in the same rounds.
struct Q4Rows {
    f_ref: CudaFunction,
    recs: Vec<Q4Rec>,
    xh: CudaSlice<u16>,
    y_ref: CudaSlice<u16>,
    _tiled: Vec<CudaSlice<u32>>,
    ref_series: Vec<f64>,
    arms: Vec<Q4Arm>,
    bytes: u64,
}

/// binary16 bits with a random sign and mantissa and an exponent field in
/// [lo, hi]: the host fixture's `Rng::h16`, on the bench's generator.
fn h16(rng: &mut llvq_core::SplitMix64, lo: u64, hi: u64) -> u16 {
    let r = rng.next();
    let e = lo + (r >> 20) % (hi - lo + 1);
    (((r >> 40) & 1) << 15 | e << 10 | (r & 0x3ff)) as u16
}

/// The served pass: `tv_q4_hh` on each record, as the eager path launches it
/// (one warp a row, 256 threads, the activation staged in f32).
fn q4hh_pass(m: &Cuda, q: &Q4Rows) -> Result<(), String> {
    let mut at = 0u64;
    for r in &q.recs {
        let cfg = LaunchConfig { grid_dim: (r.d_out / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 4 * r.d_in };
        let gpr = r.d_in / 128;
        let y = ptr(&q.y_ref, m) + 2 * at;
        let mut b = m.stream().launch_builder(&q.f_ref);
        b.arg(&r.wq).arg(&r.sc).arg(&r.bi).arg(&q.xh).arg(&y).arg(&r.d_in).arg(&gpr);
        // SAFETY: tv_q4_hh's own arguments on the record's live buffers; y
        // points at the record's rows inside y_ref.
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_q4_hh: {e}"))?;
        at += u64::from(r.d_out);
    }
    Ok(())
}

fn q4rows_launch(m: &Cuda, a: &Q4Arm, nsites: u32, xh: &CudaSlice<u16>) -> Result<(), String> {
    let cfg = LaunchConfig { grid_dim: (a.grid, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: a.shared };
    let (sp, ep) = (0u64, 0u64);
    let mut b = m.stream().launch_builder(&a.f);
    b.arg(&a.table).arg(&nsites).arg(xh).arg(&sp).arg(&ep);
    // SAFETY: the kernel's own arguments; every pointer of the site table is
    // a live buffer of the section; no stamps.
    unsafe { b.launch(cfg) }.map_err(|e| format!("{}: {e}", a.name))?;
    Ok(())
}

/// K11's setup: 84 synthetic records of the 4B's int4 shapes, seeded (the
/// values do not move the time; identity is held on these same values), the
/// served reference rows, then each arm checked row by row before it is timed.
fn q4rows_setup(m: &Cuda, served: &Cuda, sm: u32) -> Result<Q4Rows, String> {
    let sites = pers::q4rows_sites();
    let bytes = pers::q4rows_bytes(&sites);
    let mut rng = llvq_core::SplitMix64::new(0x7e75);
    let mut recs = Vec::with_capacity(sites.len());
    for &(d_out, d_in, _, _) in &sites {
        let gpr = (d_out * (d_in / 128)) as usize;
        let w: Vec<u32> = (0..(d_out as usize * d_in as usize / 8)).map(|_| rng.next() as u32).collect();
        let sc: Vec<u16> = (0..gpr).map(|_| h16(&mut rng, 6, 10)).collect();
        let bi: Vec<u16> = (0..gpr).map(|_| h16(&mut rng, 8, 12)).collect();
        recs.push(Q4Rec { wq: m.up_u32(&w)?, sc: m.up_u16(&sc)?, bi: m.up_u16(&bi)?, d_out, d_in });
    }
    let xh_host: Vec<u16> = (0..9728).map(|_| crate::f16_bits(rng.next_gaussian() as f32)).collect();
    let xh = m.up_u16(&xh_host)?;
    let rows: usize = sites.iter().map(|s| s.0 as usize).sum();
    let mut q = Q4Rows {
        f_ref: served.func("tv_q4_hh")?,
        recs,
        xh,
        y_ref: m.zeros_u16(rows)?,
        _tiled: Vec::new(),
        ref_series: Vec::new(),
        arms: Vec::new(),
        bytes,
    };
    println!(
        "\n  Int4 rows alone (K11): {} synthetic records of the 4B's int4 shapes, {rows} rows, {:.4} GB (words, scales, biases)",
        sites.len(),
        bytes as f64 / 1e9
    );
    q4hh_pass(m, &q)?;
    m.sync()?;
    let want = m.down_u16(&q.y_ref)?;
    let relayout = m.func("tv_q4rows_tile_relayout")?;
    let shared = pers::q4rows_smem_bytes(9728);
    for &(name, sym, owed) in pers::Q4ROWS_ARMS.iter() {
        let y = m.zeros_u16(rows)?;
        let mut words = Vec::with_capacity(sites.len() * 6);
        let mut at = 0u64;
        for r in &q.recs {
            let wp = if name == "q4t" {
                let buf = m.zeros_u32(r.d_out as usize * r.d_in as usize / 8)?;
                let nwords = r.d_in / 8;
                let cfg = LaunchConfig { grid_dim: (1024, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                let mut b = m.stream().launch_builder(&relayout);
                b.arg(&r.wq).arg(&buf).arg(&r.d_out).arg(&nwords);
                // SAFETY: the relayout's own arguments, both buffers d_out x nwords.
                unsafe { b.launch(cfg) }.map_err(|e| format!("tv_q4rows_tile_relayout: {e}"))?;
                let p = ptr(&buf, m);
                q._tiled.push(buf);
                p
            } else {
                ptr(&r.wq, m)
            };
            words.extend(pers::q4rows_site_words(wp, ptr(&r.sc, m), ptr(&r.bi, m), ptr(&y, m) + 2 * at, r.d_out, r.d_in));
            at += u64::from(r.d_out);
        }
        let f = m.func_dynamic_shared(sym, shared)?;
        let rep = m.report(sym)?;
        let occ = f
            .occupancy_max_active_blocks_per_multiprocessor(512, shared as usize, None)
            .map_err(|e| format!("occupancy: {e}"))?;
        println!(
            "  {name:<7} {sym}: grid {sm} (1 a SM, {occ} fit), 512 threads, {} registers, {} B local, {shared} B shared",
            rep.num_regs, rep.local_bytes
        );
        let mut a = Q4Arm { name, f, table: m.up_u64(&words)?, _y: y, grid: sm, shared, valid: rep.local_bytes == 0 && occ >= 1, series: Vec::new() };
        if !a.valid {
            println!("  {name}: SPILLS {} B or does not fit, not timed", rep.local_bytes);
            q.arms.push(a);
            continue;
        }
        q4rows_launch(m, &a, sites.len() as u32, &q.xh)?;
        m.sync()?;
        let got = m.down_u16(&a._y)?;
        let same = got.iter().zip(&want).filter(|(g, w)| g == w).count();
        if !owed {
            println!("  {name}: diagnostic, rows not owed ({same} of {rows} happen to equal tv_q4_hh)");
        } else if same == rows {
            println!("  {name}: {rows} rows identical BIT FOR BIT to tv_q4_hh");
        } else {
            let k = got.iter().zip(&want).position(|(g, w)| g != w).unwrap_or(0);
            println!("  {name} INVALIDATED: {same} of {rows} rows equal; first: row {k}: {:#06x} against {:#06x}", got[k], want[k]);
            a.valid = false;
        }
        q.arms.push(a);
    }
    Ok(q)
}

/// One round of K11: the served pass, then every valid arm.
fn q4rows_round(m: &Cuda, q: &mut Q4Rows, timed: bool) -> Result<(), String> {
    let t0 = Instant::now();
    q4hh_pass(m, q)?;
    m.sync()?;
    if timed {
        q.ref_series.push(t0.elapsed().as_secs_f64());
    }
    let n = q.recs.len() as u32;
    for a in q.arms.iter_mut().filter(|a| a.valid) {
        let t0 = Instant::now();
        q4rows_launch(m, a, n, &q.xh)?;
        m.sync()?;
        if timed {
            a.series.push(t0.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

fn q4rows_report(q: &Q4Rows) {
    println!("\n  Int4 rows alone (K11), rounds interleaved with A4; bytes = {:.4} GB of words, scales and biases:", q.bytes as f64 / 1e9);
    let line = |name: &str, s: &[f64]| {
        let (m, lo, hi) = pers::med_range(s);
        let gb: Vec<f64> = s.iter().map(|t| q.bytes as f64 / t / 1e9).collect();
        let (b, bl, bh) = pers::med_range(&gb);
        format!("  {name:<7} {:.3} ms [{:.3}-{:.3}]  {b:.0} GB/s [{bl:.0}-{bh:.0}]", 1e3 * m, 1e3 * lo, 1e3 * hi)
    };
    if q.ref_series.is_empty() {
        println!("  q4hh    not timed");
    } else {
        println!("{} (tv_q4_hh, 84 launches)", line("q4hh", &q.ref_series));
    }
    let q4 = q.arms.iter().find(|a| a.name == "q4" && a.valid && !a.series.is_empty()).map(|a| a.series.clone());
    for a in &q.arms {
        if !a.valid || a.series.is_empty() {
            println!("  {:<7} not timed", a.name);
            continue;
        }
        let mut s = line(a.name, &a.series);
        let speed = |base: &[f64]| {
            let r: Vec<f64> = base.iter().zip(&a.series).map(|(b, t)| b / t).collect();
            pers::med_range(&r)
        };
        if !q.ref_series.is_empty() {
            let (m, lo, hi) = speed(&q.ref_series);
            s.push_str(&format!("  speed over q4hh: x{m:.3} [{lo:.3}-{hi:.3}]"));
        }
        if let (Some(b), true) = (&q4, a.name != "q4") {
            let (m, lo, hi) = speed(b);
            s.push_str(&format!("  speed over q4: x{m:.3} [{lo:.3}-{hi:.3}]"));
            if a.name == "q4nc" || a.name == "q4t" {
                let v = if m >= pers::K11_LEVER { "LEVER: at or over x1.10" } else { "not a lever: under x1.10" };
                s.push_str(&format!("\n  {:<7} K11 rule: x{m:.3} over q4, {v}", ""));
            }
        }
        println!("{s}");
    }
}

/// K3b: the counter kernels (`pers_ctr.cu`), one cooperative launch of N hops
/// each, at 4E's geometry, the arms interleaved rep by rep so every
/// difference is formed within one rep. Timed on the host around launch and
/// sync, as the L1 microbenchmarks. After the reps, every counter must hold
/// what the hops put there (`pers::ctr_expected`), or the arm is void.
fn ctr_micro(m: &Cuda, sm: u32) -> Result<(), String> {
    const N: u32 = 10_000;
    const REPS: usize = 7;
    const DISCARD: usize = 2;
    // About 1 us at the 2,456-2,502 MHz of pers-rowloop's stamps.
    const WORK: u32 = 2_500;
    let hdr = pers::SMEM_HEADER;
    let grid = sm;
    println!(
        "\n  Counters (K3b): {grid} x 512 cooperative, {N} hops a launch, {REPS} reps interleaved, {DISCARD} discarded; \
         split and work spin {WORK} cycles a hop"
    );
    let mut fs = Vec::new();
    for &(name, sym) in pers::CTR_ARMS.iter() {
        let f = m.func(sym)?;
        let occ = f
            .occupancy_max_active_blocks_per_multiprocessor(512, hdr as usize, None)
            .map_err(|e| format!("occupancy: {e}"))?;
        if occ < 1 {
            return Err(format!("{sym}: no CTA of 512 fits a SM"));
        }
        let lines = pers::ctr_groups(grid) as usize * 32;
        fs.push((name, f, m.up_u32(&vec![0u32; lines])?, Vec::<f64>::new(), true));
    }
    let mut abort = m.up_u32(&[0])?;
    for rep in 0..REPS {
        for (name, f, ctr, series, ok) in fs.iter_mut() {
            if !*ok {
                continue;
            }
            let base = rep as u32 * N;
            let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: hdr };
            let t0 = Instant::now();
            let mut b = m.stream().launch_builder(f);
            b.arg(&mut *ctr).arg(&base).arg(&N).arg(&WORK).arg(&mut abort);
            // SAFETY: the kernel's own arguments; the grid passed the
            // occupancy check, and a cooperative launch refuses a grid that
            // cannot be co-resident.
            unsafe { b.launch_cooperative(cfg) }.map_err(|e| format!("tv_ctr {name}: {e}"))?;
            m.sync()?;
            let us = t0.elapsed().as_secs_f64() / f64::from(N) * 1e6;
            if m.down_u32(&abort)?[0] != 0 {
                println!("  {name}: ABORTED on the watchdog, void");
                *ok = false;
                abort = m.up_u32(&[0])?;
                continue;
            }
            if rep >= DISCARD {
                series.push(us);
            }
        }
    }
    for (name, _, ctr, _, ok) in fs.iter_mut() {
        if !*ok {
            continue;
        }
        let got = m.down_u32(ctr)?;
        let want: Vec<u32> = pers::ctr_expected(name, grid, N).iter().map(|v| v * REPS as u32).collect();
        let heads: Vec<u32> = (0..want.len()).map(|p| got[32 * p]).collect();
        if heads == want {
            println!("  {name}: counts OK ({} counter{})", want.len(), if want.len() > 1 { "s" } else { "" });
        } else {
            println!("  {name} INVALIDATED: counters {heads:?}, expected {want:?}");
            *ok = false;
        }
    }
    let get = |n: &str| fs.iter().find(|f| f.0 == n && f.4).map(|f| f.3.clone());
    for (name, _, _, series, ok) in &fs {
        if !*ok {
            println!("  {name:<6} void");
            continue;
        }
        let (md, lo, hi) = pers::med_range(series);
        let mut s = format!("  {name:<6} {md:7.3} us a hop [{lo:.3}-{hi:.3}]");
        if ["tight", "rel", "ctr8"].contains(name) {
            if let Some(bar) = get("bar") {
                let g: Vec<f64> = bar.iter().zip(series).map(|(b, x)| b - x).collect();
                let (gm, gl, gh) = pers::med_range(&g);
                let v = if gm >= pers::K3B_GAIN_US { "LEVER: under bar by 0.3 us or more" } else { "not a lever" };
                s.push_str(&format!("  under bar, rep by rep: {gm:.3} us [{gl:.3}-{gh:.3}]; K3b rule: {v}"));
            }
        }
        if *name == "split" {
            if let Some(w) = get("work") {
                let g: Vec<f64> = series.iter().zip(&w).map(|(x, y)| x - y).collect();
                let (gm, gl, gh) = pers::med_range(&g);
                s.push_str(&format!("  exposed over work, rep by rep: {gm:.3} us [{gl:.3}-{gh:.3}]"));
            }
        }
        println!("{s}");
    }
    Ok(())
}
