//! Stage 2 of the decode-speed research phase's verifier (MODE RECHERCHE, R2,
//! `docs/mesures/recherche-debit-r1-cadrage-2026-11-12.txt`): one card run that
//! scores every registered row program on the sealed 4B's own records.
//!
//! ```text
//! rowbench <sealed.bin> [<bare-tetra.bin>] [--rounds N] [--dump DIR]
//! ```
//!
//! What one invocation does, in this order:
//!   1. reads the SEALED file's records: its Tetra records (scored) and its int4
//!      g128 records (the reference). With a second path, the bare all-Tetra
//!      encode the old journals used (pers-rowloop, step1-micro), as a control;
//!   2. checks every arm on the card before it is timed: the served
//!      `tv_tetra48_h` gives the reference rows of every record; the registry's
//!      `served` arm must equal them bit for bit, and gives the reference f32
//!      values; every other arm is held to what it owes (`rowcand::Owes`): the
//!      bits, a declared bound (max and 0.99 quantile of the relative error), or
//!      nothing (a diagnostic). An arm that fails is REFUSED and no rate is
//!      printed for it;
//!   3. times, in interleaved rounds after a warmup, on every set: the
//!      registered arms (`kernels/rowcand.cu`), the row-loop arms of the
//!      journals as controls (`pers_trows.cu`), the int4 arms of K11
//!      (`pers_q4rows.cu`) and the served eager `tv_q4_hh` on the sealed int4
//!      records, and the bare read probe (`bwprobe.cu`) over a 2 GiB buffer;
//!   4. prints a line a round and an arm (`ROUND`), the medians with their
//!      round-by-round ranges, the speed of each arm over its base formed round
//!      by round, the fidelity lines of the prereg, and one TSV summary (`TSV`).
//!
//! The sets: each d_in of the file's Tetra records apart, all of them in file
//! order, and 4E's gate/up phase (`tv_rowc_gu_*`: pairs of rows, gate then up,
//! at 4E's grid, with no other traffic on the card). GB/s are given in two
//! accountings: packed words only (row stride included), and words plus
//! binary16 tails and f32 row scales (the journals' 0.9467 GB accounting, for
//! the fidelity rule). Times are taken on the host around launch and sync, as
//! the row-loop bench took them.
//!
//! `--dump DIR` writes the module text (`rowbench_unit.cu`) and the driver's
//! cubin of it, for the offline SASS count (`ops/rowcand-sass.sh`).

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("rowbench targets NVIDIA GPUs; there is nothing to run here.");
}

#[cfg(target_os = "linux")]
fn main() {
    if let Err(e) = linux::run() {
        eprintln!("rowbench: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
    use llvq_cuda::gpu::{Cuda, KernelSource};
    use llvq_cuda::pers;
    use llvq_cuda::rowcand::{self, Owes, RowErrors};
    use std::time::Instant;

    const WARMUP: usize = 2;
    const THREADS: u32 = 512;

    /// The journals' medians of each known arm's speed over its base, on the
    /// bare encode (pers-rowloop-2026-11-05 lines 26-33, step1-micro-2026-11-07
    /// lines 27-31): (arm, base, median). The fidelity rule's order.
    const JOURNAL_SPEEDS: [(&str, &str, f64); 11] = [
        ("v3", "v2", 0.849),
        ("ilp2", "v2", 0.928),
        ("v2u", "v2", 0.825),
        ("l1w", "v2u", 1.214),
        ("bcast", "v2u", 0.987),
        ("xbcast", "v2u", 0.995),
        ("v2_o2", "v2_s", 1.076),
        ("nc", "v2", 0.810),
        ("nc256", "v2", 0.845),
        ("cpa", "v2", 0.960),
        ("tile", "v2", 1.002),
    ];
    /// bal is in step1-micro only: x0.934.
    const JOURNAL_BAL: f64 = 0.934;
    /// The fidelity window of the served row on the bare encode, GB/s in the
    /// journals' accounting: pers-rowloop 379 [368-383], step1-micro 374 [368-378].
    const FIDELITY_GBS: (f64, f64) = (368.0, 383.0);
    /// How far a known arm's speed over its base may sit from the journal's.
    const ORDER_TOL: f64 = 0.03;
    /// The known arms of the journals (pers.rs::TROWS_ARMS), timed as controls.
    const KNOWN: [&str; 14] = ["v2", "v3", "ilp2", "v2u", "l1w", "bcast", "xbcast", "v2_s", "v2_o2", "nc", "nc256", "cpa", "tile", "bal"];

    fn ptr<T>(s: &CudaSlice<T>, m: &Cuda) -> u64 {
        s.device_ptr(m.stream()).0
    }

    /// One Tetra record on the card.
    struct TRec {
        name: String,
        d_out: u32,
        d_in: u32,
        nb: u32,
        tw: u32,
        stride: u32,
        words: CudaSlice<u32>,
        gscale: CudaSlice<f32>,
        rscale: CudaSlice<f32>,
        tail: CudaSlice<u16>,
        /// tv_trows_tile's relaid copy, made once if the tile arm runs.
        tiled: Option<CudaSlice<u32>>,
        /// tv_tetra48_h's rows on the bench activation.
        refy: Vec<u16>,
    }

    /// One int4 g128 record on the card.
    struct QRec {
        d_out: u32,
        d_in: u32,
        wq: CudaSlice<u32>,
        sc: CudaSlice<u16>,
        bi: CudaSlice<u16>,
        tiled: Option<CudaSlice<u32>>,
    }

    /// The decoder tables' buffers, held for the run.
    type TableBufs = (CudaSlice<u32>, CudaSlice<u32>, CudaSlice<u16>, CudaSlice<u32>, CudaSlice<f32>);

    struct Tables {
        rows: u64,
        pref: u64,
        br: u64,
        suf: u64,
        inv: u64,
        _keep: TableBufs,
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Rows,
        GateUp,
    }

    /// A list of Tetra records of one file, scored as one launch an arm.
    struct Set {
        file: &'static str,
        name: String,
        mode: Mode,
        recs: Vec<usize>,
    }

    enum Launch {
        /// tv_rowc_* or tv_trows_*: (sites, nsites, rows, pref, br, suf, invnorm, x [, x2, stamps, ends]).
        Rows { f: CudaFunction, table: CudaSlice<u64>, nsites: u32, grid: u32, shared: u32, trows: bool },
        /// The served eager int4 rows, one tv_q4_hh a record.
        Q4hh { f: CudaFunction, recs: Vec<usize> },
        /// A tv_q4rows_* arm.
        Q4 { f: CudaFunction, table: CudaSlice<u64>, nsites: u32, grid: u32, shared: u32 },
        /// The read probe.
        Probe { f: CudaFunction, grid: u32, threads: u32, shared: u32, pf: u32, part: CudaSlice<u64> },
    }

    /// One timed thing: an arm on a set.
    struct Run {
        set: String,
        arm: String,
        kind: &'static str,
        owes: String,
        launch: Option<Launch>,
        bytes_w: u64,
        bytes_old: u64,
        refused: Option<String>,
        err: Option<RowErrors>,
        series: Vec<f64>,
        _bufs: Vec<CudaSlice<u16>>,
        _fbufs: Vec<CudaSlice<f32>>,
    }

    impl Run {
        fn new(set: &str, arm: &str, kind: &'static str, owes: String) -> Self {
            Run {
                set: set.to_string(),
                arm: arm.to_string(),
                kind,
                owes,
                launch: None,
                bytes_w: 0,
                bytes_old: 0,
                refused: None,
                err: None,
                series: Vec::new(),
                _bufs: Vec::new(),
                _fbufs: Vec::new(),
            }
        }
    }

    fn owes_str(o: Owes) -> String {
        match o {
            Owes::Bits => "bits".into(),
            Owes::Bound(b) => format!("bound {b:e}"),
            Owes::Nothing => "nothing".into(),
        }
    }

    fn pack_u8(b: &[u8]) -> Vec<u32> {
        b.chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                u32::from_le_bytes(w)
            })
            .collect()
    }

    /// The records of one `.llvq` file: Tetra (transcoded to the served
    /// stream) and int4, in file order, with their names.
    #[allow(clippy::type_complexity)]
    fn read_file(path: &str) -> Result<(Vec<llvq_artifact::RawMatrix>, Vec<llvq_artifact::Int4Matrix>), String> {
        let f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let mut r = std::io::BufReader::new(f);
        let h = llvq_artifact::read_header(&mut r).map_err(|e| format!("{path}: {e}"))?;
        let (mut t, mut q) = (Vec::new(), Vec::new());
        for _ in 0..h.matrices {
            match llvq_artifact::read_record(&mut r, h.version).map_err(|e| format!("{path}: {e}"))? {
                llvq_artifact::Record::Int4(m) => q.push(m),
                llvq_artifact::Record::Lattice(m) => {
                    if m.kind != llvq_artifact::CodeKind::Tetra {
                        return Err(format!("{path}: {} is a {} record, not Tetra", m.name, m.kind.name()));
                    }
                    if m.centroids.len() != 2 {
                        return Err(format!("{path}: {}: {} centroids, the decode reads 1 gain bit", m.name, m.centroids.len()));
                    }
                    t.push(m);
                }
            }
        }
        Ok((t, q))
    }

    fn upload_tetra(m: &Cuda, recs: &[llvq_artifact::RawMatrix]) -> Result<Vec<TRec>, String> {
        let mut v = Vec::with_capacity(recs.len());
        for r in recs {
            let nb = r.d_in / llvq_core::DIM;
            let tw = (r.d_in % llvq_core::DIM) as u32;
            if tw != 8 && tw != 16 {
                return Err(format!("{}: tail of {tw}, the row loops read 8 or 16", r.name));
            }
            if r.d_out % 8 != 0 {
                return Err(format!("{}: d_out {} is not whole launches of 8 rows", r.name, r.d_out));
            }
            let b = llvq_artifact::tetra48::transcode_tetra48(&r.indices, &r.gains, r.d_out, nb).map_err(|e| e.to_string())?;
            let words: Vec<u32> = b.data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            v.push(TRec {
                name: r.name.clone(),
                d_out: r.d_out as u32,
                d_in: r.d_in as u32,
                nb: nb as u32,
                tw,
                stride: b.stride_u32 as u32,
                words: m.up_u32(&words)?,
                gscale: m.up_f32(&[r.centroids[0] as f32, r.centroids[1] as f32])?,
                rscale: m.up_f32(&r.row_scales.iter().map(|&x| x as f32).collect::<Vec<_>>())?,
                tail: m.up_u16(&r.tail.iter().map(|&x| llvq_cuda::f16_bits(x as f32)).collect::<Vec<_>>())?,
                tiled: None,
                refy: Vec::new(),
            });
        }
        Ok(v)
    }

    fn upload_int4(m: &Cuda, recs: &[llvq_artifact::Int4Matrix]) -> Result<Vec<QRec>, String> {
        let mut v = Vec::with_capacity(recs.len());
        for q in recs {
            if q.group != 128 || q.bits != 4 || q.d_in % 256 != 0 || q.d_out % 8 != 0 {
                return Err(format!("{}: int4 g{} {}x{}: K11's rows read g128, d_in a multiple of 256, whole launches of 8 rows", q.name, q.group, q.d_out, q.d_in));
            }
            // little-endian: nibble i at bits 4 (i % 8) of word i / 8 (fused_cuda.rs)
            let w: Vec<u32> = q.packed.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            v.push(QRec { d_out: q.d_out as u32, d_in: q.d_in as u32, wq: m.up_u32(&w)?, sc: m.up_u16(&q.scales)?, bi: m.up_u16(&q.biases)?, tiled: None });
        }
        Ok(v)
    }

    /// tv_tetra48_h on every record: the reference rows.
    fn reference_rows(m: &Cuda, served: &Cuda, recs: &mut [TRec], t: &Tables, x: u64, shared: u32) -> Result<(), String> {
        let f = served.func("tv_tetra48_h")?;
        for r in recs.iter_mut() {
            let y = m.zeros_u16(r.d_out as usize)?;
            let cfg = LaunchConfig { grid_dim: (r.d_out / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: shared };
            let (w, gs, rs, tl, yp) = (ptr(&r.words, m), ptr(&r.gscale, m), ptr(&r.rscale, m), ptr(&r.tail, m), ptr(&y, m));
            let mut b = m.stream().launch_builder(&f);
            b.arg(&w).arg(&r.stride).arg(&t.rows).arg(&t.pref).arg(&t.br).arg(&t.suf).arg(&gs).arg(&t.inv).arg(&rs).arg(&tl).arg(&x).arg(&yp).arg(&r.nb).arg(&r.tw);
            // SAFETY: tv_tetra48_h's own arguments on live allocations of this record.
            unsafe { b.launch(cfg) }.map_err(|e| format!("tv_tetra48_h: {e}"))?;
            m.sync()?;
            r.refy = m.down_u16(&y)?;
        }
        Ok(())
    }

    fn rows_launch(m: &Cuda, l: &Launch, t: &Tables, x: u64) -> Result<(), String> {
        match l {
            Launch::Rows { f, table, nsites, grid, shared, trows } => {
                let cfg = LaunchConfig { grid_dim: (*grid, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: *shared };
                let z = 0u64;
                let mut b = m.stream().launch_builder(f);
                b.arg(table).arg(nsites).arg(&t.rows).arg(&t.pref).arg(&t.br).arg(&t.suf).arg(&t.inv).arg(&x);
                if *trows {
                    b.arg(&z).arg(&z).arg(&z);
                }
                // SAFETY: the kernel's own arguments; every pointer of the site
                // table names a live buffer of this run; no stamps, no x2.
                unsafe { b.launch(cfg) }.map_err(|e| format!("rows launch: {e}"))?;
            }
            _ => unreachable!("rows_launch on a non-row launch"),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(m: &Cuda, l: &Launch, t: &Tables, x: u64, xh: &CudaSlice<u16>, q4: &[QRec], q4y: &CudaSlice<u16>, probe: &CudaSlice<u32>) -> Result<(), String> {
        match l {
            Launch::Rows { .. } => rows_launch(m, l, t, x),
            Launch::Q4hh { f, recs } => {
                for &k in recs {
                    let r = &q4[k];
                    let cfg = LaunchConfig { grid_dim: (r.d_out / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 4 * r.d_in };
                    let gpr = r.d_in / 128;
                    let mut b = m.stream().launch_builder(f);
                    b.arg(&r.wq).arg(&r.sc).arg(&r.bi).arg(xh).arg(q4y).arg(&r.d_in).arg(&gpr);
                    // SAFETY: tv_q4_hh's own arguments on the record's live buffers; q4y holds the widest d_out.
                    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_q4_hh: {e}"))?;
                }
                Ok(())
            }
            Launch::Q4 { f, table, nsites, grid, shared } => {
                let cfg = LaunchConfig { grid_dim: (*grid, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: *shared };
                let z = 0u64;
                let mut b = m.stream().launch_builder(f);
                b.arg(table).arg(nsites).arg(xh).arg(&z).arg(&z);
                // SAFETY: the kernel's own arguments; every pointer of the site table is a live buffer; no stamps.
                unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| format!("q4 rows: {e}"))
            }
            Launch::Probe { f, grid, threads, shared, pf, part } => {
                let cfg = LaunchConfig { grid_dim: (*grid, 1, 1), block_dim: (*threads, 1, 1), shared_mem_bytes: *shared };
                let n = rowcand::BW_WORDS;
                let mut b = m.stream().launch_builder(f);
                b.arg(probe).arg(&n).arg(part).arg(pf);
                // SAFETY: the probe's own arguments: the 2^29-word buffer and one u64 a CTA.
                unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| format!("probe: {e}"))
            }
        }
    }

    pub fn run() -> Result<(), String> {
        let a: Vec<String> = std::env::args().skip(1).collect();
        let mut paths = Vec::new();
        let (mut rounds, mut dump) = (15usize, None::<std::path::PathBuf>);
        let mut i = 0;
        while i < a.len() {
            match a[i].as_str() {
                "--rounds" => {
                    rounds = a.get(i + 1).ok_or("--rounds N")?.parse().map_err(|e| format!("--rounds: {e}"))?;
                    i += 2;
                }
                "--dump" => {
                    dump = Some(a.get(i + 1).ok_or("--dump DIR")?.into());
                    i += 2;
                }
                p => {
                    paths.push(p.to_string());
                    i += 1;
                }
            }
        }
        if paths.is_empty() || paths.len() > 2 || rounds < 3 {
            return Err("usage: rowbench <sealed.bin> [<bare-tetra.bin>] [--rounds N >= 3] [--dump DIR]".into());
        }

        // ---- the files -------------------------------------------------------------------------------------
        let t0 = Instant::now();
        let (st, sq) = read_file(&paths[0])?;
        println!("sealed {}: {} Tetra records, {} int4 g128 records ({:.1} s)", paths[0], st.len(), sq.len(), t0.elapsed().as_secs_f64());
        if sq.is_empty() {
            return Err(format!("{}: no int4 record: this is not the sealed file (the sealed 4B carries v, o and 12 downs in int4)", paths[0]));
        }
        let bare = match paths.get(1) {
            Some(p) => {
                let t0 = Instant::now();
                let (bt, bq) = read_file(p)?;
                println!("bare {p}: {} Tetra records, {} int4 records ({:.1} s)", bt.len(), bq.len(), t0.elapsed().as_secs_f64());
                Some(bt)
            }
            None => {
                println!("bare: none given; the fidelity rule on the old encode is VOID");
                None
            }
        };

        // ---- the card and the modules ----------------------------------------------------------------------
        let tile = llvq_cuda::tile::resolve(llvq_cuda::gpu::probe_compute_cap()?);
        let tile_define = tile.define();
        let text = rowcand::rowc_unit_text(&tile_define);
        let src = KernelSource::new(&[text.as_str()]);
        let m = Cuda::new(&src)?;
        let dev = m.device()?;
        println!(
            "card {} sm_{}{}, {} SMs, L2 {} B, nameplate {:.0} GB/s (not a measurement); {}",
            dev.name,
            dev.compute_cap.0,
            dev.compute_cap.1,
            dev.sm_count,
            dev.l2_bytes,
            dev.nameplate_bandwidth_gbs(),
            tile.provenance()
        );
        println!("rowbench module: {} bytes, sha256 {}", src.text.len(), src.sha256);
        let served_text = pers::served_unit_text(&format!("{tile_define}{}", llvq_cuda::tile::Prefill::resolve()?.defines()));
        let ssrc = KernelSource::new(&[served_text.as_str()]);
        println!("reference: the served unit, {} bytes, sha256 {}", ssrc.text.len(), ssrc.sha256);
        let served = Cuda::on_stream(m.stream().clone(), &ssrc)?;
        if let Some(dir) = &dump {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            std::fs::write(dir.join("rowbench_unit.cu"), &src.text).map_err(|e| e.to_string())?;
            llvq_cuda::persbench::dump_cubin(dir, "rowbench", &src.text)?;
        }
        let sm = dev.sm_count as u32;
        // tv_tetra48_h's staging: the tile's blocks of 24 f32 (tile::Tile::shared_bytes)
        let ref_shared = tile.shared_bytes();

        let tr = llvq_bench::f1::Trellis::new();
        let table = llvq_bench::f1::rank::RankTable::build();
        if table.rows.len() != 4096 || table.n0_mixed != 1240 {
            return Err(format!("rank table: {} rows, N0 {}; the decoder reads 4096 and 1240", table.rows.len(), table.n0_mixed));
        }
        let mut invn = vec![0.0f32; llvq_artifact::tetra48::TETRA48_SHELLS];
        for (k, e) in invn.iter_mut().enumerate().skip(1) {
            *e = (1.0f64 / ((16 * k) as f64).sqrt()) as f32;
        }
        let keep = (
            m.up_u32(&table.rows)?,
            m.up_u32(&pack_u8(&llvq_bench::f1::rank::prefix_bytes(&tr)))?,
            m.up_u16(&llvq_bench::f1::rank::branch_words(&tr))?,
            m.up_u32(&pack_u8(&llvq_bench::f1::rank::suffix_bytes(&tr)))?,
            m.up_f32(&invn)?,
        );
        let t = Tables { rows: ptr(&keep.0, &m), pref: ptr(&keep.1, &m), br: ptr(&keep.2, &m), suf: ptr(&keep.3, &m), inv: ptr(&keep.4, &m), _keep: keep };

        // The bench activation: 9,728 values of a seeded gaussian (the row loops' K2 seed), and its binary16
        // for the int4 rows.
        let mut rng = llvq_core::SplitMix64::new(0x7e72);
        let xh_host: Vec<f32> = (0..9728).map(|_| rng.next_gaussian() as f32).collect();
        let x_buf = m.up_f32(&xh_host)?;
        let x = ptr(&x_buf, &m);
        let xh = m.up_u16(&xh_host.iter().map(|&v| llvq_cuda::f16_bits(v)).collect::<Vec<_>>())?;

        let mut sealed = upload_tetra(&m, &st)?;
        let q4 = upload_int4(&m, &sq)?;
        drop(st);
        reference_rows(&m, &served, &mut sealed, &t, x, ref_shared)?;
        let mut bare_recs = match bare {
            Some(b) => {
                // the old row loops' order: the served schedule of the 4B, 252 sites
                let mut recs = upload_tetra(&m, &b)?;
                drop(b);
                let order: Vec<String> = pers::served_4b_schedule()
                    .iter()
                    .map(|s| {
                        let p = match s.proj {
                            pers::Proj::Q => "self_attn.q_proj",
                            pers::Proj::K => "self_attn.k_proj",
                            pers::Proj::V => "self_attn.v_proj",
                            pers::Proj::O => "self_attn.o_proj",
                            pers::Proj::Gate => "mlp.gate_proj",
                            pers::Proj::Up => "mlp.up_proj",
                            pers::Proj::Down => "mlp.down_proj",
                        };
                        format!("model.layers.{}.{p}.weight", s.layer)
                    })
                    .collect();
                if recs.len() == order.len() && order.iter().all(|n| recs.iter().any(|r| &r.name == n)) {
                    recs.sort_by_key(|r| order.iter().position(|n| n == &r.name).unwrap_or(usize::MAX));
                    println!("bare: {} Tetra records in the served schedule's order (the row loops' order)", recs.len());
                } else {
                    println!("bare: {} Tetra records in file order (not the 252 of the served schedule)", recs.len());
                }
                reference_rows(&m, &served, &mut recs, &t, x, ref_shared)?;
                recs
            }
            None => Vec::new(),
        };

        // ---- the sets --------------------------------------------------------------------------------------
        let mut sets: Vec<Set> = Vec::new();
        let shapes = |recs: &[TRec]| -> Vec<u32> {
            let mut d: Vec<u32> = recs.iter().map(|r| r.d_in).collect();
            d.sort_unstable();
            d.dedup();
            d
        };
        let list = |file: &str, recs: &[TRec]| {
            for d in shapes(recs) {
                let mut kinds: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
                for r in recs.iter().filter(|r| r.d_in == d) {
                    let proj = r.name.rsplit('.').nth(1).unwrap_or("?");
                    *kinds.entry(format!("{proj} {}x{}", r.d_out, r.d_in)).or_insert(0) += 1;
                }
                println!("  {file} Tetra d_in {d}: {}", kinds.iter().map(|(k, n)| format!("{k} x{n}")).collect::<Vec<_>>().join(", "));
            }
        };
        list("sealed", &sealed);
        for d in shapes(&sealed) {
            sets.push(Set { file: "sealed", name: format!("sealed:d_in={d}"), mode: Mode::Rows, recs: (0..sealed.len()).filter(|&k| sealed[k].d_in == d).collect() });
        }
        sets.push(Set { file: "sealed", name: "sealed:all".into(), mode: Mode::Rows, recs: (0..sealed.len()).collect() });
        {
            let mut pairs = Vec::new();
            for (k, r) in sealed.iter().enumerate() {
                if r.name.contains("mlp.gate_proj") {
                    let up = r.name.replace("mlp.gate_proj", "mlp.up_proj");
                    if let Some(u) = sealed.iter().position(|s| s.name == up) {
                        if sealed[u].d_out == r.d_out && sealed[u].d_in == r.d_in {
                            pairs.push(k);
                            pairs.push(u);
                        }
                    }
                }
            }
            println!("  sealed gate/up pairs at 4E's geometry: {} layers", pairs.len() / 2);
            if !pairs.is_empty() {
                sets.push(Set { file: "sealed", name: "sealed:gu4e".into(), mode: Mode::GateUp, recs: pairs });
            }
        }
        if !bare_recs.is_empty() {
            list("bare", &bare_recs);
            for d in shapes(&bare_recs) {
                sets.push(Set { file: "bare", name: format!("bare:d_in={d}"), mode: Mode::Rows, recs: (0..bare_recs.len()).filter(|&k| bare_recs[k].d_in == d).collect() });
            }
            sets.push(Set { file: "bare", name: "bare:all".into(), mode: Mode::Rows, recs: (0..bare_recs.len()).collect() });
        }

        // ---- the runs: built and checked before any timing -------------------------------------------------
        let arms = rowcand::arms();
        let mut runs: Vec<Run> = Vec::new();
        let relay_any = arms.iter().any(|a| a.relay);
        let mut relaid: Vec<(String, &'static str, usize, CudaSlice<u32>)> = Vec::new();
        if relay_any {
            println!("relay arms: {}", arms.iter().filter(|a| a.relay).map(|a| a.name.as_str()).collect::<Vec<_>>().join(", "));
        }
        let tile_relayout = m.func("tv_trows_tile_relayout")?;
        for set in &sets {
            let recs: &mut Vec<TRec> = if set.file == "sealed" { &mut sealed } else { &mut bare_recs };
            let rows_of: Vec<usize> = set.recs.iter().map(|&k| recs[k].d_out as usize).collect();
            let total: usize = rows_of.iter().sum();
            let refy: Vec<u16> = set.recs.iter().flat_map(|&k| recs[k].refy.iter().copied()).collect();
            let (bw, bo) = set.recs.iter().fold((0u64, 0u64), |(a, b), &k| {
                let (w, o) = rowcand::rows_bytes(recs[k].d_out, recs[k].stride, recs[k].tw);
                (a + w, b + o)
            });
            println!("\nset {}: {} records, {} rows, {:.4} GB of words, {:.4} GB with tails and row scales", set.name, set.recs.len(), total, bw as f64 / 1e9, bo as f64 / 1e9);
            // the registry, the served arm first: its f32 rows are the reference of the others
            let mut served_f: Option<Vec<f32>> = None;
            for arm in &arms {
                let (sym, label) = match set.mode {
                    Mode::Rows => (arm.symbol(), arm.name.clone()),
                    Mode::GateUp => (arm.gu_symbol(), format!("gu_{}", arm.name)),
                };
                let mut run = Run::new(&set.name, &label, "rowc", owes_str(arm.owes));
                run.bytes_w = bw;
                run.bytes_old = bo;
                if served_f.is_none() && !arm.is_served() {
                    run.refused = Some("no reference: the served arm failed on this set".into());
                    runs.push(run);
                    continue;
                }
                let y = m.zeros_u16(total)?;
                let yf = m.zeros_f32(total)?;
                // the arm's words: the served stream, or its relaid copy (same stride)
                let mut wp: Vec<u64> = set.recs.iter().map(|&k| ptr(&recs[k].words, &m)).collect();
                if arm.relay {
                    let rf = m.func(&arm.relay_symbol())?;
                    for (j, &k) in set.recs.iter().enumerate() {
                        let key = (arm.name.clone(), set.file, k);
                        let have = relaid.iter().position(|(n, f, r, _)| (n, f, r) == (&key.0, &key.1, &key.2));
                        let p = match have {
                            Some(h) => ptr(&relaid[h].3, &m),
                            None => {
                                let r = &recs[k];
                                let buf = m.zeros_u32(r.d_out as usize * r.stride as usize)?;
                                let cfg = LaunchConfig { grid_dim: (1024, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                                let mut b = m.stream().launch_builder(&rf);
                                b.arg(&r.words).arg(&buf).arg(&r.stride).arg(&r.d_out).arg(&r.nb);
                                // SAFETY: the relay's own arguments: the record's words and a zeroed buffer of the same size.
                                unsafe { b.launch(cfg) }.map_err(|e| format!("{}: {e}", arm.relay_symbol()))?;
                                let p = ptr(&buf, &m);
                                relaid.push((key.0, key.1, key.2, buf));
                                p
                            }
                        };
                        wp[j] = p;
                    }
                    m.sync()?;
                }
                let mk_table = |with_f: bool| -> Vec<u64> {
                    let mut words = Vec::new();
                    let mut at = 0u64;
                    for (j, &k) in set.recs.iter().enumerate() {
                        let r = &recs[k];
                        words.extend(rowcand::rowc_site_words(
                            wp[j],
                            ptr(&r.gscale, &m),
                            ptr(&r.rscale, &m),
                            ptr(&r.tail, &m),
                            ptr(&y, &m) + 2 * at,
                            if with_f { ptr(&yf, &m) + 4 * at } else { 0 },
                            r.d_out,
                            r.nb,
                            r.tw,
                            r.stride,
                        ));
                        at += u64::from(r.d_out);
                    }
                    words
                };
                let shared = rowcand::rowc_smem_bytes(9728);
                let f = m.func_dynamic_shared(&sym, shared)?;
                let rep = m.report(&sym)?;
                let occ = f.occupancy_max_active_blocks_per_multiprocessor(THREADS, shared as usize, None).map_err(|e| format!("occupancy: {e}"))?;
                println!("  {label:<12} {sym}: grid {sm} x {THREADS}, {} registers, {} B local, {shared} B shared, {occ} a SM fit", rep.num_regs, rep.local_bytes);
                if rep.local_bytes != 0 || occ < 1 {
                    run.refused = Some(format!("spills {} B or does not fit ({occ} a SM)", rep.local_bytes));
                    runs.push(run);
                    continue;
                }
                let check = Launch::Rows { f: f.clone(), table: m.up_u64(&mk_table(true))?, nsites: set.recs.len() as u32, grid: sm, shared, trows: false };
                rows_launch(&m, &check, &t, x)?;
                m.sync()?;
                let gy = m.down_u16(&y)?;
                let gf = m.down_f32(&yf)?;
                let y_equal = gy.iter().zip(&refy).filter(|(a, b)| a == b).count();
                if arm.is_served() {
                    if y_equal == total {
                        println!("  {label:<12} {total} rows identical BIT FOR BIT to tv_tetra48_h: the reference f32 rows of this set");
                        served_f = Some(gf.clone());
                    } else {
                        run.refused = Some(format!("{y_equal} of {total} rows equal tv_tetra48_h: the served arm is not the served row"));
                        println!("  {label:<12} REFUSED: {}", run.refused.as_ref().unwrap());
                        runs.push(run);
                        continue;
                    }
                }
                let want_f = served_f.as_ref().expect("set above");
                let e = rowcand::row_errors(&gy, &gf, &refy, want_f, &rows_of);
                let ok = e.passes(arm.owes);
                println!(
                    "  {label:<12} owes {}: y equal {}/{}, f32 equal {}/{}, max rel err {:.3e}, q99 {:.3e}: {}",
                    owes_str(arm.owes),
                    e.y_equal,
                    e.rows,
                    e.f_equal,
                    e.rows,
                    e.max,
                    e.q99,
                    match arm.owes {
                        Owes::Nothing => "diagnostic",
                        _ if ok => "PASS",
                        _ => "REFUSED",
                    }
                );
                if !ok {
                    run.refused = Some(format!("owes {} and misses: max rel err {:.3e}", owes_str(arm.owes), e.max));
                }
                run.err = Some(e);
                run.launch = Some(Launch::Rows { f, table: m.up_u64(&mk_table(false))?, nsites: set.recs.len() as u32, grid: sm, shared, trows: false });
                run._bufs.push(y);
                run._fbufs.push(yf);
                runs.push(run);
            }
            // the known row-loop arms of the journals, on row sets
            if set.mode != Mode::Rows {
                continue;
            }
            let Some(want_f) = served_f else { continue };
            for &(name, sym, owed, minb) in pers::TROWS_ARMS.iter().filter(|a| KNOWN.contains(&a.0)) {
                let pick: Vec<usize> = (0..set.recs.len()).filter(|&j| !pers::trows_small_only(name) || recs[set.recs[j]].d_in <= 4096).collect();
                let mut run = Run::new(&set.name, name, "known", if owed { "bits".into() } else { "nothing".into() });
                if pick.is_empty() {
                    continue;
                }
                let (tile, cpa) = (name == "tile", name == "cpa");
                if cpa && !pick.iter().all(|&j| pers::trows_cpa_ok(recs[set.recs[j]].stride)) {
                    run.refused = Some("a row stride is not a multiple of 16 B".into());
                    runs.push(run);
                    continue;
                }
                let sub_rows: Vec<usize> = pick.iter().map(|&j| rows_of[j]).collect();
                let sub_total: usize = sub_rows.iter().sum();
                let y = m.zeros_u16(sub_total)?;
                let yf = m.zeros_f32(sub_total)?;
                let (mut words_c, mut words_t) = (Vec::new(), Vec::new());
                let (mut at, mut bw2, mut bo2) = (0u64, 0u64, 0u64);
                for &j in &pick {
                    let k = set.recs[j];
                    if tile && recs[k].tiled.is_none() {
                        let r = &recs[k];
                        let ts = pers::trows_tile_stride(r.nb);
                        if ts != r.stride {
                            return Err(format!("{}: the tile stride {ts} is not the served {}", r.name, r.stride));
                        }
                        let buf = m.zeros_u32(r.d_out as usize * ts as usize)?;
                        let cfg = LaunchConfig { grid_dim: (1024, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                        let mut b = m.stream().launch_builder(&tile_relayout);
                        b.arg(&r.words).arg(&r.stride).arg(&buf).arg(&ts).arg(&r.d_out).arg(&r.nb);
                        // SAFETY: the relayout's own arguments: the record's words and a zeroed buffer of d_out x ts.
                        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_trows_tile_relayout: {e}"))?;
                        recs[k].tiled = Some(buf);
                    }
                    let r = &recs[k];
                    let wp = if tile { ptr(r.tiled.as_ref().expect("relaid above"), &m) } else { ptr(&r.words, &m) };
                    for (with_f, out) in [(true, &mut words_c), (false, &mut words_t)] {
                        out.extend(rowcand::rowc_site_words(
                            wp,
                            ptr(&r.gscale, &m),
                            ptr(&r.rscale, &m),
                            ptr(&r.tail, &m),
                            ptr(&y, &m) + 2 * at,
                            if with_f { ptr(&yf, &m) + 4 * at } else { 0 },
                            r.d_out,
                            r.nb,
                            r.tw,
                            r.stride,
                        ));
                    }
                    let (w, o) = rowcand::rows_bytes(r.d_out, r.stride, r.tw);
                    bw2 += w;
                    bo2 += o;
                    at += u64::from(r.d_out);
                }
                m.sync()?;
                run.bytes_w = bw2;
                run.bytes_old = bo2;
                let shared = match (cpa, minb) {
                    (true, _) => pers::trows_cpa_smem_bytes(16),
                    (false, 1) => pers::trows_smem_bytes(9728),
                    _ => pers::trows_smem_bytes(4096),
                };
                let f = m.func_dynamic_shared(sym, shared)?;
                let rep = m.report(sym)?;
                let occ = f.occupancy_max_active_blocks_per_multiprocessor(THREADS, shared as usize, None).map_err(|e| format!("occupancy: {e}"))?;
                let grid = sm * minb;
                if rep.local_bytes != 0 || occ < minb {
                    run.refused = Some(format!("spills {} B or does not fit ({occ} a SM, {minb} needed)", rep.local_bytes));
                    runs.push(run);
                    continue;
                }
                let n = pick.len() as u32;
                let check = Launch::Rows { f: f.clone(), table: m.up_u64(&words_c)?, nsites: n, grid, shared, trows: true };
                rows_launch(&m, &check, &t, x)?;
                m.sync()?;
                let gy = m.down_u16(&y)?;
                let gf = m.down_f32(&yf)?;
                let sub_refy: Vec<u16> = pick.iter().flat_map(|&j| recs[set.recs[j]].refy.iter().copied()).collect();
                let mut sub_f = Vec::with_capacity(sub_total);
                let mut at = 0usize;
                for (j, &n_rows) in rows_of.iter().enumerate() {
                    if pick.contains(&j) {
                        sub_f.extend_from_slice(&want_f[at..at + n_rows]);
                    }
                    at += n_rows;
                }
                let e = rowcand::row_errors(&gy, &gf, &sub_refy, &sub_f, &sub_rows);
                let ok = !owed || e.bit_equal();
                println!(
                    "  {name:<12} {sym} (known, grid {grid}): y equal {}/{}, f32 equal {}/{}: {}",
                    e.y_equal,
                    e.rows,
                    e.f_equal,
                    e.rows,
                    if !owed { "diagnostic, owes nothing" } else if ok { "BIT FOR BIT" } else { "REFUSED" }
                );
                if !ok {
                    run.refused = Some(format!("owes the bits and moves {} rows", e.rows - e.f_equal.min(e.y_equal)));
                }
                run.err = Some(e);
                run.launch = Some(Launch::Rows { f, table: m.up_u64(&words_t)?, nsites: n, grid, shared, trows: true });
                run._bufs.push(y);
                run._fbufs.push(yf);
                runs.push(run);
            }
        }

        // ---- the int4 arms on the sealed int4 records --------------------------------------------------------
        let mut q4_shapes: Vec<u32> = q4.iter().map(|r| r.d_in).collect();
        q4_shapes.sort_unstable();
        q4_shapes.dedup();
        let mut q4_sets: Vec<(String, Vec<usize>)> = q4_shapes.iter().map(|&d| (format!("sealed:int4:d_in={d}"), (0..q4.len()).filter(|&k| q4[k].d_in == d).collect())).collect();
        q4_sets.push(("sealed:int4:all".into(), (0..q4.len()).collect()));
        let max_q4_out = q4.iter().map(|r| r.d_out).max().unwrap_or(8) as usize;
        let q4y = m.zeros_u16(max_q4_out)?;
        let mut q4 = q4;
        {
            let relayout = m.func("tv_q4rows_tile_relayout")?;
            for r in q4.iter_mut() {
                let buf = m.zeros_u32(r.d_out as usize * r.d_in as usize / 8)?;
                let nwords = r.d_in / 8;
                let cfg = LaunchConfig { grid_dim: (1024, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                let mut b = m.stream().launch_builder(&relayout);
                b.arg(&r.wq).arg(&buf).arg(&r.d_out).arg(&nwords);
                // SAFETY: the relayout's own arguments, both buffers d_out x nwords.
                unsafe { b.launch(cfg) }.map_err(|e| format!("tv_q4rows_tile_relayout: {e}"))?;
                r.tiled = Some(buf);
            }
            m.sync()?;
        }
        let f_q4hh = served.func("tv_q4_hh")?;
        let qshared = pers::q4rows_smem_bytes(9728);
        for (name, recs) in &q4_sets {
            let bytes: u64 = recs.iter().map(|&k| u64::from(q4[k].d_out) * (u64::from(q4[k].d_in) / 2 + 4 * u64::from(q4[k].d_in / 128))).sum();
            let total: usize = recs.iter().map(|&k| q4[k].d_out as usize).sum();
            println!("\nset {name}: {} int4 records, {total} rows, {:.4} GB of words, scales and biases", recs.len(), bytes as f64 / 1e9);
            // the served rows, record by record, then each arm against them
            let mut want = Vec::with_capacity(total);
            for &k in recs {
                let r = &q4[k];
                let y = m.zeros_u16(r.d_out as usize)?;
                let cfg = LaunchConfig { grid_dim: (r.d_out / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 4 * r.d_in };
                let gpr = r.d_in / 128;
                let mut b = m.stream().launch_builder(&f_q4hh);
                b.arg(&r.wq).arg(&r.sc).arg(&r.bi).arg(&xh).arg(&y).arg(&r.d_in).arg(&gpr);
                // SAFETY: tv_q4_hh's own arguments on the record's live buffers.
                unsafe { b.launch(cfg) }.map_err(|e| format!("tv_q4_hh: {e}"))?;
                m.sync()?;
                want.extend(m.down_u16(&y)?);
            }
            let mut run = Run::new(name, "q4hh", "int4", "served".into());
            run.bytes_w = bytes;
            run.bytes_old = bytes;
            run.launch = Some(Launch::Q4hh { f: f_q4hh.clone(), recs: recs.clone() });
            runs.push(run);
            for &(arm, sym, owed) in pers::Q4ROWS_ARMS.iter() {
                let mut run = Run::new(name, arm, "int4", if owed { "bits".into() } else { "nothing".into() });
                run.bytes_w = bytes;
                run.bytes_old = bytes;
                let y = m.zeros_u16(total)?;
                let mut words = Vec::new();
                let mut at = 0u64;
                for &k in recs {
                    let r = &q4[k];
                    let wp = if arm == "q4t" { ptr(r.tiled.as_ref().expect("relaid above"), &m) } else { ptr(&r.wq, &m) };
                    words.extend(pers::q4rows_site_words(wp, ptr(&r.sc, &m), ptr(&r.bi, &m), ptr(&y, &m) + 2 * at, r.d_out, r.d_in));
                    at += u64::from(r.d_out);
                }
                let f = m.func_dynamic_shared(sym, qshared)?;
                let rep = m.report(sym)?;
                let occ = f.occupancy_max_active_blocks_per_multiprocessor(THREADS, qshared as usize, None).map_err(|e| format!("occupancy: {e}"))?;
                if rep.local_bytes != 0 || occ < 1 {
                    run.refused = Some(format!("spills {} B or does not fit", rep.local_bytes));
                    runs.push(run);
                    continue;
                }
                let l = Launch::Q4 { f, table: m.up_u64(&words)?, nsites: recs.len() as u32, grid: sm, shared: qshared };
                launch(&m, &l, &t, x, &xh, &q4, &q4y, &m.zeros_u32(4)?)?;
                m.sync()?;
                let got = m.down_u16(&y)?;
                let same = got.iter().zip(&want).filter(|(a, b)| a == b).count();
                println!(
                    "  {arm:<12} {sym}: {same} of {total} rows equal tv_q4_hh: {}",
                    if !owed { "diagnostic, owes nothing" } else if same == total { "BIT FOR BIT" } else { "REFUSED" }
                );
                if owed && same != total {
                    run.refused = Some(format!("owes the bits, {same} of {total} rows equal"));
                }
                run.launch = Some(l);
                run._bufs.push(y);
                runs.push(run);
            }
        }

        // ---- the read probe -------------------------------------------------------------------------------
        let seed = 0x0b0b_u32;
        let probe = m.zeros_u32(rowcand::BW_WORDS as usize)?;
        {
            let f = m.func("tv_bw_fill")?;
            let cfg = LaunchConfig { grid_dim: (sm * 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let n = rowcand::BW_WORDS;
            let mut b = m.stream().launch_builder(&f);
            b.arg(&probe).arg(&n).arg(&seed);
            // SAFETY: the fill's own arguments on the 2^29-word buffer.
            unsafe { b.launch(cfg) }.map_err(|e| format!("tv_bw_fill: {e}"))?;
            m.sync()?;
        }
        let t0 = Instant::now();
        let expected = rowcand::bw_expected_sum(rowcand::BW_WORDS, seed);
        println!(
            "\nread probe: {} words ({:.3} GB, {:.1} x the card's L2), filled with bw_hash, expected sum {expected:#x} ({:.1} s on the host)",
            rowcand::BW_WORDS,
            rowcand::BW_WORDS as f64 * 4.0 / 1e9,
            rowcand::BW_WORDS as f64 * 4.0 / dev.l2_bytes.max(1) as f64,
            t0.elapsed().as_secs_f64()
        );
        for (k, threads, per_sm, pf) in rowcand::bw_configs() {
            let (kname, sym, _, _) = rowcand::BW_KERNELS[k];
            let label = format!("{kname}/t{threads}/c{per_sm}/pf{pf}");
            let mut run = Run::new("probe", &label, "probe", "the host's sum".into());
            run.bytes_w = rowcand::BW_WORDS * 4;
            run.bytes_old = run.bytes_w;
            let shared = rowcand::bw_smem_bytes(threads);
            let f = m.func(sym)?;
            let occ = f.occupancy_max_active_blocks_per_multiprocessor(threads, shared as usize, None).map_err(|e| format!("occupancy: {e}"))?;
            if occ < per_sm {
                run.refused = Some(format!("{occ} CTAs of {threads} fit a SM, {per_sm} asked"));
                println!("  {label:<18} REFUSED: {}", run.refused.as_ref().unwrap());
                runs.push(run);
                continue;
            }
            let grid = sm * per_sm;
            let part = m.stream().alloc_zeros::<u64>(grid as usize).map_err(|e| format!("alloc: {e}"))?;
            run.launch = Some(Launch::Probe { f, grid, threads, shared, pf, part });
            runs.push(run);
        }

        // ---- the rounds ------------------------------------------------------------------------------------
        let total_rounds = WARMUP + rounds;
        println!("\n{WARMUP} warmup rounds, then {rounds} timed; every valid run once a round, in the order above");
        for rd in 0..total_rounds {
            for run in runs.iter_mut() {
                if run.refused.is_some() {
                    continue;
                }
                let Some(l) = run.launch.as_ref() else { continue };
                let t0 = Instant::now();
                launch(&m, l, &t, x, &xh, &q4, &q4y, &probe)?;
                m.sync()?;
                let s = t0.elapsed().as_secs_f64();
                if let Launch::Probe { part, .. } = l {
                    // the result is used: the partials come back every round and must sum to the host's sum
                    let p: Vec<u64> = m.stream().clone_dtoh(part).map_err(|e| format!("probe D2H: {e}"))?;
                    let got = p.iter().fold(0u64, |a, &v| a.wrapping_add(v));
                    if got != expected {
                        run.refused = Some(format!("round {rd}: sum {got:#x}, expected {expected:#x}"));
                        println!("  probe {} REFUSED: {}", run.arm, run.refused.as_ref().unwrap());
                        continue;
                    }
                }
                if rd >= WARMUP {
                    run.series.push(s);
                    println!(
                        "ROUND\t{}\t{}\t{}\t{:.4}\t{:.1}\t{:.1}",
                        rd - WARMUP,
                        run.set,
                        run.arm,
                        1e3 * s,
                        run.bytes_w as f64 / s / 1e9,
                        run.bytes_old as f64 / s / 1e9
                    );
                }
            }
        }
        for run in runs.iter_mut() {
            if run.refused.is_some() {
                run.series.clear();
            }
        }

        // ---- the report ------------------------------------------------------------------------------------
        let series = |set: &str, arm: &str| runs.iter().find(|r| r.set == set && r.arm == arm && r.refused.is_none() && !r.series.is_empty()).map(|r| r.series.clone());
        let over = |base: &[f64], s: &[f64]| -> (f64, f64, f64) { pers::med_range(&base.iter().zip(s).map(|(b, t)| b / t).collect::<Vec<_>>()) };
        let mut cur = String::new();
        println!("\nmedians over {rounds} rounds [round-by-round range]; GB/s of packed words, then with tails and row scales");
        for r in &runs {
            if r.set != cur {
                println!("\n{}", r.set);
                cur = r.set.clone();
            }
            if let Some(why) = &r.refused {
                println!("  {:<18} REFUSED, no rate: {why}", r.arm);
                continue;
            }
            if r.series.is_empty() {
                continue;
            }
            let (ms, lo, hi) = pers::med_range(&r.series);
            let (g, gl, gh) = pers::med_range(&r.series.iter().map(|t| r.bytes_w as f64 / t / 1e9).collect::<Vec<_>>());
            let (o, ol, oh) = pers::med_range(&r.series.iter().map(|t| r.bytes_old as f64 / t / 1e9).collect::<Vec<_>>());
            let base = match r.kind {
                "rowc" if r.arm.starts_with("gu_") => Some("gu_served"),
                "rowc" => Some("served"),
                "known" => Some(match r.arm.as_str() {
                    // v2_s runs a subset of the sites: no ratio over v2
                    "v2_o2" | "v2_s" => "v2_s",
                    "l1w" | "bcast" | "xbcast" => "v2u",
                    _ => "v2",
                }),
                "int4" => Some("q4"),
                _ => None,
            };
            let vs = base
                .filter(|b| *b != r.arm)
                .and_then(|b| series(&r.set, b).map(|s| (b, over(&s, &r.series))))
                .map(|(b, (m2, l2, h2))| format!("  x{m2:.3} [{l2:.3}-{h2:.3}] over {b}"))
                .unwrap_or_default();
            let diag = if r.owes == "nothing" { "  (diagnostic)" } else { "" };
            println!("  {:<18} {:.3} ms [{:.3}-{:.3}]  {g:.0} GB/s [{gl:.0}-{gh:.0}]  {o:.0} [{ol:.0}-{oh:.0}]{vs}{diag}", r.arm, 1e3 * ms, 1e3 * lo, 1e3 * hi);
        }

        // ---- fidelity (the prereg's rule D) ----------------------------------------------------------------
        println!();
        match series("bare:all", "served") {
            None => println!("FIDELITY\tserved\tVOID\tno bare encode, or the served arm was refused on it"),
            Some(s) => {
                let r = runs.iter().find(|r| r.set == "bare:all" && r.arm == "served").expect("found above");
                let (g, gl, gh) = pers::med_range(&s.iter().map(|t| r.bytes_old as f64 / t / 1e9).collect::<Vec<_>>());
                let pass = g >= FIDELITY_GBS.0 && g <= FIDELITY_GBS.1;
                println!(
                    "FIDELITY\tserved\t{}\tbare:all {:.4} GB, median {g:.1} GB/s [{gl:.1}-{gh:.1}] against [{}, {}]",
                    if pass { "PASS" } else { "FAIL" },
                    r.bytes_old as f64 / 1e9,
                    FIDELITY_GBS.0,
                    FIDELITY_GBS.1
                );
            }
        }
        if let Some(v2) = series("bare:all", "v2") {
            let r = runs.iter().find(|r| r.set == "bare:all" && r.arm == "v2").expect("found above");
            let (g, gl, gh) = pers::med_range(&v2.iter().map(|t| r.bytes_old as f64 / t / 1e9).collect::<Vec<_>>());
            let pass = g >= FIDELITY_GBS.0 && g <= FIDELITY_GBS.1;
            println!("FIDELITY\tv2\t{}\tbare:all median {g:.1} GB/s [{gl:.1}-{gh:.1}] against [{}, {}]", if pass { "PASS" } else { "FAIL" }, FIDELITY_GBS.0, FIDELITY_GBS.1);
        }
        let mut order_fail = 0usize;
        let mut order_n = 0usize;
        for (arm, base, j) in JOURNAL_SPEEDS.iter().copied().chain(std::iter::once(("bal", "v2", JOURNAL_BAL))) {
            match (series("bare:all", arm), series("bare:all", base)) {
                (Some(s), Some(b)) => {
                    let (m2, l2, h2) = over(&b, &s);
                    let pass = (m2 - j).abs() <= ORDER_TOL;
                    order_n += 1;
                    order_fail += usize::from(!pass);
                    println!("ORDER\t{arm}\t{}\tx{m2:.3} [{l2:.3}-{h2:.3}] over {base}, journal x{j:.3}, tolerance {ORDER_TOL}", if pass { "PASS" } else { "FAIL" });
                }
                _ => println!("ORDER\t{arm}\tVOID\tno series on bare:all"),
            }
        }
        println!(
            "ORDER\tall\t{}\t{order_n} known arms against their journal speeds, {order_fail} outside +-{ORDER_TOL}",
            if order_n == 0 { "VOID" } else if order_fail == 0 { "PASS" } else { "FAIL" }
        );

        // ---- the TSV ---------------------------------------------------------------------------------------
        println!("\nTSV\tset\tarm\tkind\towes\tstatus\tbytes_words\tbytes_old\tms_med\tms_lo\tms_hi\tgbs_med\tgbs_lo\tgbs_hi\tgbs_old_med\tgbs_old_lo\tgbs_old_hi\ty_equal\tf32_equal\trows\tmax_rel\tq99_rel");
        for r in &runs {
            let (yq, fq, rows, mx, q) = r.err.as_ref().map(|e| (e.y_equal, e.f_equal, e.rows, e.max, e.q99)).unwrap_or((0, 0, 0, 0.0, 0.0));
            let status = match (&r.refused, r.owes.as_str()) {
                (Some(_), _) => "REFUSED",
                (None, "nothing") => "diagnostic",
                _ => "valid",
            };
            let rates = if r.refused.is_none() && !r.series.is_empty() {
                let (ms, lo, hi) = pers::med_range(&r.series);
                let (g, gl, gh) = pers::med_range(&r.series.iter().map(|t| r.bytes_w as f64 / t / 1e9).collect::<Vec<_>>());
                let (o, ol, oh) = pers::med_range(&r.series.iter().map(|t| r.bytes_old as f64 / t / 1e9).collect::<Vec<_>>());
                format!("{:.4}\t{:.4}\t{:.4}\t{g:.1}\t{gl:.1}\t{gh:.1}\t{o:.1}\t{ol:.1}\t{oh:.1}", 1e3 * ms, 1e3 * lo, 1e3 * hi)
            } else {
                "\t\t\t\t\t\t\t\t".to_string()
            };
            println!("TSV\t{}\t{}\t{}\t{}\t{status}\t{}\t{}\t{rates}\t{yq}\t{fq}\t{rows}\t{mx:.3e}\t{q:.3e}", r.set, r.arm, r.kind, r.owes, r.bytes_w, r.bytes_old);
        }
        Ok(())
    }
}
