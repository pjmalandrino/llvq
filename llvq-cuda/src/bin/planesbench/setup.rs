//! `run`'s first steps: the arm selection, the NVRTC unit, the decoder tables, the Tetra file.

use super::*;

/// The arm selection, parsed and checked before anything is built.
pub(super) fn select_arms() -> Result<(Vec<ArmSet>, Option<String>, ArmSet), String> {
    // The arm selection, BEFORE any build: an invalid value must fail
    // here, not after twenty minutes of transcoding.
    // ⚠️ It does NOT touch the NVRTC translation unit below. The selection
    // changes the buffers and the dispatch, never the compiled text nor
    // the register report (the opposite would redo É1).
    let arms_var = std::env::var("LLVQ_BENCH_ARMS").ok();
    let mut phases = arms::parse_phases(arms_var.as_deref())?;
    // The Tetra arm needs a SECOND file, and `runnable()` cannot know
    // whether this invocation was given one: it says a kernel exists, not
    // that anything can feed it.
    //
    // A bare `planesbench <ball.llvq>` is the invocation every published
    // run used, so a DEFAULT selection is narrowed rather than refused —
    // loudly, on its own line, because an arm that vanishes without a word
    // is the failure this whole step is about. An EXPLICIT
    // `LLVQ_BENCH_ARMS=…,tetra48,…` is refused instead, further down: that
    // is a selection the operator made, and narrowing it would be making
    // a different one for them.
    let tetra_path = std::env::args().nth(2);
    if tetra_path.is_none() && arms_var.is_none() {
        for p in phases.iter_mut() {
            p.remove(arms::TETRA48);
        }
        println!(
            "no second file: the tetra48 arm is OUT of this run. It reads a Tetra file, \
             which\n  cannot be the same file as the ball arms' — pass it as the second \
             argument."
        );
    }
    let phases = phases;
    let union: ArmSet = {
        let mut u = ArmSet::empty();
        for p in &phases {
            for a in p.iter() {
                u.insert(a);
            }
        }
        u
    };
    // 🚨 Every arm this binary can DISPATCH, and nothing else.
    //
    // `arms::HAS_KERNEL` says an arm's kernel exists **in the repository**;
    // it does not say this binary has a branch for it, and the two drifted
    // on 2026-09-09. `HAS_KERNEL[TETRA48]` went true when the arm entered
    // the registry, `parse_phases(None)` returns `ArmSet::runnable()`, and
    // for one day a bare `planesbench <model.llvq>` selected arm 17 and
    // reached `unreachable!("unknown arm")` — after the three minutes of
    // transcoding, on a rented card, with nothing on the development Mac
    // able to see it: this file is entirely `cfg(target_os = "linux")` and
    // the failure is a runtime panic, not a type error.
    //
    // So the list is stated here and checked in the first second. It is
    // hand-written because nothing can derive it from three `match` arms —
    // but a hand-written list that fails loud and early is worth a great
    // deal more than a derived one that fails late.
    for a in union.iter() {
        let dispatched = matches!(
            a,
            arms::SLOT32
                | arms::PLANES14
                | arms::PLANES12X
                | arms::GOLAY70V1
                | arms::GOLAY70V2
                | arms::E1V
                | arms::TETRA48
                | arms::NULLK
                | arms::FP16
                | arms::CUBLASF16
                | arms::AWQ
                | arms::QTIP
        );
        if !dispatched {
            return Err(format!(
                "planesbench cannot dispatch \"{}\" (arm {a}). It is registered in \
                 `arms.rs` and `HAS_KERNEL` says its kernel exists, but this binary has \
                 no branch for it — wire it, or take it out of the selection. Refused \
                 here rather than in `unreachable!` three minutes from now.",
                arms::ARM_NAMES[a]
            ));
        }
    }
    if arms_var.is_some() {
        println!("LLVQ_BENCH_ARMS selection:");
        for (k, p) in phases.iter().enumerate() {
            println!("  phase {}/{}: {}", k + 1, phases.len(), p.label());
        }
    }
    Ok((phases, tetra_path, union))
}

/// The NVRTC unit, its register report, and the card it runs on.
pub(super) fn compile(
    tile: llvq_cuda::tile::Tile,
    union: ArmSet,
) -> Result<(bool, llvq_cuda::persbench::Flags, KernelSource, Cuda, llvq_cuda::gpu::DeviceReport), String> {
    // Parts concatenated in dependency order: llvq_planes.cuh needs
    // llvq_slot.cuh (ClassRec, ext24), planes.cu needs matvec.cu
    // (warp_sum, TILE_COLS). NVRTC has no filesystem, the host includes.
    //
    // ⚠️ tv_slot and tv_f16 sit in a larger translation unit than in
    // `bin/matvec`: their register report below is the drift detector.
    let base = llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu"])?;
    let (pcuh, pcu, planes_overridden) = load_planes_sources()?;
    let (p12cuh, p12cu, planes12_overridden) = load_planes12_sources()?;
    let (gcuh, gcu, golay_overridden) = load_golay_sources()?;
    let (segcu, seg_overridden) = load_planes_seg_source()?;
    // The Tetra fusion arm, OFF by default. Its bit-exact comparison is
    // fatal when it runs — that is the whole point of it — and on
    // 2026-09-20 it fired and took the ten-arm cost table down with it,
    // before the table was printed. An experimental arm must not be able
    // to do that to the table the bench exists for.
    //
    // Refused by name on any other value, like every flag in this file.
    let seg_tetra = match std::env::var("LLVQ_SEG_TETRA").as_deref() {
        Err(_) | Ok("0") => false,
        Ok("1") => true,
        Ok(v) => {
            return Err(format!(
                "LLVQ_SEG_TETRA={v:?}: the values are 0 and 1, and unset is 0"
            ))
        }
    };
    // The persistent sections (llvq_cuda::persbench): L1's one-launch
    // Tetra pass, the row-loop microbenchmark, the driver cubins. Off
    // unless asked; read here, before anything is built.
    let pers_flags = llvq_cuda::persbench::Flags::from_env(seg_tetra, union.has(arms::TETRA48))?;
    if pers_flags.any() || pers_flags.dump_cubin.is_some() {
        println!(
            "Persistent sections: LLVQ_TETRA_OCC={} LLVQ_TROWS={} LLVQ_DUMP_CUBIN={}",
            u8::from(pers_flags.tetra_occ),
            u8::from(pers_flags.trows),
            pers_flags.dump_cubin.as_ref().map_or("unset".to_string(), |d| d.display().to_string())
        );
    }
    let (awqcu, awq_overridden) = load_awq_source()?;
    let (gv1cu, golay_v1_overridden) = load_golay_v1_source()?;
    // P1c. Same loader as the base pair, so `LLVQ_KERNEL_DIR` overrides it
    // the same way and the override is reported below like every other.
    let e1v = llvq_cuda::load_sources_many(&["llvq_e1v.cuh", "e1v.cu"])?;
    // The floor (P4 §2.5). It has no header of its own: `matvec.cu` is
    // enough, and it is concatenated after everything else like every
    // arrival.
    let nullk = llvq_cuda::load_sources_many(&["nullk.cu"])?;
    // QTIP (F2). Absent unless `ops/fetch-qtip.sh` has run and
    // `LLVQ_KERNEL_DIR` points at its output; the empty strings below then
    // contribute nothing to the translation unit — and, deliberately,
    // nothing to its hash either, so a machine without the fetch compiles
    // the SAME source as every published run.
    // The F1 family and the served Tetra decode (6.8a, 2026-09-10).
    // Always in the unit, selected or not: the unit
    // never varies with the selection, so every arm's sha256 moves
    // together or none does.
    //
    // The order is the caller's contract, stated identically in two files
    // (`tv_tetra48_h.cu:36-37`, `tetra48_v3g.cu:24-25`): llvq_slot.cuh,
    // matvec.cu, llvq_f1rank.cuh, llvq_f1rank_v3.cuh, llvq_tetra48.cuh,
    // then the arm. The first two are already `base.parts[0..1]` at the
    // head, so the four below go at the END of the unit — adding an arm
    // must never move a fragment of an arm that carries a published
    // number, and appending is the only placement that cannot.
    //
    // `f1rank.cu` and the v1/v2 variants are NOT here: `tetra48_v3g.cu`
    // names only the three headers, and `cuhcheck` assembles exactly that
    // unit.
    let f1 = llvq_cuda::load_sources_many(&[
        "llvq_f1rank.cuh",
        "llvq_f1rank_v3.cuh",
        "llvq_tetra48.cuh",
        "tetra48_v3g.cu",
        // A4 for Tetra, appended after the served arm for the rule that
        // governs every arrival: adding a kernel must never move a
        // fragment of an arm that carries a published number.
        "tetra48_seg.cu",
    ])?;
    let qtip_src = load_qtip_sources()?;
    let (qtip_cuh, qtip_glue) = match &qtip_src {
        Some((cuh, glue)) => (cuh.as_str(), glue.as_str()),
        None => ("", ""),
    };
    let defines = tile.define();
    let parts = [
        defines.as_str(),
        base.parts[0].as_str(),
        base.parts[1].as_str(),
        pcuh.as_str(),
        pcu.as_str(),
        // After planes.cu: it uses `planes_dot` and `TILE_COLS`, and this
        // is the order `tests/host_planes.cpp` compiles them in.
        segcu.as_str(),
        p12cuh.as_str(),
        p12cu.as_str(),
        gcuh.as_str(),
        gcu.as_str(),
        // The competitor last: adding an arm must never reorder the
        // fragments of the in-house arms.
        awqcu.as_str(),
        // Then the v1 witness (2026-08-11), latest arrival, same rule.
        gv1cu.as_str(),
        // And E1v last (P1c, 2026-08-16): the same rule again, and
        // `e1v.cu` needs `matvec.cu` (warp_sum, TILE_BLOCKS), already
        // concatenated at the head.
        e1v.parts[0].as_str(),
        e1v.parts[1].as_str(),
        nullk.parts[0].as_str(),
        // QTIP last, and for the rule that has governed every arrival:
        // adding an arm must never reorder the fragments of the arms that
        // produced a published number. The device half first, then our
        // shims, which name it.
        qtip_cuh,
        qtip_glue,
        // Last of all, and after QTIP for the same rule QTIP arrived
        // under. Nothing before this line moves by one notch.
        f1.parts[0].as_str(),
        f1.parts[1].as_str(),
        f1.parts[2].as_str(),
        f1.parts[3].as_str(),
        f1.parts[4].as_str(),
    ];
    let src = KernelSource::new(&parts);
    // On the assembled text: the one place where what NVRTC compiles and
    // what this report prints are both visible.
    tile.assert_defined_in(&src.text);
    println!("NVRTC source: {} bytes, sha256 {}", src.text.len(), src.sha256);
    if let Some(d) = &base.overridden_from {
        println!("  WARNING: Slot32 SOURCES OVERRIDDEN from {d}");
    }
    // The four F1/Tetra fragments were the only ones whose override was
    // never announced. `load_sources_many` returns it and it was dropped:
    // every `tetra48` figure — b/weight, ms, GB/s, the ratio, the register
    // report on `tv_f1r_v3g` — would have been attributed to the committed
    // kernel while a substituted one produced them.
    if let Some(d) = &f1.overridden_from {
        println!("  WARNING: F1/Tetra48 SOURCES OVERRIDDEN from {d}");
    }
    if let Some(d) = &planes_overridden {
        println!("  WARNING: Planes14 SOURCES OVERRIDDEN from {d}");
    }
    if let Some(d) = &planes12_overridden {
        println!("  WARNING: Planes12x SOURCES OVERRIDDEN from {d}");
    }
    if let Some(d) = &awq_overridden {
        println!("  WARNING: AWQ SOURCE OVERRIDDEN from {d}");
    }
    if let Some(d) = &golay_overridden {
        println!("  WARNING: Golay70 SOURCES OVERRIDDEN from {d}");
    }
    if let Some(d) = &golay_v1_overridden {
        println!("  WARNING: Golay70 v1 WITNESS SOURCE OVERRIDDEN from {d}");
    }
    if let Some(d) = &seg_overridden {
        println!("  WARNING: segmented Planes14 SOURCE OVERRIDDEN from {d}");
    }
    if let Some(d) = &e1v.overridden_from {
        println!("  WARNING: E1v SOURCES OVERRIDDEN from {d}");
    }
    if let Some(d) = &nullk.overridden_from {
        println!("  WARNING: nullk SOURCE OVERRIDDEN from {d}");
    }
    match &qtip_src {
        Some(_) => println!(
            "  WARNING: QTIP KERNEL LOADED (GPL v3, not redistributed by this \
             repository, see docs/qtip-provenance.md)"
        ),
        None => {
            if union.has(arms::QTIP) {
                return Err("the qtip arm is selected but its kernel is absent: run \
                            ops/fetch-qtip.sh and point LLVQ_QTIP_DIR at it"
                    .to_string());
            }
        }
    }

    // Built before the module so a shape without a shim is a Rust-side
    // error message, never a failed symbol lookup on a rented card.
    let qtip_names: Vec<String> = match &qtip_src {
        None => Vec::new(),
        Some(_) => llvq_cuda::qtip_host::QTIP_SHAPES
            .iter()
            .map(|&(d_out, d_in)| {
                llvq_cuda::qtip_host::kernel_name(d_out, d_in)
                    .expect("QTIP_SHAPES must name its own shims")
            })
            .collect(),
    };

    let cuda = Cuda::new(&src)?;
    let dev = cuda.device()?;
    println!(
        "\n{}: {} SM, L2 {:.1} MB (read), {} B of shared memory per block",
        dev.name,
        dev.sm_count,
        dev.l2_bytes as f64 / 1e6,
        dev.shared_per_block
    );
    // The segmented kernels sit in the same translation unit as the five
    // measured ones, so their register reports are part of the same drift
    // detector: `tv_planes_seg` differs from `tv_planes` by one warp-uniform
    // load, and anything more than a register or two between them means the
    // compiler did something the source does not say.
    for name in [
        "tv_slot",
        "tv_slot_seg",
        "tv_f16",
        "tv_planes",
        "tv_planes_seg",
        "tv_planes12x",
        "tv_golay70",
        "tv_golay70_v1",
        "awq_gemv_g128",
        // 🚨 `tv_e1v` is here for the `local_bytes != 0` line just below,
        // not for the table: the FLAT body was chosen over a body 24%
        // faster on Metal precisely so that nothing spills, and if this
        // kernel spills then that choice was wrong and the number measures
        // something else (E1v-CUDA preregistration §5.4).
        "tv_e1v",
        "tv_nullk",
        // The served Tetra decode. A kernel of OURS, so it falls into the
        // hard-stop branch below and a spill kills the run — which is what
        // F1d wants: the 6.4 kill read `num_regs ≤ 40` and
        // `local_bytes == 0` off this very function, and this is the only
        // evidence that NVRTC accepts it in a translation unit this much
        // larger than `f1rankfloor`'s.
        "tv_f1r_v3g",
    ]
    .iter()
    .copied()
    // The QTIP shims, when their source is present. They are reported for
    // the same reason `tv_e1v` is — the `local_bytes != 0` line below —
    // and for one more that is specific to this arm: **this report is the
    // only proof that NVRTC accepted the fetched kernel text at all**.
    // Everything about QTIP up to here was established by reading and by
    // host-side tests; whether libcu++-free inline PTX, `__shfl_sync` and
    // `__byte_perm` survive NVRTC is a question only a device compile
    // answers, and a successful lookup of these five names IS that answer.
    // They are listed even while `arms::HAS_KERNEL[QTIP]` is false,
    // deliberately: the translation unit never varies with arm selection
    // (`arms.rs`), so the compile can be proved a job before the arm is
    // allowed to run.
    .chain(qtip_names.iter().map(|s| s.as_str()))
    {
        let r = cuda.report(name)?;
        println!(
            "  {:<10} {:>3} registers, {} local bytes, sm_{}",
            r.name, r.num_regs, r.local_bytes, r.binary_version
        );
        if r.local_bytes != 0 {
            // 🚨 A spill in OUR kernel stops everything: it means a design
            // choice is wrong and the number would measure something
            // else. In a COMPETITOR's kernel ported as shipped it is a
            // FACT to report, not a defect to fix, their tuning at our
            // occupancy. Refusing to measure would be choosing not to
            // know.
            if qtip_names.iter().any(|q| q == name) {
                println!(
                    "  WARNING: {name} SPILLS {} B, competitor arm ported as shipped, \
                     measured anyway and declared (F2 prereg §7 A1)",
                    r.local_bytes
                );
            } else {
                return Err(format!("{name}: {} spill bytes", r.local_bytes));
            }
        }
    }
    Ok((seg_tetra, pers_flags, src, cuda, dev))
}

/// The decoder tables, the activation, and their device copies, uploaded once.
pub(super) fn upload_inputs(
    cuda: &Cuda,
    dev: &llvq_cuda::gpu::DeviceReport,
    union: ArmSet,
    g70_needed: bool,
) -> Result<Inputs, String> {
    // Enough repetitions that the LIGHTEST arm streams past the L2 twice
    // over. Sizing on a heavier arm would let the light one replay from
    // cache. With the E2 arm the lower bound is Golay70's 9-byte main
    // stream (its exceptions only add bytes), so size on 9, not 12.
    let one_pass: u64 = SHAPES
        .iter()
        .map(|&(_, o, i)| (o * (i / DIM)) as u64 * 9)
        .sum();
    let reps = (2 * dev.l2_bytes as u64 / one_pass + 1).max(2) as usize;

    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    // The Slot32 arm's five-word window bound, re-asserted as every
    // consumer of slot_dot must.
    assert!(24 + table.worst_width_slot() <= 160, "five-word window exceeded");
    // The Planes14 bound: three bit-planes address 8 levels; the layout
    // is only bijective while every class has at most 5.
    assert!(
        (0..table.n_entries()).all(|e| table.record(e).len <= 5),
        "a class exceeds 5 levels: 3 planes are no longer enough"
    );

    let mut tab = vec![0u32; TABLE_ENTRIES * REC_WORDS];
    for e in 0..TABLE_ENTRIES {
        tab[e * REC_WORDS + REC_WORDS - 1] = 1;
    }
    for ci in 0..fd.n_classes() {
        let lv = fd.levels(ci);
        let norm = ((16 * lv.shell) as f64).sqrt();
        let base = (1 + ci) * REC_WORDS;
        for k in 0..lv.len {
            tab[base + k] = ((lv.values[k] as f64 / norm) as f32).to_bits();
        }
        tab[base + REC_WORDS - 1] = lv.len as u32;
    }
    let d_tab = cuda.up_u32(&tab)?;

    // The served Tetra decode's tables: the universal 16 KiB rank table,
    // the trellis's three streams, and `1/sqrt(16 m)`. Built by the
    // reference (`llvq_bench::f1`) and uploaded once — `bin/f1rankfloor`
    // builds them from the same four calls, and a second construction here
    // would be a second thing to keep in step.
    let tetra_tabs = match union.has(arms::TETRA48) {
        false => None,
        true => {
            let tr = llvq_bench::f1::Trellis::new();
            let table = llvq_bench::f1::rank::RankTable::build();
            assert_eq!(table.rows.len(), 4096, "the rank table is not 4096 rows");
            assert_eq!(
                table.n0_mixed, 1240,
                "RankTable::n0_mixed is {}, F1R_N0_MIXED in llvq_f1rank.cuh is 1240: \
                 the split of the mixed order differs",
                table.n0_mixed
            );
            println!(
                "tetra48: rank table {} B, prefixes 128 B, branches 2048 B, suffixes 128 B, \
                 invnorm {} B, N0 = {}",
                table.rows.len() * 4,
                llvq_artifact::tetra48::TETRA48_SHELLS * 4,
                table.n0_mixed
            );
            Some(TetraTabs {
                rows: cuda.up_u32(&table.rows)?,
                prefixes: cuda.up_u32(&pack_u8(&llvq_bench::f1::rank::prefix_bytes(&tr)))?,
                branches: cuda.up_u16(&llvq_bench::f1::rank::branch_words(&tr))?,
                suffixes: cuda.up_u32(&pack_u8(&llvq_bench::f1::rank::suffix_bytes(&tr)))?,
                invnorm: cuda.up_f32(&invnorm_table())?,
            })
        }
    };

    // The Golay70 arm's two constant tables: the dedicated class table
    // (residue pairs / level values + mod-4 flags + coset flag) and the
    // canonical 4096-codeword table (16 KiB) the 12-bit rank resolves
    // through. Uploaded once, shared by every matrix and by BOTH golay70
    // arms — and not at all when neither is selected. ⚠️ Conditional on
    // the UNION of the phases, not on the current phase: unlike the
    // streams (Staged), these ~32 KiB are resident from the start when a
    // golay70 arm enters only at phase 2. Disclosed in the `Staged`
    // comment, along with the three other union-level buffers.
    // (`golay`/`g70cls`, host side, also serve the conditional transcode
    // in `build`.)
    let golay = llvq_core::Golay::new();
    let g70cls = golay70_classes(&fd);
    let d_gtab = match g70_needed {
        true => Some(cuda.up_u32(&golay70_gpu_table(&g70cls))?),
        false => None,
    };
    let d_cw = match g70_needed {
        true => Some(cuda.up_u32(&golay70_gpu_codewords(&golay))?),
        false => None,
    };

    // The E1v arm's three constant tables, plus the canonical Golay
    // codewords. Conditional on the UNION like golay70's, and declared
    // here for the same reason: ~54 KiB of records, 2 KiB of widths,
    // 2.5 KiB of binomials and 16 KiB of codewords are resident from the
    // start when the arm enters only at phase 2.
    //
    // 🚨 The Golay words are those of `Golay::codewords()`, NOT golay70's
    // reworked table: `llvq_e1v.cuh` indexes `golay[golay_base + gi]` in
    // the format's canonical order, and `golay70_gpu_codewords` produces
    // another one. Two tables that look alike, and only one is right.
    let (d_e1vtab, d_e1vpay, d_e1vbinom, d_e1vcw) = match union.has(arms::E1V) {
        true => {
            let (tabw, pay, binom) = e1v_tables(&fd, &golay);
            (
                Some(cuda.up_u32(&tabw)?),
                Some(cuda.up_u32(&pay)?),
                Some(cuda.up_u32(&binom)?),
                Some(cuda.up_u32(golay.codewords())?),
            )
        }
        false => (None, None, None, None),
    };

    let mut rng = SplitMix64::new(0x6_D07);
    // 16,384 until 2026-10-04, which refused the 14B `down_proj` at
    // `d_in` 17,408 and made the 14B unbenchable (census-14b-base).
    //
    // Extending is the one edit here that cannot move a published number.
    // `SplitMix64` is sequential, so the first 16,384 draws are the same
    // values they have always been, and a matrix reads `x[0..d_in]`: every
    // arm at every `d_in` <= 16,384 sees a bit-identical activation.
    // Reseeding or shortening would not have that property.
    const XLEN: usize = 20480;
    let x: Vec<f32> = (0..XLEN).map(|_| rng.next_gaussian() as f32).collect();
    let d_x = cuda.up_f32(&x)?;
    // The activation as the AWQ kernel reads it: binary16, by float4 of
    // eight. The LLVQ and FP16 arms read the f32. That is a format
    // difference between competitors, not a tuning knob, and it is
    // declared. It exists only when an arm with a binary16 input runs:
    // AWQ, since 2026-08-18 cublasf16 (GemmEx requires A and B of the
    // same type, so x must be binary16 like the weights), and since
    // 2026-08-20 qtip, whose kernel reads `const half2* x`.
    //
    // 🚨 This list is a list of ARMS, not a list of two names: the QTIP
    // arm was wired on 2026-08-20 reading `d_xh` without being added
    // here, so the preregistration command, `fp16,qtip` without AWQ,
    // would have panicked on `expect` after the transcode, on a rented
    // card. Found by an adversarial review of the code, not by the
    // typecheck: a missing `Option` is a perfectly typed state.
    let d_xh = match union.has(arms::AWQ)
        || union.has(arms::CUBLASF16)
        || union.has(arms::QTIP)
    {
        true => {
            let xh: Vec<u16> = x.iter().map(|&v| f16_bits(v)).collect();
            Some(cuda.up_u16(&xh)?)
        }
        false => None,
    };

    // The L = 5 swap of the Planes12x arm needs the exact searcher the
    // artifact was encoded with; built once, shared by every transcode.
    let searcher = Searcher::new();
    Ok(Inputs {
        reps, fd, table, d_tab, tetra_tabs, golay, g70cls, d_gtab, d_cw, d_e1vtab, d_e1vpay, d_e1vbinom,
        d_e1vcw, rng, x, d_x, d_xh, searcher,
    })
}

/// The Tetra records of the second file, by name, and its int4 count.
pub(super) fn read_tetra(
    tetra_path: &Option<String>,
    union: ArmSet,
) -> Result<(std::collections::HashMap<String, TetraSrc>, usize), String> {
    // ---- the SECOND file, read before the first ----------------------
    //
    // Tetra and Planes14 cannot share a file: they are different code
    // alphabets, and the served Tetra file also carries its `v_proj` as
    // int4 g128. So the bench takes two paths and reads each format from
    // its own — every arm still in ONE process, which is what rule 5 is
    // about, and what F1d promises.
    //
    // Read FIRST, and entirely, for the reason the arm selection is parsed
    // in the first second: a bad path or an unreadable record must kill
    // the job before the three minutes of transcoding the first file
    // costs, not after.
    //
    // `read_record` is the only entry that walks a mixed file
    // (`format.rs:1167`). Every other one — `read_matrix_raw` included,
    // which is what this bench used for four months — refuses an int4
    // record AFTER `get_record_head` has consumed the head, and the body
    // has no length prefix a stranded reader could skip over. That is
    // deliberate (`format.rs:1899-1907`), and it is why the loop below
    // does not simply filter.
    let mut tetra: std::collections::HashMap<String, TetraSrc> =
        std::collections::HashMap::new();
    let mut tetra_int4 = 0usize;
    if let Some(path) = &tetra_path {
        let f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let mut r = std::io::BufReader::new(f);
        let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
        for _ in 0..h.matrices {
            match llvq_artifact::read_record(&mut r, h.version).map_err(|e| e.to_string())? {
                llvq_artifact::Record::Int4(m) => {
                    // Counted, never timed. `v_proj` is stored weights, a
                    // second kernel's business (`tv_q4_h`), and the arm
                    // measured here is the lattice decode. What matters is
                    // that the count reaches the report: an arm covering
                    // 216 of 252 matrices is not comparable, launch for
                    // launch, to one covering 252.
                    tetra_int4 += 1;
                    let _ = m;
                }
                llvq_artifact::Record::Lattice(m) => {
                    if m.kind != llvq_artifact::CodeKind::Tetra {
                        return Err(format!(
                            "{path}: {} is a {} record. The second path is the \
                             Tetra file; the first is the ball one, and swapping \
                             them is the one mistake that would otherwise run",
                            m.name,
                            m.kind.name()
                        ));
                    }
                    assert_eq!(
                        m.centroids.len(),
                        2,
                        "{}: the tetra decode hard-codes 1 gain bit",
                        m.name
                    );
                    let nblocks = m.d_in / DIM;
                    let blocks = llvq_artifact::tetra48::transcode_tetra48(
                        &m.indices, &m.gains, m.d_out, nblocks,
                    )
                    .map_err(|e| e.to_string())?;
                    tetra.insert(
                        m.name.clone(),
                        TetraSrc {
                            d_out: m.d_out,
                            d_in: m.d_in,
                            blocks,
                            centroids: [m.centroids[0] as f32, m.centroids[1] as f32],
                            rscale: m.row_scales.iter().map(|&v| v as f32).collect(),
                            tail: m.tail.iter().map(|&v| v as f32).collect(),
                        },
                    );
                }
            }
        }
        println!(
            "  {path} — {} matrices: {} Tetra, {tetra_int4} int4 g128 (counted, not timed)",
            h.matrices,
            tetra.len()
        );
    }
    if union.has(arms::TETRA48) && tetra.is_empty() {
        return Err("tetra48 is selected and no Tetra file was given. Pass the ball file \
                    as the first argument and the Tetra file as the SECOND — the two \
                    formats do not share a file. Or drop tetra48 from LLVQ_BENCH_ARMS."
            .to_string());
    }
    Ok((tetra, tetra_int4))
}
