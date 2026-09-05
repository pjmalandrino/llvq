//! The compiled floor of the F1 universal-table decoder: the word stream and
//! the full decode, in `tv_nullk`'s geometry, checked on the card against the
//! Rust reference before a millisecond is printed.
//!
//! Preregistered: `proofs/preregistration-f1-rang-plancher-2026-09-05.md`.
//! Not a gate, and it kills nothing (§1): it says under what form `tv_l3e8`
//! gets written, and whether the decoder is rewritten first.
//!
//! ## The ladder, and why no arm is read alone
//!
//! ```text
//!   nullk   the same pass without one byte of weights     (the floor, in THIS process)
//!   word    nullk + the 6-byte word per block, folded into a float, no decode
//!   f1r     word + the full decode: 3 rows, 3 pattern bytes, 24 coordinates, 24 FMAs
//!
//!   S  = t(word) − t(nullk)     the F1 stream in our geometry
//!   Du = t(f1r)  − t(word)      table + arithmetic decode
//!   T  = t(f1r)  − t(nullk)     what F1 spends on stream AND decode
//! ```
//!
//! `T` is read against `B = t(Planes14) − t(nullk) = 2.797 ms` from another
//! process (prereg §4): a difference read against a difference, never a time
//! subtracted from another process's time. Three arms every round, in an
//! order that ROTATES (round r opens with arm r mod 3 — the table floor's
//! negative `Didx` was possibly a position effect), differences formed round
//! by round, medians with ranges.
//!
//! ## The controls, and if one falls no number is printed
//!
//! 1. the card's decoder returns the reference's points: 256 blocks of every
//!    stream copy are decoded by `tv_f1r_dump` and compared coordinate by
//!    coordinate to `llvq_bench::f1::rank::decode_word`, on words the host
//!    replays from the mixer written once in `llvq_f1rank.cuh`;
//! 2. nothing is elided: `f1r`'s output differs from `word`'s and `nullk`'s,
//!    `word`'s from `nullk`'s;
//! 3. everything is observable: every output row written, finite, not all zero;
//! 4. the stream does not fit the L2: word bytes ≥ 4× the card's attribute;
//! 5. one process, one geometry, `nullk`'s;
//! 6. registers and local bytes of the three kernels, from the function
//!    attributes.
//!
//! ## What this bench cannot be
//!
//! A production cost: no gain scale, uniform labels rather than a model's,
//! no Planes14 in the process. A quality measurement: nothing here touches a
//! model. What it is: the first F1 decoder compiled and verified on the card
//! against a reference, which the table floor was not.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("f1rankfloor targets NVIDIA GPUs; there is nothing to run here.");
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    linux::run()
}

#[cfg(target_os = "linux")]
mod linux {
    use cudarc::driver::PushKernelArg;
    use llvq_bench::f1::rank::{branch_words, decode_word, prefix_bytes, suffix_bytes, RankTable};
    use llvq_bench::f1::Trellis;
    use llvq_core::{SplitMix64, DIM};
    use llvq_cuda::gpu::{Cuda, KernelSource};
    use llvq_cuda::TILE_BLOCKS;
    use std::time::Instant;

    // Eleven rounds, two discarded: nine kept, a multiple of three, so with the
    // rotating order every arm opens a round exactly three times.
    const ROUNDS: usize = 11;
    const WARMUP: usize = 2;
    const THREADS: u32 = 256;
    const LAYERS: usize = 36;
    /// Blocks decoded on the card per stream copy for control 1, in row-major
    /// block order — five of the seven shapes have 106 blocks a row, so "256
    /// blocks of row 0" does not exist there; the first 256 blocks of the
    /// stream span rows 0..2 and exercise the row stride on the way.
    const NDUMP: u32 = 256;
    const SEED: u64 = 0x00F1_2026_0906;
    /// `t(Planes14) − t(nullk)` measured in ANOTHER process
    /// (`docs/format-noyau.md` §6): the scale `T` is read against, never a
    /// subtrahend. The ratio beside it is `5.103 / 2.306`.
    const B_MS: f64 = 2.797;
    const B_RATIO: f64 = 5.103 / 2.306;

    /// The seven projection shapes of Qwen3-4B — `planesbench`'s table and
    /// `nullkbench`'s, so 252 launches a round in the geometry every published
    /// floor uses.
    const SHAPES: [(&str, usize, usize); 7] = [
        ("q_proj", 4096, 2560),
        ("k_proj", 1024, 2560),
        ("v_proj", 1024, 2560),
        ("o_proj", 2560, 4096),
        ("gate_proj", 9728, 2560),
        ("up_proj", 9728, 2560),
        ("down_proj", 2560, 9728),
    ];

    const ARMS: [&str; 3] = ["nullk", "word", "f1r"];

    struct Shape {
        name: &'static str,
        d_out: u32,
        nblocks: u32,
        tail_w: u32,
        /// Row stride of the word stream, in u32: round_up(6·nblocks, 8) / 4.
        stride_u32: u32,
        rscale: cudarc::driver::CudaSlice<f32>,
        tail: cudarc::driver::CudaSlice<f32>,
        x: cudarc::driver::CudaSlice<f32>,
        y: cudarc::driver::CudaSlice<f32>,
    }

    /// One layer's copy of one shape's word stream, and the seed that filled it.
    struct Stream {
        seed: u32,
        words: cudarc::driver::CudaSlice<u32>,
    }

    /// The four tables, uploaded once. The two byte tables travel packed four
    /// to a u32, little-endian, the way `planesbench` packs QTIP's u16 stream:
    /// the kernel casts nothing, it declares `const unsigned char*` and reads
    /// byte i at byte i, which on a little-endian device is what the packing
    /// put there.
    struct Tables {
        rows: cudarc::driver::CudaSlice<u32>,
        prefixes: cudarc::driver::CudaSlice<u32>,
        branches: cudarc::driver::CudaSlice<u16>,
        suffixes: cudarc::driver::CudaSlice<u32>,
    }

    struct Fns {
        nullk: cudarc::driver::CudaFunction,
        word: cudarc::driver::CudaFunction,
        f1r: cudarc::driver::CudaFunction,
    }

    /// round_up(6·nblocks, 8) / 4.
    fn stride_u32(nblocks: u32) -> u32 {
        (6 * nblocks).div_ceil(8) * 2
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

    /// `f1r_mix32` of `kernels/llvq_f1rank.cuh`, bit for bit.
    fn mix32(mut h: u32) -> u32 {
        h ^= h >> 16;
        h = h.wrapping_mul(0x7feb_352d);
        h ^= h >> 15;
        h = h.wrapping_mul(0x846c_a68b);
        h ^= h >> 16;
        h
    }

    /// The first `n` u32 of a stream copy, as `f1r_fill` wrote them, as bytes.
    fn host_bytes(seed: u32, n: usize) -> Vec<u8> {
        (0..n as u32)
            .flat_map(|i| mix32(seed.wrapping_add(i)).to_le_bytes())
            .collect()
    }

    /// Block `jb` of row `row`: the 48 bits at byte `4·row·stride + 6·jb`, read
    /// off the byte stream and NOT through `f1r_load`'s shift arithmetic — so a
    /// mistake in that arithmetic is a dump mismatch, not a shared error.
    fn host_word(bytes: &[u8], stride_u32: u32, row: u32, jb: u32) -> u64 {
        let at = (row * stride_u32 * 4 + 6 * jb) as usize;
        let mut b = [0u8; 8];
        b[..6].copy_from_slice(&bytes[at..at + 6]);
        u64::from_le_bytes(b)
    }

    fn build(cuda: &Cuda, rng: &mut SplitMix64) -> Result<Vec<Shape>, String> {
        let mut out = Vec::new();
        for &(name, d_out, d_in) in SHAPES.iter() {
            assert_eq!(d_out as u32 % (THREADS / 32), 0, "{name}: rows must fill whole blocks");
            let mut f = |n: usize| -> Vec<f32> {
                (0..n).map(|_| 0.5 + (rng.next() >> 40) as f32 / 16_777_216.0).collect()
            };
            let nblocks = (d_in / DIM) as u32;
            let tail_w = (d_in % DIM) as u32;
            let stride = stride_u32(nblocks);
            // `f1r_load` of the last block reads up to u32 index (3j >> 1) + 1,
            // i.e. up to byte 4·((3j >> 1) + 2); the row's stride must cover
            // it or the last row's last block reads past the buffer.
            assert!(
                4 * (((3 * (nblocks - 1)) >> 1) + 2) <= stride * 4,
                "{name}: row stride {} B does not cover the last block's two-word window",
                stride * 4
            );
            let x = f(d_in);
            let tail = f(d_out * tail_w as usize);
            let rscale = f(d_out);
            out.push(Shape {
                name,
                d_out: d_out as u32,
                nblocks,
                tail_w,
                stride_u32: stride,
                rscale: cuda.up_f32(&rscale)?,
                tail: cuda.up_f32(&tail)?,
                x: cuda.up_f32(&x)?,
                y: cuda.zeros_f32(d_out)?,
            });
        }
        Ok(out)
    }

    /// One stream copy, filled on the device.
    fn fill(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        rng: &mut SplitMix64,
        n: usize,
    ) -> Result<Stream, String> {
        let seed = rng.next() as u32;
        let n32 = n as u32;
        let mut words = cuda.zeros_u32(n)?;
        let c = cudarc::driver::LaunchConfig {
            grid_dim: (n32.div_ceil(THREADS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut b = cuda.stream().launch_builder(f);
            b.arg(&mut words).arg(&n32).arg(&seed);
            unsafe { b.launch(c) }.map_err(|e| format!("f1r_fill: {e}"))?;
        }
        Ok(Stream { seed, words })
    }

    fn cfg(s: &Shape, shared: u32) -> cudarc::driver::LaunchConfig {
        cudarc::driver::LaunchConfig {
            grid_dim: (s.d_out * 32 / THREADS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared,
        }
    }

    /// One timed round over the 252 launches of `tv_nullk`.
    fn round_null(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        shapes: &mut [Shape],
        shared: u32,
    ) -> Result<f64, String> {
        let t = Instant::now();
        for _ in 0..LAYERS {
            for s in shapes.iter_mut() {
                let c = cfg(s, shared);
                let mut b = cuda.stream().launch_builder(f);
                b.arg(&s.rscale).arg(&s.tail).arg(&s.x).arg(&mut s.y).arg(&s.nblocks).arg(&s.tail_w);
                unsafe { b.launch(c) }.map_err(|e| format!("nullk/{}: {e}", s.name))?;
            }
        }
        cuda.sync()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// One timed round of `tv_f1r_word`, layer `l` reading its own copy.
    fn round_word(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        shapes: &mut [Shape],
        streams: &[Vec<Stream>],
        shared: u32,
    ) -> Result<f64, String> {
        let t = Instant::now();
        for layer in streams.iter() {
            for (s, st) in shapes.iter_mut().zip(layer.iter()) {
                let c = cfg(s, shared);
                let mut b = cuda.stream().launch_builder(f);
                b.arg(&st.words)
                    .arg(&s.stride_u32)
                    .arg(&s.rscale)
                    .arg(&s.tail)
                    .arg(&s.x)
                    .arg(&mut s.y)
                    .arg(&s.nblocks)
                    .arg(&s.tail_w);
                unsafe { b.launch(c) }.map_err(|e| format!("word/{}: {e}", s.name))?;
            }
        }
        cuda.sync()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// One timed round of `tv_f1r`.
    fn round_f1r(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        shapes: &mut [Shape],
        streams: &[Vec<Stream>],
        tab: &Tables,
        shared: u32,
    ) -> Result<f64, String> {
        let t = Instant::now();
        for layer in streams.iter() {
            for (s, st) in shapes.iter_mut().zip(layer.iter()) {
                let c = cfg(s, shared);
                let mut b = cuda.stream().launch_builder(f);
                b.arg(&st.words)
                    .arg(&s.stride_u32)
                    .arg(&tab.rows)
                    .arg(&tab.prefixes)
                    .arg(&tab.branches)
                    .arg(&tab.suffixes)
                    .arg(&s.rscale)
                    .arg(&s.tail)
                    .arg(&s.x)
                    .arg(&mut s.y)
                    .arg(&s.nblocks)
                    .arg(&s.tail_w);
                unsafe { b.launch(c) }.map_err(|e| format!("f1r/{}: {e}", s.name))?;
            }
        }
        cuda.sync()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// Arm `k` of [`ARMS`], one round.
    fn run_arm(
        k: usize,
        cuda: &Cuda,
        fns: &Fns,
        shapes: &mut [Shape],
        streams: &[Vec<Stream>],
        tab: &Tables,
        shared: u32,
    ) -> Result<f64, String> {
        match k {
            0 => round_null(cuda, &fns.nullk, shapes, shared),
            1 => round_word(cuda, &fns.word, shapes, streams, shared),
            _ => round_f1r(cuda, &fns.f1r, shapes, streams, tab, shared),
        }
    }

    fn median_range(v: &[f64]) -> (f64, f64, f64) {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        (s[s.len() / 2], s[0], s[s.len() - 1])
    }

    /// Every output row written, finite, and different from every arm named
    /// in `others`. The last part is what catches an elided load: a compiler
    /// that deleted the fetches would leave the multiplier constant and the
    /// output would match another arm's to the bit.
    fn observable(
        cuda: &Cuda,
        shapes: &[Shape],
        others: &[(&str, &[Vec<f32>])],
        who: &str,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::new();
        for (i, s) in shapes.iter().enumerate() {
            let y = cuda.down_f32(&s.y)?;
            if y.iter().any(|v| !v.is_finite()) || y.iter().all(|v| *v == 0.0) {
                return Err(format!("{who}/{}: output not observable", s.name));
            }
            for &(oname, oy) in others {
                if y == oy[i] {
                    return Err(format!(
                        "{who}/{}: output identical to {oname}'s — the loads were elided",
                        s.name
                    ));
                }
            }
            out.push(y);
        }
        Ok(out)
    }

    /// Control 1: `tv_f1r_dump` against `decode_word`, every coordinate of
    /// `NDUMP` blocks of every stream copy. Returns the blocks compared.
    fn dump_check(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        shapes: &[Shape],
        streams: &[Vec<Stream>],
        tab: &Tables,
        table: &RankTable,
        tr: &Trellis,
    ) -> Result<usize, String> {
        let ndump = NDUMP;
        let mut out = cuda.zeros_u32((NDUMP as usize * DIM).div_ceil(4))?;
        let mut checked = 0usize;
        for (l, layer) in streams.iter().enumerate() {
            for (s, st) in shapes.iter().zip(layer.iter()) {
                let c = cudarc::driver::LaunchConfig {
                    grid_dim: (NDUMP.div_ceil(THREADS), 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                };
                {
                    let mut b = cuda.stream().launch_builder(f);
                    b.arg(&st.words)
                        .arg(&s.stride_u32)
                        .arg(&tab.rows)
                        .arg(&tab.prefixes)
                        .arg(&tab.branches)
                        .arg(&tab.suffixes)
                        .arg(&mut out)
                        .arg(&s.nblocks)
                        .arg(&ndump);
                    unsafe { b.launch(c) }.map_err(|e| format!("dump/{}: {e}", s.name))?;
                }
                cuda.sync()?;
                let got: Vec<i8> = cuda
                    .down_u32(&out)?
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .map(|b| b as i8)
                    .collect();
                // Rows touched by the first NDUMP blocks, plus the two-word
                // window past the last one.
                let nrows = NDUMP.div_ceil(s.nblocks);
                let bytes = host_bytes(st.seed, (nrows * s.stride_u32 + 2) as usize);
                for j in 0..NDUMP {
                    let (row, jb) = (j / s.nblocks, j % s.nblocks);
                    let word = host_word(&bytes, s.stride_u32, row, jb);
                    let want = decode_word(word, table, tr);
                    let j = j as usize;
                    for (cidx, (&g, &w)) in got[j * DIM..(j + 1) * DIM].iter().zip(want.iter()).enumerate() {
                        if g as i32 != w {
                            return Err(format!(
                                "dump: layer {l}, {}, block {j} (row {row}, block {jb}), coordinate {cidx}: \
                                 the card decoded {g}, the Rust reference {w}, on word {word:#014x}",
                                s.name
                            ));
                        }
                    }
                }
                checked += NDUMP as usize;
            }
        }
        Ok(checked)
    }

    pub fn run() -> Result<(), String> {
        let base = llvq_cuda::load_sources_many(&[
            "llvq_slot.cuh",
            "matvec.cu",
            "llvq_f1rank.cuh",
            "f1rank.cu",
            "nullk.cu",
        ])?;
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let mut parts: Vec<&str> = vec![defines.as_str()];
        parts.extend(base.parts.iter().map(String::as_str));
        let src = KernelSource::new(&parts);
        println!("F1 rank-table floor — the stream and the compiled decode, 252 launches, one process");
        println!("NVRTC source: {} bytes, sha256 {}", src.text.len(), src.sha256);
        if let Some(d) = &base.overridden_from {
            println!("⚠️ kernel sources overridden from {d}; the sha256 above is the only provenance");
        }
        let cuda = Cuda::new(&src)?;
        let dev = cuda.device()?;
        println!(
            "card: {} · {} SM · shared/block {} default, {} opt-in · L2 {:.0} MiB",
            dev.name,
            dev.sm_count,
            dev.shared_per_block,
            dev.shared_per_block_optin,
            dev.l2_bytes as f64 / (1024.0 * 1024.0)
        );

        // Control 6: registers and local bytes, from the function attributes.
        // Printed, and the prereg's thresholds (§7: ≤ 64 registers, 0 local)
        // flagged beside them; a spill does not stop the run, it is the
        // finding.
        println!("\n  kernel attributes (prereg §7 reads: registers ≤ 64, local bytes = 0):");
        for name in ["tv_nullk", "tv_f1r_word", "tv_f1r"] {
            let r = cuda.report(name)?;
            let flag = if r.local_bytes != 0 {
                "   🚨 LOCAL MEMORY: a spill on the hottest path"
            } else if r.num_regs > 64 {
                "   ⚠️ over 64 registers"
            } else {
                ""
            };
            println!(
                "  {:<12} {:>3} registers, {} local bytes, sm_{}{flag}",
                r.name, r.num_regs, r.local_bytes, r.binary_version
            );
        }

        let tile = (TILE_BLOCKS * DIM * 4) as u32;
        let mut rng = SplitMix64::new(SEED);
        let mut shapes = build(&cuda, &mut rng)?;

        // The tables, built by the reference and uploaded once.
        let tr = Trellis::new();
        let table = RankTable::build();
        assert_eq!(table.rows.len(), 4096, "the rank table is not 4096 rows");
        assert_eq!(
            table.n0_mixed, 1240,
            "RankTable::n0_mixed is {}, F1R_N0_MIXED in llvq_f1rank.cuh is 1240: the split of the mixed order differs",
            table.n0_mixed
        );
        let tab = Tables {
            rows: cuda.up_u32(&table.rows)?,
            prefixes: cuda.up_u32(&pack_u8(&prefix_bytes(&tr)))?,
            branches: cuda.up_u16(&branch_words(&tr))?,
            suffixes: cuda.up_u32(&pack_u8(&suffix_bytes(&tr)))?,
        };
        println!(
            "tables: rows {} B, prefixes 128 B, branches 2048 B, suffixes 128 B, N0 = {}",
            table.rows.len() * 4,
            table.n0_mixed
        );

        let fns = Fns {
            nullk: cuda.func("tv_nullk")?,
            word: cuda.func("tv_f1r_word")?,
            f1r: cuda.func("tv_f1r")?,
        };
        let f_dump = cuda.func("tv_f1r_dump")?;
        let f_fill = cuda.func("f1r_fill")?;

        // The stream: LAYERS distinct copies per shape, generated on the
        // device. Control 4 before anything is timed.
        let mut streams: Vec<Vec<Stream>> = Vec::with_capacity(LAYERS);
        let mut total_bytes = 0u64;
        for _ in 0..LAYERS {
            let mut layer = Vec::with_capacity(shapes.len());
            for s in shapes.iter() {
                let n = (s.d_out * s.stride_u32) as usize;
                total_bytes += n as u64 * 4;
                layer.push(fill(&cuda, &f_fill, &mut rng, n)?);
            }
            streams.push(layer);
        }
        cuda.sync()?;
        println!(
            "stream: {LAYERS} distinct copies × {} shapes, {:.3} GB of words on the device ({:.1}× the L2)",
            shapes.len(),
            total_bytes as f64 / 1e9,
            total_bytes as f64 / dev.l2_bytes as f64
        );
        if total_bytes < 4 * dev.l2_bytes as u64 {
            return Err(format!(
                "the word stream is {total_bytes} B against an L2 of {} B; below 4× it is a hit rate, not a stream",
                dev.l2_bytes
            ));
        }

        // Control 1 before any round.
        let checked = dump_check(&cuda, &f_dump, &shapes, &streams, &tab, &table, &tr)?;
        println!("control 1: {checked} blocks decoded on the card equal the Rust reference, every coordinate");

        // Interleaved rounds: every arm every round, the order rotating by
        // one each round, warmup discarded, differences formed ROUND BY
        // ROUND and never as a quotient of minima.
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
        for rep in 0..ROUNDS {
            for pos in 0..ARMS.len() {
                let k = (pos + rep) % ARMS.len();
                let t = run_arm(k, &cuda, &fns, &mut shapes, &streams, &tab, tile)?;
                if rep >= WARMUP {
                    times[k].push(t);
                }
            }
        }

        // Controls 2 and 3, on a dedicated untimed pass of each arm.
        run_arm(0, &cuda, &fns, &mut shapes, &streams, &tab, tile)?;
        let y_null = observable(&cuda, &shapes, &[], "nullk")?;
        run_arm(1, &cuda, &fns, &mut shapes, &streams, &tab, tile)?;
        let y_word = observable(&cuda, &shapes, &[("nullk", y_null.as_slice())], "word")?;
        run_arm(2, &cuda, &fns, &mut shapes, &streams, &tab, tile)?;
        observable(
            &cuda,
            &shapes,
            &[("nullk", y_null.as_slice()), ("word", y_word.as_slice())],
            "f1r",
        )?;
        println!("controls 2, 3: every output finite and written; f1r ≠ word ≠ nullk, f1r ≠ nullk");

        println!(
            "\n  {ROUNDS} rounds, {WARMUP} discarded; round r opens with arm r mod 3; every difference formed ROUND BY ROUND\n"
        );
        for (i, name) in ARMS.iter().enumerate() {
            let (m, lo, hi) = median_range(&times[i]);
            println!("  {name:<8} {m:8.3} ms  [{lo:.3}–{hi:.3}]");
        }

        let diff = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x - y).collect() };
        let (m, lo, hi) = median_range(&diff(&times[1], &times[0]));
        println!("\n  S  = word − nullk   {m:8.3} ms  [{lo:.3}–{hi:.3}]   the F1 stream in our geometry");
        let (m, lo, hi) = median_range(&diff(&times[2], &times[1]));
        println!("  Du = f1r − word     {m:8.3} ms  [{lo:.3}–{hi:.3}]   table + arithmetic decode (against D(16 KiB) = 0.663 ms of the table floor)");
        let (m, lo, hi) = median_range(&diff(&times[2], &times[0]));
        println!("  T  = f1r − nullk    {m:8.3} ms  [{lo:.3}–{hi:.3}]   stream AND decode, read against B = {B_MS} ms (Planes14 − nullk, ANOTHER process)");
        let ratio: Vec<f64> = times[2].iter().zip(&times[0]).map(|(a, b)| a / b).collect();
        let (m, lo, hi) = median_range(&ratio);
        println!(
            "  t(f1r)/t(nullk)     {m:8.4}     [{lo:.4}–{hi:.4}]   against 5.103/2.306 = {B_RATIO:.2}, same reserve"
        );
        println!("\n  WARNING: a FLOOR, not a cost. No gain scale, uniform labels rather than a model's,");
        println!("  and no Planes14 in this process. B is a scale to read T against, never a subtrahend:");
        println!("  no time here compares to a time from ANOTHER process.");
        Ok(())
    }
}
