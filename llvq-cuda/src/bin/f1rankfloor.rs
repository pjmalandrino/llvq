//! The compiled floor of the F1 universal-table decoder: the word stream,
//! the full decode, and three other arithmetics for the same decode, in
//! `tv_nullk`'s geometry, checked on the card against the Rust reference and
//! against each other before a millisecond is printed.
//!
//! Preregistered twice, and neither is a gate (§1 of both): the floor,
//! `proofs/preregistration-f1-rang-plancher-2026-09-05.md`, and the three
//! variants against it, `proofs/preregistration-f1-rang-variantes-2026-09-05.md`.
//! The second decides under which writing the decoder goes on, nothing else.
//!
//! ## The ladder, and why no arm is read alone
//!
//! ```text
//!   nullk    the same pass without one byte of weights     (the floor, in THIS process)
//!   word     nullk + the 6-byte word per block, folded into a float, no decode
//!   f1r      word + the full decode: 3 rows, 3 pattern bytes, 24 coordinates, 24 FMAs
//!   f1r_v1   f1r, the floats built without the I2F pipe: byte lanes, a LOP3 sign mux, 2²³ bias
//!   f1r_v2   f1r, the 3 dependent small-table loads replaced by F₂ algebra on the word
//!   f1r_v3   f1r, the values from byte tables in registers, looked up by prmt
//!
//!   S     = t(word)   − t(nullk)     the F1 stream in our geometry
//!   Du    = t(f1r)    − t(word)      table + arithmetic decode
//!   T     = t(f1r)    − t(nullk)     what F1 spends on stream AND decode
//!   Du_vk = t(f1r_vk) − t(word)      read against Du of the SAME process
//!   T_vk  = t(f1r_vk) − t(nullk)     read against T of the SAME process
//! ```
//!
//! `T` and the `T_vk` are read against `B = t(Planes14) − t(nullk) = 2.797 ms`
//! from another process (floor prereg §4): a difference read against a
//! difference, never a time subtracted from another process's time. Six arms
//! every round, in an order that ROTATES (round r opens with arm r mod 6 — the
//! table floor's negative `Didx` was possibly a position effect), differences
//! formed round by round, medians with ranges. Fourteen rounds, two of warmup:
//! twelve kept, a multiple of six, so every arm opens a round exactly twice.
//!
//! The three variants read the SAME word stream and the SAME 16 KiB table as
//! `tv_f1r`, through the SAME argument list (`f1rank_v1.cu`, `f1rank_v2.cu`,
//! `f1rank_v3.cu` copy `tv_f1r`'s signature argument for argument), so one
//! launch routine serves the four table arms.
//!
//! ## The controls, and if one falls no number is printed
//!
//! 1. the card's decoder returns the reference's points: 256 blocks of every
//!    stream copy are decoded by `tv_f1r_dump` and compared coordinate by
//!    coordinate to `llvq_bench::f1::rank::decode_word`, on words the host
//!    replays from the mixer written once in `llvq_f1rank.cuh`;
//! 2. nothing is elided: every table arm's output differs from `word`'s and
//!    `nullk`'s, `word`'s from `nullk`'s;
//! 3. everything is observable: every output row written, finite, not all zero;
//! 4. the stream does not fit the L2: word bytes ≥ 4× the card's attribute;
//! 5. one process, one geometry, `nullk`'s;
//! 6. registers and local bytes of the six kernels, from the function
//!    attributes;
//! 7. the variants compute what `tv_f1r` computes: on the last round's
//!    outputs, for EVERY ROW of every shape, `|y_vk[r] − y[r]| ≤ 1e-5 · max(1, |y[r]|)`
//!    — the prereg's §4.2, to the letter. The same 24 products per block,
//!    summed in `tv_f1r`'s coordinate order, with at most one extra rounding
//!    per block (V2 and V3 sum each block from zero and add; V1 runs
//!    `tv_f1r`'s very FMA chain and is expected at Δ = 0). The rounding
//!    drift of that reordering, replayed on the host with this bench's exact
//!    arithmetic (32 lanes each chaining its blocks, the `warp_sum`
//!    butterfly, the real table and word stream, all 30,720 rows): worst
//!    3.3e-7 per row, no row anywhere near zero (`min |y| ≥ 279`, every
//!    factor positive) — 30× under the tolerance (*computed*, review of
//!    2026-09-05; a first draft of this gate rested on a synthetic model
//!    that put 539 rows over it, and was wrong). A single wrong coordinate in
//!    a single block moves a row by at least `min x · min rscale = 0.25`,
//!    which is 3× the tolerance on `down_proj` (`max |y| ≈ 8,500`) and 6 to
//!    10× elsewhere: thin on one shape, but control 1 already holds the
//!    decode itself to the reference coordinate by coordinate; this control
//!    catches a variant that computes something else. The shape's ∞-norm
//!    reading is printed beside it, as information. A variant over the
//!    tolerance is HORS JEU — its times are not printed — and the other arms
//!    are read (prereg §6, first row); the times are refused wholesale only
//!    if `tv_f1r` itself fails controls 1 to 5.
//!
//! Controls 2, 3 and 7 all read the outputs captured during the LAST round,
//! one download per arm after its sync, outside every timed span.
//!
//! ## What this bench cannot be
//!
//! A production cost: no gain scale, uniform labels rather than a model's,
//! no Planes14 in the process. A quality measurement: nothing here touches a
//! model. What it is: four writings of the same decoder, compiled, verified on
//! the card against one reference, and timed side by side.

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

    // Fourteen rounds, two discarded: twelve kept, a multiple of six, so with
    // the rotating order every arm opens a round exactly twice.
    const ROUNDS: usize = 14;
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
    /// Control 7: `|y_vk[r] − y[r]| ≤ TOL · max(1, |y[r]|)` on every row —
    /// the prereg's §4.2 as written. The module header carries the replayed
    /// drift (3.3e-7 worst) and the margin on a wrong coordinate.
    const TOL: f64 = 1e-5;

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

    /// The six arms, in the order the rotation walks them. Arms 2..6 are the
    /// four table arms: `tv_f1r` and its three variants, one argument list.
    const ARMS: [&str; 6] = ["nullk", "word", "f1r", "f1r_v1", "f1r_v2", "f1r_v3"];
    /// The kernel behind each arm, same index.
    const KERNELS: [&str; 6] = ["tv_nullk", "tv_f1r_word", "tv_f1r", "tv_f1r_v1", "tv_f1r_v2", "tv_f1r_v3"];
    /// Index of `f1r` in [`ARMS`]; the variants are the arms after it.
    const F1R: usize = 2;

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
        /// `tv_f1r`, `tv_f1r_v1`, `tv_f1r_v2`, `tv_f1r_v3` — [`ARMS`]`[F1R..]`,
        /// launched by one routine with one argument list.
        table: [cudarc::driver::CudaFunction; 4],
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

    /// One timed round of a table arm — `tv_f1r` or one of its three
    /// variants, which copy its signature argument for argument:
    /// `(words, row_stride_u32, rows, prefixes, branches, suffixes, rscale,
    /// tail, x, y, nblocks, tail_w)`. `who` names the arm in an error.
    fn round_table(
        who: &str,
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
                unsafe { b.launch(c) }.map_err(|e| format!("{who}/{}: {e}", s.name))?;
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
            k => round_table(ARMS[k], cuda, &fns.table[k - F1R], shapes, streams, tab, shared),
        }
    }

    fn median_range(v: &[f64]) -> (f64, f64, f64) {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        (s[s.len() / 2], s[0], s[s.len() - 1])
    }

    /// The output rows of every shape, as the last launch left them.
    fn capture(cuda: &Cuda, shapes: &[Shape]) -> Result<Vec<Vec<f32>>, String> {
        shapes.iter().map(|s| cuda.down_f32(&s.y)).collect()
    }

    /// Controls 2 and 3 on one arm's captured outputs: every row written,
    /// finite, and different from every arm named in `others`. The last part
    /// is what catches an elided load: a compiler that deleted the fetches
    /// would leave the multiplier constant and the output would match
    /// another arm's to the bit.
    fn observable(
        shapes: &[Shape],
        ys: &[Vec<f32>],
        others: &[(&str, &[Vec<f32>])],
        who: &str,
    ) -> Result<(), String> {
        for (i, s) in shapes.iter().enumerate() {
            let y = &ys[i];
            if y.len() != s.d_out as usize {
                return Err(format!("{who}/{}: {} rows captured for {} rows", s.name, y.len(), s.d_out));
            }
            if y.iter().any(|v| !v.is_finite()) || y.iter().all(|v| *v == 0.0) {
                return Err(format!("{who}/{}: output not observable", s.name));
            }
            for &(oname, oy) in others {
                if *y == oy[i] {
                    return Err(format!(
                        "{who}/{}: output identical to {oname}'s — the loads were elided",
                        s.name
                    ));
                }
            }
        }
        Ok(())
    }

    /// What control 7 measured on one variant against `tv_f1r`.
    struct Drift {
        /// The gate: the largest `|Δ_r| / max(1, |y_r|)` over every row of
        /// every shape; where it sits; how many rows read over [`TOL`].
        worst: f64,
        shape: usize,
        row: usize,
        over: usize,
        /// Information, not the gate: the largest, over shapes, of
        /// `max_r |Δ_r| / max(1, max_r |y_r|)` — the shape's ∞-norm reading.
        worst_inf: f64,
    }

    /// Control 7, one variant: its captured outputs against `tv_f1r`'s, every
    /// row of every shape. Measures and does not refuse: every variant is
    /// measured before any is set aside, so one job says which shape and row
    /// broke the tolerance, and by how much, for all three.
    fn drift(shapes: &[Shape], y_ref: &[Vec<f32>], y_v: &[Vec<f32>], who: &str) -> Result<Drift, String> {
        let mut d = Drift { worst: 0.0, shape: 0, row: 0, over: 0, worst_inf: 0.0 };
        for (i, s) in shapes.iter().enumerate() {
            let (a, b) = (&y_ref[i], &y_v[i]);
            if a.len() != b.len() {
                return Err(format!("control 7: {who}/{}: {} rows against f1r's {}", s.name, b.len(), a.len()));
            }
            let inf = a.iter().fold(0f64, |m, &y| m.max((y as f64).abs())).max(1.0);
            for (r, (&y, &yv)) in a.iter().zip(b).enumerate() {
                let abs = (yv as f64 - y as f64).abs();
                let rel = abs / (y as f64).abs().max(1.0);
                if rel > d.worst {
                    d.worst = rel;
                    d.shape = i;
                    d.row = r;
                }
                if rel > TOL {
                    d.over += 1;
                }
                d.worst_inf = d.worst_inf.max(abs / inf);
            }
        }
        Ok(d)
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
        // One string for NVRTC: the floor's four parts, then each variant's
        // header and arm, then `nullk.cu`. `bin/cuhcheck` parses this very
        // list as one unit, so a name two variants both define fails there.
        let base = llvq_cuda::load_sources_many(&[
            "llvq_slot.cuh",
            "matvec.cu",
            "llvq_f1rank.cuh",
            "f1rank.cu",
            "llvq_f1rank_v1.cuh",
            "f1rank_v1.cu",
            "llvq_f1rank_v2.cuh",
            "f1rank_v2.cu",
            "llvq_f1rank_v3.cuh",
            "f1rank_v3.cu",
            "nullk.cu",
        ])?;
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let mut parts: Vec<&str> = vec![defines.as_str()];
        parts.extend(base.parts.iter().map(String::as_str));
        let src = KernelSource::new(&parts);
        println!(
            "F1 rank-table floor — the stream, the compiled decode and its three variants, 252 launches, one process"
        );
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

        // Control 6: registers and local bytes of the six kernels, from the
        // function attributes. Printed, and the preregs' thresholds (≤ 64
        // registers, 0 local) flagged beside them; a spill does not stop the
        // run, it is the finding.
        println!("\n  kernel attributes (prereg reads: registers ≤ 64, local bytes = 0):");
        for name in KERNELS {
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
            nullk: cuda.func(KERNELS[0])?,
            word: cuda.func(KERNELS[1])?,
            table: [
                cuda.func(KERNELS[F1R])?,
                cuda.func(KERNELS[F1R + 1])?,
                cuda.func(KERNELS[F1R + 2])?,
                cuda.func(KERNELS[F1R + 3])?,
            ],
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
        // ROUND and never as a quotient of minima. During the last round the
        // outputs of every arm are captured — after its sync, before the
        // next arm's clock starts — for controls 2, 3 and 7.
        let n = ARMS.len();
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut last: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n];
        for rep in 0..ROUNDS {
            for pos in 0..n {
                let k = (pos + rep) % n;
                let t = run_arm(k, &cuda, &fns, &mut shapes, &streams, &tab, tile)?;
                if rep >= WARMUP {
                    times[k].push(t);
                }
                if rep + 1 == ROUNDS {
                    last[k] = capture(&cuda, &shapes)?;
                }
            }
        }

        // Controls 2 and 3, on the last round's outputs. A table arm is
        // checked against `word` and `nullk`, not against `f1r`: V1 is
        // EXPECTED to equal `f1r` to the bit (control 7), so that equality
        // is not an elision.
        observable(&shapes, &last[0], &[], ARMS[0])?;
        observable(&shapes, &last[1], &[(ARMS[0], last[0].as_slice())], ARMS[1])?;
        for k in F1R..n {
            observable(
                &shapes,
                &last[k],
                &[(ARMS[0], last[0].as_slice()), (ARMS[1], last[1].as_slice())],
                ARMS[k],
            )?;
        }
        println!("controls 2, 3: every output finite and written; f1r, f1r_v1, f1r_v2, f1r_v3 ≠ word ≠ nullk");

        // Control 7: every variant against `tv_f1r`, every row of every
        // shape, on the last round's outputs. Every variant is measured and
        // printed; a variant over the tolerance is set aside (its times are
        // not printed) and the other arms are read — prereg §6, first row.
        let rows: u32 = shapes.iter().map(|s| s.d_out).sum();
        let mut hors_jeu = vec![false; n];
        for k in F1R + 1..n {
            let d = drift(&shapes, &last[F1R], &last[k], ARMS[k])?;
            let s = &shapes[d.shape];
            println!(
                "control 7: {:<7} per row, |Δ| / max(1, |y|) worst {:.2e} ({}, row {}: {} against f1r's {}), \
                 {} of {rows} rows over {TOL:e}; max |Δ| / max|y| over a shape {:.2e} (information)",
                ARMS[k],
                d.worst,
                s.name,
                d.row,
                last[k][d.shape][d.row],
                last[F1R][d.shape][d.row],
                d.over,
                d.worst_inf
            );
            if d.worst > TOL {
                hors_jeu[k] = true;
                println!("control 7: {} is HORS JEU — it does not compute what tv_f1r computes; its times are not printed", ARMS[k]);
            }
        }
        if hors_jeu.iter().all(|h| !h) {
            println!(
                "control 7: every variant equals f1r within {TOL:e} of each row, on all {rows} rows of the 7 shapes \
                 (V1 runs f1r's FMA chain and is expected at 0)"
            );
        }

        println!(
            "\n  {ROUNDS} rounds, {WARMUP} discarded; round r opens with arm r mod {n}; every difference formed ROUND BY ROUND\n"
        );
        for (i, name) in ARMS.iter().enumerate() {
            if hors_jeu[i] {
                println!("  {name:<8}      hors jeu (contrôle 7)");
                continue;
            }
            let (m, lo, hi) = median_range(&times[i]);
            println!("  {name:<8} {m:8.3} ms  [{lo:.3}–{hi:.3}]");
        }

        let diff = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x - y).collect() };
        let (m, lo, hi) = median_range(&diff(&times[1], &times[0]));
        println!("\n  S  = word − nullk   {m:8.3} ms  [{lo:.3}–{hi:.3}]   the F1 stream in our geometry");
        let (m, lo, hi) = median_range(&diff(&times[F1R], &times[1]));
        println!("  Du = f1r − word     {m:8.3} ms  [{lo:.3}–{hi:.3}]   table + arithmetic decode (against D(16 KiB) = 0.663 ms of the table floor)");
        let (m, lo, hi) = median_range(&diff(&times[F1R], &times[0]));
        println!("  T  = f1r − nullk    {m:8.3} ms  [{lo:.3}–{hi:.3}]   stream AND decode, read against B = {B_MS} ms (Planes14 − nullk, ANOTHER process)");
        let ratio: Vec<f64> = times[F1R].iter().zip(&times[0]).map(|(a, b)| a / b).collect();
        let (m, lo, hi) = median_range(&ratio);
        println!(
            "  t(f1r)/t(nullk)     {m:8.4}     [{lo:.4}–{hi:.4}]   against 5.103/2.306 = {B_RATIO:.2}, same reserve"
        );

        // The variants: Du_vk and T_vk as the prereg names them, read against
        // Du and T of THIS process, and the direct difference to f1r, round
        // by round — the variant's own gain, negative is faster.
        for k in F1R + 1..n {
            if hors_jeu[k] {
                continue;
            }
            let v = &ARMS[k][4..];
            println!();
            let (m, lo, hi) = median_range(&diff(&times[k], &times[1]));
            println!("  Du_{v} = {} − word     {m:8.3} ms  [{lo:.3}–{hi:.3}]   read against Du", ARMS[k]);
            let (m, lo, hi) = median_range(&diff(&times[k], &times[0]));
            println!(
                "  T_{v}  = {} − nullk    {m:8.3} ms  [{lo:.3}–{hi:.3}]   read against T of THIS process, B = {B_MS} ms as a scale",
                ARMS[k]
            );
            let (m, lo, hi) = median_range(&diff(&times[k], &times[F1R]));
            println!(
                "  {} − f1r         {m:8.3} ms  [{lo:.3}–{hi:.3}]   the variant's own gain, negative is faster",
                ARMS[k]
            );
        }
        println!("\n  WARNING: a FLOOR, not a cost. No gain scale, uniform labels rather than a model's,");
        println!("  and no Planes14 in this process. B is a scale to read T against, never a subtrahend:");
        println!("  no time here compares to a time from ANOTHER process.");
        Ok(())
    }
}
