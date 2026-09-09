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
    use llvq_artifact::runtime::ClassTable;
    use llvq_cuda::gpu::{Cuda, KernelSource};
    use llvq_search::fastdec::FastDecoder;
    use llvq_search::tetra::Tetra;
    /// Blocks staged per tile — `llvq_cuda::TILE_BLOCKS` unless
    /// `LLVQ_TILE_BLOCKS` overrides it.
    ///
    /// The tile is the one knob that trades shared memory for barrier count,
    /// and **no journal has ever varied it**. It is worth varying now: on
    /// 2026-09-08 the served Tetra decode measured 1.47× Planes14 on sm_120
    /// and 0.58× on sm_89, and the table floor of 2026-09-09 refuted both the
    /// capacity and the clock explanations — the table is *faster* on
    /// Blackwell at every footprint. What is left is the geometry, and the
    /// tile is what sets residency: `TILE_BLOCKS · 24 · 4` bytes per CTA.
    ///
    /// 32 is the floor. Below it `for (j = jlo + lane; j < jhi; j += 32)`
    /// leaves lanes idle, which is a different kernel, not a smaller tile.
    fn tile_blocks() -> usize {
        match std::env::var("LLVQ_TILE_BLOCKS") {
            Ok(v) => {
                let n: usize = v.parse().unwrap_or_else(|e| {
                    panic!("LLVQ_TILE_BLOCKS={v:?}: expected an integer ({e})")
                });
                assert!(
                    (32..=512).contains(&n) && n.is_power_of_two(),
                    "LLVQ_TILE_BLOCKS={n}: expected a power of two in 32..=512; \
                     below 32 the lane stride idles lanes and it is another kernel"
                );
                n
            }
            Err(_) => llvq_cuda::TILE_BLOCKS,
        }
    }
    use std::time::Instant;

    // Eighteen rounds, two discarded: sixteen kept, a multiple of EIGHT, so
    // with the rotating order every arm opens a round exactly twice.
    //
    // It was fourteen until 2026-09-09, written when there were six arms, and
    // the comment still claimed the property the constant no longer had: at
    // fourteen with eight arms, `f1r`/`v1`/`v2`/`v3` opened two rounds and
    // `nullk`/`word`/`v3g`/`planes14` one. That did **not** bias R — all three
    // of its terms were in the once-opening group, checked arm by arm — but it
    // did bias `v3g − f1r_v3`, which crosses the two groups, and the invariant
    // a comment asserts has to be the one the code holds.
    const ROUNDS: usize = 18;
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

    /// The eight arms, in the order the rotation walks them. Arms 2..6 are the
    /// four table arms: `tv_f1r` and its three variants, one argument list.
    /// Arms 6 and 7 were added on 2026-09-08 and are of a different nature —
    /// see [`V3G`] and [`PLANES`].
    const ARMS: [&str; 8] =
        ["nullk", "word", "f1r", "f1r_v1", "f1r_v2", "f1r_v3", "f1r_v3g", "planes14"];
    /// The kernel behind each arm, same index.
    const KERNELS: [&str; 8] = [
        "tv_nullk",
        "tv_f1r_word",
        "tv_f1r",
        "tv_f1r_v1",
        "tv_f1r_v2",
        "tv_f1r_v3",
        "tv_f1r_v3g",
        "tv_planes",
    ];
    /// Index of `f1r` in [`ARMS`]; the three variants are the arms after it.
    const F1R: usize = 2;
    /// `f1r_v3g`: the **served** Tetra decode — v3 plus the gain bit, the
    /// magnitude, the trio permutation and the origin. It does NOT compute
    /// what `tv_f1r` computes and is deliberately outside control 7: its
    /// arithmetic is pinned by `tests/tetra48_matches_rust.rs` on the dev
    /// machine and by control 8 on the card.
    const V3G: usize = 6;
    /// `planes14`: the served layout, **in this process**. Every F1 journal so
    /// far had to read `B = 2.797 ms` off another one and said so
    /// (`f1-rang-plancher-2026-09-05.txt:97`, *"B vient d'un autre processus"*).
    /// This arm is what ends that, and it is the arm the whole run exists for:
    /// a Tetra time and a Planes14 time formed round by round, same rounds,
    /// same rotation, same clock.
    const PLANES: usize = 7;
    /// The gain centroids both served arms are given, `planesbench`'s own, so
    /// the two layouts are scaled by the same two numbers.
    const GSCALE: [f32; 2] = [0.625, 1.375];
    /// Entries of the inverse-norm table of `llvq_tetra48.cuh`.
    const TETRA48_SHELLS: usize = 32;
    /// Planes14's uniform record stride, `llvq_planes.cuh`.
    const PLANES_STRIDE: usize = 14;
    /// `ClassRec` table, as `planesbench` builds it: 512 entries of 6 u32.
    const TABLE_ENTRIES: usize = 512;
    const REC_WORDS: usize = 6;

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
        /// `tv_f1r_v3g`: the table list plus `gscale` and `invnorm`.
        v3g: cudarc::driver::CudaFunction,
        /// `tv_planes`: a different stream, a different table, its own list.
        planes: cudarc::driver::CudaFunction,
    }

    /// The two constants the served Tetra decode adds, uploaded once.
    struct Tetra48 {
        gscale: cudarc::driver::CudaSlice<f32>,
        invnorm: cudarc::driver::CudaSlice<f32>,
    }

    /// Planes14's own table and gain pair.
    struct PlanesTab {
        tab: cudarc::driver::CudaSlice<u32>,
        gscale: cudarc::driver::CudaSlice<f32>,
    }

    /// `1/sqrt(16 m)`, entry 0 zero — the origin, reconstructed without a
    /// branch and without a division. `m <= 27` on this codebook, derived over
    /// the whole table by `llvq-bench/examples/tetrashell.rs`; 32 is that
    /// bound rounded up, and `llvq_tetra48.cuh` masks with it.
    fn invnorm_table() -> Vec<f32> {
        let mut t = vec![0.0f32; TETRA48_SHELLS];
        for (m, e) in t.iter_mut().enumerate().skip(1) {
            *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
        }
        t
    }

    /// u32 of one Planes14 stream copy for a shape: `14` bytes per block over
    /// `d_out · nblocks` blocks, plus the four-word read window of the last
    /// record (`llvq_planes.cuh`, "The read window").
    fn planes_words(d_out: u32, nblocks: u32) -> usize {
        let blocks = d_out as usize * nblocks as usize;
        (PLANES_STRIDE * blocks + 16).div_ceil(4)
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

    /// One timed round of `tv_f1r_v3g` — the table list with `gscale` and
    /// `invnorm` spliced in after `suffixes`, exactly where the kernel
    /// declares them.
    #[allow(clippy::too_many_arguments)]
    fn round_tetra(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        shapes: &mut [Shape],
        streams: &[Vec<Stream>],
        tab: &Tables,
        t48: &Tetra48,
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
                    .arg(&t48.gscale)
                    .arg(&t48.invnorm)
                    .arg(&s.rscale)
                    .arg(&s.tail)
                    .arg(&s.x)
                    .arg(&mut s.y)
                    .arg(&s.nblocks)
                    .arg(&s.tail_w);
                unsafe { b.launch(c) }.map_err(|e| format!("f1r_v3g/{}: {e}", s.name))?;
            }
        }
        cuda.sync()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// One timed round of `tv_planes` — the served layout, on its own stream.
    ///
    /// `tv_planes` addresses blocks **flat**, `row · nblocks + j` at a uniform
    /// 14-byte stride, where every F1 arm addresses them through a per-row u32
    /// stride. That difference is the layout, not the harness: it is what
    /// makes Planes14 read 14 bytes where Tetra reads 6, and it is the whole
    /// quantity the run exists to price.
    fn round_planes(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        shapes: &mut [Shape],
        streams: &[Vec<Stream>],
        pt: &PlanesTab,
        shared: u32,
    ) -> Result<f64, String> {
        let t = Instant::now();
        for layer in streams.iter() {
            for (s, st) in shapes.iter_mut().zip(layer.iter()) {
                let c = cfg(s, shared);
                let mut b = cuda.stream().launch_builder(f);
                b.arg(&st.words)
                    .arg(&pt.tab)
                    .arg(&pt.gscale)
                    .arg(&s.rscale)
                    .arg(&s.tail)
                    .arg(&s.x)
                    .arg(&mut s.y)
                    .arg(&s.nblocks)
                    .arg(&s.tail_w);
                unsafe { b.launch(c) }.map_err(|e| format!("planes14/{}: {e}", s.name))?;
            }
        }
        cuda.sync()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// Arm `k` of [`ARMS`], one round.
    #[allow(clippy::too_many_arguments)]
    fn run_arm(
        k: usize,
        cuda: &Cuda,
        fns: &Fns,
        shapes: &mut [Shape],
        streams: &[Vec<Stream>],
        pstreams: &[Vec<Stream>],
        tab: &Tables,
        t48: &Tetra48,
        pt: &PlanesTab,
        shared: u32,
    ) -> Result<f64, String> {
        match k {
            0 => round_null(cuda, &fns.nullk, shapes, shared),
            1 => round_word(cuda, &fns.word, shapes, streams, shared),
            V3G => round_tetra(cuda, &fns.v3g, shapes, streams, tab, t48, shared),
            PLANES => round_planes(cuda, &fns.planes, shapes, pstreams, pt, shared),
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

    /// Control 8: `tv_tetra48_dump` against `llvq_search::tetra` and the
    /// reconstruction the artifact reader uses.
    ///
    /// Control 1 pins `tv_f1r` against `llvq_bench::f1::rank::decode_word`,
    /// the bench yardstick. This one deliberately reaches for the **production**
    /// module instead — `Tetra::decode`, in natural order, plus the exact
    /// `centroids[g] / sqrt(16 m)` of `reconstruct_shape_gain` — because the
    /// permutation and the magnitude are precisely what the yardstick cannot
    /// see: it speaks trio order and applies no scale.
    ///
    /// The tolerance is relative and generous by design. The claim being
    /// checked is that the card runs the arithmetic the `clang++` harness
    /// already proved bit for bit (`tests/tetra48_matches_rust.rs`); a wrong
    /// permutation or an unsigned `__dp4a` misses by orders of magnitude, not
    /// by an ulp.
    #[allow(clippy::too_many_arguments)]
    fn dump_check_tetra(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        shapes: &[Shape],
        streams: &[Vec<Stream>],
        tab: &Tables,
        t48: &Tetra48,
        tetra: &Tetra,
        inv: &[f32],
    ) -> Result<usize, String> {
        let ndump = NDUMP;
        let mut out = cuda.zeros_f32(NDUMP as usize * DIM)?;
        let mut checked = 0usize;
        let mut worst = 0.0f64;
        for (s, st) in shapes.iter().zip(streams[0].iter()) {
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
                    .arg(&t48.gscale)
                    .arg(&t48.invnorm)
                    .arg(&mut out)
                    .arg(&s.nblocks)
                    .arg(&ndump);
                unsafe { b.launch(c) }.map_err(|e| format!("tetra48 dump/{}: {e}", s.name))?;
            }
            cuda.sync()?;
            let got = cuda.down_f32(&out)?;
            let nrows = NDUMP.div_ceil(s.nblocks);
            let bytes = host_bytes(st.seed, (nrows * s.stride_u32 + 2) as usize);
            for j in 0..NDUMP {
                let (row, jb) = (j / s.nblocks, j % s.nblocks);
                let word = host_word(&bytes, s.stride_u32, row, jb);
                let y = tetra.decode(word);
                let n2: u32 = y.iter().map(|&v| (v * v) as u32).sum();
                if !n2.is_multiple_of(16) {
                    return Err(format!(
                        "{}: word {word:#014x} decodes to ‖y‖² = {n2}, not a multiple of 16",
                        s.name
                    ));
                }
                let m = (n2 / 16) as usize;
                if m >= TETRA48_SHELLS {
                    return Err(format!(
                        "{}: word {word:#014x} lands on shell {m}, past the {TETRA48_SHELLS}-entry table",
                        s.name
                    ));
                }
                let g = ((word >> 47) & 1) as usize;
                let scale = GSCALE[g] as f64 * inv[m] as f64;
                for (k, &v) in y.iter().enumerate() {
                    let want = v as f64 * scale;
                    let g = got[j as usize * DIM + k] as f64;
                    let d = (g - want).abs() / want.abs().max(1.0);
                    worst = worst.max(d);
                    if d > TOL {
                        return Err(format!(
                            "{}: word {word:#014x} coordinate {k}: card {g}, llvq_search {want} (|Δ|rel {d:.2e})",
                            s.name
                        ));
                    }
                }
            }
            checked += NDUMP as usize;
        }
        println!(
            "control 8: {checked} blocks SERVED on the card equal llvq_search::tetra scaled by \
             reconstruct_shape_gain, every coordinate, worst |Δ|rel {worst:.2e}"
        );
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
            // The served Tetra decode and its arm — after v3, which it builds
            // on, and after matvec.cu, whose `warp_sum` it reduces with.
            "llvq_tetra48.cuh",
            "tetra48_v3g.cu",
            // Planes14, so the comparison is formed in ONE process.
            "llvq_planes.cuh",
            "planes.cu",
            "nullk.cu",
        ])?;
        let tb = tile_blocks();
        let defines = format!("#define TILE_BLOCKS {tb}u\n");
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
        // The whole residency card, not a third of it.
        //
        // Until 2026-09-09 this line printed the name, the SM count, the two
        // shared-per-BLOCK limits and the L2 — and none of the three numbers
        // that decide how many CTAs an SM actually holds. That is what the
        // two-card split turned on, and the fields were in `DeviceReport` the
        // whole time. The older L40S journals even printed the clock; this
        // bench had lost it.
        println!(
            "card: {} sm_{}{} · {} SM @ {:.0} MHz · L2 {:.0} MiB · mem {:.0} MHz × {} bit",
            dev.name,
            dev.compute_cap.0,
            dev.compute_cap.1,
            dev.sm_count,
            dev.clock_khz as f64 / 1000.0,
            dev.l2_bytes as f64 / (1024.0 * 1024.0),
            dev.mem_clock_khz as f64 / 1000.0,
            dev.mem_bus_bits
        );
        println!(
            "  per SM: {} threads, {} registers, {} B shared · per block: {} B default, {} opt-in",
            dev.max_threads_per_sm, dev.regs_per_sm, dev.shared_per_sm,
            dev.shared_per_block, dev.shared_per_block_optin
        );

        // Control 6: registers and local bytes of the six kernels, from the
        // function attributes. Printed, and the preregs' thresholds (≤ 64
        // registers, 0 local) flagged beside them; a spill does not stop the
        // run, it is the finding.
        // The tile in bytes: the one term of residency this bench controls.
        let tile = (tb * DIM * 4) as u32;
        println!(
            "\n  tile: {tb} blocks = {tile} B of shared per CTA{}",
            if tb == llvq_cuda::TILE_BLOCKS { String::new() }
            else { format!("  ⚠️ LLVQ_TILE_BLOCKS overrides the served {}", llvq_cuda::TILE_BLOCKS) }
        );
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
            // Residency beside the register count, because the register
            // count alone does not say what it costs. `occ::residency` is a
            // model and an upper bound (it rounds the granules down); it is
            // printed as one, and it is the same arithmetic `A3` sized its
            // persistent grid with.
            let ctas = llvq_cuda::occ::residency(
                r.num_regs as u32,
                THREADS,
                tile,
                dev.max_threads_per_sm as u32,
                dev.regs_per_sm as u32,
                dev.shared_per_sm as u32,
            );
            let warps = ctas * THREADS / 32;
            let full = 100.0 * (ctas * THREADS) as f64 / dev.max_threads_per_sm as f64;
            println!(
                "  {:<12} {:>3} registers, {} local bytes, sm_{} · {ctas} CTA/SM = {warps} warps \
                 = {full:.0}% of the SM (model){flag}",
                r.name, r.num_regs, r.local_bytes, r.binary_version
            );
        }

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

        // The served Tetra decode's two constants: 128 bytes and 8, against a
        // 16 KiB table already in flight. If `f1r_v3g` is slower than
        // `f1r_v3`, it is the six `__dp4a` and the two multiplies.
        let inv = invnorm_table();
        let t48 = Tetra48 { gscale: cuda.up_f32(&GSCALE)?, invnorm: cuda.up_f32(&inv)? };
        let tetra = Tetra::new();

        // Planes14's `ClassRec` table, built exactly as `planesbench` builds
        // it — same source, same normalisation, same origin convention — so
        // the arm in this process is the arm the published bench times.
        let fd = FastDecoder::new();
        let ctab = ClassTable::new(&fd, 1);
        assert!(
            (0..ctab.n_entries()).all(|e| ctab.record(e).len <= 5),
            "a class exceeds 5 levels: Planes14's three bit-planes are no longer enough"
        );
        let mut ptab = vec![0u32; TABLE_ENTRIES * REC_WORDS];
        for e in 0..TABLE_ENTRIES {
            ptab[e * REC_WORDS + REC_WORDS - 1] = 1;
        }
        for ci in 0..fd.n_classes() {
            let lv = fd.levels(ci);
            let norm = ((16 * lv.shell) as f64).sqrt();
            let base = (1 + ci) * REC_WORDS;
            for k in 0..lv.len {
                ptab[base + k] = ((lv.values[k] as f64 / norm) as f32).to_bits();
            }
            ptab[base + REC_WORDS - 1] = lv.len as u32;
        }
        let pt = PlanesTab { tab: cuda.up_u32(&ptab)?, gscale: cuda.up_f32(&GSCALE)? };
        println!(
            "planes14: ClassRec table {} B, {} classes, stride {PLANES_STRIDE} B/block",
            TABLE_ENTRIES * REC_WORDS * 4,
            fd.n_classes()
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
            v3g: cuda.func(KERNELS[V3G])?,
            planes: cuda.func(KERNELS[PLANES])?,
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
        // Planes14's own stream: 14 bytes a block against Tetra's 6, flat
        // over `d_out · nblocks` blocks rather than row-strided. Filled by the
        // SAME `f1r_fill`, so neither layout gets a friendlier generator; a
        // random 9-bit class field lands in 0..511, inside the 512-entry
        // table, and `planes_dot` selects values by a predicated tree over the
        // three plane bits — never `vals[idx]` — so its cost does not depend
        // on which class a block draws. That is what makes a random stream a
        // fair timing stream for this layout and it is asserted, not assumed,
        // by control 3.
        let mut pstreams: Vec<Vec<Stream>> = Vec::with_capacity(LAYERS);
        let mut ptotal_bytes = 0u64;
        for _ in 0..LAYERS {
            let mut layer = Vec::with_capacity(shapes.len());
            for s in shapes.iter() {
                let n = planes_words(s.d_out, s.nblocks);
                ptotal_bytes += n as u64 * 4;
                layer.push(fill(&cuda, &f_fill, &mut rng, n)?);
            }
            pstreams.push(layer);
        }
        cuda.sync()?;
        println!(
            "stream: {LAYERS} distinct copies × {} shapes, {:.3} GB of words on the device ({:.1}× the L2)",
            shapes.len(),
            total_bytes as f64 / 1e9,
            total_bytes as f64 / dev.l2_bytes as f64
        );
        println!(
            "stream planes14: {:.3} GB ({:.1}× the L2), {:.3}× the Tetra stream — the layout's whole claim",
            ptotal_bytes as f64 / 1e9,
            ptotal_bytes as f64 / dev.l2_bytes as f64,
            ptotal_bytes as f64 / total_bytes as f64
        );
        if ptotal_bytes < 4 * dev.l2_bytes as u64 {
            return Err(format!(
                "the planes14 stream is {ptotal_bytes} B against an L2 of {} B; below 4× it is a hit rate",
                dev.l2_bytes
            ));
        }
        if total_bytes < 4 * dev.l2_bytes as u64 {
            return Err(format!(
                "the word stream is {total_bytes} B against an L2 of {} B; below 4× it is a hit rate, not a stream",
                dev.l2_bytes
            ));
        }

        // Control 1 before any round.
        let checked = dump_check(&cuda, &f_dump, &shapes, &streams, &tab, &table, &tr)?;
        println!("control 1: {checked} blocks decoded on the card equal the Rust reference, every coordinate");
        let f_dump48 = cuda.func("tv_tetra48_dump")?;
        dump_check_tetra(&cuda, &f_dump48, &shapes, &streams, &tab, &t48, &tetra, &inv)?;

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
                let t = run_arm(k, &cuda, &fns, &mut shapes, &streams, &pstreams, &tab, &t48, &pt, tile)?;
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
        println!(
            "controls 2, 3: every output finite and written; f1r, f1r_v1, f1r_v2, f1r_v3, f1r_v3g, planes14 ≠ word ≠ nullk"
        );

        // Control 7: every variant against `tv_f1r`, every row of every
        // shape, on the last round's outputs. Every variant is measured and
        // printed; a variant over the tolerance is set aside (its times are
        // not printed) and the other arms are read — prereg §6, first row.
        let rows: u32 = shapes.iter().map(|s| s.d_out).sum();
        let mut hors_jeu = vec![false; n];
        for k in F1R + 1..V3G {
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
        for k in F1R + 1..V3G {
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
        // THE LINE THE RUN EXISTS FOR. Both arms served — the gain bit read,
        // the point normalised, the coordinates in natural order on one side;
        // the published layout on the other — in ONE process, over the same
        // rounds, with the difference formed round by round.
        if !hors_jeu[V3G] && !hors_jeu[PLANES] {
            println!("\n  ── the two served arms, IN THIS PROCESS ──");
            let (m, lo, hi) = median_range(&diff(&times[V3G], &times[PLANES]));
            println!(
                "  v3g − planes14      {m:8.3} ms  [{lo:.3}–{hi:.3}]   negative is Tetra faster, same clock"
            );
            let r: Vec<f64> = times[V3G].iter().zip(&times[PLANES]).map(|(a, b)| a / b).collect();
            let (m, lo, hi) = median_range(&r);
            println!("  t(v3g)/t(planes14)  {m:8.4}     [{lo:.4}–{hi:.4}]   the RAW ratio");
            let rn: Vec<f64> = times[V3G]
                .iter()
                .zip(&times[PLANES])
                .zip(&times[0])
                .map(|((a, b), z)| (a - z) / (b - z))
                .collect();
            let (m, lo, hi) = median_range(&rn);
            println!(
                "  (v3g−nullk)/(planes14−nullk) {m:6.4} [{lo:.4}–{hi:.4}]   the SAME-HEAD ratio, launch floor removed (rule 4)"
            );
            let (m, lo, hi) = median_range(&diff(&times[V3G], &times[V3G - 1]));
            println!(
                "  v3g − f1r_v3        {m:8.3} ms  [{lo:.3}–{hi:.3}]   what serving costs over the floor: \
                 the gain bit, the magnitude, the permutation, the origin"
            );
        }

        println!("\n  WARNING: still a FLOOR for the six original arms — uniform labels rather than a");
        println!("  model's, one launch shape rather than a token. What is NEW on 2026-09-08 is that");
        println!("  `v3g` and `planes14` are both SERVED decodes in ONE process, so their ratio is a");
        println!("  measurement and not a reading. B = {B_MS} ms stays what it always was: a scale from");
        println!("  ANOTHER process, printed beside T and never subtracted from anything here.");
        println!("  This bench does not give tok/s: 48% of a token is outside the matmuls (attribution");
        println!("  of 2026-08-05), so a kernel ratio compresses end to end. F1e is what measures that.");
        Ok(())
    }
}
