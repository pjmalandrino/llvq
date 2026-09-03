//! **V0 of P1 — exactness, and not one nanosecond.**
//!
//! `proofs/preregistration-p1-2026-08-13.md` §3: *no millisecond is timed
//! before the decoder is proven*. This binary is that proof for the new arms.
//! It prints no timing, no throughput and no derived tok/s — deliberately. A
//! number of that kind here would be read as a P1 result, and P1 has not run.
//!
//! ## The arms do not have the same standard, and confusing them is the trap
//! ## this file exists to avoid
//!
//! | arm | etalon |
//! |---|---|
//! | `cascade_uniform` | the dot product of `FastDecoder::decode`'s point, in f64 — it decodes the **archive's** order, so equality is the requirement |
//! | `decode_walk` | the dot product `binomial_walk` (the CPU reference) gives **on the same ranks**, over levels re-derived from `FastDecoder::levels` — it decodes **its own** combinatorial order |
//! | `decode_e1v` | the dot product of `cns_decode`'s point — the CNS is what the E1v stream numbers, and P5 C2 swept it against `FastDecoder::decode` over the 150,681,600 blocks of the 4B |
//!
//! Amendment É2 of the pre-registration settles the second: relating a
//! binomial walk's order to the archive's multiset-permutation order *is* the
//! CNS re-bijection, which is P5, which this bench gates. Checking the walk
//! against `fd.decode` would demand the transcoder its own verdict authorises
//! — a circularity, and a V0 it could never pass. So the walk arm is fed
//! ranks the host draws, and the GPU is required to agree with the Rust on
//! those ranks. It is a round trip across the CPU/GPU boundary on one
//! bijection, not a comparison to the archive.
//!
//! The **third** arm is here for a reason of *ordering* rather than of
//! coverage. Its own pre-registration (P1c) puts the draw's V0 inside
//! `bin/rankbench`, which is where the draw lives — but that bench refuses to
//! start until three documents are stamped, and a stamp is a one-way door. This
//! binary establishes the decode exact over the whole table *before* anyone is
//! asked to spend one. It runs the fixture only; the 2^24 blocks stay the
//! bench's job.
//!
//! ## Two passes, and the second cannot replace the first
//!
//! **Pass 1 — the synthetic fixture (pre-registration §1.6).** A real draw's
//! coverage is bounded by the *file*, not by the codebook: the whole published
//! 4B touches 286 of the table's 384 entries, holds **no origin block** and no
//! shell-13 class. 98 entries — the origin branch, the 82 classes of shell 13
//! and the 15 classes of the cap-12 ball the file never uses — are therefore
//! unreachable from any prefix of any matrix, however long. They are covered
//! here instead, on indices built from the class table itself: both ends and
//! the middle of **every** class, the origin, and for the walk every entry at
//! rank zero, at its last rank, and on random draws. The coverage is
//! **asserted**, not printed — a fixture that quietly missed shell 13 would
//! print a green line while covering nothing, which is the §5 pattern of the
//! dossier.
//!
//! **Pass 2 — the real draw.** Class *mix*: the proportions of the published
//! model, which no fixture reproduces and which is the whole reason P1 exists
//! (§1.5). It reaches fewer entries and it reaches them at their real
//! frequencies.
//!
//! Neither pass is redundant: the fixture covers what the file cannot reach,
//! the draw covers the distribution a fixture cannot imitate.
//!
//! Run: `cargo run --release -p llvq-metal --bin p1v0 [N] [model.llvq]`

use llvq_artifact::e1v::{transcode_e1v, E1vBlocks};
use llvq_core::{Golay, SplitMix64, DIM};
use llvq_metal::p1host::{
    binom_table, block_records, cascade_ends, cascade_records, div_table, dot_of, e1v_fixture,
    e1v_payload_bits, etalon_cascade, etalon_cns, walk_feed, walk_levels, walk_radix_table,
    walk_records, GpuBlockRec, GpuCascadeRec, GpuDivTab, GpuWalkRec, Levels, WalkFeed,
};
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};
use llvq_search::rankdec::{binomial_rank, binomial_walk, walk_cardinality};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::BufReader;

/// Blocks drawn by default from the archive. V0 is an exactness gate, not a
/// measurement: the pre-registration's `2^24` is a floor on a *timed* arm
/// (§1.2), and it buys nothing here. Overridable in argv.
const N_DEFAULT: usize = 1 << 20;

/// Threads per threadgroup. Both shaders stage the 24 activations with
/// `if (tid < DIM)` and a barrier, so a final threadgroup narrower than 24
/// would read an unwritten `xs`. Every draw is truncated to a multiple of this.
const GROUP: usize = 256;

/// Deterministic feed for the walk arm on the real draw — printed, so the run
/// is replayable.
const SEED: u64 = 0x0000_B1A0_5EED;

/// Deterministic feed for the fixture, kept **apart** from [`SEED`]: the two
/// passes must not share a stream, or lengthening one would silently change
/// what the other covers.
const FIXTURE_SEED: u64 = 0x0000_F1C7_5EED;

/// Largest class cardinality the bijection check enumerates in full, and the
/// total block budget it may spend doing so.
///
/// Both are arbitrary and both are printed: a class of cardinality 2,704,156
/// cannot be enumerated at any budget this binary is willing to hold, so the
/// bijection is established **where enumeration is affordable** and the round
/// trip carries the rest. Saying which is which is the point of printing the
/// count of entries covered.
const EXHAUSTIVE_CARD: u64 = 1 << 16;
const EXHAUSTIVE_BUDGET: usize = 1 << 21;

/// Gain centroids. **Not the artifact's**: the published matrices each carry
/// their own pair, and this bench mixes 252 of them. The gain scale is a
/// scalar applied *after* the decode, identical on both sides of every
/// comparison, so a single pair checks exactly what V0 is about — the decode —
/// and claims nothing about the centroids on disk. Same constant as
/// `decreal.rs:102`, for comparability.
const GSCALE: [f32; 2] = [0.9, 1.1];

/// Tolerance shape of `decreal.rs:129` — a floor OR a relative term, whichever
/// is larger — with the floor made **block-relative** rather than the absolute
/// `2e-3`. `cascade_uniform.metal` §4 asks for exactly that, and gives the
/// reason: an absolute floor that coarse cannot see a flipped sign bit on a
/// small dot, and "a fire alarm is not a guard".
const REL: f64 = 2e-3;

fn xvec() -> Vec<f32> {
    // Distinct magnitudes per slot, so a *permuted* arrangement moves the dot.
    // Same activation as `decreal`.
    (0..DIM).map(|i| 1.0 + i as f32 * 0.125).collect()
}

// ---------------------------------------------------------------------------
// The arms, each holding its constant tables on the device
// ---------------------------------------------------------------------------

/// The `cascade-uniformisée` arm: archive index in, one dot product out.
struct CascadeArm {
    k: llvq_metal::Kernel,
    b_ends: metal::Buffer,
    b_recs: metal::Buffer,
    b_golay: metal::Buffer,
    b_dv: metal::Buffer,
    b_gs: metal::Buffer,
    b_x: metal::Buffer,
}

impl CascadeArm {
    fn new(
        ends: &[u64],
        recs: &[GpuCascadeRec],
        golay: &Golay,
        dv: &GpuDivTab,
        x: &[f32],
    ) -> Result<Self, String> {
        let src = include_str!("../../shaders/cascade_uniform.metal");
        let k = llvq_metal::Kernel::new(src, "cascade_uniform")?;
        Ok(Self {
            b_ends: k.buffer(ends),
            b_recs: k.buffer(recs),
            b_golay: k.buffer(golay.codewords()),
            b_dv: k.buffer(std::slice::from_ref(dv)),
            b_gs: k.buffer(&GSCALE),
            b_x: k.buffer(x),
            k,
        })
    }

    fn run(&self, idx: &[u64], gains: &[u8]) -> Vec<f32> {
        let n = idx.len();
        assert_eq!(n, gains.len(), "one gain per block");
        assert!(
            n > 0 && n.is_multiple_of(GROUP),
            "{n} blocks is not a multiple of {GROUP}: the last threadgroup would \
             read an unwritten `xs`"
        );
        let b_idx = self.k.buffer(idx);
        let b_gain = self.k.buffer(gains);
        let b_out = self.k.empty::<f32>(n);
        self.k.dispatch(n as u64, GROUP as u64, |enc| {
            enc.set_buffer(0, Some(&b_idx), 0);
            enc.set_buffer(1, Some(&b_gain), 0);
            enc.set_buffer(2, Some(&self.b_ends), 0);
            enc.set_buffer(3, Some(&self.b_recs), 0);
            enc.set_buffer(4, Some(&self.b_golay), 0);
            enc.set_buffer(5, Some(&self.b_dv), 0);
            enc.set_buffer(6, Some(&self.b_gs), 0);
            enc.set_buffer(7, Some(&self.b_x), 0);
            enc.set_buffer(8, Some(&b_out), 0);
        });
        // SAFETY: the dispatch above completed and wrote n floats into b_out.
        unsafe { self.k.read(&b_out, n) }
    }
}

/// The `marche-binomiale` arm, and its instrumented twin.
///
/// Two entry points of **one** source: `decode_walk`, which writes one float
/// per block and is the arm the bench will time, and `walk_arrangement`
/// (`binomial_walk.metal` §11), which writes the 24 kind indices instead. The
/// twin exists because a dot product is a sum, and a sum is not injective: É2's
/// standard is `rank → arrangement → rank`, and the arrangement never leaves
/// the lane in the timed kernel. Adding a store to `decode_walk` to get at it
/// would change the thing being measured — defect no. 1 of the 2026-07-31 bench,
/// turned into a rule.
struct WalkArm {
    k: llvq_metal::Kernel,
    k_arr: llvq_metal::Kernel,
    b_tab: metal::Buffer,
    b_binom: metal::Buffer,
    b_gs: metal::Buffer,
    b_x: metal::Buffer,
    b_tab_arr: metal::Buffer,
    b_binom_arr: metal::Buffer,
}

impl WalkArm {
    fn new(walk: &[GpuWalkRec], binom: &[u32], x: &[f32]) -> Result<Self, String> {
        let src = include_str!("../../shaders/binomial_walk.metal");
        let k = llvq_metal::Kernel::new(src, "decode_walk")?;
        let k_arr = llvq_metal::Kernel::new(src, "walk_arrangement")?;
        Ok(Self {
            b_tab: k.buffer(walk),
            b_binom: k.buffer(binom),
            b_gs: k.buffer(&GSCALE),
            b_x: k.buffer(x),
            // Each `Kernel` owns its device and queue, so a buffer belongs to
            // the one that made it. Two pipelines, two copies of the tables —
            // 22 KB, and nothing here is timed.
            b_tab_arr: k_arr.buffer(walk),
            b_binom_arr: k_arr.buffer(binom),
            k,
            k_arr,
        })
    }

    fn nblocks(words: &[u32]) -> usize {
        assert_eq!(words.len() % 3, 0, "three words per block");
        let n = words.len() / 3;
        assert!(
            n > 0 && n.is_multiple_of(GROUP),
            "{n} blocks is not a multiple of {GROUP}"
        );
        n
    }

    fn run(&self, words: &[u32]) -> Vec<f32> {
        let n = Self::nblocks(words);
        let b_words = self.k.buffer(words);
        let b_out = self.k.empty::<f32>(n);
        self.k.dispatch(n as u64, GROUP as u64, |enc| {
            enc.set_buffer(0, Some(&b_words), 0);
            enc.set_buffer(1, Some(&self.b_tab), 0);
            enc.set_buffer(2, Some(&self.b_binom), 0);
            enc.set_buffer(3, Some(&self.b_gs), 0);
            enc.set_buffer(4, Some(&self.b_x), 0);
            enc.set_buffer(5, Some(&b_out), 0);
        });
        // SAFETY: the dispatch above completed and wrote n floats into b_out.
        unsafe { self.k.read(&b_out, n) }
    }

    /// The twin: `DIM` kind indices per block, slot-major.
    fn run_arrangement(&self, words: &[u32]) -> Vec<u8> {
        let n = Self::nblocks(words);
        let b_words = self.k_arr.buffer(words);
        let b_out = self.k_arr.empty::<u8>(n * DIM);
        self.k_arr.dispatch(n as u64, GROUP as u64, |enc| {
            enc.set_buffer(0, Some(&b_words), 0);
            enc.set_buffer(1, Some(&self.b_tab_arr), 0);
            enc.set_buffer(2, Some(&self.b_binom_arr), 0);
            enc.set_buffer(3, Some(&b_out), 0);
        });
        // SAFETY: the dispatch above completed and wrote n*DIM bytes into b_out.
        unsafe { self.k_arr.read(&b_out, n * DIM) }
    }
}

/// The `e1v-flux` arm of P1c: the **real** E1v stream in, one dot product out.
///
/// It holds the same three tables `marche-bloc` reads — block records,
/// binomials, Golay — plus the payload-width table the warp-scan sums. The
/// stream itself is not a table and is handed to [`Self::run`], because the
/// whole point of this arm is that where a record sits depends on what its
/// neighbours are.
struct E1vArm {
    k: llvq_metal::Kernel,
    b_tab: metal::Buffer,
    b_pay: metal::Buffer,
    b_binom: metal::Buffer,
    b_golay: metal::Buffer,
    b_gs: metal::Buffer,
    b_x: metal::Buffer,
}

impl E1vArm {
    fn new(
        brecs: &[GpuBlockRec],
        pay: &[u32],
        binom: &[u32],
        golay: &Golay,
        x: &[f32],
    ) -> Result<Self, String> {
        let src = include_str!("../../shaders/e1v_flux.metal");
        let k = llvq_metal::Kernel::new(src, "decode_e1v")?;
        Ok(Self {
            b_tab: k.buffer(brecs),
            b_pay: k.buffer(pay),
            b_binom: k.buffer(binom),
            b_golay: k.buffer(golay.codewords()),
            b_gs: k.buffer(&GSCALE),
            b_x: k.buffer(x),
            k,
        })
    }

    fn run(&self, stream: &E1vBlocks) -> Vec<f32> {
        let n = stream.n_blocks;
        assert!(
            n > 0 && n.is_multiple_of(GROUP),
            "{n} blocks is not a multiple of {GROUP}"
        );
        // Two words of padding so the three-word window of the last lane cannot
        // run off the end. Not part of the stream, and never counted as bytes
        // read (`thesis.rs:731`).
        let mut data = stream.data.clone();
        data.extend_from_slice(&[0u8; 8]);
        let b_data = self.k.buffer(&data);
        let b_bases = self.k.buffer(&stream.bases);
        let b_out = self.k.empty::<f32>(n);
        self.k.dispatch(n as u64, GROUP as u64, |enc| {
            enc.set_buffer(0, Some(&b_data), 0);
            enc.set_buffer(1, Some(&b_bases), 0);
            enc.set_buffer(2, Some(&self.b_tab), 0);
            enc.set_buffer(3, Some(&self.b_pay), 0);
            enc.set_buffer(4, Some(&self.b_binom), 0);
            enc.set_buffer(5, Some(&self.b_golay), 0);
            enc.set_buffer(6, Some(&self.b_gs), 0);
            enc.set_buffer(7, Some(&self.b_x), 0);
            enc.set_buffer(8, Some(&b_out), 0);
        });
        // SAFETY: the dispatch above completed and wrote n floats into b_out.
        unsafe { self.k.read(&b_out, n) }
    }
}

// ---------------------------------------------------------------------------
// The synthetic fixture — pre-registration §1.6
// ---------------------------------------------------------------------------

/// Cascade side: both ends and the middle of **every** class, one uniform draw
/// inside each, and the origin.
///
/// The two ends are not decoration. A class boundary off by one — an `ends`
/// table shifted, a `partition_point` convention flipped — sends a block to its
/// neighbour, and only the first and last index of a class can see it: any
/// interior index survives a shift of one. `cascade_ends` asserts the two host
/// accessors agree; this asserts the *kernel* agrees with them.
///
/// Shape borrowed from `llvq-artifact/tests/e1c_format.rs:81` (`fixture_indices`),
/// which is the repo's precedent for "every class at both ends, plus the
/// origin mid-stream".
fn fixture_cascade(fd: &FastDecoder, rng: &mut SplitMix64) -> Vec<u64> {
    let mut v: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        v.push(first);
        v.push(last);
        v.push(first + (last - first) / 2);
        v.push(first + rng.next() % (last - first + 1));
    }
    // The origin, which no real block is: the kernel takes it through the same
    // nine-step class search and zeroes the scale at the end
    // (`cascade_uniform.metal`, step 0). Padded with more of it, so the draw is
    // a whole number of threadgroups without inventing a distribution.
    while !v.len().is_multiple_of(GROUP) {
        v.push(0);
    }
    v
}

/// Walk side: **every** table entry — the origin included — at rank zero, at
/// its last rank, and on two random draws, with the sign mask at 0, all-ones
/// and random.
///
/// The last rank is the load-bearing one: it is the only draw that fills every
/// rank field to its declared width, so a `wbits` off by one, or a field packed
/// at the wrong offset, has nowhere to hide. Rank zero is its opposite — every
/// field empty — and a packer that dropped a field entirely would agree with
/// the reference on it.
fn fixture_walk(
    walk: &[GpuWalkRec],
    radices: &[[u64; MAX_KINDS]],
    rng: &mut SplitMix64,
) -> WalkFeed {
    let mut feed = WalkFeed::default();
    let full = (1u32 << DIM) - 1;
    for (id, rec) in walk.iter().enumerate() {
        let k = rec.k as usize;
        let rad = &radices[id];
        let zero = [0u64; MAX_KINDS];
        let mut max = [0u64; MAX_KINDS];
        for (j, m) in max.iter_mut().enumerate().take(k) {
            *m = rad[j] - 1;
        }
        // Four blocks per entry: the two extreme ranks with the two extreme
        // sign masks, then two interior draws.
        let rnd = |rng: &mut SplitMix64| {
            let mut r = [0u64; MAX_KINDS];
            for (j, rj) in r.iter_mut().enumerate().take(k) {
                *rj = if rad[j] > 1 { rng.next() % rad[j] } else { 0 };
            }
            r
        };
        let r1 = rnd(rng);
        let r2 = rnd(rng);
        for (i, (r, s)) in [
            (zero, 0u32),
            (max, full),
            (r1, (rng.next() as u32) & full),
            (r2, (rng.next() as u32) & full),
        ]
        .into_iter()
        .enumerate()
        {
            feed.push(id as u32, (i % 2) as u8, r, s);
        }
    }
    feed.pad(GROUP);
    feed
}

/// **Every rank of every class small enough to enumerate**, for the bijection
/// half of É2: pairwise distinct arrangements, and exactly as many as the
/// class holds.
///
/// This is the check a round trip cannot make. `rank → arrangement → rank`
/// closing says the map is injective *where it was probed*; it says nothing
/// about the map's image, and nothing at all about a rank never probed.
/// `the_walk_spans_exactly_the_class` makes it on the CPU by full enumeration;
/// this makes it on the GPU, over the same classes.
///
/// `walk_cardinality` is the walk's own product of radices, so enumerating
/// `0..cardinality` through the odometer visits each rank tuple exactly once by
/// construction — the count assertion downstream is about what the *shader*
/// produced, not about the loop that fed it.
fn exhaustive_walk(
    walk: &[GpuWalkRec],
    radices: &[[u64; MAX_KINDS]],
    max_card: u64,
    budget: usize,
) -> (WalkFeed, Vec<usize>) {
    let mut feed = WalkFeed::default();
    let mut picked = Vec::new();
    for (id, rec) in walk.iter().enumerate() {
        let k = rec.k as usize;
        let card = walk_cardinality(&rec.counts, k, DIM);
        // Entry 0 is skipped on purpose: it is what the feed pads with, and a
        // padded block is an extra arrangement for whichever id it carries.
        // Its own bijection is trivial (cardinality 1) and the fixture already
        // decodes it four times.
        if id == 0 || card > max_card || feed.len() + card as usize > budget {
            continue;
        }
        picked.push(id);
        let rad = &radices[id];
        let mut idx = [0u64; MAX_KINDS];
        loop {
            // The sign mask varies with the rank so the arrangements are not
            // all read through the same signs; it plays no part in the
            // bijection, which is about slots.
            feed.push(id as u32, 0, idx, (feed.len() as u32) & ((1 << DIM) - 1));
            let mut j = 0;
            while j < k {
                idx[j] += 1;
                if idx[j] < rad[j] {
                    break;
                }
                idx[j] = 0;
                j += 1;
            }
            if j >= k {
                break;
            }
        }
    }
    feed.pad(GROUP);
    (feed, picked)
}

// ---------------------------------------------------------------------------
// É2's standard, on the GPU's own arrangement
// ---------------------------------------------------------------------------

/// `rank → arrangement → rank`, on the arrangement the **GPU** produced.
///
/// Four assertions per block, and none of them subsumes another:
///
/// 1. **The round trip closes.** `binomial_rank` of the GPU's arrangement is
///    the rank the host packed. This is É2's standard, and until now it existed
///    only on the CPU: `p1v0` compared *dot products*, and a dot is a sum, so
///    two different arrangements can share one.
/// 2. **The arrangement realises the class's multiset.** The round trip cannot
///    see this: `binomial_rank` walks `0..k-1`, the final kind having no rank
///    of its own, so a walk that misplaced the last kind closes the trip
///    anyway. Exactly the mutation that survived five tests in `rankdec.rs`.
/// 3. **The GPU's arrangement equals the CPU's**, slot for slot — the exact
///    equality the dot could only approximate.
/// 4. **The dot rebuilt from the GPU's arrangement equals what `decode_walk`
///    wrote.** This is the only thread tying the instrumented twin to the arm
///    that will actually be timed (`binomial_walk.metal` §11). Without it, 1-3
///    would be a complete proof about a kernel nobody runs.
fn verify_arrangements(
    lev: &Levels,
    feed: &WalkFeed,
    arr: &[u8],
    dots: &[f32],
    x: &[f32],
    label: &str,
) -> usize {
    let n = feed.len();
    assert_eq!(arr.len(), n * DIM, "{label}: DIM bytes per block");
    assert_eq!(dots.len(), n, "{label}: one float per block");
    let mut bad = 0usize;
    let mut cpu = [0u8; DIM];
    let mut worst_dot = 0.0f64;
    for b in 0..n {
        let id = feed.ids[b] as usize;
        let (counts, vals, k) = &lev[id];
        let gpu: &[u8; DIM] = arr[b * DIM..(b + 1) * DIM].try_into().expect("DIM bytes");

        // 2 — the multiset, before anything else: a slot carrying a symbol
        // outside `0..k` would index `vals` out of range below.
        let mut ok = gpu.iter().all(|&s| (s as usize) < *k);
        if ok {
            for (j, &c) in counts.iter().enumerate().take(*k) {
                ok &= gpu.iter().filter(|&&s| s == j as u8).count() == c as usize;
            }
        }
        // 1 — the round trip.
        if ok {
            let back = binomial_rank(gpu, counts, *k);
            for (j, &want) in feed.ranks[b].iter().enumerate().take(k.saturating_sub(1)) {
                ok &= back[j] == want;
            }
        }
        // 3 — against the CPU walk, slot for slot.
        if ok {
            binomial_walk(&feed.ranks[b], counts, *k, &mut cpu);
            ok &= &cpu == gpu;
        }
        // 4 — the pin to the timed kernel.
        if ok {
            let (dot, mag) = dot_of(gpu, vals, feed.smasks[b], GSCALE[feed.gains[b] as usize] as f64, x);
            let d = (dots[b] as f64 - dot).abs();
            let rel = if mag > 0.0 { d / mag } else { d };
            worst_dot = worst_dot.max(rel);
            ok &= !d.is_nan() && d <= (REL * mag).max(REL * dot.abs());
        }
        if !ok {
            bad += 1;
        }
    }
    println!(
        "  {label:<18} {n} arrangements — round trip closed, multiset realised, \
         equal to the CPU slot by slot,\n  {:<18} pinned on decode_walk at {worst_dot:.3e} \
         relative to Σ|w·x|",
        ""
    );
    if bad > 0 {
        println!("  {:<18} RED — {bad} block(s) of {n}", "");
    }
    bad
}

/// The bijection half: over `0..cardinality` of a class, the arrangements the
/// GPU produces are **pairwise distinct** and there are **exactly as many as
/// the class holds**.
///
/// Injectivity plus a matching count is surjectivity onto a set of that size —
/// which, with `verify_arrangements`'s multiset check placing every arrangement
/// *inside* the class, is the bijection. Neither half alone is: a map that
/// collapsed two ranks and invented an arrangement outside the class would pass
/// a count, and a map onto half the class would pass distinctness.
fn verify_bijection(lev: &Levels, feed: &WalkFeed, arr: &[u8], picked: &[usize]) -> usize {
    use std::collections::HashSet;
    let mut bad = 0usize;
    for &id in picked {
        let (counts, _, k) = &lev[id];
        let card = walk_cardinality(counts, *k, DIM);
        let mut seen: HashSet<&[u8]> = HashSet::new();
        let mut n = 0u64;
        for (b, &fid) in feed.ids.iter().enumerate() {
            if fid as usize != id {
                continue;
            }
            n += 1;
            if !seen.insert(&arr[b * DIM..(b + 1) * DIM]) {
                bad += 1;
            }
        }
        if n != card || seen.len() as u64 != card {
            bad += 1;
            println!(
                "  {:<18} RED — entry {id}: {n} ranks supplied, {} distinct \
                 arrangements, cardinality {card}",
                "",
                seen.len()
            );
        }
    }
    bad
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Worst deviation seen over an arm, and where.
struct Worst {
    abs: f64,
    rel: f64,
    at: usize,
    got: f64,
    want: f64,
    bad: usize,
}

/// Compare one arm's output to its own etalon. `want[b] = (expected, scale)`,
/// where `scale` is `Σ|wᵢxᵢ|` — the magnitude the accumulation actually ran at,
/// which is what a tolerance on a dot product has to be relative to.
fn verify(name: &str, got: &[f32], want: &[(f64, f64)]) -> Worst {
    assert_eq!(got.len(), want.len());
    // 🚨 A NaN is not a discrepancy to measure, it is a failed dispatch, and it
    // must kill the run rather than flatter it. Every comparison below is
    // written so NaN counts against the arm — but a kernel that returned NaN
    // *everywhere* would still slip through a purely relative reading, because
    // `f64::max` ignores NaN and the worst error would stay at zero: the arm
    // would print a BETTER line than a correct one. This assertion is what
    // makes that impossible.
    let n_bad = got.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        n_bad, 0,
        "{name}: {n_bad} non-finite value(s) returned by the kernel — \
         this is not a deviation, it is a failed dispatch"
    );
    let mut w = Worst {
        abs: 0.0,
        rel: 0.0,
        at: 0,
        got: 0.0,
        want: 0.0,
        bad: 0,
    };
    for (b, (&g, &(exp, mag))) in got.iter().zip(want).enumerate() {
        let d = (g as f64 - exp).abs();
        let tol = (REL * mag).max(REL * exp.abs());
        let rel = if mag > 0.0 { d / mag } else { d };
        if rel > w.rel {
            w.rel = rel;
            w.at = b;
            w.got = g as f64;
            w.want = exp;
        }
        w.abs = w.abs.max(d);
        // NaN is named, not implied. `d > tol` alone is FALSE when `d` is
        // NaN, so the block would escape the count entirely; writing the NaN
        // case out says so, and reads better than the `!(d <= tol)` that
        // clippy rightly calls hard to refactor.
        if d.is_nan() || d > tol {
            w.bad += 1;
        }
    }
    println!(
        "  {name:<18} {} blocks verified — worst deviation {:.3e} absolute, {:.3e} relative \
         to Σ|w·x| (block {}: GPU {:.9} / CPU {:.9})",
        got.len(),
        w.abs,
        w.rel,
        w.at,
        w.got,
        w.want
    );
    if w.bad > 0 {
        println!(
            "  {:<18} RED — {} blocks out of tolerance of {}",
            "",
            w.bad,
            got.len()
        );
    }
    w
}

fn main() -> Result<(), String> {
    let mut n_req = N_DEFAULT;
    let mut path = format!(
        "{}/llvq-q4b.llvq",
        std::env::var("HOME").map_err(|_| "$HOME is not set, so there is no default path")?
    );
    for a in std::env::args().skip(1) {
        match a.parse::<usize>() {
            Ok(v) => n_req = v,
            Err(_) => path = a,
        }
    }

    println!("P1 — V0: exactness only. No timing in this binary.\n");

    // ---- host tables ------------------------------------------------------
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let ends = cascade_ends(&fd);
    let recs = cascade_records(&fd, &golay);
    let dv = div_table();
    let walk = walk_records(&fd);
    let radices = walk_radix_table(&walk);
    let lev = walk_levels(&fd);
    let binom = binom_table();
    let brecs = block_records(&fd, &golay);
    println!(
        "host tables: {} cascade classes (88 B), {} walk entries (52 B), \
         DivTab 600 B, binomials {}×{}, Golay {} words",
        recs.len(),
        walk.len(),
        DIM + 1,
        DIM + 1,
        golay.codewords().len()
    );

    let x = xvec();
    let cascade = CascadeArm::new(&ends, &recs, &golay, &dv, &x)?;
    let walk_arm = WalkArm::new(&walk, &binom, &x)?;
    println!("GPU: {}", cascade.k.device_name());

    // =======================================================================
    // PASS 1 — the synthetic fixture (§1.6)
    // =======================================================================
    println!("\n{}", "=".repeat(78));
    println!("PASS 1 — synthetic fixture: what no real draw reaches");
    println!("{}", "=".repeat(78));

    let mut frng = SplitMix64::new(FIXTURE_SEED);
    let fidx = fixture_cascade(&fd, &mut frng);
    let fgains: Vec<u8> = (0..fidx.len()).map(|i| (i % 2) as u8).collect();

    // 🚨 The coverage is ASSERTED. This fixture's entire reason to exist is the
    // 98 table entries a real draw cannot reach; a fixture that missed them
    // would print exactly the same green line as one that did not.
    let fseen: BTreeSet<Option<usize>> = fidx.iter().map(|&i| fd.class_of(i)).collect();
    assert!(
        fseen.contains(&None),
        "the fixture does not contain the origin, which is half its reason to exist"
    );
    for ci in 0..fd.n_classes() {
        assert!(
            fseen.contains(&Some(ci)),
            "class {ci} missing from the cascade fixture"
        );
    }
    let s13 = (0..fd.n_classes())
        .filter(|&ci| fd.levels(ci).shell == 13)
        .count();
    println!(
        "\n{} synthetic blocks — {} classes of {} (ALL), origin included; \
         of which {s13} shell-13 classes, which no block of the published 4B inhabits",
        fidx.len(),
        fseen.len() - 1,
        fd.n_classes()
    );
    println!("  fixture seed: {FIXTURE_SEED:#x}");

    println!("\nCASCADE arm — etalon: f64 dot product of FastDecoder::decode");
    let fc = verify(
        "cascade_uniform",
        &cascade.run(&fidx, &fgains),
        &etalon_cascade(&fd, &fidx, &fgains, &GSCALE, &x)?,
    );

    let ffeed = fixture_walk(&walk, &radices, &mut frng);
    let wseen: BTreeSet<u32> = ffeed.ids.iter().copied().collect();
    assert_eq!(
        wseen.len(),
        walk.len(),
        "the walk fixture must touch the {} entries of the table",
        walk.len()
    );
    let (fwords, fwant) = walk_feed(&walk, &lev, &ffeed, &GSCALE, &x);
    println!(
        "\n{} synthetic blocks — {} walk entries of {} (ALL), origin included; \
         rank 0, last rank, and two draws per entry",
        ffeed.len(),
        wseen.len(),
        walk.len()
    );
    println!("WALK arm — etalon: binomial_walk (CPU) on THE SAME ranks (É2), not fd.decode");
    let fdots = walk_arm.run(&fwords);
    let fw = verify("decode_walk", &fdots, &fwant);
    let fwa = verify_arrangements(
        &lev,
        &ffeed,
        &walk_arm.run_arrangement(&fwords),
        &fdots,
        &x,
        "walk_arrangement",
    );

    // =======================================================================
    // PASS 1ter — the E1v stream, on the same fixture (P1c)
    // =======================================================================
    //
    // This arm decodes nothing that `marche-bloc` does not decode — its body is
    // the same text, byte for byte, and a test requires it. What it adds is the
    // ADDRESSING: base word, fixed-stride header, SIMD prefix sum over the 32
    // widths of the group, three-word window. So addressing is what this pass
    // verifies, and the fixture is built for it: the origin (whose payload is
    // EMPTY, so it contributes nothing to the prefix sum and whose table entry
    // is 0 instead of 1+ci) and a whole group of the widest class, which is the
    // largest prefix sum the addressing can face.
    println!("\n{}", "-".repeat(78));
    let pay = e1v_payload_bits(&fd, &golay, &brecs);
    let (e1x, e1g) = e1v_fixture(&fd, &pay, GROUP);
    let e1seen: BTreeSet<Option<usize>> = e1x.iter().map(|&i| fd.class_of(i)).collect();
    assert!(
        e1seen.contains(&None),
        "the E1v fixture does not contain the origin, which the 4B does not carry and \
         which no draw will therefore ever reach"
    );
    for ci in 0..fd.n_classes() {
        assert!(e1seen.contains(&Some(ci)), "class {ci} missing from the E1v fixture");
    }
    let stream = transcode_e1v(
        &fd,
        &golay,
        &e1x,
        &e1g.iter().map(|&g| u32::from(g)).collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "E1v arm — etalon: f64 dot product of cns_decode (P1c §2), not fd.decode\n  \
         {} blocks, {} groups, {} B of stream + {} B of base words, worst-case payload \
         width {} bits",
        stream.n_blocks,
        stream.bases.len(),
        stream.data.len(),
        4 * stream.bases.len(),
        pay.iter().max().expect("the table is not empty")
    );
    let e1v_arm = E1vArm::new(&brecs, &pay, &binom, &golay, &x)?;
    let fe = verify(
        "decode_e1v",
        &e1v_arm.run(&stream),
        &etalon_cns(&fd, &golay, &e1x, &e1g, &GSCALE, &x)?,
    );

    // =======================================================================
    // PASS 1bis — the bijection, by exhaustive enumeration (É2)
    // =======================================================================
    println!("\n{}", "-".repeat(78));
    let (efeed, picked) = exhaustive_walk(&walk, &radices, EXHAUSTIVE_CARD, EXHAUSTIVE_BUDGET);
    let (ewords, _) = walk_feed(&walk, &lev, &efeed, &GSCALE, &x);
    let earr = walk_arm.run_arrangement(&ewords);
    let ecard: u64 = picked
        .iter()
        .map(|&id| walk_cardinality(&lev[id].0, lev[id].2, DIM))
        .sum();
    println!(
        "bijection — {} entries of cardinality ≤ {EXHAUSTIVE_CARD} enumerated IN FULL: \
         {ecard} ranks,\n  {} blocks with the origin padding, of the {} entries of the \
         table. The others are out\n  of reach of an enumeration and stay bounded by \
         the round trip alone.",
        picked.len(),
        efeed.len(),
        walk.len()
    );
    let eb = verify_bijection(&lev, &efeed, &earr, &picked);
    let ea = verify_arrangements(&lev, &efeed, &earr, &walk_arm.run(&ewords), &x, "exhaustive");
    if eb == 0 {
        println!(
            "  {:<18} pairwise distinct arrangements, count equal to the cardinality",
            ""
        );
    }

    // =======================================================================
    // PASS 2 — the real draw (§1.5)
    // =======================================================================
    println!("\n{}", "=".repeat(78));
    println!("PASS 2 — real draw: the class mix of the published model");
    println!("{}", "=".repeat(78));

    // House rule: a test that skips when its file is missing must FAIL. P1's
    // whole reason to exist is the real class mix (pre-registration §1.5);
    // there is no substitute draw and no degraded mode.
    let f = File::open(&path).map_err(|e| {
        format!(
            "the sealed 4B archive is not on this machine: {path} ({e})\n\n\
             The fixture above passed, but it replaces NOTHING: it covers the table \
             entries, not the distribution.\n\
             P1 is measured on the REAL class mix of the published model — there is no \
             substitute and no skip (pre-registration §1.5).\n\
             To fetch it:\n\n    \
             hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-llvq.bin --local-dir .\n\n\
             then pass it as an argument:\n\n    \
             cargo run --release -p llvq-metal --bin p1v0 -- {n_req} ./qwen3-4b-llvq.bin\n\n\
             (see LAUNCH_ME.md for the download, README.md for the invocation of \
             `bin/smoke` that produces one.)"
        )
    })?;

    let mut r = BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let per_matrix = n_req.div_ceil(h.matrices as usize).max(1);

    let mut indices: Vec<u64> = Vec::with_capacity(n_req);
    let mut gains: Vec<u8> = Vec::with_capacity(n_req);
    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).map_err(|e| e.to_string())?;
        // Both shaders hardcode ONE gain bit, like every other MSL of this
        // repo. Asserted BEFORE any work: with two, every field after the gain
        // is shifted by one and the walk arm would decode the wrong classes
        // while looking perfectly healthy.
        assert_eq!(
            m.centroids.len().next_power_of_two().trailing_zeros(),
            1,
            "{}: both P1 shaders hardcode 1 gain bit",
            m.name
        );
        let take = per_matrix.min(m.indices.len()).min(n_req - indices.len());
        indices.extend_from_slice(&m.indices[..take]);
        gains.extend(m.gains[..take].iter().map(|&g| {
            assert!(g < 2, "{}: gain rank {g} with 1 gain bit", m.name);
            g as u8
        }));
    }

    // Truncate to a whole number of threadgroups: see GROUP.
    let n = indices.len() / GROUP * GROUP;
    assert!(
        n > 0,
        "{} blocks read, less than one threadgroup of {GROUP}",
        indices.len()
    );
    indices.truncate(n);
    gains.truncate(n);

    println!(
        "\n{n} real blocks ({} matrices, contiguous prefixes) — {path}",
        h.matrices
    );

    let mut classes = Vec::with_capacity(n);
    for &idx in &indices {
        classes.push(
            fd.class_of(idx)
                .ok_or_else(|| format!("index {idx} outside the ball"))?,
        );
    }
    let seen: BTreeSet<usize> = classes.iter().copied().collect();
    let origin = indices.iter().filter(|&&i| i == 0).count();
    println!(
        "  {} distinct classes of {} in the table, {origin} origin block(s) — \
         the entries not touched here are covered by pass 1",
        seen.len(),
        fd.n_classes()
    );
    println!("  seed of the rank draw (walk arm): {SEED:#x}");

    println!("\nCASCADE arm — etalon: f64 dot product of FastDecoder::decode");
    let wc = verify(
        "cascade_uniform",
        &cascade.run(&indices, &gains),
        &etalon_cascade(&fd, &indices, &gains, &GSCALE, &x)?,
    );

    // No transcoder exists (amendment É2), so the feed is: for the REAL class
    // of each real block, ranks drawn uniformly in `walk_radices`. The etalon
    // is the CPU walk on those very ranks — NOT `fd.decode`, which decodes
    // another order entirely.
    let mut rng = SplitMix64::new(SEED);
    let mut rfeed = WalkFeed::default();
    for (b, &ci) in classes.iter().enumerate() {
        let id = 1 + ci;
        let k = walk[id].k as usize;
        let mut r = [0u64; MAX_KINDS];
        for (j, rj) in r.iter_mut().enumerate().take(k) {
            let radix = radices[id][j];
            *rj = if radix > 1 { rng.next() % radix } else { 0 };
        }
        rfeed.push(id as u32, gains[b], r, (rng.next() as u32) & ((1 << DIM) - 1));
    }
    let (words, want_walk) = walk_feed(&walk, &lev, &rfeed, &GSCALE, &x);

    println!("\nWALK arm — etalon: binomial_walk (CPU) on THE SAME ranks (É2), not fd.decode");
    let dots = walk_arm.run(&words);
    let ww = verify("decode_walk", &dots, &want_walk);
    let wwa = verify_arrangements(
        &lev,
        &rfeed,
        &walk_arm.run_arrangement(&words),
        &dots,
        &x,
        "walk_arrangement",
    );

    // ---- verdict ----------------------------------------------------------
    println!("\n{}", "-".repeat(78));
    let ok = fc.bad == 0 && fw.bad == 0 && fwa == 0 && fe.bad == 0 && ea == 0 && eb == 0
        && wc.bad == 0
        && ww.bad == 0
        && wwa == 0;
    if ok {
        println!(
            "V0 GREEN — the three arms, plus the bijection.\n\n  \
             cascade : {} fixture blocks (all {} classes, origin included) \
             + {n} real blocks,\n            point for point against FastDecoder::decode\n  \
             walk    : {} fixture blocks + {n} real blocks, round trip rank → \
             arrangement → rank\n            CLOSED ON THE GPU'S ARRANGEMENT, multiset \
             realised, equal to the CPU slot by slot\n  E1v     : {} fixture blocks — all \
             the classes, the origin the 4B does not carry,\n            and a whole group \
             of the widest class, against cns_decode\n  bijection : {} entries enumerated \
             in full ({ecard} ranks), distinct arrangements\n\n\
             Tolerance {REL:.0e}·Σ|w·x|. What it authorises: writing the bench, and \
             nothing else.\nWARNING: the E1v arm has NO real-draw pass here; its \
             pre-registration (P1c §5.2)\n   puts it in `bin/rankbench`, on the 2^24 \
             blocks of the draw. What is established here\n   is exactness on the table \
             entries — the origin included, which no draw will reach.\nWARNING: the \
             arrangement comes out of the instrumented twin `walk_arrangement`,\n   not \
             out of `decode_walk`, which emits only a sum — the link between the two is\n   \
             the dot product, not the arrangement (binomial_walk.metal §11).",
            fidx.len(),
            fd.n_classes(),
            ffeed.len(),
            e1x.len(),
            picked.len(),
        );
    } else {
        println!(
            "V0 RED — fixture: cascade {} block(s) out of tolerance, walk {}, \
             arrangements {fwa}, E1v {};\n  bijection: {eb} entries, arrangements {ea}; \
             real draw: cascade {}, walk {}, arrangements {wwa}.\n\
             No timing is authorised.",
            fc.bad, fw.bad, fe.bad, wc.bad, ww.bad
        );
    }
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
