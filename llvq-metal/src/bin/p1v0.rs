//! **V0 of P1 — exactness, and not one nanosecond.**
//!
//! `proofs/preregistration-p1-2026-08-13.md` §3: *aucune milliseconde n'est
//! chronométrée avant que le décodeur soit prouvé*. This binary is that proof
//! for the two new arms. It prints no timing, no throughput and no derived
//! tok/s — deliberately. A number of that kind here would be read as a P1
//! result, and P1 has not run.
//!
//! ## The two arms do not have the same standard, and confusing them is the
//! ## trap this file exists to avoid
//!
//! | arm | etalon |
//! |---|---|
//! | `cascade_uniform` | the dot product of `FastDecoder::decode`'s point, in f64 — it decodes the **archive's** order, so equality is the requirement |
//! | `decode_walk` | the dot product `binomial_walk` (the CPU reference) gives **on the same ranks**, over levels re-derived from `FastDecoder::levels` — it decodes **its own** combinatorial order |
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

use llvq_core::{Golay, Leech, SplitMix64, DIM};
use llvq_metal::p1host::{
    binom_table, cascade_ends, cascade_records, div_table, pack_walk_block, walk_radix_table,
    walk_records, GpuCascadeRec, GpuDivTab, GpuWalkRec,
};
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};
use llvq_search::rankdec::binomial_walk;
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
// The two arms, each holding its constant tables on the device
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
        assert_eq!(n, gains.len(), "un gain par bloc");
        assert!(
            n > 0 && n.is_multiple_of(GROUP),
            "{n} blocs n'est pas un multiple de {GROUP} : le dernier threadgroup \
             lirait un `xs` non écrit"
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

/// The `marche-binomiale` arm: a packed 96-bit record in, one dot product out.
struct WalkArm {
    k: llvq_metal::Kernel,
    b_tab: metal::Buffer,
    b_binom: metal::Buffer,
    b_gs: metal::Buffer,
    b_x: metal::Buffer,
}

impl WalkArm {
    fn new(walk: &[GpuWalkRec], binom: &[u32], x: &[f32]) -> Result<Self, String> {
        let src = include_str!("../../shaders/binomial_walk.metal");
        let k = llvq_metal::Kernel::new(src, "decode_walk")?;
        Ok(Self {
            b_tab: k.buffer(walk),
            b_binom: k.buffer(binom),
            b_gs: k.buffer(&GSCALE),
            b_x: k.buffer(x),
            k,
        })
    }

    fn run(&self, words: &[u32]) -> Vec<f32> {
        assert_eq!(words.len() % 3, 0, "trois mots par bloc");
        let n = words.len() / 3;
        assert!(
            n > 0 && n.is_multiple_of(GROUP),
            "{n} blocs n'est pas un multiple de {GROUP}"
        );
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
}

// ---------------------------------------------------------------------------
// The two etalons
// ---------------------------------------------------------------------------

/// The archive decoder's own point, dotted in f64 — the cascade's etalon.
///
/// `Leech::shell_index` recomputes ‖p‖²/16 from the decoded point, so a wrong
/// `inv_norm` in the host table cannot cancel against itself here. The origin
/// has no shell, and its dot is zero rather than an error: `decode(0)` is a
/// legitimate point of the codebook and the fixture feeds it on purpose.
fn etalon_cascade(
    fd: &FastDecoder,
    idx: &[u64],
    gains: &[u8],
    x: &[f32],
) -> Result<Vec<(f64, f64)>, String> {
    let mut want = Vec::with_capacity(idx.len());
    for (b, &i) in idx.iter().enumerate() {
        let p = fd
            .decode(i)
            .ok_or_else(|| format!("index {i} hors de la boule"))?;
        let (mut dot, mut mag) = (0.0f64, 0.0f64);
        if let Some(m) = Leech::shell_index(&p).filter(|&m| m > 0) {
            let s = GSCALE[gains[b] as usize] as f64 / ((16 * m) as f64).sqrt();
            for (&v, &xi) in p.iter().zip(x) {
                let t = v as f64 * s * xi as f64;
                dot += t;
                mag += t.abs();
            }
        }
        want.push((dot, mag));
    }
    Ok(want)
}

/// The etalon's **own** reading of the codebook: counts and level values,
/// derived from [`FastDecoder::levels`] in f64 — never read back out of
/// [`GpuWalkRec`].
///
/// 🚨 **This is not redundancy, it is the point.** The first version of this
/// file built the walk's etalon from `rec.counts` and `rec.vals`, i.e. from the
/// very table it was supposed to check, and a mutation that gave every
/// shell-13 class the norm of shell 12 **survived**: the shader read the wrong
/// value, the CPU reference read the same wrong value, and the two agreed to
/// nine decimals. The cascade arm caught it at once, because its etalon is
/// `FastDecoder::decode`. That is exactly the "malentendu partagé" §3.1 of the
/// pre-registration asks the reference to be written independently of the
/// shader to avoid — and the walk arm had it.
///
/// What É2 exempts the walk from is comparing its *order* to the archive's. It
/// does not exempt it from checking that the level values are the codebook's:
/// that half has an independent source, and this is it.
///
/// Entry 0 is the origin — one kind, 24 slots, value zero — matching
/// [`walk_records`]'s own convention, restated rather than imported.
fn walk_levels(fd: &FastDecoder) -> Vec<([u8; MAX_KINDS], [f64; MAX_KINDS], usize)> {
    let mut out = Vec::with_capacity(fd.n_classes() + 1);
    let mut origin = [0u8; MAX_KINDS];
    origin[0] = DIM as u8;
    out.push((origin, [0.0f64; MAX_KINDS], 1));
    for ci in 0..fd.n_classes() {
        let lv = fd.levels(ci);
        let norm = ((16 * lv.shell) as f64).sqrt();
        let (mut counts, mut vals) = ([0u8; MAX_KINDS], [0.0f64; MAX_KINDS]);
        for j in 0..lv.len {
            counts[j] = lv.counts[j];
            vals[j] = lv.values[j] as f64 / norm;
        }
        out.push((counts, vals, lv.len));
    }
    out
}

/// Pack the walk's feed and compute its etalon in one pass — the CPU walk on
/// the very ranks the GPU is handed (É2).
///
/// Deliberately one function: a feed built in one place and an etalon built in
/// another can drift, and the drift would read as a decode divergence. The
/// *packing* reads `walk` — it is the record's own field widths that are being
/// tested — while the *etalon* reads `lev`, which owes it nothing.
fn walk_feed(
    walk: &[GpuWalkRec],
    lev: &[([u8; MAX_KINDS], [f64; MAX_KINDS], usize)],
    ids: &[u32],
    gains: &[u8],
    ranks: &[[u64; MAX_KINDS]],
    smasks: &[u32],
    x: &[f32],
) -> (Vec<u32>, Vec<(f64, f64)>) {
    let n = ids.len();
    assert_eq!(n, gains.len());
    assert_eq!(n, ranks.len());
    assert_eq!(n, smasks.len());
    assert_eq!(walk.len(), lev.len());
    let mut words = Vec::with_capacity(3 * n);
    let mut want = Vec::with_capacity(n);
    let mut kinds = [0u8; DIM];
    for b in 0..n {
        let id = ids[b] as usize;
        let rec: &GpuWalkRec = &walk[id];
        words.extend_from_slice(&pack_walk_block(
            rec,
            ids[b],
            gains[b] as u32,
            smasks[b],
            &ranks[b],
        ));

        let (counts, vals, k) = &lev[id];
        binomial_walk(&ranks[b], counts, *k, &mut kinds);
        let g = GSCALE[gains[b] as usize] as f64;
        let (mut dot, mut mag) = (0.0f64, 0.0f64);
        for (i, &kd) in kinds.iter().enumerate() {
            let v = vals[kd as usize];
            let v = if (smasks[b] >> i) & 1 == 1 { -v } else { v };
            let t = v * g * x[i] as f64;
            dot += t;
            mag += t.abs();
        }
        want.push((dot, mag));
    }
    (words, want)
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
) -> (Vec<u32>, Vec<[u64; MAX_KINDS]>, Vec<u32>) {
    let mut ids = Vec::new();
    let mut ranks = Vec::new();
    let mut smasks = Vec::new();
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
        for (r, s) in [
            (zero, 0u32),
            (max, full),
            (r1, (rng.next() as u32) & full),
            (r2, (rng.next() as u32) & full),
        ] {
            ids.push(id as u32);
            ranks.push(r);
            smasks.push(s);
        }
    }
    // Whole threadgroups, padded with the origin at rank zero — the one entry
    // whose dot is zero whatever the sign mask, so the padding cannot flatter
    // the worst error of a real entry.
    while !ids.len().is_multiple_of(GROUP) {
        ids.push(0);
        ranks.push([0u64; MAX_KINDS]);
        smasks.push(0);
    }
    (ids, ranks, smasks)
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
        "{name}: {n_bad} valeur(s) non finies rendues par le noyau — \
         ce n'est pas un écart, c'est un dispatch en échec"
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
        "  {name:<18} {} blocs vérifiés — pire écart {:.3e} absolu, {:.3e} relatif à Σ|w·x| \
         (bloc {}: GPU {:.9} / CPU {:.9})",
        got.len(),
        w.abs,
        w.rel,
        w.at,
        w.got,
        w.want
    );
    if w.bad > 0 {
        println!(
            "  {:<18} ROUGE — {} blocs hors tolérance sur {}",
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

    println!("P1 — V0 : exactitude seule. Aucun chronométrage dans ce binaire.\n");

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
    println!(
        "tables hôtes : {} classes cascade (88 o), {} entrées marche (52 o), \
         DivTab 600 o, binomiaux {}×{}, Golay {} mots",
        recs.len(),
        walk.len(),
        DIM + 1,
        DIM + 1,
        golay.codewords().len()
    );

    let x = xvec();
    let cascade = CascadeArm::new(&ends, &recs, &golay, &dv, &x)?;
    let walk_arm = WalkArm::new(&walk, &binom, &x)?;
    println!("GPU : {}", cascade.k.device_name());

    // =======================================================================
    // PASSE 1 — la fixture synthétique (§1.6)
    // =======================================================================
    println!("\n{}", "=".repeat(78));
    println!("PASSE 1 — fixture synthétique : ce qu'aucun tirage réel n'atteint");
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
        "la fixture ne contient pas l'origine, qui est la moitié de sa raison d'être"
    );
    for ci in 0..fd.n_classes() {
        assert!(
            fseen.contains(&Some(ci)),
            "classe {ci} absente de la fixture cascade"
        );
    }
    let s13 = (0..fd.n_classes())
        .filter(|&ci| fd.levels(ci).shell == 13)
        .count();
    println!(
        "\n{} blocs synthétiques — {} classes sur {} (TOUTES), origine comprise ; \
         dont {s13} classes de coquille 13, qu'aucun bloc du 4B publié n'habite",
        fidx.len(),
        fseen.len() - 1,
        fd.n_classes()
    );
    println!("  graine de la fixture : {FIXTURE_SEED:#x}");

    println!("\nbras CASCADE — étalon : produit scalaire f64 de FastDecoder::decode");
    let fc = verify(
        "cascade_uniform",
        &cascade.run(&fidx, &fgains),
        &etalon_cascade(&fd, &fidx, &fgains, &x)?,
    );

    let (fids, franks, fsmasks) = fixture_walk(&walk, &radices, &mut frng);
    let fwgains: Vec<u8> = (0..fids.len()).map(|i| (i % 2) as u8).collect();
    let wseen: BTreeSet<u32> = fids.iter().copied().collect();
    assert_eq!(
        wseen.len(),
        walk.len(),
        "la fixture marche doit toucher les {} entrées de la table",
        walk.len()
    );
    let (fwords, fwant) = walk_feed(&walk, &lev, &fids, &fwgains, &franks, &fsmasks, &x);
    println!(
        "\n{} blocs synthétiques — {} entrées de marche sur {} (TOUTES), origine comprise ; \
         rang 0, dernier rang, et deux tirages par entrée",
        fids.len(),
        wseen.len(),
        walk.len()
    );
    println!("bras MARCHE — étalon : binomial_walk (CPU) sur LES MÊMES rangs (É2), pas fd.decode");
    let fw = verify("decode_walk", &walk_arm.run(&fwords), &fwant);

    // =======================================================================
    // PASSE 2 — le tirage réel (§1.5)
    // =======================================================================
    println!("\n{}", "=".repeat(78));
    println!("PASSE 2 — tirage réel : le mélange de classes du modèle publié");
    println!("{}", "=".repeat(78));

    // House rule: a test that skips when its file is missing must FAIL. P1's
    // whole reason to exist is the real class mix (pre-registration §1.5);
    // there is no substitute draw and no degraded mode.
    let f = File::open(&path).map_err(|e| {
        format!(
            "l'archive scellée du 4B n'est pas sur cette machine : {path} ({e})\n\n\
             La fixture ci-dessus est passée, mais elle ne remplace RIEN : elle couvre les \
             entrées de table, pas la distribution.\n\
             P1 se mesure sur le mélange de classes RÉEL du modèle publié — il n'y a ni \
             substitut ni saut (pré-enregistrement §1.5).\n\
             La récupérer :\n\n    \
             hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-llvq.bin --local-dir .\n\n\
             puis la passer en argument :\n\n    \
             cargo run --release -p llvq-metal --bin p1v0 -- {n_req} ./qwen3-4b-llvq.bin\n\n\
             (cf. LAUNCH_ME.md pour le téléchargement, README.md pour l'invocation de \
             `bin/smoke` qui en produit une.)"
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
            "{}: les deux shaders de P1 codent en dur 1 bit de gain",
            m.name
        );
        let take = per_matrix.min(m.indices.len()).min(n_req - indices.len());
        indices.extend_from_slice(&m.indices[..take]);
        gains.extend(m.gains[..take].iter().map(|&g| {
            assert!(g < 2, "{}: rang de gain {g} avec 1 bit de gain", m.name);
            g as u8
        }));
    }

    // Truncate to a whole number of threadgroups: see GROUP.
    let n = indices.len() / GROUP * GROUP;
    assert!(
        n > 0,
        "{} blocs lus, moins d'un threadgroup de {GROUP}",
        indices.len()
    );
    indices.truncate(n);
    gains.truncate(n);

    println!(
        "\n{n} blocs réels ({} matrices, préfixes contigus) — {path}",
        h.matrices
    );

    let mut classes = Vec::with_capacity(n);
    for &idx in &indices {
        classes.push(
            fd.class_of(idx)
                .ok_or_else(|| format!("index {idx} hors de la boule"))?,
        );
    }
    let seen: BTreeSet<usize> = classes.iter().copied().collect();
    let origin = indices.iter().filter(|&&i| i == 0).count();
    println!(
        "  {} classes distinctes sur {} de la table, {origin} bloc(s) origine — \
         les entrées non touchées ici le sont par la passe 1",
        seen.len(),
        fd.n_classes()
    );
    println!("  graine du tirage de rangs (bras marche) : {SEED:#x}");

    println!("\nbras CASCADE — étalon : produit scalaire f64 de FastDecoder::decode");
    let wc = verify(
        "cascade_uniform",
        &cascade.run(&indices, &gains),
        &etalon_cascade(&fd, &indices, &gains, &x)?,
    );

    // No transcoder exists (amendment É2), so the feed is: for the REAL class
    // of each real block, ranks drawn uniformly in `walk_radices`. The etalon
    // is the CPU walk on those very ranks — NOT `fd.decode`, which decodes
    // another order entirely.
    let mut rng = SplitMix64::new(SEED);
    let mut ids = Vec::with_capacity(n);
    let mut ranks = Vec::with_capacity(n);
    let mut smasks = Vec::with_capacity(n);
    for &ci in &classes {
        let id = 1 + ci;
        let k = walk[id].k as usize;
        let mut r = [0u64; MAX_KINDS];
        for (j, rj) in r.iter_mut().enumerate().take(k) {
            let radix = radices[id][j];
            *rj = if radix > 1 { rng.next() % radix } else { 0 };
        }
        ids.push(id as u32);
        ranks.push(r);
        smasks.push((rng.next() as u32) & ((1 << DIM) - 1));
    }
    let (words, want_walk) = walk_feed(&walk, &lev, &ids, &gains, &ranks, &smasks, &x);

    println!("\nbras MARCHE — étalon : binomial_walk (CPU) sur LES MÊMES rangs (É2), pas fd.decode");
    let ww = verify("decode_walk", &walk_arm.run(&words), &want_walk);

    // ---- verdict ----------------------------------------------------------
    println!("\n{}", "-".repeat(78));
    let ok = fc.bad == 0 && fw.bad == 0 && wc.bad == 0 && ww.bad == 0;
    if ok {
        println!(
            "V0 VERT sur les deux bras, aux deux passes — fixture {} blocs (toutes les \
             {} entrées de table),\ntirage réel {n} blocs, tolérance {REL:.0e}·Σ|w·x|.\n\n\
             Ce que ça autorise : rien d'autre que d'écrire le banc. Restent dus avant \
             tout chronométrage —\n  · le sweep intégral des 150 681 600 blocs ;\n  \
             · pour la marche, l'aller-retour rang → arrangement → rang CÔTÉ GPU : il \
             existe côté CPU\n    (`the_walk_round_trips_on_its_own_bijection`) et ne borne \
             rien sur le shader.",
            fidx.len().max(fids.len()),
            walk.len()
        );
    } else {
        println!(
            "V0 ROUGE — fixture : cascade {} bloc(s) hors tolérance, marche {} ; \
             tirage réel : cascade {}, marche {}. Aucun chronométrage n'est autorisé.",
            fc.bad, fw.bad, wc.bad, ww.bad
        );
    }
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
