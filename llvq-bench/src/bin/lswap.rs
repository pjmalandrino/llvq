//! Swap every 5-level block of a real artifact for its best L ≤ 4 direction —
//! the "Job B" pre-pass of the CUDA format experiment.
//!
//! `lcap` priced the L ≤ 4 cap on a Gaussian source (−0.26 pt of retention);
//! this tool converts that figure into something measurable in perplexity, by
//! producing a *file*: a copy of the sealed artifact in which each block whose
//! class holds 5 distinct magnitudes is re-encoded under
//! `BallSearcher::with_level_cap(4)` — **direction only**. The gain bit codes
//! the block norm and is copied unchanged, so the reconstructed magnitude is
//! identical; the only thing that moves is the direction, by the angle
//! reported below. Everything else — headers, row scales, centroids, tails,
//! every L ≤ 4 block, and the trailing raw/blob sections — is copied
//! bit-identically, and a control reload verifies exactly that.
//!
//! Run: `cargo run --release -p llvq-bench --bin lswap -- in.llvq out.llvq`

use llvq_artifact::{
    read_header, read_matrix_raw, write_matrix_raw, RawMatrix, MAGIC, MAGIC_V1, MAGIC_V2,
};
use llvq_core::{Point, DIM};
use llvq_search::fastdec::{FastDecoder, MAX_LEVELS};
use llvq_search::generic::BallSearcher;
use llvq_search::index::Indexer;
use llvq_search::Searcher;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

/// The cap under study: at most 4 distinct magnitudes per block, zero
/// included — the knob whose runtime stride is `9 + 1 + 24·4 = 106 b = 14 o`.
const LEVEL_CAP: usize = 4;
/// The published file is `leech1c12`: ball m ≤ 12, 47 index bits. The swap
/// searches the same ball, so every new index fits the stream's width.
const SHELL_CAP: u32 = 12;
/// L = 5 blocks of the published 4B artifact — the exhaustive count of
/// `bin/rtbits` (docs/mesures/k1c-rtbits-2026-08-05.txt). The swap must touch
/// exactly these and nothing else.
const EXPECTED_SWAPS: u64 = 5_096_688;

/// Distinct |values| of a lattice point, zero included — the definition
/// `n_levels()` uses, so this check and the searcher's filter agree.
fn distinct_magnitudes(p: &Point) -> usize {
    let mut seen = [false; 16];
    for &v in p {
        seen[v.unsigned_abs() as usize] = true;
    }
    seen.iter().filter(|&&b| b).count()
}

#[derive(Default)]
struct Stats {
    swapped: u64,
    /// cos(old point, new point) per swapped block — the angular price paid.
    cosines: Vec<f64>,
    /// Blocks by level count of their class; the origin counts as 1 level,
    /// mirroring `bin/rtbits` so the histograms can be compared line by line.
    hist: [u64; MAX_LEVELS + 1],
    origin: u64,
}

impl Stats {
    fn absorb(&mut self, other: Stats) {
        self.swapped += other.swapped;
        self.cosines.extend(other.cosines);
        for (a, b) in self.hist.iter_mut().zip(other.hist) {
            *a += b;
        }
        self.origin += other.origin;
    }
}

/// Re-encode the L = 5 blocks of one slice of indices, in place.
fn swap_slice(slice: &mut [u64], s: &Searcher, fd: &FastDecoder, ix: &Indexer) -> Stats {
    let mut ball = BallSearcher::with_level_cap(LEVEL_CAP);
    ball.set_shell_cap(SHELL_CAP);
    let mut st = Stats::default();
    for idx in slice.iter_mut() {
        let Some(ci) = fd.class_of(*idx) else {
            assert_eq!(*idx, 0, "non-origin index outside the ball");
            st.origin += 1;
            st.hist[1] += 1;
            continue;
        };
        let l = fd.levels(ci).len;
        st.hist[l] += 1;
        if l <= LEVEL_CAP {
            continue;
        }
        let p = fd.decode(*idx).expect("class_of accepted the index");
        let x: [f64; DIM] = core::array::from_fn(|i| p[i] as f64);
        let f = ball.nearest_angular(s, &x);
        // The three properties the experiment rests on, asserted per block:
        // the new point is a capped-codebook member, it round-trips through
        // the v1 index, and the record's gain (the norm code) is untouched —
        // the last one by construction: this loop never reads `gains`.
        assert!(
            (2..=SHELL_CAP).contains(&f.shell),
            "new point outside the m ≤ {SHELL_CAP} ball"
        );
        assert!(
            distinct_magnitudes(&f.point) <= LEVEL_CAP,
            "capped searcher returned a {}‑level point",
            distinct_magnitudes(&f.point)
        );
        let q = ix.encode(&f.point).expect("searched point is a codebook member");
        assert_eq!(
            fd.decode(q).as_ref(),
            Some(&f.point),
            "index round-trip diverged"
        );
        st.cosines.push(f.cosine(&x));
        *idx = q;
        st.swapped += 1;
    }
    st
}

/// Compare source and produced file record by record — the control reload.
///
/// Every field of every matrix must be bit-identical except the indices of
/// the swapped blocks, which must go from a 5-level class to a ≤ 4-level one;
/// the trailing raw/blob bytes must match exactly. Returns the number of
/// differing indices.
fn verify(src: &str, dst: &str, fd: &FastDecoder) -> u64 {
    let mut a = BufReader::with_capacity(1 << 20, File::open(src).expect("reopen source"));
    let mut b = BufReader::with_capacity(1 << 20, File::open(dst).expect("reopen output"));
    let ha = read_header(&mut a).expect("source header");
    let hb = read_header(&mut b).expect("output header");
    assert_eq!(ha.version, hb.version, "header version changed");
    assert_eq!(ha.matrices, hb.matrices, "matrix count changed");

    let mut diffs = 0u64;
    for _ in 0..ha.matrices {
        let ma = read_matrix_raw(&mut a).expect("source matrix");
        let mb = read_matrix_raw(&mut b).expect("output matrix");
        assert_eq!(ma.name, mb.name);
        assert_eq!((ma.d_out, ma.d_in), (mb.d_out, mb.d_in), "{}", ma.name);
        assert_eq!(ma.shell_cap, mb.shell_cap, "{}", ma.name);
        assert_eq!(ma.rotation_seed, mb.rotation_seed, "{}", ma.name);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&ma.centroids), bits(&mb.centroids), "{}", ma.name);
        assert_eq!(bits(&ma.row_scales), bits(&mb.row_scales), "{}", ma.name);
        assert_eq!(bits(&ma.tail), bits(&mb.tail), "{}: tail", ma.name);
        assert_eq!(ma.gains, mb.gains, "{}: a gain moved", ma.name);
        for (i, (&ia, &ib)) in ma.indices.iter().zip(&mb.indices).enumerate() {
            if ia == ib {
                continue;
            }
            diffs += 1;
            let la = fd.class_of(ia).map(|c| fd.levels(c).len).unwrap_or(1);
            let lb = fd.class_of(ib).map(|c| fd.levels(c).len).unwrap_or(1);
            assert_eq!(la, 5, "{}: block {i} swapped from L = {la}", ma.name);
            assert!(lb <= LEVEL_CAP, "{}: block {i} swapped to L = {lb}", ma.name);
        }
    }

    // Whatever follows the matrices — raw tensors, blobs, or the two zero
    // counts of a projections-only file — must be byte-identical.
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    a.read_to_end(&mut ta).expect("source tail");
    b.read_to_end(&mut tb).expect("output tail");
    assert_eq!(ta, tb, "trailing sections differ");
    diffs
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (src, dst) = match (args.first(), args.get(1)) {
        (Some(s), Some(d)) => (s.clone(), d.clone()),
        _ => {
            eprintln!("usage: lswap <in.llvq> <out.llvq>");
            std::process::exit(2);
        }
    };
    let t0 = std::time::Instant::now();
    let nthreads = std::thread::available_parallelism().map_or(8, |n| n.get());

    let s = Searcher::new();
    let fd = FastDecoder::new();
    let ix = Indexer::new();

    let mut r = BufReader::with_capacity(1 << 20, File::open(&src).unwrap_or_else(|e| {
        panic!("open {src}: {e}");
    }));
    let h = read_header(&mut r).expect("valid artifact header");
    let magic: &[u8; 4] = match h.version {
        1 => MAGIC_V1,
        2 => MAGIC_V2,
        _ => MAGIC,
    };
    // Write to a temporary path and rename only after `verify()` has passed:
    // an interrupted run must never leave a plausible-looking file at `dst`
    // (a truncated artifact opens fine and scores wrong).
    let tmp = format!("{dst}.tmp");
    let mut w = BufWriter::with_capacity(
        1 << 20,
        File::create(&tmp).unwrap_or_else(|e| panic!("create {tmp}: {e}")),
    );
    w.write_all(magic).expect("write magic");
    w.write_all(&h.matrices.to_le_bytes()).expect("write count");

    println!(
        "{src} : format v{}, {} matrices — swap L = 5 → L ≤ {LEVEL_CAP}, \
         boule m ≤ {SHELL_CAP}, {nthreads} threads",
        h.version, h.matrices
    );

    let mut total = Stats::default();
    for mi in 0..h.matrices {
        let mut m: RawMatrix = read_matrix_raw(&mut r).expect("valid matrix");
        assert_eq!(
            m.shell_cap, SHELL_CAP,
            "{}: file is not a leech1c12 artifact",
            m.name
        );
        let chunk = m.indices.len().div_ceil(nthreads);
        let stats: Vec<Stats> = std::thread::scope(|sc| {
            let mut handles = Vec::new();
            for slice in m.indices.chunks_mut(chunk) {
                let (s, fd, ix) = (&s, &fd, &ix);
                handles.push(sc.spawn(move || swap_slice(slice, s, fd, ix)));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let before = total.swapped;
        for st in stats {
            total.absorb(st);
        }
        eprintln!(
            "  {:>3}/{} {:<44} {:>7} blocs échangés ({:.0} s)",
            mi + 1,
            h.matrices,
            m.name,
            total.swapped - before,
            t0.elapsed().as_secs_f64()
        );
        write_matrix_raw(&mut w, &m).expect("write matrix");
    }
    // Copy the raw-tensor and blob sections (or a projections-only file's two
    // zero counts) byte for byte.
    let copied = std::io::copy(&mut r, &mut w).expect("copy trailing sections");
    w.flush().expect("flush output");
    drop(w);

    let blocks: u64 = total.hist.iter().sum();
    println!("\n{} blocs, dont {} origine — passe en {:.0} s", blocks, total.origin, t0.elapsed().as_secs_f64());
    println!("  niveaux de magnitude par bloc (avant swap)");
    for l in 1..=MAX_LEVELS {
        if total.hist[l] > 0 {
            println!("  {l} niveaux{:>16} blocs", total.hist[l]);
        }
    }
    println!("  sections traînantes copiées : {copied} octets");

    println!("\n  blocs échangés : {} (attendu {EXPECTED_SWAPS})", total.swapped);
    assert_eq!(
        total.swapped, EXPECTED_SWAPS,
        "swap count disagrees with the rtbits census"
    );

    // The angular price: the gain is copied, so the reconstruction changes
    // only by the angle between old and new direction.
    let mut cs = total.cosines;
    cs.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = cs.len();
    assert_eq!(n as u64, EXPECTED_SWAPS);
    let mean = cs.iter().sum::<f64>() / n as f64;
    let p1 = cs[n / 100];
    println!("  cos(ancien, nouveau) : min {:.6}   p1 {:.6}   moyen {:.6}   max {:.6}", cs[0], p1, mean, cs[n - 1]);
    println!("  gain : inchangé par construction (jamais lu ni écrit par la passe)");

    // Control reload: the produced file must open with the sealed reader and
    // differ from the source at exactly the swapped indices.
    println!("\n  rechargement de contrôle…");
    let diffs = verify(&src, &tmp, &fd);
    println!("  indices différents : {diffs} (= blocs échangés)");
    assert_eq!(diffs, total.swapped, "diff count disagrees with the swap count");

    // Every check has passed: the file may now claim its real name.
    std::fs::rename(&tmp, &dst).unwrap_or_else(|e| panic!("rename {tmp} → {dst}: {e}"));

    let (src_b, dst_b) = (
        std::fs::metadata(&src).expect("stat source").len(),
        std::fs::metadata(&dst).expect("stat output").len(),
    );
    println!("\n── {dst}");
    println!("   taille : {dst_b} octets (source {src_b})");
    println!("   vérifié : hors swap bit-identique, gains intacts, relit par le lecteur — {:.0} s total", t0.elapsed().as_secs_f64());
}
