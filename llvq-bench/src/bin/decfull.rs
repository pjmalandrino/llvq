//! A complete fast decoder for format v1 — same bits, same points, no u128,
//! no factorials, no allocations.
//!
//! `decfast` showed the unrank recurrence alone is 4.7× faster than the
//! factorial reference. This is the end-to-end version: class lookup,
//! mixed-radix split, both unranks, signs and placement, in fixed buffers —
//! everything `Indexer::decode` does, decoded from the same index to the same
//! point. The implementation now lives in `llvq_search::fastdec` (the
//! load-time transcoder calls it); this bin keeps the measurement.
//!
//! Correctness comes first and is exhaustive where it matters: every class's
//! first and last index (383 × 2 boundary cases), plus 200,000 uniform draws,
//! all compared bit-for-bit against `Indexer::decode` before a single
//! measurement is taken. (The same sweep is pinned as a release-mode test in
//! `llvq-search/tests/fastdec.rs`.)
//!
//! Run: `cargo run --release -p llvq-bench --bin decfull`

use llvq_core::SplitMix64;
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::{Indexer, N13};
use std::time::Instant;

fn main() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();

    // ---- correctness: every class boundary, then 200,000 uniform draws ----
    let mut checked = 0usize;
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        for idx in [first, last] {
            let a = ix.decode(idx).expect("valid");
            let b = fd.decode(idx).expect("valid");
            assert_eq!(a, b, "boundary index {idx} disagrees");
            checked += 1;
        }
    }
    let mut rng = SplitMix64::new(0x6_F0FF);
    const N: usize = 200_000;
    let idx: Vec<u64> = (0..N).map(|_| 1 + rng.next() % N13).collect();
    for &i in &idx {
        let a = ix.decode(i).expect("valid");
        let b = fd.decode(i).expect("valid");
        assert_eq!(a, b, "index {i} disagrees");
        checked += 1;
    }
    assert_eq!(fd.decode(0), Some([0; 24]));
    assert_eq!(fd.decode(N13 + 1), None);

    // ---- then time, both on the same indices ----
    let mut sink = 0i64;
    let t = Instant::now();
    for &i in &idx {
        sink += ix.decode(i).expect("valid")[0] as i64;
    }
    let t_ref = t.elapsed().as_secs_f64() / N as f64;

    let t = Instant::now();
    for &i in &idx {
        sink += fd.decode(i).expect("valid")[0] as i64;
    }
    let t_fast = t.elapsed().as_secs_f64() / N as f64;

    println!(
        "full decode, {N} uniform indices + {} class boundaries (sink {sink})\n",
        2 * fd.n_classes()
    );
    println!("  {:<40}{:>10}  {:>8}", "decoder", "ns/block", "blocks/s");
    println!("  {}", "-".repeat(64));
    println!(
        "  {:<40}{:>10.1}  {:>10.2e}",
        "Indexer::decode (u128, factorials, Vec)",
        t_ref * 1e9,
        1.0 / t_ref
    );
    println!(
        "  {:<40}{:>10.1}  {:>10.2e}",
        "FastDecoder (u64, recurrence, fixed buffers)",
        t_fast * 1e9,
        1.0 / t_fast
    );
    println!("  {}", "-".repeat(64));
    println!("  {:>50.1}× faster", t_ref / t_fast);
    println!(
        "\n  {checked} points compared bit for bit — same format v1, same 2.1595\n  \
         bits/weight on the file, not one bit changes."
    );

    // What it means for the load-time transcode of a 4B model.
    let blocks_4b = 3_633_315_840u64 / 24;
    let secs = blocks_4b as f64 * t_fast / 12.0;
    println!(
        "\n  load-time transcode of a 4B ({} blocks, 12 cores): ~{secs:.1} s",
        blocks_4b
    );
}
