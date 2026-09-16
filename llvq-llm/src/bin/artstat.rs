//! What an artifact actually holds: tails, gain levels, and their balance.
//!
//! ```text
//! cargo run --release -p llvq-llm --bin artstat -- ~/llvq-4b-tetra.llvq
//! ```
//!
//! ## Why this is worth a binary
//!
//! Two operations that look identical are not. `artscale` multiplies a
//! matrix's **centroids**, which moves every lattice-coded block and leaves
//! the tail alone. `errmap` probes by multiplying the **reconstructed
//! tensor**, which moves the tail too. They coincide exactly when the tail is
//! empty — and on the 4B it is empty nowhere (*measured*, 252 records of 252).
//!
//! So a map fitted with the second and applied with the first is answering a
//! slightly different question than it was asked. The difference is small,
//! `d_in % 24` columns out of `d_in`, and it is not zero. Anything that fits
//! centroids must probe centroids.
//!
//! It reads records raw — the gain level is a field, so nothing is decoded and
//! a 4B file is walked without rebuilding a single weight.

use std::collections::BTreeSet;
use std::io::BufReader;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).context_msg()?;
    let mut r = BufReader::with_capacity(1 << 20, std::fs::File::open(&path)?);
    let head = llvq_artifact::read_header(&mut r)?;

    let (mut lattice, mut int4, mut with_tail, mut tail_values) = (0usize, 0usize, 0usize, 0usize);
    let (mut level0, mut level1) = (0u64, 0u64);
    let mut centroid_counts = BTreeSet::new();
    let mut widths = BTreeSet::new();
    for _ in 0..head.matrices {
        match llvq_artifact::read_record(&mut r, head.version)? {
            llvq_artifact::Record::Lattice(m) => {
                lattice += 1;
                centroid_counts.insert(m.centroids.len());
                if !m.tail.is_empty() {
                    with_tail += 1;
                    tail_values += m.tail.len();
                    widths.insert(m.d_in % 24);
                }
                for &g in &m.gains {
                    if g == 0 {
                        level0 += 1;
                    } else {
                        level1 += 1;
                    }
                }
            }
            llvq_artifact::Record::Int4(_) => int4 += 1,
        }
    }

    println!("{path}");
    println!("  records          {lattice} lattice, {int4} int4");
    println!("  centroid counts  {centroid_counts:?}");
    println!(
        "  tails            {with_tail} of {lattice} records carry one, {tail_values} values, \
         widths {widths:?} columns"
    );
    let blocks = level0 + level1;
    println!(
        "  gain levels      {level0} on level 0, {level1} on level 1 ({:.1} % on level 1, of \
         {blocks} blocks)",
        100.0 * level1 as f64 / blocks as f64
    );
    // A ratio between the two levels only exists as a knob if both are
    // populated: an artifact that put every block on one level would make the
    // second gain parameter a no-op, and that has to be read off the file
    // rather than assumed from the format having a bit.
    let share = level1 as f64 / blocks as f64;
    if !(0.01..=0.99).contains(&share) {
        println!("  NOTE: the two levels are not both populated — a ratio between them is a no-op");
    }
    Ok(())
}

/// `Option::context` without pulling anyhow's trait into scope for one call.
trait ContextMsg<T> {
    fn context_msg(self) -> anyhow::Result<T>;
}

impl<T> ContextMsg<T> for Option<T> {
    fn context_msg(self) -> anyhow::Result<T> {
        self.ok_or_else(|| anyhow::anyhow!("usage: artstat <artifact.llvq>"))
    }
}
