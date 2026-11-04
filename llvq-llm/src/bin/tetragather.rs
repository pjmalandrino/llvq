//! What one warp's loads cost the L1/shared pipe in the Tetra decode, counted
//! on a real file's words.
//!
//! Usage: `cargo run --release -p llvq-llm --bin tetragather -- <file> [row_step] [tile]`
//!
//! Replays, for every row of every Tetra matrix (or every `row_step`-th row),
//! the warp-iterations of `tv_tetra48_seg` / `tv_tetra48_persall`: lane `l`
//! takes block `jlo + 32k + l` of a `tile`-block tile (default 64, the served
//! tile on sm_89). For each load of the block decode it counts, over the
//! lanes active in that iteration:
//!
//! - **lines**: distinct 128-byte lines, the L1TEX wavefronts of a global load
//!   under the model of `docs/mesures/pers-sass-2026-11-05.txt` (one tag a
//!   cycle; an assumption, which the gather arms of the next lot measure);
//! - **sectors**: distinct 32-byte sectors, what L1 moves on a hit;
//! - **banks**: the shared-memory wavefronts if the table sat in shared
//!   memory, 128-byte aligned: the largest number of distinct 4-byte words any
//!   one bank is asked for (one word read by many lanes is a broadcast).
//!
//! It also counts the activation reads (six 128-bit loads a lane at stride 24
//! or trio stride 28, 24 scalar loads at stride 25) and the word loads.
//! A control replays the same geometry on uniformly drawn 48-bit labels, so a
//! skew of the file's indices shows as a gap. Nothing here runs a kernel: every
//! number is *computed* from the file's bytes.

use llvq_core::SplitMix64;
use llvq_llm::fused::{self, FusedLayout, HostStream};

const N0_MIXED: u32 = 1240; // F1R_N0_MIXED, llvq_f1rank.cuh

/// The table loads of one block, as `f1r_v3_fetch` and `tetra48_dot` perform
/// them: element indices into each table.
struct Fetch {
    row1: u32,
    row2: u32,
    row3: u32,
    branch: u32,
    prefix: u32,
    suffix: u32,
    gscale: u32,
}

fn fetch(lo: u32, hi16: u32, branches: &[u16]) -> Fetch {
    let p = lo & 1;
    let r = (lo >> 1) & 1;
    let s8 = (lo >> 2) & 63;
    let b1 = (lo >> 8) & 1;
    let i1 = (lo >> 9) & 0x7ff;
    let b2 = (lo >> 20) & 15;
    let i2 = ((lo >> 24) | ((hi16 & 7) << 8)) & 0x7ff;
    let b3 = (hi16 >> 3) & 1;
    let i3 = (hi16 >> 4) & 0x7ff;
    let branch = 16 * s8 + b2;
    let br = branches[branch as usize] as u32;
    let s16 = (br >> 8) & 63;
    let mid = i2 < N0_MIXED;
    let idx2 = if mid { i2 } else { (2048 - N0_MIXED) + i2 };
    let delta = if mid { 0 } else { 1 };
    let r3 = (p ^ r ^ delta) & 1;
    let row1 = 2048 * r + i1;
    let row3 = 2048 * r3 + i3;
    Fetch {
        row1,
        row2: idx2,
        row3,
        branch,
        prefix: 2 * s8 + b1,
        suffix: 2 * s16 + b3,
        gscale: (hi16 >> 15) & 1,
    }
}

#[derive(Default, Clone, Copy)]
struct Tally {
    lines: u64,
    sectors: u64,
    banks: u64,
}

/// Distinct lines, distinct sectors and shared wavefronts of one warp load of
/// `width`-byte elements at `idx` in a table that starts 128-byte aligned.
fn count(idx: &[u32], width: u32) -> Tally {
    let mut lines: Vec<u32> = idx.iter().map(|&i| i * width / 128).collect();
    let mut sectors: Vec<u32> = idx.iter().map(|&i| i * width / 32).collect();
    let mut words: Vec<u32> = idx.iter().map(|&i| i * width / 4).collect();
    lines.sort_unstable();
    lines.dedup();
    sectors.sort_unstable();
    sectors.dedup();
    words.sort_unstable();
    words.dedup();
    let mut per_bank = [0u32; 32];
    for w in &words {
        per_bank[(w % 32) as usize] += 1;
    }
    Tally {
        lines: lines.len() as u64,
        sectors: sectors.len() as u64,
        banks: *per_bank.iter().max().unwrap_or(&0) as u64,
    }
}

/// Shared wavefronts of one 128-bit load a lane: four phases of eight lanes,
/// each the largest number of distinct 16-byte chunks one bank group (16 B, 8
/// groups) is asked for.
fn lds128(byte_addr: &[(u32, u32)]) -> u64 {
    let mut total = 0u64;
    for phase in 0..4u32 {
        let mut chunks: Vec<u32> = byte_addr
            .iter()
            .filter(|(lane, _)| lane / 8 == phase)
            .map(|(_, a)| a / 16)
            .collect();
        if chunks.is_empty() {
            continue;
        }
        chunks.sort_unstable();
        chunks.dedup();
        let mut per = [0u32; 8];
        for c in &chunks {
            per[(c % 8) as usize] += 1;
        }
        total += *per.iter().max().unwrap() as u64;
    }
    total
}

/// Shared wavefronts of one 32-bit load a lane at word addresses `w`.
fn lds32(w: &[u32]) -> u64 {
    let mut v = w.to_vec();
    v.sort_unstable();
    v.dedup();
    let mut per = [0u32; 32];
    for x in &v {
        per[(x % 32) as usize] += 1;
    }
    *per.iter().max().unwrap() as u64
}

const SITES: [&str; 9] = ["word0", "word1", "row1", "row2", "row3", "branch", "prefix", "suffix", "invnorm"];

#[derive(Default)]
struct Acc {
    iters: u64,
    lanes: u64,
    site: [Tally; 9],
    gscale: Tally,
    x24: u64,
    x28: u64,
    x25: u64,
}

impl Acc {
    fn line(&self, label: &str) {
        let n = self.iters as f64;
        println!(
            "{label}: {} warp-iterations, {:.2} active lanes on average",
            self.iters,
            self.lanes as f64 / n
        );
        println!("  load      lines  sectors  banks   (per warp-iteration)");
        for (k, s) in SITES.iter().enumerate() {
            let t = self.site[k];
            println!(
                "  {:<8} {:6.2}  {:7.2}  {:5.2}",
                s,
                t.lines as f64 / n,
                t.sectors as f64 / n,
                t.banks as f64 / n
            );
        }
        let g = self.gscale;
        println!(
            "  {:<8} {:6.2}  {:7.2}  {:5.2}",
            "gscale",
            g.lines as f64 / n,
            g.sectors as f64 / n,
            g.banks as f64 / n
        );
        let tables_l1: u64 = (2..9).map(|k| self.site[k].lines).sum::<u64>() + g.lines;
        let tables_sh: u64 = (2..9).map(|k| self.site[k].banks).sum::<u64>() + g.banks;
        let words: u64 = self.site[0].lines + self.site[1].lines;
        println!(
            "  sums      words {:.2} | tables in L1 {:.2} | tables in shared {:.2} | x: stride 24 x128 {:.2}, trio 28 x128 {:.2}, stride 25 x32 {:.2}",
            words as f64 / n,
            tables_l1 as f64 / n,
            tables_sh as f64 / n,
            self.x24 as f64 / n,
            self.x28 as f64 / n,
            self.x25 as f64 / n
        );
    }
}

/// The trio staging's six 16-byte chunks of block b at stride 28: chunk i is
/// trio slots 4i..4i+4, at float offset 28 b + 4 i.
fn replay(
    acc: &mut Acc,
    blocks: &[(u32, u32, u32)], // (lane, lo, hi16) of the active lanes, j - jlo in the tag
    word_idx: &[(u32, u32)],    // (w0 index, w1 index) in the row's u32 words
    jrel: &[u32],               // j - jlo of each active lane
    t: &fused::Tetra48Tables,
) {
    acc.iters += 1;
    acc.lanes += blocks.len() as u64;
    let f: Vec<Fetch> = blocks.iter().map(|&(_, lo, hi)| fetch(lo, hi, &t.branches)).collect();
    let w0: Vec<u32> = word_idx.iter().map(|w| w.0).collect();
    let w1: Vec<u32> = word_idx.iter().map(|w| w.1).collect();
    let sites: [Vec<u32>; 9] = [
        w0,
        w1,
        f.iter().map(|x| x.row1).collect(),
        f.iter().map(|x| x.row2).collect(),
        f.iter().map(|x| x.row3).collect(),
        f.iter().map(|x| x.branch).collect(),
        f.iter().map(|x| x.prefix).collect(),
        f.iter().map(|x| x.suffix).collect(),
        // invnorm: 32 floats, one line and one word a bank, so any shell
        // index costs one line and one wavefront; the decode is not needed.
        blocks.iter().map(|_| 0).collect(),
    ];
    let widths = [4u32, 4, 4, 4, 4, 2, 1, 1, 4];
    for k in 0..9 {
        let c = count(&sites[k], widths[k]);
        acc.site[k].lines += c.lines;
        acc.site[k].sectors += c.sectors;
        acc.site[k].banks += c.banks;
    }
    let g = count(&f.iter().map(|x| x.gscale).collect::<Vec<_>>(), 4);
    acc.gscale.lines += g.lines;
    acc.gscale.sectors += g.sectors;
    acc.gscale.banks += g.banks;
    for i in 0..6u32 {
        let a24: Vec<(u32, u32)> =
            blocks.iter().zip(jrel).map(|(&(l, _, _), &r)| (l, (24 * r + 4 * i) * 4)).collect();
        let a28: Vec<(u32, u32)> =
            blocks.iter().zip(jrel).map(|(&(l, _, _), &r)| (l, (28 * r + 4 * i) * 4)).collect();
        acc.x24 += lds128(&a24);
        acc.x28 += lds128(&a28);
    }
    for c in 0..24u32 {
        let w: Vec<u32> = jrel.iter().map(|&r| 25 * r + c).collect();
        acc.x25 += lds32(&w);
    }
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let path = a.first().ok_or("usage: tetragather <file> [row_step] [tile]")?;
    let step: usize = a.get(1).map(|s| s.parse().map_err(|e| format!("{e}"))).transpose()?.unwrap_or(1);
    let tile: u32 = a.get(2).map(|s| s.parse().map_err(|e| format!("{e}"))).transpose()?.unwrap_or(64);
    if tile == 0 || !tile.is_multiple_of(32) {
        return Err(format!("tile {tile}: a multiple of 32 is owed (lane order)"));
    }
    let tetra = llvq_search::tetra::Tetra::new();
    let t = fused::tetra48_tables(&tetra);
    let m = fused::load(path, FusedLayout::Tetra48)?;
    let mut by_din: std::collections::BTreeMap<usize, Acc> = Default::default();
    let mut ctl: std::collections::BTreeMap<usize, Acc> = Default::default();
    let mut rng = SplitMix64::new(0x7E7_A9A7);
    let mut names = Vec::new();
    for mx in &m.matrices {
        let HostStream::Tetra48 { words, stride_u32 } = &mx.stream else { continue };
        names.push(mx.name.clone());
        let nb = mx.nblocks as u32;
        for row in (0..mx.d_out).step_by(step) {
            let wrow = &words[row * *stride_u32 as usize..];
            let mut jlo = 0u32;
            while jlo < nb {
                let jhi = (jlo + tile).min(nb);
                let mut base = jlo;
                while base < jhi {
                    let mut blocks = Vec::with_capacity(32);
                    let mut widx = Vec::with_capacity(32);
                    let mut jrel = Vec::with_capacity(32);
                    let mut rnd = Vec::with_capacity(32);
                    for lane in 0..32u32 {
                        let j = base + lane;
                        if j >= jhi {
                            break;
                        }
                        let w = (3 * j) >> 1;
                        let (w0, w1) = (wrow[w as usize], wrow[w as usize + 1]);
                        let (lo, hi16) = if j & 1 == 1 { ((w0 >> 16) | (w1 << 16), w1 >> 16) } else { (w0, w1 & 0xffff) };
                        blocks.push((lane, lo, hi16));
                        widx.push(((row as u32) * stride_u32 + w, (row as u32) * stride_u32 + w + 1));
                        jrel.push(j - jlo);
                        let r = rng.next();
                        rnd.push((lane, r as u32, ((r >> 32) as u32) & 0xffff));
                    }
                    replay(by_din.entry(mx.d_in).or_default(), &blocks, &widx, &jrel, &t);
                    replay(ctl.entry(mx.d_in).or_default(), &rnd, &widx, &jrel, &t);
                    base += 32;
                }
                jlo = jhi;
            }
        }
    }
    println!("file {path}, every {step}th row, tile {tile}");
    println!("{} Tetra matrices, e.g. {}", names.len(), names.iter().take(3).cloned().collect::<Vec<_>>().join(", "));
    for (d, acc) in &by_din {
        acc.line(&format!("d_in {d}, the file's words"));
        ctl[d].line(&format!("d_in {d}, uniform labels (control)"));
    }
    Ok(())
}
