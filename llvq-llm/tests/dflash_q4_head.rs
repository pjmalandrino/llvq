//! The drafter's head through the target's int4 g64 table (`dflash_q4`,
//! `tv_q4_mma` of `kernels/tv_batch_mma.cu`) against the dense path the
//! fixture was made with (`DenseVocab`: the same table rounded to f16
//! weights), on the fixture's own head inputs.
//!
//! For each of the fixture's six rounds (two prompts, three rounds), the 15
//! rows `p{p}.r{r}.out` (the drafter's hidden state entering the head, from
//! the MLX reference) go through:
//!
//! * the kernel, executed on the CPU by `host_batch_mma.cpp` (whole CTAs,
//!   the MMA modelled lane by lane, its sums in k order where the card's
//!   tensor core has its own), on the fixture's candidate rows: every id of
//!   the 15 rows' top 64, padded to a multiple of 128 rows with ids outside
//!   it, at one slice of K, the split the card takes at 15 rows of the 4B
//!   (`CudaBatch::split`: the partials of two slices do not fit);
//! * the dense path in f16 (`hidden` and table f16, f32 accumulation, an f16
//!   store) and in f32, modelled in f64 and rounded as the card would.
//!
//! What is asserted: every logit within the bound the two arithmetics allow,
//! `sum_c |w_c - f16(w_c)| |x_c|` (the weights' rounding, which the int4
//! kernel does not do) plus two f16 ulps of the output plus 1e-5 of
//! `sum_c (|s q_c x_c| + |b x_c|)` (f32 sums in either order, the kernel's
//! two terms apart); that the bound is smaller
//! than the gap between the 1st and the 64th logit of every row, so no id
//! outside the candidates can become a top-4; and the chain's argmax and the
//! tree's top 4 agree with the dense f16 path wherever the dense gap exceeds
//! twice the bound. The agreements are printed in full.
//!
//! Needs `~/llvq-dflash-fixture-2026-11-10.safetensors` and the packed 4B
//! table `~/q4b-hf-2026-09-28/model.safetensors` (its three embedding
//! tensors only, 0.22 GB; no model is loaded), and `clang++`; fails naming
//! what is missing.

#![cfg(any(target_os = "macos", feature = "cuda"))]

use std::collections::BTreeSet;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use candle_core::{DType, Device};
use half::f16;

fn home(p: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join(p)
}

fn probe() -> PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_dflash_q4_{}", std::process::id()));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let res = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-pthread"])
        .arg(dir.join("host_batch_mma.cpp"))
        .arg("-o")
        .arg(&out)
        .output()
        .expect("clang++ is on PATH: the batched kernel text is checked by it (rule 10)");
    assert!(res.status.success(), "clang++ refused host_batch_mma.cpp:\n{}", String::from_utf8_lossy(&res.stderr));
    out
}

/// The kernel's f16 outputs `[n_rows][d_out]`.
#[allow(clippy::too_many_arguments)]
fn kernel(exe: &PathBuf, words: &[u32], sc: &[u16], bi: &[u16], x: &[u16], d_out: usize, d_in: usize, n_rows: usize) -> Vec<f32> {
    let mut f = Vec::new();
    for v in [1u32, d_out as u32, d_in as u32, 6, n_rows as u32, 1] {
        f.extend(v.to_le_bytes());
    }
    words.iter().for_each(|v| f.extend(v.to_le_bytes()));
    sc.iter().for_each(|v| f.extend(v.to_le_bytes()));
    bi.iter().for_each(|v| f.extend(v.to_le_bytes()));
    x.iter().for_each(|v| f.extend(v.to_le_bytes()));
    let mut ch = Command::new(exe).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("probe runs");
    let mut stdin = ch.stdin.take().expect("stdin");
    let w = std::thread::spawn(move || stdin.write_all(&f).expect("input written"));
    let out = ch.wait_with_output().expect("probe exits");
    w.join().expect("writer");
    assert!(out.status.success(), "probe failed: {}", String::from_utf8_lossy(&out.stderr));
    let n = n_rows * d_out;
    assert_eq!(out.stdout.len(), n * 6);
    out.stdout[..2 * n].chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
}

fn ulp16(v: f64) -> f64 {
    let a = v.abs().max(6.1e-5);
    2f64.powi(a.log2().floor() as i32 - 10)
}

#[test]
#[ignore = "needs the fixture, the packed 4B table and clang++"]
fn the_int4_head_matches_the_dense_head_on_the_fixture() {
    let fixture = home("llvq-dflash-fixture-2026-11-10.safetensors");
    assert!(fixture.exists(), "missing {}: run ops/dflash_fixture.py", fixture.display());
    let packed = home("q4b-hf-2026-09-28/model.safetensors");
    assert!(packed.exists(), "missing {}", packed.display());
    let fx = candle_core::safetensors::load(&fixture, &Device::Cpu).unwrap();
    let get = |k: String| fx.get(&k).unwrap_or_else(|| panic!("fixture: no {k}")).clone();
    // SAFETY: a read-only mmap of a file nothing in this process writes.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&packed) }.unwrap();
    let q = st.load("model.embed_tokens.qweight", &Device::Cpu).unwrap();
    let (vocab, half_h) = q.dims2().unwrap();
    let d = half_h * 2;
    let gpr = d / 64;
    let qw = q.flatten_all().unwrap().to_vec1::<u8>().unwrap();
    let bits = |k: &str| -> Vec<u16> {
        st.load(k, &Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f16>().unwrap().iter().map(|v| v.to_bits()).collect()
    };
    let (scales, biases) = (bits("model.embed_tokens.scales"), bits("model.embed_tokens.biases"));
    assert_eq!(scales.len(), vocab * gpr);
    let exe = probe();
    let (mut rows_n, mut argmax_same, mut argmax_decided, mut top4_same, mut top4_decided, mut worst_ratio, mut worst_rel) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0f64, 0f64);
    for p in 0..2 {
        for r in 0..3 {
            let out = get(format!("p{p}.r{r}.out")).to_dtype(DType::F32).unwrap();
            let n = out.dim(0).unwrap();
            let h32 = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let h16: Vec<f16> = h32.iter().map(|v| f16::from_f32(*v)).collect();
            let top = get(format!("p{p}.r{r}.top_ids")).to_dtype(DType::U32).unwrap().to_vec2::<u32>().unwrap();
            // the candidates: every row's top 64, padded with ids outside them
            let mut set: BTreeSet<u32> = top.iter().flatten().copied().collect();
            let mut pad = 0u32;
            while !set.len().is_multiple_of(128) {
                set.insert(pad);
                pad += 1;
            }
            let ids: Vec<u32> = set.into_iter().collect();
            let m = ids.len();
            let mut words = Vec::with_capacity(m * d / 8);
            let (mut sc, mut bi) = (Vec::with_capacity(m * gpr), Vec::with_capacity(m * gpr));
            for &id in &ids {
                let row = &qw[id as usize * half_h..(id as usize + 1) * half_h];
                words.extend(row.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
                sc.extend_from_slice(&scales[id as usize * gpr..(id as usize + 1) * gpr]);
                bi.extend_from_slice(&biases[id as usize * gpr..(id as usize + 1) * gpr]);
            }
            let x: Vec<u16> = h16.iter().map(|v| v.to_bits()).collect();
            let y = kernel(&exe, &words, &sc, &bi, &x, m, d, n);
            for i in 0..n {
                // dense f16 path and the bound, per candidate
                let mut dense16 = vec![0f64; m];
                let mut bound = vec![0f64; m];
                for (j, &id) in ids.iter().enumerate() {
                    let (mut s16, mut sround, mut sabs) = (0f64, 0f64, 0f64);
                    for c in 0..d {
                        let nib = (qw[id as usize * half_h + c / 2] >> ((c % 2) * 4)) & 0xf;
                        let g = id as usize * gpr + c / 64;
                        let (s, b) = (f16::from_bits(scales[g]).to_f32(), f16::from_bits(biases[g]).to_f32());
                        let w = nib as f32 * s + b; // f32, as the kernel's s q + b
                        let wr = f16::from_f32(w).to_f32() as f64; // the dense table's weight
                        let x16 = h16[i * d + c].to_f32() as f64;
                        s16 += wr * x16;
                        sround += (w as f64 - wr).abs() * x16.abs();
                        sabs += (nib as f64 * s as f64 * x16).abs() + (b as f64 * x16).abs();
                    }
                    dense16[j] = f16::from_f64(s16).to_f64();
                    bound[j] = sround + 2.0 * ulp16(s16) + 1e-5 * sabs;
                }
                let k = &y[i * m..(i + 1) * m];
                for j in 0..m {
                    let e = (k[j] as f64 - dense16[j]).abs();
                    assert!(e <= bound[j], "p{p} r{r} row {i} id {}: int4 {} dense {} |error| {e:.3e} > {:.3e}", ids[j], k[j], dense16[j], bound[j]);
                    worst_ratio = worst_ratio.max(e / bound[j]);
                    worst_rel = worst_rel.max(e / dense16.iter().fold(0f64, |a, v| a.max(v.abs())));
                }
                let order = |v: &[f64]| -> Vec<usize> {
                    let mut o: Vec<usize> = (0..m).collect();
                    o.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
                    o
                };
                let kd: Vec<f64> = k.iter().map(|v| *v as f64).collect();
                let (ok, od) = (order(&kd), order(&dense16));
                let bmax = bound.iter().fold(0f64, |a, v| a.max(*v));
                // no id outside the candidates can reach the top 4: the 64th
                // of this row's own top ids sits more than the bound below its 4th
                let row_top: Vec<usize> = top[i].iter().map(|t| ids.binary_search(t).unwrap()).collect();
                let last = row_top.iter().map(|&j| dense16[j]).fold(f64::INFINITY, f64::min);
                assert!(dense16[od[3]] - last > 2.0 * bmax, "p{p} r{r} row {i}: the top 64 does not cover the top 4 under the bound");
                rows_n += 1;
                if dense16[od[0]] - dense16[od[1]] > 2.0 * bmax {
                    argmax_decided += 1;
                    assert_eq!(ok[0], od[0], "p{p} r{r} row {i}: argmax moved past twice the bound");
                }
                argmax_same += usize::from(ok[0] == od[0]);
                if (0..4).all(|t| dense16[od[t]] - dense16[od[t + 1]] > 2.0 * bmax) {
                    top4_decided += 1;
                    assert_eq!(ok[..4], od[..4], "p{p} r{r} row {i}: top 4 moved past twice the bound");
                }
                top4_same += usize::from(ok[..4] == od[..4]);
            }
        }
    }
    println!(
        "int4 head against the dense f16 head on {rows_n} fixture rows: argmax equal on {argmax_same} ({argmax_decided} decided past twice the bound), \
         top 4 in order equal on {top4_same} ({top4_decided} decided); worst error {worst_ratio:.3} of its bound, {worst_rel:.2e} of the row's largest logit"
    );
}
