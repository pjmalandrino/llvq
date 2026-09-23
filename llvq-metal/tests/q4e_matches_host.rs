//! The q4 embedding kernels in MSL, against llvq-artifact's own dequantizer.
//!
//! `q8_matches_host.rs` at half the weight width; its header gives the method
//! and the reasons. The expected value of every weight comes from
//! `llvq_artifact::RawTensor::to_f32`, the packing mirrors
//! `embedquant::quantize_affine(t, 4, 64)` (nibbles low first at the global
//! flat index), and the head's lane order comes from the CUDA original
//! `tv_emb_q4_h`: lane `l` takes words `l, l + 32, ...`, the eight columns of
//! a word accumulate in order, and the lanes meet in a butterfly. Every
//! assertion is an equality.
//!
//! Nothing runs through candle here: this proves the kernels, not the binding
//! in `fused_metal.rs`.

#![cfg(target_os = "macos")]

use llvq_artifact::{f16_to_f32, QuantData, RawData, RawTensor};
use llvq_core::SplitMix64;
use llvq_metal::{f16_bits, Kernel};

const SOURCE: &str = include_str!("../../llvq-llm/kernels/emb_q4.metal");

const GROUP_WIDTH: usize = 64;
const LANES: usize = 32;
const THREADS: usize = 256;
const TG_LIMIT: u64 = 32_768;
const F16_ONE: u16 = 0x3c00;

struct Table {
    rows: usize,
    d: usize,
    gpr: usize,
    words: Vec<u32>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    deq: Vec<f32>,
}

/// `embedquant::quantize_affine(t, 4, 64)`, mirrored.
fn quantize_q4(weights: &[u16], rows: usize, d: usize) -> QuantData {
    let gpr = d.div_ceil(GROUP_WIDTH);
    let mut packed = vec![0u8; (rows * d).div_ceil(2)];
    let mut scales = Vec::with_capacity(rows * gpr);
    let mut biases = Vec::with_capacity(rows * gpr);
    for row in 0..rows {
        for g in 0..gpr {
            let lo = row * d + g * GROUP_WIDTH;
            let hi = row * d + ((g + 1) * GROUP_WIDTH).min(d);
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &b in &weights[lo..hi] {
                let v = f16_to_f32(b);
                mn = mn.min(v);
                mx = mx.max(v);
            }
            let s = f16_bits((mx - mn) / 15.0);
            let graded = f16_to_f32(s) > 0.0;
            let s = if graded { s } else { F16_ONE };
            let b16 = f16_bits(mn);
            scales.push(s);
            biases.push(b16);
            let (sf, bf) = (f16_to_f32(s), f16_to_f32(b16));
            for (k, &b) in weights[lo..hi].iter().enumerate() {
                let q = match graded {
                    true => ((f16_to_f32(b) - bf) / sf).round().clamp(0.0, 15.0) as u8,
                    false => 0,
                };
                let i = lo + k;
                packed[i / 2] |= q << (4 * (i % 2));
            }
        }
    }
    QuantData {
        bits: 4,
        group: GROUP_WIDTH,
        packed,
        scales,
        biases,
    }
}

/// A random table with one constant group planted, the ungraded branch.
fn table(seed: u64, rows: usize, d: usize) -> Table {
    assert!(
        d.is_multiple_of(8),
        "the kernels read rows as u32 words of eight nibbles"
    );
    let mut rng = SplitMix64::new(seed);
    let mut weights: Vec<u16> = (0..rows * d)
        .map(|_| f16_bits(rng.next_gaussian() as f32))
        .collect();
    for w in weights.iter_mut().take(GROUP_WIDTH) {
        *w = f16_bits(0.25);
    }
    let q = quantize_q4(&weights, rows, d);
    // Word `i` holds columns 8i..8i+8, column 8i in the low nibble of the low
    // byte. Little-endian, spelled out.
    let words = q
        .packed
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (scales, biases, gpr) = (q.scales.clone(), q.biases.clone(), d.div_ceil(GROUP_WIDTH));
    let tensor = RawTensor {
        name: "embed.weight".into(),
        dims: vec![rows, d],
        data: RawData::Quant(q),
    };
    Table {
        rows,
        d,
        gpr,
        words,
        scales,
        biases,
        deq: tensor.to_f32(),
    }
}

fn activation(seed: u64, d: usize, off: usize) -> Vec<u16> {
    let mut rng = SplitMix64::new(seed);
    (0..off + d)
        .map(|_| f16_bits(rng.next_gaussian() as f32))
        .collect()
}

fn compile(name: &str, exact: bool) -> Kernel {
    match exact {
        true => Kernel::new_exact(SOURCE, name),
        false => Kernel::new(SOURCE, name),
    }
    .expect("the q4 embedding kernels compile")
}

fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, v: &u32) {
    enc.set_bytes(index, 4, v as *const u32 as *const std::ffi::c_void);
}

fn run_gather(t: &Table, ids: &[u32], ids_off: u32, exact: bool) -> Vec<f32> {
    let k = compile("emb_q4_gather_metal", exact);
    let ntok = ids.len() - ids_off as usize;
    let b_w = k.buffer(&t.words);
    let b_s = k.buffer(&t.scales);
    let b_b = k.buffer(&t.biases);
    let b_i = k.buffer(ids);
    let b_y = k.empty::<f32>(ntok * t.d);
    let (d, gpr) = (t.d as u32, t.gpr as u32);
    k.dispatch((ntok * THREADS) as u64, THREADS as u64, |enc| {
        enc.set_buffer(0, Some(&b_w), 0);
        enc.set_buffer(1, Some(&b_s), 0);
        enc.set_buffer(2, Some(&b_b), 0);
        enc.set_buffer(3, Some(&b_i), 0);
        enc.set_buffer(4, Some(&b_y), 0);
        set_u32(enc, 5, &d);
        set_u32(enc, 6, &gpr);
        set_u32(enc, 7, &ids_off);
    });
    unsafe { k.read::<f32>(&b_y, ntok * t.d) }
}

fn run_head(
    t: &Table,
    x: &[u16],
    x_off: u32,
    y_off: u32,
    y_len: usize,
    sentinel: f32,
    exact: bool,
) -> Vec<f32> {
    let k = compile("tv_emb_q4_metal", exact);
    assert!(
        (t.rows * LANES).is_multiple_of(THREADS),
        "a partial threadgroup would split a row"
    );
    let tg_bytes = ((t.d * 4) as u64).next_multiple_of(16);
    assert!(tg_bytes <= TG_LIMIT, "{tg_bytes} B of threadgroup memory");
    let b_w = k.buffer(&t.words);
    let b_s = k.buffer(&t.scales);
    let b_b = k.buffer(&t.biases);
    let b_x = k.buffer(x);
    let b_y = k.buffer(&vec![sentinel; y_len]);
    let (d, gpr) = (t.d as u32, t.gpr as u32);
    k.dispatch((t.rows * LANES) as u64, THREADS as u64, |enc| {
        enc.set_buffer(0, Some(&b_w), 0);
        enc.set_buffer(1, Some(&b_s), 0);
        enc.set_buffer(2, Some(&b_b), 0);
        enc.set_buffer(3, Some(&b_x), 0);
        enc.set_buffer(4, Some(&b_y), 0);
        set_u32(enc, 5, &d);
        set_u32(enc, 6, &gpr);
        set_u32(enc, 7, &x_off);
        set_u32(enc, 8, &y_off);
        enc.set_threadgroup_memory_length(0, tg_bytes);
    });
    unsafe { k.read::<f32>(&b_y, y_len) }
}

fn gather_reference(t: &Table, ids: &[u32], ids_off: usize) -> Vec<f32> {
    ids[ids_off..]
        .iter()
        .flat_map(|&id| t.deq[id as usize * t.d..(id as usize + 1) * t.d].to_vec())
        .collect()
}

/// The head's arithmetic in `tv_emb_q4_h`'s order: eight columns a word.
fn head_reference(t: &Table, x: &[u16], x_off: usize) -> Vec<f32> {
    let xs: Vec<f32> = (0..t.d).map(|i| f16_to_f32(x[x_off + i])).collect();
    let mut y = vec![0f32; t.rows];
    for (row, out) in y.iter_mut().enumerate() {
        let mut lanes = [0f32; LANES];
        for wi in 0..t.d / 8 {
            let c = wi * 8;
            let acc = &mut lanes[wi % LANES];
            for k in 0..8 {
                *acc = t.deq[row * t.d + c + k].mul_add(xs[c + k], *acc);
            }
        }
        for k in [16usize, 8, 4, 2, 1] {
            let mut next = [0f32; LANES];
            for (l, n) in next.iter_mut().enumerate() {
                *n = lanes[l] + lanes[l ^ k];
            }
            lanes = next;
        }
        *out = lanes[0];
    }
    y
}

fn assert_same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what} {i}: metal {g} against {w}"
        );
    }
}

#[test]
fn the_gather_matches_the_artifact_dequantizer() {
    let t = table(0xE_04A0, 40, 128);
    let ids = [37u32, 5, 0, 12, 39, 1, 26];
    assert_same(
        &run_gather(&t, &ids, 2, true),
        &gather_reference(&t, &ids, 2),
        "value",
    );
}

/// 104 = 64 + 40: the second group is short.
#[test]
fn the_gather_matches_across_a_short_last_group() {
    let t = table(0xE_04A1, 24, 104);
    assert_eq!(t.gpr, 2);
    let ids = [23u32, 0, 17, 9];
    assert_same(
        &run_gather(&t, &ids, 1, true),
        &gather_reference(&t, &ids, 1),
        "value",
    );
}

/// 128 columns is 16 words, so half the lanes take none and enter the
/// butterfly at zero.
#[test]
fn the_head_matches_the_host_with_idle_lanes() {
    let t = table(0xE_04A2, 16, 128);
    let x = activation(0xE_04B2, t.d, 7);
    assert_same(
        &run_head(&t, &x, 7, 0, t.rows, f32::NAN, true),
        &head_reference(&t, &x, 7),
        "row",
    );
}

/// 104 columns, 13 words, across a short last group.
#[test]
fn the_head_matches_the_host_across_a_short_last_group() {
    let t = table(0xE_04A3, 24, 104);
    let x = activation(0xE_04B3, t.d, 3);
    assert_same(
        &run_head(&t, &x, 3, 0, t.rows, f32::NAN, true),
        &head_reference(&t, &x, 3),
        "row",
    );
}

/// 2,048 columns is 256 words, eight a lane: the stride is read seven times.
#[test]
fn the_head_matches_the_host_when_every_lane_takes_eight_words() {
    let t = table(0xE_04C0, 16, 2048);
    assert_eq!(t.d / 8 / LANES, 8);
    let x = activation(0xE_04D0, t.d, 0);
    assert_same(
        &run_head(&t, &x, 0, 0, t.rows, f32::NAN, true),
        &head_reference(&t, &x, 0),
        "row",
    );
}

/// 1,600 columns against 256 threads: every thread takes several columns.
#[test]
fn the_gather_matches_when_every_thread_takes_several_columns() {
    let t = table(0xE_04C1, 24, 1600);
    let ids = [23u32, 0, 17, 9, 11];
    assert_same(
        &run_gather(&t, &ids, 1, true),
        &gather_reference(&t, &ids, 1),
        "value",
    );
}

#[test]
fn the_head_writes_at_its_offset_and_touches_nothing_else() {
    let t = table(0xE_04A4, 16, 128);
    let x = activation(0xE_04B4, t.d, 0);
    let sentinel = -12345.0f32;
    let got = run_head(&t, &x, 0, t.rows as u32, 3 * t.rows, sentinel, true);
    assert_same(&got[t.rows..2 * t.rows], &head_reference(&t, &x, 0), "row");
    for (i, g) in got.iter().enumerate() {
        if !(t.rows..2 * t.rows).contains(&i) {
            assert_eq!(*g, sentinel, "slot {i} was written and should not be");
        }
    }
}

/// The shipped path compiles through candle with fast math on; the pragma,
/// not the option, must carry the arithmetic.
#[test]
fn the_same_source_gives_the_same_numbers_with_fast_math_either_way() {
    let t = table(0xE_04A6, 16, 256);
    let x = activation(0xE_04B6, t.d, 2);
    let exact = run_head(&t, &x, 2, 0, t.rows, f32::NAN, true);
    let fast = run_head(&t, &x, 2, 0, t.rows, f32::NAN, false);
    assert_same(&fast, &exact, "fast-math row");
    assert_same(&exact, &head_reference(&t, &x, 2), "row");
    let ids = [0u32, 9, 15];
    assert_same(
        &run_gather(&t, &ids, 0, false),
        &gather_reference(&t, &ids, 0),
        "fast-math value",
    );
}
