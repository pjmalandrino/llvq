/// A CPU int4 projection that declares `k` rows a launch, for the port's
/// plumbing: `matvec_rows` is its own matvec row by row, as the port's
/// contract demands of a real kernel, and it counts its launches.
struct RowsInt4 {
    w: Tensor,
    k: usize,
    launches: std::sync::atomic::AtomicUsize,
}

impl crate::device::Int4Proj for RowsInt4 {
    fn name(&self) -> &str {
        "fake.int4"
    }
    fn d_out(&self) -> usize {
        self.w.dim(0).unwrap()
    }
    fn d_in(&self) -> usize {
        self.w.dim(1).unwrap()
    }
    fn matvec(&self, x: &Tensor, out_dims: &[usize]) -> Result<Tensor> {
        let y = x.reshape((1, self.d_in()))?.matmul(&self.w.t()?)?;
        let mut d = out_dims.to_vec();
        *d.last_mut().expect("rank >= 1") = self.d_out();
        y.reshape(d)
    }
    fn rows_per_launch(&self) -> usize {
        self.k
    }
    fn matvec_rows(&self, x: &Tensor, n: usize) -> Result<Tensor> {
        assert!((1..=self.k).contains(&n), "{n} rows for a launch of {}", self.k);
        assert_eq!(x.dims(), [n, self.d_in()], "the chunk as [rows, d_in]");
        self.launches.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let ys = (0..n)
            .map(|i| self.matvec(&x.narrow(0, i, 1)?, &[1, self.d_in()]))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&ys, 0)
    }
}

/// An int4 projection that declares rows a launch gets ceil(rows / k)
/// launches, every row in its place, and the caller's shape back; the
/// same numbers as the row-by-row loop it replaces.
#[test]
fn the_int4_rows_port_chunks_in_order_and_keeps_the_shape() {
    let dev = candle_core::Device::Cpu;
    let w = Tensor::randn(0f32, 1.0, (5, 8), &dev).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, 7, 8), &dev).unwrap();
    let fake = |k| std::sync::Arc::new(RowsInt4 { w: w.clone(), k, launches: 0.into() });
    let (one, three) = (fake(1), fake(3));
    let want = Proj::Int4(one.clone()).forward(&x).unwrap();
    let got = Proj::Int4(three.clone()).forward(&x).unwrap();
    assert_eq!(got.dims(), [1, 7, 5]);
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(flat(&got), flat(&want), "rows out of place or recomputed");
    assert_eq!(three.launches.load(std::sync::atomic::Ordering::SeqCst), 3, "7 rows at 3 a launch");
    assert_eq!(one.launches.load(std::sync::atomic::Ordering::SeqCst), 0, "k = 1 never reaches matvec_rows");
}

/// 4E's cache API on the Grow store: reserve keeps every held position
/// and rounds the capacity up to the chunk; set_len refuses past it.
#[test]
fn grow_reserve_keeps_the_positions() {
    let dev = Device::Cpu;
    let mut c = KvCache::with_store(KvMode::F16, crate::kvq::KvStore::Grow(4));
    let k = Tensor::arange(0f32, 2.0 * 3.0 * 8.0, &dev).unwrap().reshape((1, 2, 3, 8)).unwrap();
    let v = (k.clone() * 2.0).unwrap();
    c.append(&k, &v).unwrap();
    assert_eq!(c.len(), 3);
    assert_eq!(c.grow_buffers().unwrap().0.dim(2).unwrap(), 4);
    c.reserve(9).unwrap();
    let (kb, vb) = c.grow_buffers().unwrap();
    assert_eq!(kb.dim(2).unwrap(), 12, "rounded up to the chunk");
    let held: Vec<f32> = kb.narrow(2, 0, 3).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(held, k.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    let vh: Vec<f32> = vb.narrow(2, 0, 3).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(vh, v.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    assert_eq!(c.len(), 3, "reserve moves no length");
    c.set_len(10).unwrap();
    assert_eq!(c.len(), 10);
    assert!(c.set_len(13).is_err());
    let mut cat = KvCache::new(KvMode::F16);
    assert!(cat.reserve(4).is_err(), "only the Grow store is written in place");
}

use super::*;

fn dense(d_out: usize, d_in: usize) -> Proj {
    let w = Tensor::zeros((d_out, d_in), DType::F32, &Device::Cpu).expect("weights");
    Proj::Dense(Linear::new(w, None))
}

/// **T6.** A projection handed a rotation that is not its own refuses,
/// naming itself.
///
/// This is the one wiring mistake the value-based design leaves open, and
/// it is three lines wide in `forward_cached`: the q/k/v group and the
/// gate/up group both take a `hidden_size`-wide activation, so passing one
/// group's [`Rotated`] to the other type-checks, runs, and returns finite
/// plausible wrong numbers. Lives in the crate rather than in
/// `tests/rotplan.rs` because [`Rotated`]'s fields are private — which is
/// the point: outside this module the only way to get one is to rotate an
/// activation.
#[test]
fn a_rotated_activation_of_the_wrong_key_is_refused() {
    let p = dense(4, 3);
    let x = Tensor::zeros((1usize, 3usize), DType::F32, &Device::Cpu).expect("x");
    let foreign = Rotated {
        key: Some((2560, 0xdead_beef)),
        t: x.clone(),
    };
    let e = p
        .forward_with(&foreign, &x)
        .expect_err("a foreign rotation must be refused");
    let msg = format!("{e}");
    assert!(msg.contains("2560"), "the message must cite the key: {msg}");
    assert!(
        msg.contains("dense projection"),
        "the message must name the site: {msg}"
    );
}

/// `prepare` on a dense projection is a strict no-op: the same tensor, no
/// key, no allocation. Every path that is not the fused one goes through
/// it, including the forward pass `bin/oracle` pins at `|Δhidden| = 0`.
#[test]
fn preparing_a_dense_projection_moves_nothing() {
    let p = dense(4, 3);
    let x = Tensor::arange(0f32, 3f32, &Device::Cpu)
        .expect("x")
        .reshape((1usize, 3usize))
        .expect("shape");
    let r = p.prepare(&x).expect("dense prepare");
    assert_eq!(r.key(), None);
    assert_eq!(
        x.id(),
        r.t.id(),
        "the dense prepare must return the tensor itself, not a copy"
    );
    assert!(p.forward_with(&r, &x).is_ok());
}

/// The row split and its inverse, on shapes a CPU tensor can carry.
///
/// `group_forward`'s rotated branch is reachable only with a fused
/// projection, so no test on this machine runs it end to end. Its shape
/// bookkeeping — which row goes where, and what rank comes back — is the
/// part that can be silently wrong, and it is pinned here: a transposed
/// row order or a `reshape` on the wrong dims would return finite,
/// plausible, wrong tokens on a card, with nothing red anywhere.
#[test]
fn the_row_split_is_undone_by_the_regroup() {
    for dims in [vec![1usize, 4], vec![1, 3, 4], vec![2, 3, 4]] {
        let n: usize = dims.iter().product();
        let x = Tensor::arange(0f32, n as f32, &Device::Cpu)
            .expect("x")
            .reshape(dims.clone())
            .expect("shape");
        let rows = row_views(&x).expect("split");
        let expected_rows: usize = dims[..dims.len() - 1].iter().product();
        assert_eq!(rows.len(), expected_rows);
        for r in &rows {
            assert_eq!(r.dims(), &[1, *dims.last().expect("rank >= 1")]);
        }
        let back = regroup(&rows, &dims, *dims.last().expect("rank >= 1")).expect("regroup");
        assert_eq!(back.dims(), dims.as_slice());
        assert_eq!(
            back.flatten_all().expect("flat").to_vec1::<f32>().expect("vals"),
            x.flatten_all().expect("flat").to_vec1::<f32>().expect("vals"),
            "dims {dims:?}: the row split is not returned identically"
        );

        // The order is load-bearing, and this says so: reversing the rows
        // must change the answer whenever there is more than one.
        if expected_rows > 1 {
            let mut rev = rows.clone();
            rev.reverse();
            let other =
                regroup(&rev, &dims, *dims.last().expect("rank >= 1")).expect("regroup");
            assert_ne!(
                other.flatten_all().expect("f").to_vec1::<f32>().expect("v"),
                back.flatten_all().expect("f").to_vec1::<f32>().expect("v"),
                "dims {dims:?}: the row order changes nothing, the test proves nothing"
            );
        }
    }
}

/// The width changes between the two halves: `d_in` in, `d_out` out.
#[test]
fn the_regroup_carries_the_output_width() {
    let rows: Vec<Tensor> = (0..3)
        .map(|r| {
            Tensor::arange(0f32, 5f32, &Device::Cpu)
                .expect("row")
                .affine(1.0, r as f64)
                .expect("offset")
                .reshape((1usize, 5usize))
                .expect("shape")
        })
        .collect();
    let out = regroup(&rows, &[1, 3, 7], 5).expect("regroup");
    assert_eq!(out.dims(), &[1, 3, 5]);
}

/// A group of projections in the natural basis is driven identically in
/// both modes — the branch that keeps every dense path byte-identical.
#[test]
fn a_group_without_rotation_ignores_the_mode() {
    let a = dense(2, 3);
    let b = dense(5, 3);
    let x = Tensor::arange(0f32, 6f32, &Device::Cpu)
        .expect("x")
        .reshape((1usize, 2usize, 3usize))
        .expect("shape");
    for share in [RotShare::Off, RotShare::On] {
        let out = group_forward(&[&a, &b], &x, share).expect("group");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].dims(), &[1, 2, 2]);
        assert_eq!(out[1].dims(), &[1, 2, 5]);
    }

    // And it does not go through the row loop at all — observable, because
    // the row loop carries `MAX_ROWS`, which a dense `Linear` has no reason
    // to obey. A refactor that routed dense projections through the rotated
    // branch "for uniformity" would break scoring on any window wider than
    // 256, and would otherwise be invisible: the numbers would be right.
    let wide = Tensor::zeros((1usize, MAX_ROWS + 44, 3usize), DType::F32, &Device::Cpu)
        .expect("wide");
    let out = group_forward(&[&a], &wide, RotShare::On).expect("dense on a wide window");
    assert_eq!(out[0].dims(), &[1, MAX_ROWS + 44, 2]);
}

/// The padding of the fixed window must be EXACTLY inert through the
/// whole attention arithmetic: same scores→softmax→ctx bytes over `len`
/// keys as over `w` keys whose tail is zeros behind −inf. `exp(−inf) = 0`
/// and `x + 0.0 = x` are what make this bitwise, not approximate — if
/// either stops holding (a mask built at a finite floor, say), this test
/// is the tripwire. It exercises `build_causal_mask` itself, not a
/// re-derivation.
#[test]
fn wide_mask_padding_is_exactly_inert() {
    let dev = Device::Cpu;
    let (b, h, l, hd, offset, len, w) = (1usize, 2, 1, 8, 3, 4, 16);
    let q = Tensor::from_vec(
        (0..b * h * l * hd).map(|i| (i as f32).sin()).collect::<Vec<f32>>(),
        (b, h, l, hd),
        &dev,
    )
    .unwrap();
    let mk = |n: usize, sgn: f32| {
        Tensor::from_vec(
            (0..b * h * n * hd).map(|i| sgn * (i as f32).cos()).collect::<Vec<f32>>(),
            (b, h, n, hd),
            &dev,
        )
        .unwrap()
    };
    let (k, v) = (mk(len, 1.0), mk(len, -1.0));
    let zpad = Tensor::zeros((b, h, w - len, hd), DType::F32, &dev).unwrap();
    let kw = Tensor::cat(&[&k, &zpad], 2).unwrap().contiguous().unwrap();
    let vw = Tensor::cat(&[&v, &zpad], 2).unwrap().contiguous().unwrap();
    let attn = |k: &Tensor, v: &Tensor, width: usize| -> Vec<f32> {
        let mask = build_causal_mask(b, l, offset, width, DType::F32, &dev).unwrap();
        let scores = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
        let scores = scores.broadcast_add(&mask).unwrap();
        let p = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        p.matmul(v).unwrap().flatten_all().unwrap().to_vec1().unwrap()
    };
    assert_eq!(attn(&k, &v, len), attn(&kw, &vw, w), "the padding is not inert");
}

/// The f32 logits of a chunked prefill, then of one decode step per
/// token of `feed`. Feeding fixed tokens rather than each arm's own
/// greedy choice compares every arm on the same inputs: a near tie that
/// one rounding flips would otherwise send two arms down different texts.
fn logit_trace(m: &Qwen3, prompt: &[u32], chunk: usize, feed: &[u32]) -> Vec<Vec<f32>> {
    let mut c = m.fresh_caches();
    let mut out = vec![m.prefill_chunked(prompt, chunk, &mut c).unwrap()];
    for (offset, &t) in (prompt.len()..).zip(feed) {
        out.push(m.step_logits(t, offset, &mut c).unwrap());
    }
    out.iter().map(|l| l.flatten_all().unwrap().to_vec1::<f32>().unwrap()).collect()
}

fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn same_bits(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

/// Bit equality between two computations that reach candle's CPU matmul
/// with different shapes or layouts (the `Shared` form, a `Grow` view).
/// Strict on macOS, where it was measured. Elsewhere the gemm may order a
/// reduction by shape, which is how three tests fell on the Linux runner
/// on 2026-11-07 (`f43a774`, `abdbaae`), so the two sides are held to
/// `bound` instead. Same shape, same kernel: use [`same_bits`].
fn same_bits_across_shapes(a: &[Vec<f32>], b: &[Vec<f32>], bound: f32) -> bool {
    if cfg!(target_os = "macos") {
        same_bits(a, b)
    } else {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| x.len() == y.len())
            && max_abs_diff(a, b) <= bound
    }
}

/// The three attention forms over the stores that hand attention exactly
/// the history. Against `Cat` with expanded heads, every published path:
/// the `Grow` store alone and the `Shared` form are bit for bit, in the
/// chunked prefill (a masked chunk at an offset), in 20 decode steps and
/// in the uncached full-window forward; `Grouped` changes the matmul
/// shape, and is held to a bound on the logits instead, about ten times
/// the largest difference measured on 2026-11-01. It is not held to the
/// same greedy text: on this random model, whose logits tie often, one
/// rounding difference flipped a token at the ninth step.
/// Three head layouts: 4 over 2; 8 over 2, groups of 4 as in Qwen3-4B,
/// where a group and a key head no longer have the same count; and 10
/// over 2, groups of 5 as in Qwen3-14B (40 over 8), an odd group size.
#[test]
fn attention_forms_agree() {
    use crate::kvq::{AttnHeads, KvStore};
    for (dtype, bound, n_heads) in [
        (DType::F32, 6e-7f32, 4),
        (DType::F16, 4e-3, 4),
        (DType::F32, 6e-7, 8),
        (DType::F16, 4e-3, 8),
        (DType::F32, 6e-7, 10),
        (DType::F16, 4e-3, 10),
    ] {
        let mut m = tiny_model_heads(dtype, n_heads, 2);
        let prompt = [3u32, 17, 42, 7, 9, 11];
        let window = Tensor::from_slice(&[5u32, 60, 11, 90, 2, 33, 71, 8, 13], (1, 9), &Device::Cpu).unwrap();
        let full = |m: &Qwen3| {
            vec![m.logits(&window, &mut NoCapture).unwrap().flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap()]
        };
        let feed = [5u32, 60, 11, 90, 2, 33, 71, 8, 13, 44, 1, 96, 27, 64, 38, 19, 80, 6, 52, 23];
        let reference = (logit_trace(&m, &prompt, 4, &feed), full(&m));
        for (store, heads, exact) in [
            (KvStore::Grow(4), AttnHeads::Expanded, true),
            (KvStore::Cat, AttnHeads::Shared, true),
            (KvStore::Grow(4), AttnHeads::Shared, true),
            (KvStore::Grow(1), AttnHeads::Shared, true),
            (KvStore::Cat, AttnHeads::Grouped, false),
            (KvStore::Grow(4), AttnHeads::Grouped, false),
        ] {
            m.set_kv_store(store);
            m.set_attn_heads(heads);
            let arm = format!("{dtype:?} {n_heads}/2 {} {}", store.label(), heads.label());
            let got = (logit_trace(&m, &prompt, 4, &feed), full(&m));
            if exact {
                assert!(same_bits_across_shapes(&reference.0, &got.0, bound), "{arm}: cached logits not bit for bit");
                assert!(same_bits_across_shapes(&reference.1, &got.1, bound), "{arm}: full-window logits not bit for bit");
            } else {
                let d = max_abs_diff(&reference.0, &got.0).max(max_abs_diff(&reference.1, &got.1));
                println!("{arm}: max |dlogit| {d:e}");
                assert!(d <= bound, "{arm}: max |dlogit| {d:e} above {bound:e}");
                // Liveness, on macOS only: there the grouped matmul rounds
                // differently from the expanded one, so equal bits would
                // mean the arm never took it. On x86-64 Linux candle's CPU
                // gemm returns the same bits for both layouts (CI,
                // 2026-11-07, F32 4/2), so the check would fail on a
                // correct grouped path; the bound above still holds there.
                #[cfg(target_os = "macos")]
                assert!(d > 0.0, "{arm}: bit for bit, so this arm is not testing the grouped matmul");
            }
        }
    }
}

/// What the unexpanded forms hand their matmuls: the transposed output of
/// a projection is laid out again, rows row-major; a strided view of a
/// `Grow` buffer keeps its storage and strides, so no history is copied.
#[test]
fn rows_contiguous_copies_only_what_a_matmul_refuses() {
    let dev = Device::Cpu;
    let x = Tensor::arange(0f32, 24.0, &dev).unwrap().reshape((1, 4, 2, 3)).unwrap();
    let t = x.transpose(1, 2).unwrap();
    assert_eq!(t.stride(), &[24, 3, 6, 1]);
    let r = rows_contiguous(&t).unwrap();
    assert_eq!(r.stride(), &[24, 12, 3, 1]);
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(flat(&r), flat(&t));
    let buf = Tensor::arange(0f32, 48.0, &dev).unwrap().reshape((1, 2, 8, 3)).unwrap();
    let view = buf.narrow(2, 0, 5).unwrap();
    let r = rows_contiguous(&view).unwrap();
    assert_eq!(r.stride(), &[48, 24, 3, 1], "a Grow view must not be copied");
    assert_eq!(flat(&r), flat(&view));
}

/// The teacher-forced decode scores the next true token, as the prefill
/// does: on one arm twice it agrees with itself bit for bit and with
/// [`Qwen3::window_nll`] to rounding. Against the grouped form it moves by
/// rounding only, and against the shared form not at all on the CPU.
#[test]
fn paired_decode_scores_what_the_window_scores() {
    use crate::kvq::{AttnHeads, KvStore};
    let mut m = tiny_model_heads(DType::F32, 8, 2);
    let tokens = [5u32, 60, 11, 90, 2, 33, 71, 8, 13, 44, 1, 96, 27, 64, 38, 19];
    let published = (KvStore::Cat, AttnHeads::Expanded);
    let same = m.decode_nll_paired(&tokens, [published, published]).unwrap();
    assert_eq!(same.count, tokens.len() - 1);
    assert_eq!(same.nll[0].to_bits(), same.nll[1].to_bits());
    assert_eq!((same.top1_agree, same.max_dlogit), (same.count, 0.0));
    m.set_kv_store(KvStore::Cat);
    m.set_attn_heads(AttnHeads::Expanded);
    let (window, n) = m.window_nll(&tokens, &mut NoCapture).unwrap();
    assert_eq!(n, same.count);
    let rel = (same.nll[0] / window - 1.0).abs();
    assert!(rel < 1e-5, "decode against prefill: relative {rel:e}");
    let shared = m.decode_nll_paired(&tokens, [published, (KvStore::Grow(4), AttnHeads::Shared)]).unwrap();
    // Bit for bit on macOS; elsewhere the shared form's matmul shape may
    // round differently, so the grouped form's bounds hold instead (see
    // `same_bits_across_shapes`).
    if cfg!(target_os = "macos") {
        assert_eq!(shared.nll[0].to_bits(), shared.nll[1].to_bits());
        assert_eq!(shared.max_dlogit, 0.0);
    } else {
        assert!(shared.max_dlogit < 6e-7, "{shared:?}");
        assert!((shared.nll[1] / shared.nll[0] - 1.0).abs() < 1e-6, "{shared:?}");
    }
    let grouped = m.decode_nll_paired(&tokens, [published, (KvStore::Grow(4), AttnHeads::Grouped)]).unwrap();
    assert!(grouped.max_dlogit < 6e-7, "{grouped:?}");
    // Above zero on macOS, where the grouped matmul rounds differently;
    // x86-64 Linux returns the same bits (see `attention_forms_agree`).
    #[cfg(target_os = "macos")]
    assert!(grouped.max_dlogit > 0.0, "{grouped:?}");
    assert!((grouped.nll[1] / grouped.nll[0] - 1.0).abs() < 1e-6, "{grouped:?}");
    assert!(m.decode_nll_paired(&tokens[..1], [published, published]).is_err());
}

/// At every append, across chunk boundaries and appends wider than a
/// chunk, the `Grow` store hands attention the very history `Cat` does,
/// in buffers sized to a whole number of chunks.
#[test]
fn grow_store_holds_what_cat_holds() {
    use crate::kvq::{KvMode, KvStore};
    let dev = Device::Cpu;
    let mut cat = KvCache::with_store(KvMode::F16, KvStore::Cat);
    let mut grow = KvCache::with_store(KvMode::F16, KvStore::Grow(4));
    let mut seed = 1u32;
    for l in [5usize, 1, 1, 3, 1, 1, 1, 1, 9, 1, 2] {
        let mut step = || {
            let n = 2 * l * 3;
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32
                })
                .collect();
            Tensor::from_vec(v, (1, 2, l, 3), &dev).unwrap()
        };
        let (k, v) = (step(), step());
        let (ck, cv) = cat.append(&k, &v).unwrap();
        let (gk, gv) = grow.append(&k, &v).unwrap();
        for (c, g) in [(ck, gk), (cv, gv)] {
            assert_eq!(c.dims(), g.dims());
            let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(flat(&c), flat(&g), "after an append of {l}");
        }
        let cap = grow.k.as_ref().unwrap().dim(2).unwrap();
        assert!(cap.is_multiple_of(4) && cap >= grow.len && cap < grow.len + 4, "capacity {cap} for {}", grow.len);
    }
    let mut zero = KvCache::with_store(KvMode::F16, KvStore::Grow(0));
    let k = Tensor::zeros((1, 2, 1, 3), DType::F32, &dev).unwrap();
    assert!(format!("{}", zero.append(&k, &k).unwrap_err()).contains("chunk"));
}

/// Decoding again after a truncation to the prompt gives the logits of
/// the first decode, bit for bit, on every store: `LLVQ_GQA_AB` times
/// several decodes after one prefill on the strength of this.
#[test]
fn a_truncated_cache_decodes_like_a_fresh_one() {
    use crate::kvq::{AttnHeads, KvStore};
    let mut m = tiny_model(DType::F32);
    let prompt = [3u32, 17, 42, 7, 9, 11];
    for (store, heads) in [
        (KvStore::Cat, AttnHeads::Expanded),
        (KvStore::Grow(4), AttnHeads::Grouped),
    ] {
        m.set_kv_store(store);
        m.set_attn_heads(heads);
        let mut c = m.fresh_caches();
        let first = argmax_last(&m.prefill_chunked(&prompt, 4, &mut c).unwrap()).unwrap();
        let decode = |c: &mut [KvCache]| -> Vec<Vec<f32>> {
            let mut next = first;
            (prompt.len()..prompt.len() + 7)
                .map(|offset| {
                    let l = m.step_logits(next, offset, c).unwrap();
                    next = argmax_last(&l).unwrap();
                    l.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                })
                .collect()
        };
        let a = decode(&mut c);
        for cache in c.iter_mut() {
            cache.truncate(prompt.len()).unwrap();
        }
        let b = decode(&mut c);
        assert!(same_bits(&a, &b), "{}: a truncated cache decodes differently", store.label());
        assert!(c[0].truncate(prompt.len() + 100).is_err(), "{}: truncating up must refuse", store.label());
    }
}

/// A decode step on the `Cat` store skips its mask, which is all zeros.
/// This replays each step twice from the same cache state, once through
/// `hidden_cached` (no mask) and once with the zero mask added explicitly
/// as before, and requires the hidden states to agree bit for bit.
#[test]
fn a_decode_step_without_its_zero_mask_is_bit_identical() {
    for dtype in [DType::F32, DType::F16] {
        let model = tiny_model(dtype);
        let dev = Device::Cpu;
        let prompt = [3u32, 17, 42, 7];
        let input = Tensor::from_slice(&prompt, (1, prompt.len()), &dev).unwrap();
        let mut a = model.fresh_caches();
        let mut b = model.fresh_caches();
        model.hidden_cached(&input, 0, &mut a, &mut NoCapture).unwrap();
        model.hidden_cached(&input, 0, &mut b, &mut NoCapture).unwrap();
        for (offset, tok) in (prompt.len()..).zip([5u32, 60, 11, 90, 2, 33, 71, 8]) {
            assert!(
                model.step_mask(1, 1, offset).unwrap().is_none(),
                "the decode step must take the unmasked path, or this test certifies nothing"
            );
            let x = Tensor::from_slice(&[tok], (1, 1), &dev).unwrap();
            let ha = model.hidden_cached(&x, offset, &mut a, &mut NoCapture).unwrap();
            let mask = model.causal_mask_offset(1, 1, offset).unwrap();
            let mut hb = model.embed.forward(&x).unwrap();
            for (i, blk) in model.blocks.iter().enumerate() {
                hb = blk
                    .forward_cached(&hb, &model.rotary, Some(&mask), offset, &mut b[i], i, &mut NoCapture)
                    .unwrap();
            }
            let hb = model.norm.forward(&hb).unwrap();
            let bits = |t: &Tensor| -> Vec<u32> {
                t.flatten_all()
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect()
            };
            assert_eq!(bits(&ha), bits(&hb), "{dtype:?}, step at offset {offset}");
        }
    }
}

/// `generate_timed` decodes what `generate` decodes, whole or in chunks.
#[test]
fn a_timed_generation_returns_the_tokens_of_generate() {
    let model = tiny_model(DType::F32);
    let prompt: Vec<u32> = (0..13u32).map(|i| (i * 7 + 3) % 97).collect();
    let want = model.generate(&prompt, 9, &mut NoCapture).unwrap();
    let whole = model.generate_timed(&prompt, 9, MAX_PREFILL_ROWS).unwrap();
    assert_eq!(whole.tokens, want, "one chunk is generate, call for call");
    let chunked = model.generate_timed(&prompt, 9, 4).unwrap();
    assert_eq!(chunked.tokens, want, "a prompt in chunks of 4 decodes the same tokens");
    assert!(whole.prompt_s > 0.0 && whole.decode_s > 0.0);
}

/// `prefill_next` in chunks leaves the caches and the first token of an
/// unchunked prompt, and the decode after it is `generate`'s: what the 4E
/// driver needs at an 8,192-token context, at chunk sizes a host reaches.
#[test]
fn a_chunked_prefill_next_is_the_prefill_of_generate() {
    // 4E's store: Grow(256), whose count `len` tracks.
    let mut model = tiny_model(DType::F32);
    model.set_kv_store(crate::kvq::KvStore::Grow(crate::kvq::GROW_CHUNK));
    let prompt: Vec<u32> = (0..13u32).map(|i| (i * 11 + 5) % 97).collect();
    let want = model.generate(&prompt, 6, &mut NoCapture).unwrap();
    for chunk in [1usize, 4, 12, 13, MAX_PREFILL_ROWS] {
        let mut caches = model.fresh_caches();
        let first = model.prefill_next_in(&prompt, chunk, &mut caches).unwrap();
        assert!(caches.iter().all(|c| c.len() == prompt.len()), "chunk {chunk}: the caches hold the prompt");
        let got = model.decode_greedy(first, prompt.len(), 6, &mut caches).unwrap();
        assert_eq!(got, want, "chunk {chunk}: the tokens of generate");
    }
    let mut caches = model.fresh_caches();
    assert_eq!(model.prefill_next(&prompt, &mut caches).unwrap(), want[0], "the served chunk is one call here");
}

/// A CPU lattice projection over a dense weight: rotation key `(0, 0)`,
/// the identity as its rotation, `k` rows a launch. The dense arithmetic,
/// through the rotated branch of `group_forward` and its row budget.
struct HostLattice {
    w: Tensor,
    k: usize,
}

impl crate::device::LatticeProj for HostLattice {
    fn name(&self) -> &str {
        "fake.lattice"
    }
    fn d_out(&self) -> usize {
        self.w.dim(0).unwrap()
    }
    fn d_in(&self) -> usize {
        self.w.dim(1).unwrap()
    }
    fn rotation(&self) -> Option<RotKey> {
        Some((0, 0))
    }
    fn prepare(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
    fn prepare_rows(&self, xs: &Tensor, _rows: usize) -> Result<Tensor> {
        Ok(xs.clone())
    }
    fn matvec(&self, xr: &Tensor, out_dims: &[usize]) -> Result<Tensor> {
        let y = xr.reshape((1, self.d_in()))?.matmul(&self.w.t()?)?;
        let mut d = out_dims.to_vec();
        *d.last_mut().expect("rank >= 1") = self.d_out();
        y.reshape(d)
    }
    fn rows_per_launch(&self) -> usize {
        self.k
    }
    fn matvec_rows(&self, xr: &Tensor, n: usize) -> Result<Tensor> {
        assert!((1..=self.k).contains(&n), "{n} rows for a launch of {}", self.k);
        xr.matmul(&self.w.t()?)
    }
}

/// The tiny model with every `q_proj` served by [`HostLattice`] at `k`
/// rows a launch, and RoPE tables for 512 positions.
fn tiny_model_lattice_q(k: usize) -> Qwen3 {
    let mut m = tiny_model_long(DType::F32, 4, 2, 16, 512);
    for b in &mut m.blocks {
        let Proj::Dense(l) = &b.q_proj else { panic!("the tiny model's q_proj is dense") };
        b.q_proj = Proj::Lattice(std::sync::Arc::new(HostLattice { w: l.weight().clone(), k }));
    }
    m
}

/// A turn longer than the model takes in one call goes in chunks the
/// model takes. On the served Metal path a projection loops one launch a
/// row and refuses more than 256 rows a call; until 2026-11-07
/// `Session::feed` cut at 4,096, so a 257-token turn failed there.
#[test]
fn a_session_feeds_a_long_turn_in_chunks_the_model_takes() {
    let dense = tiny_model_long(DType::F32, 4, 2, 16, 512);
    assert_eq!(dense.prefill_rows(), MAX_PREFILL_ROWS, "nothing rotated: the batched budget");
    assert_eq!(tiny_model_lattice_q(4).prefill_rows(), MAX_PREFILL_ROWS, "four rows a launch");
    let looped = tiny_model_lattice_q(1);
    assert_eq!(looped.prefill_rows(), MAX_ROWS, "one launch a row: the served Metal path");

    let ids: Vec<u32> = (0..300u32).map(|i| (i * 13 + 7) % 97).collect();
    let x = Tensor::from_slice(&ids, (1, ids.len()), &Device::Cpu).unwrap();
    let e = looped
        .hidden_cached(&x, 0, &mut looped.fresh_caches(), &mut NoCapture)
        .expect_err("300 rows in one call of the looped path");
    assert!(e.to_string().contains("more than 256"), "{e}");

    let mut s = crate::session::Session::new(&looped);
    let h = s.feed(&looped, &ids).expect("a 300-token turn on the looped path");
    assert_eq!(s.len(), ids.len(), "every position held");
    let mut d = crate::session::Session::new(&dense);
    let hd = d.feed(&dense, &ids).expect("the dense model takes it in one call");
    let next = |m: &Qwen3, h: &Tensor| argmax_last(&m.logits_last(h).unwrap()).unwrap();
    assert_eq!(next(&looped, &h), next(&dense, &hd), "the dense model's next token");
}

/// 🚨 The served config's pair reaches the model, BOTH halves of it.
///
/// `served::Served::apply` is the one place `bin/fusedrun`, `bin/mmlu` and
/// `bin/gsm8k` set the store and the form, and a half-wired apply is the
/// defect that would not show: a `Grouped` form on a `Cat` store decodes
/// the same tokens, slowly, and an `Expanded` form on a `Grow` store
/// decodes them too. Only the getters tell them apart, so they are
/// asserted, and then the pair is made to decode so the config cannot name
/// a combination that falls over.
#[test]
fn a_served_config_sets_the_store_and_the_attention_form() {
    use crate::kvq::{AttnHeads, KvStore, GROW_CHUNK};
    let file = |attention: Option<&str>| crate::served::ServedFile {
        layout: "tetra48".into(),
        embed: "q4".into(),
        rot_share: "1".into(),
        fuse: "0".into(),
        kv: "f16".into(),
        attention: attention.map(str::to_string),
        decode: None,
        rows: None,
        note: None,
    };
    let resolve = |attention: Option<&str>| {
        crate::served::Served::of(file(attention), std::path::PathBuf::from("t.json"))
            .expect("resolves")
    };

    let mut m = tiny_model(DType::F32);
    assert_eq!(m.attn_heads(), AttnHeads::Expanded, "the model's default");
    assert_eq!(m.kv_store(), KvStore::Cat);
    let prompt: Vec<u32> = (0..9u32).map(|i| (i * 11 + 5) % 97).collect();
    let want = m.generate(&prompt, 7, &mut NoCapture).unwrap();

    // A config that says nothing leaves the published path in place.
    resolve(None).apply(&mut m, &Device::Cpu).expect("expanded applies");
    assert_eq!((m.attn_heads(), m.kv_store()), (AttnHeads::Expanded, KvStore::Cat));

    // A config that says `grouped` moves both halves, and the pair decodes.
    resolve(Some("grouped")).apply(&mut m, &Device::Cpu).expect("grouped applies");
    assert_eq!(m.attn_heads(), AttnHeads::Grouped, "the form");
    assert_eq!(m.kv_store(), KvStore::Grow(GROW_CHUNK), "and the store with it");
    let got = m.generate(&prompt, 7, &mut NoCapture).unwrap();
    assert_eq!(got, want, "on the CPU in f32 the two forms agree token for token");

    // `Served::serve`, the door of `bin/gsm8k` and `bin/chat`: the same
    // pair, then the engine. A grouped config on a fresh model leaves it
    // grouped on `Grow` with the eager engine; an expanded one puts it
    // back. Before 2026-11-06 `bin/chat` printed the provenance and
    // skipped this step, so a grouped config ran the expanded form.
    let mut m = tiny_model(DType::F32);
    let dec = resolve(Some("grouped")).serve(&mut m, &Device::Cpu).expect("grouped serves");
    assert_eq!((m.attn_heads(), m.kv_store()), (AttnHeads::Grouped, KvStore::Grow(GROW_CHUNK)));
    assert!(!dec.is_persistent() && dec.max_positions().is_none(), "the eager engine");
    resolve(None).serve(&mut m, &Device::Cpu).expect("expanded serves");
    assert_eq!((m.attn_heads(), m.kv_store()), (AttnHeads::Expanded, KvStore::Cat));
}

/// The `kernel` form on the CPU, where its kernel is the canonical replay
/// the card is checked against. Against the grouped form on the same
/// `Grow` store: a prefill chunk of more than 8 rows takes the grouped
/// path and is bit for bit; chunks and decode steps of 8 rows or fewer go
/// through the kernel, whose f32 softmax differs from the grouped f16
/// one, and are held to a bound, about ten times the largest difference
/// measured on 2026-11-04, and to differing at all.
#[test]
fn the_kernel_form_attends_through_the_replay() {
    use crate::kvq::{AttnHeads, KvStore, GROW_CHUNK};
    let mut m = tiny_model_hd(DType::F16, 8, 2, 128);
    let prompt = [3u32, 17, 42, 7, 9, 11, 5, 60, 11, 90, 2];
    let feed = [5u32, 60, 11, 90, 2, 33, 71, 8, 13, 44, 1, 96];
    m.set_kv_store(KvStore::Grow(GROW_CHUNK));
    m.set_attn_heads(AttnHeads::Grouped);
    let grouped_long = logit_trace(&m, &prompt, 16, &feed);
    let grouped_short = logit_trace(&m, &prompt, 4, &feed);

    m.set_attn_heads(AttnHeads::Kernel);
    let mut c = m.fresh_caches();
    let e = m.prefill_chunked(&prompt, 4, &mut c).expect_err("no kernel installed");
    assert!(e.to_string().contains("no kernel installed"), "{e}");

    m.set_attn_kernel(Some(crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).expect("CPU kernel")));
    // At the served threshold no position of this window reaches the
    // kernel: every step is the grouped form's, bit for bit.
    assert_eq!(m.kernel_from(), crate::kvq::KERNEL_FROM);
    assert!(same_bits(&logit_trace(&m, &prompt, 16, &feed), &grouped_long), "below the threshold");
    assert!(same_bits(&logit_trace(&m, &prompt, 4, &feed), &grouped_short), "below the threshold");
    m.set_kernel_from(0);
    let long = logit_trace(&m, &prompt, 16, &feed);
    assert!(
        same_bits(&grouped_long[..1], &long[..1]),
        "a chunk of 11 rows takes the grouped path: its logits are the grouped form's"
    );
    let d_decode = max_abs_diff(&grouped_long[1..], &long[1..]);
    let short = logit_trace(&m, &prompt, 4, &feed);
    let d_short = max_abs_diff(&grouped_short, &short);
    println!("kernel against grouped: decode max |dlogit| {d_decode:e}, chunks of 4 {d_short:e}");
    for (what, d) in [("decode", d_decode), ("chunks of 4", d_short)] {
        assert!(d > 0.0, "{what}: bit for bit with grouped, so the kernel did not run");
        assert!(d <= 5e-3, "{what}: max |dlogit| {d:e}");
    }

    // The served door installs it: the form, the store and the kernel.
    let mut m = tiny_model_hd(DType::F16, 8, 2, 128);
    let served_file_kernel = || crate::served::ServedFile {
        layout: "tetra48".into(),
        embed: "q4".into(),
        rot_share: "1".into(),
        fuse: "0".into(),
        kv: "f16".into(),
        attention: Some("kernel".into()),
        decode: None,
        rows: None,
        note: None,
    };
    let f = served_file_kernel();
    let served = crate::served::Served::of(f, "kernel.json".into()).expect("kernel parses");
    served.apply(&mut m, &Device::Cpu).expect("kernel applies on the CPU");
    assert_eq!(m.attn_heads(), AttnHeads::Kernel);
    assert_eq!(m.kv_store(), KvStore::Grow(GROW_CHUNK));
    assert_eq!(m.kernel_from(), crate::kvq::KERNEL_FROM, "the door keeps the form's threshold");
    assert!(served.provenance().contains("attention kernel from position 1280 on grow(256)"));
    m.set_kernel_from(0);
    assert_eq!(logit_trace(&m, &prompt, 4, &feed), short, "the door's kernel is the same arithmetic");
    let e = served.check_device(true).expect_err("the kernel form stays refused on Metal");
    assert!(e.contains("no T8 of the form on Metal"), "{e}");

    // The persistent decode: refused on a model that is not the 4B, before
    // anything moves; past that check, every decode row goes to the
    // kernel (threshold 0), the arithmetic 4E owes its tokens to.
    let p = crate::served::Served::of(
        crate::served::ServedFile { decode: Some("persistent".into()), ..served_file_kernel() },
        "4e.json".into(),
    )
    .expect("4E parses");
    let mut toy = tiny_model_hd(DType::F16, 8, 2, 128);
    let e = p.apply(&mut toy, &Device::Cpu).expect_err("a toy is not the 4B");
    assert!(e.contains("4B"), "{e}");
    assert_eq!(toy.kernel_from(), crate::kvq::KERNEL_FROM, "a refused apply moves nothing");
    p.apply_form(&mut toy, &Device::Cpu).expect("the form applies");
    assert_eq!((toy.attn_heads(), toy.kernel_from()), (AttnHeads::Kernel, 0));
    assert_eq!(logit_trace(&toy, &prompt, 4, &feed), short, "threshold 0: the kernel on every decode row");
    served.apply(&mut toy, &Device::Cpu).expect("back to the eager kernel config");
    assert_eq!(toy.kernel_from(), crate::kvq::KERNEL_FROM, "an eager config restores the constant");
    // f32 is refused at the first decode step, far below the threshold;
    // a prefill chunk takes the grouped path in any dtype.
    let mut wide = tiny_model_hd(DType::F32, 8, 2, 128);
    served.apply(&mut wide, &Device::Cpu).expect("kernel applies on the CPU");
    let mut c = wide.fresh_caches();
    wide.prefill_chunked(&prompt, 16, &mut c).expect("an 11-row chunk is grouped");
    let e = wide.step_logits(feed[0], prompt.len(), &mut c).expect_err("f32 decode");
    assert!(e.to_string().contains("f16 only"), "{e}");
    // A head dimension the kernel does not compile is refused at the door.
    let mut small = tiny_model(DType::F16);
    let e = served.apply(&mut small, &Device::Cpu).expect_err("head dim 16");
    assert!(e.contains("head dim 16"), "{e}");
}

/// The T8 harness of the kernel form, on the CPU replay: the decode path
/// pairs the grouped and the kernel arms in lockstep and scores both, and
/// the prefill path of the two arms is the same forward (no call of more
/// than 8 rows reaches the kernel), bit for bit.
#[test]
fn the_kernel_arm_runs_in_the_paired_decode() {
    use crate::kvq::{AttnHeads, KvStore, GROW_CHUNK};
    let mut m = tiny_model_hd(DType::F16, 8, 2, 128);
    m.set_attn_kernel(Some(crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).expect("CPU kernel")));
    m.set_kernel_from(8);
    let win = [3u32, 17, 42, 7, 9, 11, 5, 60, 11, 90, 2, 33, 71, 8, 13, 44, 1, 96, 27, 64];
    let g = (KvStore::Grow(GROW_CHUNK), AttnHeads::Grouped);
    let k = (KvStore::Grow(GROW_CHUNK), AttnHeads::Kernel);
    let d = m.decode_nll_split(&win, [g, k], m.kernel_from()).expect("paired decode");
    assert_eq!(d.count, win.len() - 1);
    assert_eq!(d.below_differ, 0, "positions 0..8 are the grouped form's, bit for bit");
    assert_eq!(d.tail_count, win.len() - 1 - 8, "positions 8..19 scored past the split");
    assert!(d.tail_top1_agree <= d.tail_count && d.top1_agree <= d.count);
    assert!(d.max_dlogit > 0.0, "bit for bit: the kernel did not run");
    assert!(d.max_dlogit <= 5e-3, "max |dlogit| {:e}", d.max_dlogit);
    let rel = (d.nll[1] - d.nll[0]).abs() / d.nll[0];
    assert!(rel < 1e-3, "nll {} against {}", d.nll[1], d.nll[0]);
    let tail = (d.tail_nll[1] - d.tail_nll[0]).abs() / d.tail_nll[0];
    assert!(tail > 0.0 && tail < 1e-3, "tail nll {} against {}", d.tail_nll[1], d.tail_nll[0]);
    // The whole difference lies past the split: below it the arms agree.
    assert_eq!((d.nll[1] - d.tail_nll[1]).to_bits(), (d.nll[0] - d.tail_nll[0]).to_bits());
    // A split past the threshold sees the kernel's steps, each one.
    let late = m.decode_nll_split(&win, [g, k], 12).expect("paired decode");
    assert_eq!(late.below_differ, 4, "positions 8, 9, 10 and 11 are the kernel's");
    assert_eq!(late.tail_count, win.len() - 1 - 12);
    // Split 0 is the unsplit call.
    let whole = m.decode_nll_split(&win, [g, k], 0).expect("paired decode");
    assert_eq!(whole, m.decode_nll_paired(&win, [g, k]).expect("paired decode"));
    assert_eq!((whole.tail_nll, whole.tail_count, whole.below_differ), (whole.nll, whole.count, 0));
    let mut pre = |heads| {
        m.set_kv_store(KvStore::Grow(GROW_CHUNK));
        m.set_attn_heads(heads);
        m.window_nll(&win, &mut NoCapture).unwrap().0
    };
    let (a, b) = (pre(AttnHeads::Grouped), pre(AttnHeads::Kernel));
    assert_eq!(a.to_bits(), b.to_bits(), "the prefill path differs between the two arms");
}

/// The `Kernel` form's dispatch of one call, on the CPU replay. A row at
/// the threshold or past it returns the kernel's bits for that row alone,
/// whatever call carried it (the kernel's rows are invariant); a row
/// below it is the grouped path's over the history it sees, and not the
/// kernel's. Every way a call can sit against the threshold: below, at
/// its first row, straddling it at each row, past it.
#[test]
fn a_call_is_split_at_the_threshold_by_position() {
    use crate::attn::canon;
    let kern = crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).expect("CPU kernel");
    let t = |shape: (usize, usize, usize, usize), seed: u64| {
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        let v: Vec<half::f16> =
            canon::fill(n, seed, 2.0).into_iter().map(half::f16::from_bits).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).expect("tensor")
    };
    let (l, n) = (8usize, 20usize);
    let q = t((1, 8, l, 128), 11);
    let (k, v) = (t((1, 2, n, 128), 12), t((1, 2, n, 128), 13));
    let mask = build_causal_mask(1, l, n - l, n, DType::F16, &Device::Cpu).expect("mask");
    let scale = 1.0 / 128f64.sqrt();
    let rows = |x: &Tensor| -> Vec<Vec<u16>> {
        (0..l)
            .map(|r| {
                let v = x.narrow(2, r, 1).unwrap().flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
                v.into_iter().map(half::f16::to_bits).collect()
            })
            .collect()
    };
    // Each row through the kernel alone, over the history it sees.
    let alone: Vec<Vec<u16>> = (0..l)
        .map(|r| {
            let seen = n - l + r + 1;
            let o = kern
                .attend(
                    &q.narrow(2, r, 1).unwrap().contiguous().unwrap(),
                    &k.narrow(2, 0, seen).unwrap(),
                    &v.narrow(2, 0, seen).unwrap(),
                )
                .unwrap();
            rows(&o.broadcast_as((1, 8, l, 128)).unwrap().contiguous().unwrap())[0].clone()
        })
        .collect();
    let grouped = grouped_attention(&q, &rows_contiguous(&k).unwrap(), &rows_contiguous(&v).unwrap(), Some(&mask), scale)
        .expect("grouped");
    let grouped_rows = rows(&grouped);
    let h2f = |b: u16| half::f16::from_bits(b).to_f32();
    // Thresholds from all-grouped (20) to all-kernel (12 and below).
    for from in [0usize, 12, 13, 15, 19, 20, 64] {
        let got = rows(&kernel_rows(kern.as_ref(), &q, &k, &v, Some(&mask), scale, from).expect("dispatch"));
        for r in 0..l {
            let pos = n - l + r;
            if pos >= from {
                assert_eq!(got[r], alone[r], "from {from}: position {pos} is the kernel's, as alone");
            } else {
                assert_ne!(got[r], alone[r], "from {from}: position {pos} reached the kernel");
                let d = got[r].iter().zip(&grouped_rows[r]).map(|(a, b)| (h2f(*a) - h2f(*b)).abs()).fold(0.0, f32::max);
                assert!(d <= 2e-3, "from {from}: position {pos} is not the grouped path's ({d:e})");
            }
        }
        if from >= n {
            assert_eq!(got, grouped_rows, "a call wholly below the threshold is one grouped call");
        }
    }
    // A decode step, no mask: one row on each side.
    let one = q.narrow(2, l - 1, 1).unwrap();
    let at = rows(&kernel_rows(kern.as_ref(), &one, &k, &v, None, scale, n - 1).unwrap().broadcast_as((1, 8, l, 128)).unwrap().contiguous().unwrap());
    assert_eq!(at[0], alone[l - 1], "position n - 1 at threshold n - 1 is the kernel's");
    let below = kernel_rows(kern.as_ref(), &one, &k, &v, None, scale, n).unwrap();
    let g1 = grouped_attention(&one, &rows_contiguous(&k).unwrap(), &rows_contiguous(&v).unwrap(), None, scale).unwrap();
    let bits = |x: &Tensor| x.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
    assert_eq!(bits(&below), bits(&g1), "position n - 1 at threshold n is the grouped form's decode step");
}

/// The served threshold itself, in a model: a prefill of 1,270 positions
/// (grouped, a chunk of more than 8 rows), then decode steps from 1,270
/// to 1,289 under the grouped and the kernel forms. Below 1,280 every
/// step is bit for bit; from 1,280 on, every step differs, within the
/// bound of the replay test. A threshold one position off, or none, or
/// read from anything but the position, fails here.
#[test]
fn the_served_threshold_is_position_1280() {
    use crate::kvq::{AttnHeads, KvStore, GROW_CHUNK, KERNEL_FROM};
    assert_eq!(KERNEL_FROM, 1280);
    let mut m = tiny_model_long(DType::F16, 8, 2, 128, 1300);
    m.set_attn_kernel(Some(crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).expect("CPU kernel")));
    let mut s = 0x2545f4914f6cdd1du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % 97) as u32
    };
    let prompt: Vec<u32> = (0..1270).map(|_| next()).collect();
    let feed: Vec<u32> = (0..20).map(|_| next()).collect();
    m.set_kv_store(KvStore::Grow(GROW_CHUNK));
    let mut trace = |heads| {
        m.set_attn_heads(heads);
        logit_trace(&m, &prompt, 512, &feed)
    };
    let (g, k) = (trace(AttnHeads::Grouped), trace(AttnHeads::Kernel));
    assert!(same_bits(&g[..1], &k[..1]), "the prefill takes the grouped path");
    for (i, (a, b)) in g[1..].iter().zip(&k[1..]).enumerate() {
        let pos = prompt.len() + i;
        let same = same_bits(std::slice::from_ref(a), std::slice::from_ref(b));
        if pos < KERNEL_FROM {
            assert!(same, "position {pos}: below the threshold, the grouped form's bits");
        } else {
            assert!(!same, "position {pos}: past the threshold, and the kernel did not run");
            let d = max_abs_diff(std::slice::from_ref(a), std::slice::from_ref(b));
            assert!(d <= 5e-3, "position {pos}: max |dlogit| {d:e}");
        }
    }
}

/// A two-layer Qwen3 with deterministic pseudo-random weights, on the CPU.
fn tiny_model(dtype: DType) -> Qwen3 {
    tiny_model_heads(dtype, 4, 2)
}

/// The tiny model with `n_heads` query heads over `n_kv` key heads.
fn tiny_model_heads(dtype: DType, n_heads: usize, n_kv: usize) -> Qwen3 {
    tiny_model_hd(dtype, n_heads, n_kv, 16)
}

/// The tiny model at a head dimension of its own: the attention kernel
/// compiles 128, Qwen3's, and refuses anything else.
fn tiny_model_hd(dtype: DType, n_heads: usize, n_kv: usize, head_dim: usize) -> Qwen3 {
    tiny_model_long(dtype, n_heads, n_kv, head_dim, 64)
}

/// The tiny model with RoPE tables for `max_pos` positions.
fn tiny_model_long(dtype: DType, n_heads: usize, n_kv: usize, head_dim: usize, max_pos: usize) -> Qwen3 {
    use crate::kvq::KvMode;
    use candle_nn::VarBuilder;
    let dev = Device::Cpu;
    let cfg = Config {
        vocab_size: 97,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: n_heads,
        head_dim,
        attention_bias: false,
        num_key_value_heads: n_kv,
        max_position_embeddings: max_pos,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
    };
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &dev);
    let model = Qwen3::new(&cfg, vb, KvMode::F16).expect("model");
    // Deterministic pseudo-random weights — zeros would make every path
    // agree trivially and certify nothing. The variables are walked in
    // name order: the map's own order changes from one process to the
    // next, and with it every weight.
    let mut state = 0x9e3779b97f4a7c15u64;
    let data = varmap.data().lock().unwrap();
    let mut vars: Vec<_> = data.iter().collect();
    vars.sort_by(|a, b| a.0.cmp(b.0));
    for (_, var) in vars {
        let n = var.elem_count();
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.1
            })
            .collect();
        let t = Tensor::from_vec(vals, var.shape(), &dev)
            .and_then(|t| t.to_dtype(dtype))
            .expect("rand");
        var.set(&t).expect("set");
    }
    model
}
