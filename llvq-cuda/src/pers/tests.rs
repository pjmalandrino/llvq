//! The unit tests of the persistent host half, among them the pinned
//! digests of the served units. One file so the digest tests and the
//! module tests they share helpers with stay together.

#[test]
fn the_tree_pass_fits_the_l40s_at_the_three_sizes() {
    // 512 threads, 32 staged rows: under the L40S's 101,376 B with the
    // header and the tables, at the 4B, 8B and 14B widths
    for (hidden, inter, d_ctx, groups) in [(2560u32, 9728u32, 4096u32, 4u32), (4096, 12288, 4096, 4), (5120, 17408, 5120, 5)] {
        let area = p4t_area_floats(16, 32, hidden.max(inter).max(d_ctx), groups);
        assert!(SMEM_HEADER + TABLE_BYTES + 4 * area <= 101_376, "{hidden}: {area} floats");
    }
    assert_eq!(p4t_area_floats(16, 32, 9728, 4), 17_728);
}

#[test]
fn the_tree_splits_are_one_unit_a_cta_and_never_depend_on_rows() {
    // the 4B on 142 SMs: q/k/v 24 blocks, o 10, gate/up 76, down 10, head 594
    assert_eq!(p4t_splits(2560, 9728, 4096, 8, 151_936, 16, 142), [6, 14, 2, 14, 1]);
    assert_eq!(p4t_split(160, 16, 96, 142), 1, "never past half the sub-steps");
    assert!(p4t_fpart_floats([6, 14, 2, 14, 1], 2560, 9728, 4096, 8, 151_936) <= 2_000_000);
    assert_eq!(p4t_ftickets(2560, 9728, 4096, 8, 151_936), 9496);
    assert_eq!(n_bar_4et(36, 0), 327);
}

#[test]
fn the_tree_phases_sum_the_layers() {
    let (layers, g) = (2u32, 3usize);
    let n_bar = n_bar_4et(layers, 0) as usize;
    let events = 1 + 2 * n_bar;
    // leaving wait w at 1000 (w + 1) + c, reaching it 10 (c + 1) earlier
    let st: Vec<u64> = (0..events)
        .flat_map(|e| {
            (0..g).map(move |c| {
                let w = e.div_ceil(2) as u64;
                if e % 2 == 1 { 1000 * (w + 1) + c as u64 - 10 * (c as u64 + 1) } else { 1000 * (w + 1) + c as u64 }
            })
        })
        .collect();
    let p = p4t_phases(&st, layers, g as u32).unwrap();
    assert_eq!(p[0][0], 1000.0);
    assert_eq!(p[1][0], 2000.0, "ln1 twice");
    assert_eq!(p[10][0], 1000.0);
    assert_eq!(p[11][0], 1000.0);
    assert_eq!(p.iter().map(|x| x[0]).sum::<f64>(), 1000.0 * n_bar as f64);
    // busy: 1000 - 10 (c + 1) a wait, CTA 0 the longest
    assert_eq!(p[0][3], 990.0);
    assert_eq!(p[0][1], 970.0);
    assert_eq!(p[1][2], 2.0 * 980.0);
    assert!(p4t_phases(&st[1..], layers, g as u32).is_err());
}

#[test]
fn the_n1_plan_fills_the_grid_at_the_4b() {
    // S1's plan at the 4B on 142 SMs: q/k/v 144 units in two waves, gate/up 152
    let s1 = p4t_plan([6, 14, 2, 14, 1], 2560, 9728, 4096, 8, 151_936, 16, 142);
    assert_eq!((s1[0].0, s1[0].1), (144, 2));
    assert_eq!((s1[2].0, s1[2].1), (152, 2));
    let a1 = [11, 14, 7, 14, 1];
    assert_eq!(p4t_splits_balanced(a1, 2560, 9728, 4096, 8, 151_936, 16, 142), a1, "a1 reads 90 % or more");
    let p = p4t_plan(a1, 2560, 9728, 4096, 8, 151_936, 16, 142);
    assert_eq!((p[0].0, p[0].1), (264, 2));
    assert_eq!((p[2].0, p[2].1), (532, 4));
    assert!(p[0].2 >= 0.9 && p[2].2 >= 0.9);
    // a poor proposal is replaced by the best fill
    assert_eq!(p4t_splits_balanced([6, 14, 2, 14, 1], 2560, 9728, 4096, 8, 151_936, 16, 142)[0], 11);
    assert_eq!(p4t_tiled_words(false, 4096, 2560), 256 * 54 * 48);
    assert_eq!(p4t_tiled_words(true, 151_936, 2560), 9496 * 54 * 96);
}

#[test]
fn the_n2_stream_plan_fits_the_area_and_fills_the_grid() {
    // the 4B on 142 SMs, 16 warps: a2's area, so a2's shared bytes
    let area = p4t_area_floats(16, 32, 9728, 4);
    let smem = SMEM_HEADER + TABLE_BYTES + 4 * area;
    assert_eq!((area, smem), (17_728, 89_984));
    assert!(smem <= 101_376);
    let cap = p4t_stream_cap(area, 16);
    assert_eq!(cap, 13, "32 rows of 24 x 13 + 4 words beside the A tiles");
    assert!(16 * 16 * 28 + 32 * (24 * cap + 4) + 2 * 32 <= area && 16 * 16 * 28 + 32 * (24 * (cap + 1) + 4) + 2 * 32 > area);
    let (h, i, d, kv, v) = (2560, 9728, 4096, 8, 151_936);
    let s = p4t_splits_stream(h, i, d, kv, v, 16, 142, cap).unwrap();
    let names = ["q/k/v", "o", "gate/up", "down", "head"];
    let d_ins = [h, d, h, i, h];
    let floor: Vec<u32> = d_ins.iter().map(|&x| (1..=32).find(|&s| p4t_slice_steps(x, s) <= cap).unwrap()).collect();
    assert_eq!(floor, [5, 7, 5, 16, 5]);
    let p = p4t_plan(s, h, i, d, kv, v, 16, 142);
    for k in 0..5 {
        eprintln!(
            "4B, 142 CTAs: {} S {} (floor {}), units {}, waves {}, fill {:.1} %, at most {} sub-steps a slice",
            names[k], s[k], floor[k], p[k].0, p[k].1, 100.0 * p[k].2, p4t_slice_steps(d_ins[k], s[k])
        );
        assert!(s[k] >= floor[k] && p4t_slice_steps(d_ins[k], s[k]) <= cap);
    }
    assert_eq!(s, [11, 14, 7, 28, 5]);
    let fp = p4t_fpart_floats(s, h, i, d, kv, v);
    eprintln!("slice partials {fp} floats, {:.1} MB", 4.0 * fp as f64 / 1e6);
    assert_eq!(fp, 5 * 32 * 151_936);
    // the host twin: 3 CTAs of 2 warps, its area the widest row's
    let ht = p4t_area_floats(2, 32, 9728, 4);
    let hcap = p4t_stream_cap(ht, 2);
    assert_eq!(hcap, 18);
    let hs = p4t_splits_stream(h, i, d, kv, 1024, 2, 3, hcap).unwrap();
    let hp = p4t_plan(hs, h, i, d, kv, 1024, 2, 3);
    for k in 0..5 {
        eprintln!(
            "host twin, 3 CTAs of 2 warps: {} S {}, units {}, waves {}, fill {:.1} %, at most {} sub-steps a slice",
            names[k], hs[k], hp[k].0, hp[k].1, 100.0 * hp[k].2, p4t_slice_steps(d_ins[k], hs[k])
        );
    }
    assert_eq!(hs, [4, 6, 4, 13, 4]);
    assert!(p4t_splits_stream(h, i, d, kv, v, 16, 142, 3).is_err(), "the down needs S > 32 at 3 sub-steps a slice");
}

#[test]
fn the_stream_body_is_the_tree_body() {
    // pers_4et_stream.cu carries pers_4et.cu's pass with three
    // substitutions, and nothing else of it changed
    let src = P4T_PART.1;
    let a = src.find("template <int FORM, bool TILED>\n__device__ __forceinline__ void pers_4et_body(").unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let b = a + src[a..].find(end).unwrap() + end.len();
    let body = src[a..b]
        .replacen("template <int FORM, bool TILED>\n", "template <int FORM>\n", 1)
        .replacen("void pers_4et_body(", "void pers_4et_stream_body(", 1)
        .replace("p4t_phase<TILED>(", "p4t_phase_stream(");
    assert!(!body.contains("TILED"));
    assert_eq!(P4T_STREAM_PART.1.matches(body.as_str()).count(), 1, "the stream body drifted from pers_4et_body");
    let t = p4t_stream_unit_text(SERVED_DEFINES_SM89);
    assert!(t.contains(&p4t_tiled_unit_text(SERVED_DEFINES_SM89)));
    for f in 0..3 {
        assert_eq!(t.matches(p4t_stream_symbol(f).unwrap()).count(), 1);
        assert_eq!(p4t_unit_text(SERVED_DEFINES_SM89).matches(p4t_stream_symbol(f).unwrap()).count(), 0);
    }
    // the module texts of a0/a1 and a2, and a3's (the bench builds them from the runtime's tile define)
    for (n, x) in [("rows", p4t_unit_text(SERVED_DEFINES_SM89)), ("tiled", p4t_tiled_unit_text(SERVED_DEFINES_SM89)), ("stream", t)] {
        eprintln!("tree module {n} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", x.len(), crate::sha256::sha256_hex(x.as_bytes()));
        // LLVQ_P4T_DUMP=<dir>: the texts NVRTC compiles, for an offline nvcc SASS probe
        if let Ok(dir) = std::env::var("LLVQ_P4T_DUMP") {
            std::fs::write(std::path::Path::new(&dir).join(format!("tree_{n}.cu")), &x).unwrap();
        }
    }
}

#[test]
fn the_a4_body_is_the_stream_body() {
    // pers_4et_a4.cu carries pers_4et_stream.cu's unit, deal and pass with
    // the substitutions below, and nothing else of them changed
    let src = P4T_STREAM_PART.1;
    let a = src.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
    let b = a + src[a..].find("// p4t_phase's deal, the stream unit.").unwrap();
    let unit = src[a..b]
        .replace("p4t_unit_stream(", "p4t_unit_a4(")
        .replace(
            "                                                u32* area, unsigned long long* cmax)\n",
            "                                                u32* area, unsigned long long* cmax, unsigned long long* tm)\n",
        )
        .replace("    p4t_stream_stage(g, xs, sub_lo, ns, R, RS, XS);\n", "    P4T_A4_MARK(0);\n    p4t_stream_stage(g, xs, sub_lo, ns, R, RS, XS);\n")
        .replace("    p4t_cp_wait<0u>();\n    __syncthreads();\n", "    p4t_cp_wait<0u>();\n    __syncthreads();\n    P4T_A4_MARK(1);\n")
        .replace(
            "    __syncthreads();\n    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);\n",
            "    __syncthreads();\n    P4T_A4_MARK(2);\n    if (has) p4t_store_a4(g, L, A, sl, acc, wrow, lane, cmax);\n    P4T_A4_MARK(3);\n",
        );
    let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_stream_body(").unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + src[c..].find(end).unwrap() + end.len();
    let dn = "            for (u32 i = threadIdx.x; i < I; i += blockDim.x) {\n                const unsigned short a = exact_silu_mul(M.silu, __ldcg(gr + i), __ldcg(ur + i));\n                if (Y.tetra_down) {\n                    const float f = h2f(a);\n                    S[i] = ((M.signs_i[i >> 5] >> (i & 31u)) & 1u) ? -f : f;\n                } else {\n                    dst[i] = a;\n                }\n            }\n";
    assert_eq!(src[c..d].matches(dn).count(), 1);
    let body = src[c..d]
        .replacen("void pers_4et_stream_body(", "void pers_4et_a4_body(", 1)
        .replace("p4t_phase_stream(", "p4t_phase_a4(")
        .replace("p4t_ln(", "p4t_ln_a4(")
        .replace("p4t_out(", "p4t_out_a4(")
        .replace(dn, "            p4t_dn_in_a4(M, Y.tetra_down, gr, ur, dst, S, I);\n");
    let both = format!("{unit}{body}");
    assert_eq!(P4T_A4_PART.1.matches(both.as_str()).count(), 1, "the a4 pass drifted from the stream pass");
    let t = p4t_a4_unit_text(SERVED_DEFINES_SM89, true, true);
    assert!(t.contains(&p4t_stream_unit_text(SERVED_DEFINES_SM89)), "a3's text inside a4's, unchanged");
    for f in 0..3 {
        assert_eq!(t.matches(p4t_a4_symbol(f).unwrap()).count(), 1);
        assert_eq!(p4t_stream_unit_text(SERVED_DEFINES_SM89).matches(p4t_a4_symbol(f).unwrap()).count(), 0);
    }
    eprintln!("tree module a4 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
}

#[test]
fn the_a5_body_is_the_a4_body() {
    // pers_4et_a5.cu carries pers_4et_a4.cu's pass with three prefetch
    // calls inserted, and nothing else of it changed
    let src = P4T_A4_PART.1;
    let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a4_body(").unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + src[c..].find(end).unwrap() + end.len();
    let mut body = src[c..d].replacen("void pers_4et_a4_body(", "void pers_4et_a5_body(", 1);
    for (x, y) in [
        ("        unsigned short* tv_l = L.tv + (unsigned long long)l * P4T_RMAX * dkv;\n", "        p4t_pf_qkv(Y, L);\n"),
        ("        p4t_attention(A, L, Y.kc, Y.vc, tk_l, tv_l, reinterpret_cast<unsigned char*>(S));\n", "        p4t_pf_o_gu(Y, L, H, dq);\n"),
        ("            p4t_phase_a4(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);\n", "            p4t_pf_down(Y, L, H);\n"),
    ] {
        assert_eq!(body.matches(x).count(), 1, "{x}");
        body = body.replacen(x, &format!("{x}{y}"), 1);
    }
    assert_eq!(P4T_A5_PART.1.matches(body.as_str()).count(), 1, "the a5 pass drifted from the a4 pass");
    let t = p4t_a5_unit_text(SERVED_DEFINES_SM89);
    assert!(t.contains(&p4t_a4_unit_text(SERVED_DEFINES_SM89, true, true)), "a4's text inside a5's, unchanged");
    for f in 0..3 {
        assert_eq!(t.matches(p4t_a5_symbol(f).unwrap()).count(), 1);
    }
    eprintln!("tree module a5 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
}

#[test]
fn the_a6_family_is_the_stream_body() {
    // pers_4et_a6.cu carries a3's pass three times with the substitutions below
    let src = P4T_STREAM_PART.1;
    let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_stream_body(").unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + src[c..].find(end).unwrap() + end.len();
    let body = &src[c..d];
    let dn = "        for (u32 r = c0; r < R; r += G) {\n            const unsigned short* gr = L.g_raw";
    assert_eq!(body.matches(dn).count(), 1);
    let b6 = body.replacen("void pers_4et_stream_body(", "void pers_4et_a6_body(", 1).replacen(
        dn,
        "        if (!Y.tetra_down && (I & 7u) == 0u) p4t_dn_in_grid(M, L, R, I);\n        else for (u32 r = c0; r < R; r += G) {\n            const unsigned short* gr = L.g_raw",
        1,
    );
    let mut b7 = b6.replacen("void pers_4et_a6_body(", "void pers_4et_a7_body(", 1);
    for (x, y) in [
        ("        unsigned short* tv_l = L.tv + (unsigned long long)l * P4T_RMAX * dkv;\n", "        p4t_pf_qkv(Y, L);\n"),
        ("        p4t_attention(A, L, Y.kc, Y.vc, tk_l, tv_l, reinterpret_cast<unsigned char*>(S));\n", "        p4t_pf_o_gu(Y, L, H, dq);\n"),
        ("            p4t_phase_stream(sg, su, su, 2u, L, A, tab, t_inv, area, cmax);\n", "            p4t_pf_down(Y, L, H);\n"),
    ] {
        assert_eq!(b7.matches(x).count(), 1, "{x}");
        b7 = b7.replacen(x, &format!("{x}{y}"), 1);
    }
    let a = src.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
    let b = a + src[a..].find("// ---- the pass:").unwrap();
    let uf = src[a..b]
        .replace("p4t_unit_stream(", "p4t_unit_floor(")
        .replace("p4t_phase_stream(", "p4t_phase_floor(")
        .replace("    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);\n", "    if (has) p4t_store_floor(g, L, A, sl, acc, wrow, lane, cmax);\n");
    let bf = body.replacen("void pers_4et_stream_body(", "void pers_4et_a3f_body(", 1).replace("p4t_phase_stream(", "p4t_phase_floor(");
    for (n, x) in [("a6", b6.clone()), ("a7", b7), ("a3f", format!("{uf}{bf}"))] {
        assert_eq!(P4T_A6_PART.1.matches(x.as_str()).count(), 1, "the {n} pass drifted from the stream pass");
        let t = p4t_a6_unit_text(n, SERVED_DEFINES_SM89).unwrap();
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a6_symbol(n, f).unwrap()).count(), 1);
        }
        eprintln!("tree module {n} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    }
    assert!(p4t_a6_unit_text("a6", SERVED_DEFINES_SM89).unwrap().contains(&p4t_stream_unit_text(SERVED_DEFINES_SM89)));
}

#[test]
fn the_a8_family_is_the_stream_body() {
    // pers_4et_a8.cu carries a3's unit, deal and pass three times, only the store changed
    let src = P4T_STREAM_PART.1;
    let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_stream_body(").unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + src[c..].find(end).unwrap() + end.len();
    let a = src.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
    let b = a + src[a..].find("// ---- the pass:").unwrap();
    for (tag, store) in [("a3p", "p4t_store_proto<false>"), ("a3t", "p4t_store_proto<true>"), ("a8", "p4t_store_lean")] {
        let u = src[a..b]
            .replace("p4t_unit_stream(", &format!("p4t_unit_{tag}("))
            .replace("p4t_phase_stream(", &format!("p4t_phase_{tag}("))
            .replace("    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);\n", &format!("    if (has) {store}(g, L, A, sl, acc, wrow, lane, cmax);\n"));
        let bd = src[c..d]
            .replacen("void pers_4et_stream_body(", &format!("void pers_4et_{tag}_body("), 1)
            .replace("p4t_phase_stream(", &format!("p4t_phase_{tag}("));
        assert_eq!(P4T_A8_PART.1.matches(format!("{u}{bd}").as_str()).count(), 1, "the {tag} pass drifted from the stream pass");
        let t = p4t_a8_unit_text(tag, SERVED_DEFINES_SM89).unwrap();
        assert!(t.contains(&p4t_stream_unit_text(SERVED_DEFINES_SM89)));
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a8_symbol(tag, f).unwrap()).count(), 1);
        }
        eprintln!("tree module {tag} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    }
}

#[test]
fn the_a9_body_is_the_stream_body() {
    // pers_4et_a9.cu carries a3's unit with a9's store and a3's pass with a
    // wait and a fold step after each row phase
    let src = P4T_STREAM_PART.1;
    let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_stream_body(").unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + src[c..].find(end).unwrap() + end.len();
    let a = src.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
    let b = a + src[a..].find("// ---- the pass:").unwrap();
    let u = src[a..b]
        .replace("p4t_unit_stream(", "p4t_unit_a9(")
        .replace("p4t_phase_stream(", "p4t_phase_a9(")
        .replace("    if (has) p4t_store(g, L, A, sl, acc, wrow, lane, cmax);\n", "    if (has) p4t_store_a9(g, L, A, sl, acc, wrow, lane, cmax);\n");
    let mut bd = src[c..d].replacen("void pers_4et_stream_body(", "void pers_4et_a9_body(", 1);
    for segs in ["sq, sk, sv, 3u", "so, so, so, 1u", "sg, su, su, 2u", "sd, sd, sd, 1u"] {
        let x = format!("            p4t_phase_stream({segs}, L, A, tab, t_inv, area, cmax);\n        }}\n");
        assert_eq!(bd.matches(x.as_str()).count(), 1);
        bd = bd.replace(
            &x,
            &format!("            p4t_phase_a9({segs}, L, A, tab, t_inv, area, cmax);\n            P4T_WAIT();\n            p4t_fold_a9({segs}, L, A, cmax);\n        }}\n"),
        );
    }
    let fl = "        if (threadIdx.x < R && cmax[threadIdx.x] != 0ull) atomicMax(L.slots + threadIdx.x, cmax[threadIdx.x]);\n";
    let x = format!("        p4t_phase_stream(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);\n        __syncthreads();\n{fl}");
    assert_eq!(bd.matches(x.as_str()).count(), 1);
    bd = bd.replace(
        &x,
        &format!("        p4t_phase_a9(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);\n        __syncthreads();\n{fl}        P4T_WAIT();\n        p4t_fold_a9(sh, sh, sh, 1u, L, A, cmax);\n        __syncthreads();\n{fl}"),
    );
    assert_eq!(P4T_A9_PART.1.matches(format!("{u}{bd}").as_str()).count(), 1, "the a9 pass drifted from the stream pass");
    assert_eq!(bd.matches("P4T_WAIT();").count(), src[c..d].matches("P4T_WAIT();").count() + 5);
    let t = p4t_a9_unit_text(SERVED_DEFINES_SM89);
    assert!(t.contains(&p4t_stream_unit_text(SERVED_DEFINES_SM89)));
    for f in 0..3 {
        assert_eq!(t.matches(p4t_a9_symbol(f).unwrap()).count(), 1);
    }
    eprintln!("tree module a9 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    // the waits: 13 a layer, 3 + 1 at the ends; the phases sum the layers
    assert_eq!(n_bar_a9(36, 0), 327 + 145);
    let (layers, g) = (2u32, 3usize);
    let n_bar = n_bar_a9(layers, 0) as usize;
    let st: Vec<u64> = (0..1 + 2 * n_bar).flat_map(|e| (0..g).map(move |c| 1000 * (e as u64 + 1) + c as u64)).collect();
    let p = p4t_phases_a9(&st, layers, g as u32).unwrap();
    assert_eq!(p.len(), 17);
    assert_eq!(p[3][0], 2.0 * 2000.0, "qkv fold twice");
    assert_eq!(p[16][0], 2000.0, "head fold once");
    assert_eq!(p[14][0], 2000.0, "norm once");
    assert_eq!(p.iter().map(|x| x[0]).sum::<f64>(), 2000.0 * n_bar as f64);
}

#[test]
fn the_a10_body_is_the_a9_body() {
    // pers_4et_a10.cu carries a9's pass with the prologue and fold-step
    // substitutions; the waits are a9's (P4T_WAIT_GAP counts as one)
    let src = P4T_A9_PART.1;
    let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a9_body(").unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + src[c..].find(end).unwrap() + end.len();
    let a9 = &src[c..d];
    // the generator's substitutions, replayed (scratch gen_a10.py's pairs)
    let gap = r##"#define P4T_WAIT_GAP(...)                                                                            \
    do {                                                                                             \
        if (L.stamps != nullptr) {                                                                   \
            __syncthreads();                                                                         \
            if (threadIdx.x == 0u) L.stamps[(size_t)(2u * nb + 1u) * G + c0] = pers_now();           \
        }                                                                                            \
        if (!pers_grid_wait_gap(A->bar, L.bar_base + (++nb) * G, A->abort_flag, [&] { __VA_ARGS__; })) return; \
        if (L.stamps != nullptr && threadIdx.x == 0u) L.stamps[(size_t)(2u * nb) * G + c0] = pers_now(); \
    } while (0)
    // a10: the staged rotation area (null: the prologues keep a9's path)
    float* const v7a10 = p4t_v7_a10(S, L.area_floats, 2u * H > A->layers[0].mlp.inter ? 2u * H : A->layers[0].mlp.inter);
"##;
    let dn0 = a9.find("        for (u32 r = c0; r < R; r += G) {\n            const unsigned short* gr = L.g_raw").unwrap();
    let dn_end = "            } else {\n                __syncthreads();\n            }\n        }\n";
    let dn1 = dn0 + a9[dn0..].find(dn_end).unwrap() + dn_end.len();
    let dn_old = &a9[dn0..dn1];
    let pairs: Vec<(String, String)> = vec![
        ("void pers_4et_a9_body(".into(), "void pers_4et_a10_body(".into()),
        ("    } while (0)\n".into(), format!("    }} while (0)\n{gap}")),
        (r##"                   H * (c0 + 1u) / G);
    P4T_WAIT();
    for (u32 l = 0; l < NL; ++l) {
"##.into(),
         r##"                   H * (c0 + 1u) / G);
    P4T_WAIT_GAP(if (v7a10 != nullptr) pers_v7_stage_rot(v7a10, A->layers[0].signs_a, H, A->layers[0].small_a, A->layers[0].k_a));
    for (u32 l = 0; l < NL; ++l) {
"##.into()),
        (r##"            p4t_ln(L.x_a + (size_t)r * H, Y.ln1, Y.signs_a, A->eps, H, S, L.xn + (size_t)r * H);
            pers_rot_finish(S, Y.small_a, H, Y.m_a, Y.k_a, Y.inv_a);
            p4t_out(S, H, L.xin + (size_t)r * L.ldx);
"##.into(),
         r##"            p4t_ln_rot_a10(L.x_a + (size_t)r * H, Y.ln1, Y.signs_a, Y.small_a, A->eps, H, Y.m_a, Y.k_a, Y.inv_a, S, v7a10,
                           L.xn + (size_t)r * H, L.xin + (size_t)r * L.ldx);
"##.into()),
        (r##"            p4t_fold_a9(so, so, so, 1u, L, A, cmax);
        }
        P4T_WAIT();
"##.into(),
         r##"            p4t_fold_a9(so, so, so, 1u, L, A, cmax);
        }
        P4T_WAIT_GAP(if (v7a10 != nullptr) pers_v7_stage_rot(v7a10, M.signs_h, H, M.small_h, M.k_h));
"##.into()),
        (r##"            p4t_ln(L.x_b + (size_t)r * H, M.alpha, M.signs_h, M.eps, H, S, nullptr);
            pers_rot_finish(S, M.small_h, H, M.m_h, M.k_h, M.inv_h);
            p4t_out(S, H, L.xin + (size_t)r * L.ldx);
"##.into(),
         r##"            p4t_ln_rot_a10(L.x_b + (size_t)r * H, M.alpha, M.signs_h, M.small_h, M.eps, H, M.m_h, M.k_h, M.inv_h, S, v7a10,
                           nullptr, L.xin + (size_t)r * L.ldx);
"##.into()),
        (r##"            p4t_fold_a9(sg, su, su, 2u, L, A, cmax);
        }
        P4T_WAIT();
"##.into(),
         r##"            p4t_fold_gu_a10(sg, su, M, L);
        }
        P4T_WAIT_GAP(if (v7a10 != nullptr && Y.tetra_down) pers_v7_stage_rot(v7a10, M.signs_i, I, M.small_i, M.k_i));
"##.into()),
        (dn_old.to_string(), r##"        for (u32 r = c0; r < R; r += G) p4t_dn_in_a10(M, Y.tetra_down, L.xin + (size_t)r * L.ldx, S, I, v7a10);
"##.into()),
        (r##"            p4t_fold_a9(sd, sd, sd, 1u, L, A, cmax);
        }
        P4T_WAIT();
    }
"##.into(),
         r##"            p4t_fold_a9(sd, sd, sd, 1u, L, A, cmax);
        }
        P4T_WAIT_GAP(if (v7a10 != nullptr) {
            if (l + 1u < NL) pers_v7_stage_rot(v7a10, A->layers[l + 1u].signs_a, H, A->layers[l + 1u].small_a, A->layers[l + 1u].k_a);
            else p4t_v7_zero_signs_a10(v7a10, H);
        });
    }
"##.into()),
        (r##"        p4t_ln(L.x_a + (size_t)r * H, A->final_norm, nullptr, A->eps, H, S, L.xin + (size_t)r * L.ldx);
"##.into(),
         r##"        p4t_ln_final_a10(L.x_a + (size_t)r * H, A->final_norm, A->eps, H, S, v7a10, L.xin + (size_t)r * L.ldx);
"##.into()),
        ("#undef P4T_WAIT\n}\n".into(), "#undef P4T_WAIT_GAP\n#undef P4T_WAIT\n}\n".into()),
    ];
    let mut b10 = a9.to_string();
    for (x, y) in &pairs {
        assert_eq!(b10.matches(x.as_str()).count(), 1, "{}", &x[..x.len().min(60)]);
        b10 = b10.replacen(x.as_str(), y, 1);
    }
    assert_eq!(P4T_A10_PART.1.matches(b10.as_str()).count(), 1, "the a10 pass drifted from the a9 pass");
    let waits = |t: &str| t.matches("P4T_WAIT();").count() + t.matches("P4T_WAIT_GAP(if").count();
    assert_eq!(waits(a9), waits(&b10), "a10 keeps a9's waits");
    let t = p4t_a10_unit_text(SERVED_DEFINES_SM89);
    assert!(t.contains(&p4t_a9_unit_text(SERVED_DEFINES_SM89)));
    for f in 0..3 {
        assert_eq!(t.matches(p4t_a10_symbol(f).unwrap()).count(), 1);
    }
    eprintln!("tree module a10 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    // the staged area fits above the widest prologue row at the 4B and on the twin
    for (nw, widest) in [(16u32, 9728u32), (2, 9728)] {
        let area = p4t_area_floats(nw, 32, widest, 4);
        let off = (area - 2 * P4T_RMAX - 1840) & !3;
        eprintln!("area {area} floats at {nw} warps: the staged rotation at {off}, the prologue needs 9728");
        assert!(off >= 9728);
    }
}

#[test]
fn the_a11_pass_is_a10s_with_prefetches_in_its_gaps() {
    // a11's pass is a10's with prefetch calls inside wait gaps only: the
    // same waits, and every a10 line outside the five rewritten sites kept
    let src = P4T_A10_PART.1;
    let c = src.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a10_body(").unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + src[c..].find(end).unwrap() + end.len();
    let a10 = &src[c..d];
    let a11s = P4T_A11_PART.1;
    let e = a11s.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a11_body(").unwrap();
    let f = e + a11s[e..].find(end).unwrap() + end.len();
    let a11 = &a11s[e..f];
    let waits = |t: &str| t.matches("P4T_WAIT();").count() + t.matches("P4T_WAIT_GAP(").count();
    assert_eq!(waits(a10), waits(a11), "a11 keeps a10's waits");
    assert_eq!(a11.matches("p4t_pf11_").count(), 5, "five prefetch calls");
    for line in a11.lines().filter(|l| l.contains("p4t_pf11_")) {
        assert!(!line.trim_start().starts_with("p4t_pf11_") || line.contains("));"), "a prefetch outside a gap: {line}");
    }
    let mut at = 0usize;
    for line in a10.lines().filter(|l| !l.contains("pers_4et_a10_body(") && !l.contains("P4T_WAIT") && !l.contains("p4t_attention(") && !l.contains("pers_v7_stage_rot(v7a10, M.signs_i") && !l.contains("p4t_v7_zero_signs_a10") && !l.trim().starts_with("});")) {
        let i = a11[at..].find(line).unwrap_or_else(|| panic!("a10's line {line:?} missing or out of order in a11's pass"));
        at += i + line.len();
    }
    let t = p4t_a11_unit_text(SERVED_DEFINES_SM89);
    assert!(t.contains(&p4t_a10_unit_text(SERVED_DEFINES_SM89)));
    for f in 0..3 {
        assert_eq!(t.matches(p4t_a11_symbol(f).unwrap()).count(), 1);
    }
    eprintln!("tree module a11 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    // the volumes at the 4B, plan [5, 14, 5, 28, 5], 142 CTAs of 16 warps
    for int4_down in [false, true] {
        let v = p4t_prefetch_volume([5, 14, 5, 28, 5], 2560, 9728, 4096, 8, int4_down, 16, 142);
        let total: u64 = v.iter().map(|x| x.0).sum();
        eprintln!(
            "4B, {} down: q/k/v {:.2} MB (a CTA at most {:.1} KB), o {:.2} MB ({:.1} KB), gate/up {:.2} MB ({:.1} KB), down {:.2} MB ({:.1} KB); a layer {:.2} MB of the L40S's 96 MB L2",
            if int4_down { "int4" } else { "Tetra" },
            v[0].0 as f64 / 1e6, v[0].1 as f64 / 1e3, v[1].0 as f64 / 1e6, v[1].1 as f64 / 1e3,
            v[2].0 as f64 / 1e6, v[2].1 as f64 / 1e3, v[3].0 as f64 / 1e6, v[3].1 as f64 / 1e3, total as f64 / 1e6
        );
        assert!(total < 96_000_000);
    }
}

#[test]
fn the_a11i_pass_is_a11s_with_marks() {
    // a11i's pass is a11's with its phase calls renamed; its unit is a9's
    // with the marks only (every a9 unit line kept, in order)
    let a10 = P4T_A11_PART.1;
    let c = a10.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a11_body(").unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + a10[c..].find(end).unwrap() + end.len();
    let body = a10[c..d].replacen("void pers_4et_a11_body(", "void pers_4et_a11i_body(", 1).replace("p4t_phase_a9(", "p4t_phase_a11i(");
    assert_eq!(P4T_A11I_PART.1.matches(body.as_str()).count(), 1, "the a11i pass drifted from a11's");
    let a9 = P4T_A9_PART.1;
    let a = a9.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
    let b = a + a9[a..].find("// p4t_phase's deal, the stream unit.").unwrap();
    let ui = P4T_A11I_PART.1;
    let mut at = ui.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
    for line in a9[a..b].lines().filter(|l| !l.contains("p4t_unit_a9(") && !l.contains("u32* area, unsigned long long* cmax)")) {
        let i = ui[at..].find(line).unwrap_or_else(|| panic!("a9's unit line {line:?} missing or out of order in a11i's"));
        at += i + line.len();
    }
    let t = p4t_a11i_unit_text(SERVED_DEFINES_SM89);
    assert!(t.contains(&p4t_a11_unit_text(SERVED_DEFINES_SM89)));
    for f in 0..3 {
        assert_eq!(t.matches(p4t_a11i_symbol(f).unwrap()).count(), 1);
    }
    eprintln!("tree module a11i at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    // the wait classes: arrival 10 (c + 1) before leaving, busy 1000 - 10 (c + 1)
    let (layers, g) = (2u32, 3usize);
    let n_bar = n_bar_a9(layers, 0) as usize;
    let st: Vec<u64> = (0..1 + 2 * n_bar)
        .flat_map(|e| (0..g).map(move |c| { let w = e.div_ceil(2) as u64; if e % 2 == 1 { 1000 * (w + 1) - 10 * (c as u64 + 1) } else { 1000 * (w + 1) } }))
        .collect();
    let (cls, pro) = p4t_wait_classes(&st, layers, g as u32, 1).unwrap();
    // 3 prologue waits a layer + the norm (7), 4 row waits a layer + the head (9), 4 folds + 1 (9)
    assert_eq!(cls[0], [20.0 * 7.0, 30.0 * 7.0]);
    assert_eq!(cls[1], [20.0 * 9.0, 30.0 * 9.0]);
    assert_eq!(cls[2][1], 30.0 * 9.0);
    assert_eq!(pro[0], 2.0 * 990.0, "ln1 on CTA 0, twice");
}

#[test]
fn the_row_cap_passes_are_a10s() {
    // pers_4et_a10rc.cu: a9's unit and deal over RC n8 tiles, a10's pass
    // with the row cap; nothing else of them changed
    let a9 = P4T_A9_PART.1;
    let a = a9.find("// One unit: tiles [blk nw, blk nw + nw) of g").unwrap();
    let b = a + a9[a..].find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a9_body(").unwrap();
    let unit = a9[a..b]
        .replacen("__device__ __forceinline__ void p4t_unit_a9(", "template <u32 RC, bool IP = false>\n__device__ __forceinline__ void p4t_unit_rc(", 1)
        // a12ip (M1): the words read in place under IP, the copy otherwise
        .replacen(
            "// once and streamed by each warp alone.\n",
            "// once and streamed by each warp alone. IP: the words read in place, from\n// the served order (p4t_ip_load), instead of the tree-tiled copy.\n",
            1,
        )
        .replacen(
            "        const u32* wp = g.w + ((unsigned long long)tile * nsub + sub_lo) * SW + (g.kind == 0u ? 3u * r + h : 6u * r + 3u * h);\n        u32 c0, c1, c2;\n        p4t_stream_load(g.kind, wp, c0, c1, c2);\n",
            "        const u32* wp = IP ? p4t_ip_row(g, wrow + r)\n                           : g.w + ((unsigned long long)tile * nsub + sub_lo) * SW + (g.kind == 0u ? 3u * r + h : 6u * r + 3u * h);\n        u32 c0, c1, c2;\n        if (IP) p4t_ip_load(g, wp, sub_lo, h, c0, c1, c2);\n        else p4t_stream_load(g.kind, wp, c0, c1, c2);\n",
            1,
        )
        .replacen(
            "            if (s + 1u < ns) p4t_stream_load(g.kind, wp + SW, n0, n1, n2);\n",
            "            if (IP) {\n                if (s + 1u < ns) p4t_ip_load(g, wp, sub_lo + s + 1u, h, n0, n1, n2);\n            } else if (s + 1u < ns) {\n                p4t_stream_load(g.kind, wp + SW, n0, n1, n2);\n            }\n",
            1,
        )
        .replacen("            wp += SW;\n", "            if (!IP) wp += SW;\n", 1)
        .replacen("    float acc[4][4];\n#pragma unroll\n    for (u32 nt = 0; nt < 4u; ++nt)", "    float acc[RC][4];\n#pragma unroll\n    for (u32 nt = 0; nt < RC; ++nt)", 1)
        .replacen("                for (u32 nt = 0; nt < 4u; ++nt) {", "                for (u32 nt = 0; nt < RC; ++nt) {", 1)
        .replacen("    if (has) p4t_store_a9(g, L, A, sl, acc, wrow, lane, cmax);\n", "    if (has) p4t_store_rc<RC>(g, L, A, sl, acc, wrow, lane, cmax);\n", 1)
        .replacen("__device__ __forceinline__ void p4t_phase_a9(", "template <u32 RC, bool IP = false>\n__device__ __forceinline__ void p4t_phase_rc(", 1)
        .replacen("        p4t_unit_a9(g, lu % g.S, lu / g.S, L, A, tab, t_inv, area, cmax);", "        p4t_unit_rc<RC, IP>(g, lu % g.S, lu / g.S, L, A, tab, t_inv, area, cmax);", 1);
    assert!(!unit.contains("nt < 4u"), "every n8-tile loop of the unit is over RC");
    let a10 = P4T_A10_PART.1;
    let c = a10.find("template <int FORM>\n__device__ __forceinline__ void pers_4et_a10_body(").unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + a10[c..].find(end).unwrap() + end.len();
    assert_eq!(a10[c..d].matches("if (R == 0u || R > P4T_RMAX ||").count(), 1);
    let body = a10[c..d]
        .replacen("template <int FORM>\n__device__ __forceinline__ void pers_4et_a10_body(", "template <int FORM, u32 RC>\n__device__ __forceinline__ void pers_4et_rc_body(", 1)
        .replace("p4t_phase_a9(", "p4t_phase_rc<RC>(")
        .replacen("if (R == 0u || R > P4T_RMAX ||", "if (R == 0u || R > 8u * RC ||", 1);
    assert_eq!(P4T_A10RC_PART.1.matches(format!("{unit}{body}").as_str()).count(), 1, "the row-cap pass drifted from a10's");
    for rows in [8, 16] {
        let t = p4t_a10rc_unit_text(SERVED_DEFINES_SM89, rows);
        assert!(t.contains(&p4t_a10_unit_text(SERVED_DEFINES_SM89)));
        for f in 0..3 {
            assert_eq!(t.matches(p4t_a10rc_symbol(rows, f).unwrap()).count(), 1);
        }
        eprintln!("tree module a10r{rows} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    }
}

#[test]
fn the_thread_count_arms_are_a10s_pass() {
    // pers_4et_a10t.cu adds symbols around a10's body and nothing else
    let t = P4T_A10T_PART.1;
    assert_eq!(t.matches("pers_4et_a10_body<FORM_>(A, L);").count(), 1);
    assert_eq!(t.matches("__device__").count(), 0, "no device function of its own");
    for threads in [384u32, 256] {
        let m = p4t_a10t_unit_text(SERVED_DEFINES_SM89, threads);
        assert!(m.contains(&p4t_a10_unit_text(SERVED_DEFINES_SM89)));
        for f in 0..3 {
            assert_eq!(m.matches(p4t_a10t_symbol(threads, f).unwrap()).count(), 1);
        }
        // the 4B at this warp count: area, shared bytes, cap, the stream rule's plan
        let nw = threads / 32;
        let area = p4t_area_floats(nw, 32, 9728, 4);
        let cap = p4t_stream_cap(area, nw);
        let smem = SMEM_HEADER + TABLE_BYTES + 4 * area;
        let rule = p4t_splits_stream(2560, 9728, 4096, 8, 151_936, nw, 142, cap).unwrap();
        for (name, s) in [("[5, 14, 5, 28, 5]", [5u32, 14, 5, 28, 5]), ("the stream rule's", rule)] {
            let p = p4t_plan(s, 2560, 9728, 4096, 8, 151_936, nw, 142);
            let steps: Vec<u32> = s.iter().zip([2560u32, 4096, 2560, 9728, 2560]).map(|(&x, d)| p4t_slice_steps(d, x)).collect();
            eprintln!(
                "4B at {threads} threads ({nw} warps): area {area} floats, {smem} B shared, cap {cap}; plan {name} {s:?}: units, waves, fill {:?}; sub-steps a slice {steps:?}",
                p.iter().map(|x| (x.0, x.1, (x.2 * 1000.0).round() / 10.0)).collect::<Vec<_>>()
            );
            assert!(steps.iter().all(|&x| x <= cap));
        }
        eprintln!("tree module a10t{threads} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", m.len(), crate::sha256::sha256_hex(m.as_bytes()));
    }
}

#[test]
fn the_second_thread_count_arms_are_a10s_passes() {
    // pers_4et_a10t2.cu adds symbols around a10's and the row cap's bodies, nothing else
    let t = P4T_A10T2_PART.1;
    assert_eq!(t.matches("pers_4et_a10_body<FORM_>(A, L);").count(), 1);
    assert_eq!(t.matches("pers_4et_rc_body<FORM_, 2u>(A, L);").count(), 1);
    assert_eq!(t.matches("__device__").count(), 0);
    for (arm, threads, base) in [("a10t448", 448u32, p4t_a10_unit_text(SERVED_DEFINES_SM89)), ("a10t320", 320, p4t_a10_unit_text(SERVED_DEFINES_SM89)), ("a10t384r16", 384, p4t_a10rc_unit_text(SERVED_DEFINES_SM89, 16))] {
        let m = p4t_a10t2_unit_text(SERVED_DEFINES_SM89, arm).unwrap();
        assert!(m.contains(&base));
        for f in 0..3 {
            assert_eq!(m.matches(p4t_a10t2_symbol(arm, f).unwrap()).count(), 1);
        }
        let nw = threads / 32;
        let area = p4t_area_floats(nw, 32, 9728, 4);
        let cap = p4t_stream_cap(area, nw);
        let smem = SMEM_HEADER + TABLE_BYTES + 4 * area;
        let rule = p4t_splits_stream(2560, 9728, 4096, 8, 151_936, nw, 142, cap).unwrap();
        for (name, s) in [("[5, 14, 5, 28, 5]", [5u32, 14, 5, 28, 5]), ("[8, 10, 8, 20, 6]", [8, 10, 8, 20, 6]), ("the stream rule's", rule)] {
            let p = p4t_plan(s, 2560, 9728, 4096, 8, 151_936, nw, 142);
            let steps: Vec<u32> = s.iter().zip([2560u32, 4096, 2560, 9728, 2560]).map(|(&x, d)| p4t_slice_steps(d, x)).collect();
            eprintln!(
                "4B, {arm} ({nw} warps): area {area} floats, {smem} B shared, cap {cap}; plan {name} {s:?}: units, waves, fill {:?}; sub-steps a slice {steps:?}",
                p.iter().map(|x| (x.0, x.1, (x.2 * 1000.0).round() / 10.0)).collect::<Vec<_>>()
            );
        }
        eprintln!("tree module {arm} at SERVED_DEFINES_SM89: {} bytes, sha256 {}", m.len(), crate::sha256::sha256_hex(m.as_bytes()));
    }
}

#[test]
fn the_a12_pass_is_the_row_cap_pass_with_taps() {
    // pers_4et_a12.cu: pers_4et_rc_body renamed, two tap calls added
    let rc = P4T_A10RC_PART.1;
    let c = rc.find("template <int FORM, u32 RC>\n__device__ __forceinline__ void pers_4et_rc_body(").unwrap();
    // the a12ip edits (M1): IP threaded through the body's five phases, nothing else
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + rc[c..].find(end).unwrap() + end.len();
    let body = rc[c..d]
        .replacen("template <int FORM, u32 RC>\n__device__ __forceinline__ void pers_4et_rc_body(", "template <int FORM, u32 RC, bool IP = false>\n__device__ __forceinline__ void pers_4et_a12_body(", 1)
        .replace("p4t_phase_rc<RC>(", "p4t_phase_rc<RC, IP>(")
        .replacen(
            "        // ln1: the natural values for the int4 v rows, the signed and\n",
            "        p4t_taps_a12(L, H, R, l, c0, G);\n        // ln1: the natural values for the int4 v rows, the signed and\n",
            1,
        )
        .replacen(
            "    // the final norm, natural: the head's input\n",
            "    p4t_taps_a12(L, H, R, NL, c0, G);\n    // A launch with L.split_head == 0 (the chain prefill's passes before the\n    // prompt's last, docs/mesures/large-prefill-contexte-2026-11-11.txt) skips\n    // the final norm and the head, every wait kept: no logits, no argmax (the\n    // ring reads u32 max), the rows' residuals, K, V and taps untouched.\n    const bool head_on = L.split_head != 0u;\n    // the final norm, natural: the head's input\n",
            1,
        )
        // the head switch (96c501a): the final norm, the head's phase and
        // its fold run only when split_head is not 0; every wait and
        // every slot write stay unconditional
        .replacen(
            "    for (u32 r = c0; r < R; r += G)\n        p4t_ln_final_a10(",
            "    if (head_on)\n        for (u32 r = c0; r < R; r += G)\n            p4t_ln_final_a10(",
            1,
        )
        .replacen(
            "        p4t_io(sh, L.xin, L.ldx, nullptr, 0u, nullptr, P4T_HEAD, L.split_head);\n        unsigned long long po = 0ull;\n        u32 to = 0u;\n        p4t_place(sh, L, po, to);\n        p4t_phase_rc<RC, IP>(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);\n",
            "        if (head_on) {\n            p4t_io(sh, L.xin, L.ldx, nullptr, 0u, nullptr, P4T_HEAD, L.split_head);\n            unsigned long long po = 0ull;\n            u32 to = 0u;\n            p4t_place(sh, L, po, to);\n            p4t_phase_rc<RC, IP>(sh, sh, sh, 1u, L, A, tab, t_inv, area, cmax);\n        }\n",
            1,
        )
        .replacen("        p4t_fold_a9(sh, sh, sh, 1u, L, A, cmax);\n", "        if (head_on) p4t_fold_a9(sh, sh, sh, 1u, L, A, cmax);\n", 1);
    assert_eq!(P4T_A12_PART.1.matches(body.as_str()).count(), 1, "the a12 pass drifted from the row-cap pass");
    let t = p4t_a12_unit_text(SERVED_DEFINES_SM89);
    assert!(t.contains(&p4t_a10rc_unit_text(SERVED_DEFINES_SM89, 16)));
    for f in 0..3 {
        assert_eq!(t.matches(p4t_a12_symbol(f).unwrap()).count(), 1);
    }
    eprintln!("tree module a12 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
    // a12ip: a12's text under one more define; its symbols and the load
    // check, and not a12's
    let ip = p4t_a12ip_unit_text(SERVED_DEFINES_SM89);
    assert_eq!(ip, format!("#define P4T_A12IP 1\n{t}"));
    assert_eq!(P4T_A12_PART.1.matches("pers_4et_a12_body<FORM_, 2u, true>(A, L);").count(), 1);
    assert_eq!(P4T_A12_PART.1.matches("extern \"C\" __global__ void p4t_ip_check(").count(), 1);
    for f in 0..3 {
        assert_eq!(ip.matches(p4t_a12ip_symbol(f).unwrap()).count(), 1);
    }
    assert!(p4t_a12ip_symbol(3).is_err());
    eprintln!("tree module a12ip at SERVED_DEFINES_SM89: {} bytes, sha256 {}", ip.len(), crate::sha256::sha256_hex(ip.as_bytes()));
}

/// The card's tile define on the L40S (`TileConfig::define`, 64 blocks).
const CARD_TILE_SM89: &str = "#define TILE_BLOCKS 64u\n";

/// The module texts the card compiled in large-prefill D1, at their
/// sha256 (d1.txt's loader lines): 4E's tree module, a12ip's, a12ipg's.
/// An arm added beside them must leave all three where they are.
#[test]
fn the_d1_tree_modules_keep_their_bytes() {
    for (name, text, sha) in [
        ("4E tree", p4t_unit_text(CARD_TILE_SM89), "ce1c354d2583eb178ae8622ff99e10f9b457543972bb5039ce89ede9eec31e44"),
        ("a12ip", p4t_a12ip_unit_text(CARD_TILE_SM89), "63155df95ef832aabbff328bd92062b6981763ed57358e32dc118d150df7c6ad"),
        ("a12ipg", p4t_a12ipg_unit_text(CARD_TILE_SM89), "246e99b18895a7f9452411372b8ff4bbb686276408318e547656fb809b722da8"),
        // the 32-row arm (large-prefill-r32-mac) and the served 4E v9 module
        ("a12ipg32", p4t_a12ipg32_unit_text(CARD_TILE_SM89), "a35f2b83061dc8a22ee65e6805c9e73454a93ec773ec919a14afc80fcf60aa8f"),
        ("4E v9", format!("{}{}", p4e_rows_define(7).unwrap(), pers_unit_text(CARD_TILE_SM89)), "1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530"),
    ] {
        assert_eq!(crate::sha256::sha256_hex(text.as_bytes()), sha, "{name}: {} bytes", text.len());
    }
}

/// Each full-tile module is its parent's text with the ft edit and nothing
/// else: the anchor once in the parent, the inserted part before it and
/// one branch after it; taking both out gives the parent back. v9ft is
/// E1's v9 module with the same edit. The edit is refused on a text
/// without the anchor or with two.
#[test]
fn the_ft_modules_are_their_parents_with_the_full_tile_branch() {
    let (_, from, to) = pers_ft_edit();
    assert!(to.starts_with(PERS_FT_PART.1) && to.ends_with("        return;\n    }\n"));
    assert_eq!(to.matches("pers_attn_tile_full(").count(), 2, "one definition, one call");
    assert_eq!(to.matches("if (nv == PERS_TT) {").count(), 1);
    for (arm, parent) in [
        ("a12ipft", p4t_a12ip_unit_text(CARD_TILE_SM89)),
        ("a12ipgft", p4t_a12ipg_unit_text(CARD_TILE_SM89)),
        ("a12ipg32ft", p4t_a12ipg32_unit_text(CARD_TILE_SM89)),
    ] {
        let ft = p4t_ft_unit_text(CARD_TILE_SM89, arm).unwrap();
        assert_eq!(parent.matches(from.as_str()).count(), 1, "{arm}");
        assert_eq!(ft.replacen(to.as_str(), &from, 1), parent, "{arm}: the parent and the ft edit, nothing else");
    }
    let v9 = p4e_e1_unit_text(CARD_TILE_SM89, "v9").unwrap();
    let v9ft = p4e_e1_unit_text(CARD_TILE_SM89, "v9ft").unwrap();
    assert_eq!(v9ft.replacen(to.as_str(), &from, 1), v9, "v9ft: E1's v9 and the ft edit, nothing else");
    assert!(p4t_ft_unit_text(CARD_TILE_SM89, "a12ft").is_err());
    assert!(pers_ft_apply("no tile here").is_err());
    assert!(pers_ft_apply(&format!("{from}{from}")).is_err());
}

/// The ft modules at the card's tile define, at the sha256 the job scripts
/// check (ops/jobs/large-prefill-ft.sh, ops/jobs/large-attn-ft-4e.sh): an
/// edit to pers_attn_full.cuh moves these and must move the scripts too.
#[test]
fn the_ft_modules_keep_their_bytes() {
    for (name, text, sha) in [
        ("a12ipft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipft").unwrap(), "544f1dd5ff2dd0b7eb52ecb60002af677748dbf402182ea3d526a0ccabd73132"),
        ("a12ipgft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipgft").unwrap(), "e0fd30be4318032e371854eb4f485c14b232292481603ee2810001c0415086f2"),
        ("a12ipg32ft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipg32ft").unwrap(), "19b734a3fb6029848d87339c7d505e9bcfc612f182256793026ea964d09853e2"),
        ("4E v9ft", p4e_e1_unit_text(CARD_TILE_SM89, "v9ft").unwrap(), "22ff8320cb881b39690cbbfe6f1fd9ee8c1715142adfb4ee1ffbe932b061619a"),
        ("4E E1 v9", p4e_e1_unit_text(CARD_TILE_SM89, "v9").unwrap(), "d55a097b146e8b962829045dfea6975470b408b942c1e0c22c038ad4d9d00860"),
    ] {
        assert_eq!(crate::sha256::sha256_hex(text.as_bytes()), sha, "{name}: {} bytes", text.len());
    }
}

/// The rc arms' modules at the card's tile define, at the sha256 the R6
/// job script checks (ops/jobs/recherche-debit-jeton.sh, control 3): an
/// edit to pers_4e_rc.cuh or to the rc edits moves these and must move
/// the script and its prereg too.
#[test]
fn the_rc_modules_keep_their_bytes() {
    for (name, sha) in [
        ("v9ep", "f4166ccf9d857b30cae8f7691e9e4f83c2726218cb48c29dc512cb42b1a668dd"),
        ("v9fl", "9925b4ad769c791d2372d52a14b68e650a549ce15edd05eecae0dececfa4f03d"),
        // R6b's variants (ops/jobs/recherche-debit-charge.sh, recherche-debit-jeton2.sh)
        ("v9ep2", "a7d9a04d4bb963aac4ed831f1d7b41f6473e939dbc6c7c047d3b9e90b9801aa9"),
        ("v9ep3", "60dfa277ef82cf5c94b5a4b0b5745610324abac432c8ef4725ea96ba150714f7"),
        ("v9fl2", "d863288607ba961223b3ac8501b99f2e4c97dfc082cd237b80300df8b33dbd11"),
        ("v9fl3", "713ed5877aef63da831b8d149d31418889e42ac2580589294fa303e408ccf3fe"),
        // R6d's arms (ops/jobs/recherche-debit-jeton3.sh): 264,720, 264,835 and 264,835 bytes
        ("v9fn", "ce902a755173d05c4d50353e68d7bf62a326e13566fd4f7b84bf97e9933b6f4f"),
        ("v9fg", "36b617b0811d24cb8aa475fa9c3f0cd49ced1138c918a0fbdd7556ceac1c67bb"),
        ("v9eg", "1971b46969e4579c64327852a47c49887bd2acb0ffc5be802d2bed614117c29d"),
    ] {
        let text = p4e_e1_unit_text(CARD_TILE_SM89, name).unwrap();
        assert_eq!(crate::sha256::sha256_hex(text.as_bytes()), sha, "4E {name}: {} bytes", text.len());
    }
}

/// Writes the tree modules at the card's tile define into
/// `target/tree-modules/` for an offline compile (ops/tree-sass-probe.sh).
#[test]
#[ignore = "a dump tool, not a test: writes target/tree-modules/ for ops/tree-sass-probe.sh; run by hand with --ignored"]
fn dump_tree_modules() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/tree-modules");
    std::fs::create_dir_all(&dir).unwrap();
    let v9 = format!("{}{}", p4e_rows_define(7).unwrap(), pers_unit_text(CARD_TILE_SM89));
    for (name, text) in [
        ("4e-v9", v9),
        ("a12ip", p4t_a12ip_unit_text(CARD_TILE_SM89)),
        ("a12ipg", p4t_a12ipg_unit_text(CARD_TILE_SM89)),
        ("a12ipg32", p4t_a12ipg32_unit_text(CARD_TILE_SM89)),
        ("a12ipft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipft").unwrap()),
        ("a12ipgft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipgft").unwrap()),
        ("a12ipg32ft", p4t_ft_unit_text(CARD_TILE_SM89, "a12ipg32ft").unwrap()),
        ("4e-v9ft", p4e_e1_unit_text(CARD_TILE_SM89, "v9ft").unwrap()),
        ("4e-v9e1", p4e_e1_unit_text(CARD_TILE_SM89, "v9").unwrap()),
        ("4e-v9ep", p4e_e1_unit_text(CARD_TILE_SM89, "v9ep").unwrap()),
        ("4e-v9fl", p4e_e1_unit_text(CARD_TILE_SM89, "v9fl").unwrap()),
        ("4e-v9ep2", p4e_e1_unit_text(CARD_TILE_SM89, "v9ep2").unwrap()),
        ("4e-v9ep3", p4e_e1_unit_text(CARD_TILE_SM89, "v9ep3").unwrap()),
        ("4e-v9fl2", p4e_e1_unit_text(CARD_TILE_SM89, "v9fl2").unwrap()),
        ("4e-v9fl3", p4e_e1_unit_text(CARD_TILE_SM89, "v9fl3").unwrap()),
        ("4e-v9pe0", p4e_e1_unit_text(CARD_TILE_SM89, "v9pe0").unwrap()),
        ("4e-v9peq", p4e_e1_unit_text(CARD_TILE_SM89, "v9peq").unwrap()),
        ("4e-v9peg", p4e_e1_unit_text(CARD_TILE_SM89, "v9peg").unwrap()),
        ("4e-v9ped", p4e_e1_unit_text(CARD_TILE_SM89, "v9ped").unwrap()),
        ("4e-v9peo", p4e_e1_unit_text(CARD_TILE_SM89, "v9peo").unwrap()),
        ("4e-v9fn", p4e_e1_unit_text(CARD_TILE_SM89, "v9fn").unwrap()),
        ("4e-v9fg", p4e_e1_unit_text(CARD_TILE_SM89, "v9fg").unwrap()),
        ("4e-v9eg", p4e_e1_unit_text(CARD_TILE_SM89, "v9eg").unwrap()),
        ("4e-v9k8", p4e_e1_unit_text(CARD_TILE_SM89, "v9k8").unwrap()),
    ] {
        std::fs::write(dir.join(format!("{name}.cu")), &text).unwrap();
        eprintln!("{name}: {} bytes, sha256 {}", text.len(), crate::sha256::sha256_hex(text.as_bytes()));
    }
}

#[test]
fn the_a12ipg_pass_is_a12ip_with_the_grouped_attention() {
    // pers_4et_a12g.cu: a12's body renamed, its area check and its
    // attention call swapped for the grouped ones, nothing else
    let a12 = P4T_A12_PART.1;
    let head = "template <int FORM, u32 RC, bool IP = false>\n__device__ __forceinline__ void pers_4et_a12_body(";
    let c = a12.find(head).unwrap();
    let end = "#undef P4T_WAIT\n}\n";
    let d = c + a12[c..].find(end).unwrap() + end.len();
    let body = a12[c..d]
        .replacen(head, "template <int FORM, u32 RC, bool IP>\n__device__ __forceinline__ void pers_4et_a12g_body(", 1)
        .replacen("L.area_floats < p4t_area_floats(", "L.area_floats < p4g_area_floats(", 1)
        .replacen("        p4t_attention(A, L,", "        p4g_attention(A, L,", 1);
    assert_eq!(P4T_A12G_PART.1.matches(body.as_str()).count(), 1, "the a12ipg pass drifted from a12's");
    assert_eq!(P4T_A12G_PART.1.matches("pers_4et_a12g_body<FORM_, 2u, true>(A, L);").count(), 1);
    assert_eq!(P4T_A12G_PART.1.matches("#define P4G_RG 4u ").count(), 1, "P4G_RG mirrors the kernel's");
    let ip = p4t_a12ip_unit_text(SERVED_DEFINES_SM89);
    let g = p4t_a12ipg_unit_text(SERVED_DEFINES_SM89);
    assert_eq!(g, format!("#define P4T_A12IPG 1\n{ip}\n{}", P4T_A12G_PART.1));
    for f in 0..3 {
        assert_eq!(g.matches(p4t_a12ipg_symbol(f).unwrap()).count(), 1);
        assert_eq!(ip.matches(p4t_a12ipg_symbol(f).unwrap()).count(), 0, "a12ip's text carries no a12ipg symbol");
    }
    assert!(p4t_a12ipg_symbol(3).is_err());
    // the 4B at 12 warps: 18,624 floats, 93,568 B a CTA with the tables, under sm_89's 101,376
    let a = p4g_area_floats(12, 32, 9728, 4);
    assert_eq!(a, 18_624);
    assert_eq!(a, p4t_area_floats(12, 32, 9728, 4) + 3_680);
    assert!(SMEM_HEADER + TABLE_BYTES + 4 * a <= 101_376, "{} B", SMEM_HEADER + TABLE_BYTES + 4 * a);
    eprintln!("tree module a12ipg at SERVED_DEFINES_SM89: {} bytes, sha256 {}; area {a} floats, {} B shared", g.len(), crate::sha256::sha256_hex(g.as_bytes()), SMEM_HEADER + TABLE_BYTES + 4 * a);
}

#[test]
fn the_a12ipg32_pass_is_a12ipg_at_four_tiles() {
    // pers_4et_a12g32.cu: a12ipg's body instantiated at RC = 4, nothing else
    let part = P4T_A12G32_PART.1;
    assert_eq!(part.matches("pers_4et_a12g_body<").count(), 1);
    assert_eq!(part.matches("pers_4et_a12g_body<FORM_, 4u, true>(A, L);").count(), 1);
    assert_eq!(part.matches("__global__").count(), 1, "one kernel macro, no other code");
    let g = p4t_a12ipg_unit_text(SERVED_DEFINES_SM89);
    let t = p4t_a12ipg32_unit_text(SERVED_DEFINES_SM89);
    assert_eq!(t, format!("#define P4T_A12IPG32 1\n{g}\n{part}"));
    for f in 0..3 {
        assert_eq!(t.matches(p4t_a12ipg32_symbol(f).unwrap()).count(), 1);
        assert_eq!(g.matches(p4t_a12ipg32_symbol(f).unwrap()).count(), 0, "a12ipg's text carries no a12ipg32 symbol");
    }
    assert!(p4t_a12ipg32_symbol(3).is_err());
    // the area is a12ipg's, which already stages 32 rows
    assert_eq!(p4g_area_floats(12, 32, 9728, 4), 18_624);
    eprintln!("tree module a12ipg32 at SERVED_DEFINES_SM89: {} bytes, sha256 {}", t.len(), crate::sha256::sha256_hex(t.as_bytes()));
}

use super::e1::{
    p4e_rc_gu_edits, rc_padded_floats, RcGu, P4E_K8_PART, P4E_RC_ARMS, P4E_RC_GU_ARMS, P4E_RC_GU_TWO_CALLS,
};
use super::rows::trows_warp_rows;
use super::verify::{
    p4t_split, pers_ft_apply, P4T_A10RC_PART, P4T_A10T2_PART, P4T_A10T_PART, P4T_A10_PART, P4T_A11I_PART, P4T_A11_PART,
    P4T_A12G32_PART, P4T_A12G_PART, P4T_A12_PART, P4T_A4_PART, P4T_A5_PART, P4T_A6_PART, P4T_A8_PART, P4T_A9_PART,
    P4T_PART, P4T_STREAM_PART, P4V_PART, PERS_FT_PART,
};
use super::*;

#[test]
fn the_verify_phases_sum_their_events() {
    // two layers, three CTAs: 15 events; CTA 1 is the median at each
    let (layers, g) = (2u32, 3usize);
    let events = 1 + n_bar_4e(layers) as usize;
    let mut st = vec![0u64; events * g];
    let mut t = 100u64;
    for e in 0..events {
        t += 10 * (e as u64 + 1);
        st[e * g] = t - 3;
        st[e * g + 1] = t;
        st[e * g + 2] = t + 50;
    }
    let p = p4v_phases(&st, layers, g as u32).unwrap();
    // phase of event e lasts 10 (e + 1)
    assert_eq!(p[0], 20.0);
    assert_eq!(p[1], 30.0 + 90.0);
    assert_eq!(p[6], 80.0 + 140.0);
    assert_eq!(p[7], 150.0);
    assert_eq!(p.iter().sum::<f64>(), (st[(events - 1) * g + 1] - st[1]) as f64);
    assert!(p4v_phases(&st[1..], layers, g as u32).is_err());
    st[5] = 0;
    assert!(p4v_phases(&st, layers, g as u32).is_err());
}

#[test]
fn the_trows_unit_carries_every_arm() {
    let t = trows_unit_text("#define TILE_BLOCKS 64u");
    assert!(t.starts_with(&pers_unit_text("#define TILE_BLOCKS 64u")), "4E's module bytes come first, unchanged");
    for (name, sym, _, minb) in TROWS_ARMS {
        if name == "v2_s" || name == "v2r64" {
            continue; // the symbols of v2 and v2_o2, on other sites or CTAs a SM
        }
        let arm = sym.trim_start_matches("tv_trows_").trim_end_matches("_o2").to_uppercase();
        assert!(t.contains(&format!("TROWS_KERNEL({sym}, TROWS_{arm}, {minb})")), "{sym}");
    }
    assert_eq!(trows_smem_bytes(9728), 57_984, "4E's own shared bytes at d_in 9,728");
    assert_eq!(trows_k2_smem_bytes(), 96_896);
    assert!(t.contains("#define TROWS_X2_AT 9728u"), "K2's second activation sits after the widest first one");
    assert!(trows_small_only("v2_o2") && trows_small_only("v2_s") && !trows_small_only("v2"));
    assert!(!trows_small_only("v2r64"), "v2r64 runs every site at one CTA a SM");
    assert!(trows_counters("gb") && trows_counters("gbc") && !trows_counters("v2") && !trows_counters("bal"));
    // two sites, two CTAs, one warp: CTA busy at w = 1
    let st = [0u32, 10, 0, 30, 0, 20, 0, 20];
    assert!((trows_cta_spread(&st, 2, 2, 1) - 50.0 / 40.0).abs() < 1e-12, "slowest 30 + 20 over mean 20 + 20");
    assert!(t.contains("if (!pers_grid_wait(ctr, (s + 1u) * g, ctr + 1)) return;"), "gb and gbc wait on the grid after a site");
    assert_eq!(trows_cpa_smem_bytes(16), 67_200, "CPA's ring after 4E's own bytes");
    for (name, sym, _) in Q4ROWS_ARMS {
        assert!(t.contains(&format!("extern \"C\" __global__ void __launch_bounds__(512, 1) {sym}(")) || t.contains(&format!("Q4ROWS_KERNEL({sym}, ")), "{name}");
    }
    for (name, sym) in CTR_ARMS {
        assert!(t.contains(&format!("extern \"C\" __global__ void {sym}(")), "{name}");
    }
}

#[test]
fn the_tile_layout_keeps_the_served_bytes_at_the_4b_widths() {
    // 2,560, 4,096 and 9,728 columns: 106, 170 and 405 blocks
    for nb in [106u32, 170, 405] {
        assert_eq!(trows_tile_stride(nb), tetra_stride(nb), "nb {nb}");
        assert!(trows_cpa_ok(tetra_stride(nb)), "nb {nb}");
    }
    assert_eq!(trows_tile_stride(128), 192);
    assert_eq!(trows_tile_stride(5), 8);
    assert_eq!(trows_tile_stride(129), 196);
}

#[test]
fn the_int4_sites_are_the_84_records_of_the_4b() {
    let s = q4rows_sites();
    assert_eq!(s.len(), 84);
    assert_eq!(s.iter().filter(|r| r.3 == "down").count(), 12);
    let b = q4rows_bytes(&s);
    assert_eq!(b, 409_436_160, "0.409 GB, plan-300 §4");
    assert!(s.iter().all(|r| r.1 % 256 == 0));
    assert_eq!(ctr_expected("ctr8", 142, 3), [vec![24u32; 17], vec![18]].concat());
    assert_eq!(ctr_expected("bar", 142, 3), vec![426]);
}

#[test]
fn the_i16_check_reads_ulps_and_the_rms_bound() {
    // site 1: 1.0, 2.0, -1.0 ; site 2: 0.5, 1e-4 (a row near zero)
    let want = [0x3c00u16, 0x4000, 0xbc00, 0x3800, 0x068e];
    let same = vrows_i16_check(&want, &want, &[3, 2]);
    assert_eq!((same.same, same.one_ulp, same.within_rms, same.out), (5, 0, 0, 0));
    // one ulp up on 1.0, two ulps on 2.0 (out: 2 ulps of 2.0 is 3.9e-3 of
    // site 1's rms 1.41), a near-zero row moved by many ulps but 1e-4 of 0.35
    let got = [0x3c01u16, 0x4002, 0xbc00, 0x3800, 0x0600];
    let c = vrows_i16_check(&got, &want, &[3, 2]);
    assert_eq!((c.same, c.one_ulp, c.within_rms, c.out), (2, 1, 1, 1), "{c:?}");
    assert!(c.worst > 2e-3 && c.worst < 3e-3, "{c:?}");
    // the two zeros are one value
    assert_eq!(vrows_i16_check(&[0x8000], &[0x0000], &[1]).same, 1);
    assert_eq!(vrows_i16_check(&[0x8000], &[0x0000], &[1]).out, 0);
}

#[test]
fn the_verify_row_shared_bytes_fit_one_cta() {
    for (name, _, k, _) in VROWS_ARMS {
        let b = vrows_smem_bytes(name, k, 9728);
        assert!(b <= 101_376, "{name}: {b} B over the 101,376 a CTA may opt into on sm_89");
    }
    assert_eq!(vrows_smem_bytes("i16k4", 4, 9728), 256 + 19_584 + 4 * 20_320);
}

#[test]
fn trows_stamps_fold_by_class() {
    // 2 sites of d_in 2,560 (nit 4) and 1 of 9,728 (nit 13), grid 2, 4 warps.
    let sites = [(10u32, 2560u32), (6, 2560), (3, 9728)];
    let (grid, nw) = (2u32, 4u32);
    let mut st = vec![0u32; sites.len() * 2 * 5];
    for (s, &(d_out, _)) in sites.iter().enumerate() {
        for c in 0..grid {
            let lo = d_out * c / grid;
            let hi = d_out * (c + 1) / grid;
            let at = (s * 2 + c as usize) * 5;
            for w in 0..nw {
                st[at + w as usize] = 100 * trows_warp_rows(lo, hi, w, nw);
            }
            st[at + 4] = 100 * (hi - lo).div_ceil(nw);
        }
    }
    let cl = trows_classes(&st, &sites, grid, nw);
    assert_eq!(cl.len(), 2);
    assert_eq!((cl[0].d_in, cl[0].sites, cl[1].d_in, cl[1].sites), (2560, 2, 9728, 1));
    // site 0: 5 rows a CTA over 4 warps (2,1,1,1); site 1: 3 rows (1,1,1,0)
    assert_eq!(cl[0].steps, 4 * (5 + 5 + 3 + 3));
    assert_eq!(cl[0].busy_warps, 4 + 4 + 3 + 3);
    assert_eq!(cl[0].cycles_per_step(), 25.0);
    assert_eq!(cl[0].cta_cycles, 200 + 200 + 100 + 100);
    assert!((cl[0].balance(nw) - 1600.0 / 2400.0).abs() < 1e-12);
    // site 2: CTA 0 rows [0, 1), CTA 1 rows [1, 3)
    assert_eq!(cl[1].steps, 13 * 3);
    assert_eq!(trows_clock_mhz(&[0, 0, 2_520, 1_000, 10, 5, 2_530, 1_005]), 2_520.0);
}


#[test]
fn slices_partition_every_site() {
    for d in [16u32, 1024, 2560, 4096, 9728] {
        for grid in [1u32, 3, 4, 142, 284] {
            let sl = slices(d, grid, 1);
            assert_eq!(sl.len(), grid as usize);
            assert_eq!(sl.first().unwrap().0, 0);
            assert_eq!(sl.last().unwrap().1, d);
            assert!(sl.windows(2).all(|w| w[0].1 == w[1].0));
            let (mn, mx) = sl.iter().fold((u32::MAX, 0), |(a, b), &(l, h)| (a.min(h - l), b.max(h - l)));
            assert!(mx - mn <= 1, "rows balanced within one: {mn}..{mx}");
        }
    }
    let sl = slices(9728, 142, 8);
    assert!(sl.iter().all(|&(l, h)| l % 8 == 0 && h % 8 == 0));
    assert_eq!(sl.last().unwrap().1, 9728);
}

#[test]
fn bank_maps() {
    assert_eq!(scalar_conflict_ways(24), 8);
    assert_eq!(scalar_conflict_ways(25), 1);
    assert_eq!(scalar_conflict_ways(28), 4);
    assert_eq!(quad_conflict_ways(24), 2);
    assert_eq!(quad_conflict_ways(28), 1);
}

#[test]
fn inv_order_inverts_order() {
    for c in 0..24usize {
        assert_eq!(ORDER[INV_ORDER[c] as usize] as usize, c);
    }
}

/// The order the header carries, read out of the header, so a drift
/// between the kernel and this table fails here.
#[test]
fn order_is_the_headers() {
    let src = include_str!("../../kernels/llvq_tetra48.cuh");
    let body = src.split("TETRA48_ORDER[24] = {").nth(1).unwrap().split('}').next().unwrap();
    let hdr: Vec<u8> = body.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    assert_eq!(hdr, ORDER.to_vec());
    let src = include_str!("../../kernels/llvq_pers.cuh");
    let body = src.split("PERS_INV_ORDER[24] = {").nth(1).unwrap().split('}').next().unwrap();
    let inv: Vec<u8> = body.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    assert_eq!(inv, INV_ORDER.to_vec());
}

#[test]
fn staging_is_a_bijection() {
    for (stride, trio) in [(24, false), (25, false), (28, true)] {
        for d_in in [2560u32, 4096, 9728] {
            let mut seen = std::collections::HashSet::new();
            let bound = (d_in / 24) * stride + d_in % 24;
            for i in 0..d_in {
                let j = staged_index(i, d_in, stride, trio);
                assert!(j < bound);
                assert!(seen.insert(j));
            }
        }
    }
}

#[test]
fn bytes_match_the_bench_ledger() {
    let (w, f16, f32) = bare_file_bytes();
    assert_eq!(w, 908_328_960);
    // banc-252 sweep-64 prints 0.98 GB for the f32 arm
    assert!((f32 as f64 / 1e9 - 0.98).abs() < 0.005);
    assert!((f16 as f64 / 1e9 - 0.9467).abs() < 0.0005);
    assert!(f16 < f32 && w < f16);
}

#[test]
fn rotation_plans() {
    assert_eq!(rot_mk(2560).unwrap(), (512, 5));
    assert_eq!(rot_mk(9728).unwrap(), (512, 19));
    assert_eq!(rot_mk(4096).unwrap(), (4096, 1));
    assert_eq!(rot_mk(12288).unwrap(), (4096, 3));
    assert_eq!(rot_mk(5120).unwrap(), (1024, 5));
    assert_eq!(rot_mk(17408).unwrap(), (1024, 17));
    assert!(rot_mk(25600).is_ok());
    assert!(rot_mk(24 * 100).is_err());
}

#[test]
fn the_4e_schedule_has_the_design_counts() {
    for m in [ModelShape::Q4B, ModelShape::Q8B, ModelShape::Q14B] {
        assert_eq!(grid_waits_4e(&m), waits_4e(&m), "{m:?}");
        let s = schedule_4e(&m);
        assert_eq!(s.len() as u32, 2 + 4 * m.layers);
        assert_eq!(s.first(), Some(&(Phase4E::Embed, Wait4E::None)));
        assert_eq!(s.last(), Some(&(Phase4E::Head, Wait4E::Grid)));
        // every phase but the embedding and layer 0's q/k/v reads what
        // other CTAs wrote, and waits for it
        for (k, (p, w)) in s.iter().enumerate() {
            let free = matches!(p, Phase4E::Embed | Phase4E::Qkv(0));
            assert_eq!(*w == Wait4E::None, free, "phase {k}: {p:?}");
        }
    }
}

/// The card's three words as a 4E launch leaves them, for
/// [`p4e_settle`]: each barrier adds `grid` to the counter, a fault
/// raises its flag and stops every CTA at that barrier.
struct FakeCounters {
    bar: u32,
    abort: u32,
    nan: u32,
}

impl P4eCounters for FakeCounters {
    fn flags(&mut self) -> Result<(u32, u32), String> {
        Ok((self.abort, self.nan))
    }
    fn zero(&mut self) -> Result<(), String> {
        (self.bar, self.abort, self.nan) = (0, 0, 0);
        Ok(())
    }
}

impl FakeCounters {
    /// One launch of `barriers` barriers from the host's base. The
    /// kernel waits for `base + k * grid` at barrier k: a counter that
    /// does not start at the base either deadlocks (behind it, the
    /// watchdog aborts) or lets the barriers pass early (ahead of it).
    /// Returns the barriers it passed.
    fn launch(&mut self, base: u32, barriers: u32, grid: u32, fault: Option<(u32, P4eFault)>) -> u32 {
        if self.nan != 0 {
            return 0; // the head reads the flag and leaves at its first token
        }
        if self.bar != base {
            self.abort = 1; // early passes are a race; count them as the abort they risk
            return 0;
        }
        for k in 1..=barriers {
            if let Some((at, f)) = fault {
                if k == at {
                    match f {
                        P4eFault::Abort => self.abort = 1,
                        P4eFault::Nan => self.nan = 1,
                    }
                    self.bar = self.bar.wrapping_add(grid / 2); // some CTAs arrived
                    return k - 1;
                }
            }
            self.bar = self.bar.wrapping_add(grid);
        }
        barriers
    }
}

#[test]
fn a_faulted_4e_launch_does_not_poison_the_next() {
    let (grid, per) = (142u32, n_bar_4e(36));
    for fault in [P4eFault::Abort, P4eFault::Nan] {
        let mut dev = FakeCounters { bar: 0, abort: 0, nan: 0 };
        let mut base = 0u32;
        // a clean launch of 3 tokens
        assert_eq!(dev.launch(base, 3 * per, grid, None), 3 * per);
        assert_eq!(p4e_settle(&mut dev, &mut base), Ok(None));
        let (_, next) = p4e_after_launch(&[1, 2, 3], 3, base, 36, grid);
        base = next;
        assert_eq!(dev.bar, base, "the host's base follows the counter");
        // a launch that faults in its second token
        dev.launch(base, 3 * per, grid, Some((per + 5, fault)));
        assert_eq!(p4e_settle(&mut dev, &mut base), Ok(Some(fault)), "{fault:?} is reported");
        assert_eq!((dev.bar, base), (0, 0), "{fault:?}: counter and base start over together");
        // the next launches run, and stay in step
        for _ in 0..2 {
            assert_eq!(dev.launch(base, 2 * per, grid, None), 2 * per, "{fault:?}: the next launch runs");
            assert_eq!(p4e_settle(&mut dev, &mut base), Ok(None), "{fault:?}: the next launch is clean");
            base = p4e_after_launch(&[4, 5], 2, base, 36, grid).1;
            assert_eq!(dev.bar, base);
        }
    }
    // the NaN flag wins nothing over the abort: both set reads as Abort
    let mut dev = FakeCounters { bar: 7, abort: 1, nan: 1 };
    let mut base = 3u32;
    assert_eq!(p4e_settle(&mut dev, &mut base), Ok(Some(P4eFault::Abort)));
    assert_eq!((dev.bar, dev.abort, dev.nan, base), (0, 0, 0, 0));
    // a reset outside a fault (a launch the host cannot account for)
    let mut dev = FakeCounters { bar: 99, abort: 0, nan: 0 };
    let mut base = 42u32;
    p4e_reset(&mut dev, &mut base).unwrap();
    assert_eq!((dev.bar, base), (0, 0));
}

#[test]
fn the_4e_launch_bookkeeping() {
    assert_eq!(n_bar_4e(36), 218);
    let (made, next) = p4e_after_launch(&[5, 9, 2, u32::MAX], 4, 10, 36, 142);
    assert_eq!(made, vec![5, 9, 2]);
    assert_eq!(next, 10 + 3 * 218 * 142);
    // stopped on EOS after one token: one token's barriers
    let (made, next) = p4e_after_launch(&[7, u32::MAX, u32::MAX], 3, 0, 2, 3);
    assert_eq!((made, next), (vec![7], 14 * 3));
    // the shared bytes: the 4B's row vector (9,728 floats) is larger than
    // attention's 26,752 bytes; a narrow model takes attention's
    assert_eq!(p4e_smem_bytes(2560, 9728, 4), (SMEM_HEADER + TABLE_BYTES + 4 * 9728).div_ceil(16) * 16);
    let cuh = include_str!("../../kernels/llvq_pers.cuh");
    assert!(cuh.contains("#if PERS_4E_ROWS == 3\n#define PERS_4E_SCRATCH 16384u"), "p4e_scratch_bytes mirrors the scratch");
    assert!(cuh.contains("#elif PERS_4E_ROWS == 4\n#define PERS_4E_SCRATCH 4096u"), "form 4's scratch too");
    assert!(cuh.contains("#elif PERS_4E_ROWS >= 5\n#define PERS_4E_SCRATCH 7360u"), "forms 5 and 6's scratch too");
    assert!(cuh.contains("#define PERS_V7_FLOATS 1840u"), "form 5's scratch is the v7 area");
    let scratch: Vec<u32> = (1..=7).map(p4e_scratch_bytes).collect();
    assert_eq!(scratch, vec![0, 0, 16_384, 4_096, 7_360, 7_360, 7_360]);
    // the v7 area, and v8's ring, at the 4B, the 8B (4,096 wide, inter
    // 12,288, 4 groups) and the 14B (5,120, 17,408, 5 groups) fit under
    // sm_89's 101,376-byte opt-in
    for form in [5u32, 6, 7] {
        for (h, i, g) in [(2560u32, 9728u32, 4u32), (4096, 12_288, 4), (5120, 17_408, 5)] {
            let b = p4e_smem_bytes_form(form, h, i, g) + p4e_scratch_bytes(form);
            assert!(b <= 101_376, "form {form} at {h}/{i}/{g}: {b} B");
        }
    }
    assert_eq!(p4e_smem_bytes_form(6, 2560, 9728, 4), (SMEM_HEADER + TABLE_BYTES + 3 * 16_384 + 8_320 + 2_048).div_ceil(16) * 16);
    assert_eq!(p4e_smem_bytes_form(5, 2560, 9728, 4), p4e_smem_bytes(2560, 9728, 4));
    assert!(cuh.contains("#if PERS_4E_ROWS >= 6\n#define PERS_ATTN_RING 3u\n#else\n#define PERS_ATTN_RING 1u"), "p4e_attn_ring mirrors the ring");
    let inter_14b: u32 = 17_408;
    assert!(inter_14b.div_ceil(32) <= 576, "the 14B's down signs fit PERS_V7_SIGNS_MAX");
    // form 4's scratch holds every row of a KMAX x KMAX `small`
    assert_eq!(p4e_scratch_bytes(4) as usize, 4 * 32 * 32);
    assert_eq!(p4e_rows_define(1).unwrap(), "");
    assert_eq!(p4e_rows_define(3).unwrap(), "#define PERS_4E_ROWS 3u\n");
    assert_eq!(p4e_rows_define(4).unwrap(), "#define PERS_4E_ROWS 4u\n");
    assert_eq!(p4e_rows_define(5).unwrap(), "#define PERS_4E_ROWS 5u\n");
    assert_eq!(p4e_rows_define(6).unwrap(), "#define PERS_4E_ROWS 6u\n");
    assert_eq!(p4e_rows_define(7).unwrap(), "#define PERS_4E_ROWS 7u\n");
    assert!(p4e_rows_define(8).is_err());
    assert_eq!(p4e_smem_bytes_form(7, 2560, 9728, 4), p4e_smem_bytes_form(6, 2560, 9728, 4));
    assert!(P4E_BENCH_FORMS.iter().all(|f| P4E_ROWS_FORMS.contains(f)), "the bench forms are row forms");
    let ok = vec![("act".to_string(), 0x7000_0000u64), ("layer 0 ln1".to_string(), 0x7000_0100)];
    assert!(p4e_v6_inputs(&ok, 2560, 9728).is_ok());
    let bad = vec![("act".to_string(), 0x7000_0000u64), ("layer 3 ln2".to_string(), 0x7000_0108)];
    assert!(p4e_v6_inputs(&bad, 2560, 9728).unwrap_err().contains("layer 3 ln2"));
    assert!(p4e_v6_inputs(&ok, 2564, 9728).is_err());
    assert!(cuh.contains("#ifndef PERS_4E_ROWS\n#define PERS_4E_ROWS 1u"), "form 1 is the default");
    assert_eq!(p4e_smem_bytes(256, 512, 5), (SMEM_HEADER + TABLE_BYTES + 16384 + 10400 + 2560).div_ceil(16) * 16);
    let cuh = include_str!("../../kernels/llvq_pers.cuh");
    for line in [
        "#define PERS_STAGE_U4 (2u * PERS_STAGE * (PERS_HD / 8u))",
        "#define PERS_STAGE (PERS_UPS * PERS_TT)",
        "#define PERS_UPS 4u",
        "#define PERS_PSTRIDE 130u",
        "#define PERS_ATTN_SMEM(groups) (16u * PERS_STAGE_U4 * PERS_ATTN_RING + 4u * PERS_UPS * (groups) * PERS_PSTRIDE + 4u * (groups) * PERS_HD)",
    ] {
        assert!(cuh.contains(line), "p4e_smem_bytes mirrors {line:?}");
    }
    // a full ring past n is not read
    let (made, _) = p4e_after_launch(&[1, 2, 3], 2, 0, 1, 1);
    assert_eq!(made, vec![1, 2]);
    // across the wrap
    let (_, next) = p4e_after_launch(&[1], 1, u32::MAX - 1, 1, 1);
    assert_eq!(next, 6);
    assert_eq!(p4e_symbol(1, 512).unwrap(), "pers_4e_tokens_f1_t512");
    assert!(p4e_symbol(3, 768).is_err() && p4e_symbol(0, 1024).is_err());
    // the symbols the table names are the kernel's
    let cu = include_str!("../../kernels/pers_4e.cu");
    for f in 0..3u32 {
        for t in [768u32, 512] {
            let pf = p4e_rows_prefetch(t);
            assert!(cu.contains(&format!("PERS_4E({}, {f}, {t}, false, {pf})", p4e_symbol(f, t).unwrap())), "{f} {t}");
            assert!(cu.contains(&format!("PERS_4E({}, {f}, {t}, true, {pf})", p4e_stamp_symbol(f, t).unwrap())), "stamps {f} {t}");
        }
        assert!(p4e_stamp_symbol(f, 1024).is_err());
    }
    assert_eq!(p4e_events(36), 543);
}

/// The phase report on stamps made by hand: two CTAs, one layer, two
/// tokens; every phase's wall, slowest and mean work known in advance.
#[test]
fn the_4e_phase_report_reads_the_stamps() {
    let (layers, grid, n) = (1u32, 2usize, 2u32);
    let ev = p4e_events(layers) as usize; // 18
    // Per token, CTA 0 and CTA 1, in order: the embedding (a wait), then
    // the layer's 15 events. Each grid wait: CTA 0 works 10, CTA 1 works
    // 10 + 2k (k the wait's index), a release 1 after the last arrival.
    // The two inner stamps of q/k/v and MLP split their phase's work: q/k/v
    // 3 then the rest; MLP 2, then gate/up to the inner wait's arrival, the
    // inner wait, 1, the rest. The head then takes 7 on both.
    let mut st = vec![0u64; n as usize * ev * grid];
    let mut clock = 100u64;
    let put = |st: &mut Vec<u64>, t: usize, e: usize, v: [u64; 2]| {
        for c in 0..2 {
            st[(t * ev + e) * grid + c] = v[c];
        }
    };
    for t in 0..n as usize {
        let mut rel = [clock; 2];
        // embedding, q/k/v, norm, attention, o, MLP: arrival offsets
        let arrivals = [0usize, 3, 5, 7, 9, 15];
        for (k, &e) in arrivals.iter().enumerate() {
            let arr = [rel[0] + 10, rel[1] + 10 + 2 * k as u64];
            if k == 1 {
                put(&mut st, t, 2, [rel[0] + 3, rel[1] + 3]);
            }
            if k == 5 {
                put(&mut st, t, 11, [rel[0] + 2, rel[1] + 2]);
                let gu = [rel[0] + 5, rel[1] + 6];
                put(&mut st, t, 12, gu);
                let inner = gu[0].max(gu[1]) + 1;
                put(&mut st, t, 13, [inner, inner]);
                put(&mut st, t, 14, [inner + 1, inner + 1]);
            }
            let arr = if k == 5 { [arr[0].max(st[(t * ev + 14) * grid] + 1), arr[1].max(st[(t * ev + 14) * grid + 1] + 1)] } else { arr };
            let r = arr[0].max(arr[1]) + 1;
            put(&mut st, t, e, arr);
            put(&mut st, t, e + 1, [r, r]);
            rel = [r, r];
        }
        put(&mut st, t, ev - 1, [rel[0] + 7, rel[1] + 7]);
        clock = rel[0] + 7;
    }
    let r = p4e_phase_report(&st, n, layers, grid as u32).unwrap();
    assert_eq!(r.tokens_read, 1);
    assert_eq!(r.grain, 1);
    for (k, p) in r.phases.iter().enumerate().take(5) {
        let w = 10.0 + 2.0 * k as f64;
        assert_eq!(p.crit.0, w, "{}", p.name);
        assert_eq!(p.mean.0, (10.0 + w) / 2.0, "{}", p.name);
        assert_eq!(p.wall.0, w + 1.0, "{}", p.name);
    }
    assert_eq!(r.phases[6].wall.0, 7.0);
    let walls: f64 = r.phases.iter().map(|p| p.wall.0).sum();
    assert_eq!(walls, r.token.0);
    // the parts add up to their phase's mean work, CTA by CTA
    let sum = |lo: usize, hi: usize| r.subs[lo..hi].iter().map(|p| p.mean.0).sum::<f64>();
    assert_eq!(sum(0, 2), r.phases[1].mean.0, "q/k/v parts");
    assert_eq!(sum(2, 7), r.phases[5].mean.0, "MLP parts");
    assert_eq!(r.subs[0].mean.0, 3.0, "ln1 + rotation");
    assert_eq!(r.subs[2].mean.0, 2.0, "ln2 + rotation");
    assert_eq!(r.subs[3].crit.0, 4.0, "gate/up: CTA 1's 6 - 2");
    assert_eq!(r.subs[4].mean.0, 1.5, "the inner wait: 2 on CTA 0, 1 on CTA 1");
    assert_eq!(r.subs[5].mean.0, 1.0, "the down's input");
    // a release before the last arrival of its wait is refused
    let mut bad = st.clone();
    bad[(ev + 4) * grid] = bad[(ev + 3) * grid + 1] - 1;
    assert!(p4e_phase_report(&bad, n, layers, grid as u32).is_err());
    // so is an inner release before a gate/up arrival
    let mut bad = st.clone();
    bad[(ev + 13) * grid] = bad[(ev + 12) * grid + 1] - 1;
    assert!(p4e_phase_report(&bad, n, layers, grid as u32).is_err());
    // an unwritten stamp is refused
    let mut bad = st.clone();
    bad[ev * grid + 5] = 0;
    assert!(p4e_phase_report(&bad, n, layers, grid as u32).is_err());
    assert!(p4e_phase_report(&st[..ev * grid], 1, layers, grid as u32).is_err(), "one token reads nothing");
}

#[test]
fn the_4e_wait_counts() {
    assert_eq!(waits_4e(&ModelShape::Q4B), 97);
    assert_eq!(waits_4e(&ModelShape::Q8B), 128);
    assert_eq!(waits_4e(&ModelShape::Q14B), 102);
    assert_eq!(step_a_plan(&ModelShape::Q4B).iter().filter(|p| p.tetra_down).count(), 24);
    assert_eq!(step_a_plan(&ModelShape::Q8B).iter().filter(|p| p.tetra_down).count(), 19);
    assert_eq!(step_a_plan(&ModelShape::Q14B).iter().filter(|p| p.tetra_down).count(), 21);
}

#[test]
fn the_served_unit_is_the_published_one() {
    let t = served_unit_text(SERVED_DEFINES_SM89);
    assert_eq!(t.len(), 101_625);
    assert_eq!(
        crate::sha256::sha256_hex(t.as_bytes()),
        "6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812",
        "the served unit moved: the L2 reference would not be the served kernel"
    );
}

/// 4E v9's dynamic shared bytes at the three sealed shapes, against the
/// 101,376 B a CTA may ask on sm_89: the 14B's int4 down input (17,408
/// floats) fits, which is why the 14B's refusal is its untied head and
/// not its width. The v7 area holds every width's sign words (576).
#[test]
fn the_4e_v9_shared_bytes_at_the_three_sealed_shapes() {
    let at = |hidden: u32, inter: u32, groups: u32| p4e_smem_bytes_form(7, hidden, inter, groups) + p4e_scratch_bytes(7);
    assert_eq!(at(2560, 9728, 4), 85_952, "the 4B, as step3-4e printed");
    assert_eq!(at(4096, 12288, 4), 85_952);
    assert_eq!(at(5120, 17408, 5), 96_064);
    for (h, i) in [(2560u32, 9728u32), (4096, 12288), (5120, 17408)] {
        assert!(at(h, i, 5) <= 101_376);
        assert!(i.div_ceil(32) <= 576 && h.div_ceil(32) <= 576, "the v7 area at {h}/{i}");
    }
}

/// 4E v9's module at the L40S's tile, the one the served-v9 doors gate
/// on: the step-3 module with the head's own pointers and the Tetra o
/// phase added (step3-4e timed d9da09aa…, 250,057 B).
#[test]
fn the_4e_v9_module_is_the_one_the_doors_gate_on() {
    let t = format!("{}{}", p4e_rows_define(7).unwrap(), pers_unit_text("#define TILE_BLOCKS 64u\n"));
    assert_eq!(t.len(), 252_757);
    assert_eq!(
        crate::sha256::sha256_hex(t.as_bytes()),
        "1b422b62328be6752df4add43280565fd2a49cfa20ee26953b301abe7a15c530",
        "the 4E v9 module moved: the served-v9 doors gate on this digest"
    );
}

/// E1's reference module is the served v9 text with the logits edits
/// only: the token and stamp symbols' text is v9's, the template
/// parameter defaults to false, and three logits symbols follow.
#[test]
fn the_e1_reference_is_v9_and_its_logits_symbols() {
    let tile = "#define TILE_BLOCKS 64u\n";
    let v9 = format!("{}{}", p4e_rows_define(7).unwrap(), pers_unit_text(tile));
    let r = p4e_e1_unit_text(tile, "v9").unwrap();
    assert_ne!(r, v9);
    // the reference minus the logits edits is v9 byte for byte
    let mut back = r.clone();
    for (_, from, to) in p4e_e1_edits("v9").unwrap() {
        assert_eq!(back.matches(to.as_str()).count(), 1, "{to:?}");
        back = back.replacen(to.as_str(), &from, 1);
    }
    assert_eq!(back, v9, "the E1 reference is v9 and the logits edits, nothing else");
    for f in 0..3 {
        assert!(r.contains(&format!("PERS_4E_LOGITS({}, {f})", p4e_logits_symbol(f).unwrap())));
    }
    assert!(p4e_logits_symbol(3).is_err());
    assert!(p4e_e1_unit_text(tile, "v9x").is_err());
}

/// v9ep and v9fl are E1's v9 reference with the rc edits and nothing
/// else: undoing the edits gives the reference back byte for byte; the body
/// is inserted once with its define; no call site of v9's warp lists is
/// left on a Tetra row phase of the 512-thread path; the served v9 module
/// and the served unit keep their pins (their own tests).
#[test]
fn the_rc_arms_are_v9_with_the_body_on_every_tetra_phase() {
    let tile = "#define TILE_BLOCKS 64u\n";
    let r = p4e_e1_unit_text(tile, "v9").unwrap();
    for (arm, body) in P4E_RC_ARMS {
        let t = p4e_e1_unit_text(tile, arm).unwrap();
        let mut back = t.clone();
        for (_, from, to) in p4e_rc_edits(body).unwrap() {
            assert_eq!(back.matches(to.as_str()).count(), 1, "{arm}: {to:?}");
            back = back.replacen(to.as_str(), &from, 1);
        }
        assert_eq!(back, r, "{arm}: E1's v9 and the rc edits, nothing else");
        assert_eq!(t.matches(&format!("#define PERS_RC_ARM {body}u\n")).count(), 1);
        assert_eq!(t.matches("#ifndef PERS_4E_RC_CUH").count(), 1);
        assert_eq!(t.matches("#ifndef PERS_4E_RC2_CUH").count(), usize::from(body >= 3), "{arm}");
        let call = if body >= 3 { "pers_rc2_rows<PERS_RC_ARM>(" } else { "pers_rc_rows<PERS_RC_ARM>(" };
        assert_eq!(t.matches(call).count(), 4, "{arm}: q/k, gate/up, down, o");
        assert_eq!(t.matches("_rows<PERS_RC_ARM>(").count(), 4, "{arm}: no call of the other family");
        // pers_4e_rows keeps its definition and no caller
        assert_eq!(t.matches("pers_4e_rows(").count(), 1, "{arm}");
        assert_eq!(p4e_e1_owes_bits(arm), !rc_body_is_flat(body), "{arm}");
        assert_eq!(p4e_rc_body(arm), Some(body));
    }
    assert!(p4e_rc_edits(0).is_err() && p4e_rc_edits(7).is_err());
    for arm in ["v9", "v9np", "v9mv", "v9ft", "v9ep", "v9ep2", "v9ep3", "v9eg"] {
        assert!(p4e_e1_owes_bits(arm), "{arm}");
    }
    for arm in ["v9fl", "v9fl2", "v9fl3", "v9fn", "v9fg"] {
        assert!(!p4e_e1_owes_bits(arm), "{arm}");
    }
    // every research arm is an E1 arm
    assert!(P4E_RC_ARMS.iter().all(|a| P4E_E1_ARMS.contains(&a.0)));
    // R6d: E1's v9 plus exactly the body's edits 0, 1, 3, 4 and, at gate/up, nothing (v9fn) or the two calls
    let gu_v9 = p4e_rc_edits(2).unwrap()[2].1.clone();
    for (arm, body, gu) in P4E_RC_GU_ARMS {
        assert!(P4E_E1_ARMS.contains(&arm) && !P4E_RC_ARMS.iter().any(|a| a.0 == arm), "{arm}");
        assert_eq!(p4e_rc_body(arm), Some(body), "{arm}");
        assert_eq!(p4e_e1_owes_bits(arm), !rc_body_is_flat(body), "{arm}");
        let full = p4e_rc_edits(body).unwrap();
        let edits = p4e_rc_gu_edits(arm).unwrap();
        let want: Vec<_> = match gu {
            RcGu::V9 => full.iter().enumerate().filter(|(i, _)| *i != 2).map(|(_, e)| e.clone()).collect(),
            RcGu::TwoCalls => {
                let mut w = full.clone();
                w[2].2 = P4E_RC_GU_TWO_CALLS.to_string();
                w
            }
        };
        assert_eq!(edits, want, "{arm}: the body's edits, gate/up aside");
        let t = p4e_e1_unit_text(tile, arm).unwrap();
        let mut back = t.clone();
        for (_, from, to) in &edits {
            assert_eq!(back.matches(to.as_str()).count(), 1, "{arm}: {to:?}");
            back = back.replacen(to.as_str(), from, 1);
        }
        assert_eq!(back, r, "{arm}: E1's v9 and its edits, nothing else");
        assert_eq!(t.matches(&format!("#define PERS_RC_ARM {body}u\n")).count(), 1, "{arm}");
        assert_eq!(t.matches("#ifndef PERS_4E_RC_CUH").count(), 1, "{arm}");
        assert_eq!(t.matches("#ifndef PERS_4E_RC2_CUH").count(), 0, "{arm}");
        assert_eq!(t.matches("pers_rc2_rows<PERS_RC_ARM>(").count(), 0, "{arm}");
        let (calls, loops, keeps_v9) = match gu {
            RcGu::V9 => (3, 2, 1),
            RcGu::TwoCalls => (5, 1, 0),
        };
        assert_eq!(t.matches("pers_rc_rows<PERS_RC_ARM>(").count(), calls, "{arm}: the rc calls");
        assert_eq!(t.matches("pers_4e_rows(").count(), loops, "{arm}: pers_4e_rows' definition and its callers");
        assert_eq!(t.matches(gu_v9.as_str()).count(), keeps_v9, "{arm}: v9's gate/up text");
        assert_eq!(t.matches(P4E_RC_GU_TWO_CALLS).count(), 1 - keeps_v9, "{arm}: the two-call site");
        // the one-call site's shared values are in no R6d module
        assert!(!t.contains("gu_vals") && !t.contains("sgu[np + p]"), "{arm}");
    }
    assert!(p4e_rc_gu_edits("v9fl").is_err() && p4e_rc_gu_edits("v9").is_err());
    // the load probes: endpass's text and one call site each, v9 elsewhere; none is an E1 arm
    let ep = p4e_rc_edits(1).unwrap();
    assert_eq!(ep.len(), 5, "the text, q/k, gate/up, down, o");
    for (arm, site) in P4E_RC_PROBES {
        assert!(!P4E_E1_ARMS.contains(&arm) && p4e_rc_body(arm).is_none(), "{arm}");
        let t = p4e_e1_unit_text(tile, arm).unwrap();
        assert_eq!(t.matches("#define PERS_RC_ARM 1u\n").count(), 1, "{arm}");
        assert_eq!(t.matches("pers_rc_rows<PERS_RC_ARM>(").count(), usize::from(site > 0), "{arm}");
        assert_eq!(t.matches("pers_4e_rows(").count(), 1 + 4 - usize::from(site > 0), "{arm}: v9's row loop on the other sites");
        let mut back = t.clone();
        for (i, (_, from, to)) in ep.iter().enumerate() {
            assert_eq!(back.matches(to.as_str()).count(), usize::from(i == 0 || i == site), "{arm}: edit {i}");
            back = back.replace(to.as_str(), from);
        }
        assert_eq!(back, r, "{arm}: E1's v9, the text and one site, nothing else");
    }
}

/// Where the rc bodies sit in 4E v9's buffer at the three sealed shapes
/// (*computed*): the 4B and the 8B fit every phase in the attention
/// phase's region, the 14B's down input fills its row vector to the v7
/// area and is refused by name, for both bodies. The numbers are the
/// shared-memory report of the R6 prereg.
#[test]
fn the_rc_layout_at_the_three_sealed_shapes() {
    assert_eq!(rc_padded_floats(2560), 2576);
    assert_eq!(rc_padded_floats(9728), 9780);
    assert_eq!(rc_padded_floats(4096), 4120);
    assert_eq!(rc_padded_floats(12288), 12352);
    let ends = |body: u32, h: u32, i: u32, ctx: u32, o: bool, g: u32| {
        p4e_rc_layout(body, h, i, ctx, o, true, g, 142).map(|v| v.iter().map(|p| (p.phase, p.end, p.limit)).collect::<Vec<_>>())
    };
    // the 4B: 142 CTAs, 69 gate/up pairs at most
    assert_eq!(ends(1, 2560, 9728, 4096, false, 4).unwrap(), vec![("q/k", 7168, 14880), ("gate/up", 7237, 14880), ("down", 11828, 14880)]);
    assert_eq!(ends(2, 2560, 9728, 4096, false, 4).unwrap(), vec![("q/k", 9744, 14880), ("gate/up", 9813, 14880), ("down", 11828, 14880)]);
    // the 8B: its Tetra o at the context's width
    assert_eq!(
        ends(2, 4096, 12288, 4096, true, 4).unwrap(),
        vec![("q/k", 14360, 14880), ("gate/up", 14447, 14880), ("down", 14400, 14880), ("o", 6168, 14880)]
    );
    assert!(ends(1, 4096, 12288, 4096, true, 4).is_ok());
    // the 14B: refused by name, both bodies; flat's q/k (17,436 floats) already passes the v7 area at 17,408
    for (body, phase) in [(1, "the down phase"), (2, "the q/k phase")] {
        let e = ends(body, 5120, 17408, 5120, false, 5).unwrap_err();
        assert!(e.contains(phase) && e.contains("refused"), "{e}");
    }
    assert!(p4e_rc_layout(7, 2560, 9728, 4096, false, true, 4, 142).is_err());
    // the variants: their families' regions; the one-chunk bodies fit the 4B on 142 CTAs (largest slice: flat's
    // gate/up, 138 rows x 14 runs = 1,932 slots), and fl2, fl3 are refused on the 8B (174 x 22 = 3,828)
    assert_eq!(ends(3, 2560, 9728, 4096, false, 4).unwrap(), ends(1, 2560, 9728, 4096, false, 4).unwrap());
    assert_eq!(ends(4, 2560, 9728, 4096, false, 4).unwrap(), ends(1, 2560, 9728, 4096, false, 4).unwrap());
    for body in [5, 6] {
        assert_eq!(ends(body, 2560, 9728, 4096, false, 4).unwrap(), ends(2, 2560, 9728, 4096, false, 4).unwrap());
        let e = ends(body, 4096, 12288, 4096, true, 4).unwrap_err();
        assert!(e.contains("one chunk") && e.contains("gate/up phase's slice needs 3828"), "{e}");
    }
    assert!(ends(4, 4096, 12288, 4096, true, 4).is_ok(), "ep3 on the 8B: 174 rows a slice");
    // the twin's grids of 3 to 5 CTAs do not fit one chunk: the one-chunk arms are twin-tested on 142
    assert!(p4e_rc_layout(5, 2560, 9728, 4096, false, true, 4, 3).is_err());
}

/// The arms differ from the reference in prefetch lines only
/// ([`p4e_e1_structure`]), and each does what it says: np issues no
/// prefetch in the attention phase, mv issues its own and keeps v9's
/// down prefetch and layer 0's q/k/v prefetch.
#[test]
fn the_e1_arms_differ_from_v9_in_prefetch_lines_only() {
    let tile = "#define TILE_BLOCKS 64u\n";
    let r = p4e_e1_unit_text(tile, "v9").unwrap();
    let np = p4e_e1_unit_text(tile, "v9np").unwrap();
    let mv = p4e_e1_unit_text(tile, "v9mv").unwrap();
    p4e_e1_structure(&r, &r).unwrap();
    p4e_e1_structure(&r, &np).unwrap();
    p4e_e1_structure(&r, &mv).unwrap();
    assert_eq!(r.matches("prefetch_next();").count(), 2);
    for t in [&np, &mv] {
        assert_eq!(t.matches("prefetch_next();").count(), 0, "no o or gate/up prefetch in the attention phase");
        assert_eq!(t.matches("pers_l2_prefetch(A->down.words").count(), 1, "the down prefetch stays");
    }
    assert!(!np.contains("pers_e1_"));
    assert_eq!(mv.matches("if (IN4E) pers_e1_prefetch_staged();").count(), 1);
    assert_eq!(mv.matches("pers_e1_stage_next(Ag->layers, l + 1u,").count(), 1);
    assert_eq!(mv.matches("if (l == 0u) {  // e1-pf").count(), 1);
    // the served v9 module's digest is pinned by
    // the_4e_v9_module_is_the_one_the_doors_gate_on, which this does not touch
}

/// v9k8's module is E1's v9 with the int8 cache's edits and nothing else:
/// undoing them gives the reference back byte for byte; the k8 text is in
/// once; v9's append and attention calls are gone from the token loop, so
/// the layer's kc and vc (the int8 caches' addresses in this arm) reach
/// only the k8 calls; the fill symbol is defined once.
#[test]
fn the_k8_module_is_v9_with_the_int8_cache() {
    let tile = "#define TILE_BLOCKS 64u\n";
    let r = p4e_e1_unit_text(tile, "v9").unwrap();
    let t = p4e_e1_unit_text(tile, "v9k8").unwrap();
    let mut back = t.clone();
    for (_, from, to) in p4e_k8_edits() {
        assert_eq!(back.matches(to.as_str()).count(), 1, "{to:?}");
        back = back.replacen(to.as_str(), &from, 1);
    }
    assert_eq!(back, r, "v9k8: E1's v9 and the k8 edits, nothing else");
    assert_eq!(t.matches("#ifndef LLVQ_PERS_K8_CUH").count(), 1);
    assert_eq!(t.matches("pers_attention_k8<PF ? 2u : 1u>(").count(), 1);
    assert_eq!(t.matches("pers_attention<PF ? 2u : 1u>(").count(), 0);
    assert_eq!(t.matches("pers_4e_norm_rope_append_k8<FORM>(Y.qkv.q").count(), 1);
    assert_eq!(t.matches("pers_4e_norm_rope_append<FORM>(").count(), 0);
    assert_eq!(t.matches("Y.kc").count(), 2, "kc reaches the append and the attention, nothing else");
    assert_eq!(t.matches("Y.vc").count(), 2);
    assert_eq!(t.matches(&format!("void {P4E_K8_FILL_SYMBOL}(")).count(), 1);
    assert_eq!(t.matches(&format!("#define PERS_K8_BLK {P4E_K8_BLK}u")).count(), 1);
    assert_eq!(P4E_K8_BLK as usize, crate::kv8_ref::BLK);
    assert!(p4e_e1_k8("v9k8") && !p4e_e1_k8("v9") && P4E_E1_ARMS.contains(&"v9k8"));
    assert!(!p4e_e1_owes_bits("v9k8"));
    assert_eq!(p4e_e1_debt("v9"), "the bits");
    assert_eq!(p4e_e1_debt("v9fl"), "a bound on rows");
    assert!(p4e_e1_debt("v9k8").starts_with("the int8 cache's values"));
}

/// v9k8's launch rule: the handoff after a prefill fills the prefix
/// once; a second handoff at the same prompt fills nothing; a continuation
/// fills nothing and keeps `hand`; a later handoff fills only what the
/// eager path appended; a moved cache refills from 0; the capacity grows
/// by 256 and never shrinks; anything else is refused.
#[test]
fn the_load_rule_waives_local_memory_only_when_asked() {
    // the rule every served and measured module has loaded under
    assert!(p4e_token_symbol_loads(0, 1, false));
    assert!(!p4e_token_symbol_loads(8, 1, false), "8 B of local memory is refused");
    assert!(!p4e_token_symbol_loads(0, 0, false), "no CTA a SM is refused");
    // the measurement: local memory passes, a module that does not fit a SM never does
    assert!(p4e_token_symbol_loads(8, 1, true));
    assert!(p4e_token_symbol_loads(0, 1, true));
    assert!(!p4e_token_symbol_loads(8, 0, true), "the waiver is on local memory alone");
}

#[test]
fn the_k8_launch_rule() {
    let m = 8704;
    let s0 = K8State::default();
    // the handoff after an 8,192-token prefill, 64 tokens
    let a = p4e_k8_step(s0, true, 8192, 8192, 64, m).unwrap();
    assert_eq!(a, K8Step { grow: true, copy_old: false, fill: Some((0, 8192)), next: K8State { cap: 8448, hand: 8192, own: 0 } });
    let after = K8State { own: 8192 + 64, ..a.next };
    // the harness truncates and hands the same prompt again: nothing to fill
    let b = p4e_k8_step(after, false, 8192, 8192, 64, m).unwrap();
    assert_eq!((b.grow, b.fill, b.next.hand), (false, None, 8192));
    // the arm continues its own decode past the binary16 caches' length
    let c = p4e_k8_step(after, false, 8192, 8256, 200, m).unwrap();
    assert_eq!((c.grow, c.copy_old, c.fill, c.next.hand, c.next.cap), (true, true, None, 8192, 8704));
    // the eager path appended 10 positions, then hands them over
    let d = p4e_k8_step(after, false, 8202, 8202, 8, m).unwrap();
    assert_eq!((d.fill, d.next.hand), (Some((8192, 8202)), 8202));
    // a new context: the buffers moved, every position refilled, the capacity kept
    let e = p4e_k8_step(after, true, 5, 5, 64, m).unwrap();
    assert_eq!((e.grow, e.fill, e.next), (false, Some((0, 5)), K8State { cap: 8448, hand: 5, own: 0 }));
    // refused: a gap, a continuation from elsewhere, a moved cache mid-decode, past the scratch
    assert!(p4e_k8_step(after, false, 8192, 8300, 8, m).is_err());
    assert!(p4e_k8_step(after, false, 8100, 8192, 8, m).is_err());
    assert!(p4e_k8_step(after, true, 8192, 8256, 8, m).is_err());
    assert!(p4e_k8_step(s0, true, 8192, 8192, 600, m).is_err());
    assert!(p4e_k8_step(s0, false, 0, 7, 1, m).is_err(), "no continuation before a first launch");
}

#[test]
fn the_k8_release_rule() {
    let m = 8704;
    let s0 = K8State::default();
    // the hand-over after an 8,192-token prefill: everything filled, the buffers made
    let a = p4e_k8_step_release(s0, true, 8192, 8192, 64, m).unwrap();
    assert_eq!(a, K8Step { grow: true, copy_old: false, fill: Some((0, 8192)), next: K8State { cap: 8448, hand: 8192, own: 0 } });
    let after = K8State { own: 8192 + 64, ..a.next };
    // the arm continues with no binary16 cache: no fill, the capacity kept
    let b = p4e_k8_step_release(after, false, 0, 8256, 64, m).unwrap();
    assert_eq!(b, K8Step { grow: false, copy_old: false, fill: None, next: after });
    // past the capacity: grown, what the arm wrote copied
    let c = p4e_k8_step_release(after, false, 0, 8256, 300, m).unwrap();
    assert_eq!((c.grow, c.copy_old, c.fill, c.next.cap, c.next.own), (true, true, None, 8704, 8256));
    // a new prompt prefilled into fresh caches: refilled from 0, the capacity kept
    let d = p4e_k8_step_release(after, true, 5, 5, 64, m).unwrap();
    assert_eq!(d, K8Step { grow: false, copy_old: false, fill: Some((0, 5)), next: K8State { cap: 8448, hand: 5, own: 0 } });
    // refused: a gap at hand-over, an empty prefill, a continuation beside binary16 buffers,
    // a continuation from elsewhere, no launch before, past the scratch
    assert!(p4e_k8_step_release(s0, true, 8100, 8192, 8, m).is_err());
    assert!(p4e_k8_step_release(s0, true, 0, 0, 8, m).is_err());
    assert!(p4e_k8_step_release(after, true, 8192, 8256, 8, m).is_err());
    assert!(p4e_k8_step_release(after, false, 0, 8300, 8, m).is_err());
    assert!(p4e_k8_step_release(s0, false, 0, 0, 8, m).is_err());
    assert!(p4e_k8_step_release(s0, true, 8192, 8192, 600, m).is_err());
}

/// The code lines (comments and blank lines dropped, each trimmed) of the
/// function whose definition starts at `head` in `text`, to its closing
/// brace at column 0.
fn code_lines(text: &str, head: &str) -> Vec<String> {
    let a = text.find(head).unwrap_or_else(|| panic!("no {head:?}"));
    let b = a + text[a..].find("\n}\n").expect("a closing brace");
    text[a..b + 3]
        .lines()
        .map(|l| if l.trim_start().starts_with('#') { l.trim() } else { l.split("//").next().unwrap_or("").trim() })
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// Lines of `a` not matched one for one in `b`.
fn lines_minus(a: &[String], b: &[String]) -> Vec<String> {
    let mut left: Vec<&String> = b.iter().collect();
    a.iter()
        .filter(|l| match left.iter().position(|x| x == l) {
            Some(i) => {
                left.remove(i);
                false
            }
            None => true,
        })
        .cloned()
        .collect()
}

/// The int8 attention is section 10's text with the reader swapped, and
/// nothing else: against pers_attention, pers_attn_tile and
/// pers_attn_issue, the lines each side holds alone are exactly these (the
/// signatures, the head's address, the staging map, the reader, and v9's
/// register path below form 6, which the k8 text does not carry). A line of
/// arithmetic that drifts in either text fails here.
#[test]
fn the_k8_attention_is_section_10_with_the_int8_reader() {
    let v9 = PERS_UNIT_PARTS.iter().find(|p| p.0 == "llvq_pers.cuh").unwrap().1;
    let k8 = P4E_K8_PART.1;
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<String>>();
    let cases: [(&str, &str, Vec<String>, Vec<String>); 3] = [
        (
            "__device__ __forceinline__ void pers_attention(",
            "__device__ __forceinline__ void pers_attention_k8(",
            s(&[
                "__device__ __forceinline__ void pers_attention(const unsigned short* q, const unsigned short* kc,",
                "const unsigned short* vc, u32 n_heads, u32 groups, u32 cap, u32 l,",
                "const u32 rounds = (PERS_STAGE_U4 + PF * blockDim.x - 1u) / (PF * blockDim.x);",
                "const unsigned short* kh = kc + (unsigned long long)h * cap * PERS_HD;",
                "const unsigned short* vh = vc + (unsigned long long)h * cap * PERS_HD;",
                "#if PERS_ATTN_RING > 1",
                "pers_attn_issue(stage + st * PERS_STAGE_U4, kh, vh, base, st, nsteps, tpu, l);",
                "(void)rounds;",
                "#else",
                "uint4 pf[PF];",
                "if (rounds == 1u) pers_attn_load<PF>(pf, kh, vh, base, 0u, tpu, l, 0u);",
                "#endif",
                "#if PERS_ATTN_RING > 1",
                "#else",
                "uint4* const sbuf = stage;",
                "__syncthreads();",
                "if (rounds == 1u) {",
                "pers_attn_store<PF>(pf, stage, 0u);",
                "} else {",
                "for (u32 r = 0; r < rounds; ++r) {",
                "pers_attn_load<PF>(pf, kh, vh, base, st, tpu, l, r * PF * blockDim.x);",
                "pers_attn_store<PF>(pf, stage, r * PF * blockDim.x);",
                "}",
                "}",
                "__syncthreads();",
                "if (rounds == 1u && st + 1u < nsteps) pers_attn_load<PF>(pf, kh, vh, base, st + 1u, tpu, l, 0u);",
                "#endif",
                "const u32 sl = (u * tpu + t) * PERS_TT;",
                "pers_attn_tile(qs + ge * PERS_HD, ust + (u * groups + ge) * PERS_PSTRIDE, own, sbuf + sl * 16u,",
                "sbuf + (PERS_STAGE + sl) * 16u, nv, c, lane);",
                "#if PERS_ATTN_RING > 1",
                "pers_attn_issue(sbuf, kh, vh, base, st + PERS_ATTN_RING, nsteps, tpu, l);",
                "#endif",
                "#if PERS_ATTN_RING > 1",
                "#endif",
            ]),
            s(&[
                "__device__ __forceinline__ void pers_attention_k8(const unsigned short* q, const unsigned char* kc8,",
                "const unsigned char* vc8, u32 n_heads, u32 groups, u32 cap, u32 l,",
                "const unsigned char* kh = kc8 + (unsigned long long)h * (cap / PERS_TT) * PERS_K8_BLK;",
                "const unsigned char* vh = vc8 + (unsigned long long)h * (cap / PERS_TT) * PERS_K8_BLK;",
                "pers_attn_issue_k8(stage + st * PERS_STAGE_U4, kh, vh, base, st, nsteps, tpu, l);",
                "const unsigned char* const sb8 = reinterpret_cast<const unsigned char*>(sbuf);",
                "const u32 ts = u * tpu + t;",
                "pers_attn_tile_k8(qs + ge * PERS_HD, ust + (u * groups + ge) * PERS_PSTRIDE, own,",
                "sb8 + ts * PERS_K8_BLK, sb8 + (PERS_UPS + ts) * PERS_K8_BLK, nv, c, lane);",
                "pers_attn_issue_k8(sbuf, kh, vh, base, st + PERS_ATTN_RING, nsteps, tpu, l);",
            ]),
        ),
        (
            "__device__ __forceinline__ void pers_attn_tile(",
            "__device__ __forceinline__ void pers_attn_tile_k8(",
            s(&[
                "__device__ __forceinline__ void pers_attn_tile(const float* qrow, float* row, bool own, const uint4* ks,",
                "const uint4* vs, u32 nv, float c, u32 lane)",
                "const u32 l16 = lane & 15u, dl = l16 * 8u;",
                "pers_unpack8(ks[j * 16u + l16], kf);",
                "pers_unpack8(vs[j * 16u + l16], vf);",
            ]),
            s(&[
                "__device__ __forceinline__ void pers_attn_tile_k8(const float* qrow, float* row, bool own, const unsigned char* kb,",
                "const unsigned char* vb, u32 nv, float c, u32 lane)",
                "const u32 l16 = lane & 15u, dl = l16 * 8u, gsb = PERS_K8_CODES + (l16 >> 3) * 4u;",
                "pers_k8_unpack8(kb + j * PERS_HD + dl, kb + gsb + j * 8u, kf);",
                "pers_k8_unpack8(vb + j * PERS_HD + dl, vb + gsb + j * 8u, vf);",
            ]),
        ),
        (
            "__device__ __forceinline__ void pers_attn_issue(",
            "__device__ __forceinline__ void pers_attn_issue_k8(",
            s(&[
                "__device__ __forceinline__ void pers_attn_issue(uint4* stage, const unsigned short* kh, const unsigned short* vh,",
                "for (u32 idx = threadIdx.x; idx < PERS_STAGE_U4; idx += blockDim.x) {",
                "const u32 rem = idx % (PERS_STAGE * 16u), pl = rem >> 4;",
                "const u32 tl = pl / PERS_TT, u = tl / tpu, t = tl - u * tpu;",
                "const u32 p = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT + pl % PERS_TT;",
                "const unsigned short* src = (idx < PERS_STAGE * 16u ? kh : vh);",
                "const bool valid = p < lr;",
                "reinterpret_cast<const uint4*>(src + (unsigned long long)(valid ? p : 0u) * PERS_HD) + (rem & 15u),",
            ]),
            s(&[
                "__device__ __forceinline__ void pers_attn_issue_k8(uint4* stage, const unsigned char* kh, const unsigned char* vh,",
                "for (u32 idx = threadIdx.x; idx < PERS_K8_STAGE_U4; idx += blockDim.x) {",
                "const u32 rem = idx % PERS_K8_HALF_U4, ts = rem / PERS_K8_BLK_U4, ch = rem - ts * PERS_K8_BLK_U4;",
                "const u32 u = ts / tpu, t = ts - u * tpu;",
                "const u32 p0 = base + u * PERS_UNIT + (st * tpu + t) * PERS_TT;",
                "const unsigned char* src = idx < PERS_K8_HALF_U4 ? kh : vh;",
                "const bool valid = p0 < lr;",
                "reinterpret_cast<const uint4*>(src + (unsigned long long)(valid ? p0 / PERS_TT : 0u) * PERS_K8_BLK) + ch,",
            ]),
        ),
    ];
    let sorted = |mut v: Vec<String>| {
        v.sort();
        v
    };
    for (h9, h8, only9, only8) in cases {
        let (a, b) = (code_lines(v9, h9), code_lines(k8, h8));
        assert_eq!(sorted(lines_minus(&a, &b)), sorted(only9), "{h8}: the lines only v9 holds");
        assert_eq!(sorted(lines_minus(&b, &a)), sorted(only8), "{h8}: the lines only the k8 text holds");
    }
}

/// The v9k8 module at the card's tile define, at the sha256 its job
/// script checks: an edit to pers_4e_k8.cuh or to the k8 edits moves it.
#[test]
fn the_k8_module_keeps_its_bytes() {
    let text = p4e_e1_unit_text(CARD_TILE_SM89, "v9k8").unwrap();
    assert_eq!(crate::sha256::sha256_hex(text.as_bytes()), "8b53b1cdb8d4ff6264fea1685a355045231655f8d6ac1a368c89a8b605c224f9", "4E v9k8: {} bytes", text.len());
}

/// The structural check kills what it must: a line of arithmetic added
/// untagged or tagged, a line dropped, a barrier or a foreign store in
/// the helpers' block.
#[test]
fn the_e1_structure_check_kills_its_mutants() {
    let tile = "#define TILE_BLOCKS 64u\n";
    let r = p4e_e1_unit_text(tile, "v9").unwrap();
    let mv = p4e_e1_unit_text(tile, "v9mv").unwrap();
    let once = |t: &str, from: &str, to: &str| {
        assert_eq!(t.matches(from).count(), 1, "{from:?}");
        t.replacen(from, to, 1)
    };
    let anchor = "            PERS_WAIT(e0 + 5u);\n";
    for (tag, m) in [
        ("untagged", once(&mv, anchor, &format!("{anchor}            v[0] = 0.0f;\n"))),
        ("tagged store", once(&mv, anchor, &format!("{anchor}            v[0] = 0.0f;  // e1-pf\n"))),
        ("tagged prefetch store", once(&mv, anchor, &format!("{anchor}            v[0] = 0.0f;  // prefetch\n"))),
        ("dropped line", once(&mv, anchor, "")),
        ("helper barrier", once(&mv, "    if (l >= n_layers) return;\n", "    if (l >= n_layers) return;\n    __syncthreads();\n")),
        ("helper store", once(&mv, "    if (l >= n_layers) return;\n", "    if (l >= n_layers) return;\n    layers[0].qkv.q[0] = 0u;\n")),
        ("helper deref store", once(&mv, "    if (l >= n_layers) return;\n", "    if (l >= n_layers) return;\n    *pers_hdr() = 0u;\n")),
        ("helper open", once(&mv, "// e1-pf end\n", "")),
    ] {
        assert!(p4e_e1_structure(&r, &m).is_err(), "structure mutant {tag} survived");
    }
}

/// The verify pass's module at the L40S's tile: the 4E v9 module the
/// served doors gate on, unchanged, then `pers_4ev.cu`. Pinned at the
/// module spec-final ran (288,366 B at d7686d86, with the phase stamps of
/// 486fa45); spec-gate and spec-lookup ran the earlier 287,744 B at ab162e7f.
#[test]
fn the_verify_module_is_the_v9_module_and_the_verify_pass() {
    let tile = "#define TILE_BLOCKS 64u\n";
    let v9 = format!("{}{}", p4e_rows_define(7).unwrap(), pers_unit_text(tile));
    let t = p4v_unit_text(tile);
    assert!(t.starts_with(&v9) && t[v9.len()..] == format!("\n{}", P4V_PART.1), "the verify module is the v9 module and pers_4ev.cu");
    assert_eq!(t.len(), 288_366);
    assert_eq!(
        crate::sha256::sha256_hex(t.as_bytes()),
        "d7686d86b990b079c47a754ccb5675e090c0b9d19e1b3710558967614c3944e5",
        "the verify module moved: the speculative gate gates on this digest"
    );
}

/// The unit a v9 config compiles: the served unit unchanged, then the
/// tiled int4 readers. Pinned so the door job's digest gate is this text.
#[test]
fn the_tiled_served_unit_is_the_served_unit_and_the_tile_readers() {
    let (plain, tiled) = (served_unit_text(SERVED_DEFINES_SM89), served_unit_text_tiled(SERVED_DEFINES_SM89));
    assert!(tiled.starts_with(&plain), "the served unit's text moved under v9");
    for sym in ["tv_q4_ht(", "tv_q4_hht("] {
        assert_eq!(tiled.matches(sym).count(), 1, "{sym}");
        assert!(!plain.contains(sym), "{sym} leaked into the other configs' unit");
    }
    assert_eq!(tiled.len(), 106_947);
    assert_eq!(
        crate::sha256::sha256_hex(tiled.as_bytes()),
        "e269ea5d11b7e40ba0682f536f52ed9f09abfaa9fd6f3185eb306edca5c60e53",
        "the v9 served unit moved: the served-v9 door gates on this digest"
    );
}

#[test]
fn the_pers_unit_carries_every_symbol() {
    let t = pers_unit_text("#define TILE_BLOCKS 64u\n");
    for sym in ["tv_tetra48_persall", "tv_pers_bar", "tv_pers_ring", "tv_pers_empty", "pers_ref_rmsnorm", "pers_seg_tetra_v0", "pers_seg_q4_v0"] {
        assert!(t.contains(sym), "{sym}");
    }
    // the pass's variants: each symbol defined once, with the flags its
    // name says, and the tables-in-shared column agreeing with the name
    for (name, sym, ts) in PERSALL_VARIANTS.iter().skip(1) {
        let (pf, l2) = (name.contains("_pf"), name.contains("_l2"));
        assert_eq!(*ts, name.contains("_ts"), "{name}");
        assert!(t.contains(&format!("TOCC_PERSALL_V({sym}, {pf}, {ts}, {l2})")), "{sym}");
    }
    assert!(t.contains("extern \"C\" __global__ void tv_tetra48_persall(const unsigned long long* __restrict__ sites,"));
    // composition: every part's guard is defined by an earlier part
    let pos = |n: &str| PERS_UNIT_PARTS.iter().position(|p| p.0 == n).unwrap();
    assert!(pos("matvec.cu") < pos("llvq_pers.cuh"));
    assert!(pos("llvq_tetra48.cuh") < pos("tetra48_persall.cu"));
    assert!(pos("llvq_rot.cuh") < pos("llvq_pers.cuh"));
    assert!(pos("llvq_pers.cuh") < pos("pers_segment.cu"));
    assert!(pos("pers_segment.cu") < pos("pers_4e.cu"));
    for sym in ["pers_4e_tokens_f0", "pers_4e_tokens_f1", "pers_4e_tokens_f2"] {
        assert!(t.contains(sym), "{sym}");
    }
}

#[test]
fn persall_sites_pack_as_the_kernel_unpacks() {
    let w = persall_site_words(1, 2, 3, 4, 5, 6, 405, 8, 1216, 1234, 320);
    assert_eq!(w[6] & 0xffff, 405);
    assert_eq!((w[6] >> 16) & 0xffff, 8);
    assert_eq!(w[6] >> 32, 1216);
    assert_eq!(w[7] & 0xffff_ffff, 1234);
    assert_eq!(w[7] >> 32, 320);
    let cu = include_str!("../../kernels/tetra48_persall.cu");
    assert!(cu.contains("#define LLVQ_TOCC_SITE_WORDS 8u"));
    let (g0, total) = persall_groups(&[6144, 2560, 19456, 2560]);
    assert_eq!(g0, vec![0, 768, 1088, 3520]);
    assert_eq!(total, 3840);
}

#[test]
fn med_range_is_the_middle() {
    assert_eq!(med_range(&[3.0, 1.0, 2.0]), (2.0, 1.0, 3.0));
    assert_eq!(med_range(&[4.0, 1.0, 2.0, 3.0]), (2.5, 1.0, 4.0));
    assert!(med_range(&[]).0.is_nan());
}

#[test]
fn the_f64_row_reference() {
    // one block, y = e_0 * 4 (n2 = 16), gain 0 centroid 2: w_0 = 2 * 4 / 4 = 2
    let mut y = [0i32; 24];
    y[0] = 4;
    let x: Vec<f32> = (0..26).map(|i| i as f32 + 1.0).collect();
    let (v, m) = tetra_row_f64(1, |_| (y, 0), [2.0, 3.0], 0.5, &[0.25, -1.0], &x);
    assert_eq!(v, 2.0 * 0.5 * 1.0 + 0.25 * 25.0 - 26.0);
    assert_eq!(m, 1.0 + 6.25 + 26.0);
}
