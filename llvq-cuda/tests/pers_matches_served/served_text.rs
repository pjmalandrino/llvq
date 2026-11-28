//! The served text itself: rotation, norm, Tetra and int4 rows, the argmax key,
//! the struct mirrors and the fused kernel site by site. One file because each
//! gate here holds a persistent building block against its served original.

use super::*;

#[test]
fn the_rotation_prologue_is_rot_apply() {
    for &n in &ROT_WIDTHS {
        for threads in [64u32, 256] {
            let cmd = rot_cmd(n, threads, 0x5eed_0000 + n as u64);
            let a = run(served(), &cmd);
            let b = run(persistent(), &cmd);
            assert_eq!(a.len(), 4 * n as usize);
            assert!(a == b, "n = {n}, {threads} threads: the persistent rotation differs from rot_apply");
        }
    }
}

/// A column whose last real term underflows to -0: rot_mix's padded terms
/// turn it into +0, and so must the persistent mix's single `+ 0.0f`.
#[test]
fn the_mix_turns_minus_zero_into_plus_zero_as_the_padding_does() {
    let (m, k) = (32u32, 3u32);
    let mut s = vec![0.0f32; (k * m) as usize];
    let mut small = vec![0.0f32; KMAX * KMAX];
    for j in 0..m as usize {
        s[2 * m as usize + j] = 1e-30;
        s[j] = if j % 2 == 0 { 0.0 } else { 0.5 };
    }
    for g in 0..k as usize {
        small[g * KMAX + 2] = -1e-20; // -1e-50 underflows to -0 under round to nearest
        small[g * KMAX] = if g == 0 { 1.0 } else { 0.0 };
    }
    let cmd = Cmd::new(b'M').u32(m).u32(k).f32(1.0).f32s(&s).f32s(&small).0;
    let a = run(served(), &cmd);
    let b = run(persistent(), &cmd);
    assert_eq!((-1e-20f32).mul_add(1e-30, 0.0).to_bits(), 0x8000_0000, "the fixture's last term rounds to -0");
    assert!(a == b, "the persistent mix differs from rot_mix");
    assert!(u32s(&a).contains(&0), "some output is +0 (the column the padding fixed)");
    let mutant = mutant("mix_no_fadd", "llvq-cuda/kernels/llvq_pers.cuh", "if (k < LLVQ_ROT_KMAX) acc = __fadd_rn(acc, 0.0f);", "");
    let c = run(&mutant, &cmd);
    assert!(a != c, "dropping the + 0.0f survived the -0 column");
}

#[test]
fn rotation_mutants_are_killed() {
    let cmd = rot_cmd(4096, 128, 7);
    let cmd2 = rot_cmd(2560, 128, 9);
    let want = run(served(), &cmd);
    let want2 = run(served(), &cmd2);
    let cases = [
        ("wht_lane_stage_order", "for (u32 len = 1u; len < 16u; len <<= 1) {", "for (u32 len = 8u; len >= 1u; len >>= 1) {"),
        ("wht_upper_sign", "v[i] = upper ? (p - v[i]) : (v[i] + p);", "v[i] = upper ? (v[i] - p) : (v[i] + p);"),
        ("wht_shfl_order", "for (u32 sh = 1u; sh < 32u; sh <<= 1) {", "for (u32 sh = 16u; sh >= 1u; sh >>= 1) {"),
        ("smem_stage_start", "for (u32 len = 512u; len < m; len <<= 1) {", "for (u32 len = 1024u; len < m; len <<= 1) {"),
        ("scale_k1", "for (u32 i = threadIdx.x; i < n; i += blockDim.x) s[i] = s[i] * inv;", "for (u32 i = threadIdx.x; i < n; i += blockDim.x) s[i] = s[i];"),
        ("rot_sign", "s[i] = ((signbits[i >> 5] >> (i & 31u)) & 1u) ? -v : v;\n    }\n}\n\n// Phases", "s[i] = ((signbits[i >> 5] >> (i & 30u)) & 1u) ? -v : v;\n    }\n}\n\n// Phases"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, "llvq-cuda/kernels/llvq_pers.cuh", from, to);
        let killed = run(&b, &cmd) != want || run(&b, &cmd2) != want2;
        assert!(killed, "rotation mutant {tag} survived");
    }
}

fn norm_cmd(ncols: u32, variant: u32, threads: u32, seed: u64) -> Vec<u8> {
    Cmd::new(b'N').u32(ncols).f32(1e-6).u32(variant).u32(threads).u64(seed).0
}

/// candle's rmsnorm: block 1,024 from 1,024 columns, 32 below (q/k-norm at
/// 128), both contraction variants, on CTAs of 64 to 768 threads.
#[test]
fn the_norm_is_candles_tree() {
    let mut variants_differ = false;
    for ncols in [2560u32, 4096, 5120, 1024, 1000, 128] {
        for threads in [64u32, 256, 768] {
            for seed in [1u64, 2] {
                let a0 = run(served(), &norm_cmd(ncols, 0, threads, seed));
                let a1 = run(served(), &norm_cmd(ncols, 1, threads, seed));
                variants_differ |= a0 != a1;
                assert!(a0 == run(persistent(), &norm_cmd(ncols, 0, threads, seed)), "ncols {ncols}, {threads} threads, variant 0");
                assert!(a1 == run(persistent(), &norm_cmd(ncols, 1, threads, seed)), "ncols {ncols}, {threads} threads, variant 1");
            }
        }
    }
    // xi is a widened binary16 (11 significant bits), so xi * xi is exact in
    // binary32 and the contracted and split accumulations round the same
    // sums: the variant nvcc picked for candle cannot matter.
    assert!(!variants_differ, "the contracted and split norms differ on binary16 inputs");
}

#[test]
fn norm_mutants_are_killed() {
    let mut cmds: Vec<Vec<u8>> = Vec::new();
    for seed in 3..9u64 {
        for (n, v) in [(2560u32, 0u32), (128, 0), (5120, 1), (4096, 0)] {
            cmds.push(norm_cmd(n, v, 256, seed));
        }
    }
    let want: Vec<Vec<u8>> = cmds.iter().map(|c| run(served(), c)).collect();
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    let cases = [
        ("norm_butterfly_up", "for (int mask = 16; mask > 0; mask >>= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));\n        if (lane == 0u) s_sum[vw] = tmp;", "for (int mask = 1; mask < 32; mask <<= 1) tmp = __fadd_rn(tmp, __shfl_xor_sync(0xffffffffu, tmp, mask));\n        if (lane == 0u) s_sum[vw] = tmp;"),
        ("norm_block_768", "const u32 block = ncols < 1024u ? 32u : 1024u;", "const u32 block = ncols < 1024u ? 32u : 768u;"),
        ("norm_block_switch", "const u32 block = ncols < 1024u ? 32u : 1024u;", "const u32 block = ncols < 128u ? 32u : 1024u;"),
        ("norm_mean_mul", "const float mean = __fdiv_rn(tot, (float)ncols);", "const float mean = __fmul_rn(tot, 1.0f / (float)ncols);"),
        ("norm_apply_order", "return f2h(__fmul_rn(__fmul_rn(scale, h2f(x)), h2f(alpha)));", "return f2h(__fmul_rn(scale, __fmul_rn(h2f(x), h2f(alpha))));"),
        ("norm_square_rounded", "[&](u32 col) { return h2f(__ldcg(x + col)); }", "[&](u32 col) { return __fmul_rn(h2f(__ldcg(x + col)), 1.0009765625f); }"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = cmds.iter().zip(&want).any(|(c, w)| &run(&b, c) != w);
        assert!(killed, "norm mutant {tag} survived");
    }
}

fn rows_cmd(d_out: u32, d_in: u32, variant: u32, grid: u32, threads: u32, seed: u64) -> Vec<u8> {
    Cmd::new(b'T').u32(d_out).u32(d_in).u32(variant).u32(grid).u32(threads).u64(seed).0
}

/// The replay of `tv_tetra48_h` with nothing fused IS the served text: the
/// same lane order, tiling, butterfly and epilogue, bit for bit on y and on
/// the f32 value before `f2h`. With the three PTX fusions it differs on some
/// rows: the sites are exercised.
#[test]
fn the_served_replay_is_the_served_text() {
    let mut fused_differs = false;
    for &(d_out, d_in) in &[(64u32, 2560u32), (48, 4096), (40, 9728), (16, 120)] {
        let out = run(served(), &Cmd::new(b'U').u32(d_out).u32(d_in).u64(0x7e7a_0000 + d_in as u64).0);
        let k = 6 * d_out as usize;
        assert_eq!(out.len(), 3 * k);
        assert!(out[..k] == out[k..2 * k], "{d_out}x{d_in}: the unfused replay is not the served text");
        fused_differs |= out[k..2 * k] != out[2 * k..];
    }
    assert!(fused_differs, "the fused and unfused replays never differ: the three sites are not exercised");
}

/// A persistent Tetra row equals the served row, as NVVM compiled it, in
/// every staging layout and grid: y and the f32 value before `f2h`.
#[test]
fn tetra_rows_are_the_served_rows() {
    for &(d_out, d_in) in &[(64u32, 2560u32), (48, 4096), (40, 9728), (16, 120)] {
        let seed = 0x7e7a_0000 + d_in as u64;
        let want = run(served(), &rows_cmd(d_out, d_in, 0, 1, 64, seed));
        assert_eq!(want.len(), 6 * d_out as usize, "y in binary16, then the f32 value before f2h");
        for variant in 0..4u32 {
            for &(grid, threads) in &[(3u32, 96u32), (2, 64)] {
                let got = run(persistent(), &rows_cmd(d_out, d_in, variant, grid, threads, seed));
                assert!(got == want, "{d_out}x{d_in}, variant {variant}, grid {grid}x{threads}: rows or their f32 values differ");
            }
        }
    }
}

#[test]
fn tetra_row_mutants_are_killed() {
    let cmds: Vec<Vec<u8>> = [(64u32, 2560u32), (40, 9728)].iter().map(|&(o, i)| rows_cmd(o, i, 2, 3, 96, 11)).collect();
    let want: Vec<Vec<u8>> = cmds.iter().map(|c| run(served(), c)).collect();
    let f = "llvq-cuda/kernels/llvq_pers.cuh";
    let cases = [
        ("site1_unfused", "return __fmaf_rn(__fmul_rn(acc_b, gscale[g]), invnorm[m], acc);", "return __fadd_rn(acc, __fmul_rn(__fmul_rn(acc_b, gscale[g]), invnorm[m]));"),
        ("site1_assoc", "return __fmaf_rn(__fmul_rn(acc_b, gscale[g]), invnorm[m], acc);", "return __fmaf_rn(acc_b, __fmul_rn(gscale[g], invnorm[m]), acc);"),
        ("site2_unfused", "for (u32 i = 0; i < tail_w; ++i) tv = __fmaf_rn(h2f(tail_row[i]), xt[i], tv);", "for (u32 i = 0; i < tail_w; ++i) tv = __fadd_rn(tv, __fmul_rn(h2f(tail_row[i]), xt[i]));"),
        ("site3_unfused", "return __fmaf_rn(acc, rscale, tv);", "return __fadd_rn(__fmul_rn(acc, rscale), tv);"),
        ("trio_component", "a = __fmaf_rn(f1r_v3_float(q[i], 1u), v.y, a);", "a = __fmaf_rn(f1r_v3_float(q[i], 1u), v.z, a);"),
        ("inv_order", "0, 1, 2, 3, 4, 16, 8, 5, 17, 18,", "0, 1, 2, 3, 4, 8, 16, 5, 17, 18,"),
        ("lane_order", "for (u32 j = lane; j < nblocks; j += 32u) {\n            u32 lo, hi16;", "for (u32 j = nblocks - 1u - lane; j < nblocks; j -= 32u) {\n            u32 lo, hi16;"),
    ];
    for (tag, from, to) in cases {
        let b = mutant(tag, f, from, to);
        let killed = cmds.iter().zip(&want).any(|(c, w)| &run(&b, c) != w);
        assert!(killed, "Tetra row mutant {tag} survived");
    }
}

#[test]
fn int4_rows_are_the_served_rows() {
    for &(d_out, d_in) in &[(64u32, 9728u32), (40, 2560), (16, 4096)] {
        let cmd = Cmd::new(b'Q').u32(d_out).u32(d_in).u32(3).u32(96).u64(0x9400 + d_in as u64).0;
        assert!(run(served(), &cmd) == run(persistent(), &cmd), "{d_out}x{d_in}");
    }
    let cmd = Cmd::new(b'Q').u32(64).u32(2560).u32(3).u32(96).u64(5).0;
    let want = run(served(), &cmd);
    // Fusing the dequantization is an EQUIVALENT mutant: an 11-bit scale
    // times a 4-bit code is exact in binary32. Checked as such.
    let b = mutant("q4_deq_fused", "llvq-cuda/kernels/llvq_pers.cuh", "return __fadd_rn(__fmul_rn(h2f((unsigned short)sbits), (float)q), h2f((unsigned short)bbits));", "return __fmaf_rn(h2f((unsigned short)sbits), (float)q, h2f((unsigned short)bbits));");
    assert!(run(&b, &cmd) == want, "the fused dequantization differs: the exactness argument is wrong");
    let b = mutant("q4_group", "llvq-cuda/kernels/llvq_pers.cuh", "const u32 g = s0 + (c >> 7);", "const u32 g = s0 + (c >> 6) / 2u + (wi & 1u);");
    assert!(run(&b, &cmd) != want, "a wrong group index survived");
    let b = mutant("q4_lane_order", "llvq-cuda/kernels/llvq_pers.cuh", "for (u32 wi = lane; wi < nwords; wi += 32u) {\n            const u32 p = wq[w0 + wi];\n            const u32 c = wi << 3;\n            const u32 g", "for (u32 wi = nwords - 1u - lane; wi < nwords; wi -= 32u) {\n            const u32 p = wq[w0 + wi];\n            const u32 c = wi << 3;\n            const u32 g");
    assert!(run(&b, &cmd) != want, "a reversed int4 lane order survived");
    // the PREFETCH form (odd rows of the command): its next word's group, and
    // the hand-over of the word in flight
    let b = mutant("q4_pf_group", "llvq-cuda/kernels/llvq_pers.cuh", "sb2 = scales[s0 + ((wi + 32u) >> 4)];", "sb2 = scales[s0 + ((wi + 32u) >> 5)];");
    assert!(run(&b, &cmd) != want, "a wrong group of the prefetched word survived");
    let b = mutant("q4_pf_stale", "llvq-cuda/kernels/llvq_pers.cuh", "            p = p2;\n            sb = sb2;", "            (void)p2;\n            sb = sb2;");
    assert!(run(&b, &cmd) != want, "a prefetched word never handed over survived");
}

#[test]
fn the_argmax_key_is_tok_argmax_key() {
    let cmd = Cmd::new(b'K').0;
    assert_eq!(u32s(&run(persistent(), &cmd)), vec![0], "the persistent key differs from tok_argmax.cu's");
    let b = mutant("key_negzero", "llvq-cuda/kernels/llvq_pers.cuh", "if (b == 0x8000u) b = 0u;\n    return (b & 0x8000u)", "return (b & 0x8000u)");
    assert_ne!(u32s(&run(&b, &cmd)), vec![0], "the -0 fold of the key survived");
    let b = mutant("key_brev", "llvq-cuda/kernels/llvq_pers.cuh", "(1023u - (__brev(i & 1023u) >> 22))", "(1023u - (i & 1023u))");
    assert_ne!(u32s(&run(&b, &cmd)), vec![0], "the bit reversal of the key survived");
}

/// The host mirrors of the device structs (`pers/launch.rs`, `pers/verify.rs`) against the kernels'
/// own definitions: a field one word off would read a wrong pointer on the
/// card, with no error.
#[test]
fn the_struct_mirrors_match_the_kernel_layouts() {
    let o = u32s(&run(persistent(), &Cmd::new(b'L').0));
    let mut want = vec![48];
    want.extend([0, 8, 16, 24, 32, 36, 40, 44]);
    want.push(pers::SEG_ARGS_BYTES as u32);
    want.extend([0, 8, 16, 24, 32, 40, 88, 136, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288]);
    want.extend([296, 300, 304, 308, 312, 316, 320, 324, 328, 332]);
    want.push(pers::SEG_LAUNCH_BYTES as u32);
    want.extend([0, 8, 16]);
    want.push(pers::P4E_QKV_BYTES as u32);
    want.extend([40, 72, 80, 104, 112, 120, 128, 144]);
    want.push(pers::P4E_LAYER_BYTES as u32);
    want.extend([24, 32, pers::P4E_LAYER_QKV as u32, 192, 240, 248, 252, pers::P4E_LAYER_MLP as u32, pers::P4E_LAYER_OT as u32]);
    want.push(pers::P4E_ARGS_BYTES as u32);
    // invnorm, attn_ticket, n_layers, cap, eps, c, head_w, head_b
    want.extend([160, 168, 176, 204, 208, 212, pers::P4E_ARGS_HEAD as u32, pers::P4E_ARGS_HEAD as u32 + 16]);
    want.push(pers::P4E_OT_BYTES as u32);
    // t, signs, small, m, k, inv
    want.extend([0, 48, 56, 64, 68, 72]);
    want.push(pers::P4E_LAUNCH_BYTES as u32);
    want.extend([16, 24, 32]);
    assert_eq!(o, want);
    assert_eq!(std::mem::size_of::<pers::P4eLaunch>(), pers::P4E_LAUNCH_BYTES);
    let lay = pers::P4eLayerHost { o_gpr: 32, tetra_down: true, qkv: pers::P4eQkvHost { d_in: 2560, out_v: 9, ..Default::default() }, ..Default::default() }.bytes();
    assert_eq!(lay.len(), pers::P4E_LAYER_BYTES);
    assert_eq!(u32::from_le_bytes(lay[248..252].try_into().unwrap()), 32);
    assert_eq!(u32::from_le_bytes(lay[252..256].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(lay[40 + 120..40 + 124].try_into().unwrap()), 2560);
    assert_eq!(u64::from_le_bytes(lay[40 + 144..40 + 152].try_into().unwrap()), 9);
    let lay = pers::P4eLayerHost { ot: 0xa7, ..Default::default() }.bytes();
    assert_eq!(u64::from_le_bytes(lay[pers::P4E_LAYER_OT..pers::P4E_LAYER_OT + 8].try_into().unwrap()), 0xa7);
    let ot = pers::P4eOTHost { t: pers::SegTetraHost { d_out: 4096, ..Default::default() }, small: 3, k: 19, inv: 0.5, ..Default::default() }.bytes();
    assert_eq!(ot.len(), pers::P4E_OT_BYTES);
    assert_eq!(u32::from_le_bytes(ot[44..48].try_into().unwrap()), 4096);
    assert_eq!(u64::from_le_bytes(ot[56..64].try_into().unwrap()), 3);
    assert_eq!(u32::from_le_bytes(ot[68..72].try_into().unwrap()), 19);
    assert_eq!(f32::from_le_bytes(ot[72..76].try_into().unwrap()), 0.5);
    let mut ptrs = [0u64; 22];
    ptrs[21] = 0x7e;
    let args = pers::P4eArgsHost { ptrs, cap: 8448, c: 0.25, head: [0x11, 0x22, 0x33], ..Default::default() }.bytes();
    assert_eq!(u64::from_le_bytes(args[216..224].try_into().unwrap()), 0x11);
    assert_eq!(u64::from_le_bytes(args[232..240].try_into().unwrap()), 0x33);
    assert_eq!(args.len(), pers::P4E_ARGS_BYTES);
    assert_eq!(u64::from_le_bytes(args[168..176].try_into().unwrap()), 0x7e);
    assert_eq!(u32::from_le_bytes(args[204..208].try_into().unwrap()), 8448);
    assert_eq!(f32::from_le_bytes(args[212..216].try_into().unwrap()), 0.25);
    assert_eq!(std::mem::size_of::<pers::SegLaunch>(), pers::SEG_LAUNCH_BYTES);
    let seg = pers::SegArgsHost { act: 5, gpr: 76, eps: 1e-6, down: pers::SegTetraHost { d_out: 2560, ..Default::default() }, ..Default::default() }.bytes();
    assert_eq!(seg.len(), pers::SEG_ARGS_BYTES);
    assert_eq!(u64::from_le_bytes(seg[32..40].try_into().unwrap()), 5);
    assert_eq!(u32::from_le_bytes(seg[136 + 44..136 + 48].try_into().unwrap()), 2560);
    assert_eq!(f32::from_le_bytes(seg[328..332].try_into().unwrap()), 1e-6);
    assert_eq!(u32::from_le_bytes(seg[332..336].try_into().unwrap()), 76);
}

/// The L1 one-launch pass walks its work list onto the right rows: a fused
/// group (two parts, two centroid pairs) and a single matrix, on grids
/// smaller and larger than the 7 work groups, equal to `tv_tetra48_seg` on
/// each site, bit for bit.
#[test]
fn persall_is_the_fused_kernel_site_by_site() {
    // L1's kernel (variant 0) and its four variants (pers::PERSALL_VARIANTS)
    for variant in 0..pers::PERSALL_VARIANTS.len() as u32 {
        for grid in [1u32, 3, 7, 12] {
            let cmd = Cmd::new(b'A').u32(grid).u64(0xa11).u32(variant).0;
            assert!(run(served(), &cmd) == run(persistent(), &cmd), "variant {variant}, grid {grid}");
        }
    }
    let cmd = Cmd::new(b'A').u32(3).u64(0xa11).u32(0).0;
    let want = run(served(), &cmd);
    let b = mutant("persall_scan", "llvq-cuda/kernels/tetra48_persall.cu", "uniform too.\n        while (si + 1u < nsites && w >= (u32)(sites[(si + 1u) * LLVQ_TOCC_SITE_WORDS + 7u] & 0xffffffffull)) ++si;", "uniform too.\n        while (si + 1u < nsites && w > (u32)(sites[(si + 1u) * LLVQ_TOCC_SITE_WORDS + 7u] & 0xffffffffull)) ++si;");
    assert!(run_mutant(&b, &cmd) != Some(want), "an off-by-one site scan survived");
    // the variants' own loop: the prefetched words handed over, the tables
    // read from where they were staged
    let cmd = Cmd::new(b'A').u32(3).u64(0xa11).u32(3).0;
    let want = run(served(), &cmd);
    let f = "llvq-cuda/kernels/tetra48_persall.cu";
    for (tag, from, to) in [
        ("persall_pf_stale", "                    lo = lo2;\n                    hi16 = hi2;", "                    lo = lo2;\n                    (void)hi2;"),
        ("persall_ts_unstaged", "for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) s_br[i] = branches[i];", "for (u32 i = threadIdx.x; i < 1024u; i += blockDim.x) s_br[i] = branches[i ^ 1u];"),
        ("persall_pf_tile_restart", "        for (u32 t = 0; t < ntiles; ++t) {\n            const u32 jlo = t * TILE_BLOCKS;\n            const u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;\n            const u32 n = (jhi - jlo) * LLVQ_DIM;\n            __syncthreads();\n            for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];\n            __syncthreads();\n            for (; j < jhi;", "        for (u32 t = 0; t < ntiles; ++t) {\n            const u32 jlo = t * TILE_BLOCKS;\n            const u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;\n            const u32 n = (jhi - jlo) * LLVQ_DIM;\n            __syncthreads();\n            for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];\n            __syncthreads();\n            for (j = jlo + lane + (t > 0u ? 32u : 0u); j < jhi;"),
    ] {
        let b = mutant(tag, f, from, to);
        assert!(run_mutant(&b, &cmd) != Some(want.clone()), "persall mutant {tag} survived");
    }
}
