//! The three gates of S0 on the base pass: identity across R, slots, grids and
//! parts; closeness to 4E; the mutants. One file because every later arm reuses
//! these gate functions and the FIX fixture.

use super::*;

// ---- gate 1: identity ---------------------------------------------------------

/// The full tree in node order, then every pass of `sizes`: an
/// ancestor-closed subset of that many nodes in a random slot order, on its
/// own scheduler seed; then the full tree permuted. Every row of every pass
/// equals the same node of the first.
fn gate_subsets(bin: &Path, f: &Fix, t: &Tree, sizes: &[usize], pos0: u32) -> Result<(), String> {
    let n = t.tok.len();
    let full: Vec<usize> = (0..n).collect();
    let mut passes = vec![Pass::of(t, &full, pos0, 0x7e01)];
    let mut nodes = vec![full.clone()];
    for (i, &s) in sizes.iter().enumerate() {
        let sub = t.subset(s, 0x5b00 + i as u64);
        passes.push(Pass::of(t, &sub, pos0, 0x7e10 + i as u64));
        nodes.push(sub);
    }
    let perm = t.subset(n, 0x5bff);
    passes.push(Pass::of(t, &perm, pos0, 0x7eff));
    nodes.push(perm);
    let out = exec(bin, f, None, &passes)?;
    let base = &out.passes[0];
    for (pi, (rows, ns)) in out.passes.iter().zip(&nodes).enumerate().skip(1) {
        for (s, (row, &node)) in rows.iter().zip(ns).enumerate() {
            if let Some(what) = diff(row, &base[node]) {
                return Err(format!("pass {pi} ({} rows), slot {s}: node {node}'s {what} differs from the full tree's", ns.len()));
            }
        }
    }
    Ok(())
}

/// The full tree on two fixtures that differ only in grid shape: the same
/// bits.
fn gate_grids(bin: &Path, a: &Fix, b: &Fix, t: &Tree, pos0: u32) -> Result<(), String> {
    let full: Vec<usize> = (0..t.tok.len()).collect();
    let pa = [Pass::of(t, &full, pos0, 0x7e21)];
    let pb = [Pass::of(t, &t.subset(t.tok.len(), 0x5b21), pos0, 0x7e22)];
    let oa = exec(bin, a, None, &pa)?;
    let ob = exec(bin, b, None, &pb)?;
    for (s, &node) in t.subset(t.tok.len(), 0x5b21).iter().enumerate() {
        if let Some(what) = diff(&ob.passes[0][s], &oa.passes[0][node]) {
            return Err(format!("grid {}x{} against {}x{}: node {node}'s {what}", b.grid, b.threads, a.grid, a.threads));
        }
    }
    Ok(())
}

/// For each node of `check`: the tree, then an R = 1 pass whose root is the
/// node's token at its position and whose first phase moves the node's
/// ancestors from the tree's slots into the cache. The root's row equals the
/// node's row in the tree.
fn gate_chain(bin: &Path, f: &Fix, t: &Tree, check: &[usize], pos0: u32) -> Result<(), String> {
    let order = t.subset(t.tok.len(), 0x5c00);
    let mut passes = Vec::new();
    for (i, &node) in check.iter().enumerate() {
        passes.push(Pass::of(t, &order, pos0, 0x7f00 + i as u64));
        let slot = |n: usize| order.iter().position(|&m| m == n).unwrap() as u32;
        let path = t.path(node);
        let mv: Vec<u32> = path[..path.len() - 1].iter().map(|&a| slot(a)).collect();
        passes.push(Pass::root(t.tok[node], pos0 + t.depth[node] as u32, 0x7f80 + i as u64, pos0, mv));
    }
    let out = exec(bin, f, None, &passes)?;
    for (i, &node) in check.iter().enumerate() {
        let s = order.iter().position(|&m| m == node).unwrap();
        if let Some(what) = diff(&out.passes[2 * i + 1][0], &out.passes[2 * i][s]) {
            return Err(format!("node {node} at depth {}: decoded as a root, its {what} differs from its tree row", t.depth[node]));
        }
    }
    Ok(())
}

// ---- gate 2: closeness to 4E -----------------------------------------------------

fn h(v: u16) -> f64 {
    llvq_cuda::f16_to_f64(v)
}

/// sqrt(sum (a - b)^2 / sum b^2) over the vocabulary.
fn rel_rms(a: &[u16], b: &[u16]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        num += (h(x) - h(y)).powi(2);
        den += h(y).powi(2);
    }
    (num / den).sqrt()
}

/// One 4E token at pos0, then the R = 1 pass on the same token and position:
/// the relative RMS of the pass's logits against 4E's, and the two argmaxes.
pub(super) fn gate_close(bin: &Path, f: &Fix, tok: u32, pos0: u32) -> Result<(f64, u32, u32), String> {
    let out = exec(bin, f, Some((pos0, tok)), &[Pass::root(tok, pos0, 0x7a01, 0, vec![])])?;
    let (t4e, l4e) = out.reference.unwrap();
    let row = &out.passes[0][0];
    let e = rel_rms(&row.logits, &l4e);
    if e.is_nan() || e >= 1e-2 {
        return Err(format!("form {}, variant {}: relative RMS {e:.3e} against 4E's logits", f.form, f.variant));
    }
    Ok((e, t4e, row.tok))
}

// ---- the fixtures -------------------------------------------------------------------

pub(super) const FIX: Fix = Fix { form: 1, grid: 3, threads: 64, cap: 48, variant: 0, seed: 0x7e5ed, split: SPLIT };

/// Gate 1 at every R from 1 to 32 in one tree of 32 nodes, depth up to 7.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn tree_rows_keep_their_bits_at_every_r_and_slot() {
    let t = tree(32, 7, 0x3e1);
    let sizes: Vec<usize> = (1..=31).collect();
    gate_subsets(host(), &FIX, &t, &sizes, 9).unwrap();
}

/// Gate 1 across grid shapes and the 8B's parts (Tetra o, untied head),
/// the other RoPE forms, and a root past the first split (lengths 255 to
/// 262).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn tree_rows_keep_their_bits_across_grids_and_parts() {
    let t = tree(12, 4, 0x3e2);
    let b = Fix { grid: 5, threads: 96, ..FIX };
    gate_grids(host(), &FIX, &b, &t, 9).unwrap();
    for (form, variant) in [(0u32, 16u32), (2, 0)] {
        let f = Fix { form, variant, cap: 288, ..FIX };
        gate_subsets(host(), &f, &t, &[1, 5, 9], 254).unwrap();
    }
}

/// Gate 1, the decode it serves: a row of a tree equals the same token
/// decoded as a root after its ancestors were moved into the cache.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_tree_row_is_the_root_it_becomes() {
    let t = tree(12, 4, 0x3e3);
    let check: Vec<usize> = (1..12).filter(|&i| t.depth[i] >= 2).take(4).chain([1]).collect();
    assert_eq!(check.len(), 5, "four nodes at depth 2 or more, and node 1");
    eprintln!("nodes {check:?} at depths {:?}", check.iter().map(|&i| t.depth[i]).collect::<Vec<_>>());
    gate_chain(host(), &FIX, &t, &check, 9).unwrap();
    let f = Fix { form: 2, variant: 16, cap: 288, ..FIX };
    gate_chain(host(), &f, &t, &check[..2], 254).unwrap();
}

/// Gate 2: R = 1 within 1e-2 relative RMS of 4E's logits.
#[test]
fn one_row_is_close_to_4e() {
    for (form, variant, tok, pos0) in [(1u32, 0u32, 7u32, 9u32), (0, 0, 517, 30), (2, 16, 1023, 9), (1, 16 | 64, 64, 270)] {
        let f = Fix { form, variant, cap: 288, ..FIX };
        let (e, t4e, tt) = gate_close(host(), &f, tok, pos0).unwrap();
        eprintln!("form {form}, variant {variant}, token {tok} at {pos0}: relative RMS {e:.3e}, argmax 4E {t4e}, tree {tt}");
    }
}

/// `pers/verify.rs::P4tLaunch` against `Pers4tLaunch`: its size, every field's
/// offset in order, and the row bound.
#[test]
fn the_tree_launch_mirror_matches_the_kernel() {
    use std::mem::offset_of;
    let out = run(host(), &(b't' as u32).to_le_bytes()).unwrap();
    let got: Vec<u32> = out.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    type L = pers::P4tLaunch;
    let want: Vec<u32> = [
        std::mem::size_of::<L>(),
        offset_of!(L, bar_base),
        offset_of!(L, pos0),
        offset_of!(L, n_rows),
        offset_of!(L, area_floats),
        offset_of!(L, n_move),
        offset_of!(L, move_pos0),
        offset_of!(L, part_stride),
        offset_of!(L, ldx),
        offset_of!(L, split_qkv),
        offset_of!(L, split_o),
        offset_of!(L, split_gu),
        offset_of!(L, split_dn),
        offset_of!(L, split_head),
        offset_of!(L, pad),
        offset_of!(L, toks),
        offset_of!(L, depth),
        offset_of!(L, path),
        offset_of!(L, mv),
        offset_of!(L, ring),
        offset_of!(L, logits),
        offset_of!(L, x_a),
        offset_of!(L, x_b),
        offset_of!(L, xin),
        offset_of!(L, xn),
        offset_of!(L, q_raw),
        offset_of!(L, k_raw),
        offset_of!(L, v_raw),
        offset_of!(L, q_out),
        offset_of!(L, ctx),
        offset_of!(L, g_raw),
        offset_of!(L, u_raw),
        offset_of!(L, tk),
        offset_of!(L, tv),
        offset_of!(L, part),
        offset_of!(L, fpart),
        offset_of!(L, tickets),
        offset_of!(L, ftickets),
        offset_of!(L, slots),
        offset_of!(L, stamps),
    ]
    .into_iter()
    .map(|v| v as u32)
    .chain([pers::P4T_RMAX])
    .collect();
    assert_eq!(got, want, "Pers4tLaunch and pers::P4tLaunch disagree");
    assert_eq!(std::mem::size_of::<L>(), pers::P4T_LAUNCH_BYTES);
}

// ---- gate 3: mutants -----------------------------------------------------------------

/// The gates a mutant must fail, on a smaller tree.
fn gates(bin: &Path) -> Result<(), String> {
    let t = tree(12, 4, 0x3e4);
    gate_close(bin, &Fix { cap: 288, ..FIX }, 7, 9)?;
    gate_subsets(bin, &FIX, &t, &[1, 3, 9], 9)?;
    let check: Vec<usize> = (1..12).filter(|&i| t.depth[i] >= 2).take(2).collect();
    gate_chain(bin, &FIX, &t, &check, 9)?;
    Ok(())
}

/// Each guard has a mutant, killed by a gate (the error it fails names it).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn tree_verify_mutants_are_killed() {
    gates(host()).expect("the unmutated pass passes the mutants' gates");
    let f = "llvq-cuda/kernels/pers_4et.cu";
    let muts: &[(&str, &[Edit])] = &[
        // row order: a row's B column read from its neighbour's slot; the
        // activation rows staged in reverse
        ("row_b_column", &[(f, "const u32* xr = xs + (nt * 8u + gid) * P4T_XW + kw;", "const u32* xr = xs + (nt * 8u + (gid ^ 1u)) * P4T_XW + kw;")]),
        ("row_staged_reversed", &[(f, "g.x + (unsigned long long)(ok ? n : 0u) * g.ldx", "g.x + (unsigned long long)(ok ? R - 1u - n : 0u) * g.ldx")]),
        ("row_rope_slot", &[(f, "hh = w - r * per, pos = L.pos0 + L.depth[r];", "hh = w - r * per, pos = L.pos0 + r;")]),
        // scale association: the other centroid, the neighbour row's scale,
        // an int4 scale and bias swapped
        ("scale_gscale", &[(f, "g.gscale[(hi16 >> 15) & 1u]", "g.gscale[(hi16 >> 14) & 1u]")]),
        ("scale_rscale", &[(f, "t_inv[m]), g.rscale[row]);", "t_inv[m]), g.rscale[row ^ 1u]);")]),
        ("scale_q4_bias", &[(f, "p4t_pack(pers_q4_deq((wv >> (8u * e)) & 0xfu, sb, bb),", "p4t_pack(pers_q4_deq((wv >> (8u * e)) & 0xfu, bb, sb),")]),
        // the tree mask: ancestors read as the slots after the root's; the row's length
        // as the root's; the move one position late
        ("mask_contiguous", &[(f, "(isk ? tkh : tvh) + (unsigned long long)path_r[p - pos0] * slot_stride", "(isk ? tkh : tvh) + (unsigned long long)(path_r[0] + p - pos0) * slot_stride")]),
        ("mask_length", &[(f, "const u32 l = L.pos0 + L.depth[r] + 1u;", "const u32 l = L.pos0 + 1u;")]),
        ("mask_move", &[(f, "l = it - i * NL, pos = L.move_pos0 + i;", "l = it - i * NL, pos = L.move_pos0 + i + 1u;")]),
        // fold order: the folding warp starts from its own slice, so the
        // order follows the arrivals
        (
            "fold_arrival",
            &[
                (f, "float v = __ldcg(g.part + (unsigned long long)n * g.d_out + row);", "float v = __ldcg(g.part + ((unsigned long long)sl * R + n) * g.d_out + row);"),
                (f, "__ldcg(g.part + ((unsigned long long)k * R + n) * g.d_out + row)", "__ldcg(g.part + ((unsigned long long)((sl + k) % g.S) * R + n) * g.d_out + row)"),
            ],
        ),
    ];
    for (tag, edits) in muts {
        let b = mutant(tag, edits);
        let r = gates(&b);
        let _ = std::fs::remove_file(&b);
        match r {
            Err(e) => eprintln!("{tag}: killed by {}", e.lines().next().unwrap_or("")),
            Ok(()) => panic!("tree verify mutant {tag} survived every gate"),
        }
    }
}
