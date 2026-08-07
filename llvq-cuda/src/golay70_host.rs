// Host-side Golay70: the class analysis, the transcoder from raw
// `(index, gain)` codes, the GPU tables, and the bit-exact CPU decoder the
// GPU kernel is checked against.
//
// ⚠️ This file is `include!`d — by `bin/planesbench.rs` and by
// `tests/golay70_decoder_matches_rust.rs` — rather than being a module of
// `lib.rs`, so that adding the Golay70 arm touches no existing library file.
// It therefore declares no `use` items: every path is fully qualified, so it
// can land in any module without colliding with the host's imports.
//
// ## The single point to re-plug
//
// [`golay70_transcode`] is a stopgap: the artifact-side transcoder is being
// written in parallel against the same frozen spec. When it exists, callers
// replace this function and delete this file; until then it is the reference
// the bench and the host test run on.
//
// Main-stream record, from bit 0, LSB-first, 9 bytes per block at offset 9·b:
//
//     [class : 9][gain : 1][golay : 12][A : 24][B : 24][0 : 2]
//
// `golay` is the codeword's rank in the canonical `Golay::codewords()` order.
// EVEN class: the codeword is `{i : |x_i| ≡ 2 (mod 4)}` (a Golay codeword by
// the Λ₂₄ construction, asserted lethally per block); A picks within the
// slot's residue pair (A = 0 canonical on single-value residues); B is the
// sign mask, exact Slot32 semantics (zero slots written 0). ODD class: the
// codeword is `{i : x_i ≡ 3 (mod 4)}`; A/B are the two level bit-planes
// (`level = A | B<<1`, canonical level order); signs are computed, not
// stored (`x_i > 0 ⇔ c_i == [value ≡ 3 (mod 4)]`), re-derived and asserted
// per slot during transcoding.
//
// Exceptions — even classes with > 2 distinct |values| in a residue mod 4
// (zero counted in r0) and ALL 5-level classes of either coset; odd classes
// at L <= 4 are never exceptions — reuse the Planes12x machinery verbatim:
// a u32 block index plus the exact 14-byte Planes14 record. The main stream
// carries the ORIGIN record (class 0, gain copied) for every exception, so
// its contribution is zero and the correction pass adds the exact term.

/// Bytes per Golay70 main-stream record — 72 bits, 70 useful, 3.0 b/weight.
pub const GOLAY70_STRIDE: usize = 9;
/// Bytes one exception costs: a `u32` block index plus a 14-byte Planes14
/// record of the exact block — 144 bits.
pub const GOLAY70_EXC_BYTES: usize = 18;
/// `u32` words per entry of the GPU class table (`GolayClassRec`).
pub const GOLAY70_REC_WORDS: usize = 8;
/// Entries of the GPU class table — 512 so the 9-bit class field cannot
/// index out of bounds even from a corrupt stream.
pub const GOLAY70_TABLE_ENTRIES: usize = 512;

/// What the Golay70 layout knows about one class id (0 = the origin).
///
/// Everything here is a function of the class alone — the E2 exception rule
/// included — so the per-block transcode is table lookups plus one Golay
/// rank query.
pub struct Golay70Class {
    pub is_odd: bool,
    /// The E2 exception rule of `classhist` section 5: 5 levels (either
    /// coset), or an even class with > 2 distinct |values| in some residue
    /// mod 4. Exception classes keep zeroed fields below — the main stream
    /// never names them.
    pub is_exc: bool,
    /// Levels of the class (1 for the origin).
    pub len: usize,
    pub shell: u32,
    /// The unified 4-value select table, integer magnitudes.
    /// Odd: canonical level values (index = level, slots >= len zero).
    /// Even: `[r0[0], r0[1], r2[0], r2[1]]` — the residue pairs in canonical
    /// level order, single-value residues duplicated.
    pub vals: [i32; 4],
    /// Odd only: bit k = 1 iff `vals[k] ≡ 3 (mod 4)`. Even: 0.
    pub flags: u32,
    /// Even only: distinct |values| per residue (index 0 = ≡ 0 mod 4, zero
    /// included; index 1 = ≡ 2 mod 4) — what the canonical-A assert needs.
    pub res_len: [usize; 2],
}

/// One entry per class id: 0 is the origin, `1 + ci` is class `ci`.
pub fn golay70_classes(fd: &llvq_search::fastdec::FastDecoder) -> Vec<Golay70Class> {
    let mut out = Vec::with_capacity(fd.n_classes() + 1);
    out.push(Golay70Class {
        is_odd: false,
        is_exc: false,
        len: 1,
        shell: 0,
        vals: [0; 4],
        flags: 0,
        res_len: [0, 0],
    });
    for ci in 0..fd.n_classes() {
        let lv = fd.levels(ci);
        // Distinct |values| per residue mod 4, zero counted in r0 — the same
        // profile `classhist` section 5 decided the E2 bracket from.
        let mut r = [0u32; 4];
        for k in 0..lv.len {
            r[(lv.values[k] % 4) as usize] += 1;
        }
        let is_exc = lv.len == 5 || (!lv.odd && r.iter().any(|&n| n > 2));
        let mut c = Golay70Class {
            is_odd: lv.odd,
            is_exc,
            len: lv.len,
            shell: lv.shell,
            vals: [0; 4],
            flags: 0,
            res_len: [0, 0],
        };
        if !is_exc {
            if lv.odd {
                for k in 0..lv.len {
                    c.vals[k] = lv.values[k];
                    if lv.values[k] % 4 == 3 {
                        c.flags |= 1 << k;
                    }
                }
            } else {
                let mut pair = [[0i32; 2]; 2];
                let mut n = [0usize; 2];
                for k in 0..lv.len {
                    debug_assert_eq!(lv.values[k] % 2, 0, "even class holds even values");
                    let ri = usize::from(lv.values[k] % 4 == 2);
                    pair[ri][n[ri]] = lv.values[k];
                    n[ri] += 1;
                }
                // Single-value residues duplicate — the branchless pick of
                // the kernel; the transcoder writes A = 0 canonically.
                for (p, &nn) in pair.iter_mut().zip(&n) {
                    if nn == 1 {
                        p[1] = p[0];
                    }
                }
                c.vals = [pair[0][0], pair[0][1], pair[1][0], pair[1][1]];
                c.res_len = n;
            }
        }
        out.push(c);
    }
    out
}

/// The 512-entry GPU class table, 8 `u32` words per entry, matching
/// `GolayClassRec` of `llvq_golay.cuh` field for field. Values divided by
/// sqrt(16·shell), like every other arm's table. Exception classes and the
/// origin stay all-zero — the main stream never names an exception class,
/// and the origin decodes to zero through a zero entry.
pub fn golay70_gpu_table(cls: &[Golay70Class]) -> Vec<u32> {
    assert!(cls.len() <= GOLAY70_TABLE_ENTRIES);
    let mut tab = vec![0u32; GOLAY70_TABLE_ENTRIES * GOLAY70_REC_WORDS];
    for (id, c) in cls.iter().enumerate().skip(1) {
        if c.is_exc {
            continue;
        }
        let base = id * GOLAY70_REC_WORDS;
        let norm = f64::from(16 * c.shell).sqrt();
        for k in 0..4 {
            tab[base + k] = ((f64::from(c.vals[k]) / norm) as f32).to_bits();
        }
        tab[base + 4] = c.flags;
        tab[base + 5] = u32::from(c.is_odd);
    }
    tab
}

/// The canonical 4096-codeword table the kernel resolves the 12-bit rank
/// through — `Golay::codewords()` order, the order frozen by format v1.
pub fn golay70_gpu_codewords(golay: &llvq_core::Golay) -> Vec<u32> {
    let cw = golay.codewords();
    assert_eq!(cw.len(), 4096, "the canonical Golay table has 4096 codewords");
    cw.to_vec()
}

/// A transcoded Golay70 stream: 9-byte main records plus the exception
/// table — the same three arrays Planes12x uploads.
pub struct Golay70Blocks {
    pub n_blocks: usize,
    /// The bit-packed main stream, LSB-first within each byte,
    /// `9·n_blocks` bytes exactly.
    pub data: Vec<u8>,
    /// Matrix-wide block index of each exception, strictly ascending.
    pub exc_idx: Vec<u32>,
    /// One 14-byte Planes14 record per exception — the **exact** block.
    pub exc_data: Vec<u8>,
}

/// The exact 14-byte Planes14 record of one block — the exception table's
/// entry format, encoded from the frozen offsets (9/1/24/24/24/24, 6 zeros).
fn golay70_planes14_record(
    fd: &llvq_search::fastdec::FastDecoder,
    table: &llvq_artifact::runtime::ClassTable,
    idx: u64,
    gain: u32,
) -> u128 {
    let ci = fd.class_of(idx).expect("exceptions are never the origin");
    let id = ci + 1;
    let rec = table.record(id);
    let p = fd.decode(idx).expect("class_of validated the index");
    let mut smask = 0u128;
    let mut planes = [0u128; 3];
    for (i, &v) in p.iter().enumerate() {
        let lev = rec.values[..rec.len as usize]
            .iter()
            .position(|&u| u == v.abs())
            .expect("decoded value belongs to its class");
        if v < 0 {
            smask |= 1 << i;
        }
        for (pl, plane) in planes.iter_mut().enumerate() {
            if lev >> pl & 1 == 1 {
                *plane |= 1 << i;
            }
        }
    }
    u128::from(id as u32)
        | u128::from(gain) << 9
        | smask << 10
        | planes[0] << 34
        | planes[1] << 58
        | planes[2] << 82
}

/// One transcoding chunk's output — main-stream bytes, exception indices
/// (matrix-wide, ascending) and exception records.
type Golay70Chunk = (Vec<u8>, Vec<u32>, Vec<u8>);

fn golay70_chunk(
    fd: &llvq_search::fastdec::FastDecoder,
    golay: &llvq_core::Golay,
    cls: &[Golay70Class],
    table: &llvq_artifact::runtime::ClassTable,
    indices: &[u64],
    gains: &[u32],
    b0: usize,
) -> Golay70Chunk {
    let mut data = vec![0u8; indices.len() * GOLAY70_STRIDE];
    let mut exc_idx: Vec<u32> = Vec::new();
    let mut exc_data: Vec<u8> = Vec::new();
    for (l, (&idx, &gain)) in indices.iter().zip(gains).enumerate() {
        assert!(gain < 2, "block {}: gain {gain} overflows the 1-bit field", b0 + l);
        let id = match fd.class_of(idx) {
            Some(ci) => ci + 1,
            None => {
                assert_eq!(idx, 0, "block {}: index {idx} outside the ball", b0 + l);
                0
            }
        };
        let rec: u128 = if id == 0 {
            // The origin: header only, everything else canonical zero.
            u128::from(gain) << 9
        } else if cls[id].is_exc {
            // Exception: the main stream carries the ORIGIN record (gain
            // copied) — zero contribution — and the exact block goes to the
            // exception table as a Planes14 record.
            exc_idx.push((b0 + l) as u32);
            let r14 = golay70_planes14_record(fd, table, idx, gain);
            exc_data.extend_from_slice(&r14.to_le_bytes()[..14]);
            u128::from(gain) << 9
        } else {
            let p = fd.decode(idx).expect("class_of validated the index");
            let c = &cls[id];
            let mut cw = 0u32;
            let mut a_plane = 0u32;
            let mut b_plane = 0u32;
            if c.is_odd {
                for (i, &v) in p.iter().enumerate() {
                    if v.rem_euclid(4) == 3 {
                        cw |= 1 << i;
                    }
                }
                for (i, &v) in p.iter().enumerate() {
                    let lev = c.vals[..c.len]
                        .iter()
                        .position(|&u| u == v.abs())
                        .expect("odd magnitude is one of the class's levels");
                    a_plane |= (lev as u32 & 1) << i;
                    b_plane |= (lev as u32 >> 1) << i;
                    // The forced-sign rule, re-derived and asserted: a block
                    // whose signs do not follow it is not a valid odd-coset
                    // point and must die here, not decode wrong on the card.
                    let neg = (cw >> i & 1) ^ (c.flags >> lev & 1);
                    assert_eq!(
                        neg == 1,
                        v < 0,
                        "block {}: odd sign disagrees with the mod-4 rule",
                        b0 + l
                    );
                }
            } else {
                for (i, &v) in p.iter().enumerate() {
                    if v.abs() % 4 == 2 {
                        cw |= 1 << i;
                    }
                }
                for (i, &v) in p.iter().enumerate() {
                    let ri = (cw >> i & 1) as usize;
                    assert!(
                        c.res_len[ri] >= 1,
                        "block {}: slot {i} holds a value outside both residues",
                        b0 + l
                    );
                    let a = if v.abs() == c.vals[ri * 2] {
                        0u32 // single-value residues land here: A = 0 canonical
                    } else {
                        assert_eq!(
                            v.abs(),
                            c.vals[ri * 2 + 1],
                            "block {}: slot {i} value is not in its residue pair",
                            b0 + l
                        );
                        1u32
                    };
                    a_plane |= a << i;
                    if v < 0 {
                        b_plane |= 1 << i; // zero slots: v == 0, bit written 0
                    }
                }
            }
            let g = golay
                .rank(cw)
                .expect("the residue-mod-4 plane is a Golay codeword") as u128;
            u128::from(id as u32)
                | u128::from(gain) << 9
                | g << 10
                | u128::from(a_plane) << 22
                | u128::from(b_plane) << 46
        };
        debug_assert_eq!(rec >> 70, 0, "record exceeds 70 bits");
        data[l * GOLAY70_STRIDE..(l + 1) * GOLAY70_STRIDE]
            .copy_from_slice(&rec.to_le_bytes()[..GOLAY70_STRIDE]);
    }
    (data, exc_idx, exc_data)
}

/// Transcode raw `(index, gain)` codes into a [`Golay70Blocks`] stream —
/// one fast decode per block plus one Golay rank query, threaded by chunks.
/// Every lethal assert of the per-block path (codeword membership, the
/// forced-sign rule, residue pairs) fires here, at the source.
pub fn golay70_transcode(
    fd: &llvq_search::fastdec::FastDecoder,
    golay: &llvq_core::Golay,
    cls: &[Golay70Class],
    table: &llvq_artifact::runtime::ClassTable,
    indices: &[u64],
    gains: &[u32],
) -> Golay70Blocks {
    assert_eq!(indices.len(), gains.len(), "one gain per block");
    assert_eq!(
        table.gain_bits(),
        1,
        "Golay70 freezes the gain field at bit 9, one bit wide"
    );
    let n = indices.len();
    assert!(u32::try_from(n).is_ok(), "{n} blocks: exception indices must fit u32");
    let nthreads = std::thread::available_parallelism().map_or(8, |t| t.get());
    let chunk = n.div_ceil(nthreads).max(1);
    let outs: Vec<Golay70Chunk> = std::thread::scope(|sc| {
        let handles: Vec<_> = indices
            .chunks(chunk)
            .zip(gains.chunks(chunk))
            .enumerate()
            .map(|(t, (ic, gc))| {
                sc.spawn(move || golay70_chunk(fd, golay, cls, table, ic, gc, t * chunk))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    let mut out = Golay70Blocks {
        n_blocks: n,
        data: Vec::with_capacity(n * GOLAY70_STRIDE),
        exc_idx: Vec::new(),
        exc_data: Vec::new(),
    };
    for (d, ei, ed) in outs {
        out.data.extend_from_slice(&d);
        out.exc_idx.extend_from_slice(&ei);
        out.exc_data.extend_from_slice(&ed);
    }
    debug_assert!(out.exc_idx.windows(2).all(|w| w[0] < w[1]));
    out
}

/// Decode one 14-byte Planes14 record from the exception table — lethal on
/// padding bits, on out-of-range levels, and on a non-5-level... no: the E2
/// exception set includes L <= 4 even violators, so unlike Planes12x the
/// class's level count is NOT asserted to be 5 — the exception rule is
/// re-checked by the caller against [`Golay70Class::is_exc`] instead.
fn golay70_exc_decode(
    exc_data: &[u8],
    table: &llvq_artifact::runtime::ClassTable,
    e: usize,
    b: usize,
) -> (llvq_core::Point, u32, usize) {
    let mut buf = [0u8; 16];
    buf[..14].copy_from_slice(&exc_data[e * 14..(e + 1) * 14]);
    let r = u128::from_le_bytes(buf);
    assert_eq!(r >> 106, 0, "block {b}: exception padding bits 106..112 must be zero");
    let id = (r & 0x1ff) as usize;
    let gain = ((r >> 9) & 1) as u32;
    assert_ne!(id, 0, "block {b}: the origin is never an exception");
    let rec = table.record(id);
    let smask = ((r >> 10) & 0xff_ffff) as u32;
    let p0 = ((r >> 34) & 0xff_ffff) as u32;
    let p1 = ((r >> 58) & 0xff_ffff) as u32;
    let p2 = ((r >> 82) & 0xff_ffff) as u32;
    let mut p = [0i32; llvq_core::DIM];
    for (j, pj) in p.iter_mut().enumerate() {
        let lev = (p0 >> j & 1) | (p1 >> j & 1) << 1 | (p2 >> j & 1) << 2;
        assert!(
            (lev as usize) < rec.len as usize,
            "block {b} slot {j}: level {lev} outside class {id} (len {})",
            rec.len
        );
        let v = rec.values[lev as usize];
        *pj = if smask >> j & 1 == 1 { -v } else { v };
    }
    (p, gain, id)
}

/// Decode block `b` back to its lattice point and gain rank — the CPU
/// reference the GPU kernel is checked against, same signature family as the
/// other layouts' `decode_block`. Exceptions resolve from their Planes14
/// record, so the result equals `RuntimeBlocks::decode_block` (Slot32) on
/// the same input, bit for bit — the central claim of the layout.
///
/// Lethal on purpose: nonzero padding bits, a non-origin main record under
/// an exception, an exception class named by the main stream, an
/// out-of-range level, a nonzero refinement bit on a single-value residue,
/// or a sign bit on a zero slot all panic rather than decoding something
/// plausible.
pub fn golay70_decode_block(
    blocks: &Golay70Blocks,
    cls: &[Golay70Class],
    golay: &llvq_core::Golay,
    table: &llvq_artifact::runtime::ClassTable,
    b: usize,
) -> (llvq_core::Point, u32) {
    assert!(b < blocks.n_blocks, "block {b} of {}", blocks.n_blocks);
    let mut buf = [0u8; 16];
    buf[..GOLAY70_STRIDE]
        .copy_from_slice(&blocks.data[b * GOLAY70_STRIDE..(b + 1) * GOLAY70_STRIDE]);
    let r = u128::from_le_bytes(buf);
    assert_eq!(r >> 70, 0, "block {b}: bits 70..72 must be zero");
    let id = (r & 0x1ff) as usize;
    let gain = ((r >> 9) & 1) as u32;
    let grank = ((r >> 10) & 0xfff) as usize;
    let a_plane = ((r >> 22) & 0xff_ffff) as u32;
    let b_plane = ((r >> 46) & 0xff_ffff) as u32;

    if let Ok(e) = blocks.exc_idx.binary_search(&(b as u32)) {
        // The invariant the kernel's zero-contribution shortcut rests on:
        // the main stream holds the canonical origin record, gain copied.
        assert_eq!(id, 0, "block {b}: exception's main record is not the origin");
        assert_eq!(
            (grank, a_plane, b_plane),
            (0, 0, 0),
            "block {b}: exception's main record carries payload bits"
        );
        let (p, egain, eid) = golay70_exc_decode(&blocks.exc_data, table, e, b);
        assert!(
            cls[eid].is_exc,
            "block {b}: exception record's class {eid} does not violate the E2 rule"
        );
        assert_eq!(
            egain, gain,
            "block {b}: exception gain differs from the main stream's copy"
        );
        return (p, egain);
    }

    if id == 0 {
        assert_eq!(
            (grank, a_plane, b_plane),
            (0, 0, 0),
            "block {b}: origin record carries payload bits"
        );
        return ([0; llvq_core::DIM], gain);
    }
    let c = &cls[id];
    assert!(!c.is_exc, "block {b}: exception class {id} in the main stream");
    let cw = golay.codewords()[grank];
    let mut p = [0i32; llvq_core::DIM];
    for (i, pi) in p.iter_mut().enumerate() {
        let cbit = cw >> i & 1;
        let abit = a_plane >> i & 1;
        let bbit = b_plane >> i & 1;
        if c.is_odd {
            let lev = (abit | bbit << 1) as usize;
            assert!(
                lev < c.len,
                "block {b} slot {i}: level {lev} outside class {id} (len {})",
                c.len
            );
            let v = c.vals[lev];
            let neg = cbit ^ (c.flags >> lev & 1);
            *pi = if neg == 1 { -v } else { v };
        } else {
            let ri = cbit as usize;
            assert!(
                c.res_len[ri] >= 1,
                "block {b} slot {i}: empty residue {ri} selected"
            );
            if c.res_len[ri] == 1 {
                assert_eq!(
                    abit, 0,
                    "block {b} slot {i}: refinement bit must be 0 on a single-value residue"
                );
            }
            let v = c.vals[ri * 2 + abit as usize];
            if v == 0 {
                assert_eq!(
                    bbit, 0,
                    "block {b} slot {i}: zero slot must carry a written-0 sign"
                );
            }
            *pi = if bbit == 1 { -v } else { v };
        }
    }
    (p, gain)
}
