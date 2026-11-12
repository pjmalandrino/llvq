//! The verifier of the decode-speed research phase (MODE RECHERCHE, step R2,
//! `docs/mesures/recherche-debit-r1-cadrage-2026-11-12.txt`): the portable half.
//!
//! The candidates are row programs registered in `kernels/rowcand.cu`, one
//! line of its `ROWC_ARMS` list each. This module reads that list from the
//! kernel text itself, so the card's bench (`bin/rowbench`), the Mac's
//! stage-1 gate (`tests/rowcand_gate.rs`) and the SASS count
//! (`ops/rowcand-sass.sh`) see the same arms and nothing is registered twice.
//! It also holds what both stages judge with: the site words, the error of a
//! row against the served row, and the bandwidth probe's expected sum.
//!
//! Nothing here is served. The module text is the row-loop unit
//! ([`crate::pers::trows_unit_text`]) followed by `bwprobe.cu` and
//! `rowcand.cu`, so every served module keeps its bytes.

/// `rowcand.cu`, embedded so the Mac reads the registry the card compiles.
pub const ROWC_PART: (&str, &str) = ("rowcand.cu", include_str!("../kernels/rowcand.cu"));

/// `bwprobe.cu`, the bare read probe.
pub const BWPROBE_PART: (&str, &str) = ("bwprobe.cu", include_str!("../kernels/bwprobe.cu"));

/// The bench module: the row-loop unit (the K4 and K11 arms of the journals,
/// as controls), the probe, the candidates.
pub fn rowc_unit_text(tile_define: &str) -> String {
    format!("{}\n{}\n{}", crate::pers::trows_unit_text(tile_define), BWPROBE_PART.1, ROWC_PART.1)
}

/// What a registered arm owes the served row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Owes {
    /// y and its f32 value before `f2h`, bit for bit.
    Bits,
    /// `|f32 - served f32| / rms(served f32 of the site) <= bound` on every row.
    Bound(f64),
    /// Nothing: a diagnostic, whose rate is never a candidate's.
    Nothing,
}

/// One line of `ROWC_ARMS`.
#[derive(Clone, Debug, PartialEq)]
pub struct RowArm {
    pub name: String,
    pub strukt: String,
    pub owes: Owes,
    /// Reads its own copy of the words, rewritten at load in the served stride.
    pub relay: bool,
    /// Blocks a lane folds in one iteration of the inner loop.
    pub bpi: u32,
}

impl RowArm {
    /// The 4E row geometry over a list of sites.
    pub fn symbol(&self) -> String {
        format!("tv_rowc_{}", self.name)
    }
    /// 4E's gate/up phase alone, pairs of sites.
    pub fn gu_symbol(&self) -> String {
        format!("tv_rowc_gu_{}", self.name)
    }
    /// The load-time rewrite of a RELAY arm.
    pub fn relay_symbol(&self) -> String {
        format!("tv_rowc_relay_{}", self.name)
    }
    /// The served arm, the reference of every other.
    pub fn is_served(&self) -> bool {
        self.name == "served"
    }
}

/// The registry, parsed from `text` (a module or `rowcand.cu`): the lines
/// `X(name, Struct, OWES, bound, RELAY, BPI)` inside `#define ROWC_ARMS(X)`.
/// Refuses a malformed line, a duplicate name, a bound on a non-BOUND arm, and
/// a registry without `served` first.
pub fn parse_arms(text: &str) -> Result<Vec<RowArm>, String> {
    let start = text.find("#define ROWC_ARMS(X)").ok_or("no `#define ROWC_ARMS(X)` in the text")?;
    let mut arms: Vec<RowArm> = Vec::new();
    let mut lines = text[start..].lines();
    lines.next();
    for line in lines {
        let t = line.trim().trim_end_matches('\\').trim();
        if t.is_empty() {
            break;
        }
        let inner = t
            .strip_prefix("X(")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| format!("ROWC_ARMS: {t:?} is not X(name, Struct, OWES, bound, RELAY, BPI)"))?;
        let f: Vec<&str> = inner.split(',').map(str::trim).collect();
        if f.len() != 6 {
            return Err(format!("ROWC_ARMS: {t:?} has {} fields, expected 6", f.len()));
        }
        let name = f[0].to_string();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("ROWC_ARMS: {name:?} is not an identifier"));
        }
        let bound: f64 = f[3].parse().map_err(|e| format!("ROWC_ARMS {name}: bound {:?}: {e}", f[3]))?;
        let owes = match f[2] {
            "ROWC_BITS" => Owes::Bits,
            "ROWC_BOUND" => Owes::Bound(bound),
            "ROWC_NONE" => Owes::Nothing,
            o => return Err(format!("ROWC_ARMS {name}: OWES {o:?}, expected ROWC_BITS, ROWC_BOUND or ROWC_NONE")),
        };
        match owes {
            Owes::Bound(b) if !(b > 0.0 && b < 1.0) => return Err(format!("ROWC_ARMS {name}: a bound must lie in (0, 1), got {b}")),
            Owes::Bits | Owes::Nothing if bound != 0.0 => return Err(format!("ROWC_ARMS {name}: a bound on an arm that is not ROWC_BOUND")),
            _ => {}
        }
        let relay = match f[4] {
            "0" => false,
            "1" => true,
            r => return Err(format!("ROWC_ARMS {name}: RELAY {r:?}, expected 0 or 1")),
        };
        let bpi: u32 = f[5].parse().map_err(|e| format!("ROWC_ARMS {name}: BPI {:?}: {e}", f[5]))?;
        if bpi == 0 {
            return Err(format!("ROWC_ARMS {name}: BPI 0"));
        }
        if arms.iter().any(|a| a.name == name) {
            return Err(format!("ROWC_ARMS: {name} registered twice"));
        }
        arms.push(RowArm { name, strukt: f[1].to_string(), owes, relay, bpi });
    }
    match arms.first() {
        Some(a) if a.is_served() && a.owes == Owes::Bits && !a.relay => Ok(arms),
        _ => Err("ROWC_ARMS: the first arm must be `served`, ROWC_BITS, RELAY 0: every other arm is judged against it".into()),
    }
}

/// The registry of the embedded `rowcand.cu`.
pub fn arms() -> Vec<RowArm> {
    parse_arms(ROWC_PART.1).expect("kernels/rowcand.cu: ROWC_ARMS parses (tested)")
}

/// One `tv_rowc_*` site (pers_trows.cu's `TROWS_SITE_WORDS` layout). `yf`:
/// the f32 value of each row before `f2h`, or 0.
#[allow(clippy::too_many_arguments)]
pub fn rowc_site_words(
    words: u64,
    gscale: u64,
    rscale: u64,
    tail_f16: u64,
    y_f16: u64,
    yf: u64,
    d_out: u32,
    nblocks: u32,
    tail_w: u32,
    stride_u32: u32,
) -> [u64; 10] {
    [
        words,
        gscale,
        rscale,
        tail_f16,
        y_f16,
        u64::from(d_out) | (u64::from(nblocks) << 32),
        u64::from(tail_w) | (u64::from(stride_u32) << 32),
        yf,
        0,
        0,
    ]
}

/// Dynamic shared bytes of a `tv_rowc_*` launch: 4E's header, tables and
/// natural staging of the widest activation (the row loops' own,
/// [`crate::pers::trows_smem_bytes`]).
pub fn rowc_smem_bytes(max_d_in: u32) -> u32 {
    crate::pers::trows_smem_bytes(max_d_in)
}

/// Bytes a pass over rows reads, two accountings. `words`: the packed words
/// only, row stride included (the phase's GB/s). `old`: words, binary16 tails
/// and f32 row scales, the accounting of pers-rowloop and step1-micro (their
/// 0.9467 GB), kept for the fidelity rule.
pub fn rows_bytes(d_out: u32, stride_u32: u32, tail_w: u32) -> (u64, u64) {
    let w = u64::from(d_out) * 4 * u64::from(stride_u32);
    (w, w + u64::from(d_out) * (2 * u64::from(tail_w) + 4))
}

/// Row errors of a candidate against the served rows, site by site.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowErrors {
    pub rows: usize,
    /// Rows whose binary16 y equals the served one.
    pub y_equal: usize,
    /// Rows whose f32 value equals the served one, bit for bit.
    pub f_equal: usize,
    /// max over rows of |f - served f| / rms(served f of the row's site).
    pub max: f64,
    /// The 0.99 quantile of the same, over every row (nearest rank).
    pub q99: f64,
    /// Rows with a non-finite value where the served one is finite.
    pub nonfinite: usize,
}

impl RowErrors {
    pub fn bit_equal(&self) -> bool {
        self.y_equal == self.rows && self.f_equal == self.rows && self.rows > 0
    }

    /// Whether the arm keeps what it owes.
    pub fn passes(&self, owes: Owes) -> bool {
        match owes {
            Owes::Bits => self.bit_equal(),
            Owes::Bound(b) => self.rows > 0 && self.nonfinite == 0 && self.max <= b,
            Owes::Nothing => true,
        }
    }
}

/// `got` against `want`, binary16 rows and f32 values, `sites` the row count
/// of each site in order. A site whose served rows are all zero is measured
/// against an rms of 1.
pub fn row_errors(got_y: &[u16], got_f: &[f32], want_y: &[u16], want_f: &[f32], sites: &[usize]) -> RowErrors {
    let rows: usize = sites.iter().sum();
    assert!(got_y.len() == rows && got_f.len() == rows && want_y.len() == rows && want_f.len() == rows, "row counts differ");
    let mut e = RowErrors { rows, ..Default::default() };
    let mut rel = Vec::with_capacity(rows);
    let mut at = 0;
    for &n in sites {
        let w = &want_f[at..at + n];
        let ms = w.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / n.max(1) as f64;
        let rms = if ms > 0.0 { ms.sqrt() } else { 1.0 };
        for r in at..at + n {
            if got_y[r] == want_y[r] {
                e.y_equal += 1;
            }
            if got_f[r].to_bits() == want_f[r].to_bits() {
                e.f_equal += 1;
            }
            if !got_f[r].is_finite() && want_f[r].is_finite() {
                e.nonfinite += 1;
                rel.push(f64::INFINITY);
                continue;
            }
            rel.push((f64::from(got_f[r]) - f64::from(want_f[r])).abs() / rms);
        }
        at += n;
    }
    rel.sort_by(|a, b| a.total_cmp(b));
    e.max = rel.last().copied().unwrap_or(0.0);
    e.q99 = if rel.is_empty() { 0.0 } else { rel[((rel.len() as f64 * 0.99).ceil() as usize).clamp(1, rel.len()) - 1] };
    e
}

/// `bw_hash` of `bwprobe.cu`, word `i` of the probe's buffer.
pub fn bw_hash(i: u64, seed: u32) -> u32 {
    let mut h = (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(((i >> 32) as u32).wrapping_mul(0x85eb_ca6b)).wrapping_add(seed);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

/// The sum the probe owes over `n` words: every word once, as a u64.
pub fn bw_expected_sum(n: u64, seed: u32) -> u64 {
    (0..n).fold(0u64, |s, i| s.wrapping_add(u64::from(bw_hash(i, seed))))
}

/// The probe's kernels: name, symbol, words a load, loads in flight a step.
pub const BW_KERNELS: [(&str, &str, u32, u32); 3] =
    [("v1u1", "tv_bw_read_v1u1", 1, 1), ("v4u1", "tv_bw_read_v4u1", 4, 1), ("v4u4", "tv_bw_read_v4u4", 4, 4)];

/// Dynamic shared bytes of a probe launch: the header, one u64 a warp.
pub fn bw_smem_bytes(threads: u32) -> u32 {
    crate::pers::SMEM_HEADER + 8 * threads.div_ceil(32)
}

/// The probe's buffer: 2 GiB of u32, 2^29 words. At least 20 times the L2 of
/// any card the image targets (96 MB on an L40S, 40 MB on an A100; the RTX PRO
/// 6000's 128 MiB is 16 times under it), so a read of it is a DRAM read
/// whatever the L2 kept from the previous round.
pub const BW_WORDS: u64 = 1 << 29;

/// The probe's configurations: (kernel index into [`BW_KERNELS`], threads,
/// CTAs a SM, prefetch distance in grid steps). Every kernel at 4E's geometry
/// (512 threads, one CTA a SM), with and without the prefetch; then the
/// 128-bit unrolled kernel over thread counts and residencies.
pub fn bw_configs() -> Vec<(usize, u32, u32, u32)> {
    let mut v = Vec::new();
    for k in 0..BW_KERNELS.len() {
        for pf in [0u32, 2] {
            v.push((k, 512, 1, pf));
        }
    }
    for threads in [256u32, 512, 1024] {
        for per_sm in [1u32, 2, 4] {
            // 1,536 threads a SM on sm_89 (the L40S); the card's occupancy check refuses the rest at launch.
            if threads * per_sm > 1536 || (threads == 512 && per_sm == 1) {
                continue;
            }
            for pf in [0u32, 2] {
                v.push((2, threads, per_sm, pf));
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_parses_and_names_the_served_arm_first() {
        let a = arms();
        assert!(a.len() >= 2);
        assert_eq!(a[0].name, "served");
        assert_eq!(a[0].strukt, "RowcServed");
        let names: Vec<&str> = a.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, ["served", "served_np", "acc2", "pword"]);
        assert_eq!(a[2].owes, Owes::Bound(1e-5));
        assert_eq!(a[3].owes, Owes::Nothing);
        assert_eq!(a[2].bpi, 2);
        assert_eq!(a[1].symbol(), "tv_rowc_served_np");
        assert_eq!(a[1].gu_symbol(), "tv_rowc_gu_served_np");
        assert_eq!(a[1].relay_symbol(), "tv_rowc_relay_served_np");
    }

    #[test]
    fn the_module_carries_every_arm_and_keeps_the_row_loop_unit_first() {
        let t = rowc_unit_text("#define TILE_BLOCKS 64u");
        assert!(t.starts_with(&crate::pers::trows_unit_text("#define TILE_BLOCKS 64u")), "the row-loop unit, unchanged, first");
        for a in arms() {
            assert!(t.contains(&format!("X({}, {},", a.name, a.strukt)), "{}", a.name);
            assert!(t.contains(&format!("struct {} {{", a.strukt)), "{}: no struct {}", a.name, a.strukt);
        }
        for (_, sym, _, _) in BW_KERNELS {
            assert!(t.contains(&format!("BW_KERNEL({sym}, ")), "{sym}");
        }
    }

    #[test]
    fn a_malformed_registry_is_refused() {
        let ok = "#define ROWC_ARMS(X) \\\n    X(served, RowcServed, ROWC_BITS, 0, 0, 1) \\\n    X(b, B, ROWC_BOUND, 1e-4, 1, 2)\n\n";
        let a = parse_arms(ok).unwrap();
        assert_eq!(a[1].owes, Owes::Bound(1e-4));
        assert!(a[1].relay);
        for bad in [
            "#define ROWC_ARMS(X) \\\n    X(b, B, ROWC_BITS, 0, 0, 1)\n",
            "#define ROWC_ARMS(X) \\\n    X(served, RowcServed, ROWC_BITS, 0, 0, 1) \\\n    X(served, R, ROWC_BITS, 0, 0, 1)\n",
            "#define ROWC_ARMS(X) \\\n    X(served, RowcServed, ROWC_BITS, 0, 0, 1) \\\n    X(b, B, ROWC_BOUND, 0, 0, 1)\n",
            "#define ROWC_ARMS(X) \\\n    X(served, RowcServed, ROWC_BITS, 0, 0, 1) \\\n    X(b, B, ROWC_BITS, 1e-3, 0, 1)\n",
            "#define ROWC_ARMS(X) \\\n    X(served, RowcServed, ROWC_BITS, 0, 0, 1) \\\n    X(b, B, ROWC_MAYBE, 0, 0, 1)\n",
            "#define ROWC_ARMS(X) \\\n    X(served, RowcServed, ROWC_BITS, 0, 0, 1) \\\n    X(b, B, ROWC_BITS, 0, 2, 1)\n",
            "#define ROWC_ARMS(X) \\\n    X(served, RowcServed, ROWC_BITS, 0, 0, 1) \\\n    X(b, B, ROWC_BITS, 0, 0, 0)\n",
            "#define ROWC_ARMS(X) \\\n    X(served, RowcServed, ROWC_BITS, 0, 0) \\\n",
            "#define ROWC_ARMS(X) \\\n    X(served, RowcServed, ROWC_BITS, 0, 1, 1)\n",
        ] {
            assert!(parse_arms(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn row_errors_count_and_scale_by_the_site_rms() {
        let want_f = [3.0f32, -4.0, 0.0, 0.0];
        let want_y: Vec<u16> = want_f.iter().map(|&v| crate::f16_bits(v)).collect();
        let e = row_errors(&want_y, &want_f, &want_y, &want_f, &[2, 2]);
        assert!(e.bit_equal() && e.passes(Owes::Bits) && e.max == 0.0);
        let mut got = want_f;
        got[1] = -4.0 + 1e-2;
        let gy: Vec<u16> = got.iter().map(|&v| crate::f16_bits(v)).collect();
        let e = row_errors(&gy, &got, &want_y, &want_f, &[2, 2]);
        assert!(!e.passes(Owes::Bits));
        let rms = (12.5f64).sqrt();
        let want_max = (f64::from(got[1]) - f64::from(want_f[1])).abs() / rms;
        assert_eq!(e.max, want_max);
        assert_eq!((e.y_equal, e.f_equal), (3, 3));
        assert!(e.passes(Owes::Bound(1e-2)) && !e.passes(Owes::Bound(1e-3)));
        got[3] = f32::NAN;
        let e = row_errors(&gy, &got, &want_y, &want_f, &[2, 2]);
        assert_eq!(e.nonfinite, 1);
        assert!(!e.passes(Owes::Bound(0.5)));
    }

    #[test]
    fn the_probe_hash_is_not_constant_and_its_sum_is_order_free() {
        let v: Vec<u32> = (0..1024).map(|i| bw_hash(i, 7)).collect();
        let mut d = v.clone();
        d.sort();
        d.dedup();
        assert!(d.len() > 1000, "the buffer would be near constant");
        assert_ne!(bw_hash(5, 7), bw_hash(5, 8), "the seed moves the words");
        assert_ne!(bw_hash(5, 7), bw_hash(5 + (1 << 32), 7), "the high half of the index moves the words");
        let s: u64 = v.iter().map(|&x| u64::from(x)).sum();
        assert_eq!(bw_expected_sum(1024, 7), s);
        const { assert!(BW_WORDS * 4 >= 20 * 96_000_000, "the buffer is 20 L40S L2 at least") };
    }

    #[test]
    fn the_probe_configs_fit_a_sm() {
        let c = bw_configs();
        assert_eq!(c.len(), 6 + 2 * 5);
        for &(k, t, per, _) in &c {
            assert!(k < BW_KERNELS.len() && t * per <= 1536 && t % 32 == 0);
        }
        assert_eq!(bw_smem_bytes(512), crate::pers::SMEM_HEADER + 128);
    }

    /// Writes the bench module at the L40S's tile (64) for the offline SASS
    /// count, `ops/rowcand-sass.sh`. Ignored: it writes under target/.
    #[test]
    #[ignore]
    fn dump_rowcand_module() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/rowcand");
        std::fs::create_dir_all(&dir).unwrap();
        // the card's define on sm_89 (tile::Tile::define), so the digest is the one rowbench prints there
        let text = rowc_unit_text("#define TILE_BLOCKS 64u\n");
        assert_eq!(text.matches("#define TILE_BLOCKS").count(), 1);
        std::fs::write(dir.join("rowcand_unit.cu"), &text).unwrap();
        eprintln!("rowcand_unit: {} bytes, sha256 {}", text.len(), crate::sha256::sha256_hex(text.as_bytes()));
    }

    #[test]
    fn the_two_byte_accountings() {
        // gate/up of the 4B: 9,728 rows of 106 blocks, stride 160 u32, tail 16
        let (w, old) = rows_bytes(9728, 160, 16);
        assert_eq!(w, 9728 * 640);
        assert_eq!(old, 9728 * (640 + 32 + 4));
    }
}
