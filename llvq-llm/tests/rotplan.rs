//! Lot A4 — the rotation of a shared activation, computed once.
//!
//! Everything the hoist rests on is checked here, on a machine with no card:
//! that the file's rotation keys really do partition its activation sites,
//! that the sequencing hands every use the rotation of *its own* activation,
//! that the hoist actually removes launches, and that it does so for the three
//! runtime layouts alike. What cannot be checked here is listed in the lot's
//! report — everything inside `llvq-llm/src/fused_cuda.rs`, which compiles on
//! no target this machine has.
//!
//! ## Why correctness and efficiency are two disjoint tests
//!
//! `every_use_gets_the_rotation_of_its_own_activation` passes on an
//! implementation that rotates unconditionally, every time: that
//! implementation is *correct*, it is merely useless. `the_hoist_actually_
//! hoists` passes on an implementation that returns a stale rotation: that one
//! is fast and wrong. Neither test can be dropped, and a single test asserting
//! "fast and right" would be two assertions wearing one name.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use llvq_artifact::runtime::{ClassTable, PLANES14_BYTES};
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused::{
    planes_source_names, FusedLayout, FusedMatrix, HostStream, RotKey, Transcoder,
};
use llvq_llm::model::Act;
use llvq_llm::rotplan::{
    act_of_suffix, check_key, check_rotation_partition, drive_rows, rot_launches_per_token,
    rotation_sites, RotShare, RotSite,
};
use llvq_quant::rotation::Rotation;
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::N13;

// ---------------------------------------------------------------------------
// Synthetic models
// ---------------------------------------------------------------------------

/// The seven projections of a Qwen3 block, in the order the artifact writes
/// them, with the activation each consumes.
const PLAN: [(&str, Act); 7] = [
    ("self_attn.q_proj", Act::Attn),
    ("self_attn.k_proj", Act::Attn),
    ("self_attn.v_proj", Act::Attn),
    ("self_attn.o_proj", Act::AttnOut),
    ("mlp.gate_proj", Act::Mlp),
    ("mlp.up_proj", Act::Mlp),
    ("mlp.down_proj", Act::MlpOut),
];

/// Qwen3-4B's real activation widths, so the width partition and the key
/// partition are genuinely different objects — `Attn` and `Mlp` are both
/// `hidden_size` wide, which is precisely what makes `d_in` an unusable key.
fn width_4b(act: Act) -> usize {
    match act {
        Act::Attn | Act::Mlp => 2560,
        Act::AttnOut => 4096,
        Act::MlpOut => 9728,
    }
}

/// The seed the quantizer derives for one `(block, activation)` pair —
/// `calib::effective_rotation_seed`, restated rather than called, so that a
/// change to the derivation shows up as a disagreement rather than propagating
/// silently into the yardstick.
fn seed_of(base: u64, layer: usize, act: Act) -> u64 {
    base ^ ((layer as u64) << 32) ^ (act.index() << 16)
}

fn matrix(layer: usize, suffix: &str, d_in: usize, rotation: Option<RotKey>) -> FusedMatrix {
    FusedMatrix {
        name: format!("model.layers.{layer}.{suffix}.weight"),
        d_out: 8,
        d_in,
        nblocks: d_in / DIM,
        tail_w: d_in % DIM,
        stream: HostStream::Planes14 { words: Vec::new() },
        gscale: [0.5, 1.5],
        rscale: vec![1.0; 8],
        tail: Vec::new(),
        rotation,
        bytes: 0,
    }
}

/// `layers` blocks of seven projections, keyed exactly the way the quantizer
/// keys them.
fn model_4b_shaped(layers: usize, base: u64) -> Vec<FusedMatrix> {
    let mut out = Vec::new();
    for layer in 0..layers {
        for (suffix, act) in PLAN {
            let w = width_4b(act);
            out.push(matrix(layer, suffix, w, Some((w, seed_of(base, layer, act)))));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// T2 / T3 — the partition check itself
// ---------------------------------------------------------------------------

/// **T2.** A file whose two consumers of one activation carry different
/// rotations is refused, and the message names both matrices.
///
/// Both directions, because they are two statements and a check of one passes
/// on a file the other breaks:
///
///  * two consumers of one activation disagreeing — the hoist would hand one
///    of them the other's basis;
///  * two activations sharing one key — the launch count would be wrong, and a
///    future file could make two different activations look interchangeable.
#[test]
fn a_file_whose_groups_disagree_is_refused() {
    // Direction 1: k_proj drifts off q_proj's rotation.
    let mut m = model_4b_shaped(2, 0xABCD);
    m[8].rotation = Some((2560, 0xdead_beef));
    let e = check_rotation_partition(&m).expect_err("a split group must be refused");
    assert!(e.contains("model.layers.1.self_attn.q_proj"), "{e}");
    assert!(e.contains("model.layers.1.self_attn.k_proj"), "{e}");

    // Direction 2: gate/up of layer 0 handed q/k/v's rotation. Same width, so
    // nothing about the shapes is wrong — only the identity of the basis.
    let mut m = model_4b_shaped(2, 0xABCD);
    let attn0 = m[0].rotation;
    m[4].rotation = attn0;
    m[5].rotation = attn0;
    let e = check_rotation_partition(&m).expect_err("two shared activations must be refused");
    assert!(e.contains("self_attn.q_proj"), "{e}");
    assert!(e.contains("mlp.gate_proj"), "{e}");

    // A matrix with no rotation at all: the fused path cannot read it, and it
    // is named rather than skipped.
    let mut m = model_4b_shaped(1, 0xABCD);
    m[6].rotation = None;
    let e = check_rotation_partition(&m).expect_err("a matrix with no rotation must be refused");
    assert!(e.contains("mlp.down_proj"), "{e}");

    // And the well-formed model passes, so the three failures above are not a
    // check that refuses everything.
    check_rotation_partition(&model_4b_shaped(2, 0xABCD)).expect("well-formed model");
}

/// **T3.** The key partition is *not* the width partition — the mutation
/// anyone extending the existing scratch pool would reach for first.
///
/// `fused_cuda.rs` used to key its rotated-activation pool on `d_in`. On
/// Qwen3-4B that key merges `Attn` and `Mlp` inside every layer (both 2560)
/// **and** merges all 36 layers with each other. Keying the hoist the same way
/// would rotate `h` once and reuse it for `post_attention_layernorm(x)`, which
/// is a different vector: finite, plausible, wrong.
#[test]
fn the_key_partition_is_not_the_width_partition() {
    let m = model_4b_shaped(36, 0x5EED);
    let sites = rotation_sites(&m).expect("well-formed partition");
    assert_eq!(sites.len(), 144, "36 layers × 4 activations");

    let widths: HashSet<usize> = m.iter().map(|x| x.d_in).collect();
    assert_eq!(widths.len(), 3, "2560, 4096, 9728");
    assert!(
        widths.len() < sites.len(),
        "a key on d_in alone would merge {} sites into {} classes",
        sites.len(),
        widths.len()
    );

    // The lethal instance, spelled out rather than left to the counting: two
    // projections of the *same layer* and the *same width* that must not share
    // a rotation.
    let q = m
        .iter()
        .find(|x| x.name == "model.layers.7.self_attn.q_proj.weight")
        .expect("q_proj");
    let gate = m
        .iter()
        .find(|x| x.name == "model.layers.7.mlp.gate_proj.weight")
        .expect("gate_proj");
    assert_eq!(q.d_in, gate.d_in);
    assert_ne!(
        q.rotation, gate.rotation,
        "same width, same layer, two activations: d_in cannot be the key"
    );

    // Same width across layers, too.
    let q13 = m
        .iter()
        .find(|x| x.name == "model.layers.13.self_attn.q_proj.weight")
        .expect("q_proj");
    assert_ne!(q.rotation, q13.rotation);
}

/// The consumer histogram the whole lot rests on: 3 sites of three consumers,
/// 1 of two... per layer. Stated here on the synthetic model so that T1's
/// assertion on the real files is a comparison, not a discovery.
#[test]
fn a_block_has_four_activation_sites() {
    let sites = rotation_sites(&model_4b_shaped(1, 1)).expect("partition");
    let mut by_size: HashMap<usize, usize> = HashMap::new();
    for s in &sites {
        *by_size.entry(s.consumers).or_default() += 1;
    }
    assert_eq!(sites.len(), 4);
    assert_eq!(by_size.get(&3), Some(&1), "q/k/v");
    assert_eq!(by_size.get(&2), Some(&1), "gate/up");
    assert_eq!(by_size.get(&1), Some(&2), "o_proj and down_proj");
}

// ---------------------------------------------------------------------------
// T4 — correctness of the sequencing
// ---------------------------------------------------------------------------

/// One activation row as it sits in memory: a shared allocation and the offset
/// the row starts at.
///
/// The two fields are the point. A prefill's rows are `narrow` views into one
/// contiguous buffer, so they share `base` and differ only by `off` — which is
/// exactly the shape that kills a cache keyed on the base pointer (mutant M3).
#[derive(Clone)]
struct Row {
    base: usize,
    off: usize,
    v: Vec<f64>,
}

/// One projection of a group: what it is called, the rotation it owes, and a
/// tag unique over the whole script.
///
/// The tag is not decoration. The same projection is visited on five passes,
/// each with a different activation, so keying a result by `(name, row)` would
/// silently compare pass 4's value against pass 0's oracle — and *pass*, not
/// name, is what a stale cache would get wrong. That mistake made the first
/// draft of this test fail on a correct implementation, which is the friendly
/// direction but the same defect.
struct Site {
    name: String,
    key: RotKey,
    tag: usize,
}

/// A rotated activation, as the model's `Rotated` is: a value carrying the key
/// that produced it.
struct HostRotated {
    key: RotKey,
    v: Vec<f64>,
}

/// The host stand-in for `rot_apply`, counting its launches.
///
/// The transform itself is memoized by key — that is `RotationTables::build`'s
/// own behaviour and it is sound, because `Rotation::new(n, seed)` is a pure
/// function of its arguments. What is *not* memoized, here or in the model, is
/// the **result** of applying it to an activation.
struct HostRotator {
    transforms: HashMap<RotKey, Rotation>,
    applies: Cell<usize>,
}

impl HostRotator {
    fn new(keys: &[RotKey]) -> Self {
        Self {
            transforms: keys
                .iter()
                .map(|&(n, seed)| ((n, seed), Rotation::new(n, seed)))
                .collect(),
            applies: Cell::new(0),
        }
    }

    fn rotate(&self, key: RotKey, row: &Row) -> HostRotated {
        self.applies.set(self.applies.get() + 1);
        let mut v = row.v.clone();
        self.transforms[&key].apply(&mut v);
        HostRotated { key, v }
    }

    fn applies(&self) -> usize {
        self.applies.get()
    }
}

/// One group of projections consuming one activation, over some rows.
struct Group {
    key: RotKey,
    sites: Vec<Site>,
    rows: Vec<Row>,
}

/// Run a scripted sequence of groups through the shared driver, returning
/// every `(site, row)` result the sites saw.
fn run(
    share: RotShare,
    rot: &HostRotator,
    script: &[Group],
) -> Result<HashMap<(usize, usize), Vec<f64>>, String> {
    let mut seen = HashMap::new();
    for g in script {
        let out = drive_rows(
            share,
            &g.sites,
            g.rows.len(),
            |s: &Site, row: usize| -> Result<HostRotated, String> {
                Ok(rot.rotate(s.key, &g.rows[row]))
            },
            |s: &Site, r: &HostRotated, row: usize| -> Result<((usize, usize), Vec<f64>), String> {
                check_key(&s.name, Some(s.key), Some(r.key))?;
                Ok(((s.tag, row), r.v.clone()))
            },
        )?;
        for per_site in out {
            for (k, v) in per_site {
                assert!(seen.insert(k, v).is_none(), "a (site, row) returned twice");
            }
        }
    }
    Ok(seen)
}

/// The scripted decode: `layers` blocks × 4 activations, a 3-row prefill
/// followed by 4 one-row decode steps.
///
/// The contents are what make the test lethal, and each property is asserted
/// below rather than described here.
fn script(layers: usize, base: u64) -> Vec<Group> {
    let mut rng = SplitMix64::new(0xA4_0000 ^ base);
    let mut out = Vec::new();
    // pass 0 is the prefill (3 rows sharing one allocation); passes 1..5 are
    // decode steps of one row each.
    for pass in 0..5usize {
        let rows_in_pass = if pass == 0 { 3 } else { 1 };
        for layer in 0..layers {
            for act in Act::ALL {
                let n = width_4b(act);
                let key = (n, seed_of(base, layer, act));
                // One allocation per (pass, layer, act): the prefill's rows are
                // views into it at three different offsets.
                let alloc = out.len() + 1;
                let mut rows: Vec<Row> = (0..rows_in_pass)
                    .map(|r| Row {
                        base: alloc,
                        off: r * n,
                        v: (0..n).map(|_| rng.next_gaussian()).collect(),
                    })
                    .collect();
                // One ulp apart, on the prefill's first two rows: a cache that
                // compared contents "approximately" would collapse them, and
                // the oracle below is a bit-for-bit equality precisely so that
                // it cannot.
                if rows_in_pass == 3 {
                    rows[1].v = rows[0].v.clone();
                    let x = rows[1].v[0];
                    rows[1].v[0] = f64::from_bits(x.to_bits() + 1);
                }
                let sites = act
                    .consumers()
                    .iter()
                    .enumerate()
                    .map(|(i, suffix)| Site {
                        name: format!("model.layers.{layer}.{suffix}.weight"),
                        key,
                        tag: 8 * out.len() + i,
                    })
                    .collect();
                out.push(Group { key, sites, rows });
            }
        }
    }
    out
}

/// **T4.** Every projection is handed the rotation of *its own* activation,
/// under both modes, bit for bit.
///
/// Kills **M2** (a cache keyed on `RotKey` with no invalidation — dies only
/// because the fixture revisits every key on five passes with different
/// contents) and **M3** (a cache keyed on `(RotKey, base pointer)` — dies only
/// because the prefill's three rows share one allocation and differ by their
/// offset).
///
/// The oracle is recomputed per use from the raw activation, so a mistake in
/// the driver cannot also be in the yardstick.
#[test]
fn every_use_gets_the_rotation_of_its_own_activation() {
    let script = script(3, 0x4B);
    let keys: Vec<RotKey> = script.iter().map(|g| g.key).collect();

    // --- lethality guards, as assertions ---
    let total_prepares: usize = script
        .iter()
        .map(|g| g.sites.len() * g.rows.len())
        .sum::<usize>();
    let distinct: HashSet<RotKey> = keys.iter().copied().collect();
    assert!(
        distinct.len() < total_prepares,
        "a fixture that revisits no key cannot kill a cache: \
         {} keys for {total_prepares} uses",
        distinct.len()
    );
    // At least two consecutive groups on the same key with different contents.
    let repeats = script
        .windows(1)
        .enumerate()
        .filter(|(i, _)| {
            script[*i..]
                .iter()
                .skip(1)
                .any(|g| g.key == script[*i].key && g.rows[0].v != script[*i].rows[0].v)
        })
        .count();
    assert!(repeats >= 2, "revisited keys with different contents are needed");
    // One ulp apart, sharing an allocation, differing by offset alone.
    let prefill = &script[0];
    assert_eq!(prefill.rows.len(), 3);
    assert_eq!(prefill.rows[0].base, prefill.rows[1].base);
    assert_ne!(prefill.rows[0].off, prefill.rows[1].off);
    assert_ne!(prefill.rows[0].v, prefill.rows[1].v);
    assert_eq!(
        (prefill.rows[0].v[0].to_bits() as i128 - prefill.rows[1].v[0].to_bits() as i128).abs(),
        1,
        "the first two prefill rows must differ by a single ulp"
    );

    // --- the property ---
    for share in [RotShare::Off, RotShare::On] {
        let rot = HostRotator::new(&keys);
        let seen = run(share, &rot, &script).expect("valid sequence");
        assert_eq!(
            seen.len(),
            total_prepares,
            "{share:?}: every (site, row) must return a result"
        );
        // Independent oracle, per use.
        let mut checked = 0usize;
        for g in &script {
            for s in &g.sites {
                for (r, row) in g.rows.iter().enumerate() {
                    let mut want = row.v.clone();
                    Rotation::new(g.key.0, g.key.1).apply(&mut want);
                    let got = seen
                        .get(&(s.tag, r))
                        .unwrap_or_else(|| panic!("{} row {r} never seen", s.name));
                    assert_eq!(
                        got, &want,
                        "{share:?}: {} row {r} got the rotation of another activation",
                        s.name
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, total_prepares);
    }
}

/// A projection handed another group's rotation is refused by name — the same
/// guard `Proj::forward_with` plants, seen through the driver.
#[test]
fn a_group_whose_sites_disagree_is_refused_at_use() {
    let mut script = script(1, 7);
    let foreign = script[1].key;
    script[0].sites[2].key = foreign;
    let keys: Vec<RotKey> = script
        .iter()
        .flat_map(|g| g.sites.iter().map(|s| s.key).chain(std::iter::once(g.key)))
        .collect();
    let rot = HostRotator::new(&keys);
    let e = run(RotShare::On, &rot, &script).expect_err("foreign key");
    assert!(e.contains("v_proj"), "{e}");
}

// ---------------------------------------------------------------------------
// T5 / T9 — the hoist removes launches, for every layout
// ---------------------------------------------------------------------------

/// **T5.** The hoist actually hoists: 144 rotations a token instead of 252.
///
/// Counted through the very driver `model::group_forward` runs, so this is the
/// launch count the model issues rather than a restatement of it. Kills
/// **M4** — a `prepare` that re-rotates unconditionally, which
/// `every_use_gets_the_rotation_of_its_own_activation` cannot see because it
/// is *correct*.
#[test]
fn the_hoist_actually_hoists() {
    // One decode step over the published 4B's 36 blocks.
    let one_step: Vec<Group> = script(36, 0x11)
        .into_iter()
        .filter(|g| g.rows.len() == 1)
        .take(36 * 4)
        .collect();
    assert_eq!(one_step.len(), 144, "36 layers × 4 activations");
    let sites: usize = one_step.iter().map(|g| g.sites.len()).sum();
    assert_eq!(sites, 252, "36 layers × 7 projections");

    for (share, want) in [(RotShare::On, 144usize), (RotShare::Off, 252)] {
        let keys: Vec<RotKey> = one_step.iter().map(|g| g.key).collect();
        let rot = HostRotator::new(&keys);
        run(share, &rot, &one_step).expect("valid sequence");
        assert_eq!(rot.applies(), want, "{share:?}: rot launches/token");
    }

    // And the number `bin/fusedrun` prints agrees with the number the driver
    // issues. Two independent computations of the same quantity: one counts
    // launches through the sequencing, the other reads the model's matrices.
    let m = model_4b_shaped(36, 0x11);
    assert_eq!(rot_launches_per_token(RotShare::On, &m), 144);
    assert_eq!(rot_launches_per_token(RotShare::Off, &m), 252);
}

/// **T9.** The hoist is not gated on the runtime layout.
///
/// The rotation acts on `x`, the layout describes `W`, and the two are
/// orthogonal — but the implementation that gates both behind one flag is a
/// real temptation, and it would cost `Planes12x` 108 launches for nothing.
/// `Planes12x` in particular *cannot* be segmented (its overlay is indexed by
/// local row), which is exactly why it must get the half of A4 that does not
/// need segmentation.
///
/// Real streams, transcoded by the real `Transcoder` for each layout, so this
/// exercises the three arms rather than asserting a tautology about a constant.
#[test]
fn the_hoist_is_not_gated_on_the_layout() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let _ = (&fd, &table); // built once so the three arms are comparable work
    for layout in [
        FusedLayout::Planes14,
        FusedLayout::Planes12x,
        FusedLayout::Slot32,
    ] {
        let tr = Transcoder::new(layout).expect("transcodeur");
        let mut rng = SplitMix64::new(0xC0FFEE ^ layout.name().len() as u64);
        let mut matrices = Vec::new();
        for layer in 0..4usize {
            for (suffix, act) in PLAN {
                // Tiny matrices: what is under test is the launch count, not
                // the payload. One block per row keeps `Planes12x`'s exact
                // re-encode of 5-level blocks affordable in a unit test.
                let (d_out, nblocks) = (8usize, 1usize);
                let idx: Vec<u64> = (0..d_out * nblocks).map(|_| 1 + rng.next() % N13).collect();
                let gains: Vec<u32> = (0..d_out * nblocks).map(|_| (rng.next() & 1) as u32).collect();
                let (stream, bytes) = tr
                    .stream(&idx, &gains, d_out, nblocks)
                    .expect("transcode");
                let w = width_4b(act);
                matrices.push(FusedMatrix {
                    stream,
                    bytes,
                    ..matrix(layer, suffix, w, Some((w, seed_of(0x9, layer, act))))
                });
            }
        }
        check_rotation_partition(&matrices).expect("partition");
        assert_eq!(
            rot_launches_per_token(RotShare::On, &matrices),
            16,
            "{}: 4 layers × 4 activations",
            layout.name()
        );
        assert_eq!(
            rot_launches_per_token(RotShare::Off, &matrices),
            28,
            "{}: 4 layers × 7 projections",
            layout.name()
        );
    }
}

/// **T10.** The source list handed to NVRTC, restated where the rotation lot
/// can see it.
///
/// Not a style check. NVRTC receives one concatenated string; adding a file —
/// even one nothing calls — can move the register allocation of `tv_planes_h`,
/// and a register allocation that moves is invisible to every correctness test
/// in this repository while being exactly what the published 88.4 tok/s and
/// 4.804 b/weight were measured on.
///
/// 🚨 **This test used to be named "…is unchanged" and to assert that lot A4
/// adds no kernel source. Lot D1 adds one**, `tv_planes_seg_h.cu`, and the
/// consequence above is not hypothetical: the served unit is no longer
/// byte-identical to the one the 2026-08-06 figures were measured on. That is
/// accepted deliberately and mitigated by the shape of the measurement rather
/// than by the code — **the reference of the fusion A/B is the `LLVQ_FUSE=0`
/// arm of the same job**, which shares this very translation unit, never a
/// published absolute. See `fused::planes_source_names` for the argument in
/// full. What this assertion still buys: the list cannot drift again without
/// two files disagreeing.
#[test]
fn the_nvrtc_source_list_is_what_the_lot_declares() {
    assert_eq!(planes_source_names(FusedLayout::Slot32), &[] as &[&str]);
    assert_eq!(
        planes_source_names(FusedLayout::Planes14),
        &["llvq_planes.cuh", "planes.cu", "tv_planes_h.cu", "tv_planes_seg_h.cu"]
    );
    assert_eq!(
        planes_source_names(FusedLayout::Planes12x),
        &[
            "llvq_planes.cuh",
            "planes.cu",
            "tv_planes_h.cu",
            "tv_planes_seg_h.cu",
            "llvq_planes12.cuh",
            "tv_planes12x_h.cu",
        ]
    );
    // The record width the four-word window is arithmetic on, restated so a
    // silent change of layout geometry cannot slip past this file either.
    assert_eq!(PLANES14_BYTES, 14);
}

/// `LLVQ_ROT_SHARE` refuses anything that is not a mode — the same contract as
/// `LLVQ_FUSED_LAYOUT`, restated here beside its siblings. (The exhaustive
/// case list is a unit test in `src/rotplan.rs`; this is the reminder that the
/// two variables answer to one rule.)
#[test]
fn rot_share_parse_refuses_anything_else() {
    assert_eq!(RotShare::parse(Some("1")), Ok(RotShare::On));
    assert!(RotShare::parse(Some("planes14")).is_err());
    assert!(FusedLayout::parse(Some("1")).is_err());
}

// ---------------------------------------------------------------------------
// The gate is wired into `fused::load`, not merely available beside it
// ---------------------------------------------------------------------------

/// A minimal self-contained artifact: `layers` blocks of seven one-block
/// matrices, plus the two blobs `fused::load` insists on.
///
/// Small enough (56 blocks at four layers) that a whole `fused::load` runs in
/// milliseconds — which is what makes it possible to check that the partition
/// gate is *called* rather than merely present. Without this, deleting the
/// call from `fused::load` would kill no test on this machine: every other
/// exercise of `load` needs a multi-gigabyte archive.
fn synthetic_sealed(path: &std::path::Path, layers: usize, sabotage: bool) {
    let mut rng = SplitMix64::new(0xDEC0DE);
    let f = std::fs::File::create(path).expect("create");
    let mut w = llvq_artifact::ArtifactWriter::new(std::io::BufWriter::new(f), (layers * 7) as u32)
        .expect("header");
    for layer in 0..layers {
        for (i, (suffix, act)) in PLAN.iter().enumerate() {
            let (d_out, d_in) = (8usize, DIM);
            // The sabotage: layer 0's k_proj drifts off q_proj's rotation,
            // which is precisely what the hoist may not survive.
            let seed = if sabotage && layer == 0 && i == 1 {
                0xBAD_5EED
            } else {
                seed_of(0x33, layer, *act)
            };
            w.push_raw(&llvq_artifact::RawMatrix {
                name: format!("model.layers.{layer}.{suffix}.weight"),
                d_out,
                d_in,
                indices: (0..d_out).map(|_| 1 + rng.next() % N13).collect(),
                gains: (0..d_out).map(|_| (rng.next() & 1) as u32).collect(),
                row_scales: vec![1.0; d_out],
                centroids: vec![0.5, 1.5],
                rotation_seed: Some(seed),
                shell_cap: 13,
                tail: Vec::new(),
            })
            .expect("matrix");
        }
    }
    w.seal(
        &[],
        &[
            llvq_artifact::Blob { name: "config.json".into(), bytes: b"{}".to_vec() },
            llvq_artifact::Blob { name: "tokenizer.json".into(), bytes: b"{}".to_vec() },
        ],
    )
    .expect("seal");
}

/// `fused::load` refuses a file whose rotation keys do not partition its
/// activation sites — and accepts one that does.
///
/// The second half is not decoration: a gate that refuses everything would
/// pass the first assertion alone, and this is the load path every fused run
/// on a card goes through.
#[test]
fn the_load_path_runs_the_partition_gate() {
    let dir = std::env::temp_dir().join(format!("llvq-a4-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("directory");

    let good = dir.join("good.llvq");
    synthetic_sealed(&good, 4, false);
    let m = match llvq_llm::fused::load(good.to_str().expect("utf8"), FusedLayout::Planes14) {
        Ok(m) => m,
        Err(e) => panic!("a well-formed file must load: {e}"),
    };
    assert_eq!(m.matrices.len(), 28);
    assert_eq!(rot_launches_per_token(RotShare::On, &m.matrices), 16);

    let bad = dir.join("bad.llvq");
    synthetic_sealed(&bad, 4, true);
    let e = match llvq_llm::fused::load(bad.to_str().expect("utf8"), FusedLayout::Planes14) {
        Ok(_) => panic!("a broken partition must be refused at load"),
        Err(e) => e,
    };
    assert!(e.contains("self_attn.q_proj"), "{e}");
    assert!(e.contains("self_attn.k_proj"), "{e}");

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// T1 — the real files
// ---------------------------------------------------------------------------

/// Where a sealed artifact lives, or a loud failure naming it.
///
/// `#[ignore]`d rather than skipped: the form this replaces —
/// `eprintln!("SKIP"); return;` — reports `ok` on every machine without the
/// file, which is a statement about the filesystem dressed up as a statement
/// about the artifact. See `llvq-artifact/tests/common/mod.rs`, whose contract
/// this restates for a second crate rather than sharing (the two crates have no
/// common test crate, and a third copy of the message would be a third chance
/// to drift).
fn sealed(env: &str, default: &str) -> std::path::PathBuf {
    let p = match (std::env::var_os(env), std::env::var_os("HOME")) {
        (Some(p), _) => std::path::PathBuf::from(p),
        (None, Some(h)) => std::path::Path::new(&h).join(default),
        (None, None) => panic!("neither ${env} nor $HOME: no path to try"),
    };
    assert!(
        p.exists(),
        "the sealed artifact is not on this machine: {}\n\n\
         This sweep checks on the REAL file that the rotation seeds partition the \
         activation sites, the premise of lot A4. It is #[ignore] so it never \
         runs by accident; being here means it was asked for by name or with \
         --include-ignored, so it fails instead of printing SKIP and returning `ok`.\n\n    \
         {env}=/path/to/the.llvq \\\n        cargo test --release -p llvq-llm -- --include-ignored",
        p.display()
    );
    p
}

/// The rotation keys and names of a sealed artifact, without transcoding it.
///
/// `fused::load` would answer the same question and cost 142 s of transcoding
/// (much more under `Planes12x`, which re-encodes every 5-level block). What
/// T1 is about is the *file's* keys, and those are in the matrix headers.
fn matrices_of(path: &std::path::Path) -> Vec<FusedMatrix> {
    let f = std::fs::File::open(path).expect("open");
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r).expect("header");
    assert!(
        head.is_self_contained(),
        "{}: projections-only artifact, not a sealed model",
        path.display()
    );
    (0..head.matrices)
        .map(|_| {
            let m = llvq_artifact::read_matrix_raw(&mut r).expect("matrix");
            let rotation = m.rotation_seed.map(|s| (m.d_in, s));
            let mut fm = matrix(0, "self_attn.q_proj", m.d_in, rotation);
            fm.name = m.name;
            fm.d_out = m.d_out;
            fm
        })
        .collect()
}

/// **T1.** On the published artifacts themselves: the rotation keys partition
/// the activation sites, exactly.
///
/// This is the premise the whole lot rests on, and it is a property of the
/// *files*, not of the code — so no amount of source review establishes it. It
/// would fall to a future run whose seeds collided, or to a one-character
/// change in `calib::effective_rotation_seed` (`<< 32` → `<< 16` would fold
/// layer 1's `Attn` onto layer 0's `MlpOut`), and the design would become
/// unfounded without a line of this lot changing.
#[test]
#[ignore = "reads a multi-gigabyte sealed artifact, absent from the repository"]
fn rotation_keys_partition_the_sites() {
    for (env, default, layers) in [
        ("LLVQ_SEALED_4B", "qwen3-4b-llvq.bin", 36usize),
        ("LLVQ_SEALED_8B", "qwen3-8b-llvq.bin", 36),
    ] {
        let path = sealed(env, default);
        let m = matrices_of(&path);
        let sites: Vec<RotSite> = rotation_sites(&m)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));

        assert_eq!(m.len(), layers * 7, "{}: projections", path.display());
        assert_eq!(sites.len(), layers * 4, "{}: sites", path.display());

        let mut hist: HashMap<usize, usize> = HashMap::new();
        for s in &sites {
            *hist.entry(s.consumers).or_default() += 1;
        }
        assert_eq!(hist.get(&3), Some(&layers), "{}: q/k/v", path.display());
        assert_eq!(hist.get(&2), Some(&layers), "{}: gate/up", path.display());
        assert_eq!(
            hist.get(&1),
            Some(&(2 * layers)),
            "{}: o_proj and down_proj",
            path.display()
        );

        // Every activation of every block is present, and every suffix mapped.
        for layer in 0..layers {
            for act in Act::ALL {
                assert!(
                    sites.iter().any(|s| s.layer == layer && s.act == act),
                    "{}: {act:?} of layer {layer} missing",
                    path.display()
                );
            }
        }
        for fm in &m {
            let (_, suffix) = llvq_artifact::split_name(&fm.name).expect("name");
            assert!(
                act_of_suffix(&suffix).is_some(),
                "{}: unknown suffix {suffix}",
                path.display()
            );
        }

        println!(
            "{}: {} matrices → {} sites, histogram {:?}",
            path.display(),
            m.len(),
            sites.len(),
            hist
        );
    }
}
