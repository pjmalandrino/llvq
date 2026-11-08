//! The served config file: what it accepts, and above all what it refuses.
//!
//! Every assertion here is about a way a config could be wrong and still look
//! right. A file that silently means `planes14` is worse than no file: it
//! would make a run claim a provenance it does not have.
//!
//! Portable on purpose. None of this needs a card, and the defect class it
//! covers — a typo reading as a default — is exactly the one that costs a
//! billed job to discover.

use llvq_llm::fused::{EmbedMode, FuseMode, FusedLayout};
use llvq_llm::kvq::{AttnHeads, KvMode, KvStore, GROW_CHUNK};
use llvq_llm::rotplan::RotShare;
use llvq_llm::served::{
    check_env, check_persistent_object, check_removed, Decode, Rows4e, Served, ServedFile, BELOW_THE_DOOR,
    MEASUREMENT_MODES, REMOVED,
};
use std::path::PathBuf;

fn served() -> ServedFile {
    ServedFile {
        layout: "tetra48".into(),
        embed: "q8".into(),
        rot_share: "1".into(),
        fuse: "0".into(),
        kv: "f16".into(),
        attention: None,
        decode: None,
        rows: None,
        note: Some("Qwen3-4B, Tetra + 36 v_proj int4 g128".into()),
    }
}

fn of(f: ServedFile) -> Result<Served, String> {
    Served::of(f, PathBuf::from("test.json"))
}

/// The served object of 2026-09-08, read back as the five resolved values.
#[test]
fn the_served_config_resolves_to_the_served_values() {
    let s = of(served()).expect("the served config parses");
    assert_eq!(s.layout, FusedLayout::Tetra48);
    assert_eq!(s.embed, EmbedMode::Q8);
    assert_eq!(s.rot_share, RotShare::On);
    assert_eq!(s.fuse, FuseMode::Off);
    assert_eq!(s.kv, KvMode::F16);
    assert!(s.provenance().contains("tetra48"), "{}", s.provenance());
}

/// 🚨 An empty field must NOT read as the default.
///
/// Every `parse` in this workspace maps `Some("")` to its default — that is
/// right for an unset environment variable and wrong for a file whose whole
/// reason to exist is that it has no defaults. `"layout": ""` would resolve to
/// `planes14` and a run would report a layout nobody wrote.
#[test]
fn an_empty_field_is_refused_and_not_read_as_the_default() {
    for (name, mutate) in [
        ("layout", (|f: &mut ServedFile| f.layout = String::new()) as fn(&mut ServedFile)),
        ("embed", |f| f.embed = String::new()),
        ("rot_share", |f| f.rot_share = String::new()),
        ("fuse", |f| f.fuse = String::new()),
        ("kv", |f| f.kv = String::new()),
    ] {
        let mut f = served();
        mutate(&mut f);
        let e = match of(f) {
            Err(e) => e,
            Ok(_) => panic!("an empty {name} must be refused"),
        };
        assert!(e.contains(name), "the message must name the field: {e}");
    }

    // And whitespace is empty: "  " trimmed is "", and `parse` would never see
    // it as a value either.
    let mut f = served();
    f.layout = "   ".into();
    assert!(of(f).is_err(), "a whitespace field is an empty field");
}

/// An unknown key is a typo, not a default. `deny_unknown_fields` is what makes
/// `"embeding": "q8"` an error instead of a silent f16 embedding.
#[test]
fn an_unknown_key_is_refused() {
    let text = r#"{"layout":"tetra48","embed":"q8","rot_share":"1","fuse":"0",
                   "kv":"f16","embeding":"q8"}"#;
    let e = serde_json::from_str::<ServedFile>(text).expect_err("must be refused");
    assert!(format!("{e}").contains("embeding"), "{e}");
}

/// A missing key is refused too, and for the same reason: there is no value
/// here to fall back to.
#[test]
fn a_missing_key_is_refused() {
    for drop in ["layout", "embed", "rot_share", "fuse", "kv"] {
        let mut o = serde_json::Map::new();
        for (k, v) in [
            ("layout", "tetra48"),
            ("embed", "q8"),
            ("rot_share", "1"),
            ("fuse", "0"),
            ("kv", "f16"),
        ] {
            if k != drop {
                o.insert(k.into(), serde_json::Value::String(v.into()));
            }
        }
        let text = serde_json::to_string(&serde_json::Value::Object(o)).expect("json");
        let e = match serde_json::from_str::<ServedFile>(&text) {
            Err(e) => e,
            Ok(_) => panic!("a config missing {drop} must be refused"),
        };
        assert!(format!("{e}").contains(drop), "the message must name {drop}: {e}");
    }
}

/// The vocabulary is the environment variables', not a second one.
#[test]
fn the_file_accepts_exactly_what_the_variables_accept() {
    let mut f = served();
    f.layout = "Tetra48".into();
    let e = of(f).expect_err("the parsers are case-sensitive and so is the file");
    assert!(e.contains("Tetra48"), "{e}");

    let mut f = served();
    f.rot_share = "on".into();
    assert!(of(f).is_err(), "`RotShare::parse` refuses \"on\", so the file must");

    // And every layout the variable takes, the file takes.
    for (name, want) in [
        ("planes14", FusedLayout::Planes14),
        ("planes12x", FusedLayout::Planes12x),
        ("slot32", FusedLayout::Slot32),
        ("golay70", FusedLayout::Golay70),
        ("tetra48", FusedLayout::Tetra48),
    ] {
        let mut f = served();
        f.layout = name.into();
        assert_eq!(of(f).expect("a known layout").layout, want);
    }
}

/// The file this repository ships IS the object arbitrated on 2026-09-08.
///
/// Not a restatement of the struct: the shipped file is read from disk, by
/// the code a runner runs, and compared against the five values `docs/ETAT.md`
/// attributes 56.95 MMLU and 2.8138 b/param to. A config that drifted from the
/// dossier would make every run under it cite a provenance it does not have.
#[test]
fn the_shipped_config_is_the_served_object() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../configs/qwen3-4b-tetra-q5.json");
    // `read_file` + `of`, not `read`: `read` consults the shell, and a test
    // whose verdict depends on what the developer exported is not a test of
    // the file. Found the hard way — this line failed on any shell with
    // `LLVQ_FUSED_LAYOUT` set, and passed on a clean one.
    let f = Served::read_file(&path).expect("the shipped config parses");
    let s = Served::of(f, path.clone()).expect("the shipped config resolves");
    assert_eq!(s.layout, FusedLayout::Tetra48);
    assert_eq!(s.embed, EmbedMode::Q8);
    assert_eq!(s.rot_share, RotShare::On);
    assert_eq!(s.kv, KvMode::F16);
    // `fuse` is 0 because Tetra48 carries no segmented kernel — a fact about
    // the layout, asserted here against the layout rather than against the
    // file, so the two cannot drift apart in silence.
    assert_eq!(s.fuse, FuseMode::Off);
    assert!(
        !llvq_llm::fused::planes_source_names(s.layout).contains(&"tv_planes_seg_h.cu"),
        "Tetra48 gained a segmented kernel: the shipped `fuse` value is now a choice, \
         not a fact, and this config has to be revisited"
    );
}

/// The 8B config is the 4B's five values on another object, and nothing more.
///
/// No field of a served config is model-specific: the five name a layout, an
/// embedding mode, a hoist, a fusion mode and a KV store, and the loader reads
/// every shape from the sealed file (`fused_cuda::load_resolved` parses the
/// file's own `config.json`). What differs is the note, and the path the run
/// records on its `# config=` line — which is why the 8B gets a file of its
/// own rather than borrowing the 4B's, whose path would then name the wrong
/// object in every 8B dump header.
///
/// Mutation: change any one of the 8B file's five values and this fails; the
/// 4B file is read beside it so the two cannot drift apart in silence.
#[test]
fn the_8b_config_is_the_4b_served_values_on_the_8b_object() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let read = |name: &str| {
        let path = dir.join(name);
        let f = Served::read_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        (f.clone(), Served::of(f, path).unwrap_or_else(|e| panic!("{name}: {e}")))
    };
    let (f4, s4) = read("qwen3-4b-tetra-q5.json");
    let (f8, s8) = read("qwen3-8b-tetra-q5.json");
    assert_eq!(
        (s8.layout, s8.embed, s8.rot_share, s8.fuse, s8.kv),
        (s4.layout, s4.embed, s4.rot_share, s4.fuse, s4.kv),
        "the 8B serves the 4B's recipe; a value that moved is a new object, not this file"
    );
    let note = f8.note.expect("the 8B config says which object it belongs to");
    assert!(note.contains("Qwen3-8B"), "{note}");
    assert!(note.contains("qwen3-8b-dclm"), "{note}");
    assert_ne!(Some(note), f4.note, "a copied note would print the 4B as this run's object");
}

/// The 14B config is the 4B's five values on the 14B object, and nothing more.
///
/// Same argument as the 8B test above: the five values name runtime choices,
/// the loader reads every shape from the sealed file, and the file exists for
/// its note and its path, both of which a run prints. Qwen3-14B has 40 layers,
/// so 40 `v_proj` int4 g128 records where the 4B and the 8B have 36; the note
/// says so, because a note copied from the 8B would state the wrong count on
/// the provenance line of every 14B run.
///
/// Mutation: change any one of the 14B file's five values and this fails; the
/// 4B and 8B files are read beside it so none of the three can drift from the
/// others in silence.
#[test]
fn the_14b_config_is_the_4b_served_values_on_the_14b_object() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let read = |name: &str| {
        let path = dir.join(name);
        let f = Served::read_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        (f.clone(), Served::of(f, path).unwrap_or_else(|e| panic!("{name}: {e}")))
    };
    let (f4, s4) = read("qwen3-4b-tetra-q5.json");
    let (f8, _) = read("qwen3-8b-tetra-q5.json");
    let (f14, s14) = read("qwen3-14b-tetra-q5.json");
    assert_eq!(
        (s14.layout, s14.embed, s14.rot_share, s14.fuse, s14.kv),
        (s4.layout, s4.embed, s4.rot_share, s4.fuse, s4.kv),
        "the 14B serves the 4B's recipe; a value that moved is a new object, not this file"
    );
    let note = f14.note.expect("the 14B config says which object it belongs to");
    assert!(note.contains("Qwen3-14B"), "{note}");
    assert!(note.contains("qwen3-14b-dclm"), "{note}");
    assert!(note.contains("40 v_proj int4 g128"), "{note}");
    assert!(note.contains("Untied head"), "{note}");
    assert_ne!(Some(&note), f4.note.as_ref(), "a copied note would print the 4B as this run's object");
    assert_ne!(Some(&note), f8.note.as_ref(), "a copied note would print the 8B as this run's object");
}

/// Every file in `configs/` resolves. A config added without a test of its own
/// would otherwise reach a card unread: `ops/Dockerfile.cuda` copies the whole
/// directory and checks the names it lists with `test -f`, which proves
/// presence, not parsing.
#[test]
fn every_shipped_config_resolves() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let mut n = 0;
    for entry in std::fs::read_dir(&dir).expect("configs/") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let f = Served::read_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Served::of(f, path.clone()).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        n += 1;
    }
    assert!(n >= 3, "{n} configs found in {}", dir.display());
}

/// A lookup that answers from a fixed table — the environment, without the
/// environment. What makes the contradiction branch testable at all.
fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |v| pairs.iter().find(|(k, _)| *k == v).map(|(_, val)| val.to_string())
}

/// A variable that contradicts the file is refused BY NAME, and one that
/// agrees passes. Not a precedence rule in either direction.
#[test]
fn a_contradicting_variable_is_refused_and_an_agreeing_one_passes() {
    let f = served();
    check_env(&f, env_of(&[])).expect("no variables, nothing to contradict");
    check_env(&f, env_of(&[("LLVQ_FUSED_LAYOUT", "tetra48"), ("LLVQ_EMBED", "q8")]))
        .expect("agreeing variables pass");

    for (var, wrong) in [
        ("LLVQ_FUSED_LAYOUT", "planes14"),
        ("LLVQ_EMBED", "f16"),
        ("LLVQ_ROT_SHARE", "0"),
        ("LLVQ_FUSE", "1"),
        ("LLVQ_KV", "q8"),
    ] {
        let e = match check_env(&f, env_of(&[(var, wrong)])) {
            Err(e) => e,
            Ok(()) => panic!("{var}={wrong} must contradict the served file"),
        };
        assert!(e.contains(var), "the refusal must name the variable: {e}");
        assert!(e.contains(wrong), "and its value: {e}");
    }
}

/// 🚨 The comparison is of MEANINGS, through the parsers, not of strings.
///
/// `LLVQ_FUSE=""` is how a shell says "unset but exported", and `FuseMode::parse`
/// reads it as `Off` — the same thing `"fuse": "0"` says. A string comparison
/// would refuse a run whose two answers agree.
#[test]
fn the_comparison_goes_through_the_parsers() {
    let f = served();
    check_env(&f, env_of(&[("LLVQ_FUSE", "")])).expect("\"\" and \"0\" both mean Off");
    check_env(&f, env_of(&[("LLVQ_ROT_SHARE", "1")])).expect("same spelling, same meaning");
    // And a variable that does not parse is its own error, naming itself.
    let e = check_env(&f, env_of(&[("LLVQ_EMBED", "int8")])).expect_err("not a mode");
    assert!(e.contains("LLVQ_EMBED"), "{e}");
}

/// The variables that reach below the served door are refused by name,
/// whatever their value, and the card variable is not.
#[test]
fn variables_below_the_door_are_refused_by_name() {
    let f = served();
    assert_eq!(
        BELOW_THE_DOOR,
        ["LLVQ_KERNEL_DIR", "LLVQ_TILE_BLOCKS", "LLVQ_PREFILL", "LLVQ_METAL_KERNEL"]
    );
    for var in BELOW_THE_DOOR {
        let e = match check_env(&f, env_of(&[(var, "anything")])) {
            Err(e) => e,
            Ok(()) => panic!("{var} beside LLVQ_CONFIG must be refused"),
        };
        assert!(e.contains(var), "{e}");
    }
    // `LLVQ_NVRTC_ARCH` names the card, not the object: an A100 needs it.
    check_env(&f, env_of(&[("LLVQ_NVRTC_ARCH", "compute_80")]))
        .expect("the compile target is the card's business, not the config's");
}

/// `Served::of` reads no environment: the same file resolves the same way
/// whatever `check_env` would have said about the shell.
///
/// Pinned because it was false until 2026-09-11 — `of` called the environment
/// check itself, and four of these six tests failed on any shell that
/// exported `LLVQ_FUSED_LAYOUT`.
#[test]
fn resolving_the_file_does_not_read_the_environment() {
    // A contradiction that `check_env` refuses…
    let f = served();
    assert!(check_env(&f, env_of(&[("LLVQ_FUSED_LAYOUT", "planes14")])).is_err());

    // …is invisible to `of`, which sees only the file. The contradiction is
    // put in the REAL environment of this process for the duration of the
    // call: a mutant that made `of` consult the shell survived this test on
    // a clean shell, because a clean shell has nothing to contradict. No other
    // test in this file reads the real environment — they all hand `check_env`
    // a closure — so the variable is nobody else's business.
    std::env::set_var("LLVQ_FUSED_LAYOUT", "planes14");
    let r = of(served());
    std::env::remove_var("LLVQ_FUSED_LAYOUT");
    let s = r.expect("of() is pure: the shell says planes14, the file says tetra48, of() reads the file");
    assert_eq!(s.layout, FusedLayout::Tetra48);
}

/// The sealed paper-2 object carries its embedding int4 g64, so its config
/// names `q4`, and every other value is the 4B served one. `q4` goes through
/// `EmbedMode::parse` like every other value; the kernels it selects are
/// pinned by `tests/embed_q4.rs` and `llvq-metal/tests/q4e_matches_host.rs`.
#[test]
fn the_q4_config_is_the_4b_served_values_with_a_q4_embedding() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let read = |name: &str| {
        let path = dir.join(name);
        let f = Served::read_file(&path).expect("parses");
        Served::of(f, path).expect("resolves")
    };
    let e4 = read("qwen3-4b-tetra-e4.json");
    let q5 = read("qwen3-4b-tetra-q5.json");
    assert_eq!(e4.embed, EmbedMode::Q4);
    assert_eq!(
        (e4.layout, e4.rot_share, e4.fuse, e4.kv),
        (q5.layout, q5.rot_share, q5.fuse, q5.kv),
        "only the embedding may differ from the 4B served object"
    );
}

// ---------------------------------------------------------------------------
// The `attention` field, added 2026-11-02 with the grouped form.
// ---------------------------------------------------------------------------

/// 🚨 A config written before the field existed must serve what it served.
///
/// This is the whole reason the field is optional and the reason the operator
/// chose a field over an engine default: `expanded` is the form every
/// published number ran, so the six files that predate 2026-11-02 keep their
/// logits, and only a file that SAYS `grouped` changes path.
#[test]
fn an_absent_attention_field_is_the_published_form() {
    // In the struct.
    let s = of(served()).expect("the field is optional");
    assert_eq!(s.attention, AttnHeads::Expanded);
    assert_eq!(s.store, KvStore::Cat);

    // In a file that does not carry the key at all.
    let text = r#"{"layout":"tetra48","embed":"q4","rot_share":"1","fuse":"0","kv":"f16"}"#;
    let f: ServedFile = serde_json::from_str(text).expect("five keys are enough");
    assert_eq!(f.attention, None);
    let r = Served::of(f, PathBuf::from("old.json")).expect("resolves");
    assert_eq!((r.attention, r.store), (AttnHeads::Expanded, KvStore::Cat));

    // And on disk: every shipped config except the grouped, kernel and 4E ones
    // is expanded; every one but the 4E configs decodes on the eager path.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let mut expanded = 0;
    for entry in std::fs::read_dir(&dir).expect("configs/") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let name = path.file_name().expect("name").to_string_lossy().to_string();
        let f = Served::read_file(&path).expect("parses");
        let r = Served::of(f, path.clone()).expect("resolves");
        let four_e = name.contains("-4e");
        assert_eq!(r.decode, if four_e { Decode::Persistent } else { Decode::Eager }, "{name}");
        match (name.contains("grouped"), name.contains("kernel") || four_e) {
            (true, _) => assert_eq!(r.attention, AttnHeads::Grouped, "{name}"),
            (false, true) => assert_eq!(r.attention, AttnHeads::Kernel, "{name}"),
            (false, false) => {
                assert_eq!(r.attention, AttnHeads::Expanded, "{name} must not have moved");
                assert_eq!(r.store, KvStore::Cat, "{name}");
                expanded += 1;
            }
        }
    }
    assert!(expanded >= 6, "{expanded} configs predate the field, 6 expected");
}

/// The grouped config is the e4 object and nothing else about it moved.
///
/// It names the SAME file, the same layout, embedding, hoist, fusion and KV
/// mode: the only difference between the two is how attention reads the cache,
/// which is what makes a pair of runs under them an A/B of one thing.
#[test]
fn the_grouped_config_is_the_e4_object_with_the_grouped_attention() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let read = |name: &str| {
        let path = dir.join(name);
        let f = Served::read_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        Served::of(f, path).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let e4 = read("qwen3-4b-tetra-e4.json");
    let g = read("qwen3-4b-tetra-e4-grouped.json");

    assert_eq!(g.attention, AttnHeads::Grouped);
    assert_eq!(g.store, KvStore::Grow(GROW_CHUNK), "the store rides the form");
    assert_eq!(e4.attention, AttnHeads::Expanded, "the published 4B config did not move");
    assert_eq!(
        (g.layout, g.embed, g.rot_share, g.fuse, g.kv),
        (e4.layout, e4.embed, e4.rot_share, e4.fuse, e4.kv),
        "only the attention form may differ from the e4 object"
    );
    // The provenance names the form, so no log can be silent about which one
    // produced its number.
    let line = g.provenance();
    assert!(line.contains("attention grouped on grow(256)"), "{line}");
    assert!(e4.provenance().contains("attention expanded on cat"), "{}", e4.provenance());
}

/// The 4B kernel config (operator T8, 2026-11-05) is the grouped config with
/// the kernel form, and nothing else about it moved.
///
/// Below position 1,280 the kernel form runs the grouped arithmetic bit for
/// bit, so the two files must agree on everything but the field.
#[test]
fn the_kernel_config_is_the_grouped_config_with_the_kernel_form() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let read = |name: &str| {
        let path = dir.join(name);
        let f = Served::read_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        Served::of(f, path).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let g = read("qwen3-4b-tetra-e4-grouped.json");
    let k = read("qwen3-4b-tetra-e4-kernel-from1280.json");

    assert_eq!(k.attention, AttnHeads::Kernel);
    assert_eq!(k.store, g.store, "the kernel form reads the grouped store");
    assert_eq!(
        (k.layout, k.embed, k.rot_share, k.fuse, k.kv),
        (g.layout, g.embed, g.rot_share, g.fuse, g.kv),
        "only the attention form may differ from the grouped config"
    );
    assert!(k.provenance().contains("attention kernel from position 1280"), "{}", k.provenance());
}

/// An `attention` nobody can serve is refused, and never read as the default.
#[test]
fn an_attention_the_file_cannot_serve_is_refused() {
    // Empty, and whitespace, like every other field of this file.
    for empty in ["", "   "] {
        let mut f = served();
        f.attention = Some(empty.into());
        let e = match of(f) {
            Err(e) => e,
            Ok(_) => panic!("an empty attention must be refused, not read as expanded"),
        };
        assert!(e.contains("attention"), "the message must name the field: {e}");
        assert!(e.contains("empty"), "and say what is wrong with it: {e}");
    }
    // `shared` is refused BY NAME: it is slower than grouped everywhere and
    // not bit for bit either, so no object is served under it.
    let mut f = served();
    f.attention = Some("shared".into());
    let e = of(f).expect_err("shared is not a served form");
    assert!(e.contains("gqa-ab-2026-11-01"), "the refusal cites its journal: {e}");
    // And a typo stays a typo.
    for bad in ["Grouped", "gqa", "grouped ", "unexpanded"] {
        let mut f = served();
        f.attention = Some(bad.into());
        assert!(of(f).is_err(), "{bad:?} must be refused");
    }
}

/// 🚨 The grouped form has never run on Metal, so a Metal build refuses it.
///
/// The form is backend-independent code in `model.rs`, which is exactly what
/// makes a silent acceptance easy: the Mac would serve a form whose tolerance
/// gate was passed on an L40S, with no throughput, no perplexity and no
/// reading of Metal's own rounding. The predicate is handed in, so this runs
/// on a machine with no Metal device.
#[test]
fn the_grouped_form_is_refused_on_metal_and_named_in_the_refusal() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let path = dir.join("qwen3-4b-tetra-e4-grouped.json");
    let g = Served::of(Served::read_file(&path).expect("parses"), path).expect("resolves");

    let e = g.check_device(true).expect_err("grouped on Metal is refused");
    assert!(e.contains("Metal"), "{e}");
    assert!(e.contains("grouped"), "the message names the form: {e}");
    assert!(e.contains("gqa-ppl-2026-11-01"), "and the measurement that would lift it: {e}");
    assert!(
        e.contains("qwen3-4b-tetra-e4-grouped.json"),
        "and which config asked for it: {e}"
    );
    g.check_device(false).expect("grouped on a card is what was measured");

    // The expanded form is the Mac's served path and stays accepted.
    let s = of(served()).expect("resolves");
    s.check_device(true).expect("expanded on Metal is the published path");
    s.check_device(false).expect("expanded on a card too");
}


// ---- the decode engine (2026-11-05) ----------------------------------------

fn four_e() -> ServedFile {
    ServedFile {
        layout: "tetra48".into(),
        embed: "q4".into(),
        rot_share: "1".into(),
        fuse: "0".into(),
        kv: "f16".into(),
        attention: Some("kernel".into()),
        decode: Some("persistent".into()),
        rows: None,
        note: None,
    }
}

/// The 4E config is the kernel config with the persistent decode, and nothing
/// else about the object moved; its threshold is 0 and its line says so.
#[test]
fn the_4e_config_is_the_kernel_config_with_the_persistent_decode() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let read = |name: &str| {
        let path = dir.join(name);
        let f = Served::read_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        Served::of(f, path).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let k = read("qwen3-4b-tetra-e4-kernel-from1280.json");
    let p = read("qwen3-4b-tetra-e4-4e.json");
    assert_eq!(p.decode, Decode::Persistent);
    assert_eq!(k.decode, Decode::Eager, "the served config did not move");
    assert_eq!(
        (p.layout, p.embed, p.rot_share, p.fuse, p.kv, p.attention, p.store),
        (k.layout, k.embed, k.rot_share, k.fuse, k.kv, k.attention, k.store),
        "only the decode engine may differ from the kernel config"
    );
    assert_eq!(p.decode.kernel_from(), 0);
    assert_eq!(k.decode.kernel_from(), llvq_llm::kvq::KERNEL_FROM);
    let line = p.provenance();
    assert!(line.contains("attention kernel from position 0 on grow(256), decode persistent"), "{line}");
    assert!(!k.provenance().contains("decode persistent"), "an eager config's line reads as before: {}", k.provenance());
    let note = p.note.expect("a note");
    assert!(note.contains("NOT SERVED YET") && note.contains("886391a8"), "{note}");
}

/// The field: absent and "eager" are the eager path, "persistent" is 4E, and
/// the empty string or any other word is refused by name.
#[test]
fn the_decode_field_accepts_two_engines_and_refuses_the_rest() {
    let mut f = four_e();
    assert_eq!(of(f.clone()).expect("4E").decode, Decode::Persistent);
    f.decode = None;
    assert_eq!(of(f.clone()).expect("absent").decode, Decode::Eager);
    f.decode = Some("eager".into());
    assert_eq!(of(f.clone()).expect("eager").decode, Decode::Eager);
    for bad in ["", "  ", "Persistent", "4e", "graph"] {
        f.decode = Some(bad.into());
        let e = of(f.clone()).expect_err(bad);
        assert!(e.contains("decode"), "{bad:?}: {e}");
    }
    // In a file, the key is optional and a typo of it is an unknown key.
    let text = r#"{"layout":"tetra48","embed":"q4","rot_share":"1","fuse":"0","kv":"f16","attention":"kernel","decod":"persistent"}"#;
    assert!(serde_json::from_str::<ServedFile>(text).is_err(), "a typo must not read as the eager default");
}

/// 4E computes the kernel form and reads the served 4B's fields: any other
/// attention or field beside the persistent decode is refused by name.
#[test]
fn the_persistent_decode_is_refused_off_the_served_4b_fields() {
    for attention in [None, Some("expanded"), Some("grouped")] {
        let mut f = four_e();
        f.attention = attention.map(str::to_string);
        let e = of(f).expect_err("not the kernel form");
        assert!(e.contains("persistent") && e.contains("kernel"), "{attention:?}: {e}");
    }
    type Edit = (&'static str, fn(&mut ServedFile));
    let edits: [Edit; 5] = [
        ("layout", |f| f.layout = "planes14".into()),
        ("embed", |f| f.embed = "q8".into()),
        ("rot_share", |f| f.rot_share = "0".into()),
        ("kv", |f| f.kv = "q8".into()),
        ("fuse", |f| {
            f.fuse = "1".into();
            f.layout = "planes14".into();
        }),
    ];
    for (what, edit) in edits {
        let mut f = four_e();
        edit(&mut f);
        let e = of(f).expect_err(what);
        assert!(e.contains("served 4B's fields"), "{what}: {e}");
    }
}

/// Beside the persistent decode, every measurement mode is refused by name;
/// the two prompt-length variables and the card variable are not.
#[test]
fn measurement_modes_are_refused_beside_the_persistent_decode() {
    let f = four_e();
    for var in MEASUREMENT_MODES {
        let e = check_env(&f, env_of(&[(var, "1")])).expect_err(var);
        assert!(e.contains(var) && e.contains("measurement mode"), "{e}");
    }
    for var in ["LLVQ_CONTEXTS", "LLVQ_PREFILL_TOKENS", "LLVQ_NVRTC_ARCH"] {
        check_env(&f, env_of(&[(var, "5,8192")])).unwrap_or_else(|e| panic!("{var}: {e}"));
    }
    // An eager config keeps what it accepted before the field existed.
    check_env(&served(), env_of(&[("LLVQ_TIME_EVENTS", "1")])).expect("not this check's business on an eager config");
    assert!(MEASUREMENT_MODES.contains(&"LLVQ_PERS_4E") && MEASUREMENT_MODES.contains(&"LLVQ_GQA_PPL"));
}

/// Refused on Metal by name, before the kernel form's own refusal.
#[test]
fn the_persistent_decode_is_refused_on_metal() {
    let p = of(four_e()).expect("4E");
    let e = p.check_device(true).expect_err("Metal");
    assert!(e.contains("persistent") && e.contains("Metal"), "{e}");
    p.check_device(false).expect("CUDA");
}

/// On the sealed shapes only: the 4B, the 8B, the 14B and the 0.6B drafter;
/// any other shape (here the 1.7B's, and a 14B with one layer less) is refused
/// by name.
#[test]
fn the_persistent_decode_runs_on_the_sealed_shapes_only() {
    let cfg = |layers, hidden, inter, heads, kv| {
        let text = format!(
            r#"{{"vocab_size":151936,"hidden_size":{hidden},"intermediate_size":{inter},"num_hidden_layers":{layers},
            "num_attention_heads":{heads},"head_dim":128,"attention_bias":false,"num_key_value_heads":{kv},
            "max_position_embeddings":40960,"sliding_window":null,"max_window_layers":36,"tie_word_embeddings":true,
            "rope_theta":1000000.0,"rms_norm_eps":1e-6,"use_sliding_window":false,"hidden_act":"silu"}}"#
        );
        serde_json::from_str::<candle_transformers::models::qwen3::Config>(&text).expect("a qwen3 config")
    };
    check_persistent_object(&cfg(36, 2560, 9728, 32, 8)).expect("the 4B");
    check_persistent_object(&cfg(36, 4096, 12288, 32, 8)).expect("the 8B");
    check_persistent_object(&cfg(40, 5120, 17408, 40, 8)).expect("the 14B");
    check_persistent_object(&cfg(28, 1024, 3072, 16, 8)).expect("the 0.6B drafter");
    for (name, c) in [("1.7B", cfg(28, 2048, 6144, 16, 8)), ("14B less a layer", cfg(39, 5120, 17408, 40, 8))] {
        let e = check_persistent_object(&c).expect_err(name);
        assert!(e.contains("persistent") && e.contains("sealed Qwen3 shapes"), "{name}: {e}");
    }
}

// ---- 4E's row loop (2026-11-05) ---------------------------------------------

/// The v4 config is the 4E config with the v4 row loop, and nothing else moved;
/// its line names the loop, and the v3 config's line reads as it did.
#[test]
fn the_v4_config_is_the_4e_config_with_the_v4_rows() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let read = |name: &str| {
        let path = dir.join(name);
        let f = Served::read_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        Served::of(f, path).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let v3 = read("qwen3-4b-tetra-e4-4e.json");
    let v4 = read("qwen3-4b-tetra-e4-4e-v4.json");
    assert_eq!(v3.rows, Rows4e::V3);
    assert_eq!(v4.rows, Rows4e::V4);
    assert_eq!(v4.rows.form(), 2);
    assert_eq!(v3.rows.form(), 1);
    assert_eq!(
        (v4.layout, v4.embed, v4.rot_share, v4.fuse, v4.kv, v4.attention, v4.store, v4.decode),
        (v3.layout, v3.embed, v3.rot_share, v3.fuse, v3.kv, v3.attention, v3.store, v3.decode),
        "only the row loop may differ from the 4E config"
    );
    assert!(v4.provenance().contains("decode persistent (4E rows v4, one launch a token run)"), "{}", v4.provenance());
    assert!(v3.provenance().contains("decode persistent (4E, one launch a token run)"), "{}", v3.provenance());
    let note = v4.note.expect("a note");
    assert!(note.contains("SERVED from 2026-11-05") && note.contains("886391a8"), "{note}");
}

/// The v6 config is the v4 config with the v6 row loop, and nothing else
/// moved; its line names the loop, and it resolves to the module's form 4.
#[test]
fn the_v6_config_is_the_v4_config_with_the_v6_rows() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let read = |name: &str| {
        let path = dir.join(name);
        let f = Served::read_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        Served::of(f, path).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let v4 = read("qwen3-4b-tetra-e4-4e-v4.json");
    let v6 = read("qwen3-4b-tetra-e4-4e-v6.json");
    assert_eq!(v6.rows, Rows4e::V6);
    assert_eq!(v6.rows.form(), 4);
    assert_eq!(
        (v6.layout, v6.embed, v6.rot_share, v6.fuse, v6.kv, v6.attention, v6.store, v6.decode),
        (v4.layout, v4.embed, v4.rot_share, v4.fuse, v4.kv, v4.attention, v4.store, v4.decode),
        "only the row loop may differ from the v4 config"
    );
    assert!(
        v6.provenance().contains("attention kernel from position 0 on grow(256), decode persistent (4E rows v6, one launch a token run)"),
        "{}",
        v6.provenance()
    );
    let note = v6.note.as_deref().expect("a note");
    assert!(note.contains("886391a8") && note.contains("row form 4"), "{note}");
    // The files themselves: the same keys in the same order, one line apart
    // besides the free-text note.
    let raw = |name: &str| -> Vec<String> {
        std::fs::read_to_string(dir.join(name))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .lines()
            .filter(|l| !l.trim_start().starts_with("\"note\""))
            .map(str::to_string)
            .collect()
    };
    let (a, b) = (raw("qwen3-4b-tetra-e4-4e-v4.json"), raw("qwen3-4b-tetra-e4-4e-v6.json"));
    let differ: Vec<_> = a.iter().zip(&b).filter(|(x, y)| x != y).collect();
    assert_eq!(a.len(), b.len(), "the same keys");
    assert_eq!(differ, vec![(&"  \"rows\": \"v4\",".to_string(), &"  \"rows\": \"v6\",".to_string())]);
    // The 4E install refuses Metal and every object but the 4B, as under v4.
    let e = v6.check_device(true).expect_err("Metal");
    assert!(e.contains("persistent") && e.contains("Metal"), "{e}");
}

/// The v9 config is the v6 config with the v9 row loop, and nothing else
/// moved; its line names the loop, it resolves to the module's form 7, and
/// it is the one config that lays the int4 words tiled on the card.
#[test]
fn the_v9_config_is_the_v6_config_with_the_v9_rows() {
    use llvq_llm::fused::Int4Layout;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let read = |name: &str| {
        let path = dir.join(name);
        let f = Served::read_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        Served::of(f, path).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let v6 = read("qwen3-4b-tetra-e4-4e-v6.json");
    let v9 = read("qwen3-4b-tetra-e4-4e-v9.json");
    assert_eq!(v9.rows, Rows4e::V9);
    assert_eq!(v9.rows.form(), 7);
    assert_eq!(
        (v9.layout, v9.embed, v9.rot_share, v9.fuse, v9.kv, v9.attention, v9.store, v9.decode),
        (v6.layout, v6.embed, v6.rot_share, v6.fuse, v6.kv, v6.attention, v6.store, v6.decode),
        "only the row loop may differ from the v6 config"
    );
    assert!(
        v9.provenance().contains("attention kernel from position 0 on grow(256), decode persistent (4E rows v9, one launch a token run)"),
        "{}",
        v9.provenance()
    );
    // The int4 words: tiled and held once under v9, the file's order under
    // every other config, so no config reads words in the wrong order.
    assert_eq!(v9.int4_layout(), Int4Layout::Tiled);
    for name in [
        "qwen3-4b-tetra-e4-4e.json",
        "qwen3-4b-tetra-e4-4e-v4.json",
        "qwen3-4b-tetra-e4-4e-v6.json",
        "qwen3-4b-tetra-e4-grouped.json",
        "qwen3-8b-tetra-e4-grouped.json",
        "qwen3-14b-tetra-e4-grouped.json",
    ] {
        assert_eq!(read(name).int4_layout(), Int4Layout::Served, "{name}");
    }
    for r in [Rows4e::V3, Rows4e::V4, Rows4e::V6] {
        assert_eq!(r.int4_layout(), Int4Layout::Served, "{r:?}");
    }
    let note = v9.note.as_deref().expect("a note");
    assert!(note.contains("886391a8") && note.contains("row form 7") && note.contains("once"), "{note}");
    let raw = |name: &str| -> Vec<String> {
        std::fs::read_to_string(dir.join(name))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .lines()
            .filter(|l| !l.trim_start().starts_with("\"note\""))
            .map(str::to_string)
            .collect()
    };
    let (a, b) = (raw("qwen3-4b-tetra-e4-4e-v6.json"), raw("qwen3-4b-tetra-e4-4e-v9.json"));
    let differ: Vec<_> = a.iter().zip(&b).filter(|(x, y)| x != y).collect();
    assert_eq!(a.len(), b.len(), "the same keys");
    assert_eq!(differ, vec![(&"  \"rows\": \"v6\",".to_string(), &"  \"rows\": \"v9\",".to_string())]);
    let e = v9.check_device(true).expect_err("Metal");
    assert!(e.contains("persistent") && e.contains("Metal"), "{e}");
}

/// The 8B and 14B v9 configs carry the 4B v9 config's keys, line for line
/// beside the note; each names its own object and resolves to v9 with the
/// int4 words tiled.
#[test]
fn the_8b_and_14b_v9_configs_are_the_4b_v9_config_on_their_objects() {
    use llvq_llm::fused::Int4Layout;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let raw = |name: &str| -> Vec<String> {
        std::fs::read_to_string(dir.join(name))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .lines()
            .filter(|l| !l.trim_start().starts_with("\"note\""))
            .map(str::to_string)
            .collect()
    };
    let four = raw("qwen3-4b-tetra-e4-4e-v9.json");
    for (name, object) in [("qwen3-8b-tetra-e4-4e-v9.json", "7bdb9a55"), ("qwen3-14b-tetra-e4-4e-v9.json", "61db37fe")] {
        assert_eq!(raw(name), four, "{name}: only the note may differ from the 4B v9 config");
        let path = dir.join(name);
        let c = Served::of(Served::read_file(&path).unwrap(), path).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((c.rows, c.decode, c.int4_layout()), (Rows4e::V9, Decode::Persistent, Int4Layout::Tiled), "{name}");
        let note = c.note.as_deref().expect("a note");
        assert!(note.contains(object) && note.contains("untied"), "{name}: {note}");
    }
}

/// Every served row loop is a row form of the module, under the name the
/// module gives it: a `form()` that pointed at v5's number would compile
/// v5's loop under v6's provenance. It names `llvq_cuda`, a dev-dependency on
/// macOS only, so it carries that gate (a Linux CPU build has no such crate).
#[cfg(any(target_os = "macos", feature = "cuda"))]
#[test]
fn the_served_rows_are_the_modules_row_forms() {
    use llvq_cuda::pers::P4E_ROWS_FORMS;
    for r in [Rows4e::V3, Rows4e::V4, Rows4e::V6, Rows4e::V9] {
        assert!(P4E_ROWS_FORMS.contains(&(r.form(), r.label())), "{r:?}: ({}, {}) is not a module form", r.form(), r.label());
        assert_eq!(Rows4e::parse(Some(r.label())), Ok(r), "{r:?} parses from its own label");
        llvq_cuda::pers::p4e_rows_define(r.form()).unwrap_or_else(|e| panic!("{r:?}: {e}"));
    }
    // v5, v7 and v8 are module forms and not served ones.
    for (form, label) in [(3, "v5"), (5, "v7"), (6, "v8")] {
        assert!(P4E_ROWS_FORMS.contains(&(form, label)));
        assert!(Rows4e::parse(Some(label)).is_err(), "{label}");
    }
}

/// The field: absent and "v3" are form 1, "v4" form 2, "v6" form 4; "v5" is refused by
/// name, and so are the empty string and any other word; beside the eager
/// decode the key itself is refused.
#[test]
fn the_rows_field_serves_v3_v4_and_v6_and_refuses_the_rest() {
    let mut f = four_e();
    assert_eq!(of(f.clone()).expect("absent").rows, Rows4e::V3);
    f.rows = Some("v3".into());
    assert_eq!(of(f.clone()).expect("v3").rows, Rows4e::V3);
    f.rows = Some("v4".into());
    assert_eq!(of(f.clone()).expect("v4").rows, Rows4e::V4);
    f.rows = Some("v6".into());
    assert_eq!(of(f.clone()).expect("v6").rows, Rows4e::V6);
    f.rows = Some("v9".into());
    assert_eq!(of(f.clone()).expect("v9").rows, Rows4e::V9);
    let e = {
        f.rows = Some("v5".into());
        of(f.clone()).expect_err("v5")
    };
    assert!(e.contains("v5") && e.contains("measurement"), "{e}");
    let e = Rows4e::parse(Some("v7")).expect_err("v7");
    assert!(e.contains("local memory") && e.contains("does not install"), "v7 is refused by its cause: {e}");
    let e = Rows4e::parse(Some("v8")).expect_err("v8");
    assert!(e.contains("measurement") && e.contains("x1.035"), "v8 is refused by its cause: {e}");
    let named = Rows4e::parse(Some("v10")).expect_err("v10");
    assert!(named.contains("\"v3\", \"v4\", \"v6\" and \"v9\""), "the refusal names the served forms: {named}");
    for bad in ["", " ", "V4", "V6", "V9", "4", "v2", "v7", "v8", "6", "9"] {
        f.rows = Some(bad.into());
        let e = of(f.clone()).expect_err(bad);
        assert!(e.contains("rows"), "{bad:?}: {e}");
    }
    for v in ["v4", "v6", "v9"] {
        let mut g = four_e();
        g.decode = None;
        g.rows = Some(v.into());
        let e = of(g).expect_err("rows beside the eager decode");
        assert!(e.contains("eager decode") && e.contains(v), "{e}");
    }
}

// ---- the doors of bin/chat and bin/gsm8k (2026-11-06) ------------------------

/// `decode` and `rows` are read as written, as `attention` is: a value with a
/// space around it is refused, not trimmed into the value it resembles. A
/// whitespace-only value is refused as empty, with the field's name.
#[test]
fn the_optional_fields_are_read_as_written() {
    for v in [" persistent", "persistent ", "\tpersistent"] {
        let e = Decode::parse(Some(v)).expect_err(v);
        assert!(e.contains("decode"), "{v:?}: {e}");
        let e = AttnHeads::parse(Some(&v.replace("persistent", "kernel"))).expect_err(v);
        assert!(e.contains("attention"), "{v:?}: {e}");
    }
    for v in [" v4", "v4 ", " v3", " v6", "v6 ", " v9", "v9 "] {
        let e = Rows4e::parse(Some(v)).expect_err(v);
        assert!(e.contains("rows"), "{v:?}: {e}");
    }
    for (v, field) in [("  ", "decode"), ("\t", "decode")] {
        let e = Decode::parse(Some(v)).expect_err(v);
        assert!(e.contains(field) && e.contains("empty"), "{v:?}: {e}");
        let e = Rows4e::parse(Some(v)).expect_err(v);
        assert!(e.contains("rows") && e.contains("empty"), "{v:?}: {e}");
        let e = AttnHeads::parse(Some(v)).expect_err(v);
        assert!(e.contains("attention") && e.contains("empty"), "{v:?}: {e}");
    }
    assert_eq!(Decode::parse(Some("persistent")), Ok(Decode::Persistent));
    assert_eq!(Rows4e::parse(Some("v4")), Ok(Rows4e::V4));
    // Every shipped config still resolves: none relied on the trim.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs");
    for name in [
        "qwen3-4b-tetra-e4-4e.json",
        "qwen3-4b-tetra-e4-4e-v4.json",
        "qwen3-4b-tetra-e4-4e-v6.json",
        "qwen3-4b-tetra-e4-4e-v9.json",
    ] {
        let path = dir.join(name);
        let f = Served::read_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        Served::of(f, path).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// A removed variable is refused by name, by `check_removed` and by every
/// served config; unset passes, and a live variable passes.
#[test]
fn removed_variables_are_refused_by_name() {
    check_removed(env_of(&[])).expect("unset");
    check_removed(env_of(&[("LLVQ_KV", "q8")])).expect("a live variable");
    for var in REMOVED {
        let e = check_removed(env_of(&[(var, "1")])).expect_err(var);
        assert!(e.contains(var) && e.contains("pre-clean-2026-11"), "{e}");
        let e = check_env(&served(), env_of(&[(var, "1")])).expect_err(var);
        assert!(e.contains(var), "{e}");
        assert!(!MEASUREMENT_MODES.contains(&var), "{var} is both removed and a mode");
    }
    assert!(REMOVED.contains(&"LLVQ_KV_PREALLOC") && REMOVED.contains(&"LLVQ_PERS_SEGMENT"));
}

/// The chat door: the store variable under any config, and a temperature
/// above zero under the persistent decode, are refused before the load.
#[test]
fn the_chat_door_refuses_what_its_config_cannot_serve() {
    use llvq_llm::repl::check_served_chat;
    let eager = of(served()).expect("eager");
    let fe = of(four_e()).expect("4E");
    check_served_chat(&eager, 0.7, env_of(&[])).expect("an eager config samples");
    check_served_chat(&fe, 0.0, env_of(&[])).expect("4E at temperature 0");
    let e = check_served_chat(&fe, 0.7, env_of(&[])).expect_err("4E cannot sample");
    assert!(e.contains("persistent") && e.contains("LLVQ_CHAT_TEMP"), "{e}");
    for cfg in [&eager, &fe] {
        let e = check_served_chat(cfg, 0.0, env_of(&[("LLVQ_KV_PREALLOC", "4096")])).expect_err("removed");
        assert!(e.contains("LLVQ_KV_PREALLOC") && e.contains("removed"), "{e}");
    }
}
