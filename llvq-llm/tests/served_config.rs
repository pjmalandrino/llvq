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
use llvq_llm::kvq::KvMode;
use llvq_llm::rotplan::RotShare;
use llvq_llm::served::{check_env, Served, ServedFile, BELOW_THE_DOOR};
use std::path::PathBuf;

fn served() -> ServedFile {
    ServedFile {
        layout: "tetra48".into(),
        embed: "q8".into(),
        rot_share: "1".into(),
        fuse: "0".into(),
        kv: "f16".into(),
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

/// The two variables that reach below the served door are refused by name,
/// whatever their value, and the card variable is not.
#[test]
fn variables_below_the_door_are_refused_by_name() {
    let f = served();
    assert_eq!(BELOW_THE_DOOR, ["LLVQ_KERNEL_DIR", "LLVQ_TILE_BLOCKS"]);
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
