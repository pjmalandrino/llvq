//! The `codebook` positional: the strict parser that refuses every spelling it
//! does not name, and the journal line that prints the codebook it resolved to.
//! One file because the grammar and its printed form must agree field by field.

use llvq_llm::calib::Codebook;

/// Gain bits this binary will accept. The published configurations use 0, 1
/// and 2; `Planes14`, the reference runtime layout, freezes the field at one
/// bit. 8 is already 256 levels re-fitted by 40 k-means passes over every
/// block of every matrix — past it, `fit_gain_centroids`'s `1 << k_bits` is
/// the next thing to break, and it breaks three hours in.
const MAX_GAIN_BITS: u32 = 8;

/// Parse the `codebook` positional. **Nothing here falls back.**
///
/// Grammar:
///
/// ```text
/// identity | grid | direction
/// leech[<gain>][c<shell>][L<levels>][f]
/// tetra[<gain>]
/// ```
///
/// in that order — `leech1c12L3` is the 8B configuration, `leech1c12f` charges
/// the pre-2026-07-31 free f16 magnitude. An absent `<gain>` means 1, which is
/// what the bare default `leech` has always meant.
///
/// What the previous parser did instead, and why each one mattered:
///
/// * `strip_prefix("leech").unwrap_or("1")` — `leach1c12` became `leech1`:
///   the **full ball**, 48 index bits, a different rate and a different file;
///
/// `tetra` is the Tetra word map (roadmap §2.2 quater): 47 bits of label and one
/// gain bit, the same 48 bits per block as `leech1c12` and deliberately so —
/// the two arms are compared at a constant rate. It takes no shell, no level
/// cap and no `f`: the map is one fixed label set, and the word has a gain bit
/// whether or not it is spent. Each of those is refused by name rather than
/// ignored, for the reason the whole of this function exists.
///
/// * `c.parse().unwrap_or(13)` — `leech1cx` became the full ball too;
/// * `split_once('c')` is case-sensitive, so `leech1C12` also became the full
///   ball;
/// * `l.parse().unwrap_or(5)` — `leech1L3c12` (fields out of order) became
///   `leech1` with no level cap, silently dropping *both* modifiers.
pub(super) fn parse_codebook(spec: &str) -> Result<Codebook, String> {
    match spec {
        "identity" => return Ok(Codebook::Identity),
        "grid" => return Ok(Codebook::Grid { step: 0.01 }),
        "direction" => return Ok(Codebook::Direction),
        _ => {}
    }
    if spec.starts_with("int") {
        return parse_int(spec);
    }
    if spec.starts_with("tetra") {
        return parse_tetra(spec);
    }
    let grammar = "accepted values `identity`, `grid`, `direction`, \
                   `int<bits>g<group>`, `tetra[<gain>]` and \
                   `leech[<gain>][c<shell>][L<levels>][f]` — \
                   e.g. `int3g24`, `tetra`, `leech1`, `leech1c12`, `leech1c12L3`, `leech1c12f`";
    let rest = spec
        .strip_prefix("leech")
        .ok_or_else(|| format!("codebook {spec:?}: {grammar}"))?;

    let (gain, rest) = split_digits(rest);
    let gain_bits: u32 = if gain.is_empty() {
        1
    } else {
        gain.parse()
            .map_err(|e| format!("codebook {spec:?}: gain {gain:?} unreadable ({e})"))?
    };
    // The `f` suffix sits at the very end, so it comes off before the fields
    // are read left to right. Order is then enforced by requiring an empty
    // remainder — which is what rejects `leech1L3c12`.
    let (rest, free_magnitude) = match rest.strip_suffix('f') {
        Some(r) => (r, true),
        None => (rest, false),
    };
    let (rest, max_shell) = take_field(spec, rest, 'c', 13)?;
    let (rest, level_cap) =
        take_field(spec, rest, 'L', llvq_search::generic::MAX_LEVELS_ANY as u32)?;
    if !rest.is_empty() {
        return Err(format!("codebook {spec:?}: {rest:?} left over — {grammar}"));
    }

    if gain_bits > MAX_GAIN_BITS {
        return Err(format!(
            "codebook {spec:?}: {gain_bits} gain bits, maximum {MAX_GAIN_BITS}"
        ));
    }
    // `BallSearcher::set_shell_cap` asserts this range three hours into a run;
    // `enumerate_classes` asserts the upper half of it one line below.
    if !(2..=llvq_search::classes::MAX_SHELL).contains(&max_shell) {
        return Err(format!(
            "codebook {spec:?}: shell {max_shell}, accepted 2..={}",
            llvq_search::classes::MAX_SHELL
        ));
    }
    let cap = llvq_search::generic::MAX_LEVELS_ANY as u32;
    if level_cap == 0 || level_cap > cap {
        return Err(format!(
            "codebook {spec:?}: {level_cap} levels, accepted 1..={cap} ({cap} excludes nothing)"
        ));
    }
    let level_cap = level_cap as usize;
    // The cap combination has to leave something to search. `L1` passes every
    // range check above and empties the codebook up to shell 5; from shell 6
    // it keeps the one class `(±2)²⁴`. An empty codebook would otherwise
    // surface as a searcher with no classes, well after the model is loaded.
    let classes = class_count(max_shell, level_cap);
    if classes == 0 {
        return Err(format!(
            "codebook {spec:?}: no class survives (shell ≤ {max_shell}, \
             levels ≤ {level_cap}) — the codebook would be empty"
        ));
    }
    Ok(Codebook::ShapeGain {
        gain_bits,
        max_shell,
        free_magnitude,
        level_cap,
    })
}

/// Parse `tetra[<gain>][post]` — the Tetra word map.
///
/// **Nothing here falls back either**, and the fields it refuses are the ones
/// `leech` accepts: a shell cap, a level cap and the free-magnitude `f`. Each
/// would be silently meaningless on this map — there is one label set, and
/// the gain bit is in the word whether or not it is used — and a run whose
/// log printed `tetra1c12` while the file was written without a cap is exactly
/// the class of defect `parse_codebook` was rewritten for. `f` gets its own
/// message because it is the one that would change the *rate*: 47 + 16 bits
/// is not what this arm measures.
///
/// The gain is 1 or absent. `llvq_artifact`'s writer refuses any other count
/// three hours into a run; refusing it here costs microseconds.
fn parse_tetra(spec: &str) -> Result<Codebook, String> {
    let grammar = "grammar `tetra[<gain>][post]` with `<gain>` = 1 — e.g. `tetra`, `tetra1` or `tetrapost`. \
                   A Tetra word is 47 bits of label and one gain bit: it takes no shell \
                   cap (`c`), no level cap (`L`) and no free magnitude (`f`)";
    let rest = spec
        .strip_prefix("tetra")
        .ok_or_else(|| format!("codebook {spec:?}: {grammar}"))?;
    // The `f` suffix is named on its own: `leech1c12f` is a documented arm at
    // 63 bits per block, and someone typing `tetraf` is asking for a rate this
    // map cannot have rather than making a typo.
    if let Some(head) = rest.strip_suffix('f') {
        return Err(format!(
            "codebook {spec:?}: `tetra{head}` plus a free magnitude — the gain bit is inside \
             the 48-bit word, so there is nothing to free and the rate would not be the \
             48 bits/block this arm exists to compare. {grammar}"
        ));
    }
    let (rest, post_shape_gain) = match rest.strip_suffix("post") {
        Some(head) => (head, true),
        None => (rest, false),
    };
    let (gain, rest) = split_digits(rest);
    if !rest.is_empty() {
        return Err(format!("codebook {spec:?}: {rest:?} left over — {grammar}"));
    }
    let gain_bits: u32 = if gain.is_empty() {
        1
    } else {
        gain.parse()
            .map_err(|e| format!("codebook {spec:?}: gain {gain:?} unreadable ({e})"))?
    };
    if gain_bits != 1 {
        return Err(format!(
            "codebook {spec:?}: {gain_bits} gain bits — a Tetra word carries exactly one, \
             at bit {}. {grammar}",
            llvq_search::tetra::LABEL_BITS
        ));
    }
    Ok(Codebook::Tetra { gain_bits, post_shape_gain })
}

/// Parse `int<bits>g<groupe>` — the affine scalar arm.
///
/// Both fields are mandatory, and neither falls back. `int3` alone would have
/// to invent a group size, and the group is not a detail: it *is* the GPTQ
/// block (see [`Codebook::block_len`]), so a defaulted one would silently
/// change the error-feedback granularity of the run rather than just its rate.
///
/// `g24` pairs with a Leech arm — same block, same feedback, one variable.
/// `g128` is the field default of AutoGPTQ, and costs `bits + 0.25` instead of
/// `bits + 1.33`.
fn parse_int(spec: &str) -> Result<Codebook, String> {
    let grammar = "grammar `int<bits>g<group>` — bits 1..=8, group 8..=1024, \
                   e.g. `int3g24` (paired with the Leech block) or `int4g128` \
                   (the AutoGPTQ default)";
    let rest = spec
        .strip_prefix("int")
        .ok_or_else(|| format!("codebook {spec:?}: {grammar}"))?;
    let (digits, rest) = split_digits(rest);
    if digits.is_empty() {
        return Err(format!("codebook {spec:?}: bits missing — {grammar}"));
    }
    let bits: u32 = digits
        .parse()
        .map_err(|e| format!("codebook {spec:?}: bits {digits:?} unreadable ({e})"))?;
    let (rest, group) = take_field_required(spec, rest, 'g', grammar)?;
    if !rest.is_empty() {
        return Err(format!("codebook {spec:?}: {rest:?} left over — {grammar}"));
    }
    // `1u32 << bits` overflows at 32, and a 9-bit "INT" is not a thing anyone
    // ships; the bound is the useful one, not the representable one.
    if !(1..=8).contains(&bits) {
        return Err(format!(
            "codebook {spec:?}: {bits} bits, accepted 1..=8 — {grammar}"
        ));
    }
    // The lower bound keeps the f16 scale and zero from dominating the rate
    // (at 8 they already cost 4 bits/weight); the upper one is the narrowest
    // `d_in` of the models this harness loads (1024 on Qwen3-0.6B), past
    // which `TailPolicy::KeepExact` would quantize nothing at all and the run
    // would report a baseline as if it were a result.
    if !(8..=1024).contains(&group) {
        return Err(format!(
            "codebook {spec:?}: group {group}, accepted 8..=1024 — {grammar}"
        ));
    }
    Ok(Codebook::ScalarGroup {
        bits,
        group: group as usize,
    })
}

/// Read a mandatory `<tag><digits>` field off the front of `rest`.
///
/// The optional twin is [`take_field`]; this one exists because
/// [`parse_int`]'s group has no defensible default.
fn take_field_required<'a>(
    spec: &str,
    rest: &'a str,
    tag: char,
    grammar: &str,
) -> Result<(&'a str, u32), String> {
    let after = rest
        .strip_prefix(tag)
        .ok_or_else(|| format!("codebook {spec:?}: `{tag}` expected — {grammar}"))?;
    let (digits, rest) = split_digits(after);
    if digits.is_empty() {
        return Err(format!(
            "codebook {spec:?}: `{tag}` must be followed by a number — {grammar}"
        ));
    }
    let v = digits
        .parse()
        .map_err(|e| format!("codebook {spec:?}: `{tag}{digits}` unreadable ({e})"))?;
    Ok((rest, v))
}

/// Split off the leading run of ASCII digits.
fn split_digits(s: &str) -> (&str, &str) {
    let n = s.len() - s.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    s.split_at(n)
}

/// Read an optional `<tag><digits>` field off the front of `rest`.
///
/// A tag with no digits behind it is an error rather than the default: `c` and
/// `cx` both used to mean "shell 13".
fn take_field<'a>(
    spec: &str,
    rest: &'a str,
    tag: char,
    default: u32,
) -> Result<(&'a str, u32), String> {
    let Some(after) = rest.strip_prefix(tag) else {
        return Ok((rest, default));
    };
    let (digits, tail) = split_digits(after);
    if digits.is_empty() {
        return Err(format!(
            "codebook {spec:?}: `{tag}` must be followed by a number"
        ));
    }
    let v = digits
        .parse()
        .map_err(|e| format!("codebook {spec:?}: `{tag}{digits}` unreadable ({e})"))?;
    Ok((tail, v))
}

/// Lattice classes the codebook actually keeps: shells `2..=max_shell`, then
/// the level filter `BallSearcher::with_level_cap` applies.
///
/// Printed in the journal because it is the quantity this project keeps
/// arguing about — 301 classes for Λ₂₄(12), 383 for Λ₂₄(13) (CLAUDE.md, §une
/// seule coquille) — and because it is the one number that separates two
/// codebooks at a glance in an archived log.
pub(super) fn class_count(max_shell: u32, level_cap: usize) -> usize {
    let cs = llvq_search::classes::enumerate_classes(max_shell);
    cs.even.iter().filter(|c| c.n_levels() <= level_cap).count()
        + cs.odd.iter().filter(|c| c.n_levels() <= level_cap).count()
}

/// One line describing the codebook that will actually be used.
///
/// This is the line the run logs were missing. `docs/fiche-4b.md` §2.3 had to
/// reconstruct the shipped configuration from the *result* line, because the
/// configuration line was a literal that said "0 gain bits" while the run used
/// one — on all three published runs.
pub(super) fn codebook_line(spec: &str, c: &Codebook) -> String {
    let what = match c {
        Codebook::Identity => "lossless control, no code".to_string(),
        Codebook::Grid { step } => format!("scalar grid, step {step}"),
        // The rate is spelled out because it is the one thing a reader will
        // want to compare against a lattice arm, and `bits + 32/group` is not
        // something to do in one's head at the top of a log.
        Codebook::ScalarGroup { bits, group } => format!(
            "affine INT{bits} (AutoGPTQ sym=false), f16 scale + {bits}-bit zero \
             per group of {group}, {:.4} b/weight before the per-row f16 \
             (which this arm does not use), no artifact",
            *bits as f64 + (16.0 + *bits as f64) / *group as f64
        ),
        Codebook::Direction => {
            "direction only, magnitude left a free float — NOT a 2-bit code".to_string()
        }
        Codebook::ShapeGain {
            gain_bits,
            max_shell,
            free_magnitude,
            level_cap,
        } => format!(
            "shape–gain, {gain_bits} gain bit{}, shell ≤ {max_shell}, levels ≤ {level_cap}, \
             {} classes, magnitude {}",
            if *gain_bits == 1 { "" } else { "s" },
            class_count(*max_shell, *level_cap),
            if *free_magnitude {
                "a free f16 (charged as 16 bits)"
            } else {
                "on the gain grid"
            }
        ),
        // Named for what separates it from the `leech1c12` arm it is compared
        // against: the same 48 bits, a different map for the direction. The
        // class count has no counterpart here — Tetra's label set is one fixed
        // trellis, not a union of shells — so the word's own geometry takes
        // its place on the line.
        Codebook::Tetra {
            gain_bits,
            post_shape_gain,
        } => format!(
            "Tetra word map, {} bits of label + {gain_bits} gain bit{}, \
             3 octads of 8, {} rows per class, magnitude on the gain grid, gain selected {}",
            llvq_search::tetra::LABEL_BITS,
            if *gain_bits == 1 { "" } else { "s" },
            llvq_search::tetra::CLASS_ROWS,
            if *post_shape_gain {
                "after direction"
            } else {
                "from source norm"
            }
        ),
    };
    format!("{spec} → {what}, {:.0} bits/block", c.block_bits())
}
