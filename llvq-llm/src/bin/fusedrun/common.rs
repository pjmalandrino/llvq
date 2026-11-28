//! What several modes of `fusedrun` share: the round protocol, the rate
//! statistics, the token comparison, the parsers of the mode variables and the
//! attention arms. One file because each piece is small and each has two users.

/// The measurement protocol, aligned on the kernel benches (2026-08-18):
/// one discarded generation per arm (kernel selection, allocator growth,
/// clock ramp), then `ROUNDS_TIMED` timed ones, reported as median + min–max
/// range. Until then every published tok/s was a single point — in
/// contradiction with rule 2 of CLAUDE.md §7, which the benches obeyed and
/// this binary did not.
///
/// Unlike `planesbench`, the two arms can NOT interleave round by round:
/// each arm loads its model exclusively (the card does not have to hold
/// both), so their rounds never coexist and the speed ratio is a **quotient
/// of two medians**, not a median of per-round ratios. The summary labels it
/// as such and prints the conservative envelope
/// `[fused_lo/dense_hi ; fused_hi/dense_lo]` beside it.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) const ROUNDS_TIMED: usize = 5;

/// Median and min–max of per-round rates. Not cfg-gated: the Mac must
/// type-check what a paid image build would otherwise be the first to compile.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn rate_stats(rounds: &[f64]) -> (f64, f64, f64) {
    assert!(!rounds.is_empty(), "rate_stats over zero rounds");
    let mut sorted = rounds.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    };
    (med, sorted[0], sorted[sorted.len() - 1])
}

/// Position of the first token the two arms disagree on, or `None` when they
/// agree everywhere.
///
/// Refuses arms of different lengths instead of comparing their common prefix.
/// `zip` truncates, so an arm that returned fewer tokens would have printed
/// "N tokens identical" over the prefix and said nothing about the rest — a
/// green that means nothing, which is the failure mode this binary's own gate
/// clause exists to prevent. Out here rather than inline for the reason
/// `rotplan::arms_are_discriminating` is out of this file: a comparison written
/// inside the cfg-gated body is checked by no machine in this workspace.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn first_divergence(a: &[u32], b: &[u32]) -> Result<Option<usize>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "{} tokens against {}: the two arms did not decode the same length, and \
             comparing their common prefix would print a green that says nothing",
            a.len(),
            b.len()
        ));
    }
    Ok(a.iter().zip(b).position(|(x, y)| x != y))
}

/// The prompt lengths of `LLVQ_CONTEXTS`, comma-separated, each above zero.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn parse_contexts(v: &str) -> Result<Vec<usize>, String> {
    parse_lengths("LLVQ_CONTEXTS", v)
}

/// Prompt lengths, comma-separated, each above zero, read from `var`.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn parse_lengths(var: &str, v: &str) -> Result<Vec<usize>, String> {
    let lengths = v
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{var}={v:?}: {e}"))?;
    if lengths.is_empty() || lengths.contains(&0) {
        return Err(format!("{var}={v:?}: every prompt length must be above zero"));
    }
    Ok(lengths)
}

/// The value of `LLVQ_GQA_PPL`: `<ctx>,<windows>` scores the grouped form
/// against the published one from window 0, as every control so far did;
/// `<ctx>,<windows>,<first>,kernel` scores the kernel form against the grouped
/// one (the served form) from window index `<first>`, so that a T8 control
/// takes windows no earlier control scored. `<ctx>,<windows>,<first>,ladder`
/// scores the cache of the `LLVQ_KV_LADDER` arm against the f16 cache, both
/// under the kernel form on every decode row (the served 4E attention).
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
#[derive(Debug, PartialEq, Eq)]
pub(super) struct GqaPplSpec {
    pub(super) ctx: usize,
    pub(super) windows: usize,
    pub(super) first: usize,
    pub(super) kernel: bool,
    /// `kernel0`: MEASUREMENT FORM, every decode row on the kernel from
    /// position 0, the T8 extension below `KERNEL_FROM` that serving 4E at a
    /// short prompt would need. Implies `kernel`.
    pub(super) every_row: bool,
    /// `ladder`: the two arms share the `kernel0` form and differ by their
    /// cache, f16 against `LLVQ_KV_LADDER`'s arm. Implies `every_row`.
    pub(super) ladder: bool,
}

#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn parse_gqa_ppl(v: &str) -> Result<GqaPplSpec, String> {
    let parts: Vec<&str> = v.split(',').map(str::trim).collect();
    let (nums, kernel, every_row, ladder) = match parts.as_slice() {
        [c, w] => (vec![*c, *w], false, false, false),
        [c, w, f, "kernel"] => (vec![*c, *w, *f], true, false, false),
        [c, w, f, "kernel0"] => (vec![*c, *w, *f], true, true, false),
        [c, w, f, "ladder"] => (vec![*c, *w, *f], true, true, true),
        _ => {
            return Err(format!(
                "LLVQ_GQA_PPL={v:?}: expected <ctx>,<windows> (grouped against published, from \
                 window 0), <ctx>,<windows>,<first>,kernel (kernel against grouped), \
                 <ctx>,<windows>,<first>,kernel0 (kernel on every row against grouped) or \
                 <ctx>,<windows>,<first>,ladder (LLVQ_KV_LADDER's cache against f16)"
            ))
        }
    };
    let n = nums
        .iter()
        .map(|s| s.parse::<usize>().map_err(|e| format!("LLVQ_GQA_PPL={v:?}: {s:?}: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    let spec = GqaPplSpec { ctx: n[0], windows: n[1], first: n.get(2).copied().unwrap_or(0), kernel, every_row, ladder };
    if spec.ctx < 2 || spec.windows < 2 {
        return Err(format!("LLVQ_GQA_PPL={v:?}: two windows of two tokens at least"));
    }
    // The published-against-grouped mode scores the prefill path of every
    // window in one forward, bounded by the fused prefill's rows. The kernel
    // mode scores that path on its first window only, and only when it fits.
    if !spec.kernel && spec.ctx > llvq_llm::model::MAX_PREFILL_ROWS {
        return Err(format!(
            "LLVQ_GQA_PPL: a window of {} exceeds the fused prefill's {} rows",
            spec.ctx,
            llvq_llm::model::MAX_PREFILL_ROWS
        ));
    }
    // The kernel form sends a position to its kernel from KERNEL_FROM on: a
    // window must reach past it, or the mode scores the grouped form twice.
    let from = llvq_llm::kvq::KERNEL_FROM;
    if spec.kernel && !spec.every_row && spec.ctx <= from + 1 {
        return Err(format!(
            "LLVQ_GQA_PPL: a window of {} never reaches the kernel, which takes positions {from} \
             and later; ask {} tokens at least",
            spec.ctx,
            from + 2
        ));
    }
    Ok(spec)
}

/// The arms of `LLVQ_GQA_AB`: the published attention (`Cat` store, expanded
/// heads), and the two forms that read the cached heads unexpanded from a
/// store grown in place by [`llvq_llm::kvq::GROW_CHUNK`] positions.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GqaArm {
    Published,
    Shared,
    Grouped,
    /// Decode and verify rows through the split-KV attention kernel
    /// (`llvq_llm::attn`), longer calls through the grouped path. Not in
    /// [`GqaArm::ALL`]: only `LLVQ_GQA_PPL`'s kernel mode runs it, which
    /// installs the kernel first.
    Kernel,
}

#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
impl GqaArm {
    pub(super) const ALL: [GqaArm; 3] = [GqaArm::Published, GqaArm::Shared, GqaArm::Grouped];

    pub(super) fn store(self) -> llvq_llm::kvq::KvStore {
        match self {
            GqaArm::Published => llvq_llm::kvq::KvStore::Cat,
            GqaArm::Shared | GqaArm::Grouped | GqaArm::Kernel => {
                llvq_llm::kvq::KvStore::Grow(llvq_llm::kvq::GROW_CHUNK)
            }
        }
    }

    pub(super) fn heads(self) -> llvq_llm::kvq::AttnHeads {
        match self {
            GqaArm::Published => llvq_llm::kvq::AttnHeads::Expanded,
            GqaArm::Shared => llvq_llm::kvq::AttnHeads::Shared,
            GqaArm::Grouped => llvq_llm::kvq::AttnHeads::Grouped,
            GqaArm::Kernel => llvq_llm::kvq::AttnHeads::Kernel,
        }
    }

    /// What every line of the table prints: the store and the form, as set.
    pub(super) fn name(self) -> String {
        format!("{}+{}", self.store().label(), self.heads().label())
    }

    pub(super) fn apply(self, m: &mut llvq_llm::model::Qwen3) {
        m.set_kv_store(self.store());
        m.set_attn_heads(self.heads());
    }
}

/// The prompt lengths `LLVQ_PERS_4E` runs after the seed prompt when its
/// value names none: 8,192, the long context of every reading before
/// 2026-11-11.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
#[cfg(any(target_os = "macos", feature = "cuda"))] // names llvq_cuda, a dependency on macOS and under `cuda` only
const PERS_4E_LONG: [usize; 1] = [8192];

/// `LLVQ_PERS_4E`'s value: a token count, 2 to 256, alone (the bench forms,
/// `pers::P4E_BENCH_FORMS`) or followed by `:` and E1 arms
/// (`pers::P4E_E1_ARMS`), comma-separated, each once, which replace the
/// bench forms; then optionally `:` and the prompt lengths run after the seed
/// prompt, comma-separated, strictly ascending, each 64 to 8,192 (default
/// [`PERS_4E_LONG`]). One repetition is taken: the first arm named again as
/// the LAST of three or more (`v9,v9ep,v9fl,v9`), the fidelity control of
/// rule F1 (docs/mesures/recherche-debit-r3-espace-2026-11-12.txt): the first
/// arm's runtime timed again at the end of every round, no second module.
/// The fourth value is that control.
/// [`parse_pers_4e`]'s value: the token count, the E1 arms, the prompt
/// lengths, and whether the first arm is timed again last (rule F1).
#[cfg(any(target_os = "macos", feature = "cuda"))]
type Pers4eSpec = (usize, Vec<&'static str>, Vec<usize>, bool);

#[cfg(any(target_os = "macos", feature = "cuda"))] // names llvq_cuda, a dependency on macOS and under `cuda` only
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn parse_pers_4e(v: &str) -> Result<Pers4eSpec, String> {
    let mut fields = v.splitn(3, ':');
    let count = fields.next().unwrap_or_default();
    let arms = fields.next();
    let lengths = match fields.next() {
        None => PERS_4E_LONG.to_vec(),
        Some(l) => {
            let lengths = parse_lengths("LLVQ_PERS_4E", l)?;
            if lengths.iter().any(|n| !(64..=8192).contains(n)) || lengths.windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!("LLVQ_PERS_4E={v:?}: lengths strictly ascending, each 64 to 8192"));
            }
            lengths
        }
    };
    let n: usize = count.parse().map_err(|_| format!("LLVQ_PERS_4E={v:?}: a token count, 2 to 256, then optionally :<E1 arms>"))?;
    if !(2..=256).contains(&n) {
        return Err(format!("LLVQ_PERS_4E={v:?}: 2 to 256 tokens a run"));
    }
    let mut out: Vec<&'static str> = Vec::new();
    let mut again = false;
    if let Some(a) = arms {
        let names: Vec<&str> = a.split(',').collect();
        let names = match names.as_slice() {
            [first, .., last] if names.len() >= 3 && first == last => {
                again = true;
                &names[..names.len() - 1]
            }
            _ => &names[..],
        };
        for &name in names {
            let arm = llvq_cuda::pers::P4E_E1_ARMS
                .iter()
                .find(|x| **x == name)
                .ok_or_else(|| format!("LLVQ_PERS_4E={v:?}: no E1 arm {name:?}; the arms are {:?}", llvq_cuda::pers::P4E_E1_ARMS))?;
            if out.contains(arm) {
                return Err(format!("LLVQ_PERS_4E={v:?}: arm {name} named twice"));
            }
            out.push(arm);
        }
    }
    Ok((n, out, lengths, again))
}

/// `LLVQ_PERS_4E`'s `+local` marks (`pers::P4E_LOCAL_SUFFIX`): an arm named
/// `<arm>+local` is timed even where its token symbol holds local memory
/// (`persist4e_cuda::install_e1_local`), a measurement of what the load rule
/// protects. Returns the value with the marks removed, for [`parse_pers_4e`],
/// and the arms marked. Refused on the first arm: the reference of every
/// ratio loads under the rule or the run has none.
#[cfg(any(target_os = "macos", feature = "cuda"))]
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn split_pers_4e_local(v: &str) -> Result<(String, Vec<String>), String> {
    let suffix = llvq_cuda::pers::P4E_LOCAL_SUFFIX;
    let mut fields: Vec<String> = v.splitn(3, ':').map(str::to_string).collect();
    let mut marked: Vec<String> = Vec::new();
    if let Some(arms) = fields.get_mut(1) {
        let mut names = Vec::new();
        for (i, name) in arms.split(',').enumerate() {
            match name.strip_suffix(suffix) {
                Some(_) if i == 0 => return Err(format!("LLVQ_PERS_4E={v:?}: the first arm is the reference and takes no {suffix}")),
                Some(bare) if marked.iter().any(|m| m == bare) => return Err(format!("LLVQ_PERS_4E={v:?}: arm {bare} named twice")),
                Some(bare) => {
                    marked.push(bare.to_string());
                    names.push(bare);
                }
                None => names.push(name),
            }
        }
        *arms = names.join(",");
    }
    Ok((fields.join(":"), marked))
}

/// `LLVQ_PERS_4E=check:<arms>`'s arms: E1 arms (`pers::P4E_E1_ARMS`),
/// comma-separated, each once, at least one.
#[cfg(any(target_os = "macos", feature = "cuda"))] // names llvq_cuda, a dependency on macOS and under `cuda` only
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn parse_pers_4e_check(v: &str) -> Result<Vec<&'static str>, String> {
    let mut out: Vec<&'static str> = Vec::new();
    for name in v.split(',') {
        // the load probes (pers::P4E_RC_PROBES) are named here and nowhere else: never timed
        let arm = llvq_cuda::pers::P4E_E1_ARMS
            .iter()
            .chain(llvq_cuda::pers::P4E_RC_PROBES.iter().map(|p| &p.0))
            .find(|x| **x == name)
            .ok_or_else(|| {
                format!(
                    "LLVQ_PERS_4E=check:{v}: no E1 arm or load probe {name:?}; the arms are {:?}, the probes {:?}",
                    llvq_cuda::pers::P4E_E1_ARMS,
                    llvq_cuda::pers::P4E_RC_PROBES.map(|p| p.0)
                )
            })?;
        if out.contains(arm) {
            return Err(format!("LLVQ_PERS_4E=check:{v}: arm {name} named twice"));
        }
        out.push(arm);
    }
    Ok(out)
}
