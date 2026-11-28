//! Unit tests of the parsers, the statistics and the door of `fusedrun`, one
//! file because each reaches its subject through `super`.

use super::common::{first_divergence, parse_gqa_ppl, rate_stats, GqaArm, GqaPplSpec};
use super::ppl::paired_ppl_pct;
use super::{alone_in, ALONE};
use llvq_llm::eval::t95;

/// `LLVQ_PERS_4E`: a count alone keeps the bench forms; a count and E1
/// arms names those arms in order, then optionally the long prompt
/// lengths (8,192 alone when absent); anything else is refused.
#[test]
#[cfg(any(target_os = "macos", feature = "cuda"))]
fn pers_4e_takes_a_count_and_optional_e1_arms() {
    use super::common::parse_pers_4e;
    assert_eq!(parse_pers_4e("64"), Ok((64, vec![], vec![8192], false)));
    assert_eq!(parse_pers_4e("64:v9,v9np,v9mv"), Ok((64, vec!["v9", "v9np", "v9mv"], vec![8192], false)));
    assert_eq!(parse_pers_4e("64:v9mv"), Ok((64, vec!["v9mv"], vec![8192], false)));
    assert_eq!(parse_pers_4e("64:v9,v9ft"), Ok((64, vec!["v9", "v9ft"], vec![8192], false)));
    assert_eq!(parse_pers_4e("64:v9,v9ft:1024,4096,8192"), Ok((64, vec!["v9", "v9ft"], vec![1024, 4096, 8192], false)));
    // the F1 control: the first arm again as the last of three or more
    assert_eq!(parse_pers_4e("256:v9,v9ep,v9fl,v9"), Ok((256, vec!["v9", "v9ep", "v9fl"], vec![8192], true)));
    assert_eq!(parse_pers_4e("256:v9,v9fl,v9:8192"), Ok((256, vec!["v9", "v9fl"], vec![8192], true)));
    // the load check's arms
    use super::common::parse_pers_4e_check;
    assert_eq!(parse_pers_4e_check("v9,v9ep2,v9fl3"), Ok(vec!["v9", "v9ep2", "v9fl3"]));
    for bad in ["", "v9,v9", "v9,v8", "v9 ", "v9ep,"] {
        assert!(parse_pers_4e_check(bad).is_err(), "check:{bad}");
    }
    assert!(parse_pers_4e("check:v9").is_err(), "the check is dispatched before parse_pers_4e");
    // the load probes: the check names them, a timed run never does
    assert_eq!(parse_pers_4e_check("v9pe0,v9peq,v9peg,v9ped,v9peo"), Ok(vec!["v9pe0", "v9peq", "v9peg", "v9ped", "v9peo"]));
    assert!(parse_pers_4e_check("v9peq,v9peq").is_err());
    // R6d's arms: E1 arms, so the check and a timed run both name them
    assert_eq!(parse_pers_4e_check("v9,v9fg,v9fn,v9eg"), Ok(vec!["v9", "v9fg", "v9fn", "v9eg"]));
    assert_eq!(parse_pers_4e("256:v9,v9fg,v9eg,v9"), Ok((256, vec!["v9", "v9fg", "v9eg"], vec![8192], true)));
    assert_eq!(parse_pers_4e("256:v9,v9fn,v9"), Ok((256, vec!["v9", "v9fn"], vec![8192], true)));
    assert_eq!(parse_pers_4e("256:v9fg"), Ok((256, vec!["v9fg"], vec![8192], false)));
    // O13's int8 cache arm: the check and a timed run name it beside v9
    assert_eq!(parse_pers_4e_check("v9,v9k8"), Ok(vec!["v9", "v9k8"]));
    assert_eq!(parse_pers_4e("256:v9,v9k8,v9"), Ok((256, vec!["v9", "v9k8"], vec![8192], true)));
    for probe in llvq_cuda::pers::P4E_RC_PROBES {
        assert!(parse_pers_4e(&format!("64:v9,{}", probe.0)).is_err(), "{} is never timed", probe.0);
    }
    // `+local`: the mark is split off, the value left is an ordinary one
    use super::common::split_pers_4e_local;
    assert_eq!(split_pers_4e_local("256:v9,v9k8,v9"), Ok(("256:v9,v9k8,v9".to_string(), vec![])));
    assert_eq!(split_pers_4e_local("256:v9,v9k8+local,v9:8192"), Ok(("256:v9,v9k8,v9:8192".to_string(), vec!["v9k8".to_string()])));
    assert_eq!(split_pers_4e_local("64"), Ok(("64".to_string(), vec![])));
    let (bare, marked) = split_pers_4e_local("256:v9,v9k8+local,v9").unwrap();
    assert_eq!(parse_pers_4e(&bare), Ok((256, vec!["v9", "v9k8"], vec![8192], true)));
    assert_eq!(marked, ["v9k8"]);
    // never the reference, never twice, and the check does not know the mark
    assert!(split_pers_4e_local("256:v9+local,v9k8,v9").is_err());
    assert!(split_pers_4e_local("256:v9,v9k8+local,v9k8+local").is_err());
    assert!(split_pers_4e_local("256:v9k8+local").is_err());
    assert!(parse_pers_4e_check("v9,v9k8+local").is_err());
    // a mark that names no arm, or sits anywhere but the end of a name, is refused by the parse that follows
    for bad in ["256:v9,v8+local,v9", "256:v9,+local,v9", "256:v9,v9k8+local+local,v9", "256:v9,v9k8+localx,v9"] {
        let (bare, _) = split_pers_4e_local(bad).unwrap();
        assert!(parse_pers_4e(&bare).is_err(), "{bad}");
    }
    for bad in [
        "1", "257", "x", "64:", "64:v9,v9", "64:v8", "64:v9 ,v9np", ":v9", "64:v9:", "64::8192", "64:v9:8192,4096",
        "64:v9:4096,4096", "64:v9:8193", "64:v9:5", "64:v9:0", "64:v9:1024:4096", "64:v9,v9ep,v9ep", "64:v9,v9,v9ep",
        "64:v9,v9ep,v9,v9", "64:v9ep,v9,v9",
    ] {
        assert!(parse_pers_4e(bad).is_err(), "{bad}");
    }
}

fn env_of<'a>(set: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |v| set.iter().find(|(k, _)| *k == v).map(|(_, x)| x.to_string())
}

/// A mode set alone passes its own check. One mode failed this from
/// 2026-11-04 to 2026-11-07, when its list named itself.
#[test]
fn no_mode_refuses_itself() {
    for mode in ALONE {
        alone_in(mode, &["LLVQ_KERNEL_DIR"], env_of(&[(mode, "1")]))
            .unwrap_or_else(|e| panic!("{mode} alone: {e}"));
    }
}

/// Every pair is refused both ways, naming both variables, whatever the
/// order the modes are dispatched in.
#[test]
fn every_pair_is_refused_both_ways() {
    for mode in ALONE {
        for other in ALONE.iter().filter(|o| **o != mode) {
            let e = alone_in(mode, &[], env_of(&[(mode, "1"), (other, "1")]))
                .expect_err(&format!("{mode} beside {other}"));
            assert!(e.contains(mode) && e.contains(other), "{e}");
        }
    }
    // The hole the eight lists left: GQA_PPL returned before the modes
    // dispatched after it were read.
    assert!(alone_in("LLVQ_GQA_PPL", &[], env_of(&[("LLVQ_PREFILL_TOKENS", "1")])).is_err());
}

/// The extra names are refused beside their mode only, and an unrelated
/// variable is not.
#[test]
fn extras_belong_to_their_mode() {
    let dir = [("LLVQ_KERNEL_DIR", "/tmp/k")];
    assert!(alone_in("LLVQ_PERS_4E", &["LLVQ_KERNEL_DIR"], env_of(&dir)).is_err());
    alone_in("LLVQ_GQA_AB", &[], env_of(&dir)).expect("GQA_AB takes a kernel directory");
    alone_in("LLVQ_GQA_AB", &[], env_of(&[("LLVQ_CONFIG", "x.json")])).expect("not a mode");
}

/// The two forms of `LLVQ_GQA_PPL`, and what each refuses.
#[test]
fn gqa_ppl_spec() {
    assert_eq!(
        parse_gqa_ppl("4096,12").unwrap(),
        GqaPplSpec { ctx: 4096, windows: 12, first: 0, kernel: false, every_row: false, ladder: false }
    );
    assert_eq!(
        parse_gqa_ppl("8192,6,6,kernel").unwrap(),
        GqaPplSpec { ctx: 8192, windows: 6, first: 6, kernel: true, every_row: false, ladder: false }
    );
    assert_eq!(
        parse_gqa_ppl(" 4096 , 12 , 60 , kernel ").unwrap(),
        GqaPplSpec { ctx: 4096, windows: 12, first: 60, kernel: true, every_row: false, ladder: false }
    );
    // Above one prefill call: refused for the first form only.
    assert!(parse_gqa_ppl("8192,6").unwrap_err().contains("4096 rows"));
    // The kernel mode needs a window that reaches past the threshold:
    // position KERNEL_FROM must be an input with a target after it.
    let from = llvq_llm::kvq::KERNEL_FROM;
    let at = |ctx: usize| parse_gqa_ppl(&format!("{ctx},12,0,kernel"));
    assert!(at(from + 1).unwrap_err().contains("never reaches the kernel"));
    assert_eq!(at(from + 2).unwrap().ctx, from + 2);
    assert!(parse_gqa_ppl("1024,12").is_ok(), "the first form has no threshold");
    // kernel0: every row on the kernel, so a short window is the point.
    assert_eq!(
        parse_gqa_ppl("1280,24,40,kernel0").unwrap(),
        GqaPplSpec { ctx: 1280, windows: 24, first: 40, kernel: true, every_row: true, ladder: false }
    );
    assert!(parse_gqa_ppl("1280,24,40,kernel").unwrap_err().contains("never reaches the kernel"));
    assert!(parse_gqa_ppl("1280,24,kernel0").is_err(), "kernel0 needs its first window");
    for bad in ["4096", "4096,12,60", "4096,12,60,grouped", "4096,1", "1,12", "4096,x", "4096,12,60,kernel,1"] {
        assert!(parse_gqa_ppl(bad).is_err(), "{bad} must be refused");
    }
    // ladder: the kernel0 form, two caches.
    assert_eq!(
        parse_gqa_ppl("8192,12,0,ladder").unwrap(),
        GqaPplSpec { ctx: 8192, windows: 12, first: 0, kernel: true, every_row: true, ladder: true }
    );
    assert!(parse_gqa_ppl("8192,12,ladder").is_err(), "ladder needs its first window");
    // The kernel arm reads the Grow store, and is not one of LLVQ_GQA_AB's arms.
    assert_eq!(GqaArm::Kernel.store(), llvq_llm::kvq::KvStore::Grow(llvq_llm::kvq::GROW_CHUNK));
    assert_eq!(GqaArm::Kernel.name(), "grow(256)+kernel");
    assert!(!GqaArm::ALL.contains(&GqaArm::Kernel));
}

#[test]
fn median_is_the_middle_rate_odd_and_even() {
    let (m, lo, hi) = rate_stats(&[3.0, 1.0, 2.0]);
    assert_eq!((m, lo, hi), (2.0, 1.0, 3.0));
    let (m, lo, hi) = rate_stats(&[4.0, 1.0, 2.0, 3.0]);
    assert_eq!((m, lo, hi), (2.5, 1.0, 4.0));
}

/// The comparison refuses arms of different lengths instead of walking
/// their common prefix and printing a tick.
///
/// The hole this closes is real and predates the lot: `zip` truncates, so
/// an arm that returned fewer tokens compared green over the prefix. The
/// assertion is on the *refusal*, because the truncating version passes
/// every "identical tokens" test there is.
#[test]
fn a_short_arm_is_refused_rather_than_compared_on_its_prefix() {
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), Ok(None));
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 9, 3]), Ok(Some(1)));
    assert_eq!(first_divergence(&[9, 2], &[1, 2]), Ok(Some(0)));
    let e = first_divergence(&[1, 2, 3], &[1, 2]).expect_err("unequal lengths");
    assert!(e.contains('3') && e.contains('2'), "{e}");
    assert_eq!(first_divergence(&[], &[]), Ok(None));
}

/// The paired perplexity interval: zero for identical arms, the exact
/// ratio for a constant shift, and the t(1) half-width worked by hand for
/// two windows that disagree.
#[test]
fn paired_ppl_interval() {
    let a = [2.0, 2.1, 2.2, 2.3];
    assert_eq!(paired_ppl_pct(&a, &a), (0.0, 0.0, 0.0));
    let b: Vec<f64> = a.iter().map(|x| x + 0.01).collect();
    let (m, lo, hi) = paired_ppl_pct(&a, &b);
    let want = 0.01f64.exp_m1() * 100.0;
    assert!((m - want).abs() < 1e-9 && (lo - want).abs() < 1e-9 && (hi - want).abs() < 1e-9);
    // d = [0, 0.02]: mean 0.01, sd 0.02 / sqrt(2), half-width 12.706 * 0.01 = 0.12706.
    let (m, lo, hi) = paired_ppl_pct(&[1.0, 1.0], &[1.0, 1.02]);
    assert!((m - want).abs() < 1e-9, "{m}");
    assert!((lo - (0.01f64 - 0.12706).exp_m1() * 100.0).abs() < 1e-6, "{lo}");
    assert!((hi - (0.01f64 + 0.12706).exp_m1() * 100.0).abs() < 1e-6, "{hi}");
    assert_eq!((t95(11), t95(30), t95(31)), (2.201, 2.042, 1.96));
    assert!(t95(0).is_nan());
}
