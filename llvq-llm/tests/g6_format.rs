//! The one format test that genuinely needs the model side: the rotation seed
//! is derived from a block index and an activation kind, both of which live in
//! `llvq-llm`. Everything else about the format is tested in `llvq-artifact`,
//! which has no dependencies.

/// The rotation seed must separate every (block, activation) pair.
///
/// A mutation test showed that storing the run's *base* seed instead of the
/// effective one survived the whole suite — nothing forbade it. The formula
/// now lives in one function used by both the builder and the writer, so the
/// two cannot drift; this pins what that function must guarantee.
#[test]
fn the_rotation_seed_separates_every_block_and_activation() {
    use llvq_llm::calib::effective_rotation_seed;
    use llvq_llm::model::Act;
    let base = 0x11_0FEED;
    let mut seen = std::collections::HashSet::new();
    let mut same_as_base = 0usize;
    for block in 0..64usize {
        for act in Act::ALL {
            let s = effective_rotation_seed(base, block, act);
            assert!(
                seen.insert(s),
                "block {block} / {act:?} collides with an earlier pair — two \
                 matrices would share a rotation, and the artifact could not \
                 tell them apart"
            );
            if s == base {
                same_as_base += 1;
            }
        }
    }
    // The XOR is nil for the very first pair, so exactly one (block,
    // activation) legitimately equals the base seed. More than one would mean
    // the block or activation term is not being mixed in at all — which is
    // precisely the defect a mutation test found nothing forbidding.
    assert_eq!(
        same_as_base, 1,
        "{same_as_base} pairs return the base seed; only (block 0, first \
         activation) may"
    );
}
