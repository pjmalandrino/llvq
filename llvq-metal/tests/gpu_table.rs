//! The GPU class table is load-bearing: the shaders index magnitudes, sign
//! bases and the zero level straight out of it, and a wrong entry decodes
//! plausible wrong weights. Pin every field against the search crate's class
//! metadata — the same source `llvq_artifact::runtime::ClassTable` is built
//! from, but recomputed here rather than copied.

use llvq_search::fastdec::FastDecoder;

#[test]
fn gpu_table_matches_class_metadata() {
    let fd = FastDecoder::new();
    let recs = llvq_metal::gpu_class_table(&fd);
    assert_eq!(recs.len(), fd.n_classes() + 1);

    // Entry 0: the origin.
    assert_eq!(recs[0].len, 1);
    assert_eq!(recs[0].nz, 0);
    assert_eq!(recs[0].vals, [0.0; 5]);

    for ci in 0..fd.n_classes() {
        let lv = fd.levels(ci);
        let r = &recs[ci + 1];
        assert_eq!(r.len as usize, lv.len, "class {ci}: level count");
        assert_eq!(r.nz, lv.nonzero, "class {ci}: nonzero count");

        let norm = ((16 * lv.shell) as f64).sqrt();
        let mut sbase = 0u8;
        for k in 0..lv.len {
            assert_eq!(
                r.counts.get(k).copied().unwrap_or(0),
                if k < 4 { lv.counts[k] } else { 0 },
                "class {ci}: counts[{k}]"
            );
            let want = (lv.values[k] as f64 / norm) as f32;
            assert_eq!(r.vals[k], want, "class {ci}: vals[{k}]");
            if lv.values[k] == 0 {
                assert_eq!(r.zlev as usize, k, "class {ci}: zero level");
            }
            assert_eq!(r.sbase[k], sbase, "class {ci}: sign base of level {k}");
            if lv.values[k] != 0 {
                sbase += lv.counts[k];
            }
        }
        assert_eq!(sbase, lv.nonzero, "class {ci}: bases must sum to nz");
        if lv.values[..lv.len].iter().all(|&v| v != 0) {
            assert_eq!(r.zlev, 255, "class {ci}: no zero level");
        }
    }
}
