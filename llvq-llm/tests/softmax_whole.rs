//! The `LLVQ_SOFTMAX_WHOLE` switch, as `bin/fusedrun` reads it.
//!
//! The library's own tests replace the switch by a per-thread knob, so they
//! cannot reach it. An integration test links the library built without
//! `cfg(test)`, and therefore the process-wide atomic `bin/fusedrun` sets.
//! The atomic is global, so this file holds a single test and is its own
//! test binary.

use llvq_llm::model::{set_softmax_whole_for_measurement, softmax_form};

#[test]
fn the_softmax_whole_switch_moves_the_form_and_back() {
    assert_eq!(softmax_form(), "in place", "the default form");
    set_softmax_whole_for_measurement(true);
    assert_eq!(softmax_form(), "whole tensor (LLVQ_SOFTMAX_WHOLE=1)");
    set_softmax_whole_for_measurement(false);
    assert_eq!(softmax_form(), "in place", "the switch goes back");
}
