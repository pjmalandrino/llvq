"""The teacher is the student's dense twin, or the run is refused."""

import pytest

from llvqtune.domain.loop import WiringError
from llvqtune.domain.pairing import ARCH_KEYS, check_pairing

# The two configs as the Hub ships them (Qwen/Qwen3-8B b968826d, Qwen/Qwen3-4B
# 1cfa9a72). Written out rather than read from a cache, so the test runs
# anywhere.
QWEN3_8B = {
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}
QWEN3_4B = {
    "model_type": "qwen3",
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "intermediate_size": 9728,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
}


def test_the_4b_teacher_is_refused_for_the_8b_student():
    """The default train.sh carried until 2026-09-21."""
    with pytest.raises(WiringError, match="hidden_size: student 4096, teacher 2560"):
        check_pairing(QWEN3_8B, QWEN3_4B)


def test_the_4b_and_the_8b_share_depth_and_vocabulary():
    """Why depth alone is not a guard: it would have let the pair through."""
    assert QWEN3_8B["num_hidden_layers"] == QWEN3_4B["num_hidden_layers"]
    assert QWEN3_8B["vocab_size"] == QWEN3_4B["vocab_size"]


def test_the_twin_is_accepted_whatever_its_storage_dtype():
    """The export says float16 where the Hub says bfloat16: not architecture."""
    export = dict(QWEN3_8B, torch_dtype="float16")
    check_pairing(export, QWEN3_8B)


@pytest.mark.parametrize("key", ARCH_KEYS)
def test_each_architecture_key_is_checked(key):
    """One case per key, so deleting any key from ARCH_KEYS fails one case."""
    teacher = dict(QWEN3_8B)
    teacher[key] = "changed"
    with pytest.raises(WiringError, match=key):
        check_pairing(QWEN3_8B, teacher)


def test_the_two_keys_the_task_names_are_in_the_list():
    assert "hidden_size" in ARCH_KEYS
    assert "num_hidden_layers" in ARCH_KEYS


def test_a_key_present_on_one_side_only_is_a_difference():
    teacher = dict(QWEN3_8B)
    del teacher["intermediate_size"]
    with pytest.raises(WiringError, match="intermediate_size: student 12288, teacher absent"):
        check_pairing(QWEN3_8B, teacher)


def test_a_key_absent_on_both_sides_is_skipped():
    student = {k: v for k, v in QWEN3_8B.items() if k != "num_key_value_heads"}
    check_pairing(student, dict(student))
