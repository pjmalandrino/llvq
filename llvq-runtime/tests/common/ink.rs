//! Inkling 2's markers on a byte-level tokenizer, for the tests that need its
//! template and no model: `llvq-llm/tests/inkling_chat.rs` builds the same.

#![allow(dead_code)]

use llvq_llm::inkchat::mark;

pub const MARKS: [&str; 15] = [
    mark::END_OF_TEXT,
    mark::USER,
    mark::MODEL,
    mark::SYSTEM,
    mark::TOOL,
    mark::TEXT,
    mark::IMAGE,
    mark::END_SAMPLING,
    mark::THINKING,
    mark::END_MESSAGE,
    mark::TOOL_ERROR,
    mark::XML,
    mark::INVOKE_JSON,
    mark::INVOKE_TEXT,
    "<|unused_200054|>",
];

/// GPT-2's byte alphabet, one token a byte, and Inkling's markers as special
/// added tokens from id 300.
pub fn byte_tokenizer() -> tokenizers::Tokenizer {
    let chars: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut vocab = serde_json::Map::new();
    let mut extra = 0;
    for b in 0..=255u32 {
        let c = if chars.contains(&b) {
            b
        } else {
            extra += 1;
            255 + extra
        };
        vocab.insert(char::from_u32(c).unwrap().to_string(), b.into());
    }
    let added: Vec<serde_json::Value> = MARKS
        .iter()
        .enumerate()
        .map(|(i, m)| {
            serde_json::json!({"id": 300 + i, "content": m, "single_word": false, "lstrip": false,
                               "rstrip": false, "normalized": false, "special": true})
        })
        .collect();
    let bl = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    let j = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": bl, "post_processor": null, "decoder": bl,
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "dropout": null, "unk_token": null,
                  "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,
                  "byte_fallback": false, "ignore_merges": false}
    });
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&j).unwrap()).unwrap()
}

/// The vocabulary of [`byte_tokenizer`]: the bytes, a gap, the markers.
pub const VOCAB: usize = 300 + MARKS.len();
