//! Qwen3's chat format, built from token ids.
//!
//! Moved out of `bin/chat` on 2026-09-26 so that `bin/gsm8k` puts to the model
//! exactly the prompt the REPL does. Two copies of a template are two
//! templates, and the first place they would drift apart is a scored
//! benchmark, where nothing would say so.

use anyhow::Context;

/// Qwen3's chat markers, resolved from the tokenizer the SEALED FILE carries
/// rather than hard-coded. A wrong id here does not crash: it produces a model
/// that never stops, which is the failure a chat or a generation harness most
/// needs to avoid.
pub struct Marks {
    pub im_start: u32,
    pub im_end: u32,
    pub eot: u32,
    pub nl: u32,
    /// The id of `"\n\n"`, which is ONE token (271) and not two of `nl`.
    /// Qwen3's own template prefills the empty reasoning block with it, and a
    /// model trained on 271 does not read 198 twice as the same thing — which
    /// is exactly what the first attempt got wrong.
    pub nl2: u32,
    /// `<think>` and `</think>`, present on Qwen3 and absent on a model that
    /// does not reason out loud. `Option`, because a caller must not refuse
    /// an artifact for lacking them.
    pub think: Option<(u32, u32)>,
}

impl Marks {
    pub fn resolve(t: &tokenizers::Tokenizer) -> anyhow::Result<Self> {
        let id = |s: &str| -> anyhow::Result<u32> {
            t.token_to_id(s)
                .with_context(|| format!("the tokenizer has no {s:?}; is this a Qwen3 artifact?"))
        };
        let nl = t
            .encode("\n", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .first()
            .copied()
            .context("the tokenizer encodes a newline to nothing")?;
        let nl2 = t
            .encode("\n\n", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .first()
            .copied()
            .context("the tokenizer encodes a blank line to nothing")?;
        Ok(Marks {
            im_start: id("<|im_start|>")?,
            im_end: id("<|im_end|>")?,
            eot: id("<|endoftext|>")?,
            nl,
            nl2,
            think: t
                .token_to_id("<think>")
                .zip(t.token_to_id("</think>")),
        })
    }

    pub fn stops(&self) -> [u32; 2] {
        [self.im_end, self.eot]
    }
}

/// `<|im_start|>{role}\n{body}<|im_end|>\n`, built from ids.
///
/// Built from ids and not from a formatted string: the string form would go
/// back through the tokenizer, and a BPE merge across a marker boundary would
/// silently produce a different prompt than the one intended.
pub fn turn(
    t: &tokenizers::Tokenizer,
    m: &Marks,
    role: &str,
    body: &str,
) -> anyhow::Result<Vec<u32>> {
    let mut v = vec![m.im_start];
    v.extend(
        t.encode(role, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids(),
    );
    v.push(m.nl);
    v.extend(
        t.encode(body, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids(),
    );
    v.push(m.im_end);
    v.push(m.nl);
    Ok(v)
}

/// The assistant's opening, which carries no body: the model writes it.
///
/// With `think` false the opening is PRE-FILLED with an empty reasoning block,
/// `<think>\n\n</think>\n\n`, which is how Qwen3 is told not to reason out
/// loud. It is not a stop token and not a filter: the block is closed before
/// the model writes a word, so there is nothing to strip afterwards and the
/// budget goes to the answer.
///
/// Off by default in `bin/chat`. A 48-token budget on the first smoke went
/// entirely into the model thinking about the question, which is correct
/// behaviour and a useless chat.
pub fn open_assistant(
    t: &tokenizers::Tokenizer,
    m: &Marks,
    think: bool,
) -> anyhow::Result<Vec<u32>> {
    let mut v = vec![m.im_start];
    v.extend(
        t.encode("assistant", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids(),
    );
    v.push(m.nl);
    if !think {
        if let Some((open, close)) = m.think {
            v.push(open);
            v.push(m.nl2);
            v.push(close);
            v.push(m.nl2);
        }
    }
    Ok(v)
}
