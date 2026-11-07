//! A request's messages as ids, and the text streamed back.
//!
//! An HTTP chat is stateless: every request carries the whole history. The
//! engine keeps the cache of the last conversation, so the cost of a turn is
//! its new tokens only if the history renders to the ids the cache holds.
//! Rendering an assistant message from its text does not give those ids: the
//! cache holds the ids the model wrote, with the empty reasoning block of the
//! opening, and a byte-level BPE does not always encode a decoded text back to
//! the same ids. So the engine remembers the segments of the last
//! conversation ([`Segment`]) and [`render`] reuses a segment's ids when the
//! message at that place is the same role and the same text. The first message
//! that differs, and every one after it, is rendered by `chatfmt`.

use llvq_llm::chatfmt::{turn, Marks};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "system" | "developer" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "tool" => Err("role \"tool\": tool calls are not served by llvq".to_string()),
            o => Err(format!("role {o:?}: the roles are system, user and assistant")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Msg {
    pub role: Role,
    pub content: String,
}

/// A run of ids in the cache, and the message it renders when there is one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub msg: Option<Msg>,
    pub ids: Vec<u32>,
}

/// The ids of `msgs`, one segment a message, reusing `memo` while it agrees.
pub fn render(
    tok: &tokenizers::Tokenizer,
    marks: &Marks,
    memo: &[Segment],
    msgs: &[Msg],
) -> anyhow::Result<Vec<Segment>> {
    let mut out = Vec::with_capacity(msgs.len());
    let mut agree = true;
    for (i, m) in msgs.iter().enumerate() {
        agree = agree && memo.get(i).is_some_and(|s| s.msg.as_ref() == Some(m));
        let ids = match agree {
            true => memo[i].ids.clone(),
            false => turn(tok, marks, m.role.label(), &m.content)?,
        };
        out.push(Segment { msg: Some(m.clone()), ids });
    }
    Ok(out)
}

/// What a growing text may stream now, against the stop strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scan {
    /// No stop string yet: the first `n` bytes are safe to send. The rest
    /// could be the start of a stop string, or a character still arriving.
    Safe(usize),
    /// A stop string starts at this byte: the answer ends there.
    Stop(usize),
}

/// Scan `text` for the first stop string, and otherwise hold back the longest
/// tail that could still grow into one, and any trailing U+FFFD (a
/// character split across tokens, see `repl::printable`).
pub fn scan(text: &str, stops: &[String]) -> Scan {
    if let Some(at) = stops.iter().filter(|s| !s.is_empty()).filter_map(|s| text.find(s.as_str())).min() {
        return Scan::Stop(at);
    }
    let mut safe = llvq_llm::repl::printable(text).len();
    for s in stops.iter().filter(|s| !s.is_empty()) {
        for (k, _) in s.char_indices().skip(1) {
            if text.ends_with(&s[..k]) {
                safe = safe.min(text.len() - k);
            }
        }
    }
    Scan::Safe(safe)
}

/// One stream of text decoded from ids that arrive a few at a time: the
/// answer, or its reasoning.
///
/// [`TextStream::full`] is the text of every id, as `tokenizers` decodes the
/// whole list, but it decodes only the ids since the last whole character:
/// decoding the whole answer at every token cost 73 ns a token already said,
/// 0.6 ms a token at 8,192 (*measured*, `docs/runtime-audit-2026-11.md`).
#[derive(Default)]
pub struct TextStream {
    pub ids: Vec<u32>,
    /// What was sent so far.
    pub sent: String,
    /// The text of `ids[..read]`, which ends on a whole character.
    done: String,
    /// The ids the next decode starts from, `ids[from..read]`, and their
    /// text decoded alone. A window that starts a few ids back gives a
    /// decoder the context it would have had in the whole list (the space
    /// a word-level decoder puts between two words, for one).
    from: usize,
    read: usize,
    head: String,
}

impl TextStream {
    /// The text of every id so far, as `tokenizers` decodes the whole list.
    ///
    /// Exact for a byte-level decoder (Qwen3's): the text of a list of byte
    /// strings is the text of its first part followed by the text of the
    /// rest whenever the first part ends on a whole character, and `done`
    /// grows only then, when the new text does not end with U+FFFD. A
    /// decoder whose window text does not start with its context text falls
    /// back to the whole list.
    pub fn full(&mut self, tok: &tokenizers::Tokenizer) -> anyhow::Result<String> {
        let decode = |ids: &[u32]| tok.decode(ids, false).map_err(|e| anyhow::anyhow!("{e}"));
        let window = decode(&self.ids[self.from..])?;
        let Some(tail) = window.strip_prefix(self.head.as_str()) else {
            return decode(&self.ids);
        };
        let mut full = String::with_capacity(self.done.len() + tail.len());
        full.push_str(&self.done);
        full.push_str(tail);
        if !tail.is_empty() && !tail.ends_with('\u{FFFD}') {
            self.done.push_str(tail);
            self.from = self.read;
            self.read = self.ids.len();
            self.head = decode(&self.ids[self.from..self.read])?;
        }
        Ok(full)
    }

    /// The part of `full[..upto]` not yet sent, and mark it sent. Nothing
    /// when the decoded text no longer starts with what was sent: the stream
    /// never rewinds, it waits.
    pub fn take(&mut self, full: &str, upto: usize) -> String {
        let mut upto = upto.min(full.len());
        while !full.is_char_boundary(upto) {
            upto -= 1;
        }
        if upto <= self.sent.len() || !full.starts_with(self.sent.as_str()) {
            return String::new();
        }
        let fresh = full[self.sent.len()..upto].to_string();
        self.sent.push_str(&fresh);
        fresh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stop_string_cuts_at_its_first_occurrence() {
        let stops = vec!["END".to_string(), "\n\n".to_string()];
        assert_eq!(scan("hello END there", &stops), Scan::Stop(6));
        assert_eq!(scan("a\n\nb END", &stops), Scan::Stop(1));
        assert_eq!(scan("plain", &stops), Scan::Safe(5));
    }

    #[test]
    fn a_tail_that_could_become_a_stop_string_is_held() {
        let stops = vec!["END".to_string()];
        assert_eq!(scan("hello E", &stops), Scan::Safe(6));
        assert_eq!(scan("hello EN", &stops), Scan::Safe(6));
        assert_eq!(scan("hello ENx", &stops), Scan::Safe(9));
        // The replacement character of a split character is held too.
        assert_eq!(scan("ab\u{FFFD}", &[]), Scan::Safe(2));
        // Empty stop strings stop nothing.
        assert_eq!(scan("abc", &[String::new()]), Scan::Safe(3));
    }

    #[test]
    fn a_multibyte_stop_string_is_held_on_character_boundaries() {
        let stops = vec!["é!".to_string()];
        assert_eq!(scan("caf\u{e9}", &stops), Scan::Safe(3));
        assert_eq!(scan("caf\u{e9}!", &stops), Scan::Stop(3));
    }

    #[test]
    fn the_stream_sends_each_byte_once_and_never_rewinds() {
        let mut s = TextStream::default();
        assert_eq!(s.take("Hel", 3), "Hel");
        assert_eq!(s.take("Hello", 5), "lo");
        assert_eq!(s.take("Hello", 5), "");
        // A decode that changed what was sent waits instead of rewinding.
        assert_eq!(s.take("Help!", 5), "");
        assert_eq!(s.take("Hello!", 6), "!");
        assert_eq!(s.sent, "Hello!");
        // Never inside a character.
        let mut s = TextStream::default();
        assert_eq!(s.take("a\u{e9}", 2), "a");
        assert_eq!(s.take("a\u{e9}", 3), "\u{e9}");
    }

    /// GPT-2's byte-level alphabet: one token a byte, so that a character
    /// of two to four bytes is split across ids, as Qwen3's BPE splits rare
    /// characters, and an invalid byte decodes to U+FFFD.
    fn byte_tokenizer() -> tokenizers::Tokenizer {
        let mut chars: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
        let mut vocab = serde_json::Map::new();
        let mut extra = 0;
        for b in 0..=255u32 {
            let c = match chars.contains(&b) {
                true => b,
                false => {
                    extra += 1;
                    255 + extra
                }
            };
            vocab.insert(char::from_u32(c).unwrap().to_string(), b.into());
        }
        chars.clear();
        let j = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
            "model": {"type": "BPE", "vocab": vocab, "merges": [], "dropout": null, "unk_token": null,
                      "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,
                      "byte_fallback": false, "ignore_merges": false}
        });
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&j).unwrap()).unwrap()
    }

    #[test]
    fn the_stream_decodes_as_the_whole_list_at_every_id() {
        let tok = byte_tokenizer();
        assert_eq!(tok.decode(&[0xC3, 0xA9], false).unwrap(), "\u{e9}", "the test tokenizer is byte-level");
        let mut texts: Vec<Vec<u32>> = vec![
            "caf\u{e9} \u{1F600} \u{4E2D}\u{6587} plain".bytes().map(u32::from).collect(),
            // A lone continuation byte, a truncated character, then text.
            vec![0x61, 0x80, 0x62, 0xE2, 0x82, 0x63, 0xF0, 0x9F, 0x98, 0x80],
            // The replacement character itself, written by the model.
            "a\u{FFFD}b\u{FFFD}".bytes().map(u32::from).collect(),
        ];
        let mut state = 0x5EEDu64;
        for _ in 0..40 {
            let n = 1 + (state % 60) as usize;
            texts.push(
                (0..n)
                    .map(|_| {
                        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                        // Mostly ASCII, with lead and continuation bytes.
                        match state >> 61 {
                            0 => 0xC0 | (state >> 32) as u32 % 0x30,
                            1 => 0x80 | (state >> 32) as u32 % 0x40,
                            _ => 0x20 + (state >> 32) as u32 % 0x5F,
                        }
                    })
                    .collect(),
            );
        }
        for ids in texts {
            for step in [1usize, 3] {
                let mut s = TextStream::default();
                for part in ids.chunks(step) {
                    s.ids.extend_from_slice(part);
                    assert_eq!(s.full(&tok).unwrap(), tok.decode(&s.ids, false).unwrap(), "ids {:?}", s.ids);
                    // And it stays incremental: the fallback to the whole list
                    // would give the same text at a cost that grows with it.
                    assert!(s.ids.len() - s.from <= 16, "a window of {} ids at {:?}", s.ids.len() - s.from, s.ids);
                }
            }
        }
    }

    #[test]
    fn roles_refuse_tools_by_name() {
        assert_eq!(Role::parse("user"), Ok(Role::User));
        assert!(Role::parse("tool").unwrap_err().contains("tool"));
        assert!(Role::parse("bot").is_err());
    }
}
