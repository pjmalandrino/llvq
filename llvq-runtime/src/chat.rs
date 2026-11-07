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
#[derive(Default)]
pub struct TextStream {
    pub ids: Vec<u32>,
    /// What was sent so far.
    pub sent: String,
}

impl TextStream {
    /// The text of every id so far, as `tokenizers` decodes it.
    pub fn full(&self, tok: &tokenizers::Tokenizer) -> anyhow::Result<String> {
        tok.decode(&self.ids, false).map_err(|e| anyhow::anyhow!("{e}"))
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

    #[test]
    fn roles_refuse_tools_by_name() {
        assert_eq!(Role::parse("user"), Ok(Role::User));
        assert!(Role::parse("tool").unwrap_err().contains("tool"));
        assert!(Role::parse("bot").is_err());
    }
}
