//! Prompt files for `bin/specbench`: JSONL, one prompt a line,
//! `{"id": .., "kind": "chat"|"code"|"math"|"rag"|"other", "prompt": ".."}` or
//! `{"id": .., "kind": .., "messages": [{"role": "user"|"system"|"assistant",
//! "content": ".."}, ..]}`. A line without `id` (a `{"_comment": ..}` header)
//! and a blank line are skipped. The set `ops/data/spec-prompts-v1.jsonl` is
//! one such file.

use serde_json::Value;

/// The kinds a prompt may carry, in the order aggregates print them.
pub const KINDS: [&str; 5] = ["chat", "code", "math", "rag", "other"];

/// One prompt: its id, its kind and its messages, `(role, content)`, in
/// order; the chat template is applied by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptRec {
    pub id: String,
    pub kind: String,
    pub messages: Vec<(String, String)>,
}

/// Every prompt of a JSONL text, in file order. Refuses a line that is not
/// a JSON object, a record without `kind` or with an unknown one, a record
/// with neither or both of `prompt` and `messages`, an unknown role, and a
/// repeated id, naming the line.
pub fn parse(text: &str) -> Result<Vec<PromptRec>, String> {
    let mut out: Vec<PromptRec> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let n = i + 1;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {n}: {e}"))?;
        let o = v.as_object().ok_or_else(|| format!("line {n}: not a JSON object"))?;
        let Some(id) = o.get("id") else { continue };
        let id = id.as_str().ok_or_else(|| format!("line {n}: id is not a string"))?.to_string();
        let kind = o.get("kind").and_then(Value::as_str).ok_or_else(|| format!("line {n} ({id}): no kind"))?.to_string();
        if !KINDS.contains(&kind.as_str()) {
            return Err(format!("line {n} ({id}): kind {kind:?}, one of {KINDS:?}"));
        }
        let messages = match (o.get("prompt"), o.get("messages")) {
            (Some(p), None) => vec![("user".to_string(), p.as_str().ok_or_else(|| format!("line {n} ({id}): prompt is not a string"))?.to_string())],
            (None, Some(m)) => {
                let arr = m.as_array().ok_or_else(|| format!("line {n} ({id}): messages is not an array"))?;
                let mut v = Vec::with_capacity(arr.len());
                for x in arr {
                    let role = x.get("role").and_then(Value::as_str).ok_or_else(|| format!("line {n} ({id}): a message without role"))?;
                    let content = x.get("content").and_then(Value::as_str).ok_or_else(|| format!("line {n} ({id}): a message without content"))?;
                    if !["system", "user", "assistant"].contains(&role) {
                        return Err(format!("line {n} ({id}): role {role:?}"));
                    }
                    v.push((role.to_string(), content.to_string()));
                }
                if v.is_empty() {
                    return Err(format!("line {n} ({id}): no message"));
                }
                v
            }
            _ => return Err(format!("line {n} ({id}): one of prompt and messages")),
        };
        if out.iter().any(|p| p.id == id) {
            return Err(format!("line {n}: id {id} repeated"));
        }
        out.push(PromptRec { id, kind, messages });
    }
    if out.is_empty() {
        return Err("no prompt".into());
    }
    Ok(out)
}

/// A histogram of accepted counts 0..=15 a pass.
pub fn histogram(accepted: &[usize]) -> [usize; 16] {
    let mut h = [0usize; 16];
    for &a in accepted {
        h[a.min(15)] += 1;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_file_parses_and_refuses() {
        let text = "{\"_comment\": \"a house set\"}\n\n{\"id\": \"a\", \"kind\": \"math\", \"prompt\": \"2+2?\"}\n\
                    {\"id\": \"b\", \"kind\": \"rag\", \"messages\": [{\"role\": \"system\", \"content\": \"s\"}, {\"role\": \"user\", \"content\": \"u\"}]}\n";
        let p = parse(text).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].messages, vec![("user".to_string(), "2+2?".to_string())]);
        assert_eq!(p[1].kind, "rag");
        assert_eq!(p[1].messages.len(), 2);
        for (bad, why) in [
            ("{\"id\": \"a\", \"kind\": \"poetry\", \"prompt\": \"x\"}", "kind"),
            ("{\"id\": \"a\", \"kind\": \"chat\"}", "one of"),
            ("{\"id\": \"a\", \"kind\": \"chat\", \"prompt\": \"x\", \"messages\": []}", "one of"),
            ("{\"id\": \"a\", \"kind\": \"chat\", \"messages\": [{\"role\": \"tool\", \"content\": \"x\"}]}", "role"),
            ("{\"id\": \"a\", \"kind\": \"chat\", \"prompt\": \"x\"}\n{\"id\": \"a\", \"kind\": \"chat\", \"prompt\": \"y\"}", "repeated"),
            ("not json", "line 1"),
            ("{\"_comment\": \"only\"}", "no prompt"),
        ] {
            let e = parse(bad).unwrap_err();
            assert!(e.contains(why), "{bad}: {e}");
        }
        assert_eq!(histogram(&[0, 3, 3, 20])[3], 2);
        assert_eq!(histogram(&[0, 3, 3, 20])[15], 1);
    }

    #[test]
    fn the_house_set_parses() {
        let text = include_str!("../../ops/data/spec-prompts-v1.jsonl");
        let p = parse(text).unwrap();
        let count = |k: &str| p.iter().filter(|r| r.kind == k).count();
        assert_eq!((p.len(), count("math"), count("code"), count("chat"), count("rag")), (44, 20, 10, 10, 4));
    }
}
