//! The dependency-free JSON reader the whole-model accounting uses on a
//! safetensors header; it was an inline module and is the same module here.
#[derive(Debug, Clone)]
pub enum Value {
    Null,
    /// A boolean; its value is not read by this bench.
    Bool,
    Num(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Num(x) if *x >= 0.0 && x.fract() == 0.0 => Some(*x as u64),
            _ => None,
        }
    }
}

pub fn parse(s: &str) -> Result<Value, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let v = value(b, &mut i)?;
    ws(b, &mut i);
    if i != b.len() {
        return Err(format!("trailing data at byte {i}"));
    }
    Ok(v)
}

fn ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn value(b: &[u8], i: &mut usize) -> Result<Value, String> {
    ws(b, i);
    match b.get(*i) {
        Some(b'{') => {
            *i += 1;
            let mut kv = Vec::new();
            ws(b, i);
            if b.get(*i) == Some(&b'}') {
                *i += 1;
                return Ok(Value::Object(kv));
            }
            loop {
                ws(b, i);
                let k = string(b, i)?;
                ws(b, i);
                if b.get(*i) != Some(&b':') {
                    return Err(format!("expected ':' at byte {i}"));
                }
                *i += 1;
                kv.push((k, value(b, i)?));
                ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b'}') => {
                        *i += 1;
                        return Ok(Value::Object(kv));
                    }
                    _ => return Err(format!("expected ',' or '}}' at byte {i}")),
                }
            }
        }
        Some(b'[') => {
            *i += 1;
            let mut a = Vec::new();
            ws(b, i);
            if b.get(*i) == Some(&b']') {
                *i += 1;
                return Ok(Value::Array(a));
            }
            loop {
                a.push(value(b, i)?);
                ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b']') => {
                        *i += 1;
                        return Ok(Value::Array(a));
                    }
                    _ => return Err(format!("expected ',' or ']' at byte {i}")),
                }
            }
        }
        Some(b'"') => Ok(Value::Str(string(b, i)?)),
        Some(b't') if b[*i..].starts_with(b"true") => {
            *i += 4;
            Ok(Value::Bool)
        }
        Some(b'f') if b[*i..].starts_with(b"false") => {
            *i += 5;
            Ok(Value::Bool)
        }
        Some(b'n') if b[*i..].starts_with(b"null") => {
            *i += 4;
            Ok(Value::Null)
        }
        Some(_) => {
            let start = *i;
            while *i < b.len() && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                *i += 1;
            }
            std::str::from_utf8(&b[start..*i])
                .ok()
                .and_then(|t| t.parse::<f64>().ok())
                .map(Value::Num)
                .ok_or_else(|| format!("bad value at byte {start}"))
        }
        None => Err("unexpected end".into()),
    }
}

fn string(b: &[u8], i: &mut usize) -> Result<String, String> {
    if b.get(*i) != Some(&b'"') {
        return Err(format!("expected a string at byte {i}"));
    }
    *i += 1;
    let mut out = String::new();
    loop {
        match b.get(*i) {
            Some(b'"') => {
                *i += 1;
                return Ok(out);
            }
            Some(b'\\') => {
                let c = *b.get(*i + 1).ok_or("unterminated escape")?;
                *i += 2;
                match c {
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'u' => {
                        let h = std::str::from_utf8(b.get(*i..*i + 4).ok_or("short \\u")?)
                            .map_err(|_| "bad \\u")?;
                        let cp = u32::from_str_radix(h, 16).map_err(|_| "bad \\u")?;
                        out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        *i += 4;
                    }
                    other => out.push(other as char),
                }
            }
            Some(_) => {
                // A whole UTF-8 sequence at once.
                let start = *i;
                *i += 1;
                while *i < b.len() && (b[*i] & 0xC0) == 0x80 {
                    *i += 1;
                }
                out.push_str(std::str::from_utf8(&b[start..*i]).map_err(|_| "bad UTF-8")?);
            }
            None => return Err("unterminated string".into()),
        }
    }
}
