//! The checkpoint side: just enough JSON to read a safetensors header, the BF16
//! tensor reader on top of it, and the resolution of a Hub id in the local cache.

use super::*;

// ---------------------------------------------------------------------------
// Just enough JSON, and just enough safetensors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Json {
    Null,
    /// The value is dropped: nothing this bench reads from a safetensors
    /// header or a shard index is a boolean, and a field no one reads is dead
    /// weight the compiler is right to flag.
    Bool,
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(n) => Some(*n as usize),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match *self.b.get(self.i).ok_or("end of input")? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => self.lit("true", Json::Bool),
            b'f' => self.lit("false", Json::Bool),
            b'n' => self.lit("null", Json::Null),
            _ => self.number(),
        }
    }
    fn lit(&mut self, s: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            Err(format!("expected {s} at byte {}", self.i))
        }
    }
    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self.i < self.b.len() && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .map(Json::Num)
            .ok_or_else(|| format!("bad number at byte {start}"))
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // the opening quote
        let mut out = String::new();
        loop {
            let c = *self.b.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let hex = std::str::from_utf8(&self.b[self.i..self.i + 4])
                                .map_err(|_| "bad \\u")?;
                            let cp = u32::from_str_radix(hex, 16).map_err(|_| "bad \\u")?;
                            self.i += 4;
                            out.push(char::from_u32(cp).ok_or("bad code point")?);
                        }
                        other => out.push(other as char),
                    }
                }
                _ => {
                    // Multi-byte UTF-8 passes through one byte at a time.
                    let start = self.i - 1;
                    let n = match c {
                        0x00..=0x7f => 1,
                        0xc0..=0xdf => 2,
                        0xe0..=0xef => 3,
                        _ => 4,
                    };
                    self.i = start + n;
                    out.push_str(std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad utf8")?);
                }
            }
        }
    }
    fn array(&mut self) -> Result<Json, String> {
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(out));
        }
        loop {
            out.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(out));
                }
                _ => return Err(format!("expected , or ] at byte {}", self.i)),
            }
        }
    }
    fn object(&mut self) -> Result<Json, String> {
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(out));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            if self.b.get(self.i) != Some(&b':') {
                return Err(format!("expected : at byte {}", self.i));
            }
            self.i += 1;
            let v = self.value()?;
            out.push((k, v));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(out));
                }
                _ => return Err(format!("expected , or }} at byte {}", self.i)),
            }
        }
    }
}

fn parse_json(b: &[u8]) -> Result<Json, String> {
    Parser { b, i: 0 }.value()
}

/// Where one tensor lives: which shard, which byte range, what shape.
struct TensorEntry {
    shard: usize,
    start: u64,
    end: u64,
    shape: Vec<usize>,
}

/// A checkpoint read without a tensor runtime: the safetensors header is JSON
/// and the payload is a byte range, which is all this bench needs. BF16 only,
/// refused loudly otherwise — a silent f16-as-bf16 read would move every ρ.
pub(super) struct Checkpoint {
    shards: Vec<PathBuf>,
    index: HashMap<String, TensorEntry>,
}

impl Checkpoint {
    pub(super) fn open(dir: &Path) -> Result<Self, String> {
        let mut shards: Vec<PathBuf> = Vec::new();
        let single = dir.join("model.safetensors");
        if single.exists() {
            shards.push(single);
        } else {
            let mut names: Vec<String> = std::fs::read_dir(dir)
                .map_err(|e| format!("{}: {e}", dir.display()))?
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".safetensors"))
                .collect();
            names.sort();
            shards.extend(names.into_iter().map(|n| dir.join(n)));
        }
        if shards.is_empty() {
            return Err(format!("no safetensors under {}", dir.display()));
        }
        let mut index = HashMap::new();
        for (si, path) in shards.iter().enumerate() {
            let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut n = [0u8; 8];
            f.read_exact(&mut n).map_err(|e| e.to_string())?;
            let hlen = u64::from_le_bytes(n) as usize;
            let mut hb = vec![0u8; hlen];
            f.read_exact(&mut hb).map_err(|e| e.to_string())?;
            let head = parse_json(&hb)?;
            let Json::Obj(kv) = &head else {
                return Err("safetensors header is not an object".into());
            };
            for (name, v) in kv {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = v.get("dtype").and_then(Json::as_str).unwrap_or("");
                if dtype != "BF16" {
                    return Err(format!("{name}: dtype {dtype}, this bench reads BF16 only"));
                }
                let off = v
                    .get("data_offsets")
                    .and_then(Json::as_arr)
                    .ok_or("no data_offsets")?;
                let shape: Vec<usize> = v
                    .get("shape")
                    .and_then(Json::as_arr)
                    .ok_or("no shape")?
                    .iter()
                    .filter_map(Json::as_usize)
                    .collect();
                let base = 8 + hlen as u64;
                index.insert(
                    name.clone(),
                    TensorEntry {
                        shard: si,
                        start: base + off[0].as_usize().ok_or("bad offset")? as u64,
                        end: base + off[1].as_usize().ok_or("bad offset")? as u64,
                        shape,
                    },
                );
            }
        }
        Ok(Self { shards, index })
    }

    /// One tensor as f32. BF16 widens to f32 exactly (the low 16 bits are
    /// zero), so nothing is lost between the file and the inner products.
    pub(super) fn tensor(&self, name: &str, d_out: usize, d_in: usize) -> Result<Vec<f32>, String> {
        let e = self
            .index
            .get(name)
            .ok_or_else(|| format!("{name}: absent from the checkpoint"))?;
        if e.shape != [d_out, d_in] {
            return Err(format!(
                "{name}: checkpoint shape {:?}, artifact {}x{}",
                e.shape, d_out, d_in
            ));
        }
        let mut f = File::open(&self.shards[e.shard]).map_err(|x| x.to_string())?;
        f.seek(SeekFrom::Start(e.start)).map_err(|x| x.to_string())?;
        let n = (e.end - e.start) as usize;
        if n != d_out * d_in * 2 {
            return Err(format!("{name}: {n} bytes for {} bf16", d_out * d_in));
        }
        let mut raw = vec![0u8; n];
        f.read_exact(&mut raw).map_err(|x| x.to_string())?;
        Ok(raw
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect())
    }
}

/// A local directory, or a Hub repo id resolved in the local cache. No
/// download: a bench that silently fetches 8 GB is a bench that costs money.
pub(super) fn resolve_checkpoint(spec: &str) -> Result<PathBuf, String> {
    let direct = PathBuf::from(spec);
    if direct.is_dir() {
        return Ok(direct);
    }
    let (org, name) = spec
        .split_once('/')
        .ok_or_else(|| format!("{spec}: neither a directory nor an org/name repo id"))?;
    let hub = match std::env::var("HF_HOME") {
        Ok(h) => PathBuf::from(h).join("hub"),
        Err(_) => PathBuf::from(std::env::var("HOME").map_err(|e| e.to_string())?)
            .join(".cache/huggingface/hub"),
    };
    let repo = hub.join(format!("models--{org}--{name}"));
    let head = repo.join("refs/main");
    if let Ok(sha) = std::fs::read_to_string(&head) {
        let d = repo.join("snapshots").join(sha.trim());
        if d.is_dir() {
            return Ok(d);
        }
    }
    let mut snaps: Vec<PathBuf> = std::fs::read_dir(repo.join("snapshots"))
        .map_err(|e| format!("{}: {e}", repo.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    snaps.sort();
    snaps
        .pop()
        .ok_or_else(|| format!("{}: no snapshot in the cache", repo.display()))
}
