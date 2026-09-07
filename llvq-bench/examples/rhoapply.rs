//! M1 — the radial correction applied to a finished artifact, both arms.
//!
//! The preregistration is `proofs/preregistration-m1-rho-2026-09-07.md`
//! (sha256 `ac83cdc2…`, timestamped before this file existed). It fixes the
//! two arms, the six controls and the decision rule. **This bench decides
//! nothing**: it writes two artifacts and prints numbers.
//!
//! ```text
//!   ρ_i      = ⟨w_i , r_i⟩ / ⟨r_i , r_i⟩          (prereg §2, verbatim)
//!   ρ_global = Σ_i ⟨w_i, r_i⟩ / Σ_i ⟨r_i, r_i⟩
//!   arm A: every row scale × ρ_global
//!   arm B: row scale i × ρ_i
//! ```
//!
//! `w_i` is the checkpoint row, `r_i` the row [`llvq_artifact::decode_matrix`]
//! rebuilds — both in the **natural** basis, since the decoder undoes the
//! incoherence rotation on its way out.
//!
//! ## What the bench establishes before it computes anything
//!
//! The whole design rests on one claim: multiplying `row_scales[i]` by ρ
//! multiplies row `i` of the decoded matrix by ρ, in the natural basis, and
//! touches no other row. Step 1 proves it on the real file rather than
//! asserting it, and the proof turns up the one qualifier the prereg's formula
//! does not carry:
//!
//! **The tail is not scaled.** A record keeps `d_in % 24` trailing columns
//! unquantized (`tail`), and [`llvq_quant::quantizer::reconstruct_shape_gain`]
//! — the only consumer of `row_scales` — never sees them. So the decoded row
//! is *affine* in ρ, not linear:
//!
//! ```text
//!   r_i(ρ) = ρ·B_i + T_i        B_i = the blocks' image, T_i = the tail's
//! ```
//!
//! On Qwen3-4B every matrix has a tail (2560 % 24 = 16, 4096 % 24 = 16,
//! 9728 % 24 = 8), so `T_i` is never zero. Rows stay independent — the
//! rotation is applied per row (`Rotation::unrotate_weight_rows`) — so the
//! bench is sound; but ρ as the prereg defines it is not exactly the minimizer
//! of `‖ΔW‖²` over the parameter that actually moves. The bias is
//! `(1−ρ*)·‖T‖²/‖r‖²`, and the bench prints both the prereg's ρ (which is what
//! the two arms are written with, because the prereg is timestamped) and the
//! exact optimum `ρ̃_i = ⟨w_i − T_i, B_i⟩ / ⟨B_i, B_i⟩`, so the gap is on the
//! record instead of in a footnote.
//!
//! Because `r_i(ρ)` is affine, every `‖ΔW‖²` below is computed exactly from
//! `(‖u‖², ⟨u,B⟩, ‖B‖²)` with `u = w − T`, not from a re-decode.
//!
//! ## What two adversarial reviews changed here, and what they did not
//!
//! Both reviews attack the same joint, and they are right about it: the arms
//! minimise `‖ΔW‖²`, the encoder minimises `Tr(ΔW H ΔWᵀ)`, and the prereg says
//! so in one sentence (§7). The quantities that turn that sentence into
//! numbers are cheap, so they are computed and printed rather than left as a
//! caveat. Nothing in the two arms changed: the prereg is timestamped, and the
//! files it fixes are the files that go to MMLU.
//!
//! * **"`ρ_global` is not the radial bias."** Taken, and answered by
//!   measurement instead of subtraction. `ρ_i = L_i · c_i` exactly, row by
//!   row: a length ratio `‖w‖/‖r‖` and a cosine. The cosine is the quantity M0
//!   simulated; the length ratio is the one M0 could not see. Where the review
//!   computes the metric-blind share as `δ̄/δ_I`, it is *assuming* `δ_H = δ̄` —
//!   giving an unmeasured quantity a measured one's value. That is printed as
//!   a scenario, under that word, with the split it implies by type.
//! * **"`δ_H` is not measured and its sign is not pinned."** Correct, and now
//!   printed with its bound: `δ_H = δ_I − ⟨e_⊥ H rᵀ⟩/⟨r H rᵀ⟩`, the transverse
//!   ratio, and the cosine the cross term must stay under for arm A to help.
//!   This is the honest form of the point above, and it *contradicts* it: a
//!   review cannot call `δ_H` free over a range ten times the stake and also
//!   assign it a value to three decimals. Both are printed, adjacent, so the
//!   operator sees the tension rather than one of its halves.
//! * **"Arm B is mis-named."** Refused as stated. The prereg's arm B is per
//!   **row**, not per type; the decomposition the review supplies puts 57 % of
//!   the variance *inside* one matrix, which is per-row structure and is
//!   precisely what §2 went looking for. What the decomposition does show is
//!   that 43 % of it is reachable with 252 numbers, so a per-matrix arm C is
//!   computed — in `‖ΔW‖²` only. It is **not written**: a third artifact is a
//!   third MMLU job, and that is an operator decision, not a bench's.
//! * **"80 % of the dispersion is a tail artifact."** The direction is right —
//!   `sd(ρ̃) < sd(ρ)` was already printed — but the arithmetic is not: the two
//!   are strongly correlated, so `1 − (sd ρ̃ / sd ρ)²` is not the share of
//!   anything. The correlation and `sd(ρ − ρ̃)` are printed instead.
//! * **Clipping.** A clipped arm B′, `ρ_i ≥ 1 − 2δ̄`, is computed for the same
//!   reason as arm C, and for the same reason is not written.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p llvq-bench --example rhoapply -- <artifact.llvq> [checkpoint] [out-dir]
//! ```
//!
//! `checkpoint` is a local directory or a Hub repo id resolved in the local
//! cache (default `Qwen/Qwen3-4B`); `out-dir` defaults to the artifact's own
//! directory. Nothing is re-encoded and no job is launched.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use llvq_artifact::{Codebook, QuantizedMatrix, RawMatrix};
use llvq_quant::quantizer::BlockCode;

const DIM: usize = 24;

// ---------------------------------------------------------------------------
// SHA-256, because a core crate has no dependencies
// ---------------------------------------------------------------------------

/// FIPS 180-4 SHA-256. Self-tested against two published vectors at startup:
/// a hash function that is wrong in the same way on both files would report a
/// false idempotence, which is exactly the control it serves.
struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    len: usize,
    total: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            len: 0,
            total: 0,
        }
    }

    fn block(&mut self, b: &[u8]) {
        let mut w = [0u32; 64];
        for (i, wi) in w.iter_mut().take(16).enumerate() {
            *wi = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = self.h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for (a, b) in self.h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.len > 0 {
            let take = (64 - self.len).min(data.len());
            self.buf[self.len..self.len + take].copy_from_slice(&data[..take]);
            self.len += take;
            data = &data[take..];
            if self.len == 64 {
                let b = self.buf;
                self.block(&b);
                self.len = 0;
            } else {
                // `take` emptied `data`, and the buffer is still short of a
                // block. Falling through would overwrite it with nothing and
                // lose what is held — the shape that made `finish` spin
                // forever the first time this was run.
                return;
            }
        }
        while data.len() >= 64 {
            let (b, rest) = data.split_at(64);
            self.block(b);
            data = rest;
        }
        self.buf[..data.len()].copy_from_slice(data);
        self.len = data.len();
    }

    fn finish(mut self) -> String {
        let bits = self.total * 8;
        self.update(&[0x80]);
        while self.len != 56 {
            self.update(&[0]);
        }
        let b = bits.to_be_bytes();
        self.update(&b);
        let mut s = String::new();
        for x in self.h {
            let _ = write!(s, "{x:08x}");
        }
        s
    }
}

fn sha256_selftest() {
    let mut a = Sha256::new();
    a.update(b"");
    assert_eq!(
        a.finish(),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    let mut b = Sha256::new();
    b.update(b"abc");
    assert_eq!(
        b.finish(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // A message longer than one block, fed in awkward pieces, to exercise the
    // buffering path that a one-shot vector never reaches.
    let mut c = Sha256::new();
    for chunk in b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".chunks(7) {
        c.update(chunk);
    }
    assert_eq!(
        c.finish(),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

fn sha256_file(path: &Path) -> std::io::Result<(String, u64)> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((h.finish(), n))
}

/// A sink that hashes and counts on the way through, so a rewrite pays one
/// pass instead of two.
struct HashWriter<W: Write> {
    inner: W,
    hash: Sha256,
    bytes: u64,
}

impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hash.update(&buf[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

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
struct Checkpoint {
    shards: Vec<PathBuf>,
    index: HashMap<String, TensorEntry>,
}

impl Checkpoint {
    fn open(dir: &Path) -> Result<Self, String> {
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
    fn tensor(&self, name: &str, d_out: usize, d_in: usize) -> Result<Vec<f32>, String> {
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
fn resolve_checkpoint(spec: &str) -> Result<PathBuf, String> {
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

// ---------------------------------------------------------------------------
// Decoding a record
// ---------------------------------------------------------------------------

/// `raw` rebuilt in the natural basis, with every row scale multiplied by
/// `factor(i)`. The path is the shipped one — [`llvq_artifact::decode_matrix`]
/// — so what this bench measures is what the model would hold.
fn decode_scaled(raw: &RawMatrix, cb: &Codebook, factor: &dyn Fn(usize) -> f64) -> Vec<f32> {
    let mut codes = Vec::with_capacity(raw.indices.len());
    for (&idx, &gain) in raw.indices.iter().zip(&raw.gains) {
        let point = cb
            .decode(idx, gain)
            .unwrap_or_else(|| panic!("{}: index {idx} outside the map", raw.name));
        codes.push(BlockCode { point, gain });
    }
    let q = QuantizedMatrix {
        name: raw.name.clone(),
        d_out: raw.d_out,
        d_in: raw.d_in,
        codes,
        row_scales: raw
            .row_scales
            .iter()
            .enumerate()
            .map(|(i, s)| s * factor(i))
            .collect(),
        centroids: raw.centroids.clone(),
        rotation_seed: raw.rotation_seed,
        shell_cap: raw.shell_cap,
        tail: raw.tail.clone(),
    };
    llvq_artifact::decode_matrix(&q)
}

// ---------------------------------------------------------------------------
// Step 1 — the foundation, proved on the real file
// ---------------------------------------------------------------------------

/// Prove, on a record taken from the artifact itself, that `row_scales[i] × ρ`
/// scales row `i` of the decoded matrix and nothing else.
///
/// Three one-matrix files are written and decoded: ρ = 1, ρ = 2 and ρ = 0 on a
/// single row. The claim has two halves, and each has its own check:
///
/// * **rows are independent** — every row `j ≠ i` must come back *bit
///   identical* across the three files. If the rotation mixed rows this fails,
///   and the whole bench would be void.
/// * **the row is affine in ρ** — `r(2) = 2·r(1) − r(0)` to rounding. The
///   intercept `r(0)` is the tail's image, and it is not zero, which is why
///   the map is affine rather than linear.
fn prove_row_scale(raw: &RawMatrix, cb: &Codebook, out_dir: &Path) -> Result<(), String> {
    let row = raw.d_out / 3; // an interior row, not an edge case of either end
    println!("STEP 1 — row_scales[i] × ρ scales row i, and only row i");
    println!("  record  {} ({}x{})", raw.name, raw.d_out, raw.d_in);
    println!("  probe row {row}, tail width {}", raw.d_in % DIM);

    let mut decoded = Vec::new();
    for (tag, rho) in [("one", 1.0f64), ("two", 2.0), ("zero", 0.0)] {
        let path = out_dir.join(format!("rhoapply-proof-{tag}.llvq"));
        let f = File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut w = BufWriter::with_capacity(1 << 20, f);
        llvq_artifact::write_header_kinds(
            &mut w,
            llvq_artifact::VERSION,
            1,
            raw.kind,
            llvq_artifact::KindSet::of(raw.kind),
        )
        .map_err(|e| e.to_string())?;
        let mut copy = clone_raw(raw);
        copy.row_scales[row] *= rho;
        llvq_artifact::write_matrix_raw(&mut w, llvq_artifact::VERSION, &copy)
            .map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())?;
        drop(w);

        // Read it back through the file, not from memory: the claim is about
        // what a reader of the artifact sees.
        let mut r = BufReader::with_capacity(1 << 20, File::open(&path).map_err(|e| e.to_string())?);
        let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
        let back = llvq_artifact::read_matrix_raw(&mut r, h.version).map_err(|e| e.to_string())?;
        decoded.push(decode_scaled(&back, cb, &|_| 1.0));
        std::fs::remove_file(&path).map_err(|e| e.to_string())?;
    }
    let (r1, r2, r0) = (&decoded[0], &decoded[1], &decoded[2]);

    let n = raw.d_in;
    let mut differing_rows = Vec::new();
    for j in 0..raw.d_out {
        let a = &r1[j * n..(j + 1) * n];
        let b = &r2[j * n..(j + 1) * n];
        let c = &r0[j * n..(j + 1) * n];
        if a != b || a != c {
            differing_rows.push(j);
        }
    }
    println!(
        "  rows that moved when row {row} was rescaled: {:?} (bit-exact comparison over {} rows)",
        differing_rows, raw.d_out
    );
    if differing_rows != vec![row] {
        return Err(
            "the rotation does not keep rows independent, or the writer touched another row: \
             the whole bench is void"
                .into(),
        );
    }

    let a = &r1[row * n..(row + 1) * n];
    let b = &r2[row * n..(row + 1) * n];
    let c = &r0[row * n..(row + 1) * n];
    let scale = b.iter().fold(0.0f64, |m, v| m.max(f64::from(*v).abs()));
    let dev = a
        .iter()
        .zip(b)
        .zip(c)
        .fold(0.0f64, |m, ((x, y), z)| {
            m.max((2.0 * f64::from(*x) - f64::from(*z) - f64::from(*y)).abs())
        });
    let nb: f64 = a
        .iter()
        .zip(c)
        .map(|(x, z)| (f64::from(*x) - f64::from(*z)).powi(2))
        .sum();
    let nt: f64 = c.iter().map(|z| f64::from(*z).powi(2)).sum();
    println!("  affinity  max|r(2) − (2·r(1) − r(0))| / max|r(2)| = {:.3e}", dev / scale);
    println!(
        "  intercept ‖T‖²/(‖B‖²+‖T‖²) = {:.6e}  — the tail, which no row scale reaches",
        nt / (nb + nt)
    );
    if dev / scale > 1e-6 {
        return Err("the decoded row is not affine in the row scale".into());
    }
    println!("  PROVED: r_i(ρ) = ρ·B_i + T_i, rows independent.\n");
    Ok(())
}

fn clone_raw(m: &RawMatrix) -> RawMatrix {
    RawMatrix {
        name: m.name.clone(),
        d_out: m.d_out,
        d_in: m.d_in,
        kind: m.kind,
        indices: m.indices.clone(),
        gains: m.gains.clone(),
        row_scales: m.row_scales.clone(),
        centroids: m.centroids.clone(),
        rotation_seed: m.rotation_seed,
        shell_cap: m.shell_cap,
        tail: m.tail.clone(),
    }
}

// ---------------------------------------------------------------------------
// The rewrite
// ---------------------------------------------------------------------------

/// Copy `src` to `dst` changing nothing but the contents of `row_scales`.
///
/// Every other field — the indices, the gains, the centroids, the rotation
/// seed, the tail, the shell cap, the header and its two fingerprints — goes
/// through [`llvq_artifact::read_matrix_raw`] and
/// [`llvq_artifact::write_matrix_raw`] untouched, which is the pair whose
/// byte-identity is pinned by `raw_passthrough_is_byte_identical`. Nothing is
/// decoded and nothing is re-encoded.
fn rewrite(
    src: &Path,
    dst: &Path,
    factor: &dyn Fn(usize, usize) -> f64,
) -> Result<(String, u64), String> {
    let mut r = BufReader::with_capacity(1 << 22, File::open(src).map_err(|e| e.to_string())?);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let mut w = HashWriter {
        inner: BufWriter::with_capacity(1 << 22, File::create(dst).map_err(|e| e.to_string())?),
        hash: Sha256::new(),
        bytes: 0,
    };
    llvq_artifact::write_header_kinds(&mut w, h.version, h.matrices, h.default_kind, h.kinds)
        .map_err(|e| e.to_string())?;
    for mi in 0..h.matrices as usize {
        let mut m = llvq_artifact::read_matrix_raw(&mut r, h.version).map_err(|e| e.to_string())?;
        for (i, s) in m.row_scales.iter_mut().enumerate() {
            *s *= factor(mi, i);
        }
        llvq_artifact::write_matrix_raw(&mut w, h.version, &m).map_err(|e| e.to_string())?;
    }
    // Everything after the last record, copied verbatim. On a v3+ file that
    // is at least the two section counts `ArtifactWriter::finish` writes
    // (eight zero bytes, and the first thing the idempotence control caught
    // missing); on a sealed file it is the raw tensors and the blobs. Copying
    // the bytes rather than re-serializing the sections is what keeps this
    // tool honest about the claim it makes: it edits `row_scales` and nothing
    // else, including the parts of the format it does not model.
    std::io::copy(&mut r, &mut w).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())?;
    let bytes = w.bytes;
    Ok((w.hash.finish(), bytes))
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

fn quantiles(sorted: &[f64], q: f64) -> f64 {
    let x = q * (sorted.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    sorted[lo] + (x - lo as f64) * (sorted[hi] - sorted[lo])
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let s = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt();
    (m, s)
}

/// Pearson correlation of two series of the same length.
fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in x.iter().zip(y) {
        let (da, db) = (a - mx, b - my);
        sxy += da * db;
        sxx += da * da;
        syy += db * db;
    }
    sxy / (sxx * syy).sqrt()
}

/// Median of a copy, so the caller keeps its order.
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    quantiles(&s, 0.5)
}

/// Everything one row contributes. `u = w − T` and `B = r − T` are the
/// vectors the row scale actually moves (the tail `T` is out of its reach);
/// `w` and `r` are the ones the prereg's ρ is defined on. Six numbers per row
/// are enough to give every ρ, every cosine and every `‖ΔW‖²` below exactly,
/// with no second decode.
#[derive(Clone, Copy)]
struct RowStat {
    uu: f64,
    ub: f64,
    bb: f64,
    num: f64,
    den: f64,
    ww: f64,
}

impl RowStat {
    /// The prereg's ρ, §2 verbatim: `⟨w, r⟩ / ⟨r, r⟩`.
    fn rho(self) -> f64 {
        self.num / self.den
    }

    /// The minimiser of `‖ΔW‖²` over the parameter that actually moves.
    fn rho_tilde(self) -> f64 {
        self.ub / self.bb
    }

    /// `‖ΔW‖²` for this row when its row scale is multiplied by ρ.
    fn j(self, rho: f64) -> f64 {
        self.uu - 2.0 * rho * self.ub + rho * rho * self.bb
    }

    /// The angle factor of the shrink: `cos ∠(w, r)`. This is the quantity M0
    /// simulated (0.960740 on gaussian blocks); here it is measured, and it
    /// carries whatever angle GPTQ's compensation opened between `w` and the
    /// block the encoder was actually given.
    fn cos(self) -> f64 {
        self.num / (self.ww * self.den).sqrt()
    }

    /// The length factor of the shrink: `‖w‖/‖r‖`. M0 could not see it — its
    /// scales were taken on the encoded blocks — and `ρ_i = L_i · c_i` holds
    /// row by row, exactly.
    fn len_ratio(self) -> f64 {
        (self.ww / self.den).sqrt()
    }
}

/// Which rows of the global table belong to one matrix, and what it is.
struct MatStat {
    name: String,
    proj: String,
    layer: usize,
    first: usize,
    len: usize,
}

impl MatStat {
    fn rows<'a>(&self, all: &'a [RowStat]) -> &'a [RowStat] {
        &all[self.first..self.first + self.len]
    }
    fn slice<'a>(&self, all: &'a [f64]) -> &'a [f64] {
        &all[self.first..self.first + self.len]
    }
}

/// Three-level decomposition of the variance of one number per row: between
/// projection types, between matrices of the same type, and inside a matrix.
/// Group sizes are unequal, so every level is weighted by its count; the three
/// then add up to the total exactly, which is printed rather than asserted.
fn variance_levels(values: &[f64], mats: &[MatStat]) -> (f64, f64, f64, f64) {
    let n = values.len() as f64;
    let grand = values.iter().sum::<f64>() / n;
    let total = values.iter().map(|v| (v - grand) * (v - grand)).sum::<f64>() / n;

    let per_mat: Vec<(f64, f64, f64)> = mats
        .iter()
        .map(|m| {
            let v = m.slice(values);
            let c = v.len() as f64;
            let mean = v.iter().sum::<f64>() / c;
            let ss = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>();
            (c, mean, ss)
        })
        .collect();

    let mut type_n: HashMap<&str, f64> = HashMap::new();
    let mut type_s: HashMap<&str, f64> = HashMap::new();
    for (m, (c, mean, _)) in mats.iter().zip(&per_mat) {
        *type_n.entry(m.proj.as_str()).or_insert(0.0) += c;
        *type_s.entry(m.proj.as_str()).or_insert(0.0) += c * mean;
    }
    let between_type = type_n
        .iter()
        .map(|(k, c)| {
            let mt = type_s[k] / c;
            c * (mt - grand) * (mt - grand)
        })
        .sum::<f64>()
        / n;
    let between_mat = mats
        .iter()
        .zip(&per_mat)
        .map(|(m, (c, mean, _))| {
            let mt = type_s[m.proj.as_str()] / type_n[m.proj.as_str()];
            c * (mean - mt) * (mean - mt)
        })
        .sum::<f64>()
        / n;
    let within = per_mat.iter().map(|(_, _, ss)| ss).sum::<f64>() / n;
    (between_type, between_mat, within, total)
}

/// M0, `docs/mesures/m0-echelles-2026-09-07.txt`: the optimal row multiplier
/// on gaussian blocks, and the mean cosine that produced it.
const M0_RHO: f64 = 0.960745;
const M0_COS: f64 = 0.960740;
const M0_DELTA: f64 = 1.0 - M0_COS;
/// The scenario threshold of adversarial review 1: a shrink `δ` helps under
/// `H` iff `δ < 2·δ_H`, and the review reads `δ_H` as M0's radial `δ̄`. That
/// reading is an assumption about an unmeasured quantity, so everything
/// computed from this constant is printed as a scenario.
const TIP: f64 = 1.0 - 2.0 * M0_DELTA;

// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("rhoapply: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    sha256_selftest();
    let mut args = std::env::args().skip(1);
    let artifact = PathBuf::from(
        args.next()
            .ok_or("usage: rhoapply <artifact.llvq> [checkpoint] [out-dir]")?,
    );
    let ck_spec = args.next().unwrap_or_else(|| {
        std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-4B".to_string())
    });
    let out_dir = match args.next() {
        Some(d) => PathBuf::from(d),
        None => artifact
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
    };

    let (src_sha, src_bytes) = sha256_file(&artifact).map_err(|e| e.to_string())?;
    println!("M1 — rhoapply, arms A and B of preregistration-m1-rho-2026-09-07.md");
    println!("source   {}", artifact.display());
    println!("         {src_bytes} bytes, sha256 {src_sha}");
    let ck_dir = resolve_checkpoint(&ck_spec)?;
    println!("checkpoint {} -> {}", ck_spec, ck_dir.display());
    println!("out-dir  {}\n", out_dir.display());

    let mut r = BufReader::with_capacity(1 << 22, File::open(&artifact).map_err(|e| e.to_string())?);
    let header = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    println!(
        "format v{}, {} matrices, default kind {}, kinds {}\n",
        header.version, header.matrices, header.default_kind, header.kinds
    );
    let cb = Codebook::new(header.default_kind).map_err(|e| e.to_string())?;

    // ---- Step 1: the foundation, on the second record (the smallest one).
    let _ = llvq_artifact::read_matrix_raw(&mut r, header.version).map_err(|e| e.to_string())?;
    let probe = llvq_artifact::read_matrix_raw(&mut r, header.version).map_err(|e| e.to_string())?;
    drop(r);
    prove_row_scale(&probe, &cb, &out_dir)?;
    drop(probe);

    // ---- Step 2: idempotence. ρ = 1 everywhere must give the same bytes.
    println!("STEP 2 — idempotence: ρ = 1 everywhere");
    let id_path = out_dir.join("rhoapply-identity.llvq");
    let t = Instant::now();
    let (id_sha, id_bytes) = rewrite(&artifact, &id_path, &|_, _| 1.0)?;
    println!("  wrote {} bytes in {:.1?}", id_bytes, t.elapsed());
    println!("  sha256 source   {src_sha}");
    println!("  sha256 rewrite  {id_sha}");
    if id_sha != src_sha || id_bytes != src_bytes {
        return Err(format!(
            "idempotence FAILED: {id_bytes} bytes vs {src_bytes}. The tool does not do what it \
             says and no number below may be published (prereg §4.3)."
        ));
    }
    std::fs::remove_file(&id_path).map_err(|e| e.to_string())?;
    println!("  IDENTICAL, byte for byte. The rewrite touches nothing but row_scales.\n");

    // ---- Step 3: the ρ.
    println!("STEP 3 — ρ_i on the real weights");
    let ck = Checkpoint::open(&ck_dir)?;
    let mut r = BufReader::with_capacity(1 << 22, File::open(&artifact).map_err(|e| e.to_string())?);
    let _ = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;

    // Per matrix, per row: the six accumulators of `RowStat`, from which every
    // ρ, every cosine and every ‖ΔW‖² below is derived exactly and without a
    // second decode.
    let mut rho_of: Vec<Vec<f64>> = Vec::with_capacity(header.matrices as usize);
    let mut rows: Vec<RowStat> = Vec::new();
    let mut mats: Vec<MatStat> = Vec::with_capacity(header.matrices as usize);
    let (mut sum_num, mut sum_den) = (0.0f64, 0.0f64);
    let (mut sum_uu, mut sum_ub, mut sum_bb) = (0.0f64, 0.0f64, 0.0f64);
    let mut sum_ww = 0.0f64;
    let mut dw2_b = 0.0f64;
    let mut dw2_b_exact = 0.0f64;
    let mut above_one = 0usize;
    let t = Instant::now();
    for mi in 0..header.matrices as usize {
        let raw = llvq_artifact::read_matrix_raw(&mut r, header.version).map_err(|e| e.to_string())?;
        let w = ck.tensor(&raw.name, raw.d_out, raw.d_in)?;
        let rr = decode_scaled(&raw, &cb, &|_| 1.0);
        let tt = decode_scaled(&raw, &cb, &|_| 0.0);
        let (layer, proj) = llvq_artifact::split_name(&raw.name).map_err(|e| e.to_string())?;
        let n = raw.d_in;
        let first = rows.len();
        let mut rhos = Vec::with_capacity(raw.d_out);
        for i in 0..raw.d_out {
            let wr = &w[i * n..(i + 1) * n];
            let rv = &rr[i * n..(i + 1) * n];
            let tv = &tt[i * n..(i + 1) * n];
            let (mut num, mut den, mut uu, mut ub, mut bb, mut ww) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for k in 0..n {
                let (wk, rk, tk) = (f64::from(wr[k]), f64::from(rv[k]), f64::from(tv[k]));
                let bk = rk - tk;
                let uk = wk - tk;
                num += wk * rk;
                den += rk * rk;
                uu += uk * uk;
                ub += uk * bk;
                bb += bk * bk;
                ww += wk * wk;
            }
            let st = RowStat { uu, ub, bb, num, den, ww };
            let rho = st.rho();
            rhos.push(rho);
            rows.push(st);
            sum_num += num;
            sum_den += den;
            sum_uu += uu;
            sum_ub += ub;
            sum_bb += bb;
            sum_ww += ww;
            dw2_b += uu - 2.0 * rho * ub + rho * rho * bb;
            // What arm B would cost at the tail-exact optimum: the floor of
            // the per-row objective, and therefore the price of writing the
            // arms with the prereg's ρ rather than ρ̃.
            dw2_b_exact += uu - ub * ub / bb;
            if rho > 1.0 {
                above_one += 1;
            }
        }
        mats.push(MatStat {
            name: raw.name.clone(),
            proj,
            layer,
            first,
            len: raw.d_out,
        });
        rho_of.push(rhos);
        if mi % 28 == 27 || mi + 1 == header.matrices as usize {
            eprintln!("  {}/{} matrices, {:.0?}", mi + 1, header.matrices, t.elapsed());
        }
    }
    drop(r);
    let rho_global = sum_num / sum_den;
    let rho_global_exact = sum_ub / sum_bb;
    let dw2_orig = sum_uu - 2.0 * sum_ub + sum_bb;
    let dw2_a = sum_uu - 2.0 * rho_global * sum_ub + rho_global * rho_global * sum_bb;
    let dw2_a_exact = sum_uu - sum_ub * sum_ub / sum_bb;

    let rho_all: Vec<f64> = rows.iter().map(|s| s.rho()).collect();
    let rho_tilde_all: Vec<f64> = rows.iter().map(|s| s.rho_tilde()).collect();
    let cos_all: Vec<f64> = rows.iter().map(|s| s.cos()).collect();
    let len_all: Vec<f64> = rows.iter().map(|s| s.len_ratio()).collect();

    println!("\n  rows              {}", rows.len());
    println!("  ρ_global (prereg §2)          {rho_global:.6}");
    println!("  ρ* of M0, gaussian blocks     {M0_RHO:.6}");
    println!(
        "  |Δ|                           {:.6}   (prereg §4.4 flags > 0.02)",
        (rho_global - M0_RHO).abs()
    );
    println!(
        "  ρ_global, tail-exact optimum  {rho_global_exact:.6}   (diagnostic, not what the arms use)"
    );

    let mut sorted = rho_all.clone();
    sorted.sort_by(f64::total_cmp);
    let (m, sd) = mean_sd(&rho_all);
    println!("\n  ρ_i distribution over {} rows", rho_all.len());
    println!(
        "    mean {m:.6}   sd {sd:.6}   min {:.6}   max {:.6}",
        sorted[0],
        sorted[sorted.len() - 1]
    );
    print!("    deciles");
    for d in 1..10 {
        print!(" {:.4}", quantiles(&sorted, f64::from(d) / 10.0));
    }
    println!();
    let (mt, sdt) = mean_sd(&rho_tilde_all);
    println!("    tail-exact ρ̃_i: mean {mt:.6}  sd {sdt:.6}  (the arms use ρ_i above)");

    // ---- The shrink, split into the two factors it is actually made of.
    //
    // Adversarial review 1 subtracts M0's gaussian number from ρ_global and
    // calls the remainder "GPTQ compensation". The subtraction is not needed:
    // ρ_i = L_i·c_i holds row by row and both factors are measurable here. The
    // cosine is the same quantity M0 simulated; the length ratio is the one M0
    // could not see, because its scales were taken on the encoded blocks.
    let wcos = rows.iter().map(|s| s.den * s.cos()).sum::<f64>() / sum_den;
    let wlen = rows.iter().map(|s| s.den * s.len_ratio()).sum::<f64>() / sum_den;
    let (mc, sdc) = mean_sd(&cos_all);
    let (ml, sdl) = mean_sd(&len_all);
    let delta_i = 1.0 - rho_global;
    println!("\n  the shrink split into its two factors, measured (adversarial review 1)");
    println!("    ρ_i = L_i · c_i    L_i = ‖w_i‖/‖r_i‖ (length)   c_i = cos∠(w_i, r_i) (angle)");
    println!("    plain mean      L {ml:.6} (sd {sdl:.6})   c {mc:.6} (sd {sdc:.6})");
    println!("    ‖r‖²-weighted   L {wlen:.6}                c {wcos:.6}");
    println!("    M0, gaussian    L invisible there (scales on the encoded blocks)   c {M0_COS:.6}");
    println!(
        "    1 − c = {:.6} (angle) and 1 − L = {:.6} (length) against δ_I = 1 − ρ_global = {delta_i:.6}",
        1.0 - wcos,
        1.0 - wlen
    );
    println!(
        "    the measured angle is {:.3}× M0's δ̄ = {M0_DELTA:.6}: the artifact's rows are turned",
        (1.0 - wcos) / M0_DELTA
    );
    println!("    further from w than a gaussian block is from its own encoding target, which is");
    println!("    what GPTQ's compensation does — it moves the target away from w on purpose.");

    // ---- Per type, with the two factors and the deep-layer cut of review 1.
    const DEEP: usize = 10; // review 1's cut, kept so its table can be checked
    let mut types: Vec<&str> = mats.iter().map(|m| m.proj.as_str()).collect();
    types.sort_unstable();
    types.dedup();
    println!("\n  by projection type");
    println!(
        "    {:<18} {:>8} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>11}",
        "type", "rows", "mean", "sd", "min", "max", "L", "c", "mean L≥10"
    );
    let mut deep_delta: HashMap<&str, f64> = HashMap::new();
    for ty in &types {
        let mut v: Vec<f64> = Vec::new();
        let mut lv: Vec<f64> = Vec::new();
        let mut cv: Vec<f64> = Vec::new();
        let mut deep: Vec<f64> = Vec::new();
        for mm in mats.iter().filter(|m| m.proj == *ty) {
            v.extend_from_slice(mm.slice(&rho_all));
            lv.extend_from_slice(mm.slice(&len_all));
            cv.extend_from_slice(mm.slice(&cos_all));
            if mm.layer >= DEEP {
                deep.extend_from_slice(mm.slice(&rho_all));
            }
        }
        let (m, sd) = mean_sd(&v);
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let (mdeep, _) = mean_sd(&deep);
        deep_delta.insert(ty, 1.0 - mdeep);
        println!(
            "    {ty:<18} {:>8} {m:>10.6} {sd:>10.6} {mn:>10.6} {mx:>10.6} {:>9.6} {:>9.6} {mdeep:>11.6}",
            v.len(),
            mean_sd(&lv).0,
            mean_sd(&cv).0
        );
    }

    println!("\n  ‖ΔW‖², exact from the affine form (prereg §4.5)");
    println!("    ‖W‖²                {sum_ww:.6e}");
    println!("    original (ρ = 1)    {dw2_orig:.9e}   rel {:.6}", dw2_orig / sum_ww);
    println!("    arm A (ρ_global)    {dw2_a:.9e}   rel {:.6}", dw2_a / sum_ww);
    println!("    arm B (ρ_i)         {dw2_b:.9e}   rel {:.6}", dw2_b / sum_ww);
    println!(
        "    A/orig {:.6}   B/A {:.6}   retention-style 1 − ‖ΔW‖²/‖W‖²: {:.4} → {:.4} → {:.4} %",
        dw2_a / dw2_orig,
        dw2_b / dw2_a,
        100.0 * (1.0 - dw2_orig / sum_ww),
        100.0 * (1.0 - dw2_a / sum_ww),
        100.0 * (1.0 - dw2_b / sum_ww)
    );
    if !(dw2_b <= dw2_a && dw2_a <= dw2_orig) {
        return Err(
            "‖ΔW‖²(B) ≤ ‖ΔW‖²(A) ≤ ‖ΔW‖²(original) is violated: a tool defect, not a result \
             (prereg §4.5). Nothing is published."
                .into(),
        );
    }
    println!("    the three inequalities hold — and they hold for ‖ΔW‖², which is H = I.");
    println!("    They are not a guarantee about Tr(ΔW H ΔWᵀ); the next block says what is.");
    println!("    at the tail-exact ρ̃ instead: A {dw2_a_exact:.9e}   B {dw2_b_exact:.9e}");
    println!(
        "    the prereg's ρ leaves {:.3e} on the table in arm B, {:.4} % of ‖ΔW‖²(B)",
        dw2_b - dw2_b_exact,
        100.0 * (dw2_b - dw2_b_exact) / dw2_b
    );
    println!(
        "    rows with ρ_i > 1 (block too short, not too long): {above_one} of {}",
        rho_all.len()
    );

    // ---- What the arms do to the objective GPTQ actually minimises.
    //
    // Write e = w − r = −δ_I·r + e_⊥ with ⟨e_⊥, r⟩ = 0, which is what defines
    // δ_I. Then, for a shrink by δ = 1 − ρ,
    //
    //   J(ρ) − J(1) = ⟨r H rᵀ⟩ · δ · (δ − 2δ_H)   with   δ_H = δ_I − ⟨e_⊥ H rᵀ⟩/⟨r H rᵀ⟩
    //
    // so a shrink helps iff 0 < δ < 2δ_H. Under H = I the cross term vanishes,
    // δ_H = δ_I, and §4.5's inequalities follow. Under the real H the cross
    // term is not measured by this bench — it needs a calibration pass, which
    // prereg §3 puts out of scope. What can be stated without one is a bound.
    let eperp2 = dw2_orig - delta_i * delta_i * sum_den;
    let transverse = (eperp2 / sum_den).sqrt();
    let etot = (dw2_orig / sum_den).sqrt();
    let cos_h_bound = (delta_i / 2.0) / transverse;
    println!("\n  Tr(ΔW H ΔWᵀ), which this bench does NOT measure (prereg §7, review 3)");
    println!("    e = w − r = −δ_I·r + e_⊥,  ⟨e_⊥, r⟩ = 0 by the definition of δ_I");
    println!("      δ_H = δ_I − ⟨e_⊥ H rᵀ⟩/⟨r H rᵀ⟩      J(ρ) − J(1) = ⟨r H rᵀ⟩·δ·(δ − 2δ_H)");
    println!("      a shrink helps under H iff 0 < δ < 2δ_H. H = I gives δ_H = δ_I: §4.5.");
    println!("    δ_I                              {delta_i:.6}");
    println!("    ‖e‖/‖r‖                          {etot:.6}");
    println!("    ‖e_⊥‖/‖r‖ (Euclidean)            {transverse:.6}");
    println!(
        "    arm A survives any H with cos_H(e_⊥, r) < (δ_I/2)/(‖e_⊥‖_H/‖r‖_H) = {:.2} %",
        100.0 * cos_h_bound
    );
    println!(
        "    the transverse term is {:.2}× the stake, and GPTQ pushes its sign both ways:",
        transverse / (delta_i / 2.0)
    );
    println!("    back-propagation shrinks ‖e‖_H, and it also correlates e with r on purpose.");

    // ---- The scenario of review 1, printed as a scenario.
    let capture = delta_i * (2.0 * M0_DELTA - delta_i) / (M0_DELTA * M0_DELTA);
    println!("\n  SCENARIO, not a measurement: δ_H set equal to M0's radial δ̄ = {M0_DELTA:.6}");
    println!("    that is review 1's assumption — an unmeasured quantity given a measured one's");
    println!("    value, and review 3 above says the same quantity is free over a range 10× wider.");
    println!(
        "    tipping point δ = 2δ̄ = {:.6}; at δ_I = {delta_i:.6} arm A would capture",
        2.0 * M0_DELTA
    );
    println!(
        "    δ(2δ̄−δ)/δ̄² = {:.1} % of the gain available under H, and δ/δ̄ = {:.3}.",
        100.0 * capture,
        delta_i / M0_DELTA
    );
    println!("    {:<18} {:>10} {:>9}  under the scenario", "type (layers ≥ 10)", "δ", "δ/δ̄");
    let mut deep_types: Vec<&&str> = types.iter().collect();
    deep_types.sort_by(|a, b| deep_delta[**a].total_cmp(&deep_delta[**b]));
    for ty in deep_types {
        let d = deep_delta[*ty];
        println!(
            "    {ty:<18} {d:>10.6} {:>9.3}  {}",
            d / M0_DELTA,
            if d < 2.0 * M0_DELTA { "gain" } else { "LOSS" }
        );
    }

    // ---- Arm B row by row against that threshold, and what the low rows are.
    let below: Vec<usize> = (0..rows.len()).filter(|&i| rho_all[i] < TIP).collect();
    let e_below: f64 = below.iter().map(|&i| rows[i].ww).sum::<f64>() / sum_ww;
    println!("\n  ρ_i against the scenario threshold 1 − 2δ̄ = {TIP:.6} (review 2)");
    println!(
        "    rows below            {:>8}  ({:.2} % of rows, {:.2} % of ‖W‖²)",
        below.len(),
        100.0 * below.len() as f64 / rows.len() as f64,
        100.0 * e_below
    );
    for cut in [0.8f64, 0.5, 0.3] {
        let n = rho_all.iter().filter(|&&x| x < cut).count();
        let e: f64 = rows
            .iter()
            .zip(&rho_all)
            .filter(|(_, &x)| x < cut)
            .map(|(s, _)| s.ww)
            .sum::<f64>()
            / sum_ww;
        println!(
            "    below {cut:.2}            {n:>8}  ({:.3} % of rows, {:.4} % of ‖W‖²)",
            100.0 * n as f64 / rows.len() as f64,
            100.0 * e
        );
    }
    let logw: Vec<f64> = rows.iter().map(|s| s.ww.max(f64::MIN_POSITIVE).ln()).collect();
    println!("    corr(ρ_i, log‖w_i‖) over all rows      {:+.4}", corr(&rho_all, &logw));
    let small: Vec<f64> = rows
        .iter()
        .zip(&rho_all)
        .filter(|(_, &x)| x < 0.8)
        .map(|(s, _)| s.ww.sqrt())
        .collect();
    let big: Vec<f64> = rows
        .iter()
        .zip(&rho_all)
        .filter(|(_, &x)| x >= 0.8)
        .map(|(s, _)| s.ww.sqrt())
        .collect();
    if !small.is_empty() {
        println!(
            "    median ‖w_i‖: rows with ρ < 0.8 {:.6} against {:.6} for the rest, ratio {:.3}",
            median(&small),
            median(&big),
            median(&small) / median(&big)
        );
    }

    // ---- Where arm B's advantage over arm A lives. Exact, from the affine
    // form: the per-row difference of the two objectives, not an upper bound.
    let adv: Vec<f64> = rows.iter().map(|s| s.j(rho_global) - s.j(s.rho())).collect();
    let adv_total: f64 = adv.iter().sum();
    let neg: Vec<f64> = adv.iter().copied().filter(|x| *x < 0.0).collect();
    let mut adv_sorted = adv.clone();
    adv_sorted.sort_by(|a, b| b.total_cmp(a));
    println!("\n  where arm B's advantage over arm A lives — exact, per row (review 2)");
    println!(
        "    total  {adv_total:.4e}   (= ‖ΔW‖²(A) − ‖ΔW‖²(B) = {:.4e}, check)",
        dw2_a - dw2_b
    );
    println!(
        "    rows where B is worse than A: {} ({:.2} %), together {:.4e}",
        neg.len(),
        100.0 * neg.len() as f64 / rows.len() as f64,
        neg.iter().sum::<f64>()
    );
    for k in [1000usize, 10_000, 100_000] {
        let s: f64 = adv_sorted[..k].iter().sum();
        println!(
            "    top {k:>7} rows ({:>5.2} % of rows) carry {:>6.2} % of it",
            100.0 * k as f64 / rows.len() as f64,
            100.0 * s / adv_total
        );
    }
    let mut per_mat: Vec<(f64, usize)> = mats
        .iter()
        .enumerate()
        .map(|(k, m)| (m.slice(&adv).iter().sum::<f64>(), k))
        .collect();
    per_mat.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("    {:<44} {:>9} {:>9} {:>10}", "matrix", "share", "cumul", "rows<0.8");
    let mut cum = 0.0;
    for (v, k) in per_mat.iter().take(8) {
        cum += v / adv_total;
        let n = mats[*k].slice(&rho_all).iter().filter(|&&x| x < 0.8).count();
        println!(
            "    {:<44} {:>8.2} % {:>8.2} % {n:>10}",
            mats[*k].name,
            100.0 * v / adv_total,
            100.0 * cum
        );
    }
    let adv_below: f64 = below.iter().map(|&i| adv[i]).sum();
    let adv_small: f64 = adv
        .iter()
        .zip(&rho_all)
        .filter(|(_, &x)| x < 0.8)
        .map(|(a, _)| a)
        .sum();
    println!(
        "    the {} rows under the scenario threshold carry {:.2} % of it,",
        below.len(),
        100.0 * adv_below / adv_total
    );
    println!(
        "    and the {} rows under ρ = 0.8 carry {:.2} % — the row where the prereg's ρ",
        rho_all.iter().filter(|&&x| x < 0.8).count(),
        100.0 * adv_small / adv_total
    );
    println!("    overshoots ρ̃ the most is a row where B loses, not one where it gains.");
    // The matrices that hold the pathological rows are not the ones that carry
    // the advantage, so both lists are printed with the same two statistics.
    let mut by_small: Vec<(usize, usize)> = mats
        .iter()
        .enumerate()
        .map(|(k, m)| (m.slice(&rho_all).iter().filter(|&&x| x < 0.8).count(), k))
        .collect();
    by_small.sort_by_key(|b| std::cmp::Reverse(b.0));
    println!("    {:<44} {:>9} {:>10} {:>12}", "matrix", "rows<0.8", "share", "corr ρ,log‖w‖");
    for (n, k) in per_mat
        .iter()
        .take(3)
        .map(|(_, k)| (mats[*k].slice(&rho_all).iter().filter(|&&x| x < 0.8).count(), *k))
        .chain(by_small.iter().take(3).map(|(n, k)| (*n, *k)))
    {
        let lw: Vec<f64> = mats[k].rows(&rows).iter().map(|s| s.ww.ln()).collect();
        let share = mats[k].slice(&adv).iter().sum::<f64>() / adv_total;
        println!(
            "    {:<44} {n:>9} {:>8.2} % {:>+12.4}",
            mats[k].name,
            100.0 * share,
            corr(mats[k].slice(&rho_all), &lw)
        );
    }

    // ---- Three levels of variance (review 4), on both ρ and the tail-exact ρ̃.
    println!("\n  variance of ρ_i, three levels (review 4)");
    for (tag, v) in [("ρ_i (what the arms use)", &rho_all), ("ρ̃_i (tail-exact)", &rho_tilde_all)] {
        let (bt, bm, wi, tot) = variance_levels(v, &mats);
        println!("    {tag}");
        for (name, x) in [
            ("between types", bt),
            ("between matrices, type fixed", bm),
            ("inside one matrix", wi),
        ] {
            println!(
                "      {name:<32} {x:.6e}  {:>6.2} %  sd {:.6}",
                100.0 * x / tot,
                x.sqrt()
            );
        }
        println!(
            "      {:<32} {tot:.6e}  {:>6.2} %  sd {:.6}   (sum of the three: {:.6e})",
            "total",
            100.0,
            tot.sqrt(),
            bt + bm + wi
        );
    }
    let diff: Vec<f64> = rho_all.iter().zip(&rho_tilde_all).map(|(a, b)| a - b).collect();
    let (md, sdd) = mean_sd(&diff);
    println!(
        "    ρ_i − ρ̃_i: mean {md:+.6}  sd {sdd:.6}   corr(ρ_i, ρ̃_i) {:+.4}",
        corr(&rho_all, &rho_tilde_all)
    );
    let (_, sdte) = mean_sd(&rho_tilde_all);
    let cross = sd * sd - sdte * sdte - sdd * sdd;
    println!("    var(ρ) = var(ρ̃) + var(ρ−ρ̃) + 2cov, exactly:");
    println!(
        "      {:.6e} = {:.6e} + {:.6e} + ({:.6e})   shares {:.1} %, {:.1} %, {:.1} %",
        sd * sd,
        sdte * sdte,
        sdd * sdd,
        cross,
        100.0 * sdte * sdte / (sd * sd),
        100.0 * sdd * sdd / (sd * sd),
        100.0 * cross / (sd * sd)
    );
    let (_, _, wi_t, tot_t) = variance_levels(&rho_tilde_all, &mats);
    println!("    so `1 − (sd ρ̃ / sd ρ)²` is not the share of anything: the cross term is not");
    println!("    zero. The tail term does dominate the dispersion of the coefficient the arms");
    println!(
        "    use, and ρ̃ still keeps {:.1} % of its own variance inside one matrix.",
        100.0 * wi_t / tot_t
    );

    // ---- Arms the timestamped prereg does not fix. Computed, NOT written:
    // a third artifact is a third MMLU job, and that is an operator decision.
    let mut dw2_c = 0.0f64;
    for mm in &mats {
        let (num, den): (f64, f64) = mm
            .rows(&rows)
            .iter()
            .fold((0.0, 0.0), |(n, d), s| (n + s.num, d + s.den));
        let rho_m = num / den;
        dw2_c += mm.rows(&rows).iter().map(|s| s.j(rho_m)).sum::<f64>();
    }
    let dw2_clip: f64 = rows.iter().map(|s| s.j(s.rho().max(TIP))).sum();
    println!("\n  arms the prereg does not fix — computed here, NOT written");
    println!("    C   one ρ per matrix (252 numbers)   ‖ΔW‖² {dw2_c:.9e}   rel {:.6}", dw2_c / sum_ww);
    println!(
        "        recovers {:.1} % of what B gains over A, for 252 numbers instead of {}",
        100.0 * (dw2_a - dw2_c) / (dw2_a - dw2_b),
        rows.len()
    );
    println!(
        "    B′  arm B clipped at ρ ≥ {TIP:.6}   ‖ΔW‖² {dw2_clip:.9e}   rel {:.6}",
        dw2_clip / sum_ww
    );
    println!(
        "        keeps {:.1} % of B's gain over A while touching no row below the scenario",
        100.0 * (dw2_a - dw2_clip) / (dw2_a - dw2_b)
    );
    println!("        threshold; costs the same file, and separates the radial bias from the");
    println!("        scale defect on small rows. Writing either is an operator decision.\n");

    // ---- Step 4: the two artifacts.
    println!("STEP 4 — the two artifacts");
    let a_path = out_dir.join("q4b-tetra-rhoA.llvq");
    let b_path = out_dir.join("q4b-tetra-rhoB.llvq");
    let t = Instant::now();
    let (a_sha, a_bytes) = rewrite(&artifact, &a_path, &|_, _| rho_global)?;
    let (b_sha, b_bytes) = rewrite(&artifact, &b_path, &|mi, i| rho_of[mi][i])?;
    println!("  written in {:.1?}", t.elapsed());
    println!("  arm A  {}", a_path.display());
    println!("         {a_bytes} bytes, sha256 {a_sha}");
    println!("  arm B  {}", b_path.display());
    println!("         {b_bytes} bytes, sha256 {b_sha}");
    if a_bytes != src_bytes || b_bytes != src_bytes {
        return Err(format!(
            "size control FAILED (prereg §4.2): A {a_bytes}, B {b_bytes}, source {src_bytes}"
        ));
    }
    println!("  same size as the source, to the byte. Only row_scales differs.");

    // Re-read both, to state that what is on disk decodes and carries the
    // scales that were asked for.
    for (tag, path, per_row) in [("A", &a_path, false), ("B", &b_path, true)] {
        let mut rr =
            BufReader::with_capacity(1 << 22, File::open(path).map_err(|e| e.to_string())?);
        let h = llvq_artifact::read_header(&mut rr).map_err(|e| e.to_string())?;
        let mut rs =
            BufReader::with_capacity(1 << 22, File::open(&artifact).map_err(|e| e.to_string())?);
        let _ = llvq_artifact::read_header(&mut rs).map_err(|e| e.to_string())?;
        let mut worst = 0.0f64;
        for rhos in &rho_of {
            let got = llvq_artifact::read_matrix_raw(&mut rr, h.version).map_err(|e| e.to_string())?;
            let src = llvq_artifact::read_matrix_raw(&mut rs, h.version).map_err(|e| e.to_string())?;
            for ((&g, &s), &rho) in got.row_scales.iter().zip(&src.row_scales).zip(rhos) {
                let expect = s * if per_row { rho } else { rho_global };
                worst = worst.max((g - expect).abs() / expect.abs());
            }
        }
        println!("  arm {tag} re-read: worst relative row-scale error {worst:.3e}");
        if worst > 0.0 {
            return Err(format!("arm {tag} does not carry the scales it was written with"));
        }
    }

    // ---- Step 5: what is on the disk decodes to what the numbers claim.
    //
    // Step 4 checked the field. This checks the weights: one record is decoded
    // out of each written file and its ‖ΔW‖² compared with the value the
    // affine form predicted for it. A rewrite that corrupted an index or a
    // centroid would pass every control above and fail here.
    println!("STEP 5 — one record re-decoded out of each written file");
    let probe_index = 4usize; // layer 0 gate_proj, the widest ρ spread
    for (tag, path, per_row) in [("A", &a_path, false), ("B", &b_path, true)] {
        let mut rr =
            BufReader::with_capacity(1 << 22, File::open(path).map_err(|e| e.to_string())?);
        let h = llvq_artifact::read_header(&mut rr).map_err(|e| e.to_string())?;
        let mut got = None;
        for mi in 0..h.matrices as usize {
            let m = llvq_artifact::read_matrix_raw(&mut rr, h.version).map_err(|e| e.to_string())?;
            if mi == probe_index {
                got = Some(m);
                break;
            }
        }
        let m = got.ok_or("probe record absent")?;
        let w = ck.tensor(&m.name, m.d_out, m.d_in)?;
        let rv = decode_scaled(&m, &cb, &|_| 1.0);
        let measured: f64 = w
            .iter()
            .zip(&rv)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum();
        // The same quantity from the affine form, recomputed here for this
        // record alone.
        let src = {
            let mut rs = BufReader::with_capacity(
                1 << 22,
                File::open(&artifact).map_err(|e| e.to_string())?,
            );
            let hs = llvq_artifact::read_header(&mut rs).map_err(|e| e.to_string())?;
            let mut out = None;
            for mi in 0..hs.matrices as usize {
                let x =
                    llvq_artifact::read_matrix_raw(&mut rs, hs.version).map_err(|e| e.to_string())?;
                if mi == probe_index {
                    out = Some(x);
                    break;
                }
            }
            out.ok_or("probe record absent from the source")?
        };
        let rr0 = decode_scaled(&src, &cb, &|_| 1.0);
        let tt0 = decode_scaled(&src, &cb, &|_| 0.0);
        let n = src.d_in;
        let mut predicted = 0.0f64;
        for (i, rho) in rho_of[probe_index].iter().enumerate() {
            let rho = if per_row { *rho } else { rho_global };
            for k in i * n..(i + 1) * n {
                let (wk, rk, tk) = (f64::from(w[k]), f64::from(rr0[k]), f64::from(tt0[k]));
                predicted += (wk - tk - rho * (rk - tk)).powi(2);
            }
        }
        println!(
            "  arm {tag}  {}  ‖ΔW‖² measured {measured:.9e}  predicted {predicted:.9e}  rel {:.3e}",
            m.name,
            (measured - predicted).abs() / predicted
        );
        if (measured - predicted).abs() / predicted > 1e-6 {
            return Err(format!(
                "arm {tag}: the file on disk does not decode to what the affine form predicted"
            ));
        }
    }
    println!("  the written files decode to the weights the numbers above describe.");

    println!("\nDone. The bench decides nothing; §5 of the prereg does, after MMLU.");
    Ok(())
}
