//! M1 and M1b — the radial correction applied to a finished artifact.
//!
//! Two preregistrations, two modes, one code path up to the point where the
//! artifacts are written. **This bench decides nothing**: it writes files and
//! prints numbers.
//!
//! * **M1**, the default. `proofs/preregistration-m1-rho-2026-09-07.md`
//!   (sha256 `ac83cdc2…`). Two arms, six controls, and the ρ its §2 fixes.
//! * **M1b**, `--exact`. `proofs/preregistration-m1b-rho-exact-2026-09-07.md`
//!   (sha256 `535af730…`). One arm, five controls, and the ρ̃ its §2 derives —
//!   the exact minimiser of the same objective over the parameter that
//!   actually moves. It exists because M1's arm B collapsed (MMLU 28.11
//!   against 53.49 for Tetra) while *improving* `‖ΔW‖²`, and the cause is the
//!   affine term the paragraph below names.
//!
//! ```text
//!   ρ_i      = ⟨w_i , r_i⟩ / ⟨r_i , r_i⟩          (M1 §2, verbatim)
//!   ρ_global = Σ_i ⟨w_i, r_i⟩ / Σ_i ⟨r_i, r_i⟩
//!   ρ̃_i      = ⟨w_i − T_i , B_i⟩ / ⟨B_i , B_i⟩    (M1b §2)
//!   arm A:  every row scale × ρ_global            (M1)
//!   arm B:  row scale i × ρ_i                     (M1)
//!   arm B′: row scale i × ρ̃_i                     (M1b, --exact)
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
//! cargo run --release -p llvq-bench --example rhoapply -- [--exact] <artifact.llvq> [checkpoint] [out-dir]
//! ```
//!
//! `checkpoint` is a local directory or a Hub repo id resolved in the local
//! cache (default `Qwen/Qwen3-4B`); `out-dir` defaults to the artifact's own
//! directory. Nothing is re-encoded and no job is launched.
//!
//! `--exact` selects M1b: the same steps 1 to 3, then the five controls of its
//! §4 — size, a byte-by-byte comparison against the source outside the
//! `row_scales` zones, idempotence, `sd(ρ̃)` under 0.020, and the four
//! inequalities of §4.5 — and **one** artifact, `q4b-tetra-rhotilde.llvq`.
//! Arms A and B are not written in that mode, and the M1 mode is untouched.

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
// The byte-by-byte control (M1b §4.2)
// ---------------------------------------------------------------------------

/// Byte ranges of `path` that hold `row_scales`, half-open, in file order.
///
/// The offset is derived from the record layout `put_record_head` writes —
/// name, `d_out`, `d_in`, shell cap, kind tag from v5, centroid count,
/// rotation seed and flag, then the centroids — and then **verified** against
/// the file: the eight bytes at each computed slot must be the bit pattern
/// [`llvq_artifact::read_matrix_raw`] returned for that scale. A layout model
/// off by one field would not survive that comparison, so the ranges this
/// returns are the row-scale zones and not a guess about them.
fn row_scale_ranges(path: &Path) -> Result<Vec<(u64, u64)>, String> {
    let mut r = BufReader::with_capacity(1 << 22, File::open(path).map_err(|e| e.to_string())?);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let mut ranges = Vec::with_capacity(h.matrices as usize);
    let mut expect: Vec<Vec<u64>> = Vec::with_capacity(h.matrices as usize);
    for _ in 0..h.matrices {
        let before = r.stream_position().map_err(|e| e.to_string())?;
        let m = llvq_artifact::read_matrix_raw(&mut r, h.version).map_err(|e| e.to_string())?;
        let after = r.stream_position().map_err(|e| e.to_string())?;
        let kind_tag = u64::from(h.version >= llvq_artifact::FIRST_KINDED_VERSION);
        let prefix = 4 + m.name.len() as u64      // name length, then the name
            + 4 + 4 + 4                            // d_out, d_in, shell cap
            + 4 * kind_tag                         // the record's code kind
            + 4                                    // centroid count
            + 8 + 4                                // rotation seed, rotation flag
            + 8 * m.centroids.len() as u64;
        let start = before + prefix;
        let end = start + 8 * m.row_scales.len() as u64;
        if end > after {
            return Err(format!(
                "{}: the row-scale slot [{start}, {end}) runs past the record that ends at {after}",
                m.name
            ));
        }
        ranges.push((start, end));
        expect.push(m.row_scales.iter().map(|s| s.to_bits()).collect());
    }
    drop(r);

    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    for (&(start, end), scales) in ranges.iter().zip(&expect) {
        f.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
        buf.resize((end - start) as usize, 0u8);
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        for (k, &bits) in scales.iter().enumerate() {
            let mut w = [0u8; 8];
            w.copy_from_slice(&buf[8 * k..8 * k + 8]);
            if u64::from_le_bytes(w) != bits {
                return Err(format!(
                    "the row-scale zone model is wrong: scale {k} at offset {} reads {:#018x}, \
                     the record reader says {bits:#018x}",
                    start + 8 * k as u64,
                    u64::from_le_bytes(w)
                ));
            }
        }
    }
    Ok(ranges)
}

/// Compare two files byte by byte and split the differing bytes into those
/// that fall inside `ranges` and those that fall outside.
///
/// Returns `(inside, outside, first_outside, total)`. The prereg's control is
/// `outside == 0`; `inside` is reported because a rewrite that changed nothing
/// at all would also pass `outside == 0`.
fn diff_against(
    src: &Path,
    dst: &Path,
    ranges: &[(u64, u64)],
) -> Result<(u64, u64, Option<u64>, u64), String> {
    let mut a = BufReader::with_capacity(1 << 22, File::open(src).map_err(|e| e.to_string())?);
    let mut b = BufReader::with_capacity(1 << 22, File::open(dst).map_err(|e| e.to_string())?);
    let mut ba = vec![0u8; 1 << 22];
    let mut bb = vec![0u8; 1 << 22];
    let (mut inside, mut outside, mut first_outside, mut off) = (0u64, 0u64, None, 0u64);
    let mut ri = 0usize;
    loop {
        let n = read_up_to(&mut a, &mut ba)?;
        let m = read_up_to(&mut b, &mut bb)?;
        if n != m {
            return Err(format!("the two files disagree in length at offset {off}"));
        }
        if n == 0 {
            break;
        }
        // The common case is a chunk with no difference at all; the slice
        // comparison settles it without touching a single byte by hand.
        if ba[..n] != bb[..n] {
            for k in 0..n {
                if ba[k] == bb[k] {
                    continue;
                }
                let at = off + k as u64;
                while ri < ranges.len() && ranges[ri].1 <= at {
                    ri += 1;
                }
                if ri < ranges.len() && at >= ranges[ri].0 {
                    inside += 1;
                } else {
                    outside += 1;
                    first_outside.get_or_insert(at);
                }
            }
        }
        off += n as u64;
    }
    Ok((inside, outside, first_outside, off))
}

/// `read` until the buffer is full or the file ends. `Read::read` is allowed
/// to return short, and a short read here would misalign the two files.
fn read_up_to(r: &mut impl Read, buf: &mut [u8]) -> Result<usize, String> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]).map_err(|e| e.to_string())? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// The closed form, checked against the vectors it claims to minimise
// ---------------------------------------------------------------------------

/// Cross-check `ρ̃ = ⟨w − T, B⟩ / ⟨B, B⟩` on real rows, without using the
/// accumulators that produced it.
///
/// For each probed row the objective `‖w − (ρ·B + T)‖²` is summed **from the
/// three vectors** on a grid of ρ, and two things are required:
///
/// * the grid minimum sits on the grid point nearest `ρ̃`;
/// * the parabola through three points taken *away* from `ρ̃` (centred on
///   ρ = 1, so the check cannot pass by cancellation) has its vertex at `ρ̃`
///   to better than `1e-9` relative.
///
/// Returns the worst relative vertex error over the probed rows.
fn grid_check(
    name: &str,
    w: &[f32],
    rr: &[f32],
    tt: &[f32],
    d_in: usize,
    probes: &[usize],
) -> Result<f64, String> {
    // `‖w − (ρ·B + T)‖²` for one row, summed straight from the vectors.
    let direct = |i: usize, rho: f64| -> f64 {
        let mut s = 0.0f64;
        for k in i * d_in..(i + 1) * d_in {
            let (wk, rk, tk) = (f64::from(w[k]), f64::from(rr[k]), f64::from(tt[k]));
            let d = wk - tk - rho * (rk - tk);
            s += d * d;
        }
        s
    };
    let mut worst = 0.0f64;
    println!("  {name}, {} rows probed", probes.len());
    println!(
        "    {:>8} {:>12} {:>12} {:>12} {:>12}",
        "row", "ρ̃ closed", "ρ̃ vertex", "rel err", "grid argmin"
    );
    for &i in probes {
        let (mut ub, mut bb) = (0.0f64, 0.0f64);
        for k in i * d_in..(i + 1) * d_in {
            let (wk, rk, tk) = (f64::from(w[k]), f64::from(rr[k]), f64::from(tt[k]));
            ub += (wk - tk) * (rk - tk);
            bb += (rk - tk) * (rk - tk);
        }
        let rho_tilde = ub / bb;

        // Vertex of the parabola through ρ = 1 − h, 1, 1 + h. Centred on 1 and
        // not on ρ̃: at ρ̃ the outer difference is exactly zero and the test
        // would pass on any input.
        let h = 0.05f64;
        let (jm, j0, jp) = (direct(i, 1.0 - h), direct(i, 1.0), direct(i, 1.0 + h));
        let vertex = 1.0 - h * (jp - jm) / (2.0 * (jp - 2.0 * j0 + jm));
        let rel = (vertex - rho_tilde).abs() / rho_tilde.abs();
        worst = worst.max(rel);

        // The grid: 21 points of step 0.01 around ρ̃. Its minimum must be the
        // point nearest ρ̃, which is index 10 by construction.
        let step = 0.01f64;
        let mut best = (0usize, f64::INFINITY);
        for g in 0..21usize {
            let rho = rho_tilde + (g as f64 - 10.0) * step;
            let j = direct(i, rho);
            if j < best.1 {
                best = (g, j);
            }
        }
        println!("    {i:>8} {rho_tilde:>12.8} {vertex:>12.8} {rel:>12.3e} {:>12}", best.0);
        if best.0 != 10 {
            return Err(format!(
                "{name} row {i}: the grid minimum is at index {} and not at ρ̃ (index 10). \
                 The closed form does not minimise the objective it is derived from.",
                best.0
            ));
        }
        if rel > 1e-9 {
            return Err(format!(
                "{name} row {i}: the vertex of the measured parabola is {vertex:.12} and the \
                 closed form gives {rho_tilde:.12}, {rel:.3e} relative — beyond 1e-9."
            ));
        }
    }
    println!("    worst relative error {worst:.3e}, under 1e-9. ρ̃ = ⟨w−T, B⟩/⟨B, B⟩ confirmed.\n");
    Ok(worst)
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

/// The record every probe of this bench uses: layer 0 `gate_proj`, the widest
/// ρ spread of the file, so a check that passes there passes on the worst
/// record the artifact has.
const PROBE_INDEX: usize = 4;

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
    let mut positional: Vec<String> = Vec::new();
    let mut exact = false;
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "--exact" => exact = true,
            _ if a.starts_with("--") => {
                return Err(format!(
                    "unknown flag {a}: the only one is --exact \
                     (usage: rhoapply [--exact] <artifact.llvq> [checkpoint] [out-dir])"
                ));
            }
            _ => positional.push(a),
        }
    }
    let mut args = positional.into_iter();
    let artifact = PathBuf::from(
        args.next()
            .ok_or("usage: rhoapply [--exact] <artifact.llvq> [checkpoint] [out-dir]")?,
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
    if exact {
        println!(
            "M1b — rhoapply --exact, the single arm of \
             preregistration-m1b-rho-exact-2026-09-07.md"
        );
        println!("      ρ̃_i = ⟨w_i − T_i, B_i⟩ / ⟨B_i, B_i⟩   (§2). Arms A and B are not written.");
    } else {
        println!("M1 — rhoapply, arms A and B of preregistration-m1-rho-2026-09-07.md");
    }
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
        if mi == PROBE_INDEX {
            // The closed form checked against the objective itself, on the
            // record with the widest ρ spread, before a single ρ̃ is used.
            println!("\n  the closed form ρ̃ against ‖w − (ρ·B + T)‖² on the vectors");
            let probes: Vec<usize> = (0..5).map(|k| k * (raw.d_out - 1) / 4).collect();
            grid_check(&raw.name, &w, &rr, &tt, n, &probes)?;
        }
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

    // ---- M1b, --exact: the five controls of its §4, then one artifact.
    if exact {
        // The extreme row first. The closed form was checked in step 3 on five
        // spread rows of the widest record; the row that carries the smallest
        // ρ̃ of the whole file is the one a formula error would show on, so it
        // gets the same treatment before anything is written.
        let (arg_min, _) = rho_tilde_all
            .iter()
            .enumerate()
            .fold((0usize, f64::INFINITY), |(bi, bv), (i, &v)| {
                if v < bv { (i, v) } else { (bi, bv) }
            });
        let mk = mats
            .iter()
            .position(|m| arg_min >= m.first && arg_min < m.first + m.len)
            .ok_or("the row with the smallest ρ̃ belongs to no matrix")?;
        println!("\n  the closed form on the extreme row of the file");
        {
            let mut rs = BufReader::with_capacity(
                1 << 22,
                File::open(&artifact).map_err(|e| e.to_string())?,
            );
            let hs = llvq_artifact::read_header(&mut rs).map_err(|e| e.to_string())?;
            let mut got = None;
            for mi in 0..hs.matrices as usize {
                let m =
                    llvq_artifact::read_matrix_raw(&mut rs, hs.version).map_err(|e| e.to_string())?;
                if mi == mk {
                    got = Some(m);
                    break;
                }
            }
            let m = got.ok_or("the extreme record is absent")?;
            let w = ck.tensor(&m.name, m.d_out, m.d_in)?;
            let rr = decode_scaled(&m, &cb, &|_| 1.0);
            let tt = decode_scaled(&m, &cb, &|_| 0.0);
            grid_check(&m.name, &w, &rr, &tt, m.d_in, &[arg_min - mats[mk].first])?;
        }

        // ---- Control 4: the dispersion, published before the decision to
        // write. The prereg stops the bench above 0.020.
        let mut st = rho_tilde_all.clone();
        st.sort_by(f64::total_cmp);
        let (mt, sdt) = mean_sd(&rho_tilde_all);
        println!("CONTROL 4 (prereg §4.4) — the distribution of ρ̃ over {} rows", st.len());
        println!(
            "  mean {mt:.6}   sd {sdt:.6}   min {:.6}   max {:.6}",
            st[0],
            st[st.len() - 1]
        );
        print!("  deciles");
        for d in 1..10 {
            print!(" {:.4}", quantiles(&st, f64::from(d) / 10.0));
        }
        println!();
        println!("  the same rows under M1's ρ: sd {sd:.6}, min {:.6}", {
            let mut s2 = rho_all.clone();
            s2.sort_by(f64::total_cmp);
            s2[0]
        });
        println!("  by projection type — the column M1 read as per-row structure");
        println!(
            "  {:<18} {:>8} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}",
            "type", "rows", "mean ρ̃", "sd ρ̃", "min ρ̃", "max ρ̃", "sd ρ (M1)", "min ρ (M1)"
        );
        for ty in &types {
            let mut v: Vec<f64> = Vec::new();
            let mut vm: Vec<f64> = Vec::new();
            for mm in mats.iter().filter(|m| m.proj == *ty) {
                v.extend_from_slice(mm.slice(&rho_tilde_all));
                vm.extend_from_slice(mm.slice(&rho_all));
            }
            let (m, s) = mean_sd(&v);
            let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let (_, sm) = mean_sd(&vm);
            let mnm = vm.iter().cloned().fold(f64::INFINITY, f64::min);
            println!(
                "  {ty:<18} {:>8} {m:>10.6} {s:>10.6} {mn:>10.6} {mx:>10.6} {sm:>12.6} {mnm:>10.6}",
                v.len()
            );
        }
        if sdt > 0.020 {
            return Err(format!(
                "CONTROL 4 FAILED: sd(ρ̃) = {sdt:.6} is above 0.020, so the formula in this bench \
                 is not the one of §2 of the preregistration. No artifact is produced and no \
                 number above may be published."
            ));
        }
        println!("  sd(ρ̃) = {sdt:.6} is under 0.020: control 4 passes.\n");

        // ---- Control 5: the four inequalities.
        println!("CONTROL 5 (prereg §4.5) — the four inequalities of the objective");
        println!("  ‖ΔW‖²(ρ̃)      {dw2_b_exact:.9e}   rel {:.6}", dw2_b_exact / sum_ww);
        println!("  ‖ΔW‖²(B, M1)  {dw2_b:.9e}   rel {:.6}", dw2_b / sum_ww);
        println!("  ‖ΔW‖²(A, M1)  {dw2_a:.9e}   rel {:.6}", dw2_a / sum_ww);
        println!("  ‖ΔW‖²(orig)   {dw2_orig:.9e}   rel {:.6}", dw2_orig / sum_ww);
        println!("  M1's journal:  orig 2.497121948e5   A 2.396572380e5   B 2.392246240e5");
        if !(dw2_b_exact <= dw2_b && dw2_b <= dw2_a && dw2_a <= dw2_orig) {
            return Err(
                "CONTROL 5 FAILED: ‖ΔW‖²(ρ̃) ≤ ‖ΔW‖²(B) ≤ ‖ΔW‖²(A) ≤ ‖ΔW‖²(original) is \
                 violated. That is a tool defect and not a result; nothing is published."
                    .into(),
            );
        }
        println!("  the four hold, in that order. Control 5 passes.");
        // Row by row as well: §2 says ρ̃ beats M1's ρ on every single row.
        let worse = rows
            .iter()
            .filter(|s| s.j(s.rho_tilde()) > s.j(s.rho()) + 1e-12 * s.uu.abs())
            .count();
        println!(
            "  rows where ρ̃ is worse than M1's ρ on the objective: {worse} of {}\n",
            rows.len()
        );
        if worse != 0 {
            return Err("ρ̃ does not minimise the per-row objective it is derived from".into());
        }

        // ---- The artifact.
        println!("STEP 4 — the artifact");
        let p_path = out_dir.join("q4b-tetra-rhotilde.llvq");
        let t = Instant::now();
        let (p_sha, p_bytes) = rewrite(&artifact, &p_path, &|mi, i| {
            rho_tilde_all[mats[mi].first + i]
        })?;
        println!("  written in {:.1?}", t.elapsed());
        println!("  arm B′  {}", p_path.display());
        println!("          {p_bytes} bytes, sha256 {p_sha}");

        // Control 1: the size.
        if p_bytes != src_bytes {
            return Err(format!(
                "CONTROL 1 FAILED (prereg §4.2): {p_bytes} bytes against {src_bytes} for the source"
            ));
        }
        println!("  CONTROL 1: {p_bytes} bytes, the size of the source, to the byte.");

        // Control 2: byte by byte, against the row-scale zones of the source.
        let ranges = row_scale_ranges(&artifact)?;
        let zone_bytes: u64 = ranges.iter().map(|(a, b)| b - a).sum();
        let (inside, outside, first, total) = diff_against(&artifact, &p_path, &ranges)?;
        println!(
            "  CONTROL 2: {} row-scale zones, {zone_bytes} bytes of {total} ({:.4} % of the file)",
            ranges.len(),
            100.0 * zone_bytes as f64 / total as f64
        );
        println!(
            "             bytes that differ from the source: {inside} inside those zones, \
             {outside} outside"
        );
        if outside != 0 {
            return Err(format!(
                "CONTROL 2 FAILED (prereg §4.2): {outside} bytes differ outside the row-scale \
                 zones, the first at offset {}. The rewrite touched something else.",
                first.unwrap_or(0)
            ));
        }
        println!("             every differing byte is a row scale. Control 2 passes.");

        // The scales on disk are the ones that were asked for. The count of
        // scales that actually moved comes from here and not from `inside`: a
        // ρ̃ near 1 changes the low bytes of a double and leaves its sign and
        // exponent alone, so differing *bytes* undercount differing *scales*.
        {
            let mut rr =
                BufReader::with_capacity(1 << 22, File::open(&p_path).map_err(|e| e.to_string())?);
            let h = llvq_artifact::read_header(&mut rr).map_err(|e| e.to_string())?;
            let mut rs = BufReader::with_capacity(
                1 << 22,
                File::open(&artifact).map_err(|e| e.to_string())?,
            );
            let _ = llvq_artifact::read_header(&mut rs).map_err(|e| e.to_string())?;
            let mut worst = 0.0f64;
            let mut moved = 0usize;
            for mm in &mats {
                let got =
                    llvq_artifact::read_matrix_raw(&mut rr, h.version).map_err(|e| e.to_string())?;
                let src =
                    llvq_artifact::read_matrix_raw(&mut rs, h.version).map_err(|e| e.to_string())?;
                for (i, (&g, &s)) in got.row_scales.iter().zip(&src.row_scales).enumerate() {
                    let expect = s * rho_tilde_all[mm.first + i];
                    worst = worst.max((g - expect).abs() / expect.abs());
                    if g.to_bits() != s.to_bits() {
                        moved += 1;
                    }
                }
            }
            println!(
                "  re-read: worst relative row-scale error {worst:.3e}; {moved} of {} scales \
                 carry a different bit pattern than the source",
                rows.len()
            );
            if worst > 0.0 {
                return Err("the artifact does not carry the scales it was written with".into());
            }
        }

        // ---- Step 5: the file on disk decodes to what the numbers claim.
        println!("STEP 5 — one record re-decoded out of the written file");
        let mut rr =
            BufReader::with_capacity(1 << 22, File::open(&p_path).map_err(|e| e.to_string())?);
        let h = llvq_artifact::read_header(&mut rr).map_err(|e| e.to_string())?;
        let mut got = None;
        for mi in 0..h.matrices as usize {
            let m = llvq_artifact::read_matrix_raw(&mut rr, h.version).map_err(|e| e.to_string())?;
            if mi == PROBE_INDEX {
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
                if mi == PROBE_INDEX {
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
        for i in 0..src.d_out {
            let rho = rho_tilde_all[mats[PROBE_INDEX].first + i];
            for k in i * n..(i + 1) * n {
                let (wk, rk, tk) = (f64::from(w[k]), f64::from(rr0[k]), f64::from(tt0[k]));
                predicted += (wk - tk - rho * (rk - tk)).powi(2);
            }
        }
        println!(
            "  {}  ‖ΔW‖² measured {measured:.9e}  predicted {predicted:.9e}  rel {:.3e}",
            m.name,
            (measured - predicted).abs() / predicted
        );
        if (measured - predicted).abs() / predicted > 1e-6 {
            return Err(
                "the file on disk does not decode to what the affine form predicted".into()
            );
        }
        println!("  the written file decodes to the weights the numbers above describe.");
        println!("\nDone. The bench decides nothing; §5 of the M1b preregistration does, after MMLU.");
        return Ok(());
    }

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
    for (tag, path, per_row) in [("A", &a_path, false), ("B", &b_path, true)] {
        let mut rr =
            BufReader::with_capacity(1 << 22, File::open(path).map_err(|e| e.to_string())?);
        let h = llvq_artifact::read_header(&mut rr).map_err(|e| e.to_string())?;
        let mut got = None;
        for mi in 0..h.matrices as usize {
            let m = llvq_artifact::read_matrix_raw(&mut rr, h.version).map_err(|e| e.to_string())?;
            if mi == PROBE_INDEX {
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
                if mi == PROBE_INDEX {
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
        for (i, rho) in rho_of[PROBE_INDEX].iter().enumerate() {
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
