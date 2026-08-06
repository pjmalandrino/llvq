//! Loading the sealed artifact **without decoding it** — the shape a fused
//! kernel needs.
//!
//! ## What this is not
//!
//! [`crate::sealed::load`] rebuilds a model by calling `decode_matrix` on
//! every projection, which materializes 3.63 G weights as f16 tensors. That is
//! correct, it is what every published perplexity refers to, and it throws
//! away the entire point of the format: once decoded, the file occupies as
//! much memory as an f16 checkpoint and runs at exactly its speed. The
//! 2026-08-05 miniature run measured that plainly — 42.7 tok/s against 42.8.
//!
//! This module keeps the weights encoded. It transcodes each matrix to the
//! runtime layout the fused matvec reads — `Planes14` by default, `Slot32`
//! under `LLVQ_FUSED_LAYOUT=slot32` (see [`FusedLayout`]) — and hands the
//! host tables over. Nothing here touches a GPU: this is the portable half,
//! and it is testable without a card.
//!
//! ## The rotation tables, and why they are here
//!
//! The artifact is quantized in a rotated basis, so a kernel reading the
//! stored weights computes `W' x`, not `W x`. The identity it owes is
//!
//! ```text
//! y = W x = (W Qᵀ)(Q x) = W' · rot(x)
//! ```
//!
//! pinned by `llvq-artifact/tests/fused_path_matches_dense.rs`. A kernel
//! cannot rebuild `Q` from its seed — that would mean porting `SplitMix64` and
//! Gram–Schmidt to the device — so the host builds it once per distinct
//! `(width, seed)` and ships three small tables.
//!
//! Distinct pairs are far fewer than matrices: q/k/v share an activation and
//! therefore a rotation, and so do gate/up. Qwen3-4B has 252 projections and
//! **144** rotations, and the tables together weigh under a megabyte.

use std::collections::HashMap;
use std::io::Read;

use llvq_artifact::runtime::{
    transcode, transcode_planes14, ClassTable, Layout, PlanesBlocks, RuntimeBlocks,
};
use llvq_quant::rotation::Rotation;
use llvq_search::fastdec::FastDecoder;

/// Which runtime layout the fused path reads.
///
/// Resolved once, from `LLVQ_FUSED_LAYOUT`, before any transcoding: the whole
/// model is one layout, the kernel is chosen by it, and the two cannot drift
/// apart because every [`HostStream`] carries its variant with it.
///
/// `Planes14` is the default — the reference layout since C1 (2026-08-06,
/// 1.14× over `Slot32` at identical decoded content, 4.804 against 5.510
/// b/weight on the published 4B). `Slot32` stays as the comparison arm and
/// the fallback, bit-identical to what shipped before the switch existed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedLayout {
    Planes14,
    Slot32,
}

impl FusedLayout {
    /// Parse the value of `LLVQ_FUSED_LAYOUT`. `None` (unset) and the empty
    /// string mean the default; anything else must name a layout exactly —
    /// a typo silently falling back to a default would make an A/B lie.
    pub fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("") => Ok(Self::Planes14),
            Some("planes14") => Ok(Self::Planes14),
            Some("slot32") => Ok(Self::Slot32),
            Some(other) => Err(format!(
                "LLVQ_FUSED_LAYOUT={other} : valeurs admises « planes14 » (défaut) et « slot32 »"
            )),
        }
    }

    /// Resolve from the environment.
    pub fn from_env() -> Result<Self, String> {
        let v = std::env::var("LLVQ_FUSED_LAYOUT").ok();
        Self::parse(v.as_deref())
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Planes14 => "planes14",
            Self::Slot32 => "slot32",
        }
    }
}

/// Mirrors `LLVQ_ROT_KMAX` in `llvq-cuda/kernels/llvq_rot.cuh`.
///
/// Duplicated across a language boundary on purpose — the kernel is in another
/// crate that does not build on this machine — and pinned by
/// `the_kmax_constant_matches_the_kernel` on the CUDA side.
pub const ROT_KMAX: usize = 32;

/// One matrix's payload in one runtime layout, as the words a kernel indexes.
///
/// An enum rather than optional fields, and that is the guarantee: a
/// `Planes14` stream **has no bases** — not an empty vector, no field at all —
/// so no code path can upload or launch with a bases array on the planes path.
/// The compiler enforces what a review would otherwise have to.
pub enum HostStream {
    Slot32 {
        /// `Slot32` payload as `u32` words, padded for the kernel's five-word
        /// read window past the last block.
        words: Vec<u32>,
        /// Byte offset of each group of 32 blocks, plus a final sentinel.
        bases: Vec<u32>,
    },
    Planes14 {
        /// The uniform 14-byte records as `u32` words, padded for the
        /// kernel's four-word read window past the last block.
        words: Vec<u32>,
    },
}

/// One projection, transcoded and ready to upload.
pub struct FusedMatrix {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub nblocks: usize,
    pub tail_w: usize,
    /// The payload, in the layout [`FusedModel::layout`] names.
    pub stream: HostStream,
    /// The two gain centroids. One bit of gain is hard-coded in every decoder.
    pub gscale: [f32; 2],
    pub rscale: Vec<f32>,
    /// `d_out × tail_w`, row-major, in the rotated basis.
    pub tail: Vec<f32>,
    /// Key into [`FusedModel::rotations`], or `None` in the natural basis.
    pub rotation: Option<RotKey>,
    /// Bytes this matrix costs at runtime — payload, bases, tail, row scales.
    pub bytes: u64,
}

/// A rotation is determined by its width and its seed, and shared by every
/// matrix consuming the same activation.
pub type RotKey = (usize, u64);

/// `Q`, flattened into what a kernel can read.
pub struct RotationTables {
    pub n: usize,
    /// Power-of-two factor: the width of one Walsh–Hadamard group.
    pub m: usize,
    /// Odd factor: the side of the dense block, and the number of groups.
    pub k: usize,
    /// One bit per coordinate, set when the sign is negative.
    pub signbits: Vec<u32>,
    /// `Q_odd`, zero-padded to `ROT_KMAX × ROT_KMAX` row-major. The padding is
    /// load-bearing: the kernel's inner loops run to a compile-time bound.
    pub small: Vec<f32>,
    /// `1/√m`, narrowed once here rather than recomputed on the device where
    /// `sqrtf` and `rsqrtf` would each land somewhere else.
    pub inv: f32,
}

impl RotationTables {
    pub fn build(n: usize, seed: u64) -> Result<Self, String> {
        let rot = Rotation::new(n, seed);
        let (m, k) = (rot.pow2(), rot.odd());
        if k > ROT_KMAX {
            return Err(format!(
                "largeur {n} : facteur impair {k} au-delà de KMAX={ROT_KMAX}. Le noyau de \
                 rotation ne la traite pas — cf. llvq_rot.cuh."
            ));
        }
        let mut signbits = vec![0u32; n.div_ceil(32)];
        for (i, &s) in rot.signs().iter().enumerate() {
            if s < 0.0 {
                signbits[i >> 5] |= 1 << (i & 31);
            }
        }
        let mut small = vec![0.0f32; ROT_KMAX * ROT_KMAX];
        for g in 0..k {
            for t in 0..k {
                small[g * ROT_KMAX + t] = rot.small()[g * k + t] as f32;
            }
        }
        Ok(Self {
            n,
            m,
            k,
            signbits,
            small,
            inv: (1.0f64 / (m as f64).sqrt()) as f32,
        })
    }
}

/// Everything a fused runtime needs, still on the host.
pub struct FusedModel {
    /// The runtime layout every matrix was transcoded to.
    pub layout: FusedLayout,
    pub matrices: Vec<FusedMatrix>,
    pub rotations: HashMap<RotKey, RotationTables>,
    /// Embedding and norms, carried verbatim — name → (dims, f32 values).
    pub raw: Vec<(String, Vec<usize>, Vec<f32>)>,
    pub config_json: Vec<u8>,
    pub tokenizer_json: Vec<u8>,
    pub quantized_weights: usize,
    pub carried_weights: usize,
    /// Size of the file on disk.
    pub file_bytes: u64,
    /// Bytes the projections occupy at runtime, in the chosen layout.
    pub runtime_bytes: u64,
}

impl FusedModel {
    /// Bits per weight the runtime actually spends on quantized projections.
    ///
    /// Not the same accounting as the file's: the runtime layout pays its
    /// addressing where the file packs 48 bits per block exactly. On the
    /// published 4B: `Slot32` (byte-rounded group strides plus a `u32` base
    /// per group) measured 5.510 b/weight, `Planes14` (uniform 14-byte
    /// records, no bases) 4.804, against 2.0702 effective in the file.
    pub fn runtime_bits_per_weight(&self) -> f64 {
        self.runtime_bytes as f64 * 8.0 / self.quantized_weights as f64
    }
}

fn read_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

/// Pack a transcoded stream into `u32` words the kernel can index.
///
/// The twenty trailing zero bytes are not slack: `slot_fields` reads five
/// consecutive words from `byte >> 2`, so the last block of the stream reads
/// past its own payload by design. Without the pad that is an out-of-bounds
/// read — benign on Metal, an illegal address on CUDA, and it kills the
/// context in the middle of a billed job.
fn pack_words(rt: &RuntimeBlocks) -> Vec<u32> {
    let mut bytes = rt.data.clone();
    bytes.extend_from_slice(&[0u8; 20]);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// [`pack_words`] for a Planes14 stream — the four-word window's padding.
///
/// `planes_fields` reads four consecutive words from `(14·b) >> 2`, so the
/// last block's window reaches up to 2 bytes past the `14·n` payload. Four
/// spare bytes plus word alignment keep that read in bounds — the same
/// arithmetic as planesbench's upload, and skipping it is an illegal address
/// on CUDA, in the middle of a billed job.
fn pack_planes_words(pb: &PlanesBlocks) -> Vec<u32> {
    let mut bytes = pb.data.clone();
    bytes.extend_from_slice(&[0u8; 4]);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Transcode one matrix's raw codes into the words of the chosen layout.
///
/// Returns the stream and its **payload** bytes — the accounting the runtime
/// reports, so `Slot32` counts its bases and `Planes14` has none to count.
/// (The read-window padding is deliberately excluded, as it always was for
/// `Slot32`: it is upload slack, not format.)
///
/// Public so the tests on this machine can pin, without a card, that the
/// packed words decode to exactly the `Slot32` content — the same bijection
/// planesbench proves block by block on the device path.
pub fn transcode_stream(
    fd: &FastDecoder,
    table: &ClassTable,
    indices: &[u64],
    gains: &[u32],
    layout: FusedLayout,
) -> Result<(HostStream, u64), String> {
    match layout {
        FusedLayout::Slot32 => {
            let rt = transcode(fd, table, indices, gains, Layout::Slot32)
                .map_err(|e| e.to_string())?;
            let bytes = rt.data.len() as u64 + rt.bases.len() as u64 * 4;
            let words = pack_words(&rt);
            Ok((HostStream::Slot32 { words, bases: rt.bases }, bytes))
        }
        FusedLayout::Planes14 => {
            let pb = transcode_planes14(fd, table, indices, gains).map_err(|e| e.to_string())?;
            let bytes = pb.data.len() as u64;
            Ok((HostStream::Planes14 { words: pack_planes_words(&pb) }, bytes))
        }
    }
}

/// Load a sealed artifact in encoded form, transcoded to `layout`.
///
/// Costs one transcode of every matrix — 142 s for the 4B on eight cores,
/// measured 2026-08-05 — against `sealed::load`'s full decode, which took
/// 208.7 s on the same file and produced eight gigabytes of f16.
pub fn load(path: &str, layout: FusedLayout) -> Result<FusedModel, String> {
    let file_bytes = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    let f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    if !head.is_self_contained() {
        return Err(format!(
            "{path} n'est qu'un artefact de projections (format v{}) — il ne peut pas \
             tourner seul.",
            head.version
        ));
    }

    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    if layout == FusedLayout::Planes14 {
        // Three bit-planes address eight levels; the layout is only a
        // bijection of the Slot32 content while every class stays within
        // five. True of the v1 table, asserted rather than assumed — the
        // same guard planesbench re-asserts before any timing.
        assert!(
            (0..table.n_entries()).all(|e| table.record(e).len <= 5),
            "une classe dépasse 5 niveaux : Planes14 ne peut pas la porter"
        );
    }
    let mut matrices = Vec::with_capacity(head.matrices as usize);
    let mut rotations: HashMap<RotKey, RotationTables> = HashMap::new();
    let mut quantized_weights = 0usize;
    let mut runtime_bytes = 0u64;

    for _ in 0..head.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).map_err(|e| e.to_string())?;
        // Every decoder hard-codes one gain bit (`hdr >> 9`). A file with a
        // different gain width would transcode into a coherent stream and
        // decode into garbage — silently.
        if m.centroids.len() != 2 {
            return Err(format!(
                "{} : {} centroïdes, mais les noyaux codent 1 bit de gain en dur",
                m.name,
                m.centroids.len()
            ));
        }
        let nblocks = m.d_in / llvq_core::DIM;
        let tail_w = m.d_in % llvq_core::DIM;
        quantized_weights += m.d_out * m.d_in;

        let rotation = match m.rotation_seed {
            None => None,
            Some(seed) => {
                let key = (m.d_in, seed);
                // `entry` rather than contains/insert, but not for the reason
                // clippy gives: `RotationTables::build` runs a Gram–Schmidt and
                // allocates, and the naive form would do that work on every
                // matrix sharing a rotation — 252 builds where 144 are owed.
                if let std::collections::hash_map::Entry::Vacant(e) = rotations.entry(key) {
                    e.insert(RotationTables::build(m.d_in, seed)?);
                }
                Some(key)
            }
        };

        let (stream, payload) = transcode_stream(&fd, &table, &m.indices, &m.gains, layout)?;
        let bytes = payload + (m.d_out * tail_w) as u64 * 4 + m.d_out as u64 * 4;
        runtime_bytes += bytes;

        matrices.push(FusedMatrix {
            name: m.name,
            d_out: m.d_out,
            d_in: m.d_in,
            nblocks,
            tail_w,
            stream,
            gscale: [m.centroids[0] as f32, m.centroids[1] as f32],
            rscale: m.row_scales.iter().map(|&v| v as f32).collect(),
            tail: m.tail.iter().map(|&v| v as f32).collect(),
            rotation,
            bytes,
        });
    }

    let n_raw = read_u32(&mut r)?;
    let mut raw = Vec::with_capacity(n_raw as usize);
    let mut carried_weights = 0usize;
    for _ in 0..n_raw {
        let t = llvq_artifact::read_raw(&mut r, head.version).map_err(|e| e.to_string())?;
        carried_weights += t.len();
        raw.push((t.name.clone(), t.dims.clone(), t.to_f32()));
    }

    let n_blob = read_u32(&mut r)?;
    let mut blobs: HashMap<String, Vec<u8>> = HashMap::new();
    for _ in 0..n_blob {
        let b = llvq_artifact::read_blob(&mut r).map_err(|e| e.to_string())?;
        blobs.insert(b.name, b.bytes);
    }

    Ok(FusedModel {
        layout,
        matrices,
        rotations,
        raw,
        config_json: blobs
            .remove("config.json")
            .ok_or_else(|| format!("{path} ne porte pas de config.json"))?,
        tokenizer_json: blobs
            .remove("tokenizer.json")
            .ok_or_else(|| format!("{path} ne porte pas de tokenizer.json"))?,
        quantized_weights,
        carried_weights,
        file_bytes,
        runtime_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three tables must describe the transform `Rotation` computes.
    ///
    /// Not a restatement of `RotationTables::build`: it rebuilds `Q x` from
    /// the *tables alone*, the way a kernel would — sign bitmap, Walsh–
    /// Hadamard per group, dense block across groups — and requires the
    /// result to match. A packing bug (bit order, row-major vs column-major
    /// `small`, a wrong `inv`) shows up here and nowhere else on this machine.
    #[test]
    fn the_tables_describe_the_rotation() {
        for (n, seed) in [(240usize, 1u64), (2560, 2), (9728, 3), (4096, 4), (96, 5)] {
            let t = RotationTables::build(n, seed).expect("within KMAX");
            let rot = Rotation::new(n, seed);
            let mut rng = llvq_core::SplitMix64::new(seed ^ 0xf00d);
            let x: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();

            let mut want = x.clone();
            rot.apply(&mut want);

            // From the tables only.
            let mut s: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if (t.signbits[i >> 5] >> (i & 31)) & 1 == 1 {
                        -v
                    } else {
                        v
                    }
                })
                .collect();
            let mut len = 1usize;
            while len < t.m {
                for p in 0..n / 2 {
                    let j = (p / len) * (len << 1) + (p % len);
                    let (a, b) = (s[j], s[j + len]);
                    s[j] = a + b;
                    s[j + len] = a - b;
                }
                len <<= 1;
            }
            let inv = t.inv as f64;
            let got: Vec<f64> = if t.k == 1 {
                s.iter().map(|v| v * inv).collect()
            } else {
                let mut out = vec![0.0f64; n];
                for j in 0..t.m {
                    for g in 0..t.k {
                        let mut acc = 0.0;
                        for c in 0..t.k {
                            acc += t.small[g * ROT_KMAX + c] as f64 * s[c * t.m + j] * inv;
                        }
                        out[g * t.m + j] = acc;
                    }
                }
                out
            };

            let nrm = want.iter().map(|v| v * v).sum::<f64>().sqrt();
            let dev = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            assert!(dev / nrm < 1e-6, "n={n} : écart relatif {:.3e}", dev / nrm);
        }
    }

    /// `inv` is `1/√m`, not `1/√n`. On a width with an odd factor the two
    /// differ by `√k` — a factor of 4.4 at `k = 19` — and every output would
    /// be uniformly wrong, which is exactly the kind of error a norm check
    /// catches and an eyeball does not.
    #[test]
    fn the_scale_is_over_the_power_of_two_factor() {
        let t = RotationTables::build(9728, 7).expect("within KMAX");
        assert_eq!((t.m, t.k), (512, 19));
        assert!((t.inv as f64 - 1.0 / 512f64.sqrt()).abs() < 1e-9);
        assert!((t.inv as f64 - 1.0 / 9728f64.sqrt()).abs() > 1e-3);
    }
}
