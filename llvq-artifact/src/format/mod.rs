//! Reading and writing the `LVQ1` … `LVQ5` stream.
//!
//! ## What a matrix needs, and why each part is there
//!
//! | field | why it cannot be dropped |
//! |---|---|
//! | index + gain per block | the code itself |
//! | one scale per output row | the gain code is relative to it |
//! | the matrix's gain centroids | fitted per matrix, not global |
//! | the rotation seed | codes live in the **rotated** basis |
//! | the tail columns | `KeepExact` leaves them unquantized |
//!
//! `every_stored_field_is_load_bearing` in the test suite corrupts each of
//! them in turn and demands the reconstruction move.
//!
//! ## Bit-exactness, and where the f64s are
//!
//! Scales and centroids are stored in **f64**, not f16. `reconstruct` computes
//! `centroids[g] * row_scale` in f64; storing a rounded scale would change
//! that product and the decoded weights would differ from the evaluated ones
//! in the last bits. That costs 64 bits per output row instead of 16 —
//! 0.0146 bits/weight on Qwen3-4B — and it is **counted**, not waved away.
//! Making the scales exactly f32-representable at fit time would recover it;
//! that is an optimization, not a correctness fix, and it is not done here.
//!
//! ## A code kind per record (`LVQ5`)
//!
//! An index has always been a v1 ball index — 47 or 48 bits, one of 383
//! classes. Tetra ([`llvq_search::tetra`]) is a second map from 47 bits to Λ₂₄
//! with the same width and none of the same meaning: a Tetra word read as a
//! ball index is in range, decodes to a lattice point, and is wrong. So the
//! file says which map it uses, beside a second fingerprint for the Tetra
//! map, and every reader of a record consults the kind before it trusts any
//! width.
//!
//! *Which* map is a fact about a matrix, not about a file. Q5 serves
//! `v_proj` in int4 g128 beside Tetra matrices (`docs/ROADMAP.md` §2.3), and
//! one header field cannot say that. So from `LVQ5` every record carries its
//! own [`CodeKind`] tag, immediately after its shell cap, and the header
//! carries two facts instead of one: the file's **default** kind — what
//! [`ArtifactWriter::push`] writes — and [`KindSet`], the kinds the file
//! declares it may hold. The set is what refusals read
//! ([`crate::runtime::require_ball_kinds`]): a tool with no layout for Tetra
//! has to stop at the header of a file with one Tetra matrix in it, not at
//! whichever record it reaches first, and the header precedes every record
//! by construction.
//!
//! ## A record that stores its weights: `Int4G128`
//!
//! Tag 2 is [`CodeKind::Int4G128`]: `v_proj` served as affine int4 over
//! groups of 128, beside Tetra matrices in one file (`docs/ROADMAP.md` §2.3).
//! It stores `w = scale·q + bias` in the **natural** basis — 4 bits per
//! weight plus an f16 scale and an f16 bias per group, 4.250 b/weight
//! exactly — so its record has no centroids, no row scales, no tail and no
//! rotation seed. [`read_record`] is the only entry that reads it;
//! [`read_matrix_raw`] refuses it by name, which is what makes every
//! lattice-only tool of this repository refuse it without being touched.
//!
//! Its head is deliberately **not** readable by a lattice reader that ignores
//! the kind. Such a reader looks for the payload length after
//! `n_cent·8 + d_out·8 + tail·4` bytes; on an int4 record it overshoots by at
//! least `d_out·8` and takes a length out of the nibbles, which is a
//! [`Error::Truncated`] and not a plausible matrix. Writing `d_out` constant
//! row scales would have made the two heads agree at the cost of 64 bits per
//! output row — 0.025 b/weight on `v_proj` — for no information, and this
//! format counts its bits. The price of the choice is that the one generic
//! record walker in the repository (`llvq_llm::artifact2::shard_extent`) has
//! to know the branch.
//!
//! The record is otherwise unchanged: a Tetra record carries
//! [`TETRA_SHELL_CAP`] in its `shell_cap` field and one gain bit. That kind
//! tag is the only place a v5 record differs from a v4 one, which is why
//! every record entry point below takes the file's `version` — a v5 record
//! read as v4 mistakes its kind tag for a centroid count, a v4 record read
//! as v5 mistakes its centroid count for a kind, and both must fail loudly
//! rather than decode something plausible.

mod checks;
mod decode;
mod header;
mod int4;
mod kinds;
mod read;
mod write;
mod writer;

pub(crate) use checks::u32_field;
pub use decode::{decode_matrix, decode_raw, read_matrix, read_matrix_with, split_name};
pub use header::{
    read_all, read_header, write_header, write_header_kind, write_header_kinds, Header,
};
pub use int4::{read_record, write_matrix_int4, write_record, Int4Matrix, Record};
pub use kinds::{
    CodeKind, Codebook, Codebooks, KindSet, DEFAULT_VERSION, FIRST_FINGERPRINTED_VERSION,
    FIRST_KINDED_VERSION, INT4G128_BITS, INT4G128_GROUP, INT4G128_SHELL_CAP, MAGIC, MAGIC_V1,
    MAGIC_V2, MAGIC_V3, MAGIC_V4, MAGIC_V5, RESERVED_INT4G128_TAG, TETRA_SHELL_CAP, VERSION,
};
pub use read::read_matrix_raw;
pub use write::{
    code_stream, code_widths, write_matrix, write_matrix_raw, write_matrix_with, QuantizedMatrix,
    RawMatrix,
};
pub use writer::ArtifactWriter;
// Named only by the header's links, which resolved in the one file.
#[cfg(doc)]
use crate::Error;
