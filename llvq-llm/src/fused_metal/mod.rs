//! The Tetra kernel reached through candle, on Metal.
//!
//! The Metal twin of [`crate::fused_cuda`], and the adapter half of the port
//! in [`crate::device`]. The MSL lives in `kernels/llvq_tetra48.metal` and is
//! proved on its own by `llvq-metal/tests/tetra48_matches_rust.rs` and
//! `tetra48_matvec_matches_host.rs`, which drive it through metal-rs. This
//! module proves the BINDING: the same kernel, reached from a candle tensor.
//!
//! ## Why none of `llvq-metal` is reused
//!
//! `llvq-metal` pins `metal = "0.29"` (metal-rs). candle reaches Metal through
//! `objc2-metal`, and the two `Buffer` types are unrelated. A buffer made by
//! one cannot be bound by the other's encoder. Only the MSL text crosses.
//!
//! ## Five things about candle's Metal backend that cost a night if unknown
//!
//! Every one was measured on 2026-09-20, in a scratch crate, before this file
//! was written.
//!
//!   1. A bad MSL source **panics**; it does not return `Err`.
//!      `new_library_with_source` ends in `.unwrap()`. So the compile is
//!      wrapped in `catch_unwind` and the payload is re-raised as the error
//!      text, because `served::load_resolved` owes a refusal by name.
//!   2. Forgetting `set_threadgroup_memory_length` is **not** an error. The
//!      kernel dispatches, completes, and writes zeros. No fault, no warning.
//!   3. The allocator hands your buffers back out as scratch. `new_buffer`
//!      reuses anything at `Arc::strong_count == 1`, so the tables and the
//!      weight stream must be OWNED for the life of the model.
//!   4. Output buffers are pooled, rounded up to a power of two, and not
//!      zeroed. A row the kernel skips returns a previous tensor's bytes,
//!      which look like a plausible activation.
//!   5. candle's own `Kernels` cache cannot hold this library: `Source` is a
//!      closed enum over 15 `include_str!` constants. This module carries its
//!      own cache, and its key includes the injected tile, because a define
//!      that fails to land changes the staging and not the arithmetic.
//!
//! ## The files
//!
//! | file | holds |
//! |---|---|
//! | this file | the declarations and the re-exports, nothing else |
//! | `launch` | the quad order, the lane and group sizes, the threadgroup memory limit |
//! | `source` | the kernel texts and `source_of`, which text carries which entry point |
//! | `types` | [`MetalRuntime`], [`MetalTables`], [`MetalTetraProj`], their host copies, the pipeline key |
//! | `tuning` | [`MetalTuning`], the Tetra kernel names and row loops, the int4 tile, the threadgroup checks, `LLVQ_METAL_KERNEL` |
//! | `runtime` | the runtime's core methods, the Tetra matvec launches, the host oracle |
//! | `rot` | [`MetalRotation`] and the Walsh-Hadamard launch |
//! | `int4` | [`MetalInt4Proj`], its upload and its launches |
//! | `embed` | [`MetalEmbedTable`], the token gather and the lm_head |
//! | `port` | the `crate::device` port: the lattice and int4 projections the model sees |
//! | `load` | [`load_resolved`], the one door from a resolved file to the model's projections |
//! | `wave1` | the resident buffers, the f32 prefill rows kernel, the f32-input int4 head |
//! | `moe` | [`MetalMoeSlab`] and one decoded token through the routed experts |
//! | `moe_rows` | a prompt's rows through the resident experts |
//!
//! Each child imports what it uses by name. This file re-exports only what
//! the rest of the workspace names, plus the types its public signatures carry.

mod launch;
mod source;
mod types;
mod tuning;
mod runtime;
mod rot;
mod int4;
mod embed;
mod port;
mod load;
mod wave1;
mod moe;
mod moe_rows;

pub use tuning::{tuning_of, MetalTuning, Q4_TILE_COLS, SERVED_TETRA, TETRA_ROW_LOOPS};
pub use rot::MetalRotation;
pub use int4::MetalInt4Proj;
pub use embed::MetalEmbedTable;
pub use load::{load_resolved, load_resolved_tuned};
pub use moe::{MetalMoeSlab, MoeHostExpert, MoeHostProj, MoeShape, MoeSlabExpert, MOE_DOWN_SG_DEFAULT, MOE_MAX_SELECTED};
pub use moe_rows::MoeRowsMats;
pub use types::{MetalRuntime, MetalTables, MetalTetraProj};

#[cfg(test)]
mod tests;
