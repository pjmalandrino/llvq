//! Projections: the dense or device arms, their row caps, and the group
//! dispatch that shares one rotation between them.

use super::*;

/// One transformer block, with its seven projections reachable by name.
/// A projection: dense weights, or a device kernel reading encoded ones.
///
/// The dense arm is what every published perplexity refers to and what the
/// quantizer operates on. The device arms exist only after a model is loaded
/// from a sealed artifact, by a backend's own loader. They name no backend:
/// the trait objects of [`crate::device`] own their runtime and their
/// buffers, so this enum compiles and runs on every target, and a CPU fake
/// drives the whole dispatch under `cargo test`.
///
/// The two are *not* interchangeable in both directions: `Proj::dense` panics
/// on a device projection rather than returning something plausible, because
/// every caller of it is a quantization path and quantizing an already
/// quantized model is a bug, not a use case.
pub enum Proj {
    Dense(Linear),
    /// One projection whose weights stay encoded, read by a device kernel.
    Lattice(std::sync::Arc<dyn crate::device::LatticeProj>),
    /// One projection served as affine int4 g128 — `v_proj` in the mixed file
    /// the operator chose on 2026-09-08.
    ///
    /// It behaves like [`Proj::Dense`] in every respect that matters to the
    /// caller and like [`Proj::Lattice`] in none: the weights are STORED, in
    /// the **natural basis**, so there is no rotation to carry, no key to
    /// check against, and `prepare` hands back the activation untouched. What
    /// it shares with the other device arms is only that a kernel reads it.
    Int4(std::sync::Arc<dyn crate::device::Int4Proj>),
    /// One part of a row-concatenated group — q, k or v of one layer, or gate
    /// or up. The whole group is launched **once**, by [`group_forward`], and
    /// the result is narrowed back into the parts; there is no per-projection
    /// launch to reach from here (see [`Proj::forward_with`]'s arm).
    GroupPart {
        /// Shared by every projection of the group — `Arc::ptr_eq` on it is how
        /// [`group_forward`] recognises a group, structurally rather than by a
        /// flag that could be set on the wrong three projections.
        group: std::sync::Arc<dyn crate::device::SegGroup>,
        /// First row of this projection inside the group.
        row0: usize,
        /// This projection's own width. `group.d_out()` is the group's.
        d_out: usize,
        /// Rank inside the group, 0-based — its index in `Act::consumers()`.
        rank: usize,
    },
}

/// An activation, in the basis the projections that consume it were quantized
/// in.
///
/// **A value, not a cache entry** — that distinction is the whole correctness
/// argument of lot A4 and [`crate::rotplan`] spells it out. It is produced by
/// [`Proj::prepare`] and consumed by [`Proj::forward_with`] in the same
/// lexical scope, so the property that has to hold is *positive and local* —
/// "computed from this activation, two lines up" — rather than *negative and
/// global*, which is what every keying scheme would have needed.
///
/// ⚠️ **What that argument does and does not buy, stated exactly**, because an
/// earlier draft of this comment claimed more than the type delivers. What is
/// enforced: [`Proj::forward_with`] refuses a `Rotated` whose key is not its
/// own, so a projection can never consume an activation rotated in another
/// basis. What is *not* enforced: this value carries no trace of **which**
/// vector was rotated. Two projections of the same `(layer, act)` share a key
/// by construction, so handing one of them a `Rotated` built from a different
/// activation of that same site would be accepted, and the fused arm — which
/// reads only the rotated buffer — would silently compute on the wrong input.
///
/// Nothing reaches that today: [`Proj::group_forward`] takes a single `x` for
/// the whole group, and the two call sites in [`Block::forward_cached`] cannot
/// diverge. But `prepare` and `forward_with` are `pub`, so the guarantee is
/// held by the caller's shape, not by the type. A lifetime would not fix it —
/// it would tie the value to *a* borrow, not to *the* activation. Closing it
/// properly means carrying the source identity, and that is a design change,
/// not a comment.
///
/// The fields are private on purpose: the only way to obtain one is to rotate
/// an activation, and the only thing that can be done with it is to hand it to
/// a projection that then checks it belongs to it.
pub struct Rotated {
    pub(super) key: Option<RotKey>,
    /// Read by the device arms of [`Proj::forward_with`]. The dense arm
    /// deliberately multiplies the caller's own `x` instead, so that it stays
    /// the call `bin/oracle` certifies.
    ///
    /// It used to carry an `allow(dead_code)` off CUDA, because the only
    /// readers were behind a `cfg` no development machine here compiled. The
    /// device port removed that `cfg`, so the field is live on every target.
    pub(super) t: Tensor,
}

impl Rotated {
    /// Which rotation produced it — `None` in the natural basis, which is
    /// every dense projection.
    pub fn key(&self) -> Option<RotKey> {
        self.key
    }
}

/// Most rows a single fused call may loop over.
///
/// Not an optimization gap that can be papered over: the fused kernel is a
/// matrix–*vector* product, so `l` tokens cost `l` launches. Generation always
/// begins by running the whole prompt through, so `rows > 1` is a live path
/// and refusing it outright was wrong (the very first run found that). The cap
/// survives because a 2048-token scoring window would issue half a million
/// launches and look like a hang; `bin/ppl` keeps the dense path.
pub const MAX_ROWS: usize = 256;
// ⚠️ Read it as a LAUNCH budget, not a row budget, and read it as the budget
// of the paths that spend ONE launch a row: the segmented kernel, and every
// layout that carries no rows kernel. A decode step never approaches it — a
// decode step is one row.
//
// The batched prefill path has its own budget, [`MAX_PREFILL_ROWS`], because
// it bounds a different quantity. Until 2026-09-11 it did not: the cap read
// `MAX_ROWS · batch`, so raising one raised the other, and a served layout's
// segmented path would have moved for a reason that has nothing to do with it.

/// Most rows the batched prefill path may take in one call.
///
/// ## Why it is not [`MAX_ROWS`] times the batch
///
/// It was, and the arithmetic was right: a chunk of `batch` rows spends one
/// launch, so the same launch budget admits `batch` times the rows. What was
/// wrong is that it tied two paths together. `MAX_ROWS` also bounds the
/// segmented kernel, which is served under `Planes14` and takes one launch a
/// row; moving the prefill bound would have moved that one too, four-fold,
/// with no measurement behind it.
///
/// ## Where 4,096 comes from
///
/// From the census, not from a round number. The longest 5-shot MMLU prompt
/// under the Qwen3 tokenizer is **3,096 tokens** — `high_school_european_history`,
/// whose subject mean is 2,796 (*measured* 2026-09-11 from the cached
/// `cais/mmlu` parquet, `docs/mesures/f1e0-2026-09-10.txt`). The previous
/// bound of 1,024 refused **239 of the 2,280** questions every published bar
/// is measured on — five subjects entirely, `professional_law` and four
/// histories. 4,096 admits the whole split and leaves a third of itself spare.
///
/// ## What it costs to be wrong high
///
/// The original cap existed because "a 2048-token scoring window would issue
/// half a million launches and look like a hang". That reason survives, and
/// now it has a number instead of a guess: at the *measured* 5.507 ms a
/// prompt token, 4,096 rows is 23 s of prefill. Slow, bounded, and not a
/// hang. `bin/ppl` still keeps the dense path for its 4,096-token window.
pub const MAX_PREFILL_ROWS: usize = 4096;

/// Which of the two budgets a chunk of `batch` rows answers to.
///
/// Two budgets because two shapes of work, and a function rather than an
/// expression inline because this is the line that decides whether a census
/// question is scored or refused — the kind of line that has to be reachable
/// from a test on a machine with no card.
pub fn row_cap(batch: usize) -> usize {
    match batch > 1 {
        // One launch a chunk: bounded by the longest prompt the census holds.
        true => MAX_PREFILL_ROWS,
        // One launch a row: the budget that has always been a launch budget.
        false => MAX_ROWS,
    }
}

/// Most rows one [`group_forward`] call on `projs` takes, by the branch that
/// call would take: no bound where no projection is rotated, [`MAX_ROWS`] for
/// a segmented group, and [`row_cap`] of the group's batch otherwise.
///
/// A caller that cuts a prompt into chunks reads it so that no chunk is
/// refused. The served Metal path batches nothing, so it is 256 there.
pub fn group_row_cap(projs: &[&Proj]) -> usize {
    if projs.iter().all(|p| p.rot_key().is_none()) {
        return usize::MAX;
    }
    if projs.iter().any(|p| matches!(p, Proj::GroupPart { .. })) {
        return MAX_ROWS;
    }
    row_cap(prefill_rows_of(projs))
}

impl Proj {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // A group of one: `Off` and `On` coincide, so the mode is not a
        // parameter here and no caller has to learn about it.
        Ok(group_forward(&[self], x, RotShare::Off)?
            .pop()
            .expect("a group of one yields one result"))
    }

    /// The rotation this projection's weights were quantized under, or `None`
    /// in the natural basis — which is every dense projection.
    pub fn rot_key(&self) -> Option<RotKey> {
        match self {
            Proj::Dense(_) => None,
            Proj::Lattice(d) => d.rotation(),
            // Stored in the natural basis, so there is nothing to carry and
            // nothing for `check_key` to compare — exactly the dense answer,
            // and for exactly the dense reason.
            Proj::Int4(_) => None,
            Proj::GroupPart { group, .. } => group.rotation(),
        }
    }

    /// The name the artifact stores, for error messages. Dense projections
    /// come out of a `VarBuilder` and carry none.
    pub fn site_name(&self) -> &str {
        match self {
            Proj::Dense(_) => "(dense projection)",
            Proj::Lattice(d) => d.name(),
            Proj::Int4(d) => d.name(),
            Proj::GroupPart { group, rank, .. } => group.part_name(*rank),
        }
    }

    /// Output width.
    pub fn d_out(&self) -> Result<usize> {
        match self {
            Proj::Dense(l) => l.weight().dim(0),
            Proj::Lattice(d) => Ok(d.d_out()),
            Proj::Int4(d) => Ok(d.d_out()),
            // The **part**, not the group: this is the width of the tensor the
            // caller gets back, which is what every consumer reshapes on.
            Proj::GroupPart { d_out, .. } => Ok(*d_out),
        }
    }

    /// Rotate `x` into the basis this projection was quantized in.
    ///
    /// Dense: a strict no-op — the tensor itself, no launch, no allocation
    /// beyond candle's refcount. That arm is what `bin/oracle` certifies at
    /// `max |Δhidden| = 0` and it must stay untouched.
    ///
    /// Fused: one `rot_apply`, one f32 `[1, d_in]` result. The output is a
    /// tensor of full standing rather than a scratch buffer, which is what
    /// lets the caller keep it alive across the group's uses without any
    /// aliasing argument.
    pub fn prepare(&self, x: &Tensor) -> Result<Rotated> {
        match self {
            Proj::Dense(_) => Ok(Rotated { key: None, t: x.clone() }),
            Proj::Lattice(d) => Ok(Rotated {
                key: d.rotation(),
                t: d.prepare(x)?,
            }),
            // No `rot_apply`: the weights are quantized in the basis the
            // activation already arrives in. Byte for byte the dense arm's
            // answer, which is what makes `check_key` accept both in one group
            // and refuse a rotated one beside them.
            Proj::Int4(_) => Ok(Rotated { key: None, t: x.clone() }),
            // One `rot_apply` for the whole group: the parts share one `d_in`
            // and one rotation key by construction, checked at load time.
            Proj::GroupPart { group, .. } => Ok(Rotated {
                key: group.rotation(),
                t: group.prepare(x)?,
            }),
        }
    }

    /// [`Self::prepare`] for a whole chunk of rows, one launch.
    ///
    /// `xs` is `[len, d_in]` and **contiguous** — `group_forward` narrows it out
    /// of the one allocation `row_block` reshapes, so the rows are already
    /// `d_in` apart and nothing is copied to make them so.
    ///
    /// ## A chunk of one is [`Self::prepare`], verbatim
    ///
    /// Same discipline as [`Self::forward_rows`], and for a sharper reason
    /// here: `rotate` refuses `rows != 1` and `rotate_rows` would launch a
    /// grid of one block, which is the same arithmetic through a second entry
    /// point. Taking the one-row path is what keeps every decode step on the
    /// code `bin/oracle` certifies.
    ///
    /// ## What batches and what does not
    ///
    /// `Dense` and `FusedInt4` consume the activation in the basis it arrives
    /// in, so their "rotation" is the chunk itself and costs nothing at any
    /// length. `Fused` takes the rows kernel. `FusedSeg` does not batch — its
    /// group is one launch already and `batches_rows` is false for it, so
    /// `group_forward` never asks it for a chunk of more than one; asking
    /// anyway is refused by `rotate_group`, by name, rather than silently
    /// rotating the first row and using it for four.
    pub fn prepare_rows(&self, xs: &Tensor) -> Result<Rotated> {
        let rows = xs.dim(0)?;
        if rows == 1 {
            return self.prepare(xs);
        }
        match self {
            Proj::Dense(_) => Ok(Rotated { key: None, t: xs.clone() }),
            Proj::Lattice(d) => Ok(Rotated {
                key: d.rotation(),
                t: d.prepare_rows(xs, rows)?,
            }),
            Proj::Int4(_) => Ok(Rotated { key: None, t: xs.clone() }),
            Proj::GroupPart { group, .. } => Ok(Rotated {
                key: group.rotation(),
                t: group.prepare(xs)?,
            }),
        }
    }

    /// Whether a chunk of this projection is one launch rather than one a row.
    ///
    /// Read only to choose the chunk size — the answer changes the SHAPE of
    /// the work and never its result, because a chunk of a projection that
    /// says `false` fans out through `forward_with` row by row.
    pub fn batches_rows(&self) -> bool {
        match self {
            Proj::Lattice(d) => d.rows_per_launch() > 1,
            _ => false,
        }
    }

    /// `y = W X` for a whole chunk of rows, one launch where the layout has a
    /// kernel that takes several.
    ///
    /// ## What it is, and what it is not
    ///
    /// It is a fallback that sometimes goes fast, never a second arithmetic.
    /// A chunk of ONE row is [`Self::forward_with`] verbatim — not a batch of
    /// one — and an arm whose layout carries no rows kernel fans out through
    /// the same function, row by row. So every result this returns is a result
    /// `forward_with` would have returned, and the fast path is proved
    /// bit-identical to it by `tetra48_matches_rust.rs`.
    ///
    /// ## Why a chunk of one is not a batch of one
    ///
    /// `Tensor::cat` of a one-element slice returns that element **verbatim**,
    /// layout included. The rows here are `row_views`' narrows — views at
    /// offset `r · d_in` into one allocation — so a "batch" of one would hand
    /// the kernel a view at a nonzero offset. `FusedRowsOp` refuses that rather
    /// than reading the wrong slice, which is the right failure and still a
    /// failure. Taking the one-row path is what makes it not arise.
    ///
    /// ## What it buys, stated honestly
    ///
    /// BYTES, not launches. `Tensor::cat` copies each row into a fresh
    /// allocation, so a chunk of four costs four copies plus one matvec where
    /// four rows cost four matvecs — the launch count barely moves, and on a
    /// q+k+v chunk it goes UP. What falls is the weight stream: `q_proj`'s
    /// words are read once instead of four times, and the stream is what
    /// costs. On the served 4B that is 776 GB a question against 194.
    pub fn forward_rows(&self, r: &Rotated, xs: &Tensor, len: usize) -> Result<Vec<Tensor>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        if xs.dim(0)? != len {
            candle_core::bail!("{} rows of activation for a chunk of {len}", xs.dim(0)?);
        }
        // One row is the one-row path, byte for byte.
        if len == 1 {
            return Ok(vec![self.forward_with(r, xs)?]);
        }
        match self {
            Proj::Lattice(d) if d.rows_per_launch() > 1 => {
                check_key(self.site_name(), self.rot_key(), r.key)
                    .map_err(candle_core::Error::msg)?;
                // No `Tensor::cat` here any more, and that is the point of the
                // whole change: `prepare_rows` rotated the chunk in one launch
                // and its output is already `[len, d_in]` contiguous, which is
                // the shape this kernel wants. Stacking it cost one device copy
                // a row — 864 a chunk on the served 4B, the largest single term
                // in the prefill's operation count.
                //
                // And one `check_key` covers the chunk because there is one
                // rotation for the chunk. The loop that checked every row's key
                // against the first went with the vector of rotations it was
                // checking: a chunk cannot mix two sites' rotations when it
                // holds one.
                let y = d.matvec_rows(&r.t, len)?;
                (0..len).map(|i| y.narrow(0, i, 1)).collect()
            }
            // An int4 rows kernel: `xs`, the caller's activation, as
            // `forward_with` reads it for int4.
            Proj::Int4(d) if d.rows_per_launch() > 1 => {
                let y = int4_chunked(d.as_ref(), xs)?;
                (0..len).map(|i| y.narrow(0, i, 1)).collect()
            }
            // Dense, int4, a segmented part, or a layout with no rows kernel:
            // the same function, one row at a time. `tv_q4_h` takes no row
            // count and no offset, so int4 is here by construction and not by
            // omission.
            //
            // `r.t` is narrowed along with `xs`. In practice only `Dense` and
            // `FusedInt4` reach here with `len > 1` — a `Fused` whose layout
            // has no rows kernel is never given a chunk, because `batch` is 1
            // unless some projection of the group batches — and for both of
            // those `forward_with` reads `x` and not `r.t`. The narrow is
            // still done, because "in practice" is not a contract and a view
            // costs nothing.
            _ => (0..len)
                .map(|i| {
                    let one = Rotated { key: r.key, t: r.t.narrow(0, i, 1)? };
                    self.forward_with(&one, &xs.narrow(0, i, 1)?)
                })
                .collect(),
        }
    }

    /// `y = W x`, given `x` already rotated by [`Self::prepare`].
    ///
    /// `x` is passed beside the rotated form and is **not** redundant: it
    /// carries the caller's shape, so the result comes out with the rank the
    /// caller handed in, and on the dense arm it is the tensor the `Linear`
    /// multiplies — literally today's call.
    ///
    /// Refuses a rotation that is not this projection's, naming it. See
    /// [`check_key`].
    pub fn forward_with(&self, r: &Rotated, x: &Tensor) -> Result<Tensor> {
        check_key(self.site_name(), self.rot_key(), r.key).map_err(candle_core::Error::msg)?;
        match self {
            Proj::Dense(l) => l.forward(x),
            Proj::Lattice(d) => d.matvec(&r.t, x.dims()),
            // `x`, not `r.t` — the dense arm's choice, and the same one: `r.t`
            // IS `x` here, and reading it from the argument the caller handed
            // in is what keeps that true if `prepare` ever stops being the
            // identity for this arm.
            Proj::Int4(d) => d.matvec(x, x.dims()),
            // Deliberately unreachable, and it must stay that way: a launch per
            // projection here would compute the group's whole width and then
            // throw most of it away — silently, and at three times the cost.
            Proj::GroupPart { group, .. } => candle_core::bail!(
                "{} belongs to a fused group of {} rows: it only launches through \
                 model::group_forward, which emits ONE matvec for the whole group.",
                self.site_name(),
                group.d_out()
            ),
        }
    }

    /// The dense weights, for the quantizer.
    ///
    /// Panics on a fused projection. That is the intended behaviour: the
    /// alternative is an `Option` every caller would `unwrap`, at five sites
    /// that can none of them proceed without it.
    pub fn dense(&self) -> &Linear {
        match self {
            Proj::Dense(l) => l,
            Proj::Lattice(d) => panic!(
                "{} is a device projection: the quantizer only works on a dense model",
                d.name()
            ),
            Proj::GroupPart { group, rank, .. } => panic!(
                "{} is a device projection: the quantizer only works on a dense model",
                group.part_name(*rank)
            ),
            Proj::Int4(d) => panic!(
                "{} is served as int4: the quantizer only works on a dense model",
                d.name()
            ),
        }
    }

    pub fn dense_mut(&mut self) -> &mut Linear {
        match self {
            Proj::Dense(l) => l,
            Proj::Lattice(d) => panic!(
                "{} is a device projection: the quantizer only works on a dense model",
                d.name()
            ),
            Proj::GroupPart { group, rank, .. } => panic!(
                "{} is a device projection: the quantizer only works on a dense model",
                group.part_name(*rank)
            ),
            Proj::Int4(d) => panic!(
                "{} is served as int4: the quantizer only works on a dense model",
                d.name()
            ),
        }
    }
}

/// The single-row views of an activation, in row order.
///
/// One contiguity pass for the whole group, then `narrow` along dim 0 — which
/// yields a view at an *offset* into one allocation, and that is exactly what
/// `rot_apply`'s `x_off` argument exists for. Re-materializing each row would
/// be a copy per row per projection.
///
/// Extracted from [`group_forward`] with [`regroup`] because that function's
/// rotated branch is reachable only with a fused projection, i.e. on no
/// machine this test suite runs on. The shape bookkeeping is the part of it
/// that can be silently wrong — a transposed row order returns finite,
/// plausible, wrong tokens — so it lives in two functions that a CPU tensor
/// can exercise.
/// Rows one prefill launch carries, 1 where no projection batches.
///
/// From the PROJECTION that will take the chunk, not from a constant. Under
/// `LLVQ_PREFILL` a unit is compiled at eight or sixteen rows and
/// `tile::PREFILL_ROWS` still reads four; handing a chunk of eight to a kernel
/// compiled at four is the silent corruption this reader exists to prevent.
/// `matvec_rows` would refuse it, but only after the chunking had decided the
/// shape of the work.
///
/// One number, asked once. This used to read a `bool` and a `usize` off a
/// runtime, and those two could disagree. A single `rows_per_launch` cannot
/// disagree with itself.
///
/// Every projection of a group shares one runtime by construction, so the
/// first that batches answers for the group.
fn prefill_rows_of(projs: &[&Proj]) -> usize {
    for p in projs {
        if let Proj::Lattice(d) = p {
            let rows = d.rows_per_launch();
            if rows > 1 {
                return rows;
            }
        }
    }
    1
}

pub fn row_views(x: &Tensor) -> Result<Vec<Tensor>> {
    let flat = row_block(x)?;
    (0..flat.dim(0)?).map(|r| flat.narrow(0, r, 1)).collect()
}

/// The same rows as ONE contiguous `[rows, d_in]` tensor.
///
/// What [`row_views`] narrows out of, handed over whole. A caller that wants a
/// chunk narrows it once and gets a view; a caller that wanted a vector of
/// single rows and then stacked four of them back together was paying a device
/// copy a row to rebuild a shape it already had.
pub fn row_block(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let d_in = *dims.last().expect("rank >= 1");
    let rows: usize = dims[..dims.len() - 1].iter().product();
    x.contiguous()?.reshape((rows, d_in))
}

/// `y = W x` for the rows of `flat`, `[rows, d_in]`, through an int4 rows
/// kernel in launches of [`crate::device::Int4Proj::rows_per_launch`] rows,
/// as `[rows, d_out]`. Each row is the matvec's by the port's contract, so
/// only the launch count moves. The one place the model reaches that kernel.
fn int4_chunked(d: &dyn crate::device::Int4Proj, flat: &Tensor) -> Result<Tensor> {
    let rows = flat.dim(0)?;
    let k = d.rows_per_launch();
    let parts = (0..rows)
        .step_by(k)
        .map(|row0| {
            let len = k.min(rows - row0);
            d.matvec_rows(&flat.narrow(0, row0, len)?, len)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&parts, 0)
}

/// Reassemble one projection's per-row results into the caller's shape.
///
/// `dims` is the shape the caller handed to [`group_forward`]; the result is
/// that shape with `d_out` as its last axis. Inverse of [`row_views`] up to
/// the change of width.
pub fn regroup(rows: &[Tensor], dims: &[usize], d_out: usize) -> Result<Tensor> {
    let mut d = dims.to_vec();
    *d.last_mut().expect("rank >= 1") = d_out;
    Tensor::cat(rows, 0)?.reshape(d)
}

/// Run projections that share one activation.
///
/// This is the call site lot A4 turns on. `Block::forward_cached` hands q/k/v
/// one group and gate/up another; o_proj and down_proj are groups of one. Under
/// [`RotShare::On`] the group's activation is rotated **once**, and every
/// projection is handed that value; under [`RotShare::Off`] each rotates its
/// own, which is launch for launch what shipped before this lot.
///
/// ## Two branches, and the first one is the one that must not move
///
/// When no projection of the group carries a rotation — every dense model, so
/// every path `bin/oracle`, `bin/ppl`, `bin/mmlu` and the quantizer take —
/// there is nothing to hoist, and the group degenerates to `prepare` (a clone)
/// then `forward_with` (the `Linear`'s own `forward`) on the **whole** tensor,
/// in declaration order. No reshape, no row loop, no mode dependence. The
/// forward pass certified at `max |Δhidden| = 0` is unchanged, and
/// `the_restructured_block_is_the_same_calls_in_the_same_order` requires token
/// equality rather than a tolerance to say so.
///
/// The rotated branch does split rows, because the fused kernel is a matvec:
/// the row loop that used to live inside `FusedRuntime::forward` moves up one
/// level, so that a prefill row is rotated once for its whole group instead of
/// once per projection. Under `Off` the resulting order is still
/// rot/matvec/rot/matvec projection by projection — the old order exactly.
pub fn group_forward(projs: &[&Proj], x: &Tensor, share: RotShare) -> Result<Vec<Tensor>> {
    if projs.is_empty() {
        return Ok(Vec::new());
    }
    if projs.iter().all(|p| p.rot_key().is_none()) {
        return projs
            .iter()
            .map(|p| match p {
                // `tv_q4_h` is a matvec: one vector, no row count. A group with
                // no rotation that holds one — a lone `o_proj` or `down_proj`
                // served int4 — fans out row by row, as the rotated branch
                // does for an int4 `v_proj`. `Proj::Dense` still takes the
                // whole tensor, so the dense path does not move.
                Proj::Int4(_) => int4_rows(p, x),
                _ => {
                    let r = p.prepare(x)?;
                    p.forward_with(&r, x)
                }
            })
            .collect();
    }

    // After the dense branch, which must not move, and before the row loop:
    // a fused group replaces the loop's per-projection launches with one.
    if let Some(plan) = SegPlan::of(projs)? {
        return plan.run(x);
    }

    let dims = x.dims();
    let rows: usize = dims[..dims.len() - 1].iter().product();
    // The chunk: `PREFILL_ROWS` where a projection of this group can spend it,
    // and ONE everywhere else — which is the loop that shipped, unchanged, for
    // every layout that carries no rows kernel and for every group of one row.
    //
    // It changes the shape of the work and never its result: a projection that
    // cannot batch fans out through `forward_with`, row by row, inside the same
    // chunk. So a mixed q+k+v group of the served file issues one launch for q,
    // one for k, and four for the `v_proj` that is int4.
    let batch = match rows > 1 {
        true => prefill_rows_of(projs),
        false => 1,
    };
    let cap = row_cap(batch);
    if rows > cap {
        candle_core::bail!(
            "{}: {rows} vectors at once, more than {cap} (at {batch} rows a launch, \
             {} launches a projection). Past that the loop issues more launches than \
             it does arithmetic; `bin/ppl`'s 4096-token window keeps the dense path.",
            projs[0].site_name(),
            rows.div_ceil(batch)
        );
    }
    // The rows as ONE contiguous `[rows, d_in]`, not as a vector of views: a
    // chunk is then `narrow(0, row0, len)`, which is a view into it and costs
    // nothing, and the rotation kernel can read `len` rows from one pointer.
    let flat = row_block(x)?;
    let per_site = drive_rows_batched(
        share,
        projs,
        rows,
        batch,
        // A projection with no rotation key takes no share of the group's:
        // `Proj::Dense` and `Proj::Int4` both consume the activation in
        // the basis it arrives in.
        |p: &&Proj| p.rot_key().is_some(),
        |p: &&Proj, row0: usize, len: usize| p.prepare_rows(&flat.narrow(0, row0, len)?),
        |p: &&Proj, r: &Rotated, row0: usize, len: usize| {
            p.forward_rows(r, &flat.narrow(0, row0, len)?, len)
        },
    )?;

    per_site
        .into_iter()
        .zip(projs)
        .map(|(rows_of_site, p)| regroup(&rows_of_site, dims, p.d_out()?))
        .collect()
}

/// `y = W x` for every row of `x` through a single-vector int4 kernel,
/// reassembled into the caller's shape. One row is `forward_with` verbatim,
/// so a decode step takes the call it always took.
fn int4_rows(p: &Proj, x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let rows: usize = dims[..dims.len() - 1].iter().product();
    if rows == 1 {
        let r = p.prepare(x)?;
        return p.forward_with(&r, x);
    }
    if let Proj::Int4(d) = p {
        if d.rows_per_launch() > 1 {
            let mut out = dims.to_vec();
            *out.last_mut().expect("rank >= 1") = d.d_out();
            return int4_chunked(d.as_ref(), &row_block(x)?)?.reshape(out);
        }
    }
    let ys = row_views(x)?
        .iter()
        .map(|xi| {
            let r = p.prepare(xi)?;
            p.forward_with(&r, xi)
        })
        .collect::<Result<Vec<_>>>()?;
    regroup(&ys, dims, p.d_out()?)
}

/// One segmented launch, and the map back to the projections it stands for.
///
/// Built by [`Self::of`] from the projections `group_forward` was handed, so
/// the model states the row order **independently** of the loader that assigned
/// it. Two places have to agree and neither may assume: a group read in the
/// order k,q,v would otherwise return finite, plausible, wrong numbers with no
/// assertion anywhere.
struct SegPlan<'a> {
    /// Borrowed as the `Arc` rather than as its target: the group's identity
    /// test below is then `Arc::ptr_eq`, the only comparison that cannot be
    /// fooled, and the field is initialised straight from the enum's binding
    /// with no coercion in between.
    ///
    /// ⚠️ This is a FAT pointer, `Arc<dyn SegGroup>`. `Arc::ptr_eq` compares
    /// the data address alone and ignores the vtable, which is what makes the
    /// test correct here; that has held since Rust 1.76 and
    /// `rust-toolchain.toml` pins 1.95.0. That file's own header warns it pins
    /// FORWARD, not backward. Never `Arc::ptr_eq` a device trait object to
    /// "strengthen" this check: two coercions of one allocation are equal, but
    /// a vtable address is not guaranteed unique.
    ///
    /// The runtime is not a field any more. The group owns it.
    group: &'a std::sync::Arc<dyn crate::device::SegGroup>,
    /// `(rank, row0, d_out)` per projection, in the order the caller handed
    /// them — the shape [`crate::fused::check_seg_spans`] validates.
    spans: Vec<(usize, usize, usize)>,
}

impl<'a> SegPlan<'a> {
    /// `Some` when every projection handed in is a part of **one** group and,
    /// together and in this order, they are the whole group.
    ///
    /// `None` when none of them is segmented — every dense and every unfused
    /// build. `Err` when they are mixed, partial, or out of order: a wiring
    /// bug, never a fallback.
    fn of(projs: &[&'a Proj]) -> Result<Option<Self>> {
        if !projs.iter().any(|p| matches!(p, Proj::GroupPart { .. })) {
            return Ok(None);
        }
        let mut plan: Option<SegPlan<'a>> = None;
        // `&p` rather than `p`: the item is a `&&Proj`, and destructuring the
        // outer reference is what ties the bindings to the projections'
        // lifetime rather than to this loop's.
        for &p in projs {
            let Proj::GroupPart { group, row0, d_out, rank } = p else {
                candle_core::bail!(
                    "{} belongs to no fused group while its group neighbours do. A \
                     segmented launch cannot cover half a site",
                    p.site_name()
                );
            };
            match &mut plan {
                None => {
                    plan = Some(SegPlan {
                        group,
                        spans: vec![(*rank, *row0, *d_out)],
                    })
                }
                Some(acc) => {
                    // Pointer identity, not a key: two groups of two layers
                    // share a `d_in` and a shape, and comparing anything else
                    // would accept a q/k/v triple assembled across layers.
                    if !std::sync::Arc::ptr_eq(acc.group, group) {
                        candle_core::bail!(
                            "{} belongs to a different fused group than {}, two groups in \
                             a single call",
                            p.site_name(),
                            acc.group.name()
                        );
                    }
                    acc.spans.push((*rank, *row0, *d_out));
                }
            }
        }
        let plan = plan.expect("the `any` above found at least one segmented projection");
        crate::fused::check_seg_spans(plan.group.name(), &plan.spans, plan.group.d_out())
            .map_err(candle_core::Error::msg)?;
        Ok(Some(plan))
    }

    /// One rotation and one matvec per row, then one view per projection.
    fn run(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let dims = x.dims();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows > MAX_ROWS {
            candle_core::bail!(
                "{}: {rows} vectors at once, more than {MAX_ROWS}. The segmented kernel is \
                 still a matvec, it loops, so the cost is linear.",
                self.group.name()
            );
        }
        let slices = row_views(x)?;

        // One rotation and one matvec per row, whatever `RotShare` says: a
        // fused group is ONE site, so there is no second arm left to keep.
        // That is why `fused::check_fuse` refuses `LLVQ_FUSE=1` beside
        // `LLVQ_ROT_SHARE=0` — accepting it would let an `LLVQ_FUSE` A/B move
        // the hoist as well. `drive_rows` is deliberately not called here for
        // the same reason: its two arms would coincide, and calling it would
        // suggest a control that is gone.
        let mut per_row = Vec::with_capacity(rows);
        for slice in slices.iter().take(rows) {
            let r = self.group.prepare(slice)?;
            per_row.push(self.group.matvec(&r, slice.dims())?);
        }
        // `[rows, group.d_out]`, then one view per projection on the last axis.
        let stacked = Tensor::cat(&per_row, 0)?;
        self.spans
            .iter()
            .map(|&(_, row0, d)| {
                let mut shape = dims.to_vec();
                *shape.last_mut().expect("rank >= 1") = d;
                // `.contiguous()` unconditionally, and it is not defensive:
                //  * at `rows == 1` — the decode path this lot measures — the
                //    narrow is already contiguous (`Shape::is_contiguous` skips
                //    axes of extent ≤ 1), so this is a clone and costs nothing;
                //  * at `rows > 1` the narrow is NOT contiguous (axis 0 has
                //    stride `group.d_out` against an accumulator of `d`), so
                //    the caller's `reshape` would materialise anyway. Doing it
                //    here makes *where* the copy happens explicit, and it is
                //    the same volume `regroup`'s `Tensor::cat` already copies.
                // It also closes an aliasing retention: `KvCache::append`
                // stores `v.clone()`, and a view would keep the whole
                // 6144-wide buffer alive behind a 1024-wide tensor.
                stacked.narrow(1, row0, d)?.contiguous()?.reshape(shape)
            })
            .collect()
    }
}
