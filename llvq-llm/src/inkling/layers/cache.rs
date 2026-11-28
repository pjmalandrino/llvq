//! What one layer visit keeps between calls: [`SlotCache`] with its growth, its window bound, its
//! rollback marks and its snapshot form. One file because every method here owns the same buffers.

use candle_core::{Device, Result, Tensor};

use super::ConvState;

/// Positions a slot's key and value buffers are allocated in.
const KV_GRAIN: usize = 256;

/// The capacity after growing from `cap` to hold `need` positions: at least
/// half again, in whole grains.
fn grown_capacity(cap: usize, need: usize) -> usize {
    need.max(cap + cap / 2).div_ceil(KV_GRAIN) * KV_GRAIN
}

/// Positions past the window a bounded sliding slot keeps by default
/// ([`SlotCache::bound`]): the deepest rollback it always accepts.
pub const DEFAULT_KV_SLACK: usize = 256;

/// The storage bound of a sliding slot ([`SlotCache::bound`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KvBound {
    window: usize,
    slack: usize,
}

impl KvBound {
    /// Positions kept behind the last one when the slot drops its oldest.
    fn retain(self) -> usize {
        self.window + self.slack
    }

    /// The capacity a bounded slot settles at: what it keeps, and room for
    /// `slack` more positions before it drops its oldest again.
    fn settled_capacity(self) -> usize {
        self.window + 2 * self.slack
    }
}

/// What one layer visit keeps between calls: keys and values after their
/// norms (what `cache.update` stores, line 612), and the last `K - 1` inputs
/// of the key, value, post-attention and post-MLP convolutions (`conv_idx` 0
/// to 3, lines 563-566 and 868-873).
///
/// Keys and values live in buffers allocated for more positions than they
/// hold, grown by half when full: the new rows are written in place and the
/// attention reads a view of the positions held. Keys are held transposed,
/// `[kv_heads, d, cap]`, so the score product reads the head repeat of that
/// view with no transpose copy; values are `[kv_heads, cap, d]`. A clone
/// copies the buffers, since the next append writes into them.
///
/// A sliding slot may be bounded ([`Self::bound`]): it then keeps only the
/// positions a later row can see plus a slack, buffer row 0 holding position
/// [`Self::base`]. A full slot, and a slot never bounded, keeps every
/// position from 0.
#[derive(Debug, Default)]
pub struct SlotCache {
    k_buf: Option<Tensor>,
    v_buf: Option<Tensor>,
    /// Tokens seen, counted from position 0.
    len: usize,
    /// The position buffer row 0 holds: 0 unless the slot is bounded.
    base: usize,
    bound: Option<KvBound>,
    /// Positions the buffers hold at least once they exist ([`Self::reserve`]).
    min_capacity: usize,
    pub conv_k: ConvState,
    pub conv_v: ConvState,
    pub conv_attn: ConvState,
    pub conv_mlp: ConvState,
    /// While the fused glue steps this slot ([`LayerWeights::forward_fused`]),
    /// the post-attention and post-MLP states live in these device rings and
    /// `conv_attn` / `conv_mlp` are stale. Every other reader settles them
    /// first ([`Self::settle_rings`]); a clone and a mark read them back.
    #[cfg(ink_glue)]
    pub(super) fused: Option<super::glue::FusedRings>,
    /// While the attention kernels step this slot, the key and value states
    /// live in these device rings and `conv_k` / `conv_v` are stale. A
    /// several-row attention settles them first ([`Self::settle_attn`]); a
    /// clone and a mark read them back; a rollback drops them.
    #[cfg(ink_glue)]
    pub(super) attn: Option<super::attn_dense::AttnSlot>,
}

impl Clone for SlotCache {
    fn clone(&self) -> Self {
        let deep = |b: &Option<Tensor>| b.as_ref().map(|t| t.copy().expect("copy of a cache buffer"));
        let [conv_k, conv_v, conv_attn, conv_mlp] = self.conv_states().expect("read back of the convolution rings");
        Self {
            k_buf: deep(&self.k_buf),
            v_buf: deep(&self.v_buf),
            len: self.len,
            base: self.base,
            bound: self.bound,
            min_capacity: self.min_capacity,
            conv_k,
            conv_v,
            conv_attn,
            conv_mlp,
            #[cfg(ink_glue)]
            fused: None,
            #[cfg(ink_glue)]
            attn: None,
        }
    }
}

/// What [`SlotCache::mark`] keeps: the positions held and the four
/// convolution states (key, value, post-attention, post-MLP). The states are
/// never written in place, so a mark holds handles, not copies.
#[derive(Debug, Clone)]
pub struct SlotMark {
    len: usize,
    conv: [ConvState; 4],
}

impl SlotMark {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl SlotCache {
    /// Tokens held.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The state needed to come back here. Rings the fused glue writes in
    /// place are read back into new tensors, so the mark keeps copies.
    pub fn mark(&self) -> SlotMark {
        SlotMark { len: self.len, conv: self.conv_states().expect("read back of the convolution rings") }
    }

    /// The four convolution states (key, value, post-attention, post-MLP),
    /// the last two read back from the rings when the fused glue holds them.
    fn conv_states(&self) -> Result<[ConvState; 4]> {
        #[cfg(ink_glue)]
        let (conv_k, conv_v) = match &self.attn {
            Some(a) => {
                let (k, v) = a.states()?;
                (ConvState::Block(k), ConvState::Block(v))
            }
            None => (self.conv_k.clone(), self.conv_v.clone()),
        };
        #[cfg(not(ink_glue))]
        let (conv_k, conv_v) = (self.conv_k.clone(), self.conv_v.clone());
        #[cfg(ink_glue)]
        if let Some(r) = &self.fused {
            let (a, m) = r.states()?;
            return Ok([conv_k, conv_v, ConvState::Block(a), ConvState::Block(m)]);
        }
        Ok([conv_k, conv_v, self.conv_attn.clone(), self.conv_mlp.clone()])
    }

    /// Bring the key and value states back from the attention kernels'
    /// rings, if they hold them; the rings are dropped. Nothing changes otherwise.
    pub fn settle_attn(&mut self) -> Result<()> {
        #[cfg(ink_glue)]
        if let Some(a) = self.attn.take() {
            let (k, v) = a.states()?;
            self.conv_k = ConvState::Block(k);
            self.conv_v = ConvState::Block(v);
        }
        Ok(())
    }

    /// Whether the attention kernels' rings hold the key and value states.
    pub fn has_attn_rings(&self) -> bool {
        #[cfg(ink_glue)]
        return self.attn.is_some();
        #[cfg(not(ink_glue))]
        false
    }

    /// Bring the post-attention and post-MLP states back from the fused
    /// glue's rings, if it holds them; the rings are dropped. Nothing changes
    /// otherwise.
    pub fn settle_rings(&mut self) -> Result<()> {
        #[cfg(ink_glue)]
        if let Some(r) = self.fused.take() {
            let (a, m) = r.states()?;
            self.conv_attn = ConvState::Block(a);
            self.conv_mlp = ConvState::Block(m);
        }
        Ok(())
    }

    /// Whether the fused glue's rings hold the post-attention and post-MLP states.
    pub fn has_rings(&self) -> bool {
        #[cfg(ink_glue)]
        return self.fused.is_some();
        #[cfg(not(ink_glue))]
        false
    }

    /// Whether [`Self::rollback`] to `m` would be accepted; nothing changes.
    /// Refused when `m` holds more positions than the slot, and, in a bounded
    /// slot, when a row at `m.len()` would see a key the slot has dropped: a
    /// rollback deeper than the slack.
    pub fn check_rollback(&self, m: &SlotMark) -> Result<()> {
        if m.len > self.len {
            candle_core::bail!("slot rollback: the mark holds {} positions, the slot {}", m.len, self.len);
        }
        if let Some(b) = self.bound {
            let need_from = (m.len + 1).saturating_sub(b.window);
            if m.len > 0 && need_from < self.base {
                candle_core::bail!(
                    "slot rollback: too deep for a bounded sliding slot: the mark at position {} needs keys from \
                     position {need_from}, the slot keeps them from {} (window {}, slack {}, {} positions seen)",
                    m.len,
                    self.base,
                    b.window,
                    b.slack,
                    self.len
                );
            }
        }
        Ok(())
    }

    /// Back to `m`: the positions past `m.len()` are forgotten (the next
    /// append writes over them) and the four convolution states are those of
    /// `m`. Refused as [`Self::check_rollback`] says, the slot unchanged. The
    /// caller checks that the positions up to `m.len()` are those the mark saw.
    pub fn rollback(&mut self, m: &SlotMark) -> Result<()> {
        self.check_rollback(m)?;
        // Back to no token at all: buffer row 0 holds position 0 again.
        if m.len < self.base {
            self.base = m.len;
        }
        // the next fused step refills its rings from the states restored below
        #[cfg(ink_glue)]
        {
            self.fused = None;
            self.attn = None;
        }
        self.len = m.len;
        let [k, v, attn, mlp] = m.conv.clone();
        self.conv_k = k;
        self.conv_v = v;
        self.conv_attn = attn;
        self.conv_mlp = mlp;
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Positions the buffers can hold before they grow.
    pub fn capacity(&self) -> usize {
        self.v_buf.as_ref().map_or(0, |v| v.dim(1).expect("rank-3 values"))
    }

    /// The position of the first key and value the slot holds (buffer row 0):
    /// 0 unless the slot is bounded and has dropped its oldest positions.
    pub fn base(&self) -> usize {
        self.base
    }

    /// Positions the buffers hold: `base()..len()`.
    fn held(&self) -> usize {
        self.len - self.base
    }

    /// Keep only what a later row of a sliding layer with `window` can see,
    /// plus `slack` positions for a rollback. Set on an empty slot only.
    pub fn bound(&mut self, window: usize, slack: usize) -> Result<()> {
        if self.len > 0 || self.k_buf.is_some() {
            candle_core::bail!("slot bound: set on an empty slot, this one holds {} positions", self.len);
        }
        if window == 0 {
            candle_core::bail!("slot bound: a window of 0 positions");
        }
        self.bound = Some(KvBound { window, slack });
        Ok(())
    }

    /// Whether the slot is bounded ([`Self::bound`]).
    pub fn is_bounded(&self) -> bool {
        self.bound.is_some()
    }

    /// Views of what the slot holds, positions `base()..len()`: keys
    /// `[kv_heads, S, d]` and values `[kv_heads, S, d]`. They share the
    /// buffers: read them before the next append.
    pub fn keys_values(&self) -> Result<Option<(Tensor, Tensor)>> {
        let held = self.held();
        match (&self.k_buf, &self.v_buf) {
            (Some(k), Some(v)) => Ok(Some((k.narrow(2, 0, held)?.transpose(1, 2)?, v.narrow(1, 0, held)?))),
            _ => Ok(None),
        }
    }

    /// Write `k` and `v` `[kv_heads, T, d]` at positions `len..len + T`, in
    /// place, growing the buffers first if they are full. Returns views of
    /// every position held: keys `[kv_heads, d, S]`, values `[kv_heads, S, d]`.
    /// Make room for `positions`: the buffers, once they exist, hold at
    /// least that many before they grow again. The values held do not change.
    /// A bounded slot ignores it: its size does not follow the length.
    pub fn reserve(&mut self, positions: usize) -> Result<()> {
        if self.bound.is_some() {
            return Ok(());
        }
        self.min_capacity = self.min_capacity.max(positions);
        if positions > self.capacity() {
            if let Some(k) = self.k_buf.clone() {
                let (nkv, d, _) = k.dims3()?;
                self.grow(positions.div_ceil(KV_GRAIN) * KV_GRAIN, nkv, d, &k)?;
            }
        }
        Ok(())
    }

    /// New buffers of `cap` positions, like `like` in dtype and device, the
    /// positions held copied in once.
    fn grow(&mut self, cap: usize, nkv: usize, d: usize, like: &Tensor) -> Result<()> {
        self.relocate(self.base, cap, nkv, d, like)
    }

    /// New buffers of `cap` positions, like `like` in dtype and device, the
    /// positions `new_base..len` copied in once at row 0. The old buffers go
    /// when their last view does: these are new allocations, never a view of
    /// the old ones (a narrow would keep the whole old buffer alive).
    fn relocate(&mut self, new_base: usize, cap: usize, nkv: usize, d: usize, like: &Tensor) -> Result<()> {
        debug_assert!(new_base >= self.base && new_base <= self.len && self.len - new_base <= cap);
        let kb = Tensor::zeros((nkv, d, cap), like.dtype(), like.device())?;
        let vb = Tensor::zeros((nkv, cap, d), like.dtype(), like.device())?;
        let keep = self.len - new_base;
        if let (Some(ok), Some(ov)) = (&self.k_buf, &self.v_buf) {
            if keep > 0 {
                let from = new_base - self.base;
                kb.slice_set(&ok.narrow(2, from, keep)?.contiguous()?, 2, 0)?;
                vb.slice_set(&ov.narrow(1, from, keep)?.contiguous()?, 1, 0)?;
            }
        }
        self.k_buf = Some(kb);
        self.v_buf = Some(vb);
        self.base = new_base;
        Ok(())
    }

    /// The first position a bounded slot keeps when it drops its oldest:
    /// the last `retain` of those it holds. A row at `len` or later sees
    /// none it drops, since `retain >= window`.
    fn kept_from(&self, b: KvBound) -> usize {
        self.base.max(self.len.saturating_sub(b.retain()))
    }

    pub(super) fn append_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (nkv, t, d) = k.dims3()?;
        if self.held() + t > self.capacity() {
            match self.bound {
                None => {
                    let need = self.len + t;
                    let cap = grown_capacity(self.capacity(), need.max(self.min_capacity));
                    self.grow(cap, nkv, d, k)?;
                }
                // Drop the oldest first: the rows of this call see no key before `len + 1 - window`.
                Some(b) => {
                    let from = self.kept_from(b);
                    let cap = (self.len - from + t).max(b.settled_capacity());
                    self.relocate(from, cap, nkv, d, k)?;
                }
            }
        }
        let at = self.held();
        let (kb, vb) = (self.k_buf.as_ref().expect("allocated above"), self.v_buf.as_ref().expect("allocated above"));
        kb.slice_set(&k.transpose(1, 2)?.contiguous()?, 2, at)?;
        vb.slice_set(&v.contiguous()?, 1, at)?;
        self.len += t;
        let held = self.held();
        Ok((kb.narrow(2, 0, held)?, vb.narrow(1, 0, held)?))
    }

    /// After a call on a bounded slot whose buffers outgrew the settled
    /// capacity (a prompt, or a piece of one): keep only the last `retain`
    /// positions, in buffers of the settled capacity. Nothing otherwise. Call
    /// it once the call's views are no longer read.
    pub fn settle_window(&mut self) -> Result<()> {
        let Some(b) = self.bound else { return Ok(()) };
        let Some(k) = self.k_buf.clone() else { return Ok(()) };
        if self.capacity() <= b.settled_capacity() {
            return Ok(());
        }
        let (nkv, d, _) = k.dims3()?;
        let from = self.kept_from(b);
        self.relocate(from, b.settled_capacity(), nkv, d, &k)
    }

    /// Write the slot's state in the format of [`super::snapshot`]: what it
    /// holds, its buffers' capacity and bound, and the four convolution
    /// states, the rings read back as [`Self::mark`] reads them. Nothing
    /// changes.
    pub fn write_state(&self, w: &mut dyn std::io::Write) -> Result<()> {
        use super::snapshot::{put_tensor_bytes, put_u64, put_u8};
        put_u64(w, self.len as u64)?;
        put_u64(w, self.base as u64)?;
        put_u64(w, self.min_capacity as u64)?;
        put_u64(w, self.capacity() as u64)?;
        match self.bound {
            None => put_u8(w, 0)?,
            Some(b) => {
                put_u8(w, 1)?;
                put_u64(w, b.window as u64)?;
                put_u64(w, b.slack as u64)?;
            }
        }
        match (&self.k_buf, &self.v_buf) {
            (Some(k), Some(v)) => {
                put_u8(w, 1)?;
                let (nkv, d, _) = k.dims3()?;
                let held = self.held();
                put_u64(w, nkv as u64)?;
                put_u64(w, d as u64)?;
                put_u64(w, held as u64)?;
                put_tensor_bytes(w, &k.narrow(2, 0, held)?)?;
                put_tensor_bytes(w, &v.narrow(1, 0, held)?)?;
            }
            _ => put_u8(w, 0)?,
        }
        for c in self.conv_states()? {
            let block = match &c {
                ConvState::Empty => None,
                ConvState::Block(t) => Some(t.clone()),
                ConvState::Ring { rows, .. } => c.block(rows.len() + 1)?,
            };
            match block {
                None => put_u8(w, 0)?,
                Some(t) => {
                    put_u8(w, 1)?;
                    let (rows, cols) = t.dims2()?;
                    put_u64(w, rows as u64)?;
                    put_u64(w, cols as u64)?;
                    put_tensor_bytes(w, &t)?;
                }
            }
        }
        Ok(())
    }

    /// Load into this slot, empty and as the model made it, a state that
    /// [`Self::write_state`] wrote: refused by name when the state's bound is
    /// not this slot's, the slot unchanged. The buffers are made on `dev`.
    pub fn read_state(&mut self, r: &mut dyn std::io::Read, dev: &Device) -> Result<()> {
        use super::snapshot::{get_padded, get_tensor_bytes, get_u8, get_usize};
        if self.len > 0 || self.k_buf.is_some() {
            candle_core::bail!("slot state: loaded into a slot that holds {} positions", self.len);
        }
        let len = get_usize(r, "len")?;
        let base = get_usize(r, "base")?;
        let min_capacity = get_usize(r, "min_capacity")?;
        let capacity = get_usize(r, "capacity")?;
        let bound = match get_u8(r)? {
            0 => None,
            1 => Some(KvBound { window: get_usize(r, "window")?, slack: get_usize(r, "slack")? }),
            o => candle_core::bail!("slot state: bound flag {o}"),
        };
        if bound != self.bound {
            candle_core::bail!("slot state: the state's bound {bound:?} is not this slot's {:?}", self.bound);
        }
        if base > len {
            candle_core::bail!("slot state: base {base} past len {len}");
        }
        let held = len - base;
        let bufs = match get_u8(r)? {
            0 => None,
            1 => {
                let nkv = get_usize(r, "kv_heads")?;
                let d = get_usize(r, "head_dim")?;
                let h = get_usize(r, "held")?;
                if h != held || capacity < held {
                    candle_core::bail!("slot state: {h} positions held of {held}, in buffers of {capacity}");
                }
                // Each buffer in one allocation, the held rows at row 0, zeros past them.
                let kb = get_padded(r, nkv * d, held, 1, capacity, dev)?.reshape((nkv, d, capacity))?;
                let vb = get_padded(r, nkv, held, d, capacity, dev)?;
                Some((kb, vb))
            }
            o => candle_core::bail!("slot state: buffer flag {o}"),
        };
        if bufs.is_none() && (held > 0 || capacity > 0) {
            candle_core::bail!("slot state: {held} positions held and no buffers");
        }
        let mut conv: [ConvState; 4] = Default::default();
        for c in conv.iter_mut() {
            *c = match get_u8(r)? {
                0 => ConvState::Empty,
                1 => {
                    let rows = get_usize(r, "conv rows")?;
                    let cols = get_usize(r, "conv cols")?;
                    ConvState::Block(get_tensor_bytes(r, &[rows, cols], dev)?)
                }
                o => candle_core::bail!("slot state: convolution flag {o}"),
            };
        }
        let [conv_k, conv_v, conv_attn, conv_mlp] = conv;
        let (k_buf, v_buf) = match bufs {
            Some((k, v)) => (Some(k), Some(v)),
            None => (None, None),
        };
        *self = SlotCache {
            k_buf,
            v_buf,
            len,
            base,
            bound,
            min_capacity,
            conv_k,
            conv_v,
            conv_attn,
            conv_mlp,
            #[cfg(ink_glue)]
            fused: None,
            #[cfg(ink_glue)]
            attn: None,
        };
        Ok(())
    }
}
