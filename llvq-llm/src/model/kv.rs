//! The KV cache of one block and its two growth stores.

use super::*;

/// One block's keys and values, for every position already seen.
///
/// Stored **before** `repeat_kv`: the grouped-query expansion is a view over
/// `n_kv` heads, so caching the expanded form would hold `n_heads / n_kv`
/// times the bytes for nothing. On Qwen3-4B that is a factor 4.
///
/// ⚠️ **No `Default`, deliberately.** Every construction site must name the
/// mode it caches in. With a `Default`, an incomplete wiring compiles and an
/// unwired arm reports Δppl = 0, ΔMMLU = 0 and 1.00× throughput — three greens
/// that mean nothing. The compiler is the cheapest place to catch that.
pub struct KvCache {
    pub(super) k: Option<Tensor>,
    v: Option<Tensor>,
    mode: KvMode,
    /// How the history grows — [`KvStore::Cat`] (shipped) or chunks grown on
    /// demand. Under `Grow`, `k`/`v` hold the FULL buffers and `len` counts the
    /// valid positions; under `Cat` they hold exactly the history and `len` is
    /// unused.
    store: crate::kvq::KvStore,
    pub(super) len: usize,
}

impl KvCache {
    /// An empty cache, in the named mode, growing by concatenation.
    pub fn new(mode: KvMode) -> Self {
        Self::with_store(mode, crate::kvq::KvStore::Cat)
    }

    /// An empty cache, naming both the payload mode and the growth store.
    pub fn with_store(mode: KvMode, store: crate::kvq::KvStore) -> Self {
        Self {
            k: None,
            v: None,
            mode,
            store,
            len: 0,
        }
    }

    /// Append this step's keys and values, and return the whole history.
    ///
    /// Concatenation, not a preallocated ring: a ring needs a maximum length
    /// decided in advance, and every length in this repository is a property
    /// of the corpus rather than of the model. The copy is `O(context)` per
    /// step against the `O(context²)` full re-run it replaces.
    ///
    /// Under [`KvMode::Q8`] the **new** keys and values are quantized on the
    /// way in, once, and the history is kept in its reconstructed form —
    /// requantizing the whole history at every step would be `O(context)` of
    /// pointless work and, worse, would make a token's bytes depend on when it
    /// was read. What this function returns is therefore exactly what it
    /// stores, which is what `repeat_kv` and the attention matmul consume:
    /// there is no separate read path that could disagree with the write path.
    pub(super) fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match self.mode {
            KvMode::F16 => (k.clone(), v.clone()),
            KvMode::Q8 => (
                crate::kvq::quantize_dequantize(k)?,
                crate::kvq::quantize_dequantize(v)?,
            ),
        };
        match self.store {
            crate::kvq::KvStore::Grow(chunk) => return self.append_grow(chunk, &k, &v),
            crate::kvq::KvStore::Cat => {}
        }
        let k = match &self.k {
            None => k,
            Some(p) => Tensor::cat(&[p, &k], 2)?.contiguous()?,
        };
        let v = match &self.v {
            None => v,
            Some(p) => Tensor::cat(&[p, &v], 2)?.contiguous()?,
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    /// The `Grow` arm of [`Self::append`]: write in place, return views of
    /// the valid prefix.
    ///
    /// The buffers hold a whole number of `chunk` positions. When the new
    /// positions do not fit, both are reallocated to the next multiple and
    /// the valid prefix is copied once; at every other step the only write is
    /// the `l` new positions. What goes out are `narrow` views, whose batch
    /// stride is the capacity: the matmuls of grouped attention read them
    /// where they are (candle passes the stride to cuBLAS, Metal and the CPU
    /// gemm). Expanded attention accepts them too, and pays the strided copy
    /// of `repeat_kv`; that pairing is correct and never served.
    fn append_grow(&mut self, chunk: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        if chunk == 0 {
            candle_core::bail!("KvCache grow: a chunk of zero positions never fits anything");
        }
        let l = k.dim(2)?;
        let need = self.len + l;
        let cap = match &self.k {
            Some(b) => b.dim(2)?,
            None => 0,
        };
        if need > cap {
            let cap = need.div_ceil(chunk) * chunk;
            self.k = Some(grown(self.k.as_ref(), k, self.len, cap)?);
            self.v = Some(grown(self.v.as_ref(), v, self.len, cap)?);
        }
        let (kb, vb) = (self.k.as_ref().unwrap(), self.v.as_ref().unwrap());
        kb.slice_set(&k.contiguous()?, 2, self.len)?;
        vb.slice_set(&v.contiguous()?, 2, self.len)?;
        self.len = need;
        Ok((kb.narrow(2, 0, need)?, vb.narrow(2, 0, need)?))
    }

    /// Keep the first `len` positions and forget the rest: the next append
    /// writes at `len`. A measurement uses it to decode several times after
    /// one prefill. Under `Cat` the kept history is copied out; `Grow` only
    /// moves its count, since the positions past it are outside the view.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        let have = match self.store {
            crate::kvq::KvStore::Cat => match &self.k {
                Some(k) => k.dim(2)?,
                None => 0,
            },
            crate::kvq::KvStore::Grow(_) => self.len,
        };
        if len > have {
            candle_core::bail!("KvCache truncate: {len} positions asked, {have} held");
        }
        match self.store {
            crate::kvq::KvStore::Cat if len == 0 => {
                self.k = None;
                self.v = None;
            }
            crate::kvq::KvStore::Cat => {
                if let (Some(k), Some(v)) = (&self.k, &self.v) {
                    let (k, v) = (k.narrow(2, 0, len)?.contiguous()?, v.narrow(2, 0, len)?.contiguous()?);
                    self.k = Some(k);
                    self.v = Some(v);
                }
            }
            crate::kvq::KvStore::Grow(_) => self.len = len,
        }
        Ok(())
    }

    /// Positions held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The `Grow` store's two buffers, `(1, n_kv, cap, head_dim)`, once a
    /// first append made them. 4E reads and writes them in place.
    pub fn grow_buffers(&self) -> Option<(&Tensor, &Tensor)> {
        match (self.store, &self.k, &self.v) {
            (crate::kvq::KvStore::Grow(_), Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Grow the `Grow` buffers to hold `need` positions, by the store's own
    /// chunk rule, keeping the positions held. 4E calls it before a launch
    /// writes positions past the capacity.
    pub fn reserve(&mut self, need: usize) -> Result<()> {
        let crate::kvq::KvStore::Grow(chunk) = self.store else {
            candle_core::bail!("KvCache reserve: only the Grow store is written in place");
        };
        let (Some(k), Some(v)) = (self.k.as_ref(), self.v.as_ref()) else {
            candle_core::bail!("KvCache reserve: no buffer before a first append");
        };
        if need > k.dim(2)? {
            let cap = need.div_ceil(chunk) * chunk;
            let (nk, nv) = (grown(Some(k), k, self.len, cap)?, grown(Some(v), v, self.len, cap)?);
            self.k = Some(nk);
            self.v = Some(nv);
        }
        Ok(())
    }

    /// After 4E wrote positions `len..n` into the buffers in place.
    pub fn set_len(&mut self, n: usize) -> Result<()> {
        let cap = self.grow_buffers().map(|(k, _)| k.dim(2)).transpose()?.unwrap_or(0);
        if n > cap {
            candle_core::bail!("KvCache set_len: {n} positions, the buffers hold {cap}");
        }
        self.len = n;
        Ok(())
    }

    /// Which store this cache grows by — the block branches on it.
    pub fn store(&self) -> crate::kvq::KvStore {
        self.store
    }

    /// The history held, `(1, n_kv, len, head_dim)` views of keys and values,
    /// for a store that keeps the grouped heads (`Cat`, `Grow`). `None` before
    /// a first append. The batched driver copies a prefilled sequence into its
    /// slot through this.
    pub fn held(&self) -> Result<Option<(Tensor, Tensor)>> {
        let (Some(k), Some(v)) = (&self.k, &self.v) else {
            return Ok(None);
        };
        match self.store {
            crate::kvq::KvStore::Cat => Ok(Some((k.clone(), v.clone()))),
            crate::kvq::KvStore::Grow(_) => Ok(Some((k.narrow(2, 0, self.len)?, v.narrow(2, 0, self.len)?))),
        }
    }
}

/// A `Grow` buffer of `cap` positions, shaped like `like` on every other
/// axis, holding the first `len` positions of `old` and zeros after them.
fn grown(old: Option<&Tensor>, like: &Tensor, len: usize, cap: usize) -> Result<Tensor> {
    let (b, heads, _, hd) = like.dims4()?;
    let fresh = Tensor::zeros((b, heads, cap, hd), like.dtype(), like.device())?;
    if let Some(old) = old.filter(|_| len > 0) {
        fresh.slice_set(&old.narrow(2, 0, len)?.contiguous()?, 2, 0)?;
    }
    Ok(fresh)
}

/// `t` with its last two axes row-major, as the CUDA and Metal matmuls require
/// of each batch element, whatever its batch stride.
///
/// A view of a `Grow` buffer is already so and passes as it is: copying it
/// would be the history copy the store exists to avoid. The first append of a
/// `Cat` cache hands back the projection output as transposed, rows of one head
/// interleaved with the others; that one is copied. The CPU gemm accepts both,
/// which is why the CPU tests did not see it and the card did (deviation É1 of
/// the GQA perplexity prereg).
pub(super) fn rows_contiguous(t: &Tensor) -> Result<Tensor> {
    let (dims, stride) = (t.dims(), t.stride());
    let n = dims.len();
    let row_major = n >= 2 && stride[n - 1] == 1 && (stride[n - 2] == dims[n - 1] || dims[n - 2] == 1);
    if row_major {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}
