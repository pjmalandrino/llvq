//! The caches before a launch: grown to hold it, their addresses checked and
//! the images re-uploaded when a buffer moved, and arm v9k8's int8 caches
//! filled or released. One file because each step runs before a launch.

use candle_core::cuda_backend::cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
use llvq_cuda::pers;

use crate::model::KvCache;

use super::views::tensor_addr;
use super::{Kv8, Runtime4e, MAX_CAP};

impl Runtime4e {
    /// The caches grown to hold a launch of `n` positions from `pos0`, and
    /// the images re-uploaded when a buffer moved.
    pub(super) fn prepare(&mut self, caches: &mut [KvCache], pos0: usize, n: usize) -> anyhow::Result<()> {
        let stream = self.cuda.stream().clone();
        // Refused before any `reserve`: a buffer grown past the scratch keeps
        // its capacity, and every later launch on that cache would be refused.
        if pos0 + n > MAX_CAP {
            anyhow::bail!("4E: positions up to {}, the attention scratch holds {MAX_CAP}", pos0 + n);
        }
        // Grow every cache to hold the launch; the images carry the
        // addresses, re-uploaded only when a buffer moved.
        let mut seen = Vec::with_capacity(caches.len());
        for c in caches.iter_mut() {
            if c.len() != pos0 {
                anyhow::bail!("4E: a cache holds {} positions, the launch starts at {pos0}", c.len());
            }
            c.reserve(pos0 + n)?;
            let (k, v) = c.grow_buffers().ok_or_else(|| anyhow::anyhow!("4E: a cache with no Grow buffer"))?;
            let cap = k.dim(2)?;
            if cap > MAX_CAP {
                anyhow::bail!("4E: a cache of {cap} positions, the attention scratch holds {MAX_CAP}");
            }
            seen.push((cap, tensor_addr(k, &stream)?, tensor_addr(v, &stream)?));
        }
        if seen != self.kv_seen {
            let caps: std::collections::BTreeSet<usize> = seen.iter().map(|s| s.0).collect();
            if caps.len() != 1 {
                anyhow::bail!("4E: the layers' caches have different capacities {caps:?}");
            }
            let mut img = Vec::with_capacity(pers::P4E_LAYER_BYTES * seen.len());
            for (h, &(_, k, v)) in self.layers_host.iter_mut().zip(&seen) {
                h.kc = k;
                h.vc = v;
                img.extend(h.bytes());
            }
            stream.memcpy_htod(&img, &mut self.layers).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
            self.args_host.ptrs[0] = self.layers.device_ptr(&stream).0;
            self.args_host.cap = seen[0].0 as u32;
            stream.memcpy_htod(&self.args_host.bytes(), &mut self.args).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
            self.kv_seen = seen;
        }
        Ok(())
    }

    /// Arm v9k8 before a launch of `n` positions from `pos0`
    /// (`pers::p4e_k8_step`): grow the int8 caches (copying each head's blocks
    /// when they held something), quantize the positions the launch takes over
    /// from the binary16 caches (`pers_kv8_fill`, one launch a layer), and
    /// re-upload the images, whose kc and vc name the int8 caches and whose
    /// cap is theirs, when a buffer moved. The binary16 caches are read, never
    /// grown or written.
    pub(super) fn prepare_k8(&mut self, caches: &[KvCache], pos0: usize, n: usize) -> anyhow::Result<()> {
        let mut k8 = self.k8.take().expect("a v9k8 runtime");
        let r = self.prepare_k8_with(&mut k8, caches, pos0, n);
        self.k8 = Some(k8);
        r
    }

    /// [`Self::prepare_k8`] with the int8 state taken out of `self`.
    fn prepare_k8_with(&mut self, k8: &mut Kv8, caches: &[KvCache], pos0: usize, n: usize) -> anyhow::Result<()> {
        let stream = self.cuda.stream().clone();
        {
            let mut f16 = Vec::with_capacity(caches.len());
            let mut len16 = None;
            let mut cap16 = None;
            for c in caches.iter() {
                let (k, v) = c.grow_buffers().ok_or_else(|| anyhow::anyhow!("4E v9k8: a cache with no Grow buffer"))?;
                if *len16.get_or_insert(c.len()) != c.len() || *cap16.get_or_insert(k.dim(2)?) != k.dim(2)? {
                    anyhow::bail!("4E v9k8: the layers' binary16 caches differ in length or capacity");
                }
                f16.push((tensor_addr(k, &stream)?, tensor_addr(v, &stream)?));
            }
            let (len16, cap16) = (len16.unwrap_or(0), cap16.unwrap_or(0));
            let moved = f16 != k8.f16_seen;
            let step = pers::p4e_k8_step(k8.state, moved, len16, pos0, n, MAX_CAP).map_err(anyhow::Error::msg)?;
            let blk = pers::P4E_K8_BLK as usize;
            if step.grow {
                let (old, new) = (k8.state.cap, step.next.cap);
                let bytes = llvq_cuda::kv8_ref::cache_bytes(k8.n_kv as usize, new);
                let mut bufs = Vec::with_capacity(caches.len());
                for l in 0..caches.len() {
                    let mut kb = stream.alloc_zeros::<u8>(bytes).map_err(|e| anyhow::anyhow!("alloc int8 K: {e}"))?;
                    let mut vb = stream.alloc_zeros::<u8>(bytes).map_err(|e| anyhow::anyhow!("alloc int8 V: {e}"))?;
                    if step.copy_old {
                        let (ok, ov) = &k8.bufs[l];
                        let (ho, hn) = (old / 8 * blk, new / 8 * blk);
                        for h in 0..k8.n_kv as usize {
                            stream
                                .memcpy_dtod(&ok.slice(h * ho..(h + 1) * ho), &mut kb.slice_mut(h * hn..h * hn + ho))
                                .map_err(|e| anyhow::anyhow!("D2D int8 K: {e}"))?;
                            stream
                                .memcpy_dtod(&ov.slice(h * ho..(h + 1) * ho), &mut vb.slice_mut(h * hn..h * hn + ho))
                                .map_err(|e| anyhow::anyhow!("D2D int8 V: {e}"))?;
                        }
                    }
                    bufs.push((kb, vb));
                }
                k8.bufs = bufs;
            }
            if let Some((a, b)) = step.fill {
                let (cap8, n_kv) = (step.next.cap as u32, k8.n_kv);
                let (lo, hi, c16) = (a as u32, b as u32, cap16 as u32);
                let warps = (hi - lo) * n_kv * 2;
                let cfg = LaunchConfig { grid_dim: (warps.div_ceil(8).clamp(1, 4096), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                for (l, &(k16, v16)) in f16.iter().enumerate() {
                    let (kb, vb) = &mut k8.bufs[l];
                    let mut b = stream.launch_builder(&k8.fill);
                    b.arg(&k16).arg(&v16).arg(kb).arg(vb).arg(&n_kv).arg(&c16).arg(&cap8).arg(&lo).arg(&hi);
                    // SAFETY: pers_kv8_fill's own signature; k16 and v16 are live binary16
                    // caches of n_kv x cap16 x 128 holding positions [lo, hi), the int8
                    // buffers hold n_kv x cap8 / 8 blocks, hi <= cap8 (pers::p4e_k8_step).
                    unsafe { b.launch(cfg) }.map_err(|e| anyhow::anyhow!("pers_kv8_fill: {e}"))?;
                }
            }
            k8.state = step.next;
            k8.f16_seen = f16;
            let seen: Vec<(usize, u64, u64)> = k8
                .bufs
                .iter()
                .map(|(kb, vb)| (step.next.cap, kb.device_ptr(&stream).0, vb.device_ptr(&stream).0))
                .collect();
            if seen != self.kv_seen {
                let mut img = Vec::with_capacity(pers::P4E_LAYER_BYTES * seen.len());
                for (h, &(_, k, v)) in self.layers_host.iter_mut().zip(&seen) {
                    h.kc = k;
                    h.vc = v;
                    img.extend(h.bytes());
                }
                stream.memcpy_htod(&img, &mut self.layers).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
                self.args_host.ptrs[0] = self.layers.device_ptr(&stream).0;
                self.args_host.cap = step.next.cap as u32;
                stream.memcpy_htod(&self.args_host.bytes(), &mut self.args).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
                self.kv_seen = seen;
            }
            Ok(())
        }
    }

    /// Release the binary16 caches at each hand-over (`true`), so the int8
    /// caches are the only copy the decode holds: the NOT SERVED
    /// `decode_cache: int8` form. Refused on a runtime with no int8 caches.
    pub fn set_k8_release(&mut self, on: bool) -> anyhow::Result<()> {
        let k8 = self.k8.as_mut().ok_or_else(|| anyhow::anyhow!("4E: no int8 caches to release the binary16 ones for (arm v9k8 only)"))?;
        k8.release = on;
        Ok(())
    }

    /// [`Self::prepare_k8`] with the binary16 caches released at the
    /// hand-over (`pers::p4e_k8_step_release`). Layer by layer: the int8
    /// buffers made (copying what the arm wrote when they grow), the layer's
    /// positions quantized from its binary16 cache, and that cache released
    /// before the next layer, so the two copies never coexist whole. A
    /// continuation reads no binary16 cache.
    pub(super) fn prepare_k8_release(&mut self, caches: &mut [KvCache], pos0: usize, n: usize) -> anyhow::Result<()> {
        let mut k8 = self.k8.take().expect("a v9k8 runtime");
        let r = self.prepare_k8_release_with(&mut k8, caches, pos0, n);
        self.k8 = Some(k8);
        r
    }

    fn prepare_k8_release_with(&mut self, k8: &mut Kv8, caches: &mut [KvCache], pos0: usize, n: usize) -> anyhow::Result<()> {
        let stream = self.cuda.stream().clone();
        let held = caches.iter().filter(|c| c.grow_buffers().is_some()).count();
        if held != 0 && held != caches.len() {
            anyhow::bail!("4E v9k8: {held} of {} layers hold a binary16 cache", caches.len());
        }
        let has_f16 = held == caches.len();
        let (mut len16, mut cap16) = (0usize, 0usize);
        if has_f16 {
            for (i, c) in caches.iter().enumerate() {
                let cap = c.grow_buffers().expect("counted above").0.dim(2)?;
                if i == 0 {
                    (len16, cap16) = (c.len(), cap);
                } else if (c.len(), cap) != (len16, cap16) {
                    anyhow::bail!("4E v9k8: the layers' binary16 caches differ in length or capacity");
                }
            }
        }
        let step = pers::p4e_k8_step_release(k8.state, has_f16, len16, pos0, n, MAX_CAP).map_err(anyhow::Error::msg)?;
        let blk = pers::P4E_K8_BLK as usize;
        let (old, new) = (k8.state.cap, step.next.cap);
        let bytes = llvq_cuda::kv8_ref::cache_bytes(k8.n_kv as usize, new);
        let fresh = k8.bufs.is_empty();
        for (l, cache) in caches.iter_mut().enumerate() {
            if step.grow || fresh {
                let mut kb = stream.alloc_zeros::<u8>(bytes).map_err(|e| anyhow::anyhow!("alloc int8 K: {e}"))?;
                let mut vb = stream.alloc_zeros::<u8>(bytes).map_err(|e| anyhow::anyhow!("alloc int8 V: {e}"))?;
                if step.copy_old && !fresh {
                    let (ok, ov) = &k8.bufs[l];
                    let (ho, hn) = (old / 8 * blk, new / 8 * blk);
                    for h in 0..k8.n_kv as usize {
                        stream
                            .memcpy_dtod(&ok.slice(h * ho..(h + 1) * ho), &mut kb.slice_mut(h * hn..h * hn + ho))
                            .map_err(|e| anyhow::anyhow!("D2D int8 K: {e}"))?;
                        stream
                            .memcpy_dtod(&ov.slice(h * ho..(h + 1) * ho), &mut vb.slice_mut(h * hn..h * hn + ho))
                            .map_err(|e| anyhow::anyhow!("D2D int8 V: {e}"))?;
                    }
                }
                match fresh {
                    true => k8.bufs.push((kb, vb)),
                    false => k8.bufs[l] = (kb, vb),
                }
            }
            if let Some((a, b)) = step.fill {
                let (k16, v16) = {
                    let (k, v) = cache.grow_buffers().expect("a hand-over holds every binary16 cache");
                    (tensor_addr(k, &stream)?, tensor_addr(v, &stream)?)
                };
                let (cap8, n_kv) = (new as u32, k8.n_kv);
                let (lo, hi, c16) = (a as u32, b as u32, cap16 as u32);
                let warps = (hi - lo) * n_kv * 2;
                let cfg = LaunchConfig { grid_dim: (warps.div_ceil(8).clamp(1, 4096), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                let (kb, vb) = &mut k8.bufs[l];
                let mut lb = stream.launch_builder(&k8.fill);
                lb.arg(&k16).arg(&v16).arg(kb).arg(vb).arg(&n_kv).arg(&c16).arg(&cap8).arg(&lo).arg(&hi);
                // SAFETY: pers_kv8_fill's own signature; k16 and v16 are this layer's live binary16
                // cache of n_kv x cap16 x 128 holding positions [lo, hi), the int8 buffers hold
                // n_kv x cap8 / 8 blocks, hi <= cap8 (pers::p4e_k8_step_release).
                unsafe { lb.launch(cfg) }.map_err(|e| anyhow::anyhow!("pers_kv8_fill: {e}"))?;
                // the fill is ordered before the free on this stream; the layer's binary16 copy goes now
                cache.release();
            }
        }
        k8.state = step.next;
        k8.f16_seen = Vec::new();
        let seen: Vec<(usize, u64, u64)> =
            k8.bufs.iter().map(|(kb, vb)| (new, kb.device_ptr(&stream).0, vb.device_ptr(&stream).0)).collect();
        if seen != self.kv_seen {
            let mut img = Vec::with_capacity(pers::P4E_LAYER_BYTES * seen.len());
            for (h, &(_, k, v)) in self.layers_host.iter_mut().zip(&seen) {
                h.kc = k;
                h.vc = v;
                img.extend(h.bytes());
            }
            stream.memcpy_htod(&img, &mut self.layers).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
            self.args_host.ptrs[0] = self.layers.device_ptr(&stream).0;
            self.args_host.cap = new as u32;
            stream.memcpy_htod(&self.args_host.bytes(), &mut self.args).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
            self.kv_seen = seen;
        }
        Ok(())
    }

    /// Arm v9k8's int8 caches as they stand: capacity, bytes on the card,
    /// the positions quantized from the binary16 caches; none for another
    /// runtime.
    pub fn kv8_report(&self) -> Option<String> {
        let k8 = self.k8.as_ref()?;
        let bytes = 2 * k8.bufs.len() * llvq_cuda::kv8_ref::cache_bytes(k8.n_kv as usize, k8.state.cap);
        Some(format!(
            "4E v9k8: int8 KV caches at a capacity of {} positions, {:.1} MB on the card ({} layers), positions [0, {}) \
             quantized from the prefill's binary16 caches, {}",
            k8.state.cap,
            bytes as f64 / 1e6,
            k8.bufs.len(),
            k8.state.hand,
            if k8.release {
                "each released after its layer's fill: the int8 caches are the only copy"
            } else {
                "which stay allocated beside them for the eager path"
            }
        ))
    }

    /// Synchronise the stream (timing).
    pub fn sync(&self) -> anyhow::Result<()> {
        self.cuda.sync().map_err(anyhow::Error::msg)
    }
}
