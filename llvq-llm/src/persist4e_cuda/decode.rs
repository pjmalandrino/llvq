//! The linear launches of 4E: decoding N tokens a launch (`launch`, its
//! stamped and logits forms) and the verify pass of K rows. One file because
//! both settle the same flags and read the same caches in one cooperative
//! launch.

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use llvq_cuda::pers;

use crate::model::KvCache;

use super::{addr16, Counters4e, Mode4e, Runtime4e, Verify4e, MAX_CAP};

impl Runtime4e {
    /// Decode up to `n` tokens from `tok` at position `pos0`, the caches
    /// holding `pos0` positions, stopping after `eos`. Returns the tokens
    /// made; the caches hold `pos0 + made` positions after.
    pub fn decode(&mut self, caches: &mut [KvCache], pos0: usize, tok: u32, n: u32, eos: u32) -> anyhow::Result<Vec<u32>> {
        Ok(self.launch(caches, pos0, tok, n, eos, Mode4e::Plain)?.0)
    }

    /// [`Self::decode`] past one launch's `max_tokens`: launches of at most
    /// `chunk` tokens (0: `max_tokens`), each starting where the previous
    /// left off, at its last token and the position after its last cached
    /// one. A 4E launch computes every token from the cache and its own
    /// input alone, and leaves the cache exactly as a longer launch would
    /// after the same tokens (`launch` sets the length to the positions it
    /// wrote); only the ring and the barrier base, which the host carries,
    /// differ between one launch and two. Stops at `eos` or when a launch
    /// makes fewer tokens than asked.
    pub fn decode_chunks(&mut self, caches: &mut [KvCache], pos0: usize, tok: u32, n: usize, eos: u32, chunk: u32) -> anyhow::Result<Vec<u32>> {
        let step = if chunk == 0 { self.max_tokens } else { chunk.min(self.max_tokens) };
        let mut out: Vec<u32> = Vec::with_capacity(n);
        let (mut pos, mut t) = (pos0, tok);
        while out.len() < n {
            let k = (n - out.len()).min(step as usize) as u32;
            let made = self.decode(caches, pos, t, k, eos)?;
            let got = made.len();
            out.extend_from_slice(&made);
            if got < k as usize || made.contains(&eos) || got == 0 {
                break;
            }
            pos += got;
            t = *made.last().expect("got > 0");
        }
        Ok(out)
    }

    /// The form and the geometry, as the measurement prints them: "v4 at 512
    /// threads".
    pub fn label(&self) -> String {
        if let Some(arm) = self.e1_arm {
            let local = if self.local_bytes == 0 { String::new() } else { format!(", {} B local", self.local_bytes) };
            return format!("e1:{arm} at {} threads{local}", self.threads);
        }
        let form = pers::P4E_ROWS_FORMS.iter().find(|f| f.0 == self.rows_form).map_or("v?", |f| f.1);
        let pad = if self.padded { "s" } else { "" };
        format!("{form}{pad} at {} threads", self.threads)
    }

    /// The first position a launch may start at.
    pub fn kernel_from(&self) -> usize {
        self.kernel_from
    }

    /// Lower (or restore) the threshold, for a measurement only: below
    /// [`crate::kvq::KERNEL_FROM`] 4E's tokens are the kernel form's, which
    /// an eager config does not run there. The eager side of such a
    /// measurement must run the same form
    /// (`Qwen3::set_kernel_from_for_measurement`). A served config that names
    /// the persistent decode sets 0 through `Served::decoder` instead.
    pub fn set_kernel_from_for_measurement(&mut self, from: usize) {
        self.kernel_from = from;
    }

    /// The served config's threshold (`Served::decoder` only): 0 under a
    /// config that names the persistent decode, the kernel form the
    /// operator's T8 extension of 2026-11-05 licenses on the sealed 4B at
    /// every position.
    pub(crate) fn set_served_kernel_from(&mut self, from: usize) {
        self.kernel_from = from;
    }

    /// Does this runtime carry the phase-stamp symbol?
    pub fn can_stamp(&self) -> bool {
        self.f_stamp.is_some()
    }

    /// [`Self::decode`] on the phase-stamp symbol (a measurement): the same
    /// tokens, and the stamps `[made][p4e_events][grid]` of the grid clock,
    /// in ns, for `pers::p4e_phase_report`.
    pub fn decode_stamped(&mut self, caches: &mut [KvCache], pos0: usize, tok: u32, n: u32, eos: u32) -> anyhow::Result<(Vec<u32>, Vec<u64>)> {
        let (made, stamps, _) = self.launch(caches, pos0, tok, n, eos, Mode4e::Stamped)?;
        Ok((made, stamps))
    }

    /// Does this runtime carry an E1 logits symbol?
    pub fn can_logits(&self) -> bool {
        self.f_logits.is_some()
    }

    /// [`Self::decode`] on an E1 module's logits symbol (a measurement): the
    /// same tokens, and the head's binary16 logits of each token made,
    /// `[made][vocab]`, as bits.
    pub fn decode_logits(&mut self, caches: &mut [KvCache], pos0: usize, tok: u32, n: u32, eos: u32) -> anyhow::Result<(Vec<u32>, Vec<u16>)> {
        let (made, _, logits) = self.launch(caches, pos0, tok, n, eos, Mode4e::Logits)?;
        Ok((made, logits))
    }

    /// The layers and the grid a stamp buffer is laid out by.
    pub fn stamp_shape(&self) -> (u32, u32) {
        (self.n_layers, self.grid)
    }

    /// The two flags after a launch; on a fault, the counter, both flags and
    /// the barrier base back to 0, so one fault fails one request and not
    /// every request after it.
    pub(super) fn settle(&mut self) -> anyhow::Result<Option<pers::P4eFault>> {
        let mut dev = Counters4e { cuda: &self.cuda, bar: &mut self.bar, abort: &mut self.abort, nan: &mut self.nan };
        pers::p4e_settle(&mut dev, &mut self.bar_base).map_err(anyhow::Error::msg)
    }

    fn launch(&mut self, caches: &mut [KvCache], pos0: usize, tok: u32, n: u32, eos: u32, mode: Mode4e) -> anyhow::Result<(Vec<u32>, Vec<u64>, Vec<u16>)> {
        let stamped = mode == Mode4e::Stamped;
        if stamped && self.f_stamp.is_none() {
            anyhow::bail!("4E: no phase-stamp symbol at {} threads", self.threads);
        }
        if mode == Mode4e::Logits && self.f_logits.is_none() {
            anyhow::bail!("4E: no logits symbol (an E1 module at 512 threads carries one)");
        }
        if n == 0 || n > self.max_tokens {
            anyhow::bail!("4E: {n} tokens a launch, the runtime holds {}", self.max_tokens);
        }
        if caches.len() != self.layers_host.len() {
            anyhow::bail!("4E: {} caches for {} layers", caches.len(), self.layers_host.len());
        }
        if pos0 < self.kernel_from {
            anyhow::bail!(
                "4E: a launch at position {pos0}, below the kernel threshold {}: the served config runs the grouped \
                 arithmetic there (cuBLAS), which 4E does not compute; decode those positions on the eager path",
                self.kernel_from
            );
        }
        if self.k8.as_ref().is_some_and(|k| k.release) {
            self.prepare_k8_release(caches, pos0, n as usize)?;
        } else if self.k8.is_some() {
            self.prepare_k8(caches, pos0, n as usize)?;
        } else {
            self.prepare(caches, pos0, n as usize)?;
        }
        let stream = self.cuda.stream().clone();
        stream
            .memcpy_htod(&vec![u32::MAX; self.max_tokens as usize], &mut self.ring)
            .map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let n_stamps = if stamped { n as usize * pers::p4e_events(self.n_layers) as usize * self.grid as usize } else { 0 };
        let stamps: Option<CudaSlice<u64>> = if stamped {
            Some(stream.alloc_zeros::<u64>(n_stamps).map_err(|e| anyhow::anyhow!("alloc stamps: {e}"))?)
        } else {
            None
        };
        // The logits symbol reads the stamp pointer as [n][vocab] binary16.
        let vocab = self.args_host.vocab as usize;
        let logits: Option<CudaSlice<u16>> = if mode == Mode4e::Logits {
            Some(stream.alloc_zeros::<u16>(n as usize * vocab).map_err(|e| anyhow::anyhow!("alloc logits: {e}"))?)
        } else {
            None
        };
        let launch = pers::P4eLaunch {
            bar_base: self.bar_base,
            pos0: pos0 as u32,
            n_tokens: n,
            tok0: tok,
            eos,
            pad: 0,
            ring: self.ring.device_ptr(&stream).0,
            stamps: match (&stamps, &logits) {
                (Some(b), _) => b.device_ptr(&stream).0,
                (None, Some(b)) => b.device_ptr(&stream).0,
                (None, None) => 0,
            },
        };
        let cfg = LaunchConfig { grid_dim: (self.grid, 1, 1), block_dim: (self.threads, 1, 1), shared_mem_bytes: self.smem };
        let f = match mode {
            Mode4e::Plain => &self.f,
            Mode4e::Stamped => self.f_stamp.as_ref().expect("checked above"),
            Mode4e::Logits => self.f_logits.as_ref().expect("checked above"),
        };
        let mut b = stream.launch_builder(f);
        b.arg(&self.args).arg(&launch);
        // SAFETY (open): what holds: the two arguments are the token symbol's
        // (`const Pers4eArgs*`, `Pers4eLaunch` by value, `pers_4e.cu`;
        // `pers::P4eLaunch` mirrors the struct field for field); the grid is
        // one CTA a SM, past the occupancy check at install, which a
        // cooperative launch needs; `ring`, `stamps` and `logits` are live
        // through the launch and freed in stream order on this stream, which
        // is candle's (`Cuda::on_stream`); the KV cache addresses in the
        // images are re-read from `caches` by `prepare` above and re-uploaded
        // when a buffer moved. What does not: the images in `self.layers`
        // hold raw device addresses of the model's weights, taken from
        // `&Qwen3` at install, and `Runtime4e` keeps no borrow or `Arc` of
        // them (BUG-15). Safe code that drops the model and keeps this
        // runtime launches on freed memory. The served path keeps both in
        // one engine for the life of the server.
        unsafe { b.launch_cooperative(cfg) }.map_err(|e| anyhow::anyhow!("4E: {e}"))?;
        let ring = self.cuda.down_u32(&self.ring).map_err(anyhow::Error::msg)?;
        match self.settle()? {
            Some(pers::P4eFault::Abort) => {
                anyhow::bail!("4E: a barrier aborted on its watchdog; the counters are reset for the next launch")
            }
            Some(pers::P4eFault::Nan) => anyhow::bail!("4E: a NaN logit; the eager path is the one to read"),
            None => {}
        }
        let (made, bar_base) = pers::p4e_after_launch(&ring, n, self.bar_base, self.n_layers, self.grid);
        self.bar_base = bar_base;
        match self.k8.as_mut() {
            // v9k8 wrote its positions into the int8 caches only: the binary16
            // caches keep the length the eager path left
            Some(k8) => k8.state.own = pos0 + made.len(),
            None => {
                for c in caches.iter_mut() {
                    c.set_len(pos0 + made.len())?;
                }
            }
        }
        let stamps = match stamps {
            Some(d) => {
                let all = stream.clone_dtoh(&d).map_err(|e| anyhow::anyhow!("D2H stamps: {e}"))?;
                all[..made.len() * pers::p4e_events(self.n_layers) as usize * self.grid as usize].to_vec()
            }
            None => Vec::new(),
        };
        let logits = match logits {
            Some(d) => {
                let all = stream.clone_dtoh(&d).map_err(|e| anyhow::anyhow!("D2H logits: {e}"))?;
                all[..made.len() * vocab].to_vec()
            }
            None => Vec::new(),
        };
        Ok((made, stamps, logits))
    }

    /// Install the verify pass ([`Self::verify`]): its module (row form 7
    /// and `pers_4ev.cu`, compiled beside the served module, whose text does
    /// not move), its block size (768 where the symbol does not spill, else
    /// 512), the largest staging area the card's shared memory allows
    /// (`pers::p4v_act_floats`), and buffers for `pers::P4V_KMAX` rows.
    /// Refused below row form 7: the pass reads the int4 rows tiled.
    pub fn install_verify(&mut self) -> anyhow::Result<()> {
        if self.rows_form < 7 {
            anyhow::bail!("4E verify: the pass reads the int4 rows on tiled words (row form 7, v9); this runtime is form {}", self.rows_form);
        }
        if self.verify.is_some() {
            return Ok(());
        }
        // The pass stages a rotation's k rows of `small` in P4V_SSM floats.
        let kmax = pers::P4V_SSM / llvq_cuda::pers::ROT_KMAX;
        if let Some((l, k)) = self
            .layers_host
            .iter()
            .enumerate()
            .flat_map(|(l, y)| [(l, y.k_a), (l, y.mlp.k_h), (l, y.mlp.k_i)])
            .find(|&(_, k)| k > kmax)
        {
            anyhow::bail!("4E verify: layer {l} has a rotation of odd factor {k}, the pass stages {kmax} rows of `small`");
        }
        let stream = self.cuda.stream().clone();
        let text = pers::p4v_unit_text(&self.tile_define);
        let src = llvq_cuda::gpu::KernelSource::new(&[text.as_str()]);
        self.lines.push(format!("4E verify: module {} bytes, sha256 {}, row form 7 and pers_4ev.cu, its own module", src.text.len(), src.sha256));
        let cuda = llvq_cuda::gpu::Cuda::on_stream(stream.clone(), &src).map_err(anyhow::Error::msg)?;
        let dev = cuda.device().map_err(anyhow::Error::msg)?;
        let (hidden, n_heads, n_kv, groups) = (self.args_host.hidden, self.args_host.n_heads, self.args_host.n_kv, self.args_host.groups);
        let d_ctx = n_heads * 128;
        let act_floats = pers::p4v_act_floats(hidden, self.inter, d_ctx, groups, dev.shared_per_block_optin as u32).map_err(anyhow::Error::msg)?;
        let smem = pers::SMEM_HEADER + pers::TABLE_BYTES + 4 * act_floats;
        let k = pers::P4V_KMAX;
        // the kernel's group size: the rows after the staged `small` and the head's maxima
        let group = |w: u32| ((act_floats - pers::P4V_SSM - 2 * k) / w).clamp(1, k);
        self.lines.push(format!(
            "4E verify: {act_floats} floats of staging, {smem} B shared; rows a group at most {k}: {} at hidden {hidden}, {} at the context {d_ctx}, {} at inter {}",
            group(hidden),
            group(d_ctx),
            group(self.inter),
            self.inter
        ));
        let mut chosen = None;
        let mut why = Vec::new();
        for threads in [768u32, 512] {
            let sym = pers::p4v_symbol(self.form, threads).map_err(anyhow::Error::msg)?;
            let f = cuda.func_dynamic_shared(sym, smem).map_err(anyhow::Error::msg)?;
            let r = cuda.report(sym).map_err(anyhow::Error::msg)?;
            let occ = f
                .occupancy_max_active_blocks_per_multiprocessor(threads, smem as usize, None)
                .map_err(|e| anyhow::anyhow!("occupancy: {e}"))?;
            let line = format!("4E verify: {sym}: {} registers, {} B local, {smem} B shared, {occ} resident a SM", r.num_regs, r.local_bytes);
            self.lines.push(line.clone());
            why.push(line);
            if occ >= 1 && (r.local_bytes == 0 || threads == 512) {
                chosen = Some((f, threads));
                break;
            }
        }
        let (f, threads) = chosen.ok_or_else(|| anyhow::anyhow!("4E verify: no block size fits one CTA a SM: {}", why.join("; ")))?;
        let e = |r: Result<CudaSlice<u16>, String>| r.map_err(anyhow::Error::msg);
        let (h, dq, dkv, i) = ((k * hidden) as usize, (k * d_ctx) as usize, (k * n_kv * 128) as usize, (k * self.inter) as usize);
        let held = vec![e(cuda.zeros_u16(h))?, e(cuda.zeros_u16(h))?, e(cuda.zeros_u16(dq))?, e(cuda.zeros_u16(dkv))?, e(cuda.zeros_u16(dkv))?, e(cuda.zeros_u16(dq))?, e(cuda.zeros_u16(dq))?, e(cuda.zeros_u16(i))?];
        let mut held16_addr = [0u64; 8];
        for (a, b) in held16_addr.iter_mut().zip(&held) {
            *a = addr16(b, &stream);
        }
        let part_stride = n_kv * (MAX_CAP as u32).div_ceil(256) * groups * 130;
        let part = cuda.zeros_f32((k * part_stride) as usize).map_err(anyhow::Error::msg)?;
        let tickets = cuda.up_u32(&vec![0u32; (k * n_kv) as usize]).map_err(anyhow::Error::msg)?;
        let slots = cuda.up_u64(&vec![0u64; k as usize]).map_err(anyhow::Error::msg)?;
        let toks = cuda.up_u32(&vec![0u32; k as usize]).map_err(anyhow::Error::msg)?;
        let ring = cuda.up_u32(&vec![u32::MAX; k as usize]).map_err(anyhow::Error::msg)?;
        let bytes = 2 * (2 * h + 3 * dq + 2 * dkv + i) + 4 * (k * part_stride) as usize;
        self.lines.push(format!("4E verify: {threads} threads; {:.3} GB of row buffers on the card", bytes as f64 / 1e9));
        self.verify = Some(Verify4e { _cuda: cuda, f, threads, smem, act_floats, part_stride, toks, ring, _held16: held, held16_addr, part, tickets, slots });
        Ok(())
    }

    /// The verify pass's block size, once installed.
    pub fn verify_threads(&self) -> Option<u32> {
        self.verify.as_ref().map(|v| v.threads)
    }

    /// One verify launch: `toks.len()` rows (1 to `pers::P4V_KMAX`), row r
    /// the token `toks[r]` at position `pos0 + r`, the caches holding `pos0`
    /// positions. Returns each row's argmax, the target's token after it; the
    /// caches hold `pos0 + toks.len()` after (rewind them to what is kept).
    /// Each row is the 4E token at its position bit for bit
    /// (`tests/pers_matches_served/verify.rs::the_4e_verify_rows_are_the_served_chain`).
    pub fn verify(&mut self, caches: &mut [KvCache], pos0: usize, toks: &[u32]) -> anyhow::Result<Vec<u32>> {
        Ok(self.verify_inner(caches, pos0, toks, false)?.0)
    }

    /// [`Self::verify`] with its phases stamped: the same rows and tokens,
    /// and `[1 + n_bar_4e][grid]` clocks (`pers::p4v_phases` reads them).
    /// A measurement launch.
    pub fn verify_stamped(&mut self, caches: &mut [KvCache], pos0: usize, toks: &[u32]) -> anyhow::Result<(Vec<u32>, Vec<u64>)> {
        self.verify_inner(caches, pos0, toks, true)
    }

    /// The layers and the grid a verify stamp buffer is laid out by.
    pub fn verify_stamp_shape(&self) -> (u32, u32) {
        (self.n_layers, self.grid)
    }

    fn verify_inner(&mut self, caches: &mut [KvCache], pos0: usize, toks: &[u32], stamped: bool) -> anyhow::Result<(Vec<u32>, Vec<u64>)> {
        let n = toks.len();
        if n == 0 || n > pers::P4V_KMAX as usize {
            anyhow::bail!("4E verify: {n} rows, the pass carries 1 to {}", pers::P4V_KMAX);
        }
        if self.verify.is_none() {
            anyhow::bail!("4E verify: not installed (Runtime4e::install_verify)");
        }
        if caches.len() != self.layers_host.len() {
            anyhow::bail!("4E: {} caches for {} layers", caches.len(), self.layers_host.len());
        }
        if pos0 < self.kernel_from {
            anyhow::bail!("4E verify: a launch at position {pos0}, below the kernel threshold {}", self.kernel_from);
        }
        self.prepare(caches, pos0, n)?;
        let stream = self.cuda.stream().clone();
        let v = self.verify.as_mut().expect("checked above");
        stream.memcpy_htod(toks, &mut v.toks).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        stream.memcpy_htod(&vec![u32::MAX; pers::P4V_KMAX as usize], &mut v.ring).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let a = v.held16_addr;
        let n_stamps = if stamped { (1 + pers::n_bar_4e(self.n_layers)) as usize * self.grid as usize } else { 0 };
        let stamps: Option<CudaSlice<u64>> = if stamped {
            Some(stream.alloc_zeros::<u64>(n_stamps).map_err(|e| anyhow::anyhow!("alloc stamps: {e}"))?)
        } else {
            None
        };
        let launch = pers::P4vLaunch {
            bar_base: self.bar_base,
            pos0: pos0 as u32,
            n_rows: n as u32,
            act_floats: v.act_floats,
            toks: v.toks.device_ptr(&stream).0,
            ring: v.ring.device_ptr(&stream).0,
            x_a: a[0],
            x_b: a[1],
            q_raw: a[2],
            k_raw: a[3],
            v_raw: a[4],
            q_out: a[5],
            ctx: a[6],
            act: a[7],
            part: v.part.device_ptr(&stream).0,
            tickets: v.tickets.device_ptr(&stream).0,
            slots: v.slots.device_ptr(&stream).0,
            part_stride: v.part_stride,
            pad: 0,
            stamps: stamps.as_ref().map_or(0, |b| b.device_ptr(&stream).0),
        };
        let cfg = LaunchConfig { grid_dim: (self.grid, 1, 1), block_dim: (v.threads, 1, 1), shared_mem_bytes: v.smem };
        let mut b = stream.launch_builder(&v.f);
        b.arg(&self.args).arg(&launch);
        // SAFETY (open): as in `launch`, for the verify symbol
        // (`const Pers4eArgs*`, `Pers4vLaunch` by value): the scratch and
        // ring of `self.verify` are owned by this runtime and live through
        // the launch, `n` is at most `P4V_KMAX` (checked above), the caches
        // are re-read by `prepare`; the weight addresses in the images carry
        // no lifetime tie to the model (BUG-15).
        unsafe { b.launch_cooperative(cfg) }.map_err(|e| anyhow::anyhow!("4E verify: {e}"))?;
        let ring = stream.clone_dtoh(&v.ring).map_err(|e| anyhow::anyhow!("D2H: {e}"))?;
        match self.settle()? {
            Some(pers::P4eFault::Abort) => {
                anyhow::bail!("4E verify: a barrier aborted on its watchdog; the counters are reset for the next launch")
            }
            Some(pers::P4eFault::Nan) => anyhow::bail!("4E verify: a NaN logit; the eager path is the one to read"),
            None => {}
        }
        let made: Vec<u32> = ring[..n].to_vec();
        if made.contains(&u32::MAX) {
            // The barriers this launch passed are unknown: start over.
            let mut dev = Counters4e { cuda: &self.cuda, bar: &mut self.bar, abort: &mut self.abort, nan: &mut self.nan };
            pers::p4e_reset(&mut dev, &mut self.bar_base).map_err(anyhow::Error::msg)?;
            anyhow::bail!("4E verify: a row wrote no token ({made:?}); the counters are reset for the next launch");
        }
        self.bar_base = self.bar_base.wrapping_add(pers::n_bar_4e(self.n_layers).wrapping_mul(self.grid));
        for c in caches.iter_mut() {
            c.set_len(pos0 + n)?;
        }
        let stamps = match stamps {
            Some(d) => stream.clone_dtoh(&d).map_err(|e| anyhow::anyhow!("D2H stamps: {e}"))?,
            None => Vec::new(),
        };
        Ok((made, stamps))
    }
}
