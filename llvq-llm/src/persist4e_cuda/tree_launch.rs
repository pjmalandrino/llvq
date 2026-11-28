//! The tree pass launched: one launch of a tree of rows, its moved and chain
//! forms, and the chain prefill that sends a prompt through it. One file
//! because every entry point ends in `tree_launch`.

use super::*;

impl Runtime4e {
    /// One tree launch: row r the token `toks[r]` at position `pos0 +
    /// depth[r]`, `path[r][d]` its ancestor's row at depth d and
    /// `path[r][depth[r]] = r`, the caches holding `pos0` positions, which the
    /// launch reads and does not write (no move: the accepted rows stay in the
    /// tree buffer). Each row's argmax, its logits when `logits`, the phase
    /// clocks when `stamped` (`pers::p4t_phases`).
    #[allow(clippy::too_many_arguments)]
    pub fn tree(&mut self, arm: usize, caches: &mut [KvCache], pos0: usize, toks: &[u32], depth: &[u32], path: &[[u8; 32]], logits: bool, stamped: bool) -> anyhow::Result<TreeOut> {
        self.tree_moved(arm, caches, pos0, toks, depth, path, &[], 0, logits, stamped, false)
    }

    /// [`Self::tree`] with the previous launch's rows `mv` (its slots, root
    /// first) moved into the cache at positions `move_pos0`, `move_pos0 + 1`,
    /// ... in this launch's first phase: the caches hold `move_pos0`
    /// positions on the call and `pos0 = move_pos0 + mv.len()` after it (no
    /// move: `pos0` on both). `taps`: an a12 arm writes and returns each
    /// row's taps (`TreeOut::taps`). The speculative loop's verify step
    /// (`spec_loop::Verifier`).
    #[allow(clippy::too_many_arguments)]
    pub fn tree_moved(
        &mut self,
        arm: usize,
        caches: &mut [KvCache],
        pos0: usize,
        toks: &[u32],
        depth: &[u32],
        path: &[[u8; 32]],
        mv: &[u32],
        move_pos0: usize,
        logits: bool,
        stamped: bool,
        taps: bool,
    ) -> anyhow::Result<TreeOut> {
        self.tree_launch(arm, caches, pos0, toks, depth, path, mv, move_pos0, logits, stamped, taps, true)
    }

    /// One launch of the chain prefill (`llvq_cuda::spec_loop::chain_launches`,
    /// docs/mesures/large-prefill-contexte-2026-11-11.txt): the known tokens
    /// `toks` as a chain at `pos0` (row r at depth r, its path 0..=r), the
    /// previous launch's rows `mv` moved in first, as [`Self::tree_moved`].
    /// `head` false skips the final norm and the head (the launch's
    /// `split_head` is 0, an a12 or a12ip arm only): the ring is then
    /// meaningless and returned empty, and `logits` is refused. Taps off.
    #[allow(clippy::too_many_arguments)]
    pub fn tree_chain(
        &mut self,
        arm: usize,
        caches: &mut [KvCache],
        pos0: usize,
        toks: &[u32],
        mv: &[u32],
        move_pos0: usize,
        head: bool,
        logits: bool,
        stamped: bool,
    ) -> anyhow::Result<TreeOut> {
        let n = toks.len();
        if n == 0 || n > pers::P4T_RMAX as usize {
            anyhow::bail!("4E tree chain: {n} rows, the pass carries 1 to {}", pers::P4T_RMAX);
        }
        let depth: Vec<u32> = (0..n as u32).collect();
        let path: Vec<[u8; 32]> = (0..n).map(llvq_cuda::spec_loop::chain_path).collect();
        self.tree_launch(arm, caches, pos0, toks, &depth, &path, mv, move_pos0, logits, stamped, false, head)
    }

    /// How [`Self::install_tree`] loads the tree modules: set before it.
    /// [`TreeModules::Cubin`] with a directory reads each module from a file
    /// of its exact key ([`llvq_cuda::modcache`]) or compiles and writes it.
    pub fn set_tree_modules(&mut self, how: TreeModules) {
        self.tree_modules = how;
    }

    /// The chain prefill of `ids` after the `pos0` positions the caches hold
    /// (their Grow buffers made, no move pending), through tree arm `arm` at
    /// `r` rows a launch (`llvq_cuda::spec_loop::chain_launches`): the head on
    /// the last launch only, then the one-row hand-over launch that moves the
    /// last chunk into the caches. The caches then hold `pos0 + ids.len()`
    /// positions and 4E decodes `first` at that position. The launch loop of
    /// `bin/prefillbench --form chain`, which measured it.
    pub fn chain_prefill(&mut self, arm: usize, caches: &mut [KvCache], pos0: usize, ids: &[u32], r: usize) -> anyhow::Result<ChainPrefill> {
        use llvq_cuda::spec_loop::{chain_handover, chain_launches, ChainFault};
        let t0 = std::time::Instant::now();
        let (ls, mv, mp) = chain_launches(ids, pos0, r, ChainFault::None).map_err(anyhow::Error::msg)?;
        let n = ls.len();
        let mut first = None;
        for (i, l) in ls.iter().enumerate() {
            let head = i + 1 == n;
            let o = self.tree_chain(arm, caches, l.pos0, &l.toks, &l.mv, l.move_pos0, head, false, false)?;
            if head {
                first = o.ring.last().copied();
            }
        }
        let first = first.ok_or_else(|| anyhow::anyhow!("4E chain prefill: the last launch wrote no token"))?;
        let ttft = t0.elapsed().as_secs_f64();
        let h = chain_handover(first, &mv, mp);
        self.tree_chain(arm, caches, h.pos0, &h.toks, &h.mv, h.move_pos0, false, false, false)?;
        self.sync()?;
        Ok(ChainPrefill { first, ttft, prefill: t0.elapsed().as_secs_f64() })
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_launch(
        &mut self,
        arm: usize,
        caches: &mut [KvCache],
        pos0: usize,
        toks: &[u32],
        depth: &[u32],
        path: &[[u8; 32]],
        mv: &[u32],
        move_pos0: usize,
        logits: bool,
        stamped: bool,
        taps: bool,
        head: bool,
    ) -> anyhow::Result<TreeOut> {
        let n = toks.len();
        if !head {
            if logits {
                anyhow::bail!("4E tree: logits asked of a launch without the head");
            }
            let form = self.tree.as_ref().and_then(|t| t.arms.get(arm)).map(|a| a.form);
            if !matches!(form.map(TreeForm::parent), Some(TreeForm::A12 | TreeForm::A12ip | TreeForm::A12ipg | TreeForm::A12ipg32)) {
                anyhow::bail!("4E tree: a launch without the head runs on an a12, a12ip, a12ipg or a12ipg32 arm only, arm {arm} is {form:?}");
            }
        }
        if !mv.is_empty() {
            if mv.len() > pers::P4T_RMAX as usize || move_pos0 + mv.len() != pos0 || mv.iter().any(|&s| s >= pers::P4T_RMAX) {
                anyhow::bail!("4E tree: a move of {} rows {mv:?} from position {move_pos0} to a root at {pos0}", mv.len());
            }
            // the moved rows' positions exist once the launch's first phase wrote them
            let span = 1 + depth.iter().copied().max().unwrap_or(0) as usize;
            // before any `reserve`, which never gives capacity back
            if pos0 + span > MAX_CAP {
                anyhow::bail!("4E tree: positions up to {}, the scratch holds {MAX_CAP}", pos0 + span);
            }
            for c in caches.iter_mut() {
                if c.len() != move_pos0 {
                    anyhow::bail!("4E tree: a cache holds {} positions, the move starts at {move_pos0}", c.len());
                }
                c.reserve(pos0 + span)?;
                c.set_len(pos0)?;
            }
        }
        if n == 0 || n > pers::P4T_RMAX as usize || depth.len() != n || path.len() != n {
            anyhow::bail!("4E tree: {n} rows, {} depths, {} paths; the pass carries 1 to {}", depth.len(), path.len(), pers::P4T_RMAX);
        }
        for r in 0..n {
            let d = depth[r] as usize;
            let ok = d < 32
                && path[r][d] as usize == r
                && (0..=d).all(|e| (path[r][e] as usize) < n && depth[path[r][e] as usize] as usize == e && path[path[r][e] as usize][..=e] == path[r][..=e]);
            if !ok {
                anyhow::bail!("4E tree: row {r}'s path {:?} at depth {d} is not a path of this tree", &path[r][..=d.min(31)]);
            }
        }
        if self.tree.is_none() {
            anyhow::bail!("4E tree: not installed (Runtime4e::install_tree)");
        }
        if caches.len() != self.layers_host.len() {
            anyhow::bail!("4E: {} caches for {} layers", caches.len(), self.layers_host.len());
        }
        if pos0 < self.kernel_from {
            anyhow::bail!("4E tree: a launch at position {pos0}, below the kernel threshold {}", self.kernel_from);
        }
        let span = 1 + *depth.iter().max().expect("n > 0") as usize;
        if pos0 + span > MAX_CAP {
            anyhow::bail!("4E tree: positions up to {}, the scratch holds {MAX_CAP}", pos0 + span);
        }
        let spec = match self.tree.as_ref().expect("checked above").arms.get(arm) {
            Some(a) => a.clone(),
            None => anyhow::bail!("4E tree: arm {arm} of {} installed", self.tree.as_ref().map_or(0, |t| t.arms.len())),
        };
        if n > spec.form.rmax() as usize {
            anyhow::bail!("4E tree: arm {} carries at most {} rows a launch, {n} asked", spec.name, spec.form.rmax());
        }
        self.prepare(caches, pos0, span)?;
        // Rows, a12ip, a12ipg and a12ipg32 read the words 4E holds, through 4E's own images
        if !matches!(spec.form.parent(), TreeForm::Rows | TreeForm::A12ip | TreeForm::A12ipg | TreeForm::A12ipg32) {
            self.sync_tiled()?;
        }
        let stream = self.cuda.stream().clone();
        let v = self.tree.as_mut().expect("checked above");
        stream.memcpy_htod(toks, &mut v.toks).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        stream.memcpy_htod(depth, &mut v.depth).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        if !mv.is_empty() {
            stream.memcpy_htod(mv, &mut v.mv).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        }
        let flat: Vec<u8> = path.iter().flat_map(|p| p.iter().copied()).collect();
        stream.memcpy_htod(&flat, &mut v.path).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        stream.memcpy_htod(&vec![u32::MAX; pers::P4T_RMAX as usize], &mut v.ring).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        // a9 waits once more after each row phase: its count moves bar_base and sizes the stamps
        let n_bar = if matches!(spec.form.parent(), TreeForm::A9 | TreeForm::A10 | TreeForm::A11 | TreeForm::A11i | TreeForm::A10r8 | TreeForm::A10r16 | TreeForm::A10t384 | TreeForm::A10t256 | TreeForm::A10t448 | TreeForm::A10t320 | TreeForm::A10t384r16 | TreeForm::A12 | TreeForm::A12ip | TreeForm::A12ipg | TreeForm::A12ipg32) { pers::n_bar_a9(self.n_layers, v.n_ot) } else { pers::n_bar_4et(self.n_layers, v.n_ot) };
        let stamps: Option<CudaSlice<u64>> = if stamped {
            // an a4 launch adds its F3 slots after the phase events: 20 a CTA
            let extra = match spec.form {
                TreeForm::A4 | TreeForm::A5 => 20,
                TreeForm::A11i => 40,
                _ => 0,
            };
            Some(stream.alloc_zeros::<u64>((1 + 2 * n_bar + extra) as usize * self.grid as usize).map_err(|e| anyhow::anyhow!("alloc stamps: {e}"))?)
        } else {
            None
        };
        let a = v.held16_addr;
        let launch = pers::P4tLaunch {
            bar_base: self.bar_base,
            pos0: pos0 as u32,
            n_rows: n as u32,
            area_floats: spec.area,
            n_move: mv.len() as u32,
            move_pos0: move_pos0 as u32,
            part_stride: v.part_stride,
            ldx: v.ldx,
            split_qkv: spec.splits[0],
            split_o: spec.splits[1],
            split_gu: spec.splits[2],
            split_dn: spec.splits[3],
            // 0: the a12 body skips the final norm and the head, every wait kept
            split_head: if head { spec.splits[4] } else { 0 },
            // a12 with taps: first tapped layer 2, stride 8 (layers 2, 10, 18, 26, 34)
            pad: if taps && matches!(spec.form.parent(), TreeForm::A12 | TreeForm::A12ip | TreeForm::A12ipg | TreeForm::A12ipg32) { (8 << 16) | 2 } else { 0 },
            toks: v.toks.device_ptr(&stream).0,
            depth: v.depth.device_ptr(&stream).0,
            path: v.path.device_ptr(&stream).0,
            mv: v.mv.device_ptr(&stream).0,
            ring: v.ring.device_ptr(&stream).0,
            logits: if logits { v.logits.device_ptr(&stream).0 } else { 0 },
            x_a: a[0],
            x_b: a[1],
            xin: a[2],
            xn: a[3],
            q_raw: a[4],
            k_raw: a[5],
            v_raw: a[6],
            q_out: a[7],
            ctx: a[8],
            g_raw: a[9],
            u_raw: a[10],
            tk: a[11],
            tv: a[12],
            part: v.part.device_ptr(&stream).0,
            fpart: v.fpart.device_ptr(&stream).0,
            tickets: v.tickets.device_ptr(&stream).0,
            ftickets: v.ftickets.device_ptr(&stream).0,
            slots: v.slots.device_ptr(&stream).0,
            stamps: stamps.as_ref().map_or(0, |b| b.device_ptr(&stream).0),
        };
        // the rows module was built at 512 threads and the tree's own area
        debug_assert!(spec.form != TreeForm::Rows || (spec.smem == v.smem && spec.area == v.area && spec.threads == 512));
        let cfg = LaunchConfig { grid_dim: (self.grid, 1, 1), block_dim: (spec.threads, 1, 1), shared_mem_bytes: spec.smem };
        let (func, args) = match (spec.form, v.tiled.as_ref()) {
            (TreeForm::Rows, _) => (&v.f, &self.args),
            (TreeForm::Tiled, Some(TreeTiled { f: Some(f), args, .. })) => (f, args),
            (TreeForm::Stream, Some(TreeTiled { fs: Some(f), args, .. })) => (f, args),
            (TreeForm::A4, Some(TreeTiled { f4: Some(f), args, .. })) => (f, args),
            (TreeForm::A5, Some(TreeTiled { f5: Some(f), args, .. })) => (f, args),
            (TreeForm::A6, Some(TreeTiled { fx: [Some(f), ..], args, .. })) => (f, args),
            (TreeForm::A7, Some(TreeTiled { fx: [_, Some(f), ..], args, .. })) => (f, args),
            (TreeForm::A3f, Some(TreeTiled { fx: [_, _, Some(f), ..], args, .. })) => (f, args),
            (TreeForm::A3p, Some(TreeTiled { fx, args, .. })) if fx[3].is_some() => (fx[3].as_ref().expect("checked"), args),
            (TreeForm::A3t, Some(TreeTiled { fx, args, .. })) if fx[4].is_some() => (fx[4].as_ref().expect("checked"), args),
            (TreeForm::A8, Some(TreeTiled { fx, args, .. })) if fx[5].is_some() => (fx[5].as_ref().expect("checked"), args),
            (TreeForm::A9, Some(TreeTiled { fx, args, .. })) if fx[6].is_some() => (fx[6].as_ref().expect("checked"), args),
            (TreeForm::A10, Some(TreeTiled { fx, args, .. })) if fx[7].is_some() => (fx[7].as_ref().expect("checked"), args),
            (TreeForm::A11, Some(TreeTiled { fx, args, .. })) if fx[8].is_some() => (fx[8].as_ref().expect("checked"), args),
            (TreeForm::A11i, Some(TreeTiled { fx, args, .. })) if fx[9].is_some() => (fx[9].as_ref().expect("checked"), args),
            (TreeForm::A10r8, Some(TreeTiled { fx, args, .. })) if fx[10].is_some() => (fx[10].as_ref().expect("checked"), args),
            (TreeForm::A10r16, Some(TreeTiled { fx, args, .. })) if fx[11].is_some() => (fx[11].as_ref().expect("checked"), args),
            (TreeForm::A10t384, Some(TreeTiled { fx, args, .. })) if fx[12].is_some() => (fx[12].as_ref().expect("checked"), args),
            (TreeForm::A10t256, Some(TreeTiled { fx, args, .. })) if fx[13].is_some() => (fx[13].as_ref().expect("checked"), args),
            (TreeForm::A10t448, Some(TreeTiled { fx, args, .. })) if fx[14].is_some() => (fx[14].as_ref().expect("checked"), args),
            (TreeForm::A10t320, Some(TreeTiled { fx, args, .. })) if fx[15].is_some() => (fx[15].as_ref().expect("checked"), args),
            (TreeForm::A10t384r16, Some(TreeTiled { fx, args, .. })) if fx[16].is_some() => (fx[16].as_ref().expect("checked"), args),
            (TreeForm::A12, Some(TreeTiled { fx, args, .. })) if fx[17].is_some() => (fx[17].as_ref().expect("checked"), args),
            (TreeForm::A12ip, Some(TreeTiled { fx, .. })) if fx[18].is_some() => (fx[18].as_ref().expect("checked"), &self.args),
            (TreeForm::A12ipg, Some(TreeTiled { fx, .. })) if fx[19].is_some() => (fx[19].as_ref().expect("checked"), &self.args),
            (TreeForm::A12ipg32, Some(TreeTiled { fx, .. })) if fx[20].is_some() => (fx[20].as_ref().expect("checked"), &self.args),
            (TreeForm::A12ipFt, Some(TreeTiled { fx, .. })) if fx[21].is_some() => (fx[21].as_ref().expect("checked"), &self.args),
            (TreeForm::A12ipgFt, Some(TreeTiled { fx, .. })) if fx[22].is_some() => (fx[22].as_ref().expect("checked"), &self.args),
            (TreeForm::A12ipg32Ft, Some(TreeTiled { fx, .. })) if fx[23].is_some() => (fx[23].as_ref().expect("checked"), &self.args),
            (form, _) => anyhow::bail!("4E tree: arm {} is {form:?} and its module is not installed", spec.name),
        };
        let mut b = stream.launch_builder(func);
        b.arg(args).arg(&launch);
        // SAFETY: the kernel's own signature (Pers4eArgs*, Pers4tLaunch), the
        // layout checked on the host (tree_verify_host.rs); one CTA a SM, past
        // the occupancy check at install.
        unsafe { b.launch_cooperative(cfg) }.map_err(|e| anyhow::anyhow!("4E tree: {e}"))?;
        let ring = stream.clone_dtoh(&v.ring).map_err(|e| anyhow::anyhow!("D2H: {e}"))?;
        let lg = if logits {
            let all = stream.clone_dtoh(&v.logits).map_err(|e| anyhow::anyhow!("D2H logits: {e}"))?;
            all[..n * v.vocab as usize].to_vec()
        } else {
            Vec::new()
        };
        match self.settle()? {
            Some(pers::P4eFault::Abort) => {
                anyhow::bail!("4E tree: a barrier aborted on its watchdog; the counters are reset for the next launch")
            }
            Some(pers::P4eFault::Nan) => anyhow::bail!("4E tree: a NaN logit"),
            None => {}
        }
        let made: Vec<u32> = if head { ring[..n].to_vec() } else { Vec::new() };
        if made.contains(&u32::MAX) {
            let mut dev = Counters4e { cuda: &self.cuda, bar: &mut self.bar, abort: &mut self.abort, nan: &mut self.nan };
            pers::p4e_reset(&mut dev, &mut self.bar_base).map_err(anyhow::Error::msg)?;
            anyhow::bail!("4E tree: a row wrote no token ({made:?}); the counters are reset for the next launch");
        }
        self.bar_base = self.bar_base.wrapping_add(n_bar.wrapping_mul(self.grid));
        let stamps = match stamps {
            Some(d) => stream.clone_dtoh(&d).map_err(|e| anyhow::anyhow!("D2H stamps: {e}"))?,
            None => Vec::new(),
        };
        let taps_out = if taps && matches!(spec.form.parent(), TreeForm::A12 | TreeForm::A12ip | TreeForm::A12ipg | TreeForm::A12ipg32) {
            // [rows][5][hidden] past x_a's P4T_RMAX rows (held16 0 is x_a)
            let v = self.tree.as_ref().expect("checked above");
            let h = self.args_host.hidden as usize;
            let all = stream.clone_dtoh(&v._held16[0]).map_err(|e| anyhow::anyhow!("D2H taps: {e}"))?;
            let base = pers::P4T_RMAX as usize * h;
            all[base..base + n * 5 * h].to_vec()
        } else {
            Vec::new()
        };
        Ok(TreeOut { ring: made, logits: lg, stamps, taps: taps_out })
    }
}
