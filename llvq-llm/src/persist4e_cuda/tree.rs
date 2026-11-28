//! The tree verify pass, its types and its install: the arm forms and plans,
//! how the tree modules load, the tiled module and the tree-tiled copies.
//! One file because each type here is what `install_tree` reads or builds.

use super::*;

/// The kernel an arm of the tree pass runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeForm {
    /// `pers_4et_rows_*`: the words in their row layout (arms a0, a1).
    Rows,
    /// `pers_4et_tiled_*` (`P4T_TILED`): the tree-tiled copies staged by
    /// chunk (arm a2).
    Tiled,
    /// `pers_4et_stream_*` (`P4T_STREAM`): the tree-tiled copies streamed by
    /// each warp, the activation slice staged once a unit (arm a3).
    Stream,
    /// `pers_4et_a4_*` (`P4T_A4`): the stream unit with the fold's loads
    /// batched and the prologues on 16-byte vectors (arm a4).
    A4,
    /// `pers_4et_a5_*` (`P4T_A5`): a4 with the L2 prefetch of the next row
    /// phase's tile streams on 4E's schedule (arm a5).
    A5,
    /// `pers_4et_a6_*` (`P4T_A6`): a3 with the int4 down's input dealt over
    /// the grid (arm a6).
    A6,
    /// `pers_4et_a7_*` (`P4T_A7`): a6 with a5's prefetch points (arm a7).
    A7,
    /// `pers_4et_a3f_*` (`P4T_A3F`): a FLOOR arm, timing only, values void by
    /// design: a3 with no slice fold (slice 0 stored as if S were 1).
    A3f,
    /// `pers_4et_a3p_*`: a FLOOR arm, values void: a3f plus every slice's
    /// partial store.
    A3p,
    /// `pers_4et_a3t_*`: a FLOOR arm, values void: a3p plus the fences and
    /// tickets, no fold.
    A3t,
    /// `pers_4et_a8_*` (`P4T_A8`): a3 with the fold reading 16-byte quarters
    /// (arm a8), bit for bit with a3.
    A8,
    /// `pers_4et_a9_*` (`P4T_A9`): a3 with the slice fold after the phase's
    /// grid wait, one more wait a row phase (`pers::n_bar_a9`), bit for bit.
    A9,
    /// `pers_4et_a10_*` (`P4T_A10`): a9 with 4E's prologue path and the
    /// silu in the gate/up fold step (arm a10), a9's waits, bit for bit.
    A10,
    /// `pers_4et_a11_*` (`P4T_A11`): a10 with the L2 prefetch of the row
    /// phases' streams in its wait gaps (arm a11), bit for bit with a10.
    A11,
    /// `pers_4et_a11i_*` (`P4T_A11I`): a11 with fine stamps (40 slots a CTA
    /// after the phase events on a stamped launch), bit for bit with a11.
    A11i,
    /// `pers_4et_a10r8_*` (`P4T_A10RC 8`): a10 for at most 8 rows a launch
    /// (one n8 tile of accumulators), bit for bit with a10 at those rows.
    A10r8,
    /// `pers_4et_a10r16_*` (`P4T_A10RC 16`): a10 for at most 16 rows.
    A10r16,
    /// `pers_4et_a10_*_t384` (`P4T_A10T 384`): a10 at 12 warps a CTA.
    A10t384,
    /// `pers_4et_a10_*_t256` (`P4T_A10T 256`): a10 at 8 warps a CTA.
    A10t256,
    /// `pers_4et_a10_*_t448` (`P4T_A10T2 448`): a10 at 14 warps a CTA.
    A10t448,
    /// `pers_4et_a10_*_t320` (`P4T_A10T2 320`): a10 at 10 warps a CTA.
    A10t320,
    /// `pers_4et_a10r16_*_t384` (`P4T_A10T2 38416`): a10r16 at 12 warps.
    A10t384r16,
    /// `pers_4et_a12_*_t384` (`P4T_A12`): a10t384r16 that also writes each
    /// row's taps, the residual entering layers 2, 10, 18, 26, 34.
    A12,
    /// `pers_4et_a12ip_*_t384` (`P4T_A12IP`): a12 on the words 4E and the
    /// eager path hold, read in place (step M1 of the memory axis). No
    /// tree-tiled copy is built; the install checks, projection by
    /// projection, that every lane's words equal the copy's
    /// (`p4t_ip_check`), each copy freed after its check. Bit for bit with
    /// a12 by construction (the same words into the same decode).
    A12ip,
    /// `pers_4et_a12ipg_*_t384` (`P4T_A12IPG`): a12ip whose attention reads
    /// each split that ends at or below pos0 once for a group of
    /// `pers::P4G_RG` rows (D1 of the large-prefill study), bit for bit with
    /// a12ip; the same words read in place, the same load check, a larger
    /// shared area (`pers::p4g_area_floats`).
    A12ipg,
    /// `pers_4et_a12ipg32_*_t384` (`P4T_A12IPG32`): a12ipg's pass at RC = 4,
    /// up to 32 rows a launch (design A of the large-prefill rows study),
    /// bit for bit with a12ipg at 16; the same words, check and area.
    A12ipg32,
    /// a12ip's module with the attention tile's full-tile path
    /// (`pers::p4t_ft_unit_text`, arm a12ipft): the same symbols, words,
    /// check and area, bit for bit with a12ip.
    A12ipFt,
    /// a12ipg's module with the full-tile path (arm a12ipgft), bit for bit
    /// with a12ipg.
    A12ipgFt,
    /// a12ipg32's module with the full-tile path (arm a12ipg32ft), bit for
    /// bit with a12ipg32.
    A12ipg32Ft,
}

impl TreeForm {
    /// The form whose kernel a full-tile form compiles (its module is that
    /// form's text with the ft edit), else the form itself. Everything but the
    /// module's text follows the parent.
    pub fn parent(self) -> TreeForm {
        match self {
            TreeForm::A12ipFt => TreeForm::A12ip,
            TreeForm::A12ipgFt => TreeForm::A12ipg,
            TreeForm::A12ipg32Ft => TreeForm::A12ipg32,
            f => f,
        }
    }

    /// The rows one launch of this kernel carries at most.
    pub fn rmax(self) -> u32 {
        match self.parent() {
            TreeForm::A10r8 => 8,
            TreeForm::A10r16 | TreeForm::A10t384r16 | TreeForm::A12 | TreeForm::A12ip | TreeForm::A12ipg => 16,
            TreeForm::A12ipg32 => 32,
            _ => pers::P4T_RMAX,
        }
    }

    /// The threads of one CTA of this kernel.
    pub fn threads(self) -> u32 {
        match self.parent() {
            TreeForm::A10t384 | TreeForm::A10t384r16 | TreeForm::A12 | TreeForm::A12ip | TreeForm::A12ipg | TreeForm::A12ipg32 => 384,
            TreeForm::A10t256 => 256,
            TreeForm::A10t448 => 448,
            TreeForm::A10t320 => 320,
            _ => 512,
        }
    }
}

/// Where an arm's d_in slices come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreePlan {
    /// S1's `pers::p4t_splits`.
    S1,
    /// A proposal `pers::p4t_splits_balanced` keeps or replaces (job N1).
    Balanced([u32; 5]),
    /// `pers::p4t_splits_stream` at the stream unit's cap (job N2).
    Stream,
    /// Slices given by the bench (`arm@q.o.g.d.h`), refused when a slice
    /// exceeds the stream unit's cap or an S is outside 1..=32.
    Explicit([u32; 5]),
}

/// One arm of the tree pass: the d_in slices of q/k/v, o, gate/up, down and
/// the head, the plan they came from, and the kernel.
#[derive(Clone, Debug)]
pub struct TreeArm {
    pub name: String,
    pub splits: [u32; 5],
    pub plan: TreePlan,
    pub form: TreeForm,
    /// Why the arm cannot run (its module did not build), or None.
    pub void: Option<String>,
    /// Its CTA's threads, its area (floats after the tables) and its dynamic
    /// shared bytes, at its warp count.
    pub threads: u32,
    pub area: u32,
    pub smem: u32,
}

/// An arm asked of [`Runtime4e::install_tree`]: its name, its plan, its
/// kernel.
pub type TreeArmSpec<'a> = (&'a str, TreePlan, TreeForm);

/// What one tree launch returns: each row's argmax, its binary16 logits when
/// asked (`[rows][vocab]`), the phase clocks when stamped.
pub struct TreeOut {
    pub ring: Vec<u32>,
    pub logits: Vec<u16>,
    pub stamps: Vec<u64>,
    /// a12 with taps asked: each row's residual entering layers 2, 10, 18,
    /// 26, 34, `[rows][5][hidden]` binary16; empty otherwise.
    pub taps: Vec<u16>,
}

/// How the tree modules are loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeModules {
    /// NVRTC and the driver's PTX load at every start: every bench of the
    /// tree pass ran so, and `bin/prefillbench` still does.
    Ptx,
    /// The driver's cubin (`gpu::Cuda::on_stream_cached`): read from the
    /// directory when a file of the exact key is there, else compiled and
    /// written to it; `None` compiles the same cubin at every start and
    /// writes it nowhere. The served chain prefill's form.
    Cubin(Option<llvq_cuda::modcache::Dir>),
}

/// One tree module, loaded as `how` says; a cubin load appends its line.
fn tree_module(
    stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    src: &llvq_cuda::gpu::KernelSource,
    how: &TreeModules,
    lines: &mut Vec<String>,
) -> anyhow::Result<llvq_cuda::gpu::Cuda> {
    match how {
        TreeModules::Ptx => llvq_cuda::gpu::Cuda::on_stream(stream.clone(), src).map_err(anyhow::Error::msg),
        TreeModules::Cubin(dir) => {
            let (cuda, line) = llvq_cuda::gpu::Cuda::on_stream_cached(stream.clone(), src, dir.as_ref()).map_err(anyhow::Error::msg)?;
            lines.push(format!("4E tree: {line}"));
            Ok(cuda)
        }
    }
}

/// What a chain prefill leaves ([`Runtime4e::chain_prefill`]).
pub struct ChainPrefill {
    /// The first answer token: the last prompt row's argmax.
    pub first: u32,
    /// Seconds to the first token, and to the end of the hand-over launch.
    pub ttft: f64,
    pub prefill: f64,
}

impl Runtime4e {
    /// Install the tree verify pass ([`Self::tree`]): its module (row form 7
    /// and `pers_4et.cu`, its own module, so the served module's text does
    /// not move), 512 threads, the area `pers::p4t_area_floats` gives for 32
    /// rows, the model's slices on this grid (`pers::p4t_splits`), and
    /// buffers for `pers::P4T_RMAX` rows. Refused below row form 7 and when
    /// one CTA a SM does not fit. `arms`: the slices and forms a launch picks
    /// from ([`Self::tree`]'s `arm`); a tiled arm installs the tiled module
    /// and the tree-tiled copies.
    pub fn install_tree(&mut self, arms: &[TreeArmSpec]) -> anyhow::Result<()> {
        if self.rows_form < 7 {
            anyhow::bail!("4E tree: the pass reads the int4 rows on tiled words (row form 7, v9); this runtime is form {}", self.rows_form);
        }
        if self.tree.is_some() {
            return Ok(());
        }
        let stream = self.cuda.stream().clone();
        let text = pers::p4t_unit_text(&self.tile_define);
        let src = llvq_cuda::gpu::KernelSource::new(&[text.as_str()]);
        self.lines.push(format!("4E tree: module {} bytes, sha256 {}, row form 7 and pers_4et.cu, its own module", src.text.len(), src.sha256));
        let how = self.tree_modules.clone();
        let cuda = tree_module(&stream, &src, &how, &mut self.lines)?;
        let dev = cuda.device().map_err(anyhow::Error::msg)?;
        let (hidden, n_heads, n_kv, groups, vocab) =
            (self.args_host.hidden, self.args_host.n_heads, self.args_host.n_kv, self.args_host.groups, self.args_host.vocab);
        let (d_ctx, inter, threads) = (n_heads * 128, self.inter, 512u32);
        let nw = threads / 32;
        let widest = hidden.max(inter).max(d_ctx);
        let area = pers::p4t_area_floats(nw, 32, widest, groups);
        let smem = pers::SMEM_HEADER + pers::TABLE_BYTES + 4 * area;
        if smem > dev.shared_per_block_optin as u32 {
            anyhow::bail!("4E tree: {smem} B of shared memory a CTA, the card allows {}", dev.shared_per_block_optin);
        }
        let sym = pers::p4t_symbol(self.form).map_err(anyhow::Error::msg)?;
        let f = cuda.func_dynamic_shared(sym, smem).map_err(anyhow::Error::msg)?;
        let r = cuda.report(sym).map_err(anyhow::Error::msg)?;
        let occ = f
            .occupancy_max_active_blocks_per_multiprocessor(threads, smem as usize, None)
            .map_err(|e| anyhow::anyhow!("occupancy: {e}"))?;
        self.lines.push(format!("4E tree: {sym}: {} registers, {} B local, {smem} B shared, {occ} resident a SM", r.num_regs, r.local_bytes));
        if occ < 1 {
            anyhow::bail!("4E tree: {sym} fits no CTA on a SM at {threads} threads and {smem} B");
        }
        let s1 = pers::p4t_splits(hidden, inter, d_ctx, n_kv, vocab, nw, self.grid);
        if arms.is_empty() {
            anyhow::bail!("4E tree: no arm asked");
        }
        let mut arm_v = Vec::with_capacity(arms.len());
        let d_ins = [hidden, d_ctx, hidden, inter, hidden];
        for &(name, prop, form) in arms {
            // the arm's own geometry: its warps set its area, its slice cap and its plan
            let (thr_a, nw) = (form.threads(), form.threads() / 32);
            let area_a = if matches!(form.parent(), TreeForm::A12ipg | TreeForm::A12ipg32) { pers::p4g_area_floats(nw, 32, widest, groups) } else { pers::p4t_area_floats(nw, 32, widest, groups) };
            let smem_a = pers::SMEM_HEADER + pers::TABLE_BYTES + 4 * area_a;
            if smem_a > dev.shared_per_block_optin as u32 {
                anyhow::bail!("4E tree: arm {name}: {smem_a} B of shared memory a CTA, the card allows {}", dev.shared_per_block_optin);
            }
            let cap = pers::p4t_stream_cap(area_a, nw);
            let s1 = if nw == threads / 32 { s1 } else { pers::p4t_splits(hidden, inter, d_ctx, n_kv, vocab, nw, self.grid) };
            let splits = match prop {
                TreePlan::S1 => s1,
                TreePlan::Balanced(p) => {
                    let b = pers::p4t_splits_balanced(p, hidden, inter, d_ctx, n_kv, vocab, nw, self.grid);
                    if b != p {
                        self.lines.push(format!("4E tree: arm {name}: the proposed slices {p:?} fill under 90 % on q/k/v or gate/up; {b:?} instead"));
                    }
                    b
                }
                TreePlan::Stream => pers::p4t_splits_stream(hidden, inter, d_ctx, n_kv, vocab, nw, self.grid, cap).map_err(anyhow::Error::msg)?,
                TreePlan::Explicit(p) => {
                    if p.iter().any(|&x| x == 0 || x > 32) || d_ins.iter().zip(p).any(|(&d, x)| pers::p4t_slice_steps(d, x) > cap) {
                        anyhow::bail!("4E tree: arm {name}: the explicit plan {p:?} has an S outside 1..=32 or under the stream unit's floor ({cap} sub-steps a slice)");
                    }
                    p
                }
            };
            if !matches!(form, TreeForm::Rows | TreeForm::Tiled) && d_ins.iter().zip(splits).any(|(&d, s)| pers::p4t_slice_steps(d, s) > cap) {
                anyhow::bail!("4E tree: arm {name}: slices {splits:?} exceed the stream unit's {cap} sub-steps a slice");
            }
            let plan = pers::p4t_plan(splits, hidden, inter, d_ctx, n_kv, vocab, nw, self.grid);
            let names = ["q/k/v", "o", "gate/up", "down", "head"];
            let plan_name = match prop {
                TreePlan::S1 => "S1's plan",
                TreePlan::Balanced(_) => "the balanced plan",
                TreePlan::Stream => "the stream plan",
                TreePlan::Explicit(_) => "an explicit plan",
            };
            self.lines.push(format!(
                "4E tree: arm {name} ({}, {plan_name}): slices {splits:?}; S, units, waves, fill, sub-steps a slice at most: {}",
                match form {
                    TreeForm::Rows => "rows",
                    TreeForm::Tiled => "tiled",
                    TreeForm::Stream => "stream",
                    TreeForm::A4 => "a4",
                    TreeForm::A5 => "a5",
                    TreeForm::A6 => "a6",
                    TreeForm::A7 => "a7",
                    TreeForm::A3f => "a3f (floor, values void)",
                    TreeForm::A3p => "a3p (floor, values void)",
                    TreeForm::A3t => "a3t (floor, values void)",
                    TreeForm::A8 => "a8",
                    TreeForm::A9 => "a9",
                    TreeForm::A10 => "a10",
                    TreeForm::A11 => "a11",
                    TreeForm::A11i => "a11i (fine stamps)",
                    TreeForm::A10r8 => "a10r8 (at most 8 rows)",
                    TreeForm::A10r16 => "a10r16 (at most 16 rows)",
                    TreeForm::A10t384 => "a10t384 (12 warps)",
                    TreeForm::A10t256 => "a10t256 (8 warps)",
                    TreeForm::A10t448 => "a10t448 (14 warps)",
                    TreeForm::A10t320 => "a10t320 (10 warps)",
                    TreeForm::A10t384r16 => "a10t384r16 (12 warps, at most 16 rows)",
                    TreeForm::A12 => "a12 (a10t384r16 with the taps)",
                    TreeForm::A12ip => "a12ip (a12 on the served words, read in place, no copy)",
                    TreeForm::A12ipg => "a12ipg (a12ip, the cache read once for a group of rows)",
                    TreeForm::A12ipg32 => "a12ipg32 (a12ipg at 32 rows a launch)",
                    TreeForm::A12ipFt => "a12ipft (a12ip, the attention tile's full-tile path)",
                    TreeForm::A12ipgFt => "a12ipgft (a12ipg, the attention tile's full-tile path)",
                    TreeForm::A12ipg32Ft => "a12ipg32ft (a12ipg32, the attention tile's full-tile path)",
                },
                names
                    .iter()
                    .zip(plan)
                    .zip(splits.iter().zip(d_ins))
                    .map(|((n, (u, w, f)), (&s, d))| format!("{n} {s} {u} {w} {:.1} % {}", 100.0 * f, pers::p4t_slice_steps(d, s)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            if thr_a != threads {
                self.lines.push(format!("4E tree: arm {name} at {thr_a} threads ({nw} warps): area {area_a} floats, {smem_a} B shared, {cap} sub-steps a slice at most"));
            }
            arm_v.push(TreeArm { name: name.to_string(), splits, plan: prop, form, void: None, threads: thr_a, area: area_a, smem: smem_a });
        }
        if arm_v.iter().any(|a| !matches!(a.form, TreeForm::Rows | TreeForm::Tiled)) {
            // the 512-thread arms' cap; an arm at another warp count prints its own above
            let cap = pers::p4t_stream_cap(area, nw);
            self.lines.push(format!("4E tree: the stream unit's slices hold at most {cap} sub-steps (area {area} floats, {nw} warps, 32 staged rows)"));
        }
        let n_ot = self.layers_host.iter().filter(|y| y.ot != 0).count() as u32;
        let ldx = widest.next_multiple_of(8);
        let rm = pers::P4T_RMAX;
        let dkv = n_kv * 128;
        let l = self.layers_host.len() as u32;
        // x_a's buffer carries a12's taps past its rm rows: [rm][5][hidden]
        let sizes = [rm * hidden * 6, rm * hidden, rm * ldx, rm * hidden, rm * d_ctx, rm * dkv, rm * dkv, rm * d_ctx, rm * d_ctx, rm * inter, rm * inter, l * rm * dkv, l * rm * dkv];
        let mut held = Vec::with_capacity(sizes.len());
        let mut held16_addr = [0u64; 13];
        for (a, &n) in held16_addr.iter_mut().zip(&sizes) {
            let b = cuda.zeros_u16(n as usize).map_err(anyhow::Error::msg)?;
            *a = addr16(&b, &stream);
            held.push(b);
        }
        let part_stride = n_kv * (MAX_CAP as u32).div_ceil(256) * groups * 130;
        let fp = arm_v.iter().map(|a| pers::p4t_fpart_floats(a.splits, hidden, inter, d_ctx, n_kv, vocab)).max().unwrap_or(0).max(1) as usize;
        let nft = pers::p4t_ftickets(hidden, inter, d_ctx, n_kv, vocab) as usize;
        let e32 = |r: Result<CudaSlice<u32>, String>| r.map_err(anyhow::Error::msg);
        let part = cuda.zeros_f32((rm * part_stride) as usize).map_err(anyhow::Error::msg)?;
        let fpart = cuda.zeros_f32(fp).map_err(anyhow::Error::msg)?;
        let tickets = e32(cuda.up_u32(&vec![0u32; (rm * n_kv) as usize]))?;
        let ftickets = e32(cuda.up_u32(&vec![0u32; nft]))?;
        let slots = cuda.up_u64(&vec![0u64; rm as usize]).map_err(anyhow::Error::msg)?;
        let toks = e32(cuda.up_u32(&vec![0u32; rm as usize]))?;
        let depth = e32(cuda.up_u32(&vec![0u32; rm as usize]))?;
        let mv = e32(cuda.up_u32(&vec![0u32; rm as usize]))?;
        let ring = e32(cuda.up_u32(&vec![u32::MAX; rm as usize]))?;
        let path = stream.alloc_zeros::<u8>((rm * rm) as usize).map_err(|e| anyhow::anyhow!("alloc: {e}"))?;
        let logits = cuda.zeros_u16((rm * vocab) as usize).map_err(anyhow::Error::msg)?;
        let bytes = 2 * sizes.iter().map(|&n| n as u64).sum::<u64>() + 4 * (rm * part_stride) as u64 + 4 * fp as u64 + 2 * (rm * vocab) as u64;
        self.lines.push(format!("4E tree: {threads} threads, {area} floats of area; {:.3} GB of row buffers", bytes as f64 / 1e9));
        let want = [
            TreeForm::Tiled,
            TreeForm::Stream,
            TreeForm::A4,
            TreeForm::A5,
            TreeForm::A6,
            TreeForm::A7,
            TreeForm::A3f,
            TreeForm::A3p,
            TreeForm::A3t,
            TreeForm::A8,
            TreeForm::A9,
            TreeForm::A10,
            TreeForm::A11,
            TreeForm::A11i,
            TreeForm::A10r8,
            TreeForm::A10r16,
            TreeForm::A10t384,
            TreeForm::A10t256,
            TreeForm::A10t448,
            TreeForm::A10t320,
            TreeForm::A10t384r16,
            TreeForm::A12,
            TreeForm::A12ip,
            TreeForm::A12ipg,
            TreeForm::A12ipg32,
            TreeForm::A12ipFt,
            TreeForm::A12ipgFt,
            TreeForm::A12ipg32Ft,
        ]
        .map(|f| arm_v.iter().any(|a| a.form == f));
        let tiled = if want.iter().any(|&w| w) { Some(self.install_tiled(widest, groups, want)?) } else { None };
        // an arm whose module did not build is void; the others run
        if let Some(t) = &tiled {
            for a in arm_v.iter_mut() {
                let built = match a.form {
                    TreeForm::Rows => true,
                    TreeForm::Tiled => t.f.is_some(),
                    TreeForm::Stream => t.fs.is_some(),
                    TreeForm::A4 => t.f4.is_some(),
                    TreeForm::A5 => t.f5.is_some(),
                    TreeForm::A6 => t.fx[0].is_some(),
                    TreeForm::A7 => t.fx[1].is_some(),
                    TreeForm::A3f => t.fx[2].is_some(),
                    TreeForm::A3p => t.fx[3].is_some(),
                    TreeForm::A3t => t.fx[4].is_some(),
                    TreeForm::A8 => t.fx[5].is_some(),
                    TreeForm::A9 => t.fx[6].is_some(),
                    TreeForm::A10 => t.fx[7].is_some(),
                    TreeForm::A11 => t.fx[8].is_some(),
                    TreeForm::A11i => t.fx[9].is_some(),
                    TreeForm::A10r8 => t.fx[10].is_some(),
                    TreeForm::A10r16 => t.fx[11].is_some(),
                    TreeForm::A10t384 => t.fx[12].is_some(),
                    TreeForm::A10t256 => t.fx[13].is_some(),
                    TreeForm::A10t448 => t.fx[14].is_some(),
                    TreeForm::A10t320 => t.fx[15].is_some(),
                    TreeForm::A10t384r16 => t.fx[16].is_some(),
                    TreeForm::A12 => t.fx[17].is_some(),
                    TreeForm::A12ip => t.fx[18].is_some(),
                    TreeForm::A12ipg => t.fx[19].is_some(),
                    TreeForm::A12ipg32 => t.fx[20].is_some(),
                    TreeForm::A12ipFt => t.fx[21].is_some(),
                    TreeForm::A12ipgFt => t.fx[22].is_some(),
                    TreeForm::A12ipg32Ft => t.fx[23].is_some(),
                };
                if !built {
                    a.void = Some(format!("its {:?} module did not build", a.form));
                    self.lines.push(format!("4E tree: arm {} VOID: its {:?} module did not build", a.name, a.form));
                }
            }
        }
        self.tree = Some(Tree4e {
            _cuda: cuda, f, smem, area, arms: arm_v, tiled, part_stride, ldx, n_ot, vocab, toks, depth, path, mv, ring, logits,
            _held16: held, held16_addr, part, fpart, tickets, ftickets, slots,
        });
        Ok(())
    }

    /// The tree-tiled arms: the tiled module (`pers::p4t_tiled_unit_text`)
    /// when `want_tiled`, the stream module (`pers::p4t_stream_unit_text`)
    /// when `want_stream`, each symbol at the same shared bytes, and the
    /// tree-tiled copy of every projection the pass reads (q, k, v, o, gate,
    /// up, down, the head), built on the card by `p4t_tile_words` from the
    /// words 4E holds, once for both. A Tetra o is refused (no sealed file the
    /// job reads carries one).
    fn install_tiled(&mut self, widest: u32, groups: u32, want: [bool; 28]) -> anyhow::Result<TreeTiled> {
        if self.layers_host.iter().any(|y| y.ot != 0) {
            anyhow::bail!("4E tree: the tiled and stream arms do not copy a Tetra o_proj; this model carries one");
        }
        let stream = self.cuda.stream().clone();
        let how = self.tree_modules.clone();
        let mut cudas = Vec::new();
        let mut fns: [Option<CudaFunction>; 28] = Default::default();
        // the a12ip module's place in `cudas`: its load check lives there
        let mut ip_module: Option<usize> = None;
        for (k, want) in want.into_iter().enumerate() {
            if !want {
                continue;
            }
            let (tag, text, sym, def) = match k {
                0 => ("tiled", pers::p4t_tiled_unit_text(&self.tile_define), pers::p4t_tiled_symbol(self.form), "P4T_TILED 1"),
                1 => ("stream", pers::p4t_stream_unit_text(&self.tile_define), pers::p4t_stream_symbol(self.form), "P4T_STREAM 1 and P4T_TILED 1"),
                2 => ("a4", pers::p4t_a4_unit_text(&self.tile_define, true, true), pers::p4t_a4_symbol(self.form), "P4T_A4 1 (F1, F2), P4T_STREAM 1 and P4T_TILED 1"),
                3 => ("a5", pers::p4t_a5_unit_text(&self.tile_define), pers::p4t_a5_symbol(self.form), "P4T_A5 1, P4T_A4 1 (F1, F2), P4T_STREAM 1 and P4T_TILED 1"),
                10 => ("a9", pers::p4t_a9_unit_text(&self.tile_define), pers::p4t_a9_symbol(self.form), "P4T_A9 1 after a3's text"),
                11 => ("a10", pers::p4t_a10_unit_text(&self.tile_define), pers::p4t_a10_symbol(self.form), "P4T_A10 1 after a9's text"),
                12 => ("a11", pers::p4t_a11_unit_text(&self.tile_define), pers::p4t_a11_symbol(self.form), "P4T_A11 1 after a10's text"),
                13 => ("a11i", pers::p4t_a11i_unit_text(&self.tile_define), pers::p4t_a11i_symbol(self.form), "P4T_A11I 1 after a11's text"),
                14 => ("a10r8", pers::p4t_a10rc_unit_text(&self.tile_define, 8), pers::p4t_a10rc_symbol(8, self.form), "P4T_A10RC 8 after a10's text"),
                15 => ("a10r16", pers::p4t_a10rc_unit_text(&self.tile_define, 16), pers::p4t_a10rc_symbol(16, self.form), "P4T_A10RC 16 after a10's text"),
                16 => ("a10t384", pers::p4t_a10t_unit_text(&self.tile_define, 384), pers::p4t_a10t_symbol(384, self.form), "P4T_A10T 384 after a10's text"),
                17 => ("a10t256", pers::p4t_a10t_unit_text(&self.tile_define, 256), pers::p4t_a10t_symbol(256, self.form), "P4T_A10T 256 after a10's text"),
                21 => ("a12", pers::p4t_a12_unit_text(&self.tile_define), pers::p4t_a12_symbol(self.form), "P4T_A12 1 after a10r16's text"),
                22 => ("a12ip", pers::p4t_a12ip_unit_text(&self.tile_define), pers::p4t_a12ip_symbol(self.form), "P4T_A12IP 1 beside a12's text"),
                23 => ("a12ipg", pers::p4t_a12ipg_unit_text(&self.tile_define), pers::p4t_a12ipg_symbol(self.form), "P4T_A12IPG 1 after a12ip's text"),
                24 => ("a12ipg32", pers::p4t_a12ipg32_unit_text(&self.tile_define), pers::p4t_a12ipg32_symbol(self.form), "P4T_A12IPG32 1 after a12ipg's text"),
                25..=27 => {
                    let (arm, sym) = match k {
                        25 => ("a12ipft", pers::p4t_a12ip_symbol(self.form)),
                        26 => ("a12ipgft", pers::p4t_a12ipg_symbol(self.form)),
                        _ => ("a12ipg32ft", pers::p4t_a12ipg32_symbol(self.form)),
                    };
                    let text = pers::p4t_ft_unit_text(&self.tile_define, arm).map_err(anyhow::Error::msg)?;
                    (arm, text, sym, "its parent's text with the full-tile path (pers_attn_full.cuh)")
                }
                18..=20 => {
                    let arm = match k {
                        18 => "a10t448",
                        19 => "a10t320",
                        _ => "a10t384r16",
                    };
                    let text = pers::p4t_a10t2_unit_text(&self.tile_define, arm).map_err(anyhow::Error::msg)?;
                    (arm, text, pers::p4t_a10t2_symbol(arm, self.form), "P4T_A10T2 after a10's or a10r16's text")
                }
                7..=9 => {
                    let (arm, def) = match k {
                        7 => ("a3p", "P4T_A3P 1 after a3's text, a floor arm"),
                        8 => ("a3t", "P4T_A3T 1 after a3's text, a floor arm"),
                        _ => ("a8", "P4T_A8 1 after a3's text"),
                    };
                    let text = pers::p4t_a8_unit_text(arm, &self.tile_define).map_err(anyhow::Error::msg)?;
                    (arm, text, pers::p4t_a8_symbol(arm, self.form), def)
                }
                _ => {
                    let (arm, def) = match k {
                        4 => ("a6", "P4T_A6 1 after a3's text"),
                        5 => ("a7", "P4T_A7 1 after a5's text"),
                        _ => ("a3f", "P4T_A3F 1 after a3's text, a floor arm"),
                    };
                    let text = pers::p4t_a6_unit_text(arm, &self.tile_define).map_err(anyhow::Error::msg)?;
                    (arm, text, pers::p4t_a6_symbol(arm, self.form), def)
                }
            };
            let sym = sym.map_err(anyhow::Error::msg)?;
            // the module's CTA: 384 or 256 threads for the thread-count arms, else 512
            let threads = match k {
                16 | 20..=27 => 384u32,
                17 => 256,
                18 => 448,
                19 => 320,
                _ => 512,
            };
            let area_k = if matches!(k, 23 | 24 | 26 | 27) { pers::p4g_area_floats(threads / 32, 32, widest, groups) } else { pers::p4t_area_floats(threads / 32, 32, widest, groups) };
            let smem = pers::SMEM_HEADER + pers::TABLE_BYTES + 4 * area_k;
            let src = llvq_cuda::gpu::KernelSource::new(&[text.as_str()]);
            self.lines.push(format!("4E tree {tag}: module {} bytes, sha256 {}, {def}, its own module", src.text.len(), src.sha256));
            // a module that fails to build or to fit voids its arm only
            let built = (|| -> anyhow::Result<(llvq_cuda::gpu::Cuda, CudaFunction)> {
                let cuda = tree_module(&stream, &src, &how, &mut self.lines)?;
                let f = cuda.func_dynamic_shared(sym, smem).map_err(anyhow::Error::msg)?;
                let r = cuda.report(sym).map_err(anyhow::Error::msg)?;
                let occ = f
                    .occupancy_max_active_blocks_per_multiprocessor(threads, smem as usize, None)
                    .map_err(|e| anyhow::anyhow!("occupancy: {e}"))?;
                self.lines.push(format!("4E tree {tag}: {sym}: {} registers, {} B local, {smem} B shared, {occ} resident a SM", r.num_regs, r.local_bytes));
                if occ < 1 {
                    anyhow::bail!("{sym} fits no CTA on a SM at {threads} threads and {smem} B");
                }
                Ok((cuda, f))
            })();
            match built {
                Ok((cuda, f)) => {
                    fns[k] = Some(f);
                    // a12ip's or a12ipg's module: both carry the load check
                    if (22..=27).contains(&k) && ip_module.is_none() {
                        ip_module = Some(cudas.len());
                    }
                    cudas.push(cuda);
                }
                Err(e) => self.lines.push(format!("4E tree {tag}: module FAILED, its arms are void: {e}")),
            }
        }
        if cudas.is_empty() {
            anyhow::bail!("4E tree: no tree-tiled module built; see the lines above");
        }
        let tile_f = cudas[0].func("p4t_tile_words").map_err(anyhow::Error::msg)?;
        // Copies are held only for the arms that read them (every tiled form
        // but a12ip, a12ipg and a12ipg32). a12ip reads the words in place; its load check builds
        // each projection's copy, compares every lane's words read in place
        // with it (p4t_ip_check), and frees it before the next one.
        let keep = want[..22].iter().any(|&w| w);
        let check_f = match ip_module {
            Some(i) => Some(cudas[i].func("p4t_ip_check").map_err(anyhow::Error::msg)?),
            None => None,
        };
        let mut held: Vec<CudaSlice<u32>> = Vec::new();
        let mut bytes = 0u64;
        let (mut checked, mut served_bytes) = (0usize, 0u64);
        let (swaps, head) = {
            let mut copy = |src: u64, int4: bool, d_out: u32, stride: u32, d_in: u32, served_tiled: bool| -> anyhow::Result<u64> {
                if !d_out.is_multiple_of(16) || src == 0 {
                    anyhow::bail!("4E tree tiled: a projection of {d_out} rows at {src:#x} cannot be tiled by 16 rows");
                }
                let n = pers::p4t_tiled_words(int4, d_out, d_in);
                let buf = cudas[0].zeros_u32(n as usize).map_err(anyhow::Error::msg)?;
                let (kind, nsub, served) = (u32::from(int4), d_in.div_ceil(48), u32::from(served_tiled));
                let cfg = LaunchConfig { grid_dim: (1024, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                let mut b = stream.launch_builder(&tile_f);
                b.arg(&src).arg(&buf).arg(&kind).arg(&stride).arg(&nsub).arg(&served).arg(&n);
                // SAFETY (open): seven arguments in `p4t_tile_words`' order
                // (`pers_4et.cu`); `buf` is a fresh buffer of `n` words, the
                // count the kernel writes. `src` is an address read back from
                // `self.layers_host`, recorded at install: it is live only if
                // the model this runtime was installed from still is, which
                // nothing here guarantees (BUG-15). Assumes `src` holds
                // `d_out` rows of `stride` words, as it did at install.
                unsafe { b.launch(cfg) }.map_err(|e| anyhow::anyhow!("p4t_tile_words: {e}"))?;
                if let Some(cf) = &check_f {
                    let bad = cudas[0].up_u32(&[0]).map_err(anyhow::Error::msg)?;
                    let mut b = stream.launch_builder(cf);
                    b.arg(&src).arg(&buf).arg(&kind).arg(&stride).arg(&d_in).arg(&d_out).arg(&served).arg(&bad);
                    // SAFETY (open): eight arguments in `p4t_ip_check`'s order
                    // (`pers_4et_a12.cu`); `buf` holds the `n` words just
                    // written, `bad` one word the kernel only adds to. `src`
                    // as in the launch above: live only while the model is
                    // (BUG-15).
                    unsafe { b.launch(cfg) }.map_err(|e| anyhow::anyhow!("p4t_ip_check: {e}"))?;
                    let nbad = cudas[0].down_u32(&bad).map_err(anyhow::Error::msg)?[0];
                    if nbad != 0 {
                        anyhow::bail!(
                            "4E tree a12ip: {nbad} lanes of a projection of {d_out} x {d_in} ({}) read in place differ from its tree-tiled copy; refused",
                            if int4 { "int4" } else { "Tetra" }
                        );
                    }
                    checked += 1;
                    served_bytes += 4 * pers::p4t_served_words(int4, d_out, stride, d_in);
                }
                if !keep {
                    // a12ip only: the copy was for the check; the pass reads src
                    cudas[0].sync().map_err(anyhow::Error::msg)?;
                    drop(buf);
                    return Ok(src);
                }
                let p = buf.device_ptr(&stream).0;
                bytes += 4 * n;
                held.push(buf);
                Ok(p)
            };
            let (hidden, d_ctx, inter) = (self.args_host.hidden, self.args_host.n_heads * 128, self.inter);
            let tet = |c: &mut dyn FnMut(u64, bool, u32, u32, u32, bool) -> anyhow::Result<u64>, t: &pers::SegTetraHost| {
                c(t.words, false, t.d_out, t.stride_u32, t.nblocks * 24 + t.tail_w, false)
            };
            let mut swaps = Vec::with_capacity(self.layers_host.len());
            for y in &self.layers_host {
                let q = &y.qkv;
                let down = if y.tetra_down { tet(&mut copy, &y.mlp.down)? } else { copy(y.mlp.q4w, true, hidden, inter / 8, inter, true)? };
                swaps.push([
                    tet(&mut copy, &q.q)?,
                    tet(&mut copy, &q.k)?,
                    copy(q.vw, true, q.v_dout, q.d_in / 8, q.d_in, true)?,
                    copy(y.ow, true, hidden, d_ctx / 8, d_ctx, true)?,
                    tet(&mut copy, &y.mlp.gate)?,
                    tet(&mut copy, &y.mlp.up)?,
                    down,
                ]);
            }
            let head = copy(self.args_host.head[0], true, self.args_host.vocab, hidden / 8, hidden, false)?;
            (swaps, head)
        };
        cudas[0].sync().map_err(anyhow::Error::msg)?;
        self.lines.push(format!(
            "4E tree tiled: {} tree-tiled copies (q, k, v, o, gate, up, down a layer, the head), {:.3} GB more on the card",
            held.len(),
            bytes as f64 / 1e9
        ));
        if check_f.is_some() {
            self.lines.push(format!(
                "4E tree a12ip: {checked} projections read in place, {:.3} GB, the words 4E and the eager path hold; every lane's \
                 words equal to the tree-tiled copy's on each (p4t_ip_check, each copy freed after its check); {} copies held \
                 for the other arms",
                served_bytes as f64 / 1e9,
                held.len()
            ));
        }
        let layers = stream.clone_htod(&vec![0u8; pers::P4E_LAYER_BYTES * self.layers_host.len()]).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let args = stream.clone_htod(&vec![0u8; pers::P4E_ARGS_BYTES]).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let [f, fs, f4, f5, f6, f7, f3f, f3p, f3t, f8, f9, f10, f11, f11i, fr8, fr16, ft384, ft256, ft448, ft320, ft384r16, f12, f12ip, f12ipg, f12ipg32, f12ipft, f12ipgft, f12ipg32ft] = fns;
        Ok(TreeTiled {
            _cudas: cudas,
            f,
            fs,
            f4,
            f5,
            fx: [f6, f7, f3f, f3p, f3t, f8, f9, f10, f11, f11i, fr8, fr16, ft384, ft256, ft448, ft320, ft384r16, f12, f12ip, f12ipg, f12ipg32, f12ipft, f12ipgft, f12ipg32ft],
            _held: held, swaps, head, layers, args, seen: Vec::new() })
    }

    /// The arms installed, in the order [`Self::tree`] indexes them.
    pub fn tree_arms(&self) -> Vec<TreeArm> {
        self.tree.as_ref().map_or_else(Vec::new, |t| t.arms.clone())
    }

    /// The tiled images brought to the caches the 4E images carry.
    pub(super) fn sync_tiled(&mut self) -> anyhow::Result<()> {
        let stream = self.cuda.stream().clone();
        let Some(t) = self.tree.as_mut().and_then(|v| v.tiled.as_mut()) else {
            anyhow::bail!("4E tree: no tiled arm installed");
        };
        if t.seen == self.kv_seen {
            return Ok(());
        }
        let mut img = Vec::with_capacity(pers::P4E_LAYER_BYTES * self.layers_host.len());
        for (h, w) in self.layers_host.iter().zip(&t.swaps) {
            let mut y = h.clone();
            y.qkv.q.words = w[0];
            y.qkv.k.words = w[1];
            y.qkv.vw = w[2];
            y.ow = w[3];
            y.mlp.gate.words = w[4];
            y.mlp.up.words = w[5];
            if y.tetra_down {
                y.mlp.down.words = w[6];
            } else {
                y.mlp.q4w = w[6];
            }
            img.extend(y.bytes());
        }
        stream.memcpy_htod(&img, &mut t.layers).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        let mut a = self.args_host.clone();
        a.ptrs[0] = t.layers.device_ptr(&stream).0;
        a.head[0] = t.head;
        stream.memcpy_htod(&a.bytes(), &mut t.args).map_err(|e| anyhow::anyhow!("H2D: {e}"))?;
        t.seen = self.kv_seen.clone();
        Ok(())
    }
}
