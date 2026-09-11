# Audit of the deliverable — 2026-09-11

Scope: commits `6937458..b463f4c` (13 commits, 33 files) and the deliverable as stated by the operator: *a standalone, clean `.llvq` package, ready for the bench*. Method: seven independent reading lenses (architecture of the served door, correctness of the batched prefill, deliverable completeness, missed optimizations, test discipline and hard rules, documentation drift, code quality), each finding then put before two adversarial verifiers with distinct angles (*is it real?* / *does the fix hold and is the severity honest?*), then one synthesis. 88 agents, 925 file reads, 54 minutes. Nothing was edited or launched.

Findings: 40 verified — **35 confirmed** (both verifiers), 3 contested (judged below), 2 refuted and dropped.

Every claim in the *Blocking* section was re-read by hand after the audit; all held.

## Verdict

Not bench-ready today. The code side of the served door is in place (LLVQ_CONFIG → load_resolved, the served mmlu arm, a prefill path that admits the 3,096-token census prompt), but the package cannot reach the card it is documented to run on: ops/run.py:747 never uploads configs/, so the next publish dies at ops/Dockerfile.cuda:153 and the image on the Hub (commit 0c2d798) predates both the served arm and the config file. The single most important gap sits one step past that one-line fix: there is nothing to score the kernel against — no stamped prereg (three operator questions plus the F1d arbitration are open, the BROUILLON is stale and unstamped) and no dense dump of the mixed file, so a census launched today would be paired against the restored dump of a different file (configs/README.md:14 attributes that 56.95 to the served object without a label) and every pp of difference would be read as "the kernel" while carrying three confounds.

## Blocking — before any money is spent on the census

### 1. configs/ is outside the publish perimeter: the Space image cannot be built and the served door does not exist on the card  `[S]`

**Where.** ops/run.py:747 (UPLOAD_ALLOW) and ops/Dockerfile.cuda:153 (COPY --from=build /src/configs/)

**Why.** upload_folder (run.py:876-882) sends only UPLOAD_ALLOW; the build stage's `COPY . .` never receives configs/, so the runtime COPY has no source and the build fails after the ~12 min compile. The image at hf.co/spaces/Pier-Jean/llvq-runner-cuda is at 0c2d798 (46 min before b463f4c), so it has neither the served mmlu arm nor the file; `LLVQ_CONFIG=/usr/local/share/llvq/configs/qwen3-4b-tetra-q5.json mmlu …` on a billed L40S fails at served.rs:116 read_to_string. Same two-lists trap already journaled for fetch-qtip.sh (run.py:739-745) and nullkbench (jobs.csv:90). No test guards it (served_config.rs:139 proves the file parses locally, not that it ships).

**Fix.** In ops/run.py:747 add `"configs/**"` to UPLOAD_ALLOW (fnmatch semantics identical to the existing `llvq-*/**`; this also puts configs/ under dirty_in_upload_perimeter, which is the intended behaviour since the Dockerfile ships those bytes). Optionally add `RUN test -f /usr/local/share/llvq/configs/qwen3-4b-tetra-q5.json` in the runtime stage so a missing config fails the build by name. Republish (`uv run ops/run.py publish --cuda`, Space build unbilled), record the COMMIT, then one ~$0.03 job that only opens the config through the served arm before any census.

### 2. The census has no stamped preregistration; four operator decisions are open and the draft is stale  `[S]`

**Where.** proofs/BROUILLON-preregistration-tetra-q5-servi.md:3-8 and §6/§7; docs/ROADMAP.md:297-307; proofs/ has no preregistration-f1e*-mmlu*.md.ots

**Why.** Hard rule 2. The only document covering an F1e census is the unstamped draft: §6 prediction table empty, §7 three operator questions open (format vs product comparison object; per-arm 1.35 vs paired SE; whether disk ≤ today's is a gate), plus ROADMAP 6.9's fourth gate, the F1d arbitration. The draft predates LLVQ_CONFIG, the served mmlu arm and MAX_PREFILL_ROWS=4096, still costs F1e at ~$8 against the measured slope, and covers steps already executed (encoding, tuerie, F1d), so stamping it as-is would post-date a prereg over past measurements — the exact Golay70 failure METHODE §1 records. The prefill prereg §0 already confesses one off-prereg job (6aa32150) on this workstream.

**Fix.** Do not stamp the BROUILLON; archive it as superseded. Operator answers the four questions in one line each (recommended: comparison object = dense arm on the SAME mixed file; SE = paired at constant file, 0.79–1.44 pp measured in docs/mesures/mmlupair-4b-8b-2026-08-13.txt, NOT the 2026-08-25 seed σ which never measured MMLU; disk gate = informative; F1d arbitration written). Then write a new `proofs/preregistration-f1e-census-<date>.md` scoped to the census only: the object (configs/qwen3-4b-tetra-q5.json + sha256 of qwen3-4b-tetra-q5.bin, 1,794,564,765 B), the exact command `LLVQ_CONFIG=/usr/local/share/llvq/configs/qwen3-4b-tetra-q5.json LLVQ_MMLU_DUMP=/out/<name>/mmlu-4b-tetra-q5-file-kernel.csv LLVQ_MMLU_ALLOC=flat LLVQ_DTYPE=f16 mmlu <sealed> cuda 40` with `oracle` first, the pairing plan (mmlupair against the same-file dense dump), a signed point prediction and interval for kernel−dense micro-accuracy and discordant count, the kill criterion (a paired delta beyond the interval is a kernel defect, not a tie-break), wall clock 1.49 h (*computed* on the measured slope) + 10 % attention (*estimated*) + the lm_head term if the optimization below is not applied, `--timeout 2h`, cost cap; commit, `ots stamp`, then announce the cost and ask for the go.

### 3. No comparison arm exists for the deliverable file: the only tetra-q5 dump is of a different file and mmlupair cannot tell  `[S]`

**Where.** docs/data/mmlu-dumps/mmlu-4b-tetra-q5.csv:2 (`# model=/out/tetra-4b-2026-09-06/qwen3-4b-tetra.bin … restored v_proj at int4 g128`); llvq-llm/src/bin/mmlupair.rs:328-395 (gates on plan/fingerprint/qhash/population only, label printed at :644)

**Why.** The 56.95 dump was scored on the PURE Tetra file of 2026-09-06 with LLVQ_RESTORE_Q4 at load; the shipped mixed file of 2026-09-09 was encoded sequentially with the int4 v_proj reinjected (calib.rs:787-810), so its 216 lattice matrices differ (ppl ×1.3203 → ×1.334) and it has never scored MMLU on any arm (tetra-q5-encodage-2026-09-09.txt: "Aucune MMLU"). A kernel dump of the mixed file pairs against the restored dump with no refusal and the printed pp stacks three confounds: file (sequential re-encode), embedding (q8 vs f16 — the dense loader has no EmbedMode), kernel vs candle arithmetic.

**Fix.** Make the census one job with two arms on the mixed file: `mmlu <file> cuda 40` without LLVQ_CONFIG (dense, ~26 min / $0.78 *computed* from f1e0:216) → `/out/<name>/mmlu-4b-tetra-q5-file-dense.csv`, then with LLVQ_CONFIG (~1.49 h / $2.69 *computed*) → `…-file-kernel.csv`; total ≈1.9 h / ≈$3.5 *computed* from the journal's slope, to announce before the go. The dense arm cannot take a q8 embedding, so the embedding confound stays: either write it beside the pp, or add a third arm through a measurement config with `"embed": "f16"` kept outside configs/ (+$2.69). Name the new dumps so they cannot be mistaken for the restored one. Signed prediction for the dense-mixed arm: 56.95 minus whatever the sequential re-encode costs.

### 4. The batched prefill's only on-card gate has never seen a short tail chunk, checks one argmax on a periodic prompt, has had no mutant, and loads the wrong object when combined with LLVQ_CONFIG  `[M]`

**Where.** llvq-llm/src/bin/fusedrun.rs:215-360 (LLVQ_PREFILL_TOKENS block, `load_with` at :225, second `FusedLayout::from_env()` at :240, argmax-only refusal at :352); llvq-llm/kernels/tv_tetra48_h.cu:182 (`r < n_rows ? r : 0u`) and :203 (`r < n_rows` write guard)

**Why.** n_rows ∈ {2,3} of tv_tetra48_rows_h has executed nowhere: host_tetra48.cpp:93 hard-codes R=4 with the fold done in the harness and N_X=32; the card gate ran at 200 and 800 tokens (both ≡0 mod 4); the 5-token PROMPT chunks as 4+1 and the 1-row chunk takes the one-row path (model.rs:754). About half the 2,280 census prompts end in such a chunk, so the paid MMLU run would be its first execution. The gate itself compares a single last-position full-vocabulary argmax on 'The capital of France is' repeated; max|Δlogit| is printed, never bounded; the mutants killed this session went through host shims, none through this gate (rule 10). And the block runs BEFORE the served block (:380) through `load_with`, so `LLVQ_CONFIG=… LLVQ_PREFILL_TOKENS=800` passes agrees() (vars unset) then loads planes14/f16/rot_share=0 — check_kinds refuses the Tetra record by name at best, and the printed provenance names env vars nobody set. GRAPH_AB/KV_AB/FUSE_AB/TIME_PHASES/KV_PREALLOC/GRAPH_DIAG are silently dropped on the served arm.

**Fix.** (1) In the PREFILL_TOKENS block, when `served.is_some()` load through `fused_cuda::load_resolved(&path, &device, dtype, cfg.layout, cfg.embed, cfg.rot_share, cfg.fuse, cfg.kv)` and print `cfg.provenance()`; delete the second from_env at :240 and print `f.layout.name()`; right after :179 refuse by name `LLVQ_GRAPH_AB`, `LLVQ_GRAPH_DIAG`, `LLVQ_KV_AB`, `LLVQ_FUSE_AB`, `LLVQ_TIME_PHASES`, `LLVQ_KV_PREALLOC` beside LLVQ_CONFIG, in the shape of mmlu.rs:537-541. (2) Keep every stepwise `last_row` (Vec<Tensor>, ~4 MB at n=800) and print per-position `max|Δh|` (RMS-normalised, not max/max: Qwen3 has massive-activation dims) against `h.narrow(1, pos, 1)` — print only, the bound is written into the prereg from that measurement, not asserted (the only measured ratio is 2.8e-3 on logits). (3) Run the gate on the served config at lengths ≡1,2,3 mod 4 (e.g. 201/202/203) and on a real 5-shot block (3,096 tokens, via `block()`): cents on the L40S, needs a go, cost announced, and a prereg line. (4) Once on card against a deliberate mutant (`xr = x` in tv_tetra48_rows_h or `r = 0` in rot_apply_rows) and journal the red before P1 is called green. Host-side, the shim cannot emulate warp_sum (host_shim.h:104 identity shuffle), so a Mac driver of the __global__ wrapper is a new 32-lane emulator, not a two-line addition; at minimum make N_X in host_tetra48.cpp not a multiple of 4 so the harness's own fold fires.

### 5. The kernel dump records layout only; embed/kv/rot_share/config path are absent from the CSV that outlives the job  `[S]`

**Where.** llvq-llm/src/bin/mmlu.rs:696-709 (header: DUMP_VERSION, # model=, # dtype=, # limit=, # alloc=); :540 (`eprintln!("{}", cfg.provenance())` to stderr); :560 (label `[LLVQ 2-bit, SERVED KERNEL, {layout}]`)

**Why.** The served-vs-dense pairing differs in embedding (q8 vs f16, fused.rs:131-134: 'MMLU within sigma' — exactly what a paired census resolves) and in KV/rotation choices; the dump says none of it, so six months later the ± pp is unlabelled (rule 8) and two served dumps at different embed/kv pair without refusal. mmlupair.rs:139-146 drops unknown `# k=v` lines, so the addition is backward compatible with all 19 dumps on disk. The dense arm has the same hole for kv_mode (env-driven at :503, printed only on the result line :833).

**Fix.** On the served arm write structured keys, not one line with the free-text note (a `\n` in `note` would split the header): `# config=<path>`, `# embed=`, `# rot_share=`, `# fuse=`, `# kv=`; on the dense arm `# config=none (dense reconstruction)` and `# kv={kv_mode.name()}`. Have mmlupair print them under `A =`/`B =`. Change mmlu.rs:540 to println! so provenance sits on the same stream as the three load_resolved lines. No DUMP_VERSION bump.

### 6. `bench --timeout` silently defaults to 30m while its docstring says it is mandatory; the served census needs 1.49 h  `[S]`

**Where.** ops/run.py:1534 (`default="30m"`) vs :1010-1011 (docstring 'mandatory and has no silent default'); :1213-1219 `_timeout_minutes` single-unit parser

**Why.** One forgotten flag = an L40S killed at 30 min (~$0.90 spent) and a dump mmlupair refuses (no `# end` trailer), then a rule-9 rerun. Pre-existing (4b9bc78 / 515c23c), but the census is the first job it would certainly kill. `"2h30m"` parses to 0.0 and prints 'at worst 0.00 $'.

**Fix.** `b.add_argument("--timeout", required=True, help="the real cost ceiling; the served-arm census (mmlu … 40 under LLVQ_CONFIG) measured 1.49 h on an L40S, give it 2h or more")`; make `_timeout_minutes` refuse an unparseable string instead of returning 0.0; document `--timeout 2h` beside the census command in configs/README.md. Do NOT add the proposed 'refuse under 100 min when the command contains mmlu' guard: the positional 40 is per-subject, the dense census runs in 26 min, and `mmlupair`/`LLVQ_MMLU_DUMP=` would match the grep.

## Before any number from the census is published

### 1. ×1.402 is a cross-job slope ratio labelled *mesuré*; the same ECARTS invokes rule 5 against dividing across those two jobs  `[S]`

**Where.** docs/mesures/f1e0-2026-09-10.txt:314 ('Nature : *mesuré*'), :323 (headline), :349-365 (6,22 ms 'Mesuré', 4,80 µs '*mesurés*', 30,9 µs, 62,2 Go/s); docs/data/jobs.csv:143; commit 308d9e1 title

**Why.** Rule 7: per length the quotient of medians is ×1.390 [1.388–1.393] at 200 and ×1.399 [1.394–1.401] at 800; 1.402 sits outside both envelopes. Rule 8 and METHODE.md:88-92: a cross-job reading is 'reported and not published' and carries *computed* (D1 precedent, ×1.091). 6.22 ms = (4364.0−3119.7)/200 is a difference across jobs 6aa32150 and 6aa3a6c1. Not yet in ETAT/ROADMAP/HISTORIQUE.

**Fix.** Rewrite the §0 quater headline as per-length quotients of medians with the conservative envelope, labelled '*computed*, cross-job (6aa32150 vs 6aa3a6c1), reported not published'; keep the slope ratio as a secondary *computed* line; relabel 6,22 ms / 4,80 µs / 30,9 µs / 62,2 Go/s *computed*. The ECARTS is .ots-stamped: do not edit its 'Mesuré : 6,22 ms', add a dated addendum beside it. Do not resurrect the deleted per-row-rotation arm for an LLVQ_PREFILL_AB mode; the label is the remedy.

### 2. configs/README.md pins 56.95 and 2.8138 on the served config with no label, and 56.95 was measured on a different file by restoration  `[S]`

**Where.** configs/README.md:14; llvq-llm/tests/served_config.rs:133-136 doc-comment repeats the attribution

**Why.** Rule 8. The number comes from q5-tetra-2026-09-06.txt T2: pure Tetra file qwen3-4b-tetra.bin, LLVQ_RESTORE_Q4=v_proj, dense candle arm, L40S. The shipped qwen3-4b-tetra-q5.bin has scored MMLU on neither arm; the kernel path has never scored MMLU at all (mmlu.rs:518-524). The first served-arm score will otherwise read as a kernel regression rather than a first measurement.

**Fix.** Row → `56.95 (*measured*, restoration arm LLVQ_RESTORE_Q4=v_proj on the PURE Tetra file qwen3-4b-tetra.bin, dense candle arm, L40S, docs/mesures/q5-tetra-2026-09-06.txt T2) — the shipped qwen3-4b-tetra-q5.bin has scored MMLU on neither arm yet` and `2.8138 (*computed*, whole model, q8 embedding at 8.5 b/param, q5-tetra-2026-09-06 accounting; rtbits confirmation 6.7 still open)`.

### 3. CLAUDE.md:13 and ETAT.md still say no served kernel reads Tetra and tv_q4_h never ran on a card  `[S]`

**Where.** CLAUDE.md:13; docs/ETAT.md:56-57, :469, :518-519, :536-537; llvq-llm/src/fused.rs:635 ('has never run on a GPU')

**Why.** Loaded first every session (ETAT is step 1 of 'where to resume'). jobs.csv:140-143 and f1e0-2026-09-10.txt:119, :137-139, :343-344 record the served Tetra kernel and tv_q4_h running in the model on the L40S (92.9 tok/s pure Tetra at FUSE=0 ROT_SHARE=0; 100.8 then 101.5 tok/s in 1.39 GB on the mixed object at ROT_SHARE=1; 256 tokens identical). ROADMAP 6.5 already says DONE, so the living docs contradict each other.

**Fix.** CLAUDE.md:13 → 'the served kernel `tv_tetra48_h` runs it in the model since 2026-09-10: the 4B served object (216 Tetra + 36 `v_proj` int4, q8 embedding) gives 101.5 tok/s in 1.39 GB at `ROT_SHARE=1`, 256 tokens identical to the dense arm (*measured*, `docs/mesures/f1e0-2026-09-10.txt`); no model above 14B is served, the `rot_apply` wall of `docs/format-noyau.md` §8 closes the path whatever the format, and the 32B point has never been encoded (~$62, *estimated*, `docs/ROADMAP.md`)'. ETAT.md:56-57 → both runs with their flags, two processes, no ratio (rule 5). ETAT.md:536-537: strike only the 'no served kernel reads Tetra … until step 6' clause, keep the rot_apply wall. ETAT.md:469/518-519 and fused.rs:635 → 'ran on the L40S on 2026-09-10 (job 6aa2e938)'. Old sentences to HISTORIQUE with their date.

### 4. ROADMAP 6.5/6.6/6.9 report the int4 launcher missing and the census blocked on it; §4 says prefill is on hold  `[S]`

**Where.** docs/ROADMAP.md:292-297, :315, :366-369; last edit 5c6d2f9 predates 495ea63

**Why.** An agent costing the census from ROADMAP reports the launcher as the blocker and re-estimates F1e at ~$8 against the computed 1.49 h / $2.69 per arm; the prefill work (jobs 6aa32150, 6aa3a6c1, $0.17) and LLVQ_CONFIG appear nowhere in the table.

**Fix.** 6.5: '`tv_q4_h` ran on the L40S on 2026-09-10 beside `tv_tetra48_h`: 36 launches a token, 256 tokens identical to the dense arm (*measured*, F1e §0 bis)'. 6.6: 'done 2026-09-10 (495ea63): `FusedInt4Proj` launched by `fused_cuda`, 252 launches counted (216 → 252 fix ce9b20c)'. 6.9: prefill 5.507 → 3.929 ms per prompt token, census 1.49 h / $2.69 per kernel arm '*computed* on the *measured* slope, +10 % attention *estimated*', effort column ~$8 → the pair cost; 'the census waits on the operator's go and the F1d arbitration'; note the '239 refused' figure is obsolete since MAX_PREFILL_ROWS=4096 (4adfe3c). Add 6.10 for configs/ + LLVQ_CONFIG (2026-09-11, $0). §4: 'prefill is served in chunks of 4 rows since 2026-09-11; batch M > 1 at decode stays on hold'. Rewrite :315 to match.

### 5. HISTORIQUE.md stops at 2026-09-07: F1d, F1e §0, the prefill, the served config and the 09-08 arbitration are unhistorised  `[S]`

**Where.** docs/HISTORIQUE.md:3 (header 'to 2026-09-06'), :408-465 (last entries); jobs.csv rows 125-143

**Why.** STYLE.md:21-23: a replaced fact goes to HISTORIQUE with its date; the ETAT/CLAUDE corrections above have nowhere to go. 19 jobs since 09-08 ($17.86 for the period, project total $138.50 from jobs.csv against $100.21 last stated at :417) and five journals are unrecorded.

**Fix.** Two ten-line entries (STYLE.md:87 budget; the file is already 465/400 lines), inserted after the 09-07 entry (line 418), header :3 → 'to 2026-09-11': '2026-09-08 to 09-09. The tuerie is red (R = 1.4693 vs gate 0.90), the tile was stealing the table's L1 (64 on sm_89, 32 on sm_120, bit-identical), the served object is encoded (1,794,564,765 B, ppl ×1.334), dclm-x32 $13.50, operator chooses the mixed object' and '2026-09-10 to 09-11. F1d on two cards ($3.48), F1e §0: three refusals at $0.08, pure Tetra 92.9 [92.4–93.0] / 1.36 GB, mixed object 100.8 [100.4–100.8] then 101.5 / 1.39 GB, oracle MATCH, counter 216→252; prefill 5.507 (off-prereg, declared in the ECARTS) → 3.929 ms/prompt token, five predictions right, MAX_PREFILL_ROWS=4096 for the 3,096-token prompt, LLVQ_CONFIG and configs/qwen3-4b-tetra-q5.json (fuse=0: no segmented kernel); F1e §0 cost $0.76'.

### 6. CLAUDE.md env table names two different objects 'served' and omits LLVQ_PREFILL_TOKENS and the served mmlu command  `[S]`

**Where.** CLAUDE.md:102 (LLVQ_CONFIG row) vs :104, :106, :107 ('q8 is the served config', 'served = 1'); :83 (mmlu command without LLVQ_CONFIG); :130 (measurement-mode list)

**Why.** The config file says fuse=0; the rows below say served=1 (that is Planes14 v1, ETAT §2). An operator exporting LLVQ_FUSE=1 beside LLVQ_CONFIG is refused by name (loud), but the map contradicts itself and LLVQ_PREFILL_TOKENS is filed nowhere although fusedrun.rs:200-215 says it belongs with TIME_PHASES/GRAPH_AB.

**Fix.** Rows: `LLVQ_FUSE` → 'served = `1` under `Planes14` v1, `0` under `Tetra48` (no segmented kernel; `configs/qwen3-4b-tetra-q5.json`)'; `LLVQ_ROT_SHARE` → 'served = `1` (both objects)'; `LLVQ_EMBED` → '`q8` in both served configs'. Add `LLVQ_PREFILL_TOKENS | integer ≥ 1 | fusedrun: N prompt tokens in one call vs N calls of one token, argmax must agree; measurement mode, outside the published protocol`. Line 130 → add `LLVQ_TIME_PHASES`, `LLVQ_TIME_EVENTS`, `LLVQ_PREFILL_TOKENS`. After :83 add `LLVQ_CONFIG=configs/qwen3-4b-tetra-q5.json cargo run --release -p llvq-llm --features cuda --bin mmlu -- <sealed> cuda 40   # scores THROUGH the served kernel; without LLVQ_CONFIG mmlu scores the dense reconstruction that produced every published bar`.

### 7. served.rs module doc attributes the 100.6 tok/s bar to FUSE=0 ROT_SHARE=0; the journals say the opposite  `[S]`

**Where.** llvq-llm/src/served.rs:11-13

**Why.** d1-fusion-servie-2026-08-24.txt:79-81 gives 100.6 at ROT_SHARE=1 FUSE=1; the runs at 0/0 were the F1e §0 Tetra runs (f1e0-2026-09-10.txt:29-31). A reader would 'correct' a published bar that is right.

**Fix.** Replace with: 'A run that forgot one measured something else and said nothing — which is exactly what happened to the F1e §0 Tetra runs of 2026-09-10, measured at `FUSE=0 ROT_SHARE=0` and set beside a 100.6 tok/s bar that had been measured at the served `ROT_SHARE=1 FUSE=1` on Planes14 (`docs/mesures/d1-fusion-servie-2026-08-24.txt`, `f1e0-2026-09-10.txt` §0).'

### 8. Journal memory comparisons are GB-on-card quotients, never whole-model b/param; section-level *mesuré* blankets computed numbers; the op ledger omits 144 casts a chunk  `[S]`

**Where.** docs/mesures/f1e0-2026-09-10.txt:22, :117, :133, :139 (memory); :104, :314 (Nature); :251, :272, :355, :358, :365 (derived numbers); :263-268 and :355 (504 ops)

**Why.** Rule 6 (METHODE.md:97-98) and rule 8. ÷5.81 is a same-denominator quotient (not the 5.51/4.50 mistake) but the b/param figure is absent; :133 compares the MIXED object's 1.39 GB to ETAT's 1.390 GB, which is the PURE file's figure (mixed computes to 1.415 GB, a 2 % accounting gap ETAT §5 sexies names). The per-row `to_dtype` at fused_cuda.rs:765 adds 144 launches a 4-row chunk that no ledger counts: 648 remaining ops, residual 24.1 µs not 30.9; the 4.80 µs constant and the regroup projection are unaffected.

**Fix.** Beside :22 write 'b/param whole model, embedding q8: 2.7645 (*computed*, q5-tetra-2026-09-06 accounting); on measured bytes 16/5.91 = 2.71'; beside :117 and :139 '2.8138 (*computed*, same accounting); on measured bytes 16/5.81 = 2.75; the 2 % gap is the host-byte vs b/param accounting'; fix :133 to compare the mixed object to 1.415 computed. Per-number labels at first appearance: ms and tok/s *measured*; slopes, intercepts, GB/s, µs/op, op counts, census hours and dollars *computed*; +10 % N² and regroup 4.84 ms *estimated*. Dated addendum: ledger 1,944 → 648 ops per chunk with the 144 casts, residual 24.1 µs.

### 9. No runbook ends in 'the served object is loaded and its tokens match dense' before $2.7–5.4 is spent; ETAT §2 does not know the served door exists  `[S]`

**Where.** configs/README.md:34-38 (one mmlu line, no dump/bucket/timeout/pairing); docs/ETAT.md:11-14 (§2 'served configuration v1 = planes14 … FUSE=1'); grep LLVQ_CONFIG docs/ → nothing

**Why.** The served door (LLVQ_CONFIG → load_resolved) has never run on a card (last job at 0c2d798 predates b463f4c); the 256-token identity gate exists only in the env-var bench mode; the sequence must be written or the operator re-spells five flags by hand (f1e0 §0 shows Tetra with the f16 default diverging at token 20).

**Fix.** Write `docs/runbook-bench-4b.md` (or a section in configs/README.md): `oracle` → `fusedrun` bench mode at the served flags without LLVQ_CONFIG (256 tokens identical; or add an `LLVQ_GATE=1` measurement mode that runs the fused+dense comparison with all five values from `Served`, dense arm loaded with cfg.kv too, refusing LLVQ_FUSE_AB) → `mmlu … 1` smoke through LLVQ_CONFIG → the two-arm census with `LLVQ_MMLU_DUMP`, `--bucket`, `--timeout 2h` → `hf buckets cp` → `mmlupair a b`. Do not promise LLVQ_DATASET_REV as a pin (corpus.rs:37: cannot be used); record the resolved sha. Add one ETAT §2 line naming the wave-3 served object via `configs/qwen3-4b-tetra-q5.json` with the f1e0 §0 bis numbers and 'MMLU not yet scored through the kernel', without replacing the v1 table (fiche-4b.md is authoritative on the published file).

### 10. The served log attributes its five choices to environment variables that were not the source  `[S]`

**Where.** llvq-llm/src/fused_cuda.rs:2579, :2584, :2597 ('(LLVQ_ROT_SHARE)', '(LLVQ_FUSE)', '(LLVQ_FUSED_LAYOUT)' inside load_resolved); mmlu.rs:540 provenance on stderr

**Why.** A journal copied from a served run's stdout carries three env-labelled lines and no provenance line; values are right (agrees() guarantees it) but the attribution is wrong, and the ambiguity is what served.rs was written against.

**Fix.** Give `load_resolved` a `source: &str` argument: the served arms pass `"LLVQ_CONFIG=<path>"`, `load_with` keeps the variable names (they tell an A/B reader which variable flipped). Print it in place of the parenthesised names.

## Missed optimizations, sized on the measured 4.80 µs/op

### 1. Served mmlu arm projects the q8 lm_head on every prompt row and reads one  `[S]` — **before census**

**Where.** llvq-llm/src/bin/mmlu.rs:747-753 (`model.logits(&input)` then `logits.i((0, last))`); model.rs:1668-1671; fused_cuda.rs:1107-1137 (one tv_q8_h launch per row, each streaming the 413 MB int8 table)

**Mechanism.** Narrow the hidden states to the last position before the head, as generate (model.rs:1841-1843) and the PREFILL_TOKENS mode (fusedrun.rs:250-256) already do; per-row tv_q8_h launches are independent, so the scored row is bit-identical.

**Sizing.** *estimated* from the *measured* 0.598 ms/row lm_head (phases-2026-08-07.txt:86): +15 % per prompt token on the 3.929 ms slope; over 1,382,608 census prompt tokens ≈ 827 s ≈ 14 min ≈ $0.41; 941 MB transient logits at 3,096 rows. The journal's 1.49 h / $2.69 projection was formed with a last-row head, so bin/mmlu as written overruns it by ~10-15 %.

**Risk.** None on the served arm. The loop is SHARED with the dense arm: gate the narrow on `served.is_some()` — `let row = if served.is_some() { let h = model.hidden(&input, &mut NoCapture)?; let l = h.dim(1)?; model.project_head(&h.narrow(1, l - 1, 1)?)?.i((0, 0))? } else { model.logits(&input, &mut NoCapture)?.i((0, last))? }` — because a 1-row GEMV can move a dense logit by an f16 ulp and every published bar was measured on the N-row GEMM (mmlu.rs:526-530). Also moves the bench onto the head path the gate already exercised (rows>1 head never ran under a gate).

### 2. regroup stitches rows the kernel already wrote contiguously — step (a): return chunks whole  `[M]` — **before census**

**Where.** llvq-llm/src/model.rs:774-775 (`(0..len).map(|i| y.narrow(0, i, 1))`), :933-937 (`Tensor::cat(rows, 0)` one copy per input arg in candle 0.9.2); rotplan.rs:501, :536 (per-chunk `got.len() != len` check)

**Mechanism.** `forward_rows` returns one `[len, d_out]` tensor per chunk for Tetra sites; `drive_rows_batched` takes `Vec<(T, usize)>` (or a `rows_of` accessor) and KEEPS the per-chunk count check — the total-only check is exactly what `an_apply_that_answers_the_wrong_count_is_refused` (rotplan.rs:715-740) forbids; regroup cats chunks. The 36 int4 sites still fan out per row, so copies go 1,008 → 360 (216 + 144), not 252.

**Sizing.** *estimated* on the *measured* 4.80 µs/op: 648 × 4.80 = −3.1 ms of the 15.6 ms 4-row chunk (−20 %), ≈ −0.78 ms per prompt token, ≈ −18 min on the census. Journal already names regroup as the next term (f1e0:364-369) but not this mechanism.

**Risk.** Host-only, no kernel text change, bit-identical (same values, same order); one-row path untouched (model.rs:754). Requires rewriting the two rotplan tests that pin the per-row contract (rotplan.rs:641, :691) and re-running the B4 gate. Only worth doing before the census if it lands before that gate run; otherwise after, under its own prereg.

### 3. regroup step (b): a per-site output buffer and y_off in the rows kernel, regroup becomes a reshape  `[L]` — **after census**

**Where.** llvq-llm/kernels/tv_tetra48_h.cu:137-153 (no y_off; store at :206 `y[r*d_out+row]`); fused_cuda.rs:1621-1626 (fresh alloc per launch); emb_q8.cu:76-83 (precedent x_off/y_off)

**Mechanism.** FusedRowsOp is a CustomOp1 and must return fresh storage, so the cited emb_q8 precedent does not transfer as written. Sound variants: `candle_core::InplaceOp2` (custom_op.rs:275 in 0.9.2) with the per-site `[rows, d_out]` buffer allocated once in group_forward as `self` and the rotated chunk as `rhs`, kernel writing `y[y_off + r*d_out + row]`; or the HeadOp shape (rotate the whole prompt in one rot_apply_rows launch, one CustomOp1 per site looping chunks internally), which moves the chunk walk out of drive_rows_batched.

**Sizing.** *estimated*: −4.84 ms per 4-row chunk (−31 %), −216 output allocs per chunk; census 1.49 h → ≈1.03 h / $1.86 (journal's own estimate, f1e0:367-369).

**Risk.** Changes the served translation unit (sha256 re-journaled), kernel signature ripples into host_tetra48.cpp and tetra48_matches_rust.rs; tv_q4_h y_off is a separate kernel change. Bit-identical in every variant. Needs a timestamped prereg and a go; the closed prefill prereg §4 explicitly deferred output batching.

### 4. int4 v_proj pays a per-row f16→f32 cast launch plus its matvec; a rows entry for tv_q4_h removes 3/4 of both  `[M]` — **after census**

**Where.** llvq-llm/src/fused_cuda.rs:765 (`x.to_dtype(F32)?.contiguous()?` per row); model.rs:796-802 (fallback fan-out); kernels/tv_q4_h.cu:19-21, :95 (f32 x, no row count; `h2f` already in the unit)

**Mechanism.** `tv_q4_rows_h(…, n_rows, d_out)` on the tv_tetra48_rows_h template (stage n_rows rows, per-row accumulators, same `__fmul_rn/__fadd_rn` dequant), taking f16 `x` widened by `h2f` in the staging loop (exact, bit-identical); a `FusedInt4RowsOp`; `batches_rows()` true for FusedInt4 when loaded; extend tests/host_tv_q4_h.cpp / proj_q4.rs with an R-row fixture. Shared: 4 × 2560 × 4 = 40,960 B < 49,152.

**Sizing.** *estimated* at 4.80 µs/op: 288 → 72 ops per chunk, −1.04 ms (−6.6 %); with f16 input in the one-row kernel too, 288 → 36, −1.21 ms (−7.8 %) plus −0.17 ms of 9.92 per decode token (1.7 %).

**Risk.** `prepare_rows` for FusedInt4 returns the caller's narrow at offset row0·d_in; today the cast is what yields the offset-0 buffer that passes the `start != 0` refusal (fused_cuda.rs:1540-1545) — the rows op must launch on `x.slice(start..end)` or copy once per chunk. Changing tv_q4_h itself to f16 input touches the one-row path `oracle` certifies: oracle first on every backend (rule 10), rewrite the header at fused_cuda.rs:744-753 and tv_q4_h.cu:19-21. Adjacent latent defect: under LLVQ_DTYPE=f32 the served batched int4 path bails on row 1 of every chunk (to_dtype clone keeps the offset) — served config is f16, worth its own line.

### 5. PREFILL_ROWS is capped at 4 by the served DECODE tile; the rows kernel can take its own tile (64) and 8 rows  `[M]` — **after census**

**Where.** llvq-cuda/src/tile.rs:93 (`PREFILL_ROWS = 4`), :86-92 (deferral noted); fused_cuda.rs:380-383 (one define string for both kernels); tv_tetra48_h.cu:170-174 (rows kernel reads TILE_BLOCKS); llvq_tetra48.cuh:181-198 (decode once per word, FMA per row)

**Mechanism.** Inject a second define (`TETRA48_TILE`) used only by tv_tetra48_rows_h for xs_stride/ntiles, set it to ONE constant 64 on every card (8 × 64 × 96 = 49,152 B, the same allowance; a per-card table with a 128 fallback would refuse to load on A100/compute_80 and contradicts tile.rs:27-37's no-policy rule), PREFILL_ROWS = 8, `prefill_shared_bytes(8, 64)`, rewrite the tile.rs:277-279 test, tetra48_matches_rust.rs:197 and host_tetra48.cpp:138 to 8. Decode tile and LLVQ_TILE_BLOCKS untouched.

**Sizing.** *estimated*, unmeasured: −24 % instructions per row from decode amortisation plus 180 fewer launches per 4 rows (108 Tetra + 72 rot) × 4.80 µs = −0.86 ms; together ≈ −2.5 to −2.9 ms per 4-row chunk (16–18 %), ≈ −0.65 ms per prompt token. The 'tile 64 optimum' (f1d-2026-09-10.txt) was measured on the ONE-row kernel and its L1 mechanism does not transfer (both configurations stage 49,152 B); the gain here is launches and decode amortisation only.

**Risk.** A served-constant change: operator go and prereg before the L40S measurement (rules 1, 2); no published decode number moves (one-row kernel keeps its tile); output bit-identical (every admissible tile is a multiple of 32, per-lane j sequence unchanged); the spill check at fused_cuda.rs:459-467 hard-stops if 8 accumulators push past the 64-register contract. Already a documented deferral in two comments, not a miss; ranked last because regroup is larger and cheaper.

## Quality and maintainability

### Served door: authority, purity, tests

- **agrees() reads std::env directly, compares raw strings, and has no test; LLVQ_FUSE="" vs "0" is refused as a contradiction** — llvq-llm/src/served.rs:85-95, :123-131; llvq-llm/tests/served_config.rs (no set_var, no 'contradicts')
  - fix: Split: keep `Served::of(ServedFile, PathBuf)` pure; add `pub fn check_env(f: &ServedFile, lookup: impl Fn(&str) -> Option<String>) -> Result<(), String>` called from read()/from_env() with `|v| std::env::var(v).ok()`, comparing THROUGH the parsers via a generic helper `fn agrees<T: PartialEq>(var, env: Option<&str>, file: &str, parse: fn(Option<&str>) -> Result<T, String>)` (five different parser types, so the tuple loop cannot do it as written); run it after `of` so an empty field is reported as the file's fault first. Tests in their OWN file `tests/served_env.rs` (every existing served_config test reads the live env and would race; cargo test threads share the process): agree, disagree naming both values, unset, `""` vs `"0"` accepted, five names pinned as the prefix of each parser's error. Note: the existing file already goes red if the developer's shell exports LLVQ_FUSED_LAYOUT=planes14 (4 of 6 fail) — measured.
- **Three env reads below the resolved door: LLVQ_KERNEL_DIR (kernel text), LLVQ_TILE_BLOCKS, LLVQ_NVRTC_ARCH (never printed)** — llvq-llm/src/fused.rs:832, :642; fused_cuda.rs:66; llvq-cuda/src/lib.rs:312; llvq-cuda/src/tile.rs:262; llvq-cuda/src/gpu.rs:36; fused_cuda.rs:2529-2531 (comment claiming nothing consults the env)
  - fix: In `Served::of` refuse `LLVQ_KERNEL_DIR`, `LLVQ_TILE_BLOCKS`, `LLVQ_NVRTC_ARCH` by name when LLVQ_CONFIG is set (refuse-by-name, not config fields: tile::request() and gpu::arch() are OnceLocks seeded from env only); append `llvq_cuda::gpu::arch()` to the `NVRTC source:` line at fused_cuda.rs:417 so every served log names its compile target (two jobs at compute_80 and compute_89 carry an identical sha256 line today); correct the load_resolved doc-comment to say what it guarantees (the five choices) and 'four choices' at :2526 → five.
- **Dead third door `pub fn load` reads FuseMode::from_env; eight positional args on load_resolved spread by hand at two call sites** — llvq-llm/src/fused_cuda.rs:2496-2499 (no callers since d168f40), :2532-2542; mmlu.rs:545-556; fusedrun.rs:383-392
  - fix: Delete `pub fn load` (update the five prose mentions: fused.rs:723, :2505; model.rs:1233, :1577; fusedrun.rs:13). Optionally `pub struct Choices { layout, embed, rot_share, fuse, kv }` on `Served` with `load_resolved(path, device, dtype, &Choices)` and `impl Served { fn load(&self, …) }`; if so, `Choices::from_env` must take `fuse` (and kv) from the caller — load_with exists so fusedrun's FUSE_AB can run both arms in one process, re-reading LLVQ_FUSE would silently collapse that A/B. Low priority: all eight arguments are distinct enums, so a swap or omission is a compile error today.
- **mmlu.rs served-arm refusal names LLVQ_RESTORE_F16 for a Q4 restoration** — llvq-llm/src/bin/mmlu.rs:537-541 (`restore.describe()` = types only; RestoreF16::from_env also reads LLVQ_RESTORE_Q4)
  - fix: Add `RestoreF16::var(&self) -> &'static str` returning "LLVQ_RESTORE_F16" or "LLVQ_RESTORE_Q4" by `prec` and use it in the message (`restore.prec()` as proposed yields 'int4 g128 (dequantized)', still the wrong variable). Pre-existing convention in sealed.rs:183, :189, :259, :531 — fix there too if consistency is wanted.
- **fusedrun served path is a third hand copy of the timing loop and prints no kv_store label** — llvq-llm/src/bin/fusedrun.rs:399-416 vs :841-861 and :934-950; no set_kv_store in :380-429
  - fix: Extract `fn timed_arm(label: &str, gen: impl FnMut() -> anyhow::Result<Vec<u32>>) -> anyhow::Result<(Vec<u32>, f64, f64, f64)>` (discard, ROUNDS_TIMED loop, round-0 comparison, rate_stats) and call it from the three arms; on the served path REFUSE a non-Cat `LLVQ_KV_PREALLOC` (and a set LLVQ_TIME_PHASES) beside LLVQ_CONFIG — do not wire it, CLAUDE.md:130 says it is never a served config — and print `kv_store=cat` so the served line carries the same provenance as the bench lines.

### Copy artefacts and stale references in this session's diff

- **Eaten line continuations in three user-facing error strings** — llvq-llm/src/bin/mmlu.rs:535 and :539 (14 spaces inside the literal); llvq-llm/src/fused_cuda.rs:579-580 (`\\` + newline + 17 spaces kept in the message)
  - fix: Rewrite with real `\` continuations; at :579 drop one backslash. Detector: `grep -rnP '\S {6,}\S' --include='*.rs'` and read the string-literal hits (the proposed `[^\S\n] {6,}` matches every indented line). Same class as the memory note 'Heredoc mange les continuations'. Cosmetic: mmlu.rs:535/539 fire only on misuse, fused_cuda.rs:579 only on a card below 49,152 B default shared.
- **Phantom `row_chunk`, stale line reference, uninterpolated braces in test messages** — llvq-llm/src/model.rs:661 (`row_chunk` does not exist; the function is `row_block`); fused_cuda.rs:2518 (`fusedrun.rs:173` — the KvMode::F16 is at :923; reference predates the session); tests/served_config.rs:61, :101 (`expect_err("an empty {name} …")` plain literal)
  - fix: model.rs:661 → `row_block` + `narrow`; fused_cuda.rs:2518 → 'the dense arm of bin/fusedrun' (function, not line number); served_config.rs → `unwrap_or_else(|_| panic!("an empty {name} must be refused"))`.
- **Kernel/driver comments describe the pre-batched budget and typical prompt length loosely** — llvq-llm/kernels/tv_tetra48_h.cu:105-108 ('model::MAX_ROWS refuses past 256', no pointer to MAX_PREFILL_ROWS); fused_cuda.rs:1569-1570 ('several hundred rows'); fusedrun.rs:208-210 ('14,042 5-shot questions of several hundred tokens' — the census is 2,280); rotplan.rs:283-284 (correct in its decode scope)
  - fix: tv_tetra48_h.cu → add 'on this path `model::MAX_PREFILL_ROWS` admits 4,096 (the longest 5-shot prompt is 3,096); the one-launch-a-row paths keep `MAX_ROWS = 256`'. fused_cuda.rs:1570 → 'up to 3,096 rows (mean 606, *measured*, f1e0:208)'. fusedrun.rs:208-210 → 2,280 census questions. rotplan.rs:284 → keep the decode scope, add 'on the prefill path `drive_rows_batched` calls `prepare` once a chunk of `PREFILL_ROWS`'.

### Test fidelity and coverage of the new paths

- **served_unit.rs does not compile the text the runtime hands NVRTC: no TETRA48_ROWS define, no emb_q8.cu although the shipped config says embed=q8** — llvq-llm/tests/served_unit.rs:94, :131-136, :170-177; fused_cuda.rs:61 (EMB_Q8_CU_EMBED cfg-gated), :380-384, :392-397; tv_tetra48_h.cu:97-99 (guarded default `#define TETRA48_ROWS 4u`)
  - fix: Move `EMB_Q8_CU_EMBED`/`load_emb_sources` next to `load_int4_sources` in fused.rs; append emb_q8.cu to served_names() after the Tetra chain when the shipped config's embed is Q8 (read it via `Served::read` as the shipped-file test does); factor `defines(tile_blocks, prefill_rows) -> String` in `llvq_cuda::tile` used by both fused_cuda.rs:380 and the test (byte-identical string or re-journal the sha). Replace the header default with `#ifndef TETRA48_ROWS / #error "the host must define TETRA48_ROWS" / #endif` (matvec.cu:18-19 pattern; host_tetra48.cpp never includes tv_tetra48_h.cu) — only then can 'exactly once' be asserted, the text holds the define twice today. Also tie tetra48_matches_rust.rs:197 `ROWS = 4` and host_tetra48.cpp:138 `<4u>` to `llvq_cuda::tile::PREFILL_ROWS`. Note host_embq8.cpp:65 does parse and run emb_q8.cu standalone, so the hole is the served composition, not the file.
- **forward_rows' Dense fallback and the launch_rot_rows input bound are reachable by no portable test** — llvq-llm/src/model.rs:797-803 (fallback narrows r.t row by row; unreachable from group_forward off-card); llvq-cuda/src/gpu.rs:849-858 (checks xout only)
  - fix: Drive `Proj::forward_rows` on `Proj::Dense` with `len > 1` directly (it is pub; `prepare_rows` on Dense returns key None, `check_key(None, None)` passes). Add `x_off + (n_rows-1)*row_stride + n <= xin.len()` in launch_rot_rows as defense in depth (RotRowsOp::cuda_fwd :1190-1200 already guarantees it by construction).
- **An all-int4 group (o_proj/down_proj via LLVQ_INT4_TYPES) crashes on any prompt longer than one token with a misleading d_in error** — llvq-llm/src/model.rs:967-975 (dense shortcut keyed on `rot_key().is_none()`, true for FusedInt4); :818; fused_cuda.rs:1535-1541; fused_cuda.rs:2643-2648 (comment claiming check_key refuses a mixed group is wrong for an all-None group)
  - fix: Not on the served file (q and k carry a key). Cheapest honest fix: pin the invariant 'int4 records only at sites that sit in a group with a rotated projection (today v_proj)' at load (fused_cuda.rs:2650-2656) and at write time (smoke.rs:809), refusing other sites by name, and say so at fused_cuda.rs:156 and configs/README.md. Changing the predicate to `all(|p| matches!(p, Proj::Dense(_)))` alone only swaps the error for a `row_cap(1) = 256` refusal, since `batches_rows()` is false for FusedInt4; the proposed stub test cannot exist (Proj is a closed enum, non-Dense arms cfg-gated).

### Ops tooling

- **check-cuda.sh rebuild trigger is stale relative to its Dockerfile; rustup volume seeds from the old image; dead `${@:+}`** — ops/check-cuda.sh:18-19, :24, :25; ops/Dockerfile.check:33-45
  - fix: Key the tag on the Dockerfile (`llvq-check:$(git hash-object ops/Dockerfile.check | cut -c1-12)`) or build unconditionally (layer cache makes a no-change build seconds); remove the `llvq-rustup` volume in the SAME commit as the rebuild (alone against the stale image it reinstates the per-run download); delete `${@:+}`. Do not add `flock` (does not exist on macOS) — cargo's build-dir lock already serializes concurrent runs. Low: today 1.95.0 is cached in the volume and rust-toolchain.toml selects it regardless.
- **No check that every `COPY --from=build /src/<dir>` in Dockerfile.cuda matches UPLOAD_ALLOW** — ops/run.py:747; ops/Dockerfile.cuda:153 (third occurrence of the two-lists trap: rust-toolchain.toml, fetch-qtip.sh, nullkbench)
  - fix: A small unit test in ops/ (or a `publish` preflight) that greps Dockerfile.cuda for `COPY --from=build /src/` sources not under target/ and asserts each fnmatches a pattern in UPLOAD_ALLOW.

## Contested findings, judged

- **The kernel-arm MMLU census has no preregistration, and the served arm is already wired to run it (tests-rules lens)** (llvq-llm/src/bin/mmlu.rs:531; proofs/)
  - Stands, merged into blocking item 2. The refuting verifier is right that the lens missed proofs/BROUILLON-preregistration-tetra-q5-servi.md and that a not-yet-launched run without a prereg is the repo's normal state; but the second verifier's point wins: the BROUILLON is unstamped, its §6 is empty, it predates LLVQ_CONFIG and load_resolved, and the operator's stated next act is this census — so the gap is 'complete and stamp before the go', which is exactly what blocks spending. Both verifiers agree the fix's σ source (2026-08-25 seed σ, which never measured MMLU) is wrong; the constant-file paired SE (0.79–1.44 pp) is the anchor.
- **Kernel and rotplan doc comments still describe the pre-batched row budget and per-row rotation (docs-drift lens)** (llvq-llm/kernels/tv_tetra48_h.cu:105-108; rotplan.rs:283-284; fused_cuda.rs:1569-1570)
  - Mostly refuted; kept as a low quality item. I side with the refuting verifier: all three sentences are true in their own scope (the one-row path still refuses past 256 via row_cap(1); rot_launches is documented as a per-decode-token count and the per-chunk 144 is stated correctly at rotplan.rs:452-456; 'several hundred' matches the 606-token census mean). The kernel header's omission of MAX_PREFILL_ROWS is worth a pointer, and the second verifier's adjacent catch (fusedrun.rs:208-210 says 14,042 questions) is the one real drift.
- **check-cuda.sh never rebuilds the image on a Dockerfile change, and the rustup volume shadows the pinned toolchain layer (code-quality lens)** (ops/check-cuda.sh:18-25)
  - Mechanism confirmed, consequences refuted, severity low. Both verifiers checked the volume and found 1.95.0 downloaded once and cached, and rust-toolchain.toml selects it whichever store holds it; cargo's build-dir lock serializes concurrent runs; the 'No such image' was transient daemon state. The fix as written would break the script (`flock` absent on macOS) and RUSTUP_HOME relocation reintroduces the waste it complains about. Kept in quality/ops with the corrected fix. Outside the deliverable: nothing served depends on this script.

## Refuted and dropped

- perf-missed: Tetra has no segmented kernel, so `fuse` is 0 in the served config — and turning it on later would silently undo the prefill batching
- code-quality: No committed job template for the served MMLU census; the served runs were launched from an uncommitted scratchpad script

## What this audit did NOT cover

Declared by the synthesizer from the lenses' footprint. These are the honest holes; none of them is known to be clean.

- No lens reported a 'covered' scope (the list came back empty), so the gaps below are inferred from the seven lens names and the findings' footprint, not from a declared perimeter.
- The sealed .llvq object itself: no lens audited llvq-artifact's reader/writer for the v5 format with Int4 records beside lattice ones, the sha256 identity of qwen3-4b-tetra-q5.bin against the F1e journal, `codebook_fingerprint` behaviour on the mixed file, or the archive tests that require ~/llvq-q4b.llvq. 'Standalone and clean' was judged from the loader side only.
- The encoder side of the deliverable: calib.rs's sequential reinjection of the int4 v_proj (the reason the mixed file's lattice codes differ from the pure file's, ppl ×1.334 vs ×1.3203, 'sixth dissociation') was cited as a confound but not audited for correctness or reproducibility (LLVQ_CALIB_SEED, LLVQ_THREADS, resume shards).
- Attention and KV in batched prefill: the correctness lens covered the matvec, rotation, regroup and driver chunking; rope, causal mask, KV append and `hidden_cached` position handling for rows>1 at chunk index ≥1 rest on the argmax gate alone and were not read.
- The MMLU harness proper: prompt construction, tokenizer fingerprint, qhash, LLVQ_MMLU_ALLOC=flat sampling, the dataset revision that 'cannot be pinned' (corpus.rs:37), and the `# end` trailer logic were not audited beyond mmlupair's gating list.
- `oracle` on every backend (rule 10) for the served CUDA path under LLVQ_CONFIG: bin/oracle was not read; whether it can be entered through the served door is unknown.
- Device-side memory safety of the new kernels (OOB writes when `y` is allocated n_rows*d_out and the guard is wrong, `unsafe` in gpu.rs launch_rot_rows/launch_tetra48_rows_h) was reasoned about by reading only; no card was available to any lens and nothing ran under compute-sanitizer.
- Whether the branch is green: no lens ran `cargo test --release -- --include-ignored`, `cargo clippy --all-targets` (zero-warnings rule), or the CUDARC_CUDA_VERSION cross-check; served_config.rs was found environment-dependent by one verifier (4 of 6 fail with LLVQ_FUSED_LAYOUT exported), suggesting the suite's state on a developer shell is not established.
- ops/Dockerfile.cuda beyond the configs COPY: the bins-in-both-lists trap for any binary added this session, nvcc probe, image size, and ops/run.py's bucket mount, monitoring and jobs.csv accounting were not audited.
- The one-row decode path's bit-identity after the session's model.rs/rotplan.rs changes is asserted from the F1e journal (P5, 256 tokens identical at 0c2d798), not re-verified; commits after 0c2d798 (b463f4c, 4adfe3c) have not run on a card.
- Publication surfaces: README.md, docs/fiche-4b.md, the datasheet and paper/ were not checked for drift against the new served object; only CLAUDE.md, ETAT, ROADMAP, HISTORIQUE and configs/README.md were.
- Security and provenance: the QTIP GPL fetch path (LLVQ_QTIP_DIR), dependency pinning, and whether LLVQ_KERNEL_DIR could be set inside the image by a Space secret were not examined.
- Metal backend parity: the Mac/Metal path (llvq-metal, mmlu metal) received no attention; the prefill rows kernel has no Metal counterpart and no lens asked whether the served config is refused or misread there.
