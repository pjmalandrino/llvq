# F1b — spécification et ses trois réfutations, 2026-09-04
> Produit par une reconnaissance à huit agents lancée le 2026-09-04, AVANT toute
> ligne d'encodeur. Quatre lectures indépendantes du dépôt, une synthèse, trois
> relectures adverses. Aucun fichier du dépôt n'a été modifié par ce travail.
>
> 🚨 Ce document n'est PAS une décision. Il contient une spécification proposée et
> ce que trois relectures lui reprochent. La porte de F1b n'est pas tranchée, le
> plafond de vague 2 n'est pas posé, et rien n'est implémenté.
>
> Sa valeur immédiate : c'est la lentille « débit et comparabilité » qui a trouvé
> l'erreur d'un facteur 4 dans le compte de branches de `f1a-comptes-2026-09-04.txt`.

## 1. Spécification proposée

```
= F1b — three-section E8 coset codebook, measured on 20,000 Gaussian blocks =

SCOPE. Benchmark only. Nothing in llvq-core, llvq-search, llvq-quant, llvq-artifact or
llvq-llm changes; no served path, no format, no existing test, and `codebook_fingerprint`
stays at 0x338f_420f_1186_6319. The trio reordering lives entirely in the new module. The
only edit to an existing file is one added line, `pub mod f1;`, in llvq-bench/src/lib.rs.

--------------------------------------------------------------------------------
0. WHAT IS BEING BUILT
--------------------------------------------------------------------------------
A 48-bit word `[state 8][s1 w1][s2 w2][s3 w3][gain 1]` with w1+w2+w3 = 39, decoded to a
point of sqrt(8)*Lambda_24 in the INTEGER embedding (Point = [i32;24], ||x||^2 = 16m,
llvq-core/src/leech.rs:33-36), then used as the SHAPE of a shape-gain quantizer exactly as
`LeechShapeGain` uses the ball: direction v_hat = y/||y||, gain = 1 bit rounding the block
norm. Scored by the shipped rule, `llvq-bench/src/lib.rs:186-196`
(`shape_gain_mse_shipped`), at rate 48/24 = 2.000 b/dim.

--------------------------------------------------------------------------------
1. FIXED ARITHMETIC — DO NOT RE-DERIVE, ASSERT
--------------------------------------------------------------------------------
Membership (llvq-core/src/leech.rs:76-99, reduction at :103-109): x_j = p + 2*c_j + 4*k_j
with p in {0,1} the shared parity, c a Golay codeword, and the third constraint collapsing
to sum(k) == p (mod 2). Coordinate i is bit i of the u32 codeword, LSB first
(llvq-core/src/golay.rs:19-20).

Trio (from `Golay::of_weight(8)`, first found, f1count.rs:101-106; verified again here in
Python against the repository's own construction clmul(m,0xC75)+parity bit 23):
  A = 0x00149f, B = 0x0f6840, C = 0xf08320;  all three in the code, weight 8,
  pairwise disjoint, A^B^C = 0xffffff.
Trio order (f1count.rs:118-122), `order[j]` = natural index at trio position j:
  [0,1,2,3,4,7,10,12,  6,11,13,14,16,17,18,19,  5,8,9,15,20,21,22,23]
`inv = argsort(order)`. Permute a codeword with
`perm(w) = sum_j ((w>>order[j])&1) << j`; permute a point with
`y_natural[order[j]] = y_trio[j]`. Apply the permutation to a LOCAL COPY of
`Golay::codewords()` and to the bench's own points only — never to anything llvq-core or
llvq-search owns. Every one of these was reproduced independently during recon:

  |V8| = |V16| = 64 (exhaustive coset reduction, f1count.rs:179-193)
  64 Golay states at cut 8 and at cut 16; x4 for (p, r) -> 256 Lambda_24 states
  each state8 has EXACTLY 2 prefix bytes, differing by 0xff   (128 prefix bytes in all,
      so the prefix byte determines the state8)
  each state16 has EXACTLY 2 suffix bytes, differing by 0xff  (128 suffix bytes)
  1024 distinct edges (state8, middle byte, state16); out-degree 16 per state8;
      (state8, middle) -> state16 is single-valued; 128 distinct middle bytes
  the 16-middle sets collapse to 8 distinct cosets of the [8,4,4] extended Hamming code,
      so every middle byte belongs to exactly one of 8 M-cosets
  closing count 64 * 2 * 16 * 2 = 4096 = |Golay|

Every Golay codeword meets each trio octad evenly (self-duality), which is why the two
patterns of an end section merge into one coset of 4*E8.

Per-section point sets, given the transmitted state sigma = (g8, p, r):
  S1(sigma) = { p*1 + 2c + 4k : c in prefixes[g8], k in Z^8, sum(k) = r (mod 2) }
              a coset of 4*E8, covolume 2^16
  S2(sigma) = { p*1 + 2b + 4k : b in mids[g8] (16 bytes), k in Z^8, both parities }
              a coset of 2*sqrt2*E8 (Construction A on the extended Hamming code),
              covolume 2^12; depends on sigma only through (p, M-coset)
  S3(sigma') = { p*1 + 2c + 4k : c in suffixes[g16], sum(k) = p XOR r' (mod 2) }
              a coset of 4*E8, covolume 2^16
with sigma' = (g16(g8,b), p, r' = r XOR delta) and delta = sum(k^(2)) mod 2.
Covolume closes: 2^16 * 2^12 * 2^16 = 2^44 per state, / 256 states = 2^36 = det(sqrt8 L24).

Write S1/S2/S3 in code by that literal (p, pattern, k-parity) definition. Do NOT introduce
an abstract E8/D8 type: the literal form is what `Leech::contains` checks.

--------------------------------------------------------------------------------
2. THE SHAPING REGION — TRUNCATION AND THE BIT SPLIT
--------------------------------------------------------------------------------
The s_i field is w_i bits, so it addresses EXACTLY 2^{w_i} points of S_i. Define

  R_i(coset) = the 2^{w_i} lowest-norm points of S_i, ties broken by ascending
               lexicographic order of the 8 coordinates in trio order (as i32).

A fixed geometric radius is NOT admissible: the number of coset points inside a ball
depends on the coset offset, so the word would stop being a bijection and the rate would
stop being 47 bits. The exact-2^w rule is what makes rate accounting honest by
construction.

Counting DP (one table, used for the radius, the tie count, the lex rank and self-check
SC2). For a pattern c, parity constraint, and p, count vectors y with y_j = p+2c_j+4k_j:
  cnt[j+1][n + y_j^2][par XOR (k_j & 1)] += cnt[j][n][par],  j = 0..7,
  k_j over -8..=8 (|y_j| <= 20 covers every norm below the cap), n <= T_MAX = 460.
Sum over the admissible patterns and the admissible final parity. Cumulate to get
N_S(t) = #{y in S : ||y||^2 <= t}. Then
  rho2 = min{ t : N_S(t) >= 2^w },  n_below = N_S(rho2 - 1),  n_tie = 2^w - n_below.
Membership of y: ||y||^2 < rho2 -> in; > rho2 -> out; == rho2 -> in iff lexrank(y) < n_tie,
where lexrank(y) counts points of S with norm^2 == rho2 that are lexicographically smaller.
lexrank is the same DP read backwards: walk j = 0..7 keeping the partial norm, and for each
admissible value v < y_j at position j add the EXACT-norm count of completions on
positions j+1..7 with residual norm rho2 - partial - v^2 and the required residual parity.
Precompute the exact-norm DP suffix table per coset once.

Measured truncation radii (recon, exact DP, not an estimate):
  split 12/15/12: rho2 in 88..96 for sections 1 and 3 (256 cosets each),
                  72..80 for section 2 (16 cosets)  -> radii 9.4..9.8 / 8.5..8.9
  split 13/13/13: medians 112 / 56 / 104 -> radii 10.6 / 7.5 / 10.2, badly anisotropic
The tie shell is NOT optional: with the boundary shell dropped entirely, section 1 keeps
only 0.887 of 2^12 and section 2 only 0.638 of 2^15 (median); with it fully included the
codebook is 3.13x too large, which is worth about +2.5 pp of retention. Either shortcut
invalidates the number.

Candidate splits (w1+w2+w3 = 39): (12,15,12), (12,16,11), (11,16,12), (11,17,11),
(13,13,13). Choose the headline split by MSE on the TRAIN split; report every candidate on
eval. 13/13/13 is the roadmap's literal wording and must appear as a labelled row.

--------------------------------------------------------------------------------
3. THE ENCODER
--------------------------------------------------------------------------------
Objective. The scoring rule is `e2 = ||x||^2 - 2*g_hat*t + g_hat^2` with g_hat fixed by
||x||, so minimising e2 over the codebook is exactly maximising t = <x, y>/||y||. The
angular objective does not decompose across sections (||y|| couples them), so use a scale
sweep, which does:

  for each scale s in the grid:
      per section, per coset, find y minimising ||x_i - s*y||^2 over S_i,
          keep it only if it is in R_i (otherwise that branch is infeasible);
      combine over the trellis (below) to get the total-cost minimiser y(s);
      evaluate the TRUE criterion t(y(s)) = <x,y(s)>/||y(s)||;
  answer = argmax over s of t(y(s)).

Every candidate is feasible by construction, so the measured MSE is an upper bound on the
true F1 MSE and the retention a lower bound: the bias is one-sided and pessimistic. Its
size is measured, not assumed, by self-check SC8.

Per-section nearest point (Conway-Sloane, the D8 repair). With
z = (x_i/s - p*1 - 2c)/4, minimise ||z - k||^2 over k in Z^8 with sum(k) = delta:
  k0_j = round(z_j); if sum(k0) == delta (mod 2), k = k0;
  else j* = argmax_j |z_j - k0_j| (lowest index on a tie), and k = k0 with
  k_{j*} += sign(z_{j*} - k0_{j*}), sign(0) = +1.
Then y = p*1 + 2c + 4k, cost = 16*s^2*||z-k||^2, and ||y||^2 = sum y_j^2.
Independent rounding is wrong half the time: 4*D8 is an index-2 sublattice of 4*Z^8, so
the parity repair is mandatory, not an optimisation.
Distinct subproblems per (block, scale): 2 p x 128 prefix bytes x 2 parities = 512 for
section 1, 512 for section 2 (both parities per middle byte), 512 for section 3 —
1536 D8 solves, shared across all 256 states.

Trellis (three stages, exhaustive, no pruning needed):
  c1(sigma)         = min over the 2 prefixes of cost1(p, c, r), infeasible -> +inf
  c2(p, b, delta)   = cost2, feasibility against R2(p, M-coset(b))
  c3(sigma')        = min over the 2 suffixes of cost3(p, c, p XOR r'), infeasible -> +inf
  best = min over sigma (256) x b in mids[g8] (16) x delta (2)  of  c1 + c2 + c3
       = 8192 evaluations per (block, scale).
Track the accumulated ||y||^2 alongside the cost; recover the inner product without
rebuilding y as  <x,y> = (||x||^2 + s^2*||y||^2 - cost) / (2*s).

Scale grid. Fit the range on TRAIN and assert the argmax is interior for >= 99.5% of train
blocks; 52 points linearly spaced on [0.12, 0.80] was verified adequate in recon (the
optimum sits near s ~ 0.35, next to the spherical beta* = 0.350 of g4_full.rs:10). A finer
grid can only help — the coarseness is another one-sided pessimistic bias, and SC8
measures it.

Determinism: one block is independent of every other; parallelise with `std::thread::scope`
and per-thread workspaces the way `precompute` does (llvq-bench/src/lib.rs:78-107). The
result must not depend on the thread count.

--------------------------------------------------------------------------------
4. THE ARMS (one process, one block list, one set of gain centroids)
--------------------------------------------------------------------------------
  A1  ball13-L5     BallSearcher::with_level_cap(5), shell cap 13 (the default),
                    `nearest_angular`. index_bits(13) = 48, +1 gain -> 49 bits,
                    rate 49/24 = 2.0416667. PLUMBING ANCHOR: must reproduce
                    MSE 0.0725 / 92.72% (docs/archive/face-au-4-bits.md:198).
  A2  ball12        same searcher with `set_shell_cap(12)` — BallSearcher::new()
                    defaults to 13, so this call is mandatory.
                    index_bits(12) = 47, +1 -> 48 bits, rate 2.000.
                    This is the in-harness stand-in for the paper's 92.14%; it does not
                    exist anywhere in the repository today.
  A3  ball12-sweep  the same ball-12 codebook driven by `nearest_scaled(s)` over the same
                    scale grid, best-by-angular. Encoder-gap control (SC8).
  A4  f1-<split>    the F1 arm, 8 + 39 = 47 index bits, +1 -> 48, rate 2.000.
                    One row per candidate split; the train-chosen one is the headline.
Report per arm: index bits, gain bits, rate, MSE to 6 decimals, retention to 2 decimals,
median cosine; for F1 arms also per-section rho2 range, the infeasibility rate, and the
share of blocks whose winner touches a truncation boundary. Report explicitly the delta
A4 - A2 in percentage points: that difference, measured on identical blocks with an
identical scoring rule, is the number that decides F1b. Never put A4 next to 92.14 alone.

--------------------------------------------------------------------------------
5. SELF-CHECKS (tests/f1_codebook.rs; SC1 and SC2 are mandatory)
--------------------------------------------------------------------------------
SC1  Every point produced by the F1 encoder, un-permuted with `inv`, satisfies
     `Leech::contains`; `Leech::norm2` is a multiple of 16; `Leech::shell_index` is
     Some(m) with m >= 2. Run on all 20,000 eval blocks, every F1 arm. Recon ran this on
     300 winners of a Python prototype: 0 failures out of 300.
SC2  Theta reproduction. Count, through the three-section construction (states x branches
     x per-section EXACT-norm DP), the points with ||y||^2 = 32 and ||y||^2 = 48 over the
     UNTRUNCATED S_i. They must equal `llvq_core::leech::THETA[0].1` = 196,560 and
     `THETA[1].1` = 16,773,120. This is the check that catches a silently wrong lattice:
     it exercises the permutation, the state machine, the parity bookkeeping and every
     coset offset at once, against constants the G1 suite already pins.
SC3  Golay counting: |V8| = |V16| = 64; 64 states at each cut; 2 prefixes and 2 suffixes
     per state, each pair differing by 0xff; 1024 edges; out-degree 16; (state8, middle)
     single-valued; 8 M-cosets; 64*2*16*2 = 4096. Reproduces F1a
     (docs/mesures/f1a-comptes-2026-09-04.txt).
SC4  Bijection / round trip: encode a winner to (state, s1, s2, s3), decode, get the same
     point; all four fields in range; on a sample of 10^6 random words, decode then
     re-encode to the same word, and no two distinct words decode to the same point.
SC5  Rate honesty: for every section coset, |R_i| == 2^{w_i} exactly, by DP
     (n_below + n_tie == 2^w and n_tie <= the boundary-shell count).
SC6  D8 repair optimality: on 10^5 random z, the single-flip rule equals brute force over
     the 2^8 up/down roundings. Debug-only, in the spirit of
     `even_repair_matches_dp_reference` (llvq-search/tests/g2b_generic.rs:172).
SC7  Plumbing anchor: arm A1 gives MSE 0.0725 and 92.72% at seed 0x6_1CAB.
SC8  Encoder gap: per block, t from A3 (sweep) vs t from A2 (exact `nearest_angular`) on
     the same ball-12 codebook; report the mean and max relative deficit and the retention
     difference. In recon this gap was 0.13 pp at 52 scale points and 0.42 pp at 30.
All heavy tests carry `#[cfg_attr(debug_assertions, ignore = "...")]`, matching
g4_full.rs:28, so `cargo test --release -p llvq-bench` runs them.

--------------------------------------------------------------------------------
6. WHAT MUST NOT HAPPEN
--------------------------------------------------------------------------------
- No use of `BlockDots13` (its `d` array is 12 shells and both `t()` and `spherical_err2`
  hard-code ||v||^2 = 16m; a three-section code has neither).
- No `Leech::random_point` anywhere (its sum(k) repair skews x[0], leech.rs:107-109).
- No call to `rate_spherical13`, `rate_shape_gain13`, `rate_shape_gain13_single`, and no
  log2-of-cardinality rate.
- No `shape_gain_mse*_projected` on either side.
- No beta from `betasweep` (it fits and scores on the same blocks, betasweep.rs:14-20).
- No normalisation of the source, per block or per row.
- No claim about speed, kernel cost, or real weights. F1b measures quality on an i.i.d.
  Gaussian source, one seed, no GPTQ; label it so.
```

## 2. Protocole de mesure proposé

```
SOURCE. `let mut rng = SplitMix64::new(0x6_1CAB); let train: Vec<[f64;24]> = (0..20_000).map(|_| gauss_block(&mut rng)).collect(); let eval: Vec<[f64;24]> = (0..20_000).map(|_| gauss_block(&mut rng)).collect();` — train first, eval second, from one stream, exactly as llvq-bench/src/bin/lcap.rs:82-85. `gauss_block` is llvq-bench/src/lib.rs:70-72 (24 raw N(0,1) draws, NO normalisation; the retention formula's -0.5*log2 assumes unit variance as a premise, lib.rs:212). SplitMix64 + Box-Muller burns exactly two u64 per sample (llvq-core/src/rng.rs:35-40): no other RNG call may be interleaved, and no arm may draw its own blocks.

GAIN. `let norms: Vec<f64> = train.iter().map(|x| x.iter().map(|v| v*v).sum::<f64>().sqrt()).collect(); let centroids = lloyd_max(&norms, 1, 60);` — 1 gain bit (the published configuration), fitted on TRAIN block norms, 60 iterations, quantile init (llvq-bench/src/lib.rs:131-156). One centroid vector serves every arm; refitting per arm would measure the fit rather than the codebook.

SCORING (identical on every arm, the shipped rule of llvq-bench/src/lib.rs:186-196 and llvq-quant/src/quantizer.rs:551-578):
  for each eval block x:  xx = ||x||^2
                          t   = <x, y>/||y||   for that arm's chosen point y
                          g   = centroids[nearest_centroid(&centroids, xx.sqrt())]   // rounds ||x||, NOT t
                          e2  = xx - 2.0*g*t + g*g
  mse = sum(e2) / (24 * 20_000)          // per WEIGHT, DIM = 24
  ret = retention_pct(mse, rate)         // = 100.0 * (-0.5*mse.log2()) / rate
Print mse with 6 decimals: recomputing retention from a rounded MSE gives 92.01 where the paper says 92.14 (docs/fiche-4b.md:290).

RATE — the same denominator convention as the existing measurement, and the fix for the 2026-08-04 failure. The rate is always PACKED WHOLE BITS over 24 dimensions, `(index_bits + gain_bits) as f64 / DIM as f64`, the convention of llvq-bench/src/bin/lcap.rs:113-114 and of ops/f1a_shaping.py:23-27:
  F1 arm      : index_bits = 8 + w1 + w2 + w3, asserted == 47; + 1 gain bit = 48; rate = 48/24 = 2.000000 exactly.
  ball-12 arm : index_bits = llvq_quant::quantizer::index_bits(12) = 47; + 1 = 48; rate = 2.000000.
  ball-13 arm : index_bits(13) = 48; + 1 = 49; rate = 49/24 = 2.0416667 (this arm is the plumbing anchor, not a comparator for F1).
Forbidden as denominators, each of them the exact shape of the retracted 92.24%: rate_shape_gain13(1) = 2.041560 (lib.rs:388-390), rate_shape_gain13_single(m,k) (lib.rs:424-427, still live in main.rs:139-148), rate_spherical13() = 1.999893, log2(codebook)/24, and 47/24 (the lattice field alone — worth +1.9 pp of flattery, docs/ETAT.md:125-127). A test asserts every printed rate equals an integer bit count divided by 24.

COMPARABILITY. Absolute retention is reported, but the decision quantity is the delta between the F1 arm and the ball-12 arm measured in the SAME process, on the SAME 20,000 eval blocks, with the SAME centroids and the SAME scoring rule. 92.14% is the paper's Table 8 value for norm(Lambda_24(12)) + 1 gain bit; this repository has never measured it (docs/fiche-4b.md:288 — llvq-bench/src/main.rs:109 loops over [0,2] and never runs k = 1, and `set_shell_cap` is never called in any retention bench). The ball-12 arm A2 is the first in-repo measurement of that configuration; the journal must state its value next to 92.14 and let the reader see the harness offset before reading the F1 number.

LABELS. Gaussian source, one seed, no GPTQ, quality only. *measured* for the retention rows, *computed* for the truncation radii and the counting checks. No speed claim: the ~11 arithmetic bits per section presuppose a rank decode in E8 that has never been costed, and E1v died exactly there (docs/ETAT.md:149, llvq-search/src/rankdec.rs:33-36).
```

## 3. Prédiction signée de l'auteur de la spec

```
POINT PREDICTION: 89.6% retention for the best F1 split at 2.000 b/dim, 20,000 eval blocks. RANGE: [88.8%, 90.3%]. DIRECTION AGAINST THE CLOSED FORM: ABOVE 89.10%, by roughly +0.3 to +0.8 pp. VERDICT PREDICTED: below the 90.3% kill — I give clearing the kill about 20% and clearing the 91.0% adoption under 5%. Sign of the bet: F1b comes in red, and the state coupling buys back only a fraction of the product-to-ball gap, not the 39.6% the kill needs.

BASIS. Not a guess: recon built a working Python prototype of exactly this codebook — trio order, 256-state machine, exact-2^w truncation from the counting DP, D8 parity repair, the scale-sweep trellis — and ran it on i.i.d. Gaussian blocks with the shape-gain scoring above. Control and arms in one framework, n = 2600 blocks:
    ball-12, same approximate encoder ....... MSE 0.077907   92.05%
    F1 12/15/12, exact 2^w .................. MSE 0.082903   89.81%
    F1 12/16/11, exact 2^w .................. MSE 0.082805   89.85%
    F1 13/13/13, exact 2^w .................. MSE 0.092281   85.95%  (n = 900)
The ball-12 control lands at 92.05% against the paper's 92.14% (MSE 0.077907 vs 0.077718), so the prototype's encoder gives away only 0.13 pp on a codebook where the exact answer is known — that calibrates both the framework and the sweep. The F1 minus ball-12 delta is -2.2 pp at n = 2600 and -2.7 pp at n = 900; taking 92.14 as the anchor puts F1 at 89.4 to 89.9. Sampling noise at n = 2600 is +/-0.25 pp on retention; at 20,000 blocks it falls to +/-0.09 pp, so the range above is dominated by my prototype's sample size and by residual encoder slack, not by the final measurement's noise.

WHY ABOVE 89.10 RATHER THAN AT OR BELOW IT. Three effects push up, one pushes down.
  (+) The 89.10% closed form is a lattice-quantizer argument: it applies the 8-ball/24-ball normalised-second-moment ratio (0.3666 dB) to the served MSE. But the served format is shape-gain — the point is normalised and a gain bit is coded — so the codebook's job is angular resolution on the sphere, not radial shaping. Concentration of measure means an i.i.d. Gaussian block's direction is near-balanced across the three octads with overwhelming probability, and along balanced directions the product region reaches norm sqrt(3)*9.5 = 16.5 against the ball's 14.3. The product loses only where the source rarely goes.
  (+) The state field couples the sections: 256 states x 2^39 labels is a union of 256 differently-offset products, not one product. That is the coupling F1a said only F1b could price.
  (+) F1 saturates the code space. The served ball holds N(12) = 1.11e14 = 2^46.66 points in a 47-bit field and wastes 0.34 bits; the F1 region holds exactly 2^47. That is 21% more points for the same word, worth about +0.16 pp.
  (-) Anisotropy. The three sections have different coset densities (2^16, 2^12, 2^16), so the roadmap's literal 13/13/13 gives region volumes 2^29, 2^25, 2^29 and truncation radii 10.6 / 7.5 / 10.2. The prototype measures that as 85.95%, four points below the equalised split — worse than the +0.2116 dB the closed form predicts for it. The bit budget, 12 coset-label bits (8 in the state, 4 in the middle label) out of 47, leaves 35 point bits, hence 35/3 per end section and 4 + 35/3 for the middle: integer splits 12/15/12 or 12/16/11, radii 9.4-9.8 / 8.5-8.9. Ship the equalised split; 13/13/13 as written would fail the gate by five points and for the wrong reason.
Net: the shape-gain framing and the saturation more than cancel the residual anisotropy of the equalised split, but they do not come close to the 0.8742 dB the region would need to reach the kill.
```

## 4. Risques que la spec déclare elle-même

- Comparing the F1 number against 92.14% without an in-process ball-12 control. 92.14% is the paper's Table 8 value on its unrounded MSE 0.077718 (docs/llvq-paper-notes.md:71); this repository has never measured it, because `set_shell_cap` is called in no retention bench and main.rs:109 never runs k = 1. Our harness gives MSE 0.0725 for ball-13 + 1 gain bit where the paper gives 0.078 for ball-12 — a 7% MSE gap between harnesses, three times the entire kill-to-adopt band. Without arm A2 the F1 number is our encoder against theirs and the gate verdict is meaningless.
- Getting the truncation wrong in either direction. Including each coset's full boundary shell makes the codebook 3.13x too large (measured, median over all cosets) and inflates retention by roughly +2.5 pp — enough to turn a red into a green. Dropping the boundary shell entirely keeps only 0.887 of 2^12 in the end sections and 0.638 of 2^15 in the middle, and costs about -1.4 pp. Only the exact 2^w rule with the lexicographic tie-break gives a 47-bit word that is a bijection and a rate that is 2.000 by construction.
- The rate denominator. Dividing a 48-bit word by rate_shape_gain13(1) = 2.041560 costs 1.9 pp; dividing by the 47 bits of the lattice field alone flatters by 1.9 pp. rate_shape_gain13_single (llvq-bench/src/lib.rs:424-427) is the exact function that produced the retracted 92.24% and is still printed by main.rs:139-148. The gate band is 0.7 pp wide; any of these errors decides F1b on its own.
- Shipping the roadmap's literal 13/13/13 split. The three sections have covolumes 2^16 / 2^12 / 2^16, so equal label widths give truncation radii 10.6 / 7.5 / 10.2 and the prototype measures 85.95% — five points under the kill, for a reason that has nothing to do with whether F1's idea works. The bit budget says 12/15/12 or 12/16/11.
- Scoring the two sides with different rules. shape_gain_mse_projected rounds <x, v_hat> and is a strict lower bound, about 2% of block MSE and 0.7 pp of retention better than the encoder that ships; every docs/ table written before 2026-08-01 quoted the bound. Likewise BallSearcher::new() defaults to shell cap 13, so a ball baseline that forgets set_shell_cap(12) is a 49-bit codebook wearing a 48-bit label.
- Encoder suboptimality folded into the geometry. The scale sweep is a feasible-candidate search, not an exact argmax over the region, and it can only understate F1. It must be bounded, not assumed small: SC8 measures it on ball-12, where nearest_angular gives the exact answer (0.13 pp at 52 scale points, 0.42 pp at 30 in the prototype). Without SC8 a red verdict cannot be distinguished from a weak encoder. Note also that the rejection-on-infeasibility rule bites the F1 arm harder than the ball arm, so the residual bias is against F1.
- A silently wrong lattice. A coordinate permutation applied to the points but not to the codeword table (or the reverse) leaves `contains` returning true for a DIFFERENT lattice, and reconstruction stays plausible; no existing test catches it. SC1 (Leech::contains on every un-permuted winner) and SC2 (196,560 and 16,773,120 reproduced through the three-section construction) are the two checks that do. Neither is optional.
- Running before the prereg is stamped. proofs/ contains no F1b preregistration and docs/METHODE.md §1 puts a timestamped prereg before the first millisecond, carrying the numeric adoption and kill criteria and a signed prediction. Hard rule 2 also forbids editing it afterwards: a deviation goes in an -ECARTS.md beside it.
- Touching the index map. The trio ordering changes codebook_fingerprint (pinned at 0x338f_420f_1186_6319 by llvq-artifact/tests/codebook_fingerprint.rs:121-131) and would invalidate the published Qwen3-4B artifact. docs/ROADMAP.md:237 gates format v2 on F1b being green, so F1b must stay entirely inside llvq-bench with its own private permuted copy of the codeword table.

## 5. Questions laissées ouvertes

- Which reconstruction does F1 actually intend? `[state][s1][s2][s3][gain]` decoded by 'three lookups and two additions' (docs/ROADMAP.md:113) has no step that computes ||y||, yet shape-gain needs y/||y||. If the served rule is instead g_hat * y with two scales (spherical shaping, no normalisation), the encoder becomes an exact nearest-point search at two fixed scales and the whole convex-hull caveat disappears — but the number is then comparable to the repo's 92.23% spherical row, not to 92.14%. This spec measures the normalised rule because the gate is written against 92.14; the operator should confirm before the prereg is stamped.
- Should the split be a fixed constant or fitted per block? Per-state field widths w1(s)+w2(s)+w3(s) = 39 are legal (the state is decoded first), cost no extra bits and no extra lookups, and would make the region a union of 256 differently-shaped products. Nothing in the repository measures how much that buys. It is not in this spec; it is the obvious first thing to try if F1b lands between 90.3 and 91.0.
- Is the first trio the right trio? f1count.rs takes the first one found among the 759 octads. 8 state bits is the minimum attainable, so no trio beats it on state count, but different trios may give different shaping regions. Untested; a cheap sweep over a handful of trios could be added to the same binary.
- Should A3 (the sweep encoder on ball-12) also be run with the trio permutation applied to the input? On i.i.d. Gaussian blocks a coordinate permutation is distribution-preserving, so it changes nothing here — but the same is emphatically NOT true of real weights, where which input channel lands on which coordinate stops being exchangeable. F1c has to decide whether the permutation is applied to the codeword table (format v2) or to the data.
- The 2026-08-04 post-mortem lives only in `git show 5d59ff2:CLAUDE.md` lines 985-1050 and in prose; docs/mesures/ has no G4 or lcap journal at all. Should the F1b journal also record the ball-12 and ball-13 rows as the first written Gaussian-retention journal in the repository, retiring the prose table?

## 6. Les trois réfutations

### Lentille « lattice » — verdict : sound_with_fixes

The lattice construction itself is correct — I could not break it. I rebuilt it independently in Python from the repository's own Golay construction (clmul(m, 0xC75) + parity bit 23) and confirmed every structural claim: A = 0x00149f, B = 0x0f6840, C = 0xf08320 are codewords of weight 8, pairwise disjoint, XOR = 0xffffff; the given trio order maps them to 0x0000ff / 0x00ff00 / 0xff0000; 128 prefix and 128 suffix bytes, 64 states at each cut, exactly 2 prefixes and 2 suffixes per state differing by 0xff, 1024 edges, out-degree 16, (state8, middle) single-valued, 8 M-cosets each a coset of a [8,4,4] code containing 0xff, and 64*2*16*2 = 4096. More to the point I rebuilt the code FROM the trellis (any prefix of g8 x any middle x any suffix of g16) and got back the permuted Golay code exactly, set-for-set — so no combination of free choices emits a non-codeword. The parity closure r + delta + (p^r^delta) = p is right, the reduction of constraint (iii) to sum(k) = p mod 2 holds (Golay is doubly even, so 2*wt(c) = 0 mod 8), S1 and S3 really are cosets of 4*E8 and S2 of 2*sqrt(2)*E8, the covolumes close at 2^36, and 5,000 random points built through the construction all pass a reimplementation of Leech::contains after un-permuting. The truncation radii I recomputed match the spec (88..96 at w=12, 72..80 at w=15, 0.887 / 0.638 / 3.13x). What is broken is the verification plan, not the geometry. SC2 — the check the spec calls the one that catches a silently wrong lattice — reproduces 196,560 under the trio ordering, under the identity ordering, and under a random shuffle, because the count depends only on the Golay weight enumerator and the parity split, never on the ordering. SC1 is equally blind when the permutation is applied consistently in both directions, since contains() is permutation-invariant. SC3 is the only real guard and the spec does not say it must be computed from the permuted table. Second, the lex-rank is specified per coset rather than per pattern, which would pool 2 (or 16) patterns per position and count hybrids — 254x too many on the one coset I measured — producing a rank that is not a bijection onto R_i. Third, the mandated scoring function is hard-wired to shells 2 and 3 and cannot score this codebook at all. With those fixed, plus the origin rule and an honest bracket on the encoder gap, the measurement would mean what it claims.

**BLOCKING**

- **Problème.** SC2 is blind to exactly the failure it is billed to catch. The spec says SC2 "exercises the permutation, the state machine, the parity bookkeeping and every coset offset at once" and calls it, with SC1, one of "the two checks" that catch a silently wrong lattice. I implemented SC2 as specified (states x branches x per-section exact-norm DP) and ran it under three coordinate orderings: trio -> 196,560; identity ordering -> 196,560; a random shuffle -> 196,560. The reason is structural: the per-coordinate distribution of y_j = p + 2c_j + 4k_j depends only on c_j, so the 24-coordinate norm count depends only on the code's WEIGHT enumerator, p, and the k-parity split — never on the coordinate ordering nor on the sections being octads. SC2 does test parity bookkeeping (I broke it deliberately: S3 parity = p^r with delta dropped gives 163,408, caught), and it tests that the enumerated (prefix, middle, suffix) triple set has the Golay weight enumerator. It tests nothing about the trio. SC1 does not close the hole either: if the codeword table is permuted with sigma and the point un-permuted with sigma^-1, `Leech::contains` (llvq-core/src/leech.rs:76-99) passes for ANY sigma, correct trio or not, because all three constraints are permutation-invariant. So the only real guard is SC3, and the spec never says SC3 must be COMPUTED from the permuted local table rather than asserted against F1a's numbers.

  **Correctif proposé.** Add a two-line direct assertion on the permutation, which I confirmed is decisive: perm(0x00149f) == 0x0000ff, perm(0x0f6840) == 0x00ff00, perm(0xf08320) == 0xff0000, and all three are in the permuted table. Under the inverse ordering none of the three section supports is a codeword, so this separates order from inv immediately (it is also what f1count.rs:118-122 already builds). State explicitly that SC3 derives |V8|, |V16|, prefix/suffix pairing, edges and out-degree from the permuted local copy, and that a result of 1024 states means the ordering is wrong. Rewrite SC2's billing as what it is: a parity-and-pattern-set check, not a permutation check.

- **Problème.** The lex-rank specification pools patterns and would build a set that is not in the lattice. The counting DP in section 2 is correctly per-pattern ("For a pattern c ... Sum over the admissible patterns"), but the lexrank paragraph is written pattern-agnostically — "for each admissible value v < y_j at position j add the EXACT-norm count of completions on positions j+1..7", and "Precompute the exact-norm DP suffix table per coset once". A coset is not a pattern: S1 and S3 pool 2 complementary patterns, S2 pools 16. A per-position pooled walk counts vectors that take coordinate j from one pattern and coordinate j+1 from another; the resulting mod-4 word is generally not a Golay codeword. I measured the magnitude on one section-1 coset at rho2 = 92: the true |S cap ball| is 2,401, the per-position-pooled count is 609,553 — 254x. SC5 does not catch this (it checks n_below + n_tie from the CARDINALITY DP, which is per-pattern and stays correct); the failure surfaces only as a rank collision in SC4 or a `contains` failure in SC1, i.e. as a mystery.

  **Correctif proposé.** State normatively that both the cardinality DP and the exact-norm suffix tables are keyed per PATTERN, not per coset, and that lexrank(y) = sum over every pattern c' in the coset (including patterns other than y's own) of #{points with pattern c', norm^2 == rho2, lexicographically smaller than y}. Within a pattern, the admissible v at position j are exactly v ≡ p + 2c'_j (mod 4). Add an assertion to SC5 that unrank(i) for i in 0..2^w always lands in S_i under a single pattern.

**SERIOUS**

- **Problème.** The mandated scoring function cannot score a three-section codebook. Section 0 says "Scored by the shipped rule, llvq-bench/src/lib.rs:186-196 (`shape_gain_mse_shipped`)". That function (actually at lib.rs:190-198) takes `&[BlockDots]`, and `BlockDots::t()` at lib.rs:49-51 is `max(d2/sqrt(N2), d3/sqrt(N3))` with N2 = 32.0, N3 = 48.0 hard-coded at lib.rs:32-33. It is the two-shell spherical struct: it can only express directions on shells m = 2 and m = 3. This is the identical defect the spec correctly forbids for `BlockDots13`, one shell worse — and F1 winners at rho2 in 88..96 per section are almost never on shell 2 or 3. Feeding F1 through it either produces a silently wrong t or forces someone to back-fill d2/d3 to fake it. This is the "loads, runs and gives numbers" trap in its purest form.

  **Correctif proposé.** Delete the mandate to call `shape_gain_mse_shipped` and keep only the inline formula already spelled out in the MEASUREMENT PROTOCOL. The precedent to copy is the anchor's own harness: lcap.rs:118-127 computes `xx - 2*g*t + g*g` inline with t from `project()` at lcap.rs:66-76, which derives t = dot/||v|| from the actual point. Cite that instead.

- **Problème.** SC8 does not bound the F1 encoder gap, though the spec asserts it does ("the bias is one-sided and pessimistic. Its size is measured, not assumed, by self-check SC8"). SC8 compares the scale sweep to `nearest_angular` on ball-12, where the per-scale solve is an EXACT Euclidean argmax over the whole codebook (llvq-search/src/generic.rs:681-694). The F1 encoder carries a second, structurally different loss that SC8 never touches: the rule "nearest point of S_i, keep only if it is in R_i, otherwise the branch is infeasible" discards a branch whenever the unconstrained nearest lands on the wrong side of the truncation, even when a feasible point of the same branch is barely worse. The spec itself concedes "the rejection-on-infeasibility rule bites the F1 arm harder than the ball arm". Since the decision quantity is A4 - A2 and the gate band (kill 90.3, adopt 91.0, docs/ROADMAP.md:118) is 0.7 pp wide, an unbounded one-sided bias on A4 alone can manufacture a red verdict from a weak encoder.

  **Correctif proposé.** Report the per-section per-block rejection rate (already half-requested as "the infeasibility rate") AND add a bracketing row: the same F1 arm with truncation disabled — not a valid code, but a strict upper bound on what the region can achieve, so the true F1 value is bracketed instead of only bounded below. Separately, disambiguate the trellis wording: `c1(sigma) = min over the 2 prefixes of cost1(...), infeasible -> +inf` must be the min over FEASIBLE prefixes, not the feasibility of the argmin; the wrong reading silently throws away a legal candidate whenever the better prefix is out of region.

- **Problème.** The origin is in the F1 codebook and there is no rule for it. y = 0 (p = 0, zero pattern, k = 0) is the lowest-norm point of its coset in all three sections, so the word [state_0][0][0][0] decodes to the zero point and t = <x,y>/||y|| is 0/0. The two sides are not symmetric here: `nearest_angular` explicitly excludes it ("The origin is *not* a candidate — a direction is always required", generic.rs:707-712), while `nearest_scaled` seeds the threshold with the origin at key 0 (generic.rs:681-694) and returns shell = 0, so arm A3 can produce a zero direction too. lcap guards it (`if nn > 0.0 { dot / nn.sqrt() } else { 0.0 }`, lcap.rs:74-75); the spec does not. SC1 as written ("shell_index is Some(m) with m >= 2") would fail on it rather than handle it, and the saturation argument "the F1 region holds exactly 2^47" silently counts a word that is not a direction.

  **Correctif proposé.** Pick a rule and apply it identically to A3 and A4: either exclude the origin word from the F1 arm (and state that the F1 direction alphabet is 2^47 - 1, still saturating relative to N(12) = 2^46.66) or guard t and report how many blocks hit it. Say which, and make SC1's shell_index >= 2 assertion consistent with the choice.

**MINOR**

- **Problème.** The 13/13/13 radii "medians 112 / 56 / 104" are internally inconsistent, and the inconsistency is a bug signature in the prototype that backs the signed prediction. Sections 1 and 3 are structurally identical truncation problems: 128 even-weight prefix bytes and 128 even-weight suffix bytes, both paired {q, q^0xff}, both with one fixed k-parity, 256 cosets each. I computed both: at w = 13 the rho2 multisets are IDENTICAL (range 96..112, median 112 for section 1 and for section 3). So section 3's median cannot be 104. It is either a typo or evidence that the Python prototype keys R3 differently from R1 — e.g. by (g16, p, r) instead of (g16, p, p XOR r'), which is precisely the parity-leak this review was asked to hunt. That prototype is the sole basis of the signed prediction (89.81 / 89.85 / 85.95) and of the "13/13/13 fails by five points" argument. Everything else I re-derived reproduced exactly: sect-1 rho2 88..96 at w = 12, sect-2 72..80 at w = 15, the 0.887 and 0.638 boundary-dropped shares, and 1.76 x 1.01 x 1.76 = 3.13x for the full-shell inflation.

  **Correctif proposé.** Recompute section 3's radii from the permuted table before stamping the prereg, and if the prototype really produced 104, find out why — the answer decides whether the prediction's numbers are trustworthy.

- **Problème.** R2 is keyed by (p, M-coset) but the state transition is keyed by g8, and the spec puts both facts in the same sentence without warning that the keys differ. I checked: each of the 8 M-cosets is shared by exactly 8 state8 values, and all 224 within-coset pairs have DIFFERENT (state8, middle) -> state16 maps. Caching the transition table by M-coset, the natural thing to do once R2 is cached that way, silently sends section 3 to the wrong suffix set — which still yields a Golay codeword prefix|middle|suffix only by accident, so `contains` would fail late and confusingly.

  **Correctif proposé.** One sentence: R2 and its truncation are shared across the 8 states of an M-coset; the g16 transition is not, and must stay indexed by (g8, b).

- **Problème.** Citation drift in two places (everything else I spot-checked was exact). `shape_gain_mse_shipped` is at llvq-bench/src/lib.rs:190-198, not 186-196; `precompute` starts at lib.rs:76, not 78. Confirmed correct as cited: face-au-4-bits.md:198 (the L max 5 row, 0.0725 / 92.72 %, and MAX_LEVELS_ANY = 5 at generic.rs:56 so with_level_cap(5) is the full ball), lcap.rs seed 0x6_1CAB with TRAIN = EVAL = 20_000 and GAIN_BITS = 1 (lcap.rs:45-48, 81-83), g4_full.rs:10 and :28, g2b_generic.rs:172, llvq-paper-notes.md:71, main.rs:109 and 139-148, codebook_fingerprint.rs:120-131, ETAT.md:125-127 and :149, ROADMAP.md:118 (adopt 91.0 / kill 90.3) and :237. Also confirmed: proofs/ contains no F1b preregistration (the only f1 file is preregistration-f1-cublasf16-2026-08-18.md, unrelated), so hard rule 2 is currently unmet.

  **Correctif proposé.** Fix the two line ranges and stamp the prereg before the first measurement.

### Lentille « rate » — verdict : sound_with_fixes

The rate and denominator core survives attack. 48 bits over 24 dimensions is correct and matches the anchor it is compared to: the paper's own Table 8 states the ball-12 + 1-gain-bit rate as 1.95833 + 0.04167 = 48/24 (docs/llvq-paper-notes.md:71), the registered gate says "48 bits packed" (docs/ROADMAP.md:118), and ops/f1a_shaping.py:23-27 plus docs/ETAT.md:125-128 already name the 47/24 trap. I verified index_bits(12) = 47 against llvq-quant/tests/g5_shapegain.rs:157 and against N(12) = 111,043,117,458,000 computed from the theta series (2^46.66, inside 47 bits). MSE scaling, the beta/scale fitting, the gain-bit accounting, the scoring rule and the Gaussian stream all match between the F1 arm and the ball-12 arm: t = <x,y>/||y|| is scale-invariant in the integer embedding on both sides, e2 = ||x||^2 - 2*g*t + g^2 with g rounding ||x|| is what LeechShapeGain::quantize actually does (llvq-quant/src/quantizer.rs:552-576), one centroid set fitted on train norms serves every arm, the stream is SplitMix64(0x6_1CAB) train-then-eval exactly as llvq-bench/src/bin/lcap.rs:81-85, and nearest_by honours shell_cap for both nearest_angular and nearest_scaled (llvq-search/src/generic.rs:730) so arm A2 really is a 47-bit codebook. What does not survive is the comparability reasoning around the number and one structural count: the spec contradicts the F1a journal on the middle-section branch count in a way that makes SC3 unpassable as written, it never says whether the gate is adjudicated on the absolute retention or on the A4-A2 delta, it misdiagnoses a codebook-size difference as a harness offset, and SC8 bounds only half of the encoder bias it advertises. None of these makes the 48/24 denominator wrong; all of them can change the verdict read off the number.

**BLOCKING**

- **Problème.** SC3 asserts "1024 edges; out-degree 16 per state8" and "closing count 64 * 2 * 16 * 2 = 4096", and claims this "Reproduces F1a". It does not. docs/mesures/f1a-comptes-2026-09-04.txt:85-87 records the middle section as "1 024 aretes distinctes sur 256 etats = 4 BRANCHES PAR ETAT", with the closing count written as "512 x 4 x 2 = 4096" (lines 96-98) and the label decomposed as "2 bits codes (la branche) et ~11 bits" (line 105). The spec's 16 is the correct number: with k_passe(8) = 1 and k_futur(16) = 1 the section-2 edge count is 2^(12-1-1) = 2^10 = 1024 over the 64 Golay states at cut 8, so out-degree 16; the journal divided a Golay-level edge count by the Lambda_24-level state count (256 = 64 x 4 for (p,r)) and got 4. The spec's own covolume closure only works at 16: S2 = 2^16/16 = 2^12, and 2^16 * 2^12 * 2^16 / 256 = 2^36 = det(sqrt8 L24); at 4 middles S2 would be 2^14 and the product would be 2^38, i.e. the 47-bit word would not be a bijection onto the region. So an implementer who "reproduces F1a" literally builds a code whose rate accounting does not close, and one who follows the spec has a mandatory self-check that fails against the artefact it cites.

  **Correctif proposé.** Resolve the count before writing code, and resolve it in the spec's favour. State that section 2 has out-degree 16 per Golay state8 (4 label bits), that the journal's "4 branches par etat" divides 1024 Golay edges by 256 Lambda_24 states, and that the journal's 8.0 KiB table figure is unaffected because 256 x 4 = 64 x 16 = 1024. Hard rule 2 forbids editing the dated journal, so record the correction in docs/mesures/f1a-comptes-2026-09-04-ECARTS.txt beside it and have SC3 assert 16 while citing that note. Also correct the derived sentence "12 coset-label bits (8 in the state, 4 in the middle label)" to say the 4 comes from out-degree 16, not from the journal's "2 bits codes".

**SERIOUS**

- **Problème.** The decision rule is ambiguous, and the two candidate rules can disagree by more than the gate band. Section 4 says "the delta A4 - A2 in percentage points ... is the number that decides F1b", but the registered gate in docs/ROADMAP.md:118 is absolute: adoption at retention >= 91.0%, kill at < 90.3%. The spec never says whether the prereg registers the absolute A4 figure against 90.3/91.0, or 92.14 + (A4 - A2). The band is 0.7 pp wide; any harness offset larger than that flips the verdict between the two rules, and choosing after seeing A2 is exactly the post-hoc move docs/METHODE.md forbids.

  **Correctif proposé.** Pick one rule in the prereg, in writing, before the first run. Recommended: the gate stays absolute on A4 at 2.000 b/dim, because that is what ROADMAP.md:118 registers and what the 89.10% closed form of ETAT.md:120-122 is stated against; A2 is then a plumbing control with its own pre-declared acceptance window, and a rebased delta verdict may only be quoted as a secondary, explicitly labelled figure.

- **Problème.** The stated risk "Our harness gives MSE 0.0725 for ball-13 + 1 gain bit where the paper gives 0.078 for ball-12 - a 7% MSE gap between harnesses, three times the entire kill-to-adopt band" misdiagnoses a codebook-size difference as a harness difference. 0.0725 (docs/archive/face-au-4-bits.md:198, docs/HISTORIQUE.md:41) is a 49-bit measurement over N(13) = 280,974,212,784,720 points; 0.077718 is a 48-bit measurement over N(12) = 111,043,117,458,000 - 2.53x fewer points, 1.34 bits of resolution. Splitting 0.0725 into an angular part that scales with the direction count and a gain-quantization floor that does not, ball-12 in this harness lands near 0.0778, i.e. about 92.0-92.1%, within a couple of tenths of 92.14. There is no 7% harness gap to explain. Left as written, the sentence pre-authorises the journal to read any A2 result as "our harness differs" and to rebase the gate on it.

  **Correctif proposé.** Replace the claim with the arithmetic: state that the 0.0725-vs-0.0778 distance is accounted for by one charged bit plus 2.53x the codebook, and register a numeric acceptance window for A2 in the prereg (e.g. 91.8% to 92.4%). Declare in advance that A2 outside that window is a plumbing failure to debug, not evidence that 92.14 is inapplicable.

- **Problème.** SC8 does not bound the bias section 3 claims it bounds. Section 3 says "the bias is one-sided and pessimistic. Its size is measured, not assumed, by self-check SC8." But the F1 encoder carries two distinct losses: (a) the finite scale grid, and (b) branch rejection on infeasibility ("keep it only if it is in R_i, otherwise that branch is infeasible"). A3 measures only (a): nearest_scaled (llvq-search/src/generic.rs:681-694) searches the whole capped ball and rejects nothing, so it never pays (b). The spec concedes elsewhere that "the rejection-on-infeasibility rule bites the F1 arm harder than the ball arm", which contradicts the claim that SC8 sizes the bias. A red verdict therefore still cannot be separated from a weak encoder.

  **Correctif proposé.** Add an F1-side control that prices (b) - e.g. re-run the headline split with R_i relaxed to the full boundary shell (a strict superset, rate-dishonest but a valid upper bound on the encoder loss) and report how many eval blocks change winner plus the retention difference - or, cheaper, per section fall back to the best point of R_i by rank instead of dropping the branch, and report the residual. At minimum restate the claim honestly: SC8 measures the scale-grid component only, and name the rejection component as unmeasured.

- **Problème.** Section 3 defines no behaviour when every trellis branch is infeasible for a block, at a given scale or at every scale. At small s the per-section nearest points have large norm and fall outside R_i, so "infeasible -> +inf" can make c1, c2 or c3 uniformly infinite and leave no candidate. The MSE denominator is fixed at 24 * 20,000, so a block with no answer either silently contributes nothing (deflating MSE, flattering F1) or contributes a NaN. The spec reports "the infeasibility rate" but never says what the encoder does when it is total.

  **Correctif proposé.** Specify and test the fallback: if no branch is feasible at any scale, fall back to the lowest-norm point of R_1 x R_2 x R_3 reachable from the best state (always defined, since each R_i is non-empty by construction), count those blocks, and print the count next to the MSE. Assert that every eval block produced a point and that exactly 20,000 e2 terms were summed.

**MINOR**

- **Problème.** Two numbers in the prediction basis are wrong, and both bias the point prediction downward, toward a falsely comfortable distance from the 90.3% kill. 2^47 / N(12) = 140,737,488,355,328 / 111,043,117,458,000 = 1.267, i.e. 26.7% more points, not "21% more points". And the 0.342 spare bits are worth about 0.342/24 = 0.0142 bits of SQNR, hence roughly 0.65 to 0.71 pp of retention at rate 2.000 - not "+0.16 pp". The saturation term is understated by about half a point, most of the 0.7 pp gate band.

  **Correctif proposé.** Correct both figures (26.7%, and about +0.65 pp) and re-state whether the point prediction of 89.6% and the range [88.8, 90.3] still hold once the corrected term is used. The empirical prototype delta already contains the effect, so the point prediction may stand - say so explicitly rather than leaving an arithmetic error inside a signed bet.

- **Problème.** The scoring citation names a function neither arm can call, and the ban on BlockDots13 rests on a false reason. shape_gain_mse_shipped is at llvq-bench/src/lib.rs:190-198 (not 186-196) and takes &[BlockDots], whose t() maxes over shells 2 and 3 only (lib.rs:49-52); wiring F1 or ball-12 winners through it would silently score against a two-shell codebook. Separately, section 6 bans BlockDots13 because "t() and spherical_err2 hard-code ||v||^2 = 16m; a three-section code has neither". Every F1 winner is a Lambda_24 point and does satisfy ||y||^2 = 16m (llvq-core/src/leech.rs:14-16); what actually fails is that the F1 region reaches ||y||^2 of roughly 248-272 (m about 15.5 to 17, from the spec's own rho2 ranges), overrunning the NSHELLS = 12 slots of BlockDots13::d.

  **Correctif proposé.** Fix the line reference and say the arms reimplement the formula inline, the way llvq-bench/src/bin/lcap.rs:120-127 does. Restate the BlockDots13 ban on its real ground: the array covers m = 2..=13 only and the F1 region exceeds it.

- **Problème.** "set_shell_cap is never called in any retention bench" is defensible but reads as "the plumbing does not exist", which is false and may send an implementer building it from scratch. llvq-bench/src/bin/lswap.rs:71-72 already does BallSearcher::with_level_cap(4) followed by set_shell_cap(12), and llvq-llm/src/bin/cosdiag.rs:83-84 does the same at MAX_LEVELS_ANY.

  **Correctif proposé.** Narrow the claim to "no retention bench measures the ball at cap 12" and cite lswap.rs:71-72 as the existing call pattern to copy.

- **Problème.** Arm A3 can return the origin where A2 cannot, and SC8 does not account for it. nearest_scaled seeds the incumbent at best0 = 0.0 so the zero point is a candidate (llvq-search/src/generic.rs:681-694), while nearest_angular seeds -inf and never returns it (generic.rs:709-711). A zero winner gives ||y|| = 0 and an undefined t, so SC8's relative deficit divides by zero or produces NaN at the scales where the origin wins.

  **Correctif proposé.** Define t = 0 for a zero-point return, as llvq-bench/src/bin/lcap.rs:75 already does, state it in the A3 description, and assert in SC8 that every per-block t is finite.

- **Problème.** The prereg name collides. proofs/ already contains preregistration-f1-cublasf16-2026-08-18.md and its .ots, an unrelated "F1". A file named for F1 alone will be ambiguous in the registry and in any later ECARTS reference.

  **Correctif proposé.** Name it for the gate and the object, e.g. proofs/preregistration-f1b-codebook-trois-sections-2026-09-XX.md, stamp it with opentimestamps before the first measurement per docs/METHODE.md section 1 and hard rule 2, and carry the absolute gate thresholds (90.3 / 91.0), the A2 acceptance window and the signed prediction inside it.

### Lentille « claim » — verdict : sound_with_fixes

The construction is arithmetically sound — I re-derived it independently and it holds. S1 and S3 really are cosets of 4·E8 (the two prefixes of a state differ by 0xff, and 4D8 ∪ (2·1+4D8) = 4E8; the parity bookkeeping survives because every Golay word meets an octad evenly, self-duality); S2 is a coset of 2√2·E8 (Construction A on [8,4,4], covolume 2^12); 2^16·2^12·2^16/256 = 2^36 = det(√8·Λ24), so the word is a bijection and the state is recoverable from the point (p from the coordinates mod 2, c from (y−p)/2 mod 2, r from Σk on section 1). The rate convention is right: paper Table 8 (docs/llvq-paper-notes.md:71) reads `norm(Λ₂₄(12)) + 1 gain bit` at 1.95833 + 0.04167 = 47 + 1 packed bits, so 92.14% is genuinely a 48-bit number and A2 is the right control. index_bits(12) = 47 confirmed by recomputing N(12) = 111,043,117,458,000 from the Leech theta series. The D8 parity repair as written is the correct Conway–Sloane rule (flip the largest |z−round z| away from its rounding: cost increase 1−2|δ| is minimised by the largest |δ|). The anisotropy arithmetic checks exactly: at fixed volume, Σa_i² over a product of 8-balls with e_i = w_i + log2(covol_i) gives 0.213 dB for 13/13/13 and 0.011 dB for 12/15/12 (spec says 0.2116), and radii ratio 12.34/8.72 = 1.415 against the DP's 10.6/7.5 = 1.413.

Where it breaks is the CENTRAL CLAIM's reasoning and the PREDICTION's basis, not the code. Three things. (1) The state coupling cannot buy back shaping at all: R_i is defined per coset by norm truncation, so every one of the 256 products shares the same per-section radius envelope (88..96 / 72..80 as measured), and the union of 256 offset products is contained in a product of balls only 2.2% larger in radius than the median product. The coupling changes which 2^47 points fill the region, never the region. So the spec's second (+) effect — "the coupling F1a said only F1b could price" — is void, and F1a's open question is answerable on paper. (2) The concentration argument is quantitatively false in dimension 8: for a Gaussian block the max of the three octad energy fractions has median 0.464 (not 1/3), P(>0.5) = 0.34, P(>0.6) = 0.087. The product region's radial extent is a/max|u_i|; at 9.6/8.7/9.6 that equals the equal-volume ball's 14.12 at max f = 0.462, i.e. the median. For half of all blocks the product region is SHORTER than the ball, not "the product loses only where the source rarely goes"; √3·9.5 = 16.5 is reached on a measure-zero direction. (3) 89.10% is a category error of unknown sign: ops/f1a_shaping.py applies a normalized-second-moment ratio — the functional for a lattice/spherical-shaping quantizer with the source uniform in the region — to a shape-gain MSE, where the source direction is uniform on S^23 and the region enters only through angular density (points per solid angle ∝ ρ(u)^24, so D ∝ E_u[ρ(u)^{-48/23}]). Monte-Carlo of that functional gives 0.51 dB for 12/15/12 and 1.11 dB for 13/13/13 against the ball, versus NSM's 0.37 and 0.20. So the two functionals disagree by a factor of 2 on the anisotropy sensitivity, and nothing in the spec establishes which one governs.

That last point turns into the sharpest finding, because the spec's own prototype arbitrates against it — and against itself. Its headline row is reproduced to four significant figures by "closed form × point-count correction": 0.077718 × 10^0.03666 × (2^47/N(12))^{-1/12} = 0.082910 / 89.808%, against the reported 0.082903 / 89.81%. That is 1/100 of the row's own quoted standard error (±0.25 pp at n = 2600). Under the same law the 13/13/13 row should read 0.086857 / 88.13%, and it is reported at 0.092281 / 85.95% — 2.2 pp away. No single shaping functional fits both rows: the first requires pure NSM, the second requires something about half-way to the angular functional. One of the two rows is not measuring what it claims, and the 13/13/13 row is the one where the specified encoder should misbehave most (middle radius 7.5 against end radii 10.6 makes the infeasibility rejection bite hardest). The reconciliation also rewrites the prediction's causal story: the entire +0.7 pp above 89.10 is code-space saturation, which the spec sizes at +0.16 pp — the true figure is 2^47/N(12) = 1.267, i.e. 26.7% more points (the spec's "21%" is the same ratio read backwards), worth (1.267)^{1/12} of MSE = +0.71 pp. The two effects the spec leads with contribute nothing.

My counter-prediction: the point estimate 89.6% is roughly right by accident, the reasoning is wrong, and the risk is skewed low (87.9–89.9) rather than symmetric, because the angular functional and the uncontrolled rejection bias both push down while only saturation pushes up. Note also that the arm most likely to decide the gate is A2: extrapolating the repo's own ball-13 anchor (MSE 0.0725 at 2.0417, docs/archive/face-au-4-bits.md:198) to ball-12 gives anywhere in 91.9–92.6% depending on whether you scale by the point-count ratio (2.53×, → 91.88%) or by the one index bit (→ 92.57%). That ±0.4 pp harness offset is comparable to the whole 0.7 pp kill-to-adopt band, and with the prototype's −2.2 pp delta it puts A4 at 89.7–90.4 — straddling the 90.3 kill. Combined with an encoder bias that the stated control cannot see, F1b as specified can land in a band where it decides nothing. It is worth running with the fixes below; it is not worth running without them.

**BLOCKING**

- **Problème.** The encoder's infeasibility-rejection bias is uncontrolled, and SC8 — the control the spec offers for it — is run on an arm where the bias cannot occur. §3 rejects a branch when the per-coset unconstrained minimiser falls outside R_i, instead of finding the nearest point of S_i ∩ R_i. That is exactly the boundary where the truncation (and therefore the whole shaping question) lives. SC8 compares the scale sweep against nearest_angular on the ball-12 codebook, which has no truncation and no rejection at all (llvq-search/src/generic.rs:709 searches the full shell-capped ball), so it measures grid coarseness only. The spec's own prototype gives evidence the bias is large: the 13/13/13 row, whose middle radius 7.5 against end radii 10.6 maximises rejection, misses the NSM+saturation law by 2.2 pp while the 12/15/12 row matches it to four significant figures.

  **Correctif proposé.** Add SC9: on a 500–1000 block subsample, replace rejection by an exact constrained per-section search (enumerate the D8 candidate list — nearest, its 8 single-flip neighbours, and radial shrinks — keeping only points inside R_i) and report the retention difference against the rejecting encoder. That difference, not the ball-arm sweep gap, is the bound on F1's one-sided bias. Also change the prefix/suffix reduction from 'min over the 2, reject if outside R' to 'best feasible of the 2', which is free.

- **Problème.** The gate-reading rule is ambiguous and the two readings straddle the kill. docs/ROADMAP.md:118 writes the F1b gate on absolute retention (adopt ≥ 91.0%, kill < 90.3%), while §4 of the spec says 'the delta A4 − A2 ... is the number that decides F1b'. Those are different quantities whenever A2 ≠ 92.14. Extrapolating the repo's ball-13 anchor (0.0725 / 92.72% at 2.0417, docs/archive/face-au-4-bits.md:198) to ball-12 gives 91.88% if you scale by the point-count ratio 2.53× and 92.57% if you scale by the single index bit — a ±0.4 pp harness offset against a 0.7 pp gate band. With the prototype's −2.2 pp delta the two readings put A4 at 89.7 and 90.4, on opposite sides of the kill.

  **Correctif proposé.** State in the prereg, before measuring, which scalar the gate is read on. The defensible choice is the transported figure 92.14 + (A4 − A2), since the ROADMAP thresholds were set against the paper's 92.14 scale; if instead the absolute A4 is used, say so and record that the thresholds inherit the harness offset. Either way, publish A2 first and state its offset from 92.14 before the F1 row is read.

**SERIOUS**

- **Problème.** The (+) argument that the state field 'couples the sections ... a union of 256 differently-offset products, not one product' cannot move the shaping number, and the spec's framing hides that. R_i(coset) is a norm truncation defined per coset (§2), so every state's region is a product of balls, and the union over 256 states is contained in the product of the max-radius balls — with the measured spreads (88..96 and 72..80) that envelope is only 2.2% larger in radius than the median product, and at fixed point count the union is a rounding of the product, second order in that spread. The kill needs 0.145 dB recovered (39.6% of the 0.3666 dB gap, docs/ETAT.md:129-131). The coupling cannot plausibly supply it.

  **Correctif proposé.** Replace the (+) claim with the envelope bound and derive it explicitly in the prereg: max-over-coset radii from the same counting DP, one paragraph, no measurement. Then state F1b's real question — 'what does the actual three-section code at 48 packed bits retain' — instead of 'how much does the coupling buy back', which F1a's own open item asks and which is answerable on paper.

- **Problème.** 89.10% is the wrong functional for a shape-gain measurement, so 'ABOVE 89.10 by +0.3 to +0.8 pp' predicts the sign of an error term in a formula that does not apply. ops/f1a_shaping.py:39-58 multiplies a shape-gain MSE by the ratio of normalized second moments of two shaping regions — the correct penalty for the spherical-shaping arm (llvq-bench/src/lib.rs:62-67), not for the shape-gain arm, whose codebook is a set of directions and whose source direction is uniform on S^23. Under the angular functional (points per solid angle ∝ ρ(u)^24, D ∝ E_u[ρ(u)^{-48/23}]) I get 0.51 dB for 12/15/12 and 1.11 dB for 13/13/13, against NSM's 0.37 and 0.20 — a factor of 2 disagreement on anisotropy sensitivity.

  **Correctif proposé.** Stop treating 89.10% as the yardstick and label it in the journal as an NSM-functional estimate whose applicability to shape-gain is unestablished. Report both bracketing predictions (NSM 0.37 dB, angular 0.51 dB, plus the +0.086 dB saturation credit) and let the 13/13/13 arm arbitrate between them — that comparison, not the absolute number, is the free science in F1b.

- **Problème.** The prototype rows that carry the signed prediction are mutually inconsistent, and the headline one is implausibly close to a closed form. 0.077718 × 10^0.03666 × (2^47/N(12))^{-1/12} = 0.082910 / 89.808%, against the reported 0.082903 / 89.81% — agreement to 4 s.f., about 1/100 of the ±0.25 pp sampling error the spec itself quotes for n = 2600. The same law puts 13/13/13 at 0.086857 / 88.13% against the reported 0.092281 / 85.95%. Whichever row is right, the other is not a measurement of the same object, and 'BASIS. Not a guess' does not survive that.

  **Correctif proposé.** Before running, resolve the provenance of the two prototype rows: re-run both at the same n with the same code path and state whether the F1 rows were computed or measured. If the coincidence is genuine, say so and note that the prototype then adds nothing the closed form did not already give.

- **Problème.** The composition of the prediction is wrong even where the total is roughly right. The 'saturates the code space' term is 2^47/N(12) = 1.2674, i.e. 26.7% more points (the spec's '21%' is 1 − N(12)/2^47, the ratio read backwards), and at the high-rate exponent it is worth (1.2674)^{1/12} of MSE = +0.71 pp of retention, not the +0.16 pp claimed. That single term is the whole of the +0.7 pp that separates the prototype from the closed form; the two effects the spec leads with contribute nothing.

  **Correctif proposé.** Rewrite §'WHY ABOVE 89.10' with saturation as the sole and quantified upside (+0.71 pp, computable in one line), the coupling at ~0, and the shape-gain framing as a downside risk of 0 to −1.4 pp rather than an upside. The point prediction can stay near 89.6; its interval should become asymmetric downward, roughly [87.9, 89.9].

- **Problème.** SC3 contradicts the F1a journal it claims to reproduce, and the contradiction propagates into ETAT/ROADMAP. llvq-bench/src/bin/f1count.rs:196-224 divides 1,024 deduplicated Golay edges by trio_states = 256, which is the Λ24 state count, and prints '4 branches per state'; the honest out-degree is 1024/64 = 16 per Golay state, which is exactly what the spec's own trellis uses ('b in mids[g8] (16)') and what SC3 asserts. The 8.0 KiB table verdict survives only because 256×4×8 = 64×16×8; the derived statement in docs/ETAT.md:142-146 and the journal ('2 coded bits and ~11 arithmetic') is wrong — it is 4 coded bits, and a table indexed by the transmitted 256-state field rather than the 64 Golay states would be 32 KiB, over the 16 KiB gate, unless p is factored out arithmetically.

  **Correctif proposé.** Either drop the claim that SC3 'reproduces F1a' and file the discrepancy as an ECARTS note against the F1a journal, or fix f1count.rs to divide by the Golay state count and restate F1a's green as '64 states × 16 branches, p factored out arithmetically, 8.0 KiB'. Do not let SC3 assert 16 while the cited journal says 4.

**MINOR**

- **Problème.** The origin is in the F1 region and the spec never excludes it. With p = 0, the zero pattern and r = 0, y = 0 is the lowest-norm point of every section, so it is in R_i for state (g8=0, p=0, r=0); the shape-gain rule then divides by ‖y‖ = 0. The ball arm excludes it explicitly (llvq-search/src/generic.rs, nearest_angular: 'The origin is not a candidate — a direction is always required'), and SC1's 'shell_index Some(m) with m ≥ 2' is precisely the assertion that would fire on it, since Λ24 has no vector of norm² = 16.

  **Correctif proposé.** State that y = 0 is excluded from the F1 arm's candidate set (and that the codebook therefore offers 2^47 − 1 usable directions at an honest 47-bit field), and guard t = ⟨x,y⟩/‖y‖ against ‖y‖ = 0 in the trellis rather than relying on the argmax over s never selecting it.

- **Problème.** SC4's collision clause has no statistical power. Drawing 10^6 random words out of 2^47 and checking that no two decode to the same point has an expected collision count of ~3.5e-6 even for a map with substantial structure; it can only catch a gross failure, not the subtle one it is written for.

  **Correctif proposé.** Keep the round-trip half of SC4 (decode then re-encode to the same word), which does have power, and replace the pairwise clause with a structural injectivity argument in the test's doc comment: p, c and the section parities are all recoverable from the decoded point, hence the state is, hence the label is.

- **Problème.** Two miscitations. shape_gain_mse_shipped is at llvq-bench/src/lib.rs:190-198, not 186-196 (186 is inside the doc comment of shape_gain_mse_projected), and retention_pct is at :218, not :212 — though :212 is correct for the separate claim about sqnr_bits and unit variance. Also 'BallSearcher::with_level_cap(5)' is worth a word: MAX_LEVELS_ANY = 5 (llvq-search/src/generic.rs:56), so with_level_cap(5) is identical to BallSearcher::new() and imposes no codebook restriction — which is what makes index_bits(13) = 48 the right label for A1 and index_bits(12) = 47 the right label for A2.

  **Correctif proposé.** Correct the two line ranges and add one sentence noting that with_level_cap(5) == new(), so the level filter does not desynchronise the rate from the codebook the way lcap.rs:49-52 warns about.

## 7. Pièges relevés à la reconnaissance

- Comparing an F1b number measured in this harness against 92.14% — that value is the paper's Table 8, never measured here (docs/fiche-4b.md:288). Our harness gives MSE 0.0725 for ball-13 + 1 gain bit where the paper gives 0.078 for ball-12 + 1 gain bit, so the two harnesses differ by ~7% of MSE. The comparator must be a ball-Lambda24(12) + 1-gain-bit arm run in the SAME process on the SAME blocks — and that arm does not exist yet, because set_shell_cap is never called in any retention bench.
- Using rate_shape_gain13(1) = 2.041560 as the denominator for a 48-bit F1 word. That understates retention by about 1.9 pp. The correct denominator is 48/24 = 2.000, the packed convention of lcap.rs:113-114.
- Calling rate_shape_gain13_single (llvq-bench/src/lib.rs:424-427), still live in main.rs:139-148. It divides by log2|Shell(m)|/24 + k/24 — the exact fractional rate that produced the retracted 92.24% on 2026-08-04.
- Scoring with shape_gain_mse13_projected (lib.rs:330) instead of shape_gain_mse13_shipped (lib.rs:372). The projected rule rounds <x,v-hat> and is a strict lower bound, ~2% of block MSE and 0.7 pp of retention better than the encoder that ships. Every docs/ table written before 2026-08-01 quoted the bound.
- Reusing BlockDots13 for an F1 codebook. Its `d` array is fixed at 12 shells and both t() and spherical_err2 hardcode ||v||^2 = 16m in the sqrt(8) integer embedding. A three-section E8 coset code has neither, so the numbers would be silently meaningless rather than failing loudly.
- Taking a beta* or a comparator from betasweep. betasweep.rs:14-20 fits beta on the same 20,000 blocks it scores — no held-out split, so its retentions are optimistic relative to every main.rs and lcap.rs row.
- Sweeping a scale on an F1 shape-gain arm and calling the result comparable to the shape-gain rows. beta is a spherical-shaping-only knob (lib.rs:262-269), its window is hardcoded to (0.2, 0.9), and g4_full.rs:42 asserts beta in [0.25, 0.5).
- Normalizing the source, per block or per row. retention_pct's -0.5*log2(mse) is only retention for a unit-variance source (stated at lib.rs:212); gauss_block deliberately does no normalization.
- Changing the RNG draw order. Box-Muller consumes exactly two u64 per sample (rng.rs:36-41), so drawing eval before train, or adding one draw, changes every block and silently decorrelates the two arms.
- Decoding F1 blocks with a Viterbi/trellis pass and comparing against numbers produced by exact nearest-neighbour search (BallSearcher::shell_bests, nearest_angular). The result would fold decoder suboptimality into the codebook's geometry; the only guard in place is g4_full.rs:45 (mse > shannon), which catches only gross bugs.
- Quoting an 8,000-block test figure as the F1b result. The protocol is 20,000 eval blocks (docs/ROADMAP.md §2.2); only main.rs and lcap.rs run at that size.
- Recomputing retention from a rounded MSE. The paper's 0.078 gives 92.01, not 92.14 (docs/fiche-4b.md:290). Publish MSE to at least six digits.
- Trusting the label MSE_SERVED in ops/f1a_shaping.py:32. 0.077718 is the paper's Table 8 MSE back-computed from its SQNR 1.843, not a measurement of the served leech1c12 file on a Gaussian source. The value is a correct paper anchor; the name is misleading.
- Running F1b before timestamping a prereg. proofs/ contains no F1b preregistration, and hard rule 2 requires one before the first measurement. Related: docs/mesures/ holds no G4 or lcap journal at all, so F1b should write the first one.
- Metric. The served encoder picks the point by the ANGULAR metric (`nearest_angular`, llvq-quant/src/quantizer.rs:563), then codes the gain in 1 bit separately. An F1b bench that ranks by Euclidean distance (`nearest_ball13` / `nearest_scaled`) is measuring a different codebook and its retention number is not comparable to the 92.14% served figure quoted in docs/ROADMAP.md:126-129.
- Shell cap. `BallSearcher::new()` defaults to shell 13 (generic.rs:512), but the served artifact is `leech1c12` — shell 12, 47 bits (smoke.rs:1381, calib.rs:244-245). Benchmarking F1 against an uncapped shell-13 baseline silently inflates the baseline and understates retention.
- The ×8. `Leech::norm2` returns the INTEGER norm, 16m, not ‖v‖² = 2m. Anyone who reads "Λ₂₄" and applies the textbook min-norm 4 will be off by a factor of 8 in MSE and 2.83 in distance, and the error is uniform so nothing will look wrong.
- The two scales in one comment. `leech.rs:52-53` writes the ball as "‖x‖² ≤ 26" while the whole rest of the file uses the integer scale where that same ball is norm2 ≤ 208. Copying 26 into integer-coordinate code gives an almost-empty codebook.
- Origin off-by-one. N_SHELL_13_CUMULATIVE excludes the origin (leech.rs:49-54) but the indexer includes it as index 0 (index.rs:191). A codebook size of N13 instead of N13+1, or the reverse, shifts every index by one and still decodes to plausible lattice points.
- shells.rs cannot validate an F1 codebook. It covers m = 2 and m = 3 only (shells.rs:34, g1_invariants.rs:427). Exhaustive verification above m = 3 does not exist in this tree; the only exhaustive handle is the 196,560 points of shell 2, and everything else must be checked pointwise through `Leech::contains` plus `Indexer::encode`/`decode`.
- Permuting one side only. Every coordinate↦bit site (leech.rs:88-89, index.rs:203-209, generic.rs:297 and :327, fastdec.rs:346 and :383, lib.rs:308/:320/:390) is covariant: permute the vector without permuting the codeword table (or vice versa) and `contains` still returns true for a set of points that is a DIFFERENT lattice. No test catches it, and reconstruction stays plausible.
- Trying to get the trio ordering out of `Golay::new()`. It builds from GEN_POLY = 0xC75 and sorts weight-major (golay.rs:62-81); the generator is pinned by `generator_factorization` (golay.rs:150-157) and rebuilt from GEN_POLY in G1 (g1_invariants.rs:94-99). The permutation has to be applied to a copy of `codewords()` outside llvq-core, as f1count.rs:118-126 does. `Golay::contains`/`rank`/`rank_in_weight` are binary searches on the built table and cannot serve a permuted code.
- Gaussian versus real weights. Permuting the input block and permuting the point back is safe for the 20,000 i.i.d. Gaussian blocks of F1b (the Gaussian is rotation-invariant, so the distortion distribution is unchanged), and it is NOT safe as evidence about real weights: which input channel lands on which coordinate stops being exchangeable there. A retention number measured that way must be labelled as a Gaussian result only.
- Touching the v1 index map during F1b. `codebook_fingerprint` folds in the whole codeword table (codebook.rs:136-141) and is pinned at 0x338f_420f_1186_6319 (codebook_fingerprint.rs:58); moving it invalidates the published Qwen3-4B artifact. docs/ROADMAP.md:237 gates format v2 on F1b being green — F1b must stay entirely inside llvq-bench with its own codebook.
- Reproducibility of the block population. `SplitMix64::next_gaussian` (rng.rs:35-40) burns two u64 per sample via the cosine branch only; interleaving any other rng call changes the whole stream. And `Leech::random_point` must not be used to build the population — its Σk repair skews x[0] (leech.rs:107-109, :122-123).
- `shell_index` returns None silently for any norm2 that is not a multiple of 16 (leech.rs:139-142). A malformed F1 point does not raise; it just drops out of whatever `filter_map` consumes it, and the block count quietly falls short of 20,000.
- BallSearcher::new() defaults to shell_cap = 13, not 12 (llvq-search/src/generic.rs:387-389, :516). Every existing Gaussian harness in llvq-bench (precompute13 at lib.rs:273, lcap.rs:110, rtbits.rs:606) therefore searches Lambda_24(13), which is NOT the served codebook. An F1 arm compared against them would be compared against a 48-bit direction code plus a gain bit, i.e. 49 bits, and would look worse than it is. Call set_shell_cap(12) explicitly.
- The gain code rounds the block NORM ||x||, not the projection t = <x, v_hat>. shape_gain_mse13_projected (llvq-bench/src/lib.rs:355) is a lower BOUND, not a configuration; the two differ by the factor 2/(1+cos theta), about +2% of block MSE and 0.7 points of retention at the cos theta ~ 0.96 this codebook achieves (lib.rs:349-370). Every docs/ retention figure written before 2026-08-01 reported the bound. If F1 is scored with one rule and the baseline with the other, the comparison is meaningless.
- The rate must be exactly 2.000 b/dim (48 bits / 24 dims), not a fractional log2 of the codebook size. llvq-bench/src/main.rs uses rate_shape_gain13(k) = (log2(N(13)) + k)/24 (lib.rs:388), a fractional rate no file pays; lcap.rs uses the integer width (lcap.rs:112). ops/f1a_shaping.py:24-26 records the 2026-08-04 error explicitly: dividing by the 47 bits of the lattice field alone flatters F1 by 1.9 pp of retention.
- 92.14% is the PAPER's number on its unrounded MSE 0.077718, and docs/fiche-4b.md:290 forbids recomputing it from the rounded 0.078 (which gives 92.01). It is not an output this repository's bench has ever printed for the served configuration. Producing an in-repo baseline with the identical harness, before comparing F1 to anything, is mandatory; otherwise F1's number is compared against a differently-defined quantity.
- fit_gain_centroids (quantizer.rs:646, production) and lloyd_max (bench lib.rs:131) are two separate implementations of the same Lloyd-Max. They agree on a Gaussian source with row_scale = 1, but they are not the same code path: production skips zero-norm blocks and rows with scale <= 0, and divides by the row scale first. Use one set of centroids for both arms, as bench_matches_production.rs:45-48 does, or the comparison measures the fit rather than the codebook.
- nearest_angular is scale-invariant and the live path applies no beta. An F1 encoder built on spherical shaping (a fixed beta, the Euclidean objective of nearest_scaled) is a different quantizer family; its MSE would not be comparable to the shape-gain baseline, and the retraction that the served format relies on would no longer be a no-op.
- The origin is not a candidate for nearest_angular (best0 = NEG_INFINITY, generic.rs:709-711, asserted at g2c_reference.rs:181). Index 0 is reserved for the origin and reached only by the zero-block short circuit in quantize (quantizer.rs:553-559). An F1 word that spends a codepoint on the origin inside the shaping region, or that omits one, changes the codebook size and therefore the rate.
- The trio reordering that makes the 8-bit state field fit is a coordinate permutation, hence a different index map, hence a format v2 that breaks codebook_fingerprint (f1count.rs:113-117, and docs/mesures/f1a-comptes-2026-09-04.txt result 2). The bench itself never indexes, so this does not affect F1b's number, but the point returned by any F1 encoder must still satisfy Leech::contains in the REPOSITORY's coordinate order if it is to be checked against llvq-core, or the check must be applied to the un-permuted point.
- The retraction is a no-op only for the default LeechShapeGain (retraction_target returns None, quantizer.rs:533-548). A bench that calls quantize directly matches production for that configuration and only that one; with_free_magnitude or design C (gptq.rs:290-296) both restore a free float per block and cancel the gain code.
- The 89.10% figure in docs/ROADMAP.md:125-127 assumes three INDEPENDENT 8-balls; the 4 branches per state found in F1a are exactly the chaining that assumption ignores (docs/mesures/f1a-comptes-2026-09-04.txt, 'ce que ca ne dit pas'). It is a floor for the product form, not a prediction for F1b, and quoting it as F1's expected retention would be wrong in the pessimistic direction.
- The ~11 'arithmetic' bits per section presuppose a rank decode inside E8 that has never been costed. E1v died on exactly that, on online decode cost and not on bytes (docs/mesures/f1a-comptes-2026-09-04.txt). F1b measures quality only; nothing in it licenses a speed claim.
- nearest_level_index uses a strict < on |g - c| (quantizer.rs:629-640), so an exact tie between two gain centroids goes to the LOWER index. Any F1 gain selection that breaks ties the other way produces different weights on tie blocks, and a round-trip test would catch it only if a tie occurs in the sample.
- Treating a branch as a coset of 4*Z^8 and rounding each coordinate independently: half of all targets land outside the branch. The branch is a coset of 4*D8, an index-2 sublattice, and the parity repair is mandatory, not an optimisation.
- Mixing the two E8 scales. The end sections live on 4*E8 (covolume 2^16, min norm 32); the middle lives on 2*sqrt2*E8 (covolume 2^12, min norm 16). Using one scale for all three breaks the point count by 2^4 and silently changes the rate.
- Keeping the roadmap's literal 13/13/13 label split. It is not the equal-radius split; it makes the middle ball 1/sqrt2 smaller in radius and costs an extra 0.2116 dB (retention 87.34% instead of 89.10%). The bit budget says 35/3 per end section and 4 + 35/3 for the middle, i.e. 12/15/12 with integer fields.
- Defining the per-section codebook as 'the points of the coset inside a fixed radius'. The number of coset points inside a fixed ball depends on the coset offset, so that map is not a bijection. It must be 'the 2^w lowest-norm points of this coset', with a deterministic tie-break, or the 47-bit word is not one-to-one.
- Reusing the '4 branches per state' number as a Golay branch count. It is 1024 Golay edges divided by 256 Lambda_24 states; the Golay branch count per state is 16, and the coded-bit split of the middle label is 4 branch bits + 1 parity bit, not 2.
- Forgetting that the middle section's outgoing parity is a transmitted degree of freedom while the end sections' parities are forced by the state. Getting this backwards changes the per-section densities and therefore every volume, radius and retention number.
- Applying the trio permutation to some artefacts and not others. Leech::contains and every packed index in the repository are in the natural order; the trio order is a different index map, hence a different codebook_fingerprint and a format v2 (docs/ETAT.md:147-148).
- Counting the rate over the 47-bit lattice field instead of the 48-bit word. ops/f1a_shaping.py's own docstring records that this flattered a previous number by 1.9 points of retention.
- Costing the '~11 arithmetic bits' as free. They are a rank/unrank inside an E8 coset; E1v died on exactly that online decode cost, not on bytes (docs/ETAT.md:149, llvq-search/src/rankdec.rs:33-36).
- Assuming a state- or history-dependent per-section codebook recovers shaping gain. At a fixed count per section and an isotropic source it recovers nothing; only varying the counts (joint indexing, or per-state field widths) can.
- Ignoring overload. The branch codebook is a truncated coset, so the unconstrained D8 answer can fall outside it; without a defined fallback the encoder either emits an invalid index or silently biases the measured retention.
- Reporting the retention numbers here as measurements. They are the ball-NSM closed form of ops/f1a_shaping.py, which ignores the granular (lattice) term that the 20,000-block F1b benchmark will actually measure.
