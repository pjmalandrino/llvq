# Roadmap

La suite du projet, avec ses gates et ses coûts. État au 2026-09-02. Le passé est dans
[`HISTORIQUE.md`](HISTORIQUE.md), les règles dans `METHODE.md`.

## 1. Point de départ

Le 4B publié rend 100,6 tok/s dans 2,57 Go en config servie v1 (*mesuré*,
[d1-fusion-servie-2026-08-24](mesures/d1-fusion-servie-2026-08-24.txt), gel aux trois tailles dans
[vague2-fusion-8b-14b-2026-08-31](mesures/vague2-fusion-8b-14b-2026-08-31.txt)). Il perd 14,73 pp de
MMLU sur le f16 (*mesuré*, [a4-campagne-2026-08-06](mesures/a4-campagne-2026-08-06.txt)) et 14,45 pp
sur l'AWQ 4 bits (*mesuré*, [mmlupair-4b-8b-2026-08-13](mesures/mmlupair-4b-8b-2026-08-13.txt)),
pour 5,162 b/param contre 5,302 (*calculé* sur octets mesurés,
[rtbits-planes-8b-2026-08-09](mesures/rtbits-planes-8b-2026-08-09.txt)). A2 (CUDA Graphs) rend
+13,45 % au 4B (*mesuré*, [a2-verdict-2026-09-01](mesures/a2-verdict-2026-09-01.txt)). Il n'est pas
servi : sa fenêtre KV coûte +47 % de VRAM à 8k, +1,21 Go sur 2,57 (*calculé*,
[preregistration-a2-a3-geometrie-2026-08-31-ECARTS](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md)
§É7).

Le noyau ne rouvre pas la question produit. À 100 % de sa borne d'octets, `Planes14` plafonne à
3,33× FP16, soit 16/4,804 (*calculé*,
[plan-cloture-2026-08-27](archive/plan-cloture-2026-08-27.md)), sous l'AWQ à 3,38× (*mesuré*,
[spec-apres-awq-2026-08-10](archive/spec-apres-awq-2026-08-10.md)) et à 0,68× de QTIP (*calculé*).
Ce qui décide de la suite est la qualité. Un levier qui referme 4 à 6 pp de MMLU rouvre la question
produit ; sans lui, le volet produit se clôt sur la conclusion actuelle. La roadmap recherche est
adoptée (D0, commit `1e8583c`) avec un plafond de 5 $ pour la vague 1, définie comme M2 plus une
réplique sur une seconde graine. M2 a coûté ~2,17 $ (*calculé* sur les horodatages du bucket,
[m2-attribution-4b-2026-09-02](mesures/m2-attribution-4b-2026-09-02.txt)) ; la réplique n'a pas
tourné.

Le papier est déposé sur Zenodo (DOI de concept 10.5281/zenodo.22133606) après le renvoi sans revue
de TACO le 2026-08-27. Le 2026-09-02, une première soumission arXiv (7927047) a été refusée : le PDF
avait été téléversé à la place des sources. Les sources ont été resoumises le même jour
(`\pdfoutput=1` ajouté, commit `e721bc5`). Aucune acceptation ni identifiant arXiv n'est consigné.

## 2. Roadmap recherche

Trois axes. M mesure et outille, F cherche un format sans dépliage, Q cherche à perdre moins. Tous
les A/B à 0,6B suivent le gate du design C : 28 blocs, même graine, puis 3 graines. Toute expérience
qui recalibre se lit contre σ = 5,2 % de perplexité (*mesuré*,
[f5-graines-4b-2026-08-19](mesures/f5-graines-4b-2026-08-19.txt)) et 2,92 pp de MMLU (*mesuré*,
[bruit-mmlu-graines-4b-2026-08-25](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)).

Quatre pistes restent fermées, chacune mesurée une fois. Ce sont le volume de calibration, un format
à ALU inchangée (`Golay70`, E1c, E3), la course de décodage sur `tv_planes` et le 32B avant D4. Le
plafond de 30 $ se confirme vague par vague, jamais en cumul.

### 2.1 Axe M, mesure

| id | piste | coût (*mesuré* si fait, `jobs.csv` ; *estimé* sinon) | adoption | kill | état |
|---|---|---|---|---|---|
| M1 | shrinkage hors-diagonale de H, 0,6B, 28 blocs, 3 graines | 0 $ | étendue inter-graines divisée par 2, médiane tenue | ρ* = 1 | fait, vert |
| M2 | attribution MMLU par type de projection, fichier constant, 11 bras | ~2,17 $ (*calculé*) | mesure, critère de lecture | aucun | fait |
| M2b | `v_proj` en int4 g128 déquantifié, fichier constant | ~0,29 $ (*calculé*, [journal](mesures/m2b-v4bits-2026-09-02.txt)) | G4 ≥ 3,0 et IC > 1,5 | G4 < 1,5 | fait, règle non tranchée |
| M3 | entropie d'attention par couche ; colonne MMLU-STEM dans `mmlupair` | 0 $ | écart f16/scellé > 3 fois l'écart inter-fenêtres | aucun | à faire |
| M4 | outillage contre la dérive | 0 $ | aucun, hygiène | aucun | à faire, section 3 |

M1 est vert. Le shrinkage `H_ρ = ρ·H + (1−ρ)·diag(H)` rend, à ρ = 0,7, une étendue inter-graines de
0,6847 ppl contre 4,6214 à ρ = 1 (*mesuré*,
[m1-hessienne-shrink-2026-09-02](mesures/m1-hessienne-shrink-2026-09-02.txt)). La médiane vaut
27,4944 contre 39,6042. ρ = 0,9 rend 27,0812 / 3,1498 et ρ = 0,5 rend 27,9506 / 2,9771. Réserve :
sur trois graines l'étendue tient à une seule graine, différente selon ρ. Prédiction opposable : n/N
vaut 0,023 au 0,6B contre 0,074 au 4B (*calculé*, même journal), l'effet doit être plus grand au 4B.

M2 est rendu. Restaurer un type de projection en f16 gagne, en pp de MMLU apparié (*mesuré*,
[m2-attribution-4b-2026-09-02](mesures/m2-attribution-4b-2026-09-02.txt)) :

| projection | `gate` | `up` | `v` | `down` | `o` | `k` | `q` |
|---|---|---|---|---|---|---|---|
| gain | +5,18 | +4,94 | +4,48 | +2,96 | +2,35 | +2,09 | +1,85 |
| IC95 | [3,04 ; 7,34] | [2,72 ; 7,17] | [2,39 ; 6,61] | [0,71 ; 5,17] | [0,32 ; 4,32] | [0,34 ; 3,79] | [0,22 ; 3,50] |

L'attention entière rend +6,90, le MLP +10,78, tout +14,73. Les deux témoins reproduisent 2 280
picks sur 2 280. Le prior de littérature sur `k_proj` est réfuté. La cible est `v_proj` : 2,6 % des
poids (*calculé*, même journal) pour +4,48 pp.

M2b est rendu, sans verdict. `v_proj` en int4 g128 rend 59,19 % de MMLU, +3,60 pp [1,47 ; 5,79],
McNemar 2,0e-4, soit 80,4 % du gain f16 (*mesuré*,
[m2b-v4bits-2026-09-02](mesures/m2b-v4bits-2026-09-02.txt)). La mémoire baisse : 5,149 b/param
(*calculé*, même journal). `Planes14` déplie à 4,804 b/poids (*mesuré*,
[c1-planesbench-2026-08-06](mesures/c1-planesbench-2026-08-06.txt)). L'int4 g128 sert le même
contenu à 4,250 (*calculé*, 4 bits plus échelle et biais f16 par groupe). Servir `v_proj` en f16
coûterait +0,263 b/param, soit 5,425 (*calculé*, même journal), au-dessus de l'AWQ. La ligne 1 exige
un IC entièrement au-dessus de 1,5 ; la borne basse vaut 1,47, et sur huit graines de bootstrap
elle va de 1,42 à 1,49, jamais au-dessus de 1,50 (*mesuré*, même journal). Les
lignes 2 et 3 exigent G4 < 3,0
([preregistration-m2b-v4bits-2026-09-02-ECARTS](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).

Nouveaux boutons : `LLVQ_RESTORE_F16` et `LLVQ_RESTORE_Q4` (`mmlu`, `ppl`, exigent `LLVQ_MODEL`),
`LLVQ_H_SHRINK` (`smoke`).

### 2.2 Axe F, format sans dépliage

F1 code Λ₂₄ comme code de coset à trois sections E₈ (Forney 1988, Lepowsky-Meurman 1982). Le mot
fait 48 bits : `[état 8][s₁ ~13][s₂ ~13][s₃ ~13][gain 1]`. Le décodage coûte trois lookups et deux
additions.

| id | piste | coût (*mesuré* si fait, `jobs.csv` ; *estimé* sinon) | adoption | kill | état |
|---|---|---|---|---|---|
| F1a | compter états et alphabets pour 47 bits, prouver la bijection | 0 $, 1 sem. | tables ≤ 16 Kio par section | hors budget, passer à F2 | à faire, en premier |
| F1b | codebook dans `llvq-bench`, 20 000 blocs, 48 bits empaquetés | 0 $, 1 sem. | rétention ≥ 91,0 % | < 90,3 % | à faire |
| F1c | format v2, encodeur, 0,6B 28 blocs | 0 $ | ppl à ±1 étendue de `leech1c12` ; encodeur ≤ 656 µs/bloc | hors bande sur 3 graines | à faire |
| F1d | bras `tv_l3e8` dans `planesbench`, QTIP témoin même processus | 1 $ | t ≤ 1,15·t(QTIP) ; ≤ 2,20 b/poids noyau | t > 1,5·t(QTIP) | à faire |
| F1e | 4B scellé en v2, `fusedrun`, MMLU apparié | 8 $ | ≤ 2,6 b/param ; MMLU ≥ 55,59 − 2 SE | MMLU < 53 % | à faire |
| F2 | treillis séquentiel + trellis shaping, géométrie d'A3 seule | comme F1 | repli si F1a ou F1b meurt | aucun | non budgété |
| F3 | cap par ligne, 44 à 50 bits/bloc, guidé par M2 | 7 $ | +2 pp MMLU apparié à b/param constant | < +1 pp | après D3, conditionnel à F1c |

F1b est sous son kill dès l'estimation. La perte de forme projetée vaut +7 à +9 % de MSE
directionnelle (*estimé*,
[projection-gains-2026-09-01](archive/projection-gains-2026-09-01.md) §1.4), soit une rétention
gaussienne de 88,9 à 89,6 % (*estimé*, dérivé de ce même +7-9 %) contre un kill à 90,3 %.
F1a doit compter la rétention exacte de la région de mise en forme avant toute ligne de code. Sinon
F1 s'arrête à son gate.

Le chemin servi gèle le champ de gain à 1 bit : 8 assertions, 4 shaders,
`llvq-cuda/src/planes14_host.rs:113` refuse toute autre valeur (*mesuré*, grep). Le format v2 de
F1c, le cap par ligne de F3 et tout bras Q qui change le code rouvrent le layout runtime en plus du
quantifieur.

### 2.3 Axe Q, qualité

| id | piste | coût (*mesuré* si fait, `jobs.csv` ; *estimé* sinon) | adoption | kill | état |
|---|---|---|---|---|---|
| Q1 | shrinkage de H en production, ρ dans [0,5 ; 0,9] | 0 $ à 0,6B, 7 $ au 4B | étendue ÷ 2 tenue, médiane ≤ +étendue | aucun | à faire, ouvert par M1 |
| Q2 | cible asymétrique et pondération de sortie | 0 $ puis 7 $ | Δppl ≥ 2 étendues, 3 graines | < 1 étendue | à faire |
| Q3 | GPTQ en faisceau, K dans {2, 4, 8} | 0 $, 10 h Mac | Δppl ≥ 2 étendues, σ non augmenté, encodeur ≤ K fois | < 1 étendue | à faire |
| Q4a | équi-norme inter-couches, version VQ | 0 $ | Δppl ≥ 2 étendues | < 1 étendue | à faire |
| Q4b | cartes 24×24 côté activations, diagonale d'abord | 0 $ puis 7 $ | diagonale Δppl ≥ 3 % ; pleine +2 pp | aucun | à faire, pleine après Q6c |
| Q5 | précision mixte sur `v_proj` | 7 $ et un noyau | ≥ +3 pp apparié pour ≤ +0,10 b/poids | < +1,5 pp | qualité mesurée par M2b, noyau à faire |
| Q6a | distillation des paramètres libres du format, 0 bit de plus | 3 $ | ≥ +3 pp apparié | < +1,5 pp | à faire, après M3 |
| Q6b | EoRA / RILQ r ≤ 16 | 3 $ | ≥ +3 pp dans ≤ +0,25 b/param | aucun | après Q6a |
| Q6c | relaxation différentiable de la recherche Leech | 0 $ puis 7 $ | T → 0 bit-exact ; Q4b pleine +2 pp | aucun | après Q3 |
| Q6d | distillation KL bout-en-bout, PV-tuning | dizaines de $ | ≥ +6 pp apparié | aucun | hors plafond, go explicite |
| Q7 | composition du corpus, DCLM-edu, 3 graines | 15 $ | ≥ +3 pp STEM apparié | < +1,5 pp | après M1 et M3 |

Q3 paie l'encodeur K fois. Deux pistes n'y ont jamais été tentées : réutiliser la partition d'un
octade pour son complément (moitié des partitions paires en moins, *calculé*) et le SIMD `pulp`. Le
pré-amorçage est écarté, plafond oracle 1,37× pair et 1,07× impair (*mesuré*, `bin/encbench`,
2026-07-28) : la borne est trop lâche. Le profileur n'a jamais servi.

Les gates de Q6 et Q7 se lisent en MMLU-STEM apparié : la perplexité ne voit pas l'effondrement du
raisonnement. Tout gain racheté en octets tient dans un budget fixé d'avance, en b/param modèle
entier.

## 3. Dette et hygiène

- Tampons en attente d'ancrage. Au matin du 09-02, 28 tampons, 20 ancrés, 8 sans ancre Bitcoin
  (*mesuré*, [ots-etat-2026-09-02](mesures/ots-etat-2026-09-02.txt)) : m3-gptq2,
  vague2-gel-geometrie, protocole-piles-isolees-v2, le préreg A2/A3 du 08-31 et les quatre préregs
  A2 du 09-01. Trois de plus depuis : m2-attribution (71712e60), m1-hessienne-shrink (5a5e1027),
  m2b-v4bits (263ec52a).
- Deux tampons n'attestent plus de leur fichier, 08-10 et 08-11, réécrits par la passe
  d'anonymisation `01fdbe6`. La version attestée est irrécupérable.
- Le bucket HF n'a jamais été inventorié : 69 fichiers, 46,7 Go au 08-17 (*mesuré*, `hf buckets
  ls`). Un inventaire précède tout devis de re-run.
- `[workspace.lints.rust] unsafe_code = "forbid"` et `[lints] workspace = true` sur les cinq crates
  du cœur : `#![forbid]` dans `lib.rs` ne couvre pas les tests d'intégration.
- Compilation hôte des `.cuh` par `clang++` en CI, sur le modèle de `llvq-cuda/tests/host_e1v.cpp`.
  `ci.yml` ne la porte pas.
- `ops/status.py` à écrire : il génère `docs/ETAT.md` (compteurs de `mesures/`, `jobs.csv`,
  `otsaudit`, config servie) et un test CI échoue sur un compteur périmé.
- `docs/exp-piles-isolees-2026-08-30/MACHINES.md:50-52` donne encore `ROT_SHARE=0 FUSE=0` comme
  config publiée ; à aligner sur v1.
- Aucun tag ne pointe sur le commit déposé `e21a8bb` ; `v0.0.1` (2026-08-26) pointe sur son enfant
  direct `16c9c8b` et le contient (*mesuré* le 09-02, `git tag --contains`). Tampon dû.
- `docs/hf-model-card.md` porte 5,162 b/param depuis le 08-17 ; la carte en ligne sur le Hub n'a pas
  été republiée depuis et diverge. Republier : décision d'opérateur.

## 4. En pause

- MoE. Modèle tranché Qwen3-30B-A3B. Il manque une politique pour les experts sous rang plein : 31,4
  % des cellules (couche, expert), un expert mort, mesuré sur gpt-oss-20b, un plancher pour le
  30B-A3B (*mesuré*,
  [moe-routing-gptoss20b-2026-08-12](mesures/moe-routing-gptoss20b-2026-08-12.txt)). P2 vaut ~1,4 $
  et P6 ~69 $ (*estimé*).
- Cache KV q8 à contexte long. Qualité verte à contexte court, +0,049 % de ppl et +0,33 pp de MMLU,
  IC contenant zéro (*mesuré*, [kvq8-4b-2026-08-15](mesures/kvq8-4b-2026-08-15.txt)). Le débit long
  n'est pas mesuré : la série n_new = 1024 a dépassé son plafond, 661 s contre 600 (*mesuré*, même
  journal). Réouverture sur un banc à modèle résident seulement.
- Batch M > 1 et prefill. Batch 1 assumé depuis le 08-18, régime edge et souveraineté. Le
  transcodage paresseux redevient exact à M ≥ 8 (*calculé*,
  [audit-recherche-2026-09-01](archive/audit-recherche-2026-09-01.md)) : le format optimal dépend de
  M, à reprendre si le prefill est servi.
- Famille k. `planes14k`, k dans {1, 2, 4, 8}, `TILE_BLOCKS_K = 32`, bras `nullk`, `mvkf16`,
  `cublasf16` : non écrite. Le préreg
  [preregistration-p4-2026-08-14](../proofs/preregistration-p4-2026-08-14.md) n'est pas tamponné ;
  son §7bis reste à remplir (deux dérogations, et le run `nullk` du 08-16 était-il un job P4). K2 se
  lit `T(k=8) ≤ 4,80·T(k=1)`. Job mutualisé 0,8 à 1,0 $, pire cas 2,70 $ (*estimé*), `--timeout
  90m`. Un verdict k ne se transporte pas au débit interactif (k = 1) ; un banc k qui ignore
  `ROT_SHARE`/`FUSE` mesure un chemin qui n'est plus servi.
- Point 32B. ~62 $ et 11,4 h sur `rtx-pro-6000x2` (*estimé* sur 621 s par bloc, *mesuré* au
  dé-risquage ci-dessous, sans journal ; budget 80 $ avec marge). Gate à formuler sur la chute
  d'écart 14B → 32B avec son z ; AWQ 32B officiel à vérifier. Le chemin servi y est muré par 1 024 octets de shared
  memory, rotation du `down_proj` (*mesuré*,
  [rot-partagee-14b-2026-08-17](mesures/rot-partagee-14b-2026-08-17.txt)). Dé-risquage du
  2026-08-03 : 4 blocs sur 64, bf16, 59 min, 5,43 $ (*mesuré*). Pic `faer` 70,6 Go hôte sur 512 et
  77,4 Go VRAM sur 97 à n = 25 600 (*mesuré*). `verify_artifact` bit à bit sur 1 950 351 360 poids
  (*mesuré*). C3 (chargement bf16) est un prérequis : 131 Go f32 ne tiennent pas dans 96 Go, sinon
  `h200x2` à ~180 $ (*calculé*).
  Profil : encodeur 71,8 %, factorisation 16,5 %, ~1,9 h de Cholesky en n³ (*mesuré*). Le coût par
  poids monte de 4,77e-5 cœur-s au 8B à 6,36e-5 au 32B (*mesuré*). Le bloc prédit à ~500 s
  (*estimé*) en a coûté 621 (*mesuré*). Un encodeur ×1,5 ramène le run à ~40 $ (*estimé*) et
  compose sur tous les runs suivants.

## 5. Décisions attendues

| décision | échéance | défaut si silence |
|---|---|---|
| M2b : quelle ligne du §5 s'applique à G4 = +3,60 [1,47 ; 5,79] | avant tout noyau Q5 | aucune ligne, Q5 n'ouvre pas |
| Q1 : préreg avec « ρ dans [0,5 ; 0,9] à ré-estimer », taille et graines | avant le premier run Q1 | Q1 reste à 0,6B, 3 graines |
| plafond de la vague 2 | avant le premier job payant | aucun job carte |
| `ots upgrade` des onze tampons en attente | après ancrage | tampons non upgradés dans le dépôt |
| format v2, `codebook_fingerprint` change | à F1b vert | F1 s'arrête au banc gaussien |
| première piste Q qui reçoit le run 4B (7 $) | D2, mi-octobre | meilleur Δppl par étendue à 0,6B |
| go Q6d, hors plafond | D4, décembre | non |
| F1e passé : second papier ou révision | D4, décembre | second papier |
| venue suivante du papier | libre | préprint seul |
| point 32B : go budget et gate ancré | après D4 | non lancé |
| benchmark métier d'extraction documentaire ([arXiv:2607.08734](https://arxiv.org/abs/2607.08734)) et CSR, jamais faits ; CSR bloqué en amont, tâches non transcrites | libre | non faits |

D1 fin septembre, D2 mi-octobre, D3 mi-novembre, D4 décembre (*estimé*).
