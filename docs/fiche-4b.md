# Fiche 4B

Registre de provenance de `Qwen3-4B-LLVQ-2bit` et de son noyau fusé : chaque chiffre du fichier publié y a sa ligne, avec
source, instrument, dtype, protocole et étiquette (*mesuré*, *calculé*, *estimé*). Là où un document diverge de l'objet,
l'objet gagne. État courant : [ETAT.md](ETAT.md) ; verdicts datés : [HISTORIQUE.md](HISTORIQUE.md) ; règles de mesure :
[METHODE.md](METHODE.md) ; layouts CUDA : [format-noyau.md](format-noyau.md). Mesures Metal sur MacBook Pro Mac15,8, M3 Max,
16 cœurs CPU, 40 cœurs GPU, 68 719 476 736 o (*mesuré*, `system_profiler`), crête 400 Go/s (*estimé*, spec) ; CUDA sur L40S.

## 1. Identité
| champ | valeur | étiquette, source |
|---|---|---|
| nom | `Pier-Jean/Qwen3-4B-LLVQ-2bit`, fichier `qwen3-4b-llvq.bin` | dépôt HF au commit `f00daa7bc1dd12a720304a4483f2219d10f15c96` |
| taille | 1 770 527 533 o (1,771 Go) | *mesuré*, `shasum` ; `content-length` HF identique (2026-08-03) |
| sha256 | `9db213ef9fa9d7d7000789a8a529ce9459ce9ba6002ef5a72fd5a1c05c1c84b0` | *mesuré*, identique au `x-linked-etag` HF |
| magic | `LVQ2`, trois sections, parse à l'octet exact | *mesuré* |
| scellement | 2026-07-31 17:56 (mtime) | *mesuré* |
| binaire | commit `51d7c55` (2026-07-31 12:36:19) | *mesuré*, quatre indices (§4) |
| copie locale | `/Users/pjmalandrino/qwen3-4b-llvq.bin` | même sha256 |

Le dépôt HF contient `.gitattributes`, `LICENSE`, `README.md` et le `.bin` ; reproduction par `shasum -a 256` et
`curl -sIL https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit/resolve/main/qwen3-4b-llvq.bin`. Le fichier se charge sans
réseau ni checkpoint (`bin/run`, `bin/ppl`, `bin/mmlu`, `bin/fusedrun`, bancs `thesis`, `matvec`, `decreal`) et ne s'ouvre
dans aucun autre moteur. Le widget HF ne le sert pas : `config.json` et `tokenizer.json` sont dans le `.bin`.

## 2. Contenu octet par octet
| section | octets | contenu | étiquette |
|---|---|---|---|
| matrices | 980 790 202 | 252 matrices quantifiées | *mesuré* |
| tenseurs bruts | 778 313 898 | 146 tenseurs f16 | *mesuré* |
| blobs | 11 423 433 | `config.json` 726 o, `tokenizer.json` 11 422 654 o | *mesuré* |
| total | 1 770 527 533 | = taille du fichier, écart 0 | *mesuré* |

La section matrices se décompose en payload 980 770 752 o (7 846 166 016 bits, §5.3) et framing 19 450 o (*calculé* sur entrées
mesurées, boucle à l'octet). Le framing compte l'en-tête 8, les noms 10 370, les métadonnées 252 × 28 et les préfixes 252 × 8.

| tenseurs portés | valeurs | étiquette |
|---|---|---|
| `model.embed_tokens.weight` [151936, 2560] | 388 956 160 | *mesuré* |
| normes : 36 × (2560 + 2560 + 128 + 128) + `model.norm` 2560 | 196 096 | *mesuré* |
| total porté, 146 tenseurs | 389 152 256 | *mesuré* |

Pas de `lm_head` : `tie_word_embeddings` vaut `true`. Les 146 tenseurs sont égaux bit pour bit à f16(bf16 du checkpoint),
146/146 vérifiés (*mesuré*). Les blobs sont des copies octet pour octet du checkpoint : sha256 de `tokenizer.json` =
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`, le blob HF amont. Tokens et prompts MMLU sont donc
identiques entre les deux bras par construction. `config.json` : hidden 2560, intermediate 9728, 36 couches, head_dim 128,
32 têtes, 8 KV, vocab 151 936, bfloat16 (*mesuré*). La constante `389_070_848` de `thesis.rs:432` ne correspond à aucun
tenseur (2¹⁴ × 23 747), effet +0,03 % sur les tok/s : à corriger, jamais à citer.

## 3. Comptes de poids
Relus dans les en-têtes de matrice (*mesuré*, 2026-08-03).

| grandeur | valeur | note |
|---|---|---|
| matrices | 252 | 36 blocs × 7 projections |
| poids de projection | 3 633 315 840 | fichier et `~/llvq-run-4b-artefact.log` |
| dont quantifiés | 3 616 358 400 | |
| dont queue `KeepExact` | 16 957 440 (0,4667 %) | 471 040 par couche |
| blocs de 24 | 150 681 600 | relus, pas déduits |
| lignes de sortie | 1 105 920 | les lignes vérifiées par le banc noyau ; aussi le nombre d'échelles de ligne |
| centroïdes de gain | 504 | 2 × 252 |
| blocs au niveau de gain 0 | 72 008 871 (47,79 %) | centroïde moyen 0,8723 |
| blocs au niveau de gain 1 | 78 672 729 (52,21 %) | centroïde moyen 1,1146 ; 0 bloc codé à l'origine |
| total paramètres du modèle | 4 022 468 096 | 3 633 315 840 + 389 152 256 (*calculé*) |

Formes : q 4096 × 2560, k 1024 × 2560, v 1024 × 2560, o 2560 × 4096, gate et up 9728 × 2560, down 2560 × 9728. Queues :
2560 % 24 = 16, 4096 % 24 = 16, 9728 % 24 = 8. Par matrice, la fraction de blocs au niveau 1 va de 0,4660 à 0,7604,
médiane 0,5143 ; les centroïdes sont strictement croissants sur les 252, rapport moyen 1,2791. `~/llvq-q4b.llvq`
(980 790 202 o, magic `LVQ1`) porte aux octets [8, 980 790 202) le même sha256 que le scellé,
`5acd89c07afc143ce12ab5a04a4a24ba38f8bd7f0601d049e14e734715725a6b` (*mesuré*) : `bin/seal` ré-encode bit-identique. Son
sha256 propre est `94f60e86…` ; c'est le fichier par défaut des trois bancs Metal (`thesis.rs:191`, `matvec.rs:503`,
`decreal.rs:139`), donc l'objet des runs `thesis` du 08-01.

## 4. Configuration
| réglage | valeur livrée | ce que fait le code | preuve |
|---|---|---|---|
| codebook | `leech1c12` | `LeechShapeGain::with_caps(centroids, cap = 12, level_cap = 5)` | *mesuré* : `shell_cap = 12` sur 252/252 |
| plafond de niveaux | aucun | `MAX_LEVELS_ANY = 5`, le maximum structurel | *mesuré* : le jeton `L<n>` date du commit `fabab22`, 25 h après le scellement |
| index | 47 bits | `⌈log₂ N(12)⌉`, N(12) = 111 043 117 458 000 | *mesuré* : flux = nblocs × 6 o ; index max observé 111 043 117 450 038 |
| gain | 1 bit, 2 centroïdes par matrice | Lloyd-Max, 40 itérations, normes de bloc relatives, poids tournés | *mesuré* : 2 centroïdes sur 252/252 |
| bits par bloc | 47 + 1 = 48, 6 octets, MSB-first, sans bourrage | 2,000000 b/poids de code | *calculé*, exact |
| échelles de ligne | 1 105 920 en f64 | `row_scale = sqrt(Σ row² / (d_in/24))`, figée avant la boucle | *mesuré* : 0 sur 1 105 920 représentable en f32, 0/504 centroïdes |
| queue | `TailPolicy::KeepExact`, f32 sur disque | reçoit la rétroaction d'erreur, ne produit aucune erreur propre | *mesuré* |
| rotation | entrée seule, graine `0x110FEED` | `Q = (Q_odd ⊗ H_m) D`, graine `base ^ (bloc<<32) ^ (act<<16)`, 144 graines distinctes | *mesuré* : 252 graines reproduites sans exception ; aucun `rotate_weight_cols` |
| `group_scales` | off | arg 5 = `nogs`, et `ensure!(!cfg.group_scales)` à l'écriture | *mesuré* |
| rétraction | `true`, no-op | `retraction_target()` rend `None` sous `retract_to_level` | *mesuré* |
| amortissement | 1e-2, relatif à `mean(diag H)` | codé en dur au run | *mesuré* ; balayé au lot B, effet nul ([verdicts](archive/verdicts-lot-b-2026-08-06.md)) |
| dtype | f32 partout | `var_builder(DType::F32)` littéral ; `LLVQ_DTYPE` postérieur | *mesuré* |
| calibration | C4 validation shard 00000, 64 × 2048 = 131 072 tokens, préfixe contigu | `LLVQ_CALIB_SEED` n'existait pas | *mesuré* |
| threads d'encodage | 16 | valeur résolue, ligne 1 du log | *mesuré* |
| portée | 36 blocs sur 36, 252 matrices | | *mesuré* |

Ligne de commande qui a produit l'objet, puis scellement :
```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=/Users/pjmalandrino/llvq-q4b.llvq \
cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- \
  64 2048 12 4096 metal nogs leech1c12 999 rot
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- \
  /Users/pjmalandrino/llvq-q4b.llvq /Users/pjmalandrino/qwen3-4b-llvq.bin
```
Positionnels au commit `51d7c55` : 0 n_calib, 1 calib_len, 2 n_eval, 3 eval_ctx, 4 device, 5 teste `== "gs"`, 6 codebook,
7 limit, 8 `rot`, 9 absent. Le codebook `1c12` signifie gain 1 bit, coquille 12, pas de suffixe `f` ; toute valeur de
limit ≥ 36 équivaut. Quatre indices datent le binaire. Le mtime du `.llvq` moins 14 447 s place le démarrage vers 12:40 ;
les lignes « model dtype », « hessian damping » et « phases » sont absentes ; le compteur lit `block N/405`.
`--features fast-linalg` n'est pas traçable : le garde-fou qui l'imprime est postérieur au run. Faits sur la recette :
- La recette livrée est l'Algorithme 1 (shape-gain, reset de gain) plus une rotation d'incohérence en entrée ;
  « Spherical GPTQ » nomme le crate, pas la recette.
- La ligne de configuration des logs de `smoke` est un littéral codé en dur (« 0 gain bits, spherical retraction ») ;
  seule la ligne de résultat `leech1c12` est fiable.
- La rétraction de l'Eq. 17 est un no-op sous un gain codé : `quantize` a déjà posé le bloc sur la sphère du niveau.
- L'Algorithme 3 (`refine_group_scales`) est doublement désactivé.
- `block N/405` compte les 405 blocs de colonnes de `down_proj` (9728 / 24), pas des couches ; `/36` avant `51d7c55`.

La commande publiée reproduit la méthode sans reproduire les octets. Deux blocages (*mesuré*, git) :

| blocage | fait | conséquence |
|---|---|---|
| corpus | le commit `aba3989` (2026-08-01) déplace `LLVQ_CALIB=c4` du shard 00000 au 00001 | la commande publiée calibre à HEAD sur un autre texte ; aucune ppl C4 de l'objet n'est produisible sans contamination |
| conteneur | à `51d7c55` l'écrivain était `artifact2.rs`, magic `LVQ1`, `finish()` vide ; à HEAD `ArtifactWriter`, magic `LVQ2`, deux `u32` nuls | un re-run rend 980 790 210 o avec un autre magic ; les enregistrements de matrice restent comparables |

Un tiers sur CUDA n'obtiendra pas les mêmes poids : `calib.rs` accumule AᵀA en f32 sur l'accélérateur, écart non chiffré.

## 5. Les chiffres
### 5.1 Perplexité
Wikitext-2 test, ctx 4096, 12 fenêtres non chevauchantes, 49 140 tokens notés (4 095 × 12), logits f32 avant `log_softmax`.

| LLVQ | baseline | × | objet | dtype | instrument | trace | étiquette |
|---|---|---|---|---|---|---|---|
| 16,9617 | 12,2336 | 1,3865 | modèle en mémoire, avant et après réécriture des 252 projections | f32 | boucle `ppl` de `smoke` | `~/llvq-run-4b-artefact.log` | *mesuré* ; iso-conditions par construction, `verify_artifact` rattache ce modèle aux octets publiés |
| 16,9415 | 12,2361 | 1,3845 | octets publiés contre checkpoint, empreinte `3f1baca9033bf251` des deux côtés | f16 | `bin/ppl`, Metal | corps du commit `8c17eff` ; rejoué `~/ppl-scelle-f16-2026-08-04.log`, `~/ppl-base-f16-2026-08-04.log` | *mesuré*, reproduit au dix-millième |
| 16,9422 | 12,2369 | 1,385 | octets publiés, L40S, même empreinte | f16 | `bin/ppl`, CUDA | [a4-campagne](mesures/a4-campagne-2026-08-06.txt) | *mesuré* ; NLL par fenêtre dans [le brut](mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt) |
| 16,9358 | 12,2369 | 1,384 | variante embedding q8 (`q4b-e8.llvq`) | f16 | `bin/ppl`, CUDA | LLVQ : [campagne finale bras 4](mesures/campagne-finale-bras4-2026-08-07.txt) ; baseline : a4-campagne, même empreinte | *mesuré* des deux côtés ; le × est *calculé* entre les deux journaux |
| 15,3272 | | | overlay `~/llvq-q4b-c12.safetensors`, run de nuit, binaire antérieur au correctif `60068db`, 2,6923 b/poids réels, quantifieur du commit `db84454` | f32 | `smoke` | `~/llvq-run-nuit.log` | *mesuré*, hors référence |
| 14,2684 / 15,2909 / 14,9104 | 12,2336 | | modèles en mémoire, cap 13, annoncés à 2,1117 b/poids, débit réel 2,7338 | f32 | `smoke` | lignes de tableau, aucun log | *mesuré*, hors référence ; le couple 14,2684 / 15,2909 est l'observation de dispersion du §5.7 |

Invocations du couple f16, critère = même empreinte `3f1baca9033bf251` : `LLVQ_DTYPE=f16 cargo run --release -p llvq-llm
--features metal --bin ppl -- 4096 12 metal /Users/pjmalandrino/qwen3-4b-llvq.bin` (scellé) et `LLVQ_MODEL=Qwen/Qwen3-4B
LLVQ_DTYPE=f16 … --bin ppl -- 4096 12 metal` (checkpoint) ; attendu 16,9415 / 12,2361.

Surcoût de log-vraisemblance sur sa propre baseline (*calculé*), la seule comparaison inter-papier qui tienne :

| bras | Δ nats/token | vs QTIP |
|---|---|---|
| nous, f32 | ln(16,9617) − ln(12,2336) = 0,326772 | +3,06 % |
| nous, f16 sur le fichier | ln(16,9415) − ln(12,2361) = 0,325376 | +2,62 % |
| QTIP (17,04 / 12,41) | 0,317061 | |
| LLVQ 0 bit du papier (17,05 / 12,41) | 0,317648 | +0,19 % |

Nous sommes au-dessus de QTIP et de la config 0 bit du papier avant de payer 8,5 % de bits en plus ([notes](llvq-paper-notes.md)).

### 5.2 MMLU
Hendrycks 5-shot, dev de la même matière, logits des tokens `" A".." D"` comparés en f32, une passe avant par question.
2 280 questions sur 14 042 (40 par matière, 57 matières), tirage seedé `SplitMix64(0x6_11B0 ^ subject.len())` aux deux bras.

| bras | micro | macro | carte | dtype | trace | étiquette |
|---|---|---|---|---|---|---|
| f16 checkpoint | 70,32 ± 1,28 | | L40S | f16 | [a4-campagne](mesures/a4-campagne-2026-08-06.txt), empreinte `65dcd53655e8bfa5` | *mesuré*, référence |
| LLVQ, octets publiés | 55,59 ± 1,35 | | L40S | f16 | idem | *mesuré*, référence |
| LLVQ, embedding q8 | 55,70 ± 1,35 | | L40S | f16 | [campagne finale bras 4](mesures/campagne-finale-bras4-2026-08-07.txt) | *mesuré* ; dans le bruit |
| chute f16 → LLVQ | −14,73 pp, IC95 apparié [+11,98 ; +17,47] | | | | [mmlupair 4B/8B](mesures/mmlupair-4b-8b-2026-08-13.txt) | *calculé* ; papier −9,5 pp (60,7 / 70,2) |
| f16 checkpoint | 70,42 ± 1,28 | 72,85 | Metal | f16 | [mmlu-micro-2026-08-02.log](mmlu-micro-2026-08-02.log), 2 620 s | *mesuré*, hors référence |
| LLVQ, octets publiés | 56,09 ± 1,36 | 57,59 | Metal | f16 | même log, 2 805 s, profil par matière | *mesuré*, hors référence |
| chute, Metal | −14,33 pp | −15,26 pp | | | | *calculé*, hors référence |

« Micro » est un estimateur stratifié : les 57 taux sont repondérés par la population réelle de chaque matière.
`professional_law` pèse 1 534 / 14 042 = 10,9 %, estimé sur 40 tirages. Le ± est une erreur-type stratifiée à 1 σ, pas
un IC 95 % ; il exclut modèle, prompt et graine. La barre d'une différence est l'appariée (McNemar) : 0,43 pp à fichier
constant, 0,79 à 1,44 pp entre modèles (*mesuré*, [KV q8](mesures/kvq8-4b-2026-08-15.txt), mmlupair 4B/8B). Validation
du harnais : 70,42 contre 70,2 au papier, +0,22 pp, 0,17 σ (*calculé*). Le bras quantifié perd 0,50 pp entre Metal et
L40S sur le même fichier, écart non vérifiable (log Metal antérieur aux empreintes). Profil (Metal) : `abstract_algebra`
10/40, `professional_accounting` 10/40, `machine_learning` 12/40 ; `international_law` 33/40 ; barre par matière ±7 pp.
Invocations Metal : `cargo run --release -p llvq-llm --features metal --bin mmlu -- Qwen/Qwen3-4B metal 40` et
`… --bin mmlu -- /Users/pjmalandrino/qwen3-4b-llvq.bin metal 40` ; le `40` est la limite par matière qui définit les
2 280 questions, et le `.bin` en positionnel prouve que le bras quantifié score les octets publiés.

### 5.3 Taille et débit
Payload : 7 846 166 016 bits = 980 770 752 o (*calculé*). Somme : 150 681 600 × 48 + 1 105 920 × 64 + 504 × 64 + 16 957 440 × 32.

| dénominateur | b/poids | qui l'imprime | où il est publié | usage |
|---|---|---|---|---|
| 3 633 315 840 (projections, queue incluse) | 2,159506 | `bin/seal` | carte HF | le chiffre homogène, à afficher en nommant le dénominateur |
| 3 616 358 400 (quantifiés seuls) | 2,169632 | `bin/smoke`, ligne « artifact: » | README, CLAUDE.md | variante conservatrice, ratio mixte |
| comptabilité idéale (queue f16, échelles f16) | 2,070226 | `Report::bits_per_weight`, « effective rate » | | ne se cite jamais pour ce fichier : il décrit un fichier non écrit |

Décomposition sur 3 616 358 400 (*calculé*, boucle au septième chiffre) :

| poste | b/poids |
|---|---|
| code de réseau, 48 b/bloc | 2,000000 |
| queue en f32 | 0,150051 |
| échelles de ligne en f64 | 0,019572 |
| centroïdes f64 | 0,0000089 |
| total | 2,169632 (+8,48 % sur 2,000) |
| si queue et échelles étaient f16 | 2,0799 (+4,0 %) |

Le gisement est la queue f32 → f16, 0,075026 b/poids ; les échelles f64 ne sont pas réductibles sans casser la preuve bit
pour bit. `format.rs` documente 0,0146 (surcoût f64 sur f16), le README 0,020 (coût total) : deux conventions, aucune fausse.

| compression | valeur | étiquette |
|---|---|---|
| fichier | 1 770 527 533 o | *mesuré* |
| FP16 équivalent | 4 022 468 096 × 2 = 8 044 936 192 o | *calculé* |
| ratio | ×4,5438 | *mesuré*, imprimé par `bin/run` et `bin/seal` |
| débit du modèle entier, embedding f16 | 3,5213 b/param | *calculé* |
| `q4b-e8.llvq`, embedding int8 | 1 405 881 733 o, 2,7961 b/param | *mesuré* ; −0,02 % de ppl, en production ; section matrices bit-identique au scellé (format `LVQ3` d'époque), ce qui rattache 16,9358 et 55,70 aux mêmes 252 projections que 16,9415 et 55,59 |
| `q4b-e4.llvq`, embedding int4 g64 | 1 211 403 653 o, 2,4093 b/param | *mesuré* ; +1,52 % de ppl, non publié ; section matrices bit-identique au scellé ([verdicts lot B](archive/verdicts-lot-b-2026-08-06.md) §B4) |

### 5.4 Coût de production
| grandeur | valeur | trace | étiquette |
|---|---|---|---|
| durée du run | 14 447 s = 4,013 h | `~/llvq-run-4b-artefact.log`, « quantized 252 matrices … in 14447s » | *mesuré* |
| par couche | min 328 s, max 592 s, moyenne 401 s | 36 lignes horodatées | *mesuré* |
| threads | 16 | ligne 1 du log | *mesuré* |
| coût | 0 $, M3 Max local | | *mesuré* |
| profil par phase | aucun | l'instrumentation date du 2026-08-02 | non mesuré |

Deux durées retirées désignent deux autres runs. « ~3,5 h » est le run de nuit : 12 715 s, `~/llvq-run-nuit.log`,
`leech1c12` sur un binaire antérieur à `60068db`, 2,6923 b/poids réels. « 3,45 h » est le run cap 13 : 2,1117 b/poids
annoncés, 2,7338 réels. Seul 14 447 s = 4,013 h est le run publié ; il coûte +13,6 % sur le run de nuit (*calculé*)
parce qu'il encode les index dans la boucle et écrit en flux.

### 5.5 Preuve d'aller-retour
`3633315840 weights identical, bit for bit` (*mesuré*, `~/llvq-run-4b-artefact.log`) : comparaison `to_bits()` sur les
252 matrices, f32, queue comprise, après `decode_matrix` ; transportée aux octets publiés par l'identité des sections (§3).
Elle ne couvre pas un run natif f16 ; le narrowing étant déterministe, `f16(decode(fichier))` est le bras MMLU.

### 5.6 Débit servi sur ces octets
| configuration | tok/s [plage] | Go carte | carte | source | étiquette |
|---|---|---|---|---|---|
| v1 servie : `planes14` + q8 + `ROT_SHARE=1` + `FUSE=1` | 100,6 [99,9–100,7] | 2,57 | L40S | [D1](mesures/d1-fusion-servie-2026-08-24.txt), [vague 2](mesures/vague2-fusion-8b-14b-2026-08-31.txt) | *mesuré*, médiane de 5 rounds |
| `planes14` + q8, `ROT_SHARE=1/FUSE=0` (hissage seul) | 94,9 [94,1–95,2] | | L40S | D1 | *mesuré* |
| `planes14` + q8, `ROT_SHARE=0/FUSE=0` | 87,0 [86,8–87,0] | 2,56 | L40S | [B2](mesures/b2-fusedrun-plages-2026-08-18.txt) | *mesuré* |
| `planes12x` + q8 | 85,0 [84,7–85,1] | 2,36 | L40S | [G3](mesures/g-horloges-planes12x-2026-08-23.txt) | *mesuré* ; −2,3 % de débit pour −0,20 Go |
| `planes14`, embedding f16 (tête identique) | 48,3 [48,1–48,3] | 2,93 | L40S | B2 | *mesuré* |
| dense f16, notre chemin | 43,5 [43,4–43,5] | 8,04 | L40S | B2 | *mesuré* |
| `bin/run` avec cache KV, décodé en mémoire | 42,7 | | L40S | [mini](mesures/mini-2026-08-05.txt) | *mesuré* |

Le rapport à tête identique, ×1,11 [1,11–1,11], mesure le noyau. Le brut ×2,00 [1,99–2,00] ne se publie jamais seul : le dense
recopie 778 Mo de vocabulaire par token (*mesuré*, [phases](mesures/phases-2026-08-07.txt)). Fusion : ×1,061 [1,050–1,069]
intra-job (*mesuré*, D1) ; divergence au dense au token 89 sur 128 sous chaque bras fusé, et 128 tokens identiques
entre les deux bras fusés (F1 et F0).

### 5.7 Barres d'erreur
| barre | valeur | ce qu'elle couvre | source | étiquette |
|---|---|---|---|---|
| σ de calibration, ppl | 5,2 % (0,8202 ppl), étendue 10,3 % sur 16,7425 / 15,8836 / 15,1027 | trois runs complets du 4B, graines 1/2/3, 21,45 $ | [F5](mesures/f5-graines-4b-2026-08-19.txt) | *mesuré* ; les trois paires appariées résolues (t +4,54 / +10,92 / +7,68) |
| σ de calibration, MMLU | 2,92 pp, étendue 5,83 pp sur 58,02 / 52,19 / 55,17 | mêmes trois artefacts | [bruit MMLU](mesures/bruit-mmlu-graines-4b-2026-08-25.txt) | *mesuré* |
| excès LLVQ sur f16, ppl | +38,45 % [+33,62 ; +43,45] | échantillonnage du corpus, apparié fenêtre par fenêtre, 12/12 | [ppl appariée](mesures/ppl-appariee-4b-2026-08-17.txt) | *calculé* sur NLL mesurées |
| A/B à fichier constant | ±0,12 % ppl, SE 0,43 pp MMLU | intervalle apparié ; ne porte pas le σ de calibration | [KV q8](mesures/kvq8-4b-2026-08-15.txt) | *mesuré* |
| observation n = 2 | 14,2684 contre 15,2909, écart 7,2 % | même quantifieur (écart 7,1e-15, test `under_the_old_retraction_shape_gain_was_direction_only`), cause non tranchée | lignes de tableau, aucun log | *mesuré* |
| σ de 0,7 % (0,15 ppl) | 3 blocs de Qwen3-0.6B, lot B | pas la taille publiée : facteur 7 sous F5 | [verdicts lot B](archive/verdicts-lot-b-2026-08-06.md) | *mesuré* |

Le fichier publié est un tirage d'un processus à graine, calibré sur le shard 00000 d'avant ; il n'est pas un quatrième
tirage. Les 0,08 point sous QTIP (16,9617 contre 17,04) sont sous la dispersion mesurée et ne se revendiquent pas.

### 5.8 Coût du bit de gain
Coder le gain coûte +3,17 % de perplexité pour −0,618 b/poids (*mesuré*, `~/llvq-ab-retraction.log`, 2026-07-31). A/B :
Qwen3-0.6B, 3 blocs, ctx 2048, 12 fenêtres, baseline 19,5038.

| bras | codebook | b/poids | ppl | × |
|---|---|---|---|---|
| A | `leech1c12`, gain porté (47 + 1 = 48 b/bloc) | 2,1656 | 21,4157 | 1,098 |
| B | `leech1c12f`, magnitude f16 libre (47 + 16 = 63 b/bloc) | 2,7838 | 20,7582 | 1,064 |

Réserves : 3 blocs d'un 0.6B, et le suffixe `f` ne restaure que la magnitude libre. D'où un écart 4B (10,7 %, *calculé*
sur 16,9617 contre 15,3272, §5.1) plus grand que l'écart 0.6B (3,2 %, *calculé* sur le tableau ci-dessus). Le gate à
28 blocs du 0.6B rend, à 2,1656 b/poids, `leech0c13` 39,3309, `leech2c11` 39,5350, `leech1c12` 43,4865, `leech4c10`
47,1537 (graine 0) (*mesuré*, [gate gain](mesures/gain-ab-gate-0.6b-2026-08-25.txt)). La graine 1 inverse le classement.

### 5.9 Manques et points non tranchés
Sur les dix-sept manques relevés le 2026-08-03, dix sont mesurés depuis (dates dans [HISTORIQUE.md](HISTORIQUE.md)),
cinq restent (table ci-dessous), un était documentaire (`Cargo.toml`, README, commande ppl : 0 machine) et un n'a jamais
été fait (débit et RSS du q4 MLX rejoués et loggués, ~2 min, devenu secondaire depuis l'AWQ).
Mesurés : stdout des deux ppl f16 ; journal de `thesis` ([témoin](mesures/thesis-temoin-2026-08-04.txt), K1) ; ppl et MMLU
du 4 bits (AWQ, §7) ; σ à la profondeur publiée (F5). Mesurés aussi : plafond L ≤ 4 (+4,75 % de ppl, *mesuré*,
[verdicts lot B](archive/verdicts-lot-b-2026-08-06.md)) ; amortissement (20,6740 / 20,6643 / 20,6014, lot B) ; `Grouped32`
et `Flat32` sur le modèle entier (K1) ; rotation GPU et noyau branché sur CUDA. Restent :

| point | état |
|---|---|
| écart de ppl entre shard 0 et shard 1 de calibration | non mesuré ; 2 runs de 3 blocs, ~50 min (*estimé*) |
| `bin/seal` rejoué avec sa sortie | attendu 2,1595, 1,771 Go, fichier non identique (`LVQ2` + deux `u32`), ~10 min |
| `k = 1` dans `llvq-bench` (`main.rs:109` boucle sur `[0, 2]`) | la ligne « union + 1 bit de gain » du tableau aux auteurs est recopiée de la Table 8 |
| rétention du tableau aux auteurs | `retention_pct(mse, rate) = 100·(−½·log₂ mse)/rate` : sur la MSE arrondie 0,078 le banc imprime 92,01, le papier 92,14 sur sa MSE non arrondie (≈ 0,077718, SQNR 1,843) ; citer 92,14, ne jamais le recalculer depuis 0,078 |
| contrôle de déterminisme, deux runs identiques | tranche si les 7,2 % de l'observation n = 2 sont du bruit numérique ou une configuration non consignée ; ~8,5 h (*estimé*) |
| CSR | définition des tâches non transcrite ; 1 à 2 jours (*estimé*) |
| mécanisme du pic RSS Metal à 17,41 Go | *estimé* : double résidence hôte/buffer, ou pool de tampons candle-metal |
| gain incrémental de la rotation de sortie | la Table 9 du papier chiffre « aucune → Input+Output » (29,3 → 34,9), pas « Input → Input+Output » ; rien dans le code, bump de MAGIC requis |
| coût de `gain_bits = 0` au 4B | non mesuré ; l'A/B du §5.8 compare 1 bit à magnitude libre |
| écart CUDA contre Metal des poids produits | AᵀA accumulé en f32 sur l'accélérateur ; non chiffré |

## 6. Face au FP16
| couple | iso-conditions | comment |
|---|---|---|
| ppl f32, 12,2336 / 16,9617 | garantie par construction | `smoke` tokenise une fois, un seul `test_ids`, une seule fermeture `ppl`, un seul objet modèle avant et après réécriture ; aucune empreinte à comparer |
| ppl f16, 12,2361 / 16,9415 | empreinte `3f1baca9033bf251` identique | attendue par construction, le tokenizer scellé étant byte-identique au checkpoint |
| MMLU, 70,32 / 55,59 | même binaire, même session, même dtype imprimé, mêmes 2 280 questions, même tokenizer, empreinte `65dcd53655e8bfa5` | le log Metal du 08-02 précède l'impression des empreintes |

Écart de protocole non contrôlé en f32 (*mesuré*) : le checkpoint est bf16, `seal` écrit les tenseurs portés en f16. Sur
les 388 956 160 valeurs de l'embedding, 77 045 changent (1,98·10⁻⁴), 451 tombent à zéro, toutes sous 7,600·10⁻⁶ ; max |v|
0,250, erreur absolue max 2,98·10⁻⁸. L'embedding étant le `lm_head`, l'écart entre dans les logits. À f16 les deux bras
convergent, MMLU et ppl f16 sont propres. À f32, une ppl du fichier scellé ne se compare pas à la baseline du checkpoint
sans le dire. Le couple 12,2336 / 16,9617 tourne en mémoire et n'est pas concerné. Résidu `from_mmaped_safetensors` contre
`from_vec(f32).to_dtype()` : *estimé* négligeable. Le « FP16 » du banc n'est pas ce FP16 (§8.2).

## 7. Face au 4 bits
L'adversaire retenu est l'AWQ officiel de Qwen (décision du 2026-08-06), mesuré dans le même harnais à la même empreinte
([a4-campagne](mesures/a4-campagne-2026-08-06.txt)). Le MLX q4 reste l'objet de la comparaison disque locale ; IQ2_XXS : [ETAT.md](ETAT.md) §3.

### 7.1 Sur le disque
| objet | valeur | étiquette |
|---|---|---|
| MLX q4, `/Users/pjmalandrino/qwen3-4b-mlx-q4/` | `model.safetensors` 2 263 022 417 o, répertoire 2 274 510 217 o | *mesuré* |
| recette | `mlx_lm.convert --hf-path Qwen/Qwen3-4B -q --q-bits 4 --q-group-size 64` | *mesuré*, `config.json` |
| structure | 904 tenseurs : 253 U32, 253 `.scales`, 253 `.biases` bf16, 145 normes | *mesuré* |
| embedding | quantifié aussi (253 = 252 projections + `embed_tokens`) | *mesuré* ; nous le portons en f16 ou en q8 |
| débit | 4,500000 b/poids sur les poids quantifiés, 4,500561 tous poids | *calculé*, exact |
| total | 4 022 468 096 poids des deux côtés | *calculé* |
| AWQ w4 g128 officiel | 2,67 Go, 5,302 b/param dans son moteur | *mesuré* / *calculé* ([rtbits](mesures/rtbits-planes-8b-2026-08-09.txt)) |

### 7.2 Axe par axe
| axe | LLVQ | 4 bits | verdict | étiquette |
|---|---|---|---|---|
| disque | 1 770 527 533 o, 3,5213 b/param (1,41 Go en q8) | MLX q4 2 263 022 417 o, 4,5006 ; AWQ 2,67 Go | ×1,2782 pour nous ; projections seules 2,1595 contre 4,5000, ×2,084 | *mesuré* des deux côtés |
| VRAM, b/param modèle entier | 5,162 (`Planes14` + q8) ; 4,745 (`Planes12x` + q8) | AWQ 5,302 dans son moteur ; MLX q4 4,50 | sous l'AWQ réel de 2,6 % | *calculé* sur octets mesurés (rtbits) |
| débit | ×1,11 [1,11–1,11] à tête identique chez nous | ×2,413 [2,412 ; 2,414] pour l'AWQ dans vLLM (200,49 tok/s contre 83,09 f16) | deux piles, deux témoins, aucun quotient licite | *mesuré*, [vLLM](mesures/awq-vllm-4b-2026-08-17.txt) |
| ppl wikitext | ×1,385 | AWQ ×1,105 (13,5207 / 12,2369) | excès 0,385 contre 0,105, rapport 3,7 (*calculé*) | *mesuré*, a4-campagne |
| MMLU micro | 55,59 (−14,73 pp) | AWQ 70,04 ± 1,25 (−0,28 pp, non résolu [−1,63 ; +2,13]) | écart apparié 14,45 pp [+11,60 ; +17,27] | *mesuré*, [mmlupair](mesures/mmlupair-4b-8b-2026-08-13.txt) |

Sur un 4B, le 4 bits domine partout sauf le disque ; l'écart MMLU à l'AWQ vaut 7,49 pp au 8B et 6,09 pp au 14B
([ETAT.md](ETAT.md) §3). Notre harnais charge l'AWQ déquantifié en f16 : ni son débit ni sa mémoire ne se lisent chez nous.

### 7.3 Les trois RAM
| quantité | valeur | étiquette |
|---|---|---|
| MLX q4, « 2,39 Go » | pic de l'allocateur MLX (poids + KV + activations), prompt inconnu | *estimé*, aucune trace |
| nous, « 3,28 Go » | arithmétique poids seuls du format `Slot32`, que `bin/run` ne charge jamais | *calculé*, hors sujet pour le runner |
| nous, modèle résident de `bin/run` | 4 022 468 096 × 2 = 8 044 936 192 o | *calculé*, exact par construction |
| nous, pic RSS de `bin/run` | CPU 9,79 Go (`cpu 12`) ; Metal 17,41 Go, reproductible à 0,0006 % sur 4 lancements, mécanisme inconnu | *mesuré*, `/usr/bin/time -l` |
| nous, carte L40S sous `fusedrun` | 2,57 Go (v1), 2,56 (`Planes14` + q8), 2,36 (`Planes12x` + q8), compte d'octets hôte | *mesuré* ; D1 et vague 2 (2,57), B2 (2,56), G3 (2,36) |

À convention poids seuls, `Slot32` + `lm_head` f16 vaut 6,5245 b/poids contre 4,5006 pour le q4, ×1,45 contre nous
(*calculé*).

### 7.4 Débit
| chiffre | inclut | exclut | étiquette |
|---|---|---|---|
| MLX 129,8 tok/s | 253 matmuls, attention, normes, RoPE, KV, lm_head, échantillonnage | prefill, chargement ; `--max-tokens 256`, prompt inconnu | *estimé*, aucune trace |
| AWQ 200,49 tok/s [200,39 ; 200,61], vLLM 0.26.0, L40S, batch 1, 128 tokens | tout, prefill compris | autre pile : ne se divise avec rien de chez nous | *mesuré* |
| `thesis` 10,46 ms | 252 matvec fusés, un token, mémoire froide | attention, normes, RoPE, KV, lm_head, rotation, transcodage | *mesuré*, Metal, 2 bras du 2026-08-01 ; le rapport à publier est celui de K1 (§8.4) |
| `thesis` 78,2 tok/s | ci-dessus + lm_head modélisé | idem ; jamais exécuté | *calculé*, borne supérieure |
| `bin/run` 2,2 à 7,6 tok/s, Metal f16 | tout, bout en bout | cache KV, noyau fusé | avant le cache KV (commit `9c24d26`) ; avec cache, 42,7 tok/s sur L40S (§5.6) |
| `fusedrun` v1 100,6 tok/s [99,9–100,7] | tout, bout en bout ; divergence au dense au token 89 sur 128 | | *mesuré*, L40S, D1 |

### 7.5 Régime où le 2 bits gagne
Un seul axe est démontré : le disque, ×1,278. Le créneau structurel est la fenêtre mémoire où le 4 bits ne rentre pas.
Elle vaut 12 à 21 % (*calculé*) : 4,50 / 3,727 = ×1,21 avec `Grouped32`, 4,50 / 4,034 = ×1,12 avec L ≤ 3. Recalcul 70B,
Llama-3.1-70B, 70,554 Md, embedding et `lm_head` non liés (2,978 %) laissés en f16 (*calculé*) :

| | q4 | `Slot32` | L ≤ 3 | `Grouped32` | disque |
|---|---|---|---|---|---|
| recalculé | 39,69 Go | 51,35 | 35,58 | 32,87 | 22,77 |
| [face-au-4-bits.md](archive/face-au-4-bits.md) | 39,4 | 48,2 | 32,1 | 29,3 | 19,0 |

Les chiffres d'archive appliquaient un débit projections seules à tous les poids, optimiste de 6 à 12 % pour nous. Quatre
inconnues restent. La qualité à 70B : aucun 70B quantifié. Le cache KV : 320 Kio/token en f16, 2,68 Go à 8k (*calculé*).
Le débit `Grouped32` servi. Le format rapide, `Planes14` à 4,804 b/poids noyau, plus gros que du 4 bits. Le triplet
produit (8k, 5 Go, 32 GiB) borne b_max à 3,00 b/poids noyau ([ETAT.md](ETAT.md) §6). Le plafond L ≤ 4 est mort en
qualité (+4,75 % de ppl, §5.9).

## 8. Le noyau fusé
### 8.1 Protocole de `bin/thesis`
Metal, un token, batch 1, 252 projections, mémoire froide par construction. Deux pipelines (`tv_f16`, `tv_slot`) ; table
de 384 classes en buffer constant partagé (12 Ko) ; une activation `SplitMix64(0x6_7451)`, 16 384 f32 gaussiens, un seul
buffer pour les deux bras. Par matrice : `read_matrix_raw`, `transcode(Slot32)`, reconstruction f64, arrondi f16,
références `y_ref` / `y16_ref` en f64, upload de 6 buffers LLVQ et 1 buffer FP16 ; vérification (§8.3) avant toute
mesure. Mesure : un command buffer par bras, 252 encoders, `d_out × 32` threads en groupes de 256, tuilage identique
(128 blocs, 3 072 colonnes, 12 Ko de threadgroup memory). Chrono autour de `commit()` et `wait_until_completed()` ;
7 passes, reps 0 et 1 jetées, minimum des 5 restantes pour les runs à deux bras : leur rapport est un quotient de deux
minima, ce qui le disqualifie face à K1. Le banc à sept bras (K1) dispatche tous les bras à chaque round dans le même ordre et forme
le rapport round par round, médiane et plage. Cinq asymétries, toutes contre LLVQ ou négligeables. FP16 mesuré d'abord ;
soumission non soustraite ; queue lue en f32 contre f16 ; 9 binds contre 4 ; 12 Ko de table non comptés. Tout terme additif
commun comprime le rapport : le 2,07× est un minorant du rapport ALU/mémoire pur.

### 8.2 Le bras FP16
`w16 = f16_bits(w)`, où `w` est la reconstruction f64 des blocs LLVQ dans la base tournée. Le bras FP16 lit les mêmes
valeurs à l'arrondi près : il mesure un coût, sans rien dire de la qualité. Sur CUDA, `r = t(tv_f16) ÷ t(cuBLAS)` vaut
1,024 (2 bras) et 1,015 (5 bras) sur L40S (*mesuré*, [F1](mesures/f1-cublasf16-2026-08-18.txt)). Sur A100 le même témoin
est à 1,14× de cuBLAS (*mesuré*, [F4](mesures/f4-a100-2026-08-18.txt)). Sur Metal il n'a jamais été confronté à MPS ni à MLX.

### 8.3 Vérification numérique
| grandeur | valeur | étiquette |
|---|---|---|
| lignes vérifiées | 1 105 920, 252 matrices, les deux bras | *mesuré* |
| métrique | max sur les lignes de \|got − want\| / max(Σ\|wᵢxᵢ\|, 1e-12), référence f64 | |
| seuil dur | `assert!(e < 1e-3)`, avant toute mesure de temps | |
| pire erreur LLVQ | 3,4·10⁻⁸ · Σ\|w·x\|, identique entre exécutions et fichiers | *mesuré* |
| pire erreur FP16 | 2,8·10⁻⁸ | *mesuré* |

Réserves : granularité ligne (le bloc vit dans `bin/decreal`) ; le seuil est cinq ordres au-dessus de l'erreur, la preuve
est le pire-erreur imprimé. `thesis` ne re-vérifie pas le transcodage contre `Indexer::decode` (ce verrou vit dans
`llvq-artifact/tests/runtime_format.rs`). `slot_dot` code en dur `gain = hdr >> 9` (1 bit) ; `decreal` l'assert, `thesis` non.

### 8.4 Chiffres et dispersion
| banc | FP16 | LLVQ `Slot32` | rapport | source | étiquette |
|---|---|---|---|---|---|
| 7 bras, 2026-08-05 | 21,728 ms, 7,27 Go | 10,496 ms, 2,50 Go, 5,510 b/poids | 2,03× [2,03–2,10], médiane round par round | [K1](mesures/k1-metal-2026-08-05.txt) | *mesuré*, le chiffre à publier |
| 2 bras, trois invocations témoins | | | [2,029 ; 2,080] ; 7 bras : 2,03× · 2,06× · 2,09× | [témoin](mesures/thesis-temoin-2026-08-04.txt) | *mesuré* ; une valeur ponctuelle n'a pas de contenu |
| 2 bras, 2026-08-01 | 21,691 ms, 335,0 Go/s | 10,460 ms, 239,2 Go/s | 2,0737× ; 2ᵉ passe 2,08× | README d'époque, aucun log | *mesuré* sur l'ordre de grandeur, suspect sur les décimales ; périmé par K1, les millisecondes ne se publient pas |
| 2 bras, 2026-08-03, fichier scellé | 22,675 ms | 11,021 ms | 2,0574× | aucun log | *mesuré* sur l'ordre de grandeur, suspect sur les décimales ; périmé par K1 ; écarts à la ligne du 08-01 +4,5 % / +5,4 % / −0,8 % |

Les millisecondes dérivent d'une invocation à l'autre ; octets, b/poids et pires erreurs se reproduisent au chiffre.
Reproduction : `cargo run --release -p llvq-metal --bin thesis -- <scellé>` ; le défaut des trois bancs
(`thesis.rs:191`, `matvec.rs:503`, `decreal.rs:139`) est `~/llvq-q4b.llvq`, non publié.

### 8.5 Ce que le rapport exclut
Le rapport exclut l'attention entière (QKᵀ, softmax, AV, RoPE, cache KV) ; les 145 RMSNorm, dont `q_norm` / `k_norm` par
tête ; la SwiGLU ; les résiduels. Il exclut aussi la rotation d'incohérence sur x, 144 par token, payée par le seul bras
LLVQ, et le `lm_head` lié, rajouté analytiquement. Restent hors rapport l'échantillonnage sur 151 936 logits, le prefill
et le transcodage. La rotation coûte 0,206 % des projections en arithmétique (1,499·10⁷ ops contre 7,267·10⁹ flops,
*calculé*, `rotation.rs`). Sur CUDA, `rot_apply` rend 9,5e-8 contre f64 sur 8 formes et 8,05 µs à n = 2560 (*mesuré*,
[rotation CUDA](mesures/rotation-cuda-2026-08-05.txt)), et le bout-en-bout du §5.6 la paie. Sur Metal, aucun noyau de
rotation n'existe et `bin/run` décode en mémoire.

### 8.6 Les tok/s du banc
Projections seules : 2,07×. Avec le `lm_head` f16 modélisé : 1,88× au plus. 78,2 tok/s n'est mesuré de rien.
`thesis.rs:433-435` : `head_bytes = 389_070_848 × 2`, `bw = f16_bytes / t16` (335,0 Go/s), `head_s = 2,3228 ms` aux deux bras (*calculé*).

| bras | total | tok/s | étiquette |
|---|---|---|---|
| FP16 | 24,014 ms | 41,64 | *calculé*, borne supérieure |
| LLVQ | 12,783 ms | 78,23 | *calculé*, borne supérieure |
| rapport | 1,879 | | majorant du bout-en-bout Metal |

Ajouter une constante commune comprime le rapport (2,07 → 1,88) : le traitement est conservateur pour LLVQ. Les défauts
réels : le `lm_head` n'est jamais exécuté, le reste du pas de décodage est exclu, la constante ne correspond à rien.

### 8.7 Échelle bits contre vitesse
Une comptabilité (payload + adressage + queue f32 + échelles f32, sur tous les poids), un processus, sept bras entrelacés,
7 rounds dont 2 jetés (*mesuré*, K1, Metal) :

| layout | b/poids, métrique étroite | b/poids, métrique noyau | vs FP16 | objet | étiquette |
|---|---|---|---|---|---|
| `Grouped32` | 3,3548 (`rtbits`, 150 681 600 blocs, 6,5 s) | 3,498 → 1,589 Go | 0,69× [0,68–0,69] | modèle entier | *mesuré* |
| `Flat32` | 4,54 (gate_proj) | 5,256 → 2,39 Go | 0,91× [0,91–0,91] | modèle entier | *mesuré* |
| `Sorted32` | 4,75 | | 1,04× | gate_proj seul, `bin/matvec` | *mesuré* |
| `Fixed96` | 4,000, structurel | | | jamais en matvec | *calculé* |
| `Slot32` | 5,376 modèle / 5,375 gate_proj (`rtbits` 5,3756) | 5,510 → 2,50 Go | 2,03× [2,03–2,10] | modèle entier | *mesuré* |
| FP16 | 16,000 | | 1,00× | | |

L'écart 5,51 contre 5,375 est un écart de métrique, pas d'objet : `rtbits` (modèle entier) et `matvec` (gate_proj)
coïncident à 0,02 %. La courbe est non linéaire : `Flat32` économise 0,254 b/poids sur `Slot32` pour 2,27× le temps,
`Grouped32` 2,012 pour 3,01× (*calculé*). `float4` rend 3,5 % sur LLVQ et 5,1 % sur FP16, rapport inchangé (*mesuré*,
K1). La queue f32 pèse 2,71 % du trafic LLVQ (67 829 760 o sur 2 502 446 285) ; en f16, `Slot32` tombe à 5,435 b/poids
(*calculé*). Les 335 Go/s du bras FP16 valent 83,8 % d'un pic de 400 Go/s supposé ; le « 93 % » est celui de gate_proj
(370 Go/s). Sur CUDA, le layout servi est `Planes14` : 4,804 b/poids noyau, 2,14× [2,11–2,15] sur L40S (*mesuré*,
[C1](mesures/c1-planesbench-2026-08-06.txt), [E2](mesures/e2-golay70-bench-2026-08-07.txt)). Il rend 1,14× [1,14–1,15]
sur `Slot32` à contenu identique, et 0,79× sur A100 (*mesuré*, F4).

### 8.8 Transcodage au chargement
| chiffre | ce qu'il est | étiquette |
|---|---|---|
| « ~3 s pour un 4B » | 150 681 600 × 243 ns / 12 cœurs, avec une parallélisation qui n'existe pas (`transcode()` mono-thread) | ~37 s mono-cœur (*calculé*) |
| 128 s, `load_s` de `thesis` | couvre le dépaquetage de 981 Mo, la reconstruction f64, 3,63 Md de conversions f16, la référence CPU f64 et 7 uploads par matrice | *mesuré*, mal étiqueté |
| transcodage seul | `bin/decreal` sur 16 777 216 blocs réels, `Fixed96` et `Grouped32` ; facteur ×8,98 puis ÷2 vers le modèle entier ; `Slot32` absent | *mesuré* |
| `Planes14` / `Planes12x`, M3 Max, 16 threads | 84 s / 404 s (×4,8, recherche réseau à 5 niveaux par bloc) | *mesuré*, 2026-08-09 |
| `Planes12x` sur carte louée | 1 340 s | *mesuré*, [G3](mesures/g-horloges-planes12x-2026-08-23.txt) |
