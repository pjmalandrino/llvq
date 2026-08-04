# Protocole de mesure v2 — LLVQ 2 bits contre 4 bits contre FP16, Qwen3‑4B, sur HF Jobs / GPU NVIDIA

**Statut : plan. Aucun chiffre de résultat ci‑dessous n'est une mesure de cette campagne.** Les valeurs citées sont soit des octets relevés sur disque ou sur le Hub, soit des mesures antérieures datées et sourcées, soit de l'arithmétique exacte sur des formes lues dans `config.json`, soit des coûts extrapolés de traces existantes et étiquetés comme tels.

Ce document **remplace** le plan v1 (`docs/plan-de-test-papier.md`) sur trois points : la cible matérielle, le bras adverse, et l'axe vitesse. Il **reprend** le reste en le citant.

---

## 0. Ce qui change par rapport au plan v1, et pourquoi

### 0.1 Ce qui saute

| v1 | Cause |
|---|---|
| **Cible M3 Max** (§0, §5.10, tout le budget « 0 $ ») | Personne ne reproduit un papier sur un MacBook. Aucune preuve de run opposable. La mémoire unifiée forçait à bannir le mot VRAM (§0). |
| **Bras B = famille MLX** (§1.2 : 5 réglages `mx.quantize`) | MLX n'existe pas sur CUDA. |
| **Ticket T1** `ops/mlx_dequant.py` | idem |
| **Tableau 2×2 embedding** (§1.3) et **ticket T8** (`set_embedding`) | **Aucun** quantifieur CUDA (AWQ, GPTQ, bitsandbytes) ne touche `nn.Embedding` ni le `lm_head`. Le confondant d'iso‑périmètre disparaît par construction. Corollaire douloureux : **`~/q4b-e4.llvq` (A2, embedding int4) perd son unique comparateur.** |
| **Axe 3a/3b/3c — vitesse** (§2.3, tickets T4, T5, T7, T11) | `llvq-metal` est Apple‑only par construction (`[target.'cfg(target_os = "macos")'.dependencies]` + `compile_error!`). **Il n'existe aucun noyau fusé CUDA.** Et sur CUDA `sealed::load` décode vers des tenseurs f16 denses : le bras A et le bras C sont **le même objet** après chargement. Voir §4. |
| **Ticket T9** (sonde `MTLBuffer` / footprint) | Méthodologie macOS. Remplacée par §3.3. |
| **« Le mot VRAM est banni du papier »** (§0) | Inversé. Sur NVIDIA il a un référent. Mais il ne mesure pas ce qu'on croit — voir §3.3. |
| **Ticket T10** (ligne Slot32 dans `rtbits`) | Sans objet hors campagne noyau. |

### 0.2 Ce qui survit intégralement, et qu'on ne recopie pas

- **§2.1** — définition opérationnelle de l'axe espace (« l'ensemble minimal de fichiers que le chargeur ouvre sur une machine sans réseau »), unités en puissances de dix, contrôle d'entropie `zstd` en annexe.
- **§2.4.1** — définition figée de la perplexité (wikitext‑2‑raw‑v1 test, fenêtres non chevauchantes de 4096, dernière fenêtre jetée, sans tokens spéciaux, 73 fenêtres pleines). **Inchangée.**
- **§2.4.3** — les six contrôles avant tout chiffre du bras adverse (L1 accord de deux implémentations, L2 ré‑empaquetage, L3 noyau contre reconstruction, L4 budget de narrowing, contrôle identité, contrôle 8 bits). **Transposés au format AWQ/GPTQ en §2.4 ci‑dessous, avec L2 promu au rang de contrôle décisif.**
- **§2.5.1** — protocole MMLU 5‑shot figé, micro publié, macro nommé à côté, et **la règle du certificat** : toute modification du harnais invalide les 70,42 et impose de rejouer la baseline.
- **§2.5.3** — bootstrap apparié stratifié par matière comme statistique retenue ; McNemar en secondaire étiqueté « non pondéré » ; gain d'appariement ~1,40× en SE, **pas 3,5×**.
- **§4 en entier** — les trois sources de variance, le déterminisme comme prémisse, le delta apparié de log‑vraisemblance par fenêtre comme statistique de comparaison, les seuils de défendabilité (§4.4 : « 16,9617 < 17,04 » reste **indéfendable, à retirer**), la variance de méthode déclarée et non mesurée (§4.5). Voir §7 pour ce que CUDA y change.
- **§7.2** — les six résultats négatifs à publier nous‑mêmes. Les n° 1, 2 et 6 se **durcissent** sur CUDA.

---

## 1. La cible matérielle

### 1.1 Flavor unique : `l40sx1`

**8 vCPU · 62 Go RAM hôte · 48 Go VRAM · 1,80 $/h = 0,030 $/min.** Tarif relu sur `huggingface.co/docs/hub/en/jobs-pricing` ; la table `FLAVORS` de `ops/run.py:57-75` est exacte au cent près sur ses 11 entrées.

Quatre raisons, dans l'ordre de force.

**(a) C'est la seule carte 48 Go qui soit sm_89 NATIF.** `ops/Dockerfile.cuda:23` fige `CUDA_COMPUTE_CAP=89` (Ada) parce que le builder d'un Space n'a pas de GPU. `candle-kernels-0.9.2/build.rs` appelle `build_ptx()`, donc l'image embarque du PTX, que le driver JIT‑compile — **PTX est compatible vers l'AVANT, jamais vers l'arrière**. Conséquence à écrire dans l'en‑tête du Dockerfile :

> cap 89 = **plancher, pas cible** : natif sur Ada (`l4x1`, `l4x4`, `l40sx1`), par JIT PTX sur tout sm ≥ 89 (`h200*`, `rtx-pro-6000*`). **Exclut sm_86 (`a10g-*`) et sm_75 (`t4-*`). Exclut aussi `a100-large`, qui est sm_80.**

`a100-large` (80 Go, 2,50 $/h) figure dans `FLAVORS` et est le repli qui vient naturellement à l'esprit : **c'est celui qui ne peut charger aucun noyau.** À écrire avant que quelqu'un le tente.

**(b) 48 Go retirent le piège de dtype de la liste des façons de perdre un job.** `bin/ppl` construit en **F32 par défaut** (`ppl.rs:51`, `eval::dtype(DType::F32)`). Arithmétique à ctx 4096 sur Qwen3‑4B, formes lues dans `config.json` :

| poste | f16 | f32 |
|---|---|---|
| poids (4 022 468 096) | 8,045 Go | 16,090 Go |
| logits [1,4096,151936] + copie f32 (`model.rs:356`) + `log_softmax` non fusé (`candle-nn/src/ops.rs:31-38`, trois tampons pleine largeur vivants) | ~7,5 Go | ~7,5 Go |
| `scores` 1×32×4096×4096 matérialisé sans flash (`model.rs:271-272`), 2 à 4 tampons vivants | 2,1–4,3 Go | 4,3–8,6 Go |
| **total** | **~16,4–18,6 Go** | **~26–30 Go** |

Sur les 24 Go d'un `l4x1` (≈22,8 Go rapportés par le pilote), un `ppl` sans `LLVQ_DTYPE=f16` meurt **après** avoir facturé le pull d'image et 8 Go de téléchargement. Sur 48 Go les deux dtypes passent. La campagne impose f16 partout (§2.5), mais elle ne fait pas dépendre un job payé d'une variable d'environnement oubliée.

**(c) Bande passante.** L40S = 864 Go/s. `l4x1` = 300 Go/s, soit **en dessous** des ~400 Go/s (spec constructeur, `SUPPOSE`) du M3 Max de développement : à 0,80 $/h pour un travail potentiellement plus lent que le portable, `l4x1` n'est pas l'option économique, c'est le même prix total pour deux fois le temps mural. Une évaluation candle mono‑flux, sans batch ni cache KV, est largement limitée par la mémoire.

**(d) Ce qu'elle représente pour un lecteur.** Une L40S est la carte d'inférence standard louable partout (AWS, GCP, Lambda, RunPod), à un tarif public. « Toute la campagne tient sur une carte 48 Go à 1,80 $/h » est un énoncé qu'un tiers peut vérifier avec sa propre carte bancaire.

**Repli, un seul** : `rtx-pro-6000` (96 Go, 2,75 $/h, Blackwell, déjà éprouvé par les runs 8B et 32B, mais par JIT PTX), si le pilote (§8, E1) montre la L40S à moins de 1,5× le M3 Max. **Jamais** `a100-large`, `a10g-*`, `t4-*`.

### 1.2 Ce que la table `FLAVORS` doit gagner

Ticket **O2** (§5) : ajouter `compute_cap` et `ephemeral_gb`, compléter les 15 flavors manquantes, et **dériver la table de `huggingface_hub.list_jobs_hardware()`** (appelable sans token) plutôt que de la recopier — avec une assertion dans `cmd_selftest`. Noms API exacts à ne pas inventer : `a100x4`/`a100x8` (**pas** `a100-largex4`), `a10g-largex2`/`x4`, `l40sx4`/`x8`, `rtx-pro-6000x4`/`x8`, `cpu-basic`. Documenter que la colonne `vram` est **par carte** délibérément (candle ne pilote qu'un device).

Stockage éphémère : `l40sx1` en a assez pour le checkpoint (8,06 Go) + l'artefact (1,77 Go) + un overlay (7,27 Go) simultanément — la contrainte « un overlay à la fois, puis `rm` » de v1 §6 **tombe**.

---

## 2. Les trois bras, version CUDA

### 2.1 Définition

| Bras | Objet | Octets | Statut |
|---|---|---|---|
| **A1** | `Pier-Jean/Qwen3-4B-LLVQ-2bit`, révision `f00daa7bc1dd12a720304a4483f2219d10f15c96`, fichier `qwen3-4b-llvq.bin` | **1 770 527 533** | existe, publié, sha256 `9db213ef…c84b0` (MESURE) |
| **A2** | `~/q4b-e4.llvq` (embedding int4 g64) | 1 211 403 653 | existe, **jamais scoré, et désormais sans comparateur** (§0.1) |
| **B0** | `Qwen/Qwen3-4B-AWQ`, commit `74d4bd2bd4bff9cafc9345221320bffb08b406a3`, `model.safetensors` | **2 666 027 672** | existe, MESURE sur le Hub |
| **B1** | GPTQ w2 g128 sym, calibré sur **nos** 131 072 tokens C4 | à produire | — |
| **B2** | GPTQ w3 g128 sym, même calibration | à produire | — |
| **B3** | bitsandbytes NF4, blocksize 64, **sans** double quant | à produire | — |
| **C** | `Qwen/Qwen3-4B`, bf16→f16 | 8 044 936 192 (poids) | sur le Hub |

Dénominateurs, à nommer sur chaque ligne portant un bits/poids (inchangés de v1 §1.1) : **4 022 468 096** poids du modèle · **3 633 315 840** poids de projection (le dénominateur homogène) · **3 616 358 400** poids réellement quantifiés.

Le chiffre à afficher pour A1 reste **2,159506 b/poids**. Le 2,0702 ne doit apparaître nulle part pour ce fichier.

### 2.2 Le bras B0 : pourquoi l'AWQ officiel, et pas un 4 bits maison

`Qwen/Qwen3-4B-AWQ` est la quantification 4 bits **publiée par l'auteur du modèle**, calibrée (mise à l'échelle par saillance d'activation), 463 285 téléchargements. On passe d'un adversaire qu'on fabrique à un adversaire qu'on subit : personne ne peut écrire « vous avez affaibli l'adversaire », ni contester le choix de corpus de calibration, ni le group size.

Vérifié à l'octet sur le Hub (en‑tête safetensors lu par requêtes HTTP Range) :
- **902 tenseurs** : 252 `.qweight` (I32), 252 `.qzeros` (I32), 252 `.scales` (F16), 146 `.weight` (**BF16**). Les 252 préfixes `.qweight` sont **exactement** nos 252 clés de projection. Aucun `model.embed_tokens.qweight`.
- `quantization_config` = {bits 4, group_size 128, zero_point true, version gemm, quant_method awq, `modules_to_not_convert` null}, `tie_word_embeddings` true.
- Comptabilité qui boucle : 3 633 315 840 × 4,15625/8 = 1 887 621 120 o de payload, + embedding 777 912 320 + normes 392 192 = **2 665 925 632**, contre 2 666 027 672 mesurés → écart 102 040 o = **exactement l'en‑tête safetensors**.
- `tokenizer.json` : même oid LFS `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` et même taille 11 422 654 o que `Qwen/Qwen3-4B` **et** que le blob embarqué dans notre `.bin` (fiche‑4b §1.2). **L'identité d'empreinte de tokens entre les trois bras est prouvée par sha avant tout run**, là où le bras MLX de v1 l'espérait (son `tokenizer.json` faisait 11 422 650 o, autre sha, v1 §2.4.4).

**Réserve à publier, pas à taire** : Qwen ne documente **nulle part** comment cet AWQ a été produit — ni corpus, ni outil, ni nombre d'échantillons (carte modèle vérifiée). C'est un adversaire fort mais **opaque**. C'est précisément pourquoi B1/B2 existent à côté.

### 2.3 Le bras B1 : le point qui décide

GPTQ w2 g128 sym, calibré sur **exactement** nos 131 072 tokens de C4, produit par `gptqmodel` (support Qwen3 explicite).

Débit effectif, `g_idx` int32 par canal d'entrée compris (32/d_out, pondéré sur les 7 formes = 0,007630) :

**2 + 20/128 + 0,007630 = 2,148255 b/poids, contre nos 2,159506. Écart : 0,52 %.**

Le plan v1 ne pouvait offrir que 2,250 (4,0 % de marge sur l'abscisse) et **admettait lui‑même** (§4.4) que cette marge pouvait ruiner la figure principale. Cette collision disparaît : à 0,52 %, l'abscisse cesse d'être un argument.

**Configuration à figer explicitement et à publier**, sinon le point n'est ni reproductible ni « GPTQ vanille » — les défauts de `gptqmodel` ne sont pas ceux qu'on croit :

```python
GPTQConfig(bits=2, group_size=128, sym=True,
           desc_act=False,          # default_desc_act() rend False
           act_group_aware=False,   # ⚠ mis à True par défaut quand method==GPTQ
           gptaq=None,              # le levier GPTQv2 ; il n'existe pas de champ `v2`
           foem=None, mse=0.0, static_groups=False,
           rotation=None, smoother=None, lm_head=False,
           fallback=None)           # ⚠ défaut = RTN au-delà d'un seuil de 0,5 %
```

**Cadrage obligatoire, et il n'est pas celui qu'on croit.** Ce point ne reproduit **pas** la Table 9 du papier : ses lignes « GPTQ » vs « Spherical GPTQ » opposent deux **règles de correction** appliquées au même codebook de Leech (`docs/llvq-paper-notes.md:227-244`), pas un quantifieur affine scalaire. Et le chiffre qui circulait (91,90) a été corrigé en **191,90** le 2026‑08‑04. Formulation défendable :

> À 0,52 % de débit près, la famille affine par groupes calibrée se situe [au‑dessus / en dessous] du réseau de Leech sur Qwen3‑4B.

**Pré‑enregistré (§6.3)** : on s'attend à ce que GPTQ w2g128 se dégrade fortement. Un effondrement est un résultat, pas une victoire, et **ne doit jamais titrer une table**.

### 2.4 Le contrôle d'iso‑périmètre — et la découverte qui l'oblige à changer de nature

**L'iso‑périmètre de DÉBIT est vrai et vérifié** : les trois bras quantifient les mêmes 252 projections et portent les mêmes 389 152 256 poids à 16 bits.

**L'iso‑périmètre de CONTENU est faux pour AWQ.** Comparaison sha256 par requêtes HTTP Range, base contre AWQ, tenseur par tenseur :

| tenseur | verdict |
|---|---|
| `model.layers.{0,17,35}.input_layernorm.weight` | **DIFFÉRENT** |
| `model.layers.{0,17,35}.post_attention_layernorm.weight` | **DIFFÉRENT** |
| `q_norm`, `k_norm`, `model.norm` | identiques |
| `model.embed_tokens.weight` (3 sondages tête/milieu/queue sur 777,9 Mo) | identiques |

**Bilan : 72 tenseurs modifiés sur 146, 74 bit pour bit identiques.** AWQ replie ses échelles de saillance par canal dans les RMSNorm qui précèdent les projections.

Trois conséquences, toutes actionnables :

1. **Contrôle à exécuter une fois, 0 $, sans job** : sha des 146 tenseurs portés. Attendu **74 égaux / 72 différents** pour AWQ, **146/146 égaux** pour GPTQ et bnb (`smoother=None` ne touche pas les normes). Publier le résultat tel quel.
2. **À déclarer en section** : à débit nominal égal, AWQ dispose d'un degré de liberté dans 72 tenseurs pleine précision que **les deux bras portent gratuitement** et que LLVQ n'utilise pas. Ce n'est pas une triche, c'est la méthode AWQ — mais c'est un avantage structurel qui doit être nommé.
3. **La voie d'entrée R1 (overlay des 252 projections) est INTERDITE sur AWQ.** Elle produirait un modèle mathématiquement faux : projections à l'échelle `s`, normes du checkpoint de base sans le `1/s` compensatoire. Et `llvq_llm::artifact::load` (`artifact.rs:47-72`) n'itère que sur les 252 clés de projection — il ne **peut** pas porter les normes. R1 reste valide sur B1/B2/B3.

**Effet de bord favorable** : l'embedding d'AWQ est en **BF16** (le `torch_dtype: float16` du `config.json` ment sur les tenseurs), copié octet pour octet du checkpoint. Notre chargeur le ramène à f16 à la lecture, exactement comme `bin/seal` l'a fait pour l'artefact. **Les bras A et B partagent donc le même `lm_head` aux mêmes bits, prouvé et non supposé** — le dernier écart de protocole non contrôlé de fiche‑4b §4.2 se ferme.

### 2.5 Voie d'entrée dans notre harnais

**R2, principale, zéro ligne de Rust.** Déquantifier vers un checkpoint safetensors complet, le pousser dans un dépôt HF **PUBLIC**, le scorer par `LLVQ_MODEL=<user>/qwen3-4b-awq-deq`. `Checkpoint::fetch` accepte n'importe quel repo id, et `candle_transformers::models::qwen3::Config` n'a ni `deny_unknown_fields` ni champ `torch_dtype`, donc un `config.json` portant `quantization_config` passe.

> ⚠️ **Un dépôt PRIVÉ ne fonctionne pas.** `loader.rs:22` appelle `hf_hub::api::sync::Api::new()` → `Cache::default()` → `cache.token()`, qui lit **uniquement** le fichier `$HOME/.cache/huggingface/token`. `HF_TOKEN` n'est jamais consulté, et `HF_HOME` est intégralement ignoré (donc le `ENV HF_HOME=/scratch/hf` de `Dockerfile.cuda:65` est **inerte** pour les quatre `Api::new()` du crate). Sortie sans Rust si un dépôt privé s'impose : prélude `bash -c 'mkdir -p $HOME/.cache/huggingface && printf %s "$HF_TOKEN" > $HOME/.cache/huggingface/token && exec ppl …'`, à tester une fois pour 0,003 $ sur `cpu-upgrade`. **Publier les checkpoints déquantifiés est aussi ce qui rend la campagne rejouable par un tiers (§9, arbitrage 6).**

**R1, contrôle croisé, sur B1/B2/B3 seulement.** Overlay des 252 tenseurs `model.layers.{b}.{proj}.weight` en argv[3] de `bin/ppl` (`llvq_llm::artifact::key()` construit exactement ces noms, forme [d_out, d_in], aucune transposition). Il garantit que rien d'autre n'a bougé. Les deux voies doivent rendre la même perplexité au chiffre près ; **si ce n'est pas le cas, c'est R2 qui a un problème** — et on le sait pour 0,03 $ au lieu de le découvrir en relecture.

**Ticket T3 de v1 (branche overlay dans `mmlu.rs`) cesse d'être bloquant** : R2 fait tomber le premier chiffre de qualité du 4 bits sans toucher au workspace Rust, donc sans invalider le certificat 70,42.

### 2.6 La formule de reconstruction, et le piège qui la guette

Écrire nous‑mêmes le déquantificateur, en numpy, d'après la source d'AutoAWQ (`awq/utils/packing_utils.py`, lue verbatim). Avec `qweight` int32 [d_in, d_out/8], `qzeros` int32 [d_in/gs, d_out/8], `scales` f16 [d_in/gs, d_out] :

1. déplier par `shifts = arange(0, 32, 4)` en gardant les 8 quartets **contigus** : `pre[:, 8j+i] = (packed[:,j] >> 4i) & 0xF` ;
2. appliquer `AWQ_REVERSE_ORDER = [0,4,1,5,2,6,3,7]` **à l'intérieur de chaque groupe de 8 colonnes** ;
3. `W[in,out] = (iw − izeros.repeat_interleave(128,0)) * scales.repeat_interleave(128,0)` — **pas de −1 sur les zéros** (le −1 d'AutoAWQ n'existe que dans le chemin de repack exllama) ;
4. `weight = W.t().contiguous()` vers [d_out, d_in].

> 🚫 **Interdiction formelle d'utiliser `gptqmodel.utils.model_dequant.convert_awq_file`.** Cause exacte, vérifiée : son `unpack_cols` produit la même disposition que le dépliage d'AutoAWQ (`result[:, i::pack_factor]` place l'élément source j à l'indice 8j+i, identiquement), **mais il n'applique jamais `AWQ_REVERSE_ORDER`** (lecture intégrale des lignes 1256‑1317). Il rendrait des poids plausibles et faux — une permutation par paquets de 8 canaux de **sortie**. Et le bug vaut pour **tout** fichier AWQ GEMM, y compris ceux que gptqmodel écrit lui‑même : son `quantization/awq/utils/packing_utils.py` est une copie verbatim d'AutoAWQ et porte bien `AWQ_ORDER`/`AWQ_REVERSE_ORDER`, que son propre déquantificateur ignore.

Réciproque pour L2 : `AWQ_ORDER = [0,2,4,6,1,3,5,7]` (mutuellement inverses, vérifié : `R[O[i]] = i`).

Branches GPTQ (`w = (unpack_rows(qweight) − unpack_cols(qzeros)[g_idx]) * scales[g_idx]`, puis `.t()`, plus la correction v1 `qzeros += 1` selon `checkpoint_format`) et bitsandbytes (`dequantize_4bit`, **pas** de `.t()` — bnb stocke déjà en [d_out, d_in]). Un script unique doit **brancher sur le format**, jamais appliquer `.t()` partout.

**Arithmétique imposée** (reprise de v1 §2.4.3, argument inchangé) : calculer en float32, **un seul** `.astype(float16)` en sortie. Dtype cible **f16**.

### 2.7 Les six contrôles, avant tout chiffre publiable

| # | Contrôle | Ce qu'il ferme | Coût |
|---|---|---|---|
| **L1** | Notre reconstruction analytique contre l'extraction par forward‑identité (`W_hat = module(I_{d_in})` sur le noyau de la bibliothèque, exact car chaque sortie n'a qu'un terme non nul). Écart exigé : **0 bit** | disposition des bits, contre une seconde implémentation | 2 min |
| **L2** | **Ré‑empaquetage bit à bit** : reconstruire `qweight`/`qzeros` depuis notre sortie et exiger l'égalité **octet pour octet** avec le fichier téléchargé, sur les 252 matrices | **décisif** — ordre de quartets, group_size, `zero_point`. Ne dépend **d'aucune bibliothèque** (safetensors + numpy) | 3 min |
| **L3** | `awq_gemm(x,q,s,z)` contre `x @ dequant(W).T`, écart relatif max publié | que l'overlay mesure ce que le noyau adverse calcule | 5 min |
| **L4** | `‖f16(W)−W‖_F / ‖W‖_F` et compte de subnormaux f16 sur 252 tenseurs. Critère : ≥ 100× sous l'erreur de quantification du bras lui‑même | que l'overlay ne mesure pas notre narrowing | 1 min |
| **CI** | **Contrôle identité** : overlay reconstruit depuis le checkpoint, `ppl == baseline` à 3 fenêtres | nommage, formes, transposition, dtype | 1 run court |
| **C8** | **Contrôle 8 bits** : ppl attendue à ~0,1 % de la baseline | à 8 bits l'erreur est négligeable, donc **tout** écart accuse le script ; à 4 bits le même bug se cacherait derrière une dégradation attendue | 1 run court |

**L2 remplace L1 comme contrôle porteur, et c'est un changement de nature.** Dans v1, L1 vérifiait contre `mx.dequantize`, c'est‑à‑dire contre une bibliothèque supposée juste. Ici il n'y a plus de référence unique : AutoAWQ est **archivé** depuis le 2025‑05‑11 (dépôt GitHub `archived=true`), gptqmodel écrit un autre format. L2 remplace la confiance par une identité : la seule chose qui puisse re‑produire les octets exacts du fichier de Qwen, c'est la bonne lecture. Même raisonnement que le verrou à 15 chiffres de `classes_reproduce_theta_series`.

> ⚠️ **Conflit de dépendances réel** : `gptqmodel` 7.3.2 exige `transformers ≥ 5.4.0` et `torch ≥ 2.8.0` ; la dernière config testée d'`autoawq` 0.2.9 est torch 2.6 / transformers 4.51.3. Elles ne cohabitent pas. Deux jobs `hf jobs uv run` distincts, et **L2 comme verrou commun** puisqu'il ne dépend d'aucune des deux.

### 2.8 Candidats écartés, et pourquoi

- **autoawq** : dépôt archivé 2025‑05‑11. On lit sa source pour la formule, on n'en dépend pas.
- **torchao** `int4_weight_only` : layout tinygemm spécifique au matériel (tuiles k internes), la reconstruction n'est pas une formule affine par groupes → on perd L2.
- **HQQ** 0.2.8.post1 : dernière publication PyPI 2025‑10‑20, 9,5 mois de retard.
- **EXL3** (la famille de QTIP, le concurrent que le papier nomme) : `gptqmodel` sait le **produire** (`METHOD.EXL3`) mais **son propre déquantificateur ne sait pas le lire** — `detect_format()` renvoie `'exl3'` sur les clés `.trellis` (l.661‑663) puis `dequantize_model()` tombe dans `raise ValueError(f"Unsupported format {fmt}")` (l.1620). Un décodeur de treillis vers f16 dense est estimé 2‑4 jours humain, issue incertaine. **Trou déclaré, avec sa cause exacte.**
- **llm-compressor** 0.12.0.1 (successeur officiel d'autoawq, sait faire AWQ *et* GPTQ) : alternative légitime à gptqmodel. Écartée par choix, pas par principe.

---

## 3. Les cinq axes

Chaque table du papier porte son étiquette dans son en‑tête (règle v1 §0, inchangée) : **FORMAT** (la grandeur ne dépend que du format), **MOTEUR** (elle dépend de l'implémentation), ou **MODÈLE + point de fonctionnement**.

### 3.1 Axe 1 — Espace à froid (FORMAT)

Définition opérationnelle **inchangée de v1 §2.1**.

**La comparaison devient une ligne unique, entièrement mesurée des deux côtés** — plus de tableau 2×2, plus de cellule CALCULÉ :

| | A1 (LLVQ 2 bits) | B0 (`Qwen/Qwen3-4B-AWQ`) | rapport |
|---|---|---|---|
| fichier de poids | **1 770 527 533 o** | **2 666 027 672 o** | **×1,5058** |
| projections seules | 2,159506 b/poids | 4,156250 b/poids | **×1,9247** |

Périmètre à nommer : notre `.bin` embarque `config.json` + `tokenizer.json` (11 423 433 o), le dépôt AWQ les a à côté ; avec `tokenizer.json` des deux côtés le ratio devient ×1,5122.

**Ce que le passage à CUDA gagne ici** : les quatre ratios concurrents de v1 §1.3 (×1,2782 / ×1,5939 / ×1,8681 / ×2,0838) se réduisent à un. Plus aucun cadrage à choisir.

**Asymétrie à publier, elle joue contre nous** : 2560, 4096 et 9728 sont tous divisibles par 128 et par 64, donc AWQ, GPTQ et bnb n'ont **aucune queue**. Nous payons 16 957 440 poids de queue `KeepExact`, soit **0,150 b/poids = 7 % de notre débit**, parce que 24 ne divise aucune de ces largeurs. C'est un désavantage structurel du pas de 24, pas un détail de sérialisation.

**Vérification obligatoire de la table de débits** : les dériver du layout (`b + (b+16)/gs` + `g_idx` pour GPTQ) **et** les confronter aux octets du fichier réel. La comptabilité d'AWQ boucle à l'en‑tête près (§2.2). Le dépôt a déjà payé une fois le prix d'un débit annoncé et non compté (2,0653 → 2,7289).

**Contrôle d'entropie** (annexe, v1 §2.1) : `zstd -9 -T0 -k`, gain proche de zéro attendu sur A1 (espace de code saturé à 7 962 près), rien à conclure sur B.

**Établit** : les octets qu'un déployeur expédie. **N'établit pas** : la mémoire ni le débit.

### 3.2 Axe 2 — Qualité : perplexité (FORMAT)

Définition **figée, inchangée de v1 §2.4.1**.

**Procédure identique sur les trois bras** — un seul moteur, notre passe avant, même tokenizer (prouvé identique par sha, §2.2), même empreinte de tokens :

```bash
# bras C (baseline)
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 ppl 4096 12 cuda
# bras A1 (artefact scellé, monté en volume)
LLVQ_DTYPE=f16 ppl 4096 12 cuda /artifact/qwen3-4b-llvq.bin
# bras B0/B1/B2/B3 (checkpoints déquantifiés, R2)
LLVQ_MODEL=<user>/qwen3-4b-awq-deq LLVQ_DTYPE=f16 ppl 4096 12 cuda
```

> 🚨 **`bin/ppl` lit son modèle dans `LLVQ_MODEL`, avec pour défaut `Qwen/Qwen3-0.6B`** (`ppl.rs:57`). Un `ppl 4096 12 cuda` sans préfixe score un modèle 6,7× trop petit, **réussit**, et ne signale rien. Asymétrie à retenir : `bin/mmlu` prend le modèle en **argv[0]**, `bin/ppl` dans l'environnement, et la branche scellée de `ppl` ignore la variable. Ticket O1 : la sous‑commande `score` **refuse** de lancer un `ppl` baseline sans `LLVQ_MODEL`.

**Profondeur, deux tables** (v1 §2.4.2 inchangé) : courbe débit‑distorsion à **12 fenêtres** sur tous les points ; table de tête à **73 fenêtres** (recensement du split) sur A1, B0 et C. Les chiffres à 73 fenêtres **ne seront pas** 16,9415 / 12,2361 : ne jamais laisser coexister deux paires sans étiquette de nombre de fenêtres.

**Établit** : la restitution sur texte brut, à format seul variable. **N'établit pas** : le raisonnement — v1 §2.4.4, argument inchangé et démontré sur ce modèle même (algèbre abstraite au niveau du hasard pendant qu'histoire et droit tiennent au‑dessus de 80 %). **Ne jamais présenter la perplexité seule comme preuve de qualité.**

### 3.3 Axe 3 — Mémoire : pic, moyenne, et plancher des poids résidents

> 🚨 **Le résultat le plus important de cette section, et il faut l'écrire avant les tables : sur cette cible, le pic et la moyenne mesurés ne discriminent aucun format.**

Les trois bras tiennent des poids **f16 denses** en mémoire, par construction :
- **A1** : `sealed.rs:92` fait `decode_matrix` → `Tensor::from_vec(Vec<f32>)` → `.to_dtype(dtype)`. Le jeu de tenseurs résident est **exactement** celui du checkpoint, au même dtype. `LLVQ_DTYPE=bf16` n'y change rien.
- **B** : `artifact.rs:32-44` écrit et relit des **reconstructions f16** ; il n'existe aucun runtime int4 dans ce dépôt.
- **C** : f16 nativement.

Et le terme non‑poids (~8‑10 Go d'activations à ctx 4096, §1.1‑b) est **identique dans les trois bras** et domine l'écart de format.

**Donc trois lignes, jamais soustraites, jamais dans la même colonne.**

#### Ligne 1 — PLANCHER des poids résidents (FORMAT, arithmétique exacte, 0 s de GPU)

Dénominateur = les 4 022 468 096 poids du modèle ; embedding + normes = 389 152 256 poids à 16 bits = 778 304 512 o.

| objet | octets résidents | b/poids | statut |
|---|---|---|---|
| **A1 aujourd'hui** (f16 dense) | **8 044 936 192** | **16,000** | CALCULE exact |
| C f16 | 8 044 936 192 | 16,000 | CALCULE exact |
| B3 bnb NF4 bs64 (4,5000) dans son moteur | 2 822 044 672 | 5,613 | CALCULE |
| B0 AWQ w4g128 (4,15625) dans son moteur | 2 665 925 632 | 5,302 | CALCULE |
| **B1 GPTQ w2g128 (2,148255)** | **1 754 064 512** | **3,489** | CALCULE |
| A1 **si** Slot32 (non branché, Metal seulement) | 3 280 750 672 | 6,525 | CALCULE, layout MESURE |
| A1 **si** Grouped32 (1,589 Go proj. + 0,778) | 2 367 000 000 | 4,708 | CALCULE, layout MESURE |

> ⚠️ Les lignes `L≤4` (4,667 b/poids) et `L≤3` (3,667) de `docs/face-au-4-bits.md:108-113` sont dérivées d'un **banc sur source gaussienne**, et le même banc donne 5,667 pour L=5 là où `format-noyau.md:345` mesure **5,51** sur le 4B entier — il est 2,8 % haut. **Ne pas les mettre dans cette table sans étiquette.** L'énoncé de §7.2 doit reposer sur `Grouped32` (mesuré), pas sur `L≤3`.

**L'énoncé à publier nous‑mêmes, en section et pas en note :**

> Sur CUDA, aujourd'hui, le fichier 2 bits ne fait économiser **aucun** octet de mémoire : il tient exactement les 8 044 936 192 octets du FP16. Il en économise ×4,54 sur le disque. L'écart entre les deux nombres est, à l'octet près, le noyau qui n'est pas écrit. Et **même avec le noyau écrit**, `Slot32` (3,281 Go) reste plus gros que tout 4 bits raisonnable (2,666 à 2,822 Go) et **×1,87 plus gros que le point iso‑débit GPTQ w2g128** (1,754 Go). Seul `Grouped32` (2,367 Go) passerait devant l'AWQ, de ×1,13 — et il n'a aucune qualité publiée.

C'est le résultat négatif n° 1 de v1 §7.2, **durci** : v1 chiffrait « ×1,45 contre nous même si Slot32 était branché » en convention poids‑seuls ; à périmètre modèle entier et face au 4 bits calibré, c'est ×1,23.

#### Ligne 2 — Pic ET moyenne mesurés (MOTEUR, point de fonctionnement déclaré)

**Point de fonctionnement P1, unique et déclaré** : `ppl 4096 3 cuda` — batch 1, L = 4096, **aucun cache KV** (`Qwen3::forward` est documenté « no KV cache »), logits sur **toutes** les positions remontés en f32, poids **f16**, 3 fenêtres. C'est le seul point que les trois bras exécutent avec le **même graphe de calcul**, et c'est le point auquel la qualité est prise.

**Instrument : le plan de contrôle HF, pas un `entrypoint.sh`.** `huggingface_hub.fetch_job_metrics(job_id=…)` diffuse **un événement par seconde** portant `{"cpu_usage_pct", "memory_used_bytes", "memory_total_bytes", "gpus": {"<id>": {"utilization", "memory_used_bytes", "memory_total_bytes"}}, "replica"}`. Zéro ligne dans le conteneur, zéro rebuild d'image, zéro bucket, **même sémantique sur les deux images**, et il donne gratuitement la RAM hôte que personne ne budgète.

> 🚨 **Contrainte structurante : le flux est LIVE.** Le code de la lib passe `tolerated_status_codes=(500,)` avec le commentaire « *it returns an internal error 500 if the job has already finished, we simply ignore it* ». **Un job lancé en `--detach` et regardé après coup n'a AUCUNE métrique.** La capture doit vivre dans `cmd_monitor`, pas dans une commande post‑mortem. (Les **logs**, eux, survivent : `fetch_job_logs(follow=False)` rejoue l'historique.)

Publier : `vram_peak_bytes`, `vram_mean_bytes` (**moyenne temporelle pondérée par l'intervalle réel** entre échantillons, pas moyenne arithmétique — le flux a des keep‑alive à 30 s et des reconnexions), `vram_p50`, `vram_p95`, `vram_total_bytes`, `samples`, `period_ms`, plus `host_mem_peak_bytes`.

**Deux réserves à écrire à côté du chiffre, pas à cacher :**
1. **Échantillonnage à 1 Hz : un pic plus court qu'une seconde est invisible.** C'est la résolution de l'instrument.
2. `memory_used_bytes` mesure ce que le **contexte a réservé** (allocateur de candle compris, qui alloue par `cuMemAllocAsync` sur le pool par défaut), pas les octets de tenseurs vivants. C'est la grandeur « est‑ce que ça rentre », et elle ne se compare pas à des octets vivants — c'est exactement la faute M1/M2 que le dépôt a déjà payée (« comparer 17,41 Go de RSS à 2,39 Go de `mx.get_peak_memory()` était faux deux fois »).

**Contrôle obligatoire, 0 $, avant de promettre « pic ET moyenne » :** vérifier sur la série du pilote qu'elle **descend parfois**. Si l'allocateur ne rend jamais la mémoire au driver, la moyenne converge vers le pic en quelques secondes et publier les deux serait de la fausse précision.

#### Ligne 3 — Cache KV : le point non résolu n° 1 de v1 est tranché, par l'arithmétique

**Formule** : `octets/token = 2 · n_layers · n_kv_heads · head_dim · sizeof(dtype)`.

Qwen3‑4B f16 : 2 × 36 × 8 × 128 × 2 = **147 456 o/token = 144 Kio/token** (0,604 Go à 4096 ; 1,208 Go à 8192 ; 6,040 Go à 40 960 = `max_position_embeddings`).

Les deux autres valeurs se retirent, **chacune avec sa cause** :
- **320 Kio/token** = 2 × 80 × 8 × 128 × 2 = 327 680 → c'est le chiffre **correct de Llama‑3.1‑70B**, et il apparaît dans `docs/fiche-4b.md:381`, qui est le paragraphe de recalcul 70B. Chiffre juste, lu hors de son modèle. **80/36 = 2,2222… est exactement le rapport des nombres de couches** — le « rapport 320/144 qui ne correspond à aucune substitution évidente » (v1 §8.1) en est une, parfaitement évidente une fois posée.
- **~640 Ko/token** (`docs/pistes-battre-q4.md:97`) = 2 × 327 680 : le chiffre 70B avec le facteur K/V compté **deux fois**. Faux dans toutes les lectures.

**Le verrou de publication de v1 §8.1 est levé** : les projections mémoire à 70B redeviennent publiables. Repères : Qwen3‑8B = 147 456 · Qwen3‑32B = 262 144 · Llama‑3.1‑70B = 327 680 o/token.

**Corollaire à publier, et il est plus fort que la table elle‑même** : le cache KV est **additif et commun aux formats**, donc **tout avantage de format se dilue mécaniquement avec le contexte**. À P1 le cache vaut zéro (aucun bras ne l'alloue) ; en déploiement, avec `C_ctx` ≈ 0,5 Go et `A` ≈ 1,0 Go, l'avantage hypothétique de Slot32 sur AWQ passe de ×1,231 (poids seuls) à ×1,14 à 4096, ×1,12 à 8192, ×1,07 à 32 768. Publier cette dilution comme **arithmétique de déploiement hypothétique**, explicitement séparée de la table mesurée.

**Établit** : ce que chaque format pèserait en poids résidents, et ce que le harnais coûte à la carte. **N'établit pas** : que le 2 bits fait rentrer un modèle qui ne rentrait pas.

### 3.4 Axe 4 — MMLU (FORMAT)

Protocole **figé et certifié, inchangé de v1 §2.5.1**. Le chiffre publié est le **micro**, le macro à côté et nommé.

```bash
LLVQ_DTYPE=f16 mmlu Qwen/Qwen3-4B cuda 40                    # bras C
LLVQ_DTYPE=f16 mmlu /artifact/qwen3-4b-llvq.bin cuda 40      # bras A1
LLVQ_DTYPE=f16 mmlu <user>/qwen3-4b-awq-deq cuda 40          # bras B0
```

(`mmlu.rs:165-178` : argv[0] = modèle, argv[1] = device — **défaut `cpu`**, à toujours passer —, argv[2] = limit, dtype par défaut F16 via `LLVQ_DTYPE`.)

**Blocage matériel à lever d'abord** : `ops/Dockerfile.cuda:48-49` construit `--bin smoke --bin ppl --bin oracle`, et `:60-64` ne copie que ces trois‑là. **Aucun MMLU n'est exécutable sur la cible aujourd'hui**, et l'axe MMLU porte l'argument le mieux fondé du dossier. Ticket D1 (§5) : ajouter `--bin mmlu`, **et lui seul** — `seal` et `run` ne servent à rien ici et chacun est un lien LTO de plus sur une étape de build qui a déjà été tuée par SIGKILL (`Dockerfile.cuda:32-43`).

**Le certificat doit être réétabli en DEUX temps, une variable à la fois.** Le 70,42 ± 1,28 a été établi sur M3 Max / Metal / f16 (`docs/mmlu-micro-2026-08-02.log`). La campagne change le **harnais** (T2) *et* le **backend**. Un seul run CUDA après patch confondrait les deux et ne certifierait rien — exactement l'erreur que le dépôt a déjà payée 45 min pour un résultat ininterprétable.

1. Après T2 : rejouer MMLU limit=40 **en local sur Metal**. Si 70,42 retombe au centième, le patch est neutre. **44 min, 0 $.**
2. Puis le même sur CUDA. **L'écart restant EST le delta de backend, et c'est un résultat publiable en soi** — le dépôt signale ce chiffre comme non mesuré (`fiche-4b` §8.11). ~25‑45 min, **~0,75‑1,35 $**.

**Seuil de décision pré‑enregistré, à commiter avant le run** : `|Δ micro| ≤ 0,5 pp` **et** pas plus de 3 matières bougeant de plus de 2 questions → le certificat se transporte, on publie un seul chiffre. Au‑delà → on publie **deux** baselines avec leur backend, et aucun chiffre CUDA ne sort avant explication.

> `bin/oracle` ne verrouille **pas** ce point : il compare `Qwen3::hidden` à `candle_transformers::qwen3` **sur le même device**, avant le `lm_head`, à 64 tokens, en F32, sur un 0,6B. Il prouve que chaque backend est cohérent avec lui‑même, jamais que deux backends s'accordent. Voir §6.4.

**Profondeur** (v1 §2.5.2, mais l'économie change tout) :

| Niveau | Questions | Coût/bras sur `l40sx1` | SE |
|---|---|---|---|
| limit=40 | 2 280 | 25‑45 min = **0,75‑1,35 $** | ±1,28 / ±1,36 pp |
| recensement | 14 042 | **2,3‑4,8 h = 4,1‑8,6 $** | 0,00 |

Le recensement passait de « ~16,5 h, deux nuits de Mac, optionnel » à **une ligne de commande et 12‑26 $ pour les trois bras de tête, en trois jobs parallèles** (rien dans HF Jobs ne les sérialise, et la facturation est à la minute). C'est exactement ce que le passage au cloud achète.

> ⚠️ **`limit` change QUELLES questions sont posées, pas seulement combien.** `mmlu.rs:264` ne mélange que `if limit < picked.len()` ; un recensement saute le Fisher‑Yates. Un run limit=40 et un recensement scorent donc des ensembles différents dans des ordres différents : **les 57 comparaisons appariées du bootstrap disparaissent entre les deux.** Le recensement se compare recensement‑à‑recensement, jamais à limit=40. À écrire, sinon l'instinct « on va tout mettre en recensement » dégrade silencieusement l'instrument le plus fort de la campagne.

`timeout` obligatoire : un recensement exige `--timeout 8h` explicite. **Le défaut de la plateforme est 30 min**, et un job tué au timeout a été facturé jusque‑là.

**Établit** : un profil de capacités sur QCM 5‑shot. **N'établit pas** : la génération libre ni l'extraction documentaire. Publier le **profil par matière**, jamais l'agrégat seul, et ne jamais écrire « exactement 25 %, le hasard » (±7 pp à 40 tirages).

### 3.5 Axe 5 — Vitesse

**Vide sur cette cible.** Voir §4.

---

## 4. L'axe vitesse : l'option retenue et ce qu'elle interdit

### 4.1 L'option retenue : deux matériels, étiquetés

**La campagne CUDA ne mesure ni vitesse de noyau, ni tok/s bout en bout, ni débit de régime permanent.** Le gate G6 reste un résultat **Metal**, publié comme tel.

Trois faits qui se composent, tous vérifiés :

1. **A ≡ C par construction sur CUDA.** Après chargement, les 252 projections des deux bras sont des tenseurs f16 denses, mêmes formes, mêmes dtypes, même GEMM cuBLAS. Un tok/s publié ici mesurerait candle sur CUDA, pas LLVQ. **Ce n'est pas une faiblesse de protocole que la statistique pourrait réparer : c'est une identité.**
2. **Il n'existe aucun noyau fusé CUDA.** `llvq-metal/Cargo.toml` place `metal` et `llvq-artifact` sous `[target.'cfg(target_os = "macos")'.dependencies]`, et `lib.rs:631` porte un `compile_error!`.
3. **Le noyau fusé n'a de toute façon pas d'appelant.** `Qwen3::generate` (`model.rs:379-381`) rejoue tout le préfixe à chaque pas — le code le documente — donc il n'émet jamais de matvec.

**Et un quatrième, propre au cloud et absent du plan v1** : chaque job reçoit un conteneur frais, possiblement une carte physique différente avec des voisins différents. **Une comparaison de temps ENTRE deux jobs n'a aucune validité.** Tout ce qui est une comparaison de temps doit vivre dans un seul conteneur.

### 4.2 Ce que la campagne conserve, gratuitement

**(a) Le temps de chargement, décomposé.** C'est un résultat négatif réel : `llvq-artifact` est sans dépendance externe (aucun `rayon`, aucun `thread::spawn`), donc le décodage des 150 681 600 blocs est **mono‑thread** — « < 150 s » mesurées sur un cœur P de M3 Max (`fiche-4b:513`), donc **210‑300 s sur un vCPU loué**. **Le format coûte 3 à 5 minutes de décodage mono‑thread à chaque démarrage de processus.**

> ⚠️ « Ça se lit dans les logs, 0 ligne » est **faux** : dans `ppl.rs` comme dans `mmlu.rs`, `t0 = Instant::now()` est posé **après** le chargement du modèle *et* du corpus, et les lignes par fenêtre impriment un **cumul**. Correctif zéro‑Rust, applicable aujourd'hui : encadrer chaque invocation de `T0=$(date +%s.%N)` / `T1=$(date +%s.%N)` dans le script `bash -c`, plus un relevé après le prologue de provenance pour séparer pull + téléchargement du chargement. Découpage à 3 postes sans toucher au harnais, donc sans invalider le certificat MMLU.

**(b) Le contrôle de non‑divergence, APPARIÉ DANS UN MÊME JOB seulement.** Les temps par fenêtre des trois bras doivent être égaux au bruit près, par construction. Nommer honnêtement : « **contrôle de non‑divergence du chemin de calcul, résolution ~20 %** » (dispersion intra‑run de +75 % mesurée sur Mac), jamais « égalité des temps ».

### 4.3 Ce que l'option interdit formellement

- **Poser un tok/s CUDA à côté du 2,07× Metal.** Même règle que v1 pour `bin/run` contre `mlx_lm`.
- **Poser les 16,3 µs de leur Table 7 à côté de quoi que ce soit de chez nous.** 33,55 Mo / 16,3 µs = **2 058 Go/s**, au‑dessus du pic d'un A100 : leur chiffre décrit très probablement une matrice **L2‑résidente**, pas un régime de streaming DRAM. Consigne v1 §7.3 maintenue.
- **Revendiquer que le 2,07× se transporte sur NVIDIA.** Il n'est ni confirmé ni infirmé par cette campagne, et il ne peut pas l'être.

### 4.4 Le jalon à 1 h et 0 $ qui doit précéder toute décision de port

`docs/fiche-4b.md` §7 n° 14 chiffre « Grouped32 / Flat32 sur le modèle entier » à **~1 h de dev + 3 min de run sur Metal**, les shaders existant déjà dans `llvq-metal/src/bin/matvec.rs` (`matvec_g32`, `matvec_flat`). Cela referme le trou que `fiche-4b:476` reconnaît (« Grouped32 et Flat32 n'ont jamais tourné sur le modèle entier ») et donne la courbe bits↔vitesse **sur un seul protocole, un seul objet**, au lieu de mélanger 0,68× / 0,90× (`bin/matvec`, `gate_proj` seul, best‑of‑15) et 2,07× (`bin/thesis`, modèle entier).

C'est la question que l'arbitrage « porter le noyau sur CUDA » (§9, n° 4) pose ; elle se répond pour une heure avant d'engager des semaines.

---

## 5. Le code à écrire

Ordonné par déblocage. **T0 et O1 sont bloquants pour tout le reste.**

| # | Pièce | Fichier(s) | Effort | Vérification |
|---|---|---|---|---|
| **T0** | Commiter l'arbre (16 modifiés + 8 non suivis, dont `format.rs` LVQ3, `sealed.rs`, `embedquant.rs` — `lib.rs` déclare `pub mod embedquant;` alors que le fichier est non suivi). Retirer `Cargo.lock` du `.gitignore` **ET** : ajouter `Cargo.lock` + `rust-toolchain.toml` à `allow_patterns` (`run.py:428`), passer `cargo build --locked` (`Dockerfile.cuda:48`), faire **échouer `cmd_publish` sur `git status --porcelain` non vide** | racine, `ops/run.py`, `ops/Dockerfile.cuda`, `.gitignore` | **1 h 30** | `git status` vide ; `cargo clippy --all-targets` zéro warning ; `cargo test --release -- --include-ignored` vert. **Les quatre correctifs ensemble, ou aucun** : sans `allow_patterns` le lockfile ne monte pas dans le Space, sans `--locked` il n'est pas contraignant, et l'échec est **silencieux** dans les deux cas |
| **O1** | `ops/run.py score` : sous‑commande générique — image, flavor, `--timeout` **obligatoire** (pas de défaut), commande arbitraire chaînée en `bash -c 'set -euo pipefail; …'`, `env`, `secrets={"HF_TOKEN":…}`, volumes avec `revision`, bucket de sortie, `labels`, `name`. `cmd_launch:332` code en dur `command = ["smoke", …]` : **il n'existe aucun chemin pour lancer une mesure** | `ops/run.py` (~90 l) | **2 h** | un `ppl 4096 3 cuda` sur le bras C rend un nombre ; le refus de `LLVQ_MODEL` manquant se déclenche ; `set -euo pipefail` présent (sans lui, un `ppl` qui échoue laisse le `mmlu` suivant tourner et le job finit COMPLETED avec un résultat manquant — **le mode d'échec le plus cher du plan**) |
| **D1** | `--bin mmlu` ajouté à `Dockerfile.cuda:48-49` et au `COPY --from=build` `:60-64`. **Rien d'autre** | `ops/Dockerfile.cuda` (2 l) | 15 min | build vert ; si OOM, `CARGO_BUILD_JOBS=1` **avant** de toucher au LTO. Build **non facturé** (« there is no cost during build »), 40‑70 min de mur, prévoir 2 tentatives |
| **O3** | `cmd_monitor` : `fetch_job_logs(follow=True)` — la signature 1.26 a `follow=False` par défaut et « return immediately », donc **le monitor actuel n'affiche plus rien**. Puis second thread consommant `fetch_job_metrics` **en direct** → `metrics.jsonl` + `metrics-summary.json` (moyenne au trapèze) | `ops/run.py` (~60 l) | **2 h** | sur le pilote, `vram_peak_bytes` > 8,045e9 et `vram_total_bytes` ≈ 48e9 ; la série **descend** au moins une fois |
| **R6** | Épingler les révisions : les trois `"main"` de `corpus.rs` (l. 21 wikitext, 61 `cais/mmlu`, 195 C4) → révisions figées surchargeables par `LLVQ_DATA_REV_*`, **et** `Checkpoint::fetch` (`loader.rs:22`, `api.model(repo)` = `main` implicite) → `Repo::with_revision` + `LLVQ_MODEL_REV`. Enregistrer la révision résolue **et** le sha256 du fichier lu | `llvq-llm/src/{corpus,loader}.rs` (~30 l) | **1 h** | les révisions apparaissent dans `result.json`. **En local le cache hf_hub gèle de fait le contenu ; dans un conteneur neuf, chaque run re‑résout `main`** — c'est le trou que le changement de cible ouvre |
| **T6** | NLL et count **par fenêtre**, 9 chiffres significatifs | `llvq-llm/src/bin/ppl.rs` (~10 l) | 30 min | la moyenne des NLL reproduit `ln(ppl)` final. Débloque la barre d'erreur **et** fait passer le test de déterminisme de 1 à N points |
| **T2** | Dump CSV `subject,index,answer,pick,correct` (zipper l'index **avant** le Fisher‑Yates de `mmlu.rs:264-271` — sinon aucune clé stable) + **empreinte de tokens** sur la ligne de résultat | `llvq-llm/src/bin/mmlu.rs` (~35 l) | **1 h** | test : empreinte identique entre deux modèles, différente si `limit` change. `mmlu.rs:305-311` imprime **déjà** le détail par matière sur stderr — T2 est un reformatage, pas une mesure nouvelle |
| **JS** | `report.rs` : `result.json` à `$LLVQ_RESULT_JSON` + ligne `LLVQ_RESULT {…}` sur stdout. Contenu : provenance, argv résolu, params (dtype, ctx, fenêtres, limit, graine), **sha256 + taille de l'objet scoré** (dép. `sha2`, `llvq-llm` seulement), révisions de corpus, résultats, NLL par fenêtre, détail par matière, empreinte de tokens | `llvq-llm/src/report.rs` (~90 l) + 4 × 15 l | **3 h** | `grep -rn json llvq-llm/src/bin/*.rs` ne rend **rien** aujourd'hui : tous les chiffres du projet ont été recopiés à la main. C'est le geste qui a produit les trois chiffres orphelins |
| **PR** | Bannière de provenance : `git_sha`, `git_dirty`, `sha256(Cargo.lock)`, `rustc`, features, `built_at`. Fichier écrit à **`llvq-llm/PROVENANCE`** (il tombe sous `llvq-*/**`, donc il monte sans re‑toucher `allow_patterns`), lu par `build.rs`, dégradé en `"unknown"` s'il est absent | `llvq-llm/build.rs` + `provenance.rs` (~80 l), `ops/run.py cmd_publish` (~25 l) | **2 h** | `.dockerignore` exclut `.git` et le Space non plus : `build.rs` ne peut pas interroger git. Le sha est donc **ASSERTÉ par le publieur**, pas attesté — à écrire comme tel (§6.2) |
| **B‑py** | `ops/adversary.py` (PEP‑723, `hf jobs uv run`, aucune image à construire) : branche AWQ (formule §2.6 + `.t()`), branche GPTQ (+ correction v1 des qzeros), branche bnb (**pas** de `.t()`), contrôles L1‑L4, écriture des shards + `config.json` + copie du `tokenizer.json`, `upload_folder(private=False)` | `ops/adversary.py` (~250 l) | **4 h** | L2 doit rendre l'égalité **octet pour octet** avec le fichier téléchargé sur les 252 matrices |
| **O2** | `FLAVORS` : `compute_cap`, `ephemeral_gb`, les 15 manquantes, dérivées de `list_jobs_hardware()` + assertion dans `cmd_selftest` | `ops/run.py` (~40 l) | **1 h** | l'appel est gratuit et sans token. Transforme un dictionnaire recopié en invariant testé |
| **O5** | `ops/run.py record` / `manifest` / `verify` (§6) | `ops/run.py` (~200 l), `proofs/` | **4 h** | `verify` échoue sur un `claim_id` orphelin, un hash qui ne retombe pas, ou `git_dirty` |
| **FL** | `ops/floor.py` : plancher des poids résidents par sommation des formes×dtypes réellement produits (index safetensors pour B/C, en‑têtes de matrice pour A) + formule KV paramétrée avec 4B/8B/32B/70B en test | `ops/floor.py` (~120 l) | **2 h** | 0 GPU, 0 $. **Auditer les b/poids en sommant les formes produites, jamais en appliquant la formule du papier** — ça a déjà mordu ce projet une fois |

**Effort total : ~24 h humaines.** Tickets **retirés** de v1 : T1 (mlx_dequant), T4 (banc MLX), T5 (`thesis` apparié), T7 (`run.rs`), T8 (`set_embedding`), T9 (sonde footprint), T10 (`rtbits`), T11 (Grouped32/Flat32 — déplacé en jalon Metal §4.4). Ticket T3 (branche overlay dans `mmlu.rs`) **déclassé de bloquant à optionnel** (§2.5).

**Non planifié, non résolu** : implémentation GPU de `Rotation`, cache KV dans le runner, branchement du noyau dans `bin/run`, décodeur de treillis EXL3.

---

## 6. Preuves de run et manifeste de campagne

### 6.1 Le fait qui recadre tout

`https://huggingface.co/jobs/<user>/<job_id>` **redirige un visiteur anonyme vers un formulaire de login** (vérifié). La doc le confirme : les Jobs vivent sous `settings > Jobs`, `hf jobs inspect --namespace` exige l'appartenance, les URLs exposées demandent « an HF token with read access to the Job's namespace ».

> **Un identifiant de job n'est pas une preuve opposable à un lecteur. C'est un pointeur pour un auditeur à qui on a donné accès.** Ce qui est opposable, c'est le **bundle exporté** : vérifiable par hash, **rejouable**, adossé à un horodatage indépendant.

### 6.2 Trois classes, à ne jamais confondre

1. **Existence** — `job_id`, `url`, dump `inspect_job` (`created_at`/`started_at`/`finished_at`, `flavor`, `command`, `environment`, `volumes` avec leur `revision`, les **trois** durées), log brut, `nvidia-smi -q`, variables injectées par HF (`JOB_ID`, `ACCELERATOR`, `CPU_CORES`, `MEMORY` — **et rien d'autre**, le token n'est pas injecté). Faible : produite par nous, ré‑éditable.
2. **Validité** — empreinte de tokens identique, sha256 de l'objet scoré, dtype résolu imprimé, graine, argv complet, révisions de corpus **et** du checkpoint, `oracle` PASS sur la MÊME image et le MÊME flavor, contrôles L1‑L4 + CI + C8. **C'est la classe qui porte le dossier.**
3. **Environnement** — sha git, sha256 du `Cargo.lock`, `rustc`, features, image de base épinglée par **digest**. **La plus faible du dépôt aujourd'hui, et T0 ne la répare qu'à moitié.**

> ⚠️ **`JobInfo` n'expose NI digest d'image NI révision de Space** (champs vérifiés : `id, created_at, started_at, finished_at, docker_image, space_id, command, arguments, environment, secrets, flavor, labels, volumes, status, durations, owner, initiator, endpoint, url`). Le manifeste doit donc porter `image_identity: "asserted-by-publisher"`, et ne basculer en `"attested"` que si le build passe par GitHub Actions (§9, arbitrage 5).

> ⚠️ **Les labels sont MUTABLES et effaçables** (`hf jobs labels --label` **remplace tout**, `--clear` efface, aucune date, aucune signature). Le `name` est stocké comme label. Convention de `run_id` = `<AAAA-MM-JJ>-<axe>-<bras>-<jobid8>` : c'est de l'**indexation**, pas de la preuve. À ranger hors de la classe 1.

### 6.3 Où déposer

- **git `proofs/`** — le petit, le diffable, le porteur : `README.md`, `preregistration-<date>.md` (+ `.ots`), `manifest.jsonl`, `runs/<run_id>/meta.json` (2‑5 Ko), `verify.py`. Toute réécriture rétroactive se voit dans le diff.
- **Dataset HF public `Pier-Jean/llvq-proofs`** — le volumineux : `stdout.log`, `metrics.jsonl`, `mmlu-questions.csv`, `nvidia-smi.txt`, `result.json`, `job.json`.
- **Le lien** : le manifeste git porte le sha256 de chaque fichier du dataset et la **révision** du dataset. Citation dans le papier **par révision figée** : `…/resolve/<commit-sha>/runs/<run_id>/stdout.log`, **jamais** `/main/`.

**Ne PAS déposer les checkpoints déquantifiés (8 Go pièce) dans `proofs/`** : ce sont des reconstructions reproductibles depuis le checkpoint + `ops/adversary.py`. Déposer le **script** et les sha256.

Coût : le stockage dataset est gratuit à cette échelle ; le Storage Bucket de travail a un free tier puis un tarif au To — négligeable pour des logs, **à surveiller** s'il devait recevoir des overlays.

### 6.4 Le manifeste et son verrou

Une ligne JSONL par run, **y compris les runs ratés** (`status != COMPLETED` reste une ligne). Champs : `run_id` · `claim_ids[]` · `axis` · `arm` · `status` · `job{…, durations{scheduling_secs, running_secs, total_secs}, image_identity, command[], env{}, volumes[{type,source,revision}]}` · `build{git_sha, git_dirty, cargo_lock_sha256, rustc, features, base_image_digest}` · `hw{gpu_name, driver, cuda_runtime, compute_cap}` · `inputs[{role, ref, revision, sha256, bytes}]` · `params{dtype, ctx, windows, limit, seed}` · `result{…, token_fingerprint}` · `vram{peak_bytes, mean_bytes, p50, p95, samples, period_ms, scope, source}` · `logs{path, sha256, dataset_revision}`.

**La règle porteuse** : tout nombre du papier est marqué `[[claim:ID]]`, et **`verify` ÉCHOUE** si l'ID n'a pas de ligne, si un hash ne retombe pas, si `git_dirty` est vrai, ou si la valeur du papier diffère du JSON. Un nombre sans trace devient une **erreur de build**, pas un oubli. Même discipline que `cmd_selftest`, qui refuse déjà un estimateur qui a dérivé.

**`oracle` entre au manifeste comme un run de plein droit**, et `verify` refuse toute ligne d'axe qualité qui ne référence pas un `oracle` PASS sur la même image et le même flavor.

> ⚠️ L'oracle par défaut (`ops/run.py:603-604` : `Qwen/Qwen3-0.6B`, `oracle.rs:30` : F32, 64 tokens) **ne couvre pas la configuration qu'il protège** — c'est le motif « un paramètre à valeur neutre » de CLAUDE.md §5 appliqué au gate lui‑même. L'oracle de campagne est `LLVQ_DTYPE=f16 oracle Qwen/Qwen3-4B 4096 cuda`. Attention : il instancie **deux modèles complets simultanément** (`oracle.rs:65,69`, `vb.get` alloue une copie fraîche par modèle) → 16,1 Go en f16, **32,2 Go en F32** — d'où `l40sx1` et f16 obligatoires.

**Sur `running_secs`** : la page de tarification dit « it is only billed when the Job is **Starting or Running** », or `JobDurations` n'a que `scheduling_secs / running_secs / total_secs`, et « Starting » n'est aucun des trois. **Enregistrer les trois et réconcilier UNE fois contre la page Billing**, puis publier laquelle est la bonne. Sans cela on affirmerait « les secondes facturées » sur une grandeur jamais confrontée à la facture.

### 6.5 Pré‑enregistrement horodaté

`docs/attentes-<date>.md` commité, **tag annoté signé GPG** poussé, **plus** `ots stamp` (OpenTimestamps, ancre Bitcoin, `ots upgrade` le lendemain — à mettre dans la checklist, pas dans la mémoire). Un commit signé prouve **qui**, pas **quand** ; OpenTimestamps est le seul mécanisme qui rende l'antériorité vérifiable sans faire confiance ni à nous ni à une plateforme. Gratuit, 15 min.

`verify` contrôle mécaniquement que la date du tag précède le `created_at` du premier job du manifeste.

**Contenu obligatoire du pré‑enregistrement** (§2.3, §3.4, §9) : (i) on s'attend à ce que le 4 bits calibré nous domine sur la qualité ; (ii) on s'attend à ce que GPTQ w2g128 s'effondre — **et ce n'est pas une victoire** ; (iii) on s'attend à gagner à gauche de la courbe débit‑distorsion et à perdre à droite, **sans savoir où est le croisement** ; (iv) le seuil de décision du certificat MMLU (0,5 pp) ; (v) la définition de l'axe x de la figure principale (§7.4).

### 6.6 Ce qui reste non prouvable — section déclarée, pas note de bas de page

1. **La facture HF est privée.** On publie les secondes facturées et le tarif public ; l'arithmétique est la nôtre.
2. **HF ne publie aucune durée de rétention de logs.** L'archive **est** la preuve ; la page du job n'en est pas une.
3. **Aucune garantie d'isolation sur matériel loué.** Aucune revendication de vitesse ne sort d'un Job (§4).
4. **Le pic VRAM échantillonné à 1 Hz est une borne inférieure.**
5. **La correspondance image ↔ arbre git est ASSERTÉE**, pas attestée, tant que le build passe par un Space.
6. **Le gate G6 (2,07×) est Metal et ne peut être reproduit sur cette cible par personne.** C'est le coût du changement de cible : on achète des preuves de run sur la qualité, on **coupe entièrement le noyau du périmètre reproductible**. À assumer, pas à diluer.
7. **La variance de re‑quantification n'est pas mesurée** (n=2, configurations différentes, aucun sigma — v1 §4.5 inchangé).
8. **On mesure la RECONSTRUCTION, pas l'arithmétique fusionnée du noyau adverse.** L3 borne l'écart, il ne l'élimine pas. Point 10 des non‑résolus de v1, il survit mot pour mot avec un adversaire différent.

---

## 7. Plan statistique

### 7.1 Ce qui survit intégralement de v1 §4

- **Les trois sources de variance et leurs traitements** : mesure (se prouve une fois, puis n=1) · échantillonnage (se supprime par recensement, sinon barre appariée) · méthode (hors périmètre, déclaré).
- **La statistique publiée en perplexité, trois lignes et jamais moins** : (1) ppl au recensement 73 fenêtres, **sans barre** ; (2) **delta apparié de log‑vraisemblance par fenêtre**, sem et IC95 t apparié ; (3) ratio = `exp(delta)`, IC transporté. **Interdit** : une ppl absolue accompagnée d'une barre (±14 %, elle décrit la difficulté du corpus).
- **La barre déjà reconstruite** : sd inter‑fenêtres 0,056119, **sem 0,016200** sur 12 fenêtres, corrélation inter‑bras 0,975134. Projection 73 fenêtres : sem ≈ 0,00657 nats, 2σ = **0,0131 nats = 1,32 % sur le ratio**. Pas de correction de population finie (préfixe contigu, pas tirage sans remise).
- **Les seuils de défendabilité (§4.4)** : « 16,9617 < 17,04 » = 0,31 sem, **indéfendable, à retirer** ; l'auto‑critique « +3,06 % de nats » n'est pas plus significative (t = 0,56 sur df = 11). La formulation juste reste celle de v1.
- **MMLU : bootstrap apparié stratifié par matière** sur le dump T2 (10 000 tirages, < 1 s), McNemar en secondaire étiqueté « non pondéré ». Gain d'appariement ~1,40× en SE.
- **§4.5 en entier** : les 7 % de variance de méthode restent retirés du dossier comme observation de variance.

### 7.2 Ce que CUDA change : le déterminisme acquiert une clause

Sur Metal, « deux rejeux identiques rendent le même nombre » est plausible. Sur du matériel loué, **cuBLAS n'est documenté bit‑à‑bit reproductible qu'à version, architecture ET nombre de SM constants** — et rien ne garantit que deux jobs `l40sx1` tombent sur la même configuration. Deux étages, dans cet ordre, **T6 en prérequis** :

- **Étage 1, même job, même conteneur, même carte, deux processus** : `bash -c 'set -euo pipefail; ppl … && ppl …'` (non‑login `-c`, pas `-lc` : un login shell re‑source `/etc/profile` et peut réécrire `PATH`/`LD_LIBRARY_PATH` pour zéro bénéfice ; `&&` et non `;` sinon le job sort 0 malgré un premier échec). Attendu : égalité exacte sur les 12 NLL à 9 chiffres. Fondement : les réductions de candle passent **toujours** par `FastReduce` (`cuda_backend/mod.rs:1666`), **zéro atomique** — le kernel `sum_*` à `atomicAdd` n'est jamais atteint ; et le chemin d'évaluation n'a aucun `rayon`.
- **Étage 2, deux jobs distincts, même flavor.** **C'est LUI qui licencie n=1, et c'est LUI qui peut échouer.**

Si l'étage 2 échoue, le budget de répétition triple et toutes les barres sont à refaire. C'est pourquoi il se paie 0,45 $ maintenant plutôt qu'après.

### 7.3 Ce que CUDA change : le recensement MMLU devient une dépense, pas une nuit

Voir §3.4. Il supprime la seule barre d'échantillonnage qui pollue l'argument le mieux fondé du dossier.

### 7.4 Le piège de l'axe x, à pré‑enregistrer avant de mesurer

La campagne va produire une courbe qualité = f(bits/poids), LLVQ à 2,1595 et la famille adverse à 2,148 / 3,156 / 4,156 / 4,500. Lue naïvement, cette figure dit « à 2,16 bits nous battons le 4 bits », donc que 2,16 est une **position déployable**. Or le seul format que le noyau rapide sait lire coûte **6,525 b/poids en mémoire** (§3.3), et cette campagne ne peut ni le mesurer ni le contredire.

**Décision, à pré‑enregistrer** : publier la figure avec **deux coordonnées pour le point LLVQ** — 2,1595 (disque, mesuré sur le fichier) et 6,525 (mémoire du seul décodeur rapide, calculé, non mesurable sur cette cible) — reliées par une flèche horizontale légendée. C'est un résultat négatif qui se déclare et ne s'annule pas, et le déclarer nous‑mêmes vaut mieux qu'un relecteur qui le trouve.

**Et la collision de v1 §4.4 change de nature** : la marge sur l'abscisse tombe de 4,0 % à **0,52 %**, donc si le point LLVQ tombe à moins de 2σ apparié de la courbe interpolée, la figure montre une **frontière**, pas une victoire — mais l'objection « votre abscisse n'est pas comparable » disparaît.

---

## 8. Ordre d'exécution, temps GPU et budget

Tarifs : `l40sx1` **0,030 $/min** · `cpu-upgrade` **0,0005 $/min** · `rtx-pro-6000` 0,0458 $/min. **La facturation court pendant Starting ET Running**, donc chaque job porte un plancher : pull d'image + téléchargement du checkpoint (8,06 Go à ~42 Mo/s mesurés sur le run 32B = 3,2 min) + démarrage ≈ **6‑10 min**. Le **`timeout` est le plafond de coût** — exact, connu avant lancement, et il remplace avantageusement `--max-usd`.

> 🚫 **`ops/run.py estimate` et la garde `--max-usd` sont INAPPLICABLES à cette campagne.** `cost_table()` (`run.py:186-219`) multiplie le nombre de poids par `QUANT_CORE_SEC_PER_WEIGHT` — un modèle de l'**encodeur Leech** et de la factorisation, dont aucun n'existe dans un job de scoring. Sur Qwen3‑4B il annonce ~8 h pour ce qui en fait 30‑90 min. Il se trompe **dans les deux sens** et le plafond qu'il alimente ne protège de rien.

| Étape | Contenu | Humain | Machine | $ | Livrable si on s'arrête là |
|---|---|---|---|---|---|
| **E0** | T0, O1, O2, D1, R6, PR ; publication du Space ; pré‑enregistrement commité + tag signé + `ots stamp` | **8 h** | build 40‑70 min (non facturé) | **0 $** | Une **note de correction** publiable seule : le ratio disque en une ligne mesurée (§3.1), la résolution du cache KV (§3.3), le retrait des claims non défendables (v1 §4.4), la correction macro/micro du README, le retrait de la cellule mémoire trompeuse |
| **E1** | **J0 pré‑vol** sur `cpu-upgrade` (5 min) : montage du volume artefact, `sha256sum` = 1 770 527 533 o, `nvidia-smi` absent (contrôle négatif), `df -h`, `time hf download` de 500 Mo → **le débit Hub→Job, qui n'a qu'une seule valeur extrapolée dans tout le dépôt**. Puis **oracle** `LLVQ_DTYPE=f16 oracle Qwen/Qwen3-4B 4096 cuda`. Puis **pilote** : A1 et C, `ppl 4096 3 cuda`, appariés dans **un seul conteneur**, métriques échantillonnées en direct | **1 h** | 5 min + 10 min + 25 min | **0,01 + 0,35 + 0,80 = 1,15 $** | Quatre nombres qui n'existent nulle part : **s/fenêtre réel sur L40S** (l'incertitude passe d'un facteur 3 à 1,1 et recalibre tout le reste), **pic ET moyenne VRAM**, s de décodage du scellé sur un vCPU serveur, débit Hub→Job |
| **E2** | T6, T2, JS ; rejeu MMLU limit=40 **local sur Metal** (doit rendre 70,42) ; puis re‑certification **CUDA** | **5 h** | 44 min local + 35 min GPU | **0 $ + 1,05 $** | **Le certificat transporté (ou l'écart de backend chiffré, qui est un résultat que le dépôt n'a jamais eu)** |
| **E3** | B‑py ; dé‑risquage GPTQ sur Qwen3‑0,6B ; production B0 (déquant AWQ, CPU), B1, B2 (GPTQ), B3 (bnb) ; L1‑L4 sur chaque ; push public des 4 checkpoints | **6 h** | 20 min + 3 × 70 min + 15 min + 4 × 35 min CPU | **0,60 + 4,70 + 0,45 + 0,10 = 5,85 $** | La **famille adverse**, avec sa table de débits vérifiée à l'octet et son contrôle d'iso‑périmètre (74/72) |
| **E4** | Contrôles CI et C8 ; puis **un job par objet**, chaîné : `ppl 4096 12` + `ppl 4096 73` + `mmlu 40`, métriques échantillonnées. 6 objets : A1, C, B0, B1, B2, B3 | **2 h** | 6 × 60‑80 min | **11 – 15 $** | **La figure débit‑distorsion à deux panneaux** (ppl + MMLU), la table de tête à 73 fenêtres, le profil par matière superposé, les tests appariés. **Le papier devient écrivable.** |
| **E5** | Déterminisme étage 2 (un `ppl 4096 12` rejoué dans un job distinct) | 30 min | 20 min | **0,60 $** | La **licence de n=1** — ou son refus chiffré |
| **E6** | `ops/floor.py` ; dépouillement VRAM des séries d'E4 ; O5 (record/manifest/verify) ; publication du dataset de preuves | **5 h** | 0 | **0 $** | Les **trois lignes mémoire** (§3.3) et le **manifeste vérifiable** |
| **E7** *(opt.)* | Recensement MMLU sur A1, B0, C, **trois jobs parallèles** | 30 min | 3 × 2,3‑4,8 h | **12 – 26 $** | Barre d'échantillonnage à **zéro**, comparabilité directe aux 70,2 / 60,7 du papier |
| **E8** *(opt.)* | Jalon Metal §4.4 : Grouped32 / Flat32 sur le modèle entier | **1 h** | 3 min local | **0 $** | La courbe bits↔vitesse **sur un seul protocole**, et la réponse empirique à l'arbitrage n° 4 |
| **E9** *(opt.)* | A2 (`q4b-e4.llvq`) en `ppl 4096 12` seul | 15 min | 25 min | **0,75 $** | Le premier chiffre de qualité de l'embedding int4 — **sans comparateur**, donc à publier comme point isolé |

**Cumul E0→E6 : ~28 h humain, ~10 h GPU, 16 – 24 $. Provisionner 40 $.**
**Avec E7 + E8 + E9 : ~30 h humain, ~22 h GPU, 29 – 51 $. Provisionner 75 $.**

**Point de décision après E4** (déplacé de v1 §6, où il était après E1) : si le 4 bits calibré se dégrade à ~1‑2 % là où nous sommes à +38,5 %, **la thèse produit est tranchée sur un 4B** et le papier devient un papier de **noyau et de format**. Engager le port CUDA (arbitrage n° 4, 5‑10 jours) avant E4 serait investir dans une course dont la conclusion est peut‑être déjà écrite.

**Chaque étape laisse un livrable publiable**, et E0 en laisse un qui ne coûte pas un dollar.

---

## 9. Les arbitrages que l'utilisateur doit trancher

> ✅ **Tranchés le 2026-08-04.**
>
> | # | Décision | Conséquence |
> |---|---|---|
> | 1 | **`l40sx1`** (48 Go, 1,80 $/h, sm_89 natif) | Carte unique de toute la campagne. `a100-large`, `a10g-*`, `t4-*` **interdites** (l'image cap 89 n'y charge aucun noyau). Repli `rtx-pro-6000` seulement si E1 montre la L40S sous 1,5× le M3 Max |
> | 2 | **Trois bras, point : LLVQ 2 bits, AWQ officiel, f16 d'origine.** GPTQ w2/w4 et bnb NF4 avaient été proposés puis **écartés par l'utilisateur** — ne pas les réintroduire | On compare des **produits** (2,16 contre 4,16 b/poids), pas des codebooks à débit égal. Tout ce que le plan dit de B1/B2/B3 devient sans objet |
> | 3 | **Publier le checkpoint AWQ déquantifié** en dépôt HF public | Débloque le scoring (`hf-hub` ne lit jamais `HF_TOKEN`) **et** la rejouabilité par un tiers. Obligatoire pour AWQ : la voie overlay lui est interdite (§2.4). À étiqueter : reconstruction de mesure, pas un modèle à utiliser |
>
> Restent ouverts : n° 3 (profondeur MMLU — à décider à E4), n° 4 (port CUDA du noyau — à décider après E4), n° 5 (provenance de l'image), n° 7 (republier ou garder les deux jeux de chiffres).

### Les arbitrages, en détail

### 1. Sur quelle carte engager la campagne

| | (a) `l40sx1` **recommandé** | (b) `rtx-pro-6000` | (c) `l4x1` |
|---|---|---|---|
| $/h | 1,80 | 2,75 | 0,80 |
| VRAM | 48 Go | 96 Go | 24 Go |
| bande passante | 864 Go/s | 1792 Go/s | **300 Go/s** |
| compute cap | **sm_89 NATIF** | Blackwell, **par JIT PTX** | sm_89 natif |
| f32 possible ? | oui | oui | **non** |
| coût E0→E6 | **16 – 24 $** | 20 – 30 $ | 14 – 30 $, temps mural ×2‑3 |

**Recommandation : (a).** Le seul compromis qui ait toutes les propriétés à la fois — image exacte sans JIT, marge de dtype, ~2,2× la bande passante du Mac de dev, tarif d'une carte que le lecteur loue partout. (b) est ~1,5× plus rapide pour 1,53× le prix (donc à peu près neutre au travail unitaire) mais rouvre une question de provenance : **on ne sait pas si les runs 8B et 32B ont JIT‑é depuis compute_89 ou si l'image a été rebâtie ce jour‑là**, et c'est un trou sur deux runs déjà publiés. (c) est le faux ami : 300 Go/s est **sous** le M3 Max, donc même prix total pour deux fois le temps, plus le risque d'OOM en f32.
**Dans tous les cas : jamais `a100-large` (sm_80), `a10g-*` (sm_86) ni `t4-*` (sm_75) — l'image cap 89 ne peut y charger aucun noyau.**

### 2. Périmètre du bras adverse

| | (a) B0 seul | (b) **courbe 4 points** *recommandé* | (c) + EXL3 |
|---|---|---|---|
| humain | 4 h | **6 h** | +2‑4 jours, issue incertaine |
| machine | ~0,05 $ | **~5,85 $** | +1,4 $ machine, décodeur non borné |
| revendication | « à périmètre identique, notre fichier est ×1,506 plus petit que le 4 bits officiel de l'auteur du modèle, pour une qualité de X » — verdict binaire, très probablement perdu | **« à 0,52 % de débit près, le réseau de Leech se situe [au‑dessus/en dessous] de la famille affine calibrée »** + une pente, donc un croisement localisable | le seul concurrent 2 bits que le papier nomme (QTIP à 17,04) |

**Recommandation : (b).** L'écart entre (a) et (b) est de ~6 $ et 2 h — le prix d'un café pour la différence entre un verdict binaire et une figure. (c) est bloqué non par la production (gptqmodel sait produire de l'EXL3 pour ~1,4 $) mais par le **décodeur** : `gptqmodel` lève `Unsupported format exl3`. Trou à déclarer avec sa cause exacte, à rouvrir seulement si (b) montre que la thèse produit tient encore.

### 3. Profondeur MMLU

| | (a) limit=40 partout | (b) **(a) + recensement sur A1, B0, C** *recommandé* |
|---|---|---|
| coût | inclus dans E4 | **+12 – 26 $**, 3 jobs parallèles |
| barre | ±1,28 / ±1,36 pp | **exactement nulle** (correction de population finie) |

**Recommandation : (b), et c'est la dépense la plus rentable de la campagne.** Ce qui coûtait « deux nuits de Mac, optionnel » coûte le prix d'un repas et quelques heures de mur en parallèle. Elle supprime la seule barre qui pollue la reproduction de la baseline du papier à 0,22 pp. **Rappel : recensement et limit=40 ne s'apparient pas** (§3.4).

### 4. Porter le noyau sur CUDA — la vraie question, et elle n'est pas urgente

| | (a) **non, jalon Metal d'abord** *recommandé* | (b) port CUDA |
|---|---|---|
| humain | **1 h** (E8) | 5‑10 jours (≈220 l de CUDA C, ≈250 l de harnais, plus le tuning, non borné) |
| machine | **0 $** | 25 – 45 $ (4 cartes × rejeux internes) |
| ce qu'on apprend | la courbe bits↔vitesse Grouped32/Flat32/Slot32 sur **un seul protocole, le modèle entier** — le trou que `fiche-4b:476` reconnaît | « ×R contre le FP16, décodeur Leech multi‑coquilles fusé, 252 matrices, modèle entier, sur la classe de matériel du papier » — **ce qui n'existe nulle part, papier compris** (leur noyau est mono‑coquille M=3, mono‑couche, et déclaré plus lent que QTIP par ses auteurs) |

**Recommandation : (a) maintenant, (b) éventuellement après E4.** Trois raisons. (i) Le jalon à 1 h répond empiriquement à la question que (b) coûterait des semaines à poser. (ii) Le byte‑ratio interdit structurellement à `Slot32` de battre un 4 bits compétent en batch 1 : **×1,16 à ×1,53 contre nous** selon le group_size et le périmètre — la cible du port serait le **FP16 et leur Table 7**, pas le 4 bits, et il faut l'énoncer avant d'écrire une ligne. (iii) Trois obstacles nommés par dureté (`fiche-4b` §6.10) : pas de cache KV donc jamais de matvec (~1,5 j) ; **la rotation GPU, non bornée et payée par le seul bras quantifié** ; le prefill. **Même après le port ET le cache, tout rapport de vitesse LLVQ reste un MAJORANT** — à porter dans l'**en‑tête** de la table, pas dans une note.
Route technique si (b) : **NVRTC à l'exécution** (`candle-core 0.9.2` réexporte `cudarc`, active sa feature `nvrtc`, et expose publiquement `get_or_load_custom_func`, `cuda_stream()`, `cublas_handle()`), pas nvcc au build — 20 variantes balayées sans reconstruire le Space. `libnvrtc` est **déjà présent** dans l'image (cudarc en `dynamic-linking` émet `-lnvrtc`, donc les binaires actuels ont un DT_NEEDED dessus, et le run 8B de 4,18 h le prouve). Mais **NVRTC ne compile pas « pour la carte présente » par défaut** (`CompileOptions::arch = None`) : il faut le passer explicitement. Et la baseline doit être **cuBLAS**, pas notre `tv_f16` écrit à la main.

### 5. Provenance de l'image

| | (a) Space HF (statu quo) | (b) **GitHub Actions → ghcr.io** *recommandé si 3‑4 h* |
|---|---|---|
| coût | 0 $, 0 effort | 0 $ (GHA gratuit sur dépôt public), 3‑4 h |
| revendication | « les runs ont tourné » | **« cette image a été construite par GitHub depuis ce commit exact, voici l'attestation SLSA »** + `docker pull` possible sans compte HF |
| risque | aucun digest dans `JobInfo`, un rebuild remplace silencieusement l'image de tous les runs précédents | runner `ubuntu-latest` 4 vCPU / 16 Go / ~14 Go de disque : une image CUDA devel + `target/` sous LTO, c'est exactement ce qui a déjà OOM. Prévoir `CARGO_BUILD_JOBS=1`, 40‑70 min |

Le dépôt GitHub étant **déjà public**, (b) n'ajoute aucune surface d'exposition. C'est la différence entre une provenance **assertée par nous** et une provenance **attestée par un tiers** — dans une phase où le projet répare précisément des chiffres que personne ne pouvait rattacher.

### 6. ⚠️ Publier les checkpoints déquantifiés du bras adverse — décision de l'utilisateur, pas de l'auteur

Le scoring exige un dépôt HF **public** (§2.5 : `hf-hub` ne lit jamais `HF_TOKEN`), et la rejouabilité par un tiers l'exige aussi (`run_job` n'a **aucun** paramètre d'authentification de registre). Cela signifie **publier 4 dérivés d'un checkpoint Qwen** (~32 Go), clairement étiquetés comme des reconstructions de mesure et non des modèles à utiliser. **C'est une publication, elle engage le compte, et elle doit être autorisée explicitement.**
Alternative sans publication : le prélude bash qui écrit le token dans `$HOME/.cache/huggingface/token` (§2.5) — **à tester une fois pour 0,003 $ avant d'engager quoi que ce soit** —, au prix de la rejouabilité par un tiers.

### 7. Republier les chiffres Metal, ou garder les deux

**Recommandation : garder les deux, étiquetés par matériel**, et ne changer le couple de tête que si l'écart CUDA↔Metal dépasse la résolution établie en E5. La paire Metal a des traces primaires (corps du commit `8c17eff` pour 16,9415 / 12,2361, `docs/mmlu-micro-2026-08-02.log` pour 56,09 / 70,42 avec le profil par matière) ; la paire CUDA aura des identifiants de job **et** un bundle rejouable. L'écart entre les deux est lui‑même une mesure publiable — **la sensibilité au backend d'un résultat de quantification 2 bits, qu'aucun papier du domaine ne rapporte**.

**Ce qui doit changer sans attendre, et ne relève d'aucun arbitrage** : la section « Reproducing » du README publie uniquement des invocations `--features metal … metal`, sur un objet dont le même README dit « no CUDA kernel ». Le chemin CPU existe (`eval.rs:46-49` accepte `"cpu"`, README:121‑123 le documente) mais il est ~7× plus lent — donc le seul chemin **praticable** publié est Apple‑only. **Une commande CUDA exécutable par un tiers doit y figurer, écrite à partir de l'argv réel d'un job terminé.**

---

## 10. Points non résolus

1. **Le facteur de vitesse CUDA/Metal n'est pas mesuré.** Tous les dollars de §8 sont extrapolés du Metal (10,0‑17,5 s/fenêtre à ctx 4096 ; 2 620‑2 805 s pour 2 280 questions MMLU) avec une bande [0,5× ; 1,0×] sur L40S. **E1 referme ce trou pour 1,15 $ et rien d'autre ne le peut.** Ne pas engager E4 avant de l'avoir lu.

2. **Le pic VRAM est un CALCUL, pas une mesure**, et le comportement de l'allocateur CUDA de candle est inconnu : s'il met en cache et fragmente, le pic observé peut dépasser nettement les ~16‑19 Go estimés en f16. Mesurable en E1.

3. **La moyenne VRAM peut être égale au pic.** Si l'allocateur (`cuMemAllocAsync` sur le pool par défaut, seuil de libération 0) ne rend jamais la mémoire, « pic » et « moyenne » sont deux noms pour un seul nombre. **Contrôle à 0 $ dans E1 : la série descend‑elle ?** À vérifier avant de promettre les deux.

4. **Le certificat MMLU 70,42 tiendra‑t‑il sur CUDA ?** `oracle` garantit `max|Δhidden| = 0` sur la passe avant, **pas** le chemin logits → f32 → argmax de MMLU, qui décide par `max_by` sur quatre logits — un seuil, donc les quasi‑ex‑æquo basculent sur une dérive de dernier bit. Personne n'a la distribution des marges. Le seul point de données inter‑backend du dépôt (19,4990 contre 19,5038, 0,025 %) est **CPU‑x86 ↔ Metal, pas CUDA** (le corps du commit `00c6e2d` oppose « 59 % sur Metal » à « 12 % sur un job CPU x86 »). Dumper aussi les quatre logits (~5 lignes de plus dans T2) rendrait la question quantitative.

5. **Les corpus ont‑ils déjà bougé depuis le 2026‑08‑02 ?** Les sha des révisions de `Salesforce/wikitext`, `cais/mmlu` et `allenai/c4` ne sont pas relevés. Si l'un a bougé, **le certificat 70,42 lui‑même serait à réétablir avant tout le reste.** 10 minutes, à faire en E0.

6. **La révision du checkpoint `Qwen/Qwen3-4B` (`1cfa9a7208912126459214e8b04321603b3df60c`) n'a pas été vérifiée par moi** — elle vient d'une lentille. À confirmer au pré‑vol. Celles de l'artefact (`f00daa7b…`) et de l'AWQ (`74d4bd2b…`) sont vérifiées.

7. **`cais/mmlu` n'a jamais été téléchargé depuis un job.** Public et petit, donc risque faible, mais c'est la seule dépendance de données non prouvée à distance, et elle est sur le chemin critique du certificat.

8. **Le montage paresseux d'un volume face à une lecture séquentielle.** `sealed::load` lit 1,77 Go de bout en bout avec un `BufReader` de 1 Mio ; la doc annonce des volumes « fetched lazily ». Peut être plus lent qu'un téléchargement franc, ou transparent. À chronométrer en E1.

9. **Le décodage du scellé se paie par PROCESSUS, pas par job** — 3 à 5 min sur un cœur P de M3 Max, 210‑300 s sur un vCPU loué, et le bras A l'endure 3 fois dans un job chaîné (ppl‑12, ppl‑73, mmlu‑40). Amortir le téléchargement demande un job ; amortir le décodage demanderait **un processus**, ce que les binaires actuels ne savent pas faire. Non chiffré comme ticket.

10. **La variance de re‑quantification reste hors périmètre et non mesurée** (v1 §4.5 : n=2, configurations non identiques, aucun sigma). Aucune barre publiée ne la couvre. Et `LLVQ_CALIB_SEED` ne bouge pas que les offsets : l'objet publié n'appartient pas à la population dont on estimerait le sigma.

11. **Le trou de provenance sur deux runs déjà publiés** : le 8B (11,48 $) et le dé‑risquage 32B (5,43 $) ont tourné sur `rtx-pro-6000` (Blackwell) avec une image `CUDA_COMPUTE_CAP=89`. Soit le JIT PTX a fonctionné, soit l'image a été rebâtie ce jour‑là. **Sans conséquence pour cette campagne** (`l40sx1` est sm_89 natif), mais à trancher avant de citer l'image du 8B.

12. **`A2` (`q4b-e4.llvq`) est orphelin.** Aucun quantifieur CUDA ne touche l'embedding, donc son unique comparateur a disparu avec MLX. Il peut être scoré (E9, 0,75 $) mais le chiffre sera un point isolé, sans famille adverse en face.

13. **Le noyau reste hors du périmètre reproductible, et rien ne le remplace.** Le 2,07× reste mesuré — sur un matériel que le lecteur ne possède pas et sur lequel il ne peut lancer aucun job, c'est‑à‑dire dans exactement la situation que le déplacement de cible visait à corriger.
