# Pistes fondamentales — la campagne du 2026-08-08

> Exploration multi-agents sur les cinq critères du projet (ppl, MMLU, VRAM,
> stockage froid, vitesse). Méthode : 5 lecteurs (noyau, quantifieur, format,
> moteur, littérature) → 74 faits sourcés fichier:ligne → 5 lentilles de
> génération sous barre d'admission « ampleur du switch Slot32 → Planes14 » →
> **un sceptique par idée avec mandat de réfuter** (lentilles imposées :
> arithmétique de comptabilité / duplicata d'impasse mesurée / faisabilité
> contre le code) → synthèse. **27 idées brutes, 12 survivantes, 15 réfutées.**
> 38 agents, lecture seule, aucun run GPU.
>
> ⚠️ **Rien de ce document n'est mesuré sur carte.** Tout est de la lecture de
> code vérifiée, de l'arithmétique recoupée, et des chiffres du dépôt relus
> dans leur comptabilité d'origine. Chaque piste porte son expérience
> falsifiante et son prix.
>
> ⚠️ **Provenance de la lentille littérature** : arxiv, huggingface,
> aclanthology, proceedings.neurips, developer.nvidia et semanticscholar sont
> tous bloqués par le proxy de sortie. Les chiffres externes cités
> (EoRA, Recover-LoRA, HPTQ, AQLM, SpinQuant, KIVI, GLQ, QTIP) viennent de
> github/raw.githubusercontent ou de la connaissance du modèle — **à
> re-sourcer avant toute citation externe.**
>
> Deux corrections que cette campagne apporte au dossier existant sont dans le
> §0 : la part non attribuée du jeton n'est **pas** 5,3 ms mais quelque part
> entre 2,3 et 5,3, et les **Go/s publiés des layouts à exceptions sont faux**.

*Base : Qwen3-4B scellé (1,771 Go, 2,1696 b/poids écrits), L40S, `bin/fusedrun` Planes14 + embedding q8 = 87,7 tok/s dans 2,60 Go contre 43,5 dans 8,04.*

---

## 0. Le budget, et la seule indétermination à fermer avant tout

**Jeton de 11,40 ms** (décode, contexte ~69) :

| poste | ms | source |
|---|---|---|
| 252 projections Planes14 | 5,129 | banc E2, même carte, même fichier |
| 252 rotations d'incohérence | ~2,39 | extrapolé de `rotation-cuda` (8,05/6,72/19,30 µs), **7 par couche, pas 4** |
| ~970–1 400 lancements candle | 2,9 à 3,5 | résidu / 3,63 µs mesurés |
| lm_head q8 | 0,598 | `phases-2026-08-07` |
| embed + argmax | 0,12 | idem |

**Octets par jeton** : 2,18 Go de poids + 0,413 embed q8 + 14·C de cache KV (C = 147 456 o/position).

⚠️ **Deux soustractions incompatibles.** Bras fusé : 10,439 − 5,129 = 5,31 ms hors projections. Bras dense : 13,291 − 11,014 = 2,29 ms. Les deux bras partagent pourtant le même hors-projection. L'écart de 3,0 ms est un confondant de protocole (candle/cuBLAS contre `tv_f16` maison, et le banc E2 entrelace 5 bras). **La part non attribuée est entre 2,3 et 5,3 ms — pas 5,3.** Trois chiffres de ce dossier en dépendent (rotation, lancements, fusion candle).

> **Gate G1bis, ~1 $, une demi-journée** : instrumenter `fusedrun` par événements CUDA sur les 3 postes *dans le même processus*. Rien d'autre sur l'axe vitesse ne doit être engagé avant.

Second correctif de comptabilité, qui recadre tout l'axe VRAM : **les Go/s publiés des layouts à exceptions sont faux**, la passe de correction relit des activations non comptées. Planes12x réel = 1,97 + 0,55 = **2,52 Go → 455 Go/s**, pas 356. Golay70 = 1,63 + 1,08 = 2,71 Go → **324 Go/s**, pas 195. Corollaire décisif : **à décodeur constant (famille planes), le temps suit exactement les octets *réellement lus, exceptions comprises*.** Planes12x n'est pas lent parce qu'il est petit ; il est lent parce qu'il lit *plus* que Planes14.

---

## 1. Axe VRAM

Point de fonctionnement : **5,11 b/param modèle entier** (Planes14 + embed q8), contre **5,30 pour l'AWQ 4 bits réellement mesuré**. Payload Planes14 = 4,804 b/poids, dont 3 plans de niveau 62 %, smask 21 %, classe 7,8 %, padding 5,2 %.

### V1 — Le champ classe fait 9 bits pour 5,62 bits d'entropie ★
- **Mécanisme** : les 128 plus grosses classes couvrent 99,530 % de la boule cap 12 ; 2 des 9 bits décrivent 173 classes qui pèsent 0,47 %. Appliqués au flux principal de Planes12x (charge utile déjà 82 bits dans 96), le record tombe sur **80 bits = 10 octets pile**.
- **Chiffre** : 4,804 → **3,67–3,70 b/poids** (−23 %), modèle entier **5,11 → ~4,10 b/param**, 2,60 → 2,08 Go. Coût codebook Δ = 0,0068 bit → MSE ×1,00039, −0,04 pt de rétention (ou qualité *exacte* en routant les 0,47 % de blocs restants vers l'overlay, +0,03 b/poids). Vitesse : ~neutre (2,22 Go réels contre 2,18).
- **Blocage** : `CLASS_BITS = 9` est gelé depuis v1 comme ⌈log₂ 383⌉ — une borne d'adressage, jamais un budget d'entropie. `classhist` avait mesuré l'entropie de l'**index** (46,6536 b, conclusion « rien à gagner ») et personne n'a mesuré la marginale de **classe**.
- **Coût** : 2–3 j-h (filtre de classes = 1 ligne façon `with_level_cap`, permutation d'identifiants dans l'en-tête matrice, transcodeur Planes12x réutilisé), ~5 $.
- **Falsifieur, 0 $, CPU, quelques minutes** : faire compter à `bin/classhist` la masse cumulée des 128 classes les plus peuplées du fichier scellé. Le modèle uniforme prédit 99,53 % (il a déjà reproduit L=5 à 0,13 pp et E2 à 0,17 pp). Sous 98 %, le stride de 10 octets s'évapore.

### V2 — Le plafond L≤4 **natif** (le verdict « mort » vient d'un swap post-hoc) ★
- **Mécanisme** : `bin/lswap` remplace le point déjà choisi par un voisin L≤4 **après** la fermeture de la chaîne GPTQ — l'erreur n'est compensée par aucune colonne suivante et le gain n'est pas re-choisi. Un run natif à `level_cap = 4` maximise cos θ sur le sous-ensemble et son erreur traverse `solve_block` comme n'importe quelle autre. Sans blocs L=5, **l'overlay d'exceptions disparaît entièrement**.
- **Chiffre** : perte de codebook = 0,0515 bit (L≤4 = 96,49 % des *points* de la boule 12 — grandeur recalculée, l'usage à 96,62 % n'est pas la bonne). Coût mesuré nativement par `bin/lcap` : **+0,34 % de MSE de bloc**, pas les +4,75 % de ppl du swap. Gain : record 10 o **sans aucune exception** → **3,4769 b/poids**, 1,579 Go, **3,96 b/param**. Et c'est là que la vitesse arrive : 5,129 → ~3,97 ms de projections (le plancher de lancement de 0,915 ms ne tombe pas avec les octets), soit **+9 à +15 % de tok/s**.
- **Blocage** : une perte d'information entre un journal prudent (`verdicts-lot-b` B6 : « majorant à double titre ») et son résumé dans CLAUDE.md (« mort en qualité »). Plus une croyance héritée : « depuis Planes14 un plafond L ne rapporte rien » — vrai pour un stride de 14 o, faux pour un record de 10.
- **Coût** : l'A/B est **quasi gratuit** — `level_cap` est déjà câblé de `BallSearcher::with_level_cap` au suffixe `L<n>` du parseur de `smoke.rs`. 3 blocs natifs L4 contre L5 = ~10 min, ~1 $.
- **Falsifieur** : cet A/B. Prédiction ≤ 1 % de ppl. Au-delà de 2 %, l'argument tombe et V1 reste seul.
- ⚠️ **Coût politique** : la marge sur QTIP passe de 0,58 % (16,9415 vs 17,04) à 0,3–0,4 %, c'est-à-dire **sous la dispersion propre du projet** (σ ≈ 0,7 % sur 3 blocs, 7 % entre deux runs réputés identiques). Le titre G5 devient statistiquement indéfendable. À arbitrer explicitement.

### V3 — Queue `KeepExact` en f16 (deux axes, gratuit)
16 957 440 poids (0,467 %) stockés en **f32** sur la carte et dans le fichier, alors que le bras dense calcule déjà en f16. **−33,9 Mo de VRAM (0,075 b/poids) et −33,9 Mo de fichier**, sans toucher un poids quantifié ni le format d'index. Coût : quelques heures. Blocage : personne n'a regardé le dtype de la queue ; elle pèse 15× sa part de poids dans le fichier.

### V4 — Golay9 : le plafond de l'échelle, à ne PAS prendre comme point produit
Interdire au chercheur les 47 classes hors alphabet base-4 (7,60 % de la boule, Δ = 0,114 bit, +0,74 % de MSE) supprime *toutes* les exceptions de Golay70 : **3,145 b/poids, 3,61 b/param, 1,428 Go**. Un 70B + 8k de KV passerait de 45,9 à 31,7 Go — d'une carte 48 à une carte 32. **Mais la vitesse est réfutée** : la passe principale de Golay70 vaut ≥ 5,89 ms (calibrée sur le coût par exception de Planes12x, dont le chemin de correction fait *strictement plus* de travail), donc Golay9 ∈ [5,89 ; 7,86] ms — **0,65× à 0,87× Planes14, une régression**. La cause est celle que le dépôt a déjà nommée le 07-08 : le décodage à double coset borne le noyau en calcul (~19 instructions/slot contre ~11, plus un gather `cwtab[4096]` par bloc). À garder comme point de repère VRAM pour la thèse 70B, pas comme layout.

### V5 — Le gisement structurel intact : les signes du coset impair
Sur les 50 % de blocs impairs, les 24 bits de signe sont **entièrement déterminés** par le mot de Golay. **0,4976 b/poids = 10,4 % de Planes14**, et ni Planes14 ni Planes12x ne l'exploitent. Variante chiffrée : 1 bit de raffinement par slot côté impair au lieu de 2 plans → net **+0,405 b/poids** (Golay70 à ~3,18). Prix : record impair 46 bits, pair 70 → **fin du stride uniforme**, précisément ce qui a fait gagner Planes14 (1,14× à contenu identique). À n'ouvrir que si l'on accepte deux flux par coset. Petits compléments non exploités : signes des slots nuls (0,139 b/poids, prouvablement constants), contrainte de somme mod 8 (0,021), index d'exception u32 → delta u8 (0,074 sur Golay70).

### V6 — Le budget VRAM publié est un budget à **contexte nul**
C = 147 456 o/position. **+0,60 Go à ctx 4096, 6,04 Go à 40 960** (= `max_position_embeddings`), soit 2,3× le modèle 2 bits entier. Sur un 70B : 327 680 o/jeton → 10,74 Go à 32k, et **42,2 + 10,74 = 52,9 Go ne rentre pas dans 48**. Avec un KV à 4,5 b (recette KIVI, groupe 64) : 3,02 Go → **47,2 Go, marge 0,8 Go**. La thèse 70B se joue là, pas sur un demi-bit de payload. Falsifieur qualité, 1 j-h, ~5 $, zéro CUDA : aller-retour de quantification 4 bits groupe 64 sur k et v côté hôte dans `bin/ppl`, wikitext à 4096. Si la ppl bouge de plus de 1 %, la recette est mauvaise avant qu'une ligne de noyau soit écrite.

---

## 2. Axe vitesse — décodage

### S1 — Mémoïser la rotation par `RotKey` : 108 des 252 rotations sont des recalculs à l'identique ★★ (meilleur rapport gain/risque du dossier)
- **Mécanisme** : q/k/v partagent une activation donc une rotation, gate/up aussi — le chargeur le dit lui-même (« 252 builds where 144 are owed »). Mais `FusedOp::cuda_fwd` lance `rot_apply` par **projection**, et le pool de scratch est indexé par `d_in`, pas par `RotKey` : q, k et v réécrivent successivement le même vecteur tourné à 2560.
- **Chiffre** : 108 × 8,05 µs = **0,869 ms/jeton = 7,6 %** → 87,7 → **~94,9 tok/s**, à résultat **bit-identique**.
- **Blocage** : une clé de HashMap. Rien d'autre.
- **Coût** : ~5 lignes, 1 h, ~1 $.
- **Falsifieur** : c'est lui-même. Si le débit ne bouge pas de plus de 2 %, **tout le modèle « les lancements dominent » tombe**, et avec lui S2/S4. C'est le test le moins cher du dossier et il teste un modèle, pas une implémentation.

### S2 — Fusion q+k+v et gate+up : le noyau est écrit, prouvé, et non branché
**0,803 ms mesurés (13,8 %)** à octets rigoureusement constants (−0,00 %), 108 lancements supprimés. Cause : k_proj et v_proj lancent 128 CTAs contre une capacité résidente de 852 — **15 % de la carte, 157 Go/s contre 469**, pour un rapport de 1,06× contre FP16 sur 13 % du temps. `tv_slot_seg` existe et est bit-exact sur 921 600 lignes, **mais uniquement en Slot32 et pas dans `bin/fusedrun`**. Recoupement externe : le README de QTIP annonce 80–90 % des vitesses publiées sans fusion de matrices — même ordre. Coût : 2–3 j-h pour le jumeau Planes14 + branchement, ~3 $. Compose avec S1 (S1 supprime 108 lancements de rotation, S2 les 108 matvecs correspondants).

### S3 — La rotation d'incohérence : 2,39 ms/jeton (21 %), le seul noyau jamais optimisé
`rot_apply` tourne sur **un seul CTA** (1 SM sur 142) parce que la WHT est log₂(m) barrières et que CUDA n'a pas de barrière inter-blocs : **2,0 Go/s sur une carte qui en sert 864**. Trois leviers cumulables et indépendants : (a) dédup = S1 ; (b) **templater par k** — `rot_mix` fait 32 MAC par ligne quel que soit k, soit **6,4× de gaspillage à k=5, la forme 2560, la plus fréquente** ; (c) sortir du mono-bloc en deux lancements. ⚠️ Le chiffre publié de 1,517 ms suppose 4 rotations par couche ; le code en lance 7 — **le poste est sous-étiqueté de 57 % dans le dépôt**, et la conclusion « la rotation reprend un tiers du gain du noyau fusé » est en réalité 46 %. ⚠️ **Blocage produit dur** : `LLVQ_ROT_KMAX = 32` refuse `down_proj` de Qwen3-32B (25600 = 512·50, k=50) — **le portage 32B est bloqué côté rotation, pas côté matvec.**

### S4 — Réécrire le décodeur : 6 noyaux par couche + anneau KV préalloué + flash-decode GQA ★
- **Mécanisme** : sur un stream unique un lancement **est** une barrière de dispositif (3,63 µs mesurés). Une couche en paie 41 pour ~0,3 ms d'arithmétique. Deux gaspillages structurels : `KvCache::append` fait `Tensor::cat().contiguous()` (réalloue tout l'historique, 2C), et `repeat_kv` matérialise une copie 4× (`cat` de 4 vues, 8C) — **432 lancements/jeton** dont chacun déplace ≤ 141 Ko à contexte court, donc 100 % de latence.
- **Chiffre** : suppression des 432 copy2d = **1,57 ms → +16 %** au point publié. Repli des ~970 noyaux candle en 216 = **+7 à +18 %, central +12 %** (98 tok/s). À contexte 4096, l'anneau + flash-decode donne **×1,26 [1,26–1,60]** (74,1 → 93,0 tok/s). ⚠️ Le facteur 14C est un majorant brut : 8C des 14 sont de la relecture-après-écriture *dans* une couche, et l'ensemble vivant à 4096 (83,9 Mo) tient **sous les 100,7 Mo de L2** de la L40S — le trafic DRAM réel est ~4C.
- **Blocage** : le verdict A3 a fermé le dossier « coût de lancement » quand le CUDA Graph a échoué — mais le même journal écrit que le graph supprime le trafic pilote, **pas** la mise en route et l'arrêt des blocs sur les SM. Le graph a été réfuté ; le lancement, non. Et `model.rs:200` justifie par écrit le refus de l'anneau (« every length in this repository is a property of the corpus ») — juste pour un harnais de perplexité, faux pour un moteur.
- **Coût** : 6–10 j-h, mutualisés (le noyau flash-decode **est** le lecteur de l'anneau et remplace `repeat_kv`, le masque et le softmax d'un coup). ~20 $.
- **Falsifieur, 0,10 $** : lancer `fusedrun` avec `n_new = 2048` et imprimer le tok/s **par quart**. Modèle 14C : effondrement 88 → 56. Modèle 4C corrigé : courbe quasi plate. La mesure n'a jamais été faite et elle dit surtout si le produit publié survit à un contexte réel.
- **Plafond hôte à connaître** : ~1 480 lancements × 1,85 µs de soumission = **2,74 ms de travail CPU par jeton**, masqués sous 11,40 ms mais 30 % d'un jeton à 9 ms. Le goulot suivant est déjà identifiable, et il est hôte.

### S5 — Micro-poste gratuit
Le masque causal est **intégralement nul** au décode (à l = 1, tous les j passent) et il est pourtant reconstruit à chaque jeton : un Vec hôte, une copie H2D, une conversion f32→f16, puis 36 `broadcast_add` de zéros. ~0,14 ms (1,2 %), prouvablement neutre.

---

## 3. Axe vitesse — préfill et service (changement de classe d'usage)

### S6 — Préfill par déquantification vers un scratch f16 + GEMM cuBLAS ★
- **Mécanisme** : `FusedRuntime::forward` boucle `for r in 0..rows` — un prompt de *l* jetons relit les 2,18 Go *l* fois — avec un plafond dur `MAX_ROWS = 256`. Déquantifier **une matrice à la fois** dans un scratch f16 rend l'intensité arithmétique à sa valeur de GEMM et rend l'arithmétique aux tensor cores.
- **Chiffre** : préfill de 256 jetons, ~2,67 s aujourd'hui → **~40–55 ms**, soit **×50–65**. Scratch de pointe = la plus grosse matrice (9728×2560×2 = 49,8 Mo), donc +50 Mo, pas +7,27 Go.
- **Blocage** : `fused_cuda.rs:377` — « le noyau fusé est un matvec, donc le coût est linéaire », traité comme une propriété du **format** alors que c'est une propriété du **régime**, puis institutionnalisé par `MAX_ROWS`.
- **Coût réel** : 5–7 j-h, pas 3–4 — il faut **aussi** un `rot_apply` batché (sinon 252 × 256 = 64 512 lancements de rotation = 234 ms, six fois le budget total) et un lm_head batché si l'on veut la ppl.
- ⚠️ **Ce que ça ne fait PAS** : combler le trou de validation. `bin/mmlu` et `window_nll` appellent `logits()` — une seule passe avant : MMLU et ppl sont **100 % préfill**. Basculer le préfill sur GEMM garantit que `tv_planes` n'est *jamais* exécuté pendant ces campagnes, et ce qu'on scorerait est exactement ce que `sealed::load` score déjà. Le trou de validation (aucune métrique de qualité n'a jamais été mesurée sur le chemin fusé, et les deux bras divergent au jeton 89) doit être comblé **en décodage**, sur un sous-ensemble.
- **Falsifieur, 0,10 $** : imprimer le temps de préfill des deux bras dans `fusedrun`. Le bras dense fait déjà le préfill par GEMM sur les mêmes 256 jetons.

### S7 — Le noyau N-colonnes (service batché, et seulement lui)
Dans `planes_dot`, **240 des 270 instructions et 100 % des 14 octets** ne dépendent que du bloc ; seuls `xb[j]` et les 24 FMA dépendent de la colonne. Instructions par FMA utile : 11,25 à N=1 → 6,25 à N=2 → 3,75 à N=4 → 2,50 à N=8. **Ampleur défendable : ×3,2–4,0 à N=4, ×5,1–8,0 à N=8** (et non ×4/×8/×16 : le coût marginal par colonne n'est pas nul — 0,316 ms de partagée + 0,091 ms de FMA, et l'occupancy tombe de 48 à ≤16 warps/SM à N≥4). ⚠️ **La revendication « le rapport contre le dense reste 2,15× à N=16 » est fausse** : le modèle donne T(16) = 11,23 ms contre 11,014 au dense plat → 0,98×. C'est exactement la courbe GLQ (E8+treillis fusé sur RTX PRO 6000 : 0,98× à B=1, **0,50× à B=32**), décalée d'un cran. **Pour le préfill, S6 domine S7 d'un facteur ~7.** N-colonnes ne se justifie que pour le service concurrent et comme préalable au décodage spéculatif (la littérature mesure une **anti-synergie** spéculatif × quantification tant que le décodage n'est pas réutilisé sur les K jetons vérifiés). Falsifieur : `#define NCOL 4` dans `planesbench`, 30 lignes, ~1 $ — vérifier que **les octets lus sont identiques** à N=1 et N=4.

### S8 — Chargement : 124 s aujourd'hui, 39 min projetées à 70B
`transcode_planes14` est une bijection **sans état inter-bloc** (prouvé : `transcode_planes12x` la découpe déjà en 8 fils) et reste une boucle `for` séquentielle. Repli à **0,5 j-h** : recopier le `thread::scope` → 124 → ~70 s. Version carte : lire 0,904 Go d'index, écrire 2,110 Go = 4,57 ms. ⚠️ Les ÷40 et les 39 min sont surestimés d'un facteur ~10 (le seul temps attribué est 37 s sur 124, soit ÷1,4 par Amdahl ; et ×18,84 est le rapport des poids *linéaires*, pas du chargement entier). **Falsifieur, 10 min, 0,10 $** : un `Instant` autour de `transcode_stream`, somme sur les 252 matrices. Sous 30 s des 124, le portage GPU ne rachète rien et la bonne réponse est le `thread::scope`.

---

## 4. Axe stockage froid

Fichier de 1 770 527 533 o : code de réseau 51,06 % · embedding f16 43,94 % · queue f32 3,83 % · tokenizer 0,65 % · échelles f64 0,50 % · tout le reste 0,025 %.

**L'axe est presque clos côté code** : le fichier est à **1,2 % du plancher d'information** (2,000 b/poids de code exact contre 1,9765), et la marge entropique de l'index est **0,0045 bit** — pas 0,346. Le codage entropique est mort quantitativement, même avec un codeur parfait.

Deux gestes, tous deux à qualité déjà validée :

| geste | octets | % du fichier | coût |
|---|---|---|---|
| **C1** — sceller l'embedding en q8 **dans le fichier** (`RawData::Quant{bits:8, group:64}` passe déjà) | −364 646 400 | **−20,60 %** | 1 j-h |
| **C2** — queue `KeepExact` en f16 (= V3) | −33 914 880 | −1,92 % | 0,5 j-h |
| **total** | **1,772 → 1,372 Go** | **−22,5 %** | |

C1 supprime en prime la quantification de 389 M valeurs refaite à chaque chargement. Les échelles de ligne en f64 (8,8 Mo) et les centroïdes f64 sont du bruit — ne pas y toucher.

⚠️ **Arbitrage à connaître** : élargir le gain à 3 bits (piste P1) coûte **+0,083 b/poids sur le disque** (2,1696 → 2,2529, +37,7 Mo) et casse la comparaison iso-2,000-bits aux Tables 6/8 du papier.

---

## 5. Axe MMLU — le point faible du produit, et son cadrage

**Recadrage nécessaire avant toute dépense.** Face au 4 bits nous perdons : 55,6 contre **70,0 pour l'AWQ officiel** (−0,28 pp seulement sous FP16). Face au **2 bits publié** nous gagnons : nous retenons 79,2 % de la baseline, contre **72,6 % pour HPTQ sur un Qwen3-8B deux fois plus gros** (52,99 à 2,125 b contre 73,02 en BF16) et **46,9 % pour du GPTQ 2 bits**. AQLM retient 89,4 % sur Llama-3-8B — et son avantage n'est pas le codebook (le nôtre est meilleur, Leech > 1×16) mais **PV-tuning**, un apprentissage post-quantification. Même conclusion que le papier LLVQ, dont la meilleure ligne 4B passe de 17,05 à **9,26** de ppl par un simple apprentissage d'échelles. **« La qualité est le point faible » est vrai contre le 4 bits et faux contre l'état de l'art à 2 bits — deux énoncés à ne plus mélanger.**

### M1 — Le census MMLU + McNemar : le gate de tout l'axe ★
- **Chiffre** : l'estimateur publié est un micro **stratifié** avec correction de population finie (effet de plan 1,79, n_eff = 1277) — sa SE est 1,28/1,36 pp, exactement ce que le log imprime. **Seuil de détection 2σ aujourd'hui : 3,93 pp.** Census 14 042 questions + test apparié (McNemar sur le dump que `LLVQ_MMLU_DUMP` produit déjà, l'index parquet étant zippé avant le mélange précisément pour ça) : **0,65–0,84 pp**, soit un facteur 4,7–6,0 pour **21 min et ~1,3 $ pour deux bras**. La ré-analyse appariée du dump existant est **gratuite** (0 min de carte) et rend déjà 2,17–2,50 pp.
- **Blocage** : le dépôt a tiré du fiasco macro/micro une leçon de *protocole*, jamais de *puissance*. Le ± imprimé sert à défendre les −14,33 pp, énormes devant 1,35 ; personne n'a retourné le calcul pour demander ce qu'il permet de **détecter**.
- ⚠️ **C'est un TODO déjà écrit et déjà budgété** (`docs/experience-mesure.md`, `errata-rapport-lot-a` MINEUR item 3) — jamais exécuté. Exécution, pas découverte.
- **Corollaire à encaisser** : le profil « le raisonnement s'effondre, la restitution tient » n'est **pas établi**. Le contraste STEM/reste vaut 4,02 pp à p ≈ 0,056 sur un découpage choisi a posteriori ; « l'algèbre abstraite tombe à 25 % » est un tirage à 1,9 σ parmi 57 (elementary/high-school mathematics chutent le **moins**, 5,0 et 7,5 pp). **Ne pas construire un correctif ciblé « raisonnement » (P18) avant le census.**

### M2 — Apprentissage post-quantification : le seul levier à deux chiffres ★★
Trois variantes, poids quantifiés **gelés**, donc **noyau et artefact intacts** :
- **Recover-LoRA sur exactement notre cible** — Qwen3-4B à 2 bits, **80–95 % de récupération sur 9 benchmarks sur 12**, 10 k échantillons **synthétiques**, aucune donnée étiquetée (distillation de logits, adaptateurs sur couches sélectionnées). C'est la référence la plus directe qui existe.
- **EoRA** (NVlabs, ICLRW'26) — compensation bas-rang fermée, sans gradient, 1 024 échantillons C4. Llama3-8B à 3 bits : **+10,84 pp ARC-C, +11,45 pp GSM8K, +6,74 pp MathQA**. Rang 32 f16 sur 252 matrices = **132 Mo, +0,26 b/param**. ⚠️ Le corollaire « B partagé entre matrices » est réfuté (le résidu GPTQ est blanchi **dans la métrique H**, pas dans la base d'origine ; et `ΔᵀΔ ≈ cI` est **impossible par le rang** sur k/v/o/down, où d_out < d_in). Il faut la vraie SVD par matrice, au vrai prix.
- **Échelles par colonne apprises** (le FT du papier) : ~760 k paramètres, ~52 M tokens, +2,1 pp MMLU et 17,05 → 9,26 de ppl. ⚠️ La version **forme close** est réfutée : elle optimise le proxy local que GPTQ minimise déjà, pour 1,5–2,0 % de proxy, sous le bruit — et un scalaire par ligne **n'est pas** une jauge absorbée par le RMSNorm suivant.
- **Coût** : 3–5 j-h + quelques heures de GPU par variante. **Blocage** : classé P11/P16/P17 depuis longtemps, jamais lancé, parce que tous les A/B du projet se décident sur 3 blocs de ppl — et la ppl ne borne pas MMLU.
- **Falsifieur** : Recover-LoRA sur 3 couches, 10 k échantillons synthétiques, delta MMLU **apparié** sur le census. Si < 2 pp, la famille tombe.

### M3 — La graine de rotation n'a jamais été balayée
La rotation d'incohérence est un **tirage aléatoire** (`0x11_0FEED`, tirage unique) et le dépôt la présente comme « le plus gros levier mesuré » (×2,290 → ×1,811). SpinQuant mesure, sur Llama-2-7B W4A4 et 100 tirages, **13 points d'écart zéro-shot entre la meilleure et la pire rotation**, variance supprimée par optimisation sur la variété de Stiefel. Sa variance n'est pas celle de `LLVQ_CALIB_SEED` (qui ne bouge que les offsets de fenêtres) : c'est une variable indépendante, non mesurée. **Coût du test : 3 graines × 3 blocs, ~1 $.** C'est aussi **la barre d'erreur qui manque au projet depuis le lot B**. Mécanisme neuf, sans rapport avec la rotation de *sortie* (morte par la Table 9).

---

## 6. Axe perplexité (16,9415, ×1,384 ; QTIP 17,04 — marge 0,58 %)

### P1 — Élargir le gain de 1 à 3 bits, dans le padding
`Planes14` a **6 bits de bourrage libres** (`PLANES14_PAD_BIT = 106`) et `gain_bits` est câblé en dur à 1 par assert. Passer à 3 : **VRAM et vitesse strictement inchangées**, +0,083 b/poids sur disque. Chiffre recalculé sur le banc du dépôt (modèle Lloyd–Max sur χ₂₄, validé à 5 % près contre le pas 0→1 bit mesuré) : **−9,35 % de MSE de bloc**, soit −0,31 pt de rétention à débit constant. Sur modèle, le seul ancrage est 15,3272 (magnitude libre 16 b) contre 16,9617 (1 bit) : **borne supérieure −9,6 % de ppl**, intervalle honnête [−9 % ; 0]. ⚠️ **Sur MMLU le signal du papier est inverse** (0 bit : 60,7 ; 2 bits : 59,3) : prédiction [−1,4 pp ; 0]. ⚠️ **Ne pas coupler à la règle de projection** : les deux termes sont additifs et le « plancher » 2/(1+cos θ) ne vaut que 1,46 % ; et la règle projection rétrécit *tous* les blocs de 3,3 %, ce qui relâche exactement la rigidité de norme que `group_scales` et le design C ont déjà punie à profondeur (×1,99). Coût : 1–2 j-h, ~1 $ (déplacement d'offsets, `gain_bits` déjà paramétré dans la famille Slot32).

### P2 — Graine de rotation = M3. Le même run rend ppl et MMLU.

### P3 — Amortissement hessien à largeur réelle
`damping = 1e-2` est la seule valeur jamais passée, sur toutes les largeurs, jamais balayée. La simulation « c'est nul » tourne à **n = 240**, pas 2560/9728 : à n = 2400 l'optimum se déplace à 3e-3, 1e-2 est 0,9 % au-dessus, et la plage 3e-3…3e-1 s'étale sur **16,4 %**, pas 3 %. Effet attendu ~1 %, donc secondaire — mais le balayage annoncé comme inutile ne l'est pas tout à fait. Coût : 3 valeurs × 3 blocs, ~1 $.

### Régressions à budgéter explicitement
V1 (top-128) : +0,04 % de MSE. V2 (L≤4 natif) : +0,34 % de MSE → ppl 16,96–16,99. **Ensemble ils consomment la totalité de la marge sur QTIP**, qui est de toute façon plus étroite que la dispersion du projet. Décision à prendre : garder le titre G5, ou prendre les bits.

---

## 7. Les cinq pistes de classe « Planes14 »

Critère : ≥ 10 % sur un axe, ou gain sur deux axes à la fois.

| # | Piste | Axes | Ampleur | Coût | Pourquoi elle sort du lot |
|---|---|---|---|---|---|
| **1** | **Record 10 octets** (classe 7 bits, puis L≤4 **natif**) | VRAM + vitesse | 4,804 → 3,67 puis **3,48 b/poids** (−28 %) ; 5,11 → **3,96 b/param** ; +9 à +15 % tok/s | 3–4 j-h, ~6 $ | Le seul geste de format qui gagne sur **deux axes** ; falsifiable à **0 $** sur CPU avant d'écrire une ligne ; passe le modèle **sous les 5,30 b/param de l'AWQ réel** |
| **2** | **Apprentissage post-quantification** (Recover-LoRA / EoRA) | MMLU + ppl | **+7 à +11 pp** publiés à bit-width comparable ; poids gelés | 3–5 j-h, ~30 $ | Seul levier à deux chiffres sur l'axe où on perd ; **noyau et artefact intacts** ; c'est aussi ce qui explique les 89,4 % d'AQLM et le 9,26 du papier |
| **3** | **Décodeur en 6 noyaux + anneau KV + flash-decode** | vitesse + VRAM(contexte) | +12 à +30 % au point publié ; **×1,26–1,60 à ctx 4096** ; ferme le trou de +0,60 Go à 4k et +6,04 Go à 40k | 6–10 j-h, ~20 $ | Le poste dominant du jeton n'est **ni le décodage ni la mémoire, c'est le lancement** (~1 480/jeton) ; et c'est ce qui décide si le produit survit à un contexte réel |
| **4** | **Préfill déquant→scratch f16 + GEMM** | vitesse (classe d'usage) | préfill **×50–65** ; lève `MAX_ROWS = 256` | 5–7 j-h, ~30 $ | Change ce que le produit **peut faire** (documents, service), pas un pourcentage ; sans lui le benchmark métier d'extraction documentaire prévu au plan est infaisable |
| **5** | **Census MMLU + McNemar** | instrument | seuil **3,93 → 0,65–0,84 pp** | 0,5 j-h, **~1,3 $** | Ce n'est pas un gain, c'est la condition de mesurabilité de #2, P1, M3 : aujourd'hui aucun levier sous 3,93 pp n'est détectable |

Mentions gratuites hors classement, à faire dans la semaine : **dédup rotation** (+8 %, 5 lignes, bit-identique), **embedding q8 scellé** (−20,6 % du fichier), **queue f16** (−1,9 % de fichier et −33,9 Mo de VRAM).

---

## 8. Couplages

**Se composent :**
- S1 (dédup rotation) ⊗ S2 (fusion q+k+v) : la première supprime 108 lancements de rotation, la seconde les 108 matvecs correspondants. ~1,67 ms cumulés.
- S2 ⊗ S4 : la fusion est un sous-cas du repli en 6 noyaux ; la faire d'abord, elle est déjà écrite.
- S4 ⊗ V6 : le noyau flash-decode **est** le lecteur de l'anneau. Un seul chantier, pas deux.
- V1 ⊗ V2 : le plafond natif supprime l'overlay d'exceptions (0,202 b/poids **et 0,55 Go de trafic fantôme**). C'est V2 qui rend le record 10 o **rapide** ; sans lui V1 est neutre en vitesse. **V2 seul, sans V1, ne vaut presque rien** (Planes12x rend déjà 4,342 b/poids gratuitement).
- C1 ⊗ C2 ⊗ V3 : indépendants, additifs, tous sans perte.
- S7 (N-colonnes) ⊗ décodage spéculatif : le spéculatif ne redevient intéressant qu'**après** S7 (anti-synergie mesurée sinon).

**S'excluent :**
- **Les 2 bits libres du champ classe : VRAM ou résolution radiale, pas les deux.** Dans un record serré, classe 7 bits + gain 3 bits = 82 bits → 11 octets, et l'on reperd 0,33 b/poids. Le gain à 3 bits n'est *gratuit* que dans Planes14 à 14 octets. **Arbitrage à trancher en une décision, pas deux.**
- S6 (préfill GEMM) vs S7 (N-colonnes batché) pour le préfill : S6 domine d'un facteur ~7. Le dépôt avait raison : « deux noyaux, pas un ».
- V4 (Golay9) vs tout le reste de l'axe vitesse : −34,5 % d'octets pour −13 à −35 % de débit. Repère 70B, pas point produit.
- V5 (signes impairs) vs le stride uniforme, qui est ce qui a fait gagner Planes14.

**Budget croisé à surveiller** : EoRA r=32 coûte **+0,26 b/param**, c'est-à-dire un quart de ce que le record 10 o rend. 5,11 → 4,10 (V1) → 4,36 (V1+EoRA) : reste sous 5,30. Ne pas dépenser deux fois la même marge.

---

## 9. Ordre de bataille

**G0 — aujourd'hui, 0 $, CPU seul.** Trois mesures qui décident sans dépenser :
1. `classhist` : masse cumulée des 128 classes les plus peuplées du fichier scellé. **Décide V1** (seuil 98 %).
2. `Instant` autour de `transcode_stream`, somme sur 252 matrices. **Décide S8** (seuil 30 s sur 124).
3. Ré-analyse appariée du dump MMLU existant : imprimer b, c, b+c. **Calibre M1** et rend déjà 2,17–2,50 pp de seuil.
4. 20 lignes dans `calib.rs` : moyenne de AM/GM sur `M_p = inv(U_ppᵀU_pp)` pour 200 blocs par matrice. **Scelle le verrou blanchiment** par la mesure et pas par la simulation (si une matrice dépasse 1,15, le verrou saute).

**G1 — ~2 $, une demi-journée.** Dédup rotation par `RotKey`. Prédiction 87,7 → 94,9 tok/s. **C'est le gate du modèle « les lancements dominent »** : si le débit ne bouge pas de plus de 2 %, S2 et S4 meurent ensemble et l'effort bascule sur le format.
**G1bis — ~1 $.** Attribution in-situ par événements CUDA dans `fusedrun`. Lève l'indétermination 2,3 vs 5,3 ms. **Aucun chantier vitesse au-delà de G1 avant ce chiffre.**

**G2 — ~1,3 $, 0,5 j-h.** Census MMLU 14 042 + McNemar sur les deux bras existants. Verrouille le seuil à 0,65–0,84 pp. **Aucune décision qualité avant ce gate.**

**G3 — quand le 8B livre.** Deux précautions :
- Il tourne en `leech1c12L3` : **confondu deux fois** (plafond L3 **et** `intermediate_size = 12288 = 24·512` exactement, donc `down_proj` — 26 % des poids — n'a aucune queue). L'A/B `leech1c12` vs `leech1c12L3` sur 3 blocs (~1 $, ~15 min) est le seul moyen de le déconfondre, et il est réclamé par l'audit du 03-08 depuis quatre jours.
- **Ce qu'il faut regarder est MMLU, pas ppl.** En surcoût de nats face au papier, notre déficit **double** avec l'échelle : 4B 0,3268 contre 0,3176 = 1,03× ; 8B 0,2370 contre 0,1253 = **1,89×**. La lecture « le 8B se dégrade moins, donc signal d'échelle » est dans la liste « phrases à ne jamais écrire » de l'audit.
- Sceller le 8B exige `bin/seal` (l'artefact est LVQ1, projections seules) : checkpoint retéléchargé, ~29 Go de RAM hôte. Ce n'est ni gratuit ni un chemin déjà emprunté — le budgéter.

**G4 — ~6 $, 3–4 j-h. Format.** V1 (classe 7 bits, transcodeur Planes12x réutilisé) puis A/B natif L≤4 (~1 $) puis V2 si vert. Garde-fou obligatoire avant d'écrire le noyau : rejouer `planesbench` avec un flux sans exceptions et vérifier que le débit remonte bien vers 425–455 Go/s.

**G5 — 6–10 j-h, ~20 $. Moteur.** S2 (fusion, noyau existant à porter en Planes14), puis S4 (6 noyaux + anneau + flash-decode), puis S3(b) (templater la rotation par k). Sortie attendue : **~120–135 tok/s en décode** (les termes se recouvrent, à recaler sur G1bis), et le mur suivant devient la soumission CPU (~2,74 ms/jeton).

**G6 — 5–7 j-h, ~30 $. Préfill.** S6, avec son `rot_apply` batché. Livre le prompt long, lève `MAX_ROWS`. **Séparément**, combler le trou de validation en **décodage** : ppl/MMLU sur un sous-ensemble scoré par le chemin fusé.

**G7 — qualité.** M3 (3 graines de rotation × 3 blocs, ~1 $ — la barre d'erreur manquante) et P1 (gain 3 bits) en parallèle, décidés sur le census. Puis M2 (Recover-LoRA d'abord : synthétique, poids gelés, la référence est sur exactement notre modèle et notre bit-width).

**Hors séquence, deux verrous produit à lever quand le 32B revient au menu** : `LLVQ_ROT_KMAX = 32` refuse `down_proj` du 32B (k=50), et le coût par poids n'est **pas linéaire** (4,77e-5 cœur-s à 8B, 6,36e-5 à 32B — la factorisation passe de 5,5 % à 16,5 %).

---

## 10. Ne pas recreuser

Déjà mort **avant** ce tour : rotation de sortie ; codage entropique du froid ; run de calibration ×100 ; design C / rétraction libre (×1,99) ; `group_scales` tel qu'implémenté ; plafond L≤4 **par swap post-hoc** ; CUDA Graph ; lm_head en LLVQ ; transcodage à la volée ; mixte Slot32/Grouped32 par matrice ; rangs groupés ; décodage coopératif par ballot ; padding à 28 flottants.

Mort **par ce tour**, avec sa raison :

- **Blanchiment bloc-diagonal 24×24 / re-scoring `shell_bests` sous M_Q.** La métrique conditionnelle `M_Q = (U_QQ U_QQᵀ)⁻¹` vaut déjà I à 1,01–1,04 près **parce que la rotation d'incohérence blanchit H** (base brute : anisotropie 52 ; base tournée : 1,004). Économise ~6 j-h, 33 Mo de VRAM et 24 FMA/canal/jeton. *Là où ça vaut quelque chose (couches sans queue, 24 | d_in — `down_proj` du 8B), la réponse n'est pas un noyau mais 0,031 b/poids de `KeepExact` sur le dernier bloc.* Apport permanent : **une deuxième justification de la rotation, jamais écrite dans `rotation.rs`.**
- **Décodage spéculatif sur le 4B.** Le brouillon 0.6B paie 78 % du surcoût de lancement de la cible ; le coût de brouillon (K passes **séquentielles** par construction) valait zéro dans le calcul, soit 77 % du pas oublié. Rouvrir seulement après S7, et à 70B.
- **Golay12** (3ᵉ plan pour supprimer les exceptions de Golay70). Le format E2 marchait déjà ; ce qui l'a tué est le **décodage à double coset borné calcul**, que Golay12 conserve intégralement.
- **Ternary8.** Son codebook **est** la ligne « L max = 3 » de `bin/lcap`, mesurée : MSE 0,0870 contre 0,0725, **+20,0 %**, −2,78 pt de rétention. Et son transport est Golay70 moins un octet.
- **Noyau persistant / recalcul de la rotation dans chaque CTA du matvec.** La chaîne de dépendance d'un transformeur interdit le recouvrement ; `down_proj` (320 CTAs, 453 Go/s) réfute déjà la prémisse « le sous-remplissage explique o/down ». Et répliquer la rotation fait faire à 142 SM 8,56 fois le travail qu'un SM faisait une fois.
- **Dépaquetage des créneaux de lane (13,40 % de créneaux vides).** Le compte est exact, le prix ne l'est pas : `k_proj` et `gate_proj` ont le **même** gaspillage et rendent 157 contre 469 Go/s ; `down_proj` gaspille le moins et va moins vite que `gate/up`. Et 76 % du gain revendiqué est ce que la fusion q+k+v encaisse déjà, **à gaspillage constant**.
- **Correction de biais (premier moment).** µ est **dans** H, et la queue `KeepExact` (67,8 Mo de paramètres continus libres, 30× le biais proposé) est déjà un minimiseur des moindres carrés dans la métrique H : résidu mesuré en simulation ~1e-4, contre 0,08–0,99 annoncés.
- **EoRA avec B partagé entre matrices.** Le résidu GPTQ est blanchi dans la métrique H, pas dans la base d'origine ; et `ΔᵀΔ ≈ cI` est impossible par le rang sur 40 % des poids. EoRA reste vivante, mais au vrai prix (SVD par matrice, +0,26 b/param).
- **Refit d'échelles en forme close** (par ligne ou par colonne). Optimise le proxy local que GPTQ minimise déjà : 1,5–2,0 %, sous le bruit ; et une échelle par ligne est une échelle par canal de sortie, **que le RMSNorm suivant n'absorbe pas**.
- **Allocation de débit par matrice (water-filling).** Le 2ᵉ bit de gain vaut *moins* qu'un bit d'index — la Table 8 le mesure à débit constant (+2,56 % par bit échangé) — et le modèle utilisé avait le **signe faux** sur cet échange.
- **Le cadrage « 5,2 des 14,7 points ne sont pas de la quantité d'erreur ».** Vrai comme mécanisme, mais déjà publié au README, et le corollaire quantitatif (« facteur 15 de discontinuité ») repose sur une chute AWQ à 0,16 σ dont l'IC recouvre la bande adverse.