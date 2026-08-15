# Note produit — le barreau mémoire, les métriques et l'environnement d'expérimentation

> **STATUT : BROUILLON, soumis à relecture.** Rien ici n'est opposable tant que
> les cases du §A ne sont pas arbitrées par l'opérateur et le fichier commité.
> Une fois arbitrée, cette note devient la référence que les
> pré-enregistrements de `proofs/` citent — c'est elle qui donne un statut aux
> seuils (le `S_alt` du pré-enregistrement du 2026-08-13 n'en a aucun sans
> elle).

**Pourquoi cette note existe.** Le critère qui a enterré E3 (≤ 2,60 b/poids
noyau) n'est pas une constante physique : il se dérive de
`((VRAM − marge − KV)×8 − embedding)/68,45 Md`, et chacun de ces termes est un
**choix produit** jamais acté — 5 Go de marge, KV en f16, contexte 8k. Avec
d'autres choix aussi défendables, le même barreau 32 Go donne un seuil de
3,09. Aucun des deux n'est « le vrai » : le vrai est celui que cette page
fige.

---

## A. Les décisions à arbitrer (cases à cocher — opérateur)

**A1. Carte cible du barreau principal** *(une seule ; les autres deviennent
des segments secondaires)* :

- [ ] **RTX 5090 32 Go, PCIe gen5** *(recommandé : la carte grand public
  haut de gamme 2026, bande passante ~1,8 To/s, celle que l'étude MoE nomme)*
- [ ] 24 Go (3090/4090 d'occasion — le segment le plus « souveraineté », mais
  zéro marge opérationnelle sur tout ce qu'on sait faire)
- [ ] Mac à mémoire unifiée 64-128 Go (exige le portage Metal de la rotation,
  semaines)

**A2. Contexte servi** *(impact direct sur le seuil, table §B)* :

- [ ] 8k  - [ ] 16k  - [ ] 32k

**A3. Format du cache KV** :

- [ ] f16 (existant)
- [ ] **q8 (à construire — chantier P3, levier transversal : ~1,35 Go au lieu
  de 2,7 à 8k sur un 70B)**

**A4. Marge opérationnelle réservée** (contexte CUDA, activations, jitter) :

- [ ] 2 Go  - [ ] 5 Go

**A5. Plancher de vitesse, par segment** *(sans lui, aucun banc n'a de
critère d'admission)* :

- Extraction par lots (package B) : débit cible ______ documents/heure, ou
  secondes/document max ______ *(proposition : ≤ 60 s/doc en lot de 8)*
- Chat/agent local (packages A et C) : ______ tok/s minimum
  *(proposition : ≥ 20 — la lecture humaine confortable)*

**A6. L'offload PCIe est-il admis comme solution de référence** (ce qu'on
doit battre) **ou comme solution servie** (ce qu'on livre) ?

- [ ] référence seulement  - [ ] servi aussi (package A en dépend)

## B. Les seuils qui découlent du triplet (barreau 32 Go, 70B dense)

`b_max = ((32 − marge − KV)×8 − 17,85)/68,45` — embedding q8 facturé
**8,5 b/param** (la comptabilité de la préreg du 08-11), convention 70B de
`fiche-4b` (68,45 Md quantifiés + 2,10 Md embedding). Tous **calculés**.

| contexte | KV q8, marge 2 Go | KV q8, marge 5 Go | KV f16, marge 5 Go |
|---|---|---|---|
| 8k | **3,09** | 2,74 | 2,58 ≈ le 2,60 historique |
| 16k | 2,93 | 2,58 | 2,27 |
| 32k | 2,61 | 2,26 | 1,63 (infaisable) |

Lecture : le 2,60 historique se relit soit comme « 8k, KV f16, 5 Go », soit
comme « 32k, KV q8, 2 Go ». **Un seul triplet fera foi** — coché en §A, avant
toute lecture de résultat.

## C. Les métriques canoniques (définitions, et le piège payé derrière chacune)

| métrique | définition exacte | le piège déjà payé |
|---|---|---|
| **b/poids noyau** | bits/bloc × 0,0414723 + 0,1590915 (4B ; épinglé par le test 112 → 4,804) | ne se compare jamais à un b/param modèle entier |
| **b/param modèle entier** | octets totaux ×8 / paramètres totaux, **embedding compris** (q8 = 8,5) | le « 5,51 contre 4,50 » à deux dénominateurs — faute grave de l'errata du lot A |
| **VRAM** | mesurée sur carte (`nvidia-smi`), jamais projetée sans l'étiquette *estimé* | |
| **tok/s** | médiane du **rapport formé round par round**, 7 rounds dont 2 jetés, tous les bras dans le même processus | jamais un quotient de deux minima ; les ms dérivent (2,03–2,08 sur binaire identique), les octets non |
| **ppl** | wikitext-2 test, ctx 4096, **empreinte de tokens identique imprimée des deux bras**, même dtype | f32 vs f16 mélangés ; empreintes non vérifiées |
| **MMLU** | **micro** (= papier), f16, 2 280 questions, graine figée ; σ des différences = McNemar (0,4-0,6 pp), pas le ± de ligne | le macro sur-pondérait les matières STEM ×2,5 |
| **provenance** | chaque nombre étiqueté **mesuré / calculé / estimé**, avec sa comptabilité | trois chiffres peuvent décrire le même objet sans se contredire ; les comparer entre eux, si |

Règles de mesure héritées, non négociables : pré-enregistrement dans
`proofs/` (+ `ots stamp`) **avant** toute mesure qui décide ; `oracle` à
chaque nouveau backend (42 s, ~1 centime) ; `--features fast-linalg` partout
où l'on paie ; un A/B qui se trompe de bras doit **échouer**, pas retomber en
silence sur un défaut.

## D. L'environnement technique

**Local (0 $) — Mac M3 Max, ~69 Go, Metal.** C'est là que tout se tranche
d'abord : exactitude par sweep intégral des artefacts scellés, bancs
`decode`/`thesis`/`matvec`, `radixstudy` (largeurs), `ppl`/`mmlu` sur Metal
(~7× le CPU). Boucle de test : `cargo test` (debug, rapide) ; la suite
`--include-ignored` se compte en **dizaines de minutes** et balaie le fichier
scellé.

**Carte louée — HF Jobs (`ops/run.py`), L40S / rtx-pro-6000.** Coûts type,
tous déjà payés une fois : banc 7 rounds ~0,2-0,5 $ ; quantification 8B
11,5 $ ; 32B ~62 $ ; **70B jamais fait, estimé 150-200 $** (extrapolation
avec la marge que la règle impose — le 32B avait été sous-estimé de 25 %).
`fusedrun` A/B tokens gloutons ~0,3 $.

**Artefacts scellés disponibles** : 4B (`~/llvq-q4b.llvq`, 981 Mo,
150 681 600 blocs) ; 8B (`~/q8b-c12.llvq`, 1,82 Go, 288 571 392 blocs).
Empreintes de tokens de référence : `3f1baca9033bf251` (ppl),
`65dcd53655e8bfa5` (MMLU).

**Qualité de référence mesurée** (la seule qu'on ait — les layouts ne
changent pas le contenu décodé, donc elle vaut pour tout format servant le
même fichier) :

| | 4B | 8B |
|---|---|---|
| MMLU (f16 → LLVQ) | 70,32 → 55,59 (−14,73 pp) | 76,08 → 65,52 (−10,56 pp) |
| ppl (dégradation) | ×1,385 | ×1,220 |

⚠️ Aucun chiffre de qualité n'existe au-delà du 8B, ni sur aucun MoE.

## E. Les trois segments servis (détail : les trois packages de la passation)

| segment | package | carte | le verrou d'entrée |
|---|---|---|---|
| Chat/agent local sur MoE ~120B | A | 32 Go (+ RAM hôte 64 Go) | qualité 2 bits sur MoE : **jamais mesurée** (gate X5, ~25-55 $) |
| Extraction documentaire par lots, 70B dense | B | 24-32 Go | vitesse du décodeur d'archive (banc V0/V1, 0 $) + chemin GEMM à écrire |
| 70B dense interactif | C | 32 Go | le décodeur E1v n'existe pas ; vitesse totalement inconnue |
