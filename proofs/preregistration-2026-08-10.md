# Pré-enregistrement — campagne de comparaison kernel

**Date : 2026-08-10.** Écrit **avant** toute mesure de cette campagne. Aucun
bras concurrent n'a encore tourné ; le dernier banc en date est
`docs/mesures/e2-golay70-bench-2026-08-07.txt`, à cinq bras.

> Ce document existe parce que `paper/sections/layouts.tex:93-95` se prévaut
> d'un critère « fixed before the measurement » — le 1,6× qui a écarté
> `Golay70`. Ce critère-là avait bien été posé d'avance, mais il vivait dans
> une conversation. Celui-ci vit dans un commit.
>
> Emplacement et format : `docs/plan-de-test-v2-cuda.md` §6.3.
> *(⚠️ Ce même document nomme le fichier `docs/attentes-<date>.md` en §6.5 et
> `proofs/preregistration-<date>.md` en §6.3. On retient §6.3, qui est la
> section « où déposer ». La divergence est signalée plutôt que tranchée en
> silence.)*
>
> ⚠️ **Ce que ce fichier n'est pas** : il n'est ni signé GPG ni horodaté par
> OpenTimestamps. §6.5 exige les deux pour que l'antériorité soit vérifiable
> sans faire confiance au dépôt. **Ce sont des actions de l'opérateur** (clé
> privée), pas de l'outillage. Tant qu'elles ne sont pas faites, l'antériorité
> de ce document repose sur la date de commit git, qui est ré-éditable.

---

## 1. Ce qu'on va mesurer, et pourquoi ça vaut la peine

Le papier compare son décodeur multi-coquilles au seul FP16. FP16 est un
témoin, pas un concurrent. Trois noyaux entrent : un mono-shell (la lignée du
papier d'origine), QTIP, et le 4 bits déployé (AWQ / Marlin).

**La question centrale de la campagne n'est pas une course de vitesse.** Elle
est posée par `paper/sections/decoder.tex:66-71`, qui **affirme sans mesurer**
que le rééchelonnage inter-coquilles — ce que l'Annexe G du papier d'origine
désigne comme le coût matériel du multi-shell — « se réduit à une
multiplication par bloc ». Le code le corrobore
(`llvq-llm/src/fused_cuda.rs:236-252` replie `1/√(16·shell)` dans la table de
classes au téléversement, donc zéro instruction à l'exécution), mais aucune
mesure ne l'établit.

La question devient donc, et c'est elle qu'on pré-enregistre :

> **Un décodeur à 301 classes atteint-il la même bande passante effective
> qu'un décodeur à 79 classes, à format et à octets identiques ?**

---

## 2. Prédiction

**On prédit que oui : `Shell12` (79 classes) et `Planes14` (301 classes) rendront
des Go/s indiscernables à la résolution du banc.**

Fondement : les deux lisent le **même nombre d'octets par bloc** (record de
14 o, cf. §3), le facteur de coquille est une constante de table, et le
précédent existe — `Slot32` → `Planes14` tient 428 → 425 Go/s alors que le
format change complètement, pendant que `Golay70`, qui ajoute un décodage à
double coset, s'écroule à 195.

**Si cette prédiction est fausse, `decoder.tex:66-71` est faux et doit être
réécrit.** C'est le point de la campagne où le papier peut perdre quelque
chose, et c'est pour ça qu'il est pré-enregistré.

---

## 3. Les bras, et ce qui est déjà fixé à leur sujet

| bras | index | classes | `L` max | record | taux VRAM |
|---|---|---|---|---|---|
| `Planes14` *(incumbent)* | 47 + 1 | 301 | 5 | 14 o | 4,804 b/poids *(mesuré)* |
| **`Shell12`** | 47 + 1 | **79** | 5 | **14 o** | **identique à `Planes14`, par construction** |
| **`MonoShell3`** | 24 | **4** | **3** | 10 à 11 o | **3,33 à 3,67 b/poids** *(calculé)* |

- `Shell12` entre dans le record `Planes14` sans changer un octet : `9 + 1 + 24
  + 3×24 = 106 bits = 14 o`, et les trois plans binaires nomment jusqu'à
  8 niveaux. Le champ classe pourrait tomber de 9 à 7 bits sans effet sur le
  stride.
- `MonoShell3` : `|Shell(3)| = 16 773 120` → **exactement 1,000 bit/poids de
  payload**, et **4 classes** exhaustivement prouvées
  (`llvq-core/tests/g1_invariants.rs:445-566`), à `L ≤ 3` donc 2 plans
  binaires. ⚠️ Son taux **VRAM** n'est pas 1,0 : payload et taux noyau sont
  deux comptabilités (§6).

**Qualité, fixée d'avance et sans nouvelle mesure** : rétention en distorsion
de 90,34 % pour la coquille 12 seule contre 92,14 % pour la boule (CLAUDE.md
§6, banc gaussien 20 000 blocs, graine figée). Aucun chiffre de qualité LLM
n'est promis pour ces deux bras.

---

## 4. 🚨 La résolution du banc, et la règle de décision qui en découle

**Fait établi avant la campagne** (trois journaux, même carte, même modèle,
même protocole, seul le jeu de bras change) :

| journal | bras | Slot32 | **Planes14** | Planes12x |
|---|---|---|---|---|
| `mesures/c1-planesbench-2026-08-06.txt:23-29` | 3 | 1,89× [1,89–1,89] | **2,16× [2,16–2,16]** | — |
| `mesures/nuit-planes12x-q8-2026-08-07.txt:25-32` | 4 | 1,89× [1,89–1,89] | **2,16× [2,16–2,16]** | 2,01× [2,01–2,01] |
| `mesures/e2-golay70-bench-2026-08-07.txt:27-36` | 5 | 1,87× [1,86–1,88] | **2,14× [2,11–2,15]** | 1,98× [1,95–1,99] |

Les plages entre 4 et 5 bras sont **disjointes** : 0,93 % d'écart sur
`Planes14`, 1,5 % sur `Planes12x`. Le rapport registres/spill est **identique**
dans les trois runs — le détecteur habituel ne voit rien. Et les octets, eux,
ne bougent pas d'un chiffre.

### Le protocole de contrôle, obligatoire

Tout job qui ajoute un bras exécute, **dans le même processus**, le banc à
N bras **et** un contrôle à 5 bras identique au run publié. Le job rapporte
`Δ_contrôle` = le plus grand écart relatif observé sur les quatre bras
incumbents entre les deux.

### La règle de décision, posée maintenant

Soit `Δ_mesuré` l'écart relatif entre deux bras, et
`R = max(Δ_contrôle, demi-étendue intra-run du bras le plus dispersé)`.

| condition | verdict autorisé |
|---|---|
| `\|Δ_mesuré\| > 2R` | **séparation** — les deux bras diffèrent |
| `\|Δ_mesuré\| < R` | **indiscernable à cette résolution** — jamais « égaux », jamais « parité » tout court |
| entre les deux | **non résolu**, publié comme tel |

**Aucun rapport n'est cité contre un jeu de bras qui ne l'a pas produit.**

---

## 5. Les issues, et ce que chacune fait au papier

| Issue mesurée | Conséquence, décidée d'avance |
|---|---|
| `Shell12` et `Planes14` **indiscernables** en Go/s (§4) | La prédiction tient. Le multi-shell est gratuit, chiffré pour la première fois, et `decoder.tex:66-71` cesse d'être une affirmation. **Va en abstract.** |
| `Shell12` **séparé** au-dessus en Go/s | La prédiction est fausse. Les classes coûtent, on chiffre le prix, et l'abstract cesse de présenter le multi-shell comme sans coût. **Publiable, et plus intéressant que le cas attendu.** |
| `MonoShell3` plus rapide que `Planes14` | **Attendu, non informatif seul** : sur un noyau memory-bound, moins d'octets = plus vite. Se publie **uniquement** comme point de la courbe débit↔taux, avec son taux de code (1,000 b/poids) et sa qualité en légende. **Jamais présenté comme une course perdue ou gagnée.** |
| `Planes14` plus rapide que QTIP à taux VRAM comparable | L'affirmation la plus forte du papier devient défendable. **Va en abstract.** |
| `Planes14` plus lent que QTIP | Le papier reste publiable et devient meilleur : il documente ce que le papier LLVQ affirme sans le montrer. **L'abstract est réécrit pour ne plus suggérer un avantage vitesse global.** |
| Un concurrent ne rentre pas dans le protocole | Stratégie B avec mention en toutes lettres dans la légende **et** le texte, ou déclaré non mesuré en Limitations. |

**Aucune de ces issues ne justifie de ne pas publier.** Elles changent le titre
et l'abstract, pas la décision.

### Ce qui est exclu d'avance

- **QTIP n'entrera pas dans le balayage de batch** : son noyau est `N = 1` en
  dur (`inference.cu:462`). La case sera vide et sa légende le dira.
- **Le 4 bits ne sera pas cité sur un point unique à M = 1** : Marlin est une
  GEMM dont la plus petite tuile en M est 8, et à M = 1 tous les noyaux 4 bits
  convergent vers la même borne de bande passante. Le 4 bits n'existe que dans
  une table batchée.
- **Aucune qualité LLM ne sera revendiquée pour QTIP depuis nos mesures** : ses
  checkpoints publics sont tous des Llama. Sa qualité est **citée** depuis
  `paper/sections/evaluation.tex:127`, avec sa provenance. ⚠️ Et elle porte une
  réserve : le noyau CUDA de QTIP n'implémente que **HYB**, pas le **3INST**
  dont ce tableau cite la qualité. Vitesse et qualité viennent de deux
  configurations différentes, et ce sera écrit.

---

## 6. Comptabilité d'octets, figée avant de mesurer

Toute comparaison mémoire de cette campagne se dit en **comptabilité noyau** :
flux + queue f32 + échelles de ligne f32, rapportée à **tous** les poids de la
matrice, queue comprise. C'est celle de `llvq-bench/src/bin/rtbits.rs:63-85`,
et celle qui a produit 5,510 / 4,804 / 4,342 / 3,589.

Ce n'est **pas** le payload (5,3756 / 4,6667 / 4,2029), ni le b/param modèle
entier. 🚨 `docs/data/echelle-formats.csv` nomme aujourd'hui sa colonne
`bpw_payload` et `paper/sections/layouts.tex:36` dit « Payload rates » : les
deux sont faux et **seront corrigés avant qu'un concurrent soit posé sur cet
axe**.

Trois asymétries connues, conservées telles quelles et **documentées en
légende** plutôt que corrigées en silence :

1. le bras FP16 facture sa queue à 16 bits, les bras LLVQ la leur à 32 — parce
   que les noyaux de banc lisent vraiment une queue f32 (`gpu.rs:498-506`).
   **Ça joue contre nous** ;
2. `Golay70` facture son padding de flux (`llvq-artifact/src/runtime.rs:1546`),
   les trois autres non ;
3. les tables de classes ne sont facturées par personne, et `Golay70` en lit
   deux que les autres n'ont pas.

---

## 7. Les gardes qui doivent passer avant tout chronométrage

Repris du protocole existant, sans dérogation :

- une seule L40S, un seul processus, bras entrelacés dans chaque round, ordre
  de dispatch fixe ;
- 7 rounds, les 2 premiers jetés ; tout rapport est la **médiane des rapports
  formés round par round**, avec son étendue — jamais un quotient de deux
  minima ;
- vérification f64 **ligne à ligne de chaque bras** avant tout chronométrage,
  seuil `1e-5`, pires erreurs attendues dans 2,2–3,0·10⁻⁸ ;
- **zéro octet de mémoire locale** au rapport de registres. ⚠️ Nécessaire, pas
  suffisant : ce rapport est identique dans les trois runs du §4 qui divergent ;
- chaque bras nouveau porte **sa propre** référence f64 ; les bras qui
  prétendent au même contenu décodé restent tenus à l'égalité **bit à bit**
  entre eux ;
- coût GPU par job enregistré dans `docs/data/jobs.csv`.

---

## 8. Ce qui invaliderait ce pré-enregistrement

À dire maintenant, pour qu'on ne puisse pas le dire après :

- si le contrôle à 5 bras du §4 **ne reproduit pas** le run publié dans ses
  plages, le banc a dérivé pour une autre raison et **aucun chiffre de la
  campagne n'est publiable** avant de savoir laquelle ;
- si le ré-encodage mono-shell ne passe pas la vérification f64, le bras
  n'existe pas — on ne mesure pas un décodeur dont on n'a pas prouvé la
  reconstruction ;
- si `Shell12` ne tient pas le stride de 14 o, il n'est plus iso-taux et la
  question du §1 change de nature : il faudra le dire, pas ajuster la question.
