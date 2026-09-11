# BROUILLON — Pré-enregistrement : le fichier Tetra + Q5 servi par son noyau

**⚠️ CE FICHIER N'EST PAS TAMPONNÉ ET N'AUTORISE AUCUNE MESURE.** Il porte
`BROUILLON-` dans son nom exactement pour ça : la règle dure 2 exige un tampon
avant la première mesure, et trois questions d'opérateur (§7) restent ouvertes.
Quand elles auront des réponses, ce fichier est renommé sans le préfixe,
commité, tamponné, et **ne s'édite plus** — ce qui le corrigera ira dans un
`-ECARTS.md`.

**Go de l'opérateur du 2026-09-08 : l'objet servi est Tetra + `v_proj` en int4,
encodé sur le Mac.** Pas V32 : sa base exige le volume ×32, qui ne tient sur
aucune carte louable sous 96 Go, a coûté 13,60 $ la seule fois où il a tourné,
et dont l'incrément Q5 a un IC qui contient zéro.

**Coût annoncé** : encodage 2 h 30 sur le Mac à **0 $** ; tuerie précoce
**~0,02 $** ; F1d **~1,00 $ par carte** ; F1e **~8,00 $**. Total au pire
**~10,05 $**, dont 8 ne sont dépensés que si la tuerie précoce est verte.
Total projet avant : **134,14 $** (*mesuré*, `docs/data/jobs.csv`, 119 lignes
tarifées).

---

## §1 — L'objet, et pourquoi celui-là

Arm T2 du chantier 1, mesuré le 2026-09-06 sur carte, apparié sur 2 280
questions (*mesuré*, [journal](../docs/mesures/q5-tetra-2026-09-06.txt)) :

```
  T0  Tetra nu                53,49          2,7645 b/param
  T2  Tetra + v_proj int4     56,95          2,8138 b/param
      écart apparié          +3,47 pp   IC95 [+1,42 ; +5,57]   McNemar 8,6e-5
```

Ces 56,95 ont été obtenus par **restauration** — `LLVQ_RESTORE_Q4=v_proj` prend
`v_proj` du checkpoint f16 et le quantifie-déquantifie au chargement
(`llvq-llm/src/sealed.rs:285-307`, dont le commentaire dit lui-même *« without
pretending a 4-bit kernel exists »*). **Aucun enregistrement int4 n'a jamais été
lu par quoi que ce soit.**

Ce chantier écrit le fichier pour de vrai, et le fait lire par un noyau. C'est
la différence entre un chiffre de qualité et un objet servi.

## §2 — L'encodage, verbatim

Le même appel que le Tetra du 2026-09-06 (prereg `tetra-4b-2026-09-06` §3),
avec une variable de plus :

```bash
export LLVQ_MODEL=Qwen/Qwen3-4B
export LLVQ_CALIB=c4
export LLVQ_INT4_TYPES=v_proj
export LLVQ_ARTIFACT=$OUT/q4b-tetra-q5.llvq
export LLVQ_THREADS=12          # ncpu−4, le poste reste utilisable
nice -n 10 cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke \
  -- 64 2048 12 4096 metal nogs tetra 999 rot

LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal \
  -- $OUT/q4b-tetra-q5.llvq $OUT/qwen3-4b-tetra-q5.bin
```

Positionnels : `n_calib · calib_len · n_eval · eval_ctx · device · gs/nogs ·
codebook · limit · rot`. Rien d'autre ne change par rapport au fichier de
référence : même corpus, même rotation, même `nogs`, même graine, mêmes 64
fenêtres, ρ = 1. **Une seule variable.**

Le chemin d'écriture est déjà câblé et je l'ai relu avant d'écrire ceci :
`bin/smoke.rs:666-670` déclare `{Tetra, Int4G128}` dès que `LLVQ_INT4_TYPES`
est posée ; `calib.rs:793-812` range ces matrices en affine par groupe dans la
base naturelle — **jamais GPTQ, jamais la rotation, jamais le réseau** — et
réinjecte le tenseur déquantifié pour que les activations suivantes du bloc
voient les poids que le fichier stocke ; `seal.rs:153` fait passer les
enregistrements int4 tels quels.

## §3 — Les contrôles de l'encodage, avant toute carte

À la sortie de `smoke` et de `seal`, sur le Mac, à 0 $ :

| # | contrôle | attendu |
|---|---|---|
| C1 | l'en-tête déclare `{Tetra, Int4G128}` | les deux, pas un |
| C2 | nombre d'enregistrements int4 | **36 sur 252** — un `v_proj` par bloc |
| C3 | le débit imprimé par `smoke` | **2,8138 b/param**, une seule ligne |
| C4 | `verify_artifact` | bit pour bit |
| C5 | perplexité du scellé rouvert, f16 | finie, à 1 % de 16,16 |
| C6 | taille du scellé | comparée aux 1 770 529 149 octets du Tetra nu ; la différence est la substitution des 36 matrices |

**C3 est le contrôle qui compte.** Le 2,8138 du dossier est une arithmétique
posée à la main : `rtbits` refuse Tetra (`llvq-bench/src/bin/rtbits.rs:538`).
Si le chiffre imprimé et le chiffre calculé divergent, c'est ça le résultat de
la journée, et rien ne part sur carte avant que l'écart soit expliqué.

## §4 — La tuerie précoce, 0,02 $ et vingt secondes

Avant de dépenser les 8 $, un seul job de vingt secondes sur `l40sx1` :
`f1rankfloor` avec deux bras neufs — `tv_f1r_v3g` (le décodage servi de
`llvq_tetra48.cuh`) et **`Planes14` dans le même processus**, sur un flux
synthétique.

Signal de tuerie, à écrire dans le prereg avant le lancement :

```
  t(v3g) > 0,90 × B mesuré dans le même processus       →  l'axe s'arrête
  num_regs > 40, ou local_bytes ≠ 0, sans réduction connue →  l'axe s'arrête
```

Les registres et les octets locaux se lisent sur la fonction chargée
(`llvq-cuda/src/gpu.rs:402-425`), pas sur une projection. Médianes avec
étendues formées tour par tour (règle 7), rapport à tête identique publié à
côté du rapport brut (règle 4), Go/s contre AWQ et QTIP et jamais un × entre
cartes (règle 5).

**Ce que ce bras vaut même si Tetra meurt** : le `B` en processus est un
chiffre que `docs/format-noyau.md` §6 n'a jamais eu, et le décodeur v3g est le
décodeur de n'importe quel format qui ne déplie pas.

## §5 — Ce que F1d et F1e mesurent, et ce qu'ils tuent

Les deux portes sont des **tueries**, pas des mesures — c'est ce qui ordonne
tout le chantier.

| porte | mesure | seuil | tuerie |
|---|---|---|---|
| F1d | débit et VRAM, `planesbench`, Planes14 dans le même processus | `t ≤ t(Planes14)` et `≤ 2,20 b/poids` noyau | `t > t(Planes14)` : plus lent pour moins d'octets |
| F1e | `fusedrun` + MMLU apparié sur le scellé | noyau ≤ 3,00 b/poids, tok/s ≥ 100,6, MMLU ≥ sa barre | MMLU < 53 % |

Deux obligations sur le job à 8 $, chacune capable de le perdre entièrement :

1. **`oracle` en premier**, comme bras nommé et chronométré, sur le backend qui
   tourne (règle dure 10).
2. **`LLVQ_MMLU_DUMP` sur *chaque* bras.** `mmlupair` refuse d'apparier sans
   deux dumps, et `bin/mmlu` n'a pas de reprise : un timeout perd le bras et le
   facture. Précédent : V32, tué à 148 min contre un plafond de 120 posé sur
   une estimation fausse d'un facteur 1,7.

## §6 — Prédiction signée

À remplir **avant** le lancement, et lue contre le résultat quoi qu'il arrive.

| grandeur | prédiction | fourchette |
|---|---|---|
| MMLU du fichier servi | | |
| tok/s au 4B | | |
| Go sur carte | | |
| b/poids noyau | | |
| `num_regs` de `tv_tetra48` | | |

La projection du dossier, à titre d'ancrage et pas de prédiction : ~113 tok/s
au 4B pour 1,36 Go (*estimé*,
[F1 variantes](../docs/mesures/f1-rang-variantes-2026-09-05.txt)), contre
100,6 tok/s pour 2,57 Go servis en Planes14 (*mesuré*,
`docs/format-noyau.md:126-129`). Elle sort d'un plancher de décodage sur flux
synthétique et **elle ne porte ni la queue, ni le `v_proj` int4, ni la
permutation** — elle sera fausse, la question est de combien et dans quel sens.

## §7 — Les trois questions d'opérateur, ouvertes

Aucune ne se lit dans le code, et le fichier ne peut pas être tamponné avant
qu'elles aient des réponses.

1. **F1e parle-t-il du format ou du produit ?** Comparer ce fichier mixte aux
   55,59 publiés bouge *deux* variables — le format et la précision de
   `v_proj`. Si la porte doit attribuer au format, l'objet de comparaison est
   Tetra nu à 53,49 et le fichier mixte est un second bras.
2. **Quelle SE dans « 55,59 − 2 SE » ?** La SE par bras vaut 1,35, la SE
   appariée à fichiers différents ≈ 1,40 (`docs/ETAT.md:413-414`). La barre
   bouge de 0,1 point, ce qui n'est rien — sauf que la porte est écrite sans le
   dire, et une porte ambiguë se relit après coup.
3. **« disque ≤ celui d'aujourd'hui » est-il une vraie porte ?** Tetra est
   déjà **+1 616 octets** au-dessus du fichier publié
   (`docs/mesures/tetra-4b-2026-09-06.txt:49`), et le mixte bougera encore.

## §8 — Limite de portée, écrite d'avance

Tout ceci est **au 4B**. Le dossier a déjà mesuré que l'écart de Tetra double
du 4B au 8B sur les deux axes — perplexité de −4,64 % à +0,73 %, MMLU de
−2,10 à −3,91 pp (*mesuré*, `docs/ETAT.md` §5 septies). Un F1e vert au 4B ne
dit rien du 8B, et le prereg ne le laissera pas croire.
