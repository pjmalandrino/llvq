# Pré-enregistrement — étape 6.0 : l'objet servi, Tetra + `v_proj` en int4, encodé sur le Mac

**Écrit, commité et TAMPONNÉ le 2026-09-09, AVANT le premier bloc.** Go de
l'opérateur du 2026-09-09.

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Portée : l'étape 6.0 seule** — produire le fichier et le contrôler. F1d et
F1e ont leur propre pré-enregistrement à écrire, et les trois questions
d'opérateur encore ouvertes (F1e parle-t-il du format ou du produit ; quelle SE
dans « 55,59 − 2 SE » ; « disque ≤ aujourd'hui » est-il une vraie porte)
appartiennent à celui-là, pas à celui-ci.

**Coût : 0 $.** 2 h 30 de Mac (*estimé*, le Tetra nu du 2026-09-06 a pris
2 h 27 = 8 820 s, 245 s/bloc). Aucune carte. Total projet avant : **134,24 $**.

---

## §1 — L'objet, et pourquoi il n'existe pas encore

Arm T2 du chantier 1, mesuré sur carte le 2026-09-06 (*mesuré*,
[journal](../docs/mesures/q5-tetra-2026-09-06.txt), prereg tamponné) :

```
  T0  Tetra nu             53,49
  T2  Tetra + v_proj int4  56,95
      écart apparié       +3,47 pp   IC95 [+1,42 ; +5,57]   McNemar 8,6e-5
```

Ces 56,95 ont été obtenus par **restauration** : `LLVQ_RESTORE_Q4=v_proj`
prend `v_proj` du checkpoint f16 et le quantifie-déquantifie au chargement
(`llvq-llm/src/sealed.rs:285-307`, dont le commentaire dit lui-même *« without
pretending a 4-bit kernel exists »*). **Aucun enregistrement int4 n'a jamais
été écrit dans un fichier ni lu depuis un fichier.**

Cette étape écrit le fichier. C'est la différence entre un chiffre de
configuration et un objet.

## §2 — Le protocole, verbatim

```bash
export LLVQ_MODEL=Qwen/Qwen3-4B
export LLVQ_CALIB=c4
export LLVQ_INT4_TYPES=v_proj
export LLVQ_ARTIFACT=$OUT/q4b-tetra-q5.llvq
export LLVQ_THREADS=12                     # ncpu−4 sur 16 cœurs, le poste reste utilisable
nice -n 10 cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke \
  -- 64 2048 12 4096 metal nogs tetra 999 rot

LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal \
  -- $OUT/q4b-tetra-q5.llvq $OUT/qwen3-4b-tetra-q5.bin
```

**Une seule variable** par rapport au Tetra du 2026-09-06 : `LLVQ_INT4_TYPES`.
Même corpus, même rotation, même `nogs`, mêmes 64 fenêtres, ρ = 1, mêmes
positionnels.

Le chemin d'écriture est câblé et relu avant d'écrire ceci : `bin/smoke.rs:666`
déclare `{Tetra, Int4G128}` dès que la variable est posée ; `calib.rs:793-812`
range ces matrices en affine par groupe dans la base naturelle — **jamais GPTQ,
jamais la rotation, jamais le réseau** — et réinjecte le tenseur déquantifié
pour que les activations suivantes du bloc voient les poids stockés ;
`seal.rs:153` fait passer les enregistrements int4 tels quels.

## §3 — Les six contrôles, tous à 0 $

| # | contrôle | attendu |
|---|---|---|
| C1 | l'en-tête déclare `{Tetra, Int4G128}` | les deux |
| C2 | enregistrements int4 | **36 sur 252** — un `v_proj` par bloc |
| C3 | **le débit imprimé par `smoke`** | **relevé et confronté**, PAS gaté — voir §4 |
| C4 | `verify_artifact` | bit pour bit |
| C5 | perplexité du scellé rouvert, f16 | finie, à 1 % de 16,16 |
| C6 | taille du scellé | comparée aux 1 770 529 149 o du Tetra nu ; la différence est la substitution des 36 matrices |

## §4 — Pourquoi C3 n'est pas une porte, et c'est un écart assumé au plan

Le plan de `docs/ROADMAP.md` §2.2 quinquies écrivait : *« le débit imprimé par
`smoke` : **2,8138 b/param**, une seule ligne »*. **Ce seuil est retiré**, et la
raison est mesurée.

Trois reconstructions indépendantes du même nombre, le 2026-09-09 :

```
  2,8126   (recalcul adverse)
  2,8130   (mon recalcul : (P−V)·2,1498 + V·4,25 + E·8,5, sur T)
  2,8138   (le chiffre porté par ROADMAP et ETAT)
```

L'**incrément** est identique aux trois (+0,0493 sur le Tetra nu) ; c'est la
base qui se reconstruit à 0,0008 près. Et **aucun instrument n'a jamais imprimé
ce nombre** : `rtbits` refuse un fichier v5 (`llvq-bench/src/bin/rtbits.rs:538`).

Gater sur un chiffre que trois calculs ne reproduisent pas, ce serait poser une
porte sur une convention, pas sur une mesure. **C3 relève ce que `smoke`
imprime, l'écrit dans le journal à côté des trois reconstructions, et la
réconciliation devient le travail de l'étape 6.7** — celle qui apprend à
`rtbits` à tarifer un fichier mixte.

Ce qui EST vérifié au chiffre et le reste : **2,2044 b/poids noyau**, sous
b_max = 3,00.

## §5 — Prédiction signée

| grandeur | prédiction | fourchette |
|---|---|---|
| durée d'encodage | **2 h 35** | 2 h 10 – 3 h 10 |
| enregistrements int4 | **36** | exactement 36 |
| débit imprimé, b/param | **2,8138** | 2,8120 – 2,8145 |
| perplexité du scellé, f16 | **16,10** | 15,90 – 16,40 |
| taille du scellé | **plus grande** que le Tetra nu | +0,5 à +2,5 Mo |

Le raisonnement sur la ppl, pour qu'il soit réfutable : `v_proj` en int4 g128
est **plus précis** que le réseau à 2 bits, donc la perplexité doit s'améliorer
légèrement par rapport aux 16,1569 du Tetra nu — mais `v_proj` ne pèse que
2,60 % des poids de projection, donc l'effet est petit. Je prédis un gain, pas
une révolution.

Sur la taille : les 36 matrices passent de 48 bits/bloc à 4,25 bits/poids, soit
plus d'octets, donc **le fichier grossit**. Il était déjà +1 616 o au-dessus du
publié ; il s'en éloigne. La ligne « disque ≤ aujourd'hui » du plan est morte et
ce prereg l'enregistre plutôt que de l'ignorer.

## §6 — Ce que cette étape n'établit pas

- **Aucun débit servi.** `tv_q4_h` n'a jamais tourné sur une carte, et le
  chargeur mixte refuse encore ce fichier en trois endroits
  (`fused.rs:1775`, `:1787`, `:1791`). L'objet sera lisible par l'évaluation,
  pas par un noyau.
- **Aucune MMLU.** Elle vient après, sur carte, et elle doit reproduire les
  56,95 mesurés par restauration. `llvq-llm/tests/int4_wiring.rs:51`
  (`the_two_int4_paths_agree_bit_for_bit`) dit que les deux chemins coïncident
  bit pour bit ; c'est une raison de le prédire, pas de s'en dispenser.
- **Rien au-delà du 4B**, et le dossier a mesuré que l'écart de Tetra double du
  4B au 8B sur les deux axes.
- **Le contrôle 0 reste échoué** : le dépôt ne reproduit plus son artefact
  publié depuis le 2026-08-26, donc tout écart contre Planes14 porte le format
  ET la dérive d'encodeur. Le témoin qui les sépare coûte 4 h de Mac à 0 $ et
  n'a pas été lancé.
