# Pré-enregistrement — le quatrième barreau, et la réplication à seconde graine

**Écrit, commité et tamponné AVANT le premier des deux jobs.** Suite de
[`preregistration-bits-de-gain-2026-08-25.md`](preregistration-bits-de-gain-2026-08-25.md)
(sha256 `428f17d4…`), dont l'étage 1 est mesuré et scellé — journal
[`docs/mesures/gain-ab-gate-0.6b-2026-08-25.txt`](../docs/mesures/gain-ab-gate-0.6b-2026-08-25.txt).
Ce document ne l'amende pas et ne le rouvre pas : il en ajoute deux jobs.

**Coût : 0 $** — Mac M3 Max. Go de l'opérateur du 2026-08-25 pour le job A
seulement ; le job B part le soir même, sur go séparé.

---

## §1 — Les deux questions, et ce que ce document ne décide pas

L'étage 1 a mesuré trois points d'une courbe à débit constant (48 bits/bloc,
2,1656 b/poids effectifs) et a rendu une forme en **U** : les deux
configurations spécialisées battent celle qui partage, le codebook servi
ressortant dernier des trois de ~9,5 %.

**Job A — le quatrième barreau.** L'échelle iso-débit a quatre rungs, pas
trois, et nous n'en avons mesuré que trois. Le quatrième est `leech4c10` :
44 bits d'index + 4 bits de gain = 48. C'est aussi la dernière ligne de la
Table 8 du papier amont (`Λ₂₄(13)+0`, `Λ₂₄(12)+1`, `Λ₂₄(11)+2`,
`Λ₂₄(10)+4`). Il est le point le plus extrême du côté « tout en magnitude »,
donc celui qui dit si cette branche du U continue de descendre ou se retourne.

**Job B — la réplication.** R1 du document précédent est la seule des quatre
réserves qui peut retourner le signe du résultat : un seul tirage de
calibration, quand F5 mesure σ = 5,2 % entre graines au 4B. Les quatre bras
rejoués à `LLVQ_CALIB_SEED=1` disent si le classement est une propriété des
codebooks ou du texte qu'on leur a donné à lire.

⚠️ **Ce que ce document ne décide pas.** Ni job n'adopte quoi que ce soit :
l'interdit du §0.1 précédent tient, le 0.6B est un proxy et une perplexité
n'est pas une capacité. Le job A est un **point de courbe descriptif**, pas un
candidat à l'admission — même statut que le bras QTIP de F2, et pour la même
raison : il ne concourt pas, il situe. **Aucune issue de ce document n'annule
l'étage 1 ni ne modifie la règle d'adoption du §5.2 précédent.**

---

## §2 — Les deux jobs, verbatim

Tout est identique à l'étage 1 sauf ce qui est nommé. Une seule variable par
job.

**Job A — `leech4c10` au tirage d'ORIGINE** (préfixe contigu depuis le
token 0, `LLVQ_CALIB_SEED` non posé). C'est ce qui le rend comparable aux trois
points déjà mesurés.

```
target/release/smoke 64 2048 12 2048 metal nogs leech4c10 999 rot > ~/llvq-nuit-b/gain-ab-4c10.log 2>&1
```

**Job B — les QUATRE bras à `LLVQ_CALIB_SEED=1`**, qui tire les fenêtres à des
offsets aléatoires sur tout le corpus au lieu du préfixe.

```
LLVQ_CALIB_SEED=1 target/release/smoke 64 2048 12 2048 metal nogs leech0c13 999 rot > ~/llvq-nuit-b/gain-ab-s1-0c13.log 2>&1
LLVQ_CALIB_SEED=1 target/release/smoke 64 2048 12 2048 metal nogs leech1c12 999 rot > ~/llvq-nuit-b/gain-ab-s1-1c12.log 2>&1
LLVQ_CALIB_SEED=1 target/release/smoke 64 2048 12 2048 metal nogs leech2c11 999 rot > ~/llvq-nuit-b/gain-ab-s1-2c11.log 2>&1
LLVQ_CALIB_SEED=1 target/release/smoke 64 2048 12 2048 metal nogs leech4c10 999 rot > ~/llvq-nuit-b/gain-ab-s1-4c10.log 2>&1
```

---

## §3 — Ce qui est vérifié AVANT, et ce qui se contrôle APRÈS

**Vérifié avant d'écrire ce document**, parce que la boule 10 n'était pas plus
couverte que ne l'était la boule 11 : **`index_bits(10) == 44`**,
et c'est épinglé par une assertion ajoutée le 2026-08-25 et **prouvée létale
par mutation** (forcer 45 fait tomber `index_width_follows_the_shell_cap`).
Deux assertions de plus sont posées dans le même test : le couple (10, 4)
entre dans l'invariant `index_bits(cap) + gain_bits == 48`, et **aucune boule
ne coûte 45 bits** — la largeur saute de 44 à 46, ce qui explique pourquoi
l'échelle n'a pas de barreau à 3 bits de gain et pourquoi la Table 8 du papier
le saute aussi.

Table complète, *calculée* par la série thêta de Λ₂₄ et recoupée contre le
nombre de baisers (|Shell(2)| = 196 560) et contre `N_SHELL_13_CUMULATIVE` :

| boule | points cumulés | index | gain pour 48 |
|---|---|---|---|
| Λ₂₄(10) | 13 503 934 538 640 | **44** | **4** |
| Λ₂₄(11) | 40 556 880 458 640 | 46 | 2 |
| Λ₂₄(12) | 111 043 117 458 000 | 47 | 1 |
| Λ₂₄(13) | 280 974 212 784 720 | 48 | 0 |

**Les quatre contrôles du §4 précédent s'appliquent inchangés** à chaque bras
des deux jobs, et si l'un échoue aucun chiffre n'est publié : iso-débit
**2,1656** · configuration résolue (`leech4c10` doit imprimer 4 gain bits,
shell ≤ 10, 48 bits/block) · baseline **19,5038** · `fast-linalg` actif.

⚠️ **Amendement d'un seuil, déclaré ici et non dans le document scellé** : le
contrôle §4.4 exigeait une factorisation « sous ~5 % » et l'étage 1 a rendu
8,7-9,6 % — écart consigné dans [`README.md`](README.md). La forme retenue
pour ces deux jobs est celle qui a du contenu : **la factorisation doit rester
sous 15 %**, et l'avertissement « compilé SANS `fast-linalg` » doit être absent.

---

## §4 — Ce qui se publie, posé d'avance

- **Job A** : un quatrième point sur la courbe du 2026-08-25, dans la même
  table, avec son écart au témoin `leech1c12`. Aucun seuil, aucun verdict
  d'admission. Trois formes possibles, toutes publiables telles quelles :
  `leech4c10` **sous** `leech2c11` (la branche continue de descendre) ·
  **entre** `leech2c11` et le témoin (la branche se retourne) · **au-dessus du
  témoin** (le U n'en est pas un, et c'est `leech1c12` qui est le point
  anormal, pas les autres).
- **Job B** : la même table à quatre points sous un second tirage. Ce qui se
  lit est le **CLASSEMENT**, pas les niveaux — les niveaux bougent avec le
  tirage par construction, c'est ce que F5 mesure. Deux issues : le classement
  **tient** (R1 est levée, l'étage 2 part sur des bases sûres) ou il **bouge**
  (R1 avait raison, et aucun run 4B ne se paie sur cette base).
- ⚠️ **Confondant du job B, déclaré et non corrigé** : `LLVQ_CALIB_SEED` ne
  change pas seulement le tirage, il change le **mode d'échantillonnage**
  (préfixe contigu → offsets aléatoires). Sous wikitext-2, qui est fait
  d'articles longs, les deux ne sont pas le même échantillon. Conséquence
  écrite d'avance : si le classement tient, c'est un résultat **plus fort**
  qu'une réplication à mode constant ; s'il bouge, on ne pourra **pas** séparer
  « autre tirage » de « autre mode », et il faudra un troisième job à
  `LLVQ_CALIB_SEED=2` pour le faire.
- **Ce que ni job ne dira** : quoi que ce soit sur MMLU, sur la vitesse, sur la
  VRAM, ou sur une autre taille de modèle.

---

## §5 — Coût, durées, sorties

- **0 $**, tout en local. Job A ~25-30 min (la boule 10 a moins de classes que
  les trois autres, le sens est favorable, l'amplitude inconnue). Job B ~2 h.
- *Estimé*, pas mesuré : les durées de l'étage 1 sont 30 · 27 · 27 min, et
  aucune n'a été prise à la boule 10.
- **Sorties** : logs bruts commités dans `docs/mesures/gain-ab-2026-08-25-brut/`
  — règle du §7 de `CLAUDE.md`. Journal éditorialisé mis à jour dans
  `docs/mesures/gain-ab-gate-0.6b-2026-08-25.txt`.

---

## §6 — Divulgation datée, à la signature

- L'étage 1 a rendu 39,3309 (`leech0c13`) · 43,4865 (`leech1c12`, témoin) ·
  39,5350 (`leech2c11`), baseline 19,5038, iso-débit 2,1656 aux trois.
- Aucune quantification n'a jamais été lancée à `leech4c10`, à aucune taille.
- Prédiction de l'auteur, écrite pour être opposable : job A rend
  `leech4c10` **entre 39 et 42**, donc sous le témoin mais **pas** sous
  `leech2c11` — c'est-à-dire que la branche se retourne et que l'optimum est
  autour de 2 bits de gain. Job B : le classement **tient**, les niveaux
  bougeant de quelques pour cent.
