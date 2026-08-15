# Passation — soir du 2026-08-15 : E1v est la branche, et voici son reste à faire

> **Pour la session qui reprend.** Ce document est autonome. Il **complète**
> [`passation-exec-2026-08-15.md`](passation-exec-2026-08-15.md), dont le §2
> est **périmé sur P1 seulement** — tout ce qu'il donne comme restant (fixture,
> sweep, aller-retour GPU, banc) est fait et mesuré.
>
> 27 commits, `6d165ca..`, **0 $** — aucun job GPU loué, tout sur le M3 Max.

## 1. Ce que la journée a rendu, en six lignes

- **P1 mesuré et clos** : marche 0,3097 ns/bloc · cascade uniformisée 1,7443 ✅
  · archive 10,4093 ❌. Uniformiser la boucle vaut **un ordre de grandeur** sur
  les mêmes bits et la même table.
- **P1b mesuré** : un **bloc** coûte **0,6704 ns**, pas 0,3097. Le gate de P4
  porte sur un bloc ⇒ **l'autorisation du bras CUDA est RETIRÉE**.
- **P5 complet, quatre critères verts** — E1v a une largeur, une bijection
  prouvée, un décodeur sans division et un transcodage à 1,088×.
- **`e1c14` enterré sur papier** : son argument de diffusion ne tient pas sous
  la décomposition servie, et l'aligner le rend plus gros que `Planes14`.
- **E1v survit au même piège à +0,48 %**, et c'est ce qui en fait la branche.
- Trois pré-enregistrements horodatés, trois journaux dans `docs/mesures/`.

## 2. E1v — la branche, qualifiée

### 2.1 Pourquoi elle est intéressante, en un tableau

Le même piège de géométrie, sur les deux formats :

| | non aligné | aligné sur le warp | surcoût |
|---|---|---|---|
| `E1c14` | 4,5551 | **5,2354** | **+15,47 %** ☠️ |
| **`E1v`** | 2,3877 | **2,3983** | **+0,48 %** ✅ |

🔎 **La raison est structurelle et il faut la retenir.** Un groupe `E1c` coûte
`24·(1+plans)` mots **quelle que soit la lane occupée** : un groupe partiel de
10 blocs coûte le prix de 32. Un groupe `E1v` coûte 32 bits de mot de base
**plus la somme de ses records**, arrondie au mot : un groupe partiel coûte ce
que ses records coûtent. **La largeur variable — ce qu'E1v paie en complexité
de décodage — est exactement ce qui le sauve ici.**

### 2.2 Ce qu'E1v a déjà

| axe | état | provenance |
|---|---|---|
| mémoire | **2,3877 b/poids noyau** → 2,979 b/param → **1,50 Go** sur le 4B (contre 2,60 servi) | calculé sur 53,7370 bits/bloc **mesurés sur les octets écrits** |
| bijection | **150 681 600 blocs**, zéro écart, plus la fixture origine | mesuré, P5 C2 |
| décodeur | **zéro division** (source *et* assembleur release), ≤ 96 pas (90 max), invariant par classe | mesuré, P5 C3 |
| flux | aller-retour prouvé sur le fichier entier ; octets écrits == comptabilité à 5e-3 | mesuré |
| transcodage | **1,088×** `Planes14`, mono-thread, médiane de 3 passes | mesuré, P5 C4 |
| qualité | **inchangée par preuve** — ppl 16,9422, MMLU 55,59 % restent ceux de l'archive | par bijection |

### 2.3 Le RAF, par ordre de ce qui bloque quoi

**1 — Aucun noyau GPU, et le 0,6704 ns N'EST PAS E1v.** C'est le point à ne pas
arrondir. Le bras `marche-bloc` lit un record à **stride fixe de 12 octets**.
Le flux E1v réel a des **largeurs variables** et un **warp-scan** : une lane
doit lire le mot de base de son groupe, obtenir son propre décalage par une
somme préfixe sur les 32 largeurs du groupe, *puis* décoder. **Personne n'a
mesuré ça.** Le 0,6704 est donc une **borne inférieure**, et l'écart au gate de
0,45 ne peut que se creuser.

**2 — Le ×2,17 non attribué.** Entre décoder une marche et décoder un bloc.
Deux hypothèses testées, deux réfutées : le compte de pas prédisait ×1,002, et
le bras qui supprime le débordement de registres est **24 % plus lent**. Une
troisième explication en prose serait la quatrième erreur du même genre.
L'attribuer demande un profileur ; cette machine n'a que les Command Line
Tools, pas Xcode.

**3 — L'échelle.** Les 2,3877 sont mesurés sur le **4B seul**. La dérive
4B→8B est publiée à +0,001 bit/bloc **pour la variante par étage** ; la
variante **par genre** n'a jamais été calculée hors du 4B.

**4 — Rien de bout en bout.** E1v n'est câblé dans aucun modèle : ni tok/s, ni
Go carte mesurés. Le 1,50 Go est un chiffre de **format**.

**5 — C4 n'a qu'un point** : une matrice, la plus grosse. Conforme au §4, mais
le transcodage du modèle entier n'est pas chronométré.

## 3. Les autres phases, en une ligne chacune

| | état | RAF |
|---|---|---|
| **P1**, **P1b** | ✅ clos | — |
| **P5** | ✅ clos, 4/4 | ⚠️ le sweep n'exerce pas l'origine (le 4B n'en porte aucune) ; elle est prouvée par fixture |
| **P4** | 🔒 bloqué | voir §4 |
| **P2**, **P6** | ⏸ en pause (opérateur) | `--trace` (l'info temporelle est détruite **dans le hook**), run ~1,4 $, amendement É0 purgeant gpt-oss |
| **P7** | non ouvert | gaté sur un package validé au 8B |
| ouvert par **P3** | — | le KV q8 à contexte long ; demande un instrument qui garde le modèle résident entre les deux bras |

## 4. P4 — ce qui reste, et ce qui a disparu aujourd'hui

- ❌ bras cascade/marche : **retiré** (gate non franchi sur un bloc)
- ❌ `e1c14` : **enterré sur papier**
  ([`x3-alignement-warp-2026-08-15.txt`](../mesures/x3-alignement-warp-2026-08-15.txt))
- ⏳ **`e1c12` est sans verdict** — refaire le calcul d'alignement **avec le
  terme d'exceptions** (3,38 % des blocs), que le modèle du journal omet : il
  rend 4,1404 pour `Planes12x` là où le dépôt publie 4,342, et **les deux côtés
  du rapport en dépendent**. 0 $, une heure. **À faire en premier** : laisser un
  bras dans un état où quelqu'un pourrait le mesurer en croyant qu'il a un
  verdict est le pire des états.
- ❌ `cublasf16`, `mvkf16`, `nullk`, `planes14k`, `planes12xk`, `golay70v2k` :
  **aucun code**, plus le chronométrage par forme via events CUDA sans lequel
  K2 n'est attribuable à rien
- ⏳ décision de tuile (32 du §2.8 pour la famille k, contre 128 de l'incumbent)
- ⏳ pré-enregistrement **non horodaté**
- 🔒 critère d'admission incomplet tant qu'A2, A4 et A6 sont vides

✅ **Acquis aujourd'hui** : les huit bras sont **enregistrés en dernier** dans
`arms.rs` (ordre de dispatch figé, §2.3), le plan de phases du §2.4 est une
valeur testable, et **un bras sans noyau est refusé PAR NOM** — enregistrer un
nom n'est pas implémenter un bras, et le sélectionner dispatcherait un noyau
inexistant sur une carte louée.

## 5. Ce qui revient à l'opérateur

| | pourquoi ça bloque |
|---|---|
| **A2** (contexte), **A4** (marge), **A6** (offload) | la note n'en recommandait aucune ; sans elles le §B ne fait pas foi et P4 n'a pas de critère d'admission |
| unité de « **32 Go** » | GiB ou décimal — ±0,28 b/poids, plus que ce que le KV q8 achète |
| tampon de **P4** | avant son job |
| **Xcode** (~10 Go) | seule façon d'attribuer le ×2,17 au lieu de le constater |

✅ Arbitré le 2026-08-15 : **A1** (RTX 5090 32 Go), **A3** (KV q8), **A5**
(≤ 60 s/doc en lot de 8, ≥ 20 tok/s).

## 6. Les dettes déclarées, à ne pas redécouvrir

- **P1 a été tamponné AVANT sa mesure. P5 et P1b l'ont été APRÈS.** Ces deux-là
  prouvent l'existence du document, **pas** son antériorité, qui ne repose que
  sur les commits git. C'est écrit en tête de leurs journaux.
- Le **miroir CUDA d'X3** est une **transcription**, pas une implémentation
  partagée : rien ne détecte mécaniquement sa dérive avec le `.cuh`.
- **`g6_pack` échoue en debug** et passe en release — antérieur à ces sessions,
  `pack.rs` n'a pas bougé depuis `51d7c55`.
- `docs/note-produit-2026-08-13.md` **est entré dans l'historique** par un
  `git add -A` (commit `c201c12`) dont le message ne le mentionne pas.

## 7. Ce que la journée apprend, et qui vaut au-delà d'elle

**Cinq défauts trouvés par un garde qui pouvait tomber, et deux mesures
arrêtées avant de compter** : l'étalon de la marche relisait la table qu'il
vérifiait (mutation) ; `cascade-archive` faux sur 883 blocs sur 16,7 M (V0) ;
la parité du bloc accumulait sur les mauvais créneaux, 4,1 M blocs (V0) ; C1 et
C3 de P5 étaient contradictoires (relecture avant code) ; une largeur par genre
jamais calculée depuis le 08-12.

🚨 **Et trois prédictions fausses, toutes du même genre.** Le compte de pas
annonçait ×1,002 entre une marche et un bloc — c'est ×2,17. Le débordement de
registres était l'hypothèse suivante — le bras qui le supprime est 24 % plus
lent. C'est la **troisième fois** sur ce projet qu'un compte niveau source se
trompe d'un facteur ~2, après Golay70 et E1c.

> **La règle qui en sort, et elle est chère** : sur ce noyau, un compte
> d'opérations n'est pas une prédiction de temps, **même quand il porte sur la
> boucle qu'on croit dominante**. Ce qui reste utilisable d'une lecture de
> source, ce sont les **identités de comptage** — celles qui ont enterré E3, et
> `e1c14` aujourd'hui : elles ne dépendent d'aucun matériel et ne se
> contournent pas par un meilleur noyau.
