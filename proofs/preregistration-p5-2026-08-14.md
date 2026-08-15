# Pré-enregistrement — P5 : la ré-bijection CNS d'E1v, la forme de sa réouverture, et les critères qui la jugent

**Date : 2026-08-14.** Écrit **avant toute ligne de code E1v**. Vérifié au
commit `78bbfc8` :

- **aucune ligne de code E1v, dans aucun crate** : `grep -rn "e1v|E1v|E1V"
  --include='*.rs'` sur `llvq-artifact/src`, `llvq-llm/src`, `llvq-cuda`,
  `llvq-search/src`, `llvq-metal/src` ne rend **rien** ; pas d'`e1v.rs` dans
  `llvq-artifact/src/`. Les seules occurrences du dépôt sont dans
  `llvq-bench/src/bin/radixstudy.rs`, **un compteur de bits**, et les documents ;
- **aucune table de binomiaux `C(n≤24, k≤12)`** — `grep -rn
  "binomial|BINOM|choose("` sur `llvq-search/src` et `llvq-core/src` ne rend
  rien. Il existe une table de **factorielles** (`llvq-search/src/classes.rs:76`,
  `pub(crate)`) et un multinomial par divisions (`:88`, `pub(crate)`) ;
- **aucune combinadique** : il existe un rang de permutation de **multiensemble**
  (`llvq-search/src/index.rs:81`, `:97`, **privés**) et son unranking par
  récurrence `M' = M·c/n` (`llvq-search/src/fastdec.rs:171-194`, **division à
  `:183`**) — **une division par candidat**, exactement ce que la marche
  binomiale promet de supprimer ;
- **le verdict qui ouvrirait P5 n'existe pas** : `cascade-uniformisée` et
  `marche-binomiale` ont zéro ligne de code ([P1](preregistration-p1-2026-08-13.md) §2) ;
- **aucun tampon OpenTimestamps pour P1 ni pour le lot du 13** : `proofs/` ne
  contient que `preregistration-2026-08-10.md.ots` et `…-08-11.md.ots`.

🚨 **Ce document n'est pas dans git, donc son antériorité est NULLE à cette
minute.** Il est rédigé hors de `proofs/` ; ses liens relatifs sont écrits
**comme depuis `proofs/`**, où il doit atterrir. **Condition d'opposabilité** :
commité à `proofs/preregistration-p5-2026-08-14.md` **avant la première ligne de
code E1v**, puis `ots stamp`. D'ici là aucun critère n'est opposable — le défaut
même que le §8 relève sur le journal du 08-13.

> Hérite **sans dérogation** des gardes du [2026-08-10](preregistration-2026-08-10.md)
> (§7), de sa comptabilité (§6), de sa règle de provenance, et de la discipline
> de lecture de [P1](preregistration-p1-2026-08-13.md) §1.4 — aucun seuil ne se
> lit contre un chiffre d'un autre run.

---

## 0. Ce que P5 peut et ne peut pas conclure

**Produit** : un décodeur écrit, une bijection prouvée sur un fichier réel, une
largeur **recalculée depuis le décodeur**, un chronométrage de transcodage à modèle
de threads défini. **Ne produit pas** : un chiffre de vitesse de décodage — P5 ne
touche aucun accélérateur, et la profondeur « 48–96 pas » de `radixstudy.rs:626`
(justifiée `:73-76`) est un **compte niveau source sur du code qui n'existe pas**.

🚨 **Deux faits distincts, deux provenances — ne pas les souder.** (1) Sur Golay70
v2 le compte d'instructions promettait **1,9–2,4×**, la carte a rendu **1,77×** :
surestimation de **1,07× à 1,36×** (`docs/archive/passation-golay70-2026-08-11.md:130-131`,
qui écrit « la fourchette estimée 1,9–2,4× était elle-même optimiste »). (2) Un
compte niveau source a **par ailleurs** déjà été faux d'un facteur 2 sur ce noyau
(`CLAUDE.md`, échelle E1c). La version antérieure les confondait.

**Un vert achète** le droit de porter E1v sur carte avec le reste. **Un rouge ferme,
à 0 $, un format qu'on aurait sinon porté sur CUDA.**

## 0bis. La condition d'ouverture, énoncée exactement

🚨 **P5 ne s'ouvre PAS sur « P1 vert ». Il s'ouvre sur un seul bras.** P1 §6 :
« marche binomiale ≤ 0,45 ns → **P5 s'ouvre** ». Le « meilleur des deux
≤ 0,45 ns » est le **gate CUDA** du §4.2 de P1, règle **distincte** qui autorise
une dépense de job et rien d'autre. P1 §6 nomme la divergence : « cascade
uniformisée ≤ 0,45 ns **mais** marche binomiale > 0,45 → le bras CUDA de P4 est
autorisé et **P5 ne s'ouvre PAS** ».

⚠️ **La tolérance « capacity-first » n'est chiffrée NULLE PART.** Trois
occurrences dans `docs/` et `proofs/` —
`docs/archive/etude-moe-memoire-extreme-2026-08-12.md:180`,
`docs/archive/passation-exec-2026-08-13.md:105-107`, et P1 `:402-403` qui le
constate — **aucune ne porte de nombre**. **Règle posée ici, sans inventer de
chiffre : tant que cette tolérance n'est pas chiffrée dans un document arbitré,
une issue formulée « `cascade-archive` passe la tolérance capacity-first » NE
PEUT PAS fermer P5.** Ce serait un jugement libre rendu après la mesure.

✅ **Et P1 ne la formule plus ainsi.** Depuis son É1 (commit `78bbfc8`,
2026-08-14, §4.3 `:266-283`, §6 `:326-327`) la clause est chiffrée **par un
nombre que P1 portait déjà** : `cascade-archive` **≤ 2,0 ns/bloc** ⇒ **E1v
mort-né** — le seuil que P1 §4.1 impose à la cascade uniformisée. **C'est le seul
énoncé du dossier qui peut fermer P5 par le haut**, et P5 s'y soumet : l'archive
fait **2,1912 b/poids noyau** contre **2,3709** pour E1v, elle est plus petite, et
son décodeur existe et est prouvé.

**Priorité, décidée d'avance : le kill amont PRIME.** Si `cascade-archive` rend
≤ 2,0 ns/bloc, P5 ne s'ouvre pas, même si la marche binomiale passe 0,45 ns. Les
deux issues peuvent être vraies ensemble ; cette ligne dit laquelle gagne.

## 1. La comptabilité, figée ici

### 1.1 — Quatre comptabilités, et elles ne se comparent pas

| grandeur | ce qu'elle compte | e1v-séparé | archive |
|---|---|---|---|
| **bits/bloc, nue** | le champ seul, en-tête 10 b compris, **sans adressage** | **51,87** (pire 55) | 48,00 |
| **bits/bloc, ADRESSÉS** | + mot de base 32 b par groupe de 32, + arrondi au mot | **53,332** (ordre-fichier, warp-scan) | **49,000** |
| **b/poids NOYAU** | numérateur + queue f32 + échelles de ligne f32 ; dénominateur = **tous** les poids (§1.2) | **2,3709** | **2,1912** |
| **b/poids THESIS** | `bin/thesis` : « payload + **bases** + queue f32 + échelles de ligne f32 » (`CLAUDE.md` §G6) | **ABSENT** | **ABSENT** |

Les trois premières **[calculées]**, lues dans
[`e1v-ordre-fichier-4b-2026-08-13.txt`](../docs/mesures/e1v-ordre-fichier-4b-2026-08-13.txt)
(`:95`, `:114`, `:120`, `:129`, `:135`). 🚨 **Écrire « 53,332 bits/bloc » sans
« adressé, warp-scan, ordre-fichier » perd la moitié de l'information** : les
nombres décrivent le **même** flux.

🚨 **La quatrième ligne est une dette déclarée, et elle touche C0.** Le spec X4
écrit « ≤ **2,6 b/poids thesis** projeté »
(`docs/archive/spec-memoire-extreme-2026-08-12.md:185-188`) ; le binaire
identifie thesis ≡ noyau **en silence** (`E3_CRITERION = 2.6`,
`radixstudy.rs:283`, appliqué via `Shapes::kernel_bpw` `:813-816`) ; **aucun
document du dépôt ne prouve la coïncidence** — le numérateur thesis porte un
terme « bases » absent de `kernel_bpw`, et le dénominateur de `bin/thesis` n'est
écrit nulle part de vérifiable. **Donc P5 ne cite PAS le spec « mot pour mot »
sur l'unité** : il lit le 2,60 **dans la comptabilité noyau du §1.2** et déclare
la substitution **OUVERTE** (§7).

**Surcoût d'arrondi, par la seule identité exacte disponible** : 53,332 (FO/scan)
− 52,869 (CM/scan) = **0,463 bit/bloc** [calculé, deux termes lus à 3 décimales,
journal `:114`, ligne `radix2`], la colonne classe-majeur servant de **CONTRÔLE,
pas de verdict** (§1.3). ⚠️ Aucune décomposition ne part de la moyenne 51,87 : le
journal `:109-110` interdit d'en déduire une troisième décimale.

### 1.2 — Le b/poids noyau n'est PAS `bits_par_bloc / 24`

`Shapes::kernel_bpw` (`radixstudy.rs:813-816`) :
`(bits_par_bloc × blocs + (queue + lignes) × 32) / poids`. Sur le 4B [mesuré,
formes **lues dans le fichier**, épinglées `:794-801`, égalité exigée `:2520`] :
**3 633 315 840** poids · **16 957 440** de queue · **1 105 920** lignes ·
**150 681 600** blocs ; terme latéral f32 = **578 027 520 bits** [calculé],
identique pour tous les bras LLVQ.

⚠️ **Le dénominateur est TOUS les poids (3 633 315 840), pas 24 × blocs
(3 616 358 400).** Reproduction : `(53,332 × 150 681 600 + 578 027 520) /
3 633 315 840 = 2,370886` → **2,3709**. Contrôles : 112 bits → 4,80398
(`Planes14`, épinglé `:1933-1935`) ; 48 → 2,14976 ; 49 → 2,19123 ; 2,60 b/poids →
58,8565 bits/bloc.

### 1.3 — Le +32 bits par groupe, et la colonne interdite

`⌈Σ/32⌉·32 + 32 = ⌈(Σ+32)/32⌉·32`. Facturer `Σ + 32` sans arrondir, ou arrondir
sans le mot de base, est l'off-by-one qu'attrape le test du groupe partiel
(`radixstudy.rs:2027-2044` : 5 blocs de 48 bits → **256 + 32**, pas 240 + 32,
assertion `:2034`). Sur le 4B la distinction est inerte — **4 708 800 groupes en
ordre-fichier, 0 partiels** [mesuré, `:2554-2555`] ; un harnais E1v qui change la
découpe hérite du piège **sans que rien ne le montre**.

**La colonne classe-majeur est INTERDITE de citation dans tout verdict** :
optimiste par construction, contrôle de non-régression du log du 08-12. ⚠️ Et sa
justification usuelle est fausse comme absolu : `sweep_class_major` **ne ferme
pas** le groupe entre deux classes (`:913-914`, boucle `:924-934`), et l'en-tête
dit « almost always the *same* class » (`:201`), pas « homogène » — jusqu'à
**382 groupes** (383 classes − 1) sur 4 708 800 sont mixtes. Effet **≤ ~8e-5
bit/bloc** : le chiffre survit, c'est la **phrase** qui est fausse.

### 1.4 — La bande ]2,60 ; 3,09] porte un ⊘, et le triplet est nommé ici

`S_alt` est « a **candidate** threshold, not the project's… opposable only if the
product note of §5 keeps that triple, **and that note does not exist** »
(`radixstudy.rs:294-301`). 🚨 **Et la note produit n'est pas opposable** :
`docs/note-produit-2026-08-13.md` est **non suivi par git**, « **STATUT :
BROUILLON** … rien ici n'est opposable tant que les cases du §A ne sont pas
arbitrées » (`:3-4`). Trois défauts : (a) sa table de neuf cellules **ne se
reproduit pas depuis une convention unique** — la formule `:61-63` ne définit
**nulle part** la taille du cache KV, et *16k / KV f16 / marge 5* tombe sur la
frontière d'arrondi (2,2637 → 2,26 contre ≈2,267 → **2,27** imprimé) ; (b)
**l'unité des « 32 Go » n'est définie nulle part**, écart **+0,2758 b/poids**
[calculé : `(34 359 738 368 − 32 000 000 000) × 8 / 68,45e9`], **plus** que les
0,16 que le passage KV f16 → q8 achète dans sa propre table ; (c) **le pass d'E1v
n'est pas inconditionnel** — 2,3709 passe **6 des 9 cellules** (3,09 · 2,93 · 2,74
· 2,61 · 2,58 · 2,58) et **échoue les 3 autres** (2,27 · 2,26 · 1,63).

**Décision, pour lever une obligation autrement insatisfaisable.** La note `:71-73`
écrit que le 2,60 se relit « soit “8k, KV f16, 5 Go”, soit “32k, KV q8, 2 Go” ».
**Pour la durée de P5, le triplet retenu par défaut est (32 Go ; marge 5 Go ; KV
f16 ; 8k), cellule 2,58** — choisi **ici**, non arbitré ailleurs, retenu parce que
c'est le **plus strict** des deux, donc celui qui n'arrange pas ; 2,3709 le passe
avec 0,21 b/poids de marge. Révisé **sans discussion** dès que la note est commitée.
**Règle** : aucun verdict ne se lit contre `S_alt`, et le pass sur `S_spec` se cite
**toujours** avec ce triplet et la mention qu'il est choisi ici.

## 2. Le protocole, figé ici

### 2.1 — La forme de réouverture de la clause ≤ 24

**La clause du spec X4** (`spec-memoire-extreme-2026-08-12.md:185-188`) : « ≤ **2,6
b/poids thesis** projeté **ET** un décodeur à profondeur fixe **≤ 24 étapes** sans
état sériel inter-slot. Sinon, E3 est enterré avec le même soin que E2. »

**Conjonction que le binaire fait respecter par la machine** : `Admissible` exige
`shift_only && depth <= MAX_DEPTH`, `MAX_DEPTH = 24` (`radixstudy.rs:293`, match
`:551-557`) ; `depth: Some(96)` tombe en `DepthReopening` et **ne peut pas**
atteindre `OPEN_AT_SPEC` (`:1387`). Deux tests l'interdisent :
`the_verdict_needs_both_clauses_not_shift_only_alone` (`:2168`) et
`a_depth_reopening_never_prints_the_spec_admission_sentence` (`:2212`).
⚠️ **Ils existent parce que le binaire a publié le contraire** : son en-tête
(`:92-94`) écrit que jusqu'au 2026-08-13 il décidait sur `shift_only` seul et « a
publié "E3 est ouvert au sens de la spec" sur un point `depth: Some(96)` ».
**Les deux restent verts pendant tout P5.**

**Forme retenue : celle d'E2 → v2** (`spec-apres-awq-2026-08-10.md:78-100`) —
nommer l'ancien critère **périmé sans l'effacer**, écrire le neuf dans `proofs/`
**avant** de mesurer, prouver l'exactitude avant de chronométrer, appliquer le
critère **même quand il tue** (le lot C a refusé la v2 à 1,77× < 2,0×). La clause
≤ 24 était un **proxy de vitesse évalué sur le papier**, faute de décodeur
candidat ; P1 la remplace par **une mesure**.

🚨 **DÉROGATION NOMINATIVE ET RÉVOCABLE — la clause ne devient pas un obstacle
qu'il suffit de coder pour franchir.** (a) Remplacée **pour le seul point
`e1v-séparé`** : tout autre point du menu (`golay_tight`, `perslot`, une variante
future) exige **un nouveau pré-enregistrement**, et passer C1 ∧ C2 ∧ C3 ne l'en
exempte pas. (b) La réouverture est **PROVISOIRE** : **révoquée de plein droit,
sans nouvelle délibération**, si la mesure sur carte de P4 rend un rapport hors de
la fourchette que C3 supposait. (c) **« P1 mesure le décodeur » ne peut pas être le
critère neuf** : P5 ne s'ouvre **que si** P1 est déjà passé sous 0,45 ns, donc ce
serait un seuil qu'on est certain de passer. Le critère neuf est **interne à P5**
(§4).

### 2.2 — La preuve de bijection

**Sweep intégral des 150 681 600 blocs** du 4B scellé, harnais
`llvq-artifact/tests/e1c_format.rs`, sur le modèle de
`the_sealed_artifact_e1c_repack_is_exact` (`:186-291`).

- **L'étalon est `FastDecoder::decode`** (`llvq-search/src/fastdec.rs:321`), bloc
  à bloc, plus le gain — le motif d'`E1c14` (`e1c_format.rs:230-237`).
  ⚠️ **Contrat sur l'origine, vérifié dans le code** : `decode` rend
  **`Some([0; DIM])`** pour `idx == 0` (`fastdec.rs:322-324`) et `None`
  **seulement** hors boule (`:325-328`). Le `unwrap_or_else(|| { assert_eq!(idx,
  0, …); [0i32; DIM] })` du harnais (`e1c_format.rs:231-234`) est donc un
  **garde-fou qui panique sur un index hors boule**, pas un repli d'origine. La
  lecture inverse circule ; elle est fausse.
- ⚠️ **ABSENT : le motif exact n'existe pas côté 12x.** `E1c12` n'est comparé qu'à
  `Planes12xBlocks::decode_approx_block` (`e1c_format.rs:238-244`), le flux
  principal portant des **représentants échangés** sur les blocs à 5 niveaux.
  **À écrire**, ce n'est pas un acquis.
- **Un chunk DOIT être un nombre entier de groupes de 32** : `CHUNK = E1C_GROUP ×
  4096 = 131 072` blocs (`e1c_format.rs:203`, commentaire `:200-202`, « the one
  thing a chunked sweep could get wrong »).
- **Le sweep échoue quand le fichier manque.** `common::sealed_artifact_path()`
  (`llvq-artifact/tests/common/mod.rs:52-66`) panique en nommant le fichier ;
  `LLVQ_SEALED_ARTIFACT` **déplace** la recherche, ne la satisfait jamais. Tout
  test P5 hérite : `#[ignore]` inconditionnel **plus** échec nominatif — jamais
  `eprintln!("SKIP"); return;`.
- **Tout écart enterre E1v sans chronométrage** (règle du 08-11).

**Le bloc origine, DÉCIDÉ ICI plutôt que renvoyé au chantier.** Le 4B a **ZÉRO bloc
origine** (`radixstudy.rs:2517`) et `the_origin_tariff_is_wrong_for_fixed_width_variants`
(`:2260`) documente que le tarif est **déjà faux** pour les variantes à largeur fixe,
non réparé faute d'artefact qui l'exerce. **Décision : pour E1v, l'origine est le bloc
dont l'index est hors boule ; son enregistrement est l'en-tête seul, 10 bits, sans
champ de rang** — exactement ce que facture `WidthTable::build` (`:897-900`,
`(v.get)(&Widths::default()).max(HEADER_BITS)`, `HEADER_BITS = 10` à `:263`) et ce
qu'exige `e1v_is_never_narrower_than_the_class_rank` (`:1702`). Elle décode au vecteur
nul. **Le sweep ne l'exercera jamais** : le chemin origine se teste **séparément, sur
fixture synthétique**, comme `fixture_indices` (`e1c_format.rs:81-96`, qui pousse `0`
à `:88`). Écrire « prouvé sur 150,7 M blocs » sans cette réserve serait faux.

### 2.3 — Les deux tests qui portent la létalité de T2 — DEUX, pas un

`e1v_is_never_narrower_than_the_class_rank` (`radixstudy.rs:1660-1703`) est souvent
cité seul. **Il ne suffit pas** : sur sa **seconde** source — la boucle
`FastDecoder`, celle que le balayage lit (`:1686-1699`) — le plancher est
`floor = t.per_class[ci].exact`, soit `lg_ceil(produit des radices)`, **un modèle
comparé à lui-même**, et l'égalité exigée pour la sous-variante packée y est
**tautologique**. (Sur la première, `enumerate_classes(13)`, il compare bien à
`c.cardinality()`, `:1664` et `:1674`.) Ce qui rend ce modèle égal à la vraie
cardinalité est **un autre test** : `the_exact_column_is_the_class_cardinality`
(`:1599-1619`) — commentaire `:1594-1597` : « Everything that follows … is only as
good as this ». **Les DEUX restent verts pendant tout P5**, et le journal les nomme.

⚠️ `multiset_matches_the_class_cardinalities` (`:1918-1928`) ne boucle que sur
`cs.odd` (`:1920`) : le coset **pair** n'y est pas vérifié, et le citer comme la
garantie du produit de radices pair serait une erreur d'attribution. ⚠️ **Aucun de
ces tests ne prouve une bijection** : une largeur suffisante prouve seulement
qu'**aucune bijection ne peut exister en dessous**.

### 2.4 — Threads, machine et commande, figés D'AVANCE

🚨 **Le « 404 s contre 84 s » ne peut PAS servir de référence.** (a) **Aucun journal
dans `docs/mesures/`** : la seule source primaire est le corps du commit `3bae4fd`
(2026-08-09), **sans machine, sans threads, sans commande** ; `llvq-artifact/src/e1c.rs:24`
le reprend, ce n'est pas une seconde source. (b) **Le « (M3 Max, 16 threads) » de
`CLAUDE.md:389` n'est dans aucune source antérieure** — ajouté après coup au chiffre
de `:388`. (c) **Le code montre une asymétrie réelle** : `transcode_planes14` est
**séquentiel** (`llvq-artifact/src/runtime.rs:477-545`, aucun `thread::scope`, aucun
`chunks`), `transcode_planes12x` distribue sur `available_parallelism()` (`:1156`)
via `thread::scope` (`:1158`) — les deux chronos comparent donc **probablement** un
chemin threadé à un chemin mono-thread, et « probablement » n'est pas un protocole.

**Décision : on fige son propre modèle et on publie les deux bras dedans.**

| point | valeur figée |
|---|---|
| parallélisme | **un seul niveau**, jamais deux emboîtés |
| bras mono-thread | **N = 1** forcé dans le transcodeur, obligatoire, publié |
| bras multi-thread | **N imprimé** sur la ligne de résultat, jamais omis |
| ⚠️ `transcode_planes14` | **séquentiel par construction** (`runtime.rs:477-545`) : son bras N>1 **est** son bras N=1, publié comme tel. **C4 se lit UNIQUEMENT dans la colonne mono-thread** |
| bras chronométrés | `transcode_planes14`, `transcode_planes12x`, `transcode_e1v` — **même processus, ordre de dispatch fixe, tous les trois à chaque passe** |
| répétitions | **3 passes** ; **ratio formé PASSE PAR PASSE** ; plage des trois publiée (min · médiane · max) |
| **machine** | modèle, cœurs perf, charge concurrente déclarée — **imprimés sur la ligne de résultat** |
| **commande** | la ligne `cargo` exacte, profil compris — **imprimée en tête de journal** |
| fichier | le 4B scellé, entier ; absent ⇒ **échec**, pas skip |

⚠️ **La modification d'API requise est ADDITIVE, et c'est décidé ici.**
`transcode_planes12x` appelle `available_parallelism()` en dur (`runtime.rs:1156`) :
**aucun** moyen de forcer N = 1. On ajoute **`transcode_planes12x_with_threads(n)`** ;
`transcode_planes12x` devient un appel à `…(available_parallelism())` et **sa
signature ne bouge pas** — elle est appelée par le sweep E1c (`e1c_format.rs:226-227`)
et par le chemin modèle, et la changer ferait cesser d'être un contrôle de
non-régression le test même sur lequel P5 s'appuie. **Le sweep P5 est un NOUVEAU
test à côté de `the_sealed_artifact_e1c_repack_is_exact`, qui reste inchangé.**

🚨 **Le harnais existant sur-souscrit les threads EN CARRÉ** :
`the_sealed_artifact_e1c_repack_is_exact` lance `available_parallelism()` workers
(`e1c_format.rs:199`, `:215-217`) et **chaque** worker appelle
`transcode_planes12x` (`:226-227`), qui relance `available_parallelism()` threads.
**Donc les 459,16 s du run du 2026-08-12** (`docs/mesures/e1c-sweep-4b-2026-08-12.txt`,
total pour **3 tests**, part propre non isolée) **ne sont pas un chiffre à modèle
de threads défini.** Le harnais P5 appelle les transcodeurs avec `threads = 1`
depuis ses workers.

### 2.5 — Où vit le code, décidé d'avance

`llvq-artifact` dépend de `llvq-core`, `llvq-search`, `llvq-quant`, et d'**aucun
crate externe** (`llvq-artifact/Cargo.toml:9-12`).

- **La CNS et sa table de binomiaux vont dans `llvq-search`** — là où vivent déjà
  `FACT`, `multinomial`, `perm_rank`, `perm_unrank`, `unrank_fast` et les
  cardinalités par classe. Une table `C(n≤24, k≤12)` est un `const` : **aucune
  dépendance ajoutée**, `forbid(unsafe_code)` préservé.
- **L'étalon reste indépendant par construction** : `FastDecoder::decode` décode un
  **rang de permutation de multiensemble**, la CNS un **rang combinatoire**.
- ⚠️ **La décision d'export d'É0 est DÉJÀ appliquée** : le commit `94834f6`
  (2026-08-14) a rendu publics `CascadeClass`, `FastDecoder::cascade_class`
  (`fastdec.rs:311`) et `unrank_multiset` (`:167`), tenus par
  `the_public_cascade_table_is_complete`. `llvq-search` reste le bon endroit —
  `FACT`/`multinomial` restent `pub(crate)`, `perm_rank`/`perm_unrank` **privés** —
  mais l'argument « sinon il faudrait exporter » n'est plus vrai pour la cascade.
- Le sweep va dans `llvq-artifact/tests/`, seul endroit d'où l'on ouvre un `.llvq`.

## 3. V0 avant V1 — l'exactitude d'abord, sans exception

**Aucune seconde n'est chronométrée avant que la CNS soit prouvée.** Ordre
opposable :

1. **V0.a** — les trois tests de largeur restent verts :
   `the_exact_column_is_the_class_cardinality`,
   `e1v_is_never_narrower_than_the_class_rank` (§2.3) et
   `e1v_split_is_radix2_bit_for_bit` (`radixstudy.rs:1711-1733`), qui épingle
   l'assertion `depth > 24` et son message « une profondeur ≤ 24 déguiserait la
   réouverture de la clause du spec » (`:1729-1732`).
2. **V0.b** — la largeur est **RECALCULÉE depuis la CNS**, classe par classe (C1).
3. **V0.c** — bijection sur **fixture synthétique**, **bloc origine compris** et
   les deux bornes de chaque classe (motif `fixture_indices`).
4. **V0.d** — sweep intégral, 150 681 600 blocs, contre `fd.decode`, compte de blocs
   **imprimé** et `assert_eq!(n, 150_681_600)` — modèle `e1c_format.rs:268-291`
   (impression `:279-289`, complétude `:290`).
5. **V0.e** — **C3** : compteur de pas instrumenté, vérification zéro-division,
   invariance du compte par classe, **sur le code écrit** (§4).
6. **V1** — chronométrage du transcodage, et seulement alors, dans le modèle du §2.4.

**Tout écart en V0, C3 compris, interdit V1.**

⚠️ **La colle ordre-fichier n'est testée sur blocs réels que derrière un
`#[ignore]`** — `the_file_order_glue_runs_on_a_real_matrix` (`radixstudy.rs:2449-2450`)
exige l'archive. **Sur une machine sans le fichier, rien n'exerce sur données
réelles le chemin qui produit le 53,332 « qui fait foi ».** P5 imprime en tête de
journal que ce test a **tourné positivement**, pas qu'il est vert (sanction §7).

## 4. Les critères, posés avant la première ligne de code

**Le critère neuf qui remplace la clause ≤ 24 pour `e1v-séparé` est la conjonction
C1 ∧ C2 ∧ C3.**

| # | critère | vert | rouge |
|---|---|---|---|
| **C0** | **bits** — ≤ **2,60 b/poids noyau** (`S_spec` relu dans la comptabilité du §1.2, **pas** celle du spec ; triplet du §1.4 nommé à chaque citation) | déjà satisfait à **2,3709** | **C1 rouge ⇒ C0 est RETIRÉ** |
| **C1** | **largeur réalisée** — dérivée de la **table de binomiaux de la CNS seule**, classe par classe, égale `Widths::radix2` au bit près, en-tête de 10 b et **origine** compris ; et le flux FO/warp-scan reproduit **53,332 bits/bloc à \|Δ\| < 5e-3** ET **2,3709 b/poids noyau à \|Δ\| < 5e-5** | égalité exacte, 383 classes **+ origine**, dans les deux tolérances | tout écart ⇒ **2,3709 RETIRÉ**, E1v redevient une largeur sans décodeur |
| **C2** | **bijection** — sweep intégral, 150 681 600 blocs, contre `fd.decode`, **plus la fixture origine** | zéro écart des deux côtés | tout écart, **fixture origine comprise** ⇒ **enterré**, sans V1 |
| **C3** | **forme du décodeur**, trois clauses machine-vérifiables (ci-dessous) | les trois | l'une des trois ⇒ **réouverture REFUSÉE**, V1 **non lancé** |
| **C4** | **transcodage** — `T(E1v) ≤ 2 × T(Planes14)`, **bras mono-thread du même run** | **médiane des 3 ratios ≤ 2,0** | > 2,0 ⇒ **non adopté** pour le chemin servi ; publié comme point de courbe |

**C0 n'est pas un critère de P5** : déjà satisfait, le compter comme un pass serait
le seuil qu'on est sûr de passer. Il **ne compte dans aucun décompte de verts**. Il
figure pour une seule raison : **C1 peut le retirer.**

**Les tolérances de C1 sont celles déjà en vigueur** dans
`the_sealed_4b_reproduces_the_published_class_major_verdict` (`5e-3` sur les
moyennes en bits/bloc, `radixstudy.rs:2536`, `:2538` ; `5e-5` sur les b/poids,
`:2540`). **Et l'égalité doit être un résultat de deux chemins indépendants, jamais
une lecture** : ni la CNS ni son test de largeur n'appellent `Widths`,
`widths_of_even`, `widths_of_odd` ni quoi que ce soit de `radixstudy` — un test
échoue si l'un de ces symboles apparaît dans le module de la CNS. Sans cette
clause, C1 serait la tautologie que le §2.3 dénonce.

**C3, en trois clauses, aucune auto-attribuée.**

1. **« pas dépendant » ≡ une itération de la boucle EXTERNE sur les slots**, sommée
   sur tous les étages du décodage d'un bloc. Le compte est établi par un
   **COMPTEUR INSTRUMENTÉ exécuté sur les 150 681 600 blocs du sweep**, dont le
   **MAX est imprimé sur la ligne de résultat** — pas une lecture de source.
   **Vert : max ≤ 96**, le nombre que la variante déclare elle-même
   (`radixstudy.rs:626`, en-tête `:73-76`). Le critère transforme une déclaration
   gratuite en engagement opposable ; il échoue si un étage s'ajoute ou si une
   classe en demande davantage. ⚠️ **Le compte de la boucle INTERNE (évaluations de
   candidat) est publié à côté et n'est PAS un critère** : la déclaration `:73-76`
   ne le borne pas, et lui inventer un seuil après coup serait déplacer le poteau
   (divulgation §8).
2. **« zéro division » ≡ zéro `/` ou `%` par une valeur non constante dans la
   fonction de décodage.** Vérifié **deux fois** : un test qui **échoue** si le
   fichier source de la CNS en contient (grep épinglé sur le module), **plus** la
   constatation sur l'assembleur release. Commande fixée d'avance, hôte
   `aarch64-apple-darwin` (M3 Max) : `cargo rustc --release -p llvq-search --lib --
   --emit asm -C debuginfo=0`, puis recherche de `sdiv`/`udiv` dans le symbole de la
   CNS du `.s` produit sous `target/release/deps/`. ⚠️ **Limite** : cet assembleur
   est celui de l'**hôte CPU**, pas de la cible CUDA — il borne la référence Rust,
   rien d'autre ; seul P4 dira quelque chose du noyau.
3. **« zéro branche dépendante des données » ≡ le compteur de la clause 1 rend la
   MÊME valeur pour tous les blocs d'une même classe**, sur le sweep entier.

**Pourquoi « zéro division », et de quelles divisions on parle.** C'est la propriété
qui rend la réouverture argumentable. Le décodeur d'archive en contient à **deux
endroits distincts** : **une division par candidat** dans `unrank_fast`
(`fastdec.rs:183`, doc `:171` et `:154`), **plus** le `local % fc.m_arr` /
`local / fc.m_arr` du coset impair dans `decode` lui-même (`:335-336`), hors
`unrank_fast`. La CNS ne promet de supprimer que **les premières**. Une CNS qui les
garderait ne gagnerait rien sur un décodeur qui existe **et qui est plus petit**
(2,1912 contre 2,3709).

**C4, contenu et statistique.** **`T` ≡ le temps mural, du tableau `(indices,
gains)` d'une matrice au tampon d'octets final du layout, allocation comprise, I/O
disque exclue** — donc l'écriture du flux adressé en groupes de 32 avec mot de base
et warp-scan est **dedans** pour E1v, et toute lecture du `.llvq` **dehors**. **Le
ratio se forme PASSE PAR PASSE** ; C4 est vert si la **médiane des trois ratios**
est ≤ 2,0, et la **plage des trois** est publiée. Un ratio formé comme quotient de
deux minima n'a **aucun verdict** (§7).

🚨 **Le ×2 de C4 est LE MAILLON FAIBLE.** Le facteur 2 est un **jugement
d'ingénierie**, pas une grandeur dérivée ; l'ancre — `transcode_planes14`
mono-thread sur le 4B entier — **n'a jamais été mesurée dans un modèle de threads
défini** ; et contrairement au précédent E2, où « 2,0× » était *la vitesse d'un
layout déjà servi*, **rien ici n'est un seuil déjà atteint par un chemin servi**.
C4 est pour cela un critère d'**adoption**, jamais d'**enterrement**.

**Pas un critère, publié quand même** : la projection du coût de transcodage à 70B,
étiquetée **[calculé, extrapolation]**, sans verdict — extrapoler d'un modèle à
l'autre est le raccourci que ce dossier refuse (`CLAUDE.md` : 4,77·10⁻⁵ cœur-s/poids
à 8B contre 6,36·10⁻⁵ à 32B).

## 5. La prédiction, et ce qui ne la fonde pas

**Aucune fourchette de vitesse de décodage n'est prédite. Aucune.** **Aucune
fourchette de temps de transcodage non plus** : trois repères existent, et **aucun
ne borne C4**.

| repère | ce qu'il vaut |
|---|---|
| **404 s / 84 s** (commit `3bae4fd`, repris `e1c.rs:24`, `CLAUDE.md:388`) | sans machine, sans threads, sans commande — **inutilisable** |
| **459,16 s** (`docs/mesures/e1c-sweep-4b-2026-08-12.txt`) | threads indéfinis, sur-souscription en carré, **et** total de 3 tests — **doublement inutilisable** |
| **~37 s mono-cœur** (`runtime.rs:549-550` : « one fast decode per block (243 ns): ~37 s single-core for a 4B ») | **[estimé, aucun banc cité]** — le seul dont la découpe corresponde au dénominateur de C4, et il affaiblit la clause « le seul repère disponible » qu'écrivait la version antérieure |

**Ce qu'on peut dire, et c'est tout** : le poste de coût connu du transcodage
`Planes12x` est la recherche réseau par bloc à 5 niveaux — **5 096 688 blocs,
3,3824 %** [mesuré, `docs/mesures/shell-distribution-4b-2026-08-10.txt:394`] à
**~0,7 ms d'un cœur** chacun [**estimé**, `llvq-llm/src/fused.rs:977` et
`runtime.rs:1132-1133`, **aucun banc cité**] : ≈ 3 568 s·cœur [calculé],
**compatible en ordre de grandeur** avec l'écart 404 − 84, **pas une reproduction**.
La CNS d'E1v ne fait pas de recherche réseau. 🚨 **Ce qui NE fonde PAS le ×2 de C4** :
aucun compte d'opérations, aucun profil (le profileur n'a jamais servi sur ce
projet), aucune mesure de son propre dénominateur ; il est **conservateur dans le
mauvais sens** — il laisserait passer un chemin lent si `Planes14` mono-thread se
révélait lui-même lent.

**Si la CNS rend une largeur STRICTEMENT INFÉRIEURE à `radix2`, chercher l'erreur
avant d'en faire un titre.** `e1v-séparé` **est** `radix2` au bit près — résultat
**déjà publié le 2026-08-12**, épinglé par `e1v_split_is_radix2_bit_for_bit`. Par
ordre de vraisemblance : en-tête oublié, classe non couverte, origine mal tarifée —
et seulement en dernier une découverte ; une largeur sous `⌈log₂|classe|⌉` serait
une **bijection impossible**, donc un bug de comptage. ⚠️ **Corollaire à écrire dans
toute publication de P5** : ce que P5 apporte n'est **pas** la largeur d'E1v, c'est
le **décodeur** supposé par-dessus.

## 6. Les issues, et ce que chacune fait au dossier

| issue | conséquence, décidée d'avance |
|---|---|
| **P1 : `cascade-archive` ≤ 2,0 ns/bloc** | **E1v mort-né** (P1 §4.3, É1). L'archive est plus petite (2,1912 contre 2,3709), son décodeur existe. **P5 ne s'ouvre pas — cette issue PRIME sur la suivante** |
| **P1 : marche binomiale > 0,45 ns** | **P5 ne s'ouvre pas**, même si la cascade uniformisée passe le gate CUDA (§0bis) |
| **P1 : « la cascade passe la tolérance capacity-first »** | **NE FERME RIEN.** Cette tolérance n'est chiffrée dans aucun document arbitré (§0bis) : l'issue est inopérante tant qu'un nombre n'est pas posé ailleurs |
| **C1 rouge** | **2,3709 RETIRÉ**, **C0 RETIRÉ**, journal du 08-12 amendé ; E1v redevient une largeur sans décodeur, et la table des variantes est corrigée avant toute autre publication |
| **C2 rouge**, fixture origine comprise | **E1v enterré**, sans V1, avec le même soin qu'E2 et E3 — et on le dit |
| **C3 rouge** (max > 96 pas, ou une division, ou un compte non constant par classe) | **réouverture REFUSÉE** ; E1v reste `DepthReopening` non accordée ; **V1 n'est PAS lancé**, aucun chiffre de transcodage publié ; **le 2,3709 survit comme largeur sans chemin d'exécution** (formulation de l'issue P1 « ]0,45 ; 1,5] ns ») |
| **C1 ∧ C2 ∧ C3 verts, C4 vert** | **réouverture accordée pour le seul point `e1v-séparé`**, **provisoire et révocable** (§2.1) ; E1v devient candidat au portage CUDA, soumis au go de dépense, sans aucun chiffre de vitesse publié d'ici là |
| **C1 ∧ C2 ∧ C3 verts, C4 rouge** | **non adopté** pour le chemin servi ; publié comme point de la courbe taux↔coût de transcodage, avec sa provenance |
| **l'un des quatre tests de garde tombé** (`the_verdict_needs_both_clauses_not_shift_only_alone`, `a_depth_reopening_never_prints_the_spec_admission_sentence`, `the_exact_column_is_the_class_cardinality`, `e1v_is_never_narrower_than_the_class_rank`) | **aucun verdict n'est rendu.** On restaure le test, on relance, on consigne l'entorse au §7bis |

**Les quatre tests de garde sont exécutés et leur sortie horodatée DANS LE MÊME
journal que C1 et C2.** ⚠️ Ils ne touchent aucun fichier scellé : « les quatre sont
verts » ne prouve rien sur le chemin qui produit le 53,332 — d'où l'obligation du §7
sur les deux tests qui, eux, lisent l'archive.

**Aucune issue ne change quoi que ce soit à l'archive**, qui reste le concurrent :
2,1912 b/poids noyau, décodeur existant et prouvé, ~509 ops sérielles
(`radixstudy.rs:70-72`). **E1v est plus gros.** L'arbitrage n'est pas la place, c'est
la vitesse — et P5 n'en mesure aucune.

## 7. Ce qui invaliderait ce pré-enregistrement

- **s'il n'est pas commité dans `proofs/` avant la première ligne de code E1v** :
  pas d'antériorité, donc aucun critère opposable ;
- **si P5 est ouvert alors que `cascade-archive` a rendu ≤ 2,0 ns/bloc** (§0bis,
  kill amont prioritaire), ou **sans que la `marche-binomiale` soit passée sous
  0,45 ns** — y compris sur un « le banc est vert » visant le gate CUDA de P1 §4.2 ;
- **si une décision d'ouvrir ou de fermer P5 est prise sur « la tolérance
  capacity-first »** : elle n'est chiffrée dans aucun document arbitré, donc une
  telle décision est un jugement libre a posteriori ;
- **si le 4B scellé `~/llvq-q4b.llvq` est absent** : le sweep **échoue**, il ne saute
  pas, et aucun chiffre du run n'est publiable ;
- **si un chunk du sweep n'est pas un nombre entier de groupes de 32** : la
  transposition est fausse **sans que rien ne le montre** (§1.3, §2.2) ;
- **si le journal ne porte pas la sortie positive de
  `the_file_order_glue_runs_on_a_real_matrix` ET de
  `the_sealed_4b_reproduces_the_published_class_major_verdict`** : aucun chiffre de
  largeur du run n'est publiable ;
- **si le journal ne porte pas la machine ET la commande** (§2.4) : aucun chiffre de
  transcodage n'est publiable — le défaut même reproché au 404 s ;
- **si le chronométrage a deux niveaux de threads emboîtés**, **si C4 est lu dans un
  bras où l'un des deux transcodeurs est threadé et l'autre non**, ou **si le ratio
  est formé comme quotient de deux minima** : il n'y a **pas de verdict C4** ;
- **si un chiffre de transcodage est comparé à « 84 s » ou « 404 s »** ;
- **si un verdict cite la colonne classe-majeur** (§1.3) ou **`S_alt` (3,09) comme un
  pass** (§1.4) ;
- **si le pass sur `S_spec` est publié sans nommer le triplet du §1.4** : 2,3709
  échoue 3 des 9 cellules de la table de la note ;
- **si l'un des quatre tests de garde du §6 est modifié, désactivé ou contourné**.

**Ce qui reste OUVERT, et qu'aucune ligne de ce document ne ferme :**

1. **L'équivalence thesis ≡ noyau** (§1.1). Le spec dit « thesis », P5 lit
   « noyau » ; aucun document ne prouve l'égalité, le numérateur thesis porte un
   terme « bases » absent de `kernel_bpw`, et P5 ne l'invente pas. Tout verdict C0
   se lit sous cette réserve.
2. **Le budget de temps du chantier.** Aucun stop-loss n'est posé, faute d'un nombre
   arbitré ; la mention « ~1 semaine » de la version antérieure était une estimation
   sans conséquence et elle est **retirée**. C'est la seule ressource que P5 dépense
   et la seule qu'il ne borne pas. **À arbitrer par l'opérateur avant la première
   ligne de code.**
3. **Le motif de comparaison exact du côté `E1c12`** (§2.2), qui n'existe pas dans le
   harnais et reste à écrire.

## 7bis. Écarts au protocole — journal, tenu à chaud

*(Chaque entorse s'écrit ici le jour où elle est commise, avec sa raison et son coût
— la règle du 08-10.)*

### É0 — 2026-08-15, avant la première ligne de code E1v : **C1 et C3 ne peuvent pas être vrais ensemble**

> 🚨 **PROPOSITION, non acquise.** Elle attend l'arbitrage de l'opérateur, et le
> tampon doit venir après lui. Aucune ligne de CNS n'est écrite, aucune seconde
> n'est chronométrée.

**Ce qui s'est passé.** P1 a rendu son verdict le 2026-08-15 et P5 s'est ouvert
(marche binomiale 0,3101 ns ≤ 0,45). Avant d'écrire la CNS, une vérification de
routine — *que dit le dépôt de la forme exacte d'`e1v-séparé` ?* — a trouvé que
**deux documents le décrivent différemment, et que la différence est
structurelle**.

| source | ce qu'elle dit qu'est `e1v-séparé` |
|---|---|
| `radixstudy.rs:619-621` + `e1v_split_is_radix2_bit_for_bit` + §8 de ce document | **`radix2` au bit près** : un `⌈log₂⌉ ` par **étage de composition** — 5 pour le coset pair (mot de Golay, arrangement support, arrangement hors-support, signes de mot, signes libres) |
| `llvq-search/src/rankdec.rs:32-36` | des **champs de rang PAR GENRE**, « *and it is why it costs 53,332 bits/block* » |

**Ce sont deux objets différents, et ce n'est pas une nuance de rédaction.**
Mesuré sur les 383 classes réelles, puis pondéré par les 150 681 600 blocs du 4B
scellé [mesuré, sonde jetable, modèle validé ci-dessous] :

| granularité | bits/bloc nus, ordre fichier | classes où les deux diffèrent |
|---|---|---|
| par **étage** (`Widths::radix2`) | **51,8689** | — |
| par **genre** (ce qu'une marche binomiale lit) | **52,2206** | **101 / 383**, jusqu'à **+2 bits** |
| écart | **+0,3516 bit/bloc** | jamais négatif |

**Le modèle est validé avant d'être cru** : sa colonne par étage rend
**51,8689** là où le §8 de ce document publie **51,87**. C'est la même quantité,
et c'est ce qui autorise à croire la seconde colonne, qui n'avait jamais été
calculée.

**Pourquoi c'est structurel et non un détail d'implémentation.** Un champ par
étage porte le rang d'arrangement **empaqueté** ; l'extraire genre par genre est
un pelage à radix mixte, donc **une division par le produit des radices
suivantes**. Le seul moyen de peler par décalage est que chaque radice de genre
soit **une puissance de deux**, c'est-à-dire exactement d'arrondir **par genre** —
et c'est la colonne large. Donc :

> **largeur `radix2` ⟺ pelage empaqueté ⟺ division.**
> **zéro division ⟺ champs par genre ⟺ +0,3516 bit/bloc.**

**Conséquence sur les critères tels qu'ils sont écrits.** C1 exige que la largeur
dérivée de la CNS égale `Widths::radix2` **au bit près** ; C3.2 exige **zéro
division** dans la fonction de décodage. **Aucun objet ne satisfait les deux.**
Un décodeur qui passe C1 échoue C3, un décodeur qui passe C3 échoue C1 — et C1
rouge **retire le 2,3709** qui est l'argument d'ouverture de toute la ligne.

**Ce que ça ne change pas, et il faut le dire tout de suite** : le **C0 tient
dans les deux cas**. La variante décodable pèserait ≈ **53,68 bits/bloc** adressés
FO et ≈ **2,386 b/poids noyau** [calculé, l'écart appliqué au 53,332 publié ;
l'adressage warp-scan n'est pas rigoureusement additif, donc ce report est un
ordre de grandeur, pas une mesure], contre un critère de **2,60**. E1v survit sur
les bits. Il reste **plus gros que l'archive**, qui pèse 2,1912 et dont le
décodeur existe.

**Ce qui est proposé, et ce qui ne l'est pas.** Proposé : que C1 cesse de
demander l'égalité avec `Widths::radix2` et demande l'égalité avec une **référence
par genre**, recalculée depuis la table de binomiaux de la CNS seule — la clause
d'indépendance du §4 est *renforcée*, pas relâchée, puisque `radix2` cesse d'être
la cible. Et que le **2,3709 soit remplacé par le chiffre par genre partout où il
sert d'argument d'ouverture**, le 2,3709 restant publié comme la largeur d'une
variante **dont le décodeur divise**.

⚠️ **Ce n'est pas un déplacement de poteau, et voici pourquoi on peut le
vérifier** : l'amendement rend E1v **plus gros**, jamais plus petit. Il ne fait
passer aucun critère qui échouait ; il constate qu'un critère était
**contradictoire avec un autre**, et il choisit lequel des deux décrit l'objet
que P1 vient de mesurer — la marche binomiale, qui ne divise pas. Le sens de
l'erreur est le seul garde-fou disponible ici, et il est du bon côté.

🕳️ **Et la leçon, qui est celle du dossier** : `rankdec.rs` attribuait 53,332 à
un format par genre alors que ce nombre est celui du format par étage. Une phrase
juste sur la *forme* (les champs séparés évitent la division) et fausse sur le
*chiffre* (ce n'est pas ce que coûte 53,332), écrite le jour où le décodeur a été
codé, et jamais confrontée à la table de largeurs qui vit dans un autre crate.
**La largeur par genre n'avait jamais été calculée** — ni le 08-12, ni le 08-13,
ni en écrivant ce pré-enregistrement.

## 8. Ce qui est connu à la signature — divulgation datée

Tous ces chiffres datent du 2026-08-12 ou du 2026-08-13 ; les critères C1–C4 datent
d'aujourd'hui et portent sur des objets qui n'existent pas.

- **`e1v-séparé` est vert en largeur** : **51,87** bits/bloc nus (pire 55), **53,332**
  adressés FO/warp-scan, **57,000** FO/grp32-max, soit **2,3709** et **2,5230**
  b/poids noyau [calculés, journal `:95`, `:120`, `:135`]. **Ce n'est pas un résultat
  neuf** : `e1v-séparé` **est** `radix2` au bit près, donc cette largeur était
  imprimée dès le 08-12 sous une autre étiquette. **`e1v-packé` rend 52,975 → 2,3561**
  [`:119`, `:134`] et il est **hors décision** : pas shift-only.
- **Le rang exact de classe moyen est 41,50 bits/bloc** [calculé, épinglé
  `radixstudy.rs:2542-2549`] : le point *dans* sa classe coûte déjà 41,50 des 47 bits
  d'index — le fait arithmétique qui a enterré E3.
- **L'archive fait 2,1912 b/poids noyau** (48,00 nue, **49,000** adressée — le mot de
  base vaut exactement 1 bit/bloc), décodeur existant et prouvé, **plus petite qu'E1v**.
- **Une marche binomiale sur 24 slots à ≤ 5 types peut atteindre 120 évaluations de
  candidat** là où `:73-76` annonce « 48–96 dependent steps » en boucle externe. Les
  deux comptes sont connus **avant** l'écriture de C3 ; c'est pourquoi C3 fixe la
  définition du « pas » et publie l'autre sans le borner.
- **La dérive de largeur du 4B au 8B est de +0,001 bit/bloc** (53,332 → 53,333) contre
  une règle de kill à +2 [calculé, `docs/mesures/e1v-8b-2026-08-13.txt`], comparaison
  faite **en bits/bloc**, jamais en inversant une colonne b/poids.
- **Le sweep E1c a tourné le 2026-08-12** : 150 681 600 blocs, `Planes14` 4,6667 →
  `E1c14` **4,4167**, `Planes12x` 4,2029 → `E1c12` **3,6196** [mesuré, comptes
  d'octets construits]. ⚠️ **Exceptions comprises** : `e1c.rs:15-18` tarife les mêmes
  flux à **4,0000** et **3,4167** en stride pur ; l'écart de 0,2029 b/poids est la
  table d'exceptions.
- **Le 4B publié a ZÉRO bloc origine et 4 708 800 groupes dont 0 partiels** [mesuré,
  `radixstudy.rs:2517`, `:2554-2555`] — les deux raisons pour lesquelles le sweep **ne
  peut pas** tout prouver (§1.3, §2.2).
- **La première émission du journal du 08-13 n'est pas dans git** :
  `git log --oneline -- docs/mesures/e1v-ordre-fichier-4b-2026-08-13.txt` ne rend
  qu'**un** commit (`3879cde`). Sa phrase « les 87 valeurs numériques ne bougent pas »
  entre les deux émissions est une **déclaration d'opérateur**, pas un fait auditable.
- **L'archive scellée est sur cette machine** : `~/llvq-q4b.llvq`, **980 790 202
  octets** [vérifié] — les preuves que P5 réclame sont exécutables **pour 0 $**.
- **Aucune milliseconde, aucun compte de registres, aucun profil n'existe pour la
  marche binomiale**, sur aucun matériel. La profondeur « 48–96 » est une
  **déclaration**, et C3 existe pour la rendre opposable.
