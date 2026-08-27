# Programme de clôture de l'exploration LLVQ — 2026-08-27

> **Ce document ordonne**, il n'invente pas. Les items vivent dans
> [`BACKLOG.md`](BACKLOG.md) et dans
> [`plan-variance-2026-08-26.md`](plan-variance-2026-08-26.md) ; ce qu'on ajoute
> ici, c'est **l'ordre, les gates, et la liste de ce qu'on n'explorera pas** —
> avec la raison, pour qu'aucune session aval ne la reprenne comme piste vierge.

## 0. Ce que « refermer proprement » veut dire, et ce que ça ne veut pas dire

Le pari **produit** est clos par arithmétique et rien ici ne le rouvre :

| | b/poids VRAM | plafond = 16 ÷ b | mesuré |
|---|---|---|---|
| `Planes14` servi | 4,804 | **3,331×** | 2,15× |
| **AWQ 4 bits, même processus** | — | — | **3,38×** |
| QTIP | 2,000 | 8,00× | 4,89× |

**Un noyau parfait sur notre meilleur layout servi arrive sous l'AWQ réellement
mesuré.** Ce plafond est *calculé*, invariant en k, et il ne dépend d'aucune
qualité d'implémentation.

Ce qui reste ouvert n'est pas le produit, c'est la **force de l'énoncé de
fermeture**. Le manuscrit affirme *« cet espace de conception se referme »* ; un
énoncé de fermeture ne vaut que ce que vaut l'exploration derrière. Aujourd'hui
elle repose sur **un** examen (MMLU), **une** perplexité, **un** corpus de
calibration, **une** carte principale et **un** régime de batch.

> 🚨 **Règle de composition n°1 — aucun lot ne se justifie par « ça pourrait
> renverser le verdict ».** Un lot se justifie par un **chiffre publiable sur
> une limite**, ou par une question ouverte du papier qu'il ferme. Un lot dont
> les deux issues mènent à la même phrase est **écarté**, pas réordonné (§4).

> 🚨 **Règle n°2 — chaque lot porte un gate écrit, tamponné et ancré AVANT la
> mesure.** Sans quoi on publie l'histoire qu'on préfère.

## 1. Les trois axes demandés, et où ils tombent

| axe | lots | coût |
|---|---|---|
| **(a) qualité et calibration poussées plus loin** | 1, 2, 3, 6, 8, 9 | 0 → 2 $, puis ~95 $ sur go séparé |
| **(b) évaluation au-delà de MMLU** | 4 | 0,8–1,0 $ |
| **(c) batch > 1** | 7 | 0,8–1,0 $, pire cas 2,70 $ |
| *(transverse — la thèse du manuscrit)* | 5 | ~1,1 $ |

**Total hors 4B : 9 à 13 $**, dont ~30 h de Mac à 0 $. C'est **moins de 15 % de
ce que le projet a déjà dépensé** (87,36 $, *mesuré* sur `data/jobs.csv`).

## 2. L'ordre, et pourquoi

### Bloc gratuit — il est la barre de tout le reste

**L1 — Solder les canaux de rétention.** *0 $, ou 0,49–0,55 $ si relance. ~1 h.*

🚨 **Le job `6a8de89b984507d9db4e4664` est EN VOL depuis le 2026-08-25 au soir**
(BACKLOG §« en cours », ligne 1b) : **aucun journal, aucune ligne au registre**,
dont la dernière est le 08-24. Sa sortie n'a jamais été récoltée. La règle du §7
de CLAUDE.md s'applique **avant tout devis** — `hf jobs inspect`, puis
`hf jobs logs`, puis `hf buckets ls`. Le dépôt a déjà payé quatre fois pour
avoir chiffré un rejeu contre une absence qu'il n'avait pas vérifiée.

Ce lot produit **`s`** — l'écart-type du MMLU micro entre tirages de
calibration, **à la taille publiée**, jamais mesuré. F5 n'avait chronométré que
la perplexité. `s` est la barre de **toute** expérience de l'axe (a) qui
recalibre : D3, les bits de gain, la rotation de sortie, l'étage 3 de la
variance. Sans `s`, aucune d'elles n'est lisible.

Trois actions non-mesure y sont repliées, toutes impossibles depuis la boîte de
session (403 du proxy sur les calendriers) :
`ots stamp proofs/preregistration-variance-calibration-2026-08-26.md` — que le
§3 du protocole exige avant la première milliseconde — et
`ots upgrade proofs/*.ots` pour les quatre tampons du 08-25 encore en attente ;
plus l'inventaire du bucket (69 fichiers, 46,7 Go, jamais fait, BACKLOG §2.8).

**L2 — Gate A, le déterminisme.** *0 $, ~1 h de Mac.*

Étage 0 du protocole de variance. Deux runs identiques doivent rendre la même
ppl **à 1e-9 relatif**. Le doute est nommé d'avance : la boucle GPTQ est prouvée
déterministe (`parallel_matches_serial_exactly`), mais **l'ordre de réduction
d'un matmul Metal n'a jamais été contrôlé ici**.

🔎 Et ce gate décide plus que la grille : si le pipeline n'est pas déterministe,
**aucun A/B qui recalibre n'est lisible, à aucune taille** — les lots 8 et 9
tombent avec lui.

**L3 — Grille 0.6B, étapes 1-2.** *0 $, ~6 h de Mac.*

Les 16 premiers runs sur 68 : `σ_diff` et le contraste de famille (Leech contre
scalaire). ⚠️ **Réserve à écrire d'avance** : à deux volumes la pente est un
**signe**, pas un β — l'IC95 qui doit exclure 0 exige les six volumes, donc L6.

### Deux lots à ~1 $, chacun ferme une question ouverte du papier

Ils sont **interchangeables et parallélisables** : l'un est du Rust
d'évaluation, l'autre du CUDA de banc. Aucun fichier commun.

**L4 — Le contraste témoins/sondes : l'axe (b).** *0,8–1,0 $ au 4B. ~700 lignes
dont ~250 de tests (estimé).*

Le dossier affirme depuis le 2026-08-02 que *« le 2 bits abîme le raisonnement
plus que la restitution »*. **Cette phrase n'a jamais été mesurée par un
instrument qui teste le raisonnement** — elle est inférée d'un profil par
matière MMLU. Le design : des **témoins** de restitution contre des **sondes**
de raisonnement, verdict = ratio des déficits, IC bootstrap apparié stratifié.

🚨 **C'est le meilleur rapport décision/dollar du programme, et le seul design
du dossier qui puisse RÉFUTER au lieu de confirmer.** Si le déficit est
**uniforme**, la phrase tombe — et avec elle la motivation du suspect « corpus
curé raisonnement » qui justifie D3. **Un lot à 1 $ peut rendre inutile un lot à
24 $.**

Quatre clauses au gate, à tamponner avant la première ligne de code :
1. **Contrôle de protocole** — le bras f16 doit tomber à ±2 pp du chiffre publié
   sur chaque banc ; sinon c'est le **protocole** qu'on corrige, pas le modèle.
   C'est la discipline qui a validé notre MMLU à 0,22 pp du papier.
2. **Puissance calculée d'avance** — SE du Δ apparié ≈ 1,44·√(2280/n) (*mesuré*,
   `mmlupair-4b-8b-2026-08-13.txt`), soit ≈ 2,0 pp au census ARC-C de 1 172. Un
   déficit de 2 pp est déclaré **non résolu avant** la mesure, et pas discuté
   après. C'est la parade au motif du `p = 0,40` du palier 8B→14B.
3. **Le verdict** — ratio sondes ÷ témoins, IC excluant 1.
4. 🚨 **`--no-fpc` obligatoire sur tout banc scoré en CENSUS.** Vérifié dans le
   code : `(1 − n/population).max(0).sqrt()` vaut **exactement 0** quand n =
   population (`mmlupair.rs:483-494`), donc l'IC apparié vaut `[Δ ; Δ]` —
   **largeur nulle, lue comme précision infinie**. ARC-C, HellaSwag et PIQA sont
   des census.

⚠️ Ce lot **ne remplit pas** la colonne CSR du papier source : sa composition
n'est transcrite nulle part et le PDF est chez l'opérateur. Produire « notre
CSR » sans cette relecture fabriquerait un nombre comparable à rien.

**L5 — Contrefactuel LUT (BACKLOG §4.4).** *~1,1 $, ~2 jours de code.*

La thèse centrale du manuscrit — *« c'est la TAILLE du codebook qui impose le
dépliage »* — repose aujourd'hui sur **une arithmétique de cardinalité et sur
aucune mesure**. Un bras à codebook tabulable, dans le même processus, la
teste : si ses Go/s tombent dans [0,85 ; 1,15] × ceux de `Planes14`, la
tabulabilité achète des **octets** et pas de la vitesse, et le mécanisme est
mesuré. Hors bande, la thèse est **à qualifier**.

### Puis ce qui coûte du code ou des dollars

**L6 — Grille 0.6B étapes 3-4 + granularité + invariance de backend.**
*0 $ (Mac) + ~4 $ (contrôle CUDA). ~23 h + ~2 h de carte.*

Achève la grille et pose **Gate B**, qui est la porte du 4B.

⚠️ **Motif du contrôle CUDA, contre-intuitif donc écrit d'avance** : ce n'est
**pas** pour aller plus vite. 84 % du run est l'encodeur, qui est CPU ; porter
la grille entière sur CUDA coûterait **~48 $** (*calculé* : 28 h × 1,77 $/h
*mesuré*) pour un résultat que Metal rend à 0 $. Les 6 runs CUDA testent
**l'invariance**, rien d'autre.

**L7 — Famille k, conforme au préreg P4 : l'axe (c).** *0,8–1,0 $, pire cas
2,70 $. ~600 lignes dont ~230 de noyaux, 4-6 jours.*

✅ **Le protocole existe déjà et il est détaillé** : `proofs/preregistration-p4-2026-08-14.md`
§2.6-2.12 pose la forme (accumulateurs en registres, grille inchangée),
`TILE_BLOCKS_K = 32`, une seule unité NVRTC et un seul sha256, k ∈ {1,2,4,8}, et
les seuils K1/K2/K3. Les **noms de bras sont déjà enregistrés** dans
`llvq-cuda/src/arms.rs:59-63` (`mvkf16`, `planes14k`, `planes12xk`,
`golay70v2k`) avec le commentaire « à écrire ». Le témoin cuBLAS est **déjà un
appel GEMM**, seule la constante `1` le borne (`planesbench.rs:2100`).

Il vient **tard malgré l'insistance sur l'axe (c)**, pour trois raisons :
- c'est le lot le plus cher en jours d'ingénierie par décision, sur ~600 lignes
  de CUDA qu'**aucune machine du projet ne peut compiler** hors carte louée ;
- une de ses deux issues (K2 manqué) est **inattribuable d'avance**, faute de
  compteurs `ncu` (refusés par la plateforme, F3) — donc il ne porte que si K1
  est mesuré, donc que s'il embarque `golay70v2k` et `planes12xk` ;
- il exige **trois déblocages non techniques** : le tampon de P4 (absent), le
  ré-ancrage des seuils X3 d'E1c en comptabilité alignée (en souffrance depuis
  le 2026-08-15), et un **arbitrage d'opérateur** sur la divergence qu'il
  créerait entre le manuscrit chez l'éditeur (`tab:validity`, « batch = 1
  assumé ») et le dépôt public.

🚨 **Garde produit à écrire AVANT le chronomètre.** Le plafond amortissable est
borné par le seul terme fixe *mesuré*, `ε = 252 × 3,63 µs = 0,915 ms`
(`a3-graph-2026-08-06.txt:23`), soit un rapport plafond de
`(11,005 − 0,915) ÷ (5,102 − 0,915)` = **2,41×** contre 2,157× aujourd'hui —
+12 %, et **toujours sous les 3,38× d'AWQ du même processus**. Ce 2,41× est
*calculé* et **inter-journaux** : il borne le budget et interdit une phrase
produit, **il ne se publie pas** (règle n°2 du §7). Le « 1,55× » qui circule
(3,33 ÷ 2,15) oublie que le témoin FP16 amortit **aussi** sa part fixe : à
retirer.

⚠️ Et le concurrent 2 bits **n'est pas comparable à batch N** avec ce qui est
fetché : le shim QTIP n'a pas de dimension de batch, leur objet batché est un
`decompress + GEMM` absent du dispositif. À déclarer, pas à contourner.

**L8 — Compensation bas-rang (BACKLOG §3.3).** *~1,5–2,0 $.*

Le dernier levier de qualité dont la littérature publie un effet (+4 à 11 pp).

🔎 **Le point de lecture qui le rend budgétable à 2 $ plutôt qu'à 24 $, et qui
n'est écrit nulle part** : l'adaptateur se compare à la **même base**, sur la
même empreinte de tokens — c'est un A/B à **artefact de base constant**, donc sa
barre est l'intervalle **apparié** (±0,12 % en ppl, SE 0,43 pp en MMLU), et
**non** le σ = 5,2 % de F5, qui porte sur le niveau absolu et pas sur le delta.

🚨 **Gate à profondeur obligatoire au 0.6B (28 blocs) avant tout dollar.** La
classe « raffinement qui améliore un proxy local » s'est retournée **deux fois**
à pleine profondeur : `group_scales` (44,66 → 53,60) et design C (×1,99). Budget
d'octets fixé d'avance à **≤ 0,25 b/param modèle entier**, sans quoi on rachète
la qualité en octets et toute la comparaison AWQ est à refaire.

**L9 — Le 4B : étage 3, puis D3 en appendice.** *~69 $ + ~2 $, puis ~21-24 $.
Go budget séparé.*

🚨 **Le programme refuse de lancer D3 en solo.** D3 **recalibre**, donc il tombe
sous σ = 5,2 % en perplexité et sous une σ MMLU jamais mesurée. Un run unique
contre un témoin unique est **illisible**, et le devis de 7,6-8,0 $ qui circule
est le devis d'un bras illisible. Le devis honnête est **k ≥ 3 des deux côtés**.

⚠️ Et le témoin publié **n'est pas un quatrième tirage** : il a tourné en préfixe
contigu, quand un bras à graines change le **mode d'échantillonnage** en même
temps que le texte. La comparaison propre est **trois graines F5 contre trois
graines DCLM-edu**, jamais contre le point publié.

**Si `s > 2,0 pp`, ni l'un ni l'autre ne se lance** — et c'est alors la phrase de
clôture de l'axe (a) : *à la taille publiée, le bruit de tirage excède tout
effet de composition que cet instrument sépare, et les −14,73 pp restent non
attribués.* **C'est une clôture honnête, pas un échec** : elle ferme la question
par une mesure de résolubilité au lieu de la décréter.

## 3. Articulation avec le protocole de variance

Le programme **exécute** [`plan-variance-2026-08-26.md`](plan-variance-2026-08-26.md),
il ne le remplace pas — mais il le **déverrouille** et l'**étend**.

| | |
|---|---|
| **le précède** | L1 pose le tampon que le §3 du protocole exige, et produit `s`, que le protocole **ne produit pas** (il mesure une σ de *perplexité* au 0.6B, où MMLU est au hasard) |
| **l'exécute mot pour mot** | L2 = étage 0 · L3 = étage 1 étapes 1-2 · L6 = étapes 3-4 + étage 2 · L9 ouvre par l'étage 3 |
| **le rend plus économe** | l'étage 3 liste `leech1c12 ×1 k=3` comme « déjà payé », mais son axe capacités suppose une campagne MMLU non budgétée. L1 la produit : **le protocole gagne une cellule et perd ~0,5 $ implicites** |
| **l'étend** | axes (b) et (c) en entier, hors périmètre déclaré du protocole ; plus L5 et L8 |

🔎 **Et une limite de crédibilité que ni le protocole ni le papier ne nomment**,
sortie de la sonde : `H = AᵀA/N` est accumulée **en f32 sur l'accélérateur**
(`llvq-llm/src/calib.rs:57-58` — `to_dtype(F32)` puis `matmul` sur device), donc
**un tiers sur un autre backend n'obtient pas les mêmes poids**, là où l'encodeur
Leech est exactement déterministe. C'est le point non tranché n°11 de
`fiche-4b.md` §8, jamais repris, et c'est ce sur quoi un rapporteur qui tente la
reproduction tombe le premier jour. **Gate A en mesure la moitié intra-backend,
l'étage 2 la moitié inter-backend** : deux mesures déjà prévues répondent à une
objection que le protocole ne s'était pas posée. À écrire dans le journal de
l'étage 0, pas à découvrir en revue.

## 4. Ce qu'on n'explorera PAS, et pourquoi

> Cette section est le vrai produit du programme. Le dossier a une histoire de
> lignes rouvertes trois fois ; chaque entrée ci-dessous est un dollar et une
> semaine qu'une session aval ne dépensera pas.

| chantier | pourquoi il est écarté |
|---|---|
| **Marges et entropie MMLU depuis les logits commités** | présenté comme « un troisième instrument à 0 $ ». C'est calculé **sur MMLU** : une seconde statistique du même examen ne peut pas révéler ce que l'examen ne teste pas. Et la monotonie annoncée est un **artefact d'échelle** — en normalisant la marge par l'écart-type des logits, llvq/f16 vaut 0,686 / 0,911 / 0,893 et **le 8B passe devant le 14B**. Un troisième indicateur monotone existe déjà et est mesuré : le taux de discordance appariée (27,7 / 19,2 / 14,7 %) |
| **Échelles apprises par colonne** | la **classe entière** est mesurée deux fois, avec gate à profondeur, et retournée deux fois (`group_scales`, design C). Aucune idée neuve n'est nommée sur *pourquoi* un axe colonne échapperait à la composition |
| **Second membre du solve d'échelles (`+λ̃·1`)** | gain visé 0,35 %, sous le σ de 0,7 % du lot B et très loin sous les 5,2 % de F5. ⚠️ Et l'erreur à ne pas refaire est de croire que c'est la **crête** qui manque : elle est relative depuis la correction (`gptq.rs:448-451`) |
| **Balayage de l'amortissement `LLVQ_DAMPING`** | 20,6740 / 20,6643 / 20,6014 — écart 0,35 %, sous 1σ, avec la prédiction de nullité écrite **dans le code avant** la mesure |
| **CUDA Graph / coût de lancement** | le verdict est **dans le journal lui-même** : « le sujet est clos, pas reporté ». Le graph récupère 0,167 ms — 0,8 % d'un token — et c'est un **plafond**. ⚠️ Le chiffre survit et il est **remployé** comme garde produit de L7 |
| **`kbench` autonome hors `planesbench`** | ne porte ni `golay70v2k` ni `planes12xk`, donc ne rend **ni K1 ni K2**. Il rend K3 seul — que `bin/preflight` donne pour 0,08 $ et 3 min. 850 lignes et un job pour un critère obtenable ailleurs en trois minutes |
| **`rot_apply` multi-colonnes** | il n'y a pas deux issues : l'indépendance des colonnes d'une WHT est une propriété **du code**, pas une hypothèse. La branche défavorable serait un bug. Et la rotation déplace 39 Ko là où le matvec en déplace 2,5 Go |
| **BBH (27 familles)** | ~34 $ contre ~5 $ pour GSM8K, pour une réponse **plus bruitée** : 27 conventions de réponse, 250 exemples par tâche, et des prompts CoT qui sont un artefact **séparé** du dataset, donc sans provenance épinglable là où le dépôt épingle tous ses corpus |
| **TruthfulQA-mc2** | ne rentre pas dans le format de dump : `parse_row` vérifie `correct == (pick == answer)` (binaire) quand mc2 est une masse de probabilité **continue**. McNemar ne s'y applique pas. Une session qui l'ajoute « comme les autres » produira un dump que `mmlupair` refuse — ou pire, un `pick` fabriqué qui parse |
| **Rejouer la fusion D1 au 8B/14B** | dette de **cohérence** d'une table publiée, pas question ouverte : sa branche « les gates reproduisent » ne change aucune phrase, elle en **préserve** une. À faire avant toute publication d'un chiffre fusé — les trois tailles ou aucune |
| **Le point 32B en qualité (~70-80 $)** | écarté de **ce** programme, pas du projet. Son gate est **à formuler** (il doit porter sur la chute d'écart 14B→32B *avec son z*), et le chemin servi y est muré à **1 024 octets près** — un 32B qu'on ne peut pas servir ne ferme que la moitié de sa question, et cette moitié coûte 70 $ |

## 5. Ce qui n'est pas une mesure, et qui bloque

| | où | qui |
|---|---|---|
| `ots stamp` du prereg de variance | impossible depuis la boîte de session (403) | **le Mac**, avant L2 |
| `ots upgrade proofs/*.ots` | 4 tampons du 08-25 encore en attente | **le Mac** |
| tampon du préreg **P4** | absent de `proofs/` | avant L7 |
| ré-ancrage des seuils **X3** d'E1c en comptabilité alignée | en souffrance depuis le 2026-08-15 | **opérateur** |
| arbitrage batch ↔ manuscrit chez l'éditeur | `tab:validity` dit « batch = 1 assumé » | **opérateur** |
| relecture de la composition **CSR** du papier source | le PDF est chez l'opérateur (recette §1 de CLAUDE.md) | **opérateur** |

## 6. Le pari de fond

Aucun de ces lots ne déplace le plafond, et le programme est écrit en le sachant.
Ce qu'il achète est double : un **énoncé de fermeture qui tient** — parce qu'il
aura été exploré plutôt que décrété — et le terrain où les quatre résultats les
plus utiles de ce projet sont tombés.

`nullk` a été mesuré pour **attribuer** un temps ; il a redéfini le plafond.
QTIP a été porté comme **point de comparaison** ; il a renversé « aucun bras ne
passe sous `nullk` ». Le σ de F5 était un **contrôle** ; il est devenu le papier
n°2. Le 2026-08-26, une session est allée poser un **tampon** ; la première page
de `CLAUDE.md` s'est retournée.

Aucun des quatre n'était le résultat cherché, et le point commun est net : ils
sont tous tombés en mesurant quelque chose qu'on n'était pas obligé de mesurer.
