# Plan après dépôt — une config, la géométrie, la qualité, les familles (2026-08-29)

> **Origine.** Décision d'opérateur du 2026-08-29 : le papier est déposé
> (10.5281/zenodo.22133607), le mini-papier « calibration de la hessienne »
> est **abandonné**, et la ligne devient : *choisir une configuration servie,
> la geler, attaquer la géométrie de lancement, pousser la qualité, puis
> mesurer d'autres familles de modèles*. Ce plan ordonne ces quatre fronts,
> chiffre chaque run, et pose les critères à ancrer avant chaque mesure.
>
> **Ce qu'il amende, et ce qu'il ne rouvre pas.**
> - Il **confirme** l'ordre de priorité du chapeau de [`PLAN.md`](PLAN.md)
>   (2026-08-16) : la famille `k` / le poste des 45 %, puis la qualité. Les
>   deux fronts y étaient déjà désignés ; ce plan les exécute.
> - Il **rouvre l'axe familles comme axe de MESURE** (le 08-16 a décidé
>   « A6 = référence seulement » — on ne *sert* pas de MoE ; mesurer n'y
>   contredit rien, servir exigerait un ré-arbitrage explicite).
> - Il **ne rouvre pas l'axe format** : plancher `nullk` (4,77× de plafond,
>   `Planes14` en capture 2,16×), barreau produit 3,00 b/poids infranchi par
>   tout le portefeuille, induction Golay70 v1/v2/E1v sur le coût ALU. Rien
>   de neuf n'est connu ; la porte reste fermée.
> - Il **enterre** le mini-papier hessienne : l'oracle du lot B borne la
>   calibration à −1,6 %, et F5 montre que la variance de **graine** (10,3 %
>   d'étendue au 4B) écrase tout ce que damping et volume ont jamais rendu.
>   Travailler la hessienne, c'est raffiner un terme sous le bruit d'un autre.
>
> **La phrase qui cadre tout : les deux plafonds n'ont pas la même nature.**
> - Le plancher `nullk` (45,2 % du bras servi) est **une propriété de la
>   géométrie de lancement** — le papier l'écrit (« it measures the launch
>   geometry used here and not the card ») et D1 l'a démontré une fois :
>   fusionner 252 → 144 lancements rend ×1,061 [1,050–1,069]. Il **bouge**.
> - L'écart de trafic avec QTIP (2,40× d'octets) est **structurel au
>   codebook** (10¹⁴ points ne tiennent pas en LUT). Il **ne bouge pas** à
>   format constant, et l'axe format est fermé. Aucune course de décodage
>   contre QTIP n'est au programme.
>
> **Règles transverses** (héritées de `PLAN.md`, payées une par une) :
> rien ne part sur carte sans **go explicite**, coût annoncé avant et cumul
> après ; un pré-enregistrement ancré (.ots) par run de décision ; une
> variable par A/B ; tout mécanisme touchant aux magnitudes passe le gate à
> profondeur 0.6B/28 blocs ; chaque chiffre porte sa provenance
> (*mesuré / calculé / estimé*) et sa comptabilité. **Ce plan n'autorise
> aucun lancement** — il les chiffre.

---

## 0. Base d'estimation — d'où sortent les chiffres

Tarifs (*mesurés* : `ops/run.py` FLAVORS, recoupés par les factures des
journaux — D1 : 488 s L40S = 0,24 $ ; F5 : 2,58 h rtx-pro-6000 = 7,11 $) :

| flavor | vCPU | VRAM | $/h |
|---|---|---|---|
| `l40sx1` | 8 | 48 Go | 1,80 |
| `a100-large` | 12 | 80 Go | 2,50 |
| `rtx-pro-6000` | 23 | 96 Go | 2,75 |
| `rtx-pro-6000x2` | 46 | 96 Go | 5,50 |
| `h200` | 23 | 141 Go | 5,00 |

Coûts unitaires de référence, tous *mesurés* sur ce dépôt :

| objet | coût | source |
|---|---|---|
| banc CUDA multi-bras (L40S) | 0,08–0,30 $ | F1 163 s = 0,08 $ ; nullk 0,77 $ ; E1v 0,85 $ |
| `fusedrun` 4B, 3 bras, 128 tokens | 0,24 $ | D1 (488 s) |
| `fusedrun` 14B | 1,24 $ | 2026-08-17 |
| requantification 4B complète (GPU, f32) | **7,11 $** (2,58 h) | F5, ×3 runs concordants |
| requantification 4B complète (M3 Max local) | **0 $** (~3,5 h) | run publié |
| requantification 8B | 11,48 $ (4,18 h) | 2026-08-02 |
| requantification 14B | 27,67 $ (302 min) | 2026-08-10 |
| encodeur, cœur-s/poids | 4,77·10⁻⁵ (8B) · 6,36·10⁻⁵ (32B) | profils par phase |
| campagne MMLU, un bras (2 280 q) | ≈ 3–5 $ | *estimé* (échelle des campagnes 4B/8B/14B ; jamais isolé en $) |
| ppl 12 fenêtres, un bras | ≈ 0,2–0,4 $ | *estimé* (165 s de scoring + chargement) |

**Règle de marge** : +25 % sur toute estimation de run > 5 $ — c'est l'erreur
exacte que le dé-risquage 32B a corrigée (« l'estimation était 25 % basse »).
**Règle de dé-risquage** : tout run > 20 $ est précédé d'un pilote à ~10 %
du coût (leçon 32B : 5,43 $ pour corriger 13 $).

---

## Phase 0 — Geler la configuration servie (≤ 0,5 $, 1-2 j-h)

Quatre variantes servies coexistent dans les journaux ; le produit doit en
citer **une** :

| config | tok/s | Go carte | source |
|---|---|---|---|
| `Planes14` + q8 (B2) | 87,0 | 2,56–2,60 | papier §integration |
| `Planes12x` + q8 (G3) | 85,0 [84,7–85,1] | 2,36 | lot G |
| `Planes14` + q8 + ROT_SHARE + **FUSE_AB** (D1) | **100,6** [99,9–100,7] | 2,57 | D1 |
| `Planes12x` + q8 + ROT_SHARE + FUSE_AB | **jamais mesurée** | — | la case vide du 2×2 |

- **0.1 — Compléter le 2×2** : un job `fusedrun` L40S, 3 bras (les deux
  candidates + dense témoin), 128 tokens comparés. **≈ 0,25 $** (*calculé*
  sur D1). Bandes à pré-enregistrer sur le modèle de G3 (~−2 % de débit
  pour −0,2 Go serait la reconduction du verdict Planes12x).
  🚨 **INLANÇABLE EN L'ÉTAT (vérifié le 2026-08-29, 0 $)** : `check_fuse`
  refuse `LLVQ_FUSE=1` hors `planes14` — « seul planes14 se segmente »
  (`llvq-llm/src/fused.rs:563-576`), et le seul noyau segmenté est
  `planes_seg.cu`. La case exige d'écrire un `planes12x_seg` d'abord.
  Alternative à 0 $ : trancher sur les deux points déjà mesurés —
  planes14+FUSE 100,6 tok/s / 2,57 Go (D1) contre planes12x sans FUSE
  85,0 / 2,36 (G3), soit +18 % de débit contre +0,21 Go. Détail :
  [`mesures/etape0-vivier-2026-08-29.txt`](mesures/etape0-vivier-2026-08-29.txt).
- **0.2 — Décision et gel** : « config servie v1 » choisie (débit d'abord,
  sauf si l'écart VRAM change une classe de matériel — il ne le fait pas au
  4B). README, model card HF et billet de blog alignés dessus. 0 $.

**Livrable** : une seule ligne « configuration servie » partout, avec ses
deux formulations de débit. **Critère de sortie** : plus aucune surface
publiée ne cite deux configs concurrentes sans les nommer.

---

## Phase A — Géométrie : faire descendre le plancher (≤ 4 $, 4-8 j-h)

**État établi, à ne pas re-mesurer** : F1 — le témoin f16 maison est à
1,5-2,4 % de cuBLAS (le matvec *lui-même* n'a rien à rendre) ; D1 — la
fusion des lancements rend ×1,061 ; lot G — horloges épinglées, seuls
événements `GpuIdle` (des **creux entre noyaux**, pas du bridage) ; le
témoin f16 de vLLM tourne à 83,09 tok/s là où le nôtre rend 43,6 (×1,9 de
gisement *moteur*, attention paginée et graphes compris — borne haute, pas
attribution). **Conclusion d'orientation : le gisement est entre les
noyaux, pas dedans.**

| bras | question | coût GPU | critère à ancrer (esquisse) |
|---|---|---|---|
| **A1 — `nullk` sous géométrie fusée** | le plancher de 2,305 ms suit-il le compte de lancements (252→144) ? | 0,2 $ | s'il tombe ∝ lancements (~×0,6), le poste est la **latence par lancement** → A2 prioritaire ; s'il ne bouge pas, c'est l'occupation → A3 |
| **A2 — CUDA Graphs sur la boucle token** | capturer la séquence de lancements d'un token (notre chemin cudarc la possède de bout en bout) | 0,25 $ + 2-4 j dev | gain ≥ 8 % bout-en-bout = adopté ; < 3 % = clos |
| **A3 — variantes d'occupation** (multi-lignes par bloc, matvec persistant) | 2-3 bras de banc | 0,5 $ | ⚠️ F1 borne l'attente : viser l'inter-noyau, pas le par-noyau |
| **A4 — A100, géométrie gagnante** | lever (ou borner proprement) la réserve « résultat Ada » du papier | 0,9 $ (banc + fusedrun, a100-large) | bras réseau ≥ FP16 sur A100 = la plus grosse réserve du papier saute ; sinon, l'attribution horloge de lot G s'étend et la réserve devient un mécanisme |

**Budget : ~2 $ nominal, plafond 4 $.**
**Kill de phase** (à ancrer avant A1) : si A1 + A2 + A3 rendent < 8 %
cumulés bout-en-bout, l'axe géométrie **sous candle** est clos par mesure ;
le gisement restant est le moteur lui-même, et « servir dans un autre
moteur » devient une décision d'opérateur séparée — pas un glissement.

---

## Phase B — Qualité : le seul axe qui change le verdict produit (≤ 25 $, 5-8 j-h)

### B-a. Graines — le levier mesuré, à encadrer avant d'en profiter (≈ 4-7 $)

F5 a établi : étendue **10,3 %** de ppl sur 3 graines au 4B, les trois
paires résolues (t jusqu'à 10,9), graine 3 à **15,1027** contre 16,94
publié. À 15,10, l'artefact passerait **nettement sous QTIP (17,04)** — à
8,5 % de bits en plus, toujours.

- **B-a.1 — Protocole de sélection, écrit d'abord** (0 $). 🚨 Choisir la
  graine sur le corpus qu'on publie serait du sur-ajustement à
  l'évaluation. Règle à pré-enregistrer : sélection sur un critère
  **disjoint** (fenêtres de validation séparées, ou C4), publication de la
  **distribution complète** des graines plus l'élue, jamais l'élue seule.
- **B-a.2 — MMLU de la graine 3** (≈ 3-5 $, *estimé*) : la question qui
  compte — les −10,8 % de ppl se traduisent-ils en points de MMLU, ou
  est-ce de la restitution pure ? (Le §3ter dit que les deux se découplent ;
  c'est le test.) Apparié contre le f16 existant, empreinte identique.
- **B-a.3 — Élargir à 5-6 graines, en local** (0 $ GPU, ~3,5 h de Mac
  chacune ; évals ppl à ~0,3 $ pièce). Donne la vraie distribution — et la
  barre d'erreur que F5 réclame pour tout chiffre publié à cette taille.

### B-b. Compensation post-hoc type EoRA (≈ 4-6 $)

Le levier au plus gros gain **publié** (+4 à 11 pp MMLU dans la
littérature), jamais tenté ici. Bas risque algorithmique : fermé (SVD dans
la métrique d'activation), pas d'entraînement.

- **B-b.1 — Implémentation + gate à profondeur 0.6B/28 blocs** (0 $, 2-4 j
  dev). La règle design-C s'applique : pas de validation à 3 blocs.
- **B-b.2 — 4B : adaptateurs + ppl + MMLU** (≈ 4-5 $, dont MMLU 3-5 $).
- 🚨 **La comptabilité attrape ce que l'enthousiasme rate** : des
  adaptateurs r=32 en f16 coûtent **+0,263 b/param** (*calculé* :
  Σ(m+n) = 2 064 384 par rang, ×32 ×16 bits, sur 4,02 Md de params) — ils
  feraient repasser le 4B **au-dessus** de l'AWQ (5,162 → 5,43 vs 5,30).
  Budget bits à poser d'avance : r ≤ 16 ou adaptateurs q8 (+0,13), et le
  b/param publié inclut l'adaptateur, toujours.
- **Critère à ancrer** : ≥ +3 pp MMLU apparié à budget bits tenu = adopté ;
  < +1,5 pp = clos et documenté (la SE appariée à fichier constant vaut
  0,43 pp — mesurée, le seuil est donc à ~3,5σ).

### B-c. Composition du corpus (≤ 8 $, optionnel, après B-a/B-b)

Le seul suspect qualité que l'oracle du lot B **ne borne pas** (mécanisme
raisonnement, §3ter). A/B à une variable : C4 vs C4+math/code, 3 blocs
(0 $) → gate à profondeur (0 $) → une requantification 4B (0 $ local ou
7,11 $ GPU) + MMLU (3-5 $). ⚠️ À lire **au travers** de la variance de
graines — donc à graine fixée, appariée, et l'effet doit dépasser ce que
F5 rend au même protocole.

**Budget phase B : ~12-20 $ nominal, plafond 25 $.**

---

## Phase C0 — Le duel à moteur unique : .llvq ↔ AWQ, noyaux séparés (≈ 0,5-2 $, 3-6 j-h)

> Ajoutée le 2026-08-29 (soir) sur décision d'opérateur : la cible à terme
> est la comparaison .llvq ↔ 4 bits **à noyaux séparés dans un seul
> moteur** — celle qui rend enfin licite la case vitesse laissée vide
> depuis le lot A (« les deux rapports ne se divisent pas »).

**Ce qui existe déjà, vérifié** : le noyau AWQ (vendoré, MIT) tourne dans
le même processus que les nôtres au niveau **banc** depuis le 08-10 —
3,37× vs FP16 contre 2,16× pour `planes14`, f64 ligne à ligne, un seul
protocole (six bras, F1). **Ce qui manque** : l'étage modèle — un
`Proj::Awq` dans `fusedrun`, trois bras (.llvq, AWQ, dense témoin), même
attention, même tête, mêmes phases.

| poste | contenu | coût |
|---|---|---|
| chargeur AWQ officiel (qweight/qzeros/scales g128) → `Proj::Awq`, câblage `group_forward`, pas de rotation (base naturelle) | dev | 3-6 j |
| duel 4B, 3 bras, protocole `fusedrun` | run | ~0,5 $ |
| option : rejouer 8B et 14B (checkpoints AWQ officiels existants) — la courbe d'échelle entière dans un moteur | runs | ~1,5 $ |

**Règles d'équité, à ancrer au préreg avant toute mesure** : même tête des
deux côtés (q8 ou f16 — la règle « à tête identique » devient
structurelle) ; même KV ; même prompt et mêmes 128 tokens de protocole ;
**symétrie de fusion déclarée** — notre bras a ROT_SHARE+FUSE, le bras AWQ
reçoit l'équivalent (qkv concaténé hors ligne) ou les deux tournent non
fusés. Une asymétrie silencieuse invaliderait le duel entier. Chemin GEMV
batch 1 des deux côtés (Marlin/M≥8 hors périmètre, dit d'avance).

**Pronostic à écrire avant la mesure, et il ne nous flatte pas** : dérivé
des rapports de banc (octets ~4,3 contre 4,80 b/poids ; 584 contre 425
Go/s effectifs), le bras AWQ devrait rendre **+10-25 % de décode
bout-en-bout au 4B** (*estimé*). Le livrable n'est pas une victoire, c'est
la **table de trade-off à quatre axes dans une seule pile** — vitesse
(probablement AWQ), VRAM (nous : −2,6 / −10,6 / −5,5 % selon la taille),
disque (nous : −34 %), qualité (AWQ : +6 à +14 pp) — qu'aucun dossier
public ne possède, parce que personne n'a les deux noyaux dans un moteur.
**Kill** : aucun — les deux issues se publient ; seul un écart
d'orchestration non attribuable (bras AWQ pénalisé par un chemin que le
nôtre ne paie pas) invalide et renvoie au dev.

**Séquencement** : après le gel de la config servie (Phase 0) et de
préférence après le préfill (front 3 de l'ordre) — le bras AWQ, une fois
écrit, se réutilise tel quel pour la Phase C1 et chaque nouvelle taille.

## Phase C — Autres familles : le déficit est-il un fait Qwen ? (17 $ ; option MoE +65 $)

### C1 — Une famille dense hors Qwen (Llama-3.1-8B ou Mistral-7B) (≈ 17 $, 4-7 j-h)

Ce que ça teste : la falaise MMLU (−10,56 pp au Qwen3-8B) est-elle un fait
du 2 bits ou un fait de Qwen3 ? Une réponse dans un sens ou l'autre est
publiable — et c'est le premier point de généralité du dossier.

| poste | coût | provenance |
|---|---|---|
| portage passe avant + oracle (`max |Δhidden| = 0` contre la référence candle) | 2-4 j dev, ~0,05 $ | le harnais existe ; RoPE/normes diffèrent, l'oracle est fait pour ça |
| requantification 8B | ≈ 12 $ | *calculé* (4,77·10⁻⁵ cœur-s/poids ; recoupe les 11,48 $ mesurés du Qwen3-8B) |
| campagne qualité (ppl + MMLU appariés, f16 + LLVQ + 4 bits de référence) | ≈ 4-5 $ | *estimé* |
| vitesse `fusedrun` | 0,5 $ | *calculé* |

⚠️ **Une graine, assumée** : trois graines coûteraient 3×12 $. On publie
mono-graine **avec l'étendue F5 du 4B citée comme majorant d'incertitude**
(étiquetée : mesurée à une autre taille, autre famille) — ou on paie.

### C2 — MoE Qwen3-30B-A3B, en référence seulement (≈ 65-70 $, 6-10 j-h) — option, après B

Compatible avec l'arbitrage A6 du 08-16 (mesurer ≠ servir). C'est le seul
axe connu qui change la **classe** de machine : 61 Go en bf16 → **≈ 19 Go**
à ~4,8-5,1 b/param (*estimé* — embedding et routeur à compter au portage),
soit une carte 24 Go pour un modèle classe 30B. Et un fait arithmétique
favorable : `moe_intermediate = 768 = 24 × 32` — **zéro queue** sur les
experts (*calculé*, à vérifier au portage).

| poste | coût | provenance |
|---|---|---|
| portage routing + oracle + hessiennes par expert | 4-8 j dev, ~0,1 $ | ⚠️ risque n°1 : couverture en tokens par expert (128 experts se partagent la calibration) — à chiffrer AVANT le run |
| dé-risquage (quelques couches, règle des 10 %) | ≈ 5 $ | leçon 32B |
| quantification complète | ≈ 49 $ nominal → **60 $ avec marge** | *calculé* : ~29,9 Md poids × 4,8·10⁻⁵ cœur-s ÷ 46 vCPU ≈ 8,8 h × 5,50 $/h ; +25 % |
| qualité + vitesse | ≈ 5-6 $ | *estimé* |

**Go de C2 = décision d'opérateur explicite** : il fait passer le total du
plan au-dessus de la barre informelle des 100 $ (voir récapitulatif).

---

## Phase D — v2 des surfaces publiées (0 $, 1-2 j-h)

Après les verdicts A/B (et C s'il est joué) : version 2 du préprint sur le
même DOI de concept (Zenodo versionne), mise à jour du billet HF, des model
cards (nouveaux artefacts de graine ou de famille), du README. Les règles
de rédaction du §7 s'appliquent — chaque nouveau chiffre avec plage,
provenance, comptabilité.

---

## Récapitulatif budgétaire

| scénario | GPU ($) | dev (j-h) |
|---|---|---|
| Cœur : 0 + A + B + D | **~15-30** | 11-20 |
| + C1 (famille dense) | **~32-47** | 15-27 |
| + C2 (MoE, sur go explicite) | **~100-115** | 21-37 |

Chaque run individuel reste sous la discipline existante : préreg ancré,
coût annoncé, go explicite, cumul rapporté. Les plafonds de phase (4 / 25 /
17+70 $) sont des **plafonds d'arrêt**, pas des budgets à consommer.

**Ordre d'exécution — RÉVISÉ le 2026-08-29 (soir), après l'étape 0 du
vivier** ([`mesures/etape0-vivier-2026-08-29.txt`](mesures/etape0-vivier-2026-08-29.txt)) :

1. **Décisions à 0 $, aujourd'hui, opérateur** : (a) geler la config servie
   sur les deux points mesurés — planes14+FUSE 100,6 tok/s / 2,57 Go
   contre planes12x 85,0 / 2,36 (la 4ᵉ case du 2×2 exige un
   `planes12x_seg`, cf. le 🚨 de la Phase 0) ; (b) publier le billet HF ;
   (c) ORCID + profil Scholar.
2. **Jobs d'orientation, ~0,4 $ au total, sur go** : A1 (`nullk` sous
   géométrie fusée) + le bras d'attribution ALU/lecture (kill partagé des
   idées C, D, E du vivier). Ils décident de tout l'axe géométrie avant
   une ligne de dev.
3. **Le chantier préfill — le bloqueur produit** (vivier idée A, promue) :
   dev du chemin GEMM-préfill (le fusé boucle un matvec par token et
   refuse > 256) + phasage du préfill dans `generate_phased` (idée G) ;
   puis le job de profil (0,25 $) et le banc préfill (0,3-0,5 $).
4. **Qualité en parallèle, dev local** : protocole de sélection de graines
   écrit et ancré (B-a.1, 0 $) ; MMLU de la graine 3 (3-5 $) au premier go
   — le chiffre qui peut changer le claim qualité.
5. **Géométrie, selon le verdict d'A1** : A2 (CUDA Graphs) → B du vivier
   (mégakernel persistant — bandes de gain dans le vivier, +3-6 % seul,
   +8-15 % avec Graphs, *estimé*) → A4 (A100 avec la géométrie gagnante).
6. **Ensuite** : B-b (EoRA) → B-c (corpus) → C1 (famille dense) →
   arbitrage C2 (MoE, go explicite) → Phase D (v2 Zenodo/blog/cards).

Les fronts 3 (dev) et 4 (dev local + un job) restent parallélisables avec
les jobs courts du front 2.

## Ce que ce plan enterre, pour qu'on ne le redécouvre pas

- Le **mini-papier calibration hessienne** (oracle lot B −1,6 % ; variance
  de graine F5 7× le σ supposé — le terme est sous le bruit d'à côté).
- La **course de décodage contre QTIP** (écart de trafic structurel au
  codebook ; le papier en a fait un résultat, pas une défaite).
- La **chasse au format** (plancher 4,77×, barreau 3,00 b/poids, induction
  ALU — fermée le 08-16, rien de neuf connu).
