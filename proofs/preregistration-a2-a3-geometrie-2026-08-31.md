# Pré-enregistrement — A2 (CUDA Graphs) et A3 (occupation) — BROUILLON

> 🚨 **NON TAMPONNÉ — BROUILLON du 2026-08-31.** Les parts d'A1 sont
> **remplies** (§3, mesurées le soir même : r = 0,8158, bande mixte). Restent
> avant le tampon : **l'arbitrage des critères d'A3** (§5, marqués
> PROPOSITION) et **l'ordre A2/A3** (§3, proposition faite). Règle une fois tamponné : ce document ne s'édite
> plus jamais ; écarts →
> `proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md`, nommé ici
> d'avance. **Tampon exigé avant la première milliseconde d'un job A2 ou A3.**

## §0 — Noms, parce que « A1..A4 » est un espace de noms à trois collisions

Les A de CE document sont ceux de la **phase A géométrie**
(`docs/plan-apres-depot-2026-08-29.md:148-151`) : A1 = `nullk` sous géométrie
fusée, A2 = CUDA Graphs sur la boucle token, A3 = variantes d'occupation,
A4 = A100. Ils ne sont **ni** les A1..A4 du lot A du 2026-08-06 (branchement,
KV, graph, campagne — `docs/archive/rapport-lot-a-2026-08-06.md`), **ni** le
triplet produit du 2026-08-13. Toute citation croisée nomme sa date.

## §1 — La décision qui fonde ce document, et ce qu'elle ne change pas

Décision d'opérateur du 2026-08-31 (`deaa449`) : **A2 et A3 se font quoi
qu'il arrive**, port ou pas. Motif : la crédibilité — le papier revendique un
noyau, candle est le seul moteur auditable de bout en bout, l'argument
souverain exige que ce chemin soit rapide par lui-même. Ce que la décision ne
change pas, et qui est repris ici tel quel : **les critères d'adoption
restent** — on construit et on mesure, un résultat nul ne s'adopte pas. A1
garde son rôle du préreg vague 2 : **ordonner** A2 contre A3.

## §2 — Les priors, TOUS défavorables à A2, et déclarés avant la mesure

C'est le cœur de l'honnêteté de ce préreg : la ligne « CUDA Graphs » a déjà
été **fermée par mesure** et ce document la rouvre sur décision, pas sur fait
neuf. Un lecteur doit le savoir avant le premier chiffre.

| prior | valeur | source |
|---|---|---|
| A3(b) du lot A : « le sujet est clos, pas reporté » | le graph récupère **0,167 ms = 0,8 % d'un token**, et c'est un **plafond** | `docs/mesures/a3-graph-2026-08-06.txt:49-57` (*mesuré*) |
| le blocant technique du même verdict | un graph statique ne capture pas un `Tensor::cat` qui grandit — il faut **préallouer le cache KV** à formes fixes | même journal |
| plan de clôture du 08-27 : enterré | « le verdict est dans le journal lui-même » | `docs/plan-cloture-2026-08-27.md:260` |
| F3 : la soumission hôte est **entièrement recouverte** | écart hôte−device **0,1-0,2 %** | `docs/mesures/` F3 (*mesuré*) |
| F1 : le par-noyau est déjà au niveau cuBLAS sur L40S | témoin ≤ 1,05× | F1 (*mesuré*) — d'où l'avertissement du plan : **viser l'inter-noyau** |

Et deux priors qui laissent la porte ouverte, déclarés aussi :

| prior | valeur | source |
|---|---|---|
| D1 : 108 lancements de moins valent | 87,0 → 94,9 → **100,6 tok/s** (hissage puis fusion, *mesuré*) | `d1-fusion-servie-2026-08-24.txt` |
| l'en-tête de `nullk.cu` : 108 lancements ≈ 0,392 ms | **r ≈ 0,83 attendu** pour A1 — bande **mixte** | `nullk.cu` (*calculé*, prior déclaré au préreg vague 2 §A1) |

⚠️ Le 0,167 ms du lot A a été mesuré sous la géométrie **252 lancements,
ROT_SHARE=0** — la config servie v1 est à 144 matvec + rotations hissées. La
grandeur a pu bouger dans les deux sens ; c'est A1 qui la re-mesure, pas ce
tableau.

## §3 — Les parts d'A1 (gabarits, à remplir AVANT le tampon)

A1 (préreg vague 2 §A1, tamponné `e23e9895…`) rend
`r = t(nullk-fusé 144) ÷ t(nullk 252)` :

- **r = 0,8158 [0,8150–0,8162]** (*mesuré* le 2026-08-31, job `6a95e11b…`,
  0,01 $ — médiane du rapport round par round, 7 rounds dont 2 jetés, un seul
  processus, L40S 142 SM ; le prior déclaré de 0,83 est confirmé à 1,7 % près)
- lecture pré-posée : r ≤ 0,65 → latence par lancement, **A2 d'abord** ;
  r ≥ 0,90 → occupation, **A3 d'abord** ; **0,8158 → BANDE MIXTE** — les
  parts se publient, ni A2 ni A3 n'est éliminé. L'ordre est au choix de
  l'opérateur ; les deux pools, pour l'éclairer : le pool par-lancement est
  **mesuré directement** (Δ = 0,406 ms pour 108 lancements = **3,76
  µs/lancement**, *calculé* — cohérent avec les 3,63 µs du lot A du 08-06),
  le pool d'occupation est le **plancher résiduel à 144 lancements, 1,794
  ms**, plus gros mais sans mécanisme unique. ⚖️ PROPOSITION : A2 d'abord —
  son pool est certain et son critère tranche vite ; à arbitrer.
- part par-lancement implicite : **0,406 ms** des **2,200 ms** du plancher
  252 pour les 108 lancements retirés ; extrapolée linéairement aux 252,
  **0,947 ms ≈ 43 %** (*calculé*, hypothèse de linéarité DÉCLARÉE — seule la
  différence à 108 lancements est mesurée)
- journal : `docs/mesures/a1-nullk-252-144-2026-08-31.txt` (sha256
  `6c811c5d…`, identique aux logs du job — authenticité vérifiée avant usage)
- ⚠️ le 2,200 ms de ce processus ne se soustrait PAS au 2,306 de F2 ni au
  2,305 du 08-16 — autre processus, le banc l'imprime lui-même
- 🆕 **le même r sur A100, le même soir (A4, bras en prime)** : **0,8198
  [0,8196–0,8202]** — invariant à 0,5 % entre les deux architectures quand
  les temps absolus s'étirent de ×1,809 ≈ le rapport d'horloges 1,787 du
  lot G (*calculé*). Le poste par-lancement est une propriété de la
  GÉOMÉTRIE, pas de la carte — ce qui renforce la valeur d'A2/A3 : ce
  qu'ils gagneraient vaut sur les deux architectures
  (`docs/mesures/a4-a100-2026-08-31.txt`)

## §4 — A2 : CUDA Graphs sur la boucle token

- **Objet** : capturer la séquence de lancements d'un token de `fusedrun`
  (config servie v1 : `planes14+q8+ROT_SHARE=1+FUSE=1`) dans un CUDA Graph,
  rejouer le graph par token. Le chemin cudarc est possédé de bout en bout.
- **Le blocant de 2026-08-06 doit être levé d'abord, et il est nommé** :
  préallocation du cache KV à formes fixes (fenêtre bornée), sans quoi la
  capture est impossible. C'est un chantier de 2-4 j déclaré au plan.
- **Critères, gelés par `deaa449`** : gain bout-en-bout **≥ 8 % → adopté** ;
  **< 3 % → clos** ; entre les deux : point de courbe, non adopté, la ligne ne
  se rouvre pas sans fait neuf. La comparaison est **intra-job**, médiane sur
  5 rounds avec plage, tokens gloutons identiques exigés entre bras.
- **Coût** : ~0,25 $ de carte (*estimé*, plan :149) + le dev.

## §5 — A3 : variantes d'occupation — ⚠️ PROPOSITION, à arbitrer avant tampon

Le plan (:150) donne l'objet — 2-3 bras de banc : multi-lignes par bloc,
matvec persistant — mais **aucun critère chiffré n'existe nulle part** pour
A3, et aucun design doc. Proposition à arbitrer :

- **Étage banc (gate d'entrée au port)** : un bras d'occupation doit battre
  `planes14` (formes servies, même processus, protocole planesbench) de
  **≥ 10 %** pour mériter son port dans `fusedrun`. Sous 10 %, point de
  courbe, pas de port — le port coûte des jours et le bout-en-bout dilue.
- **Étage bout-en-bout (adoption)** : mêmes seuils qu'A2 — **≥ 8 % adopté,
  < 3 % clos** — pour qu'aucun des deux bras ne soit avantagé par son barème.
- ⚠️ F1 borne l'attente par-noyau ; l'hypothèse d'A3 est l'**inter-noyau**
  (occupation entre lancements, bulles que F3 loge dans le span device).
- **Coût** : ~0,5 $ (*estimé*, plan :150) + le dev des bras.

## §6 — Le kill de phase, ancré ICI (il ne vivait qu'en prose)

`docs/plan-apres-depot-2026-08-29.md:154-157` l'exigeait « avant A1 » et
aucun tampon ne le portait. Le voici, à la lettre du plan :

> **Si A1 + A2 + A3 rendent < 8 % cumulés bout-en-bout, l'axe géométrie
> SOUS CANDLE est clos par mesure.** Le gisement restant est le moteur
> lui-même, et « servir dans un autre moteur » devient une décision
> d'opérateur séparée — pas un glissement.

Le cumul se mesure sur le chemin servi v1, 4B, intra-job, contre la config
v1 sans les mécanismes d'A2/A3 — jamais en additionnant des pourcentages de
jobs différents.

## §7 — Budget

A2 ~0,25 $ + A3 ~0,5 $ ; **plafond de phase : 4 $** (plan :153) — distinct du
plafond 5 $ de la vague 2 (qui couvre 0.1, 0.2, A1, A4 ; dépensé à ce jour :
1,32 $, *mesuré* `docs/data/jobs.csv`). Au-delà : arrêt, retour opérateur.
