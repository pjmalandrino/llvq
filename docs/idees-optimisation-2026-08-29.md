# Bloc d'idées d'optimisation — mécanismes candidats, prix du verdict (2026-08-29)

> **Statut : vivier, pas plan.** Aucune idée ci-dessous n'est autorisée ni
> planifiée — c'est le réservoir dans lequel le plan
> ([`plan-apres-depot-2026-08-29.md`](plan-apres-depot-2026-08-29.md)) peut
> piocher. Chaque entrée suit le même gabarit que l'idée bit-serial : le
> **mécanisme**, le **poste mesuré** qu'il attaque (pas un poste supposé), le
> **gain plausible** avec sa provenance, l'**étape 0 gratuite**, le **kill**
> à ancrer avant toute mesure, et le **coût du verdict complet**.
>
> Les postes mesurés, pour mémoire — toute idée doit nommer le sien :
> - plancher `nullk` : **45,2 %** du bras servi ne lit aucun poids
>   (≈ 9 µs de géométrie par lancement : 2,305 ms / 252) ;
> - D1 : la fusion 252 → 144 lancements rend **×1,061** — le poste bouge ;
> - `planes14` au banc : **425 Go/s effectifs contre 661** pour le témoin
>   f16 (F1 : ce témoin est à 1,5-2,4 % de cuBLAS — le matvec n'est pas
>   en cause) ;
> - décodage `Planes14` : **~7 %** du temps de trafic — plafond de toute
>   idée purement ALU sur le chemin servi ;
> - moteur : le témoin f16 de vLLM rend 83,09 tok/s contre 43,6 au nôtre —
>   **×1,9** de gisement orchestration/attention, borne haute non attribuée ;
> - produit : le triplet arbitré vise **8k de contexte** — or tout le dossier
>   mesure des prompts courts. Le préfill n'a jamais été un poste ; à 8k il
>   en devient un.

---

## A — Amortir le décodage et le trafic sur M colonnes (préfill et petit batch)

**Mécanisme.** Le noyau décode un bloc puis fait UN FMA par coordonnée : à
M = 1, chaque octet de poids lu sert une seule multiplication. Décoder une
fois et accumuler sur M colonnes d'activations (M tokens de préfill, ou M
séquences) divise le trafic de poids **par token** par M — et le trafic est
le poste dominant (2 bits = memory-bound par construction).

**Poste attaqué.** Pas le décode seul : le **trafic entier**, au préfill et
en multi-flux. À M = 1 en décode mono-flux, cette idée ne rend rien — c'est
le périmètre explicite du papier. Mais le triplet produit dit 8k de
contexte : un préfill de 8k tokens traité token par token repaierait 8 000
fois les poids. ⚠️ Étape 0 avant tout : **établir ce que le chemin fusé fait
réellement au préfill** (`LLVQ_TIME_PHASES=1`, 0 $) — le protocole publié
(prompts courts, 128 tokens générés) ne le voit pas.

**Gain plausible.** Au préfill : ordre de grandeur ×M jusqu'à saturation
ALU (*estimé* — c'est l'économie Marlin standard ; la note AWQ dit déjà que
M = 1 n'est pas le régime optimal d'une GEMM 4 bits). En décode mono-flux :
zéro, à dire d'avance.

**Kill.** Si le préfill actuel passe déjà par un chemin dense groupé (à
vérifier à l'étape 0), l'idée ne s'applique qu'au multi-flux — hors
périmètre produit → classée, pas codée.
**Coût du verdict.** Étape 0 : 0 $. Noyau M-colonnes (template sur
`tv_planes`) + banc M ∈ {1, 4, 16, 128} : ~0,3 $. Préfill 8k bout-en-bout :
~0,5 $.

> ✅ **ÉTAPE 0 FAITE (2026-08-29, 0 $) — et l'idée est PROMUE : c'est un
> bloqueur produit, pas une optimisation.** Le chemin fusé traite seq > 1
> par une boucle de lignes — un matvec par token, les poids entiers relus à
> chaque fois (`model.rs:573-580`) — et **refuse tout prompt > 256 tokens**
> (`MAX_ROWS`, `model.rs:352`). Le prompt du protocole publié fait ~5
> tokens : aucun chiffre du dossier n'a jamais exercé le préfill, et la
> promesse 8k du triplet est aujourd'hui inservable sur ce chemin. Le kill
> ci-dessus est donc levé dans l'autre sens : le chemin dense groupé
> n'existe pas côté fusé. Journal :
> [`mesures/etape0-vivier-2026-08-29.txt`](mesures/etape0-vivier-2026-08-29.txt) §A.

---

## B — Mégakernel : un lancement par couche, puis par token

**Mécanisme.** D1 a réduit les lancements par fusion de matrices
(252 → 144). L'étape suivante ne fusionne plus des matrices mais des
**lancements** : un noyau persistant qui itère les 4 matvecs d'une couche
(voire les 36 couches, descripteurs de poids en mémoire device) avec
`grid.sync()` entre les étages, au lieu de rendre la main à l'hôte.

**Poste attaqué.** Le plancher `nullk` — ~9 µs de géométrie par lancement,
×144 restants. C'est le poste des 45 % nommément.

**Gain plausible.** D1 a acheté ×1,061 en retirant 108 lancements ; en
retirer ~108 de plus vaut du même ordre (*calculé* sur 0,594 ms / 11,49 ms
par token), soit **+5-8 %** ; la version « un lancement par token » vise le
reliquat du plancher, borne haute **+20-30 %** (*estimé* — le plancher
contient aussi de l'occupation et des queues de grille que `grid.sync` ne
supprime pas, et il coûte lui-même).

**Étape 0 gratuite.** Compter ce que `grid.sync` impose : la grille
coopérative plafonne le nombre de blocs résidents — vérifier que les formes
du 4B y tiennent (calcul d'occupation sur papier, puis `preflight`).
> ✅ **FAITE (2026-08-29)** : 6 blocs/SM × 142 SM = **852 blocs résidents**
> contre **2 432 requis** par gate+up fusé — la grille coopérative naïve
> est impossible, le design obligatoire est un persistant qui boucle les
> lignes (~3 vagues). Support cooperative-launch de cudarc non vérifié.
> Journal : [`mesures/etape0-vivier-2026-08-29.txt`](mesures/etape0-vivier-2026-08-29.txt) §B.
**Kill.** A1 du plan (le `nullk` sous géométrie fusée) rend l'attribution
avant d'écrire : si le plancher ne suit pas le compte de lancements, ce
mécanisme est mal ciblé → clos pour 0,2 $.
**Coût du verdict.** Dev 4-8 j ; banc 0,3 $ ; `fusedrun` 0,25 $.

---

## C — Bit-serial int8 : l'arithmétique sur les plans (l'idée d'hier)

Résumé pour mémoire — le détail et l'échelle de validation sont dans la
conversation du 2026-08-29 et ont vocation à devenir un préreg :
l'identité multilinéaire `dot = v0·Σz + Δ1·S(p0) + Δ2·S(p1) + Δ12·S(p0∧p1)
+ Δ4·S(p2)` est exacte ; en float elle ne rend rien (les sommes masquées
restent par coordonnée) ; en **activations int8 décomposées en plans de
bits**, chaque somme masquée devient LOP3+POPC sur 24 coordonnées d'un coup.

**Poste attaqué.** PAS le chemin servi (décodage ~7 % — plafond connu) :
la **viabilité d'un format sous 4 b/poids**, là où Golay70 v2 (1,77×) et
E1v (0,25×) sont morts bornés en calcul. C'est le test de « l'idée neuve
sur le coût ALU » que la fermeture du 08-16 réclamait.
**Kill d'entrée (0,2 $).** Le bras d'attribution « lectures identiques,
sélection triviale » : si les Go/s effectifs de `planes14` ne montent pas
vers ceux du témoin f16, l'ALU n'était pas le limiteur → tout clos.
**Coût du verdict complet.** ≤ 6 $ (attribution 0,2 $ ; hôte 0 $ ; banc
0,3 $ ; qualité int8 0,3-5 $). ⚠️ Les activations int8 sont une variable
de **qualité** neuve — gate apparié obligatoire, protocole KV q8.

> 🚨 **VALIDÉE PUIS DÉGRADÉE (2026-08-29, 0 $).** L'algèbre est prouvée —
> identité multilinéaire exacte (0 échec / 200 000 en entiers), sommes
> masquées popcount int8 exactes (0 / 1 000 000) — mais le recomptage
> honnête rend **~148 ops/bloc contre 96 au chemin servi** à masques de
> 24 bits : l'avantage ALU annoncé (~×1,5-2) n'existe pas à cette
> granularité. Espérance résiduelle : le dual-issue INT/FP (invisible à un
> compte statique) et des mots de masque ≥ 32 coordonnées — c'est-à-dire
> un format conçu pour. Statut : **long shot conditionné à un redesign de
> format** ; le bras d'attribution à 0,2 $ reste au programme (il sert D
> et E). Journal :
> [`mesures/etape0-vivier-2026-08-29.txt`](mesures/etape0-vivier-2026-08-29.txt) §C.

---

## D — Pipeline warp-spécialisé : recouvrir le défenêtrage par les chargements

**Mécanisme.** Aujourd'hui chaque lane charge sa fenêtre de 4 mots puis la
décortique — chargement et décodage sérialisés dans la même lane. Ada a
`cp.async` : des warps producteurs poussent les fenêtres en shared pendant
que les consommateurs décodent la vague précédente (double buffer).

**Poste attaqué.** L'écart 425 → 661 Go/s effectifs entre `planes14` et le
témoin f16 — SI l'attribution (le même bras qu'en C, kill partagé) le met
sur le compte du décode-dans-le-chemin et non du motif de lecture.

**Gain plausible.** +10-25 % au niveau noyau (*estimé* — borné par le fait
que QTIP, avec un pipeline soigné, ne dépasse pas 65 % de sa propre borne
d'octets : le recouvrement parfait n'existe pas à batch 1).
**Étape 0 gratuite.** Relire le SASS (le `preflight` imprime registres et
spill) : si le compilateur émet déjà des LDG anticipés sur la boucle de
tuiles, le recouvrement existe et l'idée est plus petite qu'elle en a l'air.
**Kill.** Le même bras d'attribution que C — un seul 0,2 $ pour deux idées.
**Coût du verdict.** Dev 2-4 j ; banc 0,2 $.

---

## E — Payer 2 octets pour des fenêtres alignées (stride 16)

**Mécanisme.** Le stride de 14 o donne des fenêtres non alignées (shift 0
ou 16 bits, quatre mots). Un stride de 16 o rend chaque bloc lisible en un
`uint4` aligné — le K-1 Metal a mesuré que la largeur de chargement paie
(float4 : +3,5-5 %).

**Poste attaqué.** Le même écart 425 → 661, hypothèse « motif de lecture »
— c'est le bras complémentaire de D : si l'attribution innocente l'ALU,
c'est cette idée-ci qui prend le relais.

**Gain plausible et prix en bits, à mettre côte à côte d'avance** : +2 o par
bloc = payload 4,667 → **5,333 b/poids** (+14,3 %). Le précédent
Slot32→Planes14 dit que sur cette portion de courbe, 14,7 % de bits ont valu
14 % de vitesse (Go/s constants). **L'espérance est donc ~nulle** : il faut
que l'alignement rende NETTEMENT plus que proportionnel aux octets pour
payer — c'est précisément ce qu'un banc tranche et qu'aucun raisonnement ne
tranche.
**Kill.** ≥ +18 % de vitesse noyau sinon clos (seuil = octets +14,3 % plus
une marge, à ré-ancrer au préreg).
**Coût du verdict.** Variante de transcodeur (le cadre 5-layouts existe) +
banc : dev 1-2 j, 0,3 $.

---

## F — Le KV q8 au contexte du produit (8k)

**Mécanisme.** Rien de neuf à coder : `LLVQ_KV=q8` est livré, qualité verte
(IC contenant zéro), mais son verdict de débit est explicitement « contexte
court seulement ». Le triplet produit dit 8k. La validation manquante est
une mesure, pas une idée — elle est ici parce que c'est le plus gros écart
connu entre « ce qui est mesuré » et « ce que le produit promet ».

**Poste attaqué.** À 8k, le cache KV et l'attention deviennent un poste de
trafic de premier ordre que tout le dossier ignore (36 couches × 8k × …
— à chiffrer à l'étape 0 sur papier, 0 $).
**Gain plausible.** Ce n'est pas un gain, c'est une **dette de validation**
: le verdict peut être « q8 indispensable à 8k » comme « le préfill 8k est
le vrai mur » (→ renvoie à A).
**Coût du verdict.** `fusedrun` avec prompts 4k/8k, f16 vs q8 : ~0,5-1 $.

> ✅ **Étape 0 papier FAITE (2026-08-29)** : à 8k, le cache vaut 1,208 Go
> f16 (0,60 q8) et sa lecture par token décodé pèse **+57 %** du flux de
> poids (+29 % en q8) — poste co-dominant, VRAM servie 3,78 Go f16 contre
> 3,17 q8. 🚨 Mais **F est gelée derrière A** : le chemin fusé refuse tout
> prompt > 256 tokens, le run 8k est impossible avant le préfill de A.
> Journal : [`mesures/etape0-vivier-2026-08-29.txt`](mesures/etape0-vivier-2026-08-29.txt) §F.

---

## G — Profiler la boucle token, enfin (l'étape 0 de toutes les autres)

Le dossier le dit lui-même : **« le profileur n'a jamais été utilisé »** —
toutes les optimisations à ce jour viennent de compteurs instrumentés.
`LLVQ_TIME_PHASES=1` existe et n'a servi qu'une fois (la découverte des
778 Mo du `lm_head` — le plus gros gain du projet, ×1,8, venu d'un profil).
Avant de jouer B, D ou E : un profil par phase de la boucle token servie
(échantillonnage, copies hôte↔device, argmax, écart entre fin de noyau et
lancement suivant). Coût : **un job à 0,25 $**, ou 0 $ s'il s'ajoute à un
run déjà payé.
> ⚠️ **Lacune trouvée avant de payer (2026-08-29)** : `generate_phased` ne
> phase **pas** le préfill — c'est documenté (`model.rs:1296`). Vu le
> verdict de A, c'est précisément le préfill qu'il faut instrumenter : le
> job G exige ce petit dev d'abord, sinon il mesure à côté du poste. Chaque µs attribuée ici change l'espérance de B — et
l'histoire du projet dit que c'est le coup à meilleur ratio information/prix
du bloc.

---

## Ce qui ressemble à une idée et n'en est pas une (déjà réfuté — ne pas redécouvrir)

- **LUT partielle en shared par classe** : c'est Golay70 v2 (logique hissée
  au bloc, deux tables) — 1,77×, sous le seuil, clos.
- **Un format plus petit sans mécanisme ALU neuf** : l'induction sur trois
  points (1,31× / 1,77× / 0,25×) tient tant que C n'a pas parlé.
- **Optimiser le matvec f16 lui-même** : F1 — il est à 1,5-2,4 % de cuBLAS.
- **Le conflit de bancs / pas de 28** : mesuré nul sur Metal (K-1) ; sur
  CUDA, à ne rouvrir qu'avec un profil qui le nomme.
- **La course de décodage contre QTIP** : structurelle au codebook, close
  par le papier lui-même.

---

## Ordre de jeu suggéré (par ratio information/prix)

| # | idée | verdict complet | ce qui peut l'arrêter avant |
|---|---|---|---|
| 1 | **G** — profil de la boucle token | 0-0,25 $ | rien : c'est l'étape 0 commune |
| 2 | **A** étape 0 — que fait le préfill ? | 0 $ | — |
| 3 | **C/D** kill partagé — bras d'attribution ALU vs lecture | 0,2 $ | — |
| 4 | **B** — mégakernel (si A1/G le désignent) | ~1 $ + dev | A1 du plan, G |
| 5 | **F** — KV q8 à 8k | 0,5-1 $ | l'étape 0 papier de F |
| 6 | **A** complet — préfill M-colonnes | ~1 $ + dev | son étape 0 |
| 7 | **D** ou **E** selon le verdict du bras d'attribution | 0,2-0,3 $ + dev | le bras du n°3 |
| 8 | **C** complet — bit-serial int8 | ≤ 6 $ | le bras du n°3 |

Trois runs à 0,25 $ ou moins (n° 1-3) orientent tout le reste : ~0,50 $
pour savoir où vivent réellement les microsecondes avant d'écrire une ligne.
Chaque entrée passe par l'entonnoir habituel — préreg ancré, coût annoncé,
go explicite — le jour où elle sort du vivier.
