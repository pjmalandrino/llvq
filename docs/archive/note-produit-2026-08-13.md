# Note produit — le barreau mémoire, les métriques et l'environnement d'expérimentation

> 🗓️ **2026-08-16 — ARBITRAGE COMPLET. Les six cases du §A sont tranchées.**
> L'opérateur a arbitré **A2 = 8k**, **A4 = marge 5 Go**, **A6 = offload en
> référence seulement**, et confirmé l'**unité : 32 GiB**. S'ajoutent aux trois
> du 2026-08-15 (**A1** RTX 5090 32 GiB · **A3** KV q8 · **A5** ≤ 60 s/doc et
> ≥ 20 tok/s). **Le §B fait foi**, au triplet ci-dessous et pas à un autre.
>
> ✅ **La condition que cette note se posait à elle-même est levée** : le
> fichier est commité et suivi — le triplet est **figé**. 🕳️ Ce paragraphe a
> maintenu « tant qu'il ne l'est pas » après le commit du fichier (constaté et
> corrigé le 2026-08-18) : la condition était satisfaite, le texte ne le
> disait pas, et le statut du triplet que `PLAN.md` déclare « fait foi »
> restait formellement ambigu.
>
> 🚨 **Et l'arbitrage déplace le seuil, dans le sens que personne n'attendait.**
> Le §B publiait 2,74 pour « 8k, KV q8, marge 5 Go ». Ce nombre supposait « 32 »
> sans unité et un KV q8 à ÷2. Avec les deux corrections que l'arbitrage impose
> — **32 GiB** (+0,2758 b/poids, le calcul est celui de
> [`preregistration-p5-2026-08-14.md`](../proofs/preregistration-p5-2026-08-14.md))
> et le **÷1,882 mesuré** de P3 au lieu du ÷2 supposé (−0,0099) — le seuil du
> triplet arbitré vaut **3,00 b/poids noyau**, pas 2,74. **Toute la table est
> refaite au §B**, et sa conséquence sur le portefeuille de layouts avec elle.
>
> Cette note devient donc ce qu'elle annonçait : la référence que les
> pré-enregistrements de `proofs/` citent, et ce qui donne un statut au `S_alt`
> du pré-enregistrement du 2026-08-13 — lequel vaut **3,09** parce qu'il est
> calculé à **marge 2 Go**, donc pas au triplet retenu.
>
> 🆕 **2026-08-25 — révisée, et l'arbitrage NE BOUGE PAS.** Trois choses ont
> changé sous cette page sans en déplacer une case : le **plafond de 4,77× FP16
> est mesuré faux** (lot F2, QTIP porté dans notre propre banc — §B et §E) ; le
> **chemin servi au 32B est muré par une arithmétique de mémoire partagée**
> (§B bis) ; et le **σ de calibration vaut 5,2 %, pas 0,7 %**, à la taille
> publiée (lot F5 — §D). `b_max` = **3,00 b/poids noyau** reste le barreau,
> aucun layout du portefeuille ne le passe, et les six cases du §A sont
> inchangées — **A3 compris**, la variance de F5 ne touchant aucun A/B à
> fichier constant. Ce qui change est la **compréhension** de l'obstacle,
> jamais son emplacement.

**Pourquoi cette note existe.** Le critère qui a enterré E3 (≤ 2,60 b/poids
noyau) n'est pas une constante physique : il se dérive de
`((VRAM − marge − KV)×8 − embedding)/68,45 Md`, et chacun de ces termes est un
**choix produit** jamais acté — 5 Go de marge, KV en f16, contexte 8k. Avec
d'autres choix aussi défendables, le même barreau 32 Go donne un seuil de
3,09. Aucun des deux n'est « le vrai » : le vrai est celui que cette page
fige.

---

## A. Les décisions à arbitrer (cases à cocher — opérateur)

**A1. Carte cible du barreau principal** *(une seule ; les autres deviennent
des segments secondaires)* :

- [x] **RTX 5090 32 Go, PCIe gen5** ✅ **ARBITRÉ 2026-08-15** *(la carte grand
  public haut de gamme 2026, bande passante ~1,8 To/s, celle que l'étude MoE
  nomme)*
  > ✅ **UNITÉ CONFIRMÉE 2026-08-16 : 32 GiB.** La 5090 porte 32 GiB et
  > `nvidia-smi` en rapporte 32 768 MiB ; c'est la lecture que le matériel
  > impose. La correction vaut **+0,2758 b/poids** sur tout le §B —
  > `(34 359 738 368 − 32 000 000 000) × 8 / 68,45e9`, le calcul étant celui
  > déjà posé au §5 de
  > [`preregistration-p5-2026-08-14.md`](../proofs/preregistration-p5-2026-08-14.md).
  > 🕳️ **La table §B d'origine était donc fausse d'un tiers de bit, en notre
  > défaveur**, et elle l'était parce qu'elle multipliait par 8 un nombre dont
  > l'unité n'était écrite nulle part. C'est le même motif que les trois règles
  > de chiffres du §7 de `CLAUDE.md` : un nombre sans sa comptabilité.
- [ ] 24 Go (3090/4090 d'occasion — le segment le plus « souveraineté », mais
  zéro marge opérationnelle sur tout ce qu'on sait faire)
- [ ] Mac à mémoire unifiée 64-128 Go (exige le portage Metal de la rotation,
  semaines)

**A2. Contexte servi** *(impact direct sur le seuil, table §B)* :

- [x] **8k** ✅ **ARBITRÉ 2026-08-16**  - [ ] 16k  - [ ] 32k

> ✅ **Seuil qui en découle : `b_max` = 3,00 b/poids noyau** (avec A4 = 5 Go,
> A3 = q8 au ÷1,882 mesuré, 32 GiB). Table complète au §B.
> 🔎 **Deux conséquences à porter, et la seconde est un gain gratuit.**
> (i) Le choix n'a pas été fait pour retomber sur le 2,60 historique — il ne le
> reproduit pas : 3,00 contre 2,60, et le 2,60 se relisait « 8k, KV f16, 5 Go »
> (2,58 à la ligne correspondante de l'ancienne table). L'interdit du déplacement
> de poteaux est tenu.
> ⚠️ **Mais l'écart ne s'attribue pas au q8, et l'écrire serait la faute que ce
> document police.** Décomposé [calculé] : 2,58 → **+0,2758 la correction GiB**
> → 2,86 → **+0,148 le passage au q8** → 3,00. **Les deux tiers de l'écart
> viennent de l'unité, pas du cache.**
> (ii) **8k maintient le KV q8 dans la région où P3 l'a mesuré.** Le verdict de
> P3 est étiqueté « contexte court seulement » parce que la série `n_new = 1024`
> a été abandonnée (661 s > 600 s posés d'avance) ; à 8k servi, la région non
> visitée n'est **pas** la région servie, donc ce verdict suffit pour A3 et
> l'instrument de mesure à contexte long **cesse d'être un prérequis produit**.
> Il redevient ce qu'il est : une question ouverte, sans échéance.
> ⚠️ Ce que ça ferme aussi : à 8k, l'argument « le cache borne la classe de
> modèle chargeable » ne joue plus au premier ordre (KV q8 8k = **1,435 Go**
> contre 2,7 en f16, sur un budget de 32 GiB).

**A3. Format du cache KV** :

- [ ] f16 (existant)
- [x] **q8** ✅ **ARBITRÉ 2026-08-15** — levier transversal, ~1,35 Go au lieu de
  2,7 à 8k sur un 70B
  > 🕳️ **« à construire » est périmé** : P3 l'a construit, mesuré et livré le
  > 2026-08-15. `LLVQ_KV=q8`, cache à **8,5 bits** (int8 + échelle et biais f16
  > par groupe de 64, soit **÷1,882 et non ÷2**), qualité verte — ppl +0,049 %,
  > MMLU +0,33 pp, les deux intervalles appariés contenant zéro
  > ([`mesures/kvq8-4b-2026-08-15.txt`](mesures/kvq8-4b-2026-08-15.txt)).
  > ⚠️ **Il n'est pas le défaut** et son verdict est étiqueté « contexte court
  > seulement » : la série `n_new = 1024` a été abandonnée (661 s > 600 s posés
  > d'avance). L'arbitrer ici, c'est décider qu'on le sert — pas constater qu'il
  > est validé à tout contexte, ce qu'il n'est pas. **Et c'est A2 qui décide si
  > la région non mesurée est la région servie.**

**A4. Marge opérationnelle réservée** (contexte CUDA, activations, jitter) :

- [ ] 2 Go  - [x] **5 Go** ✅ **ARBITRÉ 2026-08-16**

> ✅ **La marge confortable est retenue**, et elle coûte **0,35 b/poids** de
> seuil à contexte fixé (3,35 à marge 2 Go contre 3,00 à marge 5 Go, au triplet
> arbitré). C'est un jugement d'exploitation assumé, pas une grandeur dérivée :
> rien dans le dépôt ne mesure ce que consomme réellement un contexte CUDA sur
> 5090, et un dépassement se paie en OOM à l'exécution, pas en dégradation.
> ⚠️ **Ce que ce choix rend prioritaire** : si un layout venait à échouer le
> `b_max` de 3,00 **par moins de 0,35 b/poids**, la mesure de ce que consomme
> vraiment le contexte CUDA sur la carte cible deviendrait le geste le moins
> cher pour le récupérer — avant tout travail de format. Aucun layout du
> portefeuille n'est aujourd'hui dans cette bande (cf. §B), donc la question ne
> se pose pas ; elle se poserait pour un candidat neuf entre 3,00 et 3,35.

**A5. Plancher de vitesse, par segment** *(sans lui, aucun banc n'a de
critère d'admission)* :

- Extraction par lots (package B) : ✅ **ARBITRÉ 2026-08-15 — ≤ 60 s/document
  en lot de 8** (la proposition de cette note)
- Chat/agent local (packages A et C) : ✅ **ARBITRÉ 2026-08-15 — ≥ 20 tok/s**
  (la proposition de cette note ; repère : le 4B publié rend 88,4-88,5 tok/s
  sur L40S, le 8B 69,3 — mesurés, mais sur une autre carte que A1)

**A6. L'offload PCIe est-il admis comme solution de référence** (ce qu'on
doit battre) **ou comme solution servie** (ce qu'on livre) ?

- [x] **référence seulement** ✅ **ARBITRÉ 2026-08-16**  - [ ] servi aussi

> ✅ **L'offload PCIe reste la chose à battre, pas une chose qu'on livre.**
> C'était le plus lourd des trois : « servi aussi » aurait engagé le package A,
> donc le chantier MoE et le tier froid, sur un coût de miss **estimé**
> (0,35-0,75 ms) que le dépôt n'a jamais mesuré.
>
> **Trois conséquences, à ne pas redécouvrir.**
> 1. **Le package A n'est plus un engagement produit.** P2 et P6 (MoE) cessent
>    d'être sur le chemin critique : leur pause n'a plus besoin d'un déclencheur
>    de reprise, elle a une raison. Ce que P2 mesurerait — le hit d'un tier froid
>    en RAM hôte — porte sur une solution qu'on ne livre pas.
> 2. **Le point de verdict de P2 disparaît en l'état.** Le §0bis du
>    pré-enregistrement P2 range A6 comme portant **« l'architecture même du
>    test »** — le tier froid en RAM hôte *est* cet offload — et écrit que
>    « coché "référence seulement", P2 mesure une solution qu'on ne livre pas ».
>    ⚠️ **La VALEUR de α_verdict, elle, est fixée par A1, pas par A6** (32 Go ⇒
>    0,5868) ; A6 est l'un des trois arbitrages sans lesquels ce α est déclaré
>    « indéfendable », pas celui qui le calcule. Rouvrir P2 demandera donc de le
>    réécrire comme une **mesure de borne**, pas de le reprendre tel quel.
> 3. ⚠️ **Ce que ça ne dit pas** : que le MoE est sans intérêt. Il reste le seul
>    axe connu qui change la *classe* de modèle chargeable, et l'étude du
>    2026-08-12 tient. Ce qui est décidé, c'est qu'on ne le **sert** pas — donc
>    qu'aucun seuil de cette note ne dépend de lui.

## B. Le seuil, au triplet arbitré (barreau 32 GiB, 70B dense) — **fait foi**

`b_max = ((32 − marge − KV)×8 − 17,85)/68,45 + 0,2758` — embedding q8 facturé
**8,5 b/param** (comptabilité de la préreg du 08-11), convention 70B de
`fiche-4b` (68,45 Md quantifiés + 2,10 Md d'embedding), **KV q8 au ÷1,882
mesuré par P3** (et non ÷2), terme `+0,2758` = la correction **GiB** de A1.
Tous **calculés**. Contrôle : sans les deux corrections, la formule rend le
`S_alt` = 3,09 du pré-enregistrement du 08-13 (8k, marge 2, ÷2, 32 décimal) —
elle reproduit donc bien le calcul historique avant de le corriger.

| contexte | KV q8, marge 2 Go | **KV q8, marge 5 Go** | KV f16, marge 5 Go |
|---|---|---|---|
| **8k** | 3,35 | ✅ **3,00 — LE SEUIL** | 2,86 |
| 16k | 3,19 | 2,84 | 2,54 |
| 32k | 2,85 | 2,50 | 1,91 (infaisable) |

🚨 **Ce que le seuil arbitré fait au portefeuille de layouts : aucun ne le
passe, sauf celui dont le décodeur est mort.** Comparaison en **b/poids noyau**,
la seule comptabilité dans laquelle `b_max` a un sens.

| layout | b/poids noyau | vs `b_max` 3,00 | provenance |
|---|---|---|---|
| `Planes14` — **le servi** | 4,804 | **+60,0 %** | mesuré, banc CUDA (2,18 Go lus) |
| `Planes12x` | 4,342 | +44,6 % | mesuré |
| `e1c12` aligné | 4,288 | +42,8 % | calculé ([`mesures/e1c12-aligne-2026-08-16.txt`](mesures/e1c12-aligne-2026-08-16.txt)) |
| `Golay70` v2 | 3,589 | +19,5 % | mesuré (1,63 Go lus) |
| E3 `golay_tight` | 3,0444 | +1,4 % ⚠️ | **borne basse de tassage**, pas un stride payé |
| E1v | 2,398 | **−20,1 %** | mesuré (1,09 Go lus) — **décodeur fermé, 0,25× FP16** |

**Trois lectures, et elles ferment des branches plutôt qu'elles n'en ouvrent.**

1. **Le 70B dense sur 32 GiB n'est pas atteignable par le portefeuille
   actuel**, et pas de peu : le layout servi est à +60 %. Le meilleur écarté
   (`Golay70`, tué sur la **vitesse** à 1,77× contre 2,0×) est encore à +19,5 %
   — **son argument mémoire, resté orphelin depuis le 2026-08-11, est donc
   tranché ici : il ne suffisait pas non plus.** La branche se referme sur
   l'axe où elle n'avait jamais été jugée.
2. **E3 ne se rouvre pas, et pour une meilleure raison qu'avant.** Son 3,0444
   n'est qu'à +1,4 % du seuil, ce qui *paraît* rouvrir un enterrement prononcé
   contre 2,60. Il n'en est rien : ce chiffre est une **borne basse de
   tassage**, pas un stride payé. Dans la même table
   ([`mesures/radixstudy-x4-2026-08-12.txt`](mesures/radixstudy-x4-2026-08-12.txt)),
   la géométrie `perslot (= Planes)` vaut **3,2060** là où le banc mesure
   `Planes14` à **4,804** : **1,598 b/poids d'écart sur le même objet**. La
   borne d'E3 est déjà au-dessus du seuil ; son coût implémenté serait bien
   au-delà. Le journal le dit d'ailleurs de lui-même — « la colonne groupé32
   est OPTIMISTE ». ⚠️ **Ne jamais comparer une colonne de `radixstudy` à un
   `b_max`** sans cette conversion : ce serait la faute de comptabilité que
   tout ce document existe pour empêcher.
3. **Le seul format qui franchit le barreau est E1v** (−20,1 %), et son
   décodeur en ligne rend 0,25× FP16 — fermé le 2026-08-16. C'est le résumé
   exact de la quinzaine : *le format qui rentre ne se décode pas, le format
   qui se décode ne rentre pas*, et les quatre routes tentées ont toutes buté
   sur le **calcul**, jamais sur les octets.

⚠️ **Deux réserves sur cette comparaison, à ne pas lisser.** (i) Les b/poids
sont mesurés sur le **4B** ; `b_max` est dérivé pour un **70B dense**. La
transposition est une approximation — l'ordre des écarts (+60 % contre +19 %)
y survit largement, une décision à quelques pour cent n'y survivrait pas.
(ii) `b_max` suppose le modèle **entièrement en VRAM** : A6 ayant écarté
l'offload comme solution servie, c'est bien l'hypothèse retenue.

🆕 **Le barreau EST franchi — par un concurrent, pas par le portefeuille
(F2, 2026-08-21).** C'est un fait produit, et il manquait à cette page :

| point | b/poids noyau | vs `b_max` 3,00 | provenance |
|---|---|---|---|
| **QTIP 2 bits** — ⚠️ **CONCURRENT, hors portefeuille** | **2,000** | **−33,3 %** | mesuré, même banc, même processus et mêmes formes que `Planes14` ([`mesures/f2-p3-qtip-banc-2026-08-21.txt`](mesures/f2-p3-qtip-banc-2026-08-21.txt)) |

**Un code 2 bits concurrent passe donc le barreau que notre portefeuille ne
passe pas**, et il le passe avec un décodeur qui tourne — c'est ce qui le
sépare d'E1v, seul point de la table du §B sous `b_max` (−20,1 %) mais dont le
décodeur est fermé à 0,25× FP16.

⚠️ **Deux asymétries déclarées d'avance et non corrigées, à porter avec le
chiffre.** (i) QTIP ne porte **ni queue f32 ni échelle de ligne f32** — ses
cinq formes se divisent toutes par 16 [*calculé* sur les noyaux du journal],
donc il n'a **pas de queue à payer** là où `Planes14` facture la sienne dans sa
comptabilité noyau. Les 2,000 et les 4,804 sont bien dans la **même**
comptabilité, mais pas sous les mêmes charges. (ii) Le payload du banc est
pseudo-aléatoire — licite pour un code à débit fixe et un noyau sans branche
dépendante des données, mais **aucune phrase de qualité** ne peut s'appuyer sur
ce bras : il ne dit rien de ce qu'un client perdrait.

🔎 **Et la raison pour laquelle NOUS ne descendons pas est nommée, structurelle,
et ce n'est pas un défaut de format.** Un codebook de **1,1e14 points** ne tient
dans aucune table de correspondance, là où un état de treillis de **16 bits** y
tient. Notre index de réseau doit donc être **déplié** en flux de plans de bits,
à 4,80 b/poids noyau : **le dépliage est imposé par la TAILLE du codebook**, pas
par un choix d'encodage qu'un travail de format viendrait corriger
(`paper/sections/layouts.tex`). Pour cette page, la conséquence est nette :
l'obstacle au barreau n'est pas un layout mieux tassé qui resterait à écrire, il
est dans la nature du code. **L'arbitrage produit est inchangé — `b_max` = 3,00,
aucun layout du portefeuille ne le passe — c'est la compréhension de l'obstacle
qui progresse, pas son emplacement.**

> 🔎 **Ce que le barreau désigne comme suite, en creux.** Aucun travail de
> *format* ne referme +60 %. Les seules issues au segment 70B dense sont donc
> **hors format** : une carte plus grande, un modèle plus petit, ou l'axe MoE —
> que A6 vient de ranger en référence. **C'est une décision produit, pas un
> chantier technique**, et elle n'appartient pas à cette page.
>
> 🕳️ **Ce paragraphe a porté « et le plancher mesuré le 2026-08-16 borne de
> toute façon le gain de vitesse d'un format à 4,77× (dont `Planes14` capture
> déjà 2,16×) » — MESURÉ FAUX le 2026-08-21, et le mécanisme compte plus que la
> correction.** Le lot F2 a porté **QTIP** dans notre propre banc : un seul
> processus, 7 rounds dont 2 jetés, bras entrelacés, rapports formés round par
> round
> ([`mesures/f2-p3-qtip-banc-2026-08-21.txt`](mesures/f2-p3-qtip-banc-2026-08-21.txt)).
> QTIP 2 bits rend **2,246 ms [2,245–2,248], 0,91 Go lus, 2,0000 b/poids noyau,
> 405 Go/s, 4,89× FP16 [4,89–4,90]** là où `Planes14` rend, *dans le même
> processus*, **5,103 ms, 2,18 Go, 4,804 b/poids noyau, 2,15×** — tous
> *mesurés*. **Un noyau réel passe donc au-dessus du prétendu plafond**, et il
> le passe de la façon la plus nette possible : **t(QTIP) 2,246 ms < t(nullk)
> 2,306 ms**, c'est-à-dire plus vite que notre propre passe qui ne lit **aucun
> octet de poids** (séparation 2,7 % contre une résolution 2R = 0,72 %). En
> rendement d'octets, **f = 61,1 %** contre un plafond **pré-enregistré** de
> 59,6 % : dépassement de 1,5 point pour un δ de 0,2. Le tampon `.ots` interdit
> d'éditer le pré-enregistrement, donc l'**erratum est consigné dans la
> mesure**, pas dans la preuve.
> ⇒ **La formulation juste n'est plus « plafond absolu de tout travail de
> format » mais « plancher de NOTRE géométrie de lancement »** — un warp par
> ligne de sortie, 252 lancements de cette géométrie. Le 4,77× décrit notre
> grille, pas la machine ; les « 39 % latence/occupation » de l'attribution du
> 08-05 sont dans le même cas. Le papier est corrigé (`main.tex:112`) ; cette
> page l'est ici.
> ✅ **Ce que l'erratum ne change PAS, et c'est ce qui compte pour une note
> produit** : le barreau du §B est un barreau de **bits**, pas de vitesse.
> Aucune ligne de la table ci-dessus ne bouge, `b_max` reste **3,00 b/poids
> noyau**, et « aucun travail de format ne referme +60 % » tient tel quel. Un
> plafond de vitesse démenti n'ouvre pas une porte mémoire.

### B bis. Ce que le barreau ADMET — la question à l'endroit

Le §B dit ce qui ne rentre pas. Retourné, il dit ce que le produit **est
aujourd'hui**, et c'est le chiffre qui manquait à ce dossier.

Au triplet arbitré, la carte laisse **27,93 Go pour les poids** (32 GiB − 5 Go
de marge − 1,435 Go de cache q8 à 8k), soit 223,4 Gbit :

| à ce b/param modèle entier | modèle maximal | provenance |
|---|---|---|
| **5,162** (mesuré au 4B — borne **haute**, son embedding pèse 9,7 %) | **43,3 Md** | mesuré ([`mesures/rtbits-planes-8b-2026-08-09.txt`](mesures/rtbits-planes-8b-2026-08-09.txt)) |
| 4,878 (grand modèle, embedding ~2 %) | 45,8 Md | calculé |

✅ **Le produit servi par `Planes14` + embedding q8 sur une 5090, c'est donc un
modèle dense jusqu'à ~43-46 Md de paramètres, à 8k de contexte.** Qwen3-32B
tient avec de la marge ; le 70B ne tient pas, et aucun format connu ne l'y fait
tenir.

🎯 **Conséquence directe sur le plan, et elle n'était pas vue** : le point
**32B** de la Phase 3 n'est pas seulement « le dernier point de la courbe
d'échelle ». C'est **la plus grande classe de modèle que le barreau arbitré
admette** — donc le seul point de qualité qui porte sur l'objet réellement
servi. Les 4B, 8B et 14B mesurés sont des points de courbe ; le 32B serait le
produit.

🚨 **Mais le chemin SERVI au 32B est muré — par une arithmétique, pas par un
budget.** Le noyau de rotation met en mémoire partagée l'activation d'entrée de
`down_proj`, une f32 par coordonnée, quel que soit le dtype d'entrée. À
`intermediate_size` = 25 600, cela fait **102 400 octets demandés** contre
**101 376 offerts à l'opt-in**, ce dernier *mesuré* sur la carte servante
(L40S — l'attribut est **lu**, et non plus dérivé, depuis la campagne 14B :
[`mesures/rot-partagee-14b-2026-08-17.txt`](mesures/rot-partagee-14b-2026-08-17.txt),
[`mesures/fusedrun-14b-2026-08-17.txt`](mesures/fusedrun-14b-2026-08-17.txt)).
**Il manque 1 024 octets — un kibioctet sur cent — et c'est la réserve du
driver : aucun rognage côté hôte ne les récupère.** Le garde refuse le 32B
plutôt que de corrompre en silence, et c'est le comportement voulu.

> ⚠️ **Ce n'est PAS un mur de format**, et le confondre avec le §B serait la
> faute que ce document police. La rotation est la même pour `Planes14`,
> `Planes12x`, `Slot32`, `Golay70` et E1v — tous décodent des poids qui vivent
> dans la base tournée et appellent le même `rot_apply` : aucun verdict de
> format ne déplace ce mur, et aucun ne l'évite. Ce n'est pas non plus une part
> du plancher du 2026-08-16, qui chronomètre 252 projections **rotation
> exclue** ; les rapprocher demanderait de refaire l'attribution, pas de
> reporter un nombre.
> 🔎 **La seule piste nommée n'est pas chiffrée** : passer les activations
> partagées en **demi-précision**. Son coût est numérique et non structurel —
> à n = 25 600 la transformée est m = 1 024, et la profondeur d'accumulation
> d'une Hadamard vaut ~**14,6 étages**, qui se feraient alors en demi-précision.
> C'est un arbitrage numérique, il n'est **pas tranché**, et rien dans le dépôt
> ne l'a mesuré.
> ⚠️ **Pour cette page, la conséquence est de portée, pas de seuil** : le §B bis
> dit ce que le barreau **admet en bits** (~43-46 Md), et le 32B y entre
> largement. Ce mur-ci dit que le chemin **servi** n'y va pas encore sur la
> carte où nous mesurons. Les deux énoncés sont vrais et ne se soustraient pas.

⚠️ **Deux réserves.** (i) Le 5,162 est mesuré sur le 4B ; un 32B a un embedding
proportionnellement plus petit, donc un b/param **meilleur** — la borne de
43,3 Md est conservatrice, ce qui joue dans le bon sens. (ii) Ce compte est une
occupation de poids, pas une garantie d'exécution : la marge de 5 Go d'A4 est
précisément ce qui doit absorber activations et contexte CUDA, et rien dans le
dépôt ne l'a mesuré sur cette carte.

## C. Les métriques canoniques (définitions, et le piège payé derrière chacune)

| métrique | définition exacte | le piège déjà payé |
|---|---|---|
| **b/poids noyau** | bits/bloc × 0,0414723 + 0,1590915 (4B ; épinglé par le test 112 → 4,804) | ne se compare jamais à un b/param modèle entier |
| **b/param modèle entier** | octets totaux ×8 / paramètres totaux, **embedding compris** (q8 = 8,5) | le « 5,51 contre 4,50 » à deux dénominateurs — faute grave de l'errata du lot A |
| **VRAM** | mesurée sur carte (`nvidia-smi`), jamais projetée sans l'étiquette *estimé* | |
| **tok/s** | médiane du **rapport formé round par round**, 7 rounds dont 2 jetés, tous les bras dans le même processus | jamais un quotient de deux minima ; les ms dérivent (2,03–2,08 sur binaire identique), les octets non |
| **ppl** | wikitext-2 test, ctx 4096, **empreinte de tokens identique imprimée des deux bras**, même dtype | f32 vs f16 mélangés ; empreintes non vérifiées |
| **MMLU** | **micro** (= papier), f16, 2 280 questions, graine figée ; la barre d'une **différence** est la **SE appariée mesurée** (`bin/mmlupair`, bootstrap stratifié par matière), jamais le ± de ligne — et elle vaut **0,43 pp** entre deux précisions du *même fichier*, **0,79 à 1,44 pp** entre *modèles différents* (table sous celle-ci) | le macro sur-pondérait les matières STEM ×2,5 ; et le « σ McNemar 0,4-0,6 pp » que cette ligne posait comme définition n'avait **jamais été calculé** |
| **provenance** | chaque nombre étiqueté **mesuré / calculé / estimé**, avec sa comptabilité | trois chiffres peuvent décrire le même objet sans se contredire ; les comparer entre eux, si |

> 🕳️ **Corrigé le 2026-08-16 — la ligne MMLU donnait comme *définition* un
> seuil jamais calculé.** Elle posait « σ des différences = McNemar
> (0,4-0,6 pp) ». Ce 0,4-0,6 vient du §3ter de `CLAUDE.md`, où il est écrit
> « à 3-8 % de discordance » : une **estimation**, présentée ici en référence.
> Elle est mesurée depuis, et **il en faut deux, pas une, parce qu'il y a deux
> objets** :
>
> | ce qu'on compare | discordance | **SE appariée** du Δ micro stratifié | journal |
> |---|---|---|---|
> | le **même fichier** à deux précisions de cache (KV f16 ↔ q8, 4B) | 1,3 % (29 questions sur 2 280) | **0,43 pp** | [`mesures/kvq8-4b-2026-08-15.txt`](mesures/kvq8-4b-2026-08-15.txt) |
> | deux **modèles différents** (f16 ↔ AWQ ↔ LLVQ, 4B et 8B, six paires) | 7,0 à 27,7 % | **0,79 à 1,44 pp** | [`mesures/mmlupair-4b-8b-2026-08-13.txt`](mesures/mmlupair-4b-8b-2026-08-13.txt) |
>
> **C'est la discordance qui sépare les deux valeurs, et c'est elle qui les
> rend compatibles** — ce ne sont pas deux mesures qui se contredisent, ce sont
> deux objets. Un A/B à fichier constant ne déplace que 29 questions ; changer
> de modèle en déplace des centaines. Le seuil hérité était donc **plus large**
> que la barre réelle du premier cas (0,4-0,6 contre 0,43 mesuré : conservateur,
> il aurait dilué un effet réel) et **trop étroit d'un facteur 1,3 à 3,6**
> — grandeur dérivée — pour le second. Une constante unique ne pouvait pas
> servir aux deux.
>
> ⚠️ La SE se lit **sur le Δ micro stratifié**, celui que cette note publie. Le
> journal imprime aussi une SE **non pondérée** (0,21 à 0,91 pp) : c'est la
> quantité que McNemar teste, pas celle qu'on publie. Les échanger reproduirait
> exactement la faute macro/micro de la colonne de droite.

Règles de mesure héritées, non négociables : pré-enregistrement dans
`proofs/` (+ `ots stamp`) **avant** toute mesure qui décide ; `oracle` à
chaque nouveau backend (42 s, ~1 centime) ; `--features fast-linalg` partout
où l'on paie ; un A/B qui se trompe de bras doit **échouer**, pas retomber en
silence sur un défaut.

> ⚠️ **La première de ces règles n'est pas tenue partout, et cette note ne peut
> pas être la référence des pré-enregistrements sans le dire.** Inventaire
> vérifié le 2026-08-16 dans [`proofs/README.md`](../proofs/README.md) : sur
> onze pré-enregistrements, **deux** portent un tampon posé **avant** leur
> mesure *et* attestant les octets courants (`p1`, `p1c`) ; deux sont tamponnés
> **après** leurs mesures (`p1b`, `p5`) ; quatre n'ont **aucun** tampon
> (`2026-08-13`, `p2`, `p3`, `p4`), plus celui d'E1v par décision explicite de
> l'opérateur ; et les deux ancres de 08-10 / 08-11 **ne correspondent plus au
> fichier courant**, qui a été édité après elles. Le seul mécanisme du dépôt qui
> ait tenu cette règle est un **garde dans le binaire** (`bin/rankbench`), pas
> une phrase dans un document.

## D. L'environnement technique

**Local (0 $) — Mac M3 Max, ~69 Go, Metal.** C'est là que tout se tranche
d'abord : exactitude par sweep intégral des artefacts scellés, bancs
`decode`/`thesis`/`matvec`, `radixstudy` (largeurs), `ppl`/`mmlu` sur Metal
(~7× le CPU). Boucle de test : `cargo test` (debug, rapide) ; la suite
`--include-ignored` se compte en **dizaines de minutes** et balaie le fichier
scellé.

**Carte louée — HF Jobs (`ops/run.py`), L40S / rtx-pro-6000.** Coûts type,
tous déjà payés une fois : banc 7 rounds ~0,2-0,5 $ ; quantification 8B
11,5 $ ; 32B ~62 $ ; **70B jamais fait, estimé 150-200 $** (extrapolation
avec la marge que la règle impose — le 32B avait été sous-estimé de 25 %).
`fusedrun` A/B tokens gloutons ~0,3 $.

**Artefacts scellés disponibles** : 4B (`~/llvq-q4b.llvq`, 981 Mo,
150 681 600 blocs) ; 8B (`~/q8b-c12.llvq`, 1,82 Go, 288 571 392 blocs).
Empreintes de tokens de référence : `3f1baca9033bf251` (ppl),
`65dcd53655e8bfa5` (MMLU).

**Qualité de référence mesurée** (la seule qu'on ait — les layouts ne
changent pas le contenu décodé, donc elle vaut pour tout format servant le
même fichier) :

| | 4B | 8B | **14B** |
|---|---|---|---|
| MMLU (f16 → LLVQ) | 70,32 → 55,59 (−14,73 pp) | 76,08 → 65,52 (−10,56 pp) | **78,97 → 72,12 (−6,85 pp)** |
| ppl (dégradation) | ×1,385 | ×1,220 | **×1,189** |

> 🚨 **Ces lignes sont le résultat d'UN choix de fenêtres de calibration, et
> F5 a mesuré ce que ce choix vaut (2026-08-19).** Trois runs **complets** du
> 4B, `LLVQ_CALIB_SEED` ∈ {1, 2, 3}, même corpus, même codebook, même rotation,
> même empreinte de tokens `3f1baca9033bf251`, **21,45 $** *mesurés* : ppl
> scellé f16 **16,7425 / 15,8836 / 15,1027**, étendue **1,6398 ppl = 10,3 %**
> de la médiane, **σ (n = 3) = 0,8202 ppl = 5,2 %**, et les **trois** paires
> appariées sont **RÉSOLUES** (t = +4,54 / +10,92 / +7,68)
> ([`mesures/f5-graines-4b-2026-08-19.txt`](mesures/f5-graines-4b-2026-08-19.txt)).
> **Ce n'est pas du bruit de mesure** : le tirage des fenêtres de calibration
> déplace réellement la qualité de l'objet livré.
> 🕳️ **Le σ que ce dossier utilisait comme règle de travail — 0,7 %, et « tout
> effet sous ~1,5 % est du bruit » — est donc faux d'un facteur ~7 à la taille
> publiée** : il avait été mesuré sur **3 blocs de Qwen3-0.6B**, pas sur
> l'objet servi.
> ✅ **Le contrôle qui décide de la portée pour cette page** : les trois graines
> rendent **2,0702 b/poids effectifs et 1,771 Go scellés**, identiques au
> fichier publié. **Seule la qualité bouge — ni le débit, ni les octets.** Donc
> **ni le §B, ni `b_max`, ni la table b/param du §B bis ne sont touchés**, et
> l'arbitrage produit ne bouge pas d'une case.
> ⚠️ **Trois limites, sans lesquelles on sur-corrige.** (i) Les **A/B à fichier
> constant** — KV q8 d'A3, layouts runtime, embedding q8 — **ne recalibrent
> pas** : cette variance ne les touche pas, leur barre reste l'intervalle
> apparié à **±0,12 %**, et **A3 tient**. (ii) Les artefacts 4B/8B/14B ont
> **tous** tourné **sans graine**, donc sur le même préfixe contigu : la courbe
> d'échelle ci-dessus **compare des objets calibrés identiquement** et ne porte
> pas cette variance. (iii) Ce qui est réellement touché, c'est le **niveau
> absolu** de chaque dégradation — ×1,385 est la valeur d'**un** choix de
> fenêtres, non privilégié — et **toute expérience qui recalibre**. Deux
> verdicts du lot B (oracle −1,6 %, courbe de volume −1,2 %) tombent désormais
> **sous le plancher de bruit** sans que leur conclusion « le volume de
> calibration est plafonné » soit renversée : elle reposait sur des effets trop
> petits, ce que F5 confirme plutôt qu'il ne l'infirme.
>
> 🕳️ **Corrigé le 2026-08-16 : « Aucun chiffre de qualité n'existe au-delà du
> 8B » était faux le jour où cette note a été écrite.** Le point 14B est mesuré
> depuis le **2026-08-10**, dans le même harnais et sur la même empreinte de
> tokens `65dcd53655e8bfa5`
> ([`mesures/campagne-14b-qualite-2026-08-10.txt`](mesures/campagne-14b-qualite-2026-08-10.txt),
> synthèse dans [`echelle-4b-8b-2026-08-08.md`](echelle-4b-8b-2026-08-08.md)),
> et l'excès de perplexité y fond de 43 % du 4B au 8B puis de **14 %
> seulement** du 8B au 14B. Une note produit qui l'ignore extrapole depuis deux
> points.
>
> 🚨 **Cette note ajoutait « il plie la courbe » et « la tendance que le
> troisième vient de ralentir » — RETIRÉ le 2026-08-17 (matin), et rendu le
> soir SUR LA SEULE PERPLEXITÉ. 🚨 Le verdict dépend de la métrique, et une
> note produit qui n'en nomme aucune se trompe de moitié.**
> Sur l'**écart MMLU au 4 bits** — la métrique qui décrit ce qu'un client
> perd — la chute d'un palier au suivant se teste depuis que les trois écarts
> sont appariés : **4B→8B −6,96 pp, p = 0,0001, résolue** ; **8B→14B −1,40 pp,
> SE 1,68, p = 0,40, NON résolue**. Le ralentissement n'y est pas séparé par les
> barres, et p = 0,40 ne prouve pas l'égalité non plus : les données sont
> **muettes** sur ce palier.
> Sur la **perplexité**, il est **RÉSOLU** : pas 4B→8B ×0,881211
> [0,856 ; 0,907], pas 8B→14B ×0,974855 [0,959 ; 0,991], différence appariée
> **−0,100992 [−0,137670 ; −0,064313], t = −6,06**
> ([`mesures/ppl-appariee-4b-2026-08-17.txt`](mesures/ppl-appariee-4b-2026-08-17.txt)).
> 🕳️ La phrase « côté perplexité le pas 4B→8B n'est pas barrable, le journal de
> la campagne 4B est une synthèse » était vraie le matin et **démentie le
> soir** : les NLL par fenêtre vivaient dans les logs du job.
> **Pour une note produit, la conséquence ne change pas : ne pas extrapoler du
> tout, et traiter le 32B comme la mesure qui tranche.** Ce qu'un client achète
> se juge sur les capacités, où le ralentissement reste non résolu — et un
> genou de perplexité ne se transporte pas sur MMLU.
>
> ⚠️ **Ce qui reste vrai** : aucun chiffre de qualité au-delà du **14B**, et
> **aucun sur un MoE** — c'est toujours le verrou du package A (gate X5).
>
> ✅ **En revanche « le 14B n'a pas de ligne mémoire » est LEVÉ le 2026-08-17.**
> Cette note écrivait « aucun b/param modèle entier n'existe pour lui dans le
> dépôt, donc il n'entre pas dans l'arbitrage VRAM du §B ». L'artefact scellé
> n'avait jamais été rapatrié, mais il dormait dans le bucket : relu pour 0 $,
> il rend **5,106 b/param modèle entier** (`Planes14` + embedding q8) contre
> **5,404** pour l'AWQ officiel, soit **−5,5 %**
> ([`mesures/rtbits-14b-2026-08-17.txt`](mesures/rtbits-14b-2026-08-17.txt)).
> **Le 14B entre donc dans l'arbitrage VRAM du §B.** ⚠️ Deux réserves qui
> comptent pour une note produit : (i) la marge **n'est pas monotone** — 4B
> −2,6 %, 8B −10,6 %, 14B −5,5 % — et suit la **part de l'embedding** (9,7 % ·
> 15,2 % · 10,5 %), pas la méthode ; (ii) 🚨 **le point (ii) disait « ni la
> vitesse ni la VRAM carte n'ont jamais été mesurées à 14B, donc ce point n'a
> pas le troisième instrument » — DÉMENTI le 2026-08-17 (soir)** : le 14B est
> servi, **42,9 tok/s dans 9,39 Go** contre **17,0 dans 29,54** au bras dense
> ([`mesures/fusedrun-14b-2026-08-17.txt`](mesures/fusedrun-14b-2026-08-17.txt)),
> et le troisième instrument rend **5,0866 b/param**, à **−0,38 %** du 5,106
> ci-dessus. ⚠️ Pour une note produit, deux réserves demeurent : le brut ×2,53
> **ne se cite jamais seul** (son dénominateur est notre bras dense handicapé,
> et le handicap est maximal à cette taille), et **aucun rapport à tête
> identique n'existe au 14B**.
>
> ✅ **Le DISQUE du 14B, lui, est acquis** — et il l'était sans qu'aucune surface
> le dise : `qwen3-14b-llvq.bin` pèse **6 506 354 741 o = 6,506 Go** (*mesuré*,
> confirmé à l'octet par `hf buckets ls` **et** par le log de scellement). Pour
> une note produit c'est la cellule qui décide du **transport** et du stockage
> client, et elle n'est pas vide. 🕳️ **Ce point finissait sur « le triptyque du
> 14B est donc disque acquis, vitesse manquante, VRAM carte manquante : deux
> cellules à combler, pas trois » — les deux ont été comblées le 2026-08-17
> (soir)** (42,9 tok/s, 9,39 Go, cf. le 🚨 ci-dessus). **Le triptyque produit du
> 14B est complet.**

## E. Les trois segments servis (détail : les trois packages de la passation)

| segment | package | carte | le verrou d'entrée |
|---|---|---|---|
| Chat/agent local sur MoE ~120B | A | 32 Go (+ RAM hôte 64 Go) | qualité 2 bits sur MoE : **jamais mesurée** (gate X5, ~25-55 $) |
| Extraction documentaire par lots, 70B dense | B | 24-32 Go | ✅ **la moitié banc est levée** (P1, 0 $) ; ⏳ **le chemin GEMM reste à écrire** |
| 70B dense interactif | C | 32 Go | ❌ **le décodeur E1v existe, il est exact, et il est fermé** — 0,25× FP16 sur carte |

> ✅ **Package B — le verrou « vitesse du décodeur d'archive (banc V0/V1, 0 $) »
> a rendu son chiffre le 2026-08-15**, sur pré-enregistrement horodaté avant le
> run ([`mesures/p1-rankbench-2026-08-15.txt`](mesures/p1-rankbench-2026-08-15.txt),
> 16 777 216 blocs, 18 rounds dont 3 jetés) :
>
> | bras | ns/bloc | seuil de kill posé d'avance | verdict |
> |---|---|---|---|
> | `cascade-archive` (le décodeur d'archive tel quel) | **10,8115** | 2,00 | ❌ |
> | `cascade-uniformisée` (même bits, même table, boucle uniformisée) | **1,7809** | 2,00 | ✅ |
> | `marche-binomiale` (24 créneaux, **pas un bloc** — cf. P1b) | 0,3101 | 1,50 | ✅ |
>
> **Uniformiser la boucle vaut un ordre de grandeur** — 10,81 → 1,78 ns sur les
> mêmes bits, la même table et la même recherche de classe. C'est ce que ce
> verrou demandait de savoir, et il est levé **côté banc**. Ce qu'il ne lève
> pas : le **chemin GEMM**, qui n'est pas écrit, et qu'aucun de ces bras ne
> mesure (un bloc par lane, chaîne sérielle — pas un matvec).
>
> ⏳ Et le bras E1v du même banc rend **0,6795 ns/bloc** pour le vrai flux,
> adressage à largeur variable compris (+1,2 %,
> [`mesures/p1c-e1v-flux-2026-08-15.txt`](mesures/p1c-e1v-flux-2026-08-15.txt)) :
> **vert contre le kill de 1,50, mais au-dessus du gate CUDA de 0,45** de P4
> §4.2 — donc il **n'achète aucun bras CUDA**. Le régime intermédiaire du §4.2
> le dit mot pour mot : le bras survit comme point de la courbe, il faut une
> idée neuve, pas un job. ⚠️ Le gate se lit **par bloc** et non par marche :
> P1b a mesuré 0,6735 ns/bloc là où le compte de pas prédisait ×1,002 sur les
> 0,3101 ci-dessus — la mesure rend **×2,17**
> ([`mesures/p1b-marche-bloc-2026-08-15.txt`](mesures/p1b-marche-bloc-2026-08-15.txt)).
>
> ❌ **Package C — le verrou d'entrée est périmé depuis le 2026-08-16, et dans
> le sens qui ferme le segment.** Il disait « le décodeur E1v n'existe pas ;
> vitesse totalement inconnue ». Les deux moitiés sont tombées le même jour :
> il **existe**, il est **exact sur carte** (pires erreurs 2,4e-8·Σ|w·x| sur
> 1 105 920 lignes, 79 registres, zéro spill), et sa vitesse est **connue** :
> **0,25× FP16 [0,25–0,25], 25 Go/s** — 44,253 ms de médiane contre 10,988 pour
> le FP16 et 5,100 pour `Planes14`, soit **8,7× plus lent que le layout servi**
> ([`mesures/e1v-cuda-2026-08-16.txt`](mesures/e1v-cuda-2026-08-16.txt), job
> `6a814ba31f5885ae605bcb55`, 0,85 $). Contre le plancher de **1,60×** des
> critères d'X3 publiés le 2026-08-12 : **manqué d'un facteur 6,4**.
>
> **Ce que la mesure ne retire pas** : le *format* tient sa promesse au bit près
> — **1,09 Go lus contre 2,18** pour `Planes14`, soit la moitié, sur la carte et
> sur le modèle publié (même journal). En comptabilité **b/poids noyau**, P5
> l'avait chiffré la veille sur les 150 681 600 blocs : **2,3877** en groupes
> globaux, **2,3983** en coupe alignée ligne, soit **+0,44 %** sur ce couple —
> ⚠️ le « +0,48 % » qu'imprime le journal est le surcoût relatif en
> **bits/bloc** (0,2571 / 53,7370), une autre base : ne pas coller l'un à
> l'autre
> ([`mesures/p5-cns-2026-08-15.txt`](mesures/p5-cns-2026-08-15.txt) — une
> **comptabilité**, pas une mesure de temps). Ce qui est mort est son **décodeur
> en ligne**, borné en **calcul**. Le format reste disponible **hors boucle**
> (disque, transport).
>
> 🚨 **Donc ce segment n'a plus de verrou d'entrée écrit, et en inventer un
> serait choisir la cible à la place de l'opérateur.** Ce que le dépôt sait
> depuis le 2026-08-16, et qui devrait entrer dans cet arbitrage : une passe de
> projections qui ne lit **aucun poids** coûte déjà **45,2 %** du bras servi
> ([`mesures/nullk-plancher-2026-08-16.txt`](mesures/nullk-plancher-2026-08-16.txt)),
> quand `Planes14` est à 2,16×. Le seul levier nommé qui vise ces 45 % est la
> famille `k` de P4 §2.6, et **elle n'est pas écrite**.
>
> 🕳️ **Cette phrase a porté « ce qui plafonne tout travail de format à 4,77×
> FP16 » — MESURÉ FAUX le 2026-08-21, et le sens de la correction OUVRE ce
> segment au lieu de le fermer.** Le lot F2 a porté QTIP dans notre banc, un
> seul processus, bras entrelacés : **t(QTIP) 2,246 ms < t(nullk) 2,306 ms**
> — le noyau concurrent fait le même travail de projection **en lisant 0,91 Go**
> et en moins de temps que notre passe qui n'en lit aucun (séparation 2,7 %
> contre une résolution 2R = 0,72 %) ; f = **61,1 %** contre un plafond
> pré-enregistré de 59,6 %
> ([`mesures/f2-p3-qtip-banc-2026-08-21.txt`](mesures/f2-p3-qtip-banc-2026-08-21.txt),
> erratum consigné au journal, le `.ots` interdisant d'éditer la preuve).
> ⇒ **Les 45,2 % restent mesurés et vrais — mais ils mesurent NOTRE géométrie
> de lancement** (un warp par ligne de sortie, 252 lancements), pas la machine.
> La formulation juste est « plancher de notre géométrie de lancement ».
> **Pour le package C, cela déplace la question** : ce n'était pas « le format
> ne peut plus rien gagner », c'était « **notre grille** ne le laisse pas
> gagner ». La famille `k` de P4 §2.6 n'est plus le seul levier concevable sur
> ces 45 % — la géométrie de lancement elle-même en est un, et **personne ne
> l'a écrit non plus**. ⚠️ Ce qui ne change pas : E1v reste fermé à 0,25×,
> borné en **calcul**, et rien ici ne le rouvre.
>
> ⚠️ **Dette de provenance déclarée sur E1v** :
> `proofs/preregistration-e1v-cuda-2026-08-15.md` **n'est pas horodaté**, par
> décision explicite de l'opérateur — son antériorité ne repose que sur une date
> de commit. Les **seuils**, eux, sont ceux d'X3 du 2026-08-12, antérieurs par
> un chemin indépendant.

### E bis. Ce que la vitesse du 4 bits, mesurée le 2026-08-17, fait aux trois verrous — **rien, et il faut savoir pourquoi**

**Aucun des trois verrous ci-dessus ne portait « la vitesse du 4 bits est
inconnue », et aucun ne bouge.** Ils sont écrits sur *notre* chemin — qualité
2 bits sur MoE, chemin GEMM à écrire, décodeur E1v fermé — et un chiffre du
concurrent ne les lève ni ne les durcit. La note est ici parce que les trois
segments sont des régimes **batch 1**, exactement celui que la mesure vient de
sonder pour la première fois côté concurrent.

**Le fait, et sa comptabilité.** Qwen3-4B AWQ dans **vLLM 0.26.0** (image
épinglée), L40S, batch 1, 128 tokens, prefill compris, médiane de 5 rounds :
**200,49 tok/s** [200,39 ; 200,61], contre son propre témoin f16 à **83,09**
([`mesures/awq-vllm-4b-2026-08-17.txt`](mesures/awq-vllm-4b-2026-08-17.txt),
job `6a830d53e55292eada79b600`, **0,11 $**).

🚨 **Ce chiffre ne se compare à AUCUN seuil de cette note, et surtout pas au
« ≥ 20 tok/s » du §A5.** Trois raisons, et chacune suffit :

1. **Il est dans une autre pile.** Le même job mesure le f16 de vLLM à
   **83,09** là où le nôtre rend 43,6 : l'écart bout-en-bout est dominé par
   **vLLM contre candle**, pas par le décodeur de poids. La seule forme licite
   est **intra-pile** — ×2,413 pour le 4 bits chez lui, ×1,12 pour nous chez
   nous — et **ces deux rapports ne se divisent pas**.
   > ⚠️ **Le ×1,12 est un POINT, et B2 lui a donné sa plage le 2026-08-18** :
   > à tête identique le 4B rend **×1,11 [1,11–1,11]** (48,3 contre 43,5 tok/s,
   > médianes sur 5 rounds,
   > [`mesures/b2-fusedrun-plages-2026-08-18.txt`](mesures/b2-fusedrun-plages-2026-08-18.txt)).
   > L'écart au ×1,12 est un arrondi, aucune conclusion n'en dépend — mais la
   > règle « une plage, pas un point » du §C s'applique à nos propres nombres
   > avant de s'appliquer à ceux du concurrent. 🔎 **Et B2 ajoute un fait
   > produit** : à tête identique la série est **strictement croissante avec la
   > taille** — ×1,11 (4B) → ×1,29 (8B) → ×1,41 (14B) — là où la série brute
   > (×2,00 · ×2,57 · ×2,55) n'a aucun ordre. **C'est la série à tête identique
   > qui se publie.**
2. **Il est sur un 4B**, pas sur les 70B denses et le MoE ~120B que les trois
   segments servent. Rien dans ce dossier n'autorise à le transporter d'une
   taille à l'autre.
3. **Il ne majore pas** ce que l'AWQ sait faire : M = 1 n'est pas le régime
   optimal d'une GEMM Marlin (plus petite tuile en M = 8).

⚠️ **Et il n'ajoute rien à la colonne mémoire du §B** : vLLM **préalloue**, donc
ce qu'il rapporte est une *réservation*, pas une occupation. Le barreau du §B
reste dérivé de nos octets comptés.
