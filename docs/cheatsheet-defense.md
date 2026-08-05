# Cheatsheet — défendre le projet à froid

> Pour une conversation technique (Qualcomm, recrutement, conf). Chaque question :
> **la réponse à dire**, puis **si on creuse**, puis **la sortie honnête** quand tu
> atteins ta limite. Cette dernière n'est pas un aveu de faiblesse — c'est ce qui
> rend le reste croyable.
>
> ⚠️ Chiffres à resynchroniser après le run MMLU.

---

## Règle zéro : ouvre par ta faiblesse

**Ne laisse jamais quelqu'un d'autre sortir la comparaison au 4 bits.** Sors-la
toi, en premier, dans les deux premières minutes :

> « Sur un 4B, le 4 bits nous domine partout sauf le disque. C'est mesuré, même
> machine, même jour. Ce qui tient, c'est le noyau et le taux de change
> bits↔vitesse — pas le produit sur cette taille de modèle. »

Tout ce que tu diras après sera lu comme crédible. Si tu attends qu'on te le
demande, tout ce que tu as dit avant devient suspect.

---

## Les chiffres à connaître par cœur

| | |
|---|---|
| Baseline Qwen3-4B FP32, wiki, ctx 4096 | **12,2336** |
| Notre 4B quantifié | **16,9617** à **2,1696** b/poids (**×1,386**) |
| QTIP — le seuil à battre | **17,04** à 2,000 b/poids |
| Le noyau, 252 matrices, un token | `Slot32` à **5,510** b/poids : **2,09× le FP16**, plage **[2,05–2,11]** |
| RAM face au q4, **à convention identique poids seuls** | nous (`Slot32` + lm_head f16) **6,5245** b/poids · le q4 **4,5006** → **×1,45 contre nous** |
| Débit bout en bout | MLX q4 **129,8 tok/s** · nous **~78,5** (projeté) |
| Lignes vérifiées contre référence f64 | **1 105 920**, pire erreur 3,4·10⁻⁸ |

Si tu ne dois en retenir que trois : **16,9617 / 2,03× [2,03–2,10] / 6,5245 contre
4,5006.**

> ⚠️ **Deux précautions attachées à ces deux lignes, à ne jamais laisser tomber.**
>
> 1. **Le rapport de vitesse est formé round par round**, puis résumé par sa
>    médiane et sa plage sur les 5 rounds gardés — ce n'est **pas** un quotient
>    de deux minima, qui mêlerait deux rounds n'ayant jamais coexisté. Si ton
>    interlocuteur divise les millisecondes du journal, il trouvera 2,19 et pas
>    2,15 sur le bras `float4` : dis-lui d'où vient l'écart avant qu'il y voie
>    une erreur. Les ms dérivent d'un run à l'autre, les b/poids reproduisent au
>    chiffre — cite le b/poids et le rapport, renvoie au journal pour les ms :
>    `docs/mesures/k1-metal-2026-08-05.txt`.
> 2. **Ne jamais reposer « 5,51 contre 4,50 ».** C'est un mélange de métriques
>    que le dossier a déjà corrigé une fois : 5,51 est la comptabilité `thesis`
>    des **projections** (payload + bases + queue f32 + échelles de ligne f32),
>    4,50 est le q4 sur **tous** ses poids, embedding quantifié compris. La
>    forme homogène est celle du tableau : **6,5245 contre 4,5006, ×1,45 contre
>    nous** (`docs/fiche-4b.md` §5.3), et la fourchette selon le `group_size` et
>    le périmètre retenus est **×1,16 à ×1,53** (`docs/plan-de-test-v2-cuda.md`
>    §4). Le 5,51 reste juste **pour décrire notre format** (§4) ; il n'est pas
>    un terme de comparaison.

---

## 1. Pourquoi Λ₂₄, et pas une autre dimension ?

**À dire :**

> Quantifier, c'est découper l'espace en cellules et ne stocker que le numéro de
> la cellule. L'erreur, c'est la distance entre le point et le centre de sa
> cellule. À nombre de cellules fixé — donc à bits fixés — on veut des cellules
> aussi proches d'une sphère que possible, parce que la sphère minimise cette
> distance à volume donné. Or on ne peut pas paver l'espace avec des sphères. On
> prend donc le réseau dont la cellule s'en approche le plus. En dimension 24,
> c'est Λ₂₄, et son optimalité est un théorème.

**Si on creuse — pourquoi pas la dimension 48, ou 100 ?**

Deux raisons, une théorique et une pratique :

- **Le gain sature.** Plus on monte en dimension, plus on se rapproche de la
  limite de Shannon, mais les derniers points coûtent très cher. On mesure
  **92,23 % de la limite** en dimension 24 — le reste ne vaut pas le prix.
- **La structure.** Λ₂₄ se construit à partir du code de Golay [24,12,8]. C'est
  ce qui rend le test d'appartenance et l'énumération calculables. La plupart
  des réseaux en haute dimension n'ont aucune structure exploitable : la
  recherche du plus proche voisin y est intraitable. **On ne choisit pas la
  meilleure dimension dans l'absolu, on choisit celle où le décodeur est
  écrivable.**

*Analogie archi :* c'est le choix d'une fonction de hachage. Tu ne veux pas
« une » fonction en 24 dimensions, tu veux celle dont la structure te donne un
lookup rapide. Golay, c'est cette structure.

**La sortie honnête :**

> « Je ne suis pas théoricien des réseaux euclidiens. Ce que je peux te dire,
> c'est ce que j'ai vérifié : notre construction reproduit la série thêta et la
> somme cumulée N(13) = 280 974 212 784 720 exactement. C'est un verrou à
> 15 chiffres — aucune contrainte fausse ne le franchit. Le nombre de baisers
> tombe à 196 560 aussi. »

C'est une excellente réponse : elle admet la limite *et* démontre la rigueur.

---

## 2. Pourquoi l'encodeur coûte des heures et le décodeur des nanosecondes ?

**À dire :**

> Ce sont deux problèmes de nature différente. L'encodeur reçoit un vecteur
> quelconque et doit trouver **le plus proche** parmi 280 000 milliards de
> points — c'est une recherche. Le décodeur reçoit un index de 48 bits et doit
> produire le vecteur — c'est de l'arithmétique d'adresse. Il n'y a rien à
> chercher.

Les chiffres : **639 µs/bloc** pour l'encodeur sur un cœur, **0,158 ns/bloc**
pour le décodeur runtime sur GPU. Un rapport de **4 millions**.

*Analogie archi :* **c'est compiler contre exécuter.** Personne ne reproche à
gcc d'être plus lent que le binaire qu'il produit.

**Si on creuse — pourquoi c'est ça qui rend le projet viable :**

L'encodeur tourne **une fois par modèle, hors ligne**. On peut y passer 46 h sur
un 70B, ça n'a aucune importance. Le décodeur tourne **à chaque GEMM, à chaque
token**. Une nanoseconde par bloc s'y paie 150 millions de fois par passe.

D'où le principe de conception du projet : **ne jamais optimiser l'un en pensant
à l'autre.** C'est aussi ce qui explique la séparation archive/runtime (§4).

---

## 3. Pourquoi `Slot32` va 2× plus vite que `Sorted32` avec 0,6 bit de plus ?

C'est **ta meilleure question** — celle qui est purement systèmes, où ton métier
te donne un avantage sur un chercheur.

**À dire :**

> Ce n'est pas les bits. 0,6 bit d'écart, ça fait 13 % de trafic mémoire — ça ne
> peut pas expliquer un facteur 2. Ce qui l'explique, c'est que dans `Slot32`
> **chaque champ vit à un offset fixe**. Le décodeur devient 24 tours constants,
> sans boucle, sans état sériel, sans branche qui diffère entre lanes. Sur un GPU
> où 32 lanes avancent en lockstep, ça vaut 2×.

Le layout : `[classe 9][gain 1][smask 24][m₁..m₄ @ 24 bits]`. La largeur ne
dépend plus que de L. Pas de `nz` à calculer, pas de base de signes, pas de
niveau zéro à traiter.

**Si on creuse — pourquoi les formats compacts n'y arrivent pas :**

Dans les formats compacts, les masques sont **imbriqués** : le masque k ne
couvre que les slots que les niveaux précédents ont laissés libres. C'est
optimal en bits. Mais pour décoder le slot j, il faut savoir combien de slots
avant j ont déjà été pris — c'est un **popcount de préfixe en espace de rangs**,
intrinsèquement sériel, et différent d'un bloc à l'autre. Donc de la divergence.

*Analogie archi :* **c'est une struct à taille fixe contre du protobuf
varint-encodé.** Le protobuf est plus petit sur le fil ; la struct est plus
rapide à lire parce que les offsets sont des constantes de compilation. Sur du
SIMT 32-large, cette différence vaut 2×, pas 5 %.

**Le chiffre qui tue :** le décodage `Slot32` coûte **12 µs au-dessus du sol**
(le sol = mêmes loads mémoire, zéro décodage). Le format imbriqué en coûte
**~155**. **13×.** Ce n'est pas le trafic qui a bougé, c'est l'adressage.

**La sortie honnête :** trois réécritures du noyau ont échoué à contourner le
mur du popcount — sans branches 245 µs, niveaux packés deux passes 331,
demi-masques 189, toutes battues par le naïf à 206. C'est un mur mesuré, pas une
intuition.

---

## 4. Pourquoi 2,17 bits sur le disque deviennent 5,51 en RAM ?

**C'est la question centrale du projet. Elle doit sortir parfaitement.**

**À dire :**

> Le fichier stocke un **rang** — le numéro du point parmi 280 000 milliards,
> sur 48 bits. C'est le minimum théorique, on ne peut pas faire plus compact.
> Mais décoder un rang, c'est dérouler une chaîne de divisions sérielle : **8,27
> ns/bloc sur GPU**, contre 0,158 pour le format runtime. **52×.** À ce rythme,
> un 4B mettrait 1 252 ms par token rien qu'à décoder. C'est mort.
>
> Donc on transcode une fois au chargement — 3,1 s pour un 4B sur 12 cœurs — vers
> un format que le GPU décode dans l'ombre de la latence mémoire. Ce format aligne
> les champs à offsets fixes, et cet alignement coûte des bits.

**La phrase à retenir mot pour mot :**

> « Le fichier porte **2,17 bits d'information**. Le format runtime en dépense
> **5,51** pour les lire vite. Les 3,3 bits d'écart sont du **rembourrage
> d'adressage**, pas de l'information. »

*Analogie archi :* **gzip contre mmap.** On ne fait pas de requêtes sur un
fichier gzippé. La forme compressée est pour le stockage, la forme de travail
échange de la place contre de l'accès. Ce qui est nouveau ici, c'est qu'on a
**mesuré le taux de change** de bout en bout :

**Protocole A — `bin/matvec`, UNE couche (`gate_proj`), froid à 4 copies
rotatives.**

| b/poids RAM | 3,35 | 4,54 | 4,75 | **5,375** |
|---|---|---|---|---|
| vs FP16 | 0,68× | 0,90× | 1,04× | **2,21×** |

*Toujours valides : le contrôle de non-régression du 2026-08-05 les reproduit —
0,69× (`Grouped32`), 0,90× (`Flat32`), 2,20× (`Slot32`), 5,375 b/poids sur cette
couche. Des quatre b/poids de la ligne du haut, seul le 5,375 a été recoupé par
un chemin indépendant : il retrouve le 5,3756 de `bin/rtbits` sur le modèle
entier, comptabilité payload + bases.*

**Protocole B — `bin/thesis`, le MODÈLE ENTIER (252 projections), un command
buffer par bras, mémoire froide, 7 rounds dont 2 jetés, tous les bras dispatchés
à chaque round dans le même ordre ; le rapport est formé ROUND PAR ROUND, puis
résumé par sa médiane et sa plage sur les 5 rounds gardés.** b/poids en
comptabilité thesis : payload + bases + queue f32 + échelles de ligne f32,
**identique pour les quatre layouts**. 1 105 920 lignes vérifiées contre une
référence CPU f64, seuil 1e-5. Journal : `docs/mesures/k1-metal-2026-08-05.txt`.

| bras | b/poids | min ms | Go lus | Go/s | vs FP16 méd [plage] |
|---|---|---|---|---|---|
| FP16 (half4, scalaire) | 16,000 | 21,728 | 7,27 | 334 | 1,00× [1,00–1,00] |
| **LLVQ `Slot32` (scalaire@24)** | **5,510** | **10,496** | 2,50 | 241 | **2,03× [2,03–2,10]** |
| LLVQ `Flat32` | 5,256 | 23,807 | 2,39 | 99 | 0,91× [0,91–0,91] |
| LLVQ `Grouped32` | 3,498 | 31,634 | 1,59 | 50 | 0,69× [0,68–0,69] |
| FP16 (half4, float4) | 16,000 | 20,612 | 7,27 | 351 | 1,05× [1,04–1,07] |
| **LLVQ `Slot32` (float4@24)** | **5,510** | **10,126** | 2,50 | 252 | **2,14× [2,10–2,16]** |
| LLVQ `Slot32` (float4@28) | 5,510 | 10,081 | 2,50 | 248 | 2,13× [2,07–2,17] |

> ⚠️ **Si on divise les colonnes, on ne retrouve pas la dernière — et c'est
> voulu.** 21,728 / 10,126 fait 2,19 quand la colonne dit 2,15 : un minimum
> divisé par un minimum mêle deux rounds qui n'ont jamais coexisté. Le rapport
> est calculé à chaque round sur des mesures simultanées, puis médiané.
> **Dis-le avant qu'on te le demande.** Et rappelle que les ms dérivent d'un run
> à l'autre alors que les b/poids et les octets reproduisent au chiffre : le
> chiffre à citer de mémoire est le b/poids et le rapport avec sa plage.

*Pourquoi trois variantes de `Slot32` : le passage du scalaire au `float4` gagne
3,5 % sur LLVQ (10,496 → 10,126) **et 5,1 % sur FP16** (21,728 → 20,612). Ce qui
paie, c'est la largeur de chargement — un load de 128 bits au lieu de quatre de
32 — et elle paie **des deux côtés**, donc le rapport ne bouge pas : 2,04× en
float4 contre float4 (2,15 / 1,05), 2,09× en scalaire contre scalaire, chacun
dans la dispersion de l'autre. Le padding à 28 flottants, lui, ne gagne rien :
0,4 % plus lent que le `float4` dense, et sa plage [2,06–2,17] recouvre
entièrement celle du dense [2,12–2,19] par le haut — rien ne l'en distingue.
(Ces pourcentages et ce 2,04× sont des grandeurs **dérivées** de la table, pas
des lignes lues ; c'est licite parce que les deux termes viennent du même run et
de la même comptabilité.) Les deux variantes `float4` de `Slot32` sont
identiques au bit près au noyau scalaire sur les 1 105 920 lignes ; la variante
`float4` du bras FP16 ne l'est pas (3,1e-8 d'écart), sa somme étant écrite en
`+`/`*` et non en `fma` explicites — confondant déclaré, pas caché.*

> **Ne fusionne jamais les deux tableaux.** Les b/poids ne relèvent pas de la
> même comptabilité : le 5,375 de `matvec` recoupe le 5,3756 de `rtbits`
> (payload + bases) ; le 5,510 de `thesis` y ajoute la queue f32 et les échelles
> de ligne f32. Même objet, deux façons de compter les octets, toutes deux
> honnêtes — mais pas dans la même colonne. Et le layout à 4,75 / 1,04× du
> protocole A n'a **pas** de bras correspondant dans le protocole B : ne le
> cherche pas dans le second tableau.
>
> **Et si on te demande pourquoi 2,09× ici et 2,07× ailleurs dans le dépôt**
> (`format-noyau.md`, README) : le 2,07× vient du banc **à deux bras**
> antérieur, et il a sa propre dispersion — trois invocations
> consécutives du même binaire non modifié donnent **2,029× puis 2,050× puis
> 2,080×**, monotone, les deux bras accélérant ensemble, erreurs et octets
> identiques aux trois runs (`docs/mesures/thesis-temoin-2026-08-04.txt`). Le
> 2,07× publié est le **haut** d'une plage `[2,029 ; 2,080]`, reproduit au
> troisième run consécutif ; le 2,03× [2,03–2,10] du banc à sept bras est la
> même quantité mesurée avec la dispersion **incluse dans le protocole**. Les
> deux se recouvrent. Conséquence de protocole : **un effet de quelques pour
> cent ne peut pas être tranché en comparant deux invocations distinctes du
> binaire.**

**La conséquence que tu dois assumer sans broncher :** à convention identique
**poids seuls**, nous (`Slot32` + lm_head f16) pesons **6,5245 b/poids** contre
**4,5006** au q4, soit **×1,45 contre nous** (`docs/fiche-4b.md` §5.3) ; selon
le `group_size` et le périmètre retenus la fourchette va de **×1,16 à ×1,53**
(`docs/plan-de-test-v2-cuda.md` §4). Et même en descendant à **4,7083 b/poids**
avec `L ≤ 4` — mesuré le 2026-08-05 sur l'artefact réel par `bin/rtbits`,
comptabilité étroite payload + bases sur les **projections seules**, donc le
chiffre le plus favorable qu'on sache produire — on reste au-dessus des 4,5006
du q4, en ne portant que 2,17 bits de qualité. **À iso-mémoire, le q4 est
strictement meilleur sur un 4B.** Le créneau du 2 bits, c'est là où le 4 bits ne
rentre pas du tout — 70B sur 32 Go.

> ⚠️ **Attention à ne pas rejouer le mélange de métriques ici non plus.** Le
> 4,7083 et le 4,5006 ne sont pas comptés de la même façon (projections seules
> contre tous poids quantifiés, avec ou sans queue et échelles de ligne). La
> comparaison qui tient sans réserve est celle du haut, **6,5245 contre
> 4,5006** ; la ligne `L ≤ 4` sert à dire que même le plafond le plus flatteur
> ne passe pas sous le q4, pas à chiffrer un écart.

> 🕳️ **D'où venait le « 4,5 », et pourquoi il fallait le tuer.** Avant le
> 2026-08-05, le dépôt portait **trois chiffres pour la même quantité**, aucun
> mesuré : 4,5 dans cette fiche, 4,667 dans `docs/face-au-4-bits.md`, ~4,4 dans
> `docs/format-noyau.md`. Le 4,667 vient de `bin/lcap`, comptabilité qui
> facture à chaque bloc la largeur du plafond et **ne compte pas les bases** ;
> le ~4,4 était `106/24`, la largeur brute du bloc, ni arrondie à l'octet ni
> chargée de la base du groupe (généalogie complète dans `format-noyau.md`,
> section « le plafond L ≤ 4, compté ») ; le 4,5 de cette fiche est le seul
> des trois dont aucune dérivation n'est tracée dans le dépôt. Les trois ont
> été corrigés le même jour — **ne pas lire cet encadré comme si `~4,4`
> subsistait ailleurs**. La mesure est
> **4,7083**, et elle est un **majorant inconditionnel** — `L ≤ 4` implique
> `width_slot ≤ 9 + 1 + 24·4 = 106 bits = 14 octets`, donc un stride ≤ 14 o par
> groupe ; et il est atteint dès qu'un groupe porte un bloc à 4 niveaux, ce qui
> est le cas de **4 708 799 groupes sur 4 708 800**. Ce n'est pas une
> simulation, c'est un compte.
>
> **L'argument de cette fiche en sort plus fort, pas plus faible** : à 4,5 on
> était à égalité avec le q4, à 4,708 on est au-dessus. Dis-le toi-même avant
> qu'on te le sorte.

---

## 5. Ce que le β dit exactement, et ce qu'il ne dit pas

**À dire — et commence par la restriction, pas par le résultat :**

> Attention, ce n'est **pas** « on bat le papier ». Notre shape–gain 0 bit
> reproduit leur chiffre à deux dixièmes de point — 88,90 contre 89,12 — donc
> protocole et codebook sont identiques et corrects. L'écart n'existe que sur
> la variante où β est réglé.

*(88,90 et non 89,36 depuis le 2026-08-01 : le banc mesurait le gain sur la
projection, la production le mesure sur la norme du bloc — §A5.)*

**Ce que c'est :** β est le facteur d'échelle du réseau — en clair, le rayon de
la boule dans laquelle on quantifie. On l'ajuste sur un jeu d'entraînement
séparé. L'optimum est **étroit** :

| β | 0,300 | 0,325 | **0,350** | 0,375 | 0,400 |
|---|---|---|---|---|---|
| Rétention | 87,00 | 91,09 | **92,24** | 90,95 | 88,07 |

Le papier reporte 89,37 % : ça correspond à un β désaccordé d'environ ±0,04,
soit **~11 %**.

**Ce que ça dit :** leur chiffre de spherical shaping n'est probablement pas à
l'optimum, donc leur comparaison spherical shaping / shape–gain est peut-être
légèrement injuste envers le premier.

**Ce que ça ne dit pas — à énoncer toi-même :**

- **C'est un artefact de réglage, pas un meilleur code.** Même codebook, même
  protocole, bouton différent.
- C'est mesuré sur une **source gaussienne i.i.d.**, un seul harnais, une seule
  seed — pas sur de vrais poids après GPTQ.
- Le papier mesurait une *distance angulaire au plus proche voisin* sur une
  source radialement uniforme, nous une *rétention MSE* après quantifieur de
  gain. Deux métriques peuvent classer différemment : ce n'est pas une
  contradiction frontale.

**Pourquoi le dire quand même :** c'est spécifique, vérifiable de leur côté en
dix minutes, et ça prouve que tu as lu leur papier pour de vrai. C'est ton
meilleur ouvre-porte.

---

## Trois règles de survie

**1. Ne revendique jamais plus que ce que tu as mesuré.** Le projet documente
quatre pièges de mesure trouvés et corrigés — dont un banc où les buffers LLVQ
tenaient dans le cache système de 48 Mo pendant que le FP16 streamait la DRAM.
Sans cette correction, tu annoncerais 2,2× sur un chiffre faux. Raconte cette
histoire : c'est ta meilleure carte.

**2. Quand tu ne sais pas, dis-le et donne la source.**

> « Je ne sais pas. C'est mesuré dans `docs/format-noyau.md`, banc `decreal`, je
> te retrouve le chiffre exact. »

Un archi qui sait où sont ses mesures est plus crédible qu'un archi qui improvise
une réponse. Personne n'attend de toi que tu récites la théorie des réseaux.

**3. Sépare toujours trois choses**, et ne les laisse jamais se contaminer :

| | statut |
|---|---|
| **Le noyau** | contribution réelle, l'artefact que le papier dit ne pas avoir fait |
| **La reproduction** | saine, à parité, sans plus |
| **Le produit sur un 4B** | **réfuté, et c'est moi qui l'ai réfuté** |

---

## Ce qu'il ne faut jamais dire

| ❌ | ✅ |
|---|---|
| « On bat le papier » | « On reproduit le papier à iso-réglage ; notre avance sur β est un artefact de réglage » |
| « 2 bits par poids » | « 2,17 sur le disque, 5,51 en RAM — et c'est tout le problème » |
| « 5,51 contre 4,50 en RAM » | « à convention identique poids seuls, 6,5245 contre 4,5006, soit ×1,45 contre nous — le 5,51 décrit notre format, il ne se compare pas au q4 » |
| « 2,07× plus rapide » | « 2,03× [2,03–2,10] le FP16 sur les projections seules, rapport formé round par round ; contre du q4 bien réglé on est encore derrière — de combien, le dépôt ne le tranche pas, cf. `fiche-4b.md` §5.4 » |
| « ×4,63 de compression » | « ×4,63 sur le fichier, dont 44 % est un embedding non quantifié sur un 4B » |
| « Le noyau est fini » | « Le noyau est mesuré et vérifié, il n'est pas encore branché dans le runner » |

---

## Le pitch en trois phrases, si on te demande « c'est quoi ton truc »

> J'ai implémenté en Rust un papier de quantification 2 bits de Qualcomm, et
> j'ai écrit le noyau GPU fusé que les auteurs déclarent explicitement ne pas
> avoir fait — un décodeur de réseau de Leech multi-coquilles, **2,09× le FP16
> [2,05–2,11]** sur un modèle entier.
>
> En le mesurant proprement, j'ai trouvé le résultat que toute la littérature
> esquive : le format compact n'est pas décodable vite, et le format rapide
> occupe plus de RAM que du 4 bits ordinaire. La vitesse a été achetée avec les
> bits mêmes qui justifiaient le 2 bits.
>
> Ce qui reste vrai, c'est le noyau et le taux de change bits↔vitesse, que
> personne n'avait mesuré de bout en bout. Ce qui tombe, c'est l'intérêt du
> 2 bits en dessous de la taille où le 4 bits ne rentre plus.
