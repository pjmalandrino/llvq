# Golay70 — projections d'échelle et analyse du goulot de décodage

**Version 0.1, 2026-08-11.** Suite directe de
[`spec-apres-awq-2026-08-10.md`](spec-apres-awq-2026-08-10.md) : ce document
(1) relit le spec, (2) projette `Golay70` sur les modèles plus grands dans la
seule comptabilité qui compte (b/param modèle entier), (3) analyse le goulot
du noyau et propose une piste **neuve, à format inchangé**, qui subsume les
deux pistes notées dans le spec.

**Aucune mesure nouvelle ici.** Tout chiffre est étiqueté *mesuré* /
*calculé* / *estimé*, règle n°3 du §7 de `CLAUDE.md`. Les projections sont
des calculs sur des mesures existantes plus une hypothèse de transfert
explicitée en §2.3 ; l'analyse perf est un audit du source CUDA plus un
compte d'instructions, pas un profil.

---

## 1. Relecture du spec — ce qui tient, et quatre corrections

**Ce qui tient, et qu'il ne faut pas réargumenter :**

- La méthode du §3 est la bonne : le critère de 1,6× n'est pas effacé, il est
  périmé par une évidence neuve (AWQ 584 Go/s), et le critère de remplacement
  se pose **avant** de remesurer. Rien à redire.
- Le fait central du §4 est exact et vérifié sur le source : `Golay70` à
  198 Go/s n'est **pas** memory-bound — 30 % de sa borne d'octets quand
  `Planes14` en atteint 65 % et AWQ 88 %. Le goulot est dans le noyau,
  donc réparable en principe.
- L'ordre des lots (critère → sélecteur de bras → optimisation) respecte le
  pré-enregistrement. C'est la seule façon de rouvrir E2 sans tricher.

**Quatre corrections, par ordre d'importance :**

1. **Le « 4,016 b/param » du §2 refait l'erreur d'embedding que
   `rtbits` §3 documente.** La formule du spec
   (`(bpw·N_lin + 8·N_emb) / N_total`) facture le q8 à 8,0 bits ; le chemin
   `LLVQ_EMBED=q8` stocke int8 **plus** une échelle et un biais f16 par
   groupe de 64, soit **8,5 b/param** — c'est le mutant M12 de rtbits, qui a
   déjà survécu une fois à sa suite de tests. Le chiffre correct est
   **4,065 b/param** (calculé, formule de rtbits, validée au millième sur
   les 4 cellules 4B/8B connues — §2.2). Conséquence : le seuil « ≤ 4,1 » du
   critère neuf est toujours atteint, mais à 0,035 près et non 0,084, et la
   marge vs AWQ est **−23,3 %**, pas −24,2 %. Le critère survit ; la marge
   de sécurité du critère est plus mince qu'écrit.
2. **Le « ~88 → ~55 tok/s » du §7 est un majorant, pas une projection.** Il
   reporte le rapport de banc (÷1,61 sur les linéaires) sur le débit modèle
   entier. Or le token du 4B coûte ~11,3 ms dont **5,1 ms de linéaires**
   (mesuré : le banc et `fusedrun` portent le même objet) ; remplacer 5,1
   par 8,2 ms donne **~14,4 ms, soit ~69 tok/s** (estimé, additivité
   banc→modèle supposée — deux processus différents, à vérifier par
   `fusedrun`). L'arbitrage réel du §7 est donc plutôt « 24 % de mémoire
   contre ~22 % de débit », pas 37 %. Et si l'optimisation de §3 tient,
   l'arbitrage disparaît (linéaires ≈ `Planes14` → ~88 tok/s inchangés).
3. **La piste 2 du spec (« ne payer le XOR que du côté pair ») est caduque.**
   Elle date de la reconnaissance, avant l'implémentation : le noyau livré
   (`llvq_golay.cuh`) ne ré-encode **aucun** XOR depuis les générateurs — il
   lit `cwtab[f.g]`, une table de 16 Kio résidente en L1, et le mot de code
   est requis **sur les deux cosets** (résidu côté pair, signe
   `cbit ^ flag` côté impair). Il n'y a rien à économiser là.
4. **Le goulot n'est pas la divergence, c'est le compte d'instructions par
   slot.** Le décodage est déjà entièrement prédiqué (trafic identique sur
   les deux cosets, c'est documenté dans l'en-tête du `.cuh` et c'est vrai) ;
   ce qui coûte, c'est que le chemin prédiqué **paie les deux cosets en ALU à
   chaque slot**. L'audit d'instructions (§3.1) rend un ratio ~1,56 contre
   `Planes14` — le temps mesuré est ×1,61. La « spécialisation des warps par
   coset » (piste 1 du spec) attaque donc le bon coût mais par le moyen le
   plus cher ; §3.2 propose d'obtenir le même effet sans toucher ni au
   format ni à la grille.

---

## 2. Projections — Golay70 aux échelles où la mémoire est l'argument

### 2.1 Méthode

Trois étages, chacun ancré sur une mesure :

- **b/poids noyau** (comptabilité du banc CUDA) :
  `payload + (32 − payload)·f_queue + 32·(lignes/poids)`, la forme close
  validée par rtbits au 4B **et** au 8B (elle y explique les 0,052 b/poids
  d'écart poste par poste). Payload `Golay70` = `3,0 + 6·f_exc`
  (9 o/bloc + 144 bits par exception), qui rend 3,4461 au 4B — recoupé par
  `classhist` — puis 3,589 en noyau, **le chiffre du banc au millième**.
- **b/param modèle entier** : formule de rtbits §3, embedding q8 à
  **8,5** b/param, normes en f16. Validée au millième sur les quatre
  cellules mesurées (4B/8B × Planes14/Planes12x).
- **AWQ déployé** : le taux linéaire résolu depuis les deux points mesurés
  est **identique des deux côtés** — 4,1562 (4B) et 4,1557 (8B) b/poids —
  donc la projection AWQ = `(4,156·N_lin + 16·(N_emb + normes)) / N_total`
  est un fit à deux points qui n'a pas de liberté. (calculé)

Les comptes d'architecture (N_lin, embedding, normes, queues, lignes) sont
dérivés des configs et **vérifiés exactement** là où une mesure existe :
N_lin 4B = 3 633 315 840 ✓, N_lin 8B = 6 945 767 424 ✓, normes 196 096 /
308 224 ✓, et le par-couche 32B = 487 587 840 ✓ (le dé-risquage du 03-08 :
4 blocs = 1 950 351 360).

### 2.2 Le tableau

**b/param modèle entier, embedding q8 (8,5), normes f16** — la seule
comptabilité comparable à un chiffre AWQ. Poids seuls, hors KV et
activations. 4B/8B : *calculé sur fichiers mesurés* ; 14B/32B : *calculé,
artefact non encore quantifié* ; 70B (Llama-3.3) : *estimé, autre famille*.

| | 4B | 8B | 14B | 32B | 70B |
|---|---|---|---|---|---|
| `Golay70` b/poids noyau | **3,589 (mesuré)** | 3,535 | 3,487 | 3,487 | 3,474 |
| `Golay70` b/param | **4,065** | 4,290 | 4,016 | **3,725** | **3,624** |
| `Planes14` b/param | 5,162 ✓rtbits | 5,322 ✓rtbits | 5,106 | 4,886 | 4,807 |
| `Planes12x` b/param | 4,744 ✓rtbits | 4,929 ✓rtbits | 4,691 | 4,444 | 4,357 |
| AWQ déployé b/param | **5,302 (mesuré)** | **5,956 (mesuré)** | 5,404 | 4,719 | 4,509 |
| **marge `Golay70` vs AWQ** | **−23,3 %** | **−28,0 %** | **−25,7 %** | **−21,1 %** | **−19,6 %** |
| marge `Planes14` vs AWQ | −2,6 % | −10,6 % | −5,5 % | **+3,6 %** | **+6,6 %** |
| marge `Planes12x` vs AWQ | −10,5 % | −17,2 % | −13,2 % | −5,8 % | −3,4 % |
| `Golay70` Go poids | 2,04 | 4,39 | 7,41 | **15,3** | **32,0** |
| `Planes14` Go poids | 2,60 | 5,45 ✓mesuré carte | 9,43 | 20,0 | 42,4 |
| AWQ Go poids | 2,67 | 6,10 | 9,98 | 19,3 | 39,8 |
| FP16 Go | 8,04 | 16,4 | 29,5 | 65,5 | 141 |

*(La marge 4B `Planes14` −2,6 % diffère du −3,5 % du spec pour la même
raison que la correction n°1 : embedding à 8,5, pas 8,0.)*

### 2.3 Les trois lectures

1. **`Planes14` perd son argument mémoire à l'échelle — il passe DERRIÈRE
   l'AWQ à 32B.** Son avance actuelle (−10,6 % au 8B) est un artefact de
   l'embedding : l'AWQ porte deux tables f16 (15 % des poids au 8B) que
   notre q8 divise par deux. Quand l'embedding retombe à ~5 % (32B) puis
   ~3 % (70B), la comparaison converge vers payload contre payload — et là
   `Planes14` (4,67–4,71) **perd** contre AWQ (4,156) : +3,6 % à 32B,
   +6,6 % à 70B. Le « sweet spot » apparent du 8B est le point où l'AWQ est
   le plus handicapé, pas celui où nous sommes les meilleurs.
   `Planes12x` survit, mais à −6 %/−3 % — sous toute marge défendable.
2. **`Golay70` est le seul layout dont la marge est structurelle : −20 à
   −28 % à toutes les échelles.** Sa marge vient du payload (3,45 contre
   4,156), pas de l'embedding, donc elle ne se dilue pas. Et c'est
   précisément aux échelles où `Planes14` capitule que la mémoire redevient
   l'argument (§7 du spec : « le 4B est le véhicule de mesure, pas
   l'argument ») : à 32B, 15,3 Go contre 19,3 (AWQ) et 20,0 (`Planes14`) —
   sur une carte 24 Go, c'est 8,7 Go de marge KV contre 4,0 ; à 70B,
   **32,0 Go tiennent sur une seule L40S/A6000 48 Go avec 16 Go de marge**,
   là où `Planes14` (42,4) n'y laisse rien et où l'AWQ (39,8) y est à
   l'étroit. En 24 Go × 2, seul `Golay70` laisse une marge de service.
3. **Le critère du spec doit être re-dérivé par modèle, pas reporté.** Le
   « ≤ 4,1 b/param » encode « ≥ 20 % de marge vs AWQ » **au 4B**. En
   absolu, le 8B projeté (4,290) le viole — alors que sa marge y est la
   plus large du tableau (−28,0 %). L'invariant à pré-enregistrer est la
   **marge ≥ 20 % vs l'AWQ déployé du même modèle**, le 4,1 n'en étant que
   l'instance 4B.

### 2.4 Les réserves, avant qu'elles ne soient découvertes par d'autres

- **L'hypothèse porteuse est le transfert du taux d'exceptions E2**
  (7,4357 %, mesuré au 4B seulement). Ce qui la soutient : le taux
  d'exceptions `Planes12x` (L = 5) transfère du 4B au 8B à 0,010 pp,
  l'histogramme de niveaux entier transfère (30,7/65,8/3,4 des deux côtés),
  et le partage pair/impair est 50,0/50,0 sur les deux artefacts. Ce qui
  manque : la composante « pair violant » (4,05 % des 7,44 %) n'a été
  comptée qu'au 4B. Sensibilité : f_exc à 6 % → 3,643 b/param au 32B, à
  9 % → 3,814. **La conclusion (−15 à −24 % vs AWQ au 32B) survit à toute
  la plage plausible.** À trancher pour ~0 $ : `rtbits` sait déjà compter
  les classes violantes — une passe sur `qwen3-8b-llvq.bin` clôt la
  question avant tout engagement.
- **14B/32B : les artefacts n'existent pas encore** ; les lignes suivent
  l'histogramme mesuré, pas un fichier. **70B : autre famille (Llama)**,
  distribution de classes jamais observée hors Qwen3 — c'est la ligne la
  plus fragile du tableau, donnée pour l'ordre de grandeur de la thèse
  souveraineté, pas pour publication.
- **Ces projections ne disent rien de la qualité.** Le layout est
  bit-exact (E2 exact, 1 105 920 lignes vérifiées) : il hérite du déficit
  MMLU tel quel. Le pari qualité est l'axe d'échelle (−14,73 pp → −10,56 pp
  du 4B au 8B), et il est **indépendant** de ce dossier-ci — mais les deux
  paris pointent vers la même échelle : c'est à 32B que `Golay70` a une
  marge mémoire et que le déficit MMLU serait le plus resserré, si la
  tendance tient. Deux points ne font pas une loi ; le point 14B en cours
  et un éventuel 32B trancheront.
- **La vitesse projetée hors 4B est une hypothèse de proportionnalité** :
  coût ALU par poids constant, donc le 1,34× (ou son successeur optimisé)
  transfère en rapport. Les formes changent (down_proj sans queue à 8B) ;
  rien ne remplace un banc à la largeur visée.

---

## 3. Le goulot, et comment le fermer sans changer le format

### 3.1 Diagnostic : un compte d'instructions, pas une divergence

Audit du chemin chaud (`golay70_slot_value` contre le slot `planes_dot`),
en opérations entières par slot, `bj` étant une constante d'immédiat dans la
boucle déroulée (estimé — compte au source, le compilateur peut fusionner en
LOP3 ; l'ordre de grandeur est contrôlé par la mesure juste en dessous) :

| | tests de bit | sélections valeur | chemin de signe | total |
|---|---|---|---|---|
| `Planes14` | 4 (p0,p1,p2,smask) | 4 | 1 | **~9** |
| `Golay70` | 3 (cw,a,bm) | 4 (dont `hi = odd ? b : c`) | **7** (`sel = hi<<1|a`, `flags>>sel`, `&1`, `^`, `odd ? … : b`, négation) | **~14** |

Ratio ~1,56 — le temps mesuré est **8,223 / 5,111 = 1,61×**. La cohérence
de ces deux nombres est le diagnostic : le noyau est borné par l'**émission
d'instructions entières du décodage**, pas par la mémoire (30 % de sa borne
d'octets), pas par la divergence (tout est prédiqué), pas par `cwtab`
(16 Kio, L1), pas principalement par les exceptions (~1 ms sur 8,2, borné
par le delta `Planes12x`↔`Planes14` rapporté à 2,2× d'exceptions). Le
surcoût est concentré dans le **chemin de signe du coset impair et les deux
sélections `odd ?`** — payés par chaque slot, des deux cosets.

À noter : toute la famille est dans ce régime — `Planes14` ne convertit que
65 % de sa borne d'octets là où AWQ (4 valeurs par octet, décodage trivial)
en convertit 88 %. `Golay70` n'est pas d'une autre nature, il est juste plus
loin sur la même courbe ALU.

### 3.2 Piste A (neuve) : hisser la logique de coset au niveau bloc, en mots de 24 bits

> ✅ **Implémentée le jour même** (v2 de `llvq_golay.cuh`, ce commit) : zéro
> octet de format changé, identité prouvée par les trois verrous du harnais
> (probe hôte slot par slot, référence Rust bloc par bloc, records
> hand-packed), **3 mutants tués sur le prologue** (XOR du mot de signe, mux
> de flags croisé, mot haut croisé — chacun fait échouer la suite, le code
> restauré la repasse). ⚠️ Le balayage de l'artefact scellé a **sauté** sur
> la machine du port (fichier absent) — à repasser sur le Mac de dev — et
> **rien ici n'est une vitesse** : registres, spill et millisecondes restent
> au lot C, sur carte.

L'observation : **tout ce qui distingue les cosets par slot peut se calculer
une fois par bloc, sous forme de trois mots de 24 bits.** Le slot devient
alors exactement un slot `Planes14` — à trois masques au lieu de quatre.

Prologue par bloc (~10–15 ops entières, amorties sur 24 slots) :

```
// mk = diffusion 24 bits du bit k de r.flags (0x000000 ou 0xFFFFFF)
t_lo = (m1 & f.a) | (m0 & ~f.a)          // 1 LOP3
t_hi = (m3 & f.a) | (m2 & ~f.a)          // 1 LOP3
F    = (t_hi & f.bm) | (t_lo & ~f.bm)    // 1 LOP3 — F_j = flag[sel_j]
H    = odd ? f.bm : cw                   // mot « bit haut de sélection »
N    = odd ? (cw ^ F) : f.bm             // mot de signe
```

Par slot, ensuite :

```
hbit = H & bj;  abit = f.a & bj;  nbit = N & bj;        // 3 AND immédiats
v = hbit ? (abit ? v3 : v2) : (abit ? v1 : v0);          // 3 sélections
v = nbit ? -v : v;                                        // 1
```

soit **~7 ops par slot contre ~14** — et contre ~9 pour `Planes14`, qui
teste quatre masques là où le flux principal `Golay70` n'a que ≤ 4 niveaux,
donc trois. Correction : côté impair, `neg = cbit ^ flag[sel]` devient
`(cw ^ F)_j` avec `F_j = flag[(bbit_j<<1)|abit_j]` — l'algèbre est
l'identité, vérifiable slot par slot ; côté pair `H = cw`, `N = smask`,
c'est le code actuel réordonné. Les entrées paires ont `flags = 0`, donc
`F = 0` et le prologue est correct **sans branche**.

Propriétés qui comptent pour ce dossier :

- **Zéro octet de format changé.** Ni le flux 9 octets, ni les exceptions,
  ni `GolayClassRec` (les `mk` se dérivent de `flags` en 4 ops, ou se
  précalculent dans les deux `u32` de padding du record). Toute
  l'infrastructure de preuve existante — bijection, E2 exact, référence
  f64 — reste valable telle quelle.
- **Testable localement pour 0 $** : le harnais hôte
  (`tests/host_golay70.cpp` via clang++, `golay70_decoder_matches_rust.rs`)
  exécute *le même texte* que le noyau ; l'identité algébrique se prouve sur
  les 150,7 M blocs du 4B scellé sur le Mac de dev, avant toute carte.
- **Elle subsume la piste 1 du spec.** Après hissage, le code par slot est
  **identique sur les deux cosets** — il ne reste rien à spécialiser par
  warp : la spécialisation économiserait ~2 sélections *par bloc* au prix
  d'un format éclaté en deux sous-flux avec index de colonne. À ne
  considérer que si A, mesurée, ne suffit pas — et alors probablement
  enterrer E2 plutôt que payer ça.

### 3.3 Ce que ça peut rendre — et le critère qui le jugera

Bornes, dans l'ordre (toutes **estimées**, c'est exactement ce que le lot C
doit mesurer) :

| scénario | linéaires 4B | vs FP16 | provenance |
|---|---|---|---|
| aujourd'hui | 8,22 ms | 1,34× | **mesuré** |
| ALU ÷ (14/7), passe principale seule, exceptions ~1 ms inchangées | ~4,6–5,3 ms | **~2,1–2,4×** | estimé, compte d'instructions |
| plancher : memory-bound au rythme famille (428 Go/s sur 1,63 Go) | 3,8 ms | 2,9× | calculé (borne, §4 du spec) |

La fourchette utile est **1,9–2,4×** : le compte d'instructions n'est pas un
profil (le §2c de `CLAUDE.md` le rappelle : « le profileur n'a jamais été
utilisé »), les FMA et les chargements ne bougent pas, et l'émission
INT/FP32 se partage les ports sur Ada. Le critère pré-enregistré du spec
(≥ 2,0× **et** marge mémoire ≥ 20 %) est donc **plausible mais pas acquis**
— ce qui est précisément la situation pour laquelle on pré-enregistre.

Deux pistes secondaires, examinées et fermées d'avance :

- **Réduire les exceptions « pair violant »** (4,05 % des 7,44 %) : il
  faudrait un 3ᵉ niveau par résidu, donc un bit de plus par slot — le bit B
  est déjà le signe. C'est un changement de format contre ~0,5 ms ;
  rejeté.
- **Relectures d'activation de la passe d'exceptions** (~1,1 Go épars au
  4B) : elles sortent du L2 (une activation ≤ 39 Ko contre 100,7 Mo de
  L2), même argument que la réserve fermée du §1 du spec. Non poursuivi.

### 3.4 Plan de mesure — inchangé par rapport au spec, une étape ajoutée

Lots A (critère dans `proofs/`, avec la marge relative de §2.3 et
l'embedding à 8,5) et B (sélecteur de bras) tels quels. Lot C : la piste A
d'abord — preuve d'identité locale sur le 4B scellé, puis **un** job à sept
bras avec contrôle. Étape ajoutée, avant tout job : la passe `rtbits` des
classes violantes sur le 8B (§2.4), 0 $, qui transforme l'hypothèse de
transfert en compte.

---

## 4. Verdict

1. **Le spec a raison de rouvrir E2, et les projections renforcent son
   argument au-delà de ce qu'il écrit** : non seulement `Golay70` est le
   seul bras qui batte franchement l'AWQ déployé aujourd'hui, mais c'est le
   **seul dont l'avance existe encore à 32B/70B** — `Planes14` y passe
   derrière l'AWQ (+3,6 %/+6,6 %), `Planes12x` y survit sans marge
   (−6 %/−3 %). Si l'argument du projet est la mémoire à l'échelle, le
   portefeuille de layouts n'a en réalité qu'une seule carte, et c'est
   celle qui est marquée « écartée ».
2. **Le trou de perf a une cause identifiée et une attaque à format
   inchangé** : ~14 ops entières par slot dont la moitié se hisse au niveau
   bloc en mots de 24 bits (§3.2), prouvable localement pour 0 $. Fourchette
   estimée 1,9–2,4× — le critère de 2,0× se jouera à la mesure, pas au
   raisonnement, et c'est pour ça qu'il est pré-enregistré.
3. **Trois chiffres du dossier à corriger au passage** : le b/param
   `Golay70` 4B est 4,065 (pas 4,016 — embedding à 8,5), la marge
   `Planes14` est −2,6 % (pas −3,5 %), et le débit 4B projeté sous
   `Golay70` non optimisé est ~69 tok/s (pas ~55 — l'additivité
   banc→modèle, à confirmer par `fusedrun`).
