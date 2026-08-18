# Pré-enregistrement — les quatre tests à 0 $ qui jugent les pistes de sortie du verrou E3

**Date : 2026-08-13.** Écrit **avant toute mesure** : à cette heure, aucune des
quatre variantes ci-dessous n'existe dans le code, `bin/radixstudy` n'a jamais
été exécuté en ordre-fichier, l'adressage warp-scan n'a jamais été prixé pour
aucun layout, et le dump de routage
[`docs/data/moe-routing-gptoss20b-2026-08-12.json`](../docs/data/moe-routing-gptoss20b-2026-08-12.json)
n'a jamais été lu pour autre chose que ses totaux par cellule. Le dernier
verdict en date est celui du lot X
([`docs/mesures/radixstudy-x4-2026-08-12.txt`](../docs/mesures/radixstudy-x4-2026-08-12.txt)) :
E3 enterré à 3,0444 b/poids noyau contre un critère de 2,60.

> Il **hérite sans dérogation** des gardes du pré-enregistrement du
> [2026-08-10](preregistration-2026-08-10.md) (§7), de sa comptabilité
> d'octets (§6) et de sa règle de provenance — rien n'est recopié ici, c'est
> le même engagement.
>
> ⚠️ Ni signé GPG ni horodaté tant que l'opérateur ne l'a pas fait
> (`ots stamp proofs/preregistration-2026-08-13.md`, tag signé — §6.5 du plan
> de test). D'ici là l'antériorité repose sur la date de commit.

---

## 0. Ce que ces tests peuvent et ne peuvent pas conclure

Ils comptent des **bits**. Aucun ne mesure une vitesse, et la règle du dossier
tient : *aucune conclusion de vitesse ne survit sans SASS*. Un vert ici
n'ouvre qu'un droit à être mesuré plus tard ; un rouge, lui, ferme — c'est
l'asymétrie qui rend ces tests rentables à 0 $.

**Le verrou jugé** est celui du 2026-08-12 : « le point dans sa classe coûte
déjà 41,50 des 47 bits d'index ; il reste ~5,5 bits pour le choix de classe,
et toute variante qui pose un champ de classe explicite les repaie en 10 bits
d'en-tête ». La prémisse non énoncée de ce verdict est que toutes les
variantes prixées **linéarisent l'arrangement en champs par slot**. Les tests
T1–T3 attaquent cette prémisse, T4 attaque la fréquence de décodage.

## 1. La comptabilité, figée ici

Conversion noyau inchangée, celle de `radixstudy.rs:302-316`, épinglée par
`kernel_conversion_reproduces_planes14` (112 bits → 4,804) :

```
b/poids noyau = bits_par_bloc × 0,0414723 + 0,1590915      (Qwen3-4B)
```

**Deux modes d'adressage, et c'est le cœur du lot.** Un flux à largeur
variable ne se lit pas d'une seule façon, et le dossier n'en avait prixé
qu'une :

| mode | ce que le noyau paie par groupe de 32 blocs | déjà prixé ? |
|---|---|---|
| `grp32-max` | `32 × ⌈max_du_groupe/8⌉ × 8 + 32` — stride uniforme au bloc le plus large | oui, mais **en ordre classe-majeur**, donc optimiste (le binaire le dit) |
| `warp-scan` | `Σ largeurs_du_groupe + 32` d'offset de base, arrondi au mot de 32 bits — chaque bloc paie exactement sa largeur, les offsets par somme préfixe de warp | **non, jamais, pour aucune variante** |

**Interdits de citation**, opposables au rapport de sortie :

1. Ne jamais citer la colonne grp32 **classe-homogène** (l'actuelle, dont
   3,0444) dans un verdict — elle est optimiste par construction.
2. Ne jamais comparer un chiffre `warp-scan` à un seuil dérivé pour
   `grp32-max`, ni l'inverse.
3. Tout b/param modèle entier se dit embedding compris, q8 facturé **8,5**
   b/param (la correction du pré-enregistrement du 08-11 §2.1).
4. Étiqueter chaque nombre *mesuré* / *calculé* / *estimé*. Les largeurs sont
   **calculées** ; l'histogramme et l'ordre des blocs sont **mesurés** (ils
   sortent du fichier scellé) ; toute projection 70B est **estimée**.

## 2. Les seuils, et pourquoi ceux-là

| seuil | valeur | d'où il vient |
|---|---|---|
| `S_spec` | **2,60** b/poids noyau ⇔ **58,86** bits/bloc | le critère d'ouverture E3, `spec-memoire-extreme-2026-08-12.md:185-188`, inchangé |
| `S_alt` | **3,09** b/poids noyau ⇔ **70,66** bits/bloc | le même barreau 32 Go re-dérivé au triplet (marge 2 Go ; KV q8 ; 8k) avec l'embedding q8 à 8,5 : `((32−2−1,35)×8 − 17,85)/68,45`. **C'est un seuil candidat, pas le seuil du projet** — il ne devient opposable que si la note produit du §5 le retient |

⚠️ `S_alt` n'existe **qu'à contexte 8k figé** : le même calcul rend 2,93 à 16k
et 2,61 à 32k. Un résultat entre 2,60 et 3,09 ne se publie donc **jamais**
comme « ça passe » — il se publie comme « ça passe si et seulement si le
triplet produit retenu est (32 Go ; 2 Go ; KV q8 ; 8k) », et cette note
n'existe pas encore.

## 3. Les quatre tests, leurs prédictions et leurs règles de décision

Les prédictions sont celles de la contre-expertise du 2026-08-13, écrites
**avant** de mesurer. Elles sont là pour être démenties : une piste qui
atterrit loin de sa prédiction est suspecte même si elle est verte.

### T1 — `golay_tight` et tout le menu, en ordre-fichier réel

Ce que le binaire fait aujourd'hui : un histogramme par classe, puis un
groupage **classe par classe**. Ce qu'un noyau lit : 32 blocs **consécutifs
du fichier**, qui mélangent les 301 classes. T1 rend le groupage à l'ordre
réel, pour les deux modes d'adressage.

- **Prédiction** : `golay_tight` en `grp32-max` ordre-réel ≈ **3,40–3,62**
  b/poids noyau (contre 3,0444 en classe-homogène) ; en `warp-scan` ≈
  **2,90–3,00**. Mécanisme attendu : P(≥1 bloc à 5 niveaux dans un groupe de
  32) = 1 − 0,9663³² ≈ 0,67, et ces classes portent les largeurs 90–94.
- **Décision** : `grp32-max` ordre-réel > `S_spec` ⇒ **le verdict E3 du 08-12
  est confirmé et durci** (il l'était sur un chiffre optimiste ; il le sera
  sur le vrai). > `S_alt` aussi ⇒ le barreau 32 Go est fermé à cette famille
  même au seuil le plus permissif défendable, et la passation l'écrit.
- **Contrôle de non-régression obligatoire** : le mode classe-homogène doit
  reproduire **3,0444** au dix-millième sur `golay_tight` et **4,804** sur le
  contrôle `Planes14`. Sans ces deux, aucun chiffre du lot n'est lisible.

### T2 — `E1v` : le rang exact par classe sous adressage warp-scan

La variante que `radixstudy` n'a jamais eue : garder le rang dans sa classe à
sa largeur exacte (`⌈log₂|classe|⌉`, le champ `exact` qui existe déjà et vaut
41,50 en moyenne) + 10 bits d'en-tête, sous `warp-scan`. Deux sous-variantes,
et **c'est la seconde qui fait foi** :

- `e1v-packé` : un seul champ à la largeur du produit. Extraire les
  sous-rangs exige des divmod par constantes magiques ⇒ **pas shift-only**.
- `e1v-séparé` : un `⌈log₂⌉` par étage de la composition ⇒ zéro division.
  **C'est le seul point admissible**, et donc le seul qui se compare à un
  seuil.

- **Prédiction** : packé ≈ 53 bits/bloc → **2,36** ; séparé ≈ 54,5 →
  **2,42**. Sans warp-scan (`grp32-max`), 58–62 bits → 2,56–2,73, à cheval.
- **Décision (sur `e1v-séparé`, `warp-scan`, ordre-fichier réel)** :
  ≤ **58,86 bits/bloc** ⇒ vert, et l'axe bits rouvre avec une dette explicite
  (la vitesse de la marche binomiale, non jugée ici) ; > 58,86 ⇒ **l'axe des
  formats se ferme définitivement**, parce que le plancher absolu de toute
  variante à en-tête est 40,98 (entropie) + 10 + adressage ≈ 52,5 bits :
  E1v est à ≤ 2 bits de ce plancher, il n'y a pas de marche en dessous.
- **T2bis — la dette de provenance, réglée dans le même lot.** Le 41,50 est
  un chiffre **4B**. Le 8B scellé (`~/q8b-c12.llvq`) est sur la machine :
  refaire T2 dessus. **Kill si l'écart 4B→8B sur `e1v-séparé` dépasse
  +2 bits/bloc** — la distribution ne serait pas transportable et toute
  projection 70B tomberait, y compris celles déjà publiées.

### T3 — `golay_signs` : les signes reconstruits par le mot de Golay

Les cinq layouts servis dépensent 24 bits de masque de signes, y compris sur
les ~50 % de blocs impairs où les signes ne portent **aucune** information
(forcés par l'appartenance au codeword). Variante : 12 bits de message Golay
côté impair, `nonzero−1` côté pair.

- **Prédiction** : gain d'espérance −9,85 bits/bloc, mais un flux transposé
  paie le **max du groupe** et non l'espérance ⇒ **≈ 3,60** b/poids noyau en
  transposé (soit le point `Golay70` déjà mesuré à 1,77× et écarté), ≈ 3,42
  en records par bloc.
- **Décision** : la forme transposée doit rendre **≤ 3,45** b/poids noyau
  **et** améliorer `E1c12` d'au moins **6 bits/bloc net**. Sinon la piste est
  morte comme layout et ne survit que comme **brique d'E1v** (où ses bits
  sont déjà comptés dans les largeurs exactes de classe).
- Toute forme retenue doit expliquer, chiffre contre chiffre, pourquoi on la
  préférerait à `golay_tight`.

### T4 — MoE chaud/froid : le ciseau, sur le dump de routage mesuré

Sur `moe-routing-gptoss20b-2026-08-12.json`, calculer par couche
`α_min(ℓ)` = plus petite fraction de cellules chaudes telle que la masse
routée des cellules froides reste sous le budget de miss, puis le b/poids du
mélange `α·3,589 + (1−α)·2,219`.

- **Prédiction** : α_hit ≈ 0,85 contre α_VRAM ≤ 0,72 ⇒ **intervalle vide**,
  mélange ≈ **3,4–3,5** b/poids.
- **Décision** : mélange > **3,20** b/poids (le seuil qui rend 48 Go
  atteignable pour 117 Md totaux, embedding + KV ~1,2 Go déduits) ⇒ la
  variante **VRAM** est enterrée. La branche cache LRU est tuée par
  arithmétique pure (backing résident ⇒ `2,219 + α·3,589 ≤ 3,20 ⟺ α ≤ 0,27`)
  sauf hit LRU ≥ 99,8 % à α = 0,27, seuil figé **ici**.
- **Le test doit aussi chiffrer l'alternative dominante** — tier froid en
  **RAM hôte**, miss = memcpy PCIe — parce qu'un enterrement qui ne nomme pas
  ce qui le remplace n'est pas un verdict, c'est un abandon.
- ⚠️ Réserve de périmètre inscrite d'avance : le dump est un **20b à 32
  experts**, le dimensionnement vise un **120b à 128 experts**. Un vert ne se
  transporterait pas sans le dump du 120b ; un **rouge**, si, car la
  concentration ne peut qu'empirer en montant en nombre d'experts.

## 4. Ce qui invaliderait le lot entier

- Le contrôle de non-régression de T1 (3,0444 et 4,804) qui ne retombe pas.
- Une variante dont les largeurs ne sont pas vérifiées par un test contre
  `enumerate_classes` **et** contre `FastDecoder`, classe par classe.
- Un `e1v` dont la largeur serait inférieure à `⌈log₂|classe|⌉` pour une
  classe quelconque : ce serait une bijection impossible, donc un bug de
  comptage, pas une découverte. Test à écrire **avant** de lire le résultat.
- Tout écart de définition de « groupe de 32 » entre ce lot et la géométrie
  réelle d'`E1c` (`llvq-artifact/src/e1c.rs`) : à vérifier et documenter, pas
  à supposer.

## 5. Ce que ce lot NE tranche pas, et qui reste dû

1. **La vitesse.** Aucune ms ici. Les marches binomiales d'E1v, la cascade
   uniformisée, le dual-path pair/impair de T3 : tous non jugés.
2. **La note produit du barreau** — carte cible et sa génération PCIe,
   contexte servi, dtype du KV, marge, tok/s minimal. Sans elle, `S_alt` n'a
   pas de statut. Elle est **bloquante** pour toute dépense de carte, pas
   pour ce lot.
3. **Le budget GPU.** Aucun job n'est autorisé par ce document. Les ~0,2 $ du
   banc E1c (X3) restent gelés par l'arrêt formel de l'axe noyau.

---

*Écrit avant la première exécution. Les résultats vont dans
`docs/mesures/`, jamais dans ce fichier.*
