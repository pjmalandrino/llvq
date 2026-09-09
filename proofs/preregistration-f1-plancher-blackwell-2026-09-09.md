# Pré-enregistrement — le plancher de la table, rejoué sur Blackwell : la L1 est-elle la cause du décalage ?

**Écrit, commité et TAMPONNÉ le 2026-09-09, AVANT la mesure.** Go de
l'opérateur du 2026-09-09.

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Protocole inchangé.** C'est le rejeu exact de
`proofs/preregistration-f1-plancher-table-2026-09-04.md`, même binaire
`f1floorbench`, même échelle de huit empreintes, **une seule variable : la
carte**. Ce fichier n'ajoute qu'une prédiction signée et une lecture.

**Coût : ~0,01 $** (*estimé* ; le run L40S du 2026-09-05 a facturé 0,01 $ pour
18 s de calcul). Plafond de timeout 20 min, soit 0,92 $ au pire sur
`rtx-pro-6000`. Total projet avant : **134,22 $**.

---

## §1 — La question

Le banc de la tuerie a rendu deux verdicts opposés, chacun intra-processus :

```
  R = (v3g − nullk) / (planes14 − nullk)
      L40S       0,5824  [0,5801 ; 0,5865]   VERT
      Blackwell  1,4693  [1,4585 ; 1,4812]   ROUGE
```

La décomposition ferme l'arithmétique à 0,3 % (*calculé* sur des temps mesurés,
`docs/mesures/tetra48-tuerie-2026-09-08.txt`) :

```
  R(BW)/R(L40S)              = 1,4693 / 0,5824 = 2,523
  rapport(v3g)/rapport(planes) = 1,527 / 0,606 = 2,520
```

Deux paquets nets dans les rapports à tête identique Blackwell/L40S : `word`
0,623 et `planes14` 0,606 — bornés par la bande passante, qui s'améliore de
1,65× ; `v1` 1,497, `v3` 1,498, `v3g` 1,527 — **1,5× plus lents**. Ce que ces
trois-là ont en commun, et que `f1r` et `v2` n'ont pas, c'est d'avoir retiré la
conversion I2F qui masquait tout le reste.

**Donc les lectures de table coûtent 1,53× plus cher sur Blackwell.** Le
mécanisme reste ouvert. Ce banc le mesure isolé.

## §2 — Les trois hypothèses, et ce qui les distingue

| # | hypothèse | signature attendue sur `D(S)` |
|---|---|---|
| **H1** | **capacité L1** — 16 Kio de `rows` + 2 Kio de `branches` + 12 Kio de tuile = 30,4 Kio contre 24 pour Planes14 (*calculé*) ; la table déborde et chaque gather part en L2 | `D(8 Kio)` s'améliore, `D(16 Kio)` **régresse** ; le rapport `D(16)/D(8)` monte franchement |
| **H2** | **cadence soutenue** — Server Edition limitée en puissance ; mon banc n'imprime même pas l'horloge | les deux points régressent **proportionnellement**, rapport inchangé |
| **H3** | **remplissage de grille** — 188 SM contre 142, `k_proj` ne lance que 128 blocs | ce banc a sa **propre** géométrie ; si les deux points s'améliorent, la cause n'est pas dans la table et H3 monte |

Le discriminant est **`D(16 Kio) / D(8 Kio)`, formé à l'intérieur d'un seul
processus sur une seule carte**. Aucun × n'est divisé entre cartes : la règle 5
n'est pas sollicitée.

## §3 — Le protocole, verbatim

```
cargo run --release -p llvq-cuda --bin f1floorbench
```

sur `rtx-pro-6000`, image `hf.co/spaces/Pier-Jean/llvq-runner-cuda` au commit
courant. Rien d'autre ne change.

Contrôle repris tel quel du prereg d'origine : le point DRAM doit valoir ≥ 20 ×
la L2. Blackwell porte 128 Mio de L2, donc 4 Gio = **32 ×**. La garde
(`f1floorbench.rs:316`) passe — elle avait refusé de démarrer le 2026-09-04
quand elle valait 10,7 ×, pour 0,00 $.

## §4 — Référence L40S, *mesurée* le 2026-09-05

`docs/mesures/f1-plancher-table-2026-09-05.txt:45-53`, job `6a9b4b98`, 18 s :

```
  D(8 Kio)     0,344 ms  [0,343–0,351]
  D(16 Kio)    0,663 ms  [0,661–0,666]      rapport D(16)/D(8) = 1,93
  D(128 Kio)   4,666      plateau L2 plat à ±2 % jusqu'à 16 Mio
  D(4 Gio)    61,419
  Didx        −0,103
```

## §5 — Prédiction signée

| grandeur | prédiction | fourchette |
|---|---|---|
| `D(8 Kio)` | **0,27 ms** | 0,22 – 0,36 |
| `D(16 Kio)` | **1,00 ms** | 0,80 – 1,35 |
| **`D(16)/D(8)`** | **3,7** | **2,8 – 5,2** |
| plateau L2 (4 Mio) | 3,4 ms | 2,6 – 4,6 |
| point DRAM (4 Gio) | 38 ms | 28 – 50 |

**Je prédis H1.** Le raisonnement, pour qu'il soit réfutable : le journal L40S
écrit qu'à 16 Kio la table **rate déjà 7,6 % de ses accès** alors que 8 Kio
tient à 100 % ; si la portion données de la L1 est plus étroite sur Blackwell,
ce taux d'échec explose là où 8 Kio reste résident, et le rapport décolle. Les
deux termes de la prédiction sont indépendants : `D(8)` doit **s'améliorer**
d'environ le rapport d'émission (1,37×), `D(16)` doit **régresser**.

## §6 — La lecture, posée avant

Ce banc **ne décide rien** : il n'a pas de porte, c'est une mesure (prereg
d'origine §1, et la ligne « F1 floor » de `docs/ROADMAP.md:119` : *no gate*).

| lecture | ce qu'on en fait |
|---|---|
| `D(16)/D(8)` ≥ 2,8 et `D(8)` améliorée | **H1 retenue.** La table est la cause, et M3 (tuile 128→64, gratuit sur tous les critères) puis M6 (table 16→8 Kio) deviennent les correctifs désignés |
| rapport dans [1,7 ; 2,2] et les deux points régressés | **H2 retenue.** C'est la cadence ; rien à corriger dans le format, et il faut imprimer l'horloge avant tout autre chiffre |
| rapport dans [1,7 ; 2,2] et les deux points améliorés | **H1 et H2 tombent.** La cause n'est pas dans la table isolée : elle est dans la géométrie du banc de la tuerie, et H3 devient la piste |
| toute autre configuration | rien n'est décidé, et l'écart s'écrit |

## §7 — Limite de portée

Ce banc mesure **trois lectures de table dans sa propre géométrie**, pas notre
noyau. Le prereg d'origine l'écrit et rien ici ne l'assouplit : un tirage
uniforme sur toute la table est **pessimiste**, la distribution réelle des
étiquettes n'est pas mesurée, et `Dsm` ne se différencie jamais contre `nullk`.

Il ne dit rien du tok/s, rien de la qualité, rien du 8B.
