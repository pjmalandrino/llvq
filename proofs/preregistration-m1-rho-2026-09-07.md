# Pré-enregistrement — M1 : le correctif radial vaut-il un point de MMLU ?

**Écrit, commité et TAMPONNÉ le 2026-09-07, AVANT la première mesure.**
Go de l'opérateur du 2026-09-07. Deuxième porte du chantier 14.

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Coût : ~0,45 $** (deux bras MMLU sur `l40sx1`, plafond 25 min soit 0,75 $ au
pire) plus quelques minutes de Mac. Aucun réencodage. Total projet : 119,76 $.

---

## §1 — Ce que M0 a établi, et ce qu'il n'a pas pu établir

M0 mesure `ρ* = 0,960745`, identique à la sixième décimale de 6 à 1 600 blocs
par ligne, et montre qu'une seule multiplication des échelles de ligne rend
**97,6 %** de ce que la règle par bloc du papier achèterait : la rétention passe
de 88,77 à 89,42 % contre 89,44 pour la règle complète
(*mesuré*, `docs/mesures/m0-echelles-2026-09-07.txt`).

**La rétention est un intermédiaire, pas un critère fondamental.** Ce dossier a
la preuve que le lien est cassé : `leech0c13` déplace 21,5 % de perplexité pour
un MMLU indiscernable (*mesuré*, `docs/mesures/leech0c13-2026-09-07.txt`).
+0,65 pp de rétention peut valoir zéro point.

M0 porte deux limites que M1 lève :

1. **Ses blocs sont gaussiens.** Son propre §« limites » le dit : les échelles
   de ligne y sont prises sur les blocs encodés, alors que la production les fixe
   sur la ligne **originale** avant la boucle GPTQ, puis quantifie des résidus
   dont les normes ont dérivé. Ce désaccord d'échelle est **absent** de M0, donc
   son `ρ*` chiffre le biais radial `1/cos θ` seul.
2. **Il ne peut pas voir de structure par ligne**, puisque ses lignes sont
   échangeables par construction.

## §2 — Ce que M1 mesure, et pourquoi il y a deux bras et pas un

`llvq_artifact::decode_matrix` détourne la rotation à la sortie, donc les
tenseurs qu'un artefact rend sont dans la base **naturelle**, celle du
checkpoint. On peut donc calculer, pour chaque ligne `i` de chaque matrice,
l'optimum exact des moindres carrés à codes gelés, **sur les vrais poids** :

```
  ρ_i = ⟨w_i , r_i⟩ / ⟨r_i , r_i⟩
      w_i = la ligne du checkpoint, base naturelle
      r_i = la ligne reconstruite par decode_matrix, base naturelle
  ρ_global = Σ_i ⟨w_i, r_i⟩ / Σ_i ⟨r_i, r_i⟩
```

Deux bras, parce que **le champ `row_scales` existe déjà avec 1 105 920
emplacements** : appliquer un `ρ` par ligne ne coûte **aucun bit**.

```
  A  toutes les échelles de ligne multipliées par ρ_global, un seul nombre
  B  chaque échelle de ligne i multipliée par son ρ_i propre
```

B minimise le même objectif que A avec plus de liberté, donc `‖ΔW‖²(B) ≤ ‖ΔW‖²(A)`
par construction. **Ce que B mesure et que M0 ne pouvait pas voir : si le
désaccord d'échelle de la production crée une structure par ligne réelle.**

## §3 — Le protocole, verbatim

Un outil `llvq-bench` lit `q4b-tetra.llvq` (980 791 242 octets, l'artefact du
Tetra 4B du 2026-09-06), décode chaque matrice, charge la matrice homologue du
checkpoint `Qwen/Qwen3-4B`, calcule les `ρ_i` et écrit deux artefacts neufs où
seul le champ `row_scales` diffère. Puis `seal`, puis un job `l40sx1` :
MMLU avec `LLVQ_MMLU_DUMP`, empreinte `65dcd53655e8bfa5`.

**Aucun code n'est réencodé.** Les indices, la queue, les centroïdes, la graine
de rotation et la carte d'index sont copiés octet pour octet.

L'objectif minimisé est `‖ΔW‖²`, soit le cas `H = I`. La forme pondérée par la
hessienne demanderait une passe de calibration ; elle est hors de ce banc et
c'est écrit ici pour qu'on ne la confonde pas avec ce qui est mesuré.

## §4 — Les contrôles, et si l'un tombe aucun chiffre n'est publié

1. Empreinte de tokens `65dcd53655e8bfa5`, 2 280 questions, sur les deux bras.
2. Les deux artefacts ont **la même taille en octets** que l'original, au bit
   près : seul le contenu de `row_scales` change, pas sa largeur.
3. Un contrôle d'idempotence : appliquer `ρ = 1` partout doit rendre un fichier
   **identique octet pour octet** à l'original. Si ce n'est pas le cas, l'outil
   ne fait pas ce qu'il dit et rien n'est publié.
4. `ρ_global` mesuré sur les vrais poids est **publié à côté** du 0,960745 de M0.
   Un écart de plus de 0,02 dirait que la simulation gaussienne ne décrit pas
   l'artefact, ce qui est en soi une information et va aux écarts.
5. `‖ΔW‖²(B) ≤ ‖ΔW‖²(A) ≤ ‖ΔW‖²(original)` : trois inégalités que
   l'arithmétique impose. Une violation est un défaut d'outil.
6. Le témoin est le dump `t0.csv` du chantier 1, même carte, mêmes questions,
   même fichier aux échelles près. La barre est donc **0,43 pp** et non 2,92.

## §5 — La règle de décision, posée AVANT de voir le chiffre

Soit **ΔM = MMLU(meilleur des deux bras) − 53,49**, apparié contre `t0.csv`.
Barre à fichier constant : **0,43 pp**.

| | conséquence |
|---|---|
| **ΔM ≥ +1,0 pp et IC95 > 0** | **Le correctif radial est réel et il est gratuit.** Il entre dans l'encodeur servi — `nearest_level_index` sur la projection et non sur la norme — et dans tout artefact futur. La ligne 14 de la roadmap garde sa moitié ligne. |
| **+0,43 ≤ ΔM < +1,0** | **Réel mais petit.** On le prend, parce qu'il ne coûte rien, mais il ne change aucune priorité et la moitié ligne du chantier 14 se ferme sur ce gain-là. |
| **ΔM < +0,43** | **Sous la barre. L'axe ligne est clos.** La rétention était un intermédiaire trompeur de plus, le chantier 14 se réduit à sa moitié colonne, et les 1 105 920 échelles de ligne sortent définitivement de la roadmap. |
| **autrement** — B pire que A de plus de 0,43 pp, ou un contrôle du §4 qui tombe | **Rien n'est décidé.** B minimise le même objectif avec plus de liberté ; qu'il perde en MMLU dirait que l'objectif `‖ΔW‖²` est un mauvais proxy du MMLU **par ligne**, ce qui est une découverte à part entière et un préreg à part. |

## §6 — Divulgation datée, et la prédiction signée

Connu : Tetra 53,49 · `ρ*` gaussien 0,960745 · rétention 88,77 → 89,42 % ·
la règle par bloc du papier vaut +0,668 pp de rétention · barre 0,43 pp à
fichier constant · `leech0c13` déplace 21,5 % de ppl pour un MMLU indiscernable
· le volume déplace 10,2 % de ppl pour +2,98 pp de MMLU.

**Prédiction de l'auteur, opposable : ΔM entre 0 et +1,0 pp, centre +0,3, donc
la TROISIÈME ligne du §5. `ρ_global` réel entre 0,955 et 0,968. B meilleur que
A de moins de 0,3 pp.**

Motif : +0,65 pp de rétention est petit, et le taux de change rétention → MMLU
n'existe pas dans ce dossier. Je prends le correctif parce qu'il est gratuit,
pas parce que j'attends qu'il paie.

Seize prédictions signées sur ce dossier, sept fausses.

**Ce qui la rendrait fausse de façon instructive : ΔM ≥ +1,0 pp.** Cela dirait
qu'un biais systématique de 4 % sur la longueur de chaque bloc coûte un point
entier de MMLU — donc que le MMLU est sensible à un mode commun que la
perplexité ne voit pas, et l'inverse exact de ce que `leech0c13` a montré. Il
faudrait comprendre avant d'écrire quoi que ce soit d'autre.

## §7 — Ce que ce banc ne peut pas être

Une mesure de l'axe colonne, qui reste entier. Et il ne dit rien de la forme
pondérée par la hessienne : il minimise `‖ΔW‖²`, pas `Tr(ΔW H ΔWᵀ)`.
