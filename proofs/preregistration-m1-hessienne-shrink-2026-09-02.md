# Pré-enregistrement — M1 : stabilité de l'estimateur de Hessienne (shrinkage hors-diagonale), 0,6B

**Écrit et commité le 2026-09-02. BROUILLON : à tamponner (`ots stamp`) après
relecture de l'opérateur et AVANT le premier run.** Go de principe de
l'opérateur du 2026-09-02 (D0 de `docs/ROADMAP-RECHERCHE.md`, « M1 en
parallèle sur le Mac »).

**Coût : 0 $, ≈ 5 h de Mac** (*estimé* : 12 runs de 22 à 31 min mesurés le
2026-08-25 sur exactement ce protocole, `~/llvq-nuit-b/journal.txt`). Le Mac ne
porte aucun autre job Metal pendant la file.

---

## §1 — La question, et pourquoi c'est la première de l'axe Q

F5 a mesuré **σ = 5,2 % de perplexité** (étendue 10,3 %) entre trois tirages de
calibration du 4B, et 2,92 pp de MMLU. Tant que ce plancher tient, aucune piste
de qualité qui **recalibre** n'est mesurable sans trois graines par bras.

**Hypothèse testée** (audit §4.2, roadmap M1) : ce bruit vient des termes
**hors-diagonaux** de `H = AᵀA/N`, estimés à **13,5 échantillons par dimension**
sur `down_proj` (9 728² pour 131 072 tokens), par lesquels passe toute la
rétroaction d'erreur de GPTQ. La diagonale, elle, se stabilise vite. Si c'est
juste, un shrinkage linéaire vers la diagonale, `H_ρ = ρ·H + (1−ρ)·diag(H)`,
doit **réduire l'étendue inter-graines** avant de dégrader la médiane.

**Ce qui est connu à 0,6B / 28 blocs / `leech1c12`** (`gain-ab-2026-08-25-brut`) :
préfixe contigu **43,4865** ; `LLVQ_CALIB_SEED=1` **38,4507**. ⚠️ Ces deux-là
ne mesurent pas un bruit de graine : `LLVQ_CALIB_SEED` change aussi le **mode**
d'échantillonnage (préfixe → offsets aléatoires). L'étendue entre **graines 1,
2, 3**, toutes en mode offsets, n'a **jamais été mesurée** à cette taille — le
bras ρ = 1 de ce job la donne, et c'est déjà un résultat.

**Ce que ce job ne teste pas** : le volume de calibration (fermé, et F5 l'a
rendu illisible à un bras) ; le damping (mesuré nul à 3 blocs ; c'est un autre
mécanisme : il conditionne la diagonale, le shrinkage retire de l'information
hors-diagonale) ; le shrinkage **en base tournée**, qui après une Hadamard
revient à un gros damping relatif sous un autre nom (`calib::RunConfig::h_shrink`).

---

## §2 — Le job, verbatim

Binaire `target/release/smoke` reconstruit **avec `--features metal,fast-linalg`**
au commit qui porte `LLVQ_H_SHRINK`. File `ops/m1_shrink_queue.sh`, qui
**refuse de démarrer sans `proofs/preregistration-m1-hessienne-shrink-2026-09-02.md.ots`**
(la règle de `rankbench`, portée au shell).

```
for rho in 1 0.9 0.7 0.5; do
  for s in 1 2 3; do
    LLVQ_CALIB_SEED=$s LLVQ_H_SHRINK=$rho \
      target/release/smoke 64 2048 12 2048 metal nogs leech1c12 999 rot
  done
done
```

Tout le reste est le protocole du gate design C tel que le 2026-08-25 l'a
exécuté : Qwen3-0.6B, **28 blocs** (`999` = borne), calibration wikitext-2
train 64 × 2 048 = 131 072 tokens, évaluation wikitext-2 test 12 fenêtres ×
2 048, f32, Metal, rotation on, `nogs`, codebook `leech1c12`. **Une seule
variable par bras** : ρ, la graine étant un facteur de réplication.

Ordre d'exécution : **(ρ = 1, s = 1) en premier** — c'est le contrôle du §3 —
puis ρ-majeur comme écrit. Logs bruts dans `~/llvq-nuit-b/m1-r<ρ>-s<s>.log`,
recopiés dans `docs/mesures/m1-hessienne-shrink-2026-09-0X-brut/` à la fin.

---

## §3 — Contrôles, et si l'un tombe aucun chiffre n'est publié

1. **(ρ = 1, s = 1) rejoue 38,4507** au dix-millième — le binaire porte le
   bouton, et à ρ = 1 l'appel est **sauté**, pas multiplié par un : c'est le
   chemin publié, et Metal l'a déjà reproduit d'un processus à l'autre
   (`tests/resume.rs`). Un écart ici veut dire que le bouton a bougé le chemin
   publié, ou que la machine n'est pas déterministe ; dans les deux cas on
   s'arrête avant de lire quoi que ce soit.
2. **Chaque log imprime `hessian shrink ρ = <ρ>`** avec le ρ du bras, et
   **`effective rate = 2.1656 bits/weight`** sur les douze : le shrinkage
   touche H, pas le code — un débit qui bouge est un autre objet.
3. **`baseline (f32) ppl = 19.5038`** sur les douze runs : mêmes fenêtres
   d'évaluation partout.
4. **Douze logs bruts conservés**, jamais résumés avant d'être commités.

---

## §4 — Ce qui se publie, et ce qui NE se compare PAS

Par ρ : les trois ppl, la **médiane**, l'**étendue** (max − min), le rapport
**étendue(ρ) / étendue(1)** et le décalage **médiane(ρ) − médiane(1)**. Douze
nombres, quatre lignes, aucun n'est une barre d'erreur de précision (n = 3).

Ne se compare pas : au σ du 4B (autre taille, autre corpus de calibration, autre
profondeur) ; au 43,4865 du préfixe (autre mode d'échantillonnage) ; à
n'importe quel chiffre de qualité publié — ce job mesure une **dispersion**,
pas un niveau.

---

## §5 — La règle de décision, posée AVANT de voir le chiffre

Soit **E(ρ)** l'étendue inter-graines et **M(ρ)** la médiane, à 28 blocs.

| | verdict |
|---|---|
| il existe ρ < 1 avec **E(ρ) ≤ E(1) / 2** *et* **M(ρ) ≤ M(1) + E(1)** | **ρ\* = le plus grand de ces ρ** ; M1 vert, **Q1 s'ouvre** (rejouer `leech1c12` 0,6B à ρ\*, trois graines, puis seulement le 4B) |
| aucun ρ ne satisfait les deux | **kill : ρ\* = 1**. L'hypothèse « instabilité hors-diagonale » n'est pas soutenue à cette échelle ; Q1 ne s'ouvre pas ; les pistes Q qui recalibrent gardent le protocole à trois graines et son coût |
| **E(1) < 2 % de M(1)** | design **sous-puissant** : le 0,6B à 28 blocs ne porte pas assez de bruit de graine pour tester quoi que ce soit ; déclaré tel quel, **rien n'est adopté**, et l'échelle où refaire le test (4B, 3 × 7,11 $ par ρ) est un devis pour l'opérateur, pas une suite |

⚠️ La condition sur la médiane est **délibérément large** (une étendue
entière) : un shrinkage qui divise le bruit par deux au prix d'une médiane
déplacée de moins que ce bruit reste un meilleur instrument. Elle n'est pas un
critère de qualité — ce serait Q1, à trois graines, après.

---

## §6 — Divulgation datée, à la signature

- Connu : F5 (σ 5,2 % au 4B, 3 runs complets, 21,45 $) ; le bruit MMLU 2,92 pp ;
  au 0,6B/28 blocs, 43,4865 (préfixe) et 38,4507 (graine 1) pour `leech1c12`,
  et un écart de 6 à 14 % entre modes sur les quatre codebooks de l'A/B des
  bits de gain. L'étendue entre graines 1-2-3 à cette taille est **inconnue**.
- **Prédiction de l'auteur, écrite pour être opposable : kill, ρ\* = 1.**
  Motif : le bruit de graine **croît avec la profondeur** (0,7 % à 3 blocs,
  ~10 % à 28-36 blocs), ce qui est la signature d'une **composition
  séquentielle** — chaque bloc est calibré sur les activations des blocs
  amont déjà quantifiés, et une petite différence de H se propage — plutôt
  que celle d'un estimateur bruité par matrice, dont l'effet se moyennerait sur
  36 blocs. Le shrinkage réduira l'étendue seulement à ρ ≤ 0,7, en déplaçant la
  médiane de plus qu'une étendue. Si cette prédiction est fausse, c'est
  l'instrument de tout l'axe Q qui change, et ce sera le résultat le plus utile
  de l'automne.
- ⚠️ Même réserve que pour M2 : les prédictions signées de ce dépôt ont un
  historique récent médiocre. Elle engage, elle ne prouve rien.
