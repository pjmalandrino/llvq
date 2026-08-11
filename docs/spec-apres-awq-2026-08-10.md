# Spec — ce que la mesure AWQ force à reprendre

**Version 0.1, 2026-08-10.** Document de travail interne, écrit pour être repris
**dans une branche neuve** sans relire la session qui l'a produit.

> **En une phrase** : le premier bras concurrent a montré qu'un noyau 4 bits
> déployé domine `Planes14` **sur les deux axes du banc**, ce qui déplace
> l'argument du projet de la vitesse vers la mémoire — et sur la mémoire, le
> seul layout qui ait encore quelque chose à dire est celui qu'on a écarté.

---

## 1. Le fait qui force tout le reste

Job `6a7a11d32ed17c71070fda7f`, L40S, un processus, sept rounds, six bras
entrelacés. Journal : [`mesures/six-arm-awq-2026-08-10.txt`](mesures/six-arm-awq-2026-08-10.txt).

| format | b/poids noyau | méd ms | Go/s | vs FP16 | % de sa borne d'octets |
|---|---|---|---|---|---|
| FP16 (témoin) | 16,000 | 11,016 | 660 | 1,00× | — |
| Slot32 | 5,510 | 5,829 | 430 | 1,89× | 65 % |
| **Planes14** (production) | 4,804 | 5,111 | 428 | 2,16× | 65 % |
| Planes12x | 4,342 | 5,510 | 358 | 2,00× | 54 % |
| Golay70 (« écarté ») | 3,589 | 8,223 | 198 | 1,34× | 30 % |
| **AWQ w4g128** | **4,179** | **3,256** | **584** | **3,38×** | **88 %** |

**`Planes14` est dominé par AWQ sur les deux axes** : plus lent (2,16× contre
3,38×) *et* plus gros en payload (4,804 contre 4,179 b/poids). Ce n'est pas une
question d'interprétation, c'est la même carte, les mêmes rounds, le même
chronomètre.

⚠️ **La réserve qui aurait pu sauver la lecture est fermée, et contre nous.**
La colonne « Go lus » ne facture que les poids ; on pouvait espérer que le
trafic d'activation, non facturé, rétablisse l'équilibre. Calculé : **1,82 Go
chez nous contre 7,27 chez AWQ** — leur noyau relit l'activation **4× plus**,
une par canal de sortie contre une par bloc de huit lignes chez nous. Et une
activation entière pèse ≤ 39 Ko contre 100,7 Mo de L2 : les relectures sortent
du L2, pas de la DRAM. L'écart 428/584 est réel.

---

## 2. Le tableau qui n'avait jamais été fait, et qui rouvre le dossier

En **b/param modèle entier, embedding q8 compris** — la seule comptabilité dans
laquelle une comparaison mémoire a un sens (errata du lot A) :

| layout | b/poids noyau | **b/param modèle** | Go sur carte (4B) | vs AWQ déployé (5,30) |
|---|---|---|---|---|
| Slot32 | 5,510 | 5,751 | 2,89 | +8,5 % |
| **Planes14** | 4,804 | 5,113 | 2,57 | **−3,5 %** |
| Planes12x | 4,342 | 4,696 | 2,36 | −11,4 % |
| **Golay70** | **3,589** | **4,016** | **2,02** | **−24,2 %** |

*(Calculé : `(bpw·3 633 300 000 + 8·389 070 848) / 4 022 370 848`. Recoupe le
5,15 publié par `rtbits` à 0,04 près — l'écart est la queue et les échelles de
ligne, que `rtbits` compte et que cette formule approche.)*

**Golay70 est le seul bras qui batte franchement l'AWQ déployé sur la
mémoire**, et il reste **1,34× plus rapide que le FP16**. Il n'est pas lent
dans l'absolu — il est lent *par rapport à nos autres layouts*.

Pendant ce temps `Planes14` ne gagne que **3,5 %** de mémoire sur l'AWQ, une
marge que la §Limitations du papier qualifie déjà elle-même de trop faible pour
justifier 14,6 points de MMLU.

---

## 3. 🚨 Le point de méthode : comment rouvrir E2 sans tricher

`Golay70` a été écarté par un critère de **1,6× contre FP16, posé d'avance**
(`mesures/e2-golay70-bench-2026-08-07.txt`, et `layouts.tex:88-97` en fait un
argument de rigueur). Il a rendu 1,31× à l'époque, 1,34× aujourd'hui. **Le rejet
était défendable et il faut le dire ainsi.**

Le rouvrir parce qu'un autre chiffre est joli serait exactement le déplacement
de poteaux que ce dossier documente ailleurs. **Ce n'est pas ce qu'on fait.**

Ce qu'on fait, et qui doit être écrit noir sur blanc avant la première mesure :

> Le critère de 1,6× était un **seuil de vitesse appliqué à un format de
> mémoire**. C'était la bonne question tant que la thèse du projet était
> « notre noyau est rapide ». La mesure du 2026-08-10 rend cette thèse
> indéfendable : un 4 bits déployé est plus rapide *et* plus petit que notre
> layout de production. L'argument résiduel du projet est donc la mémoire, et
> un critère de vitesse ne peut pas trancher une question de mémoire.
>
> Le critère n'est pas **effacé**, il est **périmé par une évidence neuve**, et
> le nouveau se pose **avant** de remesurer.

### Le critère neuf, à commiter dans `proofs/` avant tout job

| condition | verdict |
|---|---|
| `Golay70` optimisé atteint **≥ 2,0× FP16** *et* **≤ 4,1 b/param modèle** | **adopté** pour le chemin servi : la marge mémoire contre l'AWQ dépasse 20 %, assez pour survivre à l'objection MMLU que les 3,5 % de `Planes14` ne survivaient pas |
| entre **1,6× et 2,0×** | **non adopté**, publié comme point de la courbe débit↔taux — ce qu'il est déjà |
| **< 1,6×** | le décodage à double coset est irréductible ; E2 se referme, cette fois définitivement, et le papier garde son résultat négatif |

**Pourquoi 2,0×** : c'est la vitesse de `Planes12x` aujourd'hui, donc un seuil
déjà atteint par un layout servi — pas un chiffre inventé. **Pourquoi 4,1** :
c'est ce qui met la marge mémoire au-delà de 20 %.

---

## 4. Le fait technique qui rend la réouverture plausible

`Golay70` est à **198 Go/s** quand `Planes14` est à 428 et AWQ à 584. Il n'est
**pas memory-bound** : son goulot est son propre décodage à double coset. C'est
une propriété **du noyau**, donc potentiellement réparable — pas un plancher du
format.

S'il devenait memory-bound au rythme de `Planes14` (428 Go/s), ses 1,63 Go se
liraient en **~3,8 ms**, soit **2,9× le FP16** — devant AWQ. Même à 358 Go/s
(le rythme de `Planes12x`) il rendrait **4,55 ms**, soit **2,4×**.

**Deux pistes jamais tentées**, notées lors de la reconnaissance et non
poursuivies faute de raison de le faire :

1. **Spécialiser les warps par coset.** Le décodage sépare pair et impair ; une
   grille qui n'envoie qu'un coset par warp supprime la divergence résiduelle.
2. **Ne payer le XOR que du côté pair.** Le rang de codeword Golay n'est requis
   que sur le coset pair ; le côté impair le calcule pour rien.

⚠️ Aucune des deux n'est chiffrée. Elles peuvent ne rien rendre. C'est
exactement pourquoi le critère est posé d'avance.

> 🚨 **Ce paragraphe est périmé depuis le 2026-08-11 — les deux pistes
> ci-dessus ne sont PAS celle qui a été retenue, et l'une est fausse.**
> L'analyse ([`projections-golay70-2026-08-11.md`](projections-golay70-2026-08-11.md)
> §3) a montré : (a) le goulot n'est pas la divergence — tout est déjà
> prédiqué — mais le **compte d'ops entières par slot** (~14 contre ~9 pour
> `Planes14`, cohérent avec le ×1,61 mesuré) ; (b) la piste 2 est **caduque**
> — le noyau livré ne ré-encode aucun XOR, il lit `cwtab` (16 Kio, L1),
> requis sur les deux cosets ; (c) une piste **neuve** les subsume : hisser
> la logique de coset au niveau bloc en mots de 24 bits, à format inchangé.
> **Elle est implémentée** (v2 de `llvq_golay.cuh`, branche
> `claude/golay70-memory-performance-ksbbzs`), identité prouvée sur le
> harnais hôte, 3 mutants tués. Ce qui manque : la vitesse sur carte —
> c'est le lot C, jugé par le critère du §3 ci-dessus, qui reste inchangé.
> Reprise : [`passation-golay70-2026-08-11.md`](passation-golay70-2026-08-11.md).

---

## 5. Ce qui est déjà fait, à ne pas refaire

| | où |
|---|---|
| ligne de base 5 bras, `main` corrigé | [`mesures/baseline-head-2026-08-10.txt`](mesures/baseline-head-2026-08-10.txt) |
| banc 6 bras avec AWQ | [`mesures/six-arm-awq-2026-08-10.txt`](mesures/six-arm-awq-2026-08-10.txt) |
| noyau AWQ porté (MIT, sha256 amont) | `llvq-cuda/kernels/awq_gemv.cu` |
| hôte AWQ : quantifieur w4g128, empaqueteur, lanceur | `llvq-cuda/src/bin/planesbench.rs` |
| distribution des blocs par coquille | [`mesures/shell-distribution-4b-2026-08-10.txt`](mesures/shell-distribution-4b-2026-08-10.txt) |
| reconnaissance des trois concurrents | [`kernel-comparison-recon.md`](kernel-comparison-recon.md) |
| pré-enregistrement + journal d'écarts | [`../proofs/preregistration-2026-08-10.md`](../proofs/preregistration-2026-08-10.md) |
| `f2h` hôte corrigé (rendait 0) | `llvq-cuda/kernels/matvec.cu` |

**Et deux dettes ouvertes qui bloquent la propreté de la suite :**

- **Le sélecteur de bras n'existe pas.** Le job à six bras a donc enfreint la
  règle §4 du pré-enregistrement — pas de contrôle à cinq bras dans le même
  processus. L'écart est consigné (§7bis É1). **Tant que le sélecteur n'existe
  pas, chaque nouveau bras rejouera l'entorse**, et une règle enfreinte deux
  fois cesse d'être une règle. C'est la clause que le lot 1 doit gagner : *un
  run de contrôle doit être à une variable d'environnement près*.
- **`ops/manifest.jsonl` n'existe pas.** L'outil de provenance (`ops/manifest.py`,
  `record`/`verify`/`report`) est écrit et n'a **jamais servi** : zéro entrée
  pour une centaine de nombres publiés.

---

## 6. Les lots, dans l'ordre

### Lot A — le pré-enregistrement du critère neuf · 0 $ · ½ jour
Écrire dans `proofs/` le §3 de ce document : pourquoi le critère de 1,6× est
périmé, et le critère de remplacement. **Avant** toute mesure. Signer et
horodater comme le précédent.

### Lot B — le sélecteur de bras · 0 $ · 1-2 jours
Un bras écarté ne doit **pas construire ses tampons**, sinon le contrôle ne
restitue pas la résidence VRAM qu'il mesure. C'est le trait `KernelArm` du lot 1
de la campagne kernel, ou sa moitié minimale. Solde l'entorse É1 et débloque
tous les lots suivants.

### Lot C — les deux optimisations de Golay70 · ~1 $ · 3-5 jours
Warps spécialisés par coset, XOR côté pair seulement. Chaque piste vérifiée
localement contre le décodeur Rust (`tests/golay70_decoder_matches_rust.rs`
existe et tourne sur macOS via clang++), puis un job à sept bras **avec son
contrôle**. Verdict par le critère du lot A.

> ✅ **La moitié locale de ce lot est faite (2026-08-11), par une autre piste
> que les deux nommées** — voir le 🚨 du §4. Le décodeur v2 est écrit,
> l'identité est prouvée, la vérification locale est passée. Ce qui reste du
> lot C : le sweep de l'artefact scellé sur le Mac de dev, puis **le job à
> sept bras avec son contrôle** (bloqué par le lot B, le sélecteur de bras).
> Détail et commandes : [`passation-golay70-2026-08-11.md`](passation-golay70-2026-08-11.md).

### Lot D — la conséquence papier, quel que soit le verdict · 0 $ · 2 jours
Elle est due **même si Golay70 échoue**, parce qu'elle vient de la mesure AWQ :

- l'abstract ne peut plus dire « notre noyau est rapide » sans dire à côté
  qu'un 4 bits déployé l'est davantage sur la même carte ;
- `layouts.tex` gagne le bras AWQ et la lecture en **fraction de borne
  d'octets** (65 % contre 88 %), qui est le vrai résultat ;
- la Figure 1 gagne son premier point concurrent ;
- `limitations.tex:63-65` (« AWQ speed is not comparable ») est **caduc** : il
  l'est maintenant, dans notre harnais, en stratégie A ;
- le total GPU bouge (quatre sites, cf. la tâche différée du 14B).

### Lot E — les autres concurrents · selon budget
`Shell12` (chiffre le prix du codebook, ~1 jour), `MonoShell3`, Marlin dans un
balayage de batch, QTIP. Tous décrits dans
[`kernel-comparison-recon.md`](kernel-comparison-recon.md).

---

## 7. L'arbitrage à ne pas cacher

Si `Golay70` est adopté, le 4B servi passerait de ~88 à ~55 tok/s pour 2,02 Go
au lieu de 2,57. **24 % de mémoire contre 37 % de débit.** Ce n'est pas un repas
gratuit, et le sens de l'arbitrage dépend de ce qui manque — la carte ou la
latence.

Ce que ça vaut vraiment se voit à l'échelle où la mémoire mord : à 32B, 24 % de
b/param est la différence entre tenir et ne pas tenir sur une carte donnée. À
4B, c'est un confort. **Le 4B est le véhicule de mesure ; il n'est pas
l'argument.**

---

## 8. Budgets

| campagne | dépensé | plafond |
|---|---|---|
| papier (4B + 8B) | 19,82 $ | — |
| 14B | 31,46 $ | 35 $ |
| **kernel** | **1,56 $** | **15 $** |

Les lots A à D tiennent dans ~2 $. Le plafond kernel n'est pas la contrainte —
**les jours d'ingénierie le sont.**
