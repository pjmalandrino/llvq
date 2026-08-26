# Plan de travaux — la variance de calibration

> Le **quoi faire, dans quel ordre, et avec quoi ça se termine**. Le protocole
> expérimental lui-même — facteurs, seuils, contrôles, prédictions signées — vit
> dans
> [`proofs/preregistration-variance-calibration-2026-08-26.md`](../proofs/preregistration-variance-calibration-2026-08-26.md),
> et ce document ne l'amende pas : il l'exécute.
>
> **Contexte, en une ligne.** Le pari produit LLVQ est clos (BACKLOG §0 : le mur
> d'octets est arithmétique, aucun travail de noyau ne le franchit). Ce qui
> survit est l'**instrument** — et il permet de poser une question qui ne
> dépend pas du tout de Λ₂₄.

---

## 0. Ce que ce chantier produit, et ce qu'il ne produit pas

**Produit** : une courbe σ(volume, bits, famille de quantifieur) au 0.6B, un
taux d'inversion de rang, et — si Gate B passe — un point de transfert au 4B avec
la métrique de capacités. Matière d'un papier de méthode.

**Ne produit pas** : aucune amélioration de LLVQ, aucun gain de débit, aucun
octet économisé. **Rien de ce chantier ne rouvre le verdict produit**, et il ne
faut pas le vendre comme tel.

**Budget total** : ~4 $ jusqu'à Gate B, ~70 $ au-delà sur go séparé. Le reste est
du temps de Mac, gratuit.

---

## Lot 0 — les prérequis code (0 $, ~1 journée)

Rien ne part avant que les quatre soient verts. Trois touchent du code, le
quatrième est une vérification.

### 0.1 — Lever le plafond de 8 M caractères, et le faire échouer au lieu de clamper

**Le défaut.** `llvq-llm/src/bin/smoke.rs:876` demande
`c4_calibration(8_000_000)`, puis `:897` fait
`let n_calib = n_calib.min(train_ids.len() / calib_len);` — **un clamp
silencieux**. Le lot B a mesuré ce que 8 M caractères rendent : **847 fenêtres**
(*mesuré*, `verdicts-lot-b-2026-08-06.md:33`), soit 4,61 caractères par token
(*calculé*). Les barreaux ×16 et ×32 de la grille en demandent 9,67 M et 19,3 M :
**ils retomberaient tous deux sur ×13,2 sans un mot dans le log.**

**Le correctif**, deux moitiés :

1. Dériver le budget du besoin : `n_calib × calib_len × 6` caractères (marge sur
   les 4,61 mesurés), au lieu du littéral.
2. Remplacer le clamp par un `anyhow::ensure!` qui **nomme le manque**. C'est la
   règle du §7 de `CLAUDE.md` appliquée à un run plutôt qu'à un test : *un
   dispositif qui saute quand sa ressource manque doit échouer, pas passer.*

⚠️ **Effet de bord assumé** : tout script existant qui demandait plus de fenêtres
que le corpus n'en porte échouera désormais au lieu d'en produire moins en
silence. C'est l'objet du correctif, pas un dommage.

**Létalité** : un test qui demande un volume impossible et exige l'échec nominatif.

### 0.2 — `ScalarGroupwise`, le bras qui décide de la portée du papier

**Pourquoi.** Si la variance n'est montrée que sur Λ₂₄, c'est une note de bas de
page sur une méthode morte. Sur un quantifieur scalaire groupwise — le défaut
d'AutoGPTQ, `group_size=128`, `static_groups=False` — c'est une propriété de
**GPTQ**.

**Ce qui existe déjà et sert de patron** : `llvq-quant/src/quantizer.rs:224-240`
porte `ScalarGrid { block, step }`, dont le commentaire dit exactement
*« the baseline GPTQ has to beat »* — mais son `step` est **global**, là où
l'INT-k réel dérive une échelle et un zéro **par groupe**.

**À écrire** (~50 lignes de quantifieur, plus le câblage) :

- ✅ **Fait le 2026-08-26** — `ScalarGroupwise { block, bits }`, transcrit de
  `Quantizer.find_params`/`quantize` d'AutoGPTQ (sha256 `2e0b4588…`) plutôt que
  re-dérivé, avec la grammaire `int<bits>g<groupe>` et la comptabilité
  `bits + (16 + bits)/groupe`.

🕳️ **Et le recoupement du §0.3 a fait son travail avant même d'être écrit
comme tel.** Une première implémentation, écrite de mémoire, a passé neuf tests
de propriété en étant fausse sur **trois** points, chacun déplaçant la grille :
l'étendue n'était pas étendue à zéro ; le zéro point était un flottant là où
un fichier déployé le stocke en entier packé à `bits` ; le groupe dégénéré était
traité autrement. Elle surchargeait en prime le débit de `(16−bits)/groupe`,
soit 0,54 b/poids à `int3g24`. **Aucun test de propriété écrit sans la source
sous les yeux ne pouvait les trouver** — chacune était cohérente avec
elle-même. C'est l'argument entier pour ce lot : des propriétés épinglent *un*
quantifieur, seule une transcription épingle *lequel*.

🚨 **Conséquence à connaître avant d'écrire : le bras scalaire ne produira pas
d'artefact.** `quantize_model_capturing` refuse la capture pour tout codebook
dont les codes ne décrivent pas la reconstruction, et `ScalarGroupwise` n'a pas
de `BlockCode`. Ce n'est pas un problème — la grille se lit sur la perplexité
interne de `smoke`, pas sur un fichier scellé — mais **ça impose que le bras
Leech se lise par le même chemin** (cf. §protocole, contrôle C8).

**Létalité** : test de propriété — toute sortie est sur la grille
`zero + k·scale`, l'erreur maximale est ≤ `scale/2`, et `bits_per_weight` rend la
valeur calculée d'avance. Plus le contrôle de fidélité ci-dessous.

### 0.3 — Le contrôle de fidélité, sans lequel « GPTQ standard » est une affirmation

Un relecteur dira que notre scalaire n'est pas celui du domaine. Deux niveaux,
et le premier est obligatoire :

- **Obligatoire** : une référence indépendante écrite **dans le fichier de test**
  — le patron de `correction_is_the_analytic_minimizer`, qui reconstruit son
  attendu par un chemin qui ne partage pas une ligne avec le code testé.
- **Recommandé** : un recoupement contre AutoGPTQ sur **une** matrice, même
  calibration, même graine. ~une demi-journée, Python hors workspace. C'est ce
  qui transforme « notre scalaire » en « le scalaire du domaine ».

### 0.4 — L'évaluation à 73 fenêtres

Rien à écrire : c'est un argument positionnel de `smoke`. À **vérifier** :
que WikiText-2 test porte bien 73 fenêtres de 2048 (149 504 tokens) et que la
baseline sort stable. ⚠️ **La baseline à 73 fenêtres n'est pas 19,5038** — ce
chiffre est celui de 12 fenêtres à ctx 2048. La valeur de référence du contrôle
C4 s'établit au premier run et se fige ensuite.

---

## Lot 1 — Étage 0, le gate à 0 $ (~1 h)

Trois runs, et **la grille ne part pas avant qu'ils soient lus**.

| run | ce qu'il pin |
|---|---|
| `leech1c12` ×1 graine 1, **deux fois** | le **déterminisme** — Gate A |
| `scalar-g128-b3` ×1 graine 1 | le terme fixe du bras scalaire (~160 s *estimé*) |

**Gate A** : deux runs à graine identique doivent rendre la même ppl à 1e-9
relatif. **Si ça échoue, la grille ne part pas** — σ mélangerait la calibration
et le non-déterminisme du backend, et aucune cellule ne serait interprétable. Le
chantier deviendrait alors une enquête sur le déterminisme, ce qui est un
résultat aussi, et moins cher.

---

## Lot 2 — La grille 0.6B (~28 h de Mac + ~4 h d'évaluation, 0 $)

**64 runs**, dans cet ordre pour que le premier résultat lisible arrive tôt :

1. `leech1c12`, ×1 et ×8, les 4 graines — **8 runs, ~4,3 h**. Donne σ_diff et un
   premier point de pente.
2. `scalar-g128-b3`, ×1 et ×8, les 4 graines — **8 runs, ~1,4 h**. Donne le
   contraste de famille, qui est la prédiction P4.
3. Les ancres bits : `b2` et `b4` à ×1 et ×8 — **16 runs, ~2,7 h**.
4. Le remplissage de la courbe : les deux bras porteurs aux volumes ×2, ×4, ×16,
   ×32 — **32 runs, ~20 h**.

⚠️ **L'étape 4 est la seule longue, et c'est la dernière exprès** : si les
étapes 1-2 montrent que σ_diff est dégénéré, elle ne se lance jamais.

**Livrable** : `docs/mesures/variance-grille-0.6b-<date>.txt` plus les logs bruts
dans un sous-répertoire — le format du lot du 2026-08-25, qui a permis de
retrouver le profil par phase trois jours après.

---

## Lot 3 — Le contrôle d'invariance de backend (~4 $)

Deux cellules répliquées sur `l40sx1` : `leech1c12` ×1 et `scalar-g128-b3` ×1,
k = 3 graines, **6 runs**.

⚠️ **Ce n'est pas pour aller plus vite.** 84 % du run est l'encodeur, qui est
**CPU** : louer une carte n'accélère pas le terme dominant. Porter la grille
entière sur CUDA coûterait ~48 $ (*calculé* : 28 h × 1,77 $/h *mesuré*) pour un
résultat que Metal rend gratuitement. Le contrôle sert uniquement à pouvoir
écrire que σ n'est pas une propriété de Metal.

**Prérequis d'exécution** : `oracle` d'abord, comme toujours — 42 s et ~1 centime
pour savoir si les hessiennes de ce backend valent quelque chose.

---

## Gate B — avant le premier dollar au 4B

Les quatre conditions sont au §5 du pré-enregistrement. En résumé : Gate A vert,
σ_diff non dégénéré, une pente β dont l'IC95 exclut 0, et aucun contrôle tombé
sur plus d'une cellule.

**C'est la demande explicite de l'opérateur** — vérifier qu'on tient quelque
chose de propre avant d'engager les 70 $.

---

## Lot 4 — Le transfert au 4B (~70 $, go séparé)

**9 runs** (le `leech1c12` ×1 est déjà payé : les trois graines F5, artefacts au
bucket `f5-graines-2026-08-19/seed{1,2,3}/`). C'est la seule taille où MMLU sort
du hasard, donc la seule où l'axe **capacités** existe.

**Prérequis** : `hf buckets ls` d'abord (§2.8 du backlog, toujours non fait) — la
règle des canaux de rétention a déjà payé quatre fois, dont un 0,5 $ au lieu de
21 $ le 2026-08-25.

---

## Lot 5 — Ce qu'on écrit avec

Pas avant que Lot 2 soit lu. Trois pièces, par ordre d'écriture :

1. **Le journal de mesure** — format maison, avec la lecture, les contrôles et
   les prédictions confrontées. C'est lui qui fait foi.
2. **La mise à jour de `CLAUDE.md` et du backlog** — le §3 (qualité) change de
   sens : ce n'est plus un axe qui pourrait sauver le produit, c'est le sujet.
3. **Le papier**, si la grille tient. Titre de travail : *« Combien de tirages de
   calibration faut-il pour qu'une affirmation de quantification tienne ? »* —
   l'estimand est la **résolubilité**, pas la sensibilité, et l'inversion de rang
   est la figure 1.

---

## Ce qui est délibérément hors périmètre

| | pourquoi |
|---|---|
| **Le 8B** | ~140 $ pour une pente à deux volumes (*calculé*, 11,48 $/run *mesuré* au 2026-08-02). À rouvrir seulement si la pente 0.6B → 4B est ambiguë. |
| **Une seconde architecture** | `model.rs` est une passe avant écrite à la main pour Qwen3 ; Llama-3.2-1B demande ~200-300 lignes plus un `oracle`. **C'est la limite de crédibilité n°1**, déclarée au §8 du pré-enregistrement plutôt que découverte en revue. |
| **Un quantifieur treillis** | QTIP ne peut pas entrer (on a son noyau, pas son quantifieur, et son bras de banc tourne sur payload pseudo-aléatoire). Un treillis **à nous** — Viterbi sur un état 16 bits, ~200-300 lignes — serait le substitut propre. Après la grille, pas avant. |
| **Tout item d'optimisation** | Débit, mémoire, noyau : le verdict produit est clos, et l'audit du 2026-08-26 les avait déjà classés P2 ou abandon sur la seule arithmétique du plafond 3,33×. |

---

## Ordre d'exécution, en une ligne

**Lot 0** (une journée de code) → **Lot 1** (une heure, Gate A) → **Lot 2**
étapes 1-2 (~6 h, premier résultat lisible) → **Lot 2** étapes 3-4 (~23 h) →
**Lot 3** (~4 $) → **Gate B** → **Lot 4** sur go budget.
