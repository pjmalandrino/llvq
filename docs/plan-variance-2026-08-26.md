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
page sur une méthode morte. Sur le quantifieur affine du domaine — celui
d'AutoGPTQ, `sym=False` — c'est une propriété de **GPTQ**.

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

**Létalité** : ✅ 12 tests de propriété, **11 mutants sur 11 tués** (en deux
tours : le premier laissait passer `round_ties_even` → `round`, que seul un
groupe construit sur un tie *pair* exerce).

### 0.3 — Le contrôle de fidélité, sans lequel « GPTQ standard » est une affirmation

Un relecteur dira que notre scalaire n'est pas celui du domaine. Deux niveaux,
et le premier est obligatoire :

✅ **Fait le 2026-08-26, et il était obligatoire, pas recommandé.** La source
amont a été récupérée (`AutoGPTQ@main:auto_gptq/quantization/quantizer.py`,
sha256 `2e0b4588…`) et transcrite ; `the_transcription_matches_upstream` compare
au bit près sur 8 largeurs × 4 étendues × 5 décalages, y compris le mode
d'arrondi (`torch.round` casse les ties **au pair**, pas `f64::round`).

🕳️ **Et une affirmation fausse a été évitée de justesse au passage.** J'avais
écrit dans le source que le clamp amont « se déclenche sur données réelles » ;
38,4 M de poids sur huit largeurs disent zéro déclenchement. La raison est
arithmétique — `round(maxq − t) + round(t) = maxq` pour tout `t` non
demi-entier — et elle est maintenant écrite, avec le cas de tie qui rend le
clamp vivant par test plutôt que par assertion.

### 0.4 — L'évaluation à 73 fenêtres

Rien à écrire : c'est un argument positionnel de `smoke`. À **vérifier** :
que WikiText-2 test porte bien 73 fenêtres de 2048 (149 504 tokens) et que la
baseline sort stable. ⚠️ **La baseline à 73 fenêtres n'est pas 19,5038** — ce
chiffre est celui de 12 fenêtres à ctx 2048. La valeur de référence du contrôle
C4 s'établit au premier run et se fige ensuite.

---

## Lot 1 — Étage 0 : le gate à 0 $ (~1 h de Mac)

**Rien de la grille ne part avant que ces trois runs soient lus.** Ils tiennent
en une heure et ils décident d'un chantier de 28 h.

### Ce qu'il faut avant de lancer

| | |
|---|---|
| machine | **le Mac** — cette boîte de session n'a ni Metal ni CUDA, un run y tomberait sur CPU et ne serait pas le protocole |
| build | `cargo build --release -p llvq-llm --features metal --bin smoke` |
| modèle | `Qwen/Qwen3-0.6B` en cache HF (sinon premier run = téléchargement) |
| ✅ précision | `smoke` imprime depuis le 2026-08-26 une ligne `exact-ppl … {:.12e}` — **sans elle Gate A est inobservable**, les lignes d'affichage étant à `{:.4}`, soit 5·10⁻⁶ de résolution relative à ppl ≈ 20 |
| 🚨 tampon | **`ots stamp proofs/preregistration-variance-calibration-2026-08-26.md` AVANT le premier run** — le §3 du pré-enregistrement l'exige avant la première milliseconde mesurée, et il n'est pas encore posé. **Impossible depuis la boîte de session** : les quatre calendriers y sont injoignables (403 du proxy). C'est donc une action du Mac, dans la même session que les trois runs, et **avant** eux. Commiter le `.ots` produit. |

> 🔎 **Et pendant qu'on y est, un `ots upgrade proofs/*.ots`** : quatre tampons
> du 2026-08-25 sont encore en pure attente, les seize autres portent déjà
> leurs ancres Bitcoin (mesuré le 2026-08-26,
> [`docs/mesures/ots-etat-2026-08-26.txt`](mesures/ots-etat-2026-08-26.txt)).
> Zéro coût, et ça règle la dette de vérifiabilité pour de bon.

### Les trois runs, verbatim

```bash
D=docs/mesures/variance-etage0-$(date +%Y-%m-%d) && mkdir -p $D
export LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_CALIB=c4 LLVQ_THREADS=12 LLVQ_CALIB_SEED=1

# A1 et A2 — le MÊME run, deux fois. Gate A.
target/release/smoke 64 2048 73 2048 metal nogs leech1c12 999 rot 2>&1 | tee $D/a1-leech-s1.txt
target/release/smoke 64 2048 73 2048 metal nogs leech1c12 999 rot 2>&1 | tee $D/a2-leech-s1.txt

# A3 — le terme fixe du bras scalaire, seul nombre encore estimé du modèle de coût.
target/release/smoke 64 2048 73 2048 metal nogs int3g24  999 rot 2>&1 | tee $D/a3-int3g24-s1.txt
```

Ordre des positionnels : `n_calib calib_len n_eval eval_ctx device mode codebook
limit rot`. `999` est la sentinelle « tous les blocs » (28 au 0.6B).

### Ce qu'on lit, dans l'ordre

**1. Gate A — le déterminisme.** Comparer la ligne `exact-ppl` de `a1` et `a2` :

```bash
grep '^exact-ppl' $D/a1-leech-s1.txt $D/a2-leech-s1.txt
```

- ✅ **Chiffres identiques** → le pipeline est déterministe, toute la variance
  observée dans la grille est de la calibration. La grille part.
- ❌ **Chiffres différents** → σ mélangerait deux sources et **aucune cellule
  de la grille ne serait interprétable**. La grille ne part pas ; le chantier
  devient une enquête sur le déterminisme — ce qui est un résultat aussi, et
  bien moins cher que 28 h de runs illisibles.

⚠️ **C'est la prédiction P1 du pré-enregistrement, et celle que je tiens pour la
plus sûre — donc la plus coûteuse à rater.** Le seul doute réel est le backend :
la boucle GPTQ est prouvée déterministe (`parallel_matches_serial_exactly` exige
le découpage parallèle bit-identique au sériel), les graines de rotation
dérivent d'une fonction pure, mais l'ordre de réduction d'un matmul Metal n'a
jamais été vérifié ici.

**2. Le terme fixe du bras scalaire.** Dans `a3`, le bloc `phases:` :

```bash
sed -n '/^phases/,/^$/p' $D/a3-int3g24-s1.txt
```

Le modèle de coût prédit **quantification ≈ 0** (la recherche de plus proche
voisin sur le réseau disparaît, il ne reste qu'un arrondi), factorisation
≈ 151 s, capture ≈ 100 s — soit **~4,4 min** contre 26,7 pour Leech. Si la
quantification scalaire ne s'effondre pas, **le budget de la grille est faux** et
le §4 du pré-enregistrement est à refaire avant de lancer 68 runs.

**3. La baseline à 73 fenêtres — contrôle C4.** Les trois runs doivent imprimer
la **même** ligne `baseline (f32) ppl = …`, et cette valeur devient la référence
figée du contrôle. ⚠️ **Ce ne sera pas 19,5038** : ce chiffre est celui de **12**
fenêtres à ctx 2048. La valeur à 73 s'établit ici et ne rebouge plus.

**4. Le volume réellement lu — contrôle C2.** Chaque run doit porter
`64 windows of 2048 = 131072 tokens (N available)`. À ×1 il n'y a aucun risque
de plafond ; la ligne sert de témoin de forme avant que ×16 et ×32 la sollicitent
vraiment.

### Ce que le lot produit

Un journal `docs/mesures/variance-etage0-<date>.txt` portant : le verdict de
Gate A avec les deux `exact-ppl` côte à côte, le profil par phase du bras
scalaire, la baseline à 73 fenêtres, et — s'il y a lieu — la correction du modèle
de coût. Plus les trois logs bruts dans le même répertoire : c'est ce format qui
a permis, trois jours après le gate du 2026-08-25, de retrouver un profil par
phase que personne n'avait pensé à extraire.

---

## Lot 2 — La grille 0.6B (~28,3 h de Mac + ~4 h d'évaluation, 0 $)

**68 runs**, dans cet ordre pour que le premier résultat lisible arrive tôt :

1. `leech1c12`, ×1 et ×8, les 4 graines — **8 runs, ~4,3 h**. Donne σ_diff et un
   premier point de pente.
2. `int3g24`, ×1 et ×8, les 4 graines — **8 runs, ~1,4 h**. Donne le
   contraste de famille, qui est la prédiction P4.
3. Les ancres bits : `int2g24` et `int4g24` à ×1 et ×8 — **16 runs, ~2,7 h** —
   puis le contrôle de granularité `int3g128` à ×1 — **4 runs, ~18 min**.
4. Le remplissage de la courbe : les deux bras porteurs aux volumes ×2, ×4, ×16,
   ×32 — **32 runs, ~20 h**.

⚠️ **L'étape 4 est la seule longue, et c'est la dernière exprès** : si les
étapes 1-2 montrent que σ_diff est dégénéré, elle ne se lance jamais.

**Livrable** : `docs/mesures/variance-grille-0.6b-<date>.txt` plus les logs bruts
dans un sous-répertoire — le format du lot du 2026-08-25, qui a permis de
retrouver le profil par phase trois jours après.

---

## Lot 3 — Le contrôle d'invariance de backend (~4 $)

Deux cellules répliquées sur `l40sx1` : `leech1c12` ×1 et `int3g24` ×1,
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
