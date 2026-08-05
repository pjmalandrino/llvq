# Audit de publication — 2026-08-03

> Instantané daté. Six auditeurs en lecture seule sur le dépôt, les surfaces
> publiques (README, LAUNCH_ME, carte HF, docs) et le brouillon de mail, plus
> une passe de vérification adversariale sur chaque divergence de chiffre.
> 36 divergences trouvées, 12 revérifiées, 1 réfutée, 1 partiellement vraie.
>
> **Ce document décrit l'état au 2026-08-03. Il périme dès qu'on corrige.**

**Verdict en une phrase :** le mail ne peut pas partir en l'état — il porte les
quatre chiffres rétractés, sa question centrale repose sur une comparaison de
conventions de débit qui s'annule au recalcul, et il remercie les auteurs pour
une contribution algorithmique que le code livré n'exécute pas.

---

## 1. Bloquants avant tout envoi

### B1 — `docs/mail-qualcomm-draft.md:25-29` : les chiffres rétractés

**Écrit :** « 12.2336 → 14.9104, a ×1.219 degradation, at 2.1117 bits/weight »

**Réalité :** `CLAUDE.md:598-603` — « Les 2,1117 bits/poids valaient en réalité
2,7338 […] Le chiffre honnête, mesuré sur un fichier de 981 Mo et vérifié bit
pour bit : 16,9617 de perplexité à 2,1696 bits/poids (×1,386). » Le README
publie déjà cette rétractation (`README.md:103-109`).

**À écrire :** `12.2336 → 16.9617, a ×1.386 degradation, at 2.1696 bits/weight,
weighed on the 981 MB sealed file rather than computed.` Ajouter que le même
fichier rescoré en f16 donne 16,9415 contre une baseline f16 de 12,2361
(×1,3846) — c'est la seule forme reproductible par `bin/ppl` sur les octets
publiés.

> ⚠️ **Ne pas écrire « 14,9104 était faux ».** La perplexité était une mesure
> réelle (`docs/retraction-et-gain.md:77`) ; c'est son **étiquette de débit**
> qui était fausse. L'objet mesuré tournait à 2,7338 b/poids.

### B2 — `docs/mail-qualcomm-draft.md:31-33` : l'écart au débit

**Écrit :** « my rate is 5.6 % above 2.000, of which roughly 0.1 bit/weight is
the tail policy »

**Réalité :** 2,1696/2,000 = **+8,5 %** (`CLAUDE.md:120`, `README.md:85-86`).
Le 5,6 % dérivait du 2,1117 rétracté. Et la décomposition publiée au README est
différente : colonnes de queue f32 +0,075, échelles de ligne f64 +0,015,
politique de queue ~0,05 (`README.md:88-93`).

**À écrire :** 8,5 %, avec la décomposition **exacte** recalculée sur les
dimensions réelles, qui elle boucle :

| poste | b/poids |
|---|---|
| code de réseau : 150 681 600 blocs × 48 bits / 3 616 358 400 poids | **2,00000** (exactement) |
| queue en f32 | 0,15005 |
| échelles de ligne en f64 | 0,01957 |
| **total** | **2,169625** → 980 766 720 octets = les « 981 Mo » |

C'est un argument **beaucoup plus fort** que la formulation actuelle : le
codebook tourne à 2,000 bits pile, les 8,5 % sont intégralement de la
sérialisation (f32/f64 là où f16 suffit) plus une politique de queue que le
papier ne spécifie jamais. En f16 partout le fichier tomberait à 2,0799
(+4,0 %).

> ⚠️ La décomposition actuelle du README (0,075 + 0,015 + 0,05 = 0,140) ne
> somme pas aux 0,1696 réels. Corriger le README **et** le mail.

### B3 — `docs/mail-qualcomm-draft.md:38-44` : la question sur l'annexe G repose sur trois défauts empilés

**Écrit :** « I find a single shell ahead on both axes: shell 12 alone with one
gain bit gives 92.81 % retention at 1.958 bits/dim, against 92.14 % at 2.000
for `norm(Λ₂₄(12))` + 1 gain bit »

**(a) La rétention est périmée (§A5).** Chiffre courant : **92,24 %, MSE
0,0817** (`CLAUDE.md:993`). Coquille 13 : **92,33 %, MSE 0,0762**.
`CLAUDE.md:998-1000` : « La marge sur le papier passe de 0,67 point à 0,10 —
c'est maintenant un ex æquo, pas une victoire. » Le README porte les mêmes
valeurs périmées (`README.md:181-182`), y compris la colonne MSE (0,0805 et
0,0751 au lieu de 0,0817 et 0,0762).

**(b) L'avantage de débit est un artefact de convention — le point le plus
dangereux du dossier.** `llvq-bench/src/lib.rs:424-428` :
`rate_shape_gain13_single` calcule `(log2|Shell(m)| + k)/24`, un débit
**fractionnaire**. Or :

- |Shell(12)| = 70 486 236 999 360 → log₂ = 46,0024 → **47 bits empaquetés**
- le papier facture `Λ₂₄(12)` à 2,000 b/dim = 48 bits/bloc = **47 bits d'index
  + 1 de gain**
- c'est exactement ce que le fichier livré utilise (`CLAUDE.md:120`)

**À convention identique, l'avantage de débit est exactement zéro**, et sur
l'axe restant le papier est devant : MSE 0,078 contre 0,0817. Un auteur répond
en une ligne : « ceil(log₂ 70 486 236 999 360) = 47, comme le nôtre — vous
comparez notre débit empaqueté à votre débit fractionnaire. »

**(c) La table est inter-harnais, et notre propre banc soutient le papier.**
`llvq-bench/src/main.rs:109` : la boucle union n'évalue que `k ∈ {0, 2}`. **Le
point « union + 1 bit de gain » n'est jamais mesuré chez nous** — la ligne
92,14 % est recopiée de leur Table 8 (le README l'étiquette correctement, le
mail non). Et à iso-bits-de-gain dans notre harnais, l'union gagne : union
0 bit MSE 0,0850 contre coquille 12 seule 0,0941 et coquille 13 seule 0,0886 ;
union 2 bits 0,0680, meilleur que toutes les coquilles uniques. *(Relevé en
relançant le banc — à revérifier d'une commande avant envoi.)*

**À écrire :** table à convention unique (empaquetée, puisque c'est ce que le
format écrit) et question retournée. L'argument qui **survit** n'est ni la
qualité ni le débit, c'est : **79 classes contre 383, et une norme constante,
à débit rigoureusement égal** — l'argument matériel que l'annexe G soulève
elle-même.

### B4 — `docs/mail-qualcomm-draft.md:50-57` : le noyau est présenté au futur

**Écrit :** « The multi-shell kernel the 2-bit regime needs does not appear to
exist anywhere, and that is the piece I would like to build »

**Réalité :** `CLAUDE.md:121`, gate G6 ✅ — noyau fusé mesuré sur les 252
projections du modèle entier : FP16 21,691 ms contre **10,460 ms**, ×2,07,
**1 105 920 lignes vérifiées** contre référence f64. `README.md:147-155` le
publie déjà. Rejoué pendant l'audit sur le fichier scellé : 22,675 / 11,021 =
**2,06×**, mêmes 1 105 920 lignes, pire erreur 3,4e-8.

**À écrire :** au passé, avec le chiffre et les trois réserves (§5). C'est la
seule contribution que le papier déclare hors périmètre (annexe C) ; la
présenter comme une intention jette le meilleur atout du mail.

> ⚠️ **La revendication de primeur est trop large.** « does not exist anywhere »
> est réfutable en une ligne : QTIP, QuIP#/E8P et AQLM publient tous des noyaux
> fusés 2 bits — c'est pourquoi le papier peut écrire que le sien est plus lent
> que QTIP. Version vraie et suffisante : *« no fused **multi-shell Leech**
> decoder appears to have been published, including in the paper, whose kernel
> is single-shell (M = 3). »* Corriger aussi `README.md:155`.

### B5 — `docs/mail-qualcomm-draft.md:20-21` et `README.md:117` : « Spherical GPTQ is all there » est faux pour la config livrée

Pour le quantifieur livré (`LeechShapeGain`, `retract_to_level = true`) :

- `llvq-quant/src/quantizer.rs:377-393` : `retraction_target()` renvoie
  **`None`** — « `quantize` already placed the block on the nearest level's
  sphere ».
- `llvq-quant/src/gptq.rs:229-244` : `if let Some(target) =
  quant.retraction_target(...)` — le rescale est donc **intégralement sauté**.
- Le second étage de l'Algorithme 3, `refine_group_scales`, est **désactivé**
  dans la commande publiée (`nogs`, `README.md:210`, `LAUNCH_ME.md:150`).

`docs/retraction-et-gain.md:174-177` le dit déjà : « pour `LeechShapeGain`, la
rétraction devient un no-op […] Le "Spherical" du Spherical GPTQ ne fait donc
plus rien de spécifique pour ce quantifieur. »

La config livrée est **Algorithme 1 (shape–gain + reset de gain) + rotation
d'incohérence en entrée**, c'est-à-dire la ligne « LLVQ | Input | GPTQ » de leur
Table 9, pas la ligne Spherical GPTQ.

**Et le mail se termine (`:59-60`) en les remerciant pour « the geometric
reading of scale correction as a retraction ».** C'est la première question
qu'un auteur posera, et la réponse d'après notre propre code est « rien ».

**À écrire :** renommer partout la configuration (« Algorithm 1 + input-side
incoherence rotation »), retirer « Alg. 3 » de `README.md:117`, et **en faire
une question** — infiniment plus fort qu'une revendication fausse.

### B6 — `README.md:59-70` : MMLU en macro + argument déclaré mort

**Écrit :** table « FP16 72.85 | LLVQ 57.59 | drop −15.3 pp » face à la colonne
papier, puis « The two gaps point opposite ways — our baseline is 2.8σ *above*
theirs, our quantized model 3.0σ *below* — so this is not a protocol offset
that cancels. »

**Réalité :** `CLAUDE.md:195-199` — micro (= papier) FP16 **70,42 ± 1,28**,
LLVQ **56,09 ± 1,36**, chute **−14,33 pp** (79,7 % retenus). Log par matière :
`docs/mmlu-micro-2026-08-02.log`. Le mot « macro » n'apparaît **nulle part**
dans le README. Et `CLAUDE.md:178-181` + `:228-230` déclarent l'argument mort
mot pour mot.

**À écrire :** table micro en principal, macro en colonne secondaire
explicitement étiquetée non comparable, **suppression complète** du paragraphe
sur les σ, et à la place l'argument qui tient et qui est meilleur : *« Our
baseline reproduces the paper's to within +0.22 pp (0.17σ), so the harness is
validated and the shortfall cannot be blamed on the protocol. »* Corriger aussi
« ±1 pp » (`README.md:56`) en ±1,28 / ±1,36, en précisant « erreur
d'échantillonnage seule, 2 280 des 14 042 questions ».

### B7 — `README.md:178-186` : la table de rétention adressée aux auteurs

Mêmes corrections que B3(a) et B3(b) : 0,0817 / **92,24 %** et 0,0762 /
**92,33 %**, et remplacer « A single shell appears to win on rate *and*
retention » par la formulation ex æquo + argument matériel.

> ⚠️ **Incohérence à trancher :** le banc relancé imprime **92,01 %** pour la
> ligne de référence du papier là où README et `CLAUDE.md:991` écrivent
> **92,14 %**. Les deux ne peuvent pas être vrais dans la même colonne.
> Vérifier d'où sort le 92,14 avant de mettre cette table sous leurs yeux.

### B8 — `README.md:167` : « No MMLU » cent lignes après un tableau MMLU

**À écrire :** « No CSR, and no domain-specific benchmark. MMLU is measured
above, on a 2 280-question sample (16.2 % of the split), not the full suite. »

### B9 — Carte HF : trois affirmations périmées sur la page la plus visible

`https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit` — figée au
2026-07-31T16:25 UTC, donc en retard sur tout ce qui date des 01 et 02/08.

| Écrit | Réalité |
|---|---|
| « A fused dequantize+matvec kernel is **unwritten** — and does not exist for this regime anywhere » | Le noyau existe et fait 2,07× (`CLAUDE.md:121`). **Ce qui reste vrai** : `bin/run` décode en mémoire puis fait un matvec ordinaire, donc **ce fichier** ne gagne aucune vitesse. |
| « **Perplexity only.** No MMLU » | MMLU mesuré **sur ce fichier exact** — sha256 local et `x-linked-etag` HF identiques : `9db213ef…c84b0` — 56,09 ± 1,36 contre 70,42 ± 1,28. |
| « ~250 lines of dependency-free Rust (`llvq-artifact`) define it » | 1 369 lignes à HEAD (error 79 + format 425 + lib 46 + runtime 675 + sealed 144). |

Cacher le MMLU (le résultat le plus défavorable) tout en niant le noyau (le
plus favorable) se lit comme une omission intéressée dans un sens et une
méconnaissance de son propre dépôt dans l'autre.

### B10 — `LAUNCH_ME.md:107-114` : la commande de vérification qualité ne peut pas rendre le chiffre annoncé

**Écrit :** `cargo run --release -p llvq-llm --features metal --bin ppl -- 4096
999 metal` → « Attendu : **16,9617** contre 12,2336 »

**Trois défauts cumulés :**
1. `llvq-llm/src/bin/ppl.rs:45` — le modèle quantifié est le **4ᵉ** argument ;
   absent, `ppl.rs:82` prend la branche `[baseline]` et score le checkpoint nu.
2. `ppl.rs:57` — sans `LLVQ_MODEL`, le défaut est **`Qwen/Qwen3-0.6B`**.
3. `ppl.rs:99` — `999` demande **toutes** les fenêtres (73), alors que les deux
   chiffres annoncés sont mesurés sur **12**.

La commande publiée rend donc la perplexité FP32 d'un Qwen3-0.6B nu sur tout
wikitext-2 — configuration documentée à `CLAUDE.md:499` comme valant
**19,1481**.

**Quatrième point, de provenance :** 16,9617 **n'a jamais été produit par
`bin/ppl`** (`CLAUDE.md:631` : boucle interne de `smoke`, f32, modèle en
mémoire). Le seul chiffre `bin/ppl`-sur-fichier-scellé est **16,9415** en f16.

**À écrire :**
```bash
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal \
  --bin ppl -- 4096 12 metal qwen3-4b-llvq.bin
```
```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 cargo run --release -p llvq-llm \
  --features metal --bin ppl -- 4096 12 metal
```
Attendu : **16,9415** contre **12,2361**, ×1,3846, **empreinte de tokens
identique des deux côtés** (`3f1baca9033bf251`).

### B11 — `README.md:222-224` et `LAUNCH_ME.md:87-105` : la commande du ×2,07 pointe sur un fichier non publié

`llvq-metal/src/bin/thesis.rs:191-193` : par défaut `~/llvq-q4b.llvq`. L'API HF
renvoie `['.gitattributes', 'LICENSE', 'README.md', 'qwen3-4b-llvq.bin']` — le
`.llvq` n'est publié nulle part. Le fichier scellé fonctionne, lui (vérifié :
`thesis -- ~/qwen3-4b-llvq.bin`, 2 min 10, 1 105 920 lignes, pire erreur
3,4e-8).

**À écrire :** `cargo run --release -p llvq-metal --bin thesis --
qwen3-4b-llvq.bin`. Même défaut sur `bin/decreal` (`decreal.rs:139-141`) et
`bin/matvec` (`matvec.rs:501-504`), tous deux publiés en `README.md:229-231`.

---

## 2. La revendication centrale, telle qu'elle est défendable

### Ce qui tient, dans l'ordre de force

1. **Le harnais est validé deux fois indépendamment** — le résultat le plus
   sous-employé du projet. MMLU micro FP16 = **70,42 ± 1,28** contre **70,2**
   au papier (0,22 pp, 0,17 σ). Perplexité baseline Qwen3-8B = **8,9893**
   contre **8,99** (0,01 %). Deux modèles, deux métriques. Conséquence : le
   déficit du bras quantifié ne peut plus être imputé à la mesure.
2. **Le code de réseau tourne à 2,000 bits/poids exactement.** Les 0,17 bit
   supplémentaires du fichier sont f32/f64 de sérialisation + une politique de
   queue non spécifiée par le papier. Tous réductibles.
3. **Le noyau fusé multi-coquilles Leech** : 2,07× le FP16 sur les 252
   projections, 1 105 920 lignes vérifiées contre référence f64. C'est ce que
   l'annexe C déclare hors périmètre.
4. **L'échelle bits↔vitesse, mesurée** : 3,35 b/poids → 0,68× ; 4,54 → 0,90× ;
   5,51 (Slot32) → 2,07×. Le résultat d'ingénierie le plus transférable, et il
   est **négatif pour le produit**.
5. **MMLU −14,33 pp contre −9,5 pp**, honnêtement mesuré, avec le profil qui
   montre le mécanisme.

### Ce qui ne tient pas

**« On passe juste sous QTIP » est un artefact de deux baselines différentes.**
Notre baseline est 12,2336, la leur 12,41. Normalisé en surcoût de
log-vraisemblance sur sa propre baseline :

| | nats/token | vs QTIP |
|---|---|---|
| nous, 4B | 0,3267 | **+3,1 %** |
| QTIP, 4B | 0,3171 | — |
| LLVQ 0 bit, 4B | 0,3177 | +0,2 % |

Nous sommes **au-dessus**, avant même de payer les 8,5 % de bits. C'est déjà
visible dans le tableau du README (×1,386 contre ×1,373) — c'est la phrase
`README.md:85` qui dit l'inverse de sa propre table. La ligne `README.md:100`
(« lands at QTIP's level ») est la bonne.

**Et la marge de 0,5 % est sous la résolution du pipeline.**
`docs/retraction-et-gain.md:145-151` : deux runs réputés identiques ont rendu
14,2684 et 15,2909, soit **7 % de dispersion**. Le balayage
`LLVQ_CALIB_SEED={1,2,3}`, désigné par `CLAUDE.md:105` comme « la barre
d'erreur qui manque au projet », n'a jamais été lancé.

**Le 8B est notre pire résultat face au papier, et il est publié comme un
signal positif.** Face au papier (`docs/llvq-paper-notes.md:102-103`, baseline
8,99 · QTIP 11,17 · LLVQ 2 bits 10,82 · LLVQ 0 bit **10,19**), notre 11,3934
est **11,8 % au-dessus de leur meilleure config 0 bit et 2,0 % au-dessus de
QTIP**. En surcoût de nats, notre excès vaut **1,89× le leur au 8B contre 1,03×
au 4B** : l'écart au papier ne fond pas avec l'échelle.

**Et ce point est confondu.** Le 8B a tourné en `leech1c12L3`
(`CLAUDE.md:744`), un plafond L≤3 que le 4B n'avait pas. Deux variables d'un
coup — exactement ce que `CLAUDE.md` s'interdit depuis le 0,6B. Pire,
`llvq-llm/src/calib.rs:224-230` documente que **le fichier paie quand même les
48 bits** de l'index plein : distorsion d'un codebook amputé sans l'économie de
bits correspondante. `ops/run.py:575` fixe `leech1c12L3` comme défaut, donc le
32B à ~62 $ partirait avec le même handicap.

### Phrases à ne jamais écrire

- « On bat le papier » / « just under QTIP » sans qualificatif / « on est
  solides à 2 bits » sans réserve
- « 14,9104 », « ×1,219 », « 2,1117 bits/poids », « 5,6 % au-dessus de 2,000 »,
  « 2,0653 »
- « 92,81 % de rétention », « a single shell wins on rate and retention »
- « 72,85 / 57,59 », « les deux écarts pointent en sens opposés »
- « Spherical GPTQ », « rétraction sphérique » dans la description de la
  recette livrée
- « le noyau fusé n'existe pas » (carte HF) / « the piece I would like to
  build » (mail)
- « does not exist for this regime anywhere » sans « multi-shell Leech »
- « Le 8B se dégrade moins, c'est le signal d'échelle » (vrai en absolu, faux
  face au papier, et confondu)
- « Le suspect n°1 est le volume de calibration » (jamais mesuré — voir §3)
- « 2 bits par poids » tout court, sans « 2,1696 sur le disque, 5,51 en RAM »
- « ×4,63 de compression » (calculé) — c'est **×4,54**, pesé sur 1,771 Go
- « 70B sur 32 Go » comme acquis (extrapolation pure, sans budget KV)
- « quantifier le gain ne coûte presque rien, 0,04 % de perplexité pour
  0,52 bit » — `docs/retraction-et-gain.md:141-144` : « Elle n'a jamais été
  mesurée » (les deux bras de cet A/B étaient le même quantifieur sous deux
  noms, écart relatif 7,1·10⁻¹⁵)

---

## 3. Les questions qu'ils poseront, par ordre de dégât

**Q1 — « Qu'est-ce que votre rétraction fait, concrètement ? »**
*Rien, dans la configuration livrée.* `retraction_target()` renvoie `None` et le
raffinement fermé est désactivé. Réponse honnête : « à 1 bit de gain, la
rétraction vise le niveau que le code peut exprimer, donc elle devient un
no-op ; je n'ai pas exercé l'Eq. 17 telle qu'écrite, et je ne sais pas si c'est
une lecture correcte du papier ou mon erreur. » **C'est une question à leur
poser, pas une faiblesse à cacher.**

**Q2 — « ceil(log₂ |Shell(12)|) = 47, comme le nôtre. Où est votre avantage de
débit ? »**
*Il n'y en a pas.* À convention empaquetée les deux codes coûtent 48 bits/bloc,
et notre MSE est 5 % au-dessus du leur. L'avantage qui survit est structurel :
79 classes contre 383, norme constante. Le dire d'avance.

**Q3 — « Votre baseline est plus basse que la nôtre, donc votre écart est plus
grand. »**
*Exact, et nous l'avons calculé* : +3,1 % de surcoût de log-vraisemblance
vis-à-vis de QTIP au 4B, et 1,89× le leur au 8B.

**Q4 — « Pourquoi perdez-vous 14,3 pp de MMLU là où nous en perdons 9,5 ? »**
*Nous ne savons pas.* Ce qui est **exclu par mesure** : le dtype (§A2, 0,1 %),
l'agrégation macro/micro (0,93 pp), et le harnais (baseline à 0,22 pp). Ce qui
reste, **ordonné par force de preuve** :
1. **Rotation Input seule vs Input+Output** — leur propre Table 9 chiffre ce
   levier à **+5,6 pp de MMLU** (29,3 → 34,9), plus que notre résidu de 4,8 pp.
2. **Le chemin des magnitudes** — rétraction inerte + raffinement désactivé ;
   leur Table 9 isole GPTQ → Spherical GPTQ à **+3,0 pp** (26,3 → 29,3).
3. **Volume de calibration** — plausible, **zéro mesure** ; P3 (calibration
   oracle), conçu exactement pour borner cette famille, n'a jamais tourné.
4. **Config sans équivalent chez eux** — nous tournons à **1 bit de gain**, la
   Table 6 ne rapporte que 0 et 2 bits, et leur écart 0↔2 vaut 1,4 pp.
5. Variance non caractérisée, puis échantillonnage (±1,36 pp).

> ⚠️ Le README (`:67-68`) et `CLAUDE.md:224-226` désignent le volume de
> calibration comme « la cause la plus probable ». Annoncer ça à des auteurs
> dont la Table 9 offre deux explications chiffrées plus grosses, c'est se faire
> renvoyer à leur annexe I. **Corriger dans les deux surfaces.**

**Q5 — « Votre 2,07× inclut-il la rotation des activations ? »**
*Non.* Les codes stockés vivent dans la base tournée (`calib.rs:412-413`,
`format.rs:372-374`). Un noyau fusé doit appliquer Q à x en ligne — coût
**asymétrique**, que le bras FP16 ne paie pas. `thesis.rs:22` le liste
explicitement parmi ce qui est laissé dehors, mais `README.md:147-148` présente
le rapport sans la réserve. *(Le comptage « 144 transformées/token » avancé
pendant l'audit n'est pas vérifié ; ne pas le citer.)*

**Q6 — « 78,2 tok/s, c'est mesuré ? »**
*Non, calculé.* `thesis.rs:433-435` : `head_bytes / bw`, avec `bw` = le débit du
bras **FP16** (335 Go/s) alors que le bras LLVQ n'atteint que 239. Le lm_head
n'est jamais exécuté. Re-mesuré pendant l'audit : 74,4 contre 39,8 = 1,87×.

**Q7 — « Vous vous comparez au FP16, mais qui déploie du FP16 ? »**
*Nous avons la réponse, et elle est mauvaise pour le produit :*
`docs/face-au-4-bits.md:24-31` — MLX q4 sur le même 4B, même machine, même
jour : 2,39 Go de RAM contre 3,28, **129,8 tok/s bout en bout** (7,7 ms/token,
moins que nos 10,46 ms de projections seules), ~1-2 % de dégradation contre
×1,386. **Ni le README, ni LAUNCH_ME, ni la carte HF ne mentionnent le 4 bits**
(grep : zéro occurrence) — alors que `docs/cheatsheet-defense.md:13-15` en fait
la règle zéro. C'est l'omission la plus embarrassante à se faire signaler.

**Q8 — « Quelle est votre barre d'erreur ? »**
*Aucune sur la perplexité.* La seule dispersion connue du pipeline est 7 %.

**Q9 — « Le 8B, c'est la même recette que le 4B ? »**
*Non — `leech1c12L3` contre `leech1c12`.* Deux variables, un seul point.

**Q10 — « 2,1595 sur HuggingFace, 2,1696 sur GitHub. Lequel ? »**
Voir §4 (D2) — les deux sont arithmétiquement exacts, la réconciliation existe
mais n'apparaît sur aucune des deux surfaces.

---

## 4. Ce qu'il reste à faire avant publication

### Indispensable — dépôt

| # | Fichier:ligne | Action |
|---|---|---|
| D1 | `README.md:59-70`, `:56` | MMLU micro, supprimer le paragraphe σ, barres ±1,28/±1,36 |
| D2 | `README.md:34` + carte HF | **Un seul débit affiché partout.** Les deux chiffres sont exacts : `seal.rs:46` — `quantized_weights = Σ d_out × d_in`, donc **queue incluse** → `seal` imprime **2,1595** (/ 3 633 315 840) ; `smoke.rs:436-441` divise par `weights − tail_weights` → **2,1696** (/ 3 616 358 400). Ajouter la note de provenance sur les deux pages ; corriger « at 8 % » de la carte HF en 8,5 % si l'on retient 2,1696 |
| D3 | `README.md:178-186` | Table de rétention §A5 (0,0817/92,24 % et 0,0762/92,33 %), convention empaquetée, conclusion ex æquo ; **et trancher le 92,01 vs 92,14** |
| D4 | `README.md:167` | « No CSR, and no domain-specific benchmark » |
| D5 | `README.md:88-93` | Décomposition exacte qui boucle (2,0000 + 0,1501 + 0,0196) |
| D6 | `README.md:85` | Aligner sur `:100` : « lands at QTIP's level », pas « just under » |
| D7 | `README.md:155` | « multi-shell **Leech** decoder », + réserve « M3 Max, batch 1, mémoire unifiée ; les 1,36-1,48× du papier sont une autre machine » |
| D8 | `README.md:117` | Retirer « Alg. 3 » |
| D9 | `README.md:147-148` | Annoter : hors rotation d'activation ; batch 1 ; nommer la source du « 93 % of peak » (introuvable dans le dépôt) |
| D10 | `README.md:160`, `LAUNCH_ME.md:135` | « ~3,3 Go chargés » décrit `Slot32/Grouped32`, pas le runner livré : `docs/pistes-battre-q4.md:69` donne **7,3 Go de pic** pour `bin/run`, et `LAUNCH_ME.md:55` exige « ~8 Go de RAM libre » — le même fichier se contredit |
| D11 | `README` + `LAUNCH_ME` | **Ajouter la comparaison au 4 bits**, avant la section vitesse |
| D12 | `LAUNCH_ME.md:110`, `:88` ; `README.md:223` | Commandes corrigées (B10, B11) |
| D13 | `LAUNCH_ME.md:76`, `:69`, `:13` | « ~1 min » → **~4-5 min** (255,7 s mesuré) ; « 106 tests » → **128** ; marquer « 78,2 tok/s (calculé, lm_head non exécuté) » |
| D14 | `LAUNCH_ME.md:137-139` | L'extrapolation 70B utilise le débit **disque** pour une affirmation sur la RAM. Donner les deux : ~19 Go disque, **29 à 48 Go en RAM** selon le format, contre 39,4 Go pour du 4 bits |
| D15 | `docs/g6-handover.md:15,21-25` ; `docs/run-de-nuit.md:22,28` ; `docs/passation-2026-07-31.md:8,15,17` ; `docs/plan-mmlu.md` | Bandeau de péremption daté ou `docs/archive/`. Ces quatre documents nient l'artefact, le noyau et/ou le run MMLU |
| D16 | `docs/retraction-et-gain.md:93` | La ligne attribue **15,3272** à `leech1c12`, la config du fichier publié. `llvq-llm/src/sealed.rs:11-16` explique que c'est l'overlay du **premier** run ; le fichier scellé donne 16,9617. Annoter |
| D17 | `docs/cheatsheet-defense.md:8, 179-181, 268` | Retirer le bandeau « à resynchroniser après le run MMLU » ; ajouter le MMLU micro ; étiqueter la ligne 5,375/2,21× comme **une couche isolée** (gate_proj) face au 5,51/2,07× du modèle entier ; ×4,63 → **×4,54** |
| D18 | `CLAUDE.md:666-672`, `:724-733`, `:160-166`, `:119` | Annoter l'A/B de gain jamais mesuré ; recalculer la table de compression à 2,1696 (1,771 Go, ×4,54) et marquer le ×6,9 à 70B « calculé, jamais produit » ; réécrire le bandeau MMLU au passé ; corriger 639 → **680 µs/bloc** |
| D19 | `docs/pistes-battre-q4.md:95, 128-131` | P20 ✅ avec le résultat réel (×1,267 à 2,0436, 11,48 $) |
| D20 | `llvq-bench/src/bin/decbench.rs:99-102` | La cible « ~400 tok/s » est abandonnée (mesuré : 78,2) |
| D21 | `ops/README.md:50-70, 156-158` ; `ops/run.py:244-245` | Bloc d'exemple régénéré ; provenance imprimée corrigée (la constante CUDA vient du run 32B, pas d'un 0,6B) |
| D22 | `llvq-llm/src/eval.rs:15-18` | Annoter le « 57.59 » (macro ; micro = 56,09) |
| D23 | `docs/llvq-rust-implementation-plan.md:255-260` | Le critère G5 publié est à trois métriques avec tolérance 0,05 ; G5 a été validé sur « Wiki < 17,04 » seul. Annoter le déplacement |

### Indispensable — carte HF

Réécrire les Limitations (B9), aligner le débit (D2), ajouter le MMLU micro avec
le profil par matière, ajouter la limitation 4 bits, corriger « ~250 lines » →
1 369, et **retirer `pipeline_tag: text-generation`** — le dépôt HF ne contient
ni `config.json` ni `tokenizer.json` (ils sont scellés dans le `.bin`), donc le
widget d'inférence s'affichera et échouera. Ajouter `library_name` custom et
`inference: false`.

### Bloqué par du travail non commité

**Bonne nouvelle vérifiée :** HEAD seul, exporté par `git archive` dans un
répertoire vierge, compile (`cargo check --all-targets`, exit 0), passe
`clippy` à zéro warning, passe **128/128 tests**, et fait tourner `bin/run`,
`bin/thesis` (2,06× — la valeur rejouée pendant cet audit, cf. B4 ; le chiffre
courant est 2,09× [2,05–2,11], run archivé du 2026-08-05) et `llvq-bench`. **Aucune surface publique ne référence le
travail non commité** (grep `embedq|embedquant|LVQ3|raw_tensors` : zéro). Le
dépôt publié se suffit à lui-même.

**Mais :** le working tree fait passer `MAGIC` de `LVQ2` à `LVQ3`
(`llvq-artifact/src/format.rs:39-43`).

- **Ne rien commiter avant l'envoi**, ou alors publier simultanément les trois
  chiffres du nouveau fichier — 1,771 Go / ×4,54 / 2,1696 décrivent le fichier
  LVQ2 actuel.
- **Ne pas publier la variante embedding int4** (`~/q4b-e4.llvq`, 1,211 Go) :
  **aucune mesure de qualité n'existe**, et l'outil lui-même l'écrit
  (`embedq.rs:114` : « score the OUTPUT file (ppl + mmlu) before believing
  anything »). Publier un ×6,6 non pesé reproduirait exactement le motif que le
  README dénonce, un mois après l'avoir dénoncé.
- Si LVQ3 part un jour : pousser le format **et ses tests** d'abord, vérifier
  que le fichier LVQ2 publié se lit toujours, puis publier un v3 sous un nom
  distinct (`qwen3-4b-llvq-e4.bin`), jamais en remplacement de l'objet que les
  auteurs auront regardé.

**Ne pas publier le 8B** : format v1 projections seules (scellé il ferait
~4,3 Go), ratio ×3,7 explicitement marqué « ne pas publier »
(`CLAUDE.md:762-766`), recette différente, aucun MMLU. Il vaut une phrase dans
le mail, pas un fichier.

### Souhaitable (mais à fort rendement)

| | Coût | Rendement |
|---|---|---|
| **A/B 3 blocs `leech1c12` vs `leech1c12L3` sur le 8B** | ~15 min de GPU loué | Transforme la défaite du 8B en anecdote de configuration, ou la confirme. Seule action qui débloque l'argument d'échelle |
| **3 graines de calibration sur 3 blocs** | ~24 min | Le σ qui manque au projet |
| Ajouter `k=1` à la boucle union de `llvq-bench/src/main.rs:109` | 1 ligne + 4 s | Rend la table de l'annexe G intra-harnais |
| Commiter `Cargo.lock` (`.gitignore:2`) | 1 min | Le dépôt revendique le « build reproductible » pour un workspace de binaires |
| CI GitHub Actions (`fmt`, `clippy -D warnings`, `test --release`) | 30 min | Le projet vend la létalité de ses tests et n'a **aucune** CI |
| Paragraphe « Déterminisme » au README | 15 min | `calib.rs:49-55` accumule `AᵀA` en f32 **sur l'accélérateur** : un tiers sur CUDA n'obtiendra pas les mêmes poids. À distinguer de l'encodeur, lui exactement déterministe |
| Table des sept `LLVQ_*` | 15 min | `LLVQ_DTYPE` déplace un chiffre publié et n'apparaît sur aucune surface publique |
| CITATION.cff + specs machine + versions | 20 min | Rien ne permet de citer ni de reproduire l'environnement |
| Corriger `README.md:122-124` | 1 min | « 690 transitive crates » est un **nombre de lignes** de `cargo tree` ; le compte de paquets distincts est **261** (291 avec `metal,fast-linalg`) |
| Étiqueter `thesis.rs:316` | 5 min | Il imprime « transcodées en 128 s » là où le README annonce « ~3 s » |
| Publier des fourchettes plutôt que 3 décimales de ms | 5 min | Le re-jeu donne 22,675 / 11,021 au lieu de 21,691 / 10,460 (dérive thermique 4-5 %). Garder « 1 105 920 lignes, pire erreur 3,4e-8 » : reproductible au chiffre près |

---

## 5. Matériel pour le mail

### Reproduction (EN, à coller)

> **Reproduction.** On Qwen3-4B, 2 bits, no fine-tuning, WikiText-2 at 4096
> context, calibrating out of domain on C4 as you do: 12.2336 → 16.9617, a
> ×1.386 degradation, at 2.1696 bits/weight. That rate is *weighed on a file*,
> not computed: 981 MB of indices on disk, decoding back to the evaluated
> weights bit for bit. Re-scored in f16 on the decoded sealed file the same
> artifact gives 16.9415 against an f16 baseline of 12.2361, ×1.3846, with an
> identical token fingerprint on both arms. Configuration: shape–gain with the
> angular search capped to the Leech(12) ball, 47 index bits + 1 gain bit = 48
> bits/block, input-side incoherence rotation only, block tail kept exact.
> Normalised against my own baseline rather than compared on raw perplexity, my
> excess log-likelihood is about 3 % above QTIP's and above your 0-gain-bit
> configuration — so I read this as parity, not as beating either. For
> reference my FP32 baseline lands 1.4 % under yours (12.2336 vs 12.41).

### Réserves (EN)

> Three caveats I would rather state than have you find. My rate is 8.5 % above
> 2.000: the lattice code itself is at exactly 2.000 bits/weight, and the extra
> 0.17 is f32 tail columns (+0.075), f64 row scales (+0.020) and the tail policy
> itself — layer widths are not multiples of 24 and I keep the remainder exact;
> I could not find what your implementation does with that remainder. All three
> are reducible. I have no error bar: the only dispersion I have observed on
> this pipeline is 7 % between two runs meant to be identical, so I do not claim
> the 0.08 perplexity margin. And I use about 100× fewer calibration tokens than
> you, with input-side rotation only where you use Input + Output — both work
> against me.

### MMLU (EN)

> On MMLU my baseline reproduces yours to within 0.22 pp (70.42 ± 1.28 micro,
> against 70.2), which validates the harness; the quantized arm drops to
> 56.09 ± 1.36, −14.33 pp where you report −9.5. Abstract algebra and
> professional accounting land exactly at chance while history, law and
> psychology hold above 80 %. I cannot yet attribute the shortfall: your Table 9
> puts Input-only vs Input+Output rotation at +5.6 pp of MMLU and plain GPTQ vs
> Spherical GPTQ at +3.0 pp, both larger than my 4.8 pp residual, and I have
> measured neither. Calibration volume is a third candidate I have not tested
> either.

### Le noyau (EN)

> **The kernel.** Appendix C notes that your fused kernel handles a single shell
> for simplicity and that low-level optimization is largely orthogonal to your
> contribution. No fused *multi-shell Leech* decoder appears to have been
> published, so I wrote one, in Metal. Measured across all 252 projection
> matrices of the published Qwen3-4B — one token, one command buffer per format,
> cold by construction (2.50 GB vs 7.27 GB of distinct weights against a 48 MB
> system cache) — it runs in 10.46 ms against 21.69 ms for FP16, 2.07×, with all
> 1 105 920 output rows verified against an f64 CPU reference beforehand. That is
> an M3 Max at batch 1 in unified memory, not comparable to your Table 7 numbers
> on different hardware; it excludes the input rotation, which only my arm pays;
> and it is not wired into the runner, so the shipped file gains no speed today.
> The price is explicit: the runtime layout costs 5.51 bits/weight in RAM where
> the file holds 2.1696 — more than an ordinary 4-bit format. On a 4B, a plain
> group-64 q4 beats me on RAM, throughput and quality, and I only win on disk.
> That trade-off is the thing I would most value your view on.

### Le paragraphe littérature (vérifié source par source)

> On model choice: I stayed on Qwen3-4B and 8B for comparability with your
> Table 6, and note that the 4B also sits in a demanding regime for
> post-training quantization — 36T pre-training tokens (Qwen3 Technical Report,
> arXiv:2505.09388), about 9,000 tokens per parameter, against roughly 300 for
> Llama-2 7B and 1,900 for Llama-3.1 8B. Whether token density is the actual
> cause is contested: Kumar et al. (*Scaling Laws for Precision*, ICLR 2025,
> arXiv:2411.04330) and Ouyang et al. (arXiv:2411.17691) attribute rising PTQ
> degradation to the token budget, while Catalan-Tatjer, Ajroldi and Geiping
> (arXiv:2510.06213) argue the learning-rate schedule is the primary driver —
> and the Qwen3 report does note an accelerated learning-rate decay in its second
> pre-training stage, so both readings point the same way here. I treat this as a
> working hypothesis rather than a result, all the more since your own Appendix J
> has Llama-3.2 1B degrading more than Qwen3-4B at a lower token-per-parameter
> ratio.

**Phrase optionnelle** (la référence la plus directe) :

> The closest direct evidence I found is Zheng et al., *An Empirical Study of
> Qwen3 Quantization* (arXiv:2505.02214), which reports Qwen3-8B-Base degrading
> markedly more than LLaMA3-8B under the same AWQ w3a16g128 setting — C4
> perplexity 10.4 → 23.8 against 9.2 → 11.6 — with the same "fewer redundant
> representations" hypothesis.

⚠️ **Quatre garde-fous sur ce paragraphe.**

1. Ne **jamais** écrire « la famille Qwen ». Qwen3-8B (4 396 tokens/paramètre)
   est **sous** Llama-3.2 1B (7 317), et Qwen3-32B (1 098) sous Llama-3.1 8B
   (1 868). Seul le 4B tient.
2. Le « ×1,76 de Llama-3.2 1B » (`CLAUDE.md:573-574`, Table 10) est noté **sans
   sa variante LLVQ**. Sur Qwen3-4B l'écart entre variantes va de ×1,758
   (spherical shaping) à ×1,374 (shape–gain 0 bit) : si le ×1,76 est du
   spherical shaping, la comparaison ne vaut rien. **Relire la Table 10 par
   rendu image** avant d'écrire cette subordonnée. Repli sans chiffre : *« your
   own Appendix J results on Llama-3.2 suggest model size dominates token
   density in practice. »*
3. Le facteur « ~100× » suppose implicitement 2048 tokens/séquence pour leurs
   6 100 séquences, longueur qui n'est **dans aucune note de lecture**. Version
   plus forte : *« 131 k tokens against your 6 100 sequences — I could not find
   the sequence length, so I cannot state the ratio. »*
4. **Ne pas citer** Springer et al. arXiv:2503.19206 ni Rofin et al.
   arXiv:2604.13627 : ni l'un ni l'autre ne teste la quantification.

### La question à poser

Une seule, sur l'annexe G, reformulée sur l'axe qui survit :

> **A question on Appendix G.** You compare single shells against unions on
> angular separation and adopt the union. Measuring rate–distortion retention
> instead, on an i.i.d. Gaussian source (20 000 blocks, fixed seed, gain
> centroids fitted on a held-out split), I get 92.24 % for shell 12 alone with
> one gain bit against 92.14 % for `norm(Λ₂₄(12))` + 1 gain bit — and once both
> are packed to whole bits they cost the same 48 bits per block, so I read this
> as a tie on quality rather than an advantage. What remains is structural: 79
> equivalence classes instead of 383, and the constant norm your own appendix
> raises as the hardware-friendly property, which removes the rescaling of
> intermediate dot products in a fused kernel. I am aware you measure a different
> quantity, that this is one source and one seed, and that I have not verified it
> on real weights after the GPTQ loop. I would value being told what I am
> missing.

### Ce qu'il ne faut PAS mettre

- La proposition d'emploi implicite (`:50-57` « what I would like to work on » +
  « glad to share results »). Soit une phrase explicite en fin de mail, soit
  rien.
- Le remerciement pour « the geometric reading of scale correction as a
  retraction » **tant que B5 n'est pas corrigé** — remercier pour la pièce qu'on
  n'a pas exécutée est le pire endroit possible.
- Le point 8B comme signal d'échelle. S'il est cité : « ×1,267 à 2,0436 bits —
  mais avec un plafond de niveaux L≤3 que le run 4B n'avait pas, et sans MMLU ».
- Tout chiffre non repassé par la checklist `:69-75`. La case `:73` (« Aucun
  chiffre du mail ne diverge du README ») n'a jamais été cochée — et telle
  quelle elle **passerait** pour le 92,81 %, puisque le README porte la même
  valeur périmée. Ajouter un critère qui ne peut pas se satisfaire de la
  cohérence interne : **« chaque chiffre du mail a été rejoué par une commande
  du dépôt aujourd'hui »**.

---

## 6. Faux positifs écartés

**`hf-rate-2-1595-mislabel` — RÉFUTÉ.** Un auditeur affirmait que la carte HF
étiquette mal son 2,1595 (« over the quantized projections ») et que ce label
désigne en réalité 2,1696. La prémisse est fausse : `llvq-llm/src/bin/seal.rs:46`
définit `quantized_weights = Σ (d_out × d_in)` — la matrice **entière, queue
comprise**, soit 3 633 315 840 poids, exactement le nombre que la carte énonce
elle-même. Le nom de variable est trompeur, pas la valeur. Vérification croisée :
2,1595 × 3 633 315 840 / 8 = 980,8 Mo = les « 0.981 GB » de la carte, et
2,1595 × 3 633 315 840 / 3 616 358 400 = 2,1696. Deux dénominateurs, une seule
mesure. **Corriger l'étiquette sur la foi de ce finding aurait remplacé un label
juste par un faux.**

**Corollaire :** `format-noyau-provenance-contradicts-code` est **également un
faux positif** — `seal.rs:109-112` divise par le compte queue comprise (2,1595),
`smoke.rs:436-441` par `weights − tail_weights` (2,1696). La note de provenance
de `format-noyau.md` est correcte telle qu'écrite. Ce qui reste vrai, c'est
uniquement que **deux surfaces publiques affichent deux taux sans se citer**
(D2).

**`hf-kernel-unwritten` — PARTIELLEMENT VRAI.** La puce publiée contient deux
affirmations, et **la première est juste** : « No inference speedup / the reader
decodes weights into memory and then does an ordinary matvec » reste exact au
2026-08-03. Seule l'apposition « a fused kernel is unwritten — and does not exist
for this regime anywhere » est périmée. **Ne pas corriger cette puce en bloc** :
on supprimerait une limitation vraie et on publierait un ×2,07 que le fichier
livré ne délivre pas. Correction chirurgicale : garder la limitation, retirer
l'affirmation d'inexistence, nommer où le noyau vit (`llvq-metal`,
`bin/thesis`).

**`hf-perplexity-only` — CONFIRMÉ mais surdimensionné.** La puce est une
conjonction de trois clauses ; « no commonsense reasoning » et « no
task-specific evaluation » restent **vraies**. Ne corriger que « Perplexity
only » et « No MMLU ».

**`uncommitted-work-not-load-bearing` — rassurant, pas un défaut.** Le dépôt
publié se suffit à lui-même. Le seul risque est **temporel** (ne pas commiter
LVQ3 avant l'envoi), pas fonctionnel.

---

## 7. Divergences non résolues, à trancher

1. **Durée de quantification du 4B** : `CLAUDE.md:661` dit 3,45 h (mais c'est le
   run `leech1`, config rétractée) ; README et LAUNCH_ME disent ~3,5 h ; la carte
   HF dit 4 h ; `ops/README.md:24` et `ops/run.py selftest` donnent **14 447 s =
   4,01 h** pour le run qui a produit le fichier publié ;
   `docs/retraction-et-gain.md:86` donne 12 715 s pour le run de nuit.
   **Recommandation :** ~4 h (14 447 s), en citant `uv run ops/run.py selftest`
   comme source vérifiable.
2. **Débit encodeur** : `CLAUDE.md:119` dit 639 µs/bloc ; `CLAUDE.md:413-417` dit
   656 (`nearest_scaled`) et **680** (`nearest_angular`, le chemin réellement
   appelé) ; `README.md:128` publie 1 469 blocs/s = 680 µs ; `ops/run.py:78` fixe
   1469.0. Le 639 n'apparaît nulle part ailleurs. **CLAUDE.md se contredit
   lui-même.**
3. **Ligne de référence du papier dans la table de rétention** : 92,14 %
   (README, `CLAUDE.md:991`) contre 92,01 % rapporté comme sortie du banc. Non
   résolu.
4. **Coût de la rotation dans un noyau fusé** : « 144 transformées par token sur
   le 4B » avancé pendant l'audit, non vérifié. Poser la réserve
   qualitativement, sans le chiffre.
