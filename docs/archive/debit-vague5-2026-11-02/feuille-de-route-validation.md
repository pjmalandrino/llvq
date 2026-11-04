# Débit LLVQ : feuille de route de validation des huit pistes

La première vague coûte 3,7 à 4,7 $ attendus (*estimé*) et 10,35 $ au pire (*calculé* sur les timeouts). Elle tient en trois semaines : cinq jobs sur carte, dont quatre sans code moteur, et cinq runs locaux. Elle mesure la dette GSM8K de la forme groupée au 4B et le t(k) du chemin à rangées. Elle mesure aussi la lecture du cache par un noyau GQA sur notre L40S, cinq inconnues du préremplissage et la forme groupée au 8B.

- Écrit le 2026-11-02 en lecture seule : aucun fichier du dépôt changé, aucun run, 0 $. Objet par défaut : Qwen3-4B scellé, L40S, lot 1, glouton, config groupée servie (`served-grouped-2026-11-02.txt`).
- Sources : les huit plongées, leurs préregs et leurs brouillons, sous `/private/tmp/claude-501/-Users-pjmalandrino-Documents-Pro-workspace-poc-llvq/35249513-f318-46c7-b902-3394594cb247/scratchpad/wave5-deep/`, base des chemins cités.
- Étiquettes : *mesuré* (journal nommé), *calculé* (arithmétique sur du mesuré ou sur le code), *estimé* (jugement, ou au moins une entrée estimée). Coûts et durées *estimés*, à 1,80 $/h sur les ancres de `docs/data/jobs.csv`.

Vocabulaire, repris du rapport de la vague 2 :
- **Rangée** : une ligne d'activation, donc un jeton. **Ligne** : une ligne de matrice de poids.
- **t(k)** : temps d'une passe de k rangées. **β** : coût d'une rangée de vérification de plus, en pas simples. **τ** : jetons acceptés par vérification. **q_τ** : τ contre l'objet scellé, rapporté au τ contre le Qwen3-4B d'origine. **η** : part des 864 Go/s que l'attention atteint en lisant le cache KV.
- **T8** : servir un changement numérique sous ±0,12 % de perplexité appariée, après un contrôle. **A2** : la capture de graphes CUDA, fermée hors du cœur.
- **Route P** : un noyau persistant par segment de couche. **Route D** : des fusions dans des grilles normales. **4E** : un noyau persistant qui boucle sur N jetons. **La brique** : décoder chaque mot Tetra une fois pour k rangées. **Route GEMM** : au préremplissage, déquantifier en f16 puis passer par cuBLAS.
- **Ta branche** : celle où tu réalises les étapes 2 à 5 de l'audit du 2026-10-29 (fusions, `tv_tetra48_seg`, mur int4 Metal, route GEMM).
- **Étapes** : S0, S1 et S3 sont les pas de la brique sur le Mac (`artstat`, perplexité f32, injection int16). L0, L1 et L2 sont les rejeux hôtes, le premier banc et les bras persistants. B1 est la vérification exacte avec recherche dans le prompt, B2 les noyaux int4 et tête multi-rangées, B3 une lecture du cache pour K rangées. G0 mesure τ sur le Mac, E0 la spéculation exacte sur carte. K32 est la porte de qualité du grand modèle, P0 son pilote. β0 est le premier banc de la brique. C-dense déquantifie en f64, au bit du bras dense ; d2 est un noyau d'attention de préremplissage. R1 et R8 sont tes lectures du verdict spéculatif du 2026-08-05 et du gel ROADMAP §4 pour K rangées.

## 0. Résultats du 2026-11-04

La première vague a tourné le 2026-11-04 sous un plafond de 10 $, sur la branche `validation/vague1` (worktree `.claude/worktrees/validation-vague1`). Chaque préreg a été tamponné avant sa mesure. Journaux dans `docs/mesures/` de cette branche.

- **Dette GSM8K groupée (P1), 2,02 $.** 81,96 % contre 82,49 % : Δ −0,53 point [−1,07 ; +0,00], p = 0,092, règle G1, dette GSM8K soldée au 4B (*mesuré*). Les discordances penchent contre la forme groupée, 10 contre 3, sans être séparées. Porte MMLU : mêmes 57 choix, lignes identiques à l'octet, règle M1.
- **Balayage t(k) (P2), 0,10 $.** Une rangée de plus coûte 3,93 ms à la position 0 (*mesuré*), au-delà de 2,90 : règle R-c, l'estimation de B2 se refait avant tout code. Les logits de la dernière rangée sont identiques au bit à k pas d'un jeton, à tout k de 1 à 8.
- **Ancre FlashInfer (P3), 0,03 $.** η = 0,810 à 8 192 positions (*mesuré*), contre 0,25-0,26 pour notre attention. Lecture 1 : bande [0,60 ; 0,85] tenue, écrire notre noyau d'attention maintenant ; cœurs CUDA seulement, un seul découpage.
- **`tv_tetra48_seg`, 0,14 $.** 72 groupes sur 72 identiques au bit sur carte, une fois le banc corrigé (`c68ba42`). L'échec du 2026-09-20 venait du banc.
- **Lectures locales, 0 $.** S0 : tous les rapports de niveaux de gain du 4B dans [1,26 ; 2,01], la grille int16 de la brique tient. S1 : f16 contre f32 sur Metal, +0,0225 % [−0,0353 ; +0,0803], règle B. Sonde PTX : six `ld.shared.v4.f32` et `fma.rn.f32` aux trois sites, le plafond de 358 Go/s ne s'applique pas.
- **Forme groupée au 8B (P5), 1,31 $.** Perplexité appariée +0,0000 % par le préremplissage et −0,0048 % [−0,0117 ; +0,0021] par le décodage, dans ±0,12 % ; décodage ×1,08 / ×1,34 / ×1,52 / ×2,16 à 5 / 2 048 / 4 096 / 8 192 positions, de 31,0 à 66,9 tok/s à 8k (*mesuré*). Règle R1. Tu as accordé le T8 au 8B sous condition d'un MMLU léger : 285 questions, mêmes 285 choix sous les deux configs, règle M1, 1,25 $. Porte servie groupée 105,8 tok/s contre 98,3, même texte ; 67,1 tok/s à 8k.
- **Registre.** Les cinq jobs absents sont inscrits (3,19 $). La vague a facturé 4,85 $ ; le registre lit 256,88 $ sur 248 lignes.

## 1. Le verdict

Aucune piste ne tombe à la plongée, et aucune n'est validée. Depuis le rapport de la vague 2, la route GEMM s'améliore (2,85-3,4 s au premier jeton à 8k, contre 3,5-4,4) et la route P baisse (221-298 tok/s, contre 250-300). La brique à k = 1 n'est plus sûre de gagner, et la recherche dans le prompt ne rapporte plus que ×1,01 par requête (tous *estimés*).

| piste | ce qu'elle peut donner | l'inconnue qui décide | première expérience | coût | ce qu'il faut de toi |
|---|---|---|---|---|---|
| préremplissage | premier jeton à 8k : 2,85-3,4 s par la route GEMM, 0,45-1,2 s avec un noyau d'attention de préremplissage | identité de cuBLAS sous découpe ; part de l'attention à 8k | V3-lite sur ton prochain job | 0,09 $ (*estimé*), 0,77 $ seul | le mode de mesure dans ta branche ; ensuite le gel ROADMAP §4 et un T8 |
| forme groupée 8B et 14B, gestes | à 8k : 8B de 31 à 68 tok/s, 14B de 21,5 à 44-46 ; +0,5-1,3 % au 4B par trois gestes au bit près | GSM8K groupé du 4B ; perplexité appariée au 8B | dette GSM8K groupée du 4B | 2,0 $ | un T8 par taille |
| attention de décodage | à 8k, 77,5 (*mesuré*) → 106-114 tok/s en f16, 120-125 avec un vrai KV int8 ; ×1,02-1,04 au court | η de notre noyau sur Ada ; t(8)/t(1) | ancre FlashInfer | 0,15-0,42 $ | ensuite le T8 du noyau et ta bande pour q8 |
| noyau persistant | court : 221-298 tok/s (route P), 339-394 (4E) | débit du flux Tetra persistant ; coût d'une attente globale | correctif de `tv_tetra48_seg` et sonde PTX, puis L1 | 0 $, puis 0,45 $ | go Docker ; décision 8 avant L2 |
| la brique | vérifier 8 rangées : 1,10-1,30 pas aujourd'hui | coût de la géométrie à k = 1 et sa pente en k | S0 et S1 sur le Mac, puis β0 | 0 $, puis 0,5-1,1 $ | décisions 7 et 9 avant β0 |
| spéculation exacte | ×1,15 après B2, ×1,42 après B2 et B3, en extraction ; drafter neuronal sur vérification plate : chat ×1,7-2,9 | t(k) du chemin à rangées ; q_τ | balayage t(k) (P2), puis pilote G0 | 0,12 $, puis 0 $ | R1 ; R8 avant B1 |
| Mac | 4B scellé fusé à 45-61 tok/s sur M3 Max, ×1,4-4,3 à même tête contre notre bras dense | débit in situ des noyaux int4 ; justesse des objets e4 sur Metal | le 4B sur Metal, après ton étape 4 | 0 $ | ce que vaut une lecture Mac pour ETAT §5 |
| grand modèle, murs du 32B | murs levés au bit en 3-4 jours ; rotation rapide ×1,02-1,08 au 4B ; K32 dit si un 32B vaut son encodage (70-190 $) | part de la rotation qui raccourcit le pas ; écart au 14B contre le rang 2 bits publié | K32, étapes P0 et A | 0 $, 4-7 h de Mac | rouvrir les bras GGUF rivaux |

Valeurs *estimées* sauf mention.

Ordre de validation, par information par dollar et par semaine (durées *estimées*) :
1. Les lectures à 0 $ sans code : S0, S1, la sonde PTX, K32 P0. Des minutes chacune, contre des hypothèses qui portent des semaines de code : la sonde dit si un flux persistant bute à 358 Go/s (*calculé*).
2. Le balayage t(k) de la spéculation (P2), 0,12 $ sans code. B1 coûte 1 à 2 semaines, et sa valeur seule dépend d'un β jamais mesuré.
3. L'ancre FlashInfer : elle fixe la bande de η avant 5 à 8 semaines de noyau d'attention.
4. La dette GSM8K groupée, la plus chère. ETAT §5 la met en tête de cet axe, et elle ouvre le 8B et le 14B groupés.
5. Le correctif de `tv_tetra48_seg`, que ta branche câble maintenant : son seul contrôle sur carte comparait deux matrices différentes.
6. V3-lite : cinq inconnues pour 0,09 $, contre du code dans ta branche.
7. Le Mac, dès ton étape 4.
8. Puis ce qui demande des semaines de code : L1, le banc du noyau d'attention, β0 de la brique, E0.

Le persistant porte le plus haut plafond court, mais sa première lecture payante attend 2,5 à 3,5 semaines de code : ses pas à 0 $ démarrent en semaine 1. La brique et la spéculation passent en dernier, car leur gain attend des décisions fermées par défaut et une vérification plate.

Le noyau d'attention sert aussi la vérification exacte et 4E. Les gains ne s'additionnent pas (METHODE §3) : les routes P et D partagent les fusions.

## 2. Ce qui peut démarrer tout de suite à 0 $, sans charger la machine

Du code, des tests hôtes légers (`nice`, `cargo -j 4`) et des lectures de quelques minutes, chaque run local sur ton go (règle 1). Durées *estimées*.

- Le correctif de source de `tv_tetra48_seg`, avec un test hôte à deux sources qui échoue sur HEAD : `code/persistent/seg_source_fix.rs`.
- L0, les tests hôtes du persistant : `code/persistent/pers.rs`.
- S0, S1 et la sonde PTX du §3 : `prereg-brick-1a.md`, `code/persistent/ptx_probe.py`. Docker est installé ; le 8B et le 14B scellés ne sont pas sur le Mac (7,9 Go, *calculé*).
- Le rejeu CPU et la sonde hôte du noyau d'attention, 3 à 5 jours : `code/attention/attn_ref_sketch.rs`, `code/attention/host_attn_dec_sketch.cpp`.
- Les variantes exactes du pic de scores et leurs tests hôtes, et le rejeu clang++ de C-dense pour ta branche : `code/prefill/attn_rows.rs`, `code/prefill/deq_tetra48_f64.cu`.
- Les deux tests du câblage Metal, pour ta branche : `code/mac/q4_tiled_matches_served.rs`, `code/mac/q4_metal_candle.rs`.
- Le bras `ARM=grouped` de `ops/jobs/gsm8k-served.sh` (`prereg-gestures-0.md` §2), et le script du 8B et du 14B : `code/gestures/gqa-size.sh`.
- Les scripts de l'ancre FlashInfer, à verser dans `ops/` : `code/attention/fi_decode_bench.py`, `code/attention/fi-decode-anchor.sh`.
- La fin de dump et le projecteur de K32 : `code/bigmodel/emit_dump_trailer.py`, `code/bigmodel/k32_project.py`.
- Le registre : cinq jobs facturés manquent à `docs/data/jobs.csv`, trois du 2026-09-20, un du 2026-09-19 et un du 2026-09-21, pour 3,19 $ (*calculé* sur leurs durées *mesurées*, à 1,80 $/h). La liste vient de `downloads/persistent-jobs/jobs-ps-all.json`, qui ne couvre que du 2026-09-11 au 2026-11-02.
- Les journaux bruts de chaque job, copiés dans le bucket sous 30 jours : un job du 2026-10-06 rend encore 98 lignes, un du 2026-09-26 aucune (*mesuré*). `calc/persistent/fetch_logs.py`.

## 3. Les premières demandes de go

Coûts et durées *estimés*, pires cas *calculés* sur les timeouts. Une porte échouée annule toujours le run.

Runs locaux, 0 $ chacun :
- **S1, perplexité f32 du 4B sur Metal.** 5 à 10 min, `prereg-brick-1a.md`. Elle mesure l'effet du f16 seul sur l'objet servi. Règle : un intervalle apparié dans ±0,02 % laisse l'injection int16 décider selon ses règles, et elle se lit alors comme une confirmation, un intervalle dans ±0,12 % seulement la laisse décider telle quelle, sinon l'injection décide seule et le fait te revient.
- **S0, `artstat` sur le 4B.** Secondes, sans préreg (lecture déterministe). Règle : si chaque rapport de niveaux de gain tombe dans [1/3,14 ; 3,14], la grille int16 de la brique tient, sinon les matrices nommées prennent la forme à deux accumulateurs.
- **Sonde PTX du persistant.** Minutes de CPU, sans préreg. Elle lit la largeur des lectures de l'activation et les sites de contraction FMA. Règle : des lectures scalaires mettent le plafond de 358 Go/s dans le modèle de la route P, six lectures de 128 bits le suppriment, sinon il se recalcule sur les wavefronts comptés.
- **K32, étapes P0 puis A.** 10 à 20 min, puis 3,5 à 6,5 h de nuit, 43 Go téléchargés, `prereg-bigmodel-1.md`. Elle mesure D_ours, l'écart MMLU entre notre 14B scellé et le rang UD-IQ2_M. Règle : si la borne basse de D_ours dépasse 0, l'étape B (un 32B à 2 bits) tourne, sinon la qualité du 14B passe d'abord.
- **Le 4B scellé fusé sur Metal.** 15 à 20 min sur une machine calme, `prereg-mac-1.md`, après ton étape 4 et le binaire `metalrun` (1 jour). Règle : machine restée calme, le routage tuilé devient le chemin Metal sur ton go ; sous 30 tok/s, le chiffre reste au journal ; sinon tu décides.

Jobs payants, sur l40sx1 :
- **P1. Dette GSM8K groupée du 4B, plus 57 questions MMLU sous les deux configs.** 2,0 $, 4,50 $ au pire, 65 min, `prereg-gestures-0.md`. Règle : |Δ| ≤ 1,5 point avec p ≥ 0,05 solde la moitié GSM8K de la dette et ouvre le 8B, une perte au-delà suspend le 8B et le 14B et tu revois l'octroi T8 du 4B, un gain au-delà est rapporté, sinon le run est nul. La moitié MMLU lit les 57 choix sous les deux configs : identiques, la forme groupée ne bouge aucun choix (M1) ; un seul qui diffère rend dû le recensement MMLU, environ 20 $ (*estimé*, ETAT §5). Les logits des lignes de dump bougeront sans doute de quelques centièmes, puisque la forme groupée n'est identique au bit qu'à 5 et 4 096 positions (*mesuré*, `gqa-ab-2026-11-01.txt`) : c'est rapporté, pas lu par la règle. Lancé le 2026-11-04 (job `6aeb1041`), avec un rejeu de 50 problèmes de la forme étendue en tête, qui coupe le job s'il ne réécrit pas le run 0.
- **P2. Balayage t(k) du chemin à rangées, k = 1 à 8.** 0,12 $, 0,60 $ au pire, 4 min facturées, `prereg-speculation-0.md`. Règle : β(2 048) ≤ 0,40 rend B1 seul peut-être rentable, de 0,40 à 0,60 B1 ne se fait qu'avec B2 ou attend, au-delà l'estimation de B2 se refait avant tout code, sinon pas de route.
- **P3. Ancre FlashInfer.** 0,15-0,42 $, 0,75 $ au pire, 5 à 14 min, image vLLM 0.26.0 du dépôt, `prereg-attention-1.md`. Elle mesure E8, le η du noyau GQA de FlashInfer à 8 192 positions. Règle : E8 ≥ 0,70 garde la bande [0,60 ; 0,85], E8 de 0,45 à 0,70 la baisse à [0,45 ; 0,70], sous 0,45 tu dis si le noyau CUDA s'écrit maintenant, sinon rien n'est réglé.
- **P4. V3-lite sur ton prochain job, ou V3 seul.** 0,09 $ embarqué, ou 0,77 $ en 26 min (1,80 $ au pire), `prereg-prefill-1.md`. Il mesure l'identité des variantes de découpe, la part A(8k) de l'attention, les débits FP64 et tenseur, et le noyau à rangées contre celui à une rangée. Règle : une variante exacte partout et moins gourmande devient candidate au service sans T8 après ta lecture du gel ROADMAP §4, A(8k) ≥ 1,5 s avance le noyau d'attention de préremplissage, un FP64 ≥ 0,5 T/s met C-dense d'abord, sinon c'est rapporté sans proposition.
- **P5. Forme groupée au 8B.** 1,38 $, 2,70 $ au pire, 46 min, si P1 ne montre pas de perte, `prereg-gestures-2.md`. Le lanceur lit la taille et le sha256 du 8B scellé sur sa copie locale et refuse sans elle (`gqa-size.sh:89-91`). Cette copie n'est pas sur le Mac : 2,82 Go à rapatrier du bucket avant le go, et 5,09 Go pour le 14B (*mesuré*, listing du bucket). Règle : deux intervalles dans ±0,12 % et toutes les plages de vitesse au-dessus de 1 proposent d'étendre le T8 au 8B, une plage qui touche 1 fait un point de courbe, un intervalle hors bande garde le 8B étendu, sinon le run est nul.

Cumul de la vague : 3,74 à 4,01 $ attendus avec V3-lite embarqué, 4,42 à 4,69 $ avec V3 seul (*estimé*). Au pire, 8,64 $ ou 10,35 $ (*calculé*). Avant la vague : 248,84 $ sur 237 lignes du registre (*mesuré*), 252,03 $ avec les cinq jobs facturés absents du registre (*calculé*). Après : 255,9 à 256,8 $ facturés en tout (*estimé*), P1 compté à 2,1 $.

Le pilote G0 suit (0 $, 1 à 2 h, `prereg-speculation-1.md`), après 4 à 6 jours de mise au point de son harnais (rapport de la vague 2).

## 4. Une section par piste

Durées et coûts *estimés* sauf mention.

### 4.1 Préremplissage et premier jeton (`prefill.md`)
- Conception. À partir d'un seuil R* ≥ 16 rangées, chaque projection est déquantifiée en f16 puis passe par cuBLAS. C-dense refait le bras dense au bit, C-rot16 calcule en base tournée, plus vite. Les deux sont un T8. Trois variantes exactes portent le pic de scores de 6,44 à 2,15-4,29 Go (*calculé*).
- Inconnues. Identité de cuBLAS sous découpe, a priori 0,1-0,25, car 2 remodelages sur 6 seulement ont gardé leurs bits (*mesuré*, `gqa-ab-2026-11-01.txt`). Attention à 8k, 2,21-2,55 s ; FP64, 0,15-0,23 s par passe (*estimés*).
- Échelle. V1 hôte ; P4 ; V4, porte de ton étape 5, 1,40-1,67 $ (`prereg-prefill-2.md`) ; noyau d2, 3 à 6 semaines de code ; chemin à rangées 1a.
- Décisions. Gel ROADMAP §4 et T8 de la route avant tout service ; C-dense ou C-rot16.
- Ta branche (étape 5). Vise 2,85-3,4 s, pas 2 s. `prefill.md` §5 liste 15 pièges, dont la contraction FMA de la chaîne f64.

### 4.2 Forme groupée 8B et 14B, gestes (`gestures.md`)
- Conception. La forme groupée sert tout nombre de têtes par groupe. Trois gestes gardent les bits : argmax sur les logits f16, argmax à clé dans l'ordre de candle, jeton gardé sur la carte. Un flux secondaire porte le v int4, +1,7-3,6 % aujourd'hui (*estimé*), à condition que tu admettes les événements de cudarc coupés avec deux flux. Sinon le repli, événements allumés, coûte 1,7-3,2 % (*estimé*), autant que le gain.
- Inconnues. GSM8K groupé du 4B ; perplexité appariée au 8B et au 14B ; exposition de l'hôte.
- Échelle. P1, P5, puis le 14B, 2,31 $ ; l'A/B des gestes, 0,35 $ (`prereg-gestures-1.md`), après 3,5 à 5,5 jours de code.
- Décisions. Un T8 par taille ; la boucle sur carte comme défaut servi. Le noyau d'attention pourrait remplacer directement la forme étendue au 8B et au 14B : un T8 par taille au lieu de deux, mais dans 5 à 8 semaines.
- Ta branche (étape 2). Si ta fusion écrit l'entrée de v, le flux secondaire tombe à 0-2,4 %. Si le noyau de v prend la norme en prologue, sa fenêtre survit.

### 4.3 Noyau d'attention de décodage et KV int8 (`attention.md`)
- Conception. Un noyau split-KV groupé par couche : découpe absolue de 256 positions, softmax en ligne f32, exp2 polynomiale, arrondis explicites. Décodage, vérification et préremplissage rendent les mêmes bits sur toute carte NVIDIA. Aujourd'hui l'attention lit le cache à 0,25-0,26 de la crête (*calculé*).
- Inconnues. η sur Ada, 0,75 [0,60 ; 0,85] ; t(8)/t(1), 1,0-1,7 à 8k (*estimés*).
- Échelle. Rejeu CPU ; P3 ; banc du noyau seul, 0,18-0,24 $ ; in situ ; q8 ; contrôle T8 ; GSM8K ; int8. En tout 6,4 à 7,3 $ et 5 à 8 semaines de code (*estimé*).
- Décisions. Le T8 du noyau. Ta bande pour q8, dont la seule mesure, +0,049 % [−0,071 ; +0,170] (*mesuré*, `kvq8-4b-2026-08-15.txt`), croise déjà ±0,12 %.
- Ta branche (norme q et k, RoPE, écriture KV). Choisis au chargement, contre candle, parmi deux formes de norme et trois par sortie de RoPE. Garde `rsqrtf`, et refais le contrôle à chaque chargement.

### 4.4 Noyau persistant : route P, 4E, route D (`persistent.md`)
- Conception. Un noyau coopératif de 142 CTA par segment de couche. Chaque CTA recalcule norme et rotation, et il reste 97 attentes globales par jeton. SiLU est tabulée sur ses 65 536 entrées, RMSNorm émulée et vérifiée au chargement.
- Inconnues. B_T, le débit du flux Tetra à 24-32 warps par SM, 490-670 Go/s ; h, le coût d'une attente globale, 1,5-3,5 µs (*estimés*) ; l'identité des lignes calculées dans une autre grille. Le contrôle du 2026-09-20 opposait le flux du fichier ball à la matrice du fichier Tetra : −0,3081 contre −0,5726 à la ligne 0 (*mesuré*).
- Échelle. L0, sonde PTX, correctif ; L1 ; L2, 0,36 $ (`prereg-persistent-1.md`) ; segment MLP in situ (étape A) ; ports ; 4E. Une passe L1 sous 1,80 ms recommande L2, au-delà de 2,60 ms la route D passe d'abord, entre les deux tu choisis. Code : L0 et L1, 2,5 à 3,5 semaines ; 4E, 10 à 16.
- Décisions. La 8 avant L2, la 12 avant l'étape A, la 4 avant 4E.
- Ta branche (route D). Corrige la source avant de câbler `tv_tetra48_seg`. Aucun second flux, pas de `ld.global.nc` sur une donnée écrite dans le même lancement.

### 4.5 La brique (`brick.md`)
- Conception. Chaque mot Tetra est décodé une fois en entiers, contre une activation int16 par tranche de 384. Les produits sont exacts et sommés en int64 : toute géométrie, à tout k, rend les mêmes bits. Le prix est un T8, sous 0,03 % de perplexité (*estimé*).
- Inconnue. La vitesse. Le flux par tuiles de 16 lignes force un découpage en K, et la L40S a mesuré les deux mouvements voisins en perte, de −1,87 à −24,62 % (*mesuré*, `a3-occupation-banc-2026-09-01.txt`).
- Échelle. S0, S1, S3 sur le Mac ; β0 (`prereg-brick-0.md`) après 1 à 1,5 semaine de code ; banc β, 0,8-1,5 $ (`prereg-brick-2.md`), après 6 à 9,5 semaines ; contrôle bout en bout, 4,3-9,3 $.
- Décisions. 7 et 9 avant β0 ; le T8 après S3 et le contrôle bout en bout.
- Ta branche. Garde la sortie tournée des fusions comme dernière phase séparable, et les frontières de `tv_tetra48_seg` à des multiples de 16 lignes.

### 4.6 Spéculation exacte (`speculation.md`)
- Conception. Vérifier K rangées en refaisant pour chacune le calcul du décodage, avec des tours coupés aux frontières de 256 positions et un retour arrière en O(1) sur `Grow`. D'abord la recherche dans le prompt, puis les drafters DeepSpec.
- Inconnues. Une rangée de vérification coûte β ≈ 0,50 à 5 positions, 0,58 à 2 048 et 0,71 à 8 192 (*estimé*) : il faut B2 puis B3. q_τ en chat, 0,80 [0,65 ; 0,95] (*estimé*). La règle 6 compte aussi 168 Mo de KV du drafter à 8k (*calculé*).
- Échelle. P2 ; contrôles de chargement ; pilote puis G0 complet ; E0 après B1, 1,00 $ (`prereg-speculation-2.md`) ; affinage, 18-30 $, si G0 l'y envoie. B1, 1 à 2 semaines de code.
- Décisions. R1 ; R8 avant B1 ; la règle 6 du drafter ; la licence des poids DeepSpec, non déclarée, ou DFlash-b16 de z-lab, sous MIT.
- Ta branche. Trois contrôles imprimés au chargement : rangées Tetra contre une rangée, `rot_apply_rows` contre `rot_apply`, parité int4 sur les 84 enregistrements.

### 4.7 Mac, chemin Metal fusé (`mac.md`)
- Conception. Router vers `tv_q4_metal_tiled` les seuls enregistrements int4 dont l'activation dépasse 32 768 octets, avec une tuile de 2 048 colonnes. Aucun bit ne bouge. Les fichiers pèsent 2,74, 2,70 et 2,73 b/param, tables comprises, contre 16,0 en dense (*calculé*).
- Inconnues. Débit in situ des noyaux int4, 75-250 Go/s (*estimé*) ; llama.cpp IQ2_XXS lit 142 Go/s sur le même Mac (*calculé*). Justesse des objets e4 sur Metal ; nature des 2,4-3,6 ms hors poids par jeton.
- Échelle. Tests ; banc int4 isolé ; le 4B sur Metal ; 8B et 14B après 7,9 Go de téléchargement (`prereg-mac-2.md`) ; bras nul ; formes d'attention ; noyau à rangées. Câblage et `metalrun`, 1,5 à 2 jours.
- Décisions. Ce que vaut une lecture Mac pour ETAT §5 ; le téléchargement ; la tuile Apple.
- Ta branche (étape 4). Une tuile nulle ou le buffer 7 non lié bloquent le GPU. Ne touche pas `tv_q4_h.metal`, copié et haché dans la roue `llvq-tetra`.

### 4.8 Grand modèle et murs du 32B (`bigmodel.md`)
- Conception. R0 stage la rotation en mémoire globale, mêmes fonctions et même ordre ; `tv_q4_hs` stage l'activation int4 en f16, 51 200 octets au lieu de 102 400 (*calculé*). Ils ne servent que là où les noyaux d'aujourd'hui sont refusés : aucune empreinte servie ne bouge. Les murs n'existent que sur les cartes à 99 Kio ; l'A100 et la H200 passent le 32B.
- Inconnues. La part des 0,72 ms de rotation (*calculé*) qui raccourcit le pas. D_ours se résout à ±1,8 point ; l'étape B, à ±3,9 points contre une marge prédite de 1,7, sans doute pas (*estimés*).
- Échelle. K32 P0, A, puis B ; V0c, un bras de rotation nulle, 0,24-0,30 $, avant tout code de rotation rapide ; identité sur carte, 0,30-0,39 $ (`prereg-bigmodel-2.md`) ; bout en bout ; chaîne 32B.
- Décisions. Rouvrir les bras GGUF ; corriger ETAT §4, CLAUDE.md et `format-noyau.md` §8 (25 600 vaut 1 024 × 25) ; choisir 32B, 27B ou aucun après K32.
- Ta branche (étape 2). L'exactitude de la fusion norme et rotation contre candle reste à vérifier au chargement. Le flux secondaire de v et la rotation rapide se mesurent dans le même job, chacun dans son bras, puis ensemble. Leurs gains ne s'additionnent pas (METHODE §3), et chaque rapport ne bouge qu'un mécanisme (METHODE §1).

## 5. Calendrier

Une personne, ta branche à part, aucun pari raté, coûts *estimés*.

| semaine | carte (HF) | Mac | code |
|---|---|---|---|
| 1 | P1, P2 et P3 en parallèle | S0, S1, sonde PTX ; K32 P0, puis étape A la nuit | bras GSM8K, scripts FlashInfer, fin de dump K32, correctif de `tv_tetra48_seg`, registre ; préregs tamponnés |
| 2 | P5, si P1 ne montre pas de perte | étape B de K32 si la borne basse de D_ours dépasse 0 ; le 4B sur Metal dès ton étape 4 | mode de mesure du préremplissage, 2 à 4 jours ; début de L0 |
| 3 | 14B groupé, 2,31 $, si le 8B tient la bande ; P4 et les contrôles de la spéculation sur ton prochain job | libre, ou S3 de la brique | fin de L0 ; code de L1 |
| 4 | L1 du persistant, 0,45 $, si son code est prêt ; porte servie du 8B, 0,75 $, si tu accordes son T8 | idem | fin de L1 ; rejeu CPU de l'attention |

P5 attend P1, le 14B attend P5. Le 4B sur Metal attend ton étape 4 ; P4 embarqué, ton accord sur le mode de mesure. L1 attend L0 et le correctif ; L2, L1 et la décision 8.

Les jobs carte tournent pendant que le code s'écrit. Les runs Mac se suivent : le 4B sur Metal veut une machine calme 30 min après le dernier build, et K32 occupe le GPU la nuit. Dans cet ordre, le harnais G0 et le noyau d'attention commencent en semaine 5. Si tu préfères G0 à L1, il prend les semaines 3 et 4, et L1 recule d'autant.

Sur quatre semaines, la dépense va de 2,4 $ si aucun job conditionnel ne part à 8,2 $ s'ils partent tous (*estimé*).

## 6. Les décisions de l'opérateur

Le minimum pour lancer la première vague :
1. **Le plafond de la vague** (décision 2). Par défaut, aucun job payant. Proposé : 11 $, pour 10,35 $ de pires cas.
2. **Ton go pour les runs locaux** : S0, S1, la sonde PTX dans Docker, K32 et ses 43 Go, le 4B sur Metal. Par défaut, rien ne tourne (règle 1).
3. **R1** : autoriser P2 et G0 sur la piste spéculative close le 2026-08-05, sans lever son verdict. Par défaut, le verdict tient, et ni P2 ni G0 ne tournent.
4. **Les bras GGUF rivaux**, abandonnés le 2026-09-19 après deux échecs sur carte, rouverts sur le Mac pour K32. Par défaut, non, et K32 ne tourne pas.
5. **Le véhicule commun** : ton prochain job carte porte V3-lite, les trois contrôles de la spéculation et l'A/B de frontière si son code est prêt. Chacun a sa section de préreg tamponnée. Par défaut, rien n'embarque : V3 tourne seul à 0,77 $ ou attend.
6. **Le correctif de `tv_tetra48_seg`** : dans ta branche ou dans cette piste. Par défaut, la fusion Tetra reste non prouvée, et rien ne la sert avant le contrôle corrigé.
7. **Le registre** : inscrire les cinq jobs facturés absents, 6aaf9e00, 6aafac81 et 6aafc951 du 2026-09-20, 6aae7666 du 2026-09-19 et 6ab087f2 du 2026-09-21. Par défaut, le cumul reste sous-estimé de 3,19 $ (*calculé*).

L'objectif (décision 1) peut attendre : la vague sert le décodage du 4B et la requête de 8 192 + 512 jetons. La vague suivante demande la décision 9 (gel ROADMAP §4), la 7 (cœurs tenseurs), la 8 (réglages gelés), la 12 (opérations candle), la 4 (A2), R8, et un T8 par changement servi.

## 7. Ce que cette feuille de route n'établit pas

- Aucune mesure nouvelle. Tout gain au-delà de l'existant est estimé, sur des entrées revues par les vérificateurs.
- L'état de ta branche. Les dépendances supposent qu'elle câble `tv_tetra48_seg`, la fusion norme et rotation, le mur Metal et la route GEMM dans l'ordre de l'audit.
- Les durées. Les vérificateurs ont trouvé chaque plan optimiste d'un facteur 1,5 à 2, et les files HF vont de 30 min à 9,5 h.
- Rien sur sm_120, la RTX 5090, le 32B ou le 27B, sauf K32, qui ne mesure que la qualité.

## Fichiers du dossier

Sous wave5-deep/. Brouillons jamais exécutés ; téléchargements non fiables, jamais exécutés.

Racine :
- feuille-de-route-validation.md : ce document
- attention.md
- bigmodel.md
- brick.md
- gestures.md
- mac.md
- persistent.md
- prefill.md
- speculation.md
- bucket-listing.txt : listing du bucket
- prereg-attention-1.md : P3
- prereg-attention-1b.md : balayage llama.cpp
- prereg-attention-2a.md : banc du noyau seul
- prereg-attention-2.md : A/B in situ
- prereg-attention-3.md : q8 à 8k
- prereg-bigmodel-1.md : K32
- prereg-bigmodel-2.md : identité rotation, int4
- prereg-brick-0.md : β0
- prereg-brick-1.md : injection int16
- prereg-brick-1a.md : S1
- prereg-brick-2.md : banc β
- prereg-gestures-0.md : P1
- prereg-gestures-1.md : A/B des gestes
- prereg-gestures-2.md : P5 et 14B
- prereg-mac-1.md : le 4B sur Metal
- prereg-mac-2.md : 8B et 14B sur Metal
- prereg-persistent-0.md : L1 du persistant
- prereg-persistent-1.md : L2 du persistant
- prereg-prefill-1.md : P4
- prereg-prefill-2.md : V4, route GEMM
- prereg-speculation-0.md : P2
- prereg-speculation-1.md : G0
- prereg-speculation-2.md : E0

calc/, lancé une fois :
- ttft_model.py : premier jeton
- paired_dense_vs_fused.py : dense contre fusé
- gestures-check.py : gestes
- attention/attention_v2.py : attention
- bigmodel/k32_numbers.py : K32
- brick/geometry_budget.py : géométrie
- brick/geometry_budget.out : sa sortie
- brick/synthetic_actq.py : bruit int16
- brick/synthetic_actq.out : sa sortie
- persistent/route_p_model.py : route P
- persistent/fetch_logs.py : journaux HF

code/attention/ :
- attn_dec_sketch.cu : noyau split-KV
- attn_ref_sketch.rs : rejeu CPU
- host_attn_dec_sketch.cpp : sonde hôte
- fi_decode_bench.py : banc FlashInfer
- fi-decode-anchor.sh : job FlashInfer
- attn-depth-anchor.sh : job llama.cpp

code/bigmodel/ :
- rotate_wide.cu : R0, R1, R1w
- host_rotate_wide.cpp : test hôte
- rot_null.cu
- rot_ab_modes.rs : modes V0c, V3
- rotbench_wide.rs : banc de rotation
- tv_q4_wide.cu : int4 large
- selection.rs : choix du noyau
- emit_dump_trailer.py : fin de dump
- k32.sh : pilote K32
- k32_pilot.py : étape P0
- k32_project.py : projection K32
- override-r0proxy/r0proxy.sh : job V0d
- override-r0proxy/rotate.cu : proxy de R0

code/brick/ :
- t48i_reference.hpp : référence entière
- t48i_mma_sketch.cu : pas IMMA
- warp_emu_sketch.hpp : émulateur de warp
- actq_inject_sketch.rs : injection int16

code/gestures/ :
- argmax_key_ref.rs : clé d'argmax
- tok_argmax.cu
- decode_on_device.rs : jeton sur la carte
- vstream.rs : flux secondaire
- gesture_ab.rs : mode A/B
- gqa-size.sh : job 8B et 14B

code/mac/ :
- fused_metal_wall.rs : câblage du mur
- q4_tiled_matches_served.rs : test tuilé
- q4_metal_candle.rs : test de liaison
- q4bench.rs : banc int4
- metalrun.rs : binaire de mesure
- mac_bounds.py : arithmétique

code/persistent/ :
- seg_source_fix.rs
- ptx_probe.py
- pers.rs : ordonnancement
- exact_ops.cuh : opérations exactes
- argmax_key_ref.py : clé d'argmax
- tetra48_persall.cu : passe en un lancement
- tetra48_pers.cu : noyau persistant
- culink_dump.rs : vidage des cubins
- fusedrun_boundary_ab.rs : A/B de frontière
- pers-combined.sh : job L1
- pers-arms.sh : job L2

code/prefill/ :
- attn_rows.rs : variantes du pic
- gemm_route.rs : route GEMM
- deq_tetra48_f64.cu : C-dense
- fp64_probe.cu
- tv_q4_rows_hh.cu : int4 multi-rangées
- attn_fold.cuh : pli du noyau d2
- prefill-ab.sh : job V3

code/speculation/ :
- g0_tau.py : harnais G0

downloads/bigmodel-hub-2026-11-02/, fiches GGUF du Hub :
- unsloth_Qwen3-4B-GGUF.model.json
- unsloth_Qwen3-4B-GGUF.tree.json
- unsloth_Qwen3-14B-GGUF.model.json
- unsloth_Qwen3-14B-GGUF.tree.json
- unsloth_Qwen3-32B-GGUF.model.json
- unsloth_Qwen3-32B-GGUF.tree.json

downloads/bigmodel-verify-hub/, relecture :
- unsloth_Qwen3-14B-GGUF.model.json
- unsloth_Qwen3-14B-GGUF.tree.json
- unsloth_Qwen3-32B-GGUF.model.json
- unsloth_Qwen3-32B-GGUF.tree.json

downloads/deepspec-005e03b/, DeepSpec :
- deepspec_data_parser.py
- deepspec_eval_dspark___init__.py
- deepspec_modeling_dspark___init__.py
- deepspec_modeling_dspark_qwen3___init__.py
- deepspec_modeling_dspark_qwen3_config.py
- deepspec_utils___init__.py
- scripts_data_README.md
- scripts_data_generate_train_data.py
- scripts_data_prepare_target_cache.py
- scripts_eval_eval.sh

downloads/flashinfer-decode/, FlashInfer v0.6.14 :
- decode.cuh
- cascade.cuh
- state.cuh
- utils.cuh
- main.json

downloads/llamacpp-bench/ :
- llama-bench.cpp

downloads/llamacpp-fattn/, attention de llama.cpp :
- fattn.cu
- fattn-common.cuh
- fattn-vec.cuh
- fattn-tile.cuh
- fattn-mma-f16.cuh
- fattn-wmma-f16.cuh
- common.cuh
- master.json

downloads/persistent-jobs/, jobs HF :
- jobs-ps-all.json : liste des jobs
- jobs-ps-all.err : vide
- inspect-6aafac81.json : job du 2026-09-20
- logs-6aafac81.txt : vide
- logs-6aafac81.err : vide
- logs-6aafc951.txt : vide
- logs-control-6ae83d78.txt : journal témoin
- logs-probe-6ab8145d6b030d633f695d1a.txt : rétention, vide
- logs-probe-6ac4a16efbc85ba6823b1f07.txt : rétention, 98 lignes

downloads/st-headers/, en-têtes des drafters :
- deepseek-ai_dspark_qwen3_4b_block7.hdr.json
- deepseek-ai_dspark_qwen3_4b_block7.len
- deepseek-ai_eagle3_qwen3_4b_ttt7.hdr.json
- deepseek-ai_eagle3_qwen3_4b_ttt7.len
- z-lab_Qwen3-4B-DFlash-b16.hdr.json
- z-lab_Qwen3-4B-DFlash-b16.len
- zlab_config.json

Vérifications :
- verify-attention/recompute.py : attention
- verify-bigmodel/check_plan.py : K32
- verify-brick/check_brick_numbers.py : brique
