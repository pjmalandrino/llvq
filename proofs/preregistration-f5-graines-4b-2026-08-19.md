# Pré-enregistrement F5 — variance de calibration au 4B, trois graines (2026-08-19)

Écrit et commité **avant** tout lancement. Item F5 du plan TACO ; go de
dépense opérateur du 2026-08-19, flavor `l4x4` choisi par l'opérateur.

## 0. Ce que la vérification du plan de run a changé, avant de dépenser

L'opérateur a demandé de revérifier le plan « qu'on le fasse pas pour rien ».
Quatre faits établis sur pièces, dont trois modifient le protocole :

1. **La graine n'est pas un argument de `smoke`.** L'argument 7 est `blocks`
   (`arg(&a, 7, "blocks", usize::MAX)`, smoke.rs:583) : les `999` du run 4B
   publié et les `1000000` du run 8B sont des sentinelles « tous les blocs ».
   La graine vit dans `LLVQ_CALIB_SEED`, absente des deux runs.
   🕳️ Le journal B3 du 08-18 affirmait « GRAINE 1000000 » : corrigé le
   2026-08-19 (le préreg B3, tamponné, garde l'erreur — une ancre ne se
   détache pas pour corriger le texte qu'elle atteste).
2. **Non posée, la graine signifie « préfixe contigu depuis le token 0 »**
   (`window_starts`, smoke.rs:416-427, dont la doc désigne explicitement le
   balayage à trois graines comme son usage prévu). **Tous les artefacts
   publiés sont donc des runs à préfixe, pas des tirages aléatoires** — les
   trois graines de F5 échantillonnent un régime que le publié n'habite pas.
3. **`smoke` n'imprime que la perplexité AGRÉGÉE** (accumulation sur les
   fenêtres puis un seul `exp`, smoke.rs:854-863). S'arrêter là produirait
   exactement le « journal de synthèse » que le §7 de CLAUDE.md interdit, et
   interdirait l'intervalle apparié fenêtre par fenêtre qui est l'instrument
   standard du dossier depuis le 08-17. **D'où l'étape `seal` + `ppl` dans
   chaque job**, avec `2>` capturé (les NLL par fenêtre sortent sur stderr à
   9 décimales).
4. **`LLVQ_DATASET_REV` ne peut pas épingler un sha ici** : une seule
   variable couvre **trois dépôts** (`allenai/c4`, wikitext, `cais/mmlu`,
   corpus.rs:13-30), et un sha n'est valide que dans l'un d'eux. L'épinglage
   n'est donc pas utilisé — voir la réserve R2.

## 1. La question, et la quantité produite

Toutes les barres de qualité du dossier conditionnent sur **un artefact par
taille** : le σ inter-graines n'existe qu'à 0,6B sur 3 blocs (≈ 0,7 %), un
objet différent. F5 mesure l'**étendue inter-graines à la taille publiée**,
sur le modèle publié, avec le protocole publié.

Ce n'est pas un gate : aucune bande de kill. La quantité produite est une
dispersion, et **si les trois graines tombent loin les unes des autres,
c'est le résultat** (la doc de `window_starts` le dit déjà).

## 2. Le protocole

Trois runs **indépendants et parallèles**, identiques à un seul caractère
près — `LLVQ_CALIB_SEED` ∈ {1, 2, 3} — et identiques au run publié pour tout
le reste (mêmes 9 arguments, même corpus de calibration C4, même codebook
`leech1c12`, `rot`, F32 par défaut) :

```
hf jobs run --flavor l4x4 --timeout 3h -d \
  -v hf://buckets/Pier-Jean/jobs-artifacts:/out \
  hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  -- bash -lc 'set -euo pipefail
S=<1|2|3>; D=/out/f5-graines-2026-08-19/seed$S; mkdir -p "$D"
export LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_THREADS=48
export LLVQ_CALIB_SEED=$S LLVQ_ARTIFACT=$D/q4b-s$S.llvq
smoke 64 2048 12 4096 cuda nogs leech1c12 999 rot 2>&1 | tee $D/smoke.txt
seal $D/q4b-s$S.llvq $D/q4b-s$S-sealed.llvq 2>&1 | tee $D/seal.txt
LLVQ_DTYPE=f16 ppl 4096 12 cuda $D/q4b-s$S-sealed.llvq \
  > $D/ppl-stdout.txt 2> $D/ppl-nll-par-fenetre.txt
cat $D/ppl-stdout.txt; tail -20 $D/ppl-nll-par-fenetre.txt
sha256sum $D/q4b-s$S-sealed.llvq'
```

`LLVQ_DTYPE=f16` pour `ppl` : c'est le dtype de la paire publiée
(empreinte `3f1baca9033bf251`), donc les trois runs sont comparables entre
eux **et** au point publié sur la même empreinte de tokens.

**Le premier run est un PILOTE.** Les deux autres ne partent qu'après
vérification que le pilote (a) tient dans les 24 Go de VRAM en F32, (b)
produit les trois fichiers attendus, (c) imprime l'empreinte de tokens.
C'est la réponse directe au « qu'on le fasse pas pour rien » : le risque
VRAM est estimé à ~18 Go (16,1 Go de modèle en F32 + hessiennes), jamais
mesuré à cette taille sur 24 Go, et un pilote le tranche pour 6,50 $ au lieu
de 19,50 $.

**Coût annoncé : ≤ 6,50 $ le pilote, ≤ 19,50 $ les trois** (l4x4 à 3,80 $/h,
devis estimateur 1,3 h de quantification + ~0,4 h de seal et ppl ; cap dur =
`timeout 3h`, soit 11,40 $ le pire cas par job). Cumul rapporté après.

## 3. Ce qui se publie, posé d'avance

- Les trois perplexités sur le fichier scellé, à empreinte de tokens
  identique, avec leur **étendue** (min–max) et leur médiane.
- Les **NLL par fenêtre** commitées (§7 : le brut, pas la synthèse), donc
  les intervalles appariés entre graines calculables sans rejouer un token.
- La position du point **publié** relativement aux trois, étiquetée
  **confondue** (régime de préfixe + révision de corpus antérieure), donc
  indicative et jamais présentée comme un quatrième tirage.
- Conséquence B4 : partout où le dossier écrit « toutes les barres
  conditionnent sur un tirage unique », la phrase gagne le chiffre de cette
  étendue et la réserve R1 ci-dessous.

## 4. Réserves déclarées d'avance

- **R1 — ce σ n'est pas l'erreur du point publié.** Il mesure l'étendue
  entre trois **tirages aléatoires** de fenêtres ; le publié est un
  **préfixe contigu**. La quantité est « de combien le choix des fenêtres de
  calibration déplace le résultat à 4B », et elle s'attache au point publié
  comme approximation déclarée, pas comme sa barre d'erreur.
- **R2 — corpus non épinglé** (cf. §0.4). Les trois runs lisent `main` dans
  la même heure : leur cohérence mutuelle tient par construction. La dérive
  vis-à-vis du run publié (shard C4 `00000` → `00001`) est déjà déclarée au
  README et n'est pas rouverte ici.
- **R3 — n = 3.** Une étendue sur trois points est un ordre de grandeur, pas
  un écart-type de précision — exactement le statut du σ à 0,6B, et il sera
  cité avec la même prudence.
- **R4 — la chaîne diffère du run publié ailleurs que sur la graine** :
  toolchain épinglée depuis le 08-10, CUDA au lieu de Metal, corpus dérivé.
  Ces différences sont communes aux trois runs, donc **sans effet sur
  l'étendue** ; elles n'affectent que la comparaison au point publié, déjà
  étiquetée confondue en R1/R2.

## 5. Anomalies

- **A1** — un run qui n'atteint pas `seal` : le journal le rapporte, le run
  est relancé une fois à l'identique, et l'échec reste au registre.
- **A2** — empreinte de tokens différente entre les trois `ppl` : invalide
  la comparaison, investigation avant toute publication.
- **A3** — `verify_artifact` (interne à `smoke`) doit passer sur les trois :
  un artefact qui ne se relit pas bit pour bit n'entre pas dans l'étendue.

## 6bis. AMENDEMENT DU 2026-08-19 — le pilote a tué la `l4x4`

Écrit **après le pilote, avant les runs de mesure**, et le document est
**ré-ancré** (règle du dépôt : « un fichier ancré qui doit être édité est
ré-ancré »). Le `.ots` d'origine reste dans l'historique git ; le nouveau
atteste ces octets-ci.

**Ce que le pilote a mesuré, et c'est un fait, pas une estimation** : job
`6a857b07e55292eada79d93b`, `l4x4`, **OOM à 122 s**, pendant la passe de
perplexité *baseline*, avant même que la quantification commence.
`CUDA_ERROR_OUT_OF_MEMORY`. La carte annonce **23 034 MiB** (une L4 en fait
22,5 Go, pas 24), et le budget f32 s'écrit :

| poste | Go |
|---|---|
| modèle Qwen3-4B en f32 | 16,1 |
| logits d'une fenêtre 4096 × 151 936 | 2,5 |
| cache KV, 36 couches | 1,2 |
| **sous-total, hors activations** | **19,8** contre 22,5 |

**Amendements**, tous mécaniques :

- flavor `l4x4` → **`rtx-pro-6000`** (96 Go de VRAM, 23 vCPU) — la carte sur
  laquelle le 8B a été quantifié : précédent du projet, pas un choix neuf.
- `LLVQ_THREADS` 48 → **23** (les vCPU de la carte).
- `--timeout 3h` → **`4h`** (devis 2,4 h de quantification + ~0,4 h de seal
  et ppl ; le 8B, deux fois plus gros, a pris 4,6 h).
- Coût annoncé : **≤ 7,50 $ le pilote, ≤ 22,50 $ les trois** ; pire cas au
  cap = 11 $ par job.
- **Ce qui NE change pas, et c'est le point** : F32 reste le dtype, donc le
  protocole publié est intact et aucune déviation ne s'ajoute aux réserves
  R1-R4. L'alternative examinée — rester sur `l4x4` en `bf16` — était moins
  chère de ~3 $ et introduisait une quantification depuis des poids bf16 :
  écartée pour cette raison, sur arbitrage de l'opérateur.
- **Un second pilote est relancé** sur la nouvelle carte avant les graines 2
  et 3 : le mur VRAM est levé par l'arithmétique (96 Go ≫ 20), mais la
  chaîne `smoke → seal → ppl` en un seul job n'a jamais tourné, et 7,50 $
  restent moins chers que 22,50.

Le pilote a coûté **0,13 $** pour économiser un lot à 19,50 $ lancé contre
un mur. C'est la justification de la clause pilote, et elle est datée.

## 6ter. Le job amendé, verbatim

```
hf jobs run --flavor rtx-pro-6000 --timeout 4h -d \
  -v hf://buckets/Pier-Jean/jobs-artifacts:/out \
  hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  -- bash -lc 'set -euo pipefail
S=<1|2|3>; D=/out/f5-graines-2026-08-19/seed$S; mkdir -p "$D"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv
export LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_THREADS=23
export LLVQ_CALIB_SEED=$S LLVQ_ARTIFACT=$D/q4b-s$S.llvq
smoke 64 2048 12 4096 cuda nogs leech1c12 999 rot 2>&1 | tee $D/smoke.txt
seal $D/q4b-s$S.llvq $D/q4b-s$S-sealed.llvq 2>&1 | tee $D/seal.txt
LLVQ_DTYPE=f16 ppl 4096 12 cuda $D/q4b-s$S-sealed.llvq \
  > $D/ppl-stdout.txt 2> $D/ppl-nll-par-fenetre.txt
cat $D/ppl-stdout.txt; tail -20 $D/ppl-nll-par-fenetre.txt
sha256sum $D/q4b-s$S-sealed.llvq'
```

## 6quater. Une anomalie de plus, ajoutée par l'amendement

- **A4** — OOM sur la nouvelle carte : ne pas contourner par une baisse de
  dtype ou de contexte sans **nouvel amendement ancré**. Un protocole qui
  se relâche sous la contrainte matérielle mesure autre chose que ce qu'il
  annonce, et ce document existe pour rendre ce glissement visible.

## 7. Sorties

Journal : `docs/mesures/f5-graines-4b-2026-08-19.txt`. Données brutes : les
trois `ppl-nll-par-fenetre.txt` commités sous `docs/data/f5-nll/`. Registre :
une ligne par job.
