# Pré-enregistrement — M2 : attribution de la chute MMLU par type de projection, au 4B

**Écrit et commité le 2026-09-02. BROUILLON : à tamponner (`ots stamp`) après
relecture de l'opérateur et AVANT le lancement du job.** Go de principe de
l'opérateur du 2026-09-02 (D0 de `docs/ROADMAP-RECHERCHE.md`), plafond de la
vague : **5 $**.

**Coût : ≈ 2,3 $** (*estimé* : 11 bras × ~6,5 min à 1,80 $/h sur `l40sx1`,
depuis les 0,19 $/bras mesurés au job `6a8df15645686a1580c087ea` du 2026-08-25 ;
plafond réel = le `timeout` du job, **2 h → 3,60 $ au pire**). La réplique du
§5 coûterait le même prix ; les deux tiennent sous le plafond de vague.

---

## §1 — La question, et pourquoi elle n'a jamais été posée

Le 4B scellé perd **−14,73 pp de MMLU** sur le f16 (55,59 contre 70,32,
empreinte `65dcd53655e8bfa5`, apparié). Les 252 matrices sont quantifiées sous
le même codebook, le même `cap`, le même budget de bits, et `docs/mesures/` ne
contient **aucun budget d'erreur par matrice** : on ne sait pas laquelle des
sept projections (`q`, `k`, `v`, `o`, `gate`, `up`, `down`) porte la chute.

L'expérience qui répond est la moins chère du dossier : **restaurer un type de
matrice en f16** depuis le checkpoint `Qwen/Qwen3-4B` — les 36 matrices de ce
type d'un coup — et scorer le reste tel que livré. Sept bras de ce genre, plus
le fichier livré, appariés question par question, sont l'attribution.

**Pourquoi c'est un A/B à fichier constant, et pourquoi ça compte.** Aucun bras
ne recalibre : les matrices qui restent quantifiées sont **les mêmes octets**
dans les onze bras. Le σ de calibration (5,2 % de ppl, 2,92 pp de MMLU entre
graines) ne s'applique donc pas ; la barre est la **SE appariée à fichier
constant, 0,43 pp**, mesurée le 2026-08-15. C'est ce qui rend M2 mesurable sans
attendre M1, et c'est pourquoi il passe en premier.

**Pourquoi la restauration est exacte.** `llvq_artifact::decode_matrix`
dé-rotationne à la sortie, donc les tenseurs que le fichier scellé rend sont en
**base naturelle**, celle du checkpoint ; le chemin dense de `bin/mmlu`
(`Proj::Dense`) n'applique aucune rotation. Le tenseur `(d_out, d_in)` du
checkpoint, ramené au dtype du run exactement comme le `VarBuilder` du bras f16
le fait, tombe dans le même emplacement sans transformation. Avec les sept
types restaurés, le modèle **est** le checkpoint : c'est le contrôle haut.
Code : `llvq_llm::sealed::RestoreF16` (test
`the_mmap_source_narrows_like_the_var_builder_does`).

---

## §2 — Le job, verbatim

Image `hf.co/spaces/Pier-Jean/llvq-runner-cuda` **reconstruite depuis le commit
qui porte `RestoreF16`** (`uv run ops/run.py publish Pier-Jean/llvq-runner-cuda
--cuda`, attendre `APP_STARTING` ; le fichier `COMMIT` de l'image nomme la
révision). Flavor `l40sx1`. Volumes : modèle `Pier-Jean/Qwen3-4B-LLVQ-2bit` →
`/model` (lecture seule), bucket `Pier-Jean/jobs-artifacts` → `/out`.
`HF_TOKEN` en secret pour le checkpoint (public, mais le job n'a pas de cache).

```
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda --flavor l40sx1 \
  --mount-model Pier-Jean/Qwen3-4B-LLVQ-2bit \
  --bucket Pier-Jean/jobs-artifacts --name m2-attribution-4b --timeout 2h \
  -- "$(cat <<'SH'
OUT=/out/m2-attribution-4b-2026-09-02
mkdir -p $OUT
nvidia-smi --query-gpu=name,driver_version --format=csv | tee $OUT/gpu.txt
F=/model/qwen3-4b-llvq.bin
ls -l $F | tee $OUT/artefact.txt
export LLVQ_MODEL=Qwen/Qwen3-4B
# bras 0 — contrôle bas : le fichier tel que livré
LLVQ_MMLU_DUMP=$OUT/mmlu-shipped.csv mmlu $F cuda 40 2>&1 | tee $OUT/mmlu-shipped.txt
# bras 1..7 — un type restauré en f16, dans l'ordre des consommateurs d'activation
for t in q_proj k_proj v_proj o_proj gate_proj up_proj down_proj; do
  LLVQ_RESTORE_F16=$t LLVQ_MMLU_DUMP=$OUT/mmlu-restore-$t.csv \
    mmlu $F cuda 40 2>&1 | tee $OUT/mmlu-restore-$t.txt
done
# bras 8, 9 — les deux groupes fonctionnels (interaction intra-groupe)
LLVQ_RESTORE_F16=q_proj,k_proj,v_proj,o_proj LLVQ_MMLU_DUMP=$OUT/mmlu-restore-attn.csv \
  mmlu $F cuda 40 2>&1 | tee $OUT/mmlu-restore-attn.txt
LLVQ_RESTORE_F16=gate_proj,up_proj,down_proj LLVQ_MMLU_DUMP=$OUT/mmlu-restore-mlp.csv \
  mmlu $F cuda 40 2>&1 | tee $OUT/mmlu-restore-mlp.txt
# bras 10 — contrôle haut : tout restauré = le checkpoint
LLVQ_RESTORE_F16=all LLVQ_MMLU_DUMP=$OUT/mmlu-restore-all.csv \
  mmlu $F cuda 40 2>&1 | tee $OUT/mmlu-restore-all.txt
SH
)"
```

`limit 40` par matière — le même que toutes les campagnes MMLU du projet, donc
2 280 questions. Onze bras dans un seul job, chacun son processus : la
comparaison se fait sur les dumps, question par question, aucune horloge n'est
en jeu, donc le « même processus » des bancs de vitesse n'a pas d'objet ici.

✅ **Plomberie faite, 0 $, le 2026-09-02, sur le fichier publié**
([`docs/mesures/m2-plomberie-mac-2026-09-02.txt`](../docs/mesures/m2-plomberie-mac-2026-09-02.txt)) :
la ligne `restauré en f16 (M2) = f16 restored: k_proj (36 matrices, 94371840
weights)` est rendue ; **« all restauré » rejoue le checkpoint à 114/114 picks
identiques, Δ = +0,00 pp** (le contrôle haut du §3.4, vérifié avant la carte) ;
les trois refus refusent. Un job qui découvrirait une faute de nom de tenseur
sur la carte aurait été un job facturé pour rien — ce n'est plus possible.

---

## §3 — Contrôles, et si l'un tombe aucun chiffre n'est publié

1. **Empreinte de tokens `65dcd53655e8bfa5`** sur les onze bras.
2. **Un fichier de 1 770 527 533 octets, sha256 `9db213ef…`** — le fichier
   publié (`docs/fiche-4b.md`, Hub `x-linked-etag`). ⚠️ Pas 1 770 528 125 :
   c'est la taille des artefacts graines de F5, re-scellés 592 octets plus
   gros (autre version de format), et le préreg du 08-25 les a donnés à tort
   pour « la taille du fichier publié ». Deux objets, deux tailles.
3. **Le contrôle bas rejoue 55,59** au centième (comme `mmlupair-4b` du
   2026-08-13), et son dump comparé à `docs/data/mmlu-dumps/mmlu-4b-llvq.csv`
   rend **≥ 99,5 % de picks identiques**.
4. **Le contrôle haut rejoue 70,32** au centième, dump comparé à
   `docs/data/mmlu-dumps/mmlu-4b-f16.csv`, **≥ 99,5 % de picks identiques**.
   C'est le contrôle qui prouve que « tout restauré » est le checkpoint et pas
   une approximation de lui.
5. **Chaque bras restauré imprime son compte de poids**, et il vaut ce que les
   formes du 4B imposent (*calculé* : 36 couches, hidden 2560, q/o 4096,
   k/v 1024, intermediate 9728) :

   | bras | matrices | poids restaurés |
   |---|---|---|
   | `q_proj` · `o_proj` | 36 | 377 487 360 chacun |
   | `k_proj` · `v_proj` | 36 | 94 371 840 chacun |
   | `gate_proj` · `up_proj` · `down_proj` | 36 | 896 532 480 chacun |
   | attention (`q,k,v,o`) | 144 | 943 718 400 |
   | MLP (`gate,up,down`) | 108 | 2 689 597 440 |
   | `all` | 252 | 3 633 315 840 |

   Un compte différent veut dire qu'un autre objet a été scoré.
6. **Les onze dumps sont écrits** (`# end fingerprint=… questions=2280`). Sans
   eux, aucune paire n'est formable et le job est à repayer.

---

## §4 — Ce qui se publie, et ce qui NE se compare PAS

Ce qui se publie : pour chacun des neuf bras restaurés, **Δ = restauré −
livré** en micro stratifié, avec **IC95 apparié** (bootstrap stratifié par
matière, 10 000 tirages, graine `0xb0075eed`) et **McNemar exact**, par
`mmlupair mmlu-shipped.csv mmlu-restore-*.csv` ; le contrôle haut en plafond
(attendu : +14,73) ; et, en regard de chaque bras, **ce que coûterait de servir
ce type** en f16 et à 4 bits, en b/poids noyau (*calculé*, table de l'audit
§4.6 : `k` seul +0,05 à 4 bits, `q+k` +0,26, attention entière +0,52,
`down` +0,49).

Ce qui ne se compare pas :

- **La somme des sept Δ n'a aucune raison de faire 14,73.** Chaque Δ est
  l'effet **marginal** de dé-quantifier un type *dans l'artefact livré*, dont
  les blocs aval ont été calibrés sur des activations déjà quantifiées en
  amont (calibration séquentielle). Les interactions sont réelles, et les deux
  bras de groupe existent pour les mesurer, pas pour les nier.
- **Un seul tirage de calibration.** Le classement des types est celui de
  l'artefact publié ; il peut dépendre de la graine. C'est l'objet de la
  réplique du §5, pas une réserve qu'on lève en prose.
- **Rien sur la vitesse, rien sur une autre taille, rien sur la perplexité.**

---

## §5 — La règle de lecture, posée AVANT de voir le chiffre

C'est une **mesure**, pas un gate : elle ne peut pas être « rouge ». Ce qu'elle
décide est **l'ouverture de Q5** (précision mixte par fonction) contre **Q6**
(composition) — règle reprise de `docs/ROADMAP-RECHERCHE.md` M2, précisée ici :

| lecture | condition | conséquence |
|---|---|---|
| **cible désignée** | un type seul rend Δ ≥ **+3,0 pp** *et* son IC95 apparié est entièrement au-dessus de +1,5 pp | Q5 s'ouvre sur ce type ; **réplique** obligatoire (ci-dessous) avant tout devis |
| **profil plat** | aucun type ne rend Δ > +1,5 pp | Q5 ne s'ouvre pas ; le front est la composition (Q6) ou l'allocation par ligne sous F1 (F3), pas l'allocation par type |
| **signal diffus** | entre les deux | les bras de groupe tranchent : attention ≥ +3,0 pp → Q5 sur l'attention avec sa table de coût ; sinon Q6 |

**Et le coût décide autant que le signal.** Q5 n'est « le meilleur rapport du
dossier » que si la cible est `k` (+0,05 b/poids à 4 bits) ou `q+k` (+0,26).
Si la cible est `down` ou le groupe MLP, la porter à 4 bits coûte ≥ +0,49
b/poids et ramène le 4B au niveau de l'AWQ (5,30 b/param) : Q5 est alors
**déclaré non rentable sur l'axe mémoire**, et la réponse est F3 (débit par
ligne sous F1) ou Q6 — pas un run 4B à 7 $.

**Réplique (≈ 2,3 $, sous le plafond de vague).** Si une cible est désignée, le
même job est rejoué sur l'artefact **graine 3** de F5
(`hf://buckets/Pier-Jean/jobs-artifacts/f5-graines-2026-08-19/seed3/q4b-s3-sealed.llvq`,
1 770 528 125 octets — sa taille à lui, cf. §3.2 —, meilleure ppl des trois) avec
`LLVQ_MODEL=Qwen/Qwen3-4B`.
La cible est **retenue** si elle est le type de plus grand Δ sur les deux
tirages, ou si son IC95 recouvre celui du premier ; sinon l'attribution est
publiée comme **dépendante du tirage**, et Q5 ne s'ouvre pas sur un seul
artefact.

---

## §6 — Divulgation datée, à la signature

- Connu avant le job : 55,59 (livré) / 70,32 (f16) / 70,04 (AWQ 4 bits), tous
  appariés sur l'empreinte `65dcd53655e8bfa5` ; le profil par matière — algèbre
  abstraite et comptabilité au hasard, histoire et droit tenus.
- Aucun bras restauré n'a jamais tourné, à aucune taille.
- **Ce que la littérature prédit** (audit §4.6, à lire comme un prior, pas une
  mesure) : `k_proj` et l'attention en général sont les plus sensibles à
  2 bits (APTQ, KVTuner, HyQuant).
- **Prédiction de l'auteur, écrite pour être opposable, et OPPOSÉE au prior** :
  sur Qwen3, `q` et `k` sont **renormalisés par tête après projection**
  (`q_norm`, `k_norm`) : leur erreur de magnitude est effacée avant RoPE, et
  seule la direction compte. Donc **`q_proj` et `k_proj` < +1,5 pp chacun** ;
  le plus grand Δ isolé sur **`down_proj`** (la matrice la plus large, 13,5
  échantillons par dimension, qui écrit directement dans le flux résiduel),
  suivi de `v_proj`/`o_proj` (non renormalisés) ; et **le groupe MLP > le
  groupe attention**. Conséquence si c'est juste : Q5 n'est pas rentable, la
  suite est F3/Q6.
- ⚠️ **Ce que vaut cette prédiction** : les deux prédictions signées du
  2026-08-25 étaient fausses toutes les deux. Elle est là pour être opposable,
  pas pour être crue.
