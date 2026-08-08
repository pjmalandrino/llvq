# Plan MMLU — ce qu'il faut faire, et pourquoi dans cet ordre

> 🗓️ **BANDEAU D'ÉTAT — dernière revue le 2026-08-08. Ce plan est exécuté, et
> trois fois plutôt qu'une.** Résultats, tous en agrégation **micro** (celle du
> papier), 2 280 questions sur 14 042, empreinte de tokens imprimée :
>
> | | f16 | AWQ 4 bits | LLVQ 2 bits |
> |---|---|---|---|
> | Qwen3-4B (L40S, 08-06) | 70,32 ± 1,28 | **70,04 ± 1,25** | **55,59 ± 1,35** |
> | Qwen3-8B (L40S, 08-08) | 76,08 ± 1,21 | 73,01 ± 1,26 | **65,52 ± 1,31** |
>
> Le premier run (Mac, 08-02) rendait 70,42 ± 1,28 et 56,09 ± 1,36 : la
> baseline se reproduit à 0,10 pp entre deux machines et deux sessions, ce qui
> **certifie le harnais**. Le piège que ce plan existait pour éviter — publier
> du **macro** en croyant faire du **micro** — a bien été commis une fois, puis
> corrigé le 08-02 ; il ne s'est pas reproduit.
>
> **Le verdict** : le déficit est de −14,73 pp à 4B et de **−10,56 pp à 8B**,
> et l'écart au 4 bits est **divisé par deux** (14,45 → 7,49 pp) — le déficit
> fond avec l'échelle, sur les deux métriques.
> Sources : [`campagne-finale-2026-08-07.md`](campagne-finale-2026-08-07.md),
> [`echelle-4b-8b-2026-08-08.md`](echelle-4b-8b-2026-08-08.md),
> [`mmlu-micro-2026-08-02.log`](mmlu-micro-2026-08-02.log).

> La perplexité mesure la surprise moyenne sur du texte brut. Elle ne dit
> **rien** de ce qu'un modèle sait encore faire. Notre ×1,386 de perplexité est
> un chiffre honnête et une information incomplète : on ne sait pas si le
> modèle a perdu 5 % ou 30 % de ses capacités.
>
> Référence à atteindre (papier, Table 6, Qwen3-4B, 2 bits, sans fine-tuning) :
> **baseline 70,2** · QTIP 57,4 · LLVQ 0 bit de gain **60,7**.

## Le piège à éviter d'abord

Un harnais MMLU maison qui donne 63,1 ne vaut rien si on ne peut pas le
comparer aux 70,2 du papier. MMLU est une famille de protocoles, pas un
nombre : 0-shot ou 5-shot, log-prob de la lettre (`" A"`) ou du texte complet
de la réponse, normalisation par longueur ou non, moyenne par sujet ou par
question. Deux variantes raisonnables s'écartent de plusieurs points.

**Donc : ne pas écrire de harnais avant d'avoir validé qu'on reproduit un
chiffre connu.**

## Route recommandée — exporter, puis harnais standard

L'idée : ce qu'on évalue, c'est **la quantification**, pas notre code
d'inférence. Donc on déquantifie vers un checkpoint standard et on le passe au
harnais que tout le monde utilise. Les chiffres deviennent directement
comparables au papier, et on ne valide aucun harnais.

C'est légitime parce que la chaîne est déjà verrouillée : l'artefact décode
**bit pour bit** vers les poids évalués (testé), et le noyau GPU est vérifié à
10⁻⁸ contre ces mêmes poids. Le checkpoint déquantifié *est* notre modèle.

### Étape 1 — `bin/export` (~1-2 h de travail)

Charger le checkpoint FP16 + superposer l'artefact LLVQ (`artifact::load`, qui
existe), puis écrire un répertoire HF standard : `model.safetensors` en f16,
`config.json`, `tokenizer.json`. Toutes les briques sont là — `decode_matrix`
rend déjà des `Vec<f32>`.

Sortie : ~8 Go sur disque. Vérification obligatoire : recharger l'export et
exiger les mêmes poids que l'overlay, sinon on évalue autre chose que ce qu'on
croit.

### Étape 2 — installer le harnais (⚠️ demande un go)

```bash
pip install lm_eval          # lm-evaluation-harness, ~50 Mo de dépendances
```

`mlx_lm.evaluate` est déjà installé et s'appuie dessus — il ne lui manque que
ce paquet. C'est le seul téléchargement du plan.

### Étape 3 — trois modèles, un protocole (~1-3 h de machine)

```bash
mlx_lm.convert --hf-path <export> --mlx-path qwen3-4b-llvq-mlx   # f16, non requantifié
mlx_lm.evaluate --model <chemin> --tasks mmlu --num-shots 5
```

Sur les trois, dans cet ordre de valeur :

| modèle | ce que ça répond |
|---|---|
| **Qwen3-4B FP16** | notre baseline reproduit-elle les 70,2 du papier ? *C'est le test du protocole.* |
| **LLVQ 2 bits** | où on est vraiment, contre les 60,7 du papier |
| **MLX 4 bits** | le chiffre qui décide de la suite (cf. `face-au-4-bits.md`) |

**Si la baseline ne tombe pas à ~70,2, on arrête et on corrige le protocole
avant de regarder les deux autres.** C'est la même règle que le contrôle
identité de la Phase 5 : le test qu'on relance en premier quand un résultat
paraît absurde.

## Route alternative — harnais maison en Rust

Faisable, et pas très long : `Qwen3::logits()` existe déjà, il ne manque que
la lecture structurée du parquet MMLU (`hf_parquet_text` ne sait extraire que
des colonnes texte, or MMLU a `choices` en liste et `answer` en entier) et la
construction des prompts 5-shot.

- **Pour** : zéro dépendance Python, tourne directement sur l'overlay, reste
  dans l'esprit du projet.
- **Contre** : ~1 jour, et surtout il faudrait **quand même** valider contre un
  chiffre connu — donc faire tourner le harnais standard au moins une fois. On
  paie les deux.
- **Coût machine** : ~14 000 questions × ~0,6 s (prompt 5-shot d'environ 1 000
  tokens sur Metal) ≈ **2,3 h par modèle**, contre nettement moins via MLX.

À faire **plus tard**, si on veut un CI qui surveille la qualité sans Python.
Pas maintenant.

## Ce que MMLU ne dira pas

- **Rien sur le régime batché**, donc rien de directement transposable à un
  serveur d'inférence.
- **Rien sur les tâches métier.** Cf. [arXiv:2607.08734](https://arxiv.org/abs/2607.08734) :
  perplexité et exactitude restent stables pendant que les réponses
  individuelles changent. Un benchmark d'extraction documentaire dirait autre
  chose, et c'était au plan initial.
- **Rien sur le 70B**, qui est le seul endroit où la thèse a un sens.

## Ordre de bataille suggéré

1. `bin/export` + sa vérification — c'est du travail sûr, réutilisable, et il
   débloque tout le reste (y compris de publier un checkpoint déquantifié que
   `transformers` sait lire, ce qui répond à la limite « ce n'est ni GGUF ni
   safetensors »).
2. MMLU sur les trois modèles.
3. Selon le verdict : si LLVQ tient ~60 et que le 4 bits est à ~68, la
   conclusion de `face-au-4-bits.md` se durcit et il faut attaquer le vrai
   problème (RAM du format rapide) avant toute démo commerciale.
