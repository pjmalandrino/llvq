# Lot 4, outillage payant et CI : journal

Aucune exécution locale : aucun cargo, uv, pytest ni script du dépôt. Seuls `python3 -m py_compile` sur les fichiers
Python modifiés, `bash -n` sur `train.sh`, `tomllib` et `yaml.safe_load` pour relire les manifestes et le workflow,
et `python3 -c` pour l'arithmétique.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| PY-06 | `ops/run.py` | `dequant` imprime son coût (et lit `--timeout`) avant `run_uv_job` ; un `--timeout 2h30m` refuse avant tout lancement | relu, NON COMPILÉ ; `py_compile` vert |
| PY-07 | `ops/run.py` | `oracle` appelle `cap_ok` avant `device_ok` (refus code 2 sur `a100-large`, `t4-medium`) et imprime `usd_h × 20 min` avant `run_job` ; le `"20m"` est une seule variable pour le coût et le job | relu, NON COMPILÉ ; `py_compile` vert |
| PY-03 | `ops/run.py` | `launch` passe `timeout=f"{max(60, math.ceil(wall * 1.5 * 60))}m"` ; `import math` ajouté | relu, NON COMPILÉ ; table ci-dessous |
| PY-05 | `ops/run.py` | lecture de la recette et validation de la réécriture `CUDA_COMPUTE_CAP` déplacées avant `git_state` et `create_repo` ; octets envoyés identiques | relu, NON COMPILÉ ; `py_compile` vert |
| PY-08 | `ops/run.py` | `launch --calib` accepte `dclm-edu` ; défaut `c4` inchangé | relu ; `CalibCorpus::parse` (`llvq-llm/src/corpus.rs:555`) accepte la même chaîne |
| SEC-01 | `ops/run.py` | `bench` et `dequant` ne passent `HF_TOKEN` qu'à une image `hf.co/spaces/<namespace>/llvq-runner*`, et l'annoncent sinon (`_is_runner_image`, `_hf_token_secret`) | relu, NON COMPILÉ ; `py_compile` vert |
| PY-13 | `ops/hf_publish_prep.py` | refus par nom d'un `--revision` absent ; refus si `hidden_size`, `num_hidden_layers`, `intermediate_size` ou `vocab_size` du `config.json` empaqueté diffèrent de ceux du checkpoint à cette révision (le `config.json` du checkpoint est ajouté aux motifs téléchargés) | relu, NON COMPILÉ ; `py_compile` vert |
| PY-11 | `ops/llvqtune/llvqtune/adapters/dclm_corpus.py`, `mmlu_corpus.py` | `hf_hub_download` reçoit `revision=` : `dbad8ad71224482740cd9c9d353591adbf62fe04` pour DCLM-edu (déjà dans `SHARD`, `ops/duel/inputs.py:55`, `llvq-llm/src/corpus.rs:784`), `c30699e8356da336a370243923dbaf21066bb9fe` pour `cais/mmlu` (`ops/contrast_map.py:70`, révision de la mesure de recouvrement de `ops/mmlu_aux_overlap.py:25`) ; deux tests | relu, NON COMPILÉ ; `py_compile` vert |
| PY-16 | `ops/llvqtune/train.sh` | la garde `MAX_FIRST_KL` ne laisse passer qu'une première perte finie et ≤ au plafond ; NaN et valeur vide refusent (code 4) ; test `test_a_nan_first_loss_is_refused_under_a_ceiling` | relu ; `bash -n` vert ; l'expression essayée par `python3 -c` sur 0.3, 0.5, 0.9, nan, vide : passe, passe, refuse, refuse, refuse (0.5 au plafond passe, comme avant) |
| TST-07 | `Cargo.toml` de `llvq-core`, `llvq-search`, `llvq-quant`, `llvq-artifact`, `llvq-bench` | table `[lints.rust] unsafe_code = "forbid"` ajoutée en fin de chaque manifeste ; les `#![forbid(unsafe_code)]` restent | relu, NON COMPILÉ ; `tomllib` relit les cinq tables |
| TST-08 / COH-06 | `.github/workflows/ci.yml` | `paths-ignore` devient `paths: ['**', '!**.md', '!docs/**', '!paper/**', '!paper2/**', '!proofs/**', 'docs/ENV.md', 'configs/README.md']`, pour `push` et `pull_request` | relu ; `yaml.safe_load` relit les deux listes |

PY-03, la nouvelle chaîne comparée à l'ancienne (*computed*, `python3 -c`) :

| estimation (h) | ancien timeout | nouveau timeout |
|---|---|---|
| 0,3 | 1 h | 60 m |
| 0,5 | 1 h | 60 m |
| 1,0 | 1 h | 90 m |
| 1,3 | 1 h | 118 m |
| 2,6 | 3 h | 235 m |
| 3,9 | 5 h | 351 m |

L'unité est établie par le dépôt : `cmd_oracle` passe `"20m"` à `run_job` depuis toujours, et 158 scripts de
`ops/jobs/` passent `--timeout <n>m` à `bench`, qui le transmet tel quel à `run_job`. La bibliothèque en cache
(`huggingface_hub` 2.4.0, `_jobs_api.py:673`) lit `s`, `m`, `h`, `d`. Écart à la forme du vérificateur : le plancher
de 60 minutes est gardé. Sans lui, une estimation de 0,1 h donnerait 9 minutes là où l'ancien code donnait une heure,
et l'estimation exclut le téléchargement du checkpoint et les passes avant. Avec lui, le nouveau timeout n'est jamais
inférieur à l'ancien : `int(1,5 w) · 60 ≤ 90 w ≤ ceil(90 w)`.

SEC-01, ce qui a été établi avant d'écrire. 25 scripts de `ops/jobs/` passent `--image` vers une image tierce :
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (13), `vllm/vllm-openai:v0.26.0` (7), `ghcr.io/ggml-org/llama.cpp:full-cuda`
(3), `nvidia/cuda:12.8.1-devel-ubuntu24.04` et `nvidia/cuda:12.4.1-devel-ubuntu22.04` (1 chacun). Aucun ne nomme
`HF_TOKEN` dans le conteneur ; `dclm-14b-rowscales.sh:172` et `dclm-8b-rowscales.sh:54` écrivent même que le jeton
n'est pas nécessaire. Les dépôts lus dans ces conteneurs sont publics (`Qwen/*`, `unsloth/*-GGUF`, `cais/mmlu`,
`HuggingFaceTB/dclm-edu`), et le bucket est atteint par le volume monté, pas par le jeton (le vérificateur le dit
aussi). Les `hf buckets cp` de ces scripts tournent sur la machine de l'opérateur, avant le lancement. Pour `dequant`,
`run_uv_job` reçoit l'image par défaut de la bibliothèque ; dans la version 2.4.0 en cache, le script local part par
un volume de bucket (`hf_api.py:13790-13800`), pas par un dépôt privé lu avec le jeton. Risque résiduel, non mesuré :
ces jobs téléchargent désormais en anonyme, sous les limites de débit anonymes du Hub.

## Laissé de côté

| ID | raison |
|---|---|
| PY-07, partie `bench` | la mission borne PY-07 à `oracle`. Le refus d'une saveur absente de `FLAVORS` sous `--any-flavor` reste à faire ; le vérificateur a établi qu'aucun script n'en utilise une, donc le refus ne casserait rien |
| PY-11, révision dans le journal | le constat propose aussi d'écrire la révision dans le journal de llvqtune. Cela touche l'en-tête `opened` de `__main__.py`, que des scripts relisent peut-être ; non fait. La révision est désormais une constante `REVISION` dans chaque adaptateur |
| PY-19 | voir « Décision d'opérateur ». Contradiction dans le constat : les jobs llvqtune installent `transformers==5.17.0` dans l'image `pytorch:2.5.1` (`dclm-8b-rowscales.sh`, `dclm-14b-rowscales.sh`) et ont abouti, donc la paire n'est pas « connue cassée » pour ce chemin. Passer `dtype=` au lieu de `torch_dtype=` (`__main__.py:93,139`) casse sous transformers < 4.56, que le plancher `>=4.44` admet |

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| QUA-13 | Défaut inatteignable. `rewrite` (`llvq-bench/examples/rhoapply.rs:838-852`) écrit l'en-tête du fichier réécrit avec `h.version`, la version de la source. Les trois fichiers relus avec l'en-tête source jeté (`p_path` l. 2054-2067, `a_path` et `b_path` l. 2202-2212) sortent tous de `rewrite`, donc leur version est celle de la source. Le site l. 1385 relit le même fichier que l'en-tête `header` de la l. 1332. Rien n'est modifié. Le chemin réel est `examples/rhoapply.rs`, pas `src/bin/rhoapply.rs` |

Deux remarques pour le rapport d'ensemble. La CI a déjà un job Python (`ci.yml:142-159`) qui lance les tests de
llvqtune sans torch ; « aucun test Python ne tourne en CI » est donc vrai pour `llvq-tetra` seulement. Et le nouveau
test de `test_train_sh.py` tournera dans ce job, puisqu'il ne demande pas torch.

## Décision d'opérateur

| sujet | recommandation et raison |
|---|---|
| PY-19 | Aligner l'extra `torch` de `ops/llvqtune/pyproject.toml` sur ce que les jobs installent : `transformers>=5.17` et garder `torch>=2.5`, puisque l'image `pytorch:2.5.1` avec 5.17.0 a tourné. Ensuite seulement passer `dtype=`. Ne pas borner `transformers<5` : cela exclurait la version que tous les jobs récents épinglent. Le `uv.lock` résout déjà torch 2.14.0 et transformers 5.17.0 |
| SEC-01, suite | Option recommandée : un drapeau `--hf-token` explicite pour le cas, aujourd'hui inexistant, d'une image tierce qui lit un dépôt privé, plutôt qu'un retour au jeton par défaut. Épingler les images tierces par `@sha256:` et monter un sous-chemin du bucket pour elles, comme le constat le propose ; ces deux points changent le sha256 de scripts que des préregs nomment, donc au prochain job de chaque script |
| PY-04 | Ajouter `COPY --from=build /src/COMMIT /usr/local/share/llvq/COMMIT` à `ops/Dockerfile.cuda` à la prochaine reconstruction voulue de l'image, et ne l'utiliser dans le gabarit des jobs qu'aux scripts nouveaux. Les 107 scripts qui épinglent déjà le sha du Space couvrent l'essentiel |
| PY-15 | Corriger `TIMEOUT` (refuser une valeur sans suffixe `m`) dans chaque script seulement au prochain job de ce script, avec un nouveau préreg : le sha256 de ces scripts est nommé par des préregs tamponnés |
| PY-18 | Faire écrire par `bench`, `launch`, `oracle` et `dequant` une ligne provisoire dans `docs/data/jobs.csv` (date, id, nom, saveur, statut « lancé ») que la clôture complète. C'est un changement de méthode (`docs/METHODE.md:148` dit que la clôture tient le registre à la main), donc à décider |
| SEC-08 | Restreindre le périmètre de `publish` à `git ls-files` intersecté avec `UPLOAD_ALLOW`. Plus sûr qu'une liste de motifs ignorés, mais à vérifier par un `publish` réel, car un fichier généré au build pourrait être attendu |
| SEC-10 | Épingler les deux images CUDA par `@sha256:` et les actions GitHub par sha de commit. L'image reconstruite doit repasser `oracle`, donc un job payant ; à grouper avec la prochaine reconstruction |
| Job Python en CI pour `llvq-tetra` | Ajouter `llvq-tetra` au job `python` existant demande torch et transformers, soit plusieurs minutes et ~2 Go par exécution. Recommandé seulement pour les tests qui n'importent pas torch, s'il y en a assez pour valoir le coût |
| CI, étape grep `forbid` | Une fois `[lints.rust]` vérifié par clippy, l'étape `ci.yml:127-135` devient redondante pour les cinq crates cœur ; elle reste utile pour `llvq-runtime`, qui n'a pas reçu la table. Garder l'étape, ou ajouter la table à `llvq-runtime` après un `git grep unsafe` sur ses cibles |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

Aucun.

## Commandes lancées et leur résultat

- `python3 -m py_compile` sur `ops/run.py`, `ops/hf_publish_prep.py`, `dclm_corpus.py`, `mmlu_corpus.py`,
  `test_mmlu_corpus.py`, `test_main_corpus.py`, `test_train_sh.py` : vert, aucune sortie.
- `bash -n ops/llvqtune/train.sh` : vert.
- `tomllib` sur les cinq `Cargo.toml` : `{'rust': {'unsafe_code': 'forbid'}}` pour chacun.
- `yaml.safe_load` sur `ci.yml` : les deux listes `paths` telles qu'écrites.
- `git grep -n unsafe` sur `src/`, `tests/`, `examples/`, `benches/` des cinq crates cœur : seulement des commentaires
  et les attributs `#![forbid(unsafe_code)]` (`llvq-quant/tests/schur.rs:1`, `llvq-bench/examples/hstats.rs:39`,
  `rowshrink.rs:85`, les cinq `lib.rs`). Aucun `[lints]` dans ces manifestes ni dans le `Cargo.toml` racine.

Aucun test lancé.

## À lancer quand la machine est libre

| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo clippy -j 8 -p llvq-core -p llvq-search -p llvq-quant -p llvq-artifact -p llvq-bench --all-targets` | zéro avertissement, zéro erreur | ajouter la ligne `let _ = unsafe { 0 };` au début de `main` dans `llvq-bench/src/bin/lcap.rs` : la compilation doit échouer sur `unsafe_code` ; défaire l'édition, revert vert |
| même commande, mutant dans un test : ajouter `#[test] fn m() { unsafe {} }` à la fin de `llvq-artifact/tests/` (un fichier existant) | doit échouer sur `unsafe_code` | défaire, revert vert |
| `cd ops/llvqtune && PYTHONPATH=. uv run --group dev pytest -q tests/test_train_sh.py -k "ceiling"` (ou `python3 -m pytest` comme la CI) | `3 passed` : NaN refusé (4), 0.9 refusé (4), 0.3 lancé (0) | `ops/llvqtune/train.sh`, garde `MAX_FIRST_KL` : remplacer `math.isfinite(v) and v <= float(...)` par `not (v > float(...))` ; attendu : le test NaN rougit, les deux autres restent verts |
| `cd ops/llvqtune && PYTHONPATH=. uv run --extra torch pytest -q tests/test_mmlu_corpus.py tests/test_main_corpus.py -k revision` | `2 passed` (skip si torch ou huggingface_hub absents) | `mmlu_corpus.py`, retirer `revision=REVISION,` de `fetch_split` : attendu `KeyError: 'revision'`, 1 failed ; idem dans `dclm_corpus.py` |
| `uv run ops/run.py selftest` | les lignes `ok` comme avant (lit `config.json` du Hub, rien de facturé) | aucun |
| `uv run ops/run.py launch --help` | `--calib {c4,wikitext2,wikitext2-test,dclm-edu}` | aucun |
| `python3 -c "import math; [print(w, max(60, math.ceil(w*1.5*60))) for w in (0.5, 1.3, 2.6)]"` | `60`, `118`, `235` minutes | aucun |
| Le banc sans coût ci-dessous, depuis `ops/` : il remplace `run_job` et `run_uv_job` par une fonction qui imprime ce qu'elle reçoit et sort, donc RIEN n'est lancé. `HF_TOKEN=dummy` n'est envoyé nulle part | voir les attendus sous le bloc | `ops/run.py`, `_is_runner_image` : remplacer `startswith("llvq-runner")` par `startswith("")` ; attendu : le cas pytorch reçoit `['HF_TOKEN']` |

```bash
cd ops && HF_TOKEN=dummy uv run --with 'huggingface-hub>=1.26' python - <<'PY'
import sys, huggingface_hub as h
def stop(**k):
    s = k.get("secrets")
    print("   would launch: timeout", k.get("timeout"), "secrets", None if s is None else sorted(s))
    raise SystemExit(0)
h.run_job = stop
h.run_uv_job = stop
import run
cases = [
    ["bench", "true", "--image", "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", "--timeout", "20m"],
    ["bench", "true", "--image", "hf.co/spaces/Pier-Jean/llvq-runner-cuda", "--timeout", "20m"],
    ["oracle", "--image", "x", "--flavor", "a100-large"],
    ["oracle", "--image", "x"],
    ["dequant", "--awq-repo", "Qwen/Qwen3-14B-AWQ", "--base-repo", "Qwen/Qwen3-14B",
     "--bucket", "b", "--name", "n", "--timeout", "2h30m"],
    ["dequant", "--awq-repo", "Qwen/Qwen3-14B-AWQ", "--base-repo", "Qwen/Qwen3-14B",
     "--bucket", "b", "--name", "n"],
]
for argv in cases:
    sys.argv = ["run.py", *argv]
    print("==", " ".join(argv[:4]))
    try:
        print("   returned", run.main())
    except SystemExit as e:
        print("   exit", e.code)
PY
```

Attendus du banc :
1. `bench` pytorch : la ligne de coût, puis `HF_TOKEN not passed: pytorch/... is not hf.co/spaces/<namespace>/llvq-runner*`, puis `secrets None`.
2. `bench` runner : la ligne de coût, puis `secrets ['HF_TOKEN']`.
3. `oracle` sur `a100-large` : `refused, a100-large: sm_80 < sm_89, ...`, `returned 2`, aucune ligne `would launch`.
4. `oracle` par défaut (`l4x1`) : `l4x1 at 0.80 $/h = 0.0133 $/min; ceiling 20m → at worst 0.27 $` AVANT `would launch: timeout 20m secrets None`.
5. `dequant --timeout 2h30m` : `exit --timeout '2h30m': write a number and ONE unit, ...`, aucune ligne `would launch`.
6. `dequant` par défaut : `cpu-upgrade at 0.03 $/h; ceiling 4h → at worst 0.12 $`, puis `HF_TOKEN not passed: the default uv image ...`, puis `would launch: timeout 4h secrets None`.

`hf_publish_prep.py`, sur une COPIE du fixture (le script écrit dans le répertoire quand il passe) :

| commande exacte | ce qu'elle doit montrer | mutant |
|---|---|---|
| `cp -r llvq-tetra/tests/fixtures/mini /tmp/pp && uv run --with huggingface_hub ops/hf_publish_prep.py /tmp/pp` | `REFUSED: --revision is missing ...`, code 2, rien téléchargé | aucun |
| `uv run --with huggingface_hub ops/hf_publish_prep.py /tmp/pp --revision 1cfa9a7208912126459214e8b04321603b3df60c` | télécharge les fichiers tokenizer et `config.json` du 4B, puis `REFUSED: hidden_size is 136 in /tmp/pp/config.json and 2560 in Qwen/Qwen3-4B@...`, code 2, `/tmp/pp` inchangé | aucun |
| `publish` | aucune commande sans coût ne l'exerce ; relecture seule. À voir au prochain `publish --compute-cap` réel : la ligne `refused: ... CUDA_COMPUTE_CAP` doit arriver avant `space ...` | aucun |

CI (TST-08), à voir sur GitHub au prochain lot de PR, pas localement :

| action | attendu |
|---|---|
| une PR qui ne touche que `docs/ENV.md` | la CI part |
| une PR qui ne touche que `configs/README.md` | la CI part |
| une PR qui ne touche que `docs/STYLE.md` ou `README.md` | la CI ne part pas |
| une PR qui ne touche que `.github/workflows/ci.yml` | la CI part ; si elle ne part pas, `**` n'attrape pas les chemins à point et il faut ajouter `'.github/**'` en tête de la liste |
