# Lot 1, petits correctifs Python (`llvq-tetra`, `ops/hf_cleanroom_check.py`) : journal

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| PY-01 | `llvq-tetra/llvq_tetra/metal.py` | `_extension_cuda` refuse par nom `LLVQ_HF_TILE` différent de `integration.TILE_BLOCKS`, avant le contrôle de carte ; aucun fichier de noyau touché | test `test_a_cuda_tile_the_modules_do_not_allocate_is_refused` ; mutant `if tile != TILE_BLOCKS:` → `if False:` : 1 failed, remis : 1 passed |
| PY-09 | `llvq-tetra/llvq_tetra/integration.py` | `target_device` refuse par nom un index de carte autre que la carte courante (0 par défaut, 0 sans CUDA), donc aussi plusieurs index distincts ; `"cuda"` sans index inchangé | tests `test_the_current_card_is_accepted` (5 formes) et `test_another_card_is_refused_rather_than_ignored` (5 formes) ; mutant appel `_require_current_cuda(values)` → `pass` : 5 failed 5 passed, remis : 10 passed |
| PY-14 | `ops/hf_cleanroom_check.py` | deux arguments (argv de longueur 3) refusés par nom, code 2, avant tout chargement ; un argument imprime `4B gate not run: ...` et rend 0 comme avant | test `test_the_cleanroom_check_refuses_a_packed_directory_without_its_dump` ; mutant `if len(argv) == 3:` → `if False:` : 1 failed, remis : 1 passed. `git grep -n hf_cleanroom_check` : `CLAUDE.md:88` et `proofs/preregistration-hf-testpypi-2026-10-07.md:53` passent trois arguments, `docs/mesures/hf-cleanroom-4b-2026-10-01.txt:149,177` un seul ; tous restent acceptés |
| PY-17 | `llvq-tetra/llvq_tetra/quantizer.py` | `LlvqConfig.__init__` refuse par nom `tetra_fingerprint` ou `block_dim` absent ou nul : « the quantization_config has no <champ>: this directory is not a Tetra model packed by hfpack » | test paramétré `test_a_config_without_its_map_fields_is_refused_by_name` ; mutant `if getattr(self, field) is None:` → `if False:` : 2 failed, remis : 2 passed |
| PY-20 | `llvq-tetra/llvq_tetra/checkq4guards.py`, `llvq-tetra/llvq_tetra/checkmps.py` | six `assert` de garde deviennent `if ...: raise ValueError(...)`, même condition, même message | test `test_a_mutant_at_the_wrong_site_count_is_refused` (sur `checkq4guards.mutate`, pur texte) ; mutant `raise` remis en `assert n == want` : 1 failed, remis : 1 passed |
| PY-21 | `llvq-tetra/README.md` | la section « Two ways to run it » passe par `device_map="mps"` / `"cuda"`, garde `LLVQ_HF_FUSED=1` sans `device_map`, et ne demande `ninja` et `nvcc` que pour CUDA | relu dans le code : `mps.py:1-6` (`torch.mps.compile_shader`, « no ninja, no command line tools »), `fused.py:133-138` (extension seulement sur CUDA), `quantizer.py:92-101` (ninja exigé pour `cuda` seul), `integration.py:37-49` (`LLVQ_HF_FUSED` sans `device_map`). Aucun chiffre du README changé |
| TST-10 | `llvq-tetra/tests/test_metal.py` | la docstring dit maintenant ce que fait le module : sauté en entier sans MPS, rapporté « skipped » ; le comportement ne change pas | lecture ; suite complète verte |

Écarts à la forme retenue, à trancher par la session principale :
- PY-01 : la forme du vérificateur refuse aussi `LLVQ_HF_TILE=32`, qui donnait une sortie juste avec de la mémoire partagée perdue. C'est le seul refus nouveau qui touche une entrée sans résultat faux. Je l'ai gardé parce que la mission nomme cette forme. `docs/ENV.md:107` dit encore « no range check » : fichier hors de mon domaine, à aligner.
- PY-09 : la forme minimale dit « refuser un index autre que 0 ». J'ai comparé à `torch.cuda.current_device()` (0 sans CUDA) : un utilisateur qui a fait `torch.cuda.set_device(1)` puis `device_map="cuda:1"` chargeait déjà sur la bonne carte, et ne doit pas être refusé. Sur un hôte à carte courante 0, les deux formes sont identiques. Un `"cuda:abc"` reste traité comme avant.
- PY-20 : `checkmps.py:53` n'avait pas de message ; le `raise` reprend celui des deux gardes voisines.

## Laissé de côté

| ID | raison |
|---|---|
| TST-10 (partie `test_fused.py`) | Le constat propose de sortir `test_invnorm_is_one_over_the_root_of_sixteen_m` du `skipif` de module. C'est un changement de comportement, invérifiable ici (le Mac a MPS), et la mission dit en cas de doute de corriger le texte : la docstring de `test_fused.py` ne promet rien de faux. |
| PY-02, PY-03 à PY-08, PY-10, PY-12 et les autres | Hors de ce lot, comme la mission le dit. |

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| aucun | Les sept constats ont été revérifiés dans le code avant correction ; tous tiennent. |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

Aucun fichier Rust ni CUDA. `metal.py:_extension_cuda` est le seul code Python du bras CUDA touché : la garde passe avant le contrôle de carte, et le chemin de compilation (`load(...)`) est inchangé sauf la source de `tile`, qui vaut toujours 64.

## Commandes lancées et leur résultat

- `cd llvq-tetra && nice -n 10 uv run --group dev pytest -q` (création du `.venv` depuis le cache, 41 paquets, sans échec) : `64 passed in 35.34s` ; après les mutants : `64 passed in 19.63s`.
- Mutants, un par test nouveau, par édition puis édition inverse (script dans le scratchpad, compte des sites = 1 vérifié avant chaque application) :
  - PY-17 : `2 failed, 17 deselected` puis `2 passed, 17 deselected`
  - PY-09 : `5 failed, 5 passed, 9 deselected` puis `10 passed, 9 deselected`
  - PY-01 : `1 failed, 18 deselected` puis `1 passed, 18 deselected`
  - PY-20 : `1 failed, 1 deselected` puis `1 passed, 1 deselected`
  - PY-14 : `1 failed, 1 deselected` puis `1 passed, 1 deselected`
- `python -O -m py_compile` sur `llvq_tetra/*.py`, `tests/*.py`, `ops/hf_cleanroom_check.py` : sans erreur.
- `ruff check llvq_tetra tests ../ops/hf_cleanroom_check.py` : un seul avertissement, `tests/test_fused.py:9` `json` importé et inutilisé, antérieur et hors de mes modifications.
- Aucun script `check*` lancé, aucun modèle chargé.
