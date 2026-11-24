# Lot 3, lecteur Python `llvq-tetra` : journal

Base : l'arbre après le lot 1 (`34-lot1-python.md`), non commité. Aucune exécution : seul `python3 -m py_compile` a tourné.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| PY-02 (version) | `llvq_tetra/reader.py`, `llvq_tetra/quantizer.py` | `reader.require_layout_version` refuse par nom un `llvq_layout_version` présent et différent de 1 ; appelé dans `PackedModel.__init__` (après `quant_method`) et `LlvqConfig.__init__` (avant `code_order`) ; l'absence passe | relu, NON COMPILÉ ; `py_compile` sans erreur |
| PY-10 | `llvq_tetra/fused.py`, `llvq_tetra/chat.py` | `FusedTetraLinear` et `Int4FusedLinear` surchargent `_apply` : une conversion qui change l'appareil des tenseurs résidents lève `RuntimeError` nommée avant tout déplacement ; `forward` refuse une entrée sur un autre appareil ; `chat.py` pose `LLVQ_HF_DEVICE` comme `gentokens.py` | relu, NON COMPILÉ ; `py_compile` sans erreur |

### PY-02 : ce que la lecture établit

Le champ est toujours là dans un dossier valide.
- `llvq-llm/src/hfpack.rs:409` écrit `"llvq_layout_version": LAYOUT_VERSION` (= 1, `:67`) dans `quantization_config`, à la racine de ce bloc, à côté de `quant_method`.
- Il y est depuis la création du fichier : `git log -S llvq_layout_version` donne `0c4197f` (2026-09-29, premier commit de `hfpack.rs`), et `git show 0c4197f` contient déjà la ligne.
- `fixtures/tiny/config.json:12` et `fixtures/mini/config.json:19` portent `"llvq_layout_version": 1`, au même endroit.
- `ops/hf_publish_prep.py:109-132` réécrit `config.json` en ajoutant `auto_map` et garde `quantization_config` tel quel.
- Je n'ai pas pu lire les `config.json` des trois dépôts du Hub : le cache local (`~/.cache/huggingface/hub`) n'en a aucun pour les dépôts LLVQ, et le réseau est interdit.

Forme retenue : l'absence reste acceptée. Raison : je n'ai pas lu les fichiers publiés eux-mêmes, et refuser l'absence n'ajoute rien contre le scénario du constat (un `hfpack` futur écrirait 2, refusé). Un bloc sans le champ n'est pas écrit par `hfpack`, et les refus existants (`tetra_fingerprint`, `block_dim`, `code_order`, `records`) le prennent. Une seule constante, `reader.LAYOUT_VERSION`, sert aux deux lecteurs ; `quantizer` l'importe de `reader`, qui n'importe ni torch ni transformers. La porte MLX (`ops/mlx_tetra/llvq_mlx.py:697`) passe par `PackedModel` et hérite du refus sans modification.

### PY-02 : la seconde moitié (tenseur absent, `torch.empty`), non implémentée

Chemin établi par la lecture de `transformers` 5.18.0 (le `.venv` local de `llvq-tetra`) :
1. `_process_model_before_weight_loading` remplace les couches ; les tampons naissent `torch.empty` (`modules.py:66-72`, `vocab.py:30-32`, `RotationTable`), sous le contexte d'initialisation (méta).
2. `convert_and_load_state_dict_in_model` remplit chaque clé trouvée (`core_model_loading.py:1366-1383`), la retire de `missing_keys` et pose `_is_hf_initialized = True` sur le tenseur.
3. `_finalize_model_loading` (`modeling_utils.py:4497-4530`) : `_move_missing_keys_from_meta_to_device` remplace chaque clé manquante par `torch.empty_like` sur l'appareil, donc mémoire non initialisée (`:4764-4775`) ; `_initialize_missing_keys` appelle `_init_weights`, qui ne connaît que `nn.Linear`, `nn.Embedding`, les normes (`:2315-2345`) et ne touche donc ni `TetraLinear`, ni `Int4Linear`, ni `RotationTable`, ni `Int4Vocab` ; `log_state_dict_report` écrit un avertissement et ne lève que sur `error_msgs`, `conversion_errors` ou une taille différente (`utils/loading_report.py:289-317`).
4. Ensuite seulement `hf_quantizer.postprocess_model(model)` (`modeling_utils.py:4361-4365`) appelle `_process_model_after_weight_loading(model)` **sans** la liste des clés manquantes (`quantizers/base.py:180-198`). Le crochet ne les voit pas.

Conséquences.
- L'initialisation à NaN proposée par le constat ne tient pas : `empty_like` remplace le tampon manquant après coup, NaN compris.
- Le « tripwire » (`ops/hf_tripwire_probe.py`) ne couvre pas ce cas. Il traite la méthode non enregistrée (transformers saute la quantification et réinitialise des poids denses) ; ici la méthode est enregistrée et une clé du fichier manque.
- Le refus pourrait lire le drapeau `_is_hf_initialized` dans le crochet d'après chargement. Je ne l'ai pas fait : c'est un détail interne de 5.18, le paquet accepte `transformers>=5.17` sans borne haute (`pyproject.toml:44`), et une version qui cesserait de poser ce drapeau ferait refuser un dossier valide. Je ne peux donc pas montrer qu'il ne se déclenche jamais sur un dossier valide.

Plan recommandé, à faire avec des tests qui tournent : dans `_process_model_before_weight_loading`, 5.18 passe `checkpoint_files` (`modeling_utils.py:4294-4300`). Lire les noms de l'en-tête (`safe_open(..., framework="np").keys()`, sans charger de tenseur) et refuser par nom tout tampon persistant d'un module `TetraLinear`, `Int4Linear`, `RotationTable` ou `Int4Vocab` absent du fichier. Sur un dossier valide l'ensemble vérifié est exactement celui que `test_load.py:58-66` et la porte B de `gentokens.py:137-143` trouvent sans clé manquante. Sauter le contrôle si `checkpoint_files` manque ou n'est pas une liste de `.safetensors`. Environ 20 lignes. Point à établir d'abord : si 5.17.0 passe aussi `checkpoint_files` (seule 5.18.0 est installée ici). Le scénario du constat (un layout 2) est déjà fermé par le refus de version ; ce reste ne vaut que pour un fichier tronqué ou retouché à la main.

### PY-10 : ce que la lecture établit

- `from_pretrained` ne déplace pas un module fusé vers un autre appareil. `update_device_map` charge tout sur le CPU (`quantizer.py:109-118`), les modules fusés naissent dans `finalize` sur `device` (`integration.py:212-236`), puis `finalize` fait `model.to(device)` (`:196-197`), même appareil. Après `postprocess_model`, `from_pretrained` ne fait plus de `.to()` ; `accelerate_dispatch` ne part que pour plusieurs appareils (`modeling_utils.py:4359`), exclus par `{"": "cpu"}`.
- Ce `.to()` de `finalize` reste accepté : la sonde `torch.empty(0, device=résident.device)` passée par la fonction de conversion donne `mps:0` contre `mps:0`, ou `cuda:<courante>` contre `cuda:<courante>`.
- Une conversion de dtype (`model.to(torch.float16)`, `.half()`) garde l'appareil et passe. La sonde est un tenseur vide : rien n'est copié.
- `torch.nn.Module._apply(self, fn, recurse=True)` est la signature de torch 2.14.1 (`torch/nn/modules/module.py:930`) ; `to`, `cuda`, `cpu`, `half`, `to_empty` y passent tous. Le refus vient avant `super()._apply`, donc le module fusé ne bouge pas ; les modules visités avant lui dans l'ordre de `children()` ont pu bouger, et le message demande de recharger.
- Appelants vérifiés : `gentokens.py:125,135` et `ops/hf_cuda_check.py:106` (même appareil), `checkmps.py:113` et `tests/test_mps.py` (`device_map="mps"`), `ops/hf_cleanroom_check.py:107` (`--device cpu`, chemin dense). Tous restent acceptés.
- La branche `torch.ops.llvq.tv_tetra48` hors MPS et hors CUDA de `forward` n'est plus atteignable par le paquet : `_compress` n'arme que sur `mps` ou `cuda`, et une entrée sur un autre appareil est refusée avant. Je ne l'ai pas supprimée (aucune suppression de code dans ces lots).
- `chat.py` : `setdefault("LLVQ_HF_DEVICE", device)`. Sous `LLVQ_HF_FUSED=0` (défaut) `target_device(None)` rend `"cpu"` sans lire la variable : rien ne change. Sous `LLVQ_HF_FUSED=1` sans `--device`, `pick_device` suit le même ordre que `"auto"` (cuda, mps, cpu) : rien ne change. Seul `--device` différent de l'accélérateur change, et ce cas cassait.
- Le contrôle d'appareil dans `forward` coûte une comparaison par appel, sur le bras `transformers` seulement ; aucun moteur servi ni chiffre publié n'en dépend.

## Laissé de côté

| ID | raison |
|---|---|
| PY-02 (clés manquantes) | Voir plus haut : le crochet d'après chargement ne reçoit pas les clés manquantes, et le seul signal lisible (`_is_hf_initialized`) est interne à 5.18. Plan décrit, à faire avec les tests qui tournent. |
| PY-12 | Hors mandat (les mutants de la porte MLX se rejouent sur Mac). Recommandation ci-dessous. |
| `Int4Vocab` sous `.to()` | Hors constat. Ses tampons sont de vrais tampons et suivent `.to("cpu")` ; le `forward` appellerait alors un noyau Metal sur des tenseurs CPU. À traiter avec la même garde de `forward` si l'opérateur le veut. |

## Décision d'opérateur

PY-12, recommandation. Garder `strict=False` et ajouter, juste après `model.load_weights(weights, strict=False)` dans `ops/mlx_tetra/llvq_mlx.py:754`, une comparaison d'ensembles : les noms de `weights` contre les paramètres du modèle MLX moins ceux des modules `TetraLinear` et int4 et de l'embedding posés à la main. Refuser par nom les deux écarts (nom du fichier sans paramètre, paramètre non couvert). `strict=True` seul ne marche pas, comme le vérificateur l'a lu (`20-verif-python-securite.md`, PY-12). Gravité cosmétique pour le chemin servi, épinglé par sha256 dans `served.py:30-32`. À faire sur Mac, avec un mutant de `mutate.py` qui renomme une norme dans une copie de `fixtures/mini` et doit rougir.

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| aucun | PY-02 et PY-10 tiennent tous deux à la relecture. |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

Aucun fichier Rust ni CUDA. `fused.py` est le seul fichier qui touche le bras CUDA du paquet : la garde vaut aussi pour `torch.ops.llvq_cuda.tv_tetra48`, sans changer son appel.

## Fichiers modifiés

- `llvq-tetra/llvq_tetra/reader.py` : `LAYOUT_VERSION`, `require_layout_version`, appel dans `PackedModel.__init__`.
- `llvq-tetra/llvq_tetra/quantizer.py` : import et appel dans `LlvqConfig.__init__`.
- `llvq-tetra/llvq_tetra/fused.py` : `_require_resident_device`, `_require_input_device`, deux `_apply`, deux gardes de `forward`.
- `llvq-tetra/llvq_tetra/chat.py` : `import os`, `setdefault("LLVQ_HF_DEVICE", device)`.
- `llvq-tetra/tests/test_quantizer.py` : deux tests de version.
- `llvq-tetra/tests/test_fused_device.py` : nouveau, six cas, sans carte ni MPS.

## Commandes lancées et leur résultat

- `python3 -m py_compile llvq_tetra/fused.py llvq_tetra/quantizer.py llvq_tetra/reader.py llvq_tetra/chat.py tests/test_quantizer.py tests/test_fused_device.py` : sans erreur.
- Aucun `uv`, aucun `pytest`, aucun script du dépôt.

## À lancer quand la machine est libre

| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `cd llvq-tetra && uv run --group dev pytest -q` | toute la suite verte, 64 tests du lot 1 plus 8 nouveaux | aucun ; c'est la base avant les mutants |
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_quantizer.py -k layout_version` | `2 passed` | `llvq_tetra/reader.py`, `require_layout_version` : `if version is not None and version != LAYOUT_VERSION:` → `if False:` ; attendu `2 failed`. Second mutant : retirer `version is not None and ` ; attendu `1 failed` (`test_another_layout_version_is_refused_by_name`, sur l'absence), `1 passed` |
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_quantizer.py -k reader_refuses` | `1 passed` | `llvq_tetra/reader.py`, `PackedModel.__init__` : supprimer la ligne `require_layout_version(self.qc.get("llvq_layout_version"))` ; attendu `1 failed` (le lecteur continue et échoue plus loin sur un autre message ou sur `model.safetensors` absent) |
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_fused_device.py` | `6 passed`, sans MPS ni carte | `llvq_tetra/fused.py`, `_require_resident_device` : `if target != resident.device:` → `if False:` ; attendu `2 failed, 4 passed` (`test_a_move_to_another_device_is_refused_by_name`) |
| idem | idem | `llvq_tetra/fused.py`, `_require_input_device` : `if x.device != resident.device:` → `if False:` ; attendu `2 failed, 4 passed` (`test_an_input_on_another_device_is_refused_by_name`, autre exception levée plus loin) |
| idem | idem | `llvq_tetra/fused.py`, `FusedTetraLinear._apply` : supprimer la ligne `_require_resident_device(...)` ; attendu `1 failed, 5 passed` (cas `tetra` seulement) |
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_mps.py tests/test_load.py` (sur le Mac, MPS présent) | verts comme avant : le `.to("mps")` de `finalize` passe la garde | aucun |
| `cd llvq-tetra && uv run --group dev python -m llvq_tetra.checkmps <dossier 4B>` (Mac, quand l'opérateur le décide) | sortie identique à la précédente : aucun module fusé refusé au chargement | aucun |
