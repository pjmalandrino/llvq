# Second tour, protocole et Python : constats
Périmètre réellement lu : en entier `llvq-runtime/src/chat.rs`, `openai.rs`, `api.rs`, `store.rs`, `library.rs`, `client.rs`, `main.rs`, `engine.rs:180-819` (types, `Router`, `prepare`, `run_inner`) et `:1180-1586` (`advance`, `finish`, fil du moteur, `Models`), `llvq-llm/src/chatfmt.rs`, `llvq-llm/src/session.rs:100-334`, `llvq-runtime/tests/openapi.rs`, la fixture `ollama-openapi.yaml:1-370`, `docs/plan-runtime.md:170-264`, `llvq-runtime/README.md` ; `chat.html:225-264` et `:740-826` (état, `history`, `ask`). `ops/llvqtune/` : `__main__.py`, `train.sh`, `domain/loop.py`, `domain/schedule.py`, `domain/pairing.py`, `domain/trainable.py`, tous les `adapters/*.py` sauf `__init__`, tous les `trainables/*.py`, `tests/test_row_norms.py`, `tests/test_trainables.py:20-62`, `README.md` ; `ops/mmlu_aux_overlap.py` ; `llvq-llm/src/bin/rowscale.rs:1-200` et sa fixture `:360-420`, `export.rs:1-120`, `llvq-artifact/src/format.rs:1582-1603` (`decode_matrix`), `llvq-quant/src/rotation.rs` (en-tête et signatures), `llvq-bench/examples/rhoapply.rs:20-60`. `ops/mlx_tetra/ppl.py`, `check.py`, `mutate.py`, `strip.py` en entier. `llvq-tetra/tests/` : les onze fichiers en entier, et les clés des deux fixtures.
Survolés : `engine.rs:820-1180` (chemin batch, déjà lu à fond au premier tour), `tests/server.rs` et `tests/pull.rs` (noms des tests), `llvq-llm/src/bin/ppl.rs:160-195`, `ops/jobs/dclm-14b-rowscales.sh:50-190` et les lignes `TEACHER` des jobs d'entraînement, `llvq-tetra/llvq_tetra/fused.py:110-150`.
Non couvert : le reste de `chat.html` (rendu Markdown, déjà relu en 2026-11), `tests/test_main_*.py`, `test_mmlu_corpus.py`, `test_train_sh.py`, `examples/synth_rownorms.py` de llvqtune, `ops/mlx_tetra/llvq_mlx.py` (porte déjà relue), le corps de `rowscale.rs` au-delà de `fold`.
Bilan : 0 critiques, 1 majeure, 12 mineures, 4 cosmétiques

Les fondements des « devrait recevoir » sont dits dans chaque constat. Ce que le dépôt annonce : `docs/plan-runtime.md` (décisions D2, D3, hors périmètre), le commentaire de `engine::Options` (`engine.rs:204-205`), `llvq-runtime/openapi.json` et sa comparaison à la fixture d'Ollama (`tests/openapi.rs`). Ce que je sais d'Ollama et d'OpenAI vient de leur documentation publique ; je signale chaque point dont je ne suis pas certain.

### R2-API-01 llvqtune entraîne σ dans la base naturelle, `rowscale` le plie dans la base tournée : la fonction entraînée n'est pas celle qui est écrite
- Gravité : majeure
- Confiance : CONFIRMÉ (chemin lu de bout en bout ; l'ampleur est *estimée*)
- Où : ops/llvqtune/llvqtune/trainables/row_scales.py:78-81, :107-112 ; ops/llvqtune/llvqtune/adapters/torch_model.py:3-13, :224-227 ; llvq-artifact/src/format.rs:1594-1601 ; llvq-llm/src/bin/rowscale.rs:184-188 ; ops/llvqtune/tests/test_row_norms.py:86-112
- Fait : `bin/export` passe par `decode_matrix`, qui pose la queue (les `d_in % 24` dernières colonnes, en base tournée) puis applique `unrotate_weight_rows` à toute la ligne (`format.rs:1594-1601`). La rotation est `Q = (Q_odd ⊗ H_m) D` sur tout `d_in` (`rotation.rs:26`) : elle mélange la queue dans toutes les colonnes. `RowScales.weight` multiplie les `lattice` premières colonnes DE L'EXPORT par σ et laisse passer les dernières (`row_scales.py:78-81`). `rowscale` multiplie `row_scales[i]`, donc les seuls blocs, en base tournée. Avec `B_i` l'image des blocs et `T_i` l'image de la queue (notation de `rhoapply.rs:38-44`, qui fait le calcul juste) : le run entraîne `σ·W_i[:lat] ‖ W_i[lat:]`, le fichier plié vaut `σ·B_i + T_i`. Écart : `(σ−1)·([0 ‖ W_i[lat:]] − T_i)`, nul seulement sans rotation ou sans queue. Le docstring de `torch_model.py:7-9` (« exactly `centroids[g] * row_scale * u` on the coded columns, and the verbatim tail after them ») et la phrase du README (« A per-row multiplier on the exported tensor's coded columns is a per-row multiplier on `row_scales` ») sont faux sur un enregistrement tourné. `test_the_fold_is_the_trained_function` plie en Python dans la même paramétrisation que l'entraînement, sur un modèle sans rotation : il ne peut pas voir l'écart.
- Scénario : les trois fichiers servis sont tournés et toutes leurs matrices ont une queue (4B : 16 colonnes sur 2 560 et 4 096, 8 sur 9 728 ; 8B : 16 sur 4 096 ; 14B : 8 sur 5 120 et 17 408). L'écart par ligne vaut environ `|σ−1|·√(2·tw/d_in)·‖W_i‖`, soit 11 % de l'édition voulue sur les projections 2 560 du 4B, 9 % au 8B, 3 à 6 % au 14B (*estimé*, en supposant l'énergie de la queue étalée comme celle du reste de la ligne). La KL du journal d'entraînement décrit donc un modèle qui n'est pas le fichier scellé, et σ n'est pas l'optimum de cette perte pour le fichier plié. Les gains publiés (+3,15 / +3,29 / +1,67 MMLU) restent des mesures valides des fichiers pliés : ce qui est faux, c'est le lien entre l'entraînement et ce qui est écrit. Le mode `free_params` (`free_params.py:66-70`, `__main__.py:117-120`) prend ses « queues » dans les dernières colonnes naturelles : même erreur, latente puisque son pli n'existe pas.
- Touche : outillage (instrument d'entraînement), chiffre publié par ricochet
- Correction proposée : faire écrire par `bin/export`, à côté de `llvq-int4.json`, un `llvq-tails.safetensors` : pour chaque enregistrement treillis `<nom>.tail` `[d_out, tw]` (la queue stockée) et, par rotation `(d_in, seed)`, `rot.<d_in>_<seed>.tail_rows` `[tw, d_in]` = les lignes `lat..d_in` de `Q` (obtenues par `unrotate_weight_rows` sur les vecteurs unitaires `e_lat..e_{d_in-1}`). Dans `RowScales`, remplacer la découpe par colonnes par la forme exacte `y = σ ⊙ F.linear(x, W) + (1−σ) ⊙ F.linear(F.linear(x, tail_rows), tail)` (chemin `linear`) et `W' = σ⊙W + (1−σ)⊙(tail @ tail_rows)` (chemin `weight`), sans plus aucune hypothèse sur les dernières colonnes. Réentraîner ou non les trois σ publiés est une décision d'opérateur.
- Effort : M
- Risque de la correction : moyen. Touche l'instrument d'entraînement et l'export (pas le format `.llvq`, pas l'index, aucun noyau, aucun token servi). Un réentraînement changerait les fichiers `-ft` et des chiffres publiés : opérateur.
- Vérification proposée : test Rust dans `rowscale.rs` sur la fixture existante (`rotation_seed: Some(0xABCD)`, queue de 8) : `decode_matrix` du fichier plié à σ = 1,1 sur une ligne, contre la paramétrisation Python appliquée à `decode_matrix` du fichier d'origine. Aujourd'hui l'écart vaut `0,1·‖[0‖W[lat:]] − T‖`, non nul ; après correction, égalité à 1e-12 avec `tail_rows`. Mutant : remettre `frozen[:, :lattice] * σ` ; le test redevient rouge.

### R2-API-02 σ est arrondi en bf16 dans le passage avant : la perte ne voit σ qu'au pas de 0,39 à 0,78 %
- Gravité : mineure
- Confiance : CONFIRMÉ pour le mécanisme ; PLAUSIBLE pour l'effet sur la qualité (non mesuré)
- Où : ops/llvqtune/llvqtune/trainables/row_scales.py:78, :108 ; ops/llvqtune/llvqtune/trainables/row_norms.py:98 ; ops/llvqtune/llvqtune/__main__.py:85-93, :126 ; ops/llvqtune/train.sh (`COMMON`, `--dtype bf16`)
- Fait : `train.sh` charge l'élève en bf16. σ est créé en f32 (`build` sans `dtype`), puis converti au type du poids, `sigma.unsqueeze(1).to(frozen.dtype)`, avant la multiplication. Le chemin `linear` fait de même (`.to(x.dtype)`), et `RowNorms.scale_norm` aussi pour τ. Près de 1, le pas du bf16 vaut 2^-7 = 0,0078 sur [1, 2) et 2^-8 = 0,0039 sur [0,5, 1).
- Scénario : Adam déplace σ d'environ `lr` = 3e-4 par pas ; le passage avant ne change qu'après une vingtaine de pas de même signe, quand l'arrondi bascule. L'optimisation se comporte comme un arrondi à estimateur droit sur σ, et le σ f32 exporté peut s'écarter d'un demi-pas (0,2 à 0,4 %) de la valeur que la perte a vue en dernier. Les écarts de σ à 1 se comptent probablement en pour-cent : la résolution est du même ordre que l'effet entraîné.
- Touche : outillage (instrument d'entraînement)
- Correction proposée : appliquer σ en f32 sur la sortie, dans les deux chemins : `y = (F.linear(x, frozen).float() * sigma).to(x.dtype)` (et la correction de queue de R2-API-01 de même) ; pour τ, `(y.float() * tau).to(y.dtype)`. Garder `weight` seulement pour le chemin de mesure, ou y multiplier en f32 avant l'arrondi.
- Effort : S
- Risque de la correction : faible pour le code, mais change ce qu'un réentraînement produirait : opérateur.
- Vérification proposée : test CPU sur `tiny_qwen3` en bf16 : σ = 1 + 1e-3 sur une ligne change les logits après correction, pas avant (`torch.equal` vrai aujourd'hui). Mutant : remettre `.to(x.dtype)` sur σ.

### R2-API-03 Un run plus court que son plan finit sans erreur, et `sigma.json` ne dit pas combien de pas il contient
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/llvqtune/llvqtune/domain/loop.py:121-129, :198-226, :228 ; ops/llvqtune/llvqtune/adapters/mix_corpus.py:45-56 ; ops/llvqtune/llvqtune/adapters/mmlu_corpus.py:157-167 ; ops/llvqtune/llvqtune/adapters/json_sink.py:19-31
- Fait : `check` ne refuse un corpus trop court que s'il expose `batches_available` (`loop.py:121`). `MixCorpus` ne l'expose pas et s'arrête dès qu'un côté s'épuise (`mix_corpus.py:53-55`). `MmluAuxCorpus.batches_available` est une estimation (247,9 jetons par question, mesurés une fois). Après la boucle, `run` ne compare pas `done` à `plan.steps` : seul `first is None` lève. `JsonSink` écrit `cost` et `result`, sans nombre de pas, par `write_text` non atomique, et les points de contrôle écrasent le même fichier `--out`.
- Scénario : `CORPUS=mix MIX_RATIO=0.9 STEPS=9507` à la graine 3 : le côté MMLU tient environ 8 463 lots, il en faut 8 556. Le run s'arrête vers le pas 9 400, la cosinusoïde n'est pas finie, `sigma.json` a la même forme qu'un run complet et `rowscale` le plie. C'est exactement le cas que le commentaire de `check` dit vouloir empêcher (« a run that stops at step 6,300 of 9,507 writes a plausible sigma »). Seul le `closed` du journal porte `steps_run`.
- Touche : outillage
- Correction proposée : dans `run`, après la boucle, `if done < plan.steps: raise WiringError(f"the corpus ran dry at step {done} of {plan.steps}")` après avoir écrit le journal, ou un `Outcome.short` qui fait sortir `main` en 3. Donner à `MixCorpus` un `batches_available(seed)` = `min(a/ratio, b/(1−ratio))` sur les côtés qui en ont un. Écrire `steps_run` et `steps_planned` dans le document de `JsonSink`, par fichier temporaire puis `os.replace`.
- Effort : S
- Risque de la correction : nul pour un run complet ; un run court échoue au lieu de passer.
- Vérification proposée : test de `run` avec un corpus factice qui rend 3 lots pour un plan de 5 : refus attendu ; test de `MixCorpus.batches_available`. Mutant : retirer le contrôle après la boucle.

### R2-API-04 Les paramètres de l'élève hors σ gardent `requires_grad` : leurs gradients sont calculés et s'accumulent sans jamais être remis à zéro
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/llvqtune/llvqtune/adapters/torch_model.py:49-62, :116, :124-145 ; ops/llvqtune/llvqtune/adapters/torch_optimizer.py:18-19 ; ops/llvqtune/llvqtune/__main__.py:92-96 ; ops/llvqtune/tests/test_row_norms.py:68-83
- Fait : seuls les poids des linéaires routées (copiés en tampon détaché) et, en mode `row_norms`, les normes sont gelés. `embed_tokens`, `lm_head`, `model.norm` et les normes de couche en mode `row_scales`, tous les `v_proj` (absents de `LATTICE_TYPES`) et les enregistrements int4 exclus (`o_proj`, `down_proj` des couches 12 à 23 au 4B) gardent `requires_grad=True`. `zero_grad` ne touche que les paramètres de l'optimiseur : leurs `.grad` s'additionnent pendant tout le run. Le test « nothing frozen » ne vérifie que les normes. Par ailleurs `--grad-checkpoint` ne fait rien : `from_pretrained` laisse le modèle en `eval()`, et le checkpointing de transformers n'agit qu'en `training`. Un commentaire de `ops/jobs/dclm-14b-rowscales.sh:63` le note, le drapeau reste accepté sans un mot.
- Scénario : chaque pas calcule `dW` de la tête de sortie, des `v_proj` et des int4 exclus, et garde ces gradients en mémoire : au moins 0,97 Go au 4B (plus les int4 exclus), 2,79 Go au 8B, 3,53 Go au 14B (*computed* : vocabulaire 151 936, têtes non liées au 8B et au 14B). σ n'en est pas changé : le calcul du gradient de σ ne dépend pas de ces feuilles. Le coût est en mémoire et en temps, sur un 14B que son job dit ne tenir que sur H200. Le mode `eval()` n'a pas d'effet numérique sur Qwen3, qui n'a pas de dropout actif.
- Touche : outillage
- Correction proposée : dans `__main__.main`, juste après le chargement de l'élève, `for p in student.parameters(): p.requires_grad_(False)` ; les trainables créent leurs propres feuilles. Retirer `--grad-checkpoint`, ou poser `student.train()` quand il est demandé. Étendre `test_the_gradient_reaches_tau_and_sigma_and_nothing_frozen` à tous les `named_parameters` du modèle.
- Effort : S
- Risque de la correction : faible ; σ identique au bit attendu, mémoire et temps par pas en baisse. Touche l'instrument : opérateur, avec une mesure du pas avant et après.
- Vérification proposée : le test étendu, rouge aujourd'hui sur `model.embed_tokens.weight`. Mutant : retirer la ligne de gel.

### R2-API-05 Les modes `free_params` et `low_rank` jettent ce qu'ils entraînent
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/llvqtune/llvqtune/trainables/free_params.py:76-86 ; ops/llvqtune/llvqtune/trainables/low_rank.py (`export`) ; ops/llvqtune/llvqtune/__main__.py:108-126 ; llvq-llm/src/bin/rowscale.rs:83-88
- Fait : l'export de `free_params` contient σ et `tail_shapes`, pas les valeurs des queues entraînées (94 % du jeu, d'après son docstring). Celui de `low_rank` contient `rank` et `shapes`, ni `A` ni `B`. `main` lance ces modes comme les autres, et le README les liste avec leur coût.
- Scénario : `--mode low_rank` sur une carte : le run facture ses heures et écrit un `sigma.json` sans un seul facteur. `rowscale` refuse `free_params` par nom ; aucun outil ne lit `low_rank`. Le plan du dépôt dit « refused by name rather than half-implemented ».
- Touche : outillage
- Correction proposée : refuser `free_params` et `low_rank` dans `main` hors `--dry-run` et `--repeat-one-batch`, avec un message qui nomme le pli manquant, jusqu'à ce qu'un export des valeurs (fichier safetensors à côté du JSON) et un pli existent.
- Effort : S
- Risque de la correction : nul ; aucun résultat publié ne vient de ces modes.
- Vérification proposée : `main([... "--mode", "low_rank", ...])` rend 2. Mutant : retirer le refus.

### R2-API-06 Le professeur et le tokenizer sont lus à `main` du Hub, sans révision
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/llvqtune/llvqtune/__main__.py:79-80, :138-143 ; ops/jobs/dclm-8b-rowscales.sh:66 ; ops/jobs/mmluaux-4b-rowscales.sh:74 ; ops/jobs/dclm-14b-rowscales.sh:96-97, :169-170
- Fait : `AutoConfig`, `AutoModelForCausalLM` et `AutoTokenizer` sont appelés sur `args.teacher` sans `revision`. Le tokenizer qui découpe le corpus est celui du professeur, pas celui que porte l'export. Le paquet n'a pas d'option de révision. Seul le job 14B compare la révision du Hub à celle de l'encodage, sur le Mac, avant le lancement ; les jobs 8B et `mmlu-aux` ne le font pas. PY-11 du premier tour couvrait les jeux de données, pas le professeur.
- Scénario : Qwen pousse sur `Qwen/Qwen3-8B` un nouveau `tokenizer.json` ou des poids corrigés. Un réentraînement à « même graine, même plan » distille vers un autre professeur, sur d'autres ids, et le journal ne le dit pas.
- Touche : outillage
- Correction proposée : `--teacher-revision`, obligatoire avec `--teacher`, passé aux trois `from_pretrained` et écrit dans `opened` du journal ; `TEACHER_REVISION` requis par `train.sh`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : test de `main` avec des `from_pretrained` factices qui enregistrent `revision` ; refus sans l'option.

### R2-API-07 `mmlu_aux_overlap.py` contrôle la révision du jour, pas celle sur laquelle l'entraînement lit ; il ne garantit que l'égalité exacte
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/mmlu_aux_overlap.py:25, :57-61, :64-68 ; ops/llvqtune/llvqtune/adapters/mmlu_corpus.py:62-65, :96-105
- Fait : ce que le script garantit : aucune ligne de `test`, `dev` ou `validation` n'a la même question ET les quatre mêmes choix, dans le même ordre, à la casse et aux espaces près, qu'une ligne d'`auxiliary_train`. Il télécharge sans `revision`, donc `main`, alors que son docstring dit « measured on revision c30699e » et que `mmlu_corpus.REVISION` s'appuie sur cette mesure. Un même item aux choix permutés, ou à une ponctuation près, n'est pas vu.
- Scénario : `cais/mmlu` change sur `main`. Relancé pour une nouvelle campagne, le script contrôle une autre version que celle que l'entraînement lit, et peut dire « clean » sur des données que l'entraînement ne lit pas.
- Touche : outillage (garde de contamination)
- Correction proposée : passer `revision="c30699e8356da336a370243923dbaf21066bb9fe"` aux deux téléchargements et l'imprimer ; ajouter une troisième colonne « question + choix triés » pour couvrir les permutations.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### R2-API-08 La rotation du chemin fusé de `llvq-tetra` n'est comparée à la rotation de référence par aucun test qui tourne hors Mac, et le digest du fixture `mini` n'est lu par aucun test
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-tetra/llvq_tetra/fused.py:110-134 ; llvq-tetra/tests/test_fused.py:21-24, :47-56 ; llvq-tetra/tests/test_dequant.py:93-102 ; llvq-tetra/tests/test_load.py:99-105 ; llvq-tetra/tests/fixtures/mini/llvq-dense-digest.json
- Fait : `Rotation.forward` sert les deux bras fusés, Metal et CUDA. Ses seuls tests sont `test_the_rotation_preserves_norms` (module sauté sans MPS, TST-10) et, côté hôte, `test_the_rotation_is_orthogonal_and_its_inverse_is_the_transpose`, qui ne vérifie que les normes malgré son nom. Toute application orthogonale fausse (`Q_odd` transposé, signes appliqués après au lieu d'avant, groupes permutés) les passe. Seul `test_mps.py`, sur Mac, compare des logits de bout en bout. Le fixture `mini` (deux rotations, 128 et 136 = 8 × 17, six Tetra et un int4) porte un `llvq-dense-digest.json` que seul `checkdense`, à la main, lit ; `test_load.py` ne vérifie que des poids finis et non nuls. Ce n'est pas TST-03 : ici le fichier est à jour, mais aucun test ne le consulte.
- Scénario : une édition de `Rotation.forward` remplace `"gt,...tm->...gm"` par `"tg,...tm->...gm"`. Sur Linux, `pytest` reste vert ; le premier signal est une porte sur carte.
- Touche : outillage (paquet PyPI)
- Correction proposée : un test CPU, sans `skipif`, qui vérifie `unrotate_rows(W, signs, small) @ x == W @ Rotation(signs, small, "cpu")(x)` à 1e-5 pour n = 40 et n = 136. Un test qui lit `fixtures/mini/llvq-dense-digest.json` et compare au bit le `sha256` f32 de `PackedModel(MINI).dequantize(n)` et du `weight` matérialisé par `from_pretrained`, pour les sept enregistrements.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : mutant `"gt,"` → `"tg,"` dans `fused.py:133` : rouge sur le nouveau test, vert sur la suite actuelle hors Mac.

### R2-API-09 Décision D3 non tenue sur `/v1` : en flux les avertissements ne partent jamais, et `logprobs: true` est ignoré sans un mot
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/openai.rs:227, :229-235, :320-361, :379 ; llvq-runtime/src/engine.rs:204-205 ; docs/plan-runtime.md:249-250 ; llvq-runtime/src/api.rs:805-807, :816-822, :868-873
- Fait : `Ctx.warnings` n'est lu que dans la branche sans flux (`openai.rs:379`). `HARMLESS` range `logprobs` parmi les no-op, alors que `/api/chat` avertit pour le même champ (`api.rs:805-807`). Côté Ollama, les `tool_calls` d'un message de l'historique sont désérialisés puis jetés (`api.rs:816-822`), et `system` est jeté sans avertissement sous `raw: true` (`api.rs:868-873`).
- Scénario : `POST /v1/chat/completions {"model":"qwen3:4b","messages":[...],"stream":true,"top_k":40,"frequency_penalty":0.5}` : le client reçoit ses morceaux et `[DONE]`, sans trace que deux options ont été ignorées. Le même corps avec `"stream":false` porte `warnings`. `{"logprobs":true}` donne une réponse sans `logprobs` et sans avertissement. Ce que le dépôt annonce : « Anything else a client sends is listed in the response's warnings (decision D3) » (`engine.rs:204-205`), et pour les outils « refused by name rather than half-implemented » (`plan-runtime.md:262-263`). `warnings` n'est pas un champ d'OpenAI : le mettre dans le dernier morceau ne gêne pas un client OpenAI, qui ignore les champs inconnus (à ma connaissance du SDK Python).
- Touche : runtime
- Correction proposée : dans la branche `Event::Done` du flux, poser `c["warnings"]` sur le morceau qui porte `finish_reason` quand la liste n'est pas vide. Retirer `logprobs` de `HARMLESS` et avertir quand il vaut `true`. Refuser en 400 un message de l'historique qui porte des `tool_calls` non vides, comme `refuse_unsupported` le fait pour `tools` ; avertir quand `system` est donné sous `raw: true`.
- Effort : S
- Risque de la correction : nul ; aucun token, aucun champ de `openapi.json` (les morceaux du flux n'y sont pas décrits).
- Vérification proposée : étendre `the_openai_routes_answer_as_the_ollama_ones_in_openais_shapes` : en flux avec `top_k`, un morceau porte `warnings` ; sans flux avec `logprobs: true`, `warnings` le nomme. Mutant : retirer l'insertion.

### R2-API-10 Une chaîne d'arrêt fuit son préfixe quand son caractère suivant arrive coupé en deux jetons
- Gravité : mineure
- Confiance : CONFIRMÉ (au niveau de `chat::scan`) ; le déclenchement sur un vrai texte dépend du découpage BPE
- Où : llvq-runtime/src/chat.rs:89-102 ; llvq-runtime/src/engine.rs:467-475 ; llvq-runtime/src/chat.rs:157-168
- Fait : `scan` retient un préfixe de chaîne d'arrêt en comparant la fin de `text`, U+FFFD final compris (`text.ends_with`), puis prend `safe` sur `printable(text)`, qui retire ce U+FFFD. Si le texte finit par un préfixe de la chaîne suivi du début d'un caractère multi-octet, le préfixe n'est pas reconnu, et il est envoyé. Au jeton suivant, `Stop(at)` tombe avant `sent.len()` et `take` ne peut plus rien retirer.
- Scénario : `stop: ["xé"]`, le modèle écrit « …x », puis « é » en deux jetons d'octets. Pas k : `text = "…x\u{FFFD}"`, `scan` rend `Safe` après « x », le client reçoit « x ». Pas k+1 : `Stop` à la position de « x ». La réponse contient « x », premier caractère de la chaîne d'arrêt. Ce qu'il devrait recevoir : la réponse coupée avant « x », comme pour une chaîne ASCII. Plus probable sur le chemin éager (Metal, CPU) que sous 4E, qui route 64 jetons d'un coup.
- Touche : runtime
- Correction proposée : dans `scan`, calculer `let shown = printable(text);` et faire le test des préfixes sur `shown` : `if shown.ends_with(&s[..k]) { safe = safe.min(shown.len() - k) }`.
- Effort : S
- Risque de la correction : nul ; aucun token, seul le texte retenu change dans ce cas.
- Vérification proposée : `assert_eq!(scan("ab x\u{FFFD}", &["x\u{e9}".into()]), Scan::Safe(3))` dans `chat.rs`, rouge aujourd'hui (`Safe(4)`). Mutant : remettre `text.ends_with`.

### R2-API-11 Un seul manifeste illisible casse `/api/tags`, `/v1/models`, `llvq ls` et la suppression de tout modèle, et un manifeste écrit par un `llvq` plus récent est illisible par un plus ancien
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/store.rs:191-198, :201-227, :230-237, :246-271 ; llvq-runtime/src/library.rs:49-71 ; llvq-runtime/src/api.rs:501-503, :686-690
- Fait : `list` propage la première erreur de `get` (`store.rs:220`). `Configs` est `deny_unknown_fields` (`library.rs:50`), et `Configs` a déjà gagné trois champs (`cpu`, `cuda_batch`, `cuda_eager`). `schema` n'est jamais lu. `remove` efface le manifeste, PUIS appelle `list()?` (`store.rs:259-263`) : sur une erreur, le manifeste est parti, le blob reste et la route répond 500. Le manifeste d'un modèle illisible ne s'efface pas par `llvq rm`, puisque `remove` commence par `get`. `write_manifest` passe par un temporaire au nom fixe, `<tag>.json.tmp` : deux écrivains du même nom (deux `llvq cp` vers la même cible, ou `cp` pendant un `pull`) peuvent renommer un fichier à moitié écrit.
- Scénario : l'opérateur sert avec un binaire d'avant le 2026-11-15 sur le `~/.llvq` qu'un binaire récent a rempli. Le manifeste du 4B porte `cuda_eager` : `GET /api/tags` répond 500 « unknown field `cuda_eager` », `llvq ls` échoue, `/api/chat` sur `qwen3:4b` répond 500, et `llvq rm qwen3:8b` efface le manifeste du 8B puis répond 500 en laissant son blob de 2,8 Go. Ce qu'un client devrait recevoir : la liste des modèles lisibles, comme Ollama ignore un manifeste qu'il ne lit pas (à ma connaissance, non vérifié).
- Touche : runtime
- Correction proposée : `list` saute un manifeste illisible avec un `eprintln!` qui le nomme. Dans `remove`, calculer la liste des autres manifestes avant d'effacer, et sur un manifeste propre illisible, l'effacer sans toucher aux blobs en le disant. Retirer `deny_unknown_fields` de `Configs` et garder la rigueur de `library.json` par un test qui compare ses clés à la liste connue. Nommer le temporaire `<tag>.json.<pid>.<nanos>.tmp`.
- Effort : S
- Risque de la correction : faible ; le chemin heureux ne change pas.
- Vérification proposée : test de `tests/pull.rs` : un manifeste corrompu sous un autre nom ; `list` rend le bon modèle, `remove` du bon rend `Ok(true)` et efface son blob. Mutant : remettre le `?` dans `list`.

### R2-API-12 Un `pull` lancé par HTTP ne s'arrête pas quand son client part
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/api.rs:650-657 ; llvq-runtime/src/store.rs:318-421 ; llvq-runtime/src/main.rs:159-164
- Fait : le téléchargement tourne dans `spawn_blocking` ; `report` ignore l'échec de l'envoi (`let _ = tx.blocking_send`) ; `Store::pull` n'a aucun moyen d'être annulé. `llvq pull` passe par `/api/pull` quand un serveur répond.
- Scénario : `llvq pull qwen3:14b`, puis Ctrl-C à 10 % parce que le disque est presque plein. La commande rend la main, le serveur continue à télécharger 5 Go et garde le verrou du blob. Un nouveau `llvq pull` attend « waiting for another pull of this file ». Seul l'arrêt du serveur coupe le téléchargement, et CON-09 dit que Ctrl-C sur le serveur attend justement les tâches `spawn_blocking`. Ce qu'il devrait se passer : l'arrêt du téléchargement, le fichier partiel gardé pour la reprise. Ollama annule un pull dont la requête est annulée, à ma connaissance (non vérifié dans son code).
- Touche : runtime
- Correction proposée : faire rendre à `report` un `bool` (« le client écoute encore ») et, dans la boucle de lecture de `Store::pull`, sur `false`, `out.flush()` puis `bail!("the pull was cancelled; the partial file is kept for the next pull")`. Côté CLI sans serveur, rien ne change.
- Effort : S
- Risque de la correction : faible ; change la signature interne de `report` (trois appelants).
- Vérification proposée : test de `tests/pull.rs` sur `slow_hub()` avec un `report` qui rend `false` au premier progrès : erreur en moins d'une seconde, fichier partiel présent. Mutant : ignorer le retour.

### R2-API-13 `temperature > 0` sur 4E est refusé proprement, mais seulement après le chargement du modèle
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/engine.rs:556-574, :1485-1551 ; llvq-runtime/src/library.rs:81-96 ; llvq-runtime/src/openai.rs:47-57
- Fait : la réponse est correcte sur les deux familles : 400, `{"error": "temperature 0.7: this model is served with the persistent decode (4E), which takes the argmax on the card and cannot sample. Send temperature 0, or leave it out."}` côté Ollama, la même phrase en `invalid_request_error` côté OpenAI, conforme à D2 (`plan-runtime.md:244-248`). Mais le refus vient de `prepare`, sur le fil du moteur, une fois le modèle chargé : `Models::submit` lance le chargement sans rien regarder de la requête.
- Scénario : premier `POST /v1/chat/completions {"model":"qwen3:4b","temperature":0.7,...}` sur CUDA, cache de modules froid : le client attend le chargement, 509 s à froid pour la config chaîne (`library.rs:82-84`), puis reçoit le 400. Le modèle reste chargé pour le keep-alive. Côté OpenAI, `param` vaut `null` alors qu'OpenAI y nomme le champ fautif (`"temperature"`).
- Touche : runtime
- Correction proposée : ajouter à `Loader` une méthode `greedy_only(&Manifest) -> Option<bool>` (défaut `None`), que `DeviceLoader` lit dans la config que `config_for` choisirait (champ `decode` du fichier, ou `LLVQ_BATCH`). `Models::submit` refuse en 400 avant `Engine::spawn` quand elle répond `Some(true)` et que la température demandée est positive. Poser `param: "temperature"` dans `oai_error` pour ce refus.
- Effort : M
- Risque de la correction : faible ; la règle reste celle de `prepare`, qui la garde aussi.
- Vérification proposée : test de `tests/engine_busy.rs` avec un `Loader` compté qui répond `Some(true)` : `submit` d'une requête à 0,7 rend 400 et le compteur de chargements reste à 0. Mutant : retirer le contrôle.

### R2-API-14 Sous `think: true`, `thinking` commence et finit par un saut de ligne et `content` commence par deux
- Gravité : cosmétique
- Confiance : CONFIRMÉ pour ce que le serveur envoie ; PLAUSIBLE pour l'écart à Ollama
- Où : llvq-runtime/src/engine.rs:458-466 ; llvq-runtime/src/main.rs:350-363
- Fait : `Router` route par jeton : ce qui suit `<think>` (le `\n` que Qwen3 écrit) va dans `thinking`, et le jeton 271 `\n\n` qui suit `</think>` va dans `content`.
- Scénario : `/api/chat` avec `think: true` : `message.thinking` vaut « \nOkay, … \n » et `message.content` « \n\nThe answer… ». `llvq run --think` affiche deux lignes vides de plus. À ma connaissance, l'analyseur de raisonnement d'Ollama mange ces blancs ; je n'en suis pas certain.
- Touche : runtime
- Correction proposée : retirer les sauts de ligne en tête de `content` et en tête et en queue de `thinking` au moment du routage, et garder dans le memo le texte tel qu'envoyé, pour que `render` retrouve le segment quand le client renvoie la réponse.
- Effort : S
- Risque de la correction : faible ; aucun id ne change, mais le memo doit suivre le texte émis.
- Vérification proposée : test de `Router` avec `think` : `content` ne commence pas par `\n`.

### R2-API-15 Deux messages d'erreur OpenAI imprécis
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/openai.rs:209-216, :54-55
- Fait : une partie `{"type":"text"}` sans champ `text` tombe dans le bras `(kind, _)` et reçoit « content part "text": the served models read text parts only ». `param` est toujours `null`.
- Scénario : `{"messages":[{"role":"user","content":[{"type":"text","txt":"hi"}]}]}` : le client lit qu'une partie texte n'est pas une partie texte. Il devrait lire « content part text: no `text` field ».
- Touche : runtime
- Correction proposée : un bras `(Some("text"), None) => Err("content part text: no text field")` ; `param` renseigné quand un champ est nommé.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### R2-API-16 llvqtune sort en 1 sur un critère que son propre docstring dit vide de sens hors `RepeatCorpus`, et `train.sh` s'arrête là
- Gravité : cosmétique
- Confiance : PLAUSIBLE (sur DCLM la KL baisse assez pour que `improved` soit vrai)
- Où : ops/llvqtune/llvqtune/__main__.py:212 ; ops/llvqtune/llvqtune/domain/loop.py:81-96 ; ops/llvqtune/train.sh (appel d'entraînement sous `set -e`)
- Fait : `main` rend `0 if outcome.improved else 1`. `improved` compare le premier et le dernier tiers des pertes ; son docstring dit que sur un corpus en flux « this says nothing ». L'appel d'entraînement de `train.sh` n'est pas protégé par `set +e`.
- Scénario : un run sur `mmlu-aux` dont le dernier tiers lit des questions plus dures : sortie 1 après l'écriture de `sigma.json`, le job est marqué en échec, le résumé (`date`, `ls`) n'est pas écrit.
- Touche : outillage
- Correction proposée : rendre 0 dès que le run est complet (R2-API-03 pour le cas court), et n'utiliser `improved` que sous `--repeat-one-batch`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### R2-API-17 `check.py` combine deux écarts par `max`, qui cache un NaN
- Gravité : cosmétique
- Confiance : CONFIRMÉ (sémantique Python : `max(0.0, nan)` rend `0.0`)
- Où : ops/mlx_tetra/check.py:86-87
- Fait : `e = max(rel(qf, …), rel(kf, …))` puis `hold(…, exact=True)`. Si q est au bit et k vaut NaN, `max` rend 0.0 et la ligne dit « same bits ok ».
- Scénario : un défaut du dispatch empilé qui écrit des NaN dans les lignes de k seulement passe la section A ; la section B le prendrait sur ses propres lignes, si la couche y est.
- Touche : outillage
- Correction proposée : deux `t.hold` séparés, un pour q et un pour k.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

## Connus non corrigés

- CONNU-NON-CORRIGÉ : `num_predict: 0`, ou négatif, et `max_tokens: 0` veulent dire « jusqu'à la fin du contexte » au lieu d'un 400 (`engine.rs:629-632`). Réf. `docs/archive/audit-bugs-2026-11-08.md` §4, relevé aussi dans `11-concurrence.md`.
- CONNU-NON-CORRIGÉ : un `<|im_end|>` écrit dans un message ferme son tour (`chatfmt.rs:86`, `encode` reconnaît les jetons ajoutés). Réf. `docs/runtime-audit-2026-11.md`, « Found and not fixed ». Même chose pour `<think>` et `</think>` dans un message.
- CONNU-NON-CORRIGÉ : un flux fini sans `done` est lu comme une réponse complète côté client (CON-06, « Laissé de côté » de `40-lot2-runtime.md`). Le même défaut est dans la page de chat : `chat.html:811`, `if (!t.finish) t.finish = "stop";`, qui verse la réponse tronquée dans l'historique.

## Vérifié sain

- Les compteurs. `prompt_eval_count` est tout le prompt et `prompt_eval_cached_count` la part reprise, comme la fixture d'Ollama les définit (« Number of input tokens in the prompt », « Number of prompt tokens read from the cache ») ; `prompt_eval_duration` ne compte que la part relue. `eval_count` = `said.len()` vaut exactement `num_predict` quand il coupe, sur les trois chemins. Il n'inclut jamais le jeton d'arrêt. Sous 4E, il compte les jetons calculés après une chaîne d'arrêt, jusqu'à 63, ce qui est le calcul réel. Sur le chemin batch, `cached` vaut 0, ce qui est juste : chaque requête repart d'une session neuve.
- `done_reason` vaut `stop` ou `length`, `load` et `unload` pour les requêtes vides ; `finish_reason` en est la projection. Le flux OpenAI ouvre par `{"role":"assistant","content":""}` et finit par le morceau `finish_reason`, `usage` si demandé, puis `[DONE]`, y compris après une erreur.
- `chat::scan` hors du cas R2-API-10 : une chaîne à cheval sur deux morceaux est retenue, un caractère coupé est retenu, `take` ne coupe jamais dans un caractère, et le texte retenu sort par `tail` quand l'arrêt vient d'un jeton d'arrêt ou de `num_predict`.
- Le gabarit : système, `developer` lu comme système, messages vides, rôle `tool` refusé par nom, images et parties non texte refusées par nom des deux côtés, `format` et `response_format` non texte refusés. Le rendu par texte d'un message assistant suit celui du gabarit HF de Qwen3 pour l'historique (aucun `lstrip` hors présence de `</think>`).
- Le store : disque plein pendant un `pull` laisse un partiel qui est un préfixe exact, repris par `Range` puis vérifié ; un serveur qui ignore `Range` fait repartir de zéro ; un partiel plus long que la taille épinglée est effacé ; blob sans manifeste, le `pull` suivant le revérifie et l'enregistre ; manifeste sans blob, le `pull` suivant le retélécharge. Deux serveurs sur un store se partagent le verrou par blob.
- Le pli des σ : l'ordre des lignes est le même de l'export à l'enregistrement (`decode_matrix` ligne `i`, `row_scales[i]`), σ passe en f64 sans perte par JSON, et un σ de 1,0 exact laisse le fichier identique à l'octet. Le défaut est la base, pas l'ordre (R2-API-01).
- La perte : KL sommée sur le vocabulaire puis moyennée sur lot et positions, entropie croisée décalée d'un jeton ; le professeur est en `eval()` sous `no_grad` ; aucune graine torch n'est nécessaire (σ part de 1, `low_rank` a son générateur à graine fixe). MMLU `dev`, `validation` et `test` sont refusés par nom dans `fetch_split`.
- `ppl.py` : la portée est gardée par son sha256 avant toute fenêtre, la référence Rust est lue à neuf décimales, et la fenêtre de préremplissage ne nourrit jamais le jeton qu'elle prédit. `mutate.py` compte les sites de chaque mutant et travaille sur une copie.

## Candidats au lot 1

R2-API-03, R2-API-05, R2-API-07, R2-API-08, R2-API-09, R2-API-10, R2-API-11, R2-API-15, R2-API-16, R2-API-17.

Hors lot 1 : R2-API-01, R2-API-02 et R2-API-04 touchent l'instrument d'entraînement ; R2-API-06 change l'interface de `train.sh` que des jobs tamponnés appellent ; R2-API-12 et R2-API-13 changent des signatures internes et demandent un compilateur ; R2-API-14 change le memo.
