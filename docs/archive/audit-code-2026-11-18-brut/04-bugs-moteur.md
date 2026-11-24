# Bugs de logique, moteur hôte : constats
Périmètre réellement lu : en entier `llvq-llm/src/served.rs`, `served_decode.rs`, `session.rs`, `repl.rs`, `chatfmt.rs`, `kvq.rs`, `gsm8k.rs`, `eval.rs`, `corpus.rs`, `fused_cpu.rs`, `embedquant.rs`, `attn_cuda.rs`, `bin/gsm8k.rs`, `bin/gsm8kpair.rs`, `bin/mmlu.rs`, `bin/ppl.rs`, `bin/seal.rs`, `bin/chat.rs` (main) ; `model.rs` lignes 240-1010 (KV cache, stores, formes d'attention, softmax en place, chemin noyau) et 1120-1200, 1500-1680, 1872-2402, 2546-3820 (bloc, forward, préremplissage, décode, mesures) ; `sealed.rs` 100-622 ; `mmlupair.rs` 1-785 ; `hfpack.rs` 80-540 ; `guided.rs` 1-475 ; `calib.rs` 36-225 et 1043-1342 ; `bin/smoke.rs` 560-1120 ; `fused_cuda.rs` 3790-4184 (hôte de `load_resolved`) ; `fused.rs` 196-365 ; `tetra_cpu.rs` 160-330.
Survolé : `persist4e_cuda.rs` (seulement `launch`, `prepare`, `chain_prefill`, `tree_launch` pour l'interface avec les caches), `kvcapture.rs`, la ligne de commit de `kvladder` dans `model.rs`, `loader.rs` (plan), `bin/calibtokens.rs`, `llvq-runtime/src/library.rs` (configs CPU).
Non couvert : le reste de `fused.rs` (transcodage, `load_with`), de `fused_cuda.rs` et de `calib.rs` (boucle GPTQ par matrice), `fused_metal.rs`, `tetra_cpu.rs` au-delà de la ligne 330, `speculative.rs`, `dflash*.rs`, `batch.rs`, `kvladder.rs`, `rotplan.rs`, `artifact2.rs`, et les bancs `fusedrun`, `prefillbench`, `specbench`, `vtreebench`.
Bilan : 0 critiques, 0 majeures, 6 mineures, 4 cosmétiques (plus 2 constats d'avant encore présents, en une ligne)

Aucun défaut trouvé ne change les tokens servis. Les six mineures touchent la lecture des chiffres : règle d'égalité de MMLU, traçabilité des arms, harnais qui acceptent une faute de frappe.

### BUG-01 MMLU départage une égalité de logits vers la dernière lettre
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/bin/mmlu.rs:818-820`
- Fait : `pick = (0..4).max_by(|&x, &y| options[x].total_cmp(&options[y]))`. `Iterator::max_by` rend le DERNIER maximum en cas d'égalité, donc D avant C avant B avant A. `lm-eval-harness` et tout `argmax` numpy ou torch rendent le premier. Les quatre logits sortent en f16 sur les deux bras (le modèle tourne en F16), et deux lettres tombent souvent dans le même ulp.
- Scénario : sur les 36 dumps `*FULL*.csv` de `docs/data/mmlu-dumps/`, 11 à 96 questions sur 14 042 ont un maximum à égalité. Rescorées avec la règle « premier », les micro moyennes bougent de −0,114 à +0,057 pp (*computed* depuis les logits des dumps). Le 4B publié (`mmlu-4b-abl-control-FULL.csv`) lit 63,374 en « dernier » et 63,381 en « premier ». Le bras quantifié a deux à trois fois plus d'égalités que le f16 (40 contre 19 au 4B), donc la règle ne pèse pas pareil sur les deux bras d'une paire.
- Touche : banc (chiffre MMLU publié, sous la barre)
- Correction proposée : crédit fractionnaire 1/k sur une égalité à k lettres, ou règle « premier » documentée. Les dumps portent les quatre logits : on rescore sans relancer.
- Effort : S
- Risque de la correction : faible. Elle déplace des chiffres publiés de quelques centièmes de point ; à faire avec la ligne d'HISTORIQUE qui le dit.
- Vérification proposée : le script de rescoring ci-dessus, versé dans `ops/`, sur les dumps de la Table publiée.

### BUG-02 `bin/mmlu` sous une config `prefill: chain` ou `prefill_attention: kernel` score le préremplissage groupé, et le journal annonce l'autre
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/bin/mmlu.rs:548`, `:580`, `:799-804`, `:733-743` ; `llvq-llm/src/model.rs:2118-2128`, `:2220-2245`
- Fait : le bras servi appelle `cfg.apply` (jamais de décodeur), puis `model.hidden(&input)`. `Block::forward` bâtit un `KvCache::new(self.kv)`, donc un store `Cat`. Dans la branche `Kernel` d'`attention_ctx`, `grow` vaut alors `false` : ni `kernel_rows` ni `kernel_prefill` ne tournent, tout passe par `grouped_attention`. La chaîne (passe d'arbre de 4E) ne tourne pas non plus. Pourtant `cfg.provenance()` imprime « prefill chain (tree pass ...) » ou « prefill attention kernel », et l'en-tête du dump n'écrit ni `prefill` ni `prefill_attention`.
- Scénario : `LLVQ_CONFIG=configs/qwen3-4b-tetra-e4-4e-v9-chain.json mmlu qwen3-4b-sealed.bin cuda`. Le 4B est servi par cette config depuis le 2026-11-15, et elle n'est pas au bit avec le préremplissage eager (`served.rs:386-388`). Le score imprimé est celui du préremplissage groupé, sous une ligne de provenance qui dit chaîne. Même chose pour les configs `*-kprefill.json`, qui n'existent que pour être portées par une porte de mesure. Aucun chiffre publié n'en vient : ETAT dit que MMLU a tourné sur la reconstruction dense.
- Touche : banc
- Correction proposée : refuser dans `bin/mmlu` une config dont `prefill_form != Eager` ou `prefill != Grouped`, en le nommant ; ou faire passer le bras servi par `Decoder::prefill_fresh` (le dernier `hidden` suffit). Écrire `prefill` et `prefill_attention` dans l'en-tête du dump.
- Effort : S (refus) ou M (chemin servi)
- Risque de la correction : nul pour le refus. Aucun chiffre publié ne bouge.
- Vérification proposée : test hôte sur le modèle jouet `tiny_model_kernel` : une config `prefill_attention: kernel` donne, par `hidden()`, les mêmes bits qu'une config sans le champ (ce qui prouve le défaut) ; puis le refus.

### BUG-03 Deux dumps GSM8K de préremplissages différents s'impriment identiques dans `gsm8kpair`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/gsm8k.rs:343-365` ; `llvq-llm/src/bin/gsm8kpair.rs:38-63`, `:71-79`
- Fait : `served_header` n'écrit pas `prefill_attention`. `gsm8kpair::describe` n'imprime que `attention` et `decode` (avec `rows`), jamais `config`, `prefill` ni `decode_cache`, alors que l'en-tête de `bin/gsm8k` porte les trois. Aucun contrôle ni avertissement sur `max_new` : seul `qhash` est vérifié, et il ne voit que le prompt.
- Scénario : apparier un dump de `qwen3-4b-tetra-e4-4e-v9.json` et un dump de `...-v9-chain.json` (ou `...-kprefill.json`). Les deux lignes `A =` et `B =` lisent « attention kernel | decode persistent rows v9 » : rien ne dit quel bras a tourné la chaîne. Apparier un dump à `max_new` 512 et un à 1024 donne un écart et un p de McNemar sans un mot, alors que le plafond tronque des réponses (`stop = cap`).
- Touche : banc
- Correction proposée : ajouter `prefill_attention` à `served_header` quand il vaut `kernel` ; imprimer `config`, `prefill`, `prefill_attention`, `decode_cache` dans `describe` ; avertir, comme `mmlupair` le fait pour le dtype, quand `max_new`, `dtype` ou `kv` diffèrent.
- Effort : S
- Risque de la correction : nul. Un en-tête gagne une clé optionnelle, les dumps anciens se lisent comme avant.
- Vérification proposée : test pur dans `gsm8k.rs` : `served_header` d'une config kprefill contient `prefill_attention`.

### BUG-04 La fenêtre de couches d'une restauration disparaît du libellé et de l'en-tête des dumps
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/sealed.rs:265-272`, `:440-451` ; `llvq-llm/src/bin/mmlu.rs:617-621`, `:716`
- Fait : `RestoreF16::describe` et `describe_full` joignent les seuls noms de types. `ranges` (la fenêtre `@lo-hi`) n'est imprimée nulle part. `restore_note` ajoute le nombre de matrices, qui ne sépare pas deux fenêtres de même largeur.
- Scénario : `LLVQ_RESTORE_F16=down_proj@0-11` et `LLVQ_RESTORE_F16=down_proj@24-35` écrivent tous deux « restored down_proj at f16 (12 matrices, ...) » dans `# model=` et sur la ligne de résultat. `mmlupair` les imprime sous le même libellé : deux arms d'une campagne d'attribution par profondeur ne se distinguent que par le nom du fichier.
- Touche : banc
- Correction proposée : faire écrire à `describe` `down_proj@0-11` quand une fenêtre existe.
- Effort : S
- Risque de la correction : nul. Seul le texte des libellés futurs change.
- Vérification proposée : test unitaire : `RestoreF16::parse("down_proj@0-11").describe()` contient `@0-11`.

### BUG-05 `bin/seal` prend `Qwen/Qwen3-0.6B` par défaut et ne compare jamais le modèle à celui de l'encodage
- Gravité : mineure
- Confiance : CONFIRMÉ (le chemin) ; l'erreur d'opérateur qui le déclenche est PLAUSIBLE
- Où : `llvq-llm/src/bin/seal.rs:53`, `:92-121` ; `llvq-llm/src/artifact2.rs:551` (le `.state` porte `model`)
- Fait : `LLVQ_MODEL` absent vaut le 0.6B. Le sceau copie les tenseurs non quantifiés, `config.json` et `tokenizer.json` du checkpoint nommé, sans lire le `model` que `bin/smoke` a écrit dans `<artifact>.state`, et sans vérifier qu'un nom quantifié existe dans le checkpoint avec les mêmes dimensions. `bin/mmlu` refuse ce même défaut pour les restaurations (« required, never defaulted, because the default would be the 0.6B ») ; `bin/seal` le garde.
- Scénario : sceller un 4B en oubliant `LLVQ_MODEL` donne un fichier config 0.6B + matrices 4B, refusé seulement au chargement (forme). Plus grave, sceller avec un checkpoint de mêmes formes mais d'autres poids (Qwen3-4B contre une variante 2507 ou Base) produit un fichier qui charge et tourne : normes, embedding et tokenizer de l'un, projections de l'autre, et aucun message.
- Touche : outillage (format des fichiers publiés)
- Correction proposée : refuser `LLVQ_MODEL` absent ; lire `<in>.state` quand il existe et refuser un `model` différent ; refuser une matrice dont le nom manque au checkpoint ou dont les dimensions diffèrent.
- Effort : S
- Risque de la correction : nul. Les fichiers déjà scellés ne bougent pas.
- Vérification proposée : un test qui scelle un artefact jouet avec un `.state` nommant un autre modèle et attend le refus.

### BUG-06 `bin/mmlu` et `bin/ppl` lisent leurs positionnels avec `.parse().ok().unwrap_or(défaut)`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/bin/mmlu.rs:495` ; `llvq-llm/src/bin/ppl.rs:40-41`, `:48`, `:120`
- Fait : une valeur illisible devient le défaut sans message. `bin/smoke` a corrigé exactement ce motif (`arg()`, `a_malformed_positional_is_refused_not_defaulted`), `bin/gsm8k` refuse aussi ; ces deux-là non. `limit = 0` passe aussi : chaque sujet est scoré sur 0 question.
- Scénario : `mmlu sealed.bin cuda 4O` (lettre O) lance un recensement de 14 042 questions au lieu de 2 280, soit des heures de L40S payées. `mmlu ... 0` imprime « MMLU = 0.00 % » et des `NaN %` par sujet. `ppl 4096x 12 ...` mesure à 2 048 ; un corpus mal tapé (`c4 ` ou `C4`) retombe sur WikiText-2.
- Touche : banc (coût)
- Correction proposée : la fonction `arg()` de `bin/smoke` partagée dans `eval.rs` ; refuser `limit = 0` ; refuser un corpus autre que `wikitext2` ou `c4`.
- Effort : S
- Risque de la correction : nul. Une commande valide fait la même chose.
- Vérification proposée : test unitaire de la fonction partagée sur `"4O"`, `"0"`, `""`.

### BUG-07 `bin/chat` lit la graine en f32
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/bin/chat.rs:51-58` et la construction du `Sampler` (`env_f32("LLVQ_CHAT_SEED", 0.0)? as u64`)
- Fait : la graine passe par un f32 (24 bits de mantisse) puis `as u64`. `LLVQ_CHAT_MAX_NEW` suit le même chemin (`-5` devient 0, `inf` devient `usize::MAX`).
- Scénario : `LLVQ_CHAT_SEED=123456789` et `123456790` donnent le même tirage. Une graine négative vaut 0.
- Touche : outillage (REPL seulement ; le runtime prend sa graine en u64, `engine.rs:674`)
- Correction proposée : parser la graine en `u64` et `MAX_NEW` en `usize`, refuser le reste.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### BUG-08 Le commentaire du préremplissage de `bin/gsm8k` est faux sous la forme noyau
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/bin/gsm8k.rs:302-306`, `:318` ; `llvq-llm/src/model.rs:2227-2245` ; `llvq-llm/src/session.rs:159`
- Fait : le harnais découpe le prompt par `MAX_ROWS` (256) et dit qu'« a chunk boundary changes nothing the cache does not already carry ». Sous `attention: kernel`, un morceau de 1 à 8 lignes part au noyau d'attention, un morceau plus long à la forme groupée, qui n'est pas au bit avec lui. `Session::feed` (le runtime) découpe par `prefill_rows()`, 4 096 sur CUDA. Un prompt de 257 à 264 jetons n'aurait donc pas le même cache dans le harnais et dans le runtime.
- Scénario : aucun aujourd'hui : le plus long prompt GSM8K du test fait 215 jetons (*computed* sur les 1 319 lignes de `gsm8k-8b-served.jsonl` et des autres dumps). Le commentaire mordrait le jour où le harnais servira un prompt plus long (raisonnement activé avec un système, autre jeu).
- Touche : banc
- Correction proposée : découper par `model.prefill_rows()`, comme `Session::feed`, et corriger le commentaire.
- Effort : S
- Risque de la correction : nul pour GSM8K (aucun prompt ne dépasse 256).
- Vérification proposée : aucune, lecture suffisante.

### BUG-09 `mmlu_split` lit une réponse texte vide comme « A »
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/corpus.rs:324-326`
- Fait : `"ABCD".find(s.trim())` rend `Some(0)` pour `""` et accepte des sous-chaînes (`"AB"` vaut 0, `"BC"` vaut 1).
- Scénario : un miroir de MMLU à réponses texte avec un champ vide marque la question « A » sans refus. `cais/mmlu` stocke un entier et ne passe pas par cette branche.
- Touche : banc
- Correction proposée : `match s.trim() { "A" => 0, "B" => 1, "C" => 2, "D" => 3, _ => refus }`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### BUG-10 La ligne de résultat de `bin/mmlu` imprime le `kv` de l'environnement, pas celui de la config
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/bin/mmlu.rs:889-894`
- Fait : `[kv {}]` lit `kv_mode` (`LLVQ_KV`), alors que le bras servi charge `cfg.kv` (`:563`). L'en-tête du dump écrit bien `cfg.kv` (`:741`).
- Scénario : une config `"kv": "q8"` sans `LLVQ_KV` dans le shell imprime `[kv f16]` sur la ligne MMLU. Aucune config livrée ne nomme q8, donc sans effet aujourd'hui.
- Touche : banc
- Correction proposée : imprimer `cfg.kv` sur le bras servi.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

## Constats d'audits antérieurs encore présents

- CONNU-NON-CORRIGÉ (`docs/archive/audit-bugs-2026-11-08.md` §4, `session.rs:96-106`) : `Session::feed` (`session.rs:155-166`) étend `ids` après un `hidden_cached` réussi seulement ; une erreur en milieu de couche laisse les premières couches en avance. De plus `Session::truncate` sort tôt quand `n == ids.len()` (`session.rs:129-131`), donc `reuse_prefix` ne répare pas ce cas sur un nouvel essai du même prompt ; 4E le refuse à `prepare` (`persist4e_cuda.rs:2448`), l'eager échoue sur la forme du masque ou lit des positions périmées au décode.
- CONNU-NON-CORRIGÉ (même audit, §1.6 forme CPU et §4 « CPU servi ») : le runtime CPU sert `configs/qwen3-{4b,8b,14b}-tetra-e4.json`, sans champ `attention`, donc la forme étendue sur le store `Cat` (copie de l'historique et `repeat_kv` à chaque couche) ; `prefill_rows()` vaut 4 096 sur CPU (`fused_cpu.rs:28`, `model.rs:2719-2721`), et un seul tenseur de scores f32 pèse encore 4,3 Go à 8 192 positions. Les trois tenseurs du chemin GPU sont corrigés (`softmax_in_place`).

## Candidats au lot 1

BUG-03, BUG-04, BUG-05, BUG-06, BUG-07, BUG-09, BUG-10. BUG-02 sous sa forme refus (pas sous sa forme chemin servi). BUG-01 est d'effort S mais déplace des chiffres publiés : pas au lot 1.
