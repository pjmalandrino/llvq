# Protocole de mesure — LLVQ 2 bits contre MLX q4 contre FP16, Qwen3‑4B

**Statut : plan. Aucun chiffre de résultat ci‑dessous n'est une mesure de cette campagne.** Les valeurs citées sont soit des octets relevés sur disque, soit des mesures antérieures explicitement datées et sourcées, soit des coûts extrapolés de traces existantes. Machine : MacBook Pro M3 Max, 16 cœurs CPU / 40 cœurs GPU, `hw.memsize` = 68 719 476 736 o (68,72 Go décimaux), `max_recommended_working_set_size` = 55 662 788 608 o (mesuré). 66 GiB libres au disque, capacité 93 % — c'est le facteur limitant du plan.

---

## 0. Ce que ce plan mesure, et ce qu'il ne mesure pas

Les trois bras ne vivent pas dans le même moteur. Ce fait est irréductible : `bin/run` n'a pas de cache KV et son noyau fusé n'a aucun appelant ; MLX ne sait pas lire un `.llvq`. Aucune astuce statistique ne répare un écart de moteur. Deux sorties seulement sont honnêtes, et le plan les applique axe par axe :

- **Quand on peut ramener tous les bras dans un moteur unique sans dénaturer la grandeur, on le fait.** C'est le cas de la qualité : `bin/ppl` accepte déjà un overlay `.safetensors`, donc les poids MLX sont scorés dans *notre* passe avant, avec *notre* tokenizer, sur *les mêmes* tokens. Seuls les 252 tenseurs de projection changent. C'est une mesure de FORMAT.
- **Quand on ne peut pas, on publie des rapports internes à chaque moteur**, avec le FP16 comme dénominateur commun présent des deux côtés. C'est le cas de la vitesse.

| Axe | Objet mesuré | Dispositif |
|---|---|---|
| 1. Espace à froid | **FORMAT** | `stat` sur les octets livrés, périmètre nommé |
| 2. Mémoire | **FORMAT** (poids résidents, calculé) **+ MOTEUR** (pic processus, mesuré) — deux lignes, jamais soustraites | `/usr/bin/time -l` autour des runs déjà programmés |
| 3a. Vitesse noyau | **FORMAT**, chaque format avec le noyau de ses auteurs, publié en Go/s atteints et en rapport intra‑moteur | `bin/thesis` + banc MLX sur les mêmes 252 formes |
| 3b. Vitesse bout en bout | **MOTEUR** | `mlx_lm.generate` B contre C ; le bras A s'abstient et l'abstention est un résultat |
| 4. Perplexité | **FORMAT** | `bin/ppl`, un seul moteur, empreinte de tokens identique |
| 5. MMLU | **FORMAT** | `bin/mmlu`, un seul moteur, mêmes questions, test apparié |

**Chaque table du papier porte cette étiquette dans son en‑tête.** Une table sans étiquette sera lue comme une claim de format et détruite en relecture.

**Trois objets s'appellent « FP16 » et ne doivent jamais être confondus** : *FP16‑checkpoint* (les 3 shards bf16 chargés bf16→f16 par notre harnais), *FP16‑MLX* (le même checkpoint converti et exécuté par `mlx_lm`), *FP16‑référence‑noyau* (le bras témoin de `bin/thesis`, qui est l'arrondi f16 de la reconstruction f64 des blocs LLVQ dans la base tournée — un témoin de **coût**, pas de qualité).

**Le mot « VRAM » est banni du papier.** Le M3 Max n'a pas de mémoire graphique séparée. La question « ça rentre » se traduit en « pic de footprint < 55,66 Go (working set recommandé, mesuré) ».

---

## 1. Les trois bras

### 1.1 Définition exacte

| Bras | Objet | Octets | Statut |
|---|---|---|---|
| **A1** | `/Users/pjmalandrino/qwen3-4b-llvq.bin` — LLVQ 2 bits, `leech1c12`, embedding f16, scellé auto‑suffisant (config + tokenizer dans le fichier) | 1 770 527 533 | existe, publié, sha256 `9db213ef…c84b0` |
| **A2** | `/Users/pjmalandrino/q4b-e4.llvq` — mêmes projections, embedding int4 g64 | 1 211 403 653 | existe, **jamais scoré**, magic LVQ3 lu par l'arbre de travail |
| **B** | famille MLX affine par groupes, **5 réglages** (voir 1.2) | 2 263 022 417 pour 4b‑g64 | q4‑g64 existe ; les 4 autres à produire |
| **C** | checkpoint `Qwen/Qwen3-4B`, bf16→f16 | 8 044 936 192 (poids) / 8 056 438 199 (snapshot HF) | en cache local, 7,5 Go |

Dénominateurs, à nommer sur chaque ligne où un bits/paramètre apparaît :
- **4 022 468 096** poids du modèle entier ;
- **3 633 315 840** poids de projection (queue comprise) — le dénominateur *homogène* ;
- **3 616 358 400** poids réellement quantifiés (queue `KeepExact` exclue : 16 957 440 poids, 0,4667 %).

Le chiffre à afficher pour A1 est **2,1595 b/poids** (dénominateur projections). Le 2,1696 est le conservateur (poids quantifiés seuls). **Le 2,0702 ne doit apparaître nulle part pour ce fichier** : il décrit la comptabilité idéale d'un payload, pas les octets écrits.

### 1.2 Le bras B est une courbe, pas un point

MLX quantifie à 2/3/4/5/6/8 bits × group_size arbitraire (vérifié exécutable sur les formes réelles, mlx 0.30.3). Un point unique ne produit qu'un verdict binaire, très probablement perdu (q4‑g64 est du RTN affine **sans calibration**, degradation typique de quelques pourcents, contre nos +38,5 % mesurés). Une courbe répond à la vraie question : *à débit égal, que vaut le réseau de Leech contre la quantification affine par groupes ?*

| Réglage | b/poids (projections) | Statut |
|---|---|---|
| 2 bits, g128 | 2,250 | à produire |
| 2 bits, g64 | 2,500 | à produire |
| 3 bits, g64 | 3,500 | à produire |
| 4 bits, g128 | 4,250 | à produire |
| 4 bits, g64 | **4,500** | livré, `~/qwen3-4b-mlx-q4/` |
| 8 bits, g64 | 8,500 | **contrôle**, pas un bras (voir 2.4.3) |

Notre point A1 est à 2,1595 — strictement à gauche des cinq.

**Production des 4 réglages manquants : ne pas passer par `mlx_lm.convert`.** `nn.quantize` appelle `mx.quantize` sur le poids ; on obtient exactement la même chose en appelant `mx.quantize` directement sur les tenseurs bf16 du checkpoint, sans jamais écrire de répertoire MLX. Économie : ~20‑25 min machine et ~6,8 Go de disque, et une dépendance de moins sur le comportement de `convert`. Le répertoire q4‑g64 livré sert alors de **contrôle de reproduction** : le chemin direct doit le reproduire tenseur pour tenseur (la quantification MLX est un RTN déterministe).

### 1.3 Le contrôle d'iso‑périmètre : l'embedding

MLX quantifie **253** tenseurs (252 projections + `model.embed_tokens`). Nous en quantifions **252** et laissons l'embedding en f16. Comme `tie_word_embeddings = true`, cet embedding **est** le `lm_head` : la différence entre directement dans les logits. Le confondant joue **en sens opposé sur le disque et sur la qualité** — il nous dessert sur le disque, il nous flatte sur la qualité. Le neutraliser une fois ferme les deux attaques.

**Table disque en 2×2, jamais en ligne :**

| | embedding f16 | embedding int4‑g64 |
|---|---|---|
| **projections LLVQ 2 bits** | 1 770 527 533 o (A1, **mesuré**) | 1 211 403 653 o (A2, **mesuré**) |
| **projections MLX q4‑g64** | 2 822 044 672 o (**CALCULÉ**, aucun fichier) | 2 263 022 417 o (B livré, **mesuré**) |

Ratios : diagonale produit‑contre‑produit ×1,2782 ; à embedding f16 apparié ×1,5939 ; à embedding int4 apparié ×1,8681 ; sur les projections seules 2,1595 contre 4,5000 = ×2,0838. **La cellule calculée est étiquetée CALCULÉ** — publier un fichier qui n'existe pas est exactement l'erreur de comptabilité de 2026‑07‑31.

**Sur la qualité, même dédoublement :**
- **B‑proj** = q4 sur les 252 projections, embedding f16 → iso‑périmètre avec A1. C'est le bras de la courbe.
- **B‑full** = q4 sur 253 tenseurs → produit réel de MLX, à comparer à A2.

L'écart B‑full − B‑proj *est* la mesure de ce que coûte la quantification de l'embedding, et il répond du même coup à la question ouverte sur A2, qui n'a aucune mesure de qualité.

**B‑full exige du code** (ticket T8) : les champs `embed` et `head` de `Qwen3` sont privés, et `head` est un `clone()` de l'embedding figé à la construction (`model.rs:302-306`). Écraser l'embedding après `Qwen3::new` laisserait le `lm_head` en f16 — le bras mesurerait en silence un modèle hybride. Si T8 n'est pas fait, **B‑full est déclaré absent** et on publie A1 vs B‑proj plus la ligne disque du 2×2.

### 1.4 Ce que le bras B *n'est pas*

RTN affine par groupes, **sans calibration**. Un relecteur demandera GPTQ‑int4, AWQ, ou un 2 bits calibré. Aucun n'existe localement, aucun ne tourne dans notre harnais (auto‑gptq/AutoAWQ sont CUDA‑orientés, pas de chemin Metal fiable ; ~1‑2 jours d'effort à issue incertaine). **Position tenue : le q4 MLX est déclaré comme la borne basse de la famille 4 bits.** Le battre serait nécessaire, pas suffisant. À l'inverse, dans le régime 2,25–2,50 bits, le RTN est un concurrent légitime : personne ne prétend qu'un RTN 2 bits marche.

---

## 2. Les cinq protocoles

### 2.1 Axe 1 — Espace à froid (FORMAT)

**Définition opérationnelle.** L'ensemble minimal de fichiers que le chargeur de ce bras ouvre pour construire le modèle et tokeniser, sur une machine sans réseau. A1/A2 = le fichier seul (config.json et tokenizer.json sont dedans, vérifié). B = le répertoire. C = config + tokenizer + shards + index.

**Commande** (`du -sb` n'existe pas sur macOS) :

```bash
find /Users/pjmalandrino/qwen3-4b-mlx-q4 -type f -exec stat -f '%z %N' {} + \
  | tee ~/disk-B-2026-08-04.log | awk '{s+=$1} END {print "total", s}'
stat -f '%z %N' /Users/pjmalandrino/qwen3-4b-llvq.bin /Users/pjmalandrino/q4b-e4.llvq
find ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots -type f -exec stat -f '%z %N' {} +
```

**Unités** : puissances de dix (1 Go = 10⁹ o), octets exacts entre parenthèses, jamais de Gio dans le corps du texte.

**Répétitions** : n=1, sans barre. C'est la bonne statistique : un `stat` est reproductible à l'octet.

**Contrôle d'entropie (annexe)** : `zstd -9 -T0 -k` sur les trois objets. Coût ~5‑10 min machine (le `.bin` est quasi incompressible et bascule en mode stocké ; les bras B et C compressent vraiment). L'index maximal observé dans le fichier est 111 043 117 450 038 contre N(12) = 111 043 117 458 000 — l'espace de code est saturé à 7 962 près, donc un gain de compression proche de zéro est le résultat *attendu* et il transforme un argument en mesure. Déclarer le niveau. Ne rien conclure sur B (MLX intercale des scales/biases bf16 dont la compressibilité ne dit rien du codebook).

**Hors périmètre, déclaré** : le checkpoint amont de 8,06 Go que `mlx_lm.convert` doit lire, et les 131 072 tokens de calibration + 4,01 h de machine qu'a coûtés A1. Ce sont des coûts de *production*, pas de déploiement — et l'asymétrie (4 h contre ~2 min) doit être citée là plutôt que masquée.

**Établit** : les octets qu'un déployeur doit expédier, à périmètre nommé. **N'établit pas** : la mémoire, ni le débit — le format qui rentre sur disque n'est pas le format que le noyau lit.

---

### 2.2 Axe 2 — Mémoire (FORMAT + MOTEUR, deux lignes)

**Trois grandeurs nommées, jamais additionnées ni soustraites.**

**M1 — poids résidents, ce que le moteur tient réellement.** Calculé exactement.

| Bras | M1 | Note |
|---|---|---|
| A1 dans notre harnais | 8 044 936 192 o | `sealed::load` appelle `decode_matrix` puis `Tensor::from_vec(...).to_dtype()` — **tenseurs f16 denses** |
| C dans notre harnais | 8 044 936 192 o | identique |
| B dans MLX | 2 262 920 192 o | poids quantifiés résidents (l'en‑tête safetensors de 102 225 o n'est pas un poids) |

**C'est la ligne la plus dure du dossier et il faut la publier nous‑mêmes : aujourd'hui, en mémoire, le format 2 bits ne fait économiser aucun octet.** Le q4 nous bat de ×3,55 sur cette grandeur.

**M1′ — ce que le format coûterait à un runtime qui le lit.** Étiqueté « non atteint aujourd'hui » : Slot32 2,502 Go + embedding f16 0,778 = 3,281 Go (6,525 b/param) ; Grouped32 1,589 + 0,778 = 2,367 Go (4,708 b/param). En convention poids‑seuls homogène : q4 4,5006 b/poids contre nous 6,5245 (Slot32) — **×1,45 contre nous** —, 4,034 avec L≤3, 3,727 avec Grouped32.

**M2 — pic de footprint du processus.** `peak memory footprint` de `/usr/bin/time -l` (= `phys_footprint`), avec `maximum resident set size` publié à côté.

Justification de M2 sur M1(RSS), et **validation de l'instrument déjà sondée en lecture seule** (à rejouer et logger, ticket T9) : un `MTLBuffer` StorageModeShared alloué et jamais écrit ne coûte rien (backing paresseux) ; une fois **écrit**, il compte pleinement dans RSS *et* footprint côté metal‑rs. Côté MLX, `mx.zeros` + `mx.eval` fait bouger le **footprint** linéairement mais **pas le RSS** (2 Go → RSS 31 Mo, footprint 2,18 Go). Conclusion : `peak memory footprint` est la seule grandeur qui suit linéairement dans les deux moteurs ; **comparer les 17,41 Go de RSS aux 2,39 Go de `mx.get_peak_memory()` était faux deux fois.**

`mx.get_peak_memory()` est **retiré de tout tableau croisé**. Il est republié en annexe, à côté du footprint du même processus, étiqueté « pic de l'allocateur MLX ».

**Charge de travail — et ce qu'on refuse de prétendre.** M2 n'est **pas** comparable entre moteurs, et le plan ne fait pas semblant : à ctx 4096, notre `Qwen3::logits` applique la tête à *toutes* les positions (4096 × 151 936 en f16 = 1,245 Go, recopiés en f32 par `window_nll` = 2,490 Go de plus) et notre attention matérialise une matrice de scores 32×4096×4096 par couche, sans cache KV ni flash. `mlx_lm.generate --max-tokens 1` ne calcule qu'une position et alloue un cache KV. L'écart de harnais dépasse plusieurs Go avant qu'aucun octet de poids n'intervienne. **Donc : M2 se compare INTRA‑moteur seulement, et M1 est la seule grandeur inter‑moteurs.**

**Commandes.** L'axe est gratuit : on enveloppe les runs de perplexité déjà programmés.

```bash
# les trois bras, un seul moteur, une seule charge
/usr/bin/time -l env LLVQ_DTYPE=f16 ./target/release/ppl 4096 12 metal /Users/pjmalandrino/qwen3-4b-llvq.bin 2>&1 | tee ~/mem-A1-2026-08-04.log
# côté MLX, intra-moteur
/usr/bin/time -l mlx_lm.generate --model /Users/pjmalandrino/qwen3-4b-mlx-q4 \
  --prompt "$(cat ~/bench-prompt.txt)" --ignore-chat-template --max-tokens 1 --temp 0 --seed 0 \
  2>&1 | tee ~/mem-B-mlx-2026-08-04.log
```

**Répétitions** : 3, ordre tournant, médiane et étendue min‑max. Jamais de moyenne à trois décimales (le pic a un plancher dur et une queue haute).

**Cache KV — constante format‑indépendante, à publier et à trancher.** Ni MLX ni nous ne le quantifions. Dérivation depuis `config.json` (36 couches, 8 têtes KV, head_dim 128, f16) : 36 × 8 × 128 × 2 × 2 = **147 456 o/token = 144 Kio/token**, soit 0,604 Go à 4096 et 1,208 Go à 8192. **Cette valeur contredit d'un facteur 2,22 le « 320 Kio/token » de `docs/fiche-4b.md`, qui contredisait déjà d'un facteur 2 le « ~640 Ko/token » de `pistes-battre-q4.md`.** Trois valeurs incompatibles : point non résolu (§8), et **aucune projection mémoire à 70B n'est publiable tant qu'il n'est pas tranché**.

**Établit** : ce que chaque format pèse en poids résidents dans le moteur qui le lit, et ce que chaque processus coûte à la machine. **N'établit pas** : que le 2 bits fait rentrer un modèle qui ne rentrait pas — le cache KV est un terme additif commun qui rétrécit mécaniquement tout avantage relatif à contexte long.

---

### 2.3 Axe 3 — Vitesse

#### 2.3a Niveau NOYAU (FORMAT)

**Définition.** Coût d'un pas de projection : les 252 matrices réelles du modèle (q 4096×2560, k 1024×2560, v 1024×2560, o 2560×4096, gate 9728×2560, up 9728×2560, down 2560×9728, ×36), batch 1, mémoire froide par la taille du jeu de travail. **Chaque format est mesuré avec le noyau de ses auteurs** : notre `tv_slot`, `mx.quantized_matmul` d'Apple, et le FP16 des deux côtés comme dénominateur commun.

**Statistique publiée : la bande passante atteinte (Go/s), pas seulement l'accélération.** Un matvec batch 1 sur 2 à 7 Go est purement limité par la mémoire ; le rapport 2,07× est, à bande passante égale, la borne 7,267/2,502 = 2,91× escomptée à 71 %. Publier des Go/s sépare *la qualité du noyau* de *la taille du format* — et neutralise du même coup l'objection « votre FP16 est un homme de paille », puisque deux implémentations qui saturent la même barre mémoire donnent la même barre.

**Octets de projection par token** : FP16 7,267 Go · Slot32 2,502 · Flat32 2,125 · **q4 2,044** · Grouped32 1,589.

**Pré‑enregistrement, à commiter AVANT le premier run** (`docs/attentes-2026-08-04.md`) : le q4 lit 18 % d'octets de moins que Slot32 et son décodage (shift + fma) est bien moins cher que le décodage Leech multi‑coquilles. **L'attente est que `mx.quantized_matmul` batte notre noyau fusé, d'environ 1,2× à 1,7×.** Découvrir cela après coup et le publier comme une surprise serait la pire façon de le sortir.

**Commandes.**

```bash
# nous
cargo run --release -p llvq-metal --bin thesis -- /Users/pjmalandrino/qwen3-4b-llvq.bin \
  2>&1 | tee ~/thesis-2026-08-04.log
# MLX (ticket T4, script hors dépôt)
python3 ~/scratch/mlx_kernel_bench.py --reps 7 2>&1 | tee ~/mlxkernel-2026-08-04.log
```

**Protocole du banc MLX** (T4) : construire 252 jeux **distincts** (7,27 Go en f16, 2,04 Go en q4 — au‑dessus de tout cache, froid par la taille, même argument que `thesis`) ; construire les matrices **directement en f16** (pas f32, sous peine de doubler le pic à ~20 Go) ; **un seul graphe** de 252 opérations, `mx.eval(outs)` explicite sur la liste des sorties (MLX est paresseux *et* fait de l'élimination de code mort — 252 matmuls dont les sorties ne sont pas consommées peuvent n'être jamais exécutés) ; `mx.clear_cache()` entre répétitions ; `mx.synchronize` autour du chrono ; 7 passes, 2 jetées.

**Correction obligatoire de `bin/thesis`** (ticket T5) : aujourd'hui il exécute une boucle complète de 7 répétitions sur FP16, **puis** une seconde boucle sur Slot32. Il n'existe aucun couple contemporain à apparier, et FP16 est toujours mesuré en premier (biais thermique, qui nous dessert). La statistique demandée — rapport apparié, médiane sur 5 — **n'est pas obtenable en 5 lignes** : il faut restructurer `pass` en une boucle unique alternant les deux bras à chaque tour et rendant `Vec<(f64,f64)>`. Publier **les deux** statistiques : minimum des 5 (plafond matériel, compatibilité avec l'historique 2,0737 / 2,0574) et médiane des 5 rapports (régime attendu).

**Ce que le rapport LLVQ exclut, et l'exclusion est asymétrique** : la rotation d'incohérence appliquée aux activations (144 transformées par token) n'est payée que par nous, n'a **aucune implémentation GPU**, et n'entre pas dans ce banc. **Notre rapport est donc un MAJORANT.** Le plan n'avance aucun chiffre de latence de rotation — voir §8 pour pourquoi la borne « 144 × 0,15 ms » qui circule est fausse (elle multiplie le coût d'un *command buffer* complet par un nombre de *dispatches intra‑buffer*).

**Recalage du 2,07×.** Le banc MLX donne gratuitement `T_MLX(f16)` contre `T_nous(f16)` sur les mêmes formes. Trois issues à pré‑annoncer : au niveau → le rapport tient ; plus lent → le 2,07× est gonflé et doit être recalculé contre le meilleur FP16 disponible ; plus rapide → le dire et l'expliquer. Retirer par ailleurs partout la formulation « X % du pic de bande passante » : le pic de 400 Go/s est une spec constructeur jamais mesurée, `mx.metal.device_info()` n'expose aucun champ de bande passante, et le « 93 % du pic » est déjà rétracté. **Cela inclut le code** : `llvq-bench/src/bin/rtbits.rs:398` et `:413`, `llvq-metal/src/bin/decreal.rs:274` codent 400 Go/s en dur et impriment des tok/s dérivés qu'un lecteur prendra pour des mesures.

#### 2.3b Niveau BOUT EN BOUT (MOTEUR)

**Un seul couple licite : B contre C, tous deux dans MLX.** Le bras A s'abstient, et l'abstention est publiée avec sa cause : `Qwen3::generate` reconstruit tout le préfixe à chaque pas (coût quadratique, le code le documente), et le noyau fusé n'a littéralement pas d'appelant — un runner qui fait toujours un GEMM sur tout le préfixe n'émet jamais de matvec. **Interdiction formelle de diviser un tok/s de `bin/run` par un tok/s de `mlx_lm`.**

```bash
for i in 1 2 3; do
  /usr/bin/time -l mlx_lm.generate --model /Users/pjmalandrino/qwen3-4b-mlx-q4 \
    --prompt "$(cat ~/bench-prompt.txt)" --ignore-chat-template \
    --max-tokens 256 --temp 0 --seed 0 --verbose True 2>&1 | tee ~/tps-B-$i.log
  sleep 120
done
# puis le même sur ~/qwen3-4b-mlx-f16 (converti avec --dtype float16, cf. 5.9)
```

`--ignore-chat-template` est **obligatoire** : le répertoire q4 livré n'a aucun template (son `tokenizer_config.json` fait 289 o), alors qu'un `mlx_lm.convert` frais du checkpoint en émettrait un — sans le drapeau, les deux bras recevraient des prompts de longueurs différentes. Le confondant eos, lui, est **réfuté** : `mlx_lm` lit l'eos de `config.json` (151645) et non du tokenizer_config, donc les deux bras partagent le même jeu d'arrêt. Garder quand même la garde : **vérifier dans le log que 256 tokens ont bien été produits**, sinon le run est nul.

Publier **`prompt_tps` et `generation_tps` séparément** : `mlx_lm` réinitialise son chrono après le prompt, donc un seul nombre mélangerait un prefill et un décodage.

#### 2.3c Le bras A dans notre moteur — deux quantités bornées, pas un tok/s

Ticket T7. `bin/run` ne contient **aucun chronomètre** : les 2,2–7,6 tok/s connus viennent d'un mur externe qui inclut ~150 s de décodage du fichier scellé. Un ratio construit là‑dessus serait à plus de 90 % un ratio de temps de chargement. On instrumente donc **deux quantités distinctes** :

- **temps de chargement** du format (où le 2 bits perd lourdement — c'est une vraie information : le décodage de 150 681 600 blocs est mono‑thread) ;
- **secondes pour n_new tokens depuis un préfixe de longueur L, sans cache KV**, mesurées à n_new ∈ {1, 13, 49} et publiées **avec la pente**, jamais comme un tok/s unique.

Et une branche checkpoint dans `run.rs` pour que C existe dans notre moteur : cela donne le seul énoncé de format légitime sur cet axe — *à moteur et handicap identiques, le fichier 2 bits tourne à X fois le checkpoint f16*.

#### 2.3d Contrôle de non‑divergence (gratuit)

Les temps par fenêtre imprimés par `bin/ppl` pour les trois bras **doivent être égaux au bruit près** : après chargement, les trois sont des poids f16 denses. Résultat nul attendu, par construction — c'est ce qui en fait un contrôle. **Publier la médiane des deltas des fenêtres 7..N et l'étendue, pas la moyenne sur 1..N.** Les logs du 2026‑08‑04 montrent une dispersion intra‑run de +75 % (10,0 s à 17,5 s par fenêtre) et un écart inter‑bras de 13 % entièrement confondu avec l'ordre (le bras baseline est passé en second, machine plus chaude). Nommer honnêtement : « contrôle de non‑divergence du chemin de calcul, résolution ~20 % ». Ordre alterné entre bras.

---

### 2.4 Axe 4 — Perplexité (FORMAT)

#### 2.4.1 Définition opérationnelle, figée

wikitext‑2‑raw‑v1, split `test`, parquet HF, documents concaténés dans l'ordre du fichier, tokenisé **sans tokens spéciaux**, fenêtres **non chevauchantes** de 4096, **dernière fenêtre partielle jetée** (`floor(N/ctx)`), NLL du token suivant moyennée sur 4 095 prédictions par fenêtre, ppl = exp(moyenne). Mesure sur cette machine : 299 078 tokens → **73 fenêtres pleines**, 70 tokens jetés. Épingler le sha du parquet dans le papier.

Chacun de ces choix est un axe où deux implémentations raisonnables divergent (chevauchement, dernier bloc, BOS). Les énoncer est ce qui rend le chiffre reproductible.

#### 2.4.2 Profondeur : deux tables, chacune avec son nombre de fenêtres

- **Courbe débit‑distorsion : 12 fenêtres**, tous les points (A1, A2, 5 réglages B, C). L'erreur d'échantillonnage est commune à tous les points et s'annule dans la forme de la courbe. Empreinte `3f1baca9033bf251` exigée sur chaque ligne.
- **Table de tête : 73 fenêtres** (recensement du split), pour A1, B@4b‑g64 et C. Cela supprime l'objection « votre sous‑ensemble est plus facile » — notre baseline à 12 fenêtres tombe 1,4 % sous celle du papier — et rend le chiffre directement comparable à une référence pleine population, pour ~55‑60 min de machine.

Les nouveaux chiffres à 73 fenêtres **ne seront pas** 16,9415 / 12,2361. Prévoir que la table du README change, et ne jamais laisser coexister deux paires sans étiquette de nombre de fenêtres.

#### 2.4.3 Comment le bras B entre — et les quatre contrôles avant d'y croire

Ticket T1, `ops/mlx_dequant.py`, ~130‑160 lignes, PEP‑723. Formule vérifiée sur le fichier réel : `w[i, 64g+t] = f32(scales[i,g]) * nib + f32(biases[i,g])` avec `nib = (word[i,(64g+t)//8] >> (4*((64g+t)%8))) & 0xF`, mot u32, quartets petit‑boutien. Formes : `weight` U32 [d_out, d_in/8], `scales`/`biases` **BF16** [d_out, d_in/64], groupe le long de la dimension d'entrée. Aucune transposition requise ([d_out, d_in] est ce que candle attend). Les noms MLX sont **exactement** ceux que `llvq_llm::artifact::key()` construit — rien à renommer.

**Arithmétique imposée : `mx.dequantize(..., dtype=mx.float32)` puis un seul `.astype(mx.float16)`.** Ne **pas** passer par `mlx_lm.convert -d` : son chemin fait `mx.dequantize(...).astype(mx.float16)` où `dequantize` rend le dtype des scales, c'est‑à‑dire **bf16** — soit deux arrondis là où notre bras LLVQ n'en fait qu'un, ce qui pénaliserait le q4 d'un arrondi que MLX ne paie pas en inférence.

**Quatre contrôles, tous avant le premier chiffre publiable :**

| # | Contrôle | Ce qu'il ferme | Coût |
|---|---|---|---|
| L1 | Reconstruction manuelle == `mx.dequantize` bit pour bit, 8 tenseurs échantillonnés | disposition des bits | 10 s |
| L2 | Idempotence : `mx.quantize(mx.dequantize(w,s,b,64,4),64,4)` rend `(w,s,b)` bit‑identiques sur les 253 tenseurs | `group_size`, `bits`, `mode` (mlx 0.30.3 propose affine/mxfp4/mxfp8/nvfp4 — un mode erroné rendrait des poids plausibles et faux) | 1 min |
| L3 | `mx.quantized_matmul(x,q,s,b)` contre `x @ dequant(w).T` sur les 252 formes, écart relatif max publié | que notre overlay mesure bien ce que le noyau MLX calcule (le dépôt a déjà le précédent inverse : MLX et notre pipeline divergent au 5ᵉ token sur les mêmes poids) | 2 min |
| L4 | Budget de narrowing : `‖f16(W)−W‖_F / ‖W‖_F` et compte de subnormaux f16 sur les 252 tenseurs | que l'overlay ne mesure pas notre narrowing. Critère : ≥ 100× sous l'erreur de quantification du q4 lui‑même | 60 s |

**Et deux contrôles de chaîne, dans cet ordre :**

- **Contrôle identité** : overlay construit depuis le checkpoint lui‑même (252 projections lues et réécrites en f16), `ppl(overlay identité) == ppl(baseline)` à 3 fenêtres. Exerce le nommage, les formes, la transposition et le dtype. Si l'égalité n'est qu'approchée (résidu `VarBuilder::from_mmaped_safetensors` contre `Tensor::from_vec`), **chiffrer le résidu et le déclarer comme plancher de résolution de l'axe qualité** — c'est une information, pas un échec.
- **Contrôle 8 bits**, plus tranchant : overlay q8‑g64, ppl attendue à ~0,1 % de la baseline. À 8 bits l'erreur de quantification est négligeable, donc **tout** écart significatif accuse le script (ordre de lignes, group_size, scales/biases inversés, double arrondi). À 4 bits le même bug se cacherait derrière une dégradation attendue. C'est la différence entre un contrôle qui détecte et un contrôle qui rassure.

#### 2.4.4 Commande, dtype, empreinte

```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 \
  cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal <objet> \
  2>&1 | tee ~/ppl-<bras>-12-2026-08-04.log
```

`<objet>` = `/Users/pjmalandrino/qwen3-4b-llvq.bin` (A1), `/Users/pjmalandrino/q4b-e4.llvq` (A2), le chemin de l'overlay (B), rien (C). `bin/ppl` est **f32 par défaut** : oublier `LLVQ_DTYPE=f16` sur un seul bras produit un tableau silencieusement mixte. Relire `dtype f16` sur chaque ligne de résultat.

**Critère d'acceptation non négociable : l'empreinte de tokens doit être identique sur toutes les lignes.** À défaut le run est jeté, pas rattrapé. Fait à consigner : le `tokenizer.json` du répertoire MLX n'est **pas** byte‑identique à celui du checkpoint (11 422 650 o / sha `be75606093db2094…` contre 11 422 654 o / `aeb13307a71acd8f…`) — raison technique supplémentaire, indépendante du moteur, de faire passer B par notre harnais plutôt que de le noter dans MLX.

**Établit** : la restitution sur du texte brut, à format seul variable. **N'établit pas** : le raisonnement. Le profil MMLU par matière du dépôt le démontre sur ce modèle même — algèbre abstraite et comptabilité au niveau du hasard pendant qu'histoire, droit et psychologie tiennent au‑dessus de 80 %. **Ne jamais présenter la perplexité seule comme preuve de qualité.**

---

### 2.5 Axe 5 — MMLU (FORMAT)

#### 2.5.1 Protocole, figé et certifié

Hendrycks 5‑shot, tel que `bin/mmlu` l'implémente, **sans y toucher** : les 5 exemples viennent du split `dev` de la même matière, dans l'ordre du fichier ; en‑tête `The following are multiple choice questions (with answers) about {matière}` ; bloc `question / A. / B. / C. / D. / Answer: X` ; la question notée s'arrête à `Answer:` **sans espace final** ; score = argmax des logits des tokens uniques `" A"`, `" B"`, `" C"`, `" D"` à la dernière position, comparés en f32 ; **une** passe avant par question ; aucune normalisation de longueur.

Ce protocole a un certificat : il rend **70,42** sur la baseline là où le papier annonce **70,2** (+0,22 pp = 0,17 σ). C'est l'argument le plus fort du dossier et il vaut précisément parce que rien n'a bougé. **Toute modification du harnais invalide le certificat : après les tickets T2 et T3, rejouer la baseline à limit=40 et exiger 70,42 au centième avant d'engager les runs longs.** Budget : +44 min, à faire **une** fois pour les deux patches.

Le chiffre publié est le **micro** (une question = un poids), le macro à côté et nommé. Le README publie encore la ligne macro (72,85 / 57,59) sans jamais écrire le mot — à corriger ligne à ligne.

#### 2.5.2 Profondeur

| Niveau | Questions | Coût/bras | SE |
|---|---|---|---|
| limit=40 | 2 280 | 44‑47 min | ±1,28 / ±1,36 pp |
| recensement | 14 042 | **5,23 h / 5,57 h** (reproduit exactement depuis le log par matière du 2026‑08‑02, repondéré par les populations réelles) | 0,00 |

**Décision** : `limit=40` pour tous les bras (balayage + tête) ; **recensement réservé aux trois bras de tête (A1, B@4b‑g64, C), en dernier, optionnel** — ~16,5 h, deux nuits, 0 $. Il n'y a pas de bon point intermédiaire : limit=250 coûte 3,5 h/bras pour ±0,41.

#### 2.5.3 Le test apparié — et pourquoi McNemar seul ne suffit pas

Les bras répondent aux **mêmes** questions (graine `SplitMix64(0x6_11B0 ^ subject.len())`, qui ne dépend que de la longueur du nom de matière). L'appariement est gratuit et le test indépendant est le mauvais test.

**Mais McNemar global teste le taux poolé — c'est‑à‑dire une quantité proche du macro — alors que le chiffre publié est le micro stratifié** (`mmlu.rs:116-126` repondère les 57 taux par la population réelle ; `professional_law` pèse 10,9 % en étant estimé sur 40 tirages). C'est exactement la distinction que le dépôt a passé une session à établir.

**Statistique retenue : bootstrap apparié stratifié par matière** sur le dump par question — rééchantillonner les questions dans chaque matière, recalculer le micro stratifié des deux bras, prendre la distribution de la différence (10 000 tirages, < 1 s). Elle donne la SE de la statistique réellement publiée, correction de population finie comprise. McNemar global reste publiable **en secondaire, étiqueté « non pondéré »**.

Gain d'appariement, à estimateurs égaux (non pondéré, sans fpc) : SE indépendante 1,393 pp contre SE appariée 0,997 pp, soit ~1,40× en SE et ~1,95× en questions équivalentes — **pas 3,5×**.

#### 2.5.4 Le dump, qui doit précéder les runs

Ticket T2, coût asymétrique : sans lui il faut tout relancer (0,8 h à limit=40, 16,5 h au recensement). Deux pièces :

- **CSV par question** `subject,index,answer,pick,correct`. `MmluItem` n'a **pas** d'identifiant : la seule clé stable est l'indice dans l'ordre parquet de la matière, et il est détruit par le Fisher‑Yates. Il faut zipper **avant** de mélanger (`items.iter().enumerate().collect()`).
- **Empreinte de tokens** sur la ligne de résultat (`eval::token_fingerprint` existe et ne coûte rien). C'est le **dernier trou d'iso‑conditions** de l'axe 5 : aujourd'hui l'identité des 2 280 questions est établie par lecture de code, jamais relue sur une sortie. Verrou à poser en même temps : un test qui exige l'empreinte identique entre deux modèles différents et différente si `limit` change.

#### 2.5.5 Faire entrer le bras B

Ticket T3, `mmlu.rs` : troisième branche « chemin `.safetensors` → checkpoint + overlay ». Ce n'est **pas** un copier‑coller de `ppl.rs:71-85` : dans `ppl.rs` le dépôt de base vient de `LLVQ_MODEL` et l'overlay du 4ᵉ positionnel ; dans `mmlu.rs` le **premier** positionnel *est* le dépôt, et `mmlu.rs` ne lit aucun `LLVQ_MODEL`. Décision : **lire `LLVQ_MODEL` et échouer s'il est absent** (pas de repli sur Qwen3‑0.6B, ce serait le même piège silencieux que le device fallback qu'`eval.rs` a supprimé), et prendre l'overlay en **argv[3]**, comme `ppl.rs`. Réutiliser exactement la même fonction d'overlay que `ppl.rs`, sinon les deux métriques ne portent plus sur le même objet.

**Établit** : du QCM 5‑shot à choix de lettre, sur un profil de capacités. **N'établit pas** : la génération libre, ni l'extraction documentaire (le benchmark métier prévu au plan et jamais fait). Publier le **profil par matière** plutôt que le seul agrégat — c'est lui qui montre le mécanisme. Ne jamais écrire « exactement 25 %, le hasard » : à 40 tirages la barre par matière est ±7 pp.

---

## 3. Le code à écrire

Ordonné par déblocage. **T0 est bloquant pour tout le reste.**

| # | Pièce | Fichier(s) | Effort | Vérification |
|---|---|---|---|---|
| **T0** | Commiter l'arbre (16 fichiers modifiés + 7 non suivis, dont `format.rs` LVQ3, `sealed.rs`, `embedquant.rs`) ; retirer `Cargo.lock` du `.gitignore` et le commiter | racine | 45 min | `git status --porcelain` vide ; `cargo clippy --all-targets` à zéro warning ; `cargo test --release -- --include-ignored` vert |
| **T1** | `mlx_dequant.py` : dequant q4→f16, écrit un overlay `.safetensors` des 252 projections ; option `--embed` pour B‑full ; quantification directe `mx.quantize` pour les 4 réglages manquants ; dump binaire plat des triplets pour T4 | `ops/mlx_dequant.py` (~130‑160 l) | **2,5‑3 h** | L1+L2+L3+L4 (2.4.3) imprimés dans le log ; contrôle identité ; contrôle 8 bits ; reproduction tenseur‑pour‑tenseur du q4‑g64 livré |
| **T2** | Dump CSV par question + empreinte de tokens ; zipper l'index avant Fisher‑Yates | `llvq-llm/src/bin/mmlu.rs` (~35 l) | **1 h** | test : empreinte identique entre deux modèles, différente si `limit` change ; rejeu baseline limit=40 → 70,42 |
| **T3** | Branche overlay `.safetensors` (argv[3]) + `LLVQ_MODEL` obligatoire | `llvq-llm/src/bin/mmlu.rs` (~20 l) | 45 min | même rejeu baseline que T2 (une seule fois pour T2+T3, +44 min machine) |
| **T4** | Banc noyau MLX : 252 formes, un seul graphe, `mx.eval` explicite, `clear_cache`, 7×2 jetées ; poids appariés avec notre bras via le dump de T1 | `~/scratch/mlx_kernel_bench.py` (~100 l) | **3‑4 h** | somme des temps individuels ≫ temps du graphe unique ; jeu de travail > tout cache ; sorties bit‑comparables à `x @ dequant(w).T` |
| **T5** | `thesis` : boucle unique alternant les bras → `Vec<(f64,f64)>` ; impression des 5 couples, du min par bras et de la médiane des ratios ; flag `--only <bras>` ; `assert!(centroids.len() == 2)` | `llvq-metal/src/bin/thesis.rs` (~50 l) | **1,5 h** | le min par bras reproduit l'historique (2,0737 / 2,0574) à la dérive près ; `--only` change le pic de footprint de la taille attendue |
| **T6** | Imprimer NLL et count **par fenêtre** (9 chiffres significatifs) à côté de la ppl cumulée | `llvq-llm/src/bin/ppl.rs` (~10 l) | 30 min | la moyenne des NLL par fenêtre reproduit `ln(ppl)` final ; débloque la barre d'erreur *et* renforce le contrôle de déterminisme de 1 à N points |
| **T7** | `run.rs` : branche checkpoint (`Checkpoint::fetch`), prompt optionnel en argv, deux `Instant` (chargement / génération) avec compte de tokens ; restructurer le bloc d'affichage (les champs `s.bytes`, `s.quantized_weights`… n'existent pas pour un checkpoint) | `llvq-llm/src/bin/run.rs` (~45‑60 l) | **1,5 h** | le temps de chargement du scellé retombe sur les ~150 s connus ; la pente n_new ∈ {1,13,49} reproduit la décroissance documentée |
| **T8** | `pub fn set_embedding(&mut self, t: Tensor)` remplaçant `embed` **et** réassignant `head` si `tie_word_embeddings` ; appel depuis `artifact::load` quand `model.embed_tokens.weight` est dans l'overlay | `llvq-llm/src/model.rs` + `artifact.rs` (~30 l + test) | **2 h** | test : corrompre volontairement l'embedding et exiger que la perplexité bouge ; assert que `head` et `embed` portent les mêmes valeurs après overlay |
| **T9** | Sonde footprint : allouer un `MTLBuffer` de taille N **et écrire toutes ses pages** avant lecture du footprint | `llvq-metal/src/bin/hello.rs` (~15 l) | 20 min | le footprint monte de N ± 5 % ; si seul le RSS monte, l'axe 2 retombe sur le RSS et il faut le déclarer |
| **T10** | Ligne Slot32 dans `rtbits` (il a déjà `width_slot` via `llvq_artifact::runtime` et boucle sur tous les blocs réels) | `llvq-bench/src/bin/rtbits.rs` (~15 l) | 30 min | doit reproduire le compteur de `thesis` (2,502 Go) — second compteur indépendant, le trou actuel |
| **T11** | *(optionnel)* Bras Grouped32 et Flat32 dans `thesis` (shaders existants dans `matvec.rs`, à **re‑tuiler** : ils stagent toute l'activation en threadgroup memory, 38 Ko à d_in=9728 contre 32 Ko de limite Metal ; leur lecture de queue doit passer en mémoire device) | `llvq-metal/src/bin/thesis.rs` (~120 l) | **3 h** | assert 1e‑3 par bras ; pire erreur imprimée par bras |

**Non résolu, donc non planifié** : l'implémentation Metal de `Rotation` (2‑3 jours, non bornée tant que le choix de fusion n'est pas fait), le cache KV dans notre runner (~120‑150 lignes **plus ~40 lignes de test d'équivalence** — `bin/oracle` ne teste que la passe pleine fenêtre et ne verrouillerait **rien** d'un chemin incrémentiel), et le branchement du noyau fusé dans `bin/run`.

---

## 4. Plan statistique

### 4.1 Trois sources de variance, trois traitements

| Source | Traitement | Coût |
|---|---|---|
| **Mesure** (rejouer la même commande sur le même objet) | se **prouve une fois**, puis n=1 pour toujours | 45 min |
| **Échantillonnage** (fenêtres, questions) | se **supprime par recensement**, pas par répétition ; sinon barre appariée | 1 h (ppl) / 16,5 h (MMLU) |
| **Méthode** (re‑quantifier) | **hors périmètre de cette campagne**, et déclaré tel quel | 12,6 h+ |

### 4.2 Déterminisme — la prémisse de tout le reste

Deux rejeux identiques de `ppl` (12 fenêtres, f16, metal) sur A1, et deux rejeux de `mmlu` à limit=5. **Faire T6 d'abord** : avec la NLL par fenêtre à 9 chiffres, le test passe de 1 à N points de comparaison par run pour le même temps machine. Critère : égalité exacte. Prédiction : `window_nll` accumule `picked.sum_all()` en f32 sur 4 095 termes avant conversion en f64 (`model.rs:361`) — c'est la seule réduction en précision réduite du chemin d'évaluation, et l'effet attendu (~2,4e‑7 nats/token) est trois ordres sous la dispersion inter‑fenêtres. **À citer explicitement dans le résultat.**

Si le déterminisme passe : n=1 sur les axes 4 et 5, et aucune répétition d'évaluation n'est plus jamais payée. Si non : mesurer l'amplitude, la déclarer, et tout le budget de répétition triple.

### 4.3 Perplexité — la barre existe déjà dans les logs

Les 12 lignes `running ppl` des logs du 2026‑08‑04 permettent de reconstruire la NLL par fenêtre par télescopage (`nll_w = w·ln(ppl_w) − (w−1)·ln(ppl_{w−1})`), et la reconstruction rejouée indépendamment donne : delta apparié 0,325413 nats/token, sd inter‑fenêtres 0,056119, **sem 0,016200**, corrélation inter‑bras 0,975134 (l'appariement divise l'écart‑type par 6,05).

**Deux corrections par rapport à la version initiale de ce calcul.** (i) Le « contrôle de validité » à 3,7e‑5 nats est une **identité algébrique** — le télescopage est exact par construction — et ne teste rien ; il ne mesure que l'arrondi de la dernière ligne. Le vrai contrôle est T6. (ii) **La correction de population finie ne s'applique pas** : `ppl.rs` boucle sur `(w·ctx, (w+1)·ctx)`, donc les 12 fenêtres sont le **préfixe contigu** des 73, pas un tirage aléatoire sans remise. Sans fpc, l'intervalle vaut **±3,6 %** sur le ratio, pas ±3,3 %, et il s'étiquette « généralisation à travers les fenêtres », pas « échantillonnage parmi les 73 ».

**Statistique publiée**, trois lignes et jamais moins :
1. la ppl de chaque bras au recensement 73 fenêtres, **sans barre** ;
2. le **delta apparié de log‑vraisemblance par fenêtre**, avec sem et IC95 t apparié — *la* statistique de comparaison ;
3. le ratio comme `exp(delta)`, IC transporté.

**Interdit** : publier une ppl absolue accompagnée d'une barre. La barre inter‑fenêtres de la ppl absolue vaut ±14 % et décrit la difficulté du corpus, pas la qualité du modèle.

Projection à 73 fenêtres : sem ≈ 0,05612/√73 = 0,00657 nats, 2σ = 0,0131 nats = **1,32 % sur le ratio**.

### 4.4 Seuils de défendabilité, appliqués aux claims existantes

| Claim | Valeur | Verdict |
|---|---|---|
| « 16,9617 < 17,04, on passe sous QTIP » | +0,004606 nats = 0,31 sem | **indéfendable — retirer** |
| « on paie +3,06 % de nats face à QTIP » (l'auto‑critique) | f16 : +0,008352 nats, t = 0,56 sur df=11 ; f32 : +0,009711, t = 0,66 | **non significatif non plus** |

**Le README remplace donc une revendication non défendable par une auto‑critique tout aussi non défendable.** La formulation juste, sur une seule normalisation : *« notre surcoût de log‑vraisemblance dépasse celui de QTIP de 0,0084 nats/token (+2,6 %), pour une dispersion inter‑fenêtres de ±0,0148 nats — l'écart est dans le bruit de nos propres fenêtres. Nous payons 8 % de bits en plus. »* Après recensement, le seuil devient un seuil de généralisation (2σ = 0,0131 nats) et l'écart de 0,0084 reste dessous (t = 1,27).

**Attention à une collision que le plan doit assumer** : notre point est à 2,1595 b/poids contre 2,250 pour le moins cher des réglages MLX sains — 4,0 % de marge sur **l'abscisse**. Si le point LLVQ tombe à moins de ~1,3–3,6 % de la courbe interpolée sur **l'ordonnée**, la figure principale ne pourra soutenir aucune claim de supériorité par les règles que le papier s'est données. **Elle montrera alors une frontière, pas une victoire — et c'est un résultat publiable, à condition de l'avoir écrit avant de mesurer.**

### 4.5 Variance de méthode — déclarée, pas mesurée

Le dépôt possède **une** observation de dispersion : 14,2684 contre 15,2909, ~7,2 %. Elle n'est **pas** un sigma :
- n = 2 ;
- les deux lignes portent des **débits différents** (2,7289 vs 2,1117) et des **noms de configuration différents** — ce sont deux configurations, pas deux tirages ;
- elles ont été produites par un binaire antérieur au correctif de rétraction, à l'écrivain LVQ1, sans `LLVQ_DTYPE`, sur un autre shard C4. HEAD ne peut pas les reproduire ;
- le test invoqué pour affirmer « c'était le même quantifieur » (`under_the_old_retraction_shape_gain_was_direction_only`) compare deux bras **tous deux à magnitude libre** : l'égalité est vraie par construction et n'établit rien.

**Décision : retirer les 7 % du dossier comme observation de variance, et l'étiqueter « n=2, configurations non identiques, cause non tranchée, aucun sigma ».** Écrire explicitement : *aucune barre publiée ici ne couvre la variance de re‑quantification, qui n'est pas mesurée.*

Un test de déterminisme du pipeline à 3 blocs (~65‑80 min : 3 × 401 s de quantification × 2 runs, **plus** les deux boucles ppl complètes que `smoke` lance de toute façon) reste utile — il licencie n=1 sur le pipeline de HEAD — mais il ne prouve **que** cela. Il ne tranche pas les 7 %.

Si un jour on veut le sigma : `LLVQ_CALIB_SEED` ne bouge **pas** que les offsets. Le chemin `seed=None` prend un préfixe contigu aligné depuis le token 0 ; le chemin seedé tire des offsets non alignés sur tout le corpus. **L'objet publié n'appartient pas à la population dont on estimerait le sigma** — il faudrait d'abord mesurer le décalage de moyenne. Et un sigma mesuré à 3 blocs serait faux pour l'objet publié (le seul mécanisme connu vaut 0,04 % à 3 blocs et 7 % à 36).

---

## 5. Check‑list de pré‑vol

À exécuter dans cet ordre. Une étape rouge arrête la campagne. Les étapes 5 à 8 produisent des **chiffres à comparer**, pas des commandes à voir se terminer sans erreur.

1. **Arbre propre.** `git status --porcelain` vide, commit noté, `Cargo.lock` commité et haché. *Aucun chiffre de papier ne sort d'un arbre non commité.*
2. **Versions.** `rustc --version`, `cargo --version`, `sw_vers`, `python3 -c "import mlx.core, mlx_lm; print(mlx.core.__version__, mlx_lm.__version__)"`.
3. **Identité des objets.** `shasum -a 256` sur A1 (doit rendre `9db213ef…c84b0`), A2, `qwen3-4b-mlx-q4/model.safetensors` ; `stat -f%z` sur chacun.
4. **Build et suite.** `cargo clippy --all-targets` à zéro warning, `cargo test --release -- --include-ignored` vert. **Compter 10‑20 min de reconstruction release** après T0.
5. **Identité des tokens.** Encoder wikitext‑2 test avec les deux `tokenizer.json` (checkpoint et MLX) et comparer les listes d'**IDs** — pas les fichiers, qui diffèrent de 4 octets sur `add_prefix_space`, `use_regex`, `trim_offsets`. Idem sur un échantillon de prompts MMLU.
6. **Narrowing bf16→f16.** Sur `model.embed_tokens.weight` seul, comparer `mx.astype(bf16→f16)` à un arrondi RTNE numpy, compter les discordances. Deux implémentations coexistent dans la campagne (crate `half` côté candle, `mx.astype` sur GPU) et 77 045 valeurs de cet embedding vivent dans la zone subnormale f16, 451 tombant à zéro. **2 minutes qui transforment un échec inexpliqué en note de bas de page chiffrée.**
7. **Déterminisme** (§4.2). Deux rejeux, égalité exacte. **Avant** l'étape 8, parce que c'est lui qui donne son seuil.
8. **Contrôle de passage.** Checkpoint converti par MLX en f16, rechargé chez nous : ppl doit tomber sur la baseline **à l'enveloppe mesurée en 7 près** (pas « égalité à 4 décimales »), et l'empreinte doit être `3f1baca9033bf251`.
9. **Objets dérivés construits et vérifiés** : `qwen3-4b-mlx-f16` (`mlx_lm.convert --hf-path Qwen/Qwen3-4B --mlx-path /Users/pjmalandrino/qwen3-4b-mlx-f16 --dtype float16` — **`--dtype float16` obligatoire**, sinon il retombe sur le `torch_dtype` du config, donc bf16 ; `convert()` lève si le répertoire existe déjà ; **réseau requis**, le snapshot HF local n'a ni `tokenizer_config.json` ni `generation_config.json`) ; overlays B avec leurs quatre contrôles L1‑L4.
10. **Machine.** Secteur (`pmset -g ps`), applications fermées, `uptime` relevé, `caffeinate -dimsu` armé, 120 s d'inactivité entre runs. `pmset -g therm` ne remonte rien sur cette machine et `powermetrics` exige sudo : la dérive thermique se **neutralise par le protocole** (ordre alterné, médianes, rapports appariés), elle ne s'observe pas.
11. **Disque.** `df -h` ≥ 30 Go libres avant chaque étape overlay. Un overlay pèse 7 267 Go d'octets — **7,267 Go** ; six overlays simultanés dépasseraient les 66 GiB disponibles. **L'enchaînement est séquentiel‑et‑supprimer, obligatoire.**
12. **Plan de logs.** Un `tee` par run : `~/<axe>-<bras>-<AAAA-MM-JJ>-<rep>.log`. `set -o pipefail`. **Aucun chiffre n'entre dans le papier sans un log ou un corps de commit qui le porte** — trois chiffres phares du dossier (2,39 Go, 129,8 tok/s, les trois décimales de 21,691 ms) n'en ont aucun.
13. **Pré‑enregistrement commité** (§2.3a et 4.4), horodaté **avant** le premier log.
14. **Relire `dtype f16`** sur chaque ligne de résultat de qualité.

---

## 6. Ordre d'exécution et budget

Ordonné par (ce que ça débloque) / (ce que ça coûte). **Chaque étape laisse un livrable publiable.**

| Étape | Contenu | Humain | Machine | Livrable si on s'arrête là |
|---|---|---|---|---|
| **E0** | T0 (commit, Cargo.lock) ; pré‑vol 1‑4 ; geler les définitions (§0), le 2×2 disque (§1.3), la comptabilité de sérialisation, le pré‑enregistrement | 4 h | 20‑30 min | Une **note de correction** : le 2×2 disque, les trois « FP16 », le retrait des claims non défendables (§4.4), la correction macro/micro du README, le retrait de la cellule mémoire trompeuse |
| **E1** | T1 + T6 ; contrôles L1‑L4, identité, 8 bits ; ppl 12 fenêtres × 8 objets (A1, A2, 5 réglages B‑proj, C) ; ppl 73 fenêtres × 3 bras de tête ; `/usr/bin/time -l` sur tout | 4 h | **2,5‑3 h** | **La figure débit‑distorsion en perplexité.** Le papier devient écrivable. La case décisive — la qualité du q4, aujourd'hui *vide, pas faible* — est remplie |
| **E2** | T2 + T3 ; re‑certification baseline limit=40 ; MMLU limit=40 × 8 objets ; bootstrap apparié stratifié | 2 h | **6,8 h** (dont 44 min de re‑certification) | Les **deux panneaux** de la figure ; le profil par matière superposé A/B ; les tests appariés |
| **E3** | T9 ; vitesse moteur MLX (B vs C, 3 rep) ; `mx.get_peak_memory` en annexe ; contrôle de non‑divergence dépouillé des logs E1 | 1,5 h | 40 min | Les tables **mémoire** et **vitesse‑moteur**, avec deux SUSPECT (129,8 tok/s, 2,39 Go) convertis en MESURE |
| **E4** | T5 + T4 ; banc noyau apparié des deux côtés ; recalage du 2,07× contre `mx.matmul` ; T10 | 6 h | 1 h | La table **vitesse‑format** en Go/s, et le recalage du chiffre phare du gate G6 |
| **E5** | T7 ; chargement + pente n_new dans notre moteur, A1/A2/C | 1,5 h | 45 min | Le seul énoncé de vitesse légitime pour le bras A |
| **E6** | T8 ; B‑full ; ppl + MMLU de B‑full et A2 en produit‑contre‑produit | 2 h | 1,5 h | La ligne **produit‑contre‑produit** du 2×2, avec sa qualité |
| **E7** *(opt.)* | Recensement MMLU des 3 bras de tête | 30 min | **16,5 h** (2 nuits) | Barres d'échantillonnage à zéro, comparabilité directe au papier |
| **E8** *(opt.)* | T11 ; Grouped32/Flat32 sur le modèle entier | 3 h | 15 min | La figure bits↔vitesse sur **un seul protocole** au lieu de deux |

**Cumul E0→E6 : ~21 h humain, ~13‑14 h machine, 0 $** (tout tient sur le M3 Max ; checkpoint, wikitext, C4 et MMLU sont en cache local). Avec E7+E8 : ~24 h humain, ~30 h machine.

**Point de décision après E1.** Si le q4 se dégrade à ~1‑2 % là où nous sommes à +38,5 %, la thèse **produit** est morte sur un 4B et le papier devient un papier de **noyau et de format** — ce qui change ce qu'il faut mesurer ensuite. Engager E4 (le plus intéressant techniquement, 6 h humain) avant E1 serait investir dans une course dont la conclusion est peut‑être déjà tranchée.

**Contrainte disque, à respecter littéralement** : un overlay à la fois, `ppl` **puis** `mmlu` sur le même overlay, puis `rm`. Cela **interdit de paralléliser E1 et E2** — c'est pour ça que E2 relit les overlays, et c'est déjà dans son budget.

---

## 7. Ce que le plan permettra de dire — et ce qu'il ne permettra pas

### 7.1 Claims soutenables, avec leur niveau de preuve

| Claim | Preuve | Force |
|---|---|---|
| « À périmètre d'embedding apparié, le fichier LLVQ 2 bits est ×1,59 plus petit que le q4 g64 ; ×1,87 avec embedding int4 des deux côtés ; ×2,08 sur les projections seules » | octets sur disque, n=1, périmètre nommé | **forte** — aucun moteur n'intervient |
| « À débit égal sur les projections, le réseau de Leech se situe [au‑dessus / en dessous] de la famille affine par groupes » | figure débit‑distorsion, un seul moteur, empreinte de tokens identique, 5 points MLX + 1 point LLVQ | **forte si la marge dépasse 2σ apparié**, sinon c'est une frontière et pas un classement |
| « Notre harnais reproduit la baseline du papier à 0,22 pp (0,17 σ) » | déjà mesuré, log conservé | **très forte** — c'est l'argument le mieux fondé du dossier |
| « Le décodeur Leech multi‑coquilles fusé bat le FP16 de N× sur le modèle entier, à M Go/s atteints » | `bin/thesis` apparié + recalage contre `mx.matmul` | **forte après E4**, majorant (rotation exclue) |
| « Le surcoût de log‑vraisemblance face à QTIP est de 0,0084 nats/token, dans le bruit de nos fenêtres, pour 8 % de bits en plus » | delta apparié normalisé, IC95 | **honnête** — et strictement meilleure que « on passe sous QTIP » |
| « 8 % du débit au‑dessus des 2,000 théoriques est intégralement de la sérialisation ; il n'y a aucune perte de codage » | comptabilité en quatre postes, arithmétique exacte | **forte**, zéro machine |

### 7.2 Résultats négatifs à publier — nous‑mêmes, et en section, pas en note

1. **En mémoire, le format 2 bits ne fait économiser aucun octet aujourd'hui.** `sealed::load` décode vers des tenseurs f16 denses : notre bras pèse exactement ce que pèse le FP16, et le q4 nous bat de ×3,55 sur les poids résidents (×1,45 même si Slot32 était branché). Le publier avec les 2,50 Go de Slot32 en regard transforme un aveu en feuille de route.
2. **Le noyau fusé n'a pas d'appelant**, et c'est ce qui explique tout le reste : le pic à 17,41 Go, les 2,2–7,6 tok/s, l'impossibilité de comparer la vitesse au niveau moteur. Trois obstacles nommés par dureté : pas de cache KV donc jamais de matvec ; base tournée sans rotation GPU ; le prefill exige de toute façon un chemin dense. **Ne pas annoncer de délai** — la rotation GPU n'est pas bornée, et l'attente honnête après branchement reste sous les 129,8 tok/s de MLX.
3. **Nous reproduisons leur méthode moins bien qu'eux** : −14,33 pp sur MMLU contre −9,5 pp, avec deux différences qui jouent contre nous (~100× moins de calibration ; rotation d'entrée seule contre Input+Output). C'est le résultat négatif le plus fort du dossier **parce que** la baseline reproduit le papier à 0,17 σ : le déficit ne peut plus être imputé au harnais.
4. **Le q4 va probablement battre notre noyau** au niveau matvec, d'environ 1,2× à 1,7×, parce qu'il lit 18 % d'octets de moins. Pré‑enregistré (§2.3a).
5. **Coder le gain coûte +3,17 % de perplexité pour −0,618 bit/poids** (A/B à deux bras, Qwen3‑0.6B, 3 blocs, log conservé) — ce qui **corrige** la ligne encore publiée « quantifier le gain ne coûte presque rien : 0,04 % pour 0,52 bit », issue d'un A/B où les deux bras étaient le même quantifieur.
6. **Le format qui va vite ne rentre pas mieux que du 4 bits ; le format qui rentre mieux ne va pas vite.** Frontière mesurée, publiée comme problème ouvert et non comme détail.

### 7.3 Ce qu'un relecteur exigera et qu'on n'aura pas

| Exigence | Statut | Position tenue |
|---|---|---|
| Un 4 bits **calibré** (GPTQ‑int4, AWQ) | absent, ~1‑2 jours à issue incertaine sur Mac | déclaré : le q4 RTN est la **borne basse** de la famille 4 bits ; le battre serait nécessaire, pas suffisant |
| Un 2 bits calibré concurrent (QuIP#, QTIP, AQLM) mesuré chez nous | absent | comparaison **par citation** seulement, table séparée, en nats normalisés sur chaque baseline |
| Un second modèle | Qwen3‑8B existe (2,0436 b/poids, ×1,267 contre ×1,386 sur le 4B), mais son bras B et son MMLU manquent : ~6‑7 h machine et ~30 Go de disque, **incompatible avec E1‑E6 sans nettoyage** | à décider ; ne **jamais** publier le ratio de compression du 8B (`tie_word_embeddings=false`, embedding = 57 % de l'artefact scellé, ratio ×3,7) |
| Une baseline serveur / CUDA / régime batché | hors de portée | déclaré en une phrase et sans excuse : batch 1, un token, mémoire unifiée. Ne **jamais** poser les 16,3 µs de leur Table 7 à côté de nos 10,5 ms |
| Une barre d'erreur sur la **méthode** | absente | déclarée (§4.5) : n=2, configurations différentes, aucun sigma |
| Le coût en latence de la rotation d'incohérence | non mesurable | déclaré ; tout rapport de vitesse LLVQ est un **majorant** |
| Une validation externe du harnais MMLU | possible via `mlx_lm.evaluate` (`pip install lm_eval`, ~50 Mo) | **hors plan sans accord explicite** : c'est le seul téléchargement discrétionnaire, il coûte 3‑8 h machine non bornées sans pilote `--limit 20`, et l'écart mesuré agrégerait harnais + moteur + tokenizer sans les isoler |

---

## 8. Points non résolus

1. **Le budget du cache KV ne boucle pas.** Trois valeurs incompatibles circulent : ~640 Ko/token (déjà signalé faux d'un facteur 2), 320 Kio/token (fiche §5.6 et §7), et 144 Kio/token dérivé de `config.json` (36 × 8 × 128 × 2 × 2 = 147 456 o). Le rapport 320/144 = 2,22 ne correspond à aucune substitution évidente. **Aucune projection mémoire à 70B n'est publiable tant que ce n'est pas tranché**, et rien dans le dépôt n'alloue de cache KV, donc il n'y a pas de code à lire pour arbitrer.

2. **Le mécanisme du pic Metal à 17,41 Go.** Une sonde exploratoire suggère fortement la double résidence hôte+buffer (`new_buffer_with_data` depuis un `Vec` hôte — exactement ce que fait `Tensor::from_vec` dans `sealed.rs:92` — a mesuré un pic de 4,01 Go pour 2 Go de données, facteur 2 exact ; 8,045 Go × 2 ≈ 16 Go, l'ordre des 17,41 mesurés). **À confirmer en une commande** (`footprint -p $(pgrep -f 'release/run')`, qui fonctionne sans sudo) — cela transformerait un point ouvert en mécanisme établi.

3. **La borne de latence de la rotation d'incohérence.** Le chiffre qui circule (« 144 dispatches × ~0,15 ms = 21,6 ms, donc toute l'avance effacée ») repose sur une lecture fausse : `Kernel::overhead` mesure l'aller‑retour d'un **command buffer complet**, pas le coût d'un dispatch **à l'intérieur** d'un buffer. `thesis` encode 252 dispatches dans un seul buffer pour 10,46 ms au total, soit ~41 µs par dispatch travail compris. Les 144 rotations vivraient dans le même buffer. **La borne haute n'est pas étayée** ; la mesure qui la donnerait (micro‑banc de N dispatches triviaux dans un buffer unique, N ∈ {1, 32, 144, 252}, ~30 lignes, ~2 min) n'existe pas encore.

4. **La position du point LLVQ sur la courbe débit‑distorsion.** Toute la valeur du papier en dépend, et le plan ne peut pas la prédire. L'attente pré‑enregistrée est qu'on gagne à gauche et qu'on perde à droite, sans savoir où est le croisement. **Et si la marge tombe sous 2σ apparié, aucune claim de supériorité n'est soutenable par nos propres règles** (§4.4).

5. **Les layouts L≤3 et L≤4 ne sont pas mesurables sur le fichier publié.** `L` est le nombre de magnitudes distinctes du point de Leech, une propriété du code choisi **à la quantification** : on ne peut pas le plafonner a posteriori sur des index déjà écrits. Or ce sont les seules configurations LLVQ qui passeraient sous les 4,500 b/poids du q4 **en RAM** en gardant un stride fixe (L≤4 = 4,667 ; L≤3 = 3,667). Un run de requantification coûte ~4 h machine en local, et un banc de vitesse sur un artefact L≤3 n'est publiable qu'accompagné de sa perplexité et de son MMLU (~5,5 h de plus). **À ne pas engager avant que la qualité du q4 soit connue.**

6. **La comparabilité du banc noyau MLX.** Le protocole (un seul graphe, un seul `mx.eval`) est validé exécutable, mais rien ne garantit que MLX ne réordonne ni ne fusionne un graphe de 252 opérations, ni que son noyau f16 pour une activation 1×d_in emprunte le même chemin que notre `tv_f16` tuilé, ni que `mx.quantized_matmul` passe par le chemin gemv batch 1 sur les 252 formes. Le contrôle proposé (comparer à la somme des temps individuels) peut lui‑même être trompeur si le plancher de soumission domine. **Et `thesis` ne soustrait pas le surcoût de soumission** alors que `Kernel::overhead` existe : à la valeur prédite pour q4 (6‑8,5 ms), un terme additif commun pèse proportionnellement 1,7× plus et **comprime mécaniquement l'avantage q4**.

7. **La reproductibilité par un tiers.** La commande publiée reproduit la **méthode**, pas les **octets** (déplacement du shard C4 de 00000 à 00001, conteneur LVQ1 → LVQ2/LVQ3), et `Checkpoint::fetch` n'accepte qu'un repo id Hugging Face — **pas de chemin local**. Un tiers hors ligne ne peut exécuter **aucun bras sauf le scellé**. À écrire dans la section reproductibilité, à côté de l'avertissement C4.

8. **La provenance de l'objet adverse.** `~/qwen3-4b-mlx-q4/` n'a **ni README ni chat_template**, et son `tokenizer_config.json` est un stub de 289 o — alors que `mlx_lm 0.24.0::convert()` appelle `tokenizer.save_pretrained` **puis** `create_model_card`. La version qui a produit ce fichier n'est pas établie. **Contrôle à 5 min et 2,3 Go** : relancer la commande documentée avec mlx_lm 0.24.0 épinglé dans un répertoire neuf et comparer les sha256 — la quantification MLX étant un RTN déterministe, une égalité retire la question. **À faire avant de dépenser 6,8 h de MMLU sur cet objet.**

9. **Le choix de cadrage de la table principale** : titrer avec B‑proj (comparaison de codebooks à périmètre égal, qui nous flatte) ou B‑full (le produit réel). Le plan **impose de mesurer les deux et de publier l'écart** ; il ne choisit pas lequel titre. Décision éditoriale.

10. **Ce que notre harnais mesure du q4 est la RECONSTRUCTION, pas l'arithmétique fusionnée de MLX.** Le contrôle L3 borne l'écart, il ne l'élimine pas. Si `mx.quantized_matmul` n'accumule pas comme dequant‑puis‑matmul f16, la qualité attribuée au q4 n'est pas celle qu'un utilisateur MLX obtient — et la mesurer dans MLX ferait perdre le seul dispositif qui garantit l'iso‑conditions. **C'est le nœud de la demande, et il se déclare plutôt qu'il ne se défait.**
