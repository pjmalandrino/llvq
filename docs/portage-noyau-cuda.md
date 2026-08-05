# Spécification du portage CUDA du noyau fusé LLVQ

**Cible : NVIDIA L40S (sm_89, Ada, 48 Go). Périmètre : le chemin de production `Slot32` de `bin/thesis`. Statut : aucune mesure CUDA n'existe — tout chiffre de vitesse *CUDA* dans ce document est une arithmétique de structure, jamais une performance. Les chiffres Metal du lot K−1 (2026-08-05, §3.2, §3.7, §4.6bis, §6.1) sont, eux, des mesures, et deux d'entre elles contredisent ce que cette spécification prédisait : le padding de §3.2 ne gagne rien, et le protocole de §4.6 ne suffit pas.**

---

## 0. Ce qu'on porte, et pourquoi ça vaut le coup

### 0.1 La contribution, énoncée précisément

Un **décodeur Leech multi-coquilles fusé au matvec** : un noyau qui lit des poids empaquetés sur Λ₂₄ (union des coquilles m ≤ 12, 384 classes) et produit le produit scalaire sans jamais matérialiser un poids en mémoire.

L'annexe C du papier déclare l'inverse de trois façons : leur noyau CUDA ne traite qu'**une seule coquille (M = 3)** « pour la simplicité », il est **plus lent que QTIP**, et les auteurs qualifient l'optimisation bas niveau de « largement orthogonale » à leur contribution. Le noyau multi-coquilles qu'exige le régime 2 bits/poids n'existe donc nulle part — ni chez eux, ni ailleurs.

Le nôtre existe. Il bat le FP16 de **2,09×, plage [2,05–2,11]**, sur les 252 projections du modèle entier — `Slot32` à 5,510 b/poids, 1 105 920 lignes vérifiées contre une référence f64 avant toute mesure (§3.2, §6.1, journal `docs/mesures/k1-metal-2026-08-05.txt`). Il tourne sur du Metal.

⚠️ **Le rapport est formé round par round**, puis résumé par sa médiane et sa plage sur les 5 rounds gardés — jamais en divisant deux minima, qui mêleraient deux rounds n'ayant jamais coexisté. Toute revendication qui s'appuie sur ce nombre doit citer la plage, pas le point. En `float4` le même banc rend **2,15× [2,12–2,19]**. C'est ce run-là, et lui seul, qui a un journal dans le dépôt.

🏷️ **Le « 2,06–2,08× » qui figurait ici est antérieur et n'est plus le chiffre à citer.** Il vient du banc **à deux bras** (`docs/fiche-4b.md` §6.4, désormais étiquetée section historique), agrège deux invocations distinctes du binaire — ce que §4.6bis disqualifie — et sa fourchette est plus étroite que la dispersion mesurée depuis : trois invocations consécutives du banc **non modifié** rendent 2,029× puis 2,050× puis 2,080×.

### 0.2 Ce que le portage débloque, et que Metal ne débloquera jamais

1. **La reproductibilité par un tiers.** `docs/plan-de-test-v2-cuda.md` §4 classe le gate G6 en non-prouvable n° 6 : « le 2,07× est Metal et ne peut être reproduit sur cette cible par personne ». Un lecteur du papier n'a pas de Mac ; il a un job GPU. Le port transforme un chiffre invérifiable en un chiffre rejouable pour ~0,30 $.
2. **La fermeture de l'angle hostile n° 1.** `fiche-4b.md` §6.2 le reconnaît sans détour : « le 2,07× est un rapport contre un noyau écrit par le même auteur », jamais confronté à MPS, MLX ou Accelerate. Sur CUDA, **cuBLAS existe et s'appelle en dix lignes**. C'est le seul gain méthodologique gratuit du portage.
3. **Un instrument que Metal refuse.** `llvq-metal/src/lib.rs:17-23` documente l'infirmité : metal-rs 0.29 n'expose pas `GPUStartTime`, donc tout est chronométré à l'horloge murale. Les événements CUDA donnent un temps device, et `LaunchArgs::record_kernel_launch` donne un temps **par matrice** — donc le profil par forme (q/k/v/o/gate/up/down) que le dossier n'a jamais eu.
4. **Un chiffre du pic mémoire qui soit une mesure.** `fiche-4b.md` §6.8 **rétracte** le « 335 Go/s ≈ 93 % du pic » parce que les 400 Go/s du M3 Max sont `SUPPOSE`. Un noyau de copie de dix lignes, dans le même job, sur la même carte, referme ce trou définitivement.

### 0.3 Ce que le portage n'achète pas — à écrire avant d'écrire une ligne

- **Aucun tok/s.** Les trois obstacles de `fiche-4b.md` §6.10 sont indépendants du backend et intacts : `Qwen3::generate` n'a pas de cache KV (`llvq-llm/src/model.rs:379-381`, « No KV cache: each step re-runs the whole prefix ») donc **le noyau n'a littéralement aucun appelant** ; aucune implémentation GPU de `Rotation` n'existe, sur aucun backend, et elle serait payée par le seul bras LLVQ (144 par token) ; le prefill exige un second chemin dense.
- **Aucune réponse au 4 bits.** `Slot32` coûte 5,376 b/poids (métrique étroite) en RAM. À convention identique poids seuls, `fiche-4b.md` §5.3 donne **6,5245 contre 4,5006** pour le q4, soit ×1,45 contre nous ; la fourchette selon `group_size` et périmètre est **×1,16 à ×1,53**. **Ne jamais reposer « 5,51 contre 4,50 »** : c'est un mélange de métriques que le dossier a déjà corrigé une fois.
- **Le chiffre reste un rapport sur les projections seules.** Minorant du rapport ALU/mémoire pur (tout terme additif commun comprime le rapport, §6.1) ; majorant du rapport de bout en bout (1,88× analytique avec le lm_head, §6.6). Les deux sont vrais de quantités différentes, et **une table qui n'en imprime qu'une sera lue comme l'autre**. À porter dans l'en-tête de la table publiée, pas en note.

### 0.4 Le jalon gratuit qui devait précéder la décision — fait le 2026-08-05

`docs/plan-de-test-v2-cuda.md` §4.4 posait une règle explicite : faire tourner `matvec_g32` et `matvec_flat` sur le **modèle entier** en Metal — les shaders existent (`llvq-metal/src/bin/matvec.rs`), ~1 h de dev, 3 min de run, 0 $. Cela referme le trou de `fiche-4b.md` §6.7 (« Grouped32 et Flat32 n'ont jamais tourné sur le modèle entier ») et donne la courbe bits↔vitesse **sur un seul protocole et un seul objet**, au lieu de mélanger les chiffres d'une couche isolée (`gate_proj`) et ceux du modèle entier.

S'y ajoutait un second jalon local à 0 $, identifié par ce travail de spécification : **tester le padding de tuile contre les conflits de bancs sur Metal** (§3.2), au motif que « si le noyau Metal paie déjà le conflit, le correctif améliore le 2,07× publié ».

**Les trois jalons ont tourné (lot K−1, §5). Ce paragraphe avait raison sur le principe et tort sur le pronostic :**

- le padding **ne gagne rien** sur Apple ; ce qui gagne, et des deux côtés, c'est la largeur de chargement (§3.2) ;
- la courbe bits↔vitesse existe désormais sur un seul protocole, et elle est **brutalement non linéaire** (§6.1) ;
- le plafonnement `L ≤ 4` est confirmé au quatrième chiffre, et son statut logique est plus fort qu'annoncé : un **majorant inconditionnel**, pas une probabilité (§3.7).

Le jalon s'est donc remboursé, mais pas par où on l'attendait : il a économisé un balayage de padding sur carte louée au lieu d'améliorer le chiffre publié.

---

## 1. Le noyau Metal, spécifié

Tout ce qui suit est vérifié dans le code. Assez pour réécrire sans lire une ligne de MSL.

### 1.1 Périmètre exact

Deux fonctions, et deux seulement :

| élément | fichier:lignes | taille |
|---|---|---|
| `slot_dot` — décodeur d'un bloc | `llvq-metal/src/lib.rs:543-601` | 59 l |
| `tv_slot` — matvec tuilé | `llvq-metal/src/bin/thesis.rs:104-148` | 45 l |
| `tv_f16` — bras témoin | `llvq-metal/src/bin/thesis.rs:65-98` | 34 l |
| `ext24` — extraction 24 bits | `llvq-metal/src/lib.rs:527-531` | 5 l |
| `ClassRec` (miroir MSL) | `llvq-metal/src/lib.rs:438-445` | 9 l |

Tout le reste de `llvq-metal` est de la plomberie de mesure ou des décodeurs d'autres layouts (`decode_payload` lib.rs:469-521, `cursor_g32` lib.rs:607-628) qui n'entrent **pas** dans le 2,07× : ils ne sont appelés que par `matvec.rs` et `decreal.rs`. **Ne pas les porter.**

### 1.2 Le layout `Slot32`, champ par champ

Un bloc = 24 poids. Son enregistrement, à partir du bit 0 :

```
[ classe : 9 bits ][ gain : g bits ][ smask : 24 bits ][ m₁ : 24 ][ m₂ : 24 ] … [ m_{L−1} : 24 ]
```

- `classe` ∈ [0, 383] : index dans la table (0 = l'origine, `1 + ci` = classe `ci` dans l'ordre v1).
- `gain` : g = 1 dans l'artefact publié. **Le shader code g = 1 en dur** (§1.7).
- `smask` : bit *i* = 1 ⇔ la coordonnée *i* du bloc est négative. Espace de **slots**, pas de rang.
- `m_k` : bit *i* = 1 ⇔ le slot *i* est au niveau de magnitude *k*. Le niveau 0 (le plus peuplé) est **implicite** — c'est le complément de l'union des autres. Les masques sont **disjoints** par construction de l'encodeur (`llvq-artifact/src/runtime.rs:534-542`).

**Largeur** : `width_slot = 9 + g + 24·L`, où L est le nombre de niveaux de la classe (`llvq-artifact/src/runtime.rs:130-131`). Avec g = 1 :

| L | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| bits | 34 | 58 | 82 | 106 | **130** |
| octets | 5 | 8 | 11 | 14 | **17** |

L'origine (classe 0) fait **9 + g = 10 bits** et n'a ni `smask` ni masques : l'encodeur sort après l'en-tête (`runtime.rs:507-509`).

`L ≤ 5` est verrouillé à l'exécution par `llvq-search/src/fastdec.rs:334-337` (`assert!(lv.len() <= MAX_LEVELS)`, exécuté à chaque `FastDecoder::new()`). **Ce qui n'est verrouillé nulle part, c'est le lien entre cette borne et la fenêtre de lecture du noyau** — voir §4.4.

### 1.3 Boutisme

Le flux est **LSB-first à l'intérieur de l'octet, octets croissants** : le bit *i* est `data[i/8] >> (i%8) & 1` (`BitSink::push` runtime.rs:643-652, `BitCursor::read` :666-674, épinglé par `the_bit_order_is_lsb_first`, `llvq-artifact/tests/runtime_format.rs:170-179`).

Conséquence : réinterpréter le tableau d'octets en `const uint32_t*` sur une machine little-endian (x86-64, aarch64) donne exactement les mots que le shader Metal lit. **C'est le seul endroit du portage où une erreur passerait tous les tests CPU et casserait sur GPU.**

### 1.4 Adressage groupé

Les blocs sont groupés par 32, dans l'ordre plat `b = row·nblocks + j`.

```
g      = b >> 5
base   = bases[g]
stride = (bases[g+1] − base) >> 5          // octets
byte   = base + (b & 31) · stride
```

`bases` a `ngroups + 1` entrées, la dernière valant `data.len()`. `stride` est le `ceil(width/8)` du plus large bloc du groupe. Le groupe partiel de fin est **rembourré à 32 lanes** (`runtime.rs:413`), donc la division par 32 est toujours exacte : aucun cas particulier.

**Propriété d'alignement** : `base` part de 0 et croît de `32 × stride` (`runtime.rs:432`), donc toute base de groupe est multiple de 32. D'où `sh = 8·(byte & 3) ∈ {0, 8, 16, 24}`, jamais plus. Un pointeur aligné 4 octets suffit ; `cudaMalloc` garantit 256.

⚠️ **Un warp ne couvre PAS un groupe.** `nblocks` n'est pas multiple de 32 (2560/24 = 106, 9728/24 = 405, 4096/24 = 170), donc pour la plupart des lignes les 32 blocs `j = 0..31` d'un warp **chevauchent deux groupes**, avec deux bases et deux strides. Toute optimisation qui hisse `bases[g]` ou `stride` hors de la boucle en supposant l'uniformité est **fausse**.

### 1.5 La fenêtre de 5 mots, et sa preuve

`slot_dot` lit **cinq `uint32` consécutifs** (160 bits) à partir de `byte >> 2`. Le pire cas est `sh + width = 24 + 130 = 154 ≤ 160` (lib.rs:556-559).

Les décalages ne dégénèrent jamais : `fs = sh + 10 ∈ [10, 34]`, donc `64 − fs ∈ [30, 54]`, jamais 0 ni 64. `ext24` n'est appelé qu'aux offsets littéraux 24/48/72/96.

**Corollaire obligatoire** : le tampon de poids porte **20 octets de rembourrage** après la fin du flux (`thesis.rs:274-275`), non comptés dans les octets facturés (`thesis.rs:285-288`). Sur Metal, une sur-lecture est bénigne ; sur CUDA c'est une `CUDA_ERROR_ILLEGAL_ADDRESS` qui tue le contexte, au milieu d'un job facturé. À porter littéralement, plus un assert `n_blocks·stride + 20 <= alloc_len`.

### 1.6 Décodage d'un bloc — les huit étapes

1. `g = b>>5 ; base = bases[g] ; stride = (bases[g+1]−base)>>5 ; byte = base + (b&31)·stride`
2. `w = byte>>2` ; charger `w0..w4` ; `sh = 8·(byte & 3)`
3. `lo = (w1<<32)|w0` ; `hi = (w3<<32)|w2`
4. En-tête lu **en place** : `hdr = (lo>>sh) & 0x3ff` ; `id = hdr & 0x1ff` ; `gain = hdr>>9`
5. `fs = sh+10` ; `pay_lo = (lo>>fs)|(hi<<(64−fs))` ; `pay_hi = (hi>>fs)|(w4<<(64−fs))`
6. `smask = pay_lo & 0xffffff` ; `m_k = (nlev > k) ? ext24(pay, 24k) : 0` pour k = 1..4
7. **24 tours fixes** : `v = (m1&bj)?v1 : (m2&bj)?v2 : (m3&bj)?v3 : (m4&bj)?v4 : v0` ; `v = (smask&bj) ? −v : v` ; `d[j&3] = fma(v, xb[j], d[j&3])`
8. Retour : `((d0+d1)+(d2+d3)) · gscale[gain]`

Aucun état sériel, aucun `popcount`, aucun `ctz`, aucune boucle par niveau, aucune lecture mémoire dans la boucle hormis `xb[j]` en mémoire partagée.

**Le bloc origine ne demande aucun cas particulier** : l'entrée 0 de la table a `len = 1` et `vals = [0;5]`, donc les quatre masques sont forcés à 0, tous les slots prennent 0.0f, et le bit de signe produit −0.0f, additivement neutre. Le décodeur Rust, lui, sort tôt (`runtime.rs:295-297`) — la divergence est apparente, pas réelle.

⚠️ **Cas de lecture hors enregistrement, inoffensif mais porteur.** Pour une origine (10 bits), `smask` lit quand même les bits 10..33 du créneau ; si le groupe entier est composé d'origines, `stride = 2` octets et ces 24 bits appartiennent aux **blocs suivants**. Sans conséquence (`len = 1` force tout à 0), mais cela tue deux « améliorations » qu'un porteur tentera : valider `smask` en supposant qu'il appartient au bloc, et **matérialiser** les poids au lieu de les consommer en FMA (un tel noyau écrirait des −0,0 sur les slots à zéro, et un diff bit-à-bit avec la référence Rust échouerait).

### 1.7 La table de classes

384 entrées × 32 octets = **12 288 octets** (`GpuClassRec`, lib.rs:361-375 ; construction `gpu_class_table`, lib.rs:380-419 ; épinglée par `llvq-metal/tests/gpu_table.rs`).

**`slot_dot` n'utilise que `vals[0..5]` et `len`.** `counts`, `nz`, `zlev`, `sbase` servent aux décodeurs des autres layouts. Un enregistrement CUDA dédié tient en **24 octets** (`float vals[5]; uint32 len;`), soit 9 216 octets.

**Le point qui neutralise l'objection de l'annexe G du papier** : `vals[k] = valeur_entière / √(16·m)` est précalculé **par classe** (lib.rs:392). Chaque bloc décodé est donc déjà unitaire, quelle que soit sa coquille. L'annexe G objecte qu'une union de coquilles impose un rééchelonnage entre produits scalaires ; ici il n'existe pas — le coût du multi-coquilles est un `float` de plus dans la table, pas une instruction de plus dans la boucle. **C'est ce qui distingue ce noyau du leur.**

Ordre canonique des niveaux (à ne pas re-dériver) : comptes décroissants, égalités départagées par |valeur| décroissante (`llvq-search/src/fastdec.rs:333`). Le niveau 0 est le plus peuplé, et c'est lui qui est implicite.

### 1.8 Schéma d'exécution de `tv_slot`

- **Un SIMD group (32 lanes) par ligne de sortie** ; threadgroup de **256 threads = 8 lignes**. Dispatch de `d_out × 32` threads (thesis.rs:321, :382). `row = gid >> 5`.
- **Tuilage de l'activation** : 128 blocs = 3072 colonnes = **12 288 octets** de mémoire partagée, en f32 (TILE_BLOCKS, thesis.rs:54, :59-60, :322). `ntiles = ceil(nblocks/128)`.
- Par tuile : **barrière**, remplissage coopératif `for (i = tid; i < n; i += tgs) xs[i] = x[c0+i]`, **barrière**, puis chaque lane traite `j = jlo + lane, +32, …` avec `xb = xs + (j−jlo)·24`.
- **Deux barrières, pas une** (thesis.rs:128-130 ; commentaire explicite :81-83) : la seconde ordonne le remplissage vis-à-vis des lecteurs, la première empêche le remplissage de la tuile suivante d'écraser `xs` pendant que des lanes retardataires lisent encore la précédente. `ntiles` ne dépend que de `P.nblocks`, donc les barrières sont uniformes — condition nécessaire en CUDA comme en Metal.
- **Réduction** : `acc = simd_sum(acc)` sur les 32 lanes.
- **Épilogue de queue** : lane 0 **seule** lit `tail[row·tail_w + i]` et `x[nblocks·24 + i]` en mémoire **globale** (≤ 23 colonnes), et écrit `y[row] = acc·rscale[row] + tv`. L'échelle de ligne s'applique à `acc` seul ; la queue s'ajoute après.

Les 8 SIMD groups d'un threadgroup travaillent sur 8 lignes différentes mais la **même** tuile de colonnes : les 12 Ko sont amortis 8 fois.

Origine du tuilage : sans lui, les 36 `down_proj` (d_in = 9728 → 38 912 o) dépassaient la limite Metal de 32 Ko (thesis.rs:26-32). **Sur CUDA la contrainte n'existe plus — raison de plus de ne pas y toucher** (§3.5).

### 1.9 Les neuf liaisons

| # | tampon | contenu |
|---|---|---|
| 0 | `words` | u32, flux Slot32 + 20 o de rembourrage |
| 1 | `bases` | u32[ngroups+1] |
| 2 | `tab` | 384 enregistrements, **partagé par les 252 matrices** |
| 3 | `gscale` | f32[2], les centroïdes de gain |
| 4 | `rscale` | f32[d_out], échelles de ligne |
| 5 | `tail` | f32[d_out·tail_w], ou f32[1]={0} si tail_w = 0 (thesis.rs:298 — **le tampon factice est obligatoire**) |
| 6 | `x` | f32, l'activation, **partagée par les deux bras** |
| 7 | `y` | f32[max d_out], **buffer de sortie unique** |
| 8 | `Params` | `{d_in, d_out, nblocks, tail_w}`, 4×u32, `#[repr(C)]` |
| tg0 | `xs` | 12 288 octets |

`tv_slot` n'utilise **ni** `P.d_in` **ni** `P.d_out` : la ligne vient de la grille, `c0` de `nblocks·24`. Ils ne sont là que pour partager la structure avec `tv_f16`.

### 1.10 Le bras témoin FP16 — ce qu'il est, et ce qu'il n'est pas

`tv_f16` : un SIMD group par ligne, chargements `half4` (8 o/lane, 256 o contigus par warp), même tuilage à 3072 colonnes, même mise en cache f32 de l'activation, un seul accumulateur f32, quatre multiplications-additions par itération, `simd_sum`, lane 0 écrit. **Ni échelle de ligne, ni queue** : elles sont déjà dans les poids.

**Ce n'est pas le checkpoint FP16 du modèle.** `w16 = f16_bits(w)` où `w` est la reconstruction **f64 des blocs LLVQ**, dans la **base tournée** (thesis.rs:237-253). Les deux bras calculent le même produit mathématique à l'arrondi près : c'est un baseline de **coût**, pas de qualité. À reprendre tel quel, avec l'aveu.

### 1.11 Ce que fait l'hôte

| étape | code | portable ? |
|---|---|---|
| lecture de l'artefact | `read_header`, `read_matrix_raw` | **oui, tel quel** |
| transcodage rang → Slot32 | `transcode(fd, table, indices, gains, Layout::Slot32)`, `runtime.rs:361-438` | **oui, tel quel** — mono-thread, sans dépendance externe |
| table des 384 classes | `gpu_class_table(&fd)`, lib.rs:380-419 | oui, mais elle vit dans un crate `cfg(macos)` → à extraire (§5, K0) |
| échelles, centroïdes, queue | f64 → f32, lus tels quels | oui |
| reconstruction f64 + arrondi f16 | thesis.rs:237-253 | oui |
| référence f64 par ligne | thesis.rs:258-271 | oui |

**Ce que l'hôte NE fait pas** : la rotation d'incohérence. Le banc tire une activation gaussienne quelconque et compare deux noyaux sur les mêmes octets (thesis.rs:210-212, aveu explicite :22-24). Le portage hérite du trou tel quel.

### 1.12 Le mono-verrou du bit de gain — défaut hérité, à ne pas recopier

`slot_dot` code en dur **1 bit de gain** : `id = hdr & 0x1ff`, `gain = hdr >> 9`, `fs = sh + 10` (lib.rs:567-569). `matvec.rs:519` et `decreal.rs:154-158` l'assertent ; **`thesis.rs` calcule `gain_bits` (:228) et ne l'assert jamais**.

Le format supporte g = 2 (`two_bit_gains_roundtrip`, `runtime_format.rs:147-163`) et g = 0 si `centroids.len() == 1` — et à g = 0 même la taille de l'en-tête change. Un artefact non conforme décoderait faux en silence, rattrapé seulement par le seuil 1e-3, qui est un détecteur d'incendie et non une garde. **Le port assert `gain_bits == 1`, ou spécialise à la compilation NVRTC (§2.3).**

---

> ## ✅ Inventaire vérifié sur la carte le 2026-08-04 — job `6a724dfba00abefd4b292856`
>
> Un mini-job de trente secondes a tranché les questions ouvertes de ce
> document. **Toutes vont dans le bon sens.**
>
> ```
> NVIDIA L40S, compute_cap 8.9, 46068 MiB, driver 580.159.03
> libnvrtc.so.12      → /usr/local/cuda/targets/x86_64-linux/lib/
> libcublas.so.12     → idem
> libcublasLt.so.12   → idem
> libcuda.so.1        → /usr/lib64/
> nvcc                → absent (attendu, image runtime — sans importance)
> ```
>
> - **`libnvrtc` est présente dans l'image d'EXÉCUTION**, pas seulement dans
>   celle de build. La voie A est donc praticable telle quelle : une seule
>   reconstruction d'image pour ajouter le binaire, puis toute la mise au point
>   du noyau se fait par mini-jobs sans jamais y retoucher.
> - **cuBLAS est présente**, et `ldd` montre que nos binaires actuels la lient
>   déjà — candle l'apporte. La baseline défendable de §4.5 ne demande aucune
>   installation.
> - **La carte annonce `compute_cap 8.9`**, exactement ce que fige
>   `CUDA_COMPUTE_CAP=89`. Code natif, pas de JIT PTX.
>
> ⚠️ **Correction de cadrage de ce document.** Le §2.4 justifie la voie NVRTC
> par « aucune itération locale n'est possible, donc chaque essai est un job
> distant coûteux ». La seconde moitié est fausse : un mini-job coûte quelques
> centimes et quelques minutes — le pilote complet en a coûté 0,08 $ pour six
> minutes. **On mesure au lieu de raisonner**, et l'estimation de 10,5 à 18,5
> jours du §5 est à réviser vers **4 à 7 jours** : une bonne part des lots
> était de la précaution contre une contrainte qui n'existe pas. Ce qui reste
> vrai, et qui suffit à retenir NVRTC : le binaire Rust, lui, doit être dans
> l'image, donc son premier ajout coûte une reconstruction.

## 2. La route technique retenue

### 2.1 La voie : crate autonome `llvq-cuda` + NVRTC à l'exécution

Quatre voies ont été examinées. **Retenue : un crate `llvq-cuda` sur le modèle de `llvq-metal`, compilant ses noyaux par NVRTC au démarrage, avec la source CUDA surchargeable depuis un répertoire monté.**

| voie | verdict |
|---|---|
| **A. NVRTC à l'exécution** | ✅ retenue. Compilation en centaines de ms, une seule construction d'image pour toute la campagne. |
| **B. nvcc au build (`bindgen_cuda`)** | Techniquement possible — `bindgen_cuda` 0.1.6 lit `CUDA_COMPUTE_CAP` d'abord et ne tombe sur `nvidia-smi` qu'à défaut, donc le builder sans GPU n'est **pas** un obstacle. Éliminée économiquement : chaque édition = un rebuild d'image de 40-70 min, sur une étape qui a déjà été tuée par SIGKILL (`ops/Dockerfile.cuda:31-43`). **Réservée à la livraison finale**, pas à la mise au point. |
| **C. `CustomOp1::cuda_fwd` de candle** | C'est la route d'**intégration**, pas celle du banc — elle se pose *au-dessus* de A ou B, et bute sur les trois obstacles de §0.3. Hors périmètre. |
| **D. Crate autonome sans candle** | ✅ combinée à A. `llvq-artifact`, `llvq-search`, `llvq-core` sont sans dépendance externe ; le banc n'a besoin que de cudarc. Aucun lien LTO lourd ajouté au Dockerfile. |

### 2.2 API cudarc — vérifiée dans la source, pas de mémoire

Toutes les signatures ci-dessous ont été relues dans `~/.cargo/registry/src/index.crates.io-*/cudarc-0.19.8/`.

| besoin | API | fichier:ligne |
|---|---|---|
| compiler | `nvrtc::compile_ptx_with_opts(src, opts)` | `src/nvrtc/safe.rs:116` |
| options | `CompileOptions { arch, options, maxrregcount, include_paths, … }` | `:231-243` |
| charger | `CudaContext::load_module(ptx)` → `CudaModule::load_function` | `src/driver/safe/core.rs:2173`, `:2215` |
| lancer | `CudaStream::launch_builder`, `LaunchConfig{grid_dim, block_dim, shared_mem_bytes}` | `src/driver/safe/launch.rs:63`, `:15-24` |
| chronométrer | `CudaContext::new_event(flags)`, `record`, `elapsed_ms` | `core.rs:551`, `:587`, `:603` |
| attributs carte | `CudaContext::attribute()` | `core.rs:362` |
| attributs fonction | `num_regs`, `local_size_bytes`, `set_attribute`, `occupancy_max_active_blocks_per_multiprocessor` | `core.rs:2407`, `:2422`, `:2442`, `:2278` |
| symbole `__constant__` | `CudaModule::get_global` | `core.rs:2238` |
| baseline f16 | `Gemm<half::f16>` (`gemm_ex`, force `CUDA_R_16F` / `CUBLAS_COMPUTE_32F`) | `src/cublas/safe/gemm.rs:63` |

**Quatre pièges d'API, chacun coûte une demi-journée si on ne le sait pas :**

1. **`new_event(None)` crée un événement SANS chronométrage.** `core.rs:555` : `flags.unwrap_or(CU_EVENT_DISABLE_TIMING)`. Il faut passer `Some(CU_EVENT_DEFAULT)` explicitement, sinon `elapsed_ms` échoue.
2. **`arch` est `Option<&'static str>`** (`safe.rs:240`) : une architecture calculée à l'exécution doit passer par `options: Vec<String>`, qui est déroulé *après* `--gpu-architecture=` (`:276-282`) et le surcharge donc.
3. **Sans `arch`, NVRTC compile pour `compute_75` par défaut** (doc NVIDIA). Ça tourne, ça ne dit rien de sm_89, et rien ne le signale. Garde directe : imprimer `binary_version()` (`core.rs:2438`) et asserter 89.
4. **`use_fast_math: Some(true)` n'émet que `--fmad=true`** (`safe.rs:264-266`), qui est de toute façon le défaut NVRTC. Ce champ est un no-op dans les deux sens. Pour la vraie sémantique fast-math, passer l'option par `options`.

**Il n'existe pas de `cublasHgemv`** : `impl Gemv` n'existe que pour f32 (`gemv.rs:35`) et f64 (`:63`) ; la liste des symboles gemv wrappés est {S,D,C,Z}. Le baseline cuBLAS d'une matvec f16 est **nécessairement un GEMM à n = 1**. Voir §4.5.

### 2.3 Où vit le code CUDA

- `llvq-cuda/kernels/llvq_slot.cuh` : `ClassRec`, `ext24`, `slot_dot`, la réduction de warp.
- `llvq-cuda/kernels/thesis.cu` : `tv_slot`, `tv_f16`, le noyau « sol ».

Le harnais **concatène** les deux avant de les passer à NVRTC, comme Metal fait `format!("{}{}", PAYLOAD_MSL, SRC)` (thesis.rs:195).

⚠️ **La justification n'est pas l'API.** `CompileOptions::include_paths` existe (`safe.rs:238`) et émet `--include-path=` (`:270-272`) : un `#include "llvq_slot.cuh"` compilerait. La raison de concaténer est la **preuve** : un `#include` résolu depuis un répertoire monté rend l'empreinte non close — le `.cuh` peut changer sans que la chaîne passée à NVRTC change. La concaténation donne une chaîne unique, hashable, égale à ce que le driver voit. Le manifeste porte le **sha256 de cette chaîne**, et le binaire l'imprime.

Les constantes structurelles (`GAIN_BITS`, `TILE_BLOCKS`, `TILE_STRIDE`, `CLASS_BITS`) sont émises en `#define` par l'hôte, jamais dupliquées entre le Rust et le `.cu` — c'est le défaut D5 de Metal (`TILE_BLOCKS` défini deux fois, thesis.rs:54 et :59, sans lien) qu'on ne recopie pas.

### 2.4 La boucle d'itération — la décision la plus rentable

**Aucune itération locale n'est possible** : la machine de dev est un Mac. Toute compilation, exécution et mesure CUDA vit dans un job HF.

Sans NVRTC + source surchargeable, chaque essai de padding, de taille de tuile ou de `maxrregcount` demande un rebuild d'image (40-70 min, non facturé mais sérialisant, avec un historique de SIGKILL). Avec, la campagne devient :

1. Écrire N variantes de `.cu` dans un bucket/volume.
2. Monter, lancer **un** job.
3. Le binaire prend un **répertoire** de noyaux, charge l'artefact **une fois**, et imprime une ligne de résultat par variante.

Le coût marginal d'une variante tombe à quelques centaines de millisecondes de compilation. **C'est une contrainte d'architecture du harnais, à poser avant d'écrire, pas une optimisation.**

### 2.5 Contraintes de compilation Rust

```toml
# llvq-cuda/Cargo.toml — miroir de llvq-metal/Cargo.toml
[target.'cfg(target_os = "linux")'.dependencies]
cudarc = { version = "0.19", ... }
llvq-artifact = { path = "../llvq-artifact" }
```

Trois raisons, dont deux corrigent des croyances erronées :

1. **Le `build.rs` de cudarc exécute `nvcc --version`** sous `cuda-version-from-build-system` (`build.rs:154-168`, `panic!` à :168) et émet `-l dylib=cuda/nvrtc/cublas` (`:190-213`). Sur macOS, rien de tout ça n'existe. Le target-gating règle la question ; sans lui, `cargo check` casse sur le poste de dev.
2. **Le panic d'unification de features n'est PAS le problème.** Les features par défaut de cudarc contiennent `fallback-dynamic-loading` (Cargo.toml:105-118), pas `dynamic-loading` ; le `panic!` de `build.rs:93` n'est armé que si `dynamic-loading` est explicitement activée. Ajouter cudarc à un nouveau crate ne casse rien de ce côté.
3. **Ne PAS décalquer le `compile_error!` de `llvq-metal`** (lib.rs:631-632). Le workspace liste ses membres explicitement (`Cargo.toml:3`, sans glob) et `cargo clippy --all-targets` — la porte zéro-warning de CLAUDE.md §7 — tourne sur le Mac. Hors Linux, `llvq-cuda` doit être une **coquille vide** (module et bins sous `cfg`), pas une erreur de compilation.

Règle à écrire une fois pour toutes : **un binaire qui lie candle utilise `candle_core::cuda::cudarc` ; un binaire qui ne le lie pas déclare cudarc lui-même ; jamais les deux dans le même processus.**

### 2.6 État vérifié de la chaîne d'approvisionnement

- `Cargo.lock` **est suivi par git** (`git ls-files --error-unmatch Cargo.lock` réussit) et **est dans les `allow_patterns`** de la publication du Space (`ops/run.py:480`).
- **`--locked` : les deux images l'ont.** `ops/Dockerfile.cuda:52` et `ops/Dockerfile:43` construisent tous deux avec `cargo build --release --locked`.

> 🕳️ **Deux affirmations de cette section étaient fausses ; voici d'où elles venaient.**
>
> 1. *« La ligne `Cargo.lock` du `.gitignore` est inerte et devrait être retirée par hygiène. »* **Cette ligne n'existe pas et n'a jamais existé** : le `.gitignore` du dépôt fait deux lignes, `target/` et `__pycache__/`. L'affirmation décrivait un fichier imaginé, pas lu. Rien à corriger dans le dépôt — seulement ici.
> 2. *« Ce qui manque réellement : `--locked`. `ops/Dockerfile.cuda:48` fait `cargo build` sans, vérifié par grep. »* Le défaut était réel **au moment où la phrase a été écrite**, et il a été corrigé par le commit **9de862f — celui-là même qui a ajouté cette spécification**. Le texte n'a pas suivi son propre correctif. Depuis, la ligne 48 est le **commentaire** qui justifie `--locked` et la commande est à la ligne 52, `--locked` inclus : un grep sur « Dockerfile.cuda:48 » retombe donc sur le commentaire et donne l'illusion que la phrase tient encore.
>
> **Le `--locked` réellement manquant était ailleurs** : dans `ops/Dockerfile`, l'image **CPU**, que cette spécification ne regarde jamais parce que son périmètre est le chemin CUDA. Il vient d'être ajouté (`ops/Dockerfile:43`). Leçon de la même famille que celles de `CLAUDE.md` §5 : un audit qui ne regarde qu'un fichier ne trouve le défaut que dans ce fichier-là.

---

## 3. Ce qui ne se transpose pas

### 3.1 Décidable maintenant — la table de classes n'est PAS `__constant__`

Metal passe la table en espace `constant` (thesis.rs:106) ; c'est un binding en lecture seule. `__constant__` sur CUDA est un banc de 64 Ko optimisé pour la **diffusion** : son coût croît avec le nombre d'adresses distinctes lues dans un warp. Or les 32 lanes lisent **32 classes différentes par construction**.

**Prescription : `const ClassRec* __restrict__ tab` en paramètre (mémoire globale, servie par le cache lecture-seule / L1). Interdire `__constant__`.**

Nuance de calibrage, pour ne pas surdimensionner le risque : la traduction *naïve* d'un pointeur Metal est déjà un paramètre pointeur, donc de la mémoire globale. Le piège n'est pas d'y tomber, c'est de faire du travail supplémentaire pour imiter Metal — cudarc expose `get_global` (`core.rs:2238`) et sa doc cite littéralement l'accès à `__constant__`, ce qui rend l'erreur facile. Coût réel du gather : un enregistrement fait 32 o alignés 32, soit ≤ 32 secteurs par warp et par bloc décodé ; la table (12 Ko, 9 Ko en la réduisant à `float vals[5]; uint len;`) reste résidente en L1 (128 Ko unifiés sur Ada).

### 3.2 Mesuré en Metal le 2026-08-05 — le padding ne gagne rien ; la largeur de chargement, si

> Cette section prescrivait un pas de tuile de 28 flottants avec chargements `float4` pour supprimer un conflit de bancs à 8 voies. **K−1(b) l'a testée. La prescription est infirmée sur Apple ; ce qui reste est un autre gain, plus simple, et qui ne change pas le rapport.**

**Le résultat d'abord.** Sept bras dans un **même processus**, tous dispatchés à chaque round dans le même ordre, 7 rounds dont 2 jetés, mémoire froide par construction, comptabilité d'octets identique pour tous les layouts LLVQ, 1 105 920 lignes vérifiées contre la référence CPU f64 au seuil 1e-5 (`docs/mesures/k1-metal-2026-08-05.txt`) :

| bras | b/poids | min ms | vs FP16 (half4, scalaire) [plage] |
|---|---|---|---|
| FP16 (half4, scalaire) | 16,000 | 21,775 | 1,00× [1,00–1,00] |
| FP16 (half4, `float4`) | 16,000 | 20,709 | 1,05× [1,05–1,05] |
| LLVQ Slot32 (scalaire@24) | 5,510 | 10,401 | 2,09× [2,05–2,11] |
| **LLVQ Slot32 (`float4`@24)** | **5,510** | **9,925** | **2,15× [2,12–2,19]** |
| LLVQ Slot32 (`float4`@28 — *le padding prescrit ici*) | 5,510 | 10,091 | 2,13× [2,06–2,17] |

> **Comment le rapport est formé, et pourquoi il ne se retrouve pas en divisant les colonnes** : le rapport est calculé **round par round**, puis résumé par sa **médiane** et sa plage sur les 5 rounds gardés. Diviser la colonne « min ms » d'un bras par celle d'un autre mêlerait deux rounds qui n'ont jamais coexisté — un lecteur qui fait 21,775 / 9,925 trouve 2,19 et non 2,15. Les ms sont là pour l'ordre de grandeur ; le rapport et sa plage sont le résultat.

Trois lectures, dans l'ordre :

1. **Le padding ne paie pas.** 10,091 contre 9,925, soit **1,7 % plus lent**, et sa plage de rapport [2,06–2,17] **recouvre entièrement** celle du `float4` dense [2,12–2,19] par le haut : rien ne le distingue (§4.6bis). Il coûte pourtant +2 048 octets de mémoire partagée par bloc et une division entière par thread et par itération dans la boucle de remplissage.
2. **La largeur de chargement paie, et des deux côtés.** +4,6 % sur LLVQ (10,401 → 9,925) et +4,9 % sur FP16 (21,775 → 20,709). Ce qui paie n'est donc pas la géométrie des bancs mais **un load 128 bits au lieu de quatre 32 bits**.
3. **Donc le rapport ne bouge pas.** À bras comparables : **2,05×** en `float4` contre `float4` (2,15 / 1,05), contre **2,09×** en scalaire contre scalaire. Les deux sont dans la dispersion l'un de l'autre. **Prendre le gain d'un seul côté truquerait le rapport** — la règle « traiter les deux bras ou aucun » écrite plus bas vient d'être exercée pour de vrai.

*(Les pourcentages du point 2 et le 2,05× du point 3 sont des grandeurs **dérivées** de la table ci-dessus, pas des lignes lues : `(10,401 − 9,925)/10,401`, `(21,775 − 20,709)/21,775`, `2,15 / 1,05`. Ils sont licites parce que les deux termes viennent du même run, du même processus et de la même comptabilité — ce que le protocole entrelacé de §4.6bis rend permis.)*

**Un confondant, déclaré et non caché.** Les deux variantes `float4` de `Slot32` sont **identiques au bit près** au noyau scalaire sur les 1 105 920 lignes (assertion dans le code, pire erreur 3,4e-8 des trois côtés). La variante `float4` du bras FP16 ne l'est pas : 3,1e-8 d'écart avec le bras FP16 scalaire, parce que sa somme est écrite en `+`/`*` et non en `fma` explicites, donc le compilateur contracte comme il veut. Les +4,9 % du bras témoin portent cette réserve ; les +4,6 % du bras LLVQ ne la portent pas.

**Ce qui reste de l'arithmétique NVIDIA — une transposition, pas un fait de ce matériel.** Dans `tv_slot`, la lane *L* traite le bloc `j = jlo + 32t + L` et lit `xb = xs + (j−jlo)·24` (thesis.rs:133-134, lib.rs:593-596). La banque NVIDIA est `adresse_en_mots mod 32`. Pour un slot *j* fixé, l'adresse vaut `24L + j`, et `24L mod 32` ne prend que **4 valeurs distinctes** sur 32 lanes (0, 24, 16, 8 — période 4, car 24×4 = 96 ≡ 0 mod 32) : conflit 8 voies sur chacune des 24 lectures, en accès scalaire.

| pas de tuile | accès | bancs distincts par phase | conflit *sous le modèle NVIDIA* | mémoire partagée |
|---|---|---|---|---|
| 24 flottants (actuel) | scalaire | 4 | 8 voies | 12 288 o |
| 24 flottants | `float4` (96 o aligné 16) | 4 starts × 4 mots | 2 voies | 12 288 o |
| 28 flottants | `float4` (112 o aligné 16) | 8 starts espacés de 4 × 4 mots = 32 | aucun | 14 336 o |
| 25 flottants | scalaire (100 o, **non aligné 16**) | 32 | aucun | 12 800 o |

Ce tableau reste juste **au tableau, pour NVIDIA**. Ce que K−1(b) établit, c'est qu'**il ne décrit pas ce matériel-ci** : sur Apple, la seule ligne qui devrait gagner le plus (28 + `float4`) est la plus lente des deux variantes `float4`. La mesure ne dit pas *pourquoi* — géométrie de bancs différente, ou noyau qui n'est simplement pas limité là — et il ne faut pas trancher à sa place.

⚠️ **Cela ne préjuge pas de CUDA.** La géométrie des bancs y est documentée (32 bancs de 4 octets) et l'arithmétique ci-dessus y est valide. Ce qui tombe, c'est le **statut** du gain : il n'est plus « acquis, reste à quantifier », il redevient une hypothèse comme les autres. Conséquence directe sur §5 : **K7 ne peut plus budgéter un gain de padding**, et le balayage doit commencer par la largeur de chargement — le seul effet que K−1(b) a réellement mesuré, sur les deux bras, et donc le seul qui se transporte probablement.

### 3.3 Décidable maintenant — grille, réduction, garde

- **CUDA lance des blocs entiers**, Metal autorise des threadgroups non uniformes (`dispatch_threads` prend le nombre **total** de threads, lib.rs:123-126). Le portage ne marche que parce que `assert_eq!(d_out % 8, 0)` (thesis.rs:226) rend `d_out·32/256` exact — vrai sur les six formes du 4B (4096, 1024, 1024, 2560, 9728, 4096… ) et du 8B. **Conserver l'assert ; le re-libeller « CUDA lance des blocs entiers » plutôt que « confort ».**
- **Ne PAS ajouter `if (row >= d_out) return;`** : (a) un `return` avant `__syncthreads()` est un interblocage ; (b) il invaliderait le masque `0xffffffff` des primitives de warp sous l'Independent Thread Scheduling.
- **`simd_sum` → cinq `__shfl_xor_sync(0xffffffff, acc, 1<<k)`** (ou `__shfl_down_sync`, seule la lane 0 écrit). C'est la **seule** collective du chemin de production : `slot_dot` n'a ni popcount, ni ctz, ni ballot, ni atomique.

### 3.4 Décidable maintenant — sûreté des décalages

`ext24` (lib.rs:527-531) s'écrit `(off < 64u) ? ((lo >> off) | (hi << (64u − off))) : (hi >> (off − 64u))`. Sa sûreté **n'est pas structurelle** : `off` est `unsigned`, donc `off − 64u` boucle, et à `off = 24` la branche non prise calcule un décalage par 4 294 967 256 — indéfini. Elle ne tient que parce que le compilateur replie le `?:` sur des littéraux.

**Prescription : `template<unsigned OFF> __device__ inline unsigned ext24(...)` avec `if constexpr (OFF < 64)`**, ou quatre fonctions distinctes. La sûreté devient une propriété du langage.

Option ouverte : réécrire les décalages en 32 bits avec `__funnelshift_r`/`_rc` (instruction SHF native ; les variantes `_lc`/`_rc` **clampent** à 32 au lieu de masquer par 31, ce qui supprime aussi les cas dégénérés de `take` et `cursor_g32`). Gain probable, mais **la borne `sh + width ≤ 160` est serrée à 6 bits et doit être re-prouvée sur toute réécriture**.

### 3.5 Décidable maintenant — ne pas élargir la tuile

Ada autorise 48 Ko de mémoire partagée par bloc sans opt-in et ~99 Ko avec (`cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`, exposé par `CudaFunction::set_attribute`, `core.rs:2442`). Les 38 Ko qu'exigeait `down_proj` sans tuilage passeraient donc.

**Ne pas en profiter pour le chiffre publié.** Deux raisons : (a) changer le tuilage rend les deux bancs non comparables, et c'est exactement la faute de méthode que ce dépôt documente depuis les A/B de calibration ; (b) 38 Ko/bloc à 256 threads limiterait à 2 blocs/SM (512 threads résidents contre 1 536), c'est-à-dire un tiers de la latence mémoire à masquer — or ce noyau vit précisément de se glisser dans les bulles de latence. **Levier d'optimisation ultérieur, mesuré séparément, jamais un « pendant qu'on y est ».**

⚠️ Corollaire : **porter la forme `bin/thesis` (tuilée) et non `bin/matvec`.** Un portage naïf de `matvec` compilerait, tournerait, et détruirait l'occupancy sans lever la moindre erreur.

### 3.6 Décidable maintenant — pas de tensor cores

Un matvec f16 fait 2 flops par poids de 2 octets = **1 flop/octet**. À 864 Go/s cela demande 0,864 TFLOP/s, contre 91,6 TFLOPS FP32 et 181,05 TFLOPS FP16 tensor **dense** sur L40S (les 362,05 de la fiche NVIDIA sont marqués « with sparsity »). Soit **0,94 % du FP32 et 0,48 % du tensor dense**.

Tensor cores, `ldmatrix`, `__dp4a` : sans objet. `__dp4a` est de toute façon inatteignable — les valeurs de niveau sont converties en f32 avant le GPU (lib.rs:394-397) et `x` reste en f32.

À réexaminer **uniquement** si le prefill entre au périmètre : à seq > 1 l'intensité arithmétique bascule et le chemin dense redevient tensor. Deux noyaux, pas un.

### 3.7 Exige mesure — la coalescence de `Slot32`

`stride` vaut 11, 14 ou **17 octets** ; les blocs à 5 niveaux (**3,38 % du total**, `docs/mesures/k1c-rtbits-2026-08-05.txt`, bloc « niveaux de magnitude par bloc ») fixent **66,728 %** des groupes à 17 (K−1(c), compte exact plus bas). *(Citation recalée le 2026-08-05 : elle pointait sur `format-noyau.md:297-300`, mais K−1(a) a réécrit cette zone et en a retiré le « 3,4 % », qui n'y figure plus. Ancrer sur le journal de mesure plutôt que sur un numéro de ligne d'un document en cours de correction.)* Un warp lisant un enregistrement par lane couvre 31×17 + 20 = 547 octets contigus, mais chaque lane lit 20 octets qui se chevauchent : **~17-18 secteurs de 32 o par instruction de chargement, là où l'idéal coalescé en fait 4**. Le trafic DRAM reste juste (le L1 absorbe les relectures) ; c'est le coût d'**émission LSU** qui est multiplié.

Quatre sorties, dont deux s'annulent :

| option | coût en bits | changement de format |
|---|---|---|
| **(a) staging coopératif** — le warp charge les ≤ 544 o de son groupe en shared par des LDG coalescés, puis décode depuis le shared | **0** | aucun |
| (b) transposition par mot dans le groupe, stride arrondi à 4 o | 17→20, 14→16, 11→12 : +9 à +18 % | `decode_block` + round-trips à réécrire |
| (c) enregistrement uniforme de 16 o lu en un `uint4` | 5,333 b/poids **et suppression de `bases`** | impose L ≤ 4 → requantification du 4B + reprise de ppl/MMLU |
| (d) ne rien changer et mesurer | 0 | aucun |

**Conclusion de conception à prendre maintenant : préférer (a).** (b) et (c) consomment le seul gisement de bits identifié pour acheter une coalescence que (a) obtient gratuitement. Complication de (a), à ne pas rater : un warp chevauche deux groupes (§1.4), donc la plage d'octets reste contiguë mais le code d'adressage n'est plus celui de Metal.

#### Le plafonnement `L ≤ 4` — mesuré le 2026-08-05, et son statut logique corrigé

**Chiffre corrigé dans `format-noyau.md`** : le document annonçait « ~4,4 b/poids » pour un plafonnement à L ≤ 4. L'arithmétique du format disait **4,708** ; **K−1(c) mesure 4,7083**. L'arithmétique de cette section est donc confirmée au quatrième chiffre. Le « ~4,4 » **a été corrigé là-bas le 2026-08-05** — il n'y subsiste plus que comme généalogie, dans la section « Reprendre des bits sans perdre la vitesse : le plafond L ≤ 4, compté ». *(Renvoi par titre de section et non par numéro de ligne : le lot K−1 a inséré une centaine de lignes dans ce fichier, et tout renvoi chiffré s'y périme au lot suivant.)*

Mesure : `bin/rtbits` sur l'artefact réel (`~/llvq-q4b.llvq`, 981 Mo, projections seules du Qwen3-4B publié), **4 708 800 groupes de 32 blocs, soit 150 681 600 blocs**. *(Corrigé le 2026-08-05, deuxième passe : la première rédaction écrivait « 113 011 200 blocs », qui est 4 708 800 × **24** — la taille d'un bloc appliquée à un nombre de groupes. Le chiffre imprimé par `rtbits` est 150 681 600, cf. `docs/mesures/k1c-rtbits-2026-08-05.txt`.)*

> **Comptabilité de cette sous-section, à ne pas mélanger avec les autres du document** : payload + une base u32 par groupe, **stride arrondi à l'octet** — c'est-à-dire exactement ce qu'une lane lit —, rapporté aux poids quantifiés. Elle exclut queue et échelles de ligne. C'est la « métrique étroite » de §0.3, et **pas** celle des 5,510 b/poids de `bin/thesis` (§3.2, §6.1), qui facture en plus la queue f32 et les échelles de ligne f32.

| sous cette comptabilité | b/poids |
|---|---|
| `Slot32` aujourd'hui | **5,3756** |
| `Slot32` plafonné à `L ≤ 4` | **≤ 4,7083** |
| gain | **0,667 b/poids, soit 12,4 %** |

Distribution des maxima par groupe, qui est ce qui fixe le stride :

| max de niveaux du groupe | groupes | part | stride |
|---|---|---|---|
| L = 3 | **1** | — | 11 o |
| L = 4 | 1 566 710 | 33,272 % | 14 o |
| L = 5 | 3 142 089 | 66,728 % | 17 o |

Moyenne des max par groupe : **4,667 niveaux** (moyenne par bloc : 3,726). Le compte est clos sur les 4 708 800 groupes : aucun groupe n'a un max inférieur à 3.

**Recoupement par un chemin de code indépendant** : le contrôle de non-régression de `bin/matvec` — **une** couche (`gate_proj`), protocole froid à 4 copies rotatives, donc un protocole **différent** de celui du modèle entier — reporte **5,375 b/poids** sur cette couche, et **0,69× (Grouped32) / 0,90× (Flat32) / 2,20× (Slot32)**. Le 5,375 d'un banc et le 5,3756 de `rtbits` sont calculés par deux chemins qui ne partagent pas de code. ⚠️ Les rapports de vitesse de ce banc ne sont **pas** comparables aux 2,09×/0,91×/0,69× de §6.1 : une couche contre 252 matrices, et deux protocoles de refroidissement distincts. Ne jamais les mettre dans le même tableau.

🔎 **Le statut logique du 4,7083, à écrire précisément — c'est un MAJORANT INCONDITIONNEL, pas une simulation.**

- **Majoration** : `L ≤ 4` implique `width_slot ≤ 9 + 1 + 24·4 = 106 bits = 14 octets`, donc **tout** groupe a un stride ≤ 14 o. Aucune hypothèse.
- **Atteinte** : le majorant est atteint dès qu'un groupe porte un bloc à 4 niveaux, parce qu'**un bloc déjà à `L ≤ 4` garde sa classe sous le plafond** — son mot de code est l'argmin sur la boule entière, il reste l'argmin sur un sous-ensemble qui le contient encore. Mesuré : **4 708 799 groupes sur 4 708 800** portent un tel bloc.

> 🕳️ **Ce que ce compte remplace, et pourquoi l'ancienne justification était mauvaise.** Ce paragraphe affirmait auparavant que « la part des blocs à 4 niveaux monte à **68,2 %** après plafonnement, et la probabilité qu'un groupe de 32 n'en contienne aucun est ~10⁻¹⁶ ». Les deux sont à retirer. Le 68,2 % était une part **supposée après plafonnement**, qu'aucune mesure du dépôt n'établit — seul le 65,9 % d'avant plafonnement l'était. Le 10⁻¹⁶ supposait en plus l'**indépendance** des 32 blocs d'un groupe, qui sont des blocs voisins d'une même ligne d'une même matrice et n'ont aucune raison de l'être. Le raisonnement arrivait au bon nombre par un chemin faux ; il est remplacé par un compte exhaustif et une propriété d'argmin, qui ne demandent ni loi de probabilité ni hypothèse d'indépendance.

La prime de coalescence de l'option (c) reste donc de **0,625 b/poids (13 %)** au-dessus du plafonné, pas de 0,93 : c'est la différence entre les 5,333 b/poids d'un enregistrement uniforme de 16 o et les 4,7083 mesurés ici. ⚠️ Les deux termes ne portent pas tout à fait la même comptabilité — (c) supprime `bases`, donc ses 5,333 n'ont aucun terme d'adressage, là où le 4,7083 en porte 0,042. L'écart joue **en faveur** de (c) et ne renverse rien, mais il est à dire.

### 3.8 Exige mesure — parallélisme et latence de lancement

- **Un warp par ligne sous-remplit la carte sur 72 des 252 matrices.** L40S : 142 SM × 48 warps = 6 816 emplacements. k_proj et v_proj (d_out = 1024) n'exposent que 1 024 warps = 15 % ; o_proj et down_proj 2 560 = 38 %.
- **Mais le split-K n'est probablement pas la réponse pour ces 72-là** : leur flux Slot32 fait ~1,74 Mo, du même ordre que ce qu'un lancement coûte. Le travail lui-même est trop petit pour être réparti, et ces matrices ne pèsent que ~5 % des octets du modèle. **Ordre recommandé : CUDA Graph (qui traite les 252 d'un coup) avant split-K (qui traite 5 % du volume).**
- **Une variante split-K à atomiques rendrait la pire-erreur non reproductible** au chiffre près. Arbitrage à trancher **avant** de mesurer.

### 3.9 Exige mesure — pression de registres

Ada : 64 K registres/SM, 1 536 threads/SM → **≤ 42 registres/thread** pour saturer les 48 warps. `slot_dot` tient simultanément w0..w4, lo/hi, pay_lo/pay_hi, m1..m4, v0..v4, d0..d3, plus l'état de l'appelant. La mémoire partagée ne limite pas (12-14 Ko/bloc → 6 blocs/SM, plafond atteint par les threads de toute façon).

**Ne pas dépendre de `ptxas -v`** : il exige la toolchain CUDA et mesure une compilation qui n'est pas celle qui tournera (NVRTC émet du PTX, le driver JIT). **Relever au runtime, sur la vraie carte, sans privilège** : `num_regs()`, `local_size_bytes()` (0 attendu — sinon spills), `shared_size_bytes()`, `binary_version()` (89 attendu), `occupancy_max_active_blocks_per_multiprocessor()`. ~15 lignes, 0 $, disponible au premier run. Balayage `maxrregcount` ∈ {32, 40, 48, 64, illimité} : cinq points.

⚠️ **Un piège de compilation qui se manifeste ici.** `slot_dot` tient ses quatre chaînes FMA dans `d0..d3` sélectionnées par un `switch (k)` dans une double boucle (lib.rs:585-599). Si nvcc ne déroule pas complètement, `d0..d3` deviennent un tableau indexé dynamiquement, donc de la **mémoire locale**, sur le chemin le plus chaud. `#pragma unroll` sur les deux boucles, et `local_size_bytes() == 0` comme détecteur.

### 3.10 Exige mesure — `ncu` est-il disponible ?

L'accès aux compteurs de performance NVIDIA est restreint aux administrateurs par défaut (`ERR_NVGPUCTRPERM`), et rien dans `ops/` ne suggère qu'un job HF s'exécute avec ces droits. À vérifier au premier run pour quelques centimes.

Si non : ce qui reste est le noyau « **sol** » (mêmes lectures, zéro décodage), qui isole le coût du décodage sans aucun compteur matériel — la méthode déjà employée en Metal (`matvec.rs:148-178`). Seule la question « secteurs par requête » (§3.7) reste aveugle.

**Ne pas budgéter le tuning en supposant `ncu` disponible.**

---

## 4. Le harnais : correction d'abord, vitesse ensuite

### 4.1 Principe — partager la donnée, pas le dispatch

La référence f64 (`thesis.rs:237-271`) n'appelle que `llvq-artifact`, `llvq-search`, `llvq-core` — tous sans dépendance externe. **Elle ne doit pas être portée, elle doit être partagée**, sinon deux références dérivent et le jour où l'une bouge on ne sait plus laquelle a raison.

Mais **ne pas généraliser le dispatch** : Metal passe des closures typées `impl Fn(&ComputeCommandEncoderRef, usize) -> (u64,u64)` avec `set_buffer(index, …)` ; CUDA empile des arguments **positionnels** via `LaunchArgs::arg`. Un trait à types associés fuirait au premier layout supplémentaire.

**Frontière** : un crate `llvq-kernel` (dépendances : `llvq-core`, `llvq-search`, `llvq-artifact`) expose `f16_bits`, `f16_to_f64`, la table des 384 classes, et une fonction produisant un `Vec<MatSpec>` (nom, d_out, d_in, nblocks, tail_w, words, bases, gscale, rscale, tail, w16, y_ref, y16_ref, scale) plus `worst_error`. Chaque backend écrit sa propre boucle d'upload + dispatch (~120 lignes).

Obstacle mécanique à lever d'abord : `f16_bits` (lib.rs:305), `f16_to_f64` (:342), `GpuClassRec` (:361-375) et `gpu_class_table` (:380-419) sont déjà hors du `mod gpu` cfg-macOS, mais **le crate entier est fermé par `compile_error!` (lib.rs:631-632)**.

### 4.2 Le seuil de vérification est un détecteur d'incendie — le durcir

`assert!(e < 1e-3)` (thesis.rs:351, :357) sur la métrique `|got − want| / Σ|wᵢxᵢ|` **ne peut pas attraper un bit de signe retourné** : l'erreur relative vaut ~2/d_in, soit 2,1·10⁻⁴ sur `down_proj` (d_in = 9728) et 7,8·10⁻⁴ sur `gate_proj` — les deux sous le seuil.

Trois corrections cumulables :
1. **Durcir à 1e-5** — encore ~300× au-dessus des 3,4·10⁻⁸ observés en Metal, mais sous le plancher d'un signe retourné.
2. **Toujours imprimer la pire erreur** : c'est elle la preuve, pas l'assert.
3. **Ajouter un mode de vérification par bloc** sur un échantillon (le code existe côté Metal dans `bin/decreal`, `verify()` :125-135). La granularité ligne laisse passer des erreurs qui se compensent sur 2 560 à 9 728 colonnes.

### 4.3 Fermer la boucle du transcodage sur des blocs réels

`thesis` construit sa référence par `rt.decode_block` : il vérifie donc le GPU contre le **transcodeur**, jamais le transcodeur contre `Indexer::decode`. Le verrou qui fait ça (`llvq-artifact/tests/runtime_format.rs`) tourne exclusivement sur des index **synthétiques** (bornes de classes + 200 000 tirages). Et `decreal::expected()` appelle lui aussi `decode_block` : le mode « bloc » ne ferme pas ce trou-là.

**À ajouter dans le harnais partagé, avant la référence f64** : sur un échantillon de blocs **réels** (1 sur 1024 ≈ 147 k blocs, plus le premier bloc de chaque classe rencontrée), exiger `rt.decode_block(&table, b) == Indexer::decode(indices[b])`. `llvq-search` est déjà une dépendance. Coût nul, et ça ferme la boucle sur l'objet publié.

### 4.4 Trois gardes que Metal n'a pas

1. **`assert_eq!(gain_bits, 1)`** au moment du transcodage, dans le harnais partagé (§1.12). g = 0 comme g = 2 cassent le shader.
2. **`worst_width_slot()` + `assert!(24 + worst_width_slot() <= 160)`.** `ClassTable` expose `worst_width()` et `worst_width_flat()` (runtime.rs:150, :155) mais **pas** le troisième, et aucune ligne du dépôt n'écrit l'inégalité dont dépend la lecture de 5 mots. Note : `width_slot ≥ width_flat` toujours, donc l'assertion existante à 130 sur `width_flat` **ne majore rien** côté slot — c'est une coïncidence.
3. **Borner `tab[id]`.** Le champ fait 9 bits (512 valeurs) pour une table de 384 entrées. Inatteignable depuis un flux valide (`class_id` refuse hors plage à l'écriture, runtime.rs:473-482), mais un fichier tronqué produit une faute mémoire réelle sur CUDA. Correctif à coût nul : allouer 512 enregistrements (16 Ko, ou 12 Ko en version réduite) et laisser les 128 derniers à l'origine.

### 4.5 Trois bras, et le rapport contre le meilleur

| bras | quoi | pourquoi |
|---|---|---|
| **cuBLAS** | `Gemm<half::f16>` (`gemm_ex`, n = 1) — ou cuBLASLt, qui a des noyaux dédiés aux GEMM très minces | ferme l'angle hostile « baseline maison » |
| **`tv_f16` transliteré** | chargements 128 bits (`uint4` = 8 half) plutôt que `half4` | continuité avec le chiffre Metal |
| **LLVQ `tv_slot`** | l'objet de la spec | — |
| *(+ noyau « sol »)* | mêmes lectures, zéro décodage | borne la bande passante atteignable — referme §6.8 rétracté |

**Le rapport titre est calculé contre la meilleure des deux baselines, et les trois lignes sont publiées.** Poser le résultat honnêtement dans les deux sens : si cuBLAS gagne, l'objection tombe et le rapport doit être re-cité contre lui ; s'il perd, dire qu'un GEMM générique n'est pas optimisé pour n = 1 et que la comparaison **ne valide pas** le `tv_f16` maison.

Détail à épingler : cudarc force `CUBLAS_COMPUTE_32F` pour f16 (`gemm.rs:63-97`). Un GEMM en `CUBLAS_COMPUTE_16F` accumulerait 9 728 termes en demi-précision — plus rapide **et** plus faux, et il pourrait rester juste sous le seuil et passer pour du bruit normal. Le chemin cudarc le neutralise ; cuBLASLt exige de le poser à la main. cuBLASLt n'est par ailleurs pas coûteux : cudarc alloue le workspace et pose la taille de préférence lui-même (`cublaslt/safe.rs:31-37`, :63-76, :378-379).

### 4.6 Le protocole de mesure, transposé

| aspect Metal | transposition CUDA |
|---|---|
| un command buffer, 252 encoders (`dispatch_batch`, lib.rs:~185-222) | **un stream**, 252 lancements |
| chrono **après** encodage, autour de `commit`+`wait` | événements CUDA `CU_EVENT_DEFAULT` autour du premier et du dernier lancement, `stream.synchronize()`, `elapsed_ms` |
| 7 passes, reps 0 et 1 jetées, **minimum** des 5 | identique pour les **temps** — le bruit d'un GPU partagé est au-dessus, jamais en dessous. Mais le **rapport** se forme round par round, jamais en divisant deux minima (paragraphe sous la table). ⚠️ **Nécessaire, pas suffisant : §4.6bis** |
| FP16 mesuré en premier | **entrelacer A,B,A,B,…** (voir §4.8 et §4.6bis) |
| froid par construction (2,50 / 7,27 Go distincts) | conservé : les deux bras résidents font 9,8 Go sur 48 |
| sérialisation par hazard WAW sur `y` unique | **ne pas recopier le raisonnement** : sur CUDA c'est l'ordre du stream qui sérialise. Garder `y` unique par honnêteté (mêmes octets écrits), mais dire pourquoi |

**Le rapport se forme round par round, jamais en divisant deux minima.** C'est le point de méthode que K−1 a dû corriger sur son propre banc : chaque round mesure tous les bras, on en tire un rapport par round, puis on publie la **médiane** et la **plage**. Un quotient de minima mêle deux rounds qui n'ont jamais coexisté, et il ne se retrouve pas à partir des colonnes publiées — sur le run archivé, 21,775 / 9,925 donne 2,19 quand le rapport round par round donne 2,15. **Partout où la table est reproduite, la phrase qui dit comment le rapport est formé doit l'accompagner**, sinon l'écart passe pour une erreur d'arithmétique.

**Quatre choses à ne PAS faire** : streams multiples, noyau persistant, `cudaLaunchHostFunc`, ou une sortie par dispatch — toutes laisseraient se recouvrir les 252 matrices, ce que la chaîne de dépendance d'un transformeur interdit. C'est l'avertissement déjà écrit dans `llvq-metal/src/lib.rs:146-153` et :195-198, à re-libeller mais pas à supprimer.

**Ne pas soustraire le surcoût de lancement.** `thesis` ne soustrait rien (`overhead()` n'est utilisé que par `matvec`), et §6.1 pose le corollaire : tout terme additif commun donne (T₁₆+c)/(T_slot+c) < T₁₆/T_slot, donc le rapport publié est un **minorant**. Un port qui « nettoierait » obtiendrait un chiffre plus flatteur et **non comparable** au chiffre Metal.

**Ne pas présenter l'écart horloge-murale / événements comme « le surcoût de soumission ».** Les lancements CUDA sont asynchrones : la soumission de N+1 recouvre l'exécution de N. Publier plutôt le **temps CPU de la boucle de soumission seule** : s'il est nettement sous le temps GPU, il prouve que le GPU n'a jamais été affamé — c'est la propriété qu'on veut réellement établir. Si un CUDA Graph est utilisé, publier les **deux** chiffres et ne jamais comparer un graph CUDA à un command buffer Metal sans le dire.

### 4.6bis La dispersion inter-processus — mesurée, et elle change le protocole

**C'est le résultat le plus transportable de K−1, et il ne parle pas de LLVQ : il parle de la façon de mesurer.** Le banc à **deux bras non modifié** (`bin/thesis` avant K−1, code identique aux trois runs) a été invoqué **trois fois de suite** (`docs/mesures/thesis-temoin-2026-08-04.txt`) :

| invocation | FP16 ms | LLVQ ms | rapport |
|---|---|---|---|
| 1 | 21,983 | 10,832 | **2,029×** |
| 2 | 21,783 | 10,627 | **2,050×** |
| 3 | 21,680 | 10,421 | **2,080×** |
| publié (`format-noyau.md`) | 21,690 | 10,460 | 2,074× |

Ce qui est **identique** aux trois runs : les deux pire-erreurs (3,4e-8 côté LLVQ, 2,8e-8 côté FP16) et les octets lus. **Seuls les temps bougent** — l'arithmétique n'a pas changé, donc ce n'est pas une variation de travail.

Ce qui bouge est **monotone** : les deux bras accélèrent ensemble d'un processus au suivant. Ce n'est pas du bruit symétrique, c'est un réchauffement — cache de fichier sur les 981 Mo d'artefact, état d'horloge du GPU, résidence des 9,8 Go de tampons.

**Trois conséquences, à porter telles quelles côté CUDA :**

1. **Le 2,07× publié est le HAUT d'une plage [2,029 ; 2,080]**, reproduit au **troisième** run consécutif. Un premier run à froid rend 2,029×. Publier une valeur ponctuelle, c'est publier le meilleur des trois sans le dire.
2. **Le protocole de §4.6 — « 7 passes, reps 0 et 1 jetées, minimum des 5 » — est nécessaire mais PAS suffisant.** Il ne couvre que la dérive *intra*-processus. Celle-ci vit **entre** les processus, donc aucun warm-up interne ne peut l'atteindre.
3. **Donc : un effet de quelques pour cent ne peut pas être tranché en comparant deux invocations distinctes du binaire.** Les bras doivent être **entrelacés dans un même processus** et la dispersion **imprimée**. C'est déjà appliqué côté Metal — le banc à sept bras de §3.2 dispatche tous les bras à chaque round, dans le même ordre — et c'est ce qui permet d'y conclure « **aucun gain** » sur le padding plutôt que d'y lire un effet fantôme de quelques pour cent. **À appliquer côté CUDA sans exception.**

Cette prescription **converge** avec celle de §4.8 contre le throttling : même remède, deuxième raison indépendante. Sur une carte partagée et bridée thermiquement, la dérive inter-processus a toutes les raisons d'être plus forte, pas moins.

### 4.7 Le quatrième piège du dépôt, transposé

Le piège documenté (`format-noyau.md`, section « La brèche : Slot32, 2,2× le FP16 », point 1 « Le quatrième piège de mesure du projet ») : des tampons de 11-17 Mo tenaient dans le SLC de 48 Mo du M3 Max et étaient rejoués 576 fois, rendant tous les chiffres LLVQ antérieurs optimistes. *(Renvoi recalé le 2026-08-05 : il pointait sur `format-noyau.md:242-251`, mais le lot K−1 a inséré une centaine de lignes dans ce fichier et cette plage tombe désormais à cheval sur la fin de la section précédente. Ancrer sur le **titre de section**, qu'un lot suivant ne décalera pas.)*

- **`bin/thesis` y survit** : 2,50 et 7,27 Go de poids **distincts** par passe. C'est un argument de forme, pas une constante machine : il traverse le changement de carte intact.
- **`bin/matvec` non**, et c'est une raison de plus de ne pas le porter. Le flux Slot32 de `gate_proj` fait ~16,7 Mo ; une couche entière ~67 Mo. Le protocole à 4 copies rotatives serait insuffisant.
- **Ne citer aucun chiffre de L2 de mémoire.** Les sources tierces se contredisent (48 Mo pour la SKU L40S, 96 Mo pour l'AD102 complet) et NVIDIA ne publie pas la valeur. **Lire `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE` au démarrage et l'imprimer** (`CudaContext::attribute`, `core.rs:362`), et dériver tout nombre de copies de la valeur lue. Même discipline que `simd_width()` lu et non supposé (lib.rs:78-82). Le dépôt a déjà rétracté un « 93 % du pic » construit sur un pic supposé ; ce serait la deuxième fois.

### 4.8 Deux pièges que Metal n'avait pas

1. **Le throttling.** La L40S est une carte 350 W à refroidissement passif dans un châssis partagé, et `nvidia-smi -lgc` exige des droits qu'un conteneur n'a probablement pas. Parades gratuites : **entrelacer les bras** (A,B,A,B,… au lieu de AAAAAAA puis BBBBBBB — l'asymétrie « FP16 d'abord » de §6.1 n'est conservatrice que si la dérive est monotone) ; relever `clocks.sm`, `clocks_throttle_reasons.active`, `power.draw`, `temperature.gpu` autour de chaque passe (`nvidia-smi -q -d PERFORMANCE,CLOCK` ne demande **aucun** privilège) et rejeter les passes bridées.
2. **L'ECC**, activée par défaut sur L40S. Elle frappe les deux bras également, donc le **rapport** est propre ; mais tout énoncé « x % du pic » doit utiliser un pic mesuré (noyau « sol »), jamais les 864 Go/s de la fiche.

### 4.9 Ce que le harnais ne pourra PAS faire

**L'égalité bit-à-bit Metal ↔ CUDA est impossible**, et il faut l'annoncer avant qu'on la lise comme un défaut :

- `simd_sum` et un arbre `__shfl_*_sync` somment dans des ordres différents.
- nvcc contracte les FMA selon ses propres règles.
- Une asymétrie de compilation existe et doit être documentée : `llvq-metal/src/lib.rs:53` compile avec `CompileOptions::new()` nu, et **rien dans le dépôt n'appelle `set_fast_math_enabled`** — quel que soit le défaut d'Apple, il s'applique. NVRTC, lui, compile en IEEE avec `--fmad=true`. **À vérifier et à fixer explicitement des deux côtés avant de comparer des erreurs.**

Conséquence de protocole : **ne jamais comparer `y_cuda` à `y_metal`** ; chacun contre la référence f64 partagée. Deux pire-erreurs publiées, pas un delta. Et **le 3,4·10⁻⁸ de Metal ne doit jamais être recopié dans une table CUDA** — le chiffre CUDA sera différent et légitime.

Seule la valeur de retour de `slot_dot` est entièrement ordonnée (quatre chaînes nommées, parenthésage explicite) et donc comparable au bit près. Un contrôle inter-backend est possible, mais il exige un **dump par bloc** des deux côtés — forme que `bin/decreal` possède déjà.

### 4.10 Le prologue hôte, et l'arithmétique à corriger

`bin/thesis` fait, par matrice et **sur un seul cœur** : transcodage (~37 s pour le modèle, mono-thread par construction — `llvq-artifact` est sans dépendance), reconstruction f64 de 3,63 Md de poids, conversion f16, puis une référence f64 de 1 105 920 lignes (~10¹⁰ opérations double). C'est ce que `load_s` = 128 s mesure sur M3 Max, mal étiqueté (§6.9).

**Le transfert PCIe n'est PAS le poste dominant** : 9,8 Go à 20-25 Go/s font ~0,5 s, ~2-3 s en mémoire paginable avec 252 copies séparées. Le poste dominant est le prologue CPU mono-thread, sur les **8 vCPU** d'un `l40sx1`.

**Correctif : paralléliser la boucle de matrices** (`thread::scope`, chaque itération est indépendante hors lecture séquentielle du fichier). Ce n'est pas interdit : `std::thread::scope` est de la bibliothèque standard, pas une dépendance — `llvq-quant/src/gptq.rs:478` l'utilise déjà sous la même règle. À faire dans `llvq-cuda`, pas dans `llvq-artifact` (dont on garde le `transcode()` mono-thread pour que `bits_per_weight()` et le décodeur restent lisibles ligne à ligne). **Ça divise par ~4 le budget machine de tout le chantier pour ~0,5 j de travail.**

### 4.11 Le chemin ops

- **L'image ne sait construire que quatre binaires** (`ops/Dockerfile.cuda:52-53`, `-p llvq-llm --bin smoke --bin ppl --bin oracle --bin mmlu`, copiés :70-75 — citations recalées de +4 le 2026-08-05, l'insertion du commentaire `--locked` de §2.6 ayant décalé tout ce qui suit la ligne 47). Le banc se construit avec `-p llvq-cuda` : crate léger, aucun lien candle, donc **il n'aggrave pas** le risque SIGKILL documenté :31-43. Prévoir tout de même 2 tentatives de build, 40-70 min chacune, non facturées.
- **`ops/run.py` n'a aucun chemin générique** : `cmd_launch` code `command = ["smoke", …]` (`:378`) et `cmd_oracle` code `["oracle", …]` (`:520`). Le lancement est de plus conditionné à un estimateur de coût que `plan-de-test-v2-cuda.md` §8 déclare inapplicable hors quantification. Le ticket O1 du plan (sous-commande générique, ~2 h) est un **prérequis dur**, pas un à-côté. Exiger `set -euo pipefail` et `--timeout` : sans ça, un banc qui échoue laisse la commande suivante tourner et le job finit COMPLETED sans résultat.
- **Bonne nouvelle gratuite** : un crate nommé `llvq-cuda` est déjà couvert par `allow_patterns=["Cargo.toml", "Cargo.lock", "llvq-*/**"]` (`run.py:480`), donc ses `kernels/*.cu` montent dans le Space sans toucher au script.
- **L'artefact** : `run.py:390-392` sait déjà monter un dépôt du Hub en lecture seule. Le banc prend le chemin monté en argv[1]. Corriger au passage le défaut `~/llvq-q4b.llvq` (thesis.rs:191-193), fichier **non publié** — le fichier utilisable est le scellé, `Pier-Jean/Qwen3-4B-LLVQ-2bit`.
- **Pré-vol obligatoire, ~0,01 $, à faire en tête du premier job** plutôt qu'en job dédié :
  - `ldd $(which thesis-cuda) | grep -i nvrtc` et `ls /usr/local/cuda/lib64/libnvrtc*` + `libnvrtc-builtins*`. L'argument « le run 8B de 4,18 h le prouve » **ne tient pas** : cudarc émet `-lnvrtc` sous `dynamic-linking` (`build.rs:206-207`) mais `CudaDevice::compile` de candle est `#[cfg(feature = "ug")]`, feature désactivée — aucun symbole NVRTC n'est référencé, et `--as-needed` peut avoir supprimé le DT_NEEDED. Correctif si absent : `COPY --from=build /usr/local/cuda/lib64/libnvrtc*.so* …`.
  - **Les headers CUDA sont absents de l'étage runtime** : `nvidia/cuda:12.4.1-runtime` installe `cuda-libraries-12-4` et consorts, **aucun paquet `-dev`**, donc `/usr/local/cuda/include` est vide et `#include <cuda_fp16.h>` échouera même avec `--include-path`. Deux sorties : refaire la conversion f16↔f32 par manipulation de bits dans le kernel (la version Rust existe déjà, lib.rs:305-357, ~15 lignes à transposer), ou copier les headers depuis l'étage build. **À décider avant d'écrire le kernel.**
  - `ptxas --version` pour savoir ce qui est disponible.
- **Débogage** : `compute-sanitizer` et `cuda-gdb` ne sont pas dans l'image `runtime`. Or `compute-sanitizer --tool memcheck` est exactement l'outil qui attraperait un `tab[id]` non borné ou un rembourrage de 20 octets manquant. Recommandation : image de banc sur `nvidia/cuda:12.4.1-devel` (elle n'a pas les contraintes de taille de la production), et un passage memcheck sur une seule matrice.
- **Garde de carte** : refuser de démarrer hors `l40sx1`. `l4x1` est à ~300 Go/s, **sous** les ~400 Go/s supposés du M3 Max — un « le 2,07× se transporte-t-il ? » mesuré là serait structurellement pessimiste et non interprétable. `a100-large` (sm_80), `a10g` (86), `t4` (75) ne peuvent charger aucun noyau cap 89 (`cap_ok`, `run.py:91-96`).

---

## 5. Le plan d'implémentation

Chaque étape a un livrable vérifiable. **La première est la plus petite chose qui prouve que la chaîne fonctionne.**

### K−1 — les trois jalons Metal, locaux, 0 $ ✅ **FAIT le 2026-08-05**

> Le titre disait « les deux jalons » pour trois items — corrigé. Le troisième (`rtbits`) avait été ajouté après coup sans que le titre suive.

**Livrable** : (a) `matvec_g32` et `matvec_flat` sur le modèle entier, courbe bits↔vitesse sur un seul protocole ; (b) le padding anti-conflit de bancs de §3.2 testé sur `tv_slot`/`tv_f16` en Metal ; (c) la moyenne des max par groupe sous plafond L ≤ 4, via `rtbits`.
**Coût** : 0 $, comme prévu. Logs : `docs/mesures/`.

| jalon | ce qu'il a rendu | où |
|---|---|---|
| **(a)** ✅ | La courbe bits↔vitesse sur **un seul protocole et un seul objet** : Slot32 2,09× [2,05–2,11], Flat32 0,91× [0,90–0,92], Grouped32 0,69× [0,69–0,69], aux 252 projections du modèle entier — rapports formés round par round, pas en divisant des minima. Elle est **brutalement non linéaire**, et ça tranche une décision de conception. | §6.1 |
| **(b)** ✅ | La prescription de §3.2 est **infirmée** : le padding à 28 flottants rend 2,13× [2,06–2,17] contre 2,15× [2,12–2,19] en `float4`@24, soit 1,7 % plus lent avec des plages qui se recouvrent. Le gain réel est la **largeur de chargement**, +4,6 % sur LLVQ et +4,9 % sur FP16 — donc à prendre des deux côtés, et le rapport ne bouge pas (2,05× float4/float4 contre 2,09× scalaire/scalaire). | §3.2 |
| **(c)** ✅ | `Slot32` à **5,3756 b/poids** et le plafond `L ≤ 4` à **≤ 4,7083** (0,667 b/poids, 12,4 %) — l'arithmétique de §3.7 confirmée au quatrième chiffre, et sa justification probabiliste remplacée par un compte exhaustif sur 4 708 800 groupes. | §3.7 |
| *(hors livrable)* ✅ | La **dispersion inter-processus** : trois invocations du banc non modifié rendent 2,029× / 2,050× / 2,080×. Le protocole de §4.6 est nécessaire mais pas suffisant. | §4.6bis |

**Effort réel : nettement plus que les ~0,5 j budgetés** — le budget ne comptait que le temps de run. Ce qu'il ne comptait pas : le banc à sept bras entrelacés (sans lequel le verdict (b), à 1,7 % d'écart, n'aurait pas été lisible), la comptabilité d'octets rendue identique entre les quatre layouts, l'assertion d'identité bit-à-bit des variantes `float4`, la vérification des 1 105 920 lignes contre la référence f64 sur **chaque** bras, et l'instrumentation de `rtbits` pour `Slot32`. Autrement dit : les contrôles et les mutants que `CLAUDE.md` §5 rend **obligatoires** ne sont budgetés dans aucun lot de ce plan. Le même écart est à attendre sur K1, K4 et K6, qui en portent au moins autant.

### K0 — le crate partagé `llvq-kernel`

**Livrable** : `f16_bits`, `f16_to_f64`, `GpuClassRec`, `gpu_class_table`, le chargement/transcodage des 252 matrices, la référence f64, `worst_error`, sortis de `llvq-metal`.

**Critère d'acceptation, corrigé le 2026-08-05.** Il disait : « `bin/thesis` rejoué avant/après doit rendre **le même rapport 2,06-2,08×** et la même pire erreur 3,4·10⁻⁸ ». La première moitié est inutilisable — c'est exactement une comparaison entre **deux invocations distinctes du binaire**, que §4.6bis vient de disqualifier pour tout effet de quelques pour cent : le rapport bouge de 2,029× à 2,080× sans qu'une ligne de code change. Ce qui est **invariant** aux trois runs témoins, et donc ce qu'un refactoring doit conserver : **les deux pire-erreurs (3,4·10⁻⁸ LLVQ, 2,8·10⁻⁸ FP16) et les octets facturés (2,50 et 7,27 Go)**. Le rapport n'est pas un critère de non-régression ; il est un résultat, à republier avec sa dispersion.
**Pourquoi** : sans lui les deux harnais divergent, et « même référence des deux côtés » redevient une affirmation au lieu d'une propriété.
**Effort** : 1,5-3 j.

### K0bis — hygiène du workspace

**Livrable** : `llvq-cuda` déclaré, coquille vide hors Linux, cudarc sous `[target.'cfg(target_os = "linux")'.dependencies]`, `cargo clippy --all-targets` **vert sur le Mac**.
**Effort** : 0,25 j.

> 🕳️ **Deux items retirés de ce livrable le 2026-08-05, parce qu'ils ne décrivaient rien de réel** (détail et généalogie en §2.6) : « `--locked` dans `ops/Dockerfile.cuda:48` » — l'image CUDA l'a depuis le commit 9de862f, qui est celui-là même qui a écrit ce plan ; et « retirer la ligne inerte du `.gitignore` » — cette ligne n'existe pas, le `.gitignore` fait deux lignes. Le `--locked` qui manquait vraiment était dans `ops/Dockerfile` (l'image CPU), hors du périmètre que ce document se donne ; il a été ajouté.

### K1 — le décodeur, testé sans GPU

**Livrable** : `llvq_slot.cuh` compilé **par clang sur le Mac** en C++ hôte (`__device__`/`__restrict__` neutralisés par macros), et diffé bloc par bloc contre `RuntimeBlocks::decode_block` sur les vrais blocs du 4B publié.
**Pourquoi c'est LA première étape technique** : `slot_dot` ne contient aucune primitive de warp — c'est du scalaire pur. Le test tourne en secondes, sur le Mac, à 0 $, et attrape tout ce qui n'est pas un bug de warp, de mémoire partagée ou de coalescence. Implémentation sans dépendance : un test Rust qui `Command`-e `clang++`, alimente une fixture de blocs, compare.
**Effort** : 1,5-2,5 j (décodeur + test).

### K2 — compilation NVRTC en CI, sans GPU

**Livrable** : job CI dans `nvidia/cuda:12.4.1-devel` compilant chaque variante pour `compute_89` ; rapport `ptxas -v` archivé (indicatif, pas contractuel — le SASS final vient du JIT).
**Effort** : 0,25 j. **Coût** : 0 $.

### K3 — plomberie cudarc

**Livrable** : `Kernel` CUDA à API voisine de `llvq_metal::Kernel` (compile NVRTC avec `arch = Some("compute_89")`, load_module, buffers H2D/D2H, launch, events `CU_EVENT_DEFAULT`), plus le relevé d'attributs au démarrage : `binary_version` (assert 89), `num_regs`, `local_size_bytes` (assert 0), `shared_size_bytes`, occupancy, `L2_CACHE_SIZE`, nom de la carte.
**Effort** : 1,5-2,5 j.

### K4 — `bin/thesis-cuda`, premier chiffre

**Livrable** : les 252 matrices, prologue parallélisé, **vérification des 1 105 920 lignes contre la référence f64 avant toute mesure**, trois bras (cuBLAS, `tv_f16`, `tv_slot`) plus le noyau « sol », un stream, événements, entrelacement A/B, 7 rounds dont 2 jetés avec **tous** les bras à chaque round, **rapport formé round par round puis publié en médiane + plage** (§4.6), ligne de résultat par variante de noyau, sha256 de la source compilée, relevé d'horloges.
**Effort** : 1,5-2,5 j.

### K5 — ops

**Livrable** : ticket O1 (sous-commande de lancement générique, `set -euo pipefail`, `--timeout`, garde de carte), image de banc, montage de l'artefact, répertoire de noyaux monté, pré-vol de §4.11 dans le même job.
**Effort** : 1-1,5 j.

### K6 — durcissement du harnais

**Livrable** : `assert_eq!(gain_bits, 1)` ; `worst_width_slot()` + assertion `24 + w ≤ 160` ; seuil 1e-5 ; échantillon de blocs réels `decode_block == Indexer::decode` ; table à 512 entrées ; mode de vérification par bloc ; `#pragma unroll` + contrôle de spills.
**Effort** : 0,5 j.

### K7 — réglage jusqu'au chiffre

**Livrable** : le rapport mesuré, avec sa dispersion, contre la meilleure des deux baselines, et le relevé d'attributs qui l'accompagne.
**Balayage, dans cet ordre** : largeur de chargement `float4`, **sur les deux bras** (§3.2, seul effet mesuré en Metal) → `maxrregcount` (§3.9) → staging coopératif (§3.7-a) → CUDA Graph (§3.8) → taille de tuile → padding de tuile (§3.2), **descendu en fin de liste** : il ne gagne rien sur Apple, donc son gain sur CUDA n'est plus une prévision mais une hypothèse à tester en dernier.
**Critère de sortie** : « **mesurer R avec une baseline défendable** », **pas** « atteindre R > 1 ». `plan-de-test-v2-cuda.md` §9 écrit déjà que l'écart entre backends est lui-même une mesure publiable — la sensibilité au backend d'un résultat de quantification 2 bits, qu'aucun papier du domaine ne rapporte. Sans ce critère, K7 n'a pas de condition d'arrêt.
**Effort** : 2-5 j — toute la variance du chantier.

### Récapitulatif

| lot | effort (j-h) |
|---|---|
| ~~K−1 jalons Metal locaux~~ ✅ fait | 0,5 budgeté — **dépassé**, voir ci-dessus |
| K0 crate partagé | 1,5–3 |
| K0bis hygiène workspace | 0,25 |
| K1 décodeur + test hôte | 1,5–2,5 |
| K2 CI compile-only | 0,25 |
| K3 plomberie cudarc | 1,5–2,5 |
| K4 `thesis-cuda` + 3 bras | 1,5–2,5 |
| K5 ops (O1 inclus) | 1–1,5 |
| K6 durcissement | 0,5 |
| K7 réglage | 2–5 |
| **total** | **10,5 – 18,5** |

⚠️ **Ce total est lui-même sous-estimé, et K−1 vient de le montrer sur le seul lot exécuté.** Chaque ligne budgète le travail nominal, jamais les contrôles que `CLAUDE.md` §5 rend obligatoires — comptabilité rendue identique entre bras, identités bit-à-bit épinglées, vérification exhaustive contre la référence f64, mutants. Sur K−1 ce poste a dépassé le lot entier. Aucun facteur correctif n'est avancé ici : un seul lot ne fait pas une loi. Ce qui est acquis, c'est que le total est un **plancher**, pas une fourchette centrée.

Un premier chiffre **correct mais non optimisé** tombe à K4, soit **6-9 j**. L'écart avec les « 5-10 jours » de `plan-de-test-v2-cuda.md` §9 arbitrage 4 s'explique intégralement par trois postes que le plan ne budgète pas : le crate partagé (K0), les prérequis d'infrastructure (K0bis, O1 dans K5) et le durcissement du harnais (K6). Le compte de CUDA C est d'ailleurs **≈160 lignes hors variantes, ≈200-230 avec le sol et une variante mémoire partagée** — soit l'estimation du plan, pas une révision à la baisse.

**Coût machine** : `l40sx1` à 1,80 $/h = 0,030 $/min ; plancher de facturation 6-10 min/job ; avec le prologue parallélisé un run fait ~8-12 min ≈ **0,25-0,40 $**. 15-30 runs ≈ **5-12 $**. Provisionner 30 $. Les builds d'image ne sont pas facturés. **Ce port se décide sur le temps humain, pas sur l'argent.**

Poste optionnel, à arbitrer **explicitement** et non par omission : un bras **4 bits** (Marlin, GEMV d'AWQ, noyaux GPTQ) dans le même conteneur. `CLAUDE.md` §3bis pose que le FP16 est la mauvaise référence et que le 4 bits est « le problème central à résoudre » ; le portage CUDA est précisément ce qui rend cette comparaison possible sur le même silicium, ce que Metal interdisait. **+2 à 4 j**, avec une inconnue à lever d'abord : à batch 1, quel noyau est réellement dispatché (`gemv` ou `gemm`) — à **lire** dans le code de dispatch, jamais à supposer.

---

## 6. Ce qui reste inconnu

Classé par ce qu'il faudrait faire pour le savoir.

### 6.1 Répondu localement, 0 $ — ✅ fait le 2026-08-05

| question | réponse |
|---|---|
| Le noyau Metal paie-t-il déjà le conflit de bancs 8 voies ? | **Question non tranchée telle quelle, et devenue sans objet : le correctif prescrit ne gagne rien** — 2,13× [2,06–2,17] contre 2,15× [2,12–2,19] pour le `float4` dense, soit 1,7 % plus lent avec des plages qui se recouvrent. Ce qui gagne est la largeur de chargement, sur les deux bras (§3.2). |
| Que valent Grouped32 et Flat32 sur le modèle entier ? | **0,69× et 0,91×** — voir la courbe ci-dessous, qui est le vrai résultat du jalon. |
| Quelle est la vraie moyenne des strides sous plafond L ≤ 4 ? | **4,667 niveaux de max par groupe**, `Slot32` à 5,3756 b/poids, plafond à ≤ 4,7083 (§3.7). |

#### Le fait nouveau : la courbe bits↔vitesse est brutalement non linéaire

Les 252 projections du modèle entier, un seul protocole, un seul objet, sept bras entrelacés dans un même processus, **comptabilité d'octets identique pour les quatre layouts** (payload + bases + queue f32 + échelles de ligne f32 — ce n'est **pas** la métrique étroite de §3.7) :

| bras | b/poids | min ms | Go lus | Go/s | vs FP16 méd [plage] |
|---|---|---|---|---|---|
| FP16 (half4, scalaire) | 16,000 | 21,775 | 7,27 | 334 | 1,00× [1,00–1,00] |
| **LLVQ Slot32 (scalaire@24)** | **5,510** | **10,401** | 2,50 | 241 | **2,09× [2,05–2,11]** |
| LLVQ Flat32 | 5,256 | 24,009 | 2,39 | 99 | 0,91× [0,90–0,92] |
| LLVQ Grouped32 | 3,498 | 31,494 | 1,59 | 50 | 0,69× [0,69–0,69] |
| FP16 (half4, `float4`) | 16,000 | 20,709 | 7,27 | 351 | 1,05× [1,05–1,05] |
| LLVQ Slot32 (`float4`@24) | 5,510 | 9,925 | 2,50 | 252 | 2,15× [2,12–2,19] |
| LLVQ Slot32 (`float4`@28) | 5,510 | 10,091 | 2,50 | 248 | 2,13× [2,06–2,17] |

> **La colonne « vs FP16 » n'est pas le quotient des colonnes « min ms ».** Le rapport est formé **round par round**, puis résumé par sa **médiane** et sa plage sur les 5 rounds gardés ; un minimum divisé par un minimum mêlerait deux rounds qui n'ont jamais coexisté. 21,775 / 9,925 donne 2,19, la colonne dit 2,15 — l'écart est la définition, pas une erreur. Et **les ms dérivent d'un run à l'autre** (c'est le fait même qu'établit §4.6bis) là où les b/poids et les octets reproduisent au chiffre : citer de préférence le b/poids et le rapport avec sa plage, et renvoyer à `docs/mesures/k1-metal-2026-08-05.txt` pour les ms.

Lue en bits contre temps, la courbe n'a rien de progressif :

- **Flat32 n'économise que 0,254 b/poids sur Slot32, et coûte 2,31 fois le temps.**
- **Grouped32 économise 2,012 b/poids, et coûte 3,03 fois le temps.**

*(Ces quatre nombres sont des différences et des quotients **de la table
ci-dessus**, pas des mesures séparées : `5,510 − 5,256`, `24 009/10 401`,
`5,510 − 3,498`, `31 494/10 401`. Ils sont licites parce que les deux termes
viennent du **même run**, du **même processus** et de la **même comptabilité
d'octets** — c'est exactement ce que le protocole entrelacé de §4.6bis rend
permis, et ce qu'une comparaison entre deux invocations distinctes ne
permettrait pas. Ce sont des grandeurs **dérivées**, à étiqueter comme telles.)*

Le débit effondré le dit autrement : 241 Go/s pour Slot32, 99 pour Flat32, 50 pour Grouped32. Les trois lisent **moins** d'octets que le FP16 et deux des trois sont plus lents que lui.

> **Conclusion de conception, et elle est ferme : reprendre des bits doit se faire DANS `Slot32`, jamais en changeant de layout.** Les deux layouts plus compacts ne sont pas des points d'un compromis réglable — ce sont des impasses mesurées sur le modèle entier.

**Cela classe les quatre options de §3.7 :**

- **(a) staging coopératif** — reste la sortie préférée : zéro bit, zéro changement de format, et elle vit entièrement à l'intérieur de `Slot32`.
- **(c) enregistrement uniforme de 16 o (`L ≤ 4`)** — reste la seule reprise de bits recevable, parce qu'elle **plafonne** `Slot32` au lieu d'en sortir. Ses deux termes sont maintenant chiffrés, et **il ne faut pas les confondre** : le plafonnement seul, en gardant l'adressage groupé, donne **≤ 4,7083** b/poids ; l'enregistrement uniforme de 16 o qui achète la coalescence en donne **5,333**, sans terme d'adressage. Les deux passent par une requantification du 4B et une reprise de ppl/MMLU.
- **(b) transposition par mot** — à ne pas engager : elle réécrit `decode_block` et les round-trips pour rester dans la même famille de layout, sans que rien ne dise ce qu'elle achète.
- **(d) ne rien changer** — toujours l'option par défaut tant que §6.3 n'a pas dit si le noyau est borné par l'émission LSU ou par la latence.

⚠️ **Ce classement est mesuré sur Apple.** Il ne dit pas que Flat32 et Grouped32 seraient également mauvais sur Ada — personne ne l'a mesuré, et §6.3 pose que le motif de lecture est précisément ce qui n'est pas préjugeable. Ce qu'il dit, c'est qu'aucun de ces deux layouts ne mérite un jour de portage CUDA avant que `Slot32` ait rendu son chiffre.

### 6.2 Répondu au premier job, quelques centimes

| question | comment |
|---|---|
| `libnvrtc.so.12` et `libnvrtc-builtins` sont-ils dans l'image runtime ? | `ldd` + compilation d'un kernel vide |
| Les headers CUDA sont-ils présents ? (pronostic : **non**) | `ls /usr/local/cuda/include` |
| Taille réelle de la L2 de la carte servie | `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE` |
| Compte de registres, spills, occupancy atteinte de `slot_dot` | `num_regs`, `local_size_bytes`, `occupancy_max_active_blocks…` |
| Le PTX est-il bien compilé pour sm_89 ? | `binary_version()` |
| `ncu` est-il autorisé dans un conteneur HF Jobs ? | tentative, `ERR_NVGPUCTRPERM` |
| Bande passante réellement atteignable, ECC comprise | noyau « sol », dans le même job |

### 6.3 Exige la campagne, et n'est pas préjugeable

- **Le motif de lecture Slot32 (32 lanes × 5 mots u32 à des offsets non alignés, stride 14-17 o) se comporte-t-il comme sur Apple ?** C'est la question qui décide de la fourchette de K7 (2 à 6 j). Rien dans le dépôt ne permet de la préjuger.
- **Le noyau est-il borné par l'émission LSU ou par la latence ?** Cela classe les quatre options de §3.7 ; le classement inverse ferait dépenser des jours et une décision de format avant d'avoir corrigé un défaut qui se répare en une demi-journée.
- **Le 2,07× se transporte-t-il ?** Non préjugeable. Les deux machines diffèrent sur la bande passante, le cache de dernier niveau, la mémoire partagée, la granularité de transaction et le rapport calcul/mémoire. Le résultat peut être meilleur comme franchement moins bon, **et c'est un résultat dans les deux cas**.
- **cuBLAS f16 à n = 1 est-il dégénéré sur Ada ?** Cela décide du dénominateur du rapport titre.
- **La latence de lancement réelle, et donc si le CUDA Graph est nécessaire ou cosmétique.** Aucun chiffre n'est avancé ici : les estimations circulant sur le sujet vont de 1 à 4 µs par lancement, ce qui donne entre 9 % et 26 % du plancher mémoire LLVQ — un facteur 3 sur une quantité qui décide d'un poste de travail.

### 6.4 Hors périmètre du port, et inchangé par lui

Les trois obstacles de `fiche-4b.md` §6.10 restent entiers côté CUDA : pas de cache KV donc **le noyau n'a toujours aucun appelant** ; la rotation d'incohérence GPU n'existe sur aucun backend et n'est payée que par le bras quantifié (144 par token, coût en latence **inconnu et non chiffrable** — le plancher de soumission Metal rendrait 144 dispatches naïfs suffisants pour effacer toute l'avance) ; le prefill exige un second chemin dense.

Un 2,x× CUDA sur `thesis` serait **exactement le même type d'objet** que le 2,07× Metal : un rapport sur les projections seules, minorant du rapport ALU/mémoire pur, majorant du rapport de bout en bout. La revendication à viser est précise et légitime — « décodeur Leech multi-coquilles fusé, 252 matrices, modèle entier, sur la classe de matériel du papier, face à un noyau que ses propres auteurs déclarent mono-coquille et plus lent que QTIP ». Ce n'est pas « LLVQ est plus rapide en inférence », et le glissement d'étiquette est le risque n° 1 documenté trois fois dans ce dépôt.
