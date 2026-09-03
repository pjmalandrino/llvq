# Faire tourner le modèle publié

Qwen3-4B quantifié à 2 bits sur le réseau de Leech tient dans un fichier de 1,771 Go (*mesuré*,
[docs/fiche-4b.md](docs/fiche-4b.md)) qui démarre seul : sans checkpoint, sans cache Hugging Face,
sans réseau. L'état courant du projet est dans [docs/ETAT.md](docs/ETAT.md), le fil des changements
dans [docs/HISTORIQUE.md](docs/HISTORIQUE.md), les règles de mesure dans [docs/METHODE.md](docs/METHODE.md).

## 1. Le fichier

`qwen3-4b-llvq.bin` fait 1 770 527 533 octets (*mesuré*, [docs/fiche-4b.md](docs/fiche-4b.md)) et
vit sur le Hub : [huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit).
Son sha256 est `9db213ef9fa9d7d7000789a8a529ce9459ce9ba6002ef5a72fd5a1c05c1c84b0`. Un hash
différent est un autre fichier.

| section | contenu | octets | provenance |
|---|---|---|---|
| matrices | 252 projections quantifiées, 3 633 315 840 poids, index 47 bits + 1 bit de gain par bloc de 24 | 980 790 202 | *mesuré*, fiche-4b |
| tenseurs f16 | 146 tenseurs portés : embedding 388 956 160 valeurs, normes 196 096 | 778 313 898 | *mesuré*, fiche-4b |
| blobs | `config.json` (726 o), `tokenizer.json` (11 422 654 o) | 11 423 433 | *mesuré*, fiche-4b |

Les projections pèsent 2,1595 b/poids, queue incluse au dénominateur (*calculé*, fiche-4b ;
2,1696 queue exclue, même fichier). Le modèle entier pèse 3,5213 b/param sur disque, embedding f16
compris (*calculé*, fiche-4b). Le conteneur porte le magic `LVQ2`. Le format est défini par
[`llvq-artifact`](llvq-artifact/), un crate à zéro dépendance. Son arbre fait 3 crates contre 261
pour le côté modèle (*mesuré*, [README.md](README.md)).

Ce fichier n'est ni un GGUF, ni un AWQ, ni un safetensors. `transformers`, `llama.cpp`, vLLM et TGI
ne le lisent pas. Le seul lecteur est ce dépôt. Un lecteur dans un autre langage n'existe pas ; il
lui faudrait aussi `llvq-search` et `llvq-core` pour l'index de Leech.

## 2. Télécharger et lancer

Prérequis : Rust stable. Le téléchargement est la seule étape qui touche au réseau.

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-llvq.bin --local-dir .
shasum -a 256 qwen3-4b-llvq.bin        # attendu : 9db213ef…84b0, 1 770 527 533 octets
```

Générer sur Mac (Metal) ou sur toute machine (CPU) avec `bin/run`, le chemin dense :

```bash
cargo run --release -p llvq-llm --features metal --bin run -- qwen3-4b-llvq.bin metal 24
```

Hors Mac, retirer `--features metal` et remplacer l'argument `metal` par `cpu`. La feature cargo et
l'argument sont deux choses distinctes ; demander `metal` sans la feature est une erreur. Le dernier
argument est le nombre de tokens. Le programme imprime `252 quantized matrices + 146 carried
tensors`, puis quatre prompts échantillonnés, donc non reproductibles à la lettre.

`bin/run` décode tous les poids en f16 : le modèle résident fait 8,045 Go quel que soit le fichier
(*mesuré*, fiche-4b). Pic RSS : 9,79 Go en `cpu`, 17,41 Go en Metal (*mesuré*, fiche-4b). Compter
10 Go de RAM libre en CPU et 17,4 Go en Metal. Débit : 42,7 tok/s sur L40S avec cache KV (*mesuré*,
[docs/mesures/mini-2026-08-05.txt](docs/mesures/mini-2026-08-05.txt)) ; aucune mesure Mac depuis le
cache KV.

Le chemin fusé, sur Linux + CUDA, est `bin/fusedrun`. Il décode et multiplie sans repasser par une
matrice dense. La configuration servie v1 tient en trois variables :

```bash
LLVQ_EMBED=q8 LLVQ_ROT_SHARE=1 LLVQ_FUSE=1 \
  cargo run --release -p llvq-llm --features cuda --bin fusedrun -- qwen3-4b-llvq.bin 128
```

| chemin | tok/s [plage] | Go carte | tokens gloutons | provenance |
|---|---|---|---|---|
| fusé, config v1 (`planes14` + q8 + rotation hissée + fusion) | 100,6 [99,9–100,7] | 2,57 | 128, divergence du dense au token 89 | *mesuré*, [d1-fusion-servie-2026-08-24.txt](docs/mesures/d1-fusion-servie-2026-08-24.txt) |
| dense f16 (le bras compagnon) | 43,5 [43,4–43,5] | 8,04 | référence | *mesuré*, [b2-fusedrun-plages-2026-08-18.txt](docs/mesures/b2-fusedrun-plages-2026-08-18.txt) |

Le rapport brut contre ce bras ne se cite jamais seul : il recopie 778 Mo de vocabulaire par
token. Le rapport à tête identique, ×1,11 [1,11–1,11], mesure le noyau (*calculé* sur les médianes
B2). NVRTC compile le noyau au démarrage ; `LLVQ_NVRTC_ARCH=compute_80` cible l'A100, où le même
noyau rend 0,79× FP16 (*mesuré*, [f4-a100-2026-08-18.txt](docs/mesures/f4-a100-2026-08-18.txt)).

## 3. Valider, dans l'ordre

Quatre vérifications, de la plus rapide à la plus longue.

| n° | question | commande | attendu | durée |
|---|---|---|---|---|
| 1 | le cœur mathématique est-il juste | `cargo test --release` | tout vert ; les tests `ignored` sont les sweeps d'archive | quelques minutes (*estimé*) |
| 2 | le fichier est-il autonome | `env -i HOME=/nonexistent PATH=/usr/bin:/bin ./target/release/run qwen3-4b-llvq.bin cpu 12` | une réponse, sans cache HF ni réseau | 4 à 5 min (*estimé* ; 255,7 s *mesurés* avant le cache KV, [audit-publication-2026-08-03.md](docs/archive/audit-publication-2026-08-03.md)) |
| 3 | la thèse tient-elle | `cargo run --release -p llvq-metal --bin thesis -- qwen3-4b-llvq.bin` | 1 105 920 lignes vérifiées, pire erreur 3,4e-8·Σ\|w·x\|, ×2,03 [2,03–2,10] (*mesuré*, banc 7 bras, médiane du rapport round par round, [k1-metal-2026-08-05.txt](docs/mesures/k1-metal-2026-08-05.txt)) | ~4 min (*estimé*), Mac, ~12 Go libres |
| 4 | la qualité est-elle celle annoncée | les deux commandes `ppl` ci-dessous | 16,9415 contre 12,2361 (*mesuré*, fiche-4b), empreinte `3f1baca9033bf251` des deux côtés | ~15 min (*estimé*) plus le checkpoint la première fois |

Étape 1. La boucle rapide couvre les invariants de Λ₂₄ (196 560 baisers, série thêta) et la
recherche exacte contre la force brute. Elle couvre aussi la bijectivité de l'index 48 bits, la
boucle GPTQ contre un minimiseur analytique indépendant, et les allers-retours bit pour bit des
cinq formats runtime. Les sweeps d'archive se lancent une fois le fichier téléchargé et se comptent
en dizaines de minutes : 17 min sans finir `llvq-artifact` le 2026-08-08, 10 min 51 s pour ses
45 tests seuls (*mesuré*, [CLAUDE.md](CLAUDE.md) §2 et §7) :

```bash
LLVQ_SEALED_ARTIFACT=$PWD/qwen3-4b-llvq.bin cargo test --release -- --include-ignored
```

Sans l'archive, ces sweeps échouent en nommant le fichier.

Étape 2. `HOME` inexistant et environnement vide : aucun cache Hugging Face n'est joignable. La
mesure de 255,7 s précède le cache KV du commit `9c24d26` ; aucune mesure depuis.

Étape 3. `bin/thesis` transcode les 252 matrices vers le format noyau, vérifie chaque ligne de sortie
contre une référence CPU f64, puis chronomètre un token de projections dans les deux formats. Il
mesure `Slot32` sur Metal : FP16 21,9 ms médiane pour 7,27 Go lus, LLVQ 10,7 ms pour 2,50 Go
(*mesuré*, [k1-metal-2026-08-05.txt](docs/mesures/k1-metal-2026-08-05.txt)). Le rapport est une
plage : les millisecondes dérivent de 2,029× à 2,080× sur un binaire inchangé. Ce banc n'est pas
le chemin servi, qui est `Planes14` sur CUDA. Sans argument il cherche `~/llvq-q4b.llvq`, un fichier
de travail non publié.

Étape 4. Perplexité WikiText-2, contexte 4096, 12 fenêtres, f16 des deux côtés :

```bash
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal qwen3-4b-llvq.bin
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal
```

Attendu : 16,9415 sur le fichier scellé contre 12,2361 sur le checkpoint, soit ×1,3846 (*mesuré*,
fiche-4b). Les deux lignes de résultat doivent porter la même empreinte de tokens.

## 4. Face au FP16 et au 4 bits

Au 4B, LLVQ passe devant l'AWQ sur le disque et sur la mémoire servie, et perd sur la qualité. La
référence utile est le 4 bits : personne ne
choisit entre 2 et 16 bits. Le 4 bits mesuré est l'AWQ officiel de Qwen, même carte,
même harnais, mêmes empreintes de tokens.

| axe | LLVQ 2 bits | FP16 | AWQ 4 bits officiel | provenance |
|---|---|---|---|---|
| disque | 1,771 Go | 8,045 Go | 2,67 Go | 1,771 et 8,045 *mesurés*, fiche-4b ; 2,67 *mesuré*, [campagne-finale-2026-08-07.md](docs/campagne-finale-2026-08-07.md) ; ×4,54 sur FP16 (*calculé*) |
| RAM du chemin dense (`bin/run`) | 9,79 Go cpu, 17,41 Go Metal | 8,045 Go résidents | non mesuré chez nous | *mesuré*, fiche-4b |
| VRAM du chemin fusé, b/param modèle entier | 2,56 Go, 5,162 (`Planes14` + q8 sans fusion) ; 2,57 Go en config v1, +3 686 400 octets | 8,04 Go, 16,0 | 5,302 dans son moteur | 2,56 Go *mesuré*, b2 ; 5,162 *calculé* sur octets mesurés, [rtbits-planes-8b-2026-08-09.txt](docs/mesures/rtbits-planes-8b-2026-08-09.txt) ; 2,57 Go et +3 686 400 o *mesurés*, d1 |
| vitesse, L40S | 100,6 tok/s ; ×1,11 à tête identique | 43,5 tok/s | 200,5 tok/s dans vLLM, autre pile, ne se divise pas | *mesuré*, d1, b2, [awq-vllm-4b-2026-08-17.txt](docs/mesures/awq-vllm-4b-2026-08-17.txt) |
| perplexité WikiText-2 | 16,9415 (×1,385) | 12,2361 | ×1,105 | *mesuré*, fiche-4b, [a4-campagne-2026-08-06.txt](docs/mesures/a4-campagne-2026-08-06.txt) |
| MMLU 5-shot micro | 55,59 ± 1,35 | 70,32 ± 1,28 | 70,04 | *mesuré*, a4-campagne, empreinte `65dcd53655e8bfa5` |

Mémoire : LLVQ passe sous l'AWQ en b/param modèle entier, 5,162 contre 5,302. Toute comparaison
mémoire se dit dans cette comptabilité, embedding compris ; un b/poids de projections contre un
b/param de modèle entier trompe. Qualité : LLVQ perd 14,73 pp de MMLU là où le 4 bits en perd 0,28.
Sur un 4B, le 4 bits domine partout sauf le disque. L'écart fond avec la taille (*mesuré* : 7,49 pp
au 8B, [mmlupair-4b-8b-2026-08-13.txt](docs/mesures/mmlupair-4b-8b-2026-08-13.txt) ; 6,09 pp au 14B,
[mmlupair-14b-2026-08-17.txt](docs/mesures/mmlupair-14b-2026-08-17.txt)) sans faire une loi
d'échelle ; voir [docs/ETAT.md](docs/ETAT.md). Le noyau fusé n'est pas branché dans `bin/run` : la
démo portable reste dense, le gain de vitesse vit dans `bin/fusedrun`.

## 5. Refaire le modèle soi-même

La quantification du 4B prend 4,01 h sur un M3 Max, 14 447 s (*mesuré*, fiche-4b). Le run vérifie
son propre fichier en le décodant et en exigeant les poids évalués bit pour bit :

```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=q4b.llvq \
  cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- \
  64 2048 12 4096 metal nogs leech1c12 999 rot
```

Positionnels : 64 fenêtres de 2048 tokens de calibration (131 072 tokens) ; 12 fenêtres de 4096
pour l'évaluation ; backend `metal`. Puis `nogs`, échelles de groupe désactivées ; codebook
`leech1c12`, boule Λ₂₄(12) et 1 bit de gain ; graine 999 ; rotation d'entrée. `fast-linalg` est
requis en pratique : sans lui la factorisation est 40× plus lente pour un résultat bit-identique
(*mesuré*, README). `LLVQ_THREADS=4` limite le pool d'encodage.

Le fichier produit ne porte que les projections. Le rendre autonome, avec embedding, normes, config
et tokenizer :

```bash
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- q4b.llvq qwen3-4b-llvq.bin
```

Cette recette reproduit la méthode sans reproduire les octets. Trois écarts au run publié, tous
documentés dans fiche-4b. Le shard C4 de calibration est passé de `00000` à `00001` après le run.
Le magic du conteneur est passé de `LVQ2` à `LVQ4`, qui stocke en plus l'empreinte du codebook.
`calib.rs` accumule AᵀA en f32 sur l'accélérateur, donc un re-run sur CUDA rend d'autres poids,
écart non chiffré. Un re-run rend 1,771 Go à 2,1595 b/poids dans un fichier non identique. La CI
vérifie le code : clippy, tests, garde zéro dépendance sur les cinq crates du cœur (`llvq-core`, `llvq-search`, `llvq-quant`,
`llvq-artifact`, `llvq-bench` ; [ci.yml](.github/workflows/ci.yml)). Elle ne vérifie pas
les octets de l'artefact.

## 6. Licence

Le modèle est sous Apache 2.0, héritée de [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). Le
code est sous MIT OR Apache-2.0 ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).
