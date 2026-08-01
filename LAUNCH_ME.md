# Faire tourner le modèle

Qwen3-4B quantifié à 2,1696 bits/poids. **Un fichier de 1,771 Go**, qui démarre
sans checkpoint, sans cache Hugging Face et sans réseau.

📦 **[huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit)**

| | LLVQ 2 bits | FP16 | |
|---|---|---|---|
| sur disque | **1,771 Go** | 8,045 Go | **×4,54** |
| perplexité WikiText-2 | 16,9617 | 12,2336 | ×1,386 |
| projections, un token *(noyau fusé)* | **10,46 ms** | 21,69 ms | **×2,07** |
| pas de décodage complet *(idem + lm_head)* | **78,2 tok/s** | 41,6 tok/s | **×1,88** |

⚠️ **Les deux dernières lignes ne sont pas ce que fait `bin/run`.** Elles sont
mesurées par `bin/thesis` (§ *Valider la thèse*), sur le même fichier et la même
machine, avec le noyau fusé. Ce noyau n'est **pas encore branché** dans le
runner livré : `bin/run` décode les poids en mémoire puis fait un matvec
ordinaire, et ne gagne donc aucune vitesse. C'est le dernier chantier.

---

## En trois commandes

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
```

```bash
hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-llvq.bin --local-dir .
```

```bash
cargo run --release -p llvq-llm --features metal --bin run -- qwen3-4b-llvq.bin metal 24
```

Sur autre chose qu'un Mac, enlever `--features metal` et remplacer le second
`metal` par `cpu`. Le dernier argument est le nombre de tokens à générer.

Attendu :

```
loaded qwen3-4b-llvq.bin — 252 quantized matrices + 146 carried tensors

── model
   1.771 GB on disk against 8.045 GB in FP16  →  ×4.54
   3633315840 weights quantized, 389152256 carried at f16

── "The capital of France is"
   → Paris. (True or False?)
── "In 1969, the first humans landed on"
   → the moon.
```

**Prérequis** : Rust stable et ~8 Go de RAM libre. Rien d'autre — pas de
Python, pas de CUDA, pas de compte Hugging Face pour l'exécution.

## Valider, dans l'ordre

Quatre vérifications, de la plus rapide à la plus longue. Chacune répond à une
question différente, et aucune ne demande de nous croire sur parole.

### 1. Le cœur mathématique est-il juste ? — ~45 s

```bash
cargo test --release -- --include-ignored
```

106 tests. Les invariants de Λ₂₄ (nombre de baisers 196 560, série thêta), la
recherche exacte du plus proche voisin contre la force brute, la bijectivité de
l'index 48 bits, la boucle GPTQ contre un minimiseur analytique indépendant, et
les allers-retours bit pour bit des cinq formats runtime.

### 2. Le fichier est-il vraiment autonome ? — ~1 min

Environnement vide, `HOME` inexistant : aucun cache Hugging Face n'est
joignable, aucune variable ne peut aider.

```bash
env -i HOME=/nonexistent PATH=/usr/bin:/bin ./target/release/run qwen3-4b-llvq.bin cpu 12
```

S'il répond, le fichier est bien le modèle entier.

### 3. La thèse tient-elle ? — ~4 min

```bash
cargo run --release -p llvq-metal --bin thesis
```

Prend le `.llvq`, transcode ses 252 matrices vers le format noyau, **vérifie
les 1 105 920 lignes de sortie** contre une référence CPU en f64, puis mesure
un token de projections dans les deux formats. Sortie sur M3 Max :

```
  1105920 lignes sur 252 matrices — pire erreur LLVQ 3.4e-8·Σ|w·x|

  format                            ms      Go lus        Go/s   vs FP16
  FP16                          21.691        7.27         335     1.00×
  LLVQ fusé (Slot32)            10.460        2.50         239     2.07×
```

Il faut un Mac (Metal) et ~12 Go de RAM libre. Le fichier attendu est
`~/llvq-q4b.llvq` — celui des projections, avant scellement ; passer un chemin
en argument pour un autre.

### 4. La qualité est-elle celle annoncée ? — ~20 min

```bash
cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 999 metal
```

Perplexité WikiText-2 à 4096 de contexte. Attendu : **16,9617** contre 12,2336
pour la baseline FP32.

## Ce qu'on gagne, ce qu'on ne gagne pas

**On gagne de la place** : 8,045 Go → 1,771 Go sur disque, ×4,54.

**On gagne de la vitesse, mais pas encore dans le runner** : le noyau fusé est
écrit, vérifié et mesuré à ×2,07 sur les projections du modèle entier ; il
n'est pas branché dans `bin/run`.

**On paie la vitesse en mémoire vive.** Le fichier fait 2,1696 bits/poids, mais
le format que le noyau lit en RAM en fait **5,51** : c'est le prix des offsets
fixes qui suppriment la divergence. Le transcodage se fait au chargement, une
fois. D'un même fichier on peut charger trois formats, selon ce qu'on optimise :

| format en RAM | b/poids | projections en RAM | vitesse |
|---|---|---|---|
| `Grouped32` (masques imbriqués) | 3,35 | 1,52 Go | 0,68× le FP16 |
| `Flat32` | 4,54 | 2,06 Go | 0,90× |
| **`Slot32`** *(celui mesuré)* | **5,51** | **2,50 Go** | **2,07×** |

Même au plus large, le modèle chargé fait ~3,3 Go contre 8,045 en FP16.

**Sur un 4B c'est une démonstration** — il tenait déjà partout. L'intérêt est
ailleurs : un 70B fait 140 Go en FP16 et ne tourne sur aucune machine locale ;
à ce taux il ferait ~20 Go et tiendrait dans un Mac.

## Refaire le modèle soi-même

Quantifier depuis le checkpoint d'origine — ~3,5 h sur un M3 Max. Le run
vérifie son propre fichier en le décodant et en exigeant les poids évalués, bit
pour bit :

```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=q4b.llvq \
  cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- \
  64 2048 12 4096 metal nogs leech1c12 999 rot
```

Ce fichier ne porte que les projections quantifiées. Pour le rendre autonome —
y sceller l'embedding, les normes, la config et le tokenizer :

```bash
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- q4b.llvq qwen3-4b-llvq.bin
```

`LLVQ_THREADS=4` limite le pool d'encodage si la machine doit rester
utilisable pendant ce temps.

## Le format

Défini par [`llvq-artifact`](llvq-artifact/) — **zéro dépendance**. Lire un
modèle quantifié ne doit pas exiger un runtime de tenseurs : l'arbre complet de
ce crate fait trois crates maison, contre 690 pour le côté modèle.

```
"LVQ2" · n_matrices · [matrices]  index 47 bits + gain, empaquetés dense
                    · [tenseurs bruts]  ce que le quantifieur n'a pas touché, f16
                    · [blobs]           config.json, tokenizer.json
```

Le format **sur disque** ne bouge pas : c'est le rang de permutation, optimal
en bits. Les formats **runtime** du tableau ci-dessus en sont transcodés au
chargement (~3 s pour un 4B sur 12 cœurs) et ne touchent pas au fichier.

⚠️ Ce n'est **ni** GGUF, **ni** AWQ, **ni** safetensors. `transformers`,
`llama.cpp`, vLLM et TGI ne le lisent pas. Un lecteur dans un autre langage est
direct à écrire — c'est la raison d'être du crate sans dépendance — mais il
n'existe pas encore.

## Licence

Le modèle est sous Apache 2.0, héritée de
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). Le code est
MIT OR Apache-2.0.
