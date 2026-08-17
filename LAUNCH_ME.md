# Faire tourner le modèle

Qwen3-4B quantifié à **2,1595 bits/poids** sur ses 3 633 315 840 poids de
projection (2,1696 si l'on exclut la queue du dénominateur — même fichier, deux
conventions, toutes deux exactes). **Un fichier de 1,771 Go**, qui démarre sans
checkpoint, sans cache Hugging Face et sans réseau.

📦 **[huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit)**

| | LLVQ 2 bits | FP16 | |
|---|---|---|---|
| sur disque | **1,771 Go** | 8,045 Go | **×4,54** |
| perplexité WikiText-2, f16, **sur le fichier publié** | **16,9415** | 12,2361 | ×1,385 |
| MMLU 5-shot, micro, **sur le fichier publié** | **56,09 ± 1,36** | 70,42 ± 1,28 | −14,33 pp |
| projections, un token *(noyau fusé, modèle entier)* | **10,5 – 11,0 ms** | 21,7 – 22,7 ms | **×2,06 – 2,08** |
| + lm_head f16 ajouté **analytiquement** *(jamais exécuté)* | *78,2 tok/s calculé* | *41,6 calculé* | *×1,88 — un majorant* |
| génération réelle de `bin/run` | **2,2 – 7,6 tok/s mesurés** | — | — |

⚠️ **Les deux lignes de vitesse ne sont pas ce que fait `bin/run`.** Elles sont
mesurées par `bin/thesis` (§ *Valider la thèse*), sur le même fichier et la même
machine, avec le noyau fusé. Ce noyau n'est **pas branché** dans le runner
livré : `bin/run` décode les poids en mémoire puis fait un matvec ordinaire, et
ne gagne donc aucune vitesse — **il génère entre 2,2 et 7,6 tok/s, et le débit
décroît avec la longueur** parce que `generate` rejoue tout le préfixe à chaque
pas (pas de cache KV). C'est le dernier chantier.

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

Sur autre chose qu'un Mac, enlever `--features metal` **et** remplacer le second
`metal` par `cpu` — la feature cargo et l'argument sont deux choses distinctes,
et demander `metal` sans la feature est une erreur. Le dernier argument est le
nombre de tokens à générer.

Forme de la sortie :

```
loaded qwen3-4b-llvq.bin — 252 quantized matrices + 146 carried tensors
   running at dtype f16

── model
   1.771 GB on disk against 8.045 GB in FP16  →  ×4.54
   3633315840 weights quantized, 389152256 carried at f16

── "The capital of France is"
   → …
```

…puis trois autres prompts (l'alunissage de 1969, `def fibonacci(n):`,
l'ébullition de l'eau). `bin/run` échantillonne : les continuations ne sont pas
reproductibles à la lettre.

**Prérequis** : Rust stable, et de la RAM libre — **~10 Go en `cpu` (9,79 Go de
pic RSS mesuré), et au moins 17,4 Go en Metal (17,41 Go mesurés)**. Le lecteur
décode tous les poids en mémoire : le modèle résident fait 8,045 Go de f16, quel
que soit le poids du fichier sur disque. Rien d'autre pour **faire tourner le
modèle** — pas de Python, pas de CUDA, pas de compte Hugging Face. (Le
téléchargement et la vérification n° 4 sont les deux seules étapes qui touchent
au réseau.)

## Valider, dans l'ordre

Quatre vérifications, de la plus rapide à la plus longue. Chacune répond à une
question différente, et aucune ne demande de nous croire sur parole.

### 1. Le cœur mathématique est-il juste ? — ~2 min sur un clone frais

```bash
cargo test --release -- --include-ignored
```

La totalité de la suite, sans un seul test ignoré (~20 s à chaud, le reste est
la compilation). Les invariants de Λ₂₄ (nombre de baisers 196 560, série
thêta), la recherche exacte du plus proche voisin contre la force brute, la
bijectivité de l'index 48 bits, la boucle GPTQ contre un minimiseur analytique
indépendant, et les allers-retours bit pour bit des cinq formats runtime.

### 2. Le fichier est-il vraiment autonome ? — ~4-5 min

Environnement vide, `HOME` inexistant : aucun cache Hugging Face n'est
joignable, aucune variable ne peut aider.

```bash
env -i HOME=/nonexistent PATH=/usr/bin:/bin ./target/release/run qwen3-4b-llvq.bin cpu 12
```

S'il répond, le fichier est bien le modèle entier. (255,7 s mesurées : c'est le
chemin `cpu`, le plus lent, et il n'y a pas de cache KV.)

### 3. La thèse tient-elle ? — ~4 min

```bash
cargo run --release -p llvq-metal --bin thesis -- qwen3-4b-llvq.bin
```

Prend le fichier, transcode ses 252 matrices vers le format noyau, **vérifie les
1 105 920 lignes de sortie** contre une référence CPU en f64, puis mesure un
token de projections dans les deux formats.

Il faut un Mac (Metal) et ~12 Go de RAM libre. Le banc lit le fichier scellé
téléchargé plus haut ; **sans argument il cherche `~/llvq-q4b.llvq`**, un
fichier de travail qui n'est publié nulle part.

Ordre de grandeur attendu, d'une exécution du 2026-08-01 : FP16 21,7 ms à
7,27 Go lus, LLVQ fusé 10,5 ms à 2,50 Go, soit ×2,07 — et
`pire erreur LLVQ 3.4e-8·Σ|w·x|` sur les 1 105 920 lignes. La dérive thermique
déplace les temps de 4 à 5 % d'une exécution à l'autre ; **le rapport, lui, ne
bouge que de 0,8 %**, et la ligne d'erreur est reproductible au chiffre près.

### 4. La qualité est-elle celle annoncée ? — ~15 min, plus le téléchargement du checkpoint la première fois

```bash
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal qwen3-4b-llvq.bin
```
```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal
```

Perplexité WikiText-2 à 4096 de contexte, f16 des deux côtés, mêmes fenêtres.
Attendu : **16,9415** contre **12,2361**, ×1,3846 — et la **même empreinte de
tokens `3f1baca9033bf251`** sur les deux lignes de résultat. Si les empreintes
diffèrent, le rapport ne veut rien dire : c'est à ça que sert la ligne.

## Ce qu'on gagne, ce qu'on ne gagne pas

**On gagne de la place sur le disque** : 8,045 Go → 1,771 Go, ×4,54.

**Mais la bonne référence n'est pas le FP16.** Sur ce 4B, un 4 bits ordinaire
(`mlx_lm.convert -q --q-bits 4 --q-group-size 64`, même machine, même
checkpoint) fait 2,263 Go sur disque : notre fichier est **×1,28 plus petit,
soit 22 % de disque en moins — et rien d'autre qui soit mesuré en notre
faveur**. Il génère bien plus vite, et sa qualité n'a jamais été mesurée face à
la nôtre : cette case-là est **vide, pas faible**. Analyse :
[`docs/archive/face-au-4-bits.md`](docs/archive/face-au-4-bits.md).

**On gagne de la vitesse, mais pas encore dans le runner** : le noyau fusé est
écrit, vérifié et mesuré à ×2,06–2,08 sur les projections du modèle entier ; il
n'est pas branché dans `bin/run`, qui n'a d'ailleurs pas de cache KV.

**On paie la vitesse en mémoire vive.** Le format que le noyau lit en RAM coûte
**5,51 b/poids** sur les projections — plus que les **4,179 b/poids** que le
noyau AWQ 4 bits lit dans la **même** comptabilité (banc à 7 bras du 2026-08-11,
[`docs/data/echelle-formats.csv`](docs/data/echelle-formats.csv)). Le
transcodage se fait au chargement, une fois. D'un même fichier on peut charger
trois formats :

> 🚨 **Cette phrase disait « 5,51 b/poids, soit plus que les 4,50 d'un 4 bits
> ordinaire », et c'était la faute que l'errata du lot A qualifie de GRAVE —
> corrigé le 2026-08-17.** Deux erreurs empilées : (a) 5,51 est un
> b/**poids** de **projections seules**, 4,50 un b/**param** de **modèle
> entier, embedding quantifié compris** — deux dénominateurs ; (b) le 4,50 est
> le MLX q4 g64, un artefact qui n'est entré dans **aucune** campagne, alors
> que le seul 4 bits mesuré est l'AWQ officiel de Qwen.
> **Et dans la comptabilité licite la conclusion s'inverse** : en b/param
> modèle entier, `Planes14` + embedding q8 pèse **5,162** contre **5,302**
> pour l'AWQ officiel au 4B — **sous** le 4 bits déployé, et sous lui aussi au
> 8B (5,322 contre 5,956) et au 14B (5,106 contre 5,404). Ce paragraphe
> annonçait donc l'inverse de ce que la mesure dit.
> ⚠️ Le reste de cette section décrit l'état du **2026-08-12** : le layout
> servi n'est plus `Slot32` mais `Planes14` (4,804 b/poids), et le noyau est
> branché dans `bin/fusedrun` depuis le 2026-08-06 — c'est `bin/run`, la démo
> de génération, qui reste dense.

| format en RAM | b/poids | projections en RAM | vitesse | mesurée sur |
|---|---|---|---|---|
| `Grouped32` (masques imbriqués) | 3,50 | 1,59 Go | 0,68× le FP16 | `gate_proj` seul |
| `Flat32` | 4,68 | 2,12 Go | 0,90× | `gate_proj` seul |
| **`Slot32`** *(celui mesuré)* | **5,51** | **2,50 Go** | **2,06–2,08×** | **le modèle entier** |

Métrique unique : charge utile + adressage + queue f32 + échelles de ligne, sur
tous les poids de projection.

⚠️ **Aucun de ces trois formats n'est ce que charge `bin/run`.** Le runner livré
décode en f16 dense : modèle résident 8,045 Go, pic RSS mesuré 9,79 Go en CPU et
17,41 Go en Metal. Le gain de place est sur le disque.

**Sur un 4B c'est une démonstration** — il tenait déjà partout. L'intérêt serait
ailleurs : un 70B fait 140 Go en FP16 et ne tourne sur aucune machine locale. À
ce taux il ferait ~23 Go sur disque — mais **33 Go en RAM avec `Grouped32` et
51 Go avec `Slot32`**, soit *plus* que les ~40 Go d'un 4 bits. Aucun 70B n'a
jamais été quantifié ici, et aucun cache KV n'est budgété dans ce calcul.

## Refaire le modèle soi-même

⚠️ **Cette commande reproduit la méthode, pas les octets** : le shard de
calibration C4 est passé de `00000` à `00001` après le run publié, et le magic
du conteneur a bougé. Un re-run aujourd'hui produit un fichier différent,
également valide. Il n'y a pas de CI.

Quantifier depuis le checkpoint d'origine — **~4 h** sur un M3 Max (14 447 s
mesurées). Le run vérifie son propre fichier en le décodant et en exigeant les
poids évalués, bit pour bit :

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
ce crate fait trois crates maison, contre **261 paquets** pour le côté modèle
(291 avec `metal,fast-linalg`).

```
"LVQ2" · n_matrices · [matrices]  index 47 bits + gain, empaquetés dense
                    · [tenseurs bruts]  ce que le quantifieur n'a pas touché, f16
                    · [blobs]           config.json, tokenizer.json
```

Le format **sur disque** ne bouge pas : c'est le rang de permutation, optimal
en bits. Les formats **runtime** du tableau ci-dessus en sont transcodés au
chargement — `transcode()` est **mono-thread**, et son coût n'a jamais été
chronométré pour `Slot32` — et ne touchent pas au fichier.

⚠️ Ce n'est **ni** GGUF, **ni** AWQ, **ni** safetensors. `transformers`,
`llama.cpp`, vLLM et TGI ne le lisent pas. Un lecteur dans un autre langage est
tractable — c'est la raison d'être du crate sans dépendance — mais il n'existe
pas encore, et il lui faudrait aussi `llvq-search` et `llvq-core` pour l'index
de Leech.

## Licence

Le modèle est sous Apache 2.0, héritée de
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). Le code est
MIT OR Apache-2.0.
