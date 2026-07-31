# Faire tourner le modèle

Qwen3-4B quantifié à 2,16 bits/poids. **Un fichier de 1,771 Go**, qui démarre
sans checkpoint, sans cache Hugging Face et sans réseau.

📦 **[huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit)**

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

## Vérifier qu'il est vraiment autonome

C'est le test qui définit « fini » pour ce projet : environnement vide, `HOME`
inexistant donc aucun cache Hugging Face joignable.

```bash
env -i HOME=/nonexistent PATH=/usr/bin:/bin ./target/release/run qwen3-4b-llvq.bin cpu 14
```

S'il répond, le fichier est bien le modèle entier.

## À quoi s'attendre, et à quoi ne pas s'attendre

**Ce qu'on gagne** : 8,045 Go → 1,771 Go, ×4,54. La perplexité WikiText-2
passe de 12,2336 à 16,9617.

**Ce qu'on ne gagne pas** : de la vitesse. Le lecteur décode les poids en
mémoire puis fait un produit matrice-vecteur ordinaire. Le noyau fusé, qui
seul transformerait la compression en débit, n'est pas écrit.

Sur un 4B c'est une démonstration — il tenait déjà partout. L'intérêt est
ailleurs : un 70B fait 140 Go en FP16 et ne tourne sur aucune machine locale ;
à ce taux il ferait ~20 Go et tiendrait dans un Mac.

## Refaire le modèle soi-même

Quantifier depuis le checkpoint d'origine — ~4 h sur un M3 Max. Le run vérifie
son propre fichier en le décodant et en exigeant les poids évalués, bit pour
bit :

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

Défini par [`llvq-artifact`](llvq-artifact/) — ~250 lignes, **zéro
dépendance**. Lire un modèle quantifié ne doit pas exiger un runtime de
tenseurs : l'arbre complet de ce crate fait trois crates maison, contre 690
pour le côté modèle.

```
"LVQ2" · n_matrices · [matrices]  index 47 bits + gain, empaquetés dense
                    · [tenseurs bruts]  ce que le quantifieur n'a pas touché, f16
                    · [blobs]           config.json, tokenizer.json
```

⚠️ Ce n'est **ni** GGUF, **ni** AWQ, **ni** safetensors. `transformers`,
`llama.cpp`, vLLM et TGI ne le lisent pas. Un lecteur dans un autre langage
est direct à écrire — c'est la raison d'être du crate sans dépendance — mais
il n'existe pas encore.

## Licence

Le modèle est sous Apache 2.0, héritée de
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). Le code est
MIT OR Apache-2.0.
