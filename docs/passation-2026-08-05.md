# Passation — 2026-08-05

> Remplace [`passation-2026-08-04.md`](passation-2026-08-04.md). Tout ce qu'il
> faut pour reprendre sans relire l'historique.

---

## 1. Où on en est, en cinq lignes

Le lot-gate **K−1 est passé** : les trois jalons locaux que la spécification
exigeait avant d'engager un dollar sont faits, et deux d'entre eux corrigent
des chiffres publiés. Le **décodeur Leech tourne sur CUDA** et décide
exactement ce que décide le décodeur Rust, vérifié sur une L40S louée. Ce qui
reste avant un chiffre de vitesse défendable, c'est le **matvec fusé tuilé**
(lot K4) — et il n'a plus aucune inconnue de plomberie devant lui.

**Dépensé en machine à ce jour : ~0,36 $**, dont **0,01 $** pour tout le
travail CUDA de cette session.

## 2. État du dépôt

- Branche de travail : **`noyau-cuda`**, deux commits, partie de
  `campagne-mesure`, **non poussée**.
- `campagne-mesure` reste à 13 commits devant `main`, non fusionnée.
- `cargo clippy --all-targets` : zéro warning **sur les deux cibles**
  (macOS, et Linux via `CUDARC_CUDA_VERSION=12040 cargo clippy --target
  x86_64-unknown-linux-gnu`).
- `cargo test --release -- --include-ignored` : **146 cas passent**.

## 3. Ce qui a été mesuré, et ce que ça dément

### K−1(a) — la courbe bits↔vitesse, un seul protocole

`bin/thesis` mesure désormais **sept bras** sur les 252 projections, avec la
**même comptabilité d'octets** partout (payload + bases + queue f32 + échelles
de ligne f32). Journal : [`mesures/k1-metal-2026-08-05.txt`](mesures/k1-metal-2026-08-05.txt).

| layout | b/poids | vs FP16 |
|---|---|---|
| FP16 | 16,000 | 1,00× |
| **`Slot32`** | **5,510** | **2,03× [2,03–2,10]** |
| `Flat32` | 5,256 | 0,91× |
| `Grouped32` | 3,498 | 0,69× |

**La courbe est brutalement non linéaire.** `Flat32` n'économise que
0,254 b/poids sur `Slot32` et coûte 2,27× son temps ; `Grouped32` économise
2,012 et coûte 3,01×. **Conclusion de conception : reprendre des bits se fait
*dans* `Slot32` (plafond L ≤ 4), jamais en changeant de layout.** C'est ce qui
oriente le port CUDA — inutile d'y porter les deux autres décodeurs.

### K−1(b) — le conflit de bancs prédit n'existe pas sur Apple

La §3.2 de la spécification prescrivait un pas de tuile de 28 flottants contre
un conflit de bancs à 8 voies, transposé de l'arithmétique NVIDIA. Mesuré, sur
trois points et **les deux bras** : le padding ne gagne rien. Ce qui paie est
la **largeur de chargement** (`float4`), et elle paie des deux côtés, donc le
rapport ne bouge pas.

⚠️ **Conséquence pour K7 : ne pas budgéter ce gain comme acquis côté CUDA.** La
géométrie des bancs y est documentée et différente ; à re-mesurer, pas à
supposer.

### K−1(c) — le plafond L ≤ 4 vaut 4,7083 b/poids

Et pas « ~4,4 » (`format-noyau.md`) ni « 4,5 » (`cheatsheet-defense.md`) — trois
chiffres circulaient pour la même quantité. C'est un **majorant
inconditionnel** (`L ≤ 4 ⇒ width_slot ≤ 106 b ⇒ stride ≤ 14 o`), atteint sur
**4 708 799 groupes sur 4 708 800**. Un compte, pas une probabilité.
Journal : [`mesures/k1c-rtbits-2026-08-05.txt`](mesures/k1c-rtbits-2026-08-05.txt).

### Le fait de méthode, qui vaut pour tout ce qui suit

Six invocations du banc, code **identique**, octets et pires erreurs
identiques au chiffre : le rapport tient **2,03× à 2,09×**. L'entrelacement
des bras corrige la dérive *dans* un processus ; il ne peut rien contre celle
*entre* processus, qui déplace les deux bras ensemble.

> **Une troisième décimale sur ce rapport n'a pas de contenu.** Et un écart
> CUDA ↔ Metal sous ~5 % ne sera pas interprétable sans répéter les
> invocations. Détail : [`mesures/thesis-temoin-2026-08-04.txt`](mesures/thesis-temoin-2026-08-04.txt).

## 4. Le CUDA — ce qui existe et ce qu'il a répondu

`llvq-cuda` : `kernels/llvq_slot.cuh` (le décodeur porté),
`kernels/preflight.cu` (trois sondes), `src/gpu.rs` (plomberie cudarc),
`src/bin/preflight.rs`. Journal :
[`mesures/cuda-preflight-2026-08-05.txt`](mesures/cuda-preflight-2026-08-05.txt).

**Vérifié sur la carte** : 20 767 blocs, 498 408 slots — classe, gain et
niveau **exacts** ; produit scalaire à 1,15e-7·Σ|w·x|. La fixture balaie toute
la boule m ≤ 13, donc **plus dur que le fichier scellé**, plafonné à Λ₂₄(12).

| ce que la carte dit | valeur |
|---|---|
| L40S, 142 SM, mémoire partagée | 49 152 o/bloc, 102 400 o/SM |
| **L2** | **100,7 Mo — LUE.** Les sources tierces disaient 48 ou 96 ; ni l'une ni l'autre |
| `binary_version` | **89** sur les trois noyaux — pas de repli silencieux sur `compute_75` |
| mémoire locale | **0 octet** — pas de spill, le `#pragma unroll` tient |
| registres de `slot_dot` | **38**, sous le budget de 42 pour l'occupancy pleine sur Ada |

### Trois choses à ne pas redécouvrir

**On type-checke la cible Linux depuis le Mac.** `CUDARC_CUDA_VERSION=12040
cargo clippy -p llvq-cuda --target x86_64-unknown-linux-gnu --all-targets`.
Ça valide toute l'API cudarc sans carte et sans reconstruction d'image.

**Le décodeur se teste par `clang++` sur le Mac**, à travers
`tests/host_shim.h`, contre le décodeur Rust — `cargo test --release -p
llvq-cuda -- --include-ignored`. Deux secondes, zéro dollar, et il a déjà
attrapé une lecture hors bornes de `bases` qui aurait tué le contexte CUDA en
plein job. Le test compile la forme **concaténée**, pas seulement le `.cu` :
c'est ce qui manquait quand le premier job a échoué sur un `#include` que
NVRTC ne peut pas résoudre.

**`LLVQ_KERNEL_DIR` supprime la reconstruction d'image.** Les noyaux sont
`include_str!` par défaut (un run est alors reproductible depuis le binaire
seul), mais le job peut écrire les sources par heredoc et poser la variable.
La surcharge est **divulguée**, avec le sha256 de la chaîne réellement
compilée. Sans ça, une correction d'une ligne coûte 40-70 min.

## 5. Les pièges, payés cher

**Un Space finit en `RUNTIME_ERROR`, et c'est normal.** Il ne sert que de
service de construction et tourne sur du CPU : une image CUDA n'a rien à quoi
s'attacher. Attendre `APP_STARTING` ou `RUNNING`, qui disent que le *build* a
passé. Un vrai échec de construction s'affiche `BUILD_ERROR`.

**`nvidia-smi --query-gpu=driver.version` n'existe pas** ; c'est
`driver_version`. Sous `set -euo pipefail`, ça tue le job — d'où les `|| true`
sur toutes les sondes d'environnement.

**Les pronostics de la spécification ne sont pas tous justes.** Elle annonçait
les en-têtes CUDA absentes de l'image d'exécution : elles y sont. Elle
annonçait un `--locked` manquant dans `ops/Dockerfile.cuda` : il y était déjà,
le trou était dans `ops/Dockerfile`. Elle demandait de retirer une ligne
`Cargo.lock` du `.gitignore` : cette ligne n'a jamais existé.

**Tous les pièges de la passation précédente restent valables** — voir
[`passation-2026-08-04.md`](passation-2026-08-04.md) §5 (format `LVQ2`/`LVQ3`,
estimateur de coût inapplicable hors quantification, cartes sm < 89, flux de
métriques HF, `LLVQ_DTYPE`, moyenne de VRAM).

## 6. Ce qu'il faut faire — la priorité

**Le lot K4 : le matvec fusé tuilé sur CUDA.** C'est la contribution, et c'est
tout ce qui manque à un chiffre de vitesse.

Ce qui est déjà réglé et n'est plus un risque : la plomberie cudarc, la
compilation NVRTC, la justesse du décodeur, la garde de carte, le lanceur
générique, la boucle d'itération à 0,01 $.

Ce qui reste, dans l'ordre :

1. Porter `tv_slot` — la forme **tuilée** de `bin/thesis`, pas celle de
   `bin/matvec` (qui étage la ligne entière et détruirait l'occupancy). Deux
   `__syncthreads()`, pas un ; `simd_sum` devient cinq `__shfl_xor_sync` ;
   **pas** de `if (row >= d_out) return;` — un `return` avant une barrière est
   un interblocage, et il casserait le masque `0xffffffff`.
2. Porter `tv_f16` avec des chargements 128 bits **des deux côtés** (K−1(b) :
   la largeur de chargement paie, et corriger un seul bras truquerait le
   rapport).
3. Monter le fichier scellé depuis le Hub (`run.py` sait le faire) et vérifier
   les 1 105 920 lignes contre la référence f64 **avant** toute mesure.
4. Ajouter cuBLAS `Gemm<half::f16>` à n = 1 comme troisième bras — c'est le
   seul gain méthodologique gratuit du portage, il ferme l'angle « baseline
   maison ».
5. Entrelacer les bras et publier la dispersion (cf. §3).

**Coût attendu** : le prologue CPU mono-thread est le poste dominant sur
8 vCPU — paralléliser la boucle de matrices (`thread::scope`) avant de payer.
Compter 0,30-0,50 $ par run complet.

## 7. Ce qu'il ne faut PAS faire

- **Ne pas porter `Grouped32` ni `Flat32` sur CUDA.** K−1(a) a tranché : ils
  perdent sur les deux axes à la fois.
- **Ne pas budgéter le padding anti-conflit de bancs comme un gain acquis.**
- **Ne pas publier un rapport CUDA sans sa dispersion**, ni comparer un chiffre
  CUDA à un chiffre Metal sous ~5 % d'écart.
- **Ne jamais comparer `y_cuda` à `y_metal`** : chacun contre la référence f64
  partagée. Deux pires erreurs publiées, pas un delta.
- Tout ce que dit [`passation-2026-08-04.md`](passation-2026-08-04.md) §6 reste
  vrai : pas de campagne de mesure sans redemande, pas de bras adverse en plus,
  pas de job HF sans autorisation, rien sur le 8B, jamais « 2 bits par poids »
  tout court.

## 8. Deux décisions en attente

**La carte du modèle publiée sur Hugging Face porte encore `2,06–2,08×.**
Le fichier local [`hf-model-card.md`](hf-model-card.md) est corrigé en
`2,03–2,09× sur trois invocations`, avec la sémantique du rapport et le renvoi
au journal. Republier est une action sur une surface publique.

**La branche `campagne-mesure` n'est toujours pas fusionnée**, et
`noyau-cuda` est maintenant empilée dessus.

## 9. Commandes

```bash
# la porte, sur les deux cibles
cargo clippy --all-targets && cargo test --release -- --include-ignored
CUDARC_CUDA_VERSION=12040 cargo clippy -p llvq-cuda --target x86_64-unknown-linux-gnu --all-targets

# le décodeur CUDA, testé sans carte
cargo test --release -p llvq-cuda -- --include-ignored

# le banc à sept bras (Metal)
cargo run --release -p llvq-metal --bin thesis -- ~/llvq-q4b.llvq

# la comptabilité Slot32 et le plafond L ≤ 4
cargo run --release -p llvq-bench --bin rtbits -- ~/llvq-q4b.llvq

# reconstruire l'image (seulement si le binaire Rust change)
uv run ops/run.py publish Pier-Jean/llvq-runner-cuda --cuda

# un mini-job — la commande passe sous `set -euo pipefail`, la carte est gardée
uv run ops/run.py bench --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l40sx1 --timeout 15m --name llvq-preflight 'preflight'
uv run ops/run.py monitor <job_id> --flavor l40sx1
```
