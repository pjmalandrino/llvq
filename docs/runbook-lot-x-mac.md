# Runbook — le lot X sur le Mac, avant d'engager une carte (2026-08-12)

> Ce que la session du 2026-08-12 a livré en code, et l'ordre dans lequel le
> lancer. **Tout tient sur le Mac de dev, 0 $, un seul téléchargement
> optionnel.** Chaque étape tranche quelque chose, et un rouge arrête la
> dépense correspondante — c'est le rôle que le dossier donne à ses gates.
>
> Spec : [`spec-memoire-extreme-2026-08-12.md`](spec-memoire-extreme-2026-08-12.md).
> Étude MoE : [`etude-moe-memoire-extreme-2026-08-12.md`](etude-moe-memoire-extreme-2026-08-12.md).

## Ce qui est écrit, et ce qui ne l'est pas

| item de la spec | livré | où |
|---|---|---|
| **X0** spec de layout | ✅ dans le code, en tête de module | `llvq-artifact/src/e1c.rs` |
| **X1/X2** transcodeur + référence + tests | ✅ repack + inverse + 9 tests unitaires + 3 tests d'intégration | `src/e1c.rs`, `tests/e1c_format.rs` |
| **item 1** bras `rtbits` | ✅ E1c14/E1c12 dans les trois tables + 3 tests d'acceptation | `llvq-bench/src/bin/rtbits.rs` |
| **X4** étude E3 | ✅ bin d'étude, menu de 6 variantes, verdict automatique | `llvq-bench/src/bin/radixstudy.rs` |
| **X3** bras de banc CUDA | ❌ **pas écrit** — c'est la partie carte, et elle attend les verdicts ci-dessous | — |
| noyau CUDA `e1c.cu` | ❌ pas écrit, même raison | — |

⚠️ **Rien de ce qui suit ne mesure une vitesse.** Le code livré compte des
bits et prouve une bijection. La question « le transposé va-t-il aussi vite »
reste entière et ne se tranche que sur carte (critères X3 : ≥ 1,9× pour
remplacer `Planes12x`, ≥ 2,05× pour remplacer `Planes14`).

## L'ordre

### 1. La boucle rapide (30 s)

```bash
cargo test -p llvq-artifact -p llvq-bench     # doit être vert
cargo clippy --all-targets                    # zéro warning (cudarc échoue hors Linux, normal)
```

### 2. L'étude E3 — le verdict le plus cher à obtenir autrement (2 s)

```bash
cargo run --release -p llvq-bench --bin radixstudy                      # classes à poids égal
cargo run --release -p llvq-bench --bin radixstudy ~/llvq-q4b.llvq      # blocs réels ← celui qui compte
```

Le bin imprime un `VERDICT X4` explicite. **Sur les classes à poids égal il
sort déjà rouge** — la meilleure décomposition shift-only vaut 2,73 b/poids
noyau contre un critère de 2,60. Le run pondéré par les blocs réels est celui
qui fait foi : c'est la même arithmétique sur la vraie distribution.

- 🔴 rouge → **E3 est enterré sur papier**, pour 0 $, comme E2 l'a été au banc.
  Le plafond mémoire du projet devient E1c, et la ligne « K2.6 sur un poste »
  de l'étude MoE tombe avec lui. Le consigner comme un verdict, pas comme un
  échec.
- 🟢 vert → le chantier E3 s'ouvre **au sens des bits seuls**. Rappel que
  `Golay70` a été juste, compact et mort en ALU : un vert ici n'est pas une
  promesse de vitesse.

### 3. Le compte de bits E1c sur le fichier réel (~10 s)

```bash
cargo run --release -p llvq-bench --bin rtbits ~/llvq-q4b.llvq
```

Attendu, épinglé par `e1c_rates_on_the_published_4b_match_the_spec` :

| layout | b/poids payload | b/poids noyau |
|---|---|---|
| `Planes14` | 4,6667 | 4,8040 |
| **`E1c14`** | **4,4167** | **4,5551** |
| `Planes12x` | 4,2029 | 4,3424 |
| **`E1c12`** | **3,6196** | **3,7618** |

Si la sortie diffère, **ne rien publier** : le test d'acceptation et le bin
divergeraient, et l'un des deux serait faux.

### 4. Le sweep intégral — la preuve d'exactitude (des minutes)

```bash
cargo test --release -p llvq-artifact --test e1c_format -- --include-ignored
```

150 681 600 blocs, les deux variantes contre le décodeur d'archive, plus le
flux principal `E1c12` contre celui de `Planes12x`. ⚠️ Long, et il **échoue
franchement** si `~/llvq-q4b.llvq` n'est pas là (jamais de `SKIP` vert).

Vert ici = le format est prouvé exact avant la carte, donc le bras de banc de
la semaine suivante est un pur test de vitesse à ~0,2 $.

### 5. L'histogramme de routage MoE — le risque caché du run MoE (~1 h)

Le seul item qui télécharge quelque chose (**gpt-oss-20B, ~14 Go**), et il
n'utilise pas notre pipeline : n'importe quel runtime donne les décisions du
routeur.

Compter les tokens routés **par expert** sur ~131 k tokens de C4. La question :
la distribution est-elle assez plate pour que chaque expert voie de quoi
former une hessienne non singulière ? Sur K2.6 (8 routés / 384), 131 k tokens
donneraient ~2 700 tokens par expert pour une hessienne de dimension 7 168 —
singulière. Si gpt-oss confirme une distribution très inégale, le volume de
calibration MoE explose et le devis du gate X5-MoE avec lui.

Bonus gratuit : premier contact avec un poids MXFP4 natif, c'est-à-dire la
question §4d de l'étude — que devient le contrôle identité quand le modèle
source est *déjà* quantifié ?

### 6. Le devis, avant d'annoncer quoi que ce soit (~10 min)

```bash
uv run ops/run.py selftest                              # l'estimateur contre le run 4B réel
uv run ops/run.py estimate Qwen/Qwen3-30B-A3B --dtype bf16
```

L'estimateur a déjà été **25 % bas** une fois. Le recaler avant d'annoncer les
~25-55 $ du gate X5-MoE.

## Ce qu'on saura à la fin, et ce que ça débloque

| verdict | ce qui s'ouvre | ce qui se ferme |
|---|---|---|
| X4 vert + sweep vert | banc X3 à 0,2 $, puis chantier E3 | — |
| X4 rouge + sweep vert | banc X3 à 0,2 $ seulement | E3, et « le 1T sur un poste » |
| sweep rouge | rien | le transposé, jusqu'à correction |
| routage MoE plat | gate X5-MoE à ~50 $ | — |
| routage MoE très inégal | — | le devis MoE tel qu'annoncé, à refaire |

**À ne pas faire cette semaine** : le portage Metal du noyau E1c (les verdicts
de bancs Apple ne se transportent pas sur NVIDIA — documenté deux fois) · tout
téléchargement d'un modèle à trois chiffres · toute conclusion de vitesse sans
carte (un compte niveau source a déjà été faux d'un facteur 2 sur ce noyau).
