# Étude — Mémoire extrême sur les MoE open-weights (2026-08-12)

> Étude sur papier : **rien n'a été mesuré sur un MoE**, tout ce qui suit est
> de l'arithmétique sur des layouts mesurés (dense, 4B/8B) projetée sur des
> fiches de modèles vérifiées le jour même sur le web. Étiquettes : ᵐ mesuré
> (chez nous, dense) · ᵖ prédit (compte de bits exact, conversion vérifiée) ·
> ᵉ estimé · ᵂ fiche web du 2026-08-12 (sources en fin de fichier).
>
> **Le résultat central de l'étude n'est pas un chiffre, c'est un
> renversement de critère** : sur un MoE, le trafic par token est proportionnel
> aux paramètres *actifs* pendant que la VRAM est payée sur les paramètres
> *totaux*. Le ratio actifs/totaux des MoE 2026 est de 2 à 5 % — donc le
> critère de vitesse qui structurait l'échelle dense (1,6×, celui qui a écarté
> `Golay70`) **ne s'applique plus à cette cible**, et l'échelle se relit
> capacité d'abord. `Golay70` — 3,589 b/poids ᵐ, reconstruction exacte prouvée
> sur 150,7 M blocs — redevient le meilleur point **mesuré** du projet.

## 1. Le paysage MoE open-weights (août 2026)

| modèle | totaux | actifs | ratio | notes |
|---|---|---|---|---|
| gpt-oss-20B ᵂ | 21 Md | 3,6 Md | 17 % | MXFP4 natif |
| Qwen3-30B-A3B ᵂ | 30,5 Md | 3,3 Md | 11 % | le point de gate pas cher (§7) |
| Qwen3-Next-80B-A3B ᵂ | 80 Md | 3 Md | 3,8 % | 512 experts, 10+1 actifs, hidden 2048 |
| gpt-oss-120B ᵂ | 117 Md | 5,1 Md | 4,4 % | MXFP4 natif, ~60,8 Go livrés |
| DeepSeek-V4-Flash ᵂ | 284 Md | 13 Md | 4,6 % | avril 2026, ctx 1M, MIT |
| MiniMax M3 ᵂ | ~428 Md | ~23 Md | 5,4 % | fiche la moins recoupée de la liste |
| **GLM-5.2** ᵂ | **744 Md** | **40 Md** | 5,4 % | **n°1 open-weights** (juin 2026), 256 experts 8+1, MLA, MIT |
| Kimi K2.6 ᵂ | ~1 040 Md | 32 Md | 3,1 % | **INT4 natif**, 384 experts 8+1, MLA |
| DeepSeek-V4-Pro ᵂ | 1 600 Md | 49 Md | 3,1 % | le plus gros open-weight existant |

Deux faits de structure, décisifs pour nous :

1. **L'embedding devient un epsilon.** 0,3–2 % des poids sur tout ce qui
   dépasse 80 Md (vocab ~150-200k × hidden, contre des centaines de milliards
   d'experts). La pénalité qui coûtait 26 % au 0.6B et 9,7 % au 4B — et qui a
   imposé le q8 — disparaît : **b/param modèle entier ≈ b/poids thesis**, la
   comptabilité honnête devient gratuite.
2. **Les poids au repos dominent tout.** 95-97 % des poids d'un token donné ne
   sont pas lus. La VRAM est payée sur le total, le trafic sur les actifs :
   c'est la définition même d'un problème *capacity-bound* — notre problème.

## 2. Le renversement de critère, chiffré

Sur le dense, écarter `Golay70` à 1,31× était correct : le trafic d'un token,
c'est le modèle entier, et 195 Go/s effectifs contre 425 se paient sur chaque
milliseconde. Sur un MoE, le trafic d'un token c'est **actifs × b/8** :

| actifs | Go lus/token à `Golay70` (3,589) | plafond à 195 Go/s ᵐ | à `Planes14` (425 Go/s ᵐ) |
|---|---|---|---|
| 3 Md (Qwen3-Next) | 1,35 | **144 tok/s** | 306 tok/s |
| 13 Md (V4-Flash) | 5,8 | 33 | 71 |
| 32 Md (K2.6) | 14,4 | 13,5 | 22 |
| 40 Md (GLM-5.2) | 17,9 | 10,9 | 18 |
| 49 Md (V4-Pro) | 22,0 | 8,9 | 15 |

⚠️ Plafonds de bande passante, pas des débits prédits — mais la *comparaison*
est licite : partout, même le layout « lent » reste au-dessus de la vitesse de
lecture humaine. **Le coût du décodage cesse d'être discriminant ; les bits
restent seuls en piste.** L'échelle MoE se lit donc : `Golay70` 3,589 ᵐ >
`X2` 3,77 ᵖ (qui ne le bat *pas* en bits — il ne le bat qu'en vitesse, l'axe
qui vient de perdre son poids) > `E3` ~2,4 ᵉ. Conséquence pour le lot X :
sur cible MoE, X2 perd son rôle de finaliste au profit de `Golay70` déjà
mesuré, et **E3 devient l'enjeu unique**.

## 3. La projection — poids sur la carte, en Go

`Go = totaux × b/8`, embedding q8 intégré (correction visible seulement sous
80 Md). Repère q4 : 4,6 b/param ᵉ (GGUF Q4_K_M typique), ou le natif du
modèle quand il existe.

| modèle | q4 / natif | `Planes14` ᵐ | `Golay70` ᵐ | `X2` ᵖ | `E3` ᵉ |
|---|---|---|---|---|---|
| gpt-oss-20B | 13,8 natif | 13,1 | 10,1 | 10,5 | 7,1 |
| Qwen3-30B-A3B | 17,5 | 18,6 | 14,1 | 14,7 | 9,6 |
| Qwen3-Next-80B | 46,0 | 48,5 | **36,3** | 38,0 | **24,4** |
| gpt-oss-120B | 60,8 natif | 71,0 | **53,3** | 55,8 | **35,9** |
| DeepSeek-V4-Flash | 163 | 172 | 129 | 135 | **86,6** |
| MiniMax M3 | 246 | 258 | 193 | 202 | 128 |
| GLM-5.2 | 428 | 448 | **335** | 351 | **223** |
| Kimi K2.6 | ~560 natif INT4 | 627 | **468** | 490 | **312** |
| DeepSeek-V4-Pro | 920 | 964 | 720 | 754 | **480** |

### Les classes de machine que ça ouvre

Budgets réalistes : GPU = sa VRAM moins ~2-3 Go (KV compressé MLA/CSA +
activations) ; mémoire unifiée Mac = ~75 % allocables par défaut, ~90 % en
relevant la limite wired (pratique courante, à étiqueter ⚠️ quand on en
dépend).

| verdict | q4 / natif | `Golay70` ᵐ | `E3` ᵉ |
|---|---|---|---|
| Qwen3-Next-80B | 48 Go, juste | 48 Go, à l'aise | **RTX 5090 32 Go** |
| gpt-oss-120B | déborde le 64 Go unifié | 64 Go unifié ⚠️ relevé | **une carte 48 Go** |
| V4-Flash 284B | 192 Go unifié | H200 141 / Mac 192 | **RTX PRO 6000 96 Go** |
| GLM-5.2 744B | Mac 512 ⚠️ relevé | **Mac 512, alloc défaut** | **Mac 256/384** |
| Kimi K2.6 1T | ne tient sur rien de solo | Mac 512 ⚠️ relevé, limite | **un seul Mac 384/512** |
| V4-Pro 1,6T | deux machines min. | 2×512 | 512 ⚠️ relevé, très limite |

Les trois lignes qui font le silo :

- **GLM-5.2 — le meilleur modèle open du monde — sur un Mac Studio 512 en
  allocation par défaut avec un layout déjà mesuré** (`Golay70`, 335 Go), là
  où le q4 exige de relever la limite et sature la machine.
- **Kimi K2.6 — le trillion — sur un seul poste de travail en E3** (312 Go
  sur Mac 384/512), alors que son INT4 natif (~560 Go) ne tient nulle part en
  solo.
- **DeepSeek-V4-Pro — le plus gros open existant — à portée d'une seule
  machine en E3** (480 Go), deux machines aujourd'hui quoi qu'on fasse.

## 4. Les murs nouveaux — ce que le MoE coûte, honnêtement

**(a) L'encodeur se paie sur les totaux.** 6,36·10⁻⁵ cœur-s/poids ᵐ (32B,
surlinéaire, dernière extrapolation 25 % basse — marge ×1,25 incluse) :

| modèle | cœur-h | $ (CPU spot 0,04-0,08 $/cœur-h ᵉ) |
|---|---|---|
| Qwen3-30B-A3B | 540-680 | **25-55 $** |
| Qwen3-Next-80B | 1 400-1 800 | 60-140 $ |
| V4-Flash | 5 000-6 300 | 200-500 $ |
| GLM-5.2 | 13 100-16 400 | 550-1 300 $ |
| K2.6 | 18 400-23 000 | 750-1 800 $ |
| V4-Pro | 28 300-35 300 | 1 100-2 800 $ |

Parallélisme parfait par expert (matrices indépendantes) — c'est une flotte
CPU, pas un GPU. L'ordre de grandeur reste accessible jusqu'au 1T.

**(b) La capture ne tient plus en RAM — prérequis C6.** Le pipeline actuel
garde le modèle bf16 résident : K2.6 = **2,1 To**. Il faut une capture en
streaming par couche depuis le disque (architecturalement compatible : la
passe séquentielle traite déjà bloc par bloc) ou une capture shardée
multi-nœud. Même famille de prérequis que C3 (bf16), qui avait économisé
130 $ — sauf qu'ici c'est bloquant, pas une optimisation.

**(c) La couverture hessienne par expert — le piège silencieux.** Un expert
ne voit que `T × actifs/experts` tokens : sur K2.6 (8 routés / 384), nos
131 k tokens de calibration donnent **~2 700 tokens par expert pour une
hessienne de dimension 7 168 — singulière**. Il faut 2-5 M de tokens ᵉ, un
recensement des tokens routés par expert imprimé dans le log (test létal :
un expert sous-couvert doit faire échouer le run, pas dégrader en silence),
et une politique explicite pour les experts quasi morts (fallback identité +
amortissement, consigné).

**(d) Les natifs INT4/MXFP4 n'ont pas de f16 de référence.** gpt-oss et K2.6
sont *entraînés* quantifiés : notre chaîne partirait d'un poids déjà
quantifié, et le contrôle identité ne mesure plus la même chose. Question
ouverte, à trancher sur le petit gpt-oss-20B avant de toucher aux gros.

## 5. La question scientifique — et le gate à 50 $ qui l'ouvre

**Le déficit du 2 bits suit-il les paramètres totaux ou les actifs ?** Notre
loi d'échelle mesurée (−14,73 pp → −10,56 pp de MMLU du 4B au 8B) est dense.
Si le déficit suit les *totaux*, un 1T à 2 bits serait quasi indemne — le
silo est immense. S'il suit les *actifs*, K2.6 se comporterait comme un
~32B — encore exploitable, mais tout autre. Personne n'a publié cette courbe.

**Gate X5-MoE : Qwen3-30B-A3B, ~25-55 $.** L'expérience naturelle parfaite
face au Qwen3-32B dense déjà au programme : totaux comparables (30,5 vs
32,8 Md), actifs sans commune mesure (3,3 vs 32,8 Md), même famille, même
harnais, mêmes empreintes de tokens. Une variable — exactement le protocole
du dépôt. C'est le run MoE à faire **en premier**, avant un centime sur un
modèle à trois chiffres.

## 6. Face à ce qui se fait

Le terrain existe déjà : les GGUF « dynamic » d'Unsloth servent GLM, K2 et
DeepSeek à 1,58-2,5 bpw depuis 2025 — c'est *le* concurrent de fait de la
mémoire extrême MoE, avec une qualité notoirement rugueuse aux débits bas et
une pondération hétérogène (attention gardée haute). Notre angle reste le
même qu'en dense : un code uniforme au meilleur réseau connu en dimension 24,
une reconstruction exacte prouvable, et — si la tendance qualité tient — le
meilleur rapport qualité/Go à l'extrémité de la courbe. La fenêtre est
ouverte mais elle n'attendra pas : K2.6 natif INT4 montre que les labos
intègrent la quantification à l'entraînement, et chaque génération native
abaisse la valeur d'une PTQ externe.

## 7. Conséquences sur la spec du lot X

1. **X3 (banc)** : inchangé pour la cible dense. Pour la cible MoE, le
   critère 1,6× ne s'applique pas — l'admission est *capacity-first* et
   `Golay70` est déjà admis de fait (mesuré, exact, 3,589).
2. **X2 (E1c-12)** perd son rôle de finaliste MoE (3,77 > 3,589) ; il reste
   le candidat *dense* 70B/40 Go et un banc utile.
3. **X4 (E3)** monte d'un cran : c'est la seule marche restante entre 3,589
   et le plancher, et c'est elle qui fait « le 1T sur un poste ». Son étude
   papier hérite d'un argument neuf : sur MoE, un décodeur E3 même 3× plus
   lent que `Planes14` resterait au-dessus de 10 tok/s sur K2.6 (§2).
4. **Nouveaux prérequis** avant tout run MoE : C6 (capture streaming),
   couverture hessienne par expert (§4c), politique experts morts, verdict
   INT4-natif sur gpt-oss-20B.
5. **Ordre de dépense** : X5-MoE (~50 $) → verdict → V4-Flash 284B
   (~200-500 $ d'encodeur, tient en E3 sur une seule RTX PRO 6000) comme
   première cible produit → GLM-5.2/K2.6 seulement si les deux gates sont
   verts.

## Ce que cette étude ne dit pas

La qualité 2 bits sur un MoE : **aucun chiffre, nulle part, y compris chez
nous** — c'est l'objet du gate X5-MoE. Les débits sont des plafonds de bande
passante, pas des mesures. Les fiches MiniMax M3 et le détail INT4 de K2.6
sont les moins recoupées. Et tout coût est un ordre de grandeur : la seule
constante du dépôt est que les extrapolations d'encodeur sont basses.

## Sources web (consultées le 2026-08-12)

- Kimi K2/K2.5/K2.6 : rapport technique arXiv 2507.20534, artificialanalysis.ai,
  intuitionlabs.ai, docs.api.nvidia.com — 1,04 T / 32 Md, 384 experts 8+1,
  hidden 7168, MLA, vocab 160k ; K2.6 avril 2026, INT4 natif.
- Qwen3-Next-80B-A3B : huggingface.co/Qwen, docs AWS Bedrock — 80/3 Md,
  512 experts 10+1, hidden 2048, 48 couches.
- DeepSeek-V4 : huggingface.co/deepseek-ai, arXiv 2606.19348 — V4-Pro
  1,6 T / 49 Md, V4-Flash 284/13 Md, ctx 1M, MIT, avril 2026.
- GLM-5.2 : artificialanalysis.ai, unsloth.ai/docs, huggingface.co/zai-org —
  744/40 Md, 256 experts 8+1, MLA, 78 blocs, MIT, juin 2026, n°1 de
  l'Intelligence Index open-weights.
- gpt-oss-20B/120B, Qwen3-30B-A3B, MiniMax M3 : fiches HF et synthèses —
  recoupement plus faible pour M3, étiqueté tel quel.
