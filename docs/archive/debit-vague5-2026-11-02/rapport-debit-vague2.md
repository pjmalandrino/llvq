# Débit LLVQ : programme de la vague 2

Le premier levier, un noyau persistant qui exécute chaque jeton en un seul lancement, porterait le 4B scellé de 133 à 340-425 tok/s sur L40S (*estimé*). Les trois leviers composés, avec une spéculation exacte, le porteraient à 345-1 075 tok/s en chat et 510-1 590 sur du code (*estimé*). Ces chiffres demandent huit décisions de l'opérateur, toutes fermées par défaut (§11, décisions 3 à 10), et reposent sur quatre paris non mesurés (§1). Sans ces décisions, avec un plafond de dépense et un go par job, le palier atteignable sur carte est la route D, qui fusionne des lancements sans noyau persistant : 155-182 tok/s (*estimé*). Au préremplissage, la route GEMM déquantifie chaque matrice en f16 pour cuBLAS. Le premier jeton d'un prompt de 8 192 jetons arriverait alors après 3,5-4,4 s au lieu de 35,0 s, et vers 1,7-1,9 s avec en plus un noyau d'attention sur cœurs tenseurs (*estimé*). La route GEMM demande un T8 du préremplissage et la lecture du gel ROADMAP §4 ; le noyau d'attention, un T8 de plus.

- Rapport du 2026-11-02, écrit en lecture seule. Aucun fichier du dépôt n'a changé, aucun run n'a tourné, 0 $ dépensé.
- Objet par défaut : Qwen3-4B scellé, L40S, lot 1, décodage glouton, forme groupée servie.
- Étiquettes, à la première apparition d'un nombre : *mesuré* (journal nommé) ; *calculé* (arithmétique sur des entrées mesurées ou lues dans le code) ; *estimé* (jugement, ou arithmétique sur au moins une entrée estimée, même reproduite par un vérificateur).
- Aucun × mesuré de notre fait n'est divisé entre cartes ou moteurs, et les autres moteurs sont comparés en Go/s. Une efficacité vient pourtant d'ailleurs : η, la part des 864 Go/s que l'attention atteint en lisant le cache KV, est prise à 0,40 d'après FlashInfer sur A100 et H100. Cette valeur porte les plafonds à 8k de notre noyau d'attention et de 4E (§8) ; ceux des routes P et D gardent la croissance mesurée de l'attention d'aujourd'hui.
- La méthode est en annexe A.

Vocabulaire :
- **T8** : servir un changement numérique sous la tolérance de ±0,12 % de perplexité appariée, avec un contrôle. L'opérateur l'a accordée le 2026-11-01 pour la forme groupée au 4B seulement ; ce n'est pas une politique. Chaque nouveau changement demande sa propre décision, sur préreg tamponné.
- **A2** : la capture de graphes CUDA de la boucle de jetons, fermée et hors du cœur (règle 3).
- **Rangée** : une ligne d'activation, donc un jeton traité dans la même passe. Une **ligne** reste une ligne de matrice de poids.
- **t(k)** : temps d'une passe qui traite k rangées. Une vérification est **plate** quand t(k) reste proche de t(1).
- **τ** : jetons acceptés par vérification, jeton bonus compris.
- **η** : part des 864 Go/s de la carte que l'attention atteint en lisant le cache KV. À 8k, ce terme vaut 1,208 Go / (η × 864 Go/s) : 3,50 ms à η = 0,40, 2,33 ms à η = 0,60 (*calculé*).
- **Route P** : retirer les lancements par un noyau persistant par segment de couche. **Route D** : le repli, qui fusionne et retire des lancements dans des grilles normales. **4E** : un seul noyau persistant qui boucle sur N jetons. **La brique**, ou **4K** : le noyau « décoder une fois » du §6.
- **Route GEMM** : au préremplissage, chaque matrice est déquantifiée dans un tampon f16, puis cuBLAS fait le produit (§8).

## 1. Le verdict

Aujourd'hui, au 4B (*mesuré*, médianes et plages de cinq tours, `served-grouped-2026-11-02.txt`) :
- 133,3 tok/s [133,2-133,3] à la porte servie, prompt de 5 jetons, préremplissage inclus ;
- 77,5 tok/s [77,5-77,6] de décodage à 8 192 positions ;
- 35,02 s [34,95-35,06] avant le premier jeton d'un prompt de 8 192 jetons.

Le pas de décodage dure 7,44 ms (*calculé* : 1 000 / 134,4 tok/s *mesurés*, `gqa-ab-2026-11-01.txt`). Lire ses 1,377 Go de poids à 864 Go/s prendrait 1,59 ms (*calculé*). Les 79 % restants (*calculé*) ne sont pas du flux de poids à la crête. Ils mêlent un flux plus lent que la crête, 0,4 à 1,0 ms (*estimé*, §2), et un coût fixe. Ce coût fixe vient de 929 lancements par jeton (*calculé* sur le code), d'un plancher de géométrie dans chaque projection, de 96 rotations qui n'occupent chacune qu'un SM, et d'un aller-retour hôte par jeton. Ce partage reste une inférence : aucune trace des noyaux n'a été relevée.

Les trois leviers :

| levier | idée | ce qu'il donnerait au 4B sur L40S | exactitude |
|---|---|---|---|
| un jeton, un lancement | un noyau persistant exécute le jeton entier | 340-425 tok/s court, 156-216 à 8k (*estimé*) | projections bit pour bit visées ; attention sous T8 |
| décoder une fois | chaque mot Tetra, décodé une seule fois, sert k rangées | vérifier 8 rangées coûterait 1,14-1,29 pas (*estimé*, pari) | entiers exacts ; un T8 pour l'activation int16 |
| spéculation exacte | des drafters publiés proposent, le modèle vérifie | ×1,5-3,7 sur du code, ×1,0-2,5 en chat, une fois la vérification plate (*estimé*) | sortie identique au décodage simple |

Les paliers, avec les décisions qu'ils demandent :

| palier | décisions préalables | décodage court (tok/s) | décodage à 8k (tok/s) | requête 8 192 + 512 |
|---|---|---|---|---|
| aujourd'hui | aucune | 133,3 | 77,5 | 41,7 s |
| route GEMM du préremplissage | T8 du préremplissage ; gel ROADMAP §4 | inchangé | inchangé | 10-11 s |
| route D | remplacer des opérations candle | 155-182 | 84-91 | non calculé |
| route P | réglages gelés ; remplacer des opérations candle ; lecture de A2 si elle couvre P | 250-300 | 105-113 | 8,0-9,3 s avec la route GEMM |
| route P et notre noyau d'attention | T8 de l'attention | 290-335 | 144-188 | non calculé |
| 4E | lecture de A2 ; choix du grand modèle | 340-425 | 156-216 | 4,0-5,2 s avec la route GEMM et une attention de préremplissage sur cœurs tenseurs (deux T8, gel ROADMAP §4) |
| spéculation, vérification plate | brique et T8 int16 ; verdict spéculatif ; règle 6 | 345-1 075 en chat, 510-1 590 sur du code | non calculé | non calculé |

Tout ce tableau est *estimé*, sauf la première ligne : 133,3 et 77,5 *mesurés*, 41,7 s *calculé*. La requête notée 8 192 + 512 désigne un prompt de 8 192 jetons suivi de 512 jetons générés ; aucune requête de cette forme n'a été chronométrée. Son temps d'aujourd'hui additionne les 35,02 s mesurées et 512 pas sur la droite 7,43 ms + 0,668 µs par position, ajustée sur les médianes de `served-grouped-2026-11-02.txt` et ancrée à la position moyenne 133,5 (`wave3-report/requests.py`). Les routes D et P sont deux variantes de l'étage 2. À partir de la route P, les paliers de décodage se cumulent de haut en bas, et la route GEMM s'y ajoute à part. Chaque palier payant demande aussi un plafond de dépense et un go.

Sur des requêtes courtes, au palier 4E, une réponse de 256 jetons à un prompt de 5 jetons passerait de 1,9 s (*calculé*) à 0,6-0,8 s (*estimé*). Une réponse raisonnée de 2 136 jetons à un prompt de 200 jetons passerait de 18,3 s (*calculé*) à 5,7-7,4 s (*estimé*). Le mode raisonnement demande un échantillonnage, que le chemin servi n'a pas.

Les trois leviers vont ensemble. Sans spéculation, le 4B plafonne à 627 tok/s au lot 1 (*calculé* : 864 Go/s ÷ 1,377 Go). Le palier 4E en atteindrait 54 à 68 %, la route P 40 à 48 %, la route D 25 à 29 % (*estimé*). Dépasser ce plafond demande de vérifier plusieurs jetons par lecture des poids. Il faut pour cela une vérification dont le coût ne monte pas avec k, le deuxième levier, et des brouillons souvent acceptés, le troisième.

Les fourchettes composées portent trois paris techniques : le débit d'un flux Tetra persistant sur sm_89, une vérification plate en k, et τ contre l'objet scellé. Un quatrième pari reste implicite : que la brique tienne dans le noyau persistant, où trois rangées SIMT débordent déjà la mémoire partagée (§9). Elles demandent aussi huit décisions de l'opérateur, fermées par défaut. Pour 340-425 tok/s : le dégel des réglages des CTA persistants (décision 8), le T8 de notre noyau d'attention (décision 3), la lecture de A2 (décision 4) et le choix du grand modèle (décision 10). La spéculation ajoute le chemin multi-rangées, retenu par le gel ROADMAP §4 (décision 9), les cœurs tenseurs (décision 7) avec le T8 de l'activation int16, la levée du verdict spéculatif (décision 5) et la règle 6 (décision 6). Le τ retenu est celui de DSpark. Il ne tient dans la marge du triplet qu'avec un KV q8, que le moteur ne stocke pas, et avec notre tête int4 partagée, un changement jamais entraîné ; sans ce partage, même la marge q8 ne suffit pas (§7).

Le préremplissage offre le plus gros gain par requête, mais la route GEMM seule ne suffit pas.
- Sur une requête de 8 192 + 512 jetons, le préremplissage prend 84 % du temps (*calculé* : 35,0 s sur 41,7).
- À 8k, sa part superlinéaire vaut 1,9 à 2,9 s (*estimé* : modèle carré ajusté sur 8,20, 17,03 et 35,02 s mesurés à 2 048, 4 096 et 8 192 jetons, `served-grouped-2026-11-02.txt`). On l'attribue à l'attention, par inférence.
- Cette attention garde trois tenseurs de scores f16 vivants : 6,44 Go au 4B à 8k, au-delà de la marge de 5 Go du triplet (*calculé* : 3 × 4 096 × 8 192 × 32 têtes × 2 octets ; audit ML du 2026-11-02, non versé au dépôt).
- La route GEMM seule ramènerait donc le premier jeton à 3,5-4,4 s (*estimé*). L'audit du 2026-10-29 annonçait 2 s.
- Avec en plus un noyau d'attention de préremplissage sur cœurs tenseurs, sous T8, le délai du premier jeton tomberait vers 1,7-1,9 s (*estimé*, §8).

Sur la L40S, le 4B dépasse déjà de loin les planchers de vitesse A5 du produit, définis pour la 5090 : au moins 20 tok/s en chat, et au plus 60 s par document d'extraction en lot de 8. Il fait 133,3 tok/s, et 77,5 à 8k, contre 20 en chat. Il traite un document de 8 192 + 512 jetons en 41,7 s, sous les 60 s si le plancher s'amortit sur le lot. Ce n'est qu'un ordre de grandeur, puisque la carte diffère. Les planchers mordent au 14B et au-delà : environ 99 à 106 s par document, et environ 21 tok/s à 8k en forme étendue (*estimé*). Au 14B, la forme groupée et la route GEMM bougent donc un plancher ; le gain de décodage au 4B n'en bouge aucun.

La branche `debit-trims` a livré l'étape 0 bis de l'audit du 2026-10-29 : trois des quatre nettoyages, ×1,0716 (*mesuré*, `trims-contexts-2026-10-30.txt`). Elle a aussi livré la première moitié de son étape 1, l'attention groupée sur `Grow` : ×1,10 [1,102-1,103] à 5 jetons et ×2,34 [2,338-2,341] à 8 192 (*mesuré*, rapports formés tour par tour, `gqa-ab-2026-11-01.txt`). C'est la base de ce rapport. Les autres étapes ne sont faites dans aucune branche visible de cette copie du dépôt, et la passation du 2026-11-01 les range parmi les « suites possibles, non décidées ». L'opérateur dit les réaliser sur une autre branche ; ce rapport ne l'a pas lue. Elles recoupent ce programme ainsi, et les durées et coûts du §10 les comptent déjà :
- étape 0, le profil nsys : impossible telle quelle, l'image n'a pas nsys ; l'ajouter est un changement de recette (décision 13) ;
- fin de l'étape 1, le noyau d'attention de décodage et le KV int8 réel : étage 3 et option 3b du §10, sous T8 ; un seul noyau, invariant en rangées, servirait aussi la vérification ;
- étapes 2 et 3, les fusions et `tv_tetra48_seg` : la route D de l'étage 2 ;
- étape 4, le mur int4 Metal : le poste 0.3 ;
- étape 5, la route GEMM : le poste 1c ;
- étape 6, multi-rangées puis spéculation : le chemin 1a, la brique 4K et l'étage 5 ;
- étape 7, les graphes : fermée par la règle 3, et ce programme ne la reprend pas ; 4E la remplace par un noyau persistant, si l'opérateur le lit hors de A2 (décision 4).

Pour la branche qui porte la suite de l'audit, sous les décisions du §11. Chaque mesure sur carte de cette liste attend un go et un plafond (décision 2) :
- si l'objectif est la requête de 8 192 + 512 jetons, faire l'étape 5 avant les étapes 2 et 3, une fois le T8 de la route GEMM accordé et le gel ROADMAP §4 levé (décisions 3 et 9) ; viser alors 3,5 à 4,4 s au premier jeton à 8k, pas 2 s ;
- découper l'attention du préremplissage pour sortir du pic de 6,44 Go, puis vérifier l'identité sur carte ; cette découpe appartient à l'étage 1, que le gel retient aussi par défaut ;
- écrire les fusions comme des fonctions de périphérique partagées, avec l'arbre de réduction de candle émulé, vérifié au chargement, et un repli sur l'opération candle. Elles restent alors au bit près, sans le T8 que l'audit prévoyait à l'étape 2, et la route P les réutilise ;
- attendre 155 à 182 tok/s de la route D (*estimé*), pas les 200 de l'audit, qui étaient une borne haute ;
- chronométrer `tv_tetra48_seg` sur le fichier Tetra nu : sur le fichier servi, seul gate+up fusionne, car le groupe q+k+v porte un v int4 (`planesbench.rs:3954-3956`) ;
- finir l'étape 0 bis en gardant le jeton sur la carte, avec la clé d'argmax du §4 et sans la conversion f32 des logits : environ 1 à 1,5 % (*estimé*).

Le premier pas ne coûte aucun dollar, mais chaque run y demande un go :
- quatre réponses de l'opérateur : l'objectif, un plafond de dépense, la lecture du gel ROADMAP §4, et le T8 du préremplissage pour la route GEMM (§11) ;
- G0, la mesure de τ contre le 4B scellé sur le Mac. Il faut 4 à 6 jours avec un portage d'environnement : DeepSpec fige torch 2.9.1 et triton 3.5.1, `llvq-tetra` demande torch 2.14 au moins. Il faut aussi un préreg tamponné, et un poste laissé utilisable : ncpu−4 fils de calcul et `nice` (`LLVQ_THREADS` ne règle que les binaires Rust ; en torch, `torch.set_num_threads`). G0 ne sert que la piste spéculative fermée le 2026-08-05 : l'opérateur dit s'il demande la levée de ce verdict ;
- la lecture du SASS des noyaux depuis l'étage devel de l'image : sans carte, mais au prix d'un build d'image.

Viennent ensuite deux jobs payants, chacun sur go et sous plafond. La dette de la forme groupée, GSM8K par le noyau et MMLU par la porte de 57 questions, coûte environ 2,4-2,8 $, et 3,2-3,6 $ avec la référence du pic de logits (*estimé*). Une mesure combinée sur carte coûte environ 1 à 1,6 $ (*estimé*). Elle informe le choix entre route P et route D ; l'opérateur décide.

Jusqu'au palier 4E, le programme dure 27 à 42 semaines pour une personne et coûte 17 à 26 $ (*estimé*). Le programme entier dure 49 à 79 semaines et coûte 26 à 46 $, option KV int8 comprise (*estimé*, somme du tableau du §10). Ces deux sommes excluent la forme groupée au 8B et au 14B, 10 à 13 $ de plus, et l'affinage d'un drafter, 6 à 60 $. Les durées sont déjà doublées, parce que les vérificateurs ont trouvé chaque plan optimiste d'un facteur 2. Une raison de ce doublement, un build d'image à chaque itération de noyau, ne vaut plus pour le corps des noyaux (`LLVQ_KERNEL_DIR`, §5). Les autres tiennent : files HF de 30 min à 9,5 h, ni nsys ni ncu, du code de warp que le Mac n'exerce pas. Aucune marge n'est prévue pour un pari raté.

## 2. Où part le temps aujourd'hui

Budget d'un pas du 4B, prompt de 5 jetons, forme groupée, 7,44 ms *calculé* sur le débit *mesuré* :

| poste | ms | étiquette |
|---|---|---|
| plancher de géométrie des 252 projections : staging, barrières, réduction, lancement | ~2,3 | *mesuré* au banc (`tv_f1r_v3g`, fichier Tetra nu, `banc-252-2026-10-06.txt`) ; ±0,3 *estimé* pour le mélange servi |
| flux des poids des projections, 1,158 Go | 1,7-2,3 | 1,7 *calculé* au débit du banc au-dessus du plancher, 681 Go/s ; 2,3 *estimé* si les noyaux int4 lisent moins bien |
| 96 rotations, un CTA chacune | 1,04-1,25 | 1,04 *calculé* : 72 × 8,05 + 24 × 19,30 µs (`rotation-cuda-2026-08-05.txt`) ; 1,25 *estimé* en place |
| tête liée int4, 218,8 Mo | ~0,32 | *estimé* |
| environ 580 petits lancements : normes, RoPE, cuBLAS, softmax, résidus, SiLU, argmax | 1,3-1,5 | *estimé* à ~2,2 µs l'un |
| somme | 6,7-7,7 | contre 7,44 *calculé* |

Détail des postes :
- Au banc, sur le fichier Tetra nu de 252 matrices et à la tuile servie, le plancher `nullk` prend 2,294 des 3,734 ms de la passe (*mesuré*), soit 61 % (*calculé*). Ce banc fait tourner le noyau `tv_f1r_v3g`, pas le noyau servi.
- QTIP, dans sa propre géométrie de 128 CTA de 1 024 threads, finit des projections de 0,91 Go en 2,246 ms (*mesuré*, `f2-p3-qtip-banc-2026-08-21.txt`). Notre plancher mesurait 2,306 ms le même jour. Ce plancher appartient à notre géométrie de lancement, pas à la carte.
- Chaque CTA Tetra restage toute l'activation f32 depuis le L2 : 1,71 Go par jeton, plus que les poids (*calculé*).
- Chaque rotation tourne sur un SM pendant que les 141 autres attendent.
- Chaque jeton finit sur un `to_scalar` bloquant (`model.rs:2142`), puis le jeton repart vers la carte (`model.rs:2147`).
- De 5 à 8 192 positions, le pas grandit de 5,46 ms (*calculé* : 12,90 − 7,44). Si cette croissance vient de la lecture du cache KV, soit 1,208 Go (*calculé*), cette lecture tourne à environ 221 Go/s, 26 % de la crête. L'attribution reste une inférence.
- Le préremplissage coûte 3,98 à 4,26 ms par jeton de prompt, de 2 048 à 8 192 jetons (*calculé* sur les temps au premier jeton *mesurés*, forme étendue, `trims-contexts-2026-10-30.txt`). La forme groupée fait 4,27 ms par jeton de prompt à 8k (*calculé* : 35,02 s ÷ 8 192, `served-grouped-2026-11-02.txt`).
- Chaque jeton de prompt paie 252 copies `regroup` et 168 lancements int4, sauf la première rangée de chaque appel, qui n'en paie que 84 (*calculé* sur `model.rs`).
- Sur les mêmes 1 382 608 jetons de prompt, le bras noyau a pris 20,8 fois plus longtemps que le bras dense (*calculé* : 1,514 h contre 4 min 22 s, une passe par bras, sans plage, `f1e-census-2026-09-11.txt` et `f1e0-2026-09-10.txt`, objet q5). Le préremplissage ne passe que la dernière rangée par la tête, donc le rapport à même tête serait proche du brut. Le bras dense revient à 0,19 ms par jeton au plus (*calculé*, borne haute).

Repères sur la même L40S : llama.cpp IQ2_XXS fait 312,9 ± 2,3 tok/s (*mesuré*, `m4-iq2-cuda-2026-08-30.txt`). vLLM AWQ fait 200,5 tok/s [200,4-200,6] (*mesuré*, `awq-vllm-4b-2026-08-17.txt`). Cela fait 391 et 535 Go/s (*calculé* : tok/s × taille du fichier). Nous lisons 184 Go/s (*calculé* : 133,3 × 1,377 Go).

Notre temps hors flux de poids à la crête vaut environ 160 µs par couche (*calculé* : (7,44 − 1,59) ms / 36 couches). Il comprend 11 à 28 µs de flux plus lent que la crête (*estimé*, §1) : le coût fixe tient vers 135 à 150 µs par couche (*estimé*). Le mégakernel d'AlpinDale paie 9,2 à 9,5 µs de barrières et d'instructions par couche (*mesuré* par l'auteur, Qwen3-0.6B, RTX 5090). Les deux chiffres viennent de cartes et de modèles différents : ils donnent un ordre de grandeur et ne se divisent pas.

Réduire les octets ne rapportera qu'une fois ce coût fixe retiré. Au palier 4E, les octets reprendraient environ 80 % du pas (*estimé*).

## 3. Ce que disent les publications récentes

Les rapports de vitesse de ce tableau sont ceux des auteurs, sur leur carte et contre leur propre référence. Aucun ne s'applique à nous tel quel.

| technique | référence | résultat publié | ce que ça donne chez nous |
|---|---|---|---|
| mégakernel persistant | Hazy « No Bubbles » (2025-05) ; MPK, OSDI '26 ; ForgeMegakernel (2026-09) | 78 % de la bande passante d'une H100 sur Llama-1B ; Qwen3-8B sur A100 en 12,5 ms par jeton contre 14,5 pour vLLM et SGLang ; Qwen3-4B sur H100 ×1,12 face à SGLang sous graphes | levier 1. Ces moteurs partaient de graphes ; nous partons de 929 lancements sans graphes |
| mégakernel sur sm_89 | AutoMegaKernel (2026-06) | L40S, int8 contre bf16 sous graphes : ×1,25 pour deux fois moins d'octets, soit 62,5 % de la bande passante utile de cuBLAS sous graphes, octet pour octet (*calculé* par la contre-vérification de la vague 1 ; poids aléatoires, cache vide) | seul point publié sur notre carte, et c'est une perte par octet |
| lancement dépendant (PDL) | llama.cpp PR #22522 (2026) | +13 % à tg128, +6 % à tg512, Qwen3-4B Q4_K_M, RTX PRO 6000 (branche de travail, une passe par case) | absent sur sm_89 ; levier de la carte produit |
| décoder une fois, k rangées | Marlin (PPoPP 2025) ; ExLlamaV3 dp4a int8 (2026-10) ; llama.cpp MMVQ et MMQ | Marlin environ 3,9 fois plus rapide que FP16 jusqu'au lot 16-32 | levier 2 |
| t(k) des noyaux treillis | CodeGEMM (NeurIPS 2025), table 9 | QTIP 10,5 et QuIP# 9,5 fois le temps d'une rangée pour 16 rangées (*calculé* sur leurs latences) ; l'article ne dit pas comment k > 1 a été appelé, et le noyau public de QTIP impose N = 1 | aucun noyau treillis publié ne montre un t(k) plat |
| GEMV invariant multi-rangées | magnitude (2026) | 16 rangées identiques à M = 1 bit pour bit, sous une configuration de lancement (commentaire des auteurs) | précédent direct du levier 2 |
| invariance au lot | Thinking Machines (2025-09) | sorties identiques quel que soit le lot | nos projections l'ont par l'ordre de réduction fixé par la ligne, vérifié sur l'hôte seulement ; la brique la rendrait exacte par construction, sous T8 ; l'attention cuBLAS ne l'a pas |
| drafters par blocs | DeepSpec (DeepSeek, 2026), qui livre des drafters DSpark, DFlash et EAGLE-3 ; DFlash (z-lab) | DSpark sur Qwen3-4B BF16 à T = 1 : τ 6,11 sur GSM8K, 3,64 sur MT-Bench ; DFlash-b16 de z-lab ×4,91 sur H200 à T = 0, contre une base Transformers, avec 16 rangées à vérifier | levier 3, avec des drafters pour nos trois tailles |
| EAGLE-3 | arXiv 2503.01840 | ×4,44 en moyenne, τ 6,23, LLaMA-3.1-8B à T = 0, dans le harnais des auteurs ; ×2,36 sous SGLang sur H100, MT-Bench | option à petite tête, à condition d'un checkpoint à vocabulaire réduit |
| brouillon sans octets | SuffixDecoding (NeurIPS 2025) ; Token Recycling (ACL 2025) ; Lookahead | jusqu'à ×5,3 sur du travail agentique ; environ ×2 ; jusqu'à ×1,8 sur MT-Bench (selon leurs auteurs) | aucune question de règle 6 |
| noyau de vérification | TensorRT-Edge-LLM (Jetson Thor) ; « Speculative Decoding: Performance or Illusion? » | EAGLE ×2,8-3,5 en NVFP4 contre ×1,4-1,6 en INT4 AWQ, à acceptation presque égale ; les deux vérifient sur cœurs tenseurs, et NVIDIA juge son GEMM INT4 trop faible pour des arbres de 64 à 128 jetons. La vérification prend 42 à 95 % du temps d'un tour (*mesuré* par les auteurs du second article) | le coût multi-rangées de la vérification fixe le multiplicateur ; des cœurs tenseurs ne suffisent pas |
| attention GQA split-KV | FlashInfer | « 40 %+ » de la bande passante en GQA, noyau sur cœurs CUDA, 8 têtes par groupe, A100 et H100 ; aucun chiffre GQA sur Ada | η à mesurer sur Ada |

## 4. Huit propriétés propres à Tetra48 et à notre moteur

Le noyau ne sert qu'un format. Huit propriétés de Tetra48 et de notre moteur permettent des noyaux qu'un moteur générique ne peut pas écrire. Les points 1 à 6 s'appuient sur le code. Le point 7 est une conception. Le point 8 a été vérifié par la simulation d'un vérificateur. Aucun n'a été vérifié sur carte.

1. **Des produits de bloc entiers, à portée.** Une valeur décodée vérifie |y| ≤ 10 et ‖y‖² ≤ 416 (`llvq_tetra48.cuh:26-30`). Contre une activation int16, un produit de bloc resterait sous 100 × 32 896 < 2²² (*calculé*). dp4a et IMMA int8 rendraient alors le même entier, bit pour bit, pour 1 rangée comme pour 16. Aujourd'hui l'activation est f32 : ce passage est un T8 (§6).
2. **Un treillis F₂-linéaire.** c1|c2<<8|c3<<16 = TA[s8] ^ TB[b1|b2<<1|b3<<5], avec deux tables de 64 mots, soit 512 octets (`LINEAR_COLUMNS`, `llvq-search/src/tetra/mod.rs:105`, 0 écart sur les 4 096 entrées). Une recherche dans 64 mots coûte 2 wavefronts de banque, contre 3,5 à 4 dans la table de 16 Kio (*calculé*).
3. **Un ordre de réduction fixé par la ligne.** La lane l somme les blocs j ≡ l (mod 32) par j croissant (`tv_tetra48_h.cu:67-88`). Cet ordre ne dépend ni de la tuile, ni du warp, ni du CTA. Une ligne garde donc ses bits dans une géométrie persistante, si l'expression garde sa forme sous `--fmad=true`.
4. **Un noyau multi-rangées bit-identique par rangée, sur l'hôte.** `tetra48_dot_rows` rend chaque rangée comme le noyau à une rangée dans le test hôte, compilé par clang++ avec `-ffp-contract=off` (`llvq_tetra48.cuh:135-205`). Sur carte, sous `--fmad=true`, cette identité n'a jamais été vérifiée. C'est la base d'une spéculation exacte.
5. **Une rotation sans réduction dont l'ordre dépend du partage.** Chaque sortie suit la même suite d'opérations par paires, quel que soit le découpage entre threads : signes, papillons de Walsh-Hadamard, puis une chaîne de FMA tenue par un seul thread. Toute partition qui garde l'ordre des étages reproduit donc les bits. On peut la recalculer dans chaque CTA, ou la couper en deux phases. La forme en deux phases lèverait le mur `rot_apply` du 32B : à n = 25 600, k = 25 ≤ KMAX (*calculé*). Elle n'est ni écrite ni tournée.
6. **Un cache qui grandit sans fenêtre réservée.** Sur les stores `Grow` et `Prealloc`, `truncate` ne déplace qu'une longueur (`model.rs:444`) : le retour arrière spéculatif coûte O(1). Le store `Cat` de la forme étendue recopie le préfixe ; c'est celui du 8B et du 14B aujourd'hui. Un noyau relancé tous les N jetons réserve pos + N, soit 9,4 Mo à N = 64 (*calculé*), au lieu d'une fenêtre de 1,21 Go.
7. **Un transcodage au chargement.** Le mot de 48 bits pourrait être réordonné en plans u32 et u16, tuile par tuile. Le fichier scellé et `codebook_fingerprint` ne bougeraient pas.
8. **Un argmax reproductible sur carte.** Une clé de 64 bits reproduit la règle de départage des égalités de `fast_argmax` de candle. Un vérificateur l'a testée sur 300 cas adverses : 0 écart, contre 218 pour la clé proposée en vague 1. Le jeton peut donc rester sur la carte.

## 5. Levier « un jeton, un lancement »

Avec 4E, le 4B passerait de 133,3 à 340-425 tok/s (*estimé*). À 8k, il passerait de 77,5 à 156-216 tok/s pour η de 0,40 à 0,60 (*estimé*). Le 14B passerait de 57,2 tok/s [57,1-57,2] (*mesuré* avant trims, forme étendue, `paper-table-2026-09-25.txt`) à 114-148 tok/s court et 79-107 à 8k (*estimé*). 4E attend la lecture de A2 et le choix du grand modèle. Sans 4E, la route P et notre noyau d'attention plafonnent à 290-335 tok/s court, sous T8, si l'opérateur dégèle les réglages des CTA persistants et ne range pas la route P sous A2. Sinon, la route D et notre noyau d'attention plafonnent à 159-189 tok/s (*estimé*).

| étape | 4B court | 4B à 8k | exactitude |
|---|---|---|---|
| route P : un noyau persistant par segment de couche | 250-300 | 105-113 | bit pour bit visé |
| route P et notre noyau d'attention | 290-335 | 144-188 | T8 pour l'attention |
| 4E : un noyau par N jetons | 340-425 | 156-216 | identique à l'étape précédente |
| route D : cure de lancements dans des grilles normales | 155-182 | 84-91 | bit pour bit visé |
| route D et notre noyau d'attention | 159-189 | 102-131 | T8 pour l'attention |

Toutes ces valeurs sont *estimées*, en tok/s. Deux vérificateurs les ont reproduites à 1 % près à partir des entrées du juge, sauf une ligne. Pour « route P et notre noyau d'attention », le juge écrivait 279-336 et 141-188. La contre-vérification relève le bas à 292 et 144,5, arrondis à 290 et 144 dans les tableaux ; le haut, 336, y est arrondi à 335. Dans le cas lent à 2,6 µs, l'économie d'attention vaut 0,55 ms et non 0,4.

« Bit pour bit » est la cible des routes P et D, pas un fait. Elles réécrivent la norme et RoPE de candle, dont l'identité sur carte n'est pas établie (§12). L'audit du 2026-10-29 range ces fusions sous une porte à tolérance. Si une seule rangée change d'un bit, la route relève d'un T8.

**Mécanisme.** Un noyau coopératif de 142 CTA, un par SM, de 512 à 768 threads, exécute le jeton entier.
- Les warps réclament les lignes, un warp par ligne, avec l'arithmétique d'aujourd'hui.
- Chaque CTA recalcule la norme et la rotation dont il a besoin. Le Walsh-Hadamard tient dans les registres d'un warp avec `shfl_xor` : 0,6 à 1,3 µs à n = 2 560 (*estimé*), contre 8,05 µs aujourd'hui.
- Des compteurs portent les dépendances fines : q, k, v vers l'attention, l'attention vers o, les tranches de down_proj en int4.
- Il reste 97 attentes globales par jeton au 4B : 36 normes d'entrée, 36 normes MLP, 24 rotations down, 1 argmax. Il en reste 128 au 8B et 102 au 14B (*calculé*, revu par deux vérificateurs).
- L'argmax final se fait sur carte. Une boucle bornée à N jetons supprime l'aller-retour hôte.
- La route P coupe la couche en segments persistants, par exemple ln2, rotation, gate et up avec SiLU, attente, rotation down, puis down et résidu. La route D câble `tv_tetra48_seg`, fusionne RoPE et l'écriture KV, passe les résidus en épilogue et accélère la rotation, sans CTA persistant.
- Le cycle d'itération est court. `LLVQ_KERNEL_DIR` remplace déjà toutes les sources CUDA au chargement, et il est refusé à côté de `LLVQ_CONFIG` (`served.rs:111`). Modifier le corps d'un noyau ne demande donc aucun build sur le chemin de banc. Un build reste dû pour un nouveau point d'entrée, un changement de lancement côté Rust, un run par la porte servie, ou une livraison.

**Les risques.**
- Aucune source ne publie le coût d'une attente dépendante sur sm_89. Sur RTX 5090, AlpinDale mesure 2,2 µs par barrière atomique et environ 3 µs par grid.sync.
- Le seul mégakernel publié sur sm_89 perd par octet face à cuBLAS sous graphes (§3).
- Le débit d'un flux Tetra persistant n'a jamais été mesuré. Entre 450 et 740 Go/s, 4E irait de 281 à 426 tok/s (*estimé*).
- La valeur nette d'une frontière supprimée éclaire le choix entre route P et route D. À 4 µs ou plus par frontière, le mégakernel gagnerait au moins ×1,23 sur la route D ; à 2,5 µs ou moins, au plus ×1,15 (*estimé*, `wave2/gap-2.md`).
- La fusion D1 donne 5,5 µs par frontière (`d1-fusion-servie-2026-08-24.txt`) et `persall` 8,4 µs (`a3-occupation-banc-2026-09-01.txt`), *calculés* tous deux. Aucun des deux ne se transpose à une frontière de mégakernel. D1 inclut la fusion de petites grilles et vient de deux médianes non entrelacées sur Planes14. `persall` n'a aucune dépendance entre sites. La mesure 0.6 donnera la valeur nette.
- Il n'y a pas de nsys dans l'image. L'ajouter est un changement de recette, à décider par l'opérateur (décision 13). Une trace des noyaux remplacerait trois inférences : la part hôte du pas et la part de l'attention au préremplissage (§12), et l'attribution des 5,46 ms au cache KV (§2). HF Jobs refuse les compteurs ncu ; la trace de nsys n'en dépend peut-être pas, à vérifier au premier job.

**Ce que l'opérateur doit trancher.**
- La lecture de A2. Si A2 désigne la capture de graphes avec sa fenêtre fixe, un noyau lancé normalement sur `Grow` n'est pas du A2. Seul l'opérateur peut le dire. La question vaut pour 4E, qui boucle sur N jetons sans l'hôte, et peut-être pour la route P, relancée à chaque jeton. Si l'opérateur range la route P sous A2, l'étage 2 attend aussi sa lecture.
- Trois réglages du 2026-08-05, écrits pour le noyau Slot32. Ils n'existent que dans `docs/archive`, sans ligne dans ETAT §6 : les lever demande un go (règle 1), et la règle 3 ne s'applique pas.
  - Les CTA de 512 threads, rangés « à ne pas re-dépenser » par l'audit du 2026-08-05. Avec un warp par ligne, un CTA de 512 threads porte 16 lignes, et k (1 024 lignes) ne remplit que 64 CTA, contre 128 à 256 threads. Un CTA persistant qui réclame ses lignes n'a plus ce problème.
  - Les tables en mémoire partagée. Le refus venait d'un modèle d'occupation, 24,6 Ko par bloc, donc 4 blocs par SM, tiré d'un audit statique sans job. Un CTA persistant par SM n'est pas limité en blocs et charge ses tables une fois.
  - Le décodage gelé jusqu'à la lecture du SASS. Le poste 0.4 remplit sa condition ; le lever reste une décision de l'opérateur (règle 1).

## 6. Levier « décoder une fois »

Vérifier 8 rangées coûterait 1,14 à 1,29 fois un pas simple (*estimé*, pari). Le noyau multi-rangées SIMT d'aujourd'hui paie bien plus par rangée supplémentaire (*calculé*, `wave2/gap-3.md`) :
- au moins 0,25 ms de lectures d'activation sur les 168 matrices Tetra servies, même sans conflit de banque ;
- 0,50 à 1,99 ms avec les conflits à 8 voies d'aujourd'hui, selon la largeur de chargement que le SASS dira ;
- environ 1,08 ms de plus pour relire l'int4 et la tête, qui n'ont pas de noyau multi-rangées.

À k = 1, la brique tournerait 1,16 à 1,84 fois plus vite que le noyau servi, au banc (*estimé*).

**Pourquoi les noyaux publiés ne le font pas.** Aucun noyau treillis publié ne montre un t(k) plat. CodeGEMM mesure 10,5 et 9,5 fois le temps d'une rangée pour 16 rangées avec QTIP et QuIP#, mais ne dit pas comment ces noyaux ont été appelés au-delà d'une rangée. Le noyau public de QTIP code en dur `static_assert(N == 1)`. TensorRT-Edge-LLM montre la limite des cœurs tenseurs : son GEMM INT4 sur cœurs tenseurs ne donne que ×1,4-1,6 en spéculation (§3). Il faut un décodage réutilisé par toutes les rangées, comme Marlin le fait pour l'int4.

**Mécanisme.**
- Un thread décode chaque mot Tetra48 une seule fois, en octets signés. Le décodeur v3 produit déjà des octets biaisés `val + 128`.
- À k = 1, dp4a fait le produit dans les registres.
- À k ≥ 2, les octets passent par une tuile privée de warp en mémoire partagée, adressée en XOR, puis par `ldmatrix` et `mma.m16n8k32` int8. L'activation devient un int16 par tranche de 384, coupé en deux plans int8.
- Un épilogue f32 à ordre fixe applique ensuite l'échelle de bloc.
- Les enregistrements int4 et la tête passent par le même gabarit, en trois plans, avec le zéro-point plié.
- Les tables restent en mémoire partagée, treillis F₂ de 512 octets compris.

Chaque produit de bloc serait un entier exact. Le décodage, la vérification de k rangées, l'extraction en lot, le préremplissage et une relecture C++ sur l'hôte rendraient alors les mêmes bits.

**Ce que ça coûte.**
- Un événement T8 : l'activation passe de f32 à int16 par tranche. Avec l'attention, un vérificateur en compte au moins deux sur le programme.
- 12 à 20 semaines et 5 à 10 $ (*estimé*).
- La clôture du 2026-08-05 juge les cœurs tenseurs « sans objet » pour un matvec. Elle ne les rouvre que « si le prefill entre au périmètre ». Elle n'existe que dans `docs/archive`, sans ligne dans ETAT §6 : la lever demande un go.

**Un précédent.** Le moteur magnitude, référence de l'audit, livre un GEMV à cœurs tenseurs invariant sur 16 rangées (`GEMV_ROWS = 16` dans `projection.cuh`). Il tourne aussi à M = 1. Ses codes entrent comme entiers exacts et ses activations sont arrondies en f16 ou bf16. L'identité bit pour bit est affirmée dans un commentaire ; nous ne l'avons pas mesurée.

**Sur sm_120.** La carte produit visée est une RTX 5090, que HF n'offre pas. Sur la carte de banc sm_120, une RTX PRO 6000 Server à 1 597 Go/s selon NVIDIA, la passe Tetra du banc prend 4,154 ms à sa meilleure tuile pour 0,98 Go (*mesuré*, `f1d-2026-09-10.txt`). Elle y égale Planes14, qui lit 2,23 fois plus d'octets (*calculé*). Aux tuiles 64 et 128, elle est plus lente que Planes14 : 0,86 et 0,82 fois (*mesuré*). La passe est donc limitée par le décodeur ou la L1, pas par la mémoire ; c'est une inférence, sans compteur. L'INT32 y tourne à plein débit. La brique allège ce décodage : sur sm_120, elle passerait probablement avant le noyau persistant (*estimé*).

## 7. Levier « spéculation exacte »

Une fois la vérification plate et le pas ramené à 2,6 ms par 4E, le gain serait ×1,02-2,53 en chat, ×1,50-3,74 sur du code et ×1,71-4,24 en maths (*estimé*, avec τ pris à 0,5-1,0 fois les chiffres publiés). Sur le noyau multi-rangées SIMT, le gain s'efface quand le pas raccourcit : ×0,6-1,4 en chat au palier 4E (*estimé*, temps du drafter tenu constant).

Dans le moteur d'aujourd'hui, une vérification de 4 rangées passe par le chemin de préremplissage. t(4) y vaut environ 2,06 à 2,13 fois un pas simple en contexte court (*estimé* par les vérificateurs du plan contradicteur de la vague 2, annexe A ; le t(4) servi n'est pas mesuré). Un brouillon à zéro octet y gagnerait encore ×1,1-1,4 sur du code ou de l'extraction, et rien de sûr en chat, ×0,85-1,06 (*estimé*). Ce qui bloque est l'exactitude. N rangées en un appel s'écartent de N appels d'une rangée de 4,7e-2 à 6,8e-2 sur un logit selon la longueur du prompt, écart jamais attribué (*mesuré* sur l'objet q5 de septembre, avant trims et forme groupée, `f1e0-2026-09-10.txt` et `f1e-prefill-sweep-2026-09-11.txt`). La vérification ne reproduirait donc pas le glouton sans T8. Le chemin multi-rangées 1a est le préalable.

**Les drafters publiés.** DeepSeek publie DeepSpec avec `dspark_qwen3_{4b,8b,14b}_block7`, `dflash_qwen3_*_block7` et `eagle3_qwen3_*_ttt7`. Sur Qwen3-4B en BF16, à T = 1, DSpark accepte en moyenne 6,11 jetons sur GSM8K, 5,38 sur HumanEval et 3,64 sur MT-Bench (*mesuré* par les auteurs). Bloc 7 signifie 7 propositions et 8 rangées à vérifier, donc τ ≤ 8.
- La licence MIT couvre le code. Les dépôts de poids n'ont aucune licence déclarée : redistribuer des poids de drafter ré-encodés n'est couvert par rien.
- Les drafters ont été entraînés en mode sans raisonnement. Le README conseille un nouvel affinage pour le mode raisonnement.
- Chaque checkpoint embarque sa propre copie BF16 de la tête et de l'embedding complets. Le checkpoint DFlash 4B compte ainsi 1,315 milliard de paramètres, dont 0,78 milliard pour ces deux tables (*calculé* : 2 × 151 936 × 2 560). Partager notre table int4 à la place serait un changement jamais entraîné ; G0 doit le mesurer.
- Le drafter EAGLE-3 de DeepSpec garde le vocabulaire complet de 151 936 lignes. Une option à tête de 32k demande un autre checkpoint, ou un réentraînement, et son τ n'est pas celui de la table publiée.

**Pourquoi la sortie resterait identique.** La vérification gloutonne compare l'argmax de chaque rangée vérifiée au jeton proposé. Si chaque rangée vérifiée a les mêmes bits que le décodage simple, la sortie spéculative est identique, sans tolérance.
- Notre noyau multi-rangées rend chaque rangée bit pour bit dans le test hôte. Sur carte, ce n'est pas vérifié (§12).
- Il manque une attention invariante en rangées, des noyaux int4 et tête multi-rangées, et la fin des copies `regroup`.
- Au 8B et au 14B, il faut d'abord la forme groupée sur `Grow`, pour un retour arrière en O(1).
- Il faut aussi un test à drafter oracle : il rejoue la suite gloutonne comme propositions et vérifie chaque rangée de chaque tour.

**Ce qui n'est pas mesuré.**
- τ contre l'objet scellé. Les 74,45 % d'accord top-1 cités jusqu'ici (*mesuré*, `kl-temperature-4b-2026-09-16.txt`) viennent de l'artefact Tetra nu, sans row scales ni parts int4. Aucun τ n'est publié pour une cible à 2-3 bits.
- La platitude de t(k) sur de vrais mots.
- Le coût de l'étage Markov de DSpark dans notre moteur.
- Le préremplissage du drafter. Si l'objectif est une requête de 8 192 + 512 jetons (objectif (b) du §11), il doit aussi lire les 8 192 positions du prompt : +4,9 s à 8k sur le chemin à rangées (*estimé*, plan contradicteur de la vague 2). Il faut le différer ou le loger dans la capacité libre du préremplissage, sinon le premier jeton recule.

**La règle 6.** Un drafter ajoute des octets. Au 4B, en Tetra, avec la tête partagée (*estimé*) :

| drafter | b/poids noyau | b/param |
|---|---|---|
| DSpark | +0,41-0,42 | +0,37-0,38 |
| DFlash | +0,33 | +0,29 |
| couche EAGLE-3 et tête int4 de 32k | +0,17-0,18 | +0,16-0,17 |

La marge du triplet au 4B vaut 0,458 b/poids noyau avec un KV q8, soit 0,414 b/param. Elle vaut 0,318 b/poids noyau avec le KV f16 que le moteur stocke, soit 0,287 b/param (*calculé* : b_max de 3,00 ou 2,86 moins les 2,542 b/poids noyau du 4B scellé). b_max n'est défini qu'en b/poids noyau. Il est dérivé pour un 70B dense (`note-produit-2026-08-13.md`) ; transposé au 4B, il ne départage pas un écart de quelques centièmes.
- Avec le KV f16, DSpark dépasse la marge. DFlash la dépasse de 0,012 b/poids noyau, moins que la précision de son estimation.
- Ce tableau suppose la tête partagée. Sans partage, la tête et l'embedding propres au drafter ajoutent +3,09 b/param en BF16, tel que publié, ou +0,82 à +0,87 en int4 g64 (*calculé* : 2 × 151 936 × 2 560 paramètres sur les 4,02 G du 4B).
- Un vérificateur conteste aussi la division par les 3,633 G poids de projection du 4B, que la formule du triplet ne prévoit pas.
- L'opérateur fixe l'unité, l'hypothèse KV et le partage de marge avec le passage de down_proj en int4.

**Ce qui ne demande aucune décision de règle 6.** Le brouillon à zéro octet : recherche dans le prompt, SuffixDecoding, Token Recycling. Il ne lit aucun poids supplémentaire. Il rapporte surtout sur l'extraction, le code et le texte agentique.

**Calendrier.** 2 à 4 semaines pour le brouillon à zéro octet, 6 à 10 pour un drafter neuronal, 4 à 8 $ (*estimé*). Il faut ajouter 6 à 60 $ si G0 montre qu'un affinage sur l'export scellé est dû.

## 8. Le préremplissage et les autres compléments

**Préremplissage**, de la pièce la moins chère à la plus chère :
- Découper l'attention du préremplissage par blocs de 512 à 1 024 rangées. Le correctif ne coûte aucun bit de format : c'est le « 0 bit » de l'audit ML. L'identité des sorties n'est pas acquise, car cuBLAS peut changer d'arrondi avec la forme des produits, comme l'A/B de la forme groupée l'a montré sur carte. Un contrôle d'identité sur carte le dira ; s'il échoue, la découpe devient un T8 du chemin de préremplissage. Le pic tomberait sous les 6,44 Go d'aujourd'hui au 4B et sous les 8,05 Go au 14B (*calculé*, 40 têtes au 14B).
- Le chemin multi-rangées : noyaux int4 et tête multi-rangées, fin des copies `regroup`. Le premier jeton à 8k tomberait à 7,2-11,1 s (*estimé*). C'est aussi le socle de la vérification et du lot.
- La route GEMM : chaque matrice est déquantifiée dans un tampon f16, puis cuBLAS fait le produit. Le tampon d'une matrice MLP pèse 50 Mo au 4B et 178 Mo au 14B (*calculé*). Avec l'attention actuelle, le premier jeton à 8k arriverait après 3,5-4,4 s (*estimé*). Son arithmétique est celle du bras dense : T8 sur le chemin de préremplissage. 1 à 2 semaines et 1 à 3 $ (*estimé* par le juge ; la revue du 2026-10-20 comptait 2 à 4 jours).
- Un noyau d'attention de préremplissage sur cœurs tenseurs, sous T8. Le premier jeton à 8k tomberait vers 1,7-1,9 s (*estimé*, sur une attention supposée à 0,1-0,3 s à 8k). Sur cœurs CUDA, l'attention causale à 8k demande au moins 0,22 s au pic FP32 de 91,6 TFLOPS (*calculé*) : compter alors environ 2 s. Au pic tenseur de 362 TFLOPS que retient la vague 1, l'attention coûterait 0,055 s et les projections 0,164 s (*calculé*). Ce pic n'est pas vérifié en accumulation f32, et `portage-noyau-cuda.md` lit 181 TFLOPS en dense : le préremplissage entier à 8k, projections et attention comprises, ne descendrait donc pas sous 0,22 à 0,44 s (*calculé*).

Le noyau split-KV de l'étage 3 sert le décodage et la vérification. Un premier jeton vers 1,7-1,9 s demande en plus le noyau de préremplissage sur cœurs tenseurs du point précédent. Ce noyau aurait ses propres bits, sous son propre T8 : c'est une exception au premier invariant du §10, à trancher. Aucun étage du §10 ne le chiffre.

Le 14B met aujourd'hui 74 à 81 s pour son premier jeton à 8k (*estimé*). Une requête de 8 192 + 512 jetons y prendrait environ 99 à 106 s en lot 1 sur la L40S (*estimé*). Le plancher A5 du produit, au plus 60 s par document, est défini en lot de 8 sur la 5090 : la comparaison ne donne qu'un ordre de grandeur.

**Notre noyau d'attention.** Un noyau split-KV groupé : découpe fixe et alignée de 256 positions, combinaison à ordre fixe, longueur lue en mémoire carte. Il donnerait les mêmes bits au décodage et à la vérification. Son η reste inconnu sur Ada.
- Le « 40 %+ » de FlashInfer vient d'un noyau GQA sur cœurs CUDA, à 8 têtes par groupe, sur A100 et H100.
- Ces cartes ont 5 à 10 fois moins de FP32 par octet de bande passante que la L40S (*calculé* sur les fiches : environ 20 FLOP par octet sur H100, 10 à 12,5 sur A100, 106 sur L40S).
- Qwen3-4B n'a que 4 têtes par groupe, donc deux fois moins de calcul par octet de KV.
- Les deux écarts vont dans le même sens. FlashInfer attribue son « 40 %+ » au manque de calcul des cœurs CUDA de l'A100 et de la H100 ; sur la L40S, à 4 têtes par groupe, ce noyau serait bien moins limité par le calcul. Rien ne dit quelle autre limite prendrait sa place : le chiffre ne se transpose pas, et η reste à mesurer.

**Mur du 32B.**
- Le `rot_apply` à un bloc reste muré : 102 400 octets de mémoire partagée contre 101 376 en opt-in. Le commentaire de `llvq_rot.cuh:46-48` se trompe de factorisation et de raison, mais sa conclusion tient pour ce noyau.
- La forme en deux phases tiendrait : k = 25 ≤ KMAX à n = 25 600 (*calculé*).
- Ce mur ne vaut que sur les cartes à 99 Kio d'opt-in, L40S et RTX, donc aussi la 5090. L'A100, à 166 912 octets, passerait le 32B sans levée (*calculé* sur les limites de mémoire partagée, audit ML du 2026-11-02 ; jamais tourné). ETAT §4 et CLAUDE.md disent encore que ce mur ferme le 32B quel que soit le format ; l'opérateur tranche avant toute correction.
- Un second mur existe si down_proj reste int4 : `int4_shared_bytes` (`fused.rs:734`) stage l'activation en f32, 102 400 octets à d_in 25 600. Un staging f16, exact, en demande 51 200.
- L'audit ML chiffre la levée par staging en groupes à 1 à 3 jours, exacte au bit. Le juge chiffre l'ensemble du poste 1b, avec une rotation à un CTA plus rapide, à 1,5-2 semaines et environ 0,6 $ (*estimé*). Cette rotation rapide gagnerait 0,2 à 0,55 ms par jeton au 4B (*estimé*).
- Le 32B n'a jamais été encodé. La chaîne réaliste coûte environ 120 à 180 $ (*estimé*). ROADMAP §4 garde aussi le point 32B derrière une porte sur l'écart du 14B au 32B.

**Forme groupée au 8B et au 14B.** Le code existe : `grouped_attention` prend n'importe quel nombre de têtes par groupe. Au 14B, environ 63 tok/s court et 40 à 46 à 8k, contre environ 21 à 8k en forme étendue (*estimé*). Ces chiffres additionnent des gains mesurés dans des jobs séparés ; METHODE §3 dit qu'une telle somme fabrique un chiffre, donc il faut mesurer. Chaque taille demande son fichier de config, son T8 et ses scores MMLU et GSM8K par le noyau. C'est aussi le préalable d'un retour arrière en O(1).

**Le Mac.** Le noyau `tv_q4_metal_tiled` existe dans `llvq-llm/kernels/tv_q4_h.metal`, et `llvq-tetra` l'utilise. `fused_metal.rs` refuse pourtant d_in × 4 > 32 768 octets : aucun fichier scellé ne passe le chemin fusé Metal. Le câbler coûte 1 à 2 jours et 0 $. Le gain n'est pas chiffrable proprement :
- aucune vitesse Rust des fichiers scellés n'est journalisée sur le Mac, hors un relevé de 347 ms par jeton sur une machine chargée (*mesuré*, `gsm8k-pilot-2026-09-26.txt`, non exploité) ;
- `llvq-tetra` fait tourner le 4B scellé en dense f16 sur mps à 35-37 tok/s (*mesuré* hors banc, `hf-chat-4b-2026-10-04.txt`), soit 280 à 300 Go/s s'il lit les 8,04 Go du modèle f16 (*calculé*) ;
- l'ancien objet q5 tournait fusé à 48-50 tok/s (*mesuré* sans journal, selon des messages de commit) ;
- llama.cpp IQ2_XXS fait 114,0 ± 0,6 tok/s sur le même Mac (*mesuré*, `m3-iq2-metal-2026-08-30.txt`), soit 143 Go/s (*calculé* : × 1,25 Go). Ces moteurs se comparent en Go/s, jamais par un rapport de tok/s.

**Trois gestes au bit près**, hors des fusions de l'audit et sans T8 :
- garder le jeton sur la carte avec la clé d'argmax du §4, sans la conversion f32 des logits (`model.rs:2141`) : la fin de l'étape 0 bis de l'audit, environ 1 à 1,5 % (*estimé*) ;
- un régime hôte, avec tampons réutilisés et poignées de fonctions en cache, là où la mesure 0.6 montre du temps hôte exposé. Les graphes avaient mesuré +12,6 % de débit au 4B (ETAT §6) ; ce régime en reprendrait une part sans toucher A2 ;
- des flux concurrents pour k et v, mesurés seuls.

Le passage de down_proj en int4 au lieu d'o_proj (ROADMAP §2.2) déplace des rotations et des octets : chiffrer sa vitesse à côté de sa porte de qualité.

**KV int8 réel.** 0,64 Go à 8k au lieu de 1,21 (*calculé*, ROADMAP §4). Gain ×1,15-1,25 à 8k après le noyau d'attention (*estimé*).

**Lot de 8.** À 8k, chaque pas lit 8 × 1,208 Go de KV ; le gain maximal par rapport au lot 1 n'est que de ×1,87 (*calculé*). L'extraction en lot dépend du préremplissage en lot et du KV int8 plus que d'un noyau de poids plat.

**Les octets, ensuite.** Au palier 4E, les enregistrements int4 et la tête porteraient 46 % des octets, à 4,25-4,5 b/poids (*calculé*, vague 1). Une lecture progressive de la tête par plans de bits ne lirait que 50 à 60 % de ses octets en décodage glouton (*estimé*, vague 1).

## 9. Ce que ce programme ne propose pas

Aucune ligne de ce tableau n'est une clôture. Une piste ne se ferme que sur décision de l'opérateur, et une estimation n'en prononce aucune (METHODE §1).

| idée | statut | raison |
|---|---|---|
| graphes CUDA | clôture existante (A2, ETAT §6) | hors du cœur, règle 3 |
| servir dans un autre moteur | décision existante (`deaa449`) | une décision séparée de l'opérateur |
| plusieurs lignes de sortie par warp | mesure existante | mr2 −10,45 %, mr4 −24,62 % (*mesuré* sur Planes14, `a3-occupation-banc-2026-09-01.txt` ; jamais sur Tetra48, sans ligne en ETAT §6) |
| drafter résident en L2 | différé | ×1,01-1,03 dans un moteur limité par les lancements (*estimé*) |
| repli QuaRot global | différé | ×1,02-1,07 après le premier levier, pour un ré-encodage de chaque taille (*estimé*) |
| auto-drafting par LoRA à codes gelés | différé | 5-8 semaines, 50-200 $, probabilité de succès estimée à 0,3-0,5 |
| drafter 0,6B séparé à 2 bits | différé | perplexité ×2,148 (*mesuré*, `tetrapost-ppl-0.6b-2026-09-15.txt`) |
| sparsité d'activation | écart proposé | à 50 % de sparsité, un bloc de 24 peut être sauté avec une probabilité d'environ 6 × 10⁻⁸ (*calculé*) |
| GEMV par table des activations | écart proposé | 12 recherches dépendantes des données par bloc ; au plus 5 à 8 % de gain, pour une nouvelle géométrie (*estimé*) |
| brouillon tiré d'une partie du mot Tetra | écart proposé | le mot n'a pas de structure du grossier au fin ; erreur relative d'au moins 0,7 (*calculé*) |
| activations FP8 | écart proposé | 3 bits de mantisse |
| argmax certifié par bornes au pire cas | écart proposé | 50 fois trop lâche à d = 2 560 (*calculé*) |
| auto-spéculation par saut de couches | écart proposé | ×1,3-2 publiés, sur des cibles en pleine précision |
| k ≤ 3 rangées SIMT dans le noyau persistant | écart proposé | 135 552 octets de mémoire partagée à k = 3, contre 101 376 (*calculé*) |
| choisir une route à cœurs tenseurs par compte d'opérations | écart proposé | l'écart tient dans la marge d'erreur du compte |

## 10. La séquence

L'ordre dépend de l'objectif, que choisit l'opérateur (décision 1). Ce rapport lit le programme avec l'objectif (b), une requête de 8 192 + 512 jetons sans raisonnement ; ce choix de lecture n'engage aucun travail. ETAT §4 nomme le préremplissage et le nombre de lancements comme les deux murs suivants ; ce programme met le préremplissage d'abord.

Deux invariants valent pour tous les étages :
- Une seule sémantique par rangée et par opération. Le décodage, la vérification de k rangées, les blocs de préremplissage et le noyau persistant la reproduisent au bit. Elle est vérifiée au chargement sur carte, sur le modèle de `check_int4_half_parity` (`fused_cuda.rs:2224`).
- Des fonctions de périphérique écrites une fois : norme, phases de rotation, produits de ligne, épilogues, attention, clé d'argmax. Un étage hérite ainsi des bits du précédent au lieu de payer un nouveau T8. Les fusions de la branche de l'audit sont le premier endroit où les appliquer.

| étage | contenu | effort | payé | décision préalable |
|---|---|---|---|---|
| 0 | décisions, dette groupée, Mac, outillage, ancre llama.cpp, mesure combinée, option sm_120 | 3-4 semaines | 6-8 $ | plafond de dépense ; réglages gelés pour les bras persistants de 0.6 (décision 8) ; `--any-flavor` pour sm_120 |
| 1 | découpe de l'attention de préremplissage, chemin multi-rangées (1a), levée des murs (1b), route GEMM (1c), forme groupée au 8B et au 14B (1d) | 5-8 semaines | 3-5 $, plus 10-13 $ pour 1d | T8 pour 1c et pour chaque taille de 1d ; gel ROADMAP §4 pour la découpe, 1a et 1c |
| 2 | retrait des lancements, route P ou D | 5-8 semaines (P), 4-6 (D) | 1,5-2,5 $ | réglages gelés (P) ; lecture de A2 si elle couvre P ; remplacer des opérations candle |
| 3 | noyau d'attention (3a), option KV int8 (3b) | 4-6 semaines, plus 2-3 | 3-4 $, plus 1-2 | T8 pour l'attention ; T8 de qualité à 8k pour 3b ; nouvelle référence de la porte 4 si la ligne bouge |
| 4E | un noyau par N jetons | 10-16 semaines | 3-6 $ | lecture de A2 ; choix du grand modèle |
| 4K | la brique « décoder une fois » | 12-20 semaines | 5-10 $ | cœurs tenseurs, dp4a et ldmatrix dans le périmètre ; T8 |
| 5 | spéculation exacte | 2-4 puis 6-10 semaines | 4-8 $, plus 6-60 $ d'affinage | verdict spéculatif ; règle 6 |

Tous les efforts et coûts de ce tableau sont *estimés*. 4E et 4K se recouvrent à k = 1 et se complètent pour k ≥ 2 : seule la brique aplatit t(k). 4K passe devant 4E si l'opérateur retient la route D au vu de l'étage 0, si sm_120 se montre limité par le décodeur, ou si l'objectif est le lot de 8.

L'étage 0 en détail. Chaque job de l'étage 0, payant ou local, demande son go, son préreg tamponné et son coût annoncé ; un plafond ne vaut pas go (règles 1 et 2).
- **0.1 Décisions**, 0 $ (§11).
- **0.2 Dette de la forme groupée**, environ 2,4-2,8 $. GSM8K par le noyau sous la config groupée (le bras étendu du 4B a coûté 2,13 $, *mesuré*, `docs/data/jobs.csv`), MMLU par la porte de 57 questions. En option, environ 0,8 $ : le chemin publié contre lui-même, pour expliquer les pas isolés où un logit bouge de 13,13 (ETAT §4 ; ce n'était pas un critère). Un GSM8K hors de la barre appariée du bras étendu serait un argument contre toute nouvelle extension T8. Quel que soit le résultat, chaque extension reste une décision de l'opérateur.
- **0.3 Mac**, 0 $. Câbler `tv_q4_metal_tiled` ; écrire le harnais G0 ; relever les activations aux 84 entrées int4.
- **0.4 Outillage**, 0 $. `LLVQ_KERNEL_DIR` existe déjà pour itérer sans build ; un job doit fournir les 10 fichiers de l'unité servie, 101 541 octets. Restent à faire : porter `LLVQ_TIME_EVENTS`, qui n'existe que dans `planesbench`, dans `fusedrun`, et lire le SASS de `tv_nullk`, `tv_tetra48_h` et du noyau multi-rangées.
- **0.5 Ancre llama.cpp** sur la même L40S, environ 0,05-0,1 $ : débit KV en profondeur et débit de préremplissage, en Go/s.
- **0.6 Mesure combinée**, environ 1 à 1,6 $, en deux processus sur `l40sx1` et sous un préreg tamponné.
  - `planesbench` sur le fichier Tetra nu : bras persistants, noyau multi-rangées de R = 1 à 16 avec identité rangée à rangée, micro-bancs de barrière. Son bras Tetra actuel est `tv_f1r_v3g` ; il faut lui ajouter le noyau servi.
  - `fusedrun` sous la porte, sans `LLVQ_CONFIG`, car la porte servie refuse tout mode de mesure. Les bras découpés et les lancements vides n'existent pas dans `fusedrun` : `LLVQ_SEG_ARMS` appartient à `planesbench`, et `fusedrun` ne fait que le refuser. Ils sont à écrire sur le chemin de banc, avec un moyen d'y choisir la forme groupée, que seul `LLVQ_GQA_AB` sélectionne aujourd'hui, pour ses propres bras. Ce code s'ajoute au poste 0.4. Ces bras donnent la valeur d'une frontière.
  - Le coût suppose des builds courts. Les 148 s de `banc-252` valaient pour 4 bras ; 10 bras ont pris 1 495 s, et le processus L40S de f1d a été tué au plafond pendant son build.
- **0.7 sm_120**, environ 1,7 à 2,75 $, sur go explicite. Le chemin servi une fois, et le même banc, sur `rtx-pro-6000`, avec un bras PDL. Que l'image CUDA 12.4 fasse tourner l'attention cuBLAS du chemin servi sur sm_120 n'est pas vérifié. HF n'offre aucune RTX 5090. Dans `ops/run.py`, seule la sous-commande `bench` lance un binaire arbitraire, et elle exige `--any-flavor` hors `l40sx1`.

La mesure 0.6 informe l'opérateur, qui choisit la route. Aucune ligne n'est une porte : la passe chaînée et l'attente sont des mesures de banc, et la porte de chaque route reste le débit de bout en bout. Lecture proposée, qui couvre tous les cas :
- noyau multi-rangées non identique sur carte : route D, et 4K passe devant 4E ;
- sinon, passe persistante chaînée ≤ 2,0 ms et coût d'une attente dépendante ≤ 2,5 µs : route P, 250-300 tok/s attendus (*estimé*) ;
- sinon, passe chaînée ≤ 2,6 ms : route P ou D ;
- autrement : route D.

Les portes de chaque étage, toutes à préenregistrer. Chaque arrêt est une proposition ; il revient à l'opérateur.
- **1a, chemin multi-rangées** : premier jeton à 8k ≤ 12 s, décodage pas plus lent, porte 4 inchangée, N jetons en un appel identiques à N appels. Arrêt proposé au-delà de 1,5 ms par jeton de prompt.
- **1b, levée des murs** : carte égale à l'hôte au bit à 2 560, 9 728, 17 408 et 25 600, plus un test synthétique à la largeur du 32B ; décodage pas plus lent.
- **1c, route GEMM** : perplexité appariée du préremplissage dans ±0,12 % sur fenêtres neuves, porte 4 revérifiée, VRAM transitoire déclarée, premier jeton à 8k ≤ 4 s. Cette dernière porte est à risque : l'estimation donne 3,5 à 4,4 s avec l'attention actuelle.
- **2, route P, étape A** : au moins 8 % de bout en bout sur la config groupée. 3 à 8 % est un point de courbe ; moins de 3 %, ou un jeton changé que le repli candle n'efface pas, propose l'arrêt. **Route D** : garder chaque fusion qui montre un gain apparié de plus de 1 % et l'identité.
- **3a, attention** : les deux intervalles de perplexité dans la bande, +15 % à 8k sur l'étage 2, contexte court pas plus lent. Arrêt proposé si η reste sous 0,40 après une révision, ou si le contrôle échoue deux fois.
- **4E** : au moins 8 % sur l'étage 3, même texte. Arrêt proposé sous 3 %.
- **4K** : T8 de la sémantique int16 ; t(8)/t(1) ≤ 1,3 en place ; k = 1 pas plus lent que le noyau servi.
- **5** : sortie spéculative identique au bit ; médiane ≥ ×1,25 sur code et extraction, ≥ ×1,0 en chat ; sinon, brouillon à zéro octet sur l'extraction seule.

Cumul payé à ce jour : au moins 248,84 $ (*calculé* : somme des 232 montants de `docs/data/jobs.csv` ; 5 des 237 lignes, des 2026-08-05 et 08-06, n'ont pas de montant). Les postes 0.1 à 0.6 coûtent ensemble environ 4 à 5,5 $, option de 0.2 comprise (*estimé*). Le poste 0.7 ajoute 1,7 à 2,75 $.

## 11. Les décisions qui reviennent à l'opérateur

Chaque décision a une valeur par défaut, qui s'applique si l'opérateur ne dit rien.

1. **L'objectif.** « Quel chiffre unique le programme doit-il faire bouger, et sur quel objet : (a) tok/s de décodage du 4B, lot 1, L40S ; (b) durée d'une requête 8 192 + N, avec N et le mode raisonnement ; (c) tok/s agrégé en lot de 8, avec 60 s par document amorti ou en latence ; (d) les planchers A5 à 8k sur un objet ≥ 14B sur sm_120 ; (e) le Mac ? » Par défaut : aucun objectif retenu, aucun travail engagé ; ce rapport se lit avec (b), N = 512, sans raisonnement. Le mode raisonnement change le régime : la fiche Qwen3 y déconseille le décodage glouton, et la spéculation passerait à l'acceptation stochastique.
2. **Le plafond de dépense.** Par défaut : aucun job payant. Proposé : 8 $ pour l'étage 0, puis un plafond par étage.
3. **La politique T8.** Par défaut : aucune extension. Restent ouverts à un go les chantiers qui gardent les bits : chemin multi-rangées, levée des murs, découpe de l'attention de préremplissage si son contrôle d'identité passe, route D. Le chemin multi-rangées et la découpe relèvent du gel ROADMAP §4 sur le préremplissage et le lot (décision 9). La levée des murs n'en relève pas ; elle tombe au plus sous la ligne « The 32B point » du même §4, qui met d'abord une porte sur l'écart du 14B au 32B.
4. **La lecture de A2**, avant 4E, et peut-être avant la route P. Les conditions de réouverture sont écrites au §É7 de `proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md`. Elles sont trois : une cible de contexte plus basse, un KV q8 par défaut, ou un mécanisme qui rend la capture compatible avec un cache qui grandit. Même rouverte, A2 reste hors du cœur (ETAT §6, règle 3). Par défaut : 4E ne démarre pas.
5. **Le verdict spéculatif du 2026-08-05.** Sa condition de réouverture parle du 70B, pour un drafter 0,6B. Il n'existe que dans `docs/archive`, sans ligne dans ETAT §6 : le lever demande un go (règle 1). Le code de spéculation en a besoin. Pour G0, une mesure sans coût qui ne sert que cette piste, l'opérateur dit s'il en a besoin. Par défaut : le verdict tient.
6. **La règle 6 pour un drafter** : l'unité (b/poids noyau ou b/param), l'hypothèse KV (q8 ou f16), le partage de la tête, et le partage de marge avec le passage de down_proj en int4. Par défaut : brouillon à zéro octet seulement, et seulement si le verdict spéculatif est levé (décision 5).
7. **Cœurs tenseurs, dp4a et ldmatrix dans le périmètre.** Par défaut : la brique ne démarre pas.
8. **Les réglages gelés** pour des CTA persistants : 512 threads, tables en mémoire partagée, décodage gelé jusqu'au SASS. Par défaut : ils restent gelés, et la route P n'est pas ouverte.
9. **Le gel ROADMAP §4 sur le préremplissage et le lot.** Confirmer qu'un travail de préremplissage qui ne change pas le format échappe à ce gel. Par défaut : lecture littérale ; la découpe, le chemin multi-rangées (1a) et la route GEMM (1c) attendent. La levée des murs (1b) et la forme groupée au 8B et au 14B (1d) ne sont ni du préremplissage ni du lot.
10. **Le grand modèle** : 32B dense ou 27B hybride, avant de figer le jeu d'opérations de 4E. Côté débit, le 27B n'a pas de mur `rot_apply`. Sa tête MTP de 0,425 G paramètres, que ROADMAP-27B laisse tomber par défaut, serait un drafter entraîné avec le modèle. DeltaNet y ajoute environ 2 160 lancements par jeton, +7 à 11 ms (*estimé*, ROADMAP-27B) : le retrait des lancements y vaudrait plus. Par défaut : aucun encodage.
11. **Le lieu sm_120** : une mesure hors `ops/run.py`, ou le proxy `rtx-pro-6000` sous `--any-flavor` avec son identité déclarée. Par défaut : aucun run sm_120.
12. **Remplacer des opérations candle** (normes, RoPE, résidus) par nos noyaux, préalable de l'étage 2. La décision `deaa449` garde candle comme moteur auditable. Par défaut : un remplacement n'avance que s'il rend les mêmes bits que l'opération candle, vérifié au chargement.
13. **nsys dans l'image.** C'est un changement de recette, et c'était l'étape 0 de l'audit. Par défaut : non.

Sous toutes ces valeurs par défaut, aucun run ne part. Seul du code à 0 $ peut s'écrire, sans carte : le câblage Metal, l'outillage, du code de route D sans validation sur carte, et le harnais G0 si l'opérateur dit que G0 ne demande pas la levée du verdict spéculatif (décision 5). Quatre réponses lancent l'ordre (b) : le choix de l'objectif (b), un plafond, la lecture du gel ROADMAP §4, et le T8 de la route GEMM.

## 12. Ce que ce rapport n'établit pas

- Aucune mesure nouvelle. Tout chiffre au-delà d'aujourd'hui est une estimation sur des entrées corrigées par les vérificateurs.
- Le débit d'un flux Tetra persistant sur sm_89, et le coût d'une attente dépendante.
- L'identité sur carte de la norme candle émulée, et celle du noyau multi-rangées sous `--fmad=true`. Le seul test multi-rangées tourne sur l'hôte, compilé par clang++ avec `-ffp-contract=off`.
- L'identité de la découpe de l'attention de préremplissage, que cuBLAS peut arrondir autrement selon la forme.
- η d'une attention split-KV GQA sur Ada.
- La part exacte de l'attention dans le préremplissage à 8k. Les 1,9-2,9 s viennent d'un modèle carré ajusté sur trois temps mesurés.
- Le temps d'une attention de préremplissage sur cœurs tenseurs, supposé à 0,1-0,3 s à 8k.
- τ contre l'objet scellé, avec ou sans partage de notre tête int4 ; la platitude de t(k) ; le coût de l'étage Markov de DSpark dans notre moteur.
- La qualité de chaque changement T8, et la récurrence des pas où un logit bouge de 13,13.
- Le 8B et le 14B après trims, en forme groupée ou à 8k ; toute vitesse servie sur sm_120 ; la 5090.
- La qualité d'un 32B à environ 2,6-2,7 b/param ; tout chiffre du 27B hybride.
- La part hôte du pas servi, la bande passante L2 de la L40S, l'horloge soutenue sous une charge qui sature à la fois la mémoire et le calcul.
- Le calendrier, qui ne prévoit aucun pari raté.

## Annexe A. Méthode

- La vague 1 a cartographié notre moteur et la littérature récente : 8 rapports, 5 contre-vérifications de la littérature.
- La vague 2 a opposé quatre architectes rivaux : moteur, spéculation, brique noyau, contradicteur. Deux vérificateurs ont attaqué chaque plan, l'un sur la physique, l'autre sur les règles du projet. Un critique, trois enquêtes ciblées et un juge ont ensuite consolidé l'ensemble.
- Une troisième passe a revérifié 67 affirmations du juge : 46 confirmées, 20 corrigées, 1 réfutée. Chaque point contesté a reçu un second avis, pris dans le code, les journaux ou les sources.
- Quatre relecteurs ont enfin lu ce rapport : fidélité des chiffres, règles du projet, style, utilité pour l'opérateur. Leurs 127 remarques ont été traitées. Trois agents ont ensuite contrôlé les corrections, et leurs 61 remarques ont aussi été traitées.

## Annexe B. Constats annexes sur le code et les documents

Les vérificateurs ont relevé ces écarts en passant. Aucun n'a été corrigé, puisque cette session est en lecture seule.

1. `llvq-cuda/kernels/llvq_rot.cuh:46-48`, copié dans `llvq-llm/kernels/llvq_rot.metal:97-98`, factorise 25 600 en 512 × 50 et en conclut que k dépasse KMAX. En fait 25 600 = 1 024 × 25, et k = 25 ≤ 32. Sa conclusion tient pour le noyau à un bloc, muré par la mémoire partagée.
2. `llvq-cuda/src/bin/planesbench.rs:3952-3953` et le libellé de `:4696` disent que q et k fusionnent sur le fichier servi. Le code de `:3954-3956` les laisse séparés quand v est int4.
3. `llvq-cuda/src/lib.rs:239-242` dit que les sources de noyau font « quelques kilo-octets ». L'unité Tetra servie fait 10 fichiers et 101 541 octets.
4. `docs/ROADMAP.md:121` donne encore la forme étendue comme objet servi par défaut. ETAT §2 dit la forme groupée depuis le 2026-11-02.
5. `docs/ROADMAP.md:119` cite arXiv:2607.08734 comme banc d'extraction. C'est un article sur les effets de la quantification, « The Illusion of Equivalency ».
6. `docs/archive/revue-structurelle-2026-10-20.md:82-83` étiquette *mesuré* les 0,19 ms par jeton du bras dense. C'est un quotient *calculé*, sur l'objet q5, et une borne haute.
7. `docs/archive/audit-debit-2026-10-29.md:183` donne 7,5-9,6 tok/s pour les fichiers scellés sur Mac ; aucun journal ne porte cette fourchette. Le même audit dit, à `:232`, que le GEMV de magnitude va « jusqu'à 8 rangées » ; le code du même jour dit 16.
8. `docs/archive/audit-debit-2026-10-29.md` annonce la route GEMM « de 31 s vers 2 s à 8k ». Avec l'attention de préremplissage actuelle, elle s'arrêterait vers 3,5-4,4 s.

## Annexe C. Sources

Mégakernels et coût fixe :
- Hazy Research, « Look Ma, No Bubbles! », 2025-05-27. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Mirage Persistent Kernel, arXiv 2512.22219, OSDI '26. https://arxiv.org/abs/2512.22219
- ForgeMegakernel, arXiv 2609.12379, 2026-09. https://arxiv.org/abs/2609.12379
- AutoMegaKernel, arXiv 2606.09682, 2026-06. https://arxiv.org/abs/2606.09682
- AlpinDale, décodage sur RTX 5090, 2026-02. https://blog.alpindale.net/posts/5090_decode_optimization/
- llama.cpp PR #22522, PDL sur sm_120. https://github.com/ggml-org/llama.cpp/pull/22522

Noyaux bas bits et invariance au lot :
- Marlin, arXiv 2408.11743. https://arxiv.org/abs/2408.11743
- CodeGEMM, arXiv 2512.17970, NeurIPS 2025. https://arxiv.org/abs/2512.17970
- QTIP, arXiv 2406.11235, et son noyau. https://github.com/Cornell-RelaxML/qtip
- ExLlamaV3, GEMV dp4a à activations int8. https://github.com/turboderp-org/exllamav3/commit/bd1f519
- llama.cpp MMVQ et MMQ, PR #8495. https://github.com/ggml-org/llama.cpp/pull/8495
- Thinking Machines, « Defeating Nondeterminism in LLM Inference », 2025-09-10. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- magnitude, moteur de référence de l'audit. https://github.com/magnitudedev/magnitude

Spéculation :
- DeepSpec (DeepSeek). https://github.com/deepseek-ai/DeepSpec
- DFlash, arXiv 2602.06036, et `z-lab/Qwen3-4B-DFlash-b16`. https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16
- EAGLE-3, arXiv 2503.01840. https://arxiv.org/abs/2503.01840
- TensorRT-Edge-LLM, bancs de performance et guide des plugins. https://nvidia.github.io/TensorRT-Edge-LLM/0.8.0/user_guide/performance/performance-benchmarks.html
- « Speculative Decoding: Performance or Illusion? », arXiv 2601.11580. https://arxiv.org/abs/2601.11580
- SuffixDecoding, arXiv 2411.04975, NeurIPS 2025. https://arxiv.org/abs/2411.04975
- Token Recycling, arXiv 2408.08696, ACL 2025. https://arxiv.org/abs/2408.08696
- Lookahead decoding, arXiv 2402.02057. https://arxiv.org/abs/2402.02057

Attention et modèle :
- FlashInfer, billet de lancement et arXiv 2501.01005. https://flashinfer.ai/2024/02/02/introduce-flashinfer.html
- Fiche Qwen3-4B. https://huggingface.co/Qwen/Qwen3-4B

## Annexe D. Fichiers de travail

Tous sous `/private/tmp/claude-501/-Users-pjmalandrino-Documents-Pro-workspace-poc-llvq/35249513-f318-46c7-b902-3394594cb247/scratchpad/` :
- `wave1/` : les 8 rapports de la vague 1 et les 5 contre-vérifications ;
- `wave2-brief.md` : le dossier commun de la vague 2 ;
- `wave2/` : les 4 plans, leurs 8 vérifications, le critique, les 3 enquêtes et le rapport du juge ;
- `wave2-judge/consolidated.py` : l'arithmétique du juge ;
- `wave3-verify/` : la contre-vérification des 67 affirmations, `digest.md` en résumé ;
- `wave3-report/requests.py` : les temps de requête recalculés ;
- `wave4-review/` : les 127 remarques des quatre relecteurs.
