# Écarts au pré-enregistrement du corpus de calibration

> Le pré-enregistrement
> [`preregistration-dclm-2026-09-07.md`](preregistration-dclm-2026-09-07.md)
> (sha256 `9e10776a…`) est tamponné. Il ne s'édite pas.

Écrit le 2026-09-07, **avant le lancement du bras**. Quatre écarts, dont deux
touchent une phrase du §3 et un touche l'enveloppe de coût.

## É1 — La révision n'est pas épinglée, et ne peut pas l'être

**Ce que le préreg dit** (§3) : « Sa révision est épinglée comme `corpus.rs`
l'exige. »

**Ce qui est vrai** : le shard est lu à `main`, une branche.
`LLVQ_DATASET_REV` est une variable unique pour les quatre dépôts de datasets
du harnais, et un sha n'existe que dans un dépôt. `bin/smoke` lit
`Salesforce/wikitext` pour sa perplexité de référence **avant** le bras de
calibration (`llvq-llm/src/bin/smoke.rs:1057`), donc poser le sha DCLM tue le
run sur un 404 avant la première hessienne. Le mécanisme se lit dans le code,
il n'est pas déduit : `dataset_revision()` est appelée par `hf_dataset_file`
pour les quatre dépôts sans distinction (`llvq-llm/src/corpus.rs:105-118`).

Le dépôt connaissait déjà le trou et l'écrivait pour trois dépôts
(`ARTIFACT-EVALUATION.md`, corrigé en quatre le 2026-09-07). Ce qui change
ici : wikitext et MMLU sont des jeux de référence gelés, où la branche suffit
en pratique ; DCLM-edu est un corpus de pré-entraînement récent, et le texte de
calibration **est** l'expérience. Un re-filtrage en amont changerait les
hessiennes sans changer une ligne de commande.

**Ce qui est fait à la place** : la révision effectivement lue est enregistrée.
`BoundedRead::revision` porte le sha que `hf_hub` a résolu, et le journal du run
l'imprime dans la ligne de lecture bornée. La figure n'est pas reproductible,
elle est auditable après coup. Le sha lu sur le Mac le 2026-09-07 est
`dbad8ad71224482740cd9c9d353591adbf62fe04` : la ligne
`bounded read: 1061885 chars for 1048576 asked, 162 of 776000 rows, 1 of 1 row
groups, revision dbad8ad7…` imprimée par le test `the_dclm_shard_is_read_bounded`
(*mesuré*). C'est un fait daté, pas un épinglage, et le sha du conteneur peut
différer.

**Le témoin V32 est dans le même régime** : `allenai/c4` a été lu à `main` lui
aussi. La comparaison à une variable du §3 tient donc entièrement. C'est la
promesse d'épinglage qui tombe, pas le banc.

Un vrai pin demanderait une révision par dépôt dans `hf_dataset_file`, contre
ce que la doc de `corpus.rs` argumente aujourd'hui. Décision d'opérateur, hors
de ce chantier.

## É2 — Le shard descend en entier : 2,91 Go d'entrée non annoncés

**Ce que le préreg dit** : coût ~14 $, plafond 200 min, dimensionné sur la forme
de V32.

**Ce qui est vrai** : la borne écrite est une borne de mémoire et de décodage,
pas une borne d'octets rapatriés. `hf_hub` télécharge les fichiers entiers.

| entrée | octets |
|---|---|
| shard DCLM `data/000_00000.parquet` | 2 905 491 151 |
| shard C4 de calibration lu par V32 | 40 675 053 |

Facteur 71 (*mesuré*, cache HF du Mac, 2026-09-07). `ops/run.py` ne monte aucun
volume de dataset, donc le conteneur paie ce téléchargement et doit avoir le
disque pour lui, en plus du 4B et de l'artefact. V32 est mort en `Job timeout` à
148 min contre un plafond de 120 (`docs/data/jobs.csv`, 2026-09-07), sur une
estimation fausse d'un facteur 1,7 ; l'enveloppe de ce bras hérite de cette
forme.

**Ce qui est fait** : la docstring de `parquet_text_bounded` dit maintenant
explicitement que la borne porte sur la mémoire et le décodage, pas sur les
octets rapatriés. **Ce qui reste à l'opérateur** : vérifier le disque de la
flavor et accepter l'écart d'entrée avant le go. Aucun chiffre du §5 n'en
dépend.

## É3 — Le nombre de tokens du shard est calculé, pas mesuré

**Ce que le préreg dit** (§3) : « environ 1 169 M tokens (*mesuré*,
2026-09-07) : 280 fois le besoin. »

**Ce qui est vrai** : ce nombre est 4 678 M caractères divisés par 4
caractères/token **supposés**. Mesuré au tokenizer Qwen3-4B sur les 25,19 Mo
réellement lus : **4,5356 octets/token** sur 5 552 980 tokens (*mesuré*,
2026-09-07). Le shard porte donc ~1 031 M tokens, et le facteur au besoin de
4 194 304 tokens est **246×**, pas 280×.

Sans conséquence sur le bras : 246 reste très au-dessus de 1. C'est la règle 8
qui est en cause, un nombre calculé publié comme mesuré.

**Ce qui est fait** : `corpus.rs` compare désormais des caractères à des
caractères, ce que le code fait réellement : 4,68 G contre un budget de
25,2 M, soit 186×. Il donne le ratio octets/token à part, avec sa mesure. La
phrase du préreg ne s'édite pas.

## É4 — Le corpus emporte une seconde variable mesurable : la densité de frontières de documents

**Ce que le préreg dit** (§2) : tout ΔC positif serait de l'alignement de
domaine. Le §7 liste ce que le banc ne peut pas être, sans mentionner ceci.

**Ce qui est vrai**, mesuré au budget de caractères du bras (25,17 Mo de part et
d'autre, *mesuré*, 2026-09-07) :

| corpus | documents lus | octets/document, moyenne | médiane |
|---|---|---|---|
| DCLM-edu | 3 968 | 6 347 | 3 406 |
| C4, shard de calibration | 11 686 | 2 154 | 1 163 |

Une fenêtre de 2 048 tokens pèse ~9,3 ko à 4,54 o/token. Elle enjambe donc
**~1,5 document en DCLM contre ~4,3 en C4** : trois fois moins de ruptures de
sujet à l'intérieur d'une fenêtre de calibration. Le découpage lui-même est
`llvq-llm/src/bin/smoke.rs:607-609` et `1155-1160`.

Une hessienne est une statistique du texte de calibration. Des fenêtres qui
recollent quatre documents sans rapport ne produisent pas les mêmes directions
d'activation que des fenêtres cohérentes, indépendamment du domaine. Ce n'est
pas une violation de la variable unique du §3, c'est dedans : le corpus est une
variable, et elle en contient deux.

**Réserve de lecture au §5** : un ΔC ≥ +2,0 pp ne démontrerait pas à lui seul
l'alignement de domaine du §2. La porte du §5 ne bouge pas, la conclusion
qu'elle autorise se lit « le corpus porte une part réelle de l'écart », sans
attribuer cette part au domaine.

Le contrôle qui séparerait les deux est gratuit en conception : un bras C4 dont
les documents courts sont filtrés à ≥ 3 ko avant concaténation, à volume de
tokens constant. Il n'est pas lancé et ne le sera pas sans go.

---

*Les écarts É1 à É4 ci-dessus ont été écrits avant le lancement, par la revue du chantier de
code. Les suivants sont écrits après la mesure. Un heredoc de l'auteur les a écrasés le
2026-09-08 à 03:15 ; ils sont restaurés depuis `7f48c8a^`.*

## É5 — La règle du §5 est posée sur un point là où l'intervalle décide

`ΔC = −0,55` tombe sous le seuil de −0,5 de la quatrième ligne, « le corpus
nuit ». **En substance, non** : l'IC95 va de −3,21 à +2,13, le contrôle non
pondéré est **de signe opposé** (+1,05), et McNemar donne 0,347. Il n'y a aucun
effet mesurable, dans aucun sens, et la lecture juste est la troisième ligne —
le corpus n'explique rien.

C'est le même défaut que celui écrit en É1 du préreg M0 : une porte posée sur
une estimation ponctuelle alors que l'intervalle est ce qui tranche. **Deux fois
en deux jours.** Les prochains préregs poseront leurs seuils sur les bornes de
l'IC, pas sur le point.

## É6 — La prédiction signée était juste sur la perplexité et fausse sur le MMLU

Le §6 prédit `ΔC` entre 0 et +3 avec un centre à +1,2, et une perplexité
dégradée de 2 à 10 %. Mesuré : **−0,55** et **+2,12 %**.

Le motif du §2 se décompose donc en deux moitiés, et une seule tient. Le
déplacement de domaine **est réel et mesurable** — la hessienne a bougé, la
perplexité de texte web s'est dégradée exactement comme prédit. Ce qui ne tient
pas est le second maillon : rien ne dit que cette hessienne-là serve mieux le
MMLU, et la mesure dit qu'elle ne le sert pas.

## É7 — Le préreg annonçait ~14 $ et le job en a coûté 13,50

Enveloppe tenue. Le plafond de timeout à 200 min, posé après qu'un plafond de
120 eut tué V32 à la fin de son travail, n'a pas eu à servir : 147 min.
