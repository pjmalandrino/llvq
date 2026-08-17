# docs/data — les données propres des mesures (2026-08-05 → 07)

Fichiers CSV prêts pour tracer, décimales en points, unités dans les
en-têtes. Chaque valeur provient d'un fichier de `docs/mesures/` (colonne
source ou README du fichier) ; rien n'est lissé.

🕳️ **« Une seule colonne est DÉRIVÉE » était vrai jusqu'au 2026-08-17 ; il y
en a DEUX depuis, et les deux sont recalculées par le filet.**

1. `echelle-formats.csv::pct_byte_bound` n'existe dans aucun journal. C'est
   `round(gbps / 661 × 100)`, où 661 Go/s est le bras FP16 du même run —
   autrement dit la fraction de sa borne d'octets qu'un noyau convertit en
   temps. Elle est stable au choix de round : recalculée sur les **médianes**
   plutôt que sur les minima du banc, elle rend les mêmes entiers
   (100/65/65/54/30/40/88).
2. 🆕 `echelle-4b-8b.csv::vram_margin_vs_awq_pct` (−2,6 / −10,6 / −5,5) est
   `round((llvq2 / awq4 − 1) × 100, 1)` sur les deux `vram_bits_per_param` de
   la même taille. Aucun journal ne l'écrit sous cette forme ; c'est la marge
   que la prose cite, et elle doit suivre des deux taux ou l'un des trois est
   faux.

`paper/scripts/check_tables.py` les recalcule toutes deux à chaque `make` et
refuse le build si le CSV et le tableau du papier divergent.

🚨 **Et une hygiène qui n'était contrôlée par personne l'est depuis le
2026-08-17 : ces CSV doivent être RECTANGULAIRES.** Trois lignes ne l'étaient
pas — deux de `campagne-finale.csv`, une de `jobs.csv` — parce qu'une virgule
non échappée dans un champ libre (`notes`, `what`) le coupe : `csv.DictReader`
jette tout ce qui suit dans la clé `None`, et la note se lit tronquée en plein
milieu de phrase. **Rien n'échouait**, précisément parce qu'aucun tableau
vérifié ne lit ces colonnes — c'est ainsi que le défaut a survécu depuis
l'écriture des fichiers. Convention : dans un champ libre, on sépare les
clauses par `;` ou ` - `, **jamais par une virgule**, et
`check_tables.py::check_csv_shape` le fait respecter.

| fichier | contenu | source |
|---|---|---|
| `campagne-finale.csv` | le tableau 4 bras × 5 facteurs (disque, VRAM, vitesse, ppl, MMLU) | a4-campagne + campagne-finale-bras4 |
| `echelle-formats.csv` | les 7 bras au banc (b/poids **noyau**, ms, Go/s, % de la borne d'octets, ratio vs FP16 avec plage) | golay70-v2-sept-bras (le run à 7 bras, phase 2) |
| `phases.csv` | le temps par phase d'un token, 4 profils (fencé — attribution, pas total) | phases-2026-08-07 |
| `progression.csv` | l'arc de la semaine : VRAM/débit/b-param à chaque étape | mini, a1, planes14-fusedrun, nuit |
| `echelle-4b-8b.csv` | l'échelle des modèles, 3 bras × 3 tailles (ppl, ratio, MMLU micro, **b/param modèle entier**) | a4-campagne + campagne-finale-bras4 (4B), campagne-8b-qualite (8B), campagne-14b-qualite (14B) ; `params_total` et les colonnes `vram_*` viennent d'ailleurs — rtbits-planes-8b (4B, 8B) et rtbits-14b-2026-08-17 (14B) |
| **`mmlu-appariee.csv`** 🆕 | les **écarts APPARIÉS** de MMLU : 3 tailles × 3 paires, point + IC95 + SE + McNemar + taux de discordance | mmlupair-4b-8b-2026-08-13 (4B, 8B), campagne-14b-qualite (14B `f16 −` …) et **mmlupair-14b-2026-08-17** (14B `awq4 − llvq2`) |
| **`ppl-appariee.csv`** 🆕 | les **intervalles APPARIÉS de perplexité** : 3 tailles × 3 paires, excès en % + IC95 + ratio + moyenne et SE des différences de NLL + t | ppl-appariee-4b-2026-08-17 (4B), ppl-appariee-8b-14b-2026-08-17 (8B et 14B) |
| **`ppl-genou.csv`** 🆕 | le **ralentissement EN PERPLEXITÉ** : 2 pas × 2 références, plus les 2 tests de genou (la différence des deux pas) | ppl-appariee-4b-2026-08-17 |
| `mmlu-dumps/` | les dumps MMLU **question par question**, 3 tailles × 3 bras — la matière première des paires ci-dessus | campagnes 4B/8B/14B ; les trois fichiers 14B commités le 2026-08-17 (cf. la leçon du bucket, plus bas) |
| `jobs.csv` | chaque job GPU : id, durée, coût, ce qu'il a mesuré | moniteur ops/run.py |

### 🆕 Pourquoi DEUX fichiers de perplexité, et pas une colonne de plus (2026-08-17)

Les neuf intervalles de perplexité auraient pu s'écrire dans
`echelle-4b-8b.csv`. **Ils n'y sont pas, et c'est le même dispositif que pour
MMLU : la séparation physique.**

1. **Le grain n'est pas le même.** `echelle-4b-8b.csv` est indexé par
   *(modèle, bras)* — une ligne par artefact. Un intervalle apparié porte sur
   une *(modèle, PAIRE)* : il n'appartient à aucun des deux bras, il est la
   comparaison. Le loger dans un fichier par bras obligerait soit à le
   dupliquer des deux côtés, soit à inventer des lignes hybrides bras-paire —
   exactement la confusion que la note « aucune colonne appariée » plus bas
   existe pour empêcher.
2. **On ne peut pas soustraire par accident deux colonnes qui ne sont pas dans
   le même fichier.** C'est l'argument littéral du 🚨 sur `mmlu-appariee.csv`,
   et il vaut mot pour mot ici : soustraire deux `ppl` de `echelle-4b-8b.csv`
   ne produit pas un écart testé, et le genou est précisément le nombre qu'on
   serait tenté de fabriquer ainsi.
3. **Le genou a encore un autre grain** — *(pas, référence)*, pas
   *(modèle, paire)* — d'où le second fichier plutôt qu'une colonne
   supplémentaire ou une clé en union. `ppl-genou.csv` porte les deux pas sur
   chacune des deux références, plus les deux tests de genou, tous à la même
   forme.

⚠️ **Aucun tableau du papier ne les consomme aujourd'hui** : ces nombres y sont
en prose (§ « Perplexity gets error bars »). Le filet les vérifie quand même,
et de trois façons indépendantes — épinglage littéral sur les journaux
(`PPL_PAIRED`, `PPL_KNEE`), **dérivations internes exactes** (ratio =
exp(différence moyenne), excès = ratio − 1, t = moyenne / SE, additivité des
trois paires d'une taille, et le genou = différence des deux pas), et
**recoupement croisé** avec `echelle-4b-8b.csv`, dont `ppl_ratio_vs_f16` est
la même grandeur atteinte par une autre route. 18 mutants tués sur 18 lors de
l'écriture du garde. Un garde écrit avant le tableau est le but : le tableau
qui viendra en héritera au lieu d'en réclamer un.

🕳️ **Une colonne échappe volontairement à la dérivation** :
`ppl-appariee.csv::ratio` de la paire `llvq2_over_awq4`. Elle vaut **1,1458**
au 14B (exp de la différence moyenne, exact) alors que le quotient des deux
ratios déjà arrondis de `echelle-4b-8b.csv` rend **1,1457**. Les deux sont
justes dans leur comptabilité ; dériver celle-là imposerait un artefact
d'arrondi au lieu de vérifier un accord. Elle est épinglée et tenue par
l'identité d'additivité, pas par le quotient.

✅ **`echelle-4b-8b.csv::params_total` était VIDE sur les trois lignes 14B ;
il est rempli depuis le 2026-08-17, et la consigne « il faut le laisser vide »
qui figurait ici est LEVÉE.** Elle ne l'est pas par un assouplissement : elle
l'est parce que sa condition a été satisfaite. Elle disait « **aucun passage
de `rtbits` sur un 14B scellé n'existe dans le dépôt** » — c'était exact, et
ça ne l'est plus :
[`mesures/rtbits-14b-2026-08-17.txt`](../mesures/rtbits-14b-2026-08-17.txt)
est ce passage. L'artefact n'avait jamais été rapatrié après la campagne du
08-10 ; il dormait dans le bucket `Pier-Jean/jobs-artifacts`
(`qwen3-14b-c12-3f21abde/qwen3-14b-llvq.bin`, 6 506 354 741 o), d'où il a été
relu pour 0 $ — bande passante seule, aucun GPU. Le compte est donc désormais
**exact comme les deux autres** : 4B 4 022 468 096 · 8B 8 190 735 360 ·
**14B 14 768 307 200**, les deux premiers lus dans
[`mesures/rtbits-planes-8b-2026-08-09.txt`](../mesures/rtbits-planes-8b-2026-08-09.txt)
(l. 114 et 275), le troisième dans le journal du 08-17 — et recoupé là par une
seconde route, l'arithmétique de l'architecture (§3 du journal : les huit
entiers, dont les 163 tenseurs portés que la note de reprise avait posés
d'avance comme critère de scellement).

🚨 **Le piège que cette note signalait reste entier, et il faut le garder** :
le seul compte 14B qui circulait alors était **13 212 057 600 poids
quantifiés**
([`archive/reprise-14b-2026-08-09.md`](../archive/reprise-14b-2026-08-09.md), l. 38).
C'est le numérateur des **projections**, **pas** un total modèle entier — il
manque 1 555 824 640 d'embedding et 424 960 de normes. Le mettre dans cette
colonne aurait été exactement la confusion de dénominateurs que l'errata du
lot A qualifie de GRAVE, et il en est à 10,6 % près. La case vide était le bon
choix tant que le vrai compte manquait ; c'est le compte qui a changé, pas la
règle.

🆕 **Et la ligne mémoire du 14B, que ce trou bloquait, existe maintenant** :
`Planes14` + embedding q8 pèse **5,106 b/param modèle entier** contre
**5,404** pour `Qwen/Qwen3-14B-AWQ` (octets safetensors du dépôt officiel lus
par l'API du Hub, ÷ `params_total`) — **sous l'AWQ de 5,5 %**, comme au 4B
(−2,6 %) et au 8B (−10,6 %). ⚠️ La marge **n'est pas monotone** et ne raconte
aucune tendance : elle suit la part de l'embedding (9,7 % · 15,2 % · 10,5 %),
que l'AWQ laisse en f16 et que nous passons en q8. Détail et étiquettes de
provenance au §2 du journal.

🚨 **`echelle-4b-8b.csv` ne porte toujours AUCUNE colonne appariée, et c'est
délibéré : soustraire deux de ses `mmlu_micro_pct` ne produit pas un écart
testé.** Le piège est structurel — `awq4 − llvq2` s'y lit à vue, et c'est de là
que sortait le « 6,09 pp » du 14B (78,21 − 72,12), une différence nue. **Les
neuf écarts appariés vivent désormais dans un fichier séparé,
[`mmlu-appariee.csv`](mmlu-appariee.csv)** — 3 tailles × 3 paires, chacune avec
son point, son IC95, sa SE, son McNemar, son taux de discordance et son
journal. La séparation physique est le dispositif : on ne peut pas soustraire
par accident deux colonnes qui ne sont pas dans le même fichier.

🕳️ **Ce paragraphe disait « au 14B, seules `f16 − AWQ` et `f16 − LLVQ`
existent » — périmé le 2026-08-17, voir juste dessous.** Les trois paires 14B
existent.

⚠️ **Une divergence d'arrondi à connaître avant de croire à un écart de
0,01 pp** : `echelle-4b-8b.csv::mmlu_delta_pp` vaut −10,56 au 8B
(c'est 65,52 − 76,08, deux micros déjà arrondis) tandis que
`mmlu-appariee.csv::f16_minus_llvq2` vaut 10,57 (le Δ stratifié calculé sur
les questions, avant arrondi). **Les deux sont justes dans leur comptabilité**
et ne se corrigent pas l'un l'autre ; le papier cite 10,57, qui est le nombre
apparié.

✅ **CORRIGÉ LE 2026-08-17 — ce paragraphe disait « La paire AWQ − LLVQ
n'existe pas au 14B, et la recalculer exige de refaire la campagne MMLU 14B :
ce n'est pas une correction à 0 $ ». C'ÉTAIT FAUX, et la paire a été calculée
pour 0 $.** Elle vaut **+6,09 pp, IC95 [+3,62 ; +8,52], SE 1,25 pp, McNemar
p = 1,143e-11**
([`mesures/mmlupair-14b-2026-08-17.txt`](../mesures/mmlupair-14b-2026-08-17.txt)).
🕳️ **Le mécanisme de l'erreur mérite d'être gardé, parce qu'il est
reproductible.** La note ci-dessous concluait à la perte de « vérifié le
2026-08-16 : plus aucune trace **sur la machine** » — une recherche à UN seul
endroit. Or le job de campagne n'écrivait pas sur la machine mais dans le
**bucket monté**, qui existe précisément pour que les sorties survivent au
conteneur : les trois dumps y dormaient depuis le 2026-08-10
(`hf://buckets/Pier-Jean/jobs-artifacts/campagne-14b-qualite/`). Coût réel de
la « correction impossible » : **579 ko de bande passante**.

> ### 🧭 Règle permanente — inventorier le bucket AVANT de budgéter un re-run
>
> **Toute sortie déclarée perdue mérite un `hf buckets ls` avant qu'on chiffre
> sa reproduction.** Le bucket `hf://buckets/Pier-Jean/jobs-artifacts/`
> contient **69 fichiers et 46,7 Go**, et **personne ne l'a inventorié depuis
> sa création le 2026-08-02**. C'est le dispositif que `ops/run.py --bucket`
> existe pour alimenter (« sans `--bucket`, rien de ce que le job écrit ne
> survit au conteneur ») : une recherche « sur la machine » ne le voit pas, par
> construction.
>
> **Deux prises le même jour, pour deux budgets évités** : les trois dumps
> MMLU du 14B (579 ko, contre une campagne MMLU rebudgétée) et l'**artefact
> 14B scellé** lui-même (`qwen3-14b-c12-3f21abde/qwen3-14b-llvq.bin`,
> 6 506 354 741 o, ~9 min de bande passante — contre les **27,67 $ et 302 min**
> qu'a coûté sa quantification).
>
> ⚠️ **Et la règle n'est pas une garantie, sinon elle mentirait** : le **8B
> scellé** a été cherché aux deux endroits et il est perdu — la machine ne l'a
> pas, et le bucket n'héberge que sa version *projections seules*. `hf buckets
> ls` change ce qu'on sait, pas ce qui existe.
Les trois dumps sont **désormais commités** dans
[`mmlu-dumps/`](mmlu-dumps/) (`mmlu-14b-{f16,awq,llvq}.csv`), ce que la note
d'origine reprochait justement de n'avoir pas fait — la perte ne peut plus se
reproduire. Leur authenticité est établie avant usage : les trois micros
stratifiés rejouent 78,97 / 78,21 / 72,12, et `f16 − LLVQ` rejoue ses quatre
grandeurs publiées (+6,85 [+4,52 ; +9,12], SE 1,16, McNemar 8,666e-16).

🚨 **Conséquence : la suite « 14,45 → 7,49 → 6,09 » ne mélange PLUS deux
espèces de nombre** — les trois termes sont appariés et portent un IC. Mais
elle perd autre chose, et c'est plus gênant : **le « genou » entre 8B et 14B
n'est pas résolu SUR MMLU.** La chute de l'écart vaut 6,96 pp du 4B au 8B
(SE 1,82, p = 1e-4, **résolue**) et seulement 1,40 pp du 8B au 14B (SE 1,68,
p = 0,40, **NON résolue** — SE composées en quadrature, *calculé*). Les
phrases du dossier qui font du ralentissement un résultat — « il y a un
genou », « la décroissance ralentit » — reposent donc, **sur cette
métrique**, sur des points estimés que les barres ne séparent pas. Ce qui
reste testé : l'écart est bien plus petit à 14B qu'à 4B (8,36 pp, p ≈ 1e-5).
⚠️ Et p = 0,40 ne prouve pas l'égalité non plus : sur ce palier les données
sont muettes, pas concluantes.

> 🚨🚨 **AMENDÉ LE 2026-08-17 (soir) — ET C'EST L'AMENDEMENT LE PLUS
> IMPORTANT DE CE FICHIER.** Le paragraphe ci-dessus a été écrit ce matin, en
> toute bonne foi sur l'état d'alors, et **il reste vrai — pour MMLU.** Il
> était alors le seul verdict disponible, parce que la perplexité n'avait pas
> de barre au 4B et que le pas 4B→8B n'était donc pas testable.
> **Il l'est depuis** ([`ppl-genou.csv`](ppl-genou.csv),
> [`mesures/ppl-appariee-4b-2026-08-17.txt`](../mesures/ppl-appariee-4b-2026-08-17.txt)),
> et il répond l'inverse :
>
> | métrique | pas 4B→8B | pas 8B→14B | le ralentissement |
> |---|---|---|---|
> | **perplexité** *(apparié, 12 fenêtres, même texte aux trois tailles)* | ×0,881211 [0,856 ; 0,907] | ×0,974855 [0,959 ; 0,991] | ✅ **RÉSOLU** — pas1 − pas2 = −0,1010 [−0,1377 ; −0,0643], t = −6,06 |
> | **écart MMLU au 4 bits** *(non apparié entre tailles, SE en quadrature)* | −6,96 pp, p = 1e-4 | −1,40 pp, p = 0,40 | ❌ **NON RÉSOLU** sur le second pas |
>
> **CE N'EST PAS UNE CONTRADICTION, C'EST UNE INFORMATION** : deux métriques,
> deux verdicts. La perplexité est appariée *entre tailles* (même fenêtre,
> même texte, empreinte commune) et teste donc avec beaucoup plus de
> puissance ; MMLU compose deux campagnes indépendantes. Et les deux ne
> mesurent pas la même chose — le §3ter du dossier établit depuis le
> 2026-08-02 que le 2 bits abîme le **raisonnement** bien plus que la
> **restitution**, et c'est la restitution qu'un corpus de perplexité mesure
> surtout.
>
> 🚨 **RÈGLE DE RÉDACTION, IMPÉRATIVE : toute phrase sur le genou doit NOMMER
> SA MÉTRIQUE.** « Le genou tient » nu est faux de moitié ; « le genou ne
> tient pas » nu l'est de l'autre moitié. La forme juste : *le ralentissement
> est résolu en perplexité et ne l'est pas sur l'écart MMLU au 4 bits.*

🚨 **`jobs.csv` couvre QUATRE campagnes depuis le 2026-08-17, et la somme de
la colonne n'est plus le chiffre que le papier revendique pour lui-même.**

| campagne | lignes | somme | dans le total du papier ? |
|---|---|---|---|
| papier 4B + 8B | jusqu'au 2026-08-08 inclus | 19,82 $ | ✅ |
| **kernel** (bancs 5, 6 et 7 bras) | marquées `[kernel]` | **2,33 $** | ✅ **depuis le lot D (2026-08-11)** |
| 14B | marquées `[14B]` | 30,20 $ | ✅ |
| **`[phase 1.2]`** (rejeu MMLU apparié) | marquées `[phase 1.2]` | **1,30 $** | ✅ |
| 🆕 **`[plancher]`** (E1v sur carte + `nullk`) | marquées `[plancher]` | **1,62 $** | ❌ **et délibérément pas** |

🆕 **Le total de la colonne passe de 55,59 à 57,21 $ le 2026-08-17**, en
soldant une dette : `jobs.csv` s'arrêtait au 2026-08-13 et **manquait les deux
jobs du 08-16** — `6a814ba31f5885ae605bcb55` (llvq-e1v, l40sx1, 28 min,
0,85 $) et `6a81b2b71f5885ae605bdcc9` (llvq-nullk, l40sx1, 26 min, 0,77 $).
Les deux durées et les deux montants sont **lus dans l'en-tête de leur propre
journal** ([`e1v-cuda-2026-08-16`](../mesures/e1v-cuda-2026-08-16.txt),
[`nullk-plancher-2026-08-16`](../mesures/nullk-plancher-2026-08-16.txt)), et
recoupés par le tarif l40sx1 qu'impliquent les lignes déjà présentes
(≈ 0,030 $/min : 0,0304 et 0,0296 ici) — *calculé*, pas une seconde mesure.

⚠️ **Et le 55,59 n'a PAS bougé pour autant : c'est maintenant un
sous-total.** Aucune cellule du papier ne repose sur les deux jobs du 08-16 —
E1v et le plancher `nullk` n'y apparaissent nulle part — donc les fondre dans
« le coût de cette évidence » aurait gonflé le chiffre sans rien ajouter à ce
qu'il paie. Le papier dit désormais les deux : **57,21 $ au registre, dont
55,59 $ derrière ses propres nombres.**

🕳️ **« Le total vit dans quatre sites » était faux, et vérifié le
2026-08-17 : il en vit DEUX.** `paper/main.tex` (abstract, l. 74-75) et
`paper/sections/evaluation.tex` (« Cost of evidence »). `sections/intro.tex`
mentionne la pratique — « every claim traces to a dated GPU job with its
billed cost » — **sans citer de montant**, et `sections/conclusion.tex` n'en
cite aucun non plus. Chercher un chiffre dans les deux sites fantômes faisait
conclure à tort qu'il avait déjà été déplacé. Le total n'est régénéré par
aucun script : `paper/scripts/make_figures.py` n'ouvre jamais ce fichier et
`check_tables.py` ne vérifie pas cette phrase.

⚠️ Donc : ne jamais resommer la colonne entière, et **ne pas confondre les
deux jobs Golay70** — `e2-golay70-bench` (0,74 $, 08-07, la découverte du
résultat négatif) est dans les 19,82 ; `golay70-v2-sept-bras` (0,77 $, 08-11,
la tentative de réparation) est dans les 2,33. Le papier les additionne à
1,51 $ **en le disant**, parce qu'ils sont de deux campagnes. Le total n'est
régénéré par aucun script — `paper/scripts/make_figures.py` n'ouvre jamais ce
fichier.

Conventions : VRAM en b/param = modèle entier embedding compris (jamais
payload seul — cf. errata-rapport-lot-a) ; les ratios vitesse = médiane des
rapports formés round par round, avec plage ; MMLU micro = protocole du
papier, ± = erreur d'échantillonnage seule ; les phases sont bornées par
synchronisation (elles s'attribuent, leur somme ne fait pas un tok/s).

## Étiquettes de provenance — qui est MESURÉ, qui est CALCULÉ

La colonne `vram_bits_per_param` fait se côtoyer **deux provenances
différentes**, et rien ne le disait avant le 2026-08-17. C'est maintenant
écrit dans les `notes` de chaque ligne concernée, et résumé ici :

| grandeur | statut | comment |
|---|---|---|
| **nos** b/param (5,162 · 5,322 · 5,106) | **MESURÉ** | `rtbits` sur les octets réels du fichier scellé — l'embedding y est *modélisé* à 8,5 b/param, modèle validé le 08-09 contre un vrai fichier q8 (`q8b-e8.llvq`, porté mesuré 8,502) |
| **AWQ** b/param (5,302 · 5,956 · 5,404) | **CALCULÉ** | octets safetensors du dépôt officiel (LUS par l'API du Hub) ÷ `params_total`. ⚠️ Le 8B a été obtenu par la route **taux** (5,956) ; la route **octets** rend 5,9566. Indiscernables au millième, deux routes quand même |
| disques **f16** (8,04 · 16,382 Go) | **CALCULÉ** | 2 octets × paramètres. Aucun fichier f16 n'est pesé |
| disques **AWQ** (2,67 · 6,099 Go) | **LU** au dépôt | 2 666 027 672 o et 6 098 581 864 o |
| nos disques (1,77 · 1,41 · 4,324 · 3,157 Go) | **LU** sur le fichier | `qwen3-4b-llvq.bin`, `q4b-e8.llvq`, `qwen3-8b-llvq.bin`, `q8b-e8.llvq` |
| `params_total` | **MESURÉ** puis **RECOUPÉ** | lu dans le fichier scellé ; le 14B est en plus recoupé par l'arithmétique de l'architecture (rtbits-14b §3, huit entiers) |

🕳️ **Deux valeurs qui ressemblent à des mesures et n'en sont pas.**

1. **Le `15,999` de `tableau-8b.csv` est un ARTEFACT D'ARRONDI**, pas une
   mesure : c'est `16,38 Go affichés × 8 ÷ params_total`. La construction vaut
   **16,000 exactement** (deux octets par paramètre), et c'est ce que porte
   `echelle-4b-8b.csv`. `check_tables.py` épingle le 16,000 pour que le
   15,999 ne migre pas.
2. **Les `5,323` et `6,461` de `tableau-8b.csv`** ne sont pas non plus des
   verdicts `rtbits` : ce sont les **rapports VRAM du moteur** (5,45 et
   6,62 Go carte ÷ `params_total`). Ils recoupent `rtbits` à **0,001** — donc
   ils valent comme **troisième instrument**, ce qui est précieux — mais le
   chiffre à publier est celui de `rtbits` (5,322 et 6,462), et c'est lui que
   porte `echelle-4b-8b.csv`. ⚠️ Même mécanisme que le `5,15` du 4B, qui était
   la division de l'affichage carte « 2,60 Go » et a été retiré ; la
   différence est qu'ici les deux routes tombent d'accord.

✅ **Contrôle passé le 2026-08-17 : aucune surface vivante n'oppose plus un
b/poids de projections à un b/param de modèle entier** — l'interdit posé
d'avance (`docs/archive/portage-noyau-cuda.md:31`) et enfreint une fois
(errata du lot A, « erreur GRAVE »). Une seule violation restait,
`LAUNCH_ME.md` (« 5,51 b/poids, soit plus que les 4,50 d'un 4 bits
ordinaire ») : corrigée sur place, avec la mention de ce qu'elle disait.
Les autres occurrences de « 5,51 » et « 4,50 » sont soit des citations de
l'interdit lui-même (CLAUDE.md, cheatsheet, note-produit, HISTORIQUE), soit
de l'archive, soit — `README.md`, `paper/sections/intro.tex` — des
comparaisons **licites** de b/poids à b/poids (5,510 contre les 4,179 du
noyau AWQ, `echelle-formats.csv`). `docs/fiche-4b.md:392` compare bien
4,50 à 3,727/4,034, qui sont des b/param **modèle entier** (vérifié : ils se
redérivent de sa table 70B).
