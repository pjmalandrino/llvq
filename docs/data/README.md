# docs/data — les données propres des mesures (2026-08-05 → 07)

Fichiers CSV prêts pour tracer, décimales en points, unités dans les
en-têtes. Chaque valeur provient d'un fichier de `docs/mesures/` (colonne
source ou README du fichier) ; rien n'est lissé.

⚠️ **Une seule colonne est DÉRIVÉE, et il faut la nommer** :
`echelle-formats.csv::pct_byte_bound` n'existe dans aucun journal. C'est
`round(gbps / 661 × 100)`, où 661 Go/s est le bras FP16 du même run —
autrement dit la fraction de sa borne d'octets qu'un noyau convertit en
temps. Elle est stable au choix de round : recalculée sur les **médianes**
plutôt que sur les minima du banc, elle rend les mêmes entiers
(100/65/65/54/30/40/88). `paper/scripts/check_tables.py` la recalcule à
chaque `make` et refuse le build si le CSV et le tableau du papier
divergent.

| fichier | contenu | source |
|---|---|---|
| `campagne-finale.csv` | le tableau 4 bras × 5 facteurs (disque, VRAM, vitesse, ppl, MMLU) | a4-campagne + campagne-finale-bras4 |
| `echelle-formats.csv` | les 7 bras au banc (b/poids **noyau**, ms, Go/s, % de la borne d'octets, ratio vs FP16 avec plage) | golay70-v2-sept-bras (le run à 7 bras, phase 2) |
| `phases.csv` | le temps par phase d'un token, 4 profils (fencé — attribution, pas total) | phases-2026-08-07 |
| `progression.csv` | l'arc de la semaine : VRAM/débit/b-param à chaque étape | mini, a1, planes14-fusedrun, nuit |
| `jobs.csv` | chaque job GPU : id, durée, coût, ce qu'il a mesuré | moniteur ops/run.py |

🚨 **`jobs.csv` couvre TROIS campagnes, et la somme de la colonne n'est pas
le chiffre du papier.** Le total cité par le papier est
**19,82 + 2,33 = 22,15 $** :

| campagne | lignes | somme | dans le papier ? |
|---|---|---|---|
| papier 4B + 8B | jusqu'au 2026-08-08 inclus | 19,82 $ | ✅ |
| **kernel** (bancs 5, 6 et 7 bras) | marquées `[kernel]` | **2,33 $** | ✅ **depuis le lot D (2026-08-11)** |
| 14B | marquées `[14B]` | — | ❌ pas encore |

Le total vit dans **quatre sites**, à déplacer ensemble : `paper/main.tex`
(abstract), `sections/intro.tex` (dernier §), `sections/evaluation.tex`
(« Cost of evidence ») et `sections/conclusion.tex`.

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
