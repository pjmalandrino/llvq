# docs/data — les données propres des mesures (2026-08-05 → 07)

Fichiers CSV prêts pour tracer, décimales en points, unités dans les
en-têtes. Chaque valeur provient d'un fichier de `docs/mesures/` (colonne
source ou README du fichier) ; rien n'est recalculé ni lissé.

| fichier | contenu | source |
|---|---|---|
| `campagne-finale.csv` | le tableau 4 bras × 5 facteurs (disque, VRAM, vitesse, ppl, MMLU) | a4-campagne + campagne-finale-bras4 |
| `echelle-formats.csv` | les 5 layouts au banc (b/poids, ms, Go/s, ratio vs FP16 avec plage) | e2-golay70-bench (le run à 5 bras) |
| `phases.csv` | le temps par phase d'un token, 4 profils (fencé — attribution, pas total) | phases-2026-08-07 |
| `progression.csv` | l'arc de la semaine : VRAM/débit/b-param à chaque étape | mini, a1, planes14-fusedrun, nuit |
| `jobs.csv` | chaque job GPU : id, durée, coût, ce qu'il a mesuré | moniteur ops/run.py |

🚨 **`jobs.csv` couvre désormais DEUX campagnes, et leur somme n'est pas le
chiffre du papier.** Les lignes datées **jusqu'au 2026-08-08 inclus** sont la
campagne du papier — elles somment à **19,82 $**, et c'est ce total que
`paper/main.tex:70`, `intro.tex:79`, `evaluation.tex:193` et
`conclusion.tex:26` citent, à juste titre : le papier écrit « the full campaign
behind this paper … both models », et les deux modèles sont le 4B et le 8B.
Les lignes marquées **`[14B]`** appartiennent à la campagne Qwen3-14B, qui
**n'est pas dans le papier à ce jour**.

⚠️ Donc : ne jamais resommer la colonne entière pour obtenir le chiffre du
papier, et si le 14B entre au papier comme troisième point d'échelle, déplacer
les **quatre** sites en même temps. Le total n'est régénéré par aucun script —
`paper/scripts/make_figures.py` n'ouvre jamais ce fichier.

Conventions : VRAM en b/param = modèle entier embedding compris (jamais
payload seul — cf. errata-rapport-lot-a) ; les ratios vitesse = médiane des
rapports formés round par round, avec plage ; MMLU micro = protocole du
papier, ± = erreur d'échantillonnage seule ; les phases sont bornées par
synchronisation (elles s'attribuent, leur somme ne fait pas un tok/s).
