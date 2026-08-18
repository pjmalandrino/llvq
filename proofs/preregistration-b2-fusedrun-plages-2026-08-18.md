# Pré-enregistrement B2 — débits bout-en-bout à plages, trois tailles (2026-08-18)

Écrit et commité **avant** tout lancement. Item B2 du plan TACO
([`docs/plan-taco-2026-08-18.md`](../docs/plan-taco-2026-08-18.md)) ; go de
dépense opérateur du 2026-08-18.

## 1. Ce que ce lot corrige, et ce qu'il ne décide pas

Tous les tok/s bout-en-bout publiés sont des **points uniques** — en
contradiction avec la règle « une plage, pas un point » (CLAUDE.md §7, r. 2)
que les bancs de noyau respectent. `fusedrun` porte depuis le commit
`a6817e6` le protocole des bancs : **1 génération jetée + 5 chronométrées par
bras, médiane + min–max**, tokens vérifiés identiques entre rounds, et le
rapport de vitesse étiqueté **quotient de deux médianes** (les rounds des
deux bras ne coexistent jamais) avec son enveloppe conservatrice.

Ce lot **mesure**, il ne gate rien : aucun seuil de kill. Il remplace des
points par des distributions et produit **une cellule neuve** — le rapport
« à tête identique » du 14B (`LLVQ_EMBED=f16`), jamais mesuré, non dérivable
du profil fencé (écart 71 % — journal du 08-17).

## 2. Les trois jobs, verbatim (image ≥ commit `1a585d0`)

Tous : `l40sx1`, `--timeout 40m`, bucket `Pier-Jean/jobs-artifacts` monté sur
`/out`, sorties tee vers `/out/b2-plages-2026-08-18/`. 128 tokens, le compte
des campagnes.

**B2-4B** (artefact HF monté en lecture seule) :
```
-v hf://Pier-Jean/Qwen3-4B-LLVQ-2bit:/model
LLVQ_EMBED=q8 fusedrun /model/qwen3-4b-llvq.bin 128   # le servi
fusedrun /model/qwen3-4b-llvq.bin 128                  # embedding f16
```

**B2-14B** (artefact du bucket) :
```
LLVQ_EMBED=q8 fusedrun /out/qwen3-14b-c12-3f21abde/qwen3-14b-llvq.bin 128   # le servi (rejoue le 08-17)
fusedrun /out/qwen3-14b-c12-3f21abde/qwen3-14b-llvq.bin 128                  # ⭐ tête identique — la cellule manquante
```

**B2-8B** (artefact re-scellé par B3 — conditionnel : ne se lance que si B3
rend C1 vert, et après la vérification C2 locale ou, à défaut, avec la
réserve C2 déclarée dans le journal) :
```
fusedrun /out/qwen3-8b-c12-77e76284/qwen3-8b-llvq.bin 128                    # embedding f16
LLVQ_EMBED=q8 fusedrun /out/qwen3-8b-c12-77e76284/qwen3-8b-llvq.bin 128     # le q8
```

**Coût annoncé avant lancement : ≤ 4 $ pour les trois jobs** (téléchargements
dominants ; chaque invocation ajoute ~6 générations de 128 tokens par bras).
Cumul rapporté après, au registre.

## 3. Ce qui se publie, posé d'avance

- Par taille et par bras : **médiane [min–max] sur 5 rounds**. Le × est le
  quotient des médianes avec l'enveloppe `[lo_f/hi_d ; hi_f/lo_d]`, étiqueté
  comme dans le binaire. Aucune troisième décimale.
- Les anciens points uniques (88,4 · 48,7 · 69,3 · 34,4 · 42,9 · 43,5 · 26,5
  · 17,0 tok/s) deviennent des **ancrages de comparaison**, pas des critères.

## 4. Anomalies, définies avant la mesure

Chacune suspend la publication du chiffre concerné et ouvre une
investigation (rien ne se « corrige » en silence) :

- **A1** — divergence de tokens entre bras au token ≤ 5 (une divergence
  tardive de tie-break est attendue et se rapporte avec sa position).
- **A2** — tokens non identiques ENTRE rounds d'un même bras (le binaire
  l'imprime) : le décodage glouton doit être déterministe à poids fixés.
- **A3** — médiane hors **±10 %** de l'ancrage historique correspondant
  (l'écart 88,4/86,9 inter-journaux est ~2 % : 10 % le couvre largement).
  La cellule 14B tête-identique n'a PAS d'ancrage : elle est la mesure.
- **A4** — plage (max−min)/médiane > 10 % sur un bras : dispersion à
  expliquer avant de publier la médiane.

## 5. Sorties

Journal : `docs/mesures/b2-fusedrun-plages-2026-08-18.txt` (sorties brutes
des trois jobs). Registre : trois lignes à `docs/data/jobs.csv`. Ces mesures
remplacent les points uniques dans `paper/` au gel des tables (B4), avec la
rétractation de forme qui va avec dans les documents vivants.
