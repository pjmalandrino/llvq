# Pré-enregistrement B3 — re-scellement du 8B depuis le bucket (2026-08-18)

Écrit et commité **avant** le lancement du job. Item B3 du plan TACO
([`docs/plan-taco-2026-08-18.md`](../docs/plan-taco-2026-08-18.md)), décision
D2 arbitrée par l'opérateur le 2026-08-18 (« re-sceller »), go de dépense
donné le même jour.

## 1. Contexte, et ce que les canaux de rétention ont rendu

Le 8B **scellé** est perdu (machine et bucket vérifiés le 2026-08-16). Ce que
l'inventaire du 2026-08-18 établit :

- La version **projections-seules** survit au bucket :
  `Pier-Jean/jobs-artifacts/qwen3-8b-c12-77e76284/qwen3-8b-c12.llvq`,
  **1 822 874 562 octets**, mtime 2026-08-07 20:26 — cohérent avec le
  « 1,823 Go » de CLAUDE.md §6.
- `hf jobs inspect 6a76008a3e1f34a7e32bd74c` rend la **commande exacte** du
  run de quantification d'origine :
  `smoke 64 2048 12 4096 cuda nogs leech1c12 1000000 rot` avec
  `LLVQ_CALIB=c4`, `LLVQ_MODEL=Qwen/Qwen3-8B`, `LLVQ_THREADS=23`. **La graine
  est 1000000** : la branche « graine irrécupérable » que ce préreg devait
  trancher est **sans objet** — les projections du bucket SONT celles du run
  d'origine, il n'y a aucun re-tirage de calibration.

Le scellement ne quantifie rien : `bin/seal` complète le fichier avec les
tenseurs que le quantifieur n'a pas touchés (complément lu du checkpoint,
f16), config et tokenizer. Recette identique au job `seal-14b`
(`6a7960eeda2af92a634f0d5e`, cpu-xl, 24 min, 0,40 $).

## 2. Le job, verbatim

```
hf jobs run --flavor cpu-xl --timeout 45m -d \
  -v hf://buckets/Pier-Jean/jobs-artifacts:/out \
  hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  bash -lc 'set -euo pipefail
export LLVQ_MODEL=Qwen/Qwen3-8B
seal /out/qwen3-8b-c12-77e76284/qwen3-8b-c12.llvq /out/qwen3-8b-c12-77e76284/qwen3-8b-llvq.bin
sha256sum /out/qwen3-8b-c12-77e76284/qwen3-8b-llvq.bin
ls -la /out/qwen3-8b-c12-77e76284/'
```

**Coût annoncé avant lancement : ≤ 0,80 $** (le 14B, plus gros sur tous les
axes, a coûté 0,40 $ ; timeout à 45 min = pire cas ~0,75 $ au tarif cpu-xl
observé). Cumul rapporté après, au registre `docs/data/jobs.csv`.

## 3. Critères, posés avant la première milliseconde

- **C1 — le scellement aboutit et rend la taille du journal.** `seal`
  imprime son décompte ; le total sur disque doit tomber dans
  **[4,25 ; 4,40] Go** — le journal du 2026-08-08 donne « 4,32 Go scellé »
  pour ce modèle à embedding f16. Hors bande : échec, ne pas publier, ouvrir
  l'investigation.
- **C2 — `rtbits` reproduit le journal du 08-09** (vérification LOCALE,
  après téléchargement du fichier re-scellé — `rtbits` n'est pas dans
  l'image) : b/param modèle entier `Planes14` + embedding q8 = **5,322 au
  millième**, et `params_total` = **8 190 735 360 exactement**
  ([`docs/mesures/rtbits-planes-8b-2026-08-09.txt`](../docs/mesures/rtbits-planes-8b-2026-08-09.txt)).
  Hors bande : le fichier re-scellé N'EST PAS l'objet des campagnes — ne pas
  le substituer en silence.
- **C3 — l'empreinte de codebook est celle du build** : le fichier re-scellé
  s'ouvre avec `codebook_fingerprint = 0x338f_420f_1186_6319` (vérifié par
  `read_header` à toute ouverture locale — le refus est mécanique).

## 4. Réserves déclarées d'avance

- **Le re-scellé ne peut pas être comparé à l'octet à l'original** :
  l'original est perdu, aucun sha n'en a survécu. C2 est une égalité de
  *grandeurs dérivées*, pas d'octets — c'est le maximum que la perte permet,
  et c'est dit ici plutôt qu'après.
- **Le checkpoint amont peut avoir bougé depuis le 08-07** (le fetch
  d'origine était sur `main`, non épinglé — même classe de risque que le
  drift C4 documenté au README). Si `Qwen/Qwen3-8B` a changé ses tenseurs
  non quantifiés, C1/C2 peuvent passer (tailles identiques) avec un contenu
  différent. Le filet est en aval : le bras B2-8B (`fusedrun` re-scellé
  contre dense, tokens gloutons comparés) et, au besoin, une ppl 12 fenêtres
  contre le journal de campagne. Déclaré maintenant pour ne pas être
  découvert après.
- **Le format écrit sera LVQ4** (l'original était antérieur au champ
  d'empreinte) : version différente, enveloppe différente, records de
  matrices identiques par construction — la comparabilité des campagnes
  passe par C2 et par B2, pas par le magic.

## 5. Sorties

Journal : `docs/mesures/b3-8b-reseal-2026-08-18.txt` (sortie brute du job +
vérification rtbits locale). Registre : ligne ajoutée à
`docs/data/jobs.csv`. Le fichier re-scellé reste au bucket ET une copie
locale est rapatriée (le 8B n'a aujourd'hui aucune copie locale).
