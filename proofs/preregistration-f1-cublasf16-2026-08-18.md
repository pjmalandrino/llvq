# Pré-enregistrement F1 — le dénominateur cuBLAS (2026-08-18)

Écrit et commité **avant** le lancement. Item F1 du plan TACO ; go de dépense
opérateur du 2026-08-18.

## 1. La question, et pourquoi elle prime

Tous les × publiés du banc divisent par un témoin **maison** (`tv_f16`,
« LLVQ reconstruit arrondi en binary16 »), jamais confronté à cuBLAS. Le bras
`cublasf16` — enregistré `HAS_KERNEL=false` depuis P4 §2.5 — est écrit au
commit `1a585d0` : un `cublasGemmEx` (16F/16F/16F, compute 32F, n = 1) sur le
**même** tampon `w16` que le témoin, vérifié contre **sa propre** référence
f64 (poids f16 × entrée binary16, la somme que GemmEx calcule), tolérance
binary16 comme AWQ et pour la même raison.

## 2. Le job, verbatim (image ≥ commit `1a585d0`)

`l40sx1`, `--timeout 30m`, artefact 4B monté en lecture seule
(`-v hf://Pier-Jean/Qwen3-4B-LLVQ-2bit:/model`), bucket sur `/out` :

```
LLVQ_BENCH_ARMS="fp16,cublasf16;fp16,cublasf16,planes14,awq,nullk" \
  planesbench /model/qwen3-4b-llvq.bin
```

Deux phases (la seconde contient la première, contrat du parseur) : la
première isole le duel témoin/cuBLAS, la seconde le replace dans le contexte
des bras publiés et rend le Δ_contrôle inter-phases que le banc imprime.
Protocole inchangé : 7 rounds dont 2 jetés, tous les bras chaque round,
rapports formés round par round, vérification f64 ligne à ligne avant tout
chronométrage.

**Coût annoncé avant lancement : ≤ 0,60 $** (le banc 7 bras du 08-11 a coûté
0,74 $ pour plus de bras ; timeout 30 min = pire cas ~0,90 $ — cap réel du
lot). Cumul rapporté après.

## 3. Les bandes de conséquence, posées avant la mesure

Soit `r = médiane(t_tv_f16 / t_cublasf16)` formé round par round dans la
phase 1 (les deux bras coexistent : le rapport par round est licite ici).

- **r ≤ 1,05** — le témoin maison est au niveau : les × publiés tiennent,
  et le papier gagne un paragraphe qui le dit avec la mesure.
- **r > 1,05** — le témoin sous-performe cuBLAS de plus de 5 % : **tous les
  « vs FP16 » publiés sont majorés** et se ré-ancrent sur cuBLAS dans le
  papier (B4). Pas un kill — une requalification obligatoire, décidée ici,
  pas après avoir vu le chiffre.
- **r < 0,95** — le témoin BAT cuBLAS de plus de 5 % : résultat en soi
  (un matvec spécialisé bat la GEMM générale à n = 1), à publier comme tel —
  et vérifier d'abord que la comparaison est bien à octets lus identiques.
- La **vérification f64 doit passer** (seuil binary16 1e-3) : un bras cuBLAS
  qui ne reproduit pas sa référence invalide le job, pas la référence.

## 4. Réserves déclarées d'avance

- cuBLAS choisit son algorithme (`CUBLAS_GEMM_DEFAULT`) : le chiffre est
  « cuBLAS tel que candle l'appelle », pas « le meilleur GemmEx possible » —
  même statut que la réserve M = 1 de la mesure vLLM.
- Le bras sort en binary16 quand `tv_f16` sort en f32 : différence de format
  déclarée (comme AWQ), au bénéfice du bras cuBLAS (moitié des octets
  écrits) — négligeable sur un matvec borné par la lecture des poids, et
  dit ici plutôt que découvert en revue.

## 5. Sorties

Journal : `docs/mesures/f1-cublasf16-2026-08-18.txt`. Registre : une ligne à
`docs/data/jobs.csv`. Le verdict de bande s'applique à B4 (gel des tables).
