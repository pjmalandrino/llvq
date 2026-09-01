# Pré-enregistrement — A2 étape 1 : l'A/B prealloc-contre-cat (2026-09-01)

> Sous le préreg de phase `preregistration-a2-a3-geometrie-2026-08-31.md`
> (sha256 `802006c5…`, tamponné), qui exige cette mesure AVANT toute capture :
> « la prealloc se mesure dans son propre A/B (prealloc contre `Tensor::cat`,
> sans graph), avant ». Ce document ne s'édite jamais ; écarts →
> `proofs/preregistration-a2-etape1-prealloc-2026-09-01-ECARTS.md`, nommé ici
> d'avance. Tampon avant la première milliseconde mesurée.

## La question

Que coûte — ou rend — le stockage préalloué du cache KV, seul, sur le chemin
servi v1, à géométrie et noyaux constants ?

## Le protocole, figé

- `fusedrun` sous `LLVQ_KV_AB=1` (commit qui introduit le mode) : **UN**
  modèle fusé chargé une fois — config servie v1 : `planes14 + LLVQ_EMBED=q8
  + LLVQ_ROT_SHARE=1 + LLVQ_FUSE=1` — puis **5 paires de rounds
  entrelacées**, le store basculé par `set_kv_store` entre deux `generate`
  du même processus : mêmes poids, même unité NVRTC, même prompt, 128
  tokens. Une génération jetée PAR store (sélection de noyaux, montée en
  fréquence, allocations de fenêtre).
- **Fenêtre : `LLVQ_KV_PREALLOC=256`** (prompt ~5 + 128 ≤ 256, la borne du
  préfill existant). Le mode refuse de démarrer sans fenêtre explicite.
- **Le rapport se forme round par round** : `r = tok/s(prealloc) ÷
  tok/s(cat)` par paire, médiane et plage sur les 5 paires — jamais un
  quotient de médianes de bras séparés.
- **Gate de justesse, avant tout chiffre** : les tokens de CHAQUE round de
  CHAQUE store sont comparés à la référence — une divergence tue le job, on
  ne chronomètre pas des bras qui ne rendent pas les mêmes tokens. (Sur Mac,
  l'identité octet par octet cat/prealloc est déjà prouvée par
  `prealloc_matches_cat_at_every_step`, F16 et Q8.)
- Carte : l40sx1, image `llvq-runner-cuda`, modèle 4B publié monté depuis
  `Pier-Jean/Qwen3-4B-LLVQ-2bit`. Coût ~0,25 $, plafond de phase 4 $.

## La lecture, posée d'avance

- **r ≥ 0,97** → la prealloc ne régresse pas au-delà du bruit : elle devient
  la BASE des deux bras de l'étape 3 (graph contre non-graph, à prealloc
  constante — la règle `check_fuse` du préreg de phase). Un r > 1 se publie
  comme gain de la prealloc seule, mais ce n'est pas ce qu'on attend.
- **r < 0,97** → régression : arrêt, retour à l'opérateur — l'étape 2 ne
  s'engage pas sur un store qui coûte 3 %.

## Le prior, déclaré

**r ≈ 1,00, et un écart notable serait une surprise à expliquer.** Le
`Tensor::cat` retiré ne copiait que l'histoire ×1 par pas, quand
`repeat_kv().contiguous()` — présent dans LES DEUX bras — la recopie ×4 à
chaque pas (model.rs:1001-1002 ; cadrage A2 §2). À 128 tokens le poste
retiré est petit devant ce qui reste. Le motif de la prealloc n'est pas ce
gain : c'est la **capturabilité** (un graph statique ne capture pas un cat
qui grandit). Cette mesure existe pour que l'étape 3 attribue au graph ce
qui est au graph — pas pour rendre un chiffre spectaculaire.

## Budget

~0,25 $ (*estimé*, gabarit D1 : 0,24 $ mesuré). Phase A : 0 $ dépensé sur 4.
