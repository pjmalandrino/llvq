# Pré-enregistrement — A2 étape 1b : le store étendu à fenêtre fixe (2026-09-01)

> Suite de la branche STOP de l'étape 1 (r = 0,8919 < 0,97, préreg
> `ad77df46…`) et de l'arbitrage d'opérateur du 2026-09-01 : issue 3 de
> [`docs/a2-e1-mecanisme-2026-09-01.md`](../docs/a2-e1-mecanisme-2026-09-01.md).
> Ce document ne s'édite jamais ; écarts →
> `proofs/preregistration-a2-etape1b-etendu-2026-09-01-ECARTS.md`, nommé ici
> d'avance. Tampon avant la première milliseconde mesurée sur carte.

## Ce qui change contre l'étape 1, et rien d'autre

Le store `Prealloc(w)` stocke désormais la forme **ÉTENDUE** (`n_heads`,
32 au 4B) dans des buffers fixes `[b, 32, w, hd]` (~151 Mo à w = 256,
*calculé*), écrit le pas étendu par `slice_set`, et rend les **buffers
pleins contigus** — plus de vue `narrow`, plus de `repeat_kv` par pas sur ce
chemin. L'attention lit toute la fenêtre derrière un masque causal large
dont la règle causale EST la règle de validité (les colonnes au-delà du
préfixe valide sont −inf). Trois preuves à 0 $ AVANT la carte :
- préfixe du buffer == `repeat_kv` du chemin cat, octet par octet, à chaque
  pas, F16 ET Q8 ; queue exactement nulle
  (`prealloc_expanded_matches_repeat_kv_of_cat_at_every_step`) ;
- le rembourrage est **inerte au bit près** à travers
  scores→softmax→ctx, en exerçant `build_causal_mask` lui-même
  (`wide_mask_padding_is_exactly_inert`) ;
- débordement de fenêtre = erreur nommée (inchangé).

## Le protocole, figé — IDENTIQUE à l'étape 1

`fusedrun` sous `LLVQ_KV_AB=1 LLVQ_KV_PREALLOC=256`, config servie v1, un
seul processus, 5 paires de rounds entrelacées, rapport `r = tok/s(prealloc)
÷ tok/s(cat)` formé round par round, gate de tokens sur chaque round de
chaque bras. l40sx1, même image, même montage. ~0,1 $.

## La lecture, posée d'avance — IDENTIQUE à l'étape 1

- **r ≥ 0,97** → le store étendu devient la base des deux bras de l'étape 3
  (graph contre non-graph à store constant).
- **r < 0,97** → seconde régression : arrêt, retour opérateur — et la ligne
  « base fixe » serait alors sérieusement entamée.

## Le prior, déclaré — et cette fois il est POSITIF

**r ≥ 1 attendu, mesuré en direction sur Metal avant ce tampon (0 $).** Le
bras cat paie `repeat_kv().contiguous()` — la copie ×4 de TOUTE l'histoire —
à chaque pas ; le bras étendu ne copie que le pas (×4 de l = 1) et ne
recopie jamais l'histoire. Sonde Metal (bin/run, 64 tokens/prompt, régime
stationnaire) : cat 9,4–9,7 tok/s, étendu 9,0–9,7 — **la régression de
−29 % de la v1 a disparu, parité dans la dispersion de la sonde** ; et le
témoin `LLVQ_VERIFY_CACHE` rend « identique au chemin sans cache » sur les
4 prompts, store étendu actif. Sur carte, où le −11 % de la v1 était plus
petit qu'en Metal, le gain de la copie ×4 supprimée peut émerger — ou rester
sous la dispersion : les deux issues se publient. ⚠️ Si r < 0,97 malgré ce
prior, le mécanisme est à instruire avant toute étape 2 — un prior positif
réfuté vaut autant qu'un négatif.

## Budget

~0,1 $. Phase A dépensée : 0,09 $ sur 4.
