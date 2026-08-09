# Remontée amont — `broadcast_matmul` recopie son rhs (candle)

✅ **Ouverte le 2026-08-09 :
[huggingface/candle#3871](https://github.com/huggingface/candle/issues/3871).**

| fichier | quoi |
|---|---|
| [`ISSUE.md`](ISSUE.md) | **archive verbatim** du corps posté — ne pas l'éditer, son intérêt est qu'il corresponde à l'issue |
| [`candle-broadcast-matmul.patch`](candle-broadcast-matmul.patch) | le correctif + son test, `git diff` contre `main` @ `6f74e7c`. Copie éditable |
| [`repro/`](repro) | le reproducteur autonome, CPU, sans GPU ni modèle. Copie éditable |
| [`../../figures/broadcast-matmul-bug.svg`](../../figures/broadcast-matmul-bug.svg) | le schéma du mécanisme, deux couloirs |

Le repro et le patch sont **inlinés dans le corps de l'issue** : les liens
relatifs ne résolvent pas depuis le tracker de candle, et un rapport qui dépend
d'un lien vers un dépôt tiers vieillit mal. Ils restent ici en fichiers séparés
parce que c'est eux qu'on modifie ; `ISSUE.md` n'est qu'une empreinte.

## Ce qui est établi, et comment

1. **Le défaut existe** et il est sur `main` (`6f74e7c`, 0.11.0-dev) comme sur la
   0.9.2 qu'on utilise : `Tensor::broadcast_matmul`, bras `(false, true)`,
   matérialise `rhs.broadcast_as(...).contiguous()`. Pour une tête de sortie
   `(1,1,2560) × (151936,2560)ᵀ` en f16, c'est **778 Mo recopiés par appel**, et
   depuis une vue transposée — donc un *gather* strié, pas un memcpy.
2. **Mesuré des deux côtés.** CPU (4 vCPU, `main`, release) : 8 104 ms contre
   76,6 ms pour le repli manuel en f16 ; 23 663 contre 151 en f32. GPU (L40S, notre
   job du 07-08, [`mesures/phases-2026-08-07.txt`](../../mesures/phases-2026-08-07.txt))
   : phase tête **26,7 ms/token** contre 13,3 ms pour les 36 blocs réunis.
3. **Le correctif tient.** Repli des dimensions de tête du lhs dans les lignes —
   le tour que `candle_nn::Linear::forward` fait déjà, remonté dans la primitive.
   Sur `main` : `cargo test -p candle-core --release` vert en entier (`grad_tests`
   compris), `cargo fmt --check` propre, et le test ajouté **meurt** si on mute le
   repli (`reshape((batch*m, k))` → `reshape((m, batch*k))`). Après correctif,
   `broadcast_matmul` et le repli manuel sont le même code : 81,0 ms en f16.
4. **Bonus non cherché : le repli est aussi plus juste en f16.** Contre le produit
   f32 des mêmes entrées, erreur relative **1,37e-2** pour le chemin diffusé contre
   **3,41e-4** pour le repli — ×40. Le chemin par lots semble accumuler dans un type
   plus étroit ; mécanisme non poursuivi.

## ✅ L'attribution a été corrigée dans tout le dépôt (2026-08-09)

En remontant à la source pour écrire l'issue, un point de nos publications ne
tenait pas : **`candle_nn::Linear::forward` évite déjà ce chemin**,
délibérément, avec le commentaire « we avoid using a broadcasted matmul as it is
much slower ». Il replie les dimensions de tête exactement comme notre
correctif, et `candle_transformers::models::qwen3::ModelForCausalLM::forward`
applique sa tête via `Linear`. **Donc le qwen3 de candle ne paie pas cette
copie.**

Celui qui la paie, c'est `Head::project` — **notre** code,
`llvq-llm/src/model.rs:553`, `h.broadcast_matmul(&t.t()?)`. Le bras dense de
`bin/fusedrun` charge par `sealed::load`, donc c'est bien ce chemin-là qui rend
les 26,7 ms.

Le chiffre était juste ; c'est l'étiquette qui ne l'était pas. **Les cinq
formulations relevées ici ont été corrigées** dans le commit *« La copie de
778 Mo était la nôtre, et seize documents disaient candle »*, plus onze autres
trouvées au balayage : `README.md`, `docs/hf-model-card.md`, `paper/main.tex`,
`paper/sections/{integration,related,conclusion}.tex`, `CLAUDE.md` (deux
endroits) et neuf docs. Le journal daté `verdicts-nuit-2026-08-07.md` a reçu un
errata en tête plutôt qu'une réécriture.

**Ce qui n'a pas bougé** : aucun chiffre. Ni le ×2,03, ni le ×1,12, ni les
26,7 ms, ni les 778 Mo. L'argument en sort **durci** — la baseline handicapée
étant la nôtre, le ×2,03 ne porte plus rien seul, et le papier le dit maintenant
explicitement au lieu de le laisser sous-entendre.

⚠️ **Restent justes et non touchés** : `verdict-a2-repeat-kv-2026-08-06.md`,
`passation-lot-a-2026-08-06.md`, `rapport-lot-a-2026-08-06.md` et
`tableau-8b-2026-08-07.md`. Ils décrivent le comportement de `broadcast_matmul`
sans l'attribuer aux modèles de candle.

## Ce qui reste à décider

1. **Corriger `Head::project`** (`llvq-llm/src/model.rs:553`) par le même repli.
   Une ligne, qualité inchangée — mais ça déplace le tok/s du **bras dense**,
   donc la référence de tous les tableaux publiés. À faire, mais en le disant :
   c'est un nouveau témoin, pas une correction silencieuse.
2. **La PR amont.** Le patch est prêt et vérifié sur `main` @ `6f74e7c`, l'issue
   dit « happy to send this as a PR » ; il ne manque qu'un fork où pousser.
