# Run de nuit — `Λ₂₄(12)` + 1 bit de gain sur Qwen3-4B

## La commande

```bash
cd /Users/pjmalandrino/Documents/Pro/workspace/poc/llvq
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 nohup cargo run --release -p llvq-llm \
  --features metal,fast-linalg --bin smoke -- \
  64 2048 12 4096 metal nogs leech1c12 999 rot ~/llvq-q4b-c12.safetensors \
  > ~/llvq-run-nuit.log 2>&1 &
```

Durée : **~3,5 h**. Suivi : `tail -f ~/llvq-run-nuit.log`.

Les arguments, dans l'ordre : 64 fenêtres de calibration × 2048 tokens · 12
fenêtres d'évaluation × 4096 · Metal · pas d'échelles de groupe · codebook
`leech1c12` (1 bit de gain, direction plafonnée à la coquille 12) · tous les
blocs · rotation d'entrée · chemin de sauvegarde.

## Pourquoi ce run

Le chiffre publié (**14,9104 à 2,1117 bits/poids**) porte une réserve : notre
débit est 5,6 % au-dessus des 2,000 bits du papier. Plafonner la recherche de
direction à `Λ₂₄(12)` fait tomber l'index de **48 à 47 bits**, ce qui paie le
bit de gain au même débit total. C'est littéralement la meilleure ligne de
leur Table 8 : `norm(Λ₂₄(12))` + 1 bit de gain, 92,14 % de rétention.

Débit attendu : **2,0702 bits/poids** (contre 2,1117), artefact 1,72 Go, ×4,68.
Le résidu au-dessus de 2,000 est la politique de queue — les colonnes non
alignées sur 24 restent en pleine précision — et le papier ne dit jamais ce
qu'il en fait.

## Ce qu'on attend

L'A/B sur 3 blocs de Qwen3-0.6B, config identique par ailleurs :

| codebook | ppl | bits/poids |
|---|---|---|
| `leech1` (boule complète) | 20,3102 | 2,2068 |
| **`leech1c12`** | **20,2938** | **2,1656** |

Meilleur sur les deux axes. Cohérent avec le constat coquille unique de G4 :
un codebook plus petit mais mieux réparti bat un codebook plus large.

Donc l'attendu sur le 4B est **≈ 14,9 ou légèrement mieux, à 2,07 bits/poids**.

⚠️ Si le résultat sort nettement **au-dessus** de 14,91, ce serait une
surprise, et il faudrait regarder avant de conclure : le cap change le
codebook, pas la boucle GPTQ, donc une dégradation franche signalerait un
problème dans l'application du cap plutôt qu'un effet réel.

## Au réveil

```bash
tail -6 ~/llvq-run-nuit.log
```

Trois choses à relever : la perplexité, le débit effectif, et le fait que la
baseline soit toujours **12,2336** (sinon quelque chose a bougé dans le
harnais, et le reste du run n'est pas comparable).

Puis, si tu veux la sonde de génération sur l'artefact :

```bash
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --features metal,fast-linalg --bin probe -- ~/llvq-q4b-c12.safetensors metal 20
```

Et si le chiffre est bon, mettre à jour **trois endroits** : le tableau du
`README.md`, la ligne du `CLAUDE.md` (§ Qwen3-4B), et les chiffres du
`docs/mail-qualcomm-draft.md` — la checklist en fin de brouillon demande
justement qu'aucun chiffre du mail ne diverge du README.

## Précautions

- **Pas de `cargo build` sur `llvq-llm` pendant le run** : le binaire
  `target/release/smoke` est en cours d'exécution, le recompiler le
  remplacerait sous ses pieds. Le reste du workspace se compile sans risque.
- **Aucun checkpoint.** Un redémarrage ou une fermeture de session perd tout.
  Une mise en veille, non.
- Pour suspendre puis reprendre sans rien perdre :

```bash
kill -STOP $(pgrep -f "target/release/smoke")
```

```bash
kill -CONT $(pgrep -f "target/release/smoke")
```

## Ce qui a été ajouté pour ce run

`BallSearcher::set_shell_cap` restreint la recherche à `2..=cap`. Les tables
de classes gardent leurs bornes et leur ordre, donc l'élagage reste valide :
une classe sautée n'aurait pas pu améliorer l'incumbent, et le `break` porte
toujours sur une borne qui domine toutes les suivantes.

Trois tests le tiennent : `shell_cap_restricts_the_codebook_exactly` (le
gagnant plafonné égale le meilleur des coquilles `2..=cap`, et une recherche
non plafonnée atteint toujours la coquille 13 — sinon le cap aurait fuité),
`index_width_follows_the_shell_cap` (48 bits pour la boule complète, 47 pour
`Λ₂₄(12)`), et `capped_quantizer_stays_inside_its_ball` (aucune direction
émise au-dessus du cap, vérifié en retrouvant la coquille du point du réseau).

71 tests verts, zéro warning clippy.
