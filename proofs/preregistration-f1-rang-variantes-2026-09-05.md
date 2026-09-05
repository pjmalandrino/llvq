# Pré-enregistrement — trois arithmétiques pour le même décodeur F1 (`tv_f1r_v1/v2/v3` contre `tv_f1r`)

**Écrit le 2026-09-05 au soir, commité et TAMPONNÉ avant la première milliseconde.**
Vague 2, plafond **2,00 $**, dépensé 0,03 $ ; ce job coûte **~0,01 $**, plafond propre **0,10 $**
(go opérateur du 2026-09-05 : « go pour le décodeur pour tester et sinon on revient à la version stabilisée »).
Code mesuré : commit `2df33e7` (branche `f1/plancher-table`).

🚨 **Ce fichier ne s'édite plus.** Ce qui le corrige va dans un `-ECARTS.md`.

## §1 — Ce n'est pas une porte

Règle des 04 et 05 (`docs/METHODE.md` §1) : une porte se pose sur un critère fondamental ; un kill est à
l'opérateur. Ce banc compare des noyaux entre eux dans le même processus. Il décide **quelle écriture du
décodeur F1d emporte**, rien d'autre. Le codebook, la table de 16 Kio et le mot de 48 bits sont les mêmes
dans tous les bras ; ce qui change est la suite d'instructions qui rend les 24 coordonnées.

## §2 — La question

Le plancher compilé du 05 (`docs/mesures/f1-rang-plancher-2026-09-05.txt`) : `T = 3,346 ms` contre
`B = 2,797`, dont `Du = 2,701` pour la table et le décodage, la table seule valant 0,663. **~2,0 ms
d'arithmétique.** Deux suspects nommés dans les écarts : les 24 conversions entier → flottant par bloc (pipe I2F
à 1/8 du débit FMA sur sm_89) et la chaîne de trois petites lectures dépendantes `s8 → branches → s16 → suffixes`.
Trois variantes, écrites indépendamment contre une même spécification, attaquent chacune un suspect :

```
  v1  « sans I2F »        flottant fabriqué par masque et FADD, signe par XOR ; tables inchangées
  v2  « sans chaîne »     les octets de motif par algèbre sur F₂ (parités de masques) au lieu de trois
                          lectures dépendantes ; la table de rangs reste lue
  v3  « tables PRMT »     magnitudes en octets par permutation d'octets, flottant par PRMT + FADD
```

Chaque variante rend, pour chaque ligne, **le même produit scalaire** que `tv_f1r` à l'arrondi flottant près.

## §3 — Le montage

`bin/f1rankfloor`, géométrie de `tv_nullk`, 252 lancements par tour ; six bras à chaque tour, ordre tournant
sur les six, 14 tours dont 2 de chauffe (12 gardés, chaque bras à chaque position deux fois) ; différences
formées tour par tour, médiane [plage] :

```
  nullk · word · f1r · f1r_v1 · f1r_v2 · f1r_v3

  Du      = t(f1r)    − t(word)      référence, attendue ≈ 2,70 (05)
  Du_vk   = t(f1r_vk) − t(word)
  T_vk    = t(f1r_vk) − t(nullk)     à lire contre T = t(f1r) − t(nullk) du MÊME processus,
                                     et contre B = 2,797 (autre processus, échelle seulement)
```

Même flux de mots (36 copies, 0,908 Go), mêmes tables, même graine que le 05. Commande : image
`ops/run.py publish` sur le commit ci-dessus, puis `f1rankfloor 2>&1 | tee $OUT/f1rankfloor.txt`.

## §4 — Les contrôles ; si l'un des contrôles 1, 3, 4 ou 5 tombe, aucun chiffre n'est publié

1. **Sur le Mac, avant la carte** : pour chaque variante, 10 000 mots décodés en flottants par clang++
   égalent `(float) decode_word` bit à bit, et le produit scalaire contre 200 vecteurs égale la référence f64 à
   1e-5 relatif (`llvq-cuda/tests/f1rank_v*_matches_rust.rs`). La fonction `__byte_perm` du shim hôte est
   testée sur 16 cas contre la définition PTX de `prmt.b32` ; ⚠️ l'intrinsèque CUDA masque le sélecteur à
   3 bits par quartet (0x7777), donc aucun en-tête expédié ne s'appuie sur le mode de réplication de signe —
   trouvé par la relecture avant la carte, V3 corrigé (masque par multiplication, +1 IMAD par quadruplet).
2. **Sur la carte** : pour chaque variante, sur chaque ligne de chaque forme (dernier tour),
   `|y_vk − y| ≤ 1e-5 · max(1, |y|)` où `y` est la sortie de `tv_f1r`. Une ligne hors tolérance : **la
   variante est hors jeu**, ses temps ne sont pas publiés, les autres bras se lisent (§6, première ligne).
   Rejeu hôte de la dérive d'arrondi avec l'arithmétique exacte du banc : 3,3e-7 au pire, 30× sous la
   tolérance ; une coordonnée fausse déplace une ligne d'au moins 3× la tolérance (*calculé*).
3. Contrôle 1 du 05 maintenu : `tv_f1r_dump` = référence Rust sur 64 512 blocs.
4. Élision : chaque bras de table diffère de `word` et de `nullk`.
5. Sorties finies, un processus, flux ≥ 4 × L2, registres et locaux imprimés pour les six noyaux.

## §5 — Ce qui est publié, et ce qui ne se compare pas

Publié : les six temps, `Du`, `Du_vk`, `T`, `T_vk`, registres et locaux, les contrôles.
Ne se compare pas : à Planes14 par soustraction (B est une échelle) ; à QTIP ; à un tok/s.

## §6 — Ce que le résultat décide

La résolution de ces bancs est ±0,1 ms (É7 du plancher de table). Soit `v*` la variante de `T_v` minimal
parmi celles qui passent les contrôles, avec registres ≤ 64 et locaux = 0.

| résultat | suite |
|---|---|
| un contrôle tombe pour une variante | cette variante est hors jeu ; les autres se lisent |
| aucune variante ne passe les contrôles | rien n'est publié ; bug, à corriger |
| T_v* ≤ T − 0,3 ms | **F1d prend v\***, et sa combinaison avec les autres suspects s'écrit dans F1d |
| T − 0,3 < T_v* ≤ T − 0,1 | gain réel mais mince ; F1d prend v\* si ses registres ≤ ceux de `tv_f1r`, sinon `tv_f1r` |
| T_v* > T − 0,1 ms | **retour à la version mesurée du 05** (`tv_f1r`) ; l'arithmétique n'est pas le levier qu'on croyait, et le journal dit lequel des deux suspects a été innocenté |
| autrement | non tranché, décision d'opérateur |

Aucune ligne n'est un kill de F1.

## §7 — Prédiction signée, opposable

**Du_v1 entre 1,8 et 2,3 ms** (les I2F non masqués valent ~0,6 ms ; leur remplacement par LOP+FADD sur le pipe
FMA en rend la moitié au moins). **Du_v2 entre 2,0 et 2,5** (la chaîne coûte de la latence, que 48 warps masquent
en partie ; les parités de masques coûtent des instructions). **Du_v3 entre 1,6 et 2,2** (PRMT remplace à la fois
les sélections et l'I2F). **La meilleure variante : T_v* entre 2,6 et 3,1 ms**, contre T = 3,35 ; probabilité
qu'elle passe sous B = 2,797 : 40 %. Registres : v1 ≤ 48, v2 ≤ 56, v3 ≤ 48. Locaux 0 partout.

**Ce qui la rendrait fausse de façon instructive** : Du_v1 ≥ 2,6 ms — l'I2F n'était pas le coût, et la chaîne
de lectures ou l'occupation le sont ; ou Du_v3 < 1,3 ms — l'arithmétique était presque tout, et F1d passe sous
Planes14 dès ce décodeur.

⚠️ Historique des prédictions signées : celle du plancher compilé, trois heures plus tôt, était fausse trois
fois sur cinq nombres (S, Du, T). Celle-ci est opposable, pas crédible d'avance.

## §8 — Ce que ce banc ne peut pas être

Un coût (pas d'échelle de gain, étiquettes uniformes, pas de Planes14 dans le processus), ni une qualité. Ce
qu'il est : trois écritures du même décodeur, vérifiées contre la même référence, chronométrées côte à côte.
