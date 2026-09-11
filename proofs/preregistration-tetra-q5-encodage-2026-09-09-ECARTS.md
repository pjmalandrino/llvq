# Écarts — pré-enregistrement de l'encodage Tetra + Q5 du 2026-09-09

Le prereg est tamponné (`c084a47c00f0a0509783fdbe8c544f38709fdcec4feb5ac971ec1f91909027d2`,
quatre calendriers) et ne s'édite plus.

**Aucun écart ne touche le protocole.** L'encodage a tourné exactement comme §2
l'écrit, une seule variable. Les trois ci-dessous sont des prédictions fausses,
et la troisième est un résultat.

---

## É1 — La durée est sortie sous la fourchette

Prédit **2 h 35** dans [2 h 10 ; 3 h 10]. Mesuré **1 h 58** (7 100 s), soit
**197 s/bloc** contre les 245 du Tetra nu.

Cause, lisible dans le fichier : les 36 `v_proj` sont rangées en affine par
groupe **sans passer par GPTQ, la rotation ni le réseau**
(`calib.rs:793-812`). Elles ne coûtent presque rien. J'ai ancré ma fourchette
sur le run du 2026-09-06 sans retirer ce que la nouvelle variable enlève.

Sans conséquence sur l'objet.

## É2 — La taille est fausse d'un facteur dix, et c'est une multiplication oubliée

Prédit **+0,5 à +2,5 Mo**. Mesuré **+24,0 Mo** (1 794 564 765 contre
1 770 529 149).

Je disposais de l'incrément — **+0,0493 b/param** — et je ne l'ai pas converti :

```
0,0493 × 4 022 468 096 / 8 = 24,8 Mo
```

Ce n'est pas une erreur de modèle, c'est une arithmétique que je n'ai pas
faite. Elle était à une ligne.

**Conséquence retenue** : la ligne « disque ≤ celui d'aujourd'hui » de la porte
F1e (`docs/ROADMAP.md:124`) est définitivement morte. Le Tetra nu était déjà
+1 616 octets au-dessus du publié ; le fichier mixte est +24 Mo. Ce n'est pas
une surprise du run, c'est une propriété du format qui aurait dû être écrite
dans le plan.

## É3 — La perplexité va dans le mauvais sens, et c'est le résultat de l'étape

Prédit **16,10** dans [15,90 ; 16,40], avec le raisonnement écrit :
*« `v_proj` en int4 g128 est plus précis que le réseau à 2 bits, donc la
perplexité doit s'améliorer. »*

Mesuré **16,3161** — dans la fourchette, **le sens inverse**.

```
  Tetra nu     ×1,3203  (ppl carte f16, 2026-09-06)
  Tetra + Q5   ×1,3340  (ppl Metal f32, ce run)
```

Les deux rapports sont pris contre la ligne de base f32 de leur propre run, ce
qui est la comparaison la moins fausse disponible : aucune ppl f32 d'encodage
n'existe pour le Tetra nu. Environ **+1 %**, et le signe est robuste à cette
imprécision.

**Hypothèse, pas conclusion** : les `v_proj` gagnent des bits et **perdent la
compensation d'erreur de GPTQ**. Plus de précision brute, moins de correction.
Le bras qui trancherait — `v_proj` quantifié à 4 bits *à travers* GPTQ —
n'existe pas.

**C'est la sixième dissociation du dossier**, et elle a la forme des
précédentes : Q5 achète +3,47 pp de MMLU (résolu, McNemar 8,6e-5) et paie ~1 %
de perplexité, exactement comme la correction radiale de M1 achetait +1,65 pp
pour +9,5 %.

---

## Ce qui était juste

Les 36 enregistrements int4, au chiffre. Et `verify_artifact` :
**3 633 315 840 poids identiques bit pour bit** contre le modèle évalué.

Deux prédictions sur cinq. Le prereg a fait son travail : il a rendu trois
erreurs lisibles au lieu de les laisser se dissoudre dans un commentaire écrit
après coup.
