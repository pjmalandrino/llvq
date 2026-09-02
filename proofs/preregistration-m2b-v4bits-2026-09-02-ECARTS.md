# Écarts au pré-enregistrement M2b — le §5 a un trou, et la mesure est tombée dedans

> Le pré-enregistrement
> [`preregistration-m2b-v4bits-2026-09-02.md`](preregistration-m2b-v4bits-2026-09-02.md)
> (sha256 `263ec52a…`) est tamponné et ne s'édite pas. Ce qui le corrige est ici.

## É1 — La règle de décision n'est pas exhaustive, et le cas mesuré n'est couvert par aucune ligne

**Ce que le §5 pose** :

| ligne | condition | conséquence |
|---|---|---|
| 1 | `G4 ≥ +3,0` **et** IC95 entièrement `> +1,5` | encaissable |
| 2 | `+1,5 ≤ G4 < +3,0` | partiel |
| 3 | `G4 < +1,5` | mort |

**Ce qui est mesuré** : `G4 = +3,60`, IC95 `[+1,47 ; +5,79]`.

La ligne 1 échoue sur sa seconde condition (1,47 n'est pas > 1,5). Les lignes
2 et 3 échouent sur `G4`. **Aucune ligne ne s'applique.** Le tableau
partitionnait l'axe `G4` mais ajoutait à la seule ligne 1 une condition sur
l'IC, ouvrant une zone — `G4 ≥ 3,0` avec un IC large — que rien ne couvre.

⚠️ **Ce n'est pas un raté de justesse imputable au hasard du bootstrap.**
Borne basse sur 8 graines de rééchantillonnage, le Δ point étant invariant :
**1,47 · 1,49 · 1,47 · 1,49 · 1,48 · 1,42 · 1,45 · 1,46** — jamais au-dessus
de 1,50. La condition échoue de manière stable.

**Ce qui n'est PAS fait ici, et ne doit pas l'être.** Choisir une ligne après
coup, ou récrire le §5 pour que le résultat y entre. La règle est tamponnée ;
son défaut se déclare, il ne se répare pas rétroactivement. Le choix revient à
l'opérateur, sur les faits ci-dessous.

**Les faits, indépendants du trou** :

- l'écart est **résolu contre zéro** — McNemar p = 2,0e-4, IC excluant zéro ;
- le **point estimé** (+3,60) est dans le régime « encaissable » ;
- l'**incertitude n'exclut pas** le régime « partiel » (la borne basse est à
  0,03 pp du seuil qui l'en sépare) ;
- le **coût mémoire est négatif** : 5,149 b/param contre 5,162 aujourd'hui.

Autrement dit : *ce qui est acquis est que le gain existe et qu'il est gratuit
en mémoire ; ce qui n'est pas tranché est s'il est assez gros pour justifier à
lui seul un second format dans le noyau.*

## É2 — Comment écrire la règle la prochaine fois

Le défaut est reproductible et vaut d'être nommé : **une règle de décision doit
partitionner son espace, pas seulement énumérer des cas favorables.** Trois
correctifs, à appliquer au prochain préreg qui pose un seuil :

1. **Une seule grandeur par frontière.** Mêler un point estimé et une borne
   d'intervalle dans la même ligne crée des zones non couvertes dès que les
   deux ne sont pas monotones ensemble.
2. **Tester la partition sur les cas limites AVANT de tamponner** : prendre
   trois valeurs plausibles de la grandeur et vérifier que chacune tombe dans
   exactement une ligne. Ici, `G4 = 3,60` avec un IC large aurait suffi.
3. **Déclarer d'avance ce qu'on fait d'un cas non couvert** — une ligne
   « sinon : non tranché, décision d'opérateur » — plutôt que de découvrir le
   trou avec le résultat sous les yeux.

## É3 — La prédiction signée est juste sur le nombre et fausse sur la conclusion

Le §6 annonçait « +3,5 à +4,3 pp, donc la ligne 1 ». Mesuré **+3,60** : dans
la fourchette. C'est la première prédiction signée juste de la session. Mais
la conclusion attachée est fausse, et **pas parce que le nombre l'est** :
parce que la règle censée le lire était incomplète. Prédire juste et conclure
faux est un mode d'échec distinct de ceux que ce dossier a déjà catalogués, et
il vient du protocole.

La faille que le §6 déclarait s'est par ailleurs matérialisée : l'AWQ protège
ses canaux saillants pendant sa calibration, cet affine nu ne protège rien, et
**19,6 % du gain est perdu**. L'analogie « la perte sera une fraction de
0,28 pp » était trop optimiste — elle vaut 0,88 pp sur cette seule matrice.
