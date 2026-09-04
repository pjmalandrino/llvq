# Écarts et corrections au pré-enregistrement M2 — écrits après le job, jamais dans le tampon

> Le pré-enregistrement
> [`preregistration-m2-attribution-4b-2026-09-02.md`](preregistration-m2-attribution-4b-2026-09-02.md)
> (sha256 `71712e60…`) est tamponné, et le journal
> [`docs/mesures/m2-attribution-4b-2026-09-02.txt`](../docs/mesures/m2-attribution-4b-2026-09-02.txt)
> est un fait brut : ni l'un ni l'autre ne s'édite. Ce qui les corrige s'écrit ici.

## É1 — Le journal ne chiffre le coût de la cible QU'EN f16, et en conclut trop

**Ce que le journal dit** (section « le coût mémoire dit que le gain n'est pas
encaissable sous v1 ») : servir une cible en f16 coûte au mieux **+0,263
b/param**, ce qui porte le 4B à 5,425 contre 5,302 pour l'AWQ, donc au-dessus.
C'est **exact**, et la table qui l'accompagne est juste.

**Ce qu'il omet, et l'omission change la conclusion.** Il ne calcule le coût
que pour f16, alors que f16 n'est pas le seul niveau concevable — c'est
seulement le seul que le chemin servi sait produire *aujourd'hui*. Or notre
format 2 bits **coûte 4,804 b/poids en VRAM** parce qu'il déplie l'index. Un
vrai 4 bits en coûte **moins**. En b/param modèle entier (*calculé*, servi
5,162 ; part des projections 90,3 % ; v_proj = 2,6 % des projections) :

| `v_proj` servi en… | b/poids | Δ b/param | total | contre AWQ 5,302 |
|---|---|---|---|---|
| Leech déplié `Planes14` (aujourd'hui) | 4,804 | +0,000 | 5,162 | ✅ |
| f16 | 16,000 | **+0,263** | 5,425 | ❌ |
| 4 bits type AWQ w4 g128 | 4,156 | **−0,015** | **5,147** | ✅ |
| 4 bits, hypothèse prudente | 4,500 | −0,007 | 5,155 | ✅ |

**Servir `v_proj` en vrai 4 bits coûterait MOINS de VRAM qu'aujourd'hui.** Ce
n'est pas un paradoxe, c'est la thèse du papier retournée contre nous : le
dépliage fait payer 4,80 bits par poids pour 2,00 bits d'information, donc un
format honnête à 4 bits est plus compact que notre 2 bits en mémoire.

**La bonne conclusion, qui remplace celle du journal** : Q5 n'est pas « non
rentable ». Q5 est **potentiellement gratuit sur l'axe mémoire, et positif sur
l'axe qualité** — et ce qui manque n'est ni un budget ni une cible, c'est un
**chemin de précision mixte** dans le noyau fusé et dans le format d'archive.
La conclusion « chantier de format » du journal tient donc ; ce qui tombe est
le mot « non rentable », qui la faisait lire comme un abandon.

⚠️ **Trois choses que ce calcul ne dit pas, et elles sont lourdes.**

1. **Le +4,48 pp est mesuré à f16, pas à 4 bits.** À 4 bits le gain serait
   moindre. De combien : **non mesuré**. Le seul repère du dossier est que
   l'AWQ 4 bits perd 0,28 pp sur le f16 *sur le modèle entier* au 4B, ce qui
   suggère qu'une matrice seule en garderait l'essentiel — mais c'est une
   analogie, pas une mesure, et elle porte sur un autre objet.
2. **Le coût DISQUE augmente** : 2,0702 → 2,1333 b/poids effectifs (+0,063).
   Sur l'axe où nous gagnons contre tout le monde, c'est une concession.
3. **Le chemin n'existe pas.** Il faut un second format dans le noyau fusé, et
   l'archive doit le porter. Ce n'est pas de la recherche, c'est du travail.

## É2 — L'expérience qui tranche É1 est bon marché, et elle n'est pas lancée

`LLVQ_RESTORE_F16` restaure depuis le checkpoint **en f16**. Le même mécanisme
avec une quantification scalaire 4 bits au chargement — exactement ce que
`LLVQ_EMBED=q8` fait déjà pour l'embedding — donnerait le bras manquant :
**`v_proj` à 4 bits, tout le reste tel que livré, MMLU apparié**. Un bras, ~7
min de L40S, **≈ 0,20 $** (*estimé* au tarif mesuré de M2), sur les 2,83 $ qui
restent de la vague.

Il tranche la réserve n° 1 ci-dessus, qui est la seule qui décide. Il demande
son propre pré-enregistrement — le tampon du 2026-09-02 ne couvre pas un bras
qui n'y figure pas — et un go d'opérateur.

## É3 — Ce que le §4 avait posé d'avance et qui s'est vérifié

La somme des sept gains marginaux vaut **23,85 pp** pour un déficit de
**14,73 pp**. Le §4 interdisait d'avance de lire ces sept nombres comme une
décomposition du déficit ; la mesure montre qu'il avait raison, et de 62 %.

## É4 — La réplique du §5 part le 2026-09-04 : quelle clause la lit, déclarée AVANT

Go d'opérateur le 2026-09-04. La réplique est **déjà pré-enregistrée** : le §5
du tampon `71712e60` nomme l'artefact, la commande et la règle de lecture. Rien
n'est rouvert ici. Ce qui suit déclare **quelle clause de cette règle
s'applique**, et le déclare avant la mesure — c'est le correctif n° 3 de
[`…-m2b-v4bits-2026-09-02-ECARTS.md`](preregistration-m2b-v4bits-2026-09-02-ECARTS.md)
§É2, appliqué pour la première fois.

**Le problème.** Le §5 retient la cible « si elle est le type de plus grand Δ
sur les deux tirages, **ou** si son IC95 recouvre celui du premier ». La cible
retenue est `v_proj`, et `v_proj` **n'est pas** le plus grand Δ du premier
tirage : `gate` rend +5,18 et `up` +4,94 contre +4,48. La première clause ne
peut donc pas s'appliquer, quel que soit le résultat de la graine 3.

**Pourquoi la cible est quand même `v_proj`** : le §5 le dit lui-même, « le
coût décide autant que le signal ». `v_proj` pèse 94 371 840 poids par type
contre 896 532 480 pour `gate` (*calculé*, §3.5 du préreg) ; c'est le seul
grand Δ qui se sert sans dépasser l'AWQ en mémoire.

**Ce qui est déclaré maintenant, sans connaître la graine 3** :

1. La lecture se fait sur la **seconde clause** — recouvrement des IC95 de
   `v_proj` entre les deux tirages. Ce n'est pas un assouplissement : c'est la
   seule des deux clauses qui pouvait s'appliquer dès le 2026-09-02.
2. « Recouvre » se lit littéralement : les deux intervalles appariés à 95 % ont
   une intersection non vide. IC du premier tirage : **[2,39 ; 6,61]**.
3. **Sinon**, l'attribution est publiée comme **dépendante du tirage** et Q5 ne
   s'ouvre pas sur un seul artefact — la troisième issue que le §5 prévoyait.
4. Rapporté en plus, sans que rien n'en dépende : le **classement complet** des
   sept types sur la graine 3, et si le trio de tête `gate` / `up` / `v` tient.

**Ce qui est reconnu** : cette déclaration est écrite en connaissant M2, pas la
graine 3. Elle ne choisit pas un seuil, elle constate qu'une des deux clauses
était inapplicable dès l'écriture du §5.

## É5 — L'image n'est PAS reconstruite, et c'est délibéré

La réplique tourne sur **l'image de M2, inchangée**. Le commit `8a808f9`
(2026-09-03) a réécrit commentaires et messages de 104 fichiers de code, dont
`llvq-llm/src/bin/mmlu.rs` ; reconstruire mettrait un **second facteur** entre
les deux tirages, alors que la règle du dépôt est qu'un A/B ne bouge qu'un
mécanisme (`docs/METHODE.md` §1). Le seul écart voulu entre M2 et sa réplique
est le fichier quantifié.

**Coût et plafond.** M2 a coûté 2,17 $ pour 72,3 min (*mesuré*). La vague 1 a
dépensé 2,46 $ sur 5, il reste **2,54 $**, soit **84,6 min** à 1,80 $/h. Le job
part donc avec `--timeout 84m` et non les `2h` de M2 : le plafond de vague tient
**même au pire cas**, 4,98 $ sur 5. Marge sur le temps mesuré de M2 : 16 %.
Si le job est tué par ce `timeout`, la dépense est perdue et le constat est
écrit ici — c'est le prix pour ne pas dépasser un plafond posé d'avance.

## É6 — Deux contrôles que M2 n'avait pas, gratuits, posés avant le job

Le §3 du tampon donne ses six contrôles pour le fichier publié. Deux d'entre
eux se transposent à la graine 3 avec des valeurs **déjà mesurées**, donc plus
serrées que « ≥ 99,5 % de picks » :

1. **Contrôle bas.** Le bras « tel que livré » sur `q4b-s3-sealed.llvq` doit
   rendre **55,17 %** en micro et son dump doit reproduire
   `docs/data/bruit-mmlu-graines/mmlu-s3.csv`, sha256 `5f14bd34…` (*mesuré*,
   [`docs/mesures/bruit-mmlu-graines-4b-2026-08-25.txt`](../docs/mesures/bruit-mmlu-graines-4b-2026-08-25.txt)).
   C'est le même objet scoré par le même binaire : l'écart attendu est nul, pas
   « petit ». Un écart non nul veut dire qu'un des deux jobs n'a pas scoré ce
   qu'il croyait.
2. **Contrôle haut.** « Tout restauré » doit rendre **70,32 %**, le checkpoint,
   **exactement comme dans M2** — la graine du fichier quantifié n'entre plus
   dans un modèle dont les 252 matrices viennent du checkpoint. C'est le seul
   nombre que les deux jobs doivent partager au centième, et il croise les deux
   dépenses l'une par l'autre.

Si l'un des deux tombe, aucun Δ de la réplique n'est publié.

**Ce qui n'est PAS refait** : le sha256 du fichier de 1,77 Go sur la carte. Sa
taille, 1 770 528 125 octets, le distingue déjà du fichier publié (1 770 527 533,
§3.2 du tampon), et hacher 1,77 Go depuis un volume monté coûte du temps sur un
`timeout` volontairement serré.
