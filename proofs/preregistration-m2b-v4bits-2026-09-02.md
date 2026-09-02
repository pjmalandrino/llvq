# Pré-enregistrement — M2b : le gain de `v_proj` survit-il à quatre bits ?

**Écrit, commité et TAMPONNÉ le 2026-09-02, AVANT le lancement.** Go de
l'opérateur du 2026-09-02. Reste de la vague 1 : **2,83 $** sur 5.

🚨 **Ce fichier ne s'édite plus** ; un fait démenti s'écrit dans un `-ECARTS.md`
à côté. Code mesuré : commit de `LLVQ_RESTORE_Q4`, plomberie vérifiée à 0 $ sur
le fichier publié avant le lancement (étiquette et refus des deux variables).

**Coût : ≈ 0,20 $** (*estimé* : 1 bras × ~6,5 min à 1,80 $/h, depuis les 72 min
mesurées pour 11 bras au job `6a97ea8e…`). Plafond réel = le `timeout`, posé à
**30 min → 0,90 $ au pire**.

---

## §1 — La question, et pourquoi elle décide

M2 a mesuré ce que vaut `v_proj` **à f16** : **+4,48 pp de MMLU**, IC95
[+2,39 ; +6,61], pour 2,6 % des poids. Mais f16 n'est pas servable — il coûte
+0,263 b/param et fait repasser le 4B **au-dessus** de l'AWQ (5,425 contre
5,302), or c'est l'axe mémoire qui porte la thèse du projet.

Le fait qui rouvre la question est dans l'écart É1 de M2 : **notre format
dépense 4,804 b/poids de VRAM pour 2,070 b/poids d'information**. Un vrai
4 bits en coûte moins. Servir `v_proj` en 4 bits coûterait donc **−0,013
b/param** (*calculé*, affine int4 g128 à 4,25 b/poids : 4 bits + scale et biais
f16 par groupe de 128) : **5,149 contre 5,162 aujourd'hui**. Mémoire en baisse
et qualité en hausse, sur les deux axes à la fois.

**Tout cela repose sur une inconnue, et c'est la seule que ce job mesure :
combien du +4,48 survit quand la matrice n'a plus 16 bits mais 4.**

## §2 — Le job, verbatim

Image `hf.co/spaces/Pier-Jean/llvq-runner-cuda` reconstruite au commit qui
porte `LLVQ_RESTORE_Q4`. Flavor `l40sx1`. Volumes : modèle
`Pier-Jean/Qwen3-4B-LLVQ-2bit` → `/model`, bucket `Pier-Jean/jobs-artifacts`
→ `/out`.

```
OUT=/out/m2b-v4bits-2026-09-02
mkdir -p $OUT
nvidia-smi --query-gpu=name,driver_version --format=csv | tee $OUT/gpu.txt
F=/model/qwen3-4b-llvq.bin
ls -l $F | tee $OUT/artefact.txt
export LLVQ_MODEL=Qwen/Qwen3-4B
LLVQ_RESTORE_Q4=v_proj LLVQ_MMLU_DUMP=$OUT/mmlu-v4.csv \
  mmlu $F cuda 40 2>&1 | tee $OUT/mmlu-v4.txt
```

Un seul bras. Les deux témoins nécessaires — le livré (55,59) et `v_proj` à
f16 (60,07) — **existent déjà**, mesurés dans le même protocole, à la même
empreinte, et leurs dumps sont commités dans `docs/data/m2-attribution/`. Les
repayer serait payer deux fois le même chiffre.

## §3 — Contrôles, et si l'un tombe aucun chiffre n'est publié

1. Empreinte de tokens **`65dcd53655e8bfa5`**, 2 280 questions.
2. Artefact **1 770 527 533 octets**.
3. Le bras imprime **`restored v_proj at int4 g128 (dequantized) (36
   matrices, 94371840 weights)`** — mêmes 36 matrices et mêmes 94 371 840
   poids que le bras f16 de M2, sinon ce n'est pas la même cible.
4. Le dump par question est écrit, avec son trailer.
5. Le micro du bras **n'est égal ni à 55,59 ni à 60,07** au centième : une
   égalité exacte avec l'un des deux témoins voudrait dire que la
   quantification n'a rien fait, ou qu'elle a tout détruit. Le test unitaire
   `the_q4_round_trip_quantizes_rather_than_copying` couvre déjà le premier
   cas hors carte ; ce contrôle-ci le couvre sur les octets réels.

## §4 — Ce qui se publie, et ce qui NE se compare PAS

Se publie : **G4 = q4 − livré** et **Gf = f16 − livré (= +4,48)**, appariés
sur les dumps (bootstrap stratifié 10 000 tirages, graine `0xb0075eed`,
McNemar exact), et le **taux de survie G4/Gf**. Plus le coût mémoire
correspondant, *calculé* dans la comptabilité b/param modèle entier.

Ne se compare pas :
- **Ce bras n'est pas un noyau.** Il quantifie puis déquantifie ; aucun octet
  de 4 bits n'est lu par un noyau. Il mesure un **coût d'information**, pas une
  vitesse ni une empreinte réelle. Un chemin servi à précision mixte n'existe
  pas et ce job n'en crée aucun.
- **4,25 b/poids n'est pas 4,156.** Notre affine stocke scale *et* biais en
  f16 ; l'AWQ empaquette son zéro. Écart déclaré de 0,09 b/poids, en notre
  défaveur.
- Un seul tirage de calibration, une seule taille, une seule matrice.

## §5 — La règle de décision, posée AVANT de voir le chiffre

Soit **s = G4 / Gf** le taux de survie du gain.

| | conséquence |
|---|---|
| **G4 ≥ +3,0 pp et son IC95 entièrement > +1,5 pp** | ✅ **Le gain est encaissable.** Q5 devient un chantier d'ingénierie chiffré : un chemin de précision mixte dans le noyau fusé et le format d'archive, pour un gain net sur les DEUX axes. Il passe devant F3 dans la roadmap. |
| **+1,5 pp ≤ G4 < +3,0 pp** | ⚠️ **Partiel.** Le gain existe mais ne justifie pas seul un second format ; il est mis en réserve et rejoint F1, où le débit par bloc le rend gratuit à écrire. |
| **G4 < +1,5 pp** | ❌ **Mort.** Le +4,48 était un artefact de la pleine précision. Q5 se ferme sur `v_proj`, et la conclusion de M2 (« chantier de format ») reste, mais sans cette cible pour le motiver. |

⚠️ **Le seuil porte sur G4, pas sur s.** Un taux de survie flatteur sur un
gain devenu petit ne décide rien, et c'est l'erreur que ce tableau existe pour
empêcher.

## §6 — Divulgation datée, à la signature

- Connu : livré 55,59 · `v_proj` f16 60,07 (+4,48 [+2,39 ; +6,61]) · f16
  complet 70,32 · AWQ 4 bits **70,04**, soit −0,28 pp du f16 **sur le modèle
  entier**.
- **Prédiction de l'auteur, écrite pour être opposable : G4 entre +3,5 et
  +4,3 pp, donc la première ligne du §5.** Motif : l'AWQ montre qu'à 4 bits la
  perte est de l'ordre du quart de point *sur les 252 matrices à la fois* ;
  sur une seule matrice de 2,6 % des poids, la perte devrait être une fraction
  de cela, donc l'essentiel du +4,48 devrait survivre.
- ⚠️ **Ce que vaut cette prédiction** : sur ce dossier l'auteur en a signé
  quatre, et **trois étaient fausses** (les deux du 08-25, le kill de M1, et
  deux tiers de celle de M2). Elle est là pour être opposable, pas pour être
  crue.
- ⚠️ **Et le motif a une faille connue** : l'AWQ quantifie *avec* une
  calibration qui protège les canaux saillants, là où ce bras applique un
  affine nu, groupe par groupe, sans aucune protection. Si la saillance de
  `v_proj` est concentrée, l'affine nu perdra davantage que l'analogie ne le
  laisse croire.
