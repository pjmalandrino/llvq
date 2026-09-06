# Pré-enregistrement — le 4B en Trio : le premier fichier `.llvq` du format, et sa qualité

**Rédigé le 2026-09-06. Statut : BROUILLON.** Il ne sera commité et tamponné qu'après deux choses :
le test de fumée du 0,6B (étape 4) vert, et la confirmation par l'opérateur des seuils du §5.
Vague 2, plafond **2,00 $**, dépensé 0,04 $ ; **ce travail coûte 0,00 $** — tout tourne sur le Mac.
Code mesuré : commit `__COMMIT__` (branche `f1/plancher-table`).

🚨 Une fois tamponné, ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

## §1 — Ce que c'est, et ce que ça décide

C'est **une porte**, la première de l'axe F sur un critère fondamental : la qualité du format Trio,
lue sur le 4B, sur perplexité et MMLU. Les planchers du 05 (table, décodeur compilé, trois
arithmétiques) ne décidaient rien ; celui-ci décide si Trio mérite les mesures de carte (F1d, F1e).

Le kill reste l'affaire de l'opérateur (`docs/METHODE.md` §1) : ce préreg lui donne la règle qu'il a
confirmée à l'avance, pas un verdict automatique.

## §2 — L'objet mesuré

Un artefact 4B quantifié en Trio : mot de 48 bits par bloc de 24 poids, `[p 1][r 1][s8 6][b1 1]
[i1 11][b2 4][i2 11][b3 1][i3 11][gain 1]`, table universelle de rangs de 16 Kio, format de fichier
v5 (`LVQ5`), kind `Trio` par matrice. Débit écrit **identique au servi** : 48 bits par bloc des deux
côtés, 2,159506 b/poids de projection, queue `KeepExact` inchangée
(*mesuré*, [fiche-4b](../docs/fiche-4b.md)).

Ce qui change par rapport au fichier servi, et rien d'autre : le codebook. Même corpus de
calibration, même rotation, même `nogs`, même graine, même nombre de fenêtres, même ρ = 1.

## §3 — Le protocole

Deux bras, la même machine, le même binaire, une seule variable.

```
  témoin :  LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=q4b-v1.llvq \
            cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke \
              -- 64 2048 12 4096 metal nogs leech1c12 999 rot
  Trio   :  le même, avec `trio` à la place de `leech1c12` et LLVQ_ARTIFACT=q4b-trio.llvq
```

Un seul bras est encodé. **Le témoin est le fichier publié**, `~/qwen3-4b-llvq.bin` : ppl 16,9415 et
MMLU 55,59 (*mesuré*, [fiche-4b](../docs/fiche-4b.md) §3.1, empreinte de tokens `3f1baca9033bf251`),
décision d'opérateur du 2026-09-06. Il a été calibré **sans graine**, donc sur le même préfixe contigu
de 131 072 tokens de C4 que la commande ci-dessus : le tirage de calibration est identique des deux
côtés, et ce n'était pas acquis — c'est vérifié dans fiche-4b et c'est ce qui rend le témoin publié
utilisable.

Puis, sur le bras Trio : `bin/seal`, puis la paire f16 de `fiche-4b` §3.1 ; et sur le fichier publié,
les mêmes commandes, ce jour, avec le même binaire :

```
  LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal <scellé>
  cargo run --release -p llvq-llm --features metal --bin mmlu -- <scellé> metal 40
```

## §3 bis — Le contrôle qui rend le témoin publié légitime, et son coût

Le fichier publié date du 2026-08-03. Douze commits ont touché `llvq-quant`, `llvq-core`, les modules
d'indexation de `llvq-search` et `llvq-llm/src/calib.rs` depuis (*mesuré*, `git log`). Aucun n'est
censé changer ce que `leech1c12` écrit — les boutons de M1 et M2 sont désactivés par défaut, le design
C est un drapeau, et la traduction en anglais est prouvée identique sur 127 fichiers sur 128 — mais
« censé » n'est pas un contrôle, et comparer un Trio d'aujourd'hui à un fichier d'août mêlerait sinon
le format et quatre mois de dépôt.

**Contrôle 0, à passer AVANT l'encodage Trio, ~10 min de Mac :**

```
  LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=<scratch>/q4b-drift.llvq \
    cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke \
      -- 64 2048 12 4096 metal nogs leech1c12 1 rot
```

Un seul bloc transformer, donc les sept matrices du bloc 0. Leurs enregistrements doivent être
**identiques octet pour octet** à ceux du bloc 0 du fichier publié, une fois retirée la différence
d'en-tête (le publié est scellé, celui-ci est un artefact d'encodage : la comparaison porte sur les
octets d'enregistrement, pas sur le fichier).

- Identiques : le chemin de quantification n'a pas bougé, et 16,9415 / 55,59 sont un témoin propre.
  C'est la lecture attendue.
- Différents : la dérive est exposée, avec sa taille. Aucun chiffre Trio n'est publié avant que
  l'opérateur ait arbitré entre rejouer le témoin (4 h) et caractériser la dérive.

⚠️ Ce contrôle prouve que **l'encodeur** n'a pas bougé. Il ne prouve pas que **l'évaluation** n'a pas
bougé ; c'est le contrôle 2 du §4 qui s'en charge, en rejouant `ppl` sur le fichier publié ce jour.

## §4 — Les contrôles, et si l'un tombe aucun chiffre n'est publié

1. **Oracle d'abord**, sur les deux backends (règle dure 10) : `bin/oracle` sur le 4B, Metal et CPU.
2. **Le fichier publié rejoue 16,9415** ce jour, avec le binaire de ce jour (§3) : c'est ce qui
   sépare une dérive de l'évaluation d'une différence de format. S'il dérive, aucun chiffre de ce
   préreg n'est lisible.
3. **Même débit des deux côtés** : la ligne « effective rate » de `smoke` sur le bras Trio donne
   2,159506 b/poids au dix-millionième, le chiffre du fichier publié (*mesuré*, fiche-4b §4). Un
   écart signifie que la queue ou les échelles ont bougé, donc que la comparaison n'est plus à débit
   constant.
4. **`verify_artifact` bit pour bit** sur le bras Trio : les poids relus du fichier égalent les
   poids évalués pendant l'encodage.
5. **Empreinte de tokens identique** sur les quatre évaluations (ppl et MMLU, Trio et publié).
6. **Tout point Trio est dans Λ₂₄** : la garde de `Trio::encode` refuse à l'écriture ; zéro refus
   attendu, et un seul refus arrête l'encodage.
7. Le fichier Trio porte le magic `LVQ5`, le kind `Trio`, l'empreinte v1 **inchangée** et l'empreinte
   Trio épinglée.

## §5 — La règle de décision, à confirmer par l'opérateur AVANT le tampon

Repères servis (*mesuré*, `docs/ETAT.md` §2) : ppl 16,94, MMLU micro 55,59, f16 70,32.
Bruit de tirage de calibration au 4B : σ = 5,2 % en ppl, 2,92 pp en MMLU (*mesuré*, §4 de `ETAT`).
Bruit d'un A/B à fichier constant : ±0,12 % en ppl, 0,43 pp en MMLU (*mesuré*, kvq8-4b).
Ici les deux fichiers partagent le tirage de calibration (aucune graine, même préfixe contigu) mais
pas le codebook. Le bruit de tirage ne s'applique donc pas au premier ordre ; il reste comme borne
haute de ce que le contrôle 0 ne couvre pas, et c'est à ce titre qu'il figure dans la table.

| résultat, Trio contre le fichier publié | suite |
|---|---|
| ppl ≤ +2,5 % **et** MMLU ≥ −1,5 pp | Trio est adopté sur la qualité ; F1d et F1e s'écrivent |
| ppl ≤ +5,2 % (un σ de tirage) **et** MMLU ≥ −2,92 pp (un σ) | dans le bruit : Trio n'est ni adopté ni tué ; une seconde graine tranche (4 h de Mac de plus par bras) |
| ppl > +10,3 % (l'étendue de trois tirages) **ou** MMLU < 53 % en absolu (la ligne F1e) | Trio perd trop pour ce qu'il rend ; l'opérateur arbitre contre la VRAM divisée par deux |
| autrement | non tranché, décision d'opérateur |

**Ce que la table ne fait pas** : elle ne compare pas Trio à AWQ ni à QTIP. Le gain de Trio est la
VRAM (2,76 contre 5,162 b/param, *calculé*) et le débit projeté ; sa qualité se juge contre le format
qu'il remplace, sur le même tirage.

## §6 — Prédiction signée, opposable

**ppl entre +1,5 % et +4 % du témoin, valeur centrale +2,5 %** ; **MMLU entre −2,5 et 0 pp,
valeur centrale −1,2 pp**.

Motif : la rétention gaussienne de Trio est 88,89 % contre 92,00 pour la boule-12 servie
(*mesuré*, `docs/mesures/f1-encodeur-blocs-reels-2026-09-05.txt` sur des blocs réels du 0,6B :
88,68 contre 91,87, le même écart). Cela fait +8,5 % de MSE de quantification. Sur le 4B, l'excès de
log-vraisemblance du format servi est 0,3254 nats contre le f16 (*calculé*, fiche-4b §3.1) ; si
l'excès suit la MSE, +8,5 % d'excès donne +2,2 % de perplexité. La relation n'est pas établie — c'est
la partie faible de cette prédiction, et le dépôt a déjà mesuré qu'elle est lâche (le papier lit
17,05 ppl pour 60,7 % de MMLU là où nous lisons 16,94 pour 55,59).

**Ce qui la rendrait fausse de façon instructive** : ppl sous +1 % — la rétention gaussienne
surestimerait la perte sur des poids réels, et la boule-12 aurait été payée trop cher pendant un an ;
ou MMLU sous −4 pp pour une ppl dans la fourchette — MMLU verrait quelque chose que la perplexité ne
voit pas, ce qui rouvrirait M2 (l'attribution par type de projection) sur Trio.

⚠️ Historique des prédictions signées de ce dossier au 2026-09-05 : F1b juste à 0,05 point ; le
plancher de table juste sur deux nombres et faux sur le troisième ; le plancher compilé faux sur
trois nombres sur cinq ; les trois arithmétiques fausses sur les quatre temps, dans le sens que la
clause instructive nommait. Celle-ci est opposable, pas crédible d'avance.

## §7 — Le coût, et ce qui n'est pas mesuré ici

~10 min pour le contrôle 0, ~4 h d'encodage Trio, ~2 h d'évaluation des deux côtés :
**~6 h de Mac, 0,00 $.** Le témoin publié n'est pas réencodé (décision d'opérateur du 2026-09-06),
ce qui épargne 4 h et déplace la charge de la preuve sur le contrôle 0.
Ne sont pas mesurés ici : le débit en tokens par seconde, la VRAM sur carte, la classe de modèle.
Ce sont F1d et F1e, sur carte, après cette porte.
