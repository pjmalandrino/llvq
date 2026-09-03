# BACKLOG — état au 2026-08-25

> Le RAF courant du projet, ordonné par ce qui décide chaque sujet et non par
> thème. Établi par balayage des 53 commits depuis le 2026-08-18, des onze
> journaux de mesure postérieurs, des quatre documents de plan et du registre
> de coûts. Établi à HEAD `e21a8bb` ; **mis à jour au fil de la journée du
> 2026-08-25** (branche `docs/etat-courant-2026-08-25`).
>
> 🧭 **Ce document remplace la liste de tête de [`PLAN.md`](PLAN.md)** comme RAF
> courant. `PLAN.md` reste le document de niveau projet (les trois phases et
> leurs gates) ; [`HISTORIQUE.md`](HISTORIQUE.md) reste le fil chronologique.

---

## 🧭 Reprise — état au SOIR du 2026-08-25

**Ce bloc est le point d'entrée d'une session neuve.** Il périme la lecture
« matin » du reste de ce document sur un point : la piste calibration, que
§0 et la liste donnaient pour close, est **rouverte et en cours de mesure**.

### Ce qui s'est passé dans la journée

**L'échelle des bits de gain est RÉFUTÉE, et par sa propre réplication.** Quatre
codebooks à débit constant (48 bits/bloc) mesurés au 0.6B pleine profondeur, à
deux tirages de calibration — journal
[`mesures/gain-ab-gate-0.6b-2026-08-25.txt`](mesures/gain-ab-gate-0.6b-2026-08-25.txt),
**0 $**. Le classement s'inverse complètement d'un tirage à l'autre ; un seul
bras bouge de **13,9 %** en ne changeant que le texte de calibration, là où
l'écart entre les quatre bras n'était que de 10,6 %. **Le bruit dépasse le
signal.** Rien n'est adopté, aucun étage 2 au 4B n'est justifié, et le sujet de
papier envisagé sur ces chiffres est mort.

Deux prédictions signées de l'assistant, **fausses toutes les deux**. Trois
hypothèses réfutées dans la journée : le codebook de gain biaisé, le biais
radial comme explication, puis l'effet lui-même.

**Ce qui survit du lot** : le **biais radial** mesuré par `bin/cosdiag` — le
code de gain quantifie `‖w‖` là où l'optimum à direction fixée est la
projection `⟨w, û⟩`. ~3,7 % sur la configuration servie, de la géométrie pure,
indépendant de tout tirage. **Piste de qualité gratuite, non traitée.**

### Ce qui est en cours, et pourquoi

Le résultat ci-dessus a rouvert une piste que le dossier donnait pour close.
La borne qui avait enterré la calibration le 2026-08-06 (l'« oracle », −1,6 %)
a été mesurée sur **3 blocs de Qwen3-0.6B**, en **perplexité**. Or notre
déficit est sur **MMLU**, à la **taille publiée**. La piste n'a donc jamais été
testée là où elle compte.

Le mécanisme se chiffre : sur `down_proj` du 4B, la hessienne fait 9 728² et on
l'estime sur 131 072 tokens — **13,5 exemples par dimension**. Le papier amont
en utilise ~96× plus.

| | | statut |
|---|---|---|
| **1b — bruit de MMLU entre tirages, au 4B** | job `6a8de89b984507d9db4e4664`, `l40sx1`, ~0,5 $ | 🔄 **lancé le 2026-08-25 au soir** |
| **2 — échelle de volume de calibration** | ×8 → ×32 → ×96, `rtx-pro-6000` | ⏸️ **gaté sur la sortie de 1b** |

**1b** passe MMLU sur les trois artefacts 4B des graines F5, retrouvés dans le
bucket (`f5-graines-2026-08-19/seed{1,2,3}/`) — **0,5 $ au lieu de ~21 $** de
requantification. Quatrième fois que la règle des canaux de rétention paie.
Il produit **s**, l'écart-type MMLU entre tirages, qui n'a jamais été mesuré
(F5 n'avait chronométré que la perplexité).

**2** est intégralement pré-enregistré et tamponné AVANT tout lancement :
[`proofs/preregistration-volume-calibration-2026-08-25.md`](../proofs/preregistration-volume-calibration-2026-08-25.md)
(sha256 `33fd4932…`). Une seule variable, préfixes **emboîtés** ×1 ⊂ ×8 ⊂ ×32 ⊂
×96 du même texte C4, témoin déjà payé (l'artefact publié, MMLU 55,59), cinq
contrôles, et une règle d'arrêt qui borne le pire cas à **deux barreaux et
~19 $**.

🚨 **Le barreau de départ N'EST PAS AU CHOIX** — il est décidé par `s` :

| | départ |
|---|---|
| `s ≤ 1,0 pp` | **×8** (2,9 h, ~8 $) |
| `1,0 < s ≤ 2,0 pp` | **×32** (3,8 h, ~11 $) — ×8 serait illisible |
| `s > 2,0 pp` | **aucun** — le design est à repenser |

Coûts et durées *estimés* par un modèle calibré sur le profil mesuré du 4B
(222,5 s fixes par bloc + 4,06 s par bloc et par unité de volume), qui
**reproduit à ×1 les 155 min facturés** — c'est sa seule validation.

⚠️ **Une inconnue reste, dans les contrôles** : le shard C4 a-t-il ~50 Mo de
caractères pour ×96 ? `smoke` imprime ce qu'il a réellement lu ; si ça
plafonne, le barreau se publie **à son volume réel**. Sans effet sur ×8 ni ×32.

### Deux choses qui n'appartiennent qu'à l'opérateur

1. **Corriger la fiche de soumission ScholarOne** — elle porte « MALANDRINO,
   MALANDRINO » avec une affiliation parasite, et c'est ce qui serait publié.
2. **Transmettre la page de titre corrigée au bureau éditorial.** La committer
   ne l'envoie pas.

---

## 0. Le cadre — ce que le projet a produit, et la seule question qui reste

Il faut écrire ce paragraphe avant la liste, parce qu'il change ce que la liste
vaut.

**Sur l'axe qu'il visait, le projet a perdu, et la mesure du 2026-08-21 le dit
sans ambiguïté.** Le concurrent vivant de la ligne 2 bits n'est pas QuIP#/E8P —
que nous battons largement, 16,9617 contre 21,15 de perplexité et 55,59 contre
48,6 de MMLU — c'est **QTIP**, et QTIP nous bat sur les quatre axes mesurables :

| | QTIP (3INST) | nous (`Planes14`) |
|---|---|---|
| b/poids en VRAM | **2,000** (formes divisibles, ni queue ni échelle) | 4,804 |
| temps du noyau, L40S, même processus | **2,246 ms** | 5,103 ms |
| MMLU micro, Qwen3-4B, 2 bits sans FT | **57,4** *(papier source)* | 55,59 *(mesuré chez nous)* |
| excès de log-vraisemblance vs sa propre baseline | **0,3171** | 0,3268 |

**Et le mécanisme est structurel, pas un défaut d'implémentation.** La boule
Λ₂₄(12) porte 1,1·10¹⁴ points. C'est *exactement* ce qui la fait mieux
quantifier qu'E8 — et c'est *exactement* ce qui la rend intabulable. Un
décodeur a deux voies : lire une table, ou calculer. QTIP lit une table de
2 KiB ; nous ne le pouvons pas, à aucune échelle de matériel. Restent deux
options, et les deux sont mesurées :

- **décoder la forme compacte** (2,000 bits) → E1v rend **0,25× FP16**, soit
  quatre fois plus lent que le FP16 lui-même ;
- **déplier pour rendre le décodage trivial** → `Planes14`, 4,804 b/poids,
  donc **2,4× les octets de QTIP** — et un GEMV à batch 1 est borné par les
  octets lus.

L'avantage et le coût ont donc la **même cause**, et on ne peut pas garder l'un
sans l'autre. Une boule de Leech assez petite pour tenir en table ne serait plus
un code de Leech : la coquille 12 seule porte encore 7·10¹³ points, et le banc
G4 mesure qu'elle quantifie *moins bien* que la boule (90,34 % contre 92,72 %
de rétention).

🚨 **Le gate de sortie du projet, relu avec ce qu'on sait aujourd'hui, ne passe
pas.** G5 était écrit d'avance : *« Si LLVQ ne bat pas QuIP#/QTIP en perplexité
sur Qwen3-4B, toute la thèse tombe et il faut le dire, pas optimiser un noyau
pour une méthode qui ne tient pas ses promesses. Wiki < 17,04. »* Nous rendons
16,9617 — vert de **0,46 %**. Or F5 a mesuré la dispersion inter-graines de
calibration à **σ = 5,2 %** sur exactement cet objet. La marge vaut un onzième
de σ, [`fiche-4b.md`](fiche-4b.md) la qualifie déjà de « pas défendable », et
elle est acquise en payant **2,17 b/poids écrits contre 2,000**.

**Ce qui survit, et qui n'est pas rien.**

1. **Un résultat négatif mesuré sur toute une famille de méthodes.** Combien
   coûte un codebook trop grand pour être tabulé, mesuré contre un concurrent
   qui tabule, dans le même harnais et sur la même carte. Personne n'avait ce
   chiffre — le papier source déclare son propre noyau *« plus lent que QTIP »*
   et range l'optimisation bas niveau comme *« largement orthogonale »* à sa
   contribution. C'est le papier soumis, et c'est sa contribution réelle.
2. **Le codebook, sur son propre axe.** À débit de code apparié, Λ₂₄
   multi-coquilles quantifie nettement mieux qu'E8P. C'est un énoncé sur les
   *codes*, pas sur les noyaux, et il tient quel que soit le noyau — donc
   partout où le coût de décodage ne borne pas : **disque, transport,
   archivage**, où E1v tient sa promesse au bit près (1,09 Go lus contre 2,18,
   bijection prouvée sur 150 681 600 blocs).
3. **Une courbe d'échelle non tranchée.** Le déficit MMLU fond avec la taille
   (−14,73 → −10,56 → −6,85 pp) et l'écart au 4 bits aussi (14,45 → 7,49 →
   6,09 pp). Le dernier palier n'est pas résolu (p = 0,40) et p = 0,40 ne
   prouve pas l'égalité : **les données sont muettes**, pas concluantes.
4. **L'instrument.** Pré-enregistrements horodatés, gates posés avant la
   mesure, provenance étiquetée, preuves de décodage bit-pour-bit, assertions
   tuées par mutation. Il a attrapé ses propres erreurs une dizaine de fois, y
   compris deux cette semaine.

**La seule chose qui rouvrirait la question produit est la qualité.** Si un
levier referme 4 à 6 pp du déficit MMLU, alors à 14B/32B on tient un 2 bits à
1-2 pp du 4 bits pour 5 à 10 % de mémoire en moins : un arbitrage réel. Sinon,
la conclusion honnête est déjà écrite plus haut, et le §2 de ce backlog est le
dernier chapitre. **Le §4 (noyau) ne rouvre rien** — il est conservé parce que
deux de ses items sont du matériel de révision, pas parce qu'ils changeraient un
verdict.

---

## Vue d'ensemble

| § | bloc | items | coût | ce qui le décide |
|---|---|---|---|---|
| 1 | Fenêtre courte | 1 | 0 $ | se périme en jours |
| 2 | Dette de vérité | 7 (**5 soldés**) | 0 $ | aucune raison d'attendre |
| 3 | Qualité | 3 (**§3.1 étage 1 mesuré**) | 0 → 15 $ | **le seul axe qui bascule le verdict** |
| 4 | Noyau | 4 | ≤ 2,5 $ | borné — matériel de révision |
| 5 | Échelle et produit | 4 | ~73 $ | go budget explicite |
| 6 | Attente éditeur | 2 | ~43 $ | lettre de décision TACO |
| 7 | En pause | 3 | — | décision opérateur |

---

## 1. Fenêtre courte

### 1.1 Fiche ScholarOne — nom d'auteur et affiliation parasite

La soumission porte « MALANDRINO, MALANDRINO » : le prénom n'a jamais été saisi.
S'y ajoute une seconde affiliation parasite (institution vide, MACAU 33460 FR,
adresse personnelle). Invisible des relecteurs en double aveugle, mais **c'est
la fiche qui serait publiée**. `main.tex` et `titlepage.tex` portent tous deux
le bon nom.

- **Coût** — 0 $, un mail au bureau éditorial (adresse dans le mail de confirmation)
- **Fenêtre** — jusqu'au camera-ready, mais rien ne gagne à attendre

> ✅ **Réglé le 2026-08-25** : la visibilité du dépôt. Décision opérateur — le
> dépôt **reste public**, et la page de titre a été modifiée pour l'expliquer à
> l'éditeur (manuscrit anonymisé, URL données pour que l'artefact soit
> localisable sans que le manuscrit brise sa propre anonymité). Toute phrase du
> dossier qui suppose « dépôt privé pendant la revue » est périmée.

---

## 2. Dette de vérité — 0 $

Aucun document du dépôt ne portait l'état courant : `HISTORIQUE.md`, `PLAN.md`
et `CLAUDE.md` s'arrêtaient tous au 2026-08-18, pendant que onze lots de mesure
en renversaient deux affirmations structurantes. **Cette dette a coûté une
erreur de raisonnement documentée cette semaine** — une session a conclu que
QTIP ne nous battait pas en vitesse en lisant `PLAN.md`, quatre jours après la
mesure qui dit le contraire.

| # | item | état |
|---|---|---|
| 2.1 | Retirer le plafond de 4,77× FP16 (11 lignes, 5 documents) | ✅ fait le 2026-08-25 |
| 2.2 | Propager le σ de F5 (0,7 % → 5,2 %) et ses trois réserves | ✅ fait le 2026-08-25 |
| 2.3 | Rattraper `HISTORIQUE.md` (9 campagnes + la soumission) | ✅ fait le 2026-08-25 |
| 2.4 | Deux affirmations démenties encore publiées | ✅ fait le 2026-08-25 |
| 2.5 | `ots upgrade` des 16 ancrages | ✅ **fait le 2026-08-25** — les 16 portent leur attestation Bitcoin |
| 2.6 | Tag + `.ots` sur la version soumise | ☐ non fait |
| 2.7 | Armer les gardes déjà écrits (CI Linux, `check_tables`, manifeste) | ☐ non fait |
| 2.8 | Inventorier le bucket HF | ☐ non fait |

### 2.5 `ots upgrade` — cinq minutes, et c'est la revendication centrale du dossier

Aucun des 16 ancrages n'a jamais été upgradé : tous portent **4
`PendingAttestation` et 0 `BitcoinBlockHeaderAttestation`**. Les quatre
calendriers *détiennent* les attestations et les servent à la demande — elles ne
sont simplement jamais redescendues dans les fichiers commités. Tant qu'elles
n'y sont pas, « vérifiable sans nous faire confiance » dépend de la survie de
quatre serveurs tiers, y compris pour `p1` et `p1c`, les deux documents que le
dossier cite comme exemplaires. Le plus ancien tampon a quinze jours, très
au-delà du délai de confirmation Bitcoin.

- **Gate** — `ots info` rend ≥ 1 attestation Bitcoin et 0 pendante sur chacun des 16
- **Coût** — une commande, puis un commit des fichiers grossis

### 2.6 Tag et tampon sur la version soumise

`git describe` rend `paper-v2-68-ge21a8bb` : **aucun tag ne couvre le commit
soumis**, et aucun des 16 `.ots` ne couvre le papier. La version envoyée à TACO
n'est identifiée que par un SHA consigné hors dépôt. Le jalon 4 du plan TACO
l'exigeait avant gel.

### 2.7 Armer les gardes qui existent déjà

Trois filets sont écrits et aucun n'est invoqué.

- **Typecheck Linux** — `ops/devtools/nvcc` fonctionne (0 erreur, 0 warning
  clippy sur `x86_64-unknown-linux-gnu`) mais aucune automatisation ne
  l'appelle, et la CI exclut toujours `llvq-cuda` et `llvq-metal`. Le trou qu'il
  ferme a déjà coûté une journée : l'image a été **incompilable du 08-15 au
  08-16 sans que personne puisse le savoir**. Étendre à `llvq-llm --features
  cuda` couvrirait en plus les 30+ sites `cfg(linux+cuda)` de `fused_cuda.rs`.
  *Gate : la CI rougit sur un warning de la cible linux, létalité prouvée par
  mutation.*
- **`check_tables.py`** — `tab:lit` et `tab:attribution` demeurent déclarées non
  couvertes. C'est une brèche dans le contrat chiffres→CSV→journaux que le
  papier revendique comme argument de soumission.
- **`ops/manifest.jsonl`** — une entrée pour une centaine de nombres au papier.
  L'outil est complet depuis le 2026-08-10 et implémente la règle que le hachage
  seul ne voit pas : la valeur déclarée doit se retrouver littéralement dans son
  log. *Trancher d'abord `ops/` ou `proofs/` comme emplacement faisant foi.*

### 2.8 Inventorier le bucket HF

69 fichiers, 46,7 Go, jamais inventoriés depuis la création le 2026-08-02, et le
compte a grossi depuis (F5 y a déposé trois artefacts scellés, B3 y a rescellé
le 8B). Le bucket a déjà sauvé deux « pertes » devisées contre rien : les dumps
MMLU du 14B (une campagne à refaire → 579 ko de bande passante) et l'artefact
14B scellé (~9 min contre 27,67 $ et 302 min de requantification).

⚠️ Ce n'est pas une garantie de récupération : le **8B scellé** a été cherché aux
deux endroits et il est réellement perdu — le bucket n'en héberge que la version
projections seules.

---

## 3. Qualité — le seul axe qui bascule le verdict

Le point dur est inchangé et il est ancien : **−14,73 pp de MMLU au 4B**,
−10,56 au 8B, −6,85 au 14B, contre −0,28 pp pour l'AWQ 4 bits au 4B. Le profil
par matière dit le mécanisme : algèbre abstraite et comptabilité tombent au
hasard pendant qu'histoire et droit tiennent au-dessus de 80 %. **Le 2 bits
abîme le raisonnement bien plus que la restitution.**

🚨 **Réserve transverse, et elle est neuve.** Toute expérience qui **recalibre**
tombe désormais sous la barre de F5 — **σ = 5,2 %, étendue 10,3 %** — et non
sous les 0,7 % hérités du lot B. Les gates ci-dessous sont écrits contre
l'ancienne barre et doivent être relus avant lancement. Ce qui n'est *pas*
touché : les A/B à fichier constant, qui gardent leur barre appariée à ±0,12 %.

### 3.1 A/B du partage des 48 bits — ✅ ÉTAGE 1 MESURÉ le 2026-08-25

> ✅ **Gate rendu, 0 $, 86 min.** Les quatre contrôles passent — iso-débit
> **2,1656 b/poids aux trois bras**, la valeur calculée et écrite avant le
> lancement. Résultats, baseline 19,5038 :
>
> | bras | gain | boule | ppl | vs témoin |
> |---|---|---|---|---|
> | `leech0c13` | 0 | 13 | **39,3309** | **−9,56 %** |
> | `leech1c12` ← servi | 1 | 12 | 43,4865 | — |
> | `leech2c11` | 2 | 11 | **39,5350** | **−9,09 %** |
>
> **Les deux candidats sont VERTS**, et le codebook **servi ressort dernier des
> trois** à débit strictement identique. Le classement est en U : les deux
> configurations spécialisées battent celle qui partage. Aucun mécanisme n'est
> revendiqué, et le repère gaussien du banc G4 pointe en sens inverse (92,14 %
> de rétention pour le témoin contre 88,90 %) — exactement la réserve que le §6
> de `CLAUDE.md` posait d'avance sur des poids non gaussiens après GPTQ.
>
> 🚨 **Le gate ne peut RIEN adopter** (§0.1 du pré-enregistrement) : un bras
> vert a le droit d'être mesuré au 4B, il n'est pas meilleur. Quatre réserves,
> dont celle qui décide — **R1, un seul tirage de calibration**, quand F5
> mesure σ = 5,2 % entre graines.
>
> Journal :
> [`mesures/gain-ab-gate-0.6b-2026-08-25.txt`](mesures/gain-ab-gate-0.6b-2026-08-25.txt),
> logs bruts dans `mesures/gain-ab-2026-08-25-brut/`.
>
> **Suite recommandée, avant l'étage 2** : rejouer les trois bras à une seconde
> graine — ~90 min, 0 $, contre ~24 h pour deux runs 4B. C'est la seule des
> quatre réserves qui peut retourner le signe. Demande son propre
> pré-enregistrement, celui du 08-25 étant scellé sur un tirage.
> ⚠️ Et si l'étage 2 élit un candidat, la suite n'est pas « refaire le 4B » mais
> « refaire le 4B **et** rouvrir le layout runtime » (C6) : le chemin servi gèle
> le champ de gain à un bit sur huit assertions et quatre shaders.

<details><summary>Le raisonnement d'origine, conservé — il est intact</summary>

**Le choix de 1 bit de gain n'a jamais été fondé par un A/B au niveau modèle, à
aucune taille.** Il vient d'un argument de débit — plafonner à Λ₂₄(12) fait
tomber l'index à 47 bits, « ce qui paie le bit de gain au même débit total » —
et d'une table de distorsion sur source gaussienne (Table 8 du papier source).

Or le papier donne **trois réponses contradictoires selon le protocole** :

| protocole | optimum |
|---|---|
| Table 8, source gaussienne | **1 bit** (le nôtre) |
| Table 6, perplexité LLM | **2 bits** (15,54 contre 17,05) |
| Table 6, MMLU LLM | **0 bit** — **60,7** contre 59,3 |
| Annexe I, sous Spherical GPTQ | **0 bit** |

**60,7 est le meilleur MMLU 2 bits sans fine-tuning de toute leur table, devant
QTIP (57,4), et il est 5,1 pp au-dessus de nos 55,59.** Notre configuration
n'a *aucun* équivalent dans la Table 6 : elle n'existe que dans la table de
distorsion.

Trois raisons de la faire en premier :

- **Iso-débit.** Le triplet à 48 bits/bloc est {cap 13 + 0 bit, cap 12 + 1 bit,
  cap 11 + 2 bits}. L'expérience ne coûte pas un bit, ne touche ni la VRAM ni la
  comparaison AWQ.
- **Aucune ligne de Rust.** Le paramètre est câblé du quantifieur jusqu'à
  l'archive : `fit_gain_centroids` accepte `k_bits ∈ {0,1,2}`, la CLI parse
  `leech0c13` et `leech2c11`, le format dérive `gain_bits` de
  `centroids.len()`, et `bin/ppl`/`bin/mmlu` scorent le fichier tel quel. Seul
  le chemin **fusé** est gelé à 1 bit (8 assertions + 4 shaders), ce qui
  n'empêche pas de mesurer la qualité.
- **Le bras 1 bit est déjà en magasin** (16,9415 f16 / MMLU 55,59) : deux bras
  neufs suffisent.

**Coût** — 0 $ : ~2 h de gate, ~8 h de requantification locale (4,01 h par bras
sur M3 Max), ~1 h 40 de MMLU.
**Gate** — à profondeur sur 0.6B (28 blocs) **obligatoire** : le bit de gain
*est* le code de magnitude, et la règle du dossier l'exige.
⚠️ **Réserve** — la seule mesure voisine (bras à magnitude f16 libre, meilleur
de 3,17 % sur 3 blocs) a exactement la signature « proxy local meilleur », qui
s'est retournée deux fois à pleine profondeur (`group_scales`, design C). Le
gate n'est pas une formalité.
⚠️ Le bras témoin du script de gate tourne sur `leech` (boule 13, 1 bit) et non
sur `leech1c12` : à ré-ancrer avant.

</details>

### 3.2 D3 — recalibrer sur DCLM-edu

Le suspect que le papier nomme lui-même en limitations : la source calibre sur
un corpus curé éducation-raisonnement, nous sur C4. Le mécanisme prédit
exactement la signature observée — **l'excès de perplexité reproduit la source
(×1,38 contre ×1,37) pendant que le déficit MMLU ne la reproduit pas**, ce
qu'une calibration curée raisonnement laisserait. Les trois autres suspects sont
bornés : rotation de sortie (marginale dans l'ablation source), design C
(×1,99 sur proxy), volume (oracle −1,6 %, ×13 de volume −1,2 % — et ces deux-là
tombent désormais **sous** le plancher de bruit F5).

- **Coût** — ~15 $ estimé, aucun devis dans le dépôt
- **Gate** — aucun posé ; à pré-enregistrer **et ancrer** avant lancement
- **Bloqué par** — gardé délibérément comme l'expérience du tour de *Major
  revision*, que TACO n'accorde qu'une fois

### 3.3 Compensation bas-rang post-hoc (EoRA / Recover-LoRA)

Le plus gros gain publié dans la littérature — **+4 à 11 pp de MMLU** — jamais
tenté ici, et la conclusion du papier le nomme comme non testé. Design : par
couche, un adaptateur `A·B` de rang *r* ajusté sur le résidu `W − Ŵ` dans la
métrique hessienne (les hessiennes de calibration existent), servi comme
correction additive f16 à côté du chemin fusé.

- **Coût** — jours d'ingénierie, pas de dollars annoncés
- **Gate** — refermer **≥ 4 pp** du déficit 4B (la moitié basse du publié), en
  apparié, **dans un budget d'octets fixé d'avance : ≤ 0,25 b/param modèle
  entier**, soit *r* ≈ 16 sur les projections du 4B. Sans ce budget on rachète
  la qualité en octets et toute la comparaison AWQ est à refaire.

---

## 4. Noyau — borné, et conservé comme matériel de révision

⚠️ **Lire le §0 avant ce bloc.** Aucun de ces items ne rouvre le verdict : même
à 100 % de sa borne d'octets, `Planes14` plafonne à 16/4,804 = **3,33× FP16**,
sous l'AWQ (3,38×) et à **0,68× QTIP**. Ils sont ici parce que deux d'entre eux
sont du matériel de réponse aux rapporteurs.

### 4.1 Géométrie de lancement — l'ex-« famille k », requalifiée

Le poste que `nullk` désignait **n'est pas un sol matériel** : QTIP tourne
dessous dans le même processus (2,246 ms contre 2,306). C'est le coût de *notre*
géométrie — un warp par ligne de sortie, 252 lancements — et **D1 en a déjà
repris une part** en fusionnant à 144 lancements (×1,061, 100,6 tok/s). La
conclusion du papier le désigne explicitement : *« a successor should attack
launch geometry and unfolding cost, not the factor against FP16. »*

La famille *k* (le même noyau servant *k* colonnes par lancement) reste le seul
levier écrit pour le reste, et elle n'est pas codée.

> ✅ **Rendu le 2026-09-01 → 09-02 (phase A du plan d'après-dépôt).** La part
> de ce poste que la géométrie pouvait rendre est **mesurée et bornée** :
> A2 (CUDA Graphs, hybride) **adopté au 4B, +13,45 % [13,36–13,58]** et au 8B
> (+10,1 %), point de courbe au 14B ; A3 (huit variantes d'occupation au
> banc, `mesures/a3-occupation-banc-2026-09-01.txt`) : **aucun bras portable
> ≥ 10 %**, le meilleur (`pers`) à +1,56 %, et le bras de banc `persall`
> (un lancement par round) borne le matvec fusé à **+26,36 %**, ce que les
> graphs ont déjà encaissé sur le chemin servi. ⚠️ **Le sous-remplissage de
> o/down n'est PAS le résidu** — un split-K qui porte leurs grilles à
> 640/1 280 CTAs rend −1,87 %. Ce qui reste ici après A2 vaut ~1,6 % ; la
> famille *k* garde son statut (non codée, garde produit à *k* > 1), sans
> que rien de la phase A ne l'ait rendue plus urgente.

- ⚠️ **Garde produit à poser avant d'écrire** — elle n'amortit qu'à *k* > 1,
  donc en prefill et en lot ; à *k* = 1 (chat interactif) le plancher reste
  entier, et un verdict de *k* **ne se transporte pas** au débit interactif.
- **Coût** — code sur Mac 0 $, puis job mutualisé **0,8-1,0 $**, pire cas 2,70 $.
  Le « 0,3-0,5 $ » qui circule est faux d'un facteur 2 : tout job `planesbench`
  à ≥ 5 bras paie 1 468-1 481 s de transcodage hôte avant le premier round.
  `--timeout 90m` à poser explicitement.
- **Gate** — K1 se lit sur le **rapport vs FP16**, jamais sur le temps · K2 par
  colonne : `T(k=8)/8 ≤ 0,60 × T(k=1)` · K3 zéro spill aux six sites
- **Bloqué par** — `ots stamp` sur `proofs/preregistration-p4-2026-08-14.md`,
  que le pré-enregistrement exige de lui-même avant le premier noyau *k* ; et le
  §7bis (resté vide) doit consigner les deux dérogations déjà commises

### 4.2 Banc E1c — seuils à ré-ancrer avant tout chronomètre

Le seul layout jamais dispatché par un banc, à aucune largeur. L'exactitude est
acquise (sweep intégral de 150 681 600 blocs) ; la question est un **motif de
lecture**, pas un décodage — il n'hérite donc pas du verdict d'E1v, qui est mort
d'être borné en calcul.

🚨 **Les trois seuils X3 sont invalides en l'état** : posés en comptabilité **non
alignée** (3,7618 b/poids), celle que le chemin servi ne peut pas lire, alors
que le bras à mesurer lit le flux **aligné** (4,2880, +14 % d'octets). Lancer
sans les amender publierait « E1c est lent » alors qu'on aurait mesuré un
désalignement. Dans la comptabilité alignée, seul « ≥ 2,05× contre `Planes14` »
garde un sens.

- **Coût** — ~0,2 $ mutualisé avec le job de la famille *k*
- **Bloqué par** — ré-ancrage, décision en souffrance depuis le 2026-08-15

### 4.3 QTIP sur A100

La table A100 n'a **aucun point 2 bits concurrent** et le papier le dit. Le
mécanisme est prêt : `LLVQ_NVRTC_ARCH=compute_80` a déjà tourné deux fois.

- **Coût** — ≤ 1,20 $ annoncé au pré-enregistrement F2
- **Bloqué par** — go de dépense ; décision laissée hors du préreg F2

### 4.4 Contrefactuel LUT

Chronométrer dans notre harnais un codebook de réseau qui **tient** en table —
E8P de QuIP# (2¹⁶ entrées) ou les IQ2 de llama.cpp (256 à 1 024 points). C'est
le test direct de la thèse du §0. Le papier le nomme et l'écarte lui-même :
*« timing them would confirm that side rather than test the unfolding cost »*.
AQLM et VPTQ sont dans le même cas, cités et non mesurés.

- **Coût** — non chiffré ; repère : le bras QTIP entier a coûté 1,44 $

---

## 5. Échelle et produit

### 5.1 Rejouer la fusion au 8B et au 14B

D1 n'a mesuré la fusion qu'au 4B, et le journal le déclare. **La table à trois
tailles repose sur une configuration identique partout** (`ROT_SHARE=0`,
`FUSE=0`) — propriété qu'elle *utilise*, et qu'un 4B fusé isolé casserait. Donc :
les trois tailles rejouées, ou aucune.

- **Coût** — ~1,90 $ calculé sur les jobs B2 comparables (0,63 $ au 8B, 1,27 $ au 14B)
- **Gate** — les six critères de D1 (C1, C2, C3, L1, M1, V1) sont réutilisables tels quels

### 5.2 `Planes12x` servi au 8B et au 14B

G3 l'a mesuré servi bout-en-bout **au 4B seulement** : 85,0 tok/s [84,7–85,1]
dans 2,36 Go, ×1,96 sur le dense, ÷3,41 de mémoire carte, tokens gloutons
identiques — le point servi le plus compact mesuré. Rien aux deux autres
tailles, où le dossier prédit depuis le 2026-08-09 qu'il « reprend son gain
plein » (la part de queue retombe à ~10 % puis ~5 %). Les deux artefacts sont
disponibles au bucket.

- **Coût** — repère : le job 4B complet a coûté 0,79 $, dont 1 340 s de transcodage
- **Gate** — aucune bande posée aux autres tailles ; à poser d'avance

### 5.3 Bras 8B AWQ dans vLLM

Refusé faute d'épinglage : `ops/awq_speed.py` porte le 8B en `pinned=False` et
ses deux révisions n'ont **aucune entrée `EXPECTED`** dans le dépôt — elles
n'ont donc jamais passé les contrôles structurels. *« Une révision que personne
n'a validée n'est pas un épinglage, c'est un instantané. »*

- **Coût** — 0,11 $ au tarif du 4B
- **Bloqué par** — faire passer `ops/awq_dequant.py check` sur les deux révisions
  et écrire l'entrée

### 5.4 Le point 32B — qualité seulement

Le seul point qui puisse séparer les deux lectures de la courbe de capacités.
Sur l'écart MMLU au 4 bits, le palier 8B→14B est **muet** (1,40 pp, SE 1,68,
p = 0,40), et p = 0,40 ne prouve pas l'égalité. Depuis l'arbitrage produit du
2026-08-16, c'est aussi **la plus grande classe que la carte cible admette** :
au barreau arbitré elle laisse 27,93 Go pour les poids, soit ~43-46 Md de
paramètres — le 70B ne rentre pas, et aucun format connu ne l'y fait rentrer.

🚨 **Le chemin servi est muré par une arithmétique, pas par un budget.** La
rotation Walsh–Hadamard du `down_proj` demande **102 400 octets** de mémoire
partagée par bloc contre **101 376 en opt-in mesuré** sur la carte servante : il
manque **1 024 octets**, la réserve du driver, qu'aucun rognage hôte ne
récupère, et **toutes les dispositions appellent la même rotation**. La seule
piste nommée — activations partagées en demi-précision, profondeur
d'accumulation ~14,6 étages à n = 25 600 — n'est pas chiffrée.

- **Coût** — ~70 $ estimé : quantification ~62 $ (621 s/bloc mesurés au
  dé-risquage, bf16 validé), campagne ppl/MMLU ~3 $. Budget avec marge : 80 $.
- **Gate** — **à formuler** : le seuil doit porter sur la **chute d'écart
  14B→32B avec son z**, jamais sur une différence nue, sinon le run reproduit le
  défaut du 2026-08-17. Pré-enregistré **et ancré** avant lancement.
- **Bloqué par** — (1) qualité tranchée d'abord, sinon on paie deux fois ;
  (2) vérifier qu'un AWQ 32B officiel existe et se score dans notre harnais ;
  (3) **go budget explicite** — aucun plafond n'est en vigueur

---

## 6. En attente de la lettre TACO

TACO n'accorde qu'une *Major revision*. La recommandation de la revue interne
est de garder ce matériel pour la fenêtre de réponse, **sur demande de
rapporteur**, pas de l'acheter d'avance.

### 6.1 Matériel de réponse aux rapporteurs

Trois lignes non achetées de l'option 3 de la revue du 2026-08-22 (les trois
lignes sous 5 $ de la même liste sont, elles, faites — D1, G3, G1/G2) :

- **WikiText entier + MMLU complet sur les neuf bras** — ~40 $. Ferme
  l'objection de puissance statistique et aligne le protocole sur le papier
  source.
- **MMLU des trois graines F5** — ~3 $. Dirait si les −14,7 pp sont la méthode
  ou le tirage. *C'est la ligne la moins chère et la plus décisive des trois.*
- **Un TTFT à 512 tokens** — non chiffré.

⚠️ Ce que la revue dit explicitement que cette enveloppe **n'achète pas** : un
noyau qui rattrape QTIP.

### 6.2 DOI, arXiv, APC

Le badge ACM « Available » au sens strict demande un hôte archivistique : ni
GitHub ni Hugging Face n'en sont un, et **aucun DOI n'est frappé** — ce qui
suppose un tag gelé, donc la version relue. arXiv vient après le premier tour,
une fois l'endossement réglé (aucun endossement n'existe à ce jour). L'APC se
traite **après acceptation**, jamais dans la lettre de soumission : l'ACM est en
tout-ouvert depuis le 2026-01-01, Scub n'est pas dans ACM Open et la France
n'est dans aucun palier de dispense géographique. Montant non récupéré.

---

## 7. En pause — décision opérateur

### 7.1 MoE (P2, P6) — il manque une politique, pas un devis

Modèle tranché : **Qwen3-30B-A3B** (128 experts, top-8, 6,3 % actifs, f16
propre) ; `gpt-oss-20b` écarté sur le critère de référence f16. Le routage
mesuré montre que **31,4 % des cellules (couche, expert) sont sous le rang
plein** et **qu'une est morte** — zéro routage, qu'aucun corpus ne ressuscite.
Couvrir 90 % des cellules demanderait ×12 de calibration, soit **+13 % de run
seulement** : le devis n'explose pas, **c'est la politique « expert mort » qui
manque**, notre pipeline supposant partout une hessienne inversible.

⚠️ `gpt-oss` active 12,5 % de ses experts quand la cible en active 6,3 % : ce
tableau est un **plancher** de difficulté. Et sur ce MoE, **63 % de l'artefact**
serait porté en 16 bits.

Reste le seul axe connu qui **change la classe de modèle chargeable**.

- **Coût** — P2 ~1,4 $ (décide P6, rapport 1:50) · P6 ~69 $

### 7.2 Cache KV q8 à contexte long

Qualité verte sur les deux axes, les deux intervalles contenant zéro
(ppl +0,049 %, MMLU +0,33 pp), mais **non servi par défaut** : la série
`n_new = 1024` a été abandonnée en entier quand sa première invocation a mis
661 s contre un seuil de 600 posé d'avance, et la règle interdit de baisser un
seuil après avoir vu l'horloge. Or **c'est précisément la région où
l'allègement devrait payer** (à `n_new = 1024` le bras f16 tombe à 5,6 tok/s
contre 9,6 à 128) : on a mesuré la facture, pas la recette.

Toute réouverture demande un **instrument** — un banc qui garde le modèle
résident entre les deux bras — pas une relecture du même run.

⚠️ Depuis l'arbitrage à 8k du 2026-08-16, ce n'est plus un prérequis produit.

### 7.3 Batch M > 1 et prefill

Déclarés hors domaine dans la table d'enveloppe de validité du papier : le
chemin servi n'a pas de prefill (un prompt de ℓ tokens coûte ℓ passes, et le
87,0 tok/s est mesuré sur un prompt de cinq tokens), et le régime batché — là où
opèrent les noyaux 4 bits déployés — n'est pas mesuré. C'est le risque résiduel
que la revue interne qualifie de **« qui ne s'achète pas »**.

- **Bloqué par** — décision arbitrée du 2026-08-18 : batch = 1 assumé, régime
  edge/souveraineté défendu dès l'introduction

---

## Ordre recommandé

1. **§1.1** — la fiche ScholarOne, aujourd'hui.
2. **§2.5 et §2.6** — `ots upgrade` et le tag : cinq minutes chacun, et ils
   portent la revendication centrale du dossier.
3. **§3.1** — l'A/B des bits de gain. 0 $, aucune ligne de Rust, et il vise
   l'écart le plus inexpliqué du dossier.
4. **§2.7 et §2.8** — armer les gardes, inventorier le bucket.
5. **Décision** — si §3.1 (et le cas échéant §3.3) ne referme rien, écrire la
   conclusion du §0 et clore le volet produit. Le §5.4 ne se paie que si la
   qualité a bougé.
