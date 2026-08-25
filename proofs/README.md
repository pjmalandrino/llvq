# `proofs/` — ce qui est opposable à un lecteur

Ce répertoire porte le **petit, le diffable, le porteur** : ce qu'un auditeur
peut lire, hasher et confronter sans accès à notre compte Hugging Face.
Format et raison d'être : [`docs/archive/plan-de-test-v2-cuda.md`](../docs/archive/plan-de-test-v2-cuda.md) §6.

> **Un identifiant de job n'est pas une preuve opposable.**
> `https://huggingface.co/jobs/<user>/<id>` redirige un visiteur anonyme vers
> un formulaire de login. C'est un pointeur pour un auditeur à qui on a donné
> accès — pas une pièce.

## Ce qui vit ici

| fichier | rôle |
|---|---|
| `preregistration-<lot>-<date>.md` | ce qu'on s'engage à conclure **avant** de mesurer : prédiction, règle de décision chiffrée, table des issues, et ce qui invaliderait l'ensemble |
| `<le même>.md.ots` | l'ancre OpenTimestamps du `.md`, qui atteste **un sha256 à une date** — pas un nom de fichier, d'où tout ce qui suit |
| `<le même>.vN-<motif>.md.ots` | l'ancre **gelée** d'une version antérieure, conservée quand un document a dû être ré-ancré (un seul cas : F5, cf. la fin de ce fichier) |

**Au 2026-08-25 : 22 pré-enregistrements, 16 fichiers `.ots`** (donc 15
documents ancrés, plus une ancre gelée). Recompté par commande, pas de
mémoire — `ls proofs/*.md | wc -l` moins ce README, `ls proofs/*.ots | wc -l`.

## Ce qui n'y vit pas

- les **logs bruts**, les dumps MMLU, les `nvidia-smi` : volumineux, ils vont
  au dataset public, cité **par révision figée** et jamais par `/main/` ;
- les **checkpoints déquantifiés** : reproductibles depuis le checkpoint et
  `ops/awq_dequant.py`. On dépose le script et les sha256, pas 8 Go ;
- les **chiffres eux-mêmes** : ils sont dans `docs/data/*.csv`, et c'est de là
  que le papier se régénère.

## Écarts constatés APRÈS ancrage — ils vivent ici, jamais dans le document scellé

*Un pré-enregistrement tamponné est en lecture seule pour toujours (§ « Règle
d'édition après ancrage »). Un écart découvert **pendant** la mesure ne peut
donc pas aller dans son §7bis : il va ici.*

**2026-08-25 — `preregistration-bits-de-gain-2026-08-25.md`, contrôle §4.4.**
Le document exigeait que la factorisation retombe « **sous ~5 %** » du profil
par phase, comme preuve que le binaire avait bien pris `fast-linalg`. Mesuré
sur les trois bras : **8,7 % · 9,4 % · 9,6 %**. Le seuil numérique n'est pas
atteint.

- *Ce qui est établi malgré tout* : la feature est active, par trois signes
  indépendants — l'avertissement « compilé SANS » est absent des trois logs, le
  binaire porte 54 symboles `faer` (zéro avant reconstruction), et la
  factorisation tombe de 43,3 % (gate du 2026-08-07, sans la feature) à ~9 %,
  soit ×4,8.
- *Mécanisme du défaut* : le « ÷40 » d'où venait le ~5 % est repris d'un
  commentaire de `smoke.rs` qui chiffre la factorisation d'**une couche**
  (28,4 s → 0,7 s), pas celle d'un run entier. Le vrai facteur est ~4,8. La
  même erreur explique l'estimation de durée du §9 (« ~20 min/bras » contre
  27-30 mesurées).
- *Effet sur le verdict* : **nul**. Les deux chemins de factorisation sont
  bit-identiques par construction (`both_factorizations_agree`) ; ce contrôle
  portait sur la confiance à accorder aux **durées**, pas aux perplexités.
- *Ce que ça n'autorise pas* : réinterpréter le seuil après coup. Il est écrit,
  il n'est pas atteint, et c'est consigné — pas amendé.

Détail complet dans
[`../docs/mesures/gain-ab-gate-0.6b-2026-08-25.txt`](../docs/mesures/gain-ab-gate-0.6b-2026-08-25.txt).

---

## L'état honnête de ce répertoire

> 🕳️ *Cette section s'ouvrait sur « Il vient d'être créé et il est incomplet »
> — vrai le 2026-08-10, répété pendant quinze jours et vingt documents de plus.
> Une excuse de jeunesse qui survit à sa propre jeunesse est une excuse tout
> court, et elle a couvert deux corrections que le lecteur méritait : le
> recompte, et l'état des ancres.*

⚠️ Il reste **incomplet par rapport à ce que §6 décrit**. Manquent, dans
l'ordre où ça compte :

1. **L'horodatage est fait ; la signature reste due.** §6.5 exige un tag
   annoté signé GPG *et* un `ots stamp` (OpenTimestamps) sur chaque
   pré-enregistrement. **Les deux pré-enregistrements portent leur `.ots`**
   (`preregistration-2026-08-10.md.ots`, `preregistration-2026-08-11.md.ots`,
   ce dernier ré-ancré après sa correction É0 et avant le job qu'il jugeait),
   donc l'antériorité ne repose plus sur une date de commit ré-éditable et se
   vérifie sans nous faire confiance :
   ```bash
   ots verify proofs/preregistration-2026-08-11.md.ots
   ```
   > 🚨 **CETTE COMMANDE ÉCHOUE AUJOURD'HUI, et sur les DEUX fichiers.**
   > Vérifié le 2026-08-16 : les deux `.md` ont été **édités après leur
   > ancrage**, donc leur SHA256 courant n'est plus celui que le `.ots`
   > atteste. Le paragraphe ci-dessus était vrai le jour où il a été écrit et
   > il ne l'est plus — c'est exactement le motif que ce dépôt documente
   > partout, cette fois dans le fichier qui promet le contraire. **Le détail,
   > les empreintes et ce qui reste opposable : section « L'inventaire des
   > tampons » ci-dessous.** Le répertoire n'a par ailleurs plus deux
   > pré-enregistrements mais **vingt-deux**, dont **quinze** portent une ancre
   > (seize fichiers `.ots` : le seizième gèle une version amendée, cf. la fin
   > de ce fichier). 🕳️ *Cette phrase a dit « onze » du 2026-08-16 au
   > 2026-08-25 — recomptée par `ls proofs/*.md | wc -l` moins ce README.*
   >
   > ⚠️ **Et « se vérifie sans nous faire confiance » reste faux pour une
   > seconde raison, indépendante de l'édition des deux fichiers : aucune des
   > seize ancres n'est upgradée.** `ots verify` échoue ici sur l'empreinte,
   > et sur la v1 gelée faute d'objet à hasher ; sur les **treize** autres il
   > a de quoi comparer, mais contre des attestations **pending**, c'est-à-dire
   > contre la parole de quatre calendriers tiers et non contre la chaîne
   > Bitcoin. Détail, commande de contrôle et remède : fin de la section
   > « L'inventaire des tampons ».

   ⚠️ **La signature GPG, elle, n'est pas faite** — elle prouverait *qui* et
   demande une clé privée : action de l'opérateur.

   🚨 **Les deux pré-enregistrements sont désormais GELÉS, et leur en-tête
   ment sciemment.** Chacun s'ouvre sur « ni signé GPG ni horodaté … » :
   c'était vrai à la seconde où il a été écrit, et ça ne l'est plus. Cette
   phrase **ne sera pas corrigée**, parce que corriger le fichier
   changerait son empreinte et invaliderait le `.ots` — et le ré-ancrer
   maintenant produirait un horodatage *postérieur* à la mesure qu'il
   juge, c'est-à-dire exactement l'objet sans valeur qu'on cherchait à
   éviter. **Un pré-enregistrement horodaté est en lecture seule pour
   toujours.** La seule exception jamais consentie est celle du 2026-08-11
   (correction É0), faite le jour même, **avant** le job, et suivie d'un
   ré-ancrage — le §7bis du fichier la consigne, et l'ancien `.ots` reste
   opposable dans l'historique git (`f56ae30`).

   > 🕳️ **Le gel a été enfreint, et il l'a été APRÈS avoir été écrit ici.**
   > Les deux fichiers ont été édités depuis — le 08-11 par la refonte
   > documentaire du 2026-08-12 (`b799c32`, trois réécritures de liens
   > `docs/` → `docs/archive/`), le 08-10 quatre fois. La règle « lecture
   > seule pour toujours » ne vivait que dans cette prose, et une règle qui
   > ne vit que dans la prose se saute — c'est la phrase que le garde de
   > `bin/rankbench` a fini par écrire dans un binaire. Détail et empreintes :
   > section « L'inventaire des tampons ».

   Corollaire pour toute session future : les mises à jour d'état vont
   **ici**, jamais dans un fichier scellé.
2. 🕳️ **Le manifeste et son `verify` — CORRECTION du 2026-08-10.** Une version
   antérieure de ce fichier disait qu'ils « restent à écrire ». **C'est faux, et
   l'erreur aurait envoyé quelqu'un réécrire un outil qui existe** :
   [`ops/manifest.py`](../ops/manifest.py) implémente déjà `record`, `verify`,
   `report` et `selftest`, avec exactement la règle porteuse — hachage du log et
   de l'objet mesuré, commit et propreté de l'arbre, identifiant
   `[[claim:ID]]`, et surtout `value_evidence()`, qui exige que **la valeur
   déclarée se retrouve littéralement dans son log**. Son message d'échec
   nomme la falsification que le hachage seul ne voit pas : « une valeur a
   quitté son log alors que le log, lui, est intact — l'entrée a été éditée à
   la main après l'enregistrement, et c'est celle qui atteint directement le
   papier ».

   **Ce qui manque n'est donc pas l'outil, c'est son usage :
   `ops/manifest.jsonl` n'existe pas.** Zéro entrée, alors que le papier porte
   une centaine de nombres. L'instrument est bâti, le registre est vide — le
   motif que ce dossier connaît par cœur : un garde-fou écrit et jamais armé.

   > 🕳️ **« N'existe pas » et « zéro entrée » sont périmés depuis le
   > 2026-08-18 — et de très peu.** Le fichier existe
   > ([`ops/manifest.jsonl`](../ops/manifest.jsonl), 1 393 octets) et porte
   > **une** entrée : le verdict `golay70v2_vs_fp16_l40s_sept_bras` à 1,77×,
   > enregistré *rétroactivement* sur le job du 2026-08-11, avec son
   > `value_in_log: true` et le sha256 de son journal. ⚠️ **Le diagnostic du
   > paragraphe ci-dessus n'est donc pas renversé, il est déplacé d'un cran** :
   > le registre n'est plus vide, il est **anecdotique** — une entrée contre la
   > centaine de nombres du papier, et son champ `claim` vaut `null`, donc
   > elle n'est reliée à aucune cellule. Un garde-fou armé une fois n'est pas
   > un garde-fou armé.

   ⚠️ Et il vit dans `ops/`, pas ici, alors que §6.3 le range dans `proofs/`.
   Décider **une fois** lequel des deux emplacements fait foi, plutôt que de
   laisser deux conventions coexister. *(Toujours non tranché au 2026-08-25 :
   `proofs/manifest.jsonl` n'existe pas.)*

Tant que le registre reste à une entrée, ce répertoire tient pour la partie
pré-enregistrement — et encore, aux réserves ci-dessous — et documente une
intention pour la partie provenance. Le dire est moins coûteux que de laisser
croire l'inverse.

> ⚠️ **Et « tient pour la partie pré-enregistrement » est trop généreux depuis
> le 2026-08-16.** Sur les **vingt-deux** documents du répertoire, **onze**
> tiennent la promesse entière — tampon posé avant la mesure *et* attestant les
> octets qu'on lit aujourd'hui. L'inventaire ci-dessous dit lesquels, et de
> quelle façon chacun des onze autres décroche.
>
> 🕳️ **Cette phrase a porté « sur les onze documents … deux tiennent la
> promesse entière » jusqu'au 2026-08-25, et les deux nombres étaient faux
> dans le même sens : le répertoire s'est amélioré sans que sa page d'audit le
> dise.** Onze documents et deux conformes était l'état du 2026-08-16 ; onze
> pré-enregistrements ont été écrits depuis (B2, B3, D1, F1, F2, F3, F4, F5,
> G, `awq-vllm`, `fusedrun14b`), et **neuf d'entre eux sont tamponnés avant
> leur lancement et attestent leurs octets courants**. Une surface d'audit qui
> sous-déclare son propre état est le même défaut qu'une surface qui
> sur-déclare : dans les deux cas le lecteur ne sait pas ce qui a été vérifié.

## L'inventaire des tampons — recompté le 2026-08-25

Ce répertoire promet qu'un lecteur peut « lire, hasher et confronter sans
accès à notre compte ». **La promesse ne vaut que pour les lignes ✅
ci-dessous**, et il faut deux colonnes pour la dire, parce qu'un tampon peut
échouer de deux façons indépendantes : être posé **trop tard**, ou attester une
**version qui n'est plus là**.

🚨 **Et il y a une troisième façon, qui n'a pas de colonne parce qu'elle les
frappe TOUTES : aucune ancre n'est upgradée.** Elle est traitée sous la table,
et elle doit être lue avant les ✅ — sans elle, ce tableau se lit comme une
preuve alors qu'il décrit une promesse de calendrier.

La colonne de droite est reproductible **hors ligne**, document par document :

```bash
for f in proofs/*.md.ots; do
  a=$(ots info "$f" | head -1 | awk '{print $4}')      # empreinte ancrée
  c=$(shasum -a 256 "${f%.ots}" | awk '{print $1}')     # empreinte courante
  [ "$a" = "$c" ] && echo "OK   $f" || echo "DÉTACHÉ $f"
done
```

| pré-enregistrement | `.ots` | posé **avant** sa mesure ? | atteste le **fichier courant** ? |
|---|---|---|---|
| `preregistration-2026-08-10.md` | oui | ✅ oui (`.ots` commité le 08-10 09:55) | ❌ **non** — édité **4 fois** depuis |
| `preregistration-2026-08-11.md` | oui | ✅ oui (ré-ancré après É0, avant le job) | ❌ **non** — édité **1 fois** depuis |
| `preregistration-2026-08-13.md` | **aucun** | ❔ rien ne l'établit — mtime et date de commit seuls | — |
| **`preregistration-p1-2026-08-13.md`** | oui | ✅ **oui** — tampon avant la première milliseconde | ✅ `5109b35f…` |
| `preregistration-p1b-2026-08-15.md` | oui | ❌ **non** — tamponné **après** ses mesures | ✅ `d027c9d2…` |
| **`preregistration-p1c-2026-08-15.md`** | oui | ✅ **oui** — le garde de `rankbench` refuse de démarrer sans lui | ✅ `5b2ccc3b…` |
| `preregistration-p2-2026-08-14.md` | **aucun** | — | — |
| `preregistration-p3-2026-08-14.md` | **aucun** | — | — |
| `preregistration-p4-2026-08-14.md` | **aucun** | — | — |
| `preregistration-p5-2026-08-14.md` | oui | ❌ **non** — tamponné **après** ses mesures | ✅ `3b45b450…` |
| `preregistration-e1v-cuda-2026-08-15.md` | **aucun** | ❌ non — **décision explicite de l'opérateur** | — |
| `preregistration-fusedrun14b-2026-08-17.md` | **aucun** | ❔ commité avant le lancement, mais **rien ne l'ancre** | — |
| `preregistration-awq-vllm-2026-08-17.md` | **aucun** | ❔ commité avant le lancement, mais **rien ne l'ancre** | — |
| **`preregistration-b3-8b-seal-2026-08-18.md`** | oui | ✅ **oui** — tampon puis commit `6dd9b4a` **avant** le lancement | ✅ `6c2ee7a4…` |
| **`preregistration-f1-cublasf16-2026-08-18.md`** | oui | ✅ **oui** — tampon + commit `2a35bdb` avant lancement | ✅ `8f40ccd6…` |
| **`preregistration-b2-fusedrun-plages-2026-08-18.md`** | oui | ✅ **oui** — tampon + commit `2a35bdb` avant lancement | ✅ `33c3fbc6…` |
| **`preregistration-f3-events-2026-08-18.md`** | oui | ✅ **oui** — tampon + commit `c6febc3` avant lancement | ✅ `08155e78…` |
| **`preregistration-f4-a100-2026-08-18.md`** | oui | ✅ **oui** — tampon + commit `9f8f160` avant lancement | ✅ `f318dbe7…` |
| **`preregistration-f5-graines-4b-2026-08-19.md`** | oui | ✅ **oui** — tamponné avant le pilote, **ré-ancré** avant les runs de mesure (§6bis) | ✅ `5ae2de8d…` |
| ↳ `…-f5-…-2026-08-19.v1-l4x4.md.ots` | *(ancre seule)* | ✅ oui — c'est la version d'avant l'amendement, tamponnée avant le pilote | ❔ **sans objet** : `c860eccb…` atteste un `.md` **absent de l'arbre**, récupérable par `git show da86975:…` (cf. « geler sous un nom propre » plus bas) |
| **`preregistration-f2-qtip-2026-08-20.md`** | oui | ✅ **oui** — tampon posé **avant le premier job** | ✅ `9b0bc1cf…` |
| **`preregistration-g-2026-08-23.md`** | oui | ✅ oui **selon le document** (« écrit et tamponné avant tout lancement ») ⚠️ mais le **commit** est postérieur aux jobs (08-24 12:01 contre des jobs du 08-23) | ✅ `dfd65c42…` |
| **`preregistration-d1-2026-08-24.md`** | oui | ✅ **oui** — tampon puis commit `d168f40` **avant** le lancement | ✅ `c419d67d…` |

**Ce que ça donne, en clair.** **Onze** documents — `p1`, `p1c`, `b3`, `f1`,
`b2`, `f3`, `f4`, `f5`, `f2`, `g`, `d1` — tiennent la promesse entière : tampon
posé avant la mesure *et* attestant les octets qu'on lit aujourd'hui. Les onze
autres décrochent de quatre façons distinctes, et il faut les compter
séparément parce qu'elles ne se réparent pas de la même manière :

| comment ça décroche | combien | lesquels |
|---|---|---|
| tampon intègre, mais **posé après** la mesure | 2 | `p1b`, `p5` |
| posé à temps, mais **atteste une version disparue** | 2 | `2026-08-10`, `2026-08-11` |
| **aucun tampon**, antériorité au mieux par date de commit | 6 | `2026-08-13`, `p2`, `p3`, `p4`, `fusedrun14b`, `awq-vllm` |
| **aucun tampon, par décision assumée** | 1 | `e1v-cuda` |

`p1b` et `p5` ont un tampon intègre mais **posé après** leurs mesures : il
prouve que le document n'a pas bougé *depuis*, pas qu'il précédait le chiffre —
leurs journaux portent la dette et la déclarent (`p5-cns-2026-08-15.txt` : « ce
document n'est PAS horodaté … P1, lui, a été tamponné AVANT sa mesure »). Les
deux ancêtres du 08-10 et du 08-11 sont dans le cas inverse : posés à temps,
mais **l'objet attesté n'est plus dans l'arbre de travail**.

⚠️ **La ligne `g` est la seule qui demande un jugement plutôt qu'une
commande.** Son antériorité repose sur ce que le document dit de lui-même, pas
sur un commit : le `.md` et son `.ots` sont entrés dans le dépôt le lendemain
des jobs qu'ils jugent. Le tampon, lui, est intègre et le journal du lot G
n'affirme pas « AVANT lancement » comme le font B2/B3/F1/F3/F4/D1. C'est un
cran plus faible que les dix autres ✅, et ce cran est invisible tant que la
ligne suivante n'est pas lue.

🚨 **ET AUCUN DES SEIZE `.ots` N'A JAMAIS ÉTÉ UPGRADÉ — ce qui rabat *toutes*
les colonnes ✅ ci-dessus sur la confiance en quatre serveurs tiers.** Vérifié
le 2026-08-25, `ots info` sur les seize : chacun porte **4
`PendingAttestation` et 0 `BitcoinBlockHeaderAttestation`**, les mêmes quatre
calendriers partout — `alice.btc.calendar.opentimestamps.org`,
`bob.btc.calendar.opentimestamps.org`, `btc.calendar.catallaxy.com`,
`finney.calendar.eternitywall.com`.

```bash
for f in proofs/*.ots; do
  printf '%s pending=%s btc=%s\n' "$f" \
    "$(ots info "$f" | grep -c PendingAttestation)" \
    "$(ots info "$f" | grep -c BitcoinBlockHeaderAttestation)"
done          # 2026-08-25 : seize lignes, toutes « pending=4 btc=0 »
```

Une attestation *pending* est une **promesse** de calendrier : ces serveurs
disent détenir l'engagement, ils ne l'ont pas encore ancré dans un en-tête de
bloc Bitcoin. Tant qu'ils ne sont pas upgradés, la phrase de ce fichier
« l'antériorité … se vérifie **sans nous faire confiance** » est fausse au
sens strict — elle déplace la confiance de nous vers eux, ce qui est un
progrès, pas la propriété annoncée. **Le remède est une commande, et elle n'a
jamais été lancée** :

```bash
ots upgrade proofs/*.ots       # puis committer les .ots réécrits
```

⚠️ Deux réserves sur ce remède, pour qu'il ne soit pas tenté à l'aveugle.
(1) `ots upgrade` **demande le réseau** et ne rend rien tant que le calendrier
n'a pas agrégé son commitment dans une transaction confirmée — quelques heures
au minimum, et il se relance sans dommage. (2) Il **réécrit les `.ots`** : le
`.md` ne bouge pas, donc le gel du §1 n'est pas enfreint, mais les fichiers
d'ancre changent d'octets et doivent être recommités. C'est la seule
modification d'un `.ots` que ce répertoire autorise, parce qu'elle ne change
pas l'empreinte attestée — elle ne fait qu'y ajouter la preuve de bloc que la
promesse annonçait.

### Le défaut du 08-10 et du 08-11, et comment le vérifier hors ligne

Un `.ots` atteste un SHA256, pas un nom de fichier. Éditer le `.md` — même pour
réécrire un lien — détache l'ancre. La vérification ne demande **aucun réseau** :

```bash
ots info proofs/preregistration-2026-08-11.md.ots | head -1   # empreinte ancrée
shasum -a 256 proofs/preregistration-2026-08-11.md            # empreinte courante
```

| document | empreinte **ancrée** | empreinte **courante** | la version ancrée est |
|---|---|---|---|
| 2026-08-10 | `21aa2f97a3fd7814…` | `fa3140b444296220…` | `git show 8a25792:proofs/preregistration-2026-08-10.md` |
| 2026-08-11 | `1903c32cedea94e3…` | `01a5c4d246964fcd…` | `git show 7678348:proofs/preregistration-2026-08-11.md` |

Les deux blobs git ci-dessus **rendent exactement l'empreinte ancrée** —
vérifié. Donc rien n'est perdu : ce qui est opposable est la **version
historique**, pas le fichier courant, et il faut le dire au lecteur plutôt que
de lui donner une commande qui échoue.

🕳️ **La cause n'est pas une négligence isolée, et c'est ce qui la rend
intéressante.** Le 08-11 a été détaché par la **refonte documentaire du
2026-08-12** (`b799c32`), qui réécrivait `docs/…` en `docs/archive/…` partout
— trois lignes de lien dans un fichier scellé. Le commit contrôlait « 0 lien
cassé, cargo check vert, make check du papier vert » ; **rien ne contrôlait les
ancres**, parce qu'un `.ots` n'est ni un lien ni du code. Le 08-10 avait déjà
été édité trois fois avant ça, dont une par un commit dont le titre est
« Un pré-enregistrement qui ne consigne pas ses entorses ne vaut rien » : la
règle du **gel** existait, le §1 ci-dessus l'énonce, et elle a été enfreinte
avant d'être écrite.

⏳ **Ce qui reste à décider, et ça revient à l'opérateur.** Ré-ancrer serait la
mauvaise réponse **pour ces deux fichiers-là** — le §1 le dit : un ré-ancrage
postérieur à la mesure produit exactement l'objet sans valeur qu'un tampon
existe pour éviter. Les voies possibles étaient trois : déposer à côté du
`.ots` un fichier d'empreinte + révision git de la version ancrée ; **geler la
version ancrée sous un nom propre** et renvoyer le fichier vivant vers elle ;
ou automatiser un garde qui casse le build quand une empreinte ancrée cesse de
correspondre — la forme qu'a prise le garde de `rankbench`, qui est le seul
mécanisme du dépôt à avoir tenu.

> 🕳️ **Ce paragraphe s'achevait sur « Aucune de ces trois n'est retenue à ce
> jour », et c'est faux depuis le 2026-08-19 : la deuxième A ÉTÉ retenue, et
> elle a servi.** Quand F5 a dû être amendé entre son pilote et ses runs de
> mesure (le pilote a tué la flavor `l4x4` par OOM, §6bis du document), le
> fichier a été **ré-ancré** — licitement, l'amendement étant écrit *avant* les
> mesures qu'il juge — et l'ancre de la version précédente a été **conservée à
> côté, sous un nom propre** :
> `preregistration-f5-graines-4b-2026-08-19.v1-l4x4.md.ots`, qui atteste
> `c860eccb…`. Les deux ancres coexistent donc dans l'arbre, la vivante et la
> gelée, et le journal les cite toutes les deux
> ([`docs/mesures/f5-graines-4b-2026-08-19.txt`](../docs/mesures/f5-graines-4b-2026-08-19.txt)).
>
> ⚠️ **Trois précisions pour ne pas sur-lire ce précédent.** (1) Il ne répare
> **pas** le 08-10 ni le 08-11 : geler après coup une version qu'on n'a plus
> n'est pas la même opération que geler au moment où on ré-ancre. Pour ces
> deux-là, ce qui est opposable reste le blob git nommé dans la table
> ci-dessus. (2) Le `.md` de la v1 n'est **pas** dans l'arbre de travail —
> seule son ancre l'est, ce qui laisse au lecteur une empreinte sans son objet,
> exactement le défaut du 08-10/08-11 sous une autre forme. Le `.md` gelé se
> récupère par
> `git show da86975:proofs/preregistration-f5-graines-4b-2026-08-19.md`, et ce
> blob **rend exactement `c860eccb…`** — vérifié le 2026-08-25, comme les deux
> blobs de la table ci-dessus. (3) La troisième voie — le garde automatique qui
> casse le build sur une ancre détachée — **n'est toujours pas retenue**, et
> c'est la seule qui aurait empêché la faute plutôt que de la documenter.
