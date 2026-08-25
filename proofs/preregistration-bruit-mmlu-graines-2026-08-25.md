# Pré-enregistrement — le bruit de MMLU entre tirages de calibration, au 4B

**Écrit, commité et tamponné AVANT le lancement du job.** Go de l'opérateur du
2026-08-25.

**Coût : ~0,5 $** (`l40sx1`, ~20 min *estimé* d'après `mmlupair-4b` du
2026-08-13 : 987 s de running pour trois bras au 4B).

---

## §1 — La question, et pourquoi elle n'a jamais été posée

F5 (2026-08-19, 21,45 $) a produit **trois artefacts 4B complets** aux graines
de calibration 1, 2 et 3, et a mesuré leur dispersion **en perplexité** :
étendue 10,3 %, σ = 5,2 %. Il n'a **jamais mesuré MMLU dessus.**

Conséquence : on ne sait pas ce que vaut un écart MMLU entre deux artefacts
calibrés différemment, à la taille publiée. C'est exactement le chiffre dont
dépend la lisibilité du bras « volume de calibration » envisagé ensuite — un
bras unique contre un témoin unique n'a de sens que si l'on connaît son
plancher de bruit.

Les trois artefacts **ont survécu dans le bucket** (`hf buckets ls`, vérifié le
2026-08-25) : `f5-graines-2026-08-19/seed{1,2,3}/q4b-s{1,2,3}-sealed.llvq`,
1 770 528 125 octets chacun. Le chiffre coûte donc 0,5 $ au lieu des ~21 $
d'une requantification. Troisième fois que la règle des canaux de rétention
paie.

---

## §2 — Le job, verbatim

Image `hf.co/spaces/Pier-Jean/llvq-runner-cuda`, flavor `l40sx1`, bucket
`Pier-Jean/jobs-artifacts` monté en lecture-écriture sur `/out`.

```
for s in 1 2 3; do
  LLVQ_MMLU_DUMP=/out/bruit-mmlu-graines-2026-08-25/mmlu-s$s.csv \
    mmlu /out/f5-graines-2026-08-19/seed$s/q4b-s$s-sealed.llvq cuda 40
done
```

`limit 40` par matière — le même que toutes les campagnes MMLU du projet, donc
2 280 questions.

---

## §3 — Contrôles, et si l'un tombe aucun chiffre n'est publié

1. **Empreinte de tokens `65dcd53655e8bfa5`** imprimée sur les trois bras. Sans
   elle les trois ne scorent pas les mêmes questions et rien n'est comparable.
2. **Trois fichiers de 1 770 528 125 octets** exactement — la taille du fichier
   publié. Une taille différente veut dire un autre objet.
3. **2 280 questions** sur les trois.
4. Les **dumps par question** doivent être écrits : sans eux, aucune paire
   appariée n'est formable plus tard, et c'est la faute que ce projet a déjà
   payée deux fois.

---

## §4 — Ce qui se publie, et ce qui NE se compare PAS

Ce qui se publie : les trois micros MMLU, leur étendue, leur σ (n = 3, donc un
ordre de grandeur, pas un écart-type de précision), et les trois écarts
appariés deux à deux (McNemar + bootstrap stratifié par matière, sur les dumps).

🚨 **Ce qui ne se compare pas : le 55,59 publié.** Il a tourné en **préfixe
contigu** quand les trois graines tirent des offsets aléatoires — deux modes
d'échantillonnage, pas deux tirages du même. C'est la réserve exacte que F5 a
posée sur la perplexité, et elle s'applique ici à l'identique. **Ce qui est
propre est l'étendue ENTRE LES TROIS.**

Ce que ce job ne dira pas : rien sur le volume de calibration, rien sur la
vitesse, rien sur une autre taille, rien sur la perplexité (déjà mesurée par
F5).

---

## §5 — La règle de décision, posée AVANT de voir le chiffre

Soit **s** l'écart-type des trois micros MMLU. Elle décide du design du bras
« volume » et de lui seul :

| | design du bras volume |
|---|---|
| **s ≤ 1,0 pp** | bras unique à **×32** — un effet de ~2 pp serait lisible |
| **1,0 < s ≤ 2,0 pp** | bras unique à **×96** — on maximise l'effet plutôt que de réduire le bruit |
| **s > 2,0 pp** | **on ne lance pas** — un bras unique ne déciderait rien, le design est à repenser |

⚠️ **Cette règle est conservatrice par construction, et c'est voulu.** Les trois
graines sont des tirages **indépendants** ; le bras volume comparera deux
préfixes **emboîtés** (le long contient le court), donc corrélés, donc d'écart
moins variable. **s majore le bruit du bras volume.**

---

## §6 — Divulgation datée, à la signature

- Aucune évaluation MMLU n'a jamais tourné sur ces trois artefacts.
- Ce qui est connu d'eux : ppl scellé f16 **16,7425 / 15,8836 / 15,1027**, et
  2,0702 b/poids effectifs identiques aux trois.
- Repères MMLU au 4B : f16 **70,32**, AWQ 4 bits **70,04**, LLVQ publié
  **55,59** (préfixe contigu, cf. §4).
- **Prédiction de l'auteur, écrite pour être opposable** : **s entre 0,5 et
  1,5 pp**, donc la règle du §5 devrait tomber sur la première ou la deuxième
  ligne. Motif : MMLU est un instrument plus grossier que la perplexité
  (2 280 questions binaires contre 49 140 tokens scorés), et le seul précédent
  du dossier — le swap L≤4, +4,75 % de perplexité pour −0,66 pp de MMLU — va
  dans le sens d'une faible sensibilité.
- ⚠️ **Ce que vaut cette prédiction** : l'auteur en a signé deux le 2026-08-25,
  et **les deux étaient fausses** (`leech4c10` prédit entre 39 et 42, mesuré
  47,15 ; « le classement tient », il s'est inversé). Elle est là pour être
  opposable, pas pour être crue.
