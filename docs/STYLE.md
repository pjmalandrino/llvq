# Règles d'écriture

Ces règles valent pour tout document vivant du dépôt : `README.md`, `CLAUDE.md`,
`docs/*.md`, les préregs à venir, les journaux à venir. Elles ne s'appliquent
pas rétroactivement aux journaux de `docs/mesures/`, aux préregs tamponnés ni
à `docs/archive/`, qui sont figés.

## Le lecteur

Un humain pressé, compétent, qui n'a pas lu le reste du dépôt et qui doit
pouvoir décider ou reprendre le travail avec ce seul document. On écrit pour
lui. Pas pour la postérité, pas pour un agent, pas pour se couvrir.

## Les six règles

1. **Le fait d'abord.** La première phrase d'une section donne le résultat,
   avec son chiffre. La raison vient après. La réserve vient en dernier.
2. **Une phrase, une idée, vingt mots.** Une phrase qui a deux virgules et un
   « ce qui » se coupe en deux.
3. **Un document vivant ne raconte pas sa propre histoire.** Quand un fait
   change, on remplace la phrase. L'ancien fait va dans `HISTORIQUE.md` avec sa
   date. Jamais de « cette ligne disait X, c'était faux » dans un document
   vivant : c'est ce qui a fait grossir `CLAUDE.md` jusqu'à 2 971 lignes.
4. **Un chiffre porte son étiquette une fois.** *mesuré*, *calculé* ou
   *estimé*, avec un lien vers le journal, à sa première apparition. Ensuite on
   le cite nu. On ne le recopie pas en prose dans trois documents.
5. **Pas de bannières.** Aucun emoji, dans la prose comme dans les tableaux.
   « Faux depuis le 08-21 : » ou « Réserve : » disent la même chose en mots.
   Dans un tableau, « passe » et « échoue » remplacent les coches.
6. **Pas de tournures d'IA.** Liste ci-dessous. Un document qui en contient
   une se corrige avant d'être commité.

## Tournures interdites

Grep-ables, donc vérifiables. Le tiret cadratin est interdit partout : deux
phrases ou une virgule.

```
—  (tiret cadratin)
autrement dit · en d'autres termes · c'est-à-dire que
il faut le dire · il faut le savoir · et c'est le point · et c'est ça qui compte
ce qui est acquis · ce qui n'est pas tranché · ce qui survit · ce qui reste vrai
à dessein · il convient · force est de constater · cela étant · de fait · en somme
notons que · rappelons que · soulignons que · il est à noter · pour mémoire
et ce n'est pas rien · et ce n'est pas un détail · et il faut le dire deux fois
la bonne lecture · la seule lecture autorisée · lu comme
deux lectures · trois choses · en trois phrases · en une phrase
```

Ainsi que : les listes à trois éléments par réflexe, les parallélismes
(« pas X, mais Y »), les questions rhétoriques, les phrases qui commentent la
phrase précédente, le gras sur une phrase entière, les titres-slogans.

## Avant et après, sur du texte réel du dépôt

Avant (HISTORIQUE, 2026-08-18) :

> 🆕 **LE FAIT NEUF QUE LES PLAGES RÉVÈLENT, et c'est le résultat de B2** : **à
> tête identique — la seule formulation qui mesure le noyau — le gain est
> STRICTEMENT CROISSANT avec la taille, ×1,11 → ×1,29 → ×1,41**, là où la série
> brute (×2,00 · ×2,57 · ×2,55) **n'a aucun ordre**, dominée par le handicap
> variable du bras dense.

Après :

> À tête identique, le gain du noyau croît avec la taille : ×1,11, ×1,29,
> ×1,41 du 4B au 14B (*mesuré*, [B2](mesures/b2-fusedrun-plages-2026-08-18.txt)).
> La série brute n'a pas d'ordre, parce que le bras dense est handicapé
> différemment à chaque taille.

Avant (CLAUDE.md, en-tête) :

> 🕳️ **RENVERSEMENT 1 — « tout travail de format plafonne à 4,77× FP16 » EST
> MESURÉ FAUX, et c'est cet en-tête qui le portait.**

Après, dans `ETAT.md` :

> Le plancher `nullk` (4,77×) est celui de notre géométrie de lancement, pas de
> la carte : QTIP passe dessous (2,246 ms contre 2,306).

Et dans `HISTORIQUE.md`, à la date :

> 2026-08-21. Le plafond de 4,77× annoncé depuis le 08-16 est réfuté par F2.

## Longueurs cibles

| document | lignes | contenu |
|---|---|---|
| `README.md` | 120 | ce que c'est, les chiffres, comment lancer |
| `CLAUDE.md` | 120 | règles pour l'agent, carte du dépôt |
| `docs/ETAT.md` | 100 | config servie, chiffres de tête, décisions ouvertes |
| `docs/HISTORIQUE.md` | 400 | une entrée par période, dix lignes chacune |
| `docs/ROADMAP.md` | 150 | la suite, avec gates et coûts |
| `docs/METHODE.md` | 150 | les règles du labo |
| résumé d'expérience | 40 | gabarit `templates/experience.md` |
| préreg | 80 | gabarit `templates/prereg.md` |

Un document qui dépasse sa cible de moitié se découpe ou se raccourcit. Il ne
grossit pas.

## Ce qui ne change pas

La rigueur. Chaque chiffre reste étiqueté, chaque préreg reste tamponné avant
la première mesure, chaque écart à un préreg s'écrit à côté et jamais dedans,
chaque prédiction signée reste opposable. Ces règles sont dans `METHODE.md`.
C'est la forme qui devient directe, pas le fond qui s'allège.
