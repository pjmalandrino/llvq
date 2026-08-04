# L'expérience de mesure — prête à lancer, non lancée

> **État au 2026-08-04.** Le code est écrit, l'image est construite, le pilote a
> tourné et a recalé les estimations. **L'expérience elle-même n'est pas
> lancée** : la priorité est passée au portage CUDA du noyau fusé, qui est la
> contribution du projet et qui est aujourd'hui prisonnier d'Apple.
>
> Ce document est ce qu'il faut relire pour la lancer. Le protocole détaillé
> reste [`plan-de-test-v2-cuda.md`](plan-de-test-v2-cuda.md) ; celui-ci en est
> la version exécutable, corrigée par ce que le pilote a mesuré.

---

## 1. Ce qu'on compare

Trois objets, tous du Qwen3-4B, sur **une seule carte** — une NVIDIA L40S
louée chez Hugging Face, 48 Go, 1,80 $/h.

| | Quoi | Débit | D'où il vient |
|---|---|---|---|
| **LLVQ 2 bits** | notre fichier scellé | 2,1595 b/poids | publié, `Pier-Jean/Qwen3-4B-LLVQ-2bit` |
| **AWQ 4 bits** | quantification **officielle de Qwen** | 4,15625 b/poids | `Qwen/Qwen3-4B-AWQ` |
| **f16** | le checkpoint d'origine | 16 b/poids | `Qwen/Qwen3-4B` |

Deux points de méthode, et ils ne sont pas décoratifs.

**L'adversaire n'est pas fabriqué par nous.** Personne ne pourra écrire qu'on
l'a affaibli, ni contester son corpus de calibration ou sa taille de groupe.
En contrepartie il est **opaque** : Qwen ne documente nulle part comment cet
AWQ a été produit. À déclarer.

**Les trois bras sont scorés dans notre moteur.** On ne peut pas faire
l'inverse — personne d'autre ne sait lire notre format — donc on fait entrer
l'adversaire chez nous, en écrivant en clair les nombres que son noyau
calculerait de toute façon. Ce que ça coûte : on mesure la **reconstruction**
d'AWQ, pas son arithmétique fusionnée. Borné par un contrôle, pas éliminé.

> ⚠️ **AWQ replie ses échelles de saillance dans 72 des 146 tenseurs portés**
> (les RMSNorm qui précèdent les projections). Vérifié par comparaison de sha256
> tenseur par tenseur : 74 identiques, 72 différents. Conséquence : on ne peut
> pas se contenter de remplacer les 252 projections, il faut un checkpoint
> complet — sinon le modèle est mathématiquement faux, projections à l'échelle
> `s` et normes sans le `1/s` compensatoire.

---

## 2. Les cinq mesures

### Taille sur disque

Les octets, périmètre déclaré : ce qu'il faut télécharger pour générer un
token. Rien à lancer, c'est un `stat`.

### Mémoire

**Trois nombres, jamais mélangés.**

Le **plancher** — ce que pèsent les poids une fois chargés — est calculable
exactement et ne dépend que du format. C'est `ops/floor.py`.

Le **pic** et la **moyenne**, mesurés. Le pilote a établi comment : voir §4.

### Vitesse

Mesurée et étiquetée « moteur », pas « format ». **Notre noyau fusé n'existe
qu'en Metal**, donc sur une carte NVIDIA notre bras ne fait que « décoder puis
multiplier ». C'est précisément ce que le portage CUDA doit changer, et c'est
pourquoi il passe avant.

### Perplexité

WikiText-2, contexte 4096, fenêtres non chevauchantes, dernière fenêtre
partielle jetée, sans tokens spéciaux. Le corpus donne **73 fenêtres pleines**.

Deux profondeurs : 12 fenêtres pour la courbe, 73 pour la table de tête — ce
qui supprime l'objection « votre sous-ensemble est plus facile ».

**Critère d'acceptation non négociable : l'empreinte de tokens imprimée doit
être identique sur toutes les lignes.** À défaut le run est jeté, pas rattrapé.

### MMLU

57 matières, quatre réponses possibles. On montre cinq questions déjà résolues
de la même matière, on pose la sixième, on s'arrête sur `Answer:`, et on
compare les logits des quatre tokens ` A`, ` B`, ` C`, ` D`. Une passe avant par
question.

C'est la mesure qui montre ce que la perplexité cache : notre modèle perd
14,3 points pendant que sa perplexité bouge à peine, et le profil par matière
dit pourquoi — l'algèbre abstraite et la comptabilité tombent au niveau du
hasard, l'histoire et le droit tiennent au-dessus de 80 %.

**Le chiffre publié est le micro** (une question = un poids, ce que rapporte le
papier), le macro à côté et nommé.

> **Décision prise après le pilote : passer le test complet, 14 042 questions.**
> Il coûtait 5,2 h par bras sur le Mac, ce qui en faisait une option qu'on
> repoussait. À la vitesse mesurée sur la carte il tombe autour de 30 min par
> bras. Ça met la barre d'échantillonnage à **exactement zéro** et rend le
> chiffre directement comparable aux 70,2 et 60,7 du papier.
>
> ⚠️ Un test complet et un test à 40 questions par matière **ne se comparent
> pas**. Les trois bras au même format, ou rien.

---

## 3. Le certificat, à revérifier en premier

Notre harnais MMLU rend **70,42** sur la baseline là où le papier annonce
**70,2** — 0,22 point, soit 0,17 σ. C'est l'argument le mieux fondé du dossier,
et il vaut précisément parce que rien n'a bougé dans le protocole.

**Toute modification du harnais l'invalide.** `bin/mmlu` a changé depuis (dump
par question, empreinte de tokens), donc la première chose à faire est de
rejouer la baseline et d'exiger 70,42 avant d'engager quoi que ce soit.

---

## 4. Ce que le pilote a mesuré, et ce qu'il a corrigé

Deux jobs, **0,27 $ en tout**. Traces : jobs `6a72053ca00abefd4b291fb7` et
`6a722f716b79c09949c2270e`.

### La vitesse : un facteur 10 à 16 contre l'hypothèse

**1,0 seconde par fenêtre de 4096 tokens sur L40S**, contre 10 à 17,5 sur le
M3 Max. Tout le budget de la campagne supposait la carte louée *entre deux fois
plus lente et aussi rapide* que le portable. Elle est un ordre de grandeur
au-dessus.

Conséquence directe : le MMLU complet, chiffré à 6,8 h, tombe sous l'heure, et
le devis global de 16 à 24 $ est largement pessimiste.

### Le backend déplace la perplexité de 0,0065 %

Le pilote a rejoué la baseline sur 12 fenêtres et a imprimé **la même empreinte
de tokens** que le run Metal du matin, `3f1baca9033bf251` — donc exactement les
mêmes tokens des deux côtés.

| | perplexité |
|---|---|
| Metal | 12,2361 |
| CUDA | **12,2369** |

Huit dix-millièmes. Trois ordres de grandeur sous la seule dispersion connue du
projet. **Le certificat MMLU se transportera**, et cet écart est lui-même
publiable : la sensibilité au backend d'un résultat de quantification 2 bits,
qu'aucun papier du domaine ne rapporte.

### L'oracle passe sur CUDA

```
Qwen/Qwen3-4B on Cuda — max |Δhidden| = 0.000e0   (relative to max|h| = 9.482e1)
```

Notre passe avant écrite à la main est exactement d'accord avec celle de
candle sur cette carte. Zéro, pas « proche ». Toutes les hessiennes du projet
en descendent.

### La mémoire : la série redescend, et la moyenne est un piège

| | |
|---|---|
| **Pic** | **23,616 Go** |
| **Moyenne pendant le calcul** | **16,609 Go** |
| Plancher au repos | 0,446 Go (contexte CUDA seul) |
| Moyenne sur tout le job | 3,288 Go — **à ne pas publier** |

**124 des 152 échantillons valent moins d'un giga** : ce sont le téléchargement
et le chargement, pendant lesquels le GPU ne fait rien. Une moyenne calculée
sur toute la durée d'un job dit surtout combien de temps on a passé à
télécharger. **La moyenne doit être calculée sur la fenêtre de calcul**, et la
fenêtre doit être déclarée.

Bonne nouvelle en revanche : la série **redescend** franchement, elle retombe à
0,45 Go entre deux modèles, et `nvidia-smi` en fin de job affiche 0 MiB. Pic et
moyenne sont donc bien deux chiffres distincts, ce qui n'était pas acquis.

> ⚠️ **Le pic à 23,6 Go dépasse l'estimation du protocole (16 à 19 Go). Sur une
> carte de 24 Go, ce job serait mort en mémoire.** Le choix de la L40S à 48 Go
> n'était pas du confort.

### Deux défauts corrigés, tous deux dans notre outillage

`LLVQ_DTYPE=f16` imposé à tout le conteneur tuait l'oracle, qui construit en
f32 pour se comparer à la référence de candle. Chaque binaire a maintenant son
dtype.

Le moniteur s'abonnait au flux de métriques pendant que le job attendait sa
carte. L'endpoint n'a rien à diffuser pour un job qui ne tourne pas, donc le
générateur revenait immédiatement, le fil sortait **sans erreur**, et l'axe
mémoire disparaissait en silence. On savait que s'abonner trop tard perdait le
flux ; s'y abonner **trop tôt** le perd exactement pareil. Le moniteur se
rattache maintenant en boucle jusqu'à un état terminal.

---

## 5. Ce qu'il faut changer avant de lancer, pour avoir des figures

Le pilote a produit une courbe pauvre : sur 151 secondes de job, **28 seulement
sont du calcul**. Cent vingt-quatre points plats et vingt-huit points utiles ne
font pas une figure.

Trois corrections, toutes gratuites.

**Échantillonner plus finement.** Le flux de Hugging Face est bloqué à une
mesure par seconde. Mais `nvidia-smi --query-gpu=memory.used --format=csv -lms 200`
lancé **dans** le conteneur donne une mesure toutes les 200 ms, écrite dans le
log du job. Cinq fois plus de points, même grandeur, zéro coût.

**Allonger les phases de calcul.** Douze fenêtres font 12 secondes ; 73 en font
75, et un MMLU complet plusieurs minutes. Là on a des centaines de points et
une vraie forme.

**Un job par bras, avec des marqueurs horodatés.** Aujourd'hui deux bras
partagent un conteneur et leurs courbes se mélangent. Séparés, et avec un
`date +%s.%N` avant et après chaque phase, la courbe s'annote : chargement,
calcul, libération.

### Les figures visées

1. **Mémoire dans le temps**, les trois bras superposés, phases annotées.
2. **Perplexité par fenêtre**, les trois bras — `bin/ppl` imprime maintenant la
   NLL par fenêtre à neuf chiffres, donc la courbe existe déjà dans les logs.
3. **Profil MMLU par matière**, les trois bras, trié — c'est celle qui montre
   que le 2 bits abîme le raisonnement et pas la mémoire.
4. **Taille contre qualité**, un point par bras, deux panneaux (perplexité et
   MMLU).

---

## 6. La séquence, quand on la lancera

| | Quoi | Machine | $ |
|---|---|---|---|
| 1 | Déquantifier l'AWQ, ses six contrôles, publier le checkpoint | ~20 min | ~0,1 |
| 2 | Re-certifier la baseline MMLU sur CUDA — **exiger 70,42** | ~10 min | ~0,3 |
| 3 | Perplexité 12 et 73 fenêtres, un job par bras, échantillonnage fin | ~30 min | ~1 |
| 4 | MMLU complet, un job par bras | ~1,5 h | ~3 |
| 5 | Dépouillement, figures, manifeste | local | 0 |

**Ordre de grandeur : 5 $, provisionner 15 $.** L'estimation vient du pilote,
donc elle repose sur une mesure et non sur une extrapolation depuis le Mac.

> ⚠️ **Ne pas utiliser `ops/run.py estimate` ni `--max-usd` pour ces jobs.**
> L'estimateur modélise une **quantification** : il multiplie un nombre de poids
> par un coût mesuré de l'encodeur Leech et de la factorisation, dont aucun ne
> tourne dans un job de scoring. Il se trompe d'un facteur ~8. Le plafond utile
> est le `timeout` du job, exact et connu d'avance.

---

## 7. Ce qui est prêt, et ce qui ne l'est pas

**Prêt.** L'image CUDA, avec `mmlu` dedans — il n'y était pas, donc l'axe le
mieux fondé du dossier n'était pas exécutable sur la cible. Le fichier scellé se
monte par volume, sha256 conforme. `bin/ppl` imprime la NLL par fenêtre,
`bin/mmlu` imprime une empreinte de tokens et sait vider un CSV par question.
`ops/awq_dequant.py` passe ses six contrôles, dont un qui ré-empaquette et exige
les octets exacts du fichier de Qwen, et dont le balayage de mutation tue 26
mutants sur 27. `ops/floor.py` et `ops/manifest.py` sont verts.

**Pas prêt, et hors périmètre de cette campagne.** Le noyau fusé sur CUDA. La
vitesse mesurée ici serait celle des moteurs, pas des formats — c'est le
chantier qui commence maintenant.

**Non résolu.** Ce que notre harnais mesure de l'AWQ est sa reconstruction et
non son arithmétique fusionnée ; ça se déclare, ça ne se défait pas. Et la
variance de re-quantification reste non mesurée : aucune barre publiée ne la
couvre.
