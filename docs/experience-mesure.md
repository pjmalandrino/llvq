# L'expérience de mesure — prête à lancer, non lancée

> **État au 2026-08-04.** Le code est écrit, l'image est construite, le pilote a
> tourné et a recalé les estimations. **L'expérience elle-même n'est pas
> lancée** : la priorité est passée au portage CUDA du noyau fusé, qui est la
> contribution du projet et qui est aujourd'hui prisonnier d'Apple.
>
> **Mise à jour du 2026-08-05 : le protocole passe de trois bras à quatre.** Le
> quatrième est le noyau fusé lui-même, et il est **conditionné à l'achèvement
> du matvec tuilé CUDA** — voir §1 pour la raison du changement, §2 pour la
> réserve qui l'encadre, et §7 pour l'état exact du noyau.
>
> Ce document est ce qu'il faut relire pour la lancer. Le protocole détaillé
> reste [`plan-de-test-v2-cuda.md`](plan-de-test-v2-cuda.md) ; celui-ci en est
> la version exécutable, corrigée par ce que le pilote a mesuré.

---

## 1. Ce qu'on compare

**Quatre bras sur trois objets**, tous du Qwen3-4B, sur **une seule carte** —
une NVIDIA L40S louée chez Hugging Face, 48 Go, 1,80 $/h.

| | Quoi | Débit | D'où il vient | Chemin d'exécution |
|---|---|---|---|---|
| **1. f16** | le checkpoint d'origine | 16 b/poids | `Qwen/Qwen3-4B` | dense |
| **2. AWQ 4 bits** | quantification **officielle de Qwen** | 4,15625 b/poids | `Qwen/Qwen3-4B-AWQ` | dense, par reconstruction |
| **3. LLVQ 2 bits, sans noyau** | notre fichier scellé, **décodé en dense au chargement** | 2,1595 b/poids | publié, `Pier-Jean/Qwen3-4B-LLVQ-2bit` | dense — c'est `bin/run` aujourd'hui |
| **4. LLVQ 2 bits, noyau fusé** | **le même fichier**, transcodé en `Slot32` et lu par le noyau | même fichier | idem | fusé, **projections seules** |

### Pourquoi quatre, alors que le protocole en figeait trois

**Jusqu'au 2026-08-05 ce protocole était à trois bras** — 1, 2 et 3 — et il
disait de ne pas en rajouter. Il en ajoute un, pour une raison qui n'est pas un
élargissement de périmètre mais une correction : **aucun des trois premiers ne
mesure la contribution du projet.** Le bras 3, c'est notre fichier *sans* notre
noyau.

- **(3, 4) isole exactement l'apport du noyau**, toutes choses égales par
  ailleurs : mêmes poids, même fichier, même carte, même protocole. C'est la
  seule différence entre les deux lignes.
- **(4, 2) attaque la question que [`face-au-4-bits.md`](face-au-4-bits.md)
  pose comme centrale** : est-ce que 2 bits **plus notre noyau** bat le 4 bits ?
  Ce document conclut aujourd'hui que « le format qui va vite ne rentre pas
  mieux que du 4 bits ; le format qui rentre mieux ne va pas vite », et cette
  conclusion n'a jamais été confrontée au q4 sur le même silicium.

Ni l'un ni l'autre ne se lit sur trois bras. On mesurait la pertinence de la
contribution en la supposant.

> **Ce que ce passage à quatre ne rouvre pas.** GPTQ et bitsandbytes **restent
> écartés** — la décision est dans [`plan-de-test-v2-cuda.md`](plan-de-test-v2-cuda.md)
> et rien ici ne la révise. Ce n'est pas la même question : ceux-là seraient des
> **concurrents supplémentaires**, le bras 4 est un **second chemin d'exécution
> sur un objet déjà présent**. Le nombre d'adversaires n'a pas bougé ; il y en a
> toujours un, l'AWQ officiel.

### Trois points de méthode, et ils ne sont pas décoratifs

**L'adversaire n'est pas fabriqué par nous.** Personne ne pourra écrire qu'on
l'a affaibli, ni contester son corpus de calibration ou sa taille de groupe.
En contrepartie il est **opaque** : Qwen ne documente nulle part comment cet
AWQ a été produit. À déclarer.

**Les trois bras de bout en bout sont scorés dans notre moteur.** On ne peut pas
faire l'inverse — personne d'autre ne sait lire notre format — donc on fait
entrer l'adversaire chez nous, en écrivant en clair les nombres que son noyau
calculerait de toute façon. Ce que ça coûte : on mesure la **reconstruction**
d'AWQ, pas son arithmétique fusionnée. Borné par un contrôle, pas éliminé.

**Les quatre bras ne sont pas symétriques, et ce document ne fera pas semblant
qu'ils le sont.** Les bras 1, 2 et 3 existent de bout en bout et se comparent
sur les cinq mesures. Le bras 4 n'existe pas de bout en bout : il s'ajoute sur
**un seul axe**, la vitesse, et sur un objet plus étroit que les autres. La
réserve est en tête du §2 parce qu'elle commande la lecture de tout le reste.

> ⚠️ **AWQ replie ses échelles de saillance dans 72 des 146 tenseurs portés**
> (les RMSNorm qui précèdent les projections). Vérifié par comparaison de sha256
> tenseur par tenseur : 74 identiques, 72 différents. Conséquence : on ne peut
> pas se contenter de remplacer les 252 projections, il faut un checkpoint
> complet — sinon le modèle est mathématiquement faux, projections à l'échelle
> `s` et normes sans le `1/s` compensatoire.

---

## 2. Les cinq mesures

### Qui produit quoi — à lire avant les cinq sections

| axe | bras 1 (f16) | bras 2 (AWQ) | bras 3 (LLVQ dense) | bras 4 (LLVQ fusé) |
|---|---|---|---|---|
| disque | ✅ | ✅ | ✅ | **= bras 3** — même fichier |
| mémoire | ✅ | ✅ | ✅ | ❌ **rien à mesurer** (voir ci-dessous) |
| vitesse | ✅ bout en bout | ✅ bout en bout | ✅ bout en bout | ⚠️ **projections seules — autre objet, autre table** |
| perplexité | ✅ | ✅ | ✅ | **= bras 3** |
| MMLU | ✅ | ✅ | ✅ | **= bras 3** |

**« = bras 3 » veut dire : le même fichier, les mêmes codes, les mêmes poids
reconstruits.** Il n'y a pas deux chiffres à produire sur ces axes, il y en a
**un** — celui du bras 3 — et le recopier dans une colonne « bras 4 » ferait
croire à deux mesures indépendantes. **Une seule ligne LLVQ** sur le disque, la
perplexité et le MMLU. On ne mesure pas deux fois le même objet pour remplir un
tableau.

> ⚠️ **Sur la mémoire, la case du bras 4 est vide, et elle ne doit pas être
> remplie par un calcul.** Le bras 3 ne fait économiser **aucun octet de RAM** :
> `sealed::load` décode chaque matrice en tenseur dense au dtype d'exécution
> (`llvq-llm/src/sealed.rs:92`, `decode_matrix`), donc l'artefact 2 bits tient en
> mémoire **exactement les octets du checkpoint f16** — `ops/floor.py` en fait
> son titre. C'est précisément ce que le noyau fusé changerait, et c'est aussi
> pourquoi la case reste vide : le plancher `Slot32` est calculable exactement,
> mais **un plancher calculé n'est pas un pic mesuré** et n'a rien à faire dans
> la même colonne. L'écart entre le ×4,54 de disque et le ×1,00 de mémoire est,
> à l'octet près, le noyau qui n'est pas branché.

### Taille sur disque

Les octets, périmètre déclaré : ce qu'il faut télécharger pour générer un
token. Rien à lancer, c'est un `stat`.

### Mémoire

**Trois nombres, jamais mélangés.**

Le **plancher** — ce que pèsent les poids une fois chargés — est calculable
exactement et ne dépend que du format **tel qu'il est résident**, ce qui n'est
pas le format sur disque : pour le bras 3, c'est du f16 dense (voir l'avertissement
ci-dessus). C'est `ops/floor.py`.

Le **pic** et la **moyenne**, mesurés. Le pilote a établi comment : voir §4.

### Vitesse — et la réserve qui commande le bras 4

**Bout en bout : bras 1, 2 et 3.** Mesurée et étiquetée « moteur », pas
« format ». Les trois passent par le **même chemin dense en f16** — le
checkpoint d'origine, l'AWQ déquantifié en clair, et notre fichier scellé que
`sealed::load` décode en tenseurs denses au chargement. Donc **cet axe est un
contrôle avant d'être une mesure** : trois arithmétiques identiques doivent
rendre trois débits identiques à la dispersion près. Un écart franc entre eux
serait un défaut de harnais, pas un résultat de format.

**Le bras 4 ne peut pas produire une vitesse de bout en bout, et il ne le pourra
pas dans cette campagne.** Ce n'est pas une limite de la carte ni du budget,
c'est un fait de code, vérifié dans le dépôt :

1. **Le noyau n'a littéralement aucun appelant.** `Qwen3::generate` n'a pas de
   cache KV et re-exécute tout le préfixe à chaque token
   (`llvq-llm/src/model.rs:379` : « No KV cache: each step re-runs the whole
   prefix »). Le noyau fusé est un **matvec** ; le runner ne fait jamais de
   matvec, il fait un GEMM sur tout le préfixe.
2. **Aucune implémentation GPU de la rotation d'incohérence n'existe**, sur
   aucun backend. Elle serait payée par le **seul** bras LLVQ — 144 applications
   par token, coût en latence non chiffré.
3. **Le prefill exige un second chemin dense** de toute façon, pour `seq > 1`.

Les trois sont **indépendants du backend** et le portage CUDA n'en lève aucun :
c'est écrit tel quel dans [`portage-noyau-cuda.md`](portage-noyau-cuda.md) §0.3
(« Ce que le portage n'achète pas ») et §6.4 (« Hors périmètre du port, et
inchangé par lui »), et dans [`fiche-4b.md`](fiche-4b.md) §6.10.

**Donc, précisément :**

- **Ce que le bras 4 mesure.** Le temps d'**un token de projections** — les 252
  matrices d'un Qwen3-4B, un command buffer, protocole froid, sortie vérifiée
  ligne à ligne contre une référence f64 — pour le noyau fusé lisant `Slot32`,
  rapporté au **même travail** fait en FP16 dense. C'est l'objet de
  `bin/thesis`, transposé sur la carte.
- **Ce qu'il ne mesure pas.** Une génération. Pas d'attention, pas de normes,
  pas de RoPE, pas de softmax, pas de `lm_head`, pas de cache KV, pas de
  rotation, pas de prefill. **Aucun tok/s n'en sort, et aucun n'en sortira par
  un facteur d'échelle.**

Le rapport publié est donc un **minorant** du rapport ALU/mémoire pur — tout
terme additif commun comprime un rapport — et un **majorant** du rapport de bout
en bout : le portage §6.6 donne 1,88× analytique en ajoutant le seul `lm_head`.
Les deux sont vrais de quantités différentes, et **une table qui n'en imprime
qu'une sera lue comme l'autre.** La mention va dans l'en-tête de la table
publiée, pas en note de bas de page.

> 🚨 **Le glissement d'étiquette est le risque n° 1 du dossier, documenté trois
> fois dans ce dépôt.** « Le noyau est 2× plus rapide sur les projections » et
> « LLVQ est 2× plus rapide en inférence » ne sont pas la même phrase, et la
> seconde est fausse aujourd'hui. La revendication à écrire, et elle est déjà
> légitime : *décodeur Leech multi-coquilles fusé, 252 matrices, modèle entier,
> sur la classe de matériel du papier, face à un noyau que ses propres auteurs
> déclarent mono-coquille et plus lent que QTIP.*

> ⚠️ **Le couple (4, 2) ne se lit pas sur l'axe vitesse dans cette campagne.**
> Le bras 2 n'a pas de noyau ici : on mesure sa **reconstruction** dense, donc
> son chiffre de vitesse est celui du bras 1 à la dispersion près, pas celui du
> GEMV d'AWQ. Opposer le bras 4 au bras 2 en vitesse exigerait un bras
> **AWQ/Marlin réellement dispatché** dans le même conteneur — poste optionnel
> chiffré à **+2 à 4 j** dans [`portage-noyau-cuda.md`](portage-noyau-cuda.md)
> §5, avec une inconnue à lever d'abord : à batch 1, quel noyau part vraiment,
> `gemv` ou `gemm` ? À **lire** dans le code de dispatch, jamais à supposer.
> Sur les axes disque, mémoire, perplexité et MMLU, en revanche, (4, 2) se lit
> — parce que sur ces axes-là le bras 4 **est** le bras 3.

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
> pas**. Les trois bras de bout en bout au même format, ou rien. (Le bras 4 ne
> passe pas de MMLU : c'est le bras 3, mêmes poids, même score.)

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

1. **Mémoire dans le temps**, les trois bras de bout en bout superposés, phases
   annotées.
2. **Perplexité par fenêtre**, les trois bras de bout en bout — `bin/ppl`
   imprime maintenant la NLL par fenêtre à neuf chiffres, donc la courbe existe
   déjà dans les logs.
3. **Profil MMLU par matière**, les trois bras de bout en bout, trié — c'est
   celle qui montre que le 2 bits abîme le raisonnement et pas la mémoire.
4. **Taille contre qualité**, un point par bras de bout en bout, deux panneaux
   (perplexité et MMLU).
5. **Projections : `Slot32` fusé contre FP16 dense** — le bras 4, **seul, sur sa
   propre figure et avec son propre en-tête**. Un token de projections, jamais
   un tok/s. Ne pas la superposer aux quatre précédentes : ce n'est ni le même
   objet mesuré ni la même unité, et une figure partagée serait exactement le
   glissement d'étiquette que le §2 interdit. Conditionnée à K4.

---

## 6. La séquence, quand on la lancera

| | Quoi | Bras | Machine | $ |
|---|---|---|---|---|
| 1 | Déquantifier l'AWQ, ses six contrôles, publier le checkpoint | 2 | ~20 min | ~0,1 |
| 2 | Re-certifier la baseline MMLU sur CUDA — **exiger 70,42** | 1 | ~10 min | ~0,3 |
| 3 | Perplexité 12 et 73 fenêtres, un job par bras, échantillonnage fin | 1, 2, 3 | ~30 min | ~1 |
| 4 | MMLU complet, un job par bras | 1, 2, 3 | ~1,5 h | ~3 |
| 5 | **`thesis-cuda` : 252 projections, `Slot32` fusé contre FP16 dense** ⚠️ conditionné à K4 | **4** | ~10 min | ~0,3 |
| 6 | Dépouillement, figures, manifeste | — | local | 0 |

**Ordre de grandeur : 5,5 $, provisionner 15 $.** L'estimation vient du pilote,
donc elle repose sur une mesure et non sur une extrapolation depuis le Mac. Le
quatrième bras ajoute un job court — plancher de facturation 6-10 min sur
`l40sx1` à 1,80 $/h — donc il ne déplace pas l'ordre de grandeur, et la
provision reste large.

> ⚠️ **Ce budget ne finance pas le portage, il consomme un noyau achevé.** Les
> 15 à 30 runs de mise au point du noyau (lots K3 à K7) sont budgétés à part :
> **5 à 12 $, provision 30 $**, [`portage-noyau-cuda.md`](portage-noyau-cuda.md)
> §5. Ne pas fondre les deux lignes — la campagne se lance quand le noyau est
> fini, pas pendant.
>
> **L'étape 5 saute si le matvec fusé n'est pas achevé** (§7). Dans ce cas la
> campagne reste exactement celle à trois bras, publiable telle quelle, et le
> bras 4 s'ajoute plus tard sans rien réinvalider — c'est l'intérêt de l'avoir
> mis sur un job et une figure séparés.

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

**Pas prêt — et c'est ce qui conditionne le bras 4.** Le **décodeur** `slot_dot`
est porté et validé sur L40S : 20 767 blocs, 498 408 slots, classe / gain /
niveau **exacts**, produit scalaire à 1,15·10⁻⁷·Σ|w·x| de pire erreur
([`docs/mesures/cuda-preflight-2026-08-05.txt`](mesures/cuda-preflight-2026-08-05.txt)).
**Le matvec fusé tuilé, lui, est en cours d'écriture.** Tant qu'il n'est pas
achevé et vérifié contre la référence f64, **le bras 4 n'existe pas** et la
séquence s'arrête à l'étape 4. Un décodeur juste n'est pas un noyau rapide : il
décide bien, il ne multiplie pas encore.

**Non résolu.** Ce que notre harnais mesure de l'AWQ est sa reconstruction et
non son arithmétique fusionnée ; ça se déclare, ça ne se défait pas — et le
bras 4 ne le répare pas, il déplace seulement la question sur un axe où le
bras 2 n'a rien à opposer (§2). Les trois obstacles au bout-en-bout du noyau —
cache KV, rotation GPU, prefill — restent entiers et hors périmètre. Et la
variance de re-quantification reste non mesurée : aucune barre publiée ne la
couvre.
