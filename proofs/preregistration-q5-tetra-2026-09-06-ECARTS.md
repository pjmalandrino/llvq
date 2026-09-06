# Écarts au pré-enregistrement du chantier 1 — écrits après le tampon, jamais dedans

> Le pré-enregistrement
> [`preregistration-q5-tetra-2026-09-06.md`](preregistration-q5-tetra-2026-09-06.md)
> (sha256 `866aaa26…`) est tamponné. Il ne s'édite pas. Ce qui le corrige s'écrit ici.

## É1 — La clause de financement du §5 est fausse

**Ce que le préreg écrit**, première ligne de la règle de décision : « la queue
f32 → f16 (ligne 7) la finance : −0,0675 b/param contre +0,0493 ».

**Ce qui est vrai** : la carte tient déjà la queue en f16, et depuis le
2026-08-09. `TAIL_BYTES = 2` (`llvq-llm/src/fused.rs:737`, lot A7a). Il n'y a
donc **aucun bit à libérer sur la carte**. Les −0,0675 b/param sont une
correction de comptabilité du fichier et de `rtbits`, pas une économie neuve.

**Ce que ça change, et ce que ça ne change pas.** La marge réelle avant le b_max
du triplet est **0,0747 b/poids noyau plus grande** que ce que le dossier
publie, ce qui va dans le bon sens : les +0,0493 b/param de Q5 tiennent
largement. Mais la phrase du préreg présente comme un financement ce qui est
déjà encaissé, et un lecteur qui la lit comme une ressource disponible
compterait ces bits deux fois.

**La règle de décision elle-même ne bouge pas.** Elle porte sur `G4`, mesuré
directement par les bras T0 à T3, et sur son IC. Aucun seuil ne dépend de la
clause de financement. La première ligne du §5 reste applicable, moins sa
proposition subordonnée.

Trouvé par la passe adverse du recensement des pistes mémoire, le 2026-09-06 à
20:04, après le tampon de 19:39 et avant la fin du premier bras de traitement.

## É2 — Le §1 dit « adopté sur une base qui n'est plus l'objet », ce qui est juste, et omet que le bras n'est pas servi

Le préreg dit au §4 qu'aucun bras n'est un noyau. C'est écrit et c'est suffisant.
Il faut lui ajouter une conséquence que le §1 ne tire pas : `LLVQ_RESTORE_Q4`
**déquantifie `v_proj` en f16 avant le produit matrice-vecteur**. Donc `G4`
mesure un coût d'information à 4 bits, et non la qualité que servirait un noyau
lisant réellement quatre bits. Le +3,60 pp de M2b a la même limite, et le
dossier l'a toujours écrit ; la répéter ici évite qu'un lecteur du seul §5 la
manque.

## É3 — Le témoin T0 est le même objet, mesuré avant le tampon

Rappel du contrôle 6, pour qu'il ne se perde pas : T0 et le bras `Planes14` de
transfert ont été lancés à 19:26, treize minutes avant le tampon. Le préreg le
déclare. Leurs valeurs attendues (53,49 et 55,59) étaient publiées avant d'être
mesurées sur Metal, donc aucun degré de liberté n'a pu être exploité. Si le
transfert dépasse 0,5 pp sur l'un des deux, tout repart sur carte.

## É4 — Le contrôle 6 est posé en absolu ; la lecture appariée est meilleure, et elle est plus dure

**Ce que le préreg demande** : « `Planes14` publié doit rendre 55,59 et Tetra
53,49 [...] si l'écart final dépasse 0,5 pp sur l'un des deux, tout repart sur
carte ».

**Ce qui est mesuré** :

| bras | carte | Metal | écart brut |
|---|---|---|---|
| `Planes14` publié | 55,59 | **56,09** | +0,50 |
| `Tetra` | 53,49 | **53,50** | +0,01 |

Le contrôle passe : 0,50 ne dépasse pas 0,5, et 0,01 est nul. Mais il passe à la
barre sur le premier bras, et l'absolu n'est pas la bonne lecture.

**Apparié** (`bin/mmlupair`, dump carte de `docs/data/mmlu-dumps/mmlu-4b-llvq.csv`
contre le dump Metal, mêmes 2 280 questions, même empreinte) : **7 questions
basculent sur 2 280**, 2 dans un sens et 5 dans l'autre, 99,7 % de concordance,
McNemar exact **p = 0,45**. Le Δ non pondéré vaut −0,13 pp, IC95
[−0,34 ; +0,06], qui contient zéro. Les +0,50 pp du micro stratifié viennent de
**où** quatre de ces sept bascules tombent : `professional law` pèse 10,9 % de la
strate à lui seul et porte −0,27 des −0,50.

**Ce que ça change pour ce travail** : rien, et pour une raison qui doit être
écrite. Les quatre bras T0 à T3 tournent tous sur Metal et l'inférence est
déterministe, donc `Gf`, `G4` et `Ga` sont des différences appariées
intra-Metal. L'écart au CUDA est commun à tous les bras et ne réapparaît pas
dans les gains.

**Ce que le préreg aurait dû écrire** : un contrôle apparié, avec un seuil sur le
nombre de questions discordantes, plutôt qu'une comparaison de deux nombres
absolus dont l'un est pondéré par des strates.

## É5 — Le témoin livre gratuitement l'intervalle apparié du banc du 4B, et il contient zéro

Les deux dumps Metal permettent la mesure que le banc du 2026-09-06 n'avait pas
faite, faute de `LLVQ_MMLU_DUMP`. Hors périmètre de ce préreg, consigné ici
parce que c'est le témoin de ce travail qui la produit.

`Planes14` contre `Tetra`, appariés, même device, mêmes questions :

```
  Δ micro stratifié   = +2,59 pp   IC95 [-0,32 ; +5,58]   SE 1,50 pp
  Δ non pondéré       = +1,75 pp   IC95 [-0,14 ; +3,69]   SE 0,98 pp
  McNemar exact       p = 0,1260
  discordantes        650 sur 2 280, soit 28,5 % — 345 d'un côté, 305 de l'autre
```

**Les deux intervalles contiennent zéro.** Le coût MMLU de Tetra au 4B n'est pas
résolu par ce protocole. Et 28,5 % de questions discordantes pour un écart net
de 1,75 pp dit que le changement de format **rebrasse** les réponses plutôt
qu'il ne dégrade uniformément.

Conséquence sur le journal du 8B, qui ne s'édite pas : il lit « l'écart-type
apparié de la différence est de l'ordre de 1,1 pp (repère des campagnes M2 à
fichier constant). 3,91 pp fait ~3,5 SE ». Ce repère est trop petit. Les
campagnes M2 comparent le **même** fichier ; ici deux fichiers différents
donnent une SE appariée **mesurée à 1,50 pp**. Sur ce repère, les 3,91 pp du 8B
font **~2,6 SE**, pas 3,5. L'intervalle propre au 8B demande ses propres dumps,
que son banc n'a pas écrits.
