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
