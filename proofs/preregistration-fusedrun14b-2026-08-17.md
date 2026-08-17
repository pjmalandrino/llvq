# Pré-enregistrement — `fusedrun` 14B (vitesse servie et VRAM carte)

> Écrit et commité **avant le lancement**. Statut du tampon : ce document ne
> sera pas horodaté par OpenTimestamps, et il faut dire pourquoi plutôt que de
> le laisser croire. Cette mesure **ne décide de rien** — elle ne porte aucun
> seuil d'adoption, aucun bras ne se trouve retenu ou écarté par elle. Elle
> remplit une cellule vide d'un tableau publié. Ce qui doit être figé d'avance
> n'est donc pas un critère mais **quatre choix de présentation**, qui sinon se
> feraient en voyant les nombres.

## §0 — Divulgation datée

Sont connus à la signature :

- les débits servis aux deux autres tailles — 4B **43,5 / 88,4-88,5 tok/s**,
  8B **26,6 / 69,3** (configuration `Planes14` + `LLVQ_EMBED=q8`) ;
- les rapports à tête identique **×1,12** (4B) et **×1,30** (8B) ;
- le b/param modèle entier du 14B servi, **5,106**, obtenu le 2026-08-17 par
  `rtbits` sur le fichier scellé ([`mesures/rtbits-14b-2026-08-17.txt`](../docs/mesures/rtbits-14b-2026-08-17.txt)) ;
- le disque du 14B scellé, **6 506 354 741 o**, confirmé à l'octet ;
- le plancher `nullk` (45,2 % du bras servi) et l'attribution du 2026-08-05
  (latence/occupation 39 %, flux 33 %, décodage 19 %).

**Aucun tok/s de quelque bras que ce soit n'existe au 14B, sur aucun matériel.**

## §1 — La configuration publiée, choisie d'avance

La ligne qui entrera dans les tables est **`Planes14` + `LLVQ_EMBED=q8`** — la
configuration **servie**, la même qu'au 4B et au 8B. Le bras dense f16 du même
processus est son témoin.

Si un second bras est mesuré (embedding f16), il est **contrôle**, pas
candidat : il ne remplace pas la ligne publiée, quelle que soit sa valeur.

## §2 — Le recoupement par le troisième instrument

`fusedrun` imprime les « Go carte », qui constituent une **troisième route**
vers le b/param — celle que `rtbits-14b-2026-08-17.txt` déclare manquante au
14B, et qui recoupait **au millième** aux deux autres tailles.

> Si la VRAM rapportée ne rend pas **5,106 b/param à ±0,5 %**, c'est un
> **résultat à expliquer**, et non un nombre à départager contre le `rtbits`.
> Aucune des deux valeurs ne sera choisie parce qu'elle arrange une table.

## §3 — 🚨 « Une plage, pas un point » : ce que la règle impose ICI

La règle de maison n°2 exige une plage. **`fusedrun` n'en produit pas** : il
fait une génération jetée puis **une seule** génération chronométrée par bras
(`llvq-llm/src/bin/fusedrun.rs`), et une invocation de plus repaie le
chargement entier (~470 s pour le bras fusé au 14B).

**Le fait qui tranche** : les chiffres publiés du 4B et du 8B **sont eux-mêmes
des points uniques**. La règle n'a jamais été appliquée à `fusedrun` dans ce
dossier, et le papier l'assume déjà en limitations.

> **Décision prise d'avance : le 14B sera un POINT**, comme les deux autres, et
> la colonne entière sera déclarée « quotients à run unique, sans plage ».
> Donner une plage au seul 14B rendrait la colonne **hétérogène** — trois
> cellules dont une seule barrée — c'est-à-dire exactement la faute que la
> ligne MMLU vient de payer avec le « 6,09 nu » aligné sur deux appariés.

La dispersion se documente **gratuitement**, sans job : sur le *même* 4B, même
layout, même embedding q8, un autre run rend **86,9 tok/s**
([`mesures/paliers-4b-2026-08-10.txt`](../docs/mesures/paliers-4b-2026-08-10.txt))
contre 88,4-88,5 publié — soit **~2 % sur le rapport** (×1,99 contre ×2,03),
le bras dense étant lui reproduit à 0,2 % près (43,7 / 43,6 / 43,5). C'est
cette dispersion qui sera citée, et non une plage fabriquée.

## §4 — Les deux formulations du débit, fixées avant le run

Le rapport brut se dit contre **notre** bras dense, qui est **handicapé** :
`Head::project` appelle `broadcast_matmul`, qui recopie le vocabulaire à chaque
token. **Au 14B le handicap est plus gros qu'ailleurs** — têtes déliées, deux
tables de 777,9 M poids — donc le ×brut sera spectaculaire.

> Il **ne sera jamais publié seul**. Toute citation du rapport 14B donne les
> deux formulations, comme aux deux autres tailles.

La correction « à tête identique » demande le coût de la phase `lm_head`, donc
`LLVQ_TIME_PHASES=1` — une passe supplémentaire, **hors protocole publié**, à
déclarer comme telle si elle est posée. **Décidé d'avance : elle est posée**,
et son statut hors-protocole est imprimé dans le journal.

## §5 — Ce qui invaliderait le run

- moins de 128 tokens générés, ou un `n_tokens` différent de celui des deux
  autres tailles ;
- le garde `arms_are_discriminating` qui refuse de démarrer (il l'a été gardé
  à 128 exprès) ;
- une VRAM carte hors de la bande du §2 **non expliquée** ;
- un OOM hôte pendant `fused::load` — auquel cas le repli `rtx-pro-6000` change
  la carte, donc **le chiffre n'est plus comparable aux deux autres tailles**
  et doit être déclaré tel, ou jeté.

## §6 — Le budget

Attendu **~20 min, ~0,60 $** sur `l40sx1` (extrapolé du 8B : 629 s pour
289 406 976 blocs, contre 549 806 080 au 14B, soit ×1,90). Le plafond réel est
le `--timeout 1h` posé explicitement, soit **1,80 $**. Plafond de lot accordé
par l'opérateur : **10 $**, dont ~2 $ réservés au bras AWQ.

## §7 — Journal des écarts, tenu à chaud

*(vide à la signature)*
