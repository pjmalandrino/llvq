# Méthode

Les règles du laboratoire, une par paragraphe : la règle, sa raison, la date
où elle a été payée. Les gabarits sont dans `docs/templates/`.

## 1. Avant toute mesure

- Un préreg écrit et tamponné (`ots stamp`) précède la première milliseconde.
  Payé le 2026-08-07 : le critère 1,6× de `Golay70` n'a pour antériorité que le
  commit `caef2ac`, 52 min avant la mesure (*mesuré* git).
- Le préreg porte les critères d'adoption et de kill, chiffrés, sur la quantité
  exacte qu'ils nomment. Payé le 2026-08-15 : la marche passe 0,3101 ns/bloc
  sous le gate de 0,45, le bloc rend 0,6735 (*mesuré*,
  [P1b](mesures/p1b-marche-bloc-2026-08-15.txt)). Le bras CUDA a été autorisé 57
  min.
- La règle de décision partitionne l'espace des résultats et se termine par un «
  sinon ». Payé le 2026-09-02 : M2b rend +3,60 pp [1,47 ; 5,79] et aucune ligne
  du §5 ne s'applique (*mesuré*, [M2b](mesures/m2b-v4bits-2026-09-02.txt),
  [écarts](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).
- Un préreg tamponné ne s'édite jamais. L'ancre atteste des octets ; un écart
  s'écrit à côté, dans `<préreg>-ECARTS.md`, et se cite. Payé le 2026-08-26 : la
  passe d'anonymisation (`01fdbe6`) a réécrit les préregs du 08-10 et du 08-11.
  Aucun des 128 blobs `.md` de l'histoire git ne rend le condensat attesté
  (*mesuré*, [ots](mesures/ots-etat-2026-08-26.txt)).
- Le préreg porte une prédiction signée, opposable, avec son signe et son ordre
  de grandeur. Payé le 2026-09-02 : le kill de M1 prédisait ρ* = 1, la mesure
  rend ρ = 0,7 (*mesuré*, [M1](mesures/m1-hessienne-shrink-2026-09-02.txt)). Le
  prior « k_proj » tombe : v_proj rend +4,48 pp contre +2,09 (*mesuré*,
  [M2](mesures/m2-attribution-4b-2026-09-02.txt)).
- Un seuil ne se baisse pas après avoir vu l'horloge, et une série ne se réduit
  pas après avoir vu ses points. Payé le 2026-08-15 : la série n_new = 1024 est
  abandonnée à 661 s contre un seuil de 600 (*mesuré*, [KV
  q8](mesures/kvq8-4b-2026-08-15.txt)).
- Un A/B ne bouge qu'un mécanisme ; `check_fuse` refuse `FUSE=1` avec
  `ROT_SHARE=0` pour cette raison. Payé le 2026-07-28 : deux variables changées
  entre deux runs du 0.6B, 45 min (*mesuré*, `bin/smoke`) sans verdict.

## 2. Les chiffres

- Chaque nombre porte *mesuré*, *calculé* ou *estimé*, et sa comptabilité
  (`rtbits`, `matvec`, `thesis` ou inférence). Le fichier 4B publié pèse 2,1595
  b/poids sur 3 633 315 840 poids de projection (*calculé* sur comptages
  mesurés, `bin/seal`, carte HF, [fiche 4B](fiche-4b.md)) et 2,1696 queue exclue
  du dénominateur. Le 2,0702 imprimé par `smoke` est un taux idéal : il ne se
  cite pas pour ce fichier. QTIP au banc lit 2,0000 (*mesuré*,
  [F2](mesures/f2-p3-qtip-banc-2026-08-21.txt)), troisième comptabilité. Payé le
  2026-09-02 : un « +47 % de VRAM » circulait comme mesuré alors qu'il est
  calculé ([écarts
  A2](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md)).
- Une plage, jamais un point. Les millisecondes dérivent entre invocations du
  même binaire : 2,029× / 2,050× / 2,080× (*mesuré*,
  [K1](mesures/k1-metal-2026-08-05.txt)). Payé le 2026-08-18 : les points
  uniques 88,4 tok/s et ×1,12 deviennent 87,0 [86,8 ; 87,0] et ×1,11 (*mesuré*,
  [B2](mesures/b2-fusedrun-plages-2026-08-18.txt)).
- Un banc de décodage n'écrit jamais les poids décodés, charge l'activation une
  fois par threadgroup et amortit la soumission sous ~12 %. Le régime DRAM se
  force à froid, 4 copies en rotation : le SLC de 48 Mo du M3 Max rendait les
  buffers cache-résidents. Payé le 2026-07-31 : le verdict « 25 tok/s, c'est
  mort » était 5× trop pessimiste (*calculé* avant correction des trois défauts
  de banc, sans journal ; ces défauts sont décrits dans
  [format noyau](format-noyau.md)).
- Un rapport se forme round par round, dans un seul processus, tous les bras
  entrelacés dans le même ordre à chaque round. Quand les bras ne coexistent
  pas, on publie le quotient des médianes avec enveloppe. Une lecture inter-jobs
  se rapporte sans se publier : le ×1,091 du hissage de rotation (*calculé*,
  [D1](mesures/d1-fusion-servie-2026-08-24.txt)).
- Le débit se dit en deux formulations, toujours ensemble : brut contre notre
  dense, ×2,00 [1,99 ; 2,00], et à tête identique, ×1,11 [1,11 ; 1,11]
  (*mesuré*, B2). Notre dense recopie 778 Mo de vocabulaire par token (*mesuré*,
  [phases](mesures/phases-2026-08-07.txt)).
- Toute comparaison mémoire se dit en b/param modèle entier, embedding compris :
  5,162 contre 5,302 pour l'AWQ au 4B (*calculé* sur octets mesurés,
  [rtbits](mesures/rtbits-planes-8b-2026-08-09.txt)). Payé le 2026-08-06 : «
  5,51 contre 4,50 » mêlait deux dénominateurs et deux quatre-bits
  ([errata](archive/errata-rapport-lot-a-2026-08-06.md)). Le b/poids des
  linéaires n'est jamais le taux de compression du modèle. Au 0.6B l'embedding
  lié pèse 26 % et le ratio réel vaut ×2,77 contre ×7,4 nominal (*calculé* sur
  les comptes de poids du smoke du 2026-07-28, sans journal). Au 8B les têtes
  déliées font 57 % du fichier scellé, ×3,7 (*calculé* sur les comptes de poids
  du run du 2026-08-02 ; tailles scellées 4,32 Go f16 et 2,49 Go de tables dans
  [HISTORIQUE](HISTORIQUE.md), 2026-08-08).
- Les × inter-cartes ne se divisent pas. Le ×1,78 entre L40S et A100 est le
  rapport d'horloges 2 520 / 1 410 MHz (*mesuré*, nvidia-smi à 1 Hz,
  [G](mesures/g-horloges-planes12x-2026-08-23.txt)), soit 1,787 (*calculé*,
  même journal). Un « vs FP16 » porte sa
  carte : `Planes14` rend 2,14× sur L40S (*mesuré*, G) et 0,79× sur A100
  (*mesuré*, [F4](mesures/f4-a100-2026-08-18.txt)).
- Les rapports intra-pile ne se divisent pas entre piles. ×2,413 pour l'AWQ chez
  vLLM (*mesuré*, [vLLM](mesures/awq-vllm-4b-2026-08-17.txt)) et ×1,11 pour nous
  chez candle se citent côte à côte. vLLM préalloue sa VRAM : aucun chiffre de
  mémoire n'en sort. M = 1 n'est pas le régime optimal de Marlin (tuile minimale
  M = 8) : le ×2,413 ne majore pas l'AWQ ([préreg
  vLLM](../proofs/preregistration-awq-vllm-2026-08-17.md)).
- Entre formats à bits différents, la grandeur comparable est les Go/s : QTIP
  lit 2,000 b/poids et `Planes14` 4,804 (*mesuré*, F2).

## 3. Le bruit

- Tout ce qui recalibre se lit contre le σ de calibration au 4B. En perplexité :
  5,2 %, étendue 10,3 % sur trois runs complets à 21,45 $ (*mesuré*,
  [F5](mesures/f5-graines-4b-2026-08-19.txt)). Sur MMLU : 2,92 pp (*mesuré*,
  [bruit MMLU](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)). Un effet plus
  petit exige une réplication à deux tirages. Payé le 2026-08-19 : le σ de 0,7 %
  hérité de 3 blocs de 0.6B (*mesuré*, [lot
  B](archive/verdicts-lot-b-2026-08-06.md)) était faux d'un facteur 7
  (*calculé*) à la taille publiée. Le 2026-08-25, le classement des bits de gain
  s'inverse au second tirage (*mesuré*,
  [gain](mesures/gain-ab-gate-0.6b-2026-08-25.txt)).
- Un A/B à fichier constant ne porte pas ce σ. Sa barre est l'intervalle apparié
  : SE 0,43 pp sur MMLU et ±0,12 % de perplexité (*mesuré*, KV q8). Entre
  modèles différents, la SE appariée vaut 0,79 à 1,44 pp (*mesuré*,
  [mmlupair](mesures/mmlupair-4b-8b-2026-08-13.txt)).
- Deux IC appariés intra-taille qui ne se recouvrent pas ne testent rien entre
  tailles : `mmlupair` n'apparie pas deux modèles. La chute d'un écart entre
  tailles se teste avec les SE en quadrature : 4B→8B 6,96 pp, z = 3,82 ; 8B→14B
  1,40 pp, z = 0,83 (*calculé*, [mmlupair
  14B](mesures/mmlupair-14b-2026-08-17.txt)).
- La SE d'une différence MMLU est l'appariée (McNemar), jamais le ± d'un bras.
  Le compte se fait en micro. Payé le 2026-08-30 : un gate rendu en macro à
  72,85 % retombe à 70,36 % en micro (*mesuré*, [gate
  M3](mesures/m3-gate-mmlu-vllm-2026-08-30.txt)).
- Un A/B à 3 blocs ne valide pas un mécanisme qui touche aux magnitudes. Le gate
  à profondeur (28 blocs du 0.6B) est obligatoire et automatisé avant de payer
  une carte. Un proxy local meilleur a prédit deux fois une composition pire.
  Payé le 2026-08-07 : le design C rend ×1,99 à pleine profondeur et le gate
  bloque un run 4B de 4 h (*mesuré*,
  [verdicts](archive/verdicts-nuit-2026-08-07.md)).
- Le kill d'une phase se mesure intra-job sur le chemin servi. Additionner des
  gains de jobs distincts fabrique un nombre (préreg de la phase A, 2026-08-31).

## 4. Les tests

- Un test se déclare vert après mutation : on casse le code et la suite doit
  échouer. Un mutant qui survit dit test faible ou code mort. Payé le 2026-07-28
  : un accumulateur neutralisé ne faisait échouer aucun test, et le balayage de
  suffixes de la réparation de parité était du code mort.
- Un test qui n'exerce un paramètre qu'à sa valeur neutre ne teste rien : étage
  Golay neutralisé, monotonie non stricte, crête testée à λ = 0.
- Un test qui saute quand son fichier manque doit échouer. `#[ignore]` déclare
  l'absence dans la boucle rapide ; invoqué, le test nomme le fichier manquant
  (`llvq-artifact/tests/common/mod.rs`). Payé le 2026-08-08 : huit sites `SKIP`
  passaient au vert sur toute machine sans l'archive.
- Le texte d'un noyau porté s'exécute contre une référence indépendante ; il ne
  se relit pas. Payé le 2026-08-16 : un décalage de 64 bits dans le `peek` d'E1v
  corrompait l'index Golay, attrapé par `llvq-cuda/tests/host_e1v.cpp`.
- `unsafe` est autorisé aux frontières matérielles (mmap, lancement de noyau,
  lecture d'un buffer device) et interdit ailleurs. Cinq crates portent
  `#![forbid(unsafe_code)]` ; `llvq-metal`, `llvq-cuda`, `llvq-llm` en ont 12,
  13 et 11 mentions (*mesuré* grep, 2026-08-08). L'attribut ne couvre pas les
  tests d'intégration : toute phrase d'auditabilité porte cette réserve.
- Un sélecteur refuse toute valeur inconnue (`LLVQ_FUSED_LAYOUT`).
- Un instrument prouve qu'il distingue avant de rendre 0. Payé le 2026-08-26 :
  `grep` sur un `.ots` rend 0 sur un fichier ancré comme sur un fichier en
  attente (étiquette binaire de 8 octets).

## 5. Les canaux de rétention

- Avant de chiffrer un rejeu, épuiser `hf buckets ls`, `hf jobs logs`, `hf jobs
  inspect` et `hf jobs ps -a`. Cinq sorties « perdues » vivaient ailleurs, un
  devis posé contre chacune (*mesuré*, [HISTORIQUE](HISTORIQUE.md)). Payé le
  2026-08-17 : dumps MMLU du 14B au bucket, 579 ko contre une campagne
  (*mesuré*, [mmlupair 14B](mesures/mmlupair-14b-2026-08-17.txt)). NLL du 4B
  dans les logs du job, 0 $ contre 0,25 $ devisés (*mesuré* contre *estimé*,
  [brut 4B](mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt)).
- Le bucket n'héberge que ce qu'un job y a écrit : le 8B scellé d'origine n'y
  était pas. Il compte 69 fichiers, 46,7 Go (*mesuré* le 2026-08-17, mmlupair
  14B), jamais inventoriés.
- Garder la sortie brute et la commiter. Un journal de synthèse est une perte
  dès que le canal expire. `bin/ppl` imprime les NLL par fenêtre sur stderr :
  sans `2>`, elles sont perdues.
- Un `ots upgrade` suit l'ancrage de chaque tampon, sur go d'opérateur. État au
  2026-09-02 : 28 tampons, 20 ancrés (*mesuré*,
  [ots](mesures/ots-etat-2026-09-02.txt)). Les ancres sont lues dans les
  fichiers ; la confrontation à la chaîne exige un nœud et n'a jamais eu lieu.

## 6. Les machines

- Aucun run lancé ou arrêté sans go explicite. Le coût s'annonce avant, le cumul
  après. Chaque vague porte un plafond dans son préreg. Phase A : 4 $ pour 1,11
  $ dépensés. Vague 1 recherche : 5 $ pour 2,46 $ (*mesuré*,
  `docs/data/jobs.csv`, 104 jobs pour 94,97 $ au soir du 2026-09-02).
- Un devis se vérifie contre le registre après le job. Payé le 2026-08-03 : le
  32B prédit à ~500 s/bloc (*estimé*) rend 621 (*mesuré*,
  [HISTORIQUE](HISTORIQUE.md)), soit 25 % de sous-estimation (*calculé*). Le
  coût par poids n'est pas linéaire (n³ de la factorisation).
- `oracle` d'abord, à chaque backend : 42 s (*mesuré*, [rapport lot
  A](archive/rapport-lot-a-2026-08-06.md)). `--features fast-linalg` partout où
  l'on paie : sans lui, 40× plus lent (*mesuré*,
  `llvq-llm/src/bin/smoke.rs:1095`) pour un résultat bit-identique.
- Un job local plafonne `LLVQ_THREADS` à environ `ncpu − 4` et part sous `nice`
  dès le lancement. Payé le 2026-09-02 : la file M1 passée en `nice 10` à la
  cinquième mesure, à 1 470 % de CPU (*mesuré*, [écarts
  M1](../proofs/preregistration-m1-hessienne-shrink-2026-09-02-ECARTS.md)),
  perplexités bit-exactes.
- Aucun `cargo build` pendant une file qui appelle un binaire par chemin. La
  file M1 identifie son binaire par sha256 dans le journal.
- Un `planesbench` à cinq bras ou plus se lance avec `--timeout 90m` : le
  transcodage hôte coûte ~1 470 s (*mesuré*,
  [E2](mesures/e2-golay70-bench-2026-08-07.txt)). Payé le 2026-08-18 : le job
  14B tué par timeout à 42,5 min pour 40 demandées (*mesuré*, B2).
- Un job HF se lance par l'API avec `['bash', '-lc', script]` et un assert
  d'identité contre `hf jobs inspect` ; la CLI parse `-lc` comme `--label c`.
  Payé le 2026-08-31 : A1 est mort quatre fois avant un chiffre, trois
  d'infrastructure et une de lanceur, pour 0,02 $ (*mesuré*, [écarts vague
  2](../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md)).
- Toute retouche sous `cfg(linux)` se type-checke depuis le Mac
  (`CUDARC_CUDA_VERSION=12040 cargo check --target x86_64-unknown-linux-gnu`).
  Payé du 2026-08-15 au 08-16 : image CUDA incompilable sans détection.

## 7. Les mots

- 2026-07-31 : une magnitude libre non facturée vaut 0,66 b/poids (*calculé* :
  2,0653 annoncés, 2,7289 réels, [rétraction et
  gain](archive/retraction-et-gain.md)).
- 2026-08-15 : un compte d'opérations ne prédit pas un temps (×1,002 *estimé*,
  ×2,17 *mesuré*, P1b).
- 2026-08-16 : `git log -S` ne lit pas les messages de commit ; un outil sans
  périmètre énoncé ne prouve pas une absence.
- 2026-08-17 : une phrase sur une courbe nomme sa métrique. Le genou d'échelle
  est résolu en perplexité (t = −6,06, *mesuré*, [ppl
  appariée](mesures/ppl-appariee-4b-2026-08-17.txt)) et muet sur MMLU (p = 0,40,
  mmlupair 14B).
- 2026-08-21 : `nullk` est le plancher de notre géométrie de lancement ; un
  noyau autrement formé passe dessous (2,246 ms contre 2,306, *mesuré*, F2).
- 2026-08-30 : un résultat favorable attendu exige plus de vérification (bras
  GPTQ dégénéré, 24,74 %, *mesuré*, [M3
  gptq2](mesures/m3-gptq2-mmlu-2026-08-30.txt)).
- 2026-09-01 : une hypothèse réfutée dans son signe se consigne avec son chiffre
  (split-K −1,87 %, *mesuré*, [A3](mesures/a3-occupation-banc-2026-09-01.txt)).
