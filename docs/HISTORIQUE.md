# Historique

Le fil chronologique du projet, une entrée par période, du 2026-07-24 au 2026-09-02. L'état courant est dans ETAT.md, les règles du labo dans METHODE.md, livrés avec ce fichier ; tant qu'ils ne sont pas dans `docs/`, [CLAUDE.md](../CLAUDE.md) fait foi.

## 2026-07-24 au 07-28. Fondations, G1 à G4

- Golay [24,12,8] et Λ₂₄ tenus par invariants exacts : 196 560 baisers, série thêta, N(13) = 280 974 212 784 720 (*calculé*, `classes_reproduce_theta_series`).
- Recherche NN exacte m ≤ 13 et indexage bijectif 48 bits, format v1 (générateur Golay `0xC75`, ordre des codewords, ordre des classes).
- G4 : 92,23 % de rétention à 1,9999 b/dim, β* = 0,350, MSE 0,0775 (*mesuré*, `llvq-bench`, 20 000 blocs). Le papier donne 89,37 % avec un β désaccordé de ±0,04.
- Encodeur : 639 µs/bloc/cœur, 5,5× le départ ; `nearest_angular` 680 µs (*mesuré*, `encbench`). Sommes télescopiques par runs : une classe en ≤ 5 opérations.
- Passe avant maison contre candle : max |Δhidden| = 0 (*mesuré*, `bin/oracle`). Qwen3-0.6B FP32 : ppl 19,1481 sur 73 fenêtres (*mesuré*).
- Smoke 0.6B/28 blocs : ×1,811 avec rotation contre ×2,290 sans, ×2,748 avec `group_scales` (*mesuré*, `bin/smoke`, 131 k tokens).
- Décisions : shape–gain plutôt que spherical shaping ; `TailPolicy::KeepExact` ; 4 hessiennes par bloc ; `group_scales` désactivé ; A/B sur 3 blocs à une variable.
- Le balayage multi-types de la réparation de parité est du code mort : le maximum est toujours en j = w−1 (trouvé par mutation).
- L'extraction texte du PDF du papier est corrompue ; le rendu image est fiable, transcrit dans [llvq-paper-notes.md](llvq-paper-notes.md).

## 2026-07-29 au 07-31. G5, premier 4B

- Le 07-31, le premier 4B annoncé à 2,0653 puis 2,1117 b/poids (cap 13, 14,9104 ppl) est réfuté. La magnitude libre 16 bits n'était pas facturée : réels 2,7289 et 2,7338 (*calculé*, [archive/retraction-et-gain.md](archive/retraction-et-gain.md)).
- Fichier scellé `leech1c12` (cap 12, 47 + 1 bits) : 16,9617 ppl à 2,1696 b/poids (*mesuré*, `~/llvq-run-4b-artefact.log`). Il pèse 981 Mo de projections, 1,771 Go avec l'embedding f16. Les 2,0702 b/poids effectifs sont le taux idéal imprimé par smoke ; ils ne décrivent pas le fichier écrit (cf. [fiche-4b.md](fiche-4b.md)).
- Gate G5 : QTIP 17,04 à 2,000 (papier). Vert avec 8,5 % de bits en plus.
- Le 07-31, « quantifier le gain ne coûte presque rien » est réfuté : l'A/B comparait un bras à lui-même (écart 7,1e-15). Valeur juste : +3,17 % de ppl pour −0,618 b/poids (*mesuré*, 0.6B, 3 blocs).
- Baseline 4B FP32 12,2336 contre 12,41 au papier, 12 fenêtres contre 73 (*mesuré*).
- Calibrer sur C4 rend 14,91 contre 15,29 sur wikitext (*mesuré*, [fiche-4b.md](fiche-4b.md)) : « calibrer en domaine flatte de 12 % » est réfuté, l'écart mesurait la difficulté du corpus.
- 4B en 3,45 h avec `faer`, 6,3 h sans (*mesuré*).

## 2026-08-01. Audit A

- Confondant dtype nul : ppl f32 contre f16 à 0,1 % (*mesuré*). Fichier scellé décodé f16 : 16,9415, ×1,3846 (*mesuré*, `bin/ppl`), empreinte `3f1baca9033bf251`.
- MMLU rapporté en micro, comme le papier. La prédiction « chute ~−10 pp après correction » est réfutée le 08-02 : l'agrégation valait 0,93 pp (*calculé*, [mmlu-micro-2026-08-02.log](mmlu-micro-2026-08-02.log)).
- Banc G4 recalé sur `LeechShapeGain` (gain codé sur la norme) : shape–gain 0 bit 88,90 %, MSE 0,0850 (*mesuré*, `llvq-bench`).
- `bin/thesis` Metal : FP16 21,69 ms contre LLVQ 10,46 ms, 252 projections, 1 105 920 lignes vérifiées contre f64 (*mesuré*, [mesures/thesis-temoin-2026-08-04.txt](mesures/thesis-temoin-2026-08-04.txt)).
- Le 2,07× publié est le haut d'une plage [2,029 ; 2,080] sur trois invocations (*mesuré*, [mesures/thesis-temoin-2026-08-04.txt](mesures/thesis-temoin-2026-08-04.txt)).

## 2026-08-02 au 08-04. MMLU micro, 8B, 32B, coquille unique

- MMLU micro 4B Metal : 70,42 ± 1,28 contre 56,09 ± 1,36, −14,33 pp (*mesuré*, [mmlu-micro-2026-08-02.log](mmlu-micro-2026-08-02.log)) ; baseline à +0,22 pp du papier. Remplacé par la mesure L40S du 08-06.
- Profil par matière : algèbre abstraite et comptabilité au hasard (25 %), histoire et droit au-dessus de 80 %. Le 2 bits abîme le raisonnement plus que la restitution.
- 8B `leech1c12L3` sur HF Jobs : ×1,267 à 2,0436 b/poids, 4,18 h, 11,48 $ (*mesuré*). Périmé comme point d'échelle par la requantification du 08-08.
- 32B dé-risqué sur 4 blocs : 621 s/bloc contre ~500 prédits, 5,43 $ (*mesuré*) ; run complet re-devisé à ~11,4 h et ~62 $ (*estimé*). C3 (bf16) prérequis.
- Le 08-04, « la coquille unique bat l'union » est réfuté : à 49 bits, union 0,0725 contre coquille 13 0,0762 (*mesuré*, `llvq-bench`). La rétention 92,24 % divisait par un débit fractionnaire qu'aucun fichier ne paie.
- Λ₂₄(12) compte 301 classes, pas 383 (*calculé*, `enumerate_classes`).
- Le 08-03, « 690 paquets » est corrigé : 261 pour `llvq-llm`, 3 pour `llvq-artifact` (*mesuré*).

## 2026-08-05. Lot K-1, portage CUDA

- Échelle Metal à comptabilité unique, 7 rounds dont 2 jetés : Slot32 5,510 b/poids 2,03× [2,03–2,10], Flat32 5,256 0,91×, Grouped32 3,498 0,69× (*mesuré*, [mesures/k1-metal-2026-08-05.txt](mesures/k1-metal-2026-08-05.txt)).
- L'ancienne échelle « 3,35 nested 0,68× ; 4,54 Flat32 0,90× ; 5,51 Slot32 2,07× » mêlait plusieurs comptabilités. Périmée.
- Même binaire, trois invocations : 2,029×, 2,050×, 2,080×. Règle : publier une plage, et former le rapport round par round.
- Le conflit de bancs prédit n'existe pas sur Apple ; `float4` rend 3,5 % et 5,1 % des deux côtés (*calculé*, [mesures/k1-metal-2026-08-05.txt](mesures/k1-metal-2026-08-05.txt)).
- Plafond L ≤ 4 : ≤ 4,7083 b/poids, 4 708 799 groupes sur 4 708 800 portent un bloc à 4 niveaux (*calculé*).
- Noyau de rotation CUDA écrit, 15 mutants tués, jamais tourné sur carte ce jour.
- Attribution du gisement CUDA : 2,04 ms/token, flux 33 %, latence et décodage 59 % (*mesuré*, [mesures/attribution-cuda-2026-08-05.txt](mesures/attribution-cuda-2026-08-05.txt)). Le découpage du 59 % en latence-occupation 39 % (0,803 ms) et décodage résiduel 19 % (~0,396 ms) est mesuré le même jour en faisant varier l'occupation (*mesuré*, [mesures/fusion-qkv-cuda-2026-08-05.txt](mesures/fusion-qkv-cuda-2026-08-05.txt)). Requalifié le 08-21 en propriété de notre géométrie.
- Décision : Metal d'abord (banc gratuit), CUDA ensuite (reproductible) ; `wgpu` jamais.

## 2026-08-06. Lot A, le noyau dans le modèle

- `fusedrun` Slot32 sur L40S : 47,0 tok/s dans 3,28 Go contre 43,5 dans 8,04 dense, 88 tokens identiques (*mesuré*, [archive/passation-lot-a-2026-08-06.md](archive/passation-lot-a-2026-08-06.md)). Point unique, périmé par B2.
- Campagne 4 bras 4B : MMLU f16 70,32 ± 1,28, AWQ 70,04, LLVQ 55,59 ± 1,35 ; ppl ×1,105 contre ×1,384 (*mesuré*, [mesures/a4-campagne-2026-08-06.txt](mesures/a4-campagne-2026-08-06.txt)). Sur un 4B, le 4 bits domine partout sauf le disque.
- Le bras quantifié perd 0,50 pp entre Metal (56,09) et CUDA (55,59) : dette de provenance, log antérieur aux empreintes.
- C1 : Planes14 1,14× [1,14–1,15] plus rapide que Slot32 à 4,804 b/poids, contenu identique (*mesuré*, [mesures/c1-planesbench-2026-08-06.txt](mesures/c1-planesbench-2026-08-06.txt)). Servi le jour même : 48,7 tok/s dans 2,96 Go (*mesuré*, [mesures/planes14-fusedrun-2026-08-06.txt](mesures/planes14-fusedrun-2026-08-06.txt)).
- Lot B, 0.6B 3 blocs (*mesuré*, [archive/verdicts-lot-b-2026-08-06.md](archive/verdicts-lot-b-2026-08-06.md)) : σ inter-graines 0,7 % ; oracle −1,6 % ; volume −1,2 % pour ×13. Damping 0,35 % ; swap L ≤ 4 +4,75 %.
- Le run de calibration ×100 est enterré ; L ≤ 4 est mort en qualité. Le σ 0,7 % sera réfuté le 08-19 à la taille publiée.
- Errata lot A : « 5,51 contre 4,50 » interdit, deux dénominateurs et deux quatre-bits. Règle : b/param modèle entier, embedding compris.
- Le 08-06, « 5 gates sur 7 », « le noyau n'est pas branché » et « le point de décision suivant est C1 » sont périmés.

## 2026-08-07. Design C, Golay70, embedding q8

- Design C : ×1,99 de ppl à 28 blocs (35,98 → 71,42), gate automatique, 0 $ (*mesuré*, [archive/verdicts-nuit-2026-08-07.md](archive/verdicts-nuit-2026-08-07.md)). Réfuté ; la rigidité de norme est porteuse à profondeur.
- Golay70 v1 : 3,589 b/poids, 1,31× [1,29–1,32], 195 Go/s contre un critère de 1,6× (*mesuré*, [mesures/e2-golay70-bench-2026-08-07.txt](mesures/e2-golay70-bench-2026-08-07.txt)). Écarté.
- Même banc : Slot32 1,87× [1,86–1,88] 428 Go/s ; Planes14 2,14× [2,11–2,15] 425 ; Planes12x 4,342 b/poids 1,98× [1,95–1,99], qualité exacte.
- Embedding q8 en production : ppl 16,9358, MMLU 55,70 (*mesuré*, [mesures/campagne-finale-bras4-2026-08-07.txt](mesures/campagne-finale-bras4-2026-08-07.txt)). Le journal donne `fusedrun` à 88,4 tok/s dans 2,60 Go affichés ; la synthèse de campagne écrit 88,4-88,5 (*mesuré*, [campagne-finale-2026-08-07.md](campagne-finale-2026-08-07.md)), point unique.
- Mécanisme du saut de débit : notre bras dense recopie 778 Mo de vocabulaire par token, ~26 ms (*mesuré*, [mesures/phases-2026-08-07.txt](mesures/phases-2026-08-07.txt)), `Head::project` → `broadcast_matmul`.
- Règle : toujours deux formulations de débit, brute et à tête identique. Le dilemme vitesse contre taille est levé : Planes14 est plus petit et plus rapide que Slot32.
- « L'échelle des formats est close » est écrit ce jour ; elle rouvre le 08-10.

## 2026-08-08. Échelle 4B→8B à une variable

- 8B `leech1c12`, même codebook et même corpus : ppl ×1,2201, MMLU 76,08 ± 1,21 → 65,52 ± 1,31, −10,56 pp (*mesuré*, [mesures/campagne-8b-qualite-2026-08-08.txt](mesures/campagne-8b-qualite-2026-08-08.txt)). Écart au 4 bits 14,45 → 7,49 pp.
- Vitesse 8B : dense 26,5, f16 34,4 (×1,30), q8 69,3 tok/s (×2,61) dans 5,45 Go (*mesuré*, [mesures/campagne-8b-q8-2026-08-08.txt](mesures/campagne-8b-q8-2026-08-08.txt)). Points uniques, remplacés par B2.
- Têtes déliées : 2,49 Go de tables en f16 ; scellé 4,32 Go f16, 3,157 Go q8 (*mesuré*). Sans q8, le 8B ne renverse rien.
- `codebook_fingerprint` épinglé à `0x338f_420f_1186_6319` ; `forbid(unsafe_code)` posé sur `llvq-artifact` ; `#[ignore]` inconditionnel pour les archives (11 min 26 s → 2,3 s).
- « suite complète ~45 s » est réfuté : dizaines de minutes (*mesuré*, 17 min sans finir le premier crate). « sept crates » devient huit, « unsafe exclusif à llvq-llm » devient metal 12, cuda 13, llm 11.
- « 26 min de téléchargement sur 65,5 Go » est un nombre circulaire ; hors-boucle borné à ≤ 846 s (*calculé*).

## 2026-08-09. Planes12x câblé, 5,162 b/param

- `rtbits` : 4B Planes14 + q8 = 5,162 b/param, sous l'AWQ 5,302 ; 8B 5,322 contre 5,956 (*calculé sur octets mesurés*, [mesures/rtbits-planes-8b-2026-08-09.txt](mesures/rtbits-planes-8b-2026-08-09.txt)).
- « 5,11 » (embedding à 8 bits nus) et « ≈ 5,15 » (affichage carte 2,60 Go) sont périmés ; un embedding q8 g64 vaut 8,5 b/param.
- Planes12x câblé dans `LLVQ_FUSED_LAYOUT` : 5 096 688 exceptions (3,3824 %) sur 150 681 600 blocs (*mesuré*) ; transcodage 404 s contre 84 s, ×4,8 (*mesuré*, M3 Max 16 threads). Planes12x + q8 : 4,745 b/param (*calculé*, [mesures/rtbits-planes-8b-2026-08-09.txt](mesures/rtbits-planes-8b-2026-08-09.txt)).
- Non défaut au 8B : la VRAM y est déjà gagnée (~11 % sous l'AWQ), le débit coûterait ~7 % (*estimé*).
- « un chemin candle » est réfuté : le chemin est le nôtre, remonté amont (candle#3871).

## 2026-08-10. AWQ au banc, le 14B

- AWQ porté dans notre banc : 584 Go/s, 3,38×, 88 % de sa borne d'octets contre 65 % pour nous (*mesuré*, [mesures/six-arm-awq-2026-08-10.txt](mesures/six-arm-awq-2026-08-10.txt)).
- Le critère de vitesse 1,6× d'E2 est périmé ; E2 rouvert sur l'axe mémoire avec un seuil de 2,0× tamponné le lendemain ([../proofs/preregistration-2026-08-11.md](../proofs/preregistration-2026-08-11.md)).
- 14B : ppl ×1,1894, MMLU 78,97 ± 1,19 → 72,12 ± 1,24, −6,85 pp, IC95 apparié [+4,52 ; +9,12], McNemar 8,7e-16 (*mesuré*, [mesures/campagne-14b-qualite-2026-08-10.txt](mesures/campagne-14b-qualite-2026-08-10.txt)). AWQ 78,21.
- Écart AWQ−LLVQ 6,09 pp écrit comme différence nue ; apparié le 08-17.
- « La courbe a un genou » et « −43 % puis −14 % » sont écrits sur points nus ; requalifiés le 08-17 par métrique.
- Trois points ne font pas une loi d'échelle ; le 32B trancherait.

## 2026-08-11. Golay70 v2

- Décodeur v2 (logique de coset hissée au bloc) : 1,77× [1,76–1,78], 263 Go/s, 1,32× sur v1, 40 % de la borne d'octets (*mesuré*, [mesures/golay70-v2-sept-bras-2026-08-11.txt](mesures/golay70-v2-sept-bras-2026-08-11.txt)).
- Non adopté : sous le seuil de 2,0× ([../proofs/preregistration-2026-08-11.md](../proofs/preregistration-2026-08-11.md)). Plus de piste connue à format inchangé.
- Chaîne préreg 09:30:36, `.ots` 09:31:06, mesure 13:34:31 (*mesuré*, git).
- `golay70` câblé dans `LLVQ_FUSED_LAYOUT` : mesurable, non servi.

## 2026-08-12. Papier, audit externe, refonte

- Papier dégraissé de 16 % de mots, tag `paper-v1` (*mesuré*).
- Audit externe : ~40 chiffres retracés, coût 22,83 $ recalculé exact. Le point 14B manque au papier, au README et à `CLAUDE.md`. « 25 % de mémoire en moins au 8B » est réfuté, réel ~11 % (*calculé*).
- Verdict : l'axe noyau s'arrête, l'actif est le papier et la qualité. Rouvert plus tard par le plan P1→P7 puis la phase A.
- Refonte documentaire : 36 documents déplacés vers `docs/archive/`, `HISTORIQUE.md` créé comme fil unique, `PLAN.md` comme suite.

## 2026-08-12 (suite). Lot X, E1c et E3

- E1c14 et E1c12, transposés sur le groupe de 32 blocs : sweep intégral 150 681 600 blocs exact, 401 s (*mesuré*, [mesures/e1c-sweep-4b-2026-08-12.txt](mesures/e1c-sweep-4b-2026-08-12.txt)).
- Bits non alignés : 4,5551 et 3,7618 b/poids noyau (*mesuré*, [mesures/rtbits-e1c-4b-2026-08-12.txt](mesures/rtbits-e1c-4b-2026-08-12.txt)). Périmés le 08-15 : le matvec servi ne lit pas cette comptabilité.
- Seuils X3 posés : ≥ 2,05× remplace Planes14, ≥ 1,9× remplace Planes12x, < 1,6× ferme. Posés en comptabilité non alignée, à ré-ancrer.
- E3 enterré sur papier : meilleur point 3,0444 b/poids contre un critère de 2,60 (*mesuré*, [mesures/radixstudy-x4-2026-08-12.txt](mesures/radixstudy-x4-2026-08-12.txt)). Le point dans sa classe coûte 41,50 des 47 bits.
- MoE : 31,4 % des cellules (couche, expert) de gpt-oss-20b sous rang plein à 131 k tokens (*mesuré*, [mesures/moe-routing-gptoss20b-2026-08-12.txt](mesures/moe-routing-gptoss20b-2026-08-12.txt)) ; couvrir 90 % exige ×12.

## 2026-08-13. Rejeu apparié 4B et 8B

- Six bras rejoués au centième, empreinte `65dcd53655e8bfa5`, 1,30 $ (*mesuré*, [mesures/mmlupair-4b-8b-2026-08-13.txt](mesures/mmlupair-4b-8b-2026-08-13.txt)).
- AWQ − LLVQ : +14,45 [+11,60 ; +17,27] au 4B, +7,49 [+5,28 ; +9,70] au 8B, IC disjoints. f16 − LLVQ : +14,73 et +10,57, IC se recouvrent.
- f16 − AWQ au 4B : +0,27 [−1,63 ; +2,13], non résolu en micro ; +1,97 [+0,92 ; +3,02] en non pondéré. La phrase du papier tient dans une seule comptabilité.
- SE appariée entre modèles différents : 0,79 à 1,44 pp (*mesuré*).

## 2026-08-14 au 08-15. Cache KV int8

- KV q8, 0 $ et ~2 h 45 de Mac : ppl +0,049 % [−0,071 ; +0,170], MMLU +0,33 pp [−0,45 ; +1,22], McNemar p = 1,0000 (*mesuré*, [mesures/kvq8-4b-2026-08-15.txt](mesures/kvq8-4b-2026-08-15.txt)).
- Débit 0,927× et 0,945× à n_new = 128, série 1024 abandonnée (661 s contre 600). Livré, non défaut : contexte court seulement. Mémoire KV ÷1,882 (*calculé*).
- Contrôle f16 : 16,9415 et 56,09 %, reproduits au dix-millième et à l'identique.
- Barre d'un A/B à fichier constant : ±0,12 % en ppl, SE 0,43 pp en MMLU (*mesuré*). « σ McNemar 0,4-0,6 pp », jamais calculé, est périmé.
- Préregs P2 à P5 réécrits après une revue adversariale à 18 bloquants. MoE (P2, P6) en pause, modèle tranché Qwen3-30B-A3B. Estimateur `ops/run.py` corrigé : 3,34 contre 30,5 Md params (*calculé*, [../proofs/preregistration-p2-2026-08-14.md](../proofs/preregistration-p2-2026-08-14.md)).

## 2026-08-15. P1 mesuré

- `rankbench`, 2^24 blocs, préreg tamponné à 13:37 (sha256 `5109b35f`) : marche-binomiale 0,3101 ns/bloc (kill 1,50), cascade-uniformisée 1,7809 (kill 2,00), cascade-archive 10,8115 (*mesuré*, [mesures/p1-rankbench-2026-08-15.txt](mesures/p1-rankbench-2026-08-15.txt)).
- Uniformiser la boucle vaut un ordre de grandeur : 10,81 → 1,78 ns sur les mêmes bits. La marche rend 3,84× le sol.
- P5 s'ouvre (marche ≤ 0,45) ; bras CUDA de P4 autorisé au commit `b18fe52` (13:42:02).
- V0 : 883 blocs sur 16 777 216 en échec au premier run de cascade-archive (*mesuré*, [mesures/p1-rankbench-2026-08-15.txt](mesures/p1-rankbench-2026-08-15.txt)), corrigé.

## 2026-08-15 (soir). P1b, P1c, P5

- P1b : la marche par bloc rend 0,6735 ns/bloc, ×2,17 contre ×1,002 prédit par le compte de pas (*mesuré*, [mesures/p1b-marche-bloc-2026-08-15.txt](mesures/p1b-marche-bloc-2026-08-15.txt)). Vert au kill 1,50, au-dessus du gate 0,45.
- Autorisation du bras CUDA retirée au commit `c40641b` (14:39:33) : 57 min (*mesuré*, git). « une demi-journée » est périmé.
- Hypothèse débordement réfutée : bras plat 0,8346 contre 0,6704 ns/bloc (*mesuré*, [mesures/p1b-marche-bloc-2026-08-15.txt](mesures/p1b-marche-bloc-2026-08-15.txt)). Le ×2,17 reste non attribué.
- P1c : flux E1v décodé 0,6795 ns/bloc, surcoût d'adressage +1,2 % (*mesuré*, [mesures/p1c-e1v-flux-2026-08-15.txt](mesures/p1c-e1v-flux-2026-08-15.txt)).
- P5 : E1v 2,3877 b/poids, transcodage 1,088× [1,087–1,090] contre 2,0, 0 division (*mesuré*, [mesures/p5-cns-2026-08-15.txt](mesures/p5-cns-2026-08-15.txt)). P5 clos 4/4 : droit de porter E1v sur carte.
- Alignement warp : 0 bloc sur 150 681 600 tombe dans un warp aligné ; bourrage +15,47 % au 4B ; E1c14 aligné 5,2354 contre 4,8040 (*calculé*, [mesures/x3-alignement-warp-2026-08-15.txt](mesures/x3-alignement-warp-2026-08-15.txt)). E1c14 enterré au 4B.
- Les `.ots` de P1b et P5 sont posés après mesure (15:23) : dette déclarée.

## 2026-08-16. E1v fermé, plancher nullk

- E1v CUDA : 0,25× [0,25–0,25], 25 Go/s, 44,253 ms, 0,85 $ (*mesuré*, [mesures/e1v-cuda-2026-08-16.txt](mesures/e1v-cuda-2026-08-16.txt)) contre un plancher de 1,60×. Fermé pour le chemin servi.
- Le format tient : 1,09 Go lus contre 2,18, 2,3983 b/poids en coupe alignée ligne, 79 registres, 0 spill. Le décodeur en ligne multiplie le poste décodage par 17 (*calculé*).
- `nullk`, aucun octet de poids : 2,305 ms contre 5,102 pour Planes14, 45,2 %, 4,77× [4,74–4,77], 0,77 $ (*mesuré*, [mesures/nullk-plancher-2026-08-16.txt](mesures/nullk-plancher-2026-08-16.txt)). Planes14 achète 3,11× net ; décodage ~7 %.
- Écrit ce jour : « plafond absolu de tout travail de format = 4,77× ». Réfuté le 08-21.
- E1c12 aligné 4,2880 contre 4,3424 pour Planes12x, −1,3 % (*calculé*, [mesures/e1c12-aligne-2026-08-16.txt](mesures/e1c12-aligne-2026-08-16.txt)). Payload : 5,3756 · 4,6667 · 4,2029 ; la table du 08-07 est en comptabilité noyau.
- Critère 1,6× d'E2 : antériorité par le message de commit `caef2ac` (10:36:27), mesure `4a09d8b` (11:28:59), sans tampon. « aucune trace avant la mesure » est réfuté : `git log -S` ne lit pas les messages.
- Image CUDA incompilable depuis le 08-15 (N_ARMS 7 → 15) : `arms.rs`, `bin/cuhcheck`. Leçon : faire exécuter le texte d'un noyau porté (`host_e1v.cpp`, décalage de 64).
- « SKIP proprement » remplacé par un échec nominatif : huit sites passaient au vert sans l'archive.

## 2026-08-17. Le bucket, le 14B apparié

- Dumps MMLU 14B retrouvés dans le bucket : 579 ko, 0 $ (*mesuré*). « perdus, campagne à refaire » est réfuté. Bucket : 69 fichiers, 46,7 Go, jamais inventorié (*mesuré*, `hf buckets ls`, [mesures/mmlupair-14b-2026-08-17.txt](mesures/mmlupair-14b-2026-08-17.txt)).
- AWQ − LLVQ 14B : +6,09 pp [+3,62 ; +8,52], SE 1,25, McNemar 1,143e-11, 230/106 discordantes (*mesuré*, [mesures/mmlupair-14b-2026-08-17.txt](mesures/mmlupair-14b-2026-08-17.txt)). Neuf paires existent.
- Chute de l'écart MMLU : 4B→8B 6,96 pp, p = 0,0001 ; 8B→14B 1,40 pp, p = 0,40 non résolu ; 4B→14B 8,36 pp (*calculé*). p = 0,40 ne prouve pas l'égalité.
- « L'écart fond deux fois plus vite », « se referme vers 16-32B » sont retirés.
- `rtbits` 14B : 14 768 307 200 params ; 5,106 contre 5,404 AWQ ; marge −2,6 / −10,6 / −5,5 % non monotone, mécanisme = part de l'embedding (*calculé*, [mesures/rtbits-14b-2026-08-17.txt](mesures/rtbits-14b-2026-08-17.txt)).
- Au 14B, E1c14 aligné 4,6410 < 4,7063 et bourrage +4,18 % (*calculé*, [mesures/rtbits-14b-2026-08-17.txt](mesures/rtbits-14b-2026-08-17.txt)) : « E1c14 enterré » devient un verdict 4B.
- ppl 8B et 14B appariées : excès LLVQ/f16 +22,01 % [+19,37 ; +24,70] et +18,94 % [+17,22 ; +20,68] (*calculé*, [mesures/ppl-appariee-8b-14b-2026-08-17.txt](mesures/ppl-appariee-8b-14b-2026-08-17.txt)).
- Règles : `hf buckets ls`, `hf jobs logs`, `hf jobs inspect` avant tout devis.

## 2026-08-17 (soir). Les NLL du 4B et le genou

- NLL du 4B retrouvées dans `hf jobs logs` (36 lignes, sha256 `07bf4119`), 0 $ contre ~0,25 $ devisés ([mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt](mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt)).
- Excès 4B apparié : LLVQ/f16 +38,45 % [+33,62 ; +43,45] (*calculé*, [mesures/ppl-appariee-4b-2026-08-17.txt](mesures/ppl-appariee-4b-2026-08-17.txt)).
- Genou de perplexité résolu : pas 4B→8B ×0,881211, pas 8B→14B ×0,974855, différence −0,100992 [−0,137670 ; −0,064313], t = −6,06. Fonte −42,8 % [−51,8 ; −33,5] puis −13,9 % [−22,8 ; −4,9].
- « Le genou n'est pas testable en ppl » du matin est réfuté le soir. « −42 % » était une troncature.
- Sur référence AWQ, le pas 8B→14B exclut zéro de 0,005 (t 2,2063 contre 2,200985) : jamais « significativement ».
- Règle : toute phrase sur le genou nomme sa métrique. Règle : garder la sortie brute.

## 2026-08-17 (second lot du soir). AWQ dans vLLM

- vLLM 0.26.0, L40S, batch 1, 128 tokens : f16 83,09 tok/s, AWQ Marlin 200,49 [200,39 ; 200,61], ×2,413 [2,412 ; 2,414] intra-pile, 0,11 $ (*mesuré*, [mesures/awq-vllm-4b-2026-08-17.txt](mesures/awq-vllm-4b-2026-08-17.txt)).
- Le témoin vLLM rend ×1,91 notre dense (*calculé*) : confondant moteur non décomposable. Aucune division inter-piles ; « plus rapide que le 4 bits » ne se dit à aucune échelle.
- « aucune mesure contre l'AWQ dans son moteur » est levé ; l'interdit de comparaison reste.
- Forcer `awq` charge le même noyau Marlin deux fois (écart 0,10 %, *mesuré*, [mesures/awq-vllm-4b-2026-08-17.txt](mesures/awq-vllm-4b-2026-08-17.txt)) : la clause « M = 1 tous les noyaux convergent » reste non testée.
- Bras AWQ 8B bloqué : deux révisions Hub non validées.

## 2026-08-17 (troisième lot du soir). Le 14B servi

- 14B, Planes14 + q8 : 42,9 tok/s dans 9,39 Go contre 17,0 dans 29,54, ÷3,14, ×2,53, 128 tokens identiques, 1,24 $ (*mesuré*, [mesures/fusedrun-14b-2026-08-17.txt](mesures/fusedrun-14b-2026-08-17.txt)).
- « ni la vitesse ni la VRAM mesurées à 14B » du matin est réfuté.
- Recoupement : 9,39 Go × 8 / params = 5,0866 contre 5,106 par `rtbits`, −0,38 % (*calculé*).
- Handicap du dense maximal ici : 1 555,8 Mo recopiés par token, 53,9 ms contre 1,2 ms (*mesuré*, profil fencé, [mesures/fusedrun-14b-2026-08-17.txt](mesures/fusedrun-14b-2026-08-17.txt)).
- Reconstructions à tête identique ×1,78 et ×1,24 par profil fencé : périmées le 08-18. « ×2,53 le plus élevé des trois » est réfuté par le 8B.
- Registre `jobs.csv` soldé : 57,56 $ (*mesuré*, [data/jobs.csv](data/jobs.csv)).

## 2026-08-18. B2, B3, F1

- B2 : médianes sur 5 rounds aux trois tailles, ~2,25 $ sur trois jobs (0,35 + 0,63 + 1,27 ; *calculé*, [data/jobs.csv](data/jobs.csv) ; journal [mesures/b2-fusedrun-plages-2026-08-18.txt](mesures/b2-fusedrun-plages-2026-08-18.txt)). 4B q8 87,0 [86,8–87,0] dans 2,56 Go, ×2,00 ; f16 48,3 [48,1–48,3], ×1,11 [1,11–1,11].
- 8B : 68,2 q8, 34,1 f16, ×2,57 et ×1,29. 14B : 43,3 q8, 23,9 f16, ×2,55 et ×1,41 [1,40–1,41].
- Série à tête identique croissante : ×1,11, ×1,29, ×1,41. La série brute (×2,00 · ×2,57 · ×2,55) n'a pas d'ordre.
- Tous les points uniques (47,0 ; 48,7 ; 88,4-88,5 ; 69,3 ; 42,9 ; 2,60 Go) sont périmés, écarts de −1,6 à +0,9 %. « 2,60 Go » était l'affichage carte arrondi.
- B3 : 8B re-scellé depuis le bucket, 5,322 b/param au millième, 0,24 $ (*mesuré*, [mesures/b3-8b-reseal-2026-08-18.txt](mesures/b3-8b-reseal-2026-08-18.txt)) contre 12,61 $ provisionnés.
- F1 : témoin f16 maison à 1,024 (2 bras) et 1,015 (5 bras) de cuBLAS, critère ≤ 1,05, 0,08 $ (*mesuré*, [mesures/f1-cublasf16-2026-08-18.txt](mesures/f1-cublasf16-2026-08-18.txt)). Tous les « vs FP16 » L40S tiennent.
- Préreg B3 « graine 1000000 » était une sentinelle : erratum au journal, préreg non édité. Règle : un préreg tamponné ne s'édite jamais.
- `g6_pack` : « échoue en debug, pas une régression » était un bug réel (shift 64), corrigé `a32163e`. Lot « dépôt sans contradiction » : 8 prises.

## 2026-08-19. F3, F4, F5

- F3 : écart hôte−device 0,1-0,2 %, 4-8 µs par round entier, 0,86 $ (*mesuré*, [mesures/f3-events-2026-08-19.txt](mesures/f3-events-2026-08-19.txt)) contre 0,5-2 ms attendus. `ncu` refusé (ERR_NVGPUCTRPERM), clos. Driver 580.159.03 capturé.
- F4 sur A100-SXM4-80GB, ~1,00 $ (*estimé*) : Planes14 0,79×, Slot32 0,73×, Planes12x 0,73×, Golay70 v2 0,62×, AWQ 1,82×, cuBLAS 1,14×, nullk 1,68× (*mesuré*, [mesures/f4-a100-2026-08-18.txt](mesures/f4-a100-2026-08-18.txt)).
- Go/s effectifs 425 → 250 et 428 → 266 : bornés par le calcul sur A100. « decode at matvec speed » devient un énoncé L40S/Ada.
- F5, trois runs complets du 4B, 21,45 $ : graines 1/2/3 à 16,7425 / 15,8836 / 15,1027. Étendue 10,3 %, σ 5,2 %, paires résolues t +4,54 / +10,92 / +7,68 (*mesuré*, [mesures/f5-graines-4b-2026-08-19.txt](mesures/f5-graines-4b-2026-08-19.txt)).
- Le σ 0,7 % du lot B et le seuil « bruit sous 1,5 % » sont réfutés à la taille publiée. Les trois graines rendent 2,0702 b/poids et 1,771 Go identiques.
- Oracle −1,6 % et volume −1,2 % passent sous le bruit ; « plafonné » est maintenu. Un A/B à fichier constant ne porte pas ce σ.
- Journée à 23,31 $ (*calculé*).

## 2026-08-20 au 08-21. F2, QTIP au banc

- QTIP dans notre banc, un processus, 7 rounds dont 2 jetés, 0,89 $ : 2,246 ms [2,245–2,248], 0,91 Go, 2,0000 b/poids, 405 Go/s, 4,89× (*mesuré*, [mesures/f2-p3-qtip-banc-2026-08-21.txt](mesures/f2-p3-qtip-banc-2026-08-21.txt)).
- Même processus : Planes14 5,103 ms, 2,18 Go, 2,15× ; nullk 2,306 ms. r = t(Planes14) ÷ t(QTIP) = 2,27× [2,27–2,28] ; trafic 2,40× (*calculé*).
- t(QTIP) < t(nullk) : séparation 2,7 % contre 2R = 0,72 %. Le 08-21, « tout travail de format plafonne à 4,77× » est réfuté : nullk est le plancher de notre géométrie.
- f = 61,1 % contre 59,6 % tamponné : erratum au journal, préreg non édité.
- Mécanisme : un codebook de 1,1·10¹⁴ points ne tient pas en LUT, un état de treillis 16 bits tient en 2 Kio ; l'index se déplie à 4,80 b/poids (*calculé*).
- Pire erreur 5,4e-8·Σ|w·x| contre seuil 1e-5. Aucune phrase de qualité sur ce bras (payload pseudo-aléatoire).

## 2026-08-23. Lot G, les horloges

- L40S 2 520 MHz, A100 1 410, épinglées au boost max, rapport 1,787 ∈ [1,60 ; 1,95] ; nullk ×1,772 (G) et ×1,781 (F4), 1,00 $ (*mesuré*, [mesures/g-horloges-planes12x-2026-08-23.txt](mesures/g-horloges-planes12x-2026-08-23.txt)).
- Le ×1,78 de la table A100 est le rapport d'horloges. Cette preuve porte sur l'horloge seule, sans profil d'occupation.
- G3 : Planes12x servi au 4B, 85,0 tok/s [84,7–85,1] dans 2,36 Go, ×1,96 [1,95–1,96], ÷3,41, divergence au token 89, 0,79 $ (*mesuré*). Contre Planes14 : −2,3 % de débit, −0,20 Go.
- Planes12x reste non défaut par arbitrage : transcodage au chargement 1 340 s (*mesuré*, [mesures/g-horloges-planes12x-2026-08-23.txt](mesures/g-horloges-planes12x-2026-08-23.txt)). « câblé n'est pas mesuré » est périmé.

## 2026-08-24. Soumission TACO, D1

- Papier soumis à ACM TACO (TACO-2026-428) au commit `e21a8bb`, QTIP dans le corps. Desk reject le 08-27.
- D1, 0,24 $ : fusion `q+k+v` et `gate+up` par lignes, 252 → 144 matvec/token, ×1,061 [1,050–1,069] intra-job, bande [1,00 ; 1,12] (*mesuré*, [mesures/d1-fusion-servie-2026-08-24.txt](mesures/d1-fusion-servie-2026-08-24.txt)).
- Décomposition : 87,0 → 94,9 [94,1–95,2] (hissage de rotation) → 100,6 [99,9–100,7] tok/s. Le ×1,091 du hissage est inter-jobs, non publiable.
- Six critères verts : 128 tokens identiques, divergence au token 89, +3 686 400 octets exacts, même sha256 NVRTC (64 776 octets).
- Écrit ce jour : « les tables publiées restent à ROT_SHARE=0/FUSE=0 ». Levé le 08-31.
- Le front du projet est désormais la géométrie de lancement, celle que `nullk` mesure.

## 2026-08-25 au 08-27. Bits de gain, bruit MMLU, tampons, Zenodo

- Le dépôt reste public pendant la revue (08-25). « dépôt privé » de la note de soumission est périmé.
- Bits de gain, 0.6B/28 blocs, iso-débit 2,1656 b/poids, 86 min de Mac : leech0c13 39,3309, leech2c11 39,5350, leech1c12 43,4865, leech4c10 47,1537 (*mesuré*, [mesures/gain-ab-gate-0.6b-2026-08-25.txt](mesures/gain-ab-gate-0.6b-2026-08-25.txt)).
- L'échelle des bits de gain est réfutée : la graine 1 inverse le classement, un bras bouge de 13,9 % contre 10,6 % d'écart entre les quatre. Biais radial +3,69 % (*mesuré*, [mesures/cosdiag-biais-radial-0.6b-2026-08-25.txt](mesures/cosdiag-biais-radial-0.6b-2026-08-25.txt)).
- Bruit MMLU inter-graines au 4B : 58,02 / 52,19 / 55,17, s = 2,92 pp, 0,58 $ (*mesuré*, [mesures/bruit-mmlu-graines-4b-2026-08-25.txt](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)). Prédiction 0,5-1,5 pp réfutée ; échelle de volume non lancée, ~19 $ économisés (*estimé*).
- 08-26 : 16 des 20 `.ots` portent 3-4 ancres Bitcoin (*mesuré*, [mesures/ots-etat-2026-08-26.txt](mesures/ots-etat-2026-08-26.txt)). « 0 ancre, 4 pending » du 08-25 est réfuté : grep aveugle à une étiquette binaire de 8 octets.
- Préregs 08-10 et 08-11 : la passe d'anonymisation (`01fdbe6`) a réécrit leurs octets ; aucun des 128 blobs git ne rend le condensat. Cette dette se déclare dans le papier.
- 08-27 : desk reject TACO sur le périmètre ; `ots upgrade` 20/20 ancrés ([mesures/ots-etat-2026-08-27.txt](mesures/ots-etat-2026-08-27.txt)) ; Zenodo DOI de concept 10.5281/zenodo.22133606.
- Plan de clôture : 9 lots, 9 à 13 $ (*estimé*). La passation du soir devisait 0,49-0,55 $ pour un job déjà réussi : cinquième prise de la règle de rétention.

## 2026-08-28 au 08-30. Plan d'après-dépôt, piles isolées

- Plan d'après-dépôt (08-29) : gel ~0,25 $, géométrie ~2-4 $, qualité ~12-25 $, familles ~17 $, MoE ~65 $ (*estimé*). Mini-papier « calibration de la hessienne » enterré. Brouillons de diffusion écrits, non publiés.
- Phase P (port vLLM avant la géométrie) posée le soir ; renversée le 08-31 (`deaa449`).
- Premier gate M3 rouge sur nous : agrégat macro 72,85 ; en micro 70,36 ; f16 sur quatre moteurs [70,3 ; 70,9] (*mesuré*, [mesures/m3-gate-mmlu-vllm-2026-08-30.txt](mesures/m3-gate-mmlu-vllm-2026-08-30.txt)) ; second gate 70,34 (*mesuré*, [mesures/m3-gate2-mmlu-vllm-2026-08-30.txt](mesures/m3-gate2-mmlu-vllm-2026-08-30.txt)).
- IQ2_XXS sur Metal : 2,0625 bpw, ×2,6287, MMLU 39,39 ; LLVQ − IQ2_XXS +16,20 pp [+12,64 ; +19,72], seuil de lecture ~6 pp (*mesuré*, [mesures/m3-iq2-metal-2026-08-30.txt](mesures/m3-iq2-metal-2026-08-30.txt)). Servi 2,479 contre 5,162 b/param.
- Même GGUF sur CUDA : ×3,688, MMLU 38,87, 96 désaccords (*mesuré*, [mesures/m4-iq2-cuda-2026-08-30.txt](mesures/m4-iq2-cuda-2026-08-30.txt)). llama.cpp f16 84,83 tok/s, vLLM 83,09 : accord 2,1 %.
- GPTQ 2 bits : artefact 1 754 463 312 octets, 3,489 b/param (*mesuré*, [mesures/m3-gptq2-production-2026-08-30.txt](mesures/m3-gptq2-production-2026-08-30.txt)) ; « 3,182 » au dénominateur gptqmodel périmé. MMLU 24,74 dégénéré, non publiable.
- Campagne M3/M4 : 1,29 $ sur 11 lignes (*mesuré*, [data/jobs.csv](data/jobs.csv)) ; le protocole en compte 12, écart non élucidé.

## 2026-08-31. Vague 2, gel v1

- Fusion aux trois tailles : ×1,055 [1,054–1,058] au 8B, ×1,028 [1,027–1,029] au 14B, bande [1,00 ; 1,12] ; surcoûts +4 423 680 et +6 717 440 octets exacts (*mesuré*, [mesures/vague2-fusion-8b-14b-2026-08-31.txt](mesures/vague2-fusion-8b-14b-2026-08-31.txt)).
- Config servie v1 gelée : Planes14 + q8 + ROT_SHARE=1 + FUSE=1, 100,6 / 75,5 / 46,8 tok/s dans 2,57 / 5,41 / 9,40 Go. Règle écrite avant les chiffres.
- L'interdit « un 4B fusé isolé casserait la propriété » est levé par le gel. La série à tête identique n'est pas re-mesurée sous v1.
- Préreg commité 77 s avant la création du job (*mesuré*, git). Space en BUILD_ERROR ~9 h 40 (*mesuré*, [../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md](../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md) §É2).
- Décision d'opérateur `deaa449` : A2 et A3 avant le port vLLM. Protocole piles isolées v2 tamponné, constantes ancrées : 4 022 468 096 params (*mesuré*, quatre instruments), étalon f16 [70,3 ; 70,9] ([../proofs/protocole-piles-isolees-v2-2026-08-31.md](../proofs/protocole-piles-isolees-v2-2026-08-31.md)).
- Vérification adversariale de l'alignement v1 : 25 agents, 7 surfaces.

## 2026-08-31 (soir). A1, A4

- A1 : nullk 144 contre 252 lancements, 1,794 contre 2,200 ms, r = 0,8158 [0,8150–0,8162] (*mesuré*, [mesures/a1-nullk-252-144-2026-08-31.txt](mesures/a1-nullk-252-144-2026-08-31.txt)) ; 3,76 µs/lancement (*calculé*, 0,406 ms sur 108 lancements). Prior 0,83 confirmé à 1,7 %.
- A1 est mort quatre fois avant de rendre un chiffre, trois fois d'infrastructure et une de lanceur, pour 0,02 $ ; chaque mort est au fichier d'écarts ([../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md](../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md)).
- r tombe dans la bande mixte, entre les seuils 0,65 et 0,90. L'ordre A2/A3 revient à l'opérateur.
- A4 sur A100 : r = 0,8198, temps étirés ×1,809 (horloges 1,787) ; fusion ×1,063 ; fusé 63,4 contre dense 51,4 tok/s, 0,83 $ (*mesuré*, [mesures/a4-a100-2026-08-31.txt](mesures/a4-a100-2026-08-31.txt)). F4 reproduit (0,79×, 0,73×, 1,14×, 1,69×).
- Vague 2 complète : 2,17 $ sur un plafond de 5 (*mesuré*, [data/jobs.csv](data/jobs.csv)).

## 2026-09-01. Préreg A2/A3

- Arbitrage : A2 (CUDA Graphs) d'abord, commit `833d630`, préreg sha256 `802006c5` tamponné avant tout job ([../proofs/preregistration-a2-a3-geometrie-2026-08-31.md](../proofs/preregistration-a2-a3-geometrie-2026-08-31.md)).
- Pool par-lancement extrapolé à 252 : 0,947 ms ≈ 43 % du plancher, linéarité déclarée (*calculé*).
- Seuils : adoption ≥ 8 % bout-en-bout, clôture < 3 %, gate banc A3 ≥ 10 %, kill de phase < 8 % cumulés, plafond 4 $.
- Priors déclarés défavorables : CUDA Graphs fermé au lot A à 0,167 ms = 0,8 % d'un token (*mesuré*, lot A), rouvert sur décision. Dev préallocation KV : 2-4 jours (*estimé*).

## 2026-09-01 au 09-02. A2 et A3 rendus

- A2 étape 1 : prealloc/cat 0,8919 [0,8884–0,8953], prior 1,00 réfuté ; store étendu 0,9917 [0,9883–0,9935] (*mesuré*, [mesures/a2-verdict-2026-09-01.txt](mesures/a2-verdict-2026-09-01.txt)).
- A2 graph hybride au 4B : 99,2 → 112,5 tok/s [112,4–112,6], +13,45 % [13,36–13,58] ; 8B +10,1 % ; 14B +6,1 % (*mesuré*, [mesures/a2-transfert-verdict-2026-09-01.txt](mesures/a2-transfert-verdict-2026-09-01.txt)), 0,87 $. Adopté au critère ; point de courbe au 14B ; pas de gel v2.
- A3, huit variantes d'occupation, 1 105 920 lignes bit-exactes (*mesuré*, [mesures/a3-occupation-banc-2026-09-01.txt](mesures/a3-occupation-banc-2026-09-01.txt)). pers rend +1,56 % [+1,01 ; +1,86], sous le gate. persall rend +26,36 % [+25,31 ; +26,61], bras de banc non portable.
- Split-K sk1 rend −1,87 % : « le sous-remplissage de o/down est le résidu » est réfuté. Kill de phase non déclenché. Phase A : 1,11 $ sur 4.
- 09-02, décision d'opérateur : A2 n'est pas servi. Fenêtre KV 8k : +1,21 Go sur 2,57, +47 % de VRAM pour +12,6 % de débit (*calculé*, jamais mesuré, [../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md) §É7). Au 8B +22 %, au 14B +14 % ; à 2k +12 %.
- Seule fenêtre ayant tourné : prealloc(256), 0,038 Go (*calculé*, ECARTS §É7) ; le −0,83 % de é1b est un coût en temps (*mesuré*, [mesures/a2-verdict-2026-09-01.txt](mesures/a2-verdict-2026-09-01.txt)). `KvStore::Cat` reste le défaut ; `LLVQ_KV_PREALLOC` et `LLVQ_GRAPH_AB` sont des modes de mesure.
- Compteurs : 102 jobs pour 92,51 $, 28 `.ots` dont 20 ancrés (*mesuré*, [mesures/ots-etat-2026-09-02.txt](mesures/ots-etat-2026-09-02.txt)). « 89 jobs / 90,55 $ » du 08-31 périmé.

## 2026-09-02. D0, la roadmap recherche

- Roadmap recherche adoptée en trois OK, fusionnée dans `main` (`1e8583c`), plafond de 5 $ pour la vague 1, M1 en parallèle sur le Mac.
- M2 passe devant M1 : A/B à fichier constant, barre 0,43 pp, devis ≈ 2,3 $ (*estimé*, 0,19 $/bras). Q5 s'ouvre si la cible est `k` (+0,05 b/poids) ; une cible `down` coûterait ≥ +0,49 (*calculé*).
- Plomberie vérifiée sur Mac : k_proj 94 371 840 poids, « all restauré » = checkpoint à 114/114 picks (*mesuré*, [mesures/m2-plomberie-mac-2026-09-02.txt](mesures/m2-plomberie-mac-2026-09-02.txt)).
- Livrés : `LLVQ_RESTORE_F16=<types>|all` dans `bin/mmlu` et `bin/ppl` (exige `LLVQ_MODEL`, refuse l'inconnu) ; `LLVQ_H_SHRINK=ρ` dans `bin/smoke`. Branche `recherche/m1-m2-vague1`, `main` = `origin/main`.

## 2026-09-02 (suite). M2, M2b, M1

- M2, job `6a97ea8e`, 72 min, 2,17 $ : 11 bras, contrôles 55,59 et 70,32 à 2280/2280 picks (*mesuré*, [mesures/m2-attribution-4b-2026-09-02.txt](mesures/m2-attribution-4b-2026-09-02.txt)). Devis 2,3 $ périmé.
- Gains appariés par type restauré, en pp de MMLU (*mesuré*, même journal) :

  | restauré | gain [IC95] |
  |---|---|
  | gate | +5,18 [3,04 ; 7,34] |
  | up | +4,94 [2,72 ; 7,17] |
  | v | +4,48 [2,39 ; 6,61] |
  | down | +2,96 [0,71 ; 5,17] |
  | o | +2,35 |
  | k | +2,09 |
  | q | +1,85 |
  | attention | +6,90 |
  | MLP | +10,78 |
  | tout | +14,73 |

- Le prior littérature (k_proj, attention) est réfuté. Cible v_proj : 2,6 % des poids, rendement 8× la meilleure cible MLP (*calculé*).
- Écart É1 : v_proj en f16 = +0,263 b/param (5,425 > AWQ 5,302) ; en int4 g128 = −0,013 (5,149) (*calculé*). Cause : Planes14 déplie à 4,804 b/poids pour 2,07 d'information.
- M2b, job `6a986698`, 10 min, 0,29 $ : v_proj int4 g128 déquantifié rend MMLU 59,19, +3,60 [1,47 ; 5,79], McNemar 2,0e-4, 80,4 % du gain f16 (*mesuré*, [mesures/m2b-v4bits-2026-09-02.txt](mesures/m2b-v4bits-2026-09-02.txt)).
- La règle du préreg a un trou : ligne 1 exige IC > 1,5 (borne 1,47), lignes 2-3 exigent G4 < 3,0. Décision d'opérateur en attente ([../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).
- M1, 0 $, 12 runs Mac, 0.6B/28 blocs, médiane et étendue sur 3 graines : ρ = 1 39,6042 / 4,6214 ; 0,9 27,0812 / 3,1498 ; 0,7 27,4944 / 0,6847 ; 0,5 27,9506 / 2,9771 (*mesuré*, [mesures/m1-hessienne-shrink-2026-09-02.txt](mesures/m1-hessienne-shrink-2026-09-02.txt)). Contrôle 38,4507 rejoué.
- Par la règle du préreg : ρ* = 0,7, M1 vert, prédiction signée de kill (ρ* = 1) réfutée. Sur n = 3 l'étendue tient à une graine ; robuste : le signe et l'ordre de grandeur (−12 ppl, graines 2-3 de 3,47 à ≤ 0,54). Q1 adopte ρ ∈ [0,5 ; 0,9] à ré-estimer ; n/N 0,023 contre 0,074 au 4B (*calculé*).
- Écart M1 : file passée en nice 10 à la 5e mesure (CPU 1470 %, RSS 1,22 Go), ppl bit-exactes ; règle `LLVQ_THREADS ≈ ncpu−4` et nice dès le lancement. Note F1 : rétention projetée 88,9-89,6 % sous le kill 90,3 (*estimé*), F1a compte avant de coder.
- Soumission arXiv 7927047 refusée : `paper.pdf` téléversé à la place des sources ; `\pdfoutput=1` ajouté à `main.tex` pour la resoumission (`e721bc5`, git). Préregs tamponnés : m2-attribution `71712e60`, m1-hessienne-shrink `5a5e1027`, m2b-v4bits `263ec52a`, ancrage en attente. Vague 1 : 2,46 $ dépensés sur 5.
