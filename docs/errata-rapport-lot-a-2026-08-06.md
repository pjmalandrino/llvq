# Errata — rapport du lot A (2026-08-06)

> Vérification adversariale du [`rapport-lot-a-2026-08-06.md`](rapport-lot-a-2026-08-06.md)
> contre les mesures brutes, menée le jour même par deux vérificateurs
> indépendants (arithmétique, protocole). **Les trois conclusions de fond
> survivent** — domination du 4 bits sur ce 4B, noyau réel à ÷2,45 mesuré,
> CUDA Graph négatif — chacune portée par des mesures directes. Mais une
> erreur grave et sept mineures sont à porter au dossier.

## GRAVE — §1/§6/§8 : deux quatre-bits confondus, et un critère interdit

1. **Le rapport mélange deux objets sans le dire.** Le « 4,156 b/poids » de
   §1 est l'**AWQ w4 g128 mesuré** (fichier réel 2,666 Go, comptabilité
   bouclée à l'octet). Le « ~4,50 b/poids » de §6 et le « 2,26 Go » de
   disque sont le **MLX q4 g64 — un artefact absent de la campagne**.
   L'empreinte VRAM de l'AWQ dans son propre moteur est chiffrée au dossier
   à **5,302 b/param** (`plan-de-test-v2-cuda.md:278`), pas 4,50. La
   conclusion « seul avantage : le disque » tient (1,77 < 2,67 Go), mais le
   chiffre imprimé appartient à un autre fichier.
2. **Le critère du §8 — « descendre les 5,51 sous les 4,50 » — est la
   comparaison que le dossier interdit** (`portage-noyau-cuda.md:31` : « Ne
   jamais reposer “5,51 contre 4,50” »). 5,51 = projections seules, hors
   embedding ; 4,50 = q4 modèle entier, embedding quantifié compris. La
   forme homogène sur le 4B (b/param modèle entier, embedding compris) :

   | | b/param modèle entier |
   |---|---|
   | LLVQ Slot32 + embedding f16 (aujourd'hui) | **6,52** |
   | AWQ w4 g128 dans son moteur | **5,30** |
   | MLX q4 g64 | 4,50 |
   | C1 (Planes14 ~4,80 thesis) + embedding f16 | 5,88 |
   | **C1 + embedding int8** (validé sans perte au lot B) | **5,11 — passe sous l'AWQ réel** |
   | E1b′ overlay (~4,36) + embedding int8 | 4,71 |
   | E2 (~3,3) + embedding int8 | 3,76 — passe sous tout |

   **Conséquence pour C1 : le run garde tout son sens** (la question posée —
   Planes14 aussi vite que Slot32 à −0,71 b/poids, qualité identique — ne
   dépend pas du critère), mais **la condition de victoire face au 4 bits
   se dit en b/param modèle entier, et la chaîne honnête est C1 + embedding
   int8** (dont la gratuité qualité a été mesurée au lot B). Battre le
   MLX q4 (4,50) exige le barreau overlay ; battre l'AWQ réel (5,30) est
   atteignable dès C1+int8.

## MINEUR — sept items, conclusions intactes

1. Table §1 : « 42,7 → 47,0 (×1,08) » — le ×1,08 est formé sur le 43,5 du
   même job (juste), mais imprimé face au 42,7 du protocole miniature
   (47,0/42,7 = ×1,10). Même motif ligne chargement : « 209 → 128 (÷1,47) »
   mélange trois jobs (209 = miniature ; le ÷1,47 = 187,8/127,4).
2. Le glissement absolu −0,50 pp du bras LLVQ entre le run Metal du 02-08
   (56,09) et le run CUDA (55,59), 5× celui de la baseline, n'est ni
   commenté ni vérifiable par empreinte (le log du 02-08 est antérieur à
   l'impression des empreintes). Les *deltas* restent cohérents.
3. « −0,28 pp, sous ±1,25 » : les questions étant appariées, l'écart-type
   pertinent est celui des paires discordantes (McNemar), jamais calculé —
   à 3-8 % de discordance, σ ≈ 0,4-0,6 pp et la conclusion tient, mais la
   justification écrite est incorrecte.
4. « Trois instruments concordent » : la source elle-même écrit « Ne pas
   confondre » — 1,85 µs est la soumission CPU seule, 3,63 le bout-à-bout ;
   et `g`/`ε` changent de valeur (×2) entre `attribution-cuda` et `a3-graph`
   sous le même symbole. Les conclusions d'A3 reposent sur le job à trois
   bras et tiennent.
5. « ~2,2 $ sur quatorze jobs » (rapport) contre « 2,19 $ sur douze »
   (passation), même jour ; « trois erreurs pour 0,34 $ » — la source n'en
   facture que deux ; « 209 s payées par les échecs » — c'était 184-188 s.
6. Le ×1,08 est une invocation unique, sans plage (le job à 32 tokens rend
   ×1,05) — contraire à la discipline « une plage, pas un point » ; l'écart
   est expliqué (amortissement du warmup) mais jamais présenté en fourchette.
7. Deux affirmations du §2 dépassent la mesure : « deux candidats à un
   arrondi l'un de l'autre » (inférence — `fusedrun` compare des tokens, pas
   des logits) et « grammaticales et de même qualité » (jugement non
   archivé). La classification tie-break reste indépendamment soutenue.

## Ce qui a été vérifié et tient tel quel

÷2,45 · ε = 0,915 ms, 18 %, 0,8 % d'un token · −14,73 pp et les ratios de
perplexité · 1,89× [1,88–1,90] · toute l'attribution du §4 · la décomposition
plomberie du §3 · le 4,15625 lui-même, bouclé à l'octet · la fidélité de la
déquantification AWQ (contrôles L2/L1, mutation 26/27, échelles repliées des
RMSNorm retrouvées par sha256) · le bornage « un seul modèle, axe d'échelle
non testé » du verdict produit.
