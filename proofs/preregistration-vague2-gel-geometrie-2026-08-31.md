# Pré-enregistrement — vague 2 : geler la config servie, et deux bras de géométrie (2026-08-31)

> Écrit **avant tout job**, sous le protocole ancré
> `proofs/protocole-piles-isolees-v2-2026-08-31.md` (sha256 `987a07f4…`), dont
> les gates G1-G7 s'appliquent tels quels. Fronts choisis par l'opérateur le
> 2026-08-31 : **Phase 0** (gel) et **Phase A, bras inconditionnels A1 + A4**
> du [`plan-apres-depot-2026-08-29`](../docs/plan-apres-depot-2026-08-29.md).
>
> Ce document ne s'édite jamais. Écarts →
> `proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md`, nommé ici
> d'avance. Tampon à poser avant la première milliseconde.

## §1 — Ce qui est connu à la signature

| grandeur | valeur | provenance |
|---|---|---|
| D1, 4B : fusion sur le chemin servi | **×1,061 [1,050–1,069]** à ROT_SHARE constant, bande pré-posée [1,00 ; 1,12], six critères verts | `d1-fusion-servie-2026-08-24.txt` |
| D1, 4B : décomposition | 87,0 → 94,9 (hissage) → **100,6 [99,9–100,7]** tok/s | même journal |
| B2, 8B : servi q8 · tête id. · dense | 68,2 [68,2–68,3] · 34,1 [34,0–34,1] · 26,4 [26,4–26,5] | `b2-fusedrun-plages-2026-08-18.txt` |
| B2, 14B : servi q8 · tête id. · dense | 43,3 [43,2–43,4] · 23,9 [23,8–24,0] · 17,0 [17,0–17,0] | même journal |
| F4, A100 : aucun bras à décodage ne bat FP16 | `Planes14` **0,79×** · `nullk` 1,68× · AWQ 1,82× · cuBLAS 1,14× | `f4-a100-2026-08-18.txt` |
| lot G : le ×1,78 A100/L40S EST le rapport d'horloges | 2 520/1 410 = 1,787 ; `nullk` ×1,772/×1,781 | `g-horloges-planes12x-2026-08-23.txt` |
| plancher `nullk` 4B, L40S, géométrie 252 | 2,306 ms (0 octet de poids lu) | F2 |
| coûts de référence | fusedrun 8B 0,63 $ · 14B 1,27 $ · banc ~0,2 $ · a100-large ~0,9-1,0 $ | `jobs.csv` |

🚨 **Le biais de cette vague** : D1 a rendu un beau chiffre au 4B, et le gel sur
FUSE=1 est le résultat *attendu*. Un 8B ou un 14B qui raterait sa bande serait
le résultat défavorable — c'est lui qui exige la lecture la plus soigneuse, pas
l'inverse.

## §2 — Le protocole, figé

### 0.1 — Rejouer la fusion au 8B et au 14B (~1,90 $)

La table publiée aux trois tailles repose sur « une seule configuration
partout » ; un 4B fusé isolé la casserait (CLAUDE.md §2). Donc : les trois
tailles, ou aucune.

- Forme **D1 exactement** : `LLVQ_FUSE_AB=1` — les deux bras (FUSE=0/FUSE=1,
  ROT_SHARE=1 des deux côtés, `check_fuse` l'exige) dans **un seul processus**,
  médianes 5 rounds, `planes14` + `q8`.
- **Les six critères de D1, réutilisés tels quels** : tokens identiques entre
  bras fusés ; divergence au dense au même token que B2 ; delta d'octets exact
  et calculé d'avance ; même sha256 NVRTC des deux côtés ; médianes à plage ;
  rapport intra-job seulement.
- **Bande, la même que D1** : gain de fusion ∈ **[1,00 ; 1,12]** à chaque
  taille. En dessous de 1,00 : la fusion ne transfère pas — fait à publier.

### 0.2 — Le gel, et sa règle écrite avant les chiffres

**Si** 8B et 14B tiennent leur bande → « **config servie v1** » =
`planes14 + q8 + ROT_SHARE=1 + FUSE=1`, aux trois tailles, et toutes les
surfaces publiées s'alignent (0 $).
**Sinon** → la config servie reste `ROT_SHARE=0, FUSE=0` (les médianes B2), le
gel se fait dessus, et le ×1,061 du 4B reste un résultat D1, pas une config.
Règle du plan Phase 0 : débit d'abord — l'écart VRAM (+0,21 Go au 4B) ne change
pas de classe de matériel.

### A1 — `nullk` sous géométrie fusée (~0,2 $ + petit dev)

**La question** : le plancher de 2,306 ms suit-il le compte de lancements
(252 → 144) ?

- ⚠️ **Dev préalable, déclaré** : le banc n'a pas de `nullk` sur formes fusées
  (vérifié le 2026-08-31 : `tv_planes_seg` existe, pas de `nullk_seg`). Le
  bras = le noyau `tv_nullk` existant sur la **liste de formes fusées**
  (d_out sommés q+k+v et gate+up) — une liste de formes, pas un noyau neuf.
- Un seul processus, `nullk`(252) et `nullk-fusé`(144) entrelacés, 7 rounds
  dont 2 jetés.
- **Lecture, posée d'avance** — et le test discrimine (gate G7) :
  `r = t(nullk-fusé) ÷ t(nullk)` ; **r ≤ 0,65** → le poste est la latence par
  lancement, A2 (CUDA Graphs) devient prioritaire ; **r ≥ 0,90** → le poste est
  l'occupation, A3 ; entre les deux → mixte, publier les parts, aucun des deux
  n'est éliminé.

### A4 — La géométrie gagnante sur A100 (~0,9-1,0 $)

- `a100-large`, `LLVQ_NVRTC_ARCH=compute_80`, banc (`planes14`, `planes14_seg`,
  `nullk`, `f16`, `cublasf16`) **+** `fusedrun` dans la config issue de 0.2.
- **Lecture, posée d'avance** : un bras réseau **≥ 1,00× FP16** sur A100 → la
  réserve « résultat Ada » du papier saute ; sinon → l'attribution horloge du
  lot G s'étend à la géométrie fusée et la réserve devient un **mécanisme**
  mesuré à deux points. Les deux issues se publient.
- 🚨 Aucun × A100 ne se divise par un × L40S — jamais (protocole v2 §4.1).

## §3 — Issues, y compris celle qui manquait

| issue | conséquence |
|---|---|
| 8B et 14B dans la bande | gel FUSE=1, surfaces alignées, la table à trois tailles migre |
| une taille rate sa bande | gel sur B2 (ROT_SHARE=0/FUSE=0) ; le fait défavorable se publie avec son mécanisme |
| **un bras ne tourne pas** (OOM 14B, refus de forme, image) | le job s'arrête, le refus est daté, rien n'est lu — leçon n°10 |
| A1 rend r intermédiaire | les parts se publient, ni A2 ni A3 n'est éliminé |
| A4 : aucun bras ≥ FP16 | la réserve A100 devient mécanisme — c'est une issue de publication, pas un échec |

## §4 — Budget

Nominal **~3,0 $** (0,63 + 1,27 + 0,2 + 0,9). **Plafond : 5 $** — au-delà,
arrêt et retour à l'opérateur. Ordre : 0.1 (8B puis 14B) → 0.2 (0 $) → A1
(après son dev) → A4 (dans la config gelée).
