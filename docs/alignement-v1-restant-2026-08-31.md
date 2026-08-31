# Alignement config servie v1 — ce qui RESTE, surface par surface (2026-08-31)

> Le gel (vague 2, préreg `e23e9895…` §0.2) dit : « les surfaces publiées la
> citent ». Le commit `5ff5c55` en a aligné trois (README, model card locale,
> CLAUDE.md) ; une vérification systématique du 2026-08-31 (25 agents,
> adversariale) en trouve **sept autres** qui donnent encore l'ancienne config
> comme courante. Deux corrigées ce soir (`format-noyau.md`, compteurs et note
> ots de `CLAUDE.md`). Le reste est listé ici avec ce qui le décide — les
> entrées ⚖️ attendent un arbitrage d'opérateur, les autres sont mécaniques.

## Mécaniques (aucune décision, juste du soin)

| surface | ce qui est faux | la correction |
|---|---|---|
| `docs/fiche-4b.md:612` | « 88,4-88,5 tok/s dans 2,60 Go » et « ×1,12 » donnés comme courants — **deux générations en arrière** (ni B2, ni v1) | annotation datée : médianes B2 (87,0 / ×1,11) puis v1 (100,6, `vague2-…-2026-08-31.txt`) |
| `docs/cheatsheet-defense.md:116` | même chose (48,7 · 88,4-88,5 · ×2,03 · ×1,12) | même correction, même source |
| `docs/HISTORIQUE.md` | « État courant (au 2026-08-25) » ; **aucune entrée après le 08-24** — ni desk-reject TACO (08-27), ni campagnes m3/m4, ni vague 2, ni gel ; et sa fin répète le « 0 Bitcoin » démenti le 08-26 | nouvel « État courant » + entrées 08-25→08-31 ; c'est le plus gros morceau et le plus important : c'est LE fil |
| `CLAUDE.md` (en-têtes de reprise) | l'en-tête du 08-25 donne encore 87,0 comme débit servi dans plusieurs phrases de narration | annotations 🕳️ datées aux endroits qui affirment, pas une réécriture |

## ⚖️ À arbitrer (chaque ligne engage autre chose que de la cohérence)

| surface | le conflit | l'arbitrage |
|---|---|---|
| `docs/hf-blog-article.md:100-101` | « 87.0 tok/s », « 2.60 GB » — et le plan (`plan-apres-depot-2026-08-29.md:395-397`) liste « publier le billet » comme décision d'opérateur en attente | aligner la copie locale AVANT publication (sinon une 3ᵉ config entre en circulation) — mais le choix des chiffres du billet (v1 ? B2 ? les deux ?) est éditorial |
| `paper/sections/evaluation.tex:51`, `integration.tex:83-85`, `availability.tex:24-28` | le papier donne 87,0/68,2/43,3 comme « served » et une commande `fusedrun` **sans** ROT_SHARE/FUSE | le manuscrit est entre deux venues (desk-reject TACO 08-27) : intégrer v1 est une décision de révision, pas un alignement mécanique |
| `docs/exp-piles-isolees-2026-08-30/MACHINES.md:50-52` | « Configuration publiée : ROT_SHARE=0 FUSE=0. 🚨 Ne pas activer la fusion » — instruction ACTIVE d'une campagne dont le **protocole v2 est tamponné** | si la campagne doit mesurer v1, ça s'écrit dans SON fichier d'écarts (le protocole ne s'édite pas) ; si elle reste à B2, l'instruction est juste et c'est le motif qui doit être re-daté |

## Déjà fait ce soir (2026-08-31)

- `docs/format-noyau.md` : « la configuration servie publiée » datée ; le
  « 8B/14B pas rejoués sous fusion » levé avec les chiffres de la vague 2.
- `CLAUDE.md` : compteurs re-mesurés (89 jobs / 90,55 $ / 88 entrées / 34 md /
  23 ots) ; la note « quatre tampons du 08-25 en attente » annotée (ancrés le
  08-27 ; les trois en attente sont du 08-30/31).
