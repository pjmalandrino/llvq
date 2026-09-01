# Écarts au préreg A2 é2+é3 (af6c12d2…)

## É1 — Gate 2 rouge au premier essai, mécanisme INSTRUIT puis corrigé (2026-09-01, 0,18 $ cumulés de diagnostic)

Premier job (`6a96bd7a…`, 0,09 $) : gate 1 VERT (la capture existe — le pas
entier s'enregistre, noyaux candle + cuBLAS + les nôtres, sans
invalidation), gate 2 ROUGE (divergence éager/graph au token 2, replay 1
juste). Aucun chrono rendu — le préreg l'exigeait.

Diagnostic instrumenté (`LLVQ_GRAPH_DIAG=1`, job suivant, 0,09 $) : token
par token, replay PUIS éager sur le même état, logits comparés en entier,
canaux relus depuis le device. **Verdict mesuré** : le PREMIER lancement du
graph rend des logits inexacts (max|Δ| = 1,12e1) puis TOUS les suivants
sont EXACTS au bit près (0,000e0, douze tokens) — les relectures device
étaient justes partout. Le graph est un rejoueur parfait sauf à son
lancement inaugural — celui qui matérialise les nœuds d'allocation
AUTO_FREE. Explication unifiée du rouge : replay 1 légèrement faux
(l'argmax avait survécu), ses écritures KV polluées faisaient dériver le
replay 2 ; au diag, l'éager re-nettoyait l'état à chaque pas.

Correctif : un lancement JETABLE consommé sitôt la capture rendue (à
blanc, sur l'état de la capture, que chaque génération re-préfille) ; en
secours un mode HYBRIDE (premier token de décodage éager, replay ensuite),
les deux gates imprimés, le chrono sur le bras qui passe. Le protocole et
la lecture du préreg sont inchangés.
