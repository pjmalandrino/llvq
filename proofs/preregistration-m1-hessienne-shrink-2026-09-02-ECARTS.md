# Écarts au pré-enregistrement M1 — écrits pendant le job, jamais dans le tampon

> Le pré-enregistrement
> [`preregistration-m1-hessienne-shrink-2026-09-02.md`](preregistration-m1-hessienne-shrink-2026-09-02.md)
> (sha256 `5a5e1027…`) est tamponné : il ne s'édite pas. Ce qui s'en écarte, ou
> ce qu'il n'avait pas prévu, s'écrit ici. Règle du §7 de `CLAUDE.md`.

## É1 — La file a été passée en `nice 10` à la 5ᵉ mesure sur 12 (2026-09-02, ~13 h 30)

**Ce qui a changé.** `renice 10` sur le processus de la file
(`ops/m1_shrink_queue.sh`, pid 39007) et sur le `smoke` en cours. Les bras
suivants en héritent. Avant : `nice 5`, ~1 470 % de CPU, soit ~15 cœurs sur 16.

**Pourquoi.** Demande de l'opérateur pendant le job : la machine est son poste
de travail et la file la rendait pénible. Mesuré au moment de la demande :
**CPU 1 470 %** contre **RSS 1,22 Go sur 64 Go, 68 % de mémoire libre** — la
ressource saturée est le processeur, pas la mémoire.

**Ce que ça ne change pas, et c'est la raison pour laquelle ça a été fait sans
attendre la fin de la file.** Le découpage de la quantification est **par
ligne**, exact et non approché ; `parallel_matches_serial_exactly` l'exige au
bit près. Le nombre de threads effectivement actifs ne déplace donc **aucune**
des grandeurs du §4 — les trois ppl, la médiane, l'étendue. Les contrôles du §3
(baseline 19,5038, débit 2,1656, ρ imprimé) sont eux aussi invariants, et ils
sont vérifiés sur les 12 bras.

**Ce que ça change.** Les **durées** consignées dans
`~/llvq-nuit-b/journal.txt` (« OK en N min »). Les quatre premiers bras ont
tourné à `nice 5`, les huit suivants à `nice 10`, sur une machine par ailleurs
utilisée. ⚠️ **Ces durées ne sont donc pas comparables entre elles**, et
aucune n'est un chiffre de performance : ce job mesure une dispersion de
perplexité, pas un temps.

**Ce qui aurait dû être fait, et qui vaut pour la prochaine file.**
Plafonner `LLVQ_THREADS` (≈ `ncpu − 4`) et lancer sous `nice` **dès le
départ**, plutôt que de corriger à mi-parcours. Le bouton existe et son propre
commentaire le prescrit (`llvq-llm/src/bin/smoke.rs` : *« an A/B launched on a
machine someone is working on should not take the whole machine »*). Il n'a pas
été utilisé parce que personne ne s'était demandé qui d'autre se servait de la
machine — ce n'est pas un oubli technique mais un défaut de cadrage, et il est
consigné comme tel. ⚠️ La file n'a **pas** été tuée pour être relancée
plafonnée : les quatre bras déjà rendus auraient été à refaire pour un gain de
confort, ce qui est un mauvais échange.
