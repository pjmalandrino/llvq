# Brief du second tour d'audit (2026-11-19)

Lis d'abord `BRIEF.md` dans ce dossier : ses règles tiennent toutes (lecture seule, aucun `cargo`, aucun `uv`,
aucun test, aucun réseau, pas de sous-agent, format d'un constat, exigence de preuve, message de retour). Ce fichier
ajoute ce qui change au second tour.

## L'état de l'arbre

La racine reste `<the audit worktree>`. Elle n'est plus
`main` à `925930d` nu : les corrections des lots 1, 2, 3, 4 et 7 du premier tour y sont appliquées, NON commitées.
`git diff` montre ce qui a changé depuis `925930d` (75 fichiers, environ 2 750 lignes ajoutées). Tu audites l'arbre
tel qu'il est maintenant.

Ne lance AUCUN processus qui calcule : l'opérateur utilise la machine. Lire, `git grep`, `git diff`, `git log`,
`wc`, `python3 -c` pour une arithmétique courte. Rien d'autre.

## Ce qui est déjà connu : ne le redis pas

- Le rapport du premier tour : `docs/archive/audit-code-2026-11-18.md`. Lis-le en entier d'abord.
- Ses constats bruts : fichiers `01-` à `12-` et `05b-` de ce dossier ; leurs vérifications : `20-` à `23-`.
- Les journaux des corrections : `30-` à `34-` (lot 1) et `40-` à `45-` (lots 2 à 7). Chaque journal a une section
  « Laissé de côté » et « Décision d'opérateur » : ce qui y figure est connu.
- Les audits antérieurs cités par `BRIEF.md`.

Un constat déjà écrit dans un de ces fichiers ne vaut rien ici. Tu cherches ce qu'ils ont manqué.

## Les ID

Préfixe `R2-` suivi du préfixe de ta mission et d'un numéro : `R2-DIF-01`, `R2-SRC-01`, etc. Ton message de mission
donne le préfixe exact.

## Ce qui sera fait de tes constats

Ils seront vérifiés, puis corrigés directement par un autre agent, avec un compilateur. Donc, pour chaque constat,
la « Correction proposée » doit être assez précise pour être appliquée : la fonction, la ligne, le changement, et le
test qui la garde avec son mutant. Dis toujours si la correction touche un fichier de noyau (`*.cu`, `*.cuh`,
`*.metal`), un format, la carte d'index, un token servi, un chiffre publié ou un instrument de mesure : ces
corrections-là ne seront pas appliquées sans l'opérateur.
