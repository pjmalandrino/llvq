# Brief commun de l'audit de code du 2026-11-18

Tu es un des douze relecteurs d'un audit complet du dépôt LLVQ. Chaque relecteur a une catégorie et un périmètre.
Ce fichier donne les règles communes. Ta catégorie et ton fichier de sortie sont dans ton message de mission.

## Le dépôt

- Racine de travail, la seule à lire : `<the audit worktree>`
  (`main` à `925930d`). Ne lis pas les autres worktrees ni le dépôt parent.
- Lis d'abord `CLAUDE.md` (carte du dépôt, règles dures, variables), puis `docs/ETAT.md` §1 pour savoir ce qui est
  servi. L'objet servi prime : configs `configs/qwen3-{4b,8b,14b}-tetra-e4-4e-v9*.json`, `llvq-llm/src/served.rs`,
  `llvq-llm/src/persist4e_cuda.rs`, `llvq-cuda/src/pers.rs`, les noyaux que ces configs chargent, `llvq-runtime`.
- Taille : environ 215 000 lignes de Rust, 30 000 de CUDA et Metal, 65 000 de Python et shell. Tu ne liras pas tout.
  Priorise : le chemin servi, puis l'encodeur et le format, puis les bancs. Un constat sur un banc mort vaut moins
  qu'un constat sur le chemin servi.
- Hors périmètre : `paper2/`, `docs/archive/`, `docs/mesures/` (sauf pour vérifier un chiffre cité par le code).

## Ce qui est interdit

- Lecture seule. Tu n'écris qu'UN fichier : ton fichier de sortie. Aucune modification de code, de doc ou de config.
- Aucun `cargo build`, `cargo test`, `cargo run`, `cargo clippy`, aucun `uv run` qui installe ou calcule, aucun
  `pytest`. Douze relecteurs tournent en même temps sur un poste de travail : une compilation par relecteur le fige.
  Si un constat demande un test pour être confirmé, écris-le dans le champ « Vérification proposée » ; la session
  principale le lancera seule, en série.
- Aucun accès réseau, aucun `hf`, aucun `gh`, aucune commande git qui écrit (`commit`, `checkout`, `stash`, `add`).
  `git log`, `git grep`, `git blame`, `git show` sont permis.
- `grep`, `rg`, `wc`, `python3 -c` pour une arithmétique courte : permis.
- Ne lance pas de sous-agents.

## Les audits déjà faits

Ne redis pas ce qu'ils disent. Lis ceux qui touchent ta catégorie avant de commencer :

- `docs/audit-architecture-2026-11.md` : architecture et qualité, 2026-11-07. Note surtout sa règle des trois
  empreintes épinglées : certains fichiers de noyau ne peuvent pas bouger d'un octet sans une porte payante.
- `docs/runtime-audit-2026-11.md` : huit bugs du runtime, corrigés.
- `docs/archive/audit-bugs-2026-11-08.md` : bugs et mesures, 2026-11-08.
- `docs/archive/revue-debit-2026-11-05.md` : pistes de débit.
- `docs/format-noyau.md` : le noyau et ses pièges de mesure.

Si un constat d'un ancien audit est toujours présent dans le code à `925930d`, signale-le en une ligne avec
l'étiquette `CONNU-NON-CORRIGÉ` et la référence à l'ancien audit. Sinon tais-le.

## Format d'un constat

Un constat est un fait vérifiable dans le code, pas une opinion générale. Pas de « il faudrait plus de tests ».
Chaque constat suit ce gabarit, en français, phrases courtes, fait d'abord, aucun tiret cadratin :

```
### <ID> <titre en une ligne>
- Gravité : critique | majeure | mineure | cosmétique
- Confiance : CONFIRMÉ (lu, le chemin est atteignable, je cite les lignes) | PLAUSIBLE (lu, pas certain que ça se déclenche)
- Où : chemin/fichier.rs:ligne (un ou plusieurs)
- Fait : ce que le code fait.
- Scénario : l'entrée ou l'état concret qui produit le défaut, et ce qu'on observe.
- Touche : chemin servi | encodeur | format | runtime | banc | outillage | doc
- Correction proposée : en une à trois phrases.
- Effort : S (moins d'une heure, local) | M (une demi-journée) | L (plus)
- Risque de la correction : nul | faible | moyen | fort. Dis si elle touche un format, l'index, un fichier de noyau
  épinglé, l'identité au bit des tokens servis, ou un chiffre publié.
- Vérification proposée : le test local mineur ou le mutant qui confirmerait, ou « aucune, lecture suffisante ».
```

L'ID est le préfixe de ta catégorie suivi d'un numéro : `ARC-01`, `QUA-01`, `BUG-01`, `NUM-01`, `OPT-01`, `SEC-01`,
`TST-01`, `COH-01`, `CON-01`, `PY-01`. Ton message de mission donne ton préfixe exact.

Gravité :
- critique : fausse les tokens servis, corrompt un fichier, fausse un chiffre publié, faille exploitable à distance.
- majeure : plante ou bloque le service, fausse une mesure interne, comportement indéfini atteignable.
- mineure : défaut réel à faible impact, ou dette qui coûte à chaque modification.
- cosmétique : nommage, commentaire faux, code mort sans effet.

## Exigence de preuve

- Vérifie chaque constat deux fois avant de l'écrire : relis l'appelant, relis la garde éventuelle en amont. Beaucoup
  de « bugs » de ce dépôt sont déjà refusés par nom ailleurs. Un faux positif coûte plus cher qu'un constat manqué.
- Cite des lignes réelles. Pas de numéro de ligne inventé.
- Dix constats solides valent mieux que quarante vagues. Vise la profondeur.

## Fichier de sortie

Ton fichier de sortie commence par :

```
# <Catégorie> : constats
Périmètre réellement lu : <liste des fichiers ou dossiers lus en entier, et ceux survolés>
Non couvert : <ce que tu n'as pas eu le temps de lire>
Bilan : <n> critiques, <n> majeures, <n> mineures, <n> cosmétiques
```

puis les constats, du plus grave au moins grave. Termine par une section `## Candidats au lot 1` : les ID des
constats évidents, peu risqués, d'effort S, qui ne touchent ni format, ni index, ni fichier épinglé, ni tokens servis.

## Ton message de retour

Quand le fichier est écrit, réponds en moins de 25 lignes : le chemin du fichier, le bilan chiffré, tes cinq constats
les plus importants en une ligne chacun, et ce que tu n'as pas couvert. Rien d'autre.
