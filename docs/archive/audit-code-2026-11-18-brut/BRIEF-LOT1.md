# Brief commun du lot 1 (corrections évidentes et peu critiques)

Tu es un des quatre agents qui appliquent le lot 1 de l'audit du 2026-11-18. Chaque agent possède une liste de
constats et une liste de fichiers. Ta liste est dans ton message de mission. Les trois autres agents écrivent en même
temps dans le même arbre : ne touche QUE tes fichiers.

## Où

- Racine de travail, la seule : `<the audit worktree>`,
  branche `audit/code-2026-11-18`, partie de `main` à `925930d`.
- Les constats sont dans `docs/archive/audit-code-2026-11-18-brut/` : fichiers `01-` à `12-` (constats), `20-` à `23-`
  (vérification contradictoire). Pour chaque ID de ta liste, lis le constat, puis cherche son ID dans les fichiers
  `20-` à `23-` : si un vérificateur l'a NUANCÉ, c'est SA version qui fait foi, y compris sur la forme de la
  correction (« Correction sûre pour un lot 1 ? »).
- Lis `CLAUDE.md` (du worktree) et `docs/STYLE.md` avant d'écrire.

## Ce qu'est le lot 1

Des corrections évidentes, locales, qui ne changent ni un format, ni la carte d'index, ni un chiffre publié, ni un
token servi, ni le comportement du chemin servi. Dans le doute, tu ne corriges pas : tu écris le constat dans ta
section « Laissé de côté » avec la raison. Un constat laissé de côté n'est pas un échec.

Avant chaque correction, revérifie le constat toi-même dans le code. S'il est faux, ne corrige rien et dis-le.

## Interdits absolus

- Ne modifie jamais : `CLAUDE.md` ; tout fichier `*.cu`, `*.cuh`, `*.metal` ; `llvq-tetra/llvq_tetra/data/*` ;
  `configs/*.json` ; `docs/archive/*` (sauf ton journal) ; `docs/mesures/*` ; `docs/data/jobs.csv` ; `proofs/*` ;
  `paper2/*` ; `ops/jobs/*` (des préregs nomment leur sha256) ; `ops/Dockerfile*` ; `.github/*` ; `Cargo.toml` et
  `Cargo.lock` ; les fixtures de test.
- Ne change jamais une chaîne imprimée sans avoir vérifié qu'aucun script ne la lit : `git grep -nF '<la chaîne>' --
  ops/ paper2/ docs/METHODE.md .github/`. Les portes des jobs lisent des lignes par `grep -F`.
- Aucune suppression de fonction, de fichier ou de binaire. Le code mort est un autre lot.
- Aucun `git add`, `commit`, `checkout`, `restore`, `stash`, `reset`, `push`. Tes modifications restent non commitées.
  ATTENTION : pour annuler un essai ou un mutant, défais-le par l'édition inverse. `git checkout -- fichier`
  effacerait aussi le travail déjà fait, le tien et celui des autres.
- Aucun réseau hors résolution de dépendances Python déjà en cache. Aucun `hf`, aucun `gh`, aucun job.
- Pas de sous-agents.

## Compilation et tests : légers, ciblés, en série

Le poste est un portable partagé. Règles :
- Toujours `nice -n 10`, toujours `-j 8`, toujours en debug. Jamais `--release`, jamais `--workspace`, jamais
  `--include-ignored`, jamais de feature `metal` ou `cuda`.
- Seulement des commandes ciblées : `nice -n 10 cargo check -p <crate> --all-targets -j 8` ou
  `nice -n 10 cargo test -p <crate> --test <nom> -j 8` ou `... --lib <filtre>`.
- Cargo sérialise les compilations entre agents par son verrou : si ta commande attend (« Blocking waiting for file
  lock »), c'est normal, laisse-la attendre. Donne-lui un délai de dix minutes.
- Le code Linux/CUDA (`cfg(target_os = "linux")`) ne compile pas sur ce Mac : ne lance pas de compilation croisée,
  la session principale s'en charge à la fin. Signale dans ton journal chaque fichier Linux que tu as touché.

## Règles d'écriture

- Code, commentaires et documentation en anglais. Journal en français.
- `docs/STYLE.md` : le fait d'abord, une idée par phrase, aucun tiret cadratin, pas de bannière. Un fait qui change est
  remplacé, pas empilé.
- Un commentaire dit ce que le code fait maintenant, sans raconter l'audit ni la correction. Pas de « fixed », pas de
  « was », pas de référence à un ID d'audit dans le code.
- Zéro avertissement clippy. Garde le style du fichier.
- Tout test nouveau doit pouvoir échouer : casse le code qu'il garde par une édition, constate qu'il rougit, défais
  l'édition, constate qu'il reverdit. Note le mutant dans ton journal. Avant de conclure qu'un mutant survit,
  vérifie qu'il est bien appliqué (le remplacement a trouvé son site).
- Ne tronque jamais un log de test avec `head` : compte les lignes `test result:`.

## Ton journal

Écris-le dans le fichier que ton message de mission nomme. Format :

```
# Lot 1, <ton domaine> : journal
## Corrigé
| ID | fichiers | ce qui change, en une ligne | vérification faite |
## Laissé de côté
| ID | raison |
## Constat faux ou déjà corrigé
| ID | preuve |
## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)
## Commandes lancées et leur résultat (les lignes `test result:` telles quelles)
```

## Ton message de retour

Moins de 20 lignes : le chemin du journal, le nombre de constats corrigés, laissés, faux, la liste des fichiers
modifiés, et l'état des tests que tu as lancés (vert ou rouge, avec la ligne). Rien d'autre.
