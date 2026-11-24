# Brief commun des lots 2 à 7 (écriture sans exécution)

Tu es un des agents qui appliquent les lots 2 à 7 de l'audit du 2026-11-18. Lis d'abord `BRIEF-LOT1.md` dans ce même
dossier : ses règles tiennent toutes (où travailler, revérifier chaque constat, la version du vérificateur fait foi,
interdits git, règles d'écriture, format du journal, message de retour), SAUF les points que ce fichier remplace.

Le rapport d'ensemble est `docs/archive/audit-code-2026-11-18.md`. Le lot 1 est déjà appliqué dans l'arbre, non
commité : tu pars de cet état, tu ne le défais pas.

## Ce qui change par rapport au lot 1

### 1. AUCUNE exécution locale. C'est la règle la plus importante.

L'opérateur utilise sa machine. Tu ne lances RIEN qui calcule : aucun `cargo` (ni check, ni test, ni clippy, ni
build), aucun `uv`, aucun `pytest`, aucun script du dépôt, aucun `rustc`, aucun `rustfmt`. Permis seulement : lire,
`git grep`, `git log`, `git diff`, `wc`, `python3 -c` pour une arithmétique courte, `python3 -m py_compile` sur un
fichier Python que tu as modifié.

Ton code ne sera donc pas compilé avant plus tard. Conséquences :
- Écris peu et sûr. Préfère la plus petite modification qui corrige. Pas de refactorisation d'opportunité.
- Avant d'appeler une fonction, un type ou une méthode, ouvre sa définition et recopie sa signature exacte. Vérifie
  les imports, la visibilité (`pub`), les durées de vie, les traits en portée, les `cfg`. Calque-toi sur un appel
  voisin qui compile déjà.
- Relis chaque fonction modifiée en entier, deux fois, comme le ferait le compilateur : types de chaque expression,
  emprunts, `?` sur le bon type d'erreur, bras de `match` exhaustifs, variables inutilisées (clippy est à zéro
  avertissement dans ce dépôt).
- Si une correction demande plus de trente lignes de code de production nouveau, ou touche une signature publique
  appelée ailleurs, range-la dans « Laissé de côté : à faire avec un compilateur » et décris le plan exact.
- Écris les tests quand même. Ils ne tourneront pas maintenant.

### 2. Ton journal porte une file de vérification

En plus des sections du lot 1, ton journal a une section :

```
## À lancer quand la machine est libre
| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
```

Une ligne par test nouveau ou modifié, plus la commande de compilation de chaque crate touchée. Pour le code
Linux/CUDA : `ops/check-linux-cuda.sh clippy -p llvq-llm --features cuda --all-targets`.

Dans la colonne « vérification faite » du tableau « Corrigé », écris honnêtement « relu, NON COMPILÉ ».

### 3. Périmètre de risque élargi, avec ses limites

Le lot 1 interdisait de toucher au chemin servi. Ces lots y touchent, mais seulement comme ton message de mission le
dit. Restent interdits pour tous :
- tout fichier `*.cu`, `*.cuh`, `*.metal`, et `llvq-tetra/llvq_tetra/data/*` ;
- `CLAUDE.md`, `docs/archive/*` (sauf ton journal), `docs/mesures/*`, `docs/data/jobs.csv`, `proofs/*`, `paper2/*`,
  `ops/jobs/*`, `ops/Dockerfile*`, `configs/*.json`, les fixtures ;
- tout changement qui modifie un token servi, un octet d'un format, la carte d'index, ou un chiffre publié dans un
  document ;
- toute chaîne imprimée qu'un script lit (`git grep -nF` dans `ops/` avant de la changer) ;
- toute décision qui revient à l'opérateur : une valeur de délai, un défaut qui change, un choix de structure
  (graphe de crates, features), un instrument de mesure. Dans ce cas, n'implémente pas : écris l'option que tu
  recommandes et pourquoi dans « Décision d'opérateur ».

Un refus nouveau ne doit se déclencher que sur une entrée qui produisait déjà une panique, un résultat faux ou un
état cassé. Une entrée valide se comporte exactement comme avant.

### 4. Tes fichiers

Tu ne modifies que les fichiers de ta mission. D'autres agents écrivent en même temps ailleurs dans l'arbre.
