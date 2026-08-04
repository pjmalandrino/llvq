# Passation — 2026-08-04

> À coller dans une nouvelle session. Tout ce qu'il faut pour reprendre sans
> relire l'historique.

---

## 1. Où on en est, en cinq lignes

Le modèle Qwen3-4B quantifié à 2 bits est **publié** sur Hugging Face et démarre
seul. Les surfaces publiques (README, carte du modèle) viennent d'être
**corrigées de 36 divergences** trouvées par audit. La campagne de mesure
comparative est **prête et volontairement non lancée**. La priorité est passée
au **portage CUDA du noyau fusé**, qui est la contribution du projet et qui ne
tourne aujourd'hui que sur Mac.

**Dépensé en machine à ce jour : ~0,35 $.**

## 2. État du dépôt

- Branche de travail : **`campagne-mesure`**, 12 commits, poussée, **non
  fusionnée dans `main`**.
- `cargo clippy --all-targets` : zéro warning.
- `cargo test --release -- --include-ignored` : **144 cas passent**.
- La carte du modèle sur Hugging Face a été **republiée deux fois** aujourd'hui
  (commits `1198670` puis `ad2d78a`).

⚠️ **La branche n'est pas fusionnée.** Décider si on la fusionne avant de
continuer, ou si le travail CUDA continue dessus.

## 3. Ce qu'il faut faire — la priorité

**Porter le noyau fusé de Metal vers CUDA.**

Pourquoi c'est la priorité et pas la campagne de mesure : le papier LLVQ
déclare lui-même son noyau CUDA **mono-coquille (M = 3), mono-couche, « pour
la simplicité », et plus lent que QTIP** (annexe C). Le décodeur Leech
**multi-coquilles** qu'exige le régime 2 bits n'existe nulle part. Le nôtre
existe, bat le FP16 de 2,06-2,08× sur les 252 projections du modèle entier —
mais uniquement en Metal, donc sur un matériel que le lecteur du papier ne
possède pas et sur lequel il ne peut rien rejouer.

**La spécification complète est dans
[`docs/portage-noyau-cuda.md`](portage-noyau-cuda.md)** : le noyau Metal
spécifié bit par bit, la route technique, les écarts matériels, le harnais.
C'est le document à lire avant d'écrire une ligne.

### Le plan, révisé après vérification sur la carte

| | Quoi | Temps | Coût |
|---|---|---|---|
| 1 | Écrire le crate `llvq-cuda` : plomberie `cudarc` + compilation à la volée + décodeur | 1–2 j | 0 |
| 2 | Une reconstruction d'image | 1 h non surveillée | 0 |
| 3 | **Mini-job : le noyau compile-t-il, et décode-t-il un bloc juste ?** | minutes | ~0,05 $ |
| 4 | Faire grossir : 252 matrices, vérification du million de lignes, baseline cuBLAS, protocole froid | 1–2 j | centimes par essai |
| 5 | Régler jusqu'à un chiffre défendable | 1–3 j | idem |

**4 à 7 jours.** La spécification annonce 10,5 à 18,5 : ce chiffre est
**périmé**, il supposait qu'on ne pouvait pas itérer. On peut, pour quelques
centimes.

### Les faits déjà vérifiés sur la carte

Job `6a724dfba00abefd4b292856`, trente secondes :

```
NVIDIA L40S, compute_cap 8.9, 46068 MiB, driver 580.159.03
libnvrtc.so.12    → présente dans l'image d'EXÉCUTION
libcublas.so.12   → présente, et nos binaires la lient déjà
nvcc              → absent (sans importance)
```

Conséquence : **une seule reconstruction d'image pour ajouter le binaire, puis
la mise au point du noyau se fait par mini-jobs sans jamais y retoucher** — à
condition que le code du noyau vive **en dehors** du binaire et soit compilé au
démarrage.

### Deux trouvailles de la spécification, obtenues au tableau

**La table de classes ne doit pas être `__constant__`.** Metal la passe en
espace constant ; la traduction naïve ferait pareil. Mais ce banc CUDA est
optimisé pour la **diffusion**, et ici les 32 voies lisent 32 classes
différentes par construction. Passer un pointeur `const __restrict__` ordinaire.

**La tuile en mémoire partagée a un conflit de bancs à 8 voies.** L'adresse lue
par la voie *L* vaut `24·L + j`, et `24·L mod 32` ne prend que quatre valeurs
distinctes sur trente-deux voies. Chacune des 24 lectures est donc sérialisée
huit fois. **Ce défaut existe peut-être déjà en Metal** — le tester là-bas est
gratuit, et s'il se confirme, le corriger améliore le 2,07× déjà publié.

## 4. Ce qui est prêt et ne doit pas être refait

**La campagne de mesure**, décrite dans
[`docs/experience-mesure.md`](experience-mesure.md). Trois bras : notre LLVQ
2 bits, l'**AWQ officiel de Qwen** (`Qwen/Qwen3-4B-AWQ`), et le checkpoint f16.
Cinq mesures : disque, mémoire, vitesse, perplexité, MMLU. Sur une L40S louée
chez Hugging Face.

Un pilote a déjà tourné et **ses résultats sont acquis, à réutiliser plutôt
qu'à remesurer** :

| | |
|---|---|
| Vitesse | **1,0 s par fenêtre de 4096 tokens** contre 10-17,5 sur le Mac |
| Écart Metal ↔ CUDA sur la perplexité | **0,0065 %** (12,2361 contre 12,2369, même empreinte de tokens) |
| Oracle sur CUDA | `max \|Δhidden\| = 0.000e0` |
| Mémoire | pic **23,6 Go**, moyenne **16,6 Go** sur la fenêtre de calcul |

**L'outillage est écrit et vert** : `ops/awq_dequant.py` (26 mutants tués sur
27), `ops/floor.py`, `ops/manifest.py`, la capture VRAM dans `ops/run.py`,
`bin/ppl` qui imprime la NLL par fenêtre, `bin/mmlu` qui imprime une empreinte
de tokens et sait vider un CSV par question.

## 5. Les pièges, payés cher

**Le format sur disque est `LVQ2`, le writer écrit `LVQ3`.** La
rétrocompatibilité porte 1,77 Go d'artefact publié et n'était couverte par
aucun test avant aujourd'hui. Le verrou est
`an_untagged_v2_raw_tensor_still_reads`.

**L'estimateur de coût de `ops/run.py` ne vaut que pour une quantification.**
Il multiplie un nombre de poids par un coût de l'encodeur Leech. Sur un job de
mesure il se trompe d'un facteur ~8. Le plafond utile est le `timeout` du job.

**Trois cartes de la table `FLAVORS` ne peuvent charger aucun noyau** :
`a100-large` (sm_80), `a10g-*` (sm_86), `t4-*` (sm_75). L'image fige
`CUDA_COMPUTE_CAP=89` et le PTX n'est compatible que vers l'avant. `launch` les
refuse désormais avant de facturer. **`a100-large` est le piège** : 80 Go à bon
prix, c'est exactement la carte vers laquelle on tend la main.

**Le flux de métriques de Hugging Face est live, dans les deux sens.** Un job
lancé en `--detach` et regardé après coup n'a aucune métrique — mais s'abonner
**trop tôt**, pendant que le job attend une carte, la perd exactement pareil :
l'endpoint n'a rien à diffuser, le générateur revient, le fil sort sans erreur.
Le moniteur se rattache maintenant en boucle.

**`bin/ppl` construit en f32 par défaut, `bin/mmlu` en f16.** Passer
`LLVQ_DTYPE` explicitement. Et ne pas l'imposer à tout un conteneur : `oracle`
construit en f32 pour se comparer à candle, et meurt si on lui impose f16.

**La moyenne de VRAM sur toute la durée d'un job ne veut rien dire.** 124 des
152 échantillons du pilote valaient moins d'un giga : c'est le téléchargement.
La moyenne doit être calculée sur la fenêtre de calcul, et la fenêtre déclarée.

## 6. Ce qu'il ne faut PAS faire

- **Ne pas relancer la campagne de mesure** sans que l'utilisateur le
  redemande. Elle est prête ; la priorité est le noyau.
- **Ne pas ajouter de bras adverse.** Trois bras : LLVQ, AWQ officiel, f16.
  GPTQ et bitsandbytes ont été proposés et **écartés explicitement**.
- **Ne jamais lancer un job Hugging Face sans autorisation explicite.** Ça
  coûte de l'argent.
- **Ne pas publier le ratio de compression du 8B** ni le 8B en général : hors
  périmètre.
- **Ne pas écrire « 2 bits par poids » tout court.** C'est 2,1595 sur le
  disque, 5,51 en RAM avec le format rapide.
- **Ne pas comparer un MMLU à 40 questions par matière avec un MMLU complet.**

## 7. Commandes utiles

Vérifier que tout tient :
```bash
cargo clippy --all-targets && cargo test --release -- --include-ignored
```

Faire tourner le modèle publié :
```bash
cargo run --release -p llvq-llm --features metal --bin run -- ~/qwen3-4b-llvq.bin metal 24
```

Le banc du noyau, Metal, celui qui produit le 2,07× :
```bash
cargo run --release -p llvq-metal --bin thesis -- ~/qwen3-4b-llvq.bin
```

Contrôler le déquantificateur AWQ sans rien télécharger :
```bash
uv run ops/awq_dequant.py check
```

Lancer un mini-job sur la carte (adapter la commande) :
```bash
uv run ops/run.py monitor <job_id> --flavor l40sx1 --metrics-out serie.json
```

## 8. Les documents, par ordre d'utilité

| Fichier | Ce qu'il contient |
|---|---|
| [`docs/portage-noyau-cuda.md`](portage-noyau-cuda.md) | **La spécification du portage.** À lire en premier |
| [`docs/fiche-4b.md`](fiche-4b.md) | La vérité terrain sur le modèle publié : chaque chiffre avec sa provenance et son statut |
| [`docs/experience-mesure.md`](experience-mesure.md) | La campagne de mesure, prête à lancer |
| [`docs/plan-de-test-v2-cuda.md`](plan-de-test-v2-cuda.md) | Le protocole détaillé dont le précédent est le résumé exécutable |
| [`docs/audit-publication-2026-08-03.md`](audit-publication-2026-08-03.md) | L'audit qui a déclenché les corrections |
| [`CLAUDE.md`](../CLAUDE.md) | Le journal de bord. ⚠️ Carnet de laboratoire, pas spécification : il porte encore des chiffres que les documents ci-dessus rétractent |

## 9. Deux choses non résolues

**Ce que notre harnais mesurera de l'AWQ est sa reconstruction, pas son
arithmétique fusionnée.** Si son noyau accumule autrement, la qualité qu'on lui
attribue n'est pas exactement celle qu'un utilisateur obtient. Ça se déclare,
ça ne se défait pas.

**La variance de re-quantification n'est pas mesurée.** Aucune barre publiée ne
la couvre. La seule dispersion observée (~7 %) porte sur deux configurations
différentes, pas sur deux tirages — donc ce n'est pas un sigma.
