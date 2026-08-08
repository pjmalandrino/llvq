# Brouillon de mail aux auteurs — RELIRE AVANT ENVOI

> Version du 2026-08-08, **écrite sur la structure et le ton de l'utilisateur**
> (courte, directe, première personne). Les versions antérieures sont mortes :
> elles revendiquaient sur l'annexe G l'inverse de ce qu'on sait (92,81 % à
> 1,958 bit/dim, obtenus en divisant la MSE par un débit fractionnaire
> qu'aucun fichier ne paie), donnaient la perplexité d'avant la découverte que
> la rétraction annulait le code de gain, présentaient le noyau comme un
> projet et non comme une mesure, et annonçaient une cible **Apple Metal** que
> le papier déclare inexistante.
>
> Adresses en première page du papier :
> `{touderaa, mart, pwhatmou, markusn}@qti.qualcomm.com`.
>
> **Claude n'envoie rien.** L'envoi est ton geste.
>
> **Pas de pièce jointe, et pas besoin** : `paper/` est dans le dépôt public,
> donc pointer le lien suffit. Un PDF de treize pages non sollicité dans une
> boîte de chercheur se lit « je lirai ça » ; un lien se clique.
>
> **Pas de demande d'endorsement arXiv ici** — mais il en faudra une, et
> ailleurs dans ce fichier on a d'abord écrit une bêtise là-dessus : arXiv
> fournit le *mécanisme* (un code à six caractères et un lien à transmettre),
> **pas l'endosseur**. Le demandeur doit trouver un humain qualifié lui-même.
>
> Et depuis le **21 janvier 2026**, l'adresse institutionnelle ne suffit plus
> comme unique qualificatif : il faut soit une adresse académique **et** un
> papier déjà accepté dans le même domaine d'endorsement, soit l'endorsement
> personnel d'un auteur arXiv établi. Avec une adresse `scub.net` et aucune
> publication antérieure, seule la seconde voie est ouverte
> (https://info.arxiv.org/help/endorsement.html, et le billet de politique du
> 2026-01-21 sur `blog.arxiv.org`).
>
> Ça ne change pas la consigne pour CE mail : demander un jugement d'abord, un
> service ensuite. Deux choses à retenir pour le second message : la charge
> pour l'endosseur est faible (« We do not expect you to read the paper in
> detail, or verify that the work is correct, but you should check that the
> paper is appropriate for the subject area »), et arXiv recommande
> explicitement de **joindre le papier à la demande d'endorsement** — donc la
> règle « pas de pièce jointe » ci-dessus vaut pour le premier contact, pas
> pour la demande elle-même.

---

**Objet :** Rust reimplementation of LLVQ — reproduction, and Appendix C

Hello,

I read your LLVQ paper a few months ago and ended up reimplementing it in
Rust. The repository is public: https://github.com/pjmalandrino/llvq. It
covers the lattice, the exact nearest-neighbour search, the bijective 48-bit
indexing and Spherical GPTQ. The mathematical core has no external
dependencies, so it can be read on its own.

My first goal was to reproduce your numbers. The second one, which took most
of the time, was Appendix C. Your kernel decodes a single shell, and you say
the low-level work is orthogonal to your contribution. I could not find the
multi-shell version anywhere, so I wrote it: a fused dequantize-plus-matvec
over the full 301-class codebook. It runs at 2.14× an FP16 matvec while
reading 4.80 bits per weight, and every output row is checked against an f64
reference. CUDA only for now. The measurements are in the repository, with a
draft paper I would like to publish.

Where I differ from you is the part I would like your opinion on. My
perplexity matches yours. My MMLU does not.

The FP16 baseline is fine: MMLU 70.32 ± 1.28 against your 70.2, WikiText-2
12.2369 against your 12.41. At 2 bits I use norm(Λ₂₄(12)) with one gain bit,
48 bits per block. Perplexity degrades by ×1.384, close to the ×1.374 of your
0-gain-bit line. But MMLU drops to 55.59, below both of your shape-gain
configurations. That is −14.6 points where you report −9.5 without
fine-tuning, on a baseline that agrees with yours.

I tried the obvious explanations and none of them held: output-side rotation,
the free-magnitude variant with the closed-form scale solve, and calibration
volume, which I bounded with a deliberate-contamination oracle. The main
difference I know of is that I calibrate on about 100× fewer tokens. On 8B
the gap narrows to −10.6 points.

Two smaller things. My rate is 2.07 bits per weight of payload instead of
2.000. About 0.1 bit of that is the tail: layer widths are not multiples of
24 and I keep the remainder in full precision. I could not find what you do
there, and I would like to know. Also, I redid the single-shell versus union
comparison from Appendix G on a Gaussian source. At equal packed rate it
agrees with your choice, 92.14 % retention against 90.34 % for shell 12
alone. I had assumed the opposite before I measured it.

I would be glad to hear what you think, about the implementation as much as
about the MMLU gap.

Thank you for the paper. Reading the scale correction as a retraction is the
part that made everything else make sense to me.

Best regards,
Pier-Jean Malandrino

---

## À vérifier avant d'envoyer

- [x] Le dépôt est public — vérifié le 2026-08-08, `github.com/pjmalandrino/llvq`
- [x] `LICENSE-MIT` et `LICENSE-APACHE` présents, cohérents avec le
      `license = "MIT OR Apache-2.0"` des `Cargo.toml`
- [ ] Le README affiche le tableau de reproduction **et** la section
      « what is not here » — c'est cette dernière qui rend le reste crédible
- [ ] Aucun chiffre du mail ne diverge du README ni du papier
- [ ] Relu à voix haute : c'est ton mail, il doit sonner comme toi

### Destinataires

Note de première page de `2603.11021v2` (7 juillet 2026) : « Correspondence to:
`{touderaa, mart, pwhatmou, markusn}@qti.qualcomm.com` ».

| adresse | auteur |
|---|---|
| `touderaa@qti.qualcomm.com` | Tycho F. A. van der Ouderaa — destinataire principal |
| `mart@qti.qualcomm.com` | Mart van Baalen |
| `pwhatmou@qti.qualcomm.com` | Paul Whatmough |
| `markusn@qti.qualcomm.com` | Markus Nagel |

---

# Mail séparé — demande d'endorsement arXiv

> **À n'envoyer qu'après avoir démarré la soumission sur arXiv et récupéré le
> code à six caractères.** Une demande accompagnée du code et du lien est un
> clic ; une demande de principe est une charge mentale.
>
> Cible : quelqu'un que tu connais réellement. arXiv préfère explicitement ça
> (« You should know the person that you endorse »).
>
> **Le domaine d'endorsement pour l'informatique est l'archive `cs` entière**,
> donc un dossier `cs.SE` vaut pour `cs.LG`. Fondé sur la phrase d'arXiv
> « most high-level subject areas are currently endorsement domains, with the
> notable exception of physics » : l'informatique n'y est pas nommée, mais la
> physique est donnée comme la seule exception. Inférence solide, pas
> citation — le lien d'endorsement tranche définitivement et gratuitement.
>
> Second critère, celui-là explicite : les papiers de l'endosseur doivent
> avoir été déposés **entre 3 mois et 5 ans**. Sa page auteur arXiv le dit.
>
> Le registre ci-dessous suppose une relation professionnelle cordiale mais
> pas intime — ajuste l'ouverture si vous êtes plus proches.

**Objet :** Un coup de main pour un endorsement arXiv (cs.LG) ?

Bonjour Romain,

J'espère que tu vas bien. Je me permets de te solliciter pour quelque chose de
très ponctuel.

J'ai passé les derniers mois à réimplémenter en Rust un papier de Qualcomm AI
Research sur la quantification de LLM à 2 bits, et à écrire le noyau GPU que
leur papier laissait explicitement de côté. Le dépôt est public
(https://github.com/pjmalandrino/llvq) et j'aimerais déposer le papier qui en
sort sur arXiv, en cs.LG.

Problème : arXiv demande un endorsement pour un premier dépôt, et depuis leur
changement de politique de janvier 2026 une adresse d'entreprise ne suffit
plus. Il me faut donc l'endorsement d'un auteur arXiv établi du domaine.

Est-ce que ce serait quelque chose que tu pourrais faire, ou que quelqu'un de
ton entourage pourrait faire ? Je précise, parce que la catégorie peut faire
hésiter : chez arXiv le domaine d'endorsement pour l'informatique est
l'archive `cs` dans son ensemble, pas la sous-catégorie. Un dossier en
`cs.SE` compte donc pour `cs.LG`, et le lien confirme de toute façon la
qualification dans un sens ou dans l'autre.

Concrètement c'est un lien à ouvrir avec un code à six caractères. Leur
consigne est explicite : il ne s'agit pas d'une relecture, juste de confirmer
que le papier relève bien de la catégorie.

Mon code est : `XXXXXX`
Le lien : https://arxiv.org/auth/endorse?x=XXXXXX

Le papier est dans le dépôt, dans `paper/`, si tu veux y jeter un œil avant —
et si ce n'est pas quelque chose que tu souhaites faire, aucun souci, dis-le
moi simplement et je chercherai ailleurs.

Merci d'avance, et bonne continuation,
Pier-Jean

---

### Provenance des chiffres cités

| chiffre | source |
|---|---|
| 70,32 ± 1,28 · 55,59 · ×1,384 · 12,2369 | `paper/sections/evaluation.tex`, table du 4B |
| −14,6 points, −10,6 au 8B | idem, et l'abstract du papier |
| leurs 70,2 · 12,41 · 17,05 · 60,7 · 59,3 · −9,5 | `CLAUDE.md` §6, Table 6 du papier relue au rendu image |
| ×1,374 pour leur ligne 0 bit de gain | dérivé : 17,05 / 12,41 |
| 92,14 % contre 90,34 % à 48 bits/bloc | `CLAUDE.md` §6, table révisée le 2026-08-04 |
| 2,14× · 4,80 b/poids · 301 classes | `paper/sections/layouts.tex` et `tab:layouts` |
| 2,07 b/poids de payload | `calib.rs::bits_per_weight`, note de provenance `CLAUDE.md` §G5 |
