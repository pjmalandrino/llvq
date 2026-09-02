# <Nom court de l'expérience> (<AAAA-MM-JJ>)

**Question.** <Une phrase.>

**Réponse.** <Une phrase, avec le chiffre et son intervalle.>

## Dispositif

- Objet : <le fichier, le modèle, la taille>
- Variable unique : <ce qui change entre les bras>
- Témoins : <ce qui ne change pas, et où ils ont déjà été mesurés>
- Coût : <mesuré, en $ ou en heures de machine> ; durée <mesurée>

## Résultat

| bras | valeur | intervalle | source |
|---|---|---|---|
| témoin | | | |
| <bras 1> | | | |

## Contrôles

| contrôle | attendu | obtenu | verdict |
|---|---|---|---|
| | | | passe / échoue |

Si un contrôle échoue, rien ne se publie et la ligne « Réponse » le dit.

## Ce que ça n'établit pas

- <une réserve par ligne, la plus lourde en premier>

## Décision

<Ce que le résultat ouvre ou ferme, par la règle posée d'avance. Si la règle
ne couvre pas le cas, le dire, et nommer qui décide.>

## Provenance

- préreg : `proofs/<fichier>.md`, sha256 `<8 premiers>`, tamponné le <date>
- écarts : `proofs/<fichier>-ECARTS.md` (ou « aucun »)
- code : commit `<7 car.>`
- job : `<id>`, <carte>, <minutes>, <$>
- brut : `docs/mesures/<fichier>.txt` et `docs/mesures/<fichier>-brut/`
- données : `docs/data/<...>`
