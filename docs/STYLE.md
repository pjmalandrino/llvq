# Writing rules

These rules apply to every living document: `README.md`, `CLAUDE.md`, `docs/*.md`, and the preregs and journals to
come. They do not apply to `docs/mesures/`, to timestamped preregs or to `docs/archive/`, which are frozen.

Documents are in English. Conversation with the operator stays in French.

## The reader

A competent engineer in a hurry, who has not read the rest of the repository and must be able to decide or resume the
work from this one document. Write the way you would explain it to a colleague.

## The rules

1. **Fact first.** A section opens with the result and its number. The reason comes next, the caveat last.
2. **Short sentences.** One idea a sentence, about twenty words. A sentence with two commas and a "which" gets split.
3. **No history in a living document.** When a fact changes, replace the sentence and put one dated line in
   `HISTORIQUE.md`. Never write "this said X, it was wrong".
4. **A number carries its label once.** *measured*, *computed* or *estimated*, with its journal, where it first
   appears. After that, cite it bare, and do not copy it into three documents.
5. **No banners.** No emoji, no "Caveat:", no check marks. In a table, write "passes" or "fails".
6. **No AI phrasing.** The list below is grep-able. A document that contains one gets fixed before it is committed.

## Banned

```
—  (em dash: use two sentences or a comma)
in other words · that is to say · simply put
it is worth noting · it's important to note · note that · recall that
that said · that being said · at the end of the day · ultimately
delve · seamless · leverage (verb) · robust · comprehensive
crucially · notably · importantly · significantly (as filler)
here's the thing · the key insight is · the takeaway is
```

Also banned: the "not X, but Y" pattern, reflex lists of three, rhetorical questions, sentences that comment on the
previous one, bold on a whole sentence, slogan headings.

## Example

Before:

> 🆕 **THE NEW FACT THE RANGES REVEAL**: **at identical head, the only formulation that measures the kernel, the gain
> is STRICTLY INCREASING with size, ×1.11 → ×1.29 → ×1.41**, where the raw series has no order at all.

After:

> At identical head, the kernel gain grows with size: ×1.11, ×1.29, ×1.41 from 4B to 14B (*measured*,
> [B2](mesures/b2-fusedrun-plages-2026-08-18.txt)). The raw series has no order, because the dense arm is handicapped
> differently at each size.

## Target lengths

| document | lines |
|---|---|
| `README.md` | 150 |
| `CLAUDE.md` | 180, most of it commands and variables |
| `docs/ETAT.md` | 150 |
| `docs/ROADMAP.md` | 150 |
| `docs/METHODE.md` | 150 |
| `docs/CERVEAU.md` | no line target; one to two sentences a claim, and its five fields |
| `docs/MODES.md` | a short document |
| `docs/how-it-works.md` | 300 |
| experiment summary | 40, template `templates/experience.md` |
| prereg | 80, template `templates/prereg.md` |

A document that passes its target by half gets cut or split.

## What does not change

The rigour. Every number keeps its label, every prereg is stamped before the first measurement, every deviation is
written beside the prereg, and every signed prediction stays on the record. `METHODE.md` holds those rules.
