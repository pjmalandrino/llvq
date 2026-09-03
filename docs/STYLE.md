# Writing rules

These rules govern every living document in the repository: `README.md`, `CLAUDE.md`,
`docs/*.md`, preregs to come, journals to come. They do not apply retroactively to the
journals in `docs/mesures/`, to timestamped preregs, or to `docs/archive/`, which are frozen.

Documents are written in English. Conversation with the operator stays in French.

## The reader

A competent human in a hurry, who has not read the rest of the repository, and who must be
able to decide or resume the work from this document alone. Write for them. Not for
posterity, not for an agent, not to cover yourself.

## The six rules

1. **Fact first.** The first sentence of a section gives the result, with its number. The
   reason comes second. The caveat comes last.
2. **One sentence, one idea, twenty words.** A sentence with two commas and a "which" gets
   cut in two.
3. **A living document does not narrate its own history.** When a fact changes, replace the
   sentence. The old fact goes to `HISTORIQUE.md` with its date. Never write "this line said
   X, it was wrong" in a living document. That is what grew `CLAUDE.md` to 2,971 lines.
4. **A number carries its label once.** *measured*, *computed* or *estimated*, with a link to
   its journal, at first appearance. After that, cite it bare. Do not copy it into the prose
   of three separate documents.
5. **No banners.** No emoji, in prose or in tables. "Wrong since 08-21:" and "Caveat:" say the
   same thing in words. In a table, "passes" and "fails" replace check marks.
6. **No AI phrasing.** List below. A document that contains one gets fixed before it is
   committed.

## Banned phrasing

Grep-able, therefore checkable. The em dash is banned everywhere: use two sentences or a comma.

```
—  (em dash)
in other words · that is to say · simply put · put simply
it is worth noting · it's important to note · note that · let us note · recall that
that said · that being said · at the end of the day · ultimately (as filler)
delve · seamless · leverage (as a verb) · robust (as filler) · comprehensive (as filler)
what is settled · what remains open · what survives · what still holds
crucially · notably · importantly · significantly (as filler)
here's the thing · the key insight is · the takeaway is
two readings · three things · in three sentences · in a word
```

Also banned: reflex three-item lists, the "not X, but Y" parallel, rhetorical questions,
sentences that comment on the previous sentence, bold on a whole sentence, slogan headings.

## Before and after, on real repository text

Before (HISTORIQUE, 2026-08-18):

> 🆕 **THE NEW FACT THE RANGES REVEAL, and this is the B2 result**: **at identical head, the
> only formulation that measures the kernel, the gain is STRICTLY INCREASING with size,
> ×1.11 → ×1.29 → ×1.41**, where the raw series (×2.00 · ×2.57 · ×2.55) **has no order at
> all**, dominated by the varying handicap of the dense arm.

After:

> At identical head, the kernel gain grows with size: ×1.11, ×1.29, ×1.41 from 4B to 14B
> (*measured*, [B2](mesures/b2-fusedrun-plages-2026-08-18.txt)). The raw series has no order,
> because the dense arm is handicapped differently at each size.

Before (CLAUDE.md, header):

> 🕳️ **REVERSAL 1: "all format work is capped at 4.77× FP16" IS MEASURED FALSE, and it is
> this header that carried it.**

After, in `ETAT.md`:

> The `nullk` floor (4.77×) belongs to our launch geometry, not to the card: QTIP goes below
> it (2.246 ms against 2.306).

And in `HISTORIQUE.md`, at its date:

> 2026-08-21. The 4.77× ceiling announced since 08-16 is refuted by F2.

## Target lengths

| document | lines | content |
|---|---|---|
| `README.md` | 120 | what it is, the numbers, how to run it |
| `CLAUDE.md` | 120 | rules for the agent, repository map |
| `docs/ETAT.md` | 100 | served config, headline numbers, open decisions |
| `docs/HISTORIQUE.md` | 400 | one entry per period, ten lines each |
| `docs/ROADMAP.md` | 150 | what comes next, with gates and costs |
| `docs/METHODE.md` | 150 | the lab rules |
| experiment summary | 40 | template `templates/experience.md` |
| prereg | 80 | template `templates/prereg.md` |

A document that exceeds its target by half gets split or cut. It does not grow.

## What does not change

The rigour. Every number keeps its label, every prereg stays timestamped before the first
measurement, every deviation from a prereg is written beside it and never inside it, every
signed prediction stays on the record. Those rules live in `METHODE.md`. The form becomes
direct, the substance does not get lighter.
