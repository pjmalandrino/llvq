# Brouillon de mail aux auteurs — À RELIRE ET RÉÉCRIRE AVANT ENVOI

> Ce n'est qu'un point de départ. Le ton d'un premier contact qui engage
> professionnellement doit être le tien. Adresses en première page du papier :
> `{touderaa, mart, pwhatmou, markusn}@qti.qualcomm.com`.
>
> Trois principes tenus dans le brouillon : **court** (les chercheurs reçoivent
> beaucoup), **les réserves posées d'emblée** (c'est ce qui distingue du bruit),
> et **une question précise** plutôt qu'une annonce (on répond bien plus
> volontiers à une question sur son propre travail).

---

**Objet :** Independent Rust reimplementation of LLVQ — reproduction, and a
question on Appendix G

Dear Dr. van der Ouderaa and colleagues,

I have reimplemented LLVQ from scratch in Rust and would like to submit the
results to your scrutiny. The lattice, the exact nearest-neighbour search, the
bijective indexing and Spherical GPTQ are all there, with the mathematical
core kept dependency-free so it can be audited end to end:
[link]

**Reproduction.** On Qwen3-4B, 2 bits, no fine-tuning, WikiText-2 at 4096
context, calibrating out of domain (C4) as you do: 12.2336 → 14.9104, a ×1.219
degradation, at 2.1117 bits/weight. For reference my FP32 baseline lands 1.4 %
under yours (12.2336 vs 12.41), which I attribute to evaluating on 12 windows
rather than the full test set.

Two caveats I would rather state than have you find: my rate is 5.6 % above
2.000, of which roughly 0.1 bit/weight is the tail policy — layer widths are
not multiples of 24 and I leave the remainder at full precision. I could not
find what your implementation does with that remainder. Conversely I use about
100× fewer calibration tokens than you and only input-side rotation, which
works the other way.

**A question on Appendix G.** You compare single shells against unions on
angular separation and adopt the union. Measuring rate–distortion retention
instead, on an i.i.d. Gaussian source, I find a single shell ahead on both
axes: shell 12 alone with one gain bit gives 92.81 % retention at 1.958
bits/dim, against 92.14 % at 2.000 for `norm(Λ₂₄(12))` + 1 gain bit — with 4.8×
fewer equivalence classes and the constant norm your own appendix notes as the
hardware-friendly property.

I am aware these are different quantities and that mine is one source, one
seed, and not yet verified on real weights after the GPTQ loop. I would value
knowing what I am missing.

**What I would like to work on.** Appendix C notes that your fused kernel
handles a single shell for simplicity and that low-level optimization is
largely orthogonal to your contribution. The multi-shell kernel the 2-bit
regime needs does not appear to exist anywhere, and that is the piece I would
like to build — targeting Apple Metal, where unified memory removes the VRAM
ceiling that makes a 2-bit 70B interesting in the first place. I would be glad
to share results as they come, and equally glad to be told the idea is a dead
end and why.

Thank you for the paper — the geometric reading of scale correction as a
retraction is the part that made the rest click.

Best regards,
Pier-Jean Malandrino

---

## Points à vérifier avant d'envoyer

- [ ] Le dépôt est public et le lien fonctionne
- [ ] `LICENSE` présent (MIT + Apache-2.0, annoncés dans les `Cargo.toml`)
- [ ] Le README affiche le tableau de reproduction **et** la section
      « what is not here » — c'est cette dernière qui rend le reste crédible
- [ ] Aucun chiffre du mail ne diverge du README
- [ ] Relu à voix haute : si une phrase sonne comme une revendication plutôt
      qu'un rapport, la réécrire
