# Tetra logo

Four nested chevrons on a 24 × 24 grid, the outer one in terracotta, followed by the name in Geist Mono.

## Construction

- The apex angle is 60°. Every layer stands on the same base line, y = 21, and the outer apex sits at y = 3.
- The stroke is 1.0 unit. Layer k is the outer layer moved inward by d = c(√5 − √m), for m = 5, 4, 3, 2, with c set so that the innermost layer sits 6.4 units inside. The offsets are 0, 1.838, 3.925 and 6.4.
- The √m are the radii of the first four shells of Λ₂₄: shell m holds the points with ‖x‖² = 16m, as in [`how-it-works.md`](../how-it-works.md). That is why the layers spread out toward the centre.
- There are four layers because the Tetra word has four fields, `[state][s₁][s₂][s₃]`. The coloured outer layer stands for the first one.
- The wordmark is `tetra` in Geist Mono Regular, outlined, on the same base line as the mark. The gap between mark and word equals the depth of the layers, 6.4 units.

## Colours

| name | hex | use |
|---|---|---|
| Ink | `#17171B` | inner layers and wordmark on light grounds, dark backgrounds |
| Paper | `#F2F1ED` | inner layers and wordmark on dark grounds, light backgrounds |
| Terracotta | `#B4502D` | the outer layer only |

On a terracotta ground the whole mark is Paper. In one colour the whole mark is Ink or Paper.

## Sizes

- From 48 px up, use the four-layer mark (`svg/tetra-mark*.svg`).
- Below 48 px, use the two-layer mark, stroke 2.4 (`svg/tetra-mark-small.svg`, `svg/favicon.svg`, `favicon.ico`). Four layers blur into one shape at that size.
- Keep a clear space of a quarter of the mark's height on every side.

Do not change the angle, the colours of the inner layers or the spacing, and do not add shadows, gradients or outlines.

## Terminal

The mark writes itself in ASCII: four chevrons of `/` and `\`, the outer one in the accent colour.

```text
   /\
  //\\
 ///\\\    tetra
////\\\\
```

`ascii/banner.txt` is plain text and `ascii/banner.ansi` carries 24-bit colour codes. The `llvq` runtime builds
either with `banner(color: bool)` in [`llvq-runtime/src/banner.rs`](../../llvq-runtime/src/banner.rs), where a test
holds it equal to these two files, and prints it when the server starts and when a chat opens. The colour is left
out under `NO_COLOR`, on a `dumb` terminal and when the output is not a terminal.

## README header

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/svg/tetra-lockup-on-dark.svg">
  <img src="docs/brand/svg/tetra-lockup.svg" alt="tetra" width="320">
</picture>
```

## Files

| file | use |
|---|---|
| `svg/tetra-mark.svg`, `tetra-mark-on-dark.svg` | the mark on light and dark grounds |
| `svg/tetra-mark-mono.svg`, `tetra-mark-mono-white.svg` | one colour |
| `svg/tetra-mark-small.svg` | below 48 px |
| `svg/tetra-lockup.svg`, `tetra-lockup-on-dark.svg` | mark and name |
| `svg/favicon.svg`, `favicon.ico`, `png/favicon-*.png` | browser tab; the SVG follows the system dark mode |
| `png/apple-touch-icon-180.png` | home screen icon |
| `png/avatar-512-dark.png`, `avatar-512-light.png` | GitHub and Hugging Face avatars |
| `png/social-card-1200x630.png` | repository social preview, link cards |
| `ascii/` | terminal banner |

## Font

Geist Mono is © 2023 Vercel, under the SIL Open Font License 1.1. The wordmark is outlined in the SVG files, so the font is not needed to show them.
