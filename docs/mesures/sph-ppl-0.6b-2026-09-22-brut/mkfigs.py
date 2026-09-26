#!/usr/bin/env python3
"""Figures for the spherical-feedback refutation of 2026-09-22.

Every number drawn here comes from sphgeom.txt (Gaussian bench, same
directory) or from the journal beside it. Nothing is illustrative except the
angle theta, which is drawn wider than its measured 16.1 degrees so the
triangle stays legible, and the figure says so.

Colours are written as concrete attributes so any renderer shows the light
palette; a <style> block adds the dark steps for browsers that ask for them.
Renderers without CSS support simply keep the light values.

    python3 mkfigs.py          # writes fig1..fig4 .svg here
"""
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
W = 980

# Validated categorical slots 1-3 plus surfaces and ink, light and dark.
# scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --pairs all : all pass.
LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd6",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a", on1="#ffffff")
DARK = dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#44443f",
            s1="#3987e5", s2="#d95926", s3="#199e70", on1="#ffffff")

FONT = 'ui-sans-serif, system-ui, "Segoe UI", Helvetica, Arial, sans-serif'


def style_block():
    out = ["@media (prefers-color-scheme: dark){"]
    for k, v in DARK.items():
        out.append(f".f-{k}{{fill:{v}}}")
        out.append(f".s-{k}{{stroke:{v}}}")
    out.append("}")
    out.append(".bar:hover{opacity:.75}")
    return "<style>" + "".join(out) + "</style>"


def svg(h, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" '
        f'width="{W}" height="{h}" role="img" font-family=\'{FONT}\'>'
        f"{style_block()}"
        f'<rect class="f-surface" width="{W}" height="{h}" fill="{LIGHT["surface"]}"/>'
        f"{body}</svg>"
    )


def txt(x, y, s, size=14.5, role="ink", anchor="start", weight=None, extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" class="f-{role}" fill="{LIGHT[role]}"{w}{extra}>'
        f"{s}</text>"
    )


def line(x1, y1, x2, y2, role="grid", w=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'class="s-{role}" stroke="{LIGHT[role]}" stroke-width="{w}" '
        f'stroke-linecap="round"{d}/>'
    )


def circle(cx, cy, r, fill=None, stroke=None, sw=1.6):
    cls = ([f"f-{fill}"] if fill else []) + ([f"s-{stroke}"] if stroke else [])
    a = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" class="{" ".join(cls)}"'
    a += f' fill="{LIGHT[fill]}"' if fill else ' fill="none"'
    if stroke:
        a += f' stroke="{LIGHT[stroke]}" stroke-width="{sw}"'
    return a + "/>"


def path(d, fill=None, stroke=None, sw=2, extra=""):
    cls = ([f"f-{fill}"] if fill else []) + ([f"s-{stroke}"] if stroke else [])
    a = f'<path d="{d}" class="{" ".join(cls)}"'
    a += f' fill="{LIGHT[fill]}"' if fill else ' fill="none"'
    if stroke:
        a += f' stroke="{LIGHT[stroke]}" stroke-width="{sw}" stroke-linecap="round"'
    return a + extra + "/>"


def rect(x, y, w, h, rx=0, fill=None, stroke=None, sw=1.6, dash=None, op=None,
         title=None, cls=""):
    c = ([cls] if cls else []) + ([f"f-{fill}"] if fill else []) + \
        ([f"s-{stroke}"] if stroke else [])
    a = f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"'
    a += f' class="{" ".join(c)}"'
    a += f' fill="{LIGHT[fill]}"' if fill else ' fill="none"'
    if stroke:
        a += f' stroke="{LIGHT[stroke]}" stroke-width="{sw}"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
    if op is not None:
        a += f' opacity="{op}"'
    if title:
        return a + f"><title>{title}</title></rect>"
    return a + "/>"


def head(title, sub):
    return txt(40, 40, title, 21, "ink", weight=650) + txt(40, 66, sub, 14.5, "ink2")


# --------------------------------------------------------------------------
# measured constants (sphgeom.txt, seed 0xf1b20260922)
# --------------------------------------------------------------------------
COS = 0.96080
G0, G1 = 0.875003, 1.106938
SD, MEAN = 0.09200, 1.00684


def read_hist(label):
    rows, on = [], False
    for ln in open(os.path.join(HERE, "sphgeom.txt")):
        if ln.startswith("HIST "):
            on = ln.split()[1] == label
            continue
        if on and re.match(r"\s+[\d.]+\s+\d+\s*$", ln):
            a, b = ln.split()
            rows.append((float(a), int(b)))
        elif on and not ln.startswith(" "):
            on = False
    return rows


# ==========================================================================
# FIG 1
# ==========================================================================
def fig1():
    H = 752
    b = [head(
        "Où chaque design pose le point, et contre quoi la boucle compense",
        "Un bloc de 24 poids. Toute la décision tient sur la demi-droite portée par la direction codée u.")]

    # ---- inset -----------------------------------------------------------
    ox, oy, L = 95, 235, 235
    ang = math.radians(30)
    xx, xy = ox + L * math.cos(ang), oy - L * math.sin(ang)
    b += [
        line(ox, oy, ox + 300, oy, "ink2", 1.8),
        path(f"M {ox+300} {oy} l -10 -5 v 10 z", fill="ink2"),
        txt(ox + 310, oy + 5, "u", 15, "ink2", weight=600),
        line(ox, oy, xx, xy, "ink", 2.2),
        circle(xx, xy, 5, fill="ink"),
        txt(xx + 12, xy - 4, "x, le bloc", 14, "ink", weight=600),
        line(xx, xy, xx, oy, "s2", 1.8, dash="4 4"),
        txt(xx + 12, (xy + oy) / 2 - 4, "erreur angulaire", 13, "s2"),
        txt(xx + 12, (xy + oy) / 2 + 13, "0,277 · ‖x‖", 13, "s2"),
    ]
    r = 44
    b += [
        path(f"M {ox+r} {oy} A {r} {r} 0 0 0 "
             f"{ox + r*math.cos(ang):.1f} {oy - r*math.sin(ang):.1f}",
             stroke="ink2", sw=1.4),
        txt(ox + r + 9, oy - 11, "θ", 14, "ink2"),
        line(ox, oy - 6, ox, oy + 6, "ink2", 1.8),
        txt(ox, oy + 25, "0", 14, "ink", "middle"),
    ]
    # t and ‖x‖ are 3,9 % apart: stagger the labels with leaders
    pt, pn = ox + L * COS * math.cos(ang), ox + L * math.cos(ang)
    b += [
        line(pt, oy - 6, pt, oy + 6, "ink2", 1.8),
        line(pn, oy - 6, pn, oy + 6, "ink2", 1.8),
        line(pt, oy + 8, pt - 16, oy + 22, "grid", 1.3),
        txt(pt - 20, oy + 27, "t", 14, "ink", "end"),
        line(pn, oy + 8, pn + 16, oy + 40, "grid", 1.3),
        txt(pn + 20, oy + 45, "‖x‖", 14, "ink"),
    ]
    b += [
        txt(ox, oy + 78, "t = ‖x‖ · cos θ, l’optimum sur la droite.", 13, "ink2"),
        txt(ox, oy + 97, "θ est dessiné large pour la lisibilité ; mesuré à 16,1°.", 13, "ink2"),
    ]

    # ---- right column -----------------------------------------------------
    tx = 470
    b += [
        txt(tx, 118, "Le fichier stocke ĝ · u : une direction codée et une", 15),
        txt(tx, 140, "magnitude ĝ. Deux quantités peuvent diverger.", 15),
        circle(tx + 9, 175, 8.5, fill="s1"),
        txt(tx + 28, 180, "ce que le fichier stocke", 15, "ink", weight=600),
        path(f"M {tx+9} 205 l -7 -11 h 14 z", fill="s2"),
        txt(tx + 28, 208, "ce contre quoi la boucle compense", 15, "ink", weight=600),
        txt(tx, 242, "Si les deux coïncident, le design est cohérent.", 15, "ink2"),
        txt(tx, 264, "Sinon la boucle corrige un fichier qui n’existe pas.", 15, "ink2"),
    ]

    # ---- three rulers -----------------------------------------------------
    LO, HI = 0.80, 1.25
    x0, x1 = 300, 860

    def px(v):
        return x0 + (v - LO) / (HI - LO) * (x1 - x0)

    rows = [
        ("A. publié (nogs)", "48 bits/bloc", 1.107, 1.107, "cohérent", False),
        ("B. magnitude libre", "47 + 16 bits/bloc", 1.000, 1.000, "cohérent", False),
        ("sph, ce qui a été mesuré", "48 bits/bloc", 1.107, 1.000, "incohérent", True),
    ]
    b += [
        txt(40, 372, "Les trois designs, sur la même règle", 15, "ink", weight=650),
        txt(40, 393, "position en unités de ‖x‖, la norme du bloc", 13, "ink2"),
    ]
    y = 478
    for i, (name, rate, stored, fed, verdict, gap) in enumerate(rows):
        yy = y + i * 88
        b += [
            line(x0, yy, x1, yy, "grid", 2),
            txt(40, yy - 20, name, 15, "ink", weight=600),
            txt(40, yy + 2, rate, 13, "ink2"),
            txt(40, yy + 23, verdict, 13.5, "s2" if gap else "ink2",
                weight=600 if gap else None),
        ]
        for v in (G0, G1):
            b.append(circle(px(v), yy, 5.5, stroke="ink2"))
        for v in (COS, 1.0):
            b.append(line(px(v), yy - 10, px(v), yy + 10, "grid", 1.6))
        if i == 0:
            b += [
                txt(px(COS) - 8, yy + 30, "t", 13, "ink2", "end"),
                txt(px(1.0) + 8, yy + 30, "‖x‖", 13, "ink2"),
                txt(px(G0), yy - 40, "niveau 0", 13, "ink2", "middle"),
                txt(px(G1), yy - 40, "niveau 1", 13, "ink2", "middle"),
                txt((px(G0) + px(G1)) / 2, yy - 62,
                    "les deux magnitudes que le mot de 48 bits sait dire", 13, "ink2", "middle"),
            ]
        if gap:
            a, c = sorted((px(stored), px(fed)))
            b.append(rect(a, yy - 14, c - a, 28, 3, fill="s2", op=0.18))
            b.append(txt((a + c) / 2, yy + 50, "l’écart : (‖x‖ − ĝ) · u", 13.5,
                         "s2", "middle", weight=600))
        b.append(circle(px(stored), yy, 8.5, fill="s1", stroke="surface", sw=2))
        b.append(path(f"M {px(fed):.1f} {yy-15} l -7 -11 h 14 z", fill="s2"))

    b.append(txt(40, 728,
                 "Le design B est la ligne 5 de l’Algorithme 3 appliquée à la lettre : elle écrit dans les poids stockés, et coûte 16 bits par bloc.",
                 13.5, "ink2"))
    return svg(H, "".join(b))


# ==========================================================================
# FIG 2
# ==========================================================================
def fig2():
    H = 648
    hist = read_hist("tetra")
    total = 20000
    b = [head(
        "Ce que le fichier stocke comme magnitude, rapporté à la norme du bloc",
        "20 000 blocs gaussiens, encodeur de production Tetra, un bit de gain. Banc sphgeom, graine 0xf1b20260922.")]
    x0, x1, y0, y1 = 120, 910, 150, 420
    LO, HI = 0.5, 1.5
    pmax = max(c for _, c in hist) / total * 100

    def px(v):
        return x0 + (v - LO) / (HI - LO) * (x1 - x0)

    def pyp(p):
        return y1 - p / pmax * (y1 - y0)

    for p in (2, 4, 6, 8, 10):
        b.append(line(x0, pyp(p), x1, pyp(p), "grid", 1))
        b.append(txt(x0 - 12, pyp(p) + 5, f"{p} %", 13, "ink2", "end"))
    bw = (x1 - x0) / len(hist)
    for c0, c in hist:
        if c == 0:
            continue
        pc = c / total * 100
        b.append(rect(px(c0) - bw / 2 + 1, pyp(pc), bw - 2, y1 - pyp(pc), 3,
                      fill="s1", cls="bar",
                      title=f"{c0:.3f} : {c} blocs, {pc:.2f} %"))
    b.append(line(x0, y1, x1, y1, "ink2", 1.6))
    for v in (0.6, 0.8, 1.0, 1.2, 1.4):
        b.append(line(px(v), y1, px(v), y1 + 6, "ink2", 1.6))
        b.append(txt(px(v), y1 + 26, f"{v:.1f}".replace(".", ","), 14, "ink", "middle"))
    b.append(txt((x0 + x1) / 2, y1 + 52, "ĝ / ‖x‖", 16, "ink", "middle", weight=600))

    b += [
        line(px(1.0), y0 - 40, px(1.0), y1, "s3", 3),
        txt(px(1.0) + 12, y0 - 44, "magnitude exacte : les 20 000 blocs sont ici",
            15, "s3", weight=650),
        txt(px(1.0) + 12, y0 - 24,
            "le cas à 0 bit de gain, et ce que la rétraction du papier fabrique", 13, "ink2"),
    ]
    ya = y1 + 88
    b += [
        line(px(MEAN - SD), ya, px(MEAN + SD), ya, "ink2", 2),
        line(px(MEAN - SD), ya - 6, px(MEAN - SD), ya + 6, "ink2", 2),
        line(px(MEAN + SD), ya - 6, px(MEAN + SD), ya + 6, "ink2", 2),
        txt(px(MEAN), ya + 21, "un écart-type de part et d’autre : 0,092", 13, "ink2", "middle"),
    ]
    b += [
        txt(40, 578, "moyenne 1,007  ·  écart-type 0,092  ·  p1 0,824  ·  p50 1,000  ·  p99 1,297",
            15, "ink", weight=600),
        txt(40, 604,
            "L’énergie de cet écart vaut 8,7 % de l’erreur que le fichier porte, et c’est exactement ce que sph cache à la boucle.", 13.5, "ink2"),
        txt(40, 626,
            "À 0 bit de gain il vaut zéro pour tout bloc, par construction : le test norm_preserving_direction_code_makes_the_flag_inert le mesure à 1e-12.", 13.5, "ink2"),
    ]
    return svg(H, "".join(b))


# ==========================================================================
# FIG 3
# ==========================================================================
def fig3():
    H = 510
    b = [head(
        "Pourquoi un écart par bloc devient +31,9 % de perplexité",
        "GPTQ pousse l’erreur de chaque bloc sur les colonnes à sa droite. Ce qui n’est pas remis n’est jamais réparé.")]
    nb, bx, bw, gp = 11, 250, 52, 9

    def strip(yy, hide):
        out = []
        for i in range(nb):
            x = bx + i * (bw + gp)
            out.append(rect(x, yy, bw, 38, 5,
                            fill="s1" if i == 0 else None, stroke="grid", sw=1.6))
        ax0, ax1 = bx + bw + 6, bx + nb * (bw + gp) - gp
        role = "s2" if hide else "s1"
        out.append(line(ax0, yy - 22, ax1 - 11, yy - 22, role, 3))
        out.append(path(f"M {ax1} {yy-22} l -12 -6 v 12 z", fill=role))
        return out

    y1, y2 = 172, 348
    b += strip(y1, False)
    b += [
        txt(40, y1 + 12, "nogs, le chemin publié", 15, "ink", weight=600),
        txt(40, y1 + 34, "48 bits/bloc", 13, "ink2"),
        txt(bx + 64, y1 - 32, "e₀, l’erreur vraie du bloc 0", 13.5, "s1", weight=600),
        txt(bx, y1 + 66,
            "L’aval absorbe exactement ce que le bloc 0 a raté. Il ne reste rien.", 13.5, "ink2"),
    ]
    b += strip(y2, True)
    b += [
        txt(40, y2 + 12, "sph", 15, "ink", weight=600),
        txt(40, y2 + 34, "48 bits/bloc", 13, "ink2"),
        txt(bx + 64, y2 - 32, "e₀ privée de sa tranche radiale", 13.5, "s2", weight=600),
        txt(bx, y2 + 66,
            "La tranche (‖x‖ − ĝ) · u n’est remise à personne. 85 blocs par ligne, 196 matrices, 28 couches.", 13.5, "ink2"),
    ]
    # the slice that falls out, branching off the arrow itself
    dx = bx + 450          # right of the arrow label, so nothing is crossed
    b += [
        circle(dx, y2 - 22, 4, fill="s2"),
        line(dx, y2 - 22, dx + 34, y2 - 64, "s2", 2, dash="4 4"),
        path(f"M {dx+26} {y2-82} l 16 16 M {dx+42} {y2-82} l -16 16", stroke="s2", sw=2.6),
        txt(dx + 52, y2 - 68, "perdue, définitivement", 13.5, "s2", weight=600),
    ]
    b.append(txt(40, 470,
                 "Mesure de bout en bout, Qwen3-0.6B, 28 blocs, une seule variable de configuration : 41,8875 contre 55,2478 de perplexité.",
                 13.5, "ink2"))
    return svg(H, "".join(b))


# ==========================================================================
# FIG 4
# ==========================================================================
def fig4():
    H = 540
    b = [head(
        "La Table 9 du papier ne remplit qu’une diagonale",
        "Le +1,9 pp que nos documents lui font dire paie deux changements à la fois, pas la rétraction seule.")]
    cx, cy, cw, ch = 360, 185, 260, 100
    cols = ["recherche euclidienne", "recherche angulaire"]
    rows = ["correction : GPTQ", "correction : Spherical GPTQ"]
    cells = [[("34,1 MMLU", True), ("jamais mesuré", False)],
             [("jamais mesuré", False), ("36,0 MMLU", True)]]
    for j, c in enumerate(cols):
        b.append(txt(cx + j * (cw + 14) + cw / 2, cy - 18, c, 15, "ink", "middle", weight=600))
    for i, r in enumerate(rows):
        y = cy + i * (ch + 14)
        b.append(txt(cx - 20, y + ch / 2 + 5, r, 15, "ink", "end", weight=600))
        for j in range(2):
            x = cx + j * (cw + 14)
            filled = cells[i][j][1]
            b.append(rect(x, y, cw, ch, 8,
                          fill="s1" if filled else None,
                          stroke=None if filled else "grid", sw=2,
                          dash=None if filled else "6 5"))
            b.append(txt(x + cw / 2, y + ch / 2 + 6, cells[i][j][0], 16,
                         "on1" if filled else "ink2", "middle", weight=650))
    b += [
        txt(40, 432,
            "Les deux cases bleues sont la paire que le papier compare. Elles diffèrent par la ligne ET par la colonne.", 14, "s2", weight=600),
        txt(40, 458,
            "Les six paires de la table bougent la métrique de recherche et la correction ensemble : elle ne peut attribuer son écart ni à l’une ni à l’autre.", 13.5, "ink2"),
        txt(40, 482,
            "Notre bras sph n’a changé que la ligne. L’encodeur Tetra est déjà l’angulaire : cos θ = 0,961, mesuré.", 13.5, "ink2"),
        txt(40, 506,
            "Deux endroits du dépôt reposent sur cette lecture : ROADMAP-QUALITY ligne 3, et la ligne Spherical GPTQ +1,90 du journal des row scales.", 13.5, "ink2"),
    ]
    return svg(H, "".join(b))


for name, fn in (("fig1-demi-droite", fig1), ("fig2-magnitude", fig2),
                 ("fig3-accumulation", fig3), ("fig4-table9", fig4)):
    p = os.path.join(HERE, name + ".svg")
    open(p, "w").write(fn())
    print("wrote", os.path.basename(p), os.path.getsize(p), "B")
