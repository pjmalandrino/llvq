#!/usr/bin/env python3
"""Build cours-spherical-gptq.html: one self-contained page, figures inlined.

    python3 mkfigs.py && python3 mkcours.py

Every figure comes from mkfigs.py, every number from sphgeom.txt or from the
journal one level up. The page has no external dependency.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = ["fig1-demi-droite", "fig2-magnitude", "fig3-accumulation", "fig4-table9"]


def inline(name):
    """Read a figure and make it responsive: keep viewBox, drop width/height."""
    s = open(os.path.join(HERE, name + ".svg")).read()
    s = re.sub(r'\swidth="\d+"\sheight="\d+"', ' class="fig"', s, count=1)
    return s


PAGE = r"""<!doctype html>
<html lang="fr">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pourquoi Spherical GPTQ ne marche pas sur Tetra</title>
<style>
  :root {
    color-scheme: light dark;
    --bg:#fcfcfb; --card:#f4f3ef; --ink:#0b0b0b; --ink2:#52514e; --rule:#e2e0da;
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --code:#eeece6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:#1a1a19; --card:#232322; --ink:#ffffff; --ink2:#c3c2b7; --rule:#3a3a37;
      --s1:#3987e5; --s2:#d95926; --s3:#199e70; --code:#2a2a28;
    }
  }
  * { box-sizing:border-box }
  body {
    margin:0; background:var(--bg); color:var(--ink);
    font:17px/1.62 ui-sans-serif, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
    -webkit-text-size-adjust:100%;
  }
  .wrap { max-width:900px; margin:0 auto; padding:56px 24px 120px }
  header { border-bottom:2px solid var(--rule); padding-bottom:28px; margin-bottom:8px }
  h1 { font-size:34px; line-height:1.2; margin:0 0 10px; letter-spacing:-.01em }
  .sub { color:var(--ink2); font-size:18px; margin:0 }
  .meta { color:var(--ink2); font-size:14px; margin-top:18px }
  h2 {
    font-size:23px; margin:56px 0 4px; letter-spacing:-.005em;
    padding-top:22px; border-top:1px solid var(--rule);
  }
  h2 .n { color:var(--s1); font-variant-numeric:tabular-nums; margin-right:10px }
  h3 { font-size:18px; margin:32px 0 6px }
  p { margin:14px 0 }
  a { color:var(--s1) }
  .lede { font-size:19px; color:var(--ink2) }
  figure { margin:30px 0 8px }
  figure svg.fig { width:100%; height:auto; display:block;
                   border:1px solid var(--rule); border-radius:10px }
  figcaption { color:var(--ink2); font-size:14.5px; margin-top:10px }
  .key {
    background:var(--card); border-left:4px solid var(--s1);
    border-radius:0 10px 10px 0; padding:20px 24px; margin:28px 0;
  }
  .key h3, .warn h3 { margin-top:0 }
  .warn { background:var(--card); border-left:4px solid var(--s2);
          border-radius:0 10px 10px 0; padding:20px 24px; margin:28px 0 }
  table { border-collapse:collapse; width:100%; margin:22px 0; font-size:15.5px }
  th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--rule) }
  th { font-weight:650; color:var(--ink2); font-size:14px;
       text-transform:uppercase; letter-spacing:.04em }
  td.num, th.num { font-variant-numeric:tabular-nums; text-align:right;
                   white-space:nowrap }
  figure, .demo, .key, .warn, table { break-inside:avoid }
  tr.hi td { background:rgba(235,104,52,.10) }
  code, pre { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace }
  code { background:var(--code); padding:2px 6px; border-radius:5px; font-size:.88em }
  pre { background:var(--code); padding:16px 18px; border-radius:10px;
        overflow-x:auto; font-size:14px; line-height:1.55 }
  pre code { background:none; padding:0 }
  ul, ol { padding-left:22px } li { margin:9px 0 }
  .toc { background:var(--card); border-radius:10px; padding:18px 24px; margin:32px 0 }
  .toc ol { margin:6px 0; padding-left:20px } .toc li { margin:5px 0 }
  .tag { display:inline-block; font-size:12px; font-weight:650; letter-spacing:.05em;
         text-transform:uppercase; padding:3px 9px; border-radius:20px;
         background:var(--card); color:var(--ink2); vertical-align:2px }
  .tag.m { color:var(--s3); box-shadow:inset 0 0 0 1.5px currentColor }
  /* ---- interactive ruler ---- */
  .demo { background:var(--card); border-radius:12px; padding:24px; margin:30px 0 }
  .demo h3 { margin:0 0 4px }
  .demo .hint { color:var(--ink2); font-size:14.5px; margin:0 0 18px }
  .demo svg { width:100%; height:auto; display:block }
  .ctl { display:flex; align-items:center; gap:16px; margin-top:8px; flex-wrap:wrap }
  .ctl label { font-size:15px; color:var(--ink2) }
  .ctl input[type=range] { flex:1; min-width:220px; accent-color:var(--s1) }
  .read { display:flex; gap:26px; flex-wrap:wrap; margin-top:16px;
          font-variant-numeric:tabular-nums }
  .read div { font-size:14px; color:var(--ink2) }
  .read b { display:block; font-size:21px; color:var(--ink); font-weight:650 }
  .read b.gap { color:var(--s2) }
  footer { margin-top:70px; padding-top:26px; border-top:1px solid var(--rule);
           color:var(--ink2); font-size:14.5px }
</style>

<div class="wrap">
<header>
  <h1>Pourquoi Spherical GPTQ ne marche pas sur Tetra</h1>
  <p class="sub">Un levier du papier LLVQ qui vaut +1,9 point chez Qualcomm, et
     −31,9 % de perplexité chez nous. Ce cours explique pourquoi, et ce que
     l&rsquo;écart apprend sur les deux codebooks.</p>
  <p class="meta">LLVQ · 22 septembre 2026 · toutes les mesures sont dans
     <code>docs/mesures/sph-ppl-0.6b-2026-09-22.txt</code></p>
</header>

<div class="key">
  <h3>La réponse en trois lignes</h3>
  <p style="margin-bottom:0">La rétraction du papier <b>écrit dans les poids
  stockés</b> : elle restaure la norme exacte du bloc, et la rétroaction devient
  honnête <i>en conséquence</i>. Ça se paie en bits. Nous avons voulu prendre la
  rétroaction sans payer la magnitude, et il n&rsquo;y a rien à prendre : un
  résidu n&rsquo;est honnête que s&rsquo;il décrit ce que le fichier contient.</p>
</div>

<div class="toc">
  <b>Au programme</b>
  <ol>
    <li>Le décor : un bloc, une direction, une magnitude</li>
    <li>La boucle GPTQ en une phrase</li>
    <li>Tout tient sur une demi-droite <span class="tag">interactif</span></li>
    <li>Les trois designs, et pourquoi il y en a trois</li>
    <li>L&rsquo;écart, mesuré</li>
    <li>Comment 8,7 % par bloc devient +31,9 %</li>
    <li>Le résultat, et ses contrôles</li>
    <li>« Mais le papier annonce +1,9 point »</li>
    <li>Ce qu&rsquo;il faut retenir</li>
    <li>Refaire les mesures soi-même</li>
  </ol>
</div>

<h2><span class="n">1</span>Le décor : un bloc, une direction, une magnitude</h2>
<p class="lede">LLVQ ne quantifie pas les poids un par un. Il les prend par
paquets de 24 et code chaque paquet sur un mot de 48 bits.</p>
<p>Dans le format <b>Tetra</b>, ce mot se découpe en 47 bits de direction et
1 bit de gain. Le décodeur rebâtit le bloc en trois opérations :</p>
<pre><code>ŵ  =  ĝ · u        u  = la direction codée, de norme 1
                   ĝ  = l'un des 2 niveaux de gain de la ligne</code></pre>
<p>La direction est très bonne et très régulière. Sur 20 000 blocs gaussiens,
le cosinus entre un bloc et sa direction Tetra vaut 0,9608 avec un écart-type de
0,0076 <span class="tag m">mesuré</span>. Autrement dit l&rsquo;erreur angulaire
est presque la même pour tout le monde. <b>Tout ce qui reste à décider, c&rsquo;est
où poser le point le long de cette direction.</b></p>

<h2><span class="n">2</span>La boucle GPTQ en une phrase</h2>
<p>Une matrice est quantifiée bloc par bloc, de gauche à droite. Après chaque
bloc, l&rsquo;erreur commise est projetée sur les colonnes pas encore traitées,
à travers la hessienne des activations. Chaque bloc suivant est donc choisi
<i>en sachant ce que les précédents ont raté</i>.</p>
<div class="key">
  <p style="margin:0">Le point qui décide de tout ce cours : la boucle ne connaît
  le fichier qu&rsquo;à travers ce résidu. Si le résidu ment, personne ne le
  rattrape plus tard.</p>
</div>

<h2><span class="n">3</span>Tout tient sur une demi-droite</h2>
<p>Soit un bloc <b>x</b> et sa direction codée <b>u</b>. Trois positions comptent
sur la demi-droite portée par u :</p>
<ul>
  <li><b>t = ‖x‖ · cos θ</b>, la projection. C&rsquo;est l&rsquo;optimum euclidien :
      le point de la droite le plus proche de x.</li>
  <li><b>‖x‖</b>, la norme du bloc. C&rsquo;est là que la rétraction sphérique
      pose le point.</li>
  <li><b>ĝ</b>, l&rsquo;un des deux niveaux du code de gain. C&rsquo;est le seul
      des trois que le fichier sait écrire.</li>
</ul>

__FIG1__
<figcaption>L&rsquo;encart de gauche pose la géométrie. Les trois règles du bas
montrent où chaque design pose le point (disque bleu) et contre quoi il fait
compenser la boucle (chevron orange).</figcaption>
</figure>

<div class="demo">
  <h3>Essayez : déplacez la norme du bloc</h3>
  <p class="hint">Les deux niveaux de gain sont fixes pour toute la ligne, aux
  valeurs mesurées 0,875 et 1,107. Seul le bloc bouge. Regardez l&rsquo;écart
  orange apparaître et changer de côté.</p>
  <svg id="ruler" viewBox="0 0 860 200" role="img" aria-label="Règle interactive"></svg>
  <div class="ctl">
    <label for="sl">norme du bloc ‖x‖</label>
    <input id="sl" type="range" min="0.62" max="1.38" step="0.002" value="1.02">
  </div>
  <div class="read">
    <div>‖x‖<b id="rx">1,020</b></div>
    <div>niveau choisi ĝ<b id="rg">1,107</b></div>
    <div>ĝ / ‖x‖<b id="rr">1,085</b></div>
    <div>l&rsquo;écart, ‖x‖ − ĝ<b class="gap" id="rd">−0,087</b></div>
  </div>
</div>

<h2><span class="n">4</span>Les trois designs, et pourquoi il y en a trois</h2>
<p>La ligne 5 de l&rsquo;Algorithme 3 du papier s&rsquo;écrit ainsi :</p>
<pre><code>W̃ ← (‖W̃‖ / ‖Q(W̃)‖) · Q(W̃)        puis        E ← W − W̃</code></pre>
<p>Le détail qui change tout : <b>elle affecte W̃</b>, les poids stockés. La
rétraction n&rsquo;est pas une astuce sur la rétroaction, c&rsquo;est une
amélioration de la reconstruction. La rétroaction devient honnête parce que le
bloc stocké <i>est</i> devenu ‖x‖·u. Et une norme exacte par bloc, ça coûte
un flottant libre, soit 0,67 bit par poids.</p>
<p>D&rsquo;où trois designs cohérents ou non, et pas deux :</p>
<table>
  <thead><tr><th>Design</th><th>Le fichier stocke</th><th>La boucle compense contre</th><th class="num">Débit</th><th>Cohérent</th></tr></thead>
  <tbody>
    <tr><td><b>A</b>, le chemin publié</td><td>ĝ · u</td><td>ĝ · u</td><td class="num">48 bits</td><td>oui</td></tr>
    <tr><td><b>B</b>, l&rsquo;Éq. 17 à la lettre</td><td>‖x‖ · u</td><td>‖x‖ · u</td><td class="num">47 + 16 bits</td><td>oui</td></tr>
    <tr class="hi"><td><b>sph</b>, ce qu&rsquo;on a mesuré</td><td>ĝ · u</td><td>‖x‖ · u</td><td class="num">48 bits</td><td><b>non</b></td></tr>
  </tbody>
</table>
<p>Le design B était déjà connu du dépôt depuis juillet 2026, et écarté non pas
sur la qualité mais sur le débit : 16 bits par bloc que la comptabilité ne
facturait pas. Le design <code>sph</code> était la tentative de garder le débit
de A et la rétroaction de B.</p>
<div class="key">
  <p style="margin:0"><b>À 0 bit de gain, la question ne se pose pas.</b> Le code
  stocke ‖x‖ exactement, donc A, B et sph sont le même objet. C&rsquo;est
  pourquoi le papier ne rencontre jamais ce problème : sa configuration
  recommandée tranche la magnitude <i>après</i> la boucle, par une résolution
  close, quand plus aucune rétroaction ne peut être trompée. Notre test
  <code>norm_preserving_direction_code_makes_the_flag_inert</code> le mesure :
  sur un code à magnitude libre, activer <code>sph</code> déplace la couche de
  1e-12.</p>
</div>

<h2><span class="n">5</span>L&rsquo;écart, mesuré</h2>
<p>Quelle est la taille de ce que <code>sph</code> cache à la boucle ? C&rsquo;est
le vecteur <b>(‖x‖ − ĝ) · u</b>, purement radial. Voici la distribution de
ĝ/‖x‖ sous l&rsquo;encodeur de production.</p>

__FIG2__
<figcaption>20 000 blocs gaussiens, encodeur Tetra de production. La barre verte
marque 1,000, où un codebook à 0 bit de gain met <i>tous</i> ses blocs. Survolez
une barre pour le compte exact.</figcaption>
</figure>

<table>
  <thead><tr><th>Quantité</th><th class="num">0 bit de gain</th><th class="num">Tetra, 1 bit</th></tr></thead>
  <tbody>
    <tr><td>ĝ / ‖x‖, moyenne</td><td class="num">1,00000</td><td class="num">1,00684</td></tr>
    <tr class="hi"><td>ĝ / ‖x‖, écart-type</td><td class="num">0</td><td class="num">0,09200</td></tr>
    <tr><td>p1 … p99</td><td class="num">1,000 … 1,000</td><td class="num">0,824 … 1,297</td></tr>
    <tr><td>énergie de l&rsquo;écart / erreur du fichier</td><td class="num">0</td><td class="num">8,7 %</td></tr>
  </tbody>
</table>
<p>Le biais est de 0,7 %, négligeable. C&rsquo;est <b>la dispersion de 9,2 %</b>
qui compte, parce qu&rsquo;elle change à chaque bloc. Aucune constante globale
ne la rattrape après coup.</p>

<h2><span class="n">6</span>Comment 8,7 % par bloc devient +31,9 %</h2>
<p>Parce que ça ne s&rsquo;oublie pas, ça s&rsquo;empile. Une ligne de
<code>down_proj</code> compte 85 blocs consécutifs, et chacun remet son erreur
au suivant.</p>

__FIG3__
<figcaption>Sous <code>nogs</code>, l&rsquo;aval absorbe exactement ce que le
bloc a raté. Sous <code>sph</code>, la tranche radiale sort de la flèche et
n&rsquo;est remise à personne, 85 fois par ligne, dans 196 matrices, sur
28 couches.</figcaption>
</figure>

<h2><span class="n">7</span>Le résultat, et ses contrôles</h2>
<p>Qwen3-0.6B, 28 blocs, un seul mot de configuration change entre les deux bras.
Préenregistrement tamponné avant le premier bloc, 35 minutes de Mac, 0 $.</p>
<table>
  <thead><tr><th>Bras</th><th>Rétroaction formée contre</th><th class="num">Perplexité</th><th class="num">Dégradation</th><th class="num">b/poids</th></tr></thead>
  <tbody>
    <tr><td><code>nogs</code>, publié</td><td>le bloc sur la grille de gain</td><td class="num">41,8875</td><td class="num">×2,148</td><td class="num">2,1656</td></tr>
    <tr class="hi"><td><code>sph</code></td><td>le bloc ramené à sa norme exacte</td><td class="num">55,2478</td><td class="num">×2,833</td><td class="num">2,1656</td></tr>
  </tbody>
</table>
<p>La prédiction signée avant le run disait « B sous A de 1 à 4 % ». Elle est
réfutée en sens inverse, à huit fois la largeur de son intervalle. Les cinq
contrôles passent : débit identique, base f32 identique à 19,5038, même nombre
de poids, un seul mot différent dans le diff des deux journaux, et le bras A
rejoue au dix-millième le 41,8875 mesuré une semaine plus tôt.</p>

<h2><span class="n">8</span>« Mais le papier annonce +1,9 point »</h2>
<p>Oui, et cette lecture ne tient pas non plus. En relisant la Table 9, chaque
ligne <i>GPTQ</i> porte la métrique de recherche euclidienne et chaque ligne
<i>Spherical GPTQ</i> porte l&rsquo;angulaire. Les deux variables bougent
ensemble dans les six paires.</p>

__FIG4__
<figcaption>La table ne remplit qu&rsquo;une diagonale. Son écart paie donc deux
changements à la fois, et l&rsquo;encodeur Tetra possède déjà le premier.</figcaption>
</figure>

<div class="warn">
  <h3>Ce que ça ne dit pas</h3>
  <ul style="margin-bottom:0">
    <li>Que le papier a tort. La Table 9 n&rsquo;isole pas ce que nos documents
        lui font dire, c&rsquo;est tout.</li>
    <li>Rien sur un codebook <b>sans</b> bit de gain sous notre boucle
        (<code>leech0c13</code>). C&rsquo;est la configuration du papier, et elle
        n&rsquo;a pas tourné ici.</li>
    <li>Une explication par la « dérive radiale » du code boule. Elle a été
        testée et réfutée : le code boule lit un écart-type de 0,083 sur sa
        magnitude, contre 0,092 pour Tetra. Le sien est plus serré, pas plus
        large.</li>
    <li>Un modèle, une graine, un jeu d&rsquo;évaluation, en perplexité
        seulement. À +31,9 % sur un seul mot, les graines ne changeraient pas le
        signe.</li>
  </ul>
</div>

<h2><span class="n">9</span>Ce qu&rsquo;il faut retenir</h2>
<ol>
  <li><b>Un résidu n&rsquo;est honnête que s&rsquo;il décrit ce que le fichier
      contient.</b> Toute optimisation qui change l&rsquo;un sans l&rsquo;autre
      fabrique une incohérence que 28 couches amplifient.</li>
  <li><b>Le bénéfice de la rétraction est dans la magnitude stockée, pas dans la
      rétroaction.</b> On ne peut pas en prendre la moitié gratuite : elle
      n&rsquo;existe pas.</li>
  <li><b>Un levier d&rsquo;un papier se lit avec la configuration du papier.</b>
      Ici : 0 bit de gain, magnitude tranchée après la boucle. Transposé à un
      gain codé dans la boucle, le même geste change de nature.</li>
  <li><b>Cinquième cas dans ce dépôt</b> d&rsquo;un gain local qui compose mal en
      profondeur, après design C, <code>group_scales</code>, gptq2 et tetrapost.
      Le motif est maintenant une règle de méthode, plus une anecdote.</li>
</ol>

<h2><span class="n">10</span>Refaire les mesures soi-même</h2>
<pre><code># la géométrie et les distributions, 4 secondes, aucun modèle
cargo run --release -p llvq-bench --example sphgeom

# les neuf tests qui verrouillent le drapeau
cargo test -p llvq-quant --test g5_spherical

# les deux bras du 0.6B, 35 min sur un Mac, une seule variable
LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_THREADS=12 \
  cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- \
  64 2048 12 2048 metal {nogs|sph} tetra 999 rot</code></pre>

<footer>
  Figures générées par <code>mkfigs.py</code>, page par <code>mkcours.py</code>,
  toutes deux dans ce dossier. Journal complet, préenregistrement tamponné et
  écarts signés dans <code>docs/mesures/sph-ppl-0.6b-2026-09-22.txt</code> et
  <code>proofs/preregistration-sph-ppl-0.6b-2026-09-22.md</code>.
</footer>
</div>

<script>
(function () {
  var G0 = 0.875003, G1 = 1.106938, COS = 0.96080;
  var LO = 0.55, HI = 1.45, X0 = 70, X1 = 790, Y = 124;
  var svg = document.getElementById("ruler"), sl = document.getElementById("sl");
  var NS = "http://www.w3.org/2000/svg";
  function px(v) { return X0 + (v - LO) / (HI - LO) * (X1 - X0); }
  function el(n, a, t) {
    var e = document.createElementNS(NS, n);
    for (var k in a) e.setAttribute(k, a[k]);
    if (t) e.textContent = t;
    return e;
  }
  function fr(v, d) { return v.toFixed(d).replace(".", ","); }
  function draw() {
    var xn = parseFloat(sl.value);
    var g = Math.abs(xn - G0) < Math.abs(xn - G1) ? G0 : G1;   // the shipped rule
    var t = xn * COS;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var ink2 = "var(--ink2)", rule = "var(--rule)";
    // fixed legend: the labels never move, only the markers do
    svg.appendChild(el("circle", {cx:X0+8, cy:30, r:8, fill:"var(--s1)"}));
    svg.appendChild(el("text", {x:X0+24, y:35, fill:"var(--s1)", "font-size":14,
                                "font-weight":600}, "ce que le fichier stocke : ĝ"));
    svg.appendChild(el("path", {d:"M 428 34 l -7.5 -12 h 15 z", fill:"var(--s2)"}));
    svg.appendChild(el("text", {x:446, y:35, fill:"var(--s2)", "font-size":14,
                                "font-weight":600}, "ce contre quoi sph compense : ‖x‖"));
    svg.appendChild(el("line", {x1:X0, y1:Y, x2:X1, y2:Y, stroke:rule, "stroke-width":3,
                                "stroke-linecap":"round"}));
    svg.appendChild(el("path", {d:"M "+X1+" "+Y+" l -11 -5.5 v 11 z", fill:rule}));
    svg.appendChild(el("text", {x:X1+12, y:Y+5, fill:ink2, "font-size":15,
                                "font-weight":600}, "u"));
    // the gap band
    var a = Math.min(px(g), px(xn)), c = Math.max(px(g), px(xn));
    if (c - a > 1)
      svg.appendChild(el("rect", {x:a, y:Y-14, width:c-a, height:28, rx:3,
                                  fill:"var(--s2)", opacity:.20}));
    // the two legal magnitudes
    [[G0,"niveau 0"],[G1,"niveau 1"]].forEach(function (p) {
      svg.appendChild(el("circle", {cx:px(p[0]), cy:Y, r:6, fill:"none",
                                    stroke:ink2, "stroke-width":1.8}));
      svg.appendChild(el("text", {x:px(p[0]), y:Y+44, fill:ink2, "font-size":13,
                                  "text-anchor":"middle"}, p[1]));
      svg.appendChild(el("text", {x:px(p[0]), y:Y+62, fill:ink2, "font-size":13,
                                  "text-anchor":"middle"}, fr(p[0],3)));
    });
    // t, the euclidean optimum
    svg.appendChild(el("line", {x1:px(t), y1:Y-11, x2:px(t), y2:Y+11,
                                stroke:rule, "stroke-width":2}));
    svg.appendChild(el("text", {x:px(t), y:Y-20, fill:ink2, "font-size":13,
                                "text-anchor":"middle"}, "t"));
    // what the file stores
    svg.appendChild(el("circle", {cx:px(g), cy:Y, r:9, fill:"var(--s1)",
                                  stroke:"var(--card)", "stroke-width":2}));
    // what sph compensates against
    svg.appendChild(el("path", {d:"M "+px(xn)+" "+(Y-16)+" l -7.5 -12 h 15 z",
                                fill:"var(--s2)"}));
    document.getElementById("rx").textContent = fr(xn,3);
    document.getElementById("rg").textContent = fr(g,3);
    document.getElementById("rr").textContent = fr(g/xn,3);
    document.getElementById("rd").textContent = (xn>=g?"+":"−") + fr(Math.abs(xn-g),3);
  }
  sl.addEventListener("input", draw);
  draw();
})();
</script>
</html>
"""

out = PAGE
for i, name in enumerate(FIGS, 1):
    out = out.replace(f"__FIG{i}__", "<figure>" + inline(name))
p = os.path.join(HERE, "cours-spherical-gptq.html")
open(p, "w").write(out)
print("wrote", os.path.basename(p), os.path.getsize(p), "B")
