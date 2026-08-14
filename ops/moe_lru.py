#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Simulateur LRU sur une trace de routage temporelle — P2, §2.4 et V0.4.

Répond à une question que le dump agrégé ne peut pas poser : **la localité
temporelle du routage MoE**. Le dump du 2026-08-12 est agrégé par cellule
(couche, expert), donc il ne borne qu'un *oracle statique* — le meilleur choix
d'un ensemble figé de cellules chaudes. Un LRU peut battre cet oracle s'il
existe de la localité, et le ciseau le dit lui-même
(`ops/moe_ciseau.py:445-450` : « trancher exigerait un dump TEMPOREL, qui
n'existe pas »).

**Ce que ce script mesure** : `hit_LRU(ordre décode)`, le taux de cellules déjà
résidentes au moment où la couche en a besoin, en régime décode mono-séquence
(batch 1), cache global sur les cellules (couche, expert).

**Ce qu'il ne mesure pas** : le temps. Un hit n'est pas gratuit, un miss coûte
une copie PCIe, et aucune des deux n'est chronométrée ici. Le script compte des
événements ; c'est le §4.0 du pré-enregistrement qui les convertit en budget.

Stdlib seule : aucun GPU, aucun réseau, aucune dépendance — il tourne sur la
trace, pas sur le modèle.

Usage :
    python3 ops/moe_lru.py --trace ~/llvq-moe/trace-....u8
    python3 ops/moe_lru.py --selftest        # les cinq cas de V0.4, sans trace
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict


# --------------------------------------------------------------------------
# Le simulateur
# --------------------------------------------------------------------------


class Lru:
    """LRU vrai : admission au miss, **rafraîchissement de la récence au hit**,
    éviction du moins récemment utilisé.

    Le rafraîchissement au hit est le paramètre porteur : c'est le seul
    mécanisme par lequel un LRU peut battre un oracle statique. Un cache qui
    n'ordonne que par date d'insertion est un FIFO, et les cas 1 à 3 de V0.4 ne
    les distinguent pas — seul le cas 4 le fait.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError(f"capacité négative : {capacity}")
        self.capacity = capacity
        self.slots: OrderedDict[int, None] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.compulsory = 0  # premiers accès : inévitables quelle que soit la taille
        self._seen: set[int] = set()

    def touch(self, cell: int) -> bool:
        """Accède à `cell`. Rend True si elle était résidente."""
        if cell in self.slots:
            self.slots.move_to_end(cell)  # ← le rafraîchissement, cf. la docstring
            self.hits += 1
            return True
        self.misses += 1
        if cell not in self._seen:
            self._seen.add(cell)
            self.compulsory += 1
        if self.capacity == 0:
            return False
        self.slots[cell] = None
        if len(self.slots) > self.capacity:
            self.slots.popitem(last=False)
        return False

    @property
    def total(self) -> int:
        return self.hits + self.misses

    def rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


def simulate(stream, capacity: int) -> Lru:
    c = Lru(capacity)
    for cell in stream:
        c.touch(cell)
    return c


def decode_order(trace: bytes, n_tokens: int, n_layers: int, top_k: int, n_experts: int):
    """Les cellules dans l'ordre du DÉCODE : `pour t, pour ℓ, les K`.

    Ce n'est pas l'ordre d'exécution du dump — un prefill traite tous les
    tokens d'une fenêtre à la couche 0, puis à la couche 1. C'est l'ordre dans
    lequel un cache travaillerait en génération, qui est le régime du package A.
    L'argument qui autorise la transposition (routeur causal, donc mêmes
    décisions par token) est un ARGUMENT, pas une mesure : il est déclaré au
    §2.4 du pré-enregistrement, et non vérifié.

    Une cellule est `ℓ * n_experts + e` — l'unité opposable, jamais l'expert
    seul (le piège d'agrégation du §1).
    """
    stride = n_layers * top_k
    for t in range(n_tokens):
        base = t * stride
        for l in range(n_layers):
            off = base + l * top_k
            for k in range(top_k):
                yield l * n_experts + trace[off + k]


# --------------------------------------------------------------------------
# V0.4 — les cinq cas, à passer AVANT de lire une ligne de trace
# --------------------------------------------------------------------------


def selftest() -> int:
    """Les cinq cas du §V0.4. Rend 0 si tous passent."""
    fails = []

    def check(name, got, want, why):
        ok = abs(got - want) < 1e-9 if isinstance(want, float) else got == want
        print(f"  [{'ok ' if ok else 'ÉCHEC'}] {name}: {got} (attendu {want})")
        if not ok:
            fails.append(f"{name} — {why}")

    print("V0.4 — le simulateur, sur flux synthétiques à réponse connue\n")

    # 1. ensemble de travail ≤ taille ⇒ 100 % hors miss obligatoires.
    c = simulate([0, 1, 2] * 100, capacity=3)
    check("1. ensemble de travail tenant en cache",
          c.hits, 300 - 3,
          "un cache qui contient tout doit tout garder")
    check("1bis. miss obligatoires comptés", c.compulsory, 3,
          "les premiers accès sont inévitables et doivent être comptés à part")

    # 2. tourniquet sur taille + 1 ⇒ 0 hit. Le pire cas d'un LRU.
    c = simulate([0, 1, 2, 3] * 50, capacity=3)
    check("2. tourniquet sur taille+1", c.hits, 0,
          "chaque cellule est évincée juste avant d'être redemandée")

    # 3. i.i.d. plat ⇒ hit ≈ taille / cellules. Générateur déterministe.
    n_cells, cap, n = 100, 25, 200_000
    x = 12345
    flat = []
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        flat.append(x % n_cells)
    c = simulate(flat, capacity=cap)
    got, want = c.rate(), cap / n_cells
    ok3 = abs(got - want) < 0.02
    print(f"  [{'ok ' if ok3 else 'ÉCHEC'}] 3. i.i.d. plat: {got:.4f} (attendu ≈ {want:.4f})")
    if not ok3:
        fails.append("3. i.i.d. plat — sur une source sans localité, un LRU vaut sa taille relative")

    # 4. LE cas qui tue le FIFO. A B A C A, taille 2.
    #    LRU : A(miss) B(miss) A(HIT, rafraîchit A) C(miss, évince B) A(HIT) = 2.
    #    FIFO: A(miss) B(miss) A(HIT)  C(miss, évince A) A(miss)       = 1.
    c = simulate([0, 1, 0, 2, 0], capacity=2)
    check("4. A B A C A, taille 2 (tue le FIFO)", c.hits, 2,
          "sans rafraîchissement au hit c'est un FIFO, qui rendrait 1 — et alors "
          "aucun LRU ne pourrait battre l'oracle statique, donc tout le §4.2 tombe")

    # 5. LE cas qui exerce l'ordre des K. taille 2, K = 2.
    #    Décroissant : X Y | Z X  →  X(miss) Y(miss) Z(miss, évince X) X(miss) = 0.
    #    Inverse     : Y X | X Z  →  Y(miss) X(miss) X(HIT) Z(miss)            = 1.
    dec = simulate([10, 11, 12, 10], capacity=2)
    inv = simulate([11, 10, 10, 12], capacity=2)
    check("5. ordre des K, score décroissant", dec.hits, 0,
          "l'ordre d'insertion des K change le verdict : il doit être figé (§2.1)")
    check("5bis. ordre des K, inverse", inv.hits, 1,
          "le contraste avec 5 est ce qui prouve que l'ordre est exercé")

    print()
    if fails:
        print("V0.4 ROUGE — aucune ligne de trace ne doit être lue :")
        for f in fails:
            print(f"  · {f}")
        return 1
    print("V0.4 VERT — les cinq cas passent, la lecture de la trace est autorisée.")
    return 0


# --------------------------------------------------------------------------
# Lecture d'une trace réelle
# --------------------------------------------------------------------------


def read_trace(path: str):
    """En-tête JSON d'une ligne, puis le tableau plat. Un fichier absent
    ÉCHOUE — il ne saute pas (convention de maison, forme de
    `ops/moe_ciseau.py:206-208`)."""
    if not os.path.exists(path):
        sys.exit(
            f"trace introuvable : {path}\n"
            "  Elle se produit par :\n"
            "    uv run ops/moe_routing.py --model openai/gpt-oss-20b --tokens 131072 \\\n"
            "        --ctx 1024 --device mps --dataset allenai/c4 \\\n"
            "        --json docs/data/moe-routing-....json --trace ~/llvq-moe/trace-....u8\n"
            "  Ce script ne fabrique aucun substitut : un run sur une trace absente\n"
            "  serait un fumigène."
        )
    with open(path, "rb") as f:
        head = json.loads(f.readline().decode("utf-8"))
        body = f.read()
    return head, body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", help="trace temporelle produite par moe_routing.py --trace")
    ap.add_argument("--selftest", action="store_true", help="les cinq cas de V0.4, sans trace")
    ap.add_argument(
        "--alphas",
        default="0.2733,0.35,0.45,0.50,0.607,0.7161",
        help="fractions de cellules résidentes à simuler",
    )
    a = ap.parse_args()

    if a.selftest or not a.trace:
        rc = selftest()
        if not a.trace:
            return rc
        if rc:
            return rc
        print()

    head, body = read_trace(a.trace)
    n_tokens = head["tokens"]
    n_layers = len(head["layer_names"])
    top_k = head["top_k"]
    n_experts = head["n_experts"]
    cells = n_layers * n_experts
    want = n_tokens * n_layers * top_k
    if n_experts > 256:
        sys.exit("trace uint16 : non implémenté (règle posée d'avance, §2.1)")
    if len(body) != want:
        sys.exit(f"trace tronquée : {len(body)} octets pour {want} attendus")

    print(f"# trace   : {a.trace}")
    print(f"# sha256  : {hashlib.sha256(body).hexdigest()}")
    print(f"# modèle  : {head.get('model')}  transformers {head.get('transformers_version')}")
    print(f"# tokens  : {n_tokens}  couches {n_layers}  top_k {top_k}  experts {n_experts}")
    print(f"# cellules: {cells}   visites/token {n_layers * top_k}")
    print(f"# tokens_sha256 : {head.get('tokens_sha256')}")
    print()
    print(f"  {'α':>8}{'cellules':>10}{'hit LRU':>10}{'miss/token':>12}{'obligatoires':>14}")
    print("  " + "-" * 54)
    for alpha in [float(s) for s in a.alphas.split(",")]:
        cap = int(alpha * cells)  # convention ceil/floor : figée à floor, imprimée
        c = simulate(decode_order(body, n_tokens, n_layers, top_k, n_experts), cap)
        print(
            f"  {alpha:>8.4f}{cap:>10}{c.rate() * 100:>9.3f}%"
            f"{c.misses / n_tokens:>12.4f}{c.compulsory / max(c.misses, 1) * 100:>13.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
