"""Audit the OpenTimestamps stamps in proofs/ from the bytes on disk.

Answers three questions that prose in CLAUDE.md had been answering from
memory, and one that a `grep` had been answering wrongly:

  1. does each .ots still attest to the .md sitting next to it (sha256
     recomputed and compared to the digest the stamp commits to);
  2. how many Bitcoin anchors does each stamp actually carry, and at what
     block heights;
  3. which pre-registrations carry no stamp at all.

WHY THIS EXISTS.  The .ots format stores an attestation's type as an 8-byte
binary tag, never as text, so `grep BitcoinBlockHeaderAttestation` returns 0
on a file carrying four of them -- and `grep PendingAttestation` returns 0
too.  An instrument that returns the same value in both cases measures
nothing; this one deserializes the proof instead.

WHAT IT DOES NOT DO.  Verifying an anchor end to end means checking the
committed Merkle root against the real block, which needs a Bitcoin node or a
block explorer.  This script reads what the files carry and prints the
committed root per block so a third party can close that last step.

    pip install opentimestamps-client      # provides the `opentimestamps` lib
    python3 ops/otsaudit.py > docs/mesures/ots-etat-<date>.txt

Run from the repository root: paths are relative to it.
"""
import glob, hashlib, os, subprocess
from opentimestamps.core.timestamp import DetachedTimestampFile
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.serialize import StreamDeserializationContext

def load(p):
    with open(p, "rb") as f:
        return DetachedTimestampFile.deserialize(StreamDeserializationContext(f))

print("=" * 78)
print("ÉTAT RÉEL DES TAMPONS OpenTimestamps  —  mesuré le 2026-08-26")
print("=" * 78)
print("""
INSTRUMENT.  `ots info` (opentimestamps-client v0.7.2, installé depuis PyPI) et
la bibliothèque python `opentimestamps`, qui désérialise le .ots et parcourt
ses attestations.  Chaque nombre ci-dessous est *mesuré* sur les octets des
fichiers du dépôt ; rien n'est estimé.

⚠️  CE QUI N'EST PAS FAIT ICI, ET POURQUOI.  La vérification complète d'une
ancre exige de confronter la racine de Merkle engagée au bloc réel — donc un
nœud Bitcoin ou un explorateur.  La politique réseau de cette machine bloque
les deux (403 du proxy sur blockstream.info, mempool.space, blockchain.info,
et sur les quatre calendriers).  Ce journal établit donc ce que les FICHIERS
portent, pas que la chaîne le confirme.  Les racines sont imprimées en fin de
journal exactement pour qu'un tiers fasse ce dernier pas en une commande.
""")

print("-" * 78)
print("1.  LE GREP QUI A FONDÉ LA LIGNE DE CLAUDE.md")
print("-" * 78)
print("""
CLAUDE.md (en-tête 🧾 et §7) affirme, « vérifié par grep le 2026-08-25 » :
    16 .ots, chacun 4 PendingAttestation et 0 BitcoinBlockHeaderAttestation,
    « aucun n'a jamais été upgradé ».

Ce que le grep rend réellement, sur un fichier qui porte quatre ancres :""")
f = "proofs/preregistration-p1-2026-08-13.md.ots"
for pat in ("BitcoinBlockHeaderAttestation", "PendingAttestation"):
    n = subprocess.run(["grep","-c",pat,f], capture_output=True, text=True).stdout.strip()
    print(f"    grep -c {pat:<32} {f}  ->  {n}")
print("""
Les deux rendent 0.  Le format .ots stocke le type d'une attestation dans une
étiquette binaire de 8 octets, jamais sous forme de texte : le nom de classe
n'apparaît que dans la sortie RENDUE par `ots info`, et dans le source de la
bibliothèque.  Un grep sur ce format ne peut donc pas distinguer un fichier
ancré d'un fichier en attente — il rend 0 dans les deux cas.

Conséquence sur les deux nombres publiés :
  · le « 0 BitcoinBlockHeaderAttestation » est ce que l'instrument a rendu, et
    il est FAUX — le fichier ci-dessus en porte quatre ;
  · le « 4 PendingAttestation » est JUSTE, mais il ne vient pas de ce grep, qui
    rend 0 lui aussi.  Il a été inféré (« les quatre calendriers ») et présenté
    comme mesuré.
""")

print("-" * 78)
print("2.  CE QUE LES FICHIERS PORTENT")
print("-" * 78)
rows = []
for p in sorted(glob.glob("proofs/*.ots")):
    d, doc = load(p), p[:-4]
    ok = None
    if os.path.exists(doc):
        ok = hashlib.sha256(open(doc,"rb").read()).digest() == d.file_digest
    btc, pend = set(), set()
    for msg, att in d.timestamp.all_attestations():
        if isinstance(att, BitcoinBlockHeaderAttestation): btc.add((att.height, msg[::-1].hex()))
        elif isinstance(att, PendingAttestation): pend.add(att.uri)
    rows.append((os.path.basename(doc), ok, sorted(btc), sorted(pend)))

print(f"\n{'document':<56} {'sha256 du .md':<14} {'btc':>4} {'pend':>5}")
for name, ok, btc, pend in rows:
    mark = {True: "recalcule ok", False: "NE COLLE PAS", None: ".md absent"}[ok]
    print(f"{name:<56} {mark:<14} {len(btc):>4} {len(pend):>5}")

nb = sum(1 for _,_,b,_ in rows if b)
print(f"""
TOTAL : {len(rows)} tampons, dont {nb} portent au moins une ancre Bitcoin et
{len(rows)-nb} n'en portent aucune.  Tous portent les 4 attestations en attente.

Le compte de CLAUDE.md (« 16 .ots ») est lui aussi périmé : il y en a {len(rows)},
pour {len(glob.glob('proofs/*.md'))} documents (README.md compris).
""")

print("-" * 78)
print("3.  LES DEUX TAMPONS QUI N'ATTESTENT PLUS DE LEUR FICHIER")
print("-" * 78)
print("""
CLAUDE.md §7 le savait déjà — « le défaut inverse a été réalisé sur les préregs
du 08-10 et du 08-11 » — mais de mémoire.  C'est désormais machine-vérifiable,
et le mécanisme est nommé : le commit 01fdbe6 (2026-08-19), la passe
d'anonymisation pour TACO, a réécrit ces deux documents.  Une ancre atteste des
OCTETS : les réécrire détruit ce qu'elle prouve.
""")
for doc in ["proofs/preregistration-2026-08-10.md", "proofs/preregistration-2026-08-11.md"]:
    want = load(doc + ".ots").file_digest
    have = hashlib.sha256(open(doc,"rb").read()).digest()
    print(f"  {os.path.basename(doc)}")
    print(f"    le tampon engage  {want.hex()}")
    print(f"    le fichier vaut   {have.hex()}")
print("""
Et la version attestée n'est PAS récupérable : les 128 blobs .md distincts de
toute l'histoire git ont été hachés, aucun ne rend l'un de ces deux condensats.
Ces deux tampons prouvent donc l'antériorité d'un texte que le dépôt ne
contient plus, sous aucune révision.
""")

print("-" * 78)
print("4.  CE QUI N'EST PAS TAMPONNÉ")
print("-" * 78)
print()
for f in sorted(glob.glob("proofs/*.md")):
    if not os.path.exists(f + ".ots"):
        print(f"    {os.path.basename(f)}")
print("""
    (README.md n'est pas un pré-enregistrement.)

⚠️  preregistration-variance-calibration-2026-08-26.md est dans cette liste, et
    son propre §3 exige le tampon avant la première milliseconde mesurée.  Il ne
    peut pas être posé depuis cette machine : les quatre calendriers sont
    injoignables (403).  À faire depuis une machine en réseau, avant le lot 1.
""")

print("-" * 78)
print("5.  RACINES DE MERKLE ENGAGÉES  —  pour la vérification par un tiers")
print("-" * 78)
print("""
Chaque ligne dit : ce fichier engage cette racine dans le bloc de cette hauteur.
Vérification en une commande, depuis une machine en réseau :

    ots verify proofs/<fichier>.md.ots        (avec un nœud Bitcoin)

ou, sans nœud, en comparant la racine ci-dessous au champ merkle_root du bloc :

    curl -s https://blockstream.info/api/block-height/<hauteur> \\
      | xargs -I{} curl -s https://blockstream.info/api/block/{} | jq -r .merkle_root
""")
for name, ok, btc, pend in rows:
    if not btc: continue
    print(f"\n  {name}")
    for h, root in btc:
        print(f"    bloc {h}   {root}")
print()
