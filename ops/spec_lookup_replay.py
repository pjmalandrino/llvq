# /// script
# dependencies = ["tokenizers"]
# ///
"""Replay prompt-lookup policies offline (2026-11-08, prereg spec-final section 1).

Greedy decoding makes acceptance a function of the greedy text alone, so a policy's rounds replay exactly from the
prompt and the greedy completion. The completions here are spec-lookup's printed texts, retokenized: newlines were
printed as spaces, so the code prompts (edit above all) replay lower than the card read. A round costs the measured
verify pass of its rows in greedy tokens (spec-gate), or one token without a draft (4E's token launch).

  uv run ops/spec_lookup_replay.py <tokenizer.json of Qwen3>
"""
import re, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
src = open('llvq-llm/src/bin/specrun.rs').read()
def const(name):
    m = re.search(r'const %s: &str = "(.*?)";' % name, src, re.S)
    s = m.group(1).replace('\\\n', '')
    return s.replace('\\"', '"')
msgs = {
 'chat': const('CHAT'), 'code': const('CODE'),
 'summary': const('SUMMARY') + "\n\n" + const('SUMMARY_TEXT'),
 'edit': const('EDIT') + "\n\n" + const('EDIT_CODE'),
 'rag': const('RAG') + "\n\n" + const('SUMMARY_TEXT'),
}
def enc(s): return tok.encode(s, add_special_tokens=False).ids
def prompt(m): return enc("<|im_start|>user\n" + m + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
cost = {'4b': [1, 1.95, 2.61, 3.39], '8b': [1, 2.05, 2.81, 3.79], '14b': [1, 2.25, 3.10, 4.11]}
def draft(ctx, k, mn, mx=3):
    L = len(ctx)
    for n in range(min(mx, L - 1), mn - 1, -1):
        tail = ctx[L-n:]
        for i in range(L - n - 1, -1, -1):
            if ctx[i:i+n] == tail:
                return ctx[i+n:i+n+k], n
    return [], 0
def run(P, G, c, k, mn, route=True, gate=0):
    # G: greedy output incl. first. gate: draft only after `gate` consecutive accepted drafts (copy mode), 0 = always
    ctx = P + G[:1]; m = 1; t = 0.0; streak = 0
    while m < len(G):
        kk = min(k, len(G) - m - 1)
        d, n = draft(ctx, kk, mn) if kk > 0 else ([], 0)
        if gate and streak < gate and d:
            # probe with one draft only
            d = d[:1]
        a = 0
        while a < len(d) and m + a < len(G) and d[a] == G[m + a]: a += 1
        t += c[len(d)] if (d or not route) else 1.0
        if not d and not route: t += 0.26
        streak = streak + a if (d and a == len(d)) else (0 if d else streak)
        e = min(a + 1, len(G) - m)
        ctx += G[m:m+e]; m += e
    return (len(G) - 1) / t
texts = {}
for size in ['4b', '8b', '14b']:
    f = 'docs/mesures/spec-lookup-2026-11-08-brut/' + ('spec-4b.txt' if size == '4b' else 'r1/spec-%s.txt' % size)
    lines = [l for l in open(f) if l.startswith('  text: ')]
    texts[size] = [l[len('  text: '):].rstrip('\n') for l in lines]
names = ['chat', 'code', 'summary', 'edit', 'rag']
for size in ['4b', '8b', '14b']:
    c = cost[size]
    print('==', size)
    rows = []
    for i, nm in enumerate(names):
        P = prompt(msgs[nm]); G = enc(texts[size][i])
        res = {}
        res['old k1'] = run(P, G, c, 1, 1, route=False)
        for k in (1, 2, 3):
            for mn in (1, 2, 3, 4):
                res['k%d n%d' % (k, mn)] = run(P, G, c, k, mn)
        for g in (1, 2, 3):
            for mn in (2, 3):
                res['k3 n%d gate%d' % (mn, g)] = run(P, G, c, 3, mn, gate=g)
        rows.append(res)
    keys = list(rows[0].keys())
    print('%-14s' % 'policy' + ''.join('%9s' % n for n in names) + '   geo-free  min')
    for kname in keys:
        v = [r[kname] for r in rows]
        geo = (v[0] * v[1] * v[2]) ** (1/3)
        print('%-14s' % kname + ''.join('%9.3f' % x for x in v) + '   %7.3f %6.3f' % (geo, min(v)))
