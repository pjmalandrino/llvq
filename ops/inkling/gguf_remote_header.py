# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Parse the header and tensor table of a remote GGUF by HTTP range requests only.

Usage: uv run gguf_remote_header.py <url or local path> <out.json>
"""
import json, struct, sys, os, requests

def token():
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else None

class Remote:
    def __init__(self, url):
        self.url = url; self.buf = b""; self.h = {}
        t = token()
        if t: self.h["Authorization"] = f"Bearer {t}"
    def ensure(self, n):
        if not self.url.startswith("http"):
            if len(self.buf) < n:
                with open(self.url, "rb") as f:
                    self.buf = f.read(max(n, len(self.buf) + (8 << 20)))
            return
        while len(self.buf) < n:
            start = len(self.buf); end = max(n, start + (8 << 20)) - 1
            assert end - start < (64 << 20), "header read too large"
            r = requests.get(self.url, headers={**self.h, "Range": f"bytes={start}-{end}"}, timeout=120)
            assert r.status_code == 206, r.status_code
            self.buf += r.content

TYPES = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",8:"str",9:"arr",10:"u64",11:"i64",12:"f64"}
FMT = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}

def parse(url):
    R = Remote(url); pos = 0
    def rd(fmt):
        nonlocal pos
        n = struct.calcsize(fmt); R.ensure(pos + n); v = struct.unpack_from(fmt, R.buf, pos); pos += n; return v[0]
    def rstr():
        nonlocal pos
        n = rd("<Q"); R.ensure(pos + n); s = R.buf[pos:pos+n].decode("utf-8", "replace"); pos += n; return s
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            et = rd("<I"); n = rd("<Q")
            vals = [rval(et) for _ in range(n)]
            return vals if n <= 64 else {"array_type": TYPES[et], "len": n}
        return rd(FMT[t])
    assert R.ensure(4) or R.buf[:4] == b"GGUF"
    pos = 4; ver = rd("<I"); nt = rd("<Q"); nkv = rd("<Q")
    kv = {}
    for _ in range(nkv):
        k = rstr(); t = rd("<I"); kv[k] = rval(t)
    tensors = []
    for _ in range(nt):
        name = rstr(); nd = rd("<I"); dims = [rd("<Q") for _ in range(nd)]; typ = rd("<I"); off = rd("<Q")
        tensors.append({"name": name, "ne": dims, "type": typ, "offset": off})
    align = kv.get("general.alignment", 32)
    data_start = (pos + align - 1) // align * align
    return {"url": url, "version": ver, "kv": kv, "tensors": tensors, "data_start": data_start}

if __name__ == "__main__":
    h = parse(sys.argv[1])
    json.dump(h, open(sys.argv[2], "w"), indent=1)
    print(h["version"], len(h["tensors"]), h["data_start"])
