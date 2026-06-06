"""Cross-modal binding on a REAL VIDEO (Prelinger PD narrated film), scaled from digits.

video_to_stream.py turned a real narrated film into per-window byte streams:
   [16x16 frame raster] SEP [mu-law audio] SEP [whisper caption] WSEP
One byte-LM trains next-byte on these (no per-modality work). Binding test on HELD-OUT windows:
   does the model predict the CAPTION text better when given the window's OWN frame+audio
   than when given a MISMATCHED window's frame+audio?   matched bpb << mismatched => bound.
(The caption is the transcript of the concurrent speech, so audio<->text is a real signal,
 plus the visual scene — exactly the 'baby watching video' tri-modal co-occurrence.)
"""
from __future__ import annotations
import sys, pathlib, json, math, random, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_rope import RoPECausalBlock
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG, AUD, TXT = 256, 256, 24
SEP, WSEP = 254, 255
N = IMG + 1 + AUD + 1 + TXT + 1                 # 539
TXT0 = IMG + 1 + AUD + 1                          # 514: text region start
TXT1 = TXT0 + TXT                                 # 538


def load_streams(path, captioned_only=True):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if captioned_only and not r.get("caption"): continue
        b = bytes.fromhex(r["bytes_hex"])
        if len(b) == N: out.append(b)
    return out


def feats_of(streams):
    F_ = torch.zeros(len(streams), N, FEAT_DIM); I = torch.zeros(len(streams), N, dtype=torch.long)
    for i, b in enumerate(streams):
        f, _ = signal_features(b); F_[i] = f; I[i] = torch.tensor(list(b))
    return F_, I


class ByteLM(nn.Module):
    def __init__(self, dim, layers, heads):
        super().__init__()
        self.proj = nn.Linear(FEAT_DIM, dim)
        self.blocks = nn.ModuleList([RoPECausalBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, 256)
    def forward(self, feats):
        x = self.proj(feats)
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))


def text_bpb(model, feats, ids, bs=32):
    model.eval(); tot = 0.0; ntok = 0
    with torch.no_grad():
        for s in range(0, feats.shape[0], bs):
            lo = model(feats[s:s+bs].to(DEV))
            tgt = ids[s:s+bs, TXT0:TXT1].to(DEV)
            pred = lo[:, TXT0-1:TXT1-1]
            tot += float(F.cross_entropy(pred.reshape(-1, 256), tgt.reshape(-1), reduction="sum")); ntok += tgt.numel()
    return (tot / max(ntok, 1)) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=r"C:\Users\gguni\holo_v6_data\video_streams.jsonl")
    ap.add_argument("--dim", type=int, default=256); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    rng = random.Random(0); torch.manual_seed(0)
    streams = load_streams(a.streams)
    print(f"real-video tri-modal binding | windows={len(streams)} | [img{IMG}|aud{AUD}|txt{TXT}] N={N} | {DEV}", flush=True)
    rng.shuffle(streams)
    k = int(len(streams) * 0.85)
    tr, te = streams[:k], streams[k:]
    print(f"train windows {len(tr)} | held-out {len(te)}", flush=True)
    ftr, itr = feats_of(tr)
    model = ByteLM(a.dim, a.layers, a.heads).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(a.steps):
        idx = torch.randint(0, ftr.shape[0], (16,))
        lo = model(ftr[idx].to(DEV)); tgt = itr[idx][:, 1:].to(DEV)
        loss = F.cross_entropy(lo[:, :-1].reshape(-1, 256), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step+1) % 1000 == 0: print(f"  step {step+1}  loss {loss.item():.3f}", flush=True)

    # held-out: matched vs mismatched (swap frame+audio from a different held-out window)
    fm, im = feats_of(te)
    mis = te[:]; perm = list(range(len(te))); rng.shuffle(perm)
    mismatched = []
    for i, b in enumerate(te):
        j = perm[i] if perm[i] != i else (perm[i]+1) % len(te)
        donor = te[j]
        mismatched.append(donor[:TXT0] + b[TXT0:])     # frame+audio from j, caption from i
    fx, ix = feats_of(mismatched)
    bm = text_bpb(model, fm, im); bx = text_bpb(model, fx, ix)
    print(f"\n[HELD-OUT real-video windows]")
    print(f"  CAPTION bits/byte  matched(own frame+audio): {bm:.3f}   mismatched: {bx:.3f}   Δ={bx-bm:+.3f}")
    if bx > bm:
        print("  => caption predicted BETTER with the matching frame+audio: real-video cross-modal binding.")
    else:
        print("  => no binding gap on this clip (honest null).")
    print("\namateur POC on ONE real PD film — pipeline + binding, not a superiority claim.")


if __name__ == "__main__":
    main()
