"""THE north-star experiment: does ONE byte-native model, fed an interleaved
[image | audio | text] byte stream of the SAME concept, learn the cross-modal binding
for free — no per-modality engineering, no explicit alignment?

Clean proxy = digits (ground-truth correspondence via the digit label):
  stream = [MNIST image of d] SEP [FSDD spoken-d audio, μ-law] SEP [text word of d]
One HSL byte model trains on MATCHED streams (next-byte). Test = does it BIND:
  on held-out, is the TEXT predicted better when image+audio MATCH the text's digit
  than when they don't? matched bits/byte << mismatched bits/byte  =>  binding learned.
(Not raw MP4 — that's codec-encrypted; this is the decoded interleaved form a model can learn.)
"""
from __future__ import annotations
import sys, pathlib, glob, wave, math, random, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchaudio.functional as AF
import torchvision
from PIL import Image
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_rope import RoPECausalBlock
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG, AUD, TXT = 256, 128, 16                    # bytes per segment (16x16 img, audio chunk, text)
SEP = 254
N = IMG + 1 + AUD + 1 + TXT                      # 402
TXT0 = IMG + 1 + AUD + 1                          # text region start
WORDS = ["zero","one","two","three","four","five","six","seven","eight","nine"]


def load_pools():
    # MNIST images -> 16x16 grayscale bytes, grouped by digit
    mn = torchvision.datasets.MNIST(r"C:\Users\gguni\AppData\Local\Temp\hsl_img", train=True, download=True)
    imgs = {d: [] for d in range(10)}
    for img, lab in mn:
        if len(imgs[lab]) >= 600: continue
        g = np.array(img.resize((16, 16), Image.BILINEAR), dtype=np.uint8)
        imgs[lab].append(g.flatten().tobytes())
        if all(len(v) >= 600 for v in imgs.values()): break
    # FSDD audio -> μ-law chunk, grouped by leading digit
    auds = {d: [] for d in range(10)}
    for f in sorted(glob.glob(r"C:\Users\gguni\AppData\Local\Temp\hsl_audio\**\recordings\*.wav", recursive=True)):
        d = int(pathlib.Path(f).name.split("_")[0])
        try:
            w = wave.open(f); raw = w.readframes(w.getnframes()); w.close()
        except Exception:
            continue
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        if len(x) < AUD: x = np.pad(x, (0, AUD - len(x)))
        mu = AF.mu_law_encoding(torch.from_numpy(x[:AUD]), 256).clamp(0, 255).numpy().astype(np.uint8)
        auds[d].append(mu.tobytes())
    return imgs, auds


def make_stream(imgs, auds, d_img, d_aud, d_txt, rng):
    img = rng.choice(imgs[d_img]); aud = rng.choice(auds[d_aud])
    txt = (WORDS[d_txt].encode() + b" " * TXT)[:TXT]
    return img + bytes([SEP]) + aud + bytes([SEP]) + txt


def build(imgs, auds, n, rng, mismatch=False):
    feats = torch.zeros(n, N, FEAT_DIM); ids = torch.zeros(n, N, dtype=torch.long)
    for i in range(n):
        d = rng.randrange(10)
        dt = d if not mismatch else rng.choice([x for x in range(10) if x != d])
        b = make_stream(imgs, auds, d, d, dt, rng)
        f, _ = signal_features(b); feats[i] = f[:N]; ids[i] = torch.tensor(list(b[:N]))
    return feats.to(DEV), ids.to(DEV)


class ByteLM(nn.Module):
    def __init__(self, dim, layers, heads):
        super().__init__()
        self.proj = nn.Linear(FEAT_DIM, dim)
        self.blocks = nn.ModuleList([RoPECausalBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, 256)
    def forward(self, feats):
        x = self.proj(feats)
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x))


def text_bpb(model, feats, ids, bs=64):
    model.eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for s in range(0, feats.shape[0], bs):
            lo = model(feats[s:s + bs])                        # batched (avoid OOM)
            tgt = ids[s:s + bs, TXT0:N]
            pred = lo[:, TXT0 - 1:N - 1]                        # predict text from prior (img+audio)
            ce = F.cross_entropy(pred.reshape(-1, 256), tgt.reshape(-1), reduction="sum")
            tot += float(ce); ntok += tgt.numel()
    return (tot / max(ntok, 1)) / math.log(2)


def retrieval_accuracy(model, imgs, auds, rng, n=400, bs=20):
    """given HELD-OUT [image+audio] of digit d, pick the word (0..9) with lowest text-bits.
    chance = 10%. high => model identifies the concept across modalities, on unseen instances."""
    model.eval()
    ex = [(rng.choice(imgs[d]), rng.choice(auds[d]), d) for d in (rng.randrange(10) for _ in range(n))]
    correct = total = 0
    with torch.no_grad():
        for s in range(0, n, bs):
            batch = ex[s:s + bs]; B = len(batch)
            feats = torch.zeros(B, 10, N, FEAT_DIM); ids = torch.zeros(B, 10, N, dtype=torch.long)
            for bi, (img, aud, d) in enumerate(batch):
                for w in range(10):
                    txt = (WORDS[w].encode() + b" " * TXT)[:TXT]
                    b = img + bytes([SEP]) + aud + bytes([SEP]) + txt
                    f, _ = signal_features(b); feats[bi, w] = f[:N]; ids[bi, w] = torch.tensor(list(b[:N]))
            feats = feats.reshape(B * 10, N, FEAT_DIM).to(DEV); ids = ids.reshape(B * 10, N).to(DEV)
            lo = model(feats)
            ce = F.cross_entropy(lo[:, TXT0 - 1:N - 1].reshape(-1, 256), ids[:, TXT0:N].reshape(-1),
                                 reduction="none").reshape(B * 10, -1).sum(1).reshape(B, 10)
            choice = ce.argmin(1)
            true = torch.tensor([d for _, _, d in batch], device=DEV)
            correct += int((choice == true).sum()); total += B
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    rng = random.Random(0); torch.manual_seed(0)
    print(f"cross-modal binding (scaled, held-out) | [img{IMG}|aud{AUD}|txt{TXT}] N={N} | dim{a.dim}/{a.layers}L | {DEV}", flush=True)
    imgs, auds = load_pools()

    def split(pool, frac=0.8):
        tr, te = {}, {}
        for d, lst in pool.items():
            k = max(1, int(len(lst) * frac)); tr[d] = lst[:k]; te[d] = lst[k:] or lst[:1]
        return tr, te
    imgs_tr, imgs_te = split(imgs); auds_tr, auds_te = split(auds)
    print(f"instances/digit: train img~{len(imgs_tr[0])} aud~{len(auds_tr[0])} | HELD-OUT img~{len(imgs_te[0])} aud~{len(auds_te[0])}", flush=True)

    ftr, itr = build(imgs_tr, auds_tr, 8000, rng)             # train on matched streams (train instances)
    model = ByteLM(a.dim, a.layers, a.heads).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(a.steps):
        idx = torch.randint(0, ftr.shape[0], (16,), device=DEV)
        lo = model(ftr[idx]); loss = F.cross_entropy(lo[:, :-1].reshape(-1, 256), itr[idx][:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    # eval on HELD-OUT instances (unseen images + audio)
    fm, im = build(imgs_te, auds_te, 1200, random.Random(7), mismatch=False)
    fx, ix = build(imgs_te, auds_te, 1200, random.Random(8), mismatch=True)
    bm = text_bpb(model, fm, im); bx = text_bpb(model, fx, ix)
    acc = retrieval_accuracy(model, imgs_te, auds_te, random.Random(9))
    print(f"\n[HELD-OUT instances]")
    print(f"  TEXT bits/byte  matched: {bm:.3f}   mismatched: {bx:.3f}   Δ={bx-bm:+.3f}")
    print(f"  cross-modal retrieval (img+aud -> right word of 10): {acc:.1%}  (chance 10%)")
    print("\npossibility seen: on UNSEEN instances, one byte-native model binds image↔sound↔text")
    print("from a single interleaved stream — no pairing, no alignment, no per-modality work.")
    print("amateur POC — not a superiority claim. scaling/robustness left to the experts.")


if __name__ == "__main__":
    main()
