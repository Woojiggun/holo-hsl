"""Canonical encoder eval — raw / nonlearned-HSL / learned-HSL, corrected features.

Both HSL arms use the same withheld byte→signal feature substrate (exact, invertible;
recipe not disclosed — see hsl_signal_encoder).
Discrimination task (minor/shuffled/different) on real POC text. Same plain body for
raw/nonlearned; the learned arm is the full LearnedHSLEncoder (VQ + phase attention).
"""
from __future__ import annotations
import sys, pathlib, json, random, time, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hsl_signal_encoder import signal_features, pack_batch, FEAT_DIM
from hsl_learned_encoder import LearnedHSLEncoder

POC_TRAIN = r"G:\내 드라이브\홀로빗 POC\data\train.jsonl"
SEP = 255
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_texts(path, n, min_bytes):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("text") or d.get("serialized") or d.get("output") or d.get("input") or ""
            b = t.encode("utf-8", "replace")
            if len(b) >= min_bytes:
                out.append(b)
            if len(out) >= n:
                break
    return out


def minor_edit(b, rng, frac=12):
    a = bytearray(b)
    for _ in range(max(1, len(a) // frac)):
        i = rng.randrange(len(a)); a[i] = rng.randrange(256)
    return bytes(a)


def make_examples(bases, half, rng):
    X, Y = [], []
    m = len(bases)
    for i, full in enumerate(bases):
        base = full[:half]
        if len(base) < 8:
            continue
        other = bases[(i + m // 3 + 1) % m][:half]
        sh = bytearray(base); rng.shuffle(sh)
        for var, lab in [(minor_edit(base, rng), 0), (bytes(sh), 1), (other, 2)]:
            X.append(base + bytes([SEP]) + var); Y.append(lab)
    return X, Y


def pack(byte_list, max_len):
    feats, phase, mask = pack_batch(byte_list, max_len)
    bid = torch.zeros(len(byte_list), max_len, dtype=torch.long)
    for i, data in enumerate(byte_list):
        arr = list(data[:max_len]); bid[i, :len(arr)] = torch.tensor(arr, dtype=torch.long)
    return feats, phase, mask, bid


class Body(nn.Module):
    def __init__(self, dim, layers, heads, max_len, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        x = x + self.pos[:, : x.shape[1]]
        return self.norm(self.enc(x, src_key_padding_mask=(mask == 0)))


class PlainReader(nn.Module):
    def __init__(self, arm, dim, layers, heads, max_len):
        super().__init__()
        self.arm = arm
        if arm == "raw":
            self.emb = nn.Embedding(256, dim)
        else:
            self.proj = nn.Linear(FEAT_DIM, dim)
        self.body = Body(dim, layers, heads, max_len)
        self.head = nn.Linear(dim, 3)

    def forward(self, feats, bid, mask):
        x = self.emb(bid) if self.arm == "raw" else self.proj(feats)
        h = self.body(x, mask)
        m = mask.unsqueeze(-1)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1.0))


def run_plain(arm, tr, va, dim, layers, heads, max_len, steps, bs=32, lr=1e-3, seed=11):
    ftr, ptr, mtr, btr, ytr = tr
    fva, pva, mva, bva, yva = va
    torch.manual_seed(seed)
    model = PlainReader(arm, dim, layers, heads, max_len).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = ftr.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        loss = F.cross_entropy(model(ftr[idx], btr[idx], mtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(fva, bva, mva).argmax(1)
    return _report(pred, yva, sum(p.numel() for p in model.parameters()))


def run_learned(tr, va, dim, layers, heads, max_len, steps, num_codes=64, bs=32, lr=1e-3, seed=11):
    ftr, ptr, mtr, btr, ytr = tr
    fva, pva, mva, bva, yva = va
    torch.manual_seed(seed)
    model = LearnedHSLEncoder(dim=dim, num_codes=num_codes, layers=layers, heads=heads,
                              n_classes=3, max_len=max_len).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = ftr.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        logits, vq, codes, _ = model(ftr[idx], ptr[idx], mtr[idx])
        loss = F.cross_entropy(logits, ytr[idx]) + vq
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, _, _, _ = model(fva, pva, mva)
        pred = logits.argmax(1)
    return _report(pred, yva, sum(p.numel() for p in model.parameters()))


def _report(pred, y, params):
    acc = float((pred == y).float().mean())
    per = [float((pred[y == l] == l).float().mean()) for l in (0, 1, 2)]
    return acc, per, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--val-rows", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    args = ap.parse_args()
    random.seed(11); torch.manual_seed(11); np.random.seed(11)
    half = args.max_len // 2

    texts = load_texts(POC_TRAIN, args.rows + args.val_rows, min_bytes=half + 4)
    random.Random(11).shuffle(texts)
    tr_rows, va_rows = texts[:args.rows], texts[args.rows:args.rows + args.val_rows]
    Xtr, Ytr = make_examples(tr_rows, half, random.Random(13))
    Xva, Yva = make_examples(va_rows, half, random.Random(14))
    tr = (*[t.to(DEV) for t in pack(Xtr, args.max_len)], torch.tensor(Ytr).to(DEV))
    va = (*[t.to(DEV) for t in pack(Xva, args.max_len)], torch.tensor(Yva).to(DEV))
    print(f"encoder eval (corrected: exact phase + codec Δ/∂²/boundary) | FEAT_DIM {FEAT_DIM} "
          f"| train ex {len(Xtr)} val ex {len(Xva)} | body dim{args.dim}/{args.layers}L | {DEV}\n", flush=True)
    print(f"{'arm':12s} {'disc acc':>9s}  minor/shuf/diff   params")
    for arm in ("raw", "nonlearned"):
        t0 = time.time()
        acc, per, p = run_plain(arm, tr, va, args.dim, args.layers, args.heads, args.max_len, args.steps)
        print(f"{arm:12s} {acc:9.3f}  {per[0]:.3f}/{per[1]:.3f}/{per[2]:.3f}   {p}  ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    acc, per, p = run_learned(tr, va, args.dim, args.layers, args.heads, args.max_len, args.steps)
    print(f"{'learned':12s} {acc:9.3f}  {per[0]:.3f}/{per[1]:.3f}/{per[2]:.3f}   {p}  ({time.time()-t0:.0f}s)", flush=True)
    print("\nboth HSL arms use the same withheld feature substrate (recipe not disclosed).")


if __name__ == "__main__":
    main()
