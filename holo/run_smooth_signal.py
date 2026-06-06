"""Multimodal proof, step 1 (synthetic): does HSL beat raw on SMOOTH signals?

The make-or-break test. On TEXT we measured HSL ≈ raw (text is broadband). Prediction:
on SMOOTH signals (audio/image-like) the signal features (Δ/Δ²/Fourier/phase) ARE the
natural representation and should beat raw bytes. Control = random bytes (should be parity,
like text). 2×2: {smooth, random} × {raw, hsl}, next-byte bits/byte. Same model both arms.

8-bit signals (1 byte/sample) so byte-level HSL features == sample-level features.
"""
from __future__ import annotations
import sys, pathlib, math, time, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_rope import RoPECausalBlock
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smooth_signal(N, rng):
    """sum of a few low-frequency sinusoids -> normalize -> 8-bit. random per sequence."""
    t = np.arange(N)
    sig = np.zeros(N)
    for _ in range(rng.integers(2, 5)):
        f = rng.uniform(1, 6)                         # low freq = smooth
        sig += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t / N + rng.uniform(0, 2 * np.pi))
    sig = (sig - sig.min()) / (np.ptp(sig) + 1e-9)
    return (sig * 255).round().astype(np.uint8).tobytes()


def random_signal(N, rng):
    return rng.integers(0, 256, N, dtype=np.uint8).tobytes()


def make_set(kind, n, N, seed):
    rng = np.random.default_rng(seed)
    gen = smooth_signal if kind == "smooth" else random_signal
    feats = torch.zeros(n, N, FEAT_DIM); ids = torch.zeros(n, N, dtype=torch.long)
    for i in range(n):
        b = gen(N, rng)
        f, _p = signal_features(b); feats[i] = f[:N]
        ids[i] = torch.tensor(list(b))
    return feats, ids


class LM(nn.Module):
    def __init__(self, arm, dim, layers, heads):
        super().__init__()
        self.arm = arm
        if arm == "raw":
            self.emb = nn.Embedding(256, dim)
        else:
            self.proj = nn.Linear(FEAT_DIM, dim)
        self.blocks = nn.ModuleList([RoPECausalBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, 256)

    def forward(self, feats, ids):
        x = self.emb(ids) if self.arm == "raw" else self.proj(feats)
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x))


def run(arm, tr, va, dim, layers, heads, steps, bs=32, lr=1e-3):
    ftr, itr = tr; fva, iva = va
    torch.manual_seed(0)
    m = LM(arm, dim, layers, heads).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=lr)
    n = ftr.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        lo = m(ftr[idx], itr[idx])
        loss = F.cross_entropy(lo[:, :-1].reshape(-1, 256), itr[idx][:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        lo = m(fva, iva)
        nll = F.cross_entropy(lo[:, :-1].reshape(-1, 256), iva[:, 1:].reshape(-1))
    return float(nll) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256); ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--n", type=int, default=8000)
    a = ap.parse_args()
    print(f"smooth-signal proof | N={a.N} (8-bit, 1 byte/sample) | dim{a.dim}/{a.layers}L | {DEV}")
    print("prediction: smooth -> HSL beats raw ; random -> parity (like text)\n", flush=True)
    print(f"{'signal':>8} {'raw bpb':>9} {'hsl bpb':>9} {'Δ(hsl-raw)':>11} {'winner':>8}")
    for kind in ("smooth", "random"):
        ftr, itr = make_set(kind, a.n, a.N, 1); fva, iva = make_set(kind, a.n // 5, a.N, 2)
        tr = (ftr.to(DEV), itr.to(DEV)); va = (fva.to(DEV), iva.to(DEV))
        r = run("raw", tr, va, a.dim, a.layers, a.heads, a.steps)
        h = run("hsl", tr, va, a.dim, a.layers, a.heads, a.steps)
        win = "HSL" if h < r - 0.02 else ("raw" if r < h - 0.02 else "tie")
        print(f"{kind:>8} {r:>9.3f} {h:>9.3f} {h-r:>+11.3f} {win:>8}", flush=True)
    print("\nHSL<raw on smooth + tie on random = the differentiator (HSL's reason-to-exist) is REAL.")


if __name__ == "__main__":
    main()
