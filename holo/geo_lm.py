"""Transplant the space structure INTO a live LM and optimize it.

A real byte-level LM (embedding table + RoPE transformer + head). We add the
free-energy geometry (PPMI attraction + entropy/symmetry spread) as an AUX LOSS
on the LIVE embedding table, so the model's own byte representations organize by
radius=hub / direction=category WHILE it learns to predict.

Tests: (1) feasibility — trains stably? (2) does aux help/hurt bits/byte?
(3) does the embedding organize (frequent 'hub' bytes pulled central)?
Sweep λ (geometry weight) = optimize.
"""
from __future__ import annotations
import sys, math, argparse
sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
import torch.nn.functional as F
from hsl_rope import RoPECausalBlock

CORPUS = (
    b"the quick brown fox jumps over the lazy dog. "
    b"holographic signal layer encodes bytes as phase. "
    b"the sea is calm and the waves move slowly to the shore. "
    b"a singer plays the piano as the melody fills the quiet room. "
) * 12


def ppmi_from_corpus(data: bytes, device):
    """adjacent-byte co-occurrence -> PPMI-weighted pairs over bytes present."""
    pairs, deg = {}, {}
    for a, b in zip(data[:-1], data[1:]):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    total = float(sum(pairs.values()))
    present = sorted(deg); sub = {b: k for k, b in enumerate(present)}   # byte -> subset index
    ks = list(pairs)
    i = torch.tensor([sub[k[0]] for k in ks], device=device)            # subset indices
    j = torch.tensor([sub[k[1]] for k in ks], device=device)
    cij = torch.tensor([pairs[k] for k in ks], dtype=torch.float32, device=device)
    ci = torch.tensor([deg[k[0]] for k in ks], dtype=torch.float32, device=device)
    cj = torch.tensor([deg[k[1]] for k in ks], dtype=torch.float32, device=device)
    ppmi = (torch.log((cij * total) / (ci * cj) + 1e-9)).clamp(min=0.0)
    degv = torch.tensor([deg[b] for b in present], dtype=torch.float32)
    return (i, j, ppmi), present, degv


def geometry_loss(E, pos, beta=1.0):
    """free energy on the embedding table: PPMI attraction + entropy spread."""
    i, j, w = pos
    d_pos = (E[i] - E[j]).norm(dim=1)
    attraction = (w * d_pos).sum() / (w.sum() + 1e-9)
    dirs = E / E.norm(dim=1, keepdim=True).clamp(min=1e-6)
    cos = dirs @ dirs.t()
    n = E.shape[0]
    crowd = (cos.sum() - n) / (n * (n - 1))                 # mean off-diagonal cosine (entropy term)
    return attraction + beta * crowd


class GeoLM(nn.Module):
    def __init__(self, dim=128, layers=3, heads=4):
        super().__init__()
        self.emb = nn.Embedding(256, dim)
        self.blocks = nn.ModuleList([RoPECausalBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, 256)

    def forward(self, tok):
        x = self.emb(tok)
        m = torch.ones(tok.shape, device=tok.device)
        for b in self.blocks:
            x = b(x, phase=None, key_padding_mask=(m == 0))
        return self.head(self.norm(x))


def batch(data, seq, bs, device):
    s = torch.randint(0, len(data) - seq - 1, (bs,))
    tok = torch.stack([torch.tensor(list(data[k:k + seq]), dtype=torch.long) for k in s.tolist()])
    tgt = torch.stack([torch.tensor(list(data[k + 1:k + seq + 1]), dtype=torch.long) for k in s.tolist()])
    return tok.to(device), tgt.to(device)


def run(lam, steps, device, seed=0):
    torch.manual_seed(seed)
    model = GeoLM().to(device)
    pos, present, degv = ppmi_from_corpus(CORPUS, device)
    present_t = torch.tensor(present, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        tok, tgt = batch(CORPUS, 48, 32, device)
        logits = model(tok)
        ce = F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1))
        loss = ce
        if lam > 0:
            loss = ce + lam * geometry_loss(model.emb.weight[present_t], pos, beta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval
    with torch.no_grad():
        tok, tgt = batch(CORPUS, 48, 256, device)
        bpb = F.cross_entropy(model(tok).reshape(-1, 256), tgt.reshape(-1)).item() / math.log(2)
        E = model.emb.weight[present_t].cpu()
        radius = E.norm(dim=1)
        d = (degv - degv.mean()) / degv.std().clamp(min=1e-6)
        r = (radius - radius.mean()) / radius.std().clamp(min=1e-6)
        hub = (d * r).mean().item()
    print(f"  λ={lam:<4} bits/byte={bpb:.3f}  hub(degree↔radius)={hub:+.3f}")
    return bpb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"transplant test: free-energy geometry as aux loss on live embedding (device={device})")
    print("λ=0 is the plain baseline; λ>0 = space structure applied")
    for lam in [0.0, 0.05, 0.1, 0.3, 1.0]:
        run(lam, args.steps, device)


if __name__ == "__main__":
    main()
