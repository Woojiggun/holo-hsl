"""Does 'connection mass' warp space? — flat (Euclidean) vs curved (hyperbolic)
embedding of word co-occurrence. User's idea: hubs (frequent connectors like
the/of/and, or Korean 조사) should distort space and get pulled to the center,
while topic words spread to the rim and cluster.

We embed each word in 2D twice (Euclidean disk and Poincaré/hyperbolic disk),
trained so co-occurring words are close (skip-gram negative sampling). Then we
measure: do high-degree HUB words sit near the origin? In hyperbolic geometry
they should (that's why Poincaré embeddings exist); in flat space, weakly.
"""
from __future__ import annotations
import sys, re, math
sys.stdout.reconfigure(encoding="utf-8")
import torch

CORPUS = """
the sea was calm and the waves moved slowly over the smooth sand.
a sailor watched the water and the sky as the ship sailed to the harbor.
the ocean is deep and the waves are cold and the wind is strong.
the city was loud and the streets were full of cars and people.
a driver waited at the corner as the lights changed and the road cleared.
the streets are busy and the buildings are tall and the noise is constant.
the music was soft and the song moved slowly over the quiet room.
a singer played the piano and the notes rose as the melody filled the air.
the song is gentle and the rhythm is slow and the sound is warm.
the sea and the city and the song all share the calm of the evening.
""" * 6


def tokenize(t):
    return re.findall(r"[a-z]+", t.lower())


def build(tokens, window=2):
    vocab = sorted(set(tokens))
    idx = {w: i for i, w in enumerate(vocab)}
    pairs = {}
    deg = [0] * len(vocab)
    for i, w in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if j == i:
                continue
            a, b = idx[w], idx[tokens[j]]
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
            deg[a] += 1
    return vocab, pairs, torch.tensor(deg, dtype=torch.float32)


def poincare_dist(u, v):
    diff = ((u - v) ** 2).sum(-1)
    nu = (u ** 2).sum(-1).clamp(max=1 - 1e-5)
    nv = (v ** 2).sum(-1).clamp(max=1 - 1e-5)
    return torch.acosh(1 + 2 * diff / ((1 - nu) * (1 - nv)) + 1e-7)


def dist(u, v, hyperbolic):
    return poincare_dist(u, v) if hyperbolic else (u - v).norm(dim=-1)


def train(vocab, pairs, hyperbolic, steps=2000, K=10, device="cpu"):
    """Poincaré-style loss: for each co-occurrence (i,j), the true j must be the
    NEAREST among {j, K random negatives}. Negatives get pushed away -> no collapse."""
    n = len(vocab)
    pos = torch.tensor(list(pairs.keys()), device=device)            # [P,2]
    w = torch.tensor(list(pairs.values()), dtype=torch.float32, device=device)
    emb = torch.nn.Parameter((torch.rand(n, 2, device=device) - 0.5) * 0.01)
    opt = torch.optim.Adam([emb], lr=0.01)
    i, j = pos[:, 0], pos[:, 1]
    for _ in range(steps):
        if hyperbolic:                                               # keep inside unit disk
            with torch.no_grad():
                norm = emb.norm(dim=1, keepdim=True)
                emb.mul_(torch.clamp(norm, max=0.999) / norm.clamp(min=1e-9))
        ui = emb[i]
        d_pos = dist(ui, emb[j], hyperbolic)                         # [P]
        neg = torch.randint(0, n, (pos.shape[0], K), device=device)  # [P,K]
        d_neg = dist(ui[:, None, :], emb[neg], hyperbolic)           # [P,K]
        logits = -torch.cat([d_pos[:, None], d_neg], dim=1)          # nearest should be pos (idx 0)
        ce = torch.nn.functional.cross_entropy(
            logits, torch.zeros(pos.shape[0], dtype=torch.long, device=device), reduction="none")
        loss = (w * ce).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return emb.detach()


def report(name, vocab, emb, deg):
    radius = emb.norm(dim=1)
    # hub centrality: do high-degree words sit near origin? -> negative correlation wanted
    d = (deg - deg.mean()) / deg.std().clamp(min=1e-6)
    r = (radius - radius.mean()) / radius.std().clamp(min=1e-6)
    corr = (d * r).mean().item()
    order = radius.argsort()
    central = [vocab[k] for k in order[:8].tolist()]
    rim = [vocab[k] for k in order[-8:].tolist()]
    print(f"\n[{name}]  degree↔radius corr = {corr:+.3f}  (negative = hubs central)")
    print(f"  CENTER (most central): {central}")
    print(f"  RIM    (most outer)  : {rim}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = tokenize(CORPUS)
    vocab, pairs, deg = build(tokens)
    print(f"tokens={len(tokens)} vocab={len(vocab)} pairs={len(pairs)} device={device}")
    top = deg.argsort(descending=True)[:8]
    print("top hubs by degree:", [vocab[k] for k in top.tolist()])
    for name, hyp in [("EUCLIDEAN (flat)", False), ("HYPERBOLIC (curved)", True)]:
        emb = train(vocab, pairs, hyp, device=device)
        report(name, vocab, emb.cpu(), deg)


if __name__ == "__main__":
    main()
