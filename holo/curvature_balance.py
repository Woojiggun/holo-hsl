"""PMI + entropy balance — free energy of the concept space.

Two fixes in one:
  (1) PMI weighting: discount hub domination (the/and co-occur with everything),
      so the ANGLE can encode topic instead of being washed out by hubs.
  (2) Entropy / symmetry term: connection (attraction) = energy, pulling toward
      hubs (bias). Add a spreading force (maximize angular spread) = entropy.
      loss = attraction(PPMI) + β · angular_crowding.  Balancing them = free energy.
      β is the 'temperature': 0 = pure structure (collapsed), high = pure symmetry.

Metrics: hub(radius↔degree), topic(angular within vs across cos),
         symmetry(||mean direction|| — lower = more balanced/대칭).
"""
from __future__ import annotations
import sys, math
sys.stdout.reconfigure(encoding="utf-8")
import torch
from emergent_curvature import CORPUS, tokenize, build
from curvature_viz import TOPICS, topic_of


def ppmi_pairs(vocab, pairs, deg):
    """positive pointwise mutual info weight per co-occurrence pair."""
    total = float(sum(pairs.values()))
    ks = list(pairs.keys())
    cij = torch.tensor([pairs[k] for k in ks], dtype=torch.float32)
    i = torch.tensor([k[0] for k in ks]); j = torch.tensor([k[1] for k in ks])
    ci, cj = deg[i], deg[j]
    pmi = torch.log((cij * total) / (ci * cj) + 1e-9)
    ppmi = pmi.clamp(min=0.0)                                  # discount hubs -> ~0
    return torch.stack([i, j], 1), ppmi


def train(vocab, pos, w, beta, steps=2500, K=10, dim=3, device="cpu"):
    n = len(vocab)
    pos = pos.to(device); w = w.to(device)
    emb = torch.nn.Parameter((torch.rand(n, dim, device=device) - 0.5) * 0.01)
    opt = torch.optim.Adam([emb], lr=0.01)
    i, j = pos[:, 0], pos[:, 1]
    for _ in range(steps):
        ui = emb[i]
        d_pos = (ui - emb[j]).norm(dim=1)
        neg = torch.randint(0, n, (pos.shape[0], K), device=device)
        d_neg = (ui[:, None, :] - emb[neg]).norm(dim=2)
        logits = -torch.cat([d_pos[:, None], d_neg], dim=1)
        ce = torch.nn.functional.cross_entropy(
            logits, torch.zeros(pos.shape[0], dtype=torch.long, device=device), reduction="none")
        attraction = (w * ce).sum() / (w.sum() + 1e-9)
        # entropy term: push DIRECTIONS apart (minimize crowding) -> angular symmetry
        dirs = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-6)
        cos = dirs @ dirs.t()
        crowd = (cos.sum() - n) / (n * (n - 1))               # mean off-diagonal cosine
        loss = attraction + beta * crowd
        opt.zero_grad(); loss.backward(); opt.step()
    return emb.detach().cpu()


def metrics(vocab, emb, deg):
    radius = emb.norm(dim=1)
    d = (deg - deg.mean()) / deg.std().clamp(min=1e-6)
    r = (radius - radius.mean()) / radius.std().clamp(min=1e-6)
    hub = (d * r).mean().item()
    dirs = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
    ids = {t: [k for k, ww in enumerate(vocab) if topic_of(ww) == t] for t in TOPICS}
    def avgcos(A, B, same):
        s, c = 0.0, 0
        for a in A:
            for b in B:
                if same and a >= b:
                    continue
                s += (dirs[a] * dirs[b]).sum().item(); c += 1
        return s / max(c, 1)
    within = sum(avgcos(v, v, True) for v in ids.values()) / len(ids)
    ts = list(ids); across = 0.0; n = 0
    for x in range(len(ts)):
        for y in range(x + 1, len(ts)):
            across += avgcos(ids[ts[x]], ids[ts[y]], False); n += 1
    across /= max(n, 1)
    symmetry = dirs.mean(0).norm().item()                     # ||mean dir||: lower = balanced
    return hub, within, across, symmetry


def plot3d(vocab, emb, path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
    except Exception as e:
        print(f"  (no matplotlib: {e})"); return
    colors = {"sea": "#1f77b4", "city": "#d62728", "music": "#2ca02c", "function": "#888888"}
    e = emb.numpy(); fig = plt.figure(figsize=(10, 9)); ax = fig.add_subplot(111, projection="3d")
    for k, wd in enumerate(vocab):
        t = topic_of(wd)
        ax.scatter(e[k, 0], e[k, 1], e[k, 2], c=colors[t], s=22)
        ax.text(e[k, 0], e[k, 1], e[k, 2], wd, fontsize=6, color=colors[t])
    ax.scatter([0], [0], [0], c="black", s=60, marker="*")
    ax.set_title("PMI + entropy balance (free energy): radius=hub, direction=topic")
    fig.savefig(path, dpi=110, bbox_inches="tight"); print(f"  saved: {path}")


def main():
    tokens = tokenize(CORPUS)
    vocab, pairs, deg = build(tokens)
    pos, w = ppmi_pairs(vocab, pairs, deg)
    print("β=temperature (0=pure structure → high=pure symmetry)")
    print(f"{'β':>5} {'hub corr':>9} {'within':>8} {'across':>8} {'topic Δ':>8} {'symmetry':>9}")
    best = None
    for beta in [0.0, 0.5, 1.0, 2.0, 4.0]:
        emb = train(vocab, pos, w, beta)
        hub, within, across, sym = metrics(vocab, emb, deg)
        print(f"{beta:>5.1f} {hub:>+9.3f} {within:>+8.3f} {across:>+8.3f} {within-across:>+8.3f} {sym:>9.3f}")
        if beta == 1.0:
            best = emb
    if best is not None:
        plot3d(vocab, best, "curvature_balance.png")


if __name__ == "__main__":
    main()
