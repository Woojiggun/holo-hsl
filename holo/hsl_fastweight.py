"""Tier-2 memory: fast-weight (Hebbian / linear-attention) associative memory.

Between [dense window (attention, O(N²))] and [RAG (retrieval)] sits a CONTINUOUS
short-term memory that carries out-of-window same-conversation state in O(d²) state:
  M_t = Σ_{i≤t} φ(k_i) ⊗ v_i      (causal cumulative)
  o_t = φ(q_t) · M_t / (φ(q_t)·Σφ(k_i))
Persists across windows if M is carried. Recalls beyond the attention window cheaply.

Smoke replicates the parallel-room finding (window-attn ≈ random vs fast-weight ≈ 1.0)
on a beyond-window associative-recall task, IN OUR codebase.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class FastWeightMemory(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.h = heads; self.dk = dim // heads
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))               # DIAL: starts at 0 (off); opens iff it pays

    @staticmethod
    def phi(x):
        return F.elu(x) + 1.0                                   # positive feature map

    def forward(self, x, chunk=128):                            # [B,L,d] causal
        """CHUNKED-RECURRENT linear attention. Carries state S[B,h,dk,dk] across chunks
        instead of materializing the naive [B,h,L,dk,dk] cumsum (which OOMs at dim512/long L).
        Memory O(B*h*(chunk^2 + dk^2)) per step; mathematically identical to the cumsum form."""
        B, L, d = x.shape
        qkv = self.qkv(self.norm(x)).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # [B,h,L,dk]
        pq, pk = self.phi(q), self.phi(k)
        S = x.new_zeros(B, self.h, self.dk, self.dk)            # Hebbian state  S[a,b]=Σ pk[a]·v[b]
        kz = x.new_zeros(B, self.h, self.dk)                    # Σ pk  (denominator state)
        outs = []
        for c0 in range(0, L, chunk):
            c1 = min(c0 + chunk, L)
            pqc, pkc, vc = pq[:, :, c0:c1], pk[:, :, c0:c1], v[:, :, c0:c1]   # [B,h,C,dk]
            num_inter = pqc @ S                                              # past chunks
            A = (pqc @ pkc.transpose(-1, -2)).tril()                        # [B,h,C,C] intra, causal incl. diagonal
            num = num_inter + A @ vc                                        # read [B,h,C,dk]
            kz_full = kz.unsqueeze(2) + pkc.cumsum(dim=2)                   # [B,h,C,dk]
            z = (pqc * kz_full).sum(-1, keepdim=True) + 1e-6
            outs.append(num / z)
            S = S + pkc.transpose(-1, -2) @ vc                             # update state over chunk
            kz = kz + pkc.sum(dim=2)
        o = torch.cat(outs, dim=2).transpose(1, 2).reshape(B, L, d)
        return x + torch.tanh(self.gate) * self.proj(o)        # tanh(gate)=0 at init -> no effect until learned


class WindowAttn(nn.Module):
    """Baseline: causal self-attn limited to the last W positions (a moving window)."""
    def __init__(self, dim, heads=4, W=8):
        super().__init__()
        self.h = heads; self.dk = dim // heads; self.W = W
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim); self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, L, d = x.shape
        qkv = self.qkv(self.norm(x)).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        s = (q @ k.transpose(-1, -2)) / (self.dk ** 0.5)
        i = torch.arange(L, device=x.device)[:, None]; j = torch.arange(L, device=x.device)[None, :]
        mask = (j > i) | (j <= i - self.W)                     # causal + only last W
        s = s.masked_fill(mask[None, None], float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, L, d)
        return x + self.proj(o)


if __name__ == "__main__":
    import random
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); rng = random.Random(0)
    NK, P, W, dim = 32, 24, 8, 64                              # 24 pairs, window 8 (query is far beyond)
    SEP = 2 * NK
    VOCAB = 2 * NK + 1
    L = 2 * P + 2                                              # k1 v1 ... kP vP SEP kq -> predict vq

    def make(B):
        x = torch.zeros(B, L, dtype=torch.long); y = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            keys = rng.sample(range(NK), P); vals = [rng.randrange(NK) for _ in range(P)]
            seq = []
            for kk, vv in zip(keys, vals): seq += [kk, NK + vv]
            qi = rng.randrange(P)
            seq += [SEP, keys[qi]]
            x[b] = torch.tensor(seq); y[b] = NK + vals[qi]     # the value paired with the queried key
        return x.to(dev), y.to(dev)

    class M(nn.Module):
        def __init__(self, kind, layers=3):
            super().__init__(); self.emb = nn.Embedding(VOCAB, dim)
            self.pos = nn.Parameter(torch.randn(1, L, dim) * 0.02)
            mk = (lambda: FastWeightMemory(dim)) if kind == "fast" else (lambda: WindowAttn(dim, W=W))
            self.blocks = nn.ModuleList([mk() for _ in range(layers)])
            self.head = nn.Linear(dim, VOCAB)
        def forward(self, x):
            h = self.emb(x) + self.pos[:, :x.shape[1]]
            for b in self.blocks: h = b(h)
            return self.head(h[:, -1])                          # predict at final position

    for kind in ("window", "fast"):
        torch.manual_seed(0)
        m = M(kind).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        for _ in range(4000):
            x, y = make(64); loss = F.cross_entropy(m(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            xv, yv = make(512); acc = float((m(xv).argmax(-1) == yv).float().mean())
        print(f"{kind:7} beyond-window recall acc = {acc:.3f}  (window W={W}, query {2*P}+ tokens back)")
    print("expect: window ~ random(1/32≈0.03), fast-weight -> high. confirms tier-2 in our code.")
