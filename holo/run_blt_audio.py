"""Dynamic-patching BLT on AUDIO. HSL-encoded input (proven > raw on audio) + dynamic
patch boundaries from a signal change-rate criterion vs fixed stride. Byte-AR output.

  local encoder (HSL feats, 2L) -> segment-mean pool by patch_id -> MAIN transformer
  (on P patches, cheap) -> per-byte cond = prior-patch ctx -> local byte-AR decoder (2L).
Dynamic patch_id: cuts at top-(P-1) |Δsample| (signal transitions). Fixed: uniform stride.
Compare bits/sample dynamic vs fixed (same P). Causal, no future leak.
"""
from __future__ import annotations
import sys, pathlib, glob, wave, math, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchaudio.functional as AF
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_rope import RoPECausalBlock
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mulaw(glob_pat, max_bytes):
    out = bytearray()
    for f in sorted(glob.glob(glob_pat)):
        if f.lower().endswith(".npy"):
            x = np.load(f).astype(np.float32)
        else:
            try:
                w = wave.open(f); raw = w.readframes(w.getnframes())
                sw, ch = w.getsampwidth(), w.getnchannels(); w.close()
            except Exception:
                continue
            if sw != 2: continue
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1: x = x.reshape(-1, ch).mean(1)
        mu = AF.mu_law_encoding(torch.from_numpy(x), 256).clamp(0, 255).numpy().astype(np.uint8)
        out.extend(mu.tobytes())
        if len(out) >= max_bytes: break
    return bytes(out[:max_bytes])


def make_set(stream, N, n):
    nw = min(n, len(stream) // N)
    feats = torch.zeros(nw, N, FEAT_DIM); ids = torch.zeros(nw, N, dtype=torch.long)
    for i in range(nw):
        b = stream[i*N:(i+1)*N]; f, _ = signal_features(b)
        feats[i] = f[:N]; ids[i] = torch.tensor(list(b))
    return feats, ids


def patch_ids(ids, P, mode):
    """[B,N] byte ids -> patch_id [B,N] in 0..P-1. dynamic=cuts at top |Δ|, fixed=uniform."""
    B, N = ids.shape
    if mode == "fixed":
        K = N // P
        pid = (torch.arange(N, device=ids.device) // K).clamp(max=P - 1)
        return pid[None].expand(B, -1).contiguous()
    d = (ids[:, 1:].float() - ids[:, :-1].float()).abs()       # |Δ sample| transition energy
    d = torch.cat([torch.zeros(B, 1, device=ids.device), d], 1)
    d[:, 0] = -1                                               # never cut at 0
    cut = torch.zeros(B, N, device=ids.device)
    topi = d.topk(P - 1, dim=1).indices                        # P-1 highest-transition positions
    cut.scatter_(1, topi, 1.0)
    pid = (cut.cumsum(1) - cut[:, :1]).clamp(0, P - 1).long()   # increment at each cut
    pid = cut.cumsum(1).clamp(max=P - 1).long()
    return pid


class BLTAudio(nn.Module):
    def __init__(self, dim, P, heads, local_layers=2, main_layers=4, dec_layers=2):
        super().__init__()
        self.P = P; self.dim = dim
        self.hsl_proj = nn.Linear(FEAT_DIM, dim)               # HSL-encoded INPUT (the proven edge)
        self.enc = nn.ModuleList([RoPECausalBlock(dim, heads) for _ in range(local_layers)])
        self.main = nn.ModuleList([RoPECausalBlock(dim, heads) for _ in range(main_layers)])
        self.byte_emb = nn.Embedding(256, dim)
        self.cond_proj = nn.Linear(dim, dim)
        self.bos = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dec = nn.ModuleList([RoPECausalBlock(dim, heads) for _ in range(dec_layers)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, 256)

    def forward(self, feats, ids, pid):
        B, N, _ = feats.shape; P = self.P
        x = self.hsl_proj(feats)
        for blk in self.enc: x = blk(x)                        # local encoder (bytes)
        # segment-mean pool by patch_id -> patch_emb [B,P,dim]
        pe = torch.zeros(B, P, self.dim, device=x.device)
        pe.scatter_add_(1, pid.unsqueeze(-1).expand(-1, -1, self.dim), x)
        cnt = torch.zeros(B, P, device=x.device).scatter_add_(1, pid, torch.ones_like(pid, dtype=torch.float))
        pe = pe / cnt.clamp(min=1).unsqueeze(-1)
        pc = pe
        for blk in self.main: pc = blk(pc)                     # MAIN transformer (patches, causal)
        # per-byte cond = prior-patch ctx (patch_id-1); patch 0 -> BOS
        prior = (pid - 1).clamp(min=0)
        cond = torch.gather(pc, 1, prior.unsqueeze(-1).expand(-1, -1, self.dim))
        cond = torch.where((pid == 0).unsqueeze(-1), self.bos.expand(B, N, -1), cond)
        h = self.byte_emb(ids) + self.cond_proj(cond)          # byte-AR input + compressed prior context
        for blk in self.dec: h = blk(h)                        # local byte decoder (causal)
        return self.head(self.norm(h))                         # [B,N,256] predict next byte


def run(mode, tr, va, P, dim, heads, steps, bs=32, lr=1e-3):
    ftr, itr = tr; fva, iva = va
    torch.manual_seed(0)
    m = BLTAudio(dim, P, heads).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=lr)
    n = ftr.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV); f, i = ftr[idx], itr[idx]
        pid = patch_ids(i, P, mode)
        lo = m(f, i, pid)
        loss = F.cross_entropy(lo[:, :-1].reshape(-1, 256), i[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pid = patch_ids(iva, P, mode); lo = m(fva, iva, pid)
        nll = F.cross_entropy(lo[:, :-1].reshape(-1, 256), iva[:, 1:].reshape(-1))
    return float(nll) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=r"C:\Users\gguni\AppData\Local\Temp\hsl_audio\**\recordings\*.wav")
    ap.add_argument("--label", default="speech")
    ap.add_argument("--N", type=int, default=256); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--mb", type=int, default=4_000_000)
    a = ap.parse_args()
    P = a.N // a.K
    stream = load_mulaw(a.glob, a.mb)
    print(f"BLT-audio[{a.label}] | {len(stream)} bytes | N={a.N}, K={a.K} -> {P} patches | dim{a.dim} | {DEV}")
    print(f"main attention: {P} patches vs {a.N} bytes (~{a.K**2}× cheaper)\n", flush=True)
    half = int(len(stream) * 0.85)
    tr = tuple(t.to(DEV) for t in make_set(stream[:half], a.N, 8000))
    va = tuple(t.to(DEV) for t in make_set(stream[half:], a.N, 1500))
    print(f"{'patching':>10} {'bits/sample':>12}")
    fx = run("fixed", tr, va, P, a.dim, a.heads, a.steps)
    print(f"{'fixed':>10} {fx:>12.3f}", flush=True)
    dy = run("dynamic", tr, va, P, a.dim, a.heads, a.steps)
    print(f"{'dynamic':>10} {dy:>12.3f}", flush=True)
    print(f"\nΔ(dynamic-fixed)={dy-fx:+.3f}  {'dynamic wins (boundaries at structure help)' if dy<fx-0.01 else ('fixed' if fx<dy-0.01 else 'tie')}")
    print(f"both: {a.K}× fewer patch positions, ~{a.K**2}× cheaper main attention.")


if __name__ == "__main__":
    main()
