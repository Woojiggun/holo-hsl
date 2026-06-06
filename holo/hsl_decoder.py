"""HSL generative decoder — RoPE causal transformer, byte vocab + EOS/MASK.

Closes spec gaps:
  - RoPE positions (hsl_rope) -> length-extrapolatable; sliding-window + streaming gen.
  - vocab 258 = 256 bytes + MASK(256) + EOS(257); EOS lets generation terminate.
Input per position: HSL signal features (FEAT_DIM-wide; recipe withheld — see hsl_signal_encoder)
for byte tokens; a learned embedding for the special tokens. THE TRANSFORMER LIVES HERE.

Generation:
  - generate()        : sliding-window (recompute) — correct for ANY length (streaming need).
  - generate_stream() : KV-cache incremental — true streaming.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
import torch.nn as nn

from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_rope import RoPECausalBlock

VOCAB = 258
MASK_ID = 256
EOS_ID = 257


class HSLDecoder(nn.Module):
    def __init__(self, dim=512, layers=12, heads=8, dropout=0.1, feat_dim=FEAT_DIM, rope_base=10000.0):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(feat_dim, dim)          # HSL features -> model dim
        self.special_emb = nn.Embedding(2, dim)             # [MASK, EOS]
        self.blocks = nn.ModuleList([RoPECausalBlock(dim, heads, dropout, rope_base) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB)

    def embed(self, feats, tok):
        x = self.input_proj(feats)                          # [B,L,d]
        is_special = tok >= 256
        if bool(is_special.any()):
            sp = self.special_emb((tok - 256).clamp(min=0, max=1))
            x = torch.where(is_special.unsqueeze(-1), sp, x)
        return x

    def forward(self, feats, tok, mask, kv_caches=None, pos_offset=0):
        x = self.embed(feats, tok)
        kpm = (mask == 0)
        for i, blk in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = blk(x, key_padding_mask=kpm, kv_cache=cache, pos_offset=pos_offset)
        return self.head(self.norm(x))                      # [B,L,VOCAB]

    # ---- generation -------------------------------------------------------
    def _feats_tok(self, ctx: bytes, device):
        f, _p = signal_features(ctx)
        L = f.shape[0]
        feats = f.unsqueeze(0).to(device)
        tok = torch.tensor([list(ctx)], dtype=torch.long, device=device)
        mask = torch.ones(1, L, device=device)
        return feats, tok, mask

    @staticmethod
    def _pick(logits, temperature):
        if temperature <= 0:
            return int(logits.argmax())
        p = (logits / temperature).softmax(-1)
        return int(torch.multinomial(p, 1))

    @torch.no_grad()
    def generate(self, seed: bytes, n_new: int, window: int, device, temperature: float = 0.0):
        """Sliding-window autoregressive. Correct for arbitrary length (window slides)."""
        self.eval()
        out = bytearray(seed) if seed else bytearray(b"\x00")
        for _ in range(n_new):
            ctx = bytes(out[-window:])
            feats, tok, mask = self._feats_tok(ctx, device)
            nxt = self._pick(self.forward(feats, tok, mask)[0, -1], temperature)
            if nxt >= 256:                                  # EOS / MASK -> stop
                break
            out.append(nxt)
        return bytes(out)

    @torch.no_grad()
    def generate_stream(self, seed: bytes, n_new: int, device, temperature: float = 0.0):
        """KV-cache incremental streaming. New-byte Δ feature uses the last 2 bytes."""
        self.eval()
        kv = [{"k": None, "v": None} for _ in self.blocks]
        out = bytearray(seed) if seed else bytearray(b"\x00")
        feats, tok, mask = self._feats_tok(bytes(out), device)
        logits = self.forward(feats, tok, mask, kv_caches=kv, pos_offset=0)
        pos = len(out)
        last = logits[0, -1]
        for _ in range(n_new):
            nxt = self._pick(last, temperature)
            if nxt >= 256:
                break
            out.append(nxt)
            tail = bytes(out[-2:])                          # Δ needs the previous byte
            f, _p = signal_features(tail)
            feats = f[-1:].unsqueeze(0).to(device)          # [1,1,29]
            tk = torch.tensor([[nxt]], dtype=torch.long, device=device)
            mk = torch.ones(1, 1, device=device)
            logits = self.forward(feats, tk, mk, kv_caches=kv, pos_offset=pos)
            pos += 1
            last = logits[0, -1]
        return bytes(out)
