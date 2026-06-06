"""RoPE (rotary position) causal attention — length-extrapolatable, streaming-ready.

Why RoPE: positions are encoded as rotations of Q/K, so the model is NOT tied to a
fixed max_len (unlike a learned absolute pos table). This enables:
  - sliding-window generation past the trained length,
  - chunked / streaming inference (KV-cache appends with a running position offset).

Block = prenorm -> RoPE causal multi-head attention -> prenorm -> GELU FFN (residual).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_tables(seq_len: int, head_dim: int, base: float = 10000.0,
                device=None, dtype=torch.float32, offset: int = 0):
    """cos/sin of shape [seq_len, head_dim]. `offset` shifts positions (KV-cache)."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # [L, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)                # [L, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B,H,L,hd]; cos,sin: [L,hd]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q2 = q * cos + _rotate_half(q) * sin
    k2 = k * cos + _rotate_half(k) * sin
    return q2, k2


class RoPECausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1, base: float = 10000.0,
                 causal: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.h = heads; self.dk = dim // heads; self.base = base; self.causal = causal
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, kv_cache=None, pos_offset=0):
        """x [B,L,d]. key_padding_mask [B,L] (True=pad). kv_cache: optional dict for streaming."""
        B, L, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # [B,h,L,dk]
        cos, sin = rope_tables(L, self.dk, self.base, x.device, q.dtype, offset=pos_offset)
        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None and kv_cache.get("k") is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)       # prepend cached keys/values
            v = torch.cat([kv_cache["v"], v], dim=2)
        if kv_cache is not None:
            kv_cache["k"], kv_cache["v"] = k, v
        Lk = k.shape[2]

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)   # [B,h,L,Lk]
        if self.causal:                                            # encoder uses causal=False (bidirectional)
            qpos = torch.arange(L, device=x.device).unsqueeze(1) + (Lk - L)
            kpos = torch.arange(Lk, device=x.device).unsqueeze(0)
            scores = scores.masked_fill((kpos > qpos)[None, None], float("-inf"))
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, -Lk:]
            scores = scores.masked_fill(pad, float("-inf"))
        attn = scores.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, d)
        x = x + self.drop(self.proj(out))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
