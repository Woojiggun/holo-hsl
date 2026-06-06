"""Asymmetric HSL — the finalized architecture.

DESYNCHRONIZED input/output (the key decision):
  INPUT  : DENSE. K bytes packed per embedding (K=8) -> big window read cheaply.
           bidirectional RoPE encoder -> input memory.
  OUTPUT : BYTE-by-byte autoregressive (1 byte/step) -> generation quality kept.
  BRIDGE : the byte decoder CROSS-ATTENDS to (input memory ⊕ retrieved RAG memory),
           so a small model pulls in needed info on demand instead of memorizing.

Window math: input memory = N_in patches × K bytes. N_in=2048, K=8 -> ~16 KB window.
Anything beyond the window -> RAG: retrieved chunks are encoded and concatenated to
memory, and the decoder cross-attends to them (the `retrieved` argument).

Core dims confirmed: dim 512 / heads 8 / RoPE / vocab 258 (256 bytes + MASK + EOS).
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
import torch.nn as nn

from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_rope import RoPECausalBlock
from hsl_fastweight import FastWeightMemory

VOCAB = 258
MASK_ID = 256
EOS_ID = 257
DEFAULT_K = 8                                    # bytes per INPUT embedding (dense)


class CrossAttnBlock(nn.Module):
    """Decoder cross-attention: byte queries attend to (input ⊕ retrieved) memory."""
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.h = heads; self.dk = dim // heads
        self.nq = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim); self.kv = nn.Linear(dim, dim * 2); self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, mem_key_mask=None):
        B, Lx, d = x.shape; Lm = memory.shape[1]
        h = self.nq(x)
        q = self.q(h).view(B, Lx, self.h, self.dk).transpose(1, 2)              # [B,h,Lx,dk]
        kv = self.kv(memory).view(B, Lm, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                     # [B,h,Lm,dk]
        scores = (q @ k.transpose(-1, -2)) / (self.dk ** 0.5)                   # [B,h,Lx,Lm]
        if mem_key_mask is not None:
            scores = scores + mem_key_mask[:, None, None, :]                    # [B,1,1,Lm]
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, Lx, d)
        x = x + self.drop(self.proj(out))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class AsymHSL(nn.Module):
    def __init__(self, dim=512, enc_layers=4, dec_layers=12, heads=8, K=DEFAULT_K, dropout=0.1):
        super().__init__()
        self.K = K
        # dense INPUT encoder (bidirectional)
        self.in_proj = nn.Linear(K * FEAT_DIM, dim)
        self.enc = nn.ModuleList([RoPECausalBlock(dim, heads, dropout, causal=False) for _ in range(enc_layers)])
        self.enc_norm = nn.LayerNorm(dim)
        # byte-AR DECODER (causal self-attn + cross-attn to memory)
        self.out_proj = nn.Linear(FEAT_DIM, dim)
        self.self_blocks = nn.ModuleList([RoPECausalBlock(dim, heads, dropout, causal=True) for _ in range(dec_layers)])
        self.mem_blocks = nn.ModuleList([FastWeightMemory(dim, heads) for _ in range(dec_layers)])   # tier-2 (gated dial, 0=off)
        self.cross_blocks = nn.ModuleList([CrossAttnBlock(dim, heads, dropout) for _ in range(dec_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB)

    def encode_input(self, inp_feats, inp_mask=None):
        """inp_feats [B, N_in, K, FEAT_DIM] -> memory [B, N_in, dim] (dense, bidirectional)."""
        B, N, K, F = inp_feats.shape
        x = self.in_proj(inp_feats.reshape(B, N, K * F))
        kpm = (inp_mask == 0) if inp_mask is not None else None
        for blk in self.enc:
            x = blk(x, key_padding_mask=kpm)
        return self.enc_norm(x)

    def forward(self, inp_feats, out_feats, retrieved=None, inp_mask=None, out_mask=None):
        """inp_feats [B,N_in,K,F] dense input; out_feats [B,L,F] output bytes (teacher-forced).
        retrieved [B,M,dim] optional RAG memory concatenated to input memory."""
        memory = self.encode_input(inp_feats, inp_mask)          # [B,N_in,dim]
        if retrieved is not None:
            memory = torch.cat([memory, retrieved], dim=1)
        x = self.out_proj(out_feats)                             # [B,L,dim]
        kpm = (out_mask == 0) if out_mask is not None else None
        for s, mem, c in zip(self.self_blocks, self.mem_blocks, self.cross_blocks):
            x = s(x, key_padding_mask=kpm)                       # tier-1: window self-attn (byte AR)
            x = mem(x)                                           # tier-2: fast-weight recall (gated dial, 0=off)
            x = c(x, memory)                                     # tier-3: cross-attn (input ⊕ RAG)
        return self.head(self.norm(x))                           # [B,L,VOCAB]

    @torch.no_grad()
    def generate(self, seed: bytes, n_new: int, window: int = 256, device: str = "cpu", temperature: float = 0.8, top_k: int = 40):
        self.eval()
        inp_feats = pack_input(seed, self.K).unsqueeze(0).to(device)
        out_bytes = bytearray()
        
        for _ in range(n_new):
            cur_bytes = bytes(out_bytes[-window:]) if out_bytes else b"\x00"
            of = out_features(cur_bytes).unsqueeze(0).to(device)
            logits = self(inp_feats, of)
            
            last_logits = logits[0, -1]
            if temperature <= 0:
                next_token = int(torch.argmax(last_logits).item())
            else:
                last_logits = last_logits / max(temperature, 1e-6)
                if 0 < top_k < last_logits.numel():
                    v, i = torch.topk(last_logits, k=top_k)
                    probs = torch.softmax(v, dim=-1)
                    next_token = int(i[torch.multinomial(probs, num_samples=1)].item())
                else:
                    probs = torch.softmax(last_logits, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())
                    
            if next_token > 255:
                break
            out_bytes.append(next_token)
            
        return seed + bytes(out_bytes)



# ----------------------- helpers -----------------------
def pack_input(data: bytes, K: int = DEFAULT_K):
    """bytes -> dense input feats [N_in, K, FEAT_DIM] (truncate to multiple of K)."""
    n = (len(data) // K) * K
    if n == 0:
        data = (data + b"\x00" * K)[:K]; n = K
    f, _p = signal_features(data[:n])                            # [n, FEAT_DIM]
    return f.reshape(n // K, K, FEAT_DIM)


def out_features(data: bytes):
    f, _p = signal_features(data if data else b"\x00")
    return f                                                     # [L, FEAT_DIM]


if __name__ == "__main__":
    # smoke: runs end-to-end + loss drops
    import torch.nn.functional as F, random
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(0); torch.manual_seed(0)
    K, IN, OUT, B = 8, 128, 64, 16
    base = ("Holo HSL 비대칭 입출력 테스트. dense input, byte output, cross-attention RAG. " * 40).encode("utf-8")
    def batch():
        inpf = torch.zeros(B, IN // K, K, FEAT_DIM); of = torch.zeros(B, OUT, FEAT_DIM); oid = torch.zeros(B, OUT, dtype=torch.long)
        for i in range(B):
            s = rng.randrange(0, len(base) - (IN + OUT) - 1)
            inpf[i] = pack_input(base[s:s+IN], K)
            seg = base[s+IN:s+IN+OUT]; of[i] = out_features(seg); oid[i] = torch.tensor(list(seg))
        return inpf.to(dev), of.to(dev), oid.to(dev)
    m = AsymHSL(dim=128, enc_layers=2, dec_layers=2, heads=4, K=K).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    inpf, of, oid = batch()
    print(f"device={dev} | params={sum(p.numel() for p in m.parameters())/1e6:.2f}M | "
          f"input window={IN}B ({IN//K} patches × {K}B) | output={OUT}B byte-AR")
    first = None
    for step in range(120):
        logits = m(inpf, of)                                     # [B,OUT,VOCAB]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), oid[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if first is None: first = float(loss)
    print(f"forward logits {tuple(logits.shape)} | loss {first:.3f} -> {float(loss):.3f}")
    # RAG hook: cross-attend to extra retrieved memory
    retr = torch.randn(B, 5, 128, device=dev)
    lo2 = m(inpf, of, retrieved=retr)
    print(f"with retrieved RAG memory [{tuple(retr.shape)}] -> logits {tuple(lo2.shape)}  (cross-attn injection OK)")
    gates = [round(float(torch.tanh(b.gate)), 3) for b in m.mem_blocks]
    print(f"tier-2 fast-weight gates (dial; 0=off at init, opens iff it pays): {gates}")
    print("ASYM SMOKE: GREEN" if float(loss) < first else "CHECK")
