"""Step 3 — curvature READ-ONLY probe on a TRAINED checkpoint (no forcing, lens only).

geo_lm.py already proved forcing geometry HURTS (lambda=0 best). So here we do NOT add any
loss — we just LOOK at the universe the model already built, layer by layer, and report where
its self-organized geometry is richest.

Per layer we measure, over byte-value reps (mean hidden state per byte 0..255):
  - hub axis : Pearson corr( log-frequency , radius=||centered rep|| ).  (sign = which way hubs go)
  - class geometry : how well the 4 natural byte-classes (ascii-letter / digit / punct&space /
                     utf8-continuation 0x80-0xFF) separate ANGULARLY (cosine silhouette).
The "richest" depth = strongest |hub corr| and/or class separation. Pure observation.
"""
from __future__ import annotations
import sys, pathlib, json, math, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_decoder import HSLDecoder, EOS_ID, VOCAB
from hsl_asym import AsymHSL, pack_input
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POC_VAL = r"G:\내 드라이브\홀로빗 POC\data\val.jsonl"


def byte_class(b):
    if 48 <= b <= 57: return 1           # digit
    if (65 <= b <= 90) or (97 <= b <= 122): return 0   # ascii letter
    if b >= 128: return 3                # utf8 continuation / lead (Korean etc.)
    return 2                             # punct / space / control


def load_stream(path, n_docs, max_bytes):
    buf = bytearray()
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n_docs: break
            try: d = json.loads(line)
            except json.JSONDecodeError: continue
            t = d.get("text") or d.get("output") or d.get("input") or ""
            buf.extend(t.encode("utf-8", "replace")); buf.append(EOS_ID & 0xFF)
            if len(buf) >= max_bytes: break
    return bytes(buf[:max_bytes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "hsl_lm_main_512" / "checkpoint_step_020000.pt"))
    ap.add_argument("--max-bytes", type=int, default=120_000)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--arch", choices=["decoder", "asym"], default="decoder")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEV)
    cfg = ck.get("config", {})
    dim, heads = cfg.get("dim", 512), cfg.get("heads", 8)
    if args.arch == "asym":
        layers = cfg.get("dec_layers", 12)
        model = AsymHSL(dim=dim, enc_layers=cfg.get("enc_layers", 4), dec_layers=layers,
                        heads=heads, K=cfg.get("K", 8)).to(DEV).eval()
        blocks = model.self_blocks                            # the decoder's byte-AR self-attn stack
        K = cfg.get("K", 8)
    else:
        layers = cfg.get("layers", 12)
        model = HSLDecoder(dim, layers, heads).to(DEV).eval()
        blocks = model.blocks
        K = None
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd)
    print(f"loaded {pathlib.Path(args.ckpt).name}  arch={args.arch} dim{dim}/L{layers}/h{heads}  step={ck.get('step','?')}")

    acts = {}
    def mk(i):
        def hook(_m, _inp, out):
            acts[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook
    for i, blk in enumerate(blocks):
        blk.register_forward_hook(mk(i))

    def run_chunk(feats, tok, mask, seg_bytes):
        """run one chunk; return (emb_rep[L,dim], fills acts). handles both archs."""
        if args.arch == "asym":
            # pad seg to a multiple of K for dense input packing; output = the seg bytes
            inpf = pack_input(seg_bytes, K).unsqueeze(0).to(DEV)      # [1,N_in,K,F]
            emb = model.out_proj(feats)                              # [1,L,dim] per-byte output embedding
            _ = model(inpf, feats)                                   # triggers self_blocks hooks
            return emb
        emb = model.embed(feats, tok)
        _ = model(feats, tok, mask)
        return emb

    data = load_stream(POC_VAL, 4000, args.max_bytes)
    by = np.frombuffer(data, dtype=np.uint8)
    L = args.seq
    nchunk = len(by) // L

    # accumulate sum + count per byte value, per layer (+ embedding layer = -1)
    sum_rep = {l: np.zeros((256, dim), np.float64) for l in range(-1, layers)}
    cnt = np.zeros(256, np.int64)
    with torch.no_grad():
        for c in range(nchunk):
            seg = bytes(by[c*L:(c+1)*L])
            f, _ = signal_features(seg)
            feats = f.unsqueeze(0).to(DEV)
            tok = torch.tensor([list(seg)], dtype=torch.long, device=DEV)
            mask = torch.ones(1, L, device=DEV)
            emb = run_chunk(feats, tok, mask, seg)
            ids = np.frombuffer(seg, np.uint8)
            for l in range(-1, layers):
                rep = (emb if l == -1 else acts[l])[0].float().cpu().numpy()
                np.add.at(sum_rep[l], ids, rep)
            np.add.at(cnt, ids, 1)
    seen = cnt > 0
    freq = cnt[seen] / cnt.sum()
    logf = np.log(freq)
    cls = np.array([byte_class(b) for b in range(256)])[seen]

    print(f"\nbytes seen={seen.sum()}  chunks={nchunk}  ({len(by)} bytes)\n")
    print(f"{'layer':>6} | {'hub corr(logfreq,radius)':>26} | {'angular class silhouette':>26}")
    print("-" * 66)
    best = (None, -1)
    for l in range(-1, layers):
        mean = sum_rep[l][seen] / cnt[seen][:, None]
        c = mean - mean.mean(0, keepdims=True)          # center
        radius = np.linalg.norm(c, axis=1)
        hub = float(np.corrcoef(logf, radius)[0, 1])
        # angular class silhouette: cosine sim, mean(intra)-mean(inter) per class, normalized
        u = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
        cos = u @ u.T
        sil = []
        for k in range(u.shape[0]):
            same = cls == cls[k]; same[k] = False
            other = cls != cls[k]
            if same.sum() == 0 or other.sum() == 0: continue
            sil.append(cos[k][same].mean() - cos[k][other].mean())
        silh = float(np.mean(sil)) if sil else 0.0
        tag = "emb" if l == -1 else f"L{l}"
        score = abs(hub) + max(0.0, silh)
        if score > best[1]: best = (tag, score)
        print(f"{tag:>6} | {hub:>+26.3f} | {silh:>+26.3f}")
    print("-" * 66)
    print(f"richest self-built geometry: {best[0]}  (|hub|+class-sep = {best[1]:.3f})")
    print("\nNOTE: read-only. Positive hub corr = frequent bytes to RIM (model's own grid, non-human).")
    print("No forcing (geo_lm: forcing hurts). This is the lens on the universe it already built.")


if __name__ == "__main__":
    main()
