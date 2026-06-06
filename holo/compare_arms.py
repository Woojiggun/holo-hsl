"""TO-VALIDATE #1 (spec): does AsymHSL (dense-input + byte-AR) match/beat the 1:1 byte LM?

Fair head-to-head on IDENTICAL held-out target bytes:
  - both predict the same TARGET (256 bytes) given preceding context + preceding target bytes
  - 1:1 HSLDecoder conditions on CTX256 (256 immediate bytes, its trained window)
  - AsymHSL conditions on CTX512 (512 bytes packed DENSE into 64 enc positions = 8x) via cross-attn
=> tests whether asym's bigger CHEAP context (same attn budget) lowers bits/byte vs 1:1.
Lower bits/byte = better. No superiority claim beyond the measured number.
"""
from __future__ import annotations
import sys, pathlib, json, math, random, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torch.nn.functional as F
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_decoder import HSLDecoder, VOCAB
from hsl_asym import AsymHSL, pack_input, out_features
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL = r"C:\Users\gguni\holo_v6_data\val.jsonl"
CTX1, CTX2, TGT = 256, 512, 256                  # 1:1 context, asym context, target len


def load_buffer(path, gb):
    cap = int(gb * 1e9); buf = bytearray()
    for line in open(path, encoding="utf-8", errors="replace"):
        try: t = json.loads(line).get("text") or ""
        except json.JSONDecodeError: continue
        buf.extend(t.encode("utf-8", "replace")); buf.append(10)
        if len(buf) >= cap: break
    return np.frombuffer(bytes(buf), dtype=np.uint8)


def load_decoder(ckpt):
    ck = torch.load(ckpt, map_location=DEV); cfg = ck.get("config", {})
    m = HSLDecoder(cfg.get("dim", 512), cfg.get("layers", 12), cfg.get("heads", 8)).to(DEV).eval()
    m.load_state_dict(ck["model"] if "model" in ck else ck); return m, ck.get("step", "?")


def load_asym(ckpt):
    ck = torch.load(ckpt, map_location=DEV); cfg = ck.get("config", {})
    m = AsymHSL(dim=cfg.get("dim", 512), enc_layers=cfg.get("enc_layers", 4), dec_layers=cfg.get("dec_layers", 12),
                heads=cfg.get("heads", 8), K=cfg.get("K", 8)).to(DEV).eval()
    m.load_state_dict(ck["model"] if "model" in ck else ck); return m, ck.get("step", "?"), cfg.get("K", 8)


@torch.no_grad()
def bpb_decoder(model, buf, wins):
    tot = 0.0; ntok = 0
    for s in wins:
        seg = buf[s + CTX2 - CTX1: s + CTX2 + TGT]        # [CTX256 + TGT] = 512 bytes
        f, _ = signal_features(seg.tobytes())
        feats = f.unsqueeze(0).to(DEV); tok = torch.tensor([seg.tolist()], dtype=torch.long, device=DEV); mask = torch.ones(1, len(seg), device=DEV)
        lo = model(feats, tok, mask)
        tgt = tok[0, CTX1:CTX1+TGT]; pred = lo[0, CTX1-1:CTX1+TGT-1]
        tot += float(F.cross_entropy(pred, tgt, reduction="sum")); ntok += tgt.numel()
    return (tot / ntok) / math.log(2)


@torch.no_grad()
def bpb_asym(model, K, buf, wins):
    tot = 0.0; ntok = 0
    for s in wins:
        ctx = buf[s: s + CTX2].tobytes(); tgt = buf[s + CTX2: s + CTX2 + TGT].tobytes()
        inpf = pack_input(ctx, K).unsqueeze(0).to(DEV); of = out_features(tgt).unsqueeze(0).to(DEV)
        oid = torch.tensor([list(tgt)], device=DEV)
        lo = model(inpf, of)
        tot += float(F.cross_entropy(lo[0, :-1], oid[0, 1:], reduction="sum")); ntok += oid[0, 1:].numel()
    return (tot / ntok) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(HERE / "checkpoints" / "v6_scratch_512x12" / "latest.pt"))
    ap.add_argument("--asym", default=str(HERE / "checkpoints" / "v6_asym_101m" / "latest.pt"))
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--val-gb", type=float, default=0.08)
    a = ap.parse_args()
    buf = load_buffer(VAL, a.val_gb); rng = random.Random(0)
    wins = [rng.randint(0, len(buf) - CTX2 - TGT - 2) for _ in range(a.n)]
    print(f"compare on {a.n} identical held-out targets ({TGT}B each) | {DEV}\n")
    dec, ds = load_decoder(a.baseline); b1 = bpb_decoder(dec, buf, wins)
    print(f"  1:1 HSLDecoder (38M, step {ds})   ctx={CTX1}B   bits/byte = {b1:.4f}")
    del dec; torch.cuda.empty_cache()
    asy, as_, K = load_asym(a.asym); b2 = bpb_asym(asy, K, buf, wins)
    print(f"  AsymHSL (101M, step {as_})        ctx={CTX2}B(dense)  bits/byte = {b2:.4f}")
    print(f"\n  Δ(asym - 1:1) = {b2-b1:+.4f} bits/byte  =>  ", end="")
    print("asym BEATS 1:1 (bigger cheap context pays)" if b2 < b1 else
          "asym does NOT beat 1:1 here (honest; 1:1 remains the validated baseline per spec)")


if __name__ == "__main__":
    main()
