"""From-scratch byte-LM training on the NEW validated corpus (streaming features).

Why new: the old POC chat data overfit past ~10k steps. This trains the canonical
512/12/8 HSL decoder FROM SCRATCH on FineWeb-Edu(EN)+Korean-Wikipedia (3.8GB), sampling
random windows so the model effectively never repeats data (no overfit from tiny corpora).

Streaming, NOT precomputed: each step samples random byte windows from one big in-RAM
buffer and computes signal_features on the fly (the old precompute-all pipeline would need
hundreds of GB for this corpus). Δ is per-window (signal_features uses window-origin), so
windows are self-contained.
"""
from __future__ import annotations
import sys, os, json, math, time, argparse, random
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torch.nn.functional as F
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_decoder import HSLDecoder, VOCAB
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LICENSE = {"en": "FineWeb-Edu ODC-By", "ko": "Korean Wikipedia CC BY-SA", "use": "research-only, no redistribution"}


def load_buffer(path, max_gb, log):
    cap = int(max_gb * 1e9); buf = bytearray()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try: t = json.loads(line).get("text") or ""
            except json.JSONDecodeError: continue
            buf.extend(t.encode("utf-8", "replace")); buf.append(10)
            if len(buf) >= cap: break
    log(f"  loaded {len(buf)/1e9:.2f}GB from {os.path.basename(path)}")
    return np.frombuffer(bytes(buf), dtype=np.uint8)


def batch_feats(buf, bs, seq, rng):
    """sample bs random windows of seq+1 bytes -> feats[bs,seq,F], tok[bs,seq], tgt[bs,seq]."""
    feats = torch.zeros(bs, seq, FEAT_DIM); tok = torch.zeros(bs, seq, dtype=torch.long); tgt = torch.zeros(bs, seq, dtype=torch.long)
    for i in range(bs):
        s = rng.randint(0, len(buf) - seq - 2)
        win = buf[s:s + seq + 1]
        f, _ = signal_features(win[:seq].tobytes())
        feats[i] = f; tok[i] = torch.from_numpy(win[:seq].astype(np.int64)); tgt[i] = torch.from_numpy(win[1:seq+1].astype(np.int64))
    return feats.to(DEV), tok.to(DEV), tgt.to(DEV)


@torch.no_grad()
def eval_bpb(model, buf, seq, n=64, bs=16):
    model.eval(); rng = random.Random(123); tot = 0.0; ntok = 0
    for s in range(0, n, bs):
        feats, tok, tgt = batch_feats(buf, bs, seq, rng)
        lo = model(feats, tok, torch.ones(bs, seq, device=DEV))
        ce = F.cross_entropy(lo.reshape(-1, VOCAB), tgt.reshape(-1), reduction="sum")
        tot += float(ce); ntok += tgt.numel()
    model.train(); return (tot / ntok) / math.log(2)


def lr_at(step, warm, total, base, lo):
    if step < warm: return base * step / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    return lo + 0.5 * (base - lo) * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v6_scratch_512x12")
    ap.add_argument("--train", default=r"C:\Users\gguni\holo_v6_data\train.jsonl")
    ap.add_argument("--val", default=r"C:\Users\gguni\holo_v6_data\val.jsonl")
    ap.add_argument("--max-gb", type=float, default=2.0)        # RAM-safe slice, sampled randomly
    ap.add_argument("--val-gb", type=float, default=0.1)
    ap.add_argument("--dim", type=int, default=512); ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=24); ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1); ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=606)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    ckpt = HERE / "checkpoints" / args.run_name; logd = HERE / "logs" / args.run_name
    ckpt.mkdir(parents=True, exist_ok=True); logd.mkdir(parents=True, exist_ok=True)
    lf = (logd / "train.log").open("a", encoding="utf-8")
    def log(m): print(m, flush=True); lf.write(m + "\n"); lf.flush()
    (logd / "config.json").write_text(json.dumps({**vars(args), "license": LICENSE, "feat_dim": FEAT_DIM}, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"=== HSL v6 FROM-SCRATCH === {args.run_name} === RESEARCH-ONLY ===")
    log(f"corpus: FineWeb-Edu(EN)+KoWiki | streaming windows | {DEV}")
    tr = load_buffer(args.train, args.max_gb, log); va = load_buffer(args.val, args.val_gb, log)
    model = HSLDecoder(args.dim, args.layers, args.heads).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    use_bf16 = DEV.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV.type == "cuda" and not use_bf16))
    log(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M | amp {'bf16' if use_bf16 else 'fp16'} | bs {args.batch_size}x{args.accum} seq {args.seq}\n")
    rng = random.Random(args.seed); metrics = (logd / "metrics.jsonl").open("a", encoding="utf-8"); t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step, args.warmup, args.steps, args.lr, args.min_lr)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            feats, tok, tgt = batch_feats(tr, args.batch_size, args.seq, rng)
            with torch.autocast("cuda", dtype=amp, enabled=(DEV.type == "cuda")):
                lo = model(feats, tok, torch.ones(args.batch_size, args.seq, device=DEV))
                loss = F.cross_entropy(lo.reshape(-1, VOCAB), tgt.reshape(-1)) / args.accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt); scaler.update()
        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            vb = eval_bpb(model, va, args.seq); tb = float(loss) * args.accum / math.log(2)
            row = {"step": step, "lr": round(opt.param_groups[0]["lr"], 6), "train_bpb": round(tb, 4),
                   "val_bpb": round(vb, 4), "min": round((time.time()-t0)/60, 1)}
            metrics.write(json.dumps(row) + "\n"); metrics.flush(); log(json.dumps(row, ensure_ascii=False))
        if step % args.save_every == 0 or step == args.steps:
            tmp = ckpt / "latest.tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "config": vars(args), "license": LICENSE}, tmp); tmp.replace(ckpt / "latest.pt")
            torch.save({"model": model.state_dict(), "step": step, "config": vars(args)}, ckpt / f"checkpoint_step_{step:06d}.pt")
    log(f"\ndone. {ckpt}"); metrics.close(); lf.close()


if __name__ == "__main__":
    main()
