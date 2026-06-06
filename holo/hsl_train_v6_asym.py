"""From-scratch training of the FINALIZED architecture AsymHSL (101M) on the new corpus.

AsymHSL is asymmetric: a DENSE bidirectional encoder reads K bytes/embedding of CONTEXT,
a byte-AR decoder generates the CONTINUATION while cross-attending to the encoded context
(+ optional RAG). So the LM objective is split per window:

   window = [ CTX_BYTES = K * n_in ]  +  [ OUT_BYTES ]
   input  feats = pack_input(ctx)            -> encoder (reads context densely)
   output feats = out_features(out)          -> decoder (byte-AR), cross-attends to context
   loss = next-byte CE over the OUT region

This actually exercises encoder + cross-attn + fast-weight (tier-2 dial) together — the plain
HSLDecoder run did not. Streaming windows from the 3.8GB FineWeb-Edu+KoWiki buffer => no overfit.
"""
from __future__ import annotations
import sys, os, json, math, time, argparse, random
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torch.nn.functional as F
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_asym import AsymHSL, pack_input, out_features, VOCAB, DEFAULT_K
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


def batch(buf, bs, K, n_in, out_len, rng):
    ctx_len = K * n_in; W = ctx_len + out_len
    inpf = torch.zeros(bs, n_in, K, FEAT_DIM); of = torch.zeros(bs, out_len, FEAT_DIM); oid = torch.zeros(bs, out_len, dtype=torch.long)
    for i in range(bs):
        s = rng.randint(0, len(buf) - W - 1)
        ctx = buf[s:s + ctx_len].tobytes(); out = buf[s + ctx_len:s + ctx_len + out_len].tobytes()
        inpf[i] = pack_input(ctx, K); of[i] = out_features(out); oid[i] = torch.tensor(list(out))
    return inpf.to(DEV), of.to(DEV), oid.to(DEV)


@torch.no_grad()
def eval_bpb(model, buf, K, n_in, out_len, n=48, bs=12):
    model.eval(); rng = random.Random(123); tot = 0.0; ntok = 0
    for _ in range(0, n, bs):
        inpf, of, oid = batch(buf, bs, K, n_in, out_len, rng)
        lo = model(inpf, of)
        ce = F.cross_entropy(lo[:, :-1].reshape(-1, VOCAB), oid[:, 1:].reshape(-1), reduction="sum")
        tot += float(ce); ntok += oid[:, 1:].numel()
    model.train(); return (tot / ntok) / math.log(2)


def lr_at(step, warm, total, base, lo):
    if step < warm: return base * step / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    return lo + 0.5 * (base - lo) * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v6_asym_101m")
    ap.add_argument("--train", default=r"C:\Users\gguni\holo_v6_data\train.jsonl")
    ap.add_argument("--val", default=r"C:\Users\gguni\holo_v6_data\val.jsonl")
    ap.add_argument("--max-gb", type=float, default=2.0); ap.add_argument("--val-gb", type=float, default=0.1)
    ap.add_argument("--dim", type=int, default=512); ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=12); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--K", type=int, default=DEFAULT_K); ap.add_argument("--n-in", type=int, default=64)
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=16000); ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=12); ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1); ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=606)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>/latest.pt")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    ckpt = HERE / "checkpoints" / args.run_name; logd = HERE / "logs" / args.run_name
    ckpt.mkdir(parents=True, exist_ok=True); logd.mkdir(parents=True, exist_ok=True)
    lf = (logd / "train.log").open("a", encoding="utf-8")
    def log(m): print(m, flush=True); lf.write(m + "\n"); lf.flush()
    (logd / "config.json").write_text(json.dumps({**vars(args), "license": LICENSE, "feat_dim": FEAT_DIM, "arch": "AsymHSL"}, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"=== AsymHSL v6 FROM-SCRATCH (101M finalized arch) === {args.run_name} === RESEARCH-ONLY ===")
    log(f"obj: dense CTX={args.K*args.n_in}B (enc) -> AR OUT={args.out_len}B (dec+cross-attn) | {DEV}")
    tr = load_buffer(args.train, args.max_gb, log); va = load_buffer(args.val, args.val_gb, log)
    model = AsymHSL(dim=args.dim, enc_layers=args.enc_layers, dec_layers=args.dec_layers, heads=args.heads, K=args.K).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    use_bf16 = DEV.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV.type == "cuda" and not use_bf16))
    start = 0
    if args.resume and (ckpt / "latest.pt").exists():
        rc = torch.load(ckpt / "latest.pt", map_location=DEV)
        model.load_state_dict(rc["model"])
        try: opt.load_state_dict(rc["opt"])
        except Exception as e: log(f"(opt state not restored: {e})")
        start = int(rc.get("step", 0)); log(f"RESUMED from step {start} -> {args.steps} (warm-restart cosine)")
    log(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M | amp {'bf16' if use_bf16 else 'fp16'} | bs {args.batch_size}x{args.accum} | enc {args.enc_layers} dec {args.dec_layers}\n")
    rng = random.Random(args.seed + start); metrics = (logd / "metrics.jsonl").open("a", encoding="utf-8"); t0 = time.time()
    for step in range(start + 1, args.steps + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step, args.warmup, args.steps, args.lr, args.min_lr)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            inpf, of, oid = batch(tr, args.batch_size, args.K, args.n_in, args.out_len, rng)
            with torch.autocast("cuda", dtype=amp, enabled=(DEV.type == "cuda")):
                lo = model(inpf, of)
                loss = F.cross_entropy(lo[:, :-1].reshape(-1, VOCAB), oid[:, 1:].reshape(-1)) / args.accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt); scaler.update()
        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            vb = eval_bpb(model, va, args.K, args.n_in, args.out_len); tb = float(loss) * args.accum / math.log(2)
            gates = [round(float(torch.tanh(b.gate)), 3) for b in model.mem_blocks]
            row = {"step": step, "lr": round(opt.param_groups[0]["lr"], 6), "train_bpb": round(tb, 4),
                   "val_bpb": round(vb, 4), "gate_absmax": round(max(abs(g) for g in gates), 3), "min": round((time.time()-t0)/60, 1)}
            metrics.write(json.dumps(row) + "\n"); metrics.flush(); log(json.dumps(row, ensure_ascii=False))
        if step % args.save_every == 0 or step == args.steps:
            tmp = ckpt / "latest.tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "config": vars(args), "license": LICENSE}, tmp); tmp.replace(ckpt / "latest.pt")
            torch.save({"model": model.state_dict(), "step": step, "config": vars(args)}, ckpt / f"checkpoint_step_{step:06d}.pt")
    log(f"\ndone. {ckpt}"); metrics.close(); lf.close()


if __name__ == "__main__":
    main()
