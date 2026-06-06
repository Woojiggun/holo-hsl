"""Unified MULTIMODAL training on AsymHSL (the POC architecture — encoder-decoder, NOT 1:1).

One byte model, mixed batches:
  - TEXT  : dense CTX (enc) -> AR text continuation (dec+cross-attn)          [language]
  - VIDEO : dense window_t (enc) -> AR window_{t+1} = [frame|audio|caption]   [perception + video world-model]
Same AsymHSL.forward(inpf, of) for both; batches alternate by --p-video. The model thus learns to
GENERATE text AND tri-modal video windows (pixels + mu-law audio + caption) — byte-native, no tokenizer.
Streaming text windows (no overfit on text); video windows oversampled (small corpus). Long run OK.
Resume supported. Encoder stays AsymHSL throughout (no 1:1 confusion).
"""
from __future__ import annotations
import sys, os, json, math, time, argparse, random, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torch.nn.functional as F
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_asym import AsymHSL, pack_input, out_features, VOCAB, DEFAULT_K, EOS_ID
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LICENSE = {"text": "FineWeb-Edu ODC-By + KoWiki CC BY-SA", "video": "Prelinger public-domain", "use": "research-only"}


def load_text(path, max_gb, log):
    """token stream (int16) with EOS_ID(257) closing each doc => the model learns the '1' (closure)."""
    cap = int(max_gb*1e9); parts = []; n = 0; eos = np.array([EOS_ID], np.int16)
    for line in open(path, encoding="utf-8", errors="replace"):
        try: t = json.loads(line).get("text") or ""
        except json.JSONDecodeError: continue
        b = np.frombuffer(t.encode("utf-8", "replace"), np.uint8).astype(np.int16)
        parts.append(b); parts.append(eos); n += len(b) + 1
        if n >= cap: break
    toks = np.concatenate(parts) if parts else np.zeros(2, np.int16)
    log(f"  text: {len(toks)/1e9:.2f}G tokens (EOS-closed docs)"); return toks


def feats_for_tokens(toks):
    """[L,FEAT_DIM] features; byte-runs get signal_features, special tokens (>=256) get 0."""
    L = len(toks); f = torch.zeros(L, FEAT_DIM); i = 0
    while i < L:
        if int(toks[i]) >= 256: i += 1; continue
        j = i
        while j < L and int(toks[j]) < 256: j += 1
        ff, _ = signal_features(bytes(toks[i:j].astype(np.uint8)))
        f[i:j] = ff
        i = j
    return f


def load_video(path, log):
    """group tri-modal windows by source film, keep temporal order (for next-window prediction)."""
    films = {}
    if not os.path.exists(path):
        log(f"  video: {path} MISSING"); return []
    for line in open(path, encoding="utf-8", errors="replace"):
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        b = bytes.fromhex(r["bytes_hex"]); src = r.get("src", "single")
        films.setdefault(src, []).append(b)
    seqs = [w for w in films.values() if len(w) >= 2]
    log(f"  video: {sum(len(w) for w in seqs)} windows / {len(seqs)} films")
    return seqs


def batch_text(toks, bs, K, n_in, out_len, rng):
    """dense CTX (enc) -> AR continuation (dec); EOS(257) in the stream teaches closure."""
    cl = K*n_in; W = cl+out_len
    inpf = torch.zeros(bs, n_in, K, FEAT_DIM); of = torch.zeros(bs, out_len, FEAT_DIM); oid = torch.zeros(bs, out_len, dtype=torch.long)
    for i in range(bs):
        s = rng.randint(0, len(toks)-W-1)
        ctx = toks[s:s+cl]; out = toks[s+cl:s+cl+out_len]
        inpf[i] = feats_for_tokens(ctx).reshape(n_in, K, FEAT_DIM)
        of[i] = feats_for_tokens(out); oid[i] = torch.from_numpy(out.astype(np.int64))
    return inpf.to(DEV), of.to(DEV), oid.to(DEV)


def load_chat(path, n):
    """POC chat data: split serialized at <|assistant|> -> (prompt bytes, response bytes)."""
    A = b"<|assistant|>"; out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        try: s = json.loads(line).get("serialized")
        except json.JSONDecodeError: continue
        if not s: continue
        b = s.encode("utf-8", "replace"); j = b.rfind(A)
        if j < 0: continue
        prompt, resp = b[:j+len(A)], b[j+len(A):]
        if len(resp) >= 4: out.append((prompt, resp))
        if len(out) >= n: break
    return out


def batch_chat(samples, bs, K, n_in, out_len, rng):
    """prompt (system+user) = dense input (enc); assistant answer + EOS = AR output (dec) -> dialogue + closure."""
    cl = K*n_in
    inpf = torch.zeros(bs, n_in, K, FEAT_DIM); of = torch.zeros(bs, out_len, FEAT_DIM); oid = torch.zeros(bs, out_len, dtype=torch.long)
    for i in range(bs):
        prompt, resp = samples[rng.randrange(len(samples))]
        pt = list(prompt)[-cl:]; pt = [0]*(cl-len(pt)) + pt                       # left-pad prompt to cl
        rt = list(resp)[:out_len-1] + [EOS_ID]; rt = rt + [EOS_ID]*(out_len-len(rt))  # answer + EOS, pad with EOS
        inpf[i] = feats_for_tokens(np.array(pt, np.int16)).reshape(n_in, K, FEAT_DIM)
        of[i] = feats_for_tokens(np.array(rt, np.int16)); oid[i] = torch.tensor(rt, dtype=torch.long)
    return inpf.to(DEV), of.to(DEV), oid.to(DEV)


def batch_video(seqs, bs, K, rng):
    """input = window_t (dense), output = window_{t+1} (AR). All windows same length."""
    wlen = len(seqs[0][0])
    n_in = wlen // K
    inpf = torch.zeros(bs, n_in, K, FEAT_DIM); of = torch.zeros(bs, wlen, FEAT_DIM); oid = torch.zeros(bs, wlen, dtype=torch.long)
    for i in range(bs):
        film = seqs[rng.randrange(len(seqs))]; t = rng.randrange(len(film)-1)
        wt, wn = film[t], film[t+1]
        inpf[i] = pack_input(wt, K); of[i] = out_features(wn); oid[i] = torch.tensor(list(wn))
    return inpf.to(DEV), of.to(DEV), oid.to(DEV)


def lr_at(step, warm, total, base, lo):
    if step < warm: return base*step/max(1, warm)
    p = (step-warm)/max(1, total-warm); return lo + 0.5*(base-lo)*(1+math.cos(math.pi*min(1.0, p)))


@torch.no_grad()
def eval_loss(model, fn, *a):
    model.eval(); inpf, of, oid = fn(*a)
    lo = model(inpf, of); bpb = float(F.cross_entropy(lo[:, :-1].reshape(-1, VOCAB), oid[:, 1:].reshape(-1)))/math.log(2)
    model.train(); return bpb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v6_mm_asym")
    ap.add_argument("--text", default=r"C:\Users\gguni\holo_v6_data\train.jsonl")
    ap.add_argument("--video", default=r"C:\Users\gguni\holo_v6_data\video_streams_multi.jsonl")
    ap.add_argument("--video-fallback", default=r"C:\Users\gguni\holo_v6_data\video_streams.jsonl")
    ap.add_argument("--max-gb", type=float, default=2.0)
    ap.add_argument("--dim", type=int, default=512); ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=12); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--K", type=int, default=DEFAULT_K); ap.add_argument("--n-in", type=int, default=64); ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--p-video", type=float, default=0.30)
    ap.add_argument("--chat", default=r"G:\내 드라이브\홀로빗 POC\data\train.jsonl")
    ap.add_argument("--p-chat", type=float, default=0.25); ap.add_argument("--chat-rows", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=60000); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=12); ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1); ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=606); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    ckpt = HERE/"checkpoints"/args.run_name; logd = HERE/"logs"/args.run_name
    ckpt.mkdir(parents=True, exist_ok=True); logd.mkdir(parents=True, exist_ok=True)
    lf = (logd/"train.log").open("a", encoding="utf-8")
    def log(m): print(m, flush=True); lf.write(m+"\n"); lf.flush()
    (logd/"config.json").write_text(json.dumps({**vars(args), "license": LICENSE, "arch": "AsymHSL-multimodal"}, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"=== AsymHSL UNIFIED MULTIMODAL (text + video) === {args.run_name} === RESEARCH-ONLY ===")
    tr = load_text(args.text, args.max_gb, log)
    vpath = args.video if os.path.exists(args.video) else args.video_fallback
    seqs = load_video(vpath, log)
    if not seqs: log("WARNING: no video — text-only run"); args.p_video = 0.0
    chats = load_chat(args.chat, args.chat_rows) if os.path.exists(args.chat) else []
    if not chats: log("WARNING: no chat data"); args.p_chat = 0.0
    log(f"  chat: {len(chats)} turns (prompt->answer+EOS) | mix: video {args.p_video} / chat {args.p_chat} / text {round(1-args.p_video-args.p_chat,2)}")
    model = AsymHSL(dim=args.dim, enc_layers=args.enc_layers, dec_layers=args.dec_layers, heads=args.heads, K=args.K).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=args.weight_decay)
    use_bf16 = DEV.type=="cuda" and torch.cuda.is_bf16_supported(); amp = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV.type=="cuda" and not use_bf16))
    start = 0
    if args.resume and (ckpt/"latest.pt").exists():
        rc = torch.load(ckpt/"latest.pt", map_location=DEV); model.load_state_dict(rc["model"])
        try: opt.load_state_dict(rc["opt"])
        except Exception: pass
        start = int(rc.get("step", 0)); log(f"RESUMED from {start}")
    log(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M | p_video {args.p_video} | bs {args.batch_size}x{args.accum} | {DEV}\n")
    rng = random.Random(args.seed+start); metrics = (logd/"metrics.jsonl").open("a", encoding="utf-8"); t0 = time.time()
    for step in range(start+1, args.steps+1):
        for g in opt.param_groups: g["lr"] = lr_at(step, args.warmup, args.steps, args.lr, args.min_lr)
        opt.zero_grad(set_to_none=True)
        r = rng.random()
        mode = "video" if (seqs and r < args.p_video) else \
               ("chat" if (chats and r < args.p_video + args.p_chat) else "text")
        for _ in range(args.accum):
            if mode == "video":  inpf, of, oid = batch_video(seqs, args.batch_size, args.K, rng)
            elif mode == "chat": inpf, of, oid = batch_chat(chats, args.batch_size, args.K, args.n_in, args.out_len, rng)
            else:                inpf, of, oid = batch_text(tr, args.batch_size, args.K, args.n_in, args.out_len, rng)
            with torch.autocast("cuda", dtype=amp, enabled=(DEV.type=="cuda")):
                lo = model(inpf, of); loss = F.cross_entropy(lo[:, :-1].reshape(-1, VOCAB), oid[:, 1:].reshape(-1)) / args.accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); scaler.step(opt); scaler.update()
        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            tb = eval_loss(model, batch_text, tr, args.batch_size, args.K, args.n_in, args.out_len, random.Random(123))
            vb = eval_loss(model, batch_video, seqs, args.batch_size, args.K, random.Random(7)) if seqs else 0.0
            cb = eval_loss(model, batch_chat, chats, args.batch_size, args.K, args.n_in, args.out_len, random.Random(9)) if chats else 0.0
            row = {"step": step, "lr": round(opt.param_groups[0]["lr"],6), "text_bpb": round(tb,4), "chat_bpb": round(cb,4), "video_bpb": round(vb,4), "min": round((time.time()-t0)/60,1)}
            metrics.write(json.dumps(row)+"\n"); metrics.flush(); log(json.dumps(row, ensure_ascii=False))
        if step % args.save_every == 0 or step == args.steps:
            tmp = ckpt/"latest.tmp"; torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": vars(args)}, tmp); tmp.replace(ckpt/"latest.pt")
            torch.save({"model": model.state_dict(), "step": step, "config": vars(args)}, ckpt/f"checkpoint_step_{step:06d}.pt")
    log(f"\ndone -> {ckpt}"); metrics.close(); lf.close()


if __name__ == "__main__":
    main()
