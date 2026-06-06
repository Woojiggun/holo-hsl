"""Conversation-like generation wiring (NO training): chat template + <|eos|> stop + decoding.

The 3 no-training pieces:
  1. chat template  : wrap the question as <|system|>..<|user|>..<|assistant|>\n  (the POC format)
  2. EOS stop       : stop when the model emits the literal "<|eos|>" (its trained terminator) or a new turn
  3. decoding       : temperature + repetition penalty + max length  -> fluent-shaped output, no loops

Works best on a chat-trained / SFT'd checkpoint. On a raw web/wiki base it will emit the markers as
unknown bytes (-> run hsl_sft.py first). Byte-level: markers are literal byte strings.
"""
from __future__ import annotations
import sys, pathlib, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
from hsl_signal_encoder import signal_features
from hsl_decoder import HSLDecoder, VOCAB
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOS_STR = b"<|eos|>"; STOPS = [b"<|eos|>", b"<|user|>", b"<|system|>", b"<|assistant|>"]


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEV); cfg = ck.get("config", {})
    m = HSLDecoder(cfg.get("dim", 512), cfg.get("layers", 12), cfg.get("heads", 8)).to(DEV).eval()
    m.load_state_dict(ck["model"] if "model" in ck else ck); return m, ck.get("step", "?")


@torch.no_grad()
def chat(model, question, system="You are a helpful assistant.", task="reasoning",
         max_new=240, temp=0.8, rep=1.3, rep_window=64, window=320):
    prompt = (f"<|system|>\n{system}\n<|user|><|task:{task}|>\n{question}\n<|assistant|>\n").encode("utf-8")
    out = bytearray(prompt); produced = []
    for _ in range(max_new):
        ctx = bytes(out[-window:])
        f, _ = signal_features(ctx)
        logits = model(f.unsqueeze(0).to(DEV), torch.tensor([list(ctx)], device=DEV),
                       torch.ones(1, len(ctx), device=DEV))[0, -1].float()
        for b in set(produced[-rep_window:]):                       # repetition penalty on recent bytes
            if logits[b] > 0: logits[b] /= rep
            else: logits[b] *= rep
        nxt = int(logits.argmax()) if temp <= 0 else int(torch.multinomial((logits/temp).softmax(-1), 1))
        if nxt >= 256: break                                        # EOS/MASK token
        out.append(nxt); produced.append(nxt)
        tail = bytes(out[-10:])
        if any(s in tail for s in STOPS): break                     # literal terminator / new turn
    resp = bytes(out)[len(prompt):]
    for s in STOPS: resp = resp.split(s)[0]
    return resp.decode("utf-8", "replace").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE/"checkpoints"/"v6_scratch_512x12"/"latest.pt"))
    ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--rep", type=float, default=1.3)
    ap.add_argument("--max-new", type=int, default=240)
    ap.add_argument("--q", default=None, help="single question; omit for REPL")
    a = ap.parse_args()
    model, step = load(a.ckpt); print(f"loaded {pathlib.Path(a.ckpt).name} (step {step}) | temp {a.temp} rep {a.rep}\n")
    if a.q is not None:
        print("🧑", a.q); print("🤖", chat(model, a.q, temp=a.temp, rep=a.rep, max_new=a.max_new)); return
    print("chat REPL — type a question (Ctrl-C to quit)")
    while True:
        try: q = input("🧑 ").strip()
        except (EOFError, KeyboardInterrupt): break
        if q: print("🤖", chat(model, q, temp=a.temp, rep=a.rep, max_new=a.max_new), "\n")


if __name__ == "__main__":
    main()
