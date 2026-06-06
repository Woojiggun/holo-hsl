"""Verify CLOSURE (the '1' endpoint) on a TRAINED checkpoint — not the toy, real held-out docs.

'0과 1의 요동': 0 = origin anchor (lossless recon, codec-proven). 1 = closure (the end-point).
This checks the 1 on a real model: at a TRUE document end, does the model predict the closure
token (EOS_ID=257, or a chosen terminator byte) far more than at mid-document positions?
  closure works  <=>  P(term | end) >> P(term | mid)  AND argmax at end == term often.
Models trained with EOS between docs (hsl_train.py) should pass; newline-only v6 won't (honest).
"""
from __future__ import annotations
import sys, json, math, argparse, random, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from hsl_signal_encoder import signal_features
from hsl_decoder import HSLDecoder, EOS_ID, VOCAB
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_docs(path, n, min_len=40, max_len=255):
    docs = []
    for line in open(path, encoding="utf-8", errors="replace"):
        try: d = json.loads(line)
        except json.JSONDecodeError: continue
        t = d.get("serialized") or d.get("text") or d.get("output") or ""
        b = t.encode("utf-8", "replace")
        if len(b) >= min_len:
            docs.append(b[:max_len])
        if len(docs) >= n: break
    return docs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE/"checkpoints"/"hsl_lm_main_512"/"checkpoint_step_020000.pt"))
    ap.add_argument("--val", default=r"G:\내 드라이브\홀로빗 POC\data\val.jsonl")
    ap.add_argument("--term", default="eos", help="eos (token 257) | a byte value e.g. 10 for newline")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    term = EOS_ID if a.term == "eos" else int(a.term)
    ck = torch.load(a.ckpt, map_location=DEV); cfg = ck.get("config", {})
    m = HSLDecoder(cfg.get("dim",512), cfg.get("layers",12), cfg.get("heads",8)).to(DEV).eval()
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    docs = load_docs(a.val, a.n)
    print(f"loaded {pathlib.Path(a.ckpt).name} step {ck.get('step','?')} | term={a.term}({term}) | {len(docs)} real docs\n")

    p_end, p_mid, hit_end, hit_mid, nmid = [], [], 0, 0, 0
    for b in docs:
        L = len(b)
        feats = signal_features(b)[0].unsqueeze(0).to(DEV)
        tok = torch.tensor([list(b)], dtype=torch.long, device=DEV); mask = torch.ones(1, L, device=DEV)
        probs = m(feats, tok, mask)[0].softmax(-1)                 # [L, VOCAB]
        # position L-1 predicts what follows the LAST content byte -> should be the terminator at a true end
        pe = float(probs[L-1, term]); p_end.append(pe); hit_end += int(int(probs[L-1].argmax()) == term)
        for t in random.sample(range(0, L-2), min(3, max(1, L-2))):  # mid positions
            pm = float(probs[t, term]); p_mid.append(pm); hit_mid += int(int(probs[t].argmax()) == term); nmid += 1

    me, mm = sum(p_end)/len(p_end), sum(p_mid)/len(p_mid)
    print(f"  P({a.term} | TRUE doc end)  = {me:.4f}   argmax==term at end: {hit_end}/{len(docs)} = {hit_end/len(docs):.1%}")
    print(f"  P({a.term} | mid-document)  = {mm:.4f}   argmax==term at mid: {hit_mid}/{nmid} = {hit_mid/max(nmid,1):.1%}")
    ratio = me/max(mm, 1e-9)
    print(f"\n  closure margin: P(end)/P(mid) = {ratio:.1f}x   Δ = {me-mm:+.4f}")
    if me > 0.5 and ratio > 5:
        print("  => CLOSURE WORKS on the trained model: it strongly predicts the end-marker at true ends, not mid.")
    elif me > mm * 3:
        print("  => closure PARTIAL: end-marker raised at true ends, but not dominant.")
    else:
        print("  => closure WEAK/ABSENT on this model (likely not trained with this terminator).")


if __name__ == "__main__":
    main()
