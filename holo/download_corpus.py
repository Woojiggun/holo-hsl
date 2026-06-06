"""Download a rich, validated, license-clean text corpus for from-scratch byte-LM training.

Sources (both deduplicated + clean, research-use OK):
  - English: HuggingFaceFW/fineweb-edu  (sample-10BT)  — ODC-By, high-quality edu web, deduped
  - Korean : wikimedia/wikipedia 20231101.ko           — CC BY-SA, clean encyclopedic

Streams to a byte budget per source (no full-dataset download), writes mixed shards with
per-row provenance + license. Output: <out>/train.jsonl + <out>/val.jsonl  (val = held-out tail).
Bytes are what the byte-LM consumes, so budgets are in bytes of UTF-8 text.
"""
from __future__ import annotations
import sys, os, json, time, argparse, itertools
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from datasets import load_dataset

SOURCES = [
    dict(key="fineweb-edu-en", lic="ODC-By", repo="HuggingFaceFW/fineweb-edu",
         cfg="sample-10BT", field="text"),
    dict(key="wikipedia-ko", lic="CC BY-SA 4.0", repo="wikimedia/wikipedia",
         cfg="20231101.ko", field="text"),
]


def stream_source(src, budget_bytes, min_len, log):
    ds = load_dataset(src["repo"], src["cfg"], split="train", streaming=True)
    got = 0
    for rec in ds:
        t = rec.get(src["field"]) or ""
        b = len(t.encode("utf-8", "replace"))
        if b < min_len:
            continue
        yield {"text": t, "source": src["key"], "license": src["lic"]}
        got += b
        if got >= budget_bytes:
            log(f"  [{src['key']}] reached {got/1e9:.2f} GB"); return
    log(f"  [{src['key']}] source exhausted at {got/1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Users\gguni\holo_v6_data")
    ap.add_argument("--en-gb", type=float, default=2.5)
    ap.add_argument("--ko-gb", type=float, default=1.5)
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.01)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logp = os.path.join(args.out, "download.log")
    lf = open(logp, "a", encoding="utf-8")
    def log(m):
        print(m, flush=True); lf.write(m + "\n"); lf.flush()

    budgets = {"fineweb-edu-en": int(args.en_gb * 1e9), "wikipedia-ko": int(args.ko_gb * 1e9)}
    trainp = os.path.join(args.out, "train.jsonl")
    valp = os.path.join(args.out, "val.jsonl")
    log(f"=== corpus download === out={args.out}  EN={args.en_gb}GB KO={args.ko_gb}GB")
    t0 = time.time()
    n_train = n_val = 0; bytes_by_src = {}
    with open(trainp, "w", encoding="utf-8") as ftr, open(valp, "w", encoding="utf-8") as fva:
        # interleave the two sources round-robin for a mixed stream
        gens = []
        for src in SOURCES:
            b = budgets[src["key"]]
            if b > 0:
                gens.append(stream_source(src, b, args.min_len, log))
        i = 0
        for rec in roundrobin(gens):
            i += 1
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if (i % int(1 / args.val_frac)) == 0:
                fva.write(line); n_val += 1
            else:
                ftr.write(line); n_train += 1
            bytes_by_src[rec["source"]] = bytes_by_src.get(rec["source"], 0) + len(rec["text"].encode("utf-8", "replace"))
            if i % 20000 == 0:
                tot = sum(bytes_by_src.values()) / 1e9
                log(f"  rows={i}  total={tot:.2f}GB  {dict((k, round(v/1e9,2)) for k,v in bytes_by_src.items())}  {(time.time()-t0)/60:.1f}min")
    log(f"DONE: train rows={n_train} val rows={n_val}  bytes={ {k: round(v/1e9,2) for k,v in bytes_by_src.items()} }  {(time.time()-t0)/60:.1f}min")
    log(f"  -> {trainp}\n  -> {valp}")
    lf.close()


def roundrobin(iterables):
    iters = [iter(it) for it in iterables]
    active = list(range(len(iters)))
    while active:
        nxt = []
        for k in active:
            try:
                yield next(iters[k]); nxt.append(k)
            except StopIteration:
                pass
        active = nxt


if __name__ == "__main__":
    main()
