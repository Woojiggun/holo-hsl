"""Step 1 (real audio): does HSL beat raw on REAL audio? μ-law 8-bit (1 byte/sample),
same setup as the synthetic smooth proof. Mirrors run_smooth_signal but on real WAV.
"""
from __future__ import annotations
import sys, pathlib, glob, wave, math, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch
import torchaudio.functional as AF
from run_smooth_signal import LM, run, FEAT_DIM
from hsl_signal_encoder import signal_features
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mulaw_bytes(wav_glob, max_bytes):
    out = bytearray()
    files = sorted(glob.glob(wav_glob))
    for f in files:
        if f.lower().endswith(".npy"):
            x = np.load(f).astype(np.float32)               # music: pre-decoded float [-1,1]
        else:
            try:
                w = wave.open(f); n = w.getnframes(); raw = w.readframes(n)
                sw, ch = w.getsampwidth(), w.getnchannels(); w.close()
            except Exception:
                continue
            if sw != 2:
                continue
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                x = x.reshape(-1, ch).mean(1)
        mu = AF.mu_law_encoding(torch.from_numpy(x), 256).clamp(0, 255).numpy().astype(np.uint8)
        out.extend(mu.tobytes())
        if len(out) >= max_bytes:
            break
    return bytes(out[:max_bytes]), len(files)


def make_set(stream, N, n):
    nw = min(n, len(stream) // N)
    feats = torch.zeros(nw, N, FEAT_DIM); ids = torch.zeros(nw, N, dtype=torch.long)
    for i in range(nw):
        b = stream[i*N:(i+1)*N]
        f, _p = signal_features(b); feats[i] = f[:N]
        ids[i] = torch.tensor(list(b))
    return feats, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=r"C:\Users\gguni\AppData\Local\Temp\hsl_audio\**\recordings\*.wav")
    ap.add_argument("--label", default="speech")
    ap.add_argument("--N", type=int, default=256); ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--mb", type=int, default=4_000_000)
    a = ap.parse_args()
    stream, nf = load_mulaw_bytes(a.glob, a.mb)
    print(f"audio[{a.label}]: {len(stream)} μ-law bytes from {nf} files | N={a.N} | dim{a.dim}/{a.layers}L | {DEV}", flush=True)
    if len(stream) < a.N * 100:
        print("not enough audio loaded"); return
    half = int(len(stream) * 0.85)
    ftr, itr = make_set(stream[:half], a.N, 8000); fva, iva = make_set(stream[half:], a.N, 1500)
    tr = (ftr.to(DEV), itr.to(DEV)); va = (fva.to(DEV), iva.to(DEV))
    print(f"\n{'arm':>6} {'bits/sample':>12}")
    r = run("raw", tr, va, a.dim, a.layers, a.heads, a.steps)
    h = run("hsl", tr, va, a.dim, a.layers, a.heads, a.steps)
    print(f"{'raw':>6} {r:>12.3f}")
    print(f"{'hsl':>6} {h:>12.3f}", flush=True)
    win = "HSL" if h < r - 0.02 else ("raw" if r < h - 0.02 else "tie")
    print(f"\nΔ(hsl-raw)={h-r:+.3f}  winner={win}")
    print("HSL<raw on REAL audio = differentiator confirmed on a real modality (not just synthetic).")


if __name__ == "__main__":
    main()
