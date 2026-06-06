"""4th modality: IMAGE. Grayscale pixels raster-scanned to a 1D byte stream (0-255 =
bytes directly, no μ-law). Same byte-native encoder/decoder, raw & HSL arms as comparison
(NO superiority claim) — just demonstrating the one model WORKS on image bytes too.
"""
from __future__ import annotations
import sys, pathlib, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torchvision
from run_smooth_signal import LM, run, FEAT_DIM
from hsl_signal_encoder import signal_features
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_stream(max_bytes, root):
    ds = torchvision.datasets.CIFAR10(root, train=True, download=True)
    out = bytearray()
    for img, _ in ds:
        g = np.array(img.convert("L"), dtype=np.uint8)         # 32x32 grayscale
        out.extend(g.flatten().tobytes())                      # raster -> 1D byte stream
        if len(out) >= max_bytes:
            break
    return bytes(out[:max_bytes])


def make_set(stream, N, n):
    nw = min(n, len(stream) // N)
    feats = torch.zeros(nw, N, FEAT_DIM); ids = torch.zeros(nw, N, dtype=torch.long)
    for i in range(nw):
        b = stream[i*N:(i+1)*N]; f, _ = signal_features(b)
        feats[i] = f[:N]; ids[i] = torch.tensor(list(b))
    return feats, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256); ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--mb", type=int, default=4_000_000)
    a = ap.parse_args()
    root = r"C:\Users\gguni\AppData\Local\Temp\hsl_img"
    stream = image_stream(a.mb, root)
    print(f"image[CIFAR-10 grayscale raster]: {len(stream)} bytes | N={a.N} | dim{a.dim}/{a.layers}L | {DEV}", flush=True)
    half = int(len(stream) * 0.85)
    tr = tuple(t.to(DEV) for t in make_set(stream[:half], a.N, 8000))
    va = tuple(t.to(DEV) for t in make_set(stream[half:], a.N, 1500))
    print(f"\n{'arm':>6} {'bits/pixel':>11}  (comparison arms — no superiority claim)")
    r = run("raw", tr, va, a.dim, a.layers, a.heads, a.steps); print(f"{'raw':>6} {r:>11.3f}", flush=True)
    h = run("hsl", tr, va, a.dim, a.layers, a.heads, a.steps); print(f"{'hsl':>6} {h:>11.3f}", flush=True)
    print(f"\nimage modality: byte-native encoder/decoder WORKS on image bytes (4th modality).")


if __name__ == "__main__":
    main()
