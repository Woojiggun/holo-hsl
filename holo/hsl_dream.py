"""The model DREAMS across modalities — generate bytes, decode to image/audio/text.

AsymHSL (multimodal) generates a tri-modal window [16x16 frame | mu-law audio | caption] byte-AR,
optionally rolling forward window->window (a tiny dreamed 'video'). We then DECODE the bytes back:
  frame bytes  -> 16x16 grayscale PNG (+ animated GIF over the rollout)
  audio bytes  -> mu-law decode -> WAV (8 kHz), windows concatenated
  caption bytes-> text
Quality is garbled/low (toy scale) — but it's REAL cross-modal GENERATION, not just classification.
Point --ckpt at the multimodal run (v6_mm_asym) once trained; on a text-only ckpt it dreams noise.
"""
from __future__ import annotations
import sys, os, json, argparse, pathlib, random
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import glob, subprocess
import numpy as np, torch
import torchaudio.functional as AF
from PIL import Image
import soundfile as sf
from hsl_asym import AsymHSL, pack_input, out_features
from video_to_stream import FFMPEG                            # ffmpeg path (winget Gyan.FFmpeg)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# window layout (from video_to_stream.py): IMG | SEP | AUD | SEP | TXT | WSEP
IMG, AUD, TXT = 256, 256, 24
I0, A0, T0 = 0, IMG+1, IMG+1+AUD+1
WLEN = IMG+1+AUD+1+TXT+1                                     # 539
SR = 8000
FPS = SR / AUD                                               # 31.25 — frame i is shown for its 256-sample (0.032s) audio chunk => synced


def make_mp4(outdir):
    """assemble the dreamed frames + audio into ONE synced .mp4 (the 4th modality: motion)."""
    frames = sorted(glob.glob(os.path.join(outdir, "frame_*.png")))
    wav = os.path.join(outdir, "dream.wav"); mp4 = os.path.join(outdir, "dream.mp4")
    if not frames: return None
    cmd = [FFMPEG, "-y", "-framerate", f"{FPS}", "-i", os.path.join(outdir, "frame_%03d.png")]
    if os.path.exists(wav): cmd += ["-i", wav]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=256:256:flags=neighbor"]
    if os.path.exists(wav): cmd += ["-c:a", "aac", "-shortest"]
    cmd += [mp4]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120); return mp4
    except Exception as e:
        print(f"(mp4 mux failed: {repr(e)[:120]})"); return None


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEV); cfg = ck.get("config", {})
    m = AsymHSL(dim=cfg.get("dim",512), enc_layers=cfg.get("enc_layers",4), dec_layers=cfg.get("dec_layers",12),
               heads=cfg.get("heads",8), K=cfg.get("K",8)).to(DEV).eval()
    m.load_state_dict(ck["model"] if "model" in ck else ck); return m, cfg.get("K",8), ck.get("step","?")


@torch.no_grad()
def gen_window(model, seed_bytes, K, temp=0.8):
    """AsymHSL: dense input = seed window, AR-generate the next WLEN bytes."""
    inpf = pack_input(seed_bytes, K).unsqueeze(0).to(DEV)
    out = bytearray(b"\x00")
    for _ in range(WLEN - 1):
        of = out_features(bytes(out)).unsqueeze(0).to(DEV)
        logits = model(inpf, of)[0, -1].float()
        nxt = int(logits.argmax()) if temp <= 0 else int(torch.multinomial((logits[:256]/temp).softmax(-1), 1))
        out.append(nxt)
    return bytes(out[:WLEN])


def render(windows, outdir):
    os.makedirs(outdir, exist_ok=True)
    frames, audio, caps = [], [], []
    for w in windows:
        g = np.frombuffer(w[I0:I0+IMG], np.uint8).reshape(16, 16)
        frames.append(Image.fromarray(g, "L").resize((160, 160), Image.NEAREST))
        mu = torch.tensor(list(w[A0:A0+AUD]), dtype=torch.float)
        audio.append(AF.mu_law_decoding(mu, 256).numpy())
        caps.append(w[T0:T0+TXT].split(b"\xff")[0].decode("utf-8", "replace").strip())
    for i, f in enumerate(frames): f.save(os.path.join(outdir, f"frame_{i:03d}.png"))
    if frames:
        frames[0].save(os.path.join(outdir, "dream.gif"), save_all=True, append_images=frames[1:], duration=400, loop=0)
    if audio:
        sf.write(os.path.join(outdir, "dream.wav"), np.concatenate(audio).astype(np.float32), SR)
    open(os.path.join(outdir, "captions.txt"), "w", encoding="utf-8").write("\n".join(caps))
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE/"checkpoints"/"v6_mm_asym"/"latest.pt"))
    ap.add_argument("--seed-video", default=r"C:\Users\gguni\holo_v6_data\video_streams.jsonl")
    ap.add_argument("--n-windows", type=int, default=48); ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--out", default=str(HERE/"dream_out"))
    a = ap.parse_args()
    if not os.path.exists(a.ckpt):
        print(f"no checkpoint at {a.ckpt} — train v6_mm_asym first."); return
    model, K, step = load(a.ckpt); print(f"loaded {pathlib.Path(a.ckpt).name} (step {step})")
    # seed = a real window (so the model has perceptual context to roll from)
    seed = None
    if os.path.exists(a.seed_video):
        for line in open(a.seed_video, encoding="utf-8", errors="replace"):
            try: seed = bytes.fromhex(json.loads(line)["bytes_hex"]); break
            except Exception: continue
    if seed is None: seed = bytes(WLEN)
    windows = []
    cur = seed
    for _ in range(a.n_windows):
        cur = gen_window(model, cur, K, a.temp); windows.append(cur)        # rollout: each dream feeds the next
    caps = render(windows, a.out)
    mp4 = make_mp4(a.out)
    print(f"dreamed {len(windows)} windows -> {a.out}")
    dur = a.n_windows * AUD / SR
    print(f"  🎬 VIDEO -> dream.mp4 ({dur:.2f}s, {FPS:.1f}fps, frame+audio synced)" if mp4 else "  (mp4 skipped)")
    print(f"  also: frame_*.png + dream.gif | dream.wav | captions.txt")
    print(f"  4-modal generated: image(frame) + audio(sound) + text(caption) + video(motion). captions:")
    for c in caps[:12]: print("   ", repr(c))


if __name__ == "__main__":
    main()
