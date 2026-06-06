"""Video -> synchronized byte stream (the 'baby watches video' substrate).

Takes a real video (frames + audio + subtitles) and emits ONE interleaved byte stream
per time-window, matching run_crossmodal's convention so the same byte-LM ingests it:

   per window t:  [frame raster bytes] SEP [audio mu-law bytes] SEP [caption text bytes] WSEP

No per-modality model — image/sound/text become bytes that co-occur in time, so binding
can emerge from prediction alone (the north-star, scaled from digits to real video).

Tooling: ffmpeg (winget Gyan.FFmpeg), opencv (frames), torchaudio (mu-law), webvtt/srt subs.
"""
from __future__ import annotations
import sys, os, glob, json, subprocess, tempfile, argparse, re, pathlib
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np


def find_ffmpeg():
    import shutil
    for name in ("ffmpeg",):
        p = shutil.which(name)
        if p: return p
    cands = glob.glob(os.path.expanduser(
        r"~/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/**/bin/ffmpeg.exe"), recursive=True)
    if cands: return cands[0]
    raise RuntimeError("ffmpeg not found")


FFMPEG = find_ffmpeg()
SEP, WSEP = 254, 255                                   # segment / window separators
IMG_HW = 16                                            # frame -> 16x16 grayscale (256 bytes)
AUD_BYTES = 256                                        # mu-law bytes per window
TXT_BYTES = 24                                         # caption bytes per window
SR = 8000                                              # audio sample rate


def parse_subs(path):
    """parse .srt/.vtt -> list of (start_sec, end_sec, text)."""
    if not path or not os.path.exists(path): return []
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = []
    pat = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")
    blocks = re.split(r"\n\s*\n", txt)
    for b in blocks:
        m = pat.search(b)
        if not m: continue
        h1,m1,s1,ms1,h2,m2,s2,ms2 = map(int, m.groups())
        st = h1*3600+m1*60+s1+ms1/1000; en = h2*3600+m2*60+s2+ms2/1000
        lines = [l for l in b.splitlines() if not pat.search(l) and "-->" not in l and not l.strip().isdigit()]
        cap = " ".join(l.strip() for l in lines if l.strip() and not l.startswith("WEBVTT"))
        if cap: out.append((st, en, cap))
    return out


def caption_at(subs, t):
    for st, en, cap in subs:
        if st <= t <= en: return cap
    return ""


def extract(video, subs_path, fps, max_windows, log):
    import cv2, torch, torchaudio.functional as AF, soundfile as sf
    # --- audio: decode to 8k mono wav via ffmpeg ---
    tmpwav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run([FFMPEG, "-y", "-i", video, "-ac", "1", "-ar", str(SR), "-vn", tmpwav],
                   check=True, capture_output=True)
    audio, _ = sf.read(tmpwav, dtype="float32"); os.unlink(tmpwav)
    if audio.ndim > 1: audio = audio.mean(1)
    # --- frames via opencv ---
    cap = cv2.VideoCapture(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    subs = parse_subs(subs_path)
    log(f"  video fps={vfps:.1f}  audio={len(audio)/SR:.1f}s  subs={len(subs)} cues  sampling at {fps}fps")
    streams = []; t = 0.0; step = 1.0 / fps; win_samp = int(SR / fps)
    while len(streams) < max_windows:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok: break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (IMG_HW, IMG_HW), interpolation=cv2.INTER_AREA).astype(np.uint8)
        img_b = g.flatten().tobytes()                                  # 256 bytes
        a0 = int(t * SR); seg = audio[a0:a0 + win_samp]
        if len(seg) < AUD_BYTES: seg = np.pad(seg, (0, AUD_BYTES - len(seg)))
        mu = AF.mu_law_encoding(torch.from_numpy(seg[:AUD_BYTES].copy()), 256).clamp(0,255).numpy().astype(np.uint8)
        aud_b = mu.tobytes()                                           # 256 bytes
        cap_txt = caption_at(subs, t + step/2)
        txt_b = (cap_txt.encode("utf-8","replace") + b" "*TXT_BYTES)[:TXT_BYTES]
        stream = img_b + bytes([SEP]) + aud_b + bytes([SEP]) + txt_b + bytes([WSEP])
        streams.append({"t": round(t,2), "caption": cap_txt, "bytes_hex": stream.hex()})
        t += step
    cap.release()
    return streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--subs", default=None)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-windows", type=int, default=4000)
    ap.add_argument("--out", default=r"C:\Users\gguni\holo_v6_data\video_streams.jsonl")
    a = ap.parse_args()
    def log(m): print(m, flush=True)
    log(f"ffmpeg={FFMPEG}\nextracting {a.video}")
    streams = extract(a.video, a.subs, a.fps, a.max_windows, log)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        for s in streams: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    capt = sum(1 for s in streams if s["caption"])
    log(f"wrote {len(streams)} windows ({capt} captioned) -> {a.out}")
    log(f"per-window bytes = {IMG_HW*IMG_HW}(img)+1+{AUD_BYTES}(aud)+1+{TXT_BYTES}(txt)+1 = {IMG_HW*IMG_HW+AUD_BYTES+TXT_BYTES+3}")


if __name__ == "__main__":
    main()
