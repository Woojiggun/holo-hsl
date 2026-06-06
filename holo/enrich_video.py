"""Enrich the multimodal corpus: extract tri-modal byte streams from MANY public-domain films.

For each archive.org Prelinger ID: download a small mp4 -> Whisper-transcribe -> video_to_stream
([16x16 frame | mu-law audio | caption] windows) -> append to video_streams_multi.jsonl.
Public domain (Prelinger), research use. Robust: skips any film that fails.
"""
from __future__ import annotations
import sys, os, json, subprocess, tempfile, pathlib, datetime
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from video_to_stream import extract, FFMPEG

def fetch_ids(n=60):
    """pull many Prelinger (public-domain, narrated) movie IDs from archive.org, by popularity."""
    import urllib.request, urllib.parse
    seed = ["AboutBan1935", "Doctorin1946", "HealthYo1953", "FromtheG1954", "Sleepfor1950",
            "JoanAvoi1947", "right_to_health_1", "Automoti1940", "Careofth1949", "Sniffles1955",
            "ParkCons1938", "Personal1950", "EatforHe1954"]
    try:
        q = "collection:(prelinger) AND mediatype:(movies)"
        url = ("https://archive.org/advancedsearch.php?q=" + urllib.parse.quote(q) +
               "&fl[]=identifier&rows=" + str(n) + "&sort[]=downloads+desc&output=json")
        d = json.load(urllib.request.urlopen(url, timeout=30))
        ids = [doc["identifier"] for doc in d["response"]["docs"]]
        for s in seed:
            if s not in ids: ids.append(s)
        return ids
    except Exception:
        return seed


IDS = fetch_ids(60)
OUT = r"C:\Users\gguni\holo_v6_data\video_streams_multi.jsonl"
VIDDIR = r"C:\Users\gguni\holo_v6_data\video"
_WH = {"m": None}


def transcribe_srt(mp4, srt):
    from faster_whisper import WhisperModel
    if _WH["m"] is None:
        _WH["m"] = WhisperModel("base", device="cuda", compute_type="float16")
    def ts(s):
        td = datetime.timedelta(seconds=s); h, r = divmod(td.seconds, 3600); m, sec = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{sec:02d},{int(td.microseconds/1000):03d}"
    segs, _ = _WH["m"].transcribe(mp4, language="en", vad_filter=True)
    n = 0
    with open(srt, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{ts(s.start)} --> {ts(s.end)}\n{s.text.strip()}\n\n"); n += 1
    return n


def main():
    os.makedirs(VIDDIR, exist_ok=True)
    fout = open(OUT, "w", encoding="utf-8")
    def log(m): print(m, flush=True)
    total = 0; ok = 0
    for vid in IDS:
        mp4 = os.path.join(VIDDIR, f"{vid}.mp4"); srt = os.path.join(VIDDIR, f"{vid}.srt")
        try:
            if not os.path.exists(mp4):
                log(f"[{vid}] downloading...")
                subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", "--playlist-items", "1",
                                "-f", "mp4", "-S", "+size,res", "-o", mp4, f"https://archive.org/details/{vid}"],
                               check=True, capture_output=True, timeout=900)
            log(f"[{vid}] transcribing...")
            ncap = transcribe_srt(mp4, srt)
            log(f"[{vid}] extracting streams ({ncap} captions)...")
            streams = extract(mp4, srt, fps=2.0, max_windows=2000, log=lambda m: None)
            for s in streams:
                s["src"] = vid; fout.write(json.dumps(s, ensure_ascii=False) + "\n")
            fout.flush(); total += len(streams); ok += 1
            log(f"[{vid}] +{len(streams)} windows  (total {total})")
        except Exception as e:
            log(f"[{vid}] SKIP: {repr(e)[:160]}")
    fout.close()
    log(f"\nDONE: {ok}/{len(IDS)} films, {total} tri-modal windows -> {OUT}")


if __name__ == "__main__":
    main()
