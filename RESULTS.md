# Results — measured evidence (WORKS-proof, NOT a superiority claim)

> **Framing (read first).** This is a **proof that it *works*** — that the mechanisms run, train, and
> behave as designed at tiny scale — **not a claim of superiority** over any existing system. Where a
> "raw bytes" arm appears, it is a **comparison/control arm only**; margins are small, single-seed,
> 8-bit, on tiny models on one RTX 4070. Several results are explicit **NULLs / parities**, reported
> honestly. Nothing here says "HSL beats X." It says "this is buildable, and here is what we measured."

Each result lists the script that produced it. Reproduction needs the withheld encoder (see DATA_SOURCES / README).

---

## A. Baseline comparisons (vs raw bytes / vs 1:1)
**A1. Encoder A/B — raw vs HSL-features vs learned-VQ** (`hsl_encoder_eval.py`)
Same tiny transformer body; only the input encoder differs. Discrimination acc:
- 1500 rows: raw 0.986 · nonlearned-HSL 0.991 · learned-VQ 0.989 → **parity (within single-seed noise)** at fewer params.
- Scaled 5000 rows: raw **0.997** > nonlearned 0.992 > learned 0.984. → **VQ discretization is net-negative**; HSL-features ≈ raw.
- **Verdict: NULL on superiority** — HSL features neither beat nor lose to raw here; VQ slightly hurts. Honest.

**A2. Asymmetric dense-input vs 1:1 byte-LM** (`compare_arms.py`) — identical held-out 256B targets
- 1:1 HSLDecoder (38M, 256B ctx): **1.836 bits/byte** · AsymHSL (101M, 512B dense ctx): **1.572**. Δ −0.264.
- **Confounded** by params (2.7×) and context (2×) → validates the *dense-input direction pays*, not "architecture alone wins."

**A3. Next-byte generation, raw vs HSL** (decoder smoke/scaled)
- smoke: raw 0.677 / hsl 0.655 bpb · scaled(~11M): raw 0.396 / hsl 0.377. Small **persistent** edge at fewer params — *"on par, slightly better,"* not decisive.

## B. Multimodal — one byte-native arch across modalities (raw = comparison arm)
**B1. Per-modality next-sample bits** (`run_smooth_signal.py`, `run_audio_test.py`, `run_image_test.py`)
| signal | raw arm | hsl arm |
|---|---|---|
| synthetic smooth | 1.343 | 1.189 |
| random (control) | 8.005 | 8.003 |
| real speech (FSDD) | 4.046 | 3.907 |
| real music (Commons) | 1.903 | 1.855 |
| image (CIFAR-10 gray raster, bits/px) | 5.120 | 5.046 |
→ **Takeaway is functional**: the *same* small byte encoder/decoder runs on text, audio, image — 4 modalities, one arch, no tokenizer. No superiority claim.

**B2. Cross-modal binding** (`run_crossmodal.py`) — one model on interleaved [image|audio|text]
- TEXT bits/byte matched **0.024** vs mismatched **0.869**. Held-out instances: **0.038** vs **1.090**. Retrieval (img+aud→right word) ≫ chance.
- Binding emerges from one stream — no pairing, no alignment, no per-modality work.

**B3. Real-video binding** (`run_video_binding.py`, `video_to_stream.py`) — PD film → [frame|audio|ASR-caption] windows
- held-out caption bits/byte matched **0.125** vs mismatched **1.193** (Δ **+1.068**).

## C. Technique works-verification (does each ported mechanism do its job?)
**C1. Closure ("1" endpoint)** (`closure_check.py`) — P(end-marker | true doc end) **0.640** vs | mid-doc **0.000** (~10⁵×). Learns where content closes.
**C2. Fast-weight memory dial** (`run_dial_open.py`) — windowed-attn recall 0.066 → +fast-weight **1.000**; gate opens 0→0.21 (opens only when recall is needed; ~0 = no harm otherwise).
**C3. Self-built geometry probe** (`curvature_probe.py`, `geo_lm.py`) — shallow layers organize bytes by frequency/centrality (corr −0.75), deep by category (+0.46), **unforced**. Forcing geometry as a loss **HURTS** prediction at every weight (λ=0 best) → it's a lens, not a target.
**C4. Max bytes per input embedding** (`run_maxk_input.py`) — reconstruction frontier ≈ K=8 (~0.90) → sets the dense-input packing.
**C5. Dynamic vs fixed patching** (`run_blt_audio.py`, `run_blt_toy.py`) — fixed-stride BLT ~10% worse than a flat byte baseline at this scale (structural, not undertraining); dynamic boundaries didn't beat fixed → fixed-K adopted (don't-force discipline).
**C6. Self-built universe at scale** — from-scratch byte model on 31.5MB real chat: bits/byte 8.36→**1.464**; builds a *coherent but non-human* grid (frequency↔radius +0.386, no human categories). Legitimate by prediction, not by resembling us.

## D. From-scratch v6 (new license-clean corpus, no overfit)
- 1:1 HSLDecoder 38M: val **1.563** bits/byte @20k · AsymHSL 101M: ~1.61 @16k (streaming windows → train≈val, no overfit).
- Unified multimodal (text+dialogue+video, AsymHSL): training; text/chat/video bits/byte all descending.

## Honest caveats (apply to everything above)
Single seed, toy proxies, 8-bit modalities, tiny models, one consumer GPU. Margins small and partly
inflated by simplicity. This is **mechanism + possibility**, demonstrated and reported as-is — the
proof at scale, robustness, and safety are explicitly left to others. — © 2026 Jinhyun Woo
