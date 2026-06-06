# HoLo / HSL — a byte-native, modality-agnostic substrate

*One small byte-native encoder/decoder that processes text, dialogue, audio, image, and video with a
single architecture — no per-modality tokenizer. An amateur, single-GPU (RTX 4070) proof of concept.*

**Frame (honest, on purpose):** I'm an independent amateur. This is **not** a benchmark-beating system
and makes **no performance-superiority claim**. The claim is narrower and, I think, more interesting:
*the possibility looks real, it runs, and the numbers are reported as-is.* Proving it at scale,
robustly, and safely is for people with the resources. — Jinhyun Woo

## The idea
Train AI a little like a baby: raw, unlabeled, multisensory streams, no pre-imposed symbols. Everything
becomes **bytes**, and a single **modality-agnostic feature substrate** turns bytes into a compact
signal representation. Correlated signals (a picture + a sound + a word) get bound on their own from one
interleaved stream. Founding intuition: *"everything is information — a fluctuation between 0 and 1."*
The substrate anchors that with two endpoints: **0** = an origin enabling lossless reconstruction, and
**1** = a learned **closure** signal ("this is the end") the model can predict.

## What's measured (reported as-is; raw is a comparison arm, no superiority claim)
| finding | result |
|---|---|
| **Closure ("1")** — does the model predict the end-marker at TRUE ends? | P(end-marker \| true end) **0.640** vs \| mid-document **0.000** (real held-out docs) |
| **Asymmetric dense-input** vs 1:1 byte-LM (same held-out targets) | bits/byte **1.572** (101M, 512B dense ctx) vs **1.836** (38M, 256B). Δ −0.264. (confounded by params+ctx; validates the *direction*) |
| **Cross-modal binding** (one byte model, interleaved image\|audio\|text) | caption bits/byte matched **0.024** vs mismatched **0.869**; held-out instances **0.038** vs **1.090** |
| **Real-video binding** (frames + audio + ASR captions → byte windows) | held-out matched **0.125** vs mismatched **1.193** (Δ +1.068) |
| **Self-built geometry** (read-only probe) | shallow layers = frequency/hub axis; deep layers = category clustering. Emerges unforced; *forcing* it hurts prediction. |
| **Multimodal generation** | one model autoregressively generates text, dialogue, image (raster), audio (μ-law), and short video (frames+sound) — garbled at this scale, but real cross-modal generation. |

## Architecture (fully open here)
- **AsymHSL** (the finalized model): a **dense bidirectional input encoder** (K bytes packed per
  embedding → big cheap context) + a **byte-autoregressive decoder** that cross-attends to the encoded
  input (and optional retrieved memory), with a 3-tier memory (window self-attn → gated fast-weight
  recall → cross-attn). RoPE positions; byte vocab + MASK + EOS.
- Trained from scratch on a mixed byte stream: **text + dialogue turns + video tri-modal windows**,
  each closed with an EOS/window marker (the "1").
- All of the above — model, training, multimodal pipeline, generation, interactive 3D probe — is in
  this repo and runs on **any** byte→feature map of width `FEAT_DIM`.

## 🔒 Encoder access (the one withheld piece)
The **byte→signal feature substrate** (`hsl_signal_encoder`, `hsl_codec`) — the core IP — is **not
included**; the files here are non-functional stubs. It is an exact, invertible map from bytes to a
compact feature vector combining **bit-level, change-rate, spectral, and complex-phase** components.
To reproduce with the original substrate, use the **HSL Encoder API** (contact the author), or drop in
your own `byte → [L, FEAT_DIM]` feature extractor — the rest of the repo works either way.

## Honest limits
Toy scale, single seed, 8-bit modalities, tiny models on one 4070. Margins are small and partly inflated
by simplicity. Generation is "fluent-shaped babble," not coherent content. This demonstrates *mechanism
and possibility*, not competitiveness.

## Data & attribution
All training data is public and **not redistributed here** (scripts fetch from original hosts). Full
source/license/attribution: **`DATA_SOURCES.md`**. Because several dialogue sources are CC BY-NC and
the corpus mixes CC BY-SA / ODC-By, **any trained weights are research / non-commercial only** and must
carry attribution + share-alike where applicable.

## Evidence
Full measured results — baselines, comparisons, and per-technique works-verification, with the
**"works-proof, not superiority"** framing — are in **`RESULTS.md`**. Data attribution: **`DATA_SOURCES.md`**.

## Author / contact / cite
- **Jinhyun Woo** — Independent Researcher
- ✉️ ggunio5782@gmail.com
- 💼 LinkedIn: `TODO — add URL`
- 🆔 ORCID: `TODO — create at orcid.org (free) and add`
- 🐙 GitHub: `TODO` · 🤗 Hugging Face: `TODO`

Original independent work, created on own time and equipment; this timestamped public release
establishes authorship and prior disclosure. Cite via `CITATION.cff` (or the Zenodo DOI once minted).
See `LICENSE.md` (open code Apache-2.0; encoder withheld; weights research/non-commercial).
The proof, the scale, and the safety — over to the experts.
