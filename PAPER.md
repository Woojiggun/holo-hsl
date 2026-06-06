# Everything is Information: a byte-native, modality-agnostic substrate and the seed of a world model

**Jinhyun Woo** (independent researcher, ggunio5782@gmail.com) · 2026 · *preliminary preprint*

## Abstract
We present a single **byte-native, modality-agnostic** encoder/decoder that processes text, dialogue,
audio, image, and short video with one architecture and **no per-modality tokenizer**. Built and trained
by an independent amateur on a single consumer GPU (RTX 4070), the system is offered as a **possibility
proof, not a benchmark result**: it *works* across modalities, and we report all numbers as-is with
explicit confounds. Bytes are mapped by an exact, invertible feature substrate (recipe withheld) into a
compact signal representation; a transformer with asymmetric **dense-input / byte-autoregressive-output**
and cross-attention models the stream. We measure: (i) a learned **closure** signal — the model predicts
an end-marker at true document ends (P=0.64) but essentially never mid-document (P≈0), giving a clean
"end-of-content vs end-of-window" distinction; (ii) the asymmetric dense-input design lowers bits/byte
vs a 1:1 byte-LM on identical targets (1.572 vs 1.836; confounded by parameters and context, so this
validates the *direction*); (iii) **cross-modal binding emerges for free** from a single interleaved
stream — captions are predicted far better when paired with the matching image+audio than with
mismatched ones, on held-out instances and on real video; (iv) the model builds a **non-human but
coherent geometry** of its byte space (frequency↔centrality at shallow layers, categories at deep
layers) that *emerges unforced* — explicitly forcing such geometry hurts prediction. The same model
generates all modalities autoregressively (garbled at this scale, but real cross-modal generation).

## 1. Motivation
Tokenizers impose language-specific symbols and break on new scripts/modalities. We ask whether a
substrate *below* language — raw information, "a fluctuation between 0 and 1" — can unify modalities by
their **rate of change** rather than by tokens, and whether correlations across senses bind themselves,
the way an infant's do, from unlabeled multisensory experience.

## 2. Substrate (withheld) and architecture (open)
Bytes are mapped by an exact, invertible feature substrate combining bit-level, change-rate, spectral,
and complex-phase components, synchronized with a lossless codec (recipe withheld; available via API).
Two endpoints anchor the stream: an **origin** (enabling lossless reconstruction) and a learned
**closure** marker. The model (**AsymHSL**) packs K bytes per input embedding into a dense bidirectional
encoder, then generates output byte-by-byte while cross-attending to the encoded context and a gated
fast-weight memory. It is trained from scratch on a mixed byte stream of text, dialogue turns, and video
tri-modal windows (frame | audio | caption), each closed with an end-marker.

## 3. Experiments (single-GPU, single-seed; raw = comparison arm only)
- **Closure.** On real held-out documents, P(end-marker | true end) = 0.640 (argmax = end-marker 64%)
  vs P(· | mid) ≈ 0.000 — a ~10⁵× contrast. The model learns *where content closes*.
- **Asymmetric dense input.** On identical 256-byte held-out targets: 1.572 bits/byte (101M params,
  512-byte dense context) vs 1.836 (38M, 256-byte context); Δ −0.264. Confounded by parameter count and
  context size — interpreted as "the dense-input direction pays," not "architecture alone wins."
- **Cross-modal binding.** One byte model on interleaved [image | audio | text] of the same concept:
  text bits/byte matched 0.024 vs mismatched 0.869; on held-out instances 0.038 vs 1.090. On a real
  public-domain narrated film decoded to [frame | audio | ASR-caption] windows: held-out matched 0.125
  vs mismatched 1.193 (Δ +1.068). Binding emerges with no pairing, alignment, or per-modality work.
- **Emergent geometry.** A read-only probe shows shallow layers organize bytes by frequency/centrality
  and deep layers by category, unforced. Adding an explicit geometry loss *hurts* prediction at every
  weight — the geometry is a lens, not a target.

## 4. Limitations
Toy scale, single seed, 8-bit modalities, tiny models. Margins are small and partly inflated by
simplicity. Generation is fluent-shaped but semantically incoherent. No superiority over existing
systems is claimed or implied; reproduction at scale, robustness, and safety are open.

## 5. Toward a world model
Next-byte prediction over a modality-unified change-rate stream is world-modeling-by-prediction; the
emergent geometry is a candidate latent state-space. Adding generative dynamics (e.g., diffusion-based
imagination) and action-conditioning is the natural next step — left as future work and as an invitation.

## Availability
Open repository (architecture, training, multimodal pipeline, generation, probes, results) under
Apache-2.0; the encoder/codec substrate is withheld and accessed via API. © 2026 Jinhyun Woo.
