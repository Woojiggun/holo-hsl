"""learned_hsl — HSL encoders, split into two clearly-separated files.

Two files, both HSL-prefixed:
  hsl_signal_encoder.py  : 비학습형 (algorithmic signal substrate, no params) —
                           signal_features / pack_batch. The STABLE foundation the
                           production + decoder path depends on.
  hsl_learned_encoder.py : 학습형 (learned cluster=VQ + schema=phase attention),
                           built ON TOP of the signal substrate.

All HSL signal formulas preserved (Δ change-rate, Δ²/boundary, rFFT, complex phase,
multi-hop Feynman attention) AND the keystone raw bits kept. Minimal intervention:
codes/relations self-organize, nothing hand-named.
"""
from .hsl_signal_encoder import signal_features, pack_batch, FEAT_DIM, FEAT_NAMES, ORIGIN, CLOSURE
from .hsl_learned_encoder import LearnedHSLEncoder, VectorQuantizer, PhaseSchemaBlock

__all__ = [
    # non-learned signal substrate
    "signal_features", "pack_batch", "FEAT_DIM", "FEAT_NAMES", "ORIGIN", "CLOSURE",
    # learned cluster + schema
    "LearnedHSLEncoder", "VectorQuantizer", "PhaseSchemaBlock",
]
