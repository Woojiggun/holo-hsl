"""HSL signal encoder — WITHHELD (core IP, not included in this open release).

This module is the byte->signal feature SUBSTRATE: an exact, invertible map from raw bytes to a
compact multi-component feature vector (bit-level + change-rate + spectral + complex-phase
components), kept synchronized with a lossless codec so that decode(encode(x)) == x.

The exact feature recipe is intentionally withheld. Everything else in this repository — the
architecture, training, multimodal pipeline, generation, probes, and results — is fully open and
runs on ANY byte->[L, FEAT_DIM] feature map. To reproduce with the original substrate, use the
HSL Encoder API (see README, "Encoder access"), or drop in your own feature extractor.

FEAT_DIM is published so the open architecture is runnable once any encoder is provided.

Author: Jinhyun Woo (ggunio5782@gmail.com). See LICENSE. The encoder/codec are proprietary.
"""
from __future__ import annotations

FEAT_DIM = 29  # dimensionality of the per-byte feature vector (recipe withheld)


def signal_features(data: bytes):
    raise NotImplementedError(
        "The HSL signal encoder is withheld (core IP). Options:\n"
        "  1) Call the HSL Encoder API: bytes -> [L, FEAT_DIM] features (see README 'Encoder access').\n"
        "  2) Provide your own byte->feature map of width FEAT_DIM.\n"
        "The architecture / training / probes in this repo are fully open and work with either."
    )
