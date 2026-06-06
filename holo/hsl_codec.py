"""Lossless byte<->signal codec — WITHHELD (paired with the encoder IP, not in this open release).

Provides the invertible byte<->signal mapping the encoder is synchronized with. Access via the
HSL Encoder API (see README, "Encoder access"). Stubs below keep imports resolvable.

Author: Jinhyun Woo. Proprietary. See LICENSE.
"""
from __future__ import annotations


def encode_bytes(data: bytes):
    raise NotImplementedError("Codec withheld (core IP). See README 'Encoder access'.")


def decode_frame(*args, **kwargs):
    raise NotImplementedError("Codec withheld (core IP). See README 'Encoder access'.")
