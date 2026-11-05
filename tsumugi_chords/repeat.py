"""Repeated bars hear each other: the chord model's probabilities of a bar mixed with those of the bars that repeat it.

Each bar is cut into ``SLOTS`` equal slots and described by what the model hears in each: the chord tones it expects
(12 pitch classes) and its bass (12 + N).  Two bars, a and b, are compared under every transposition s (b moved up s
semitones), so a chorus a step up still matches whatever the estimated keys say; the pair's transposition is the one
under which the ``k`` bars on either side match best (the context).  b repeats a when that context similarity reaches
``context_min`` and the two bars themselves reach ``local_min`` (a bar the frames read a little differently still
matches; a bar that differs, such as a chorus's last bar, does not).  a's root x quality probabilities are then mixed
with b's at the same place in the bar, transposed by s:  p'_a = (p_a + beta * sum_b w_b p_b) / (1 + beta * sum_b w_b),
w_b the context similarity.  The bass is not mixed (the labels' slash chords lose by it).

On the validation songs this lifts tetrads 0.814 -> 0.823 and notes 0.763 -> 0.774 (slash 0.607 -> 0.609); a larger
``beta`` or looser thresholds pull different bars together and lose."""
from __future__ import annotations

import numpy as np

from .theory import chord_tone_templates

FPS = 25
SLOTS = 8
BASS_WEIGHT = 0.7
_TONES: np.ndarray | None = None


def tones() -> np.ndarray:
    """``[745, 12]`` chord tones of every root x quality class (N: none)."""

    global _TONES
    if _TONES is None:
        _TONES = np.r_[chord_tone_templates(), np.zeros((1, 12), np.float32)]
    return _TONES


def bar_spans(downbeats, frames: int) -> np.ndarray:
    """``[bars, 2]`` frame spans from downbeat to downbeat."""

    db = np.round(np.asarray(downbeats, np.float64) * FPS).astype(int)
    db = db[(db >= 0) & (db < frames)]
    spans = np.c_[db[:-1], db[1:]] if len(db) > 1 else np.zeros((0, 2), int)
    return spans[spans[:, 1] > spans[:, 0]]


def slot_means(x: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """``[bars, SLOTS, ...]`` means of ``x`` over each bar's slots."""

    out = np.zeros((len(spans), SLOTS) + x.shape[1:], np.float32)
    for i, (a, b) in enumerate(spans):
        edges = np.linspace(a, b, SLOTS + 1).round().astype(int)
        for s in range(SLOTS):
            out[i, s] = x[edges[s] : max(edges[s + 1], edges[s] + 1)].mean(0)
    return out


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def similarities(detail: np.ndarray, bass: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """``[12, bars, bars]``: the slots' mean cosine of bar a and bar b moved up s semitones."""

    chroma = unit(slot_means(detail @ tones(), spans))
    low = slot_means(bass, spans) * BASS_WEIGHT
    a = unit(np.concatenate([chroma, low], 2))
    out = np.zeros((12, len(spans), len(spans)), np.float32)
    for s in range(12):
        b = unit(np.concatenate([np.roll(chroma, s, 2), np.roll(low[..., :12], s, 2), low[..., 12:]], 2))
        out[s] = np.einsum("asd,bsd->ab", a, b) / SLOTS
    return out


def context(sim: np.ndarray, k: int) -> np.ndarray:
    """Mean of ``sim[..., a + d, b + d]`` over ``0 < |d| <= k`` (the pairs that exist)."""

    n = sim.shape[-1]
    total, count = np.zeros_like(sim), np.zeros(sim.shape[-2:], np.float32)
    for d in [d for d in range(-k, k + 1) if d]:
        lo, hi = max(0, -d), min(n, n - d)
        total[..., lo:hi, lo:hi] += sim[..., lo + d : hi + d, lo + d : hi + d]
        count[lo:hi, lo:hi] += 1
    return total / np.maximum(count, 1)


def roll_detail(x: np.ndarray, shift: int) -> np.ndarray:
    """Root x quality probabilities ``[T, 12 * qualities + 1]`` moved up ``shift`` semitones (N stays)."""

    if shift % 12 == 0:
        return x
    q = x[:, :-1].reshape(len(x), -1, 12)
    return np.c_[np.roll(q, shift, axis=2).reshape(len(x), -1), x[:, -1:]]


def smooth(detail: np.ndarray, bass: np.ndarray, downbeats, beta: float = 0.5, context_min: float = 0.9,
           local_min: float = 0.75, k: int = 2, min_lag: int = 2) -> np.ndarray:  # fmt: skip
    """``detail`` ``[T, 745]`` with each bar mixed with the bars that repeat it (``bass`` ``[T, 13]`` and the downbeats
    in seconds only find them)."""

    spans = bar_spans(downbeats, len(detail))
    n = len(spans)
    if beta <= 0 or n < 2 * k + 2:
        return detail
    sim = similarities(detail, bass, spans)
    ctx = context(sim, k)
    shift = ctx.argmax(0)
    a, b = np.indices((n, n))
    weight = ctx[shift, a, b]
    match = (np.abs(a - b) >= min_lag) & (weight >= context_min) & (sim[shift, a, b] >= local_min)
    out = detail.copy()
    for i in range(n):
        partners = np.flatnonzero(match[i])
        if not len(partners):
            continue
        a0, a1 = spans[i]
        place = (np.arange(a0, a1) - a0) / (a1 - a0)
        mixed, total = np.zeros((a1 - a0, detail.shape[1]), np.float32), 0.0
        for j in partners:
            b0, b1 = spans[j]
            frames = np.clip(np.round(b0 + place * (b1 - b0)).astype(int), 0, len(detail) - 1)
            mixed += weight[i, j] * roll_detail(detail[frames], int(shift[i, j]))
            total += weight[i, j]
        out[a0:a1] = (detail[a0:a1] + beta * mixed) / (1 + beta * total)
    return out
