"""Chord segments from the model's 25 Hz probabilities: the semi-Markov decoder (``semi_crf.decode``) over the beat
grid, with chord-length and beat-position tables and a key-relative chord bigram counted on the training labels, then
one quality per segment (``chords.segments``) written as mir_eval labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import semi_crf as semi
from .chord_crf import NS, key_frames
from .chords import segments
from .heads import FRAME_RATE
from .theory import QUALITIES, chord_to_mir_eval

NO_CHORD = 12 * len(QUALITIES)


def bigram(path: Path) -> np.ndarray:
    """``[NS, NS]`` log P(next | previous) of key-relative chord states (Tsumugi-teacher and label counts, smoothed
    towards the unigram; no self-transitions)."""

    counts = np.load(path)
    table = counts["teacher"] + counts["labels"]
    np.fill_diagonal(table, 0.0)
    unigram = (table.sum(0) + 0.5) / (table.sum() + 0.5 * NS)
    prob = (table + 20.0 * unigram[None]) / (table.sum(1, keepdims=True) + 20.0)
    np.fill_diagonal(prob, 1e-12)
    return np.log(prob / prob.sum(1, keepdims=True))


@dataclass
class Chord:
    start: float  # seconds
    end: float
    label: str  # mir_eval syntax, e.g. "A:min7/b7"; "N" for no chord
    root: int | None  # pitch class 0..11 (C = 0)
    quality: str | None
    bass: int | None


class ChordDecoder:
    def __init__(self, tables: Path, lm: Path, params: dict):
        self.tables = semi.tables_from_counts(dict(np.load(tables)))
        self.lm = bigram(lm)
        self.params = semi.Params(**params)

    def __call__(self, detail: np.ndarray, bass: np.ndarray, boundary: np.ndarray, beats: np.ndarray, downbeats: np.ndarray,
                 keys: list[dict]) -> list[Chord]:  # fmt: skip
        """``detail`` ``[T, 745]`` / ``bass`` ``[T, 13]`` / ``boundary`` ``[T]`` probabilities at 25 Hz; the beats,
        downbeats (seconds) and keys (``{"start", "end", "key"}``) the model read."""

        detail = detail.astype(np.float32)
        tonic = key_frames(keys, len(detail))
        cp, bp = semi.decode(detail, bass.astype(np.float32), boundary.astype(np.float32), tonic, np.asarray(beats, np.float64),
                             np.asarray(downbeats, np.float64), self.params, self.tables, self.lm)  # fmt: skip
        out = []
        for seg in segments(detail, cp, bp):
            start, end = seg.start / FRAME_RATE, seg.end / FRAME_RATE
            if seg.detail >= NO_CHORD:
                out.append(Chord(start, end, "N", None, None, None))
            else:
                root, quality = seg.detail % 12, QUALITIES[seg.detail // 12]
                bass_pc = seg.bass if seg.bass < 12 else None
                out.append(Chord(start, end, chord_to_mir_eval(root, quality, bass_pc), root, quality, bass_pc))
        return out
