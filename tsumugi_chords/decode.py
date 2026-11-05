"""Chord segments from the model's 25 Hz probabilities: the semi-Markov decoder (``semi_crf.decode``) over the beat
grid, with chord-length and beat-position tables and a key-relative chord bigram counted on the training labels, then
one quality per segment (``chords.segments``) written as mir_eval labels and as chord symbols (``Am7/G``)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import repeat
from . import semi_crf as semi
from .chord_crf import NS, PARENT, SHARP, beat_position, key_frames
from .chords import merge_fifth_bass, merge_short_slash, merge_similar, segments
from .heads import FRAME_RATE
from .theory import QUALITIES, chord_to_mir_eval

NO_CHORD = 12 * len(QUALITIES)
FLAT = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
LETTERS = "CDEFGAB"
NATURAL = (0, 2, 4, 5, 7, 9, 11)
# Letter steps of the 12 degrees above the tonic: 1, b2, 2, b3, 3, 4, #4, 5, b6, 6, b7, 7
DEGREE_STEP = (0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6)


def spell(pc: int, key: str | None = None, raised: bool = False) -> str:
    """The name of pitch class ``pc`` in ``key`` (e.g. ``"A:minor"``): the key's tonic spelt by its key signature
    (flats when its parent major is F, Bb, Eb, Ab or Db), the other notes by their degree above it (b3, #4, b7 ...),
    so G# is the leading tone of A minor and Bb the flat seventh of C major; with ``raised`` a note outside the key
    is spelt as a raised degree instead (#1, #2, #5, #6: the root of a diminished chord leads up, A#dim7 -> Bm in
    D major).  Cb, Fb, E# and B# are written B, E, F and C.  No key: C major's spelling."""

    tonic, _, mode = (key or "C:major").partition(":")
    if tonic not in SHARP and tonic not in FLAT:
        tonic, mode = "C", "major"
    t = SHARP.index(tonic) if tonic in SHARP else FLAT.index(tonic)
    names = FLAT if (t + PARENT.get(mode, 0)) % 12 in (1, 3, 5, 8, 10) else SHARP
    parent, degree = (t + PARENT.get(mode, 0)) % 12, (pc - t) % 12
    step = DEGREE_STEP[degree]
    if raised and (pc - parent) % 12 not in NATURAL and degree in (1, 3, 8, 10):
        step -= 1
    letter = LETTERS[(LETTERS.index(names[t][0]) + step) % 7]
    shift = (pc - NATURAL[LETTERS.index(letter)] + 6) % 12 - 6
    name = letter + {0: "", 1: "#", -1: "b"}[shift] if shift in (-1, 0, 1) else names[pc]
    return names[pc] if name in ("Cb", "Fb", "E#", "B#") else name  # as lead sheets write them


def chord_symbol(root: int | None, quality: str | None, bass: int | None, key: str | None = None) -> str:
    """A chord symbol as in lead sheets, e.g. ``Am7/G``, ``FM7``, ``G7(b9)``, ``Bm7-5``; ``N`` for no chord."""

    if root is None:
        return "N"
    name = spell(root, key, raised=quality.startswith("dim"))
    symbol = name + quality
    if bass is None or bass == root:
        return symbol
    # the bass by its interval above the root (B7/D#, C/Bb), else as the key spells it
    interval = (bass - root) % 12
    step = DEGREE_STEP[interval]
    if interval == 6 and ("-5" in quality or quality.startswith("dim")):
        step = 4  # b5
    elif interval == 8 and "aug" in quality:
        step = 4  # #5
    letter = LETTERS[(LETTERS.index(name[0]) + step) % 7]
    shift = (bass - NATURAL[LETTERS.index(letter)] + 6) % 12 - 6
    low = letter + {1: "#", -1: "b"}.get(shift, "") if shift in (-1, 0, 1) else None
    return symbol + "/" + (low if low and low not in ("Cb", "Fb", "E#", "B#") else spell(bass, key))


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
    symbol: str = "N"  # chord symbol spelt in the chord's key, e.g. "Am7/G"


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
        if self.params.repeat_beta and len(downbeats):
            detail = repeat.smooth(detail, bass.astype(np.float32), downbeats, self.params.repeat_beta, self.params.repeat_context,
                                   self.params.repeat_local)  # fmt: skip
        tonic = key_frames(keys, len(detail))
        cp, bp = semi.decode(detail, bass.astype(np.float32), boundary.astype(np.float32), tonic, np.asarray(beats, np.float64),
                             np.asarray(downbeats, np.float64), self.params, self.tables, self.lm)  # fmt: skip
        segs = segments(detail, cp, bp, self.params.symmetric)
        if self.params.merge_fifth_bass:
            segs = merge_fifth_bass(segs, self.params.merge_fifth_bass if isinstance(self.params.merge_fifth_bass, str) else "first")
        if self.params.short_slash:
            grid = np.asarray(beats, np.float64)
            segs = merge_short_slash(segs, lambda f: float(beat_position(grid, np.array([f / FRAME_RATE]))[0]), self.params.short_slash)
        if self.params.merge_similar:
            segs = merge_similar(segs, detail, self.params.symmetric)
        out = []
        for seg in segs:
            start, end = seg.start / FRAME_RATE, seg.end / FRAME_RATE
            if seg.detail >= NO_CHORD:
                out.append(Chord(start, end, "N", None, None, None))
            else:
                root, quality = seg.detail % 12, QUALITIES[seg.detail // 12]
                bass_pc = seg.bass if seg.bass < 12 else None
                key = next((k["key"] for k in keys if k["start"] <= (start + end) / 2 < k["end"]), keys[-1]["key"] if keys else None)
                out.append(Chord(start, end, chord_to_mir_eval(root, quality, bass_pc), root, quality, bass_pc,
                                 chord_symbol(root, quality, bass_pc, key)))  # fmt: skip
        return out
