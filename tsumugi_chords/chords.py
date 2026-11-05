"""Chord segments from per-frame chord probabilities: root, family and bass decoded together, one quality per chord.

A chord's quality falls into three levels (``theory.QUALITIES``):

* the **family** drops the tensions (``m7(9)``, ``m7(11)`` -> ``m7``; ``add9`` -> major, ``madd9`` -> ``m``, ``69`` -> ``6``,
  ``m69`` -> ``m6``): ``FAMILIES``, 20 of them;
* the **triad** drops the sixth and the seventh as well (``m7``, ``m6``, ``mM7`` -> ``m``; ``7``, ``M7``, ``6`` -> major;
  ``m7-5``, ``dim7`` -> ``dim``; ``7sus4`` -> ``sus4``): ``TRIADS``.

``joint_path`` is a Viterbi over the states (root x family, bass) plus N: 12 x 20 x 12 + 1.  A frame's score is the
log-probability of the root x family (the qualities of the family summed, so that a chord whose probability is split
between ``m7`` and ``m7(9)`` stays strong) plus ``bass_weight`` x that of the bass, times ``weight`` (frames' posteriors
are correlated; below 1 they are not counted as independent evidence).  Only three moves are allowed, so a step costs
O(states) and not O(states^2):

* stay (the same chord and bass): ``stay[t]``;
* change the bass only (C -> C/B): ``stay[t] + bass_penalty``;
* change the chord: ``penalty + change[t]`` from the best state of the previous frame.

``change`` / ``stay`` carry where chords tend to change (``boundary_costs``: a chord-boundary probability).  N has one
state (no bass).

``segments`` then picks one quality per segment of the path: the quality of its family with the highest mean share of the
family's probability over the segment's frames.  It also reports how sure that is: ``family_share`` (the family's mean
share of its triad, e.g. ``m7`` against ``m``, ``m6``, ``mM7``) and ``quality_share`` (the quality's share of its family),
so a reading can be left at the triad or the family where the frames do not decide it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np

from .theory import QUALITIES

_FAMILY_RENAME = {"add9": "", "madd9": "m", "69": "6", "m69": "m6"}
_TRIAD_OF = {"6": "", "7": "", "M7": "", "7-5": "-5", "7sus4": "sus4", "aug7": "aug", "augM7": "aug", "dim7": "dim",
             "m7-5": "dim", "m6": "m", "m7": "m", "mM7": "m"}  # fmt: skip


def family_of(quality: str) -> str:
    base = re.sub(r"\(.*\)", "", quality)
    return _FAMILY_RENAME.get(base, base)


def triad_of(family: str) -> str:
    return _TRIAD_OF.get(family, family)


FAMILIES = tuple(dict.fromkeys(family_of(q) for q in QUALITIES))
TRIADS = tuple(dict.fromkeys(triad_of(f) for f in FAMILIES))
QUALITY_FAMILY = np.array([FAMILIES.index(family_of(q)) for q in QUALITIES])  # [62] family of each quality
FAMILY_TRIAD = np.array([TRIADS.index(triad_of(f)) for f in FAMILIES])  # [20] triad of each family
NO_CHORD = 12 * len(QUALITIES)  # the detail classes' N (12 * quality + root, N last)


def detail_groups(level: str) -> np.ndarray:
    """``[745]`` the group of each root x quality class at ``level`` ("family" or "triad"): ``12 * group + root``, N last."""

    group = QUALITY_FAMILY if level == "family" else FAMILY_TRIAD[QUALITY_FAMILY]
    count = (len(FAMILIES) if level == "family" else len(TRIADS)) * 12
    classes = np.arange(NO_CHORD)
    return np.r_[12 * group[classes // 12] + classes % 12, count]


def family_probs(detail: np.ndarray) -> np.ndarray:
    """``[T, 12 * families + 1]`` probabilities of root x family from ``[T, 745]`` root x quality ones."""

    return detail.astype(np.float32) @ one_hot(detail_groups("family"))


# Chords whose notes are the same from several roots: aug (Caug = Eaug = G#aug) and dim7 (Cdim7 = D#dim7 = F#dim7 = Adim7).
SYMMETRIC = {FAMILIES.index("aug"): 4, FAMILIES.index("dim7"): 3}  # family -> the root step that keeps its notes


def equivalent_roots(family: int, root: int) -> list[int]:
    """The roots that spell the same notes as ``root`` in ``family`` (just ``root`` for a chord that is not symmetric)."""

    step = SYMMETRIC.get(family)
    return [root] if step is None else [(root + k * step) % 12 for k in range(12 // step)]


def merge_symmetric(family: np.ndarray) -> np.ndarray:
    """``[T, 12 * families + 1]`` root x family probabilities with each symmetric chord's roots summed: every spelling of
    one set of notes gets their total, so the frames cannot switch between spellings (Baug -> D#aug) of one sound."""

    out = family.copy()
    for fam, step in SYMMETRIC.items():
        cols = family[:, 12 * fam : 12 * fam + 12]
        out[:, 12 * fam : 12 * fam + 12] = sum(np.roll(cols, k * step, axis=1) for k in range(12 // step))
    return out


def one_hot(groups: np.ndarray) -> np.ndarray:
    out = np.zeros((len(groups), groups.max() + 1), dtype=np.float32)
    out[np.arange(len(groups)), groups] = 1.0
    return out


def boundary_costs(boundary: np.ndarray, weight: float) -> tuple[np.ndarray, np.ndarray]:
    """``change`` / ``stay`` costs from a chord-boundary probability per frame."""

    b = np.clip(boundary.astype(np.float32), 1e-4, 1 - 1e-4)
    return -weight * np.log(b), -weight * np.log(1 - b)


def joint_path(
    family: np.ndarray, bass: np.ndarray, penalty: float, bass_penalty: float, weight: float = 1.0,
    bass_weight: float = 1.0, change: np.ndarray | None = None, stay: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The best (root x family, bass) path: ``[T]`` classes ``12 * family + root`` (N = ``12 * len(FAMILIES)``) and
    ``[T]`` bass pitch classes (12 under N).  ``family`` ``[T, 241]`` and ``bass`` ``[T, 12+]`` (C..B, N ignored) are
    probabilities."""

    frames, chords = family.shape
    n = chords - 1
    zeros = np.zeros(frames, dtype=np.float32)
    change = zeros if change is None else change
    stay = zeros if stay is None else stay
    chord_lp = weight * np.log(np.clip(family, 1e-6, None))
    bass_lp = weight * bass_weight * np.log(np.clip(bass[:, :12], 1e-6, None))
    emit = chord_lp[:, :, None] + bass_lp[:, None, :]  # [T, chords, 12]
    emit[:, n, 1:] = -np.inf  # N has one state
    emit[:, n, 0] = chord_lp[:, n]
    choice = np.zeros((frames, chords, 12), dtype=np.uint8)  # 0 stay, 1 bass change, 2 chord change
    bass_from = np.zeros((frames, chords), dtype=np.int8)  # the best bass of each chord at t - 1
    chord_from = np.zeros(frames, dtype=np.int64)  # the best state at t - 1 (flat index)
    score = emit[0].copy()
    for t in range(1, frames):
        best_bass = score.max(1)
        bass_from[t] = score.argmax(1)
        flat = int(score.argmax())
        chord_from[t] = flat
        hold = score - stay[t]
        moved_bass = (best_bass - stay[t] - bass_penalty)[:, None]
        moved_chord = score.flat[flat] - penalty - change[t]
        new = hold
        pick = np.zeros(score.shape, dtype=np.uint8)
        better = moved_bass > new
        new, pick = np.where(better, moved_bass, new), np.where(better, 1, pick)
        better = moved_chord > new
        new, pick = np.where(better, moved_chord, new), np.where(better, 2, pick)
        choice[t] = pick
        score = new + emit[t]
    chord_path = np.empty(frames, dtype=np.int64)
    bass_path = np.empty(frames, dtype=np.int64)
    c, b = np.unravel_index(int(score.argmax()), score.shape)
    for t in range(frames - 1, -1, -1):
        chord_path[t], bass_path[t] = c, b
        if t == 0:
            break
        move = choice[t, c, b]
        if move == 1:
            b = int(bass_from[t, c])
        elif move == 2:
            c, b = np.unravel_index(int(chord_from[t]), score.shape)
    bass_path[chord_path == n] = 12
    return chord_path, bass_path


@dataclass
class Segment:
    start: int  # first frame
    end: int  # one past the last frame
    detail: int  # root x quality class (12 * quality + root), NO_CHORD for N
    bass: int  # pitch class, 12 for N
    family_share: float  # mean share of the family within its triad over the frames (1 for N)
    quality_share: float  # mean share of the quality within its family


def segments(detail: np.ndarray, chord_path: np.ndarray, bass_path: np.ndarray, symmetric: bool = False) -> list[Segment]:
    """The runs of ``joint_path``'s path, each with one quality (``detail`` ``[T, 745]``: root x quality
    probabilities); with ``symmetric`` an aug or dim7 run is spelt from the root its frames favour (``SYMMETRIC``)."""

    detail = detail.astype(np.float32)
    family = family_probs(detail)
    n_family = family.shape[1] - 1
    triad_groups = np.r_[12 * FAMILY_TRIAD[np.arange(n_family) // 12] + np.arange(n_family) % 12, 12 * len(TRIADS)]
    triad = family @ one_hot(triad_groups)
    out = []
    edges = np.flatnonzero((np.diff(chord_path) != 0) | (np.diff(bass_path) != 0)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(chord_path)]):
        c = int(chord_path[a])
        if c == n_family:
            out.append(Segment(int(a), int(b), NO_CHORD, 12, 1.0, 1.0))
            continue
        root, fam = c % 12, c // 12
        roots = equivalent_roots(fam, root) if symmetric else [root]
        if len(roots) > 1:  # a symmetric chord: the spelling the frames give the most probability over the segment
            root = roots[int(family[a:b, [12 * fam + r for r in roots]].sum(0).argmax())]
            c = 12 * fam + root
        members = np.flatnonzero(QUALITY_FAMILY == fam)
        fam_p = np.clip(family[a:b, c], 1e-9, None)
        shares = (detail[a:b][:, 12 * members + root] / fam_p[:, None]).mean(0)
        best = int(shares.argmax())
        family_share = float((fam_p / np.clip(triad[a:b, triad_groups[c]], 1e-9, None)).mean())
        out.append(Segment(int(a), int(b), int(12 * members[best] + root), int(bass_path[a]), family_share, float(shares[best])))
    return out


def merge_similar(segs: list[Segment], detail: np.ndarray, symmetric: bool = False) -> list[Segment]:
    """Neighbouring segments over the same bass whose notes (tensions and bass included) are a subset of one another
    but whose roots differ (Baug/F -> F7(9,#11,13)) joined into one: the labels next to never change so (11 of 98k
    changes in the training songs), the frames do when they split one sound between its readings.  The joined segment
    keeps the reading with the more probability over both segments' frames."""

    from .theory import chord_tone_templates

    tones = chord_tone_templates().astype(bool)

    def notes(seg: Segment) -> set[int]:
        return set(np.flatnonzero(tones[seg.detail]).tolist()) | {seg.bass}

    def weight(seg: Segment, a: int, b: int) -> float:
        quality, root = seg.detail // 12, seg.detail % 12
        roots = equivalent_roots(int(QUALITY_FAMILY[quality]), root) if symmetric else [root]
        return float(detail[a:b, [12 * quality + r for r in roots]].sum())

    out: list[Segment] = []
    for seg in segs:
        prev = out[-1] if out else None
        if (prev is not None and NO_CHORD not in (prev.detail, seg.detail) and prev.bass == seg.bass
                and prev.detail % 12 != seg.detail % 12 and (notes(prev) <= notes(seg) or notes(seg) <= notes(prev))):  # fmt: skip
            keep = prev if weight(prev, prev.start, seg.end) >= weight(seg, prev.start, seg.end) else seg
            out[-1] = replace(keep, start=prev.start, end=seg.end)
        else:
            out.append(seg)
    return out


def merge_fifth_bass(segs: list[Segment], keep: str = "first") -> list[Segment]:
    """A bass-only change between the chord's root and its fifth (B/F# -> B, C -> C/G) dropped: the bass going back and
    forth between them is one chord in the labels (5 -> 1 is 43 of 98k labelled changes, 1 -> 5 fewer still), so the
    segment keeps the bass it started with."""

    out: list[Segment] = []
    for seg in segs:
        prev = out[-1] if out else None
        if prev is not None and prev.detail == seg.detail and seg.detail < NO_CHORD:
            root = seg.detail % 12
            if {(prev.bass - root) % 12, (seg.bass - root) % 12} == {0, 7}:
                bass = {"first": prev.bass, "longer": prev.bass if prev.end - prev.start >= seg.end - seg.start else seg.bass,
                        "root": root}[keep]  # (checks)
                out[-1] = replace(prev, end=seg.end, bass=bass)
                continue
        out.append(seg)
    return out
