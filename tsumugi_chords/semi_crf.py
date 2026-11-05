"""A semi-Markov chord decoder: segments of (root x family, bass) between candidate boundaries on the beat grid, scored
by their frames, their length in beats and where they start in the bar (C8, semi-CRF step 3: decoding only).

``chords.joint_path`` decides frame by frame, so a short chord whose edges are unsure is split over many paths (one
per placement of its edges) and loses to the path that keeps the previous chord.  Here a segment can only start at a
candidate (``candidates``: the no-decoder beats cut in halves, thirds and quarters, plus the peaks of the chord-boundary
output), so a chord has a few placements, and each segment carries the naturalness the frames do not see:

* ``dur``: log P(length in beats) of the labels' segments, by how the segment began (chord change / bass-only change);
* ``pos``: the labels' change rate at the start's position (beat in the bar x place in the beat, ``POSITIONS``) relative
  to how often the grid offers that position, by the same two kinds;
* the chord change's bigram (key-relative, as ``chord_crf``) and the penalties of ``joint_path``.

A segment's score is the sum of its frames' ``w * log P`` (root x family plus bass, as ``joint_path``) with the
chord-boundary costs (``-log b`` where it starts, ``-log(1 - b)`` inside), plus ``gamma_dur * dur + gamma_pos * pos``
and, for a chord change, ``lam * LM - penalty`` (a bass-only change: ``- bass_penalty``).  A segment is at most
``max_beats`` long; a chord that lasts longer goes on in a continuation segment (no penalty, length or position terms),
which may only follow a segment of at least ``long_beats``.  After the search each boundary may move by up to
``refine`` frames to where the frames score best (the grid can sit a frame off the change).

The search runs over candidates (``M`` of them, ~14 a second): for each end, every start within ``max_beats`` and every
state kept (root x family whose probability reaches ``prune`` somewhere in the song, x 12 basses): O(M x L x K x 12).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numba
import numpy as np
import torch

from .chord_crf import N_STATE, NS, beat_position
from .chords import SYMMETRIC, family_probs, merge_symmetric

FPS = 25
DIVISIONS = (2, 3, 4)
SUBS = (0.0, 0.25, 1 / 3, 0.5, 2 / 3, 0.75)  # places in the beat; anything else is "off"
SUB_TOLERANCE = 0.07  # beats
POSITIONS = 4 * (len(SUBS) + 1) + 1  # beat in the bar (0..3) x place in the beat (+ off); last: unknown
DUR_EDGES = np.array([0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 10.0, 14.0])  # beats
DURATIONS = len(DUR_EDGES) + 1  # ~1/4, 1/2, 3/4, 1, 1.5, 2, 3, 4, 6, 8, 12, 16+ beats
NEG = -1e9

# Labelled chords of the training songs by family (``chords.FAMILIES`` order) and the bass's interval above the root
# (1, b2, 2, b3, 3, 4, b5, 5, #5, 6, b7, 7): ``bass_prior``'s table when the model's ``semi_tables`` has none.
BASS_INTERVAL_COUNTS = np.array([
    (335, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 5
    (25657, 6, 1677, 3, 4376, 146, 3, 1351, 1, 8, 427, 28),  # major
    (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # -5
    (6838, 3, 14, 410, 6, 119, 0, 214, 0, 10, 145, 11),  # m
    (678, 0, 2, 40, 3, 0, 9, 2, 0, 4, 19, 0),  # dim
    (1245, 0, 97, 10, 8, 0, 1326, 12, 0, 0, 14, 0),  # aug
    (84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # sus2
    (1707, 52, 50, 49, 15, 0, 1, 17, 28, 19, 47, 0),  # sus4
    (972, 0, 2, 0, 13, 0, 0, 0, 0, 0, 0, 0),  # 6
    (11967, 10, 7, 1, 626, 30, 2, 42, 0, 0, 612, 0),  # 7
    (9541, 0, 632, 0, 25, 2, 0, 54, 0, 0, 0, 1),  # M7
    (368, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 0),  # m6
    (19973, 1, 6, 12, 1, 2091, 1, 44, 0, 0, 51, 4),  # m7
    (405, 0, 34, 5, 0, 70, 0, 18, 0, 2, 0, 0),  # mM7
    (215, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),  # 7-5
    (2122, 0, 2, 7, 0, 159, 105, 0, 0, 0, 24, 0),  # m7-5
    (244, 0, 0, 0, 2, 0, 9, 0, 0, 0, 0, 0),  # aug7
    (43, 0, 42, 0, 0, 0, 5, 0, 0, 0, 1, 0),  # augM7
    (251, 0, 0, 0, 0, 5, 0, 3, 0, 0, 19, 0),  # 7sus4
    (1268, 9, 0, 4, 6, 0, 0, 10, 0, 0, 0, 0),  # dim7
], dtype=np.float64)  # fmt: skip


@dataclass
class Params:
    w: float = 1.0
    penalty: float = 8.0
    bass_penalty: float = 2.0
    nct_penalty: float = 0.0  # extra cost of a bass-only change to a bass outside the held chord's family tones
    bass_prior: float = 0.5  # weight of log P(bass interval | family) (over its likeliest interval) at every segment's start
    symmetric: bool = True  # aug / dim7: the roots that spell one set of notes share their summed probability
    merge_fifth_bass: bool | str = False  # (after the search, chords.merge_fifth_bass) root <-> fifth bass moves: one chord
    short_slash: float = 1.25  # (after the search, chords.merge_short_slash) an opening bass shorter than this many beats
    repeat_beta: float = 0.5  # (before the search, repeat.smooth) how much a bar hears the bars that repeat it; 0: off
    repeat_context: float = 0.9  # ... the similarity the bars around the two must reach
    repeat_local: float = 0.75  # ... and the two bars themselves
    merge_similar: bool = True  # (after the search, chords.merge_similar) one sound split between two readings: joined
    boundary_weight: float = 1.0
    lam: float = 0.0
    gamma_dur: float = 0.0
    gamma_pos: float = 0.0
    peak: float = 0.2  # chord-boundary peaks above this are candidates too
    divisions: tuple = DIVISIONS
    max_beats: float = 16.0
    long_beats: float = 8.0
    max_candidates: int = 200  # a segment spans at most this many candidates
    prune: float = 0.01
    refine: int = 0
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ the beat grid


def extended_beats(beats: np.ndarray, seconds: float) -> np.ndarray:
    """The beats with the median beat repeated before the first and after the last (the grid covers the song)."""

    step = float(np.median(np.diff(beats)))
    before = np.arange(beats[0] - step, -step, -step)[::-1]
    after = np.arange(beats[-1] + step, seconds + step, step)
    return np.r_[before, beats, after]


def candidates(beats: np.ndarray, boundary: np.ndarray, frames: int, params: Params) -> np.ndarray:
    """Candidate boundary frames (0 and ``frames`` included), increasing."""

    if "cand" in params.extra:  # (checks) given candidates
        return np.asarray(params.extra["cand"])
    if params.extra.get("every_frame"):  # (a check: with free continuations this is chords.joint_path)
        return np.arange(frames + 1)
    b = extended_beats(beats, frames / FPS)
    points = [b]
    for n in params.divisions:
        for k in range(1, n):
            points.append(b[:-1] + k / n * np.diff(b))
    out = [np.round(np.concatenate(points) * FPS).astype(np.int64)]
    if params.peak is not None and boundary is not None:
        p = sigmoid_if_logits(boundary)
        inner = (p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] > params.peak)
        out.append(np.nonzero(inner)[0] + 1)
    c = np.unique(np.clip(np.concatenate(out), 0, frames))
    return np.unique(np.r_[0, c[(c > 0) & (c < frames)], frames])


def sigmoid_if_logits(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return 1 / (1 + np.exp(-x)) if x.min() < 0 or x.max() > 1 else x


def bar_beats(beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    """Per beat its place in the bar (0..3, later beats 3; -1 before the first downbeat or without downbeats)."""

    if len(downbeats) == 0:
        return np.full(len(beats), -1)
    first = np.unique(np.abs(downbeats[:, None] - beats[None]).argmin(1))
    index = np.arange(len(beats))
    last = np.searchsorted(first, index, side="right") - 1
    return np.where(last >= 0, np.minimum(index - first[np.maximum(last, 0)], 3), -1)


def positions(u: np.ndarray, in_bar: np.ndarray) -> np.ndarray:
    """``POSITIONS`` index of beat coordinates ``u`` (``beat_position``) given each beat's place in the bar."""

    whole = np.floor(u + 1e-6).astype(np.int64)
    frac = u - whole
    d = np.abs(frac[:, None] - np.r_[SUBS, 1.0][None])
    sub = d.argmin(1)
    near = d.min(1) <= SUB_TOLERANCE
    up = sub == len(SUBS)  # just before the next beat: that beat's start
    whole = np.where(up & near, whole + 1, whole)
    sub = np.where(up, 0, sub)
    sub = np.where(near, sub, len(SUBS))
    inside = (whole >= 0) & (whole < len(in_bar))
    bar = np.where(inside, in_bar[np.clip(whole, 0, len(in_bar) - 1)], -1)
    return np.where(bar >= 0, bar * (len(SUBS) + 1) + sub, POSITIONS - 1)


def duration_bins(beats: np.ndarray) -> np.ndarray:
    return np.digitize(beats, DUR_EDGES)


# ------------------------------------------------------------------ decoding


def decode(detail: np.ndarray, bass: np.ndarray, boundary: np.ndarray, tonic: np.ndarray, beats: np.ndarray,
           downbeats: np.ndarray, params: Params, tables: dict, lm: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """``[T]`` root x family classes (N = ``N_STATE``) and ``[T]`` basses (12 under N) of the best segmentation.
    ``detail`` ``[T, 745]`` / ``bass`` ``[T, 13]`` / ``boundary`` ``[T]`` probabilities; ``tables``: ``dur_chord``,
    ``dur_bass`` ``[DURATIONS]``, ``pos_chord``, ``pos_bass`` ``[POSITIONS]``; ``lm`` ``[NS, NS]`` relative bigram."""

    frames = len(detail)
    family = family_probs(detail)
    if params.symmetric:
        family = merge_symmetric(family)
    keep = np.nonzero(family.max(0) >= params.prune)[0]
    if params.extra.get("keep") is not None:  # (checks) given states
        keep = np.asarray(params.extra["keep"])
    keep = np.union1d(keep, [N_STATE])
    K = len(keep)
    n_idx = K - 1  # N is the last kept state
    chord_lp = params.w * np.log(np.clip(family[:, keep], 1e-6, None)).astype(np.float64)
    bass_lp = params.w * np.log(np.clip(bass[:, :12], 1e-6, None)).astype(np.float64)
    b = np.clip(sigmoid_if_logits(boundary), 1e-4, 1 - 1e-4)
    change = -params.boundary_weight * np.log(b)
    stay = -params.boundary_weight * np.log1p(-b)
    cand = candidates(beats, boundary, frames, params)
    M = len(cand) - 1
    CP = np.vstack([np.zeros((1, K)), np.cumsum(chord_lp, 0)])[cand]  # [M + 1, K]
    BP = np.vstack([np.zeros((1, 12)), np.cumsum(bass_lp, 0)])[cand]  # [M + 1, 12]
    SP = np.r_[0.0, np.cumsum(stay)]  # SP[t] = sum of stay[u] for u < t
    at = np.minimum(cand, frames - 1)
    g_chord = SP[np.minimum(cand + 1, frames)] - np.where(cand > 0, change[at], 0.0)  # how a segment starts at c_i
    g_stay = SP[np.minimum(cand + 1, frames)] - np.where(cand > 0, stay[at], 0.0)
    ext = extended_beats(beats, frames / FPS)
    u = beat_position(ext, cand / FPS)
    in_bar = bar_beats(ext, downbeats) if len(downbeats) else np.full(len(ext), -1)
    pos = positions(u, in_bar)
    pos_c = params.gamma_pos * tables["pos_chord"][pos]
    pos_b = params.gamma_pos * tables["pos_bass"][pos]
    dur_c, dur_b = params.gamma_dur * tables["dur_chord"], params.gamma_dur * tables["dur_bass"]
    # chord-change transitions between kept states, per tonic (key-relative bigram, no self move)
    rel_of = lambda k: np.where(keep == N_STATE, N_STATE, 12 * (keep // 12) + (keep % 12 - k) % 12)  # noqa: E731
    trans = {}
    for k in np.unique(tonic):
        t = np.zeros((K, K)) if lm is None or params.lam == 0 else params.lam * lm[np.ix_(rel_of(k), rel_of(k))]
        if not params.extra.get("allow_self"):  # (joint_path also lets a chord move keep the chord)
            np.fill_diagonal(t, NEG)
        trans[int(k)] = t
    tonic_c = tonic[at]
    tonics = sorted(trans)
    trans_all = np.stack([trans[k] for k in tonics])
    tonic_idx = np.searchsorted(np.array(tonics), tonic_c).astype(np.int64)
    nct = np.zeros((K, 12))
    if params.nct_penalty:  # the family's first quality's tones (e.g. m7 for the m7 family)
        from .chords import QUALITY_FAMILY
        from .theory import chord_tone_templates

        templ = chord_tone_templates().astype(bool)
        for idx, state in enumerate(keep[:-1]):
            q = int(np.flatnonzero(QUALITY_FAMILY == state // 12)[0])
            nct[idx] = params.nct_penalty * ~templ[12 * q + state % 12]
    prior = np.zeros((K, 12))  # the chord-bass pair's prior, paid once per segment (not by a continuation)
    if params.bass_prior:
        for idx, state in enumerate(keep[:-1]):
            fam, root = state // 12, state % 12
            table = tables["bass_interval"][fam]
            step = SYMMETRIC.get(fam)  # a symmetric chord: the bass's best reading over its spellings
            rots = [0] if step is None else [k * step for k in range(12 // step)]
            prior[idx] = params.bass_prior * np.max([np.roll(table, root + r) for r in rots], 0)
    out = _forward(CP, BP, SP[cand], g_chord, g_stay, u, pos_c, pos_b, dur_c, dur_b, DUR_EDGES, trans_all, tonic_idx,
                   float(params.penalty), float(params.bass_penalty), float(params.max_beats), float(params.long_beats),
                   int(params.max_candidates), n_idx, nct, prior)  # fmt: skip
    end, bp_i, bp_type, end_long, bpl_i, bpl_type, prev_k, prev_kb, top1, top2 = out
    if params.extra.get("keep") is not None or params.extra.get("want_score"):
        params.extra["score"] = float(np.where(np.ones((K, 12), dtype=bool), end[M], NEG).max())
    bass_ok = np.ones((K, 12), dtype=bool)
    bass_ok[n_idx, 1:] = False
    # back-trace
    chord_path = np.empty(frames, dtype=np.int64)
    bass_path = np.empty(frames, dtype=np.int64)
    k, bb = np.unravel_index(int(np.where(bass_ok, end[M], NEG).argmax()), (K, 12))
    j, use_long = M, False
    while j > 0:
        i = int((bpl_i if use_long else bp_i)[j, k, bb])
        typ = int((bpl_type if use_long else bp_type)[j, k, bb])
        chord_path[cand[i] : cand[j]], bass_path[cand[i] : cand[j]] = keep[k], bb
        if i == 0:
            break
        if typ == 0:
            k, bb = int(prev_k[i, k]), int(prev_kb[i, k])
            use_long = False
        elif typ == 1:
            bb = int(top2[i, k] if top1[i, k] == bb else top1[i, k])
            use_long = False
        else:
            use_long = True
        j = i
    if params.refine:
        chord_path, bass_path = refine(chord_path, bass_path, family, bass, change, stay, params)
    bass_path[chord_path == N_STATE] = 12
    return chord_path, bass_path


@numba.njit(cache=True)
def _forward(CP, BP, SPc, g_chord, g_stay, u, pos_c, pos_b, dur_c, dur_b, edges, trans, tonic_idx, penalty, bass_penalty,
             max_beats, long_beats, max_cand, n_idx, nct, prior):  # fmt: skip
    """The max-sum forward over candidates (see ``decode``); returns the ends' scores and the back pointers."""

    M = CP.shape[0] - 1
    K = CP.shape[1]
    start_c = np.full((M + 1, K, 12), NEG)  # a segment starting at c_i after a chord change (or the song's start)
    start_b = np.full((M + 1, K, 12), NEG)  # after a bass-only change
    start_n = np.full((M + 1, K, 12), NEG)  # a continuation of a long segment
    end = np.full((M + 1, K, 12), NEG)
    end_long = np.full((M + 1, K, 12), NEG)
    bp_i = np.zeros((M + 1, K, 12), dtype=np.int32)
    bp_type = np.zeros((M + 1, K, 12), dtype=np.int8)
    bpl_i = np.zeros((M + 1, K, 12), dtype=np.int32)
    bpl_type = np.zeros((M + 1, K, 12), dtype=np.int8)
    prev_k = np.zeros((M + 1, K), dtype=np.int32)  # start_c: the previous chord
    prev_kb = np.zeros((M + 1, K), dtype=np.int8)  # ... and its bass
    top1 = np.zeros((M + 1, K), dtype=np.int8)  # start_b: the best and second best bass at c_i
    top2 = np.zeros((M + 1, K), dtype=np.int8)
    for k in range(K):
        for b in range(12 if k != n_idx else 1):
            start_c[0, k, b] = g_chord[0] + prior[k, b]
    best_k = np.empty(K)
    best_kb = np.empty(K, dtype=np.int64)
    for j in range(1, M + 1):
        for i in range(max(0, j - max_cand), j):
            length = u[j] - u[i]
            if length > max_beats and i != j - 1:
                continue
            db = np.searchsorted(edges, length, side="right")
            dc, dbb = dur_c[db], dur_b[db]
            long = length >= long_beats
            for k in range(K):
                ce = CP[j, k] - CP[i, k]
                for b in range(12 if k != n_idx else 1):
                    e = ce + (BP[j, b] - BP[i, b] if k != n_idx else 0.0)
                    v, t = start_c[i, k, b] + dc, 0
                    w = start_b[i, k, b] + dbb
                    if w > v:
                        v, t = w, 1
                    w = start_n[i, k, b]
                    if w > v:
                        v, t = w, 2
                    v += e
                    if v > end[j, k, b]:
                        end[j, k, b], bp_i[j, k, b], bp_type[j, k, b] = v, i, t
                    if long and v > end_long[j, k, b]:
                        end_long[j, k, b], bpl_i[j, k, b], bpl_type[j, k, b] = v, i, t
        for k in range(K):
            for b in range(12):
                end[j, k, b] -= SPc[j]
                end_long[j, k, b] -= SPc[j]
        if j == M:
            break
        # segments starting at c_j
        for k in range(K):
            m, mb, m2, mb2 = NEG, 0, NEG, 0
            for b in range(12 if k != n_idx else 1):
                x = end[j, k, b]
                if x > m:
                    m2, mb2, m, mb = m, mb, x, b
                elif x > m2:
                    m2, mb2 = x, b
            best_k[k], best_kb[k] = m, mb
            top1[j, k], top2[j, k] = mb, mb2
            if k != n_idx:
                for b in range(12):
                    other = m2 if b == mb else m
                    start_b[j, k, b] = other - bass_penalty - nct[k, b] + prior[k, b] + pos_b[j] + g_stay[j]
            for b in range(12 if k != n_idx else 1):
                start_n[j, k, b] = end_long[j, k, b] + g_stay[j]
        tr = trans[tonic_idx[j]]
        for k in range(K):
            m, mk = NEG, 0
            for kp in range(K):
                x = best_k[kp] + tr[kp, k]
                if x > m:
                    m, mk = x, kp
            prev_k[j, k], prev_kb[j, k] = mk, best_kb[mk]
            sc = m - penalty + pos_c[j] + g_chord[j]
            for b in range(12 if k != n_idx else 1):
                start_c[j, k, b] = sc + prior[k, b]
    return end, bp_i, bp_type, end_long, bpl_i, bpl_type, prev_k, prev_kb, top1, top2


def refine(chord_path, bass_path, family, bass, change, stay, params: Params):
    """Move each boundary by up to ``params.refine`` frames to where the two sides' frames score best."""

    lp = params.w * np.log(np.clip(family, 1e-6, None))
    blp = params.w * np.log(np.clip(bass[:, :12], 1e-6, None))
    frames = len(chord_path)
    bounds = np.nonzero((chord_path[1:] != chord_path[:-1]) | (bass_path[1:] != bass_path[:-1]))[0] + 1
    edges = np.r_[0, bounds, frames]
    for n in range(1, len(edges) - 1):
        a, t, e = edges[n - 1], edges[n], edges[n + 1]
        ca, ba, cb, bb_ = chord_path[a], bass_path[a], chord_path[t], bass_path[t]
        score_a = lp[:, ca] + (blp[:, ba] if ca != N_STATE else 0.0)
        score_b = lp[:, cb] + (blp[:, bb_] if cb != N_STATE else 0.0)
        best, best_t = None, t
        for s in range(max(a + 1, t - params.refine), min(e - 1, t + params.refine) + 1):
            v = score_a[a:s].sum() + score_b[s:e].sum() - change[s] + stay[s]
            if best is None or v > best:
                best, best_t = v, s
        if best_t != t:
            chord_path[a:best_t], bass_path[a:best_t] = ca, ba
            chord_path[best_t:e], bass_path[best_t:e] = cb, bb_
            edges[n] = best_t
    return chord_path, bass_path


# ------------------------------------------------------------------ tables from labelled segments


def label_segments(rows: list[dict], family_index, pitch_class) -> list[tuple[float, float, int, int]]:
    """Merge the label rows into (start, end, root x family state, bass) runs."""

    out = []
    for r in rows:
        state, bass = family_index(r), pitch_class(r)
        if out and out[-1][2] == state and out[-1][3] == bass and abs(out[-1][1] - r["start_time"]) < 1e-3:
            out[-1] = (out[-1][0], r["end_time"], state, bass)
        else:
            out.append((r["start_time"], r["end_time"], state, bass))
    return out


def count_tables(songs: list[tuple[list, np.ndarray, np.ndarray, float]], family_index, pitch_class) -> dict:
    """Counts for ``tables`` from (label rows, beats, downbeats, seconds) of training songs."""

    dur = {"chord": np.zeros(DURATIONS), "bass": np.zeros(DURATIONS)}
    hits = {"chord": np.zeros(POSITIONS), "bass": np.zeros(POSITIONS)}
    offered = np.zeros(POSITIONS)
    for rows, beats, downbeats, seconds in songs:
        segs = label_segments(rows, family_index, pitch_class)
        if len(segs) < 2:
            continue
        ext = extended_beats(beats, seconds)
        in_bar = bar_beats(ext, downbeats)
        starts = np.array([s[0] for s in segs])
        ends = np.array([s[1] for s in segs])
        u0, u1 = beat_position(ext, starts), beat_position(ext, ends)
        p = positions(u0, in_bar)
        for n in range(1, len(segs)):
            kind = "chord" if segs[n][2] != segs[n - 1][2] else "bass"
            dur[kind][duration_bins(np.array([u1[n] - u0[n]]))[0]] += 1
            hits[kind][p[n]] += 1
        grid = []
        for d in DIVISIONS:
            for k in range(d):
                grid.append(ext[:-1] + k / d * np.diff(ext))
        grid = np.unique(np.round(np.concatenate(grid) * FPS)) / FPS
        grid = grid[(grid > starts[0]) & (grid < ends[-1])]
        offered += np.bincount(positions(beat_position(ext, grid), in_bar), minlength=POSITIONS)
    return {"dur_chord": dur["chord"], "dur_bass": dur["bass"], "hits_chord": hits["chord"], "hits_bass": hits["bass"],
            "offered": offered}  # fmt: skip


def tables_from_counts(counts: dict, off: float = -2.0) -> dict:
    """``dur_*``: log P(length bin), its best bin 0; ``pos_*``: log of the change rate at a position over the mean
    rate (grid positions; off-grid starts get ``off``, unknown 0); ``bass_interval``: log P(bass interval | family)
    over the family's likeliest interval (``counts["bass_interval"]`` or ``BASS_INTERVAL_COUNTS``)."""

    out = {}
    for kind in ("chord", "bass"):
        d = counts[f"dur_{kind}"] + 1.0
        lp = np.log(d / d.sum())
        out[f"dur_{kind}"] = lp - lp.max()
        hits, offered = counts[f"hits_{kind}"], counts["offered"]
        rate = (hits + 0.5) / (offered + 1.0)
        grid = np.array([(i % (len(SUBS) + 1)) < len(SUBS) for i in range(POSITIONS - 1)] + [False])
        mean = hits[grid].sum() / max(offered[grid].sum(), 1)
        pos = np.where(grid, np.log(rate / mean), off)
        pos[-1] = 0.0
        out[f"pos_{kind}"] = pos
    c = np.asarray(counts.get("bass_interval", BASS_INTERVAL_COUNTS), dtype=np.float64)  # [families, 12]
    lp = np.log((c + 20.0 * (c.sum(0) + 1.0) / (c.sum() + 12.0)) / (c.sum(1, keepdims=True) + 20.0))  # towards all families'
    out["bass_interval"] = lp - lp.max(1, keepdims=True)
    return out


# ------------------------------------------------------------------ training: the semi-CRF's likelihood (torch)

GRID_TWELFTHS = (0, 3, 4, 6, 8, 9)  # the subdivisions 1/2, 1/3, 1/4 of a beat in twelfths


def frame_beats(beats: np.ndarray, downbeats: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per frame its beat coordinate (``beat_position`` on the extended grid) and ``POSITIONS`` index."""

    ext = extended_beats(beats, frames / FPS)
    u = beat_position(ext, np.arange(frames) / FPS)
    return u.astype(np.float32), positions(u, bar_beats(ext, downbeats)).astype(np.int64)


def crop_candidates(u: np.ndarray, boundary: np.ndarray, peak: float | None) -> np.ndarray:
    """Candidates of a crop from its frames' beat coordinates ``u`` (the 1/2, 1/3, 1/4 grid) and boundary peaks."""

    frames = len(u)
    half = np.r_[u[0] - (u[1] - u[0]) / 2, (u[1:] + u[:-1]) / 2, u[-1] + (u[-1] - u[-2]) / 2] * 12  # frame edges
    a, b = np.floor(half[:-1]), np.floor(half[1:])
    crossed = (b > a) & np.isin(b.astype(np.int64) % 12, GRID_TWELFTHS)
    out = [np.nonzero(crossed)[0]]
    if peak is not None:
        inner = (boundary[1:-1] >= boundary[:-2]) & (boundary[1:-1] > boundary[2:]) & (boundary[1:-1] > peak)
        out.append(np.nonzero(inner)[0] + 1)
    c = np.unique(np.concatenate(out))
    return np.r_[0, c[(c > 0) & (c < frames)], frames]


class SemiCRF(torch.nn.Module):
    """The decoder's model as a CRF over segmentations, for a training loss: ``loss`` = log Z - log Z(gold) per frame.

    Learnt: the length and position tables (started at ``gamma_dur`` / ``gamma_pos`` x the label counts' tables), a
    correction ``delta`` of the key-relative bigram (started at 0), and the scalars ``penalty``, ``bass_penalty``,
    ``boundary_weight``, ``lam`` (and ``w`` with ``learn_w``; otherwise the frames' weight stays fixed so that the
    likelihood cannot just shrink the frames' evidence).  An L2 penalty (``reg``) keeps the tables near their start.
    The gold path is the labels per candidate interval (their middle frame), so a change between candidates moves to
    the nearest one."""

    def __init__(self, tables: dict, lm: np.ndarray | None, init: dict | None = None, learn_w: bool = False,
                 max_beats: float = 8.0, long_beats: float = 4.0, max_candidates: int = 64, prune: float = 0.03,
                 peak: float | None = 0.2):  # fmt: skip
        super().__init__()
        init = {"w": 1.0, "penalty": 8.0, "bass_penalty": 2.0, "boundary_weight": 1.0, "lam": 1.0, "gamma_dur": 2.0,
                "gamma_pos": 1.0} | (init or {})  # fmt: skip
        inv = lambda x: float(np.log(np.expm1(max(x, 1e-3))))  # noqa: E731
        names = ["penalty", "bass_penalty", "boundary_weight", "lam"] + (["w"] if learn_w else [])
        self.raw = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(inv(init[k]))) for k in names})
        self.fixed = {} if learn_w else {"w": float(init["w"])}
        for k in ("dur_chord", "dur_bass", "pos_chord", "pos_bass"):
            start = torch.tensor(np.asarray(tables[k], dtype=np.float32) * init["gamma_" + k[:3]])
            self.register_parameter(k, torch.nn.Parameter(start.clone()))
            self.register_buffer(k + "_init", start)
        lm = np.zeros((NS, NS), dtype=np.float32) if lm is None else lm.astype(np.float32)
        np.fill_diagonal(lm, 0.0)
        self.register_buffer("lm", torch.from_numpy(lm))
        self.delta = torch.nn.Parameter(torch.zeros(NS, NS))
        self.register_buffer("edges", torch.tensor(DUR_EDGES, dtype=torch.float32))
        self.max_beats, self.long_beats, self.max_candidates = max_beats, long_beats, max_candidates
        self.prune, self.peak = prune, peak
        self.block = 32  # candidates per checkpointed block of the forward
        self.compile = False
        self.backend = "triton"  # or "torch" (the reference)
        self._compiled = None

    def value(self, name: str) -> torch.Tensor:
        if name in self.fixed:
            return torch.tensor(self.fixed[name], device=self.lm.device)
        return torch.nn.functional.softplus(self.raw[name])

    def values(self) -> dict[str, float]:
        return {k: round(float(self.value(k)), 4) for k in list(self.raw) + list(self.fixed)}

    def reg(self) -> torch.Tensor:
        out = self.delta.square().sum()
        for k in ("dur_chord", "dur_bass", "pos_chord", "pos_bass"):
            out = out + (getattr(self, k) - getattr(self, k + "_init")).square().sum()
        return out

    def tables(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k).detach().cpu().numpy() for k in ("dur_chord", "dur_bass", "pos_chord", "pos_bass", "delta")}

    def _step(self):
        if getattr(self, "_compiled", None) is None:
            self._compiled = torch.compile(_semi_step, dynamic=True) if self.compile else _semi_step
        return self._compiled

    def _prepare(self, detail, boundary, gold_detail, gold_bass, tonic, ubeat, spos):
        """Per crop (numpy, detached): candidates, kept states, gold per candidate interval; then padded tensors."""

        from .chord_crf import GROUP

        dev = detail.device
        B, T = tonic.shape
        prob = torch.softmax(detail.detach().float(), -1)
        fam = torch.zeros(B, T, NS, device=dev).index_add_(-1, torch.as_tensor(GROUP, device=dev), prob)
        fam_max = fam.max(1).values.cpu().numpy()
        bnd = torch.sigmoid(boundary.detach().float()).cpu().numpy()
        group = np.asarray(GROUP)
        gd, gb = gold_detail.cpu().numpy(), gold_bass.cpu().numpy()
        u_all, pos_all = ubeat.cpu().numpy(), spos.cpu().numpy()
        crops = []
        for n in range(B):
            cand = crop_candidates(u_all[n], bnd[n], self.peak)
            mid = np.minimum((cand[:-1] + cand[1:]) // 2, T - 1)
            g_state = np.where(gd[n, mid] >= 0, group[np.clip(gd[n, mid], 0, None)], -1)
            g_bass = np.where((g_state >= 0) & (g_state != N_STATE) & (gb[n, mid] >= 0) & (gb[n, mid] < 12), gb[n, mid], -1)
            keep = np.union1d(np.nonzero(fam_max[n] >= self.prune)[0], g_state[g_state >= 0])
            keep = np.union1d(keep, [N_STATE])
            crops.append((cand, keep, g_state, g_bass))
        M1 = max(len(c[0]) for c in crops)
        K = max(len(c[1]) for c in crops)
        cand_t = np.full((B, M1), T, dtype=np.int64)
        keep_t = np.zeros((B, K), dtype=np.int64)
        kmask = np.zeros((B, K), dtype=bool)
        ncand = np.zeros(B, dtype=np.int64)
        disallow = np.zeros((B, M1 - 1, K, 12), dtype=np.float32)  # gold pass: intervals x states other than the gold
        for n, (cand, keep, g_state, g_bass) in enumerate(crops):
            m = len(cand)
            cand_t[n, :m], ncand[n] = cand, m - 1
            keep_t[n, : len(keep)], kmask[n, : len(keep)] = keep, True
            for i in np.nonzero(g_state >= 0)[0]:
                row = np.ones((K, 12), dtype=np.float32)
                k = int(np.searchsorted(keep, g_state[i]))
                if g_bass[i] >= 0:
                    row[k, g_bass[i]] = 0.0
                else:
                    row[k, :] = 0.0
                disallow[n, i] = row
        t = lambda x, dt=None: torch.as_tensor(x, device=dev, dtype=dt)  # noqa: E731
        cand_t, keep_t, kmask, ncand = t(cand_t), t(keep_t), t(kmask), t(ncand)
        at = cand_t.clamp(max=T - 1)
        u = torch.gather(t(u_all), 1, at)
        beyond = (torch.arange(M1, device=dev)[None] - ncand[:, None]).clamp(min=0).float()
        u = u + 1e4 * beyond  # padded candidates: too far to join
        pos = torch.gather(t(pos_all), 1, at)
        ton = torch.gather(tonic, 1, at)
        known = torch.cat((torch.zeros(B, 1, K, 12, device=dev), torch.cumsum(t(disallow), 1)), 1)  # [B, M1, K, 12]
        return cand_t, keep_t, kmask, ncand, u, pos, ton, known

    def loss(self, detail, bass, boundary, gold_detail, gold_bass, tonic, ubeat, spos, return_z: bool = False) -> torch.Tensor:
        """Mean over the crops of ``(log Z - log Z(gold)) / frames``; logits ``detail`` ``[B, T, 745]``, ``bass``
        ``[B, T, 13]``, ``boundary`` ``[B, T]``; ``gold_*`` as ``chord_crf``; ``ubeat`` / ``spos`` from
        ``frame_beats``."""

        from .chord_crf import GROUP

        dev = detail.device
        B, T = tonic.shape
        cand, keep, kmask, ncand, u, pos, ton, known = self._prepare(detail, boundary, gold_detail, gold_bass, tonic, ubeat, spos)
        M1, K = cand.shape[1], keep.shape[1]
        NEGT = -1e7
        w = self.value("w")
        prob = torch.softmax(detail.float(), -1)
        fam = torch.zeros(B, T, NS, device=dev).index_add_(-1, torch.as_tensor(GROUP, device=dev), prob)
        chord = w * torch.log(torch.gather(fam, 2, keep[:, None].expand(B, T, K)).clamp_min(1e-6))  # [B, T, K]
        blp = w * torch.log_softmax(bass[..., :12].float(), -1)  # [B, T, 12]
        CP = torch.cat((torch.zeros(B, 1, K, device=dev), torch.cumsum(chord, 1)), 1)
        CP = torch.gather(CP, 1, cand[..., None].expand(B, M1, K))
        BP = torch.cat((torch.zeros(B, 1, 12, device=dev), torch.cumsum(blp, 1)), 1)
        BP = torch.gather(BP, 1, cand[..., None].expand(B, M1, 12))
        bw = self.value("boundary_weight")
        p = torch.sigmoid(boundary.float()).clamp(1e-4, 1 - 1e-4)
        change, stay = -bw * torch.log(p), -bw * torch.log1p(-p)
        SP = torch.cat((torch.zeros(B, 1, device=dev), torch.cumsum(stay, 1)), 1)  # [B, T + 1]
        at = cand.clamp(max=T - 1)
        nxt = (cand + 1).clamp(max=T)
        SPc = torch.gather(SP, 1, cand)
        zero = torch.zeros_like(SPc)
        g_chord = torch.gather(SP, 1, nxt) - torch.where(cand > 0, torch.gather(change, 1, at), zero)
        g_stay = torch.gather(SP, 1, nxt) - torch.where(cand > 0, torch.gather(stay, 1, at), zero)
        pos_c, pos_b = self.pos_chord[pos], self.pos_bass[pos]  # [B, M1]
        # transitions per candidate between the kept states (key-relative bigram + delta), no self move
        kk = keep[:, None]
        rel = torch.where(kk == N_STATE, N_STATE, 12 * (kk // 12) + (kk % 12 - ton[..., None]) % 12)  # [B, M1, K]
        score = self.value("lam") * self.lm + self.delta
        trans = score[rel[..., :, None], rel[..., None, :]]  # [B, M1, K, K]
        bad = (~kmask[:, None, :, None]) | (~kmask[:, None, None, :]) | torch.eye(K, dtype=torch.bool, device=dev)[None, None]
        trans = trans.masked_fill(bad, NEGT)
        is_n = keep == N_STATE  # [B, K]
        bass_ok = kmask[..., None] & ~(is_n[..., None] & (torch.arange(12, device=dev) > 0))  # [B, K, 12]
        has_bass = (~is_n).float()[..., None]  # N has no bass
        # both passes at once: [full; gold]
        two = lambda x: torch.cat((x, x))  # noqa: E731
        CP2, BP2, SPc2, gc2, gs2, u2, pc2, pb2, tr2 = map(two, (CP, BP, SPc, g_chord, g_stay, u, pos_c, pos_b, trans))
        KN2 = torch.cat((torch.zeros_like(known), known)) * NEGT
        ok2, hb2 = two(bass_ok), two(has_bass)
        can_b = ok2 & hb2.bool()
        penalty, bass_penalty = self.value("penalty"), self.value("bass_penalty")
        L = self.max_candidates
        # windows of the L candidates before each j (front-padded), all j at once: [2B, M1, L, ...]
        pad = lambda x, v=0.0: torch.cat((torch.full((x.shape[0], L) + x.shape[2:], v, device=dev, dtype=x.dtype), x), 1)  # noqa: E731
        win = lambda x: pad(x).unfold(1, L, 1)[:, : x.shape[1]].movedim(-1, 2)  # noqa: E731
        uw = pad(u2, -1e9).unfold(1, L, 1)[:, :M1]  # [2B, M1, L]
        length = u2[..., None] - uw
        okw = (length <= self.max_beats) | (torch.arange(L, device=dev) == L - 1)
        longw = okw & (length >= self.long_beats)
        db = torch.bucketize(length.contiguous(), self.edges, right=True).clamp(max=DURATIONS - 1)
        dur3 = torch.stack((self.dur_chord[db], self.dur_bass[db], torch.zeros_like(length)), -1).masked_fill(~okw[..., None], NEGT)
        CPw, BPw, KNw = win(CP2), win(BP2), win(KN2)  # [2B, M1, L, K] / [.., 12] / [.., K, 12]
        if self.backend == "triton" and dev.type == "cuda":
            from . import semi_crf_triton as st

            KP = max(16, 1 << (K - 1).bit_length())
            padk = lambda x, v=0.0, dim=2: torch.nn.functional.pad(x, [0, 0] * (x.dim() - 1 - dim) + [0, KP - K], value=v)  # noqa: E731
            pad16 = lambda x, v=0.0: torch.nn.functional.pad(x, [0, 4], value=v)  # noqa: E731
            CPp = padk(CP2)
            BPp = pad16(BP2)
            KNp = pad16(padk(KN2))
            TRp = torch.nn.functional.pad(tr2, [0, KP - K, 0, KP - K], value=NEGT)
            OKp = pad16(padk(ok2.float(), dim=1)) > 0
            CANBp = pad16(padk(can_b.float(), dim=1)) > 0
            HBp = torch.nn.functional.pad(hb2[..., 0], [0, KP - K])
            z = st.log_z(CPp, BPp, SPc2, gc2, gs2, pc2 - penalty, pb2 - bass_penalty, dur3, TRp, 0.0, 0.0, KNp, longw,
                         OKp, CANBp, HBp, two(ncand))  # fmt: skip
            if return_z:
                return z.view(2, B)
            return ((z[:B] - z[B:]) / T).mean()
        neg = torch.full((2 * B, K, 12), NEGT, device=dev)
        last_j = two(ncand)  # each crop's last candidate
        step = self._step()

        def block(j0, j1, hist, final, CP2, BP2, KN2, CPw, BPw, KNw, SPc2, gc2, gs2, pc2, pb2, tr2, dur3, penalty, bass_penalty):
            """Candidates j0..j1-1: the starts there and the crops' final ends (``hist``: the L starts before j0)."""

            rows = list(hist.unbind(1))
            for j in range(j0, j1):
                new, end = step(torch.stack(rows[-L:], 1), dur3[:, j], longw[:, j], CP2[:, j], CPw[:, j], BP2[:, j],
                                BPw[:, j], KN2[:, j], KNw[:, j], SPc2[:, j], tr2[:, j], pc2[:, j], pb2[:, j], gc2[:, j],
                                gs2[:, j], penalty, bass_penalty, ok2, can_b, hb2)  # fmt: skip
                final = torch.where((last_j == j)[:, None, None], end, final)
                rows.append(new)
            return torch.stack(rows[-(j1 - j0) :], 1), final

        from torch.utils.checkpoint import checkpoint

        first = torch.stack((torch.where(ok2, gc2[:, 0, None, None].expand(2 * B, K, 12), neg), neg, neg), 1)  # [2B, 3, K, 12]
        starts = [torch.full_like(first, NEGT)] * (L - 1) + [first]
        final = neg
        for j0 in range(1, M1, self.block):
            j1 = min(M1, j0 + self.block)
            args = (torch.stack(starts[-L:], 1), final, CP2, BP2, KN2, CPw, BPw, KNw, SPc2, gc2, gs2, pc2, pb2, tr2, dur3,
                    penalty, bass_penalty)  # fmt: skip
            if torch.is_grad_enabled():
                new, final = checkpoint(lambda *a, j0=j0, j1=j1: block(j0, j1, *a), *args, use_reentrant=False)
            else:
                new, final = block(j0, j1, *args)
            starts.extend(new.unbind(1))
        z = torch.logsumexp(final.flatten(1), -1)
        if return_z:  # (checks) log Z and log Z(gold) per crop
            return z.view(2, B)
        return ((z[:B] - z[B:]) / T).mean()


NEG_STEP = -1e7


def _semi_step(hist, dur3, longw, CPj, CPw, BPj, BPw, KNj, KNw, SPj, trj, pcj, pbj, gcj, gsj, penalty, bass_penalty, ok2,
               can_b, hb2):  # fmt: skip
    """One candidate j of ``SemiCRF.loss``'s forward: from the L previous starts ``hist`` ``[2B, L, 3, K, 12]`` (chord
    change, bass-only change, continuation) to the ends at j and the starts there ``[2B, 3, K, 12]``."""

    emis = (CPj[:, None] - CPw)[..., None] + hb2[:, None] * (BPj[:, None] - BPw)[:, :, None, :] + (KNj[:, None] - KNw)
    tot = hist + dur3[..., None, None] + emis[:, :, None]  # [2B, L, 3, K, 12]
    end = torch.logsumexp(tot.flatten(1, 2), 1) - SPj[:, None, None]
    end = torch.where(ok2, end, NEG_STEP)
    end_long = torch.logsumexp(tot.masked_fill(~longw[..., None, None, None], NEG_STEP).flatten(1, 2), 1) - SPj[:, None, None]
    per_k = torch.logsumexp(end, -1)  # [2B, K]
    moved = torch.logsumexp(per_k[..., :, None] + trj, 1)  # [2B, K]
    new_c = torch.where(ok2, (moved - penalty + pcj[:, None] + gcj[:, None])[..., None], NEG_STEP)
    others = per_k[..., None] + torch.log((-torch.expm1((end - per_k[..., None]).clamp(max=-1e-6))).clamp_min(1e-30))
    new_b = torch.where(can_b, others - bass_penalty + pbj[:, None, None] + gsj[:, None, None], NEG_STEP)
    new_n = torch.where(ok2, end_long + gsj[:, None, None], NEG_STEP)
    return torch.stack((new_c, new_b, new_n), 1), end
