"""A linear-chain CRF over (root x family, bass): ``chords.joint_path`` made trainable (C8 stage 3).

States are those of ``chords.joint_path``: ``12 * family + root`` (N last) times the bass (N has one).  A frame's score
is ``w * log P(root x family) + w_bass * log P(bass)`` from the model's ``detail`` and ``bass`` outputs.  Three moves:

* stay: ``- stay[t]``;
* change the bass only: ``- stay[t] - bass_penalty - beta_bass * phase_bass[t]``;
* change the chord: ``lam * LM[p, c] - penalty - change[t] - beta * phase_chord[t]``, any bass.

``change`` / ``stay`` are ``-boundary_weight * log(b)`` / ``-boundary_weight * log(1 - b)`` of the chord-boundary output;
``phase_*`` are ``-log(h(t) / mean h)`` of the beat-phase change rates (``frame_hazard``: the training labels' changes
by position in the beat, from SheetSage2-Refined's no-decoder beats); ``LM`` is a bigram over chord changes in states
relative to the key's parent major (``key_frames``, no-decoder keys), counted on the teacher's segments and the
labels.  The moves exclude staying put (a bass move changes the bass, a chord move the chord), so every state sequence
has one path and the likelihood is exact.  The CRF runs in key-relative states, so the bigram is one fixed matrix; at a
key change the forward variables are re-rotated.

With ``tables`` the CRF also learns what post-processing cannot fix by hand: a cost of a chord change and of a
bass-only change by position (``pos``: the beat in the bar, 0..3 with later beats as 3, times ``POS_BINS`` positions in
the beat; one entry for frames outside the beats or before the first downbeat), and a correction ``delta`` of the
key-relative transition scores (``lam * LM + delta``).  They start at 0 and an L2 penalty (``reg``) keeps rarely seen
entries there.

``loss`` is the negative log-likelihood of the frames' known (state, bass): ``log Z - log Z(known)``, frames with an
unknown chord left free and a known chord with an unknown bass free in the bass.  ``decode`` is the matching Viterbi on
whole songs (numpy).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .chords import FAMILIES, detail_groups

FAM = len(FAMILIES)
NS = 12 * FAM + 1  # chord states, N last
N_STATE = NS - 1
GROUP = detail_groups("family")  # [745] detail class -> state
NEG = -1e4
BINS = 48
POS_BINS = 12  # positions in the beat of the learnt tables
POS = 4 * POS_BINS + 1  # beat in the bar x position in the beat; the last entry: unknown
SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PARENT = {"major": 0, "minor": 3, "harmonic_minor": 3, "dorian": -2, "phrygian": -4, "lydian": -5, "mixolydian": -7,
          "locrian": 1, "phrygian_dominant": -4}  # fmt: skip


# ------------------------------------------------------------------ song-level inputs (keys, beats)


def read_song(path: Path) -> dict | None:
    """A song's no-decoder keys and beats (``{"duration", "keys", "beats", "downbeats"}``), or None."""

    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return d if len(d.get("beats", [])) >= 4 else None


def key_frames(keys: list[dict], frames: int) -> np.ndarray:
    """Per frame the parent major's tonic (0..11)."""

    out = np.zeros(frames, dtype=np.int64)
    end = 0
    for k in keys:
        tonic, _, mode = k["key"].partition(":")
        a, b = int(round(k["start"] * 25)), int(round(k["end"] * 25))
        out[max(a, 0) : max(min(b, frames), 0)] = (SHARP.index(tonic) + PARENT.get(mode, 0)) % 12
        end = max(end, min(b, frames))
    if 0 < end < frames:
        out[end:] = out[end - 1]
    return out


def beat_position(beats: np.ndarray, times: np.ndarray) -> np.ndarray:
    step = float(np.median(np.diff(beats)))
    out = np.interp(times, beats, np.arange(len(beats), dtype=np.float64))
    out = np.where(times < beats[0], (times - beats[0]) / step, out)
    return np.where(times > beats[-1], len(beats) - 1 + (times - beats[-1]) / step, out)


def frame_hazard(beats: np.ndarray, frames: int, hist: np.ndarray) -> np.ndarray:
    """Expected changes per frame: the per-beat density ``hist`` integrated over the frame's stretch of beat."""

    cum = np.r_[0.0, np.cumsum(hist)] / BINS
    total = cum[-1]
    u = beat_position(beats, np.arange(frames + 1) / 25.0)
    whole = np.floor(u)
    frac = (u - whole) * BINS
    i = np.minimum(frac.astype(np.int64), BINS - 1)
    return np.diff(whole * total + cum[i] + (frac - i) * (cum[i + 1] - cum[i]))


def phase_costs(beats: np.ndarray, frames: int, phase: dict) -> tuple[np.ndarray, np.ndarray]:
    """``-log(h / mean h)`` (floored) of chord changes and of bass-only changes, per frame of the whole song."""

    out = []
    for key in ("chord", "bass"):
        h = frame_hazard(beats, frames, phase[key])
        out.append((-np.log(np.clip(h / max(h.mean(), 1e-9), float(phase["floor"]), None))).astype(np.float32))
    return out[0], out[1]


def frame_positions(beats: np.ndarray, downbeats: np.ndarray, frames: int) -> np.ndarray:
    """Per frame (its start, where a change into it happens) the index into the position tables (``POS - 1``
    outside the beats, before the first downbeat or without downbeats)."""

    out = np.full(frames, POS - 1, dtype=np.int64)
    if len(downbeats) == 0:
        return out
    u = beat_position(beats, np.arange(frames) / 25.0)
    whole = np.floor(u).astype(np.int64)
    within = np.minimum(((u - whole) * POS_BINS).astype(np.int64), POS_BINS - 1)
    first = np.unique(np.abs(downbeats[:, None] - beats[None]).argmin(1))  # downbeats as beat indices
    index = np.arange(len(beats))
    last = np.searchsorted(first, index, side="right") - 1  # the bar's downbeat of each beat
    in_bar = np.where(last >= 0, np.minimum(index - first[np.maximum(last, 0)], 3), -1)
    inside = (whole >= 0) & (whole < len(beats) - 1)
    bar = np.where(inside, in_bar[np.clip(whole, 0, len(beats) - 1)], -1)
    ok = bar >= 0
    out[ok] = bar[ok] * POS_BINS + within[ok]
    return out


# ------------------------------------------------------------------ the CRF


def inv_softplus(x: float) -> float:
    return float(np.log(np.expm1(x)))


class ChordCRF(nn.Module):
    def __init__(self, lm: np.ndarray | None, init: dict | None = None, learn_lm: bool = False, tables: bool = False):
        super().__init__()
        init = {"w": 1.0, "w_bass": 1.0, "penalty": 8.0, "bass_penalty": 2.0, "boundary_weight": 1.0, "beta": 0.0,
                "beta_bass": 0.0, "lam": 0.0} | (init or {})  # fmt: skip
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.tensor(inv_softplus(max(v, 1e-3)))) for k, v in init.items()})
        lm = np.zeros((NS, NS), dtype=np.float32) if lm is None else lm.astype(np.float32)
        np.fill_diagonal(lm, 0.0)
        self.register_buffer("lm", torch.from_numpy(lm))
        self.register_buffer("off_diag", 1.0 - torch.eye(NS))
        self.delta = nn.Parameter(torch.zeros(NS, NS)) if learn_lm or tables else None
        self.pos_chord = nn.Parameter(torch.zeros(POS)) if tables else None
        self.pos_bass = nn.Parameter(torch.zeros(POS)) if tables else None

    def reg(self) -> Tensor:
        """Sum of squares of the learnt tables (their starting values are 0)."""

        tables = [t for t in (self.delta, self.pos_chord, self.pos_bass) if t is not None]
        return sum(t.square().sum() for t in tables) if tables else self.raw["w"] * 0

    def tables(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k).detach().cpu().numpy() for k in ("delta", "pos_chord", "pos_bass")
                if getattr(self, k) is not None}  # fmt: skip

    def value(self, name: str) -> Tensor:
        return F.softplus(self.raw[name])

    def values(self) -> dict[str, float]:
        return {k: round(float(self.value(k)), 4) for k in self.raw}

    def kernel(self) -> Tensor:
        """``exp(lam * LM + delta)`` with zeros on the diagonal (a chord move changes the chord)."""

        score = self.value("lam") * self.lm + (self.delta if self.delta is not None else 0.0)
        return torch.exp(score) * self.off_diag

    def emissions(self, detail: Tensor, bass: Tensor) -> Tensor:
        """``[B, T, NS, 12]`` from the logits ``detail`` ``[B, T, 745]`` and ``bass`` ``[B, T, 13]``."""

        p = torch.softmax(detail.float(), -1)
        fam = torch.zeros(*p.shape[:-1], NS, device=p.device).index_add_(-1, torch.as_tensor(GROUP, device=p.device), p)
        chord = self.value("w") * torch.log(fam.clamp_min(1e-6))
        b = self.value("w_bass") * torch.log_softmax(bass[..., :12].float(), -1)
        e = chord[..., None] + b[..., None, :]
        n = torch.full_like(e[..., N_STATE, :], NEG)
        n[..., 0] = chord[..., N_STATE]
        return torch.cat((e[..., :N_STATE, :], n[..., None, :]), -2)

    @staticmethod
    def rotate(e: Tensor, tonic: Tensor) -> Tensor:
        """Absolute -> key-relative states (root and bass minus the tonic); N untouched.  ``tonic`` ``[B, T]``."""

        B, T = tonic.shape
        idx = (torch.arange(12, device=e.device)[None, None] + tonic[..., None]) % 12  # [B, T, 12]: rel -> abs
        body = e[:, :, :N_STATE].unflatten(2, (FAM, 12))  # [B, T, FAM, root, bass]
        body = torch.gather(body, 3, idx[:, :, None, :, None].expand(B, T, FAM, 12, 12))
        body = torch.gather(body, 4, idx[:, :, None, None, :].expand(B, T, FAM, 12, 12))
        return torch.cat((body.flatten(2, 3), e[:, :, N_STATE:]), 2)

    def logz(self, e: Tensor, change: Tensor, stay: Tensor, bass_cost: Tensor, tonic: Tensor) -> Tensor:
        """log of the sum over paths, ``e`` ``[B, T, NS, 12]`` key-relative emissions; costs ``[B, T]``."""

        B, T = tonic.shape
        kernel = self.kernel()
        penalty = self.value("penalty")
        moves = (tonic[:, 1:] - tonic[:, :-1]) % 12  # rotation of the forward variables at a key change
        rotate_at = torch.nonzero(moves.any(0)).flatten().tolist()
        alpha = e[:, 0]
        for t in range(1, T):
            if rotate_at and t - 1 in rotate_at:
                alpha = self._shift(alpha, moves[:, t - 1])
            per_chord = torch.logsumexp(alpha, -1)  # [B, NS]
            top = per_chord.max(-1, keepdim=True).values
            chord_in = torch.log((torch.exp(per_chord - top) @ kernel).clamp_min(1e-30)) + top  # [B, NS]
            others = per_chord[..., None] + torch.log((-torch.expm1((alpha - per_chord[..., None]).clamp_max(-1e-6))).clamp_min(1e-30))
            s = stay[:, t, None, None]
            alpha = e[:, t] + torch.logsumexp(torch.stack((
                alpha - s,
                others - s - bass_cost[:, t, None, None],
                (chord_in - penalty - change[:, t, None])[..., None].expand_as(alpha),
            )), 0)  # fmt: skip
        return torch.logsumexp(alpha.flatten(1), -1)

    @staticmethod
    def _shift(alpha: Tensor, d: Tensor) -> Tensor:
        """Re-rotate key-relative forward variables when the tonic moves by ``d`` ``[B]`` (rel' = rel - d)."""

        B = alpha.shape[0]
        idx = (torch.arange(12, device=alpha.device)[None] + d[:, None]) % 12  # rel' -> rel
        body = alpha[:, :N_STATE].unflatten(1, (FAM, 12))
        body = torch.gather(body, 2, idx[:, None, :, None].expand(B, FAM, 12, 12))
        body = torch.gather(body, 3, idx[:, None, None, :].expand(B, FAM, 12, 12))
        return torch.cat((body.flatten(1, 2), alpha[:, N_STATE:]), 1)

    def loss(self, detail: Tensor, bass: Tensor, boundary: Tensor, gold_detail: Tensor, gold_bass: Tensor, tonic: Tensor,
             phase_chord: Tensor, phase_bass: Tensor, pos: Tensor | None = None) -> Tensor:  # fmt: skip
        """Mean over the crops of ``(log Z - log Z(known)) / frames``.  ``gold_detail`` ``[B, T]`` detail classes
        (< 0 unknown), ``gold_bass`` 0..11 (12 = N, < 0 unknown)."""

        e = self.emissions(detail, bass)
        known = gold_detail >= 0
        state = torch.as_tensor(GROUP, device=e.device)[gold_detail.clamp_min(0)]
        allowed = torch.zeros(e.shape, dtype=torch.bool, device=e.device)
        allowed[~known] = True
        bass_known = known & (gold_bass >= 0) & (gold_bass < 12) & (state != N_STATE)
        b_idx, t_idx = torch.nonzero(bass_known, as_tuple=True)
        allowed[b_idx, t_idx, state[bass_known], gold_bass[bass_known]] = True
        free = known & ~bass_known & (state != N_STATE)
        b_idx, t_idx = torch.nonzero(free, as_tuple=True)
        allowed[b_idx, t_idx, state[free]] = True
        b_idx, t_idx = torch.nonzero(known & (state == N_STATE), as_tuple=True)
        allowed[b_idx, t_idx, N_STATE, 0] = True
        both = torch.cat((e, e.masked_fill(~allowed, NEG)))
        bw = self.value("boundary_weight")
        b = torch.sigmoid(boundary.float()).clamp(1e-4, 1 - 1e-4)
        change = -bw * torch.log(b) + self.value("beta") * phase_chord
        stay = -bw * torch.log1p(-b)
        bass_cost = self.value("bass_penalty") + self.value("beta_bass") * phase_bass
        if self.pos_chord is not None and pos is not None:
            change = change + self.pos_chord[pos]
            bass_cost = bass_cost + self.pos_bass[pos]
        tonic2 = torch.cat((tonic, tonic))
        z = self.logz(self.rotate(both, tonic2), torch.cat((change, change)), torch.cat((stay, stay)),
                      torch.cat((bass_cost, bass_cost)), tonic2)  # fmt: skip
        B, T = tonic.shape
        return ((z[:B] - z[B:]) / T).mean()


# ------------------------------------------------------------------ whole-song decoding (numpy)


def decode(detail: np.ndarray, bass: np.ndarray, boundary: np.ndarray, tonic: np.ndarray, phase_chord: np.ndarray,
           phase_bass: np.ndarray, params: dict, lm: np.ndarray | None, tables: dict | None = None,
           pos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The best (state, bass) path of a whole song from probabilities (``detail`` ``[T, 745]``, ``bass`` ``[T, 13]``,
    ``boundary`` ``[T]``) with the CRF's ``params`` (``ChordCRF.values``) and bigram ``lm`` (relative states);
    ``tables`` (``ChordCRF.tables``) with the frames' ``pos`` add the learnt tables."""

    from .chords import family_probs

    frames = len(detail)
    family = family_probs(detail)
    chord_lp = params["w"] * np.log(np.clip(family, 1e-6, None))
    bass_lp = params["w_bass"] * np.log(np.clip(bass[:, :12] / bass[:, :12].sum(-1, keepdims=True), 1e-6, None))
    emit = chord_lp[:, :, None] + bass_lp[:, None, :]
    emit[:, N_STATE, 1:] = -np.inf
    emit[:, N_STATE, 0] = chord_lp[:, N_STATE]
    b = np.clip(boundary.astype(np.float64), 1e-4, 1 - 1e-4)
    change = -params["boundary_weight"] * np.log(b) + params["beta"] * phase_chord
    stay = -params["boundary_weight"] * np.log1p(-b)
    bass_cost = params["bass_penalty"] + params["beta_bass"] * phase_bass
    tables = tables or {}
    if "pos_chord" in tables and pos is not None:
        change = change + tables["pos_chord"][pos]
        bass_cost = bass_cost + tables["pos_bass"][pos]
    lm = np.zeros((NS, NS)) if lm is None else lm.astype(np.float64)
    trans_rel = params["lam"] * lm + tables.get("delta", 0.0)
    np.fill_diagonal(trans_rel, -np.inf)
    states = np.arange(NS)
    trans = {}
    for k in np.unique(tonic):
        rel = np.where(states == N_STATE, N_STATE, 12 * (states // 12) + (states % 12 - k) % 12)
        trans[int(k)] = trans_rel[np.ix_(rel, rel)]
    choice = np.zeros((frames, NS, 12), dtype=np.uint8)
    bass_from = np.zeros((frames, NS), dtype=np.int8)
    chord_from = np.zeros((frames, NS), dtype=np.int16)
    score = emit[0].copy()
    for t in range(1, frames):
        best_bass = score.max(1)
        bass_from[t] = score.argmax(1)
        # (a bass move from the same bass never beats staying, so the max needs no exclusion)
        hold = score - stay[t]
        moved_bass = (best_bass - stay[t] - bass_cost[t])[:, None]
        cand = best_bass[:, None] + trans[int(tonic[t])]
        prev = cand.argmax(0)
        moved_chord = (cand[prev, states] - params["penalty"] - change[t])[:, None]
        chord_from[t] = prev
        new, pick = hold, np.zeros(score.shape, dtype=np.uint8)
        better = moved_bass > new
        new, pick = np.where(better, moved_bass, new), np.where(better, 1, pick)
        better = moved_chord > new
        new, pick = np.where(better, moved_chord, new), np.where(better, 2, pick)
        choice[t] = pick
        score = new + emit[t]
    chord_path, bass_path = np.empty(frames, dtype=np.int64), np.empty(frames, dtype=np.int64)
    c, bb = np.unravel_index(int(score.argmax()), score.shape)
    for t in range(frames - 1, -1, -1):
        chord_path[t], bass_path[t] = c, bb
        if t == 0:
            break
        move = choice[t, c, bb]
        if move == 1:
            bb = int(bass_from[t, c])
        elif move == 2:
            c = int(chord_from[t, c])
            bb = int(bass_from[t, c])
    bass_path[chord_path == N_STATE] = 12
    return chord_path, bass_path
