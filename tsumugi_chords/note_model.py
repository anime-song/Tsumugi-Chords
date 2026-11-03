"""The note model: audio -> notes with instruments, after Tsumugi's AMT (instrument_agnostic_amt V2), decoded on a
beat grid.  Chords are meant to be read from its MIDI afterwards (as Tsumugi does), not from this model.

* ``NoteFrontend``: the axis model's HCQT (stereo, 5 harmonics, 312 bins, 50 Hz) with the onset saliency of each
  channel (``|dphi| * |X|``, ``axis.AxisConfig.phase``) and three stereo cues of the fundamental's bins: the sine and
  cosine of the left-right phase difference and the level difference, weighted by the bin's compressed energy (an
  instrument panned elsewhere looks different without separating the sources).
* ``StemConv`` (time / 8, frequency / 4) and ``stem_extra`` residual 3 x 3 blocks at that low resolution (where
  parameters are cheap), then per 6.25 Hz step a row of tokens (78 bands + 88 key queries) through ``layers`` axis
  Transformer blocks (along the row, then along time), as ``axis.AxisModel`` without the chord tokens.
* ``GridPool``: the key tokens up-sampled x4 to 25 Hz (a transposed convolution, never materialised) and averaged
  over the cells of a beat grid (``grid_points``: 1/4 + 1/3 beat, or 1/8 + 1/6 beat below 100 BPM) -> features
  ``[B, cells, 88, D]``.
* ``NoteHead`` (V2): a residual adapter, then per (instrument, pitch) pair a semi-CRF track whose interval score is
  ``query[begin] . key[end]`` (+ ``diag`` on one cell) with ``query = pitch_query[cell, pitch] + instrument_query[i]``
  (and so on: ``IntervalScorer.project_factorized``); a pair gate per window (cells pooled) says which pairs sound.

Training (``note_loss``): per window of ``window`` cells, the pairs with notes, the gate's top-k other pairs and
random ones; the semi-CRF NLL with the cost-augmented partition function (``fn_cost`` per gold cell left uncovered,
``fp_cost`` per silent cell covered: instrument_agnostic_amt's ``semi_crf_false_negative_cost`` /
``false_positive_cost``), averaged over tracks, plus the gate's BCE (positive weight negatives / positives, 1-50).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .axis import HOP, RATE, AxisConfig, Block, HCQT, PitchQuery, RMSNorm, StemConv
from .note_crf import TaskFeatureAdapter, log_z, viterbi

KEYS = 88
LOWEST = 21
INSTRUMENTS = 36  # Tsumugi's taxonomy (instrument_agnostic_amt.taxonomy); the drums (7) are left out of the targets
DRUM = 7
FRAME_RATE = 25
FINE = [0, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4]
SLOW = sorted(set([i / 8 for i in range(8)] + [i / 6 for i in range(6)]))
SLOW_BPM = 100.0


@dataclass
class NoteConfig:
    rate: int = RATE
    hop: int = HOP
    channels: int = 2
    fmin: float = 27.5
    bins: int = 312
    bins_per_octave: int = 36
    filter_scale: float = 0.475
    harmonics: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    phase: bool = True  # the onset saliency of each channel
    stereo: bool = True  # the left-right cues
    base_ch: int = 64  # the stem's width (its output is 4 * base_ch)
    stem_extra: int = 4  # residual blocks at time / 8, frequency / 4
    dim: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.1
    crf_dim: int = 256
    gate_dim: int = 128
    instruments: int = INSTRUMENTS
    checkpoint: bool = False  # gradient checkpointing of the stem (training)
    checkpoint_layers: bool = True  # ... and of the Transformer layers
    loops: int = 1  # passes through the Transformer layers (F): h1 = F(x), h_k+1 = h_k + alpha (F(h_k) - h_k); the stem once
    loop_alpha: float = 0.0  # the learnt alpha's start (0: the looped model starts as the one-pass one)

    def axis(self) -> AxisConfig:
        return AxisConfig(rate=self.rate, hop=self.hop, channels=self.channels, fmin=self.fmin, bins=self.bins,
                          bins_per_octave=self.bins_per_octave, filter_scale=self.filter_scale, harmonics=self.harmonics,
                          phase=self.phase)  # fmt: skip

    @property
    def in_channels(self) -> int:
        return self.channels * (len(self.harmonics) + int(self.phase)) + 3 * int(self.stereo)


def _pad_frames(x: Tensor, frames: int) -> Tensor:
    return F.pad(x, (0, max(0, frames - x.shape[-1])))[..., :frames]


class NoteFrontend(nn.Module):
    """``[B, 2, samples]`` -> ``[B, in_channels, frames, bins]`` (fp32): the HCQT (normalised per clip), the onset
    saliency (normalised on its own) and the stereo cues (``[-1, 1]``)."""

    def __init__(self, config: NoteConfig):
        super().__init__()
        self.config = config
        self.hcqt = HCQT(config.axis())

    def _complex(self, x: Tensor) -> Tensor:
        """The recursive CQT's complex bins ``[N, actual bins, frames]`` (low first), as ``RecursiveCQT.forward``."""

        cqt = self.hcqt.cqt
        frames = math.ceil(x.shape[-1] / cqt.hop)
        out = []
        for i, size in enumerate(cqt.fft_sizes):
            stft = torch.stft(x, size, cqt.hop // 2**i, window=getattr(cqt, f"window_{i}"), center=True,
                              pad_mode="reflect", return_complex=True)  # fmt: skip
            c = torch.view_as_real((stft.transpose(-1, -2) @ getattr(cqt, f"kernel_{i}").T).transpose(-1, -2))
            out.append(torch.view_as_complex(_pad_frames(c.movedim(-1, 0), frames).movedim(0, -1).contiguous()))
            if i < cqt.octaves - 1:
                x = F.conv1d(x[:, None], cqt.fir, stride=2, padding=cqt.fir.shape[-1] // 2)[:, 0]
        return torch.cat(out[::-1], 1)

    @torch.autocast("cuda", enabled=False)
    def forward(self, wave: Tensor) -> Tensor:
        h, c = self.hcqt, self.config
        batch = wave.shape[0]
        z = self._complex(wave.float().flatten(0, 1))  # [B * C, actual, frames]
        large = F.pad(z.abs(), (0, 0, 0, h.large - h.actual))
        spec = large[:, h.lower] + h.alpha * (large[:, h.upper] - large[:, h.lower])  # [B * C, H, bins, frames]
        parts = [h.normalise(spec.unflatten(0, (batch, c.channels)).flatten(1, 2), None)]
        fund = z[:, : c.bins]  # harmonic 1 is unshifted: the fundamental's bins
        if c.phase:
            dphi = torch.angle(fund[..., 1:] * fund[..., :-1].conj()).abs()
            sal = F.pad(dphi * fund[..., 1:].abs(), (1, 0))
            parts.append(h.normalise(sal.unflatten(0, (batch, c.channels)), None))
        if c.stereo:
            lr = fund.unflatten(0, (batch, c.channels))
            left, right = lr[:, 0], lr[:, 1]  # [B, bins, frames]
            cross = left * right.conj()
            energy = cross.abs()
            weight = torch.log1p(energy / energy.mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)).clamp(max=4.0) / 4.0
            unit = cross / energy.clamp_min(1e-12)
            al, ar = left.abs(), right.abs()
            floor = 1e-2 * (al + ar).mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            ild = (torch.log((al + floor) / (ar + floor)).clamp(-3.0, 3.0) / 3.0) * (weight > 0)
            cues = torch.stack((unit.imag * weight, unit.real * weight, ild), 1)  # [B, 3, bins, frames]
            parts.append(cues.transpose(-1, -2).contiguous())
        return torch.cat(parts, 1)


class ResBlock2d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, ch), nn.GELU(), nn.Conv2d(ch, ch, 3, padding=1),
                                 nn.GroupNorm(8, ch), nn.GELU(), nn.Conv2d(ch, ch, 3, padding=1))  # fmt: skip

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class GridPool(nn.Module):
    """Key tokens ``[B, T, 88, D]`` (6.25 Hz) -> the means over grid cells ``[B, cells, 88, D]`` of their x4
    transposed convolution (25 Hz), from ``pool`` ``[B, cells, 4T]`` (each cell's weights over the 25 Hz frames):
    the 25 Hz map is never built (frame ``4t + j`` is ``W_j x_t + b``)."""

    def __init__(self, dim: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=4)

    def forward(self, x: Tensor, pool: Tensor) -> Tensor:
        steps = x.shape[1]
        pool = F.pad(pool, (0, 4 * steps - pool.shape[-1]))[..., : 4 * steps].unflatten(-1, (steps, 4)).to(x.dtype)
        out = None
        for j in range(4):
            mixed = torch.einsum("bct,btkd->bckd", pool[..., j], x)
            part = mixed @ self.up.weight[:, :, j].to(x.dtype)  # [in, out]
            out = part if out is None else out + part
        return out + self.up.bias.to(x.dtype) * pool.sum((-1, -2))[..., None, None]


class InstrumentPairGate(nn.Module):
    """instrument_agnostic_amt's pair gate: per window, logits ``[W, instruments, 88]`` of which pairs sound."""

    def __init__(self, dim: int, gate_dim: int, instruments: int, dropout: float):
        super().__init__()
        self.pitch_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, gate_dim), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(gate_dim, gate_dim))  # fmt: skip
        self.instrument_proj = nn.Linear(dim, gate_dim, bias=False)
        self.instrument_bias = nn.Parameter(torch.zeros(instruments))
        self.scale = gate_dim**-0.5

    def forward(self, pooled: Tensor, instrument_embeddings: Tensor) -> Tensor:  # pooled [W, 88, D]
        gate = torch.einsum("wpg,ig->wip", self.pitch_proj(pooled), self.instrument_proj(instrument_embeddings)) * self.scale
        return gate + self.instrument_bias[None, :, None]


class NoteHead(nn.Module):
    def __init__(self, config: NoteConfig):
        super().__init__()
        dim = config.dim
        self.head_dim = config.crf_dim
        self.adapter = TaskFeatureAdapter(dim, config.dropout)
        self.instrument_embedding = nn.Embedding(config.instruments, dim)
        self.gate = InstrumentPairGate(dim, config.gate_dim, config.instruments, config.dropout)
        self.proj = nn.Linear(dim, 2 * config.crf_dim + 1)
        self.query_scale = 1.0 / math.sqrt(config.crf_dim)

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q, k, d = x.split([self.head_dim, self.head_dim, 1], -1)
        return q * self.query_scale, k, d.squeeze(-1)

    def forward(self, feats: Tensor) -> dict[str, Tensor]:
        """``feats`` ``[B, cells, 88, D]`` -> the pitch projections ``[B, cells, 88, h]`` (diag ``[B, cells, 88]``),
        the instruments' ``[instruments, h]`` (no bias: their sum is ``proj(pitch + instrument)``) and the adapted
        features (for the gate)."""

        x = self.adapter(feats)
        pq, pk, pd = self._split(self.proj(x))
        iq, ik, idg = self._split(F.linear(self.instrument_embedding.weight.to(x.dtype), self.proj.weight.to(x.dtype)))
        return {"features": x, "pq": pq, "pk": pk, "pd": pd, "iq": iq, "ik": ik, "id": idg}

    def gate_logits(self, features: Tensor, windows: list[tuple[int, int, int]]) -> Tensor:
        """``[len(windows), instruments, 88]`` for windows ``(crop, first cell, end cell)``."""

        pooled = torch.stack([features[b, s:e].float().mean(0) for b, s, e in windows])
        return self.gate(pooled.to(features.dtype), self.instrument_embedding.weight.to(features.dtype)).float()


class NoteModel(nn.Module):
    def __init__(self, config: NoteConfig):
        super().__init__()
        self.config = config
        dim, stem_out = config.dim, 4 * config.base_ch
        self.frontend = NoteFrontend(config)
        self.stem = StemConv(config.in_channels, config.base_ch, config.bins)
        self.stem_extra = nn.Sequential(*[ResBlock2d(stem_out) for _ in range(config.stem_extra)])
        self.stem_proj = nn.Conv2d(stem_out, dim, 1) if stem_out != dim else nn.Identity()
        self.bands = math.ceil(math.ceil(config.bins / 2) / 2)
        self.pitch_query = PitchQuery(dim)
        self.types = nn.Parameter(torch.randn(2, dim) * 0.02)  # band, key
        self.row_layers = nn.ModuleList(Block(dim, config.heads, config.dropout) for _ in range(config.layers))
        self.time_layers = nn.ModuleList(Block(dim, config.heads, config.dropout) for _ in range(config.layers))
        self.norm = RMSNorm(dim)
        if config.loops > 1:
            self.loop_alpha = nn.Parameter(torch.tensor(float(config.loop_alpha)))
        self.grid_pool = GridPool(dim)
        self.head = NoteHead(config)

    def _stem(self, spec: Tensor) -> Tensor:
        return self.stem_proj(self.stem_extra(self.stem(spec)))

    def _layer(self, i: int, x: Tensor) -> Tensor:
        batch, steps, row, dim = x.shape
        x = self.row_layers[i](x.reshape(batch * steps, row, dim)).reshape(batch, steps, row, dim)
        x = x.transpose(1, 2).reshape(batch * row, steps, dim)
        return self.time_layers[i](x).reshape(batch, row, steps, dim).transpose(1, 2)

    def stack(self, x: Tensor, ckpt: bool = False) -> Tensor:
        """One pass through the Transformer layers (F)."""

        for i in range(self.config.layers):
            x = torch.utils.checkpoint.checkpoint(self._layer, i, x, use_reentrant=False) if ckpt else self._layer(i, x)
        return x

    def layers(self, x: Tensor, ckpt: bool = False) -> Tensor:
        """``config.loops`` passes of the layers: ``h1 = F(x)``, then ``h <- h + alpha (F(h) - h)``."""

        h = self.stack(x, ckpt)
        for _ in range(self.config.loops - 1):
            h = h + self.loop_alpha.to(h.dtype) * (self.stack(h, ckpt) - h)
        return h

    def body(self, spec: Tensor, pool: Tensor) -> dict[str, Tensor]:
        """Everything after the frontend (what ``--compile`` compiles)."""

        ckpt = self.config.checkpoint and self.training
        stem = torch.utils.checkpoint.checkpoint(self._stem, spec, use_reentrant=False) if ckpt else self._stem(spec)
        bands = stem.permute(0, 2, 3, 1)  # [B, T, bands, D]
        batch, steps = bands.shape[:2]
        keys = self.pitch_query()[None, None].expand(batch, steps, -1, -1).to(bands.dtype)
        x = torch.cat((bands + self.types[0], keys + self.types[1]), 2)
        x = self.layers(x, ckpt and self.config.checkpoint_layers)
        x = self.norm(x)[:, :, self.bands :]  # the key tokens [B, T, 88, D]
        return self.head(self.grid_pool(x, pool))

    def forward(self, wave: Tensor, pool: Tensor) -> dict[str, Tensor]:
        """``wave`` ``[B, 2, samples]`` at ``RATE``; ``pool`` ``[B, cells, frames]`` (``pool_matrix``)."""

        return self.body(self.frontend(wave), pool)


def save_notes(model: NoteModel, path: Path) -> None:
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path), metadata={"note_config": json.dumps(asdict(model.config))})


def load_notes(path: Path, device="cpu") -> NoteModel:
    with safe_open(str(path), "pt") as f:
        raw = json.loads(f.metadata()["note_config"])
    raw["harmonics"] = tuple(raw["harmonics"])
    model = NoteModel(NoteConfig(**raw)).to(device).eval()
    model.load_state_dict(load_file(str(path), device=str(device)))
    model.requires_grad_(False)
    return model


# --- the beat grid -------------------------------------------------------------------------------------------------


def grid_points(beats: np.ndarray, duration: float) -> np.ndarray:
    """Grid times over the whole song: each beat cut at ``FINE`` (or ``SLOW`` below ``SLOW_BPM``, the song's median
    tempo), beats extended at the median beat length before the first and after the last to cover 0 .. duration."""

    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 2:
        beats = np.arange(0.0, duration + 0.5, 0.5)
    gap = float(np.median(np.diff(beats)))
    sub = np.asarray(SLOW if 60.0 / gap < SLOW_BPM else FINE)
    before = beats[0] - gap * np.arange(math.ceil(beats[0] / gap), 0, -1)
    after = beats[-1] + gap * np.arange(1, math.ceil(max(0.0, duration - beats[-1]) / gap) + 2)
    full = np.concatenate((before, beats, after))
    points = (full[:-1, None] + np.diff(full)[:, None] * sub[None]).ravel()
    return np.append(points, full[-1])


def crop_cells(points: np.ndarray, start: float, seconds: float) -> tuple[int, np.ndarray, np.ndarray]:
    """The cells of a crop: the index of its first grid point, and each cell's 25 Hz frame bounds ``[s, e)``
    (relative to ``start``; at least one frame each)."""

    frames = int(round(seconds * FRAME_RATE))
    first = int(np.searchsorted(points, start - 1e-9))
    last = int(np.searchsorted(points, start + seconds + 1e-9)) - 1
    inner = points[first : last + 1]
    s = np.clip(np.floor((inner[:-1] - start) * FRAME_RATE + 0.5), 0, frames - 1).astype(np.int64)
    e = np.clip(np.floor((inner[1:] - start) * FRAME_RATE + 0.5), 0, frames).astype(np.int64)
    return first, s, np.maximum(e, s + 1)


def pool_matrix(starts: list[np.ndarray], ends: list[np.ndarray], frames: int) -> Tensor:
    """``[B, max cells, frames]``: each cell's mean over its frames (padded cells all 0)."""

    cells = max(len(s) for s in starts)
    out = torch.zeros(len(starts), cells, frames)
    for b, (s, e) in enumerate(zip(starts, ends)):
        index = torch.arange(frames)
        s_, e_ = torch.from_numpy(s)[:, None], torch.from_numpy(e)[:, None]
        inside = (index[None] >= s_) & (index[None] < e_)
        out[b, : len(s)] = inside.float() / (e_ - s_).float()
    return out


def snap_notes(events: dict, points: np.ndarray) -> np.ndarray:
    """Tsumugi's notes (``cls``, ``pitch``, ``start``, ``end``; drums and keys off the piano left out) on the grid:
    ``[N, 4]`` (instrument, key 0..87, first cell, last cell), the begin at the nearest grid point, the end at the
    cell before the grid point nearest the note's end (at least the first cell); per (instrument, key) a note that
    starts on another's cell is dropped, and an earlier note ends before the next one's first cell."""

    cls, pitch = np.asarray(events["cls"]).astype(np.int64), np.asarray(events["pitch"]).astype(np.int64)
    start, end = np.asarray(events["start"], np.float64), np.asarray(events["end"], np.float64)
    keep = (cls != DRUM) & (pitch >= LOWEST) & (pitch < LOWEST + KEYS) & (end > start) & (cls >= 0) & (cls < INSTRUMENTS)
    cls, pitch, start, end = cls[keep], pitch[keep] - LOWEST, start[keep], end[keep]

    def nearest(t: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(points, t), 1, len(points) - 1)
        return np.where(t - points[i - 1] < points[i] - t, i - 1, i)

    b = nearest(start)
    e = np.maximum(b, nearest(end) - 1)
    order = np.lexsort((-e, b, pitch, cls))
    cls, pitch, b, e = cls[order], pitch[order], b[order], e[order]
    out = []
    for j in range(len(b)):
        if out and out[-1][0] == cls[j] and out[-1][1] == pitch[j]:
            if b[j] <= out[-1][2]:
                continue
            if b[j] <= out[-1][3]:
                out[-1][3] = b[j] - 1
        out.append([cls[j], pitch[j], b[j], e[j]])
    return np.asarray(out, dtype=np.int64).reshape(-1, 4)


def crop_notes(notes: np.ndarray, first: int, cells: int) -> np.ndarray:
    """The song's grid notes inside a crop's cells ``first .. first + cells - 1``, clipped, relative to ``first``."""

    if not len(notes):
        return notes
    inside = (notes[:, 3] >= first) & (notes[:, 2] < first + cells)
    out = notes[inside].copy()
    out[:, 2] = np.maximum(out[:, 2], first) - first
    out[:, 3] = np.minimum(out[:, 3], first + cells - 1) - first
    return out


# --- the loss and the decoding -------------------------------------------------------------------------------------


def windows_of(cells: list[int], window: int, minimum: int = 8) -> list[tuple[int, int, int]]:
    """``(crop, first cell, end cell)`` windows of ``window`` cells (a last shorter one kept from ``minimum``)."""

    out = []
    for b, n in enumerate(cells):
        for s in range(0, n, window):
            e = min(n, s + window)
            if e - s >= minimum or s == 0:
                out.append((b, s, e))
    return out


def _track_tensors(head: dict[str, Tensor], b: Tensor, inst: Tensor, pitch: Tensor, s: int, e: int):  # (decode)
    q = head["pq"][b, s:e, pitch] + head["iq"][inst][:, None]  # [n, L, h]
    k = head["pk"][b, s:e, pitch] + head["ik"][inst][:, None]
    d = head["pd"][b, s:e, pitch] + head["id"][inst][:, None]  # [n, L]
    return q.float(), k.float(), d.float()


def interval_scores(q: Tensor, k: Tensor, d: Tensor) -> Tensor:
    """``score[end, begin, track] = q[track, begin] . k[track, end]`` (+ ``d`` on the diagonal), ``[L, L, n]``."""

    score = torch.einsum("nbh,neh->ebn", q, k)
    i = torch.arange(score.shape[0], device=score.device)
    score[i, i] = score[i, i] + d.transpose(0, 1)
    return score


def active_cost(active: Tensor, fn_cost: float, fp_cost: float) -> Tensor:
    """instrument_agnostic_amt's ``_build_interval_active_cost``: ``[L, L, n]`` cost of covering ``[begin, end]``,
    ``fp * length - (fn + fp) * gold cells covered`` (the weighted Hamming loss up to a gold constant), ``active``
    ``[n, L]`` the gold cells."""

    n, length = active.shape
    prefix = F.pad(active.float().cumsum(1), (1, 0))  # [n, L + 1]
    covered = prefix[:, 1:].transpose(0, 1)[:, None] - prefix[:, :-1].transpose(0, 1)[None]  # [end, begin, n]
    idx = torch.arange(length, device=active.device)
    size = (idx[:, None] - idx[None] + 1).clamp_min(0).float()[..., None]
    cost = fp_cost * size - (fn_cost + fp_cost) * covered
    return cost * torch.ones(length, length, device=active.device).tril()[..., None]


def _to(x: np.ndarray, device) -> Tensor:
    """A host array on ``device`` without waiting for the GPU's queue (pinned, non-blocking)."""

    t = torch.from_numpy(np.ascontiguousarray(x))
    return t.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else t.to(device)


def select_pairs(gate: Tensor, positive: np.ndarray, topk: int, randoms: int, max_pairs: int) -> list[np.ndarray]:
    """Per window the positive pair ids (instrument * 88 + key), then the gate's top-k other pairs and random ones, at
    most ``max_pairs`` (AMT's ``_select_pair_candidates``), from ``positive`` ``[windows, pairs]`` (bool); all windows
    at once on the GPU, one transfer back."""

    flat = gate.detach().flatten(1).float()
    count, total = flat.shape
    pos = _to(positive, flat.device)
    npos = positive.sum(1)
    hard_cap = np.minimum(np.minimum(topk, np.maximum(0, max_pairs - npos)), total - npos)
    rand_cap = np.minimum(randoms, np.maximum(0, max_pairs - npos - hard_cap))
    k_hard, k_rand = int(hard_cap.max(initial=0)), int(rand_cap.max(initial=0))
    hard = flat.masked_fill(pos, float("-inf")).topk(k_hard, 1).indices if k_hard else flat.new_zeros(count, 0, dtype=torch.long)
    taken = pos.clone()
    rank = torch.arange(k_hard, device=flat.device)[None] < _to(hard_cap, flat.device)[:, None]
    taken.scatter_(1, hard, rank) if k_hard else None
    taken |= pos
    noise = torch.rand(count, total, device=flat.device).masked_fill(taken, -1.0)
    rand = noise.topk(k_rand, 1) if k_rand else None
    hard_np = hard.cpu().numpy()
    rand_ids, rand_ok = (rand.indices.cpu().numpy(), (rand.values >= 0).cpu().numpy()) if rand is not None else (None, None)
    out = []
    for w in range(count):
        ids = [np.flatnonzero(positive[w]), hard_np[w, : hard_cap[w]]]
        if rand is not None:
            r = rand_ids[w, : rand_cap[w]]
            ids.append(r[rand_ok[w, : rand_cap[w]]])
        out.append(np.concatenate(ids).astype(np.int64))
    return out


def note_loss(head: dict[str, Tensor], gate: Tensor, windows: list[tuple[int, int, int]], notes: list[np.ndarray],
              fn_cost: float = 2.0, fp_cost: float = 0.3, topk: int = 256, randoms: int = 128, max_pairs: int = 512,
              chunk: int = 512, backend: str = "triton") -> tuple[Tensor, Tensor, dict[str, float]]:  # fmt: skip
    """The semi-CRF NLL (mean per track) and the gate's BCE.  ``notes[b]`` ``[N, 4]`` (instrument, key, first, last
    cell) of crop ``b``; ``gate`` ``[len(windows), instruments, 88]``.  The tracks and their gold notes are laid out
    on the host and sent once per chunk, so the GPU's queue is not drained on the way (one wait, for the selection)."""

    device = gate.device
    pairs = gate.shape[1] * gate.shape[2]
    positive = np.zeros((len(windows), pairs), dtype=bool)
    gold = []  # per window: pair id, first, last (relative to the window) of its notes
    for w, (b, s, e) in enumerate(windows):
        nb = notes[b]
        inside = nb[(nb[:, 3] >= s) & (nb[:, 2] < e)] if len(nb) else nb.reshape(0, 4)
        pid = inside[:, 0] * KEYS + inside[:, 1]
        positive[w, pid] = True
        gold.append((pid, np.maximum(inside[:, 2], s) - s, np.minimum(inside[:, 3], e - 1) - s))
    presence = _to(positive.astype(np.float32), device).view(gate.shape)
    npos = float(positive.sum())
    pos_weight = min(50.0, max(1.0, (positive.size - npos) / max(npos, 1.0)))
    gate_loss = F.binary_cross_entropy_with_logits(gate, presence, pos_weight=gate.new_tensor(pos_weight))
    chosen = select_pairs(gate, positive, topk, randoms, max_pairs)
    # every track (window, pair) and its gold notes, grouped by window length
    groups: dict[int, list[int]] = {}
    for w, (b, s, e) in enumerate(windows):
        groups.setdefault(e - s, []).append(w)
    total, tracks, intervals = gate.new_zeros(()), 0, 0
    for length, ws in groups.items():
        t_win = np.concatenate([np.full(len(chosen[w]), w) for w in ws])
        t_pid = np.concatenate([chosen[w] for w in ws])
        g_track, g_first, g_last, offset = [], [], [], 0
        for w in ws:
            pid, first, last = gold[w]
            order = np.argsort(chosen[w], kind="stable")
            g_track.append(offset + order[np.searchsorted(chosen[w], pid, sorter=order)])
            g_first.append(first), g_last.append(last)
            offset += len(chosen[w])
        g_track, g_first, g_last = np.concatenate(g_track), np.concatenate(g_first), np.concatenate(g_last)
        info = np.stack([np.asarray([windows[w][0] for w in t_win]), np.asarray([windows[w][1] for w in t_win]), t_pid // KEYS, t_pid % KEYS])
        for c in range(0, len(t_pid), chunk):
            n = min(chunk, len(t_pid) - c)
            sel = (g_track >= c) & (g_track < c + n)
            b_, s_, inst, key = _to(info[:, c : c + n].astype(np.int64), device)
            gt, gf, gl = _to(np.stack([g_track[sel] - c, g_first[sel], g_last[sel]]).astype(np.int64), device)
            cells = s_[:, None] + torch.arange(length, device=device)[None]  # [n, L]
            q = (head["pq"][b_[:, None], cells, key[:, None]] + head["iq"][inst][:, None]).float()  # [n, L, h]
            k = (head["pk"][b_[:, None], cells, key[:, None]] + head["ik"][inst][:, None]).float()
            d = (head["pd"][b_[:, None], cells, key[:, None]] + head["id"][inst][:, None]).float()  # [n, L]
            score = interval_scores(q, k, d)
            noise = score.new_zeros(length - 1, n)
            if fn_cost or fp_cost:
                edges = torch.zeros(n, length + 1, device=device)
                edges.index_put_((gt, gf), torch.ones_like(gf, dtype=edges.dtype), accumulate=True)
                edges.index_put_((gt, gl + 1), -torch.ones_like(gl, dtype=edges.dtype), accumulate=True)
                active = edges.cumsum(1)[:, :length].clamp(0, 1)
                logz = log_z((score + active_cost(active, fn_cost, fp_cost)).contiguous(), noise, backend)
            else:
                logz = log_z(score.contiguous(), noise, backend)
            total = total + logz.sum() - score[gl, gf, gt].sum()
            tracks += n
            intervals += int(sel.sum())
    nll = total / max(tracks, 1)
    return nll, gate_loss, {"note_nll": float(nll.detach()), "gate_loss": float(gate_loss.detach()),
                            "tracks_per_window": tracks / max(len(windows), 1), "notes_per_window": intervals / max(len(windows), 1)}  # fmt: skip


@torch.no_grad()
def decode(head: dict[str, Tensor], gate: Tensor, windows: list[tuple[int, int, int]], threshold: float = -3.0,
           max_pairs: int = 512, note_bias: float = 0.0, chunk: int = 512) -> list[list[tuple[int, int, int, int]]]:  # fmt: skip
    """Per crop, notes ``(instrument, key, first cell, last cell)``: per window the pairs whose gate logit is at
    least ``threshold`` (at most ``max_pairs``, highest first), each decoded by the semi-CRF's Viterbi."""

    crops = 1 + max(b for b, _, _ in windows)
    out: list[list[tuple[int, int, int, int]]] = [[] for _ in range(crops)]
    for w, (b, s, e) in enumerate(windows):
        if e - s < 2:
            continue
        flat = gate[w].flatten()
        ids = (flat >= threshold).nonzero().flatten()
        ids = ids[flat[ids].argsort(descending=True)][:max_pairs]
        for c in range(0, len(ids), chunk):
            part = ids[c : c + chunk]
            q, k, d = _track_tensors(head, torch.full_like(part, b), part // KEYS, part % KEYS, s, e)
            score = interval_scores(q, k, d) + note_bias
            paths = viterbi(score.contiguous(), score.new_zeros(e - s - 1, len(part)))
            for pid, path in zip(part.tolist(), paths):
                out[b].extend((pid // KEYS, pid % KEYS, s + i, s + j) for i, j in path)
    return out


def note_scores(est: list[tuple[int, int, int, int]], ref: np.ndarray) -> dict[str, float]:
    """Grid-note counts: matches by (key, first cell) [pitch], + instrument [inst], + last cell within max(1, 20 % of
    the length) [offset]; ``*_tp``, ``n_est``, ``n_ref`` for summing over crops."""

    ref_l = [tuple(r) for r in np.asarray(ref).reshape(-1, 4).tolist()]
    out = {"n_est_pitch": 0, "n_ref_pitch": 0, "pitch_tp": 0, "n_est": len(est), "n_ref": len(ref_l), "inst_tp": 0, "offset_tp": 0}
    ep, rp = {(k, f) for _, k, f, _ in est}, {(k, f) for _, k, f, _ in ref_l}
    out["n_est_pitch"], out["n_ref_pitch"], out["pitch_tp"] = len(ep), len(rp), len(ep & rp)
    ref_map = {(i, k, f): l for i, k, f, l in ref_l}
    for i, k, f, l in est:
        if (i, k, f) in ref_map:
            out["inst_tp"] += 1
            rl = ref_map[(i, k, f)]
            if abs(l - rl) <= max(1, round(0.2 * (rl - f + 1))):
                out["offset_tp"] += 1
    return out


def f_measures(counts: dict[str, float]) -> dict[str, float]:
    def f(tp, ne, nr):
        p, r = tp / max(ne, 1), tp / max(nr, 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)

    out = {}
    for name, tp, ne, nr in (("pitch", "pitch_tp", "n_est_pitch", "n_ref_pitch"), ("inst", "inst_tp", "n_est", "n_ref"),
                             ("offset", "offset_tp", "n_est", "n_ref")):  # fmt: skip
        out[f"{name}_p"], out[f"{name}_r"], out[f"{name}_f"] = f(counts[tp], counts[ne], counts[nr])
    return out
