"""The note semi-CRF (``axis.AxisConfig.note_crf``): instrument_agnostic_amt's V1 pitch-wise interval head on the
axis model's key tokens, as a training loss that teaches the notes as intervals (where each note starts and ends,
repeated notes apart) rather than the frame roll alone.

The key tokens (6.25 Hz) are up-sampled x4 to the 25 Hz frames by a transposed convolution, through a residual MLP
(``TaskFeatureAdapter``) and projected to a query, a key and a diagonal per key and frame (``IntervalScorer``).  The
score of a note on frames ``[begin, end]`` is ``query[begin] . key[end]`` (+ ``diag`` when ``begin == end``); a key's
notes in a window are one path of Yujia Yan's neural semi-CRF (no noise score, no length scaling: the AMT defaults).
The DP, its gradient and the Viterbi decoding are copied from instrument_agnostic_amt
(amt/modeling/heads/semi_crf.py, common.py), the Triton backend in ``note_crf_triton``.

The loss cuts each crop into windows of ``window`` frames (8 s, the AMT crops), keeps the windows whose notes are all
known, and averages the NLL over every key of those windows (all 88, silent ones too)."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Interval = Tuple[int, int]


class TaskFeatureAdapter(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class NoteCRFHead(nn.Module):
    """Key tokens ``[B, T, 88, D]`` -> query, key ``[B, 4T, 88, head_dim]`` and diag ``[B, 4T, 88]`` (cut to ``frames``)."""

    def __init__(self, dim: int, head_dim: int, dropout: float):
        super().__init__()
        self.head_dim = head_dim
        self.up = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=4)
        self.adapter = TaskFeatureAdapter(dim, dropout)
        self.proj = nn.Linear(dim, 2 * head_dim + 1)
        self.query_scale = 1.0 / math.sqrt(head_dim)

    def forward(self, pitches: Tensor, frames: int) -> tuple[Tensor, Tensor, Tensor]:
        batch, steps, keys, dim = pitches.shape
        x = self.up(pitches.permute(0, 2, 3, 1).reshape(batch * keys, dim, steps))  # [B * 88, D, 4T]
        x = x.reshape(batch, keys, dim, 4 * steps).permute(0, 3, 1, 2)[:, :frames]  # [B, frames, 88, D]
        query, key, diag = self.proj(self.adapter(x)).split([self.head_dim, self.head_dim, 1], -1)
        return query * self.query_scale, key, diag.squeeze(-1)


@torch.jit.script
def forward_backward(score: Tensor, noiseScore: Tensor):
    """log Z of ``score`` ``[T, T, B]`` (``score[end, begin]``) and ``noiseScore`` ``[T - 1, B]``, with the exact
    gradients (instrument_agnostic_amt, unchanged)."""
    T = int(score.shape[0])
    nBatch = int(score.shape[2])

    scoreFlip = torch.flip(score, dims=[0, 1]).transpose(0, 1)
    noiseScoreFlip = torch.flip(noiseScore, dims=(0,))

    scoreFB = torch.cat([score, scoreFlip], dim=-1)  # [T, T, 2B]
    noiseScoreFB = torch.cat([noiseScore, noiseScoreFlip], dim=-1)

    singleScoreSP = F.softplus(torch.diagonal(scoreFB, dim1=0, dim2=1)).transpose(-1, -2)

    v = score.new_zeros(T, nBatch * 2)
    v[0] = singleScoreSP[0, :]

    for end in range(1, T):
        subScore = scoreFB[end, :end, :]
        v[end] = torch.logaddexp(
            v[end - 1, :] + noiseScoreFB[end - 1, :],  # skip
            torch.logsumexp(v[:end, :] + subScore, dim=0),
        )
        v[end] += singleScoreSP[end, :]

    v, q = torch.chunk(v, 2, dim=-1)
    q = torch.flip(q, dims=(0,))
    logZ = v[-1]

    diag_softplus = F.softplus(torch.diagonal(score, dim1=0, dim2=1))
    grad = v.unsqueeze(0) + (q.unsqueeze(1) - logZ) + score
    grad = grad - 2 * torch.diag_embed(diag_softplus, dim1=0, dim2=1)

    lowerMask = torch.ones(T, T, device=grad.device).tril().unsqueeze(-1)
    grad = (grad * lowerMask).exp() * lowerMask

    gradNoise = (v[:-1] + q[1:] + noiseScore - logZ).exp()

    return logZ, grad, gradNoise


class _LogZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score: Tensor, noiseScore: Tensor) -> Tensor:
        logz, grad, gradNoise = forward_backward(score, noiseScore)
        ctx.save_for_backward(grad, gradNoise)
        return logz

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad, gradNoise = ctx.saved_tensors
        return grad * grad_output, gradNoise * grad_output


def log_z(score: Tensor, noise: Tensor, backend: str = "torch") -> Tensor:
    if backend == "triton":
        from .note_crf_triton import compute_log_z_triton

        return compute_log_z_triton(score, noise)
    return _LogZ.apply(score, noise)


@torch.jit.script
def viterbi(score: Tensor, noiseScore: Tensor) -> List[List[Tuple[int, int]]]:
    """The best intervals of each track (instrument_agnostic_amt's forward Viterbi, without forced starts)."""
    T = int(score.shape[0])
    nBatch = int(score.shape[2])
    v = score.new_zeros(T, nBatch)
    ptr = []
    v[0] = score[0, 0, :] * (score[0, 0, :] > 0)
    for end in range(1, T):
        subScore = score[end, :end, :]
        candidates = torch.cat([v[end - 1 : end, :] + noiseScore[end - 1, :], v[:end, :] + subScore], dim=0)
        bestValue, selection = candidates.max(dim=0)
        ptr.append(selection - 1)
        singletonMask = score[end, end, :] > 0
        v[end] = bestValue + score[end, end, :] * singletonMask
    ptr = torch.stack(ptr, dim=0).cpu()
    diagInclusion = (torch.diagonal(score, dim1=0, dim2=1) > 0).cpu()
    result: List[List[Tuple[int, int]]] = []
    for batchIdx in range(nBatch):
        pos = T - 1
        batchResult: List[Tuple[int, int]] = []
        curDiag = diagInclusion[batchIdx]
        while pos > 0:
            selection = int(ptr[pos - 1][batchIdx])
            if bool(curDiag[pos]):
                batchResult.append((pos, pos))
            if selection < 0:
                pos -= 1
            else:
                begin = selection
                batchResult.append((begin, pos))
                pos = begin
        if score[0, 0, batchIdx] > 0:
            batchResult.append((0, 0))
        batchResult.reverse()
        result.append(batchResult)
    return result


def interval_score(query: Tensor, key: Tensor, diag: Tensor) -> Tensor:
    """``[T, T, N]`` float32 scores (``score[end, begin] = query[begin] . key[end]``) of tracks ``query`` / ``key`` ``[T, N, D]``, ``diag`` ``[T, N]``."""
    score = torch.einsum("tnd,snd->stn", query, key).float()
    i = torch.arange(score.shape[0], device=score.device)
    score[i, i, :] = score[i, i, :] + diag.float()
    return score


def note_bounds(on: Tensor, onset: Tensor) -> tuple[Tensor, Tensor]:
    """Begin and end marks ``[N, T]`` of the notes of tracks with sounding frames ``on`` and note starts ``onset``
    (bool ``[N, T]``): a note begins where a key starts sounding or is struck again, and ends before the next
    begin or where it stops sounding."""
    prev = F.pad(on[:, :-1], (1, 0))
    begin = on & (onset | ~prev)
    nxt_on = F.pad(on[:, 1:], (0, 1))
    nxt_begin = F.pad(begin[:, 1:], (0, 1))
    end = on & (~nxt_on | nxt_begin)
    return begin, end


def loss(query: Tensor, key: Tensor, diag: Tensor, notes: Tensor, onsets: Tensor, mask: Tensor, window: int = 200,
         backend: str = "torch", chunk: int = 512) -> tuple[Tensor, dict[str, float]]:  # fmt: skip
    """Mean NLL per (window, key) track.  ``query`` / ``key`` ``[B, F, 88, D]``, ``diag`` ``[B, F, 88]``; ``notes`` (sounding)
    and ``onsets`` (note starts) ``[B, F', 88]``, ``mask`` ``[B, F']`` (1 = the notes are known)."""
    frames = min(query.shape[1], notes.shape[1])
    n = frames // window
    if n == 0:
        return query.sum() * 0, {}
    cut = lambda x: x[:, : n * window].unflatten(1, (n, window))  # noqa: E731  [B, n, W, ...]
    keep = cut(mask > 0).all(-1)  # [B, n]
    if not keep.any():
        return query.sum() * 0, {}
    q, k, d = cut(query)[keep], cut(key)[keep], cut(diag)[keep]  # [M, W, 88, D], [M, W, 88]
    on, start = cut(notes > 0.5)[keep], cut(onsets > 0.5)[keep]  # [M, W, 88]
    tracks = q.shape[0] * q.shape[2]
    q, k = q.permute(1, 0, 2, 3).reshape(window, tracks, -1), k.permute(1, 0, 2, 3).reshape(window, tracks, -1)
    d = d.permute(1, 0, 2).reshape(window, tracks)
    begin, end = note_bounds(on.permute(0, 2, 1).reshape(tracks, window), start.permute(0, 2, 1).reshape(tracks, window))
    bt, bf = begin.nonzero(as_tuple=True)  # sorted by track, then frame: the i-th begin pairs with the i-th end
    _, ef = end.nonzero(as_tuple=True)
    gold = (q[bf, bt].float() * k[ef, bt].float()).sum(-1) + torch.where(bf == ef, d[ef, bt].float(), 0.0)
    total = gold.new_zeros(())
    noise = q.new_zeros(window - 1, min(chunk, tracks), dtype=torch.float32)
    for s in range(0, tracks, chunk):
        e = min(s + chunk, tracks)
        score = interval_score(q[:, s:e], k[:, s:e], d[:, s:e])
        total = total + log_z(score, noise[:, : e - s], backend).sum()
    value = (total - gold.sum()) / tracks
    return value, {"note_crf_nll": float(value.detach()), "note_crf_notes": float(len(bt)) / tracks}


@torch.no_grad()
def decode(query: Tensor, key: Tensor, diag: Tensor, window: int = 200, chunk: int = 256) -> list[list[Interval]]:
    """The notes of one song, per key: ``query`` / ``key`` ``[F, 88, D]``, ``diag`` ``[F, 88]`` -> 88 lists of
    ``(begin, end)`` frames, decoded window by window (a note across a window edge is split there)."""
    frames, keys = query.shape[:2]
    out: list[list[Interval]] = [[] for _ in range(keys)]
    for w0 in range(0, frames, window):
        w1 = min(w0 + window, frames)
        T = w1 - w0
        if T < 2:
            break
        score = interval_score(query[w0:w1], key[w0:w1], diag[w0:w1])
        for s in range(0, keys, chunk):
            paths = viterbi(score[:, :, s : s + chunk].contiguous(), score.new_zeros(max(T - 1, 0), min(chunk, keys - s)))
            for j, path in enumerate(paths):
                out[s + j].extend((w0 + a, w0 + b) for a, b in path)
    return out
