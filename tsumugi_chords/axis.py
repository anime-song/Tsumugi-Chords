"""The audio layers of the chord model's note backbone (``note_model``), after Tsumugi's AMT
(instrument_agnostic_amt), and the audio loader.

* ``HCQT``: stereo waveform at ``RATE`` -> a constant-Q magnitude per channel and harmonic (1..5 times the
  bin's frequency, read off one large CQT by interpolation, bins above Nyquist zero), normalised per
  clip, at 50 Hz (``HOP``), 36 bins per octave from A0.  As Tsumugi's ``AudioFeatureExtractor``.
* ``StemConv``: Tsumugi's stem (time / 8, frequency / 4) with a GELU after its first two convolutions.
* ``Block``: one axis-Transformer layer (RMSNorm, attention with RoPE, MLP); ``PitchQuery``: one query per piano key.

Precision: the HCQT runs in fp32; under bf16 autocast the attention runs in bf16, but RoPE's angles and
rotation, the norms and the softmax-side heads stay in fp32 (bf16 positions above 256 are rounded, so
autocast's bf16 einsum of positions x frequencies scrambles the far positions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .heads import KEYS

RATE, HOP = 25_600, 512  # 50 Hz spectrogram frames, two per output frame


@dataclass
class AxisConfig:
    rate: int = RATE
    hop: int = HOP
    channels: int = 2
    fmin: float = 27.5
    bins: int = 312
    bins_per_octave: int = 36
    filter_scale: float = 0.475
    harmonics: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    base_ch: int = 32  # tokens are 4 * base_ch wide
    layers: int = 4
    heads: int = 4
    globals: int = 4
    dropout: float = 0.1
    refine: tuple[int, ...] = (1, 2)
    condition: bool = True
    checkpoint: bool = False  # gradient checkpointing of the stem and the layers
    phase: bool = False  # the onset saliency |dphi| * |X| as extra stem channels, and an onset head
    phase_enc: bool = False  # the phase encoder (sin / cos of the phase-prediction residual)
    phase_ch: int = 16  # its stem width (tokens 4 x wider)
    phase_layers: int = 2
    fine: bool = False  # the 25 Hz path for the chord outputs
    fb_decoded: bool = False  # the decoded (joint Viterbi, one-hot) feedback
    note_crf: bool = False  # the note semi-CRF head on the key tokens
    note_crf_dim: int = 256
    fine_ch: int = 32
    fine_layers: int = 2
    fine_kernel: int = 9
    extra: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return 4 * self.base_ch


class RecursiveCQT(nn.Module):
    """Tsumugi's recursive CQT: one STFT per octave from the top down, halving the rate between octaves with
    a fixed low-pass; ``bins`` need not fill the lowest octave."""

    def __init__(self, rate: int, hop: int, fmin: float, bins: int, bins_per_octave: int, filter_scale: float,
                 resampling_quality: int = 64):  # fmt: skip
        super().__init__()
        octaves = math.ceil(bins / bins_per_octave)
        if hop % 2 ** (octaves - 1):
            raise ValueError(f"hop {hop} must be divisible by 2^{octaves - 1} (one halving per octave)")
        self.hop, self.octaves, self.rate = hop, octaves, rate
        self.q = filter_scale / (2.0 ** (1.0 / bins_per_octave) - 1.0)
        freqs = fmin * 2.0 ** (np.arange(bins) / bins_per_octave)
        self.fft_sizes = []
        current = float(rate)
        for i in range(octaves):  # top octave first, counted down from the top bin (the lowest may be partial)
            octave = freqs[max(0, bins - (i + 1) * bins_per_octave) : bins - i * bins_per_octave]
            if octave[-1] > current / 2:
                raise ValueError("CQT bins above the Nyquist frequency")
            size = 2 ** int(np.ceil(np.log2(self.q * current / octave[0])))
            self.fft_sizes.append(size)
            self.register_buffer(f"kernel_{i}", self._kernel(octave, current, size), persistent=False)
            self.register_buffer(f"freq_{i}", torch.from_numpy(octave).float(), persistent=False)
            self.register_buffer(f"window_{i}", torch.from_numpy(scipy.signal.get_window("hann", size)).float(), persistent=False)
            current /= 2
        fir = scipy.signal.firwin(resampling_quality + 1, rate / 4.0, fs=rate, window=("kaiser", 5.0))
        self.register_buffer("fir", torch.from_numpy(fir).float().view(1, 1, -1), persistent=False)

    def _kernel(self, freqs: np.ndarray, rate: float, size: int) -> Tensor:
        kernel = torch.zeros(len(freqs), size // 2 + 1, dtype=torch.complex64)
        for k, f in enumerate(freqs):
            length = min(self.q * rate / f, size)
            window = torch.from_numpy(scipy.signal.get_window("hann", int(np.ceil(length)))).float()
            t = torch.arange(len(window), dtype=torch.float32)
            atom = window * torch.exp(2j * np.pi * f * t / rate)
            atom = atom / atom.abs().sum().clamp_min(1e-8)
            padded = torch.zeros(size, dtype=torch.complex64)
            start = (size - len(atom)) // 2
            padded[start : start + len(atom)] = atom
            kernel[k] = torch.fft.fft(padded)[: size // 2 + 1]
        return kernel

    def forward(self, x: Tensor, phase: bool = False, residual: bool = False) -> Tensor | tuple[Tensor, ...]:
        """``[B, samples]`` -> magnitudes ``[B, bins, ceil(samples / hop)]``, low bins first; with ``phase`` also
        the onset saliency ``|dphi| * |X|`` (``dphi`` the wrapped phase difference from the previous frame); with
        ``residual`` instead the sine and cosine of the phase-prediction residual (0 on the first frame and where the
        bin is silent)."""

        frames = math.ceil(x.shape[-1] / self.hop)
        out, saliency, res_sin, res_cos = [], [], [], []
        for i, size in enumerate(self.fft_sizes):
            stft = torch.stft(x, size, self.hop // 2**i, window=getattr(self, f"window_{i}"), center=True,
                              pad_mode="reflect", return_complex=True)  # fmt: skip
            complex_ = (stft.transpose(-1, -2) @ getattr(self, f"kernel_{i}").T).transpose(-1, -2)
            octave = complex_.abs()
            out.append(F.pad(octave, (0, max(0, frames - octave.shape[-1])))[..., :frames])
            if residual:  # one hop advances a steady tone at f by 2 pi f hop / rate (each octave: hop / 2^i at rate / 2^i)
                step = torch.polar(torch.ones_like(getattr(self, f"freq_{i}")), -2 * math.pi * getattr(self, f"freq_{i}") * self.hop / self.rate)
                z = complex_[..., 1:] * complex_[..., :-1].conj() * step[:, None]
                z = z / z.abs().clamp_min(1e-12)
                for part, dest in ((z.imag, res_sin), (z.real, res_cos)):
                    part = F.pad(part, (1, 0))
                    dest.append(F.pad(part, (0, max(0, frames - part.shape[-1])))[..., :frames])
            if phase:
                dphi = torch.angle(complex_[..., 1:] * complex_[..., :-1].conj()).abs()
                sal = F.pad(dphi * octave[..., 1:], (1, 0))
                saliency.append(F.pad(sal, (0, max(0, frames - sal.shape[-1])))[..., :frames])
            if i < self.octaves - 1:  # the next octave down at half the rate
                x = F.conv1d(x[:, None], self.fir, stride=2, padding=self.fir.shape[-1] // 2)[:, 0]
        if residual:
            return torch.cat(out[::-1], 1), torch.cat(res_sin[::-1], 1), torch.cat(res_cos[::-1], 1)
        if phase:
            return torch.cat(out[::-1], 1), torch.cat(saliency[::-1], 1)
        return torch.cat(out[::-1], 1)


class HCQT(nn.Module):
    """``[B, channels, samples]`` -> ``[B, channels * harmonics, frames, bins]`` (fp32, normalised per clip)."""

    def __init__(self, config: AxisConfig):
        super().__init__()
        self.channels, self.bins, self.phase, self.phase_enc = config.channels, config.bins, config.phase, config.phase_enc
        self.harmonics = len(config.harmonics)
        shifts = [config.bins_per_octave * math.log2(h / min(config.harmonics)) for h in config.harmonics]
        self.large = math.ceil(config.bins + max(shifts))
        below_nyquist = math.floor(config.bins_per_octave * math.log2(config.rate / 2 / config.fmin / min(config.harmonics)) + 1)
        self.actual = min(self.large, below_nyquist)
        self.cqt = RecursiveCQT(config.rate, config.hop, config.fmin * min(config.harmonics), self.actual,
                                config.bins_per_octave, config.filter_scale)  # fmt: skip
        position = (torch.arange(config.bins, dtype=torch.float32)[None] + torch.tensor(shifts)[:, None]).clamp(0, self.large - 1)
        lower = position.floor().long()
        self.register_buffer("lower", lower, persistent=False)
        self.register_buffer("upper", (lower + 1).clamp(max=self.large - 1), persistent=False)
        self.register_buffer("alpha", (position - lower)[..., None], persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(self, wave: Tensor, valid: Tensor | None = None) -> Tensor:
        """``valid`` ``[B, frames]`` (bool): the frames of the song, the rest (padding) left out of the
        normalisation and set to 0 after it."""

        batch = wave.shape[0]
        if self.phase_enc:  # (not with ``phase``)
            large, rs, rc = self.cqt(wave.float().flatten(0, 1), residual=True)
            large = F.pad(large, (0, 0, 0, self.large - self.actual))
            spec = large[:, self.lower] + self.alpha * (large[:, self.upper] - large[:, self.lower])
            spec = self.normalise(spec.unflatten(0, (batch, self.channels)).flatten(1, 2), valid)  # [B, C * H, T, F]
            frames = spec.shape[2]
            keep = 1.0 if valid is None else valid[:, None, :, None].to(spec.dtype)
            rs = rs[:, : self.bins].unflatten(0, (batch, self.channels)).transpose(-1, -2)  # [B, C, T, F]
            rc = rc[:, : self.bins].unflatten(0, (batch, self.channels)).transpose(-1, -2)
            mag = spec[:, :: self.harmonics]  # the fundamental of each channel (harmonic 1 first)
            pin = torch.stack((rs[:, :, :frames] * keep, rc[:, :, :frames] * keep, mag), 2).flatten(1, 2)  # [B, 3C, T, F]
            return spec, pin.contiguous()
        if self.phase:
            large, saliency = self.cqt(wave.float().flatten(0, 1), phase=True)  # [B * C, bins, frames] each
        else:
            large = self.cqt(wave.float().flatten(0, 1))  # [B * C, bins, frames]
        large = F.pad(large, (0, 0, 0, self.large - self.actual))
        spec = large[:, self.lower] + self.alpha * (large[:, self.upper] - large[:, self.lower])  # [B * C, H, bins, frames]
        spec = spec.unflatten(0, (batch, self.channels)).flatten(1, 2)
        if self.phase:  # the fundamental's bins (harmonic 1 is the first, unshifted), normalised on their own
            sal = self.normalise(saliency[:, : self.bins].unflatten(0, (batch, self.channels)), valid)
            return torch.cat((self.normalise(spec, valid), sal), 1)
        return self.normalise(spec, valid)

    @staticmethod
    def normalise(spec: Tensor, valid: Tensor | None) -> Tensor:
        """``[B, C, bins, frames]`` -> ``[B, C, frames, bins]``, zero mean and unit variance per clip (``valid``
        frames only, the rest 0)."""

        if valid is None:
            mean = spec.mean(dim=(1, 2, 3), keepdim=True)
            std = spec.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
            return ((spec - mean) / std).transpose(-1, -2).contiguous()  # [B, C * H, frames, bins]
        keep = valid[:, None, None, :].to(spec.dtype)
        count = (keep.sum(dim=(1, 2, 3), keepdim=True) * spec.shape[1] * spec.shape[2]).clamp_min(2.0)
        mean = (spec * keep).sum(dim=(1, 2, 3), keepdim=True) / count
        std = ((((spec - mean) * keep) ** 2).sum(dim=(1, 2, 3), keepdim=True) / (count - 1)).sqrt().clamp_min(1e-8)
        return ((spec - mean) / std * keep).transpose(-1, -2).contiguous()


class StemConv(nn.Module):
    """Tsumugi's stem (time / 8, frequency / 4), with a GELU after each of the first two convolutions."""

    def __init__(self, in_ch: int, base_ch: int, bins: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_ch, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=5, padding=2)
        self.freq_embed = nn.Parameter(torch.randn(1, base_ch, 1, bins) * 0.02)
        self.block1 = self._block(base_ch, base_ch * 2, (2, 1))
        self.block2 = self._block(base_ch * 2, base_ch * 4, (2, 2))
        self.block3 = self._block(base_ch * 4, base_ch * 4, (2, 2))
        self.block4 = nn.Sequential(nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1), nn.GroupNorm(4, base_ch * 4))

    @staticmethod
    def _block(in_ch: int, out_ch: int, stride: tuple[int, int]) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), nn.GroupNorm(4, out_ch), nn.GELU())

    def forward_mid(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``forward`` without ``valid``, also giving the first block's output ``[B, 2 * base_ch, T / 2, F]``."""

        x = F.gelu(self.conv1(x) + self.freq_embed[..., : x.shape[-1]])
        mid = self.block1(F.gelu(self.conv2(x)))
        return self.block4(self.block3(self.block2(mid))), mid

    def forward(self, x: Tensor, valid: Tensor | None = None, run=None) -> Tensor:  # [B, C, T, F] -> [B, D, T / 8, F / 4]
        """``valid`` ``[B, T]`` (bool): the frames of the song; the padding is kept at 0 after every layer and left
        out of the group norms' statistics, so a padded song gives what it gives alone.  ``run(stage, x, valid)``
        runs each of the six stages (e.g. a checkpoint per stage: the 50 Hz stages are the largest tensors)."""

        if valid is None:
            x = F.gelu(self.conv1(x) + self.freq_embed[..., : x.shape[-1]])
            x = F.gelu(self.conv2(x))
            return self.block4(self.block3(self.block2(self.block1(x))))
        run = run or (lambda stage, *args: stage(*args))
        x = run(self._first, x, valid)
        x = run(self._second, x, valid)
        for block in (self.block1, self.block2, self.block3, self.block4):
            x = run(block_stage(block), x, valid)
            if block[0].stride[0] == 2:
                valid = valid[:, ::2]
        return x * valid[:, None, :, None].to(x.dtype)

    def _first(self, x: Tensor, valid: Tensor) -> Tensor:
        h = self.conv1(x * valid[:, None, :, None].to(x.dtype))
        return F.gelu(h + self.freq_embed[..., : x.shape[-1]].to(h.dtype))

    def _second(self, x: Tensor, valid: Tensor) -> Tensor:
        return F.gelu(self.conv2(x * valid[:, None, :, None].to(x.dtype)))


def block_stage(block: nn.Sequential):
    """A stem block (convolution, group norm over the song's frames, GELU) as a function of ``(x, valid)``."""

    def stage(x: Tensor, valid: Tensor) -> Tensor:
        h = block[0](x * valid[:, None, :, None].to(x.dtype))
        h = masked_group_norm(block[1], h, valid[:, ::2] if block[0].stride[0] == 2 else valid)
        return block[2](h) if len(block) > 2 else h

    return stage


def masked_group_norm(norm: nn.GroupNorm, x: Tensor, valid: Tensor) -> Tensor:
    """``norm`` over the ``valid`` frames ``[B, T]`` of ``x`` ``[B, C, T, F]`` only; the sums in fp32, the tensor
    itself left in its own precision (no fp32 copy of a 50 Hz feature map)."""

    batch, channels, frames, bins = x.shape
    groups = x.view(batch, norm.num_groups, channels // norm.num_groups, frames, bins)
    keep = valid[:, None, None, :, None].to(x.dtype)
    count = (valid.sum(-1).float() * groups.shape[2] * bins).clamp_min(1.0)[:, None]  # [B, 1]
    kept = groups * keep
    mean = kept.sum(dim=(2, 3, 4), dtype=torch.float32) / count  # [B, G]
    var = (kept * groups).sum(dim=(2, 3, 4), dtype=torch.float32) / count - mean**2
    rstd = torch.rsqrt(var.clamp_min(0.0) + norm.eps)
    per = channels // norm.num_groups
    scale = rstd.repeat_interleave(per, 1) * norm.weight  # [B, C]
    shift = norm.bias - mean.repeat_interleave(per, 1) * scale
    return x * scale[..., None, None].to(x.dtype) + shift[..., None, None].to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps) * self.gamma
        return y.to(x.dtype)


def rope(x: Tensor, inv_freq: Tensor) -> Tensor:
    """Rotary position embedding of ``x`` ``[N, H, T, D]`` along T (pairs interleaved, as Tsumugi's), in fp32."""

    with torch.autocast(x.device.type, enabled=False):
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None]  # an fp32 product: no bf16 rounding of the positions
        cos, sin = angles.cos(), angles.sin()
        y = x.float()
        even, odd = y[..., 0::2], y[..., 1::2]
        y = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
    return y.to(x.dtype)


class Attention(nn.Module):
    """Tsumugi's gated self-attention (RMSNorm, RoPE, a sigmoid gate per head)."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.gates = nn.Linear(dim, heads)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """``mask`` ``[N, T]`` (bool): the keys that may be attended to (a row without any attends to all)."""

        h = self.norm(x)
        q, k, v = self.qkv(h).unflatten(-1, (3, self.heads, self.head_dim)).permute(2, 0, 3, 1, 4)  # 3 x [N, H, T, d]
        q, k = rope(q, self.inv_freq), rope(k, self.inv_freq)
        if mask is not None:  # an all-False row would give NaN, which masked keys still carry into the sums
            mask = (mask | ~mask.any(-1, keepdim=True))[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask)
        y = y * torch.sigmoid(self.gates(h)).transpose(1, 2)[..., None].to(y.dtype)
        return x + self.dropout(self.out(y.transpose(1, 2).flatten(-2)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attention = Attention(dim, heads, dropout)
        self.ffn = nn.Sequential(RMSNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))  # fmt: skip

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.attention(x, mask)
        return x + self.ffn(x)


class PitchQuery(nn.Module):
    """Tsumugi's key queries: an MLP of (pitch / 128, the pitch class on a circle, 1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        pitch = torch.arange(21, 21 + KEYS, dtype=torch.float32)
        angle = 2 * math.pi * pitch / 12
        self.register_buffer("features", torch.stack((pitch / 128, angle.sin(), angle.cos(), torch.ones_like(pitch)), -1), persistent=False)

    def forward(self) -> Tensor:
        return self.mlp(self.features)  # [88, D]


def load_stereo(path: Path) -> Tensor:
    """A whole song as ``[2, samples]`` at ``RATE`` (a mono file twice)."""

    import soundfile as sf
    import torchaudio

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.T)
    wave = wave[:2] if len(wave) >= 2 else wave.expand(2, -1)
    return torchaudio.functional.resample(wave, rate, RATE) if rate != RATE else wave
