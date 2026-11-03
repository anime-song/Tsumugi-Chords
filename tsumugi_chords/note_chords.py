"""A chord model on the note model (``note_model.NoteModel``): the note model's row of tokens per 6.25 Hz step (78
frequency bands + 88 key queries) gets three more, a key token, a beat token and a chord query, and the chord
outputs come off the chord query at 25 Hz.

* The key token reads the song's keys as given (no-decoder keys, ``outputs/nodec_keys``): per 25 Hz frame a
  one-hot of the 24 keys (tonic + 12 x minor; other modes by their parent major) and a present flag, four frames
  per token (a strided convolution).
* The beat token reads the beats as given (``beat_features``): beat / downbeat on the frame, beat and bar phase
  (sin, cos), the time since the last and until the next beat and downbeat (so each frame carries where the
  beats are on both sides), present.  A model that can see the beats does not use them unless it is told where
  they are (the decoder's experience), hence the explicit past and future.
* The chord query is a learnt token; its final state is up-sampled x4 (a transposed convolution), passed through
  a residual adapter and read by one linear layer: root (13), bass (13), chord tones (12), root x quality
  (``DETAIL``) and the chord boundary, as ``axis.Heads``.
* With ``lookahead`` another linear layer reads the chord's neighbourhood (``LOOKAHEAD``, training targets that
  make the chord query carry the progression on both sides): the root interval from the previous chord and to the
  next one (``(r_next - r) mod 12``, 12 each), the next chord's family (``chords.FAMILIES`` and N) and the beats
  until the next chord change and since the last one (``DISTANCE_BINS``: half beats, 8 beats or more last); with
  ``BASS_LOOKAHEAD`` also the bass intervals from the previous and to the next segment.  ``lookahead`` is True (these
  ``LOOKAHEAD`` heads) or a list of ``(name, classes)``, kept in the saved config.

With ``reader="gru"`` the note model is frozen (eval mode, no gradients) and gets no extra tokens: its final key
tokens ``[B, T, 88, D]`` (6.25 Hz) are cut to ``GRU_KEY_DIM`` per key, concatenated with the key and beat
features (the same strided convolutions as the tokens above) and read by a 3-layer bidirectional GRU, whose states go
through the same up-sampling, adapter and outputs.

``chords`` > 1 chord queries (as the axis model's global tokens) are merged by one linear layer into the chord
features.  With ``note_head`` the note model keeps its note head (grid pool, semi-CRF projections, pair gate) and,
given a grid ``pool``, also returns its outputs off the final key tokens (``note_head``), for learning the notes
alongside the chords; otherwise the note head is not used.  ``save_note_chords`` keeps the note model's
config with the weights; ``load_note_chords`` rebuilds it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .chords import FAMILIES
from .heads import DETAIL, FRAME_RATE
from .note_model import NoteConfig, NoteModel

BEAT_FEATURES = 11
KEY_FEATURES = 25
OUTPUTS = (("root", 13), ("bass", 13), ("tones", 12), ("detail", DETAIL), ("chord_boundary", 1))
DISTANCE_BINS = 17  # half beats 0 .. 7.5, then 8 beats or more
LOOKAHEAD = (("prev_interval", 12), ("next_interval", 12), ("next_family", len(FAMILIES) + 1), ("remaining", DISTANCE_BINS),
             ("elapsed", DISTANCE_BINS))  # fmt: skip
BASS_LOOKAHEAD = (("prev_bass_interval", 12), ("next_bass_interval", 12))
GRU_KEY_DIM, GRU_INPUT, GRU_HIDDEN, GRU_LAYERS = 16, 512, 256, 3
BASS = 13  # bass classes (12 pitch classes, none)
AR_DURATIONS = 33  # an AR chord's length: half beats 0 .. 15.5, then 16 beats or more
AR_MAX = 64  # chords per crop the AR decoder reads (30 s: median 21, at most 60)


class ChordAR(nn.Module):
    """The auxiliary AR decoder: one position per label chord (runs of root x quality + bass), reading the chord
    tokens (6.25 Hz) by cross-attention.  Position i gets chord i - 1 (root x quality, bass, length; BOS at the start)
    and predicts chord i in three steps: root x quality, then its bass given the true root x quality, then its
    length given both (the chord's own tokens in order, as NotaGen's character decoder, without its Transformer).
    No onset is given: where a chord starts follows from the lengths and the audio."""

    def __init__(self, dim: int, layers: int = 3, heads: int = 8, dropout: float = 0.1, memory_steps: int = 512):
        super().__init__()
        self.detail_emb = nn.Embedding(DETAIL + 1, dim)  # + BOS
        self.bass_emb = nn.Embedding(BASS + 1, dim)
        self.length_emb = nn.Embedding(AR_DURATIONS + 1, dim)
        self.pos = nn.Embedding(AR_MAX, dim)
        self.memory_pos = nn.Embedding(memory_steps, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(dim))
        self.detail_out = nn.Linear(dim, DETAIL)
        self.bass_out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, BASS))
        self.length_out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, AR_DURATIONS))

    def forward(self, memory: Tensor, detail: Tensor, bass: Tensor, length: Tensor, count: Tensor) -> dict[str, Tensor]:
        """``memory`` ``[B, T, D]``; the chords ``[B, L]`` (padding and unknown lengths negative), ``count`` ``[B]``
        (each at least 1) -> logits ``ar_detail`` / ``ar_bass`` / ``ar_length`` ``[B, L, n]``."""

        steps, L = memory.shape[1], detail.shape[1]
        shift = lambda x, bos: torch.cat((torch.full_like(x[:, :1], bos), x[:, :-1].clamp(min=0)), 1)  # noqa: E731
        x = self.detail_emb(shift(detail, DETAIL)) + self.bass_emb(shift(bass, BASS)) + self.length_emb(shift(length, AR_DURATIONS))
        x = x + self.pos.weight[:L]
        memory = memory + self.memory_pos.weight[:steps].to(memory.dtype)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        pad = torch.arange(L, device=x.device)[None] >= count[:, None]
        h = self.decoder(x.to(memory.dtype), memory, tgt_mask=causal, tgt_key_padding_mask=pad)
        hb = h + self.detail_emb(detail.clamp(min=0)).to(h.dtype)
        hl = hb + self.bass_emb(bass.clamp(min=0)).to(h.dtype)
        return {"ar_detail": self.detail_out(h).float(), "ar_bass": self.bass_out(hb).float(), "ar_length": self.length_out(hl).float()}
SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}
PARENT = {"major": 0, "minor": 3, "harmonic_minor": 3, "dorian": -2, "phrygian": -4, "lydian": -5, "mixolydian": -7,
          "locrian": 1}  # fmt: skip
GAP = 8.0  # seconds: the cap of the since / until features


def _since_until(times: np.ndarray, marks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seconds since the last mark and until the next one, ``log1p`` of at most ``GAP`` scaled to 0..1 (1 where none)."""

    scale = np.log1p(GAP)
    if len(marks) == 0:
        return np.ones_like(times), np.ones_like(times)
    after = np.searchsorted(marks, times, side="right")
    prev = np.where(after > 0, times - marks[np.clip(after - 1, 0, len(marks) - 1)], GAP)
    nxt = np.where(after < len(marks), marks[np.clip(after, 0, len(marks) - 1)] - times, GAP)
    return np.log1p(np.clip(prev, 0, GAP)) / scale, np.log1p(np.clip(nxt, 0, GAP)) / scale


def phase(times: np.ndarray, marks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each time's position between consecutive marks (0..1) and whether it lies between two."""

    if len(marks) < 2:
        return np.zeros_like(times), np.zeros(len(times), dtype=bool)
    index = np.searchsorted(marks, times, side="right") - 1
    inside = (index >= 0) & (index < len(marks) - 1)
    index = np.clip(index, 0, len(marks) - 2)
    return np.where(inside, (times - marks[index]) / (marks[index + 1] - marks[index]), 0.0), inside


def beat_features(beats: np.ndarray, downbeats: np.ndarray, start: float, frames: int) -> np.ndarray:
    """``[frames, BEAT_FEATURES]`` of a crop from ``start`` (s): beat, downbeat on the frame; beat phase and bar
    phase (sin, cos; 0 outside the marks); since / until the beat and the downbeat; present."""

    out = np.zeros((frames, BEAT_FEATURES), dtype=np.float32)
    beats, downbeats = np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64)
    if len(beats) < 2:
        return out
    times = start + (np.arange(frames) + 0.5) / FRAME_RATE
    for column, marks in ((0, beats), (1, downbeats)):
        index = np.floor((marks - start) * FRAME_RATE).astype(np.int64)
        out[index[(index >= 0) & (index < frames)], column] = 1.0
    for column, marks in ((2, beats), (4, downbeats)):
        value, inside = phase(times, marks)
        out[:, column] = np.where(inside, np.sin(2 * np.pi * value), 0.0)
        out[:, column + 1] = np.where(inside, np.cos(2 * np.pi * value), 0.0)
    out[:, 6], out[:, 7] = _since_until(times, beats)
    out[:, 8], out[:, 9] = _since_until(times, downbeats)
    out[:, 10] = 1.0
    return out


def key_index(name: str) -> int | None:
    """``"E:major"`` -> 4, ``"A:minor"`` -> 21 (tonic + 12 x minor; other modes by their parent major)."""

    tonic, _, mode = name.partition(":")
    tonic = FLAT.get(tonic, tonic)
    if tonic not in SHARP:
        return None
    if mode == "minor":
        return SHARP.index(tonic) + 12
    return (SHARP.index(tonic) + PARENT.get(mode, 0)) % 12


def key_features(keys: list[dict], start: float, frames: int) -> np.ndarray:
    """``[frames, KEY_FEATURES]`` of a crop: the one-hot key (``{"start", "end", "key"}`` segments) and present."""

    out = np.zeros((frames, KEY_FEATURES), dtype=np.float32)
    times = start + (np.arange(frames) + 0.5) / FRAME_RATE
    for k in keys:
        index = key_index(k["key"])
        inside = (times >= k["start"]) & (times < k["end"])
        if index is not None and inside.any():
            out[inside, index] = 1.0
            out[inside, 24] = 1.0
    return out


def rotate_keys(keys: Tensor, steps: int) -> Tensor:
    """A pitch shift of ``key_features`` (``[..., KEY_FEATURES]``): each mode's 12 tonics rolled by ``steps``."""

    major, minor, present = keys[..., :12], keys[..., 12:24], keys[..., 24:]
    return torch.cat((torch.roll(major, steps, -1), torch.roll(minor, steps, -1), present), -1)


class NoteChordModel(nn.Module):
    def __init__(self, config: NoteConfig, dropout: float = 0.1, lookahead: bool | list = False, reader: str = "query",
                 chords: int = 1, note_head: bool = False, aux: bool = False):  # fmt: skip
        super().__init__()
        self.notes = NoteModel(config)
        if not note_head:
            self.notes.grid_pool = self.notes.head = None  # (the note outputs are not used)
        dim = config.dim
        self.reader, self.chords, self.note_head = reader, chords, note_head
        self.key_in = nn.Conv1d(KEY_FEATURES, dim, kernel_size=4, stride=4)
        self.beat_in = nn.Conv1d(BEAT_FEATURES, dim, kernel_size=4, stride=4)
        if reader == "gru":
            self.notes.requires_grad_(False)
            self.key_proj = nn.Linear(dim, GRU_KEY_DIM)
            width = 88 * GRU_KEY_DIM + 2 * dim
            self.gru_in = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, GRU_INPUT), nn.GELU(), nn.Dropout(dropout))
            self.gru = nn.GRU(GRU_INPUT, GRU_HIDDEN, num_layers=GRU_LAYERS, batch_first=True, bidirectional=True, dropout=dropout)
            self.gru_out = nn.Linear(2 * GRU_HIDDEN, dim)
        else:
            self.chord_token = nn.Parameter(torch.randn(dim) * 0.02 if chords == 1 else torch.randn(chords, dim) * 0.02)
            self.types = nn.Parameter(torch.randn(3, dim) * 0.02)  # key, beat, chord
            self.merge = nn.Linear(chords * dim, dim) if chords > 1 else None
        self.up = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=4)
        self.adapter = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.out = nn.Linear(dim, sum(n for _, n in OUTPUTS))
        self.heads = tuple(tuple(h) for h in lookahead) if isinstance(lookahead, (list, tuple)) else LOOKAHEAD if lookahead else ()
        self.neighbours = nn.Linear(dim, sum(n for _, n in self.heads)) if self.heads else None
        self.dropout = dropout
        self.aux = aux  # the auxiliary CTC heads (25 Hz: root x quality + blank, bass + blank) and AR decoder (training only)
        if aux:
            self.ctc = nn.Linear(dim, DETAIL + 1 + BASS + 1)
            with torch.no_grad():  # blank first (p ~ 0.95): otherwise the initial CTC loss (~ frames / labels x log classes) swamps the rest
                self.ctc.bias.zero_()
                self.ctc.bias[DETAIL] = float(np.log(DETAIL)) + 3.0
                self.ctc.bias[DETAIL + 1 + BASS] = float(np.log(BASS)) + 3.0
            self.ar = ChordAR(dim, dropout=dropout)

    def new_parameters(self) -> list[nn.Parameter]:
        """Everything but the note model's (for their own learning rate)."""

        return [p for n, p in self.named_parameters() if not n.startswith("notes.")]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.reader == "gru":
            self.notes.eval()  # (frozen: no dropout)
        return self

    def forward(self, wave: Tensor, beats: Tensor, keys: Tensor, pool: Tensor | None = None, note_items: Tensor | None = None,
                aux_outputs: bool = False) -> dict[str, Tensor]:  # fmt: skip
        """``wave`` ``[B, 2, samples]`` at ``RATE``; ``beats`` ``[B, frames, BEAT_FEATURES]`` and ``keys``
        ``[B, frames, KEY_FEATURES]`` at 25 Hz.  Logits at 25 Hz; with ``note_head`` and a ``pool`` (``note_model.pool_matrix``)
        also the note head's outputs (``note_head``), of the crops ``note_items`` (indices; all when None); with ``aux``
        and ``aux_outputs`` the CTC logits (``ctc_chord``, ``ctc_bass``) and the chord tokens for the AR decoder
        (``memory``, 6.25 Hz)."""

        notes, config = self.notes, self.notes.config
        frames = wave.shape[-1] // (config.rate // FRAME_RATE)
        spec = notes.frontend(wave)
        ckpt = config.checkpoint and self.training
        ckpt = ckpt and self.reader != "gru"
        if self.reader == "gru":
            with torch.no_grad():
                stem = notes._stem(spec)
        else:
            stem = torch.utils.checkpoint.checkpoint(notes._stem, spec, use_reentrant=False) if ckpt else notes._stem(spec)
        bands = stem.permute(0, 2, 3, 1)  # [B, T, bands, D]
        batch, steps, _, dim = bands.shape
        queries = notes.pitch_query()[None, None].expand(batch, steps, -1, -1).to(bands.dtype)

        def tokens(conv: nn.Conv1d, x: Tensor) -> Tensor:
            x = F.pad(x.to(bands.dtype), (0, 0, 0, max(0, 4 * steps - x.shape[1])))[:, : 4 * steps]
            return conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, D]

        if self.reader == "gru":
            with torch.no_grad():
                x = torch.cat((bands + notes.types[0], queries + notes.types[1]), 2)
                for i in range(config.layers):
                    x = notes._layer(i, x)
                x = notes.norm(x)[:, :, notes.bands :]  # the key tokens [B, T, 88, D]
            x = torch.cat((self.key_proj(x).flatten(2), tokens(self.key_in, keys), tokens(self.beat_in, beats)), -1)
            with torch.autocast(wave.device.type, enabled=False):
                c = self.gru_out(self.gru(self.gru_in(x.float()))[0])
            return self._outputs(c, frames)
        key = tokens(self.key_in, keys) + self.types[0]
        beat = tokens(self.beat_in, beats) + self.types[1]
        chord = (self.chord_token + self.types[2]).reshape(-1, dim)[None, None].expand(batch, steps, -1, -1)
        x = torch.cat((bands + notes.types[0], queries + notes.types[1], key[:, :, None], beat[:, :, None], chord), 2)
        for i in range(config.layers):
            x = torch.utils.checkpoint.checkpoint(notes._layer, i, x, use_reentrant=False) if ckpt and config.checkpoint_layers else notes._layer(i, x)
        x = notes.norm(x)
        c = x[:, :, -self.chords :]  # the chord queries [B, T, chords, D]
        c = self.merge(c.flatten(2)) if self.merge is not None else c[:, :, 0]
        out = self._outputs(c, frames, aux_outputs)
        if self.aux and aux_outputs:
            out["memory"] = c
        if self.note_head and pool is not None:
            x = x[:, :, notes.bands : notes.bands + 88]
            out["note_head"] = notes.head(notes.grid_pool(x if note_items is None else x[note_items], pool))
        return out

    def _outputs(self, c: Tensor, frames: int, aux_outputs: bool = False) -> dict[str, Tensor]:
        g = self.up(c.transpose(1, 2)).transpose(1, 2)[:, :frames]
        g = g + self.adapter(g)
        parts = self.out(g).float().split([n for _, n in OUTPUTS], -1)
        out = {name: part for (name, _), part in zip(OUTPUTS, parts)}
        out["chord_boundary"] = out["chord_boundary"][..., 0]
        if self.neighbours is not None:
            out |= {name: part for (name, _), part in zip(self.heads, self.neighbours(g).float().split([n for _, n in self.heads], -1))}
        if self.aux and aux_outputs:
            out["ctc_chord"], out["ctc_bass"] = self.ctc(g).float().split([DETAIL + 1, BASS + 1], -1)
        return out


def from_note_model(path: Path, device, checkpoint: bool = False, dropout: float = 0.1, lookahead: bool | list = False,
                    reader: str = "query", chords: int = 1, note_head: bool = False, aux: bool = False) -> NoteChordModel:  # fmt: skip
    """A chord model whose note model starts from a trained one (``note_model.save_notes``); the rest is new."""

    with safe_open(str(path), "pt") as f:
        raw = json.loads(f.metadata()["note_config"])
    raw["harmonics"], raw["checkpoint"] = tuple(raw["harmonics"]), checkpoint
    model = NoteChordModel(NoteConfig(**raw), dropout, lookahead, reader, chords, note_head, aux).to(device)
    state = {k: v for k, v in load_file(str(path), device=str(device)).items() if note_head or not k.startswith(("grid_pool.", "head."))}
    missing, unexpected = model.notes.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model


def save_note_chords(model: NoteChordModel, path: Path) -> None:
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    config = asdict(model.notes.config) | {"checkpoint": False}
    save_file(state, str(path), metadata={"note_chords_config": json.dumps({"notes": config, "dropout": model.dropout, "lookahead": [list(h) for h in model.heads], "reader": model.reader, "chords": model.chords, "note_head": model.note_head, "aux": model.aux})})


def init_from_chords(model: NoteChordModel, path: Path) -> list[str]:
    """Start ``model`` from a trained note-chords model's weights where names and shapes match (e.g. stage 2 from stage
    1 with new lookahead heads); returns the names left at their initial values."""

    state, own = load_file(str(path), device=str(next(model.parameters()).device)), model.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return sorted(set(own) - set(keep))


def load_note_chords(path: Path, device="cpu") -> NoteChordModel:
    """A trained model (frozen, eval mode)."""

    with safe_open(str(path), "pt") as f:
        raw = json.loads(f.metadata()["note_chords_config"])
    raw["notes"]["harmonics"] = tuple(raw["notes"]["harmonics"])
    model = NoteChordModel(NoteConfig(**raw["notes"]), raw["dropout"], raw.get("lookahead", False), raw.get("reader", "query"),
                           raw.get("chords", 1), raw.get("note_head", False), raw.get("aux", False)).to(device).eval()  # fmt: skip
    model.load_state_dict(load_file(str(path), device=str(device)))
    model.requires_grad_(False)
    return model


@torch.no_grad()
def song_logits(model: NoteChordModel, wave: Tensor, info: dict | None, window: float = 30.0, context: float = 5.0,
                batch: int = 2) -> dict[str, Tensor]:  # fmt: skip
    """The 25 Hz logits over a whole song (``[2, samples]`` at ``RATE``; ``info`` the no-decoder ``beats``,
    ``downbeats`` and ``keys``, or None): windows of ``window`` seconds (the training crops), each kept but for
    ``context`` seconds at either side (the song's ends excepted)."""

    device = next(model.parameters()).device
    per_frame = model.notes.config.rate // FRAME_RATE
    frames = wave.shape[-1] // per_frame
    size, margin = int(window * FRAME_RATE), int(context * FRAME_RATE)
    core = size - 2 * margin
    wave = F.pad(wave, (0, max(0, size * per_frame - wave.shape[-1])))
    pieces = []
    for kept in range(0, frames, core):
        start = min(max(0, kept - margin), max(0, frames - size))
        pieces.append((start, kept, min(kept + core, frames)))
    out: dict[str, list[Tensor]] = {}
    for i in range(0, len(pieces), batch):
        group = pieces[i : i + batch]
        waves = torch.stack([wave[:, s * per_frame : (s + size) * per_frame] for s, _, _ in group]).to(device)
        if info is None:
            beats = np.zeros((len(group), size, BEAT_FEATURES), np.float32)
            keys = np.zeros((len(group), size, KEY_FEATURES), np.float32)
        else:
            beats = np.stack([beat_features(info["beats"], info.get("downbeats", []), s / FRAME_RATE, size) for s, _, _ in group])
            keys = np.stack([key_features(info.get("keys", []), s / FRAME_RATE, size) for s, _, _ in group])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):  # (fp32 on the CPU: bf16 there is much slower)
            logits = model(waves, torch.from_numpy(beats).to(device), torch.from_numpy(keys).to(device))
        for j, (s, a, b) in enumerate(group):
            for k, v in logits.items():
                out.setdefault(k, []).append(v[j, a - s : b - s].float())
    return {k: torch.cat(v) for k, v in out.items()}
