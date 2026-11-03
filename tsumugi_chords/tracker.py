"""HeadChords: the chords of a whole song.

1. Beats, downbeats and keys from SheetSage2-Refined's decoder-free trackers (``sheetsage2_refined.keys.HeadKeys``),
   the inputs the chord model was trained with.
2. The chord model (a note-transcription backbone with a chord reader, ``note_chords``) over 30 s windows, the middle
   20 s of each kept: per 25 Hz frame root x quality (745), bass (13) and chord-boundary probabilities.
3. The semi-Markov decoder on the beat grid (``decode.ChordDecoder``) -> chord segments in mir_eval syntax."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .axis import load_stereo
from .decode import Chord, ChordDecoder
from .note_chords import NoteChordModel, load_note_chords, song_logits

DEFAULT_MODEL = "anime-song/Tsumugi-Chords"


def model_file(model: Path | str, name: str, revision: str | None = None) -> Path:
    """A file of a local model folder, or of a Hugging Face model repo (downloaded on first use)."""

    folder = Path(model)
    if folder.is_dir():
        return folder / name
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(str(model), name, revision=revision))


class HeadChords:
    def __init__(self, model: NoteChordModel, decoder: ChordDecoder, device: torch.device, beats_keys: dict):
        self.model, self.decoder, self.device = model, decoder, device
        self.beats_keys = beats_keys  # how to load HeadKeys: {"model", "revision"}
        self._keys = None

    @classmethod
    def load(cls, model: Path | str = DEFAULT_MODEL, device: str | None = None, revision: str | None = None,
             keys_model: Path | str | None = None) -> "HeadChords":  # fmt: skip
        """The chord model and decoder of a model folder or Hugging Face repo id (at ``revision``) on ``device``
        (default: CUDA if there is one).  SheetSage2-Refined (for the beats and keys; the model's ``beats_keys``, or
        ``keys_model``: a local SheetSage2-Refined folder or repo id) loads on first use."""

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config = json.loads(model_file(model, "config.json", revision).read_text(encoding="utf-8"))
        tracker = config["chord_tracker"]
        net = load_note_chords(model_file(model, tracker["model"], revision), device)
        decoder = ChordDecoder(model_file(model, tracker["tables"], revision), model_file(model, tracker["bigram"], revision), tracker["decoding"])
        beats_keys = dict(config["beats_keys"])
        if keys_model is not None:
            beats_keys = {"model": str(keys_model), "revision": None}
        return cls(net, decoder, device, beats_keys)

    def beats_and_keys(self, audio: Path | str) -> dict:
        """SheetSage2-Refined's decoder-free beats, downbeats and keys of ``audio``."""

        if self._keys is None:
            from sheetsage2_refined.keys import HeadKeys

            self._keys = HeadKeys.load(self.beats_keys["model"], str(self.device), self.beats_keys.get("revision"))
        r = self._keys.transcribe(audio)
        return {"duration": r["duration"], "beats": [float(b) for b in r["beats"]], "downbeats": [float(b) for b in r["downbeats"]], "keys": r["keys"]}

    @torch.no_grad()
    def probabilities(self, audio: Path | str, info: dict) -> dict[str, np.ndarray]:
        """The model's 25 Hz ``detail`` / ``bass`` / ``boundary`` probabilities of ``audio`` given ``info``
        (``beats``, ``downbeats``, ``keys``)."""

        o = song_logits(self.model, load_stereo(Path(audio)), info)
        return {"detail": torch.softmax(o["detail"], -1).cpu().numpy(), "bass": torch.softmax(o["bass"], -1).cpu().numpy(),
                "boundary": torch.sigmoid(o["chord_boundary"]).cpu().numpy()}  # fmt: skip

    def transcribe(self, audio: Path | str, info: dict | None = None) -> dict:
        """``{"duration", "beats", "downbeats", "keys", "chords": [Chord, ...]}`` of ``audio``; ``info`` (beats,
        downbeats, keys as ``beats_and_keys`` returns them) skips SheetSage2-Refined."""

        info = info or self.beats_and_keys(audio)
        p = self.probabilities(audio, info)
        chords = self.decoder(p["detail"], p["bass"], p["boundary"], info["beats"], info.get("downbeats", []), info.get("keys", []))
        return {"audio": str(audio)} | info | {"chords": chords}


def to_lab(chords: list[Chord], notation: str = "harte") -> str:
    """``start<TAB>end<TAB>label`` lines; ``notation`` "harte" (mir_eval, ``A:min7/b7``) or "symbol" (``Am7/G``)."""

    return "".join(f"{c.start:.3f}\t{c.end:.3f}\t{c.symbol if notation == 'symbol' else c.label}\n" for c in chords)
