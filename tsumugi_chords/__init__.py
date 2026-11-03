"""Tsumugi-Chords: detailed chords of whole songs (root, quality with tensions, bass) from a note-transcription
backbone with a chord reader, decoded on SheetSage2-Refined's beat grid."""

__version__ = "0.1.0"

from .decode import Chord, ChordDecoder  # noqa: E402
from .tracker import DEFAULT_MODEL, HeadChords  # noqa: E402

__all__ = ["Chord", "ChordDecoder", "DEFAULT_MODEL", "HeadChords", "__version__"]
