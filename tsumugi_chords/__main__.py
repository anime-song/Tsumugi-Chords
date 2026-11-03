"""Chords of audio files: python -m tsumugi_chords AUDIO [AUDIO ...] -> OUT_DIR/<stem>.chords.lab (start, end,
label: chord symbols such as Am7/G, or with --notation harte mir_eval's Harte syntax such as A:min7/b7) and
<stem>.chords.json (the chords with both labels, root / quality / bass, plus the beats, downbeats and keys used)."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .tracker import DEFAULT_MODEL, HeadChords, to_lab


def main() -> None:
    parser = argparse.ArgumentParser(prog="tsumugi-chords", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local model folder")
    parser.add_argument("--revision", help="the Hugging Face repo's branch, tag or commit")
    parser.add_argument("--keys-model", help="SheetSage2-Refined model for the beats and keys (default: the one config.json names)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "chords")
    parser.add_argument("--notation", choices=("harte", "symbol"), default="symbol",
                        help="labels of the .lab file: symbol (Am7/G) or harte (mir_eval, A:min7/b7)")
    parser.add_argument("--memory-fraction", type=float, default=0.6, help="cap on this process's share of GPU memory")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device).index or 0)
    tracker = HeadChords.load(args.model, args.device, args.revision, args.keys_model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for audio in args.audio:
        started = time.perf_counter()
        result = tracker.transcribe(audio)
        (args.out_dir / f"{audio.stem}.chords.lab").write_text(to_lab(result["chords"], args.notation), encoding="utf-8")
        record = result | {"chords": [asdict(c) for c in result["chords"]]}
        (args.out_dir / f"{audio.stem}.chords.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        print(f"{audio.name}: {len(result['chords'])} chords in {time.perf_counter() - started:.1f} s -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
