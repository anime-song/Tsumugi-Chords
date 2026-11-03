# Tsumugi-Chords

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Tsumugi--Chords-yellow)](https://huggingface.co/anime-song/Tsumugi-Chords)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Weights: CC BY-NC 4.0](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-orange)](https://creativecommons.org/licenses/by-nc/4.0/)

Detailed chords of whole songs: root, quality (62 qualities, tensions included) and bass note, so slash chords
and tension chords come out as they are played.  A small note-transcription backbone with a chord reader
(16.9 M parameters) and a semi-Markov decoder on the beat grid; the beats, downbeats and keys come from
[SheetSage2-Refined](https://github.com/anime-song/SheetSage2-Refined).  Details and all scores are on the
[model card](https://huggingface.co/anime-song/Tsumugi-Chords).

## Results

Held-out songs (mostly Japanese pop and anime songs, hand-annotated), `mir_eval` weighted accuracies:

<!-- tables:summary -->
| Test songs (95) | SheetSage2 | SheetSage2-Refined | Tsumugi-Chords |
|---|---|---|---|
| root | 0.821 | 0.864 | **0.899** |
| tetrads | 0.678 | 0.770 | **0.818** |
| notes (tensions included) | 0.586 | 0.689 | **0.737** |
| slash chords (bass, exact) | 0.366 | 0.520 | **0.641** |
| tension chords: recall / precision | 0.000 / — | 0.232 / **0.485** | **0.550** / 0.410 |
<!-- /tables:summary -->

The chord model that Tsumugi-Chords partly learnt from was trained on data that includes these test songs; see the
[model card](https://huggingface.co/anime-song/Tsumugi-Chords#results).

## Install

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/anime-song/Tsumugi-Chords
```

This installs SheetSage2-Refined too.  The weights are downloaded from Hugging Face on first use: this model
(68 MB) and the parts of SheetSage2-Refined that the beats and keys need (2.8 GB).  A CUDA GPU is recommended
(a four-minute song in about 4 s); the CPU works too (about 2.5 minutes).

## Usage

```bash
tsumugi-chords song.mp3                       # -> outputs/chords/song.chords.lab and song.chords.json
tsumugi-chords song.mp3 --notation symbol     # chord symbols (Am7/G) in the .lab instead of Harte (A:min7/b7)
tsumugi-chords *.mp3 --out-dir chords --device cpu
```

```python
from tsumugi_chords import HeadChords

tracker = HeadChords.load()
result = tracker.transcribe("song.mp3")
for chord in result["chords"]:
    print(chord.start, chord.end, chord.label, chord.symbol)    # e.g. 12.48 13.71 A:min7/b7 Am7/G
```

`result` also holds the `beats`, `downbeats` and `keys` the chord model read.  To use beats and keys of your own,
pass them: `tracker.transcribe("song.mp3", {"beats": [...], "downbeats": [...], "keys": [{"start", "end", "key"}]})`.

Each chord has its Harte label (`mir_eval`) and a chord symbol spelt in its key (`symbol`).  Options:
`--notation` (`harte` or `symbol` for the .lab file), `--model` (a Hugging Face repo id or a local model folder), `--revision`, `--keys-model` (another
SheetSage2-Refined folder or repo), `--device`, `--memory-fraction` (the share of GPU memory the process may use).

## License

The code is MIT ([LICENSE](LICENSE)).  The weights, this model's and SheetSage2-Refined's, are
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial use only.
