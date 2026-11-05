---
license: cc-by-nc-4.0
tags:
- audio
- music
- music-transcription
- chord-recognition
---

# Tsumugi-Chords

Whole songs in, chords out: **root, quality (62 qualities: triads, sixths, sevenths, 9ths, 11ths, 13ths, add, sus,
power chords) and bass note**, so slash chords and tension chords are written as they are played, on the song's
beat grid.

Tsumugi-Chords is a small audio model (16.9 M parameters) of its own: a note-transcription backbone with a chord
reader on top, trained from scratch, plus a semi-Markov decoder over the beats.  It does not use SheetSage2's
weights for the chords.  It reads the beats, downbeats and keys of
[SheetSage2-Refined](https://huggingface.co/anime-song/SheetSage2-Refined)'s decoder-free trackers, which it
loads as a dependency.

- Code (MIT): https://github.com/anime-song/Tsumugi-Chords
- Beats and keys: [anime-song/SheetSage2-Refined](https://huggingface.co/anime-song/SheetSage2-Refined) v0.5.0
- Weights: CC BY-NC 4.0, non-commercial use only

## Quick start

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/anime-song/Tsumugi-Chords
tsumugi-chords song.mp3                 # -> outputs/chords/song.chords.lab and song.chords.json
tsumugi-chords song.mp3 --notation harte    # .lab with Harte labels (A:min7/b7, for mir_eval) instead of Am7/G
```

The first run downloads this model (68 MB) and the parts of SheetSage2-Refined the beats and keys need
(its encoder, 2.7 GB, and four small heads).  A CUDA GPU is recommended; the CPU works (fp32).

```python
from tsumugi_chords import HeadChords

tracker = HeadChords.load()            # device: CUDA if there is one
result = tracker.transcribe("song.mp3")
for chord in result["chords"]:
    print(f"{chord.start:7.2f} {chord.end:7.2f} {chord.label} {chord.symbol}")   # e.g. "  12.48   13.71 A:min7/b7 Am7/G"
# result["beats"], result["downbeats"], result["keys"]: what the chord model read
```

Beats, downbeats and keys of your own can be given instead: `tracker.transcribe(audio, info)` with
`info = {"beats": [...], "downbeats": [...], "keys": [{"start", "end", "key"}, ...]}` (seconds; keys such as
`A:major`), which skips SheetSage2-Refined.  The model was trained on SheetSage2-Refined's, so other trackers may
score lower.

Speed for a four-minute song: about 4 s on an RTX 4070 Ti (beats and keys ~2 s, chords ~2 s); about 2.5 minutes
on an Intel Core i5-12400F (6 threads; beats and keys 65 s, chords 84 s, decoding 1 s).

## How it works

1. **Beats, downbeats and keys** from SheetSage2-Refined's decoder-free trackers (`sheetsage2_refined.keys`).
2. **The chord model** reads stereo audio (25.6 kHz) in 30 s windows, keeping the middle 20 s of each.
   - *Backbone*: a harmonic CQT (5 harmonics, 36 bins per octave, 50 Hz), a convolutional stem and 6 axis-Transformer
     layers (along the frequency row, then along time; width 256), after the AMT model of Tsumugi
     (the author's piano-cover transcription system).  It was first trained as a note transcriber (88 keys,
     instruments) on Tsumugi's MIDI of 18k songs; the note head is not shipped.
   - *Chord reader*: 4 chord query tokens per step, fed the beats (positions in the beat and bar) and the key, merged
     into the chord features.
   - *Outputs* at 25 Hz: root × quality (745 classes: 12 × 62 + no chord), bass (12 + none) and chord boundaries.
3. **The decoder**: a semi-Markov Viterbi over the beat grid that chooses chord segments from the frame
   probabilities, the boundary peaks, chord-length and beat-position tables, and a key-relative chord bigram
   (counted on the training labels and on Tsumugi's chords).  Chords may change on the beats' halves, thirds and
   quarters; a bass-only change to a note outside the held chord costs extra (passing bass notes), and every segment
   pays the labels' log-probability of its bass's interval above the root for its chord family.  Each segment gets
   one root × quality and one bass.
   - *Repeated bars*: before the search each bar's root × quality probabilities are mixed with those of the bars that
     repeat it (bars matched by the chord tones and bass the model hears, under the transposition the bars around
     them agree on, so a chorus a step up still counts), which keeps repeated passages consistent.
   - *After the search*: aug and dim7 chords are spelt from the root their frames favour (their roots share one
     probability), a sound split between two readings over the same bass (Baug/F, F7(9,#11,13)) is joined, and an
     opening bass shorter than 1.25 beats before the root (DM7(9)/F# for a beat, then DM7(9)) is dropped.

### Training

- **Note backbone**: Tsumugi's MIDI transcriptions of 18,266 songs (1,065 hours), 40k steps.
- **Chords, stage 1**: the backbone fine-tuned to Tsumugi's chord model outputs on 2,221 songs.
- **Chords, stage 2**: half Tsumugi's outputs, half the chord annotations of 847 hand-labelled songs.  Two such runs
  (one with CTC and autoregressive auxiliary losses) were weight-averaged.
- **Ensemble distillation**: a further run learnt, on 2,500 more songs, the outputs of an ensemble (the averaged model
  with pitch-shift test-time augmentation, a model that reads the transcribed notes, and an earlier audio chord
  model); the released weights are the average of that run and the averaged model.
- Songs of the unlabelled collections that match an evaluation song by their chords were removed (151 songs).

## Results

SheetSage2 (as released, `chord_full` prompt) and SheetSage2-Refined (its decoder with frame-head rescoring)
against Tsumugi-Chords, on the same audio.

- **Pop songs, test / valid**: 95 and 47 songs, mostly Japanese pop and anime songs, with hand-made beat,
  chord and key annotations (not public).  Tsumugi-Chords never saw their audio or labels.  **But** Tsumugi's chord
  model, whose outputs on other songs Tsumugi-Chords learnt from, was trained on a set that includes these songs, so
  some knowledge of them may have reached the model indirectly.

Chord scores are `mir_eval` weighted accuracies over the annotated span; `notes` compares the root and every
chord tone, tensions included; `slash chords` asks for the bass note of inverted chords; the tension row counts
chords with 9ths, 11ths, 13ths or added notes (SheetSage2's vocabulary cannot write them); `segmentation` is
`mir_eval`'s min(under-, over-segmentation).

<!-- tables:chords -->
| Chords | SheetSage2 | SheetSage2-Refined | Tsumugi-Chords |
|---|---|---|---|
| root, test | 0.821 | 0.864 | **0.901** |
| root, valid | 0.822 | 0.880 | **0.889** |
| majmin, test | 0.871 | 0.886 | **0.918** |
| majmin, valid | 0.845 | 0.885 | **0.904** |
| sevenths, test | 0.762 | 0.829 | **0.867** |
| sevenths, valid | 0.777 | 0.825 | **0.872** |
| tetrads, test | 0.678 | 0.770 | **0.822** |
| tetrads, valid | 0.704 | 0.780 | **0.823** |
| notes (tensions included), test | 0.586 | 0.689 | **0.745** |
| notes (tensions included), valid | 0.636 | 0.727 | **0.774** |
| slash chords (bass, exact), test | 0.366 | 0.520 | **0.635** |
| slash chords (bass, exact), valid | 0.386 | 0.515 | **0.609** |
| tension chords, test: recall / precision | 0.000 / — | 0.232 / **0.485** | **0.545** / 0.441 |
| tension chords, valid: recall / precision | 0.000 / — | 0.221 / **0.606** | **0.531** / 0.522 |
| segmentation, test | 0.914 | 0.928 | **0.948** |
| segmentation, valid | 0.877 | 0.909 | **0.921** |
<!-- /tables:chords -->

## Output

`tsumugi-chords` writes two files per song:

- `<song>.chords.lab`: one chord per line, `start<TAB>end<TAB>label` (seconds).  By default the labels are chord symbols
  as in lead sheets, e.g. `DM7`, `B/D#`, `Em7(9)`, `G7(b9)`, `N` for no chord; with `--notation harte` they are in
  Harte syntax as in `mir_eval`: `D:maj7`, `B:maj/3`, `E:min7(9)`, `G:7(b9)`.
- `<song>.chords.json`: `{"audio", "duration", "beats", "downbeats", "keys", "chords"}`; each chord is
  `{"start", "end", "label", "root", "quality", "bass", "symbol"}` (root and bass as pitch classes, C = 0; `null` for
  no chord or no separate bass).

The chord symbols are spelt in the key the chord sits in: sharps or flats by its key signature, other notes by their
degree (Bb7 in C major, E7/G# in A minor, A#dim7 before Bm in D major).  Qualities are written `M7`, `m7`, `m7-5`,
`7-5`, `dim7`, `aug`, `sus4`, `add9`, `69`, tensions in parentheses (`7(9,13)`).

## Limitations

- Long chords are sometimes split: about one in six annotated chords of 6 beats or more in the valid songs is broken
  up (by half a beat or more), mostly by a chord with another root.  Many of these are pedal or vamp passages where the
  annotation writes one long chord over a moving bass line or riff, so part of this is a labelling convention.
- Written tension chords are right about half the time (precision 0.44 / 0.52 on test / valid).
- Chord changes are placed on the tracked beat grid, so an error of the beat tracker (e.g. half tempo, odd meters)
  also moves the chords.
- The training songs are mostly Japanese pop and anime songs; other styles were not measured.

## License

The weights in this repository and the decoder's tables are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see `LICENSE`) and may not be used for commercial
purposes.  The beats and keys come from SheetSage2-Refined, whose weights (CC BY-NC 4.0, adapted from SheetSage2
and MERT-v2-FullSong) are downloaded from its own repository.  The inference code is MIT.

## Acknowledgements

The beats and keys come from SheetSage2-Refined, which is built on [SheetSage2](https://huggingface.co/m-a-p/SheetSage2)
and [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) by the Multimodal Art Projection (m-a-p) team.

## 日本語

曲全体の音声からコード（根音 × 62 種の分類（テンション込み）とベース音）を推定するモデルです。分数コードや
テンションコードもそのまま書き出します。
SheetSage2 の重みは使わず、音符の採譜で学習したバックボーン（16.9 M パラメータ、ゼロから学習）にコード読み取り部を
載せ、拍の上で半マルコフ型の Viterbi でコードを決めます。拍・小節線・キーは SheetSage2-Refined の
デコーダーなしのトラッカーから受け取るため、SheetSage2-Refined に依存しています。
テストの曲（95 曲）で tetrads 0.822、分数コード 0.635 です（SheetSage2-Refined は 0.770 / 0.520）。
ただし、学習で出力をまねた Tsumugi のコードモデルはテスト曲を含むデータで学習しているので、この数字は
やや楽観的な可能性があります。
重みは CC BY-NC 4.0（非商用のみ）、推論コードは MIT です。
