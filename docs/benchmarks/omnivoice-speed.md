# OmniVoice sentence speed (2026-10-31)

Where the time of one OmniVoice voice-cloning sentence goes, and what made it faster.

## Setup

- M2 Max, 32 GB. Release build, compiled `mlx.metallib`.
- `aufklarer/OmniVoice-MLX-fp16`, 16 steps, Polish, a fixed duration per sentence.
- One reference clip of 8.2 s (205 reference tokens), encoded once with `encodeReference`.
- 3 timing sentences: short (1.1 s of audio), medium (4.8 s), long (8.5 s).
- Each comparison is paired: baseline and candidate alternate per sentence and repeat, in the same process.
  Every comparison ran in two processes, except the precision sweep (one process per variant).
- Noise floor: the same code in two processes differed by 4-6%, and up to 7% inside one run.

## Where the time goes (before)

| sentence | positions (cond / target) | time | cond pass | uncond pass | audio heads | CFG + top-k | codec |
|---|---|---|---|---|---|---|---|
| short | 292 / 30 | ~1.2 s | 76% | 20% | 2% | 1% | 1% |
| medium | 407 / 122 | ~1.7 s | 67% | 27% | 2% | 1% | 3% |
| long | 523 / 219 | ~2.4 s | 64% | 28% | 2% | 1% | 4% |

The two backbone passes per step are 93-96% of the time. The cond pass carries the reference clip and the text
on every step, so a short sentence costs about as long as its own audio.

## Changes

1. **Packed guidance forward** (`OmniVoiceModel.guidanceLogits`, `generateTokens(packed: true)`, the default).
   The cond sequence and the uncond sequence (the target alone) run as one backbone forward. Each segment
   attends only to itself and its RoPE positions start at 0, so no mask is needed. Only the target positions go
   through the audio heads. 6-12% faster (18 of 18 pairs faster). Same samples for 2 of 3 sentences; the third
   differed in 1 of 976 tokens (float summation order).
2. **fp16 compute.** `loadWeights` keeps the bundle's dtype instead of an fp32 upcast; the guidance math
   (log-softmax, CFG, top-k) stays in fp32. About 21% faster on top of change 1 (8 pairs, -16% to -26%). Most
   tokens differ from the fp32 run (a new rendering, not a copy), so it was checked by ASR read-back and by
   listening: WER on 13 sentences (113 words) was the same as fp32, and the rendering sounded as good.

## Tried, not kept

| candidate | result |
|---|---|
| `compile` of the packed forward | No gain once compiled (-4% to +1%). A new sentence length retraces, so it costs more in normal use. |
| bf16 compute | -6%. |
| int8 weights, fp32 activations | +2% (slower). |
| int8 weights, fp16 activations | -19%: close to fp16, uses less memory. |
| slicing before the audio heads alone | The heads are ~2% of the time: below the noise floor. |

## Not measured

- fp16 range on other voices and texts.
- Voice likeness (for example with a speaker-embedding score).
