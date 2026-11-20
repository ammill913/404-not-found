# AI vs Original Music — hackathon data

Can a pretrained audio embedding tell **which text-to-music generator** made a track, and what happens with an **unknown generator** or **real music**?

## Data

```
audio/{train,val,test}/<source>/<ytid>.wav   700 / 150 / 150 captions x 6 sources
audio/holdout/<source>/<name>.wav            61 other captions, Level 3 (test-only)
audio/level4/<model>/<name>.wav              61 clips from each of 3 newer generators (not in metadata.csv, test-only)
metadata.csv                                 one row per clip
features/mert_v1_{95m,330m}.npz              pre-extracted MERT embeddings (see below)
evaluate.py                                  optional scorer
example_submission/                          example prediction files and the scores evaluate.py gives for them
```

- **Sources:** `real` (MusicCaps), 5 generators from FakeMusicCaps (`musicgen`, `musicldm`, `audioldm2`, `stable_audio_open`, `mustango`) and `suno` (SunoCaps; holdout only).
- **Holdout:** 61 clips per source, and 243 Suno clips (4 versions per caption; two prompted with the caption, two with the aspect list).
- **Level 4 (extra, test-only):** `audio/level4/{acestep15,sa3_medium,yue2}/` holds 61 clips each from three newer generators: ACE-Step 1.5, SA3 Medium and YuE 2. They are not in `metadata.csv` and have no features in `features/`. Use them only to test the final models, with no training, tuning or model selection on them.
- **Format:** all audio is 16 kHz, mono, 16-bit, exactly 10.000 s. All sources were converted to this same format before release, so format differences are not a cue.
- **Captions:** a split never shares a caption with another split. Clips with the same `caption_id` share a prompt, so keep them together in cross-validation (`GroupKFold`).
- **Columns:** `id, caption_id, source, label (real/fake), split, path, caption, aspect_list`, plus the original `ytid, start_s, end_s`, audio format fields, `suno_version`, `suno_prompt_type` (caption/aspect) and `suno_instrumental`.

## Pre-extracted features

`features/` holds frozen [MERT](https://huggingface.co/m-a-p/MERT-v1-95M) embeddings for every clip (all splits), so you can start modelling right away:

| File | Model | `mean`, `std` shape | Size |
|---|---|---|---|
| `features/mert_v1_95m.npz` | `m-a-p/MERT-v1-95M` | `[6609, 13, 768]` | ~0.26 GB |
| `features/mert_v1_330m.npz` | `m-a-p/MERT-v1-330M` | `[6609, 25, 1024]` | ~0.68 GB |

- `ids[i]` is the `id` of row `i`, and the rows are in the same order as `metadata.csv`.
- **Axis 1 is the layer.** Index 0 is the CNN feature-encoder output, and 1..L are the transformer layers. Different layers carry different information (low layers sit closer to the signal, high layers to musical content), so try more than the last one.
- `mean` and `std` are the mean and standard deviation over time of each layer's frame embeddings (~75 frames/s, ~750 frames per clip), stored as float16.
- **Preprocessing:** the 16 kHz clips were resampled to 24 kHz (MERT's rate), and the MERT feature extractor normalised each clip to zero mean and unit variance. The input therefore has no content above 8 kHz, and absolute loudness is removed.
- Each file also stores `model_id`, `revision`, `sr_in`, `sr_model` and `pooling`.

```python
import numpy as np, pandas as pd
meta = pd.read_csv("metadata.csv")
z = np.load("features/mert_v1_95m.npz")
assert (z["ids"] == meta["id"].values).all()
X = z["mean"][:, 6].astype("float32")   # layer 6; or np.concatenate([z["mean"][:, 6], z["std"][:, 6]], 1)
train = meta["split"] == "train"; y = meta["source"]
```

To use features for modified clips (see *Cue investigations*), or features from another model, extract them yourself. The same pipeline applies: `transformers` 4.x with `trust_remote_code=True` (MERT's code does not work with transformers 5), resample to 24 kHz, pass `output_hidden_states=True`, and pool over time.

## Levels

Use pretrained models for the features and do not fine-tune them.

| Level | Model | Train on | Evaluate on | Question |
|---|---|---|---|---|
| 1 | `gen5` | 5 generators (**no real**) | `test`, 750 clips | Which generator? |
| 2 | `gen6` | 5 generators + real | `test`, 900 clips | Which source, and real vs generated? |
| 3A | `gen5` | as Level 1 | `holdout`, 609 clips | Real and Suno are unknown |
| 3B | `gen6` | as Level 2 | `holdout`, 609 clips | Suno is unknown |

In Level 3 a model may answer `unknown` instead of one of its classes. When to do so is up to you (a threshold, a distance, ...). Choose the rule on `val`.

## Metrics to report

- **Level 1:** accuracy, macro-F1, confusion matrix.
- **Level 2:** accuracy, macro-F1, confusion matrix, and real-vs-fake balanced accuracy (a clip is "fake" if its label is not `real`).
- **Level 3 (A and B):**
  - for each unknown group, the share of its clips labelled `unknown`;
  - the share of **known** clips wrongly labelled `unknown` (false alarm);
  - a table of where each source's clips go (share per label).

Everything else (other metrics, uncertainty, ablations, plots) is your own choice. With 61 captions, differences of a few points are likely noise. Say how you checked whether a difference is real.

## What to submit

Besides the report, one CSV of predictions per level (see `example_submission/`). The ids are in `metadata.csv`.

| File | Columns | Rows |
|---|---|---|
| `level1.csv` | `id, pred` | test clips of the 5 generators (750) |
| `level2.csv` | `id, pred` | all test clips (900) |
| `level3.csv` | `id, pred_gen5, pred_gen6` | all holdout clips (609) |

- `pred` is a source name. Level 3 also allows `unknown`. The `gen5` column must not contain `real`.
- Leave out a Level 3 column if you did not run that model.

## Optional scorer

```
python evaluate.py --level1 level1.csv --level2 level2.csv --level3 level3.csv --out results/
```

It prints the metrics above and writes `results/scores.json` and `results/report.txt`. Give any subset of the levels. The files in `example_submission/` are simulated predictions with roughly 75% accuracy, not a real model. They only show the file format, and `example_submission/scores/` shows what the scorer prints for them. Don't treat their scores as a baseline.

## Cue investigations

Your model may use shortcuts that have nothing to do with the generator (silence at the start or end, loudness, compression history, vocals, clipping, ...). Test this: modify the clips (add or remove silence, normalise loudness, MP3 round trip, ...), predict again, save the result in the same format (for example `level2_add_tail.csv`), and compare the metrics with the unmodified run.

## Rules

1. Train on `train` and tune on `val`. `holdout` is for the final run only: no training, no threshold tuning and no model selection on it, and report the result of a single run.
2. The `gen5` model never sees real audio in training.
3. In your report, say what each model was trained on and anything that deviates from these rules.
