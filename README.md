# Fly sex scoring

Score every tracked fly in a sampled video frame, then rank the flies within each well from highest to lowest model score. Higher scores indicate stronger model evidence for male sex.

Daily use runs from the command line. It requires recording stills, FlyTracker tables, calibration files, and the supplied model weights. It does not require a stocking sheet, male counts, or annotation labels. Research notebooks remain available for building training data, training, and evaluating the model.

## Start here

Use Python 3.10 or newer. Open a terminal in this repository and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Copy `configs/example.json` to `configs/local_trial.json`. Set its paths, wells, and verified `frame_zero`, then run:

```bash
python -m flysex predict --config configs/local_trial.json
```

Start with `max_frames: 5` and inspect the preview crops and position overlays. Confirm both spatial and time alignment before processing the full recording. The [quickstart](docs/QUICKSTART.md) explains the input layout and each step.

For a first run with the original July 14 snapshot export, put that session folder at `data/07142026/` and use `configs/july14_preview.json`. It contains the frame origin checked for that specific export and processes five stills each from wells 2 and 6. Other recordings or regenerated tracker exports need their own verified configuration.

Once the scores have been reviewed, export ranked copies of the tracker tables:

```bash
python -m flysex export \
  --session /path/to/your-trial \
  --scores results/trial/ranked_scores.csv \
  --output results/trial/ranked_tracks
```

## Outputs and interpretation

| Output | Purpose |
|---|---|
| `ranked_scores.csv` | Every scored fly, its original tracker ID, model score, and rank within that well and frame |
| Preview contact sheets and overlays | Check crop quality and whether tracker positions match the stills |
| `run_summary.json` | Record the run configuration and coverage |
| Ranked tracker tables | Optional copies reordered at the exact rows with complete scores |
| `rank_coverage.csv` | Identify the exported rows for which rank order is available |
| `rank_mapping.csv` | Trace each exported rank back to the original tracker ID and score |

In exported tables, `fly1.csv` holds the highest-scoring fly at a scored row, `fly2.csv` the next highest, and so on. A rank is not proof of sex or a persistent animal identity. An all-female group still has a highest-scoring fly, and the animal at rank 1 may change between frames.

**Downstream analyses using ranked tracker tables must filter by `rank_coverage.csv` where `ranked == True`.** Unscored rows retain their original tracker order. The exporter does not fill gaps or carry a ranking forward between sampled frames.

The optional sex call (`score > 0.5`) and confidence flag (`score < 0.1` or `score > 0.9`) describe individual predictions. They do not select flies or change their ranking.

## Model evidence

Held-out validation across three recording dates gave mean accuracy **0.870** and mean AUC **0.962**. Retaining only predictions outside the 0.1 to 0.9 score band gave mean accuracy **0.951**, covering about **72%** of crops. These results describe those three dates; they do not establish accuracy for a new recording or night footage. See [methods and limitations](docs/METHODS.md).

## Repository layout

```text
configs/       Trial configuration examples
src/flysex/    Inference, ranking, export, and frame-extraction code
model/         flysex_seed0.pt and flysex_seed1.pt
notebooks/     Optional dataset, training, and evaluation notebooks
docs/          Quickstart and methods
tests/         Checks for the command-line workflow
```

Recordings, annotations, generated results, and personal project notes are kept outside version control.

For the research notebooks and optional video extraction dependency:

```bash
python -m pip install -e '.[research,video]'
```

Existing weights are sufficient for routine prediction. Retraining is a separate research step that uses annotation labels and evaluation split by recording date.

For code changes, follow [the code style](docs/CODE_STYLE.md). The [verification record](docs/VERIFICATION.md) describes the checks performed before sharing this version.
