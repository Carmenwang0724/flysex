# Research notebooks

These notebooks cover supervised dataset preparation, model training, and error
analysis. For routine scoring and ranking, use the command line workflow in the
[repository README](../README.md).

Human annotations are used only as training targets and evaluation references.
Inference uses image scores and ranks and takes no expected male count.

| Notebook | Purpose | Required inputs | Main outputs |
| --- | --- | --- | --- |
| `01_build_dataset.ipynb` | Extract aligned crops for supervised learning | Human annotation workbook, original stills, FlyTracker CSVs and calibration files | `../data/labeled_crops.npz` |
| `02_train_and_validate.ipynb` | Validate across recording dates, then train deployment models | Labelled crop dataset; cached or downloadable ImageNet weights | `../data/cv_predictions.npz`, `../model/flysex_seed0.pt`, `../model/flysex_seed1.pt` |
| `06_error_analysis.ipynb` | Review held-out scores against human annotations | Crop dataset, matching held-out predictions, original session files | Plots and review CSVs under `../results/research-audit/` |

## Setup

From the repository root:

```bash
python -m pip install -e ".[research]"
cd notebooks
jupyter lab
```

Run each notebook from its own directory. Place private research inputs in this
layout, or edit the configuration cell for your local folders:

```text
data/
  annotations.xlsx
  labeled_crops.npz
  cv_predictions.npz
  sessions/
    07142026/
    07172026/
    7212026/
model/
notebooks/
results/
```

The session names and arena coordinates are configured for the existing July 2026
recordings. Each session needs `frame_*.png` stills, either directly in the session
folder or in `photos/`. Each arena needs one `*_arena_N-trackfeat.csv/` directory
with `fly1.csv` through `fly5.csv` and `calibration.mat`; an arena subfolder is
accepted. Adapt the geometry and naming configuration for different recordings.

The annotation workbook uses per-arena tabs such as `0714_a2`, annotation image
names in the first column, and five fly columns. `M`, `F`, and `?` mean male,
female, and unknown. A blank fly cell is interpreted as female only in a row that
has at least one annotation, following the original annotation convention. Unknown
flies are omitted. Confirm this convention before using another workbook.

The dataset and prediction files must be trusted: their NumPy object arrays need
`allow_pickle=True`. The error analysis checks provenance and label alignment
before using cached predictions. Research inputs and generated outputs are ignored
by Git and are not supplied with these notebooks.

## Training and Colab

Notebook 02 also searches the current working directory for an uploaded
`labeled_crops.npz`, so it can run on Colab. It requires the same Python
dependencies. By default it trains two models for each held-out date, then two
deployment models; a GPU is recommended. The first model construction downloads
ImageNet weights when they are not already cached. Routine inference does not
need this training step.

Training writes model files to `../model/`. On Colab, explicitly download those
files before the runtime expires. Locally, preserve any released checkpoints
before replacing files with the same names.

Notebook 06 can read uploaded NumPy files too, but its tracker-context analysis
still requires the matching session folders. A dataset upload alone is enough
for notebook 02, not for rebuilding crops or running every error-analysis cell.

## Interpreting the analysis

The research notebooks use a score above 0.5 for binary evaluation calls and
measure confidence outside the 0.1 to 0.9 band. These are evaluation settings;
they impose no count quota on inference. Scores are model outputs, and high
confidence does not guarantee a correct classification. Report retained coverage
alongside accuracy when filtering uncertain crops.

Validation holds out recording dates to reduce leakage between near-duplicate
frames. It measures crop classification against human annotations on those dates.
It does not establish continuous fly identity or guarantee performance on a new
recording. Review score/annotation disagreements visually before revising labels.

## Verification of this migration

The notebook structure remains at 17, 20, and 26 cells respectively. Saved outputs
and execution counts are cleared.

- Notebook 01: all definitions loaded; workbook parsing, aligned sampling,
  quantisation, still indexing, and calibration reading checked. The workspace
  workbook had zero usable annotated rows, so a full dataset rebuild was not run.
- Notebook 02: definitions, loading of 1,124 existing labelled crops, sample plot,
  augmentation, and model construction with cached ImageNet weights checked.
  Training, cross-validation, and checkpoint saving were not rerun.
- Notebook 06: every code cell executed against the existing matched dataset,
  cached predictions, and session files, using temporary in-memory path overrides.
  It produced plots and a review list of 146 score/annotation disagreements from
  1,124 crops. These are cached-model results, not a new validation run.

Verification outputs were saved under the ignored `results/research-audit/` folder.
