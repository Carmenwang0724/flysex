# Run one trial

The routine workflow is: prepare a trial, confirm alignment, score and review the sampled frames, then optionally export ranked tracker tables. No stocking sheet, male-count setting, or annotation workbook is used.

## 1. Install

Use Python 3.10 or newer. In a terminal opened at the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows activation is `.venv\Scripts\activate`. Keep both supplied weights in `model/`: `flysex_seed0.pt` and `flysex_seed1.pt`.

## 2. Prepare the trial

A trial needs full-frame stills, the per-fly FlyTracker CSV files for each requested well, and its `calibration.mat`. These layouts are supported:

```text
your-trial/
    photos/
        frame_000120_00-00-04.png
        ...
    arena_2/
        your-trial_arena_2/
            your-trial_arena_2-trackfeat.csv/
                fly1.csv
                fly2.csv
                ...
        calibration.mat
```

```text
your-trial/
    frame_000120_00-00-04.png
    ...
    your-trial_arena_2-trackfeat.csv/
        fly1.csv
        fly2.csv
        ...
        calibration.mat
```

Default geometry assumes stills of 2592 × 1944 pixels and the original six-well layout. Inspect the actual images. A correctly sized image can still be truncated or partly black; replace damaged stills from the original video before use.

If stills are missing, install the optional video dependency and extract them:

```bash
python -m pip install -e '.[video]'
python -m flysex extract \
  --video /path/to/recording.mp4 \
  --output /path/to/your-trial/photos \
  --stride 5
```

This samples every fifth decoded video frame. The numeric part after `frame_` is the zero-based video frame number. Extraction alone does not establish the relationship between the video and FlyTracker rows.

## 3. Configure and verify alignment

For the original July 14 snapshot export, place the existing session folder at `data/07142026/` and use the supplied `configs/july14_preview.json`. Run:

```bash
python -m flysex predict --config configs/july14_preview.json
```

This preview processes five stills each from wells 2 and 6. Its `frame_zero: 54647` was checked against that original export. It is not a default for other trials or newly generated tracker tables. The raw session data is supplied separately from the repository.

For a different trial, follow the configuration steps below.

Copy `configs/example.json` to `configs/local_trial.json`. Edit:

| Setting | What to enter |
|---|---|
| `session_directory` | Trial folder containing the tracker outputs |
| `photos_directory` | Optional separate folder containing the stills |
| `model_directory` | Folder containing both model weight files |
| `output_directory` | Destination for scores and review images |
| `wells` | Well numbers to process, such as `[2]` |
| `frame_zero` | Video frame number corresponding to tracker row index 0 |
| `max_frames` | Use `5` for an initial review, then remove this limit for the full run |
| `device` | `"auto"` selects an available device |
| `seed` | Keep a fixed integer, such as `0`, for reproducible sampling |

Relative paths are resolved from the configuration file's directory. For a config in `configs/`, `"../model"` points to the repository's `model/` folder, and `"../results/trial"` points to its results folder. Absolute paths are also supported.

`frame_zero` is required; replace the example's `null` with a verified integer. The mapping is:

```text
tracker row index = video frame number - frame_zero
```

Tracker row indices start at zero and exclude the CSV header. Use `frame_zero: 0` only after confirming that tracker row 0 is video frame 0. A tracker run made from a trimmed video or an offset segment needs a different value.

For some historical files, the old notebook mapping was `row = frame - first_original_still + 1`. That corresponds to `frame_zero = first_original_still - 1`. Confirm it separately for each recording. Do not infer the origin from the first still currently in a folder: missing, filtered, or sampled images can change that number without changing tracker timing.

The default `well_origins` are inherited from the original plate layout. If the camera, plate, panorama crop, or tracker input geometry changed, supply the correct origins in the configuration. Object keys are well numbers as strings, for example `"2"`.

## 4. Run a small preview

With `max_frames` set to `5`:

```bash
python -m flysex predict --config configs/local_trial.json
```

Open the generated contact sheets and overlays in your configured output folder. Check that:

- Markers lie on the fly bodies in each requested well.
- Crops contain the intended flies with plausible body orientation and scale.
- Matching is correct in several frames where flies have moved, not only in a nearly static frame.

A misplaced marker can indicate incorrect geometry or a frame offset. Use the source video and tracker positions at multiple known frame numbers to distinguish them. Stills taken far apart can help reveal a time mismatch. Do not export for analysis until this relationship is verified.

Remove `max_frames` from the configuration and choose a new, empty `output_directory`, then rerun the same command to process all usable sampled frames. The preview output remains available for comparison.

## 5. Review the scores

`ranked_scores.csv` contains one row per scored fly. Within each well and sampled frame, every scored fly receives a rank ordered by descending score. The original tracker ID remains available so the source measurement can be traced.

| Column | Meaning |
|---|---|
| `frame`, `well` | Input video frame number and well number |
| `tracker_row` | Zero-based data row in the original tracker tables |
| `fly` | Original tracker ID at this frame |
| `score` | Model score from 0 to 1 |
| `rank` | Position in descending score order, starting at 1 |
| `rank_gap` | Score minus the next-ranked fly's score; blank for the last rank |

Equal scores are ordered by original tracker ID. The tie-breaking order adds no evidence about sex.

Higher scores mean stronger model evidence for male sex. Rank 1 means highest score in that group; it does not establish that the fly is male. The same physical fly need not keep the same rank or tracker ID between frames.

`sex` and `confident` are descriptive columns. The sex threshold is `score > 0.5`; confidence is `score < 0.1` or `score > 0.9`. All scored flies participate in ranking, including uncertain predictions. Inspect the score values and review images instead of interpreting the rank as a confidence guarantee.

Check `run_summary.json` for the run configuration and actual coverage. Skipped or unavailable frames are not predictions.

## 6. Export ranked tracker tables

Prediction creates scores and review images. Export is a separate command:

```bash
python -m flysex export \
  --session /path/to/your-trial \
  --scores results/trial/ranked_scores.csv \
  --output results/trial/ranked_tracks
```

These command-line paths are relative to your terminal's current directory, unlike paths inside a configuration file. Adjust them to match your run. Choose a new or empty export directory, separate from the source trial folder.

The exported copies preserve the original tracker table structure. At each exactly scored row with a complete ranking, the highest-scoring fly goes into `fly1.csv`, the next into `fly2.csv`, and so on. Unscored rows keep the original tracker order. Original files are not edited, and rankings are not propagated across gaps.

The export root contains `rank_mapping.csv`, `rank_coverage.csv`, and a README. Per-well tracker copies and calibration files are under `wellN/`. The mapping table records the original fly ID and score at each ranked row.

**Read the exported `rank_coverage.csv` and retain only rows where `ranked == True` whenever using the tables as ranked data.** With a stride of 5, intervening rows normally remain unranked. A full-length output CSV is not evidence of full-length score coverage.

Do not join consecutive `fly1.csv` positions into an individual animal's trajectory without a separate identity-tracking method.

## Common problems

| Symptom | Check |
|---|---|
| No usable stills | Photo path, filename format, image completeness, and the configured frame mapping |
| Missing tracker files or calibration | Trial layout and well number |
| Overlay markers miss the flies | `frame_zero`, well origins, and image geometry |
| Crops are dark or incomplete | Original stills and whether the requested well is actually visible |
| Scores look uncertain or unexpected | Preview crops and performance on this recording; rank alone cannot validate sex |
| Most exported rows are unranked | Sampling stride, `max_frames`, skipped frames, and `rank_coverage.csv` |
