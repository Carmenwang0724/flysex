# Verification of the initial release

Checked locally on macOS with CPU inference. This is a software migration check, not a new model accuracy estimate.

- Editable installation and both CLI entry points succeeded.
- All 30 automated tests passed, including score ties, invalid/incomplete frames, nonfinite coordinates, missing/corrupt images, frame offsets, video extraction, and export source preservation.
- The unchanged two-model ensemble scored five real stills in each of two wells from the July 14 recording: 50 fly scores across ten well/frame groups.
- Crop contact sheets and position overlays were visually inspected on the real-data preview.
- Exported rows were compared against the source CSVs. Each scored row was exactly the intended permutation. For each well, the other 18,690 rows remained unchanged.
- The cached research audit ran fully on 1,124 labelled crops and reproduced 146 disagreements. Training notebook definitions, dataset loading, plotting, and model construction were exercised without retraining.
- Dataset notebook helper functions were exercised. A complete dataset rebuild was not run because the available workbook had no filled annotation rows.
- Research notebooks retain their cell counts and contain no saved outputs or local absolute paths.

The model checksums are in `model/manifest.json`. Raw inputs and verification outputs are local files excluded from the repository. GitHub Actions is configured; a remote CI result will only exist after the repository is pushed.

Run the automated checks after installation:

```bash
python -m unittest discover -s tests -v
```

Install the `video` extra to include the synthetic video extraction test. A fresh full dependency installation on Windows or Linux has not been exercised locally.
