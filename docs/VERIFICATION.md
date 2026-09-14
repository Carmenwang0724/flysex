# Verification of the initial release

Checked locally on macOS with CPU inference. This is a software migration check, not a new model accuracy estimate.

## Code style and installation audit

The Python source, tests, and all three research notebooks were audited together. Project configuration names now use lowercase `snake_case`. Functions have one-sentence docstrings, logical blocks have operation comments, and multi-argument calls are expanded. External API names and saved data field names are preserved.

- All 30 automated tests passed in the existing environment and in a separate environment installed from a clean source copy without system site packages.
- The clean installation used Python 3.12 on macOS with NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, PyTorch 2.14.0, torchvision 0.29.0, Pillow 12.3.0, and PyAV 18.1.0.
- The clean copy completed prediction and export for five stills in each of two wells. All 50 fly ranks matched the pre-audit preview. Export comparisons confirmed the intended row permutations and 18,690 unchanged rows per well.
- Notebook 01 executed all cells with a small workbook reconstructed from one complete existing annotation group, producing five crops.
- Notebook 02 executed setup, dataset loading, plotting, and one training epoch on a single eight-crop batch, followed by inference. Released checkpoints were not changed.
- Notebook 06 executed all cells against the existing cached data and again produced 146 disagreements from 1,124 crops.
- Notebook cell counts remain 17, 20, and 26. Saved outputs and private paths are excluded from the notebook files.

## Original migration checks

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

Install the `video` extra to include the synthetic video extraction test. Windows and Linux installation have not been exercised locally. The configured GitHub Actions workflow has not run remotely because the repository has not been published yet.
