# Methods and interpretation

This repository scores images of tracked flies and ranks all flies within each well and sampled frame by descending model score. Higher scores indicate stronger model evidence for male sex. Inference and export use no stocking records, male-count quotas, or annotation labels. Human labels are used for supervised model training and held-out evaluation.

## Inputs and image preparation

Inputs are full-frame recording stills, FlyTracker per-fly feature tables, and per-well calibration files. Tracker positions and body orientation identify the image crop for each fly.

The original geometry uses 2592 × 1944 stills and tracker coordinates from a 1944 × 1458 panorama. Add the well's crop origin to its local tracker coordinates, then multiply by `2592 / 1944` to map into the still image.

| Well | Panorama crop origin (x, y) |
|---|---|
| 1 | (260, 171) |
| 2 | (832, 163) |
| 3 | (1402, 171) |
| 4 | (267, 748) |
| 5 | (824, 748) |
| 6 | (1387, 733) |

These origins describe the original acquisition setup. They must be rechecked when the plate position, camera, or tracker input crop changes.

Video and tracker time alignment is configured explicitly:

```text
tracker row index = video frame number - frame_zero
```

`frame_zero` is the video frame corresponding to tracker row index 0. It cannot safely be recovered from the first retained still. Spatial overlays and source-frame comparisons across multiple moving frames are required to check this mapping.

Image preparation follows the trained model's crop construction:

1. Flatten illumination using the still divided by a 121 × 121 local box-filtered image.
2. Sample a 96 × 96 crop along the tracked body axis, using the negative of FlyTracker's orientation to account for the image coordinate convention.
3. Normalize physical scale with the well's calibrated pixels per millimetre. Sampling step is proportional to the well's pixels per millimetre, with reference value 15.7 and the original image-to-panorama scale.
4. Map the flattened intensity interval 0.35 to 1.15 to an 8-bit crop for model input.

Incorrect geometry, timing, calibration, or incomplete source images can invalidate the score even when the software completes successfully. Contact sheets and overlays support this input check.

## Model and training

The model is an ImageNet-pretrained ResNet-18 adapted to grayscale crops and a single binary output. The original three input-channel kernels are summed to initialize a one-channel first convolution. The final classifier is replaced with a single linear output.

The original training freezes the early network through `layer2` and fine-tunes `layer3`, `layer4`, and the output layer. Training uses class-weighted binary cross entropy, AdamW with weight decay 0.01, a OneCycle learning-rate schedule with maximum learning rate 0.0006, batch size 64, and 40 epochs. Augmentation includes flips, modest spatial transforms, and brightness changes. Predictions combine the two supplied seed models and test-time augmentation.

The labeled dataset contains 1,124 crops, including 434 male and 690 female crops, from nine wells across three recording dates. Repeated images do not represent independent animals. The dataset includes only about 50 animals, which limits the evidence available for generalization.

The optional research notebooks preserve the dataset-building, training, and error-analysis workflow. Labels remain necessary for those research tasks; they are not input to routine prediction or export.

## Ranking and export

Each scored fly receives its own score. All scored flies in a well and frame are sorted from highest to lowest score, with ties ordered by original tracker ID. `rank_gap` records the difference from the next-ranked score and is empty for the last rank. No count is supplied to decide how many flies should be male.

Two additional columns describe the individual score:

| Column | Rule | Interpretation |
|---|---|---|
| `sex` | Male if `score > 0.5`, female otherwise | A model classification, not a confirmed label |
| `confident` | `score < 0.1` or `score > 0.9` | Score is outside the middle band |

Neither column changes the ranking or removes uncertain flies from it. Confidence here is a score-band flag, not a guarantee or an independently calibrated probability of correctness.

Export reorders complete tracker records at the exact rows for which all tracked flies have scores. At a ranked row, `fly1.csv` contains rank 1, `fly2.csv` rank 2, and so forth. Unscored rows retain their original order. The exporter does not carry a ranking into a later frame or interpolate between sampled frames.

The accompanying `rank_coverage.csv` distinguishes these cases. Analyses that interpret the exported tables as ranked data must use only rows where `ranked == True`.

Ranking gives relative order within a frame. An all-female well still has a rank 1, and two flies with nearly equal scores still have an ordering. Neither rank nor a high score establishes a persistent identity. In particular, joining successive rank-1 positions can join different animals.

## Held-out evaluation

Validation holds out one recording date at a time: train on two dates and evaluate on the third. Splitting frames randomly would mix near-duplicate images of the same animals between training and evaluation.

The reported classifications use the fixed 0.5 score threshold without stocking counts:

| Held-out recording date | Crops | Accuracy | AUC |
|---|---:|---:|---:|
| 2026-07-14 | 569 | 0.923 | 0.977 |
| 2026-07-17 | 410 | 0.785 | 0.938 |
| 2026-07-21 | 145 | 0.903 | 0.971 |
| Mean across dates | 1,124 total | **0.870** | **0.962** |

These are means across dates, not pooled crop accuracy. Across the full set, there were 146 classification errors: 95 males classified as female and 51 females classified as male.

Restricting evaluation to scores below 0.1 or above 0.9 retains approximately 71.7% of crops. Accuracy in that subset is 0.979, 0.891, and 0.983 for the three held-out dates, giving a date-wise mean of **0.951**. That figure applies to the retained subset and does not describe all crops or every rank-1 selection.

Historical agreement between selected high-scoring flies and annotations on July 14 snapshots was measured within the existing dataset. It is not an independent generalization test and does not validate continuous identities between those snapshots.

## Limits of the current evidence

Performance varies substantially between recording dates. The July 17 held-out recording accounts for much of the missed-male problem. A new recording can have a shifted score distribution because its imaging conditions or animals differ from the training data. The three-date results do not establish accuracy on unseen sessions or night footage.

FlyTracker IDs can switch between physical flies. Ranking each sampled frame independently avoids assuming that a tracker ID consistently identifies one animal, but it does not solve continuous identity tracking. This export therefore supports analysis of scored rows and rank-associated measurements; it does not establish complete male trajectories or stable identities among multiple males.

Image preprocessing was designed to reduce shortcuts from well illumination and scale. Any future change to crop generation requires renewed held-out evaluation and checks that the model distinguishes flies within wells, rather than merely distinguishing the wells themselves.

Additional validation should use annotations from independent recording dates and representative imaging conditions. Those annotations can assess the model's errors without imposing counts on its predictions.
