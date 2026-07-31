# Flower Classification & Semantic Segmentation in MATLAB

Two deep-learning models built from scratch in MATLAB for the Computer Vision module of my
MSc Computer Science with AI at the University of Nottingham:

1. **Classification** — a convolutional network that identifies 17 flower species
2. **Segmentation** — a U-Net that labels flower pixels against background

Both networks are defined layer by layer rather than fine-tuned from a pretrained model, so
the repository is a reasonable reference for how to assemble, train and evaluate custom
architectures with the MATLAB Deep Learning Toolbox.

## Results

| Task | Architecture | Dataset | Metric | Result |
|---|---|---|---|---|
| Classification | 4-block CNN | Oxford 17 Flowers (17 classes, 1,360 images) | Validation accuracy | **73.2%** (95% CI 67.6–78.1, n = 272) |
| | | | Macro-F1 | **0.735** |
| | | | Top-3 accuracy | 0.879 |
| Segmentation | U-Net | DaffodilSeg | **Mean IoU** | **0.976** (flower 0.963, background 0.989) |
| | | | Boundary F1 | 0.930 |
| | | | Pixel accuracy | 0.991 |

Measured with the suite in `eval/`, which is itself verified by 38 checks against
hand-computed values. Full method and per-class breakdowns in **[RESULTS.md](RESULTS.md)**.

### Reading those numbers honestly

**Mean IoU, not pixel accuracy.** Flower-versus-background is imbalanced, so pixel accuracy
flatters it: labelling every pixel "background" scores 0.766 here. Mean IoU gives each class
equal weight, and the same trivial predictor reaches only 0.383. Reporting 0.976 against that
0.383 ceiling is a claim the metric can actually support; 98.5% pixel accuracy was not.

**The classification figure moved because a defect was found and fixed.** The original run
reported 64.3%. Re-evaluating it revealed a 34-point gap between training and validation
accuracy, traced to an `imageDataAugmenter` that `classification.m` constructs and then never
attaches to the training datastore — so the augmentation described in the write-up never ran.
Retraining with it applied (`src/classification_augmented.m`, seeded split) gives 73.2%, and
halves the generalisation gap to 15.4 points. The improvement is significant at 95%
(z = 2.24, p = 0.025).

**What augmentation did not fix.** The surviving errors are colour confusions — Snowdrop with
Lily Valley (both white), Daffodil with Cowslip (both yellow), Tulip with Cowslip. Rotation,
reflection and scaling leave colour untouched, so this is the expected residue. Colour jitter,
or transfer learning from a backbone with richer texture features, is the indicated next step;
a fine-tuned ResNet-50 would very likely reach 90%+. The from-scratch constraint here was
pedagogical, to build the architecture rather than adapt one.

## Classification model

`src/classification.m`

- Input 256×256×3
- Four convolutional blocks — 5×5/32, then 3×3 at 64, 128, 256 — each with batch normalisation, ReLU and 2×2 max pooling at stride 2
- Fully connected 512 → ReLU → dropout 0.5
- Fully connected 17 → softmax
- Adam, initial learning rate 3×10⁻⁴, 75 epochs, GPU execution
- Labels derived from folder names via `imageDatastore`; 80/20 stratified split with `splitEachLabel`
- Augmentation: ±30° rotation, random X reflection, 0.8–1.2 X/Y scaling

`src/classification_augmented.m` is the corrected run. `classification.m` is kept unchanged as
the original: it builds the augmenter on line 25 and then calls
`trainNetwork(trainSet, ...)` with the plain datastore, so the augmentation is never applied.
The corrected version differs in exactly two ways — `rng(seed)` before the split so the
validation set is recorded and the result is reproducible, and the augmenter actually attached
via `augmentedImageDatastore`. Everything else is identical, so the comparison is controlled.

## Segmentation model

`src/segmentation.m`

- `unet(imageSize, 2)` converted to a `layerGraph`, with a `pixelClassificationLayer` added and connected to the final softmax
- `pixelLabelDatastore` pairing images to masks, combined via `pixelLabelImageDatastore`
- Images normalised with `im2double` and resized to 256×256; masks resized without normalisation
- Adam, learning rate 3×10⁻⁴, 50 epochs, mini-batch 16, GPU execution

**Note on classes.** The DaffodilSeg masks encode five regions (sky, grass, background, null,
daffodil). This implementation collapses the problem to two classes by selecting label IDs
`[1, 3]`, so the remaining regions are not learned. Extending to the full five-class problem is
listed below.

Quantified: the discarded regions are **17.3% of all labelled pixels** (values 0, 2 and 4 at
13.26%, 3.19% and 0.85%), and 11.3% of the validation set specifically. Undeclared values
become `<undefined>`, which MATLAB excludes from both the training loss and the accuracy
calculation — so every number above is computed over 83% of the annotation, not all of it.
`eval/recon_dataset.m` audits this.

## Requirements

- MATLAB R2023a or later
- Deep Learning Toolbox
- Image Processing Toolbox
- Computer Vision Toolbox (`pixelLabelDatastore`, `semanticseg`)
- Parallel Computing Toolbox — for `'ExecutionEnvironment', 'gpu'`; change to `'cpu'` if unavailable

## Getting the data

Datasets are not committed. Place them under `data/` so the relative paths resolve:

```
data/
  17flowers/            # one subfolder per species — folder names become the labels
  daffodilSeg/
    ImagesRsz256/
    LabelsRsz256/
```

The Oxford 17 Flowers dataset is published by the Visual Geometry Group, University of Oxford:
https://www.robots.ox.ac.uk/~vgg/data/flowers/17/

## Running

```matlab
cd src
classification              % original run — trains and saves classnet.mat
classification_augmented    % corrected run — seeded split, augmentation applied
segmentation                % trains and saves segmentnet.mat
```

Trained `.mat` files are gitignored — each is over 100 MB, past GitHub's per-file limit.

### Evaluation

```matlab
cd eval
verify_metrics          % 38 checks on the metric code itself
recon_dataset           % dataset inventory and label-value audit
evaluate_coursework     % both trained models, full reports
```

`evaluate_coursework` reads dataset and model locations from `paths_local.m` — copy
`paths_local.example.m` and edit it. That file is gitignored so machine-specific paths stay
out of the repository.

## Next steps

- [x] Report mean IoU and Dice for segmentation instead of pixel accuracy
- [x] Add a confusion matrix to identify which species are being confused
- [x] Seed the train/validation split so results are reproducible
- [ ] Extend segmentation to all five annotated classes
- [ ] Compare the from-scratch CNN against a fine-tuned ResNet-50 baseline
- [ ] Add colour jitter to attack the confusions that survived augmentation
- [ ] Hold out a genuine test set, separate from the validation split used for tuning
- [ ] Calibrate the classifier — expected calibration error is currently 0.107

## Licence

MIT — see `LICENSE`.
