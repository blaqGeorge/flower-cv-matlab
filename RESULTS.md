# Results — MSc computer vision coursework

Going back to a finished project with better metrics than it was originally
reported with, to see what the original numbers did and did not establish.

Evaluated from the trained networks against the `17flowers` (1,360 images,
17 classes) and `daffodilSeg` (71 images, figure-ground) datasets. The metric
code itself is verified by 38 checks against hand-computed values
(`verify_metrics.m`), because a subtly broken metric returns a plausible number
and you believe it.

Headline: the segmentation result is considerably stronger than the metric
originally used could show, the classification figure is corroborated but not
reproducible, and three defects turned up in the training scripts.

---

## Segmentation — the headline result

| Metric | Value |
|---|---|
| **Mean IoU** | **0.9760** |
| Flower IoU | 0.9634 |
| Background IoU | 0.9886 |
| Mean Dice | 0.9878 |
| Mean class accuracy | 0.9899 |
| Boundary F1 (contour agreement) | 0.9297 |
| Pixel accuracy | 0.9912 |

Validation set: images 58–71 of 71, exactly as the training script splits them.
That split is deterministic, so this is reproducible.

**Against the trivial baseline** — label every pixel "background":

| | Trivial | Achieved | Margin |
|---|---|---|---|
| Pixel accuracy | 0.7657 | 0.9912 | +0.226 |
| Mean IoU | 0.3828 | 0.9760 | **+0.593** |

Flower pixels are 23.4% of the evaluated set, not the 1–2% at which pixel
accuracy becomes meaningless. So the originally reported 98.5% was not a hollow
number — but mean IoU is the stronger claim by a wide margin, because the
trivial predictor can reach 0.766 pixel accuracy and only 0.383 mean IoU.

Boundary F1 at 0.930 against IoU at 0.976 is the expected relationship: region
overlap is near-perfect, edges are slightly softer. Worth reporting both, since
IoU alone is forgiving of ragged contours on large regions.

---

## Classification

**The reported 64.3% is corroborated, but cannot be reproduced.**

`classification.m` line 23 calls `splitEachLabel(imds, 0.8, 'randomized')` with
no RNG seed set, so the validation set was never recorded. Attempting to recover
it by replaying the default RNG state fails — every candidate split scores ~91.5%,
i.e. indistinguishable from the full-set figure, which means each one leaks
training images.

The reported number still checks out arithmetically. The split is exactly 80/20
per class, so `fullAcc = 0.8·trainAcc + 0.2·valAcc`:

| | |
|---|---|
| Measured accuracy over all 1360 images | 91.7% |
| Reported validation accuracy | 64.3% |
| ⇒ implied accuracy on the 1088 training images | **98.5%** |
| ⇒ generalisation gap | **34.2 points** |

98.5% train against 64.3% validation is a heavily overfit model — consistent
with 75 epochs on 1088 un-augmented images (see finding 1).

**Per-class structure.** Absolute values below are inflated, because 80% of these
images were in training. The *relative* pattern is still diagnostic:

| Class | F1 | Precision | Recall | Most confused with |
|---|---|---|---|---|
| Tulip | 0.828 | 0.844 | 0.812 | Buttercup (3) |
| Daffodil | 0.861 | 0.969 | 0.775 | Buttercup (8) |
| Cowslip | 0.867 | 0.929 | 0.812 | Colts'Foot (4) |
| Snowdrop | 0.882 | 0.877 | 0.887 | Lily Valley (4) |
| Dandelion | 0.894 | 0.889 | 0.900 | Colts'Foot (3) |

The confusions are between the yellow flowers (Daffodil/Buttercup/Colts'Foot/
Dandelion/Cowslip) and between the white ones (Snowdrop/Lily Valley) — i.e. the
model is largely separating on colour, and struggles where colour does not
separate. That is a real, explainable observation about what the network learned.

**Note on metric choice.** The 17 classes are perfectly balanced at 80 images
each, so accuracy and macro-F1 agree to within 0.0007. On *this* dataset accuracy
is a defensible headline. The confidence interval is the part that matters: on
272 validation images, 64.3% carries a 95% interval of roughly 58.5–69.8%.

Chance level is 5.9%; Cohen's κ and top-3 accuracy are in the full report.

---

## Three findings in the source

### 1. The augmentation was never applied

`classification.m` line 25 constructs an `imageDataAugmenter` with rotation,
reflection and scaling. Line 75 then calls:

```matlab
flowerNet = trainNetwork(trainSet, layers, options);
```

`trainSet` is a plain `imageDatastore` straight from `splitEachLabel`. The
augmenter is never referenced again, and there is no `augmentedImageDatastore`
anywhere in the project (verified across all three copies of the script). The
augmentation object was created and dropped.

This explains the 34-point generalisation gap, and it means the CV claim
"with augmentation to offset a 1,360-image dataset" describes code that did not
run. One line fixes it:

```matlab
augTrain = augmentedImageDatastore([256 256 3], trainSet, 'DataAugmentation', augmenter);
flowerNet = trainNetwork(augTrain, layers, options);
```

### 1b. Retrained with the fix — the finding is confirmed

`src/classification_augmented.m` re-runs the training with exactly two changes: the split
is seeded (`rng(42)`), and the augmenter is attached to the training datastore.
Architecture, optimiser, learning rate, epoch count and split ratio are identical
to the original script.

| | original | with augmentation |
|---|---|---|
| Training accuracy | 98.5% *(inferred)* | **88.6%** *(measured)* |
| Validation accuracy | 64.3% | **73.2%** (95% CI 67.6–78.1) |
| Generalisation gap | 34.2 points | **15.4 points** |
| Macro-F1 | not measured | 0.735 |
| Top-3 accuracy | not measured | 0.879 |
| Cohen's κ | not measured | 0.715 |

**+8.9 points of validation accuracy, and the generalisation gap more than
halved.** Both figures are on 272 held-out images; treating them as independent
samples, the difference is 8.9 pp against a standard error of 3.96 pp — z = 2.24,
two-tailed p = 0.025. Significant at 95%, though not overwhelmingly so, and the
two runs used different random splits, so this is not a paired comparison.

Training accuracy falling from 98.5% to 88.6% is the mechanism made visible: the
network is no longer able to memorise the training set, so capacity goes into
features that generalise instead.

Remaining weaknesses on the held-out set, which the original evaluation could not
have shown:

| Class | F1 | Most confused with |
|---|---|---|
| Tulip | 0.353 | Cowslip |
| Snowdrop | 0.606 | Lily Valley |
| Cowslip | 0.647 | Tulip |
| Lily Valley | 0.647 | Snowdrop |
| Daffodil | 0.649 | Cowslip |

Snowdrop ↔ Lily Valley are both white; Daffodil and Cowslip are both yellow. The
colour-confusion pattern seen in the earlier full-set evaluation survives the
augmentation, which makes sense — rotation, reflection and scaling do not perturb
colour. A colour-jitter augmentation, or transfer learning from a network with
richer texture features, is the indicated next step.

Expected calibration error is 0.107, i.e. the model is meaningfully
overconfident. That is normal for a network of this kind and worth stating
rather than omitting.

### 2. The validation split was not seeded

`splitEachLabel(..., 'randomized')` without `rng(seed)` means the result cannot
be reproduced, by you or by anyone else. Adding `rng(42)` above line 23 makes
every number in the report regenerable.

### 3. 17.3% of labelled pixels are silently discarded

`segmentation.m` line 15 declares `labelIDs = [1, 3]`. The label maps actually
contain five values:

| Pixel value | Share of all label pixels | Declared? |
|---|---|---|
| 0 | 13.26% | no → `<undefined>` |
| 1 (flower) | 25.67% | yes |
| 2 | 3.19% | no → `<undefined>` |
| 3 (background) | 57.03% | yes |
| 4 | 0.85% | no → `<undefined>` |

Values 0, 2 and 4 become `<undefined>`, which MATLAB excludes from both the
training loss and the accuracy calculation. So 17.3% of the annotation was
dropped — 11.3% of the validation pixels specifically. The model was never
trained on those regions and is not scored on them.

Whether that is wrong depends on what those values mean in the source dataset
(plausibly leaf, stem, or a void/border band). If they are real classes, this is
a 4- or 5-class problem solved as a 2-class one; if value 0 is a void band, that
part is standard practice. Either way it should be stated rather than implicit,
because "98.5% pixel accuracy" computed over 83% of the pixels is not the same
claim as over all of them.

---

## Reproducing

```matlab
verify_metrics          % 38 checks on the metric code itself
recon_dataset           % dataset inventory and label-value audit
evaluate_coursework     % both models, full reports
```
