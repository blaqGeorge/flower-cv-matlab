# Evaluation suite

Metrics for the classification and segmentation models in `../src`, chosen so
that a trivial predictor cannot reach them.

| Task | Headline metric | Instead of | Why |
|---|---|---|---|
| Classification | **Macro-F1** | Accuracy | Accuracy is dominated by well-populated classes; macro-F1 weights every class equally, so a class the model never predicts drags the number down rather than hiding in it |
| Segmentation | **Mean IoU** | Pixel accuracy | Pixel accuracy is dominated by the background; a model predicting nothing but background can score 90–98% while segmenting nothing |

Both reports print the **trivial baseline** — what "predict the majority class"
scores on the same data — so the gap between a real result and an artefact of
class imbalance is visible on the page rather than left to the reader.

For the 17 Flowers dataset specifically, the classes are perfectly balanced at
80 images each, so accuracy and macro-F1 agree to within 0.003 and accuracy is
a defensible headline. The confidence interval is the part that matters there:
on 272 validation images, a point estimate carries roughly ±5.5 points.

---

## The pixel accuracy problem

`demo_reported_numbers.m` computes what a reported 98.5% pixel accuracy is
consistent with, at different foreground fractions:

```
    foreground    trivial pixel  mean IoU        headroom
    fraction           accuracy  (consistent)  over trivial
    1.5%                   98.5%         0.659         0.0 pp
    5.0%                   95.0%         0.862         3.5 pp
   10.0%                   90.0%         0.922         8.5 pp
   20.0%                   80.0%         0.955        18.5 pp
```

Top row: at a 1.5% foreground fraction, labelling every pixel "background"
scores 98.5% pixel accuracy having segmented nothing.

On the actual DaffodilSeg validation set the foreground is 23.4% of evaluated
pixels, so the trivial predictor reaches 0.766 pixel accuracy and only 0.383
mean IoU — against the measured 0.976. The original figure was not hollow, but
mean IoU is the far stronger claim.

---

## What each metric adds

**Classification** — `evalClassification.m`

| Metric | What it tells you that accuracy doesn't |
|---|---|
| Macro-F1 | Whether performance is even across all 17 classes |
| Balanced accuracy | Mean per-class recall — unaffected by class sizes |
| Accuracy ± 95% CI | Wilson score interval, which behaves near 0 and 1 where the normal approximation does not |
| Cohen's κ | Accuracy corrected for chance agreement (chance here is 5.9%) |
| Top-3 accuracy | Whether the right answer is nearly found or missed entirely |
| Expected calibration error | Whether the model's confidence means anything — deep networks are systematically overconfident |
| Per-class table | The weakest classes, and what each is most often confused with |

**Segmentation** — `evalSegmentation.m`, `segConfusion.m`

| Metric | What it tells you |
|---|---|
| Mean IoU | Per-class intersection-over-union, equally weighted — the standard in PASCAL VOC, Cityscapes and COCO |
| Per-class IoU | Which class is failing; the mean alone hides it |
| Dice / F1 | Same information weighted differently; the medical-imaging convention |
| Mean class accuracy | Per-class recall averaged — catches missed foreground |
| Frequency-weighted IoU | The compromise between mean IoU and pixel accuracy |
| Boundary F1 | Whether the *contours* are right. IoU is forgiving of ragged edges on large regions, so a blobby mask and a crisp one can score alike |

---

## Verification

`verify_metrics.m` — **38 checks, all passing.** Every expected value is derived
on paper, not taken from a previous run of this same code.

Evaluation code is the code you are least likely to notice is wrong: a subtly
broken metric returns a plausible number and you believe it. Worked cases
include a 2×2 confusion matrix with hand-computed precision, recall, F1 and κ;
a Wilson interval checked against its published reference value; the 90/10
all-background segmentation case (pixel accuracy 0.90, mean IoU 0.45); IoU
symmetry under exchanging false positives and false negatives; and the ignore
label being excluded from pixel totals.

---

## Usage

```matlab
verify_metrics          % 38 checks on the metric code itself
recon_dataset           % dataset inventory and label-value audit
evaluate_coursework     % both trained models, full reports
demo_reported_numbers   % what a reported figure does and does not constrain
```

`evaluate_coursework` reads paths from `paths_local.m` — copy
`paths_local.example.m` and edit it. That file is gitignored so
machine-specific paths stay out of the repository.

Against your own predictions:

```matlab
% Classification -- yTrue/yPred categorical or numeric, scores optional
R = evalClassification(yTrue, yPred, 'Scores', softmaxScores, 'ClassNames', names);

% Segmentation -- label images with values in 1..K, 0 = ignore
C = segConfusion(predLabels, gtLabels, K);
R = evalSegmentation(C, 'ClassNames', names);
```

## Files

| File | Purpose |
|---|---|
| `evalClassification.m` | Full classification report |
| `evalSegmentation.m` | Full segmentation report from a pixel confusion matrix |
| `segConfusion.m` | Label images → pixel confusion matrix, with optional boundary F1 |
| `wilsonInterval.m` | Confidence interval for a proportion |
| `expectedCalibrationError.m` | ECE and reliability-diagram data |
| `verify_metrics.m` | 38 checks against hand-computed values |
| `recon_dataset.m` | Dataset inventory and label-value audit |
| `evaluate_coursework.m` | Runs both trained models through the suite |
| `demo_reported_numbers.m` | Applies the above to the originally reported figures |
| `retrain_report.txt` | Full 75-epoch training log from the corrected run |

The metric code itself has no dependency on the Statistics or Deep Learning
toolboxes; boundary F1 needs Image Processing Toolbox and is optional.
