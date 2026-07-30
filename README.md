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
| Classification | 4-block CNN | Oxford 17 Flowers (17 classes, 1,360 images) | Validation accuracy | **64.3%** |
| Segmentation | U-Net | DaffodilSeg | Validation pixel accuracy | **98.5%** |

### Reading those numbers honestly

**The 98.5% is pixel accuracy, which flatters this problem.** Flower-versus-background is a
heavily imbalanced two-class task — background dominates the frame, so a model that guessed
"background" everywhere would still score highly. Mean IoU or Dice is the appropriate metric
and is the correct next measurement to add.

**64.3% on 17 classes is modest, and the cause is data.** 1,360 images across 17 categories is
roughly 80 examples per class, which is thin for training a CNN from scratch. Augmentation
(rotation, reflection, scaling) was added to compensate. Transfer learning from a pretrained
backbone such as ResNet-50 would very likely reach 90%+ — the from-scratch constraint here was
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
classification    % trains and saves classnet.mat
segmentation      % trains and saves segmentnet.mat
```

Trained `.mat` files are gitignored — each is over 100 MB, past GitHub's per-file limit.

## Next steps

- [ ] Report mean IoU and Dice for segmentation instead of pixel accuracy
- [ ] Extend segmentation to all five annotated classes
- [ ] Compare the from-scratch CNN against a fine-tuned ResNet-50 baseline
- [ ] Add a confusion matrix to identify which species are being confused
- [ ] Hold out a genuine test set, separate from the validation split used for tuning

## Licence

MIT — see `LICENSE`.
