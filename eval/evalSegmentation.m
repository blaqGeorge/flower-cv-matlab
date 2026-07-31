function R = evalSegmentation(C, opts)
%EVALSEGMENTATION  Evaluation report for semantic segmentation.
%
%   R = evalSegmentation(C) takes a K-by-K pixel confusion matrix (rows =
%   ground truth, columns = prediction; build one with segConfusion) and
%   returns a metrics struct.
%
%   Headline metric: MEAN IoU, not pixel accuracy.
%
%   Pixel accuracy is the fraction of pixels labelled correctly. It is
%   dominated by whichever class covers the most pixels, and in a typical
%   figure-ground segmentation that is the background. If background is 90%
%   of the pixels, a model that predicts nothing but background scores 90%
%   pixel accuracy while segmenting nothing at all.
%
%   Intersection-over-Union (the Jaccard index) is computed per class and
%   then averaged with equal weight per class:
%
%       IoU_c = TP_c / (TP_c + FP_c + FN_c)
%
%   It penalises both missed foreground (FN) and spilled foreground (FP),
%   and gives the background no more weight than anything else. This is the
%   metric PASCAL VOC, Cityscapes and COCO report, and it is what a computer
%   vision reviewer will ask for.
%
%   The report prints the trivial "predict the majority class everywhere"
%   baseline next to your result on both metrics, so the difference between
%   a real result and an artefact of class imbalance is visible on the page.

arguments
    C               double {mustBeNonnegative}
    opts.ClassNames string  = string.empty
    opts.Print      logical = true
end

K = size(C, 1);
assert(size(C,2) == K, 'Confusion matrix must be square.');

names = opts.ClassNames;
if isempty(names), names = "class" + string(1:K); end

total   = sum(C(:));
TP      = diag(C);
support = sum(C, 2);                 % true pixels per class
predCnt = sum(C, 1).';
FP      = predCnt - TP;
FN      = support - TP;

R.confusion  = C;
R.classNames = names;
R.nPixels    = total;

R.iou       = safeDiv(TP, TP + FP + FN);
R.dice      = safeDiv(2*TP, 2*TP + FP + FN);
R.recall    = safeDiv(TP, support);
R.precision = safeDiv(TP, predCnt);
R.frequency = support / total;

present = support > 0;
R.meanIoU          = mean(R.iou(present));         % headline
R.meanDice         = mean(R.dice(present));
R.pixelAccuracy    = sum(TP) / total;
R.meanAccuracy     = mean(R.recall(present));
R.freqWeightedIoU  = sum(R.frequency(present) .* R.iou(present));

R.perClass = table(names(:), support, R.frequency, R.iou, R.dice, ...
                   R.precision, R.recall, ...
    'VariableNames', {'Class','Pixels','Frequency','IoU','Dice','Precision','Recall'});

% ------------------------------- the trivial baseline --------------------
% "Label every pixel with the most common class." Computed in closed form:
% the majority class gets TP = its own support, FP = everything else, FN = 0,
% so its IoU is exactly its frequency; every other class scores 0.
[~, maj] = max(support);
R.baseline.class         = names(maj);
R.baseline.pixelAccuracy = support(maj) / total;
R.baseline.meanIoU       = (support(maj) / total) / nnz(present);

if opts.Print, printReport(R); end
end

% =========================================================================
function printReport(R)
K = numel(R.classNames);
fprintf('\n=== Segmentation: %d classes, %s pixels ====================\n', ...
        K, addComma(R.nPixels));
fprintf('  MEAN IoU (headline)      %.4f\n', R.meanIoU);
fprintf('  Mean Dice                %.4f\n', R.meanDice);
fprintf('  Mean class accuracy      %.4f\n', R.meanAccuracy);
fprintf('  Frequency-weighted IoU   %.4f\n', R.freqWeightedIoU);
fprintf('  Pixel accuracy           %.4f   <-- inflated by class imbalance\n', ...
        R.pixelAccuracy);

fprintf('\n  Trivial baseline: label every pixel "%s"\n', R.baseline.class);
fprintf('    pixel accuracy         %.4f   (you beat it by %+.4f)\n', ...
        R.baseline.pixelAccuracy, R.pixelAccuracy - R.baseline.pixelAccuracy);
fprintf('    mean IoU               %.4f   (you beat it by %+.4f)\n', ...
        R.baseline.meanIoU, R.meanIoU - R.baseline.meanIoU);

fprintf('\n  %-14s %10s %10s %8s %8s %10s %8s\n', ...
        'class', 'pixels', 'frequency', 'IoU', 'Dice', 'precision', 'recall');
T = sortrows(R.perClass, 'IoU', 'ascend');
for i = 1:height(T)
    fprintf('  %-14s %10s %10.4f %8.4f %8.4f %10.4f %8.4f\n', ...
        T.Class(i), addComma(T.Pixels(i)), T.Frequency(i), T.IoU(i), ...
        T.Dice(i), T.Precision(i), T.Recall(i));
end
fprintf('\n');
end

% =========================================================================
function y = safeDiv(a, b)
y = a ./ b;
y(b == 0) = 0;
end

function s = addComma(x)
s = regexprep(sprintf('%d', round(x)), '\d{1,3}(?=(\d{3})+$)', '$&,');
end
