function verify_metrics()
%VERIFY_METRICS  Check every metric against a case computed by hand.
%
%   Evaluation code is the code you are least likely to notice is wrong: a
%   subtly broken metric produces a plausible number and you believe it.
%   Every check below has an expected value derived on paper, not from a
%   previous run of this same code.

fprintf('\n=== Metric verification ===================================\n');
n = 0; f = 0;

% ---------------------------------------------- Wilson interval ----------
% Published reference value for 50/100 at 95%: (0.4038, 0.5962).
[lo, hi] = wilsonInterval(50, 100);
[n,f] = chk(n, f, 'Wilson CI, 50/100, lower', lo, 0.40383, 1e-4);
[n,f] = chk(n, f, 'Wilson CI, 50/100, upper', hi, 0.59617, 1e-4);

% Degenerate case: the normal approximation gives a zero-width interval at
% k = n; Wilson does not.
[lo, hi] = wilsonInterval(10, 10);
[n,f] = chk(n, f, 'Wilson CI, 10/10, upper is 1',  hi, 1.0,    1e-12);
[n,f] = chk(n, f, 'Wilson CI, 10/10, lower < 1',   lo < 1,  true, 0);

% ------------------------------------------------- classification --------
% Perfect prediction on 3 balanced classes.
y = [1 1 2 2 3 3].';
R = evalClassification(y, y, 'Print', false);
[n,f] = chk(n, f, 'Perfect: accuracy',  R.accuracy, 1, 1e-12);
[n,f] = chk(n, f, 'Perfect: macro-F1',  R.macroF1,  1, 1e-12);
[n,f] = chk(n, f, 'Perfect: kappa',     R.kappa,    1, 1e-12);

% Hand-worked 2-class case.
%   yTrue = 1 1 1 1 2 2 2 2 ; yPred = 1 1 1 2 2 2 1 1
%   C = [3 1 ; 2 2]
%   class 1: TP 3, FP 2, FN 1 -> P 0.6,     R 0.75, F1 0.666667
%   class 2: TP 2, FP 1, FN 2 -> P 0.66667, R 0.50, F1 0.571429
%   accuracy 5/8 = 0.625, macro-F1 0.619048
%   pe = (4*5 + 4*3)/64 = 0.5 -> kappa = (0.625-0.5)/0.5 = 0.25
yT = [1 1 1 1 2 2 2 2].';
yP = [1 1 1 2 2 2 1 1].';
R  = evalClassification(yT, yP, 'Print', false);
[n,f] = chk(n, f, 'Hand case: accuracy',       R.accuracy,          0.625,    1e-12);
[n,f] = chk(n, f, 'Hand case: macro-F1',       R.macroF1,           0.619048, 1e-6);
[n,f] = chk(n, f, 'Hand case: kappa',          R.kappa,             0.25,     1e-12);
[n,f] = chk(n, f, 'Hand case: class 1 recall', R.perClass.Recall(1), 0.75,    1e-12);
[n,f] = chk(n, f, 'Hand case: class 2 prec.',  R.perClass.Precision(2), 2/3,  1e-12);

% Majority baseline on support [6 3 1]:
%   accuracy 0.6; macro-F1 = (0.75 + 0 + 0)/3 = 0.25
yT = [ones(1,6) 2*ones(1,3) 3].';
R  = evalClassification(yT, yT, 'Print', false);
[n,f] = chk(n, f, 'Baseline: majority accuracy', R.baselines.majorityAccuracy, 0.6,  1e-12);
[n,f] = chk(n, f, 'Baseline: majority macro-F1', R.baselines.majorityMacroF1,  0.25, 1e-12);

% A class that is never predicted must score 0, not NaN.
yT = [1 1 2 2].'; yP = [1 1 1 1].';
R  = evalClassification(yT, yP, 'Print', false);
[n,f] = chk(n, f, 'Unpredicted class F1 is 0', R.perClass.F1(2), 0, 1e-12);
[n,f] = chk(n, f, 'Unpredicted class no NaN',  any(isnan(R.perClass.F1)), false, 0);

% Top-k: truth is class 3, scores rank it second -> top-1 fails, top-2 passes.
S = [0.5 0.1 0.4; 0.5 0.1 0.4];
R = evalClassification([3;3], [1;1], 'Scores', S, 'TopK', 3, 'Print', false);
[n,f] = chk(n, f, 'Top-1 accuracy', R.topK(1), 0, 1e-12);
[n,f] = chk(n, f, 'Top-2 accuracy', R.topK(2), 1, 1e-12);

% ------------------------------------------------------ calibration ------
% 100 predictions at confidence 0.9 of which half are correct -> ECE = 0.4.
conf    = 0.9*ones(100,1);
correct = [true(50,1); false(50,1)];
[ece, ~] = expectedCalibrationError(conf, correct);
[n,f] = chk(n, f, 'ECE, overconfident case', ece, 0.4, 1e-12);

% Confidence matches accuracy exactly in every bin -> ECE = 0.
conf    = [ones(80,1); 0.5*ones(20,1)];
correct = [true(80,1); true(10,1); false(10,1)];
[ece, ~] = expectedCalibrationError(conf, correct);
[n,f] = chk(n, f, 'ECE, calibrated case', ece, 0, 1e-12);

% ------------------------------------------------- segmentation ----------
% Perfect segmentation.
C = [50 0; 0 50];
R = evalSegmentation(C, 'Print', false);
[n,f] = chk(n, f, 'Seg perfect: mean IoU',       R.meanIoU,       1, 1e-12);
[n,f] = chk(n, f, 'Seg perfect: pixel accuracy', R.pixelAccuracy, 1, 1e-12);

% THE TRAP. 90% background, model predicts background everywhere.
%   C = [90 0 ; 10 0]
%   IoU_bg = 90/(90+10+0) = 0.90 ; IoU_fg = 0/(0+0+10) = 0 -> mIoU = 0.45
%   pixel accuracy = 90/100 = 0.90
%   mean class accuracy = (1 + 0)/2 = 0.5
%   frequency-weighted IoU = 0.9*0.9 + 0.1*0 = 0.81
C = [90 0; 10 0];
R = evalSegmentation(C, 'Print', false);
[n,f] = chk(n, f, 'Trap: pixel accuracy',   R.pixelAccuracy,   0.90, 1e-12);
[n,f] = chk(n, f, 'Trap: mean IoU',         R.meanIoU,         0.45, 1e-12);
[n,f] = chk(n, f, 'Trap: mean class acc',   R.meanAccuracy,    0.50, 1e-12);
[n,f] = chk(n, f, 'Trap: freq-weighted IoU',R.freqWeightedIoU, 0.81, 1e-12);
[n,f] = chk(n, f, 'Trap: equals its own baseline', ...
                  R.meanIoU, R.baseline.meanIoU, 1e-12);

% Hand-worked partial overlap. C = [8 2 ; 3 7]
%   IoU_1 = 8/13 = 0.615385 ; IoU_2 = 7/12 = 0.583333 -> mIoU = 0.599359
%   Dice_1 = 16/21 = 0.761905 ; pixel accuracy = 15/20 = 0.75
C = [8 2; 3 7];
R = evalSegmentation(C, 'Print', false);
[n,f] = chk(n, f, 'Seg hand: IoU class 1',    R.iou(1),        8/13,     1e-12);
[n,f] = chk(n, f, 'Seg hand: mean IoU',       R.meanIoU,       0.599359, 1e-6);
[n,f] = chk(n, f, 'Seg hand: Dice class 1',   R.dice(1),       16/21,    1e-12);
[n,f] = chk(n, f, 'Seg hand: pixel accuracy', R.pixelAccuracy, 0.75,     1e-12);

% IoU must be symmetric in the two ways a pixel can be wrong.
[n,f] = chk(n, f, 'IoU symmetric in FP/FN', ...
    isequal(evalSegmentation([8 2;3 7],'Print',false).meanIoU, ...
            evalSegmentation([8 3;2 7],'Print',false).meanIoU), true, 0);

% ------------------------------------------------ confusion builder ------
% 4x4 image, gt is two vertical halves, prediction shifted one column right.
gt   = repmat([1 1 2 2], 4, 1);
pred = repmat([1 1 1 2], 4, 1);
C = segConfusion(pred, gt, 2);
[n,f] = chk(n, f, 'segConfusion: C(1,1)', C(1,1), 8, 1e-12);
[n,f] = chk(n, f, 'segConfusion: C(2,1)', C(2,1), 4, 1e-12);
[n,f] = chk(n, f, 'segConfusion: C(2,2)', C(2,2), 4, 1e-12);
[n,f] = chk(n, f, 'segConfusion: total',  sum(C(:)), 16, 1e-12);

% Ignore label 0 must be excluded from the totals.
gt(:,1) = 0;
C = segConfusion(pred, gt, 2);
[n,f] = chk(n, f, 'segConfusion: ignores label 0', sum(C(:)), 12, 1e-12);

% Batches of images accumulate.
C = segConfusion({pred, pred}, {repmat([1 1 2 2],4,1), repmat([1 1 2 2],4,1)}, 2);
[n,f] = chk(n, f, 'segConfusion: batch accumulates', sum(C(:)), 32, 1e-12);

fprintf('  ---------------------------------------------------------\n');
fprintf('  %d/%d checks passed\n\n', n-f, n);
if f > 0, error('verify:metrics', '%d metric check(s) failed.', f); end
end

% =========================================================================
function [n, f] = chk(n, f, name, got, want, tol)
n = n + 1;
if islogical(want) || tol == 0
    ok = isequal(logical(got), logical(want));
    gotStr = string(logical(got)); wantStr = string(logical(want));
else
    ok = abs(got - want) <= tol;
    gotStr = sprintf('%.6f', got); wantStr = sprintf('%.6f', want);
end
if ~ok, f = f + 1; verdict = 'FAIL'; else, verdict = 'PASS'; end
fprintf('  %-36s %10s / %-10s %s\n', name, gotStr, wantStr, verdict);
end
