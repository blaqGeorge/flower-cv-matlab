function R = evalClassification(yTrue, yPred, opts)
%EVALCLASSIFICATION  Evaluation report for a multi-class classifier.
%
%   R = evalClassification(yTrue, yPred) returns a metrics struct and prints
%   a report. Labels may be categorical, string, or numeric class indices.
%
%   R = evalClassification(..., 'Scores', S) additionally computes top-k
%   accuracy and calibration error from the N-by-C score matrix S.
%
%   Headline metric: MACRO-F1, not accuracy.
%
%   Accuracy answers "what fraction did it get right", which is only a fair
%   summary when the classes are balanced and equally important. It is
%   dominated by the classes with the most examples, so a model can look
%   respectable while failing entirely on several classes. Macro-F1 averages
%   per-class F1 with equal weight per class, so a class the model never
%   predicts drags the number down instead of hiding in it.
%
%   Both are reported. The gap between them is the diagnostic: a large gap
%   means performance is uneven across classes.

arguments
    yTrue
    yPred
    opts.Scores     double  = []
    opts.ClassNames string  = string.empty
    opts.TopK       double  = 3
    opts.Alpha      double  = 0.05
    opts.Print      logical = true
end

[t, p, names] = toIndices(yTrue, yPred, opts.ClassNames);
K = numel(names);
N = numel(t);

% ------------------------------------------------------ confusion --------
C = accumarray([t(:), p(:)], 1, [K K]);     % rows = truth, cols = prediction
R.confusion  = C;
R.classNames = names;
R.N          = N;
R.K          = K;

TP      = diag(C);
support = sum(C, 2);                        % true instances per class
predCnt = sum(C, 1).';                      % predicted instances per class
FP      = predCnt - TP;
FN      = support - TP;

% ------------------------------------------------------ headline ---------
R.accuracy = sum(TP) / N;
[R.accuracyCI(1), R.accuracyCI(2)] = wilsonInterval(sum(TP), N, opts.Alpha);

precision = safeDiv(TP, TP + FP);
recall    = safeDiv(TP, TP + FN);
f1        = safeDiv(2*precision.*recall, precision + recall);

present = support > 0;                      % classes actually in the truth
R.macroF1        = mean(f1(present));
R.macroPrecision = mean(precision(present));
R.macroRecall    = mean(recall(present));
R.balancedAcc    = mean(recall(present));   % = macro recall, by definition
R.weightedF1     = sum(support(present).*f1(present)) / sum(support(present));

% Cohen's kappa: accuracy corrected for agreement expected by chance.
pe       = sum(support .* predCnt) / N^2;
R.kappa  = (R.accuracy - pe) / (1 - pe);

R.perClass = table(names(:), support, precision, recall, f1, ...
    'VariableNames', {'Class', 'Support', 'Precision', 'Recall', 'F1'});

% Which class does each class get mistaken for most often?
Coff = C; Coff(1:K+1:end) = -1;
[~, worstIdx] = max(Coff, [], 2);
confusedCount = Coff(sub2ind([K K], (1:K).', worstIdx));
R.perClass.ConfusedWith = reshape(names(worstIdx), [], 1);
R.perClass.ConfusedN    = reshape(max(confusedCount, 0), [], 1);

% ------------------------------------------------------ baselines --------
% What a model with no information scores. Any headline number should be
% quoted against these, not against zero.
[~, majority] = max(support);
Cmaj = zeros(K); Cmaj(:, majority) = support;
R.baselines.majorityAccuracy = support(majority) / N;
R.baselines.majorityMacroF1  = macroF1From(Cmaj);
R.baselines.randomAccuracy   = 1 / K;
R.baselines.randomMacroF1    = macroF1From(support * (support.' / N));

% ------------------------------------------------------ score-based ------
R.topK = [];
R.ece  = [];
if ~isempty(opts.Scores)
    S = opts.Scores;
    assert(size(S,1) == N && size(S,2) == K, ...
        'Scores must be %d-by-%d.', N, K);

    ks = 1:min(opts.TopK, K);
    R.topK = zeros(1, numel(ks));
    [~, order] = sort(S, 2, 'descend');
    for i = ks
        R.topK(i) = mean(any(order(:, 1:i) == t(:), 2));
    end

    conf    = max(S, [], 2);
    correct = (p(:) == t(:));
    [R.ece, R.reliability] = expectedCalibrationError(conf, correct);
end

if opts.Print, printReport(R, opts); end
end

% =========================================================================
function printReport(R, opts)
fprintf('\n=== Classification: %d classes, %d samples ==================\n', R.K, R.N);
fprintf('  MACRO-F1 (headline)      %.4f\n', R.macroF1);
fprintf('  Accuracy                 %.4f   (95%% CI %.4f - %.4f)\n', ...
        R.accuracy, R.accuracyCI(1), R.accuracyCI(2));
fprintf('  Balanced accuracy        %.4f\n', R.balancedAcc);
fprintf('  Weighted F1              %.4f\n', R.weightedF1);
fprintf('  Cohen''s kappa            %.4f\n', R.kappa);
if ~isempty(R.topK)
    for i = 1:numel(R.topK)
        fprintf('  Top-%d accuracy            %.4f\n', i, R.topK(i));
    end
    fprintf('  Expected calibration err %.4f\n', R.ece);
end

fprintf('\n  Trivial baselines (what "no information" scores)\n');
fprintf('    predict majority class   accuracy %.4f   macro-F1 %.4f\n', ...
        R.baselines.majorityAccuracy, R.baselines.majorityMacroF1);
fprintf('    predict at random        accuracy %.4f   macro-F1 %.4f\n', ...
        R.baselines.randomAccuracy, R.baselines.randomMacroF1);

gap = R.accuracy - R.macroF1;
fprintf('\n  Accuracy - macro-F1 gap  %+.4f', gap);
if abs(gap) > 0.05
    fprintf('   <-- uneven across classes');
end
fprintf('\n');

T = sortrows(R.perClass, 'F1', 'ascend');
n = min(5, height(T));
fprintf('\n  Weakest %d classes\n', n);
fprintf('    %-14s %7s %9s %7s %6s  %s\n', ...
        'class', 'support', 'precision', 'recall', 'F1', 'most confused with');
for i = 1:n
    fprintf('    %-14s %7d %9.3f %7.3f %6.3f  %s (%d)\n', ...
        T.Class(i), T.Support(i), T.Precision(i), T.Recall(i), T.F1(i), ...
        T.ConfusedWith(i), T.ConfusedN(i));
end
fprintf('\n');
end

% =========================================================================
function [t, p, names] = toIndices(yTrue, yPred, names)
if iscategorical(yTrue) || isstring(yTrue) || iscellstr(yTrue)
    yTrue = categorical(yTrue);
    yPred = categorical(yPred, categories(yTrue));
    if isempty(names), names = string(categories(yTrue)); end
    t = double(yTrue);  p = double(yPred);
else
    t = double(yTrue(:));  p = double(yPred(:));
    if isempty(names)
        names = "class" + string(1:max([t; p]));
    end
end
assert(numel(t) == numel(p), 'yTrue and yPred must be the same length.');
assert(~any(isnan(p)), 'yPred contains labels not present in yTrue.');
end

function y = safeDiv(a, b)
y = a ./ b;
y(b == 0) = 0;          % a class never predicted has precision 0, not NaN
end

function f = macroF1From(C)
TP = diag(C); sup = sum(C,2); pc = sum(C,1).';
pr = safeDiv(TP, pc); re = safeDiv(TP, sup);
f1 = safeDiv(2*pr.*re, pr + re);
f  = mean(f1(sup > 0));
end
