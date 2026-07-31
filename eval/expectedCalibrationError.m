function [ece, rel] = expectedCalibrationError(conf, correct, nBins)
%EXPECTEDCALIBRATIONERROR  How much a model's confidence is to be believed.
%
%   Accuracy says how often the model is right. Calibration says whether its
%   confidence means anything: of the predictions made at 90% confidence,
%   were 90% of them actually correct?
%
%   ECE bins predictions by confidence and averages |accuracy - confidence|
%   within each bin, weighted by bin population. 0 is perfect. Modern deep
%   networks are systematically overconfident, so this is usually non-zero
%   and worth reporting -- especially for a model at 64% accuracy that may
%   be claiming 95% confidence.
%
%   rel is a table of per-bin counts, mean confidence and accuracy, which is
%   what you plot as a reliability diagram.

arguments
    conf    (:,1) double
    correct (:,1) logical
    nBins   (1,1) double = 15
end

edges = linspace(0, 1, nBins+1);
bin   = discretize(conf, edges);
bin(conf == 1) = nBins;                 % include the closed upper endpoint

n = numel(conf);
count = zeros(nBins,1); mconf = zeros(nBins,1); macc = zeros(nBins,1);

for b = 1:nBins
    m = (bin == b);
    count(b) = sum(m);
    if count(b) > 0
        mconf(b) = mean(conf(m));
        macc(b)  = mean(correct(m));
    end
end

ece = sum(count .* abs(macc - mconf)) / n;

rel = table((1:nBins).', edges(1:end-1).', edges(2:end).', ...
            count, mconf, macc, ...
    'VariableNames', {'Bin','LowEdge','HighEdge','Count','MeanConf','Accuracy'});
end
