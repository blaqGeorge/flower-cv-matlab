function classification_augmented(seed)
%CLASSIFICATION_AUGMENTED  The 17-class CNN with the augmentation applied.
%
%   Controlled comparison against the original run in classification.m.
%   Architecture, optimiser, learning rate, epoch count and split ratio are
%   all identical to it. Exactly two things change:
%
%     1. the split is seeded, so the validation set is recorded and every
%        number below can be regenerated;
%     2. the imageDataAugmenter is actually attached to the training
%        datastore, which the original script constructed and then dropped.
%
%   Writes classnet_aug.mat (the original classnet.mat is not touched) and
%   retrain_report.txt.

arguments
    seed (1,1) double = 42
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'eval'));

diary('retrain_report.txt');
diary on;
cleanup = onCleanup(@() diary('off'));

[root, ~] = paths_local();
t0 = tic;

fprintf('\n=== Retraining with augmentation ==========================\n');
fprintf('  seed %d, started %s\n', seed, datetime('now'));

imds = imageDatastore(fullfile(root, '17flowers', '17flowers'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.ReadFcn = @(f) imresize(imread(f), [256 256]);
imds.Labels  = categorical(imds.Labels);

% ---- the seeded split. This is finding 2 from RESULTS.md, fixed. --------
rng(seed);
[trainSet, valSet] = splitEachLabel(imds, 0.8, 'randomized');
fprintf('  train %d, validation %d\n', numel(trainSet.Files), numel(valSet.Files));

augmenter = imageDataAugmenter( ...
    'RandRotation',    [-30, 30], ...
    'RandXReflection', true, ...
    'RandXScale',      [0.8, 1.2], ...
    'RandYScale',      [0.8, 1.2]);

% ---- the one line the original script was missing. ---------------------
augTrain = augmentedImageDatastore([256 256 3], trainSet, ...
                                   'DataAugmentation', augmenter);

layers = [
    imageInputLayer([256 256 3])

    convolution2dLayer(5, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(512)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(17)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',    0.0003, ...
    'MaxEpochs',           75, ...
    'ValidationData',      valSet, ...
    'ValidationFrequency', 30, ...
    'Verbose',             true, ...
    'VerboseFrequency',    30, ...
    'Plots',               'none', ...
    'ExecutionEnvironment','gpu');

flowerNet = trainNetwork(augTrain, layers, options);

valFiles = string(valSet.Files);
save('classnet_aug.mat', 'flowerNet', 'seed', 'valFiles', '-v7.3');
fprintf('\n  trained in %.1f min, saved classnet_aug.mat\n', toc(t0)/60);

% ------------------------------------------------ evaluate --------------
fprintf('\n=== Held-out validation set (seeded, reproducible) =========\n');
[yPred, scores] = classify(flowerNet, valSet, 'ExecutionEnvironment', 'gpu');
names = string(categories(imds.Labels));
Rval  = evalClassification(valSet.Labels, yPred, 'Scores', scores, ...
                           'ClassNames', names, 'TopK', 3);

yTr   = classify(flowerNet, trainSet, 'ExecutionEnvironment', 'gpu');
trAcc = mean(yTr == trainSet.Labels);

fprintf('=== Comparison against the original run ====================\n');
fprintf('  %-34s %10s %10s\n', '', 'original', 'augmented');
fprintf('  %-34s %9.1f%% %9.1f%%\n', 'training accuracy', 98.5, 100*trAcc);
fprintf('  %-34s %9.1f%% %9.1f%%\n', 'validation accuracy', 64.3, 100*Rval.accuracy);
fprintf('  %-34s %9.1f  %9.1f\n', 'generalisation gap (points)', ...
        34.2, 100*(trAcc - Rval.accuracy));
fprintf('  %-34s %10s %9.3f\n', 'macro-F1', 'not measured', Rval.macroF1);
fprintf('\n  Original training accuracy is inferred, not measured: the\n');
fprintf('  original split was never seeded. See RESULTS.md.\n');
fprintf('  finished %s\n\n', datetime('now'));
end
