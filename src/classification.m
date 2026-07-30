
clc; clear; close all;



% Define dataset path
datasetPath = fullfile('..','data','17flowers');   % see README for dataset download

% Define image size (must match CNN input layer)
imageSize = [256, 256, 3]; 


imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames'); % Assign labels from folder names

imds.ReadFcn = @(filename) imresize(imread(filename), [256, 256]);

imds.Labels = categorical(imds.Labels);

I = read(imds);

[trainSet, valSet] = splitEachLabel(imds, 0.8, 'randomized');

augmenter = imageDataAugmenter( ...
    'RandRotation', [-30, 30], ...   % Random rotation (-30° to 30°)
    'RandXReflection', true, ...     % Random horizontal flipping
    'RandXScale', [0.8, 1.2], ...    % Random width scaling
    'RandYScale', [0.8, 1.2]);       % Random height scaling
    

numClasses = numel(categories(imds.Labels));
disp(['Number of classes detected: ', num2str(numClasses)]);

layers = [
    imageInputLayer([256, 256, 3])

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

    fullyConnectedLayer(512) % Larger Dense Layer
    reluLayer
    dropoutLayer(0.5) % Prevent overfitting

    fullyConnectedLayer(17) % Must match dataset categories
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0003, ...
    'MaxEpochs', 75, ...
    'ValidationData', valSet, ...
    'Verbose', true, ...
    'Plots', 'training-progress',...
    'ExecutionEnvironment', 'gpu');

flowerNet = trainNetwork(trainSet, layers, options);

save('classnet.mat', 'flowerNet');