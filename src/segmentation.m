clc; clear; close all;

% Define dataset paths for images and labels
imagePath = fullfile('..','data','daffodilSeg','ImagesRsz256');   % see README
labelPath = fullfile('..','data','daffodilSeg','LabelsRsz256');   % see README

% Define image size (RGB format)
imageSize = [256, 256, 3];

% Load flower images
imds = imageDatastore(imagePath);

% Define segmentation labels (flower vs. background)
classNames = ["flower", "background"]; 
labelIDs = [1, 3]; % Ensure values match dataset pixel classes

% Load segmentation masks
pxds = pixelLabelDatastore(labelPath, classNames, labelIDs);

% Verify label assignment
disp(pxds.ClassNames);
disp(pxds.Files);

resizeFcn = @(x) imresize(x, [256, 256]);
normalizationFcn = @(x) im2double(x);

imds.ReadFcn = @(filename) normalizationFcn(resizeFcn(imread(filename)));
pxds.ReadFcn = @(filename) resizeFcn(imread(filename)); % Labels don’t need normalization

numImages = numel(imds.Files);
splitRatio = 0.8;
numTrain = round(numImages * splitRatio);

% Manually split images & labels
trainImages = subset(imds, 1:numTrain);
valImages = subset(imds, numTrain+1:numImages);
trainLabels = subset(pxds, 1:numTrain);
valLabels = subset(pxds, numTrain+1:numImages);

% Create paired datastore for training and validation
trainData = pixelLabelImageDatastore(trainImages, trainLabels);
valData = pixelLabelImageDatastore(valImages, valLabels);

net = unet(imageSize, 2);  % Generate U-Net model
lgraph = layerGraph(net);  % Convert to layer graph

% Add pixel classification layer
lgraph.Layers
outputLayer = pixelClassificationLayer('Name', 'pixelClassificationLayer');
lgraph = addLayers(lgraph, outputLayer);

% Connect the output layer to the final softmax layer
lgraph = connectLayers(lgraph, "FinalNetworkSoftmax-Layer", "pixelClassificationLayer");

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0003, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'ValidationData', valData, ...
    'Verbose', true, ...
    'Plots', 'training-progress',...
    'ExecutionEnvironment', 'gpu');

segmentNet = trainNetwork(trainData, lgraph, options);
save('segmentnet.mat', 'segmentNet'); % Saves trained segmentation model
%{
testImage = im2double(imresize(imread(trainImages.Files{1}), [256, 256]));

%testImage = im2double(imresize(imread('test_daffodil.jpg'), [256, 256]));

% Predict segmentation
predictedMask = semanticseg(testImage, segmentNet);

imshowpair(testImage, predictedMask, 'montage');
%}