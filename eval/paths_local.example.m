function [root, modelDir] = paths_local()
%PATHS_LOCAL  Local dataset and model locations.
%
%   Copy this file to paths_local.m and edit. paths_local.m is gitignored,
%   so machine-specific paths stay out of the repository.
%
%   root      folder containing 17flowers/ and daffodilSeg/
%   modelDir  subfolder of root containing classnet.mat and segmentnet.mat

root     = 'C:\path\to\courseWork';
modelDir = 'trained';
end
