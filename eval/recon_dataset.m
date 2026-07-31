function recon_dataset()
%RECON_DATASET  Inspect the coursework data before evaluating anything.

root = 'C:\Users\geokw\OneDrive\Desktop\ComputerVis\courseWork';

% ------------------------------------------------- classification --------
imds = imageDatastore(fullfile(root, '17flowers', '17flowers'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
fprintf('\n=== 17flowers ==============================================\n');
fprintf('  images %d, classes %d\n', numel(imds.Files), numel(categories(imds.Labels)));
cnt = countEachLabel(imds);
fprintf('  per class: min %d, max %d\n', min(cnt.Count), max(cnt.Count));
fprintf('  80/20 split -> %d train, %d validation\n', ...
        round(0.8*numel(imds.Files)), numel(imds.Files) - round(0.8*numel(imds.Files)));

% ------------------------------------------------- segmentation ----------
imgPath = fullfile(root, 'daffodilSeg', 'daffodilSeg', 'ImagesRsz256');
lblPath = fullfile(root, 'daffodilSeg', 'daffodilSeg', 'LabelsRsz256');
imgs = imageDatastore(imgPath);
lbls = imageDatastore(lblPath);

fprintf('\n=== daffodilSeg ============================================\n');
fprintf('  images %d, label maps %d\n', numel(imgs.Files), numel(lbls.Files));
n = numel(imgs.Files);
fprintf('  80/20 split -> train 1:%d, validation %d:%d\n', ...
        round(0.8*n), round(0.8*n)+1, n);

% Do the file lists line up pairwise? A mismatch here silently evaluates
% every image against somebody else's mask.
[~, ia] = fileparts(string(imgs.Files));
[~, ib] = fileparts(string(lbls.Files));
fprintf('  first 3 image files : %s\n', strjoin(ia(1:3).', ', '));
fprintf('  first 3 label files : %s\n', strjoin(ib(1:3).', ', '));
fprintf('  names identical     : %d\n', isequal(ia, ib));

% What pixel values actually occur? The training script declares
% labelIDs = [1 3]. Anything else present is silently dropped.
vals = [];
for i = 1:numel(lbls.Files)
    L = imread(lbls.Files{i});
    vals = union(vals, unique(L(:)));
end
fprintf('  pixel values present in label maps: %s\n', mat2str(double(vals(:).')));
fprintf('  training script declares labelIDs : [1 3]\n');
missing = setdiff(double(vals(:).'), [1 3]);
if ~isempty(missing)
    fprintf('  UNDECLARED VALUES: %s  <-- these become <undefined>\n', mat2str(missing));
end

% Class balance across the whole label set -- this is what decides how
% badly pixel accuracy flatters the result.
tot = zeros(1, max(double(vals))+1);
for i = 1:numel(lbls.Files)
    L = double(imread(lbls.Files{i}));
    tot = tot + accumarray(L(:)+1, 1, [numel(tot) 1]).';
end
fprintf('\n  pixel frequency over all %d masks:\n', numel(lbls.Files));
for v = find(tot > 0)
    fprintf('    value %d : %12s px  (%.2f%%)\n', v-1, addComma(tot(v)), 100*tot(v)/sum(tot));
end
fprintf('\n');
end

function s = addComma(x)
s = regexprep(sprintf('%d', round(x)), '\d{1,3}(?=(\d{3})+$)', '$&,');
end
