function [C, bf] = segConfusion(pred, gt, K, opts)
%SEGCONFUSION  Accumulate a pixel confusion matrix over a set of images.
%
%   C = segConfusion(pred, gt, K) where pred and gt are label matrices with
%   values in 1..K, or cell arrays of such matrices, or 3-D stacks.
%
%   [C, bf] = segConfusion(..., 'BoundaryF1', true) also returns the mean
%   boundary F1 score (Image Processing Toolbox). IoU is forgiving of ragged
%   edges on large regions -- a blobby mask and a crisp one can score alike.
%   Boundary F1 measures contour agreement specifically, so reporting both
%   says whether the shapes are right as well as the areas.
%
%   Pixels labelled 0 in gt are treated as "ignore" and excluded, which is
%   the usual convention for unlabelled or void regions.

arguments
    pred
    gt
    K                 (1,1) double
    opts.BoundaryF1   (1,1) logical = false
    opts.BoundaryTol  (1,1) double  = 3
end

pred = toCell(pred);
gt   = toCell(gt);
assert(numel(pred) == numel(gt), 'pred and gt must contain the same number of images.');

C  = zeros(K);
bfAcc = [];

for i = 1:numel(pred)
    p = double(pred{i}(:));
    t = double(gt{i}(:));
    assert(numel(p) == numel(t), 'Image %d: pred and gt differ in size.', i);

    keep = t > 0;                       % 0 = ignore
    p = p(keep); t = t(keep);
    assert(all(p >= 1 & p <= K), 'Image %d: prediction outside 1..%d.', i, K);

    C = C + accumarray([t, p], 1, [K K]);

    if opts.BoundaryF1
        s = bfscore(categorical(pred{i}, 1:K), categorical(gt{i}, 1:K), ...
                    opts.BoundaryTol);
        bfAcc = [bfAcc, s(:)]; %#ok<AGROW>
    end
end

if opts.BoundaryF1
    bf = mean(bfAcc(~isnan(bfAcc)));
else
    bf = [];
end
end

% =========================================================================
function c = toCell(x)
if iscell(x)
    c = x;
elseif ndims(x) == 3
    c = squeeze(num2cell(x, [1 2]));
else
    c = {x};
end
end
