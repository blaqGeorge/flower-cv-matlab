function demo_reported_numbers()
%DEMO_REPORTED_NUMBERS  What "64.3% accuracy" and "98.5% pixel accuracy" mean.
%
%   Pure arithmetic on the two reported figures -- nothing here is simulated
%   or assumed about the models themselves. The question in both cases is
%   the same: how much does the quoted number actually constrain?

fprintf('\n');
partA();
partB();
end

% =========================================================================
function partA()
fprintf('=== A. 64.3%% accuracy, 17 classes ==========================\n\n');
fprintf('  Chance level (1/17)              %.1f%%\n', 100/17);
fprintf('  Reported                         64.3%%\n');
fprintf('  Above chance by                  %.1f percentage points\n\n', 64.3 - 100/17);

fprintf('  The interval depends on how many images the number was measured on.\n');
fprintf('  A 1,360-image dataset splits many ways, and the bullet does not say:\n\n');
fprintf('    %-28s %8s   %s\n', 'validation split', 'n', '95% CI on 64.3%');
p = 0.643;
for n = [136 204 272 408]
    k = round(p*n);
    [lo, hi] = wilsonInterval(k, n);
    fprintf('    %-28s %8d   %.1f%% - %.1f%%  (+-%.1f pp)\n', ...
        sprintf('%.0f%% of 1360', 100*n/1360), n, 100*lo, 100*hi, 100*(hi-lo)/2);
end

fprintf('\n  So "64.3%%" is quoting three significant figures on a measurement\n');
fprintf('  whose uncertainty is around +-6 points. That is not a small point:\n');
fprintf('  it means 64.3%% and 60%% are the same result, and a reader who knows\n');
fprintf('  that will wonder whether you do. Quoting the interval, or at least\n');
fprintf('  the split size, converts a soft number into a defensible one.\n\n');
end

% =========================================================================
function partB()
fprintf('=== B. 98.5%% pixel accuracy, figure-ground segmentation =====\n\n');

PA = 0.985;
N  = 1e6;
E  = (1 - PA) * N;          % misclassified pixels, split evenly FP / FN

fprintf('  Pixel accuracy is dominated by whichever class owns the most pixels.\n');
fprintf('  Below: the same 98.5%% result, at different foreground fractions.\n\n');

fprintf('    %-12s %14s %14s %12s\n', ...
        'foreground', 'trivial pixel', 'your mean IoU', 'headroom');
fprintf('    %-12s %14s %14s %12s\n', ...
        'fraction', 'accuracy', '(consistent)', 'over trivial');

for f = [0.015 0.05 0.10 0.20 0.30 0.50]
    fg = f*N;  bg = N - fg;
    iouFg = (fg - E/2) / (fg + E/2);
    iouBg = (bg - E/2) / (bg + E/2);
    mIoU  = mean([max(iouFg,0), iouBg]);

    trivialPA   = max(1-f, f);
    trivialMIoU = trivialPA / 2;

    fprintf('    %-12.1f%% %13.1f%% %14.3f %12.1f pp\n', ...
            100*f, 100*trivialPA, mIoU, 100*(PA - trivialPA));
end

fprintf('\n  Read the top row. If the flowers occupy 1.5%% of each image, then\n');
fprintf('  labelling every pixel "background" scores 98.5%% pixel accuracy --\n');
fprintf('  exactly the reported figure, having segmented nothing whatsoever.\n');
fprintf('  Your U-Net is almost certainly far better than that. The problem is\n');
fprintf('  that the metric quoted cannot demonstrate it.\n\n');

fprintf('  Mean IoU can. At a 10%% foreground fraction the same result is a\n');
fprintf('  mean IoU around 0.92, which is a genuinely strong number and one\n');
fprintf('  that a trivial predictor cannot reach: its ceiling is 0.45.\n\n');

% The trap, run through the actual evaluator.
fprintf('  Demonstrated on the real code -- an all-background prediction\n');
fprintf('  where foreground is 10%% of pixels:\n');
C = [900000 0; 100000 0];
evalSegmentation(C, 'ClassNames', ["background", "flower"]);
end
