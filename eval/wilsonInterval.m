function [lo, hi] = wilsonInterval(k, n, alpha)
%WILSONINTERVAL  Confidence interval for a proportion (Wilson score method).
%
%   [lo, hi] = wilsonInterval(k, n) returns the 95% interval for k successes
%   out of n trials.
%
%   Why this and not k/n +- 1.96*sqrt(p(1-p)/n): the normal approximation
%   misbehaves near 0 and 1 (it can produce intervals outside [0,1]) and is
%   poor for small n. The Wilson score interval is well behaved everywhere
%   and is the standard recommendation for reporting accuracy on a test set.
%
%   This matters more than it looks. A validation accuracy quoted to one
%   decimal place on a few hundred images implies a precision the sample
%   size does not support: 64.3% on 272 images is 64.3% (95% CI 58.4-69.7).
%   Reporting the interval is not hedging, it is stating what was measured.

arguments
    k     (1,1) double {mustBeNonnegative}
    n     (1,1) double {mustBePositive}
    alpha (1,1) double {mustBePositive} = 0.05
end

p = k / n;
z = -sqrt(2) * erfcinv(2 * (1 - alpha/2));      % normal quantile, no toolbox

denom  = 1 + z^2/n;
centre = (p + z^2/(2*n)) / denom;
half   = (z/denom) * sqrt(p*(1-p)/n + z^2/(4*n^2));

lo = max(0, centre - half);
hi = min(1, centre + half);
end
