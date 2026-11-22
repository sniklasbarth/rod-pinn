function ys = smooth_moving(y, span)
%SMOOTH_MOVING Centered moving average of the column vector y, as smooth(y,
% span) of the Curve Fitting Toolbox: the span is rounded down to an odd
% number, and near the ends the window shrinks symmetrically (1, 3, 5, ...).
span = floor(span);
if mod(span,2) == 0
    span = span - 1;
end
y = y(:);
n = numel(y);
k = (1:n)';
half = min([repmat((span-1)/2,n,1), k-1, n-k], [], 2);
c = [0; cumsum(y)];
ys = (c(k+half+1) - c(k-half)) ./ (2*half+1);
end
