function [leg, obj, hout, mout] = legend_compact(h, labels, varargin)
%LEGEND_COMPACT Legend for the line handles h with the given labels. The
% options of the plot calls (ncol, anchor, buffer, ...) are ignored. A second
% call on the same axes replaces the first legend.
ax = ancestor(h(1), 'axes');
leg = legend(ax, h(:), labels, 'Interpreter', 'latex', 'Location', 'best');
obj = []; hout = []; mout = [];
end
