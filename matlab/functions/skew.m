function [Y] = skew(v)
%CROSSPRODUCT_MATRIX create crossproduct matrix from 
% symbolic vectorial input $v \in R^{3}$
%
% https://de.wikipedia.org/wiki/Kreuzprodukt

% generators
e1 = ([1;0;0]);
e2 = ([0;1;0]);
e3 = ([0;0;1]);
if isa(v,'sym')
    e1 = sym(e1);
    e2 = sym(e2);
    e3 = sym(e3);
else
    
end
Y = [0 -e3.'*v e2.'*v; e3.'*v 0 -e1.'*v; -e2.'*v e1.'*v 0];
end

