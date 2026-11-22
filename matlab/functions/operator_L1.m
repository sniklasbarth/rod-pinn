function out = operator_L1(x1)
%OPERATOR_L1 see documentation of derivation or Artola et al. 2021
out = (zeros(6));

if isa(x1,'sym')
    out = sym(out);
end
out(1:3,1:3) = skew(x1(4:6));
out(4:6,1:3) = skew(x1(1:3));
out(4:6,4:6) = skew(x1(4:6));
end

