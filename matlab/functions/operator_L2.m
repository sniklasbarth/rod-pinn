function out = operator_L2(x2)
%OPERATOR_L2 see documentation of derivation or Artola et al. 2021
out = (zeros(6));

if isa(x2,'sym')
    out = sym(out);
end
out(1:3,4:6) = skew(x2(1:3));
out(4:6,1:3) = skew(x2(1:3));
out(4:6,4:6) = skew(x2(4:6));
end

