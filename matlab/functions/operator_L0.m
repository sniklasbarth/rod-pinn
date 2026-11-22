function out = operator_L0(x0)
%OPERATOR_L2 see documentation of derivation or Artola et al. 2021
v0 = x0(1:3);
u0 = x0(4:6);
out = (zeros(6));

if isa(x0,'sym')
    out = sym(out);
end
out(1:3,1:3) = skew(u0);
out(4:6,1:3) = skew(v0);
out(4:6,4:6) = skew(u0);
end

