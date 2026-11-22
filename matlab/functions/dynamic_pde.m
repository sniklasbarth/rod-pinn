function dxds= dynamic_pde(x,tau, xt, x0, C, M, D)
% DYNAMIC_ODE validated
% Check the latex documentation to understand the content of each
% expression
x1 = x(1:6); % velocities
x2 = x(7:12); % forces/moments
OO = zeros(size(M));

% w = x1(4:6);
% Omega = [skew(w),zeros(3);zeros(3),skew(w)];

f1 = [OO, C; M, -(operator_L1(C*x2)+operator_L0(x0))*D]*xt;
A2 = [-operator_L0(x0).', OO; OO, operator_L0(x0)];
A1 = [ OO, -operator_L1(x1).'; operator_L1(x1),  operator_L2(x2)];
f2 = -(A2 - A1*[M,OO; OO, C])*x;
f3 = -[zeros(size(x1));tau];

dxds = f1+f2+f3;

if any(isnan(dxds))
   warning('PDE has nan entries.')
end
end
