function T_EE = rodEndEffector(xi, xi_base,L)
%% Output equation for RodEndEffector
% xi is already interpolated using pls

% transform xi to matrix, so that it becomes easier to loop over all nodes
xi_mat = reshape(xi, 6, []);
N  = size(xi_mat,2);

% parameters required for discrete integration of distributed graviational
% load and discrete reconstruction of orientations via exponential update
ds = L/(N-1);
T0 = SE3Group.expMap(xi_base);
temp = T0;
for i = 1:N-1
    temp = temp * SE3Group.expMap(xi_mat(:,i)*ds);
end
T_EE = temp;
end


