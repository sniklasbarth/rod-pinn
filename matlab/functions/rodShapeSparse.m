function T_EE = rodShapeSparse(xi, xi_base, L)
%% Output equation for RodEndEffector
% The computation from strains to configuration/EE pose is done using the
% exponetial update. xi is the piecewise linear interpolated strain field

% transform xi to matrix, so that it becomes easier to loop over all nodes
xi_mat = reshape(xi, 6, []);
N  = size(xi_mat,2);

% parameters required for discrete integration of distributed graviational
% load and discrete reconstruction of orientations via exponential update
ds = L/(N-1);
T0 = SE3Group.expMap(xi_base);
temp = T0;
T_EE(:,:,1) = temp;
for i = 1:N-1
    temp = temp * SE3Group.expMap(ds*xi_mat(:,i));
    T_EE(:,:,i+1) = temp;
end
end


