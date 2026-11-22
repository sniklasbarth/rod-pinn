function u_g = gravity(xi, param, N_nodes)
%GRAVITY is a manual implementation of discrete gravitational load
% It is used in the wrapper class in combination with the exported state to
% strain function. The respective gradients are derived without symbolic
% tools as well and are implemented by derivative_gravity.
% The first absolute frame of the system is always assumed to be identity.
%
% parse graviation parameters

temp = param(1:3);
if isrow(temp)
    g = zeros(3,1);
    for i = 1:length(g)
        g(i) = temp(i);
    end
else
    g = temp;
end
L = param(4); % length of the rod
M3 = param(5); % 3rd diagonal entry of inertia density matrix
M_tip = param(6);
% transform xi to matrix, so that it becomes easier to loop over all nodes
xi_mat = reshape(xi, 6, []);

% this linear map adds 3 zero components to ug (for moments)
mapR3toR6 = [eye(3);zeros(3)];

% this linear map removes the first 3 components of xi
mapR6toR3 = [zeros(3), eye(3)]; %

% reduce to relevant strains
xi_mat = mapR6toR3 * xi_mat;
N  = size(xi_mat,2);
N_ = (N-1)/(N_nodes-1);
assert(mod(N_,1) ==  0, 'N_ is not an integer. Assert that input N_nodes is correct.')
% parameters required for discrete integration of distributed graviational
% load and discrete reconstruction of orientations via exponential update
ds = L/(N-1);
R0 = eye(3);
R = R0;
% assumption: nodes evenly distributed
gload = - M3*g; %
u_g_mat_loc = mapR3toR6*gload; % init local gravity forces
%
for i = 1:N-1
    % integrate orientations with linear interpolation of strains
        R = R * SO3Group.expMap(xi_mat(:,i)*ds);
        % additional mass at tip
        if i >= N-(N_+1)
            gload = -(M3 + M_tip)*g;
        end
        % concatenate local graviational loads
        u_g_mat_loc = [u_g_mat_loc, mapR3toR6*R'*gload];
        %     u_g_mat = u_g_mat - M3*ds*g; % update u_g_mat (discrete integration)
     
end
u_g = u_g_mat_loc(:);
end

