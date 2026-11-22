function [J_ug_xi] = gravity_derivative(xi,param, N_nodes)
%% gravity_derivative This is the Jacobian of the function 'gravity'.
% w.r.t. to the strains xi.
%
% It has been derived using group theory, although symbolic dervation.
% Although symbolic derivation using some simple substitution worked as
% well in the end, which could have saved me a lot of time and thinking.
%
% "Denken ist wie Schaukeln. Man ist beschäftigt, aber man kommt nicht
% voran." - Spruch auf einer Karte
temp = param(1:3);
if isrow(temp)
    g = zeros(3,1);
    for i = 1:length(g)
        g(i) = temp(i);
    end
else
    g = temp;
end

norm_g = norm(g);
if norm_g ~= 0
    e_g = g/norm_g;
else
    e_g = zeros(size(g));
end

L = param(4);
M3 = param(5);
M_tip = param(6);
% transform xi to matrix, so that it becomes easier to loop over all nodes
xi_mat = reshape(xi, 6, []);
xi_mat(1:3,:) = [];
N  = size(xi_mat,2);
% assumption: nodes evenly distributed
ds = L/(N-1);
G0 = L * M3 * norm_g;
R0 = eye(3);

N  = size(xi_mat,2);
N_ = (N-1)/(N_nodes-1);
assert(mod(N_,1) ==  0, 'N_ is not an integer. Assert that input N_nodes is correct.')

%% Compute all sub jacobians Jik of the Jacobian du/dxi
%
% Jik = u_i/tau_k
%
% This is done column-wise. Gradient wrt. tau_k for all u_i.
X0 = eye(3);
temp_matrix = zeros(length(xi(:)));
for k = 1:N
    xi_k = ds*xi_mat(:,k);
    X_k = SO3Group.expMap(xi_k);
%     subCol = [];
    j_i_k = zeros(3);
    for i = 1:N
        % Cases:    G_i * ds * Jik(X0,X1,xi_k*ds,e_g) for i>k
        %           0   else
        % with     G_i = (G_0 - i*mu*g*ds)
        if i > k
            % set X0, Xk and X1 so that
            % R_i = X0 Xk X1 = R0 Exp(xi_0) .. Exp(x_i-1)
            % with R_0 = Identity
            % and
            % ug_i = R_i^T * (g_i)
            %
            % It follows that:
            % X0 = R0 * Exp(xi_k-1)
            % Xk = Exp(xi_k)
            % X1 = Exp(xi_k+1) ... Exp(xi_i-1)
            
            % compute X1
            X1 = eye(3);
            for j = k+1:i-1
                X1 = X1*SO3Group.expMap(ds*xi_mat(:,j));
            end
            if i<=N-(N_+1) % anpassung
                u_g = -M3*norm_g;
            else
                u_g = -(M3 + M_tip)*norm_g;
            end
            %             j_i_k = (G0 - (i-1)*M3*norm_g*ds)*ds*anaGrad(X0,X1,xi_k,e_g);
            j_i_k = u_g*ds*anaGrad(X0,X1,xi_k,e_g);
            
            
        else
            j_i_k = zeros(3);
        end
        % append rows
        
        temp_matrix(((i-1)*6 + 1):(i*6), ((k-1)*6 + 1):((k)*6)) = ...
        [zeros(size(j_i_k)),j_i_k;
        ;zeros(size(j_i_k)),zeros(size(j_i_k))];
    end
    % append full column to gradient
%     temp = [temp,zeros(size(subCol)),subCol];
    % update X0 for the gradient xi_k+1
    X0 = X0*X_k;
end
J_ug_xi = temp_matrix;
end

function J_Zg_tau2 = anaGrad(R1,R3,tauk,g)
%% gradient of D(R^T*g)/Dtau
% see Eq.(xy) from paper: TODO
X_k     = SO3Group.expMap(tauk);
Y       = R1*X_k*R3;
Z       = SO3Group.inverse(Y);
skw_g   = SO3Group.skewSymmetric(g);

dZgdZ = -Z*skw_g;
dZdY = -SO3Group.adjoint(Y);
dYdX  = SO3Group.adjoint(SO3Group.inverse(R3));
dXdTau = SO3Group.rightJacobian(tauk);
J_Zg_tau2 =dZgdZ*dZdY*dYdX*dXdTau;

end
