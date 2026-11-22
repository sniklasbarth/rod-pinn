classdef SE3Group
    % SE3Group Class for operations on the SE(3) group (rigid body transformations)
    % Includes methods for exponential map, action, inverse, adjoint, and right Jacobian.
    % generate with chat.openai
    methods (Static)
        function T = expMap(xi)
            %% expMap Exponential map for the SE(3) group
            %
            % T = expMap(xi)
            %
            % Inputs:
            %   xi - A 6x1 vector representing the element in the Lie algebra se(3)
            %        (first 3 elements are the translation part, last 3 elements are the rotation part)
            %
            % Outputs:
            %   T - A 4x4 transformation matrix in the SE(3) group
            %
            % Note: One could ofcourse use expm(SE3Group.hat(xi)).
            % However, the analytical formulation here helps to understand
            % what is going on and where Jacobians are derived from. 
            % Regarding implementation: no comparison of computational
            % speed has been made.
            %
            % A batch xi (Nx6, N > 1) returns T as 4x4xN (expMapBatch);
            % not supported in generated code (MATLAB Coder).
            if coder.target('MATLAB') && numel(xi) ~= 6 % batch only in MATLAB, not in generated code
                T = SE3Group.expMapBatch(xi);
                return
            end
            if isrow(xi)
               xi = xi'; 
            end
            v = xi(1:3);
            omega = xi(4:6);
            theta = norm(omega);
            
            if theta == 0
                R = eye(3);
                V = eye(3);
            else
                omega_hat = SO3Group.skewSymmetric(omega);
                R =  SO3Group.expMap(omega);
                V = eye(3) + (1 - cos(theta)) / (theta^2) * omega_hat + (theta - sin(theta)) / (theta^3) * (omega_hat * omega_hat);
            end
            
            T = [R, V * v;
                0, 0, 0, 1];
        end
        
        function xi = logMap(T)
            % logMap Logarithm map for the SE(3) group
            %
            % xi = logMap(T)
            %
            % Inputs:
            %   T - A 4x4 transformation matrix in the SE(3) group
            %
            % Outputs:
            %   xi - A 6x1 vector representing the element in the Lie algebra se(3)
            %        (first 3 elements are the translation part, last 3 elements are the rotation part)
            
            R = T(1:3, 1:3);
            p = T(1:3, 4);
            
            omega = SO3Group.logMap(R);
            theta = norm(omega);
            
            if theta == 0
                V_inv = eye(3);
            else
                omega_hat = SO3Group.skewSymmetric(omega);
                V_inv = eye(3) - 0.5 * omega_hat + (1 - theta / (2 * tan(theta / 2))) / (theta^2) * (omega_hat * omega_hat);
            end
            
            v = V_inv * p;
            xi = [ v;omega;];
        end
        
        function v_trans = action(T, v)
            %% action Apply the action of a transformation matrix on a 3D vector
            %
            % v_trans = action(T, v)
            %
            % Inputs:
            %   T - A 4x4 transformation matrix
            %   v - A 3x1 vector to be transformed
            %
            % Outputs:
            %   v_trans - The transformed 3x1 vector
            
            v_hom = [v; 1];
            v_trans_hom = T * v_hom;
            v_trans = v_trans_hom(1:3);
        end
        
        function T_inv = inverse(T)
            %% inverse Compute the inverse of a transformation matrix
            %
            % T_inv = inverse(T)
            %
            % Inputs:
            %   T - A 4x4 transformation matrix
            %
            % Outputs:
            %   T_inv - The inverse transformation matrix
            
            R = T(1:3, 1:3);
            p = T(1:3, 4);
            T_inv = [R', -R' * p;
                0, 0, 0, 1];
        end
        
        function Ad_T = adjoint(T)
            %% adjoint Compute the adjoint action of a transformation matrix
            %
            % Ad_T = adjoint(T)
            %
            % Inputs:
            %   T - A 4x4 transformation matrix
            %
            % Outputs:
            %   Ad_T - The 6x6 adjoint matrix
            
            R = T(1:3, 1:3);
            p = T(1:3, 4);
            p_hat = SO3Group.skewSymmetric(p);
            Ad_T = [R, p_hat * R;
                zeros(3, 3), R];
        end
        
        function Jr = rightJacobian(xi)
            % rightJacobian Compute the right Jacobian for the SE(3) group
            %
            % Jr = rightJacobian(xi)
            %
            % Inputs:
            %   xi - A 6x1 vector representing the element in the Lie algebra se(3)
            %        (first 3 elements are the translation part, last 3 elements are the rotation part)
            %
            % Outputs:
            %   Jr - The 6x6 right Jacobian matrix
            Jr = SE3Group.leftJacobian(-xi);
        end
        
        function Jl = leftJacobian(xi)
            %% leftJacobian Compute the left Jacobian for the SE(3) group
            %
            % Jl = leftJacobian(xi)
            %
            % Inputs:
            %   xi - A 6x1 vector representing the element in the Lie algebra se(3)
            %        (first 3 elements are the translation part, last 3 elements are the rotation part)
            %
            % Outputs:
            %   Jl - The 6x6 left Jacobian matrix
            %
            % Source: [1] Barfoot et al., Asociating Uncertainty With Three-Dimensional Poses for Use in Estimation Problems, T-RO, 2014
            %
            % 'Owing to the nature of the cot(φ/2) function, there
            % are singularities associated with J (i.e., the inverse does not exist) at
            % theta= (2 * pi *  n) with m a non-zero integer' - Barfoot,
            % p.224, 2022
            
            omega = xi(4:6);
            v = xi(1:3);
            theta = norm(omega);
            
            if theta == 0
                Jl = eye(6);
            else
                omega_hat = SO3Group.skewSymmetric(omega);
                v_hat = SO3Group.skewSymmetric(v);
                % [1], Eq. (102)
                temp1 = 1/2*v_hat + ((theta - sin(theta))/theta^3)*(omega_hat*v_hat + v_hat*omega_hat +  omega_hat*v_hat*omega_hat)... % corect ...
                    - (1-(theta^2/2)-cos(theta))/theta^4*(omega_hat*omega_hat*v_hat + v_hat*omega_hat*omega_hat -  3*omega_hat*v_hat*omega_hat); % correct
                temp2 = - 1/2*((1 - (theta^2/2) - cos(theta))/theta^4 - 3*(theta - sin(theta) - (theta^3)/6)/theta^5);
                %                 temp2skew = SO3Group.skewSymmetric(temp2);
                Q = temp1 + temp2*(omega_hat*v_hat*omega_hat*omega_hat + omega_hat*omega_hat*v_hat*omega_hat);
                Jl = [SO3Group.leftJacobian(omega), Q;
                    zeros(3, 3), SO3Group.leftJacobian(omega)];
            end
        end
        function xi_hat = hat(xi)
            % hat(xi) computes the heteromorphism of the group
            %
            %
            v = xi(1:3);
            omega = xi(4:6);
            omega_hat = SO3Group.skewSymmetric(omega);
            xi_hat = [omega_hat, v;zeros(1,3), 0];
        end
        function omega_hat = skewSymmetric(omega)
            %% skewSymmetric Create a skew-symmetric matrix from a vector
            %
            % omega_hat = skewSymmetric(omega)
            %
            % Inputs:
            %   omega - A 3x1 vector
            %
            % Outputs:
            %   omega_hat - A 3x3 skew-symmetric matrix
            %
            % A batch omega (Nx3, N > 1) returns omega_hat as 3x3xN.
            if coder.target('MATLAB') && numel(omega) ~= 3 % batch only in MATLAB, not in generated code
                omega_hat = SO3Group.skewSymmetricBatch(omega);
                return
            end
            
            omega_hat = [  0,    -omega(3),  omega(2);
                omega(3),   0,   -omega(1);
                -omega(2),  omega(1),   0];
        end
        
        function T = expMapBatch(xi)
            %% expMapBatch Exponential map for a batch of se(3) elements
            %
            % Inputs:
            %   xi - Nx6 array (one element per row, translation part first)
            %
            % Outputs:
            %   T - 4x4xN transformation matrices (numeric only)
            bs = size(xi,1);
            v(:,1,:) = xi(:,1:3)';
            omega = xi(:,4:6);
            theta = vecnorm(omega,2,2);
            
            omega_hat = SO3Group.skewSymmetricBatch(omega);
            R = SO3Group.expMapBatch(omega);
            V = repmat(eye(3),1,1,bs) + reshape((1 - cos(theta))./(theta.^2),1,1,bs).* omega_hat + ...
                reshape((theta - sin(theta)) ./ (theta.^3),1,1,bs) .* pagemtimes(omega_hat, omega_hat);
            
            T = [R, pagemtimes(V,v);
                repmat([0, 0, 0, 1],1,1,bs)];
            % Replace transformations where theta is zero
            zeros_idx = (theta == 0);
            if any(zeros_idx)
                num_zeros = nnz(zeros_idx);
                T(:,:,zeros_idx) = [repmat(eye(3),1,1,num_zeros), v(:,:,zeros_idx); repmat([0, 0, 0, 1],1,1,num_zeros)];
            end
        end
        function R = rotation(T)
            % rotation(T) returns 3x3 Rotation-Matrix in T
            % T = [R, t; 0 0];
            %
            R = T(1:3,1:3);
        end
        function t = translation(T)
            % translation(T) returns 3x1 Trasnlation-Vector in T
            % T = [R, t; 0 0];
            %
            t = T(1:3,4);
        end
        
        function T = random()
           xi = rand(6,1);
           T = SE3Group.expMap(xi);
        end
    end
end
