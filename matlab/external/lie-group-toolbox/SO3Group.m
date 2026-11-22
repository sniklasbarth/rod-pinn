classdef SO3Group
    % SO3Group Class for operations on the SO(3) group (rotation matrices)
    % Includes methods for exponential map, action, inverse, adjoint, and right Jacobian.
    % generate with chat.openai
    
    methods (Static)
        function R = expMap(omega)
            %% expMap Exponential map for the SO(3) group
            %
            % R = expMap(omega)
            %
            % Inputs:
            %   omega - A 3x1 vector representing the element in the Lie algebra so(3)
            %
            % Outputs:
            %   R - A 3x3 rotation matrix in the SO(3) group
            %
            % A batch omega (Nx3, N > 1) returns R as 3x3xN (expMapBatch);
            % not supported in generated code (MATLAB Coder).
            if coder.target('MATLAB') && numel(omega) ~= 3 % batch only in MATLAB, not in generated code
                R = SO3Group.expMapBatch(omega);
                return
            end
            
            theta = sum((omega).^2)^(1/2);
            
            if theta == 0
                R = (eye(3));
            else
                omega_hat = SO3Group.skewSymmetric(omega);
                R = eye(3) + sin(theta) / theta * omega_hat + (1 - cos(theta)) / (theta^2) * (omega_hat * omega_hat);
            end
        end
        
        function omega = logMap(R)
            % logMap Logarithm map for the SO(3) group
            %
            % omega = logMap(R)
            %
            % Inputs:
            %   R - A 3x3 rotation matrix in the SO(3) group
            %
            % Outputs:
            %   omega - A 3x1 vector representing the element in the Lie algebra so(3)
            temp = (trace(R) - 1) / 2;
            
            warningACOS = ~(abs(temp)<=1);
            if warningACOS
                warning(['Ill-defined acos function call in SO3Group.logMap'])
            end
            theta = acos((trace(R) - 1) / 2);
            
            if theta == 0
                omega = ([0; 0; 0]); % Zero vector
            else
                omega_hat = theta*(R - R') / (2 * sin(theta));
                omega = [omega_hat(3, 2); omega_hat(1, 3); omega_hat(2, 1)];
            end
        end
        
        function v_rot = action(R, v)
            %% action Apply the action of a rotation matrix on a 3D vector
            %
            % v_rot = action(R, v)
            %
            % Inputs:
            %   R - A 3x3 rotation matrix
            %   v - A 3x1 vector to be rotated
            %
            % Outputs:
            %   v_rot - The rotated 3x1 vector
            
            v_rot = R * v;
        end
        
        function R_inv = inverse(R)
            %% inverse Compute the inverse of a rotation matrix
            %
            % R_inv = inverse(R)
            %
            % Inputs:
            %   R - A 3x3 rotation matrix
            %
            % Outputs:
            %   R_inv - The inverse rotation matrix
            
            R_inv = R';
        end
        
        function Ad_R = adjoint(R)
            %% adjoint Compute the adjoint action of a rotation matrix
            %
            % Ad_R = adjoint(R)
            %
            % Inputs:
            %   R - A 3x3 rotation matrix
            %
            % Outputs:
            %   Ad_R - The 3x3 adjoint matrix
            
            Ad_R = R;
        end
        
        function Jr = rightJacobian(omega)
            %% rightJacobian Compute the right Jacobian for the SO(3) group
            %
            % Jr = rightJacobian(omega)
            %
            % Inputs:
            %   omega - A 3x1 vector representing the element in the Lie algebra so(3)
            %
            % Outputs:
            %   Jr - The 3x3 right Jacobian matrix
            
            theta = norm(omega);
            
            if theta == 0
                Jr = eye(3);
            else
                I = eye(3);
                omega_hat = SO3Group.skewSymmetric(omega);
                Jr = I - (1 - cos(theta)) / (theta^2) * omega_hat + (theta - sin(theta)) / (theta^3) * (omega_hat * omega_hat);
            end
        end
        
        
        function Jl = leftJacobian(omega)
            %% leftJacobian Compute the left Jacobian for the SO(3) group
            %
            % Jl = leftJacobian(omega)
            %
            % Inputs:
            %   omega - A 3x1 vector representing the element in the Lie algebra so(3)
            %
            % Outputs:
            %   Jl - The 3x3 left Jacobian matrix
            
            theta = norm(omega);
            
            if theta == 0
                Jl = eye(3);
            else
                I = eye(3);
                omega_hat = SO3Group.skewSymmetric(omega);
                Jl = I + (1 - cos(theta)) / (theta^2) * omega_hat + (theta - sin(theta)) / (theta^3) * (omega_hat * omega_hat);
            end
        end
        
        function J = leftJacobianVariant(omega)
            % Eq. 7.37a in T. Barfoot's book on state estimation in robotics
            % Inputs:
            %   phi - Scalar angle
            %   a   - 3x1 unit vector (axis of rotation)
            % Output:
            %   J   - 3x3 matrix (Jacobian)
            
            % Skew-symmetric matrix of vector a
            w = norm(omega);
            a = omega/w;
            a_hat =[  0    -a(3)   a(2);
                a(3)   0     -a(1);
                -a(2)  a(1)    0   ];
            
            % Compute the matrix J
            J = (sin(w)/w) * eye(3) + ...
                (1 - sin(w)/w) * (a * a') + ...
                ((1 - cos(w))/w) * a_hat;
        end
        function J = leftJacobianSeries(omega,N)
            % Eq. 7.37a in T. Barfoot's book on state estimation in robotics
            % Inputs:
            %   phi - Scalar angle
            %   a   - 3x1 unit vector (axis of rotation)
            % Output:
            %   J   - 3x3 matrix (Jacobian)
            
            % Skew-symmetric matrix of vector a
            omega_hat = [  0    -omega(3)   omega(2);
                omega(3)   0     -omega(1);
                -omega(2)  omega(1)    0   ];
            
            % Compute the matrix J
            J = zeros(3);
            for i = 0:N
               J = J + 1/factorial(i+1) * omega_hat^i;
            end
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
        
        function R = expMapBatch(omega)
            %% expMapBatch Exponential map for a batch of so(3) elements
            %
            % Inputs:
            %   omega - Nx3 array (one element per row)
            %
            % Outputs:
            %   R - 3x3xN rotation matrices (numeric only)
            bs = size(omega,1);
            theta = vecnorm(omega,2,2);
            
            omega_hat = SO3Group.skewSymmetricBatch(omega);
            R = repmat(eye(3),1,1,bs) + reshape(sin(theta)./theta,1,1,bs).*omega_hat + ...
                reshape((1.-cos(theta))./(theta.^2),1,1,bs).* pagemtimes(omega_hat,omega_hat);
            % Replace transformations where theta is zero
            zeros_idx = (theta == 0);
            if any(zeros_idx)
                R(:,:,zeros_idx) = repmat(eye(3),1,1,nnz(zeros_idx));
            end
        end
        
        function omega_hat = skewSymmetricBatch(omega)
            %% skewSymmetricBatch Skew-symmetric matrices for a batch of vectors
            %
            % Inputs:
            %   omega - Nx3 array (one vector per row)
            %
            % Outputs:
            %   omega_hat - 3x3xN skew-symmetric matrices (numeric only)
            bs = size(omega,1);
            omega_hat = zeros(3,3,bs);
            omega_hat(1,[2 3],:) = [-omega(:,3),  omega(:,2)]';
            omega_hat(2,[1 3],:) = [omega(:,3),   -omega(:,1)]';
            omega_hat(3,[1 2],:) = [-omega(:,2),  omega(:,1)]';
        end
        
        function omega = skewInverse(omega_hat)
            %% skewInverse Inverse of the skew-symmetric matrix to vector
            %
            % omega = skewInverse(omega_hat)
            %
            % Inputs:
            %   omega_hat - A 3x3 skew-symmetric matrix
            %
            % Outputs:
            %   omega - A 3x1 vector
            
            omega = [omega_hat(3, 2);
                omega_hat(1, 3);
                omega_hat(2, 1)];
        end
        
    end
end
