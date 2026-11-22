classdef S3Group
    % S3Group Class for operations on the S^3 group (unit quaternions)
    % Includes methods for action, inverse, adjoint, and right Jacobian.
    
    methods (Static)
        function i = identity
           %  identity gives the neutral element of the group
           i = [1,0,0,0]; 
        end
        function q = expMap(v)
            % expMap Exponential map for the S^3 group
            %
            % q = expMap(v)
            %
            % Inputs:
            %   v - A 3x1 vector representing the element in the Lie algebra su(2)
            %
            % Outputs:
            %   q - A 1x4 vector representing the unit quaternion in the S^3 group
            %
            % A batch v (Nx3, N > 1) returns q as Nx4 (expMapBatch).
            if coder.target('MATLAB') && numel(v) ~= 3 % batch only in MATLAB, not in generated code
                q = S3Group.expMapBatch(v);
                return
            end
            
            norm_v = norm(v);
            
            if norm_v == 0
                q = S3Group.identity; % Identity quaternion
            else
                q0 = cos(norm_v / 2);
                q123 = (v / norm_v) * sin(norm_v / 2);
                q = [q0, q123'];
            end
        end
        
        function v = logMap(q)
            %% logMap Logarithm map for the S^3 group
            %
            % v = logMap(q)
            %
            % Inputs:
            %   q - A 1x4 vector representing the unit quaternion in the S^3 group
            %
            % Outputs:
            %   v - A 3x1 vector representing the element in the Lie algebra su(2)
            %
            % A batch q (Nx4, N > 1) returns v as Nx3 (logMapBatch).
            if coder.target('MATLAB') && numel(q) ~= 4 % batch only in MATLAB, not in generated code
                v = S3Group.logMapBatch(q);
                return
            end
            
            if q(1) == 1
                v = zeros(3,1); % Zero vector
            else
                q_vec = q(2:4);
                norm_q_vec = norm(q_vec);
                theta = 2 * atan2(norm_q_vec, q(1));
                v = (theta / norm_q_vec) * q_vec';
            end
        end
        
        function q = expMapBatch(v)
            %% expMapBatch Exponential map for a batch of su(2) elements
            %
            % Inputs:
            %   v - Nx3 array (one element per row)
            %
            % Outputs:
            %   q - Nx4 unit quaternions (numeric only)
            norm_v = vecnorm(v,2,2);
            q = [cos(norm_v / 2), (v ./ norm_v) .* sin(norm_v ./ 2)];
            % replace if norm_v == 0
            zeros_idx = (norm_v == 0);
            if any(zeros_idx)
                q(zeros_idx,:) = repmat([1,0,0,0],nnz(zeros_idx),1);
            end
        end
        
        function v = logMapBatch(q)
            %% logMapBatch Logarithm map for a batch of unit quaternions
            %
            % Inputs:
            %   q - Nx4 unit quaternions (one per row)
            %
            % Outputs:
            %   v - Nx3 elements of su(2) (numeric only)
            q_vec = q(:,2:4);
            norm_q_vec = vecnorm(q_vec,2,2);
            theta = 2 .* atan2(norm_q_vec, q(:,1));
            v = (theta ./ norm_q_vec) .* q_vec;
            % replace if q(1) == 1
            zeros_idx = (q(:,1) == 1);
            if any(zeros_idx)
                v(zeros_idx,:) = repmat(zeros(1,3),nnz(zeros_idx),1);
            end
        end
        
        function v_rot = action(q, v)
            % action Apply the action of a unit quaternion on a 3D vector
            %
            % v_rot = action(q, v)
            %
            % Inputs:
            %   q - A 1x4 unit quaternion
            %   v - A 3x1 vector to be rotated
            %
            % Outputs:
            %   v_rot - The rotated 3x1 vector
            
            if isrow(v)
                v = v';
            end
            
            v_quat = [0, v'];
            q_conj = S3Group.inverse(q);
            v_rot_quat = S3Group.quatmultiply(S3Group.quatmultiply(q, v_quat), q_conj);
            v_rot = v_rot_quat(2:4)';
        end
        
        function q_inv = inverse(q)
            % inverse Compute the inverse of a unit quaternion
            %
            % q_inv = inverse(q)
            %
            % Inputs:
            %   q - A 1x4 unit quaternion
            %
            % Outputs:
            %   q_inv - The inverse quaternion
            if iscolumn(q)
               q = q.'; 
            end
            q_inv = [q(1), -q(2:4)];
        end
        
        function Ad_q = adjoint(q)
            % adjoint Compute the adjoint action of a unit quaternion
            %
            % Ad_q = adjoint(q)
            %
            % Inputs:
            %   q - A 1x4 unit quaternion
            %
            % Outputs:
            %   Ad_q - The 3x3 adjoint matrix
            
            q_vec = q(2:4);
            q0 = q(1);
            Q = [0, -q_vec(3), q_vec(2);
                q_vec(3), 0, -q_vec(1);
                -q_vec(2), q_vec(1), 0];
            Ad_q = q0 * eye(3) + Q;
        end
        
        function Jr = rightJacobian(v)
            % rightJacobian Compute the right Jacobian for the S^3 group
            %
            % Jr = rightJacobian(v)
            %
            % Inputs:
            %   v - A 3x1 vector representing the element in the Lie algebra su(2)
            %
            % Outputs:
            %   Jr - The 3x3 right Jacobian matrix
            
            norm_v = norm(v);
            
            if norm_v == 0
                Jr = eye(3);
            else
                I = eye(3);
                V = S3Group.skewSymmetric(v);
                theta = norm_v;
                
                Jr = I - (1 - cos(theta)) / (theta^2) * V + (theta - sin(theta)) / (theta^3) * (V * V);
            end
        end
        
        function V = skewSymmetric(v)
            % skewSymmetric Create a skew-symmetric matrix from a vector
            %
            % V = skewSymmetric(v)
            %
            % Inputs:
            %   v - A 3x1 vector
            %
            % Outputs:
            %   V - A 3x3 skew-symmetric matrix
            
            V = [  0,    -v(3),  v(2);
                v(3),   0,   -v(1);
                -v(2),  v(1),   0];
        end
        
        function q_out = quatmultiply(q1, q2)
            % Quaternion multiplication
            if iscolumn(q1)
               q1 = q1.'; 
            end
            if iscolumn(q2)
               q2 = q2.'; 
            end
            q_out = [q1(1)*q2(1) - dot(q1(2:4), q2(2:4)), ...
                q1(1)*q2(2:4) + q2(1)*q1(2:4) + cross(q1(2:4), q2(2:4))];
        end
        
        function R = quaternionToRotationMatrix(q)
            %% Converts (non-)/unit quaternion to rotation matrix
            % Input:
            %   q - 4x1 quaternion [q0; q1; q2; q3]
            % Output:
            %   R - 3x3 rotation matrix
            
            
            % Extract quaternion components
            q0 = q(1);
            q1 = q(2);
            q2 = q(3);
            q3 = q(4);
            
            % Compute the rotation matrix
            norm_q_sq = norm(q)^2;
            
            R = (1/norm_q_sq) * [
                q0^2 + q1^2 - q2^2 - q3^2, 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2);
                2 * (q1*q2 + q0*q3), q0^2 + q2^2 - q1^2 - q3^2, 2 * (q2*q3 - q0*q1);
                2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), q0^2 + q3^2 - q1^2 - q2^2
                ];
        end
        
    end
end
