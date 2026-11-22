classdef TestLieGroups < TestSuperClass
    % The three As of unit testing: arrange, act, assert
    %
    % Summary of all tests performed:
    %
    % - Exp and Log test: u = Log(Exp(u))
    % - Action test
    % - Consistency between S^3 and SO(3)
    % - Test all inverse operators (x^-1)*x = Identity
    % - Test the analytic Jacobian by computing the numerical Jacobian
    %
    % Copyright(C)2024, Institut fuer Mechatronische Systeme, Leibniz
    % Universitaet Hannover
    % Author: Max Bartholdt (max.bartholdt[at]imes.uni-hannover.de)
    %
    methods (Test)
        %% Exp and Log test: u = Log(Exp(u))
        function testS3GroupLogExpMap(testCase)
            % Test exponential and logarithm map for S^3 group
            disp('Testing S3Group...');
            
            % Create a random vector in the Lie algebra su(2)
            u = randn(3, 1);
            
            % Compute the exponential map
            q = S3Group.expMap(u);
            
            % Compute the logarithm map
            u_reconstructed = S3Group.logMap(q);
            
            % Verify that the reconstructed vector is close to the original
            testCase.verifyLessThan(abs(u - u_reconstructed), 1e-10);
        end
        
        function testSO3GroupLogExpMap(testCase)
            % Test exponential and logarithm map for SO(3) group
            disp('Testing SO3Group...');
            
            % Create a random vector in the Lie algebra so(3) with a rotation
            % angle below pi (the logarithm is unique only there)
            u = randn(3, 1);
            u = 0.9*pi*rand*u/norm(u);
            
            % Compute the exponential map
            R = SO3Group.expMap(u);
            
            % Compute the logarithm map
            u_reconstructed = SO3Group.logMap(R);
            
            % Verify that the reconstructed vector is close to the original
            testCase.verifyLessThan(abs(u - u_reconstructed), 1e-10);
        end
        
        function testSE3GroupLogExpMap(testCase)
            % Test exponential and logarithm map for SE(3) group
            disp('Testing SE3Group...');
            
            % Create a random vector in the Lie algebra se(3) with a rotation
            % angle below pi (the logarithm is unique only there)
            u = randn(6, 1);
            u(4:6) = 0.9*pi*rand*u(4:6)/norm(u(4:6)); % [translation; rotation]
            
            % Compute the exponential map
            T = SE3Group.expMap(u);
            
            % Compute the logarithm map
            u_reconstructed = SE3Group.logMap(T);
            
            % Verify that the reconstructed vector is close to the original
            testCase.verifyLessThan(abs(u - u_reconstructed), 1e-10);
        end
        %% Action test
        
        function testS3GroupAction(testCase)
            % Test the action method for S^3 group
            disp('Testing S3Group action...');
            
            % Create a random vector in the Lie algebra su(2)
            u = randn(3, 1);
            
            % Compute the exponential map to get a quaternion
            q = S3Group.expMap(u);
            
            % Create a random vector to act upon
            v = randn(3, 1);
            
            % Apply the action
            v_rot = S3Group.action(q, v);
            
            % Verify that the action produces a rotated vector
            % Use the quaternion rotation formula directly for comparison
            q_conj = S3Group.inverse(q);
            v_quat = [0, v'];
            v_rot_quat = S3Group.quatmultiply(S3Group.quatmultiply(q, v_quat), q_conj);
            v_rot_expected = v_rot_quat(2:4)';
            
            % Verify that the rotated vectors are close
            testCase.verifyLessThan(abs(v_rot - v_rot_expected), 1e-10);
        end
        
        function testSO3GroupAction(testCase)
            % Test the action method for SO(3) group
            disp('Testing SO3Group action...');
            
            % Create a random vector in the Lie algebra so(3)
            u = randn(3, 1);
            
            % Compute the exponential map to get a rotation matrix
            R = SO3Group.expMap(u);
            
            % Create a random vector to act upon
            v = randn(3, 1);
            
            % Apply the action
            v_rot = SO3Group.action(R, v);
            
            % Verify that the action produces a rotated vector
            v_rot_expected = R * v;
            
            % Verify that the rotated vectors are close
            testCase.verifyLessThan(abs(v_rot - v_rot_expected), 1e-10);
        end
        
        function testSE3GroupAction(testCase)
            % Test the action method for SE(3) group
            disp('Testing SE3Group action...');
            
            % Create a random vector in the Lie algebra se(3)
            u = randn(6, 1);
            
            % Compute the exponential map to get a transformation matrix
            T = SE3Group.expMap(u);
            
            % Create a random vector to act upon
            v = randn(3, 1);
            
            % Apply the action
            v_rot_homogeneous = SE3Group.action(T,v);
            v_rot = v_rot_homogeneous(1:3);
            
            % Verify that the action produces a transformed vector
            R = T(1:3, 1:3);
            p = T(1:3, 4);
            v_rot_expected = R * v + p;
            
            % Verify that the transformed vectors are close
            testCase.verifyLessThan(abs(v_rot - v_rot_expected), 1e-10);
        end
        %% Consistency between S3 and SO(3)
        function testS3andSO3ActionConsistency(testCase)
            % Test that the action of S^3 and SO(3) groups produce the same result
            disp('Testing S3Group and SO3Group action consistency...');
            
            % Create a random vector in the Lie algebra su(2) or so(3)
            u = randn(3, 1);
            
            % Compute the exponential map to get a quaternion and a rotation matrix
            q = S3Group.expMap(u);
            R = SO3Group.expMap(u);
            
            % Create a random vector to act upon
            v = randn(3, 1);
            
            % Apply the actions
            v_rot_s3 = S3Group.action(q, v);
            v_rot_so3 = SO3Group.action(R, v);
            
            % Verify that the rotated vectors are close
            testCase.verifyLessThan(abs(v_rot_s3 - v_rot_so3), 1e-10);
        end
        %% Test all inverse operators (x^-1)*x = Identity
         function testS3GroupInverse(testCase)
            % Test the inverse method for S^3 group
            disp('Testing S3Group inverse...');
            
            % Create a random vector in the Lie algebra su(2)
            u = randn(3, 1);

            % Compute the exponential map to get a quaternion
            q = S3Group.expMap(u);

            % Compute the inverse quaternion
            q_inv = S3Group.inverse(q);

            % Verify that q * q_inv = identity quaternion [1, 0, 0, 0]
            identity_quat = S3Group.quatmultiply(q, q_inv);
            testCase.verifyLessThan(abs(identity_quat - [1, 0, 0, 0]), 1e-10);
        end

        function testSO3GroupInverse(testCase)
            % Test the inverse method for SO(3) group
            disp('Testing SO3Group inverse...');

            % Create a random vector in the Lie algebra so(3)
            u = randn(3, 1);

            % Compute the exponential map to get a rotation matrix
            R = SO3Group.expMap(u);

            % Compute the inverse rotation matrix
            R_inv = SO3Group.inverse(R);

            % Verify that R * R_inv = identity matrix
            identity_matrix = R * R_inv;
            identity_expected = eye(3);
            testCase.verifyLessThan(abs(identity_matrix(:) - identity_expected(:)), 1e-10);
        end

        function testSE3GroupInverse(testCase)
            % Test the inverse method for SE(3) group
            disp('Testing SE3Group inverse...');

            % Create a random vector in the Lie algebra se(3)
            u = randn(6, 1);

            % Compute the exponential map to get a transformation matrix
            T = SE3Group.expMap(u);

            % Compute the inverse transformation matrix
            T_inv = SE3Group.inverse(T);

            % Verify that T * T_inv = identity transformation matrix
            identity_matrix = T * T_inv;
            identity_expected = eye(4);
            testCase.verifyLessThan(abs(identity_matrix(:) - identity_expected(:)), 1e-10);
        end
        
        %% Test the analytic Jacobian by computing the numerical Jacobian
        % and comparing the Frobenius norm
        
        function testS3GroupRightJacobian(testCase)
            % Test the right Jacobian for S^3 group
            disp('Testing S3Group right Jacobian...');
            
            % Create a random vector in the Lie algebra su(2)
            u = randn(3, 1);

            % Compute the analytical right Jacobian
            Jr_analytical = S3Group.rightJacobian(u);

            % Compute the numerical right Jacobian using finite differences
            epsilon = 1e-6;
            Jr_numerical = zeros(3, 3);
            for i = 1:3
                du = zeros(3, 1);
                du(i) = epsilon;
                exp_u_plus_du = S3Group.expMap(u + du);
                exp_u = S3Group.expMap(u);
                log_diff = S3Group.logMap(S3Group.quatmultiply(S3Group.inverse(exp_u), (exp_u_plus_du)));
                Jr_numerical(:, i) = log_diff / epsilon;
            end

            % Verify that the analytical and numerical Jacobians are close
            testCase.verifyLessThan(norm(Jr_analytical - Jr_numerical, 'fro'), 1e-6);
        end

        function testSO3GroupRightJacobian(testCase)
            % Test the right Jacobian for SO(3) group
            disp('Testing SO3Group right Jacobian...');

            % Create a random vector in the Lie algebra so(3)
            u = randn(3, 1);

            % Compute the analytical right Jacobian
            Jr_analytical = SO3Group.rightJacobian(u);

            % Compute the numerical right Jacobian using finite differences
            epsilon = 1e-6;
            Jr_numerical = zeros(3, 3);
            for i = 1:3
                du = zeros(3, 1);
                du(i) = epsilon;
                exp_u_plus_du = SO3Group.expMap(u + du);
                exp_u = SO3Group.expMap(u);
                log_diff = SO3Group.logMap(SO3Group.inverse(exp_u) * (exp_u_plus_du));
                Jr_numerical(:, i) = log_diff / epsilon;
            end

            % Verify that the analytical and numerical Jacobians are close
            testCase.verifyLessThan(norm(Jr_analytical - Jr_numerical, 'fro'), 1e-6);
        end

        function testSE3GroupRightJacobian(testCase)
            % Test the right Jacobian for SE(3) group
            disp('Testing SE3Group right Jacobian...');

            % Create a random vector in the Lie algebra se(3)
            u = randn(6, 1);

            % Compute the analytical right Jacobian
            Jr_analytical = SE3Group.rightJacobian(u);

            % Compute the numerical right Jacobian using finite differences
            epsilon = 1e-6;
            Jr_numerical = zeros(6, 6);
            for i = 1:6
                du = zeros(6, 1);
                du(i) = epsilon;
                exp_u_plus_du = SE3Group.expMap(u + du);
                exp_u = SE3Group.expMap(u);
                log_diff = SE3Group.logMap(SE3Group.inverse(exp_u) * (exp_u_plus_du));
                Jr_numerical(:, i) = log_diff / epsilon;
            end

            % Verify that the analytical and numerical Jacobians are close
            testCase.verifyLessThan(norm(Jr_analytical - Jr_numerical, 'fro'), 1e-6);
        end
        
        %% Test Orientation in SE(3) and SO(3) Integration with exponential update
        function testOrientationIntegration(testCase)
            N = 20;
            xi_se3 = zeros(6,N) + [1:20];
            xi_so3 = xi_se3(4:6,:);
            
            h0 = S3Group.expMap([0 0 0]);
            R0 = eye(3);
            T0 = eye(4);
            for i = 1:N
                h0 = S3Group.quatmultiply(h0,S3Group.expMap(xi_so3(:,i)));
                R0 = R0*SO3Group.expMap(xi_so3(:,i));
                T0 = T0*SE3Group.expMap(xi_se3(:,i));
            end
            h0inSO3 = S3Group.quaternionToRotationMatrix(h0);
            
            errSE3_SO3 = norm((T0(1:3,1:3)' * R0) - eye(3));
            errS3_SO3 = norm(h0inSO3' * R0 - eye(3));
            testCase.verifyLessThan(errSE3_SO3, 1e-6);
            testCase.verifyLessThan(errS3_SO3, 1e-6);
        end
        
    end
end
