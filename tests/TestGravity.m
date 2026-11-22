classdef TestGravity < TestSuperClass
    % Tests of the distributed gravitational load (gravity) and its
    % Jacobian w.r.t. the strains (gravity_derivative), which is derived
    % with group theory instead of symbolically. Uses the MEX functions if
    % they have been compiled (matlab/compile_mex) and compares them with
    % the .m files.
    %
    % Run from the repository root: runtests('tests')
    %
    % Copyright(C)2024, Institut fuer Mechatronische Systeme, Leibniz
    % Universitaet Hannover
    % Author: Max Bartholdt (max.bartholdt[at]imes.uni-hannover.de)
    %
    properties
        su
        xi % strains on all sub-nodes (6 x sub-nodes)
    end

    methods (TestMethodSetup)
        function createStrains(testCase)
            %% Arrange: strains of a deflected rod on the sub-nodes
            testCase.su = testCase.paperSimulation(1);
            testCase.su.setInputConstant([1.4, 1.31, 1.1]*1e5);
            testCase.xi = testCase.su.eta(0, testCase.testState(testCase.su));
        end
    end

    methods (Test)
        function testGravityDerivative(testCase)
            % analytical vs. numerical Jacobian w.r.t. the strains
            p = testCase.su.param_gravity;
            N = testCase.su.N;
            j_ana = gravity_derivative(testCase.xi, p, N);
            j_num = testCase.numJacobian(@(t,xi)gravity(reshape(xi, 6, []), p, N), testCase.xi(:));
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testGravityDerivativeTipMass(testCase)
            % same without tip mass (M_tip enters only the last elements)
            p = testCase.su.param_gravity;
            p(6) = 0;
            N = testCase.su.N;
            j_ana = gravity_derivative(testCase.xi, p, N);
            j_num = testCase.numJacobian(@(t,xi)gravity(reshape(xi, 6, []), p, N), testCase.xi(:));
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testStraightRodLoad(testCase)
            % straight vertical rod: all loads point along -M*g, no moments
            p = testCase.su.param_gravity;
            xi_straight = zeros(size(testCase.xi));
            xi_straight(3,:) = 1;
            ug = reshape(gravity(xi_straight, p, testCase.su.N), 6, []);
            testCase.verifyLessThan(max(abs(ug(4:6,:)), [], 'all'), 1e-12);
            testCase.verifyLessThan(max(abs(ug(1:2,:)), [], 'all'), 1e-12);
            testCase.verifyEqual(ug(3,1), -p(5)*p(3), 'RelTol', 1e-12);
            testCase.verifyEqual(ug(3,end), -(p(5) + p(6))*p(3), 'RelTol', 1e-12);
        end

        function testMexMatchesMFile(testCase)
            % compiled functions (if present) give the results of the .m files
            testCase.assumeTrue(endsWith(which('gravity'), mexext), 'MEX functions not compiled (matlab/compile_mex).');
            p = testCase.su.param_gravity;
            N = testCase.su.N;
            ug_m = testCase.callMFile('gravity', testCase.xi, p, N);
            jug_m = testCase.callMFile('gravity_derivative', testCase.xi, p, N);
            testCase.verifyEqual(gravity(testCase.xi, p, N), ug_m, 'AbsTol', 1e-12);
            testCase.verifyEqual(gravity_derivative(testCase.xi, p, N), jug_m, 'AbsTol', 1e-12);
        end
    end
end
