classdef TestSymbolicExports < TestSuperClass
    % Tests of the generated model functions (matlab/model/TRO_N_7):
    % every exported Jacobian is compared with the numerical Jacobian of the
    % exported function it belongs to, for random states, inputs and loads
    % and the parameters of the paper.
    %
    % Run from the repository root: runtests('tests')
    %
    % Copyright(C)2024, Institut fuer Mechatronische Systeme, Leibniz
    % Universitaet Hannover
    % Author: Max Bartholdt (max.bartholdt[at]imes.uni-hannover.de)
    %
    properties
        param % parameter array of the exported functions
        x0 % state
        xt0 % time derivative of the state (implicit model)
        w0 % actuation wrench
        u0 % distributed pressure loads
        ug0 % distributed gravitational loads
    end

    methods (TestMethodSetup)
        function createInputs(testCase)
            %% Arrange: random arguments of the exported functions
            su = testCase.paperSimulation(1);
            s = RandStream('mt19937ar', 'Seed', 1);
            testCase.param = su.param;
            testCase.x0 = testCase.testState(su);
            testCase.xt0 = randn(s, size(testCase.x0));
            testCase.w0 = randn(s, 6, 1);
            testCase.u0 = randn(s, numel(u_p(testCase.x0, testCase.w0, testCase.param)), 1);
            testCase.ug0 = randn(s, size(j_f_ug([], testCase.param), 2), 1);
        end
    end

    methods (Test)
        %% Explicit model f_
        function testJacobianFx(testCase)
            j_ana = j_f_x(testCase.x0, testCase.w0, [], [], testCase.param);
            j_num = testCase.numJacobian(@(t,x)f_(x, testCase.w0, testCase.u0, testCase.ug0, testCase.param), testCase.x0);
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testJacobianFu(testCase)
            j_ana = j_f_u([], [], [], testCase.param);
            j_num = testCase.numJacobian(@(t,u)f_(testCase.x0, testCase.w0, u, testCase.ug0, testCase.param), testCase.u0);
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testJacobianFug(testCase)
            j_ana = j_f_ug([], testCase.param);
            j_num = testCase.numJacobian(@(t,ug)f_(testCase.x0, testCase.w0, testCase.u0, ug, testCase.param), testCase.ug0);
            testCase.verifyJacobian(j_ana, j_num);
        end

        %% Implicit model c_
        function testJacobianCx(testCase)
            j_ana = j_c_x(testCase.x0, testCase.xt0, testCase.w0, [], [], testCase.param, []);
            j_num = testCase.numJacobian(@(t,x)c_(x, testCase.xt0, testCase.w0, testCase.u0, testCase.ug0, testCase.param), testCase.x0);
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testJacobianCu(testCase)
            j_ana = j_c_u(testCase.x0, testCase.xt0, testCase.w0, [], [], testCase.param, []);
            j_num = testCase.numJacobian(@(t,u)c_(testCase.x0, testCase.xt0, testCase.w0, u, testCase.ug0, testCase.param), testCase.u0);
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testJacobianCug(testCase)
            j_ana = j_c_ug(testCase.x0, testCase.xt0, testCase.w0, [], [], testCase.param, []);
            j_num = testCase.numJacobian(@(t,ug)c_(testCase.x0, testCase.xt0, testCase.w0, testCase.u0, ug, testCase.param), testCase.ug0);
            testCase.verifyJacobian(j_ana, j_num);
        end

        %% Compiled model functions (matlab/compile_mex)
        function testMexMatchesMFile(testCase)
            testCase.assumeTrue(endsWith(which('f_'), mexext) && endsWith(which('j_f_x'), mexext), ...
                'MEX functions not compiled (matlab/compile_mex).');
            args_f = {testCase.x0, testCase.w0, testCase.u0, testCase.ug0, testCase.param};
            args_j = {testCase.x0, testCase.w0, [], [], testCase.param};
            testCase.verifyEqual(f_(args_f{:}), testCase.callMFile('f_', args_f{:}), 'RelTol', 1e-12);
            testCase.verifyEqual(j_f_x(args_j{:}), testCase.callMFile('j_f_x', args_j{:}), 'AbsTol', 1e-12*norm(j_f_x(args_j{:}), 'fro'));
        end

        %% Strains and distributed pressure loads
        function testJacobianEtaPls(testCase)
            j_ana = j_eta_pls_x(testCase.param(1:6));
            j_num = testCase.numJacobian(@(t,x)eta_pls(x, testCase.w0, testCase.param), testCase.x0);
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testJacobianUp(testCase)
            j_ana = j_up_x(testCase.x0, testCase.w0, testCase.param);
            j_num = testCase.numJacobian(@(t,x)u_p(x, testCase.w0, testCase.param), testCase.x0);
            testCase.verifyJacobian(j_ana, j_num);
        end
    end
end
