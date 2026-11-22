classdef TestRodSimulation < TestSuperClass
    % Tests of the Cosserat rod simulation (SimulationSetup) with the
    % parameters of the paper:
    % - analytical vs. numerical Jacobians of the ODE (with and without
    %   inputs and gravity), the gravitational and pressure loads and the
    %   implicit (static) model, in the reference configuration and in a
    %   deflected, moving state
    % - distributed loads vanish without gravity and inputs
    % - strains and end-effector pose of the reference configuration
    % - consistency of the explicit and the implicit model export
    %
    % Run from the repository root: runtests('tests')
    %
    % Copyright(C)2024, Institut fuer Mechatronische Systeme, Leibniz
    % Universitaet Hannover
    % Author: Max Bartholdt (max.bartholdt[at]imes.uni-hannover.de)
    %
    properties
        su % simulation setup
        errortolerance = 1e-9;
        u_test = [1.4, 1.31, 1.1]*1e5; % absolute pressures in Pa
    end

    properties (TestParameter)
        state = {'reference', 'deflected'};
    end

    methods (TestMethodSetup)
        function createSimulation(testCase)
            %% Arrange: create simulation setup for every test
            TOTAL_SIM_TIME = 1; % in seconds
            testCase.su = testCase.paperSimulation(TOTAL_SIM_TIME);
            testCase.su.setInputConstant(testCase.u_test);
        end
    end

    methods
        function x = getState(testCase, state)
            if strcmp(state, 'reference')
                x = testCase.su.x0;
            else
                x = testCase.testState(testCase.su);
            end
        end
    end

    %% Test Functions
    methods (Test)
        %% Total gradient of the ODE (num vs. ana)
        function testODEGradient(testCase, state)
            x = testCase.getState(state);
            j_ana = testCase.su.j_f(0, x);
            j_num = testCase.numJacobian(@(t,x)testCase.su.f(t,x), x);
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testODEGradientWithoutInputs(testCase, state)
            testCase.su.setInputsZero;
            x = testCase.getState(state);
            j_ana = testCase.su.j_f(0, x);
            j_num = testCase.numJacobian(@(t,x)testCase.su.f(t,x), x);
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testODEGradientWithoutGravity(testCase, state)
            testCase.su.setGravity(zeros(3,1));
            x = testCase.getState(state);
            j_ana = testCase.su.j_f(0, x);
            j_num = testCase.numJacobian(@(t,x)testCase.su.f(t,x), x);
            testCase.verifyJacobian(j_ana, j_num);
        end

        %% Gradients of the distributed loads and the implicit model
        function testGravityGradient(testCase, state)
            x = testCase.getState(state);
            j_ana = testCase.su.j_ug_eta(0, x) * testCase.su.j_eta_x(0, x);
            j_num = testCase.numJacobian(@(t,x)testCase.su.u_g(t,x), x);
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testPressureGradient(testCase, state)
            x = testCase.getState(state);
            j_ana = testCase.su.j_up_x(0, x);
            j_num = testCase.numJacobian(@(t,x)testCase.su.u_p(t,x), x);
            testCase.verifyJacobian(j_ana, j_num);
        end

        function testStaticsGradient(testCase, state)
            x = testCase.getState(state);
            xt = zeros(size(x));
            j_ana = testCase.su.j_c_x(0, x, xt);
            j_num = testCase.numJacobian(@(t,x)testCase.su.c(t,x,xt), x);
            testCase.verifyJacobian(j_ana, j_num);
        end

        %% Distributed loads
        function ugIsZero(testCase)
            testCase.su.setGravity(zeros(3,1));
            ug = testCase.su.u_g(0, testCase.testState(testCase.su));
            testCase.verifyLessThan(norm(ug), testCase.errortolerance)
        end

        function uIsZero(testCase)
            testCase.su.setInputsZero;
            u = testCase.su.u(0, testCase.testState(testCase.su));
            testCase.verifyLessThan(norm(u), testCase.errortolerance)
        end

        %% Strains and kinematics of the reference configuration
        function testStrains(testCase)
            testCase.su.setInputsZero;
            eta_ = testCase.su.eta(0, testCase.su.x0);
            eta_expected = zeros(size(eta_));
            eta_expected(3,:) = 1; % straight, unstretched rod
            testCase.verifyLessThan(norm(eta_(:) - eta_expected(:)), testCase.errortolerance);
        end

        function testSE3Integration(testCase)
            testCase.su.setInputsZero;
            EE = testCase.su.EE(0, testCase.su.x0);
            % base pose, straight rod of length L, offset of the measured end effector
            EE_expected = SE3Group.expMap(testCase.su.xi_base) * ...
                [eye(3), [0; 0; testCase.su.L]; 0 0 0 1] * ...
                [eye(3), testCase.su.EE_pos_offset; 0 0 0 1];
            err = SE3Group.logMap(SE3Group.inverse(EE)*EE_expected);
            testCase.verifyLessThan(norm(err), testCase.errortolerance);
        end

        %% Consistency of the model exports
        function testStateEquation(testCase)
            % time-invariant state equation equals the simulation ODE
            x = testCase.testState(testCase.su);
            dx_sim = testCase.su.f(0, x);
            dx_ssm = testCase.su.stateEquation(x, testCase.u_test);
            testCase.verifyLessThan(norm(dx_sim - dx_ssm)/norm(dx_sim), testCase.errortolerance);
        end

        function testStateEquationWithImplicitModel(testCase)
            % the explicit state equation solves the implicit model c = 0
            % along a simulated trajectory
            y = testCase.su.run([0 0.05 0.1]);
            err = zeros(size(y,1),1);
            for i = 1:size(y,1)
                x = y(i,:)';
                dx = testCase.su.f(0, x);
                err(i) = norm(testCase.su.c(0, x, dx));
            end
            testCase.verifyLessThan(max(err), 1e-6);
        end
    end
end
