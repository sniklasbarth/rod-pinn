classdef TestPINN < TestSuperClass
    % Tests of the trained DD-PINN (pinn/): the plain-text weights load
    % consistently and the PINN prediction of the measured input trajectory
    % of Exp. 2 matches the Cosserat rod simulation (paper: 0.55 mm mean
    % end-effector deviation over 6 s).
    %
    % Run from the repository root: runtests('tests')
    %
    % Copyright(C)2025, Institut fuer Mechatronische Systeme, Leibniz
    % Universitaet Hannover
    % Author: Max Bartholdt (max.bartholdt[at]imes.uni-hannover.de)
    %
    properties
        root
        pinn
        cdat
    end

    methods (TestMethodSetup)
        function createPINN(testCase)
            %% Arrange: PINN with the paper parameters and the Exp. 2 input
            testCase.root = fileparts(fileparts(mfilename('fullpath')));
            testCase.cdat = model_config();
            TOTAL_SIM_TIME = 1; % in seconds
            testCase.pinn = PINNModel(fullfile(testCase.root, 'pinn'), TOTAL_SIM_TIME);
            testCase.pinn.setRodParams(fullfile(testCase.root, 'parameters', 'rod_parameters.json'));
            testCase.pinn.setInputDataFromFile(fullfile(testCase.root, 'data', 'exp2_measurement.csv'));
        end
    end

    methods (Test)
        function testPINNDimensions(testCase)
            % 72 states of the model and one supplementary state (bending compliance)
            testCase.verifyEqual(testCase.pinn.xdim, 12*(testCase.cdat.N-1));
            testCase.verifyEqual(testCase.pinn.N, testCase.cdat.N);
            testCase.verifyEqual(testCase.pinn.sdim, 1);
        end

        function testBatchOutputsMatchExport(testCase)
            % batch versions of eta_pls and out_ft_sensor in PINNModel vs.
            % the generated export, with C_4 = C_5 from the supplementary state
            s = RandStream('mt19937ar', 'Seed', 1);
            B = 5;
            X = 0.3*randn(s, testCase.pinn.xdim, B);
            w = randn(s, 6, B);
            c_bend = 30 + 20*rand(s, B, 1);
            p = testCase.pinn.param;
            eta_b = testCase.pinn.eta_pls(X, w, p, c_bend);
            fts_b = testCase.pinn.out_ft_sensor(X, w, p, c_bend);
            for b = 1:B
                p_b = [p(1:3), c_bend(b)*[1 1], p(6:end)];
                testCase.verifyEqual(eta_b(:,:,b), eta_pls(X(:,b), w(:,b), p_b), 'RelTol', 1e-12);
                testCase.verifyEqual(fts_b(:,b), out_ft_sensor(X(:,b), w(:,b), p_b), 'RelTol', 1e-12);
            end
        end

        function testPINNtrajectory(testCase)
            % PINN prediction vs. simulation of the end-effector position
            testCase.pinn.predictTraj();
            r_EE = testCase.pinn.calcEE();
            testCase.pinn.runSimulation(testCase.cdat);
            err = mean(sqrt(sum((r_EE - testCase.pinn.sim_res.rEE).^2, 2)));
            testCase.verifyLessThan(err, 1e-3); % in m
        end
    end
end
