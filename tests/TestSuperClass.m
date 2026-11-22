classdef TestSuperClass < matlab.unittest.TestCase
    % Common helpers of the unit tests: numerical Jacobians (central
    % differences) and comparison of numerical and analytical Jacobians.
    %
    % Copyright(C)2024, Institut fuer Mechatronische Systeme, Leibniz
    % Universitaet Hannover
    % Author: Max Bartholdt (max.bartholdt[at]imes.uni-hannover.de)
    %
    properties
        tolerance_eigenvalue = 1e-3; % tolerance of equalJacobians
        tolerance_relative = 1e-5; % tolerance of relativeError (Frobenius norm)
        DEBUG = 0; % 1: plot the sparsity patterns of the compared Jacobians
    end

    methods (TestClassSetup)
        function addRepositoryPaths(testCase)
            % make the tests independent of the current folder and of setup.m
            root = fileparts(fileparts(mfilename('fullpath')));
            matlab_dir = fullfile(root, 'matlab');
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(matlab_dir));
            for d = ["classes", "functions", "external"]
                testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                    fullfile(matlab_dir, d), 'IncludingSubfolders', true));
            end
            cdat = model_config();
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(cdat.path));
        end
    end

    methods (Static)
        function su = paperSimulation(t_total)
            % simulation setup with the parameters of the paper (Table II)
            root = fileparts(fileparts(mfilename('fullpath')));
            rod = jsondecode(fileread(fullfile(root, 'parameters', 'rod_parameters.json')));
            su = SimulationSetup(model_config(), t_total);
            su.setGravity(rod.gravity');
            su.setReferenceLength(rod.ref_length);
            su.CM = diag(rod.C);
            su.DM = diag(rod.D);
            su.MM = diag(rod.M);
            su.M_tip = rod.M_tip;
            su.compressiongain = rod.compression_gain;
            su.updateParameters();
        end

        function x = testState(su)
            % deflected, moving state: reference configuration plus a
            % reproducible random perturbation
            s = RandStream('mt19937ar', 'Seed', 42);
            x = su.x0 + 0.05*randn(s, size(su.x0));
        end

        function J = numJacobian(testhandle, x0)
            % central differences of testhandle(t, x) w.r.t. x at x0
            epsilon = 1e-6;
            for i = 1:length(x0)
                dx0_1 = x0;
                dx0_2 = x0;
                dx0_1(i) = dx0_1(i) - epsilon;
                dx0_2(i) = dx0_2(i) + epsilon;
                y1 = testhandle(0, dx0_1);
                y2 = testhandle(0, dx0_2);
                J(:, i) = (y2(:) - y1(:))./(2*epsilon); %#ok<AGROW>
            end
        end

        function J = numJacobianSE3(testhandle, x0)
            % right Jacobian of an SE(3)-valued testhandle(t, x)
            epsilon = 1e-6;
            for i = 1:length(x0)
                dx0_1 = x0;
                dx0_2 = x0;
                dx0_1(i) = dx0_1(i) - epsilon;
                dx0_2(i) = dx0_2(i) + epsilon;
                y1 = testhandle(0, dx0_1);
                y2 = testhandle(0, dx0_2);
                J(:, i) = SE3Group.logMap(SE3Group.inverse(y1)*y2)./(2*epsilon); %#ok<AGROW>
            end
        end

        function out = equalJacobians(A, B)
            % Eigenvalues of A\B (square) or singular values (otherwise)
            % compared with those of identical matrices
            if diff(size(A)) == 0
                EV_temp = eig(A\B);
                out = norm(EV_temp - ones(size(EV_temp)));
            else
                out = norm(svd(A) - svd(B));
            end
        end

        function out = relativeError(A, B)
            % element-wise comparison: ||A - B||_F / ||A||_F
            out = norm(A - B, 'fro')/norm(A, 'fro');
        end

        function out = callMFile(name, varargin)
            % call the .m file of a function that is shadowed by its MEX file
            fun_dir = fileparts(which(name));
            tmp = tempname;
            mkdir(tmp);
            txt = fileread(fullfile(fun_dir, [name, '.m']));
            txt = regexprep(txt, ['\<', name, '\s*\('], [name, '_mfile('], 'once'); % rename the function
            fid = fopen(fullfile(tmp, [name, '_mfile.m']), 'w');
            fwrite(fid, txt);
            fclose(fid);
            addpath(tmp);
            out = feval([name, '_mfile'], varargin{:});
            rmpath(tmp);
            rmdir(tmp, 's');
        end

        function inspectMatrixStructure(j_ana, j_num)
            figure
            subplot(1, 2, 1)
            spy(j_ana)
            title('Analytic')
            subplot(1, 2, 2)
            spy(j_num)
            title('Numeric')
        end
    end

    methods
        function verifyJacobian(testCase, j_ana, j_num)
            % assert that the analytical Jacobian matches the numerical one
            if testCase.DEBUG
                testCase.inspectMatrixStructure(j_ana, j_num)
            end
            testCase.verifySize(j_ana, size(j_num));
            testCase.verifyLessThan(testCase.relativeError(j_num, j_ana), testCase.tolerance_relative);
            if diff(size(j_ana)) ~= 0 || rcond(j_ana) > eps
                testCase.verifyLessThan(testCase.equalJacobians(j_ana, j_num), testCase.tolerance_eigenvalue);
            end
        end
    end
end
