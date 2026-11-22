%% Step 1: generate the symbolic Cosserat rod model
% Derives the discretized Cosserat rod model (local frame formulation,
% implicit midpoint collocation, N = 7 nodes, gravity) and exports all
% functions with matlabFunction to model/TRO_N_7. The model is the one the
% PINN loss was generated from (IMES local_frame_formulation, commit 3ef4dd0);
% python/matlab_to_torch.py translates its c_, eta_pls and u_p to PyTorch.
% The export is part of the repository, so this step is only needed to
% reproduce it. It needs the Symbolic Math Toolbox (about 2 min with R2024b).
% Other MATLAB releases than R2020b/R2024b name the temporary variables
% differently, so the files are equal numerically, not textually.
%
% Run from the repository root after setup.

clear all
close all
cdat = model_config();
rmpath(cdat.path) % the functions are regenerated
if ~exist(cdat.path,'dir'), mkdir(cdat.path); end

%% fluidic actuation
fluid_act = SymFluidicActuation(3,1,cdat);
fluid_act.export();

%% symbolic discretisation of Cosserat rod pde
sym_cosserat = SymCosseratRod(cdat);
sym_cosserat.exportall

%% mass and compliance matrices
T03_symbolic_parameters
matlabFunction(computeMass,...
    'Vars',symvar(computeMass) ,...
    'Output',{'mass_matrix'}, ...
    'File',[cdat.path,filesep,'computeMass']);
matlabFunction(computeCompliance,...
    'Vars',symvar(computeCompliance) ,...
    'Output',{'compliance_matrix'}, ...
    'File',[cdat.path,filesep,'computeCompliance']);

createParameterDisplay(sym_cosserat.param,[cdat.path,filesep,'displayParameters.m'])
addpath(cdat.path)

function createParameterDisplay(param, filename)
    % Open the file for writing
    fileID = fopen(filename, 'w');
    if fileID == -1
        error('Cannot open file: %s', filename);
    end
    %%Header
    fprintf(fileID, 'function displayParameters(values)\n');

    % Begin the paramLabels array in the file
    fprintf(fileID, 'paramLabels = {\n');

    % Iterate through each symbolic variable and write the name to the file
    for i = 1:length(param)
        varName = char(param(i)); % Convert symbolic variable to string
        if i == length(param)
            fprintf(fileID, '''%s''\n', varName); % Last element without ellipsis
        else
            fprintf(fileID, '''%s'', ...\n', varName); % With ellipsis
        end
    end

    % End the paramLabels array
    fprintf(fileID, '};\n');
    fprintf(fileID,'if isrow(values), values = values'';,end\n')
    fprintf(fileID, ...
    ['T = table(paramLabels'', values, ''VariableNames'', {''Parameter'', ''Value''});\n']);
    fprintf(fileID, ...
    ['disp(T)\n end']);
    % Close the file
    fclose(fileID);
end
