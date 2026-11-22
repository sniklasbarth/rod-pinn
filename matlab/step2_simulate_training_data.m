%% Step 2: simulate the trajectory for the PINN training
% Simulates the Cosserat rod model with the identified parameters
% (parameters/rod_parameters.json) for the training pressure trajectory
% (three tanh pulses, 5 s, 70 Hz) and writes results/training_simulation.csv
% (time, absolute pressures in Pa, 72 states). The training takes the
% sampling ranges of states and inputs and the reference trajectory for its
% plots from this file by default. data/training_simulation.csv is the file
% used for the published PINN; this script reproduces it up to the solver
% tolerances (see README).
%
% Run from the repository root after setup.

clear all
close all
cdat = model_config();
root = fileparts(fileparts(mfilename('fullpath')));

TOTAL_SIM_TIME = 5; % in s
T_sample = 1/70;    % sampling time of the PINN in s

rod = jsondecode(fileread(fullfile(root,'parameters','rod_parameters.json')));
su = SimulationSetup(cdat,TOTAL_SIM_TIME);
su.setGravity(rod.gravity');
su.setReferenceLength(rod.ref_length);
su.CM = diag(rod.C);
su.DM = diag(rod.D);
su.MM = diag(rod.M);
su.M_tip = rod.M_tip;
su.compressiongain = rod.compression_gain;
su.updateParameters();

p = pressure_input2(1,0,TOTAL_SIM_TIME); % absolute pressure in Pa
t = (0:T_sample:TOTAL_SIM_TIME)';
pressure = cell2mat(arrayfun(p,t,'UniformOutput',false));
su.setInputData(t',pressure);

%% Run simulation
opts = odeset('Jacobian',@su.j_f,'NormControl','on');
disp('Starting simulation ...')
tic
[t,y] = ode23s(@su.f,t,su.x0,opts);
disp(['Computation time: ',num2str(toc),' s'])

%% Save
state_names = jsondecode(fileread(fullfile(root,'pinn','state_header.txt')));
if ~exist(fullfile(root,'results'),'dir'), mkdir(fullfile(root,'results')); end
out_file = fullfile(root,'results','training_simulation.csv');
writecell([{'t','u1','u2','u3'},state_names(1:size(y,2))'],out_file)
writematrix([t,pressure,y],out_file,'WriteMode','append')
fprintf('Saved %s\n',out_file)
