%% Exp. 3: estimation of the bending compliance in simulation
% Simulates the Cosserat rod model with a bending compliance of 45 for a
% dynamic pressure trajectory (100 s) and estimates states and bending
% compliance with the UKF on the DD-PINN, starting from the identified value
% 49.9 (T-RO Fig. 9).
%
% Run from the repository root after setup.

clear all
close all
cdat = model_config();
root = fileparts(fileparts(mfilename('fullpath')));

c_bend = 45; % bending compliance of the simulated ground truth

numParticle = 1000; % only for the particle filter
t_sample_se = 1/70; % sampling time of state estimation
t_total = 100;
t_start = 0;
used_measurements = ["position"];
pinn_ukf = PINNStateEst(fullfile(root,'pinn'),numParticle,t_sample_se,t_start,t_total,used_measurements);

%% UKF parameters
ukf = jsondecode(fileread(fullfile(root,'parameters','ukf_parameters.json')));
pinn_ukf.setProcessNoise(ukf.sigma_velocity,ukf.sigma_angular_velocity,ukf.sigma_force,ukf.sigma_force_z,ukf.sigma_moment);
pinn_ukf.setProcessNoiseBendingComp(ukf.sigma_bending_compliance);
pinn_ukf.setMeasurementNoise(ukf.R_position,[],[],[]);
pinn_ukf.setUKFParams(ukf.alpha,ukf.beta,ukf.kappa);
pinn_ukf.setInitialCovariance(ukf.sigma_velocity,ukf.sigma_angular_velocity,ukf.sigma_force,ukf.sigma_force_z,ukf.sigma_moment);

%% Simulate the measurements with c_bend
pinn_ukf.setRodParams(fullfile(root,'parameters','rod_parameters.json'));
c_bend_ident = pinn_ukf.C(4,4);
pinn_ukf.C = diag([diag(pinn_ukf.C(1:3,1:3))',c_bend,c_bend,pinn_ukf.C(6,6)]);
p_max = 0.9*10^5;
base_freq = 0.1;
validation_flag = 1;
[input_pressures,t_set] = smooth_input_complex(p_max,t_total,t_sample_se,base_freq,validation_flag);
input_pressures = input_pressures(:,1:3)+1e5;
pinn_ukf.setInputsUsingSim(cdat,input_pressures)
pinn_ukf.C = diag([diag(pinn_ukf.C(1:3,1:3))',c_bend_ident,c_bend_ident,pinn_ukf.C(6,6)]); % UKF starts from the identified value

%% Estimate
pinn_ukf.ukf(1);
t_avg_comp_ms = median(pinn_ukf.computation_time)*1000

%% Results
results_dir = fullfile(root,'results');
if ~exist(results_dir,'dir'), mkdir(results_dir); end
save(fullfile(results_dir,'exp3_ukf_parameter_estimation.mat'),"pinn_ukf")

c_bend_est = pinn_ukf.X_plus(:,pinn_ukf.xdim+pinn_ukf.sdim);
in_window = pinn_ukf.t >= 20;
fprintf('Estimated bending compliance (mean over [20, 100] s): %.2f (ground truth %g, relative error %.1f %%)\n', ...
    mean(c_bend_est(in_window)), c_bend, 100*abs(mean(c_bend_est(in_window))-c_bend)/c_bend)

fig_1 = figure("Position",[0 0 400 300]);
tiledlayout(1,1,"TileSpacing","compact","Padding","compact");
ax = nexttile;
pinn_ukf.plotSupplStates(0,ax)
hold(ax,'on')
yline(ax,c_bend,'r--','LineWidth',1.5)
ylim(ax,[43 52])
legend('UKF','Start value UKF','Ground truth (Simulation)')
exportgraphics(fig_1,fullfile(results_dir,'exp3_ukf_param_est.pdf'),'ContentType','vector')
