%% Exp. 2: model validation and PINN verification
% Predicts the measured test bench trajectory (data/exp2_measurement.csv)
% with the DD-PINN and the Cosserat rod simulation and compares both with
% the measurement (T-RO Fig. 6, Fig. 7 and Table III).
%
% Run from the repository root after setup.

clear all
close all
cdat = model_config();
root = fileparts(fileparts(mfilename('fullpath')));

%% Initiate params and PINNModel class
t_total = 6;
pinn_folder = fullfile(root,'pinn');
rod_pinn = PINNModel(pinn_folder,t_total);
rod_pinn.setRodParams(fullfile(root,'parameters','rod_parameters.json'));

%% Set inputs
plot_data_flag = 1;
input_data_file = fullfile(root,'data','exp2_measurement.csv');
rod_pinn.setInputDataFromFile(input_data_file);

%% Predict and calculate outputs
predStates = rod_pinn.predictTraj();
[r_EE,orientations] = rod_pinn.calcEE();
forces_torques = rod_pinn.calcForceTorque();

rod_pinn.runSimulation(cdat);

% Calculate errors
err_pinn_sim.position = mean(sqrt(sum((r_EE - rod_pinn.sim_res.rEE).^2,2)));
err_pinn_sim.rotation = mean(sqrt(sum((S3Group.logMap(rotm2quat(orientations))-S3Group.logMap(rotm2quat(rod_pinn.sim_res.orientations))).^2,2)));
err_pinn_sim.force = mean(sqrt(sum((forces_torques(:,1:3) - rod_pinn.sim_res.FTS(:,1:3)).^2,2)));
err_pinn_sim.torque = mean(sqrt(sum((forces_torques(:,4:6) - rod_pinn.sim_res.FTS(:,4:6)).^2,2)));

err_pinn_sim_std.position = std(sqrt(sum((r_EE - rod_pinn.sim_res.rEE).^2,2)));
err_pinn_sim_std.rotation = std(sqrt(sum((S3Group.logMap(rotm2quat(orientations))-S3Group.logMap(rotm2quat(rod_pinn.sim_res.orientations))).^2,2)));
err_pinn_sim_std.force = std(sqrt(sum((forces_torques(:,1:3) - rod_pinn.sim_res.FTS(:,1:3)).^2,2)));
err_pinn_sim_std.torque = std(sqrt(sum((forces_torques(:,4:6) - rod_pinn.sim_res.FTS(:,4:6)).^2,2)));

data_standard_struct = load_measurement(input_data_file);
subsampling_step = rod_pinn.t_sample/0.001; % measurement sampled at 1 kHz
start_idx  = find(1e-6>abs(data_standard_struct.time-0));
end_idx  = find(1e-6>abs(data_standard_struct.time-t_total));
t_new = 0:rod_pinn.t_sample:t_total;
xEE_smooth = zeros(size(data_standard_struct.x_is));
for i = 1:size(data_standard_struct.x_is,2)
    xEE_smooth(:,i) = smooth_moving(data_standard_struct.x_is(:,i),subsampling_step);
end
positionData = interp1(data_standard_struct.time(start_idx:end_idx),xEE_smooth(start_idx:end_idx,:),t_new);
force_torque_smooth = zeros(size(data_standard_struct.force_torque));
for i = 1:size(data_standard_struct.force_torque,2)
    force_torque_smooth(:,i) = smooth_moving(data_standard_struct.force_torque(:,i),subsampling_step);
end
forceData = interp1(data_standard_struct.time(start_idx:end_idx),force_torque_smooth(start_idx:end_idx,1:3),t_new);
torqueData = interp1(data_standard_struct.time(start_idx:end_idx),force_torque_smooth(start_idx:end_idx,4:6),t_new);
h_smooth = smoothdata(S3Group.logMap(data_standard_struct.h_is),1,'movmean',subsampling_step);
h_new = interp1(data_standard_struct.time(start_idx:end_idx),h_smooth(start_idx:end_idx,:),t_new);
orientationData = S3Group.expMap(h_new);

err_sim_meas.position = mean(sqrt(sum((positionData - rod_pinn.sim_res.rEE).^2,2)));
err_sim_meas.rotation = mean(sqrt(sum((S3Group.logMap(orientationData)-S3Group.logMap(rotm2quat(rod_pinn.sim_res.orientations))).^2,2)));
err_sim_meas.force = mean(sqrt(sum((forceData - rod_pinn.sim_res.FTS(:,1:3)).^2,2)));
err_sim_meas.torque = mean(sqrt(sum((torqueData - rod_pinn.sim_res.FTS(:,4:6)).^2,2)));

err_sim_meas_std.position = std(sqrt(sum((positionData - rod_pinn.sim_res.rEE).^2,2)));
err_sim_meas_std.rotation = std(sqrt(sum((S3Group.logMap(orientationData)-S3Group.logMap(rotm2quat(rod_pinn.sim_res.orientations))).^2,2)));
err_sim_meas_std.force = std(sqrt(sum((forceData - rod_pinn.sim_res.FTS(:,1:3)).^2,2)));
err_sim_meas_std.torque = std(sqrt(sum((torqueData - rod_pinn.sim_res.FTS(:,4:6)).^2,2)));

%% Table III (position in mm, rotation in rad, force in N, torque in Nmm)
unit_scale = [1000 1 1 1000];
quantities = {'position','rotation','force','torque'};
errors = zeros(4,4);
for i = 1:4
    errors(:,i) = unit_scale(i)*[err_sim_meas.(quantities{i}); err_sim_meas_std.(quantities{i}); ...
        err_pinn_sim.(quantities{i}); err_pinn_sim_std.(quantities{i})];
end
disp(array2table(errors,'VariableNames',{'r_EE (mm)','theta_EE (rad)','f_react (N)','m_react (Nmm)'}, ...
    'RowNames',{'Sim-meas mean','Sim-meas SD','PINN-sim mean','PINN-sim SD'}))

%% Plots
results_dir = fullfile(root,'results');
if ~exist(results_dir,'dir'), mkdir(results_dir); end

fig_1 = figure("Position",[50 50 600 900]);
tl_1 = tiledlayout(5,1,"TileSpacing","compact","Padding","compact");
set_color_order_imes_default;
ax = nexttile;
rod_pinn.plotPressures(ax)
xlabel(ax,'')
xticklabels(ax,{})
set_color_order_rgb_default;
ax = nexttile;
rod_pinn.plotEE(plot_data_flag,ax,2)
xlabel(ax,'')
xticklabels(ax,{})
ax = nexttile;
rod_pinn.plotOrientation(plot_data_flag,ax)
legend(ax,'off');
xlabel(ax,'')
xticklabels(ax,{})
ax(1) = nexttile(4);
ax(2) = nexttile(5);
rod_pinn.plotFTS(plot_data_flag,ax)
legend(ax(1),'off');
xlabel(ax(1),'')
xticklabels(ax(1),{})
legend(ax(2),'off');
exportgraphics(fig_1,fullfile(results_dir,'exp2_pinn_prediction.pdf'),'ContentType','vector')

fig_states = figure("Position",[100 100 1300 700]);
tiledlayout(1,1,"TileSpacing","compact","Padding","tight");
nexttile
ax_states = rod_pinn.plotStates(fig_states);
exportgraphics(fig_states,fullfile(results_dir,'exp2_pinn_verification.pdf'),'ContentType','vector')
