%% Exp. 4: plot of the NEMPC results in simulation (T-RO Fig. 11)
% Loads results/exp4_tracking.mat and results/exp4_setpoint.mat, written
% by exp4_nempc_simulation.m, and computes the end-effector and shape
% errors (discrete Frechet distance between reference and controlled shape,
% each consisting of the rod nodes and the measured end effector).
%
% Run from the repository root after setup.

clear all
close all
root = fileparts(fileparts(mfilename('fullpath')));
results_dir = fullfile(root,'results');
plot_data_flag = 0;
xlimits = [2 5];

fig_1 = figure("Position",[100 100 650 850]);
tl_1 = tiledlayout(4,1,"TileSpacing","compact","Padding","compact");

%% a) trajectory tracking
load(fullfile(results_dir,'exp4_tracking.mat'),'pinn_mpc');
err_traj_pos = mean(sqrt(sum((pinn_mpc.rEE - pinn_mpc.sim_res.rEE).^2,2)));
frechet_traj = shape_errors(pinn_mpc);
fprintf('Exp. 4a: end-effector error %.2f mm, shape error (Frechet) mean %.2f mm, max %.2f mm\n', ...
    1000*err_traj_pos, 1000*mean(frechet_traj), 1000*max(frechet_traj))

ax = nexttile;
title(ax,'a) Dynamic trajectory tracking')
v = [xlimits(1) 0; xlimits(1) 80; xlimits(2) 80; xlimits(2) 0];
f = [1 2 3 4];
xlim(ax,[0 5])
patch('Faces',f,'Vertices',v,'FaceColor','b','FaceAlpha',.07,'LineStyle','none');
hold(ax,"on")
ax = pinn_mpc.plotPressures(ax,1);
ylim(ax,[0 80])
xlabel(ax,'')
xticklabels(ax,{})
set_color_order_rgb_default;
ax = nexttile;
v2 = [xlimits(1) -0.05; xlimits(1) 0.05; xlimits(2) 0.05; xlimits(2) -0.05];
patch('Faces',f,'Vertices',v2,'FaceColor','b','FaceAlpha',.07,'LineStyle','none');
hold(ax,"on")
pinn_mpc.plotEE(plot_data_flag,ax,2)
xlim(ax,[0 5])
xlabel(ax,'')
xticklabels(ax,{})

fig_states = figure("Position",[100 100 1300 700]);
tiledlayout(1,1,"TileSpacing","compact","Padding","tight");
nexttile
pinn_mpc.plotStates(fig_states,xlimits);
exportgraphics(fig_states,fullfile(results_dir,'exp4_states.pdf'),'ContentType','vector')

%% b) setpoint control
figure(fig_1)
load(fullfile(results_dir,'exp4_setpoint.mat'),'pinn_mpc');
err_set_pos = sqrt(sum((pinn_mpc.rEE(end,:) - pinn_mpc.sim_res.rEE(end,:)).^2,2));
frechet_setp = shape_errors(pinn_mpc);
fprintf('Exp. 4b: final end-effector error %.2f mm, final shape error (Frechet) %.2f mm\n', ...
    1000*err_set_pos, 1000*frechet_setp(end))

set_color_order_imes_default;
ax = nexttile(tl_1,3);
title(ax,'b) Setpoint control')
ax = pinn_mpc.plotPressures(ax,0);
xlim(ax,[0 5])
ylim(ax,[0 80])
xlabel(ax,'')
xticklabels(ax,{})
set_color_order_rgb_default;
ax = nexttile(tl_1,4);
pinn_mpc.plotEE(plot_data_flag,ax,0)
xlim(ax,[0 5])
ylim(ax,[-0.05,0.05])
set_color_order_imes_default;
exportgraphics(fig_1,fullfile(results_dir,'exp4_pos_pres.pdf'),'ContentType','vector')

function frechet = shape_errors(pinn_mpc)
% Frechet distance between the reference and the controlled shape over time
T_base = [eye(3),pinn_mpc.r0;0 0 0 1];
pinn_mpc.xi_base = SE3Group.logMap(T_base);
pinn_mpc.sim_res.setup.xi_base = SE3Group.logMap(T_base);
frechet = zeros(size(pinn_mpc.t,2),1);
for i = 1:size(pinn_mpc.t,2)
    controlled_shape = pinn_mpc.configuration(i);
    controlled_shape = squeeze(controlled_shape(1:3,4,:));
    ref_shape = pinn_mpc.sim_res.setup.configuration(pinn_mpc.t(i),pinn_mpc.sim_res.states(i,:));
    ref_shape = squeeze(ref_shape(1:3,4,:));
    frechet(i) = discrete_frechet_distance(ref_shape',controlled_shape');
end
end
