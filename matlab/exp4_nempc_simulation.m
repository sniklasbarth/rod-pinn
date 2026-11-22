%% Exp. 4: NEMPC in simulation
% Closed loop of the NEMPC (Python, python/NEMPC_matlab_interface.py) and the
% UKF, both with the DD-PINN, controlling the Cosserat rod simulation. The
% plant input is filtered with the identified pressure model, the NEMPC
% predicts with a faster first-order model (parameters/pressure_models.json).
% Gaussian noise is added to the simulated end-effector position.
%   task = 'tracking': Exp. 4a, dynamic trajectory tracking (T-RO: 2.68 mm)
%   task = 'setpoint': Exp. 4b, setpoint control (T-RO: 2.82 mm)
% Each run takes several minutes (about 1 s per time step on a CPU). The
% NEMPC is stochastic, so the errors vary from run to run (4a: 2.6-2.8 mm).
%
% Run from the repository root after setup (Python environment needed).

clear all
close all
cdat = model_config();
root = fileparts(fileparts(mfilename('fullpath')));

task = 'tracking'; % 'tracking' (Exp. 4a) or 'setpoint' (Exp. 4b)
plotresults = 1;
debugging = 0;

%% Init MPC with PINN
t_sample = 1/70;
N_x = 5; % prediction horizon (the N = 8 in the T-RO text applies to the test bench experiments)
t_total = 5+(N_x-1)*t_sample;
N_samples = t_total/t_sample;

pinn_folder = fullfile(root,'pinn');
rod_param_file = fullfile(root,'parameters','rod_parameters.json');
pinn_mpc = PINNMPC(pinn_folder,t_sample,N_x,t_total+N_x*t_sample);
pinn_mpc.setRodParams(rod_param_file);

%% Calc reference
switch task
    case 'tracking'
        % reference states of a pressure trajectory with oscillations up to 1.5 Hz
        p_max = 0.8*10^5;
        base_freq = 0.15;
        [P_Mat,t_set] = smooth_input_complex(p_max,t_total+N_x*t_sample,t_sample,base_freq,0);
        u_ref = [P_Mat(:,1:3)]+1e5;
        t_set = 0:t_sample:(t_sample*(size(u_ref,1)-1));
        x_set = pinn_mpc.calcSetpoint(cdat,u_ref); % simulation with the parameters of the PINN
    case 'setpoint'
        % static setpoint of constant pressures (absolute, Pa)
        u_ref = [1.0,1.3,1.6]*10^5;
        x_set = pinn_mpc.calcSetpoint(cdat,u_ref)';
    otherwise
        error('task must be ''tracking'' or ''setpoint''')
end

%% Pressure models: plant input filter (identified) and NEMPC prediction model
pm = jsondecode(fileread(fullfile(root,'parameters','pressure_models.json')));
p_sys_d = c2d(tf(pm.identified.num',pm.identified.den'),t_sample);
p_num = p_sys_d.Numerator{1};
p_den = p_sys_d.Denominator{1};
pinn_mpc.use_pressure_model = true;

nempc_sys_d = c2d(tf(pm.nempc.num',pm.nempc.den'),t_sample);
NEMPC_params.p_num = py.numpy.array(nempc_sys_d.Numerator{1});
NEMPC_params.p_den = py.numpy.array(nempc_sys_d.Denominator{1});


%% NEMPC parameters
ne = jsondecode(fileread(fullfile(root,'parameters','nempc_parameters.json')));
NEMPC_path = fullfile(root,'python');
NEMPC_params.numSims = int16(ne.numSims);
NEMPC_params.horizon = int16(N_x);
NEMPC_params.R_factor = ne.R_factor;
NEMPC_params.parentQuota = ne.parentQuota;
NEMPC_params.strangerQuota = ne.strangerQuota;
NEMPC_params.numSegments = int16(ne.numSegments);
NEMPC_params.mutation_noise_factor = ne.mutation_noise_factor;
NEMPC_params.mutationNoise_init = ne.mutationNoise_init;
NEMPC_params.mutation_noise_min = ne.mutation_noise_min;
NEMPC_params.mutation_noise_max = ne.mutation_noise_max;
NEMPC_params.mutation_probability = ne.mutation_probability;
NEMPC_params.fix_first_knot = ne.fix_first_knot;
NEMPC_params.u_min_bar = ne.u_min_bar; % minimum absolute pressure in bar
NEMPC_params.u_max_bar = ne.u_max_bar; % maximum absolute pressure in bar
Q = pinn_mpc.calcQ(ne.Q_weights.vel,ne.Q_weights.angvel,ne.Q_weights.force,ne.Q_weights.forceZ,ne.Q_weights.moment);
NEMPC_params.Q = py.numpy.array(Q);

use_position_control = 0; % state cost
pinn_mpc.init_mpc(NEMPC_params,NEMPC_path,use_position_control)

%% Setup UKF
numParticle = 1000;  % irrelevant for UKF
t_total_ukf = 30; % irrelevant for step-wise eval
t_start = 0; % irrelevant for step-wise eval
used_measurements = ["position"];
pinn_ukf = PINNStateEst(pinn_folder,numParticle,t_sample,t_start,t_total_ukf,used_measurements);

ukf = jsondecode(fileread(fullfile(root,'parameters','ukf_parameters.json')));
pinn_ukf.setProcessNoise(ukf.sigma_velocity,ukf.sigma_angular_velocity,ukf.sigma_force,ukf.sigma_force_z,ukf.sigma_moment);
pinn_ukf.setInitialCovariance(ukf.sigma_velocity,ukf.sigma_angular_velocity,ukf.sigma_force,ukf.sigma_force_z,ukf.sigma_moment);
pinn_ukf.setProcessNoiseBendingComp(ukf.sigma_bending_compliance);
pinn_ukf.setMeasurementNoise(ukf.R_position,[],[],[]);
pinn_ukf.setUKFParams(ukf.alpha,ukf.beta,ukf.kappa);
pinn_ukf.setRodParams(rod_param_file);
pinn_ukf.ukf_init(1)

%% Run MPC + UKF
pre_steps = 0;
if isrow(x_set)
    x_set = repmat(x_set,N_samples+N_x,1);
    u_ref = repmat(u_ref,N_samples+N_x,1);
end
x_set_norm = normalize_list(x_set,pinn_mpc.props_names(1:pinn_mpc.xdim),pinn_mpc.m_x,pinn_mpc.b_x,false,false,true);
u_MPC = zeros(N_samples+1,pinn_mpc.dimU);
u_MPC(1,:) = 10^5*ones(1,pinn_mpc.dimU);
u_filt = zeros(N_samples+1,pinn_mpc.dimU);
u_filt(1,:) = 10^5*ones(1,pinn_mpc.dimU);
x_hat = zeros(N_samples+1,pinn_mpc.xdim+pinn_mpc.sdim);
x_hat(1,:) = pinn_mpc.X0;
x_sim = zeros(N_samples+1,pinn_mpc.xdim);
x_sim(1,:) = pinn_mpc.X0(1:pinn_mpc.xdim);
y_sim = randn(N_samples+1,pinn_ukf.mdim)*chol(pinn_ukf.R); % sensor noise

for k = 1:N_samples
    ep_start_t = tic;
    % optimize u using MPC
    xEE_meas = pinn_mpc.EE(x_hat(k,1:pinn_mpc.xdim),u_MPC(k,:),x_hat(k,pinn_mpc.xdim+1));
    if k == 1
        u_normalized = pinn_mpc.optimize_first(x_set_norm(k:k+N_x,:),xEE_meas,debugging); % first u optimization based on x0
    else
        x_normalized = normalize_list(x_hat(k,:),pinn_mpc.props_names(1:(pinn_mpc.xdim+pinn_mpc.sdim)),pinn_mpc.m_x,pinn_mpc.b_x,false,false,true);
        u_normalized = pinn_mpc.optimize(x_normalized(1:pinn_mpc.xdim),x_normalized(pinn_mpc.xdim+1:pinn_mpc.xdim+pinn_mpc.sdim),x_set_norm(k:k+N_x,:),u_norm_filt,xEE_meas,debugging); % optimize u based on x from UKF and w
    end
    % Rescale u
    u_MPC(k+1,:) = normalize_list(u_normalized,pinn_mpc.props_names(pinn_mpc.xdim+pinn_mpc.sdim+1:pinn_mpc.xdim+pinn_mpc.sdim+pinn_mpc.dimU),pinn_mpc.m_x,pinn_mpc.b_x,false,false,false);

    % simulate the plant (Cosserat rod model) with the filtered input
    u_filt(1:k+1,:) = filter(p_num,p_den,u_MPC(1:k+1,:),u_MPC(1,:));
    [x_sim(k+1,:),y_temp] = pinn_mpc.simStep([x_sim(k,:),x_hat(k,pinn_mpc.xdim+1:pinn_mpc.xdim+pinn_mpc.sdim)],u_filt(k+1,:),t_sample);
    y_sim(k+1,:) = y_sim(k+1,:) + y_temp;
    % estimate state x from measurements y using UKF
    u_norm_filt = normalize_list(u_filt(k+1,:),pinn_mpc.props_names(pinn_mpc.xdim+pinn_mpc.sdim+1:pinn_mpc.xdim+pinn_mpc.sdim+pinn_mpc.dimU),pinn_mpc.m_x,pinn_mpc.b_x,false,false,true);
    [x_hat(k+1,:),error] = pinn_ukf.ukf_step(y_sim(k+1,:),u_norm_filt,pre_steps+k);
    if error
        break
    end
    disp("Time step "+num2str(k)+" took "+num2str(toc(ep_start_t))+ " s.")
end

%% Errors
pinn_mpc.storeDataForPlotting(x_hat,u_filt,x_set,cdat)
x_hat_norm = normalize_list(x_hat,pinn_mpc.props_names(1:pinn_mpc.xdim),pinn_mpc.m_x,pinn_mpc.b_x,false,false,true);
x_set_norm = normalize_list(x_set,pinn_mpc.props_names(1:pinn_mpc.xdim),pinn_mpc.m_x,pinn_mpc.b_x,false,false,true);
state_rmse = mean(sqrt(sum((x_hat_norm(:,1:pinn_mpc.xdim) - x_set_norm(1:length(x_hat_norm),:)).^2,2)));
ee_rmse = mean(sqrt(sum((pinn_mpc.rEE - pinn_mpc.sim_res.rEE).^2,2)));
ee_final = sqrt(sum((pinn_mpc.rEE(end,:) - pinn_mpc.sim_res.rEE(end,:)).^2,2));
fprintf('End-effector position error: mean %.2f mm, final %.2f mm; mean normalized state error %.3f\n', ...
    1000*ee_rmse, 1000*ee_final, state_rmse)

results_dir = fullfile(root,'results');
if ~exist(results_dir,'dir'), mkdir(results_dir); end
save(fullfile(results_dir,['exp4_',task,'.mat']),"pinn_mpc","pinn_ukf","ee_rmse","ee_final","state_rmse", ...
    "x_sim","y_sim","x_set","u_ref","u_MPC","u_filt")

%% Plots
if plotresults
    pinn_mpc.plotStates()
    pinn_mpc.plotEE(0)
    pinn_mpc.plotEE3d(0)
    pinn_mpc.plotPressures()
end
