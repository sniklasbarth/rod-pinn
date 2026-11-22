classdef PINNStateEst<PINNModel
    %PINNUKF Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        P0
        Q
        R
        measData
        mdim
        np
        sigma_state
        sigma0_state
        sigma_bend_comp
        sigma_pressure
        X_plus
        Qxx_plus
        diverged
        t_sample_se
        t_start
        nSteps
        pos_flag = 0
        force_flag = 0
        torque_flag = 0
        orientation_flag = 0
        ang_vel_flag = 0
        accel_flag = 0
        pressure_flag = 0 % should pressure be included as a measurement with noise, otherwise assumed as an input without noise/covariance
        pdim = 0
        alpha
        beta
        kappa
        lambda
        orientationData
        positionData
        nan_counter = 0
        used_meas
        n
        W_m
        W_c
        disp_flag
        sim
        pred_duration = 0
    end
    
    methods
        function obj = PINNStateEst(importedPinnFile,numParticle,t_sample_se,t_start,t_total,meas_setup)
            %PINNUKF Construct an instance of this class
            obj@PINNModel(importedPinnFile,t_total);
            obj.t_start = t_start;
            obj.used_meas = meas_setup;
            if any(strcmp("position",meas_setup))
            obj.pos_flag = 1;
            end
            if any(strcmp("force",meas_setup))
            obj.force_flag = 1;
            end
            if any(strcmp("torque",meas_setup))
            obj.torque_flag = 1;
            end
            if any(strcmp("orientation",meas_setup))
            obj.orientation_flag = 1;
            end
            if any(strcmp("ang_vel",meas_setup))
                obj.ang_vel_flag = 1;
            end
            if any(strcmp("acceleration",meas_setup))
                obj.accel_flag = 1;
            end
            if any(strcmp("pressure",meas_setup))
                obj.pressure_flag = 1;
                obj.pdim = 3;
                obj.X0 = [obj.X0,10000*ones(1,obj.pdim)];
            end
            obj.mdim = size(meas_setup,1)*3;
            obj.np = numParticle;                   % Number of particles
            obj.t_sample_se = t_sample_se;
            obj.t_ratio = obj.t_sample_se/obj.t_sample;
            assert(floor(obj.t_ratio)==obj.t_ratio,"t_sample_se must be multiple of t_sample!")
            obj.nSteps = obj.t_total/obj.t_sample_se;

            % set standard values for Covariances            
            obj.sigma0_state.vel = 2*9.25169198383252e-07;
            obj.sigma0_state.angvel = 2*3.40026340836065e-05;
            obj.sigma0_state.force = 2*2.2402e-06;
            obj.sigma0_state.forceZ = 2*0.00101659802551008;
            obj.sigma0_state.moment = 2*3.14358611014490e-07;
            obj.sigma_bend_comp = 1e-1;
            obj.sigma_pressure = 10;
            updateP0Matrix(obj);                             % Covariance for the initial state        
            
            % Process noise covariance -> means of calculated vals below
            obj.sigma_state.vel = 2*9.25169198383252e-07;
            obj.sigma_state.angvel = 2*3.40026340836065e-05;
            obj.sigma_state.force = 2*2.2402e-06;
            obj.sigma_state.forceZ = 2*0.00101659802551008;
            obj.sigma_state.moment = 2*3.14358611014490e-07;
            obj.sigma_bend_comp = 1e-1;
            obj.sigma_pressure = 10;
            updateQMatrix(obj);
            % Values calculated using mean diff of the simulated states
            % obj.Q = diag(2*[1.82055184632961e-06	2.26290485649189e-06	0.000506621705372718	7.27487486450741e-08	5.21661793310367e-07	3.13063180205066e-08	1.54464581701991e-07	1.45003110035312e-09	2.40938271511076e-07	3.03881486670420e-06	4.18253489682867e-06	2.77773600952231e-06	7.87100902869344e-07	8.23728329018952e-07	0.000552178698778094	3.37064278307720e-07	7.78835278836388e-08	1.21754142044290e-08	3.40534380797106e-07	9.80110884344650e-08	5.62999444707962e-07	5.49772197373386e-06	2.81753369962497e-05	2.09214519306047e-06	1.03019474197433e-06	8.75022638789536e-07	0.000715505921692836	2.30103992937795e-07	9.08984024885193e-08	1.55721419446571e-10	8.60944941015309e-07	1.47058946011519e-07	7.17840177384252e-07	3.24704477646851e-06	2.05659374194451e-05	1.24117886049408e-06	3.34056585017498e-07	1.57809589118203e-06	0.000612713624633398	2.39485048293899e-07	8.23994368052099e-08	2.53189391101877e-09	1.10796607730753e-07	2.19809477026921e-07	7.65222919512327e-07	3.14352845583060e-06	6.58686205984267e-05	2.34959030871340e-07	1.97037334841133e-06	4.18028938727606e-07	0.000411613758599627	9.86004091985139e-08	4.22335865106119e-07	3.98915327722055e-09	1.40574813323152e-08	2.60436836458478e-07	9.87453208691372e-07	2.83943211851645e-05	3.99461068604522e-06	7.24954454645230e-07	4.26695453771503e-07	1.11435001140034e-06	0.000251160367453568	8.48854355780402e-08	5.20514434736045e-07	4.87625006846158e-10	1.12354490189530e-06	4.62050672318098e-07	1.25890881781967e-06	5.12352809584743e-05	7.86166447066006e-05	2.99233568389326e-06]);
            
            % R-Matrix for sensors (variance from measurement)
            R_vec = [];
            if obj.pos_flag
                R_vec = [R_vec,ones(1,3)*10^(-3)];
            end
            if obj.force_flag
                force_noise = 10*[0.000147870814712213,9.573e-05,0.0004195];
                R_vec = [R_vec,force_noise];
            end
            if obj.torque_flag
                torque_noise = 10*[6.308e-08,5.283e-08,2.585e-08];
                R_vec = [R_vec,torque_noise];
            end
            if obj.orientation_flag
                ortientation_noise = 10*[0.0004195,6.308e-08,5.283e-08,2.585e-08];
                R_vec = [R_vec,ortientation_noise];
            end
            if obj.ang_vel_flag
                ang_vel_noise = 10*[0.0004195,6.308e-08,5.283e-08,2.585e-08];
                R_vec = [R_vec,ang_vel_noise];
            end
            if obj.pressure_flag
                ang_vel_noise = 10*[0.0004195,6.308e-08,5.283e-08,2.585e-08];
                R_vec = [R_vec,ang_vel_noise];
            end
            obj.R = diag(R_vec);

            % set standard values for UKF parameters
            obj.alpha = 0.5;
            obj.beta = 2;
            obj.kappa = 0;
        end
        
        function obj = setProcessNoise(obj,sigma_vel,sigma_angvel,sigma_force,sigma_forceZ,sigma_moment)
            obj.sigma_state.vel = sigma_vel;
            obj.sigma_state.angvel = sigma_angvel;
            obj.sigma_state.force = sigma_force;
            obj.sigma_state.forceZ = sigma_forceZ;
            obj.sigma_state.moment = sigma_moment;
            updateQMatrix(obj);
        end
        function obj = setProcessNoiseBendingComp(obj,sigma_bend_comp)
            obj.sigma_bend_comp = sigma_bend_comp;
            updateQMatrix(obj);
        end
        function obj = setProcessNoisePressure(obj,sigma_pressure)
            obj.sigma_pressure = sigma_pressure;
            updateQMatrix(obj);
        end

        function obj = updateQMatrix(obj)
            % after changing uncertainty of state equation or parameter
            % this has to be called to update the Q Matrix
            sigma_vel = obj.sigma_state.vel*ones(1,3);
            sigma_angvel = obj.sigma_state.angvel*ones(1,3);
            sigma_internal_force = [obj.sigma_state.force*ones(1,2), obj.sigma_state.forceZ];
            sigma_internal_moment = obj.sigma_state.moment*ones(1,3);
            
            % This has to be changed if the boundary conditions of the
            % system are not adapted.
            for i = 1:obj.N
                if i == 1
                    temp = [sigma_internal_force,sigma_internal_moment];
                elseif i == obj.N
                    temp = [temp,...
                        [sigma_vel,sigma_angvel]];
                else
                    
                    temp = [temp,...
                        [sigma_vel,sigma_angvel,...
                        sigma_internal_force,sigma_internal_moment]];
                end
            end
            assert(length(temp) == obj.N*12-12);
            if obj.sdim==1
                temp = [temp,obj.sigma_bend_comp];
            end
            if obj.pdim ~= 0
                temp = [temp,obj.sigma_pressure*ones(1,obj.pdim)];
            end
            obj.Q = diag(temp);
        end

        function obj = setInitialCovariance(obj,sigma_vel,sigma_angvel,sigma_force,sigma_forceZ,sigma_moment)
            obj.sigma0_state.vel = sigma_vel;
            obj.sigma0_state.angvel = sigma_angvel;
            obj.sigma0_state.force = sigma_force;
            obj.sigma0_state.forceZ = sigma_forceZ;
            obj.sigma0_state.moment = sigma_moment;
            updateP0Matrix(obj);
        end

        function obj = updateP0Matrix(obj)
            % after changing uncertainty of state equation or parameter
            % this has to be called to update the P0 Matrix
            sigma_vel = obj.sigma0_state.vel*ones(1,3);
            sigma_angvel = obj.sigma0_state.angvel*ones(1,3);
            sigma_internal_force = [obj.sigma0_state.force*ones(1,2), obj.sigma0_state.forceZ];
            sigma_internal_moment = obj.sigma0_state.moment*ones(1,3);
            
            % This has to be changed if the boundary conditions of the
            % system are not adapted.
            for i = 1:obj.N
                if i == 1
                    temp = [sigma_internal_force,sigma_internal_moment];
                elseif i == obj.N
                    temp = [temp,...
                        [sigma_vel,sigma_angvel]];
                else
                    
                    temp = [temp,...
                        [sigma_vel,sigma_angvel,...
                        sigma_internal_force,sigma_internal_moment]];
                end
            end
            assert(length(temp) == obj.N*12-12);
            if obj.sdim==1
                temp = [temp,obj.sigma_bend_comp];
            end
            if obj.pdim ~= 0
                temp = [temp,obj.sigma_pressure*ones(1,obj.pdim)];
            end
            obj.P0 = diag(temp);
        end

        function obj = setMeasurementNoise(obj,sigma_meas_pos,sigma_meas_force,sigma_meas_torque,sigma_meas_orientation,sigma_meas_ang_vel,sigma_meas_accel,sigma_meas_pressure)
            R_vec = [];
            if obj.pos_flag
                assert(~isempty(sigma_meas_pos),'Position sensor noise must be set!')
                R_vec = [R_vec,sigma_meas_pos*ones(1,3)];
            end
            if obj.force_flag
                assert(~isempty(sigma_meas_force),'Force sensor noise must be set!')
                R_vec = [R_vec,sigma_meas_force*ones(1,3)];
            end
            if obj.torque_flag
                assert(~isempty(sigma_meas_torque),'Torque sensor noise must be set!')
                R_vec = [R_vec,sigma_meas_torque*ones(1,3)];
            end
            if obj.orientation_flag
                assert(~isempty(sigma_meas_orientation),'Orientation sensor noise must be set!')
                R_vec = [R_vec,sigma_meas_orientation*ones(1,3)];
            end
            if obj.ang_vel_flag
                assert(~isempty(sigma_meas_ang_vel),'Angular velocity sensor noise must be set!')
                R_vec = [R_vec,sigma_meas_ang_vel*ones(1,3)];
            end
           if obj.accel_flag
                assert(~isempty(sigma_meas_accel),'Acceleration sensor noise must be set!')
                R_vec = [R_vec,sigma_meas_accel*ones(1,3)];
            end
            if obj.pressure_flag
                assert(~isempty(sigma_meas_pressure),'Pressure sensor noise must be set!')
                R_vec = [R_vec,sigma_meas_pressure*ones(1,3)];
            end
            obj.R = diag(R_vec);
        end


        function obj = setInputAndMeas(obj,data_file) 
            obj.data_file = data_file;
            data_standard_struct = load_measurement(obj.data_file);
            assert(obj.t_start+obj.t_total<=data_standard_struct.time(end),"Total time exceeds data duration!")
            subsampling_step = round(obj.t_sample_se/data_standard_struct.time(2));
            start_idx  = find(1e-6>abs(data_standard_struct.time-obj.t_start));
            end_idx  = find(1e-6>abs(data_standard_struct.time-(obj.t_total+obj.t_start)));
            t_new = obj.t_start:obj.t_sample_se:(obj.t_total+obj.t_start);
            xEE_smooth = zeros(size(data_standard_struct.x_is));
            for i = 1:size(data_standard_struct.x_is,2)
                xEE_smooth(:,i) = smooth_moving(data_standard_struct.x_is(:,i),subsampling_step);
            end
            obj.positionData = interp1(data_standard_struct.time(start_idx:end_idx),xEE_smooth(start_idx:end_idx,:),t_new);
            force_torque_smooth = zeros(size(data_standard_struct.force_torque));
            for i = 1:size(data_standard_struct.force_torque,2)
                force_torque_smooth(:,i) = smooth_moving(data_standard_struct.force_torque(:,i),subsampling_step);
            end

            % save orientation for cost fun
            h_smooth = smoothdata(S3Group.logMap(data_standard_struct.h_is),1,'movmean',subsampling_step);
            h_new = interp1(data_standard_struct.time(start_idx:end_idx),h_smooth(start_idx:end_idx,:),t_new);
            obj.orientationData = S3Group.expMap(h_new);

            obj.measData = [];
            if obj.pos_flag
                obj.measData = [obj.measData,obj.positionData];
            end
            if obj.force_flag
                obj.measData = [obj.measData,interp1(data_standard_struct.time(start_idx:end_idx),force_torque_smooth(start_idx:end_idx,1:3),t_new)];
            end
            if obj.torque_flag
                obj.measData = [obj.measData,interp1(data_standard_struct.time(start_idx:end_idx),force_torque_smooth(start_idx:end_idx,4:6),t_new)];
            end
            if obj.orientation_flag
                obj.measData = [obj.measData,S3Group.logMap(obj.orientationData)];
            end
            if obj.ang_vel_flag
                ang_vel_smooth = zeros(size(data_standard_struct.ang_vel));
                for i = 1:size(data_standard_struct.ang_vel,2)
                    ang_vel_smooth(:,i) = smooth_moving(data_standard_struct.ang_vel(:,i),subsampling_step);
                end
                obj.measData = [obj.measData,interp1(data_standard_struct.time(start_idx:end_idx),ang_vel_smooth(start_idx:end_idx,:),t_new)];
            end
            if obj.accel_flag
                accel_smooth = zeros(size(data_standard_struct.accel));
                for i = 1:size(data_standard_struct.accel,2)
                    accel_smooth(:,i) = smooth_moving(data_standard_struct.accel(:,i),subsampling_step);
                end
                obj.measData = [obj.measData,interp1(data_standard_struct.time(start_idx:end_idx),accel_smooth(start_idx:end_idx,:),t_new)];
            end
            if obj.pressure_flag
                ang_vel_smooth = zeros(size(data_standard_struct.ang_vel));
                for i = 1:3
                    p_smooth(:,i) = smooth_moving(data_standard_struct.p_is(:,i),subsampling_step);
                end
                p_interp = interp1(data_standard_struct.time(start_idx:end_idx),p_smooth(start_idx:end_idx,:),t_new);
                obj.measData = [obj.measData,p_interp+10^5];
                obj.t = obj.t_start:obj.t_sample_se:(obj.t_total+obj.t_start);
            else
                f_cutoff = 10;
                f_sample = 1/data_standard_struct.time(2);
                [b,a] = butter(2,f_cutoff/(f_sample/2)); % causal filter
                pressure_smooth = filter(b,a,data_standard_struct.p_is(:,1:3),data_standard_struct.p_is(1:2,1:3),1);
                pressure = interp1(data_standard_struct.time(start_idx:end_idx),pressure_smooth(start_idx:end_idx,:),t_new);

                % pressure = interp1(data_standard_struct.time(start_idx:end_idx),data_standard_struct.p_filt(start_idx:end_idx,:),t_new);
                pressure = pressure + 100000; % absolute pressure: P0 = 100000
                obj.setInputData(t_new'-obj.t_start,pressure);
                obj.t = obj.t_start:obj.t_sample_se:(obj.t_total+obj.t_start);
                obj.pressure = obj.pressure_fun(obj.t-obj.t_start);
            end
        end

        function obj = setInputsUsingSim(obj,cdat,input_pressures)
            obj.sim = SimulationSetup(cdat,obj.t_total);
            obj.sim.setGravity([0,0,9.81]);
            obj.sim.setReferenceLength(obj.L);
            obj.sim.M_tip = obj.M_tip;
            obj.sim.DM = obj.D;
            obj.sim.CM = obj.C;
            obj.sim.MM = obj.M;
            obj.sim.compressiongain = obj.compressiongain;
            obj.sim.updateParameters();
            obj.t = 0:obj.t_sample:(obj.t_total);
            obj.sim.setInputData(obj.t,input_pressures);
            x_sim = obj.sim.run(obj.t);
            obj.sim_res = SimulationResults(obj.t,x_sim,[],cdat,obj.sim);
            start_idx  = find(1e-6>abs(obj.t -obj.t_start));
            end_idx  = find(1e-6>abs(obj.t -(obj.t_total+obj.t_start)));
            obj.positionData = obj.sim_res.rEE(start_idx:end_idx,:);
            obj.orientationData = rotm2quat(obj.sim_res.orientations(:,:,start_idx:end_idx));

            obj.measData = [];
            if obj.pos_flag
                obj.measData = [obj.measData,obj.positionData];
            end
            if obj.force_flag
                obj.measData = [obj.measData,obj.sim_res.FTS(start_idx:end_idx,1:3)];
            end
            if obj.torque_flag
                obj.measData = [obj.measData,obj.sim_res.FTS(start_idx:end_idx,4:6)];
            end
            if obj.orientation_flag
                obj.measData = [obj.measData,S3Group.logMap(obj.orientationData)];
            end
            if obj.ang_vel_flag
                obj.measData = [obj.measData,obj.getAngularVeloEE(x_sim)];
            end

            obj.setInputData(obj.t(start_idx:end_idx)-obj.t_start,input_pressures(start_idx:end_idx,:));
            obj.t = obj.t_start:obj.t_sample_se:(obj.t_total+obj.t_start);
            obj.pressure = input_pressures(start_idx:end_idx,:);
            obj.sim_flag = 1;
        end

        function obj = setStartTime(obj,start_time)
	        obj.t_start = start_time;
            obj.setInputAndMeas(obj.data_file);
        end

        function obj = setUKFParams(obj,alpha,beta,kappa)
	        obj.alpha = alpha;
            obj.beta = beta;
            obj.kappa = kappa;
        end
        
        function m = hfun(obj,X_curr,u_curr)
            bs = size(X_curr,1);
            if obj.sdim == 0
                supp_vals = obj.supp_vals;
            else
                supp_vals = X_curr(:,(obj.xdim+1):(obj.xdim+obj.sdim));
            end
            if obj.pos_flag || obj.orientation_flag || obj.accel_flag
                [rEE,orientation] = obj.EE(X_curr,u_curr,supp_vals);
            end
            if obj.force_flag || obj.torque_flag
                force_torque = obj.FTS(X_curr,u_curr,supp_vals);
            end
            % Store selected measurements in return value
            m = [];
            if obj.pos_flag
                m = [m,rEE];
            end
            if obj.force_flag
                m = [m,force_torque(:,1:3)];
            end
            if obj.torque_flag
                m = [m,force_torque(:,4:6)];
            end
            if obj.orientation_flag
                m = [m,S3Group.logMap(rotm2quat(orientation))];
            end
            if obj.ang_vel_flag
                m = [m,obj.getAngularVeloEE(X_curr)];
            end
            if obj.accel_flag
                m = [m,obj.getAccelerationEE(X_curr,u_curr,supp_vals,orientation)];
            end
            if obj.pressure_flag
                m = [m,repmat(X_curr(end-2:end),bs,1)];
            end
        end

        function x_next = ffun(obj,X_curr,u_norm_sel,t)
            % predictStep uses the PINN to predict one time step
            % Inputs:   X0 (BATCHSIZE x 72) Current states
            %           supp_vals (BATCHSIZE x sdim) estimated parameters: 1 bending damping, 2 tip_mass
            %           u_in (BATCHSIZE x 9) pressure inputs for current time [1:3], an intermediate step [4:6] next time step [7:9]
            % Outputs:  x_next (BATCHSIZE x 72) state for next time step
            bs = size(X_curr,1); % Batchsize
            if obj.pdim ~= 0 % Drücke durchschleifen, wenn sie als Zustände gesehen werden
                pressures = X_curr(:,end-2:end);
                X_curr = X_curr(:,1:end-3);
                u_norm = normalize_list(pressures,obj.props_names(obj.xdim+obj.sdim+1:obj.xdim+obj.sdim+size(pressures,2)),obj.m_x,obj.b_x,false,false,true);
                
                if obj.u_degree == 2 % Drücke interpolieren wenn von Netz erwartet
                    if t == 0
                        idx_last = 1; % X0
                    else
                        idx_last = round(t/obj.t_sample_se);
                    end
                    u_norm_last = normalize_list(obj.X_plus(idx_last,end-2:end),obj.props_names(obj.xdim+obj.sdim+1:obj.xdim+obj.sdim+size(pressures,2)),obj.m_x,obj.b_x,false,false,true);
                    u_in1 = repmat(u_norm_last,bs,1);         % Equivalent to u_in[:-2:2]
                    u_in2 = (repmat(u_norm_last,bs,1)+u_norm)./2;         % Mean of last and current
                    u_in3 = u_norm;           % Equivalent to u_in[2::2]
                    u_norm_sel = [u_in1,u_in2,u_in3];
                else
                    u_norm_sel = u_norm;
                end
            end

            X0_normal = toDevice(normalize_list(X_curr,obj.props_names(1:(obj.xdim+obj.sdim)),obj.m_x,obj.b_x,false,false,true));
            t_normal = toDevice(normalizeVar(obj.t_sample*ones(size(X_curr,1),1), obj.m_x.dt, obj.b_x.dt, true));
            x_normal = [X0_normal,repmat(u_norm_sel(1,:),bs,1),t_normal];
            % Actual prediction loop
            for i = 1:obj.t_ratio
                d_x_out = dd_pinn_forward(obj.weights,obj.biases,x_normal,obj.xdim,obj.n_funs);
                if i<obj.t_ratio
                    x_normal = [double(d_x_out(:,1:obj.xdim)),X0_normal(:,(obj.xdim+1):(obj.xdim+obj.sdim)),repmat(u_norm_sel(i+1,:),bs,1),t_normal];
                end
            end


            % Rescale
            x_next = [normalize_list(gather(d_x_out),obj.props_names(1:obj.xdim),obj.m_x,obj.b_x,false,false,false),X_curr(:,obj.xdim+1:(obj.xdim+obj.sdim))];
            obj.y = x_next;

            if obj.pdim ~= 0
                idx = round(t/obj.t_sample_se+2);
                x_next = [x_next, repmat(obj.measData(idx,end-2:end),bs,1)];
            end
        end

        function [X_plus,Qxx_plus] = pf(obj,disp_flag)
            %   function of the Particle Filter (PF) algorithm
            %
            %   OUTPUT:
            %       X_plus: filtered state vector
            %       Qxx_plus: VCM of the filtered states
            
            % initialize matrix of updated state matrices and VCMs (memory allocation)
            obj.X_plus = zeros(obj.nSteps+1,obj.xdim+obj.sdim);
            if obj.pdim ~= 0
                obj.X0 = [obj.X0, 10^5*ones(1,obj.pdim)];
            end
            obj.X_plus(1,:) = obj.X0;
            obj.Qxx_plus = cell(obj.nSteps+1,1);
            obj.Qxx_plus(1) = {obj.P0};

             obj.computation_time = zeros(obj.nSteps,1);
            
            %% generate the particles and store them in a variable called "S" %% (instead of if ep == 1)
            X_particles = mvnrnd(obj.X0,obj.P0,obj.np);
            
            for ep = 1:obj.nSteps
                if disp_flag
                    tic
                end
                t = (ep-1)*obj.t_sample_se;
                %% prediction step %%
                X_particles = X_particles + (ep>1)*mvnrnd(zeros(1,obj.xdim+obj.sdim+obj.pdim),obj.Q,obj.np); % add process noise
                start_idx = round(t/obj.t_sample+1);
                u_norm_sel = obj.u_normal(start_idx:start_idx+obj.t_ratio,:);
                X_particles = obj.ffun(X_particles,u_norm_sel,t); 
                
                %% update step of the filter %%
                if obj.pressure_flag
                    u_curr = X_particles(:,1:end-3);
                else
                    u_curr = obj.pressure(round(t/obj.t_sample_se+1),:);
                end
                Particles_PredObv = obj.hfun(X_particles,u_curr); % predict observation using particles as state representation
                res = obj.measData(ep,:) - Particles_PredObv; % deviations from real measurement
            
                % estimate the likelihood of the particles -> importance weights
                q = (mvnpdf(res,zeros(1,obj.mdim),obj.R) + 10^-60)'; % inputs of mvnpdf: samples/particles, mu/mean, sigma/vcm
                if sum(q,"omitnan")<1e-35              % Divergence check
                   obj.diverged = true;
                   obj.X_plus   = zeros(obj.nSteps+1,obj.xdim+obj.sdim);
                   obj.y = obj.X_plus;
                   disp(['Weights close to zero at t=',num2str(t),' !!!']);
                   return;
                end
                % derive the importance weights by using the derived likelihoods and
                % store the normalized importance weights w_tilde
                w_star = 1/obj.np*q;
                w_tilde = w_star/sum(w_star,"omitnan"); % normalization
                w_tilde(isnan(w_tilde)) = 0; % replace NaNs with weight of zero
                replaced_nans = obj.np-nnz(w_tilde); % for debugging
                obj.nan_counter = obj.nan_counter + replaced_nans;
                % resample the particles by using the "ResidualResample" function 
                new_i = obj.residualResample(1:obj.np,w_tilde); % get index of resampled points
                X_particles = X_particles(new_i,:); % select resampled points according to index and store them in the same variable (overwrite)
            
                %% derive the state vector "X" along with its VCM "Qxx" and store them persistently%%
                obj.X_plus(ep+1,:) = (1/obj.np*sum(X_particles,1));
                obj.Qxx_plus(ep+1) = {cov(X_particles)}; % calculate covariance matrix from particles and store it
                obj.computation_time(ep) = toc;
                if disp_flag
                    disp(['PF: Epoch ',num2str(ep),' of ',num2str(obj.nSteps),' took ',num2str(toc),' seconds. Removed ',num2str(replaced_nans),' NaNs.'])
                end
            end
            obj.y = obj.X_plus;
            X_plus = obj.X_plus;
            Qxx_plus = obj.Qxx_plus;
        end

        function [X_plus,Qxx_plus] = ukf(obj,disp_flag)
            %   function of the Unscented Kalman Filter (UKF) algorithmn for all-at-once evaluation
            %
            %   INPUT:
            %       disp_flag: decides if epochs should be displayed
            %
            %   OUTPUT:
            %       X_plus: filtered state vector
            %       Qxx_plus: VCM of the filtered states
            
            % initialize matrix of updated state matrices and VCMs (memory allocation)
            n = obj.xdim+obj.sdim+obj.pdim;
            obj.X_plus = zeros(obj.nSteps+1,obj.xdim+obj.sdim+obj.pdim);
            if obj.pdim ~= 0
                obj.X0 = [obj.X0, 10^5*ones(1,obj.pdim)];
            end
            obj.X_plus(1,:) = obj.X0;
            obj.Qxx_plus = cell(obj.nSteps+1,1);
            obj.Qxx_plus(1) = {obj.P0};

            obj.np = 2*(obj.xdim+obj.sdim+obj.pdim)+1; % Number of Sigma points
            obj.lambda   = (obj.alpha^2)*(n+obj.kappa)-n; % distribution parameter
            % Weights for mean and covariance
            W_m = zeros(obj.np,1);
            W_c = zeros(obj.np,1);
            W_m(1,1) = obj.lambda/(n + obj.lambda);
            W_c(1,1) = obj.lambda/(n + obj.lambda)+(1-obj.alpha^2+obj.beta);  
            W_m(2:obj.np,1) = 1/(2*(n+obj.lambda));
            W_c(2:obj.np,1) = 1/(2*(n+obj.lambda));

            obj.computation_time = zeros(obj.nSteps,1);

            x_sig = obj.unscented_trafo(obj.X0,obj.P0); % use initial state and VCM for first epoch
            
            for ep = 1:obj.nSteps
                if disp_flag
                    tic
                end
                t = (ep-1)*obj.t_sample_se;
                %% prediction step of the filter %%
                start_idx = round(t/obj.t_sample+1);
                u_norm_sel = obj.u_normal(start_idx:start_idx+obj.t_ratio,:);
                x_sig = obj.ffun(x_sig,u_norm_sel,t); 
                replaced_nans = nnz(any(isnan(x_sig),2)); % for debugging
                if replaced_nans>0
                    x_sig(any(isnan(x_sig),2),:) = zeros(replaced_nans,n); % replace NaNs with weight of zero
                end
                obj.nan_counter = obj.nan_counter + replaced_nans;
                %  weighted mean of the predicted Sigma points
                predX = sum(x_sig.*W_m,1);
                
                % derive the predicted VCM of the sigma points
                predP = zeros(n,n);
                e_x = x_sig - predX;
                for i=1:obj.np
                    predP = predP + ((e_x(i,:))' * (e_x(i,:)) .* W_c(i,1));
                end
                predP  = predP + obj.Q;
                
                try chol(predP); % Divergence check
                   % disp('Matrix is symmetric positive definite.')
                catch ME
                   obj.diverged = 1;
                   obj.y = obj.X_plus;
                   disp(['Covariance matrix not pos. def. at t=',num2str(t),' !!!']);
                   return;
                end

                % derive the new Sigma points based on the predicted states
                x_sig = obj.unscented_trafo(predX,predP);

                %% update step of the filter %%
                % predict the observations
                if obj.pressure_flag
                    u_curr = x_sig(:,1:end-3);
                else
                    u_curr = obj.pressure(round(t/obj.t_sample_se+1),:);
                end
                obs_sig = obj.hfun(x_sig,u_curr); % predict observation using particles as state representation
            
                %  weighted mean of the predicted observation Sigma points
                obs_mean = sum(obs_sig.*W_m,1);
                
                % derive the predicted VCM of the sigma points
                P_obs_obs = zeros(obj.mdim,obj.mdim);
                for i=1:obj.np
                    P_obs_obs = P_obs_obs + ((obs_sig(i,:)-obs_mean)'*(obs_sig(i,:)-obs_mean) .* W_c(i,1));
                end
                P_obs_obs  = P_obs_obs + obj.R;
                
                % derive the cross VCM between the state and observation
                e_x = x_sig - predX; % must be calculated again as sigma points from the update step should be used.
                P_x_obs = zeros(n,obj.mdim);
                for i=1:obj.np
                    P_x_obs = P_x_obs + (e_x(i,:)'*(obs_sig(i,:)-obs_mean) .* W_c(i,1));
                end
            
                % derive the Kalman gain (K)
                K  = P_x_obs/(P_obs_obs);
            
                %% derive the updated state vector along with its corresponding VCM and save them to persistend matrices
                obj.X_plus(ep+1,:) = (predX + (K * (obj.measData(ep,:)-obs_mean)')'); % double transposing because row vectors are used
                obj.Qxx_plus(ep+1) = {predP - K*P_obs_obs*K'}; 
                
                try chol(cell2mat(obj.Qxx_plus(ep+1))); % Divergence check
                catch ME
                   obj.diverged = 1;
                   obj.y = obj.X_plus;
                   disp(['Covariance matrix not pos. def. at t=',num2str(t),' !!!']);
                   return;
                end
                obj.computation_time(ep) = toc;
                if disp_flag
                    disp(['UKF epoch ',num2str(ep),' of ',num2str(obj.nSteps),' took ',num2str(toc),' seconds. Removed ',num2str(replaced_nans),' NaNs.'])
                end
                % Apply unscented transform
                x_sig = obj.unscented_trafo(obj.X_plus(ep+1,:),cell2mat(obj.Qxx_plus(ep+1))); % use updated state and VCM from last epoch
            end
            obj.y = obj.X_plus;
            Qxx_plus = obj.Qxx_plus;
        end

        function ukf_init(obj,disp_flag)
            % Initialization for step-wise evaluation
            obj.disp_flag = disp_flag;
            obj.n = obj.xdim+obj.sdim+obj.pdim;
            obj.X_plus = zeros(obj.nSteps+1,obj.xdim+obj.sdim+obj.pdim);
            if obj.pdim ~= 0
                obj.X0 = [obj.X0, 10^5*ones(1,obj.pdim)];
            end
            obj.X_plus(1,:) = obj.X0;
            obj.Qxx_plus = cell(obj.nSteps+1,1);
            obj.Qxx_plus(1) = {obj.P0};

            obj.np = 2*(obj.xdim+obj.sdim+obj.pdim)+1; % Number of Sigma points
            obj.lambda   = (obj.alpha^2)*(obj.n+obj.kappa)-obj.n; % distribution parameter
            % Weights for mean and covariance
            obj.W_m = zeros(obj.np,1);
            obj.W_c = zeros(obj.np,1);
            obj.W_m(1,1) = obj.lambda/(obj.n + obj.lambda);
            obj.W_c(1,1) = obj.lambda/(obj.n + obj.lambda)+(1-obj.alpha^2+obj.beta);  
            obj.W_m(2:obj.np,1) = 1/(2*(obj.n+obj.lambda));
            obj.W_c(2:obj.np,1) = 1/(2*(obj.n+obj.lambda));
        end

        function [x,error] = ukf_step(obj,measurement,u_norm,ep)
            % UKF algorithm for step-wise evaluation
            error = 0;
            if obj.disp_flag
                tic
            end
            t = (ep-1)*obj.t_sample_se;
            if ep == 1
                x_sig = obj.unscented_trafo(obj.X0,obj.P0); % use initial state and VCM for first epoch
            else
                x_sig = obj.unscented_trafo(obj.X_plus(ep,:),cell2mat(obj.Qxx_plus(ep)));
            end
            %% prediction step of the filter
            if obj.u_degree == 2
                u_norm_reshaped = [u_norm(1,:),u_norm(2,:),u_norm(3,:)];
            else
                u_norm_reshaped = u_norm;
            end
            x_sig = obj.ffun(x_sig,u_norm_reshaped,t); 
            replaced_nans = nnz(any(isnan(x_sig),2)); % for debugging
            if replaced_nans>0
                x_sig(any(isnan(x_sig),2),:) = zeros(replaced_nans,obj.n); % replace NaNs with weight of zero
            end
            obj.nan_counter = obj.nan_counter + replaced_nans;
            %  weighted mean of the predicted Sigma points
            predX = sum(x_sig.*obj.W_m,1);
            
            % derive the predicted VCM of the sigma points
            % predP = sum(W_c.*(x_sig-predX)'*(x_sig-predX)+obj.Q,1);
            predP = zeros(obj.n,obj.n);
            e_x = x_sig - predX;
            for i=1:obj.np
                predP = predP + (e_x(i,:)' * e_x(i,:) .* obj.W_c(i,1));
            end
            predP  = predP + obj.Q;
            
            try chol(predP); % Divergence check
               % disp('Matrix is symmetric positive definite.')
            catch ME
               obj.diverged = 1;
               obj.y = obj.X_plus;
               disp(['Covariance matrix not pos. def. at t=',num2str(t),' !!!']);
               error = 1;
               x = zeros(1,obj.xdim+obj.sdim);
               return;
            end

            % derive the new Sigma points based on the predicted states
            x_sig = obj.unscented_trafo(predX,predP);

            %% update step of the filter %%
            % predict the observations
            if obj.pressure_flag
                u_curr = x_sig(:,1:end-3);
            else
                u_curr = normalize_list(u_norm(end,:),obj.props_names(obj.xdim+obj.sdim+1:obj.xdim+obj.sdim+3),obj.m_x,obj.b_x,false,false,false);
            end
            obs_sig = obj.hfun(x_sig,u_curr); % predict observation using particles as state representation
        
            %  weighted mean of the predicted observation Sigma points
            obs_mean = sum(obs_sig.*obj.W_m,1);
            
            % derive the predicted VCM of the sigma points
            % P_obs_obs = sum(W_c.*(obs_sig-meas_mean)*(obs_sig-meas_mean)'+obj.R,1);
            P_obs_obs = zeros(obj.mdim,obj.mdim);
            for i=1:obj.np
                P_obs_obs = P_obs_obs + ((obs_sig(i,:)-obs_mean)'*(obs_sig(i,:)-obs_mean) .* obj.W_c(i,1));
            end
            P_obs_obs  = P_obs_obs + obj.R;
            
            % derive the cross VCM between the state and observation
            e_x = x_sig - predX; % must be calculated again as sigma points from the update step should be used.
            P_x_obs = zeros(obj.n,obj.mdim);
            for i=1:obj.np
                P_x_obs = P_x_obs + (e_x(i,:)'*(obs_sig(i,:)-obs_mean) .* obj.W_c(i,1));
            end
        
            % derive the Kalman gain (K)
            K  = P_x_obs/(P_obs_obs);
        
            %% derive the updated state vector along with its corresponding VCM and save them to persistend matrices
            obj.X_plus(ep+1,:) = (predX + (K * (measurement-obs_mean)')'); % double transposing because row vectors are used
            obj.Qxx_plus(ep+1) = {predP - K*P_obs_obs*K'}; 
            
            try chol(cell2mat(obj.Qxx_plus(ep+1))); % Divergence check
            catch ME
               obj.diverged = 1;
               obj.y = obj.X_plus;
               disp(['Covariance matrix not pos. def. at t=',num2str(t),' !!!']);
               error = 1;
               x = zeros(1,obj.xdim+obj.sdim);
               return;
            end
            
            if obj.disp_flag
                disp(['UKF epoch ',num2str(ep),' took ',num2str(toc),' seconds. Removed ',num2str(replaced_nans),' NaNs.'])
            end

            x = obj.X_plus(ep+1,:);
            obj.y = obj.X_plus;
        end


        function x_sig = unscented_trafo(obj,x,P)
            % Unscented transform
            n = obj.xdim+obj.sdim+obj.pdim;
            P_an = (sqrt(n+obj.lambda).*chol(P)); % Wurzelmatrix mittels Cholesky-Zerlegung
            x_sig   = zeros(obj.np,n);
            x_sig(1,:) = x;
            x_sig(2:(n+1),:) = x + P_an;
            x_sig(n+2:2*n+1,:) = x - P_an; %is equivalent to loop
        end

        function ax_states = plotStatesPF(obj,plot_data_flag,fig)
            szy = size(obj.X_plus);
            states_and_BV = [zeros(szy(1),obj.K/2),obj.X_plus(:,1:obj.xdim),zeros(szy(1),obj.K/2)];
            physicQuant = sort_solution(states_and_BV,'3D');
            if plot_data_flag
                data_standard_struct = load_measurement(obj.data_file);
                szy = size(data_standard_struct.states);
                states_and_BV_testbench = [zeros(szy(1),obj.K/2),data_standard_struct.states(:,1:obj.xdim),zeros(szy(1),obj.K/2)];
                physicQuant_testbench = sort_solution(states_and_BV_testbench,'3D');


            end
            N_groups = 4;
            % stateLabels = {'Velocities $q$ in m/s','Ang. Velocities $\omega$ in rad./s',...
            %             'Elastic Forces in N','Elastic Moments in Nm'};
            stateLabels = {'\boldmath$q$ in m/s','\boldmath$\omega$ in rad/s',...
                '\boldmath$n$ in N','\boldmath$m$ in Nm'};
            custom_corder =  [zeros(obj.N,1),linspace(1,0,obj.N)',linspace(0,1,obj.N)'];
            set_color_order_default(custom_corder)
            if nargin<3
                fig = figure;
            end
            ax_states = multi_col_fig(fig,N_groups,obj.K/N_groups,ones(obj.K,2));
            iAx = 0;
            fnames = fieldnames(physicQuant);
            for iState = 1:N_groups
                for i = 1:obj.K/N_groups
                    disp(['Plotting ',num2str(iAx)])
                    iAx = iAx + 1;
                    plot(ax_states(iAx), obj.t,physicQuant.(fnames{iAx}));
                    if plot_data_flag
                        hold(ax_states(iAx),'on')
                        plot(ax_states(iAx), data_standard_struct.time,physicQuant_testbench.(fnames{iAx}),'--');
                    end
                    if obj.sim_flag
                        hold(ax_states(iAx),'on')
                        plot(ax_states(iAx), obj.sim_res.time,obj.sim_res.physicQuant.(fnames{iAx}),'--');
                    end
                    box(ax_states(iAx),'on')
                    grid(ax_states(iAx),'on')
                    xlim(ax_states(iAx),[obj.t_start,obj.t_start+obj.t_total])
                    xticklabels(ax_states(iAx),{})
                    xticks(ax_states(iAx),'auto')
                    if i == 1
                        ylabel(ax_states(iAx),[stateLabels{iState}]);
                    end
                    if iState == N_groups
                        xlabel(ax_states(iAx),'t in s');
                        xticklabels(ax_states(iAx),'auto')
                    end
                    if iAx == 1
                        title(ax_states(iAx),'x');
                    elseif iAx == 2
                        title(ax_states(iAx),'y');
                    elseif iAx == 3
                        title(ax_states(iAx),'z');
                    end
                end
            end
            colormap(custom_corder)
            colorbar('Ticks',[0,1],'TickLabels',{'Base','Tip'})
            set_color_order_imes_default;
            if plot_data_flag
                legend('','','','','','Offline-UKF','','','','','','Online-UKF')
            end
            if obj.sim_flag
                legend('','','','','','Offline-UKF','','','','','','Simulation')
            end
        end

        function ax2 = plotSupplStates(obj,plot_data_flag,ax2)
            if nargin<3
                figure
                ax2 = axes;
            end
            if plot_data_flag
                data_standard_struct = load_measurement(obj.data_file);
            end
            set_color_order_imes_default
            plot(ax2,obj.t, obj.X_plus(:,obj.xdim+obj.sdim),'DisplayName','PINN-UKF')
            hold(ax2,'on')
            if plot_data_flag
                plot(ax2,data_standard_struct.time, data_standard_struct.states(:,obj.xdim+obj.sdim),'--','DisplayName','Online-UKF')
            end
            yline(ax2,obj.supp_vals,':',"LineWidth",1.5,'DisplayName','Identified value')
            ylabel(ax2,'$c_\mathrm{b}$ in $\frac{\mathbf{rad}}{\mathbf{Nm}\cdot \mathbf{m}}$') % zweites m für s
            xlabel(ax2,'$t$ in s')
            grid(ax2,'on');
            box(ax2,'on')
            xlim(ax2,[obj.t(1),obj.t(end)])
            legend(ax2,'Location','southwest')
        end

        function ax2 = plotEEAngVel(obj,plotDataFlag,ax2)
            if nargin<3
                figure
                ax2 = axes;
            end
            set_color_order_rgb_default
            plot(obj.t, obj.X_plus(:,(obj.xdim-2):obj.xdim),'DisplayName','PINN-UKF')
            if plotDataFlag
                data_standard_struct = load_measurement(obj.data_file);
                hold(ax2,'on')
                plot(ax2,data_standard_struct.time,data_standard_struct.ang_vel,'--','DisplayName','Measurement');
            end
            ylabel(ax2,'Angular velocity $\omega$ in rad/s')
            xlabel(ax2,'$t$ in s')
            grid(ax2,'on');
            box(ax2,'on')
            xlim(ax2,[obj.t(1),obj.t(end)])
            legend(ax2,'NumColumns',2)
            set_color_order_imes_default
        end

        function ax2 = plotEEAccel(obj,plotDataFlag,ax2)
            if nargin<3
                figure
                ax2 = axes;
            end
            set_color_order_rgb_default
            accel_ukf = obj.getAccelerationEE(obj.X_plus,obj.pressure, obj.X_plus(:,(obj.xdim+1):(obj.xdim+obj.sdim)),obj.orientation);
            plot(obj.t, accel_ukf,'DisplayName','PINN-UKF')
            if plotDataFlag
                data_standard_struct = load_measurement(obj.data_file);
                hold(ax2,'on')
                plot(ax2,data_standard_struct.time,data_standard_struct.accel,'--','DisplayName','Measurement');
            end
            ylabel(ax2,'Tip acceleration in $m/s^2$')
            xlabel(ax2,'$t$ in s')
            grid(ax2,'on');
            box(ax2,'on')
            xlim(ax2,[obj.t(1),obj.t(end)])
            legend(ax2,'NumColumns',2)
            set_color_order_imes_default
        end

        function ax = plotPressures(obj,ax)
            if nargin<2
                figure
                ax = axes;
            end
            if obj.pressure_flag
                plot(obj.t,obj.X_plus(:,end-2:end),"DisplayName","PINN-UKF")
                hold on
                plot(obj.t,obj.measData(:,end-2:end),"--","DisplayName","Measurement")
                legend (ax,'NumColumns',2)
                xlim(ax,[obj.t(1),obj.t(end)])
                xlabel(ax,"Time in s")
                ylabel(ax,"Abs. Pressure in Pa")
            else 
                plot(obj.t,obj.pressure)
                xlim(ax,[obj.t(1),obj.t(end)])
                xlabel(ax,"Time in s")
                ylabel(ax,"Pressure in Pa")
            end

        end

        function t_pf = plotEEAnimation(obj)
            figure
            t_pf = plotTransforms([obj.rEE(1,1:3);obj.measData(1,1:3)],[rotm2quat(obj.orientation(:,:,1));obj.orientationData(1,:)],'FrameSize',0.01);
            hold on
            scatter3(t_pf,obj.rEE(1,1),obj.rEE(1,2),obj.rEE(1,3),'r.');
            scatter3(t_pf,obj.positionData(1,1),obj.positionData(1,2),obj.positionData(1,3),'b.');
            xlabel("x")
            ylabel("y")
            zlabel("z")
            ax = gca;
            grid on
            ax.DataAspectRatio = [1 1 1];
            xlim([-0.1,0.1])
            ylim([-0.1,0.1])
            zlim([0,0.2])
            legend("PINN-UKF","Measurement","AutoUpdate","off")
            drawnow()

            for i = 2:size(obj.rEE,1)
                t_pf.Children((i-1)*2+1).Children(1).Children.Matrix = [obj.orientation(:,:,i),obj.rEE(i,:)';0 0 0 1];
                t_pf.Children((i-1)*2+1).Children(2).Children.Matrix = [quat2rotm(obj.orientationData(i,:)),obj.positionData(i,1:3)';0 0 0 1];
                scatter3(t_pf,obj.rEE(i,1),obj.rEE(i,2),obj.rEE(i,3),'r.');
                scatter3(t_pf,obj.positionData(i,1),obj.positionData(i,2),obj.positionData(i,3),'b.');
                pause(0.01)
                drawnow()
            end
        end
        
        function outIndex = residualResample(~,inIndex, weights)
            % RESIDUALRESAMPLE  Residual Resampling from Doucet et a.
            %   outIndex = ResidualResample(inIndex, weights)
            %
            %   INPUT
            %          inIndex        (r-vector) Input particle indices
            %          weights        (r-vector) Normalized importance weights
            %   OUTPUT
            %          outIndex          Resampled particle indices

            % Number of particles
            S = length(weights);       

            % initialize output vector
            outIndex = zeros(1,S);  

            N_kind= zeros(1,S);

            weights_res = S*weights;
            N_kind = fix(weights_res);
            % residual number of particles that need to be generated
            N_res = S-sum(N_kind);

            if N_res
                weights_res = (weights_res-N_kind)/N_res;
                cumDist = cumsum(weights_res);

                % Generate N_res sorted particles from a uniform dristribution [0,1]
                u = fliplr(cumprod(rand(1,N_res).^(1./(N_res:-1:1))));
                j=1;
                for i=1:N_res
                    while j<size(cumDist,2) && (u(1,i)>=cumDist(1,j))
                        j=j+1;
                    end
                    N_kind(1,j)=N_kind(1,j)+1;
                end
            end


            index=1;
            for i=1:S
                if (N_kind(1,i)>0)
                    for j=index:index+N_kind(1,i)-1
                        outIndex(j) = inIndex(i);
                    end
                end
                index = index+N_kind(1,i);
            end
        end
    end
end

