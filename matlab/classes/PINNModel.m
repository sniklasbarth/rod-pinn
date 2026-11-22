classdef PINNModel < matlab.mixin.Copyable
    %PINNMODEL Cosserat rod model using PINN as surrogate
    %   There are two modes trajectory prediction or single step prediction
    %   For trajectory prediction set an input using setInputData and then
    %   call the predict_traj function. For single step prediction the
    %   current state and input is given directly as a function argument
    %   for the predict step function.

    properties
        pinnFilename
        data_file
        net
        weights
        biases
        n_funs
        m_x
        b_x
        props_names
        xdim
        udim
        u_degree
        sdim
        supp_vals
        t_sample
        t_total
        pressure_fun
        pressure
        u_normal
        X0
        x_normal
        t_normal
        computation_time
        y
        C
        D
        L % Reference length
        N
        K = 12; % states per node
        xi_idx
        r0 = [0;0;0.024];  % base point in m
        fts_pos_offset= [0;0;0.024]; % Offset of FT-sensor to bending body in m
        EE_pos_offset = [0;0;0.01]; % Offset of rod calculation EE and measured EE
        radius = 0.02115;  % radius of cross-section
        M_base = 0.0167 + 1077*pi*0.02115^2*0.01; % adapter(incl. screws) + Dragonskin cap
        compressiongain
        M
        M_tip
        A_p
        r_p
        p0
        input_wrench
        xi0 = [0 0 1 0 0 0] 
        param
        rEE
        orientation
        force_torque
        ang_vel
        t
        sim_flag = 0
        y_sim
        sim_res
        t_ratio = 1
        strains
        imu_offset = [0;0;0.005]
        accel
        xi_base
    end

    methods
        function obj = PINNModel(importedPinnFile,t_total)
            %PINNModel Construct an instance of this class
            %   Detailed explanation goes here

            % Store data in class object
            obj.pinnFilename = importedPinnFile;
            pinn = load_pinn(obj.pinnFilename);
            for i = 1:length(pinn.weights)
                obj.weights{i} = toDevice(double(pinn.weights{i}'));
                obj.biases{i} = toDevice(double(pinn.biases{i}'));
            end
            n_funs = pinn.n_funs; m_x = pinn.m_x; b_x = pinn.b_x; props_names = pinn.props_names;
            xdim = pinn.xdim; udim = pinn.udim; sdim = pinn.sdim; t_sample = pinn.t_sample;
            obj.n_funs = n_funs;
            props_names = cellstr(props_names);

            obj.m_x = m_x;
            obj.b_x = b_x;
            obj.props_names = props_names;
            obj.xdim = xdim;
            obj.N = obj.xdim/12+1;
            obj.udim = udim;
            if udim == 9
                obj.u_degree = 2;
            elseif udim == 3
                obj.u_degree = 0;
            end
            obj.sdim = sdim;
            % obj.supp_vals = supp_vals;
            obj.t_sample = t_sample;
            obj.t_normal = normalizeVar(obj.t_sample, obj.m_x.dt, obj.b_x.dt, true);
            try
                obj.t_total = t_total;
            catch
                warning("t_total wasn't set. Only step-wise evaluation possible.")
            end
        end

        function obj = setInputData(obj,time,pressureData)
            %% setInputData sets ABSOLUTE (!) pressure
            % pressureData - Pressure as Nx3 in Pa (absolute!)
            obj.t = 0:obj.t_sample:(obj.t_total);
            obj.pressure_fun =  @(t)interp1(time', pressureData, t);
            obj.pressure = obj.pressure_fun(obj.t);
            if obj.u_degree == 2
                t_NN = [0:obj.t_sample/2:(obj.t_total+obj.t_sample)]'; % second degree excitation needs 1 value in between
                u_in = obj.pressure_fun(t_NN);
                u_norm = normalize_list(u_in,obj.props_names(obj.xdim+obj.sdim+1:obj.xdim+obj.sdim+size(u_in,2)),obj.m_x,obj.b_x,false,false,true);
                u_in1 = u_norm(1:2:end-2,:);         % Equivalent to u_in[:-2:2]
                u_in2 = u_norm(2:2:end-1,:);         % Equivalent to u_in[1:-1:2]
                u_in3 = u_norm(3:2:end,:);           % Equivalent to u_in[2::2]
                obj.u_normal = [u_in1,u_in2,u_in3];
            else
                u_in = obj.pressure_fun(obj.t);
                u_norm = normalize_list(u_in,obj.props_names(obj.xdim+obj.sdim+1:obj.xdim+obj.sdim+size(u_in,2)),obj.m_x,obj.b_x,false,false,true);
                obj.u_normal = u_norm;
            end
        end

        function obj = setInputDataFromFile(obj,data_file) 
            obj.data_file = data_file;
            data_standard_struct = load_measurement(obj.data_file);
            assert(obj.t_total<=data_standard_struct.time(end),"Set total time exceeds measurement duration.")
            end_idx  = find((abs(data_standard_struct.time-obj.t_total)<=0.0001));
            t_new = 0:obj.t_sample:obj.t_total;

            f_cutoff = 10;
            f_sample = 1/data_standard_struct.time(2);
            [b,a] = butter(2,f_cutoff/(f_sample/2)); % causal filter
            pressure_smooth = filter(b,a,data_standard_struct.p_is(:,1:3),[],1);
            pressure = interp1(data_standard_struct.time(1:end_idx),pressure_smooth(1:end_idx,:),t_new);
            pressure = pressure + 100000; % absolute pressure: P0 = 100000
            obj.setInputData(t_new,pressure);
            obj.pressure = obj.pressure_fun(obj.t);
        end

        function obj = setRodParams(obj,rodParamFile)
            % rodParamFile: parameters/rod_parameters.json
            rod = jsondecode(fileread(rodParamFile));
            obj.C = diag(rod.C);
            obj.D = diag(rod.D);
            obj.L = rod.ref_length;
            obj.xi_idx = [7:12]+(obj.K*[0:obj.N-2]');
            obj.xi_idx = obj.xi_idx(:);
            obj.compressiongain = rod.compression_gain;
            obj.p0 = rod.p0;
            obj.A_p = obj.compressiongain*(5.15/1000)^2*pi;
            obj.r_p = rod.chamber_radius;
            obj.M = diag(rod.M);
            obj.M_tip = rod.M_tip;
            obj.param = [diag(obj.C)', diag(obj.D)', obj.L, diag(obj.M)', obj.M_tip, obj.xi0(4:6), obj.xi0(1:3)];
            obj.supp_vals = [];
            if any(strcmp(obj.props_names,'C45'))
                obj.supp_vals = [obj.supp_vals,obj.C(4,4)];
            end
            if any(strcmp(obj.props_names,'M_tip'))
                obj.supp_vals = [obj.supp_vals,obj.M_tip];
            end
            % Initialize model inputs
            obj.X0 = [zeros(1,obj.xdim),obj.supp_vals];
            % calculated X0 with grav
            % obj.X0 = [7.82390047362269e-05	5.37689512363073e-06	1.88075590808755	9.43149657252116e-06	-0.000137237362462995	-2.42410616690967e-23	0	0	0	0	0	0	-0.000582530919459926	-4.00338382468939e-05	1.72497189157487	-2.60061583078164e-05	0.000378414660549019	-4.95494923748637e-24	0	0	0	0	0	0	-0.00110130616286513	-7.56861332361997e-05	1.52067479322508	-2.71885570017277e-05	0.000395619700797334	1.23617794137664e-23	0	0	0	0	0	0	-0.00148531941783647	-0.000102077049186960	1.31637754743300	-2.90093879410818e-05	0.000422114545352124	7.19484943211465e-24	0	0	0	0	0	0	-0.00172958262949645	-0.000118863787158445	1.11208022398595	-3.12780684797102e-05	0.000455126033083520	2.68634322690401e-24	0	0	0	0	0	0	-0.00181972836632659	-0.000125058960197931	0.907782917521278	-3.37888942174404e-05	0.00049166	1.871e-24	0	0	0	0	0	0 obj.supp_vals];
        end

        function predStates = predictTraj(obj)
            X0_normal = normalize_list(obj.X0,obj.props_names(1:(obj.xdim+obj.sdim)),obj.m_x,obj.b_x,false,false,true);
            % X0_normal = toDevice(X0_normal);
            obj.t_normal = normalizeVar(obj.t_sample*ones(obj.t_total/obj.t_sample+1,1), obj.m_x.dt, obj.b_x.dt, true);
            obj.x_normal = zeros((size(obj.t_normal,1)),obj.xdim+obj.sdim+obj.udim+1);
            obj.x_normal(1,:) = [X0_normal,obj.u_normal(1,:),obj.t_normal(1)];
            d_x_out = zeros(size(obj.t_normal,1),obj.xdim);
            num_steps = round(obj.t_total/obj.t_sample);
            % Actual prediction loop
            start_pred = tic;
            for i = 1:num_steps
                d_x_out(i,:) = dd_pinn_forward(obj.weights,obj.biases,obj.x_normal(i,:),obj.xdim,obj.n_funs);
                obj.x_normal(i+1,:) =  [d_x_out(i,:),X0_normal(obj.xdim+1:obj.xdim+obj.sdim),obj.u_normal(i+1,:),obj.t_normal(i+1)];
            end
            obj.computation_time = toc(start_pred);
            disp(['Mean computation time per step for PINN: ',num2str(obj.computation_time/num_steps),' s']);
            % Rescale
            obj.y = normalize_list(obj.x_normal(:,1:obj.xdim),obj.props_names(1:obj.xdim),obj.m_x,obj.b_x,false,false,false);
            predStates = obj.y;

        end

        function x_next = step(obj,x_norm,supp_vals,u_norm,t_step) % u must include value from last step
            t_ratio = t_step/obj.t_sample;
            assert(floor(t_ratio)==t_ratio,"t_sample_se must be multiple of t_sample_pinn!")

            x_normal = [x_norm,supp_vals,u_norm(1,:),obj.t_normal];
            % Actual prediction loop
            for i = 1:t_ratio
                d_x_out = dd_pinn_forward(obj.weights,obj.biases,x_normal,obj.xdim,obj.n_funs);
                % disp(['NaNs in PINN-output:',num2str(nnz(extractdata(isnan(d_x_out))))])
                if i<t_ratio
                    x_normal = [d_x_out(:,1:obj.xdim),supp_vals,u_normal(i+1,:),obj.t_normal];
                end
            end
            
            x_next = d_x_out;
        end

        function [rEE,orientation] = calcEE(obj)
            if (isa(obj,'PINNStateEst') || isa(obj,'PINNRealResults'))  && obj.sdim ~= 0
                if isa(obj,'PINNStateEst') && obj.pressure_flag
                    [rEE,orientation] = EE(obj,obj.X_plus(:,1:obj.xdim),obj.X_plus(:,(obj.xdim+obj.sdim+1):end), obj.X_plus(:,(obj.xdim+1):(obj.xdim+obj.sdim)));
                else
                    [rEE,orientation] = EE(obj,obj.X_plus(:,1:obj.xdim),obj.pressure, obj.X_plus(:,(obj.xdim+1):(obj.xdim+obj.sdim)));
                end
            else
                [rEE,orientation] = EE(obj,obj.y,obj.pressure_fun(obj.t),obj.supp_vals);
            end
            obj.rEE = rEE;
            obj.orientation = orientation;
        end

        function [rEE,orientation] = EE(obj,X_curr,u_curr,supp_vals)
            T_base  = [eye(3),obj.r0;0 0 0 1];
            xi_base  = SE3Group.logMap(T_base);

            input_wrench = obj.actuation_model_pressure(u_curr');
            if iscolumn(input_wrench)
                input_wrench = repmat(input_wrench,1,size(X_curr,1));
            end
            eta = obj.eta_pls(X_curr(:,1:obj.xdim)',input_wrench,obj.param,supp_vals);
            obj.strains = eta;

            % This is basically the RodEndEffector function for batchwise evaluation
            num  = size(eta,2);
            ds = obj.L/(num-1);
            bs = size(eta,3); % batchsize
            T0 = SE3Group.expMap(xi_base);
            temp = repmat(T0,1,1,bs);
            for i = 1:num-1
                % piecewise constant integration to approximate linear strain function
                temp = pagemtimes(temp, SE3Group.expMap(squeeze(eta(:,i,:))'.*ds));
            end
            endeffector_pose = temp;
            
            T_rodEE_measEE = [diag([1 1 1]),obj.EE_pos_offset;0 0 0 1];
            endeffector_pose = pagemtimes(endeffector_pose, repmat(T_rodEE_measEE,1,1,bs));
            rEE = squeeze(endeffector_pose(1:3,4,:))';
            orientation = endeffector_pose(1:3,1:3,:);
        end

        function force_torque = calcForceTorque(obj)
            if isa(obj,'PINNStateEst') && obj.sdim ~= 0
                if obj.pressure_flag
                    force_torque = FTS(obj,obj.X_plus(:,1:obj.xdim),obj.X_plus(:,(obj.xdim+obj.sdim+1):end), obj.X_plus(:,(obj.xdim+1):(obj.xdim+obj.sdim)));
                else
                    force_torque = FTS(obj,obj.X_plus(:,1:obj.xdim),obj.pressure, obj.X_plus(:,(obj.xdim+1):(obj.xdim+obj.sdim)));
                end
            else
                force_torque = FTS(obj,obj.y,obj.pressure_fun(obj.t),obj.supp_vals);
            end
            obj.force_torque = force_torque;
        end

        function force_torque = FTS(obj,X_curr,u_curr,supp_vals)
            %% Output model for state estimation
            input_wrench = obj.actuation_model_pressure(u_curr');
            if iscolumn(input_wrench)
                input_wrench = repmat(input_wrench,1,size(X_curr,1));
            end
            wrench =  obj.out_ft_sensor(X_curr(:,1:obj.xdim)', input_wrench,obj.param,supp_vals)';
            force_torque = wrench + [repmat([0,0,9.81]*obj.M_base,size(wrench,1),1),-cross(repmat(obj.fts_pos_offset',size(wrench,1),1), wrench(:,1:3),2)];
        end

        function ang_vel = getAngularVeloEE(obj,X_curr)
            %% Output of angular velocity at tip for state estimation
            ang_vel = X_curr(:,(obj.xdim-2):obj.xdim);
            obj.ang_vel = ang_vel;
        end

        function accel = getAccelerationEE(obj,X_curr,u_curr,supp_vals,orientation)
            %% Output of acceleration at tip for state estimation
            if size(supp_vals,1) == 1
                supp_vals = repmat(supp_vals,size(X_curr,1),1);
            end
            input_wrench = obj.actuation_model_pressure(u_curr');
            if iscolumn(input_wrench)
                input_wrench = repmat(input_wrench,1,size(X_curr,1));
            end
            param_gravity  = [0,0,9.81,obj.L,obj.param(1,16),obj.M_tip]; 
            tip_accel_local = zeros(size(X_curr,1),6);
            for i = 1:size(X_curr,1)
                param = cat(2,obj.param(1:3),supp_vals(i)*ones(1,2),obj.param(6:end));
                tau_p = u_p(X_curr(i,:)', input_wrench(:,i), param);
                eta = eta_pls(X_curr(i,:)', input_wrench(:,i), param);
                u_g = gravity(eta, param_gravity,obj.N);
                dx = f_(X_curr(i,:)', input_wrench(:,i), reshape(tau_p, [], 1), u_g, param);
                tip_accel_local(i,:) = dx(end-5:end); % tip node: last six state derivatives
            end
            lin_accel_rod = tip_accel_local(:,1:3); % Linear acceleration at rod tip
            ang_accel = tip_accel_local(:,4:6);
            ang_vel = obj.getAngularVeloEE(X_curr);
            lin_accel_atImu = lin_accel_rod; %+ cross(ang_accel,obj.imu_offset) + cross(ang_vel,cross(ang_vel,obj.imu_offset));
            accel = lin_accel_atImu + squeeze(pagemtimes(orientation,[0;0;9.81]))';
            obj.accel = accel;
        end

        function obj = plotStates(obj,fig,xlimits)
            if nargin<2
                fig = figure;
            end
            szy = size(obj.y(:,1:obj.xdim));
            states_and_BV = [zeros(szy(1),obj.K/2),obj.y(:,1:obj.xdim),zeros(szy(1),obj.K/2)];
            physicQuant = sort_solution(states_and_BV,'3D');
            N_groups = 4;
            % stateLabels = {'Velocities $q$ in m/s','Ang. Velocities $\omega$ in rad./s',...
            %             'Elastic Forces in N','Elastic Moments in Nm'};
            stateLabels = {'\boldmath$q$ in m/s','\boldmath$\omega$ in rad/s',...
                        '\boldmath$n$ in N','\boldmath$m$ in Nm'};
            % custom_corder =  [zeros(obj.N,1),linspace(1,0,obj.N)',linspace(0,1,obj.N)'];
            custom_corder = parula(obj.N+2);
            custom_corder = custom_corder(2:obj.N+1,:);
            set_color_order_default(custom_corder)
            
            ax_states = multi_col_fig(fig,N_groups,obj.K/N_groups,ones(obj.K,2));
            iAx = 0;
            fnames = fieldnames(physicQuant);
            for iState = 1:N_groups
                for i = 1:obj.K/N_groups
                    disp(['Plotting ',num2str(iAx)])
                    iAx = iAx + 1;
                    plot(ax_states(iAx), obj.t,physicQuant.(fnames{iAx}));
                    if obj.sim_flag || isa(obj,"PINNMPC")
                        hold(ax_states(iAx),'on')
                        plot(ax_states(iAx), obj.t,obj.sim_res.physicQuant.(fnames{iAx}),':');
                    end
                    box(ax_states(iAx),'on')
                    grid(ax_states(iAx),'on')
                    if nargin==3
                        xlim(ax_states(iAx),xlimits)
                    end
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
            if isa(obj,"PINNStateEst")
                legend('','','','','','PINN-UKF','','','','','','Simulation','Location','northwest')
            elseif isa(obj,"PINNMPC")
                legend('','','','','','NEMPC','','','','','','Reference','Location','northwest')
            elseif isa(obj,"PINNModel")
                legend('','','','','','PINN','','','','','','Simulation','Location','northwest')
            end
            colormap(custom_corder)
            colorbar('Ticks',[0,1],'TickLabels',{'Base','Tip'})
            set_color_order_imes_default;
        end

        function [ax] = plotEE3d(obj,plotDataFlag,ax)  
            % 3d trajectory plot
            set_color_order_imes_default
            if nargin<3
                figure
                ax = axes;
            end
            if isa(obj,"PINNStateEst")
                p1 = plot3(ax,obj.rEE(:,1),obj.rEE(:,2),obj.rEE(:,3),'DisplayName','PINN-UKF');
            elseif isa(obj,"PINNMPC")
                p1 = plot3(ax,obj.rEE(:,1),obj.rEE(:,2),obj.rEE(:,3),'DisplayName','PINN-NEMPC');
                hold(ax,'on')
                if isrow(unique(obj.sim_res.rEE,"rows"))
                    p4 = plot3(ax,obj.sim_res.rEE(:,1),obj.sim_res.rEE(:,2),obj.sim_res.rEE(:,3),'*','DisplayName','Ground truth');
                else
                    p4 = plot3(ax,obj.sim_res.rEE(:,1),obj.sim_res.rEE(:,2),obj.sim_res.rEE(:,3),'--','DisplayName','Ground truth');
                end
            else
                p1 = plot3(ax,obj.rEE(:,1),obj.rEE(:,2),obj.rEE(:,3),'DisplayName','PINN');
            end
            p2 = xline(ax,0,'k--','DisplayName','');
            p3 = yline(ax,0,'k--','DisplayName','');
            ax.ZDir = 'reverse';
            grid(ax,'on')
            box(ax,'on')
            daspect(ax,[1 1 1])
            xlabel(ax,'x in m')
            ylabel(ax,'y in m')
            zlabel(ax,'z in m')
            if plotDataFlag
                data_standard_struct = load_measurement(obj.data_file);
                hold(ax,'on')
                p4 = plot3(ax,data_standard_struct.x_is(:,1),data_standard_struct.x_is(:,2),data_standard_struct.x_is(:,3),'--','DisplayName','Measurement');
            end
            if obj.sim_flag
                hold(ax,'on')
                p4 = plot3(ax,obj.sim_res.rEE(:,1),obj.sim_res.rEE(:,2),obj.sim_res.rEE(:,3),':','DisplayName','Simulation');
            end
            legend(ax,[p1,p4],'Location','northeast')
        end

        function [ax2] = plotEE(obj,plotDataFlag,ax2,legend_flag)  
           % time vs ee position
            if nargin<3
                figure
                ax2 = axes;
            end  
            if nargin<4
                legend_flag = 1;
            end 
            if plotDataFlag
                data_standard_struct = load_measurement(obj.data_file);
            end
            set_color_order_rgb_default
            if isa(obj,"PINNStateEst")
                plt_pinn = plot(obj.t, [obj.rEE(:,1:2),obj.rEE(:,3)-obj.L],'DisplayName','PINN-UKF');
                label = {'PINN-UKF'};
            elseif isa(obj,"PINNMPC")
                plt_pinn = plot(obj.t, [obj.rEE(:,1:2),obj.rEE(:,3)-obj.L],'DisplayName','PINN-NEMPC');
                hold(ax2,'on')
                plt_pinn = [plt_pinn, plot(ax2,obj.t,[obj.sim_res.rEE(:,1:2),obj.sim_res.rEE(:,3)-obj.L],'--','DisplayName','Ground truth')]
                label = {'NEMPC','Reference\,'};
            else
                % colororder([1.0 0.0 0.0; % Red
                %             0.0 0.4 0.0; % Green
                %             0 0.5 1.0])
                % yyaxis left
                plt_pinn = plot(obj.t, [obj.rEE(:,1:2),obj.sim_res.rEE(:,3)-obj.L],'DisplayName','PINN');
                % yyaxis right
                % plt_pinn = [plt_pinn; plot(obj.t, obj.rEE(:,3),'DisplayName','PINN')];
                label = {'PINN'};
            end
            ylabel(ax2,'\boldmath${\tilde{r}}_\mathrm{EE}$ in m')
            xlabel(ax2,'$t$ in s')
            grid(ax2,'on');
            box(ax2,'on')
            xlim(ax2,[obj.t(1),obj.t(end)])
            if plotDataFlag
                
                hold(ax2,'on')
                plt_pinn = [plt_pinn,plot(ax2,data_standard_struct.time,[data_standard_struct.x_is(:,1:2),data_standard_struct.x_is(:,3)-obj.L],'--','DisplayName','Measurement')];
                % legend(ax2,'PINN x','PINN y','PINN z','Measurement x','Measurement y','Measurement z')
                label{end+1} = 'Measurement';
            end
            if obj.sim_flag
                hold(ax2,'on')
                plt_pinn = [plt_pinn,plot(ax2,obj.t,[obj.sim_res.rEE(:,1:2),obj.sim_res.rEE(:,3)-obj.L],':','DisplayName','Simulation')];
                label{end+1} = 'Simulation';
            end
            if legend_flag == 1
                legend(ax2, plt_pinn(1:3,1), {'x', 'y', 'z'}, 'NumColumns', 3, 'Location', 'southwest');
            elseif legend_flag == 2
                % colors: coordinates, line styles: PINN/measurement/simulation
                legend(ax2, [plt_pinn(1:3,1); plt_pinn(1,1:end)'], [{'x', 'y', 'z'}, label], ...
                    'NumColumns', 3, 'Location', 'southwest');

                % % For PINN prediction
                % [hl(2).leg, hl(2).obj, hl(2).hout, hl(2).mout] = ...
                % legend_compact(plt_pinn(1:3,1), {'x', 'y','z\,'}, ...
                % 'ncol',3,...
                % 'anchor', {'sw','sw'}, ...
                % 'buffer', [-4 -17], ...
                % 'fontsize',12, ...
                % 'xscale',0.8, ...
                % 'title', 'Color');
                % label = {'PINN','Meas','Sim\,'};
                % [hl(3).leg, hl(3).obj, hl(3).hout, hl(3).mout] = ...
                % legend_compact(plt_pinn(1,1:end), label, ...
                % 'ncol',3,...
                % 'ref', hl(2).leg, ...
                % 'anchor', {'e','w'}, ...
                % 'buffer', [3 0], ...
                % 'fontsize', 12', ...
                % 'title', 'Line');

                % % For PINN var bend compliance
                % [hl(2).leg, hl(2).obj, hl(2).hout, hl(2).mout] = ...
                % legend_compact(plt_pinn(1:3,1), {'x', 'y','z'}, ...
                % 'ncol',3,...
                % 'anchor', {'sw','sw'}, ...
                % 'buffer', [5 -15], ...
                % 'fontsize',12, ...
                % 'xscale',0.5, ...
                % 'title', 'Color');
                % label = {'PINN','Sim'};
                % [hl(3).leg, hl(3).obj, hl(3).hout, hl(3).mout] = ...
                % legend_compact(plt_pinn(1,1:end), label, ...
                % 'ref', hl(2).leg, ...
                % 'anchor', {'e','w'}, ...
                % 'buffer', [3 10], ...
                % 'fontsize', 12', ...
                % 'title', 'Line');

                % For MPC sim presentation
                % [hl(1).leg, hl(1).obj, hl(1).hout, hl(1).mout] = ...
                % legend_compact(plt_pinn(1,1:2), label, ...
                % 'ncol',2,...
                % 'anchor', {'sw','sw'}, ...
                % 'buffer', [5 5], ...
                % 'fontsize', 12', ...
                % 'title', 'Line');
                % [hl(2).leg, hl(2).obj, hl(2).hout, hl(2).mout] = ...
                % legend_compact(plt_pinn(1:3,1), {'x', 'y','z'}, ...
                % 'ncol',3,...
                % 'anchor', {'ne','nw'}, ...
                % 'ref', hl(1).leg, ...
                % 'buffer', [5 0], ...
                % 'fontsize',12, ...
                % 'xscale',0.5, ...
                % 'title', 'Color');
            end
            
        end



        function ax = plotOrientation(obj,plotDataFlag,ax,legend_flag)
            if nargin<3
                figure
                ax = axes;
            end
            if nargin<4
                legend_flag = 0;
            end
            set_color_order_rgb_default
            h_pinn_tang = S3Group.logMap(rotm2quat(obj.orientation));
            if isa(obj,"PINNStateEst")
                h_meas_tang = S3Group.logMap(obj.orientationData);
                meas_t = obj.t;
                h_pinn_tang = S3Group.logMap(rotm2quat(obj.orientation));
                plt_pinn = plot(ax,obj.t,h_pinn_tang,'DisplayName','PINN-UKF');
                label = {'UKF'};
            elseif isa(obj,"PINNMPC")
                plot(obj.t, h_pinn_tang,'DisplayName','PINN-NEMPC')
                hold(ax,'on')
                h_gt_tang = S3Group.logMap(rotm2quat(obj.sim_res.orientation));
                plot(ax,obj.t,h_gt_tang,'--','DisplayName','Ground truth')
            else
                plt_pinn = plot(obj.t, h_pinn_tang,'DisplayName','PINN');
                label = {'PINN'};
            end
            grid(ax,'on')
            
            if plotDataFlag
                data_standard_struct = load_measurement(obj.data_file);
                h_meas_tang = S3Group.logMap(data_standard_struct.h_is);
                meas_t = data_standard_struct.time;
                hold(ax,'on')
                plt_pinn = [plt_pinn,plot(ax,meas_t,h_meas_tang,'--','DisplayName','Measurement')];
                label{end+1} = 'Meas';
            end
            if obj.sim_flag
                hold(ax,'on')
                plt_pinn = [plt_pinn,plot(ax,obj.t,S3Group.logMap(rotm2quat(obj.sim_res.orientations)),':','DisplayName','Simulation')];
                label{end+1} = 'Sim';
            end
            % legend (ax,'NumColumns',2)
            if legend_flag
                [hl(2).leg, hl(2).obj, hl(2).hout, hl(2).mout] = ...
                legend_compact(plt_pinn(1:3,1), {'x', 'y','z'}, ...
                'ncol',3,...
                'anchor', {'nw','nw'}, ...
                'buffer', [5 0], ...
                'fontsize',12, ...
                'xscale',0.5, ...
                'title', 'Color');
            end
            if legend_flag == 2
                [hl(3).leg, hl(3).obj, hl(3).hout, hl(3).mout] = ...
                legend_compact(plt_pinn(1,1:end), label, ...
                'ref', hl(2).leg, ...
                'ncol',2,...
                'anchor', {'ne','nw'}, ...
                'buffer', [3 0], ...
                'fontsize', 12', ...
                'title', 'Line');
            end
            xlim(ax,[obj.t(1),obj.t(end)])
            xlabel(ax,"t in s")
            ylabel(ax,"\boldmath${\theta}_\mathrm{EE}$ in rad") % (Tangent space of SO3)
            set_color_order_imes_default
        end
        
        function [ax] = plotFTS(obj,plotDataFlag,ax,legend_flag) 
            if nargin<3
                figure
                tiledlayout(2,1,"TileSpacing","compact")
                ax(1) = nexttile;
                ax(2) = nexttile;
            end  
            if nargin<4
                legend_flag = 0;
            end
            set_color_order_rgb_default
            if isa(obj,"PINNStateEst")
                plot(ax(1),obj.t,obj.force_torque(:,1:3),'DisplayName','PINN-UKF');
                plt = plot(ax(2),obj.t,obj.force_torque(:,4:6),'DisplayName','PINN-UKF');
                label = {'UKF'};
            elseif isa(obj,"PINNMPC")
                plot(ax(1),obj.t,obj.force_torque(:,1:3),'DisplayName','PINN-NEMPC');
                plot(ax(2),obj.t,obj.force_torque(:,4:6),'DisplayName','PINN-NEMPC');
                hold(ax(1),'on')
                hold(ax(2),'on')
                plot(ax(1),obj.t,obj.sim_res.FTS(:,1:3),':','DisplayName','Ground truth');
                plot(ax(2),obj.t,obj.sim_res.FTS(:,4:6),':','DisplayName','Ground truth');
                label = {'NEMPC','Ground truth'};
            else
                plot(ax(1),obj.t,obj.force_torque(:,1:3),'DisplayName','PINN');
                plt = plot(ax(2),obj.t,obj.force_torque(:,4:6),'DisplayName','PINN');
                label = {'PINN'};
            end
            ylabel(ax(1),'\boldmath${f}_\mathrm{react}$ in N')
            ylabel(ax(2),'\boldmath${m}_\mathrm{react}$ in Nm')
            xlabel(ax(2),'t in s')
            for i = 1:length(ax)
                box(ax(i),'on')
                grid(ax(i),'on')
            end
            xlim(ax,[obj.t(1),obj.t(end)])
            if plotDataFlag
                data_standard_struct = load_measurement(obj.data_file);
                force_torque_smooth = zeros(size(data_standard_struct.force_torque));
                for i = 1:size(data_standard_struct.force_torque,2)
                    force_torque_smooth(:,i) = smooth_moving(data_standard_struct.force_torque(:,i),20);
                end
                hold(ax(1),'on')
                hold(ax(2),'on')
                plot(ax(1),data_standard_struct.time,force_torque_smooth(:,1:3),'--','DisplayName','Measurement');
                plt = [plt,plot(ax(2),data_standard_struct.time,force_torque_smooth(:,4:6),'--','DisplayName','Measurement')];
                label{end+1} = 'Meas';
                % legend('PINN x','PINN y','PINN z','Measurement x','Measurement y','Measurement z')
            end
            if obj.sim_flag
                hold(ax(1),'on')
                hold(ax(2),'on')
                plot(ax(1),obj.t,obj.sim_res.FTS(:,1:3),':','DisplayName','Simulation');
                plt = [plt,plot(ax(2),obj.t,obj.sim_res.FTS(:,4:6),':','DisplayName','Simulation')];
                label{end+1} = 'Sim\,';
            end

            if legend_flag == 1
                [hl(2).leg, hl(2).obj, hl(2).hout, hl(2).mout] = ...
                legend_compact(plt(1:3,1), {'x', 'y','z'}, ...
                'ncol',3,...
                'anchor', {'nw','nw'}, ...
                'buffer', [5 -5], ...
                'fontsize',12, ...
                'xscale',0.5, ...
                'title', 'Color');
            end
            if legend_flag == 2
             % For comparison UKF measurements
            [hl(2).leg, hl(2).obj, hl(2).hout, hl(2).mout] = ...
            legend_compact(plt(1:3,1), {'x', 'y','z'}, ...
            'ncol',3,...
            'anchor', {'sw','sw'}, ...
            'buffer', [0 25], ...
            'fontsize',12, ...
            'xscale',0.5, ...
            'title', 'Color');
            [hl(3).leg, hl(3).obj, hl(3).hout, hl(3).mout] = ...
            legend_compact(plt(1,1:end), label, ...
            'ncol',2,...
            'ref', hl(2).leg, ...
            'anchor', {'e','w'}, ...
            'buffer', [3 0], ...
            'fontsize', 12', ...
            'title', 'Line');
             % For comparison UKF perturbation
            % [hl(2).leg, hl(2).obj, hl(2).hout, hl(2).mout] = ...
            % legend_compact(plt(1:3,1), {'x', 'y','z'}, ...
            % 'ncol',3,...
            % 'anchor', {'nw','nw'}, ...
            % 'buffer', [-715 20], ...
            % 'fontsize',12, ...
            % 'xscale',0.5, ...
            % 'title', 'Color');
            % [hl(3).leg, hl(3).obj, hl(3).hout, hl(3).mout] = ...
            % legend_compact(plt(1,1:end), label, ...
            % 'ncol',2,...
            % 'ref', hl(2).leg, ...
            % 'anchor', {'ne','nw'}, ...
            % 'buffer', [3 0], ...
            % 'fontsize', 12', ...
            % 'title', 'Line');
            end
            set_color_order_imes_default
        end

        function plotStrains(obj,fig)
            %% This function is a comparison of orientations.
            % Since orientations can be obtained by integration of the time
            % or spatial ODEs, the error introduced by just using time
            % integration is investigated.
            % First the SE(3) elements representing the robot's
            % configuration in space are computed (and stored). Afterwards
            % the orientations from the quaternion states are plotted and
            % an error (in axis-angle) is given.
            % waveplots
            if nargin<2
                fig = figure;
            end
            for i = 1:6
                % Add strain to physical quantities
                physicQuant.(['eta',num2str(i)])= squeeze(obj.strains(i,:,:));
                sz = size(physicQuant.(['eta',num2str(i)]));
                ax = subplot(2,3,i);
                %generate waveplots for strains
                s_ = linspace(0, obj.L, sz(1));
                
                waveplot(ax,s_,obj.t,physicQuant.(['eta',num2str(i)]))
                view(ax,[30,30])
                title(ax,['$\epsilon_',num2str(i),'(s,t)$'])
            end
        end

        function ax = plotPressures(obj,ax,legend_flag)
            if nargin<2
                figure
                ax = axes;
            end
            if nargin<3
                legend_flag = 1;
            end
            grid(ax,'on') 
            hold(ax,'on') 
            if isa(obj,"PINNMPC")
                u_ref = obj.sim.pressure_handle(obj.t);
                plt = plot(obj.t,(obj.u_mpc-1e5)/1e3);
                plt = [plt, plot(obj.t,(u_ref-1e5)/1e3,'--')];
                label = {'NEMPC','Ground truth'};
            else
                plt = plot(obj.t,(obj.pressure-1e5)/1e3);

            end

            ylabel("\boldmath${p}$ in kPa")
            ylim([0 1e2])
            box(ax,'on')

            if legend_flag == 1
                [hl(2).leg, hl(2).obj, hl(2).hout, hl(2).mout] = ...
                legend_compact(plt(1:3,1), {'$p_1$','$p_2$','$p_3\,$'}, ...
                'ncol',3,...
                'anchor', {'nw','nw'}, ...
                'buffer', [10 -5], ...
                'fontsize',12, ...
                'xscale',0.8,...
                'title', 'Color');
            end
            if legend_flag == 2
                [hl(2).leg, hl(2).obj, hl(2).hout, hl(2).mout] = ...
                legend_compact(plt(1:3,1), {'$p_1$','$p_2$','$p_3\,$'}, ...
                'ncol',3,...
                'anchor', {'nw','nw'}, ...
                'buffer', [10 -5], ...
                'fontsize',12, ...
                'xscale',0.5, ...
                'title', 'Color');
                [hl(3).leg, hl(3).obj, hl(3).hout, hl(3).mout] = ...
                legend_compact(plt(1,1:end), label, ...
                'ref', hl(2).leg, ...
                'anchor', {'ne','nw'}, ...
                'buffer', [0 0], ...
                'fontsize', 12', ...
                'title', 'Line');
            end
        end

        function out = configuration(obj,iT)
            % Basis
            % T_base  = [eye(3),obj.r0;0 0 0 1];
            % obj.xi_base  = SE3Group.logMap(T_base);
            config = rodShapeSparse(obj.strains(:,:,iT),obj.xi_base,obj.L);
            % measured end effector (marker frame) as last point of the shape
            T_rodEE_measEE = [diag([1 1 1]),obj.EE_pos_offset;0 0 0 1];
            config(:,:,end+1) = config(:,:,end) * T_rodEE_measEE;
            out = config;
        end

        function out = configurationSparse(obj,iT)
            % T_base  = [obj.R0,obj.r0;0 0 0 1];
            % obj.xi_base  = SE3Group.logMap(T_base);
            config = rodShapeSparse(obj.strains(:,:,iT),obj.xi_base,obj.L);
            % only output every (Ng-1) nodes
            % Ng is the number of interpolation nodes
            % spatial coordinate --->
            %                       x---x---x   N  = 3 with
            %                       x-.-x-.-x   if Ng = 3 (example 1)
            %                       x...x...x   if Ng = 5 (example 2)
            % where x are part of the N nodes and the . are intermediate
            % nodes for spatial integration of position and rotation, which
            % are required for gravity, thus the name Ng.
            %
            out = config(:,:,(1:obj.Ng-1:end));
        end


         function p = plotCurrentConfiguration(obj,ax,iT,lineStyle)             
                rodConfig = obj.configuration(iT);       
                % position orientation
                r = squeeze(rodConfig(1:3,4,:,iT));
                hold(ax,'on')
                if nargin>3
                    p = plot3(ax,r(1,:),r(2,:),r(3,:),lineStyle);
                else
                    p = plot3(ax,r(1,:),r(2,:),r(3,:));
                end
        end



        function y_sim = runSimulation(obj,cdat)
            sim = SimulationSetup(cdat,obj.t_total);
            sim.setGravity([0,0,9.81]);
            sim.setReferenceLength(obj.L);
            sim.M_tip = obj.M_tip;
            sim.DM = obj.D;
            sim.CM = obj.C;
            sim.MM = obj.M;
            sim.compressiongain = obj.compressiongain;
            sim.updateParameters();
            sim.setInputData(obj.t,obj.pressure_fun(obj.t));
            opts = odeset('Jacobian',@sim.j_f,'NormControl','on');
            start_sim = tic;
            [t,y_sim] = ode23s(@sim.f,obj.t,sim.x0,opts);
            sim_duration = toc(start_sim);
            disp(['Mean computation time per step for simulation: ',num2str(sim_duration/length(t)),' s']);
            obj.sim_res = SimulationResults(t,y_sim,[],cdat,sim);
            obj.y_sim = y_sim;
            obj.sim_flag = 1;
        end

        function eta_pls = eta_pls(~,in1,in2,in3,supp_vals)
            %ETA_PLS
            %    ETA_PLS = ETA_PLS(IN1,IN2,IN3)

            %    This function was generated by the Symbolic Math Toolbox version 8.7.
            %    22-Oct-2024 16:39:39
            %    Batch version (samples as columns); C_4 = C_5 from supp_vals,
            %    external tip wrench (export input IN3) removed as in model/TRO_N_7

            C_1 = in3(:,1);
            C_2 = in3(:,2);
            C_3 = in3(:,3);
            if exist('supp_vals')
                C_4 = supp_vals(:,1)';
                C_5 = supp_vals(:,1)';
            else
                C_4 = in3(:,4);
                C_5 = in3(:,5);
            end
            C_6 = in3(:,6);
            u0_1 = in3(:,21);
            u0_2 = in3(:,22);
            u0_3 = in3(:,23);
            v0_1 = in3(:,24);
            v0_2 = in3(:,25);
            v0_3 = in3(:,26);
            w1 = in2(1,:);
            w2 = in2(2,:);
            w3 = in2(3,:);
            w4 = in2(4,:);
            w5 = in2(5,:);
            w6 = in2(6,:);
            x10_1 = in1(4,:);
            x10_2 = in1(16,:);
            x10_3 = in1(28,:);
            x10_4 = in1(40,:);
            x10_5 = in1(52,:);
            x10_6 = in1(64,:);
            x11_1 = in1(5,:);
            x11_2 = in1(17,:);
            x11_3 = in1(29,:);
            x11_4 = in1(41,:);
            x11_5 = in1(53,:);
            x11_6 = in1(65,:);
            x12_1 = in1(6,:);
            x12_2 = in1(18,:);
            x12_3 = in1(30,:);
            x12_4 = in1(42,:);
            x12_5 = in1(54,:);
            x12_6 = in1(66,:);
            x7_1 = in1(1,:);
            x7_2 = in1(13,:);
            x7_3 = in1(25,:);
            x7_4 = in1(37,:);
            x7_5 = in1(49,:);
            x7_6 = in1(61,:);
            x8_1 = in1(2,:);
            x8_2 = in1(14,:);
            x8_3 = in1(26,:);
            x8_4 = in1(38,:);
            x8_5 = in1(50,:);
            x8_6 = in1(62,:);
            x9_1 = in1(3,:);
            x9_2 = in1(15,:);
            x9_3 = in1(27,:);
            x9_4 = in1(39,:);
            x9_5 = in1(51,:);
            x9_6 = in1(63,:);
            mt1(1,:,:) = vertcat(v0_1+C_1.*(w1+x7_1),v0_2+C_2.*(w2+x8_1),v0_3+C_3.*(w3+x9_1),u0_1+C_4.*(w4+x10_1),u0_2+C_5.*(w5+x11_1),u0_3+C_6.*(w6+x12_1),v0_1+(C_1.*x7_2)./7.0+C_1.*(w1+x7_1).*(6.0./7.0),v0_2+(C_2.*x8_2)./7.0+C_2.*(w2+x8_1).*(6.0./7.0),v0_3+(C_3.*x9_2)./7.0+C_3.*(w3+x9_1).*(6.0./7.0),u0_1+(C_4.*x10_2)./7.0+C_4.*(w4+x10_1).*(6.0./7.0),u0_2+(C_5.*x11_2)./7.0+C_5.*(w5+x11_1).*(6.0./7.0),u0_3+(C_6.*x12_2)./7.0+C_6.*(w6+x12_1).*(6.0./7.0),v0_1+C_1.*x7_2.*(2.0./7.0)+C_1.*(w1+x7_1).*(5.0./7.0),v0_2+C_2.*x8_2.*(2.0./7.0)+C_2.*(w2+x8_1).*(5.0./7.0),v0_3+C_3.*x9_2.*(2.0./7.0)+C_3.*(w3+x9_1).*(5.0./7.0));
            mt2(1,:,:) = vertcat(u0_1+C_4.*x10_2.*(2.0./7.0)+C_4.*(w4+x10_1).*(5.0./7.0),u0_2+C_5.*x11_2.*(2.0./7.0)+C_5.*(w5+x11_1).*(5.0./7.0),u0_3+C_6.*x12_2.*(2.0./7.0)+C_6.*(w6+x12_1).*(5.0./7.0),v0_1+C_1.*x7_2.*(3.0./7.0)+C_1.*(w1+x7_1).*(4.0./7.0),v0_2+C_2.*x8_2.*(3.0./7.0)+C_2.*(w2+x8_1).*(4.0./7.0),v0_3+C_3.*x9_2.*(3.0./7.0)+C_3.*(w3+x9_1).*(4.0./7.0),u0_1+C_4.*x10_2.*(3.0./7.0)+C_4.*(w4+x10_1).*(4.0./7.0),u0_2+C_5.*x11_2.*(3.0./7.0)+C_5.*(w5+x11_1).*(4.0./7.0),u0_3+C_6.*x12_2.*(3.0./7.0)+C_6.*(w6+x12_1).*(4.0./7.0),v0_1+C_1.*x7_2.*(4.0./7.0)+C_1.*(w1+x7_1).*(3.0./7.0),v0_2+C_2.*x8_2.*(4.0./7.0)+C_2.*(w2+x8_1).*(3.0./7.0),v0_3+C_3.*x9_2.*(4.0./7.0)+C_3.*(w3+x9_1).*(3.0./7.0));
            mt3(1,:,:) = vertcat(u0_1+C_4.*x10_2.*(4.0./7.0)+C_4.*(w4+x10_1).*(3.0./7.0),u0_2+C_5.*x11_2.*(4.0./7.0)+C_5.*(w5+x11_1).*(3.0./7.0),u0_3+C_6.*x12_2.*(4.0./7.0)+C_6.*(w6+x12_1).*(3.0./7.0),v0_1+C_1.*x7_2.*(5.0./7.0)+C_1.*(w1+x7_1).*(2.0./7.0),v0_2+C_2.*x8_2.*(5.0./7.0)+C_2.*(w2+x8_1).*(2.0./7.0),v0_3+C_3.*x9_2.*(5.0./7.0)+C_3.*(w3+x9_1).*(2.0./7.0),u0_1+C_4.*x10_2.*(5.0./7.0)+C_4.*(w4+x10_1).*(2.0./7.0),u0_2+C_5.*x11_2.*(5.0./7.0)+C_5.*(w5+x11_1).*(2.0./7.0),u0_3+C_6.*x12_2.*(5.0./7.0)+C_6.*(w6+x12_1).*(2.0./7.0),v0_1+C_1.*x7_2.*(6.0./7.0)+(C_1.*(w1+x7_1))./7.0,v0_2+C_2.*x8_2.*(6.0./7.0)+(C_2.*(w2+x8_1))./7.0,v0_3+C_3.*x9_2.*(6.0./7.0)+(C_3.*(w3+x9_1))./7.0);
            mt4(1,:,:) = vertcat(u0_1+C_4.*x10_2.*(6.0./7.0)+(C_4.*(w4+x10_1))./7.0,u0_2+C_5.*x11_2.*(6.0./7.0)+(C_5.*(w5+x11_1))./7.0,u0_3+C_6.*x12_2.*(6.0./7.0)+(C_6.*(w6+x12_1))./7.0,v0_1+C_1.*x7_2,v0_2+C_2.*x8_2,v0_3+C_3.*x9_2,u0_1+C_4.*x10_2,u0_2+C_5.*x11_2,u0_3+C_6.*x12_2,v0_1+C_1.*x7_2.*(6.0./7.0)+(C_1.*x7_3)./7.0,v0_2+C_2.*x8_2.*(6.0./7.0)+(C_2.*x8_3)./7.0,v0_3+C_3.*x9_2.*(6.0./7.0)+(C_3.*x9_3)./7.0,u0_1+C_4.*x10_2.*(6.0./7.0)+(C_4.*x10_3)./7.0,u0_2+C_5.*x11_2.*(6.0./7.0)+(C_5.*x11_3)./7.0,u0_3+C_6.*x12_2.*(6.0./7.0)+(C_6.*x12_3)./7.0,v0_1+C_1.*x7_2.*(5.0./7.0)+C_1.*x7_3.*(2.0./7.0),v0_2+C_2.*x8_2.*(5.0./7.0)+C_2.*x8_3.*(2.0./7.0));
            mt5(1,:,:) = vertcat(v0_3+C_3.*x9_2.*(5.0./7.0)+C_3.*x9_3.*(2.0./7.0),u0_1+C_4.*x10_2.*(5.0./7.0)+C_4.*x10_3.*(2.0./7.0),u0_2+C_5.*x11_2.*(5.0./7.0)+C_5.*x11_3.*(2.0./7.0),u0_3+C_6.*x12_2.*(5.0./7.0)+C_6.*x12_3.*(2.0./7.0),v0_1+C_1.*x7_2.*(4.0./7.0)+C_1.*x7_3.*(3.0./7.0),v0_2+C_2.*x8_2.*(4.0./7.0)+C_2.*x8_3.*(3.0./7.0),v0_3+C_3.*x9_2.*(4.0./7.0)+C_3.*x9_3.*(3.0./7.0),u0_1+C_4.*x10_2.*(4.0./7.0)+C_4.*x10_3.*(3.0./7.0),u0_2+C_5.*x11_2.*(4.0./7.0)+C_5.*x11_3.*(3.0./7.0),u0_3+C_6.*x12_2.*(4.0./7.0)+C_6.*x12_3.*(3.0./7.0),v0_1+C_1.*x7_2.*(3.0./7.0)+C_1.*x7_3.*(4.0./7.0),v0_2+C_2.*x8_2.*(3.0./7.0)+C_2.*x8_3.*(4.0./7.0),v0_3+C_3.*x9_2.*(3.0./7.0)+C_3.*x9_3.*(4.0./7.0),u0_1+C_4.*x10_2.*(3.0./7.0)+C_4.*x10_3.*(4.0./7.0));
            mt6(1,:,:) = vertcat(u0_2+C_5.*x11_2.*(3.0./7.0)+C_5.*x11_3.*(4.0./7.0),u0_3+C_6.*x12_2.*(3.0./7.0)+C_6.*x12_3.*(4.0./7.0),v0_1+C_1.*x7_2.*(2.0./7.0)+C_1.*x7_3.*(5.0./7.0),v0_2+C_2.*x8_2.*(2.0./7.0)+C_2.*x8_3.*(5.0./7.0),v0_3+C_3.*x9_2.*(2.0./7.0)+C_3.*x9_3.*(5.0./7.0),u0_1+C_4.*x10_2.*(2.0./7.0)+C_4.*x10_3.*(5.0./7.0),u0_2+C_5.*x11_2.*(2.0./7.0)+C_5.*x11_3.*(5.0./7.0),u0_3+C_6.*x12_2.*(2.0./7.0)+C_6.*x12_3.*(5.0./7.0),v0_1+(C_1.*x7_2)./7.0+C_1.*x7_3.*(6.0./7.0),v0_2+(C_2.*x8_2)./7.0+C_2.*x8_3.*(6.0./7.0),v0_3+(C_3.*x9_2)./7.0+C_3.*x9_3.*(6.0./7.0),u0_1+(C_4.*x10_2)./7.0+C_4.*x10_3.*(6.0./7.0),u0_2+(C_5.*x11_2)./7.0+C_5.*x11_3.*(6.0./7.0),u0_3+(C_6.*x12_2)./7.0+C_6.*x12_3.*(6.0./7.0));
            mt7(1,:,:) = vertcat(v0_1+C_1.*x7_3,v0_2+C_2.*x8_3,v0_3+C_3.*x9_3,u0_1+C_4.*x10_3,u0_2+C_5.*x11_3,u0_3+C_6.*x12_3,v0_1+C_1.*x7_3.*(6.0./7.0)+(C_1.*x7_4)./7.0,v0_2+C_2.*x8_3.*(6.0./7.0)+(C_2.*x8_4)./7.0,v0_3+C_3.*x9_3.*(6.0./7.0)+(C_3.*x9_4)./7.0,u0_1+C_4.*x10_3.*(6.0./7.0)+(C_4.*x10_4)./7.0,u0_2+C_5.*x11_3.*(6.0./7.0)+(C_5.*x11_4)./7.0,u0_3+C_6.*x12_3.*(6.0./7.0)+(C_6.*x12_4)./7.0,v0_1+C_1.*x7_3.*(5.0./7.0)+C_1.*x7_4.*(2.0./7.0),v0_2+C_2.*x8_3.*(5.0./7.0)+C_2.*x8_4.*(2.0./7.0),v0_3+C_3.*x9_3.*(5.0./7.0)+C_3.*x9_4.*(2.0./7.0),u0_1+C_4.*x10_3.*(5.0./7.0)+C_4.*x10_4.*(2.0./7.0),u0_2+C_5.*x11_3.*(5.0./7.0)+C_5.*x11_4.*(2.0./7.0),u0_3+C_6.*x12_3.*(5.0./7.0)+C_6.*x12_4.*(2.0./7.0));
            mt8(1,:,:) = vertcat(v0_1+C_1.*x7_3.*(4.0./7.0)+C_1.*x7_4.*(3.0./7.0),v0_2+C_2.*x8_3.*(4.0./7.0)+C_2.*x8_4.*(3.0./7.0),v0_3+C_3.*x9_3.*(4.0./7.0)+C_3.*x9_4.*(3.0./7.0),u0_1+C_4.*x10_3.*(4.0./7.0)+C_4.*x10_4.*(3.0./7.0),u0_2+C_5.*x11_3.*(4.0./7.0)+C_5.*x11_4.*(3.0./7.0),u0_3+C_6.*x12_3.*(4.0./7.0)+C_6.*x12_4.*(3.0./7.0),v0_1+C_1.*x7_3.*(3.0./7.0)+C_1.*x7_4.*(4.0./7.0),v0_2+C_2.*x8_3.*(3.0./7.0)+C_2.*x8_4.*(4.0./7.0),v0_3+C_3.*x9_3.*(3.0./7.0)+C_3.*x9_4.*(4.0./7.0),u0_1+C_4.*x10_3.*(3.0./7.0)+C_4.*x10_4.*(4.0./7.0),u0_2+C_5.*x11_3.*(3.0./7.0)+C_5.*x11_4.*(4.0./7.0),u0_3+C_6.*x12_3.*(3.0./7.0)+C_6.*x12_4.*(4.0./7.0),v0_1+C_1.*x7_3.*(2.0./7.0)+C_1.*x7_4.*(5.0./7.0),v0_2+C_2.*x8_3.*(2.0./7.0)+C_2.*x8_4.*(5.0./7.0));
            mt9(1,:,:) = vertcat(v0_3+C_3.*x9_3.*(2.0./7.0)+C_3.*x9_4.*(5.0./7.0),u0_1+C_4.*x10_3.*(2.0./7.0)+C_4.*x10_4.*(5.0./7.0),u0_2+C_5.*x11_3.*(2.0./7.0)+C_5.*x11_4.*(5.0./7.0),u0_3+C_6.*x12_3.*(2.0./7.0)+C_6.*x12_4.*(5.0./7.0),v0_1+(C_1.*x7_3)./7.0+C_1.*x7_4.*(6.0./7.0),v0_2+(C_2.*x8_3)./7.0+C_2.*x8_4.*(6.0./7.0),v0_3+(C_3.*x9_3)./7.0+C_3.*x9_4.*(6.0./7.0),u0_1+(C_4.*x10_3)./7.0+C_4.*x10_4.*(6.0./7.0),u0_2+(C_5.*x11_3)./7.0+C_5.*x11_4.*(6.0./7.0),u0_3+(C_6.*x12_3)./7.0+C_6.*x12_4.*(6.0./7.0),v0_1+C_1.*x7_4,v0_2+C_2.*x8_4,v0_3+C_3.*x9_4,u0_1+C_4.*x10_4,u0_2+C_5.*x11_4,u0_3+C_6.*x12_4,v0_1+C_1.*x7_4.*(6.0./7.0)+(C_1.*x7_5)./7.0,v0_2+C_2.*x8_4.*(6.0./7.0)+(C_2.*x8_5)./7.0);
            mt10(1,:,:) = vertcat(v0_3+C_3.*x9_4.*(6.0./7.0)+(C_3.*x9_5)./7.0,u0_1+C_4.*x10_4.*(6.0./7.0)+(C_4.*x10_5)./7.0,u0_2+C_5.*x11_4.*(6.0./7.0)+(C_5.*x11_5)./7.0,u0_3+C_6.*x12_4.*(6.0./7.0)+(C_6.*x12_5)./7.0,v0_1+C_1.*x7_4.*(5.0./7.0)+C_1.*x7_5.*(2.0./7.0),v0_2+C_2.*x8_4.*(5.0./7.0)+C_2.*x8_5.*(2.0./7.0),v0_3+C_3.*x9_4.*(5.0./7.0)+C_3.*x9_5.*(2.0./7.0),u0_1+C_4.*x10_4.*(5.0./7.0)+C_4.*x10_5.*(2.0./7.0),u0_2+C_5.*x11_4.*(5.0./7.0)+C_5.*x11_5.*(2.0./7.0),u0_3+C_6.*x12_4.*(5.0./7.0)+C_6.*x12_5.*(2.0./7.0),v0_1+C_1.*x7_4.*(4.0./7.0)+C_1.*x7_5.*(3.0./7.0),v0_2+C_2.*x8_4.*(4.0./7.0)+C_2.*x8_5.*(3.0./7.0),v0_3+C_3.*x9_4.*(4.0./7.0)+C_3.*x9_5.*(3.0./7.0),u0_1+C_4.*x10_4.*(4.0./7.0)+C_4.*x10_5.*(3.0./7.0));
            mt11(1,:,:) = vertcat(u0_2+C_5.*x11_4.*(4.0./7.0)+C_5.*x11_5.*(3.0./7.0),u0_3+C_6.*x12_4.*(4.0./7.0)+C_6.*x12_5.*(3.0./7.0),v0_1+C_1.*x7_4.*(3.0./7.0)+C_1.*x7_5.*(4.0./7.0),v0_2+C_2.*x8_4.*(3.0./7.0)+C_2.*x8_5.*(4.0./7.0),v0_3+C_3.*x9_4.*(3.0./7.0)+C_3.*x9_5.*(4.0./7.0),u0_1+C_4.*x10_4.*(3.0./7.0)+C_4.*x10_5.*(4.0./7.0),u0_2+C_5.*x11_4.*(3.0./7.0)+C_5.*x11_5.*(4.0./7.0),u0_3+C_6.*x12_4.*(3.0./7.0)+C_6.*x12_5.*(4.0./7.0),v0_1+C_1.*x7_4.*(2.0./7.0)+C_1.*x7_5.*(5.0./7.0),v0_2+C_2.*x8_4.*(2.0./7.0)+C_2.*x8_5.*(5.0./7.0),v0_3+C_3.*x9_4.*(2.0./7.0)+C_3.*x9_5.*(5.0./7.0),u0_1+C_4.*x10_4.*(2.0./7.0)+C_4.*x10_5.*(5.0./7.0),u0_2+C_5.*x11_4.*(2.0./7.0)+C_5.*x11_5.*(5.0./7.0),u0_3+C_6.*x12_4.*(2.0./7.0)+C_6.*x12_5.*(5.0./7.0));
            mt12(1,:,:) = vertcat(v0_1+(C_1.*x7_4)./7.0+C_1.*x7_5.*(6.0./7.0),v0_2+(C_2.*x8_4)./7.0+C_2.*x8_5.*(6.0./7.0),v0_3+(C_3.*x9_4)./7.0+C_3.*x9_5.*(6.0./7.0),u0_1+(C_4.*x10_4)./7.0+C_4.*x10_5.*(6.0./7.0),u0_2+(C_5.*x11_4)./7.0+C_5.*x11_5.*(6.0./7.0),u0_3+(C_6.*x12_4)./7.0+C_6.*x12_5.*(6.0./7.0),v0_1+C_1.*x7_5,v0_2+C_2.*x8_5,v0_3+C_3.*x9_5,u0_1+C_4.*x10_5,u0_2+C_5.*x11_5,u0_3+C_6.*x12_5,v0_1+C_1.*x7_5.*(6.0./7.0)+(C_1.*x7_6)./7.0,v0_2+C_2.*x8_5.*(6.0./7.0)+(C_2.*x8_6)./7.0,v0_3+C_3.*x9_5.*(6.0./7.0)+(C_3.*x9_6)./7.0,u0_1+C_4.*x10_5.*(6.0./7.0)+(C_4.*x10_6)./7.0,u0_2+C_5.*x11_5.*(6.0./7.0)+(C_5.*x11_6)./7.0,u0_3+C_6.*x12_5.*(6.0./7.0)+(C_6.*x12_6)./7.0);
            mt13(1,:,:) = vertcat(v0_1+C_1.*x7_5.*(5.0./7.0)+C_1.*x7_6.*(2.0./7.0),v0_2+C_2.*x8_5.*(5.0./7.0)+C_2.*x8_6.*(2.0./7.0),v0_3+C_3.*x9_5.*(5.0./7.0)+C_3.*x9_6.*(2.0./7.0),u0_1+C_4.*x10_5.*(5.0./7.0)+C_4.*x10_6.*(2.0./7.0),u0_2+C_5.*x11_5.*(5.0./7.0)+C_5.*x11_6.*(2.0./7.0),u0_3+C_6.*x12_5.*(5.0./7.0)+C_6.*x12_6.*(2.0./7.0),v0_1+C_1.*x7_5.*(4.0./7.0)+C_1.*x7_6.*(3.0./7.0),v0_2+C_2.*x8_5.*(4.0./7.0)+C_2.*x8_6.*(3.0./7.0),v0_3+C_3.*x9_5.*(4.0./7.0)+C_3.*x9_6.*(3.0./7.0),u0_1+C_4.*x10_5.*(4.0./7.0)+C_4.*x10_6.*(3.0./7.0),u0_2+C_5.*x11_5.*(4.0./7.0)+C_5.*x11_6.*(3.0./7.0),u0_3+C_6.*x12_5.*(4.0./7.0)+C_6.*x12_6.*(3.0./7.0),v0_1+C_1.*x7_5.*(3.0./7.0)+C_1.*x7_6.*(4.0./7.0),v0_2+C_2.*x8_5.*(3.0./7.0)+C_2.*x8_6.*(4.0./7.0));
            mt14(1,:,:) = vertcat(v0_3+C_3.*x9_5.*(3.0./7.0)+C_3.*x9_6.*(4.0./7.0),u0_1+C_4.*x10_5.*(3.0./7.0)+C_4.*x10_6.*(4.0./7.0),u0_2+C_5.*x11_5.*(3.0./7.0)+C_5.*x11_6.*(4.0./7.0),u0_3+C_6.*x12_5.*(3.0./7.0)+C_6.*x12_6.*(4.0./7.0),v0_1+C_1.*x7_5.*(2.0./7.0)+C_1.*x7_6.*(5.0./7.0),v0_2+C_2.*x8_5.*(2.0./7.0)+C_2.*x8_6.*(5.0./7.0),v0_3+C_3.*x9_5.*(2.0./7.0)+C_3.*x9_6.*(5.0./7.0),u0_1+C_4.*x10_5.*(2.0./7.0)+C_4.*x10_6.*(5.0./7.0),u0_2+C_5.*x11_5.*(2.0./7.0)+C_5.*x11_6.*(5.0./7.0),u0_3+C_6.*x12_5.*(2.0./7.0)+C_6.*x12_6.*(5.0./7.0),v0_1+(C_1.*x7_5)./7.0+C_1.*x7_6.*(6.0./7.0),v0_2+(C_2.*x8_5)./7.0+C_2.*x8_6.*(6.0./7.0),v0_3+(C_3.*x9_5)./7.0+C_3.*x9_6.*(6.0./7.0),u0_1+(C_4.*x10_5)./7.0+C_4.*x10_6.*(6.0./7.0));
            mt15(1,:,:) = vertcat(u0_2+(C_5.*x11_5)./7.0+C_5.*x11_6.*(6.0./7.0),u0_3+(C_6.*x12_5)./7.0+C_6.*x12_6.*(6.0./7.0),v0_1+C_1.*x7_6,v0_2+C_2.*x8_6,v0_3+C_3.*x9_6,u0_1+C_4.*x10_6,u0_2+C_5.*x11_6,u0_3+C_6.*x12_6,v0_1+C_1.*x7_6.*(6.0./7.0)+(C_1.*w1)./7.0,v0_2+C_2.*x8_6.*(6.0./7.0)+(C_2.*w2)./7.0,v0_3+C_3.*x9_6.*(6.0./7.0)+(C_3.*w3)./7.0,u0_1+C_4.*x10_6.*(6.0./7.0)+(C_4.*w4)./7.0,u0_2+C_5.*x11_6.*(6.0./7.0)+(C_5.*w5)./7.0,u0_3+C_6.*x12_6.*(6.0./7.0)+(C_6.*w6)./7.0,v0_1+C_1.*x7_6.*(5.0./7.0)+C_1.*w1.*(2.0./7.0),v0_2+C_2.*x8_6.*(5.0./7.0)+C_2.*w2.*(2.0./7.0));
            mt16(1,:,:) = vertcat(v0_3+C_3.*x9_6.*(5.0./7.0)+C_3.*w3.*(2.0./7.0),u0_1+C_4.*x10_6.*(5.0./7.0)+C_4.*w4.*(2.0./7.0),u0_2+C_5.*x11_6.*(5.0./7.0)+C_5.*w5.*(2.0./7.0),u0_3+C_6.*x12_6.*(5.0./7.0)+C_6.*w6.*(2.0./7.0),v0_1+C_1.*x7_6.*(4.0./7.0)+C_1.*w1.*(3.0./7.0),v0_2+C_2.*x8_6.*(4.0./7.0)+C_2.*w2.*(3.0./7.0),v0_3+C_3.*x9_6.*(4.0./7.0)+C_3.*w3.*(3.0./7.0),u0_1+C_4.*x10_6.*(4.0./7.0)+C_4.*w4.*(3.0./7.0),u0_2+C_5.*x11_6.*(4.0./7.0)+C_5.*w5.*(3.0./7.0),u0_3+C_6.*x12_6.*(4.0./7.0)+C_6.*w6.*(3.0./7.0),v0_1+C_1.*x7_6.*(3.0./7.0)+C_1.*w1.*(4.0./7.0),v0_2+C_2.*x8_6.*(3.0./7.0)+C_2.*w2.*(4.0./7.0));
            mt17(1,:,:) = vertcat(v0_3+C_3.*x9_6.*(3.0./7.0)+C_3.*w3.*(4.0./7.0),u0_1+C_4.*x10_6.*(3.0./7.0)+C_4.*w4.*(4.0./7.0),u0_2+C_5.*x11_6.*(3.0./7.0)+C_5.*w5.*(4.0./7.0),u0_3+C_6.*x12_6.*(3.0./7.0)+C_6.*w6.*(4.0./7.0),v0_1+C_1.*x7_6.*(2.0./7.0)+C_1.*w1.*(5.0./7.0),v0_2+C_2.*x8_6.*(2.0./7.0)+C_2.*w2.*(5.0./7.0),v0_3+C_3.*x9_6.*(2.0./7.0)+C_3.*w3.*(5.0./7.0),u0_1+C_4.*x10_6.*(2.0./7.0)+C_4.*w4.*(5.0./7.0),u0_2+C_5.*x11_6.*(2.0./7.0)+C_5.*w5.*(5.0./7.0),u0_3+C_6.*x12_6.*(2.0./7.0)+C_6.*w6.*(5.0./7.0),v0_1+(C_1.*x7_6)./7.0+C_1.*w1.*(6.0./7.0),v0_2+(C_2.*x8_6)./7.0+C_2.*w2.*(6.0./7.0));
            mt18(1,:,:) = vertcat(v0_3+(C_3.*x9_6)./7.0+C_3.*w3.*(6.0./7.0),u0_1+(C_4.*x10_6)./7.0+C_4.*w4.*(6.0./7.0),u0_2+(C_5.*x11_6)./7.0+C_5.*w5.*(6.0./7.0),u0_3+(C_6.*x12_6)./7.0+C_6.*w6.*(6.0./7.0),v0_1+C_1.*w1,v0_2+C_2.*w2,v0_3+C_3.*w3,u0_1+C_4.*w4,u0_2+C_5.*w5,u0_3+C_6.*w6);
            eta_pls = reshape(cat(2,mt1,mt2,mt3,mt4,mt5,mt6,mt7,mt8,mt9,mt10,mt11,mt12,mt13,mt14,mt15,mt16,mt17,mt18),6,43,[]);
        end

        function w = actuation_model_pressure(obj,u)
            %ACTUATION_MODEL_PRESSURE
            %    W = ACTUATION_MODEL_PRESSURE(IN1,IN2)
            
            %    This function was generated by the Symbolic Math Toolbox version 24.2.
            %    04-Nov-2024 12:33:02
            bs = size(u,2);
            t2 = -u(1,:);
            t3 = -u(2,:);
            t4 = -u(3,:);
            t5 = sqrt(3.0);
            t6 = obj.p0+t2;
            t7 = obj.p0+t3;
            t8 = obj.p0+t4;
            w = [zeros(1,bs);zeros(1,bs);-obj.A_p.*t6-obj.A_p.*t7-obj.A_p.*t8;obj.A_p.*obj.r_p.*t6.*1.224646799147353e-16+(obj.A_p.*obj.r_p.*t5.*t7)./2.0-(obj.A_p.*obj.r_p.*t5.*t8)./2.0;-obj.A_p.*obj.r_p.*t6+(obj.A_p.*obj.r_p.*t7)./2.0+(obj.A_p.*obj.r_p.*t8)./2.0;zeros(1,bs)];
        end

        function out_ft_sensor = out_ft_sensor(~,in1,in2,in3,supp_vals)
            %OUT_FT_SENSOR
            %    OUT_FT_SENSOR = OUT_FT_SENSOR(IN1,IN2,IN3)
            
            %    This function was generated by the Symbolic Math Toolbox version 8.7.
            %    22-Oct-2024 16:39:39
            %    Batch version (samples as columns); C_4 = C_5 from supp_vals,
            %    external tip wrench (export input IN3) removed as in model/TRO_N_7
            
            C_1 = in3(:,1);
            C_2 = in3(:,2);
            C_3 = in3(:,3);
            if isempty(supp_vals)
                C_4 = in3(:,4);
                C_5 = in3(:,5);
            else
                C_4 = supp_vals(:,1)';
                C_5 = supp_vals(:,1)';
            end
            C_6 = in3(:,6);
            D_1 = in3(:,7);
            D_2 = in3(:,8);
            D_3 = in3(:,9);
            D_4 = in3(:,10);
            D_5 = in3(:,11);
            D_6 = in3(:,12);
            L = in3(:,13);
            u0_1 = in3(:,21);
            u0_2 = in3(:,22);
            u0_3 = in3(:,23);
            v0_1 = in3(:,24);
            v0_2 = in3(:,25);
            v0_3 = in3(:,26);
            w1 = in2(1,:);
            w2 = in2(2,:);
            w3 = in2(3,:);
            w4 = in2(4,:);
            w5 = in2(5,:);
            w6 = in2(6,:);
            x10_1 = in1(4,:);
            x10_2 = in1(16,:);
            x10_3 = in1(28,:);
            x10_4 = in1(40,:);
            x10_5 = in1(52,:);
            x10_6 = in1(64,:);
            x11_1 = in1(5,:);
            x11_2 = in1(17,:);
            x11_3 = in1(29,:);
            x11_4 = in1(41,:);
            x11_5 = in1(53,:);
            x11_6 = in1(65,:);
            x12_1 = in1(6,:);
            x12_2 = in1(18,:);
            x12_3 = in1(30,:);
            x12_4 = in1(42,:);
            x12_5 = in1(54,:);
            x12_6 = in1(66,:);
            x1_2 = in1(7,:);
            x1_3 = in1(19,:);
            x1_4 = in1(31,:);
            x1_5 = in1(43,:);
            x1_6 = in1(55,:);
            x1_7 = in1(67,:);
            x2_2 = in1(8,:);
            x2_3 = in1(20,:);
            x2_4 = in1(32,:);
            x2_5 = in1(44,:);
            x2_6 = in1(56,:);
            x2_7 = in1(68,:);
            x3_2 = in1(9,:);
            x3_3 = in1(21,:);
            x3_4 = in1(33,:);
            x3_5 = in1(45,:);
            x3_6 = in1(57,:);
            x3_7 = in1(69,:);
            x4_2 = in1(10,:);
            x4_3 = in1(22,:);
            x4_4 = in1(34,:);
            x4_5 = in1(46,:);
            x4_6 = in1(58,:);
            x4_7 = in1(70,:);
            x5_2 = in1(11,:);
            x5_3 = in1(23,:);
            x5_4 = in1(35,:);
            x5_5 = in1(47,:);
            x5_6 = in1(59,:);
            x5_7 = in1(71,:);
            x6_2 = in1(12,:);
            x6_3 = in1(24,:);
            x6_4 = in1(36,:);
            x6_5 = in1(48,:);
            x6_6 = in1(60,:);
            x6_7 = in1(72,:);
            x7_1 = in1(1,:);
            x7_2 = in1(13,:);
            x7_3 = in1(25,:);
            x7_4 = in1(37,:);
            x7_5 = in1(49,:);
            x7_6 = in1(61,:);
            x8_1 = in1(2,:);
            x8_2 = in1(14,:);
            x8_3 = in1(26,:);
            x8_4 = in1(38,:);
            x8_5 = in1(50,:);
            x8_6 = in1(62,:);
            x9_1 = in1(3,:);
            x9_2 = in1(15,:);
            x9_3 = in1(27,:);
            x9_4 = in1(39,:);
            x9_5 = in1(51,:);
            x9_6 = in1(63,:);
            et1 = ((x6_2-(L.*((u0_2.*x4_2)./2.0-(u0_1.*x5_2)./2.0-(C_4.*x5_2.*(w4./2.0+x10_1./2.0+x10_2./2.0))./2.0+(C_5.*x4_2.*(w5./2.0+x11_1./2.0+x11_2./2.0))./2.0))./6.0).*1.2e+1)./(C_6.*L)+((x6_6-x6_7+(L.*(u0_2.*(x4_6./2.0+x4_7./2.0)-u0_1.*(x5_6./2.0+x5_7./2.0)+C_5.*(x4_6./2.0+x4_7./2.0).*(w5./2.0+x11_6./2.0)-C_4.*(x5_6./2.0+x5_7./2.0).*(w4./2.0+x10_6./2.0)))./6.0).*1.2e+1)./(C_6.*L);
            et2 = ((x6_2-x6_3+(L.*(u0_2.*(x4_2./2.0+x4_3./2.0)-u0_1.*(x5_2./2.0+x5_3./2.0)-C_4.*(x10_2./2.0+x10_3./2.0).*(x5_2./2.0+x5_3./2.0)+C_5.*(x11_2./2.0+x11_3./2.0).*(x4_2./2.0+x4_3./2.0)))./6.0).*1.2e+1)./(C_6.*L);
            et3 = ((x6_3-x6_4+(L.*(u0_2.*(x4_3./2.0+x4_4./2.0)-u0_1.*(x5_3./2.0+x5_4./2.0)-C_4.*(x10_3./2.0+x10_4./2.0).*(x5_3./2.0+x5_4./2.0)+C_5.*(x11_3./2.0+x11_4./2.0).*(x4_3./2.0+x4_4./2.0)))./6.0).*-1.2e+1)./(C_6.*L);
            et4 = ((x6_4-x6_5+(L.*(u0_2.*(x4_4./2.0+x4_5./2.0)-u0_1.*(x5_4./2.0+x5_5./2.0)-C_4.*(x10_4./2.0+x10_5./2.0).*(x5_4./2.0+x5_5./2.0)+C_5.*(x11_4./2.0+x11_5./2.0).*(x4_4./2.0+x4_5./2.0)))./6.0).*1.2e+1)./(C_6.*L);
            et5 = ((x6_5-x6_6+(L.*(u0_2.*(x4_5./2.0+x4_6./2.0)-u0_1.*(x5_5./2.0+x5_6./2.0)-C_4.*(x10_5./2.0+x10_6./2.0).*(x5_5./2.0+x5_6./2.0)+C_5.*(x11_5./2.0+x11_6./2.0).*(x4_5./2.0+x4_6./2.0)))./6.0).*-1.2e+1)./(C_6.*L);
            et6 = ((x5_2+(L.*((u0_3.*x4_2)./2.0-(u0_1.*x6_2)./2.0-(C_4.*x6_2.*(w4./2.0+x10_1./2.0+x10_2./2.0))./2.0+(C_6.*x4_2.*(w6./2.0+x12_1./2.0+x12_2./2.0))./2.0))./6.0).*1.2e+1)./(C_5.*L)-((-x5_6+x5_7+(L.*(u0_3.*(x4_6./2.0+x4_7./2.0)-u0_1.*(x6_6./2.0+x6_7./2.0)+C_6.*(x4_6./2.0+x4_7./2.0).*(w6./2.0+x12_6./2.0)-C_4.*(x6_6./2.0+x6_7./2.0).*(w4./2.0+x10_6./2.0)))./6.0).*1.2e+1)./(C_5.*L);
            et7 = ((-x5_2+x5_3+(L.*(u0_3.*(x4_2./2.0+x4_3./2.0)-u0_1.*(x6_2./2.0+x6_3./2.0)-C_4.*(x10_2./2.0+x10_3./2.0).*(x6_2./2.0+x6_3./2.0)+C_6.*(x12_2./2.0+x12_3./2.0).*(x4_2./2.0+x4_3./2.0)))./6.0).*-1.2e+1)./(C_5.*L);
            et8 = ((-x5_3+x5_4+(L.*(u0_3.*(x4_3./2.0+x4_4./2.0)-u0_1.*(x6_3./2.0+x6_4./2.0)-C_4.*(x10_3./2.0+x10_4./2.0).*(x6_3./2.0+x6_4./2.0)+C_6.*(x12_3./2.0+x12_4./2.0).*(x4_3./2.0+x4_4./2.0)))./6.0).*1.2e+1)./(C_5.*L);
            et9 = ((-x5_4+x5_5+(L.*(u0_3.*(x4_4./2.0+x4_5./2.0)-u0_1.*(x6_4./2.0+x6_5./2.0)-C_4.*(x10_4./2.0+x10_5./2.0).*(x6_4./2.0+x6_5./2.0)+C_6.*(x12_4./2.0+x12_5./2.0).*(x4_4./2.0+x4_5./2.0)))./6.0).*-1.2e+1)./(C_5.*L);
            et10 = ((-x5_5+x5_6+(L.*(u0_3.*(x4_5./2.0+x4_6./2.0)-u0_1.*(x6_5./2.0+x6_6./2.0)-C_4.*(x10_5./2.0+x10_6./2.0).*(x6_5./2.0+x6_6./2.0)+C_6.*(x12_5./2.0+x12_6./2.0).*(x4_5./2.0+x4_6./2.0)))./6.0).*1.2e+1)./(C_5.*L);
            et11 = ((x4_2-(L.*((u0_3.*x5_2)./2.0-(u0_2.*x6_2)./2.0-(C_5.*x6_2.*(w5./2.0+x11_1./2.0+x11_2./2.0))./2.0+(C_6.*x5_2.*(w6./2.0+x12_1./2.0+x12_2./2.0))./2.0))./6.0).*1.2e+1)./(C_4.*L)+((x4_6-x4_7+(L.*(u0_3.*(x5_6./2.0+x5_7./2.0)-u0_2.*(x6_6./2.0+x6_7./2.0)+C_6.*(x5_6./2.0+x5_7./2.0).*(w6./2.0+x12_6./2.0)-C_5.*(x6_6./2.0+x6_7./2.0).*(w5./2.0+x11_6./2.0)))./6.0).*1.2e+1)./(C_4.*L);
            et12 = ((x4_2-x4_3+(L.*(u0_3.*(x5_2./2.0+x5_3./2.0)-u0_2.*(x6_2./2.0+x6_3./2.0)-C_5.*(x11_2./2.0+x11_3./2.0).*(x6_2./2.0+x6_3./2.0)+C_6.*(x12_2./2.0+x12_3./2.0).*(x5_2./2.0+x5_3./2.0)))./6.0).*1.2e+1)./(C_4.*L);
            et13 = ((x4_3-x4_4+(L.*(u0_3.*(x5_3./2.0+x5_4./2.0)-u0_2.*(x6_3./2.0+x6_4./2.0)-C_5.*(x11_3./2.0+x11_4./2.0).*(x6_3./2.0+x6_4./2.0)+C_6.*(x12_3./2.0+x12_4./2.0).*(x5_3./2.0+x5_4./2.0)))./6.0).*-1.2e+1)./(C_4.*L);
            et14 = ((x4_4-x4_5+(L.*(u0_3.*(x5_4./2.0+x5_5./2.0)-u0_2.*(x6_4./2.0+x6_5./2.0)-C_5.*(x11_4./2.0+x11_5./2.0).*(x6_4./2.0+x6_5./2.0)+C_6.*(x12_4./2.0+x12_5./2.0).*(x5_4./2.0+x5_5./2.0)))./6.0).*1.2e+1)./(C_4.*L);
            et15 = ((x4_5-x4_6+(L.*(u0_3.*(x5_5./2.0+x5_6./2.0)-u0_2.*(x6_5./2.0+x6_6./2.0)-C_5.*(x11_5./2.0+x11_6./2.0).*(x6_5./2.0+x6_6./2.0)+C_6.*(x12_5./2.0+x12_6./2.0).*(x5_5./2.0+x5_6./2.0)))./6.0).*-1.2e+1)./(C_4.*L);
            et16 = ((x3_6-x3_7+(L.*(u0_2.*(x1_6./2.0+x1_7./2.0)-u0_1.*(x2_6./2.0+x2_7./2.0)+v0_2.*(x4_6./2.0+x4_7./2.0)-v0_1.*(x5_6./2.0+x5_7./2.0)+C_5.*(x1_6./2.0+x1_7./2.0).*(w5./2.0+x11_6./2.0)-C_4.*(x2_6./2.0+x2_7./2.0).*(w4./2.0+x10_6./2.0)+C_2.*(x4_6./2.0+x4_7./2.0).*(w2./2.0+x8_6./2.0)-C_1.*(x5_6./2.0+x5_7./2.0).*(w1./2.0+x7_6./2.0)))./6.0).*1.2e+1)./(C_3.*L);
            et17 = ((x3_2-(L.*((u0_2.*x1_2)./2.0-(u0_1.*x2_2)./2.0+(v0_2.*x4_2)./2.0-(v0_1.*x5_2)./2.0-(C_4.*x2_2.*(w4./2.0+x10_1./2.0+x10_2./2.0))./2.0+(C_5.*x1_2.*(w5./2.0+x11_1./2.0+x11_2./2.0))./2.0-(C_1.*x5_2.*(w1./2.0+x7_1./2.0+x7_2./2.0))./2.0+(C_2.*x4_2.*(w2./2.0+x8_1./2.0+x8_2./2.0))./2.0))./6.0).*1.2e+1)./(C_3.*L);
            et18 = ((x3_2-x3_3+(L.*(u0_2.*(x1_2./2.0+x1_3./2.0)-u0_1.*(x2_2./2.0+x2_3./2.0)+v0_2.*(x4_2./2.0+x4_3./2.0)-v0_1.*(x5_2./2.0+x5_3./2.0)-C_4.*(x10_2./2.0+x10_3./2.0).*(x2_2./2.0+x2_3./2.0)+C_5.*(x11_2./2.0+x11_3./2.0).*(x1_2./2.0+x1_3./2.0)-C_1.*(x5_2./2.0+x5_3./2.0).*(x7_2./2.0+x7_3./2.0)+C_2.*(x4_2./2.0+x4_3./2.0).*(x8_2./2.0+x8_3./2.0)))./6.0).*1.2e+1)./(C_3.*L);
            et19 = ((x3_3-x3_4+(L.*(u0_2.*(x1_3./2.0+x1_4./2.0)-u0_1.*(x2_3./2.0+x2_4./2.0)+v0_2.*(x4_3./2.0+x4_4./2.0)-v0_1.*(x5_3./2.0+x5_4./2.0)-C_4.*(x10_3./2.0+x10_4./2.0).*(x2_3./2.0+x2_4./2.0)+C_5.*(x11_3./2.0+x11_4./2.0).*(x1_3./2.0+x1_4./2.0)-C_1.*(x5_3./2.0+x5_4./2.0).*(x7_3./2.0+x7_4./2.0)+C_2.*(x4_3./2.0+x4_4./2.0).*(x8_3./2.0+x8_4./2.0)))./6.0).*-1.2e+1)./(C_3.*L);
            et20 = ((x3_4-x3_5+(L.*(u0_2.*(x1_4./2.0+x1_5./2.0)-u0_1.*(x2_4./2.0+x2_5./2.0)+v0_2.*(x4_4./2.0+x4_5./2.0)-v0_1.*(x5_4./2.0+x5_5./2.0)-C_4.*(x10_4./2.0+x10_5./2.0).*(x2_4./2.0+x2_5./2.0)+C_5.*(x11_4./2.0+x11_5./2.0).*(x1_4./2.0+x1_5./2.0)-C_1.*(x5_4./2.0+x5_5./2.0).*(x7_4./2.0+x7_5./2.0)+C_2.*(x4_4./2.0+x4_5./2.0).*(x8_4./2.0+x8_5./2.0)))./6.0).*1.2e+1)./(C_3.*L);
            et21 = ((x3_5-x3_6+(L.*(u0_2.*(x1_5./2.0+x1_6./2.0)-u0_1.*(x2_5./2.0+x2_6./2.0)+v0_2.*(x4_5./2.0+x4_6./2.0)-v0_1.*(x5_5./2.0+x5_6./2.0)-C_4.*(x10_5./2.0+x10_6./2.0).*(x2_5./2.0+x2_6./2.0)+C_5.*(x11_5./2.0+x11_6./2.0).*(x1_5./2.0+x1_6./2.0)-C_1.*(x5_5./2.0+x5_6./2.0).*(x7_5./2.0+x7_6./2.0)+C_2.*(x4_5./2.0+x4_6./2.0).*(x8_5./2.0+x8_6./2.0)))./6.0).*-1.2e+1)./(C_3.*L);
            et22 = ((-x2_6+x2_7+(L.*(u0_3.*(x1_6./2.0+x1_7./2.0)-u0_1.*(x3_6./2.0+x3_7./2.0)+v0_3.*(x4_6./2.0+x4_7./2.0)-v0_1.*(x6_6./2.0+x6_7./2.0)+C_6.*(x1_6./2.0+x1_7./2.0).*(w6./2.0+x12_6./2.0)-C_4.*(x3_6./2.0+x3_7./2.0).*(w4./2.0+x10_6./2.0)+C_3.*(x4_6./2.0+x4_7./2.0).*(w3./2.0+x9_6./2.0)-C_1.*(x6_6./2.0+x6_7./2.0).*(w1./2.0+x7_6./2.0)))./6.0).*1.2e+1)./(C_2.*L);
            et23 = ((x2_2+(L.*((u0_3.*x1_2)./2.0-(u0_1.*x3_2)./2.0+(v0_3.*x4_2)./2.0-(v0_1.*x6_2)./2.0-(C_4.*x3_2.*(w4./2.0+x10_1./2.0+x10_2./2.0))./2.0+(C_6.*x1_2.*(w6./2.0+x12_1./2.0+x12_2./2.0))./2.0-(C_1.*x6_2.*(w1./2.0+x7_1./2.0+x7_2./2.0))./2.0+(C_3.*x4_2.*(w3./2.0+x9_1./2.0+x9_2./2.0))./2.0))./6.0).*-1.2e+1)./(C_2.*L);
            et24 = ((-x2_2+x2_3+(L.*(u0_3.*(x1_2./2.0+x1_3./2.0)-u0_1.*(x3_2./2.0+x3_3./2.0)+v0_3.*(x4_2./2.0+x4_3./2.0)-v0_1.*(x6_2./2.0+x6_3./2.0)-C_4.*(x10_2./2.0+x10_3./2.0).*(x3_2./2.0+x3_3./2.0)+C_6.*(x12_2./2.0+x12_3./2.0).*(x1_2./2.0+x1_3./2.0)-C_1.*(x6_2./2.0+x6_3./2.0).*(x7_2./2.0+x7_3./2.0)+C_3.*(x4_2./2.0+x4_3./2.0).*(x9_2./2.0+x9_3./2.0)))./6.0).*1.2e+1)./(C_2.*L);
            et25 = ((-x2_3+x2_4+(L.*(u0_3.*(x1_3./2.0+x1_4./2.0)-u0_1.*(x3_3./2.0+x3_4./2.0)+v0_3.*(x4_3./2.0+x4_4./2.0)-v0_1.*(x6_3./2.0+x6_4./2.0)-C_4.*(x10_3./2.0+x10_4./2.0).*(x3_3./2.0+x3_4./2.0)+C_6.*(x12_3./2.0+x12_4./2.0).*(x1_3./2.0+x1_4./2.0)-C_1.*(x6_3./2.0+x6_4./2.0).*(x7_3./2.0+x7_4./2.0)+C_3.*(x4_3./2.0+x4_4./2.0).*(x9_3./2.0+x9_4./2.0)))./6.0).*-1.2e+1)./(C_2.*L);
            et26 = ((-x2_4+x2_5+(L.*(u0_3.*(x1_4./2.0+x1_5./2.0)-u0_1.*(x3_4./2.0+x3_5./2.0)+v0_3.*(x4_4./2.0+x4_5./2.0)-v0_1.*(x6_4./2.0+x6_5./2.0)-C_4.*(x10_4./2.0+x10_5./2.0).*(x3_4./2.0+x3_5./2.0)+C_6.*(x12_4./2.0+x12_5./2.0).*(x1_4./2.0+x1_5./2.0)-C_1.*(x6_4./2.0+x6_5./2.0).*(x7_4./2.0+x7_5./2.0)+C_3.*(x4_4./2.0+x4_5./2.0).*(x9_4./2.0+x9_5./2.0)))./6.0).*1.2e+1)./(C_2.*L);
            et27 = ((-x2_5+x2_6+(L.*(u0_3.*(x1_5./2.0+x1_6./2.0)-u0_1.*(x3_5./2.0+x3_6./2.0)+v0_3.*(x4_5./2.0+x4_6./2.0)-v0_1.*(x6_5./2.0+x6_6./2.0)-C_4.*(x10_5./2.0+x10_6./2.0).*(x3_5./2.0+x3_6./2.0)+C_6.*(x12_5./2.0+x12_6./2.0).*(x1_5./2.0+x1_6./2.0)-C_1.*(x6_5./2.0+x6_6./2.0).*(x7_5./2.0+x7_6./2.0)+C_3.*(x4_5./2.0+x4_6./2.0).*(x9_5./2.0+x9_6./2.0)))./6.0).*-1.2e+1)./(C_2.*L);
            et28 = ((x1_6-x1_7+(L.*(u0_3.*(x2_6./2.0+x2_7./2.0)-u0_2.*(x3_6./2.0+x3_7./2.0)+v0_3.*(x5_6./2.0+x5_7./2.0)-v0_2.*(x6_6./2.0+x6_7./2.0)+C_6.*(x2_6./2.0+x2_7./2.0).*(w6./2.0+x12_6./2.0)-C_5.*(x3_6./2.0+x3_7./2.0).*(w5./2.0+x11_6./2.0)+C_3.*(x5_6./2.0+x5_7./2.0).*(w3./2.0+x9_6./2.0)-C_2.*(x6_6./2.0+x6_7./2.0).*(w2./2.0+x8_6./2.0)))./6.0).*1.2e+1)./(C_1.*L);
            et29 = ((x1_2-(L.*((u0_3.*x2_2)./2.0-(u0_2.*x3_2)./2.0+(v0_3.*x5_2)./2.0-(v0_2.*x6_2)./2.0-(C_5.*x3_2.*(w5./2.0+x11_1./2.0+x11_2./2.0))./2.0+(C_6.*x2_2.*(w6./2.0+x12_1./2.0+x12_2./2.0))./2.0-(C_2.*x6_2.*(w2./2.0+x8_1./2.0+x8_2./2.0))./2.0+(C_3.*x5_2.*(w3./2.0+x9_1./2.0+x9_2./2.0))./2.0))./6.0).*1.2e+1)./(C_1.*L);
            et30 = ((x1_2-x1_3+(L.*(u0_3.*(x2_2./2.0+x2_3./2.0)-u0_2.*(x3_2./2.0+x3_3./2.0)+v0_3.*(x5_2./2.0+x5_3./2.0)-v0_2.*(x6_2./2.0+x6_3./2.0)-C_5.*(x11_2./2.0+x11_3./2.0).*(x3_2./2.0+x3_3./2.0)+C_6.*(x12_2./2.0+x12_3./2.0).*(x2_2./2.0+x2_3./2.0)-C_2.*(x6_2./2.0+x6_3./2.0).*(x8_2./2.0+x8_3./2.0)+C_3.*(x5_2./2.0+x5_3./2.0).*(x9_2./2.0+x9_3./2.0)))./6.0).*1.2e+1)./(C_1.*L);
            et31 = ((x1_3-x1_4+(L.*(u0_3.*(x2_3./2.0+x2_4./2.0)-u0_2.*(x3_3./2.0+x3_4./2.0)+v0_3.*(x5_3./2.0+x5_4./2.0)-v0_2.*(x6_3./2.0+x6_4./2.0)-C_5.*(x11_3./2.0+x11_4./2.0).*(x3_3./2.0+x3_4./2.0)+C_6.*(x12_3./2.0+x12_4./2.0).*(x2_3./2.0+x2_4./2.0)-C_2.*(x6_3./2.0+x6_4./2.0).*(x8_3./2.0+x8_4./2.0)+C_3.*(x5_3./2.0+x5_4./2.0).*(x9_3./2.0+x9_4./2.0)))./6.0).*-1.2e+1)./(C_1.*L);
            et32 = ((x1_4-x1_5+(L.*(u0_3.*(x2_4./2.0+x2_5./2.0)-u0_2.*(x3_4./2.0+x3_5./2.0)+v0_3.*(x5_4./2.0+x5_5./2.0)-v0_2.*(x6_4./2.0+x6_5./2.0)-C_5.*(x11_4./2.0+x11_5./2.0).*(x3_4./2.0+x3_5./2.0)+C_6.*(x12_4./2.0+x12_5./2.0).*(x2_4./2.0+x2_5./2.0)-C_2.*(x6_4./2.0+x6_5./2.0).*(x8_4./2.0+x8_5./2.0)+C_3.*(x5_4./2.0+x5_5./2.0).*(x9_4./2.0+x9_5./2.0)))./6.0).*1.2e+1)./(C_1.*L);
            et33 = ((x1_5-x1_6+(L.*(u0_3.*(x2_5./2.0+x2_6./2.0)-u0_2.*(x3_5./2.0+x3_6./2.0)+v0_3.*(x5_5./2.0+x5_6./2.0)-v0_2.*(x6_5./2.0+x6_6./2.0)-C_5.*(x11_5./2.0+x11_6./2.0).*(x3_5./2.0+x3_6./2.0)+C_6.*(x12_5./2.0+x12_6./2.0).*(x2_5./2.0+x2_6./2.0)-C_2.*(x6_5./2.0+x6_6./2.0).*(x8_5./2.0+x8_6./2.0)+C_3.*(x5_5./2.0+x5_6./2.0).*(x9_5./2.0+x9_6./2.0)))./6.0).*-1.2e+1)./(C_1.*L);
            out_ft_sensor = [x7_1+D_1.*(et28+et29+et30+et31+et32+et33);x8_1-D_2.*(et22+et23+et24+et25+et26+et27);x9_1+D_3.*(et16+et17+et18+et19+et20+et21);x10_1+D_4.*(et11+et12+et13+et14+et15);x11_1+D_5.*(et6+et7+et8+et9+et10);x12_1+D_6.*(et1+et2+et3+et4+et5)];
        end

    end
end