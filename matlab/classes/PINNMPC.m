classdef PINNMPC < PINNModel
    %PINNMPC MPC class that interfaces with the NEMPC in Python
    % (python/NEMPC_matlab_interface.py). MATLAB must use the Python
    % environment with torch (pyenv, see README).
    
    properties
        x_set
        t_sample_mpc
        N_pred
        N_u
        dimU % is equal to udim if u_degree == 0 
        pinn_folder
        sval
        u0_mpc
        x0_mpc
        u_pred
        Q
        sim
        t_curr
        NEMPC_params
        NEMPC
        u_mpc
        use_pressure_model = false
        use_position_control = false
    end
    
    methods
        function obj = PINNMPC(imported_pinn_file,t_sample_mpc,N_pred,t_total)
            obj@PINNModel(imported_pinn_file,t_total);
            obj.pinn_folder = char(imported_pinn_file);
            obj.t_sample_mpc = t_sample_mpc;
            obj.N_pred = N_pred;
            obj.dimU = 3;
        end
        function Q = calcQ(obj,vel_weight,angvel_weight,force_weight,force_weight_z,moment_weight)
            sigma_vel = vel_weight*ones(1,3);
            sigma_angvel = angvel_weight*ones(1,3);
            sigma_internal_force = [force_weight*ones(1,2), force_weight_z];
            sigma_internal_moment = moment_weight*ones(1,3);
            
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
            Q = temp;
            obj.Q = diag(temp);
        end

        function Q = calcQ_with_rescaling(obj,vel_weight,angvel_weight,force_weight,force_weight_z,moment_weight)
            sigma_vel = vel_weight*ones(1,3);
            sigma_angvel = angvel_weight*ones(1,3);
            sigma_internal_force = [force_weight*ones(1,2), force_weight_z];
            sigma_internal_moment = moment_weight*ones(1,3);
            
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
            m_x_array = struct2array(obj.m_x);
            Q = temp./m_x_array(1:obj.xdim);
            obj.Q = diag(temp);
        end

        function x_set = calcSetpoint(obj,cdat,input_pressures)
            obj.sim = SimulationSetup(cdat,obj.t_total);
            obj.sim.setGravity([0,0,9.81]);
            obj.sim.setReferenceLength(obj.L);
            obj.sim.M_tip = obj.M_tip;
            obj.sim.DM = obj.D;
            obj.sim.CM = obj.C;
            obj.sim.MM = obj.M;
            obj.sim.compressiongain = obj.compressiongain;
            obj.sim.updateParameters();
            if isrow(input_pressures)
                opts_fsolve = optimoptions('fsolve', 'Display','none');
                obj.sim.setInputConstant(input_pressures);
                x_set = fsolve(@(x)obj.sim.c(0,x,zeros(size(x))),obj.sim.x0,opts_fsolve);
            else
                obj.t = 0:obj.t_sample:(obj.t_total);
                obj.sim.setInputData(obj.t,input_pressures);
                x_set = obj.sim.run(obj.t);
            end
        end

        function obj = init_sim(obj,cdat)
            obj.sim = SimulationSetup(cdat);
            obj.sim.setGravity([0,0,9.81]);
            obj.sim.setReferenceLength(obj.L);
            obj.sim.M_tip = obj.M_tip;
            obj.sim.DM = obj.D;
            obj.sim.CM = obj.C;
            obj.sim.MM = obj.M;
            obj.sim.compressiongain = obj.compressiongain;
            obj.sim.updateParameters();
        end

        function [x_next,y_next] = simStep(obj,X_curr,input_pressures,t_step)
            sim_ = copy(obj.sim);
            sim_.setBendingCompliance(X_curr(:,(obj.xdim+1)));
            sim_.setInputData([0 t_step]',repmat(input_pressures,2,1));
            opts = odeset('Jacobian',@sim_.j_f,'NormControl','on');
            [~,x_sim] = ode23s(@sim_.f,[0 t_step]',X_curr(:,1:obj.xdim),opts);
            x_next = x_sim(end,:);

            T_EE = sim_.EE(t_step,x_next');
            y_next = T_EE(1:3,4)';
        end

        
        function obj = init_mpc(obj,NEMPC_params,NEMPC_path,use_position_control)
            obj.use_position_control = use_position_control;
            obj.NEMPC_params = NEMPC_params;
            obj.u0_mpc = normalize_list(10^5*ones(obj.dimU,1),obj.props_names(obj.xdim+obj.sdim+1:obj.xdim+obj.sdim+obj.dimU),obj.m_x,obj.b_x,false,false,true);
            obj.t_curr = 0; 
            obj.u_pred = repmat(obj.u0_mpc,obj.N_pred,1); % property to later store predictions
            obj.sval = normalize_list(obj.supp_vals,obj.props_names((obj.xdim+1):(obj.xdim+obj.sdim)),obj.m_x,obj.b_x,false,false,true);
            % Initiate NEMPC
            if count(py.sys.path,NEMPC_path) == 0
                insert(py.sys.path,int32(0),NEMPC_path);
            end
            mod = py.importlib.import_module('NEMPC_matlab_interface');
            py.importlib.reload(mod);
            mod = py.importlib.import_module('NonlinearEMPC_pure_torch');
            py.importlib.reload(mod);
            obj.NEMPC = py.NEMPC_matlab_interface.NEMPC_wrapper(obj.pinn_folder,obj.NEMPC_params);
        end

        function u_next = optimize_first(obj,x_set,xEE_meas,debugging)
            obj.x0_mpc = normalize_list(obj.X0(1:obj.xdim),obj.props_names(1:obj.xdim),obj.m_x,obj.b_x,false,false,true);
            u_next = optimize(obj, obj.x0_mpc,obj.sval,x_set,obj.u0_mpc,xEE_meas, debugging);
        end
        
        
        function u_next = optimize(obj, x,supp_vals,x_set,u_meas,xEE_meas, debugging)
            % Calculate MPC step in Python
            u_pred_ndarray = obj.NEMPC.step(py.numpy.array(x),py.numpy.array(obj.u_pred(1,:)),py.numpy.array(supp_vals),py.numpy.array(x_set),py.numpy.array(u_meas),py.numpy.array(xEE_meas));
            obj.u_pred = double(u_pred_ndarray);      
            % disp(obj.u_pred)
            u_next = obj.u_pred(1,:);

            if debugging
                u_plot = normalize_list(obj.u_pred,obj.props_names(obj.xdim+obj.sdim+1:obj.xdim+obj.sdim+obj.dimU),obj.m_x,obj.b_x,false,false,false);
                figure(7)
                for i = 1:obj.dimU
                    subplot(obj.dimU,1,i)
                    if obj.u_degree == 2
                        plot(obj.t_curr+obj.t_sample_mpc/2*(0:1:(obj.N_pred*2-1)),u_plot(:,i))
                    else
                        plot(obj.t_curr+obj.t_sample_mpc*(0:1:obj.N_pred-1),u_plot(:,i))
                    end
                    hold on
                    ylabel("$u_"+num2str(i)+"$")
                end
                xlabel("t")
            end
            obj.t_curr = obj.t_curr + obj.t_sample_mpc;
        end

        function storeDataForPlotting(obj,x_mpc,u_mpc,x_ref,cdat)
            obj.t = 0:obj.t_sample:obj.t_sample*(length(u_mpc)-1);
            obj.sim_res = SimulationResults(obj.t,x_ref(1:length(obj.t),:),[],cdat,obj.sim);
            obj.u_mpc = u_mpc;
            obj.y = x_mpc;
            [rEE,orientation] = obj.EE(x_mpc,u_mpc,obj.supp_vals);
            obj.rEE = rEE;
            obj.orientation = orientation;
            obj.force_torque = FTS(obj,x_mpc,u_mpc,obj.supp_vals);
            obj.ang_vel = getAngularVeloEE(obj,x_mpc);
        end
    end
end