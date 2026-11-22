classdef SimulationSetup < ModelSetup
    % Cosserat rod model without the transition condition at the point loads
    % (12*(N-1) states), used with the export model/TRO_N_7 and
    % cdat = model_config().
    %SIMULATIONSETUP This class sums up all simulation parameters and
    %returns all function handles (such as state equations, output equations
    % input equations)
    
    properties
        T_TOTAL % total simulation time
        param % parameters for most functions from generate_equations
        param_gravity % parameters for distr. gravity load
        CM  % compliance matrix
        DM  % damping matrix (in s)
        MM  % mass matrix in kg/m
        M_tip = 0; % mass at tip (in kg/m)
        M_base % mass of ft_sensor and Dragonskin cap (just gets added to z-force)
        L   % ref. length
        xi0 % reference local strains
        radius % geometric radius of system
        rho % corrected density
        % r0  % base point in m
        R0 = eye(3); % rod's base orientation
        input % input function handle generating wrenches
        pressure_handle % function handle for pressure signal in bar
        gravityUsesMean = 0;
        grav
        deta_ds    % derivative of strains w.r.t s (functionhandle)
        compressiongain = 1.61;
        fts_pos_offset % Offset of FT-sensor to bending body in m
        EE_pos_offset % Offset of rod calculation EE and measured EE
        p0 = 100000; % environment pressure
        A_p  % effective area of air chamber
        r_p = 12/1000; % radius to center
        xi_base
    end
    
    methods
        function obj = SimulationSetup(config, T)
            % global simulation parameters
            obj@ModelSetup(config);
            try
                obj.T_TOTAL = T;        % Duration in sS
            catch
                warning("Did not set T_TOTAL.")
            end
            obj.setParametersGenericSoftRobot;
            obj.setGravity([0,0,9.81]);
            obj.setInputsZero;
            % displayParameters(obj.param);
        end
        
        function obj = setParameters(obj,param, tipMass)
            %% SETPARAMETERS
            %
            % ORDER:
            % C1 C2 C3 C4 C5 C6 D1 D2 D3 D4 D5 D6 Length Radius
            %
            % UNITS:
            %   - C1-3 in m/N , C4-6 rad/Nm
            %   - D1-6 in s
            %   - Length in m
            %   - Radius in m
            %
            % All other parameters are fixed in this function.
            
            % Material parameters
            % fixed
%             rho = 1080; % kg/m3
            
            % Geometry
            obj.L = param(end-1);       % Ref. Length
            
            obj.radius = param(end);  % radius of cross-section
            
            % fixed
            u0_1 = 0;       % Ref. curvature/twist
            u0_2 = 0;
            u0_3 = 0;
            v0_1 = 0;       % Ref. shear/dilatation
            v0_2 = 0;
            v0_3 = 1;
            obj.xi0 = [v0_1,v0_2,v0_3,u0_1,u0_2,u0_3];
            % Material
            obj.MM = computeMass(obj.radius,obj.rho); % mass matrix
            obj.CM = diag(param(1:6));  % compliance matrix
            obj.DM = diag(param(7:12)); % damping matrix
            obj.setTipMassInKg(tipMass);
            updateParameters(obj);
        end
        
        function obj = setParametersGenericSoftRobot(obj)
            % Cosserat rod parameters
            % Parameters in ODE (adapt if necessary)
            
            
            % Geometry
            obj.L = 0.11;       % Ref. Length of Ecoflex
            obj.radius = 0.02115;  % radius of cross-section
            u0_1 = 0;       % Ref. curvature/twist
            u0_2 = 0;
            u0_3 = 0;
            v0_1 = 0;       % Ref. shear/dilatation
            v0_2 = 0;
            v0_3 = 1;
            obj.xi0 = [v0_1,v0_2,v0_3,u0_1,u0_2,u0_3];
            % obj.r0  = [0;0;0.024]; % base couple point in m
            obj.fts_pos_offset = [0;0;0.024]; % Offset of FT-sensor to bending body in m
            obj.EE_pos_offset = [0;0;0.01]; % Offset of rod calculation EE and measured EE
            obj.M_base = 0.0167 + 1077*pi*obj.radius^2*0.01; % adapter(incl. screws) + Dragonskin cap
            
            % Material parameters
            E = 1.5*80000;      % E-Modul
            nu = 0.5;       % Poisson ratio
            rho_correction = ((pi*obj.radius^2)-(3*pi*0.005^2)) / (pi*obj.radius^2); % for air chambers (r_chamber = 0.005 m)
            obj.rho = 1069 * rho_correction;     % material density Ecoflex 00-50
            damp = 0.03;    % damping coeff
            
            % Kinetic
            grav = [0,0,9.81]; % gravity in m/s^2
            obj.MM = 0.8*computeMass(obj.radius,obj.rho); % mass matrix
            obj.CM = computeCompliance(E,nu,obj.radius);
            obj.CM(3,3) = obj.CM(3,3)*0.1;
            obj.DM = diag([2*damp,2*damp,damp,damp,damp,2*damp]);
            
            % combine parameters as wrappers for functions
            obj.setGravity(grav);
            updateParameters(obj);
            obj.setTipMassInKg(0.06); % 60 grams for marker system
        end
        
        function obj = setShearCompliance(obj, C)
            obj.CM(1,1) = C;
            obj.CM(2,2) = C;
            updateParameters(obj);
        end
        
        function obj = setBendingCompliance(obj, C)
            obj.CM(4,4) = C;
            obj.CM(5,5) = C;
            updateParameters(obj);
        end
        
        function obj = setDilatationCompliance(obj, C)
            obj.CM(3,3) = C;
            updateParameters(obj);
        end
        
        function obj = setReferenceLength(obj, L)
            obj.L = L;
            updateParameters(obj);
        end
        function obj = setTipMassInKg(obj,mass)
            ds = obj.L/(obj.N-1);
            obj.M_tip = mass/ds; % all masses given as mass of cross-section with 1 m height (gets multiplied by ds in gravity function)
            updateParameters(obj);
        end
        
        
        function obj = updateParameters(obj)
            % Create wrappers for numeric parameters
            % (This is often source of major bugs, if something is totally off
            % there is a good chance that the devil lies within these few
            % wonderful lines of code.)
            
            
            % all parameters
            nParameters = nargin('param');
            if nParameters == 25
                obj.param = [diag(obj.CM)',...
                    diag(obj.DM)',...
                    obj.L,...
                    diag(obj.MM)',...
                    obj.xi0(4:6),...
                    obj.xi0(1:3)
                    ];
            elseif nParameters == 26
                obj.param = [diag(obj.CM)',...
                    diag(obj.DM)',...
                    obj.L,...
                    diag(obj.MM)',...
                    obj.M_tip,...
                    obj.xi0(4:6),...
                    obj.xi0(1:3)
                    ];
            else
                error('Insufficient number of parameters. Check for undetected/undefined parameters.')
            end
            obj.param_gravity  = [obj.grav,obj.L,obj.MM(3,3),obj.M_tip];
            T_base  = [obj.R0,obj.fts_pos_offset;0 0 0 1];
            obj.xi_base  = SE3Group.logMap(T_base);
        end
        function obj = setGravity(obj,grav)
            %%
            if iscolumn(grav)
                grav = grav';
            end
            obj.grav = grav;
            updateParameters(obj);
        end
        
        function y = f(obj, t, x)
            % f - state equation function handle
            y = f_(x, obj.input(t), reshape(obj.u(t, x), [], 1), obj.u_g(t, x), obj.param);
        end
        
        function jac = j_f_x(obj, t, x)
            % j_f_x - jacobian f(x,u) of w.r.t. x
            jac = j_f_x(x, obj.input(t),[], [], obj.param);
        end
        
        function y = u_g(obj, t, x)
            % u_g - function handle for gravitational loads
            y = gravity(obj.eta(t, x), obj.param_gravity,obj.N);
        end
        
        function g_der = j_ug_eta(obj, t, x)
            % j_ug_eta - function handle for jacobian of u_g w.r.t x
            g_der = gravity_derivative(obj.eta(t, x), obj.param_gravity,obj.N);
        end
        
        function j = j_f(obj, t, x)
            % j_f - total jacobian for solver
            j = obj.j_f_x(t, x) + ...
                j_f_ug([], obj.param) * eye(size(j_f_ug([], obj.param),2),size(obj.j_ug_eta(t, x) ,1)) * obj.j_ug_eta(t, x) * obj.j_eta_x(t, x) + ...
                j_f_u([], [], [], obj.param) * obj.j_up_x(t, x);
        end
        
        function y = u_p(obj, t, x)
            % u_p - function handle for pressure actuation loads
            y = u_p(x, obj.input(t), obj.param);
        end
        
        function jac = j_up_x(obj, t, x)
            % j_up_x - function handle for jacobian of u_p w.r.t. x
            jac = j_up_x(x, obj.input(t), obj.param);
        end
        
        
        function y = u(obj, t, x)
            % u - function handle for external loads
            y = obj.u_p(t, x);
        end
        
        function y = eta(obj, t, x)
            % eta - strains (function handle)
            y = eta_pls(x, obj.input(t), obj.param);
        end
        
        function jac = j_eta_x(obj, t, x)
            % j_eta_x - jacobian of eta w.r.t. to x
            jac = j_eta_pls_x( diag(obj.CM)');
        end
        
        function y = c(obj, t, x, xt)
            % c_ - implicit state space with boundary condition
            y = c_(x, xt, obj.input(t), reshape(obj.u(t, x), [], 1), obj.u_g(t, x), obj.param);
        end
        
        function jac = j_c_x(obj, t, x, xt)
            % j_c_x - total jacobian of fully implicit model w.r.t. x
            jac = j_c_x(x,xt,obj.input(t),[],[],obj.param,[]) + ...
                    j_c_ug(x,xt,obj.input(t),[],[],obj.param,[]) *...
                    eye(size(j_c_ug(x,xt,obj.input(t),[],[],obj.param,[]),2) ,size(obj.j_ug_eta(t, x),1)) *...
                    obj.j_ug_eta(t, x) * obj.j_eta_x(t, x) + ...
                    j_c_u(x,xt,obj.input(t),[],[],obj.param,[]) * obj.j_up_x(t, x);
        
        end
        
        function x0 = x0(obj)
            % For automatic intialisation of IVP compute number of DOFs
            % simulation setup: initial conditions
            if obj.config.orientationIsState
                NDOF = (obj.N-1)*obj.K + obj.N_h*4;
                x0 = zeros(NDOF,1);
                x0(NDOF-4*obj.N_h+1:4:end) = 1; % assume rod to be straight initially
            else
                NDOF = (obj.N-1)*obj.K;
                x0 = zeros(NDOF,1);
            end
        end
        
        function obj = setModelParameters(obj,p)
            % p must contain diag(C), diag(D) diag(M) as a 18x1 vector
            % Bending, shearing compliance/damping: both dimensions need to
            % match ([p1,p2] [p4,p5], [p7,p8], [p10,p11]. Also first three entries
            % of mass matrix need to match and p16,p17)
            obj.CM = diag(p(1:6));
            obj.DM = diag(p(7:12));
            obj.MM = diag(p(13:18));
            isIsometric = (obj.CM(1,1) == obj.CM(1,1)) ...
                && (obj.CM(4,4) == obj.CM(5,5)) ...
                && (obj.MM(4,4) == obj.MM(5,5)) ...
                && (obj.DM(2,2) == obj.DM(3,3));
            assert(isIsometrix, 'Parameter mismatch: Only isometric parametrisations are valid in this current implementation.')
            assert(obj.MM(1,1) == obj.MM(2,2) && obj.MM(2,2) == obj.MM(3,3),'Mass matrix: First three entries must be equal')
        end
        
        function obj = setInputsZero(obj)
            % ambient pressure in all chambers: no actuation
            obj.setInputConstant(obj.p0*ones(1,3));
        end
        
        function obj = setInputConstant(obj,amp)
            %% setInputConstant sets ABSOLUTE (!) pressure constant
            % amp - Pressure as 1x3 in Pa (absolute!)
            if isrow(amp) 
                amp = amp';
            end
            if (any(amp<0.95e5))
                warning("Pressure trajectory drastically below environmental pressure!")
            end
            % Generate pressure input
            obj.A_p = obj.compressiongain*(5.15/1000)^2*pi;
            obj.pressure_handle = @(t) repmat(amp',length(t),1);
            obj.input = @(t)actuation_model_pressure( amp,[obj.A_p,obj.p0,obj.r_p]);
        end
        
        function obj = setInputData(obj,time,data)
            %% setInputConstant sets ABSOLUTE (!) pressure constant
            % amp - Pressure as 1x3 in Pa (absolute!)
            if (any(data<0.95e5,"all"))
                warning("Pressure trajectory drastically below environmental pressure!")
            end
            obj.A_p = obj.compressiongain*(5.15/1000)^2*pi;
            pressure =  @(t)fhandleFromData(t(:)',time(:)',data);
            obj.pressure_handle = pressure;
            obj.input = @(t)actuation_model_pressure( pressure(t)',[obj.A_p,obj.p0,obj.r_p]);
        end
        
        function endeffector_pose = EE(obj,t,x)
            % Calculate end-effector pose including offsets
            endeffector_pose = rodEndEffector(obj.eta(t,x),obj.xi_base,obj.L);

            T_rodEE_measEE = [diag([1 1 1]),obj.EE_pos_offset;0 0 0 1];
            endeffector_pose = endeffector_pose*T_rodEE_measEE;
        end
        function fts_sensor_output = FTS(obj,t,x)
            %% Output model for state estimation

            temp =  out_ft_sensor(x, obj.input(t),obj.param);
            
            fts_sensor_output = temp + [obj.grav'*obj.M_base;-cross(obj.fts_pos_offset,[eye(3),zeros(3)] * temp)];
        end

        function dx = stateEquation(obj,x,u)
            %% stateEquation stripped from any time dependency to directly
            % call it using state x and inputs u.
            %
            % All operations are already implemented in a nested manner for
            % simulation purposes. However, that becomes quite unhandy in
            % some cases.
            if isrow(u)
                u = u';
            end
            % pressure to wrench (with actuation parameters)
            wrench = obj.pressure2wrench(u);
            % gravity
            tau_g =  gravity(eta_pls(x, wrench, obj.param), obj.param_gravity,obj.N);
            % pressure
            tau_p = reshape(u_p(x, wrench, obj.param), [], 1);
            
            % state equation with all subexpressions
            dx = f_(x, wrench, tau_p, tau_g, obj.param);
        end

        function wrench = pressure2wrench(obj,u)
            %% computes local wrench from pressure input u \in R^3x1 ->
            % R^6x1
            
            if any(u<(0.95e5-1))
                warning('Pressure drastically below environmental pressure.')
            end
            A_p = obj.compressiongain*(5.15/1000)^2*pi; % as in setInputConstant and setInputData
            wrench = actuation_model_pressure( u,[A_p,obj.p0,obj.r_p]);
        end

        function out = configuration(obj,t,x)
            % Basis
            if isrow(x)
                x = x';
            end
            config = rodShapeSparse(obj.eta(t,x),obj.xi_base,obj.L);
            % measured end effector (marker frame) as last point of the shape
            T_rodEE_measEE = [diag([1 1 1]),obj.EE_pos_offset;0 0 0 1];
            config(:,:,end+1) = config(:,:,end) * T_rodEE_measEE;
            out = config;
        end
        
        function out = configurationSparse(obj,t,x)
            % T_base  = [obj.R0,obj.r0;0 0 0 1];
            % obj.xi_base  = SE3Group.logMap(T_base);
            config = rodShapeSparse(obj.eta(t,x),obj.xi_base,obj.L);
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

        function y = run(obj,t)
            % sim options
            opts = odeset('Jacobian',@obj.j_f,'NormControl','on');
            % opts = odeset('OutputFcn',@simOutFunc)
            % disp('Starting Simulation..')
            % tic
            [t,y] = ode23s(@obj.f,t,obj.x0,opts);
            % disp(['Total Computation Time',num2str(toc)])
        end
        
    end
end
