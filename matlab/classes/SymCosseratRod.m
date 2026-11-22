classdef SymCosseratRod <  ModelSetup
    %SYMCOSSERATROD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % :: parameters
        % ::: material
        M = diag(sym('M_',[6,1]));        %  mass-matrix in kg/m and kg*m
        tipMass = sym('M_tip');           % additional Mass parameter at tip
        C = diag(sym('C_',[6,1]));        % compliance matrix in N/m and Nm/rad
        D = diag(sym('D_',[6,1]));        % damping D_tau in s
        dynParam
        % ::: geometry
        v0 = sym('v0_',[1,3]); % reference shear/dilatiation
        u0 = sym('u0_',[1,3]); % reference curvature/twist
        xi  % this is a symbolic expression of strains (eta = xi - xi0)
        xi0 % this defines the shape of the rod in unloaded configuration (ref. config.)
        eta_pls % linear interpolation of nodes of xi in Ng steps (piecewise linear strain)
        j_eta_pls_x % jacobian of piecewise-linear strain w.r.t. strains
        eta % the strains used for spatial integration (function)
        eta_mean % mean value of strain between two nodes (for integration of orienations)
        j_eta_x % jacobian of the eta(x) w.r.t. x
        j_eta_mean_x % jacobian of the eta_mean(x) w.r.t. x
        
        deta_ds % the strain rates w.r.t. s (also used for spatial integration)
        L = sym('L') % reference length
        % materialcases
        materialcase = 'Kelvin-Voigt'; % TODO: check if this is still up
        implicitMaterial
        % ::: kinetic quantities
        gravity = sym('g',[3,1]);
        % :: variables
        s       % spatial coordiante
        s_h     % spatial coordinates of quaternion nodes
        s_pointloads    % nodes where point loads are applied
        DeltaS  % s[i+1] - s[i]
        % state space variables (_mat = matrix is used to simplfy loops on
        % the spatial nodes 1:N. x_mat(:,1) = "x(s(1))" and so on.)
        x   % all states (algebraic and differential)
        xt  % derivatives
        u   % distributed loads
        x_mat
        xt_mat
        u_mat
        h  % (4xN_h) non unit quaternion which are nodes for cubic hermite spline interpolation
        h_ % (4xN) evaluated nodes of quaternion from cubic hermite spline
        ht % (4xN_h)quaternion derivative wrt time
        hs % (4xN_h)quaternion derivative wrt space
        h_mat
        ht_mat
        hs_mat
        % Implicit
        xk % time discrete states
        uk % time discrete inputs
        param_d % implicit parameter without dynParam
        % :: BDF-alpha variable
        xh      % xt = x+xh
        xh_mat  % xh as matrix
        % acuation
        w       % actuation variable [TODO: generalize]
        u_p % distributed load fluidic actuation
        u_p_mat
        j_up_x % jacobian wrt to x
        % explicit model equations and quantities
        x_ % differential states of the ssm
        xt_ % derivatives of x_
        f_ % nonlinear state space model
        j_f_x % jacobian wrt to x
        j_f_u % jacobian wrt to u
        % gravity
        ug % distributed load gravitation
        ug_mat %
        j_f_ug % jacobian wrt to x
        % misc.
        c                   % spatially discrete model equations
        c_                  % spatially discrete model with BV substituted
        j_c_x               % jacobian of c_ w.r.t. x_ (used for solution of static model)
        j_c_u               % jacobian of c_ w.r.t. u (used for solution of static model)
        j_c_ug              % jacobian of c_ w.r.t. ug (used for solution of static model)
        c_d                 % spatially/time discrete model
        c_static            % spatially discrete static equations
        c_static_pressure   % spatially discrete static equations with pressure actuations
        bd      % boundary conditions as equations
        bd_static
        cineq = []; % inequality constraints, in case of contact [currently unused feature]
        param   % symbolic parameter in all equations c
        BV1 = [1:6];    % hardcoded boundary values
        BV2 = [7:12];   % hardcoded boundary values
        x2x_ % mapping from full vector x to state vector x_
        strain2state % function mapping the strains to states x2 (elastic contact forces/moments)
        out_ft_sensor % output equation for ft sensor
        
        
    end
    methods
        function obj = SymCosseratRod(config)
            %% Constructor and main routine
            % When creating an object of this class, the main routine is
            % automatically run.
            obj@ModelSetup(config);
            
            % Define geometric and kinetic parameters
            obj.xi0 = [obj.v0,obj.u0];
            obj.dynParam = [diag(obj.C),diag(obj.D),diag(obj.M)];
            
            % Create State Variables
            defineSymStateSpaceVariables(obj);
            defineGravitationalLoads(obj);
            defineSymOrientationStates(obj);
            
            % Continuity constraint (spatial discretization with finite diff.)
            defineContinuityConstraint(obj);
            defineStrainOverS(obj);
            defineBoundaryValues(obj);
            
            % Actuation
            defineDistributedLoadsFluidicActuation(obj);
            definePointLoadsFluidicActuation(obj);
            %             definePointLoadsTipMass(obj);
            
            % Static model validation
            % See if constant curvature holds for pneumatic actuation
            % without gravity
            deriveStaticModel(obj);
            
            % derive an state space model
            % - a few things are hardcoded here and could be more flexible
            if obj.isExplicit
                deriveExplicitStateEquation(obj);
                % All algebraic equations need to replace the respective
                % states. This has to be done for the extenal loads in a
                % seperate manner.
                insertBoundaryValues(obj);
            end
            
            % Misc. functions not required for simulation but for example
            % for identification
            deriveExplicitInverseMaterialModel(obj);
            
            % misc. functions that are used as interfaces at some point in
            % the simulation environment
            definex2x_(obj); % interface: x to x_ (with and without boundary values)
            
            generateOutputEquation(obj);
            derivePiecewiseLinearStrain(obj);
            if obj.isImplicit
                % TODO: copy implicit, refactor routine
            end
        end
        
        function obj = defineSymStateSpaceVariables(obj)
            %% The following symbolic expressions are used to build the nonlinear system
            % of equations resulting from the spatially discretized PDE system.
            % Each quantity is given as a single vector aswell as a matrix. The latter
            % contains K rows and N columns, so that each column represents the states
            % associated with one spatial node.
            K = obj.K;
            N = obj.N;
            % % This can only be used in a certain way with implicit scheme. not sure
            % if applicable to explicit schemes, so left out for now, losing some
            % accuracy regarding the actuation model.
            obj.s = linspace(0,obj.L,obj.N); % spatial nodes as symbolic vector
            obj.DeltaS = sym(zeros(1,length(obj.s)-1));
            for i = 1:N-1
                obj.DeltaS(i) = obj.s(i+1)-obj.s(i);
            end
            obj.x_mat = obj.B*sym('x',[K,N]); % discrete state matrix
            obj.x = obj.x_mat(:); % discrete state vector
            % and their derivatives w.r.t. time t
            obj.xt_mat = obj.B*sym('xt',[K,N]);
            obj.xt = obj.xt_mat(:);
            % inputs: distributed loads (actuation, gravity, etc.) in the pde
            obj.u_mat = obj.B(obj.N1:obj.M3,obj.N1:obj.M3)*sym('u',[K/2,N]);
            obj.u = obj.u_mat(:);
            % point wrenches w: used for actuation
            obj.w = obj.B(obj.N1:obj.M3,obj.N1:obj.M3)*sym('w',[6,1]); % wrench vector
            % When using the temporal discretization technique presented in (Till
            % et al., 2019) the past states are computed and stored in the
            % variables xh (h for history).
            obj.xh_mat = obj.B*sym('xh',[K,N]);
            obj.xh = obj.xh_mat(:);
        end
        
        function obj = defineSymOrientationStates(obj)
            %% defineSymOrientationStates(obj) Generates symbolic variables
            % for orientation representation by quaternions.
            %
            % It also generates the interpolation functions if N != N_h
            % The quaternion states are then used by
            N = obj.N;
            L = obj.L;
        end
        
        function obj = defineBoundaryValues(obj)
            %% defineBoundaryValues(obj) these equality constraints set the boundary values at
            % s = 0 and s = L. For now, only cantilevered beams are
            % considered: q0,w0 = 0, nL,mL=0.
            obj.bd = obj.x_mat(obj.BV1,1); % velocities s = 0 are zero
            
            
            obj.bd = [obj.bd; (obj.x_mat(obj.BV2,end) + ...
                obj.D*obj.xt_mat(obj.BV2,end));]; % forces s = L are zero
            
        end
        
        function obj = defineContinuityConstraint(obj)
            % PDE defined by the Cosserat rod model
            ode_active = @(x,u,xt) dynamic_pde(x, u, xt, ...
                obj.B(obj.Q1:obj.W3,obj.Q1:obj.W3)*obj.xi0.',...
                obj.C, obj.M, obj.D);
            % symbolic variables (states, derivatives, inputs);
            x_mat = obj.x_mat; %#ok<*PROP>
            xt_mat = obj.xt_mat;
            u_mat = obj.u_mat;
            DeltaS = obj.DeltaS;
            w = obj.w; %#ok<*NASGU>
            % configurations
            imethod = obj.imethod;
            N = obj.N;
            disp('Define (in-)equality constraints.')
            tic
            c = [];
            for i = 1:obj.N-1
                switch imethod
                    case 'implicit_midpoint'
                        % continuity constraint
                        % contribution of gravity to the gradient has to be
                        % computed using a different scheme than for other
                        % loads.
                        % In general: If the distributed load is nonlinear
                        % you cannot apply the midpoint rule like this:
                        %
                        % See Bartholdt,2024 for more information.
                        temp = obj.leftSide(x_mat(:,i+1),xt_mat(:,i:i+1),obj.D) ... % left side of Eq.(?) TODO paper ref.
                            - obj.(imethod)(DeltaS(i),x_mat(:,i:(i+1)),u_mat(:,i:(i+1)), xt_mat(:,i:(i+1)), ode_active)... % right side of Eq.(?) TODO paper ref.
                            ;                        
                        % approx. integral of gravitational load
                        d = DeltaS(i)/(obj.Ng-1);
                        for j = 1:obj.Ng-1
                            temp = temp - [sym(zeros(6,1));obj.ug_mat(:,j,i);]*d;
                        end
                        
                        if i == 1
                            %                            temp = temp + [obj.gravity * obj.M(3,3) * obj.L;zeros(3,1)];
                        end
                        
                        if i == obj.N-1
                            disp('Adding point mass to tip of system')
                            temp = subs(temp,obj.M, obj.M + obj.tipMass*diag(sym([ones(3,1);zeros(3,1)])));
                        end
                        obj.c = [obj.c; temp] ;
                    otherwise
                        error(['ERROR:', imethod,' not implemented!'])
                end
            end
        end
        
        function obj = defineGravitationalLoads(obj)
            %% This function generate distributed gravitational loads
            %
            % It will use interpolation of the orientation depending
            % on N and N_h.
            
            % compute distributed load from gravity as continues function
            % of s by using the quaternion interpolation with cubic hermite
            % splines
            N = obj.N;
            Ng = obj.Ng;
            obj.ug_mat = sym('ug',[6,obj.Ng-1,obj.N-1]);
            obj.ug = obj.ug_mat(:);
        end
        
        function obj = definePointLoadsFluidicActuation(obj)
            %% hardcoded point wrench location. exchange later on.

            %  Model version: transition condition at the point loads
            % The transition condition (compare paper) was implemented after the experiments had been carried out; it only leads to smaller numerical
            % deviations. This repository contains the version without it so that the published numbers can be reproduced.
            obj.s_pointloads = [1,obj.N];
            % replace all coordinates s- with their respective s+ coordinate
            % in every expression!
            for i = 1:size(obj.s_pointloads, 1)
                % continuity constaints
                obj.c = subs(obj.c, obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)),...
                    obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)) + obj.w(:,i)*[1 1]);
                % actutation
                obj.u_p = subs(obj.u_p, obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)),...
                    obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)) + obj.w(:,i)*[1 1]);
                % strain function
                obj.eta = subs(obj.eta, obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)),...
                    obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)) + obj.w(:,i)*[1 1]);
                obj.eta_mean = subs(obj.eta_mean, obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)),...
                    obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)) + obj.w(:,i)*[1 1]);
                % implicit material
                obj.implicitMaterial = subs(obj.implicitMaterial, obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)),...
                    obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)) + obj.w(:,i)*[1 1]);
                % derivative of strain function
                obj.deta_ds = subs(obj.deta_ds, obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)),...
                    obj.x_mat(obj.N1:obj.M3,obj.s_pointloads(i,:)) + obj.w(:,i)*[1 1]);
            end
            % also required for any new distributed load, that depends on
            % the local strains.
        end
        
        function obj = definePointLoadsTipMass(obj)
            
            % force by inertia of tip mass = M * dx1,t
            F_m = obj.tipMass * eye(3) * obj.xt_mat(obj.Q1:obj.Q3,end);
            % continuity constaints
            obj.c = subs(obj.c, obj.x_mat(obj.N1:obj.N3,end),...
                obj.x_mat(obj.N1:obj.N3,end) + F_m);
            % actutation
            obj.u_p = subs(obj.u_p, obj.x_mat(obj.N1:obj.N3,end),...
                obj.x_mat(obj.N1:obj.N3,end) + F_m);
            % strain function
            obj.eta = subs(obj.eta, obj.x_mat(obj.N1:obj.N3,end),...
                obj.x_mat(obj.N1:obj.N3,end) + F_m);
            obj.eta_mean = subs(obj.eta_mean, obj.x_mat(obj.N1:obj.N3,end),...
                obj.x_mat(obj.N1:obj.N3,end) + F_m);
            % implicit material
            obj.implicitMaterial = subs(obj.implicitMaterial, obj.x_mat(obj.N1:obj.N3,end),...
                obj.x_mat(obj.N1:obj.N3,end) + F_m);
            % derivative of strain function
            obj.deta_ds = subs(obj.deta_ds, obj.x_mat(obj.N1:obj.N3,end),...
                obj.x_mat(obj.N1:obj.N3,end) + F_m);
        end
        
        function obj = defineDistributedLoadsFluidicActuation(obj)
            %% derive discretized distributed loads by pressure actuations
            obj.u_p_mat = sym(zeros(length(distributed_actuation_pressure(obj.x_mat(:,1),obj.w,obj.C,obj.xi0)),obj.N));
            for i = 1:obj.N
                obj.u_p_mat(:,i) = distributed_actuation_pressure(obj.x_mat(:,i),obj.w,obj.C,obj.xi0);
            end
            obj.u_p = obj.u_p_mat(:); % create this actuation map to export later
        end
        
        function obj = insertBoundaryValues(obj)
            %% INSERTBOUNDARYVALUES substitutes boundary values in required functions
            %
            % If you add new functions, that are derived using states that
            % are defined by algebraic equations, e.g. boundary values at
            % the distal ends of a rod, this function has to be listed
            % herer:
            
            % distributed pressure
            subsBoundaryValues(obj,'u_p');
            % strain function
            subsBoundaryValues(obj,'eta')
            subsBoundaryValues(obj,'eta_mean')
            % implicit material model
            subsBoundaryValues(obj,'implicitMaterial')
            % We can derive the correct jacobians after substitution
            
            % jacobians for ode23s solver
            % TODO: this has to move to a different place because it is too
            % hard to find here.
            obj.j_eta_x = jacobian(obj.eta(:),obj.x_)
            obj.j_eta_mean_x = jacobian(obj.eta_mean(:),obj.x_);
            obj.j_up_x = jacobian(obj.u_p,obj.x_);
        end
        
        function obj = deriveExplicitStateEquation(obj)
            %% find vector of of differential states
            % this doesnt work if orientationIsStae
            disp('Derive explicit state space model!')
            obj.x_ = obj.x(7:end-6);    % hardcoded boundary conditions
            obj.xt_ = obj.xt(7:end-6);  % hardcoded boundary conditions
            obj.c_ = obj.c;
            
            subsBoundaryValues(obj,'c_')
            % assuming the equation c has the form
            % M_d(x)*xt + R(x) = 0
            % Boundary conditions are hard coded here:
            R_ = simplify(subs(obj.c_,obj.xt_,obj.xt_*0));
            M_ = simplify(jacobian(obj.c_,obj.xt_));
            % assertion 1: Mxt+R = c = 0
            expr1 = M_*obj.xt_ + R_;
            expr2 = obj.c_;
            assert(all(isAlways(simplify(expr1-expr2) == 0,'Unknown','false')));
            invM_ = inv(M_); % this inversion might be a bottleneck for higher order systems.
            % derive explicit ode
            obj.f_ = (invM_)*(-R_);
            % test explicit formulation
            test = simplify(subs(obj.c_,obj.xt_,obj.f_));
            assert(all(isAlways(simplify(test) == 0)),'Unknown','false');
            % jacobians
            obj.j_f_x = jacobian(obj.f_, obj.x_);
            obj.j_f_u = jacobian(obj.f_, obj.u);
            obj.j_f_ug = jacobian(obj.f_, obj.ug);
            obj.j_c_x = jacobian(obj.c_,obj.x_);
            obj.j_c_u = jacobian(obj.c_,obj.u);
            obj.j_c_ug = jacobian(obj.c_,obj.ug);
            
        end
        
        function obj = subsBoundaryValues(obj,expr)
            %% replace boundary values at s = 0 :BV1 and s = L:BV2
            % x_mat(BV1,1) = 0 ,x_mat(BV2,end) = 0;
            
            % Point Mass at Tip
            obj.(expr) = subs(obj.(expr), obj.x_mat(obj.BV2,end), obj.x_mat(obj.BV2,end)*0);
            obj.(expr) = subs(obj.(expr), obj.x_mat(obj.BV1,1), obj.x_mat(obj.BV1,1)*0);
            obj.(expr) = subs(obj.(expr), obj.x_mat(obj.BV2,end), obj.x_mat(obj.BV2,end)*0);
            obj.(expr) = subs(obj.(expr), obj.xt_mat(obj.BV1,1), obj.x_mat(obj.BV1,1)*0);
            obj.(expr) = subs(obj.(expr), obj.xt_mat(obj.BV2,end), obj.xt_mat(obj.BV2,end)*0);
            
        end
        
        function deriveStaticModel(obj)
            %% This currently only validates the actuation by pressure in a
            % symbolic manner.
            % Change in Curvature with constant pressure actuation is
            % always zero in statics.
            
            % replace derivatives and velocities
            obj.c_static = subs(obj.c, obj.xt,sym(zeros(size(obj.xt))));
            obj.c_static = subs(obj.c_static, obj.x_mat(obj.Q1:obj.W3,:),sym(zeros(size(obj.x_mat(obj.Q1:obj.W3,:)))));
            obj.bd_static = subs(obj.bd, obj.xt,sym(zeros(size(obj.xt))));
            % remove all 0 = 0 equations
            obj.c_static(isAlways(obj.c_static == 0,'Unknown','false')) = [];
            obj.bd_static(isAlways(obj.bd_static == 0,'Unknown','false')) = [];
            
            % replace pressure actuation
            if ~isempty(obj.u_p)
                obj.c_static_pressure = subs(obj.c_static, obj.u, obj.u_p);
            end
            
            % test solution of static_pressure model ():
            test = subs(obj.c_static_pressure,obj.x_mat(7:end,2:obj.N-1),obj.w*ones(1,obj.N-2));
            test = subs(test,obj.ug,obj.ug*0);
            test = simplify(test);
            test = subs(test,obj.x_mat(7:end,[1,obj.N]),0*(obj.x_mat(7:end,[1,obj.N])));
            test = subs(test,obj.tipMass,0)
            assert(all(isAlways(test == 0,'Unknown','false')), 'Static Pressure model does not add up!')
            
            % test static model without pressure and with gravity
            c_static_gravity = subs(obj.c_static,obj.u,0*obj.u);
            % test static model without pressure and with gravity
            c_static_gravity = subs(c_static_gravity,obj.w,0*obj.w);
            % initially straight rod
            c_static_gravity = subs(c_static_gravity,obj.xi0,sym([0,0,1,0,0,0]));
            %
            %             ug_expr = gravity(obj, [obj.gravity.',obj.L,obj.M(3,3)]);
            % fill in boundary condition
            obj.c_static = subs(obj.c_static,obj.bd_static,sym(zeros(size(obj.bd_static))));
        end
     
        function obj = defineStrainOverS(obj)
            %% defineStrainOverS
            % This function generates an expression used to get the actual
            % strain at an arbitrary position s. This considers all
            % pointloads and the piecewise linear assumption, which is
            % implicitly introduced by discretization with the implicit-midpoint
            % method.
            %            assert(isempty(obj.s_pointloads),'Pointloads already substituted. Change order in main routine.');
            % get all relevant states
            x2_mat = obj.x_mat(obj.N1:obj.M3,:);
            % Later on: add all point loads/ replace s- cooridnates with s+ coodinates
            % -point loads need to generalized in the framework and stored
            % in a variable. This happens in a different routine
            
            % Use compliance to compute xi
            xi_mat = obj.C*x2_mat;
            % add xi0
            xi_mat = xi_mat + obj.xi0.';
            obj.eta = [xi_mat];
            obj.eta_mean = (obj.eta(:,1:obj.N-1) + obj.eta(:,2:obj.N))/2;
            
            % The mapping from strains to elastic forces for identification
            % is a bit more tricky, but also not awefully complicated
            % and derived here in an isolated manner.
            obj.xi = sym('xi',[6, obj.N]);
            % implicit material equation
            obj.implicitMaterial =  obj.C\(obj.xi - obj.xi0.') - x2_mat;
            % subs boundary values
            
            %
            
            for i = 1:obj.N-1
                dxi_mat(:,i) = (xi_mat(:,i+1) - xi_mat(:,i))./obj.DeltaS(i);
            end
            obj.deta_ds = dxi_mat;
            % jacobian w.r.t. to x
            % ** currently located at insertBoundaryCondition
        end
        
        function obj = generateOutputEquation(obj)
            % output equation for FT-sensor (special case)
            y_out = obj.x_mat(obj.N1:obj.M3, 1) + obj.D*obj.xt_mat(obj.N1:obj.M3, 1);
            y_out = subs(y_out, obj.xt_, obj.f_);
            obj.out_ft_sensor = y_out;
        end
        
        function obj = deriveExplicitInverseMaterialModel(obj)
            %% This expression computs x_2 = [n, m] from xi
            % with x_2 = (xi)
            %
            x2_mat = obj.x_mat(obj.N1:obj.M3,:);
            obj.strain2state = -jacobian(obj.implicitMaterial(:),  x2_mat(:)) * subs(obj.implicitMaterial(:), x2_mat(:), x2_mat(:)*0);
        end
        
        function obj = definex2x_(obj)
            %% x2x_ is a helper function, mapping the full states vector to
            % to the states vector with inserted boundary values
            obj.x2x_ = jacobian(obj.x_,obj.x)*obj.x;
        end
        function obj = derivePiecewiseLinearStrain(obj)
            obj.eta_pls = pls(obj.eta, obj.L, obj.Ng);
            obj.j_eta_pls_x = jacobian(obj.eta_pls(:),obj.x_(:));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% EXPORT FUNCTIONS %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%
        function export_param = export(obj,expr, vars)
            %%EXPORT This function exports symbolic expr with vars as input
            % to the path defined in the configuration.
            if ~isempty(obj.(expr))
                tic
                if ~isempty(vars)
                    % concatenate vars
                    var_ = [];
                    for i = 1:numel(vars)
                        var_ = [var_;obj.(vars{i})(:)];
                        inputs{i} = obj.(vars{i});
                    end
                    % determine parameter vector (all quantities not defined by vars)
                    export_param = symvar(subs(obj.(expr), var_, sym([1:numel(var_)].')));
                    inputs{end+1} = export_param;
                    
                    disp(['Exporting: ', expr])
                    disp(['... to',obj.path,filesep,expr])
                    % matlab function export
                    matlabFunction(obj.(expr),...
                        'Vars',inputs,...
                        'Output',{expr}, ...
                        'File',[obj.path,filesep,expr],...
                        'Sparse',obj.export_sparse,...
                        'Optimize',obj.export_optimization);
                    disp(['::Done after ', num2str(toc),' seconds.'])
                else
                    disp(['Exporting: ', expr])
                    disp(['... to',obj.path,filesep,expr])
                    % matlab function export
                    matlabFunction(obj.(expr),...
                        'Output',{expr}, ...
                        'File',[obj.path,filesep,expr],...
                        'Sparse',obj.export_sparse,...
                        'Optimize',obj.export_optimization);
                    disp(['::Done after ', num2str(toc),' seconds.'])
                end
            end
        end
        function exportall(obj)
            %% export all symbolic quantities required for simulation
            obj.param = export(obj, 'c', {'x','xt','w','u','ug'});
            export(obj,'param',{}); % exports parameters as a function for interfacing
            export(obj, 'c_',       {'x_','xt_','w','u','ug','param'});
            export(obj, 'j_c_x',       {'x_','xt_','w','u','ug','param'});
            export(obj, 'j_c_u',       {'x_','xt_','w','u','ug','param'});
            export(obj, 'j_c_ug',       {'x_','xt_','w','u','ug','param'});
            
            export(obj, 'bd',       {'x','xt','w','u','param'});
            export(obj, 'f_',       {'x_','w','u','ug','param'});
            export(obj, 'j_f_x',    {'x_','w','u','ug','param'});
            export(obj, 'j_f_u',    {'x_','w','u','param'});
            export(obj, 'j_f_ug',   {'x_','param'});
            export(obj, 'u_p',      {'x_','w','param'});
            export(obj, 'j_up_x',   {'x_','w','param'});
            export(obj, 'eta',      {'x_','w','param'});
            export(obj, 'eta_mean', {'x_','w','param'});
            export(obj, 'j_eta_x',  {'x_','w','param'});
            export(obj, 'j_eta_mean_x',  {'x_','w','param'});
            export(obj, 'deta_ds',  {'x','w','param'});
            export(obj, 'x2x_', {'x','param'});
            export(obj, 'strain2state', {'xi','w','param'});
            export(obj, 'out_ft_sensor', {'x_','w','param'});
            export(obj, 'eta_pls',{'x_','w','param'});
            export(obj, 'j_eta_pls_x',{'param'});
            
        end
        
    end
    methods(Static)
        
        function out = leftSide(x, xt, D)
            out = x + [zeros(6), zeros(6);zeros(6),D]*(xt(:,2)-xt(:,1));
        end
        
        function out = implicit_midpoint(ds, x, u, xt, dx)
            %% This function implements the implicit midpoint method
            % for solving a differential equation.
            %
            % Input arguments:
            %   - dt: step size.
            %   - x: Matrix representing the current and next state
            % of the system.
            %   - u: Matrix representing the control input.
            %   - xt: Matrix representing the target state.
            %   - dx: User-defined function that calculates a
            % change in the state based on the current state (x),
            % control input (u), and target state (xt).
            %
            % Output:
            %   - out: The updated state of the system after one
            % time step.
            assert(size(x,2)==2, 'Input for implicit scheme incorrect.')
            % Extract the first column of the matrix x
            x0 = x(:, 1);
            
            % Calculate the sum of rows of x, u, and xt,
            % and divide by 2 to obtain midpoint values.
            x_sum = sum(x, 2) / 2;
            u_sum = sum(u, 2) / 2;
            xt_sum = sum(xt, 2) / 2;
            
            % Calculate a change in the state using the
            % user-defined function (PDE) dx
            delta_x = dx(x_sum, u_sum, xt_sum);
            
            % Update the state by adding the change multiplied by
            % the step
            out = x0 + delta_x * ds;
        end
        
        function out = implicit_euler(ds, x, u, xt, dx)
            x0 = x(:,1);
            x1 = x(:,2);
            u0 = u(:,1);
            u1 = u(:,2);
            xt0 = xt(:,1);
            xt1 = xt(:,2);
            out = x0 + dx(x1,u1,xt1)*ds;
        end
        
        function out = explicit_euler(ds, x, u, xt, dx)
            out = x + dx(x,u,xt)*ds;
        end
    end
end


