classdef SymFluidicActuation < ModelSetup
    %% internal acutation
    % This model is the mapping from pressure to wrench for
    % our current pressure driven actuator with 3 chambers.
    % For the current development this is rather
    % unimportant.
    % For the PINNs, this relation can be used to reduce
    % the domain for the collocation points during
    % training. wrench \in R6 --> pressure \in R3.
    
    properties
        thetha;
        r_p = sym('r_p');   % radius from rod centre to chamber centre
        p0 = sym('p0');     % env. pressure
        A_p = sym('A_p');   % chamber radius
        p_act
        tau_pressure
        act_param
    end
    
    methods
        function obj = SymFluidicActuation(n_chambers,n_actuators,config)
            %SYMFLUIDICACTUATION Construct an instance of this class
            %   A total of n_actuators with n_chambers is created. The
            %   chambers are evenly distributed on a circle around the
            %   rod's centre with radius r_p. Each chamber has the same
            %   cross-section with area A_p. The pressure is directed into
            %   the rod's plane normal e_z = [0 0 1]^T.
          
            obj@ModelSetup(config);
            thetha = linspace(-pi,pi,n_chambers+1);
            thetha = thetha(1:3);
            rpi = obj.r_p * [cos(thetha);sin(thetha);zeros(size(thetha))];
            obj.p_act = sym('p',[n_chambers,n_actuators]);
            unitz = sym([0;0;1]);
            n_p = sym(zeros(3,n_actuators));
            m_p = sym(zeros(3,n_actuators));
            for i = 1:n_actuators
                for j = 1:n_chambers
                    n_p(:,i) = n_p(:,i) + ...
                        (obj.p_act(j,i)-obj.p0)*obj.A_p*unitz;
                    m_p(:,i) = m_p(:,i) + ...
                        (cross(rpi(:,j), (obj.p_act(j,i)-obj.p0)*obj.A_p*unitz));
                end
            end
            obj.tau_pressure = [n_p;m_p];
            obj.act_param = symvar(obj.tau_pressure);
            obj.act_param = symvar(subs(obj.act_param,...
                obj.p_act, sym(zeros(size(obj.p_act)))));
        end
        
        
        function export(obj)
            %EXPORT Creates Matlab files for the actuator model
            % export pressure actuation model
            matlabFunction(obj.tau_pressure,...
                'Vars', {obj.p_act(:),obj.act_param},...
                'Output',{'w'}, ...
                'File',[obj.path,filesep,'actuation_model_pressure']);
        end
    end
end

