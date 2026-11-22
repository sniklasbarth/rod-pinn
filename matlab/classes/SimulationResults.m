classdef SimulationResults < matlab.mixin.Copyable
    % Results of SimulationSetup (Cosserat rod model, 12*(N-1) states)
    properties
        time        % Time vector     (N_t x 1)
        states      % State variables (N_t x N)
        strains     % local strains (N x 6 x N_t)
        outputs     % Output variables (dimension depends on output)
        config      % general configuration of project (struct)
        setup       % object of SimulationSetup class
        rEE         % End-effector positions
        FTS         % Force-Torque Sensor Readings
        % physical quantities
        orientations  % nodal quaternions representing cross-sections
        physicQuant   % struct containing all physical quantities
        states_and_BV % states including boundary values
        timeXstateXnode % Tensor (N_t x K x N) containing states
        weight
        weightTipMass
        
        % Input
        pressure % time series of pressure input
        
        % numeric quantities
        K   % number of DoFs per node
        N   % number of nodes
        N_h % number of quaternion nodes
        % matrices / system parameters
        C   % compliance matrix 6x6
        Cse % compliance matrix 3x3 for shear and extension
        Cbt % compliance matrix 3x3 for bending an torsion
        D   % damping time constants matrix 6x6
        Dse % damping time constants 3x3 in s for shear and extension
        Dbt % damping time constants 3x3 in s for bending and torsion
        % help variables/plots
        stateOrder = {'velocities','angular_velocities',...
            'elastic_forces','elastic_moments'};
        stateLabels = {'Velocities $q$ in m/s','Ang. Velocities $\omega$ in rad./s',...
            'Elastic Forces in N','Elastic Moments in Nm'}
        % misc.
        ax_states
        ax_actuation
        ax_gravity
        ax_orientation
        ax_endeffector
    end
    
    methods
        function obj = SimulationResults(time, states, outputs, config, setup)
            % Constructor TODO: Tidy this up
            
            % set color order from green to blue (base to tip)
            try
                
            catch e
                warning('No instance of set-plot-defaults found.')
            end
            
            disp('::: Prepare results for evaluations.')
            obj.time = time;
            obj.states = states;
            obj.outputs = outputs;
            obj.config = config;
            obj.setup = setup;
            obj.K = 12;
            
            %%
            obj.N = obj.config.N;
            %             obj.N_h = obj.config.N_h;
            
            % check which dimension is the time domain
            N_t = length(obj.time);
            dim_states = find(N_t == size(obj.states));
            % assert that each column is a respective state
            if dim_states == 2
                obj.states = obj.states';
            end
            disp(['::: Stored time series. Values: ', num2str(N_t)])
            
            % separate orientations
            obj.separateStateOrientations;
            %
            obj.appendBoundaryValues;
            %
            obj.separatePhysicalQuantities;
            obj.createStrainTensor;
            % get compliance matrices from setup
            obj.C = setup.CM;
            obj.D = setup.DM;
            obj.Cse = setup.CM(1:3,1:3);
            obj.Cbt = setup.CM(4:6,4:6);
            %
            obj.createStateTensor;
            obj.evaluateInputFunctions;
            obj.weight= obj.setup.MM(3,3)*obj.setup.L;
            obj.weightTipMass = obj.setup.M_tip*(obj.setup.L/(obj.N-1));
            obj.computeEE;
            obj.computeFTS;
            disp('::: Simulation results processed and stored.')
        end
        
        function obj = separateStateOrientations(obj)
            %% this will no longer be required but is kept "in case".
            if obj.config.orientationIsState
                N_states = (obj.N-1) * obj.K;
                obj.orientations = obj.states;
                obj.orientations(:,1:N_states) = [];
                obj.states = obj.states(:,1:N_states);
                disp('::: Orientations and other states separated.')
            else
                obj.orientations = [];
            end
        end
        
        function obj = plotStates(obj)
            % Plot state variables over time
            % There are K states on N nodes (Total of 12 plots)
            % They are in groups of 4 simliar quantities.
            custom_corder =  [zeros(obj.config.N,1),linspace(1,0,obj.config.N)',linspace(0,1,obj.config.N)'];
            set_color_order_default(custom_corder)
            
            fig = figure;
            N_groups = 4;
            obj.ax_states = multi_col_fig(fig,N_groups,obj.K/N_groups,ones(obj.K,2));
            iAx = 0;
            fnames = fieldnames(obj.physicQuant);
            for iState = 1:N_groups
                for i = 1:obj.K/N_groups
                    disp(['Plotting ',num2str(iAx)])
                    iAx = iAx + 1;
                    plot(obj.ax_states(iAx), obj.time,obj.physicQuant.(fnames{iAx}));
                    box(obj.ax_states(iAx),'on')
                    grid(obj.ax_states(iAx),'on')
                    xlabel(obj.ax_states(iAx),'t in s');
                    ylabel(obj.ax_states(iAx),[obj.stateLabels{iState},' ',num2str(i)]);
                    
                end
            end
            colormap(custom_corder)
            colorbar('Ticks',[0,1],'TickLabels',{'Base','Tip'})
            set_color_order_imes_default;
        end
        
        function ax = plotActuation(obj)
            % Generate plots of point and distributed loads
            
            w = zeros(length(obj.setup.input(obj.time(1))),length(obj.time));
            fig = figure;
            fig.Position(1:2)=[0,0];
            ax = multi_col_fig(fig,3 ,1, ones(3,2))
            for i = 1:length(obj.time)
                w(:,i) = obj.setup.input(obj.time(i));
            end
            
            plot(ax(1),obj.time,w(1:3,:));
            ylabel(ax(1),'Force in N')
            l = legend(ax(1),{'x','y','z'})
            l.Location = 'eastoutside';
            
            plot(ax(2),obj.time,w(4:6,:));
            ylabel(ax(2),'Moment in Nm')
            l = legend(ax(2),{'x','y','z'})
            l.Location = 'eastoutside';
            
            plot(ax(3),obj.time,obj.pressure);
            ylabel(ax(3),'Pressure in Bar')
            l = legend(ax(3),{'Chamber 1','Chamber 2','Chamber 3'})
            l.Location = 'eastoutside';
            
            for i = 1:length(ax)
                hold(ax(i),'on');
                grid(ax(i),'on');
                if i<length(ax)
                    ax(i).XTickLabel = [];
                end
            end
            
        end
        
        function plotGravitationalLoad(obj)
            
            try
                temp = [obj.states, obj.orientations];
                for i = 1:length(obj.time)
                    g(:,i) = obj.setup.u_g(obj.time(i),temp(i,:)');
                end
                s_ = linspace(0, obj.setup.L, size(g,1));
                
                fig = figure;
                fig.Position(1:2)=[0,0];
                obj.ax_gravity = multi_col_fig(fig,1,3,ones(3,2));
                for i = 1:length(obj.ax_gravity)
                    gtemp  = g(i:6:end,:);
                    s_ = linspace(0, obj.setup.L, size(gtemp,1));
                    waveplot(obj.ax_gravity(i),s_,obj.time, gtemp);
                    view(obj.ax_gravity(i),30,30)
                    obj.ax_gravity(i).LineWidth = 1;
                end
                for j = 1:length(obj.time)
                    ugsum(:,j) = sqrt((g(1:6:end,j).^2 +  g(2:6:end,j).^2+ g(3:6:end,j).^2));
%                     disp(["Frame indifferent change in gravload",num2str(mean(diff))])
                end
                title(obj.ax_gravity(1), '$\mu g_x$ in N/m')
                title(obj.ax_gravity(2), '$\mu g_y$ in N/m')
                title(obj.ax_gravity(3), '$\mu g_z$ in N/m')
%                 sum(gdir)
            catch e
                warning(e.message)
            end
            
        end
        
        function plotStrains(obj)
            %% This function is a comparison of orientations.
            % Since orientations can be obtained by integration of the time
            % or spatial ODEs, the error introduced by just using time
            % integration is investigated.
            % First the SE(3) elements representing the robot's
            % configuration in space are computed (and stored). Afterwards
            % the orientations from the quaternion states are plotted and
            % an error (in axis-angle) is given.
            % waveplots
            fig = figure;
            for i = 1:6
                % Add strain to physical quantities
                obj.physicQuant.(['eta',num2str(i)])= squeeze(obj.strains(i,:,:));
                sz = size(obj.physicQuant.(['eta',num2str(i)]));
                ax = subplot(2,3,i);
                %generate waveplots for strains
                s_ = linspace(0, obj.setup.L, sz(1));
                
                waveplot(ax,s_,obj.time,obj.physicQuant.(['eta',num2str(i)]))
                view(ax,[30,30])
                title(ax,['$\epsilon_',num2str(i),'(s,t)$'])
            end
        end
        
        function obj = computeEE(obj)
            %% computes trajectory of robot's End Effector
            for i = 1:length(obj.time)
                xEE(:,:,i) = obj.setup.EE(obj.time(i),obj.states(i,:)');
            end
            obj.rEE = squeeze(xEE(1:3,4,:))';
            obj.orientations = xEE(1:3,1:3,:);
        end
        
        function [ax ,ax2] = plotEE(obj,ax)
            if isempty(ax)
                figure;
                ax  = axes;
            end
            
            % time vs ee position
            figure
            ax2 = axes
            set_color_order_rgb_default
            plot(obj.time, obj.rEE)
            ylabel(ax2,'$x_\mathrm{EE,t}$ in m')
            xlabel(ax2,'$t$ in s')
            grid(ax2,'on');
            box(ax2,'on')
            xlim(ax2,[0,obj.time(end)])
            set_color_order_imes_default
            
            plot3(ax,obj.rEE(:,1),obj.rEE(:,2),obj.rEE(:,3),'-')
            xline(ax,0,'--')
            yline(ax,0,'--')
            ax.ZDir = 'reverse';
            grid(ax,'on')
            box(ax,'on')
            daspect(ax,[1 1 1])
            xlabel(ax,'x in m')
            ylabel(ax,'y in m')
            zlabel(ax,'z in m')
            
        end
        
        function obj = computeFTS(obj)
            for i = 1:length(obj.time)
                obj.FTS(i,:) = obj.setup.FTS(obj.time(i),obj.states(i,:)');
            end
        end
        function ax = plotForceTorque(obj)
            set_color_order_rgb_default
            figure;
            ax(1) = subplot(2,1,1);
            plot(ax(1),obj.time,obj.FTS(:,1:3));
            ylabel(ax(1),'Force in N')
            ax(2) = subplot(2,1,2);
            plot(ax(2),obj.time,obj.FTS(:,4:6));
            ylabel(ax(2),'Torque in Nm')
            xlabel(ax(2),'Time in s')
            set_color_order_imes_default
            for i = 1:length(ax)
                box(ax(i),'on')
                grid(ax(i),'on')
            end
        end
        
        function obj = separatePhysicalQuantities(obj)
            disp('::: Generating struct with physical quantities.')
            obj.physicQuant = sort_solution(obj.states_and_BV,...
                obj.config.modelcase);
        end
        %% compute inputs
        function obj = evaluateInputFunctions(obj)
            % evaluates input function and stores numeric values in
            % properties
            for i = 1:length(obj.time)
                obj.pressure(i,:) = obj.setup.pressure_handle(obj.time(i));
            end
        end
        
        
        function obj = appendBoundaryValues(obj)
            % Adds the boundary values to the state matrix.
            % HARDCODED
            % This only works for cantilevered beam:
            % (q,w = 0 @ s = 0  and n,m = 0 @ s = L)
            x = obj.states;
            szy = size(x);
            obj.states_and_BV = [zeros(szy(1),obj.K/2),x,zeros(szy(1),obj.K/2)];
        end
        
        function obj = createStateTensor(obj)
            obj.timeXstateXnode = reshape(obj.states_and_BV,[length(obj.time),  obj.K, obj.N]);
        end
        
        
        function obj = createStrainTensor(obj)
            for ti = 1:length(obj.time)
                % get respective strains
                temp(:,:,ti) = obj.setup.eta(obj.time(ti),obj.states(ti,:)');
            end
            obj.strains = temp;
        end

         function obj            = plotConfiguration(obj)

            figure
            %% computes trajectory of robot's End Effector
            for i = 1:1:length(obj.time)
                rodConfig(:,:,:,i) = obj.setup.configuration(obj.time(i),obj.states(i,:)');
                rodConfigSparse(:,:,:,i) = obj.setup.configurationSparse(obj.time(i),obj.states(i,:)');
                
                % position orientation
                r = squeeze(rodConfigSparse(1:3,4,:,i));
                rFull = squeeze(rodConfig(1:3,4,:,i));
                R = squeeze(rodConfigSparse(1:3,1:3,:,i));
                
                
                ax(1) = subplot(2,1,1);
                
                p = plot3(ax(1),rFull(1,:),rFull(2,:),rFull(3,:));
                hold(ax(1),'on');
                plot3(ax(1),r(1,:),r(2,:),r(3,:),'b+');
                p.Color = ([0.1 0.7 0.2] - [0.2 0.3 0.2]) * (1-i/length(obj.time)) + [0.2 0.3 0.2];
                
                ax(1).XLim = [-0.2,0.2];
                ax(1).YLim = [-0.2,0.2];
                ax(1).ZLim = [-0.0,0.2];
                daspect(ax(1),[1,1,1])
                view(ax(1),[0 0])
                grid(ax(1),'on')
                hold(ax(1),'off');
                
                ax(2) = subplot(2,1,2);
                bar(ax(2), sqrt(sum(extForce.^2,1)))
                ylabel(ax(2), 'Contact Force in N')
                xlabel(ax(2), 'Node $i$')
                ylim(ax(2),[0,20])
                grid(ax(2),'on')
                drawnow
                pause(0.01);
            end
         end

         function p = plotCurrentConfiguration(obj,ax,iT,lineStyle)
                
                rodConfig = obj.setup.configuration(obj.time(iT),obj.states(iT,:)');       
                % position orientation
                r = squeeze(rodConfig(1:3,4,:,iT));
                hold(ax,'on')
                if nargin>3
                    p = plot3(ax,r(1,:),r(2,:),r(3,:),lineStyle);
                else
                    p = plot3(ax,r(1,:),r(2,:),r(3,:));
                end
         end
    end
end
