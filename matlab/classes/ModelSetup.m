classdef ModelSetup < matlab.mixin.Copyable
    %MODELSETUP This class stores some basic configurations of the model
    % such as spatial resolution, quadratur formula, etc.
    % These informations are critical for symbolic derivations as well as
    % numeric computations and evaluation of results (plots etc.)
    
    properties
        % numerics properties of the model
        K   % DOF per node
        N   % number of spatial nodes      
        Ng = 8 % number of graviational increments per node -> should always match N_segments, maybe define from config.m
        % model specifications
        imethod % spatial integration method (quadrature formula) used for discretisation
        modelcase % different spatial cases (1D,2D,3D)
        B % selection matrix of the DoFs
        orientationIsState %  boolean: consider orientation of crosssections as state (non-unit quaternion)
        gravityActive % boolean: consider gravitational load 
        isExplicit % boolean: derive/export explicit model equations
        isImplicit % boolean: derive/export explicit model equations
        
        % export settings
        export_sparse %bool: export large matrices as Matlab's sparse type
        export_optimization %bool: optimize expression during export of symbolic functions
        path
        % constants, describing the index of the physical quantities in full
        % state space vector.
        Q1 = 1; Q2 = 2; Q3 = 3;
        W1 = 4; W2 = 5; W3 = 6;
        N1 = 7; N2 = 8; N3 = 9;
        M1 = 10; M2 = 11; M3 = 12;
        
        % misc.
        config % config struct for pipeline
    end
    
    methods
        function obj = ModelSetup(config)
            %MODELSETUP This is used by subclasses:
            % - SymCosseratRod
            % - SimulationSetup
            obj.config = config;    % global config parameters of system
            obj.N = config.N;
            obj.config = config;
            
            % model specifics
            obj.modelcase = config.modelcase;
            obj.orientationIsState = config.orientationIsState;
            obj.gravityActive = config.gravityActive;
            obj.imethod = config.imethod;
            obj.isExplicit = config.isExplicit;
            obj.isImplicit = config.isImplicit;
            % export options
            obj.export_sparse = config.export_sparse;
            if  obj.export_sparse
                warning('Sparse export active. Compilation not possible.')
            end
            obj.export_optimization = config.export_optimization;
            if obj.export_optimization
                warning('Export might take a long time due to optimization. Consider deactivating export_optimization')
            end
            obj.path = config.path;
            % active degrees of freedom
            obj.B = zeros(12); % DoF selection
            switch config.modelcase
                case '1D'
                    obj.B(Q3,Q3) = 1;
                    obj.B(N3,N3) = 1;
                    obj.K = 2; % 1D -> 2 % TODO: check if this is still working
                case '2D'
                    obj.K = 6; % 2D -> 6 % TODO: check if this is still working
                    obj.B(W1,W1) = 1; % bending
                    obj.B(Q2,Q2) = 1; % shear
                    obj.B(Q3,Q3) = 1; % extension
                    obj.B(N2,N2) = 1; % shear force
                    obj.B(N3,N3) = 1; % normal force
                    obj.B(M1,M1) = 1; % bending moment
                case '3D'
                    obj.K = 12; % 3D -> 12
                    obj.B = eye(12); % full Cosserat rod dynamics
            end
            
        end
    end
end


