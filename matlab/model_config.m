function cdat = model_config()
%MODEL_CONFIG Configuration of the Cosserat rod model (N = 7 nodes, 72 states,
% without the transition condition at the point loads, see README) and puts
% its symbolic export model/TRO_N_7 on the path.
%
%   cdat = model_config();
%   sim = SimulationSetup(cdat, T);

root = fileparts(mfilename('fullpath'));
cdat = struct('modelcase','3D','isExplicit',1,'isImplicit',0,'saveSymbolicExpressions',1, ...
    'export_optimization',false,'export_sparse',false,'orientationIsState',false, ...
    'gravityActive',true,'N',7,'imethod','implicit_midpoint', ...
    'timeDomainMethod','explicit-euler', ...
    'path',fullfile(root,'model','TRO_N_7'));
addpath(cdat.path);
end
