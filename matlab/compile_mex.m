%% Compile the time-critical functions of the simulation to MEX
% The Cosserat rod simulation (step 2, Exp. 2 and the plant in Exp. 3 and 4)
% evaluates the following functions in every call of the ODE function or
% its Jacobian:
% - f_, j_f_x: generated state equation and its Jacobian, long loop-free
%   code (about 55000 and 190000 scalar operations). With MATLAB R2024b on
%   an Apple M4 Pro they take about 30 and 120 ms per call as .m files
%   (0.5-1 us per scalar operation), as MEX functions about 0.02 ms.
% - gravity, gravity_derivative: distributed gravitational load and its
%   Jacobian
% - fhandleFromData: interpolation of the input
% Without the MEX files the scripts run the .m files with identical results,
% but depending on the MATLAB release and platform the simulation can be
% about ten times slower (Exp. 2, R2024b, Apple M4 Pro: 160 instead of
% 14 ms per time step; T-RO with .m files and R2020b: 35.9 ms).
%
% Requires MATLAB Coder and a supported C compiler (mex -setup). The MEX
% files are written next to the .m files and take precedence over them.
% Compiling j_f_x takes about 5 min. Run once from the repository root
% after setup; delete the MEX files to fall back to the .m files.

root = fileparts(fileparts(mfilename('fullpath')));
fun_dir = fullfile(root, 'matlab', 'functions');
cdat = model_config();
model_dir = cdat.path;
build_dir = fullfile(tempdir, 'rod_pinn_codegen');

cfg = coder.config('mex');
cfg.GenerateReport = false;

% f_(x, w, u, u_g, param), j_f_x(x, w, [], [], param): state, actuation
% wrench, distributed pressure and gravitational loads, parameters
nx = 12*(cdat.N - 1);
x = coder.typeof(0, [nx 1]);
wrench = coder.typeof(0, [6 1]);
param = coder.typeof(0, [1 26]);
args_f = {x, wrench, coder.typeof(0, [Inf 1], [1 0]), coder.typeof(0, [Inf 1], [1 0]), param};
args_j_f_x = {x, wrench, coder.typeof(0, [0 0]), coder.typeof(0, [0 0]), param};
% gravity(xi, param, N_nodes), gravity_derivative(xi, param, N_nodes):
% strains xi (6 x sub-nodes), param = [g(3), L, M(3,3), M_tip] (row)
args_gravity = {coder.typeof(0, [6 Inf], [0 1]), coder.typeof(0, [1 6]), coder.typeof(0)};
% fhandleFromData(t, time, data): interp1(time, data, t) with query and
% sample times as rows and samples (one row per sample time)
args_interp = {coder.typeof(0, [1 Inf], [0 1]), coder.typeof(0, [1 Inf], [0 1]), coder.typeof(0, [Inf Inf], [1 1])};

entry_points = {
    'f_',                 model_dir, args_f
    'j_f_x',              model_dir, args_j_f_x
    'gravity',            fun_dir,   args_gravity
    'gravity_derivative', fun_dir,   args_gravity
    'fhandleFromData',    fun_dir,   args_interp};
for i = 1:size(entry_points, 1)
    [name, out_dir, args] = entry_points{i, :};
    fprintf('Compiling %s ...\n', name)
    codegen('-config', cfg, '-d', fullfile(build_dir, name), '-o', fullfile(out_dir, name), ...
        fullfile(out_dir, [name, '.m']), '-args', args)
end
clear(entry_points{:, 1}) % use the new MEX files in this session
fprintf('MEX files written to %s and %s\n', model_dir, fun_dir)
