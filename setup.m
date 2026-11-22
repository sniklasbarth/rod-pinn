%% Setup of the MATLAB path and the Python environment
% Run once per MATLAB session from the repository root, before any script.
% The NEMPC (Exp. 4) runs in Python: set python_exe to the Python of the
% environment created from python/requirements.txt (MATLAB R2024b supports
% Python 3.9-3.12). Python can only be selected before it is first loaded.

root = fileparts(mfilename('fullpath'));
if ispc
    python_exe = fullfile(root, '.venv', 'Scripts', 'python.exe');
else
    python_exe = fullfile(root, '.venv', 'bin', 'python');
end
addpath(fullfile(root, 'matlab'))
addpath(genpath(fullfile(root, 'matlab', 'classes')))
addpath(genpath(fullfile(root, 'matlab', 'functions')))
addpath(genpath(fullfile(root, 'matlab', 'external')))
model_config(); % puts the model export on the path
if ~endsWith(which('f_'), mexext)
    disp('The simulation runs without MEX functions (can be about ten times slower, depending on the MATLAB release). Run matlab/compile_mex once to compile them (MATLAB Coder).')
end

if ismac && ~strcmp(computer('arch'), 'maci64')
    setenv('ROD_PINN_DEVICE', 'cpu') % Apple Metal (MPS) lacks linalg_cholesky, needed by the NEMPC
end
pe = pyenv;
if isfile(python_exe)
    if pe.Status == "Loaded" && pe.Executable ~= python_exe
        warning('Python is already loaded from %s. Restart MATLAB and run setup first to use %s.', pe.Executable, python_exe)
    elseif pe.Status ~= "Loaded"
        pyenv('Version', python_exe);
    end
else
    % fall back to the Python environment MATLAB selects by default, if torch is available there
    try
        cuda_available = py.torch.cuda.is_available();
        pe = pyenv;
        fprintf('%s not found. Using the Python environment %s, torch is available there.\n', python_exe, pe.Executable)
        if cuda_available
            disp('CUDA is available there.')
        end
    catch
        warning('Python environment %s not found and torch is not available in the default Python environment; the NEMPC (Exp. 4) will not run. See README.', python_exe)
    end
end

% plot defaults
set_factory_default;
set_fontsize_default(14);
set_color_order_imes_default;
set_latex_default;
set_plot_line_width_default(1.4);
