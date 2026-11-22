function pinn = load_pinn(pinn_folder)
%LOAD_PINN Loads a trained DD-PINN from the plain-text files written by the
% training (python/PINN_training.py): weights.json, m_x.json, b_x.json
% and hyperparameters.json.
%
%   pinn = load_pinn(fullfile(root,'pinn'))
%
% pinn.weights{i}, pinn.biases{i}: layers in the order input, hidden, output
% (weights as out x in, as in torch), m_x/b_x: normalization of all
% PINN inputs (props_names: states, supplementary values, inputs, dt).

w = jsondecode(fileread(fullfile(pinn_folder,'weights.json')));
hp = jsondecode(fileread(fullfile(pinn_folder,'hyperparameters.json')));
layers = ["layer_in", "layers_middle_" + string(0:hp.n_hidden-1), "layer_out"];
for i = 1:numel(layers)
    pinn.weights{i} = w.(layers(i) + "_weight");
    pinn.biases{i} = w.(layers(i) + "_bias"); % column
end
pinn.m_x = jsondecode(fileread(fullfile(pinn_folder,'m_x.json')));
pinn.b_x = jsondecode(fileread(fullfile(pinn_folder,'b_x.json')));
pinn.props_names = fieldnames(pinn.m_x);
pinn.sdim = double(hp.var_bend_comp) + double(hp.var_tip_mass);
pinn.udim = 3*(hp.u_degree + 1);
pinn.xdim = numel(pinn.props_names) - pinn.sdim - pinn.udim - 1; % states, supplementary values, inputs, dt
pinn.t_sample = 1/hp.f_sample;
pinn.n_funs = hp.n_sums_DDNN;
end
