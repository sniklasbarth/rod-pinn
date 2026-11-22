# Rod-PINN: adaptive MPC of a soft continuum robot with a physics-informed neural network

Code and data to reproduce the simulation results of

> J. Licher\*, M. Bartholdt\*, H. Krauss, T.-L. Habich, T. Seel, M. Schappler,
> "Adaptive Model-Predictive Control of a Soft Continuum Robot Using a Physics-Informed Neural
> Network Based on Cosserat Rod Theory," *IEEE Transactions on Robotics*.
> (\*equal contribution)

The repository covers the full pipeline: generation of the Cosserat rod model, simulation of the
training trajectory, training of the domain-decoupled PINN (DD-PINN), and the closed loop of
unscented Kalman filter (UKF) and nonlinear evolutionary MPC (NEMPC) with the Cosserat rod
simulation as plant. The identified model parameters are included; the identification itself is
not part of this repository.

## AI Usage

This is a refactored and joined version of our original repositories. Claude Code (Claude Opus
5.5) has been used to join the repositories, merge branches and simplify the use for others.

## Contents

```
setup.m                         MATLAB path, Python environment, plot defaults
matlab/
  step1_generate_model.m        symbolic model -> matlab/model/TRO_N_7 (included)
  compile_mex.m                 MEX functions of the model and the simulation (optional, recommended)
  step2_simulate_training_data.m  training trajectory -> results/training_simulation.csv
  exp2_pinn_verification.m      Exp. 2: PINN and simulation vs. measurement (Table III, Figs. 6, 7)
  exp3_ukf_parameter_estimation.m  Exp. 3: UKF estimation of the bending compliance (Fig. 9)
  exp4_nempc_simulation.m       Exp. 4a/4b: NEMPC in simulation
  exp4_plot.m                   Exp. 4: errors and plot (Fig. 11)
  model_config.m                configuration of the model (N = 7, implicit midpoint, gravity)
  classes/, functions/          model generation (Sym*), simulation, PINN, UKF, MPC interface
  model/TRO_N_7/                generated model functions
  external/                     SO(3)/SE(3)/S3 group operations and plot defaults (IMES toolboxes)
python/
  PINN_training.py              DD-PINN training as used in the paper (usually faster on CPU)
  PINN_training_GPU.py          DD-PINN training with hyperparameters adjusted for GPU training
  PINN_Functions.py             training, sampling, DD-PINN
  PINN_ODE_loss_functions.py    physics of the model in PyTorch (loss): actuation, gravity
  PINN_ODE_loss_functions_N7.py   implicit model, strains, actuation loads (generated, included)
  matlab_to_torch.py            translation of the MATLAB export to PyTorch (step 1b)
  NEMPC_matlab_interface.py     NEMPC called from MATLAB (exp4)
  NonlinearEMPC_pure_torch.py   evolutionary MPC
  pinn_io.py, torch_device.py   plain-text weights, device selection (CUDA, Apple Metal, CPU)
parameters/
  rod_parameters.json           Cosserat rod parameters (identified: Table II)
  pressure_models.json          pressure controller models (identified and NEMPC prediction model)
  ukf_parameters.json           UKF noise and sigma point parameters
  nempc_parameters.json         NEMPC parameters and state weights
pinn/                           trained DD-PINN of the paper (weights.json, normalization, hyperparameters)
tests/                          unit tests (runtests('tests'))
data/
  training_simulation.csv       simulated trajectory of the paper PINN (reference, not used by default)
  exp2_measurement.csv          test bench measurement of Exp. 2
```

All parameters, the PINN and the data are stored as plain text (JSON, CSV). Results are
written to `results/` (not tracked).

## Requirements

- MATLAB (R2024b was used) with
  - Control System Toolbox (`tf`, `c2d`: Exp. 4)
  - Optimization Toolbox (`fsolve`: setpoint of Exp. 4b)
  - Signal Processing Toolbox (`butter`: input filter of Exp. 2)
  - Robotics System Toolbox (`rotm2quat`, `quat2rotm`)
  - Symbolic Math Toolbox (only for step 1)
  - MATLAB Coder and a C compiler (optional: MEX functions, see below)
  - Parallel Computing Toolbox (optional: PINN evaluation on an NVIDIA GPU)
- Python 3.9–3.12 (supported by MATLAB R2024b's `pyenv`) with the packages in
  `python/requirements.txt`. For training on an NVIDIA GPU install the CUDA build of torch
  (e.g. `torch==2.5.1+cu124` from the PyTorch index for CUDA 12.4).

Create the Python environment in the repository root:

```
python3.11 -m venv .venv
.venv/bin/pip install -r python/requirements.txt          # Windows: .venv\Scripts\pip ...
```

`setup.m` selects `.venv/bin/python` (Windows: `.venv\Scripts\python.exe`) for MATLAB. If i
does not exist, MATLAB's default Python environment is used, provided torch is available there
(adapt `python_exe` for another environment). On Apple Silicon it sets `ROD_PINN_DEVICE=cpu`, because Apple Metal lacks the
Cholesky decomposition used by the NEMPC.

## Usage

Start MATLAB in the repository root and run `setup` once per session (before any Python call).
All scripts are run from the repository root.

| Step | Command | Duration | Output |
|---|---|---|---|
| 1 | `run matlab/step1_generate_model` | 2 min | `matlab/model/TRO_N_7/` |
| 1b | `python python/matlab_to_torch.py matlab/model/TRO_N_7` | 10 s | `python/PINN_ODE_loss_functions_N7.py` |
| (optional) | `run matlab/compile_mex` | 7 min | MEX files in `matlab/model/TRO_N_7/`, `matlab/functions/` |
| 2 | `run matlab/step2_simulate_training_data` | 1 min | `results/training_simulation.csv` |
| 3 | `python python/PINN_training.py` or `python python/PINN_training_GPU.py` | 10 h (GPU) to 8 days (CPU), see below | `results/pinn_training/<run>/` |
| Exp. 2 | `run matlab/exp2_pinn_verification` | 2 min | error table, `results/exp2_*.pdf` |
| Exp. 3 | `run matlab/exp3_ukf_parameter_estimation` | 10 min | `results/exp3_*` |
| Exp. 4a | `run matlab/exp4_nempc_simulation` (`task = 'tracking'`) | 7 min | `results/exp4_tracking.mat` |
| Exp. 4b | same with `task = 'setpoint'` | 7 min | `results/exp4_setpoint.mat` |
| Exp. 4 | `run matlab/exp4_plot` | 1 min | errors, `results/exp4_*.pdf` |

The experiments use the trained PINN in `pinn/`; steps 1–3 reproduce the model and the PINN.

**Step 1** regenerates the symbolic export, which is already included. The model is a 3D Cosserat rod in the local frame
formulation with implicit midpoint collocation, N = 7 nodes and gravity on 8 sub-nodes per
element.
Simulation, PINN loss, UKF and NEMPC all use this model.

**Step 1b** translates the exports `c_` (implicit model and boundary values), `eta_pls` (strains
on the gravity sub-nodes) and `u_p` (distributed actuation loads) to batched PyTorch functions in
`python/PINN_ODE_loss_functions_N7.py`, which `python/PINN_ODE_loss_functions.py` imports for the
physics loss. The generated file is included. With `--check --matlab <matlab executable>` the
script evaluates the MATLAB exports for random inputs and compares them with the generated
functions.

**MEX functions.** `compile_mex` compiles the functions that the Cosserat rod simulation evaluates
in every call of the ODE function and its Jacobian: the generated state equation `f_` and its
Jacobian `j_f_x`, the distributed gravitational load (`gravity`, `gravity_derivative`) and the
input interpolation (`fhandleFromData`). The MEX files take precedence over the .m files and give
identical results, but accelerate the simulation. This affects step 2
and the plant simulation in Exp. 2–4, not the PINN.

**Step 2** simulates the training trajectory (three staggered tanh pulses, 5 s, 70 Hz). The
training only takes the sampling ranges of states and inputs (plus a 10 % margin) and the
reference trajectory for its plots from it.

**Step 3** trains the DD-PINN (23 ansatz functions, 281 neurons, one hidden layer, 70 Hz) on the physics loss only.
`python python/PINN_training.py` uses the training setup that has been used to train the PINN in the paper (Sec. V-B, PINN Training and Hyperparameter Optimization): 10^6 collocation points per epoch in batches of 1000 and an initial learning rate of 0.000688. At this batch size, the training is usually faster on the CPU but in general slower than the training optimized for the GPU below. As described in the paper it took about 8 days on two CPU cores.
`python python/PINN_training_GPU.py` uses the GPU training setup described in the discussion section of the paper (Sec. VII-A):
10^6 collocation points per epoch in batches of 20000, a fivefold initial learning rate
(5 · 0.000688), gradient clipping (maximum norm 1.0) and early stopping (150 epochs withou
improvement of the validation loss). The training ends at the minimum learning rate, by early
stopping or after 6000 epochs. The printed remaining time is an upper bound (up to 6000 epochs). A run on an NVIDIA H200 took 4400 epochs of 8.5 s each (10.3 h); on the CPU of an Apple M4 Pro an epoch takes 30 s.

Now the experiments can be run with the trained PINN in `pinn/` (or with a newly trained one). For details on the experiments see the paper. The scripts generate the figures and error metrics of the paper.


## Variation in results

The results of the paper can be reproduced with the included model, PINN and data. However, Exp. 3 and 4 contain random measurement noise (and Exp. 4 the stochastic NEMPC), so the numbers vary between runs.

## Tests

```
setup
results = runtests('tests')
```

The unit tests (about 1 min) compare all analytical Jacobians with numerical ones (central
differences): the exported Jacobians of the state equation and the implicit model, the Jacobian of
the gravitational load (derived with Lie group theory), the total Jacobian of the ODE used by
`ode23s`, and the Jacobians of the SO(3), SE(3) and S³ operations. They further check the
reference configuration, the consistency of the explicit and the implicit model, the MEX functions
against the .m files and the PINN against the simulation.



## License

MIT, see `LICENSE`. If you use this code, please cite the paper above.
