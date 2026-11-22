"""
DD-PINN training for the Cosserat rod model
@author: HK, JL, adapted from TLH

    python python/PINN_training_GPU.py

GPU training setup (Sec. VII-A of the paper): batches of 20000 collocation points, initial learning
rate 5*0.000688, gradient clipping and early stopping. On an NVIDIA H200 the training took 4400
epochs of 8.5 s each (10.3 h) until the minimum learning rate was reached, without NaN outputs. For
the training setup of the published PINN see PINN_training.py.

Results are written to results/pinn_training/<dd_mm_yyyy__hh_mm>/epoch_<n>/ every plot_interval
epochs, including the plain-text weights.json used by MATLAB and the NEMPC. At the end, the model
with the lowest validation loss is restored and saved to best_epoch_<n>/. Run from the repository
root.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import PINN_Functions
from torch_device import get_device

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--continue', dest='load_folder', default=None,
                    help='run folder of a previous training to continue (its last epoch is loaded)')
parser.add_argument('--lr', type=float, default=None, help='initial learning rate (default: 5*0.000688)')
parser.add_argument('--epochs', type=int, default=6000, help='number of epochs')
parser.add_argument('--sim-file', default=os.path.join('results', 'training_simulation.csv'),
                    help='simulated trajectory (sampling ranges, plot reference), written by '
                         'matlab/step2_simulate_training_data.m; data/training_simulation.csv is the '
                         'trajectory of the published PINN')
args = parser.parse_args()
if not os.path.isfile(args.sim_file):
    parser.error(f'{args.sim_file} not found, run matlab/step2_simulate_training_data.m first '
                 'or pass the trajectory of the published PINN with --sim-file data/training_simulation.csv')

local_run = False       # True: print more training statistics
save_everything = True  # True: save model, plots etc. each "plot_interval" epochs
cwd = Path(os.getcwd())

# Use a GPU (CUDA or Apple Metal) if available, otherwise the CPU (see torch_device.py)
device = get_device(use_gpu=True)

torch.set_default_dtype(torch.float32)

# Reproducibility of several runs
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters (training setup of J. Licher, see above)
HP_dict = {

    # System parameters
    "n_discr": 7,                   # Number of discrete points of the Cosserat model (12*(N-1) states)
    "model_dims": 3,                # Number of dimensions of the Cosserat model
    "n_ODE": 1,                     # Number of ODE residual losses to sum up to (relevant for loss balancing)
    "grav_flag": True,              # Flag to use gravity
    "var_bend_comp": True,          # Flag to use the bending compliance as additional PINN input (needed for the UKF)
    "var_tip_mass": False,          # Flag to use variable tip mass in the Cosserat model
    "disable_states_flag": False,   # Flag to disable Cosserat states that are close to 0 (experimental)
    "import_rod_params": True,      # Rod parameters from rod_param_file

    # Main neural network training parameters
    "n_data": 2000,                 # Number of data points, should be >= n_batches
    "resampling_period": 1000,      # Number of epochs until resampling of collocation points
    "multivar_normal_dist": True,   # If main states (xdim) shall be sampled using a multivariate normal distribution instead latin hypercube
    "multivar_dist_std": 0.5,       # Multivariate distribution: Standard deviation around 0 for the normal distribution (Warning: sampled values can exceed (-1,1))
    "constr_u_sample_flag": True,   # If 'u_degree'>0, constrains the sampling of excitation values u_b, u_c in certain range of u_a
    "constr_u_ratio": 0.0175,       # Ratio, by which u_b, u_c are constrained around u_a for normalized values
    "plot_interval": 200,           # Number of epochs until plotting and export of results
    "t_range": 5,                   # Range in s for which to run the model for - only related to the plot
    "split_size": 0.7,              # Ratio of training to validation data/ ic/ collocation points
    "num_epochs": args.epochs,      # Number of total epochs to compute
    "nn_type": "DDNN",              # Domain-decoupled PINN
    "n_sums_DDNN": 23,              # Number of sub-functions in 'g' of DD-PINN (per each output state!)
    "ansatz_type": "Single_Layer_damped", # Type of ansatz function
    "limit_deltas": False,          # Limit the delta or d coefficients in the DDNN to avoid runaway solutions
    "n_neurons": 281,               # Number of neurons in neural network hidden layers
    "n_hidden": 1,                  # Number of neural network hidden layers
    "lr_init": 0.000688*5,          # Initial ADAM learning rate (x5 for batch_size_collo=20000)
    "lr_min": 1e-5*5,               # Criteria of learning rate to stop training
    "optim_w_decay": 1e-6,          # Weight decay of ADAM optimizer (L2-Regularization)
    "LRS_patience": 75,             # Patience in number of epochs for learning rate scheduler "ReduceLROnPlateau"
    "LRS_schedule_red": 0.5,        # Reduction factor of learning rate for "ReduceLROnPlateau"
    "early_stopping_flag": True,    # Flag to stop training if the validation loss does not improve for "ES_patience" epochs
    "ES_patience": 150,             # Patience in number of epochs for early stopping

    # Main PINN/ DD-PINN parameters
    "f_sample": 70,                 # Sampling frequency of the PINN
    "u_degree": 0,                  # Piece-wise degree of excitation u; 0: const., 1: linear, 2: quadratic

    # Additional PINN/ DD-PINN training parameters
    "t_sample_extra": 1.1,          # Factor of t-sampling to overshoot
    "tct_flag": False,              # TCT: Flag to use (time-) causal training (TCT) after Sifan Wang, compatible only with n_ODE=1
    "eps_start_val": 1e-4,          # TCT: Start value of epsilon
    "eps_weight_criteria": 0.8,     # TCT epsilon annealing: If mean weights in epoch exceed this value, epsilon gets increased tenfold
    "eps_annealing_wait": 5,        # TCT epsilon annealing: Number of epochs to wait until starting epsilon annealing
    "ic_loss_flag": False,          # Flag to use initial condition loss; PINN: True, DD-PINN: False
    "data_loss_flag": False,        # Flag to use data points (not available here)
    "erase_SSM_gradients": False,   # Flag to erase physics loss gradients for Y_dach
    "clip_gradients": True,         # Flag to clip gradients during training
    "max_grad_norm": 1.0,           # Max. gradient norm for gradient clipping

    # Loss balancing parameters
    "n_epoch_redo_balance": 20,     # Naive loss balancing: Number of epochs until naive loss balancing is reapplied
    "loss_balance_flag": False,     # Flag to use additional loss balancing
    "loss_balance_type": "Naive",   # Loss balancing type to use: "LRA", "Naive"
    "LRA_max_loss": 3,              # LRA: Which loss is reference
    "LRA_n_avg": 10,                # LRA: Number of batches used per epoch for LRA
    "alpha_loss": 0.95,             # LRA: Moving average parameter

    # Batch/Epoch parameters
    "batch_size_collo": 20000,      # Number of collocation points per batch
    "batch_size_ic": 1000,          # Number of initial condition points per batch
    "n_batches": 50,                # Number of batches

    # Files (relative to the repository root)
    "sim_file": args.sim_file,
    "rod_param_file": os.path.join('parameters', 'rod_parameters.json'),
}
if args.lr is not None:
    HP_dict["lr_init"] = args.lr

pinn = PINN_Functions.pinn_cl(HP_dict, cwd, local_run, save_everything, device=device)

# Loading of simulation results to determine sampling ranges and for use as reference on plots
pinn.loadSimResults()

# Parameter initialization
pinn.initParams()

# Create folder to save results
pinn.initSaveDir()

pinn.initTraining(args.load_folder is None, args.load_folder)

pinn.train()
