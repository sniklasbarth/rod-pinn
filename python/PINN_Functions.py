# -*- coding: utf-8 -*-
"""
PINN and DD-PINN training for the Cosserat rod model
@author: HK, TLH, JL
"""

#### Libraries ####
import torch
from torch_device import synchronize, to_device
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import os
from pathlib import Path
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
import math
from scipy.stats import qmc
# from matplotlib.path import Path as matPath
# import matplotlib.patches as patches
import time 
import json
import PINN_ODE_loss_functions as pinn_loss
import pinn_io
class pinn_cl():
    def __init__(self,HP_dict,cwd,local_run,save_everything,device):
        # Calculation of dependent HP parameters
        HP_dict["t_sample"]=1/HP_dict["f_sample"]
        HP_dict["t_sample_max"]=HP_dict["t_sample"]*HP_dict["t_sample_extra"]
        HP_dict["tmax"]=HP_dict["t_sample_max"]
        HP_dict["batch_size_data"]= int(np.floor(HP_dict['n_data']/HP_dict['n_batches']))
        self.HP_dict=HP_dict
        self.device=device
        self.cwd=cwd
        self.local_run=local_run
        self.save_everything=save_everything

    def loadSimResults(self):
        # Simulated trajectory (results/training_simulation.csv): t, absolute pressures u1-u3, 72 states.
        # It sets the sampling ranges and serves as reference in the plots.
        sim_result = np.loadtxt(os.path.join(self.cwd, self.HP_dict["sim_file"]), delimiter=',', skiprows=1, ndmin=2)
        self.t_sim = np.clip(sim_result[:, :1], a_min=None, a_max=self.HP_dict['t_range'])
        self.u_sim = sim_result[:, 1:4]
        self.y_sim = sim_result[:, 4:]
        pinn_loss.check_sim_file(self.y_sim, self.HP_dict['n_discr'])

        # Rod parameters (parameters/rod_parameters.json)
        with open(os.path.join(self.cwd, self.HP_dict["rod_param_file"])) as f:
            rod = json.load(f)
        self.rod_dict = {"p0": rod["p0"], "compression_gain": rod["compression_gain"], "r_p": rod["chamber_radius"],
                         "ref_length": rod["ref_length"], "radius": rod["radius"]}
        for name in ["C", "D", "M"]:
            for i in range(6):
                self.rod_dict[f"{name}{i+1}"] = rod[name][i]
        self.rod_dict["M_tip"] = rod["M_tip"]

        # Calculation of dependent rod parameters
        self.rod_dict["A_p"] = self.rod_dict["compression_gain"]*(5.15/1000)**2*math.pi

    def initParams(self):
        # Cosserat rod model with 12*(N-1) states, absolute pressure input (see PINN_ODE_loss_functions)
        self.physics = pinn_loss.physics_model(self.HP_dict['n_discr'])
        self.n_state_nodes = self.HP_dict['n_discr'] - 1
        self.p_ref = self.rod_dict["p0"]

        # Creat properties and state header
        self.add_props=[]
        self.supp_props=[]
        if self.HP_dict["model_dims"] == 1:
            self.props = ['nz', 'qz'] # Properties/ states per discrete point
            self.u_props = ['unz'] # Excitations

        elif self.HP_dict["model_dims"] == 2:
            self.props = [ 'ny', 'nz', 'mx', 'qy', 'qz', 'wx']
            self.u_props = ['uny', 'unz', 'umx']

        elif self.HP_dict["model_dims"] == 3:
            self.props = ['nx', 'ny', 'nz', 'mx', 'my', 'mz','qx', 'qy', 'qz', 'wx', 'wy', 'wz']
            self.u_props = ['up1', 'up2', 'up3']
            
            if self.HP_dict["var_bend_comp"]:
                self.supp_props = self.supp_props + ['C45'] # Supplemental unsupervised properties: Variable bending compliance
            if self.HP_dict["var_tip_mass"]:
                self.supp_props = self.supp_props + ['M_tip'] # Supplemental unsupervised properties: Variable tip mass
                    
        # Define sampling ranges for properties
        self.props_max=np.ones(len(self.props)*self.n_state_nodes+len(self.add_props)+len(self.u_props)*(self.HP_dict['u_degree']+1)+1)

        self.epsilon= self.HP_dict['eps_start_val']

        umax=np.max(self.u_sim,axis=0)
        umin=np.min(self.u_sim,axis=0)

        # Determine max and min vals of sampling ranges based on values from simulation
        self.max_vals=np.max(self.y_sim,axis=0)
        self.min_vals=np.min(self.y_sim,axis=0)

        # Append the min max values for the excitation
        for i in range(self.HP_dict['u_degree']+1):
            self.max_vals=np.hstack([self.max_vals,umax])
            self.min_vals=np.hstack([self.min_vals,umin])

        # Define margins to add onto sampling ranges (ratio and constant value)
        add_interval_ratio=(self.max_vals-self.min_vals)*0.1
        add_interval=0.0001

        # Append sampling ranges for the time
        self.max_vals=np.hstack([self.max_vals,self.HP_dict['tmax']])
        self.min_vals=np.hstack([self.min_vals,[0]])

        self.props_max=self.max_vals.copy()
        self.props_max[:-1]+=add_interval_ratio/2
        self.props_max[:-1]+=add_interval

        self.props_min=self.min_vals.copy()

        self.props_min[:self.y_sim.shape[1]]-=add_interval_ratio[:self.y_sim.shape[1]]/2 # pressures do not go below environmental pressure
        self.props_min[:self.y_sim.shape[1]]-=add_interval

        # Align sampling ranges for pressures (improves loss balancing)
        self.props_max[self.y_sim.shape[1]:-1]=np.max(self.props_max[self.y_sim.shape[1]:-1])
        self.props_min[self.y_sim.shape[1]:-1]=np.min(self.props_min[self.y_sim.shape[1]:-1])

        # # Special settings for sampling ranges and supplemental properties
        supp_props_min=np.array([])
        supp_props_max=np.array([])
        self.supp_vals = np.array([])

        if (self.HP_dict['var_bend_comp']==True):
            # Define sampling ranges of bending compliance
            supp_props_min=np.append(supp_props_min,self.rod_dict['C4']*0.5)
            supp_props_max=np.append(supp_props_max,self.rod_dict['C4']*1.5)
            self.supp_vals=np.append(self.supp_vals,self.rod_dict['C4']) # Which supp values to use for 'plot_model_run'

        if (self.HP_dict['var_tip_mass']==True):
            # Define sampling ranges of bending compliance
            supp_props_min=np.append(supp_props_min,0.001)
            supp_props_max=np.append(supp_props_max,self.rod_dict['M_tip']*1.5)
            self.supp_vals=np.append(self.supp_vals,self.rod_dict['M_tip']) # Which supp values to use for 'plot_model_run'

        # Insert sampling range of bending coompliance
        self.props_max=to_device(torch.from_numpy(np.hstack([self.props_max[:self.y_sim.shape[1]],supp_props_max,self.props_max[self.y_sim.shape[1]:]])),self.device)
        self.props_min=to_device(torch.from_numpy(np.hstack([self.props_min[:self.y_sim.shape[1]],supp_props_min,self.props_min[self.y_sim.shape[1]:]])),self.device)
        self.max_vals=to_device(torch.from_numpy(np.hstack([self.max_vals[:self.y_sim.shape[1]],supp_props_max,self.max_vals[self.y_sim.shape[1]:]])),self.device)
        self.min_vals=to_device(torch.from_numpy(np.hstack([self.min_vals[:self.y_sim.shape[1]],supp_props_min,self.min_vals[self.y_sim.shape[1]:]])),self.device)

        # # Generating state header
        self.state_header=[]

        for i in range(len(self.props)*self.n_state_nodes):
            idx=str(int(np.floor((i+len(self.props)/2)/len(self.props))))
            self.state_header.append(self.props[(i)%len(self.props)]+'_'+idx)

        self.state_header+=self.add_props
        self.state_header+=self.supp_props

        u_suffix=['_a','_b','_c']
        for i in range(self.HP_dict['u_degree']+1):
            for u_now in self.u_props:
                self.state_header.append(u_now+u_suffix[i])

        self.state_header.append('dt')

        print('Model input states and parameters:')
        print(self.state_header)

        self.xdim=len(self.props)*self.n_state_nodes
        self.udim=len(self.u_props)*(self.HP_dict['u_degree']+1)
        self.adim=len(self.add_props)
        self.sdim=len(self.supp_props)
        
        self.input_dim = self.xdim+self.adim+self.sdim+self.udim+1 # +1 for time t
        self.output_dim=self.xdim+self.adim

        self.n_collo=int(self.HP_dict['batch_size_collo']*self.HP_dict['n_batches'])
        self.n_ic=int(self.HP_dict['batch_size_ic']*self.HP_dict['n_batches']) # Calculate number of initial condition points per epoch

        # Determine normalization values stored in dictionaries m_x, b_x

        m_x_vals=2/(self.props_max-self.props_min)
        b_x_vals=-2*self.props_min/(self.props_max-self.props_min)-1
        self.m_x={}
        self.b_x={}

        for i in range(len(self.props_max)):
            self.m_x[self.state_header[i]]=m_x_vals[i].detach().item()
            self.b_x[self.state_header[i]]=b_x_vals[i].detach().item()

        self.data_points=np.zeros([self.HP_dict['n_data'],self.input_dim+self.output_dim])

        # Determine "state_one_vector", that disables states close to 0
        # Will be set to one if 'disable_states_flag' == False
        threshold=1e-2 # Threshold for physical value of a state to be disabled
        temp_max_vals=self.max_vals-self.min_vals
        self.state_one_vector =torch.where(temp_max_vals < threshold, 0, 1)

        if self.HP_dict['disable_states_flag']:
            self.b_x=multiply_dict_values(self.b_x,self.state_one_vector)
        else:
            self.state_one_vector=torch.ones(len(self.state_header))


    def initSaveDir(self):
        if self.save_everything:
            today = datetime.now()
            Datum = today.strftime("%d_%m_%Y__%H_%M")
            self.base_folder = Path(self.cwd, 'results', 'pinn_training', Datum)
            os.makedirs(self.base_folder, exist_ok=True)

    def initTraining(self,new_run,load_folder):
        if new_run: # Create new PINN/ DD-PINN model
            if self.HP_dict['nn_type'] == 'DDNN':
                self.model = DDNN(self.xdim, self.adim, self.sdim, self.udim, self.state_one_vector, self.HP_dict,device=self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.HP_dict['lr_init'], weight_decay=self.HP_dict['optim_w_decay'])

        else: # Load state dict of model and optimizer from previous run

            load_last_epoch = get_last_subfolder(load_folder)
            model_path = os.path.join(load_last_epoch, 'model_checkpoint.pt')
            optimizer_path = os.path.join(load_last_epoch, 'optimizer_checkpoint.pt')

            if self.HP_dict['nn_type'] == 'DDNN':
                self.model = DDNN(self.xdim, self.adim, self.sdim, self.udim, self.state_one_vector, self.HP_dict,device=self.device)

            self.model.load_state_dict(torch.load(model_path))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.HP_dict['lr_init'])
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.HP_dict['lr_init']

        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.HP_dict['LRS_schedule_red'], patience=self.HP_dict['LRS_patience'])
        # Init losses
        self.train_DL_collo_Pinn=None
        self.val_DL_collo_Pinn=None
        self.train_DL_ic_Pinn=None
        self.val_DL_ic_Pinn=None
        self.train_DL_data_Pinn=None
        self.val_DL_data_Pinn=None

        dict_now={}
        dict_now["ic"]=[]
        dict_now["data"]=[]

        for ii in range(2,self.HP_dict['n_ODE']+2):
            dict_now['phys_ODE'+str(ii-1)]=[]
            
        dict_now["total"]=[]
        self.train_loss_history=copy.deepcopy(dict_now)
        self.train_loss_history_unscaled=copy.deepcopy(dict_now)

        self.val_loss_history=copy.deepcopy(dict_now)
        self.val_loss_history_unscaled=copy.deepcopy(dict_now)

        self.loss_lam_history=None
        self.loss_lam_final=None

        self.n_losses=len(dict_now)-1 # 'n_losses' does not include total loss
        self.loss_lam_init=[]
        self.w_causal_train_avg=[]

        self.loss_str = give_loss_str(self.HP_dict, True)
        self.lr_vals = np.zeros(self.HP_dict['num_epochs'])
        self.epsilon_vals = np.zeros(self.HP_dict['num_epochs'])

        # Init early stopping and tracking of model with best validation loss
        self.best_val_loss = np.inf
        self.best_epoch = None
        self.best_model_state = None
        self.best_optimizer_state = None
        self.epochs_no_improve = 0

    def train(self):
        # Training loop
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping of Dataframe output
        self.df_stats = pd.DataFrame() # Dataframe for storing training stats

        # start_timer = timer()
        dur_total = 0
        for epoch in range(self.HP_dict['num_epochs']):
            epoch_start_timer = timer()

            # Train one epoch
            # with torch.profiler.profile(
            #         activities=[
            #             torch.profiler.ProfilerActivity.CPU,
            #             torch.profiler.ProfilerActivity.CUDA,
            #         ],
            #         with_stack=True,
            #     ) as prof:
            start = time.time()
            self.trainOneEpoch(epoch)
            synchronize(self.device)  # Ensure all operations are complete before measuring time
            print(f"train one epoch took {time.time()-start} s")
                

            # print(prof.key_averages().table(
            #         sort_by="self_cpu_time_total",
            #         row_limit=50
            # ))

            if self.nan_flag: # Stop training if loss is nan
                print("End of training!")
                break

            # Update train loss history and determine timer
            self.train_loss = self.train_loss_history["total"][-1]
            val_loss = self.val_loss_history_unscaled["total"][-1]
            self.lr_vals[epoch] = get_lr(self.optimizer)
            self.epsilon_vals[epoch] = self.epsilon

            # Keep copy of model with best validation loss and count epochs without improvement for early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            early_stop = self.HP_dict['early_stopping_flag'] and self.epochs_no_improve >= self.HP_dict['ES_patience']

            epoch_end_timer = timer()
            dur = epoch_end_timer - epoch_start_timer
            dur_total += dur

            # Remaining time from the mean epoch duration up to num_epochs; upper bound, since the training
            # stops earlier at lr_min or by early stopping (plotting and saving are not included)
            time_left = dur_total / (epoch + 1) * (self.HP_dict['num_epochs'] - (epoch + 1)) / 60
            print("Epoch #" + str(epoch) + " took " + str(round(dur, 1)) + " seconds -> at most " + str(round(time_left)) +
                " minutes of training left!\n" + #loss_message +
                "\n###")

            # Append stats to dataframe and print progress
            self.df_stats.loc[epoch, ['Epoch', 'delta_time', 'total_time', 'lr'] + ['train_' + item + '_loss' for item in give_loss_str(self.HP_dict,True)] + ['val_' + item + '_loss' for item in give_loss_str(self.HP_dict, True)] ] = [epoch, dur, dur_total, self.lr_vals[epoch]] + [self.train_loss_history_unscaled[loss][-1] for loss in give_loss_str(self.HP_dict,True)] + [self.val_loss_history_unscaled[loss][-1] for loss in give_loss_str(self.HP_dict, True)]
            if self.local_run:
                print(self.df_stats.tail())
            else:
                print(self.df_stats.tail(1))

            # Plot and save figures each 'plot_interval' epochs
            if (epoch % self.HP_dict['plot_interval'] == 0 and epoch != 0) or epoch==self.HP_dict['num_epochs']-1 or early_stop:
                if self.save_everything: # without a result folder (save_everything=False) nothing is plotted or saved
                    epoch_folder = os.path.join(self.base_folder, f'epoch_{epoch:04}')
                    try:
                        os.mkdir(epoch_folder)
                    except:
                        print(str(epoch_folder) + " already exists")
                    self.plot_loss_history(epoch, epoch_folder)
                    self.plot_model_run(epoch_folder)
                    self.saveCheckpoint(epoch_folder)
                # Check for learning rate minimum criteria
                if self.lr_vals[epoch] <= self.HP_dict['lr_min']:
                    print('LR mininum is reached. Done.')
                    break
                # Check for early stopping criteria
                if early_stop:
                    print('No improvement of validation loss for ' + str(self.HP_dict['ES_patience']) + ' epochs. Early stopping. Done.')
                    break

            # Run learning rate scheduler step
            if epoch % self.HP_dict['resampling_period'] != 0:
                self.scheduler.step(val_loss)

        # Restore and save model with best validation loss
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.optimizer.load_state_dict(self.best_optimizer_state)
            print('Restored model of epoch #' + str(self.best_epoch) + ' with best validation loss ' + str(self.best_val_loss))
            if self.save_everything:
                best_folder = os.path.join(self.base_folder, f'best_epoch_{self.best_epoch:04}')
                os.makedirs(best_folder, exist_ok=True)
                self.plot_model_run(best_folder)
                self.saveCheckpoint(best_folder)

    def saveCheckpoint(self,epoch_folder):
        pinn_io.write_weights(self.model.state_dict(), Path(epoch_folder, "weights.json")) # plain-text weights for MATLAB and the NEMPC
        torch.save(self.model.state_dict(), Path(epoch_folder, "model_checkpoint.pt")) # model and optimizer state to continue the training
        torch.save(self.optimizer.state_dict(), Path(epoch_folder, "optimizer_checkpoint.pt"))
        self.df_stats.to_pickle(Path(epoch_folder,"df_stats.pkl"))
        hp_file_path = Path(epoch_folder, 'hyperparameters.json')
        rod_params_file_path = Path(epoch_folder, 'rod_params.json')
        m_x_file_path = Path(epoch_folder, 'm_x.json')
        b_x_file_path = Path(epoch_folder, 'b_x.json')
        sh_file_path = Path(epoch_folder, 'state_header.txt')
        with open(hp_file_path, 'w') as convert_file: 
            convert_file.write(json.dumps(self.HP_dict))
        with open(rod_params_file_path, 'w') as convert_file: 
            convert_file.write(json.dumps(self.rod_dict))
        with open(m_x_file_path, 'w') as convert_file: 
            convert_file.write(json.dumps(self.m_x))            
        with open(b_x_file_path, 'w') as convert_file: 
            convert_file.write(json.dumps(self.b_x))
        with open(sh_file_path, 'w') as convert_file: 
            convert_file.write(json.dumps(self.state_header))
                        

    def trainOneEpoch(self,epoch):
        
        if epoch% self.HP_dict['resampling_period']==0:
            
            #Sample collocation points and initial condition points und merge with data_points
            print("Resampling collocation and initial condition points")
            # Returns: train_DL_collo,val_DL_collo,train_DL_ic,val_DL_ic,train_DL_data,val_DL_data
            self.sampleColloInit()

        for train_mode in [True,False]:
            if train_mode:
                dataloader_collo=self.train_DL_collo
                dataloader_ic=self.train_DL_ic
                dataloader_data=self.train_DL_data
            else:
                dataloader_collo=self.val_DL_collo
                dataloader_ic=self.val_DL_ic
                dataloader_data=self.val_DL_data
                
            torch.set_grad_enabled(True) # Required for PINN for building physics loss (autograd)
            
            self.model.train()

            loss_sum = 0
            ic_loss_sum = 0
            data_loss_sum = 0
            loss_sum_unscaled = 0

            phys_loss_ODE_sum=np.zeros(self.HP_dict['n_ODE'])
            if train_mode:
                w_causal_train_avg_now=np.zeros(self.HP_dict['batch_size_collo'])
            if epoch==0 and train_mode:
                self.w_causal_train_avg=np.zeros(self.HP_dict['batch_size_collo'])
                self.w_causal_train_avg[0]=1
                self.loss_lam_final=np.ones([self.n_losses])
                self.loss_lam_init=np.ones([self.n_losses])
                self.loss_lam_history=np.ones([1,self.n_losses])
            # start_train_time = time.time()
            # Asynchronous host-to-device copies only with pinned memory (CUDA); on MPS the copy
            # can read the temporary float32 buffer after it has been freed (garbage/NaN inputs)
            non_blocking = self.device.type == "cuda"
            # Loop through batches
            for i, (X_daten_collo, X_daten_ic, (X_daten_data, Y_daten_data)) in enumerate(zip(dataloader_collo, dataloader_ic,dataloader_data)):
                # start_batch = time.time()
                
                X_daten_collo = torch.reshape(X_daten_collo, (self.HP_dict['batch_size_collo'], self.input_dim)).to(self.device, dtype=torch.float32, non_blocking=non_blocking) #bereits skaliert
                X_daten_ic = torch.reshape(X_daten_ic, (self.HP_dict['batch_size_ic'], self.input_dim)).to(self.device, dtype=torch.float32, non_blocking=non_blocking) #bereits skaliert

                X_daten_data = torch.reshape(X_daten_data, (self.HP_dict["batch_size_data"], self.input_dim)).to(self.device, dtype=torch.float32, non_blocking=non_blocking) #bereits skaliert
                Y_daten_data = torch.reshape(Y_daten_data, (self.HP_dict["batch_size_data"], self.output_dim)).to(self.device, dtype=torch.float32, non_blocking=non_blocking) #bereits skaliert


                # Time-causal training: Sort current batch by t
                if self.HP_dict['tct_flag']:
                    _,sorted_ts_idx=torch.sort(X_daten_collo[:,-1],dim=0)
                    X_daten_collo=X_daten_collo[sorted_ts_idx,:]

                # device-aware placeholders (keep all intermediates on GPU)
                phys_loss_ODE = torch.zeros(self.HP_dict['n_ODE'], device=self.device, dtype=torch.float32)
                ic_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                data_loss = torch.zeros((), device=self.device, dtype=torch.float32)

                # per-batch model outputs / derivatives on device
                Y_dach = torch.zeros([self.HP_dict['batch_size_collo'], self.output_dim], device=self.device, dtype=torch.float32)
                dYdt_dach = torch.zeros([self.HP_dict['batch_size_collo'], self.output_dim], device=self.device, dtype=torch.float32)
                Y_dach_rescal = torch.zeros([self.HP_dict['batch_size_collo'], self.output_dim], device=self.device, dtype=torch.float32)
                dYdt_dach_rescal = torch.zeros([self.HP_dict['batch_size_collo'], self.output_dim], device=self.device, dtype=torch.float32)
                
                # Determine NN prediction and derivative for normal PINN 'FNN', or DD-PINN 'DDNN'
                if self.HP_dict['nn_type']=='FNN':
                    for k in range(self.HP_dict['batch_size_collo']):
                        X_step = X_daten_collo[k,:-1]
                        T_step = X_daten_collo[k,-1]

                        X_step=X_step.detach() 
                        X_step.requires_grad=False
                        T_step=T_step.detach()
                        T_step.requires_grad=True
                        Y_dach[k,:] = self.model(torch.hstack([X_step,T_step])) 

                        # Autograd
                        T_step.retain_grad()
                        try:
                            T_step.grad.zero_()
                            X_step.grad.zero_()
                        except:
                            donothing=0
                        for column in range(self.xdim):
                            temp,=grad(Y_dach[k,column],T_step,retain_graph=True,create_graph=True)
                            dYdt_dach[k,column]=temp

                elif 'DDNN':
                    Y_dach,dYdt_dach = self.model(X_daten_collo,ddt_flag=True,train_flag=True)

                if torch.any(torch.isnan(Y_dach)):
                    print("NaN in Y_dach. Replacing loss with 1e6")


                ### Determine excitation u for 'u_degree' > 0
                t_pos=((X_daten_collo[:,-1]+1)*self.HP_dict["t_sample_extra"]/2).unsqueeze(1)
                udim_single=len(self.u_props)
                us_rescal=normalize_list(X_daten_collo[:,self.xdim+self.adim+self.sdim:-1],self.state_header[self.xdim+self.adim+self.sdim:-1],self.m_x,self.b_x,False,False,False)
                if self.HP_dict['u_degree'] == 0:
                    u_rescal = us_rescal
                elif self.HP_dict['u_degree'] == 1:
                    u_rescal1, u_rescal2 = us_rescal[:,:udim_single], us_rescal[:,udim_single:]
                    u_rescal = u_rescal2 * t_pos + (1 - t_pos) * u_rescal1

                elif self.HP_dict['u_degree'] == 2:
                    u_rescal1, u_rescal2, u_rescal3 = us_rescal[:,:udim_single], us_rescal[:,udim_single:2*udim_single], us_rescal[:,2*udim_single:]
                    u_rescal = (2 * u_rescal1 - 4 * u_rescal2 + 2 * u_rescal3) * t_pos**2 + (-3 * u_rescal1 + 4 * u_rescal2 - u_rescal3) * t_pos + u_rescal1


                Y_dach_rescal=normalize_list(Y_dach,self.state_header[:self.output_dim],self.m_x,self.b_x,False,False,False)
                dYdt_dach_rescal=normalize_list(dYdt_dach,self.state_header[:self.output_dim],self.m_x,self.b_x,True,False,False)
                if self.sdim>0:
                    Supp_rescal=normalize_list(X_daten_collo[:,self.xdim+self.adim:self.xdim+self.adim+self.sdim],self.state_header[self.xdim+self.adim:self.xdim+self.adim+self.sdim],self.m_x,self.b_x,False,False,False)

                # ODE loss calculation
                # This section needs to be adapted based on available ODE loss functions
                    
                if self.HP_dict['model_dims']==3:
                    param_gravity  = torch.tensor([0.0,0.0,9.81,self.rod_dict["ref_length"],self.rod_dict["M3"],self.rod_dict["M_tip"]],device=self.device, dtype=torch.float32)
                    x=Y_dach_rescal.transpose(0,1)*self.state_one_vector[:self.xdim+self.adim].unsqueeze(1)
                    xt=dYdt_dach_rescal.transpose(0,1)*self.state_one_vector[:self.xdim+self.adim].unsqueeze(1)
                    u_rescal=u_rescal*self.state_one_vector[self.xdim+self.adim:self.xdim+self.adim+len(self.u_props)].unsqueeze(0)
                    u_rescal_expanded=pinn_loss.actuation_model_pressure(u_rescal, [self.rod_dict["A_p"],self.p_ref,self.rod_dict["r_p"]])
                    current_input=u_rescal_expanded.transpose(0,1)
                    param_rod = torch.tensor([self.rod_dict["C1"],self.rod_dict["C2"],self.rod_dict["C3"],self.rod_dict["C4"],self.rod_dict["C5"],self.rod_dict["C6"],
                                            self.rod_dict["D1"],self.rod_dict["D2"],self.rod_dict["D3"],self.rod_dict["D4"],self.rod_dict["D5"],self.rod_dict["D6"],
                                            self.rod_dict["ref_length"],self.rod_dict["M1"],self.rod_dict["M2"],self.rod_dict["M3"],self.rod_dict["M4"],self.rod_dict["M5"],self.rod_dict["M6"],
                                            self.rod_dict["M_tip"],0, 0, 0, 0, 0, 1],device=self.device, dtype=torch.float32) # last values are xi0
                    if self.HP_dict['var_bend_comp']==True and self.HP_dict['var_tip_mass']==True:
                        bend_compliance = Supp_rescal[:,0].squeeze()
                        tip_mass = Supp_rescal[:,1].squeeze()
                    elif self.HP_dict['var_bend_comp']==True: # only variable bending compliance
                        bend_compliance = Supp_rescal.squeeze()
                        tip_mass = torch.tensor(self.rod_dict["M_tip"],device=self.device, dtype=torch.float32).unsqueeze(0)
                    elif self.HP_dict['var_tip_mass']==True: # only variable tip mass
                        bend_compliance = torch.tensor(self.rod_dict["C4"],device=self.device, dtype=torch.float32).unsqueeze(0)
                        tip_mass = Supp_rescal.squeeze()
                    else:
                        bend_compliance = torch.tensor(self.rod_dict["C4"],device=self.device, dtype=torch.float32).unsqueeze(0)
                        tip_mass = torch.tensor(self.rod_dict["M_tip"],device=self.device, dtype=torch.float32).unsqueeze(0)

                    # Physics model functions for the model version and number of discrete points
                    n_discr = int(self.HP_dict['n_discr'])
                    eta_pls, external_loads, implicit_ode = self.physics

                    eta = eta_pls(x, current_input, param_rod,bend_compliance) # current strain
                    u_p = external_loads(x, current_input, param_rod,bend_compliance) # external loads
                    if self.HP_dict['grav_flag']:
                        u_g = pinn_loss.gravity(eta,param_gravity,torch.tensor(n_discr,device=self.device, dtype=torch.float32),tip_mass) # gravitational loads
                    else:
                        n_g_loads = eta.shape[0] if eta.dim() == 2 else 6*eta.shape[1] # 6 load components per gravity node
                        u_g = torch.zeros([n_g_loads,u_p.shape[1]],device=self.device, dtype=torch.float32)
                    mat_loss = implicit_ode(x, xt, current_input, u_p, u_g, param_rod,bend_compliance,tip_mass).transpose(0,1)


                    # Replace NaN values with high loss
                    mat_loss[torch.isnan(mat_loss)] = 1e6

                    # Normalization of loss vector of implicit_ode
                    # Construct state header where q and n are switched to align with loss vector of implicit_ode
                    indices= []
                    for ii in range(self.n_state_nodes):
                        for j in range(len(self.props)):
                            indices.append(int(ii*len(self.props)+(j+len(self.props)/2)%len(self.props)))
                    indices = np.array(indices)
                    state_header_resorted = [self.state_header[iii] for iii in indices]+ self.state_header[self.xdim:self.xdim+self.adim] # TODO: check if this is correct

                    # sov_resorted = np.hstack([[self.state_one_vector.cpu()[iii] for iii in indices], self.state_one_vector.cpu()[self.xdim:self.xdim+self.adim]])
                    # mat_loss=normalize_list(mat_loss,state_header_resorted[:self.output_dim],self.m_x,self.b_x,False,True,True)
                    # mat_loss*=torch.tensor(sov_resorted[:self.output_dim])

                    mat_loss=normalize_list(mat_loss,state_header_resorted[:self.output_dim],self.m_x,self.b_x,False,True,True)

                # Time-causal training: Weighting
                ode_batch_loss=torch.sum(torch.square(mat_loss),dim=1)
                
                if self.HP_dict['tct_flag']:
                    w_causal_training=torch.ones(self.HP_dict['batch_size_collo'])
                    for ii in range(1,self.HP_dict['batch_size_collo']):
                        w_causal_training[ii]=torch.exp(-self.epsilon*torch.sum(ode_batch_loss[:ii]))            

                    if train_mode:
                        w_causal_train_avg_now+=w_causal_training.detach().cpu().numpy()/len(dataloader_collo)

                    phys_loss_ODE[0]=torch.mean(ode_batch_loss*w_causal_training.detach())/torch.mean(w_causal_training.detach()) # time-causal training
                
                else:
                    if self.HP_dict['n_ODE']==1:
                        phys_loss_ODE[0]=torch.sum(torch.mean(torch.square(mat_loss),dim=0))
                    if self.HP_dict['n_ODE']>1:
                        # If no special case defined, sum up loss vector to 'n_ODE' losses                    
                        for ii in range(self.HP_dict['n_ODE']):
                            phys_loss_ODE[ii]=torch.sum(torch.mean(torch.square(mat_loss),dim=0)[ii::self.HP_dict['n_ODE']])

                # IC loss calculation
                if self.HP_dict['ic_loss_flag']:
                    Y_dach_ic = self.model(X_daten_ic)       
                    ic_loss = torch.sum(torch.mean(torch.square(Y_dach_ic - X_daten_ic[:, :self.xdim+self.adim]), dim=0))
                else:
                    ic_loss = torch.tensor(0)

                # Data loss calculation
                if self.HP_dict['data_loss_flag']:
                    Y_dach_data = self.model(X_daten_data)       
                    data_loss = torch.sum(torch.mean(torch.square(Y_dach_data - Y_daten_data[:, :self.xdim+self.adim]), dim=0))  # experimental
                else:
                    data_loss = torch.tensor(0)                
                

                # Learning rate annealing (LRA)
                i_last = len(dataloader_collo)   
                if self.HP_dict['loss_balance_flag'] and i >= i_last - self.HP_dict['LRA_n_avg'] and self.HP_dict['loss_balance_type'] == 'LRA':
                    lam_target_now = np.ones(self.n_losses)

                    param_maxmean = torch.zeros(self.n_losses)
                    loss_list = [ic_loss,data_loss]
                    loss_list.extend(phys_loss_ODE)

                    for n_loss, loss_now in enumerate(loss_list):
                        
                        # Avoiding loss.backward() on non-existent ic or data loss
                        if n_loss==0 and not self.HP_dict['ic_loss_flag']:
                            lam_target_now[n_loss] = 1
                        elif n_loss==1 and not self.HP_dict['data_loss_flag']:
                            lam_target_now[n_loss] = 1
                        else:
                            self.model.zero_grad()

                            loss_now.backward(retain_graph=True)
                            for name, param in self.model.named_parameters():
                                if n_loss == self.HP_dict['LRA_max_loss']:
                                    param_maxmean[n_loss] += torch.mean(torch.abs((param.grad)))
                                else:
                                    param_maxmean[n_loss] += torch.mean(torch.abs((param.grad)))

                    for n_loss in range(self.n_losses):
                        if n_loss==0 and not self.HP_dict['ic_loss_flag']:
                            lam_target_now[n_loss] = 1
                        elif n_loss==1 and not self.HP_dict['data_loss_flag']:
                            lam_target_now[n_loss] = 1

                        elif n_loss != self.HP_dict['LRA_max_loss']:
                            lam_target_now[n_loss] = param_maxmean[self.HP_dict['LRA_max_loss']] / param_maxmean[n_loss]
                
                    try: lam_target=np.vstack([lam_target,lam_target_now])
                    except: lam_target=lam_target_now
                
                # Calculate overall loss for training
                loss = torch.zeros((), device=self.device, dtype=torch.float32)
                loss_unscaled = torch.zeros((), device=self.device, dtype=torch.float32)
                # Move loss weights to device
                loss_lam_init_torch = torch.from_numpy(self.loss_lam_init).to(device=self.device, dtype=torch.float32)
                loss_lam_final_torch = torch.from_numpy(self.loss_lam_final).to(device=self.device, dtype=torch.float32)

                if self.HP_dict['ic_loss_flag']:
                    loss+=loss_lam_init_torch[0]*loss_lam_final_torch[0]*ic_loss
                    loss_unscaled+=ic_loss.clone().detach()

                if self.HP_dict['data_loss_flag']:
                    loss+=loss_lam_init_torch[1]*loss_lam_final_torch[1]*data_loss
                    loss_unscaled+=data_loss.clone().detach()

                for ii in range(2,self.HP_dict['n_ODE']+2):
                    loss+=loss_lam_init_torch[ii]*loss_lam_final_torch[ii]*phys_loss_ODE[ii-2]
                    loss_unscaled+=phys_loss_ODE[ii-2]

                # Stop training if loss is nan
                self.nan_flag=False
                if torch.isnan(loss):
                    #print(torch.sum(in4,dim=1))
                    print(Y_dach)
                    print(loss)
                    print(ic_loss)
                    print(data_loss)
                    print(self.loss_lam_init)
                    print(self.loss_lam_final)
                    print(phys_loss_ODE)
                    print(f'Loss is NaN!')
                    self.nan_flag=True
                
                # Summation of scaled losses
                if self.n_collo>0: 
                    for ii in range(2,self.HP_dict['n_ODE']+2):
                        phys_loss_ODE_sum[ii-2]+=self.loss_lam_init[ii]*self.loss_lam_final[ii]*phys_loss_ODE[ii-2].detach().cpu().item()

                if self.n_ic>0: ic_loss_sum+=self.loss_lam_init[0]*self.loss_lam_final[0] * ic_loss.detach().cpu().item()
                if self.HP_dict['n_data']>0: data_loss_sum+=self.loss_lam_init[1]*self.loss_lam_final[1] * data_loss.detach().cpu().item()


                loss_sum += loss.detach().cpu().item()
                loss_sum_unscaled += loss_unscaled.detach().cpu().item()

                # Optimizer step
                if train_mode:
                    self.optimizer.zero_grad()
                    # loss.backward(retain_graph=True)
                    loss.backward()
                    # # Debugging: Check for parameters without gradients
                    # for name, p in self.model.named_parameters():
                    #     if p.grad is None:
                    #         print(f"No grad for {name}")
                    if self.HP_dict["clip_gradients"]:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.HP_dict["max_grad_norm"])
                    
                    self.optimizer.step()
            #     if self.device.type == 'cuda':
            #         torch.cuda.current_stream().synchronize()  # Ensure all operations are complete before measuring time
            #     print(f"One batch took {time.time()-start_batch} s")

            # if self.device.type == 'cuda':
            #     torch.cuda.current_stream().synchronize()  # Ensure all operations are complete before measuring time
            # print(f"All batches took {time.time()-start_train_time} s")

            # Time-causal training: Annealing of epsilon
            if self.HP_dict['tct_flag']:
                self.w_causal_train_avg=w_causal_train_avg_now
                if epoch>=self.HP_dict['eps_annealing_wait'] and train_mode:
                    if np.mean(self.w_causal_train_avg)>self.HP_dict['eps_weight_criteria']:
                        self.epsilon*=10

            # Calculate average loss for epoch
            n_batches_local = len(dataloader_collo)
            loss_mean = loss_sum / n_batches_local

            loss_ODE_mean=phys_loss_ODE_sum/ n_batches_local
            loss_ic_mean = ic_loss_sum/ n_batches_local
            loss_data_mean = data_loss_sum/ n_batches_local

            loss_mean_unscaled = loss_sum_unscaled/ n_batches_local
            loss_ic_mean_unscaled = ic_loss_sum/ n_batches_local/self.loss_lam_final[0]/self.loss_lam_init[0]
            loss_data_mean_unscaled = data_loss_sum/ n_batches_local/self.loss_lam_final[1]/self.loss_lam_init[0]

            loss_ODE_mean_unscaled = np.zeros(self.HP_dict['n_ODE'])
            for ii in range(2,self.HP_dict['n_ODE']+2):
                loss_ODE_mean_unscaled[ii-2] = phys_loss_ODE_sum[ii-2] / n_batches_local/self.loss_lam_final[ii]/self.loss_lam_init[ii]


            # Initial naive loss balancing and rescaling
            if self.HP_dict['loss_balance_flag'] and self.HP_dict['loss_balance_type']=="Naive" and epoch%self.HP_dict['n_epoch_redo_balance']==0 and train_mode:
                # calculate init losses
                n_loss_effective=self.n_losses-(not self.HP_dict['ic_loss_flag'])-(not self.HP_dict['data_loss_flag'])
                if self.HP_dict['ic_loss_flag']:
                    self.loss_lam_init[0]=1/loss_ic_mean*self.loss_lam_init[0]
                if self.HP_dict['data_loss_flag']:
                    self.loss_lam_init[1]=1/loss_data_mean*self.loss_lam_init[1]

                for ii in range(self.HP_dict['n_ODE']):
                    self.loss_lam_init[ii+2]=1/loss_ODE_mean[ii]*self.loss_lam_init[ii+2]

                    self.val_loss_history["total"]
                if epoch>0:
                    self.loss_lam_init=self.loss_lam_init*loss_mean # Rescaling so that total loss' nominal value is consistent with previous value
                    
                self.loss_lam_init/=n_loss_effective # Scale so that initial total loss is 1
                
            # Save loss history
            if train_mode:
                for ii in range(2,self.HP_dict['n_ODE']+2):
                    self.train_loss_history['phys_ODE'+str(ii-1)].append(loss_ODE_mean[ii-2])
                    self.train_loss_history_unscaled['phys_ODE'+str(ii-1)].append(loss_ODE_mean_unscaled[ii-2])
                self.train_loss_history["ic"].append(loss_ic_mean)
                self.train_loss_history["data"].append(loss_data_mean)
                self.train_loss_history["total"].append(loss_mean)
                self.train_loss_history_unscaled["ic"].append(loss_ic_mean_unscaled)
                self.train_loss_history_unscaled["data"].append(loss_data_mean_unscaled)
                self.train_loss_history_unscaled["total"].append(loss_mean_unscaled)

            else:           
                for ii in range(2,self.HP_dict['n_ODE']+2):
                    self. val_loss_history['phys_ODE'+str(ii-1)].append(loss_ODE_mean[ii-2])
                    self.val_loss_history_unscaled['phys_ODE'+str(ii-1)].append(loss_ODE_mean_unscaled[ii-2])
                self.val_loss_history["ic"].append(loss_ic_mean)
                self.val_loss_history["data"].append(loss_data_mean)
                self.val_loss_history["total"].append(loss_mean)   
                self.val_loss_history_unscaled["ic"].append(loss_ic_mean_unscaled)
                self.val_loss_history_unscaled["data"].append(loss_data_mean_unscaled)
                self.val_loss_history_unscaled["total"].append(loss_mean_unscaled)      

            # Calculate loss balance lambdas for LRA
            if self.HP_dict['loss_balance_flag']:  # & epoch>0:
                if self.P_dict['loss_balance_type'] == 'LRA':
                    try:
                        lam_target_mean = np.mean(lam_target[1:i, :], axis=0)
                    except:
                        lam_target_mean = lam_target

                    self.loss_lam_final = self.loss_lam_final * self.HP_dict['alpha_loss'] + (1 - self.HP_dict['alpha_loss']) * lam_target_mean
                    if train_mode and epoch > 0:
                        self.loss_lam_history = np.vstack([self.loss_lam_history, self.loss_lam_final])
                elif self.HP_dict['loss_balance_type'] == 'Naive' and train_mode and epoch > 0:
                    self.loss_lam_history=np.vstack([self.loss_lam_history,self.loss_lam_init])
                    
            elif epoch>0:
                if train_mode: self.loss_lam_history=np.vstack([self.loss_lam_history,self.loss_lam_final])

    # Sampling of collocation points
    def sampleColloInit(self):
        sampler = qmc.LatinHypercube(d=self.input_dim)
        collo_points = np.random.rand(self.n_collo, self.input_dim)

        l_bounds = -np.ones(self.input_dim)
        u_bounds = np.ones(self.input_dim)
        collo_points = qmc.scale(collo_points, l_bounds, u_bounds)

        if self.HP_dict["multivar_normal_dist"]:
            # Generate multivariate normal samples with mean at 0
            mean = np.zeros(self.xdim)
            std_deviation = self.HP_dict["multivar_dist_std"]

            covariance = np.eye(self.xdim) * std_deviation**2 # Identity covariance matrix for simplicity
            x_collo_points = np.random.multivariate_normal(mean, covariance, self.n_collo)
            #l_bounds_cov = -np.ones(input_dim)
            #u_bounds_cov = np.ones(input_dim)
            #collo_points = np.clip(collo_points, l_bounds_cov, u_bounds_cov)
            collo_points[:,:self.xdim]=x_collo_points

        # Constrain the range of excitation u_b, u_c for 'u_degree' >0
        if self.HP_dict['constr_u_sample_flag']:
            usdim = int(self.udim / (self.HP_dict['u_degree'] + 1)) # udim if u_degree was 0
            if self.HP_dict['u_degree'] > 0:
                du_range = self.HP_dict['constr_u_ratio']*2 / self.HP_dict['u_degree']
            for i in range(1, 1 + self.HP_dict['u_degree']):
                collo_points[:, self.xdim+self.adim+self.sdim + usdim * i:self.xdim +self.adim+self.sdim+ usdim * (i + 1)] = collo_points[:, self.xdim+self.adim+self.sdim + usdim * (i - 1):self.xdim+self.adim+self.sdim + usdim * i] + (np.random.rand(self.n_collo, usdim)*2-1) * du_range
            
        ic_points = sampler.random(n=self.n_ic)
        ic_points=qmc.scale(ic_points, l_bounds, u_bounds)
        ic_points[:,-1]=-1

        data_collo=np.zeros([self.n_collo,self.input_dim+1])
        data_ic=np.zeros([self.n_ic,self.input_dim+1])
        #data_data=np.zeros([np.max([HP_dict['n_data'],HPDict['n_batches']]),input_dim+1])
        data_data=np.zeros([np.max([self.HP_dict['n_data'],self.HP_dict['n_batches']]),self.input_dim+self.output_dim])


        if self.HP_dict['data_loss_flag']:
            data_data=self.data_points
        data_ic=ic_points
        data_collo=collo_points

        # Split data into training and validation data
        train_data_collo,val_data_collo=SplitData(data_collo,self.HP_dict['split_size'],self.n_collo,self.HP_dict['batch_size_collo'])
        train_data_ic,val_data_ic=SplitData(data_ic,self.HP_dict['split_size'],self.n_ic,self.HP_dict['batch_size_ic'])
        train_data_data,val_data_data=SplitData(data_data,self.HP_dict['split_size'],self.HP_dict['n_data'],self.HP_dict['batch_size_data'])
        

        # Split data into Batches using the Datensatz class
        train_data_collo=DatensatzTorchSimple(train_data_collo[:,:self.input_dim],self.input_dim, self.HP_dict['batch_size_collo'])
        val_data_collo=DatensatzTorchSimple(val_data_collo[:,:self.input_dim],self.input_dim, self.HP_dict['batch_size_collo'])
        train_data_ic=DatensatzTorchSimple(train_data_ic[:,:self.input_dim],self.input_dim, self.HP_dict['batch_size_ic'])
        val_data_ic=DatensatzTorchSimple(val_data_ic[:,:self.input_dim],self.input_dim, self.HP_dict['batch_size_ic'])
        # train_data_data=DatensatzTorchSimple(train_data_data[:,:input_dim],input_dim, HP_dict["batch_size_data"])
        # val_data_data=DatensatzTorchSimple(val_data_data[:,:input_dim],input_dim, HP_dict["batch_size_data"])
        train_data_data=DatensatzTorchData(train_data_data[:,:self.input_dim],train_data_data[:,self.input_dim:],self.input_dim, self.HP_dict["batch_size_data"])
        val_data_data=DatensatzTorchData(val_data_data[:,:self.input_dim],val_data_data[:,self.input_dim:],self.input_dim, self.HP_dict["batch_size_data"])

        # DataLoader
        self.train_DL_collo = DataLoader(train_data_collo,batch_size = 1,shuffle = False,pin_memory=self.device.type == "cuda")
        self.val_DL_collo = DataLoader(val_data_collo,batch_size = 1,shuffle = False,pin_memory=self.device.type == "cuda")
        self.train_DL_ic = DataLoader(train_data_ic,batch_size = 1,shuffle = False,pin_memory=self.device.type == "cuda")
        self.val_DL_ic = DataLoader(val_data_ic,batch_size = 1,shuffle = False,pin_memory=self.device.type == "cuda")
        self.train_DL_data = DataLoader(train_data_data,batch_size = 1,shuffle = False,pin_memory=self.device.type == "cuda")
        self.val_DL_data = DataLoader(val_data_data,batch_size = 1,shuffle = False,pin_memory=self.device.type == "cuda")

    def plot_loss_history(self,epoch,epoch_folder):
        num_colors = 20

        # Using the 'tab20' color map to generate up to 20 distinct colors
        col = plt.cm.tab20b(range(num_colors))
        n_plots=3
        if self.HP_dict['tct_flag']:
            n_plots+=1
        fig_unscaled,ax_loss1=plt.subplots(n_plots,1,figsize=(8,15))
        fig_unscaled_separate,ax_loss_sep=plt.subplots(len(self.loss_str),1,figsize=(8,16))

        # loss_lam_history shortened by ic or data loss if turned off
        loss_lam_history_cut=self.loss_lam_history
        if not self.HP_dict['ic_loss_flag']:
            loss_lam_history_cut=self.loss_lam_history[:,1:]
            if not self.HP_dict['data_loss_flag']:
                loss_lam_history_cut=loss_lam_history_cut[:,1:]
        elif not self.HP_dict['data_loss_flag']:
            loss_lam_history_cut= np.delete(self.loss_lam_history, 1, 1)
        for idl,loss_now in enumerate(self.loss_str):
            
            ax_loss1[0].plot(np.arange(epoch+1),self.train_loss_history_unscaled[loss_now],'-s',c=col[idl],label='Train: '+loss_now)
            ax_loss1[1].plot(np.arange(epoch+1),self.val_loss_history_unscaled[loss_now],'--o',c=col[idl],label='Val: '+loss_now)
            
            #if HP_dict['loss_balance_flag'] and HP_dict['loss_balance_type']=="LRA":
            if idl<len(self.loss_str)-1: # There is no lambda for total loss
                loss_lam_now=loss_lam_history_cut[:,idl]
                ax_loss1[2].plot(np.arange(epoch+1),loss_lam_now,'-s',c=col[idl],label='Lam: '+loss_now)
            
            epoch_shift=self.HP_dict["LRS_patience"] # Number of epochs to plot in seperate loss plots

            ax_loss_sep[idl].plot(np.arange(np.max([epoch+1-epoch_shift,0]),epoch+1),self.train_loss_history_unscaled[loss_now][-epoch_shift:],'-s',c=col[idl],label='Train: '+loss_now)
            ax_loss_sep[idl].plot(np.arange(np.max([epoch+1-epoch_shift,0]),epoch+1),self.val_loss_history_unscaled[loss_now][-epoch_shift:],'--o',c=col[idl],label='Val: '+loss_now)
            ax_loss_sep[idl].set_yscale('log')
            ax_loss_sep[idl].set_xlabel('Epoch')
            ax_loss_sep[idl].grid()
            ax_loss_sep[idl].set_title('Unscaled loss: '+loss_now)
            
        ax_loss1[0].plot(np.arange(epoch+1),self.lr_vals[:epoch+1],c='k',label='lr')

        if self.HP_dict['tct_flag']: # Time-causal training: Plot epsilon
            ax_loss1[0].plot(np.arange(epoch+1),self.epsilon_vals[:epoch+1],c='cyan',label='epsilon')
            ax_loss1[1].plot(np.arange(epoch+1),self.epsilon_vals[:epoch+1],c='cyan',label='epsilon')

        ax_loss1[0].legend(loc='upper right',fontsize='small',bbox_to_anchor=(1.25, 1.0))
        ax_loss1[0].set_yscale('log')
        ax_loss1[0].set_xlabel('Epoch')
        ax_loss1[0].grid()
        ax_loss1[0].set_title('Unscaled losses: Train')
        ax_loss1[1].legend(loc='upper right',fontsize='small',bbox_to_anchor=(1.25, 1.0))
        ax_loss1[1].set_yscale('log')
        ax_loss1[1].set_xlabel('Epoch')
        ax_loss1[1].grid()
        ax_loss1[1].set_title('Unscaled Losses: Val')
        ax_loss1[2].legend(loc='upper right',fontsize='small',bbox_to_anchor=(1.25, 1.0))
        ax_loss1[2].set_yscale('log')
        ax_loss1[2].set_xlabel('Epoch')
        ax_loss1[2].grid()
        ax_loss1[2].set_title('Loss scaling lambdas')
        if self.HP_dict['tct_flag']:
            ax_loss1[3].plot(self.w_causal_train_avg,'-s')
            ax_loss1[3].plot(np.arange(len(self.w_causal_train_avg)),np.ones(len(self.w_causal_train_avg))*np.mean(self.w_causal_train_avg),'-')
            ax_loss1[3].set_title('Average of time-causal training weights')
            ax_loss1[3].set_ylim([0,1])

        plt.subplots_adjust(hspace=0.8)

        if self.save_everything:
            fig_unscaled.savefig(os.path.join(epoch_folder, 'training_progress_full.pdf'), format='pdf',bbox_inches="tight")
            fig_unscaled_separate.savefig(os.path.join(epoch_folder, 'training_progress_recent.pdf'), format='pdf',bbox_inches="tight")
        
        if self.local_run:
            plt.show()
        else:
            plt.close(fig_unscaled)
            plt.close(fig_unscaled_separate)

    def plot_model_run(self,epoch_folder):
        self.run_model()
        n_props=len(self.props)

        # t = torch.arange(len(self.xs_subs[:, 0]))/len(self.xs_subs[:, 0]) * self.HP_dict['t_range']
        t_big = torch.arange(len(self.xs[:, 0]))/len(self.xs[:, 0]) * self.HP_dict['t_range']

        col = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k']
        height = [15, 25, 35]
        n_plot_quat=0

        fig, axs = plt.subplots(n_props+n_plot_quat, 1, figsize=(12, height[self.HP_dict['model_dims'] - 1]))
        fig_zoom, axz = plt.subplots(n_props+n_plot_quat, 1, figsize=(12, height[self.HP_dict['model_dims'] - 1]))
        fig_err, ax_err = plt.subplots(n_props+n_plot_quat, 1, figsize=(12, height[self.HP_dict['model_dims'] - 1]))
        fig_exc, ax_exc = plt.subplots(1, 1, figsize=(12, 4))

        # Yuanda only
        q_max=np.array([220/180*np.pi]*6)
        q_max=np.hstack([q_max,np.array([141,141,169,180,180,180])/180*np.pi])

        for j in range(self.n_state_nodes):
            for k in range(n_props):
                idx_now = (j * n_props + k)  

                axz[k].set_xlim([0,1])

                for axn in [axs,axz]:
                    lw=0.7 # Line width
                    # if self.HP_dict['f_sample']<100:
                    #     axn[k].plot(t,self.xs_subs[:,idx_now].detach().numpy(),c=col[j],label='NN',linewidth=lw)
            
                    axn[k].scatter(t_big.cpu().detach().numpy(),self.xs[:,idx_now].cpu().detach().numpy(),s=3,c=col[j])
                    
                    axn[k].plot(self.t_sim,self.y_sim[:len(self.t_sim),idx_now],'--',c=col[j],label='Sim',alpha=0.7,linewidth=lw)

                    n_now=j
                    if k>=n_props/2: n_now+=1
                    axn[k].set_title(self.props[int(k%n_props)])
                    axn[k].grid()
                    if j==0: axn[k].legend()
                    if j==self.n_state_nodes-1: 
                        axn[k].set_xlabel('time t in s')
                        y_slice = self.y_sim[:, np.arange(k, self.y_sim.shape[1], step=n_props)]
                        ymin = 1.2 * np.min(y_slice)
                        ymax = 1.2 * np.max(y_slice)
                        axn[k].set_ylim(ymin, ymax)
                        # axn[k].set_ylim(1.2*self.y_sim[:,torch.arange(k,self.y_sim.shape[1], step=n_props)].min(), 1.2*self.y_sim[:,torch.arange(k,self.y_sim.shape[1], step=n_props)].max())

                state_range = np.max(self.y_sim[:len(self.t_sim),idx_now]) - np.min(self.y_sim[:len(self.t_sim),idx_now])
                errors = np.abs(self.xs[:,idx_now].cpu().detach().numpy()-self.y_sim[:len(self.t_sim)-1,idx_now])/state_range
                ax_err[k].plot(t_big.cpu().detach().numpy(),errors,c=col[j],label='Rel. Error',linewidth=lw)
                n_now=j
                if k>=n_props/2: n_now+=1
                ax_err[k].set_title(self.props[int(k%n_props)])
                ax_err[k].grid()
                if j==0: ax_err[k].legend()
                if j==self.n_state_nodes-1: 
                    ax_err[k].set_xlabel('time t in s')
                    # ax_err[k].set_ylim(1.2*y_sim[:,torch.arange(k,y_sim.shape[1], step=n_props)].min(), 1.2*y_sim[:,torch.arange(k,y_sim.shape[1], step=n_props)].max())

                
        # Excitation plot        
        ax_exc.plot(self.u_in_c.cpu().detach().numpy())
        ax_exc.legend(self.state_header[self.xdim+self.adim+self.sdim:self.xdim+self.adim+self.sdim+int(self.udim/(self.HP_dict['u_degree']+1))])

        # Determine path deviation to simulation
        y_sim_at_run = np.empty((len(t_big.cpu().detach().numpy()), self.xdim))
        # y_sim_at_run_subs = np.empty((len(t), self.xdim))
        for i in range(self.xdim):
            y_sim_at_run[:,i] = np.interp(t_big.cpu().detach().numpy(), self.t_sim[:,0], self.y_sim[:,i])
            # y_sim_at_run_subs[:,i] = np.interp(t, self.t_sim[:,0], y_sim[:,i])

        path_dev=self.xs[:,:self.xdim].cpu().detach().numpy()-y_sim_at_run
        # path_dev_subs=self.xs_subs[:,:self.xdim].detach().numpy()-y_sim_at_run_subs

        if self.local_run:
            plt.show()
        else:
            plt.close(fig)
            plt.close(fig_zoom) 
            plt.close(fig_err)          
            plt.close(fig_exc)          

        # Save all generated paths and simulation path as csv viles
        if self.save_everything:
            fig.savefig(os.path.join(epoch_folder, 'path_full.pdf'), format='pdf',bbox_inches="tight")
            fig_zoom.savefig(os.path.join(epoch_folder, 'path_zoom.pdf'), format='pdf',bbox_inches="tight")
            fig_err.savefig(os.path.join(epoch_folder, 'path_error.pdf'), format='pdf',bbox_inches="tight")
            fig_exc.savefig(os.path.join(epoch_folder, 'excitation.pdf'), format='pdf',bbox_inches="tight")
                
            xs_np = self.xs.cpu().detach().numpy() 
            dxs_dt_np = self.dxs_dt.cpu().detach().numpy() 
            # xs_subs_np = xs_subs.detach().numpy() 
            # dxs_dt_subs_np = dxs_dt_subs.detach().numpy() 

            csv_file_path = os.path.join(epoch_folder, 'path.csv')
            np.savetxt(csv_file_path, xs_np, delimiter=',', fmt='%.6e', header=",".join(self.state_header), comments='')

            csv_file_path_dt = os.path.join(epoch_folder, 'path_dt.csv')
            np.savetxt(csv_file_path_dt, dxs_dt_np, delimiter=',', fmt='%.6e', header=",".join(self.state_header[:self.xdim+self.adim]), comments='')

            # csv_file_path_subs = os.path.join(folder, 'path_subs.csv')
            # np.savetxt(csv_file_path_subs, xs_subs_np, delimiter=',', fmt='%.6e', header=",".join(state_header[:xdim+adim]), comments='')
            
            # csv_file_path_subs_dt = os.path.join(folder, 'path_subs_dt.csv')
            # np.savetxt(csv_file_path_subs_dt, dxs_dt_subs_np, delimiter=',', fmt='%.6e', header=",".join(state_header[:xdim+adim]), comments='')

            csv_file_path_dev = os.path.join(epoch_folder, 'path_dev.csv')
            np.savetxt(csv_file_path_dev, path_dev, delimiter=',', fmt='%.6e', header=",".join(self.state_header[:self.xdim]), comments='')

            # csv_file_path_dev_subs = os.path.join(folder, 'path_dev_subs.csv')
            # np.savetxt(csv_file_path_dev_subs, path_dev_subs, delimiter=',', fmt='%.6e', header=",".join(state_header[:xdim]), comments='')
            
            csv_file_path_sim = os.path.join(epoch_folder, 'path_sim.csv')
            np.savetxt(csv_file_path_sim, y_sim_at_run, delimiter=',', fmt='%.6e', header=",".join(self.state_header[:self.xdim]), comments='')
            
            # csv_file_path_sim_subs = os.path.join(folder, 'path_sim_subs.csv')
            # np.savetxt(csv_file_path_sim_subs, y_sim_at_run_subs, delimiter=',', fmt='%.6e', header=",".join(state_header[:xdim]), comments='')


    def run_model(self):
        n_t=self.HP_dict['f_sample']*self.HP_dict['t_range']
        n_subs=10 # Sub steps to plot in single shootin range between (0, T) # TODO: Try changing this value
        t_subs = torch.linspace(-1, normalizeVar(self.HP_dict['t_sample'], self.m_x['dt'], self.b_x['dt'], True) - (1 + normalizeVar(self.HP_dict['t_sample'], self.m_x['dt'], self.b_x['dt'], True)) / n_subs, n_subs)

        for t_in, t_case in zip([torch.ones(n_t) * self.HP_dict['t_sample'], torch.zeros(n_t)], [1, 0]):
            if t_case == 0:
                break

            t_in = normalizeVar(t_in, self.m_x['dt'], self.b_x['dt'], True)

            x_in = torch.zeros(n_t, self.xdim + self.adim+self.sdim+self.udim + 1)
            dxdt = torch.zeros(n_t, self.xdim+ self.adim)

            x_in_subs = torch.zeros(n_t * n_subs, self.xdim + self.adim+ self.sdim+ self.udim + 1)
            x_out_subs = torch.zeros((n_t-1) * n_subs, self.xdim+ self.adim)
            dxdt_subs = torch.zeros((n_t-1) * n_subs, self.xdim+ self.adim)

            t = torch.arange(0, 1*self.HP_dict['t_range']+1/self.HP_dict['f_sample'], 1/self.HP_dict['f_sample'])
            if self.HP_dict['u_degree']==2:
                t = torch.arange(0, 1*self.HP_dict['t_range']+1/self.HP_dict['f_sample'], 1/self.HP_dict['f_sample']/2)

            # Create excitation u_in
            if self.HP_dict['model_dims']==1:
                uf=(torch.tanh((t - 0.25) * 20) / 2) + (torch.tanh(-(t-0.5)*200))/2
                unz=uf
        
            elif self.HP_dict['model_dims']==2:
                uf=(torch.tanh((t - 0.25) * 20) / 2) + (torch.tanh(-(t-0.5)*50))/2
                uny=uf.clone().detach()
                unz=uf.clone().detach()*0.5
                umx=uf.clone().detach()*0.1

            elif self.HP_dict['model_dims']==3: # TOCO: Change to match input excitation
                    dt_on=1
                    dt_2nd=2
                    t_on1=0.5
                    t_on2=0.75
                    t_on3=1
                    t_off1=t_on1+dt_on
                    t_off2=t_on2+dt_on
                    t_off3=t_on3+dt_on
                    tanh_steep=5

                    AMP1 = (1/2*(torch.tanh(tanh_steep*(t-t_on1))-torch.tanh(tanh_steep*(t-t_off1))))#+(1/2*(torch.tanh(tanh_steep*(t-t_on1-dt_2nd))-torch.tanh(tanh_steep*(t-t_off1-dt_2nd))))
                    AMP2 = (1/2*(torch.tanh(tanh_steep*(t-t_on2))-torch.tanh(tanh_steep*(t-t_off2))))#+(1/2*(torch.tanh(tanh_steep*(t-t_on2-dt_2nd))-torch.tanh(tanh_steep*(t-t_off2-dt_2nd))))
                    AMP3 = (1/2*(torch.tanh(tanh_steep*(t-t_on3))-torch.tanh(tanh_steep*(t-t_off3))))#+(1/2*(torch.tanh(tanh_steep*(t-t_on3-dt_2nd))-torch.tanh(tanh_steep*(t-t_off3-dt_2nd))))
                    p0=1e5
                    T_ON=0.0
                    T_OFF=5.0
                    AMP=(1/2*(torch.tanh(50*(t-T_ON))-torch.tanh(50*(t-T_OFF))))
                    up1=AMP*AMP1*p0+p0
                    up2=AMP*AMP2*p0+p0
                    up3=AMP*AMP3*p0+p0

            if self.HP_dict['model_dims']==1:
                u_in=torch.vstack([unz]).transpose(0,1)
            elif self.HP_dict['model_dims']==2:
                u_in=torch.vstack([uny,unz,umx]).transpose(0,1)
            elif self.HP_dict['model_dims']==3:
                u_in=torch.vstack([up1,up2,up3]).transpose(0,1)


            u_in=normalize_list(u_in,self.state_header[self.xdim+self.adim+self.sdim:self.xdim+self.adim+self.sdim+len(self.u_props)],self.m_x,self.b_x,False,False,True)

            self.u_in_c=u_in.clone().detach()
            if self.HP_dict['u_degree']==0:
                u_in1=u_in[:-1]
            elif self.HP_dict['u_degree']==1:
                u_in1=u_in[:-1]
                u_in2=u_in[1:]
            elif self.HP_dict['u_degree']==2:
                self.u_in_c=self.u_in_c[::2]
                u_in1=u_in[:-2:2]
                u_in2=u_in[1:-1:2]
                u_in3=u_in[2::2]
    
            X0_normal=torch.zeros(self.xdim+self.adim+self.sdim)
            # if HP_dict['grav_flag']:
            #     X0_normal[xdim::4]=1 # Start values of quaternions are (1,0,0,0)
            
            # Set supplemental values (such as bending compliance for Cosserat)
            X0_normal[self.xdim+self.adim:]=torch.as_tensor(self.supp_vals, dtype=X0_normal.dtype)
            X0=normalize_list(X0_normal,self.state_header[:self.xdim+self.adim+self.sdim],self.m_x,self.b_x,False,False,True)
            x_in[0,:self.xdim+self.adim+self.sdim]=X0
            x_in_subs[0,:self.xdim+self.adim+self.sdim]=X0
            if self.sdim>0:
                x_out_subs[0,:self.xdim+self.adim]=X0[0,:-self.sdim]
            else:
                x_out_subs[0,:self.xdim+self.adim]=X0[0,:]

            
            # Running models
            for i in range(n_t-1):
                if self.HP_dict['u_degree']==0:
                    x_in[i,:]=torch.hstack([x_in[i,:self.xdim+self.adim+self.sdim],u_in1[i,:],t_in[i]])
                elif self.HP_dict['u_degree']==1:
                    x_in[i,:]=torch.hstack([x_in[i,:self.xdim+self.adim+self.sdim],u_in1[i,:],u_in2[i,:],t_in[i]])
                elif self.HP_dict['u_degree']==2:
                    x_in[i,:]=torch.hstack([x_in[i,:self.xdim+self.adim+self.sdim],u_in1[i,:],u_in2[i,:],u_in3[i,:],t_in[i]])


                if self.HP_dict['nn_type']=='DDNN':
                    x_out_now,dxdt_out_now=self.model(x_in[i,:].unsqueeze(0),ddt_flag=True)
                else:
                    x_out_now=self.model(x_in[i,:])

                for ii in range(n_subs):
                    
                    if self.HP_dict['u_degree']==0:
                        x_in_subs[i*n_subs+ii,:]=torch.hstack([x_in[i,:self.xdim+self.adim+self.sdim],u_in1[i,:],t_subs[ii]])
                    elif self.HP_dict['u_degree']==1:
                        x_in_subs[i*n_subs+ii,:]=torch.hstack([x_in[i,:self.xdim+self.adim+self.sdim],u_in1[i,:],u_in2[i,:],t_subs[ii]])
                    elif self.HP_dict['u_degree']==2:
                        x_in_subs[i*n_subs+ii,:]=torch.hstack([x_in[i,:self.xdim+self.adim+self.sdim],u_in1[i,:],u_in2[i,:],u_in3[i,:],t_subs[ii]])

                    if self.HP_dict['nn_type']=='DDNN':
                        x_out_now_sub,dxdt_out_now_sub=self.model(x_in_subs[i*n_subs+ii,:].unsqueeze(0),ddt_flag=True)
                    else:
                        x_out_now_sub=self.model(x_in_subs[i*n_subs+ii,:])
                    
                    if ii==0:
                        x_out_subs[i*n_subs+ii,:]=x_in[i,:-(1+self.udim+self.sdim)]

                    else: 
                        x_out_subs[i*n_subs+ii,:]=x_out_now_sub
                    
                    if self.HP_dict['nn_type']=='DDNN':
                        dxdt_subs[i*n_subs+ii,:]=dxdt_out_now_sub
                        
                x_in[i+1,:self.xdim+self.adim]=x_out_now
                if self.HP_dict['nn_type']=='DDNN':
                    dxdt[i+1,:]=dxdt_out_now
            
            # Rescaling of paths before saving them        
            self.xs=normalize_list(x_in,self.state_header,self.m_x,self.b_x,False,False,False)
            self.dxs_dt=normalize_list(dxdt,self.state_header[:self.xdim+self.adim],self.m_x,self.b_x,True,False,False)
            self.x_out_subs=normalize_list(x_out_subs,self.state_header[:self.xdim+self.adim],self.m_x,self.b_x,False,False,False)
            self.dxdt_subs=normalize_list(dxdt_subs,self.state_header[:self.xdim+self.adim],self.m_x,self.b_x,True,False,False)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_last_subfolder(directory):
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    if not subfolders:
        return None
    subfolders.sort()
    last_subfolder = subfolders[-1]
    return last_subfolder


def multiply_dict_values(dictionary, vector):
    if len(dictionary) != len(vector):
        raise ValueError("Lengths of dictionary and vector must match")
    multiplied_dict = {}
    for i, (key, value) in enumerate(dictionary.items()):
        multiplied_dict[key] = value * vector[i] 
    return multiplied_dict


class DatensatzTorchSimple(Dataset):
    def __init__(self, input_data,input_dim, batch_size):
        datensatz_pytorch = ()
        tensor_inputs = torch.from_numpy(input_data)
        self.datensatz_pytorch = datensatz_pytorch + torch.split(tensor_inputs, batch_size)
        self.input_dim=input_dim
    def __len__(self):
        return len(self.datensatz_pytorch)
    def __getitem__(self,idx):
        daten = self.datensatz_pytorch[idx]
        X_daten = daten[:,:self.input_dim]
        return X_daten

class DatensatzTorchData(Dataset):
    def __init__(self, input_data,output_data,input_dim, batch_size):
        datensatz_pytorch = () ## Definieren eines Tupels --> Nur hier für Erzeugung von Batches
        tensor_inputs = torch.from_numpy(input_data)
        tensor_outputs = torch.from_numpy(output_data)
        tensor_ges = torch.hstack((tensor_inputs,tensor_outputs))
        self.datensatz_pytorch = datensatz_pytorch + torch.split(tensor_ges, batch_size)
        self.input_dim=input_dim
    def __len__(self):
        return len(self.datensatz_pytorch)
    def __getitem__(self,idx):
        daten = self.datensatz_pytorch[idx]
        X_daten = daten[:,:self.input_dim]
        Y_daten = daten[:,self.input_dim:]
        return X_daten, Y_daten


def SplitData(data,split_size,n_data,batch_size):
    #split in train und val
    split_point = int(split_size * n_data)-int(split_size * n_data)%batch_size 
    train_data = data[:split_point,:]
    val_data = data[split_point:,:]# auf volle Batches noch kürzen
    n_cut=len(val_data)%batch_size
    if n_cut>0:
        val_data = data[split_point:-n_cut,:]
    return train_data,val_data

# Functions to use as activation function for FNN or DD-PINN single layer ansatz
class Exp_act(nn.Module):
    def forward(self, x):
        return torch.exp(x)
class Sine_act(nn.Module):
    def forward(self, x):
        return torch.sin(x)
def gaussian(x):
    return torch.exp(-((x) ** 2)/2)    
def gaussian_derivative(x):
    return -x * torch.exp(-((x) ** 2)/2)
def gelu_derivative(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + 0.5 * x * (1 - torch.erf(x / math.sqrt(2)).pow(2)) * math.sqrt(2 / math.pi)
def tanh_derivative(x):
    return 1 - torch.tanh(x)**2
def sigmoid_derivative(x):
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)
def sinc_derivative(x):
    xpi=math.pi*x
    return torch.where(x == 0, torch.tensor(0.0), (xpi*torch.cos(xpi)-torch.sin(xpi))/xpi**2)

class DDNN(nn.Module):
    def __init__(self,xdim,adim,sdim,udim,state_one_vector,HP_dict,device):
        super().__init__()
        self.device = device
        self.n_exp=HP_dict['n_sums_DDNN'] # Number of ansatz function in g

        self.ansatz=HP_dict['ansatz_type'] # Definition of the ansatz used
        self.f_n_param=4 # Numbers of parameter for ansatz function
        self.d_clamping = HP_dict['limit_deltas'] # Clamping of damping term d to negative values
        # Sine ansatz: 2: Sine with amplitude and frequency, 3: ... and phase shift, 4: ... and damping term

        self.xudim=xdim+adim+sdim+udim # Input dimensions except t
        self.xdim=xdim+adim # Supervised dimensions
        self.layer_in = nn.Linear(self.xudim, HP_dict['n_neurons'])
        self.layer_out = nn.Linear(HP_dict['n_neurons'],self.n_exp*self.xdim*self.f_n_param)
        
        # Activation function of the FNN preceding the g function
        self.act=nn.GELU()

        # Activation functions for DD-PINN g function if self.ansatz = "Single_Layer" is used
        # May use: Sin/Cos, GELU/GELU_der, Tanh/Tanh_der, Gaussian/Gaussian_der

        #self.act_g=nn.GELU()
        #self.act_g_derivative=gelu_derivative
        self.act_g=torch.sin
        self.act_g_derivative=torch.cos
        #self.act_g=torch.exp
        #self.act_g_derivative=torch.exp
        # self.act_g=gaussian
        # self.act_g_derivative=gaussian_derivative
        # self.act_g=torch.sigmoid
        # self.act_g_derivative=sigmoid_derivative
        # self.act_g=torch.sinc
        # self.act_g_derivative=sinc_derivative

        self.sov=state_one_vector.float() # S
        init.xavier_uniform_(self.layer_in.weight) # Xavier initilization of weights
        init.xavier_uniform_(self.layer_out.weight)

        layers=[]
        for i in range(HP_dict['n_hidden']):
            layers.append(nn.Linear(HP_dict['n_neurons'],HP_dict['n_neurons']))
            init.xavier_uniform_(layers[i].weight)
        self.layers_middle=nn.ModuleList(layers)
        self.n_hidden=HP_dict['n_hidden']

    # def forward(self, x,ddt_flag=False,train_flag=False):
    #     x=x*self.sov  # state one vector
    #
    #     if len(x.shape) == 1:
    #         if isinstance(x, np.ndarray):
    #             x = np.expand_dims(x, axis=0)
    #         else:
    #             x=x.unsqueeze(0)
    #
    #     x_in = x[:, :self.xdim]
    #     xu_in = x[:, :self.xudim]
    #     if isinstance(x, np.ndarray):
    #         t_in = (np.expand_dims(x[:, -1], axis=0).T + 1)
    #     else:
    #         t_in = (x[:, -1].unsqueeze(0).transpose(0,1)+1)
    #
    #     l_out =self.act(self.layer_in(xu_in.to(device, dtype=torch.float32)))
    #     if self.n_hidden>0:
    #         for layer in self.layers_middle:
    #             l_out=self.act(layer(l_out))        
    #     l_out=self.layer_out(l_out)
    #
    #     # Get parameters from the output layer a,b,c,d,e dependent on number needed for given ansatz function
    #     a=l_out[:,:self.n_exp*self.xdim]
    #     if self.f_n_param>1:
    #         b=l_out[:,self.n_exp*self.xdim:2*self.n_exp*self.xdim]
    #     if self.f_n_param>2:
    #         c=l_out[:,2*self.n_exp*self.xdim:3*self.n_exp*self.xdim]     
    #     if self.f_n_param>3:
    #         d=l_out[:,3*self.n_exp*self.xdim:4*self.n_exp*self.xdim]   
    #     if self.f_n_param>4:
    #         e=l_out[:,4*self.n_exp*self.xdim:]  
    #
    #     # Calculation of DD-PINN g function
    #     if self.ansatz=='Single_Layer': # Ansatz equivalent to a sparse single layer NN (Recommended)
    #         if self.f_n_param==3:
    #             g_out=a*(self.act_g(b*t_in+c)-self.act_g(c))
    #             dgdt_out=a*b*self.act_g_derivative(b*t_in+c)
    #
    #     if self.ansatz=='Single_Layer_damped': # Ansatz equivalent to a sparse single layer NN (Most recommended)
    #         if self.f_n_param==4:
    #             if self.d_clamping:
    #                 d=torch.clamp(d, min=-10)
    #             g_out=a*(self.act_g(b*t_in+c)*torch.exp(-d*t_in)-self.act_g(c))
    #             dgdt_out=a*(b*self.act_g_derivative(b*t_in+c)-d*self.act_g(b*t_in+c))*torch.exp(-d*t_in)
    #         if self.f_n_param==5:
    #             g_out=a*(self.act_g(b*t_in+c)*torch.exp(-d*t_in)-self.act_g(c))+e
    #             dgdt_out=a*(b*self.act_g_derivative(b*t_in+c)-d*self.act_g(b*t_in+c))*torch.exp(-d*t_in)
    #
    #     if self.ansatz=='Single_Layer_damped_lin': # Ansatz equivalent to a sparse single layer NN (Works well)
    #         if self.f_n_param==5:
    #             g_out=a*(self.act_g(b*t_in+c)*torch.exp(-d*t_in+e)-self.act_g(c)*torch.exp(e))
    #             dgdt_out=a*(b*self.act_g_derivative(b*t_in+c)-d*self.act_g(b*t_in+c))*torch.exp(-d*t_in+e)
    #
    #     elif self.ansatz=='Double_Layer': # Ansatz equivalent to a sparse double layer NN (Experimental, but works)
    #         if self.f_n_param==5:
    #             g_out=e*self.act_g(c*(self.act_g(a*t_in+b))+d)-e*self.act_g(c*(self.act_g(b))+d)
    #
    #             dgdt_out=e * self.act_g_derivative(c * self.act_g(a * t_in + b) + d) * c * self.act_g_derivative(a * t_in + b) * a
    #
    #     elif self.ansatz=='Single_Layer_sqr':  # Ansatz equivalent to a sparse modified single layer NN (Experimental, does not seem to work well)
    #         if self.f_n_param==4:
    #             g_out=a*(self.act_g(b*t_in**2+c*t_in+d)-self.act_g(d))
    #             dgdt_out=a*2*b*t_in*c*self.act_g_derivative(b*t_in**2+c*t_in+d)                
    #
    #     # Additional ansatz functions that did not work well (!) but are left for reference
    #     elif self.ansatz=='Parabel': # Sum of parabolas
    #         
    #         if self.f_n_param==2:
    #             g_out=a*(t_in-b)**2-a*b**2
    #             dgdt_out=(2*a*t_in-2*b)/2
    #
    #     elif self.ansatz=='Taylor': # Taylor series expansion. Degree is equal to 'n_sums_DDNN'
    #         t_in_dt=t_in
    #         a_dt=a
    #         for i in range(self.n_exp):
    #             t_in[i::self.n_exp]**=(i+1)
    #             t_in_dt[i::self.n_exp]**=(i)
    #             a[:,i::self.n_exp]/=torch.lgamma(torch.tensor(i+1+1)).exp()
    #             a_dt[:,i::self.n_exp]/=torch.lgamma(torch.tensor(i+1)).exp()
    #         
    #         axt=a*t_in
    #         axt_dt=a_dt*t_in_dt
    #         g_out=axt
    #         dgdt_out=axt_dt/2
    #
    #     elif self.ansatz=='Legendre': # Sum of Legendre polynomials
    #         def legendre_polynomial_offset(n, x):
    #             res=np.polynomial.legendre.Legendre.basis(n)(x)-np.polynomial.legendre.Legendre.basis(n)(-torch.ones_like(x))
    #             return res
    #         def legendre_derivative(n, x):
    #             p_n = np.polynomial.legendre.Legendre.basis(n)
    #             derivative = p_n.deriv()(x)
    #             return derivative
    #         
    #         g_out=torch.zeros_like(a)
    #         dgdt_out=torch.zeros_like(a)
    #         for i in range(self.n_exp):
    #             g_out[:,i::self.n_exp]=legendre_polynomial_offset(i, t_in)
    #             dgdt_out[:,i::self.n_exp]=legendre_derivative(i, t_in)
    #
    #         g_out*=a
    #         dgdt_out*=a/2
    #
    #     # Calculate output of function g
    #     g_out_sum=g_out.view(t_in.shape[0], self.xdim, self.n_exp).sum(dim=2)
    #     out=x_in+g_out_sum
    #     out=out*self.sov[:self.xdim]
    #
    #     # If ddt_flag==True, calculate and output derivative
    #     if ddt_flag:
    #         dgdt_out_sum=dgdt_out.view(len(t_in), self.xdim, self.n_exp).sum(dim=2)
    #         dout_dt=dgdt_out_sum
    #         dout_dt=dout_dt*self.sov[:self.xdim]
    #         return out,dout_dt
    #     else:
    #         return out
        
    # Forward pass for expoort
    def forward(self, x,ddt_flag=False,train_flag=False):
        x=x*self.sov  # state one vector

        x_in = x[:, :self.xdim]
        xu_in = x[:, :self.xudim]

        t_in = (x[:, self.xudim].unsqueeze(0).transpose(0,1)+1) # -1 indexing seems to be an edge case bug in matlab

        l_out =self.act(self.layer_in(xu_in.to(self.device, dtype=torch.float32)))
        if self.n_hidden>0:
            for layer in self.layers_middle:
                l_out=self.act(layer(l_out))        
        l_out=self.layer_out(l_out)

        # Get parameters from the output layer a,b,c,d,e dependent on number needed for given ansatz function
        a=l_out[:,:self.n_exp*self.xdim]
        b=l_out[:,self.n_exp*self.xdim:2*self.n_exp*self.xdim]
        c=l_out[:,2*self.n_exp*self.xdim:3*self.n_exp*self.xdim] 
        d=l_out[:,3*self.n_exp*self.xdim:4*self.n_exp*self.xdim]    

        # Calculation of DD-PINN g function
        g_out=a*(self.act_g(b*t_in+c)*torch.exp(-d*t_in)-self.act_g(c))
        

        # Calculate output of function g
        g_out_sum=g_out.view(t_in.shape[0], self.xdim, self.n_exp).sum(dim=2)
        out=x_in+g_out_sum
        out=out*self.sov[:self.xdim]

        # If ddt_flag==True, calculate and output derivative
        if ddt_flag:
            dgdt_out=a*(b*self.act_g_derivative(b*t_in+c)-d*self.act_g(b*t_in+c))*torch.exp(-d*t_in)
            dgdt_out_sum=dgdt_out.view(len(t_in), self.xdim, self.n_exp).sum(dim=2)
            dout_dt=dgdt_out_sum
            dout_dt=dout_dt*self.sov[:self.xdim]
            return out,dout_dt
        else:
            return out


def normalizeVar(var,m_var,b_var,flag):
    if flag:
        # Normalize
        return var*m_var+b_var
    else:
        # Renormalize/ rescale
        return (var-b_var)*1/m_var  

# Normalizes an array of values where the column's states are defined in 'props_names
def normalize_list(X,props_names,m_x,b_x,derivative_flag,no_offset_flag,normalize_flag):
    # derivative_flag
    #   True: The state to normalize or rescale is a derivative towards t?
    # no_offset_flag
    #   True: Normalization only based on m_x, e.g. when renormalizing ODE-losses

    if isinstance(X, torch.Tensor):
        if X.dim()==1:
            X=X.unsqueeze(0)
    
        X_scal=X.clone()
    elif isinstance(X, np.ndarray):
        if X.ndim==1:
            X=np.expand_dims(X,0)
        X_scal=X.copy()

    for i in range(len(props_names)):
        if derivative_flag:
            if normalize_flag: #not: experimental
                X_scal[:,i]=X[:,i]/m_x['dt']*m_x[props_names[i]]
            else:
                X_scal[:,i]=X[:,i]*m_x['dt']/m_x[props_names[i]]
                #X_scal[:,i]=X[:,i]/m_x['dt']*m_x[props_names[i]]
        else:
            if no_offset_flag:
                if normalize_flag: #not: experimental
                    X_scal[:,i]=X[:,i]*m_x[props_names[i]]
                else:
                    X_scal[:,i]=X[:,i]/m_x[props_names[i]]
            else:
                X_scal[:,i]=normalizeVar(X[:,i],m_x[props_names[i]],b_x[props_names[i]],normalize_flag)
    return X_scal

# Return a list of current losses
def give_loss_str(HP_dict,incl_total=True):
    loss_str=[]

    if HP_dict['ic_loss_flag']:
        loss_str.append("ic")
    if HP_dict['data_loss_flag']:
        loss_str.append("data")     

    for i in range(1,HP_dict['n_ODE']+1):
        loss_str.append('phys_ODE'+str(i))
        
    if incl_total: loss_str.append('total')
    return loss_str
