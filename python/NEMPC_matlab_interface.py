import numpy as np
import torch
from torch_device import get_device
import pinn_io
import torch.nn as nn
import torch.nn.init as init
from NonlinearEMPC_pure_torch import NonlinearEMPC as NEMPC
import os
import json
from copy import deepcopy
import time
from scipy import signal

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

## Domain-decoupled PINN (DD-PINN)
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
        self.act_g=torch.sin
        self.act_g_derivative=torch.cos

        self.sov=state_one_vector.float() # S
        init.xavier_uniform_(self.layer_in.weight) # Xavier initilization of weights
        init.xavier_uniform_(self.layer_out.weight)

        layers=[]
        for i in range(HP_dict['n_hidden']):
            layers.append(nn.Linear(HP_dict['n_neurons'],HP_dict['n_neurons']))
            init.xavier_uniform_(layers[i].weight)
        self.layers_middle=nn.ModuleList(layers)
        self.n_hidden=HP_dict['n_hidden']
        
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

class DDNN_wrapper():
    def __init__(self,model,supp_vals_norm,t_norm):
        self.model=model
        self.supp_vals_norm=supp_vals_norm.clone()
        self.t_norm=t_norm.clone()

    def forward(self,x, u, dt):
        # start = time.time()
        # print(f"Supplementary Values: {self.supp_vals_norm}")
        bs = x.shape[0]
        x_NN = torch.cat((x, self.supp_vals_norm.expand(bs,1), u, self.t_norm.expand(bs,1)), dim=1)
        x_next=self.model.forward(x_NN)
        # time_passed = (time.time()- start)*1000
        # print(f"Forward pass took {time_passed} ms")
        return x_next
    
class NEMPC_wrapper():
    def __init__(self,pinn_folder,NEMPC_params,use_gpu=True):
        self.pinn_folder=pinn_folder
        self.use_gpu=use_gpu
        self.device = get_device(self.use_gpu)
        torch.set_default_dtype(torch.float32)

        hp_file = os.path.join(self.pinn_folder, 'hyperparameters.json')
        with open(hp_file, 'r') as input_file:
            self.HP_dict = json.load(input_file)

        m_x_file = os.path.join(self.pinn_folder, 'm_x.json')
        with open(m_x_file, 'r') as input_file:
            self.m_x = json.load(input_file)
        self.state_header = list(self.m_x.keys())

        b_x_file = os.path.join(self.pinn_folder, 'b_x.json')
        with open(b_x_file, 'r') as input_file:
            self.b_x = json.load(input_file)

        rod_params_file = os.path.join(self.pinn_folder, 'rod_params.json')
        with open(rod_params_file, 'r') as input_file:
            self.rod_params = json.load(input_file)


        if self.HP_dict['model_dims'] == 3:
            self.xdim = 12*(self.HP_dict['n_discr'] - 1)
            self.adim = 0
            self.sdim = 0
            if self.HP_dict['var_bend_comp'] == True:
                self.sdim = self.sdim + 1
            if self.HP_dict['var_tip_mass'] == True:
                self.sdim = self.sdim + 1
            self.udim = (self.HP_dict['u_degree']+1)*3
        else:
            raise ValueError('Model type not supported')

        self.state_one_vector=torch.ones(self.xdim+self.adim+self.sdim+self.udim+1)

        self.model = DDNN(self.xdim, self.adim, self.sdim, self.udim, self.state_one_vector, self.HP_dict,device=self.device)
        self.model.load_state_dict(pinn_io.read_weights(os.path.join(self.pinn_folder, 'weights.json')))
        self.model.to(self.device)

        # Controller parameters
        self.ctrl_dt = 1.0/self.HP_dict['f_sample']   # The time step of the controller.
        self.horizon = NEMPC_params['horizon']                        # The number of time steps in the controller horizon.
        self.numSims = NEMPC_params['numSims']                       # The number of simulations to run for the control test.
        self.parentQuota = NEMPC_params['parentQuota']                     # The number of parents to use in the genetic algorithm.
        self.strangerQuota = NEMPC_params['strangerQuota']             # The number of strangers to use in the genetic algorithm.
        self.numSegments = NEMPC_params['numSegments']                    # The number of knot points to use in the genetic algorithm.
        self.fix_first_knot = NEMPC_params['fix_first_knot']              # fix the first knot point to the current input.
        self.selection_method = "elitism"     # The selection method to use in the genetic algorithm.
        # tournament_size = 5                 # The tournament size to use in the genetic algorithm.
        self.crossover_method = "knot_point"     # The crossover method to use in the genetic algorithm.
        self.mutation_probability = NEMPC_params['mutation_probability']            # The mutation probability to use in the genetic algorithm.
        self.mutationNoise_init = NEMPC_params['mutationNoise_init']                  # The noise to add during mutation.
        self.mutation_noise_factor = NEMPC_params['mutation_noise_factor']
        self.mutation_noise_min = NEMPC_params['mutation_noise_min']
        self.mutation_noise_max = NEMPC_params['mutation_noise_max']

        if 'p_num' in NEMPC_params:
            self.use_pressure_model = True

            ### Scipy version
            self.p_num = np.array(NEMPC_params['p_num']).astype(np.float32).squeeze() # Pressure model numerator
            self.p_den = np.array(NEMPC_params['p_den']).astype(np.float32).squeeze() # Pressure model denominator

            def pressure_model(pressure, last_pressure):
                pressure = pressure.detach().cpu().numpy()
                last_pressure = last_pressure.detach().cpu().numpy()
                u_temp,_ = signal.lfilter(self.p_num, self.p_den, pressure, axis=1, zi=np.expand_dims(last_pressure,1))
                u_filt = torch.from_numpy(u_temp[:,1:,:]).float().to(self.device) # remove first element as it is the same as the previous input
                # u_filt = torch.from_numpy(u_temp).float().to(self.device) # don't remove to simulate the delay
                return u_filt

            ### Cupy version
            # self.p_num = cupy.array(NEMPC_params['p_num']).astype(cupy.float32).squeeze() # Pressure model numerator
            # self.p_den = cupy.array(NEMPC_params['p_den']).astype(cupy.float32).squeeze() # Pressure model denominator
            # def pressure_model(pressure, last_pressure):
            #     pressure_cupy = cupy.from_dlpack(pressure)
            #     last_pressure_cupy = cupy.from_dlpack(last_pressure)
            #     pressure_filt_cupy,_ = signal.lfilter(self.p_num, self.p_den, pressure_cupy, axis=1, zi=cupy.expand_dims(last_pressure_cupy,1))
            #     pressure_filt = torch.from_dlpack(pressure_filt_cupy)
            #     return pressure_filt
        else:
            self.use_pressure_model = False



        # X0_norm = normalize_list(torch.zeros(xdim),state_header[:xdim],m_x,b_x,False,False,True)
        self.supp_vals_norm = normalizeVar(self.rod_params['C4'],self.m_x['C45'], self.b_x['C45'], True)
        self.t_norm = normalizeVar(self.ctrl_dt, self.m_x['dt'], self.b_x['dt'], True)

        self.model_wrapped = DDNN_wrapper(self.model,torch.tensor(self.supp_vals_norm,device=self.device,dtype=torch.float32),torch.tensor(self.t_norm,device=self.device,dtype=torch.float32))

        umin = normalizeVar(NEMPC_params['u_min_bar']*10**5, self.m_x['up1_a'], self.b_x['up1_a'], True)
        umax = normalizeVar(NEMPC_params['u_max_bar']*10**5, self.m_x['up1_a'], self.b_x['up1_a'], True)
        self.umin = torch.tensor([umin,umin,umin], dtype=torch.float32).to(self.device)
        self.umax = torch.tensor([umax,umax,umax], dtype=torch.float32).to(self.device)

        self.R = NEMPC_params['R_factor'] * np.eye(self.udim)
        R = torch.from_numpy(self.R).float().to(self.device)

        self.Q = np.diag(NEMPC_params['Q']) 
        Q = torch.from_numpy(self.Q).float().to(self.device)

        def CostFunc(x: torch.Tensor, 
                    u: torch.Tensor, 
                    xgoal: torch.Tensor, 
                    ugoal: torch.Tensor, 
                    prev_u: torch.Tensor, 
                    final_timestep: bool = False) -> torch.Tensor:

            Qx = torch.mm((x - xgoal) ** 2.0, Q)
            Qu = torch.mm((u - prev_u) ** 2.0, R)
            cost = torch.sum(torch.cat((Qx,Qu),dim=1), axis=1)
            return cost


        
        self.controller = NEMPC(
            forward_sim_func=self.model_wrapped.forward,
            cost_function=CostFunc,
            numStates=self.xdim,
            numInputs=self.udim,
            umin=self.umin,
            umax=self.umax,
            horizon=self.horizon,
            dt=self.ctrl_dt,
            numSegments=self.numSegments,
            device=self.device,
            selection_method=self.selection_method,
            # tournament_size=tournament_size,
            crossover_method=self.crossover_method,
            mutation_probability=self.mutation_probability,
            numSims=self.numSims,
            numParents=int(self.numSims*self.parentQuota),
            numStrangers=int(self.numSims*self.strangerQuota),
            pressure_model=pressure_model if self.use_pressure_model else None,
            fix_first_knot = self.fix_first_knot
        )
        


    def step(self,x,u,supp_vals_norm,xgoal,u_meas:np.ndarray=None,xEE_meas:np.ndarray=None):
        self.model_wrapped.supp_vals_norm=torch.from_numpy(supp_vals_norm).float().to(self.device)
        mutationNoise = np.clip(np.sum(np.abs(x - xgoal[0,:]))*self.mutation_noise_factor,self.mutation_noise_min,self.mutation_noise_max) 
        print(f"Mutation Noise: {mutationNoise}")

        with torch.no_grad():
            u_nempc, path = self.controller.solve_for_next_u(
                torch.from_numpy(x).float().to(self.device),
                torch.from_numpy(xgoal).float().to(self.device),
                ulast=torch.from_numpy(u.reshape((1, self.controller.numInputs))).float().to(self.device),
                ugoal=torch.zeros((1, self.controller.numInputs)),
                mutation_noise=torch.tensor(mutationNoise, dtype=torch.float32).to(self.device),
                u_meas=torch.from_numpy(u_meas).float().to(self.device) if u_meas is not None else None
            )

        return u_nempc.detach().cpu().numpy()