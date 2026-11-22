
# import numpy as np
import torch
from typing import Callable
# import time

class NonlinearEMPC:
    def __init__(
        self,
        forward_sim_func: Callable,
        cost_function: Callable,
        numStates: int,
        numInputs: int,
        umin: float | torch.Tensor,
        umax: float | torch.Tensor,
        horizon: int,
        dt: float,
        numSims: int,
        numSegments: int,
        device: torch.device,
        **kwargs,
    ):
        """Nonlinear Evolutionary Model Predictive Control Class

            For algorithmic details, see this paper:

            Hyatt, Phillip, and Marc D. Killpack.
            “Real-Time Nonlinear Model Predictive Control of Robots Using a Graphics Processing Unit.”
            IEEE Robotics and Automation Letters 5, no. 2 (April 2020): 1468–75.
            https://doi.org/10.1109/LRA.2020.2965393.

        :param forward_sim_func: Function that takes in state, input, and dt and returns the next state
        :param cost_function: Function that takes in state, input, goal state, and goal input and returns the cost
        :param numStates: Number of states
        :param numInputs: Number of inputs
        :param umin: Minimum input value, size (1, numInputs)
        :param umax: Maximum input value, size (1, numInputs)
        :param horizon: Number of time steps in the controller horizon
        :param dt: Time step of the controller
        :param numSims: Number of simulations to run for the control test
        :param numSegment: Number of segments to use in the genetic algorithm
        :param useGPU: Whether or not to use the GPU
        :param kwargs: Optional arguments
            :param kwargs["selection_method"]: str: The selection method to use in the genetic algorithm
            :param kwargs["tournament_size"]: int: The tournament size to use in the genetic algorithm
            :param kwargs["crossover_method"]: str: The crossover method to use in the genetic algorithm
            :param kwargs["mutation_probability"]: float: The mutation probability to use in the genetic algorithm
            :param kwargs["numParents"]: int: The number of parents to use in the genetic algorithm
            :param kwargs["numStrangers"]: int: The number of strangers to use in the genetic algorithm
        """
        torch.set_grad_enabled(False)
        self.model = forward_sim_func
        self.cost_function = cost_function
        self.numStates = numStates
        self.numInputs = numInputs
        self.umin = umin
        self.umax = umax
        self.uRange = umax - umin
        self.horizon = horizon
        self.numSims = numSims
        self.dt = dt
        self.numSegments = numSegments

        supported_selection_methods = ["elitism", "tournament"]
        self.selection_method = kwargs.get("selection_method", "elitism")
        if self.selection_method not in supported_selection_methods:
            raise ValueError(f"{self.selection_method} isn't supported.")
        self.tournament_size = kwargs.get("tournament_size", 5)
        self.crossover_method = kwargs.get("crossover_method", "knot_point")
        self.mutation_probability = kwargs.get("mutation_probability", 0.1)
        self.numParents = kwargs.get("numParents", 10)
        self.numStrangers = kwargs.get("numStrangers", 0)
        self.pressure_model = kwargs.get("pressure_model", None)
        if self.pressure_model is None:
            self.use_pressure_model = False
        else:
            self.use_pressure_model = True
        self.fix_first_knot = kwargs.get("fix_first_knot", False)

        assert self.mutation_probability >= 0.0 and self.mutation_probability <= 1.0, "Mutation probability is out of bounds [0,1]"
        assert self.numParents + self.numStrangers < self.numSims, "Population split badly. Must have numParents + numStrangers < numSims."

        self.warmStart = False
        
        self.device = device
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        torch.set_default_device(self.device)
        torch.set_default_dtype(torch.float32)

        self.rng = torch.Generator(device=self.device)

        self.U = torch.zeros([self.numSims, self.numSegments, self.numInputs]).to(self.device)
        self.costs = torch.zeros([self.numSims]).to(self.device)
        self.costs_zero = torch.zeros([self.numSims,]).to(self.device)
        self.X = torch.zeros([self.numSims, self.horizon, self.numStates]).to(self.device)
        self.X_c_zero = torch.zeros([self.numSims, self.horizon, self.numStates]).to(self.device)

    def get_random_commands(self,method: str, numSims: int, **kwargs) -> torch.tensor:
        """
        Get random input trajectories
        :param numSims: Number of random input trajectories to generate
        :return: Random input trajectories, size (numSims, knotPoints, numInputs)"""
        if method == "uniform":
            return (torch.rand((numSims, self.numSegments, self.numInputs),self.rng) * self.uRange + self.umin)
        elif method == "normal":
            ulast = kwargs.get("ulast", torch.vstack([self.umin, self.umax]).mean(dim=0))
            u_rand = torch.zeros([numSims, self.numSegments, self.numInputs]).to(self.device)
            # print(ulast.shape[0])
            for i in range(ulast.shape[0]): # number of segments
                distr = torch.distributions.multivariate_normal.MultivariateNormal(ulast[i,:], covariance_matrix=0.3*torch.eye(3).to(self.device))
                u_rand[:,i,:] = distr.sample(sample_shape=torch.tensor([numSims]).to(self.device))
            u_rand = torch.abs(u_rand-self.umin) + self.umin # clip with reflection
            u_rand = -torch.abs(u_rand-self.umax) + self.umax # clip with reflection
            return torch.clip(u_rand, self.umin, self.umax)
        else:
            raise ValueError(f"{method} isn't a supported method.")

    def select_parents(self) -> torch.tensor:
        """
        Select parents from the population
        :return: Parents, size (numParents, numSegments, numInputs)
        """
        U_parents = torch.zeros([self.numParents, self.numSegments, self.numInputs])
        parentCount = 0
        if self.selection_method == "tournament":
            while parentCount < self.numParents:
                group_idxs = torch.randint(0, self.numSims - 1, self.tournament_size, generator=self.rng)
                tourney_winner_idx = group_idxs[torch.argmin(self.costs[group_idxs])]
                U_parents[parentCount] = self.U[tourney_winner_idx]
                parentCount += 1
        elif self.selection_method == "stochastic_acceptance":
            minCost = torch.min(self.costs)
            while parentCount < self.numParents:
                candidate = torch.randint(0,self.numSims,(1), generator=self.rng)
                cost = self.costs[candidate]
                acceptProbability = minCost / cost
                if torch.rand((1),generator=self.rng) < acceptProbability:
                    U_parents[parentCount] = self.U[candidate]
                    parentCount += 1
        else:   # elitism - just choose best numParents members of the population
            indices = self.costs.argsort().flatten()
            U_parents = self.U[indices[:self.numParents]]
        return U_parents

    def mate_parents(self, parents: torch.tensor) -> torch.tensor:
        """
        Mate parents to create children
        :param parents: Parents to mate, size (numParents, knotPoints, numInputs)
        :return: Children, size (numChildren, knotPoints, numInputs)
        """
        self.U[:self.numParents] = parents
        # self.U[self.numParents:self.numParents+self.numStrangers] = self.get_random_commands("uniform",self.numStrangers)
        self.U[self.numParents:self.numParents+self.numStrangers] = self.get_random_commands("normal",self.numStrangers,ulast=self.bestU)

        numChildren = self.numSims - (self.numParents + self.numStrangers)
        if self.crossover_method == "knot_point":
            mates = torch.randint(0, self.numParents, (2, numChildren),device=self.device)
            parents0 = parents[mates[0]]
            parents1 = parents[mates[1]]

            inheritanceMask = torch.randint(2, size=parents0.shape,device=self.device)
            children = parents0 * inheritanceMask + parents1 * (1 - inheritanceMask)
        else:
            raise ValueError(f"{self.crossover_method} isn't a supported crossover method.")
        
        return children
        
    def mutate_children(self, children: torch.tensor, mutation_noise: float) -> None:
        """
        Mutate the children with Gaussian noise
        :param children: Children to mutate, size (numChildren, knotPoints, numInputs)
        :param mutation_noise: Maximum amount that a child will be mutated (+- mutation_noise)
        :return: None
        """
        numChildren = self.numSims - (self.numParents + self.numStrangers)
        random = torch.rand(size=[numChildren],generator=self.rng,device=self.device)
        mutationMask = random > (1 - self.mutation_probability)
        numMutated = torch.sum(mutationMask)

        mutation = torch.randn(size=[numMutated, self.numSegments, self.numInputs], generator=self.rng,device=self.device) * mutation_noise
        children[mutationMask] += mutation

        children = torch.clip(children, self.umin, self.umax)
        return children

    def get_us_from_U(self) -> torch.tensor:
        """
        Get the inputs for the trajectory based on knotPoints
        :return: Inputs for the given time step, size (numSims, horizon, numInputs)
        """

        if self.numSegments == 0:
            ui = self.U.expand((-1,self.horizon,-1))
        else:
            if self.fix_first_knot:
                segmentLength = torch.tensor(float(self.horizon) / (self.numSegments)).to(self.device)
                i = torch.arange(1.0,self.horizon+1,device=self.device)
                knot = (i / segmentLength).int()
                U_with_last = torch.concatenate((self.ulast.expand((self.numSims,-1)).unsqueeze(1), self.U),dim=1)
                Ulow = U_with_last[:, knot]
                knot[-1] -= 1
                Uhigh = U_with_last[:, knot + 1]
            else:
                segmentLength = torch.tensor(float(self.horizon) / (self.numSegments-1)).to(self.device)
                i = torch.arange(1.0,self.horizon+1,device=self.device)
                knot = (i / segmentLength).int()
                Ulow = self.U[:, knot,:]
                knot[-1] -= 1
                Uhigh = self.U[:, knot + 1,:]
            ratio = (i % segmentLength) / segmentLength
            ui = (1-ratio.reshape(1,self.horizon,1))*Ulow + ratio.reshape(1,self.horizon,1)*Uhigh # linear interpolation
        ui = torch.clamp(ui, self.umin, self.umax)
        return ui
    
    # def get_costs_cpu(self, x0:np.ndarray, xgoal:np.ndarray, ugoal:np.ndarray, uprev:np.ndarray=None) -> None:
    #     """
    #     Forward simulate the population through the horizon and assign each member a cost using the cpu
    #     :param x0: Initial state of the system, size (1, numStates)
    #     :param xgoal: Goal state of the system, size (1, numStates)
    #     :param ugoal: Goal input of the system, size (1, numInputs)
    #     :param uprev: Previous input of the system, size (1, numInputs)
    #     :return: None

    #     Updates self.costs and self.X directly
    #     """
    #     self.costs = np.zeros([self.numSims])
    #     self.X[:,0] = x0
    #     U = self.get_us_from_U()

    #     for i in range(0, self.horizon - 1):
    #         self.X[:, i+1] = self.model(self.X[:, i], U[:,i], self.dt)
    #         self.costs += self.cost_function(self.X[:, i], U[:,i], xgoal, ugoal)

    #     FinalCosts = self.cost_function(self.X[:, self.horizon - 1], U[:,i-1], xgoal, ugoal, final_timestep=True)
    #     self.costs += FinalCosts

    def get_costs_gpu(self, x0:torch.tensor, xgoal:torch.tensor, ugoal:torch.tensor, u_prev:torch.tensor=None,u_meas:torch.tensor=None) -> None:
        """
        Forward simulate the population through the horizon and assign each member a cost.
        :param x0: Initial state of the system, size (1, numStates)
        :param xgoal: Goal state of the system, size (1, numStates)
        :param ugoal: Goal input of the system, size (1, numInputs)
        :param uprev: Previous input of the system, size (1, numInputs)
        :return: None

        Updates self.costs and self.X directly
        """
        # xgoal_c = torch.from_numpy(xgoal).float().to(self.device)
        # ugoal_c = torch.from_numpy(ugoal).float().to(self.device)
        # uprev = torch.from_numpy(uprev).float().to(self.device)
        self.U_c = self.get_us_from_U()
        costs = self.costs_zero.clone()

        u_prev_rep = u_prev.expand((self.numSims,-1))
        X_c = self.X_c_zero.clone()
        # x0_c = torch.tensor(x0).float().to(self.device)
        X_c[:, 0] = x0.expand((self.numSims,-1))

        if self.use_pressure_model:
            # Calculate with filtered pressures but still return unfiltered pressures as setpoint for pressure controller
            # u_temp = np.zeros(self.U_c.shape,dtype=np.float32)
            # for i in range(0, self.U_c.shape[2]):
            #     u_temp[:,:,i],_ = self.pressure_model(self.U_c[:,:,i].detach().cpu().numpy(),u_prev_rep[:,i].detach().cpu().numpy())
            u_meas_rep = u_meas.expand((self.numSims,-1))
            u_filt = self.pressure_model(self.U_c,u_meas_rep)

            for i in range(0, self.horizon - 1):
                X_c[:, i+1] = torch.as_tensor(self.model(X_c[:, i], u_filt[:,i], self.dt)).to(self.device) # predict with filtered pressures
                if i == 0:
                    costs += self.cost_function(X_c[:,i], self.U_c[:,i], xgoal[i,:].unsqueeze(0), ugoal, prev_u=u_prev_rep) # use unfiltered pressures for cost calculation
                else:
                    costs += self.cost_function(X_c[:,i], self.U_c[:,i], xgoal[i,:].unsqueeze(0), ugoal, prev_u=self.U_c[:,i-1])
            costs += self.cost_function(X_c[:, self.horizon-1], self.U_c[:,self.horizon-1], xgoal[self.horizon-1,:].unsqueeze(0), ugoal, prev_u=self.U_c[:,self.horizon - 2], final_timestep=True)

        else:
            for i in range(0, self.horizon - 1):
                X_c[:, i+1] = torch.as_tensor(self.model(X_c[:, i], self.U_c[:,i], self.dt)).to(self.device)
                if i == 0:
                    costs += self.cost_function(X_c[:,i], self.U_c[:,i], xgoal[i,:].unsqueeze(0), ugoal, prev_u=u_prev_rep)
                else:
                    costs += self.cost_function(X_c[:,i], self.U_c[:,i], xgoal[i,:].unsqueeze(0), ugoal, prev_u=self.U_c[:,i-1])
            costs += self.cost_function(X_c[:, self.horizon-1], self.U_c[:,self.horizon-1], xgoal[self.horizon-1,:].unsqueeze(0), ugoal, prev_u=self.U_c[:,self.horizon - 2], final_timestep=True)
        

        self.X = X_c
        self.costs = costs
        # time_passed = (time.time()- start_time)*1000.0
        # print(f"Time for predictions: {time_passed:.2f} ms")

    def get_best_inputs(self) -> torch.tensor:
        """
        Get the next input to be applied (the one corresponding to the lowest cost)
        :return: Best inputs, size (1, numInputs)
        """
        cost_not_nan = torch.nan_to_num(self.costs,nan = 10e14)
        best_trajectory_idx = torch.argmin(cost_not_nan)
        self.bestU = self.U[best_trajectory_idx]
        return self.U_c[best_trajectory_idx,:]
    
    def get_planned_path(self) -> torch.tensor:
        """
        Get the best trajectory that will be returned by nempc
        :return: Best trajectory, size (horizon, numStates)
        """
        cost_not_nan = torch.nan_to_num(self.costs,nan = 10e14)
        best_trajectory_idx = torch.argmin(cost_not_nan)
        return self.X[best_trajectory_idx]

    def solve_for_next_u(self, x0:torch.tensor, xgoal:torch.tensor, ugoal:torch.tensor, ulast:torch.tensor, mutation_noise:float, u_meas:torch.tensor=None) -> tuple[torch.tensor, torch.tensor]:
        """
        Solve for the next input to apply to the system
        :param x0: Initial state of the system, size (1, numStates)
        :param xgoal: Goal state of the system, size (1 or horizon, numStates)
        :param ugoal: Goal input of the system, size (1 or horizon, numInputs)
        :param ulast: Previous input of the system, size (1, numInputs)
        :param mutation_noise: Mutation noise to use for the genetic algorithm
        :return: Tuple[np.ndarray, np.ndarray]:
            u: Input to apply to the system, size (1, numInputs)
            path: Best trajectory that will be returned by nempc, size (horizon, numStates)
        """
        self.ulast = ulast
        if not self.warmStart:
            self.U = self.get_random_commands(method="normal",numSims=self.numSims,ulast=ulast.expand((self.numSegments,-1)))
            self.warmStart = True
        else:
            U_parents = self.select_parents()
            children = self.mate_parents(U_parents) #
            mutated_children = self.mutate_children(children, mutation_noise)
            self.U[self.numParents + self.numStrangers:] = mutated_children
        
        # if self.device == "cpu":
        #     self.get_costs_cpu(x0, xgoal, ugoal, uprev=ulast)
        # else:
        if self.use_pressure_model:
            self.get_costs_gpu(x0, xgoal, ugoal, u_prev=ulast,u_meas=u_meas)
        else:
            self.get_costs_gpu(x0, xgoal, ugoal, u_prev=ulast)

        u = self.get_best_inputs()
        path = self.get_planned_path()
        return u, path