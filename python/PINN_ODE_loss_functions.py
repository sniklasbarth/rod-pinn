import torch
import math
import numpy as np
from math import pi as pi
# Physics of the Cosserat rod model for the PINN loss. The model with N = 7 nodes and
# 12*(N-1) = 72 states (implicit ODE and boundary values c_, strains for gravity eta_pls and
# distributed actuation loads u_p) is generated from the MATLAB export matlab/model/TRO_N_7 by
# python/matlab_to_torch.py into PINN_ODE_loss_functions_N7.py; actuation, gravity and the SO(3)
# helpers below are independent of N and written by hand.
from PINN_ODE_loss_functions_N7 import (implicit_ode_and_boundary_values_3D_N7_gravity,
                                        eta_pls_3D_N7, external_loads_3D_N7)

def check_sim_file(y, n_discr):
    """Assert that the simulated states fit the physics model (12*(N-1) states)."""
    expected = 12*(n_discr - 1)
    if y.shape[1] != expected:
        raise ValueError(f"simulation file has {y.shape[1]} states, the model with n_discr={n_discr} has {expected}")


def physics_model(n_discr=7):
    """Physics functions of the Cosserat rod:

        eta(x, w, param, bend_compliance)                         strains for gravity
        u_p(x, w, param, bend_compliance)                         distributed actuation loads
        c(x, xt, w, u_p, u_g, param, bend_compliance, tip_mass)   implicit ODE residual (n_x, B)

    x, xt: (n_x, B); w: actuation wrench (6, B); param: 26 rod parameters as 1-D tensor.
    """
    if n_discr != 7:
        raise NotImplementedError("only the model with n_discr = 7 is available")

    def eta(x, w, param, bend_compliance):
        return eta_pls_3D_N7(x, w, param.reshape(1, -1), bend_compliance)

    def u_p(x, w, param, bend_compliance):
        return external_loads_3D_N7(x, w, param.reshape(1, -1), bend_compliance)

    def c(x, xt, w, u_p_, u_g, param, bend_compliance, tip_mass):
        return implicit_ode_and_boundary_values_3D_N7_gravity(x, xt, w, u_p_, u_g, param.reshape(1, -1), bend_compliance, tip_mass)
    return eta, u_p, c

def actuation_model_pressure(in1, in2):
    in1=torch.transpose(in1,0,1)
    A_p = in2[0]
    p0 = in2[1]
    p1 = in1[0, :]
    p2 = in1[1, :]
    p3 = in1[2, :]
    r_p = in2[2]
    t2 = -p1
    t3 = -p2
    t4 = -p3
    t5 = torch.sqrt(torch.tensor(3.0))
    t6 = p0 + t2
    t7 = p0 + t3
    t8 = p0 + t4
    wrench = torch.vstack([torch.zeros(in1.shape[1]), 
                          torch.zeros(in1.shape[1]), 
                        #   -A_p * t6 - A_p * t7 - A_p * t8,  # from Henrik! do not fit Matlab function!
                        #   A_p * r_p * t5 * t7 * (-1.0 / 2.0) + (A_p * r_p * t5 * t8) / 2.0, 
                        #   A_p * r_p * t6 - (A_p * r_p * t7) / 2.0 - (A_p * r_p * t8) / 2.0, 
                          -A_p*t6-A_p*t7-A_p*t8,
                          A_p*r_p*t6*1.224646799147353e-16+(A_p*r_p*t5*t7)/2.0-(A_p*r_p*t5*t8)/2.0,
                          -A_p*r_p*t6+(A_p*r_p*t7)/2.0+(A_p*r_p*t8)/2.0,
                          torch.zeros(in1.shape[1])])

    wrench=torch.transpose(wrench,0,1)
    return wrench

def gravity(xi, param, N_nodes,tip_mass): 
    # GRAVITY is a manual implementation of discrete gravitational load
    # Parse gravity parameters
    # torch.set_default_dtype(torch.float64)
    g = param[:3].clone()


    L = param[3]  # Length of the rod
    M3 = param[4]  # 3rd diagonal entry of inertia density matrix
    M_tip = tip_mass  # Tip mass
    if M_tip.shape == torch.Size([]):
        M_tip = M_tip.unsqueeze(0)    

    # Strains of all gravity nodes as (n, 6, B); eta_pls returns (6, n, B)
    xi_mat = xi.permute(1, 0, 2)
    bsz = xi_mat.shape[2] # batch size

    # Linear map to add 3 zero components to ug (for moments)
    mapR3toR6 = torch.cat((torch.eye(3), torch.zeros(3, 3)), dim=0)

    # Linear map to remove the first 3 components of xi
    mapR6toR3 = torch.cat((torch.zeros(3, 3), torch.eye(3)), dim=1)

    # Reduce to relevant strains
    xi_mat = torch.matmul(mapR6toR3, xi_mat)
    N = xi_mat.shape[0]
    N_ = float((N - 1) / (N_nodes - 1))
    assert math.isclose(N_, round(N_), abs_tol=1e-4), 'N_ is not an integer. Assert that input N_nodes is correct.'
    N_ = round(N_) # remove numeric inaccuracies (e.g. 49/7 = 7.0000005 for float32 N_nodes), N_ is used for the tip mass below

    # Parameters for discrete integration of distributed gravitational load
    ds = L / (N - 1)
    R0 = torch.eye(3)
    R = R0.clone()

    # Assumption: nodes evenly distributed
    gload = -M3 * g
    u_g_mat_loc = torch.matmul(mapR3toR6, gload).unsqueeze(-1).expand(bsz,-1,-1) # expand with -1 means not changing the size of that dimension

    # Loop over nodes
    for i in range(N - 1):
        # Integrate orientations with linear interpolation of strains
        R = torch.matmul(R, SO3_expMap(xi_mat[i,:, :].T * ds)) # strains as (B, 3)  
        
        # Additional mass at tip
        if i >= N - (N_+2): # No -1 because of 0-indexing
            M_sum = -(M3 + M_tip)
            gload = M_sum.unsqueeze(1) * g
            # Concatenate local gravitational loads
            u_g_mat_loc = torch.cat((u_g_mat_loc, torch.matmul(mapR3toR6, torch.matmul(R.transpose(1,2), gload.unsqueeze(2)))), dim=2)
        else:
            u_g_mat_loc = torch.cat((u_g_mat_loc, torch.matmul(mapR3toR6, torch.matmul(R.transpose(1,2), gload).unsqueeze(-1))), dim=2)
        
    u_g = u_g_mat_loc.transpose(1,2).reshape(-1,N*6) 
    return u_g.transpose(0,1)

def SO3_expMap(omega):
    """
    Exponential map for the SO(3) group

    Inputs:
        omega - A 3x1 vector representing the element in the Lie algebra so(3)

    Outputs:
        R - A 3x3 rotation matrix in the SO(3) group
    """
    # Calculate the norm (theta)
    theta = torch.linalg.vector_norm(omega,dim=1)

    # Define the skew symmetric matrix
    omega_hat = SO3_skewSymmetric(omega)
    omega_hat_squared = torch.matmul(omega_hat,omega_hat)
    omega_hat_squared = omega_hat_squared.transpose(1,2)
    omega_hat_squared = omega_hat_squared.transpose(0,2)
    omega_hat = omega_hat.transpose(1,2)
    omega_hat = omega_hat.transpose(0,2)

    # Calculate the rotation matrix using the exponential map formula
    R = (torch.sin(theta) / theta) * omega_hat + ((1 - torch.cos(theta)) / (theta ** 2)) * omega_hat_squared
    R = R.transpose(0,1)
    R = R.transpose(0,2) + torch.eye(3)

    # If theta is 0, return identity matrix -> replace NaN values
    zeros_idx = (theta == 0).nonzero()
    R[zeros_idx,:,:] = torch.eye(3)

    return R

def SO3_skewSymmetric(omega_batch):
    """
    Returns the skew-symmetric matrices for a batch of 3x1 vectors omega_batch.

    Inputs:
        omega_batch - A batch of 3x1 vectors (Bx3 where B is the batch size)

    Outputs:
        omega_hat_batch - A batch of 3x3 skew-symmetric matrices (Bx3x3)
    """
    # Ensure omega_batch has shape (B, 3)
    assert omega_batch.shape[1] == 3, "Input must have shape (B, 3)"
    
    # Extract components of each vector in the batch
    omega_x = omega_batch[:, 0]
    omega_y = omega_batch[:, 1]
    omega_z = omega_batch[:, 2]
    
    # Create skew-symmetric matrices for each vector in the batch
    zero = torch.zeros_like(omega_x)
    
    omega_hat_batch = torch.stack([
        torch.stack([zero, -omega_z, omega_y], dim=1),
        torch.stack([omega_z, zero, -omega_x], dim=1),
        torch.stack([-omega_y, omega_x, zero], dim=1)
    ], dim=1)

    return omega_hat_batch
