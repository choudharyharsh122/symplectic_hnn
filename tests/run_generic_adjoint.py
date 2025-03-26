import os
os.system("ml tqdm/4.66.2-GCCcore-13.2.0")
from tqdm import tqdm
os.system("ml PyTorch/2.2.1-foss-2023b-CUDA-12.4.0")
import torch
import torch.nn as nn   
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, TensorDataset
import math
import optuna
import csv
import logging
import sys
import pandas as pd
import time
import argparse
import distutils.util

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])


    
class HamiltonianNN(nn.Module):

    def __init__(self, model_specs):
        super(HamiltonianNN, self).__init__()

        # Create a list of linear layers based on layer_sizes
        layer_sizes = model_specs[0]
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.RANDOM_SEED = 0
        for i in range(len(layer_sizes) - 2):  # All layers except the last one
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=True))

            self.dropout_layers.append(nn.Dropout(p=0.2))
        
        # Last layer without bias
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=False))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.tanh(x)
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)
        x = self.layers[-1](x)
        return x
    
    def _apply_xavier_init(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    

"""
Computes the time derivative of the system state using Hamiltonian-based dynamics.

Parameters:
-----------
y_tensor : torch.Tensor, shape (batch_size, 2d)  
    Current state of the system, where:  
    - The first `d` elements represent the position \( q \).  
    - The last `d` elements represent the momentum \( p \).
args : tuple  
    Contains additional arguments required for computation.  
    - `args[0]` : Neural network model that approximates the Hamiltonian \( H(q, p, \theta) \).
kargs : tuple  
    Contains additional keyword arguments.  
    - `kargs[0]` : Index or step counter (not explicitly used in this function).

Returns:
--------
torch.Tensor, shape (batch_size, 2d)  
    The computed time derivative \([dq/dt, dp/dt]\), where:
    - \( dq/dt = \partial H / \partial p \)
    - \( dp/dt = -\partial H / \partial q \)
"""
def forward_ode(y_tensor, args, kargs):

    model = args[0]
    i = kargs[0]
    br_i = y_tensor.shape[1]//2
        
    with torch.enable_grad():

        y = y_tensor.clone().detach().requires_grad_(True)

        h = model(y)

    
        grad_h = torch.autograd.grad(outputs=h.sum(), inputs=y, create_graph=True, retain_graph=True, allow_unused=True)[0]

        #print("grad h: ", grad_h)
        dq_dt = grad_h[:, br_i:2*br_i]
        dp_dt = -grad_h[:, 0:br_i]

    return torch.cat((dq_dt, dp_dt), dim=-1)

"""
    Computes the time evolution of the adjoint variable in a Hamiltonian system.

    Parameters:
    -----------
    lam : torch.Tensor, shape (batch_size, 2d)
        Adjoint state variable \( \lambda(t) \).

    args : tuple
        Contains:
        - `model`: Neural network representing \( H(y) \).
        - `y_values`: Tensor of system state over time.
        - `y_gt_values`: Ground truth states (not used in the function).

    kargs : tuple
        Contains:
        - `i`: Current time step index.

    Returns:
    --------
    torch.Tensor, shape (batch_size, 2d)
        Time derivative \( \dot{\lambda} \) computed as \( \dot{\lambda} = -J_H \lambda \), where \( J_H \) is the Hessian of \( H(y) \).
    """
def adjoint_ode(lam, args, kargs):

    (model, y_values) = args
    batch_size = y_values.shape[0]
    (i,) = kargs
    br_i = y_values.shape[2]//2

    y_tensor = y_values[:,i,:].clone().detach().requires_grad_(True)
    
    h = model(y_tensor)

    # Compute first-order derivatives ∇H = [∂H/∂q, ∂H/∂p]
    grad_h = torch.autograd.grad(outputs=h.sum(), inputs=y_tensor, create_graph=True, retain_graph=True)[0]  # Shape: [batch_size, 2n]

    # Compute second-order derivatives
    J_H = torch.zeros(batch_size, 2*br_i, 2*br_i)  # Initialize Jacobian matrix

    for i in range(2 * br_i):
        if i<br_i:
            grad_i = torch.autograd.grad(outputs=grad_h[:, br_i+i], inputs=y_tensor, grad_outputs=torch.ones_like(grad_h[:, br_i+i]), create_graph=True, retain_graph=True)[0]
        else:
            grad_i = torch.autograd.grad(outputs=-grad_h[:, i-br_i], inputs=y_tensor, grad_outputs=torch.ones_like(-grad_h[:, i-br_i]), create_graph=True, retain_graph=True)[0]
        J_H[:, i, :] = grad_i  # Assign row-wise
    
    lam_tensor = lam.clone().detach().unsqueeze(2)


    lam_dot = - (torch.bmm(J_H, lam_tensor)).squeeze()

    return lam_dot


def reshape_gradients(flattened_gradients, original_shapes):
    reshaped_gradients = []
    start = 0
    for shape in original_shapes:
        size = torch.prod(torch.tensor(shape)).item()  # Calculate the number of elements in this shape
        end = start + size
        reshaped_gradients.append(flattened_gradients[start:end].reshape(shape))
        start = end
    return reshaped_gradients


"""
    Performs a single step of the explicit second-order Runge-Kutta method (midpoint method)
    for solving Hamiltonian dynamics.

    Parameters:
    -----------
    y : torch.Tensor, shape (batch_size, 2d)
        The current state of the system:
        - `q = y[:, 0:d]` (generalized coordinates)
        - `p = y[:, d:2d]` (generalized momenta)
    dt : float
        The integration time step \( h \).
    dynamics : function
        Function computing the system’s dynamics \( f(y) \).
    args : tuple
        Additional arguments for `dynamics`.
    kargs : tuple
        Contains:
        - `i`: Current time step index.

    Returns:
    --------
    torch.Tensor, shape (batch_size, 2d)
        The updated state \( y_{n+1} \) after an RK2 step.
    """
def rk2_step(y, dt, dynamics, args, kargs):
    h = dt
    i = kargs[0]
    br_i = y.shape[1]//2
    q, p = y[:, 0:br_i], y[:, br_i:2*br_i]


    dy1 = dynamics(y, args, kargs)
    q1 = q + 0.5 * dy1[:, 0:br_i] * h
    p1 = p + 0.5 * dy1[:, br_i:2*br_i] * h

    y1 = torch.cat((q1, p1), dim=-1)
    dy2 = dynamics(y1, args, kargs)

    q_new = q + dy2[:, 0:br_i] * h
    p_new = p + dy2[:, br_i:2*br_i] * h
    return torch.cat((q_new, p_new), dim=-1)


"""
    Performs a single step of the Implicit Midpoint method for Hamiltonian systems.

    Parameters:
    -----------
    y : torch.Tensor, shape (N, 2d)
        Current system state:
        - `q = y[:, 0:d]` (generalized coordinates)
        - `p = y[:, d:2d]` (generalized momenta)
    dt : float
        The integration time step \( h \).
    dynamics : function
        Function computing the system’s dynamics \( f(y) \).
    iterations : int
        Number of iterations for solving the implicit equations.
    y_init : torch.Tensor, shape (N, 2d)
        Initial condition used to compute the implicit step.
    args : tuple
        Additional arguments for `dynamics`.
    kargs : tuple
        Additional keyword arguments.

    Returns:
    --------
    torch.Tensor, shape (N, 2d)
        Updated state \( y_{n+1} \) after an implicit midpoint step.
    """
def im_step(y, dt, dynamics, iterations, y_init, args, kargs):
    h = dt
    br_i = y.shape[1] // 2
    q, p = y[:, 0:br_i], y[:, br_i:2 * br_i]
    
    y_init_concat = torch.cat((y_init[:, 0:br_i], y_init[:, br_i:2*br_i]), dim=-1)  # Shape [batch, 2]
    f_init = dynamics(y_init_concat, args, kargs)  # Compute dynamics at initial point
    
    q_new = q + 0.5 * h * f_init[:, 0:br_i]  # Shape [batch, 1]
    p_new = p + 0.5 * h * f_init[:, br_i:2*br_i]  # Shape [batch, 1]

    for _ in range(iterations):
        mid_q = 0.5 * (q + q_new)  # Shape [batch, q_shape]
        mid_p = 0.5 * (p + p_new)  # Shape [batch, p_shape]
        
        mid_concat = torch.cat((mid_q, mid_p), dim=-1)  # Ensure [batch, 2*q_shape]
        f_mid = dynamics(mid_concat, args, kargs)  # Compute dynamics at midpoint
        
        q_new = q + h * f_mid[:, 0:br_i]  # Shape [batch, q_shape]
        p_new = p + h * f_mid[:, br_i:2*br_i]  # Shape [batch, p_shape]

    return torch.cat((q_new, p_new), dim=-1)  # Final shape [batch, 2*q_shape]


"""
    Performs a single step of the Störmer-Verlet (2nd order semi-implicit symplectic) integrator.

    Parameters:
    -----------
    y : torch.Tensor, shape (N, 2d)
        The current state of the system, where:
        - `q = y[:, 0:d]` represents generalized coordinates.
        - `p = y[:, d:2d]` represents generalized momenta.
    dt : float
        The integration time step \( h \).
    dynamics : function
        The function that computes the time derivative of the state \( \dot{y} = f(y) \).
    iterations : int
        The number of fixed-point iterations for implicit updates.
    y_init : torch.Tensor, shape (N, 2d)
        The initial condition of the trajectory, used for implicit updates.
    args : tuple
        Additional arguments passed to `dynamics`.
    kargs : tuple
        Additional keyword arguments (assumed to contain at least one index `i`).

    Returns:
    --------
    torch.Tensor, shape (N, 2d)
        The updated state \( y_{n+1} \) after a full Störmer-Verlet step.

    """
def sv_step(y, dt, dynamics, iterations, y_init, args, kargs):
    h = dt
    br_i = y.shape[1] // 2
    q, p = y[:, 0:br_i], y[:, br_i:2 * br_i]
    i = kargs[0]

    p_half = p + 0.5 * h * dynamics(torch.cat((q, y_init[:, 0:br_i]), dim=-1), args, kargs)[:, br_i:2 * br_i]
    for _ in range(iterations):
        p_half = p + 0.5 * h * dynamics(torch.cat((q, p_half), dim=-1), args, kargs)[:, br_i:2 * br_i]

    q_half = q + 0.5 * h * dynamics(torch.cat((y_init[:, 0:br_i], p_half), dim=-1), args, kargs)[:, 0:br_i]
    for _ in range(iterations):
        q_half = q + 0.5 * h * dynamics(torch.cat((q_half, p), dim=-1), args, kargs)[:, 0:br_i]

    q_new = q + h * dynamics(torch.cat((q_half, p_half), dim=-1), args, kargs)[:, 0:br_i]
    p_new = p_half + 0.5 * h * dynamics(torch.cat((q_new, p_half), dim=-1), args, kargs)[:, br_i:2 * br_i]

    return torch.cat((q_new, p_new), dim=-1)


"""
Logic for Forward ODE Solver

Parameters:
-----------
dynamics : function  
    Function that computes the system dynamics \( f(y) \).
pred : bool  
    If `True`, performs an explicit step at the beginning using the solver.  
    If `False`, uses the noisy ground truth trajectory (`y_data`) for initialization.
y_data : torch.Tensor, shape (batch_size, time_steps, 2d)  
    Noisy ground truth trajectory, serving as initial values for the solver.
solver : str  
    Specifies the integration method to be used:
    - `"im"` : Implicit midpoint solver.
    - `"sv"` : Störmer-Verlet (semi-implicit symplectic) solver.
dt : float  
    Time step size \( h \) for numerical integration.
shooting_segment_length : int  
    Number of time steps in each shooting segment.
number_of_shooting_segments : int  
    Number of shooting segments (short trajectories) in the integration process.
iters : int  
    Number of Newton or fixed-point iterations per solver step.

Returns:
--------
torch.Tensor  
    The computed trajectory over all shooting segments.
"""

def solve_ivp_custom_ms_forward(dynamics, pred, y_data, solver, dt, shooting_segment_length, number_of_shooting_segments, args, iters):
    #t0, t1 = t_span
    shooting_nodes = [y_data[:, i * shooting_segment_length, :] for i in range(number_of_shooting_segments)]
    segments = []
    cont_errors = []
    
    for seg in range(number_of_shooting_segments):
        y_seg = shooting_nodes[seg]
        seg_states = [y_seg]
        
        for j in range(1, shooting_segment_length):
            step_index = seg * shooting_segment_length + j
            if pred:
                y_ = rk2_step(y_seg, dt, dynamics, args, kargs=(step_index,))
            else:
                y_ = y_data[:, step_index, :]
            
            if solver=="im":
                y_next = im_step(y_seg, dt, dynamics, iters, y_, args, kargs=(step_index,))
            elif solver=="sv":
                y_next = sv_step(y_seg, dt, dynamics, iters, y_, args, kargs=(step_index,))

            seg_states.append(y_next)
            y_seg = y_next
        
        seg_states = torch.stack(seg_states, dim=1)
        segments.append(seg_states)
        
        #cont_error = y_seg - shooting_nodes[seg + 1]
        #cont_errors.append(cont_error)
    
    full_traj = segments[0]
    for seg in segments[1:]:
        full_traj = torch.cat((full_traj, seg), dim=1)
    
    #print("traj shape:", full_traj.shape)
    return full_traj, cont_errors


"""
    Computes an integral over time using the trapezoidal rule to track parameter gradients.

    Parameters:
    -----------
    model : torch.nn.Module
        Neural network modeling the Hamiltonian function \( H(y) \).

    y_t : torch.Tensor, shape (batch_size, 2d)
        State variable at time t.

    lambda_t : torch.Tensor, shape (batch_size, 2d)
        Adjoint variable at time t.

    Returns:
    --------
    torch.Tensor, shape (num_params,)
        Integral of the parameter-dependent term over time.
    """
def calculate_grad(model, y_t, lambda_t, batch_size):

    integral_values = []
    batch_size = lambda_t.shape[0]
    br_i = y_t.shape[1]//2

    y_tensor = y_t.clone().detach().requires_grad_(True).to(y_t.device)
    
    # Perform forward pass
    h = model(y_tensor)
    
    # Compute gradients of model output w.r.t y_tensor
    grad_h = torch.autograd.grad(outputs=h.sum(), inputs=y_tensor,
                                 create_graph=True, retain_graph=True, allow_unused=True)[0]
    
    grad_w_p = torch.autograd.grad(outputs=grad_h[:, br_i:2*br_i], inputs=model.parameters(), 
                                   grad_outputs=lambda_t[:, 0:br_i],
                                   create_graph=True, retain_graph=True, allow_unused=True)
    
    grad_w_q = torch.autograd.grad(outputs=grad_h[:, 0:br_i], inputs=model.parameters(), 
                                   grad_outputs=lambda_t[:, br_i:2*br_i],
                                   create_graph=True, retain_graph=True, allow_unused=True)
    
    if grad_w_p is not None:
        grad_w_p = torch.cat([p_grad.flatten() for p_grad in grad_w_p]).unsqueeze(0)
        grad_w_p = grad_w_p.expand(batch_size, -1) / batch_size
    
    if grad_w_q is not None:
        grad_w_q = torch.cat([p_grad.flatten() for p_grad in grad_w_q]).unsqueeze(0)
        grad_w_q = grad_w_q.expand(batch_size, -1) / batch_size
    
    grad_w_combined = grad_w_p - grad_w_q

    model.zero_grad()
    
    return grad_w_combined.mean(dim=0)  # avg over batch


"""
Logic for Backward ODE/Adjoint Solver

Parameters:
-----------
dynamics : function  
    Function that computes the system dynamics \( f(y) \).
pred : bool  
    If `True`, performs an explicit step at the beginning using the solver.  
    If `False`, uses the noisy ground truth trajectory (`y_data`) for initialization.
lambda_data : torch.Tensor, shape (batch_size, time_steps, 2d)  
    Adjoint state trajectory, serving as terminal values for the solver.
solver : str  
    Specifies the integration method to be used:
    - `"im"` : Implicit midpoint solver.
    - `"sv"` : Störmer-Verlet (semi-implicit symplectic) solver.
dt : float  
    Time step size \( h \) for numerical integration.
shooting_segment_length : int  
    Number of time steps in each shooting segment.
number_of_shooting_segments : int  
    Number of shooting segments (short trajectories) in the integration process.
iters : int  
    Number of Newton or fixed-point iterations per solver step.

Returns:
--------
torch.Tensor  
    The computed trajectory over all shooting segments.
"""
def backward(dynamics, pred, lambda_data, solver, dt, shooting_segment_length, number_of_shooting_segments, args, iters):
    #t1, t0 = t_span
    shooting_nodes = [lambda_data[:, i*(shooting_segment_length) - 1, :] for i in range(number_of_shooting_segments, 0 ,-1)]
    segments = []
    cont_errors = []
    y_batch = args[1]
    num_steps = shooting_segment_length*number_of_shooting_segments
    batch_size = lambda_data.shape[0]

    num_params = sum(p.numel() for p in model.parameters())
    grad_result = torch.zeros(num_params, device=lambda_data.device)

    grad_result += (-dt / 2) * calculate_grad(model, y_batch[:,-1,:], lambda_data[:,-1,:], batch_size)
    
    for seg in range(number_of_shooting_segments, 0, -1):
        lam_seg = shooting_nodes[seg-1]
        #seg_states = [lam_seg]
        
        for j in range(1, shooting_segment_length):
            step_index = seg * shooting_segment_length - j - 1
            if pred:
                y_ = rk2_step(lam_seg, dt, dynamics, args, kargs=(step_index,))
            else:
                y_ = lambda_data[:, step_index, :]
            
            if solver=="im":
                lam_next = im_step(lam_seg, dt, dynamics, iters, y_, args, kargs=(step_index,))
            elif solver=="sv":
                lam_next = sv_step(lam_seg, dt, dynamics, iters, y_, args, kargs=(step_index,))
            
            #print("Heyyoo:",y_batch[:,num_steps-step_index-1,:].shape, lam_next.shape)
            
            #current_grad = calculate_grad(model, y_batch[:,num_steps-step_index-1,:], lam_next, batch_size)

            current_grad = calculate_grad(model, y_batch[:,step_index,:], lam_next, batch_size)

            #print("step Index: ", step_index)

            # if step_index==num_steps-1:
            #     grad_result += (-dt / 2) * (current_grad)
            if step_index==0:
                grad_result += (-dt / 2) * (current_grad)
            else:
                grad_result += (-dt) * (current_grad)
            
            lam_seg = lam_next
    return grad_result



def downsample_gt(gt_data, dt_solve, dt_gt):
    downsample_factor = int(dt_solve / dt_gt)
    return gt_data[:, ::downsample_factor, :]


def load_data(datafolder, dynamics, dt_solve, dt_gt):

    noisy_train_path = "../data/"+str(datafolder)+"/noisy_"+str(dynamics)+"_train.pt"
    noisy_val_path = "../data/"+str(datafolder)+"/noisy_"+str(dynamics)+"_val.pt"
    noisy_test_path = "../data/"+str(datafolder)+"/noisy_"+str(dynamics)+"_test.pt"

    train_path = "../data/"+str(datafolder)+"/"+str(dynamics)+"_train.pt"
    val_path = "../data/"+str(datafolder)+"/"+str(dynamics)+"_val.pt"
    test_path = "../data/"+str(datafolder)+"/"+str(dynamics)+"_test.pt"

    noisy_train_trajectories = torch.load(noisy_train_path).to(device)
    noisy_val_trajectories = torch.load(noisy_val_path).to(device)

    true_train_trajectories = torch.load(train_path).to(device)
    true_val_trajectories = torch.load(val_path).to(device)


    # Downsample ground truth data according to dt_solve
    noisy_train_trajectories = downsample_gt(noisy_train_trajectories, dt_solve, dt_gt)
    true_train_trajectories = downsample_gt(true_train_trajectories, dt_solve, dt_gt)


    noisy_val_trajectories = downsample_gt(noisy_val_trajectories, dt_solve, dt_gt)
    true_val_trajectories = downsample_gt(true_val_trajectories, dt_solve, dt_gt)


    return noisy_train_trajectories, noisy_val_trajectories, true_train_trajectories, true_val_trajectories


def objective(model, noisy_train_traj, noisy_val_traj, true_train_traj, true_val_traj, dt_gt, dt_solve, param_vals):

    start_time = time.time()
    
    num_epochs = 25

    learning_rate = param_vals["lr"]
    
    train_batch_size = param_vals["train_batch_size"]
    val_batch_size = param_vals["val_batch_size"]
    sim_len = param_vals["sim_len"]
    sims = param_vals["sims"]
    pred = param_vals["pred"]
    solver = param_vals["solver"]
    T = param_vals["t_start"] + (sims*sim_len - 1)*dt_solve + dt_solve
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    train_dataset = TensorDataset(noisy_train_traj, true_train_traj)
    val_dataset = TensorDataset(noisy_val_traj, true_val_traj)

    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    
    train_losses = []
    val_losses = []
    alpha = 1e-4
    # Training loop
    #print("Julian Time: ", )
    print(f"Params: Train size {noisy_train_traj.shape}, Val size {noisy_val_traj.shape}, Sim length {T} sec")

    for epoch in range(num_epochs):
        total_loss = 0.0
        
        logging.info(f"Progress: Step {epoch+1}")

        for batch in train_data_loader:
            y_noisy_batch, y_true_batch = batch
            y_noisy_batch = y_noisy_batch.to(device)
            y_true_batch = y_true_batch.to(device)

            pq0_batch = torch.tensor(y_true_batch, dtype=torch.float32)


            pq0_batch = y_true_batch.clone().detach().float()  # shape: [batch, total_steps, num_vars]
            
            # Forward integration using multiple shooting.
            y_pred_batch, cont_errors_fwd = solve_ivp_custom_ms_forward(
                forward_ode, pred, pq0_batch, solver, dt_solve,
                sim_len, sims, args=(model,), iters=6
            )
            y_pred = y_pred_batch.clone().detach()

            y_pred_batch.requires_grad_(True)

            loss = criterion(y_pred_batch[:,:, 0:(y_pred.shape[2]//2)], y_noisy_batch[:,:,0:(y_pred.shape[2]//2)]) + criterion(y_pred_batch[:, :, (y_pred.shape[2]//2):(y_pred.shape[2])], y_noisy_batch[:,:,(y_pred.shape[2]//2):(y_pred.shape[2])])
            
            lamb = torch.autograd.grad(loss, y_pred_batch, retain_graph=True)[0]

            y_pred_batch = y_pred_batch.detach()

            grads = backward(adjoint_ode, pred, lamb, solver, -dt_solve, sim_len, sims, args=(model, y_pred_batch), iters=6)

            #lambda_pred_batch = lambda_pred_batch.flip(1)

            #grads = calculate_integral(model, y_pred_batch, T, lambda_pred_batch)

            #Reshape the gradients to match the model parameters
            start_idx = 0
            for param in model.parameters():
                param_shape = param.shape
                param_size = param.numel()
                param_grad = grads[start_idx:start_idx + param_size].reshape(param_shape)
                param.grad = param_grad.clone().detach()
                start_idx += param_size

            # # Update the model parameters using the optimizer
            optimizer.step()
            optimizer.zero_grad()

            # Compute loss
            
            total_loss += loss.item()
        
        average_train_loss = total_loss / (train_batch_size)
        train_losses.append(average_train_loss)

        print(f'Epoch {epoch}/{num_epochs}, Train Loss: {total_loss/len(train_data_loader)}')
        scheduler.step(total_loss/len(train_data_loader))

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0

        with torch.no_grad():  # No gradient calculation during validation
            for batch in val_data_loader:
                y_noisy_batch_val, y_true_batch_val = batch
                y_noisy_batch_val = y_noisy_batch_val.to(device)
                y_true_batch_val = y_true_batch_val.to(device)

                pq0_batch_val = y_true_batch_val.clone().detach().float()

                # Forward pass
                y_pred_batch_val, _ = solve_ivp_custom_ms_forward(
                    forward_ode, pred, pq0_batch_val, solver, dt_solve,
                    sim_len, sims, args=(model,), iters=6
                )

                # Compute loss
                val_loss = criterion(y_pred_batch_val[:,:,0:(y_pred_batch_val.shape[2]//2)], y_noisy_batch_val[:,:,0:(y_pred_batch_val.shape[2]//2)]) + criterion(y_pred_batch_val[:,:,(y_pred_batch_val.shape[2]//2):(y_pred_batch_val.shape[2])], y_noisy_batch_val[:,:,(y_pred_batch_val.shape[2]//2):(y_pred_batch_val.shape[2])])
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / (val_batch_size)
        val_losses.append(average_val_loss)

        print(f'Epoch {epoch}/{num_epochs}, Validation Loss: {average_val_loss}')
        
        # Step the scheduler based on the validation loss
        scheduler.step(average_val_loss)


    end_time = time.time()
    
    # Log the time taken
    elapsed_time = end_time - start_time
    print(f"Objective function took {elapsed_time:.2f} seconds to complete")
    
    return train_losses, val_losses, model

def parse_hidden_layers(s):
    """Parse a string of comma-separated integers into a list of ints."""
    try:
        return list(map(int, s.strip('[]').split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Hidden layers must be a list of integers like [16,32,16]")
    

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enter the simulation parameters")
    parser.add_argument("--dynamics_name", type=str, required=True, choices=["mass_spring", "double_well", "coupled_ho", "henon_heiles"], help="The name of the dynamics function.")
    parser.add_argument("--data_folder", type=str, required=True, help="the ground truth data folder")
    parser.add_argument("--gt_res", type=float, required=True, help="the ground truth resolution/stepsize")
    parser.add_argument("--hid_layers", type=parse_hidden_layers, required=True,
                        help="Hidden layers as a list of integers, e.g., [16,32,16]")
    parser.add_argument("--solver_res", type=float, required=True, help="The time step length for our solver(= k*gt_res where k is an integer)")
    parser.add_argument("--noise_level", type=float, required=False, default=0.0,
                    help="The noise level (a float number from data_gen). Default is 0.0.")
    parser.add_argument("--pred", type=lambda x: bool(distutils.util.strtobool(x)), required=False, default=False, 
                    help="Boolean flag: True if you need a predictor step, False if you use GT (default: False)")
    parser.add_argument("--num_sims", type=int, required=False, default=1, help="The number of multi-shooting trajectories (default= 1 single shooting)")
    parser.add_argument("--sim_len", type=int, required=True, help="The forward simulation length of each trajectory for training.")
    parser.add_argument("--solver", type=str, required=False, default="im", choices=["im","sv"])

    args = parser.parse_args()
    
    dynamics_name = args.dynamics_name
    data_folder = args.data_folder
    dt_gt = args.gt_res
    hidden_layer_sizes = args.hid_layers
    dt_solve = args.solver_res
    noise_level = args.noise_level
    pred = args.pred
    num_sims = args.pred
    sim_len = args.sim_len
    solver = args.solver

    noisy_train, noisy_val, true_train, true_val = load_data(data_folder, dynamics_name, dt_gt, dt_solve)

    input_size = noisy_train.shape[2]
    output_size = 1

    train_set_len = int(noisy_train.shape[0])
    val_set_len = int(noisy_val.shape[0])

    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
    
    model_specs = (layer_sizes,)

    model = HamiltonianNN(model_specs).to(device) 

    params_list = [{"sim_len":sim_len, "lr":0.01, "pred":pred, "solver":solver,
                     "train_batch_size":512, "val_batch_size":512, "t_start":0, "sims": num_sims}
                    # ,{"sim_len":10, "lr":0.008,
                    #   "train_batch_size":512, "val_batch_size":512, "t_final":0.1}
                    #   ,{"sim_len":100, "lr":0.005,
                    #   "train_batch_size":512, "val_batch_size":512, "t_final":1.0}
                     ]
    
    
    train_losses = []
    val_losses = []
    models = []

    for i in range(len(params_list)):
        
        start_ind = int(params_list[i]["t_start"]/dt_solve)
        end_ind = params_list[i]["sim_len"] * params_list[i]["sims"] 

        print("Trial: ", str(i))
        
        train_loss, val_loss, model = objective(model, noisy_train[i*train_set_len:(i+1)*train_set_len, 0:end_ind, :],
                                                 noisy_val[i*val_set_len:(i+1)*val_set_len, 0:end_ind, :], 
                                                 true_train[i*train_set_len:(i+1)*train_set_len, 0:end_ind, :], 
                                                 true_val[i*val_set_len:(i+1)*val_set_len, 0:end_ind, :], 
                                                 dt_gt, dt_solve, params_list[i])

        torch.save(model, f'../models/model_{i}_{dynamics_name}_{noise_level}_adjoint_{num_sims}_{sim_len}_{solver}.pt')
        df = pd.DataFrame({
        "train_loss": train_loss,
        "val_loss": val_loss
        })
        df.to_csv(f'output_{dynamics_name}_{noise_level}_adjoint_{num_sims}_{sim_len}_{solver}.csv', index=False)
        
        

