#import matplotlib.pyplot as plt
import os
os.system("ml tqdm/4.64.1-GCCcore-12.2.0")
from tqdm import tqdm
os.system("ml PyTorch/2.2.1-foss-2023b-CUDA-12.4.0")
import torch
import numpy as np
import argparse

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

num_trajectories_train = 32678
num_trajectories_val = 8192
num_trajectories_test = 1024
T = 10
dt = 1e-2
noise_level = 0.008

def mass_spring(y, args, kargs):
    q = y[:, 0]
    p = y[:, 1]
    dq_dt = p
    dp_dt = -q
    return torch.stack((dq_dt, dp_dt), dim=-1)

def double_well(y, args, kargs):
    q = y[:, 0]
    p = y[:, 1]
    dq_dt = p
    dp_dt = q - q**3
    return torch.stack((dq_dt, dp_dt), dim=-1)

# Dynamics for the non-separable Hamiltonian system
def coupled_ho(y, args, kargs):
    alpha = args["alpha"]
    q, p = y[:, 0], y[:, 1]
    dqdt = p + alpha * q
    dpdt = -q - alpha * p
    return torch.stack((dqdt, dpdt), dim=-1)

# def rk2_step(dyn, y, dt, dynamics, args, kargs):
#     h = dt
#     i = kargs[0]
#     q, p = y[:, 0], y[:, 1]

#     y = torch.stack((q, p), dim=-1)  # Shape: (batch_size, 2)

#     dy1 = dynamics(y, args, kargs)
#     q1 = q + 0.5 * dy1[:, 0] * h
#     p1 = p + 0.5 * dy1[:, 1] * h

#     y1 = torch.stack((q1, p1), dim=-1)  # Shape: (batch_size, 2)
#     dy2 = dynamics(y1, args, kargs)

#     q_new = q + dy2[:, 0] * h
#     p_new = p + dy2[:, 1] * h
#     return q_new, p_new

# def sv_step(dyn, y, dt, dynamics, iterations, y_init, args, kargs):
#     h = dt
#     q, p = y[:, 0], y[:, 1]
#     i = kargs[0]
#     p_half = p + 0.5 * h * dynamics(torch.stack((q, y_init[:, 1]), dim=-1), args, kargs)[:, 1]
#     for _ in range(iterations):
#         p_half = p + 0.5 * h * dynamics(torch.stack((q, p_half), dim=-1), args, kargs)[:, 1]

#     q_half = q + 0.5 * h * dynamics(torch.stack((y_init[:, 0], p_half), dim=-1), args, kargs)[:, 0]
#     for _ in range(iterations):
#         q_half = q + 0.5 * h * dynamics(torch.stack((q_half, p), dim=-1), args, kargs)[:, 0]

#     q_new = q + h * dynamics(torch.stack((q_half, p_half), dim=-1), args, kargs)[:, 0]
#     p_new = p_half + 0.5 * h * dynamics(torch.stack((q_new, p_half), dim=-1), args, kargs)[:, 1]

#     return torch.stack((q_new, p_new), dim=-1)

# RK2 step function for initial guess
def rk2_step(dyn, y, dt, dynamics, args, kargs):
    h = dt
    i = kargs[0]
    q, p = y[:, 0:1], y[:, 1:2]

    y = torch.cat((q, p), dim=-1)  # Shape: (batch_size, 2)

    #print(q.shape)

    dy1 = dynamics(y, args, kargs)
    q1 = q + 0.5 * dy1[:, 0:1] * h
    p1 = p + 0.5 * dy1[:, 1:2] * h

    y1 = torch.cat((q1, p1), dim=-1)  # Shape: (batch_size, 2)
    dy2 = dynamics(y1, args, kargs)

    q_new = q + dy2[:, 0:1] * h
    p_new = p + dy2[:, 1:2] * h
    return q_new, p_new

# Implicit midpoint step (provided by user)
def im_step(dyn, y, dt, dynamics, iterations, y_init, args, kargs):
    h = dt
    q, p = y[:, 0:1], y[:, 1:2]
    y_init_concat = torch.cat((y_init[:, 0:1], y_init[:, 1:2]), dim=-1)
    f_init = dynamics(y_init_concat, args, kargs)
    q_new = q + 0.5 * h * f_init[:, 0:1]
    p_new = p + 0.5 * h * f_init[:, 1:2]

    for _ in range(iterations):
        mid_q = 0.5 * (q + q_new)
        mid_p = 0.5 * (p + p_new)
        f_mid = dynamics(torch.cat((mid_q, mid_p), dim=-1), args, kargs)
        q_new = q + h * f_mid[:, 0:1]
        p_new = p + h * f_mid[:, 1:2]

    return torch.cat((q_new, p_new), dim=-1)

def solve_ivp_custom(type, dynamics, dyn, y_init, t_span, dt, args, iters):
    t0, t1 = t_span
    if t0 > t1:
        dt = -dt
    #t_vals = np.arange(t0, t1, dt)
    t_vals = np.linspace(t0, t1, int(t1/dt)+1)
    batch_size = y_init.shape[0]

    _y = torch.zeros((batch_size, t_vals.shape[0], 2), dtype=torch.float32, requires_grad=False, device=device)
    y = y_init.clone()

    _y[:, 0, :] = y

    for i, t in enumerate(t_vals[1:], start=1):
        q_, p_ = rk2_step(dyn, y.clone(), dt, dynamics, args, kargs=(i,))
        y = im_step(dyn, y.clone(), dt, dynamics, iters, torch.cat((q_, p_), dim=-1), args, kargs=(i,))
        _y[:, i, :] = y
    
    print("generated trajectories for: ", type)
    return _y

# Function to add noise
def add_noise(trajectories, noise_level):
    # Clone the trajectories to avoid modifying the original data
    noisy_trajectories = trajectories.clone()
    
    noise = noise_level * torch.randn_like(trajectories[:, 1:, :], device=device)
    # Add the generated noise to the trajectories (excluding the first time step)
    noisy_trajectories[:, 1:, :] += noise
    
    return noisy_trajectories


# Parse command line arguments
parser = argparse.ArgumentParser(description="Enter the simulation parameters")
parser.add_argument("--dynamics_name", type=str, required=True, choices=["mass_spring", "double_well", "coupled_ho"], help="The name of the dynamics function.")
parser.add_argument("--q_range", type=float, nargs=2, required=True, help="range for q0 (pair of space seperated floating point numbers)")
parser.add_argument("--p_range", type=float, nargs=2, required=True, help="range for p0 (pair of space seperated floating point numbers)")
parser.add_argument("--sim_len", type=int, required=True, help="The total simulation time length (an int)")
parser.add_argument("--time_step", type=float, required=True, help="The time step length (< sim_len)")

args = parser.parse_args()

# Unpack the ranges directly
q_min, q_max = args.q_range
p_min, p_max = args.p_range

dynamics_name = args.dynamics_name
q_range = args.q_range
p_range = args.p_range
sim_len = args.sim_len
time_step = args.time_step
#q_min, q_max = args.q_range
#p_min, p_max = args.p_range

#print(type(q_min))

if args.dynamics_name == "mass_spring":
    dynamics = mass_spring
elif args.dynamics_name == "double_well":
    dynamics = double_well
elif args.dynamics_name == "coupled_ho":
    dynamics = coupled_ho


print(f"q_min: {q_min}, q_max: {q_max}, type of q_min: {type(q_min)}, type of q_max: {type(q_max)}")
print(f"p_min: {p_min}, p_max: {p_max}, type of p_min: {type(p_min)}, type of p_max: {type(p_max)}")

# Ensure that q_min, q_max, p_min, and p_max are tensors
q_min = torch.tensor(q_min, device=device)
q_max = torch.tensor(q_max, device=device)
p_min = torch.tensor(p_min, device=device)
p_max = torch.tensor(p_max, device=device)



# Initialize the tensors and move them to the GPU if available
inits_train = torch.cat(((torch.rand(num_trajectories_train, 1, device=device) - 0.5)*(q_max - q_min) + ((q_max + q_min)/2),  # q0 range q_min to q_max
                         (torch.rand(num_trajectories_train, 1, device=device) - 0.5)*(q_max - q_min) + ((q_max + q_min)/2)) , # p0 range p_min to p_max
                        dim=-1)

print(inits_train.shape)

inits_val = torch.cat(((torch.rand(num_trajectories_val, 1, device=device) - 0.5)*(q_max - q_min) + ((q_max + q_min)/2),  # q0 range q_min to q_max
                         (torch.rand(num_trajectories_val, 1, device=device) - 0.5)*(q_max - q_min) + ((q_max + q_min)/2)) , # p0 range p_min to p_max
                        dim=-1)

inits_test = torch.cat(((torch.rand(num_trajectories_test, 1, device=device) - 0.5)*(q_max - q_min) + ((q_max + q_min)/2),  # q0 range q_min to q_max
                         (torch.rand(num_trajectories_test, 1, device=device) - 0.5)*(q_max - q_min) + ((q_max + q_min)/2)) , # p0 range p_min to p_max
                        dim=-1)

args = {"alpha": 0.5}

trajectories_train = solve_ivp_custom("train", dynamics, dynamics_name, inits_train, [0, sim_len], time_step, args, 5)
trajectories_val = solve_ivp_custom("val", dynamics, dynamics_name, inits_val, [0, sim_len], time_step, args, 5)
trajectories_test = solve_ivp_custom("test", dynamics, dynamics_name, inits_test, [0, sim_len], time_step, args, 5)



noisy_trajectories_train = add_noise(trajectories_train, noise_level)
noisy_trajectories_val = add_noise(trajectories_val, noise_level)
noisy_trajectories_test = add_noise(trajectories_test, noise_level)

# Define the save path
savepath = os.path.join('/home/choudhar/HNNs/data', f"{dynamics_name}_{sim_len}")

os.makedirs(savepath, exist_ok=True)


torch.save(noisy_trajectories_train.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_train.pt'))
torch.save(noisy_trajectories_val.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_val.pt'))
torch.save(noisy_trajectories_test.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_test.pt'))

torch.save(trajectories_train.cpu(), os.path.join(savepath, f'{dynamics_name}_train.pt'))
torch.save(trajectories_val.cpu(), os.path.join(savepath, f'{dynamics_name}_val.pt'))
torch.save(trajectories_test.cpu(), os.path.join(savepath, f'{dynamics_name}_test.pt'))

print(f"Data successfully saved in: {savepath}")
