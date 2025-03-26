import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the spring-mass Hamiltonian system
def K_t(q, p):
    m = 1.0  # Mass
    k = 1.0  # Spring constant
    dq_dt = p / m  # dq/dt = p/m
    dp_dt = -k * q  # dp/dt = -kq
    return dq_dt, dp_dt

# Calculate the Hamiltonian H
def calculate_H(q, p):
    m = 1.0  # Mass
    k = 1.0  # Spring constant
    H = (p**2) / (2 * m) + (k * q**2) / 2
    return H

# Runge-Kutta 2 (RK2) step
def rk2_step(q, p, dt, K_t):
    h = dt
    dq1, dp1 = K_t(q, p)
    q1 = q + 0.5 * dq1 * h
    p1 = p + 0.5 * dp1 * h
    dq2, dp2 = K_t(q1, p1)
    q_new = q + dq2 * h
    p_new = p + dp2 * h
    return q_new, p_new

# Störmer-Verlet (SV) step
def sv_step(q, p, dt, K_t, iterations, x_init):
    h = dt
    p_half = p + 0.5 * h * K_t(q, x_init[1])[1]
    for _ in range(iterations):
        p_half = p + 0.5 * h * K_t(q, p_half)[1]
    q_half = q + 0.5 * h * K_t(x_init[0], p_half)[0]
    for _ in range(iterations):
        q_half = q + 0.5 * h * K_t(q_half, p)[0]
    q_new = q + h * K_t(q_half, p_half)[0]
    p_new = p_half + 0.5 * h * K_t(q_new, p_half)[1]
    return q_new, p_new

# Predictor-Corrector (PC) method
def PC(q, p, dt, K_t, eps, iterations=1, entire_trajectory=True):
    n_steps = int(np.round((np.abs(dt) / eps).max().item()))
    h = dt / n_steps
    trajectory = []

    for i_step in range(int(n_steps)):
        q_, p_ = rk2_step(q, p, h, K_t)
        q, p = sv_step(q, p, h, K_t, iterations, (q_, p_))
        H = calculate_H(q, p)
        if entire_trajectory:
            trajectory.append((q.clone(), p.clone(), H.clone()))

    if entire_trajectory:
        return trajectory
    else:
        return q, p

# Function to generate trajectories
def generate_trajectories(num_trajectories, dt, eps, K_t, iterations):
    trajectories = []
    for _ in range(num_trajectories):
        q0 = torch.tensor(2 * np.random.rand() - 1, dtype=torch.float32)
        p0 = torch.tensor(2 * np.random.rand() - 1, dtype=torch.float32)
        trajectory = PC(q0, p0, dt, K_t, eps, iterations, entire_trajectory=True)
        trajectories.append(trajectory)
    return trajectories

# Parameters
dt = 0.1
eps = 0.01
iterations = 5

# Generate training, validation, and test trajectories
train_trajectories = generate_trajectories(100, dt, eps, K_t, iterations)
val_trajectories = generate_trajectories(20, dt, eps, K_t, iterations)
test_trajectories = generate_trajectories(20, dt, eps, K_t, iterations)


data = "mass_spring"
os.makedirs("data/"+str(data), exist_ok=True)

np.save("data/"+str(data)+"/train.npy", train_trajectories)
np.save("data/"+str(data)+"/val.npy", val_trajectories)
np.save("data/"+str(data)+"/test.npy", test_trajectories)
