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
# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])


# # Define the neural network class
# class SingleLayerNN(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
#         super(SingleLayerNN, self).__init__()
#         self.hidden_layer1 = nn.Linear(input_size, hidden_size1, bias=False)
#         self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
#         self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3, bias=False)
#         self.output_layer = nn.Linear(hidden_size3, output_size, bias=False)

#     def forward(self, x):
#         hidden_output1 = torch.sigmoid(self.hidden_layer1(x))
#         hidden_output2 = torch.sigmoid(self.hidden_layer2(hidden_output1))
#         hidden_output3 = torch.sigmoid(self.hidden_layer3(hidden_output2))
#         output = torch.sigmoid(self.output_layer(hidden_output3))
#         return output
    
class HamiltonianNN(nn.Module):

    # For now the input data is passed as init parameters
    def __init__(self, model_specs):
        super(HamiltonianNN, self).__init__()

        # Create a list of linear layers based on layer_sizes
        layer_sizes = model_specs[0]
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.RANDOM_SEED = 0
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            if i < len(layer_sizes) - 2:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            #x = x.requires_grad_(True)
            x = torch.sigmoid(x)
        x = ((self.layers[-1](x)))
        return x/32.
    
class LinearBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(LinearBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Linear(inchannel, outchannel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.left(x)
        return out
    

class KTrained(nn.Module):
    def __init__(self, N, hidden_dim):
        super(KTrained, self).__init__()
        self.N = N
        self.cal_H = nn.Sequential(LinearBlock(2 * self.N, hidden_dim),
                                    LinearBlock(hidden_dim, hidden_dim),
                                    LinearBlock(hidden_dim, hidden_dim),
                                    LinearBlock(hidden_dim, hidden_dim),
                                    LinearBlock(hidden_dim, hidden_dim),
                                    nn.Linear(hidden_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-math.sqrt(6. / m.in_features), math.sqrt(6. / m.in_features))

    def forward_train(self, q, p):
        with torch.enable_grad():
            x = torch.cat([q.unsqueeze(1), p.unsqueeze(1)], dim=1)
            x = x.requires_grad_(True)
            K = self.cal_H(x.squeeze(1))
            dK = torch.autograd.grad(K.sum(), x, retain_graph=True, create_graph=True)[0]
            #print(dK.shape)
        return dK

    def forward(self, x, args, kargs):
        with torch.enable_grad():
            #x = torch.cat([q.unsqueeze(1), p.unsqueeze(1)], dim=1)
            x = x.requires_grad_(True)
            K = self.cal_H(x.squeeze(1))
            #print("Hamiltonian is : ", K.sum())
            dK = torch.autograd.grad(K.sum(), x, retain_graph=False, create_graph=False)[0]
        return dK


def forward_ode(y_tensor, args, kargs):

    model = args[0]
    i = kargs[0]
        
    with torch.enable_grad():

        y = y_tensor.clone().detach().requires_grad_(True)

        h = model(y)
    
        grad_h = torch.autograd.grad(outputs=h.sum(), inputs=y, create_graph=True, retain_graph=True, allow_unused=True)[0]
        #print("grad h: ", grad_h)
        dq_dt = grad_h[:, 1]
        dp_dt = -grad_h[:, 0]

    return torch.stack((dq_dt, dp_dt), dim=-1)


def adjoint_ode(lam, args, kargs):

    (model, y_values, y_gt_values) = args
    (i,) = kargs

    y_tensor = y_values[:,i,:].clone().detach().requires_grad_(True)
    
    y_gt_tensor = y_gt_values[:,i,:].clone().detach()

    # # Forward pass through the model (for a dynamics prediction model)
    #w = model(y_tensor)
 

    # Forward pass through the model (for a Hamiltonian prediction model)
    h = model(y_tensor)

    #print("W shape: ", w.shape)

    #system dynamics h = df/dt
    grad_h = torch.autograd.grad(outputs=h.sum(dim=0), inputs=y_tensor, create_graph=True, retain_graph=True, allow_unused=True)[0]


    grad_m_0 = torch.autograd.grad(outputs=grad_h[:,1], inputs=y_tensor, grad_outputs=torch.ones_like(grad_h[:,1]), create_graph=True, allow_unused=True)[0]
    grad_m_1 = torch.autograd.grad(outputs=-grad_h[:,0], inputs=y_tensor, grad_outputs=torch.ones_like(-grad_h[:,0]), create_graph=True, allow_unused=True)[0]

    
    #grad_m_0 = torch.autograd.grad(outputs=grad_h[:,1], inputs=y_tensor, grad_outputs=lam[:,0], create_graph=True, allow_unused=True)[0]
    #grad_m_1 = torch.autograd.grad(outputs=-grad_h[:,0], inputs=y_tensor, grad_outputs=-lam[:,1], create_graph=True, allow_unused=True)[0]

    #grad_m_0 = torch.autograd.grad(outputs=w[:,0], inputs=y_tensor, grad_outputs=torch.ones_like(w[:,0]), create_graph=True, allow_unused=True)[0]
    #grad_m_1 = torch.autograd.grad(outputs=w[:,1], inputs=y_tensor, grad_outputs=torch.ones_like(w[:,1]), create_graph=True, allow_unused=True)[0]
    
    j_w = torch.stack((grad_m_0, grad_m_1), dim=2).transpose(1, 2)

    #print("jw shape: ", j_w.shape)


    num = (y_values.shape[1])
    
    lam_tensor = lam.clone().detach().unsqueeze(2)

    #print("lamda shape: ", lam_tensor.shape)

    lam_dot = - (torch.bmm(j_w, lam_tensor)).squeeze()

    #print(lam_dot.shape)
    #print("Done once")

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


def calculate_integral(model, y_values, T, lambda_values):
    integral_values = []
    grad_vals = []

    t_eval = torch.linspace(0, T, y_values.shape[1], device=y_values.device)

    for i, t in enumerate(t_eval):
        y_val = y_values[:, i, :]  # Shape: (batch_size, 2)
        
        # Set requires_grad=True to track operations on y_val
        y_tensor = y_val.clone().detach().requires_grad_(True).to(y_values.device)
        
        # Perform the forward pass for the entire batch
        h = model(y_tensor)  # Shape: (batch_size, 1)
        
        # Compute gradients of the model output w.r.t y_tensor for the whole batch
        grad_h = torch.autograd.grad(outputs=h.sum(), inputs=y_tensor,
                                     create_graph=True, retain_graph=True, allow_unused=True)[0]  # Shape: (batch_size, 2)
        
        # print(">>>>>>", grad_h.shape)
        
                
        y_tensor.requires_grad = False

        #parameters = [p.view(-1) for p in model.parameters()]

        # Concatenate the flattened parameters into a single tensor
        #flattened_params = torch.cat(parameters)

        #print(flattened_params.shape)
        
        grad_w_0 = torch.autograd.grad(outputs=grad_h[:,1], inputs=model.parameters(), grad_outputs=lambda_values[:, i, 0], create_graph=True, retain_graph = True, allow_unused=True)
        grad_w_1 = torch.autograd.grad(outputs=-grad_h[:,0], inputs=model.parameters(), grad_outputs=lambda_values[:, i, 1], create_graph=True, retain_graph = True, allow_unused=True)
        
        
        #print(grad_w_0.shape)
        
        grad_w_0_flat = [g.view(-1) if g is not None else torch.zeros_like(param).view(-1) 
                         for g, param in zip(grad_w_0, model.parameters())]
        grad_w_1_flat = [g.view(-1) if g is not None else torch.zeros_like(param).view(-1) 
                         for g, param in zip(grad_w_1, model.parameters())]
        
        #print("grad_w_0_flat[0] shape", grad_w_0_flat[0].shape)
        
        grad_w_0_flat_tensor = torch.cat(grad_w_0_flat, dim=0)  # Shape: (batch_size, num_params)
        grad_w_1_flat_tensor = torch.cat(grad_w_1_flat, dim=0)  # Shape: (batch_size, num_params)

        #print(grad_w_0_flat_tensor.shape)
        
        # Stack the tensors
        #grad_w = torch.stack((grad_w_0_flat_tensor, grad_w_1_flat_tensor), dim=0)

        #print("grad_w shape: ", grad_w.shape)
        #print("lambda shape: ", lambda_values[:, i, :].shape)
        grad_w_combined = grad_w_0_flat_tensor + grad_w_1_flat_tensor
        #lambda_mean = lambda_values[:, i, :].mean(dim=0, keepdim=True)
        
        #integral_value = torch.matmul(lambda_values[:, i, :], grad_w)
        integral_value = grad_w_combined
        #integral_value = torch.mean(integral_value, dim=0)
                
        #print(">>>> ", grad_w)
        #integral_value = torch.matmul(lambda_mean.float(), grad_w)
        #integral_value_mean = integral_value.mean(dim=0)  # Shape: (num_params,)
    
        # Append to the integral values list
        integral_values.append(integral_value)
        
        #integral_value = torch.matmul(lambda_values[:, i, :].unsqueeze(1).float(), grad_w).squeeze(1)  # Shape: (batch_size, num_params)
        
        # Average the result across the batch
        #integral_values.append(integral_value.mean(dim=0))

    # Stack integral values and compute the final integral using the trapezoidal rule
    integral_values = torch.stack(integral_values)  # Shape: (time_stamp, num_params)
    integral_result = torch.trapz(integral_values, t_eval, dim=0)  # Shape: (num_params,)

    return integral_result


def rk2_step(dyn, y, dt, dynamics, args, kargs):
    h = dt
    i = kargs[0]
    q, p = y[:, 0], y[:, 1]

    y = torch.stack((q, p), dim=-1)  # Shape: (batch_size, 2)

    #print(q.shape)

    dy1 = dynamics(y, args, kargs)
    q1 = q + 0.5 * dy1[:, 0] * h
    p1 = p + 0.5 * dy1[:, 1] * h

    y1 = torch.stack((q1, p1), dim=-1)  # Shape: (batch_size, 2)
    dy2 = dynamics(y1, args, kargs)

    q_new = q + dy2[:, 0] * h
    p_new = p + dy2[:, 1] * h
    return torch.stack((q_new, p_new), dim=-1)


def sv_step(dyn, y, dt, dynamics, iterations, y_init, args, kargs):
    h = dt
    q, p = y[:, 0], y[:, 1]
    i = kargs[0]

    p_half = p + 0.5 * h * dynamics(torch.stack((q, y_init[:, 1]), dim=-1), args, kargs)[:, 1]
    for _ in range(iterations):
        p_half = p + 0.5 * h * dynamics(torch.stack((q, p_half), dim=-1), args, kargs)[:, 1]

    q_half = q + 0.5 * h * dynamics(torch.stack((y_init[:, 0], p_half), dim=-1), args, kargs)[:, 0]
    for _ in range(iterations):
        q_half = q + 0.5 * h * dynamics(torch.stack((q_half, p), dim=-1), args, kargs)[:, 0]

    q_new = q + h * dynamics(torch.stack((q_half, p_half), dim=-1), args, kargs)[:, 0]
    p_new = p_half + 0.5 * h * dynamics(torch.stack((q_new, p_half), dim=-1), args, kargs)[:, 1]

    return torch.stack((q_new, p_new), dim=-1)


def solve_ivp_custom(dynamics, dyn, y0_batch, t_span, dt, args, iters):
    #t = torch.arange(0, T, dt)
    batch_size = y0_batch.shape[0]
    t0, t1 = t_span
    if t0 > t1:
        dt = -dt
    num_steps = int((t1 - t0) / dt) + 1
    #y0_batch = noisy_obs[:, 0, :]
    ys_batch = [y0_batch]
    #print(y0_batch.shape)

    #_y = torch.zeros((batch_size, t_vals.shape[0], 2), dtype=torch.float32, requires_grad=False, device=y_init.device)
    #_y[:, 0, :] = y_init.clone()

    for i in range(1, num_steps):
        #y = noisy_obs[:, i-1, :]  # Use the noisy observation at the current step
        y = ys_batch[-1]
        y_ = rk2_step(dyn, y, dt, dynamics, args, kargs=(i,))
        y_next = sv_step(dyn, y, dt, dynamics, iters, y_, args, kargs=(i,))
        #print(y_next.requires_grad)
        ys_batch.append(y_next)
        #print(y_next.shape)
    ys_batch = torch.stack(ys_batch, dim=1)
    #print(ys_batch.requires_grad)
    return ys_batch

def downsample_gt(gt_data, dt_solve, dt_gt):
    downsample_factor = int(dt_solve / dt_gt)
    return gt_data[:, ::downsample_factor, :]


def load_data(datafolder, dt_solve, dt_gt):

    datafolder = "double_well_20"

    noisy_train_path = "../data/"+str(datafolder)+"/noisy_double_well_train.pt"
    noisy_val_path = "../data/"+str(datafolder)+"/noisy_double_well_val.pt"
    noisy_test_path = "../data/"+str(datafolder)+"/noisy_double_well_test.pt"

    train_path = "../data/"+str(datafolder)+"/double_well_train.pt"
    val_path = "../data/"+str(datafolder)+"/double_well_val.pt"
    test_path = "../data/"+str(datafolder)+"/double_well_test.pt"

    noisy_train_trajectories = torch.load(noisy_train_path).to(device)
    noisy_val_trajectories = torch.load(noisy_val_path).to(device)

    true_train_trajectories = torch.load(train_path).to(device)
    true_val_trajectories = torch.load(val_path).to(device)

    # noisy_train_trajectories = noisy_train_trajectories[:,0:traj_len,:]
    # noisy_val_trajectories = noisy_val_trajectories[:,0:traj_len,:]
    # true_train_trajectories = true_train_trajectories[:,0:traj_len,:]
    # true_val_trajectories = true_val_trajectories[:, 0:traj_len,:]


    # Downsample ground truth data according to dt_solve
    noisy_train_trajectories = downsample_gt(noisy_train_trajectories, dt_solve, dt_gt)
    true_train_trajectories = downsample_gt(true_train_trajectories, dt_solve, dt_gt)


    noisy_val_trajectories = downsample_gt(noisy_val_trajectories, dt_solve, dt_gt)
    true_val_trajectories = downsample_gt(true_val_trajectories, dt_solve, dt_gt)


    return noisy_train_trajectories, noisy_val_trajectories, true_train_trajectories, true_val_trajectories


def objective(model, noisy_train_traj, noisy_val_traj, true_train_traj, true_val_traj, dt_gt, dt_solve, param_vals):

    start_time = time.time()
    # Parameters
    #num_trajectories = 64
    #T = 0.1
    #noise_level = 0.008
    num_epochs = 100
    #batch_size = 256
    #input_size = 2
    #hidden_size1 = 8
    #hidden_size2 = 16
    #hidden_size3 = 8
    #output_size = 2
    #datafolder = "mass_spring"
    
    learning_rate = param_vals["lr"]
    #train_data_size = param_vals["train_data_size"]
    #val_data_size = param_vals["val_data_size"]
    train_batch_size = param_vals["train_batch_size"]
    val_batch_size = param_vals["val_batch_size"]
    T = param_vals["t_final"]
    #layer_sizes = param_vals["layer_sizes"]
    
    #layer_sizes = [input_size, hidden_size1, hidden_size2, hidden_size3, output_size]

    #model2 = KTrained(N, hidden_dim).to(device)

    #learning_rate = 0.05
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    train_dataset = TensorDataset(noisy_train_traj, true_train_traj)
    val_dataset = TensorDataset(noisy_val_traj, true_val_traj)

    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    
    train_losses = []
    val_losses = []
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
            #print("noisy shape: ",y_noisy_batch.shape)
            #print("true shape:", y_true_batch.shape)
            pq0_batch = torch.tensor(y_true_batch[:, 0, :], dtype=torch.float32)

            y_pred_batch = solve_ivp_custom(forward_ode, "forward", pq0_batch, (0, T), dt_solve, args=(model,), iters=5)

            y_pred = y_pred_batch.clone().detach()

            q_batch = y_pred[:, :, 0]
            p_batch = y_pred[:, :, 1]
            q_batch.requires_grad_(True)
            p_batch.requires_grad_(True)

            #loss = criterion(y_pred[:, :, 0], y_noisy_batch[:,:,0]) + criterion(y_pred[:, :, 1], y_noisy_batch[:,:,1])
            loss = criterion(q_batch, y_noisy_batch[:,:,0]) + criterion(p_batch, y_noisy_batch[:,:,1])
            #print("q batch 4: ", q_batch[4,:])
            #print("q batch true 4: ", y_noisy_batch[4,:, 0])
            #print("p batch 4: ", p_batch[4,:])
            #print("p batch true 4: ", y_noisy_batch[4,:, 1])
            lamb_q = torch.autograd.grad(loss, q_batch, retain_graph=True)[0]
            lamb_p = torch.autograd.grad(loss, p_batch, retain_graph=True)[0]
            lamb_0 = torch.stack((lamb_q[:,-1], lamb_p[:,-1]), dim=1)
            #lamb = torch.autograd.grad(loss, y_pred, retain_graph=True)[0]
            #lamb_0 = lamb[:, -1, :]
            #print("lamb_0 here: ", lamb_0)

            #print("lambda pred batch before: ", lamb_0[5, :])

            lambda_pred_batch = solve_ivp_custom(adjoint_ode, "adjoint", lamb_0, (T, 0), dt_solve, args=(model, y_pred_batch, y_noisy_batch), iters=5)

            lambda_pred_batch = lambda_pred_batch.flip(1)

            #print("lambda pred batch after: ", lambda_pred_batch[5, -1, :])

            grads = calculate_integral(model, y_pred_batch, T, lambda_pred_batch)

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
                pq0_batch = torch.tensor(y_true_batch_val[:, 0, :], dtype=torch.float32).to(device)

                # Forward pass
                y_pred_batch_val = solve_ivp_custom(forward_ode, "forward", pq0_batch, (0, T), dt_solve, args=(model,), iters=5)

                # Compute loss
                val_loss = criterion(y_pred_batch_val[:,:,0], y_noisy_batch_val[:,:,0]) + criterion(y_pred_batch_val[:,:,1], y_noisy_batch_val[:,:,1])
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / (val_batch_size)
        val_losses.append(average_val_loss)

        print(f'Epoch {epoch}/{num_epochs}, Validation Loss: {average_val_loss}')
        
        # Step the scheduler based on the validation loss
        scheduler.step(average_val_loss)

        # # Log the training and validation losses
        # trial.set_user_attr('train_losses', train_losses)
        # trial.set_user_attr('val_losses', val_losses)
    end_time = time.time()
    
    # Log the time taken
    elapsed_time = end_time - start_time
    print(f"Objective function took {elapsed_time:.2f} seconds to complete")

    return train_losses, val_losses, model

if __name__ == "__main__":

    dt_gt = 0.01
    dt_solve = 0.01
    input_size = 2
    output_size = 1
    noisy_train, noisy_val, true_train, true_val = load_data("double_well", dt_gt, dt_solve)

    train_set_len = int(noisy_train.shape[0])
    val_set_len = int(noisy_val.shape[0])

    layer_sizes = [input_size, 16, 32, 16, output_size]
    
    model_specs = (layer_sizes,)

    model = HamiltonianNN(model_specs).to(device)

    #model = KTrained(1, 64) 

    params_list = [{"sim_len":5, "lr":0.05,
                     "train_batch_size":512, "val_batch_size":512, "t_final":0.04}
                    # ,{"sim_len":10, "lr":0.008,
                    #   "train_batch_size":512, "val_batch_size":512, "t_final":0.1}
                    #   ,{"sim_len":100, "lr":0.005,
                    #   "train_batch_size":512, "val_batch_size":512, "t_final":1.0}
                     ]
    
    train_losses = []
    val_losses = []
    models = []

    for i in range(len(params_list)):

        print("Trial: ", str(i))
        train_loss, val_loss, model = objective(model, noisy_train[i*train_set_len:(i+1)*train_set_len, 0:params_list[i]["sim_len"], :],
                                                 noisy_val[i*val_set_len:(i+1)*val_set_len, 0:params_list[i]["sim_len"], :], 
                                                 true_train[i*train_set_len:(i+1)*train_set_len, 0:params_list[i]["sim_len"], :], 
                                                 true_val[i*val_set_len:(i+1)*val_set_len, 0:params_list[i]["sim_len"], :], 
                                                 dt_gt, dt_solve, params_list[i])
        #os.makedirs(path, exist_ok=True) 
        torch.save(model, f'../models/model_{i}_dw')
        df = pd.DataFrame({
        "train_loss": train_loss,
        "val_loss": val_loss
        })
        df.to_csv(f'output_{i}_dw.csv', index=False)
        
        

