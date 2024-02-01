#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:13:20 2023

@author: forootani
"""

import numpy as np
import sys
import os
def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir
root_dir = setting_directory(1)

from pathlib import Path
import torch
from scipy import linalg

import torch.nn as nn
import torch.nn.init as init

from PINN_DEIM.Functions.samples import Subsample_random
from PINN_DEIM.Functions.base import Dataset, get_train_test_loader
from PINN_DEIM.Functions.modules import Siren
from PINN_DEIM.Functions.utils import loss_func_AC
from PINN_DEIM.Functions.utils import leastsquares_fit
from PINN_DEIM.Functions.utils import equation_residual_AC
from PINN_DEIM.Functions.library import library_deriv
import PINN_DEIM.Functions.plot_config_file

from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from PINN_DEIM.Functions.DEIM_class import DEIM



import warnings
import time

cwd = os.getcwd()

#sys.path.append(cwd + '/my_directory')

sys.path.append(cwd)

warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

########################################
########################################

data = scipy.io.loadmat(root_dir +'/data/kdv.mat')
#data = scipy.io.loadmat('data/KdV_ESINDy.mat')
print(data.keys())
t_data = data["t"].flatten()
x_data = data["x"].flatten()
u_data = data["usol"]

#u_data=data["U_exact"]

t = data["t"].flatten()
x = data["x"].flatten()

### use the below if the data is not normalized already
min_t = t.min()
max_t = t.max()

t_std = (t - t.min(axis=0)) / (t.max(axis=0) - t.min(axis=0))
t_norm = t_std * (1 - (-0)) -0
t_scaling = 1/(t.max(axis=0) - t.min(axis=0))


x_std = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
x_norm = x_std * (1 - (-1)) -1
x_scaling = 2/(x.max(axis=0) - x.min(axis=0))

#print(x_norm)
#print(t_norm)
print(u_data.size)

########################################
########################################

t_1 = t_norm.flatten()[0:101, None]
t_2 = t_norm.flatten()[101:201, None]

Exact = np.real(data["usol"]) + 0.00 * np.random.rand(512,201)
#Exact = np.real(data["U_exact"])

X_1, T_1 = np.meshgrid(x_norm, t_1, indexing="ij")
Exact_1 = Exact[:,0:X_1.shape[1]]

X_2, T_2 = np.meshgrid(x_norm, t_2, indexing="ij")
Exact_2 = Exact[:,X_2.shape[1]+1:]


##############################
############################## X, n_d, t_o, x_o, tolerance = 1e-5,num_basis=20

t_o_1 = t_norm[0:101]
x_o_1 = x_norm
prec = 1
N_u = int(prec * len(Exact.flatten()))
deim_instance_1 = DEIM(Exact_1, 2, t_o_1, x_o_1, tolerance = 0.00011, num_basis= 1)
S_s, T_s, U_s_1 = deim_instance_1.execute()
coords_1 = deim_instance_1.coords
N_u = int(prec * len(U_s_1.flatten()))
X_star_1 = coords_1
u_star_1 = U_s_1
idx_1 = np.random.choice(X_star_1.shape[0], N_u, replace=False)
X_str_train_1 = X_star_1
u_str_train_1 = u_star_1
X_f_1 = np.hstack((X_star_1, u_star_1))
idx_f1 = np.random.choice(X_f_1.shape[0], N_u, replace=False)
#X_str_train_1, X_test, u_str_train_1, y_test = train_test_split(X_star_1, 
#                                                          u_star_1, test_size=0.98, random_state=42)
X_str_train_1 = X_star_1[idx_1,:]
u_str_train_1 = u_star_1[idx_1,:]
#X_str_train_f1, X_test, u_str_train_f1, y_test = train_test_split(X_star_1, 
#                                                          u_star_1, test_size=0.98, random_state=42)
X_str_train_f1 = X_star_1[idx_f1,:]
u_str_train_f1 = u_star_1[idx_f1,:]

####################################
#################################### deim_instance_2 = DEIM(Exact_2, 2, t_o_2, x_o_2, dec_rate=0, num_basis= 20)


t_o_2 = t_norm[101:]
x_o_2 = x_norm

deim_instance_2 = DEIM(Exact_2, 2, t_o_2, x_o_2, tolerance = 0.00011, num_basis= 1)
S_s, T_s, U_s_2 = deim_instance_2.execute()
coords_2 = deim_instance_2.coords
N_u = int(prec * len(U_s_2.flatten()))

X_f_2 = np.hstack((coords_2, U_s_2))
idx_f2 = np.random.choice(X_f_2.shape[0], N_u, replace=False)

#X_str_train_f1, X_test, u_str_train_f1, y_test = train_test_split(X_star_1, 
#                                                          u_star_1, test_size=0.98, random_state=42)

X_star_2 = coords_2
u_star_2 = U_s_2
X_str_train_2 = X_star_2[idx_f2,:]
u_str_train_2 = u_star_2[idx_f2,:]

X_str_train_f2 = X_star_2[idx_f2,:]
u_str_train_f2 = u_star_2[idx_f2,:]







#######################################
#######################################
fig, ax = plt.subplots()
im = ax.scatter(X_str_train_1[:,1], X_str_train_1[:,0], c=u_str_train_1[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)
plt.show()


fig, ax = plt.subplots()
im = ax.scatter(X_str_train_f1[:,1], X_str_train_f1[:,0], c=u_str_train_f1[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)
plt.show()


fig, ax = plt.subplots()
im = ax.scatter(X_str_train_2[:,1], X_str_train_2[:,0], c=u_str_train_2[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)
plt.show()


fig, ax = plt.subplots()
im = ax.scatter(X_str_train_f2[:,1], X_str_train_f2[:,0], c=u_str_train_f2[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)
plt.show()


fig_5, ax_5 = plt.subplots(2, 2, figsize=(8, 6), sharey=True, sharex=True)

scatter1 = ax_5[1, 1].scatter(X_str_train_f2[:, 1], X_str_train_f2[:, 0], c=u_str_train_f2[:, 0], marker="x", s=10)
scatter2 = ax_5[1, 0].scatter(X_str_train_2[:, 1], X_str_train_2[:, 0], c=u_str_train_2[:, 0], marker="o", s=10)
scatter3 = ax_5[0, 1].scatter(X_str_train_f1[:, 1], X_str_train_f1[:, 0], c=u_str_train_f1[:, 0], marker="^", s=10)
scatter4 = ax_5[0, 0].scatter(X_str_train_1[:, 1], X_str_train_1[:, 0], c=u_str_train_1[:, 0], marker="D", s=10)

ax_5[1, 0].set_xlabel('t')
ax_5[1, 0].set_ylabel('x')
ax_5[1, 1].set_xlabel('t')
ax_5[0, 0].set_ylabel('x')

#fig_5.colorbar(scatter1, ax=ax.ravel().tolist(), location='right', label='Colorbar Label')
#fig_5.colorbar.ax.set_aspect(8) 
#plt.tight_layout()
#plt.show()

#colorbar = fig_5.colorbar(scatter1, ax=ax.ravel().tolist(), location='right', aspect=50, fraction=0.01)
fig_5.suptitle('Different Data set')

plt.savefig("different_data"+".png")
plt.savefig("different_data"+".pdf")
plt.show()

###########################
###########################


in_features = 2
out_features = 1

hidden_features= 128
hidden_layers= 3

hidden_features_str = 128
hidden_layers_str = 3


###############################
###############################

noise=0

learning_rate_inr = 1e-5
hidden_features_str = 128
hidden_features = 128
hidden_layers = 3
num_epochs = 1000
noise = 0
#prec = 0.05


string_f = (
    "_LSfit_lr_"
    + str(learning_rate_inr)
    + "_hf_phi_"
    + str(hidden_features)
    + "_layers_"
    + str(int(hidden_layers))
    + str(int(hidden_features_str))
    + "_ep_"
    + str(int(num_epochs))
    + "_noise_"
    + str(int(100 * noise))
    + "_perc_"
    + str(int(100 * prec))
)

### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "./result_xpinn_kdv_1d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.makedirs(result_path)


## File saving parameters
filename_u = (
    result_path + "/" + "U_data" + string_f + ".npy"
)  ## path where the primary variable phi data is saved
filename_mu = (
    result_path + "/" + "mu_pred" + string_f + ".npy"
)  ## path where the auxilliary variable mu data is saved
filename_l_s1 = (
    result_path + "/" + "Loss_collect_s1" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved
filename_l_s2 = (
    result_path + "/" + "Loss_collect_s2" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved

filename_l_f1 = (
    result_path + "/" + "Loss_collect_f1" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved

filename_l_f2 = (
    result_path + "/" + "Loss_collect_f2" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved

filename_l_coef_s1 = (
    result_path + "/" + "coef_s_tr1" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved

filename_l_coef_f1 = (
    result_path + "/" + "coef_f_tr1" + string_f + ".npy"
)  ## path where t


filename_l_coef_s2 = (
    result_path + "/" + "coef_s_tr2" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved

filename_l_coef_f2 = (
    result_path + "/" + "coef_f_tr2" + string_f + ".npy"
)  ## path where t


filename_p = (
    result_path + "/" + "params_collect" + string_f + ".npy"
)  ## path where the parameter data for every epoch is saved
filename_model_u = (
    result_path + "/" + "model_u" + string_f + ".pt"
)  ## path where the primary SIREN network data is saved
filename_model_mu = (
    result_path + "/" + "model_mu" + string_f + ".pt"
)  ## path where the auxilliary SIREN netowrk data is saved

pltname_u = (
    result_path + "/" + "SOC" + string_f
)  ## path where the plots of primary variable phi is saved
pltname_l = (
    result_path + "/" + "LP" + string_f + ".png"
)  ## path where loss plot is saved

pltname_p1 = (
    result_path + "/" + "PP_1" + string_f + ".png"
)  ## pltname_p1 and pltname_p2 are paths where the plots of learned parameters are saved
pltname_p2 = result_path + "/" + "PP_2" + string_f + ".png"

filename_data = (
    result_path + "/" + "different_data" + string_f + ".png"
)  ## path where the parameter data for every epoch is saved



##############################
##############################


# siren model initialization
model_str_1 = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features= hidden_features_str,
    hidden_layers= hidden_layers_str,
    outermost_linear=True,
).to(device)


model_str_f1 = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features= hidden_features,
    hidden_layers= hidden_layers,
    outermost_linear=True,
).to(device)


model_str_2 = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features= hidden_features_str,
    hidden_layers= hidden_layers_str,
    outermost_linear=True,
).to(device)


model_str_f2 = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features= hidden_features,
    hidden_layers= hidden_layers,
    outermost_linear=True,
).to(device)



learning_rate_inr = 1e-5

# optimizer
optim_adam = torch.optim.Adam(
    [
        {
            "params": model_str_1.parameters(),
            "lr": learning_rate_inr,
            "weight_decay": 0,
        },
        {"params": model_str_f1.parameters(),
         "lr": learning_rate_inr,
         "weight_decay": 0,
         },
        
        {
            "params": model_str_2.parameters(),
            "lr": learning_rate_inr,
            "weight_decay": 0,
        },
        
        {
            "params": model_str_f2.parameters(),
            "lr": learning_rate_inr,
            "weight_decay": 0,
        },
        
    ]
)


### learnig scheduler cyclic damping
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optim_adam,
    base_lr= 1 * learning_rate_inr,
    max_lr= 1 * learning_rate_inr,
    cycle_momentum = False,
    mode="exp_range",
    step_size_up=1000,
)


# converting numpy to torch
X_t_1 = torch.tensor(X_str_train_1, requires_grad=True).float().to(device)
Y_t_1 = torch.tensor(u_str_train_1).float().to(device)


X_t_f1 = torch.tensor(X_str_train_f1, requires_grad=True).float().to(device)
Y_t_f1 = torch.tensor(u_str_train_f1).float().to(device)


X_t_2 = torch.tensor(X_str_train_2, requires_grad=True).float().to(device)
Y_t_2 = torch.tensor(u_str_train_2).float().to(device)


X_t_f2 = torch.tensor(X_str_train_f2, requires_grad=True).float().to(device)
Y_t_f2 = torch.tensor(u_str_train_f2).float().to(device)


batch_size_1 = X_t_1.shape[0]
batch_size_f1 = X_t_f1.shape[0]


batch_size_2 = X_t_2.shape[0]
batch_size_f2 = X_t_f2.shape[0]


# dataloader
print("now starting loading data")
train_loader_1 = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t_1, Y_t_1), batch_size=batch_size_1, shuffle=True
)

train_loader_f1 = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t_f1, Y_t_f1), batch_size=batch_size_f1, shuffle=True
)

train_loader_2 = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t_2, Y_t_2), batch_size=batch_size_2, shuffle=True
)

train_loader_f2 = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t_f2, Y_t_f2), batch_size=batch_size_f2, shuffle=True
)



data_loaders = [train_loader_1, train_loader_f1, train_loader_2, train_loader_f2]



################################
################################




def features_calc(X, u, t_scale, x_scale):
    
    #print(t_scale)
    
    u_t = t_scale * torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_x = x_scale * torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = x_scale * torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xxx = x_scale * torch.autograd.grad(
        u_xx, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    uu_x = u.multiply(u_x).reshape(-1,1) 
    
    #u_t = u_t/u_t.norm()
    #uu_x = uu_x/uu_x.norm()
    #u_xxx =  u_xxx/u_xxx.norm()
    
    
    features = torch.cat([u_t, uu_x, u_xxx], dim=1)
    
    
    
    return features


### calculation of dictionary theta


def calculate_theta(features, u_pred):
    
    thetas = torch.cat([features[:, 1:2], features[:, 2:3] ], dim=1)

    return thetas


def weight_init(m):
    if isinstance(m, nn.Linear):
        #init.xavier_uniform_(m.weight)
        init.xavier_normal_(m.weight)

        

def custom_normalize(feature):
        """minmax all features by their absolute maximum
        Args:
            feature (torch.tensor): data to be minmax normalized
        Returns:
            (torch.tensor): minmaxed data"""
        return (feature/feature.abs().max(axis=0).values)


"""
model_str_1.apply(weight_init)
model_str_f1.apply(weight_init)
model_str_2.apply(weight_init)
model_str_f2.apply(weight_init)
model_int.apply(weight_init)
"""




#num_epochs = 1000

loss_total = []

loss_tot_s1 = []
loss_tot_f1 = []
loss_tot_s2 = []
loss_tot_f2 = []
loss_tot_int = []


coef_s_1 = []
coef_f_1 = []

coef_s_2 = []
coef_f_2 = []

for epoch in range(num_epochs):
    # print(f'epoch {epoch}')
    loss_epoch = 0
    loss_data_epoch = 0
    loss_eq_epoch = 0
    loss_mu_epoch = 0

    ii = 0
    
    print(epoch)
    print("\n")
    
    start_time = time.time()
    
    loss_data_1 = 0
    for batch_idx_1, (input_1, out_1) in enumerate(train_loader_1):
        
        u_pred_1 = model_str_1(input_1)
        
        ## features calculation
        features_1 = features_calc(input_1, u_pred_1, t_scaling, x_scaling)
        time_derivs_1 = features_1[:, 0:1]

        ### calculation of dictionary theta
        thetas_1 = calculate_theta(features_1, u_pred_1)
        
        ### least squares fit of the coefficients
        coefficients_1 = leastsquares_fit(thetas_1, time_derivs_1)

        ### equation residual
        residual_1 = equation_residual_AC(thetas_1, time_derivs_1, coefficients_1)

        ### loss evaluation
        #loss_data_1 = torch.mean((out_1 - u_pred_1) ** 2)
        
        loss_data_b1, loss_data, loss_eq = loss_func_AC(out_1, u_pred_1, residual_1)
        
        loss_data_1 = loss_data_1 + loss_data_b1
        
        coef_s_1.append(coefficients_1.cpu().detach().numpy())
        
    loss_f1 = 0
    
    for batch_idx_f1, (input_f1, out_f1) in enumerate(train_loader_f1):
        
        u_pred_f1 = model_str_f1(input_f1)
    
        features_f1 = features_calc(input_f1, u_pred_f1, t_scaling, x_scaling)

        time_derivs_f1 = features_f1[:, 0:1]

        ### calculation of dictionary theta
        thetas_f1 = calculate_theta(features_f1, u_pred_f1)
        
        ### least squares fit of the coefficients
        coefficients_f1 = leastsquares_fit(thetas_f1, time_derivs_f1)
        
        residual_f1 = equation_residual_AC(thetas_f1, time_derivs_f1, coefficients_f1)
        loss_bf1, loss_data, loss_eq = loss_func_AC(out_f1, u_pred_f1, residual_f1)
        
        loss_f1 = loss_bf1 + loss_f1
        coef_f_1.append(coefficients_f1.cpu().detach().numpy())
    
    loss_data_2 = 0
    for batch_idx_2, (input_2, out_2) in enumerate(train_loader_2):
        
        u_pred_2 = model_str_2(input_2)
        
        ## features calculation
        features_2 = features_calc(input_2, u_pred_2, t_scaling, x_scaling)
        time_derivs_2 = features_2[:, 0:1]

        ### calculation of dictionary theta
        thetas_2 = calculate_theta(features_2, u_pred_2)
        
        ### least squares fit of the coefficients
        coefficients_2 = leastsquares_fit(thetas_2, time_derivs_2)

        ### equation residual
        residual_2 = equation_residual_AC(thetas_2, time_derivs_2, coefficients_2)

        ### loss evaluation
        #loss_data_2 = torch.mean((out_2 - u_pred_2) ** 2)
        
        loss_data_b2, loss_data, loss_eq = loss_func_AC(out_2, u_pred_2, residual_2)
        loss_data_2 = loss_data_b2 + loss_data_2
        coef_s_2.append(coefficients_2.cpu().detach().numpy())
    
        
    loss_f2=0
    for batch_idx_f2, (input_f2, out_f2) in enumerate(train_loader_f2):
        
        u_pred_f2 = model_str_f2(input_f2)
    
        features_f2 = features_calc(input_f2, u_pred_f2, t_scaling, x_scaling)

        time_derivs_f2 = features_f2[:, 0:1]

        ### calculation of dictionary theta
        thetas_f2 = calculate_theta(features_f2, u_pred_f2)
        
        ### least squares fit of the coefficients
        coefficients_f2 = leastsquares_fit(thetas_f2, time_derivs_f2)
        
        residual_f2 = equation_residual_AC(thetas_f2, time_derivs_f2, coefficients_f2)
        loss_bf2, loss_data, loss_eq = loss_func_AC(out_f2, u_pred_f2, residual_f2)
        
        loss_f2 = loss_bf2 + loss_f2
        
        coef_f_2.append(coefficients_f2.cpu().detach().numpy())
        
    
    
    loss_int_total = 0
    loss_int = 0
    loss_int_s1 = 0
    loss_int_f1 = 0 
    loss_int_s2 = 0 
    loss_int_f2 = 0
        
    loss = 1*loss_data_1 + 1*loss_data_2 + 1*loss_f2 + 1*loss_f1 
    #loss = loss_data_1 + loss_data_2 + loss_f2 + loss_f1 + loss_int_total

    
    loss_total.append(loss.cpu().detach().numpy())
    loss_tot_s1.append(loss_data_1.cpu().detach().numpy())
    loss_tot_f1.append(loss_f1.cpu().detach().numpy())
    loss_tot_s2.append(loss_data_2.cpu().detach().numpy())
    loss_tot_f2.append(loss_f2.cpu().detach().numpy())
    #loss_tot_int.append(loss_int_total.cpu().detach().numpy())
    
    
    
    
    optim_adam.zero_grad()
    loss.backward()
    optim_adam.step()

    # scheduler step
    scheduler.step()


loss_total = np.array(loss_total)
loss_tot_s1 = np.array(loss_tot_s1)
loss_tot_f1 = np.array(loss_tot_f1)
loss_tot_s2 =  np.array(loss_tot_s2)
loss_tot_f2 = np.array(loss_tot_f2)
loss_tot_int = np.array(loss_tot_int)


##########

fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))

axs2.semilogy(loss_tot_s1)
axs2.semilogy(loss_tot_f1)
axs2.semilogy(loss_tot_s2)
axs2.semilogy(loss_tot_f2)





axs2.set(xlabel='Iteration')
plt.title("loss functions")

plt.tight_layout()
plt.savefig(pltname_p2 +'loss_function'+'.png', dpi=300,)
plt.savefig(pltname_p2 +'loss_function'+'.pdf', dpi=300,)


######################################
######################################
# Specify the directory path where the array is stored

# Load the array from the directory
#loss_tot_s1 = np.load(result_path + "loss_tot_s1.npy")


axs2.set(xlabel='Iteration')
plt.title("loss functions")

coef_s_tr1=np.reshape(np.array(coef_s_1),(num_epochs,-1))
coef_f_tr1=np.reshape(np.array(coef_f_1),(num_epochs,-1))

coef_s_tr2=np.reshape(np.array(coef_s_2),(num_epochs,-1))
coef_f_tr2=np.reshape(np.array(coef_f_2),(num_epochs,-1))



np.save(filename_l_coef_s1, coef_s_tr1)
np.save(filename_l_coef_f1, coef_f_tr1)
np.save(filename_l_coef_s2, coef_s_tr2)
np.save(filename_l_coef_f2, coef_f_tr2)




plt.rcParams.update({
    "font.size": 20,
})

line_width=3

fig, axs = plt.subplots(1, 3, sharey=False, sharex=True, figsize=(15, 4))


axs[0].plot(coef_s_tr1[:,0], linewidth=line_width)
axs[0].plot(coef_f_tr1[:,0], linewidth=line_width)
axs[0].set(ylabel='Coefficients')
axs[0].set(xlabel='Iteration')

axs[0].plot(coef_s_tr2[:,0],linewidth=line_width)
axs[0].plot(coef_f_tr2[:,0],linewidth=line_width)


axs[1].plot(coef_s_tr1[:,1],linewidth=line_width)
axs[1].plot(coef_f_tr1[:,1],linewidth=line_width)
axs[1].plot(coef_s_tr2[:,1],linewidth=line_width)
axs[1].plot(coef_f_tr2[:,1],linewidth=line_width)
axs[1].set(ylabel='Coefficients')
axs[1].set(xlabel='Iteration')



axs[2].semilogy(loss_tot_s1)
axs[2].semilogy(loss_tot_f1)
axs[2].semilogy(loss_tot_s2)
axs[2].semilogy(loss_tot_f2)
axs[2].set(ylabel='loss')
axs[2].set(xlabel='Iteration')


fig.suptitle(r'\texttt{DEIM-XPINN}')

"""
axs[2].semilogy(loss_tot_s1)
axs[2].semilogy(loss_tot_f1)
axs[2].semilogy(loss_tot_s2)
axs[2].semilogy(loss_tot_f2)
axs[2].set(ylabel='loss')
axs[2].set(xlabel='Iteration')
"""

#plt.suptitle(r'\texttt{E-PINN}')
plt.tight_layout()
plt.savefig(pltname_p2 +'coefficients'+'.png', dpi=300,)
plt.savefig(pltname_p2 +'coefficients'+'.pdf', dpi=300,)





#plt.suptitle(r'\texttt{E-PINN}')
plt.tight_layout()
#plt.savefig(pltname_p2 +'loss_function'+'.png', dpi=300,)
#plt.savefig(pltname_p2 +'loss_function'+'.pdf', dpi=300,)
####################################
####################################


fig_5, ax_5 = plt.subplots(2, 2, figsize=(8, 6), sharey=True, sharex=True)


scatter1 = ax_5[1, 1].scatter(X_str_train_f2[:, 1], X_str_train_f2[:, 0], c=u_str_train_f2[:, 0], marker="x", s=10)
scatter2 = ax_5[1, 0].scatter(X_str_train_2[:, 1], X_str_train_2[:, 0], c=u_str_train_2[:, 0], marker="o", s=10)
scatter3 = ax_5[0, 1].scatter(X_str_train_f1[:, 1], X_str_train_f1[:, 0], c=u_str_train_f1[:, 0], marker="^", s=10)
scatter4 = ax_5[0, 0].scatter(X_str_train_1[:, 1], X_str_train_1[:, 0], c=u_str_train_1[:, 0], marker="D", s=10)

ax_5[1, 0].set_xlabel('t')
ax_5[1, 0].set_ylabel('x')
ax_5[1, 1].set_xlabel('t')
ax_5[0, 0].set_ylabel('x')

#fig_5.colorbar(scatter1, ax=ax.ravel().tolist(), location='right', label='Colorbar Label')
#fig_5.colorbar.ax.set_aspect(8) 
#plt.tight_layout()
#plt.show()

colorbar = fig_5.colorbar(scatter1, ax=ax_5.ravel().tolist(), location='right', aspect=50, fraction=0.01)
fig_5.suptitle('Different Data set')

#plt.savefig(filename_data + "different_data" + ".png")
#plt.savefig(filename_data + "different_data" + ".pdf", dpi=300,)


plt.savefig(filename_data + "different_data" + ".png", dpi=300,)
plt.savefig(filename_data + "different_data" + ".pdf", dpi=300,)
plt.show()


################################
################################

fig, axs = plt.subplots(1, 2, sharey=False, figsize=(8, 3.5))

axs[0].plot(coef_s_tr1[:,0], linewidth=line_width)
axs[0].plot(coef_f_tr1[:,0], linewidth=line_width)
axs[0].set(ylabel='Coefficients')
axs[0].set(xlabel='Iteration')

axs[0].plot(coef_s_tr2[:,0],linewidth=line_width)
axs[0].plot(coef_f_tr2[:,0],linewidth=line_width)

axs[1].plot(coef_s_tr1[:,1],linewidth=line_width)
axs[1].plot(coef_f_tr1[:,1],linewidth=line_width)
axs[1].plot(coef_s_tr2[:,1],linewidth=line_width)
axs[1].plot(coef_f_tr2[:,1],linewidth=line_width)
axs[1].set(ylabel='Coefficients')
axs[1].set(xlabel='Iteration')

axs[0].set_ylim(-8, 5.2)

axs[0].set_yticks(np.arange(-6, 5, 4))

fig.suptitle(r"Ensembled \texttt{DEIM-XPINN}", fontsize=18)

#plt.suptitle(r'\texttt{E-PINN}')
plt.tight_layout()

print(coefficients_f2)
print(coefficients_f1)
print(coefficients_2)
print(coefficients_1)

plt.savefig(filename_data + "coefficients" + ".png", dpi=600,)
plt.savefig(filename_data + "coefficients" + ".pdf", dpi=600,)
plt.show()


########################
########################




########################
########################



X_T, T_T = np.meshgrid(x_norm, t_norm, indexing= "ij")
Exact_T = Exact[:,0:X_1.shape[1]]


prec = 0.01
N_u_ran = int(prec * len(Exact_T.flatten()))


X_ran = np.hstack((X_1.flatten()[:, None], T_1.flatten()[:, None]))
u_ran = Exact_1.flatten()[:, None]
idx_ran = np.random.choice(X_ran.shape[0], N_u_ran, replace= False)


X_ran_train_1 = X_ran[idx_ran,:]
u_ran_train_1 = u_ran[idx_ran,:]




#mask = np.ones_like(X_star_2[:,0], dtype=bool)
#mask[idx_f2] = False
#X_str_mask = X_star_2[mask]
#print(X_str_test_2)
#u_str_mask = u_star_2[mask]

#idx_mask = np.random.choice(X_str_mask.shape[0], 1000, replace=False)

X_str_test = X_ran[idx_ran,:]
u_str_test = u_ran[idx_ran,:]

X_str_test_torch = torch.tensor(X_str_test).float().to(device)
u_str_test_torch = torch.tensor(u_str_test).float().to(device)

u_pred_test_torch = model_str_2(X_str_test_torch )

error = (u_str_test_torch - u_pred_test_torch).cpu().detach().numpy()

plt.rcParams.update({
    "font.size": 20,
})
fig_6, ax_6 = plt.subplots(1, 3, figsize=(15, 4.5),  sharey= True, sharex=True)

scatter1 = ax_6[0].scatter(X_str_test[:, 1], X_str_test[:, 0], c=u_str_test[:, 0], marker="x", s=10)
scatter2 = ax_6[1].scatter(X_str_test[:, 1], X_str_test[:, 0], c=u_pred_test_torch.cpu().detach().numpy()[:, 0], marker="o", s=10)
scatter3 = ax_6[2].scatter(X_str_test[:, 1], X_str_test[:, 0], c=np.abs(error[:, 0]), marker="*", s=10)

import matplotlib.cm as cm
norm = plt.Normalize(vmin=min(u_str_test[:, 0]), vmax=max(u_str_test[:, 0]))
cmap = cm.get_cmap('viridis')
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) 

norm_e = plt.Normalize(vmin=min(np.abs(error[:, 0])), vmax=max(np.abs(error[:, 0])))
cmap = cm.get_cmap('viridis')
sm_e = cm.ScalarMappable(cmap=cmap, norm=norm_e)
sm_e.set_array([]) 

#colorbar = fig_6.colorbar(scatter1, ax=ax_5.ravel().tolist(), location='right', aspect=50, fraction=0.01)
import matplotlib.ticker as ticker
cbar1 = fig_6.colorbar(mappable = scatter1, ax=ax_6[0], ticks=ticker.MultipleLocator(0.2))
cbar2 = fig_6.colorbar(mappable = scatter2, ax=ax_6[1],ticks=ticker.MultipleLocator(0.2))
cbar3 = fig_6.colorbar(mappable = scatter3, ax=ax_6[2], ticks=ticker.MultipleLocator(0.1))

ax_6[0].set_title("Test data set")
ax_6[1].set_title("Prediction")
ax_6[2].set_title("Absoluet Error")
ax_6[0].set_xlabel("t", fontsize=20)
ax_6[1].set_xlabel("t", fontsize=20)
ax_6[2].set_xlabel("t", fontsize=20)


ax_6[0].set_ylabel("x", fontsize=20)
plt.tight_layout()


plt.savefig(filename_data + "prediction_error" + ".png")
plt.savefig(filename_data + "prediction_error" + ".pdf", dpi=300,)
plt.show()



#
#xticks_step = 0.2
#for ax in ax_6:
#    ax.set_xticks(np.arange(min(X_str_test[:, 1]), max(X_str_test[:, 1]) , xticks_step))



#fig_6.colorbar()

#fig_7, ax_7 = plt.subplots(1, 1, figsize=(10, 4),  sharey= True, sharex=True)

#scatter4 = ax_7.scatter(X_str_test[:, 1], X_str_test[:, 0], c=np.abs(error[:, 0]), marker="*", s=10)

#cbar4 = fig_7.colorbar(mappable = scatter4, ax=ax_7, format=ticker.ScalarFormatter(useMathText=True))

#for ax in ax_7:
#    ax.set_xticks(np.arange(min(X_str_test[:, 1]), max(X_str_test[:, 1]) + xticks_step, xticks_step))

#fig_7.colorbar()


#plt.show()


