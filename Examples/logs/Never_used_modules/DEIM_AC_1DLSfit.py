
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:42:41 2023

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
import warnings
import time
from PINN_DEIM.Functions.DEIM_class import DEIM


cwd = os.getcwd()
sys.path.append(cwd)

warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


#########################################


def x_t_normalization(t,x):
        
    min_t = t.min()
    max_t = t.max()
    
    t_std = (t - t.min(axis=0)) / (t.max(axis=0) - t.min(axis=0))
    t_norm = t_std * (1 - (-0)) -0
    t_scaling = 1/(t.max(axis=0) - t.min(axis=0))
    
    x_std = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    x_norm = x_std * (1 - (-1)) -1
    x_scaling = 2/(x.max(axis=0) - x.min(axis=0))
    
    return t_norm, x_norm



##########################################

data = scipy.io.loadmat(root_dir+'/data/AC.mat')

t = data["tt"].flatten()[0:201, None]
t_2 = data["tt"].flatten()[0:201, None]
x = data["x"].flatten()[:, None]
Exact = np.real(data["uu"])



t_norm, x_norm = x_t_normalization(t,x)






#############################

t_o = data["tt"].flatten()[0:201, None]
x_o = data["x"].flatten()[:, None]


t_o, x_o = x_t_normalization(t_o,x_o)


#############################
#############################



X_1, T_1 = np.meshgrid(x_norm, t_norm, indexing= "ij")
Exact_1 = Exact[:,0:X_1.shape[1]]


prec = 0.00
N_u_ran = int(prec * len(Exact.flatten()))


X_ran = np.hstack((X_1.flatten()[:, None], T_1.flatten()[:, None]))
u_ran = Exact_1.flatten()[:, None]
idx_ran = np.random.choice(X_ran.shape[0], N_u_ran, replace= False)


X_ran_train_1 = X_ran[idx_ran,:]
u_ran_train_1 = u_ran[idx_ran,:]


#############################
#############################


prec = 1

deim_instance_1 = DEIM(Exact, 10, t_o, x_o, dec_rate=0, num_basis= 10)
S_s, T_s, U_s_1 = deim_instance_1.execute()
coords_1 = deim_instance_1.coords

N_u_DEIM = int(prec * len(U_s_1.flatten()))


X_star_1 = coords_1
u_star_1 = U_s_1
idx_1 = np.random.choice(X_star_1.shape[0], N_u_DEIM, replace=False)


X_str_train_1 = np.vstack((X_star_1[idx_1],X_ran_train_1))
u_str_train_1 = np.vstack((u_star_1[idx_1], u_ran_train_1))


###########################################
###########################################

#X_str_train_1 = X_ran_train_1
#u_str_train_1 = u_ran_train_1


#######################################
#######################################


fig_2, ax_2 = plt.subplots()
im = ax_2.scatter(X_str_train_1[:,1], X_str_train_1[:,0], c=u_str_train_1[:,0], marker="x", s=10)
ax_2.set_xlabel('t')
ax_2.set_ylabel('x')
fig_2.colorbar(mappable=im)
plt.show()




#######################################
#######################################


#######################################
#######################################

in_features = 2
out_features = 1

hidden_features= 128
hidden_layers= 3

hidden_features_str = 128
hidden_layers_str = 3

# siren model initialization
model_str_1 = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features= hidden_features_str,
    hidden_layers= hidden_layers_str,
    outermost_linear=True,
).to(device)




"""
model_int = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features=64,
    hidden_layers=4,
    outermost_linear=True,
).to(device)
"""





learning_rate_inr = 1e-5

# optimizer
optim_adam = torch.optim.Adam(
    [
        {
            "params": model_str_1.parameters(),
            "lr": learning_rate_inr,
            "weight_decay": 1e-6,
        },
    ]
)


### learnig scheduler cyclic damping
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optim_adam,
    base_lr=0.1 * learning_rate_inr,
    max_lr=10 * learning_rate_inr,
    cycle_momentum=False,
    mode="exp_range",
    step_size_up=1000,
)


 

# converting numpy to torch
X_t_1 = torch.tensor(X_str_train_1, requires_grad=True).float().to(device)
Y_t_1 = torch.tensor(u_str_train_1).float().to(device)



batch_size_1 = X_t_1.shape[0]

# dataloader
print("now starting loading data")
train_loader_1 = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t_1, Y_t_1), batch_size=batch_size_1, shuffle=True
)

t

data_loaders = [train_loader_1]


#######################################
#######################################



def features_calc(X, u):
    u_t = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_x = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]

    features = torch.cat([u_t, u_xx], dim=1)
    return features


### calculation of dictionary theta


def calculate_theta(features, u_pred):
    thetas = torch.cat([features[:, 1:2], (torch.pow((u_pred), 3) - u_pred)], dim=1)

    return thetas

def weight_init(m):
    if isinstance(m, nn.Linear):
        #init.xavier_uniform_(m.weight)
        init.xavier_normal_(m.weight)

# Apply weight initialization to the model
#model_str_1.apply(weight_init)

"""
model_str_1.apply(weight_init)
model_str_f1.apply(weight_init)
model_str_2.apply(weight_init)
model_str_f2.apply(weight_init)
model_int.apply(weight_init)
"""

#######################################
#######################################


num_epochs = 2000

loss_total = []
loss_tot_s1 = []
coef_s_1 = []



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
        features_1 = features_calc(input_1, u_pred_1)
        time_derivs_1 = features_1[:, 0:1]

        ### calculation of dictionary theta
        thetas_1 = calculate_theta(features_1, u_pred_1)
        
        ### least squares fit of the coefficients
        coefficients_1 = leastsquares_fit(thetas_1, time_derivs_1)

        ### equation residual
        residual_1 = equation_residual_AC(thetas_1, time_derivs_1, coefficients_1)

        ### loss evaluation
        loss_data_1 = torch.mean((out_1 - u_pred_1) ** 2)
        
        loss_data_b1, loss_data, loss_eq = loss_func_AC(out_1, u_pred_1, residual_1)
        
        loss_data_1 = loss_data_1 + loss_data_b1
        
        coef_s_1.append(coefficients_1.cpu().detach().numpy())
     
        
    
    
    loss_int_total = 0
    loss_int = 0
    loss_int_s1 = 0
    
        
    loss = 1*loss_data_1 
    #loss = loss_data_1 + loss_data_2 + loss_f2 + loss_f1 + loss_int_total

    
    loss_total.append(loss.cpu().detach().numpy())
    loss_tot_s1.append(loss_data_1.cpu().detach().numpy())
    
    
    
    
    
    optim_adam.zero_grad()
    loss.backward()
    optim_adam.step()

    # scheduler step
    scheduler.step()


loss_total = np.array(loss_total)
loss_tot_s1 = np.array(loss_tot_s1)




#############################################
#############################################
#############################################
noise=0


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
result_path = "./result_Epinn_AC_1d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.makedirs(result_path)

#os.makedirs(os.path.dirname(param_set.save_model_path), exist_ok=True)

    
    
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



################################
################################


fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))

axs2.semilogy(loss_tot_s1)

np.save(filename_l_s1, loss_tot_s1)

######################################
######################################
# Specify the directory path where the array is stored

# Load the array from the directory
#loss_tot_s1 = np.load(result_path + "loss_tot_s1.npy")




axs2.set(xlabel='Iteration')
plt.title("loss functions")


coef_s_tr1=np.reshape(np.array(coef_s_1),(num_epochs,-1))

np.save(filename_l_coef_s1, coef_s_tr1)



plt.rcParams.update({
    "font.size": 15,
})

line_width=3


fig, axs = plt.subplots(1, 3, sharey=False, figsize=(12, 3))
axs[0].plot(coef_s_tr1[:,0], linewidth=line_width)


axs[1].plot(coef_s_tr1[:,1],linewidth=line_width)

axs[1].set_ylim([-5.2,1])

axs[2].semilogy(loss_tot_s1)

axs[2].set(ylabel='loss')
axs[2].set(xlabel='Iteration')

#plt.suptitle(r'\texttt{E-PINN}')
plt.tight_layout()
plt.savefig(pltname_p2 +'loss_function'+'.png', dpi=300,)
plt.savefig(pltname_p2 +'loss_function'+'.pdf', dpi=300,)
####################################
####################################




fig_5, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True, sharex=True)

scatter4 = ax.scatter(X_str_train_1[:, 1], X_str_train_1[:, 0], c=u_str_train_1[:, 0], marker="D", s=10)

ax.set_xlabel('t')
ax.set_ylabel('x')

colorbar = plt.colorbar(scatter4, ax=ax, location='right', aspect=50, fraction=0.01)
colorbar.set_label('Colorbar Label')

fig_5.suptitle('Different Data Set')


plt.savefig(filename_data + "different_data" + ".png")
plt.savefig(filename_data + "different_data" + ".pdf", dpi=300,)

plt.tight_layout()
plt.show()




