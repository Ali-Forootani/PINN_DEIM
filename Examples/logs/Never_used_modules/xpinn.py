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
from PINN_DEIM.Functions.utils import loss_func_AC, leastsquares_fit, equation_residual_AC
from PINN_DEIM.Functions.library import library_deriv
from PINN_DEIM.Functions import plot_config_file
from PINN_DEIM.Functions.ensemble_loader import LoaderProcessor, process_loader
from PINN_DEIM.Functions.library import features_calc_AC, calculate_theta_AC, library_deriv



from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time


warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

####################### loading the data set
#############################################

data = scipy.io.loadmat(root_dir +"/data/AC.mat")


t_1 = data["tt"].flatten()[0:101, None]
t_2 = data["tt"].flatten()[101:201, None]
x = data["x"].flatten()[:, None]
Exact = np.real(data["uu"])

X_1, T_1 = np.meshgrid(x, t_1, indexing="ij")
Exact_1 = Exact[:,0:X_1.shape[1]]


X_2, T_2 = np.meshgrid(x, t_2, indexing="ij")
Exact_2 = Exact[:,X_2.shape[1]+1:]



T_int = (T_1[:,-1] + T_2[:,0])/2
Exact_int = (Exact_1[:,-1] + Exact_2[:,0])/2
X_int = x.reshape(-1,1)
T_int = T_int.reshape(-1,1)
X_star_int = np.hstack((X_int, T_int))
u_int = Exact_int.reshape(-1,1)
X_int_f = np.hstack((X_star_int, u_int))



test_data_size = 0.98


X_star_1 = np.hstack((X_1.flatten()[:, None], T_1.flatten()[:, None]))
u_star_1 = Exact_1.flatten()[:, None]
X_f_1 = np.hstack((X_star_1, u_star_1))
X_str_train_1, X_test, u_str_train_1, u_test = train_test_split(X_star_1, 
                                                          u_star_1, test_size= test_data_size, random_state=42)
X_str_train_f1, X_test, u_str_train_f1, u_test = train_test_split(X_test, 
                                                          u_test, test_size= test_data_size, random_state=42)




X_star_2 = np.hstack((X_2.flatten()[:, None], T_2.flatten()[:, None]))
u_star_2 = Exact_2.flatten()[:, None]
X_f_2 = np.hstack((X_star_2, u_star_2))
X_str_train_2, X_test, u_str_train_2, u_test = train_test_split(X_star_2, 
                                                          u_star_2, test_size= test_data_size, random_state=42)
X_str_train_f2, X_test, u_str_train_f2, u_test = train_test_split(X_test, 
                                                          u_test, test_size= test_data_size, random_state=42)





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



fig, ax = plt.subplots(2,2, sharey=True, sharex=True, figsize=(10, 8))

ax[0,0].scatter(X_str_train_1[:,1], X_str_train_1[:,0], c=u_str_train_1[:,0], marker="x", s=10)
ax[1,0].scatter(X_str_train_f1[:,1], X_str_train_f1[:,0], c=u_str_train_f1[:,0], marker="x", s=10)


ax[0,1].scatter(X_str_train_2[:,1], X_str_train_2[:,0], c=u_str_train_2[:,0], marker="x", s=10)
ax[1,1].scatter(X_str_train_f2[:,1], X_str_train_f2[:,0], c=u_str_train_f2[:,0], marker="x", s=10)


plt.show()


########################################
########################################


noise=0
learning_rate_inr = 1e-5
hidden_features_str = 128
hidden_features = 64
hidden_layers = 3
num_epochs = 1500
noise = 0
prec = 0.05


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
result_path = "./result_xpinn_AC_1d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.makedirs(result_path)

#######################################
#######################################


in_features = 2
out_features = 1

# siren model initialization
model_str_1 = Siren(
    in_features,
    hidden_features_str,
    hidden_layers,
    out_features,
    outermost_linear=True,
).to(device)


model_str_f1 = Siren(
    in_features,
    hidden_features_str,
    hidden_layers,
    out_features,
    outermost_linear=True,
).to(device)


model_str_2 = Siren(
    in_features,
    hidden_features_str,
    hidden_layers,
    out_features,
    outermost_linear=True,
).to(device)


model_str_f2 = Siren(
    in_features,
    hidden_features_str,
    hidden_layers,
    out_features,
    outermost_linear=True,
).to(device)


###############################################
###############################################

# optimizer
optim_adam = torch.optim.Adam(
    [
        {
            "params": model_str_1.parameters(),
            "lr": learning_rate_inr,
            "weight_decay": 1e-6,
        },
        {"params": model_str_f1.parameters(),
         "lr": learning_rate_inr,
         "weight_decay": 1e-6,
         },
        
        {
            "params": model_str_2.parameters(),
            "lr": learning_rate_inr,
            "weight_decay": 1e-6,
        },
        
        {
            "params": model_str_f2.parameters(),
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


X_t_f1 = torch.tensor(X_str_train_f1, requires_grad=True).float().to(device)
Y_t_f1 = torch.tensor(u_str_train_f1).float().to(device)


X_t_2 = torch.tensor(X_str_train_2, requires_grad=True).float().to(device)
Y_t_2 = torch.tensor(u_str_train_2).float().to(device)


X_t_f2 = torch.tensor(X_str_train_f2, requires_grad=True).float().to(device)
Y_t_f2 = torch.tensor(u_str_train_f2).float().to(device)


X_t_int = torch.tensor(X_star_int, requires_grad=True).float().to(device)
Y_t_int = torch.tensor(u_int).float().to(device)


batch_size_1 = X_t_1.shape[0]
batch_size_f1 = X_t_f1.shape[0]


batch_size_2 = X_t_2.shape[0]
batch_size_f2 = X_t_f2.shape[0]

batch_size_int = X_t_int.shape[0]

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


########################################
########################################


loss_total = []

loss_tot_s1 = []
loss_tot_f1 = []
loss_tot_s2 = []
loss_tot_f2 = []


coef_s_1 = []
coef_f_1 = []
coef_s_2 = []
coef_f_2 = []


ensemble_s_1 = LoaderProcessor(model_str_1, features_calc_AC,
               calculate_theta_AC, loss_func_AC)

ensemble_f_1 = LoaderProcessor(model_str_f1, features_calc_AC,
               calculate_theta_AC, loss_func_AC)

ensemble_s_2 = LoaderProcessor(model_str_2, features_calc_AC,
               calculate_theta_AC, loss_func_AC)

ensemble_f_2 = LoaderProcessor(model_str_f2, features_calc_AC,
               calculate_theta_AC, loss_func_AC)



for epoch in range(num_epochs):
    # print(f'epoch {epoch}')
    loss_epoch = 0
    loss_data_epoch = 0
    loss_eq_epoch = 0
    loss_mu_epoch = 0

    ii = 0
    
    tqdm.write(f"Epoch: {epoch}")
    
    ####################################################
    
    start_time = time.time()
    
    loss_data_1 = 0
    loss_data_1, coefficients_1 = ensemble_s_1(train_loader_1)
    
    coef_s_1.append(coefficients_1.cpu().detach().numpy())
    
    ####################################################
    
    loss_f1 = 0
    loss_f1, coefficients_f1 = ensemble_f_1(train_loader_f1)
    
    coef_f_1.append(coefficients_f1.cpu().detach().numpy())
    
    ####################################################
    
    loss_data_2 = 0
    loss_data_2, coefficients_2 = ensemble_s_2(train_loader_2)
    
    coef_s_2.append(coefficients_2.cpu().detach().numpy())
    
    ####################################################
        
    loss_f2=0
    loss_f2, coefficients_f2 = ensemble_f_2(train_loader_f2)
    
    coef_f_2.append(coefficients_f2.cpu().detach().numpy())
    
    ####################################################
    
           
    loss = loss_data_1 + loss_data_2 + loss_f2 + loss_f1 

    
    loss_total.append(loss.cpu().detach().numpy())
    loss_tot_s1.append(loss_data_1.cpu().detach().numpy())
    loss_tot_f1.append(loss_f1.cpu().detach().numpy())
    loss_tot_s2.append(loss_data_2.cpu().detach().numpy())
    loss_tot_f2.append(loss_f2.cpu().detach().numpy())
    
            
    
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


#########  Reshaping the coefficients

plt.semilogy(loss_tot_s1)
plt.semilogy(loss_total)

coef_s_tr1=np.reshape(np.array(coef_s_1),(num_epochs,-1))
coef_f_tr1=np.reshape(np.array(coef_f_1),(num_epochs,-1))

coef_s_tr2=np.reshape(np.array(coef_s_2),(num_epochs,-1))
coef_f_tr2=np.reshape(np.array(coef_f_2),(num_epochs,-1))


###############################################
###############################################


filename_loss_funcs = (
    result_path + "/" + "Loss_functions" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved

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

filename_coef_s1 = (
    result_path + "/" + "coef_s_tr1" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved

filename_coef_f1 = (
    result_path + "/" + "coef_f_tr1" + string_f + ".npy"
)  ## path where t

filename_coef_s2 = (
    result_path + "/" + "coef_s_tr2" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved

filename_coef_f2 = (
    result_path + "/" + "coef_f_tr2" + string_f + ".npy"
)  ## path where t


filename_model_s1 = (
    result_path + "/" + "model_str_1" + string_f + ".pt"
)  ## path where the primary SIREN network data is saved

filename_model_f1 = (
    result_path + "/" + "model_str_f1" + string_f + ".pt"
)  ## path where the pr

filename_model_s2 = (
    result_path + "/" + "model_str_2" + string_f + ".pt"
)  ## path where the primary SIREN network data is saved

filename_model_f2 = (
    result_path + "/" + "model_str_f2" + string_f + ".pt"
)  ## path where the pr

filename_data = (
    result_path + "/" + "data_set" + string_f
)  ##

#################################################
##################### Plotting Loss Functions ###



plt.rcParams.update({
    "font.size": 20,
})

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))

x_ticks = [0, 500, 1000, 1500]

line_width = 3

axs2.semilogy(loss_tot_s1, marker="^", markersize=5, linewidth=line_width, markevery=100, linestyle='-')
axs2.semilogy(loss_tot_f1, marker="8", markersize=5, linewidth=line_width, markevery=100, linestyle='--')
axs2.semilogy(loss_tot_s2, marker="s", markersize=5, linewidth=line_width, markevery=100, linestyle=':')
axs2.semilogy(loss_tot_f2, marker="o", markersize=5, linewidth=line_width, markevery=100, linestyle='-.')

axs2.set_xticks(x_ticks)


axs2.set(xlabel='Iteration')
plt.title("Loss functions")

plt.tight_layout()
plt.savefig(filename_loss_funcs +'loss_function'+'.png', dpi=600,)
plt.savefig(filename_loss_funcs +'loss_function'+'.pdf', dpi=600,)


np.save(filename_l_s1, loss_tot_s1)
np.save(filename_l_f1, loss_tot_f1)
np.save(filename_l_s2, loss_tot_s2)
np.save(filename_l_f2, loss_tot_f2)



###################################
###################################


coef_s_tr1=np.reshape(np.array(coef_s_1),(num_epochs,-1))
coef_f_tr1=np.reshape(np.array(coef_f_1),(num_epochs,-1))

coef_s_tr2=np.reshape(np.array(coef_s_2),(num_epochs,-1))
coef_f_tr2=np.reshape(np.array(coef_f_2),(num_epochs,-1))


np.save(filename_coef_s1, coef_s_tr1)
np.save(filename_coef_f1, coef_f_tr1)
np.save(filename_coef_s2, coef_s_tr2)
np.save(filename_coef_f2, coef_f_tr2)


####################################
#################################### Plotting the convergence of coefficients

plt.rcParams.update({
    "font.size": 20,
})

line_width = 4

fig, axs = plt.subplots(2, 4, sharey=False, figsize=(13, 6))

axs[0,0].plot(coef_s_tr1[:,0], linewidth=line_width, linestyle='-', color= colors[0])
axs[0,1].plot(coef_f_tr1[:,0], linewidth=line_width, linestyle='--', color= colors[1])
axs[0,2].plot(coef_s_tr2[:,0],linewidth=line_width, linestyle=':', color= colors[2])
axs[0,3].plot(coef_f_tr2[:,0],linewidth=line_width, linestyle='-.', color= colors[3])

axs[1,0].plot(coef_s_tr1[:,1],linewidth=line_width, linestyle='-', color= colors[0], label=r"\texttt{Ensemb}$_1$-Subdom$_1$")
axs[1,1].plot(coef_f_tr1[:,1],linewidth=line_width, linestyle='--', color= colors[1], label=r"\texttt{Ensemb}$_2$-Subdom$_1$")
axs[1,2].plot(coef_s_tr2[:,1],linewidth=line_width, linestyle=':', color= colors[2], label=r"\texttt{Ensemb}$_1$-Subdom$_2$")
axs[1,3].plot(coef_f_tr2[:,1],linewidth=line_width, linestyle='-.', color= colors[3], label=r"\texttt{Ensemb}$_2$-Subdom$_2$")

plt.subplots_adjust(wspace=0.5, hspace=0.5)  

x_ticks = [0, 750, 1500]
y_ticks = [0, -1, -3, -5]

from matplotlib.ticker import FuncFormatter

# Define a custom formatting function for y-axis tick labels
def sci_fmt_1(x, pos):
    return "${:.2f}$".format(x)

def sci_fmt_2(x, pos):
    return "${:.0f}$".format(x)

# Apply the custom formatting function to the y-axis tick labels
#for ax in axs.flat:   
#    ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt_1))

for i in range (2):
    for j in range (4):
        if i == 0 and j <= 3 :
            axs[i,j].set_ylim([-0.03, 0.01])
            axs[i,j].set_xticks(x_ticks)
            
            axs[i,j].yaxis.set_major_formatter(FuncFormatter(sci_fmt_1))
            
            if j == 0 :
                axs[i,j].set(ylabel= r'Coefficients $u_{xx}$')        
        #if i == 0 and j > 1 :
        #    axs[i,j].set_ylim([-0.01, 0.005])    
        if i == 1:
            axs[i,j].set_ylim([-6, 1])
            axs[i,j].set_yticks(y_ticks)
            axs[i,j].set_xticks(x_ticks)
            axs[i,j].set(xlabel='Iteration')
            axs[i,j].yaxis.set_major_formatter(FuncFormatter(sci_fmt_2))
            if j == 0:
                axs[i,j].set(ylabel= r'Coefficients $u,\ u^3$')
        
    
lines, labels = [], []
for ax in fig.axes:
    line, label = ax.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)

fig.legend(
    lines,
    labels,
    loc="center",  # Change the location to upper center
    ncol=4,
    bbox_to_anchor=(0.51, 0.97),  # Adjust the coordinates
    bbox_transform=fig.transFigure,
    fontsize=15,
    frameon=True,
)


plt.savefig(filename_coef_s1 +'coefficients'+'.png', dpi=300,)
plt.savefig(filename_coef_s1 +'coefficients'+'.pdf', dpi=300,)

####################################
####################################


print(coefficients_f2)
print(coefficients_f1)
print(coefficients_2)
print(coefficients_1)


###################################
###################################


fig_5, ax_5 = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
scatter1 = ax_5[1, 1].scatter(X_str_train_f2[:, 1], X_str_train_f2[:, 0],
                            c=u_str_train_f2[:, 0], marker="x", s=10, label=r"\texttt{Ensemb}$_2$-Subdom$_2$")
scatter2 = ax_5[1, 0].scatter(X_str_train_2[:, 1], X_str_train_2[:, 0],
                            c=u_str_train_2[:, 0], marker="o", s=10, label=r"\texttt{Ensemb}$_1$-Subdom$_2$")
scatter3 = ax_5[0, 1].scatter(X_str_train_f1[:, 1], X_str_train_f1[:, 0],
                            c=u_str_train_f1[:, 0], marker="^", s=10, label=r"\texttt{Ensemb}$_1$-Subdom$_2$")
scatter4 = ax_5[0, 0].scatter(X_str_train_1[:, 1], X_str_train_1[:, 0],
                            c=u_str_train_1[:, 0], marker="D", s=10, label=r"\texttt{Ensemb}$_1$-Subdom$_1$")

ax_5[1, 0].set_xlabel('t')
ax_5[1, 0].set_ylabel('x')
ax_5[1, 1].set_xlabel('t')
ax_5[0, 0].set_ylabel('x')

# Add individual legends to each subplot
ax_5[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),fontsize=20)
ax_5[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),fontsize=20)
ax_5[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),fontsize=20)
ax_5[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),fontsize=20)

plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed

colorbar = fig_5.colorbar(scatter1, ax=ax_5.ravel().tolist(), location='right', aspect=50, fraction=0.01)
#fig_5.suptitle('Sampled Data set')

plt.savefig(filename_data + "dataset_ensemble" + ".png", dpi=300)
plt.savefig(filename_data + "dataset_ensemble" + ".pdf", dpi=300)
plt.show()

##########################################
##########################################

N_t=1000

idx = np.random.choice(X_test.shape[0], N_t, replace=False)

X_str_test = X_test[idx,:]
u_str_test = u_test[idx,:]

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
cbar1 = fig_6.colorbar(mappable = scatter1, ax=ax_6[0], ticks=ticker.MultipleLocator(0.5))
cbar2 = fig_6.colorbar(mappable = scatter2, ax=ax_6[1],ticks=ticker.MultipleLocator(0.5))
cbar3 = fig_6.colorbar(mappable = scatter3, ax=ax_6[2], ticks=ticker.MultipleLocator(0.01))

ax_6[0].set_title("Test data set")
ax_6[1].set_title("Prediction")
ax_6[2].set_title("Absolute Error")
ax_6[0].set_xlabel("t", fontsize=20)
ax_6[1].set_xlabel("t", fontsize=20)
ax_6[2].set_xlabel("t", fontsize=20)


ax_6[0].set_ylabel("x", fontsize=20)
plt.tight_layout()


plt.savefig(filename_data + "prediction_error" + ".png", dpi=300,)
plt.savefig(filename_data + "prediction_error" + ".pdf", dpi=300,)
plt.show()













