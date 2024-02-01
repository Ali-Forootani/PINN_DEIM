#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:12:38 2023

@author: forootani
"""

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

data = scipy.io.loadmat(root_dir +'/data/burgers.mat')


## preparing and normalizing the input and output data
t = data["t"].flatten()

x = data["x"].flatten()

### use the below if the data is not normalized already
min_t = t.min()
max_t = t.max()


t_std = (t - t.min(axis=0)) / (t.max(axis=0) - t.min(axis=0))
t_norm = t_std * (1 - (-0)) -0

t_scaling = 1/(t.max()-t.min())




x_std = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
x_norm = x_std * (1 - (-1)) -1

x_scaling = 2/(x.max()-x.min())

#t_norm = (t-min_t)/(max_t-min_t)
#t_norm = ((t-min_t)/(max_t-min_t)-0.5)*2


#min_x = x.min()
#max_x = x.max()


#x_norm = x/1
#x_norm = ((x-min_x)/(max_x-min_x)-0.5)*2


t_1 = t_norm.flatten()[0:101, None]
t_2 = t_norm.flatten()[0:101, None]


Exact = np.real(data["usol"])


X_1, T_1 = np.meshgrid(x_norm, t_1, indexing="ij")
Exact_1 = Exact[:,0:X_1.shape[1]]


X_2, T_2 = np.meshgrid(x_norm, t_2, indexing="ij")
#Exact_2 = Exact[:,X_2.shape[1]+1:]
Exact_2 = Exact[:,0:X_2.shape[1]]


prec = 0.1
N_u = int(prec * len(Exact.flatten()))

X_star_1 = np.hstack((X_1.flatten()[:, None], T_1.flatten()[:, None]))
u_star_1 = Exact_1.flatten()[:, None]
idx_1 = np.random.choice(X_star_1.shape[0], N_u, replace=False)


X_f_1 = np.hstack((X_star_1, u_star_1))
idx_f1 = np.random.choice(X_f_1.shape[0], N_u, replace=False)


#X_str_train_1, X_test, u_str_train_1, y_test = train_test_split(X_star_1, 
#                                                          u_star_1, test_size=0.98, random_state=42)

X_str_train_1 = X_star_1[idx_1,:]
u_str_train_1 = u_star_1[idx_1,:]

#X_str_train_f1, X_test, u_str_train_f1, y_test = train_test_split(X_star_1, 
#                                                          u_star_1, test_size=0.98, random_state=42)

X_str_train_f1= X_star_1[idx_f1,:]
u_str_train_f1= u_star_1[idx_f1,:]



X_star_2 = np.hstack((X_2.flatten()[:, None], T_2.flatten()[:, None]))
u_star_2 = Exact_2.flatten()[:, None]
X_f_2 = np.hstack((X_star_2, u_star_2))

idx_2 = np.random.choice(X_star_2.shape[0], N_u, replace=False)



X_str_train_2 = X_star_2[idx_2,:]
u_str_train_2 = u_star_2[idx_2,:]

idx_f2 = np.random.choice(X_f_2.shape[0], N_u, replace=False)


#X_str_train_f2, X_test, u_str_train_f2, y_test = train_test_split(X_star_2, 
#                                                          u_star_2, test_size=0.98, random_state=42)


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


colorbar = fig_5.colorbar(scatter1, ax=ax_5.ravel().tolist(), location='right', aspect=50, fraction=0.01)

#fig_5.colorbar(scatter1, ax=ax.ravel().tolist(), location='right', label='Colorbar Label')
#fig_5.colorbar.ax.set_aspect(8) 
#plt.tight_layout()
#plt.show()

#colorbar = fig_5.colorbar(scatter1, ax=ax.ravel().tolist(), location='right', aspect=50, fraction=0.01)
fig_5.suptitle('Different Data set')

plt.savefig("different_data"+".png")
plt.savefig("different_data"+".pdf")
plt.show()



in_features = 2
out_features = 1

hidden_features= 64
hidden_layers= 4

hidden_features_str = 64
hidden_layers_str = 4

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
    max_lr= 10 * learning_rate_inr,
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
    
    #t_scale = 1
    #x_scale = 1
    
    u_t = t_scale * torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_x = x_scale * torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = x_scale * torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    
    #u_t = u_t/u_t.norm()
    #u_x = u_x/u_x.norm()
    #u_xx = u_xx/u_xx.norm()
    
    uu_x = u * u_x
    
    
    features = torch.cat([u_t, uu_x, u_xx], dim=1)
    return features


### calculation of dictionary theta


def calculate_theta(features, u_pred):
    
    thetas = torch.cat([features[:, 1:2], features[:, 2:3] ], dim=1)

    return thetas


def weight_init(m):
    if isinstance(m, nn.Linear):
        #init.xavier_uniform_(m.weight)
        init.xavier_normal_(m.weight)
        

"""
model_str_1.apply(weight_init)
model_str_f1.apply(weight_init)
model_str_2.apply(weight_init)
model_str_f2.apply(weight_init)
model_int.apply(weight_init)
"""




num_epochs = 2000

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
        loss_data_1 = torch.mean((out_1 - u_pred_1) ** 2)
        
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
        
    loss = 0.1*loss_data_1 + 0.1*loss_data_2 + 0.1*loss_f2 + 0.1*loss_f1 
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


################################
################################



##########

fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))

axs2.semilogy(loss_tot_s1)
axs2.semilogy(loss_tot_f1)
axs2.semilogy(loss_tot_s2)
axs2.semilogy(loss_tot_f2)


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






plt.rcParams.update({
    "font.size": 20,
})

line_width=3

fig, axs = plt.subplots(1, 3, sharey=False, figsize=(15, 4))



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

#plt.suptitle(r'\texttt{E-PINN}')
plt.tight_layout()
#plt.savefig(pltname_p2 +'loss_function'+'.png', dpi=300,)
#plt.savefig(pltname_p2 +'loss_function'+'.pdf', dpi=300,)
####################################
####################################


fig_5, ax = plt.subplots(2, 2, figsize=(8, 6), sharey=True, sharex=False)

scatter1 = ax[1, 1].scatter(X_str_train_f2[:, 1], X_str_train_f2[:, 0], c=u_str_train_f2[:, 0], marker="x", s=10)
scatter2 = ax[1, 0].scatter(X_str_train_2[:, 1], X_str_train_2[:, 0], c=u_str_train_2[:, 0], marker="o", s=10)
scatter3 = ax[0, 1].scatter(X_str_train_f1[:, 1], X_str_train_f1[:, 0], c=u_str_train_f1[:, 0], marker="^", s=10)
scatter4 = ax[0, 0].scatter(X_str_train_1[:, 1], X_str_train_1[:, 0], c=u_str_train_1[:, 0], marker="D", s=10)

ax[1, 0].set_xlabel('t')
ax[1, 0].set_ylabel('x')
ax[1, 1].set_xlabel('t')
ax[0, 0].set_ylabel('x')

#fig_5.colorbar(scatter1, ax=ax.ravel().tolist(), location='right', label='Colorbar Label')
#fig_5.colorbar.ax.set_aspect(8) 
#plt.tight_layout()
#plt.show()

colorbar = fig_5.colorbar(scatter1, ax=ax.ravel().tolist(), location='right', aspect=50, fraction=0.01)
fig_5.suptitle('Different Data set')

#plt.savefig(filename_data + "different_data" + ".png")
#plt.savefig(filename_data + "different_data" + ".pdf", dpi=300,)
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



#plt.suptitle(r'\texttt{E-PINN}')
plt.tight_layout()

print(coefficients_f2)
print(coefficients_f1)
print(coefficients_2)
print(coefficients_1)



################################
################################
"""
def create_data():
    
    data = scipy.io.loadmat("data/burgers.mat")

    ## preparing and normalizing the input and output data
    t = data["t"].flatten()
    x = data["x"].flatten()
    Exact = np.real(data["usol"])
    X, T = np.meshgrid(x, t, indexing="ij")
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]
    
    #coords = torch.from_numpy(np.stack((t,x), axis=-1))
    #data = torch.from_numpy(np.real(data["uu"])).unsqueeze(-1)
    # alternative way of providing the coordinates
    # coords = torch.from_numpy(np.transpose((t_v.flatten(), x_v.flatten(), y_v.flatten())))
    # data = torch.from_numpy(usol[:, :, :, 3].reshape(-1,1))
    print("The coodinates have shape {}".format(X_star.shape))
    print("The data has shape {}".format(u_star.shape))
    
    X_star = torch.tensor(X_star, requires_grad = True)
    u_star = torch.tensor(u_star, requires_grad = True)
    
    return X_star, u_star


x_t, u = create_data()

x_t = x_t.detach().cpu().numpy()
u = u.detach().cpu().numpy()
fig, ax = plt.subplots()
im = ax.scatter(x_t[:,1], x_t[:,0], c=u[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)

dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": True,
        "normalize_data": True,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 1000},
    device=device,)

coords = dataset.get_coords().detach().cpu()
data = dataset.get_data().detach().cpu()
fig, ax = plt.subplots()
im = ax.scatter(coords[:,1], coords[:,0], c=data[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)
plt.show()

coords = dataset.get_coords()
data = dataset.get_data()

"""




















