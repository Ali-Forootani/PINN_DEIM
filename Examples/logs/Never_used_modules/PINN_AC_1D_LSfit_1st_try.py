#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:52:26 2023

@author: forootani
"""



#!/usr/bin/env python3

import numpy as np
import sys
import os
from pathlib import Path
import torch

from numba import njit

from scipy import linalg

from Functions.modules import Siren
from Functions.utils import loss_func_AC
from Functions.utils import leastsquares_fit
from Functions.utils import equation_residual_AC
from Functions.base import Dataset, get_train_test_loader
from Functions.samples import Subsample_random
from Functions.library import library_deriv
from Functions.constraint import STRidgeCons
from sklearn.model_selection import train_test_split





from Functions.func_approx import NN, Siren


from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
import time
from scipy.io import loadmat


warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


## calculation of derivatives

"""
We should make it general I think !!!
"""
##########################################
##########################################
##########################################

def features_calc(X_n, u):
    
    print(type(X_n))
    print(u.size())
    
    
    u_t = torch.autograd.grad(
        u, X_n, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    
    print(u_t)
    
    u_x = torch.autograd.grad(
        u, X_n, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = torch.autograd.grad(
        u_x, X_n, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]

    features = torch.cat([u_t, u_xx], dim=1)
    return features


### calculation of dictionary theta


def calculate_theta(features, u_pred):
    thetas = torch.cat([features[:, 1:2], (torch.pow((u_pred), 3) - u_pred)], dim=1)

    return thetas

#######################################
#######################################
#######################################

## Training parameters


num_params = 2  ## number of equation parameters to be learned
learning_rate_inr = (
    1e-5  ## learning rate value for Siren network of primary variable phi
)
learning_rate_mu = (
    1e-5  ## learning rate value for Siren network of auxilliary vaiable mu
)

perc = 0.05  ## percentage of the data considered
noise = 0.3  ## percentage of noise considered

in_features = 2  ## number of input features to the neural network (for 1D spatio temporal case, use 2: for 2D spatio temporal case, use 3)
out_features = 1  ## number of output features
hidden_features_phi = 128  ## hidden features for the primary SIREN network
hidden_features_mu = 32  ## hidden features for the auxilliary SIREN network
hidden_layers = 3  ## number of hidden layers for both networks
batch_size = 4096  ## batch size for the training
num_epochs = 4000  ## total number of epoch for training

### file saving parameters

string_f = (
    "_LSfit_lr_"
    + str(learning_rate_inr)
    + "_hf_phi_"
    + str(hidden_features_phi)
    + "_layers_"
    + str(int(hidden_layers))
    + "_ep_"
    + str(int(num_epochs))
    + "_noise_"
    + str(int(100 * noise))
    + "_perc_"
    + str(int(100 * perc))
)

### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "result_AC_1d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.mkdir(result_path)
## File saving parameters
filename_u = (
    result_path + "/" + "U_data" + string_f + ".npy"
)  ## path where the primary variable phi data is saved
filename_mu = (
    result_path + "/" + "mu_pred" + string_f + ".npy"
)  ## path where the auxilliary variable mu data is saved
filename_l = (
    result_path + "/" + "Loss_collect" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved
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


#####################################
#####################################




def create_data():
    data = loadmat("data/AC.mat")
    
    data = scipy.io.loadmat("data/AC.mat")

    ## preparing and normalizing the input and output data
    t = data["tt"].flatten()[0:201, None]
    x = data["x"].flatten()[:, None]
    Exact = np.real(data["uu"])
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
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 6000},
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



####################


X_t = torch.tensor(dataset.get_coords(), requires_grad=True).float().to(device)
Y_t = torch.tensor(data).float().to(device)


X_t,X_test,Y_t,Y_test = train_test_split(x_t, u, test_size=0.94, random_state=42) 

fig, ax = plt.subplots()
im = ax.scatter(X_t[:,1], X_t[:,0], c=Y_t[:,0], marker="x", s=10)
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(mappable=im)
plt.show()



X_t=torch.tensor(X_t, requires_grad=True).float().to(device)
Y_t=torch.tensor(Y_t).float().to(device)

#####################



train_loader, test_loader = get_train_test_loader(dataset, train_test_split=0.9)



train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=True)




#siren model initialization
#model_inr = Siren(
#    in_features= in_features,
#    out_features= out_features,
#    hidden_features= hidden_features_phi,
#    hidden_layers= hidden_layers,
#    outermost_linear= True,
#).to(device)


model_inr = Siren(2, [128, 128, 128, 128], 1).to(device)



print(model_inr)



optim_adam = torch.optim.Adam(
    [{"params": model_inr.parameters(), "lr": learning_rate_inr, "weight_decay": 1e-6},]
)




### learnig scheduler cyclic damping
# scheduler = torch.optim.lr_scheduler.CyclicLR(optim_adam, base_lr=0.1*learning_rate_inr, max_lr=10*learning_rate_inr, cycle_momentum=False, mode='exp_range', step_size_up=1000)


### learnig scheduler cyclic
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optim_adam,
    base_lr=0.1 * learning_rate_inr,
    max_lr=10 * learning_rate_inr,
    cycle_momentum=False,)


Loss_collect = np.zeros((num_epochs, 3))
params_collect = np.zeros((num_epochs, num_params))
STRidge = STRidgeCons()












print("Parameters are learned")
print("Training Begins !!")

for epoch in range(num_epochs):
    # print(f'epoch {epoch}')
    loss_epoch = 0
    loss_data_epoch = 0
    loss_eq_epoch = 0

    ii = 0

    start_time = time.time()

    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    

    for batch_idx, (local_batch, output) in loop:
        
        #u_pred, coordinates = model_inr(local_batch.float())
        
        u_pred = model_inr(local_batch.float())

        ## features calculation
        
        features = library_deriv(local_batch.float(), u_pred, max_order= 2)
        
        #print(features)
        
        #features = features_calc(local_batch.float(), u_pred)

        time_derivs = features[:, 0:1]

        ### calculation of dictionary theta
        thetas = calculate_theta(features, u_pred)

        ### least squares fit of the coefficients
        coefficients = leastsquares_fit(thetas, time_derivs)
        # = STRidge.fit(thetas, time_derivs)
        
        ################################
        ################################
        ################################

        ### equation residual
        residual = equation_residual_AC(thetas, time_derivs, coefficients)

        ### loss evaluation
        loss, loss_data, loss_eq = loss_func_AC(output, u_pred, residual)

        # Backward and optimize
        optim_adam.zero_grad()
        loss.backward()
        optim_adam.step()

        # scheduler step
        scheduler.step()
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(
            loss=loss.item(), loss_data=loss_data.item(), loss_eq=loss_eq.item(),
        )
        # loop.set_postfix(loss = loss.item())

        loss_epoch += loss.item()
        loss_data_epoch += loss_data.item()
        loss_eq_epoch += loss_eq.item()

        # loop.set_postfix(residual = torch.mean(residual).item(), mu_lap = torch.mean(mu_lap).item(), time_deriv = torch.mean(features[:,0:1]).item())
    loss_epoch = loss_epoch / len(train_loader)
    loss_data_epoch = loss_data_epoch / len(train_loader)
    loss_eq_epoch = loss_eq_epoch / len(train_loader)
    if epoch % 1 == 0:
        print(
            "It: %d, Loss: %.3e, Loss_data: %.3e, Loss_eq: %.3e, Lambda_1: %.3e, Lambda_2: %.3e"
            % (
                epoch,
                loss_epoch,
                loss_data_epoch,
                loss_eq_epoch,
                coefficients[0].item(),
                coefficients[1].item(),
            )
        )
    Loss_collect[epoch, 0] = loss_epoch
    Loss_collect[epoch, 1] = loss_data_epoch
    Loss_collect[epoch, 2] = loss_eq_epoch
    params_collect[epoch, 0] = coefficients[0].item()
    params_collect[epoch, 1] = coefficients[1].item()

data = loadmat("data/AC.mat")

data = scipy.io.loadmat("data/AC.mat")

t = data["tt"].flatten()[0:201, None]
x = data["x"].flatten()[:, None]
Exact = np.real(data["uu"])
X, T = np.meshgrid(x, t, indexing="ij")
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]




X_star = torch.tensor(X_star, requires_grad=True).float().to(device)


u_pred_total = model_inr(X_star)


u_pred_total = u_pred_total.cpu()
u_pred_total = u_pred_total.detach().numpy()

X_star = X_star.cpu()
X_star = X_star.detach().numpy()

U_pred = griddata(X_star, u_pred_total, (X, T), method="nearest")

U_pred = U_pred[:, :, 0]

error = np.abs(U_pred - Exact) / linalg.norm(Exact, "fro")


U_noisy = u

u_data = {"orig": Exact, "pinn": U_pred, "noisy": U_noisy, "error": error}
np.save(filename_u, u_data)

np.save(filename_l, Loss_collect)
np.save(filename_p, params_collect)

torch.save(model_inr, filename_model_u)

### plotting



plt.rcParams.update({
    "font.family":  "serif.Times New Roman",
    "font.style":   "normal",
    "font.variant": "normal",
    "font.weight":  "bold",
    "font.stretch": "normal",
    "font.size": 20,
    "legend.fancybox":  True,
    "text.usetex": True,
    #"text.latex.preamble": [r'\usepackage{amsmath,amssymb,bm,fontenc,physics,lmodern,nicefrac}']
    "pgf.rcfonts": True,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [r'\usepackage{amsmath},\usepackage{amssymb},\usepackage{bm},\usepackge{fontenc},\usepackage{physics},\usepackge{lmodern},\usepackage{nicefrac}'],
    "figure.subplot.hspace": 0.25,
    })



fig2 = plt.figure(figsize=(6, 6))
shape_loss = Loss_collect.shape[1]
if shape_loss == 1:
    plt.semilogy(Loss_collect[:, 0:1])
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.savefig(pltname_l)
if shape_loss == 3:
    plt.semilogy(Loss_collect[:, 0:1], label="Total loss")
    plt.semilogy(Loss_collect[:, 1:2], label="Data loss")
    plt.semilogy(Loss_collect[:, 2:3], label="Equation loss")
    plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.savefig(pltname_l)
if shape_loss == 4:
    plt.semilogy(Loss_collect[:, 0:1], label="Total loss")
    plt.semilogy(Loss_collect[:, 1:2], label="Data loss")
    plt.semilogy(Loss_collect[:, 2:3], label="Equation loss")
    plt.semilogy(Loss_collect[:, 3:4], label="$\mu$ loss")
    plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.savefig(pltname_l)

fig = plt.figure(figsize=(20, 5))
ax = fig.add_subplot(141)


h0 = ax.imshow(
    U_noisy,
    interpolation="nearest",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h0, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("Noisy")
ax.set_xlabel("$t$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(142)
h1 = ax.imshow(
    Exact,
    interpolation="nearest",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h1, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("Original")
ax.set_xlabel("$t$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(143)
h2 = ax.imshow(
    U_pred,
    interpolation="nearest",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h2, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("SIREN representation")
ax.set_xlabel("$t$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(144)
h3 = ax.imshow(
    error,
    interpolation="nearest",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h3, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("error")
ax.set_xlabel("$t$", size=20)
ax.set_ylabel("$x$", size=20)

fig.tight_layout(pad=3.0)
plt.savefig(pltname_u + ".png")

fig5 = plt.figure(figsize=(6, 6))
plt.plot(params_collect[:, 0], label=("$\lambda_1$"))
plt.yscale("symlog", linthreshy=1e-4)

plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
plt.xlabel("Epoch")
plt.ylabel("Parameter")
plt.savefig(pltname_p1)


fig6 = plt.figure(figsize=(6, 6))
plt.plot(params_collect[:, 1], label=("$\lambda_2$"))
plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
plt.xlabel("Epoch")
plt.ylabel("Parameter")
plt.savefig(pltname_p2)



