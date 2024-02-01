#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:19:09 2023

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
from PINN_DEIM.Functions.utils import (loss_func_AC, leastsquares_fit, loss_func_KDV, 
    equation_residual_AC,)
from PINN_DEIM.Functions.library import library_deriv, features_calc_KDV, calculate_theta_KDV
from PINN_DEIM.Functions import plot_config_file
from PINN_DEIM.Functions.ensemble_loader import LoaderProcessor, process_loader
from PINN_DEIM.Functions.library import (
    features_calc_AC, calculate_theta_AC, library_deriv,)
from PINN_DEIM.Functions.DEIM_class import DEIM
from PINN_DEIM.Functions.data_set_preparation import KDVEquation
from PINN_DEIM.Functions.deep_simulation_framework import PDEDeepModel
from PINN_DEIM.Functions.ensemble_trainer import EnsembleTrainer



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

data = scipy.io.loadmat(root_dir + "/data/kdv.mat")



"""
Defining an instance of Allen Cahn PDE, and spliting its time domain by the
indeces that is written, e.g.=101
"""

KDV_instance = KDVEquation(data, time_indices=101)

"""
the arguments are:
t_div_s1: number of time division for the first ensemble, e.g. 2
n_basis_s1: number of selected basis for the first ensemble, e.g. 5
"""

t_div = [2, 4, 2, 2]
n_basis = [1, 1, 1, 1]

(x_train_list, u_train_list, train_loaders) = KDV_instance.prepare_data_DEIM(
    t_div,
    n_basis,
    deim_tolerance = 1e-4
)


"""
train data sets for each ensemble with their associated torch tensor loader
"""

X_str_train_1 = x_train_list[0]
X_str_train_f1 = x_train_list[1]
X_str_train_2 = x_train_list[2]
X_str_train_f2 = x_train_list[3]

u_str_train_1 = u_train_list[0]
u_str_train_f1 = u_train_list[1]
u_str_train_2 = u_train_list[2]
u_str_train_f2 = u_train_list[3]

train_loader_1 = train_loaders[0]
train_loader_f1 = train_loaders[1]
train_loader_2 = train_loaders[2]
train_loader_f2 = train_loaders[3]




fig, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 8))

ax[0, 0].scatter(
    X_str_train_1[:, 1], X_str_train_1[:, 0], c=u_str_train_1[:, 0], marker="x", s=12
)
ax[1, 0].scatter(
    X_str_train_f1[:, 1], X_str_train_f1[:, 0], c=u_str_train_f1[:, 0], marker="x", s=12
)

ax[0, 1].scatter(
    X_str_train_2[:, 1], X_str_train_2[:, 0], c=u_str_train_2[:, 0], marker="x", s=12
)
ax[1, 1].scatter(
    X_str_train_f2[:, 1], X_str_train_f2[:, 0], c=u_str_train_f2[:, 0], marker="x", s=12
)

plt.show()


##############################################
##############################################


noise = 0
learning_rate_inr = 1e-5
hidden_features_str = 128
hidden_features = 64
hidden_layers = 3
num_epochs = 1500

string_f = KDV_instance.save_string(
    noise=0,
    learning_rate_inr=1e-5,
    hidden_features_str=128,
    hidden_layers=3,
    num_epochs=1500,
    random_sampling=False,
)


### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "./result_xpinn_KDV_1d_/results" + string_f
p = Path(result_path)
if not p.exists():
    os.makedirs(result_path)

##############################################
##############################################

"""
Defining deep NN structure for AC model, i.e. optimizer, scheduler, DNN model 
"""

KDVDeepModel_instance = PDEDeepModel(
                 in_features= 2,
                 out_features=1,
                 hidden_features_str=128, 
                 hidden_layers=3,
                 learning_rate_inr=1e-5) 

models_list, optim_adam, scheduler = KDVDeepModel_instance.run()


model_str_1 = models_list[0]
model_str_f1 = models_list[1]
model_str_2 = models_list[2]
model_str_f2 = models_list[3]


########################################
########### Training the deep neural network



Ensem_Train_inst = EnsembleTrainer(
    model_str_1,
    model_str_f1,
    model_str_2,
    model_str_f2,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
    features_calc = features_calc_KDV,
    calculate_theta = calculate_theta_KDV,
    loss_func= loss_func_KDV 
)

loss_func_list, coeff_track_list = Ensem_Train_inst.train_func(
    train_loader_1, train_loader_f1, train_loader_2, train_loader_f2
)

coef_s_1 = coeff_track_list[0]
coef_f_1 = coeff_track_list[1]
coef_s_2 = coeff_track_list[2]
coef_f_2 = coeff_track_list[3]


loss_tot_s1 = loss_func_list[0]
loss_tot_f1 = loss_func_list[1]
loss_tot_s2 = loss_func_list[2]
loss_tot_f2 = loss_func_list[3]

coef_s_tr1 = np.reshape(np.array(coef_s_1), (num_epochs, coef_s_1[0].shape[0]))
coef_f_tr1 = np.reshape(np.array(coef_f_1), (num_epochs, coef_f_1[0].shape[0]))

coef_s_tr2 = np.reshape(np.array(coef_s_2), (num_epochs, coef_s_2[0].shape[0]))
coef_f_tr2 = np.reshape(np.array(coef_f_2), (num_epochs, coef_f_2[0].shape[0]))

########################################
########################################

coef_s_tr1 = np.reshape(np.array(coef_s_1), (num_epochs, -1))
coef_f_tr1 = np.reshape(np.array(coef_f_1), (num_epochs, -1))

coef_s_tr2 = np.reshape(np.array(coef_s_2), (num_epochs, -1))
coef_f_tr2 = np.reshape(np.array(coef_f_2), (num_epochs, -1))

###############################################
###############################################

"""
file path, saving the data and related parameters
"""
(
    filename_loss_funcs,
    filename_dict,
    filename_coef_dict,
    filename_model_dict
) = KDV_instance.file_path_saving(result_path)

###############################################
###############################################


plt.rcParams.update(
    {
        "font.size": 20,
    }
)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))

x_ticks = [0, 500, 1000, 1500]

line_width = 3

axs2.semilogy(
    loss_tot_s1,
    marker="^",
    markersize=5,
    linewidth=line_width,
    markevery=100,
    linestyle="-",
)
axs2.semilogy(
    loss_tot_f1,
    marker="8",
    markersize=5,
    linewidth=line_width,
    markevery=100,
    linestyle="--",
)
axs2.semilogy(
    loss_tot_s2,
    marker="s",
    markersize=5,
    linewidth=line_width,
    markevery=100,
    linestyle=":",
)
axs2.semilogy(
    loss_tot_f2,
    marker="o",
    markersize=5,
    linewidth=line_width,
    markevery=100,
    linestyle="-.",
)

axs2.set_xticks(x_ticks)


axs2.set(xlabel="Iteration")
plt.title("Loss functions")

plt.tight_layout()
plt.savefig(
    filename_loss_funcs + "loss_function" + ".png",
    dpi=600,
)
plt.savefig(
    filename_loss_funcs + "loss_function" + ".pdf",
    dpi=600,
)

np.save(filename_dict["s1"], loss_tot_s1)
np.save(filename_dict["f1"], loss_tot_f1)
np.save(filename_dict["s2"], loss_tot_s2)
np.save(filename_dict["f2"], loss_tot_f2)







