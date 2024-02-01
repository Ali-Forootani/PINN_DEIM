#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:49:45 2023

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
from PINN_DEIM.Functions.utils import (
    loss_func_KDV,
    leastsquares_fit,
    equation_residual_AC,
)
from PINN_DEIM.Functions.library import library_deriv
from PINN_DEIM.Functions import plot_config_file
from PINN_DEIM.Functions.ensemble_loader import LoaderProcessor, process_loader
from PINN_DEIM.Functions.library import (
    features_calc_KDV,
    calculate_theta_KDV,
    library_deriv,
)
from PINN_DEIM.Functions.deep_simulation_framework import PDEDeepModelEntireDomain
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time

from PINN_DEIM.Functions.data_set_preparation import KDVEquationEntireDomain
from PINN_DEIM.Functions.ensemble_trainer import EnsembleTrainer, Trainer


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

########################################################
######## Making directory and file saving settings


noise=0
learning_rate_inr=1e-5
hidden_features_str=128
hidden_layers=3
num_epochs=1500

string_f = ("_KDV_sensitivity_DEIM_random_sampling_"
            + "_noise_" + str(int(noise)) 
            + "_learning_rate_inr_" + str(int(learning_rate_inr))
            + "_hidden_features_str_" + str(int(hidden_features_str))
            + "_hidden_layers_" + str(int(hidden_layers))
            + "_nim_epochs_" + str(int(num_epochs))
            )

### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "./logs/result_DEIM_KDV_1d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.makedirs(result_path)


#############  DEIM Object

#tol_list = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5 , 1e-4, 1e-3]


#tol_list = [ 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2]

tol_list = np.logspace(-10, -2, 20)

coef_s_tr_list_DEIM = []
coef_dic_list_DEIM = []

coef_s_tr_list_random_samp = []
coef_dic_list_random_samp = []
num_samples = []

#tol_list = [1e-15, 1e-14]

num_epochs = 1500

for i in range(len(tol_list)):
    
    KDV_DEIM = KDVEquationEntireDomain(data)
    (x_train_list_DEIM, u_train_list_DEIM, train_loaders_DEIM)= KDV_DEIM.prepare_data_DEIM(
                                                    t_div = 2,
                                                    n_basis=1,
                                                    deim_tolerance = tol_list[i])
                                                    
    X_str_train_DEIM = x_train_list_DEIM[0]
    u_str_train_DEIM = u_train_list_DEIM[0]
    train_loader_DEIM = train_loaders_DEIM[0]
    
    #print(X_str_train_DEIM.shape)
    num_samples.append( X_str_train_DEIM.shape[0] )
    
    ############################################
    ############################################
    
    """
    Defining deep NN structure for AC model, i.e. optimizer, scheduler, DNN model 
    """
    
    KDVDeepModel_DEIM = PDEDeepModelEntireDomain(
        in_features=2,
        out_features=1,
        hidden_features_str=128,
        hidden_layers=3,
        learning_rate_inr=1e-5,)

    models_list_DEIM, optim_adam_DEIM, scheduler_DEIM = KDVDeepModel_DEIM.run()
    
    model_str_DEIM = models_list_DEIM[0]
    
    
    Train_DEIM = Trainer(
        model_str_DEIM,
        num_epochs= num_epochs,
        optim_adam=optim_adam_DEIM,
        scheduler=scheduler_DEIM,
        features_calc = features_calc_KDV,
        calculate_theta = calculate_theta_KDV,
        loss_func = loss_func_KDV
        )
    
    loss_func_list_DEIM, coeff_track_list_DEIM = Train_DEIM.train_func(
        train_loader_DEIM)
    
    coef_s_DEIM = coeff_track_list_DEIM[0]
    loss_tot_s_DEIM = loss_func_list_DEIM[0]
    coef_s_tr_DEIM = np.reshape(np.array(coef_s_DEIM), (num_epochs, coef_s_DEIM[0].shape[0]))
    
    coef_s_tr_list_DEIM.append([coef_s_tr_DEIM])
    
    coef_dic_list_DEIM.append( { "tol": tol_list[i],
               "coeff": coef_s_tr_DEIM[-1] } ) 
    
    ##################################################
    ##################################################
    
    #data = scipy.io.loadmat(root_dir + "/data/AC.mat")
    org_data_size = data["usol"].shape[0] * data["usol"].shape[1]
    test_data_size = 1 - (X_str_train_DEIM.shape[0]/org_data_size)
    
    KDV_random_samp = KDVEquationEntireDomain(data)

    (x_train_list_random,
     u_train_list_random,
     train_loaders_random,
     X_test_random,
     u_test_random) = KDV_random_samp.prepare_data_random(test_data_size)
    
    X_str_train_random = x_train_list_random[0]
    u_str_train_random = u_train_list_random[0]
    train_loader_random = train_loaders_random[0]
    
    
    
    KDVDeepModel_random_samp = PDEDeepModelEntireDomain(
        in_features=2,
        out_features=1,
        hidden_features_str=128,
        hidden_layers=3,
        learning_rate_inr=1e-5,)
    
    models_list_random_samp, optim_adam_random_samp, scheduler_random_samp = KDVDeepModel_random_samp.run()
    
    model_str_random_samp = models_list_random_samp[0]
    
     
    Train_inst = Trainer(
        model_str_random_samp,
        num_epochs=num_epochs,
        optim_adam=optim_adam_random_samp,
        scheduler=scheduler_random_samp,
        features_calc = features_calc_KDV,
        calculate_theta = calculate_theta_KDV,
        loss_func = loss_func_KDV
                        )
    
    loss_func_list_random_samp, coeff_track_list_random_samp = Train_inst.train_func(
        train_loader_random
        )
    
    coef_s_random_samp = coeff_track_list_random_samp[0]

    loss_tot_s_random_samp = loss_func_list_random_samp[0]

    coef_s_tr_random_samp = np.reshape(np.array(coef_s_random_samp),
                                       (num_epochs, coef_s_random_samp[0].shape[0]))
    
    coef_s_tr_list_random_samp.append([coef_s_tr_random_samp])
    
    coef_dic_list_random_samp.append( { "tol": tol_list[i],
               "coeff": coef_s_tr_random_samp[-1] } ) 
    
 
#################################################
#################################################


filename_DEIM = result_path + "/" + "coef_dic_list_DEIM" + ".npy"
filename_random_samp = result_path + "/" + "coef_dic_list_random_samp" + ".npy"

np.save(filename_DEIM, coef_dic_list_DEIM)
np.save(filename_random_samp, coef_dic_list_random_samp)




##################################################
##################################################

coef_1_ref = -6e0    
coef_2_ref = -1e0
    
error_coef_1_DEIM = np.zeros((1, len(tol_list)))
error_coef_2_DEIM = np.zeros((1, len(tol_list)))

error_coef_1_random_samp = np.zeros((1, len(tol_list)))
error_coef_2_random_samp = np.zeros((1, len(tol_list)))


for j in range(len(tol_list)):    
    #print( np.abs(coef_1_ref - coef_dic_list_DEIM[j]["coeff"][0]) )
    
    error_coef_1_DEIM[0, j] = np.abs(coef_1_ref - coef_dic_list_DEIM[j]["coeff"][0])/ np.abs(coef_1_ref)   
    error_coef_2_DEIM[0, j] = np.abs(coef_2_ref - coef_dic_list_DEIM[j]["coeff"][1])/ np.abs(coef_2_ref)   
    
    
    error_coef_1_random_samp[0, j] = np.abs(coef_1_ref - coef_dic_list_random_samp[j]["coeff"][0])/ np.abs(coef_1_ref)    
    error_coef_2_random_samp[0, j] = np.abs(coef_2_ref - coef_dic_list_random_samp[j]["coeff"][1])/ np.abs(coef_2_ref) 




filename_DEIM_e_coef_1 = result_path + "/" + "error_coef_1_DEIM" + ".npy"
filename_DEIM_e_coef_2 = result_path + "/" + "error_coef_2_DEIM" + ".npy"

filename_random_samp_e_coef_1 = result_path + "/" + "error_coef_1_random_samp" + ".npy"
filename_random_samp_e_coef_2 = result_path + "/" + "error_coef_2_random_samp" + ".npy"


np.save(filename_DEIM_e_coef_1, error_coef_1_DEIM)
np.save(filename_DEIM_e_coef_2, error_coef_2_DEIM)

np.save(filename_random_samp_e_coef_1, error_coef_1_DEIM)
np.save(filename_random_samp_e_coef_2, error_coef_2_DEIM)


#################################################
#################################################

plt.rcParams.update(
    {
        "font.size": 20,
    }
)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
line_width = 2

fig, axs = plt.subplots(1, 2, sharey=False, figsize=(9, 3))
axs[0].semilogy(np.array(num_samples), error_coef_1_DEIM[0,:], linewidth=line_width, marker= "*",linestyle="-", color=colors[0])
axs[0].semilogy(np.array(num_samples), error_coef_1_random_samp[0,:], linewidth=line_width, marker= "s",linestyle="--", color=colors[1])
axs[0].set(xlabel="Number of samples")
axs[0].set(ylabel=r"Coefficient $u_{xxx}$")

axs[1].semilogy(
    np.array(num_samples), error_coef_2_DEIM[0,:],
    linewidth=line_width,
    linestyle="-", marker= "*",
    color=colors[0],
    label=r"\texttt{DEIM} Sampling",)

axs[1].semilogy(
    np.array(num_samples),error_coef_2_random_samp[0,:],
    linewidth=line_width, marker= "s",
    linestyle="--",
    color=colors[1],
    label=r"\texttt{Random} Sampling",)

axs[1].set(xlabel="Number of samples")
axs[1].set(ylabel=r"Coefficient $uu_x$ ")


"""
# Create a twin y-axis for the first subplot (axs[0])
ax2_0 = axs[0].twiny()
ax2_0.semilogy(
    tol_list[::-1], error_coef_1_DEIM[0,:],
    linewidth=line_width, marker="o",
    linestyle="-", color=colors[5],  
    
)
#ax2_0.set(ylabel="Additional Data 0")
"""



lines, labels = [], []
for ax in fig.axes:
    line, label = ax.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)

fig.legend(
    lines, labels, loc="center",  # Change the location to upper center
    ncol=4,
    bbox_to_anchor=(0.5, 1),  # Adjust the coordinates
    bbox_transform=fig.transFigure, fontsize=15, frameon=True,)

plt.tight_layout()




plt.savefig(
    result_path + "/" + "relative_abs_error_" + "_sensitivity_analysis_" + ".png",
    dpi=300,
    bbox_inches="tight",)
plt.savefig(
    result_path + "/" + "relative_abs_error_" + "_sensitivity_analysis_" + ".pdf",
    dpi=300,
    bbox_inches="tight",)

plt.show()


##############################################
##############################################



"""
# Create a twin y-axis for the first subplot (axs[0])
ax2_0 = axs[0].twiny()
ax2_0.semilogy(
    tol_list, error_coef_1_DEIM[0,:],
    linewidth=line_width, marker="o",
    linestyle="-", color=colors[2],  
    
)
ax2_0.set(ylabel="Additional Data 0")

# Create a twin y-axis for the second subplot (axs[1])
ax2_1 = axs[1].twiny()
ax2_1.semilogy(
    tol_list, error_coef_2_DEIM[0,:],
    linewidth=line_width, marker="x",
    linestyle="-", color=colors[3],  
)
ax2_1.set(ylabel="Additional Data 1")

plt.show()

"""

