#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:07:37 2023

@author: forootani
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:58:59 2023

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
    loss_func_AC,
    leastsquares_fit,
    equation_residual_AC,
)
from PINN_DEIM.Functions.library import library_deriv
from PINN_DEIM.Functions import plot_config_file
from PINN_DEIM.Functions.ensemble_loader import LoaderProcessor, process_loader
from PINN_DEIM.Functions.library import (
    features_calc_AC,
    calculate_theta_AC,
    library_deriv,
)
from PINN_DEIM.Functions.deep_simulation_framework import AllenCahnDeepModel, AllenCahnDeepModelEntireDomain
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time

from PINN_DEIM.Functions.data_set_preparation import AllenCahnEquation, AllenCahnEquationEntireDomain
from PINN_DEIM.Functions.ensemble_trainer import EnsembleTrainer, Trainer


warnings.filterwarnings("ignore")


# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

####################### loading the data set
#############################################

data = scipy.io.loadmat(root_dir + "/data/AC.mat")




########################################################
######## Making directory and file saving settings


noise=0
learning_rate_inr=1e-5
hidden_features_str=128
hidden_layers=3
num_epochs=1500

string_f = ("_AC_sensitivity_DEIM_random_sampling_"
            + "_noise_" + str(int(noise)) 
            + "_learning_rate_inr_" + str(int(learning_rate_inr))
            + "_hidden_features_str_" + str(int(hidden_features_str))
            + "_hidden_layers_" + str(int(hidden_layers))
            + "_nim_epochs_" + str(int(num_epochs))
            )

### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "./logs/result_DEIM_AC_1d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.makedirs(result_path)


#############  
"""
tol_list = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8,
            1e-7, 1e-6, 1e-5, 5e-5,
            1e-4, 3e-4, 5e-4, 1e-3]
"""


tol_list = np.logspace(-10, -2, 20)



coef_s_tr_list_DEIM = []
coef_dic_list_DEIM = []

coef_s_tr_list_random_samp_experiment = []
coef_dic_list_random_samp_experiment = []
num_samples = []


num_epochs = 100

for i in range(len(tol_list)):
    
    AllenCahn_DEIM = AllenCahnEquationEntireDomain(data)
    (x_train_list_DEIM, u_train_list_DEIM, train_loaders_DEIM)= AllenCahn_DEIM.prepare_data_DEIM(
                                                    t_div=4,
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
    
    AllenCahnDeepModel_DEIM = AllenCahnDeepModelEntireDomain(
        in_features=2,
        out_features=1,
        hidden_features_str=128,
        hidden_layers=3,
        learning_rate_inr=1e-5,)

    models_list_DEIM, optim_adam_DEIM, scheduler_DEIM = AllenCahnDeepModel_DEIM.run()
    
    model_str_DEIM = models_list_DEIM[0]
    
    
    Train_DEIM = Trainer(
        model_str_DEIM,
        num_epochs= num_epochs,
        optim_adam=optim_adam_DEIM,
        scheduler=scheduler_DEIM,
        features_calc = features_calc_AC,
        calculate_theta = calculate_theta_AC,
        loss_func = loss_func_AC
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
    org_data_size = data["uu"].shape[0] * data["uu"].shape[1]
    test_data_size = 1 - (X_str_train_DEIM.shape[0]/org_data_size)
    
    
    for j in range(len(tol_list)):
        
        
        coef_s_tr_list_random_samp = []
        coef_dic_list_random_samp = []
        
        
        AllenCahn_random_samp = AllenCahnEquationEntireDomain(data)

        (x_train_list_random,
         u_train_list_random,
         train_loaders_random,
         X_test_random,
         u_test_random) = AllenCahn_random_samp.prepare_data_random(test_data_size)
    
        X_str_train_random = x_train_list_random[0]
        u_str_train_random = u_train_list_random[0]
        train_loader_random = train_loaders_random[0]
    
    
    
        AllenCahnDeepModel_random_samp = AllenCahnDeepModelEntireDomain(
            in_features=2,
            out_features=1,
            hidden_features_str=128,
            hidden_layers=3,
            learning_rate_inr=1e-5,)
    
        models_list_random_samp, optim_adam_random_samp, scheduler_random_samp = AllenCahnDeepModel_random_samp.run()
    
        model_str_random_samp = models_list_random_samp[0]
    
     
        Train_inst = Trainer(
            model_str_random_samp,
            num_epochs=num_epochs,
            optim_adam=optim_adam_random_samp,
            scheduler=scheduler_random_samp,
            features_calc = features_calc_AC,
            calculate_theta = calculate_theta_AC,
            loss_func = loss_func_AC
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
        
        coef_s_tr_list_random_samp_experiment.append( coef_s_tr_list_random_samp )
    
        coef_dic_list_random_samp_experiment.append( coef_dic_list_random_samp )
        
        

#################################################
#################################################    


# filename_DEIM = result_path + "/" + "coef_dic_list_DEIM" + ".npy"
# filename_random_samp = result_path + "/" + "coef_dic_list_random_samp" + ".npy"

# np.save(filename_DEIM, coef_dic_list_DEIM)
# np.save(filename_random_samp, coef_dic_list_random_samp)

 
#################################################
#################################################
    
coef_1_ref = 1e-4    
coef_2_ref = -5e0
    
error_coef_1_DEIM = np.zeros((1, len(tol_list)))
error_coef_2_DEIM = np.zeros((1, len(tol_list)))

error_coef_1_random_samp = np.zeros((1, len(tol_list)))
error_coef_2_random_samp = np.zeros((1, len(tol_list)))


for j in range(len(tol_list)):    
    #print( np.abs(coef_1_ref - coef_dic_list_DEIM[j]["coeff"][0]) )
    
    error_coef_1_DEIM[0, j] = np.abs(coef_1_ref - coef_dic_list_DEIM[j]["coeff"][0])/ np.abs(coef_1_ref)   
    error_coef_2_DEIM[0, j] = np.abs(coef_2_ref - coef_dic_list_DEIM[j]["coeff"][1])/ np.abs(coef_2_ref)   
    
    
    error_coef_1_random_samp[0, j] = np.abs(coef_1_ref - coef_dic_list_random_samp_experiment[j][0]["coeff"][0])/ np.abs(coef_1_ref)    
    error_coef_2_random_samp[0, j] = np.abs(coef_2_ref - coef_dic_list_random_samp_experiment[j][0]["coeff"][1])/ np.abs(coef_2_ref) 

for k in range(len(tol_list)):
    for l in range(2):
        error_coef_1_random_samp[0, j] = np.abs(coef_1_ref - coef_dic_list_random_samp_experiment[k*+l][0]["coeff"][0])/ np.abs(coef_1_ref)    
        error_coef_2_random_samp[0, j] = np.abs(coef_2_ref - coef_dic_list_random_samp_experiment[k+l][0]["coeff"][1])/ np.abs(coef_2_ref)
    
    pass




filename_error_coef_1_DEIM = result_path + "/" + "error_coef_1_DEIM" + ".npy"
filename_error_coef_2_DEIM = result_path + "/" + "error_coef_2_DEIM" + ".npy"

filename_error_coef_1_random_samp = result_path + "/" + "error_coef_1_random_samp" + ".npy"
filename_error_coef_2_random_samp = result_path + "/" + "error_coef_2_random_samp" + ".npy"

# np.save(filename_error_coef_1_DEIM , error_coef_1_DEIM)
# np.save(filename_error_coef_2_DEIM , error_coef_2_DEIM)

# np.save(filename_error_coef_1_random_samp , error_coef_1_random_samp)
# np.save(filename_error_coef_2_random_samp , error_coef_2_random_samp)

#################################################
#################################################

plt.rcParams.update({"font.size": 20,})

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
line_width = 3

fig, axs = plt.subplots(1, 2, sharey=False, figsize=(9, 3.5))
axs[0].loglog(np.array(num_samples), error_coef_1_DEIM[0,:], 
                linewidth=line_width, linestyle="-", marker="*", color=colors[0])
axs[0].loglog(np.array(num_samples), error_coef_1_random_samp[0,:],
                linewidth=line_width, linestyle="--", marker="s", color=colors[1])
axs[0].set(xlabel="Number of samples")
axs[0].set(ylabel=r"Coefficient $u_{xx}$")

axs[1].loglog(
    np.array(num_samples),error_coef_2_DEIM[0,:],
    linewidth=line_width,
    linestyle="-",
    marker="*",
    color=colors[0],
    label=r"\texttt{DEIM} Sampling",)

axs[1].loglog(
    np.array(num_samples),error_coef_2_random_samp[0,:],
    linewidth=line_width,
    linestyle="--",
    marker="s",
    color=colors[1],
    label=r"\texttt{Random} Sampling",)

axs[1].set(xlabel="Number of samples")
axs[1].set(ylabel=r"Coefficients $u,\ u^3$")


#plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust the value as needed
#plt.suptitle("Relative Absolute Error ", fontsize=20)

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

# plt.savefig(
#     result_path + "/" + "relative_abs_error_" + "_sensitivity_analysis_" + ".png",
#     dpi=300,
#     bbox_inches="tight",
# )
# plt.savefig(
#     result_path + "/" + "relative_abs_error_" + "_sensitivity_analysis_" + ".pdf",
#     dpi=300,
#     bbox_inches="tight",
# )




