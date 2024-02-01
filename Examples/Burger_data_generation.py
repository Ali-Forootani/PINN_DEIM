

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
from PINN_DEIM.Functions.utils import leastsquares_fit
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

from PINN_DEIM.Functions.data_set_preparation import BurgerEquationEntireDomain
from PINN_DEIM.Functions.deep_simulation_framework import PDEDeepModelEntireDomain

from PINN_DEIM.Functions.ensemble_trainer import EnsembleTrainer, Trainer
from matplotlib.ticker import FuncFormatter
from PINN_DEIM.Functions.library import library_deriv, features_calc_Burger, calculate_theta_Burger

from PINN_DEIM.Functions.utils import (
    leastsquares_fit, loss_func_Burger)


warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data = scipy.io.loadmat(root_dir +'/data/burgers.mat')


########################################################
######## Making directory and file saving settings


noise=0
learning_rate_inr=1e-5
hidden_features_str=128
hidden_layers=3


string_f = ("_Burger_"
            
            + "comparing_"
            
            + "different_time_division_"
            
            + "different_tol_"
            
            )

### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "./logs/result_DEIM_Burger_1d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.makedirs(result_path)

"""
third
"""
tol_list = np.logspace(-15, -4, 25)

"""
second
"""
tol_list = np.logspace(-13, -4, 20)

"""
first
"""
tol_list = np.logspace(-10, -2, 20)

####################################################################

coef_s_tr_list_DEIM = []
coef_dic_list_DEIM = []

coef_s_tr_list_random_samp = []
coef_dic_list_random_samp = []
num_samples = []

num_epochs = 1500

list_Burger_data_DEIM_t_div_tol = []



for j in range(1, 5):

    for i in range(len(tol_list)):
        
        
        Burger_DEIM = BurgerEquationEntireDomain(data)
        (x_train_list_DEIM, u_train_list_DEIM, train_loaders_DEIM)= Burger_DEIM.prepare_data_DEIM(
                                                    t_div= int(j),
                                                    n_basis=1,
                                                    deim_tolerance = tol_list[i])
                                                    
        X_str_train_DEIM = x_train_list_DEIM[0]
        u_str_train_DEIM = u_train_list_DEIM[0]
        
        
        train_loader_DEIM = train_loaders_DEIM[0]
    
        #print(X_str_train_DEIM.shape)
        num_samples.append( X_str_train_DEIM.shape[0] )
        
        
        Burger_DEIM_t_div_tol = {"t_div": j ,
                             "tol": i ,
                             "X_train_DEIM": X_str_train_DEIM,
                             "u_train_DEIM": u_str_train_DEIM,
                             "train_loader_DEIM": train_loader_DEIM,
                             "num_samples": X_str_train_DEIM.shape[0]
                             }
        
        list_Burger_data_DEIM_t_div_tol.append( Burger_DEIM_t_div_tol )
        
        
        BurgerDeepModel_DEIM = PDEDeepModelEntireDomain(
            in_features=2,
            out_features=1,
            hidden_features_str=128,
            hidden_layers=3,
            learning_rate_inr=1e-5,)

        models_list_DEIM, optim_adam_DEIM, scheduler_DEIM = BurgerDeepModel_DEIM.run()
        
        model_str_DEIM = models_list_DEIM[0]
        
        
        Train_DEIM = Trainer(
            model_str_DEIM,
            num_epochs= num_epochs,
            optim_adam=optim_adam_DEIM,
            scheduler=scheduler_DEIM,
            features_calc = features_calc_Burger,
            calculate_theta = calculate_theta_Burger,
            loss_func = loss_func_Burger
            )
        
        loss_func_list_DEIM, coeff_track_list_DEIM = Train_DEIM.train_func(
            train_loader_DEIM)
        
        coef_s_DEIM = coeff_track_list_DEIM[0]
        loss_tot_s_DEIM = loss_func_list_DEIM[0]
        coef_s_tr_DEIM = np.reshape(np.array(coef_s_DEIM), (num_epochs, coef_s_DEIM[0].shape[0]))
        
        coef_s_tr_list_DEIM.append({"t_div": j,
                                    "tol": tol_list[i],
                                    "num_samples": X_str_train_DEIM.shape[0],
                                    "coef_s_tr_DEIM": [coef_s_tr_DEIM]
                                    })
        
        coef_dic_list_DEIM.append( {"t_div": j,
                                    "tol": tol_list[i],
                                    "num_samples": X_str_train_DEIM.shape[0],
                                    "coeff": coef_s_tr_DEIM[-1]})
        
       
      

filename_coef_DEIM = result_path + "/" + "Burger_coef_dic_list_diff_tol_diff_tim_DEIM" + ".npy"
filename_data_t_v_tol_DEIM = result_path + "/" + "Burger_list_data_diff_tol_diff_tim_DEIM" + ".npy"
filename_coef_track_DEIM = result_path + "/" + "Burger_coef_track_dic_list_diff_tol_diff_tim_DEIM" + ".npy"


np.save(filename_coef_DEIM, coef_dic_list_DEIM)
np.save( filename_data_t_v_tol_DEIM , list_Burger_data_DEIM_t_div_tol)
np.save(filename_coef_track_DEIM, coef_s_tr_list_DEIM)



#######################################################
#######################################################


t = data["t"]
x = data["x"]
Exact = np.real(data["usol"])


t = data["t"].flatten()
t_min = t.min(axis=0)
t_max = t.max(axis=0)
t = (t - t_min) / (t_max - t_min)

x = data["x"].flatten()[:, None]
x = (x - x.min(axis=0)) / ( x.max(axis=0) - x.min(axis=0) )
x = x * (1 - (-1) ) - 1

Exact = np.real(data["usol"])
X, T = np.meshgrid( x, t, indexing="ij" )



min_num_samples = min(num_samples)
max_num_samples = max(num_samples)
step_size = ( max(num_samples) - min(num_samples) )/ 10
org_data_size = data["usol"].shape[0] * data["usol"].shape[1]

list_Burger_data_random_points = []








for k in range( min(num_samples) , max(num_samples) , int(step_size)):
    
    for l in range(1, 6):
        
        test_data_size = 1 - (k/org_data_size)
        Burger_random_samp = BurgerEquationEntireDomain(data)
        
        ##############################################
        
        
        
        X_star_random = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star_random = Exact.flatten()[:, None]

        X_str_train_random, X_test, u_str_train_random, u_test = train_test_split(
            X_star_random, u_star_random, test_size = test_data_size)
        
        X_torch = torch.tensor(X_str_train_random, requires_grad=True).float().to(device)
        Y_torch = torch.tensor(u_str_train_random).float().to(device)

        train_loader_random = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_torch, Y_torch), batch_size = X_str_train_random.shape[0], shuffle=True
        )
        
        
        Burger_Random_points = {
                             "X_train_Random": X_str_train_random,
                             "u_train_Random": u_str_train_random,
                             "train_loader_Random": train_loader_random,
                             "num_samples": X_str_train_random.shape[0]
                             }
        
        list_Burger_data_random_points.append( Burger_Random_points )
        #################################################
        
    
        BurgerDeepModel_random_samp = PDEDeepModelEntireDomain(
            in_features=2,
            out_features=1,
            hidden_features_str=128,
            hidden_layers=3,
            learning_rate_inr=1e-5,)
    
        models_list_random_samp, optim_adam_random_samp, scheduler_random_samp = BurgerDeepModel_random_samp.run()
    
        model_str_random_samp = models_list_random_samp[0]
    
     
        Train_inst = Trainer(
            model_str_random_samp,
            num_epochs=num_epochs,
            optim_adam=optim_adam_random_samp,
            scheduler=scheduler_random_samp,
            features_calc = features_calc_Burger,
            calculate_theta = calculate_theta_Burger,
            loss_func = loss_func_Burger
                        )
    
        loss_func_list_random_samp, coeff_track_list_random_samp = Train_inst.train_func(
            train_loader_random
            )
    
        coef_s_random_samp = coeff_track_list_random_samp[0]

        loss_tot_s_random_samp = loss_func_list_random_samp[0]

        coef_s_tr_random_samp = np.reshape(np.array(coef_s_random_samp),
                                           (num_epochs, coef_s_random_samp[0].shape[0]))
    
        coef_s_tr_list_random_samp.append(
                                            {"experiment_index": l,
                                             "num_samples": X_str_train_random.shape[0],
                                             "coef_s_tr_random_samp": [coef_s_tr_random_samp]
                                             }
                                            )
    
        
        coef_dic_list_random_samp.append( {"experiment_index": l,
                                           "num_samples": X_str_train_random.shape[0],
                                           "coeff": coef_s_tr_random_samp[-1]})


filename_coef_random_samp = result_path + "/" + "Burger_coef_dic_list_random_samp" + ".npy"
filename_data_random_points = result_path + "/" + "Burger_list_data_random_points" + ".npy"
filename_coef_track_random_samp = result_path + "/" + "Burger_coef_dic_track_list_random_samp" + ".npy"



np.save(filename_coef_random_samp, coef_dic_list_random_samp)
np.save(filename_data_random_points, Burger_Random_points)
np.save(filename_coef_track_random_samp, coef_s_tr_list_random_samp)


"""
to load the save array just use this
array_1 = np.load(filename_data_random_points, allow_pickle=True)
array_2 = np.load(filename_coef_random_samp, allow_pickle=True)
array_3 = np.load(filename_coef_track_random_samp, allow_pickle=True)
"""      
        














