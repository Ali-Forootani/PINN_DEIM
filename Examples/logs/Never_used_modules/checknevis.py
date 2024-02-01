#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:00:47 2023

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

from PINN_DEIM.Functions.data_set_preparation import AllenCahnEquation, AllenCahnEquationEntireDomain

from PINN_DEIM.Functions.deep_simulation_framework import AllenCahnDeepModel
from PINN_DEIM.Functions.ensemble_trainer import EnsembleTrainer


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

test_data_size = 0.95



AllenCahn_instance = AllenCahnEquationEntireDomain(data)

(
    x_train_list,
    u_train_list,
    train_loaders,
    X_test,
    u_test,
) = AllenCahn_instance.prepare_data_random(test_data_size=0.95)






#######################################
#######################################


"""
AllenCahn_instance = AllenCahnEquation(data,
                                       time_indices = 101)

x_train_list, u_train_list, train_loaders, X_test, u_test = AllenCahn_instance.prepare_data_random(
    test_data_size=0.95)

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
"""

##################################################
##################################################


"""
AllenCahnDeepModel_instance = AllenCahnDeepModel(
                 in_features= 2,
                 out_features=1,
                 hidden_features_str=128, 
                 hidden_layers=3,
                 learning_rate_inr=1e-5
    ) 

models_list, optim_adam, scheduler = AllenCahnDeepModel_instance.run()

    
model_str_1 = models_list[0]
model_str_f1 = models_list[1]
model_str_2 = models_list[2]
model_str_f2 = models_list[3]

num_epochs = 100


coefficients_1_list = []
coefficients_f1_list = []
coefficients_2_list = []
coefficients_f2_list = []


coefs_noise_lists_s_1 = []
coefs_noise_lists_f_1 = []
coefs_noise_lists_s_2 = []
coefs_noise_lists_f_2 = []


for i in range (2):
    
    AllenCahn_instance_2 = AllenCahnEquation(data,
                                           time_indices = 101, noise_level = i * 0.01)

    
    x_train_list, u_train_list, train_loaders, X_test, u_test = AllenCahn_instance_2.prepare_data_random(
        test_data_size=0.95)

   
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
    
    
########################################
########################################
    
    
    Ensem_Train_inst = EnsembleTrainer(
        model_str_1,
        model_str_f1,
        model_str_2,
        model_str_f2,
        num_epochs=num_epochs,
        optim_adam=optim_adam,
        scheduler=scheduler,
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
    
    
    my_dict = {"noise_level": i * 0.01,
               "coef_s_tr1": coef_s_tr1,
               "loss_tot_s1": loss_tot_s1
        }
    
    
    coefs_noise_lists_s_1.extend(
        {"noise_level": i * 0.01,
               "coef_s_tr1": coef_f_tr1,
               "loss_tot_s1": loss_tot_f1
        })
    
    coefs_noise_lists_f_1.extend(
        {"noise_level": i * 0.01,
               "coef_s_tr1": coef_s_tr2,
               "loss_tot_s1": loss_tot_s2
        }
        )
    
    coefs_noise_lists_s_2.extend(my_dict)
    
    coefs_noise_lists_f_2.extend(my_dict)
    

    
###################################################
###################################################
###################################################    
    
    
    
    loss_total = []
    
    loss_tot_s1 = []
    loss_tot_f1 = []
    loss_tot_s2 = []
    loss_tot_f2 = []
    
    
    coef_s_1 = []
    coef_f_1 = []
    coef_s_2 = []
    coef_f_2 = []
    
    
    ensemble_s_1 = LoaderProcessor(
        model_str_1, features_calc_AC, calculate_theta_AC, loss_func_AC
        )
    
    ensemble_f_1 = LoaderProcessor(
        model_str_f1, features_calc_AC, calculate_theta_AC, loss_func_AC
        )
    
    ensemble_s_2 = LoaderProcessor(
        model_str_2, features_calc_AC, calculate_theta_AC, loss_func_AC
        )
    
    ensemble_f_2 = LoaderProcessor(
        model_str_f2, features_calc_AC, calculate_theta_AC, loss_func_AC
        )
    
    
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

        loss_f2 = 0
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
    loss_tot_s2 = np.array(loss_tot_s2)
    loss_tot_f2 = np.array(loss_tot_f2)
    
    
    print(coefficients_f2)
    print(coefficients_f1)
    print(coefficients_2)
    print(coefficients_1)
    
    
    coefficients_1_list.append(coefficients_1)
    coefficients_f1_list.append(coefficients_f1)
    
    coefficients_2_list.append(coefficients_1)
    coefficients_f2_list.append(coefficients_f1)
    
    """
    
    
    
    
    
    
    
    
    
    
    
##############################    

"""
from PINN_DEIM.Functions.ensemble_trainer import EnsembleTrainer

EnsembleTrainer_instance = EnsembleTrainer(model_str_1,
                                           model_str_f1,
                                           model_str_2,
                                           model_str_f2,
                                           num_epochs=100,
                                           optim_adam=optim_adam,
                                           scheduler=scheduler)

loss_func_list, coeff_track_list = EnsembleTrainer_instance.train_func(train_loader_1,
                                                                       train_loader_f1,
                                                                       train_loader_2,
                               train_loader_f2)

plt.semilogy(loss_tot_s1)
plt.semilogy(loss_total)

coef_s_tr1=np.reshape(np.array(coeff_track_list[0]),(num_epochs, coeff_track_list[0][0].shape[0]))
coef_f_tr1=np.reshape(np.array(coeff_track_list[1]),(num_epochs, coeff_track_list[0][0].shape[0]))

coef_s_tr2=np.reshape(np.array(coeff_track_list[2]),(num_epochs, coeff_track_list[0][0].shape[0]))
coef_f_tr2=np.reshape(np.array(coeff_track_list[3]),(num_epochs, coeff_track_list[0][0].shape[0]))
"""


#############################    
    
"""
x_train_list, u_train_list_noisy, train_loaders_noisy, X_test, u_test_noisy = AllenCahn_instance_2.prepare_data_random(
    test_data_size=0.95)

X_str_train_1 = x_train_list[0]
X_str_train_f1 = x_train_list[1]
X_str_train_2 = x_train_list[2]
X_str_train_f2 = x_train_list[3]

u_str_train_1_noisy = u_train_list_noisy[0]
u_str_train_f1_noisy = u_train_list_noisy[1]
u_str_train_2_noisy = u_train_list_noisy[2]
u_str_train_f2_noisy = u_train_list_noisy[3]

train_loader_1 = train_loaders_noisy[0]
train_loader_f1 = train_loaders_noisy[1]
train_loader_2 = train_loaders_noisy[2]
train_loader_f2 = train_loaders_noisy[3]
"""

#######################################
#######################################


#u_star_noisy = u_star + noise * np.std(u_star) * np.random.randn(
#    u_star.shape[0], u_star.shape[1])

"""
noise=0
learning_rate_inr = 1e-5
hidden_features_str = 128
hidden_features = 64
hidden_layers = 3
num_epochs = 1500
prec = 1 - test_data_size

AllenCahnDeepModel_instance = AllenCahnDeepModel(
                 in_features= 2,
                 out_features=1,
                 hidden_features_str=128, 
                 hidden_layers=3,
                 learning_rate_inr=1e-5
    ) 

models_list, optimizer, scheduler = AllenCahnDeepModel_instance.run()

print(models_list)
print(optimizer)
print(scheduler)
"""


