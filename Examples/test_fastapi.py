#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:39:44 2023

@author: forootani
"""

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


from fastapi import FastAPI, File, UploadFile
import os
import numpy as np
import scipy.io
from pathlib import Path
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
import signal


############################################
############################################

# Function to gracefully shut down the background simulation
def shutdown_simulation():
    global stop_simulation
    stop_simulation = True
    # Implement code to stop the background simulation here
    print("Shutting down the simulation...")
    # Insert code to stop your simulation or release resources here

# Register a signal handler to catch SIGINT (Ctrl+C)
def signal_handler(sig, frame):
    print("Received Ctrl+C. Stopping the application gracefully...")
    shutdown_simulation()  # Call the shutdown function
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)




############################################
############################################

# Import your KDVEquationEntireDomain, PDEDeepModelEntireDomain, and Trainer classes here
# Make sure you have the required classes and functions defined

app = FastAPI()

@app.post("/run_kdv_code")
async def run_kdv_code(
    noise: int,
    learning_rate_inr: float,
    hidden_features_str: int,
    hidden_layers: int,
    num_epochs: int,
    data_file_path: str,  # User selects a file from a directory
    #tol_list = np.logspace(-10, -2, 20),
    root_dir = root_dir,
):
    try:
        
        data = scipy.io.loadmat(root_dir + data_file_path)
        
        tol_list = np.logspace(-4, -2, 1)
        
        #data = scipy.io.loadmat(os.path.join(root_dir, data_file_path))

        string_f = (
            "_KDV_sensitivity_DEIM_random_sampling_"
            + "_noise_" + str(int(noise)) 
            + "_learning_rate_inr_" + str(int(learning_rate_inr))
            + "_hidden_features_str_" + str(int(hidden_features_str))
            + "_hidden_layers_" + str(int(hidden_layers))
            + "_nim_epochs_" + str(int(num_epochs))
        )

        result_path = "./logs/result_DEIM_KDV_1d/results" + string_f
        p = Path(result_path)
        if not p.exists():
            os.makedirs(result_path)

        coef_s_tr_list_DEIM = []
        coef_dic_list_DEIM = []
        coef_s_tr_list_random_samp = []
        coef_dic_list_random_samp = []
        num_samples = []

        #num_epochs = 1500
        
        
        
        for i in range(len(tol_list)):
            # Your existing code for DEIM and random sampling here
            
            
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
            
            
            # Append the results to the respective lists
            
            
            filename_DEIM = result_path + "/" + "coef_dic_list_DEIM_fastapi" + ".npy"
            filename_random_samp = result_path + "/" + "coef_dic_list_random_samp_fastapi" + ".npy"

            np.save(filename_DEIM, coef_dic_list_DEIM)
            np.save(filename_random_samp, coef_dic_list_random_samp)
                        
            
        
        response_data = { 
            
             "coef_s_tr_list_DEIM": coef_s_tr_list_DEIM[-1][0][-1].tolist(),
             #"coef_dic_list_DEIM": coef_dic_list_DEIM,
             "coef_s_tr_list_random_samp": coef_s_tr_list_random_samp[-1][0][-1].tolist(),
             #"coef_dic_list_random_samp": coef_dic_list_random_samp[-1][0][-1].tolist(),
             "num_samples": num_samples, 
                                        }
        """
        # Convert numpy arrays to lists (JSON serializable)
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float32):
                return float(obj)
            return obj
        """
        
        #response_data = {k: convert_to_json_serializable(v) for k, v in response_data.items()}

        JSONResponse(content=response_data)
        return  response_data
    
            
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


"""
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
"""


























