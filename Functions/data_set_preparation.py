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
#CUDA support


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod
from PINN_DEIM.Functions.DEIM_class import DEIM


######################################################
######################################################

class DataPreparation(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def prepare_data_random(self, test_data_size):
        pass

    @abstractmethod
    def prepare_data_DEIM(self):
        pass

#####################################################
#####################################################


class AllenCahnEquation(DataPreparation):
    def __init__(self, data, time_indices, noise_level = None):
        
        """
        loading the AC model data set and
        spliting the time domain into 2,
        Args: 
            data: Allen Cahn data set
            time_indices: the index of time that we want to split our domain
        """
        
        super().__init__(data)
        self.time_indices = time_indices
        self.string_f = None
        # self.test_data_size = test_data_size
        
        self.t = self.data["tt"].flatten()
        self.t_1 = self.data["tt"].flatten()[0 : int(self.time_indices), None]
        self.t_2 = self.data["tt"].flatten()[int(self.time_indices) :, None]

        self.x = self.data["x"].flatten()[:, None]
        self.Exact = np.real(self.data["uu"])

        self.X_1, self.T_1 = np.meshgrid(self.x, self.t_1, indexing="ij")
        self.Exact_1 = self.Exact[:, 0 : self.X_1.shape[1]]

        self.X_2, self.T_2 = np.meshgrid(self.x, self.t_2, indexing="ij")
        self.Exact_2 = self.Exact[:, self.X_1.shape[1] + 1 :]
        
        self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")
        
        if noise_level is not None:
            noise_effect = noise_level * np.std(self.Exact)
            self.Exact = np.real(self.data["uu"]) + noise_effect
            self.Exact_1 = self.Exact_1 + noise_effect
            self.Exact_2 = self.Exact_2 + noise_effect
    
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        u_trains = []
        x_trains = []
        
        
        (
            X_str_train_1,
            u_str_train_1,
            X_str_train_f1,
            u_str_train_f1,
        ) = self.train_test_split(self.X_1, self.T_1, self.Exact_1, test_data_size)

        (
            X_str_train_2,
            u_str_train_2,
            X_str_train_f2,
            u_str_train_f2,
        ) = self.train_test_split(self.X_2, self.T_2, self.Exact_2, test_data_size)

        batch_size_1 = X_str_train_1.shape[0]

        train_loader_1 = self.data_loader(X_str_train_1, u_str_train_1, batch_size_1)
        train_loader_f1 = self.data_loader(X_str_train_f1, u_str_train_f1, batch_size_1)
        train_loader_2 = self.data_loader(X_str_train_2, u_str_train_2, batch_size_1)
        train_loader_f2 = self.data_loader(X_str_train_f2, u_str_train_f2, batch_size_1)

        x_trains = [X_str_train_1, X_str_train_f1, X_str_train_2, X_str_train_f2]

        u_trains = [u_str_train_1, u_str_train_f1, u_str_train_2, u_str_train_f2]

        train_loaders = [
            train_loader_1,
            train_loader_f1,
            train_loader_2,
            train_loader_f2,
        ]
        
        X_test_c, u_test_c = self.test_dataset_entire_domain(self.X, self.T, self.Exact)
        
       

        return x_trains, u_trains, train_loaders, X_test_c, u_test_c

    def prepare_data_DEIM(self, t_div_s1, t_div_f1,
                              t_div_s2, t_div_f2,
                              n_basis_s1, n_basis_f1,
                              n_basis_s2, n_basis_f2, tolerance):
        """
        t = self.data["tt"].flatten()
        t_1 = self.data["tt"].flatten()[0 : int(self.time_indices), None]
        t_2 = self.data["tt"].flatten()[int(self.time_indices) :, None]
        
        x = self.data["x"].flatten()[:, None]
        Exact = np.real(self.data["uu"])

        X_1, T_1 = np.meshgrid(x, t_1, indexing= "ij")
        Exact_1 = Exact[:,0:X_1.shape[1]]

        X_2, T_2 = np.meshgrid(x, t_2, indexing= "ij")
        Exact_2 = Exact[:,X_1.shape[1]:]
        """
        
        X_str_train_1, u_str_train_1 = self.DEIM_exec(self.x, self.t_1, self.Exact_1, t_div_s1, n_basis_s1, tolerance)
        X_str_train_f1, u_str_train_f1 = self.DEIM_exec( self.x, self.t_1, self.Exact_1, t_div_f1, n_basis_f1, tolerance)
        X_str_train_2, u_str_train_2 = self.DEIM_exec( self.x, self.t_2, self.Exact_2, t_div_s2, n_basis_s2, tolerance)
        X_str_train_f2, u_str_train_f2 = self.DEIM_exec( self.x, self.t_2, self.Exact_2, t_div_f2, n_basis_f2, tolerance)
        
        
        batch_size_1 = X_str_train_1.shape[0]
        batch_size_f1 = X_str_train_f1.shape[0]
        batch_size_2 = X_str_train_2.shape[0]
        batch_size_f2 = X_str_train_f2.shape[0]

        train_loader_1 = self.data_loader(X_str_train_1, u_str_train_1, batch_size_1)
        train_loader_f1 = self.data_loader(X_str_train_f1, u_str_train_f1, batch_size_f1)
        train_loader_2 = self.data_loader(X_str_train_2, u_str_train_2, batch_size_2)
        train_loader_f2 = self.data_loader(X_str_train_f2, u_str_train_f2, batch_size_f2)
        
        x_trains = [X_str_train_1, X_str_train_f1, X_str_train_2, X_str_train_f2]

        u_trains = [u_str_train_1, u_str_train_f1, u_str_train_2, u_str_train_f2]

        train_loaders = [
            train_loader_1,
            train_loader_f1,
            train_loader_2,
            train_loader_f2,
        ]
        
        
        return x_trains, u_trains, train_loaders, 

    def train_test_split(self, X, T, Exact, test_data_size):
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]
        X_f = np.hstack((X_star, u_star))

        X_str_train, X_test, u_str_train, u_test = train_test_split(
            X_star, u_star, test_size=test_data_size, random_state=42
        )
        X_str_train_f, X_test, u_str_train_f, u_test = train_test_split(
            X_test, u_test, test_size=test_data_size, random_state=42
        )

        return X_str_train, u_str_train, X_str_train_f, u_str_train_f
    
    
    def test_dataset_entire_domain(self, X, T, Exact_c, test_data_size=0.01):
        X_c = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_c = Exact_c.flatten()[:, None]
        
        X_train_c, X_test_c, u_train_c, u_test_c = train_test_split(
            X_c, u_c, test_size=test_data_size, random_state=42
        )
        
        return X_test_c, u_test_c
    
    def DEIM_exec(self, X, T, Exact, time_division,  num_basis, tolerance):
        deim_instance = DEIM(Exact, time_division, T, X, tolerance, num_basis)
        S_s, T_s, U_s = deim_instance.execute()
        coords = deim_instance.coords
        #X_star = coords_1
        #u_star = U_s
        X_str_train = coords
        u_str_train = U_s
        
        return X_str_train, u_str_train
    
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader

    # @staticmethod
    def save_string(
        self,
        noise=0,
        learning_rate_inr=1e-5,
        hidden_features_str=128,
        hidden_features=64,
        hidden_layers=3,
        num_epochs=1500,
        prec=0.02,
        random_sampling = True
    ):
        string_root = (
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
        
        if random_sampling == True:
            self.string_f = string_root + "random_sampling"
        else:
            self.string_f = string_root + "DEIM_sampling" 
            
        return self.string_f

    def file_path_saving(self, result_path):
        filename_loss_funcs = (
            result_path + "/" + "Loss_functions" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        
        ############
        
        """     To be implemented in the following
        suffixes = ["s1", "s2", "f1", "f2"]
        filename_dict = {}

        for suffix in suffixes:
            filename_dict[suffix] = result_path + "/" + "Loss_collect_" + suffix + self.string_f + ".npy"
        """
        
        
        filename_l_s1 = (
            result_path + "/" + "Loss_collect_s1" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        filename_l_s2 = (
            result_path + "/" + "Loss_collect_s2" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved

        filename_l_f1 = (
            result_path + "/" + "Loss_collect_f1" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved

        filename_l_f2 = (
            result_path + "/" + "Loss_collect_f2" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        
        ############        
        """        To be implemeneted in the following way
        coefficients = ["coef_s_tr1", "coef_f_tr1", "coef_s_tr2", "coef_f_tr2"]
        filename_coef_dict = {}
        for coeff in coefficients:
            filename_coef_dict[coeff] = result_path + "/" + coeff + self.string_f + ".npy"
        """
        
        filename_coef_s1 = (
            result_path + "/" + "coef_s_tr1" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved

        filename_coef_f1 = (
            result_path + "/" + "coef_f_tr1" + self.string_f + ".npy"
        )  ## path where t

        filename_coef_s2 = (
            result_path + "/" + "coef_s_tr2" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved

        filename_coef_f2 = (
            result_path + "/" + "coef_f_tr2" + self.string_f + ".npy"
        )  ## path where t


        """
        model_names = ["model_str_1", "model_str_f1", "model_str_2", "model_str_f2"]
        filename_model_dict = {}

        for name in model_names:
            filename_model_dict[name] = result_path + "/" + name + self.string_f + ".pt"
        """


        filename_model_s1 = (
            result_path + "/" + "model_str_1" + self.string_f + ".pt"
        )  ## path where the primary SIREN network data is saved

        filename_model_f1 = (
            result_path + "/" + "model_str_f1" + self.string_f + ".pt"
        )  ## path where the pr

        filename_model_s2 = (
            result_path + "/" + "model_str_2" + self.string_f + ".pt"
        )  ## path where the primary SIREN network data is saved

        filename_model_f2 = (
            result_path + "/" + "model_str_f2" + self.string_f + ".pt"
        )  ## path where the pr

        filename_data = result_path + "/" + "data_set" + self.string_f  ##

        return (
            filename_loss_funcs,
            filename_l_s1,
            filename_l_s2,
            filename_l_f1,
            filename_l_f2,
            filename_coef_s1,
            filename_l_s2,
            filename_l_f1,
            filename_l_f2,
            filename_coef_s1,
            filename_coef_f1,
            filename_coef_s2,
            filename_coef_f2,
            filename_model_s1,
            filename_model_f1,
            filename_model_s2,
            filename_model_f2,
            filename_data,
        )

#############################################
#############################################



class AllenCahnEquationEntireDomain(DataPreparation):
    def __init__(self, data, noise_level = None):
        
        """
        loading the AC model data set and
        spliting the time domain into 2,
        Args: 
            data: Allen Cahn data set
        """
        
        super().__init__(data)
        self.string_f = None
        self.t = self.data["tt"].flatten()
        self.x = self.data["x"].flatten()[:, None]
        self.Exact = np.real(self.data["uu"])
        self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")
        
        if noise_level is not None:
            noise_effect = noise_level * np.std(self.Exact)
            self.Exact = np.real(self.data["uu"]) + noise_effect
            
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        u_trains = []
        x_trains = []
        
        
        (
            X_str_train,
            u_str_train,
            X_str_test,
            u_str_test,
        ) = self.train_test_split(self.X, self.T, self.Exact, test_data_size)
        
        batch_size = X_str_train.shape[0]
        train_loader = self.data_loader(X_str_train, u_str_train, batch_size)
        x_trains = [X_str_train]
        u_trains = [u_str_train]

        train_loaders = [
            train_loader,
        ]
        
        X_test_c, u_test_c = self.test_dataset_entire_domain(self.X, self.T, self.Exact)
        
        return x_trains, u_trains, train_loaders, X_str_test, u_str_test

    def prepare_data_DEIM(self, t_div,
                              n_basis,
                              deim_tolerance
                              ):
        X_str_train, u_str_train = self.DEIM_exec(self.x, self.t, self.Exact, t_div, n_basis, deim_tolerance)
       
        batch_size = X_str_train.shape[0]
        
        train_loader = self.data_loader(X_str_train, u_str_train, batch_size)
        
        x_trains = [X_str_train]
        u_trains = [u_str_train]

        train_loaders = [
            train_loader
        ]
        
        return x_trains, u_trains, train_loaders, 

    def train_test_split(self, X, T, Exact, test_data_size):
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]
        
        X_str_train, X_test, u_str_train, u_test = train_test_split(
            X_star, u_star, test_size=test_data_size, random_state=42
        )
        return X_str_train, u_str_train, X_test, u_test
    
    
    def test_dataset_entire_domain(self, X, T, Exact_c, test_data_size=0.01):
        X_c = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_c = Exact_c.flatten()[:, None]
        X_train_c, X_test_c, u_train_c, u_test_c = train_test_split(
            X_c, u_c, test_size=test_data_size, 
        )
        return X_test_c, u_test_c
    
    
    def DEIM_exec(self, X, T, Exact, time_division, num_basis, deim_tolerance):
        deim_instance = DEIM(Exact, time_division, T, X, deim_tolerance, num_basis)
        S_s, T_s, U_s = deim_instance.execute()
        coords = deim_instance.coords
        #X_star = coords_1
        #u_star = U_s
        X_str_train = coords
        u_str_train = U_s
        
        return X_str_train, u_str_train
    
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader

    # @staticmethod
    def save_string(
        self,
        noise=0,
        learning_rate_inr=1e-5,
        hidden_features_str=128,
        hidden_features=64,
        hidden_layers=3,
        num_epochs=1500,
        prec=0.02,
        random_sampling = True
    ):
        string_root = (
            "_LSfit_lr_"
            + str(learning_rate_inr)
            + "_hf_phi_"
            + str(hidden_layers)
            + "_layers_"
            + "_hidden_features_"
            + str(int(hidden_features_str))
            + "_ep_"
            + str(int(num_epochs))
            + "_noise_"
            + str(int(100 * noise))
            #+ "_perc_"
            #+ str(int(100 * prec))
            + "_entire_domain_"
            )
        
        if random_sampling == True:
            self.string_f = string_root + "_perc_" + str(int(100 * prec)) + "random_sampling"
        else:
            self.string_f = string_root + "DEIM_sampling" 
            
        return self.string_f

    def file_path_saving(self, result_path):
        filename_loss_funcs = (
            result_path + "/" + "Loss_functions" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        
        filename_coef = (
            result_path + "/" + "coef_s_tr" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
         
        filename_model = (
            result_path + "/" + "model_str_" + self.string_f + ".pt"
        )  ## path where the primary SIREN network data is saved
        
        filename_data = result_path + "/" + "data_set" + self.string_f  ##

        return (
            filename_loss_funcs,
            filename_coef,
            filename_model,
            filename_data,
        )

###################################################
###################################################
###################################################




class AllenCahnEquationDomainDecomposition(DataPreparation):
    def __init__(self, data, time_indices, noise_level = None):
        
        """
        loading the AC model data set and
        spliting the time domain into 2,
        Args: 
            data: Allen Cahn data set
            time_indices: the index of time that we want to split our domain
        """
        
        super().__init__(data)
        self.time_indices = time_indices
        self.string_f = None
        # self.test_data_size = test_data_size
        
        self.t = self.data["tt"].flatten()
        self.t_1 = self.data["tt"].flatten()[0 : int(self.time_indices), None]
        self.t_2 = self.data["tt"].flatten()[int(self.time_indices) :, None]

        self.x = self.data["x"].flatten()[:, None]
        self.Exact = np.real(self.data["uu"])

        self.X_1, self.T_1 = np.meshgrid(self.x, self.t_1, indexing="ij")
        self.Exact_1 = self.Exact[:, 0 : self.X_1.shape[1]]

        self.X_2, self.T_2 = np.meshgrid(self.x, self.t_2, indexing="ij")
        self.Exact_2 = self.Exact[:, self.X_1.shape[1] + 1 :]
        
        self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")
        
        if noise_level is not None:
            noise_effect = noise_level * np.std(self.Exact)
            self.Exact = np.real(self.data["uu"]) + noise_effect
            self.Exact_1 = self.Exact_1 + noise_effect
            self.Exact_2 = self.Exact_2 + noise_effect
    
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        u_trains = []
        x_trains = []
        
        
        (
            X_str_train_1,
            u_str_train_1,
            X_str_train_f1,
            u_str_train_f1,
        ) = self.train_test_split(self.X_1, self.T_1, self.Exact_1, test_data_size)

        (
            X_str_train_2,
            u_str_train_2,
            X_str_train_f2,
            u_str_train_f2,
        ) = self.train_test_split(self.X_2, self.T_2, self.Exact_2, test_data_size)

        batch_size_1 = X_str_train_1.shape[0]

        train_loader_1 = self.data_loader(X_str_train_1, u_str_train_1, batch_size_1)
        #train_loader_f1 = self.data_loader(X_str_train_f1, u_str_train_f1, batch_size_1)
        train_loader_2 = self.data_loader(X_str_train_2, u_str_train_2, batch_size_1)
        #train_loader_f2 = self.data_loader(X_str_train_f2, u_str_train_f2, batch_size_1)

        x_trains = [X_str_train_1, 
                    #X_str_train_f1,
                    X_str_train_2,
                    #X_str_train_f2
                    ]

        u_trains = [u_str_train_1,
                    #u_str_train_f1,
                    u_str_train_2,
                    #u_str_train_f2
                    ]

        train_loaders = [
            train_loader_1,
            #train_loader_f1,
            train_loader_2,
            #train_loader_f2,
        ]
        
        X_test_c, u_test_c = self.test_dataset_entire_domain(self.X, self.T, self.Exact)
        
       

        return x_trains, u_trains, train_loaders, X_test_c, u_test_c

    def prepare_data_DEIM(self, t_div, 
                          n_basis,
                          deim_tolerance
                          ):
        
       
        t_div_s1 = t_div[0]
           #t_div_f1 = t_div[1]
        t_div_s2 = t_div[1]
           #t_div_f2 = t_div[3]
           
        n_basis_s1 = n_basis
           #n_basis_f1 = n_basis[1]
        n_basis_s2 = n_basis
           #n_basis_f2 = n_basis[3]
        
        
        X_str_train_1, u_str_train_1 = self.DEIM_exec(self.x, self.t_1, self.Exact_1, t_div_s1, n_basis_s1, deim_tolerance)
        #X_str_train_f1, u_str_train_f1 = self.DEIM_exec( self.x, self.t_1, self.Exact_1, t_div_f1, n_basis_f1, tolerance)
        X_str_train_2, u_str_train_2 = self.DEIM_exec( self.x, self.t_2, self.Exact_2, t_div_s2, n_basis_s2, deim_tolerance)
        #X_str_train_f2, u_str_train_f2 = self.DEIM_exec( self.x, self.t_2, self.Exact_2, t_div_f2, n_basis_f2, tolerance)
        
        
        batch_size_1 = X_str_train_1.shape[0]
        #batch_size_f1 = X_str_train_f1.shape[0]
        batch_size_2 = X_str_train_2.shape[0]
        #batch_size_f2 = X_str_train_f2.shape[0]

        train_loader_1 = self.data_loader(X_str_train_1, u_str_train_1, batch_size_1)
        #train_loader_f1 = self.data_loader(X_str_train_f1, u_str_train_f1, batch_size_f1)
        train_loader_2 = self.data_loader(X_str_train_2, u_str_train_2, batch_size_2)
        #train_loader_f2 = self.data_loader(X_str_train_f2, u_str_train_f2, batch_size_f2)
        
        x_trains = [X_str_train_1, 
                    #X_str_train_f1, 
                    X_str_train_2, 
                    #X_str_train_f2
                    ]

        u_trains = [u_str_train_1,
                    #u_str_train_f1,
                    u_str_train_2,
                    #u_str_train_f2
                    ]

        train_loaders = [
            train_loader_1,
            #train_loader_f1,
            train_loader_2,
            #train_loader_f2,
        ]
        
        
        return x_trains, u_trains, train_loaders, 

    def train_test_split(self, X, T, Exact, test_data_size):
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]
        X_f = np.hstack((X_star, u_star))

        X_str_train, X_test, u_str_train, u_test = train_test_split(
            X_star, u_star, test_size=test_data_size, random_state=42
        )
        X_str_train_f, X_test, u_str_train_f, u_test = train_test_split(
            X_test, u_test, test_size=test_data_size, random_state=42
        )

        return X_str_train, u_str_train, X_str_train_f, u_str_train_f
    
    
    def test_dataset_entire_domain(self, X, T, Exact_c, test_data_size=0.01):
        X_c = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_c = Exact_c.flatten()[:, None]
        
        X_train_c, X_test_c, u_train_c, u_test_c = train_test_split(
            X_c, u_c, test_size=test_data_size, random_state=42
        )
        
        return X_test_c, u_test_c
    
    
    def DEIM_exec(self, X, T, Exact, time_division,  num_basis, tolerance,):
        
        
        """
        self.x, self.t_1, self.Exact_1, t_div_s1, n_basis_s1, deim_tolerance
        """
        
        deim_instance = DEIM(Exact, time_division, T, X, tolerance, num_basis)
        S_s, T_s, U_s = deim_instance.execute()
        coords = deim_instance.coords
        #X_star = coords_1
        #u_star = U_s
        X_str_train = coords
        u_str_train = U_s
        
        return X_str_train, u_str_train
    
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader

    # @staticmethod
    def save_string(
        self,
        noise=0,
        learning_rate_inr=1e-5,
        hidden_features_str=128,
        hidden_features=64,
        hidden_layers=3,
        num_epochs=1500,
        prec=0.02,
        random_sampling = True
    ):
        string_root = (
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
        
        if random_sampling == True:
            self.string_f = string_root + "random_sampling"
        else:
            self.string_f = string_root + "DEIM_sampling" 
            
        return self.string_f

    def file_path_saving(self, result_path):
        filename_loss_funcs = (
            result_path + "/" + "Loss_functions" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        
        ############
        
        """     To be implemented in the following """
        suffixes = ["s1", "s2"]
        filename_dict = {}

        for suffix in suffixes:
            filename_dict[suffix] = result_path + "/" + "Loss_collect_" + suffix + self.string_f + ".npy"
        
        
       
        
        ############        
        """        To be implemeneted in the following way """
        coefficients = ["coef_s_tr1", "coef_s_tr2"]
        filename_coef_dict = {}
        for coeff in coefficients:
            filename_coef_dict[coeff] = result_path + "/" + coeff + self.string_f + ".npy"
        
       
        
        model_names = ["model_str_1", "model_str_2"]
        filename_model_dict = {}

        for name in model_names:
            filename_model_dict[name] = result_path + "/" + name + self.string_f + ".pt"
        
       
        
        return (
            filename_loss_funcs,
            filename_dict,
            filename_coef_dict,
            filename_model_dict
        )



#####################################################
#####################################################
#####################################################

class KDVEquationEntireDomain(DataPreparation):
    def __init__(self, data, noise_level = None):
        
        """
        loading the AC model data set and
        spliting the time domain into 2,
        Args: 
            data: Allen Cahn data set
            
        t_data = data["t"].flatten()
        x_data = data["x"].flatten()
        u_data = data["usol"]
            
            
        """
        
        super().__init__(data)
        self.string_f = None
        self.t = self.data["t"].flatten()
        t_min = self.t.min(axis=0)
        t_max = self.t.max(axis=0)
        self.t = (self.t - t_min) / (t_max - t_min)
        
        self.x = self.data["x"].flatten()[:, None]
        self.x = (self.x - self.x.min(axis=0)) / (self.x.max(axis=0) - self.x.min(axis=0))
        self.x = self.x * (1 - (-1) ) - 1
        
        self.Exact = np.real(self.data["usol"])
        self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")
        
        if noise_level is not None:
            noise_effect = noise_level * np.std(self.Exact)
            self.Exact = np.real(self.data["usol"]) + noise_effect
            
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        u_trains = []
        x_trains = []
        
        
        (
            X_str_train,
            u_str_train,
            X_str_test,
            u_str_test,
        ) = self.train_test_split(self.X, self.T, self.Exact, test_data_size)
        
        batch_size = X_str_train.shape[0]
        train_loader = self.data_loader(X_str_train, u_str_train, batch_size)
        x_trains = [X_str_train]
        u_trains = [u_str_train]

        train_loaders = [
            train_loader,
        ]
        
        X_test_c, u_test_c = self.test_dataset_entire_domain(self.X, self.T, self.Exact)
        
        return x_trains, u_trains, train_loaders, X_str_test, u_str_test

    def prepare_data_DEIM(self, t_div,
                              n_basis,
                              deim_tolerance
                              ):
        X_str_train, u_str_train = self.DEIM_exec(self.x, self.t, self.Exact, t_div, n_basis, deim_tolerance)
       
        batch_size = X_str_train.shape[0]
        
        train_loader = self.data_loader(X_str_train, u_str_train, batch_size)
        
        x_trains = [X_str_train]
        u_trains = [u_str_train]

        train_loaders = [
            train_loader
        ]
        
        return x_trains, u_trains, train_loaders, 

    def train_test_split(self, X, T, Exact, test_data_size):
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]
        
        X_str_train, X_test, u_str_train, u_test = train_test_split(
            X_star, u_star, test_size=test_data_size, random_state=42
        )
        return X_str_train, u_str_train, X_test, u_test
    
    
    def test_dataset_entire_domain(self, X, T, Exact_c, test_data_size=0.01):
        X_c = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_c = Exact_c.flatten()[:, None]
        X_train_c, X_test_c, u_train_c, u_test_c = train_test_split(
            X_c, u_c, test_size=test_data_size, random_state=42
        )
        return X_test_c, u_test_c
    
    
    def DEIM_exec(self, X, T, Exact, time_division, num_basis, deim_tolerance):
        deim_instance = DEIM(Exact, time_division, T, X, deim_tolerance, num_basis)
        S_s, T_s, U_s = deim_instance.execute()
        coords = deim_instance.coords
        #X_star = coords_1
        #u_star = U_s
        X_str_train = coords
        u_str_train = U_s
        
        return X_str_train, u_str_train
    
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader

    # @staticmethod
    def save_string(
        self,
        noise=0,
        learning_rate_inr=1e-5,
        hidden_features_str=128,
        hidden_features=64,
        hidden_layers=3,
        num_epochs=1500,
        prec=0.02,
        random_sampling = True
    ):
        string_root = (
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
            #+ "_perc_"
            #+ str(int(100 * prec))
            + "_entire_domain_"
            )
        
        if random_sampling == True:
            self.string_f = string_root + "_perc_" + str(int(100 * prec)) + "random_sampling"
        else:
            self.string_f = string_root + "DEIM_sampling" 
            
        return self.string_f

    def file_path_saving(self, result_path):
        filename_loss_funcs = (
            result_path + "/" + "Loss_functions" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        
        filename_coef = (
            result_path + "/" + "coef_s_tr1" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
         
        filename_model = (
            result_path + "/" + "model_str_1" + self.string_f + ".pt"
        )  ## path where the primary SIREN network data is saved
        
        filename_data = result_path + "/" + "data_set" + self.string_f  ##

        return (
            filename_loss_funcs,
            filename_coef,
            filename_model,
            filename_data,
        )

###############################################
###############################################


class KDVEquation(DataPreparation):
    def __init__(self, data, time_indices, noise_level = None):
        
        """
        loading the AC model data set and
        spliting the time domain into 2,
        Args: 
            data: Allen Cahn data set
            time_indices: the index of time that we want to split our domain
            t_data = data["t"].flatten()
            x_data = data["x"].flatten()
            u_data = data["usol"]
            
            t_data = data["t"].flatten()
            x_data = data["x"].flatten()
            u_data = data["usol"]

            #u_data=data["U_exact"]

            t = data["t"].flatten()
            x = data["x"].flatten()

            ### use the below if the data is not normalized already
            min_t = t.min()
            max_t = t.max()

            t_std = (t - t.min(axis=0)) / (t.max(axis=0) - t.min(axis=0))
            t_norm = t_std * (1 - (-0)) -0
            t_scaling = 1/(t.max(axis=0) - t.min(axis=0))


            x_std = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
            x_norm = x_std * (1 - (-1)) -1
            x_scaling = 2/(x.max(axis=0) - x.min(axis=0))
        """
        
        super().__init__(data)
        self.time_indices = time_indices
        self.string_f = None
        # self.test_data_size = test_data_size
        
        
        
        
        self.t = self.data["t"].flatten()
        
        t_min = self.t.min(axis=0)
        t_max = self.t.max(axis=0)
        
        self.t = (self.t - t_min) / (t_max - t_min)
        
        #####################################
        
        self.t_1 = self.data["t"].flatten()[0 : int(self.time_indices), None]
        self.t_1 = (self.t_1 - t_min) / (t_max - t_min)
        
        self.t_2 = self.data["t"].flatten()[int(self.time_indices) :, None]
        self.t_2 = (self.t_2 - t_min ) / (t_max - t_min)
        
        self.x = self.data["x"].flatten()[:, None]
        self.x = (self.x - self.x.min(axis=0)) / (self.x.max(axis=0) - self.x.min(axis=0))
        self.x = self.x * (1 - (-1) ) - 1
        
        
        self.Exact = np.real(self.data["usol"])

        self.X_1, self.T_1 = np.meshgrid(self.x, self.t_1, indexing="ij")
        self.Exact_1 = self.Exact[:, 0 : self.X_1.shape[1]]

        self.X_2, self.T_2 = np.meshgrid(self.x, self.t_2, indexing="ij")
        self.Exact_2 = self.Exact[:, self.X_1.shape[1] + 1 :]
        
        self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")
        
        if noise_level is not None:
            noise_effect = noise_level * np.std(self.Exact)
            self.Exact = np.real(self.data["usol"]) + noise_effect
            self.Exact_1 = self.Exact_1 + noise_effect
            self.Exact_2 = self.Exact_2 + noise_effect
    
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        u_trains = []
        x_trains = []
        
        
        (
            X_str_train_1,
            u_str_train_1,
            X_str_train_f1,
            u_str_train_f1,
        ) = self.train_test_split(self.X_1, self.T_1, self.Exact_1, test_data_size)

        (
            X_str_train_2,
            u_str_train_2,
            X_str_train_f2,
            u_str_train_f2,
        ) = self.train_test_split(self.X_2, self.T_2, self.Exact_2, test_data_size)

        batch_size_1 = X_str_train_1.shape[0]

        train_loader_1 = self.data_loader(X_str_train_1, u_str_train_1, batch_size_1)
        train_loader_f1 = self.data_loader(X_str_train_f1, u_str_train_f1, batch_size_1)
        train_loader_2 = self.data_loader(X_str_train_2, u_str_train_2, batch_size_1)
        train_loader_f2 = self.data_loader(X_str_train_f2, u_str_train_f2, batch_size_1)

        x_trains = [X_str_train_1, X_str_train_f1, X_str_train_2, X_str_train_f2]

        u_trains = [u_str_train_1, u_str_train_f1, u_str_train_2, u_str_train_f2]

        train_loaders = [
            train_loader_1,
            train_loader_f1,
            train_loader_2,
            train_loader_f2,
        ]
        
        X_test_c, u_test_c = self.test_dataset_entire_domain(self.X, self.T, self.Exact)
        
       

        return x_trains, u_trains, train_loaders, X_test_c, u_test_c

    def prepare_data_DEIM(self, t_div,
                              n_basis,
                              deim_tolerance):
        
        
        t_div_s1 = t_div[0]
        t_div_f1 = t_div[1]
        t_div_s2 = t_div[2]
        t_div_f2 = t_div[3]
        
        n_basis_s1 = n_basis[0]
        n_basis_f1 = n_basis[1]
        n_basis_s2 = n_basis[2]
        n_basis_f2 = n_basis[3]
        """
        t = self.data["tt"].flatten()
        t_1 = self.data["tt"].flatten()[0 : int(self.time_indices), None]
        t_2 = self.data["tt"].flatten()[int(self.time_indices) :, None]
        
        x = self.data["x"].flatten()[:, None]
        Exact = np.real(self.data["uu"])

        X_1, T_1 = np.meshgrid(x, t_1, indexing= "ij")
        Exact_1 = Exact[:,0:X_1.shape[1]]

        X_2, T_2 = np.meshgrid(x, t_2, indexing= "ij")
        Exact_2 = Exact[:,X_1.shape[1]:]
        """
        
        X_str_train_1, u_str_train_1 = self.DEIM_exec(self.x, self.t_1, self.Exact_1, t_div_s1, n_basis_s1, deim_tolerance)
        X_str_train_f1, u_str_train_f1 = self.DEIM_exec( self.x, self.t_1, self.Exact_1, t_div_f1, n_basis_f1, deim_tolerance)
        X_str_train_2, u_str_train_2 = self.DEIM_exec( self.x, self.t_2, self.Exact_2, t_div_s2, n_basis_s2, deim_tolerance)
        X_str_train_f2, u_str_train_f2 = self.DEIM_exec( self.x, self.t_2, self.Exact_2, t_div_f2, n_basis_f2, deim_tolerance)
        
        
        batch_size_1 = X_str_train_1.shape[0]
        batch_size_f1 = X_str_train_f1.shape[0]
        batch_size_2 = X_str_train_2.shape[0]
        batch_size_f2 = X_str_train_f2.shape[0]

        train_loader_1 = self.data_loader(X_str_train_1, u_str_train_1, batch_size_1)
        train_loader_f1 = self.data_loader(X_str_train_f1, u_str_train_f1, batch_size_f1)
        train_loader_2 = self.data_loader(X_str_train_2, u_str_train_2, batch_size_2)
        train_loader_f2 = self.data_loader(X_str_train_f2, u_str_train_f2, batch_size_f2)
        
        x_trains = [X_str_train_1, X_str_train_f1, X_str_train_2, X_str_train_f2]

        u_trains = [u_str_train_1, u_str_train_f1, u_str_train_2, u_str_train_f2]

        train_loaders = [
            train_loader_1,
            train_loader_f1,
            train_loader_2,
            train_loader_f2,
        ]
        
        
        return x_trains, u_trains, train_loaders, 

    def train_test_split(self, X, T, Exact, test_data_size):
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]
        X_f = np.hstack((X_star, u_star))

        X_str_train, X_test, u_str_train, u_test = train_test_split(
            X_star, u_star, test_size=test_data_size, random_state=42
        )
        X_str_train_f, X_test, u_str_train_f, u_test = train_test_split(
            X_test, u_test, test_size=test_data_size, random_state=42
        )

        return X_str_train, u_str_train, X_str_train_f, u_str_train_f
    
    
    def test_dataset_entire_domain(self, X, T, Exact_c, test_data_size=0.01):
        X_c = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_c = Exact_c.flatten()[:, None]
        
        X_train_c, X_test_c, u_train_c, u_test_c = train_test_split(
            X_c, u_c, test_size=test_data_size, random_state=42
        )
        
        return X_test_c, u_test_c
    
    
    def DEIM_exec(self, X, T, Exact, time_division, num_basis, deim_tolerance):
        deim_instance = DEIM(Exact, time_division, T, X, deim_tolerance, num_basis)
        S_s, T_s, U_s = deim_instance.execute()
        coords = deim_instance.coords
        #X_star = coords_1
        #u_star = U_s
        X_str_train = coords
        u_str_train = U_s
        
        return X_str_train, u_str_train
    
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader

    # @staticmethod
    def save_string(
        self,
        noise=0,
        learning_rate_inr=1e-5,
        hidden_features_str=128,
        hidden_features=64,
        hidden_layers=3,
        num_epochs=1500,
        prec=0.02,
        random_sampling = True
    ):
        string_root = (
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
        
        if random_sampling == True:
            self.string_f = string_root + "random_sampling"
        else:
            self.string_f = string_root + "DEIM_sampling" 
            
        return self.string_f

    def file_path_saving(self, result_path):
        filename_loss_funcs = (
            result_path + "/" + "Loss_functions" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        
        ############
        
        ###    To be implemented in the following
        suffixes = ["s1", "s2", "f1", "f2"]
        filename_dict = {}
        
        #string_f = "ali"
        
        for suffix in suffixes:
            filename_dict[suffix] = result_path + "/" + "Loss_collect_" + suffix + self.string_f + ".npy"
        
        
        """
        filename_l_s1 = (
            result_path + "/" + "Loss_collect_s1" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        filename_l_s2 = (
            result_path + "/" + "Loss_collect_s2" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved

        filename_l_f1 = (
            result_path + "/" + "Loss_collect_f1" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved

        filename_l_f2 = (
            result_path + "/" + "Loss_collect_f2" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        """
        
        ############        
        ###        To be implemeneted in the following way
        
        coefficients = ["coef_s_tr1", "coef_f_tr1", "coef_s_tr2", "coef_f_tr2"]
        filename_coef_dict = {}
        for coeff in coefficients:
            filename_coef_dict[coeff] = result_path + "/" + coeff + self.string_f + ".npy"
        
        
        """
        filename_coef_s1 = (
            result_path + "/" + "coef_s_tr1" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved

        filename_coef_f1 = (
            result_path + "/" + "coef_f_tr1" + self.string_f + ".npy"
        )  ## path where t

        filename_coef_s2 = (
            result_path + "/" + "coef_s_tr2" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved

        filename_coef_f2 = (
            result_path + "/" + "coef_f_tr2" + self.string_f + ".npy"
        )  ## path where t

        """
        model_names = ["model_str_1", "model_str_f1", "model_str_2", "model_str_f2"]
        filename_model_dict = {}

        for name in model_names:
            filename_model_dict[name] = result_path + "/" + name + self.string_f + ".pt"
        
        """
        filename_model_s1 = (
            result_path + "/" + "model_str_1" + self.string_f + ".pt"
        )  ## path where the primary SIREN network data is saved

        filename_model_f1 = (
            result_path + "/" + "model_str_f1" + self.string_f + ".pt"
        )  ## path where the pr

        filename_model_s2 = (
            result_path + "/" + "model_str_2" + self.string_f + ".pt"
        )  ## path where the primary SIREN network data is saved

        filename_model_f2 = (
            result_path + "/" + "model_str_f2" + self.string_f + ".pt"
        )  ## path where the pr

        filename_data = result_path + "/" + "data_set" + self.string_f  ##
        """
        
        return (
            filename_loss_funcs,
            filename_dict,
            filename_coef_dict,
            filename_model_dict
            #filename_l_s1,
            #filename_l_s2,
            #filename_l_f1,
            #filename_l_f2,
            #filename_coef_s1,
            #filename_l_s2,
            #filename_l_f1,
            #filename_l_f2,
            #filename_coef_s1,
            #filename_coef_f1,
            #filename_coef_s2,
            #filename_coef_f2,
            #filename_model_s1,
            #filename_model_f1,
            #filename_model_s2,
            #filename_model_f2,
            #filename_data,
        )

#####################################################
#####################################################


class KDVEquationDomainDecomposition(DataPreparation):
    def __init__(self, data, time_indices, noise_level = None):
        
        """
        loading the AC model data set and
        spliting the time domain into 2,
        Args: 
            data: Allen Cahn data set
            time_indices: the index of time that we want to split our domain
            t_data = data["t"].flatten()
            x_data = data["x"].flatten()
            u_data = data["usol"]
            
            t_data = data["t"].flatten()
            x_data = data["x"].flatten()
            u_data = data["usol"]

            #u_data=data["U_exact"]

            t = data["t"].flatten()
            x = data["x"].flatten()

            ### use the below if the data is not normalized already
            min_t = t.min()
            max_t = t.max()

            t_std = (t - t.min(axis=0)) / (t.max(axis=0) - t.min(axis=0))
            t_norm = t_std * (1 - (-0)) -0
            t_scaling = 1/(t.max(axis=0) - t.min(axis=0))


            x_std = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
            x_norm = x_std * (1 - (-1)) -1
            x_scaling = 2/(x.max(axis=0) - x.min(axis=0))
        """
        
        super().__init__(data)
        self.time_indices = time_indices
        self.string_f = None
        # self.test_data_size = test_data_size
        
        
        
        
        self.t = self.data["t"].flatten()
        
        t_min = self.t.min(axis=0)
        t_max = self.t.max(axis=0)
        
        self.t = (self.t - t_min) / (t_max - t_min)
        
        #####################################
        
        self.t_1 = self.data["t"].flatten()[0 : int(self.time_indices), None]
        self.t_1 = (self.t_1 - t_min) / (t_max - t_min)
        
        self.t_2 = self.data["t"].flatten()[int(self.time_indices) :, None]
        self.t_2 = (self.t_2 - t_min ) / (t_max - t_min)
        
        self.x = self.data["x"].flatten()[:, None]
        self.x = (self.x - self.x.min(axis=0)) / (self.x.max(axis=0) - self.x.min(axis=0))
        self.x = self.x * (1 - (-1) ) - 1
        
        
        self.Exact = np.real(self.data["usol"])

        self.X_1, self.T_1 = np.meshgrid(self.x, self.t_1, indexing="ij")
        self.Exact_1 = self.Exact[:, 0 : self.X_1.shape[1]]

        self.X_2, self.T_2 = np.meshgrid(self.x, self.t_2, indexing="ij")
        self.Exact_2 = self.Exact[:, self.X_1.shape[1] + 1 :]
        
        self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")
        
        if noise_level is not None:
            noise_effect = noise_level * np.std(self.Exact)
            self.Exact = np.real(self.data["usol"]) + noise_effect
            self.Exact_1 = self.Exact_1 + noise_effect
            self.Exact_2 = self.Exact_2 + noise_effect
    
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        u_trains = []
        x_trains = []
        
        
        (
            X_str_train_1,
            u_str_train_1,
            X_str_train_f1,
            u_str_train_f1,
        ) = self.train_test_split(self.X_1, self.T_1, self.Exact_1, test_data_size)

        (
            X_str_train_2,
            u_str_train_2,
            X_str_train_f2,
            u_str_train_f2,
        ) = self.train_test_split(self.X_2, self.T_2, self.Exact_2, test_data_size)

        batch_size_1 = X_str_train_1.shape[0]

        train_loader_1 = self.data_loader(X_str_train_1, u_str_train_1, batch_size_1)        
        train_loader_2 = self.data_loader(X_str_train_2, u_str_train_2, batch_size_1)

        x_trains = [X_str_train_1, X_str_train_2]
        u_trains = [u_str_train_1, u_str_train_2]

        train_loaders = [
            train_loader_1,
            train_loader_2,
        ]
        
        X_test_c, u_test_c = self.test_dataset_entire_domain(self.X, self.T, self.Exact)
        
        return x_trains, u_trains, train_loaders, X_test_c, u_test_c

    def prepare_data_DEIM(self, t_div,
                              n_basis,
                              deim_tolerance):
        
        t_div_s1 = t_div[0]
        t_div_s2 = t_div[1]
        
        n_basis_s1 = n_basis[0]
        n_basis_s2 = n_basis[1]
        
        """
        t = self.data["tt"].flatten()
        t_1 = self.data["tt"].flatten()[0 : int(self.time_indices), None]
        t_2 = self.data["tt"].flatten()[int(self.time_indices) :, None]
        
        x = self.data["x"].flatten()[:, None]
        Exact = np.real(self.data["uu"])

        X_1, T_1 = np.meshgrid(x, t_1, indexing= "ij")
        Exact_1 = Exact[:,0:X_1.shape[1]]

        X_2, T_2 = np.meshgrid(x, t_2, indexing= "ij")
        Exact_2 = Exact[:,X_1.shape[1]:]
        """
        
       
        
        X_str_train_1, u_str_train_1 = self.DEIM_exec(self.x, self.t_1, self.Exact_1,
                                                      t_div_s1, n_basis_s1,
                                                      deim_tolerance)

        X_str_train_2, u_str_train_2 = self.DEIM_exec( self.x, self.t_2, self.Exact_2,
                                                      t_div_s2, n_basis_s2,
                                                      deim_tolerance)
        
        
        batch_size_1 = X_str_train_1.shape[0]
        batch_size_2 = X_str_train_2.shape[0]
        

        train_loader_1 = self.data_loader(X_str_train_1, u_str_train_1, batch_size_1)
        train_loader_2 = self.data_loader(X_str_train_2, u_str_train_2, batch_size_2)
        
        
        x_trains = [X_str_train_1, X_str_train_2]
        u_trains = [u_str_train_1, u_str_train_2]

        train_loaders = [
            train_loader_1,
            train_loader_2,
        ]
        
        return x_trains, u_trains, train_loaders, 

    def train_test_split(self, X, T, Exact, test_data_size):
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]
        X_f = np.hstack((X_star, u_star))

        X_str_train, X_test, u_str_train, u_test = train_test_split(
            X_star, u_star, test_size=test_data_size, random_state=42
        )
        X_str_train_f, X_test, u_str_train_f, u_test = train_test_split(
            X_test, u_test, test_size=test_data_size, random_state=42
        )

        return X_str_train, u_str_train, X_str_train_f, u_str_train_f
    
    
    def test_dataset_entire_domain(self, X, T, Exact_c, test_data_size=0.01):
        X_c = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_c = Exact_c.flatten()[:, None]
        
        X_train_c, X_test_c, u_train_c, u_test_c = train_test_split(
            X_c, u_c, test_size=test_data_size, random_state=42
        )
        
        return X_test_c, u_test_c
    
    
    def DEIM_exec(self, X, T, Exact, time_division, num_basis, deim_tolerance):
        deim_instance = DEIM(Exact, time_division, T, X, deim_tolerance, num_basis)
        S_s, T_s, U_s = deim_instance.execute()
        coords = deim_instance.coords
        #X_star = coords_1
        #u_star = U_s
        X_str_train = coords
        u_str_train = U_s
        
        return X_str_train, u_str_train
    
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader

    # @staticmethod
    def save_string(
        self,
        noise=0,
        learning_rate_inr=1e-5,
        hidden_features_str=128,
        hidden_features=64,
        hidden_layers=3,
        num_epochs=1500,
        prec=0.02,
        random_sampling = True
    ):
        string_root = (
            "_LSfit_lr_"
            + str(learning_rate_inr)
            + "_hf_phi_"
            + str(hidden_features)
            + "_layers_"
            + str(int(hidden_features_str))
            + "_ep_"
            + str(int(num_epochs))
            + "_noise_"
            + str(int(100 * noise))
            + "_perc_"
            + str(int(100 * prec))
            )
        
        if random_sampling == True:
            self.string_f = string_root + "random_sampling"
        else:
            self.string_f = string_root + "DEIM_sampling" 
            
        return self.string_f

    def file_path_saving(self, result_path):
        filename_loss_funcs = (
            result_path + "/" + "Loss_functions" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        
        ############
        
        ###    To be implemented in the following
        suffixes = ["s1", "s2"]
        filename_dict = {}
        
        #string_f = "ali"
        
        for suffix in suffixes:
            filename_dict[suffix] = result_path + "/" + "Loss_collect_" + suffix + self.string_f + ".npy"
        
        
        
        ############        
        ###        To be implemeneted in the following way
        
        coefficients = ["coef_s_tr1", "coef_s_tr2"]
        filename_coef_dict = {}
        for coeff in coefficients:
            filename_coef_dict[coeff] = result_path + "/" + coeff + self.string_f + ".npy"
        
        
        model_names = ["model_str_1", "model_str_2"]
        filename_model_dict = {}

        for name in model_names:
            filename_model_dict[name] = result_path + "/" + name + self.string_f + ".pt"
        
       
        
        return (
            filename_loss_funcs,
            filename_dict,
            filename_coef_dict,
            filename_model_dict)












######################################################
######################################################

class BurgerEquationEntireDomain(DataPreparation):
    def __init__(self, data, noise_level = None):
        
        """
        loading the AC model data set and
        spliting the time domain into 2,
        Args: 
            data: Burger data set
            
        t_data = data["t"].flatten()
        x_data = data["x"].flatten()
        u_data = data["usol"]
        """
        
        super().__init__(data)
        self.string_f = None
        self.t = self.data["t"].flatten()
        t_min = self.t.min(axis=0)
        t_max = self.t.max(axis=0)
        self.t = (self.t - t_min) / (t_max - t_min)
        
        self.x = self.data["x"].flatten()[:, None]
        self.x = (self.x - self.x.min(axis=0)) / (self.x.max(axis=0) - self.x.min(axis=0))
        self.x = self.x * (1 - (-1) ) - 1
        
        self.Exact = np.real(self.data["usol"])
        self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")
        
        if noise_level is not None:
            noise_effect = noise_level * np.std(self.Exact)
            self.Exact = np.real(self.data["usol"]) + noise_effect
            
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        u_trains = []
        x_trains = []
        
        
        (
            X_str_train,
            u_str_train,
            X_str_test,
            u_str_test,
        ) = self.train_test_split(self.X, self.T, self.Exact, test_data_size)
        
        batch_size = X_str_train.shape[0]
        train_loader = self.data_loader(X_str_train, u_str_train, batch_size)
        x_trains = [X_str_train]
        u_trains = [u_str_train]

        train_loaders = [
            train_loader,
        ]
        
        X_test_c, u_test_c = self.test_dataset_entire_domain(self.X, self.T, self.Exact)
        
        return x_trains, u_trains, train_loaders, X_str_test, u_str_test

    def prepare_data_DEIM(self, t_div,
                              n_basis,
                              deim_tolerance
                              ):
        X_str_train, u_str_train = self.DEIM_exec(self.x, self.t, self.Exact, t_div, n_basis, deim_tolerance)
       
        batch_size = X_str_train.shape[0]
        
        train_loader = self.data_loader(X_str_train, u_str_train, batch_size)
        
        x_trains = [X_str_train]
        u_trains = [u_str_train]

        train_loaders = [
            train_loader
        ]
        
        return x_trains, u_trains, train_loaders, 

    def train_test_split(self, X, T, Exact, test_data_size):
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]
        
        X_str_train, X_test, u_str_train, u_test = train_test_split(
            X_star, u_star, test_size=test_data_size, random_state=42
        )
        return X_str_train, u_str_train, X_test, u_test
    
    
    def test_dataset_entire_domain(self, X, T, Exact_c, test_data_size=0.01):
        X_c = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_c = Exact_c.flatten()[:, None]
        X_train_c, X_test_c, u_train_c, u_test_c = train_test_split(
            X_c, u_c, test_size=test_data_size, random_state=42
        )
        return X_test_c, u_test_c
    
    
    def DEIM_exec(self, X, T, Exact, time_division, num_basis, deim_tolerance):
        deim_instance = DEIM(Exact, time_division, T, X, deim_tolerance, num_basis)
        S_s, T_s, U_s = deim_instance.execute()
        coords = deim_instance.coords
        #X_star = coords_1
        #u_star = U_s
        X_str_train = coords
        u_str_train = U_s
        
        return X_str_train, u_str_train
    
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader

    # @staticmethod
    def save_string(
        self,
        noise=0,
        learning_rate_inr=1e-5,
        hidden_features_str=128,
        hidden_features=64,
        hidden_layers=3,
        num_epochs=1500,
        prec=0.02,
        random_sampling = True
    ):
        string_root = (
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
            #+ "_perc_"
            #+ str(int(100 * prec))
            + "_entire_domain_"
            )
        
        if random_sampling == True:
            self.string_f = string_root + "_perc_" + str(int(100 * prec)) + "random_sampling"
        else:
            self.string_f = string_root + "DEIM_sampling" 
            
        return self.string_f

    def file_path_saving(self, result_path):
        filename_loss_funcs = (
            result_path + "/" + "Loss_functions" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        
        filename_coef = (
            result_path + "/" + "coef_s_tr1" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
         
        filename_model = (
            result_path + "/" + "model_str_1" + self.string_f + ".pt"
        )  ## path where the primary SIREN network data is saved
        
        filename_data = result_path + "/" + "data_set" + self.string_f  ##

        return (
            filename_loss_funcs,
            filename_coef,
            filename_model,
            filename_data,
        )


################################################
################################################
################################################



class BurgerEquationDomainDecomposition(DataPreparation):
    def __init__(self, data, time_indices, noise_level = None):
        
        """
        loading the Burger model data set and
        spliting the time domain into 2,
        Args: 
            data: Allen Cahn data set
            time_indices: the index of time that we want to split our domain
           
            self.string_f = None
            self.t = self.data["t"].flatten()
            t_min = self.t.min(axis=0)
            t_max = self.t.max(axis=0)
            self.t = (self.t - t_min) / (t_max - t_min)
            
            self.x = self.data["x"].flatten()[:, None]
            self.x = (self.x - self.x.min(axis=0)) / (self.x.max(axis=0) - self.x.min(axis=0))
            self.x = self.x * (1 - (-1) ) - 1
            
            self.Exact = np.real(self.data["usol"])
            self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")
            
            if noise_level is not None:
                noise_effect = noise_level * np.std(self.Exact)
                self.Exact = np.real(self.data["usol"]) + noise_effect
        """
        
        super().__init__(data)
        self.time_indices = time_indices
        self.string_f = None
        # self.test_data_size = test_data_size
        
        
        
        
        self.t = self.data["t"].flatten()
        
        t_min = self.t.min(axis=0)
        t_max = self.t.max(axis=0)
        
        self.t = (self.t - t_min) / (t_max - t_min)
        
        #####################################
        
        self.t_1 = self.data["t"].flatten()[0 : int(self.time_indices), None]
        self.t_1 = (self.t_1 - t_min) / (t_max - t_min)
        
        self.t_2 = self.data["t"].flatten()[int(self.time_indices) :, None]
        self.t_2 = (self.t_2 - t_min ) / (t_max - t_min)
        
        self.x = self.data["x"].flatten()[:, None]
        self.x = (self.x - self.x.min(axis=0)) / (self.x.max(axis=0) - self.x.min(axis=0))
        self.x = self.x * (1 - (-1) ) - 1
        
        
        self.Exact = np.real(self.data["usol"])

        self.X_1, self.T_1 = np.meshgrid(self.x, self.t_1, indexing="ij")
        self.Exact_1 = self.Exact[:, 0 : self.X_1.shape[1]]

        self.X_2, self.T_2 = np.meshgrid(self.x, self.t_2, indexing="ij")
        self.Exact_2 = self.Exact[:, self.X_1.shape[1] + 1 :]
        
        self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")
        
        if noise_level is not None:
            noise_effect = noise_level * np.std(self.Exact)
            self.Exact = np.real(self.data["usol"]) + noise_effect
            self.Exact_1 = self.Exact_1 + noise_effect
            self.Exact_2 = self.Exact_2 + noise_effect
    
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        u_trains = []
        x_trains = []
        
        
        (
            X_str_train_1,
            u_str_train_1,
            X_str_train_f1,
            u_str_train_f1,
        ) = self.train_test_split(self.X_1, self.T_1, self.Exact_1, test_data_size)

        (
            X_str_train_2,
            u_str_train_2,
            X_str_train_f2,
            u_str_train_f2,
        ) = self.train_test_split(self.X_2, self.T_2, self.Exact_2, test_data_size)

        batch_size_1 = X_str_train_1.shape[0]

        train_loader_1 = self.data_loader(X_str_train_1, u_str_train_1, batch_size_1)        
        train_loader_2 = self.data_loader(X_str_train_2, u_str_train_2, batch_size_1)

        x_trains = [X_str_train_1, X_str_train_2]
        u_trains = [u_str_train_1, u_str_train_2]

        train_loaders = [
            train_loader_1,
            train_loader_2,
        ]
        
        X_test_c, u_test_c = self.test_dataset_entire_domain(self.X, self.T, self.Exact)
        
        return x_trains, u_trains, train_loaders, X_test_c, u_test_c

    def prepare_data_DEIM(self, t_div,
                              n_basis,
                              deim_tolerance):
        
        t_div_s1 = t_div[0]
        t_div_s2 = t_div[1]
        
        n_basis_s1 = n_basis[0]
        n_basis_s2 = n_basis[1]
        
        deim_tolerance_1 = deim_tolerance[0]
        
        deim_tolerance_2 = deim_tolerance[1]
        
        """
        t = self.data["tt"].flatten()
        t_1 = self.data["tt"].flatten()[0 : int(self.time_indices), None]
        t_2 = self.data["tt"].flatten()[int(self.time_indices) :, None]
        
        x = self.data["x"].flatten()[:, None]
        Exact = np.real(self.data["uu"])

        X_1, T_1 = np.meshgrid(x, t_1, indexing= "ij")
        Exact_1 = Exact[:,0:X_1.shape[1]]

        X_2, T_2 = np.meshgrid(x, t_2, indexing= "ij")
        Exact_2 = Exact[:,X_1.shape[1]:]
        """
        
       
        
        X_str_train_1, u_str_train_1 = self.DEIM_exec(self.x, self.t_1, self.Exact_1,
                                                      t_div_s1, n_basis_s1,
                                                      deim_tolerance_1)

        X_str_train_2, u_str_train_2 = self.DEIM_exec( self.x, self.t_2, self.Exact_2,
                                                      t_div_s2, n_basis_s2,
                                                      deim_tolerance_2)
        
        
        batch_size_1 = X_str_train_1.shape[0]
        batch_size_2 = X_str_train_2.shape[0]
        

        train_loader_1 = self.data_loader(X_str_train_1, u_str_train_1, batch_size_1)
        train_loader_2 = self.data_loader(X_str_train_2, u_str_train_2, batch_size_2)
        
        
        x_trains = [X_str_train_1, X_str_train_2]
        u_trains = [u_str_train_1, u_str_train_2]

        train_loaders = [
            train_loader_1,
            train_loader_2,
        ]
        
        return x_trains, u_trains, train_loaders, 

    def train_test_split(self, X, T, Exact, test_data_size):
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]
        X_f = np.hstack((X_star, u_star))

        X_str_train, X_test, u_str_train, u_test = train_test_split(
            X_star, u_star, test_size=test_data_size, random_state=42
        )
        X_str_train_f, X_test, u_str_train_f, u_test = train_test_split(
            X_test, u_test, test_size=test_data_size, random_state=42
        )

        return X_str_train, u_str_train, X_str_train_f, u_str_train_f
    
    
    def test_dataset_entire_domain(self, X, T, Exact_c, test_data_size=0.01):
        X_c = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_c = Exact_c.flatten()[:, None]
        
        X_train_c, X_test_c, u_train_c, u_test_c = train_test_split(
            X_c, u_c, test_size=test_data_size, random_state=42
        )
        
        return X_test_c, u_test_c
    
    
    def DEIM_exec(self, X, T, Exact, time_division, num_basis, deim_tolerance):
        deim_instance = DEIM(Exact, time_division, T, X, deim_tolerance, num_basis)
        S_s, T_s, U_s = deim_instance.execute()
        coords = deim_instance.coords
        #X_star = coords_1
        #u_star = U_s
        X_str_train = coords
        u_str_train = U_s
        
        return X_str_train, u_str_train
    
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader

    # @staticmethod
    def save_string(
        self,
        noise=0,
        learning_rate_inr=1e-5,
        hidden_features_str=128,
        hidden_features=64,
        hidden_layers=3,
        num_epochs=1500,
        prec=0.02,
        random_sampling = True
    ):
        string_root = (
            "_LSfit_lr_"
            + str(learning_rate_inr)
            + "_hf_phi_"
            + str(hidden_features)
            + "_layers_"
            + str(int(hidden_features_str))
            + "_ep_"
            + str(int(num_epochs))
            + "_noise_"
            + str(int(100 * noise))
            + "_perc_"
            + str(int(100 * prec))
            )
        
        if random_sampling == True:
            self.string_f = string_root + "random_sampling"
        else:
            self.string_f = string_root + "DEIM_sampling" 
            
        return self.string_f

    def file_path_saving(self, result_path):
        filename_loss_funcs = (
            result_path + "/" + "Loss_functions" + self.string_f + ".npy"
        )  ## path where the loss data for every epoch is saved
        
        ############
        
        ###    To be implemented in the following
        suffixes = ["s1", "s2"]
        filename_dict = {}
        
        #string_f = "ali"
        
        for suffix in suffixes:
            filename_dict[suffix] = result_path + "/" + "Loss_collect_" + suffix + self.string_f + ".npy"
        
        
        
        ############        
        ###        To be implemeneted in the following way
        
        coefficients = ["coef_s_tr1", "coef_s_tr2"]
        filename_coef_dict = {}
        for coeff in coefficients:
            filename_coef_dict[coeff] = result_path + "/" + coeff + self.string_f + ".npy"
        
        
        model_names = ["model_str_1", "model_str_2"]
        filename_model_dict = {}

        for name in model_names:
            filename_model_dict[name] = result_path + "/" + name + self.string_f + ".pt"
        
       
        
        return (
            filename_loss_funcs,
            filename_dict,
            filename_coef_dict,
            filename_model_dict)




























