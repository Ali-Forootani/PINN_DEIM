#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:35:53 2023
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
from PINN_DEIM.Functions.ensemble_loader import LoaderProcessor, process_loader ####################################
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

from PINN_DEIM.Functions.data_set_preparation import AllenCahnEquation
from PINN_DEIM.Functions.deep_simulation_framework import AllenCahnDeepModel

warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod

##################################

class TrainPDEs(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def train_func(self):
        pass


################################################
################################################



class EnsembleTrainer(TrainPDEs):
    def __init__(
        self,
        model_str_1,
        model_str_f1,
        model_str_2,
        model_str_f2,
        optim_adam,
        scheduler,
        features_calc,
        calculate_theta,
        loss_func,
        num_epochs=1500,
    ):
        """
        # Usage
        # Define your train loaders, features_calc_AC, calculate_theta_AC, loss_func_AC, etc.
        # Create optimizer and scheduler objects
        # Instantiate the EnsembleTrainer class
        # Call the train method on the instance
        
        # Example Usage:
        # ensemble_trainer = EnsembleTrainer(model_str_1, model_str_f1, model_str_2, model_str_f2, num_epochs, optim_adam, scheduler)
        # ensemble_trainer.train(train_loader_1, train_loader_f1, train_loader_2, train_loader_f2, features_calc_AC, calculate_theta_AC, loss_func_AC)
        """
        
        super().__init__()
        self.model_str_1 = model_str_1
        self.model_str_f1 = model_str_f1
        self.model_str_2 = model_str_2
        self.model_str_f2 = model_str_f2
        self.num_epochs = num_epochs
        self.optim_adam = optim_adam
        self.scheduler = scheduler
        self.calculate_theta = calculate_theta
        self.features_calc = features_calc
        self.loss_func = loss_func

        self.loss_total = []
        self.loss_tot_s1 = []
        self.loss_tot_f1 = []
        self.loss_tot_s2 = []
        self.loss_tot_f2 = []
        self.coef_s_1 = []
        self.coef_f_1 = []
        self.coef_s_2 = []
        self.coef_f_2 = []

    def train_func(
        self,
        train_loader_1,
        train_loader_f1,
        train_loader_2,
        train_loader_f2,
    ):
        
        loop = tqdm(range(self.num_epochs), leave=False)
        
        ## range(self.num_epochs)
        
        for epoch in loop:
            #tqdm.write(f"Epoch: {epoch}")

            loss_data_1 = 0
            loss_f1 = 0
            loss_data_2 = 0
            loss_f2 = 0

            start_time = time.time()

            ####################################################
            ensemble_s_1 = LoaderProcessor(
                self.model_str_1, self.features_calc, self.calculate_theta, self.loss_func
            )
            loss_data_1, coefficients_1 = ensemble_s_1(train_loader_1)
            self.coef_s_1.append(coefficients_1.cpu().detach().numpy())

            ####################################################
            ensemble_f_1 = LoaderProcessor(
                self.model_str_f1, self.features_calc, self.calculate_theta, self.loss_func
            )
            loss_f1, coefficients_f1 = ensemble_f_1(train_loader_f1)
            self.coef_f_1.append(coefficients_f1.cpu().detach().numpy())

            ####################################################
            ensemble_s_2 = LoaderProcessor(
                self.model_str_2, self.features_calc, self.calculate_theta, self.loss_func
            )
            loss_data_2, coefficients_2 = ensemble_s_2(train_loader_2)
            self.coef_s_2.append(coefficients_2.cpu().detach().numpy())

            ####################################################
            ensemble_f_2 = LoaderProcessor(
                self.model_str_f2, self.features_calc, self.calculate_theta, self.loss_func
            )
            loss_f2, coefficients_f2 = ensemble_f_2(train_loader_f2)
            self.coef_f_2.append(coefficients_f2.cpu().detach().numpy())

            # print(coefficients_f2)

            ####################################################
            loss = loss_data_1 + loss_data_2 + loss_f2 + loss_f1

            self.loss_total.append(loss.cpu().detach().numpy())
            self.loss_tot_s1.append(loss_data_1.cpu().detach().numpy())
            self.loss_tot_f1.append(loss_f1.cpu().detach().numpy())
            self.loss_tot_s2.append(loss_data_2.cpu().detach().numpy())
            self.loss_tot_f2.append(loss_f2.cpu().detach().numpy())

            self.optim_adam.zero_grad()
            loss.backward()
            self.optim_adam.step()

            # scheduler step
            self.scheduler.step()

        self.loss_total = np.array(self.loss_total)
        self.loss_tot_s1 = np.array(self.loss_tot_s1)
        self.loss_tot_f1 = np.array(self.loss_tot_f1)
        self.loss_tot_s2 = np.array(self.loss_tot_s2)
        self.loss_tot_f2 = np.array(self.loss_tot_f2)

        loss_func_list = [
            self.loss_tot_s1,
            self.loss_tot_f1,
            self.loss_tot_s2,
            self.loss_tot_f2,
        ]
        coeff_track_list = [self.coef_s_1, self.coef_f_1, self.coef_s_2, self.coef_f_2]

        return loss_func_list, coeff_track_list


##########################################################



class Trainer(TrainPDEs):
    def __init__(
        self,
        model_str,
        optim_adam,
        scheduler,
        features_calc,
        calculate_theta,
        loss_func,
        num_epochs=1500,
    ):
        """
        # Usage
        # Define your train loaders, features_calc_AC, calculate_theta_AC, loss_func_AC, etc.
        # Create optimizer and scheduler objects
        # Instantiate the EnsembleTrainer class
        # Call the train method on the instance
        
        # Example Usage:
        # ensemble_trainer = EnsembleTrainer(model_str, num_epochs, optim_adam, scheduler)
        # ensemble_trainer.train(train_loader, features_calc_AC, calculate_theta_AC, loss_func_AC)
        """
        
        super().__init__()
        self.model_str = model_str
        self.num_epochs = num_epochs
        self.optim_adam = optim_adam
        self.scheduler = scheduler
        self.calculate_theta = calculate_theta
        self.features_calc = features_calc
        self.loss_func = loss_func

        self.loss_total = []
        self.coef_s = []
        

    def train_func(
        self,
        train_loader,
    ):
        #loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        loop = tqdm(range(self.num_epochs), leave=False)
        
        for epoch in loop:
            #tqdm.write(f"Epoch: {epoch}")

            loss_data = 0
            start_time = time.time()

            ####################################################
            ensemble_s = LoaderProcessor(
                self.model_str, self.features_calc, self.calculate_theta, self.loss_func
            )
            loss_data, coefficients = ensemble_s(train_loader)
            self.coef_s.append(coefficients.cpu().detach().numpy())

            ####################################################
            loss = loss_data
            self.loss_total.append(loss.cpu().detach().numpy())
            self.optim_adam.zero_grad()
            loss.backward()
            self.optim_adam.step()

            # scheduler step
            self.scheduler.step()
            
           
            
            #loop.set_description(f"Epoch [{epoch}/{self.num_epochs}]")
            loop.set_postfix(
                training_loss=loss.item(),)
            

        self.loss_total = np.array(self.loss_total)
        

        loss_func_list = [
            self.loss_total,
            
        ]
        coeff_track_list = [self.coef_s]

        return loss_func_list, coeff_track_list


################################################
################################################


class EnsembleTrainerSplitDomain(TrainPDEs):
    def __init__(
        self,
        model_str_1,
        model_str_2,
        optim_adam,
        scheduler,
        features_calc,
        calculate_theta,
        loss_func,
        num_epochs=1500,
    ):
        """
        # Usage
        # Define your train loaders, features_calc_AC, calculate_theta_AC, loss_func_AC, etc.
        # Create optimizer and scheduler objects
        # Instantiate the EnsembleTrainer class
        # Call the train method on the instance
        
        # Example Usage:
        # ensemble_trainer = EnsembleTrainer(model_str_1, model_str_f1, model_str_2, model_str_f2, num_epochs, optim_adam, scheduler)
        # ensemble_trainer.train(train_loader_1, train_loader_f1, train_loader_2, train_loader_f2, features_calc_AC, calculate_theta_AC, loss_func_AC)
        """
        
        super().__init__()
        self.model_str_1 = model_str_1
        self.model_str_2 = model_str_2
        self.num_epochs = num_epochs
        self.optim_adam = optim_adam
        self.scheduler = scheduler
        self.calculate_theta = calculate_theta
        self.features_calc = features_calc
        self.loss_func = loss_func

        self.loss_total = []
        self.loss_tot_s1 = []
        self.loss_tot_s2 = []
        self.coef_s_1 = []
        self.coef_s_2 = []

    def train_func(
        self,
        train_loader_1,
        train_loader_2,
    ):
        
        loop = tqdm(range(self.num_epochs), leave=False)
        
        ## range(self.num_epochs)
        
        for epoch in loop:
            #tqdm.write(f"Epoch: {epoch}")

            loss_data_1 = 0
            loss_data_2 = 0

            start_time = time.time()

            ####################################################
            ensemble_s_1 = LoaderProcessor(
                self.model_str_1, self.features_calc, self.calculate_theta, self.loss_func
            )
            loss_data_1, coefficients_1 = ensemble_s_1(train_loader_1)
            self.coef_s_1.append(coefficients_1.cpu().detach().numpy())


            ####################################################
            ensemble_s_2 = LoaderProcessor(
                self.model_str_2, self.features_calc, self.calculate_theta, self.loss_func
            )
            loss_data_2, coefficients_2 = ensemble_s_2(train_loader_2)
            self.coef_s_2.append(coefficients_2.cpu().detach().numpy())

            ####################################################
           
            loss = loss_data_1 + loss_data_2 

            self.loss_total.append(loss.cpu().detach().numpy())
            self.loss_tot_s1.append(loss_data_1.cpu().detach().numpy())
            self.loss_tot_s2.append(loss_data_2.cpu().detach().numpy())

            self.optim_adam.zero_grad()
            loss.backward()
            self.optim_adam.step()

            # scheduler step
            self.scheduler.step()

        self.loss_total = np.array(self.loss_total)
        self.loss_tot_s1 = np.array(self.loss_tot_s1)
        self.loss_tot_s2 = np.array(self.loss_tot_s2)

        loss_func_list = [
            self.loss_tot_s1,
            self.loss_tot_s2,
        ]
        coeff_track_list = [self.coef_s_1, self.coef_s_2]

        return loss_func_list, coeff_track_list













