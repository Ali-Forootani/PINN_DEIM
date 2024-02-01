#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:30:12 2023

@author: forootani
"""
import torch 
from PINN_DEIM.Functions.utils import leastsquares_fit, equation_residual_AC



def process_loader(loader, model_fn, features_fn, calculate_theta_fn, loss_func):
    loss_data_total = 0
    coefficients_list = []
    
    
    print(model_fn)

    for batch_idx, (input_data, output_data) in enumerate(loader):
        u_pred = model_fn(input_data)
        
        
        features = features_fn(input_data, u_pred)
        time_derivs = features[:, 0:1]
        thetas = calculate_theta_fn(features, u_pred)
        coefficients = leastsquares_fit(thetas, time_derivs)
        residual = equation_residual_AC(thetas, time_derivs, coefficients)
        loss_data_mse = torch.mean((output_data - u_pred) ** 2)
        loss_data_b, loss_data_c, loss_eq = loss_func(output_data, u_pred, residual)
        loss_data_total = loss_data_b + loss_data_mse

    return loss_data_total, coefficients


"""
class LoaderProcessor:
    def __init__(self, model_fn, features_fn, calculate_theta_fn, loss_func):
        self.model_fn = model_fn
        self.features_fn = features_fn
        self.calculate_theta_fn = calculate_theta_fn
        self.loss_func = loss_func
        
    def __call__(self, loader):
        loss_data_total = 0
        
        for batch_idx, (input_data, output_data) in enumerate(loader):
            u_pred = self.model_fn(input_data)
            features = self.features_fn(input_data, u_pred)
            time_derivs = features[:, 0:1]
            thetas = self.calculate_theta_fn(features, u_pred)
            coefficients = leastsquares_fit(thetas, time_derivs)
            residual = equation_residual_AC(thetas, time_derivs, coefficients)
            loss_data_mse = torch.mean((output_data - u_pred) ** 2)
            
            loss_data_b, loss_data_c, loss_eq = self.loss_func(output_data, u_pred, residual)
            loss_data_total = loss_data_b + loss_data_mse

        return loss_data_total, coefficients
"""



##################################
######### Debugging modules


class LoaderProcessor:
    def __init__(self, model_fn, features_fn, calculate_theta_fn, loss_func):
        self.model_fn = model_fn
        self.features_fn = features_fn
        self.calculate_theta_fn = calculate_theta_fn
        self.loss_func = loss_func
        
    def __call__(self, loader):
        loss_data_total = 0
        
        for batch_idx, (input_data, output_data) in enumerate(loader):
            u_pred = self.model_fn(input_data)
            features = self.features_fn(input_data, u_pred)
            time_derivs = features[:, 0:1]
            thetas = self.calculate_theta_fn(features, u_pred)
            coefficients = leastsquares_fit(thetas, time_derivs)
            residual = equation_residual_AC(thetas, time_derivs, coefficients)
            
            #loss_data_mse = torch.mean((output_data - u_pred) ** 2)
            
            loss_data_b, loss_data_c, loss_eq = self.loss_func(output_data, u_pred, residual)
            loss_data_total = loss_data_b + loss_data_total

        return loss_data_total, coefficients



"""
def loss_func_AC(u, u_pred, f_pred):
    loss_data = torch.mean((u - u_pred) ** 2)
    loss_eq = torch.mean(f_pred ** 2)
    loss = loss_data + loss_eq
    return loss, loss_data, loss_eq


def leastsquares_fit(thetas, time_derivs):
    Q, R = torch.qr(thetas)  # solution of lst. sq. by QR decomp.
    coeff_vectors = torch.inverse(R) @ Q.T @ time_derivs
    return coeff_vectors


### calculation of the equation residual of AC equation


def equation_residual_AC(thetas, time_derivs, coeff_vectors):
    residual = time_derivs - thetas @ coeff_vectors
    return residual
"""


"""
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
"""