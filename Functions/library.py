""" Contains the library classes that store the parameters u_t, theta"""
import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations
from functools import reduce
#from .deepmod import Library
from typing import Tuple
#from ..utils.types import TensorList


# ==================== Library helper functions =======================
def library_poly(prediction: torch.Tensor, max_order: int) -> torch.Tensor:
    """Given a prediction u, returns u^n up to max_order, including ones as first column.

    Args:
        prediction (torch.Tensor): the data u for which to evaluate the library (n_samples x 1)
        max_order (int): the maximum polynomial order up to which compute the library

    Returns:
        torch.Tensor: Tensor with polynomials (n_samples, max_order + 1)
    """
    u = torch.ones_like(prediction)
    for order in np.arange(1, max_order + 1):
        u = torch.cat((u, u[:, order - 1 : order] * prediction), dim=1)

    return u


def library_deriv(
    data: torch.Tensor, prediction: torch.Tensor, max_order: int
) :
    """Given a prediction u evaluated at data (t, x), returns du/dt and du/dx up to max_order, including ones
    as first column.

    Args:
        data (torch.Tensor): (t, x) locations of where to evaluate derivatives (n_samples x 2)
        prediction (torch.Tensor): the data u for which to evaluate the library (n_samples x 1)
        max_order (int): maximum order of derivatives to be calculated.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: time derivative and feature library ((n_samples, 1), (n_samples,  max_order + 1))
    """
    dy = grad(
        prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True
    )[0]
    time_deriv = dy[:, 0:1]

    if max_order == 0:
        du = torch.ones_like(time_deriv)
    else:
        du = torch.cat((torch.ones_like(time_deriv), dy[:, 1:2]), dim=1)
        if max_order > 1:
            for order in np.arange(1, max_order):
                du = torch.cat(
                    (
                        du,
                        grad(
                            du[:, order : order + 1],
                            data,
                            grad_outputs=torch.ones_like(prediction),
                            create_graph=True,
                        )[0][:, 1:2],
                    ),
                    dim=1,
                )
       
    #print(time_deriv)
    #print(du)    
    
    features = torch.hstack((time_deriv,du))
    
    
    return features


def features_calc_AC(X, u):
    u_t = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_x = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]

    features = torch.cat([u_t, u_xx], dim=1)
    return features


### calculation of dictionary theta


def calculate_theta_AC(features, u_pred):
    thetas = torch.cat([features[:, 1:2], (torch.pow((u_pred), 3) - u_pred)], dim=1)

    return thetas




###########################



def features_calc_KDV(X, u, t_scale = 0.05, x_scale= 0.03339856490541422):
    
    """
    t_scaling = 1/(t.max(axis=0) - t.min(axis=0)), e.g. t = [0, 20],
    x_scaling = 2/(x.max(axis=0) - x.min(axis=0)), e.g. x = [-30, 30]
    """
    
    u_t = t_scale * torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_x = x_scale * torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = x_scale * torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xxx = x_scale * torch.autograd.grad(
        u_xx, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    uu_x = u.multiply(u_x).reshape(-1,1) 
    
    #u_t = u_t/u_t.norm()
    #uu_x = uu_x/uu_x.norm()
    #u_xxx =  u_xxx/u_xxx.norm()
    
    features = torch.cat([u_t, uu_x, u_xxx], dim=1)
    
    return features


def calculate_theta_KDV(features, u_pred):
    
    thetas = torch.cat([features[:, 1:2], features[:, 2:3] ], dim=1)

    return thetas


###########################################################
###########################################################

def features_calc_Burger(X, u, t_scale = 0.1, x_scale = 0.12549019607843137):
    
    """
    t_scaling = 1/(t.max()-t.min()), e.g. t = [0, 10]
    x_scaling = 2/(x.max()-x.min()), e.g. x = [-8, 7.9375]
    """
    
    u_t = t_scale * torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_x = x_scale * torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = x_scale * torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
     
    #u_t = u_t/u_t.norm()
    #u_x = u_x/u_x.norm()
    #u_xx = u_xx/u_xx.norm()
    
    uu_x = u * u_x
    
    
    features = torch.cat([u_t, uu_x, u_xx], dim=1)
    return features


def calculate_theta_Burger(features, u_pred):
    
    thetas = torch.cat([features[:, 1:2], features[:, 2:3] ], dim=1)

    return thetas
