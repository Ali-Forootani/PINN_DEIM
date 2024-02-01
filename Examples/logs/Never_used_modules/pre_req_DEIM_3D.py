#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:30:39 2023

@author: forootani
"""


import numpy as np
import sys
import os
from pathlib import Path
import torch
from scipy import linalg
import torch.nn as nn
import torch.nn.init as init
from Functions import plot_config_file
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time
import numpy as np
from scipy.linalg import svd, qr
import itertools

############################################

cwd = os.getcwd()
#sys.path.append(cwd + '/my_directory')
sys.path.append(cwd)
warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#############################################
data = np.load('data/CH_Frame1200_X64_Y64_GammaOne_1_GammaTwo_1.npy')

#t_o = data["t"].flatten()[:, None]
#x_o = data["x"].flatten()[:, None]
#Exact = np.real(data["usol"][:,:])


t_o = np.arange(0, 1, 0.0008333333333333334)
y_o = np.arange(-1, 1, 0.03125)
x_o = np.arange(-1, 1, 0.03125)
X, Y, T = np.meshgrid(x_o, y_o, t_o, indexing="ij")
X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))


U = griddata(X_star, data.flatten()[:,None], (X, Y, T), method="nearest")
U = U[:, :, :, 0]

############################################

X_single , Y_single = np.meshgrid(x_o, y_o, indexing="ij")
X_star_sing = np.hstack((X_single.flatten()[:, None], Y_single.flatten()[:, None]))
U_single = griddata(X_star_sing, data[0,:,:].flatten()[:,None], (X_single, Y_single), method="nearest")

############################################


def deim(self, X, i):
    U, Sigma, Vt = svd(X, full_matrices=False)
    
    k = (self.num_basis - self.dec_rate * 2)  # Number of basis functions to retain
    
    # Step 3: Compute the SVD-based approximation
    Uk = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    #print(Sigma_k)
    Vk_t = Vt[:k, :]
    
    X_k = Uk @ Sigma_k @ Vk_t
    
    left = Uk @ np.sqrt(Sigma_k)
    right = np.sqrt(Sigma_k) @ Vk_t
    
    q_x, r_x, p_x = qr(Uk.T, mode='economic', pivoting=True)
    i_x = p_x[:k]
    
    
    q_t, r_t, p_t = qr(Vk_t, mode='economic', pivoting=True)
    i_t = p_t[:k]

    return i_t, i_x



###############################################
