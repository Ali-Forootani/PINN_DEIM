#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:44:18 2023

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


from Functions.modules import Siren
from Functions.utils import loss_func_AC
from Functions.utils import leastsquares_fit
from Functions.utils import equation_residual_AC
from Functions.library import library_deriv
from Functions import plot_config_file


from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split

import warnings
import time

import numpy as np
from scipy.linalg import svd
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

data = scipy.io.loadmat('data/kdv.mat')

t_o = data["t"].flatten()[:, None]
x_o = data["x"].flatten()[:, None]
Exact = np.real(data["usol"][:,100:200])

#Exact = np.real(data["uu"])
#X, T = np.meshgrid(x_o, t_o, indexing="ij")




def deim(X):
    
    U, Sigma, Vt = svd(X, full_matrices=False)
    
    # Step 2: Select the basis functions
    k = 20   # Number of basis functions to retain
    
    
    # Step 3: Compute the SVD-based approximation
    Uk = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    #Vk_t = Vt[:k, :]
    Vk_t = Vt[:, :k]
    
    #print(Sigma[:k])
    
    #X_k = Uk @ Sigma_k @ Vk_t
    #left = Uk @ np.sqrt(Sigma_k )
    #right = np.sqrt(Sigma_k) @ Vk_t
    
    q_x, r_x, p_x = linalg.qr(Uk.T, mode='economic', pivoting=True)
    
    
    #print(q_x.shape)
    #print(r_x.shape)
    #print(p_x)
    
    i_x = p_x[:k]
    
    q_t, r_t, p_t = linalg.qr(Vk_t.T, mode='economic', pivoting=True)
    i_t = p_t[:k]
    
    return i_t, i_x
    
    
    
##########################################
##########################################

n_k = Exact.shape[1]
n_d = 1
n_s = int(n_k / n_d)

i_t = []
i_x = []
u_selected = []

t_sampled = []
x_sampled = []

X_sampled = []
T_sampled = []

S_star = []
T_star = []
U_star = []

for i in range(n_d):
    
    #print(i)
    #print(i * n_s)
    #print((i+1) * n_s)
    
    i_tf, i_xf = deim(Exact[:, i * n_s: (i+1) * n_s])
    
    i_tf = i_tf + i * n_s
    
    i_t.append([i_tf])
    i_x.append([i_xf])
    
    #X, T = np.meshgrid(x_o, t_o[i * n_s: (i+1) * n_s], indexing="ij")
    
    X, T = np.meshgrid(x_o[i_xf], t_o[i_tf], indexing="ij") #values[index]
    
    
    X_sampled.append(X)
    T_sampled.append(T)    
    
    
    t, space = np.meshgrid(i_tf, i_xf, indexing="ij") #index
    u_selected.append(Exact[space,t])
    t_sampled.append(t)
    x_sampled.append(space)
    
    X_star = np.hstack((t.flatten()[:, None], space.flatten()[:, None]))
    
    #plt.scatter(X_star[:,0],X_star[:,1])
    plt.scatter(X_star[:,0],X_star[:,1], c=Exact[space,t])
    
    S_star.append(X_sampled[i].flatten())
    T_star.append(T_sampled[i].flatten())
    U_star.append(u_selected[i].flatten())

    
S_s = np.concatenate(S_star, axis=0).reshape(-1,1)
T_s = np.concatenate(T_star, axis=0).reshape(-1,1)
U_s = np.concatenate(U_star, axis=0).reshape(-1,1)

coords = np.hstack((T_s,S_s))

print(U_s.shape)


#### Cleaning the data with FFT
# https://www.youtube.com/watch?v=s2K1JfNR7Sc


#U, Sigma, Vt = svd(Exact, full_matrices=False)
#print("SVDDDDDDDDDDDDDDDD")
#print(Vt.shape)
#print(U.shape)
#print(Sigma.shape)
#print(Exact - U @ np.diag(Sigma) @ Vt)


