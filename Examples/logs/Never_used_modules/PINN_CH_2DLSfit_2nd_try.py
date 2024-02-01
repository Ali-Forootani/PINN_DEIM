#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:19:09 2023

@author: forootani
"""


### importing essential libraries

import numpy as np
import sys
import os
from pathlib import Path
import torch
from scipy import linalg

from Functions.modules import Siren
from Functions.utils import loss_func_CH_2
from Functions.utils import leastsquares_fit
#from Functions.utils import equation_residual_CH_2D
from Functions.utils import equation_residual_CH

from Functions.utils import equation_residual_mu_CH, equation_residual_u_CH
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
import time

from Functions.base import Dataset, get_train_test_loader
from Functions.samples import Subsample_random



warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")




def create_data():
    #phi_data = np.load("data/phi_2D_test.npy")
    phi_data = np.load("data/CH_Frame1200_X64_Y64_GammaOne_1_GammaTwo_1.npy")
    phi_data = phi_data [0:phi_data.shape[0]:2000, :, :]
    #phi_data = phi_data [:, :, :]
    phi_data = np.transpose(phi_data, (1, 0, 2))
    phi_data = np.transpose(phi_data, (0, 2, 1))
    t_num = phi_data.shape[2]
    x_num = phi_data.shape[0]
    y_num = phi_data.shape[1]
    ## preparing and normalizing the input and output data
    t = np.linspace(0, 1, t_num)
    x = np.linspace(-1, 1, x_num)
    y = np.linspace(-1, 1, y_num)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))
    u_star = phi_data.flatten()[:, None]
    print("The coodinates have shape {}".format(X_star.shape))
    print("The data has shape {}".format(u_star.shape))
    X_star = torch.tensor(X_star, requires_grad = True)
    u_star = torch.tensor(u_star, requires_grad = True)
    
    
    
    return X_star, u_star


X_star, u_star = create_data()


complete_data = torch.hstack((X_star,u_star))

border_points = complete_data[torch.abs(complete_data[:,3])<0.8].detach().cpu().numpy()

inner_points = complete_data[torch.abs(complete_data[:,3])>0.8].detach().cpu().numpy()



dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 1000},
    device=device,
)



coords = dataset.get_coords().detach().cpu()
data = dataset.get_data().detach().cpu()

x_str = X_star.detach().cpu().numpy()
u_str = u_star.detach().cpu().numpy()

uc_str=u_str[np.abs(u_str)<0.8]


#plt.scatter(coords[:,1], coords[:,0], c=data[:,0], marker="o", s=10)
fig_1, ax_1 = plt.subplots()
ax_1.scatter(x_str[:,1], x_str[:,0], c=u_str[:,0], marker="o", s=10)


fig_2, ax_2 = plt.subplots()
ax_2.scatter(coords[:,1], coords[:,0], c=data[:,0], marker="o", s=10)



#fig_3, ax_3 = plt.subplots()
#ax_3.scatter(border_points[:,1], border_points[:,0], c=border_points[:,3], marker="o", s=10)


#fig_4, ax_4 = plt.subplots()
#ax_4.scatter(inner_points[:,1], inner_points[:,0], c=inner_points[:,3], marker="o", s=10)




threshold = 0.9

b = np.array(filter(lambda x: x >= threshold, u_str))



"""
f2 = np.fft.fft2(u_str)
inv_f2 = np.fft.ifft2(f2)
print(inv_f2-u_str.shape)
f2_split = f2[0:2048]
inv_f2_split = np.fft.ifft2(f2_split)
f3_split = f2[2048:]
inv_f3_split = np.fft.ifft2(f3_split)
print(inv_f3_split)
print(u_str[2048:]-inv_f3_split)
"""

#fig_2, ax_2 = plt.subplots()
#ax_2.scatter(x_str[:2048,1], x_str[:2048,0], c=np.real(inv_f2_split[:,0]), marker="o", s=10)


#fig_2, ax_2 = plt.subplots()
#ax_2.scatter(x_str[2048:,1], x_str[2048:,0], c=np.real(inv_f3_split[:,0]), marker="o", s=10)



