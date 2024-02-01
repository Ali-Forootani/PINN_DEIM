#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 08:24:14 2023

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



def theta_u(X, mu):
    
    mu_x = torch.autograd.grad(
        mu, X, grad_outputs=torch.ones_like(mu), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    mu_xx = torch.autograd.grad(
        mu_x, X, grad_outputs=torch.ones_like(mu), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    mu_y = torch.autograd.grad(
        mu, X, grad_outputs=torch.ones_like(mu), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    
    mu_yy = torch.autograd.grad(
        mu_x, X, grad_outputs=torch.ones_like(mu), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    
    
    u_theta = torch.cat([mu_xx, mu_yy], dim=1)
    
    return u_theta




def grad_u(X, u):
    u_x = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    u_y = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_yy = torch.autograd.grad(
        u_y, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    
    
    u_grad = torch.cat([u_xx, u_yy], dim=1)
    
    return u_grad


def theta_mu(u, u_grad):
    
    mu_theta = torch.cat([
                        u,
                        torch.pow(u,3),
                        u_grad[:,0:1],
                        u_grad[:,1:2],
                          ],
                         dim=1)
    
    return mu_theta 



################################
################################
################################

### calculation of derivatives
def features_calc(X, u, u_mu):

    u_t = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 2:3]
    u_x = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    u_xxx = torch.autograd.grad(
        u_xx, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    u_xxxx = torch.autograd.grad(
        u_xxx, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    u_xxy = torch.autograd.grad(
        u_xx, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    
    u_xxyy = torch.autograd.grad(
        u_xxy, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    
    
    u_y = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_yy = torch.autograd.grad(
        u_y, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    
    u_yyy = torch.autograd.grad(
        u_yy, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    
    u_yyyy = torch.autograd.grad(
        u_yyy, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    
    u_yyx = torch.autograd.grad(
        u_yy, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    u_yyxx = torch.autograd.grad(
        u_yyx, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    """
    mu_x = torch.autograd.grad(
        u_mu, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    mu_xx = torch.autograd.grad(
        mu_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]

    mu_y = torch.autograd.grad(
        u_mu, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    mu_yy = torch.autograd.grad(
        mu_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    """
    
    
    #features = torch.cat([u_t, u_xx, u_yy, mu_xx, mu_yy], dim=1)
    
    features = torch.cat([u_t,
                          u_x,
                          u_xx,
                          u_xxx,
                          u_xxxx,
                          u_xxyy,
                          u_y,
                          u_yy,
                          u_yyyy,
                          u_yyxx
                          ], dim=1)
    
    
    """
    u_t –
    (3 u_xx u^2 + 6 (u_x)^2 u - u_xx - u_xxxx – u_yyxx        
     +
     3 u_yy u^2 + 6 (u_y)^2 u - u_yy - u_xxyy - u_yyyy
     ) = 0
    """
        
        
    return features


### calculation of dictionary theta


def calculate_theta(features, u_pred):
    """
    u_t –
    (3 u_xx u^2 + 6 (u_x)^2 u - u_xx - u_xxxx – u_yyxx        
     +
     3 u_yy u^2 + 6 (u_y)^2 u - u_yy - u_xxyy - u_yyyy
     ) = 0
    """
    #features = torch.cat([u_t,
    #                      u_x,
     #                     u_xx,
      #                    u_xxx,
       #                   u_xxxx,
        #                  u_xxyy,
         #                 u_y,
          #                u_yy,
           #               u_yyyy,
            #              u_yyxx
             #             ], dim=1)
    
    thetas = torch.cat(
        [   
            features[:,2:3],
            features[:,4:5],
            torch.pow(features[:,1:2],2)* u_pred,
            features[:,2:3]* u_pred* u_pred,
            features[:,5:6],
            features[:,7:8],
            features[:,8:9],
            torch.pow(features[:,6:7],2)* u_pred,
            features[:,7:8]* u_pred* u_pred,
            features[:,9:10],
            #(features[:, 1:2] + features[:, 2:3]),
            #(torch.pow((2 * u_pred - 1), 2) * u_pred * (1 - u_pred)),
            #(u_pred * (1 - u_pred) * (2 * u_pred - 1)),
            #(u_pred * (1 - u_pred)),
        ],
        dim=1,
    )
    
    return thetas


## Training parameters


num_params = 4  ## number of equation parameters to be learned
learning_rate_inr = (
    1e-5  ## learning rate value for Siren network of primary variable phi
)
learning_rate_mu = (
    1e-5  ## learning rate value for Siren network of auxilliary vaiable mu
)

perc = 1.0  ## percentage of the data considered
noise = 0  ## percentage of noise considered

in_features = 3  ## number of input features to the neural network (for 1D spatio temporal case, use 2: for 2D spatio temporal case, use 3)
out_features = 1  ## number of output features
hidden_features_phi = 256  ## hidden features for the primary SIREN network
hidden_features_mu = 256  ## hidden features for the auxilliary SIREN network
hidden_layers = 4  ## number of hidden layers for both networks
batch_size = 10000  ## batch size for the training
num_epochs = 1000 ## total number of epoch for training

### file saving parameters

string_f = (
    "_LSfit_lr_"
    + str(learning_rate_inr)
    + "_hf_phi_"
    + str(hidden_features_phi)
    + "_layers_"
    + str(int(hidden_layers))
    + "_ep_"
    + str(int(num_epochs))
    + "_noise_"
    + str(int(100 * noise))
    + "_perc_"
    + str(int(100 * perc))
)

### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "result_CH_2d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.mkdir(result_path)
## File saving parameters
filename_u = (
    result_path + "/" + "U_data" + string_f + ".npy"
)  ## path where the primary variable phi data is saved
filename_mu = (
    result_path + "/" + "mu_pred" + string_f + ".npy"
)  ## path where the auxilliary variable mu data is saved
filename_l = (
    result_path + "/" + "Loss_collect" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved
filename_p = (
    result_path + "/" + "params_collect" + string_f + ".npy"
)  ## path where the parameter data for every epoch is saved
filename_model_u = (
    result_path + "/" + "model_u" + string_f + ".pt"
)  ## path where the primary SIREN network data is saved
filename_model_mu = (
    result_path + "/" + "model_mu" + string_f + ".pt"
)  ## path where the auxilliary SIREN netowrk data is saved

pltname_u = (
    result_path + "/" + "SOC" + string_f
)  ## path where the plots of primary variable phi is saved
pltname_l = (
    result_path + "/" + "LP" + string_f + ".png"
)  ## path where loss plot is saved

pltname_p1 = (
    result_path + "/" + "PP_1" + string_f + ".png"
)  ## pltname_p1 and pltname_p2 are paths where the plots of learned parameters are saved
pltname_p2 = result_path + "/" + "PP_2" + string_f + ".png"


# loading the data



def create_data():
    
    #phi_data = np.load("data/phi_2D_test.npy")
    
    phi_data = np.load("data/CH_Frame1200_X64_Y64_GammaOne_1_GammaTwo_1.npy")
    phi_data = phi_data [0:phi_data.shape[0]:100, :, :]
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
    
    #data = loadmat("data/advection_diffusion.mat")
    #usol = np.real(data["Expression1"]).astype("float32")
    #usol = usol.reshape((51, 51, 61, 4))
    #x_v = usol[:,:,:,0]
    #y_v = usol[:,:,:,1]
    #t_v = usol[:,:,:,2]
    #u_v = usol[:,:,:,3]
    #coords = torch.from_numpy(np.stack((t, x, y), axis=-1))
    #data = torch.from_numpy(usol[:, :, :, 3]).unsqueeze(-1)
    # alternative way of providing the coordinates
    # coords = torch.from_numpy(np.transpose((t_v.flatten(), x_v.flatten(), y_v.flatten())))
    # data = torch.from_numpy(usol[:, :, :, 3].reshape(-1,1))
    #print("The coodinates have shape {}".format(coords.shape))
    #print("The data has shape {}".format(data.shape))
    return X_star, u_star


X_star, u_star = create_data()



complete_data = torch.hstack((X_star, u_star))



border_points = complete_data[torch.abs(complete_data[:,3])<0.8].detach().cpu().numpy()


inner_points = complete_data[torch.abs(complete_data[:,3])>0].detach().cpu().numpy()


X_star_border = border_points[:,:3]
u_star_border = border_points[:,3]



X_star_inner = inner_points[:,:3]
u_star_inner = inner_points[:,3]


"""
u_t –
(3 u_xx u^2 + 6 (u_x)^2 u - u_xx - u_xxxx – u_yyxx        
+
3 u_yy u^2 + 6 (u_y)^2 u - u_yy - u_xxyy - u_yyyy
) = 0

(t,x,y) \to u
du/dt = coefs * theta, least sqaure 
"""




"""
u_t - mu_xx - mu_yy = 0,
mu = gamma_2 u^3 - gamma_2 u - gamma_1 u_xx - gamma_1 u_yy,
"""


"""
(t,x,y) \to u_hat  : DNN

u_t - mu_xx - mu_yy = 0, least square, 

mu - (gamma_2 u^3 - gamma_2 u - gamma_1 u_xx - gamma_1 u_yy) = 0, 
last square,


total loss = u_loss + loss_mu + loss_

"""



u_star_noisy = u_star.detach().cpu().numpy() + noise * np.std(u_star.detach().cpu().numpy()) * np.random.randn(
u_star.shape[0], u_star.shape[1])



phi_data = np.load("data/CH_Frame1200_X64_Y64_GammaOne_1_GammaTwo_1.npy")

phi_data = phi_data [0:phi_data.shape[0]:100, :,:]

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

U_noisy = griddata(X_star.detach().cpu().numpy(), u_star_noisy
                   , (X, Y, T), method="nearest")
U_noisy = U_noisy[:, :, :, 0]





dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "noise_level": 0.00,
        "normalize_coords": False,
        "normalize_data": False,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 100000},
    device=device,
)




coords = dataset.get_coords().detach().cpu()
data = dataset.get_data().detach().cpu()


"""

phi_data = np.load(
    "data/phi_2D_test.npy"
)  ## This is a test case of the full experimental data. To run for the experimental data please load 'data/u_all.npy'. Please note that file size is large.
## The file u_all.npy cannot be uploaded to gitlab, but it is there in "BigMax" folder. Please set the path accordingly if you want to load the u_all.npy

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
N_u = u_star.shape[0]

### adding noise

print("adding noise")

u_star_noisy = u_star + noise * np.std(u_star) * np.random.randn(
    u_star.shape[0], u_star.shape[1]
)
print("create training set")
print(np.std(u_star))
N_u = round(perc * (X_star.shape[0]))
print(N_u)
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_train = X_star[idx, :]
u_train = u_star_noisy[idx, :]
U_noisy = griddata(X_star, u_star_noisy, (X, Y, T), method="nearest")
U_noisy = U_noisy[:, :, :, 0]

fig, axes = plt.subplots(ncols=3, figsize=(15, 4))

im0 = axes[0].contourf(x, y, phi_data[0,:,:], cmap='coolwarm')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('t = 0')

im1 = axes[1].contourf(x, y, phi_data[10,:,:], cmap='coolwarm')
axes[1].set_xlabel('x')
axes[1].set_title('t = 10')

im2 = axes[2].contourf(x, y, phi_data[20,:,:], cmap='coolwarm')
axes[2].set_xlabel('x')
axes[2].set_title('t= 20')

fig.colorbar(im1, ax=axes.ravel().tolist())

plt.show()

"""




# siren model initialization
model_inr = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features=hidden_features_phi,
    hidden_layers=hidden_layers,
    outermost_linear=True,
).to(device)

model_mu_2 = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features=hidden_features_mu,
    hidden_layers=hidden_layers,
    outermost_linear=True,
).to(device)




# siren model initialization
model_inr_2 = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features=hidden_features_phi,
    hidden_layers=hidden_layers,
    outermost_linear=True,
).to(device)

model_mu_4 = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features=hidden_features_mu,
    hidden_layers=hidden_layers,
    outermost_linear=True,
).to(device)





print(model_inr)

# optimizer
optim_adam = torch.optim.Adam(
    [
        {
            "params": model_inr.parameters(),
            "lr": learning_rate_inr,
            "weight_decay": 0,
        },
        {"params": model_mu_2.parameters(),
         "lr": learning_rate_mu,
         "weight_decay": 0},
    ]
)


### learnig scheduler cyclic damping
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optim_adam,
    base_lr= 1 * learning_rate_inr,
    max_lr= 10 * learning_rate_inr,
    cycle_momentum=False,
    mode="exp_range",
    step_size_up=1000,
)

### learnig scheduler cyclic
# scheduler = torch.optim.lr_scheduler.CyclicLR(optim_adam, base_lr=0.1*learning_rate_inr, max_lr=10*learning_rate_inr, cycle_momentum=False)


### change the name of the file in string_f according to the learning scheme to differentiate the results for different learning schemes.
### For a constant learning rate, please comment out line (scheduler.step())


# converting numpy to torch
#X_t = torch.tensor(coords, requires_grad=True).float().to(device)
#Y_t = torch.tensor(data).float().to(device)


X_t = torch.tensor(X_star, requires_grad=True).float().to(device)
Y_t = torch.tensor(u_star).float().to(device)


X_t_border = torch.tensor(X_star_border, requires_grad=True).float().to(device)
Y_t_border = torch.tensor(u_star_border).float().to(device)




# dataloader
print("now starting loading data")
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=True
)


train_loader_border = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t_border, Y_t_border), batch_size=batch_size, shuffle=True
)




Loss_collect = np.zeros((num_epochs, 4))
params_collect = np.zeros((num_epochs, num_params))


print("Parameters are learned")
print("Training Begins !!")

for epoch in range(num_epochs):
    # print(f'epoch {epoch}')
    loss_epoch = 0
    loss_data_epoch = 0
    loss_eq_epoch = 0
    loss_mu_epoch = 0

    ii = 0

    start_time = time.time()

    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (local_batch, output) in loop:

        u_pred = model_inr(local_batch)
        
        mu_pred = model_mu_2(local_batch)
        
        
        u_grad = grad_u(local_batch, u_pred) # gradient of u w.r.t. "x" and "y".
        
        mu_theta = theta_mu(u_pred, u_grad) # constructung mu based on u_pred & u_grad:u_xx,u_yy
        
        coefficients_mu = leastsquares_fit(mu_theta, mu_pred)
        
        
        
        #mu_pred = model_mu_2(mu_theta)
        
        u_theta = theta_u(local_batch, mu_pred) #u_theta = mu_xx + mu_yy
        
        
        
        u_t = torch.autograd.grad(
            u_pred, local_batch, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True
        )[0][:, 2:3]
        
        coefficients_u = leastsquares_fit(u_theta, u_t)
        
       
        
        
        loss_data = torch.mean((output - u_pred) ** 2)
        
        loss_res_u = torch.mean((u_t - u_theta @ coefficients_u ) ** 2)
        
        loss_res_mu = torch.mean((mu_pred - mu_theta @ coefficients_mu ) ** 2)
        
        loss = loss_data + loss_res_u + 0.1 * loss_res_mu 
        
        ## features calculation
       # features = features_calc(local_batch, u_pred, u_mu)
        
       
        

        ### calculation of dictionary theta
        #thetas = calculate_theta(features, u_pred)
        
       
        

        ### calculating coefficients using least squares fit
        
        
        #coefficients = leastsquares_fit(thetas, features[:,0:1])
        
        #mu_coefficients = 
        
        ################################################
        #
        #
        #
        #residual_1 = equation_residual_AC(thetas_1, time_derivs_1, coefficients_1)
        #loss_data_1 = torch.mean((out_1 - u_pred_1) ** 2)
        #loss_data_b1, loss_data, loss_eq = loss_func_AC(out_1, u_pred_1, residual_1)
        #loss_data_1 = loss_data_1 + loss_data_b1
        #
        #
        #
        
        ### calculating residuals mu and equation
        #mu_residual = equation_residual_mu_CH(thetas, u_mu, coefficients)
        
        #residual = equation_residual_CH(features)
        
        #u_residual = equation_residual_u_CH(thetas, features[:,0:1], coefficients)
        
        #loss, loss_data, loss_u = loss_func_CH_2(
        #    output, u_pred, u_residual)
        
        
        
        #loss_data = torch.mean((output - u_pred) ** 2)
        
        #loss = loss_data + 
        
        ## loss evaluation
        # (u, u_pred, f_pred, f_pred_u)
        
       

        # Backward and optimize
        optim_adam.zero_grad()
        loss.backward()
        optim_adam.step()

        # scheduler step
        scheduler.step()

        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(
            loss=loss.item(),
            loss_data=loss_data.item(),
            #loss_eq=loss_eq.item(),
            #loss_mu=loss_mu.item(),
        )
        # loop.set_postfix(loss = loss.item())

        loss_epoch += loss.item()
        loss_data_epoch += loss_data.item()
        #loss_eq_epoch += loss_eq.item()
        #loss_mu_epoch += loss_mu.item()

        # loop.set_postfix(residual = torch.mean(residual).item(), mu_lap = torch.mean(mu_lap).item(), time_deriv = torch.mean(features[:,0:1]).item())
    loss_epoch = loss_epoch / len(train_loader)
    loss_data_epoch = loss_data_epoch / len(train_loader)
    loss_res_u = loss_res_u / len(train_loader)
    loss_res_mu = loss_res_mu / len(train_loader)
    tmp = coefficients_mu
    if epoch % 1 == 0:
        print(
            "It: %d, Loss: %.3e, Loss_data: %.3e, Loss_eq: %.3e, Loss_mu: %.3e"
            % (epoch, loss_epoch, loss_data_epoch, loss_res_u, loss_res_mu,)
        )
        print("coeffs: ", tmp.detach().cpu().numpy().T[0, :])
    Loss_collect[epoch, 0] = loss_epoch
    Loss_collect[epoch, 1] = loss_data_epoch
    Loss_collect[epoch, 2] = loss_res_u
    Loss_collect[epoch, 3] = loss_res_mu
    params_collect[epoch, :] = tmp.detach().cpu().numpy().T[0, :]





X_star = torch.tensor(X_star, requires_grad=True).float().to(device)


X_t = torch.tensor(coords, requires_grad=True).float().to(device)
#Y_t = torch.tensor(data).float().to(device)


u_pred_total = model_inr(X_t)


u_pred_total = u_pred_total.cpu()
u_pred_total = u_pred_total.detach().numpy()

X_t = X_t.cpu()
X_t = X_t.detach().numpy()

U_pred = griddata(X_t, u_pred_total, (X, Y, T), method="nearest")

U_pred = U_pred[:, :, :, 0]

error = np.abs(U_pred[:, :, 1] - phi_data[:, :, 1]) / linalg.norm(
    phi_data[:, :, 1], "fro"
)


u_data = {"orig": phi_data, "pinn": U_pred, "noisy": U_noisy, "error": error}


np.save(filename_u, u_data, allow_pickle=True)

np.save(filename_l, Loss_collect)
np.save(filename_p, params_collect)

torch.save(model_inr, filename_model_u)


plt.rcParams.update({
    "font.size": 20,
})
### plotting

fig2 = plt.figure(figsize=(6, 6))
shape_loss = Loss_collect.shape[1]
if shape_loss == 1:
    plt.semilogy(Loss_collect[:, 0:1])
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.savefig(pltname_l)
if shape_loss == 3:
    plt.semilogy(Loss_collect[:, 0:1], label="Total loss")
    plt.semilogy(Loss_collect[:, 1:2], label="Data loss")
    plt.semilogy(Loss_collect[:, 2:3], label="Equation loss")
    plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.savefig(pltname_l)
if shape_loss == 4:
    plt.semilogy(Loss_collect[:, 0:1], label="Total loss")
    plt.semilogy(Loss_collect[:, 1:2], label="Data loss")
    plt.semilogy(Loss_collect[:, 2:3], label="Equation loss")
    plt.semilogy(Loss_collect[:, 3:4], label="$\mu$ loss")
    plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.savefig(pltname_l)




plt.rcParams.update({
    "font.size": 18,
})

fig = plt.figure(figsize=(20, 5))
ax = fig.add_subplot(141)


h0 = ax.imshow(
    U_noisy[:, :, 1],
    interpolation="nearest",
    extent=[y.min(), y.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h0, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("Noisy")
ax.set_xlabel("$y$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(142)
h1 = ax.imshow(
    phi_data[:, :, 1],
    interpolation="nearest",
    extent=[y.min(), y.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h1, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("Original")
ax.set_xlabel("$y$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(143)
h2 = ax.imshow(
    U_pred[:, :, 1],
    interpolation="nearest",
    extent=[y.min(), y.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h2, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("SIREN representation")
ax.set_xlabel("$y$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(144)
h3 = ax.imshow(
    error[:, :],
    interpolation="nearest",
    extent=[y.min(), y.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h3, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("error")
ax.set_xlabel("$y$", size=20)
ax.set_ylabel("$x$", size=20)

fig.tight_layout(pad=3.0)
plt.savefig(pltname_u + ".png")


#####################################
#####################################

from Functions import plot_config_file


plt.rcParams.update({
    "font.size": 15,
})

fig = plt.figure(figsize=(12, 4))


ax = fig.add_subplot(131)
h1 = ax.imshow(
    phi_data[:, :, 1],
    interpolation="nearest",
    extent=[y.min(), y.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h1, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("Original")
ax.set_xlabel("$y$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(132)
h2 = ax.imshow(
    U_pred[:, :, 1],
    interpolation="nearest",
    extent=[y.min(), y.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h2, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("Prediction")
ax.set_xlabel("$y$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(133)
h3 = ax.imshow(
    error[:, :],
    interpolation="nearest",
    extent=[y.min(), y.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h3, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("error")
ax.set_xlabel("$y$", size=20)
ax.set_ylabel("$x$", size=20)

fig.tight_layout(pad=3.0)
plt.savefig(pltname_u + ".png")






#####################################
#####################################


# fig5 = plt.figure(figsize=(6,6))
# plt.plot(params_collect[:,0],label = ('$\lambda_1$'))
# plt.yscale('symlog', linthreshy=1e-8)

# plt.legend(loc = 'upper right', prop={'size':17}, frameon = False)
# plt.xlabel('Epoch')
# plt.ylabel('Parameter')
# plt.savefig(pltname_p1)


# fig6 = plt.figure(figsize=(6,6))
# plt.plot(params_collect[:,1],label = ('$\lambda_2$'))
# plt.yscale('symlog', linthreshy=1e-4)
# plt.legend(loc = 'upper right', prop={'size':17}, frameon = False)
# plt.xlabel('Epoch')
# plt.ylabel('Parameter')
# plt.savefig(pltname_p2)
