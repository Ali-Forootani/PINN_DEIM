#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:59:11 2023

@author: forootani
"""


import numpy as np
import sys
import os
from pathlib import Path
import torch
from scipy import linalg

from Functions.modules import Siren
from Functions.utils import loss_func_CH
from Functions.utils import leastsquares_fit
#from Functions.utils import equation_residual_CH_2D
from Functions.utils import equation_residual_CH

from Functions.utils import equation_residual_mu_CH
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

######################
######################


# Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
"""
nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}
"""




######################
######################

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
    u_y = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_yy = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 1:2]

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

    features = torch.cat([u_t, u_xx, u_yy, mu_xx, mu_yy], dim=1)

    return features


### calculation of dictionary theta


def calculate_theta(features, u_pred):
    thetas = torch.cat(
        [
            (features[:, 1:2] + features[:, 2:3]),
            (torch.pow((2 * u_pred - 1), 2) * u_pred * (1 - u_pred)),
            (u_pred * (1 - u_pred) * (2 * u_pred - 1)),
            (u_pred * (1 - u_pred)),
        ],
        dim=1,
    )

    return thetas


#####################
#####################
def create_data():
    phi_data = np.load("data/data_ujjal.npy")
    phi_data = phi_data[14000:14002,:,:]
    #phi_data = phi_data[np.random.choice(phi_data.shape[0], size=100),:,:] 
    #for i in range(len(phi_data)):
    #    phi_data_sampled = phi_data[:,np.random.choice(phi_data.shape[1], size=250),:]
    #    phi_data_sampled = phi_data_sampled[:, :, np.random.choice(phi_data.shape[2], size=200)]
    
    phi_data_sampled = phi_data 
    
    phi_data_sampled = np.transpose(phi_data_sampled, (1, 0, 2))
    phi_data_sampled = np.transpose(phi_data_sampled, (0, 2, 1))
    
    t_num = phi_data_sampled.shape[2]
    x_num = phi_data_sampled.shape[0]
    y_num = phi_data_sampled.shape[1]
    ## preparing and normalizing the input and output data
    t = np.linspace(0, 1, t_num)
    x = np.linspace(-1, 1, x_num)
    y = np.linspace(-1, 1, y_num)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))
    u_star = phi_data.flatten()[:, None]
    print("The coodinates have shape {}".format(X_star.shape))
    print("The data has shape {}".format(u_star.shape))
    
    X_star = torch.tensor(X_star.astype(float), requires_grad = True)
    u_star = torch.tensor(u_star.astype(float), requires_grad = True)
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



dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "noise_level": 0.05,
        "normalize_coords": True,
        "normalize_data": True,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 500000},
    device=device,
)

coords = dataset.get_coords().detach().cpu()
data = dataset.get_data().detach().cpu()


######################################
######################################



num_params = 4  ## number of equation parameters to be learned
learning_rate_inr = (
    1e-5  ## learning rate value for Siren network of primary variable phi
)
learning_rate_mu = (
    1e-5  ## learning rate value for Siren network of auxilliary vaiable mu
)

perc = 1.0  ## percentage of the data considered
noise = 0.3  ## percentage of noise considered

in_features = 3  ## number of input features to the neural network (for 1D spatio temporal case, use 2: for 2D spatio temporal case, use 3)
out_features = 1  ## number of output features
hidden_features_phi = 128  ## hidden features for the primary SIREN network
hidden_features_mu = 32  ## hidden features for the auxilliary SIREN network
hidden_layers = 3  ## number of hidden layers for both networks
batch_size = 2000  ## batch size for the training
num_epochs = 200  ## total number of epoch for training


# siren model initialization
model_inr = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features=hidden_features_phi,
    hidden_layers=hidden_layers,
    outermost_linear=True,
).to(device)
model_mu = Siren(
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
            "weight_decay": 1e-6,
        },
        {"params": model_mu.parameters(), "lr": learning_rate_mu, "weight_decay": 1e-6},
    ]
)


### learnig scheduler cyclic damping
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optim_adam,
    base_lr=0.1 * learning_rate_inr,
    max_lr=10 * learning_rate_inr,
    cycle_momentum=False,
    mode="exp_range",
    step_size_up=1000,
)


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





# converting numpy to torch
X_t = torch.tensor(coords, requires_grad=True).float().to(device)
Y_t = torch.tensor(data).float().to(device)


# dataloader
print("now starting loading data")
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=True
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
        u_mu = model_mu(local_batch)

        ## features calculation
        features = features_calc(local_batch, u_pred, u_mu)

        ### calculation of dictionary theta
        thetas = calculate_theta(features, u_pred)

        ### calculating coefficients using least squares fit
        coefficients = leastsquares_fit(thetas, u_mu)

        ### calculating residuals mu and equation
        mu_residual = equation_residual_mu_CH(thetas, u_mu, coefficients)
        residual = equation_residual_CH(features)

        ## loss evaluation
        loss, loss_data, loss_eq, loss_mu = loss_func_CH(
            output, u_pred, residual, mu_residual
        )

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
            loss_eq=loss_eq.item(),
            loss_mu=loss_mu.item(),
        )
        # loop.set_postfix(loss = loss.item())

        loss_epoch += loss.item()
        loss_data_epoch += loss_data.item()
        loss_eq_epoch += loss_eq.item()
        loss_mu_epoch += loss_mu.item()

        # loop.set_postfix(residual = torch.mean(residual).item(), mu_lap = torch.mean(mu_lap).item(), time_deriv = torch.mean(features[:,0:1]).item())
    loss_epoch = loss_epoch / len(train_loader)
    loss_data_epoch = loss_data_epoch / len(train_loader)
    loss_eq_epoch = loss_eq_epoch / len(train_loader)
    loss_mu_epoch = loss_mu_epoch / len(train_loader)
    tmp = coefficients
    if epoch % 1 == 0:
        print(
            "It: %d, Loss: %.3e, Loss_data: %.3e, Loss_eq: %.3e, Loss_mu: %.3e"
            % (epoch, loss_epoch, loss_data_epoch, loss_eq_epoch, loss_mu_epoch,)
        )
        print("coeffs: ", tmp.detach().cpu().numpy().T[0, :])
    Loss_collect[epoch, 0] = loss_epoch
    Loss_collect[epoch, 1] = loss_data_epoch
    Loss_collect[epoch, 2] = loss_eq_epoch
    Loss_collect[epoch, 3] = loss_mu_epoch
    params_collect[epoch, :] = tmp.detach().cpu().numpy().T[0, :]


X_star = torch.tensor(X_star, requires_grad=True).float().to(device)


u_pred_total = model_inr(X_star)








########################
########################
########################
########################


"""

import numpy as np

phi_data = np.load("data/data_ujjal.npy")
chunk_size = 1000  # Number of elements to process at a time
n_chunks = (phi_data.size // chunk_size) + 1  # Calculate the number of chunks needed

# Create an iterator to iterate over the flattened data in chunks
it = np.nditer(phi_data, flags=['multi_index'], order='C', op_flags=['readwrite'])
for i in range(n_chunks):
    chunk = np.empty((chunk_size, 1))  # Create a buffer for the chunk
    for j in range(chunk_size):
        if not it.finished:
            chunk[j, 0] = it[0]  # Copy the element to the buffer
            it.iternext()  # Move to the next element
    # Process the chunk
    # print(chunk)


###################
###################



import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

phi_data = np.load("data/data_ujjal.npy")
#phi_data = np.transpose(phi_data, (1, 0, 2))
#phi_data = np.transpose(phi_data, (0, 2, 1))
phi_data = torch.tensor(phi_data)


phi_data = phi_data[10000:,:,:]




phi_data = np.transpose(phi_data, (0, 2, 1))



batch_size = 32
dataset = MyDataset(phi_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)



for i in range(len(dataset)):
    sample = dataset[i]
    print(type(sample))
    print(sample.shape)


    



for batch in dataloader:
    batch = batch.to('cuda')
    # Do some computations on the batch...


#################################
#################################

import numpy as np
from sklearn.decomposition import PCA

# Create a random array of shape (14408, 250, 200)
phi_data = np.load("data/data_ujjal.npy")

# Reshape the array to (14408, 50000)
phi_data_flat = phi_data.reshape(14408, -1)

# Perform PCA on the flattened data to extract the most important features
n_components = 5000  # Set the number of components to keep
pca = PCA(n_components=n_components)
phi_data_pca = pca.fit_transform(phi_data_flat)



# Reconstruct the compressed data to the original shape (14408, 250, 200)
phi_data_recon = pca.inverse_transform(phi_data_pca).reshape(14408, 250, 200)

# Check the shape of the reconstructed data
print(phi_data_recon.shape)


"""








