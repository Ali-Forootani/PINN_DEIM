#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:13:07 2023

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
    leastsquares_fit,
    loss_func_KDV
)
from PINN_DEIM.Functions.library import library_deriv, features_calc_KDV, calculate_theta_KDV
from PINN_DEIM.Functions import plot_config_file
from PINN_DEIM.Functions.ensemble_loader import LoaderProcessor, process_loader

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
from matplotlib.ticker import FuncFormatter


warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

####################### loading the data set
#############################################

data = scipy.io.loadmat(root_dir + '/data/kdv.mat')

KDV_instance = KDVEquationEntireDomain(data)

(x_train_list, u_train_list, train_loaders) = KDV_instance.prepare_data_DEIM(
                                                    t_div=2,
                                                    n_basis=1,
                                                    deim_tolerance=1e-3
                                                    )
X_str_train = x_train_list[0]
u_str_train = u_train_list[0]
train_loader = train_loaders[0]

test_data_size = 0.99


fig, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(7, 6))

ax.scatter(
    X_str_train[:, 1], X_str_train[:, 0], c=u_str_train[:, 0], marker="x", s=50
)

plt.show()

#############################################
#############################################


noise = 0
learning_rate_inr = 1e-5
hidden_features_str = 128
#hidden_features = 64
hidden_layers = 3
num_epochs = 1000
prec = 1 - test_data_size

string_f = KDV_instance.save_string(
    noise=0,
    learning_rate_inr=1e-5,
    hidden_features_str=128,
    hidden_layers=3,
    num_epochs=1000,
    prec=1 - test_data_size,
    random_sampling = False
)

### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "./logs/result_DEIM_KDV_1d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.makedirs(result_path)

#############################################
#############################################


"""
Defining deep NN structure for AC model, i.e. optimizer, scheduler, DNN model 
"""

KVDDeepModel_instance = PDEDeepModelEntireDomain(
    in_features=2,
    out_features=1,
    hidden_features_str=128,
    hidden_layers=3,
    learning_rate_inr=1e-5,)

models_list, optim_adam, scheduler = KVDDeepModel_instance.run()

model_str = models_list[0]


Train_inst = Trainer(
    model_str,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
    features_calc = features_calc_KDV,
    calculate_theta = calculate_theta_KDV,
    loss_func = loss_func_KDV
)

loss_func_list, coeff_track_list = Train_inst.train_func(
    train_loader
)


coef_s = coeff_track_list[0]
loss_tot_s = loss_func_list[0]
coef_s_tr = np.reshape(np.array(coef_s), (num_epochs, coef_s[0].shape[0]))

################################################
################################################

"""
file path, saving the data and related parameters
"""
(
    filename_loss_funcs,
    filename_coef,
    filename_model,
    filename_data,
) = KDV_instance.file_path_saving(result_path)


################################################
################################################

plt.rcParams.update(
    {
        "font.size": 20,
    }
)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))

x_ticks = [0, 500, 1000]

line_width = 3

axs2.semilogy(
    loss_tot_s,
    marker="^",
    markersize=5,
    linewidth=line_width,
    markevery=100,
    linestyle="-",
)

axs2.set_xticks(x_ticks)


axs2.set(xlabel="Iteration")
plt.title("Loss functions")

plt.tight_layout()
plt.savefig(
    filename_loss_funcs + "loss_function" + " random_sampling" + ".png",
    dpi=600,
)
plt.savefig(
    filename_loss_funcs + "loss_function" + " random_sampling" + ".pdf",
    dpi=600,
)

np.save(filename_loss_funcs, loss_tot_s)
np.save(filename_coef, coef_s_tr)

##############################################
##############################################

plt.rcParams.update({"font.size": 15,})

line_width = 3

fig, axs = plt.subplots(1, 2, sharey=False, figsize=(8, 3))

axs[0].plot(coef_s_tr[:, 0], linewidth=line_width, linestyle="-", color=colors[1])

axs[0].plot(-6*torch.ones(coef_s_tr[:, 0].shape), linewidth=line_width, linestyle="--",
            color=colors[2])


axs[1].plot( coef_s_tr[:, 1],
    linewidth=line_width, linestyle="-", color=colors[1],
    label=r"\texttt{Estimated Coefficients}",)

axs[1].plot( -1*torch.ones(coef_s_tr[:, 1].shape),
    linewidth=line_width, linestyle="--", color=colors[2],
    label=r"\texttt{Ground Truth}",)


plt.subplots_adjust(wspace=0.5, hspace=0.5)

x_ticks = [0, 500, 1000]
y_ticks_0 = [-8, -6, -4, -2, 0, ]
y_ticks_1 = [-1.5, -1, 0, 1, 1.5]



# Define a custom formatting function for y-axis tick labels
def sci_fmt_1(x, pos):
    return "${:.2f}$".format(x)

def sci_fmt_2(x, pos):
    return "${:.0f}$".format(x)

# Apply the custom formatting function to the y-axis tick labels
# for ax in axs.flat:
#    ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt_1))


for j in range(2):
    if j == 0:
        #axs[j].set_ylim([-8, -4])
        axs[j].set_xticks(x_ticks)
        #axs[j].set_yticks(y_ticks_0)
        axs[j].set(xlabel="Iteration")
        axs[j].yaxis.set_major_formatter(FuncFormatter(sci_fmt_1))
        axs[j].set(ylabel=r"Coeff $uu_{x}$ ")
        axs[j].grid(True)
        
    if j == 1:
        axs[j].set_ylim([-2, 2])
        #axs[j].set_yticks(y_ticks_1)
        axs[j].set_xticks(x_ticks)
        axs[j].set(xlabel="Iteration")
        axs[j].yaxis.set_major_formatter(FuncFormatter(sci_fmt_2))
        axs[j].set(ylabel=r"Coeff $u_{xxx}$ ")
        axs[j].grid(True)
        
##################################
##################################

lines, labels = [], []
for ax in fig.axes:
    line, label = ax.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)

fig.legend(
    lines,
    labels,
    loc="center",  # Change the location to upper center
    ncol=4,
    bbox_to_anchor=(0.51, 1),  # Adjust the coordinates
    bbox_transform=fig.transFigure,
    fontsize=15,
    frameon=True,
)


plt.savefig( filename_coef + "coefficients" + "DEIM_sampling" + ".png", bbox_inches='tight', dpi= 600,)
plt.savefig( filename_coef + "coefficients" + "DEIM_sampling" + ".pdf", bbox_inches='tight', dpi= 600,)

#################################
#################################


plt.rcParams.update({"font.size": 12})

fig_5, ax_5 = plt.subplots(1, 1, figsize=(3, 3), sharey=True, sharex=True)

scatter = ax_5.scatter(
    X_str_train[:, 1],
    X_str_train[:, 0],
    c=u_str_train[:, 0],
    marker="D",
    s=10,
    label=r"Sampled dataset: Q-DEIM",
)

ax_5.set(xlabel=r"$t$")
ax_5.set(ylabel=r"$x$")

# Add individual legends to each subplot
ax_5.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8)

plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed

colorbar = fig_5.colorbar(
    scatter, ax=ax_5, location="right", aspect=50, fraction=0.02
)
# fig_5.suptitle('Sampled Data set')

plt.savefig(filename_data + "dataset_ensemble" + " random_sampling" + ".png", bbox_inches='tight', dpi=600)
plt.savefig(filename_data + "dataset_ensemble" + " random_sampling" + ".pdf", bbox_inches='tight', dpi=600)
plt.show()

##################################
##################################




x_mesh, t_mesh = np.meshgrid( data["x"], data["t"], indexing="ij")
plt.rcParams.update({"font.size": 15})
fig_5, ax_5 = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)

ax_5[0].scatter(t_mesh, x_mesh, c=data["usol"],  s=10, label=r"Entire data set" )

scatter = ax_5[1].scatter(X_str_train[:, 1]*20, 30*(X_str_train[:, 0]+1)-30, c=u_str_train[:, 0],
                          marker="o",
    s=20,
    label=r"Greedy samples: \texttt{Q-DEIM}",)

ax_5[0].set(xlabel=r"$t$")
ax_5[0].set(ylabel=r"$x$")

ax_5[1].set(xlabel=r"$t$")
#ax_5[1].set_ylabel("x")


#for j in range(2):
#    ax_5[j].yaxis.set_label_coords(-0.12, 0.5)

ax_5[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=12)

#fig_5.text(0.1, 1, "Your Text", fontsize=12, color='red', ha='center', va='center')

#for ax in ax_5:
#    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=12)

plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed
colorbar = fig_5.colorbar(scatter, ax=ax_5, location="right", aspect=50, fraction=0.02)


plt.savefig(filename_data + "dataset" + " Q_Deim_samples" + ".png", bbox_inches='tight', dpi=600)
plt.savefig(filename_data + "dataset" + " Q_Deim_samples" + ".pdf", bbox_inches='tight', dpi=600)
plt.show()






















