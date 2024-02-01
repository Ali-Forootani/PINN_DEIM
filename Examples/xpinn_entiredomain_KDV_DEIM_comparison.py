#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:18:25 2023

@author: forootani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:19:09 2023

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
from PINN_DEIM.Functions.utils import leastsquares_fit, loss_func_KDV
from PINN_DEIM.Functions.library import library_deriv, features_calc_KDV, calculate_theta_KDV
from PINN_DEIM.Functions import plot_config_file
from PINN_DEIM.Functions.ensemble_loader import LoaderProcessor, process_loader

from PINN_DEIM.Functions.DEIM_class import DEIM
from PINN_DEIM.Functions.data_set_preparation import KDVEquationEntireDomain, KDVEquationDomainDecomposition
from PINN_DEIM.Functions.deep_simulation_framework import PDEDeepModelSplitDomain, PDEDeepModel
from PINN_DEIM.Functions.ensemble_trainer import Trainer, EnsembleTrainerSplitDomain


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
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

####################### loading the data set
#############################################

data = scipy.io.loadmat(root_dir + "/data/kdv.mat")


"""
Defining an instance of KDV PDE, and spliting its time domain by the
indeces that is written, e.g.=101
"""

KDV_instance = KDVEquationDomainDecomposition(data, time_indices=101)

"""
the arguments are:
t_div: number of time division for the first ensemble, e.g. 2
n_basis: number of selected basis for the first ensemble, e.g. 5
"""


t_div = [2, 2]

(x_train_list, u_train_list, train_loaders) = KDV_instance.prepare_data_DEIM(
    t_div,
    n_basis=[1, 1],
    deim_tolerance=1e-4)


"""
train data sets for each ensemble with their associated torch tensor loader
"""

X_str_train_1 = x_train_list[0]
X_str_train_2 = x_train_list[1]

u_str_train_1 = u_train_list[0]
u_str_train_2 = u_train_list[1]

train_loader_1 = train_loaders[0]
train_loader_2 = train_loaders[1]

############################################

fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8, 4))
ax[0].scatter(
    X_str_train_1[:, 1], X_str_train_1[:, 0], c=u_str_train_1[:, 0], marker="x", s=30)

ax[1].scatter(
    X_str_train_2[:, 1], X_str_train_2[:, 0], c=u_str_train_2[:, 0], marker="x", s=30)

plt.show()

#############################################


noise = 0
learning_rate_inr = 1e-5
hidden_features_str = 128
hidden_features = 128
hidden_layers = 3
num_epochs = 750

string_f = KDV_instance.save_string(
    noise=0,
    learning_rate_inr=1e-5,
    hidden_features_str=128,
    hidden_layers=3,
    num_epochs=750,
    random_sampling=False,
)


### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "./logs/result_xpinn_entiredomain_comparison_KDV/results" + string_f
p = Path(result_path)
if not p.exists():
    os.makedirs(result_path)

############################################
############################################

"""
Defining deep NN structure for KDV model, i.e. optimizer, scheduler, DNN model

Spliting the domain into 2 and defining separate NN for each split
"""

KDVDeepModel_instance = PDEDeepModelSplitDomain(
                 in_features= 2,
                 out_features=1,
                 hidden_features_str=128, 
                 hidden_layers=3,
                 learning_rate_inr=1e-5) 

models_list, optim_adam, scheduler = KDVDeepModel_instance.run()

model_str_1 = models_list[0]
model_str_2 = models_list[1]


Ensem_Train_inst = EnsembleTrainerSplitDomain(
    model_str_1,
    model_str_2,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
    features_calc = features_calc_KDV,
    calculate_theta = calculate_theta_KDV,
    loss_func= loss_func_KDV 
)

loss_func_list, coeff_track_list = Ensem_Train_inst.train_func(
    train_loader_1, train_loader_2)

################################################
################################################

coef_s_1 = coeff_track_list[0]
coef_s_2 = coeff_track_list[1]

loss_tot_s1 = loss_func_list[0]
loss_tot_s2 = loss_func_list[1]

coef_s_tr1 = np.reshape(np.array(coef_s_1), (num_epochs, coef_s_1[0].shape[0]))
coef_s_tr2 = np.reshape(np.array(coef_s_2), (num_epochs, coef_s_2[0].shape[0]))


###############################################
###############################################

"""
file path, saving the data and related parameters
"""
(
    filename_loss_funcs,
    filename_dict,
    filename_coef_dict,
    filename_model_dict
) = KDV_instance.file_path_saving(result_path)

###############################################
###############################################

plt.rcParams.update({"font.size": 20,})

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))

x_ticks = [0, 250, 500, 750]

line_width = 3

axs2.semilogy(
    loss_tot_s1,
    marker="^",
    markersize=5,
    linewidth=line_width,
    markevery=100,
    linestyle="-",
)

axs2.semilogy(
    loss_tot_s2,
    marker="s",
    markersize=5,
    linewidth=line_width,
    markevery=100,
    linestyle=":",
)


axs2.set_xticks(x_ticks)


axs2.set(xlabel="Iteration")
plt.title("Loss functions")

plt.tight_layout()
plt.savefig(
    filename_loss_funcs + "loss_function" + ".png",
    dpi=600,
)
plt.savefig(
    filename_loss_funcs + "loss_function" + ".pdf",
    dpi=600,
)

np.save(filename_dict["s1"], loss_tot_s1)
np.save(filename_dict["s2"], loss_tot_s2)


##########################################
##########################################
#####  Saving the coefficients of the simulation

coef_s_tr1 = np.reshape(np.array(coef_s_1), (num_epochs, -1))
coef_s_tr2 = np.reshape(np.array(coef_s_2), (num_epochs, -1))

filename_coef_s1 = filename_coef_dict["coef_s_tr1"]
filename_coef_s2 = filename_coef_dict["coef_s_tr2"]

np.save(filename_coef_s1, coef_s_tr1)
np.save(filename_coef_s2, coef_s_tr2)



#####################################################################
#####################################################################
####################################
#################################### Simulating the entire domain

"""
Defining deep NN structure for KDV model, i.e. optimizer, scheduler, DNN model
Without decomposing the domain
"""

KDV_no_domain_decompose = KDVEquationEntireDomain(data)
(x_train_list, u_train_list, train_loaders)= KDV_no_domain_decompose.prepare_data_DEIM(
                                                    t_div=2,
                                                    n_basis=1,
                                                    deim_tolerance=1e-3
                                                    )

X_str_train = x_train_list[0]
u_str_train = u_train_list[0]
train_loader = train_loaders[0]


KDVDeepModel_no_domain_decompose = PDEDeepModel(
    in_features=2,
    out_features=1,
    hidden_features_str=128,
    hidden_layers=3,
    learning_rate_inr=1e-5,)

models_list, optim_adam, scheduler = KDVDeepModel_no_domain_decompose.run()

model_str = models_list[0]


Train_no_domain_decompose = Trainer(
    model_str,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
    features_calc = features_calc_KDV,
    calculate_theta = calculate_theta_KDV,
    loss_func = loss_func_KDV
)


loss_func_list, coeff_track_list = Train_no_domain_decompose.train_func(
    train_loader)

coef_s = coeff_track_list[0]
loss_tot_s = loss_func_list[0]
coef_s_tr = np.reshape(np.array(coef_s), (num_epochs, coef_s[0].shape[0]))



string_f_no_domain_decompose = KDV_no_domain_decompose.save_string(
    noise=0,
    learning_rate_inr=1e-5,
    hidden_features_str=128,
    hidden_layers=3,
    num_epochs=750,
    random_sampling=False,)


(
    filename_loss_funcs_no_domain_decompose,
    filename_coef_dict_no_domain_decompose,
    filename_dict_no_domain_decompose,
    filename_data_no_domain_decompose
) = KDV_no_domain_decompose.file_path_saving(result_path)

np.save(filename_coef_dict_no_domain_decompose, coef_s_tr)


###################################################################
###################################################################
###################################################################

plt.rcParams.update({"font.size": 20,})

line_width = 4

fig_2, axs_2 = plt.subplots(1, 2, sharey=False, figsize=(9, 3.5))

axs_2[0].plot(coef_s_tr[:, 0], linewidth=line_width, linestyle="-", color=colors[0])
axs_2[0].plot(coef_s_tr1[:, 0], linewidth=line_width, linestyle="--", color=colors[1])
axs_2[0].plot(coef_s_tr2[:, 0], linewidth=line_width, linestyle=":", color=colors[2])


axs_2[1].plot(coef_s_tr[:, 1], linewidth=line_width, linestyle="-", color=colors[0],
    label=r"Entire domain",)
axs_2[1].plot(coef_s_tr1[:, 1], linewidth=line_width, linestyle="--", color=colors[1],
    label=r"Sub-domain $1$ ",)
axs_2[1].plot(coef_s_tr2[:, 1], linewidth=line_width, linestyle=":", color=colors[2],
    label=r"Sub-domain $2$ ",)

plt.subplots_adjust(wspace=0.5, hspace=0.5)

x_ticks = [0, 250, 500, 750]
y_ticks = [1, 0, -1, -2]

from matplotlib.ticker import FuncFormatter

# Define a custom formatting function for y-axis tick labels
def sci_fmt_1(x, pos):
    return "${:.2f}$".format(x)

def sci_fmt_2(x, pos):
    return "${:.0f}$".format(x)


for j in range(2):
    if j == 0:
        axs_2[j].set_ylim([-12, 12])
        axs_2[j].set_xticks(x_ticks)
        axs_2[j].set_yticks([-12, -6, 0, 6, 12])
        axs_2[j].yaxis.set_major_formatter(FuncFormatter(sci_fmt_1))
        axs_2[j].set(xlabel="Iteration")
        axs_2[j].set(ylabel=r"Coefficients $u_{xx}$")
        
    if j == 1:
        axs_2[j].set_ylim([-2, 1])
        axs_2[j].set_yticks(y_ticks)
        axs_2[j].set_xticks(x_ticks)
        axs_2[j].set(xlabel="Iteration")
        axs_2[j].yaxis.set_major_formatter(FuncFormatter(sci_fmt_2))
        axs_2[j].set(ylabel=r"Coefficients $u,\ u^3$")
        
####################################################

lines, labels = [], []
for ax in fig_2.axes:
    line, label = ax.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)

fig_2.legend(
    lines,
    labels,
    loc="center",  # Change the location to upper center
    ncol=4,
    bbox_to_anchor=(0.54, 0.8),  # Adjust the coordinates
    bbox_transform=fig.transFigure,
    fontsize=15,
    frameon=True,)

plt.tight_layout()
##########################################################
##########################################################

plt.savefig(filename_data_no_domain_decompose + "_comparison_domain_split_no_decompose_" 
            + "_DEIM_sampling" + ".png", dpi=300,  bbox_inches="tight")

plt.savefig(filename_data_no_domain_decompose + "_comparison_domain_split_no_decompose_" 
            + "_DEIM_sampling" + ".pdf", dpi=300, bbox_inches="tight")
plt.show()






