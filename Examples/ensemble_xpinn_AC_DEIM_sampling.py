#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:42:41 2023

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
from PINN_DEIM.Functions.utils import (loss_func_AC, leastsquares_fit,
    equation_residual_AC,)
from PINN_DEIM.Functions.library import library_deriv
from PINN_DEIM.Functions import plot_config_file
from PINN_DEIM.Functions.ensemble_loader import LoaderProcessor, process_loader
from PINN_DEIM.Functions.library import (
    features_calc_AC, calculate_theta_AC, library_deriv,)
from PINN_DEIM.Functions.DEIM_class import DEIM
from PINN_DEIM.Functions.data_set_preparation import AllenCahnEquation
from PINN_DEIM.Functions.deep_simulation_framework import AllenCahnDeepModel
from PINN_DEIM.Functions.ensemble_trainer import EnsembleTrainer


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

data = scipy.io.loadmat(root_dir + "/data/AC.mat")

#######################################
#######################################

"""
Defining an instance of Allen Cahn PDE, and spliting its time domain by the
indeces that is written, e.g.=101
"""

AllenCahn_instance = AllenCahnEquation(data, time_indices=101)

"""
the arguments are:
t_div_s1: number of time division for the first ensemble, e.g. 2
n_basis_s1: number of selected basis for the first ensemble, e.g. 5
"""

(x_train_list, u_train_list, train_loaders) = AllenCahn_instance.prepare_data_DEIM(
    t_div_s1=1,
    t_div_f1=2,
    t_div_s2=1,
    t_div_f2=3,
    n_basis_s1=1,
    n_basis_f1=1,
    n_basis_s2=1,
    n_basis_f2=1,
    tolerance = 1e-4
)


"""
train data sets for each ensemble with their associated torch tensor loader
"""

X_str_train_1 = x_train_list[0]
X_str_train_f1 = x_train_list[1]
X_str_train_2 = x_train_list[2]
X_str_train_f2 = x_train_list[3]

u_str_train_1 = u_train_list[0]
u_str_train_f1 = u_train_list[1]
u_str_train_2 = u_train_list[2]
u_str_train_f2 = u_train_list[3]

train_loader_1 = train_loaders[0]
train_loader_f1 = train_loaders[1]
train_loader_2 = train_loaders[2]
train_loader_f2 = train_loaders[3]

########################################
########################################


fig, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 8))

ax[0, 0].scatter(
    X_str_train_1[:, 1], X_str_train_1[:, 0], c=u_str_train_1[:, 0], marker="x", s=12
)
ax[1, 0].scatter(
    X_str_train_f1[:, 1], X_str_train_f1[:, 0], c=u_str_train_f1[:, 0], marker="x", s=12
)

ax[0, 1].scatter(
    X_str_train_2[:, 1], X_str_train_2[:, 0], c=u_str_train_2[:, 0], marker="x", s=12
)
ax[1, 1].scatter(
    X_str_train_f2[:, 1], X_str_train_f2[:, 0], c=u_str_train_f2[:, 0], marker="x", s=12
)

plt.show()

########################################
########################################

noise = 0
learning_rate_inr = 1e-5
hidden_features_str = 128
hidden_features = 64
hidden_layers = 3
num_epochs = 1500

string_f = AllenCahn_instance.save_string(
    noise=0,
    learning_rate_inr=1e-5,
    hidden_features_str=128,
    hidden_layers=3,
    num_epochs=1500,
    random_sampling=False,
)


### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "./result_xpinn_AC_1d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.makedirs(result_path)

#######################################
#######################################

"""
Defining deep NN structure for AC model, i.e. optimizer, scheduler, DNN model 
"""

AllenCahnDeepModel_instance = AllenCahnDeepModel(
                 in_features= 2,
                 out_features=1,
                 hidden_features_str=128, 
                 hidden_layers=3,
                 learning_rate_inr=1e-5) 

models_list, optim_adam, scheduler = AllenCahnDeepModel_instance.run()


model_str_1 = models_list[0]
model_str_f1 = models_list[1]
model_str_2 = models_list[2]
model_str_f2 = models_list[3]


########################################
########### Training the deep neural network



Ensem_Train_inst = EnsembleTrainer(
    model_str_1,
    model_str_f1,
    model_str_2,
    model_str_f2,
    num_epochs=num_epochs,
    optim_adam=optim_adam,
    scheduler=scheduler,
    features_calc = features_calc_AC,
    calculate_theta = calculate_theta_AC,
    loss_func= loss_func_AC 
)

loss_func_list, coeff_track_list = Ensem_Train_inst.train_func(
    train_loader_1, train_loader_f1, train_loader_2, train_loader_f2
)

coef_s_1 = coeff_track_list[0]
coef_f_1 = coeff_track_list[1]
coef_s_2 = coeff_track_list[2]
coef_f_2 = coeff_track_list[3]


loss_tot_s1 = loss_func_list[0]
loss_tot_f1 = loss_func_list[1]
loss_tot_s2 = loss_func_list[2]
loss_tot_f2 = loss_func_list[3]


coef_s_tr1 = np.reshape(np.array(coef_s_1), (num_epochs, coef_s_1[0].shape[0]))
coef_f_tr1 = np.reshape(np.array(coef_f_1), (num_epochs, coef_f_1[0].shape[0]))

coef_s_tr2 = np.reshape(np.array(coef_s_2), (num_epochs, coef_s_2[0].shape[0]))
coef_f_tr2 = np.reshape(np.array(coef_f_2), (num_epochs, coef_f_2[0].shape[0]))


########################################
#########  Reshaping the coefficients

coef_s_tr1 = np.reshape(np.array(coef_s_1), (num_epochs, -1))
coef_f_tr1 = np.reshape(np.array(coef_f_1), (num_epochs, -1))

coef_s_tr2 = np.reshape(np.array(coef_s_2), (num_epochs, -1))
coef_f_tr2 = np.reshape(np.array(coef_f_2), (num_epochs, -1))


###############################################
###############################################


"""
file path, saving the data and related parameters
"""
(
    filename_loss_funcs,
    filename_l_s1,
    filename_l_s2,
    filename_l_f1,
    filename_l_f2,
    filename_coef_s1,
    filename_l_s2,
    filename_l_f1,
    filename_l_f2,
    filename_coef_s1,
    filename_coef_f1,
    filename_coef_s2,
    filename_coef_f2,
    filename_model_s1,
    filename_model_f1,
    filename_model_s2,
    filename_model_f2,
    filename_data,
) = AllenCahn_instance.file_path_saving(result_path)


#################################################
##################### Plotting Loss Functions ###


plt.rcParams.update(
    {
        "font.size": 20,
    }
)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))

x_ticks = [0, 500, 1000, 1500]

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
    loss_tot_f1,
    marker="8",
    markersize=5,
    linewidth=line_width,
    markevery=100,
    linestyle="--",
)
axs2.semilogy(
    loss_tot_s2,
    marker="s",
    markersize=5,
    linewidth=line_width,
    markevery=100,
    linestyle=":",
)
axs2.semilogy(
    loss_tot_f2,
    marker="o",
    markersize=5,
    linewidth=line_width,
    markevery=100,
    linestyle="-.",
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

np.save(filename_l_s1, loss_tot_s1)
np.save(filename_l_f1, loss_tot_f1)
np.save(filename_l_s2, loss_tot_s2)
np.save(filename_l_f2, loss_tot_f2)

###################################
###################################

coef_s_tr1 = np.reshape(np.array(coef_s_1), (num_epochs, -1))
coef_f_tr1 = np.reshape(np.array(coef_f_1), (num_epochs, -1))

coef_s_tr2 = np.reshape(np.array(coef_s_2), (num_epochs, -1))
coef_f_tr2 = np.reshape(np.array(coef_f_2), (num_epochs, -1))

np.save(filename_coef_s1, coef_s_tr1)
np.save(filename_coef_f1, coef_f_tr1)
np.save(filename_coef_s2, coef_s_tr2)
np.save(filename_coef_f2, coef_f_tr2)

####################################
#################################### Plotting the convergence of coefficients

plt.rcParams.update(
    {
        "font.size": 20,
    }
)

line_width = 4

fig, axs = plt.subplots(2, 4, sharey=False, figsize=(13, 6))

axs[0, 0].plot(coef_s_tr1[:, 0], linewidth=line_width, linestyle="-", color=colors[0])
axs[0, 1].plot(coef_f_tr1[:, 0], linewidth=line_width, linestyle="--", color=colors[1])
axs[0, 2].plot(coef_s_tr2[:, 0], linewidth=line_width, linestyle=":", color=colors[2])
axs[0, 3].plot(coef_f_tr2[:, 0], linewidth=line_width, linestyle="-.", color=colors[3])

axs[1, 0].plot(
    coef_s_tr1[:, 1],
    linewidth=line_width,
    linestyle="-",
    color=colors[0],
    label=r"\texttt{Ensemb}$_1$-Subdom$_1$",
)
axs[1, 1].plot(
    coef_f_tr1[:, 1],
    linewidth=line_width,
    linestyle="--",
    color=colors[1],
    label=r"\texttt{Ensemb}$_2$-Subdom$_1$",
)
axs[1, 2].plot(
    coef_s_tr2[:, 1],
    linewidth=line_width,
    linestyle=":",
    color=colors[2],
    label=r"\texttt{Ensemb}$_1$-Subdom$_2$",
)
axs[1, 3].plot(
    coef_f_tr2[:, 1],
    linewidth=line_width,
    linestyle="-.",
    color=colors[3],
    label=r"\texttt{Ensemb}$_2$-Subdom$_2$",
)

plt.subplots_adjust(wspace=0.5, hspace=0.5)

x_ticks = [0, 750, 1500]
y_ticks = [0, -1, -3, -5]

from matplotlib.ticker import FuncFormatter


# Define a custom formatting function for y-axis tick labels
def sci_fmt_1(x, pos):
    return "${:.2f}$".format(x)


def sci_fmt_2(x, pos):
    return "${:.0f}$".format(x)


# Apply the custom formatting function to the y-axis tick labels
# for ax in axs.flat:
#    ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt_1))

for i in range(2):
    for j in range(4):
        if i == 0 and j <= 3:
            axs[i, j].set_ylim([-0.03, 0.02])
            axs[i, j].set_xticks(x_ticks)

            axs[i, j].yaxis.set_major_formatter(FuncFormatter(sci_fmt_1))

            if j == 0:
                axs[i, j].set(ylabel=r"Coefficients $u_{xx}$")
        # if i == 0 and j > 1 :
        #    axs[i,j].set_ylim([-0.01, 0.005])
        if i == 1:
            axs[i, j].set_ylim([-6, 1])
            axs[i, j].set_yticks(y_ticks)
            axs[i, j].set_xticks(x_ticks)
            axs[i, j].set(xlabel="Iteration")
            axs[i, j].yaxis.set_major_formatter(FuncFormatter(sci_fmt_2))
            if j == 0:
                axs[i, j].set(ylabel=r"Coefficients $u,\ u^3$")


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
    bbox_to_anchor=(0.51, 0.97),  # Adjust the coordinates
    bbox_transform=fig.transFigure,
    fontsize=15,
    frameon=True,
)


plt.savefig(
    filename_coef_s1 + "coefficients" + ".png",
    dpi=300,
)
plt.savefig(
    filename_coef_s1 + "coefficients" + ".pdf",
    dpi=300,
)

####################################
####################################

"""
print(coefficients_f2)
print(coefficients_f1)
print(coefficients_2)
print(coefficients_1)
"""

###################################
###################################


fig_5, ax_5 = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
scatter1 = ax_5[1, 1].scatter(
    X_str_train_f2[:, 1],
    X_str_train_f2[:, 0],
    c=u_str_train_f2[:, 0],
    marker="x",
    s=15,
    label=r"\texttt{Ensemb}$_2$-Subdom$_2$",
)
scatter2 = ax_5[1, 0].scatter(
    X_str_train_2[:, 1],
    X_str_train_2[:, 0],
    c=u_str_train_2[:, 0],
    marker="o",
    s=15,
    label=r"\texttt{Ensemb}$_1$-Subdom$_2$",
)
scatter3 = ax_5[0, 1].scatter(
    X_str_train_f1[:, 1],
    X_str_train_f1[:, 0],
    c=u_str_train_f1[:, 0],
    marker="^",
    s=15,
    label=r"\texttt{Ensemb}$_2$-Subdom$_1$",
)
scatter4 = ax_5[0, 0].scatter(
    X_str_train_1[:, 1],
    X_str_train_1[:, 0],
    c=u_str_train_1[:, 0],
    marker="D",
    s=15,
    label=r"\texttt{Ensemb}$_1$-Subdom$_1$",
)

ax_5[1, 0].set_xlabel("t")
ax_5[1, 0].set_ylabel("x")
ax_5[1, 1].set_xlabel("t")
ax_5[0, 0].set_ylabel("x")

# Add individual legends to each subplot
ax_5[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=20)
ax_5[1, 0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=20)
ax_5[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=20)
ax_5[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=20)

plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed

colorbar = fig_5.colorbar(
    scatter1, ax=ax_5.ravel().tolist(), location="right", aspect=50, fraction=0.01
)
# fig_5.suptitle('Sampled Data set')

plt.savefig(filename_data + "dataset_ensemble" + ".png", dpi=300)
plt.savefig(filename_data + "dataset_ensemble" + ".pdf", dpi=300)
plt.show()

##########################################
##########################################


N_t = 1000
t = data["tt"].flatten()
x = data["x"].flatten()[:, None]
Exact = np.real(data["uu"])
X, T = np.meshgrid(x, t, indexing="ij")
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]


X_str_train_t, X_test, u_str_train_t, u_test = train_test_split(
    X_star, u_star, test_size=0.9, random_state=42
)


idx = np.random.choice(X_test.shape[0], N_t, replace=False)

X_str_test = X_test[idx, :]
u_str_test = u_test[idx, :]

X_str_test_torch = torch.tensor(X_str_test).float().to(device)
u_str_test_torch = torch.tensor(u_str_test).float().to(device)

u_pred_test_torch = model_str_2(X_str_test_torch)
error = (u_str_test_torch - u_pred_test_torch).cpu().detach().numpy()

plt.rcParams.update(
    {
        "font.size": 20,
    }
)
fig_6, ax_6 = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, sharex=True)

scatter1 = ax_6[0].scatter(
    X_str_test[:, 1], X_str_test[:, 0], c=u_str_test[:, 0], marker="x", s=10
)
scatter2 = ax_6[1].scatter(
    X_str_test[:, 1],
    X_str_test[:, 0],
    c=u_pred_test_torch.cpu().detach().numpy()[:, 0],
    marker="o",
    s=10,
)
scatter3 = ax_6[2].scatter(
    X_str_test[:, 1], X_str_test[:, 0], c=np.abs(error[:, 0]), marker="*", s=10
)

import matplotlib.cm as cm

norm = plt.Normalize(vmin=min(u_str_test[:, 0]), vmax=max(u_str_test[:, 0]))
cmap = cm.get_cmap("viridis")
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

norm_e = plt.Normalize(vmin=min(np.abs(error[:, 0])), vmax=max(np.abs(error[:, 0])))
cmap = cm.get_cmap("viridis")
sm_e = cm.ScalarMappable(cmap=cmap, norm=norm_e)
sm_e.set_array([])

# colorbar = fig_6.colorbar(scatter1, ax=ax_5.ravel().tolist(), location='right', aspect=50, fraction=0.01)
import matplotlib.ticker as ticker

cbar1 = fig_6.colorbar(mappable=scatter1, ax=ax_6[0], ticks=ticker.MultipleLocator(0.5))
cbar2 = fig_6.colorbar(mappable=scatter2, ax=ax_6[1], ticks=ticker.MultipleLocator(0.5))
cbar3 = fig_6.colorbar(mappable=scatter3, ax=ax_6[2], ticks=ticker.MultipleLocator(0.2))

ax_6[0].set_title("Test data set")
ax_6[1].set_title("Prediction")
ax_6[2].set_title("Absolute Error")
ax_6[0].set_xlabel("t", fontsize=20)
ax_6[1].set_xlabel("t", fontsize=20)
ax_6[2].set_xlabel("t", fontsize=20)

ax_6[0].set_ylabel("x", fontsize=20)
plt.tight_layout()

plt.savefig(
    filename_data + "prediction_error" + ".png",
    dpi=300,
)
plt.savefig(
    filename_data + "prediction_error" + ".pdf",
    dpi=300,
)
plt.show()
