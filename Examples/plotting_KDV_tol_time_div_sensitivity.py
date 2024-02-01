#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:44:58 2023

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
    loss_func_KDV,
    leastsquares_fit,
    equation_residual_AC,
)
from PINN_DEIM.Functions.library import library_deriv
from PINN_DEIM.Functions import plot_config_file
from PINN_DEIM.Functions.ensemble_loader import LoaderProcessor, process_loader
from PINN_DEIM.Functions.library import (
    features_calc_KDV,
    calculate_theta_KDV,
    library_deriv,
)
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


warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


"""

## This module plots the curves that compare GS-PINN with RS-PINN for Allen-Cahn PDE
## It loads the data from the directory that previous simulations are done then
## it plots the comparison curves


Important note: This file is only a shortcut to plot the already existed data points
to avoid tedious simulation which takes a few hours

### You have to run `KDV_data_generation.py` first then you can run this module

"""


string_f = "_KDV_" + "comparing_" + "different_time_division_" + "different_tol_"


### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "./logs/result_DEIM_KDV_1d/results" + string_f

"""
loading the saved data from the directroy that were saved
"""

filename_coef_random_samp = result_path + "/" + "KDV_coef_dic_list_random_samp" + ".npy"
filename_data_random_points = result_path + "/" + "KDV_list_data_random_points" + ".npy"
filename_coef_track_random_samp = (
    result_path + "/" + "KDV_coef_dic_track_list_random_samp" + ".npy"
)

KDV_Random_points = np.load(filename_data_random_points, allow_pickle=True)
coef_dic_list_random_samp = np.load(filename_coef_random_samp, allow_pickle=True)
coef_s_tr_list_random_samp = np.load(filename_coef_track_random_samp, allow_pickle=True)

filename_coef_DEIM = (
    result_path + "/" + "KDV_coef_dic_list_diff_tol_diff_tim_DEIM" + ".npy"
)
filename_data_t_v_tol_DEIM = (
    result_path + "/" + "KDV_list_data_diff_tol_diff_tim_DEIM" + ".npy"
)
filename_coef_track_DEIM = (
    result_path + "/" + "KDV_coef_track_dic_list_diff_tol_diff_tim_DEIM" + ".npy"
)

coef_dic_list_DEIM = np.load(filename_coef_DEIM, allow_pickle=True)
list_KDV_data_DEIM_t_div_tol = np.load(filename_data_t_v_tol_DEIM, allow_pickle=True)
coef_s_tr_list_DEIM = np.load(filename_coef_track_DEIM, allow_pickle=True)


"""
first
"""
tol_list = np.logspace(-10, -2, 20)


coef_1_ref = -6
coef_2_ref = -1

coef_t_div_1 = []
num_samp_t_div_1 = []
er_coef_1_t_div_1 = []
er_coef_2_t_div_1 = []


coef_t_div_2 = []
num_samp_t_div_2 = []
er_coef_1_t_div_2 = []
er_coef_2_t_div_2 = []


coef_t_div_3 = []
num_samp_t_div_3 = []
er_coef_1_t_div_3 = []
er_coef_2_t_div_3 = []


coef_t_div_4 = []
num_samp_t_div_4 = []
er_coef_1_t_div_4 = []
er_coef_2_t_div_4 = []


for j in range(1, 5):
    for i in range(len(tol_list)):
        if j == 1:
            coef_t_div_1.append(coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"])
            num_samp_t_div_1.append(coef_dic_list_DEIM[(j - 1) * 20 + i]["num_samples"])
            er_coef_1_t_div_1.append(
                np.abs(
                    (coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"][0] - coef_1_ref)
                    / coef_1_ref
                )
            )

            er_coef_2_t_div_1.append(
                np.abs(
                    (coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"][1] - coef_2_ref)
                    / coef_2_ref
                )
            )

        if j == 2:
            coef_t_div_2.append(coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"])
            num_samp_t_div_2.append(coef_dic_list_DEIM[(j - 1) * 20 + i]["num_samples"])

            er_coef_1_t_div_2.append(
                np.abs(
                    (coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"][0] - coef_1_ref)
                    / coef_1_ref
                )
            )

            er_coef_2_t_div_2.append(
                np.abs(
                    (coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"][1] - coef_2_ref)
                    / coef_2_ref
                )
            )

        if j == 3:
            coef_t_div_3.append(coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"])
            num_samp_t_div_3.append(coef_dic_list_DEIM[(j - 1) * 20 + i]["num_samples"])

            er_coef_1_t_div_3.append(
                np.abs(
                    (coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"][0] - coef_1_ref)
                    / coef_1_ref
                )
            )

            er_coef_2_t_div_3.append(
                np.abs(
                    (coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"][1] - coef_2_ref)
                    / coef_2_ref
                )
            )

        if j == 4:
            coef_t_div_4.append(coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"])
            num_samp_t_div_4.append(coef_dic_list_DEIM[(j - 1) * 20 + i]["num_samples"])

            er_coef_1_t_div_4.append(
                np.abs(
                    (coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"][0] - coef_1_ref)
                    / coef_1_ref
                )
            )

            er_coef_2_t_div_4.append(
                np.abs(
                    (coef_dic_list_DEIM[(j - 1) * 20 + i]["coeff"][1] - coef_2_ref)
                    / coef_2_ref
                )
            )


plt.rcParams.update(
    {
        "font.size": 20,
    }
)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
line_width = 2


#############################################
#############################################
"""
Figure I
"""

fig, axs = plt.subplots(1, 2, sharey=False, figsize=(12, 5))
axs[0].loglog(
    np.array(num_samp_t_div_1),
    er_coef_1_t_div_1,
    linewidth=line_width,
    linestyle="--",
    marker="*",
    color=colors[4],
    markersize=10,
)
axs[0].loglog(
    np.array(num_samp_t_div_2),
    er_coef_1_t_div_2,
    linewidth=line_width,
    linestyle="--",
    marker="o",
    color=colors[1],
    markersize=6,
)
axs[0].loglog(
    np.array(num_samp_t_div_3),
    er_coef_1_t_div_3,
    linewidth=line_width,
    linestyle="--",
    marker="D",
    color=colors[2],
    markersize=6,
)
axs[0].loglog(
    np.array(num_samp_t_div_4),
    er_coef_1_t_div_4,
    linewidth=line_width,
    linestyle="--",
    marker="^",
    color=colors[3],
    markersize=6,
)

axs[0].set(xlabel="Number of samples")
# axs[0].set(ylabel=r"Coefficient $u_{xx}$")
axs[0].set(ylabel=r"$\texttt{error}\ uu_{x}$")


axs[1].loglog(
    np.array(num_samp_t_div_1),
    er_coef_2_t_div_1,
    linewidth=line_width,
    linestyle="--",
    marker="*",
    color=colors[4],
    markersize=10,
    label=r"$ \texttt{t}_{\texttt{div}} = 1$",
)
axs[1].loglog(
    np.array(num_samp_t_div_2),
    er_coef_2_t_div_2,
    linewidth=line_width,
    linestyle="--",
    marker="o",
    color=colors[1],
    markersize=6,
    label=r"$ \texttt{t}_{\texttt{div}} = 2$",
)
axs[1].loglog(
    np.array(num_samp_t_div_3),
    er_coef_2_t_div_3,
    linewidth=line_width,
    linestyle="--",
    marker="D",
    color=colors[2],
    markersize=6,
    label=r"$ \texttt{t}_{\texttt{div}} = 3$",
)
axs[1].loglog(
    np.array(num_samp_t_div_4),
    er_coef_2_t_div_4,
    linewidth=line_width,
    linestyle="--",
    marker="^",
    color=colors[3],
    markersize=6,
    label=r"$ \texttt{t}_{\texttt{div}} = 4$",
)

axs[1].set(xlabel="Number of samples")
axs[1].set(ylabel=r"$\texttt{error}\ u_{xxx}$")


lines, labels = [], []
for ax in fig.axes:
    line, label = ax.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)
    ax.grid(True)

fig.legend(
    lines,
    labels,
    loc="center",  # Change the location to upper center
    ncol=4,
    bbox_to_anchor=(0.53, 1),  # Adjust the coordinates
    bbox_transform=fig.transFigure,
    fontsize=20,
    frameon=True,
)


plt.tight_layout()


plt.savefig(
    result_path + "/" + "KDV_relative_abs_error_" + "GSPINN" + ".png",
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    result_path + "/" + "KDV_relative_abs_error_" + "GSPINN" + ".pdf",
    dpi=600,
    bbox_inches="tight",
)


#################################################################
#################################################################
#################################################################


data = scipy.io.loadmat(root_dir + "/data/kdv.mat")
t = data["t"]
x = data["x"]
Exact = data["usol"]


min_num_samples = min(
    min(num_samp_t_div_1),
    min(num_samp_t_div_2),
    min(num_samp_t_div_3),
    min(num_samp_t_div_4),
)

max_num_samples = max(
    max(num_samp_t_div_1),
    max(num_samp_t_div_2),
    max(num_samp_t_div_3),
    max(num_samp_t_div_4),
)
step_size = (max_num_samples - min_num_samples) / 10
org_data_size = data["usol"].shape[0] * data["usol"].shape[1]
list_AC_data_random_points = []


coef_t_rand_exp_1 = []
num_samp_t_rand_exp_1 = []
er_coef_1_t_rand_exp_1 = []
er_coef_2_t_rand_exp_1 = []


coef_t_rand_exp_2 = []
num_samp_t_rand_exp_2 = []
er_coef_1_t_rand_exp_2 = []
er_coef_2_t_rand_exp_2 = []


coef_t_rand_exp_3 = []
num_samp_t_rand_exp_3 = []
er_coef_1_t_rand_exp_3 = []
er_coef_2_t_rand_exp_3 = []


coef_t_rand_exp_4 = []
num_samp_t_rand_exp_4 = []
er_coef_1_t_rand_exp_4 = []
er_coef_2_t_rand_exp_4 = []


coef_t_rand_exp_5 = []
num_samp_t_rand_exp_5 = []
er_coef_1_t_rand_exp_5 = []
er_coef_2_t_rand_exp_5 = []

m = 0


for k in range(1, 6):
    for n in range(0, 11):
        if coef_dic_list_random_samp[(k - 1) * 11 + n]["experiment_index"] == 1:
            coef_t_rand_exp_1.append(
                coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"]
            )

            num_samp_t_rand_exp_1.append(
                coef_dic_list_random_samp[(k - 1) * 11 + n]["num_samples"]
            )

            er_coef_1_t_rand_exp_1.append(
                np.abs(
                    (
                        coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"][0]
                        - coef_1_ref
                    )
                    / coef_1_ref
                )
            )

            er_coef_2_t_rand_exp_1.append(
                np.abs(
                    (
                        coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"][1]
                        - coef_2_ref
                    )
                    / coef_2_ref
                )
            )

        if coef_dic_list_random_samp[(k - 1) * 11 + n]["experiment_index"] == 2:
            coef_t_rand_exp_2.append(
                coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"]
            )

            num_samp_t_rand_exp_2.append(
                coef_dic_list_random_samp[(k - 1) * 11 + n]["num_samples"]
            )

            er_coef_1_t_rand_exp_2.append(
                np.abs(
                    (
                        coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"][0]
                        - coef_1_ref
                    )
                    / coef_1_ref
                )
            )

            er_coef_2_t_rand_exp_2.append(
                np.abs(
                    (
                        coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"][1]
                        - coef_2_ref
                    )
                    / coef_2_ref
                )
            )

        if coef_dic_list_random_samp[(k - 1) * 11 + n]["experiment_index"] == 3:
            coef_t_rand_exp_3.append(
                coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"]
            )

            num_samp_t_rand_exp_3.append(
                coef_dic_list_random_samp[(k - 1) * 11 + n]["num_samples"]
            )

            er_coef_1_t_rand_exp_3.append(
                np.abs(
                    (
                        coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"][0]
                        - coef_1_ref
                    )
                    / coef_1_ref
                )
            )

            er_coef_2_t_rand_exp_3.append(
                np.abs(
                    (
                        coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"][1]
                        - coef_2_ref
                    )
                    / coef_2_ref
                )
            )

        if coef_dic_list_random_samp[(k - 1) * 11 + n]["experiment_index"] == 4:
            coef_t_rand_exp_4.append(
                coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"]
            )

            num_samp_t_rand_exp_4.append(
                coef_dic_list_random_samp[(k - 1) * 11 + n]["num_samples"]
            )

            er_coef_1_t_rand_exp_4.append(
                np.abs(
                    (
                        coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"][0]
                        - coef_1_ref
                    )
                    / coef_1_ref
                )
            )

            er_coef_2_t_rand_exp_4.append(
                np.abs(
                    (
                        coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"][1]
                        - coef_2_ref
                    )
                    / coef_2_ref
                )
            )

        if coef_dic_list_random_samp[(k - 1) * 11 + n]["experiment_index"] == 5:
            coef_t_rand_exp_5.append(
                coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"]
            )

            num_samp_t_rand_exp_5.append(
                coef_dic_list_random_samp[(k - 1) * 11 + n]["num_samples"]
            )

            er_coef_1_t_rand_exp_5.append(
                np.abs(
                    (
                        coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"][0]
                        - coef_1_ref
                    )
                    / coef_1_ref
                )
            )

            er_coef_2_t_rand_exp_5.append(
                np.abs(
                    (
                        coef_dic_list_random_samp[(k - 1) * 11 + n]["coeff"][1]
                        - coef_2_ref
                    )
                    / coef_2_ref
                )
            )

    m = m + 1

########################################################
########################################################
########################################################
########################################################


np.array(er_coef_2_t_rand_exp_1)
np.array(er_coef_2_t_rand_exp_2)
np.array(er_coef_2_t_rand_exp_3)
np.array(er_coef_2_t_rand_exp_4)
np.array(er_coef_2_t_rand_exp_5)

rand_2_12 = np.vstack(
    (np.array(er_coef_2_t_rand_exp_1), np.array(er_coef_2_t_rand_exp_2))
)
rand_2_34 = np.vstack(
    (np.array(er_coef_2_t_rand_exp_3), np.array(er_coef_2_t_rand_exp_4))
)
rand_2_1234 = np.vstack((rand_2_12, rand_2_34))
rand_2_12345 = np.vstack((rand_2_1234, np.array(er_coef_2_t_rand_exp_5)))

row_avg_rand_2_12345 = np.mean(rand_2_12345, axis=0)


np.array(er_coef_1_t_rand_exp_1)
np.array(er_coef_1_t_rand_exp_2)
np.array(er_coef_1_t_rand_exp_3)
np.array(er_coef_1_t_rand_exp_4)
np.array(er_coef_1_t_rand_exp_5)

rand_1_12 = np.vstack(
    (np.array(er_coef_1_t_rand_exp_1), np.array(er_coef_1_t_rand_exp_2))
)
rand_1_34 = np.vstack(
    (np.array(er_coef_1_t_rand_exp_3), np.array(er_coef_1_t_rand_exp_4))
)
rand_1_1234 = np.vstack((rand_1_12, rand_1_34))
rand_1_12345 = np.vstack((rand_1_1234, np.array(er_coef_1_t_rand_exp_5)))


row_avg_rand_1_12345 = np.mean(rand_1_12345, axis=0)


############################################################


from sklearn.cluster import KMeans

t_div_12 = np.hstack(
    (
        np.array(num_samp_t_div_1).reshape(-1, 1),
        np.array(er_coef_2_t_div_1).reshape(-1, 1),
    )
)
t_div_22 = np.hstack(
    (
        np.array(num_samp_t_div_2).reshape(-1, 1),
        np.array(er_coef_2_t_div_2).reshape(-1, 1),
    )
)

t_div_32 = np.hstack(
    (
        np.array(num_samp_t_div_3).reshape(-1, 1),
        np.array(er_coef_2_t_div_3).reshape(-1, 1),
    )
)
t_div_42 = np.hstack(
    (
        np.array(num_samp_t_div_4).reshape(-1, 1),
        np.array(er_coef_2_t_div_4).reshape(-1, 1),
    )
)

ent_t_div_2 = np.vstack((t_div_12, t_div_22, t_div_32, t_div_42))

kmeans_t_div_2 = KMeans(n_clusters=20, random_state=0, n_init=100).fit(ent_t_div_2)

kmeans_t_div_2.cluster_centers_[:, 0]
kmeans_t_div_2.cluster_centers_[:, 1]

# plt.plot( np.sort(kmeans_t_div_2.cluster_centers_[:,0],axis=0),
#         np.sort(kmeans_t_div_2.cluster_centers_[:,1],axis=0)[::-1])


t_div_11 = np.hstack(
    (
        np.array(num_samp_t_div_1).reshape(-1, 1),
        np.array(er_coef_1_t_div_1).reshape(-1, 1),
    )
)
t_div_21 = np.hstack(
    (
        np.array(num_samp_t_div_2).reshape(-1, 1),
        np.array(er_coef_1_t_div_2).reshape(-1, 1),
    )
)

t_div_31 = np.hstack(
    (
        np.array(num_samp_t_div_3).reshape(-1, 1),
        np.array(er_coef_1_t_div_3).reshape(-1, 1),
    )
)
t_div_41 = np.hstack(
    (
        np.array(num_samp_t_div_4).reshape(-1, 1),
        np.array(er_coef_1_t_div_4).reshape(-1, 1),
    )
)

ent_t_div_1 = np.vstack((t_div_11, t_div_21, t_div_31, t_div_41))
kmeans_t_div_1 = KMeans(n_clusters=20, random_state=0, n_init=100).fit(ent_t_div_1)
kmeans_t_div_1.cluster_centers_[:, 0]
kmeans_t_div_1.cluster_centers_[:, 1]


########################################################
########################################################
########################################################
#############################################################################
#############################################################################

# plt.style.use('seaborn')
plt.rcParams.update(
    {
        "font.family": "serif.Times New Roman",
        "font.style": "normal",
        "font.variant": "normal",
        "font.weight": "bold",
        "font.stretch": "normal",
        "font.size": 20,
        "legend.fancybox": True,
        "text.usetex": True,
        # "text.latex.preamble": [r'\usepackage{amsmath,amssymb,bm,fontenc,physics,lmodern,nicefrac}']
        "pgf.rcfonts": True,
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": [
            r"\usepackage{amsmath},\usepackage{amssymb},\usepackage{bm},\usepackge{fontenc},\usepackage{physics},\usepackge{lmodern},\usepackage{nicefrac}"
        ],
        "figure.subplot.hspace": 0.25,
    }
)


al_pha = 0.2
line_width = 3


fig_2, axs_2 = plt.subplots(1, 2, sharey=False, figsize=(12, 5))
axs_2[0].loglog(
    np.array(num_samp_t_rand_exp_1),
    er_coef_1_t_rand_exp_1,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)
axs_2[0].loglog(
    np.array(num_samp_t_rand_exp_2),
    er_coef_1_t_rand_exp_2,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)
axs_2[0].loglog(
    np.array(num_samp_t_rand_exp_3),
    er_coef_1_t_rand_exp_3,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)
axs_2[0].loglog(
    np.array(num_samp_t_rand_exp_4),
    er_coef_1_t_rand_exp_4,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)
axs_2[0].loglog(
    np.array(num_samp_t_rand_exp_5),
    er_coef_1_t_rand_exp_5,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)
axs_2[0].loglog(
    np.array(num_samp_t_rand_exp_5),
    row_avg_rand_1_12345,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)

axs_2[0].loglog(
    np.array(num_samp_t_div_1),
    er_coef_1_t_div_1,
    linewidth=line_width,
    linestyle="--",
    marker="*",
    alpha=al_pha,
    markersize=10,
    color="k",
)
axs_2[0].loglog(
    np.array(num_samp_t_div_2),
    er_coef_1_t_div_2,
    linewidth=line_width,
    linestyle="--",
    marker="*",
    alpha=al_pha,
    markersize=10,
    color="k",
)
axs_2[0].loglog(
    np.array(num_samp_t_div_3),
    er_coef_1_t_div_3,
    linewidth=line_width,
    linestyle="--",
    marker="*",
    alpha=al_pha,
    markersize=10,
    color="k",
)
axs_2[0].loglog(
    np.array(num_samp_t_div_4),
    er_coef_1_t_div_4,
    linewidth=line_width,
    linestyle="--",
    marker="*",
    alpha=al_pha,
    markersize=10,
    color="k",
)
axs_2[0].loglog(
    np.sort(kmeans_t_div_1.cluster_centers_[:, 0], axis=0),
    np.sort(kmeans_t_div_1.cluster_centers_[:, 1], axis=0)[::-1],
    linewidth=line_width,
    linestyle="--",
    marker="*",
    markersize=10,
    color="k",
)


axs_2[0].set(xlabel="Number of samples")
axs_2[0].set(ylabel=r"$\texttt{error}\ uu_x$")

axs_2[1].loglog(
    np.array(num_samp_t_rand_exp_1),
    er_coef_2_t_rand_exp_1,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)
axs_2[1].loglog(
    np.array(num_samp_t_rand_exp_2),
    er_coef_2_t_rand_exp_2,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)
axs_2[1].loglog(
    np.array(num_samp_t_rand_exp_3),
    er_coef_2_t_rand_exp_3,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)
axs_2[1].loglog(
    np.array(num_samp_t_rand_exp_4),
    er_coef_2_t_rand_exp_4,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)
axs_2[1].loglog(
    np.array(num_samp_t_rand_exp_5),
    er_coef_2_t_rand_exp_5,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
)
axs_2[1].loglog(
    np.array(num_samp_t_rand_exp_5),
    row_avg_rand_2_12345,
    linewidth=line_width,
    linestyle="-",
    marker="s",
    color=colors[0],
    label=r"\texttt{RS-PINN}",
)

axs_2[1].loglog(
    np.array(num_samp_t_div_1),
    er_coef_2_t_div_1,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="--",
    marker="*",
    markersize=10,
    color="k",
)
axs_2[1].loglog(
    np.array(num_samp_t_div_2),
    er_coef_2_t_div_2,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="--",
    marker="*",
    markersize=10,
    color="k",
)
axs_2[1].loglog(
    np.array(num_samp_t_div_3),
    er_coef_2_t_div_3,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="--",
    marker="*",
    markersize=10,
    color="k",
)
axs_2[1].loglog(
    np.array(num_samp_t_div_4),
    er_coef_2_t_div_4,
    alpha=al_pha,
    linewidth=line_width,
    linestyle="--",
    marker="*",
    markersize=10,
    color="k",
)
axs_2[1].loglog(
    np.sort(kmeans_t_div_2.cluster_centers_[:, 0], axis=0),
    np.sort(kmeans_t_div_2.cluster_centers_[:, 1], axis=0)[::-1],
    linewidth=line_width,
    linestyle="--",
    marker="*",
    markersize=10,
    color="k",
    label=r"\texttt{GS-PINN}",
)


axs_2[1].set(xlabel="Number of samples")
axs_2[1].set(ylabel=r"$\texttt{error}\ u_{xxx}$")

#################################################

from matplotlib.ticker import FuncFormatter


# Define a custom formatting function for y-axis tick labels
def sci_fmt_1(x, pos):
    return "${:.3f}$".format(x)


def sci_fmt_2(x, pos):
    return "${:.3f}$".format(x)


x_ticks = [100, 1000, 10000, 30000]

for j in range(2):
    if j == 0:
        # axs_2[j].set_ylim([-0.035, 0.015])
        axs_2[j].set_xticks(x_ticks)
        # axs_2[j].set_yticks([-0.03, -0.01, 0.01])
        # axs_2[j].yaxis.set_major_formatter(FuncFormatter(sci_fmt_1))
        axs_2[j].grid(True)

    if j == 1:
        # axs_2[j].set_ylim([-6, 1])
        # axs_2[j].set_yticks(y_ticks)
        axs_2[j].set_xticks(x_ticks)
        # axs_2[j].set(xlabel="Iteration")
        # axs_2[j].yaxis.set_major_formatter(FuncFormatter(sci_fmt_2))
        axs_2[j].grid(True)


##################################################

lines, labels = [], []
for ax in fig_2.axes:
    line, label = ax.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)


fig_2.legend(
    lines,
    labels,
    loc="upper center",  # Change the location to upper center
    ncol=2,
    bbox_transform=fig_2.transFigure,
    fontsize=20,
    frameon=True,
    bbox_to_anchor=(0.5, 1.08),
)


plt.tight_layout()


plt.savefig(
    result_path
    + "/"
    + "relative_abs_error_"
    + "_sensitivity_analysis_experiemnts"
    + ".png",
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    result_path
    + "/"
    + "relative_abs_error_"
    + "_sensitivity_analysis_experiments"
    + ".pdf",
    dpi=600,
    bbox_inches="tight",
)
