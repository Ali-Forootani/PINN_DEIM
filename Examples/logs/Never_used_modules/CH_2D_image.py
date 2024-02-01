#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:38:02 2023

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

#np.random.seed(1000)
#torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


cwd = os.getcwd()
sys.path.append(cwd)

###################################
###################################

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

###################################
###################################

phi_data = np.load("data/CH_Frame1200_X64_Y64_GammaOne_1_GammaTwo_1.npy")
act_num_t = phi_data.shape[0]

output_folder = 'CH_2D_image'  # Specify the folder where the images will be saved
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

for t in range( act_num_t ):
    image = phi_data[t]  # Get the time snapshot as a 2D array

    # Save the image with a unique filename based on the time index
    filename = os.path.join(output_folder, f'image_{t}.png')
    plt.imsave(filename, image, cmap='jet')

####################################
####################################

import os
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(root_dir))  # Get the list of image files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image


data_folder = cwd +'/CH_2D_image'  # Replace with the path to the folder containing your generated images


dataset = CustomDataset(data_folder, transform=ToTensor())

# Set up the data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)


image = dataset.__getitem__(3)

image_np = np.transpose(image[0], (1, 2, 0))

# Display the image
plt.imshow(image[0])
plt.axis('off')
plt.show()


image_np = image[0].numpy()

# Display the image
plt.imshow(image_np, cmap='gray')
plt.axis('off')
plt.show()


dataset.__getitem__(3)
#print(next(iter(data_loader)))

#####################################
#####################################


import os
import torch
from torchvision.transforms import ToTensor, Grayscale, Normalize, Compose
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(root_dir))  # Get the list of image files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path)

        # Convert image to single channel (grayscale)
        image = image.convert("L")

        if self.transform:
            image = self.transform(image)

        return image


data_folder = cwd + '/CH_2D_image'  # Replace with the path to the folder containing your generated images

# Define the transformation pipeline
transform = Compose([
    Grayscale(),              # Convert image to single channel (grayscale)
    ToTensor(),               # Convert image to tensor
    Normalize([0.5], [0.5])   # Normalize the image tensor to range [-1, 1]
])

# Instantiate the CustomDataset
dataset = CustomDataset(data_folder, transform=transform)

# Set up the data loader
batch_size = 2
shuffle = True
num_workers = 4
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)






####################################
####################################
model = fasterrcnn_resnet50_fpn(pretrained=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

model.eval()



# Perform inference
for images in data_loader:
    with torch.no_grad():
        # Forward pass
        predictions = model(images)

    # Process or analyze the predictions as needed

    # You can also save the predictions or perform further post-processing


