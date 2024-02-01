#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:36:47 2023

@author: forootani
"""


import sys
import os

def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
        
        
        #print("*************")
        #print(root_dir)
        
        
    return None

setting_directory(0)


#from base import *
#from constraint import *
#from func_approx import *
#from library import *
#from modules import *
#from plot_config_file import *
#from samples import *
#from utils import *
