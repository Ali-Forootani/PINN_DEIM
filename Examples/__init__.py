#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:36:47 2023

@author: forootani
"""

#from Functions import *
#from Functions.base import *
#from Functions.constraint import *
#from Functions.func_approx import *
#from Functions.library import *
#from Functions.modules import *
#from Functions.plot_config_file import *
#from Functions.samples import *
#from Functions.utils import *
#from data import *


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
        
        print(root_dir)
        
        
    return None

setting_directory(2)

