#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:03:15 2023

@author: forootani
"""

## https://web.tuat.ac.jp/~yamanaka/pcoms2019/Cahn-Hilliard-2d.html


import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt


nx = 32 # number of computational grids along x direction
ny = nx # number of computational grids along y direction
dx, dy = 2.0e-9, 2.0e-9 # spacing of computational grids [m]
c0 = 0.5 # average composition of B atom [atomic fraction]
R = 8.314 # gas constant
temp = 673 # temperature [K]
nsteps = 6000# total number of time-steps


La = 20000.-9.*temp # Atom intaraction constant [J/mol]
ac = 3.0e-14 # gradient coefficient [Jm2/mol]
Da = 1.0e-04*np.exp(-300000.0/R/temp) # diffusion coefficient of A atom [m2/s]
Db = 2.0e-05*np.exp(-300000.0/R/temp) # diffusion coefficient of B atom [m2/s]
dt = (dx*dx/Da)*0.1 # time increment [s]


fig = plt.figure(figsize=(5,5))
cc = np.linspace(0.01, 0.99, 100);
plt.plot(cc, R*temp*(cc*np.log(cc)+(1-cc)*np.log(1-cc))+La*cc*(1-cc),color='black')
plt.plot(c0, R*temp*(c0*np.log(c0)+(1-c0)*np.log(1-c0))+La*c0*(1-c0),color='r',marker='o',markersize=10)
plt.xlabel('Concentration c [at. frac]')
plt.ylabel('Chemical free energy density')
plt.show()


c = np.zeros((nx,ny)) # order parameter c at time t
c_new = np.zeros((nx,ny)) # order parameter c at time t+dt


def update_orderparameter(c,c_new):
    for j in range(ny):
        for i in range(nx):
            
            ip = i + 1
            im = i - 1
            jp = j + 1
            jm = j - 1
            ipp = i + 2
            imm = i - 2
            jpp = j + 2
            jmm = j - 2

            if ip > nx-1:  # periodic boundary condition
                ip = ip - nx
            if im < 0:
                im = im + nx
            if jp > ny-1:
                jp = jp - ny
            if jm < 0:
                jm = jm + ny
            if ipp > nx-1: 
                ipp = ipp - nx
            if imm < 0:
                imm = imm + nx
            if jpp > ny-1:
                jpp = jpp - ny
            if jmm < 0:
                jmm = jmm + ny
            
            cc = c[i,j] # at (i,j) "centeral point"
            ce = c[ip,j] # at (i+1.j) "eastern point"
            cw = c[im,j] # at (i-1,j) "western point"
            cs = c[i,jm] # at (i,j-1) "southern point"
            cn = c[i,jp] # at (i,j+1) "northern point"
            cse = c[ip,jm] # at (i+1, j-1)
            cne = c[ip,jp]
            csw = c[im,jm]
            cnw = c[im,jp]
            cee = c[ipp,j]  # at (i+2, j+1)
            cww = c[imm,j]
            css = c[i,jmm]
            cnn = c[i,jpp]
            
            mu_chem_c = R*temp*(np.log(cc)-np.log(1.0-cc)) + La*(1.0-2.0*cc) # chemical term of the diffusion potential
            mu_chem_w = R*temp*(np.log(cw)-np.log(1.0-cw)) + La*(1.0-2.0*cw) 
            mu_chem_e = R*temp*(np.log(ce)-np.log(1.0-ce)) + La*(1.0-2.0*ce) 
            mu_chem_n = R*temp*(np.log(cn)-np.log(1.0-cn)) + La*(1.0-2.0*cn)  
            mu_chem_s = R*temp*(np.log(cs)-np.log(1.0-cs)) + La*(1.0-2.0*cs) 

            mu_grad_c = -ac*( (ce -2.0*cc +cw )/dx/dx + (cn  -2.0*cc +cs )/dy/dy ) # gradient term of the diffusion potential
            mu_grad_w = -ac*( (cc -2.0*cw +cww)/dx/dx + (cnw -2.0*cw +csw)/dy/dy )
            mu_grad_e = -ac*( (cee-2.0*ce +cc )/dx/dx + (cne -2.0*ce +cse)/dy/dy )  
            mu_grad_n = -ac*( (cne-2.0*cn +cnw)/dx/dx + (cnn -2.0*cn +cc )/dy/dy ) 
            mu_grad_s = -ac*( (cse-2.0*cs +csw)/dx/dx + (cc  -2.0*cs +css)/dy/dy )              
            
            mu_c = mu_chem_c + mu_grad_c # total diffusion potental
            mu_w = mu_chem_w + mu_grad_w 
            mu_e = mu_chem_e + mu_grad_e 
            mu_n = mu_chem_n + mu_grad_n 
            mu_s = mu_chem_s + mu_grad_s 
    
            nabla_mu = (mu_w -2.0*mu_c + mu_e)/dx/dx + (mu_n -2.0*mu_c + mu_s)/dy/dy    
            dc2dx2 = ((ce-cw)*(mu_e-mu_w))/(4.0*dx*dx)
            dc2dy2 = ((cn-cs)*(mu_n-mu_s))/(4.0*dy*dy) 
            
            DbDa = Db/Da
            mob = (Da/R/temp)*(cc+DbDa*(1.0-cc))*cc*(1.0-cc) 
            dmdc = (Da/R/temp)*((1.0-DbDa)*cc*(1.0-cc)+(cc+DbDa*(1.0-cc))*(1.0-2.0*cc)) 
        
            dcdt = mob*nabla_mu + dmdc*(dc2dx2 + dc2dy2) # right-hand side of Cahn-Hilliard equation
            c_new[i,j] = c[i,j] + dcdt *dt # update order parameter c 
            
            
c = np.zeros((nx,ny)) # zero-clear
c_new = np.zeros((nx,ny)) # zero clear

c = c0 + np.random.rand(nx, ny)*0.01

plt.imshow(c, cmap='bwr')
plt.title('initial concentration')
plt.colorbar()
plt.show()



for nstep in range(1,nsteps+1):
    update_orderparameter(c,c_new)
    c[:,:] = c_new[:,:] # swap c at time t and c at time t+dt
    
    if nstep % 600 == 0:
        print('nstep = ', nstep)
        print('Maximum concentration = ', np.max(c))
        print('Minimum concentration = ', np.min(c))
        plt.imshow(c, cmap='bwr')
        plt.title('concentration of B atom')
        plt.colorbar()
        plt.show() 