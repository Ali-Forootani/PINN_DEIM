
import numpy as np
import sys
import os
from pathlib import Path
import torch
from scipy import linalg
import torch.nn as nn
import torch.nn.init as init
from Functions import plot_config_file
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time
import numpy as np
from scipy.linalg import svd, qr
import itertools
from scipy.io import loadmat


############################################

cwd = os.getcwd()
#sys.path.append(cwd + '/my_directory')
sys.path.append(cwd)
warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#############################################


"""
data = np.load('data/CH_Frame1200_X64_Y64_GammaOne_1_GammaTwo_1.npy')
t_o = np.arange(0, 1, 0.0008333333333333334)
y_o = np.arange(-1, 1, 0.03125)
x_o = np.arange(-1, 1, 0.03125)
z_o = np.vstack((x_o,y_o))
t_0 = np.arange(0, 1200, 1)
d_o = np.arange(0, 4096)
Exact = np.transpose(np.real(data.reshape(1200,-1)))
"""

#plt.imshow(Exact)
###################################################
###################################################
###################################################

"""
def create_data():
    data = loadmat("data/advection_diffusion.mat")
    usol = np.real(data["Expression1"]).astype("float32")
    print(usol.shape)
    usol = usol.reshape((51, 51, 61, 4))
    x_v = usol[:,:,:,0]
    y_v = usol[:,:,:,1]
    t_v = usol[:,:,:,2]
    u_v = usol[:,:,:,3]
    coords = torch.from_numpy(np.stack((t_v,x_v, y_v), axis=-1))
    data = torch.from_numpy(usol[:, :, :, 3]).unsqueeze(-1)
    # alternative way of providing the coordinates
    # coords = torch.from_numpy(np.transpose((t_v.flatten(), x_v.flatten(), y_v.flatten())))
    # data = torch.from_numpy(usol[:, :, :, 3].reshape(-1,1))
    print("The coodinates have shape {}".format(coords.shape))
    print("The data has shape {}".format(data.shape))
    return coords, data
coords, data = create_data()
"""

"""
data = loadmat("data/advection_diffusion.mat")
usol = np.real(data["Expression1"]).astype("float32")
usol = usol.reshape((51, 51, 61, 4))
x_v= usol[:,:,:,0]/5
y_v = usol[:,:,:,1]/5
t_v = usol[:,:,:,2]/9
u_v = usol[:,:,:,3]

fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
im0 = axes[0].contourf(x_v[:,:,60], y_v[:,:,60], u_v[:,:,60], cmap='coolwarm')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('t = 60')
im1 = axes[1].contourf(x_v[:,:,0], y_v[:,:,0], u_v[:,:,0], cmap='coolwarm')
axes[1].set_xlabel('x')
axes[1].set_title('t = 0')
im2 = axes[2].contourf(x_v[:,:,10], y_v[:,:,10], u_v[:,:,10], cmap='coolwarm')
axes[2].set_xlabel('x')
axes[2].set_title('t= 10')
fig.colorbar(im1, ax=axes.ravel().tolist())
plt.show()
x_o = x_v[:,0,0].reshape(51,-1)
y_o = y_v[0,:,0].reshape(51,-1)
z_o = np.hstack((x_o,y_o))
z_o.shape
"""

###################################################
###################################################
###################################################


class DEIM:
    def __init__(self, X, n_d, t_o, d_o, x_o, y_o, z_o, dec_rate = 0, num_basis=20):
        self.X = X
        self.n_d = n_d
        self.num_basis = num_basis
        self.i_t = []
        self.i_x = []
        self.u_selected = []
        self.t_sampled = []
        self.x_sampled = []
        self.X_sampled = []
        self.T_sampled = []
        self.S_star = []
        self.T_star = []
        self.U_star = []
        self.coords = None
        self.t_o = t_o
        self.d_o = d_o
        self.x_o = x_o
        self.y_o = y_o
        self.z_o = z_o
        self.dec_rate = dec_rate

    def deim(self, X, i):
        U, Sigma, Vt = svd(X, full_matrices=False)
        
        k = (self.num_basis - self.dec_rate * 2)  # Number of basis functions to retain
        
        precision = 1 - np.sum(Sigma[:k])/(np.sum(Sigma) )
        
        while precision >= 1e-7:
            k = k + 1
            precision = 1 - np.sum(Sigma[:k]) / np.sum(Sigma)
        
        print(k)
        
        # Step 3: Compute the SVD-based approximation
        Uk = U[:, :k]
        Sigma_k = np.diag(Sigma[:k])
        #print(Sigma_k)
        Vk_t = Vt[:k, :]
        
        X_k = Uk @ Sigma_k @ Vk_t
        
        left = Uk @ np.sqrt(Sigma_k)
        right = np.sqrt(Sigma_k) @ Vk_t
        
        q_x, r_x, p_x = qr(Uk.T, mode='economic', pivoting=True)
        i_x = p_x[:k]
        
        q_t, r_t, p_t = qr(Vk_t, mode='economic', pivoting=True)
        i_t = p_t[:k]

        return i_t, i_x

    def execute(self):
        n_k = self.X.shape[1]
        n_s = int(n_k / self.n_d)

        for i in range(self.n_d):
            i_tf, i_xf = self.deim(self.X[:, i * n_s: (i+1) * n_s], i)
            i_tf = i_tf + i * n_s
            self.i_t.append([i_tf])
            self.i_x.append([i_xf])
            
            space_o, T_o = np.meshgrid(self.d_o, self.t_o, indexing="ij")
            self.X_sampled.append(space_o)
            self.T_sampled.append(T_o)
            
            t, space = np.meshgrid(i_tf, i_xf, indexing="ij")
            
            self.u_selected.append(self.X[space, t])
          
            self.t_sampled.append(T_o[space,t])
            self.x_sampled.append(space_o[space, t])
            
            
            X_star = np.hstack((t.flatten()[:, None], space.flatten()[:, None]))
            #plt.scatter(X_star[:,0], X_star[:,1])
            plt.scatter(X_star[:,0], X_star[:,1], c=self.X[space, t])
            #plt.ylim([-50,600])
            
            
            ############################
            
            self.S_star.append(self.x_sampled[i].flatten())
            self.T_star.append(self.t_sampled[i].flatten())
            
            
            
            self.U_star.append(self.u_selected[i].flatten())
            
        
        S_s = np.concatenate(self.S_star, axis=0).reshape(-1, 1)
        T_s = np.concatenate(self.T_star, axis=0).reshape(-1, 1)
        U_s = np.concatenate(self.U_star, axis=0).reshape(-1, 1)

        self.coords = np.hstack((S_s,T_s))
        
        return S_s, T_s, U_s
    
    def run(self):
        
        S_s, T_s, U_s = self.execute()
        X_single , Y_single = np.meshgrid(self.x_o, self.y_o, indexing="ij")
        space_star_sing = np.hstack((X_single.flatten()[:, None], Y_single.flatten()[:, None]))
        
        sel_pairs = space_star_sing[S_s.flatten(),:]
        
        
        return sel_pairs, T_s, U_s
    
    
    
    

#deim_instance = DEIM(Exact[:,:], 10, t_o, d_o, x_o, y_o, z_o, num_basis = 5)
#S_s, T_s, U_s_2 = deim_instance.run()


#print(S_s.shape)
#print(T_s.shape)
#print(U_s_2.shape)

#geotime = np.hstack((S_s,T_s))



#X_rec, Y_rec, T_rec = np.meshgrid(S_s[:,0], S_s[:,1], T_s, indexing="ij")
#point_rec = np.hstack((X_rec.flatten()[:, None], Y_rec.flatten()[:, None], T_rec.flatten()[:, None]))


#U = griddata(point_rec, U_s_2.flatten()[:,None], (X_rec, Y_rec, T_rec), method="nearest")
#U = U[:, :, :, 0]

#print(U)

#############################################
#############################################
#############################################



#X_single , Y_single = np.meshgrid(x_o, y_o, indexing="ij")
#X_star_sing = np.hstack((X_single.flatten()[:, None], Y_single.flatten()[:, None]))

#print(X_star_sing[S_s.flatten(),:])

#U_s_2

#U_single = griddata(X_star_sing, data[0,:,:].flatten()[:,None], (X_single, Y_single), method="nearest")

#print(S_s)




#fig, ax = plt.subplots()
#im = ax.scatter(T_s, S_s, c=U_s_2)
#ax.set_xlabel('t')
#ax.set_ylabel('x')
#fig.colorbar(mappable=im)
#plt.show()




#print(U_s_2)


#S_star = deim_instance.S_star
#T_star = deim_instance.T_star
#U_star = deim_instance.U_star
#coords = deim_instance.coords

#print(U_s)
#print(T_s)
#print(S_s)