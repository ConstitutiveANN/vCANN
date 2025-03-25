# -*- coding: utf-8 -*-
"""
@author: Kian Abdolazizi
Institute for Conitnuum and Material Mechanics, Hamburg University of Technology, Germany

Feel free to cantact if you have questions or want to colaborate: kian.abdolazizi@tuhh.de 


VHB 4910 Dataset from Hossain et al. (2012), "Experimental study and numerical modelling of VHB 4910 polymer".
Create training and validation dataset (tf.dataset)

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
import matplotlib as mpl

#%%

SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rc('text', usetex=True)

mpl.rcParams['text.usetex'] = True 
mpl.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'cm'

prop_cycle = plt.rcParams['axes.prop_cycle']

#%%

def defGrad(lam): # Deformation gradient for incompressible uniaxial tension loading [?,3,3]
    nSamples = lam.shape[0]
    nSteps = lam.shape[1]    
    F = np.zeros([nSamples, nSteps, 3, 3])
    F[:,:,0,0] = lam
    F[:,:,1,1] = 1.0/(np.sqrt(lam))
    F[:,:,2,2] = 1.0/(np.sqrt(lam))
    
    return F


#%%

plt.close('all')

LAM = [1.5, 2.0, 2.5, 3.0]    # stretch amplituedes
LAM_DOT = [0.01, 0.03, 0.05]  # stretch rates

nSteps = 600 # number of time steps 
np.savetxt('.\\n_time_steps.txt', np.array([nSteps]))

for lam in LAM:
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Stretch $\\lambda$ [-]')
    ax.set_ylabel('Nominal stress $P$ [kPa]')
    for lam_dot in LAM_DOT:
        
        if lam == 3.0 and lam_dot == 0.03: # no data available for this stretch amplitude / stretch rate combination
            continue 
        
        data = np.loadtxt('.\\stretch_{:}\\{:}.txt'.format(lam, lam_dot), delimiter=',') # load experimental data points
        
        stretch = data[:,0]
        stress = data[:,1]
        dlam = np.abs(np.diff(stretch))
        
        dt = dlam/lam_dot
        time = np.concatenate([np.array([0]),np.cumsum(dt)])
        
        # interpolate the experimental data for higher time / stretch resolution
        f = interp1d(time, [stretch, stress])
        time_new = np.linspace(time[0], time[-1], nSteps)
        y_new = f(time_new)
        stretch_new, stress_new = y_new[0,:], y_new[1,:]   
        
        label='${:}$'.format(lam_dot)
        #plt.plot(stretch_new, stress_new, label=label)
        plt.scatter(stretch, stress, s=15, label=label)
        
        d = np.c_[stretch_new, time_new, stress_new]
        np.save('.\\stretch_{:}\\{:}.npy'.format(lam, lam_dot), d) # save the interpolated data
    
    ax.legend(title='Stretch rate $\dot{\lambda}$ [s$^{-1}$]')
    plt.savefig('.\\lam_{:}.pdf'.format(lam), format='pdf')
   
#%%

colors = ['tab:blue', 'tab:green', 'tab:orange']

fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel('Stretch $\\lambda$ [-]')
ax.set_ylabel('Nominal stress $P$ [kPa]')

for ii, lam_dot in enumerate(LAM_DOT):

    c=colors[ii]
    for lam in LAM:
        
        if lam == 3.0 and lam_dot == 0.03:
            continue
        
        data = np.loadtxt('.\\stretch_{:}\\{:}.txt'.format(lam, lam_dot), delimiter=',')
        
        stretch = data[:,0]
        stress = data[:,1]
        
        data = np.load('.\\stretch_{:}\\{:}.npy'.format(lam, lam_dot))   
        stretch_new, time_new, stress_new = data[:,0], data[:,1], data[:,2]
        
        if lam == 1.5:
            label='${:}$'.format(lam_dot)
        else:
            label=''
        #plt.plot(stretch_new, stress_new, color=c,label=label)
        plt.scatter(stretch, stress, s=15, color=c, label=label)
    
ax.legend(title='Stretch rate $\dot{\lambda}$ [s$^{-1}$]')
plt.savefig('.\\VHB_4910_data.pdf', format='pdf')


#%% Training data

LAM = [3.0]
LAM_DOT = [0.01, 0.03, 0.05,]

data_tr = []
for lam in LAM:
    for lam_dot in LAM_DOT:
        
        if lam == 3.0 and lam_dot == 0.03:
            continue
    
        d = np.load('.\\stretch_{:}\\{:}.npy'.format(lam, lam_dot))
        data_tr.append(d)    

data_tr = np.vstack(data_tr)

# make dataset with deformation gradient (3D)
lam_reshape = data_tr[:,0].reshape(-1,nSteps) # (?, nSteps)
t_reshape = data_tr[:,1].reshape(-1,nSteps) # (?, nSteps)
extra_dummy = np.ones_like(t_reshape) # dummy feature

P11_reshape = data_tr[:,2].reshape(-1,nSteps) # (?, nSteps)
P = np.zeros([P11_reshape.shape[0],nSteps,3,3])
P[:,:,0,0] = P11_reshape

F_tr = defGrad(lam_reshape) # (?, nSteps, 3, 3)
inps = (F_tr, t_reshape) #, extra_dummy)
outs = (P)

batchSize = len(data_tr)//nSteps
ds_train = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batchSize)
tf.data.experimental.save(ds_train, '.\ds_train_defGrad', compression='GZIP')


#%% Validation data   

LAM = [1.5, 2.0, 2.5]
LAM_DOT = [0.01, 0.03, 0.05]

data_val = []
for lam in LAM:
    for lam_dot in LAM_DOT:
        
        if lam == 3.0 and lam_dot == 0.03:
            continue
        
        d = np.load('.\\stretch_{:}\\{:}.npy'.format(lam, lam_dot))
        data_val.append(d)
        
data_val = np.vstack(data_val)
        
# make dataset with deformation gradient (3D)
lam_reshape = data_val[:,0].reshape(-1,nSteps) # (?, nSteps)
t_reshape = data_val[:,1].reshape(-1,nSteps) # (?, nSteps)
extra_dummy = np.ones_like(t_reshape) # dummy feature

P11_reshape = data_val[:,2].reshape(-1,nSteps) # (?, nSteps)
P = np.zeros([P11_reshape.shape[0],nSteps,3,3])
P[:,:,0,0] = P11_reshape

F_val = defGrad(lam_reshape) # (?, nSteps, 3, 3)
inps = (F_val, t_reshape)#, extra_dummy)
outs = (P)

batchSize = len(data_val)//nSteps
ds_valid = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batchSize)
tf.data.experimental.save(ds_valid, '.\ds_valid_defGrad', compression='GZIP')
