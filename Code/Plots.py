"""
@author: Kian Abdolazizi
Institute for Conitnuum and Material Mechanics, Hamburg University of Technology, Germany

Feel free to cantact if you have questions or want to colaborate: kian.abdolazizi@tuhh.de 

"""



# Standard imports

from matplotlib import cm
from mpl_toolkits.mplot3d.axis3d import Axis
from utils import R2


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.ticker as ticker
import plotly.io as pio

pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

#%% General plotting settings


SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rc('text', usetex=True)

mpl.rcParams['text.usetex'] = True 
mpl.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'cm'

prop_cycle = plt.rcParams['axes.prop_cycle']

linestyles = [(0, (1, 10)),
              (0, (1, 5)),
              (0, (1, 1)),
              (0, (5, 10)),
              (0, (5, 5)),
              (0, (5, 1)),
              (0, (3, 10, 1, 10)),
              (0, (3, 5, 1, 5)),
              (0, (3, 1, 1, 1)),
              (0, (3, 10, 1, 10, 1, 10)),
              (0, (3, 5, 1, 5, 1, 5)),
              (0, (3, 1, 1, 1, 1, 1))]

plt.rcParams['text.usetex'] = True

### Patch to remove margins from 3D plots
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
### Patch end

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

def plot_struc_tensor(model, extra_struc=None, plotly=False, outputFolder=None):
    """
    Plot the generalized strucutral tensor
    
    """

    inputs = model.get_layer('F_input').input
    outputs = model.get_layer('H').output
    model_H = tf.keras.models.Model(inputs, outputs)

    numTens = model_H.get_layer('H').input[-1].numpy()
    nSteps = model.outputs[0].shape.as_list()[1]
    numDir = model_H.get_layer('H').input[-2].numpy()

    nTheta, nPhi = 100, 100

    theta, phi = np.linspace(0, np.pi, nTheta), np.linspace(0, 2*np.pi, nPhi)
    THETA, PHI = np.meshgrid(theta, phi)

    X = np.sin(THETA)*np.cos(PHI)
    Y = np.sin(THETA)*np.sin(PHI)
    Z = np.cos(THETA)

    P = np.array([X,Y,Z])
    P = P.reshape(3,-1)

    if extra_struc != 0:
        F = np.eye(3) # dummy deformation gradient 
        F = np.expand_dims(F,(0,1))
        F = np.tile(F, (1,nSteps,1,1))    
        Hs = model_H(F)        
    else:
        extra_struc = np.expand_dims(extra_struc, axis=(0,1))
        extra_struc = np.tile(extra_struc, (1,nSteps,1))
        Hs = model_H(extra_struc)

    Hs = Hs[0,0]
    Hs = np.split(Hs, numTens, 0)

    x_max = 1.
    y_max = 1.
    z_max = 1.
    abs_H_max = 0.0
    
    for ii in range(numTens):
        H = Hs[ii].squeeze()
        abs_H = np.einsum('im,ik,km->m',P, H, P)
        abs_H = abs_H.reshape(nPhi, nTheta)
        abs_H_max = np.maximum(abs_H_max, np.max(abs_H))
        
        X_ = abs_H*np.sin(THETA)*np.cos(PHI)
        Y_ = abs_H*np.sin(THETA)*np.sin(PHI)
        Z_ = abs_H*np.cos(THETA)

    fig, axes = plt.subplots(1,numTens,figsize=(24,8), subplot_kw=dict(projection='3d'))
    if numTens == 1:
        axes = [axes,]
    
    for ii in range(numTens):
        H = Hs[ii].squeeze()
        abs_H = np.einsum('im,ik,km->m',P, H, P)
        abs_H = abs_H.reshape(nPhi, nTheta)
        
        X_ = abs_H*np.sin(THETA)*np.cos(PHI)
        Y_ = abs_H*np.sin(THETA)*np.sin(PHI)
        Z_ = abs_H*np.cos(THETA)
                
        d = np.abs(abs_H/abs_H_max)

        ax = axes[ii]
        tick_spacing = 1.
        fontsize = 32
        labelpad=20
        
        ax.set_xlabel("x'", labelpad=labelpad, fontsize=fontsize)
        ax.set_ylabel("y'", labelpad=labelpad, fontsize=fontsize)
        ax.set_zlabel("z'", labelpad=labelpad, fontsize=fontsize)
        
        ax.tick_params(axis='both', which='major', labelsize=fontsize)    
        
        ax.set_xlim([-x_max, x_max])
        ax.set_ylim([-y_max, y_max])
        ax.set_zlim([-z_max, z_max])
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        cmap = cm.coolwarm(d)               
        ax.plot_surface(X_, Y_, Z_, facecolors=cmap, rcount=200, ccount=200, linewidth=0, antialiased=True)
        ax.set_aspect('equal')
        
        zdirs = ['x','y', 'z']
        for d in zdirs:
            if d == 'y':
                offset = 1
                # offset = ax.get_ylim()[1]
            elif d == 'x':
                offset = -1
                # offset = ax.get_xlim()[0]
            elif d == 'z':        
                offset = -1   
                # offset = ax.get_zlim()[0]
            ax.contour(X_, Y_, Z_, levels=[0,], zdir=d, offset=offset, colors='tab:grey')

        # ax.set_title('Generalized structural tensor $\\tilde{{\\mathbf{{L}}}}_{:}$'.format(ii+1))
    
    plt.tight_layout()
    if outputFolder:
        fig.savefig(outputFolder+'\\generalized_structural_tensor.pdf', format='pdf',
                    bbox_inches="tight",
                    #pad_inches = 0       # remove white space
                    )    

    if plotly:
        fig.layout['scene'] =  dict(
                xaxis = dict(title_text='x', range=[-x_max, x_max]),
                yaxis = dict(title_text='y', range=[-y_max, y_max]),
                zaxis = dict(title_text='z', range=[-z_max, z_max]),
                aspectmode = 'data'
                )
        
        for ii in range(2,numTens+1):
            fig.layout['scene{:}'.format(ii)] = dict(
                    xaxis = dict(title_text='x', range=[-x_max, x_max]),
                    yaxis = dict(title_text='y', range=[-y_max, y_max]),
                    zaxis = dict(title_text='z', range=[-z_max, z_max]),
                    aspectmode='data'
                    )
    
        fig.update_traces(showscale=False)
        fig.update_layout(height=600, width=2400, title_text="Generalized structural tensors")        
        fig.show()
                
#
###
#  

def plot_stress(model, trainDs, valDs, outputFolder):
    """
    PLot the training and validation dataset in two separate subfigures

    Parameters
    ----------
    model : keras model
        the vCANN.
    trainDs : tf.dataset
        the training dataset.
    valDs : tf.dataset
        the validation dataset.
    outputFolder : str
        path to the output folder.

    Returns
    -------
    None.

    """
    
    figsize = (22,8)
    
    fig_l, axes_l = plt.subplots(1,2, figsize=figsize)
    axes_l = axes_l.flatten()

    fig_t, axes_t = plt.subplots(1,2, figsize=figsize)
    axes_t = axes_t.flatten()
    
    Ds = [trainDs, valDs]
    Title = ['Training', 'Validation']
    mse_avg = []
        
    for ii, (ds, ax_l, ax_t, t) in enumerate(zip(Ds, axes_l, axes_t, Title)): 
        MSE = []
        
        ax_t.set_xlabel(u'Time $t$ [s]')
        ax_t.set_ylabel(u'Nominal stress $P$ [MPa]')
        ax_l.set_xlabel(u'Stretch $\\lambda$ [-]')
        ax_l.set_ylabel(u'Nominal stress $P$ [MPa]')
        
        for jj, (x,y) in enumerate(ds):
            batchSize = x[0].get_shape().as_list()[0]
            stress = model.predict(x)
            
            for kk in range(batchSize):
                F = x[0][kk].numpy()
                lam = F[:,0,0]       
                                
                stress_data = y.numpy()
                MSE.append( ((stress[kk]-stress_data[kk])**2).mean(axis=None) )
                    
                if len(x) == 2:
                    time = x[1][kk].numpy()  
                    ax_t.plot(time, stress[kk][:,0,0], lw=1)
                    n=1
                    ax_t.scatter(time[::n], stress_data[kk][::n,0,0], marker='o', facecolors=None, s=10)
        
                ax_l.plot(lam, stress[kk][:,0,0], lw=1)
                n=1
                ax_l.scatter(lam[::n], stress_data[kk][::n,0,0], marker='o', facecolors=None, s=10)                
        
        mse_avg.append(np.mean(MSE))
    
    subscript = ['tr','val']           
    for axes in [axes_t, axes_l]:
        for ii, ax in enumerate(axes):
            text = 'MSE$_{{{:}}}={:5.4f}$'.format(subscript[ii], mse_avg[ii])
            ax.text(0.7, 0.05, text, color='k',transform=ax.transAxes, 
                            bbox=dict(facecolor='none', edgecolor='none'))
            ax.set_title(Title[ii])

    fig_l.savefig(outputFolder+'\\stretch_stress.pdf', format='pdf', bbox_inches="tight")    
    fig_t.savefig(outputFolder+'\\time_stress.pdf', format='pdf', bbox_inches="tight")    

#
###
#

def plot_VHB4910(model, pathToData, outputFolder):
    """
    Plot the results of for the VHB 4910 dataset by Hossain et al. (2012), "Experimental study and numerical modelling of VHB 4910 polymer"

    Parameters
    ----------
    model : keras model
        the vCANN.
    pathToData : str
        the path to the experimental data.
    outputFolder : str
        the path to the output folder.

    Returns
    -------
    None.

    """

    path = outputFolder + '\\prediction'
    if not os.path.exists(path):
        os.makedirs(path)

    LAM = [3.0, 1.5, 2.0, 2.5]
    LAM_DOT = [0.01, 0.03, 0.05]
    
    fig_p, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    captions = ['(a)','(b)','(c)','(d)']
    colors = ['tab:blue', 'tab:green', 'tab:orange']
        
    for ii, lam in enumerate(LAM):
    
        axes = axes.flatten()
        fig, ax = plt.subplots(figsize=(8.5, 7))     
        
        xlabel = 'Stretch $\\lambda$ [-]'
        ylabel = 'Nominal stress $P$ [kPa]'
        axes[ii].set_xlabel(xlabel)
        axes[ii].set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        MSE = []
        NRMSE = []
    
        for jj, lam_dot in enumerate(LAM_DOT):
    
            if lam == 3.0 and lam_dot == 0.03:
                continue
                        
            # interpolated training data
            data = np.load(pathToData + '\\stretch_{:}\\{:}.npy'.format(lam, lam_dot))   
            stretch, time, stress = data[:,0], data[:,1], data[:,2]
            
            F = defGrad(stretch.reshape(1,-1))
            time = np.expand_dims(time,0)
            stress = np.expand_dims(stress, 0)
            batch_size = len(F)

            extra = np.ones_like(time) # dummy extra variable for feature vector
            
            inps = (F, time ) # , extra)
            outs = (stress)
            ds = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
            stress_pred = model.predict(ds)
            stress_pred = stress_pred[0,:,0,0]
            
            mse = ((stress_pred - stress)**2).mean(axis=None)
            nrmse = np.sqrt(mse)/np.abs(stress).mean(axis=None)
            MSE.append(mse)
            NRMSE.append(nrmse)
           
            axes[ii].plot(stretch, stress_pred, color=colors[jj], lw=2.)
            ax.plot(stretch, stress_pred, color=colors[jj], lw=2.5)
            
            # save vCANN prediction to file
            pred_data = np.vstack([stretch, time.reshape(-1), stress_pred])
            file = path + '\\lam_{}_lam_dot_{}.npy'.format(lam, lam_dot)
            np.save(file, pred_data)
            
            # experimental data
            data = np.loadtxt(pathToData + '\\stretch_{:}\\{:}.txt'.format(lam, lam_dot), delimiter=',')   
            stretch, stress = data[:,0], data[:,1]
            
            label='${:}$'.format(lam_dot)
            axes[ii].scatter(stretch, stress, s=25, facecolors='none', edgecolor=colors[jj], label=label)
            axes[ii].grid(True)
            ax.scatter(stretch, stress, s=30, facecolors='none', edgecolor=colors[jj], label=label)
            ax.grid(True)
        
        axes[ii].legend(title='$\\dot{\\lambda}$ [s$^{-1}$]')
        axes[ii].text(.5, -.21, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
        
        ax.legend(title='$\\dot{\\lambda}$ [s$^{-1}$]')
                
        if ii == 0:
            t = 'Training'
            axes[ii].set_title(t)
            ax.set_title(t)
        else:
            t = 'Validation'
            axes[ii].set_title(t)
            ax.set_title(t)

        fig.tight_layout()        
        fig.savefig(outputFolder + '\\training_validation_{:}.pdf'.format(ii+1), format='pdf', bbox_inches='tight')

    fig_p.subplots_adjust(hspace=0.4)
    fig_p.savefig(outputFolder + '\\training_validation_panel.pdf', format='pdf', bbox_inches='tight')

