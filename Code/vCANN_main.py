# -*- coding: utf-8 -*-
"""
@author: Kian Abdolazizi
Institute for Continuum and Material Mechanics, Hamburg University of Technology, Germany

Feel free to contact if you have questions or want to collaborate: kian.abdolazizi@tuhh.de 

"""


#%%     Imports

# Standard Imports
import tensorflow as tf

tf.keras.backend.clear_session()
tf.keras.backend.set_floatx('float64')
tf_float = 'float64'
tf.random.set_seed(42)

import numpy as np
import os
import datetime
np_float = np.float64

# Own imports
import ContinuumMechanics as CM
import subANNs
import Outputs
import Plots
import Callbacks
import utils
import fit
import build


#%%
 
dataset = 'VHB4910'

date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
outputFolder = '.\\Results_{:}\\'.format(dataset) + date # output folder were all results are stored
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
    
pathToData = '.\\datasets\\{:}_data\\'.format(dataset) # path to the training and validation data

# If you want to load an already-trained model
load_model = False
modelToLoad = '.\\Results_VHB4910\\20250325-010404'

#%%

###########################################################################
### Material model settings
###########################################################################

incomp = True # incompressible
visco = True  # viscoelastic; if FALSE only elastic

nMaxwell = 5 # The number of Maxwell elements (spring-dashpot elements)
lambda_prony = 0.0 # penalty to enforce sparsity of the Prony Series

# The time constants (powers of 10) used to scale the relaxation times.
#   For example, [T_{r,1}, ... ,T_{r,nMaxwell}]  =  [10^(T_min), ... ,  10^(T_max)] 
T_min = -2
T_max = 1

###########################################################################
### Generalized structural tensors 

numTens = 1 # the number of generalized structural tensors

### there are two alternatives for determining the generalized structural tensors
#
# 1.) The preferred material directions and the weighting factors depend on 
#     some additional data embodied in the feature vector and are learned 
#     from this feature vector. Consequently, preferred material directions 
#     and weighting factors are functions of the feature vector.
#     To use this alternative "numExtraStruc" has to be greater than zero 
#     and obviously the feature vector has to be in the training and validation dataset
#
# 2.) If no data for the feature vector is available, the preferred material directions 
#     and weighting factors are simply trainable parameters optimized during training. 
#     Using this alternative, it is assumed that the fiber directions always
#     appear in pairs, symmetrically distributed about the y-axis of the Cartesian 
#     coordinate system in which also the deformation gradient is defined, c.f.
#     Figure 6 in doi:10.1098/rsif.2005.0073 for illustation. This alternative is
#     automatically activated if "numExtraStruc" is zero. Obviously, only the deformation
#     gradient and the time are then the only inputs and no additional features.

numDir = 0  # the number of preferred material directions per generalized structural tensor; if zero, isotropy is assumed
numExtraStruc = 0 # the number of extra features entering only the sub-networks representing the preferred material directions and weights of the generalized structural tensors

###########################################################################

numExtra = 0 # the number of extra features (i.e. the dimension of the feature vector); enters only the sub-networks of the strain energy function, relaxation times/coefficients

uncoupled = True # if TRUE: to each generalized structural tensor corresponds a separate Generalized Maxwell model, depending only on the invariants associated with this generalized structural tensor. 
                 # if FALSE: the strain energy function, relaxation times/coefficients of each Generalized Maxwell model depends on the invariants of all generalized structural tensors. (not thoroughly tested)

rateDependent = False # if TRUE: relaxation times and coefficients depend on the deformation rate; if FALSE: relaxation times and coefficients depend only on deformation

#%%

# for saving and loading    
custom_objects = {'dirModel': CM.dirModel,
                  'weightModel': CM.weightModel,
                  'PositiveHeNormal': subANNs.PositiveHeNormal,
                  'PsiSigmaLayer': CM.PsiSigmaLayer, 
                  'GradientLayer': CM.GradientLayer,
                  'ScaleLayer': CM.ScaleLayer, 
                  'stressUpdateLayer': CM.stressUpdateLayer,
                  'SparsityRegularizer': utils.SparsityRegularizer
                  }

#%% Hyperparameters

stochastic = False # if TRUE: use stochastic gradient descent; if FALSE: use deterministic L-BFGS-B optimizer
lr = 0.001  # learning rate of the stochastic optimizer

# some standard activation functions
acti0 = 'elu'
acti1 = 'tanh'
acti2 = 'sigmoid'
acti3 = 'softplus'
acti4 = 'linear'
 
#
EPOCHS = 200 # number of training epochs
earlyStopPatience = 20 # number of epochs without improvement in the validation loss after which training is aborted


#### Topology of the individual sub-neural networks (here only a single layer was used. However, arbitrary width and depth is possible)

## sub-networks representing the strain energy function for the elastic part
layer_size_psi = [8,] #  number of neurons per layer; the last layer of shape (1,) is automatically included in the model
activations_psi = [acti3,] # activation function of each layer

## sub-networks representing the preferred material directions
layer_size_dir = [5,] # number of neurons per layer; 
activations_dir = [acti3,] # activation function of each layer

## sub-networks representing the weights of the generalized structural tensors
layer_size_w = [5,] # number of neurons per layer; 
activations_w = [acti3,] # activation function of each layer
  
## sub-networks representing the relaxation times
layer_size_tau = [4,] # number of neurons per layer; the last layer of shape (1,) is automatically included in the model
activations_tau = [acti3,] # activation function of each layer

## sub-networks representing the relaxation coefficients
layer_size_g = [4,] # number of neurons per layer; the last layer of shape (1,) is automatically included in the model
activations_g = [acti3,]#  # activation function of each layer

    
#%% Training and validation data

# Here, we use tf.datasets for training and validation data
# the data has the format data = [F, t, (extraStruc, extra)]
# F : deformation gradients, shape=(nSamples, nSteps, 3, 3)
# t : time, shape=(nSamples, nSteps)
# extraStruc : features from which the preferred material directions and 
#              weights of generalized structural tensors are learned, shape=(nSamples, nSteps, numExtra)
# extra      : features that enter the strain energy function and 
#              relaxation timese/coefficients weights of generalized structural tensors are learned, shape=(nSamples, nSteps, numExtra)    

# the number of time steps (is connected to the training and validation data and cannot be changed independently)
nSteps = int(np.loadtxt(pathToData + 'n_time_steps.txt'))
        
# Load the training and validation data
trainDs = tf.data.experimental.load(pathToData + "ds_train_defGrad", compression='GZIP')
valDs = tf.data.experimental.load(pathToData + "ds_valid_defGrad", compression='GZIP')

# save training and validation data in results folder
tf.data.experimental.save(trainDs, outputFolder + '\\ds_train_defGrad', compression='GZIP')
tf.data.experimental.save(valDs, outputFolder + '\\ds_valid_defGrad', compression='GZIP')
    
#%% Train the model with prescribed hyperparameters

if load_model == True:
    ### load an existing vCANN
    vCANN_fit  = Outputs.loadModel(modelToLoad,'model', custom_objects)
else:  
    ### build and train a new vCANN   
    # vCANN_fit  : used for training; outputs only the total first Piola-Kirchhoff stress tensor
    # vCANN_full : used for debugging and analyzing various intermediate results of the different sub networks and the like
    vCANN_fit, vCANN_full = build.build_model(nSteps,
                                              numTens,
                                              numDir,
                                              nMaxwell,
                                              numExtra,
                                              numExtraStruc,
                                              uncoupled,
                                              rateDependent,
                                              layer_size_psi,
                                              activations_psi,
                                              layer_size_dir,
                                              activations_dir,
                                              layer_size_w,
                                              activations_w,
                                              layer_size_tau,
                                              activations_tau,
                                              layer_size_g, 
                                              activations_g,
                                              [T_min, T_max],
                                              lambda_prony,
                                              incomp,
                                              visco,
                                              tf_float
                                              )
        

    #%% Fit the vCANN using a stochastic or deterministic optimizer
    
    Maxwell_monitor = Callbacks.NonZeroWeightsMonitor(numTens, lambda_prony) # callback for tracking the number of active Maxwell elements in each generalized Maxwell model
    reg_callback = Callbacks.RegularizationCallback(lambda_prony, numTens)   # regularization callback for successively eliminating Maxwell elements during training yielding a sparse model
    
    loss = tf.keras.losses.MeanSquaredError()

    # stochastic optimizer
    if stochastic:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, name='optimizer')
        vCANN_fit.compile(
                optimizer=optimizer,
                loss=loss,
                run_eagerly=False
            )
        
        with tf.device('/device:CPU:0'):
            history, vCANN_fit =  fit.stochastic(vCANN_fit, trainDs, EPOCHS, earlyStopPatience, outputFolder, valDs=valDs, Maxwell_monitor=Maxwell_monitor)
                                       
    # deterministic optimizer L-BFGS-B
    else:
        history, vCANN_fit = fit.deterministic(vCANN_fit, trainDs, EPOCHS, earlyStopPatience, outputFolder, valDs=valDs, Maxwell_monitor=Maxwell_monitor, loss=loss)
            
    #%% Save and plot loss curves
    
    Outputs.saveLoss(history, Maxwell_monitor=Maxwell_monitor, outputFolder=outputFolder)
    Outputs.plotLoss(history, Maxwell_monitor=Maxwell_monitor, title='$\Lambda = {:}$'.format(lambda_prony), outputFolder=outputFolder,scale='log')
        
    #%% Plot computational graphs and save model (summary)
    
    Outputs.showModelSummary(vCANN_fit, outputFolder)
    Outputs.plotModelGraph(vCANN_fit, outputFolder)
    Outputs.saveModel(vCANN_fit, outputFolder)
    
#%% Plot training and validation results

Plots.plot_VHB4910(vCANN_fit, pathToData, outputFolder)
Plots.plot_stress(vCANN_fit, trainDs, valDs, outputFolder)

#%% Plot generalized strucutral tensor(s)

Plots.plot_struc_tensor(vCANN_fit, plotly=False, outputFolder=outputFolder)
    




   
