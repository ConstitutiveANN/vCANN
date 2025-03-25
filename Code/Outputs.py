# Standard imports
import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import shutil
from contextlib import redirect_stdout
from matplotlib import cm
import json

from utils import R2
from natsort import natsorted


#%%

# SMALL_SIZE = 16
# MEDIUM_SIZE = 18
# BIGGER_SIZE = 20

# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rc('text', usetex=True)

# prop_cycle = plt.rcParams['axes.prop_cycle']

# linestyles = [
#      ('loosely dotted',        (0, (1, 10))),
#      ('dotted',                (0, (1, 1))),
#      ('densely dotted',        (0, (1, 1))),

#      ('loosely dashed',        (0, (5, 10))),
#      ('dashed',                (0, (5, 5))),
#      ('densely dashed',        (0, (5, 1))),

#      ('loosely dashdotted',    (0, (3, 10, 1, 10))),
#      ('dashdotted',            (0, (3, 5, 1, 5))),
#      ('densely dashdotted',    (0, (3, 1, 1, 1))),

#      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

# plt.rcParams['text.usetex'] = True

#%%

#
###

def prepareOutputFolder(baseFolder, problemName):
    """
    Creates all necessary output folders.

    Parameters
    ----------
    baseFolder : string
        Name of the base folder for all exports.
    problemName : string
        Name of the current problem (will be used as sub folder name).
        
    Returns
    -------
    outputFolder : string
        Complete path to problem specific output folder (with filename prefix).
        
    """
    if not os.path.exists(baseFolder):
        os.makedirs(baseFolder)
        
    outputFolder = baseFolder + problemName+'\\'
    
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        
    outputFolder = outputFolder + problemName+'_'
    
    return outputFolder



#####
## Save/plot loss
#####

#
###
def saveLoss(his, Maxwell_monitor=None, outputFolder=None):
    """
    Saves the loss to MATLAB.

    Parameters
    ----------
    his : keras history object
        History object obtained during training.
    outputFolder : string
        Output folder (and filename prefix).

    """
    if outputFolder:
        loss_dir = outputFolder + '\\loss'
        os.makedirs(loss_dir)
        
    np.save(loss_dir+'\\loss.npy', his.history['loss'])
    np.save(loss_dir+'\\val_loss', his.history['val_loss'])
    if Maxwell_monitor:    
        np.save(loss_dir+'\\nMaxwell', Maxwell_monitor.non_zero_counts)



#
###
def plotLoss(his, Maxwell_monitor=None, outputFolder=False, title=None, scale='log'):
    """
    Plots the training and validation loss and the number of Maxwell elements. If more than one
    Generalized Maxwell model is used the number Maxwell elements equates to the total number
    of all Maxwell elements.    
    
    If an outputFolder is specified, saves the figure to this output folder.

    Parameters
    ----------
    his : keras history object
        History object obtained during training.
    nMaxwell_monitor : list of lists
        Each list holds the number Maxwell elements of the corresponding Generalized Maxwell model at successive epochs
    outputFolder : string, optional
        Output folder (and filename prefix).
    title : str
        The title to put above the loss plot
    scale : str
        if 'log', uses a logarithmic scale on the y-axis for the loss, else linear scale

    """
    fig, ax = plt.subplots(1,1,figsize=(9,6))
    lw=1.2
    alpha=1.0
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # ax.set_ylim([2*10**-1, 3*10**2])
    
    if type(his) == str:
        loss = np.load(outputFolder+"\\loss\\loss.npy")
        val_loss = np.load(outputFolder+"\\loss\\val_loss.npy")        
    else:
        loss = his.history['loss']
        val_loss = his.history['val_loss']
    
    ax.plot(loss, color='tab:blue', lw=lw, label='Training loss')
    ax.plot(val_loss, color='tab:orange', lw=lw, label='Validation loss')
    # ax.plot(his.history['val_loss_without_reg'], color='tab:orange', lw=lw, label='Validation')
    
    if Maxwell_monitor:
        ax_1 = ax.twinx()
        ax_1.set_ylabel('\# Maxwell elements')
        
        # compute the total number of Maxwell elements over all Generalized Maxwell models
        if type(Maxwell_monitor) == str:
            nMxawell = np.load(outputFolder+"\\loss\\nMaxwell.npy").reshape(-1) -1
        else:
            numTens = Maxwell_monitor.numTens
            nMxawell = np.sum(Maxwell_monitor.non_zero_counts, axis=0)
            nMxawell = nMxawell - numTens # subtract the equilibrium spring(s)
        
        ax_1.plot(nMxawell, color='tab:green', lw=lw, label='\# Maxwell elements', alpha=alpha)     
        # ax.grid(visible=True, which='major', color='#CCCCCC', linestyle='-')
        n_max = np.max(nMxawell)
        ax_1.set_ylim([0,n_max+1])
        
        ax_1.legend(loc='lower left')

    ax.legend(loc='best')

    if scale == 'log':
        ax.set_yscale('log')
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if isinstance(outputFolder, str):
        plt.savefig(outputFolder+'\\loss_.pdf', format='pdf')

    
#
###
def saveTrainingValidationSplit(train_indices, val_indices, outputFolder):
    """
    Saves the training/validation split into MATLAB.

    Parameters
    ----------
    train_indices : float array
        Original indices that have been put into the training data set.
    val_indices : float array
        Original indices that have been put into the validation data set.
    outputFolder : string
        Output folder (and filename prefix).

    """
    
    scipy.io.savemat(outputFolder+'trainingValidation.mat', mdict={
        'train_indices': train_indices,
        'val_indices': val_indices
        })

#
###
def saveModelConfig(model, outputFolder=False):
    """
    Writes the model/optimizer config to file.

    Parameters
    ----------
    model: Keras model or optimizer 
        Keras model or optimizer 
    outputFolder : string, optional
        Output folder (and filename prefix).

    """
    #config = model.to_json()   
    config = model.get_config()
    
    if isinstance(outputFolder, str):
        with open(outputFolder + '\\' + 'model_config.json', 'w') as f: 
            json.dump(config, f)
#
###            
def saveOptimizerConfig(optimizer, outputFolder=False):
    """
    Writes the model/optimizer config to file.

    Parameters
    ----------
    model: Keras model or optimizer 
        Keras model or optimizer 
    outputFolder : string, optional
        Output folder (and filename prefix).

    """
    config = optimizer.get_config()   
   
    if isinstance(outputFolder, str):
        with open(outputFolder + '\\' + 'optimizer_config.json', 'w') as f: 
            json.dump(str(config), f)
                
#
###
def showModelSummary(model, outputFolder=False):
    """
    Prints the model summary (layers and such) for the whole model and for each used subANN. If an outputFolder is specified, the output is saved.

    Parameters
    ----------
    model : keras model
        Keras model to be summarized.
    outputFolder : string, optional
        Output folder (and filename prefix).

    """

    if isinstance(outputFolder, str):
        summary_folder = os.path.join(outputFolder, "summaries")
        if not os.path.isdir(summary_folder):
            os.makedirs(summary_folder)
        
    # main model summary
    modelName = model.name
    print('\n\n\n')
    model.summary()
    if isinstance(summary_folder, str):
        with open(summary_folder + '\\' + modelName + '_summary.txt', 'w') as f: 
            with redirect_stdout(f):
                model.summary()
    print('\n\n\n')
    
    # submodel summaries
    for l in model.layers:
        if l.count_params() == 0: # skip all except the functional layers
            continue
        modelName = l.name
        print(l.name)
        l.summary()
        if isinstance(summary_folder, str):
            with open(summary_folder + '\\' + modelName + '_summary.txt', 'w') as f: 
                with redirect_stdout(f):
                    l.summary()           
        print('\n\n\n')

    
#
###
def plotModelGraph(model, outputFolder):
    """
    Saves a plot of the model graph.
    
    Attention: This function utilizes the software 'graphviz', which needs to be installed separately in order to work. See the documentation for k.utils.plot_model for details.

    Parameters
    ----------
    model : keras model for strain energy and damage variable
        Keras model to be summarized.
    outputFolder : string
        Output folder (and filename prefix).
    numDir : int, optional
        Number of preferred directions to use (0 for isotropy, more than 0 for anisotropy).

    """
    
    graph_path = os.path.join(outputFolder, 'graphs')
    if not os.path.isdir(graph_path):
        os.makedirs(graph_path)
        
    # main model graph
    modelName = model.name  
    tf.keras.utils.plot_model(model, show_shapes=True , dpi=400, expand_nested=True  , show_layer_names=True, to_file=graph_path + '\\' + modelName + '_graph_expanded.png')
    tf.keras.utils.plot_model(model, show_shapes=False, dpi=400, expand_nested=False , show_layer_names=True, to_file=graph_path + '\\' + modelName + '_graph.png')

    # submodel graphs
    for l in model.layers:
        if l.count_params() == 0: # skip all except the functional layers
            continue
        modelName = l.name
        tf.keras.utils.plot_model(l, show_shapes=True , dpi=400, expand_nested=True , show_layer_names=True, to_file=graph_path + '\\' + modelName + '_graph_expanded.png')
        tf.keras.utils.plot_model(l, show_shapes=False, dpi=400, expand_nested=False, show_layer_names=True, to_file=graph_path + '\\' + modelName + '_graph.png')
        
#
###        
def saveModel(model, outputFolder):
    modelName = model.name
    saveFolder = outputFolder + "\\saved_model\\" + modelName + '.keras'
    # if not os.path.exists(saveFolder):
    #     os.makedirs(saveFolder)
        
    tf.keras.models.save_model(model,
                               saveFolder,
                               overwrite=True,
                               include_optimizer=False,
                               save_format='keras',
                               signatures=None,
                               options=None,
                               save_traces=True
                               )
        

#
###
def loadModel(savedFolder, subModelName, custom_objects=None):
    model = savedFolder + "\\saved_model\\" + subModelName + '.keras'
    if os.path.isfile(model):
        loadedModel = tf.keras.models.load_model(model, custom_objects=custom_objects)
        return loadedModel
    else:
        print(model + ' does not exist. No model loaded.')
        return None
        
