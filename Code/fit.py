# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 12:57:24 2025

@author: Kian
"""

import tensorflow as tf
import os

import Callbacks
import utils
import kormos

def stochastic(model, trainDs, epochs, earlyStopPatience, folder, valDs=None, Maxwell_monitor=None, reg_callback=None): 

    #tensorboard callback    
    log_dir = os.path.join(folder, "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                                log_dir=log_dir,
                                                histogram_freq=0,
                                                write_graph=True,
                                                write_images=False,
                                                update_freq=50,
                                                profile_batch=0,
                                                embeddings_freq=0,
                                                embeddings_metadata=None,
                                                )

    
    # custom checkpoint callback to enable saving every n-th epoch  
    ckpt_dir = os.path.join(folder, 'ckpt', 'ckpt')
    model_ckpt_callback = Callbacks.MyModelCheckpoint(epoch_per_save=20,
                                                      filepath=ckpt_dir,
                                                      monitor='val_loss',
                                                      verbose=2,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode='auto',
                                                      options=None
                                                      )
        
    # early stopping based on the validation loss without regularization loss
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=earlyStopPatience,
                                                  restore_best_weights=True, # restore weights of best epoch
                                                  mode='auto')
    
    # callback to compute the validation loss without the sparsity enforcing L1 regularization
    val_loss_callback = Callbacks.ValLossCallback(validation_data=valDs)
    
    
    callbacks = [val_loss_callback, model_ckpt_callback, early_stop, tensorboard_callback]

    if reg_callback is not None:    
        # callback to write the regularization weights of the Maxwell elements to the console
        callbacks.append(reg_callback)

    if Maxwell_monitor is not None:
        callbacks.append(Maxwell_monitor)


    history = model.fit(
                  x=trainDs,
                  batch_size=None,
                  epochs=epochs,
                  verbose=2,
                  callbacks=callbacks,
                  validation_data=valDs,
                  shuffle=True,
                  # validation_freq=1,
                  max_queue_size=20,
                  workers=8,
                  use_multiprocessing=True)
    
    return history, model
 

#
###
#

def deterministic(model, trainDs, epochs, earlyStopPatience, folder, valDs=None, method='L-BFGS-B', Maxwell_monitor=None, loss=tf.keras.losses.MeanSquaredError(), batch_size=2**10):
    
    
    # define bounds/constraints
    bounds = utils.setBounds(model)
    constraints = utils.boundsAsConstraints(bounds)
    
    # set up model and optimizer
    kormos_model = kormos.models.BatchOptimizedModel(inputs=model.inputs, outputs=model.outputs)
    optimizer = kormos.optimizers.ScipyBatchOptimizer(method=method, bounds=bounds, batch_size=batch_size)            
    kormos_model.compile(loss=loss, optimizer=optimizer, metrics=['mean_squared_error'])
    options={'ftol': 1e-12, 'gtol': 1e-12} # options passed to the scipy minimizer
    
    # define callbacks
    callbacks = []
    
    ckpt_dir = os.path.join(folder, 'ckpt', 'ckpt')
    model_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir,
                                                             save_freq=10,
                                                             monitor='val_loss',
                                                             verbose=2,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode="auto",
                                                             options=None,
                                                             )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=earlyStopPatience,
                                                  restore_best_weights=True, # restore weights of best epoch
                                                  mode='auto')
    
    callbacks.append(model_ckpt_callback)
    callbacks.append(early_stop)
    
    if Maxwell_monitor is not None:
        callbacks.append(Maxwell_monitor)

    # train the model using a deterministic optimizer
    with tf.device('/device:CPU:0'):
        history = kormos_model.fit(trainDs,
                                   epochs=epochs,
                                   callbacks=callbacks,
                                   batch_size=batch_size,
                                   validation_data=valDs,
                                   options=options
                                   )
     
    # check after training if bouds are violated   
    utils.checkBounds(model, bounds)
    
    return history, model
