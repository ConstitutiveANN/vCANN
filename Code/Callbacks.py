# -*- coding: utf-8 -*-
"""
@author: Kian Abdolazizi
Institute for Continuum and Material Mechanics, Hamburg University of Technology, Germany

Feel free to contact if you have questions or want to collaborate: kian.abdolazizi@tuhh.de 

"""
import tensorflow as tf
import numpy as np

class RegularizationCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, lambda_prony, numTens):
        super().__init__()
        self.lambda_prony = lambda_prony
        self.numTens = numTens
    
    def on_epoch_end(self, epoch, logs=None):
        
        if len(self.model.losses) != 0:
                
            if epoch % 20 == 0:
                regularization_weights = []
                for ii in range(self.numTens):
                    model_g = self.model.get_layer('model_g_{:}'.format(ii))
                    regularization_layer = model_g.get_layer('regularization_layer_g_{:}'.format(ii))
                    regularization_weights.append(regularization_layer.weights[0].numpy().squeeze())
                                         
                regularization_weights = np.array(regularization_weights)
                regularization_loss = np.sum(np.abs(regularization_weights))*self.lambda_prony
                
                np.set_printoptions(precision=8)
                np.set_printoptions(linewidth=np.inf)
                
                for ii in range(self.numTens):
                    print('\n')
                    print('Generalized structural tensor {:}'.format(ii+1))
                    print('-------------------------------')
                    print('Sparsity loss:   {:}\n'.format(regularization_loss))
                    print('Sparsity weights:   {:}\n'.format(regularization_weights[ii]))
                print('\n')

#
###
#

class ValLossCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, validation_data=None):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        
        if len(self.model.losses) != 0:
            
            if logs is None:
                logs = {}
            val_loss = self.model.evaluate(self.validation_data, verbose=0)
            regularization_loss = self.model.losses[0]
            logs['val_loss_without_reg'] = val_loss - regularization_loss
        
#
###
#        
        
class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  def __init__(self, epoch_per_save=1, *args, **kwargs):
    self.epochs_per_save = epoch_per_save
    super().__init__(save_freq='epoch', *args, **kwargs)

  def on_epoch_end(self, epoch, logs):
    if epoch % self.epochs_per_save == 0:
      super().on_epoch_end(epoch, logs)

#
###
#
      
# Custom callback to monitor non-zero weights
class NonZeroWeightsMonitor(tf.keras.callbacks.Callback):
    
    def __init__(self, numTens, lambda_prony):
        super().__init__()
        self.lambda_prony = lambda_prony
        self.numTens = numTens
        self.non_zero_counts = [ [] for n in range(numTens) ]  # To store non-zero counts per epoch

    def on_epoch_end(self, epoch, logs=None):
                
        for ii in range(self.numTens):
            model_g = self.model.get_layer('model_g_{:}'.format(ii))
            regularization_layer = model_g.get_layer('regularization_layer_g_{:}'.format(ii))
            weights = regularization_layer.weights[0]
            self.non_zero_counts[ii].append(np.count_nonzero(weights))
            
        # print(f"Epoch {epoch + 1}: Non-zero weights in {self.layer_name} = {non_zero_count}")
