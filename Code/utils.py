# -*- coding: utf-8 -*-
"""
@author: Kian Abdolazizi
Institute for Continuum and Material Mechanics, Hamburg University of Technology, Germany

Feel free to contact if you have questions or want to collaborate: kian.abdolazizi@tuhh.de 

"""
import numpy as np
import tensorflow as tf


# This constraint is used to ensure, that combined with the sigmoidal activation
# function in the second last layer the damage function takes values between 0 and 1
# by restricting the weights of the last layer to positive values, that sum up to one
class NonNeg_Norm(tf.keras.constraints.Constraint):
    """
    Constrains weight tensors to sum up to `norm_value` and to be positive.
    
    """

    def __init__(self, norm_value, **kwargs):
        super(NonNeg_Norm, self).__init__(**kwargs)
        self.norm_value = norm_value

    def __call__(self, w):
        w = w * tf.cast(tf.greater_equal(w, 0.), tf.keras.backend.floatx())
        summed_weights = tf.math.reduce_sum(w)
        return w/summed_weights*self.norm_value

    def get_config(self):
        config = super(NonNeg_Norm, self).get_config()       
        config.update({'norm_value': self.norm_value})
        return config

#
###
#

class normalized_mean_squared_error(tf.keras.losses.Loss):
    """
    Compute the normalized mean squared error. Normalized means that each sample is normalized
    by its true value.
    
    """
    
    def call(self, y_true, y_pred):
    
        diff = tf.square(tf.divide(tf.subtract(y_true, y_pred), tf.add(y_true,  1e-6) ))
        loss = tf.reduce_mean(diff)
    
        return loss
#
###
#

class SparsityRegularizer(tf.keras.regularizers.Regularizer):
    """
    L1 regularizer that promotes sparsity of the Maxwell elements by indirectly penalizing
    their relaxation coefficients. The equilibrium relaxation coefficient is not penalized
    since the final vCANN is desired to resemble a generalized Maxwell model and not a series
    of Maxwell models.
    
    """
    
    def __init__(self, l1):
        self.l1 = l1

    def __call__(self, x):
        """
        Applies the L1 regularization to all relaxation coefficients but the first which
        corresponds to the equilibrium spring of the generalized Maxwell model.
        
        """
        p = 1.
        return self.l1 * tf.pow(tf.reduce_sum(tf.pow(tf.abs(x[:,1:,:]),p)), 1/p)
    
    def get_config(self):
        """
        Implement get_config to enable serialization
        """
        config = {"l1": self.l1}

        return config

#
###
#

def R2(y_true, y_pred):
    SS_res = np.sum(np.square( y_true-y_pred ), axis=-1) 
    SS_tot = np.sum(np.square( y_true - np.mean(y_true, axis=-1, keepdims=True)), axis=-1) 
    return (1 - SS_res/(SS_tot + 1e-07))

#
###
#

def setBounds(model):
    """
    Set the lower and upper weight bounds for the deterministic kormos optimizer.
    
    Parameters
    ----------
    model : Keras model
        vCANN model.
        
    Returns
    -------
    bounds : list of tuples
        Each tuple (lb, ub) holds the lower and upper bound of the vectorized weights of the model.
        
    """
    # Determine the bounds of the parameters    
    bounds = []
    
    print('Constrained model weights: \n')
    for m in model.layers:
        if m.count_params() == 0: # skip all except the functional layers
            continue
        if len(m.trainable_weights) != 0 and m.layers == []:
            print('   ' + m.name)
            print('         no kernel constraint')
            b = [(None, None)]*m.count_params()
            bounds = bounds + b
            continue

        print('   ' + m.name)    
        for ll in m.layers:
            
            if len(ll.trainable_weights) == 0: # skip input layers
                continue
            
            print('      ' + ll.name + ':')
            # kernels
            if hasattr(ll, 'kernel'):
                kernel_size = ll.kernel.numpy().size
                kernel_constr = ll.kernel.constraint
            elif hasattr(ll, 'depthwise_kernel'):
                kernel_size = ll.depthwise_kernel.numpy().size
                kernel_constr = ll.depthwise_kernel.constraint
            else:
                kernel_constr = None
            
            if kernel_constr:
                kernel_bounds = [(0.0, None)]*kernel_size
                print('         kernel constrained')
            else:
                kernel_bounds = [(None, None)]*kernel_size
                print('         no kernel constraint')
            bounds = bounds + kernel_bounds
            
        
            # biases
            if ll.use_bias == True: 
                bias_size   = ll.bias.numpy().size
                bias_constr = ll.bias.constraint
                if bias_constr:
                    bias_bounds = [(0.0, None)]*bias_size
                    print('         bias constrained')
                else:
                    bias_bounds = [(None, None)]*bias_size      
                    print('         no bias constraint')
                
                bounds = bounds + bias_bounds
        
    # check if all params have bounds
    nTrainableWeights = len(np.concatenate([w.numpy().flatten() for w in model.trainable_weights]))
    print("Total number of model weights: ", nTrainableWeights)
    print("Total number bounds set      : ", len(bounds))
    
    return bounds
    
#
###
#
    
def checkBounds(model, bounds):
    
    # check if after training the weights are actually within the bounds
    flattened_weights = []
    weight_names = []
    for weight in model.trainable_weights:
        
        w = np.ravel(weight)
        name = weight.name
        weight_names.extend(len(w)*[name])
        flattened_weights.extend(w)
    flattened_weights = np.array(flattened_weights)
    
    for ii, (w, b) in enumerate(zip(flattened_weights,bounds)):
        # lower bounds
        if b[0] == None:
            lb = -np.inf
        else:
            lb = b[0]
        # upper bounds
        if b[1] == None:
            ub = np.inf
        else:
            ub = b[1]         
        if w < lb or w > ub:
            print('Weight bound violation in: {:}'.format(weight_names[ii]))

#
###
#    
    
def boundsAsConstraints(bounds):
    """
    Construct bounds in the form of constraints
    
    """
    cons = []
    for factor in range(len(bounds)):
        lower, upper = bounds[factor]
        if lower == None:
            lower = -np.inf
        if upper == None:
            upper = np.inf
        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)
        
    return cons
