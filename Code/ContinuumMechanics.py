
# -*- coding: utf-8 -*-
"""
@author: Kian Abdolazizi
Institute for Conitnuum and Material Mechanics, Hamburg University of Technology, Germany

Feel free to cantact if you have questions or want to colaborate: kian.abdolazizi@tuhh.de 

"""

# Standard import
import tensorflow as tf
import numpy as np

# Precision
tf.keras.backend.set_floatx('float64')
tf_float = 'float64'


##########################################################################################
##########################################################################################
###### BASIC CONTINUUM MECHANICS FUNCTIONS ###############################################
##########################################################################################
##########################################################################################


### Pre-defined fiber directions and weights
class dirModel(tf.keras.models.Model):
    """
    Implements a model representing the in-plane symmetrically distributed preferred
    fiber directions based on a trainable fiber angle 'theta' withouth relying on a feature vector
    """
    
    def __init__(self, numTens, numDir, **kwargs):
        """
        Initializes the attributes of the fiber direction model

        Parameters
        ----------
        numTens : int
            the number of generalized strucutral tensors. Note that all generalized
            strucutral tensors share the same preferred material directions. However,
            their weighting of the corresponding strucutral tensors may differ
            between the strucutral models.
        numDir : int
            the number of preferred material directions. Has to be an even number.

        Raises
        ------
        ValueError
            Raises an error if numDir is odd, since it is assumed that two fiber
            families are always symmetrically distributed including the same angle 'theta'.

        Returns
        -------
        None.

        """
        super(dirModel, self).__init__(**kwargs)
        self.numTens = numTens
        self.numDir = numDir
        if numDir % 2 != 0:
            raise ValueError('Only an even number of preferred material directions is allowed, since they are assumed to occure as symmetric pairs!')
        theta_init = tf.keras.initializers.Constant(value=np.pi/4.)
        self.theta = self.add_weight(shape=(numDir//2),
                                     initializer=theta_init,
                                     trainable=True,
                                     name='theta')
               
    def call(self, inputs):
        """
        computes the preferred material directions

        Parameters
        ----------
        inputs : tf.Tensor
            input data, for example the deformation gradient.
            Only needed to derive the batch size and number of time steps

        Returns
        -------
        dirs : tf.Tensor
            the preferred material directions

        """
        # fiber orientations
        # theta = tf.nn.sigmoid(self.theta)*np.pi/2.0 # works not so good better let take on arbitrary values
        sin = tf.sin(self.theta)
        cos = tf.cos(self.theta)
        zero = tf.zeros_like(self.theta, dtype=tf_float)

        # symmetric pairs of fibers
        l_1 = tf.stack([cos,  sin, zero], axis=-1)
        l_2 = tf.stack([cos, -sin, zero], axis=-1)
        
        # interleave the single fibers such that rows 2*i and 2*i+1 belong to the symmetric fiber pair i
        dirs = tf.reshape(tf.stack([l_1, l_2], axis=1), [-1, tf.shape(l_1)[1]], name='concat_dirs') # (numDir, 3)
       
        # expand dims: add batch and time step dimension and tile the fiber directions
        batchSize = tf.shape(inputs)[0]
        nSteps = tf.shape(inputs)[1]
        dirs = tf.expand_dims(tf.expand_dims(dirs, 0), 0)
        shaper = batchSize*tf.constant([1,0,0,0,]) + nSteps*tf.constant([0,1,0,0]) + tf.constant([0,0,1,1])
        dirs = tf.tile(dirs, shaper, name='dirs')
        
        return dirs # (?, nSteps, numDirs, 3)
    
    
    def get_config(self):
        # Implement get_config to enable serialization.
        config = super(dirModel, self).get_config()
        config.update(
            {
                'numTens': self.numTens,
                'numDir': self.numDir
            }
        )
        return config

#
###
#

class weightModel(tf.keras.models.Model):
    """
    Implements the trainable weights of the generalized structural tensor
    withouth relying on the feature vector. 
    """
    
    def __init__(self, numTens, numDir, **kwargs):
        super(weightModel, self).__init__(**kwargs)
        self.numTens = numTens
        self.numDir = numDir
        w_init = tf.keras.initializers.Constant(value=0.0)
        self.w = self.add_weight(shape=(numTens,numDir+1),
                                 initializer=w_init,
                                 trainable=True,
                                 name='w')
        
    def call(self, inputs):
        
        # w = tf.math.sigmoid(self.w)
        w = tf.nn.softmax(self.w)
        
        # zero = tf.constant(0.0, dtype=tf_float)
        # one = tf.constant(1.0, dtype=tf_float)
        # # isotropic part
        # w_iso = tf.stack([one, zero, zero])
        # # anisotropic part, same weights for both fibers
        # w_ani1 = tf.stack([one-w, w, zero])
        # w_ani2 = tf.stack([one-w, zero, w])
        
        batchSize = tf.shape(inputs)[0]
        nSteps = tf.shape(inputs)[1]
        # weights = tf.stack([w_iso, w_ani1, w_ani2], name='stack_weights')
        # weights = tf.stack([w_iso, w_ani1, w_ani2], name='stack_weights')
        shaper = batchSize*tf.constant([1,0,0,0,]) + nSteps*tf.constant([0,1,0,0]) + tf.constant([0,0,1,1])
        w = tf.expand_dims(tf.expand_dims(w, 0), 0)
        w = tf.tile(w, shaper, name='weights')

        return w
    
    
    def get_config(self):
        # Implement get_config to enable serialization.
        config = super(weightModel, self).get_config()
        config.update(
            {
                'numTens': self.numTens,
                'numDir': self.numDir
            }
        )
        return config

#
###
#

def ten2_H(L, w, nSteps, numDir, numTens): # Generalized structural tensors: H_r = \sum_i w_ri * L_i, (?,nSteps,numTens,3,3)
    """
    Computes the generalized strucutral tensors from classical strucutral tensors and the
    corresponding scalar weights
    
    Parameters
    ----------
    L : tf.Tensor
        Classical structural tensors.
    w : tf.Tensor
        Scalar weights of the generalized strucutral tensors.
    nSteps : int
        Number of time steps.
    numDir : int
        Number of preferred directions to use (0 for isotropy, more than 0 for anisotropy).
    numTens : int
        Number of generalized structural tensors to use (at least 1).
        
    Returns
    -------
    H : tf.Tensor
        Generalized structural tensors.

    """
    batchSize = tf.shape(w)[0]

    # Create L_0 and add it to L
    shaper = batchSize*tf.constant([1,0,0,0,0]) + tf.constant([0,0,1,1,1]) + nSteps*tf.constant([0,1,0,0,0])
    L_0 = 1.0/3.0 * tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.eye(3, dtype=tf_float),0),0),0), shaper)
    
    L = tf.cond(numDir > 0, lambda: tf.concat([L_0, L], axis=2, name='concat_L0_L'), lambda: L_0)  
    
    # if numDir > 0:
    #     L = tf.concat([L_0, L], axis=2) # (?,nSteps, numDir+1,3,3)
    # else:
    #     L = L_0

    # Expand L (to get one for each numTens)
    shaper = numTens*tf.constant([0,0,1,0,0,0]) + tf.constant([1,1,0,1,1,1])
    L = tf.tile(tf.expand_dims(L, 2), shaper) # (?,nSteps,numTens,numDir+1,3,3)

    # Expand w
    shaper = tf.constant([1,1,1,1,3])
    w = tf.tile(tf.expand_dims(w, 4), shaper) # (?,nSteps,numTens,numDir+1,3)
    shaper = tf.constant([1,1,1,1,1,3])
    w = tf.tile(tf.expand_dims(w, 5), shaper) # (?,nSteps,numTens,numDir+1,3,3)

    # Multiply L with weights
    L_weighted = tf.math.multiply(L, w) # (?,nSteps,numTens,numDir+1,3,3)

    # Sum them up for the corresponding H
    H = tf.math.reduce_sum(L_weighted, axis=3) # (?,nSteps,numTens,3,3)

    return H

#
###
#

def invariants_I(C, H, numTens): # Generalized invariants I: I_r = trace(C*H_r) [?,nSteps,numTens]
    """
    Computes the generalized invariant I corresponding to the individual strucutral tensors H.
    Can be used for the incompressible as well as nearly incompressible case since the incompressiblity
    constraint det(C)=1 does not have to be considered.
    
    Parameters
    ----------
    C : tf.Tensor
        Right Cauchy-Green deformation tensor.
    H : tf.Tensor
        Scalar weights of the generalized strucutral tensors.
    numTens : int
        Number of generalized structural tensors to use (at least 1).
        
    Returns
    -------
    I : tf.Tensor
        Generalized invariant I corresponding to the individual generalized structural tensors.

    """
    shaper = tf.constant([1,1,0,1,1]) + numTens*tf.constant([0,0,1,0,0])
    C_tile = tf.tile(tf.expand_dims(C, 2), shaper)
    I = tf.linalg.trace(tf.matmul(C_tile,H))
    
    return I

#
###
#
    
# use the incompressible invariants
def invariants_J_incomp(C_bar, H, numTens): # Generalized polyconvex isochoric invariants J: J_r = trace( C_bar^{-T}*H_r) [?,nSteps,numTens] 
    """
    Computes the incompresssible generalized invariant J corresponding to the individual strucutral tensors H.
    The kinematic constraint det(C)=1 is explicitly taken into account in the formulaiton of the invarians.
    
    Parameters
    ----------
    C : tf.Tensor
        Right Cauchy-Green deformation tensor.
    H : tf.Tensor
        Scalar weights of the generalized strucutral tensors.
    numTens : int
        Number of generalized structural tensors to use (at least 1).
        
    Returns
    -------
    J : tf.Tensor
        Incompressible generalized invariants J corresponding to the individual generalized structural tensors.

    """

    shaper = tf.constant([1,1,0,1,1]) + numTens*tf.constant([0,0,1,0,0])
    C_bar_tile = tf.tile(tf.expand_dims(C_bar, 2), shaper)
    invTransC_bar = tf.linalg.inv(tf.transpose(C_bar_tile, perm=[0, 1, 2, 4, 3]))
    matmul = tf.matmul(invTransC_bar, H)
    J = tf.linalg.trace(matmul)
    
    return J

#
###
#

def invariants_J_comp(C, H, numTens): # Generalized invariants J: J_r = trace(cofactor(C)*H_r) [?,nSteps,numTens]
    """
    Computes the compresssible generalized invariant J corresponding to the individual strucutral tensors H.
    
    Parameters
    ----------
    C : tf.Tensor
        Right Cauchy-Green deformation tensor.
    H : tf.Tensor
        Scalar weights of the generalized strucutral tensors.
    numTens : int
        Number of generalized structural tensors to use (at least 1).
        
    Returns
    -------
    J : tf.Tensor
        Compressible generalized invariants J corresponding to the individual generalized structural tensors.

    """       
    
    shaper = tf.constant([1,1,0,1,1]) + numTens*tf.constant([0,0,1,0,0])
    C_tile = tf.tile(tf.expand_dims(C, 2), shaper)
    
    detC_tile = tf.linalg.det(C_tile)
    shaper = tf.constant([1,1,1,3])
    detC_tile = tf.tile(tf.expand_dims(detC_tile, 3), shaper)
    shaper = tf.constant([1,1,1,1,3])
    detC_tile = tf.tile(tf.expand_dims(detC_tile, 4), shaper)
    
    invTransC = tf.linalg.inv(tf.transpose(C_tile, perm=[0, 1, 2, 4, 3]))
    
    mul = tf.math.multiply(detC_tile, invTransC)
    matmul = tf.matmul(mul, H)
    J = tf.linalg.trace(matmul)
    
    return J

#
###
#

def ten2_L(dir): # Structural tensor L_i = l_i (x) l_i , shape = (?, nSteps, numDir, 3, 3)
    """
    Computes the classical strucutral tensors L = l \dyadic l.
    
    Parameters
    ----------
    dir : tf.Tensor
        Preferred material directions.
        
    Returns
    -------
    L : tf.Tensor
        Generalized strucutral tensors.

    """ 
    
    dir = tf.expand_dims(dir, -1) # (?, nSteps, numDir, 3, 1)
    dir_t = tf.transpose(dir, perm=[0, 1, 2, 4, 3]) # (?, nSteps, numDir, 1, 3)
    L = tf.linalg.matmul(dir, dir_t) # (?, nSteps, numDir, 3, 3)
    
    return L

#
###
#

def invariant_I3(C): # Third invariant of a tensor C: I3 = det(C) [?,nSteps,1]
    """
    Compute third invariant (determinant) of a tensor.
    
    Parameters
    ----------
    C : tf.Tensor
        Arbitrary 2nd-order tensor.
        
    Returns
    -------
    det_C : tf.Tensor
        The determinant of C.

    """     
    det_C = tf.expand_dims(tf.linalg.det(C), 2)
    return det_C

#
###
#

def ten2_C(F): # Right Cauchy-Green tensor: C = F^T * F [?,nSteps,3,3]
    """
    Compute the right Cauchy-Green deformation tensor from the deformation gradient.
    
    Parameters
    ----------
    dir : tf.Tensor
        Preferred material directions.
        
    Returns
    -------
    L : tf.Tensor
        Generalized strucutral tensors.

    """ 
    return tf.linalg.matmul(F,F,transpose_a=True)

#
###
#

def ten2_C_bar(C):
    """
    Compute the isochoric right Cauchy-Green deformation tensor.
    
    Parameters
    ----------
    C : tf.Tensor
        Right Cauchy-Green deformation tensor.
        
    Returns
    -------
    C_bar : tf.Tensor
        Isochoric right Cauchy-Green deformation tensors.

    """ 
    det_C = tf.linalg.det(C)
    scale = tf.math.pow(det_C, -1./3.)
    
    shaper = tf.constant([1,1,3])
    scale = tf.tile(tf.expand_dims(scale, 2), shaper)
    shaper = tf.constant([1,1,1,3])
    scale = tf.tile(tf.expand_dims(scale, 3), shaper)
    
    C_bar = tf.math.multiply(scale, C)
    
    return C_bar

#
###
#   

def ten2_C_dot(F, F_dot): # material time derivative of the right Cauchy-Green tensor [?,nSteps,3,3]
    """
    Compute the material time derivative of the right Cauchy-Green deformation gradient.
    
    Parameters
    ----------
    F : tf.Tensor
        deformation gradient.
    F_dot : tf.Tensor
        material time derivative of the deformation gradient.
        
    Returns
    -------
    C_dot : tf.Tensor
        Material time derivative of the right Cauchy-Green deformation tensor.

    """ 
    C_dot = tf.linalg.matmul(F_dot,F,transpose_a=True) + tf.linalg.matmul(F,F_dot,transpose_a=True)
    return C_dot

#
###
#

def ten2_F_ref(F): # Deformation gradient in reference configuration [?,nSteps,3,3]
    """
    Compute the deformation gradient in the reference configuration (identity matrix).
    
    Parameters
    ----------
    F : tf.Tensor
        deformation gradient.
   
    Returns
    -------
    F_ref : tf.Tensor
        Referential deformation gradient.

    """ 
    # In Order for the other formulae to work we need the correct dimension required to produce enough eye matrices/tensors 
    batchSize = tf.shape(F)[0]
    nSteps = tf.shape(F)[1]
    shaper = batchSize*tf.constant([1,0,0,0]) + nSteps*tf.constant([0,1,0,0]) + tf.constant([0,0,1,1])

    F_ref = tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(3, dtype=tf_float),0),0), shaper)
    
    return F_ref

#
###
#

def grad(Psi, C):
    """
    Simply computes the gradient of Psi with respect to C; Psi should be a scalar, C a second order tensor;
    Otherwise the elements of Psi a summed up before gradient computation

    Parameters
    ----------
    Psi : tf.Tensor
        strain energy function (or other scalar).
    C : tf.Tensor
        right Cauchy-Green tensor .

    Returns
    -------
    None.

    """    
    dPsidC = tf.gradients(Psi, C, unconnected_gradients='zero')[0]
    return tf.math.scalar_mul(2.0, dPsidC)

#
###
#

def ten2_S_incomp(S_dev, C):
    """
    Compute incompressible 2. Piola-Kirchhoff stress tensor.
    
    Parameters
    ----------
    S_dev : tf.Tensor
        Deviatoric 2. Piola-Kirchhoff stress tensor.
    C : tf.Tensor
        right Cauchy-Green deformation tensor.        
   
    Returns
    -------
    F_ref : tf.Tensor
        Referential deformation gradient.

    """ 
    CInv = tf.linalg.inv(C)

    S_33 = tf.expand_dims(S_dev[:,:,2,2],2)
    CInv_33 = tf.expand_dims(CInv[:,:,2,2],2)
    p = tf.expand_dims(tf.divide(S_33, CInv_33),2) # hydrostatic pressure
    shaper = tf.constant([1,1,3,3])
    p = tf.tile(p, shaper)
    
    S_iso = tf.math.multiply(p, CInv)
    
    return tf.subtract(S_dev, S_iso)

#
###
#

class ScaleLayer(tf.keras.layers.Layer):
    """
    Scales the relaxation times to powers of 10 within the predefined range [10^[T_min}, 10{T_max}]
    
    """
    
    def __init__(self, T_min=-4, T_max=4, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.T_min = T_min
        self.T_max = T_max
        
    def __call__(self, x):
        nPronyParams = x.shape[2]
        scale = tf.experimental.numpy.logspace(self.T_min, self.T_max, 
                                               num=nPronyParams, endpoint=True,
                                               base=10.0, axis=0) # dtype=tf_float,
        scale = tf.expand_dims(scale, 0)
        scaled = tf.math.multiply(x, scale)
        return scaled 

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(ScaleLayer, self).get_config()
        config.update(
            {
                "T_min": self.T_min,
                "T_max": self.T_max,
            }
        )
        return config

#
###
#

# @tf.keras.saving.register_keras_serializable()
class GradientLayer(tf.keras.layers.Layer):
    """
    Implements the computation of second Piola-Kirchhoff stress-like quantities
    """
    def __init__(self, scale=False, **kwargs):
        super(GradientLayer, self).__init__(**kwargs)
        self.scale = scale
        
    def call(self, y, x):
        dydx = tf.gradients(y, x, unconnected_gradients='zero')[0]
        if self.scale == True:
            dydx = tf.math.scalar_mul(2.0, dydx)
        return dydx
        
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(GradientLayer, self).get_config()
        config.update(
            {
                "scale": self.scale,
            }
        )
        return config

#
###
#

class PsiSigmaLayer(tf.keras.layers.Layer):
    """"
    Implements for each generalized structural tensor the stress normalization strain energy contribution to guarantee a stress-free reference configuration
    """
    def __init__(self, **kwargs):
        super(PsiSigmaLayer, self).__init__(**kwargs)
        
    def call(self, alpha, beta, I, J):
        """
        Computes the strain energy contribution to guarantee a stress-free reference configuration
        
        Parameters
        ----------
        alpha : tf.Tensor
            factor depending on the partial derivatives of Psi evluated in the reference configuration. Either alpha or beta is zero.
        beta : tf.Tensor
            factor depending on the partial derivatives of Psi evluated in the reference configuration. Either alpha or beta is zero.
        I : tf.Tensor
            first generalized invariant of the corresponding generalized strucutral tensor.
        J : tf.Tensor
            secodn generalized invariant of the corresponding generalized strucutral tensor.

        Returns
        -------
        PsiSigma : tf.tensor
            stess normalization contribution.
        """
        
        Psi_1 = tf.math.multiply(alpha, tf.math.subtract(I, 1.0))
        Psi_2 = tf.math.multiply(beta, tf.math.subtract(J, 1.0))
        PsiSigma = tf.math.add(Psi_1, Psi_2)
    
        return PsiSigma
        
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(PsiSigmaLayer, self).get_config()
        return config

#
###
#

# @tf.keras.saving.register_keras_serializable()
class stressUpdateLayer(tf.keras.layers.Layer):
    """
    Implements the recursive stress update formula for computing the viscoelastic stress
    
    """
    
    def __init__(self, nMaxwell, nSteps, **kwargs):
        super(stressUpdateLayer, self).__init__(**kwargs)
        self.nMaxwell = nMaxwell
        self.nSteps = nSteps
   
    def call(self, S_e, t, PronyParams):
        """
        Computes the viscoelastic 2. Piola-Kirchhoff stress tensor.
        
        Parameters
        ----------
        S_e : tf.Tensor
            Instantaneous elastic 2. Piola-Kirchhoff stress tensor
        t : tf.Tensor
            Time
        PronyParams : tf.Tensor
            Prony Parameters; relaxation times and coefficients
                                
        Returns
        -------
        S : tf.Tensor
            Viscoelastic 2. Piola-Kirchhoff stress tensor
        S_infy : tf.Tensor
            Equlibrium 2. Piola-Kirchhoff stress tensor
        Q_sum : tf.Tensor
            Viscous 2. Piola-Kirchhoff stress tensor (sum of all Maxwell elements' viscous stress)
        """
         
        batchSize = tf.shape(S_e)[0] 
       
        tau = (PronyParams[:,:,:self.nMaxwell])
        g = (PronyParams[:,:,self.nMaxwell:])  
        g_sum = tf.math.reduce_sum(g, name='g_sum', axis=-1,keepdims=True)
        g = tf.divide(g, g_sum, name='g') # norm the sum of all g's to 1
        g_infy = g[:,:,0:1]     
        g_i = g[:,:,1:]     
        
        ###
        # Prony Series with variable relaxation times and coefficients
        def recursive_update(x0, x1):
            """
            Compute the stress update for one time step.
            
            """
            
            # x[0] - Q_i, viscoelastic overstresses
            # x[1] - t, time
            # x[2] - S_e, instantaneous elastic stress
            # x[3] - tau, relaxation times
            # x[4] - g, relaxation coefficients
            
            # average values over the time interval interval delta_t = x1[1] - x0[1]
            g_bar =  (x1[4] + x0[4])/2.
            tau_bar = (x1[3] + x0[3])/2.
            # compute overstress update
            delta_t = tf.tile(tf.expand_dims(x1[1]-x0[1],-1), (1, self.nMaxwell))
            
            term_1  = tf.math.exp(-delta_t/tau_bar)
            term_1  = tf.expand_dims(tf.expand_dims(term_1,-1),-1)
            shaper  = tf.constant([1,1,3,3])
            term_1  = tf.tile(term_1, shaper) # (?,nMaxwell,3,3)
            
            term_2 = tf.math.exp(-delta_t/(2.0*tau_bar))*g_bar
            # term_2 = g_bar*tau_bar/ delta_t *(1.0 - tf.math.exp(-delta_t/tau_bar)) # alternative update rule
            term_2 = tf.expand_dims(tf.expand_dims(term_2,-1),-1)
            shaper = tf.constant([1,1,3,3])
            term_2 = tf.tile(term_2, shaper) # (?,nMaxwell,3,3)
            
            delta_S_e = x1[2]-x0[2]
            delta_S_e = tf.expand_dims(delta_S_e,1)
            shaper = tf.constant([1,self.nMaxwell,1,1])
            delta_S_e = tf.tile(delta_S_e, shaper) # (?,nMaxwell,3,3)
            
            Q_i = tf.math.add(tf.math.multiply(term_1, x0[0]), tf.math.multiply(term_2, delta_S_e)) # (?,nMaxwell,3,3)
            
            result = (Q_i, x1[1], x1[2], x1[3], x1[4])
                   
            return result
    
    
        Q_zeros = tf.zeros([batchSize, self.nSteps, self.nMaxwell,3,3], dtype='float64')
        
        # transpose to scan over the 0-th dimension which is should be the time steps and not the batches
        Q_zeros = tf.transpose(Q_zeros, perm=[1,0,2,3,4])
        S_e     = tf.transpose(S_e,     perm=[1,0,2,3])
        tau     = tf.transpose(tau,     perm=[1,0,2])
        g_i     = tf.transpose(g_i,     perm=[1,0,2])
        t       = tf.transpose(t,       perm=[1,0])
        
        initializer = (tf.zeros([batchSize, self.nMaxwell, 3, 3], dtype='float64'), -t[1], tf.zeros([batchSize, 3, 3], dtype='float64'), tau[0], g_i[0])
        
        # recursively compute the stress
        Q = tf.scan(
                recursive_update, # fn
                (Q_zeros, t, S_e, tau, g_i), #  elems
                initializer,
                parallel_iterations=10,
                name='Q'
            )
        
        Q_terms = Q[0]
        Q_sum = tf.math.reduce_sum(Q_terms, axis=2, name='sum_Q_i', keepdims=False) # accumulate the Maxwell element contributions
        
        # transpose back to (?,nSteps,3,3) such that the 0-th dimension is again the batch and not the time steps
        S_e = tf.transpose(S_e, perm=[1,0,2,3]) # instantaneous elasic stress
        Q_sum = tf.transpose(Q_sum, perm=[1,0,2,3]) # viscous overstress
        
        # equilibirum relaxation coefficient
        g_infy = tf.expand_dims(g_infy,-1)
        shaper = tf.constant([1,1,3,3])
        g_infy = tf.tile(g_infy, shaper) # (?,nSteps,3,3)
        
        # the equilibirum stress
        S_infy = tf.math.multiply(g_infy, S_e) 
        
        # the total stress
        S = S_infy + Q_sum 
        
        return S, S_infy, Q_sum
        
    
    def get_config(self):
        # Implement get_config to enable serialization.
        config = super(stressUpdateLayer, self).get_config()
        config.update(
            {
                'nMaxwell': self.nMaxwell,
                'nSteps': self.nSteps
            }
        )
        return config

#
###
#

def ten2_P(S, F): # Second Piola Kirchhoff stress tensor: P = F * S [?,nSteps,3,3]
    """
    Compute the 1. Piola-Kirchhoff stress.
    
    Parameters
    ----------
    S : tf.Tensor
        2. Piola-Kirchhoff stress tensor.
    F : tf.Tensor
        Deformation gradient.
        
    Returns
    -------
    P : tf.Tensor
        1. Piola-Kirchhoff stress tensor.
        
    """
    P = tf.matmul(F, S)
    return P

#
###
#

def ten2_sigma(P, F, J): # Cauchy stress tensor: sigma = J^-1 * P * F^T [?,nSteps,3,3]
    """
    Compute the Cauchy stress by pushing forward the 2. Piola-Kirchhoff stress.
    
    Parameters
    ----------
    S : tf.Tensor
        2. Piola-Kirchhoff stress tensor.
    F : tf.Tensor
        Deformation gradient.
        
    Returns
    -------
    sigma : tf.Tensor
        Cauchy stress tensor.
        
    """ 
    OneOverJ = tf.tile(tf.expand_dims(tf.math.divide(1.0,J),-1), tf.constant([1,1,3,3]))
    sigma = tf.math.multiply(OneOverJ, tf.matmul(P, tf.transpose(F, perm=[0, 1, 3, 2])))

    return sigma

#
###
#

def stressTensors(S, F):
    """
    Compute the 1. Piola-Kirchhoff stress and the Cauchy stress by pushing forward the 2. Piola-Kirchhoff stress.
    
    Parameters
    ----------
    S : tf.Tensor
        2. Piola-Kirchhoff stress tensor.
    F : tf.Tensor
        Deformation gradient.
        
    Returns
    -------
    P : tf.Tensor
        1. Piola-Kirchhoff stress tensor.
    P11 : tf.Tensor
        (1,1)-component of the 1. Piola-Kirchhoff stress tensor.
    sigma : tf.Tensor
        Cauchy stress tensor.
        
    """ 
    P     = tf.keras.layers.Lambda(lambda x: ten2_P(x[0], x[1]), name='P')([S, F])
    P11   = tf.keras.layers.Lambda(lambda P: P[:,:,0,0] , name='P11')(P)
    J     = tf.keras.layers.Lambda(lambda F: tf.expand_dims(tf.linalg.det(F),-1), name='J')(F)
    sigma = tf.keras.layers.Lambda(lambda x: ten2_sigma(x[0], x[1], x[2]), name='sigma')([P, F, J])
    
    return P, P11, sigma
    
    
