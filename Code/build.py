# -*- coding: utf-8 -*-
"""
@author: Kian Abdolazizi
Institute for Continuum and Material Mechanics, Hamburg University of Technology, Germany

Feel free to contact if you have questions or want to collaborate: kian.abdolazizi@tuhh.de 

"""

import tensorflow as tf

import ContinuumMechanics as CM
import subANNs
import utils


def build_model(nSteps,
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
                tau = [1e-3, 1e3],
                lambda_prony=0.001,
                incomp=True,
                visco=False,
                tf_float='float64'):

    
    ### Deformation measures
    
    # current configuration
    F = tf.keras.layers.Input(shape=(nSteps,3,3,), name='F_input') # INPUT
    F_dot = tf.keras.layers.Input(shape=(nSteps,3,3,), name='F_dot_input') # INPUT
    batchSize = tf.shape(F)[0]

    # right Cauchy-Green deformation tensor and its material time derivative    
    C = tf.keras.layers.Lambda(lambda F: CM.ten2_C(F), name='C' )(F)
    C_bar = tf.keras.layers.Lambda(lambda C: CM.ten2_C_bar(C), name='C_bar')(C)
    C_dot = tf.keras.layers.Lambda(lambda x: CM.ten2_C_dot(x[0],x[1]), name='C_dot' )([F,F_dot]) 
    
    # Deformation gradient and right Cauchy-Green deformation tensor in 
    # the reference configuration (no reference for the invariants based on C_dot necessary)
    F_ref = tf.keras.layers.Lambda(lambda F: CM.ten2_F_ref(F), name='F_ref')(F)    # DO NOT use output_shape=(None,tf.shape(F)[1],3,3) as argument to the lambda layer. Will cause massive problems when saving / serializing
    C_ref = tf.keras.layers.Lambda(lambda F: CM.ten2_C(F), name='C_ref')(F_ref)
    C_bar_ref = tf.keras.layers.Lambda(lambda C: CM.ten2_C_bar(C), name='C_bar_ref')(C_ref)   
    
    ### Other extra feature inputs that affect the material properties (e.g. temperature, filler content, ...)
    if numExtra == 0:
        extra_in = []
    else:
        extra_in = tf.keras.layers.Input(shape=(nSteps,numExtra,), name='extra_input') # INPUT    
       
    ### Preferred material directions and structure tensors
    # isotropic material
    if numDir == 0:
        extra_struc_in = []
        dir_model = []
        w_model = []
        
        dir = [] # we do not need directions (and hence their sub-ANN) at all
        w = tf.ones([batchSize,nSteps,numTens,1], dtype=tf_float) # we do not need a sub-ANN to get the weights
        L = []
        shaper = batchSize*tf.constant([1,0,0,0,0]) + tf.constant([0,0,1,1,1]) + nSteps*tf.constant([0,1,0,0,0])
        L = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.zeros([3,3], dtype=tf_float),0),0),0), shaper)
        inputs = F

    # anisotropic material; fiber directions and weights of the structural tensors do not depend on any additional input
    elif numDir != 0 and numExtraStruc == 0:
        extra_struc_in = []
        
        model_dir = CM.dirModel(numTens, numDir, name='model_dir')
        dir = model_dir(F)  # (?,nSteps,numDir,3)
        model_weight = CM.weightModel(numTens, numDir, name='model_w')
        w = model_weight(F) # (?,nSteps,numTens,numDir+1)
        L = tf.keras.layers.Lambda(lambda dir: CM.ten2_L(dir), name='L')(dir) # (?,nSteps,numDir,3,3)
        inputs = F
    
    # anisotropic material; fiber directions and weights of the structural tensors depend on additional input
    elif numDir != 0 and numExtraStruc != 0:
        extra_struc_in = tf.keras.layers.Input(shape=(nSteps, numExtraStruc,), name='extra_struc_input') # INPUT

        # Create a model from direction sub-ANN
        dir_ANNs = []
        for ii in range(numDir):
            dir_ann = subANNs.dir_subANN(extra_struc_in, layer_size_dir, activations_dir, str(ii+1)) # (?,nSteps,3)
            dir_ANNs.append(dir_ann)

        dir_ANNs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-2), name='stack_dir')(dir_ANNs) 
        dir_model = tf.keras.models.Model(inputs=extra_struc_in, outputs=dir_ANNs, name='model_dir') # (?,nSteps,numDir,3)

        # Create model from weights sub ANN
        w_ANNs = []
        for ii in range(numTens):
            w_ann = subANNs.w_subANN(extra_struc_in, layer_size_w, activations_w, numDir, str(ii+1))  # (?,nSteps,numDir+1)
            w_ANNs.append(w_ann)

        w_ANNs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-2), name='stack_w')(w_ANNs)
        w_model = tf.keras.models.Model(inputs=extra_struc_in, outputs=w_ANNs, name='model_w') # (?,nSteps,numTens,numDir+1)
        
        dir = dir_model(extra_struc_in) # (?,nSteps,numDir,3)
        w = w_model(extra_struc_in) # (?,nSteps,numTens,numDir+1) ???
        L = tf.keras.layers.Lambda(lambda dir: CM.ten2_L(dir), name='L')(dir) # (?,nSteps,numDir,3,3)
    
        inputs = extra_struc_in

    ### Compute the generalized structure tensors H from the standard structural tensors L and the weights 
    H = tf.keras.layers.Lambda(lambda x: CM.ten2_H(x[0], x[1], x[2], x[3], x[4]), name='H')([L, w, nSteps, numDir, numTens])      
    
    ### Compute the invariants
    if incomp:
        # isochoric invariants
        I = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1], x[2]), name='invars_I')([C_bar,H,numTens])
        J = tf.keras.layers.Lambda(lambda x: CM.invariants_J_incomp(x[0], x[1], x[2]), name='invars_J')([C_bar,H,numTens])
    else:
        I = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1]), name='invars_I')([C,H,numTens])
        J = tf.keras.layers.Lambda(lambda x: CM.invariants_J_comp(x[0], x[1], x[2]), name='invars_J')([C,H,numTens])
    det_C = tf.keras.layers.Lambda(lambda C: CM.invariant_I3(C), name='invar_det_C')(C)
    
    # Invariants in the stress-free reference configuration
    if incomp:
        I_ref = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1], x[2]) , name='invars_I_ref')([C_bar_ref,H,numTens])
        J_ref = tf.keras.layers.Lambda(lambda x: CM.invariants_J_incomp(x[0], x[1], x[2]) , name='invars_J_ref')([C_bar_ref,H,numTens])
    else:
        I_ref = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1], x[2]) , name='invars_I_ref')([C_ref,H,numTens])
        J_ref = tf.keras.layers.Lambda(lambda x: CM.invariants_J_comp(x[0], x[1], x[2]) , name='invars_J_ref')([C_ref,H,numTens])
    det_C_ref = tf.keras.layers.Lambda(lambda C_ref: CM.invariant_I3(C_ref), name='invar_det_C_ref')(C_ref)
        
    # invariants of the strain rate tensor
    I_dot     = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1], x[2]), name='invars_I_dot')([C_dot,H,numTens])
    J_dot     = tf.keras.layers.Lambda(lambda x: CM.invariants_J_comp(x[0], x[1], x[2]), name='invars_J_dot')([C_dot,H,numTens])
    det_C_dot = tf.keras.layers.Lambda(lambda C_dot: CM.invariant_I3(C_dot), name='invar_det_C_dot')(C_dot)    

    ### Concatenate invariants
    Invars     = tf.keras.layers.concatenate([I, J]        , name='concat_invars')
    Invars_ref = tf.keras.layers.concatenate([I_ref, J_ref], name='concat_invars_ref')
    Invars_dot = tf.keras.layers.concatenate([I_dot, J_dot, det_C_dot], name='concat_invars_dot')
    
    ### Hyperelasticity
    Invars_in = tf.keras.Input(shape=(nSteps,numTens*2,), name='Invars_in')
        
    # Loop over all generalized structural tensors
    IJ_split = tf.keras.layers.Lambda(lambda x: tf.split(x, numTens*2, axis=-1), name='split_IJ')(Invars_in)
    Psi_subANNs = []
    for ii in range(numTens):
        if numExtra == 0:
            IJ_in = tf.keras.layers.concatenate([IJ_split[ii], IJ_split[numTens+ii]], name='concat_IJ_'+str(ii))
        else:
            IJ_in = tf.keras.layers.concatenate([IJ_split[ii], IJ_split[numTens+ii], extra_in], name='concat_IJ_extra_'+str(ii))

        # create model for strain energy sub ANN
        Psi_ann = subANNs.Psi_subANN(IJ_in, layer_size_psi, activations_psi, str(ii+1))
        Psi_subANNs.append(Psi_ann)

    if numTens == 1:
        Psi_subANNs = Psi_subANNs[0] # quick-fix: otherwise tf.keras throws an error when loading the model and tf.keras.concatenate is called on a list with only one element/strain energy contribution (Issue #127 on Github tf-keras)
    else:
        Psi_subANNs = tf.keras.layers.concatenate(Psi_subANNs, axis=-1, name='concat_psi_subAnns') 
    
    if numExtra == 0:
        inputs = Invars_in
    else:
        inputs = [Invars_in, extra_in]
    model_Psi = tf.keras.models.Model(inputs=inputs, outputs=Psi_subANNs, name='model_Psi')
    
    
    # Evaluate strain energy models
    if numExtra == 0:
        Psi_    = model_Psi(Invars)
        Psi_ref = model_Psi(Invars_ref)    
    else: 
        Psi_    = model_Psi([Invars, extra_in])
        Psi_ref = model_Psi([Invars_ref, extra_in])
    
    ### isochoric contribution
    Psi_    = tf.keras.layers.Lambda(lambda x: tf.split(x, numTens, axis=-1), name='split_Psi')(Psi_)
    Psi_ref = tf.keras.layers.Lambda(lambda x: tf.split(x, numTens, axis=-1), name='split_Psi_ref')(Psi_ref)

    ### apply offset for stress-free reference configuration; follows Linden et al., 2023
    dPsidI_ref = [CM.GradientLayer(scale=False, name='dPsidI_ref_{:}'.format(ii+1))(psi, I_ref)[:,:,ii:ii+1] for ii, psi in enumerate(Psi_ref)] # [numTens * (?, nSteps, 1)]
    dPsidJ_ref = [CM.GradientLayer(scale=False, name='dPsidJ_ref_{:}'.format(ii+1))(psi, J_ref)[:,:,ii:ii+1] for ii, psi in enumerate(Psi_ref)] # [numTens * (?, nSteps, 1)]
    delta = [tf.keras.layers.Subtract(name='delta_{:}'.format(ii))([dpdi, dpdj]) for ii, (dpdi, dpdj) in enumerate(zip(dPsidI_ref, dPsidJ_ref),1) ] # [numTens * (?, nSteps, 1)]
    alpha = [tf.keras.layers.Lambda(lambda x: tf.nn.relu(-x), name='alpha_{:}'.format(ii))(d) for ii, d in enumerate(delta,1)] # [numTens * (?, nSteps, 1)]
    beta  = [tf.keras.layers.Lambda(lambda x: tf.nn.relu(x),  name='beta_{:}'.format(ii))(d)  for ii, d in enumerate(delta,1)] # [numTens * (?, nSteps, 1)]
    
    Psi_sigma = [CM.PsiSigmaLayer(name='Psi_sigma_{:}'.format(ii+1))(a, b, I[:,:,ii:ii+1], J[:,:,ii:ii+1]) for ii, (a,b) in enumerate(zip(alpha,beta))] # [numTens * (?, nSteps, 1)]
    S_sigma = [CM.GradientLayer(scale=True, name='S_sigma_{:}'.format(ii))(psi_s, C) for ii, psi_s in enumerate(Psi_sigma,1)] # [numTens * (?, nSteps, 3, 3)]

    ### stress
    S_e_ =    [CM.GradientLayer(scale=True, name='dPsidC_{:}'.format(ii))(psi, C) for ii, psi in enumerate(Psi_,1)] #  [numTens * (?, nSteps, 3, 3)]
    
    ### Apply offset for the energy-free reference configuration
    if numTens == 1: # TensorFlow requires special treatment if numTens==1, otherwise problems during prediction
        Psi_    = Psi_[0]
        Psi_ref = Psi_ref[0]
        Psi_sigma = Psi_sigma[0]
    else:
        Psi_      = tf.keras.layers.concatenate(Psi_, axis=-1, name='concat_psi') # (?, nSteps, numTens)
        Psi_ref   = tf.keras.layers.concatenate(Psi_ref, axis=-1, name='concat_psi_ref') # (?, nSteps, numTens)  
        Psi_sigma = tf.keras.layers.concatenate(Psi_sigma, axis=-1, name='concat_psi_sigma') # (?, nSteps, numTens)  
        
    Psi = tf.keras.layers.Add(name='Psi')([Psi_, -Psi_ref, Psi_sigma]) # (?, nSteps, numTens)
        
    ### Apply offset for the stress-free reference configuration
    if numTens == 1:
        S_e_ = S_e_[0]
        S_sigma = S_sigma[0]        
    else:
        S_e_    = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=2), name='stack_S_e')(S_e_) # (?, nSteps, numTens, 3, 3)
        S_sigma = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=2), name='stack_S_sigma')(S_sigma) # (?, nSteps, numTens, 3, 3)    
    
    S_e = tf.keras.layers.Add(name='S_e')([S_e_, S_sigma])  # (?, nSteps, numTens, 3, 3)
    
    # unstack instantaneous elastic stress into its individual components
    if numTens == 1:
        S_e = [S_e]
    else:
        S_e = tf.keras.layers.Lambda(lambda x: tf.unstack(x, numTens, axis=2), name='unstack_S_e')(S_e) # [numTens * (?, nSteps, 3, 3)]


    ### viscous part 
    
    # Time
    time = tf.keras.layers.Input(shape=(nSteps,), name = 'time_input')
    
    if visco == True:
           
        S = []
        S_infy = []
        Q_sum = []
        PronyParamsAll = []
        
        Invars_dot_in = tf.keras.Input(shape=(nSteps,numTens*2+1,), name='Invars_dot_in')       
        if uncoupled:
            IJ_dot_split = tf.keras.layers.Lambda(lambda x: tf.split(x, numTens*2+1, axis=-1), name='split_IJ_dot')(Invars_dot_in)
        
        else:
            IJ_in = Invars_in 
            if rateDependent:
                IJ_in = tf.keras.layer.concatenate([IJ_in, Invars_dot_in], name='prony_concat_invars_dot')
            if numExtra != 0:                
                IJ_in = tf.keras.layer.concatenate([IJ_in, extra_in], name='prony_concat_extra')
        
        # Loop over each instantanous elastic stress contribution / generalized structural tensor   
        for ii in range(numTens):
            if uncoupled:
                IJ_in = tf.keras.layers.concatenate([IJ_split[ii], IJ_split[numTens+ii]], name='prony_concat_IJ_'+str(ii))
                if rateDependent:
                    IJ_in = tf.keras.layers.concatenate([IJ_in, IJ_dot_split[ii], IJ_dot_split[numTens+ii], IJ_dot_split[-1]], name='prony_concat_IJ_dot_'+str(ii))
                if numExtra != 0:
                    IJ_in = tf.keras.layers.concatenate([IJ_in, extra_in], name='prony_concat_extra_'+str(ii))
                         
            ### deformation (rate) dependent relaxation time        
            G_subANNs = []
            for jj in range(nMaxwell+1):
                G_ann = subANNs.G_subANN(IJ_in, layer_size_g, activations_g, '{:}_{:}'.format(ii+1,jj))
                G_subANNs.append(G_ann)
                           
            G_subANNs = tf.keras.layers.concatenate(G_subANNs, axis=-1, name='concat_g_{:}'.format(ii))
                                    
            ### sparsity regularization
            if lambda_prony == 0.0:
                trainable = False
            else:
                trainable = True
            L1_Conv1D = tf.keras.layers.DepthwiseConv1D(
                                    kernel_size=1,
                                    strides=1,
                                    depth_multiplier=1,
                                    activation=None,
                                    use_bias=False,
                                    depthwise_initializer=tf.keras.initializers.ones,
                                    depthwise_regularizer=utils.SparsityRegularizer(l1=lambda_prony), #tf.keras.regularizers.L1(l1=lambda_prony),
                                    depthwise_constraint=tf.keras.constraints.NonNeg(),
                                    name='regularization_layer_g_{:}'.format(ii),
                                    trainable=trainable)
            
            G_subANNs = L1_Conv1D(G_subANNs) # (?, nSteps, nMaxwell+1)
            
            ### deformation-/stress (-rate) dependent relaxation time
            Tau_subANNs = []
            for jj in range(nMaxwell):
                Tau_ann = subANNs.Tau_subANN(IJ_in, layer_size_tau, activations_tau, '{:}_{:}'.format(ii+1,jj+1))
                Tau_subANNs.append(Tau_ann)
            Tau_subANNs = tf.keras.layers.concatenate(Tau_subANNs, axis=-1, name='concat_tau_{:}'.format(ii))
    
            # scale the relaxation times
            scale_layer = CM.ScaleLayer(T_min=tau[0], T_max=tau[1], name='scale_layer_{}'.format(ii+1))
            Tau_subANNs = scale_layer(Tau_subANNs)
            
            # inputs to the relaxation coefficient and time models
            inputs = [Invars_in]
            if rateDependent:
                inputs.append(Invars_dot_in)
            if numExtra != 0:
                inputs.append(extra_in) 

            # Construct relaxation coefficient and times models        
            model_g = tf.keras.models.Model(inputs=inputs, outputs=G_subANNs, name='model_g_{:}'.format(ii))
            model_tau = tf.keras.models.Model(inputs=inputs, outputs=Tau_subANNs, name='model_tau_{:}'.format(ii))            
            
            # Evaluate relaxation coefficient and time models
            inputs = [Invars]
            if rateDependent:
                inputs.append(Invars_dot)
            if numExtra != 0:
                inputs.append(extra_in)
            
            Gs = model_g(inputs)
            Taus = model_tau(inputs)
                        
            PronyParams = tf.keras.layers.concatenate([Taus, Gs], name='concat_pronyParams_{:}'.format(ii))
            PronyParamsAll.append(PronyParams)
                    
            S_e_i = S_e[ii] # only the isochric part of the stress should be affected by viscous effects
            
            ### Algorithmic stress update
            # S_i: total stress of the generalized Maxwell model corresponding to a specific generalized structural tensor
            # S_infyi : equilibrium stress of the generalized Maxwell model corresponding to a specific generalized structural tensor
            # Q_sum_i : viscous stress of the generalized Maxwell model corresponding to a specific generalized structural tensor
            S_i, S_infy_i, Q_sum_i = CM.stressUpdateLayer(nMaxwell, nSteps, name='S_{:}'.format(ii+1))(S_e_i, time, PronyParams)           
            
            # collect all stresses
            S.append(S_i)
            S_infy.append(S_infy_i)
            Q_sum.append(Q_sum_i)

        if numTens == 1:
            PronyParamsAll = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2), name='stack_prony_params_all')(PronyParamsAll[0])
        else:
            PronyParamsAll = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=2), name='stack_prony_params_all')(PronyParamsAll) # (?, nSteps, numTens, 2*nMaxwell+1)
            
        Taus, Gs = tf.keras.layers.Lambda(lambda x: tf.split(x, [nMaxwell, nMaxwell+1], axis=-1), name='split_prony_params')(PronyParamsAll) # (?, nSteps, numTens, nMaxwell), (?, nSteps, numTens, nMaxwell+1)
        
        # apply incompressibility condition for the viscoelastic stress
        if incomp:
            S = [tf.keras.layers.Lambda(lambda x: CM.ten2_S_incomp(x[0],x[1]), name='S_incomp_{:}'.format(ii))([s, C])  for ii, s in enumerate(S, 1)]# (?, nSteps, numTens, 3, 3)

        # apply incompressibility condition for the equilibrium stress
        if incomp:
            S_infy = [tf.keras.layers.Lambda(lambda x: CM.ten2_S_incomp(x[0],x[1]), name='S_infy_incomp_{:}'.format(ii))([s_infy, C])  for ii, s_infy in enumerate(S_infy, 1)]# (?, nSteps, numTens, 3, 3)
                    
    # apply incompressibility condition for the elastic stress
    if incomp:
        S_e = [tf.keras.layers.Lambda(lambda x: CM.ten2_S_incomp(x[0],x[1]), name='S_e_incomp_{:}'.format(ii))([s_e, C])  for ii, s_e in enumerate(S_e, 1)]# (?, nSteps, numTens, 3, 3)
    
    
    if visco == False : 
        S = S_e
        
    if numTens == 1:
        S = S[0]
    else:
        S = tf.keras.layers.add(S, name='add_S_i')
        
    P, P11, sigma = CM.stressTensors(S, F)

    # Create and return models
    inputs = [F, time]  
    if rateDependent:
        inputs.append(F_dot)
    if numExtraStruc != 0:
        inputs.append(extra_struc_in)
    if numExtra != 0:
        inputs.append(extra_in)

    model_fit  = tf.keras.models.Model(inputs, P) # The one for fitting (only P11 as output in order to not require a complicated loss function)

    # intermediate results for debugging and other purposes
    model_full =  tf.keras.models.Model(inputs, [Invars, Psi, Psi_sigma, delta, alpha, beta,
                                                 S_e_, S_sigma, S_e, Gs, Taus, P, S,
                                                 sigma, S_i, S_infy_i, Q_sum_i])
    
    return model_fit, model_full

