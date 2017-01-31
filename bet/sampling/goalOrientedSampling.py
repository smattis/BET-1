# Copyright (C) 2014-2017 The BET Development Team

"""
This module contains functions for goal-oriented sampling
"""
import math, os, glob, logging
import numpy as np
import scipy.io as sio
import bet.sampling.basicSampling as bsam
import bet.util as util
from bet.Comm import comm 
import bet.sample as sample

class wrong_critera(Exception):
    """
    Exception for when the criteria for sampling method are not met.
    """

class sampler(bsam.sampler):
    """
    This class provides methods for goal-oriented adaptive sampling of parameter 
    space to provide samples to be used by algorithms to solve inverse problems. 
    """
    def __init__(self,
                 max_samples,
                 min_samples,
                 tol,
                 lb_model_list,
                 error_estimates=False,
                 jacobians=False):
        
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.tol = tol
        self.num_levels = len(lb_model_list)
        self.lb_model_list = lb_model_list
        self.error_estimates = error_estimates
        self.jacobians = jacobians
        

    def initial_samples_random(self,
                               sample_type,
                               input_obj,
                               num_samples=1E2,
                               criterion='center',
                               globalize=True,
                               level=0,
                               savefile='initial_discretization.txt.gz'):
        initial_sampler = bsam.sampler(self.lb_model_list[level],
                                       error_estimates = self.error_estimates,
                                       jacobians = self.jacobians)
        input_obj = initial_sampler.random_sample_set(sample_type,
                                          input_obj,
                                          num_samples,
                                          criterion,
                                          globalize)
        self.disc = initial_sampler.compute_QoI_and_create_discretization(
            input_obj, savefile=savefile)

        return self.disc

    def initial_samples_regular(self,
                                input_obj,
                                level=0,
                                num_samples_per_dim=1,
                                savefile='initial_discretization.txt.gz'):
    
        initial_sampler = bsam.sampler(self.lb_model_list[level],
                                       error_estimates = self.error_estimates,
                                       jacobians = self.jacobians)
        input_obj = initial_sampler.regular_sample_set(input_obj,
                                           num_samples_per_dim)
        import pdb
        pdb.set_trace()
        self.disc = initial_sampler.compute_QoI_and_create_discretization(
            input_obj, savefile=savefile)
        return self.disc

    def pseudoinverse_samples(self, inputs, outputs):
        import numpy.linalg as nla
        if self.jacobians is None:
            raise wrong_critera("Jacobians are required for pseudoinverse")

        vals = self.disc._input_sample_set._values[inputs,:]
        jacs = self.disc._input_sample_set._jacobians[inputs, :, :]
        q0 = self.disc._output_sample_set._values[inputs, :]
        ees = self.disc._output_sample_set._error_estimates[inputs, :]
        (N, M, P) = jacs.shape
        pi_jacs = np.zeros((N, P, M))
        qs = self.disc._output_probability_set._values[outputs,:]

        # Calculate pseudoinverses
        for i in range(N):
            pi_jacs[i,:,:] = nla.pinv(jacs[i,:,:])

        # Calculate new samples
        if self.error_estimates:
            new_vals = np.einsum('ijk,ik->ij', pi_jacs, qs - q0 - ees) + vals
        else:
            new_vals = np.einsum('ijk,ik->ij', pi_jacs, qs - q0) + vals
        return new_vals

        
        

    
                                          
