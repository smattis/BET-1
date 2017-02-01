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
import bet.surrogates as surrogates

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
                 set_of_interest,
                 region_of_interest,
                 error_estimates=False,
                 jacobians=False):
        
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.tol = tol
        self.num_levels = len(lb_model_list)
        self.lb_model_list = lb_model_list
        self.error_estimates = error_estimates
        self.jacobians = jacobians
        self.total_samples = 0
        self.set_of_interest = set_of_interest
        self.region_of_interest = region_of_interest
        

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
        num = self.disc.check_nums()
        self.disc._input_sample_set.set_levels(level*np.ones((num,), dtype=int))
        self.total_samples += int(num)
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

    def pseudoinverse_samples(self, inputs, outputs, level=0):
        import numpy.linalg as nla
        if self.jacobians is None:
            raise wrong_critera("Jacobians are required for pseudoinverse")

        num_old = self.disc.check_nums()

        #Check inputs and outputs to see if they are already right
        if self.disc._io_ptr is None:
            self.disc.set_io_ptr()
        good = np.not_equal(self.disc._io_ptr[inputs], outputs)
        inputs = inputs[good]
        outputs = outputs[good]

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

        in1 = np.greater_equal(new_vals, self.disc._input_sample_set._domain[:,0])
        in2 = np.less_equal(new_vals, self.disc._input_sample_set._domain[:,1])
        in_both = np.logical_and(in1, in2)
        num_new = int(np.sum(in_both))
        new_vals = new_vals[in_both]
        new_vals = util.fix_dimensions_data(new_vals, self.disc._input_sample_set._dim)
        new_sset = sample.sample_set(self.disc._input_sample_set._dim)
        new_sset.set_domain(self.disc._input_sample_set._domain)
        new_sset.set_values(new_vals)

  
        new_inputs = inputs[in_both.flat[:]] + num_old
        new_outputs = outputs[in_both.flat[:]] 
        new_sampler = bsam.sampler(self.lb_model_list[level],
                                       error_estimates = self.error_estimates,
                                       jacobians = self.jacobians)
        new_disc = new_sampler.compute_QoI_and_create_discretization(
            new_sset,
            savefile=None,
            globalize=True)
        new_disc._input_sample_set.set_levels(level*np.ones((num_new,), dtype=int))

        self.disc._input_sample_set.append_sample_set(
            new_disc._input_sample_set)
        self.disc._output_sample_set.append_sample_set(
            new_disc._output_sample_set)
        
        self.disc.set_io_ptr()
        samps_bad = np.not_equal(self.disc._io_ptr[num_old:num_old+num_new], new_outputs)

        inputs_new = np.arange(num_old, num_old+num_new)[samps_bad]
        outputs_new = new_outputs[samps_bad]

        for i in range(self.disc._output_probability_set.check_num()):
            logging.info("Set " + `i` + " has " + `int(np.sum(np.equal(self.disc._io_ptr, i)))` + '/' + `num_old+num_new`)
            print "Set " + `i` + " has " + `int(np.sum(np.equal(self.disc._io_ptr, i)))` + '/' + `num_old+num_new`
        
        return (inputs_new, outputs_new)

    def level_refinement_inds(self,
                         indices):
        """
        Increases the level of selcted samples.

        """
        num_models = len(self.lb_model_list)
        good = np.less(self.disc._input_sample_set._levels, num_models - 1)

        indices = indices[good[indices]]
        current_levels = self.disc._input_sample_set._levels[indices]
        for i in range(num_models-1, 0, -1):
            inds = indices[np.equal(current_levels, i - 1)]
            num_new = len(inds)
            new_sset = sample.sample_set(self.disc._input_sample_set._dim)
            new_sset.set_domain(self.disc._input_sample_set._domain)
            new_sset.set_values(self.disc._input_sample_set._values[inds,:])

   
            new_sampler = bsam.sampler(self.lb_model_list[i],
                                       error_estimates = self.error_estimates,
                                       jacobians = self.jacobians)
            new_disc = new_sampler.compute_QoI_and_create_discretization(
                new_sset,
                savefile=None,
                globalize=True)
            self.disc._output_sample_set._values[inds,:] = new_disc._output_sample_set._values[:]
            self.disc._input_sample_set._levels[inds] = i
            if self.error_estimates:
                self.disc._output_sample_set._error_estimates[inds,:] = new_disc._output_sample_set._error_estimates[:]
            if self.jacobians:
                self.disc._input_sample_set._jacobians[inds,:] = new_disc._input_sample_set._jacobians[:]

    def evaluate_surrogate(self, input_sample_set, order=0):
        self.sur = surrogates.piecewise_polynomial_surrogate(self.disc)
        self.sur.generate_for_input_set(input_sample_set=input_sample_set,
                                   order=order)
        (prob, ee) = self.sur.calculate_prob_for_sample_set_region(self.set_of_interest, 
                                             regions=[self.region_of_interest], update_input=True)
        return (prob, ee)

    def h_refinement_random(self, input_sample_set, order=0,
                            num_new_samples=10, level=0):
        (prob, ee) = self.evaluate_surrogate(input_sample_set, order)
        input_sample_set.local_to_global()
        error_ids = np.abs(input_sample_set._error_id[:])
        inds = np.argsort(error_ids, axis=0)
        new_vals = input_sample_set._values[inds[0:num_new_samples],:]

        
        new_sset = sample.sample_set(self.disc._input_sample_set._dim)
        new_sset.set_domain(self.disc._input_sample_set._domain)
        new_sset.set_values(new_vals)

  
       
        new_sampler = bsam.sampler(self.lb_model_list[level],
                                       error_estimates = self.error_estimates,
                                       jacobians = self.jacobians)
        new_disc = new_sampler.compute_QoI_and_create_discretization(
            new_sset,
            savefile=None,
            globalize=True)
        new_disc._input_sample_set.set_levels(level*np.ones((num_new_samples,), dtype=int))

        self.disc._input_sample_set.append_sample_set(
            new_disc._input_sample_set)
        self.disc._output_sample_set.append_sample_set(
            new_disc._output_sample_set)
        self.disc._input_sample_set.local_to_global()
        return (prob, ee)
    
                                          
    def level_refinement(self, input_sample_set, order=0,
                            num_new_samples=10):
        (prob, ee) = self.evaluate_surrogate(input_sample_set, order)
        error_ids = np.abs(self.disc._input_sample_set._error_id[:])
        inds = np.argsort(error_ids, axis=0)
        #new_vals = self.dicinput_sample_set._values[inds[0:num_new_samples],:]
        inds = inds[0:num_new_samples]
        self.level_refinement_inds(inds)

        return (prob, ee)
