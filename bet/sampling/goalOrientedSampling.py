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
        #import pdb
        #pdb.set_trace()
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
        #good = np.less(self.disc._input_sample_set._levels, num_models - 1)

        #indices = indices[good[indices]]
        current_levels = self.disc._input_sample_set._levels[indices]
        #import pdb
        #pdb.set_trace()
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
        self.disc._io_ptr = None
        self.disc._io_ptr_local = None
        #pdb.set_trace()
            
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
        self.disc._io_ptr = None
        self.disc._io_ptr_local = None
        return (prob, ee)

    def h_refinement_cluster(self, input_sample_set, order=0,
                             num_new_samples=10, level=0, tol=0.01, match_level=True):
        import scipy.cluster as clust
        import matplotlib.pyplot as plt
        #import pdb

        if input_sample_set._error_id_local is None:
            (prob, ee) = self.evaluate_surrogate(input_sample_set, order)
            input_sample_set.local_to_global()
        else:
            (prob, ee) = (np.nan, np.nan)
        cluster_inds = np.not_equal(input_sample_set._error_id, 0.0)
        cluster_vals = input_sample_set._values[cluster_inds, :]
        # eid_vals = input_sample_set._error_id[cluster_inds]
        # eid_vals = np.floor(eid_vals/np.min(eid_vals)).astype(int) - 1
        # for i in range(len(eid_vals)):
        #     if eid_vals[i] > 0:
        #         new_clut = np.repeat(np.reshape(cluster_vals[i,:], (1, input_sample_set._dim)), eid_vals[i], axis=0)
        #         #if eid_vals[i] > 0:
        #         #import pdb
        #         #pdb.set_trace()
        #         cluster_vals = np.vstack((cluster_vals, new_clut))
            
        print cluster_vals.shape
        if np.any(cluster_inds):
        #pdb.set_trace()
            #import pdb

            eList = []
            centList = []
            krange = min(cluster_vals.shape[0] + 1, num_new_samples + 1)
            print krange
            for i in range(1, krange):
                try:
                    (new_vals, L) = clust.vq.kmeans2(cluster_vals, i, thresh=1.0e-7)
                except:
                    #import pdb
                    #pdb.set_trace()
                    #pass
                    return (prob,ee)
                centList.append(new_vals)
                error = 0.0
                for k in range(i):
                    inds = np.equal(L, k)
                    error += np.sum((cluster_vals[inds,:] - new_vals[k,:])**2)
                eList.append(error)
            #plt.figure()
            #plt.plot(range(1, num_new_samples + 1), eList)
            #plt.show()
            #pdb.set_trace()
            eList = np.array(eList)
            diffs = (eList[1::] - eList[0:-1])/np.max(eList)
            go = True
            count = 0
            while go and count < (krange-2):
                try:
                    go2 =  (diffs[count] <= 0.0 and abs(diffs[count]) <= tol and eList[count+1]/np.min(eList) < 10.0) or (diffs[count] > 0.0 and eList[count+1]/np.min(eList) < 10.0)
                except:
                    import pdb
                    pdb.set_trace()
                if go2:
                    knum = count + 1
                    go = False
                else:
                    count += 1
            if go:
                knum = krange-1 #num_new_samples

            # print knum
            # plt.figure()
            # plt.plot(range(1, num_new_samples + 1), eList)
            # plt.show()

            #import pdb
            #pdb.set_trace()
            new_vals = centList[knum-1]

            if match_level:
                (_, old_sets) = self.disc._input_sample_set.query(new_vals)
                old_levels = self.disc._input_sample_set._levels[old_sets]
                for Level in range(len(self.lb_model_list)):
                    go = np.equal(old_levels, Level)
                    if np.any(go):
                        n_vals = new_vals[go]
                        new_sset = sample.sample_set(self.disc._input_sample_set._dim)
                        new_sset.set_domain(self.disc._input_sample_set._domain)


                        new_sset.set_values(n_vals)        


                        num_new_samples = len(n_vals)
                        new_sampler = bsam.sampler(self.lb_model_list[Level],
                                                       error_estimates = self.error_estimates,
                                                       jacobians = self.jacobians)
                        new_disc = new_sampler.compute_QoI_and_create_discretization(
                            new_sset,
                            savefile=None,
                            globalize=True)
                        new_disc._input_sample_set.set_levels(Level*np.ones((num_new_samples,), dtype=int))

                        self.disc._input_sample_set.append_sample_set(
                            new_disc._input_sample_set)
                        self.disc._output_sample_set.append_sample_set(
                            new_disc._output_sample_set)
                        self.disc._input_sample_set.local_to_global()
                self.disc._io_ptr = None
                self.disc._io_ptr_local = None
                self.disc._input_sample_set.kdtree = None
                
                        
            else:

                new_sset = sample.sample_set(self.disc._input_sample_set._dim)
                new_sset.set_domain(self.disc._input_sample_set._domain)


                new_sset.set_values(new_vals)        


                num_new_samples = knum
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
                self.disc._io_ptr = None
                self.disc._io_ptr_local = None
            
        return (prob, ee)
        
    
                                          
    def level_refinement(self, input_sample_set=None, order=0,
                            num_new_samples=10):
        if input_sample_set is not None:
            (prob, ee) = self.evaluate_surrogate(input_sample_set, order)
        else:
            (prob, ee) = (np.nan, np.nan)
        error_ids = np.abs(self.disc._input_sample_set._error_id[:])
        inds = np.argsort(error_ids, axis=0)[::-1]
        #import pdb
        #pdb.set_trace()
        #new_vals = self.dicinput_sample_set._values[inds[0:num_new_samples],:]
        num_mod = len(self.lb_model_list)
        good = np.less(self.disc._input_sample_set._levels, num_mod - 1)
        #inds[good[inds]]
        #import pdb
        #pdb.set_trace()
        num_go = int(np.sum(np.not_equal(self.disc._input_sample_set._error_id[inds], 0.0)))
        num_go = min(num_go, num_new_samples)
        inds = inds[0:num_go]
        #import pdb
        #pdb.set_trace()
        self.level_refinement_inds(inds)

        return (prob, ee)

    def h_refinement_opt(self, input_sample_set=None, order=0,
                         num_new_samples=10,
                         estimate_error=None,
                         x1=None, x2=None, max_props=3):
        #if input_sample_set is not None:
        #    (prob, ee) = self.evaluate_surrogate(input_sample_set, order)
        #else:
        (prob, ee) = (np.nan, np.nan)
        inds = np.argsort(self.disc._input_sample_set._error_id, axis=0)[::-1][:num_new_samples]
        # make more general
        (_, ptr) = self.disc._input_sample_set.query(input_sample_set._values)
        values = np.copy(self.disc._input_sample_set._values)
        best_vals = []
        best_levels = []
        for ind in inds:
            in_i = np.equal(ptr, ind)
            props = input_sample_set._values[in_i][0:max_props]
            eed = []
            #import pdb
            #pdb.set_trace()
            for j in range(props.shape[0]):
                new_values = np.concatenate((np.copy(values), np.array([props[j,:]]),), axis=0)
                disc_dummy = self.disc.copy()
                disc_dummy._kdtree = None
                disc_dummy._kdtree_values = None
                #kdtree = spatial.KDTree(new_values)
                error_id = estimate_error(disc_dummy, x1, x2)
                eed.append(np.sum(error_id))
            eed = np.array(eed)
            if len(eed) > 0:
                best_arg = np.argmin(eed, axis=0)
                best_vals.append(props[best_arg,:])
                best_levels.append(self.disc._input_sample_set._levels[ind])
        #import pdb
        #pdb.set_trace()

            
        #(_, old_sets) = self.disc._input_sample_set.query(new_vals)
        #old_levels = self.disc._input_sample_set._levels[old_sets]
        old_levels = np.array(best_levels)
        new_vals = np.array(best_vals)
        for Level in range(len(self.lb_model_list)):
            go = np.equal(old_levels, Level)
            if np.any(go):
                n_vals = new_vals[go]
                new_sset = sample.sample_set(self.disc._input_sample_set._dim)
                new_sset.set_domain(self.disc._input_sample_set._domain)


                new_sset.set_values(n_vals)        


                num_new_samples = len(n_vals)
                new_sampler = bsam.sampler(self.lb_model_list[Level],
                                               error_estimates = self.error_estimates,
                                               jacobians = self.jacobians)
                new_disc = new_sampler.compute_QoI_and_create_discretization(
                    new_sset,
                    savefile=None,
                    globalize=True)
                new_disc._input_sample_set.set_levels(Level*np.ones((num_new_samples,), dtype=int))

                self.disc._input_sample_set.append_sample_set(
                    new_disc._input_sample_set)
                self.disc._output_sample_set.append_sample_set(
                    new_disc._output_sample_set)
                self.disc._input_sample_set.local_to_global()
        self.disc._io_ptr = None
        self.disc._io_ptr_local = None
        self.disc._input_sample_set.kdtree = None
        
        return (prob, ee)

    def calculate_gamma(self, F, emulate_set=None):
        if emulate_set is not None:
            self.disc.set_emulated_input_sample_set(emulate_set)
        elif self.disc.emulated_input_sample_set is None:
            raise wrong_critera("Need emulated points.")
        self.disc.set_emulated_ii_ptr()

        F_vec = np.fabs(F(self.disc.emulated_input_sample_set._values))
        #zs = np.greater(F_vec, 1.0e-12)
        #pdiff = self.disc_enhanced._input_sample_set._probabilities - self.disc._input_sample_set._probabilities
        cells = np.greater(np.fabs(self.disc_enhanced._input_sample_set._probabilities, self.disc._input_sample_set._probabilities), 1.0e-12)
        cells_em = cells[self.disc._emulated_ii_ptr]
        #cells_m
        #self.disc._emulated_ii_ptr
        #cells = np.logical_and(cell, zs)
        #vol = np.sum(self.disc._input_sample_set._volumes[cells])
        #vol = np.average(cells)
        self.gamma = np.average(F_vec[cells_em])        
        return self.gamma

    def calculate_E_prob(self, F, emulate_set=None):
        if self.gamma is None:
            self.calculate_gamma(F, emulate_set)
        E_prob = self.gamma * np.fabs(self.disc_enhanced._input_sample_set._probabilities, self.disc._input_sample_set._probabilities)
        return E_prob
                                      
                
            
class sampler_hpl_adaptive(bsam.sampler):
    """
    This class provides methods for goal-oriented adaptive sampling of parameter 
    space to provide samples to be used by algorithms to solve inverse problems. 
    """
    def __init__(self,
                 lb_model_list,
                 f,
                 max_samples=int(1.0e6),
                 min_samples=int(1.0e2),
                 tol=1.0e-3,
                 #error_estimates=False,
                 jacobians=False):
        
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.tol = tol
        self.num_levels = len(lb_model_list)
        self.lb_model_list = lb_model_list
        self.f = f
        self.error_estimates = True #error_estimates
        self.jacobians = jacobians
        self.total_samples = 0
        self.total_evals = np.zeros((self.num_levels,), dtype=np.int)
         

    def initial_samples_random(self,
                               sample_type,
                               input_obj,
                               num_samples=1E2,
                               criterion='center',
                               globalize=True,
                               level=0,
                               emulate=False,
                               emulate_num=1E5,
                               savefile='initial_discretization.txt.gz'):
        initial_sampler = bsam.sampler(self.lb_model_list[level],
                                       error_estimates = True, #self.error_estimates,
                                       jacobians = self.jacobians)
        input_obj = initial_sampler.random_sample_set(sample_type,
                                          input_obj,
                                          num_samples,
                                          criterion,
                                          globalize)
        if emulate:
            initial_sampler_em = bsam.sampler(self.lb_model_list[level],
                                              error_estimates = True, #self.error_estimates,
                                       jacobians = self.jacobians)
            emulate_set = initial_sampler_em.random_sample_set(sample_type,
                                                               input_obj,
                                                               emulate_num,
                                                               criterion,
                                                               globalize)
            
        self.disc = initial_sampler.compute_QoI_and_create_discretization(
            input_obj, savefile=savefile)
        num = self.disc.check_nums()
        self.disc._input_sample_set.set_levels(level*np.ones((num,), dtype=int))
        self.total_samples += int(num)
        self.total_evals[level] += int(num)
        if emulate:
            self.disc.set_emulated_input_sample_set(emulate_set)
            self.fabs_em = np.fabs(self.f(self.disc._emulated_input_sample_set._values))
        return self.disc

    def initial_samples_regular(self,
                                input_obj,
                                level=0,
                                num_samples_per_dim=1,
                                savefile='initial_discretization.txt.gz'):
    
        initial_sampler = bsam.sampler(self.lb_model_list[level],
                                       error_estimates = True, #self.error_estimates,
                                       jacobians = self.jacobians)
        input_obj = initial_sampler.regular_sample_set(input_obj,
                                           num_samples_per_dim)
        #import pdb
        #pdb.set_trace()
        self.disc = initial_sampler.compute_QoI_and_create_discretization(
            input_obj, savefile=savefile)
        num = self.disc.check_nums()
        self.disc._input_sample_set.set_levels(level*np.ones((num,), dtype=int))
        self.total_evals[level] += int(num)
        return self.disc

    def set_emulation_set(self,
                          emulated_sample_set):
        self.disc.set_emulated_input_sample_set(emulate_sample_set)
        self.fabs_em = np.fabs(self.f(self.disc.emulated_input_sample_set._values))
        #self.cells_nonzero = np.greater(self.fabs_em, 1.0e-12)
    def set_probabilities(self,
                          probabilities,
                          probabilites_enhanced):
        self.disc._input_sample_set.set_probabilities(probabilities)
        self.probabilities_enhanced = probabilites_enhanced
        self.prob_diff = np.fabs(probabilities - probabilites_enhanced)

    def calculate_gamma(self):
        self.disc.set_emulated_ii_ptr()
        cells = np.greater(self.prob_diff, 1.0e-12)
        cells_em = cells[self.disc._emulated_ii_ptr]
        self.gamma = np.average(self.fabs_em[cells_em])        
        return self.gamma

    def calculate_ee_prob(self):
        self.ee_prob = self.gamma*(self.prob_diff)
        return self.ee_prob

    def calculate_prob_ee_int_chain(self, x, x_enhanced):
        num = self.disc.check_nums()
        n1 = float(len(x))
        n2 = float(len(x_enhanced))
        F1 = self.f(x)
        F2 = self.f(x_enhanced)
        (_, ptr1) = self.disc._input_sample_set.query(x)
        (_, ptr2) = self.disc._input_sample_set.query(x_enhanced)

        int1 = np.zeros((num,))
        int2 = np.zeros((num,))
        prob1 = np.zeros((num,))
        prob2 = np.zeros((num,))
        
        for i in range(num):
            io1 = np.equal(ptr1, i)
            int1[i] = np.sum(F1[io1])/n1
            io2 = np.equal(ptr2, i)
            int2[i] = np.sum(F2[io2])/n2
            prob1[i] = float(np.sum(io1))/n1
            prob2[i] = float(np.sum(io2))/n2
            
        self.ee_int = np.fabs(int1-int2)
        self.disc._input_sample_set_probabilities = prob1
        self.probabilities_enhanced = prob2
        self.prob_diff = np.fabs(prob1 - prob2)
        self.F1 = F1
        self.F2 = F2
        self.ptr1 = ptr1
        self.ptr2 = ptr2
        return self.ee_int, np.sum(int1), np.sum(int2)
    
    def calculate_ee_chain(self, x, x_enhanced):
        #(_, ptr1) =  self.disc._input_sample_set.query(x)
        #(_, ptr2) =  self.disc._input_sample_set.query(x)
        #self.calculate_gamma()
        #self.calculate_ee_prob()
        (_, int1, int2) = self.calculate_prob_ee_int_chain(x, x_enhanced)
        self.calculate_gamma()
        self.calculate_ee_prob()
        self.disc._input_sample_set._error_id = self.ee_int + self.ee_prob
        return self.disc._input_sample_set._error_id, int1, int2

    def calculate_subgrid_ee_chain(self, inds, subgrid_set, x, x_enhanced):
        #num = self.disc.check_nums()
        num_s = subgrid_set.check_num()
        n1 = float(len(x))
        n2 = float(len(x_enhanced))
        #F1 = self.f(x)
        #F2 = self.f(x_enhanced)
        #(_, ptr1) = self.disc._input_sample_set.query(x)
        #(_, ptr2) = self.disc._input_sample_set.query(x_enhanced)
        #(_, ptr1) = subgrid_set.query(x)
        #(_, ptr2) = subgrid_set.query(x_enhanced)

        #int1 = np.zeros((num,))
        #int2 = np.zeros((num,))
        #prob1 = np.zeros((num,))
        #prob2 = np.zeros((num,))
        E_local = []
        proposal_local = []
        levels_local = []
        
        for j in inds:
            id1 = np.equal(self.ptr1, j)
            x_loc1 = x[id1,:]
            F1 = self.F1[id1]
            id2 = np.equal(self.ptr2, j)
            x_loc2 = x_enhanced[id2,:]
            F2 = self.F2[id2]
            (_, ptr1) = subgrid_set.query(x_loc1)
            (_, ptr2) = subgrid_set.query(x_loc2)
            #nums1 = len(x_loc1)
            #nums2 = len(x_loc2)
            #import pdb
            #pdb.set_trace()
            
            int1 = np.zeros((num_s,))
            int2 = np.zeros((num_s,))
            prob1 = np.zeros((num_s,))
            prob2 = np.zeros((num_s,))
            for i in range(num_s):
                io1 = np.equal(ptr1, i)#, np.equal(self.ptr1, j))
                if np.sum(io1) > 0:
                    int1[i] = np.sum(F1[io1])/n1
                    prob1[i] = float(np.sum(io1))/n1
                io2 = np.equal(ptr2, i)#, np.equal(self.ptr2, j))
                if np.sum(io2) > 0:
                    int2[i] = np.sum(F2[io2])/n2
                    #prob1[i] = float(np.sum(io1))/n1
                    prob2[i] = float(np.sum(io2))/n2
            
            ee_int = np.fabs(int1-int2)
            #self.disc._input_sample_set_probabilities = prob1
            #self.probabilities_enhanced = prob2
            prob_diff = np.fabs(prob1 - prob2)
            ee_prob = self.gamma * prob_diff 
            ee = ee_int + ee_prob
            E_local.append(np.sum(ee))
            imax = np.argmax(ee)
            proposal_local.append(subgrid_set._values[imax,:])
            levels_local.append(self.disc._input_sample_set._levels[j])
        E_local = np.array(E_local)
        proposal_local = np.array(proposal_local)
        return (E_local, proposal_local, levels_local)

    def hl_step_setup_chain(self, x, x_enhanced, subgrid_set, factor=0.2):
        (ee, int1, int2) = self.calculate_ee_chain(x,x_enhanced)
        max_ee = np.max(ee)
        num_go = np.sum(np.greater(ee, factor*max_ee))
        inds = np.argsort(ee)[::-1][0:num_go+1]
        E = ee[inds]
        (El, props, levels) = self.calculate_subgrid_ee_chain(inds, subgrid_set, x, x_enhanced)
        self.hList=[]
        propsList=[]
        levelsList=[]
        self.lList=[]
        for i in range(len(inds)):
            if (El[i]/E[i] < 1.1) and (self.disc._input_sample_set._levels[i] < (self.num_levels - 1)):
                print El[i]/E[i]
                self.lList.append(inds[i])
            else:
                self.hList.append(inds[i])
                propsList.append(props[i])
                levelsList.append(levels[i])
               

        self.props = np.array(propsList)
        self.levels = np.array(levelsList)
        return (int1, int2, np.sum(ee))
    def hl_step_chain(self):
        if len(self.hList) > 0:
            self.h_refinement(self.props, self.levels)
            self.disc._input_sample_set.kdtree = None
        if len(self.lList) > 0:
            self.level_refinement(self.lList)
            
        self.disc._io_ptr = None
        self.disc._io_ptr_local = None            
        
        # import pdb
        # pdb.set_trace()
        
    def level_refinement(self,
                         indices):
        """
        Increases the level of selcted samples.

        """
        num_models = len(self.lb_model_list)
        #good = np.less(self.disc._input_sample_set._levels, num_models - 1)

        #indices = indices[good[indices]]
        current_levels = self.disc._input_sample_set._levels[indices]
        #import pdb
        #pdb.set_trace()
        indices = np.array(indices)
        for i in range(0, num_models-1):
            #import pdb
            #pdb.set_trace()
            inds = indices[np.equal(current_levels, i)]
            num_new = len(inds)
            if num_new > 0:
                new_sset = sample.sample_set(self.disc._input_sample_set._dim)
                new_sset.set_domain(self.disc._input_sample_set._domain)
                new_sset.set_values(self.disc._input_sample_set._values[inds,:])


                new_sampler = bsam.sampler(self.lb_model_list[i+1],
                                           error_estimates = self.error_estimates,
                                           jacobians = self.jacobians)
                new_disc = new_sampler.compute_QoI_and_create_discretization(
                    new_sset,
                    savefile=None,
                    globalize=True)
                self.disc._output_sample_set._values[inds,:] = new_disc._output_sample_set._values[:]
                self.disc._input_sample_set._levels[inds] = i + 1
                if self.error_estimates:
                    self.disc._output_sample_set._error_estimates[inds,:] = new_disc._output_sample_set._error_estimates[:]
                if self.jacobians:
                    self.disc._input_sample_set._jacobians[inds,:] = new_disc._input_sample_set._jacobians[:]
                self.total_evals[i+1] += num_new
        #self.disc._io_ptr = None
        #self.disc._io_ptr_local = None
        #pdb.set_trace()

        #def subgrid_ee_chain(
        
   
    def h_refinement(self, props, levels):
        for level in range(self.num_levels):
            if level in levels:
                inds = np.equal(levels, level)
                new_vals = props[inds,:]
                new_sset = sample.sample_set(self.disc._input_sample_set._dim)
                new_sset.set_domain(self.disc._input_sample_set._domain)
                new_sset.set_values(new_vals)        


                num_new_samples = new_vals.shape[0]
                new_sampler = bsam.sampler(self.lb_model_list[level],
                                               error_estimates = self.error_estimates,
                                               jacobians = self.jacobians)
                new_disc = new_sampler.compute_QoI_and_create_discretization(
                    new_sset,
                    savefile=None,
                    globalize=True)
                new_disc._input_sample_set.set_levels(level*np.ones((num_new_samples,), dtype=int))
                #import pdb
                #pdb.set_trace()

                self.disc._input_sample_set.append_sample_set(
                    new_disc._input_sample_set)
                self.disc._output_sample_set.append_sample_set(
                    new_disc._output_sample_set)
                self.total_evals[level] += num_new_samples
                self.total_samples += num_new_samples
        self.disc._input_sample_set.local_to_global()
            

        #return (prob, ee)
