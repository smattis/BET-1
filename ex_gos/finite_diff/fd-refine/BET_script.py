#! /usr/bin/env python

# Copyright (C) 2016 Steven Mattis

r"""

This example generates samples for a 1-1 nonlinear map defined by
solving a 2D linear system that depends nonlinearly on the input parameter
and taking the average of the solution vector. The so-called exact
solution is solved  directly numerically inverting the system at many
samples. The so-called numerical solution is calculated by solving the 
system with a small number of iterations of the Gauss-Seidel method.

"""

import numpy as np
from tabulate import tabulate
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calcP
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import bet.surrogates as surrogates
from lbModel import lb_model, lb_model2, lb_model3
import bet.sampling.goalOrientedSampling as gos

import plot1D as p1


# Interface BET to the model.

# Define paramter set of interest A
s_set = samp.rectangle_sample_set(1)
s_set.setup(maxes=[[0.25]], mins=[[-0.25]])
s_set.set_region(np.array([0,1]))

sampler = gos.sampler(1000, 10, 1.0e-5, [lb_model, lb_model2, lb_model3], s_set, 0, error_estimates=True, jacobians=True)

#sampler_ref = bsam.sampler(lb_model_exact, error_estimates=False, jacobians=False)

# Initialize input parameter sample set object
input_samples = samp.sample_set(1)
#input_samples_ref = samp.sample_set(1)

# Set parameter domain
domain = np.array([[-0.5, 1.0]])
input_samples.set_domain(domain)
#input_samples_ref.set_domain(domain)

# Generate samples on the parameter space
#input_samples = sampler.random_sample_set('random', input_samples, num_samples=1000)
#input_samples_ref = sampler_ref.random_sample_set('random', input_samples_ref, num_samples=10000)

#my_discretization = sampler.initial_samples_regular(input_samples, num_samples_per_dim=[20])
my_discretization = sampler.initial_samples_random('r',
                                                   input_samples,
                                                   10,
                                                   level=0)

#p1.plot1D_p1(sampler.disc)
#import pdb
#pdb.set_trace()

#MC_assumption = True
# Estimate volumes of Voronoi cells associated with the parameter samples
# if MC_assumption is False:
#     input_samples.estimate_volume(n_mc_points=1E5)
#     input_samples_ref.estimate_volume(n_mc_points=1E5)
# else:
#     input_samples.estimate_volume_mc()
#     input_samples_ref.estimate_volume_mc()


# Create the discretization object using the input samples
#my_discretization = sampler.compute_QoI_and_create_discretization(
#    input_samples, savefile='linear.txt.gz')
#my_discretization_ref = sampler_ref.compute_QoI_and_create_discretization(
#    input_samples_ref, savefile='linear_ref.txt.gz')

# Define the reference parameter and compute the reference QoI
param_ref = [0.0]
Q_ref = lb_model3(np.array([param_ref]))[0]
#import pdb
#pdb.set_trace()
my_discretization._input_sample_set.set_reference_value(np.array(param_ref))
my_discretization._output_sample_set.set_reference_value(np.array(Q_ref[0,:]))
#my_discretization_ref._input_sample_set.set_reference_value(np.array(param_ref))
#my_discretization_ref._output_sample_set.set_reference_value(np.array(Q_ref[0,:]))

# Save the discretizations
samp.save_discretization(my_discretization, "control")
#samp.save_discretization(my_discretization_ref, "control_ref")

# Define the output probababilty for reference
rect_domain = np.array([[0.12, 0.15]])
simpleFunP.regular_partition_uniform_distribution_rectangle_domain(my_discretization, rect_domain=rect_domain)


inputs = np.arange(100)
outputs = np.zeros((100,), dtype='int')
#(inputs, outputs) = sampler.pseudoinverse_samples(inputs, outputs, 1)

import matplotlib.pyplot as plt
#plt.figure(0)
#plt.scatter(my_discretization._input_sample_set._values[0:100,:],
#            my_discretization._output_sample_set._values[0:100,:])


# num_old = 100
# for i in range(0):
#     num = my_discretization.check_nums()
#     #inputs = np.arange(num_old, num)
#     #outputs = np.zeros((num-num_old,),dtype='int')
#     (inputs, outputs) = sampler.pseudoinverse_samples(inputs, outputs)
    
#     plt.figure(i)
#     plt.scatter(my_discretization._input_sample_set._values[num_old:num,:],
#             my_discretization._output_sample_set._values[num_old:num,:])
#     num_old = num
#plt.show()

#sampler.level_refinement_inds(np.array([5, 50]))
# disc = my_discretization
# emulated_inputs = bsam.random_sample_set('r',
#                                          disc._input_sample_set._domain,
#                                          num_samples = int(1E4),
#                                          globalize=False)

#(prob, ee) = sampler.evaluate_surrogate(emulated_inputs, order=1)

# emulated_inputs = bsam.random_sample_set('r',
#                                          sampler.disc._input_sample_set._domain,
#                                          num_samples = int(1E5),
#                                          globalize=False)
for i in range(5):
    #disc = my_discretization
    emulated_inputs = bsam.random_sample_set('r',
                                         sampler.disc._input_sample_set._domain,
                                         num_samples = int(1E5),
                                         globalize=False)
    (prob, ee) = sampler.h_refinement_cluster(emulated_inputs, 1,
                                              15,1)
    print (prob, ee ,prob[0]-ee[0])
    #plt.scatter(my_discretization._input_sample_set._values[-5::,:],
    #            my_discretization._output_sample_set._values[-5::,:])
    # import pdb
    # pdb.set_trace()
    # plt.show()
    (prob, ee) = sampler.level_refinement(emulated_inputs, 1,
                                        10)
    p1.plot1D_p1(sampler.disc)
    
    print (prob, ee, prob[0]-ee[0])
    
p1.plot1D_p1(sampler.disc)
import pdb
pdb.set_trace()

# # Solve stochastic inverse problem for reference
# calcP.prob(my_discretization_ref)

# # Define paramter set of interest A
# s_set = samp.rectangle_sample_set(1)
# s_set.setup(maxes=[[0.25]], mins=[[-0.25]])
# s_set.set_region(np.array([0,1]))

# # Calculate reference probability for A
# P_ref = calcP.prob_from_discretization_input(my_discretization_ref, s_set)
# print P_ref[0]

# # Rename for simplicity
# disc = my_discretization

# # numbers of samples for which to calculate
# nList=[5, 10, 25, 50, 100, 250, 500, 1000]
# headers=['num of samples', 'P_N(A)', 'hat{P_N(A)}$', 'P_{tilde{N}}(A)' ,'hat{P_{tilde{N}}(A)}$']
# tList=[]

# # Loop over numbers of samples
# for n in nList:
#     sList=[n]
#     # Clip discretiztion for number of samples
#     disc_new=disc.clip(n)
#     # Define output probability set
#     simpleFunP.regular_partition_uniform_distribution_rectangle_domain(disc_new, rect_domain=rect_domain)
    
#     # Define emulated inputs
#     emulated_inputs = bsam.random_sample_set('r',
#                                              disc_new._input_sample_set._domain,
#                                              num_samples = int(1E5),
#                                              globalize=False)
#     disc_new.set_emulated_input_sample_set(emulated_inputs)

#     # Make copies for different surrogates and corrections
#     disc_no_surrogate = disc_new.copy()
#     disc_no_surrogate_corrected = disc_new.copy()
#     disc_surrogate = disc_new.copy()

#     # Solve stochastic inverse problem and compute probability for region of
#     # interest for different surrogates and corrections

#     # No surrogate, no correction
#     calcP.prob_on_emulated_samples(disc_no_surrogate, globalize=False)
#     P = calcP.prob_from_discretization_input(disc_no_surrogate, s_set)
#     sList.append(P[0])
#     print P[0]
    
#     # No surrogate, corrected
#     disc_no_surrogate_corrected._output_sample_set._values += disc_no_surrogate_corrected._output_sample_set._error_estimates
#     disc_no_surrogate_corrected._output_sample_set._values_local += disc_no_surrogate_corrected._output_sample_set._error_estimates_local
#     calcP.prob_on_emulated_samples(disc_no_surrogate_corrected, globalize=False)
#     P = calcP.prob_from_discretization_input(disc_no_surrogate_corrected, s_set)
#     sList.append(P[0])
#     print P[0]

#     # Surrogate
#     sur = surrogates.piecewise_polynomial_surrogate(disc_surrogate)
#     sur_disc_lin = sur.generate_for_input_set(emulated_inputs, order=1)
#     # No correction and error (subtract error to get corrected)
#     (Pt, error) = sur.calculate_prob_for_sample_set_region(s_set, 
#                                                          regions=[0])
#     print Pt[0]
#     print Pt[0] - error[0]
#     sList.append(Pt[0])
#     sList.append(Pt[0] - error[0])

#     tList.append(sList)
# print "True P(A): ", P_ref[0]
# # print latex table
# print tabulate(tList, headers=headers, tablefmt="latex", floatfmt="0.5e")
