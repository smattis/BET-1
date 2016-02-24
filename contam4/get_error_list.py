#! /usr/bin/env python

# Copyright (C) 2014-2015 The BET Development Team

"""
This example generates uniform samples on a 3D grid
and evaluates a linear map to a 2d space. Probabilities
in the paramter space are calculated using emulated points.
1D and 2D marginals are calculated, smoothed, and plotted.
"""

import numpy as np
import numpy.random as rand
import bet.calculateP as calculateP
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.calculateP.calculateError as calculateError
import bet.calculateP.calculateEventP as calculateEventP
import bet.postProcess.plotP as plotP
from bet.Comm import comm, MPI 
from math import *
import copy

# parameter domain
lam_domain= np.array([[10.0, 50.0],
                      [10.0, 50.0]])
                      
lam_domain_old = copy.copy(lam_domain)

# reference parameters
ref_lam = [30.0*cos(pi/5.0), 30.0*sin(pi/5.0)] #, 0.5]

# box 
box=np.array([[20.0,30.0],[10.0,20.0]])
box = box - lam_domain[:,0]
box = box/(lam_domain[:,1]-lam_domain[:,0])
box = np.array([[.1, .3],[.1,.3]])

inds = [2]

sn = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 12000]
for s in sn:
  lam_domain= np.array([[10.0, 50.0],
                      [10.0, 50.0]])
  samples = np.loadtxt("results/samples.txt")[0:s,:]#[:,inds]
  samples = samples - lam_domain[:,0]
  samples = samples/(lam_domain[:,1]-lam_domain[:,0])
  data = np.loadtxt("results/data.txt")[:][0:s]
  ee = np.loadtxt("results/ee.txt")[:][0:s]
  deriv = np.loadtxt("results/gradients1.txt")[0:s,:]
  #derivatives = np.zeros(deriv.shape)
  #derivatives[:,1] = -deriv[:,0]*np.sin(samples[:,1])*samples[:,0] + deriv[:,1]*np.cos(samples[:,1])*samples[:,0]
  #derivatives[:,0] = deriv[:,0]*np.cos(samples[:,1]) + deriv[:,1]*np.sin(samples[:,1])
  derivatives=deriv
  #import pdb
  #pdb.set_trace()

  derivatives = derivatives*(lam_domain[:,1]-lam_domain[:,0])

  #pdb.set_trace()



  # QoI map
  #Q_map = np.array([[0.503],[0.253]])
  lam_domain = np.array([[0.0,1.0], [0.0,1.0]])
  # reference QoI
  Q_ref =  np.array([0.007565667633600828,0.013768501998185518,0.016950964573704413]) #np.dot(np.array(ref_lam), Q_map)
  Q_ref = Q_ref[inds]





  (d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.uniform_hyperrectangle_binsize(data=data,
                                                                                      Q_ref=Q_ref, 
                                                                                      bin_size = [0.01417567], #bin_ratio = 0.5, #size = [0.01417567], 
                                                                                      center_pts_per_edge = 1)



  (prob_box,total_error) = calculateError.total_error_with_derivatives(samples=samples,
                                                            data=data,
                                                            derivatives=derivatives,
                                                            rho_D_M=d_distr_prob, 
                                                            rho_D_M_samples = d_distr_samples,
                                                            lam_domain = lam_domain,
                                                            event_type = "calculate_error_hyperbox_mc",
                                                            event_args = [box],
                                                            error_estimate = ee,
                                                                       num_emulate = int(5.0e5))

  if comm.rank == 0:
    #print "Box probability", prob_box, total_error
    print s, prob_box, total_error, prob_box + total_error, total_error/(0.158214237783-prob_box)






