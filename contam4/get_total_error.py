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
samples = np.loadtxt("results/samples.txt")[0:12000,:]#[:,inds]
samples = samples - lam_domain[:,0]
samples = samples/(lam_domain[:,1]-lam_domain[:,0])
data = np.loadtxt("results/data.txt")[:][0:12000]
ee = np.loadtxt("results/ee.txt")[:][0:12000]
deriv = np.loadtxt("results/gradients1.txt")[0:12000,:]
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

# se = calculateError.sampling_error(samples, lam_vol, 
#                                    rho_D_M=d_distr_prob, rho_D_M_samples = d_distr_samples, data=data)
#samples_new = se.get_new_samples(lam_domain = lam_domain, num_l_emulate=10000, index =1)


# (P, lam_vol, io_ptr) = calculateP.prob_exact(samples, data, d_distr_prob, d_distr_samples, lam_domain)
# (lam_vol,_,_) = calculateP.exact_volume(samples=samples,
#                                                lam_domain=lam_domain)
# me = calculateError.model_error(samples,
#                                 data,
#                                 error_estimate = ee,
#                                 lam_vol = lam_vol,
#                                 rho_D_M = d_distr_prob,
#                                 rho_D_M_samples = d_distr_samples)

# m = me.calculate_error_hyperbox(lam_domain = lam_domain,
#                                 box = box,
#                                 num_l_emulate = int(1.0e4))
# if comm.rank ==0:
#   print "model error ", m


# (samples_all, data_all, lam_vol_all, error_estimate_all) = calculateError.refine_with_derivatives(samples=samples,
#                                                                                      data=data,
#                                                                                      derivatives=derivatives,
#                                                                                      rho_D_M=d_distr_prob, 
#                                                                                      rho_D_M_samples = d_distr_samples,
#                                                                                      lam_domain = lam_domain,
#                                                                                      event_type = "calculate_error_hyperbox",
#                                                                                      event_args = [lam_domain, box, int(1.0e5)],
#                                                                                      tol = 2.0e-2,
#                                                                                      error_estimate=ee)

# (P, lam_vol, io_ptr) = calculateP.prob_exact(samples, data, d_distr_prob, d_distr_samples, lam_domain)

# p_event = calculateEventP.prob_event(samples = samples,
#                                      lam_domain = lam_domain,
#                                      P = P,
#                                      lam_vol = lam_vol, rho_D_M=d_distr_prob, rho_D_M_samples = d_distr_samples, data=data)
# prob_box = p_event.calculate_prob_hyperbox(box=box, 
#                                            num_l_emulate=int(1.0e5))

(prob_box,total_error) = calculateError.total_error_with_derivatives(samples=samples,
                                                          data=data,
                                                          derivatives=derivatives,
                                                          rho_D_M=d_distr_prob, 
                                                          rho_D_M_samples = d_distr_samples,
                                                          lam_domain = lam_domain,
                                                          event_type = "calculate_error_hyperbox_mc",
                                                          event_args = [box],
                                                          error_estimate = ee,
                                                          num_emulate = int(1.0e5))
                                     
if comm.rank == 0:
  print "Box probability", prob_box, total_error





colorP = data_all/np.max(data_all)
#pdb.set_trace()
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(samples_all)
fig = plt.figure()
plt.axis([0, 1, 0, 1])
plt.hold(True)
#plt.plot(vor.vertices[:,0], vor.vertices[:, 1], 'ko', ms=1)
# for vpair in vor.ridge_vertices:
#     if vpair[0] >= 0 and vpair[1] >= 0:
#         v0 = vor.vertices[vpair[0]]
#         v1 = vor.vertices[vpair[1]]
#         # Draw a line from v0 to v1.
#         plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=0.1)
# import pdb
# pdb.set_trace()
for i,val in enumerate(vor.point_region):
  region = vor.regions[val]
  #print i
  if not -1 in region:
    if P[i] > 0.0:
      #print i, P[i], colorP[i], colorP.shape
      polygon = [vor.vertices[I] for I in region]
      #kw = {'color':'r', 'edgecolor': 'r'}
      #print colorP[i]
      col_string = '%f' % colorP[i]
      #print col_string
      z = zip(*polygon)
      #plt.fill(z[0], z[1],color=col_string , edgecolor = col_string, linewidth = 0.005)
      plt.fill(z[0], z[1],color='r' , edgecolor = col_string, linewidth = 0.005)
#for i in range(samples.shape[0]):
#  if P[i] > 0.0:
#    plt.fill(samples[i,0], samples[i,1], 'r')
#plt.show()

plt.show()
plt.savefig("coarse2.pdf")


