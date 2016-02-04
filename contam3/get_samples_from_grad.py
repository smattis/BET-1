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
                      [0.0, pi/2.0]])
                      
lam_domain_old = copy.copy(lam_domain)

# reference parameters
ref_lam = [30.0, pi/5.0] #, 0.5]

# box 
box=np.array([[0.2,0.5],[0.3,0.5]])


inds = [2]
samples = np.loadtxt("results_grad/coarse/samples.txt")[0:1000,:]#[:,inds]
samples = samples - lam_domain[:,0]
samples = samples/(lam_domain[:,1]-lam_domain[:,0])
data = np.loadtxt("results_grad/finer/data.txt")[:,inds][0:1000,:]
ee = np.loadtxt("results_grad/finer/ee.txt")[:,inds][0:0000,:]
derivatives = np.loadtxt("results_grad/finer/gradients3.txt")[0:1000,:]




# QoI map
#Q_map = np.array([[0.503],[0.253]])
lam_domain = np.array([[0.0,1.0], [0.0,1.0]])
# reference QoI
Q_ref =  np.array([0.007565667633600828,0.013768501998185518,0.016950964573704413]) #np.dot(np.array(ref_lam), Q_map)
Q_ref = Q_ref[inds]





(d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.uniform_hyperrectangle_binsize(data=data,
                                                                                    Q_ref=Q_ref, 
                                                                                    bin_size = [0.01417567], 
                                                                                    center_pts_per_edge = 1)

# se = calculateError.sampling_error(samples, lam_vol, 
#                                    rho_D_M=d_distr_prob, rho_D_M_samples = d_distr_samples, data=data)
#samples_new = se.get_new_samples(lam_domain = lam_domain, num_l_emulate=10000, index =1)


# (P, lam_vol, io_ptr) = calculateP.prob_exact(samples, data, d_distr_prob, d_distr_samples, lam_domain)
print "here"
(samples_all, data_all, error_estimate_all) = calculateError.refine_with_derivatives(samples=samples,
                                                                                     data=data,
                                                                                     derivatives=derivatives,
                                                                                     rho_D_M=d_distr_prob, 
                                                                                     rho_D_M_samples = d_distr_samples,
                                                                                     lam_domain = lam_domain,
                                                                                     event_type = "calculate_error_hyperbox",
                                                                                     event_args = [lam_domain, box, int(1.0e3)],
                                                                                     tol = 1.0e-1,
                                                                                     error_estimate=ee)


# p_event = calculateEventP.prob_event(samples = samples,
#                                      lam_domain = lam_domain,
#                                      P = P,
#                                      lam_vol = lam_vol, rho_D_M=d_distr_prob, rho_D_M_samples = d_distr_samples, data=data)
# prob_box = p_event.calculate_prob_hyperbox(box=box, 
#                                            num_l_emulate=int(1.0e6))
                                     
# if comm.rank == 0:
#   print prob_box
# se = calculateError.sampling_error(samples, lam_vol, 
#                                    rho_D_M=d_distr_prob, rho_D_M_samples = d_distr_samples, data=data)
#   #(h,l) = se.calculate_error_fraction(P, 1.0)
#   #(h,l) = se.calculate_error_contour_events()
#   #(h,l) = se.calculate_error_contour_events()
#   #(h,l) = se.calculate_error_voronoi(lam_domain, samples_true, marker_true, int(1.0e5))
# (h,l) = se.calculate_error_hyperbox(lam_domain, box, int(1.0E6))
# if comm.rank ==0:
#   print h,l
# #samples_new = se.get_new_samples(lam_domain = lam_domain, num_l_emulate=10000, index =1)
# #samples_new = (lam_domain_old[:,1] - lam_domain_old[:,0]) * samples_new + lam_domain_old[:,0]
# #np.savetxt("samples_new.txt", samples_new)

# #   #ee = 0.01*np.ones(data.shape)
# me = calculateError.model_error(samples,
#                                 data,
#                                 error_estimate = ee,
#                                 lam_vol = lam_vol,
#                                 rho_D_M = d_distr_prob,
#                                 rho_D_M_samples = d_distr_samples)
# # m = me.calculate_error_contour_events()
# m = me.calculate_error_hyperbox(lam_domain = lam_domain,
#                                 box = box,
#                                 num_l_emulate = int(1.0e6))
# if comm.rank ==0:
#   print m

# #   m = me. calculate_error_fraction(P, 1.0)
# #   #m = me. calculate_error_all()

# #   print m

# # import pdb
# # pdb.set_trace()


# import matplotlib.pyplot as plt
# from scipy.spatial import Voronoi, voronoi_plot_2d
# vor = Voronoi(samples)
# fig = plt.figure()
# plt.axis([0, 1, 0, 1])
# plt.hold(True)
# #plt.plot(vor.vertices[:,0], vor.vertices[:, 1], 'ko', ms=1)
# # for vpair in vor.ridge_vertices:
# #     if vpair[0] >= 0 and vpair[1] >= 0:
# #         v0 = vor.vertices[vpair[0]]
# #         v1 = vor.vertices[vpair[1]]
# #         # Draw a line from v0 to v1.
# #         plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=0.1)
# # import pdb
# # pdb.set_trace()
# for i,val in enumerate(vor.point_region):
#   region = vor.regions[val]
#   if not -1 in region:
#     if P[i] > 0:
#       polygon = [vor.vertices[i] for i in region]
#       #kw = {'color':'r', 'edgecolor': 'r'}
#       z = zip(*polygon)
#       plt.fill(z[0], z[1],color='r' , edgecolor = 'k', linewidth = 0.005)
# #for i in range(samples.shape[0]):
# #  if P[i] > 0.0:
# #    plt.fill(samples[i,0], samples[i,1], 'r')
# #plt.show()

# #plt.show()
# plt.savefig("coarse1.eps")


