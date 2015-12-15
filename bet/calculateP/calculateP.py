# Copyright (C) 2014-2015 The BET Development Team

r""" 
This module provides methods for calulating the probability measure
:math:`P_{\Lambda}`.

* :mod:`~bet.calculateP.prob_emulated` provides a skeleton class and calculates
    the probability for a set of emulation points.
* :mod:`~bet.calculateP.calculateP.prob_samples_mc` estimates the volumes of
    the voronoi cells using MC integration
"""
from bet.Comm import comm, MPI 
import numpy as np
import scipy.spatial as spatial
import bet.util as util
import numpy.linalg as linalg
from math import *

def emulate_iid_lebesgue(lam_domain, num_l_emulate):
    """
    Parition the parameter space using emulated samples into many voronoi
    cells. These samples are iid so that we can apply the standard MC                                       
    assumuption/approximation

    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:`~numpy.ndarray` of shape (ndim, 2)  
    :param num_l_emulate: The number of emulated samples.
    :type num_l_emulate: int

    :rtype: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)
    :returns: a set of samples for emulation

    """
    num_l_emulate = (num_l_emulate/comm.size) + \
            (comm.rank < num_l_emulate%comm.size)
    lam_width = lam_domain[:, 1] - lam_domain[:, 0]
    lambda_emulate = lam_width*np.random.random((num_l_emulate,
        lam_domain.shape[0]))+lam_domain[:, 0] 
    return lambda_emulate 

def prob_emulated(samples, data, rho_D_M, d_distr_samples,
        lambda_emulate=None, d_Tree=None): 
    r"""

    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{emulate}})`, the
    probability assoicated with a set of voronoi cells defined by
    ``num_l_emulate`` iid samples :math:`(\lambda_{emulate})`.

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (M,) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (M, mdim) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :param lambda_emulate: Samples used to partition the parameter space
    :type lambda_emulate: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)
    :rtype: tuple
    :returns: (P, lambda_emulate, io_ptr, emulate_ptr, lam_vol)

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1) 
    if lambda_emulate is None:
        lambda_emulate = samples
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)
    if d_Tree is None:
        d_Tree = spatial.KDTree(d_distr_samples)
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr) = d_Tree.query(data)
    
    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)
    (_, emulate_ptr) = l_Tree.query(lambda_emulate)
    
    # Calculate Probabilties
    P = np.zeros((lambda_emulate.shape[0],))
    d_distr_emu_ptr = np.zeros(emulate_ptr.shape)
    d_distr_emu_ptr = io_ptr[emulate_ptr]
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(d_distr_emu_ptr, i)
        Itemp_sum = np.sum(Itemp)
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        if Itemp_sum > 0:
            P[Itemp] = rho_D_M[i]/Itemp_sum

    return (P, lambda_emulate, io_ptr, emulate_ptr)

def prob(samples, data, rho_D_M, d_distr_samples, d_Tree=None): 
    r"""
    
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability assoicated with a set of voronoi cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes of these voronoi
    cells are assumed to be equal under the MC assumption.

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (M,) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (M, mdim) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :rtype: tuple of :class:`~numpy.ndarray` of sizes (num_samples,),
        (num_samples,), (ndim, num_l_emulate), (num_samples,), (num_l_emulate,)
    :returns: (P, lam_vol, io_ptr) where P is the
        probability associated with samples, and lam_vol the volumes associated
        with the samples, io_ptr a pointer from data to M bins.

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1) 
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)
    if d_Tree is None:
        d_Tree = spatial.KDTree(d_distr_samples)

    # Set up local arrays for parallelism
    local_index = range(0+comm.rank, samples.shape[0], comm.size)
    samples_local = samples[local_index, :]
    data_local = data[local_index, :]
    local_array = np.array(local_index, dtype='int64')
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr) = d_Tree.query(data_local)

    # Apply the standard MC approximation and
    # calculate probabilities
    P_local = np.zeros((samples_local.shape[0],))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr, i)
        Itemp_sum = np.sum(Itemp)
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        if Itemp_sum > 0:
            P_local[Itemp] = rho_D_M[i]/Itemp_sum 
    P_global = util.get_global_values(P_local)
    global_index = util.get_global_values(local_array)
    P = np.zeros(P_global.shape)
    P[global_index] = P_global[:]

    lam_vol = (1.0/float(samples.shape[0]))*np.ones((samples.shape[0],))

    return (P, lam_vol, io_ptr)

def prob_mc(samples, data, rho_D_M, d_distr_samples,
            lambda_emulate=None, d_Tree=None): 
    r"""
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability assoicated with a set of voronoi cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes of these voronoi
    cells are approximated using MC integration.

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (M,) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (M, mdim) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :param lambda_emulate: Samples used to estimate the volumes of the Voronoi
        cells associated with ``samples``

    :rtype: tuple of :class:`~numpy.ndarray` of sizes (num_samples,),
        (num_samples,), (ndim, num_l_emulate), (num_samples,), (num_l_emulate,)
    :returns: (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr) where P is the
        probability associated with samples, lam_vol the volumes associated
        with the samples, io_ptr a pointer from data to M bins, and emulate_ptr
        a pointer from emulated samples to samples (in parameter space)

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1) 
    if lambda_emulate is None:
        lambda_emulate = samples
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)
    if d_Tree is None:
        d_Tree = spatial.KDTree(d_distr_samples)
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr) = d_Tree.query(data)
    
    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)
    (_, emulate_ptr) = l_Tree.query(lambda_emulate)

    lam_vol, lam_vol_local, local_index = estimate_volume(samples,
            lambda_emulate)

    local_array = np.array(local_index, dtype='int64')
    data_local = data[local_index, :]
    samples_local = samples[local_index, :]
    
    
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr_local) = d_Tree.query(data_local)

    # Calculate Probabilities
    P_local = np.zeros((samples_local.shape[0],))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr_local, i)
        Itemp_sum = np.sum(lam_vol_local[Itemp])
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        if Itemp_sum > 0:
            P_local[Itemp] = rho_D_M[i]*lam_vol_local[Itemp]/Itemp_sum 
    P_global = util.get_global_values(P_local)
    global_index = util.get_global_values(local_array)
    P = np.zeros(P_global.shape)
    P[global_index] = P_global[:]
    return (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr)

def prob_exact(samples, data, rho_D_M, d_distr_samples,
            lam_domain, side_ratio=0.1, d_Tree=None): 
    r"""
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability assoicated with a set of voronoi cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes of these voronoi
    cells are approximated using MC integration.

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (M,) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (M, mdim) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :param lambda_emulate: Samples used to estimate the volumes of the Voronoi
        cells associated with ``samples``

    :rtype: tuple of :class:`~numpy.ndarray` of sizes (num_samples,),
        (num_samples,), (ndim, num_l_emulate), (num_samples,), (num_l_emulate,)
    :returns: (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr) where P is the
        probability associated with samples, lam_vol the volumes associated
        with the samples, io_ptr a pointer from data to M bins, and emulate_ptr
        a pointer from emulated samples to samples (in parameter space)

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1) 
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)
    if d_Tree is None:
        d_Tree = spatial.KDTree(d_distr_samples)
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr) = d_Tree.query(data)
    
    # Determine which emulated samples match with which model run samples
    #l_Tree = spatial.KDTree(samples)
    #(_, emulate_ptr) = l_Tree.query(lambda_emulate)

    (lam_vol, lam_vol_local, local_index) = exact_volume(samples, lam_domain, side_ratio)

    # local_array = np.array(local_index, dtype='int64')
    # data_local = data[local_index, :]
    # samples_local = samples[local_index, :]
    
    
    # # Determine which inputs go to which M bins using the QoI
    # (_, io_ptr_local) = d_Tree.query(data_local)

    # # Calculate Probabilities
    # P = np.zeros((samples.shape[0],))
    # for i in range(rho_D_M.shape[0]):
    #     Itemp = np.equal(io_ptr, i)
    #     Itemp_sum = np.sum(lam_vol[Itemp])
    #     #Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
    #     if Itemp_sum > 0:
    #         P[Itemp] = rho_D_M[i]*lam_vol[Itemp]/Itemp_sum 
    # #P_global = util.get_global_values(P_local)
    # #global_index = util.get_global_values(local_array)
    # #P = np.zeros(P_global.shape)
    # #P[global_index] = P_global[:]

    local_array = np.array(local_index, dtype='int64')
    data_local = data[local_index, :]
    samples_local = samples[local_index, :]
    
    
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr_local) = d_Tree.query(data_local)

    # Calculate Probabilities
    P_local = np.zeros((samples_local.shape[0],))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr_local, i)
        Itemp_sum = np.sum(lam_vol_local[Itemp])
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        if Itemp_sum > 0:
            P_local[Itemp] = rho_D_M[i]*lam_vol_local[Itemp]/Itemp_sum 
    P_global = util.get_global_values(P_local)
    global_index = util.get_global_values(local_array)
    P = np.zeros(P_global.shape)
    P[global_index] = P_global[:]
    
    return (P, lam_vol, io_ptr)

def estimate_volume(samples, lambda_emulate=None):
    r"""
    Estimate the volume fraction of the Voronoi cells associated with
    ``samples`` using ``lambda_emulate`` as samples for Monte Carlo
    integration. Specifically we are estimating 
    :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.
    
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param lambda_emulate: Samples used to partition the parameter space
    :type lambda_emulate: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)

    :rtype: tuple
    :returns: (lam_vol, lam_vol_local, local_index) where ``lam_vol`` is the
        global array of volume fractions, ``lam_vol_local`` is the local array
        of volume fractions, and ``local_index`` a list of the global indices
        for local arrays on this particular processor ``lam_vol_local =
        lam_vol[local_index]``
    
    """

    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if lambda_emulate is None:
        lambda_emulate = samples
 
    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)
    (_, emulate_ptr) = l_Tree.query(lambda_emulate)

    # Apply the standard MC approximation to determine the number of emulated
    # samples per model run sample. This is for approximating 
    # \mu_Lambda(A_i \intersect b_j)
    lam_vol = np.zeros((samples.shape[0],)) 
    for i in range(samples.shape[0]):
        lam_vol[i] = np.sum(np.equal(emulate_ptr, i))
    clam_vol = np.copy(lam_vol) 
    comm.Allreduce([lam_vol, MPI.DOUBLE], [clam_vol, MPI.DOUBLE], op=MPI.SUM)
    lam_vol = clam_vol
    num_emulated = lambda_emulate.shape[0]
    num_emulated = comm.allreduce(num_emulated, op=MPI.SUM)
    lam_vol = lam_vol/(num_emulated)

    # Set up local arrays for parallelism
    local_index = range(0+comm.rank, samples.shape[0], comm.size)
    lam_vol_local = lam_vol[local_index]

    return (lam_vol, lam_vol_local, local_index)


    
def exact_volume(samples, lam_domain, side_ratio = 0.1):
    r"""
    Estimate the volume fraction of the Voronoi cells associated with
    ``samples`` using ``lambda_emulate`` as samples for Monte Carlo
    integration. Specifically we are estimating 
    :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.
    
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param lambda_emulate: Samples used to partition the parameter space
    :type lambda_emulate: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)

    :rtype: tuple
    :returns: (lam_vol, lam_vol_local, local_index) where ``lam_vol`` is the
        global array of volume fractions, ``lam_vol_local`` is the local array
        of volume fractions, and ``local_index`` a list of the global indices
        for local arrays on this particular processor ``lam_vol_local =
        lam_vol[local_index]``
    
    """
    (n_samp, ndim) = samples.shape
    if ndim == 1:
        add_points = np.zeros((2,0))
        val = np.min(samples)
        add_points[0] = -val 
        val = np.max(samples)
        add_points[1] = 2.0*lam_domain[0][1] - val 
        samples = np.vstack((samples,add_points))
    elif ndim == 2:
        add_points = np.less(samples[:,0], lam_domain[0][0]+side_ratio*(lam_domain[0][1] - lam_domain[0][0]))
        points_new = samples[add_points,:]
        points_new[:,0] = lam_domain[0][0] - (points_new[:,0]-lam_domain[0][0])
        samples = np.vstack((samples, points_new))


        add_points = np.greater(samples[:,0], lam_domain[0][1]-side_ratio*(lam_domain[0][1] - lam_domain[0][0]))
        points_new = samples[add_points,:]
        points_new[:,0] = lam_domain[0][1] + (-points_new[:,0]+lam_domain[0][1])
        samples = np.vstack((samples, points_new))

        add_points = np.less(samples[:,1], lam_domain[1][0]+side_ratio*(lam_domain[1][1] - lam_domain[1][0]))
        points_new = samples[add_points,:]
        points_new[:,1] = lam_domain[1][0] - (points_new[:,1]-lam_domain[1][0])
        samples = np.vstack((samples, points_new))


        add_points = np.greater(samples[:,1], lam_domain[1][1]-side_ratio*(lam_domain[1][1] - lam_domain[1][0]))
        points_new = samples[add_points,:]
        points_new[:,1] = lam_domain[1][1] + (-points_new[:,1]+lam_domain[1][1])
        samples = np.vstack((samples, points_new))

        # add_points = np.linspace(lam_domain[0][0], lam_domain[0][1], points_per_edge[0])
        # add_points1 = np.zeros((len(add_points),2))
        # add_points1[:,0] = add_points
        # add_points1[:,1] = lam_domain[1][0]
        # samples = np.vstack((samples,add_points1))
        
        # add_points1 = np.zeros((len(add_points),2))
        # add_points1[:,0] = add_points
        # add_points1[:,1] = lam_domain[1][1]
        # samples = np.vstack((samples,add_points1))

        # add_points = np.linspace(lam_domain[1][0], lam_domain[1][1], points_per_edge[1])
        # add_points1 = np.zeros((len(add_points),2))
        # add_points1[:,0] = add_points
        # add_points1[:,1] = lam_domain[0][0]
        # samples = np.vstack((samples,add_points1))
        
        # add_points1 = np.zeros((len(add_points),2))
        # add_points1[:,0] = add_points
        # add_points1[:,1] = lam_domain[0][1]
        # samples = np.vstack((samples,add_points1))
    else:
        exit(1)
        
    # new_list = []
    # ind_set = set(range(ndim))
    # for i in range(ndim):
    #     x = [np.linspace(lam_domain[i][0], lam_domain[i][1],10)]
    #     for j in list(ind_set-i):
    #         x.append(lam_domain[j,:])
    #     new_samples = 
    vor = spatial.Voronoi(samples)
    # ndim = samples.shape[1]
    #import pdb
    #pdb.set_trace()
    local_index = range(0+comm.rank, n_samp, comm.size)
    local_array = np.array(local_index, dtype='int64')
    lam_vol_local = np.zeros(local_array.shape)


    for I,i in enumerate(local_index):
        val =vor.point_region[i]
        region = vor.regions[val]
        if not -1 in region:
            polygon = [vor.vertices[k] for k in region]
            #import pdb
            #pdb.set_trace()
            delan = spatial.Delaunay(polygon)
            simplices = delan.points[delan.simplices]
            vol = 0.0
            for j in range(simplices.shape[0]):
                mat = np.empty((ndim,ndim))
                mat[:,:] = (simplices[j][1::,:] - simplices[j][0,:]).transpose()
                #import pdb
                #pdb.set_trace()
                vol += abs(1.0/factorial(ndim)*linalg.det(mat))
            #if vol > 1.0:
                #import pdb
                #pdb.set_trace()
            lam_vol_local[I] = vol
    #import pdb
    #pdb.set_trace()
    lam_size = np.prod(lam_domain[:,1] - lam_domain[:,0])
    #import pdb
    #pdb.set_trace()
    lam_vol_local  = lam_vol_local/lam_size
    lam_vol_global = util.get_global_values(lam_vol_local)
    global_index = util.get_global_values(local_array)
    lam_vol = np.zeros(lam_vol_global.shape)
    lam_vol[global_index] = lam_vol_global[:]
    return (lam_vol, lam_vol_local, local_index)

def prob_hyperbox(box, samples, P, lam_vol, lam_domain, num_l_emulate):
    
    lambda_emulate = emulate_iid_lebesgue(lam_domain, num_l_emulate)
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    rho = P/lam_vol
 
    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)
    (_, emulate_ptr) = l_Tree.query(lambda_emulate)
    
    in_A = np.logical_and(np.greater_equal(lambda_emulate,box[:,0]), np.less_equal(lambda_emulate,box[:,1]))
    in_A = np.all(in_A, axis=1)
    sum1 = np.sum(rho[emulate_ptr[in_A]])
    print comm.rank, sum1
    sum1_all = comm.allreduce(sum1, op=MPI.SUM)
    print sum1_all
    prob = float(sum1_all)/float(num_l_emulate)
    
    return prob

