from bet.Comm import comm, MPI 
import numpy as np
import scipy.spatial as spatial
import bet.util as util
import bet.calculateP.calculateP as calculateP
import bet.postProcess.postTools as postTools
import math

class prob_event(object):
    def __init__(self,
                 samples,
                 lam_domain,
                 P,
                 lam_vol,
                 rho_D_M,
                 rho_D_M_samples = None,
                 io_ptr= None,
                 data = None,
                 exact = True):
        self.lam_vol = lam_vol
        self.rho_D_M = rho_D_M
        
        if io_ptr == None:
            if len(samples.shape) == 1:
                samples = np.expand_dims(samples, axis=1) 
            if data != None:
                if len(data.shape) == 1:
                    data = np.expand_dims(data, axis=1) 

            if len(rho_D_M_samples.shape) == 1:
                rho_D_M_samples = np.expand_dims(rho_D_M_samples, axis=1)
            #import pdb
            #pdb.set_trace()
            d_Tree = spatial.KDTree(rho_D_M_samples)
        
            # Determine which inputs go to which M bins using the QoI
            (_, io_ptr) = d_Tree.query(data)
        self.io_ptr = io_ptr

        self.samples = samples
        self.P = P
        self.lam_domain = lam_domain

    def calculate_prob_hyperbox(self, box, num_l_emulate):

        lambda_emulate = calculateP.emulate_iid_lebesgue(self.lam_domain, num_l_emulate)
        # if len(self.samples.shape) == 1:
        #     samples = np.expand_dims(samples, axis=1)
            
        rho = self.P/self.lam_vol

        # Determine which emulated samples match with which model run samples
        l_Tree = spatial.KDTree(self.samples)
        (_, emulate_ptr) = l_Tree.query(lambda_emulate)

        in_A = np.logical_and(np.greater_equal(lambda_emulate,box[:,0]), np.less_equal(lambda_emulate,box[:,1]))
        in_A = np.all(in_A, axis=1)
        sum1 = np.sum(rho[emulate_ptr[in_A]])
        #print comm.rank, sum1
        sum1_all = comm.allreduce(sum1, op=MPI.SUM)
        #print sum1_all
        prob = float(sum1_all)/float(num_l_emulate)

        return prob


    def calculate_prob_voronoi(self, samples_A, id_A, num_l_emulate):
        lambda_emulate = calculateP.emulate_iid_lebesgue(self.lam_domain, num_l_emulate)
        l_tree1 = spatial.KDTree(self.samples)
        l_tree2 = spatial.KDTree(samples_A)
        
        ptr1 = l_tree1.query(lambda_emulate)[1]
        ptr2 = l_tree2.query(lambda_emulate)[1]
        #import pdb
        #pdb.set_trace()
        in_A = id_A[ptr2]

        prob = 0.0

        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                indices = np.equal(self.io_ptr,i)
                in_Ai = indices[ptr1]
                sum1 = np.sum(np.logical_and(in_A, in_Ai))
                sum2 = np.sum(in_Ai)
                sum1 = comm.allreduce(sum1, op=MPI.SUM)
                sum2 = comm.allreduce(sum2, op=MPI.SUM)
                prob  += (float(sum1)/float(sum2))*rho_D_M[i]
                #E = float(np.sum(np.logical_and(in_A, in_Ai)))/(np.sum(in_Ai))
                
               
        return prob 
