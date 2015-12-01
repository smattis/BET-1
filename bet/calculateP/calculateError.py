from bet.Comm import comm, MPI 
import numpy as np
import scipy.spatial as spatial
import bet.util as util
import bet.postProcess.postTools as postTools
import math


                 
def cell_connectivity_exact(samples, io_ptr):
    from scipy.spatial import Delaunay
    from collections import defaultdict
    import itertools
    import numpy.linalg as nlinalg

    tri = Delaunay(samples)
    neiList=defaultdict(set)
    for p in tri.vertices:
        for i,j in itertools.combinations(p,2):
            neiList[i].add(io_ptr[j])
            neiList[j].add(io_ptr[i])
    for i in range(samples.shape[0]):
        neiList[i] = list(set(neiList[i]))
    return neiList

# def cell_connectivity_approx(samples, io_ptr, lam_domain, num_per_circle = 3):
#     l_Tree = spatial.KDTree(samples)
#     Lambda_volume = 1.0
#     for i in range(lam_domain.shape[0]):
#         Lambda_volume *= lam_domain[i][1] - lam_domain[i][0]
#     rad_start = 3.0*(Lambda_volume/float(samples.shape[0]))**(1.0/float(lam_domain.shape[0]))
#     ball_points = l_Tree.query_ball_tree(l_Tree, rad_start, p=2.0)
#     for i in range(samples.shape[0]):
#         sample_ball = samples[ball_points[i],:] - samples[i,:]
#         #coords_square = np.square(sample_ball)
#         #ball_distance = np.empty((sample_ball.shape[0], sample_ball.shape[1]))
#         ball_distance = np.sqrt(np.cumsum(np.square(sample_ball)[:,::-1], axis=1))[::-1]
#         angles = np.empty((sample_ball.shape[0], sample_ball.shape[1]-1))
#         #angles[:,0:-1] = np.arctan2(sample_ball[:,0:-2],ball_distance)
#         #angles[:,-1] = 2.0*np.arctan2(sample_ball[:,-2] + (sample_ball[:,-1]**2 + sample_ball[:,-2]**2)**0.5, ball_distance)
#         angles[:,0:-1] = np.arctan2(sample_ball[:,0:-2],ball_distance[:,1::])
#         angles[:,-1] = 2.0*np.arctan2(sample_ball[:,-2] + (sample_ball[:,-1]**2 + sample_ball[:,-2]**2)**0.5, ball_distance[:,0])
#         angles += np.less(angles, 0.0)*(math.pi+angles)
#         angles = angles**-1.0
        
        
#     pass

def nearest_neighbor_orthant(samples, l_tree, radius):
    ball_points = l_Tree.query_ball_tree(l_Tree, radius, p=2.0)
    base_dict = {}
    #for i in range(samples):
    for i in range(samples.shape[0]):
        sample_ball = np.sign(samples[ball_points[i],:] - samples[i,:])
        sample_ball = sample_ball.astype(int)
        
    return neighbors

def boundary_sets(samples, nei_list, io_ptr):
    from collections import defaultdict
    B_N = defaultdict(list)
    C_N = defaultdict(list)
    for i in range(samples.shape[0]):
        contour_event = io_ptr[i]
        if nei_list[i] == [contour_event]:
            B_N[contour_event].append(i)
        for j in nei_list[i]:
            C_N[j].append(i)
    
    return (B_N, C_N)

class sampling_error(object):
    def __init__(self,
                 samples,
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

                   
        if exact:
            nei_list = cell_connectivity_exact(samples, io_ptr)
        else:
            nei_list = cell_connectivity_approx(samples, io_ptr)
        (self.B_N, self.C_N) = boundary_sets(samples, nei_list, io_ptr)
            
    def calculate_error_all(self):
        upper_bound = 0.0
        lower_bound = 0.0
        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                if i in self.B_N:
                    val1 = np.sum(self.lam_vol[self.B_N[i]])
                else:
                    return (float('nan'), float('nan'))
                val2 = np.sum(self.lam_vol[self.C_N[i]])
                
                term1 = val1/val2 - 1.0
                term2 = val2/val1 - 1.0
                upper_bound += self.rho_D_M[i]*max(term1,term2)
                lower_bound += self.rho_D_M[i]*min(term1,term2)  
        return (upper_bound, lower_bound)

    def calculate_error_fraction(self,P, fraction=0.95):
        #(num_samples, P_high, samples_high, lam_vol_high, data_high)= postTools.sample_highest_prob(top_percentile=fraction, P_samples=P, samples=self.samples, lam_vol=self.lam_vol,sort=True)
        #(P_samples, samples, lam_vol, data, indices) = postTools.sort_by_rho(P_samples = P, samples = self.samples, lam_vol = self.lam_vol)

        (num_samples, P_high, samples_high, lam_vol_high, _ , indices )= postTools.sample_highest_prob(top_percentile=fraction, P_samples=P, samples=self.samples, lam_vol=self.lam_vol,sort=True)

        #import pdb
        #pdb.set_trace()
        lam_vol = np.zeros(self.lam_vol.shape)
        lam_vol[indices] = self.lam_vol[indices]
        #lam_vol[0:num_samples] = lam_vol_high
        # import pdb
        # pdb.set_trace()

        upper_bound = 0.0
        lower_bound = 0.0
        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                e_list = np.equal(self.io_ptr, i)
                E = np.sum(lam_vol[e_list])/np.sum(self.lam_vol[e_list])
                if i in self.B_N:
                    val1 = np.sum(lam_vol[self.B_N[i]])
                    val2 = np.sum(self.lam_vol[self.B_N[i]])
                else:
                    return (float('nan'), float('nan'))
                val3 = np.sum(lam_vol[self.C_N[i]])
                val4 = np.sum(self.lam_vol[self.C_N[i]])

                term1 = val1/val4 - E
                term2 = val3/val2 - E
                upper_bound += self.rho_D_M[i]*max(term1,term2)
                lower_bound += self.rho_D_M[i]*min(term1,term2)  
        return (upper_bound, lower_bound)
        
        

class model_error(object):
    def __init__(self,
                 samples,
                 data,
                 error_estimate,
                 lam_vol,
                 rho_D_M,
                 rho_D_M_samples):
        self.lam_vol = lam_vol
        self.rho_D_M = rho_D_M

        if len(samples.shape) == 1:
            samples = np.expand_dims(samples, axis=1) 
        self.samples = samples

        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)
        if len(error_estimate.shape) == 1:
            error_estimate = np.expand_dims(error_estimate, axis=1)
        if len(rho_D_M_samples.shape) == 1:
            rho_D_M_samples = np.expand_dims(rho_D_M, axis=1)
        d_Tree = spatial.KDTree(rho_D_M_samples)

        # Determine which inputs go to which M bins using the QoI
        (_, self.io_ptr1) = d_Tree.query(data)
        (_, self.io_ptr2) = d_Tree.query(data+error_estimate)

    def calculate_error_all(self):
        er_est = 0.0
        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                ind1 = np.equal(self.io_ptr1, i)
                ind2 = np.equal(self.io_ptr2, i)
                JiA = np.sum(self.lam_vol[ind1])
                Ji = JiA
                JiAe = np.sum(self.lam_vol[ind2]) #np.sum(self.lam_vol[np.logical_and(ind1,ind2)])
                Jie = np.sum(self.lam_vol[ind2])
                er_est += self.rho_D_M[i]*((JiA*Jie - JiAe*Ji)/(Ji*Jie))

        return er_est
        # (num_samples, P_high, samples_high, lam_vol_high, _ , indices )= postTools.sample_highest_prob(top_percentile=fraction, P_samples=P, samples=self.samples, lam_vol=self.lam_vol,sort=True)
        # lam_vol = np.zeros(self.lam_vol.shape)
        # lam_vol[indices] = self.lam_vol[indices]

        # er_est = 0.0
        # for i in range(self.rho_D_M.shape[0]):
        #     if self.rho_D_M[i] > 0.0:
        #         ind1 = np.equal(self.io_ptr1, i)
        #         ind2 = np.equal(self.io_ptr2, i)
        #         JiA = np.sum(lam_vol[ind1])
        #         Ji = np.sum(self.lam_vol[ind1])
        #         JiAe = np.sum(lam_vol[ind2])
        #         Jie = np.sum(self.lam_vol[ind2])
        #         er_est += self.rho_D_M[i]*((JiA*Jie - JiAe*Ji)/(Ji*Jie))

        # return er_est

    def calculate_error_fraction(self,P, fraction=0.95):
        (num_samples, P_high, samples_high, lam_vol_high, _ , indices )= postTools.sample_highest_prob(top_percentile=fraction, P_samples=P, samples=self.samples, lam_vol=self.lam_vol,sort=True)
        lam_vol = np.zeros(self.lam_vol.shape)
        lam_vol[indices] = self.lam_vol[indices]

        er_est = 0.0
        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                ind1 = np.equal(self.io_ptr1, i)
                ind2 = np.equal(self.io_ptr2, i)
                JiA = np.sum(lam_vol[ind1])
                Ji = np.sum(self.lam_vol[ind1])
                JiAe = np.sum(lam_vol[ind2])
                Jie = np.sum(self.lam_vol[ind2])
                er_est += self.rho_D_M[i]*((JiA*Jie - JiAe*Ji)/(Ji*Jie))

        return er_est
