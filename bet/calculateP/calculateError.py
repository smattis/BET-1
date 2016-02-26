from bet.Comm import comm, MPI 
import numpy as np
import scipy.spatial as spatial
import bet.util as util
import bet.calculateP.calculateP as calculateP
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

def surrogate_from_derivatives(samples, 
                               samples_surrogate, 
                               data,
                               derivatives,
                               error_estimate = None):
    l_tree = spatial.KDTree(samples)
    ptr = l_tree.query(samples_surrogate)[1]
    #import pdb
    #pdb.set_trace()
    data_surrogate = np.sum(derivatives[ptr,:]*(samples_surrogate - samples[ptr]),axis=1)
    if len(data_surrogate.shape) == 1:
        data_surrogate = np.expand_dims(data_surrogate, axis=1)
    #import pdb
    #pdb.set_trace()
    data_surrogate += data[ptr]
    if error_estimate != None:
        error_estimate_surrogate = error_estimate[ptr]
    else:
        error_estimate_surrogate = None
    #pdb.set_trace()
    return (data_surrogate, error_estimate_surrogate, ptr)
    

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

    def calculate_error_contour_events(self):
        up_list = []
        low_list = []
    
        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                lam_vol = np.zeros(self.lam_vol.shape)
                indices = np.equal(self.io_ptr,i)
                lam_vol[indices] = self.lam_vol[indices]
                if i in self.B_N:
                    val1 = np.sum(self.lam_vol[self.B_N[i]])
                    val2 = np.sum(lam_vol[self.B_N[i]])
                
                    val3 = np.sum(self.lam_vol[self.C_N[i]])
                    val4 = np.sum(lam_vol[self.C_N[i]])
                
                    term1 = val2/val3 - 1.0
                    term2 = val4/val1 - 1.0
                    up_list.append(self.rho_D_M[i]*max(term1,term2))
                    low_list.append(self.rho_D_M[i]*min(term1,term2))
                else:
                    up_list.append(float('nan'))
                    low_list.append(float('nan'))
                
            else:
                up_list.append(0.0)
                low_list.append(0.0)
        return (up_list, low_list)
        

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
                #print val1/val4, val3/val2
                term1 = val1/val4 - E
                term2 = val3/val2 - E
                upper_bound += self.rho_D_M[i]*max(term1,term2)
                lower_bound += self.rho_D_M[i]*min(term1,term2)  
        return (upper_bound, lower_bound)
    
    def calculate_error_voronoi(self, lam_domain, samples_A, id_A, num_l_emulate):
        lambda_emulate = calculateP.emulate_iid_lebesgue(lam_domain, num_l_emulate)
        l_tree1 = spatial.KDTree(self.samples)
        l_tree2 = spatial.KDTree(samples_A)
        
        ptr1 = l_tree1.query(lambda_emulate)[1]
        ptr2 = l_tree2.query(lambda_emulate)[1]
        #import pdb
        #pdb.set_trace()
        in_A = id_A[ptr2]

        upper_bound = 0.0
        lower_bound = 0.0

        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                indices = np.equal(self.io_ptr,i)
                in_Ai = indices[ptr1]
                sum1 = np.sum(np.logical_and(in_A, in_Ai))
                sum2 = np.sum(in_Ai)
                sum1 = comm.allreduce(sum1, op=MPI.SUM)
                sum2 = comm.allreduce(sum2, op=MPI.SUM)
                E = float(sum1)/float(sum2)
                #E = float(np.sum(np.logical_and(in_A, in_Ai)))/(np.sum(in_Ai))
                
                #import pdb
                # pdb.set_trace()
                # should be faster way
                in_B_N = np.zeros(in_A.shape, dtype=np.bool)
                for j in self.B_N[i]:
                    in_B_N = np.logical_or(np.equal(ptr1,j),in_B_N)

                in_C_N = np.zeros(in_A.shape, dtype=np.bool)
                for j in self.C_N[i]:
                    in_C_N =  np.logical_or(np.equal(ptr1,j), in_C_N)
                #pdb.set_trace()
                sum3 = np.sum(np.logical_and(in_A,in_B_N))
                sum4 = np.sum(in_C_N)
                sum3 = comm.allreduce(sum3, op=MPI.SUM)
                sum4 = comm.allreduce(sum4, op=MPI.SUM)
                term1 = float(sum3)/float(sum4) - E
                #term1 = float(np.sum(np.logical_and(in_A,in_B_N)))/float(np.sum(in_C_N))- E
                sum5 = np.sum(np.logical_and(in_A,in_C_N))
                sum6 = np.sum(in_B_N)
                sum5 = comm.allreduce(sum5, op=MPI.SUM)
                sum6 = comm.allreduce(sum6, op=MPI.SUM)
                #term2 = float(np.sum(np.logical_and(in_A,in_C_N)))/float(np.sum(in_B_N))- E
                term2 = float(sum5)/float(sum6) - E
                upper_bound += self.rho_D_M[i]*max(term1,term2)
                lower_bound += self.rho_D_M[i]*min(term1,term2)  
        return (upper_bound, lower_bound)
    
    def calculate_error_hyperbox(self, lam_domain, box, num_l_emulate=10000):
        lambda_emulate = calculateP.emulate_iid_lebesgue(lam_domain, num_l_emulate)
        l_tree1 = spatial.KDTree(self.samples)
        
        ptr1 = l_tree1.query(lambda_emulate)[1]
        #print 'here'
        in_A = np.logical_and(np.greater_equal(lambda_emulate,box[:,0]), np.less_equal(lambda_emulate,box[:,1]))
        in_A = np.all(in_A, axis=1)
        #import pdb
        #pdb.set_trace()
        upper_bound = 0.0
        lower_bound = 0.0

        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                indices = np.equal(self.io_ptr,i)
                in_Ai = indices[ptr1]
                #E = float(np.sum(np.logical_and(in_A, in_Ai)))/float(np.sum(in_Ai))
                sum1 = np.sum(np.logical_and(in_A, in_Ai))
                sum2 = np.sum(in_Ai)
                sum1 = comm.allreduce(sum1, op=MPI.SUM)
                sum2 = comm.allreduce(sum2, op=MPI.SUM)
                E = float(sum1)/float(sum2)

                in_B_N = np.zeros(in_A.shape, dtype=np.bool)
                for j in self.B_N[i]:
                    in_B_N = np.logical_or(np.equal(ptr1,j),in_B_N)

                in_C_N = np.zeros(in_A.shape, dtype=np.bool)
                for j in self.C_N[i]:
                    in_C_N =  np.logical_or(np.equal(ptr1,j), in_C_N)
                #pdb.set_trace()
                #term1 = float(np.sum(np.logical_and(in_A,in_B_N)))/float(np.sum(in_C_N))- E
                #term2 = float(np.sum(np.logical_and(in_A,in_C_N)))/float(np.sum(in_B_N))- E
                sum3 = np.sum(np.logical_and(in_A,in_B_N))
                sum4 = np.sum(in_C_N)
                sum3 = comm.allreduce(sum3, op=MPI.SUM)
                sum4 = comm.allreduce(sum4, op=MPI.SUM)
                term1 = float(sum3)/float(sum4) - E
                sum5 = np.sum(np.logical_and(in_A,in_C_N))
                sum6 = np.sum(in_B_N)
                sum5 = comm.allreduce(sum5, op=MPI.SUM)
                sum6 = comm.allreduce(sum6, op=MPI.SUM)
                term2 = float(sum5)/float(sum6) - E

                upper_bound += self.rho_D_M[i]*max(term1,term2)
                lower_bound += self.rho_D_M[i]*min(term1,term2)  
        return (upper_bound, lower_bound)
        

    def get_new_samples(self, lam_domain, num_l_emulate=100, indices = None):
        if indices == None:
            indices = []
            for i in range(self.rho_D_M.shape[0]): # ,val in enumerate(self.rho_D_M):
                if self.rho_D_M[i] > 0.0:
                    indices.append(i)

        l_tree = spatial.KDTree(self.samples)
        samples_new = np.zeros((2*num_l_emulate, self.samples.shape[1]))

        refine_list = set([])
        for index in indices:
            refine_list = set(list(refine_list) + list(set(self.C_N[index]) - set(self.B_N[index])))
        refine_list = list(refine_list)
            

        go = True
        counter = 0
        while go:
            #print counter 
            lambda_emulate = calculateP.emulate_iid_lebesgue(lam_domain, num_l_emulate)
            ptr = l_tree.query(lambda_emulate)[1]
            in_refine = np.zeros((num_l_emulate,), dtype=np.bool)
            for j in refine_list:
                in_refine = np.logical_or(in_refine, np.equal(ptr,j))
            samples_new_local = lambda_emulate[in_refine,:]
            samples_new_global = util.get_global_values(samples_new_local)
            if samples_new_global != None:
                num_new = samples_new_global.shape[0]
            else:
                num_new = 0
            samples_new[counter:counter+num_new ,:] = samples_new_global 
            counter += num_new 
            if counter > num_l_emulate:
                go = False
        return samples_new[0:num_l_emulate,:]

            
            
        
                

        

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
                JiAe = np.sum(self.lam_vol[np.logical_and(ind1,ind2)])
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

    def calculate_error_contour_events(self):
        er_list = []
        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                ind1 = np.equal(self.io_ptr1, i)
                ind2 = np.equal(self.io_ptr2, i)

                JiA = np.sum(self.lam_vol[ind1])
                Ji = JiA
                JiAe = np.sum(self.lam_vol[np.logical_and(ind1,ind2)])
                Jie = np.sum(self.lam_vol[ind2])
                er_list.append(self.rho_D_M[i]*((JiA*Jie - JiAe*Ji)/(Ji*Jie)))
                
            else:
                er_list.append(0.0)
        
        return er_list 
        
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

    def calculate_error_voronoi(self, lam_domain, samples_A, id_A, num_l_emulate):
        lambda_emulate = calculateP.emulate_iid_lebesgue(lam_domain, num_l_emulate)
        l_tree1 = spatial.KDTree(self.samples)
        l_tree2 = spatial.KDTree(samples_A)
        ptr1 = l_tree1.query(lambda_emulate)[1]
        ptr2 = l_tree2.query(lambda_emulate)[1]

        in_A = id_A[ptr2]
        er_est = 0.0
        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                indices1 = np.equal(self.io_ptr1,i)
                in_Ai1 = indices1[ptr1]
                indices2 = np.equal(self.io_ptr2,i)
                in_Ai2 = indices2[ptr1]
                JiA_local = float(np.sum(np.logical_and(in_A,in_Ai1)))
                JiA = comm.allreduce(JiA_local, op=MPI.SUM)
                Ji_local = float(np.sum(in_Ai1))
                Ji = comm.allreduce(Ji_local, op=MPI.SUM)
                JiAe_local = float(np.sum(np.logical_and(in_A,in_Ai2)))
                JiAe = comm.allreduce(JiAe_local, op=MPI.SUM)
                Jie_local = float(np.sum(in_Ai2))
                Jie = comm.allreduce(Jie_local, op=MPI.SUM)
                er_est += self.rho_D_M[i]*((JiA*Jie - JiAe*Ji)/(Ji*Jie))

        return er_est

    def calculate_error_hyperbox(self, lam_domain, box, num_l_emulate):
        lambda_emulate = calculateP.emulate_iid_lebesgue(lam_domain, num_l_emulate)
        l_tree1 = spatial.KDTree(self.samples)
        ptr1 = l_tree1.query(lambda_emulate)[1]

        in_A = np.logical_and(np.greater_equal(lambda_emulate,box[:,0]), np.less_equal(lambda_emulate,box[:,1]))
        in_A = np.all(in_A, axis=1)

        er_est = 0.0
        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                indices1 = np.equal(self.io_ptr1,i)
                in_Ai1 = indices1[ptr1]
                indices2 = np.equal(self.io_ptr2,i)
                in_Ai2 = indices2[ptr1]
                JiA_local = float(np.sum(np.logical_and(in_A,in_Ai1)))
                JiA = comm.allreduce(JiA_local, op=MPI.SUM)
                Ji_local = float(np.sum(in_Ai1))
                Ji = comm.allreduce(Ji_local, op=MPI.SUM)                
                JiAe_local = float(np.sum(np.logical_and(in_A,in_Ai2)))
                JiAe = comm.allreduce(JiAe_local, op=MPI.SUM)
                Jie_local = float(np.sum(in_Ai2))
                Jie = comm.allreduce(Jie_local, op=MPI.SUM)

                er_est += self.rho_D_M[i]*((JiA*Jie - JiAe*Ji)/(Ji*Jie))

        return er_est
    def calculate_error_hyperbox_mc(self, box):

        in_A = np.logical_and(np.greater_equal(self.samples,box[:,0]), np.less_equal(self.samples,box[:,1]))
        in_A = np.all(in_A, axis=1)

        er_est = 0.0
        P = 0.0
        P2 = 0.0
        for i in range(self.rho_D_M.shape[0]):
            if self.rho_D_M[i] > 0.0:
                indices1 = np.equal(self.io_ptr1,i)
                in_Ai1 = indices1#[ptr1]
                indices2 = np.equal(self.io_ptr2,i)
                in_Ai2 = indices2#[ptr1]
                JiA_local = float(np.sum(np.logical_and(in_A,in_Ai1)))
                JiA = comm.allreduce(JiA_local, op=MPI.SUM)
                Ji_local = float(np.sum(in_Ai1))
                Ji = comm.allreduce(Ji_local, op=MPI.SUM)                
                JiAe_local = float(np.sum(np.logical_and(in_A,in_Ai2)))
                JiAe = comm.allreduce(JiAe_local, op=MPI.SUM)
                Jie_local = float(np.sum(in_Ai2))
                Jie = comm.allreduce(Jie_local, op=MPI.SUM)

                er_est += self.rho_D_M[i]*((JiA*Jie - JiAe*Ji)/(Ji*Jie))
                P += self.rho_D_M[i]*JiA/Ji
                P2 += self.rho_D_M[i]*JiAe/Jie

        return (P, er_est, P2)
        

def refine_with_derivatives(samples,
                            data,
                            derivatives,
                            rho_D_M,
                            rho_D_M_samples,
                            lam_domain,
                            event_type,
                            event_args,
                            tol=1.0e-3,
                            error_estimate = None,
                            new_per_batch=1000,
                            max_batch_num = 100):

    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    if error_estimate !=None and len(error_estimate.shape) == 1:
        error_estimate = np.expand_dims(error_estimate, axis=1)

    #import pdb
    error = 1.0
    counter  = 0
    samples_all = samples
    data_all = data
    if error_estimate != None:
        error_estimate_all = error_estimate
    else:
        error_estimate_all = None
        
    while error > tol and counter <= max_batch_num:
        (lam_vol_all,_,_) = calculateP.exact_volume(samples=samples_all,
                                               lam_domain=lam_domain)
        se = sampling_error(samples_all, 
                            lam_vol_all, 
                            rho_D_M = rho_D_M, 
                            rho_D_M_samples = rho_D_M_samples, 
                            data=data_all)
        (h,l) = getattr(se, event_type)(*event_args)
        #pdb.set_trace()
        error = max(abs(h), abs(l))
        print "Level " + `counter` + " Sampling Error Estimate: " + `error`
        if error > tol:
            counter += 1
            samples_new = se.get_new_samples(lam_domain, num_l_emulate=new_per_batch, indices = None)
            
            (data_new, ee_new, _) = surrogate_from_derivatives(samples, 
                                                               samples_new, 
                                                               data,
                                                               derivatives,
                                                               error_estimate)
            samples_all = np.vstack((samples_all, samples_new))
            data_all = np.vstack((data_all, data_new))
            if error_estimate_all != None:
                error_estimate_all = np.vstack((error_estimate_all, ee_new))
    
        
    return (samples_all, data_all, lam_vol_all, error_estimate_all)

def total_error_with_derivatives(samples,
                                 data,
                                 derivatives,
                                 rho_D_M,
                                 rho_D_M_samples,
                                 lam_domain,
                                 event_type,
                                 event_args,
                                 error_estimate,
                                 num_emulate=int(1.0e5)):
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    if len(error_estimate.shape) == 1:
        error_estimate = np.expand_dims(error_estimate, axis=1)

    #import pdb
    samples_emulate = calculateP.emulate_iid_lebesgue(lam_domain, num_emulate)
    (data_emulate, ee_emulate, _) = surrogate_from_derivatives(samples, 
                                                               samples_emulate, 
                                                               data,
                                                               derivatives,
                                                               error_estimate)
    lam_vol_emulate = 1.0/float(num_emulate)*np.ones((num_emulate,))
    
    me = model_error(samples_emulate,
                     data_emulate,
                     ee_emulate,
                     lam_vol_emulate,
                     rho_D_M,
                     rho_D_M_samples)
    error  = getattr(me, event_type)(*event_args)

    return error

