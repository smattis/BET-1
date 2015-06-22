from bet.Comm import comm, MPI 
import numpy as np
import scipy.spatial as spatial
import bet.util as util
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
    for i in range(samples
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
                 io_ptr,
                 exact = True):
        self.lam_vol = lam_vol
        self.rho_D_M = rho_D_M

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
