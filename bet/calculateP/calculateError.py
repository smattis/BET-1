from bet.Comm import comm, MPI 
import numpy as np
import scipy.spatial as spatial
import bet.util as util


                 
def cell_connectivity_exact(samples, io_ptr):
    from scipy.spatial import Delaunay
    from collections import defaultdict
    import itertools

    tri = Delaunay(samples)
    neiList=defaultdict(set)
    for p in tri.vertices:
        for i,j in itertools.combinations(p,2):
            neiList[i].add(io_ptr[j])
            neiList[j].add(io_ptr[i])
    for i in range(samples.shape[0]):
        neiList[i] = list(set(neiList[i]))
    return neiList

def cell_connectivity_approx(samples, io_ptr):
    l_Tree = spatial.KDTree(samples)
    pass

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
