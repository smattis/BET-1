import numpy as np
import bet.sample as samp
import bet.sampling.goalOrientedSampling as gos
import bet.surrogates as surrogates
from average_qoi import lb_model0, lb_model1, lb_model2, lb_model3, lb_model_exact
from ex_mcmc import run_mcmc, calc_measures
import bet.postProcess.plotVoronoi as pv
import bet.sampling.basicSampling as bsam

alpha_1 = 1.0
alpha_2 = 1.0

def F(lam):
    return np.sqrt(lam[:, 0]**2 + lam[:,1]**2)#return np.ones(lam[:,0].shape) #np.sqrt(lam[:, 0]**2 + lam[:,1]**2)

def estimate_error(F1, p_meas1, p_meas2):
    zero = np.equal(F1, 0.0)
    not_zero = np.logical_not(zero)
    e_id = np.zeros(F1.shape)
    e_id[not_zero] =  alpha_1 * np.fabs(F1[not_zero])*np.fabs(p_meas1[not_zero] - p_meas2[not_zero])
    zero_sum = alpha_2 * abs(np.sum(p_meas1[zero]) - np.sum(p_meas2[zero]))
    e_id[zero] = zero_sum * np.fabs(p_meas1[zero] - p_meas2[zero])
    return e_id
def mark(vals):
    return np.less(vals[:,0], 1.0).astype(float)

s_set = samp.rectangle_sample_set(1)
s_set.setup(maxes=[[3.0], [3.0]], mins=[[0.0], [-1.0]])
s_set.set_region(np.array([0,1]))

sampler = gos.sampler(1000, 10, 1.0e-5, [lb_model2, lb_model3], s_set, 0 , error_estimates=True, jacobians=True)

input_samples = samp.sample_set(2)
domain = np.array([[0.0, 3.0], [-1.0, 3.0]])
input_samples.set_domain(domain)

my_discretization = sampler.initial_samples_random('r',
                                                   input_samples,
                                                   100,
                                                   level=1)
for i in range(5):

    sur = surrogates.piecewise_polynomial_surrogate(my_discretization)

    N=100000

    (x2, y2, p_meas2) = run_mcmc(sur, N, order=1, ee=True)
    (x1, y1, p_meas1) = run_mcmc(sur, N, order=1, ee=False)
    F1 = F(my_discretization._input_sample_set._values)

    #(x2, y2, p_meas2) = run_mcmc(sur, N, order=1, ee=True)
    #F2 = F(my_discretization._input_sample_set._values)

    print 'P1 =  ', np.dot(F1, p_meas1)
    print 'P2 =  ', np.dot(F1, p_meas2)

    e_id = estimate_error(F1, p_meas1, p_meas2)
    #import pdb
    #pdb.set_trace()
    marked = mark(my_discretization._input_sample_set._values)
    my_discretization._input_sample_set._probabilities_local = e_id
    my_discretization._input_sample_set._probabilities = e_id

    #pv.plot_2D_voronoi(my_discretization, density=False, interactive=False)

    # proposals
    proposals = bsam.random_sample_set('random', input_obj=domain, num_samples=10000)
    (_, cellList1) = proposals.query(x1)
    (_, cellList2) = proposals.query(x2)
    prop_meas1 = calc_measures(cellList1, proposals.check_num())
    prop_meas2 = calc_measures(cellList2, proposals.check_num())

    F1_p = F(proposals._values)
    e_id_p = estimate_error(F1_p, prop_meas1, prop_meas2)

    #proposals._probabilities_local = e_id_p
    #proposals._probabilities = e_id_p

    proposals._error_id = e_id_p
    proposals._error_id_local = e_id_p

    sampler.h_refinement_cluster(proposals, order=0, num_new_samples=50, level=0)

#proposals._probabilities_local = prop_meas2
#proposals._probabilities = prop_meas2
(x2, y2, p_meas2) = run_mcmc(sur, N, order=1, ee=True)
(x1, y1, p_meas1) = run_mcmc(sur, N, order=1, ee=False)
F1 = F(my_discretization._input_sample_set._values)
my_discretization._input_sample_set._probabilities_local = p_meas2
my_discretization._input_sample_set._probabilities = p_meas2

pv.plot_2D_voronoi(my_discretization, density=False, interactive=True)



import pdb
pdb.set_trace()
