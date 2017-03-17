import numpy as np
import bet.sample as samp
import bet.sampling.goalOrientedSampling as gos
import bet.surrogates as surrogates
from average_qoi import lb_model0, lb_model1, lb_model2, lb_model3, lb_model4, lb_model_exact
from ex_mcmc import run_mcmc, calc_measures
import bet.postProcess.plotVoronoi as pv
import bet.sampling.basicSampling as bsam
import matplotlib.pyplot as plt
import matplotlib

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

sampler = gos.sampler(1000, 10, 1.0e-5, [lb_model0, lb_model1, lb_model2, lb_model4], s_set, 0 , error_estimates=True, jacobians=True)

input_samples = samp.sample_set(2)
domain = np.array([[0.0, 3.0], [-1.0, 3.0]])
input_samples.set_domain(domain)

my_discretization = sampler.initial_samples_random('r',
                                                   input_samples,
                                                   100,
                                                   level=0)
for i in range(10):
    my_discretization._input_sample_set.set_kdtree()

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
    
    my_discretization._input_sample_set._error_id_local = e_id
    my_discretization._input_sample_set._error_id = e_id
    #pv.plot_2D_voronoi(my_discretization, density=False, interactive=False)

    if i%2 == 0:
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

        sampler.h_refinement_cluster(proposals, order=0, num_new_samples=50, level=0, tol=0.005, match_level=True)
        print 'Total Samples after cluster ref. : ', sampler.disc.check_nums()

    else:
        sampler.level_refinement(None, 0, 20)
        print 'Total Samples after level ref.: ', sampler.disc.check_nums()
        for K in range(len(sampler.lb_model_list)):
            Num = int(np.sum(np.equal(sampler.disc._input_sample_set._levels, K)))
            print `Num` + " runs at level " + `K`

#proposals._probabilities_local = prop_meas2
#proposals._probabilities = prop_meas2
my_discretization._input_sample_set.set_kdtree()
(x2, y2, p_meas2) = run_mcmc(sur, N, order=1, ee=True)
(x1, y1, p_meas1) = run_mcmc(sur, N, order=1, ee=False)
F1 = F(my_discretization._input_sample_set._values)

print 'P1 =  ', np.dot(F1, p_meas1)
print 'P2 =  ', np.dot(F1, p_meas2)

my_discretization._input_sample_set._probabilities_local = p_meas1
my_discretization._input_sample_set._probabilities = p_meas1
my_discretization._input_sample_set.exact_volume_2D()
pv.plot_2D_voronoi(my_discretization, density=True, interactive=False)

my_discretization._input_sample_set._probabilities_local = p_meas2
my_discretization._input_sample_set._probabilities = p_meas2
#my_discretization._input_sample_set.exact_volume_2D()
pv.plot_2D_voronoi(my_discretization, density=True, interactive=False)

my_discretization._input_sample_set._probabilities_local = my_discretization._input_sample_set._levels.astype(float)
my_discretization._input_sample_set._probabilities = my_discretization._input_sample_set._levels.astype(float)
#my_discretization._input_sample_set.exact_volume_2D()
pv.plot_2D_voronoi(my_discretization, density=False, interactive=False)


H, xedges, yedges = np.histogram2d(x2[:,0], x2[:,1], bins=40, range=[[0.0, 3.0], [-1.0, 3.0]])
H=H.T
H = H/float(N)
fig = plt.figure()
colormap_type='BuGn'
cmap = matplotlib.cm.get_cmap(colormap_type)
plt.imshow(H, interpolation='bicubic', origin='low', cmap=cmap,
          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
sample_obj = sur.input_disc._input_sample_set
plt.axis([sample_obj._domain[0][0], sample_obj._domain[0][1], sample_obj._domain[1][0], sample_obj._domain[1][1]])
ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=0.9)
cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=np.max(H)), label=r'$P_{\Lambda}(\mathcal{V}_i)$')
text = cbar.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(size=20)
text.set_font_properties(font)
plt.show()

import pdb
pdb.set_trace()