import numpy as np
import bet.sample as samp
import bet.sampling.goalOrientedSampling as gos
import bet.surrogates as surrogates
from average_qoi import lb_model0, lb_model1, lb_model2, lb_model3, lb_model4, lb_model_exact
from ex_mcmc import run_mcmc, calc_measures, run_mcmc_exact
import bet.postProcess.plotVoronoi as pv
import bet.sampling.basicSampling as bsam
import matplotlib.pyplot as plt
import matplotlib
import plotting

alpha_1 = 1.0
alpha_2 = 1.0

def F(lam):
    return np.greater(lam[:, 0], 1.00).astype(float)
    #return np.sqrt(lam[:, 0]**2 + lam[:,1]**2)#return np.ones(lam[:,0].shape) #np.sqrt(lam[:, 0]**2 + lam[:,1]**2)

def estimate_error(F1, p_meas1, p_meas2):
    zero = np.equal(F1, 0.0)
    not_zero = np.logical_not(zero)
    e_id = np.zeros(F1.shape)
    e_id[not_zero] =  alpha_1 * np.fabs(F1[not_zero])*np.fabs(p_meas1[not_zero] - p_meas2[not_zero])
    zero_sum = alpha_2 * abs(np.sum(p_meas1[zero]) - np.sum(p_meas2[zero]))
    e_id[zero] = zero_sum * np.fabs(p_meas1[zero] - p_meas2[zero])
    return e_id

def estimate_error2(disc, x1, x2, sset_fine=None):
    F11 = F(x1)
    F12 = F(x2)
    #zero = np.logical_and(np.equal(F11, 0.0), np.equal(F12, 0.0))
    zero1 = np.equal(F11, 0.0)
    zero2 = np.equal(F12, 0.0)
    num = disc.check_nums()
    (_, id1) = disc._input_sample_set.query(x1)
    (_, id2) = disc._input_sample_set.query(x2)
    #import pdb
    #pdb.set_trace()
    e_id = np.zeros((num,))
    #e_id2 = np.zeros((num,))
    F1 = np.zeros((num,))
    F2 = np.zeros((num,))
    z1 = np.zeros((num,))
    z2 = np.zeros((num,))

    if sset_fine is not None:
        num_fine = sset_fine.check_num()
        F1_fine = np.zeros((num_fine,))
        F2_fine = np.zeros((num_fine,))
        z1_fine = np.zeros((num_fine,))
        z2_fine = np.zeros((num_fine,))
        (_, id1_fine) = sset_fine.query(x1)
        (_, id2_fine) = sset_fine.query(x2)
                        
    #num_F1 = np.zeros((num,))
    #num_F2 = np.zeros((num,))
    for i in range(num):
        ids1 = np.equal(id1,i)
        if len(ids1) > 0:
            F1[i] = np.sum(F11[ids1])/float(len(x1))
            z1[i]= float(np.sum(zero1[ids1]))/float(len(x1))
        ids2 = np.equal(id2,i)
        if len(ids2) > 0:
            F2[i] = np.sum(F12[ids2])/float(len(x2))
            z2[i]= float(np.sum(zero2[ids2]))/float(len(x2))
    # if sset_fine is not None:
    #     for i in range(num_fine):
    #         ids1_fine = np.equal(id1_fine,i)
    #         if len(ids1_fine) > 0:
    #             F1_fine[i] = np.sum(F11_fine[ids1])/float(len(x1))
    #             z1[i]= float(np.sum(zero1[ids1]))/float(len(x1))
    #         ids2 = np.equal(id2,i)
    #         if len(ids2) > 0:
    #             F2[i] = np.sum(F12[ids2])/float(len(x2))
    #             z2[i]= float(np.sum(zero2[ids2]))/float(len(x2))
        

    e_id = alpha_1 * np.fabs(F2-F1) + alpha_2 * (abs(np.average(z2) - np.average(z1))* np.fabs(z2-z1))**0.5
    # count1 = np.ones((len(x1),))
    # count2 = np.ones((len(x2),))
    # F1[id1] += F11
    # F2[id2] += F12
    # count_id1 = np.zeros((num,))
    # count_id2 = np.zeros((num,))
    # count_id1[id1] += count1
    # count_id2[id2] += count2
    # F1 = F1/count_id1
    # F2 = F2/count_id2
    # z1[id1[zero1]] += 1.0
    # z2[id2[zero2]] += 1.0
    # z1 = z1/count_id1
    # z2 = z2/count_id2
    # e_id = alpha_1 * np.fabs(F2-F1) + alpha_2 * (abs(np.average(z2) - np.average(z1))* np.fabs(z2-z1))**0.5
  
    return e_id
        
def mark(vals):
    return np.less(vals[:,0], 1.0).astype(float)




#Exact answer
(x_exact, y_exact) = run_mcmc_exact(lb_model_exact, 10000)
int_exact = np.average(F(x_exact), axis=0)
print 'Exact integral: ', int_exact

# Run with surrogate
s_set = samp.rectangle_sample_set(1)
s_set.setup(maxes=[[3.0], [3.0]], mins=[[0.0], [-1.0]])
s_set.set_region(np.array([0,1]))

sampler = gos.sampler(1000, 10, 1.0e-5, [lb_model0, lb_model1, lb_model2, lb_model3, lb_model4], s_set, 0 , error_estimates=True, jacobians=True)

input_samples = samp.sample_set(2)
domain = np.array([[0.0, 3.0], [-1.0, 3.0]])
input_samples.set_domain(domain)

my_discretization = sampler.initial_samples_random('r',
                                                   input_samples,
                                                   100,
                                                   level=0)
for i in range(30):
    my_discretization._input_sample_set.set_kdtree()

    sur = surrogates.piecewise_polynomial_surrogate(my_discretization)

    N=10000

    (x2, y2, p_meas2) = run_mcmc(sur, N, order=1, ee=True)
    (x1, y1, p_meas1) = run_mcmc(sur, N, order=1, ee=False)
    F1 = F(my_discretization._input_sample_set._values)

    #(x2, y2, p_meas2) = run_mcmc(sur, N, order=1, ee=True)
    #F2 = F(my_discretization._input_sample_set._values)
    print "exact integral = ", int_exact
    print 'P1 =  ', np.dot(F1, p_meas1), np.average(F(x1))
    print 'P2 =  ', np.dot(F1, p_meas2), np.average(F(x2))

    #e_id = estimate_error(F1, p_meas1, p_meas2)
    e_id = estimate_error2(sampler.disc, x1, x2)
    #import pdb
    #pdb.set_trace()
    marked = mark(my_discretization._input_sample_set._values)
    my_discretization._input_sample_set._probabilities_local = e_id
    my_discretization._input_sample_set._probabilities = e_id
    
    my_discretization._input_sample_set._error_id_local = e_id
    my_discretization._input_sample_set._error_id = e_id
    #pv.plot_2D_voronoi(my_discretization, density=False, interactive=False)
    plotting.plot_hist(sampler.disc, x2, False, save=True, savename='hist_'+`i`)
    plotting.plot_error_id_voronoi(sampler.disc, False, save=True, savename='errorid_'+`i`)
    plotting.plot_levels_voronoi(sampler.disc, False, save=True, savename='levels_'+`i`, data_range=[1.0, 5.0])

    if i%2 == 0:
        # proposals
        proposals = bsam.random_sample_set('random', input_obj=domain, num_samples=1000)
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

        #sampler.h_refinement_cluster(proposals, order=0, num_new_samples=50, level=0, tol=0.005, match_level=True)
        #print 'Total Samples after cluster ref. : ', sampler.disc.check_nums()
        sampler.h_refinement_opt(proposals, order=0, num_new_samples=20, estimate_error=estimate_error2, x1=x2, x2=x2)

    else:
        sampler.level_refinement(None, 0, 20)
        print 'Total Samples after level ref.: ', sampler.disc.check_nums()
        for K in range(len(sampler.lb_model_list)):
            Num = int(np.sum(np.equal(sampler.disc._input_sample_set._levels, K)))
            print `Num` + " runs at level " + `K`
    #plotting.plot_error_id_voronoi(sampler.disc, True)

#proposals._probabilities_local = prop_meas2
#proposals._probabilities = prop_meas2
N_fin = 1000000
my_discretization._input_sample_set.set_kdtree()
(x2, y2, p_meas2) = run_mcmc(sur, N_fin, order=1, ee=True)
(x1, y1, p_meas1) = run_mcmc(sur, N_fin, order=1, ee=False)
F1 = F(my_discretization._input_sample_set._values)

#print 'P1 =  ', np.dot(F1, p_meas1)
#print 'P2 =  ', np.dot(F1, p_meas2)
print "exact integral = ", int_exact
print 'P1 =  ', np.dot(F1, p_meas1), np.average(F(x1))
print 'P2 =  ', np.dot(F1, p_meas2), np.average(F(x2))

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

plotting.plot_hist(x_exact, False)
plotting.plot_hist(x2, True)

# H, xedges, yedges = np.histogram2d(x2[:,0], x2[:,1], bins=40, range=[[0.0, 3.0], [-1.0, 3.0]])
# H=H.T
# H = H/float(N)
# fig = plt.figure()
# colormap_type='BuGn'
# cmap = matplotlib.cm.get_cmap(colormap_type)
# plt.imshow(H, interpolation='bicubic', origin='low', cmap=cmap,
#           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
# sample_obj = sur.input_disc._input_sample_set
# plt.axis([sample_obj._domain[0][0], sample_obj._domain[0][1], sample_obj._domain[1][0], sample_obj._domain[1][1]])
# ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=0.9)
# cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
#                                         norm=matplotlib.colors.Normalize(vmin=0.0, vmax=np.max(H)), label=r'$P_{\Lambda}(\mathcal{V}_i)$')
# text = cbar.ax.yaxis.label
# font = matplotlib.font_manager.FontProperties(size=20)
# text.set_font_properties(font)

# ### plot exact
# H, xedges, yedges = np.histogram2d(x_exact[:,0], x_exact[:,1], bins=40, range=[[0.0, 3.0], [-1.0, 3.0]])
# H=H.T
# H = H/float(N)
# fig = plt.figure()
# colormap_type='BuGn'
# cmap = matplotlib.cm.get_cmap(colormap_type)
# plt.imshow(H, interpolation='bicubic', origin='low', cmap=cmap,
#           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
# sample_obj = sur.input_disc._input_sample_set
# plt.axis([sample_obj._domain[0][0], sample_obj._domain[0][1], sample_obj._domain[1][0], sample_obj._domain[1][1]])
# ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=0.9)
# cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
#                                         norm=matplotlib.colors.Normalize(vmin=0.0, vmax=np.max(H)), label=r'$P_{\Lambda}(\mathcal{V}_i)$')
# text = cbar.ax.yaxis.label
# font = matplotlib.font_manager.FontProperties(size=20)
# text.set_font_properties(font)


# plt.show()

import pdb
pdb.set_trace()
