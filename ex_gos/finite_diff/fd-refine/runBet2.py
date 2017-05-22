import numpy as np
import bet.sample as samp
import bet.sampling.goalOrientedSampling as gos
import bet.surrogates as surrogates
from average_qoi import lb_model0, lb_model1, lb_model2, lb_model3, lb_model4, lb_model_exact, predict_model_exact
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



#Exact answer
(x_exact, y_exact) = run_mcmc_exact(lb_model_exact, 1000000)
int_exact = np.average(predict_model_exact(x_exact), axis=0)
print 'Exact integral: ', int_exact

# Run with surrogate
#s_set = samp.rectangle_sample_set(1)
#s_set.setup(maxes=[[3.0], [3.0]], mins=[[0.0], [-1.0]])
#s_set.set_region(np.array([0,1]))

sampler = gos.sampler_hpl_adaptive([lb_model0, lb_model1, lb_model2, lb_model3, lb_model4], f=predict_model_exact, jacobians=True)

input_samples = samp.sample_set(2)
domain = np.array([[0.0, 3.0], [-1.0, 3.0]])
input_samples.set_domain(domain)

my_discretization = sampler.initial_samples_random('r',
                                                   input_samples,
                                                   100,
                                                   level=0,
                                                   emulate=True,
                                                   emulate_num=1E5)
subgrid_set = bsam.random_sample_set('r', domain, num_samples=int(1E4))

#import pdb
#pdb.set_trace()
xList = []
yList = []
errorList = []
lw=[]
up=[]

for i in range(30):
    my_discretization._input_sample_set.set_kdtree()

    sur = surrogates.piecewise_polynomial_surrogate(my_discretization)

    N=1000000

    (x2, y2, p_meas2) = run_mcmc(sur, N, order=1, ee=True)
    (x1, y1, p_meas1) = run_mcmc(sur, N, order=1, ee=False)
    (int1, int2, Ei) = sampler.hl_step_setup_chain(x1, x2, subgrid_set, 0.05)
    #sampler.set_probabilities(p_meas1, p_meas2)
    #F1 = F(my_discretization._input_sample_set._values)
    #sampler.calculate_gamma()
    #ee_prob = sampler.calculate_ee_prob()
    #(ee, int1, int2) = sampler.calculate_ee_chain(x1,x2)
    #inds = np.argsort(ee)[::-1][0:3]
    #(E, p) = sampler.calculate_subgrid_ee_chain(inds, subgrid_set, x1, x2)
    #sampler.hl_step_chain(x1, x2, subgrid_set, 0.1)
    #import pdb
    #pdb.set_trace()
    #(x2, y2, p_meas2) = run_mcmc(sur, N, order=1, ee=True)
    #F2 = F(my_discretization._input_sample_set._values)
    print "exact integral = ", int_exact
    print 'P1 =  ', int1 #np.dot(F1, p_meas1), np.average(F(x1))
    print 'P2 =  ', int2 #np.dot(F1, p_meas2), np.average(F(x2))
    print 'E_sum = ', Ei
    print "integral bound = ", (int1 - np.sum(sampler.ee_int)), (int1 + np.sum(sampler.ee_int))
    print "Model Evals = ", sampler.total_evals[:]
    xList.append(i)
    yList.append(int2)
    lw.append(int2 - (int1-np.sum(sampler.ee_int)))
    up.append(int1 + np.sum(sampler.ee_int)-int2)
    #errorList.append([lw, up])
    fig = plt.figure()
    #print errorList
    print lw[-1], up[-1]
    plt.errorbar(xList, yList, yerr=[lw,up], fmt='o')
    plt.axhline(y=int_exact, color='r')
    plt.xlabel("Iteration")
    plt.ylabel("Integral Estimate")
    axes = plt.gca()
    axes.set_xlim([xList[0]-0.1, xList[-1]+0.1])
    #axes.set_ylim([0.1, 0.3])
    xticks = range(len(xList))
    plt.xticks(xticks)
    #fig.tight_layout(pad=2)
    plt.savefig("error_plot.png")
    plt.close()
    sampler.disc._io_ptr = None
    sampler.disc._io_ptr_local = None

    #e_id = estimate_error(F1, p_meas1, p_meas2)
    #e_id = estimate_error2(sampler.disc, x1, x2)
    #import pdb
    #pdb.set_trace()
    #marked = mark(my_discretization._input_sample_set._values)

    e_id = sampler.disc._input_sample_set._error_estimates
    my_discretization._input_sample_set._probabilities_local = e_id
    
    my_discretization._input_sample_set._probabilities = e_id
    
    #my_discretization._input_sample_set._error_id_local = e_id
    #my_discretization._input_sample_set._error_id = e_id
    #pv.plot_2D_voronoi(my_discretization, density=False, interactive=False)
    plotting.plot_hist(sampler.disc, x2, False, save=True, savename='hist_'+`i`)
    plotting.plot_error_id_voronoi(sampler.disc, False, save=True, savename='errorid_'+`i`)
    plotting.plot_levels_voronoi(sampler.disc, False, save=True, savename='levels_'+`i`, data_range=[0.0, 4.0])
    sampler.hl_step_chain()

    #(x2, y2, p_meas2) = run_mcmc(sur, N, order=1, ee=True)
    #(x1, y1, p_meas1) = run_mcmc(sur, N, order=1, ee=False)
    # sampler.hl_step_chain(x1, x2, subgrid_set, 0.1)

    # if i%2 == 0:
    #     # proposals
    #     proposals = bsam.random_sample_set('random', input_obj=domain, num_samples=1000)
    #     (_, cellList1) = proposals.query(x1)
    #     (_, cellList2) = proposals.query(x2)
    #     prop_meas1 = calc_measures(cellList1, proposals.check_num())
    #     prop_meas2 = calc_measures(cellList2, proposals.check_num())

    #     F1_p = F(proposals._values)
    #     e_id_p = estimate_error(F1_p, prop_meas1, prop_meas2)

    #     #proposals._probabilities_local = e_id_p
    #     #proposals._probabilities = e_id_p

    #     proposals._error_id = e_id_p
    #     proposals._error_id_local = e_id_p

    #     #sampler.h_refinement_cluster(proposals, order=0, num_new_samples=50, level=0, tol=0.005, match_level=True)
    #     #print 'Total Samples after cluster ref. : ', sampler.disc.check_nums()
    #     sampler.h_refinement_opt(proposals, order=0, num_new_samples=20, estimate_error=estimate_error2, x1=x2, x2=x2)

    # else:
    #     sampler.level_refinement(None, 0, 20)
    #     print 'Total Samples after level ref.: ', sampler.disc.check_nums()
    #     for K in range(len(sampler.lb_model_list)):
    #         Num = int(np.sum(np.equal(sampler.disc._input_sample_set._levels, K)))
    #         print `Num` + " runs at level " + `K`
    #plotting.plot_error_id_voronoi(sampler.disc, True)

#proposals._probabilities_local = prop_meas2
#proposals._probabilities = prop_meas2
# N_fin = 1000000
# my_discretization._input_sample_set.set_kdtree()
# (x2, y2, p_meas2) = run_mcmc(sur, N_fin, order=1, ee=True)
# (x1, y1, p_meas1) = run_mcmc(sur, N_fin, order=1, ee=False)
# F1 = F(my_discretization._input_sample_set._values)

# #print 'P1 =  ', np.dot(F1, p_meas1)
# #print 'P2 =  ', np.dot(F1, p_meas2)
# print "exact integral = ", int_exact
# print 'P1 =  ', np.dot(F1, p_meas1), np.average(F(x1))
# print 'P2 =  ', np.dot(F1, p_meas2), np.average(F(x2))

# my_discretization._input_sample_set._probabilities_local = p_meas1
# my_discretization._input_sample_set._probabilities = p_meas1
# my_discretization._input_sample_set.exact_volume_2D()
# pv.plot_2D_voronoi(my_discretization, density=True, interactive=False)

# my_discretization._input_sample_set._probabilities_local = p_meas2
# my_discretization._input_sample_set._probabilities = p_meas2
# #my_discretization._input_sample_set.exact_volume_2D()
# pv.plot_2D_voronoi(my_discretization, density=True, interactive=False)

# my_discretization._input_sample_set._probabilities_local = my_discretization._input_sample_set._levels.astype(float)
# my_discretization._input_sample_set._probabilities = my_discretization._input_sample_set._levels.astype(float)
# #my_discretization._input_sample_set.exact_volume_2D()
# pv.plot_2D_voronoi(my_discretization, density=False, interactive=False)

# plotting.plot_hist(x_exact, False)
# plotting.plot_hist(x2, True)

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