import scipy.stats as stats
import math
import numpy as np
import random as rand
import numpy.random as nrand
import matplotlib.pyplot as plt
import matplotlib
from average_qoi import *

# Data
y_dat = 0.2

# Gaussian prior
prior_mean = [1.25, 1.25]
prior_std = [0.5, 0.5]

def prior_func(x,y):
    return stats.norm.pdf(x, prior_mean[0], prior_std[0])*stats.norm.pdf(y, prior_mean[1], prior_std[1])

# Gaussian Noise
noise_mean = 0.0
noise_std = 0.02

# Proposal
prop_std = [0.01, 0.01]

def likelihood(y_prop):
    return math.exp(-1.0/(2.0*noise_std**2)*(y_dat - y_prop)**2)

def calc_measures(cellList, num_vor):
    N = len(cellList)
    unique, count = np.unique(cellList, return_counts=True)
    p_meas = np.zeros((num_vor,))#np.zeros((int(np.sum(count)),))#np.zeros((sur.input_disc.check_nums(),))
    p_meas[unique] = count.astype(float)
    p_meas = p_meas/float(N)
    return p_meas

# Chain length
#N=100000

def run_mcmc(sur, N=100, order=1, ee=True):
    # Initialize state
    yList=[]
    cellList=[]
    x = np.empty((N,2))
    x[0,:] = np.array([1.5, 1.5])
    y = sur.evaluate_at_values(np.array([x[0,:]]), order, ee)#model(np.array([x[0,:]]))
    cellList.append(y[1][0])
    y = y[0][0,0]
    yList.append(y)

    go = True
    its = 1
    total = 1

    while go:
        total += 1
        x_prop0 = x[its-1,0] + nrand.normal(0.0, prop_std[0])
        x_prop1 = x[its-1,1] + nrand.normal(0.0, prop_std[1])

        y_prop = sur.evaluate_at_values(np.array([[x_prop0, x_prop1]]), order, ee)
        cell = y_prop[1][0]
        y_prop = y_prop[0][0]
        # (yp2, _, _) = lb_model_exact(np.array([[x_prop0, x_prop1]]))
        # print y_prop, yp2[0,0]
        rho_num = likelihood(y_prop)*prior_func(x_prop0, x_prop1) #stats.norm.pdf(x_prop0, prior_mean[0], prior_std[0])*stats.norm.pdf(x_prop1, prior_mean[1], prior_std[1])
        rho_den = likelihood(y)*prior_func(x[its-1,0], x[its-1,1]) #*stats.norm.pdf(x[-1][0], prior_mean[0], prior_std[0])*stats.norm.pdf(x[-1][1], prior_mean[1], prior_std[1])
        rho = min(1, rho_num/rho_den)
        # Accept?
        if rand.uniform(0.0, 1.0) < rho:
            x[its,:] = np.array([x_prop0, x_prop1]) #np.vstack((x, np.array([[x_prop0, x_prop1]])))
            yList.append(y)
            cellList.append(cell)
            y = y_prop
            its += 1
            if its == N:
                go = False
            if total%10000 == 0:
                print float(its)/float(N)
    #unique, count = np.unique(cellList, return_counts=True)
    #p_meas = np.zeros((sur.input_disc.check_nums(),))
    #p_meas[unique] = count.astype(float)
    #p_meas = p_meas/float(N)
    #import pdb
    #pdb.set_trace()
    p_meas = calc_measures(cellList, sur.input_disc.check_nums())
    print float(its)/float(total)
    H, xedges, yedges = np.histogram2d(x[:,0], x[:,1], bins=40, range=[[0.0, 3.0], [-1.0, 3.0]])
    H=H.T
    H = H/float(N)
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
    # #plt.show()
    return (x, y, p_meas)

