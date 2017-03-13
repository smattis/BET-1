import numpy as np
import bet.sample as samp
import bet.sampling.goalOrientedSampling as gos
import bet.surrogates as surrogates
from average_qoi import lb_model0, lb_model1, lb_model_exact

import scipy.stats as stats
import math
import random as rand
import numpy.random as nrand
import matplotlib.pyplot as plt
from average_qoi import *

s_set = samp.rectangle_sample_set(1)
s_set.setup(maxes=[[3.0], [3.0]], mins=[[0.0], [-1.0]])
s_set.set_region(np.array([0,1]))

sampler = gos.sampler(1000, 10, 1.0e-5, [lb_model0, lb_model1], s_set, 0 , error_estimates=True, jacobians=True)

input_samples = samp.sample_set(2)
domain = np.array([[0.0, 3.0], [-1.0, 3.0]])
input_samples.set_domain(domain)

my_discretization = sampler.initial_samples_random('r',
                                                   input_samples,
                                                   10000,
                                                   level=1)
sur = surrogates.piecewise_polynomial_surrogate(my_discretization)

# Begining of MCMC code
# Model
#model = sur

# Data
y_dat = 0.2
yList=[]
    

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

# Chain length
N=1000000

# Initialize state
x = np.empty((N,2))
x[0,:] = np.array([1.5, 1.5])
y = sur.evaluate_at_values(np.array([x[0,:]]), 1, True)#model(np.array([x[0,:]]))
y = y[0][0]
yList.append(y)

go = True
its = 1
total = 1

while go:
    total += 1
    x_prop0 = x[its-1,0] + nrand.normal(0.0, prop_std[0])
    x_prop1 = x[its-1,1] + nrand.normal(0.0, prop_std[1])
    
    y_prop = sur.evaluate_at_values(np.array([[x_prop0, x_prop1]]), 1, True)
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
        y = y_prop
        its += 1
        if its == N:
            go = False
        if total%10000 == 0:
            print float(its)/float(N)
print float(its)/float(total)
H, xedges, yedges = np.histogram2d(x[:,0], x[:,1], bins=40, range=[[0.0, 3.0], [-1.0, 3.0]])
fig = plt.figure(0)
plt.imshow(H, interpolation='bicubic', origin='low',
          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.show()
