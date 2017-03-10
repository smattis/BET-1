import numpy as np
import math
import scipy.stats as stats
import random as rand
import numpy.random as nrand
import matplotlib.pyplot as plt
from average_qoi import *

# Model
model = lb_model3

# Data
y = 0.2
yList=[y]
    

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
    return math.exp(-1.0/(2.0*noise_std**2)*(yList[0] - y_prop)**2)

# Chain length
N=1000000

# Initialize state
x = np.empty((N,2))
x[0,:] = np.array([1.5, 1.5])
(y, _, _) = model(np.array([x[0,:]]))
y = y[0][0]
                           

go = True
its = 1
total = 1

while go:
    total += 1
    x_prop0 = x[its-1,0] + nrand.normal(0.0, prop_std[0])
    x_prop1 = x[its-1,1] + nrand.normal(0.0, prop_std[1])
    
    (y_prop, _, _) = model(np.array([[x_prop0, x_prop1]]))
    y_prop = y_prop[0][0]
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

plt.savefig("approx2_posterior.png")

xs = np.linspace(xedges[0], xedges[-1], 100)
ys = np.linspace(yedges[0], yedges[-1], 100)
X,Y = np.meshgrid(xs, ys) # grid of point
Z = prior_func(X, Y) # evaluation of the function on the grid
fig = plt.figure(1)
plt.imshow(Z,interpolation='bicubic', origin='low',
          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.savefig("prior.png")

np.savetxt("approx_better.txt", x)
plt.show()
#import pdb
#pdb.set_trace()
