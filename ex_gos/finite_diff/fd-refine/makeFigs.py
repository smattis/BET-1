#! /usr/bin/env python

# Copyright (C) 2016 Steven Mattis

r"""

Make figures.

"""

import numpy as np
import matplotlib.pyplot as plt
from lbModel import *

# Number of points to evaluate model at
n = 51

# exact model
x = np.zeros((1001,1))
x[:,0] = np.linspace(-0.5, 1, 1001)
y = lb_model_exact(x)

# numerical solution
xn = np.zeros((n,1))
xn[:,0] = np.linspace(-0.5, 1, n)
h = xn[1]-xn[0]
(yn, ee, jac) = lb_model(xn)
# correct with error estimate
yn_cor = yn+ee

# Plot truth, uncorrected points and corrected points
plt.figure(1)
line_true, = plt.plot(x,y, label='test')
line_num, = plt.plot(xn,yn, '*')
line_num_cor, = plt.plot(xn,yn_cor, '>')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$q(\lambda)$')
plt.legend(['True', 'Uncorrected', 'Corrected'], loc=4)
#plt.show()
plt.savefig("points_" + `n` + ".eps")

# Plot truth, uncorrected surrogate and corrected surrogate
plt.figure(2)
plt.plot(x,y)
xList = [xn[0], xn[0]+ 0.5*h]
yList = [yn[0], yn[0] + 0.5*h*jac[0]]
plt.plot(xList, yList, '-k')
xList = [xn[0], xn[0]+ 0.5*h]
yList = [yn_cor[0], yn_cor[0] + 0.5*h*jac[0]]
plt.plot(xList, yList, '-r')
for i in range(1,len(xn)-1):
    xList = np.array([xn[i]-0.5*h, xn[i]+ 0.5*h])
    yList = np.array([ yn[i] - 0.5*h*jac[i], yn[i] + 0.5*h*jac[i]])
    plt.plot(xList.flat[:], yList.flat[:], '-k')
xList = [xn[-1], xn[-1]- 0.5*h]
yList = [yn[-1], yn[-1] - 0.5*h*jac[-1]]
plt.plot(xList, yList, '-k')


for i in range(1,len(xn)-1):
    xList = np.array([xn[i]-0.5*h, xn[i]+ 0.5*h])
    yList = np.array([ yn_cor[i] - 0.5*h*jac[i], yn_cor[i] + 0.5*h*jac[i]])
    plt.plot(xList.flat[:], yList.flat[:], '-r')
xList = [xn[-1], xn[-1]- 0.5*h]
yList = [yn_cor[-1], yn_cor[-1] - 0.5*h*jac[-1]]
plt.plot(xList, yList, '-r')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$q(\lambda)$')
plt.legend(['True', 'Uncorrected', 'Corrected'], loc=4)
#plt.show()
plt.savefig("surrogates_" + `n` + ".eps")
