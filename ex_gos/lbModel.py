import os
import scipy.io as sio
import sys
import numpy as np
import numpy.random as rand
import numpy.linalg as linalg
from math import *

def gauss(A, b, x, n):
    """
    Gauss-Seidel Method
    """
    L = np.tril(A)
    U = A - L
    for i in range(n):
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
    return x

def lb_model_exact(input_data):
    """
    Exact map.
    """

    num_runs = input_data.shape[0]
    num_runs_dim = input_data.shape[1]
    
    values = np.zeros((num_runs, 1))

    for i in range(num_runs):
        lam = input_data[i, 0]
        A = [[exp(lam), cos(pi*lam)],
             [sin(pi*lam), 2.0*exp(lam)]]
        #b = np.array([lam, 0.0])
        b = np.array([sin(10*pi*lam), exp(3.0*lam)])
        A=np.array(A)
        x = linalg.solve(A,b)
        values[i] = 0.5*np.sum(x)
    return values

def lb_model(input_data):
    """
    Approximate map.
    """

    num_runs = input_data.shape[0]
    num_runs_dim = input_data.shape[1]
    
    values = np.zeros((num_runs, 1))
    jacobians = np.zeros((num_runs, 1, 1))
    error_estimates = np.zeros((num_runs, 1))

    for i in range(num_runs):
        lam = input_data[i, 0]
        A = [[exp(lam), cos(pi*lam)],
             [sin(pi*lam), 2.0*exp(lam)]]
        #b = np.array([lam, 0.0])
        b = np.array([sin(10*pi*lam), exp(3.0*lam)])

        A=np.array(A)
        x0 = np.array([10.0, 10.0])
        # Solve with Gauss-Seidel method
        x = gauss(A, b, x0, 2)
        psi = np.array([0.5, 0.5])
        values[i] = np.dot(x, psi)
        # Solve adjoint problem exactly
        phi = linalg.solve(A.transpose(), psi)
        # Calculate error estimate
        ee = np.dot(b-np.dot(A, x), phi)
        error_estimates[i] = ee

        dAdlam = [[exp(lam), -pi*sin(pi*lam)],
                  [pi*cos(pi*lam), 2.0*exp(lam)]]
        #dbdlam = np.array([1.0, 0.0])
        dbdlam = np.array([10.0*pi*cos(10.0*pi*lam), 3.0*exp(3.0)*lam])
        # Calculate derivative
        deriv = np.dot(dbdlam - np.dot(dAdlam, x), phi)
        jacobians[i] = deriv

    return (values, error_estimates, jacobians)
