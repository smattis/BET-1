import os
import scipy.io as sio
import sys
import numpy as np
import numpy.random as rand
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg
from math import *

# interval over which solution is averaged
(ia, ib) = (0.367, 0.643)

def lb_model_exact(input_data):
    '''
    Evaluate model and calculate derivatives exactly.
    '''
    a = ia
    b = ib
    # intialize outputs
    num_runs = input_data.shape[0]
    values = np.zeros((num_runs, 1))
    jacobians = np.zeros((num_runs, 1, 2))
    error_estimates = np.zeros((num_runs, 1))

    # get parameters
    k = input_data[:,0]
    alpha = input_data[:,1]
    
    # evaluate values and derivatives
    values[:, 0] = (-2 * np.exp(alpha * a) + 2 * np.exp(alpha * b) + (a ** 2 - b ** 2) * alpha * np.exp(alpha) + (-a ** 2 + b ** 2 + 2 * a - 2 * b) * alpha) / k / alpha ** 3 / (a - b) / 2

    jacobians[:, 0, 0] = (2 * np.exp(alpha * a) - 2 * np.exp(alpha * b) + (-a ** 2 + b ** 2) * alpha * np.exp(alpha) + (a ** 2 - b ** 2 - 2 * a + 2 * b) * alpha) / k ** 2 / alpha ** 3 / (a - b) / 2

    jacobians[:, 0, 1] = ((-2 * alpha * a + 6) * np.exp(alpha * a) + (2 * alpha * b - 6) * np.exp(alpha * b) + (a - b) * ((alpha - 2) * (a + b) * np.exp(alpha) + 2 * a + 2 * b - 4) * alpha) / k / alpha ** 4 / (a - b) / 2
    
    return (values, error_estimates, jacobians)


def model_approx(input_data, h):
    """
    Solve model with centered finite differences with given  h.
    Solve adjoint problem with grid size of h/2.
    Calcuate qoi, error estimates and derivatives.
    """
    num_runs = input_data.shape[0]
    num_runs_dim = input_data.shape[1]

    #initialize outputs
    values = np.zeros((num_runs, 1))
    jacobians = np.zeros((num_runs, 1, 2))
    error_estimates = np.zeros((num_runs, 1))

    # Loop over input parameters
    for i in range(num_runs):
        alpha  = input_data[i,1]
        k = input_data[i, 0]

        #define mesh
        xval = np.arange(h, 1.0, h)
        pts=len(xval)
        #define refined mesh
        xval_adj = np.arange(0.5*h, 1.0, 0.5*h)
        pts_adj = len(xval_adj)
        
        # Setup Forward Solution:

        # Discretize ODE: \f[ -k u^\prime\prime(x_n)= \frac{2u(x_n)-u(x_{n-1})-u(x_{n+1})}{h^2} = e^{\alpha x_n} \f]
        # Uniform grid so grid size h can be moved to RHS
        b = h**2/k*np.exp(alpha*xval)
        b_adj = (0.5*h)**2/k*np.exp(alpha*xval_adj)
        
        # We use the spdiags command to map -1 2 -1 to the tridiagonal matrix A
        temp = np.hstack((-np.ones((pts,1)), 2.0*np.ones((pts,1)), -np.ones((pts,1)))).transpose()
        A = sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")

        temp_adj = np.hstack((-np.ones((pts_adj,1)), 2.0*np.ones((pts_adj,1)), -np.ones((pts_adj,1)))).transpose()
        A_adj = sparse.spdiags(temp_adj, [-1,0,1], pts_adj, pts_adj, format = "csr")
        # Discretize derivatives
        db_dlam1 = xval_adj*(0.5*h)**2/k*np.exp(alpha*xval_adj)
        dA_dlam1 = np.zeros(A_adj.shape)#sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")
        db_dlam2 =-(0.5*h)**2/(k**2)*np.exp(alpha*xval_adj)
        dA_dlam2 = np.zeros(A_adj.shape)
        # Create our approximate solution u_sol by 7 steps of CG(no preconditioner)
        # Solve the model.
        u_sol = splinalg.spsolve(A,b)

        # Interpolate onto finer mesh
        u_sol_interp = np.zeros((A_adj.shape[0],))
        u_sol_interp[1:-1:2] = u_sol
        u_sol_interp[0] = 0.5*u_sol_interp[1]
        u_sol_interp[-1] = 0.5*u_sol_interp[-2]
        u_sol_interp[2:-1:2] = 0.5*(u_sol_interp[1:-2:2] + u_sol_interp[3:-1:2])
        u_sol = u_sol_interp

        # Now set up the adjoint problem and solve it("exactly").
        # Define adjoint rhs corresponding with averaging u over (a,b)
        psi = np.zeros((pts_adj,1))
        ag0 = np.searchsorted(xval_adj, ia)
        ag1 = np.searchsorted(xval_adj, ib)
        psi[ag0-1] =  (xval_adj[ag0]-ia)**2/h
        psi[ag0] = 0.25*h +(xval_adj[ag0]-ia)/h*(0.5*h+ia-xval_adj[ag0-1])
        psi[ag1-1] = 0.25*h + (ib-xval_adj[ag1-1])/h*(0.5*h+xval_adj[ag1]-ib)
        psi[ag1] = (ib-xval_adj[ag1-1])**2/h
        psi[ag0+1:ag1-1] = 0.5*h 
        psi = psi/(ib-ia)

        # Atop=A'; Atop=A^\top not needed as A is symetric
        # Solve for adjoint solution using slash for "truth"
        phi = splinalg.spsolve(A_adj,psi)

        # Compute residual vector
        Res=b_adj-A_adj.dot(u_sol)

        # Estimate Error = \f$(R(U),\phi)\f$=(b-AU,\phi)=(b-AU)^\top\phi\f$.
        error_estimates[i,0] = np.dot(Res,phi)

        # Compute QoI
        values[i,0] = np.dot(u_sol, psi)

        # Compute derivatives
        dRes_dlam = db_dlam1 - dA_dlam1.dot(u_sol)
        jacobians[i,0,1] = np.dot(dRes_dlam, phi)
        dRes_dlam = db_dlam2 - dA_dlam2.dot(u_sol)
        jacobians[i,0,0] = np.dot(dRes_dlam, phi)
        

    return (values, error_estimates, jacobians)


def lb_model0(input_data):
    return model_approx(input_data, h=0.2)

def lb_model1(input_data):
    return model_approx(input_data, h=0.1)

def lb_model2(input_data):
    return model_approx(input_data, h=0.05)

def lb_model3(input_data):
    return model_approx(input_data, h=0.025)

def lb_model4(input_data):
    return model_approx(input_data, h=0.0125)

def lb_model5(input_data):
    return model_approx(input_data, h=0.00625)

def predict_model_exact(input_data):
    '''
    Evaluate prediction model exactly.
    '''
    #a = ia
    #b = ib
    # intialize outputs
    #num_runs = input_data.shape[0]
    #values = np.zeros((num_runs, 1))
    #jacobians = np.zeros((num_runs, 1, 2))
    #error_estimates = np.zeros((num_runs, 1))
    k = input_data[:,0]
    alpha = input_data[:,1]
    f = 1.0/(k*alpha**2) * (0.5 - np.exp(0.5*alpha) + 0.5*np.exp(alpha))
    return f
