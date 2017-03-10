import os
import scipy.io as sio
import sys
import numpy as np
import numpy.random as rand
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg
from math import *

def lb_model_exact(input_data):
    num_runs = input_data.shape[0]
    values = np.zeros((num_runs, 1))
    jacobians = np.zeros((num_runs, 1, 2))
    error_estimates = np.zeros((num_runs, 1))

    k = input_data[:,0]
    alpha = input_data[:,1]
    x = 0.5
    values[:,0] = (-np.exp(alpha * x) - x + x * np.exp(alpha) + 1) / k / alpha ** 2
    jacobians[:, 0, 0] = -(-np.exp(alpha * x) - x + x * np.exp(alpha) + 1) / k ** 2 / alpha ** 2

    jacobians[:, 0 , 1] = ((-alpha * x + 2) * np.exp(alpha * x) + x * (alpha - 2) * np.exp(alpha) + 2 * x - 2) / k / alpha ** 3
    return (values, error_estimates, jacobians)


def lb_model(input_data, h):
    """
    Approximate map.
    """
    #h = 0.1
    num_runs = input_data.shape[0]
    num_runs_dim = input_data.shape[1]
    
    values = np.zeros((num_runs, 1))
    jacobians = np.zeros((num_runs, 1, 2))
    error_estimates = np.zeros((num_runs, 1))

    for i in range(num_runs):
        alpha  = input_data[i,0]
        k = input_data[i, 0]
        #h=.05
        xval = np.arange(h, 1.0, h)
        pts=len(xval)

        xval_adj = np.arange(0.5*h, 1.0, 0.5*h)
        pts_adj = len(xval_adj)
        # Setup Forward Solution:

        # Discretize ODE: \f[ -u^\prime\prime(x_n)= \frac{2u(x_n)-u(x_{n-1})-u(x_{n+1})}{h^2} = e^{\alpha x_n} \f]

        # Uniform grid so grid size h can be moved to RHS
        b = h**2/k*np.exp(alpha*xval)
        b_adj = (0.5*h)**2/k*np.exp(alpha*xval_adj)
        # We use the spdiags command to map -1 2 1 to the tridiagonal matrix A
        temp = np.hstack((-np.ones((pts,1)), 2.0*np.ones((pts,1)), -np.ones((pts,1)))).transpose()
        A = sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")

        temp_adj = np.hstack((-np.ones((pts_adj,1)), 2.0*np.ones((pts_adj,1)), -np.ones((pts_adj,1)))).transpose()
        A_adj = sparse.spdiags(temp_adj, [-1,0,1], pts_adj, pts_adj, format = "csr")

        db_dlam1 = xval_adj*(0.5*h)**2/k*np.exp(alpha*xval_adj)
        dA_dlam1 = np.zeros(A_adj.shape)#sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")
        db_dlam2 =-(0.5*h)**2/(k**2)*np.exp(alpha*xval_adj)
        dA_dlam2 = np.zeros(A_adj.shape)
        # Create our approximate solution u_sol by 7 steps of CG(no preconditioner)
        #(u_sol,_) = splinalg.cg(A, b,tol=1.0e-20, maxiter=9)
        # Solve for "truth" solve.
        u_sol = splinalg.spsolve(A,b)
        u_sol_interp = np.zeros((A_adj.shape[0],))
        u_sol_interp[1:-1:2] = u_sol
        u_sol_interp[0] = 0.5*u_sol_interp[1]
        u_sol_interp[-1] = 0.5*u_sol_interp[-2]
        u_sol_interp[2:-1:2] = 0.5*(u_sol_interp[1:-2:2] + u_sol_interp[3:-1:2])
        u_sol = u_sol_interp
        #import pdb
        #pdb.set_trace()

        # Now set up the adjoint problem and solve it("exactly").
        # QoI is u_sol(10), so \f$ \psi = e_{10}\f$.
        psi = np.zeros((pts_adj,1))
        index = int(floor(u_sol_interp.shape[0]/2.0))
        psi[index] = 1.0

        # Atop=A'; Atop=A^\top not needed as A is symetric
        # Solve for adjoint solution using slash for "truth"
        phi = splinalg.spsolve(A_adj,psi)

        # Compute residual vector
        Res=b_adj-A_adj.dot(u_sol)

        dRes_dlam = db_dlam1 - dA_dlam1.dot(u_sol)
        # Estimated Error = \f$(R(U),\phi)\f$=(b-AU,\phi)=(b-AU)^\top\phi\f$.
        error_estimates[i,0] = np.dot(Res,phi)
        values[i,0] = u_sol[index]
        jacobians[i,0,1] = np.dot(dRes_dlam, phi)
        dRes_dlam = db_dlam2 - dA_dlam2.dot(u_sol)
        jacobians[i,0,0] = np.dot(dRes_dlam, phi)
                            
        

    return (values, error_estimates, jacobians)

def lb_model2(input_data):
    """
    Approximate map.
    """

    num_runs = input_data.shape[0]
    num_runs_dim = input_data.shape[1]
    
    values = np.zeros((num_runs, 1))
    jacobians = np.zeros((num_runs, 1, 1))
    error_estimates = np.zeros((num_runs, 1))

    for i in range(num_runs):
        alpha  = input_data[i,:]
        h=.05
        xval = np.arange(h, 1.0, h)
        pts=len(xval)
        # Setup Forward Solution:

        # Discretize ODE: \f[ -u^\prime\prime(x_n)= \frac{2u(x_n)-u(x_{n-1})-u(x_{n+1})}{h^2} = e^{\alpha x_n} \f]

        # Uniform grid so grid size h can be moved to RHS
        b = h**2*np.exp(alpha*xval)
        # We use the spdiags command to map -1 2 1 to the tridiagonal matrix A
        temp = np.hstack((-np.ones((pts,1)), 2.0*np.ones((pts,1)), -np.ones((pts,1)))).transpose()
        A = sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")

        db_dlam = xval*h**2*np.exp(alpha*xval)
        dA_dlam = np.zeros(A.shape)#sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")
        # Create our approximate solution u_sol by 7 steps of CG(no preconditioner)
        (u_sol,_) = splinalg.cg(A, b,tol=1.0e-20, maxiter=12)
        # Solve for "truth" solve.
        u = splinalg.spsolve(A,b)

        # Now set up the adjoint problem and solve it("exactly").
        # QoI is u_sol(10), so \f$ \psi = e_{10}\f$.
        psi = np.zeros((pts,1))
        psi[9] = 1

        # Atop=A'; Atop=A^\top not needed as A is symetric
        # Solve for adjoint solution using slash for "truth"
        phi = splinalg.spsolve(A,psi)

        # Compute residual vector
        Res=b-A.dot(u_sol)

        dRes_dlam = db_dlam - dA_dlam.dot(u_sol)
        # Estimated Error = \f$(R(U),\phi)\f$=(b-AU,\phi)=(b-AU)^\top\phi\f$.
        error_estimates[i,0] = np.dot(Res,phi)
        values[i,0] = u_sol[9]
        jacobians[i,0,0] = np.dot(dRes_dlam, phi)
        

    return (values, error_estimates, jacobians)

def lb_model3(input_data):
    """
    Approximate map.
    """

    num_runs = input_data.shape[0]
    num_runs_dim = input_data.shape[1]
    
    values = np.zeros((num_runs, 1))
    jacobians = np.zeros((num_runs, 1, 1))
    error_estimates = np.zeros((num_runs, 1))

    for i in range(num_runs):
        alpha  = input_data[i,:]
        h=.05
        xval = np.arange(h, 1.0, h)
        pts=len(xval)
        # Setup Forward Solution:

        # Discretize ODE: \f[ -u^\prime\prime(x_n)= \frac{2u(x_n)-u(x_{n-1})-u(x_{n+1})}{h^2} = e^{\alpha x_n} \f]

        # Uniform grid so grid size h can be moved to RHS
        b = h**2*np.exp(alpha*xval)
        # We use the spdiags command to map -1 2 1 to the tridiagonal matrix A
        temp = np.hstack((-np.ones((pts,1)), 2.0*np.ones((pts,1)), -np.ones((pts,1)))).transpose()
        A = sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")

        db_dlam = xval*h**2*np.exp(alpha*xval)
        dA_dlam = np.zeros(A.shape)#sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")
        # Create our approximate solution u_sol by 7 steps of CG(no preconditioner)
        #(u_sol,_) = splinalg.cg(A, b,tol=1.0e-20, maxiter=100)
        # Solve for "truth" solve.
        u_sol = splinalg.spsolve(A,b)

        # Now set up the adjoint problem and solve it("exactly").
        # QoI is u_sol(10), so \f$ \psi = e_{10}\f$.
        psi = np.zeros((pts,1))
        psi[9] = 1

        # Atop=A'; Atop=A^\top not needed as A is symetric
        # Solve for adjoint solution using slash for "truth"
        phi = splinalg.spsolve(A,psi)

        # Compute residual vector
        Res=b-A.dot(u_sol)

        dRes_dlam = db_dlam - dA_dlam.dot(u_sol)
        # Estimated Error = \f$(R(U),\phi)\f$=(b-AU,\phi)=(b-AU)^\top\phi\f$.
        error_estimates[i,0] = np.dot(Res,phi)
        values[i,0] = u_sol[9]
        jacobians[i,0,0] = np.dot(dRes_dlam, phi)
        

    return (values, error_estimates, jacobians)
