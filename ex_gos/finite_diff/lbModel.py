import os
import scipy.io as sio
import sys
import numpy as np
import numpy.random as rand
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg
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

        db_dlam = alpha*h**2*np.exp(alpha*xval)
        dA_dlam = np.zeros(A.shape)#sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")
        # Create our approximate solution u_sol by 7 steps of CG(no preconditioner)
        (u_sol,_) = splinalg.cg(A, b,tol=1.0e-20, maxiter=7)
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

        db_dlam = alpha*h**2*np.exp(alpha*xval)
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

        db_dlam = alpha*h**2*np.exp(alpha*xval)
        dA_dlam = np.zeros(A.shape)#sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")
        # Create our approximate solution u_sol by 7 steps of CG(no preconditioner)
        (u_sol,_) = splinalg.cg(A, b,tol=1.0e-20, maxiter=100)
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
