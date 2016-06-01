# Copyright (C) 2014-2016 The BET Development Team

"""
This example generates uniform random samples in the unit hypercube and
corresponding QoIs (data) generated by a linear map Q.  We then calculate the
gradients using an RBF scheme and use the gradient information to choose the
optimal set of 2 (3, 4, ... input_dim) QoIs to use in the inverse problem.

Every real world problem requires special attention regarding how we choose
*optimal QoIs*.  This set of examples (examples/sensitivity/linear) covers
some of the more common scenarios using easy to understand linear maps.

In this *volume_binsize_large* example we choose *optimal QoIs* to be the set of 
QoIs of size input_dim that produces the smallest support of the inverse 
solution, assuming we define the uncertainty in our data to be fixed, i.e.,
independent of the range of data maesured for each QoI (bin_size).
"""

import numpy as np
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cqoi
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.postTools as postTools
import bet.Comm as comm
import bet.sample as sample

# Let Lambda be a 5 dimensional hypercube
input_dim = 10
output_dim = 100
num_samples = 1E5
num_centers = 10

# Let the map Q be a random matrix of size (output_dim, input_dim)
np.random.seed(0)
Q = np.random.random([output_dim, input_dim])

# Initialize some sample objects we will need
input_samples = sample.sample_set(input_dim)
input_samples_centers = sample.sample_set(input_dim)
output_samples = sample.sample_set(output_dim)

# Choose random samples in parameter space to solve the model
input_samples.set_values(np.random.uniform(0, 1, [num_samples, input_dim]))

# Make the MC assumption and compute the volumes of each voronoi cell
input_samples.estimate_volume_mc()

# We will approximate the jacobian at each of the centers
input_samples_centers.set_values(input_samples.get_values()[:num_centers])

# Compute the output values with the map Q
output_samples.set_values(Q.dot(input_samples.get_values().transpose()).transpose())

# Calculate the gradient vectors at some subset of the samples.  Here the
# *normalize* argument is set to *False* because we are using bin_size to
# determine the uncertainty in our data.
input_samples.set_jacobians(grad.calculate_gradients_rbf(input_samples,
    output_samples, input_samples_centers, normalize=False))

# With these gradient vectors, we are now ready to choose an optimal set of
# QoIs to use in the inverse problem, based on minimizing the support of the
# inverse solution (volume).  The most robust method for this is
# :meth:~bet.sensitivity.chooseQoIs.chooseOptQoIs_large which returns the
# best set of 2, 3, 4 ... until input_dim.  This method returns a list of
# matrices.  Each matrix has 10 rows, the first column representing the
# expected inverse volume ratio, and the rest of the columns the corresponding
# QoI indices.
best_sets = cqoi.chooseOptQoIs_large(input_samples, max_qois_return=5,
    num_optsets_return=2, inner_prod_tol=0.9, measskew_tol=1E2, measure=True)

'''
We see here the expected volume ratios are small.  This number represents the
expected volume of the inverse image of a unit hypercube in the data space.
With the bin_size definition of the uncertainty in the data, here we expect to
see inverse solutions that have a smaller support (expected volume ratio < 1)
than the original volume of the hypercube in the data space.

This interpretation of the expected volume ratios is only valid for inverting
from a data space that has the same dimensions as the paramter space.  When
inverting into a higher dimensional space, this expected volume ratio is the
expected volume of the cross section of the inverse solution.
'''
###############################################################################

# At this point we have determined the optimal set of QoIs to use in the inverse
# problem.  Now we compare the support of the inverse solution using
# different sets of these QoIs.  We set Q_ref to correspond to the center of
# the parameter space.  We choose the set of QoIs to consider.

QoI_indices = [0, 7] # choose up to input_dim
#QoI_indices = [0, 1]
#QoI_indices = [0, 7, 34, 39, 90]
#QoI_indices = [0, 1, 2, 3, 4]

# Choose some QoI indices to solve the ivnerse problem with
output_samples._dim = len(QoI_indices)
output_samples.set_values(output_samples.get_values()[:, QoI_indices])

# Set the jacobians to None
input_samples.set_jacobians(None)

# Define the reference point in the output space to correspond to the center of
# the input space.
Q_ref = Q[QoI_indices, :].dot(0.5 * np.ones(input_dim))

# bin_ratio defines the uncertainty in our data
bin_ratio = 0.25

# Create discretization object
my_discretization = sample.discretization(input_sample_set=input_samples,
                                        output_sample_set=output_samples)


# Find the simple function approximation
simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
    data_set=my_discretization, Q_ref=Q_ref, rect_scale=bin_ratio,
    center_pts_per_edge = 1)

# Calculate probablities making the Monte Carlo assumption
calculateP.prob(my_discretization)

percentile = 1.0
# Sort samples by highest probability density and find how many samples lie in
# the support of the inverse solution.  With the Monte Carlo assumption, this
# also tells us the approximate volume of this support.
(num_samples, _, indices_in_inverse) =\
    postTools.sample_highest_prob(top_percentile=percentile,
    sample_set=input_samples,sort=True)

# Print the number of samples that make up the highest percentile percent
# samples and ratio of the volume of the parameter domain they take up
if comm.rank == 0:
    print (num_samples, np.sum(input_samples.get_volumes()[indices_in_inverse]))