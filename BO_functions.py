# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:54:30 2023

@author: weizh
"""

import numpy as np
import gpflow
from gpflow.optimizers import Scipy
import os
import sys
from scipy import special
from scipy.stats import norm
from scipy.optimize import minimize
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

np.set_printoptions(threshold=sys.maxsize)

def curve_normalize(curve, max_strain, max_stress):
    
    X = curve[:, 0]/max_strain
    Y = curve[:, 1]/max_stress
    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    curve_norm = np.hstack((X, Y))
 
    return curve_norm

    
def interp_curve_space(curve, spacing=0.02):

    # Compute the cumulative distance along the curve
    x, y = curve[:, 0], curve[:, 1]
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)  # individual distances

    tot_dist = np.sum(distances)
    cum_distances = np.append(np.zeros(1), np.cumsum(distances))

    # Create an interpolation function for x and y coordinates
    fx = interp1d(cum_distances, x, kind='linear')
    fy = interp1d(cum_distances, y, kind='linear')

    # Generate new points at regular intervals along the curve length  
    num_points = int(tot_dist / spacing) + 1
    new_distances = np.linspace(0, tot_dist*0.999, num_points)

    new_x = fx(new_distances)
    new_y = fy(new_distances)

    return np.column_stack((new_x, new_y))


def match_strain(curve1, curve2):
    # Find the max x-value for both curves
    max_x1 = np.max(curve1[:, 0])
    max_x2 = np.max(curve2[:, 0])
    max_x = max(max_x1, max_x2)
    
    # Extend the smaller curve
    if max_x1 < max_x:
        # Extend curve1
        add_point = np.array([max_x, curve1[-1, 1]])
        curve1 = np.vstack((curve1, add_point))
    elif max_x2 < max_x:
        # Extend curve2
        add_point = np.array([max_x, curve2[-1, 1]])
        curve2 = np.vstack((curve2, add_point))

    return curve1, curve2



# This objective function calculates the average minimum distance on all 
# the points of the two curves.
def obj_average(tar_curve, current_curve, biased_coeff):
    
    max_strain = np.max(tar_curve[:, 0])
    max_stress = np.max(tar_curve[:, 1])

    tar_norm = curve_normalize(tar_curve, max_strain, max_stress/np.sqrt(biased_coeff))
    current_norm = curve_normalize(current_curve, max_strain, max_stress/np.sqrt(biased_coeff))
    
    tar_norm, current_norm = match_strain(tar_norm, current_norm)
    
    tar_curve_interp = interp_curve_space(tar_norm)
    current_curve_interp = interp_curve_space(current_norm)
    
    # Compute the pairwise distances between all points in the two curves
    distances = cdist(tar_curve_interp, current_curve_interp)

    # Find the minimum distance for each point in curve1 to any point in curve2
    min_distances_1 = np.min(distances, axis=1)
    
    # Find the minimum distance for each point in curve2 to any point in curve1
    min_distances_2 = np.min(distances, axis=0)

    # Calculate the average of these minimum distances
    avg_distance = (np.mean(min_distances_1) + np.mean(min_distances_2)) / 2

    return avg_distance



def obj_HD(tar_curve, current_curve, biased_coeff):
    
    max_strain = np.max(tar_curve[:, 0])
    max_stress = np.max(tar_curve[:, 1])

    tar_norm = curve_normalize(tar_curve, max_strain, max_stress/np.sqrt(biased_coeff))
    current_norm = curve_normalize(current_curve, max_strain, max_stress/np.sqrt(biased_coeff))
    
    tar_norm, current_norm = match_strain(tar_norm, current_norm)
    
    tar_curve_interp = interp_curve_space(tar_norm)
    current_curve_interp = interp_curve_space(current_norm)
    
    # Hausdorff distance as the curve difference
    DH1 = directed_hausdorff(tar_curve_interp, current_curve_interp)[0]
    DH2 = directed_hausdorff(current_curve_interp, tar_curve_interp)[0]
    
    objective = max(DH1, DH2)
        
    return objective



# curve area calculation
def curve_area(curve):
    
    curve_interp = interp_curve_space(curve)
    
    X = curve_interp[:, 0]
    Y = curve_interp[:, 1]
    
    area = np.trapz(Y, X)
    
    return area


# the curve area
def objective_area(tar_area, current_curve):
    
    current_area = curve_area(current_curve)
    objective = abs(tar_area - current_area)
        
    return objective



## GPflow version of 1.3.0
def Train_GPR(gp_input, gp_output, input_dim):
    # Kern = gpflow.kernels.Matern52(input_dim=3)
    k1 = gpflow.kernels.RBF(lengthscales=[1.0]*input_dim)
    # k2 = gpflow.kernels.Linear(variance=1.0, active_dims=None)
    
    k2 = gpflow.kernels.Matern52(variance=1.0, lengthscales=[1.0]*input_dim)
    
    # Create the GP regression model
    model = gpflow.models.GPR(data=(gp_input, gp_output), kernel=k2, mean_function=None)
    
    def objective_closure():
        return -model.log_marginal_likelihood()
    # Training using ScipyOptimizer
    opt = Scipy()
    # opt.minimize(model.training_loss, model.trainable_variables)
    opt.minimize(objective_closure, model.trainable_variables)
    
    # Get the lengthscales
    lengthscales_value = model.kernel.lengthscales.numpy()
    print("Lengthscales:", lengthscales_value)
    
    # Get the variance
    variance_value = model.kernel.variance.numpy()
    print("Variance:", variance_value)
        
    return model


# 3. Define Acquisition function: 4 types
def probability_of_improvement(model, candidate, xi=0.0):
    mean, var = model.predict_f(candidate)
    mean, var = mean.numpy(), var.numpy() # Convert TensorFlow tensors to numpy arrays
    best_y = np.min(model.data[1].numpy())
    z = (best_y - mean - xi) / np.sqrt(var)
    PoI =  0.5 * (1.0 + special.erf(z / np.sqrt(2.0)))

    PoI_max = np.max(PoI)
    # print('Probabilities of improvement has shape of', PoI.shape)
    print('max PoI is', PoI_max)
    print('next design point is', candidate[np.argmax(PoI)], np.argmax(PoI))
    
    return candidate[np.argmax(PoI)]


def upper_confidence_bound(model, candidate, kappa=4.0):
    mean, var = model.predict_f(candidate)
    mean, var = mean.numpy(), var.numpy() # Convert TensorFlow tensors to numpy arrays
    ucb_value = mean - kappa * np.sqrt(var)
    print('the best ucb value is', np.min(ucb_value), np.argmin(ucb_value))
    return candidate[np.argmin(ucb_value)]


def expected_improvement_discretized(model, candidate, xi=0.0):
    mean, var = model.predict_f(candidate)
    mean, var = mean.numpy(), var.numpy() # Convert TensorFlow tensors to numpy arrays
    sigma = np.sqrt(var)
    f_minus = np.min(model.data[1].numpy())  # Updated this line for GPflow 2.x
    
    Z = np.zeros(mean.shape)
    mask = sigma > 0
    Z[mask] = (f_minus - mean[mask] + xi) / sigma[mask]
    
    ei_value = (xi - mean + f_minus) * norm.cdf(Z) + sigma * norm.pdf(Z)  
    return candidate[np.argmax(ei_value)]


# Expected Improvement acquisition function
def expected_improvement(x, model, epsilon=1e-08):
    # x = x.reshape(1, -1)
    mean, var = model.predict_f(x)
    std_dev = tf.math.sqrt(var)
    
    # Best observed value
    f_best = tf.math.reduce_min(model.data[1])
    
    improvement = f_best - mean
    Z = improvement / (std_dev + epsilon)
    
    # - is to maximize the EI number.
    return -((improvement * tf.math.erfc(-Z/np.sqrt(2)) / 2) + \
             (std_dev * tf.exp(-(Z**2)/2) / np.sqrt(2 * np.pi)))



def acq_max_scipy(new_lb_norm, new_ub_norm, model, tolerance, maxiter_EI, dim):
    """
    A function to find the maximum of the acquisition function using
    the scipy python
    """

    # Start with the lower bound as the argmax
    x_max = new_lb_norm
    max_acq = None

    # multi start
    for i in range(5*dim):
        # Find the minimum of the negative acquisition function        
        x_tries = np.random.uniform(new_lb_norm, new_ub_norm, size=(100*dim, dim))
    
        # evaluate        
        y_tries = expected_improvement(x_tries, model) 
        # print("y_tries shape is", y_tries.shape)
        
        #find x optimal for init
        idx_min=np.argmin(y_tries)
        # print("idx_min is", idx_min)
        x_init_min=x_tries[idx_min]
        # print("x_init_min is", x_init_min)
        
        result = minimize(
            fun=lambda x: expected_improvement(x.reshape(1, -1), model).numpy(),
            x0 = x_init_min,
            bounds=list(zip(new_lb_norm, new_ub_norm)),
            method='L-BFGS-B',
            options={'ftol': tolerance, 'maxiter': maxiter_EI},
        )       

        # # value at the estimated point
        val = result.fun
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val <= max_acq:
            x_max = result.x
            max_acq = val
            #print max_acq

    return x_max



# function to update the design space, decelerating expansion, with controller E.
def expand_space(lowest, lower_bound, upper_bound, input_dim, expand_count, E = 16):
    
    mid_points = (lower_bound + upper_bound) * 0.5
    
    ori_radii = upper_bound - mid_points
    
    alpha = np.power(E/expand_count, 1/input_dim)
    
    if alpha > 1:
        new_radii = alpha * ori_radii
    else: 
        new_radii = ori_radii
        
    new_lower_bound = mid_points - new_radii
    
    new_upper_bound = mid_points + new_radii
    
    #Add constraints to the lower bound for physics.
    for i in range(input_dim):        
        if new_lower_bound[i] < lowest[i]:
            new_lower_bound[i] = lowest[i]
    
    return new_lower_bound, new_upper_bound



def update_space(lower_bound, upper_bound, X_new_batch, input_dim):
    X_new_batch = np.array(X_new_batch)
    
    # Create copies of the lower_bound and upper_bound arrays
    lower = lower_bound.copy()
    upper = upper_bound.copy()
    
    for i in range(input_dim):
        if np.min(X_new_batch[:, i]) < lower[i]:
            lower[i] = np.min(X_new_batch[:, i])
        if np.max(X_new_batch[:, i]) > upper[i]:
            upper[i] = np.max(X_new_batch[:, i])
              
    return lower, upper

    