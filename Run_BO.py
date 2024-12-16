# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:17:12 2023

@author: weizh
"""

import numpy as np
import os
import sys
import pandas as pd
import time
import matplotlib.pyplot as plt
from BO_functions import *
from general_functions import *
import pickle

time_start = time.time()

#Define some parameters
input_dim = 5
ini_popu_size = 50  # initial population size
tolerance_BO = 0.001
max_iter = 150
stop_num = 150
top_num = 10
stay_num = 3*input_dim

# this parameter controls the biased coefficient in objective function.
# >1 means stress-emphasized. <1 means strain-emphasized.
biased_coeff = 1  

lowest = [1, 1, 0.1, 0.1, 0.1]
maxiter_EI = 2000
tolerance_optimizer = 1e-4  # Convergence tolerance for L-BFGS-B

# Specify the pickle filename
pickle_filename = "valid_plateau_ini50_ave.pkl"

db_directory = r"C:\Users\tangmingjian\Desktop\Nacre Inverse Design\DB_5param_small_50_database\DB_5param_small_50_database"
db_csv_name = "DB_5param_small_50.csv"
db_csv_path = os.path.join(db_directory, db_csv_name)
local_directory = r"C:\Users\tangmingjian\Desktop\Nacre Inverse Design\BO code"

abaqus_script = "Nacre_2D_cae_txt.py"

# Information needs to be stored
new_curves = []
ini_curves = []
objectives = []
best_so_far = []
new_X = []

# 1. Define Objective function
# Create the target curve. 
tar_txt_name = 'target_data.txt'
tar_curve = extract_curve_txt(local_directory, tar_txt_name)


# 2. Surrogate model with GPflow
df = pd.read_csv(db_csv_path, header=None)
df = np.array(df)
df = df[0:ini_popu_size, :]
            
gp_input = df
    
ini_obj = []    
for i in range(ini_popu_size):
    txt_name = 'Nacre_2D_db_run' + str(i) + '.txt'
    current_curve = extract_curve_txt(db_directory, txt_name)
    ini_curves.append(current_curve)
    curve_diff = obj_average(tar_curve, current_curve, biased_coeff)
    ini_obj.append(curve_diff)

ini_obj = np.array(ini_obj)

gp_output = np.array(ini_obj).reshape((-1, 1))

gp_input_stdd, gp_input_mean, gp_input_std = data_stdd(gp_input)
gp_output_stdd, gp_output_mean, gp_output_std = data_stdd(gp_output)

# Plot the target curve and best curve in the initial population
ini_best_index = np.argmin(gp_output)
ini_best_name = 'Nacre_2D_db_run' + str(ini_best_index) + '.txt'
ini_best_curve = extract_curve_txt(db_directory, ini_best_name)

plot_best_curves(ini_best_curve, tar_curve, 0)
plot_initial_population(tar_curve, ini_curves)
# print('ini best index is', ini_best_index)

new_curves.append(ini_best_curve)
objectives.append(np.min(gp_output))
new_X.append(gp_input[np.argmin(gp_output)])

Y_previous_next = np.min(gp_output)

#%% Define the GP model
print("gp_input shape is ", gp_input_stdd.shape)
print("gp_output shape is ", gp_output_stdd.shape)
model = Train_GPR(gp_input_stdd, gp_output_stdd, input_dim)

#%% Bayesian Optimization loop   
stop_count = 0
expand_count = 0
optimal_Y_pre = 5
lower_bound = np.array([40.0, 40, 4, 4, 10])
upper_bound = np.array([120.0, 120, 8, 8, 40])
lbs = np.array([40.0, 40, 4, 4, 10])
ubs = np.array([120.0, 120, 8, 8, 40])

# update optimization iterations
for iteration in range(1, max_iter+1):
    
    print(f'---------- in iteration {iteration} ----------------')
    
    optimal_X = model.data[0].numpy()[np.argmin(model.data[1].numpy())] * gp_input_std + gp_input_mean
    optimal_Y = np.min(model.data[1].numpy()) * gp_output_std + gp_output_mean
    best_so_far.append(optimal_Y)
    print('current optimal X is', optimal_X)
    print('current optimal Y is', optimal_Y)
    
    # stopping criteria
    if np.abs(optimal_Y - optimal_Y_pre) < tolerance_BO:       
        stop_count += 1
        if stop_count == stop_num:
            print("===============================================")
            print(f"Converged after {iteration} iterations.")
            break
        else:
            pass
    else:
        stop_count = 0
    optimal_Y_pre = optimal_Y

    
    # Expansion of the design space. expansion factor is dropping. 
    if iteration % stay_num == 1:
        # update the bounds, and then record them.
        expand_count = iteration // stay_num + 1
        
        if expand_count > 1:
            X_new_batch = new_X[iteration-stay_num : iteration]
            lower_bound, upper_bound = update_space(lower_bound, upper_bound, X_new_batch, input_dim)
            lbs = np.vstack([lbs, lower_bound])
            ubs = np.vstack([ubs, upper_bound])
        
        # new expansion of the bounds.
        new_lower_bound, new_upper_bound = expand_space(lowest, lower_bound, upper_bound, input_dim, expand_count)
    
    new_lb_stdd = (new_lower_bound - gp_input_mean) / gp_input_std
    new_ub_stdd = (new_upper_bound - gp_input_mean) / gp_input_std
    
    # Acquisition function to choose the next point to sample
    X_next = acq_max_scipy(new_lb_stdd, new_ub_stdd, model, tolerance_optimizer, maxiter_EI, input_dim)    
       
    X_real_next = X_next * gp_input_std + gp_input_mean
    
    # modify if parameter values are out of bounds.
    for j in range(input_dim):
        if X_real_next[j] < new_lower_bound[j]:
            X_real_next[j] = new_lower_bound[j]
        if X_real_next[j] > new_upper_bound[j]:
            X_real_next[j] = new_upper_bound[j]
  
    new_X.append(X_real_next)
    print('the new design point is', X_real_next)
    
    job_name = 'Nacre_2D'
    txt_name = job_name + '_ite' + str(iteration) + '.txt'
    txt_path = os.path.join(local_directory, txt_name)
    
    # pass next sample values to ABAQUS scripts, and run ABAQUS
    try:
        run_abaqus_locally(iteration, X_real_next[0], X_real_next[1], X_real_next[2], X_real_next[3], X_real_next[4], abaqus_script)
        current_curve = extract_curve_txt(local_directory, txt_name)
        new_curves.append(current_curve)
        # Evaluate objective function at the new design point
        Y_next = obj_average(tar_curve, current_curve, biased_coeff)
            
    except:
        current_curve = ini_best_curve       
        new_curves.append(current_curve)      
        Y_next = 5

    objectives.append(Y_next)
    
    print('the new objective value is', Y_next)
    
    Y_previous_next = Y_next

    # Update model with new data point
    gp_input = np.vstack([gp_input, X_real_next])
    gp_output = np.vstack([gp_output, Y_next])

    gp_input_stdd, gp_input_mean, gp_input_std = data_stdd(gp_input)
    gp_output_stdd, gp_output_mean, gp_output_std = data_stdd(gp_output)
    
    model = Train_GPR(gp_input_stdd, gp_output_stdd, input_dim)


#%% Results acquisition
optimal_X_final = model.data[0].numpy()[np.argmin(model.data[1].numpy())] * gp_input_std + gp_input_mean
optimal_Y_final = np.min(model.data[1].numpy()) * gp_output_std + gp_output_mean
best_so_far.append(optimal_Y_final)
objectives = np.array(objectives)
best_index = np.argmin(objectives)

# plot_all_curves(curves)
plot_best_curves(new_curves[best_index], tar_curve, best_index)

# plot the convergence plot
plt.figure()
plt.plot(objectives)
# plt.title("objective of the new design in each iteration")
plt.xlabel('iterations')
plt.ylabel('new objective')
plt.savefig(fname="objective of the new design")

# plot best value so far
plt.figure()
plt.plot(best_so_far)
# plt.title("best objective so far over iterations")
plt.xlabel('iterations')
plt.ylabel('best objective')
plt.savefig(fname="best objective values so far")

new_X = np.array(new_X)
best_so_far = np.array(best_so_far)

# find the top 10 best designs and plot them. 
top_indices = np.argsort(objectives)[:top_num]
top_X = new_X[top_indices]
top_Y = np.sort(objectives)[:top_num]
print(f"the top {top_num} designs are")
print(top_X)
print(f"the top {top_num} design objectives are")
print(top_Y)
plot_top_curves(top_num, tar_curve, new_curves, top_indices, objectives)

pairplot(new_X)

print('lbs is', lbs)
print('ubs is', ubs)
plot_bounds(lbs, ubs, input_dim)


# output the best design point and corresponding objective value. 
print("===============================================")
print("the best design is", optimal_X_final)
print("the best objective value is", optimal_Y_final)

entire_time = time.time() - time_start
print('the entire CPU time is', entire_time)

data = {
    "tar_curve": tar_curve,
    "ini_curves" : ini_curves, 
    "ini_best_index" : ini_best_index,
    "ini_best_curve" : ini_best_curve,
    "ini_obj" : ini_obj,
    "new_X" : new_X,    
    "new_curves" : new_curves,
    "objectives" : objectives,
    "best_so_far" : best_so_far,
    "best_index" : best_index,
    "best_curve" : new_curves[best_index],
    "optimal_X_final" : optimal_X_final, 
    "optimal_Y_final" : optimal_Y_final,
    "top_indices" : top_indices,
    "top_X" : top_X,
    "top_Y" : top_Y,
    "lbs" : lbs,
    "ubs" : ubs,
    "entire_time" : entire_time   
}

# Writing to a pickle file
with open(pickle_filename, 'wb') as file:
    pickle.dump(data, file)
    