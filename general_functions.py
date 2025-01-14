# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:54:30 2023

@author: weizh
"""

import numpy as np
import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sea
# from matplotlib.animation import FuncAnimation


np.set_printoptions(threshold=sys.maxsize)

# define parameters from ABAQUS model for stress and strain calculation
length = 8.4
thick = 1.0
area = length * thick

x_limit = 2.0
y_limit = 225.0

def extract_curve_txt(directory, txt_name):
    
    directory_txt =  os.path.join(directory, txt_name)       
    x = []
    y = []   
    file_hold = open(directory_txt, 'r')    
    for count, line in enumerate(file_hold):       
        if count < 4:
            pass        
        else:
            try:
                lines = [i for i in line.split()]
                x.append(float(lines[2])*100/length)   # in unit of %.
                if float(lines[1]) < 0:
                    lines[1] = 0
                y.append(float(lines[1])*10**6/area)     # in unit of MPa.
            except:
                pass
    x = np.array(x[:-1]).reshape(-1, 1)     # exclude the last point of the curve
    y = np.array(y[:-1]).reshape(-1, 1)     # exclude the last point of the curve
    curve = np.hstack((x, y))
    return curve


def extract_curve_csv(directory, csv_name):
    
    csv_path = os.path.join(directory, csv_name)      
    data = pd.read_csv(csv_path)
    stress = np.array(data['stress (MPa)'].tolist()).reshape(-1, 1)   
    strain = np.array(data['strain (%)'].tolist()).reshape(-1, 1)
    curve = np.hstack((strain, stress))
    
    return curve



# Data preprocessing, normalization
def data_stdd(data):
  
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    data_stdd = (data - mean)/std
    # print('data normalized is', data_norm)
    # print('means are', mean, mean.shape)
    # print('standard deviation are', std, std.shape)
    return data_stdd, mean, std


def plot_initial_population(tar_curve, ini_curves):
    
    plt.figure(figsize=(12, 8))
    plt.plot(tar_curve[:,0], tar_curve[:,1], 'k--', label = "the target curve")
    
    for i in range(len(ini_curves)):
        plt.plot(ini_curves[i][:,0], ini_curves[i][:,1])

    plt.xlabel('strains (%)', fontsize = 20)
    plt.ylabel('stress (MPa)', fontsize = 20)
    plt.xlim([0, x_limit])
    plt.ylim([0, y_limit])
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(loc ='best', fontsize=12, framealpha=0)
    # plt.title("initial population and target curve", fontsize = 25)
    plt.savefig(fname = "Initial_curves")



def plot_best_curves(current_curve, target_curve, iteration):
    
    plt.figure()
    if iteration == 0:
        curve_label = "best curve initially"
        fig_title = "best design initially vs target curve"
    else:
        curve_label = f"the new curve in ite{iteration}"
        fig_title = f"optimal and target curves after {iteration} iterations"
    plt.plot(current_curve[:,0], current_curve[:,1], label = curve_label)
    plt.plot(target_curve[:, 0], target_curve[:, 1], 'k--', label = "the target curve")
    plt.xlabel('strains (%)')
    plt.ylabel('stress (MPa)')
    plt.xlim([0, x_limit])
    plt.ylim([0, y_limit])
    plt.legend(loc ='best', fontsize='small', framealpha=0)
    # plt.title(fig_title)
    plt.savefig(fname=fig_title)

    

def plot_all_curves(curves):
    
    plt.figure(figsize=(12, 8))
    for i in range(len(curves)):
        if i == 0:
            plt.plot(curves[i][:,0], curves[i][:,1], 'k--', label = "the target curve")
        elif i == 1:
            plt.plot(curves[i][:,0], curves[i][:,1], label = "initial population")
        else:
            plt.plot(curves[i][:,0], curves[i][:,1], label = f"ite {i-1}")

    plt.xlabel('strains (%)', fontsize = 20)
    plt.ylabel('stress (MPa)', fontsize = 20)
    plt.xlim([0, x_limit])
    plt.ylim([0, y_limit])
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    # plt.legend(fontsize = 10, loc = 1, ncol=3)
    plt.legend(loc ='best', fontsize='small', framealpha=0)
    # plt.title("Evoluation of the best curves", fontsize = 25)
    plt.savefig(fname = "Evoluation of the best curves")    
        

def plot_top_curves(top_num, tar_curve, new_curves, top_indices, objectives):
    
    plt.figure(figsize=(9, 6))
    
    plt.plot(tar_curve[:,0], tar_curve[:,1], 'k--', label = "target")
    
    for i in top_indices:       
        obj = objectives[i]
        plt.plot(new_curves[i][:,0], new_curves[i][:,1], label = f"ite{i},obj={obj:.2f}")

    plt.xlabel('strains (%)', fontsize = 15)
    plt.ylabel('stress (MPa)', fontsize = 15)
    plt.xlim([0, x_limit])
    plt.ylim([0, y_limit])
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(loc ='best', fontsize='small', framealpha=0)
    plt.title(f"the best {top_num} curves", fontsize = 20)
    plt.savefig(fname = f"the best {top_num} curves") 

        
  
def plot_bounds(lbs, ubs, input_dim):
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(16, 12))
    # fig.suptitle('Update of the design space')
    x_values = range(1, ubs[:, 0].shape[0] + 1)
    
    ax1.plot(x_values, ubs[:, 0], label = "ub of para1")
    ax1.plot(x_values, lbs[:, 0], label = "lb of para1")
    # ax1.set_title('space of parameter 1')
    ax1.legend(loc='center right', framealpha=0)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    ax2.plot(x_values, ubs[:, 1], label = "ub of para2") 
    ax2.plot(x_values, lbs[:, 1], label = "lb of para2")
    # ax2.set_title('space of parameter 2')
    ax2.legend(loc='center right', framealpha=0)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    ax3.plot(x_values, ubs[:, 2], label = "ub of para3")    
    ax3.plot(x_values, lbs[:, 2], label = "lb of para3")
    # ax3.set_title('space of parameter 3')
    ax3.legend(loc='center right', framealpha=0)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    ax4.plot(x_values, ubs[:, 3], label = "ub of para4")    
    ax4.plot(x_values, lbs[:, 3], label = "lb of para4")
    # ax3.set_title('space of parameter 4')
    ax4.legend(loc='center right', framealpha=0)
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    ax5.plot(x_values, ubs[:, 4], label = "ub of para5")    
    ax5.plot(x_values, lbs[:, 4], label = "lb of para5")
    # ax3.set_title('space of parameter 4')
    ax5.legend(loc='center right', framealpha=0)
    ax5.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax5.set_xlabel('expasion count')
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.4)
        
    fig.savefig('Update of the design space.png')
    
    
    
def save_to_txt(directory, file_name, array): 
    
    file_path = os.path.join(directory, file_name)
    with open(file_path, "ab") as f:
        f.write(b"\n")
        np.savetxt(f, array, delimiter=',', fmt='%1.4f')
    

def pairplot(new_X):
    
    data = pd.DataFrame(new_X, columns=['var1', 'var2', 'var3', 'var4', 'var5'])
    sea.pairplot(pd.DataFrame(data), corner=True)
    plt.savefig(fname = "pairplot of the design variables")
    
    
def run_abaqus(param1, param2, param3, param4, param5, param6, directory, inp_file_path, job_name, abaqus_script):
    command = f"abaqus cae noGUI={abaqus_script}"
    parameters = f'{param1}, {param2}, {param3}, {param4}, {param5}, {param6}, {directory}, {inp_file_path}, {job_name}'
    os.environ['ABAQUS_PARAMS'] = parameters

    # Use subprocess to call the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ, bufsize = 0)
    stdout, stderr = process.communicate()

    # Print the output and error to the console
    print(stdout.decode())
    if stderr:
        print("Error:")
        print(stderr.decode())
        
        
        
def run_abaqus_locally(param1, param2, param3, param4, param5, param6, directory, inp_file_path, job_name, abaqus_script):
    
    # define the abaqus bat file path
    abaqus_cmd = f'C:\\SIMULIA\\Commands\\abaqus.bat'
    
    # create environment parameters. pass parameter values to it.
    parameters = f'{param1}, {param2}, {param3}, {param4}, {param5}, {param6}, {directory}, {inp_file_path}, {job_name}'
    os.environ['ABAQUS_PARAMS'] = parameters
    
    # Create the command to run Abaqus with a Python script
    command = abaqus_cmd + " cae noGUI=" + abaqus_script

    # Use subprocess to call the command, with environment value.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ, bufsize = 0)
    # process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to complete and get the output and error
    stdout, stderr = process.communicate()

    # Print the output and error to the console
    print(stdout.decode())
    if stderr:
        print("Error:")
        print(stderr.decode())
        