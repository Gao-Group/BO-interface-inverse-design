from abaqus import *
from abaqusConstants import *
import __main__
import numpy as np
import os
import time
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior


def modify_inp(inp_path, sigma_c_n, sigma_c_s, delta_d_n, delta_d_s, delta_f):
    thick_ini = 0.025
    Enn = sigma_c_n * thick_ini / delta_d_n
    G1 = sigma_c_s * thick_ini / delta_d_s
    G2 = G1
    stress_normal = sigma_c_n
    stress_1st = sigma_c_s
    stress_2nd = sigma_c_n
    disp_f = delta_f
    
    Enn = np.round(Enn, decimals=8)
    G1 = np.round(G1, decimals=8)
    G2 = np.round(G2, decimals=8)
    stress_normal = np.round(stress_normal, decimals=8)
    stress_1st = np.round(stress_1st, decimals=8)
    stress_2nd = np.round(stress_2nd, decimals=8)
    disp_f = np.round(disp_f, decimals=8)
    
    tag_list = [False, False, False]
    index_list = ['*Damage Initiation, criterion=MAXS',
                  '*Damage Evolution, type=DISPLACEMENT',
                  '*Elastic, type=TRACTION']
    with open(inp_path, 'r+') as inp_file:
        lines = inp_file.readlines()
        inp_file.seek(0)
        for line in lines:
            outer_continue = False
            for i in range(3):
                if index_list[i] in line.strip():
                    tag_list[i] = True
                    inp_file.write(line)
                    outer_continue = True
            if outer_continue:
                continue
            if tag_list[0]:
                inp_file.write(" " + str(stress_normal) + ",\t" + str(stress_1st) + ",\t" + str(stress_2nd) + '\n')
            elif tag_list[1]:
                inp_file.write(" " + str(disp_f) + ',\n')
            elif tag_list[2]:
                inp_file.write(" " + str(Enn) + ",\t" + str(G1) + ",\t" + str(G2) + '\n')
            else:
                inp_file.write(line)
            tag_list = [False, False, False]
            

def submit_job(directory, inp_file_path, job_name):
    os.chdir(directory)
    mdb.ModelFromInputFile(name='Nacre_model', inputFileName=inp_file_path)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    a = mdb.models['Nacre_model'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
        predefinedFields=ON, connectors=ON)
    mdb.Job(name=job_name, model='Nacre_model', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=2, 
        memoryUnits=GIGA_BYTES, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    
    try:
        mdb.jobs[job_name].waitForCompletion(600)
    except (AbaqusException, message):
        print("Job timed out", message) 


def Nacre_2D_odb(directory, run_count, job_name):    
    txt_name = job_name + '_ite' + run_count + '.txt'
    txt_path = os.path.join(directory, txt_name)
    odb_path = os.path.join(directory, job_name + '.odb')
    o3 = session.openOdb(name=odb_path)
    odb = session.odbs[odb_path]
    session.XYDataFromHistory(name='RF2', odb=odb, 
        outputVariableName='Reaction force: RF2 PI: PART-1-1 Node 4710 in NSET RP', 
        steps=('Step-1', ), __linkedVpName__='Viewport: 1')
    session.XYDataFromHistory(name='U2', odb=odb, 
        outputVariableName='Spatial displacement: U2 PI: PART-1-1 Node 4710 in NSET RP', 
        steps=('Step-1', ), __linkedVpName__='Viewport: 1')
    x0 = session.xyDataObjects['RF2']
    x1 = session.xyDataObjects['U2']
    session.writeXYReport(fileName=txt_path, xyData=(x0, x1))


if __name__ == '__main__':
    params = os.environ["ABAQUS_PARAMS"].split(', ')
    iteration = params[0]
    sigma_c_n = float(params[1])*1e-6
    sigma_c_s = float(params[2])*1e-6
    delta_d_n = float(params[3])*1e-3
    delta_d_s = float(params[4])*1e-3
    delta_f = float(params[5])*1e-3
    directory = params[6]
    inp_file_path = params[7]
    job_name = params[8]
    if os.path.exists(job_name + '.lck'):
        os.remove(job_name + '.lck')
    
    modify_inp(inp_file_path, sigma_c_n, sigma_c_s, delta_d_n, delta_d_s, delta_f)
    submit_job(directory, inp_file_path, job_name)
    time.sleep(3) 
    Nacre_2D_odb(directory, iteration, job_name)
    time.sleep(5)
    