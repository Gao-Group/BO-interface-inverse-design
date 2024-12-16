from abaqus import *
from abaqusConstants import *
import __main__
import numpy as np
import os
import time


def Nacre_2D_cae(directory, cae_name, job_name, thick_ini, UY, Enn, G1, G2, stress_normal, stress_1st, stress_2nd, disp_f):
    
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
    
    os.chdir(directory)
    
    openMdb(pathName=os.path.join(directory, cae_name))

    # mdb.models['Model-1'].Material(name='cohesive')
    mdb.models['Model-1'].materials['COHESIVE'].Elastic(type=TRACTION, table=((Enn, 
        G1, G2), ))
    mdb.models['Model-1'].materials['COHESIVE'].MaxsDamageInitiation(table=((stress_normal, 
        stress_1st, stress_2nd), ))
    mdb.models['Model-1'].materials['COHESIVE'].maxsDamageInitiation.DamageEvolution(
        type=DISPLACEMENT, table=((disp_f, ), ))

    # mdb.models['Model-1'].CohesiveSection(name='interface', material='COHESIVE', 
    #     response=TRACTION_SEPARATION, initialThicknessType=SPECIFY, 
    #     initialThickness=thick_ini, outOfPlaneThickness=1.0)

    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
        predefinedFields=ON, connectors=ON)
    # mdb.models['Model-1'].boundaryConditions['UY'].setValues(u2= UY)

    mdb.Job(name=job_name, model='Model-1', description='', type=ANALYSIS, 
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
    
    txt_name = job_name + '_ite' + run_count + '.txt'
    txt_path = os.path.join(directory, txt_name)
    odb_path = os.path.join(directory, job_name + '.odb')
    o3 = session.openOdb(name=odb_path)
    odb = session.odbs[odb_path]
    session.XYDataFromHistory(name='RF2', odb=odb, 
        outputVariableName='Reaction force: RF2 PI: rootAssembly Node 1 in NSET RP', 
        steps=('Step-1', ), __linkedVpName__='Viewport: 1')
    session.XYDataFromHistory(name='U2', odb=odb, 
        outputVariableName='Spatial displacement: U2 PI: rootAssembly Node 1 in NSET RP', 
        steps=('Step-1', ), __linkedVpName__='Viewport: 1')
    x0 = session.xyDataObjects['RF2']
    x1 = session.xyDataObjects['U2']
    session.writeXYReport(fileName=txt_path, xyData=(x0, x1))


if __name__ == '__main__':
    params = os.environ["ABAQUS_PARAMS"].split(',')
    iteration = params[0]
    sigma_c_n = float(params[1])*1e-6
    sigma_c_s = float(params[2])*1e-6
    delta_d_n = float(params[3])*1e-3
    delta_d_s = float(params[4])*1e-3
    delta_f = float(params[4])*1e-3

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
    
    UY = 0.25
    
    # job_name = 'Nacre_2D_run' + str(run_count)
    job_name = 'Nacre-y'
    cae_name = 'Nacre_model.cae'
    directory = r"C:\Users\tangmingjian\Desktop\Nacre Inverse Design\BO code"
    
    if os.path.exists(job_name + '.lck'):
        os.remove(job_name + '.lck')
    
    Nacre_2D_cae(directory, cae_name, job_name, thick_ini, UY, Enn, G1, G2, stress_normal, stress_1st, stress_2nd, disp_f)
        
    time.sleep(3) 
    
    Nacre_2D_odb(directory, iteration, job_name)
        
    time.sleep(5)