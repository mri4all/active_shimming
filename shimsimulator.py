# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:36:56 2023

@author: leeoralon
"""
import sys
sys.path.append("..")
import magpylib as magpy
import numpy as np
import magsimulator
import pandas as pd
import magpylib as magpy
import scipy
from scipy.spatial.transform import Rotation as R
import magsimulator
from cvxopt import matrix, solvers
from scipy.linalg import solve
from scipy.linalg import lstsq
# r- radius
# nr - number of coils azimouthally
# z - z position
# nz - number of coils in z (total coils = nr*nz)
  
def generate_shim_coils_on_cylinder(r,number_mags_azimuthal,coil_diam,delta_z,delta_theta,nz,dc_phase=0,ref_curr=1,plot=False,tofile=None):
    
    # print(r,number_mags_azimuthal,coil_diam,delta_z,delta_theta,nz,dc_phase)
    zloc = np.round(np.r_[-(nz+1)*delta_z/2+delta_z:(nz+1)*delta_z/2:delta_z],4)

    point_list = []
    
    mintheta = 360/number_mags_azimuthal
    z_inc_idx=0
    for z in zloc:   

        theta = np.linspace(0,360-mintheta,number_mags_azimuthal)*np.pi/180
        
        for jj in theta:
            # print(jj)
            x= r*np.cos(jj+z_inc_idx*delta_theta*np.pi/180+dc_phase*np.pi/180)
            y= r*np.sin(jj+z_inc_idx*delta_theta*np.pi/180+dc_phase*np.pi/180)
            rotx = 0
            roty = 0
            rotz = jj*180/np.pi +delta_theta*z_inc_idx + dc_phase
            point_list.append([x,y,z,rotx,roty,rotz,coil_diam,r,number_mags_azimuthal,coil_diam,delta_z,delta_theta,nz,ref_curr])
        
        z_inc_idx=z_inc_idx+1
    
    # print(point_list)
    if plot==True:
        loops = magpy.Collection(style_label='loops')

        for obj in point_list:
            
            r = R.from_euler('yz', [90,obj[5]], degrees=True)
            loop = magpy.current.Loop(current=ref_curr, diameter=obj[6],position=[obj[0],obj[1],obj[2]],orientation=r)
            loops.add(loop)
        
        magpy.show(loops)
        
    ledger = pd.DataFrame(point_list,columns =['X-pos', 'Y-pos', 'Z-pos','X-rot','Y-rot','Z-rot','Coil-diam','r','number_mags_aximuthal','coil_diam','delta_z','delta_theta','nz','ref_curr'])
    cols_to_round = ['X-pos', 'Y-pos', 'Z-pos','X-rot','Y-rot','Z-rot','Coil-diam','r','number_mags_aximuthal','coil_diam','delta_z','delta_theta','nz','ref_curr']
    ledger[cols_to_round] = ledger[cols_to_round].round(3)
    
    if(tofile!=None):
        ledger.to_excel(tofile, index=False)
    
    return ledger

def simulate_shim_elements(ledger,col_sensors,Bfield_component=0):
    df = ledger.copy()
    B_elements = np.zeros((col_sensors[0].position.shape[0],ledger.shape[0]))
    # loops = magpy.Collection(style_label='magnets')

    for idx_ledger, df_row in df.iterrows():
        # print('computing: ' +  str(idx_ledger) + ' out of: ' + str(df.shape[0]-1))
        r = R.from_euler('yz', [90,df_row['Z-rot']], degrees=True)
        loop = magpy.current.Loop(current=df_row['ref_curr'], diameter=df_row['Coil-diam'],position=[df_row['X-pos'],df_row['Y-pos'],df_row['Z-pos']],orientation=r)
        # Btmp = col_sensors.getB(loop)
        Btmp = col_sensors.getB(loop)
        # print('meanB0: ' + str(Btmp[-1,0]))
        B_elements[:,idx_ledger] = Btmp[:,Bfield_component]
        # loops.add(loop)
    
    # magpy.show(loops, col_sensors)
    
    return B_elements

def simulate_shim_ledger(ledger,col_sensors,Bfield_component=0,plotting=True):
    df = ledger.copy()
    loops = magpy.Collection(style_label='magnets')

    for idx_ledger, df_row in df.iterrows():
        # print('computing: ' +  str(idx_ledger) + ' out of: ' + str(df.shape[0]-1))
        r = R.from_euler('yz', [90,df_row['Z-rot']], degrees=True)
        loop = magpy.current.Loop(current=df_row['ref_curr'], diameter=df_row['Coil-diam'],position=[df_row['X-pos'],df_row['Y-pos'],df_row['Z-pos']],orientation=r)
        # Btmp = col_sensors.getB(loop)
        # print('meanB0: ' + str(Btmp[-1,0]))
        loops.add(loop)
    
    if(plotting==True):
        magpy.show(loops, col_sensors)
        
    B = col_sensors.getB(loop)

    return B, loops

# B field must be in units of Tesla!    
def shim_field_no_constraints(ledg,col_sensors,Btar,Bfield_component=0):
    A = simulate_shim_elements(ledg,col_sensors,Bfield_component)

    b = Btar[:,Bfield_component] - np.mean(Btar[:,Bfield_component])
    
    Q = np.dot(A.T,A)
    c = np.dot(-A.T,b)
    
    sol=solvers.qp(matrix(Q), matrix(c))
    solution=np.array(sol['x'])
    
    resultingfieldhomogeneity = np.std(Btar[:,0] - np.dot(A,solution).reshape(-1))*42580000
    print('std unshimmed B0=' + str(np.std(b)*42580000))
    # print('std shimmed B0=' + str(np.std(np.dot(A,m[0]).reshape(-1)*42580000)))
    print('std shimmed B0=' + str(resultingfieldhomogeneity))
    return solution, resultingfieldhomogeneity
     
    return solution

def shim_least_squares(Btar,ledg,col_sensors,Bfield_component=0):
    A = simulate_shim_elements(ledg,col_sensors,Bfield_component)
    b = Btar[:,Bfield_component] - np.mean(Btar[:,Bfield_component])

    m = lstsq(A, b)

    solution=m[0]
    resultingfieldhomogeneity = np.std(Btar[:,0] - np.dot(A,solution).reshape(-1))*42580000
    print('std unshimmed B0=' + str(np.std(b)*42580000))
    # print('std shimmed B0=' + str(np.std(np.dot(A,m[0]).reshape(-1)*42580000)))
    print('std shimmed B0=' + str(resultingfieldhomogeneity))
    return solution, resultingfieldhomogeneity

# B field must be in units of Tesla!    
def shim_field_w_constraints(ledg,col_sensors,BackgroundB,maxcur_per_chan=3,max_tot_cur=45,Bfield_component=0):
    A = simulate_shim_elements(ledg,col_sensors,Bfield_component)

    # BackgroundB = BackgroundB/1000 #convert mT to Tesla

    b = BackgroundB[:,Bfield_component] - np.mean(BackgroundB[:,Bfield_component])
    
    Q = np.dot(A.T,A)
    c = np.dot(-A.T,b)
    
    G_maxtotalcurr = np.ones((1,A.shape[1]))
    G_max_curr = np.eye(A.shape[1])
    G_min_curr = -np.eye(A.shape[1])
    G = np.concatenate((G_maxtotalcurr,G_max_curr,G_min_curr),axis=0)
    h = np.concatenate((max_tot_cur,maxcur_per_chan*np.ones((A.shape[1],1)), maxcur_per_chan*np.ones((A.shape[1],1))), axis=None)
    
    # G = np.ones((1,c.shape[0]))*G_maxtotalcurr
    # h = np.ones((1,1))*100
    
    sol=solvers.qp(matrix(Q), matrix(c), matrix(G), matrix(h))
    # sol=solvers.qp(matrix(Q), matrix(c))
    solution=np.array(sol['x'])
    
    # print(solution.T)
    # print('std (Hz) unshimmed B0=' + str(np.std(b)*42580000))
    # print('std (Hz) shimmed B0=' + str(np.std(np.dot(A,solution).reshape(-1)*42580000)))
     
    resultingfieldhomogeneity = np.std(BackgroundB[:,Bfield_component]-np.dot(A,solution).reshape(-1))*42580000
    return solution, resultingfieldhomogeneity


def extract_3D_shim_field(loop,xmin=-70,xmax=70,ymin=-70,ymax=70,zmin=-70, zmax=70, numberpoints_per_ax = 25,Bcomponent=0):
        
    xs = np.linspace(xmin, xmax, numberpoints_per_ax)
    ys = np.linspace(ymin, ymax, numberpoints_per_ax)
    zs = np.linspace(zmin,zmax,numberpoints_per_ax)
    
    B = np.zeros((xs.shape[0],ys.shape[0],zs.shape[0],3))
    
    for kk in range(zs.shape[0]):
        print('extracting slice:' + str(kk+1) + ' out of:' + str(zs.shape[0]))

        grid_xy = np.array([[(x,y,zs[kk]) for x in xs] for y in ys])

        B[:,:,kk,:] = loop.getB(grid_xy)
        
    return B[:,:,:,Bcomponent]