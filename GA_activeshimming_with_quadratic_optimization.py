# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:54:01 2023

@author: la506
"""

import os
#os.add_dll_directory("C://Users/Sebastian/anaconda3/envs/MRI4All/DLLs")
                     
import sys
sys.path.append("..")

import shimsimulator
import magsimulator
import magcadexporter
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from magpylib.magnet import Cuboid, CylinderSegment
import itertools
from scipy.spatial.transform import Rotation as R
import pandas as pd
import cProfile
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, freeze_support
from os import getpid
import time
import addcopyfighandler
import pygad
import numpy.matlib
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import multiprocessing
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool

import pickle
from cvxopt import matrix, solvers

def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

# from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.variable import Real, Integer
from pymoo.optimize import minimize

filename = 'optimization_after_neonate_magnet_Rmin_132p7mm_extrinsic_rot_DSV140mm_maxh60_maxlayers6_maxmag990.xlsx'
mag_vect = [1270,0,0]
ledger, magnets = magsimulator.load_magnet_positions(filename, mag_vect)

col_sensors = magpy.Collection(style_label='sensors')
sensor1 = magsimulator.define_sensor_points_on_filled_sphere(200,70,5,[0,0,0])
col_sensors.add(sensor1)


magnets = magpy.Collection(style_label='magnets')

eta, meanB0, col_magnet, B = magsimulator.simulate_ledger(magnets,col_sensors,mag_vect,ledger,0.06,4,True,False,None,False)
print('mean B0='+str(round(meanB0,3)) +  ' homogeneity=' + str(round(eta,3)))
data = {'Bfield':  B, 'coordinates': col_sensors[0].position}

fileObj = open(filename[:-4]+'pkl', 'wb')
pickle.dump(data,fileObj)
fileObj.close()
#%%
data=magsimulator.extract_3Dfields(col_magnet,xmin=-70,xmax=70,ymin=-70,ymax=70,zmin=-70, zmax=70, numberpoints_per_ax = 11,filename=None,plotting=True,Bcomponent=0)
# magsimulator.plot_3D_field(data['Bfield'],Bcomponent=0)
B=data['Bfield']
col_sensors = magpy.Collection(style_label='sensors')
sensor1 = magpy.Sensor(position=data['coordinates'])
col_sensors.add(sensor1)

# magpy.show(col_magnet, col_sensors)
fileObj = open(filename[:-4]+'pkl', 'wb')
pickle.dump(data,fileObj)
fileObj.close()

# magcadexporter.export_magnet_ledger_to_cad(filename,'C:\\Users\\la506\\Downloads\\cube_boolean.step')
file = open(filename[:-4]+'pkl', 'rb')
rd_data = pickle.load(file)
file.close()

#%%
class MultiObjectiveMixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        
        filename = '../data/optimization_after_neonate_magnet_Rmin_132p7mm_extrinsic_rot_DSV140mm_maxh60_maxlayers6_maxmag990.xlsx'
        file = open(filename[:-4]+'pkl', 'rb')
        self.data = pickle.load(file)
        self.B_background =data['Bfield']
        self.coordinates = data['coordinates']
        file.close()

        self.B_background =self.B_background.reshape(-1,3)/1000 #convert mT to Tesla
        self.r=70
        self.maxchannels = 64
        self.refcurr = 1
        
        self.maxtotalcurr = 45
        self.maxchannelcurr = 3
        
        variables = dict()
        
        variables["x01"] = Integer(bounds=(0, 90)) #delta dc phase
        variables["x02"] = Integer(bounds=(2,8)) # number of coils azimuthally
        variables["x03"] = Integer(bounds=(5,150)) #coil diameter
        variables["x04"] = Integer(bounds=(1,20)) #number of z rows
        variables["x05"] = Integer(bounds=(5,200)) #delta z
        variables["x06"] = Integer(bounds=(0, 360)) #delta theta between z increments
        
        # super().__init__(vars=variables,n_ieq_constr=1, n_obj=2, **kwargs)
        super().__init__(vars=variables,n_ieq_constr=1, n_obj=1, **kwargs)


    def _evaluate(self, x, out, *args, **kwargs):
        delta_dc_phase=np.array(x["x01"]).reshape((1,1)).astype(float)
        delta_dc_phase=int(delta_dc_phase[0])
        
        num_coils_azi = np.array(x["x02"]).reshape((1,1)).astype(int)
        num_coils_azi=int(num_coils_azi[0])
        
        # print(num_coils_azi)
        
        coil_diam = np.array(x["x03"]).reshape((1,1)).astype(int)
        coil_diam=int(coil_diam[0])

        nz = np.array(x["x04"]).reshape((1,1)).astype(int)
        nz=int(nz[0])

        dz = np.array(x["x05"]).reshape((1,1)).astype(int)
        dz=int(dz[0])

        delta_theta = np.array(x["x06"]).reshape((1,1)).astype(float)
        delta_theta=int(delta_theta[0])        
        
        # print(delta_dc_phase,num_coils_azi,dz,nz,coil_diam,delta_theta)
# running quadradic programming optimization

        ledg= shimsimulator.generate_shim_coils_on_cylinder(self.r,
                                                            num_coils_azi,
                                                            coil_diam,
                                                            dz,
                                                            delta_theta,
                                                            nz,
                                                            delta_dc_phase,
                                                            self.refcurr,
                                                            plot=False)
        
        # ledg.head()
        
        # col_sensors = magpy.Collection(style_label='sensors')
        # sensor1 = magsimulator.define_sensor_points_on_filled_sphere(200,70,5,[0,0,0])
        # col_sensors.add(sensor1)
        col_sensors = magpy.Collection(style_label='sensors')
        sensor1 = magpy.Sensor(position=self.coordinates)
        col_sensors.add(sensor1)
                
        # solution,stdfield_hz=shimsimulator.shim_least_squares(self.B_background,ledg,col_sensors,Bfield_component=0)
        solution, stdfield_hz = shimsimulator.shim_field_w_constraints(ledg,col_sensors,
                                                self.B_background,
                                                self.maxchannelcurr,
                                                self.maxtotalcurr,
                                                Bfield_component=0)

        g = nz*num_coils_azi-self.maxchannels
        print('unshimmed:'+ str(np.round(np.std(self.B_background[:,0]*42580000),5)) + ' ; shimmed:' + str(np.round(stdfield_hz,5)) + '; n-coils:' + str(nz*num_coils_azi))
        # out["F"] = np.column_stack([stdfield_hz,nz*num_coils_azi])
        out["F"] = stdfield_hz
        out["G"] = g
      
   
    
   
    def get_coils(self, x):
        delta_dc_phase=np.array(x["x01"]).reshape((1,1)).astype(float)
        delta_dc_phase=float(delta_dc_phase[0])
        
        num_coils_azi = np.array(x["x02"]).reshape((1,1)).astype(int)
        num_coils_azi=int(num_coils_azi[0])
        
        # print(num_coils_azi)
        
        coil_diam = np.array(x["x03"]).reshape((1,1)).astype(int)
        coil_diam=int(coil_diam[0])

        nz = np.array(x["x04"]).reshape((1,1)).astype(int)
        nz=int(nz[0])

        dz = np.array(x["x05"]).reshape((1,1)).astype(int)
        dz=int(dz[0])

        delta_theta = np.array(x["x06"]).reshape((1,1)).astype(float)
        delta_theta=float(delta_theta[0])        
        
        # print(delta_dc_phase,num_coils_azi,dz,nz,coil_diam,delta_theta)
# running quadradic programming optimization

        ledg= shimsimulator.generate_shim_coils_on_cylinder(self.r,
                                                            num_coils_azi,
                                                            coil_diam,
                                                            dz,
                                                            delta_theta,
                                                            nz,
                                                            delta_dc_phase,
                                                            self.refcurr,
                                                            plot=False)
        
        # ledg.head()
        
        # col_sensors = magpy.Collection(style_label='sensors')
        # sensor1 = magsimulator.define_sensor_points_on_filled_sphere(200,70,5,[0,0,0])
        # col_sensors.add(sensor1)
        col_sensors = magpy.Collection(style_label='sensors')
        sensor1 = magpy.Sensor(position=self.coordinates)
        col_sensors.add(sensor1)

        # solution,stdfield_hz=shimsimulator.shim_least_squares(self.B_background,ledg,col_sensors,Bfield_component=0)
        solution, stdfield_hz = shimsimulator.shim_field_w_constraints(ledg,col_sensors,
                                                self.B_background,
                                                self.maxchannelcurr,
                                                self.maxtotalcurr,
                                                Bfield_component=0)

        return ledg,solution,stdfield_hz
    
#%%
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import multiprocessing
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool
import pickle
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.gradient.automatic import AutomaticDifferentiation
from pymoo.core.mixed import MixedVariableGA
from cvxopt import matrix, solvers

import warnings
warnings.filterwarnings("ignore")

solvers.options['show_progress'] = False #used to kill the output and just display the resulting homogeneity

n_threads = 2
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

# problem = MultiObjectiveMixedVariableProblem(elementwise_runner=runner)

problem = MultiObjectiveMixedVariableProblem()

algorithm = MixedVariableGA(pop_size=30, survival=RankAndCrowdingSurvival())
# algorithm = NSGA2(pop_size=80,sampling=MixedVariableSampling(),                 mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),                  eliminate_duplicates=MixedVariableDuplicateElimination(),)

# algorithm = MixedVariableGA(pop=20) 

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)


# filename = 'opt_shimming_layers1_maxzpos_50.xlsx'

# fileObj = open(filename[:-4]+'pkl', 'wb')
# pickle.dump(res,fileObj)
# fileObj.close()
#%%
plt.figure()
plt.scatter(res.F[:,0], res.F[:,1],s=100)
plt.title("Tradeoff B0 and Homogeneity", fontsize=18, weight='bold')
plt.xlabel("Standard deviation field (Hz)", fontsize=16, weight='bold')
plt.ylabel("Number of coils", fontsize=16, weight='bold')
plt.xticks(fontsize=14, weight='bold')
plt.show()
#%%
B_rescaled = B_background /1000
 
xx=res.X
ledger,solution,stdfield_hz = problem.get_coils(xx)
ledger.to_excel('test.xlsx', index=False)
#%%
B_loops,loops = shimsimulator.simulate_shim_ledger(ledg,col_sensors,0,True)

print('std shimmed B0=' + str((np.std(Btar[:,0] - np.dot(A,m[0]).reshape(-1))*42580000)))

#%%
magcadexporter.export_magnet_ledger_to_cad('mean B0=' + str(np.round(meanB0,4)) + ' homogen=' + str(np.round(eta,4)) + ' ' + filename,[0.2,0.2,0.2])