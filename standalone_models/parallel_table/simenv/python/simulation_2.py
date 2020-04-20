import os

# 3rd party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# uraeus imports
from uraeus.nmbd.python import (import_source, multibody_system, 
                                simulation, configuration)

# ================================================================== #
#                               Helpers
# ================================================================== #

# getting to the root project directory from this file directory
dir_name = os.path.dirname(__file__)

# creating the various needed directories references
project_dir = os.path.abspath(os.path.join(dir_name, '../../'))
symdata_dir = os.path.join(project_dir, 'symenv/data/')
numdata_dir = os.path.join(project_dir, 'numenv/python/src/')
results_dir = os.path.join(project_dir, 'simenv/results/')

# ================================================================== #
#                           Initializations
# ================================================================== #

# TODO : User Declared
model_name = 'parallel_table'

# getting the configuration .json file
# TODO : User Declared
config_file = os.path.join(symdata_dir, 'parallel_table_cfg.json')

# Creating a numerical configuration instance
num_config = configuration('base')
# constructing the numerical configuration instance from
# imported JSON file
num_config.construct_from_json(config_file)

# Getting the numrical topology module
model = import_source(numdata_dir, model_name)

# Creating the numerical model from the imported module
num_model  = multibody_system(model)
# Assigning this configuration instance to the numerical model
num_model.topology.config = num_config

# ================================================================== #
#              Numerical Configuration of the System
# ================================================================== #


num_config.hps_p1.flat[:] = 0, 0, 0
num_config.hps_p2.flat[:] = 0, 200, 0
num_config.hps_p3.flat[:] = 0, 500, 0

num_config.hps_p4.flat[:] = 0, 50, 200
num_config.hps_p5.flat[:] = 0, 250, 200
num_config.hps_p6.flat[:] = 0, 450, 200

num_config.hps_p7.flat[:] = 0, 0, 400
num_config.hps_p8.flat[:] = 0, 200, 400
num_config.hps_p9.flat[:] = 0, 500, 400

num_config.vcs_x.flat[:] = 1, 0, 0
num_config.vcs_y.flat[:] = 0, 1, 0
num_config.vcs_z.flat[:] = 0, 0, 1

num_config.s_radius = 10

# Assembling the configuration and exporting a .json file that
# holds these numerical values
num_config.assemble()
num_config.export_json()

def stiffness(x):
    x = float(x)
    #print(x)
    k = 150*1e3
    return k * x

def damping(v):
    v = float(v)
    return 15*1e2 * v

num_config.fas_strut_FL = 250
num_config.UF_fas_strut_Fs = stiffness
num_config.UF_fas_strut_Fd = damping

num_config.UF_mcs_act = lambda t : 50 * np.sin(t)
# ================================================================== #
#                   Creating the Simulation Instance
# ================================================================== #
sim = simulation('sim', num_model, 'kds')

# setting the simulation time grid
sim.set_time_array(2*np.pi, 5e-3)

# Starting the simulation
sim.solve()

# Saving the results in the /results directory as csv and npz
sim.save_as_csv(results_dir, 'test_10')
sim.save_as_npz(results_dir, 'test_10')

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

sim.soln.pos_dataframe.plot(x='time', y=['rbs_l1.z', 'rbs_l1.y'], grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_l1.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_l1.z', grid=True, figsize=(10,4))

sim.soln.pos_dataframe.plot(x='time', y=['rbs_l8.z', 'rbs_l8.y'], grid=True, figsize=(10,4))

sim.soln.pos_dataframe.plot(x='time', y='rbs_table.z', grid=True, figsize=(10,4))
sim.soln.pos_dataframe.plot(x='time', y='rbs_table.y', grid=True, figsize=(10,4))


plt.show()
