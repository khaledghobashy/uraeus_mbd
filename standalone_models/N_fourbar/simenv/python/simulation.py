# standard library imports
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

model_name = 'N_fourbar'

# getting the configuration .json file
config_file = os.path.join(symdata_dir, 'N_fourbar_cfg.json')

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

# Number of additional loops 
loops = 3

# length (l) of links 
l = 300

# initial angle of inclination (theta) of vertical links
theta = 30

num_config.hps_p1.flat[:] = 0, 0, 0
num_config.hps_p2.flat[:] = 0, l*np.sin(np.deg2rad(theta)), l*np.cos(np.deg2rad(theta))
num_config.hps_p3 = num_config.hps_p2 + np.array([[0], [l], [0]])
num_config.hps_p4 = num_config.hps_p3 + np.array([[0], [-l*np.sin(np.deg2rad(theta))], [-l*np.cos(np.deg2rad(theta))]])

num_config.vcs_x.flat[:] = 1, 0, 0
num_config.s_radius = 10

# assigning numerical values for the additional four-bar loops
p_end = 4
for i in range(loops):
    p0 = getattr(num_config, 'hps_p%s'%(p_end - 1))
    p1 = getattr(num_config, 'hps_p%s'%(p_end))
    setattr(num_config, 'hps_p%s'%(p_end + 1), p0 + np.array([[0], [l], [0]]))
    setattr(num_config, 'hps_p%s'%(p_end + 2), p1 + np.array([[0], [l], [0]]))

    p_end += 2

# Assembling the configuration and exporting a .json file that
# holds these numerical values
num_config.assemble()
num_config.export_json()

# ================================================================== #
#                   Creating the Simulation Instance
# ================================================================== #
sim = simulation('sim', num_model, 'dds')

# setting the simulation time grid
sim.set_time_array(5, 5e-3)

# Starting the simulation
sim.solve()

# Saving the results in the /results directory as csv and npz
sim.save_as_csv(results_dir, 'test_2')
sim.save_as_npz(results_dir, 'test_2')

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

sim.soln.pos_dataframe.plot(x='time', y=['rbs_l1.z', 'rbs_l1.y'], grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_l1.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_l1.z', grid=True, figsize=(10,4))

plt.show()
