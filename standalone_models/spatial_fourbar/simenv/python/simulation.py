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

model_name = 'spatial_fourbar'

# getting the configuration .json file
config_file = os.path.join(symdata_dir, 'spatial_fourbar_cfg.json')

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

num_config.hps_a.flat[:] = 0, 0, 0
num_config.hps_b.flat[:] = 0, 0, 200
num_config.hps_c.flat[:] = -750, -850, 650
num_config.hps_d.flat[:] = -400, -850, 0

num_config.vcs_x.flat[:] = 1, 0, 0
num_config.vcs_y.flat[:] = 0, 1, 0
num_config.vcs_z.flat[:] = 0, 0, 1

num_config.s_radius = 20

num_config.UF_mcs_act = lambda t : -np.deg2rad(360)*t

# Assembling the configuration and exporting a .json file that
# holds these numerical values
num_config.assemble()
num_config.export_json()

# ================================================================== #
#                   Creating the Simulation Instance
# ================================================================== #
sim = simulation('sim', num_model, 'kds')

# setting the simulation time grid
sim.set_time_array(5, 5e-3)

# Starting the simulation
sim.solve()

# Saving the results in the /results directory as csv and npz
sim.save_as_csv(results_dir, 'test_1')
sim.save_as_npz(results_dir, 'test_1')

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

sim.soln.pos_dataframe.plot(x='time', y='rbs_l1.z', grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_l1.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_l1.z', grid=True, figsize=(10,4))

sim.soln.pos_dataframe.plot(x='time', y='rbs_l2.z', grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_l2.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_l2.z', grid=True, figsize=(10,4))

sim.soln.pos_dataframe.plot(x='time', y='rbs_l3.z', grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_l3.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_l3.z', grid=True, figsize=(10,4))

plt.show()
