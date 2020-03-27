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

model_name = 'stewart_platform'

# getting the configuration .json file
config_file = os.path.join(symdata_dir, 'stewart_platform_cfg.json')

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

# Helper Values
# =============
v1 = np.array([[0], [1], [0]])

C1 = np.array([[np.cos(np.deg2rad(120)), -np.sin(np.deg2rad(120)), 0], 
               [np.sin(np.deg2rad(120)),  np.cos(np.deg2rad(120)), 0],
               [0,                       0,                        1]])

C2 = np.array([[np.cos(np.deg2rad(240)), -np.sin(np.deg2rad(240)), 0], 
               [np.sin(np.deg2rad(240)),  np.cos(np.deg2rad(240)), 0],
               [0,                       0,                        1]])

v2 = C1@v1
v3 = C2@v1


num_config.hps_bottom_1.flat[:] = -400,    0, 0
num_config.hps_bottom_2.flat[:] =  200,  346, 0
num_config.hps_bottom_3.flat[:] =  200, -346, 0

num_config.hps_middle_1.flat[:] = -500,    0, 0
num_config.hps_middle_2.flat[:] =  250,  433, 0
num_config.hps_middle_3.flat[:] =  250, -433, 0

num_config.hps_upper_1.flat[:] = -400,    0, 387
num_config.hps_upper_2.flat[:] =  200,  346, 387
num_config.hps_upper_3.flat[:] =  200, -346, 387

num_config.hps_strut_upper.flat[:] =  0, 0, 348
num_config.hps_strut_lower.flat[:] =  0, 0, -190


num_config.ax1_jcs_rev_1.flat[:] = v1
num_config.ax1_jcs_rev_3.flat[:] = v2
num_config.ax1_jcs_rev_2.flat[:] = v3

num_config.vcs_x.flat[:] = 1, 0, 0
num_config.vcs_y.flat[:] = 0, 1, 0
num_config.vcs_z.flat[:] = 0, 0, 1

num_config.fas_strut_FL = 800

num_config.s_links_ro = 25
num_config.s_rockers_ro = 30
num_config.s_strut_lower_radius = 30
num_config.s_strut_upper_radius = 35

num_config.UF_mcs_act_1 = lambda t : np.deg2rad(360) * np.sin(t)
num_config.UF_mcs_act_2 = lambda t : np.deg2rad(360) * np.cos(t)
num_config.UF_mcs_act_3 = lambda t : np.deg2rad(360) * np.sin(t+1)

num_config.UF_fas_strut_Fs = lambda x : 120*1e6 * x if x > 0 else 0
num_config.UF_fas_strut_Fd = lambda v : 12*1e6 * v

# Assembling the configuration and exporting a .json file that
# holds these numerical values
num_config.assemble()
num_config.export_json()

# Overriding some configuration values
num_config.m_rbs_table = 1500*1e3
num_config.Jbar_rbs_table = np.array([[849696769400, 798659080.5, -93578452884],
                                      [798659080.5, 1829010966541, -1258130092],
                                      [-93578452884, -1258130092, 2014800214060]])
num_config.R_rbs_table.flat[:] = -18.9, -2.65, 816.3

# ================================================================== #
#                   Creating the Simulation Instance
# ================================================================== #
sim = simulation('sim', num_model, 'kds')

# setting the simulation time grid
sim.set_time_array(10, 1e-2)

# Starting the simulation
sim.solve()

# Evaluating the system reactions
sim.eval_reactions()

# Saving the results in the /results directory as csv and npz
sim.save_as_csv(results_dir, 'test_2')
sim.save_as_npz(results_dir, 'test_2')

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

coordinates = ['rbs_table.x', 'rbs_table.y', 'rbs_table.z']

sim.soln.pos_dataframe.plot(x='time', y=coordinates, grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y=coordinates, grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y=coordinates, grid=True, figsize=(10,4))

reactions = ['F_ground_jcs_rev_1.x', 'F_ground_jcs_rev_1.y', 'F_ground_jcs_rev_1.z']
sim.soln.reactions_dataframe.plot(x='time', y=reactions, grid=True, figsize=(10,4))

plt.show()
