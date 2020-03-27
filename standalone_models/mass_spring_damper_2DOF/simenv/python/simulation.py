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

model_name = 'mass_spring_damper_2DOF'

# getting the configuration .json file
config_file = os.path.join(symdata_dir, 'mass_spring_damper_2DOF_cfg.json')

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
num_config.hps_p2.flat[:] = 0, 0, 200
num_config.hps_p2.flat[:] = 0, 0, 400

num_config.vcs_v.flat[:] = 0, 0, 1

num_config.fas_TSDA_1_FL = 200
num_config.fas_TSDA_2_FL = 200

num_config.s_radius = 20*3

def stiffness(x):
    x = float(x)
    force = x * 5*1e6
    #print('x = %s'%x)
    #print('K_Force = %s'%force)
    return force

def damping(v):
    v = v[0,0]
    force = v * 0.5*1e5
    #print('v = %s'%v)
    #print('D_Force = %s'%force)
    return force


num_config.UF_fas_TSDA_1_Fs = stiffness
num_config.UF_fas_TSDA_1_Fd = damping

num_config.UF_fas_TSDA_2_Fs = stiffness
num_config.UF_fas_TSDA_2_Fd = damping

# Assembling the configuration and exporting a .json file that
# holds these numerical values
num_config.assemble()
num_config.export_json()

# ================================================================== #
#                   Creating the Simulation Instance
# ================================================================== #
sim = simulation('sim', num_model, 'dds')

# setting the simulation time grid
sim.set_time_array(5, 1e-2)

# Starting the simulation
sim.solve()

# Evaluating the system reactions
sim.eval_reactions()

# Saving the results in the /results directory as csv and npz
sim.save_as_csv(results_dir, 'test_1')
sim.save_as_npz(results_dir, 'test_1')

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

sim.soln.pos_dataframe.plot(x='time', y='rbs_body_1.z', grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_body_1.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_body_1.z', grid=True, figsize=(10,4))

sim.soln.pos_dataframe.plot(x='time', y='rbs_body_2.z', grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_body_2.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_body_2.z', grid=True, figsize=(10,4))

sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_1_fas_TSDA_1.x', grid=True, figsize=(10,4))
sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_1_fas_TSDA_1.y', grid=True, figsize=(10,4))
sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_1_fas_TSDA_1.z', grid=True, figsize=(10,4))

sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_1_jcs_trans_1.x', grid=True, figsize=(10,4))
sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_1_jcs_trans_1.y', grid=True, figsize=(10,4))
sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_1_jcs_trans_1.z', grid=True, figsize=(10,4))

plt.show()
