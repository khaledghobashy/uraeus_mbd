# 3rd party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# uraeus imports
from uraeus.nmbd.python import multibody_system, simulation, configuration

# model imports
from numenv.python.src import double_pendulum

# Creating the numerical model from the imported script
num_model  = multibody_system(double_pendulum)

# Creating a numerical configuration instance from the .json file
# created by the symbolic configuration
num_config = configuration('base')
num_config.construct_from_json('symenv/double_pendulum_cfg.json')

# Assigning this configuration instance to the numerical model
num_model.topology.config = num_config

# ================================================================== #
#              Numerical Configuration of the System
# ================================================================== #

num_config.hps_p1.flat[:] = 0, 0, 0
num_config.hps_p2.flat[:] = 0, 200, 0
num_config.hps_p3.flat[:] = 0, 400, 0
num_config.vcs_v.flat[:] = 1, 0, 0

num_config.s_radius = 20

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
sim.save_as_csv('results', 'test_1')
sim.save_as_npz('results', 'test_1')

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

sim.soln.pos_dataframe.plot(x='time', y=['rbs_body_1.z', 'rbs_body_1.y'], grid=True, figsize=(10,4))
sim.soln.pos_dataframe.plot(x='time', y=['rbs_body_2.z', 'rbs_body_2.y'], grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_body_1.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_body_1.z', grid=True, figsize=(10,4))

plt.show()
