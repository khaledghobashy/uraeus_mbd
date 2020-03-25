
# 3rd party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# uraeus imports
from uraeus.nmbd.python import multibody_system, simulation, configuration

# model imports
from numenv.python.src import MODEL_NAME as model

# Creating the numerical model from the imported script
num_model = multibody_system(model)

# Creating a numerical configuration instance from the .json file
# created by the symbolic configuration
num_config = configuration('base')
# TODO
num_config.construct_from_json('symenv/MODEL_NAME_cfg.json')

# Assigning this configuration instance to the numerical model
num_model.topology.config = num_config

# ================================================================== #
#              Numerical Configuration of the System
# ================================================================== #






# Assembling the configuration and exporting a .json file that
# holds these numerical values
num_config.assemble()
num_config.export_json()

# ================================================================== #
#                   Creating the Simulation Instance
# ================================================================== #
sim = simulation('sim', num_model, 'dds')

# setting the simulation time grid
sim.set_time_array(20, 5e-3)

# Starting the simulation
sim.solve()

# Saving the results in the /results directory as csv and npz
sim.save_as_csv('results', 'test_1')
sim.save_as_npz('results', 'test_1')

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

#sim.soln.pos_dataframe.plot(x='time', y=['rbs_body.z', 'rbs_body.y'], grid=True, figsize=(10,4))
#sim.soln.vel_dataframe.plot(x='time', y='rbs_body.z', grid=True, figsize=(10,4))
#sim.soln.acc_dataframe.plot(x='time', y='rbs_body.z', grid=True, figsize=(10,4))

plt.show()