
# 3rd party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# uraeus imports
from uraeus.nmbd.python import multibody_system, simulation, configuration

# model imports
from numenv.python.src import spatial_slider_crank as model

# Creating the numerical model from the imported script
num_model = multibody_system(model)

# Creating a numerical configuration instance from the .json file
# created by the symbolic configuration
num_config = configuration('base')
num_config.construct_from_json('symenv/spatial_slider_crank_cfg.json')

# Assigning this configuration instance to the numerical model
num_model.topology.config = num_config

# ================================================================== #
#              Numerical Configuration of the System
# ================================================================== #

num_config.hps_a.flat[:] = 0, 100, 120
num_config.hps_b.flat[:] = 0, 100, 200
num_config.hps_c.flat[:] = 200, 0, 0
num_config.hps_d.flat[:] = 200, 0, 0

num_config.hps_s1.flat[:] = 180, 0, 0
num_config.hps_s2.flat[:] = 220, 0, 0

num_config.vcs_x.flat[:] = 1, 0, 0
num_config.vcs_y.flat[:] = 0, 1, 0
num_config.vcs_z.flat[:] = 0, 0, 1

num_config.s_links_ro = 20
num_config.s_block_ro = 25

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

# Evaluating the system reactions
sim.eval_reactions()

# Saving the results in the /results directory as csv and npz
sim.save_as_csv('results', 'test_1')
sim.save_as_npz('results', 'test_1')

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

sim.soln.pos_dataframe.plot(x='time', y=['rbs_l2.z', 'rbs_l2.y'], grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_l2.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_l2.z', grid=True, figsize=(10,4))

plt.show()