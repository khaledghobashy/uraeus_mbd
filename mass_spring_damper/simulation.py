
# 3rd party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# uraeus imports
from uraeus.nmbd.python import multibody_system, simulation, configuration

# model imports
from numenv.python.src import mass_spring_damper

# Creating the numerical model from the imported script
num_model  = multibody_system(mass_spring_damper)

# Creating a numerical configuration instance from the .json file
# created by the symbolic configuration
num_config = configuration('base')
num_config.construct_from_json('symenv/mass_spring_damper_cfg.json')

# Assigning this configuration instance to the numerical model
num_model.topology.config = num_config

# ================================================================== #
#              Numerical Configuration of the System
# ================================================================== #

num_config.hps_p1.flat[:] = 0, 0, 0
num_config.hps_p2.flat[:] = 0, 0, 200
num_config.vcs_v.flat[:] = 0, 0, 1

num_config.fas_TSDA_FL = 200

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


num_config.UF_fas_TSDA_Fs = stiffness
num_config.UF_fas_TSDA_Fd = damping

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
sim.save_as_csv('results', 'test_1')
sim.save_as_npz('results', 'test_1')

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

sim.soln.pos_dataframe.plot(x='time', y='rbs_body.z', grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_body.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_body.z', grid=True, figsize=(10,4))

sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_fas_TSDA.x', grid=True, figsize=(10,4))
sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_fas_TSDA.y', grid=True, figsize=(10,4))
sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_fas_TSDA.z', grid=True, figsize=(10,4))

sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_jcs_trans.x', grid=True, figsize=(10,4))
sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_jcs_trans.y', grid=True, figsize=(10,4))
sim.soln.reactions_dataframe.plot(x='time', y='F_rbs_body_jcs_trans.z', grid=True, figsize=(10,4))

plt.show()
