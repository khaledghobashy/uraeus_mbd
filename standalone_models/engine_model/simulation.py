# 3rd party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# uraeus imports
from uraeus.nmbd.python import multibody_system, simulation, configuration

# model imports
from numenv.python.src import engine_4c

# Creating the numerical model from the imported script
num_model  = multibody_system(engine_4c)

# Creating a numerical configuration instance from the .json file
# created by the symbolic configuration
num_config = configuration('base')
num_config.construct_from_json('symenv/engine_4c_cfg.json')

# Assigning this configuration instance to the numerical model
num_model.topology.config = num_config

# ================================================================== #
#              Numerical Configuration of the System
# ================================================================== #

crank_length = 30
connect_length = 100

num_config.hps_front.flat[:] =  100, 0, 0
num_config.hps_rear.flat[:]  = -100, 0, 0

num_config.hps_a1.flat[:] =  75, 0, connect_length + crank_length
num_config.hps_a2.flat[:] =  25, 0, connect_length - crank_length
num_config.hps_a3.flat[:] = -25, 0, connect_length + crank_length
num_config.hps_a4.flat[:] = -75, 0, connect_length - crank_length

num_config.hps_b1.flat[:] =  75, 0,  crank_length
num_config.hps_b2.flat[:] =  25, 0, -crank_length
num_config.hps_b3.flat[:] = -25, 0,  crank_length
num_config.hps_b4.flat[:] = -75, 0, -crank_length

num_config.hps_center.flat[:] = 0, 0, 0

num_config.hps_p1.flat[:] =  100,  100, 0
num_config.hps_p2.flat[:] =  100, -100, 0
num_config.hps_p3.flat[:] = -100,  100, 0
num_config.hps_p4.flat[:] = -100, -100, 0

num_config.vcs_x.flat[:] = 1, 0, 0
num_config.vcs_y.flat[:] = 0, 1, 0
num_config.vcs_z.flat[:] = 0, 0, 1

num_config.s_piston_radius = 30
num_config.s_connect_radius = 10
num_config.s_crank_radius = 20

num_config.R_rbs_engine_block.flat[:] = 0, 0, 100
num_config.P_rbs_engine_block.flat[:] = 1, 0, 0, 0
num_config.m_rbs_engine_block = 30 * 1e3
num_config.Jbar_rbs_engine_block = np.array([[1e7, 0, 0],
                                             [0, 1e7, 0],
                                             [0, 0, 1e7]])

num_config.Kt_fas_bush_1 = 2 * 1e8
num_config.Kt_fas_bush_2 = 2 * 1e8
num_config.Kt_fas_bush_3 = 2 * 1e8
num_config.Kt_fas_bush_4 = 2 * 1e8

num_config.Ct_fas_bush_1 = 3 * 1e5
num_config.Ct_fas_bush_2 = 3 * 1e5
num_config.Ct_fas_bush_3 = 3 * 1e5
num_config.Ct_fas_bush_4 = 3 * 1e5

def drive_torque(t):
    peak = 100*1e6
    if t < 1:
        torque = 0
    elif t >= 1 and t < 2:
        torque = peak * (t - 1)/(1)
    else:
        torque = peak
    return torque

num_config.UF_fas_drive = drive_torque

# Assembling the configuration and exporting a .json file that
# holds these numerical values
num_config.assemble()
num_config.export_json()

# ================================================================== #
#                   Creating the Simulation Instance
# ================================================================== #
sim = simulation('sim', num_model, 'dds')

# setting the simulation time grid
sim.set_time_array(5, 1e-3)

# Starting the simulation
sim.solve()

# Saving the results in the /results directory as csv and npz
sim.save_as_csv('results', 'run_1')
sim.save_as_npz('results', 'run_1')

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

sim.soln.pos_dataframe.plot(x='time', y='rbs_piston_1.z', grid=True, figsize=(10,4))
sim.soln.pos_dataframe.plot(x='time', y='rbs_piston_2.z', grid=True, figsize=(10,4))
sim.soln.pos_dataframe.plot(x='time', y='rbs_piston_3.z', grid=True, figsize=(10,4))
sim.soln.pos_dataframe.plot(x='time', y='rbs_piston_4.z', grid=True, figsize=(10,4))

sim.soln.pos_dataframe.plot(x='time', y='rbs_engine_block.z', grid=True, figsize=(10,4))
sim.soln.vel_dataframe.plot(x='time', y='rbs_engine_block.z', grid=True, figsize=(10,4))
sim.soln.acc_dataframe.plot(x='time', y='rbs_engine_block.z', grid=True, figsize=(10,4))

plt.show()