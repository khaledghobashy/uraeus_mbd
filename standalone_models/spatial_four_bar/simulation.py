import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from uraeus.nmbd.python import multibody_system, simulation, configuration


from numenv.python.src import spatial_four_bar

num_model  = multibody_system(spatial_four_bar)

num_config = configuration('base')
num_config.construct_from_json('symenv/spatial_four_bar_cfg.json')

num_model.topology.config = num_config

# Specifying the Numerical Configuration of the System.
# ====================================================

num_config.hps_a.flat[:] = 0, 0, 0
num_config.hps_b.flat[:] = 0, 0, 200
num_config.hps_c.flat[:] = -750, -850, 650
num_config.hps_d.flat[:] = -400, -850, 0

num_config.vcs_x.flat[:] = 1, 0, 0
num_config.vcs_y.flat[:] = 0, 1, 0
num_config.vcs_z.flat[:] = 0, 0, 1

num_config.s_radius = 20

num_config.UF_mcs_act = lambda t : -np.deg2rad(360)*t

num_config.assemble()
num_config.export_json()

sim = simulation('sim', num_model, 'kds')
sim.set_time_array(5, 5e-3)
sim.solve()
sim.eval_reactions()
sim.save_as_csv('results', 'test_1')
sim.save_as_npz('results', 'test_1')



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