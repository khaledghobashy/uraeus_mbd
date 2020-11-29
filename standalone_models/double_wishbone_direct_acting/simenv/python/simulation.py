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

# TODO : User Declared
model_name = 'double_wishbone_direct_acting'

# getting the configuration .json file
# TODO : User Declared
config_file = os.path.join(symdata_dir, 'double_wishbone_direct_acting_cfg.json')

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


# Tire Radius
TR = 546

# Upper Control Arms
num_config.hpr_ucaf.flat[:] = [-131, 282, 180 + TR]
num_config.hpr_ucar.flat[:] = [ 131, 282, 180 + TR]
num_config.hpr_ucao.flat[:] = [   0, 722, 187 + TR]

# Lower Control Arms
num_config.hpr_lcaf.flat[:] = [-300, 245, -106 + TR]
num_config.hpr_lcar.flat[:] = [ 300, 245, -106 + TR]
num_config.hpr_lcao.flat[:] = [   0, 776, -181 + TR]

# Tie-Rod
num_config.hpr_tri.flat[:] = [ 402, 267, 108 + TR]
num_config.hpr_tro.flat[:] = [ 399, 720, 110 + TR]

# Struts
num_config.hpr_strut_chassis.flat[:] = [-165, 534, 639 + TR]
num_config.hpr_strut_lca.flat[:]     = [-165, 534, -79 + TR]

num_config.far_strut_FL  = 794.6
num_config.pt1_far_strut = num_config.hpr_strut_chassis
num_config.pt2_far_strut = num_config.hpr_strut_lca


# Wheel Center
num_config.hpr_wc.flat[:]  = [0, 1032, 0 + TR]
num_config.hpr_wc1.flat[:] = [0, 1164, 0 + TR]
num_config.hpr_wc2.flat[:] = [0,  900, 0 + TR]
num_config.pt1_far_tire = num_config.hpr_wc


# Helpers
num_config.vcs_x.flat[:] = [1, 0, 0]
num_config.vcs_y.flat[:] = [0, 1, 0]
num_config.vcs_z.flat[:] = [0, 0, 1]

num_config.s_tire_radius  = TR
num_config.s_hub_radius  = 0.3 * TR
num_config.s_tire_radius = TR
num_config.s_links_ro    = 35
num_config.s_strut_inner = 45
num_config.s_strut_outer = 65
num_config.s_thickness   = 35

# Assembling the configuration and exporting a .json file that
# holds these numerical values
num_config.assemble()
num_config.export_json()

print(num_config.gml_hub_cyl.R)
print(num_config.gml_hub_cyl.P)
print(num_config.gml_hub_cyl.m)
print(num_config.gml_hub_cyl.J)
print('\n')
print(num_config.gml_hub.R)
print(num_config.gml_hub.P)
print(num_config.gml_hub.m)
print(num_config.gml_hub.J)

""" wheel_inertia =  np.array([[1*1e4, 0,      0 ],
                           [0    , 50*1e9, 0 ],
                           [0    , 0, 1*1e4  ]])


num_config.Jbar_rbr_hub = wheel_inertia
num_config.Jbar_rbl_hub = wheel_inertia

num_config.m_rbr_hub = 200*1e3
num_config.m_rbl_hub = 200*1e3 """


def strut_spring(x):
    x = float(x)
    #print('defflection = %s'%x)
    k = 550*1e6
    force = k * x if x > 0 else 0
    return 0

def strut_damping(v):
    v = float(v)
    #print('velocity = %s'%v)
    force = 40*1e6 * v
    return 0

def zero_func(t):
    return np.zeros((3,1), dtype=np.float64)

def wheel_lock(t, *args):
    return 0

def wheel_travel(t, *args):
    travel = 546 + 170*np.sin(t)
    return travel

num_config.UF_mcr_wheel_travel = lambda t: 546 + 170*np.sin(t)
num_config.UF_mcl_wheel_travel = lambda t: 546 + 170*np.sin(t)

num_config.UF_mcr_wheel_lock = wheel_lock
num_config.UF_mcl_wheel_lock = wheel_lock

num_config.UF_far_tire_F = zero_func
num_config.UF_fal_tire_F = zero_func

num_config.UF_far_tire_T = zero_func
num_config.UF_fal_tire_T = zero_func


num_config.UF_far_strut_Fs = strut_spring
num_config.UF_fal_strut_Fs = strut_spring
num_config.UF_far_strut_Fd = strut_damping
num_config.UF_fal_strut_Fd = strut_damping


# ================================================================== #
#                   Creating the Simulation Instance
# ================================================================== #
sim = simulation('sim', num_model, 'kds')

# setting the simulation time grid
sim.set_time_array(2*np.pi, 5e-3)

# Starting the simulation
sim.solve()

# Saving the results in the /results directory as csv and npz
sim.save_as_csv(results_dir, 'py_pos', 'pos')
sim.save_as_csv(results_dir, 'py_vel', 'vel')
sim.save_as_csv(results_dir, 'py_acc', 'acc')

sim.save_as_npz(results_dir, 'test_2')

sim.eval_reactions()
sim.soln.reactions_dataframe.to_csv(os.path.join(results_dir, 'rct_py.csv'))

# ================================================================== #
#                   Plotting the Simulation Results
# ================================================================== #

sim.soln.pos_dataframe.plot(x='time', 
                            y=['rbr_hub.z', 'rbl_hub.z'], 
                            grid=True)

sim.soln.pos_dataframe.plot(x='time', 
                            y=['rbr_hub.x', 'rbl_hub.x'], 
                            grid=True)

sim.soln.vel_dataframe.plot(x='time', 
                            y=['rbr_hub.x', 'rbl_hub.x'], 
                            grid=True)

sim.soln.pos_dataframe.plot(x='time', 
                            y=['rbr_hub.e0', 'rbr_hub.e1',
                               'rbr_hub.e2', 'rbr_hub.e3'], 
                            grid=True)

plt.show()
