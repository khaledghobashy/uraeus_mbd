# standard library imports
import os

# uraeus imports
from uraeus.smbd.systems import standalone_topology, configuration

# getting directory of current file and specifying the directory
# where data will be saved
dir_name = os.path.dirname(__file__)
data_dir = os.path.join(dir_name, 'data')

# ============================================================= #
#                       Symbolic Topology
# ============================================================= #

# Creating the symbolic topology as an instance of the
# standalone_topology class
# TODO : User Declared
project_name = 'tchebycheff'
sym_model = standalone_topology(project_name)

# Adding Bodies
# =============
sym_model.add_body('l1')
sym_model.add_body('l2')
sym_model.add_body('l3')
sym_model.add_body('l4')
sym_model.add_body('l5')

# Adding Joints
# =============
sym_model.add_joint.revolute('a', 'ground', 'rbs_l1')
sym_model.add_joint.revolute('b', 'ground', 'rbs_l2')
sym_model.add_joint.cylinderical('c', 'rbs_l1', 'rbs_l3')
sym_model.add_joint.spherical('d', 'rbs_l2', 'rbs_l3')

sym_model.add_joint.cylinderical('e', 'rbs_l3', 'rbs_l4')
sym_model.add_joint.spherical('f', 'rbs_l4', 'rbs_l5')
sym_model.add_joint.cylinderical('h', 'rbs_l5', 'rbs_l2')


# Adding Actuators
# ================
sym_model.add_actuator.rotational_actuator('act', 'jcs_a')

# Adding Forces
# =============


# Assembling and Saving the model
sym_model.assemble()
sym_model.save(data_dir)

# ============================================================= #
#                     Symbolic Configuration
# ============================================================= #

# Symbolic configuration name.
config_name = '%s_cfg'%project_name

# Symbolic configuration instance.
sym_config = configuration(config_name, sym_model)

# Adding the desired set of UserInputs
# ====================================
sym_config.add_point.UserInput('p1')
sym_config.add_point.UserInput('p2')
sym_config.add_point.UserInput('p3')
sym_config.add_point.UserInput('p4')
sym_config.add_point.UserInput('p5')
sym_config.add_point.UserInput('p6')
sym_config.add_point.UserInput('p7')

sym_config.add_vector.UserInput('x')
sym_config.add_vector.UserInput('y')
sym_config.add_vector.UserInput('z')


# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_a', ('hps_p1',))
sym_config.add_relation.Equal_to('ax1_jcs_a', ('vcs_x',))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_b', ('hps_p2',))
sym_config.add_relation.Equal_to('ax1_jcs_b', ('vcs_x',))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_c', ('hps_p3',))
sym_config.add_relation.Equal_to('ax1_jcs_c', ('vcs_x',))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_d', ('hps_p4',))
sym_config.add_relation.Equal_to('ax1_jcs_d', ('vcs_x',))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_e', ('hps_p5',))
sym_config.add_relation.Equal_to('ax1_jcs_e', ('vcs_x',))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_f', ('hps_p6',))
sym_config.add_relation.Equal_to('ax1_jcs_f', ('vcs_x',))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_h', ('hps_p7',))
sym_config.add_relation.Equal_to('ax1_jcs_h', ('vcs_x',))

# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('radius')

sym_config.add_geometry.Cylinder_Geometry('l1', ('hps_p1', 'hps_p3', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l1', 'gms_l1')

sym_config.add_geometry.Cylinder_Geometry('l2', ('hps_p2', 'hps_p4', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l2', 'gms_l2')

sym_config.add_geometry.Cylinder_Geometry('l3', ('hps_p3', 'hps_p4', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l3', 'gms_l3')

sym_config.add_geometry.Cylinder_Geometry('l4', ('hps_p5', 'hps_p6', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l4', 'gms_l4')

sym_config.add_geometry.Cylinder_Geometry('l5', ('hps_p6', 'hps_p7', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l5', 'gms_l5')

# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import standalone_project
project = standalone_project()
project.create_dirs()

project.write_topology_code(sym_model)
