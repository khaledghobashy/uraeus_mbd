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
project_name = 'parallel_table'
sym_model = standalone_topology(project_name)

# Adding Bodies
# =============
#sym_model.add_body('table')

sym_model.add_body('l1')
sym_model.add_body('l2')
sym_model.add_body('l3')

#sym_model.add_body('l4')
#sym_model.add_body('l5')
#sym_model.add_body('l6')

sym_model.add_body('l7')
sym_model.add_body('l8')
#sym_model.add_body('l9')

# Adding Joints
# =============
sym_model.add_joint.cylinderical('a', 'ground', 'rbs_l1')
sym_model.add_joint.revolute('b', 'ground', 'rbs_l2')
sym_model.add_joint.revolute('c', 'ground', 'rbs_l3')

#sym_model.add_joint.revolute('d', 'rbs_table', 'rbs_l4')
#sym_model.add_joint.revolute('e', 'rbs_table', 'rbs_l5')
#sym_model.add_joint.revolute('f', 'rbs_table', 'rbs_l6')

sym_model.add_joint.cylinderical('h1', 'rbs_l1', 'rbs_l7')
sym_model.add_joint.spherical('k1', 'rbs_l2', 'rbs_l7')
sym_model.add_joint.cylinderical('l1', 'rbs_l3', 'rbs_l8')

#sym_model.add_joint.cylinderical('h2', 'rbs_l4', 'rbs_l7')
#sym_model.add_joint.spherical('k2', 'rbs_l5', 'rbs_l7')
#sym_model.add_joint.cylinderical('l2', 'rbs_l6', 'rbs_l8')

sym_model.add_joint.cylinderical('trans', 'rbs_l7', 'rbs_l8')

# Adding Actuators
# ================
#sym_model.add_actuator.translational_actuator('act', 'jcs_trans')
#sym_model.add_actuator.rotational_actuator('act', 'jcs_a')

# Adding Forces
# =============
sym_model.add_force.TSDA('strut', 'rbs_l7', 'rbs_l8')


print(sym_model.topology.n, sym_model.topology.nc)
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

sym_config.add_vector.UserInput('x')
sym_config.add_vector.UserInput('y')
sym_config.add_vector.UserInput('z')

# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_a', ('hps_p1',))
sym_config.add_relation.Equal_to('ax1_jcs_a', ('vcs_x',))

# Revolute Joint (b) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_b', ('hps_p2',))
sym_config.add_relation.Equal_to('ax1_jcs_b', ('vcs_x',))

# Revolute Joint (c) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_c', ('hps_p3',))
sym_config.add_relation.Equal_to('ax1_jcs_c', ('vcs_x',))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_h1', ('hps_p4',))
sym_config.add_relation.Equal_to('ax1_jcs_h1', ('vcs_x',))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_k1', ('hps_p5',))
sym_config.add_relation.Equal_to('ax1_jcs_k1', ('vcs_x',))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_l1', ('hps_p6',))
sym_config.add_relation.Equal_to('ax1_jcs_l1', ('vcs_x',))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Centered('pt1_jcs_trans', ('hps_p6', 'hps_p5'))
sym_config.add_relation.Oriented('ax1_jcs_trans', ('hps_p6', 'hps_p5'))

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_fas_strut', ('hps_p5',))
sym_config.add_relation.Equal_to('pt2_fas_strut', ('hps_p6',))

# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('radius')

sym_config.add_geometry.Cylinder_Geometry('l1', ('hps_p1', 'hps_p4', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l1', 'gms_l1')

sym_config.add_geometry.Cylinder_Geometry('l2', ('hps_p2', 'hps_p5', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l2', 'gms_l2')

sym_config.add_geometry.Cylinder_Geometry('l3', ('hps_p3', 'hps_p6', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l3', 'gms_l3')

sym_config.add_geometry.Cylinder_Geometry('l7', ('hps_p4', 'hps_p5', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l7', 'gms_l7')

sym_config.add_geometry.Cylinder_Geometry('l8', ('hps_p5', 'hps_p6', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l8', 'gms_l8')


# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import standalone_project
project = standalone_project()
project.create_dirs()

project.write_topology_code(sym_model)
