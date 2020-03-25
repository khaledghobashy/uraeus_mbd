
# uraeus imports
from uraeus.smbd.systems import standalone_project
from uraeus.smbd.systems import standalone_topology, configuration

# Creating project directories to store the various
# project files
project = standalone_project()
project.create()

# ============================================================= #
#                       Symbolic Topology
# ============================================================= #

# Creating the symbolic topology as an instance of the
# standalone_topology class
project_name = 'spatial_four_bar'
sym_model = standalone_topology(project_name)

# Adding Bodies
# =============
sym_model.add_body('l1')
sym_model.add_body('l2')
sym_model.add_body('l3')

# Adding Joints
# =============
sym_model.add_joint.revolute('a','ground','rbs_l1')
sym_model.add_joint.spherical('b','rbs_l1','rbs_l2')
sym_model.add_joint.universal('c','rbs_l2','rbs_l3')
sym_model.add_joint.revolute('d','rbs_l3','ground')

# Adding Actuators
# ================
sym_model.add_actuator.rotational_actuator('act', 'jcs_a')

# Assembling and Saving model
sym_model.assemble()
sym_model.save('symenv')

# ============================================================= #
#                     Symbolic Configuration
# ============================================================= #

# Symbolic configuration name.
config_name = '%s_cfg'%project_name

# Symbolic configuration instance.
sym_config = configuration(config_name, sym_model)

# Adding the desired set of UserInputs
# ====================================
sym_config.add_point.UserInput('a')
sym_config.add_point.UserInput('b')
sym_config.add_point.UserInput('c')
sym_config.add_point.UserInput('d')

sym_config.add_vector.UserInput('x')
sym_config.add_vector.UserInput('y')
sym_config.add_vector.UserInput('z')

# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_a', ('hps_a',))
sym_config.add_relation.Equal_to('ax1_jcs_a', ('vcs_x',))

# Spherical Joint (b) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_b', ('hps_b',))
sym_config.add_relation.Equal_to('ax1_jcs_b', ('vcs_z',))

# Universal Joint (c) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_c', ('hps_c',))
sym_config.add_relation.Oriented('ax1_jcs_c', ('hps_b', 'hps_c'))
sym_config.add_relation.Oriented('ax2_jcs_c', ('hps_c', 'hps_b'))

# Revolute Joint (d) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_d', ('hps_d',))
sym_config.add_relation.Equal_to('ax1_jcs_d', ('vcs_y',))

# Creating Geometries
# ===================
# links radius
sym_config.add_scalar.UserInput('links_ro')

# Link 1 geometry
sym_config.add_geometry.Cylinder_Geometry('l1', ('hps_a','hps_b','s_links_ro'))
sym_config.assign_geometry_to_body('rbs_l1', 'gms_l1')

# Link 2 geometry
sym_config.add_geometry.Cylinder_Geometry('l2', ('hps_b','hps_c','s_links_ro'))
sym_config.assign_geometry_to_body('rbs_l2', 'gms_l2')

# Link 3 geometry
sym_config.add_geometry.Cylinder_Geometry('l3', ('hps_c','hps_d','s_links_ro'))
sym_config.assign_geometry_to_body('rbs_l3', 'gms_l3')

# Exporing the configuration as a JSON file
sym_config.export_JSON_file('symenv')

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import standalone_project
project = standalone_project()
project.create_dirs()

project.write_topology_code(sym_model)
