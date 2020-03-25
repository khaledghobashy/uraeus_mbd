
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
project_name = 'mass_spring_damper_2DOF'
sym_model = standalone_topology(project_name)

# Adding Bodies
# =============
sym_model.add_body('body_1')
sym_model.add_body('body_2')


# Adding Joints
# =============
sym_model.add_joint.translational('trans_1', 'rbs_body_1', 'ground')
sym_model.add_joint.translational('trans_2', 'rbs_body_2', 'ground')

# Adding Spring-Damper
# ====================
sym_model.add_force.TSDA('TSDA_1', 'rbs_body_1', 'ground')
sym_model.add_force.TSDA('TSDA_2', 'rbs_body_2', 'rbs_body_1')

# Assembling and Saving the model
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
sym_config.add_point.UserInput('p1')
sym_config.add_point.UserInput('p2')
sym_config.add_point.UserInput('p3')

sym_config.add_vector.UserInput('v')


# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_trans_1', ('hps_p1',))
sym_config.add_relation.Equal_to('ax1_jcs_trans_1', ('vcs_v',))

sym_config.add_relation.Equal_to('pt1_jcs_trans_2', ('hps_p2',))
sym_config.add_relation.Equal_to('ax1_jcs_trans_2', ('vcs_v',))

sym_config.add_relation.Equal_to('pt1_fas_TSDA_1', ('hps_p1',))
sym_config.add_relation.Equal_to('pt2_fas_TSDA_1', ('hps_p2',))

sym_config.add_relation.Equal_to('pt1_fas_TSDA_2', ('hps_p2',))
sym_config.add_relation.Equal_to('pt2_fas_TSDA_2', ('hps_p3',))


# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('radius')

sym_config.add_geometry.Sphere_Geometry('body_1', ('hps_p2', 's_radius'))
sym_config.assign_geometry_to_body('rbs_body_1', 'gms_body_1')

sym_config.add_geometry.Sphere_Geometry('body_2', ('hps_p3', 's_radius'))
sym_config.assign_geometry_to_body('rbs_body_2', 'gms_body_2')

# Exporing the configuration as a JSON file
sym_config.export_JSON_file('symenv')

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import standalone_project
project = standalone_project()
project.create_dirs()

project.write_topology_code(sym_model)
