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
project_name = 'pendulum'
sym_model = standalone_topology(project_name)

# Adding Bodies
# =============
sym_model.add_body('body')

# Adding Joints
# =============
sym_model.add_joint.revolute('a', 'ground', 'rbs_body')

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

sym_config.add_vector.UserInput('v')


# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_a', ('hps_p1',))
sym_config.add_relation.Equal_to('ax1_jcs_a', ('vcs_v',))

# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('radius')

sym_config.add_geometry.Sphere_Geometry('body', ('hps_p2', 's_radius'))
sym_config.assign_geometry_to_body('rbs_body', 'gms_body')

# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import standalone_project
project = standalone_project()
project.create_dirs()

project.write_topology_code(sym_model)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.cpp.codegen import standalone_project
project = standalone_project()
project.create_dirs()

project.write_topology_code(sym_model)