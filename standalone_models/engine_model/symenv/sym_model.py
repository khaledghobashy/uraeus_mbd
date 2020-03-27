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
project_name = 'engine_4c'
sym_model = standalone_topology(project_name)

# Adding Bodies
# =============
sym_model.add_body('piston_1')
sym_model.add_body('piston_2')
sym_model.add_body('piston_3')
sym_model.add_body('piston_4')

sym_model.add_body('connect_1')
sym_model.add_body('connect_2')
sym_model.add_body('connect_3')
sym_model.add_body('connect_4')

sym_model.add_body('crank_shaft')
sym_model.add_body('engine_block')


# Adding Joints
# =============
sym_model.add_joint.cylinderical('cyl_1', 'rbs_piston_1', 'rbs_engine_block')
sym_model.add_joint.cylinderical('cyl_2', 'rbs_piston_2', 'rbs_engine_block')
sym_model.add_joint.cylinderical('cyl_3', 'rbs_piston_3', 'rbs_engine_block')
sym_model.add_joint.cylinderical('cyl_4', 'rbs_piston_4', 'rbs_engine_block')

sym_model.add_joint.spherical('sph_1', 'rbs_piston_1', 'rbs_connect_1')
sym_model.add_joint.spherical('sph_2', 'rbs_piston_2', 'rbs_connect_2')
sym_model.add_joint.spherical('sph_3', 'rbs_piston_3', 'rbs_connect_3')
sym_model.add_joint.spherical('sph_4', 'rbs_piston_4', 'rbs_connect_4')

sym_model.add_joint.spherical('bsph_1', 'rbs_crank_shaft', 'rbs_connect_1')
sym_model.add_joint.spherical('bsph_2', 'rbs_crank_shaft', 'rbs_connect_2')
sym_model.add_joint.spherical('bsph_3', 'rbs_crank_shaft', 'rbs_connect_3')
sym_model.add_joint.spherical('bsph_4', 'rbs_crank_shaft', 'rbs_connect_4')

sym_model.add_joint.revolute('crank_joint', 'rbs_crank_shaft', 'rbs_engine_block')


# Adding Bushes
# =============
sym_model.add_force.isotropic_bushing('bush_1', 'ground', 'rbs_engine_block')
sym_model.add_force.isotropic_bushing('bush_2', 'ground', 'rbs_engine_block')
sym_model.add_force.isotropic_bushing('bush_3', 'ground', 'rbs_engine_block')
sym_model.add_force.isotropic_bushing('bush_4', 'ground', 'rbs_engine_block')

# Adding Actuators
#sym_model.add_actuator.rotational_actuator('drive', 'jcs_crank_joint')

# Adding Forces
# =============
sym_model.add_force.local_torque('drive', 'rbs_crank_shaft')

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
sym_config.add_point.UserInput('front')
sym_config.add_point.UserInput('rear')

sym_config.add_point.UserInput('a1')
sym_config.add_point.UserInput('a2')
sym_config.add_point.UserInput('a3')
sym_config.add_point.UserInput('a4')

sym_config.add_point.UserInput('b1')
sym_config.add_point.UserInput('b2')
sym_config.add_point.UserInput('b3')
sym_config.add_point.UserInput('b4')

sym_config.add_point.UserInput('center')

sym_config.add_point.UserInput('p1')
sym_config.add_point.UserInput('p2')
sym_config.add_point.UserInput('p3')
sym_config.add_point.UserInput('p4')

sym_config.add_vector.UserInput('x')
sym_config.add_vector.UserInput('y')
sym_config.add_vector.UserInput('z')


# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Cylinderical Joint (cyl_1) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_cyl_1', ('hps_a1',))
sym_config.add_relation.Equal_to('ax1_jcs_cyl_1', ('vcs_z',))

# Cylinderical Joint (cyl_2) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_cyl_2', ('hps_a2',))
sym_config.add_relation.Equal_to('ax1_jcs_cyl_2', ('vcs_z',))

# Cylinderical Joint (cyl_3) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_cyl_3', ('hps_a3',))
sym_config.add_relation.Equal_to('ax1_jcs_cyl_3', ('vcs_z',))

# Cylinderical Joint (cyl_4) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_cyl_4', ('hps_a4',))
sym_config.add_relation.Equal_to('ax1_jcs_cyl_4', ('vcs_z',))

# ===================================================

# Top Spherical Joint (sph_1) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_sph_1', ('hps_a1',))
sym_config.add_relation.Equal_to('ax1_jcs_sph_1', ('vcs_z',))

# Top Spherical Joint (sph_2) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_sph_2', ('hps_a2',))
sym_config.add_relation.Equal_to('ax1_jcs_sph_2', ('vcs_z',))

# Top Spherical Joint (sph_3) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_sph_3', ('hps_a3',))
sym_config.add_relation.Equal_to('ax1_jcs_sph_3', ('vcs_z',))

# Top Spherical Joint (sph_4) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_sph_4', ('hps_a4',))
sym_config.add_relation.Equal_to('ax1_jcs_sph_4', ('vcs_z',))

# ===================================================

# Bottom Spherical Joint (bsph_1) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_bsph_1', ('hps_b1',))
sym_config.add_relation.Equal_to('ax1_jcs_bsph_1', ('vcs_z',))

# Bottom Spherical Joint (bsph_2) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_bsph_2', ('hps_b2',))
sym_config.add_relation.Equal_to('ax1_jcs_bsph_2', ('vcs_z',))

# Bottom Spherical Joint (bsph_3) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_bsph_3', ('hps_b3',))
sym_config.add_relation.Equal_to('ax1_jcs_bsph_3', ('vcs_z',))

# Bottom Spherical Joint (bsph_4) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_bsph_4', ('hps_b4',))
sym_config.add_relation.Equal_to('ax1_jcs_bsph_4', ('vcs_z',))

# ===================================================

# Bottom Revolute Joint (crank_joint) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_crank_joint', ('hps_center',))
sym_config.add_relation.Equal_to('ax1_jcs_crank_joint', ('vcs_x',))

# ===================================================

# Bottom Bush (bush_1) location and orientation
sym_config.add_relation.Equal_to('pt1_fas_bush_1', ('hps_p1',))
sym_config.add_relation.Equal_to('ax1_fas_bush_1', ('vcs_z',))

# Bottom Bush (bush_2) location and orientation
sym_config.add_relation.Equal_to('pt1_fas_bush_2', ('hps_p2',))
sym_config.add_relation.Equal_to('ax1_fas_bush_2', ('vcs_z',))

# Bottom Bush (bush_3) location and orientation
sym_config.add_relation.Equal_to('pt1_fas_bush_3', ('hps_p3',))
sym_config.add_relation.Equal_to('ax1_fas_bush_3', ('vcs_z',))

# Bottom Bush (bush_4) location and orientation
sym_config.add_relation.Equal_to('pt1_fas_bush_4', ('hps_p4',))
sym_config.add_relation.Equal_to('ax1_fas_bush_4', ('vcs_z',))

# Drive Torque Axis
sym_config.add_relation.Equal_to('ax1_fas_drive', ('vcs_x',))

# ===================================================

# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('piston_radius')
sym_config.add_scalar.UserInput('connect_radius')
sym_config.add_scalar.UserInput('crank_radius')


sym_config.add_geometry.Sphere_Geometry('piston_1', ('hps_a1', 's_piston_radius'))
sym_config.assign_geometry_to_body('rbs_piston_1', 'gms_piston_1')

sym_config.add_geometry.Sphere_Geometry('piston_2', ('hps_a2', 's_piston_radius'))
sym_config.assign_geometry_to_body('rbs_piston_2', 'gms_piston_2')

sym_config.add_geometry.Sphere_Geometry('piston_3', ('hps_a3', 's_piston_radius'))
sym_config.assign_geometry_to_body('rbs_piston_3', 'gms_piston_3')

sym_config.add_geometry.Sphere_Geometry('piston_4', ('hps_a4', 's_piston_radius'))
sym_config.assign_geometry_to_body('rbs_piston_4', 'gms_piston_4')


sym_config.add_geometry.Cylinder_Geometry('connect_1', ('hps_a1', 'hps_b1', 's_connect_radius'))
sym_config.assign_geometry_to_body('rbs_connect_1', 'gms_connect_1')

sym_config.add_geometry.Cylinder_Geometry('connect_2', ('hps_a2', 'hps_b2', 's_connect_radius'))
sym_config.assign_geometry_to_body('rbs_connect_2', 'gms_connect_2')

sym_config.add_geometry.Cylinder_Geometry('connect_3', ('hps_a3', 'hps_b3', 's_connect_radius'))
sym_config.assign_geometry_to_body('rbs_connect_3', 'gms_connect_3')

sym_config.add_geometry.Cylinder_Geometry('connect_4', ('hps_a4', 'hps_b4', 's_connect_radius'))
sym_config.assign_geometry_to_body('rbs_connect_4', 'gms_connect_4')

sym_config.add_geometry.Cylinder_Geometry('crank_shaft', ('hps_front', 'hps_rear', 's_crank_radius'))
sym_config.assign_geometry_to_body('rbs_crank_shaft', 'gms_crank_shaft')

# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import standalone_project
project = standalone_project()
project.create_dirs()

project.write_topology_code(sym_model)
