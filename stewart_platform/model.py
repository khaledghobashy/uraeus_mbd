
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
project_name = 'stewart_platform'
sym_model = standalone_topology(project_name)

# Adding Bodies
# =============
sym_model.add_body('rocker_1')
sym_model.add_body('rocker_2')
sym_model.add_body('rocker_3')

sym_model.add_body('link_1')
sym_model.add_body('link_2')
sym_model.add_body('link_3')

sym_model.add_body('strut_lower')
sym_model.add_body('strut_upper')

sym_model.add_body('table')

# Adding Joints
# =============

# Revolute joints connecting the rockers to the ground.
sym_model.add_joint.revolute('rev_1', 'ground', 'rbs_rocker_1')
sym_model.add_joint.revolute('rev_2', 'ground', 'rbs_rocker_2')
sym_model.add_joint.revolute('rev_3', 'ground', 'rbs_rocker_3')

# Sperical joints connecting the connecting-rods to the rockers.
sym_model.add_joint.revolute('bottom_rev_1', 'rbs_rocker_1', 'rbs_link_1')
sym_model.add_joint.revolute('bottom_rev_2', 'rbs_rocker_2', 'rbs_link_2')
sym_model.add_joint.revolute('bottom_rev_3', 'rbs_rocker_3', 'rbs_link_3')

# Universal joints connecting the moving-table to the connecting-rods.
sym_model.add_joint.spherical('upper_sph_1', 'rbs_link_1', 'rbs_table')
sym_model.add_joint.spherical('upper_sph_2', 'rbs_link_2', 'rbs_table')
sym_model.add_joint.spherical('upper_sph_3', 'rbs_link_3', 'rbs_table')

# Universal joint connecting the strut_upper to the moving-table.
sym_model.add_joint.universal('strut_upper', 'rbs_strut_upper', 'rbs_table')

# Universal joint connecting the strut_lower to the ground.
sym_model.add_joint.universal('strut_lower', 'rbs_strut_lower', 'ground')

# Cylinderical joint connecting the strut_lower to the strut_upper.
sym_model.add_joint.cylinderical('strut_cyl', 'rbs_strut_lower', 'rbs_strut_upper')

# Adding Actuators
# ================
# Forward Kinematics Actuators
sym_model.add_actuator.rotational_actuator('act_1', 'jcs_rev_1')
sym_model.add_actuator.rotational_actuator('act_2', 'jcs_rev_2')
sym_model.add_actuator.rotational_actuator('act_3', 'jcs_rev_3')

# Inverse Kinematics Actuators
#sym_model.add_actuator.absolute_rotator('pitch', 'rbs_table', 'ground', '')
#sym_model.add_actuator.absolute_rotator('roll', 'rbs_table', 'ground', '')
#sym_model.add_actuator.absolute_locator('travel', 'rbs_table', 'ground', 'z')

# Adding Forces
# =============
sym_model.add_force.TSDA('strut', 'rbs_strut_upper', 'rbs_strut_lower')


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
sym_config.add_point.UserInput('bottom_1')
sym_config.add_point.UserInput('bottom_2')
sym_config.add_point.UserInput('bottom_3')

sym_config.add_point.UserInput('middle_1')
sym_config.add_point.UserInput('middle_2')
sym_config.add_point.UserInput('middle_3')

sym_config.add_point.UserInput('upper_1')
sym_config.add_point.UserInput('upper_2')
sym_config.add_point.UserInput('upper_3')

sym_config.add_point.UserInput('strut_upper')
sym_config.add_point.UserInput('strut_lower')

sym_config.add_vector.UserInput('x')
sym_config.add_vector.UserInput('y')
sym_config.add_vector.UserInput('z')

# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Bottom Revolute Joints:
sym_config.add_relation.Equal_to('pt1_jcs_rev_1', ('hps_bottom_1',))
#sym_config.add_relation.Equal_to('ax1_jcs_rev_1', ('vcs_y',))

sym_config.add_relation.Equal_to('pt1_jcs_rev_2', ('hps_bottom_2',))
#sym_config.add_relation.Equal_to('ax1_jcs_rev_2', ('vcs_y',))

sym_config.add_relation.Equal_to('pt1_jcs_rev_3', ('hps_bottom_3',))
#sym_config.add_relation.Equal_to('ax1_jcs_rev_3', ('vcs_y',))

# Bottom Revolute Joints:
sym_config.add_relation.Equal_to('pt1_jcs_bottom_rev_1', ('hps_middle_1',))
sym_config.add_relation.Equal_to('ax1_jcs_bottom_rev_1', ('ax1_jcs_rev_1',))

sym_config.add_relation.Equal_to('pt1_jcs_bottom_rev_2', ('hps_middle_2',))
sym_config.add_relation.Equal_to('ax1_jcs_bottom_rev_2', ('ax1_jcs_rev_2',))

sym_config.add_relation.Equal_to('pt1_jcs_bottom_rev_3', ('hps_middle_3',))
sym_config.add_relation.Equal_to('ax1_jcs_bottom_rev_3', ('ax1_jcs_rev_3',))

# Upper Spherical Joints:
sym_config.add_relation.Equal_to('pt1_jcs_upper_sph_1', ('hps_upper_1',))
sym_config.add_relation.Equal_to('ax1_jcs_upper_sph_1', ('vcs_z',))

sym_config.add_relation.Equal_to('pt1_jcs_upper_sph_2', ('hps_upper_2',))
sym_config.add_relation.Equal_to('ax1_jcs_upper_sph_2', ('vcs_z',))

sym_config.add_relation.Equal_to('pt1_jcs_upper_sph_3', ('hps_upper_3',))
sym_config.add_relation.Equal_to('ax1_jcs_upper_sph_3', ('vcs_z',))

# Strut_Upper Universal Joint:
sym_config.add_relation.Equal_to('pt1_jcs_strut_upper', ('hps_strut_upper',))
sym_config.add_relation.Oriented('ax1_jcs_strut_upper', ('hps_strut_upper', 'hps_strut_lower'))
sym_config.add_relation.Oriented('ax2_jcs_strut_upper', ('hps_strut_lower', 'hps_strut_upper'))

# Strut_Lower Universal Joint:
sym_config.add_relation.Equal_to('pt1_jcs_strut_lower', ('hps_strut_lower',))
sym_config.add_relation.Oriented('ax1_jcs_strut_lower', ('hps_strut_upper', 'hps_strut_lower'))
sym_config.add_relation.Oriented('ax2_jcs_strut_lower', ('hps_strut_lower', 'hps_strut_upper'))

# Strut Cylinderical Joint:
sym_config.add_point.Centered('strut_mid',  ('hps_strut_upper', 'hps_strut_lower'))
sym_config.add_relation.Centered('pt1_jcs_strut_cyl', ('hps_strut_mid',))
sym_config.add_relation.Oriented('ax1_jcs_strut_cyl', ('hps_strut_upper', 'hps_strut_lower'))

# Strut Force Points
sym_config.add_relation.Equal_to('pt1_fas_strut', ('hps_strut_upper',))
sym_config.add_relation.Equal_to('pt2_fas_strut', ('hps_strut_lower',))


# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('links_ro')
sym_config.add_scalar.UserInput('rockers_ro')
sym_config.add_scalar.UserInput('strut_upper_radius')
sym_config.add_scalar.UserInput('strut_lower_radius')

sym_config.add_geometry.Triangular_Prism('table', ('hps_upper_1','hps_upper_2','hps_upper_3','s_rockers_ro'))
sym_config.assign_geometry_to_body('rbs_table', 'gms_table')

sym_config.add_geometry.Cylinder_Geometry('rocker_1', ('hps_bottom_1','hps_middle_1','s_rockers_ro'))
sym_config.assign_geometry_to_body('rbs_rocker_1', 'gms_rocker_1')

sym_config.add_geometry.Cylinder_Geometry('rocker_2', ('hps_bottom_2','hps_middle_2','s_rockers_ro'))
sym_config.assign_geometry_to_body('rbs_rocker_2', 'gms_rocker_2')

sym_config.add_geometry.Cylinder_Geometry('rocker_3', ('hps_bottom_3','hps_middle_3','s_rockers_ro'))
sym_config.assign_geometry_to_body('rbs_rocker_3', 'gms_rocker_3')

sym_config.add_geometry.Cylinder_Geometry('link_1', ('hps_upper_1','hps_middle_1','s_links_ro'))
sym_config.assign_geometry_to_body('rbs_link_1', 'gms_link_1')

sym_config.add_geometry.Cylinder_Geometry('link_2', ('hps_upper_2','hps_middle_2','s_links_ro'))
sym_config.assign_geometry_to_body('rbs_link_2', 'gms_link_2')

sym_config.add_geometry.Cylinder_Geometry('link_3', ('hps_upper_3','hps_middle_3','s_links_ro'))
sym_config.assign_geometry_to_body('rbs_link_3', 'gms_link_3')

sym_config.add_geometry.Cylinder_Geometry('strut_upper', ('hps_strut_upper','hps_strut_mid','s_strut_upper_radius'))
sym_config.assign_geometry_to_body('rbs_strut_upper', 'gms_strut_upper')

sym_config.add_geometry.Cylinder_Geometry('strut_lower', ('hps_strut_lower','hps_strut_mid','s_strut_lower_radius'))
sym_config.assign_geometry_to_body('rbs_strut_lower', 'gms_strut_lower')

# Exporing the configuration as a JSON file
sym_config.export_JSON_file('symenv')

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import standalone_project
project = standalone_project()
project.create_dirs()

project.write_topology_code(sym_model)
