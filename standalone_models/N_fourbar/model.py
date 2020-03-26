
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
project_name = 'N_fourbar'
sym_model = standalone_topology(project_name)

loops = 3
# Adding Bodies
# =============
sym_model.add_body('l1')
sym_model.add_body('l2')
sym_model.add_body('l3')

sym_model.add_joint.revolute('j1', 'ground', 'rbs_l1')
sym_model.add_joint.spherical('j2', 'rbs_l1', 'rbs_l2')
sym_model.add_joint.spherical('j3', 'rbs_l2', 'rbs_l3')
sym_model.add_joint.revolute('j4', 'rbs_l3', 'ground')


l_end = 3
j_end = 4
for i in range(loops):
    l_new = l_end + 1
    rocker_1 = 'rbs_l%s'%l_end
    coupler = 'l%s'%(l_new)
    rocker  = 'l%s'%(l_new + 1)

    j1 = 'j%s'%(j_end + 1)
    j2 = 'j%s'%(j_end + 2)
    j3 = 'j%s'%(j_end + 3)

    sym_model.add_body(coupler)
    sym_model.add_body(rocker)

    sym_model.add_joint.spherical(j1, rocker_1, 'rbs_%s'%coupler)
    sym_model.add_joint.spherical(j2, 'rbs_%s'%coupler, 'rbs_%s'%rocker)
    sym_model.add_joint.revolute(j3, 'rbs_%s'%rocker, 'ground')

    l_end += 2
    j_end += 3



# Assembling and Saving the model
sym_model.assemble()
sym_model.save('symenv')

sym_model.topology.draw_constraints_topology()

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

sym_config.add_vector.UserInput('x')
sym_config.add_vector.UserInput('y')
sym_config.add_vector.UserInput('z')

p_end = 4
for i in range(loops):
    sym_config.add_point.UserInput('p%s'%(p_end + 1))
    sym_config.add_point.UserInput('p%s'%(p_end + 2))

    p_end += 2


# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Revolute Joint (j1) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_j1', ('hps_p1',))
sym_config.add_relation.Equal_to('ax1_jcs_j1', ('vcs_x',))

# Revolute Joint (j2) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_j2', ('hps_p2',))
sym_config.add_relation.Equal_to('ax1_jcs_j2', ('vcs_x',))

# Revolute Joint (j3) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_j3', ('hps_p3',))
sym_config.add_relation.Equal_to('ax1_jcs_j3', ('vcs_x',))

# Revolute Joint (j4) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_j4', ('hps_p4',))
sym_config.add_relation.Equal_to('ax1_jcs_j4', ('vcs_x',))

j_end = 4
p_end = 4
for i in range(loops):
    # Revolute Joint (j2) location and orientation
    sym_config.add_relation.Equal_to('pt1_jcs_j%s'%(j_end + 1), ('hps_p%s'%(p_end - 1),))
    sym_config.add_relation.Equal_to('ax1_jcs_j%s'%(j_end + 1), ('vcs_x',))

    # Revolute Joint (j3) location and orientation
    sym_config.add_relation.Equal_to('pt1_jcs_j%s'%(j_end + 2), ('hps_p%s'%(p_end + 1),))
    sym_config.add_relation.Equal_to('ax1_jcs_j%s'%(j_end + 2), ('vcs_x',))

    # Revolute Joint (j4) location and orientation
    sym_config.add_relation.Equal_to('pt1_jcs_j%s'%(j_end + 3), ('hps_p%s'%(p_end + 2),))
    sym_config.add_relation.Equal_to('ax1_jcs_j%s'%(j_end + 3), ('vcs_x',))

    j_end += 3
    p_end += 2


# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('radius')

sym_config.add_geometry.Cylinder_Geometry('l1', ('hps_p1', 'hps_p2', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l1', 'gms_l1')

sym_config.add_geometry.Cylinder_Geometry('l2', ('hps_p2', 'hps_p3', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l2', 'gms_l2')

sym_config.add_geometry.Cylinder_Geometry('l3', ('hps_p3', 'hps_p4', 's_radius'))
sym_config.assign_geometry_to_body('rbs_l3', 'gms_l3')

l_end = 3
p_end = 4
for i in range(loops):
    l1 = l_end + 1
    l2 = l_end + 2

    p_up1 = p_end - 1
    p_up2 = p_end + 1
    p_do1 = p_end + 2

    sym_config.add_geometry.Cylinder_Geometry('l%s'%l1, ('hps_p%s'%p_up1, 'hps_p%s'%p_up2, 's_radius'))
    sym_config.assign_geometry_to_body('rbs_l%s'%l1, 'gms_l%s'%l1)

    sym_config.add_geometry.Cylinder_Geometry('l%s'%l2, ('hps_p%s'%p_up2, 'hps_p%s'%p_do1, 's_radius'))
    sym_config.assign_geometry_to_body('rbs_l%s'%l2, 'gms_l%s'%l2)

    l_end += 2
    p_end += 2


# Exporing the configuration as a JSON file
sym_config.export_JSON_file('symenv')

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import standalone_project
project = standalone_project()
project.create_dirs()

project.write_topology_code(sym_model)
