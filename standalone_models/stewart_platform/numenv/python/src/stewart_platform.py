
import numpy as np
from numpy import cos, sin
from scipy.misc import derivative

from uraeus.nmbd.python.engine.numerics.math_funcs import A, B, G, E, triad, skew, multi_dot

# CONSTANTS
F64_DTYPE = np.float64

I1 = np.eye(1, dtype=F64_DTYPE)
I2 = np.eye(2, dtype=F64_DTYPE)
I3 = np.eye(3, dtype=F64_DTYPE)
I4 = np.eye(4, dtype=F64_DTYPE)

Z1x1 = np.zeros((1,1), F64_DTYPE)
Z1x3 = np.zeros((1,3), F64_DTYPE)
Z3x1 = np.zeros((3,1), F64_DTYPE)
Z3x4 = np.zeros((3,4), F64_DTYPE)
Z4x1 = np.zeros((4,1), F64_DTYPE)
Z4x3 = np.zeros((4,3), F64_DTYPE)



class topology(object):

    def __init__(self,prefix=''):
        self.t = 0.0
        self.prefix = (prefix if prefix=='' else prefix+'.')
        self.config = None

        self.indicies_map = {'ground': 0, 'rbs_rocker_1': 1, 'rbs_rocker_2': 2, 'rbs_rocker_3': 3, 'rbs_link_1': 4, 'rbs_link_2': 5, 'rbs_link_3': 6, 'rbs_strut_lower': 7, 'rbs_strut_upper': 8, 'rbs_table': 9}

        self.n  = 70
        self.nc = 70
        self.nrows = 43
        self.ncols = 2*10
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_ground_jcs_rev_1', 'T_ground_jcs_rev_1', 'F_ground_mcs_act_1', 'T_ground_mcs_act_1', 'F_ground_jcs_rev_2', 'T_ground_jcs_rev_2', 'F_ground_mcs_act_2', 'T_ground_mcs_act_2', 'F_ground_jcs_rev_3', 'T_ground_jcs_rev_3', 'F_ground_mcs_act_3', 'T_ground_mcs_act_3', 'F_rbs_rocker_1_jcs_bottom_rev_1', 'T_rbs_rocker_1_jcs_bottom_rev_1', 'F_rbs_rocker_2_jcs_bottom_rev_2', 'T_rbs_rocker_2_jcs_bottom_rev_2', 'F_rbs_rocker_3_jcs_bottom_rev_3', 'T_rbs_rocker_3_jcs_bottom_rev_3', 'F_rbs_link_1_jcs_upper_sph_1', 'T_rbs_link_1_jcs_upper_sph_1', 'F_rbs_link_2_jcs_upper_sph_2', 'T_rbs_link_2_jcs_upper_sph_2', 'F_rbs_link_3_jcs_upper_sph_3', 'T_rbs_link_3_jcs_upper_sph_3', 'F_rbs_strut_lower_jcs_strut_lower', 'T_rbs_strut_lower_jcs_strut_lower', 'F_rbs_strut_lower_jcs_strut_cyl', 'T_rbs_strut_lower_jcs_strut_cyl', 'F_rbs_strut_upper_jcs_strut_upper', 'T_rbs_strut_upper_jcs_strut_upper', 'F_rbs_strut_upper_fas_strut', 'T_rbs_strut_upper_fas_strut']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def initialize(self, q, qd, qdd, lgr):
        self.t = 0
        self.assemble(self.indicies_map, {}, 0)
        self._set_states_arrays(q, qd, qdd, lgr)
        self._map_states_arrays()
        self.set_initial_states()
        self.eval_constants()

    def assemble(self, indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.rbs_link_1*2, self.rbs_link_1*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.rbs_link_1*2, self.rbs_link_1*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.rbs_link_1*2, self.rbs_link_1*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.rbs_link_2*2, self.rbs_link_2*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.rbs_link_2*2, self.rbs_link_2*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.rbs_link_2*2, self.rbs_link_2*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.rbs_link_3*2, self.rbs_link_3*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.rbs_link_3*2, self.rbs_link_3*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.rbs_link_3*2, self.rbs_link_3*2+1, self.rbs_link_1*2, self.rbs_link_1*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_link_2*2, self.rbs_link_2*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_link_3*2, self.rbs_link_3*2+1, self.rbs_table*2, self.rbs_table*2+1, self.ground*2, self.ground*2+1, self.rbs_strut_lower*2, self.rbs_strut_lower*2+1, self.ground*2, self.ground*2+1, self.rbs_strut_lower*2, self.rbs_strut_lower*2+1, self.rbs_strut_lower*2, self.rbs_strut_lower*2+1, self.rbs_strut_upper*2, self.rbs_strut_upper*2+1, self.rbs_strut_lower*2, self.rbs_strut_lower*2+1, self.rbs_strut_upper*2, self.rbs_strut_upper*2+1, self.rbs_strut_lower*2, self.rbs_strut_lower*2+1, self.rbs_strut_upper*2, self.rbs_strut_upper*2+1, self.rbs_strut_lower*2, self.rbs_strut_lower*2+1, self.rbs_strut_upper*2, self.rbs_strut_upper*2+1, self.rbs_strut_upper*2, self.rbs_strut_upper*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_strut_upper*2, self.rbs_strut_upper*2+1, self.rbs_table*2, self.rbs_table*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.rbs_link_1*2, self.rbs_link_1*2+1, self.rbs_link_2*2, self.rbs_link_2*2+1, self.rbs_link_3*2, self.rbs_link_3*2+1, self.rbs_strut_lower*2, self.rbs_strut_lower*2+1, self.rbs_strut_upper*2, self.rbs_strut_upper*2+1, self.rbs_table*2, self.rbs_table*2+1], dtype=np.intc)

    def _set_states_arrays(self, q, qd, qdd, lgr):
        self._q = q
        self._qd = qd
        self._qdd = qdd
        self._lgr = lgr

    def _map_states_arrays(self):
        self._map_gen_coordinates()
        self._map_gen_velocities()
        self._map_gen_accelerations()
        self._map_lagrange_multipliers()

    def set_initial_states(self):
        np.concatenate([self.config.R_ground,
        self.config.P_ground,
        self.config.R_rbs_rocker_1,
        self.config.P_rbs_rocker_1,
        self.config.R_rbs_rocker_2,
        self.config.P_rbs_rocker_2,
        self.config.R_rbs_rocker_3,
        self.config.P_rbs_rocker_3,
        self.config.R_rbs_link_1,
        self.config.P_rbs_link_1,
        self.config.R_rbs_link_2,
        self.config.P_rbs_link_2,
        self.config.R_rbs_link_3,
        self.config.P_rbs_link_3,
        self.config.R_rbs_strut_lower,
        self.config.P_rbs_strut_lower,
        self.config.R_rbs_strut_upper,
        self.config.P_rbs_strut_upper,
        self.config.R_rbs_table,
        self.config.P_rbs_table], out=self._q)

        np.concatenate([self.config.Rd_ground,
        self.config.Pd_ground,
        self.config.Rd_rbs_rocker_1,
        self.config.Pd_rbs_rocker_1,
        self.config.Rd_rbs_rocker_2,
        self.config.Pd_rbs_rocker_2,
        self.config.Rd_rbs_rocker_3,
        self.config.Pd_rbs_rocker_3,
        self.config.Rd_rbs_link_1,
        self.config.Pd_rbs_link_1,
        self.config.Rd_rbs_link_2,
        self.config.Pd_rbs_link_2,
        self.config.Rd_rbs_link_3,
        self.config.Pd_rbs_link_3,
        self.config.Rd_rbs_strut_lower,
        self.config.Pd_rbs_strut_lower,
        self.config.Rd_rbs_strut_upper,
        self.config.Pd_rbs_strut_upper,
        self.config.Rd_rbs_table,
        self.config.Pd_rbs_table], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p + 'ground']
        self.rbs_rocker_1 = indicies_map[p + 'rbs_rocker_1']
        self.rbs_rocker_2 = indicies_map[p + 'rbs_rocker_2']
        self.rbs_rocker_3 = indicies_map[p + 'rbs_rocker_3']
        self.rbs_link_1 = indicies_map[p + 'rbs_link_1']
        self.rbs_link_2 = indicies_map[p + 'rbs_link_2']
        self.rbs_link_3 = indicies_map[p + 'rbs_link_3']
        self.rbs_strut_lower = indicies_map[p + 'rbs_strut_lower']
        self.rbs_strut_upper = indicies_map[p + 'rbs_strut_upper']
        self.rbs_table = indicies_map[p + 'rbs_table']
    

    
    def eval_constants(self):
        config = self.config

        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.m_ground = 1.0
        self.Jbar_ground = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.F_rbs_rocker_1_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_rocker_1]], dtype=np.float64)
        self.F_rbs_rocker_2_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_rocker_2]], dtype=np.float64)
        self.F_rbs_rocker_3_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_rocker_3]], dtype=np.float64)
        self.F_rbs_link_1_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_link_1]], dtype=np.float64)
        self.F_rbs_link_2_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_link_2]], dtype=np.float64)
        self.F_rbs_link_3_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_link_3]], dtype=np.float64)
        self.F_rbs_strut_lower_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_strut_lower]], dtype=np.float64)
        self.F_rbs_strut_upper_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_strut_upper]], dtype=np.float64)
        self.T_rbs_strut_upper_fas_strut = Z3x1
        self.T_rbs_strut_lower_fas_strut = Z3x1
        self.F_rbs_table_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_table]], dtype=np.float64)

        self.Mbar_ground_jcs_rev_1 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_rev_1)])
        self.Mbar_rbs_rocker_1_jcs_rev_1 = multi_dot([A(config.P_rbs_rocker_1).T,triad(config.ax1_jcs_rev_1)])
        self.ubar_ground_jcs_rev_1 = (multi_dot([A(self.P_ground).T,config.pt1_jcs_rev_1]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_rocker_1_jcs_rev_1 = (multi_dot([A(config.P_rbs_rocker_1).T,config.pt1_jcs_rev_1]) + (-1) * multi_dot([A(config.P_rbs_rocker_1).T,config.R_rbs_rocker_1]))
        self.Mbar_ground_jcs_rev_1 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_rev_1)])
        self.Mbar_rbs_rocker_1_jcs_rev_1 = multi_dot([A(config.P_rbs_rocker_1).T,triad(config.ax1_jcs_rev_1)])
        self.Mbar_ground_jcs_rev_2 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_rev_2)])
        self.Mbar_rbs_rocker_2_jcs_rev_2 = multi_dot([A(config.P_rbs_rocker_2).T,triad(config.ax1_jcs_rev_2)])
        self.ubar_ground_jcs_rev_2 = (multi_dot([A(self.P_ground).T,config.pt1_jcs_rev_2]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_rocker_2_jcs_rev_2 = (multi_dot([A(config.P_rbs_rocker_2).T,config.pt1_jcs_rev_2]) + (-1) * multi_dot([A(config.P_rbs_rocker_2).T,config.R_rbs_rocker_2]))
        self.Mbar_ground_jcs_rev_2 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_rev_2)])
        self.Mbar_rbs_rocker_2_jcs_rev_2 = multi_dot([A(config.P_rbs_rocker_2).T,triad(config.ax1_jcs_rev_2)])
        self.Mbar_ground_jcs_rev_3 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_rev_3)])
        self.Mbar_rbs_rocker_3_jcs_rev_3 = multi_dot([A(config.P_rbs_rocker_3).T,triad(config.ax1_jcs_rev_3)])
        self.ubar_ground_jcs_rev_3 = (multi_dot([A(self.P_ground).T,config.pt1_jcs_rev_3]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_rocker_3_jcs_rev_3 = (multi_dot([A(config.P_rbs_rocker_3).T,config.pt1_jcs_rev_3]) + (-1) * multi_dot([A(config.P_rbs_rocker_3).T,config.R_rbs_rocker_3]))
        self.Mbar_ground_jcs_rev_3 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_rev_3)])
        self.Mbar_rbs_rocker_3_jcs_rev_3 = multi_dot([A(config.P_rbs_rocker_3).T,triad(config.ax1_jcs_rev_3)])
        self.Mbar_rbs_rocker_1_jcs_bottom_rev_1 = multi_dot([A(config.P_rbs_rocker_1).T,triad(config.ax1_jcs_bottom_rev_1)])
        self.Mbar_rbs_link_1_jcs_bottom_rev_1 = multi_dot([A(config.P_rbs_link_1).T,triad(config.ax1_jcs_bottom_rev_1)])
        self.ubar_rbs_rocker_1_jcs_bottom_rev_1 = (multi_dot([A(config.P_rbs_rocker_1).T,config.pt1_jcs_bottom_rev_1]) + (-1) * multi_dot([A(config.P_rbs_rocker_1).T,config.R_rbs_rocker_1]))
        self.ubar_rbs_link_1_jcs_bottom_rev_1 = (multi_dot([A(config.P_rbs_link_1).T,config.pt1_jcs_bottom_rev_1]) + (-1) * multi_dot([A(config.P_rbs_link_1).T,config.R_rbs_link_1]))
        self.Mbar_rbs_rocker_2_jcs_bottom_rev_2 = multi_dot([A(config.P_rbs_rocker_2).T,triad(config.ax1_jcs_bottom_rev_2)])
        self.Mbar_rbs_link_2_jcs_bottom_rev_2 = multi_dot([A(config.P_rbs_link_2).T,triad(config.ax1_jcs_bottom_rev_2)])
        self.ubar_rbs_rocker_2_jcs_bottom_rev_2 = (multi_dot([A(config.P_rbs_rocker_2).T,config.pt1_jcs_bottom_rev_2]) + (-1) * multi_dot([A(config.P_rbs_rocker_2).T,config.R_rbs_rocker_2]))
        self.ubar_rbs_link_2_jcs_bottom_rev_2 = (multi_dot([A(config.P_rbs_link_2).T,config.pt1_jcs_bottom_rev_2]) + (-1) * multi_dot([A(config.P_rbs_link_2).T,config.R_rbs_link_2]))
        self.Mbar_rbs_rocker_3_jcs_bottom_rev_3 = multi_dot([A(config.P_rbs_rocker_3).T,triad(config.ax1_jcs_bottom_rev_3)])
        self.Mbar_rbs_link_3_jcs_bottom_rev_3 = multi_dot([A(config.P_rbs_link_3).T,triad(config.ax1_jcs_bottom_rev_3)])
        self.ubar_rbs_rocker_3_jcs_bottom_rev_3 = (multi_dot([A(config.P_rbs_rocker_3).T,config.pt1_jcs_bottom_rev_3]) + (-1) * multi_dot([A(config.P_rbs_rocker_3).T,config.R_rbs_rocker_3]))
        self.ubar_rbs_link_3_jcs_bottom_rev_3 = (multi_dot([A(config.P_rbs_link_3).T,config.pt1_jcs_bottom_rev_3]) + (-1) * multi_dot([A(config.P_rbs_link_3).T,config.R_rbs_link_3]))
        self.Mbar_rbs_link_1_jcs_upper_sph_1 = multi_dot([A(config.P_rbs_link_1).T,triad(config.ax1_jcs_upper_sph_1)])
        self.Mbar_rbs_table_jcs_upper_sph_1 = multi_dot([A(config.P_rbs_table).T,triad(config.ax1_jcs_upper_sph_1)])
        self.ubar_rbs_link_1_jcs_upper_sph_1 = (multi_dot([A(config.P_rbs_link_1).T,config.pt1_jcs_upper_sph_1]) + (-1) * multi_dot([A(config.P_rbs_link_1).T,config.R_rbs_link_1]))
        self.ubar_rbs_table_jcs_upper_sph_1 = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_upper_sph_1]) + (-1) * multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.Mbar_rbs_link_2_jcs_upper_sph_2 = multi_dot([A(config.P_rbs_link_2).T,triad(config.ax1_jcs_upper_sph_2)])
        self.Mbar_rbs_table_jcs_upper_sph_2 = multi_dot([A(config.P_rbs_table).T,triad(config.ax1_jcs_upper_sph_2)])
        self.ubar_rbs_link_2_jcs_upper_sph_2 = (multi_dot([A(config.P_rbs_link_2).T,config.pt1_jcs_upper_sph_2]) + (-1) * multi_dot([A(config.P_rbs_link_2).T,config.R_rbs_link_2]))
        self.ubar_rbs_table_jcs_upper_sph_2 = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_upper_sph_2]) + (-1) * multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.Mbar_rbs_link_3_jcs_upper_sph_3 = multi_dot([A(config.P_rbs_link_3).T,triad(config.ax1_jcs_upper_sph_3)])
        self.Mbar_rbs_table_jcs_upper_sph_3 = multi_dot([A(config.P_rbs_table).T,triad(config.ax1_jcs_upper_sph_3)])
        self.ubar_rbs_link_3_jcs_upper_sph_3 = (multi_dot([A(config.P_rbs_link_3).T,config.pt1_jcs_upper_sph_3]) + (-1) * multi_dot([A(config.P_rbs_link_3).T,config.R_rbs_link_3]))
        self.ubar_rbs_table_jcs_upper_sph_3 = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_upper_sph_3]) + (-1) * multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.Mbar_rbs_strut_lower_jcs_strut_lower = multi_dot([A(config.P_rbs_strut_lower).T,triad(config.ax1_jcs_strut_lower)])
        self.Mbar_ground_jcs_strut_lower = multi_dot([A(self.P_ground).T,triad(config.ax2_jcs_strut_lower,triad(config.ax1_jcs_strut_lower)[0:3,1:2])])
        self.ubar_rbs_strut_lower_jcs_strut_lower = (multi_dot([A(config.P_rbs_strut_lower).T,config.pt1_jcs_strut_lower]) + (-1) * multi_dot([A(config.P_rbs_strut_lower).T,config.R_rbs_strut_lower]))
        self.ubar_ground_jcs_strut_lower = (multi_dot([A(self.P_ground).T,config.pt1_jcs_strut_lower]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbs_strut_lower_jcs_strut_cyl = multi_dot([A(config.P_rbs_strut_lower).T,triad(config.ax1_jcs_strut_cyl)])
        self.Mbar_rbs_strut_upper_jcs_strut_cyl = multi_dot([A(config.P_rbs_strut_upper).T,triad(config.ax1_jcs_strut_cyl)])
        self.ubar_rbs_strut_lower_jcs_strut_cyl = (multi_dot([A(config.P_rbs_strut_lower).T,config.pt1_jcs_strut_cyl]) + (-1) * multi_dot([A(config.P_rbs_strut_lower).T,config.R_rbs_strut_lower]))
        self.ubar_rbs_strut_upper_jcs_strut_cyl = (multi_dot([A(config.P_rbs_strut_upper).T,config.pt1_jcs_strut_cyl]) + (-1) * multi_dot([A(config.P_rbs_strut_upper).T,config.R_rbs_strut_upper]))
        self.Mbar_rbs_strut_upper_jcs_strut_upper = multi_dot([A(config.P_rbs_strut_upper).T,triad(config.ax1_jcs_strut_upper)])
        self.Mbar_rbs_table_jcs_strut_upper = multi_dot([A(config.P_rbs_table).T,triad(config.ax2_jcs_strut_upper,triad(config.ax1_jcs_strut_upper)[0:3,1:2])])
        self.ubar_rbs_strut_upper_jcs_strut_upper = (multi_dot([A(config.P_rbs_strut_upper).T,config.pt1_jcs_strut_upper]) + (-1) * multi_dot([A(config.P_rbs_strut_upper).T,config.R_rbs_strut_upper]))
        self.ubar_rbs_table_jcs_strut_upper = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_strut_upper]) + (-1) * multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.ubar_rbs_strut_upper_fas_strut = (multi_dot([A(config.P_rbs_strut_upper).T,config.pt1_fas_strut]) + (-1) * multi_dot([A(config.P_rbs_strut_upper).T,config.R_rbs_strut_upper]))
        self.ubar_rbs_strut_lower_fas_strut = (multi_dot([A(config.P_rbs_strut_lower).T,config.pt2_fas_strut]) + (-1) * multi_dot([A(config.P_rbs_strut_lower).T,config.R_rbs_strut_lower]))

    
    def _map_gen_coordinates(self):
        q = self._q
        self.R_ground = q[0:3]
        self.P_ground = q[3:7]
        self.R_rbs_rocker_1 = q[7:10]
        self.P_rbs_rocker_1 = q[10:14]
        self.R_rbs_rocker_2 = q[14:17]
        self.P_rbs_rocker_2 = q[17:21]
        self.R_rbs_rocker_3 = q[21:24]
        self.P_rbs_rocker_3 = q[24:28]
        self.R_rbs_link_1 = q[28:31]
        self.P_rbs_link_1 = q[31:35]
        self.R_rbs_link_2 = q[35:38]
        self.P_rbs_link_2 = q[38:42]
        self.R_rbs_link_3 = q[42:45]
        self.P_rbs_link_3 = q[45:49]
        self.R_rbs_strut_lower = q[49:52]
        self.P_rbs_strut_lower = q[52:56]
        self.R_rbs_strut_upper = q[56:59]
        self.P_rbs_strut_upper = q[59:63]
        self.R_rbs_table = q[63:66]
        self.P_rbs_table = q[66:70]

    
    def _map_gen_velocities(self):
        qd = self._qd
        self.Rd_ground = qd[0:3]
        self.Pd_ground = qd[3:7]
        self.Rd_rbs_rocker_1 = qd[7:10]
        self.Pd_rbs_rocker_1 = qd[10:14]
        self.Rd_rbs_rocker_2 = qd[14:17]
        self.Pd_rbs_rocker_2 = qd[17:21]
        self.Rd_rbs_rocker_3 = qd[21:24]
        self.Pd_rbs_rocker_3 = qd[24:28]
        self.Rd_rbs_link_1 = qd[28:31]
        self.Pd_rbs_link_1 = qd[31:35]
        self.Rd_rbs_link_2 = qd[35:38]
        self.Pd_rbs_link_2 = qd[38:42]
        self.Rd_rbs_link_3 = qd[42:45]
        self.Pd_rbs_link_3 = qd[45:49]
        self.Rd_rbs_strut_lower = qd[49:52]
        self.Pd_rbs_strut_lower = qd[52:56]
        self.Rd_rbs_strut_upper = qd[56:59]
        self.Pd_rbs_strut_upper = qd[59:63]
        self.Rd_rbs_table = qd[63:66]
        self.Pd_rbs_table = qd[66:70]

    
    def _map_gen_accelerations(self):
        qdd = self._qdd
        self.Rdd_ground = qdd[0:3]
        self.Pdd_ground = qdd[3:7]
        self.Rdd_rbs_rocker_1 = qdd[7:10]
        self.Pdd_rbs_rocker_1 = qdd[10:14]
        self.Rdd_rbs_rocker_2 = qdd[14:17]
        self.Pdd_rbs_rocker_2 = qdd[17:21]
        self.Rdd_rbs_rocker_3 = qdd[21:24]
        self.Pdd_rbs_rocker_3 = qdd[24:28]
        self.Rdd_rbs_link_1 = qdd[28:31]
        self.Pdd_rbs_link_1 = qdd[31:35]
        self.Rdd_rbs_link_2 = qdd[35:38]
        self.Pdd_rbs_link_2 = qdd[38:42]
        self.Rdd_rbs_link_3 = qdd[42:45]
        self.Pdd_rbs_link_3 = qdd[45:49]
        self.Rdd_rbs_strut_lower = qdd[49:52]
        self.Pdd_rbs_strut_lower = qdd[52:56]
        self.Rdd_rbs_strut_upper = qdd[56:59]
        self.Pdd_rbs_strut_upper = qdd[59:63]
        self.Rdd_rbs_table = qdd[63:66]
        self.Pdd_rbs_table = qdd[66:70]

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcs_rev_1 = Lambda[0:5]
        self.L_mcs_act_1 = Lambda[5:6]
        self.L_jcs_rev_2 = Lambda[6:11]
        self.L_mcs_act_2 = Lambda[11:12]
        self.L_jcs_rev_3 = Lambda[12:17]
        self.L_mcs_act_3 = Lambda[17:18]
        self.L_jcs_bottom_rev_1 = Lambda[18:23]
        self.L_jcs_bottom_rev_2 = Lambda[23:28]
        self.L_jcs_bottom_rev_3 = Lambda[28:33]
        self.L_jcs_upper_sph_1 = Lambda[33:36]
        self.L_jcs_upper_sph_2 = Lambda[36:39]
        self.L_jcs_upper_sph_3 = Lambda[39:42]
        self.L_jcs_strut_lower = Lambda[42:46]
        self.L_jcs_strut_cyl = Lambda[46:50]
        self.L_jcs_strut_upper = Lambda[50:54]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_ground
        x1 = self.R_rbs_rocker_1
        x2 = self.P_ground
        x3 = A(x2)
        x4 = self.P_rbs_rocker_1
        x5 = A(x4)
        x6 = x3.T
        x7 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]
        x8 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,0:1]
        x9 = self.R_rbs_rocker_2
        x10 = self.P_rbs_rocker_2
        x11 = A(x10)
        x12 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]
        x13 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,0:1]
        x14 = self.R_rbs_rocker_3
        x15 = self.P_rbs_rocker_3
        x16 = A(x15)
        x17 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]
        x18 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,0:1]
        x19 = self.R_rbs_link_1
        x20 = self.P_rbs_link_1
        x21 = A(x20)
        x22 = x5.T
        x23 = self.Mbar_rbs_link_1_jcs_bottom_rev_1[:,2:3]
        x24 = self.R_rbs_link_2
        x25 = self.P_rbs_link_2
        x26 = A(x25)
        x27 = x11.T
        x28 = self.Mbar_rbs_link_2_jcs_bottom_rev_2[:,2:3]
        x29 = self.R_rbs_link_3
        x30 = self.P_rbs_link_3
        x31 = A(x30)
        x32 = x16.T
        x33 = self.Mbar_rbs_link_3_jcs_bottom_rev_3[:,2:3]
        x34 = (-1) * self.R_rbs_table
        x35 = self.P_rbs_table
        x36 = A(x35)
        x37 = self.R_rbs_strut_lower
        x38 = self.P_rbs_strut_lower
        x39 = A(x38)
        x40 = x39.T
        x41 = self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,0:1].T
        x42 = self.P_rbs_strut_upper
        x43 = A(x42)
        x44 = self.Mbar_rbs_strut_upper_jcs_strut_cyl[:,2:3]
        x45 = self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,1:2].T
        x46 = self.R_rbs_strut_upper
        x47 = (x37 + (-1) * x46 + multi_dot([x39,self.ubar_rbs_strut_lower_jcs_strut_cyl]) + (-1) * multi_dot([x43,self.ubar_rbs_strut_upper_jcs_strut_cyl]))
        x48 = (-1) * I1

        self.pos_eq_blocks = ((x0 + (-1) * x1 + multi_dot([x3,self.ubar_ground_jcs_rev_1]) + (-1) * multi_dot([x5,self.ubar_rbs_rocker_1_jcs_rev_1])),
        multi_dot([self.Mbar_ground_jcs_rev_1[:,0:1].T,x6,x5,x7]),
        multi_dot([self.Mbar_ground_jcs_rev_1[:,1:2].T,x6,x5,x7]),
        (cos(config.UF_mcs_act_1(t)) * multi_dot([self.Mbar_ground_jcs_rev_1[:,1:2].T,x6,x5,x8]) + (-1 * sin(config.UF_mcs_act_1(t))) * multi_dot([self.Mbar_ground_jcs_rev_1[:,0:1].T,x6,x5,x8])),
        (x0 + (-1) * x9 + multi_dot([x3,self.ubar_ground_jcs_rev_2]) + (-1) * multi_dot([x11,self.ubar_rbs_rocker_2_jcs_rev_2])),
        multi_dot([self.Mbar_ground_jcs_rev_2[:,0:1].T,x6,x11,x12]),
        multi_dot([self.Mbar_ground_jcs_rev_2[:,1:2].T,x6,x11,x12]),
        (cos(config.UF_mcs_act_2(t)) * multi_dot([self.Mbar_ground_jcs_rev_2[:,1:2].T,x6,x11,x13]) + (-1 * sin(config.UF_mcs_act_2(t))) * multi_dot([self.Mbar_ground_jcs_rev_2[:,0:1].T,x6,x11,x13])),
        (x0 + (-1) * x14 + multi_dot([x3,self.ubar_ground_jcs_rev_3]) + (-1) * multi_dot([x16,self.ubar_rbs_rocker_3_jcs_rev_3])),
        multi_dot([self.Mbar_ground_jcs_rev_3[:,0:1].T,x6,x16,x17]),
        multi_dot([self.Mbar_ground_jcs_rev_3[:,1:2].T,x6,x16,x17]),
        (cos(config.UF_mcs_act_3(t)) * multi_dot([self.Mbar_ground_jcs_rev_3[:,1:2].T,x6,x16,x18]) + (-1 * sin(config.UF_mcs_act_3(t))) * multi_dot([self.Mbar_ground_jcs_rev_3[:,0:1].T,x6,x16,x18])),
        (x1 + (-1) * x19 + multi_dot([x5,self.ubar_rbs_rocker_1_jcs_bottom_rev_1]) + (-1) * multi_dot([x21,self.ubar_rbs_link_1_jcs_bottom_rev_1])),
        multi_dot([self.Mbar_rbs_rocker_1_jcs_bottom_rev_1[:,0:1].T,x22,x21,x23]),
        multi_dot([self.Mbar_rbs_rocker_1_jcs_bottom_rev_1[:,1:2].T,x22,x21,x23]),
        (x9 + (-1) * x24 + multi_dot([x11,self.ubar_rbs_rocker_2_jcs_bottom_rev_2]) + (-1) * multi_dot([x26,self.ubar_rbs_link_2_jcs_bottom_rev_2])),
        multi_dot([self.Mbar_rbs_rocker_2_jcs_bottom_rev_2[:,0:1].T,x27,x26,x28]),
        multi_dot([self.Mbar_rbs_rocker_2_jcs_bottom_rev_2[:,1:2].T,x27,x26,x28]),
        (x14 + (-1) * x29 + multi_dot([x16,self.ubar_rbs_rocker_3_jcs_bottom_rev_3]) + (-1) * multi_dot([x31,self.ubar_rbs_link_3_jcs_bottom_rev_3])),
        multi_dot([self.Mbar_rbs_rocker_3_jcs_bottom_rev_3[:,0:1].T,x32,x31,x33]),
        multi_dot([self.Mbar_rbs_rocker_3_jcs_bottom_rev_3[:,1:2].T,x32,x31,x33]),
        (x19 + x34 + multi_dot([x21,self.ubar_rbs_link_1_jcs_upper_sph_1]) + (-1) * multi_dot([x36,self.ubar_rbs_table_jcs_upper_sph_1])),
        (x24 + x34 + multi_dot([x26,self.ubar_rbs_link_2_jcs_upper_sph_2]) + (-1) * multi_dot([x36,self.ubar_rbs_table_jcs_upper_sph_2])),
        (x29 + x34 + multi_dot([x31,self.ubar_rbs_link_3_jcs_upper_sph_3]) + (-1) * multi_dot([x36,self.ubar_rbs_table_jcs_upper_sph_3])),
        (x37 + (-1) * x0 + multi_dot([x39,self.ubar_rbs_strut_lower_jcs_strut_lower]) + (-1) * multi_dot([x3,self.ubar_ground_jcs_strut_lower])),
        multi_dot([self.Mbar_rbs_strut_lower_jcs_strut_lower[:,0:1].T,x40,x3,self.Mbar_ground_jcs_strut_lower[:,0:1]]),
        multi_dot([x41,x40,x43,x44]),
        multi_dot([x45,x40,x43,x44]),
        multi_dot([x41,x40,x47]),
        multi_dot([x45,x40,x47]),
        (x46 + x34 + multi_dot([x43,self.ubar_rbs_strut_upper_jcs_strut_upper]) + (-1) * multi_dot([x36,self.ubar_rbs_table_jcs_strut_upper])),
        multi_dot([self.Mbar_rbs_strut_upper_jcs_strut_upper[:,0:1].T,x43.T,x36,self.Mbar_rbs_table_jcs_strut_upper[:,0:1]]),
        x0,
        (x2 + (-1) * self.Pg_ground),
        (x48 + multi_dot([x4.T,x4])),
        (x48 + multi_dot([x10.T,x10])),
        (x48 + multi_dot([x15.T,x15])),
        (x48 + multi_dot([x20.T,x20])),
        (x48 + multi_dot([x25.T,x25])),
        (x48 + multi_dot([x30.T,x30])),
        (x48 + multi_dot([x38.T,x38])),
        (x48 + multi_dot([x42.T,x42])),
        (x48 + multi_dot([x35.T,x35])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z3x1
        v1 = Z1x1
        v2 = I1

        self.vel_eq_blocks = (v0,
        v1,
        v1,
        (v1 + (-1 * derivative(config.UF_mcs_act_1, t, 0.1, 1)) * v2),
        v0,
        v1,
        v1,
        (v1 + (-1 * derivative(config.UF_mcs_act_2, t, 0.1, 1)) * v2),
        v0,
        v1,
        v1,
        (v1 + (-1 * derivative(config.UF_mcs_act_3, t, 0.1, 1)) * v2),
        v0,
        v1,
        v1,
        v0,
        v1,
        v1,
        v0,
        v1,
        v1,
        v0,
        v0,
        v0,
        v0,
        v1,
        v1,
        v1,
        v1,
        v1,
        v0,
        v1,
        v0,
        Z4x1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_ground
        a1 = self.Pd_rbs_rocker_1
        a2 = self.Mbar_ground_jcs_rev_1[:,0:1]
        a3 = self.P_ground
        a4 = A(a3).T
        a5 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_rbs_rocker_1
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_ground_jcs_rev_1[:,1:2]
        a13 = I1
        a14 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,0:1]
        a15 = self.Mbar_ground_jcs_rev_1[:,1:2]
        a16 = self.Mbar_ground_jcs_rev_1[:,0:1]
        a17 = self.Pd_rbs_rocker_2
        a18 = self.Mbar_ground_jcs_rev_2[:,0:1]
        a19 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]
        a20 = B(a17,a19)
        a21 = a19.T
        a22 = self.P_rbs_rocker_2
        a23 = A(a22).T
        a24 = B(a22,a19)
        a25 = self.Mbar_ground_jcs_rev_2[:,1:2]
        a26 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,0:1]
        a27 = self.Mbar_ground_jcs_rev_2[:,1:2]
        a28 = self.Mbar_ground_jcs_rev_2[:,0:1]
        a29 = self.Pd_rbs_rocker_3
        a30 = self.Mbar_ground_jcs_rev_3[:,0:1]
        a31 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]
        a32 = B(a29,a31)
        a33 = a31.T
        a34 = self.P_rbs_rocker_3
        a35 = A(a34).T
        a36 = B(a34,a31)
        a37 = self.Mbar_ground_jcs_rev_3[:,1:2]
        a38 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,0:1]
        a39 = self.Mbar_ground_jcs_rev_3[:,1:2]
        a40 = self.Mbar_ground_jcs_rev_3[:,0:1]
        a41 = self.Pd_rbs_link_1
        a42 = self.Mbar_rbs_rocker_1_jcs_bottom_rev_1[:,0:1]
        a43 = self.Mbar_rbs_link_1_jcs_bottom_rev_1[:,2:3]
        a44 = B(a41,a43)
        a45 = a43.T
        a46 = self.P_rbs_link_1
        a47 = A(a46).T
        a48 = a1.T
        a49 = B(a46,a43)
        a50 = self.Mbar_rbs_rocker_1_jcs_bottom_rev_1[:,1:2]
        a51 = self.Pd_rbs_link_2
        a52 = self.Mbar_rbs_rocker_2_jcs_bottom_rev_2[:,0:1]
        a53 = self.Mbar_rbs_link_2_jcs_bottom_rev_2[:,2:3]
        a54 = B(a51,a53)
        a55 = a53.T
        a56 = self.P_rbs_link_2
        a57 = A(a56).T
        a58 = a17.T
        a59 = B(a56,a53)
        a60 = self.Mbar_rbs_rocker_2_jcs_bottom_rev_2[:,1:2]
        a61 = self.Pd_rbs_link_3
        a62 = self.Mbar_rbs_link_3_jcs_bottom_rev_3[:,2:3]
        a63 = a62.T
        a64 = self.P_rbs_link_3
        a65 = A(a64).T
        a66 = self.Mbar_rbs_rocker_3_jcs_bottom_rev_3[:,0:1]
        a67 = B(a61,a62)
        a68 = a29.T
        a69 = B(a64,a62)
        a70 = self.Mbar_rbs_rocker_3_jcs_bottom_rev_3[:,1:2]
        a71 = self.Pd_rbs_table
        a72 = self.Pd_rbs_strut_lower
        a73 = self.Mbar_ground_jcs_strut_lower[:,0:1]
        a74 = self.Mbar_rbs_strut_lower_jcs_strut_lower[:,0:1]
        a75 = self.P_rbs_strut_lower
        a76 = A(a75).T
        a77 = a72.T
        a78 = self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,0:1]
        a79 = a78.T
        a80 = self.Pd_rbs_strut_upper
        a81 = self.Mbar_rbs_strut_upper_jcs_strut_cyl[:,2:3]
        a82 = B(a80,a81)
        a83 = a81.T
        a84 = self.P_rbs_strut_upper
        a85 = A(a84).T
        a86 = B(a72,a78)
        a87 = B(a75,a78).T
        a88 = B(a84,a81)
        a89 = self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,1:2]
        a90 = a89.T
        a91 = B(a72,a89)
        a92 = B(a75,a89).T
        a93 = self.ubar_rbs_strut_lower_jcs_strut_cyl
        a94 = self.ubar_rbs_strut_upper_jcs_strut_cyl
        a95 = (multi_dot([B(a72,a93),a72]) + (-1) * multi_dot([B(a80,a94),a80]))
        a96 = (self.Rd_rbs_strut_lower + (-1) * self.Rd_rbs_strut_upper + multi_dot([B(a75,a93),a72]) + (-1) * multi_dot([B(a84,a94),a80]))
        a97 = (self.R_rbs_strut_lower.T + (-1) * self.R_rbs_strut_upper.T + multi_dot([a93.T,a76]) + (-1) * multi_dot([a94.T,a85]))
        a98 = self.Mbar_rbs_strut_upper_jcs_strut_upper[:,0:1]
        a99 = self.Mbar_rbs_table_jcs_strut_upper[:,0:1]
        a100 = self.P_rbs_table
        a101 = a80.T

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_ground_jcs_rev_1),a0]) + (-1) * multi_dot([B(a1,self.ubar_rbs_rocker_1_jcs_rev_1),a1])),
        (multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + (2) * multi_dot([a10,B(a3,a2).T,a11,a1])),
        (multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + (2) * multi_dot([a10,B(a3,a12).T,a11,a1])),
        ((-1 * derivative(config.UF_mcs_act_1, t, 0.1, 2)) * a13 + multi_dot([a14.T,a9,(cos(config.UF_mcs_act_1(t)) * B(a0,a15) + (-1 * sin(config.UF_mcs_act_1(t))) * B(a0,a16)),a0]) + multi_dot([(cos(config.UF_mcs_act_1(t)) * multi_dot([a15.T,a4]) + (-1 * sin(config.UF_mcs_act_1(t))) * multi_dot([a16.T,a4])),B(a1,a14),a1]) + (2) * multi_dot([(cos(config.UF_mcs_act_1(t)) * multi_dot([a10,B(a3,a15).T]) + (-1 * sin(config.UF_mcs_act_1(t))) * multi_dot([a10,B(a3,a16).T])),B(a8,a14),a1])),
        (multi_dot([B(a0,self.ubar_ground_jcs_rev_2),a0]) + (-1) * multi_dot([B(a17,self.ubar_rbs_rocker_2_jcs_rev_2),a17])),
        (multi_dot([a18.T,a4,a20,a17]) + multi_dot([a21,a23,B(a0,a18),a0]) + (2) * multi_dot([a10,B(a3,a18).T,a24,a17])),
        (multi_dot([a25.T,a4,a20,a17]) + multi_dot([a21,a23,B(a0,a25),a0]) + (2) * multi_dot([a10,B(a3,a25).T,a24,a17])),
        ((-1 * derivative(config.UF_mcs_act_2, t, 0.1, 2)) * a13 + multi_dot([a26.T,a23,(cos(config.UF_mcs_act_2(t)) * B(a0,a27) + (-1 * sin(config.UF_mcs_act_2(t))) * B(a0,a28)),a0]) + multi_dot([(cos(config.UF_mcs_act_2(t)) * multi_dot([a27.T,a4]) + (-1 * sin(config.UF_mcs_act_2(t))) * multi_dot([a28.T,a4])),B(a17,a26),a17]) + (2) * multi_dot([(cos(config.UF_mcs_act_2(t)) * multi_dot([a10,B(a3,a27).T]) + (-1 * sin(config.UF_mcs_act_2(t))) * multi_dot([a10,B(a3,a28).T])),B(a22,a26),a17])),
        (multi_dot([B(a0,self.ubar_ground_jcs_rev_3),a0]) + (-1) * multi_dot([B(a29,self.ubar_rbs_rocker_3_jcs_rev_3),a29])),
        (multi_dot([a30.T,a4,a32,a29]) + multi_dot([a33,a35,B(a0,a30),a0]) + (2) * multi_dot([a10,B(a3,a30).T,a36,a29])),
        (multi_dot([a37.T,a4,a32,a29]) + multi_dot([a33,a35,B(a0,a37),a0]) + (2) * multi_dot([a10,B(a3,a37).T,a36,a29])),
        ((-1 * derivative(config.UF_mcs_act_3, t, 0.1, 2)) * a13 + multi_dot([a38.T,a35,(cos(config.UF_mcs_act_3(t)) * B(a0,a39) + (-1 * sin(config.UF_mcs_act_3(t))) * B(a0,a40)),a0]) + multi_dot([(cos(config.UF_mcs_act_3(t)) * multi_dot([a39.T,a4]) + (-1 * sin(config.UF_mcs_act_3(t))) * multi_dot([a40.T,a4])),B(a29,a38),a29]) + (2) * multi_dot([(cos(config.UF_mcs_act_3(t)) * multi_dot([a10,B(a3,a39).T]) + (-1 * sin(config.UF_mcs_act_3(t))) * multi_dot([a10,B(a3,a40).T])),B(a34,a38),a29])),
        (multi_dot([B(a1,self.ubar_rbs_rocker_1_jcs_bottom_rev_1),a1]) + (-1) * multi_dot([B(a41,self.ubar_rbs_link_1_jcs_bottom_rev_1),a41])),
        (multi_dot([a42.T,a9,a44,a41]) + multi_dot([a45,a47,B(a1,a42),a1]) + (2) * multi_dot([a48,B(a8,a42).T,a49,a41])),
        (multi_dot([a50.T,a9,a44,a41]) + multi_dot([a45,a47,B(a1,a50),a1]) + (2) * multi_dot([a48,B(a8,a50).T,a49,a41])),
        (multi_dot([B(a17,self.ubar_rbs_rocker_2_jcs_bottom_rev_2),a17]) + (-1) * multi_dot([B(a51,self.ubar_rbs_link_2_jcs_bottom_rev_2),a51])),
        (multi_dot([a52.T,a23,a54,a51]) + multi_dot([a55,a57,B(a17,a52),a17]) + (2) * multi_dot([a58,B(a22,a52).T,a59,a51])),
        (multi_dot([a60.T,a23,a54,a51]) + multi_dot([a55,a57,B(a17,a60),a17]) + (2) * multi_dot([a58,B(a22,a60).T,a59,a51])),
        (multi_dot([B(a29,self.ubar_rbs_rocker_3_jcs_bottom_rev_3),a29]) + (-1) * multi_dot([B(a61,self.ubar_rbs_link_3_jcs_bottom_rev_3),a61])),
        (multi_dot([a63,a65,B(a29,a66),a29]) + multi_dot([a66.T,a35,a67,a61]) + (2) * multi_dot([a68,B(a34,a66).T,a69,a61])),
        (multi_dot([a63,a65,B(a29,a70),a29]) + multi_dot([a70.T,a35,a67,a61]) + (2) * multi_dot([a68,B(a34,a70).T,a69,a61])),
        (multi_dot([B(a41,self.ubar_rbs_link_1_jcs_upper_sph_1),a41]) + (-1) * multi_dot([B(a71,self.ubar_rbs_table_jcs_upper_sph_1),a71])),
        (multi_dot([B(a51,self.ubar_rbs_link_2_jcs_upper_sph_2),a51]) + (-1) * multi_dot([B(a71,self.ubar_rbs_table_jcs_upper_sph_2),a71])),
        (multi_dot([B(a61,self.ubar_rbs_link_3_jcs_upper_sph_3),a61]) + (-1) * multi_dot([B(a71,self.ubar_rbs_table_jcs_upper_sph_3),a71])),
        (multi_dot([B(a72,self.ubar_rbs_strut_lower_jcs_strut_lower),a72]) + (-1) * multi_dot([B(a0,self.ubar_ground_jcs_strut_lower),a0])),
        (multi_dot([a73.T,a4,B(a72,a74),a72]) + multi_dot([a74.T,a76,B(a0,a73),a0]) + (2) * multi_dot([a77,B(a75,a74).T,B(a3,a73),a0])),
        (multi_dot([a79,a76,a82,a80]) + multi_dot([a83,a85,a86,a72]) + (2) * multi_dot([a77,a87,a88,a80])),
        (multi_dot([a90,a76,a82,a80]) + multi_dot([a83,a85,a91,a72]) + (2) * multi_dot([a77,a92,a88,a80])),
        (multi_dot([a79,a76,a95]) + (2) * multi_dot([a77,a87,a96]) + multi_dot([a97,a86,a72])),
        (multi_dot([a90,a76,a95]) + (2) * multi_dot([a77,a92,a96]) + multi_dot([a97,a91,a72])),
        (multi_dot([B(a80,self.ubar_rbs_strut_upper_jcs_strut_upper),a80]) + (-1) * multi_dot([B(a71,self.ubar_rbs_table_jcs_strut_upper),a71])),
        (multi_dot([a98.T,a85,B(a71,a99),a71]) + multi_dot([a99.T,A(a100).T,B(a80,a98),a80]) + (2) * multi_dot([a101,B(a84,a98).T,B(a100,a99),a71])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a48,a1]),
        (2) * multi_dot([a58,a17]),
        (2) * multi_dot([a68,a29]),
        (2) * multi_dot([a41.T,a41]),
        (2) * multi_dot([a51.T,a51]),
        (2) * multi_dot([a61.T,a61]),
        (2) * multi_dot([a77,a72]),
        (2) * multi_dot([a101,a80]),
        (2) * multi_dot([a71.T,a71]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = I3
        j1 = self.P_ground
        j2 = Z1x3
        j3 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]
        j4 = j3.T
        j5 = self.P_rbs_rocker_1
        j6 = A(j5).T
        j7 = self.Mbar_ground_jcs_rev_1[:,0:1]
        j8 = self.Mbar_ground_jcs_rev_1[:,1:2]
        j9 = (-1) * j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,0:1]
        j13 = self.Mbar_ground_jcs_rev_1[:,1:2]
        j14 = self.Mbar_ground_jcs_rev_1[:,0:1]
        j15 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]
        j16 = j15.T
        j17 = self.P_rbs_rocker_2
        j18 = A(j17).T
        j19 = self.Mbar_ground_jcs_rev_2[:,0:1]
        j20 = self.Mbar_ground_jcs_rev_2[:,1:2]
        j21 = B(j17,j15)
        j22 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,0:1]
        j23 = self.Mbar_ground_jcs_rev_2[:,1:2]
        j24 = self.Mbar_ground_jcs_rev_2[:,0:1]
        j25 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]
        j26 = j25.T
        j27 = self.P_rbs_rocker_3
        j28 = A(j27).T
        j29 = self.Mbar_ground_jcs_rev_3[:,0:1]
        j30 = self.Mbar_ground_jcs_rev_3[:,1:2]
        j31 = B(j27,j25)
        j32 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,0:1]
        j33 = self.Mbar_ground_jcs_rev_3[:,1:2]
        j34 = self.Mbar_ground_jcs_rev_3[:,0:1]
        j35 = self.Mbar_rbs_link_1_jcs_bottom_rev_1[:,2:3]
        j36 = j35.T
        j37 = self.P_rbs_link_1
        j38 = A(j37).T
        j39 = self.Mbar_rbs_rocker_1_jcs_bottom_rev_1[:,0:1]
        j40 = self.Mbar_rbs_rocker_1_jcs_bottom_rev_1[:,1:2]
        j41 = B(j37,j35)
        j42 = self.Mbar_rbs_link_2_jcs_bottom_rev_2[:,2:3]
        j43 = j42.T
        j44 = self.P_rbs_link_2
        j45 = A(j44).T
        j46 = self.Mbar_rbs_rocker_2_jcs_bottom_rev_2[:,0:1]
        j47 = self.Mbar_rbs_rocker_2_jcs_bottom_rev_2[:,1:2]
        j48 = B(j44,j42)
        j49 = self.Mbar_rbs_link_3_jcs_bottom_rev_3[:,2:3]
        j50 = j49.T
        j51 = self.P_rbs_link_3
        j52 = A(j51).T
        j53 = self.Mbar_rbs_rocker_3_jcs_bottom_rev_3[:,0:1]
        j54 = self.Mbar_rbs_rocker_3_jcs_bottom_rev_3[:,1:2]
        j55 = B(j51,j49)
        j56 = self.P_rbs_table
        j57 = self.P_rbs_strut_lower
        j58 = self.Mbar_ground_jcs_strut_lower[:,0:1]
        j59 = self.Mbar_rbs_strut_lower_jcs_strut_lower[:,0:1]
        j60 = A(j57).T
        j61 = self.Mbar_rbs_strut_upper_jcs_strut_cyl[:,2:3]
        j62 = j61.T
        j63 = self.P_rbs_strut_upper
        j64 = A(j63).T
        j65 = self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,0:1]
        j66 = B(j57,j65)
        j67 = self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,1:2]
        j68 = B(j57,j67)
        j69 = j65.T
        j70 = multi_dot([j69,j60])
        j71 = self.ubar_rbs_strut_lower_jcs_strut_cyl
        j72 = B(j57,j71)
        j73 = self.ubar_rbs_strut_upper_jcs_strut_cyl
        j74 = (self.R_rbs_strut_lower.T + (-1) * self.R_rbs_strut_upper.T + multi_dot([j71.T,j60]) + (-1) * multi_dot([j73.T,j64]))
        j75 = j67.T
        j76 = multi_dot([j75,j60])
        j77 = B(j63,j61)
        j78 = B(j63,j73)
        j79 = self.Mbar_rbs_table_jcs_strut_upper[:,0:1]
        j80 = self.Mbar_rbs_strut_upper_jcs_strut_upper[:,0:1]

        self.jac_eq_blocks = (j0,
        B(j1,self.ubar_ground_jcs_rev_1),
        j9,
        (-1) * B(j5,self.ubar_rbs_rocker_1_jcs_rev_1),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j7.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j2,
        multi_dot([j8.T,j10,j11]),
        j2,
        multi_dot([j12.T,j6,(cos(config.UF_mcs_act_1(t)) * B(j1,j13) + (-1 * sin(config.UF_mcs_act_1(t))) * B(j1,j14))]),
        j2,
        multi_dot([(cos(config.UF_mcs_act_1(t)) * multi_dot([j13.T,j10]) + (-1 * sin(config.UF_mcs_act_1(t))) * multi_dot([j14.T,j10])),B(j5,j12)]),
        j0,
        B(j1,self.ubar_ground_jcs_rev_2),
        j9,
        (-1) * B(j17,self.ubar_rbs_rocker_2_jcs_rev_2),
        j2,
        multi_dot([j16,j18,B(j1,j19)]),
        j2,
        multi_dot([j19.T,j10,j21]),
        j2,
        multi_dot([j16,j18,B(j1,j20)]),
        j2,
        multi_dot([j20.T,j10,j21]),
        j2,
        multi_dot([j22.T,j18,(cos(config.UF_mcs_act_2(t)) * B(j1,j23) + (-1 * sin(config.UF_mcs_act_2(t))) * B(j1,j24))]),
        j2,
        multi_dot([(cos(config.UF_mcs_act_2(t)) * multi_dot([j23.T,j10]) + (-1 * sin(config.UF_mcs_act_2(t))) * multi_dot([j24.T,j10])),B(j17,j22)]),
        j0,
        B(j1,self.ubar_ground_jcs_rev_3),
        j9,
        (-1) * B(j27,self.ubar_rbs_rocker_3_jcs_rev_3),
        j2,
        multi_dot([j26,j28,B(j1,j29)]),
        j2,
        multi_dot([j29.T,j10,j31]),
        j2,
        multi_dot([j26,j28,B(j1,j30)]),
        j2,
        multi_dot([j30.T,j10,j31]),
        j2,
        multi_dot([j32.T,j28,(cos(config.UF_mcs_act_3(t)) * B(j1,j33) + (-1 * sin(config.UF_mcs_act_3(t))) * B(j1,j34))]),
        j2,
        multi_dot([(cos(config.UF_mcs_act_3(t)) * multi_dot([j33.T,j10]) + (-1 * sin(config.UF_mcs_act_3(t))) * multi_dot([j34.T,j10])),B(j27,j32)]),
        j0,
        B(j5,self.ubar_rbs_rocker_1_jcs_bottom_rev_1),
        j9,
        (-1) * B(j37,self.ubar_rbs_link_1_jcs_bottom_rev_1),
        j2,
        multi_dot([j36,j38,B(j5,j39)]),
        j2,
        multi_dot([j39.T,j6,j41]),
        j2,
        multi_dot([j36,j38,B(j5,j40)]),
        j2,
        multi_dot([j40.T,j6,j41]),
        j0,
        B(j17,self.ubar_rbs_rocker_2_jcs_bottom_rev_2),
        j9,
        (-1) * B(j44,self.ubar_rbs_link_2_jcs_bottom_rev_2),
        j2,
        multi_dot([j43,j45,B(j17,j46)]),
        j2,
        multi_dot([j46.T,j18,j48]),
        j2,
        multi_dot([j43,j45,B(j17,j47)]),
        j2,
        multi_dot([j47.T,j18,j48]),
        j0,
        B(j27,self.ubar_rbs_rocker_3_jcs_bottom_rev_3),
        j9,
        (-1) * B(j51,self.ubar_rbs_link_3_jcs_bottom_rev_3),
        j2,
        multi_dot([j50,j52,B(j27,j53)]),
        j2,
        multi_dot([j53.T,j28,j55]),
        j2,
        multi_dot([j50,j52,B(j27,j54)]),
        j2,
        multi_dot([j54.T,j28,j55]),
        j0,
        B(j37,self.ubar_rbs_link_1_jcs_upper_sph_1),
        j9,
        (-1) * B(j56,self.ubar_rbs_table_jcs_upper_sph_1),
        j0,
        B(j44,self.ubar_rbs_link_2_jcs_upper_sph_2),
        j9,
        (-1) * B(j56,self.ubar_rbs_table_jcs_upper_sph_2),
        j0,
        B(j51,self.ubar_rbs_link_3_jcs_upper_sph_3),
        j9,
        (-1) * B(j56,self.ubar_rbs_table_jcs_upper_sph_3),
        j9,
        (-1) * B(j1,self.ubar_ground_jcs_strut_lower),
        j0,
        B(j57,self.ubar_rbs_strut_lower_jcs_strut_lower),
        j2,
        multi_dot([j59.T,j60,B(j1,j58)]),
        j2,
        multi_dot([j58.T,j10,B(j57,j59)]),
        j2,
        multi_dot([j62,j64,j66]),
        j2,
        multi_dot([j69,j60,j77]),
        j2,
        multi_dot([j62,j64,j68]),
        j2,
        multi_dot([j75,j60,j77]),
        j70,
        (multi_dot([j69,j60,j72]) + multi_dot([j74,j66])),
        (-1) * j70,
        (-1) * multi_dot([j69,j60,j78]),
        j76,
        (multi_dot([j75,j60,j72]) + multi_dot([j74,j68])),
        (-1) * j76,
        (-1) * multi_dot([j75,j60,j78]),
        j0,
        B(j63,self.ubar_rbs_strut_upper_jcs_strut_upper),
        j9,
        (-1) * B(j56,self.ubar_rbs_table_jcs_strut_upper),
        j2,
        multi_dot([j79.T,A(j56).T,B(j63,j80)]),
        j2,
        multi_dot([j80.T,j64,B(j56,j79)]),
        j0,
        Z3x4,
        Z4x3,
        I4,
        j2,
        (2) * j5.T,
        j2,
        (2) * j17.T,
        j2,
        (2) * j27.T,
        j2,
        (2) * j37.T,
        j2,
        (2) * j44.T,
        j2,
        (2) * j51.T,
        j2,
        (2) * j57.T,
        j2,
        (2) * j63.T,
        j2,
        (2) * j56.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_ground)
        m2 = G(self.P_rbs_rocker_1)
        m3 = G(self.P_rbs_rocker_2)
        m4 = G(self.P_rbs_rocker_3)
        m5 = G(self.P_rbs_link_1)
        m6 = G(self.P_rbs_link_2)
        m7 = G(self.P_rbs_link_3)
        m8 = G(self.P_rbs_strut_lower)
        m9 = G(self.P_rbs_strut_upper)
        m10 = G(self.P_rbs_table)

        self.mass_eq_blocks = (self.m_ground * m0,
        (4) * multi_dot([m1.T,self.Jbar_ground,m1]),
        config.m_rbs_rocker_1 * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbs_rocker_1,m2]),
        config.m_rbs_rocker_2 * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbs_rocker_2,m3]),
        config.m_rbs_rocker_3 * m0,
        (4) * multi_dot([m4.T,config.Jbar_rbs_rocker_3,m4]),
        config.m_rbs_link_1 * m0,
        (4) * multi_dot([m5.T,config.Jbar_rbs_link_1,m5]),
        config.m_rbs_link_2 * m0,
        (4) * multi_dot([m6.T,config.Jbar_rbs_link_2,m6]),
        config.m_rbs_link_3 * m0,
        (4) * multi_dot([m7.T,config.Jbar_rbs_link_3,m7]),
        config.m_rbs_strut_lower * m0,
        (4) * multi_dot([m8.T,config.Jbar_rbs_strut_lower,m8]),
        config.m_rbs_strut_upper * m0,
        (4) * multi_dot([m9.T,config.Jbar_rbs_strut_upper,m9]),
        config.m_rbs_table * m0,
        (4) * multi_dot([m10.T,config.Jbar_rbs_table,m10]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = Z3x1
        f1 = Z4x1
        f2 = G(self.Pd_rbs_rocker_1)
        f3 = G(self.Pd_rbs_rocker_2)
        f4 = G(self.Pd_rbs_rocker_3)
        f5 = G(self.Pd_rbs_link_1)
        f6 = G(self.Pd_rbs_link_2)
        f7 = G(self.Pd_rbs_link_3)
        f8 = self.R_rbs_strut_upper
        f9 = self.R_rbs_strut_lower
        f10 = self.ubar_rbs_strut_upper_fas_strut
        f11 = self.P_rbs_strut_upper
        f12 = A(f11)
        f13 = self.ubar_rbs_strut_lower_fas_strut
        f14 = self.P_rbs_strut_lower
        f15 = A(f14)
        f16 = (f8.T + (-1) * f9.T + multi_dot([f10.T,f12.T]) + (-1) * multi_dot([f13.T,f15.T]))
        f17 = multi_dot([f12,f10])
        f18 = multi_dot([f15,f13])
        f19 = (f8 + (-1) * f9 + f17 + (-1) * f18)
        f20 = ((multi_dot([f16,f19]))**(1.0/2.0))[0]
        f21 = 1.0/f20
        f22 = config.UF_fas_strut_Fs((config.fas_strut_FL + (-1 * f20)))
        f23 = self.Pd_rbs_strut_upper
        f24 = self.Pd_rbs_strut_lower
        f25 = config.UF_fas_strut_Fd((-1 * 1.0/f20) * multi_dot([f16,(self.Rd_rbs_strut_upper + (-1) * self.Rd_rbs_strut_lower + multi_dot([B(f11,f10),f23]) + (-1) * multi_dot([B(f14,f13),f24]))]))
        f26 = (f21 * (f22 + f25)) * f19
        f27 = G(f24)
        f28 = (2 * f22)
        f29 = (2 * f25)
        f30 = G(f23)
        f31 = G(self.Pd_rbs_table)

        self.frc_eq_blocks = (f0,
        f1,
        self.F_rbs_rocker_1_gravity,
        (8) * multi_dot([f2.T,config.Jbar_rbs_rocker_1,f2,self.P_rbs_rocker_1]),
        self.F_rbs_rocker_2_gravity,
        (8) * multi_dot([f3.T,config.Jbar_rbs_rocker_2,f3,self.P_rbs_rocker_2]),
        self.F_rbs_rocker_3_gravity,
        (8) * multi_dot([f4.T,config.Jbar_rbs_rocker_3,f4,self.P_rbs_rocker_3]),
        self.F_rbs_link_1_gravity,
        (8) * multi_dot([f5.T,config.Jbar_rbs_link_1,f5,self.P_rbs_link_1]),
        self.F_rbs_link_2_gravity,
        (8) * multi_dot([f6.T,config.Jbar_rbs_link_2,f6,self.P_rbs_link_2]),
        self.F_rbs_link_3_gravity,
        (8) * multi_dot([f7.T,config.Jbar_rbs_link_3,f7,self.P_rbs_link_3]),
        (self.F_rbs_strut_lower_gravity + f0 + (-1) * f26),
        (f1 + (8) * multi_dot([f27.T,config.Jbar_rbs_strut_lower,f27,f14]) + (f21 * (f28 + f29)) * multi_dot([E(f14).T,skew(f18).T,f19])),
        (self.F_rbs_strut_upper_gravity + f26),
        ((8) * multi_dot([f30.T,config.Jbar_rbs_strut_upper,f30,f11]) + (f21 * ((-1 * f28) + (-1 * f29))) * multi_dot([E(f11).T,skew(f17).T,f19])),
        self.F_rbs_table_gravity,
        (8) * multi_dot([f31.T,config.Jbar_rbs_table,f31,self.P_rbs_table]),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_ground_jcs_rev_1 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_ground,self.ubar_ground_jcs_rev_1).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_rev_1[:,0:1]).T,A(self.P_rbs_rocker_1),self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_rev_1[:,1:2]).T,A(self.P_rbs_rocker_1),self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]])]]),self.L_jcs_rev_1])
        self.F_ground_jcs_rev_1 = Q_ground_jcs_rev_1[0:3]
        Te_ground_jcs_rev_1 = Q_ground_jcs_rev_1[3:7]
        self.T_ground_jcs_rev_1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_rev_1])),self.F_ground_jcs_rev_1]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_rev_1]))
        Q_ground_mcs_act_1 = (-1) * multi_dot([np.bmat([[Z1x3.T],[multi_dot([((-1 * sin(config.UF_mcs_act_1(t))) * B(self.P_ground,self.Mbar_ground_jcs_rev_1[:,0:1]).T + cos(config.UF_mcs_act_1(t)) * B(self.P_ground,self.Mbar_ground_jcs_rev_1[:,1:2]).T),A(self.P_rbs_rocker_1),self.Mbar_rbs_rocker_1_jcs_rev_1[:,0:1]])]]),self.L_mcs_act_1])
        self.F_ground_mcs_act_1 = Q_ground_mcs_act_1[0:3]
        Te_ground_mcs_act_1 = Q_ground_mcs_act_1[3:7]
        self.T_ground_mcs_act_1 = (0.5) * multi_dot([E(self.P_ground),Te_ground_mcs_act_1])
        Q_ground_jcs_rev_2 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_ground,self.ubar_ground_jcs_rev_2).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_rev_2[:,0:1]).T,A(self.P_rbs_rocker_2),self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_rev_2[:,1:2]).T,A(self.P_rbs_rocker_2),self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]])]]),self.L_jcs_rev_2])
        self.F_ground_jcs_rev_2 = Q_ground_jcs_rev_2[0:3]
        Te_ground_jcs_rev_2 = Q_ground_jcs_rev_2[3:7]
        self.T_ground_jcs_rev_2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_rev_2])),self.F_ground_jcs_rev_2]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_rev_2]))
        Q_ground_mcs_act_2 = (-1) * multi_dot([np.bmat([[Z1x3.T],[multi_dot([((-1 * sin(config.UF_mcs_act_2(t))) * B(self.P_ground,self.Mbar_ground_jcs_rev_2[:,0:1]).T + cos(config.UF_mcs_act_2(t)) * B(self.P_ground,self.Mbar_ground_jcs_rev_2[:,1:2]).T),A(self.P_rbs_rocker_2),self.Mbar_rbs_rocker_2_jcs_rev_2[:,0:1]])]]),self.L_mcs_act_2])
        self.F_ground_mcs_act_2 = Q_ground_mcs_act_2[0:3]
        Te_ground_mcs_act_2 = Q_ground_mcs_act_2[3:7]
        self.T_ground_mcs_act_2 = (0.5) * multi_dot([E(self.P_ground),Te_ground_mcs_act_2])
        Q_ground_jcs_rev_3 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_ground,self.ubar_ground_jcs_rev_3).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_rev_3[:,0:1]).T,A(self.P_rbs_rocker_3),self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_rev_3[:,1:2]).T,A(self.P_rbs_rocker_3),self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]])]]),self.L_jcs_rev_3])
        self.F_ground_jcs_rev_3 = Q_ground_jcs_rev_3[0:3]
        Te_ground_jcs_rev_3 = Q_ground_jcs_rev_3[3:7]
        self.T_ground_jcs_rev_3 = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_rev_3])),self.F_ground_jcs_rev_3]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_rev_3]))
        Q_ground_mcs_act_3 = (-1) * multi_dot([np.bmat([[Z1x3.T],[multi_dot([((-1 * sin(config.UF_mcs_act_3(t))) * B(self.P_ground,self.Mbar_ground_jcs_rev_3[:,0:1]).T + cos(config.UF_mcs_act_3(t)) * B(self.P_ground,self.Mbar_ground_jcs_rev_3[:,1:2]).T),A(self.P_rbs_rocker_3),self.Mbar_rbs_rocker_3_jcs_rev_3[:,0:1]])]]),self.L_mcs_act_3])
        self.F_ground_mcs_act_3 = Q_ground_mcs_act_3[0:3]
        Te_ground_mcs_act_3 = Q_ground_mcs_act_3[3:7]
        self.T_ground_mcs_act_3 = (0.5) * multi_dot([E(self.P_ground),Te_ground_mcs_act_3])
        Q_rbs_rocker_1_jcs_bottom_rev_1 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_rocker_1,self.ubar_rbs_rocker_1_jcs_bottom_rev_1).T,multi_dot([B(self.P_rbs_rocker_1,self.Mbar_rbs_rocker_1_jcs_bottom_rev_1[:,0:1]).T,A(self.P_rbs_link_1),self.Mbar_rbs_link_1_jcs_bottom_rev_1[:,2:3]]),multi_dot([B(self.P_rbs_rocker_1,self.Mbar_rbs_rocker_1_jcs_bottom_rev_1[:,1:2]).T,A(self.P_rbs_link_1),self.Mbar_rbs_link_1_jcs_bottom_rev_1[:,2:3]])]]),self.L_jcs_bottom_rev_1])
        self.F_rbs_rocker_1_jcs_bottom_rev_1 = Q_rbs_rocker_1_jcs_bottom_rev_1[0:3]
        Te_rbs_rocker_1_jcs_bottom_rev_1 = Q_rbs_rocker_1_jcs_bottom_rev_1[3:7]
        self.T_rbs_rocker_1_jcs_bottom_rev_1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_rocker_1),self.ubar_rbs_rocker_1_jcs_bottom_rev_1])),self.F_rbs_rocker_1_jcs_bottom_rev_1]) + (0.5) * multi_dot([E(self.P_rbs_rocker_1),Te_rbs_rocker_1_jcs_bottom_rev_1]))
        Q_rbs_rocker_2_jcs_bottom_rev_2 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_rocker_2,self.ubar_rbs_rocker_2_jcs_bottom_rev_2).T,multi_dot([B(self.P_rbs_rocker_2,self.Mbar_rbs_rocker_2_jcs_bottom_rev_2[:,0:1]).T,A(self.P_rbs_link_2),self.Mbar_rbs_link_2_jcs_bottom_rev_2[:,2:3]]),multi_dot([B(self.P_rbs_rocker_2,self.Mbar_rbs_rocker_2_jcs_bottom_rev_2[:,1:2]).T,A(self.P_rbs_link_2),self.Mbar_rbs_link_2_jcs_bottom_rev_2[:,2:3]])]]),self.L_jcs_bottom_rev_2])
        self.F_rbs_rocker_2_jcs_bottom_rev_2 = Q_rbs_rocker_2_jcs_bottom_rev_2[0:3]
        Te_rbs_rocker_2_jcs_bottom_rev_2 = Q_rbs_rocker_2_jcs_bottom_rev_2[3:7]
        self.T_rbs_rocker_2_jcs_bottom_rev_2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_rocker_2),self.ubar_rbs_rocker_2_jcs_bottom_rev_2])),self.F_rbs_rocker_2_jcs_bottom_rev_2]) + (0.5) * multi_dot([E(self.P_rbs_rocker_2),Te_rbs_rocker_2_jcs_bottom_rev_2]))
        Q_rbs_rocker_3_jcs_bottom_rev_3 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_rocker_3,self.ubar_rbs_rocker_3_jcs_bottom_rev_3).T,multi_dot([B(self.P_rbs_rocker_3,self.Mbar_rbs_rocker_3_jcs_bottom_rev_3[:,0:1]).T,A(self.P_rbs_link_3),self.Mbar_rbs_link_3_jcs_bottom_rev_3[:,2:3]]),multi_dot([B(self.P_rbs_rocker_3,self.Mbar_rbs_rocker_3_jcs_bottom_rev_3[:,1:2]).T,A(self.P_rbs_link_3),self.Mbar_rbs_link_3_jcs_bottom_rev_3[:,2:3]])]]),self.L_jcs_bottom_rev_3])
        self.F_rbs_rocker_3_jcs_bottom_rev_3 = Q_rbs_rocker_3_jcs_bottom_rev_3[0:3]
        Te_rbs_rocker_3_jcs_bottom_rev_3 = Q_rbs_rocker_3_jcs_bottom_rev_3[3:7]
        self.T_rbs_rocker_3_jcs_bottom_rev_3 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_rocker_3),self.ubar_rbs_rocker_3_jcs_bottom_rev_3])),self.F_rbs_rocker_3_jcs_bottom_rev_3]) + (0.5) * multi_dot([E(self.P_rbs_rocker_3),Te_rbs_rocker_3_jcs_bottom_rev_3]))
        Q_rbs_link_1_jcs_upper_sph_1 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_link_1,self.ubar_rbs_link_1_jcs_upper_sph_1).T]]),self.L_jcs_upper_sph_1])
        self.F_rbs_link_1_jcs_upper_sph_1 = Q_rbs_link_1_jcs_upper_sph_1[0:3]
        Te_rbs_link_1_jcs_upper_sph_1 = Q_rbs_link_1_jcs_upper_sph_1[3:7]
        self.T_rbs_link_1_jcs_upper_sph_1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_link_1),self.ubar_rbs_link_1_jcs_upper_sph_1])),self.F_rbs_link_1_jcs_upper_sph_1]) + (0.5) * multi_dot([E(self.P_rbs_link_1),Te_rbs_link_1_jcs_upper_sph_1]))
        Q_rbs_link_2_jcs_upper_sph_2 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_link_2,self.ubar_rbs_link_2_jcs_upper_sph_2).T]]),self.L_jcs_upper_sph_2])
        self.F_rbs_link_2_jcs_upper_sph_2 = Q_rbs_link_2_jcs_upper_sph_2[0:3]
        Te_rbs_link_2_jcs_upper_sph_2 = Q_rbs_link_2_jcs_upper_sph_2[3:7]
        self.T_rbs_link_2_jcs_upper_sph_2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_link_2),self.ubar_rbs_link_2_jcs_upper_sph_2])),self.F_rbs_link_2_jcs_upper_sph_2]) + (0.5) * multi_dot([E(self.P_rbs_link_2),Te_rbs_link_2_jcs_upper_sph_2]))
        Q_rbs_link_3_jcs_upper_sph_3 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_link_3,self.ubar_rbs_link_3_jcs_upper_sph_3).T]]),self.L_jcs_upper_sph_3])
        self.F_rbs_link_3_jcs_upper_sph_3 = Q_rbs_link_3_jcs_upper_sph_3[0:3]
        Te_rbs_link_3_jcs_upper_sph_3 = Q_rbs_link_3_jcs_upper_sph_3[3:7]
        self.T_rbs_link_3_jcs_upper_sph_3 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_link_3),self.ubar_rbs_link_3_jcs_upper_sph_3])),self.F_rbs_link_3_jcs_upper_sph_3]) + (0.5) * multi_dot([E(self.P_rbs_link_3),Te_rbs_link_3_jcs_upper_sph_3]))
        Q_rbs_strut_lower_jcs_strut_lower = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbs_strut_lower,self.ubar_rbs_strut_lower_jcs_strut_lower).T,multi_dot([B(self.P_rbs_strut_lower,self.Mbar_rbs_strut_lower_jcs_strut_lower[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_strut_lower[:,0:1]])]]),self.L_jcs_strut_lower])
        self.F_rbs_strut_lower_jcs_strut_lower = Q_rbs_strut_lower_jcs_strut_lower[0:3]
        Te_rbs_strut_lower_jcs_strut_lower = Q_rbs_strut_lower_jcs_strut_lower[3:7]
        self.T_rbs_strut_lower_jcs_strut_lower = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_strut_lower),self.ubar_rbs_strut_lower_jcs_strut_lower])),self.F_rbs_strut_lower_jcs_strut_lower]) + (0.5) * multi_dot([E(self.P_rbs_strut_lower),Te_rbs_strut_lower_jcs_strut_lower]))
        Q_rbs_strut_lower_jcs_strut_cyl = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_strut_lower),self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,0:1]]),multi_dot([A(self.P_rbs_strut_lower),self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,1:2]])],[multi_dot([B(self.P_rbs_strut_lower,self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,0:1]).T,A(self.P_rbs_strut_upper),self.Mbar_rbs_strut_upper_jcs_strut_cyl[:,2:3]]),multi_dot([B(self.P_rbs_strut_lower,self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,1:2]).T,A(self.P_rbs_strut_upper),self.Mbar_rbs_strut_upper_jcs_strut_cyl[:,2:3]]),(multi_dot([B(self.P_rbs_strut_lower,self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,0:1]).T,((-1) * self.R_rbs_strut_upper + multi_dot([A(self.P_rbs_strut_lower),self.ubar_rbs_strut_lower_jcs_strut_cyl]) + (-1) * multi_dot([A(self.P_rbs_strut_upper),self.ubar_rbs_strut_upper_jcs_strut_cyl]) + self.R_rbs_strut_lower)]) + multi_dot([B(self.P_rbs_strut_lower,self.ubar_rbs_strut_lower_jcs_strut_cyl).T,A(self.P_rbs_strut_lower),self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,0:1]])),(multi_dot([B(self.P_rbs_strut_lower,self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,1:2]).T,((-1) * self.R_rbs_strut_upper + multi_dot([A(self.P_rbs_strut_lower),self.ubar_rbs_strut_lower_jcs_strut_cyl]) + (-1) * multi_dot([A(self.P_rbs_strut_upper),self.ubar_rbs_strut_upper_jcs_strut_cyl]) + self.R_rbs_strut_lower)]) + multi_dot([B(self.P_rbs_strut_lower,self.ubar_rbs_strut_lower_jcs_strut_cyl).T,A(self.P_rbs_strut_lower),self.Mbar_rbs_strut_lower_jcs_strut_cyl[:,1:2]]))]]),self.L_jcs_strut_cyl])
        self.F_rbs_strut_lower_jcs_strut_cyl = Q_rbs_strut_lower_jcs_strut_cyl[0:3]
        Te_rbs_strut_lower_jcs_strut_cyl = Q_rbs_strut_lower_jcs_strut_cyl[3:7]
        self.T_rbs_strut_lower_jcs_strut_cyl = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_strut_lower),self.ubar_rbs_strut_lower_jcs_strut_cyl])),self.F_rbs_strut_lower_jcs_strut_cyl]) + (0.5) * multi_dot([E(self.P_rbs_strut_lower),Te_rbs_strut_lower_jcs_strut_cyl]))
        Q_rbs_strut_upper_jcs_strut_upper = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbs_strut_upper,self.ubar_rbs_strut_upper_jcs_strut_upper).T,multi_dot([B(self.P_rbs_strut_upper,self.Mbar_rbs_strut_upper_jcs_strut_upper[:,0:1]).T,A(self.P_rbs_table),self.Mbar_rbs_table_jcs_strut_upper[:,0:1]])]]),self.L_jcs_strut_upper])
        self.F_rbs_strut_upper_jcs_strut_upper = Q_rbs_strut_upper_jcs_strut_upper[0:3]
        Te_rbs_strut_upper_jcs_strut_upper = Q_rbs_strut_upper_jcs_strut_upper[3:7]
        self.T_rbs_strut_upper_jcs_strut_upper = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_strut_upper),self.ubar_rbs_strut_upper_jcs_strut_upper])),self.F_rbs_strut_upper_jcs_strut_upper]) + (0.5) * multi_dot([E(self.P_rbs_strut_upper),Te_rbs_strut_upper_jcs_strut_upper]))
        self.F_rbs_strut_upper_fas_strut = (1.0/((multi_dot([((-1) * self.R_rbs_strut_lower.T + multi_dot([self.ubar_rbs_strut_upper_fas_strut.T,A(self.P_rbs_strut_upper).T]) + (-1) * multi_dot([self.ubar_rbs_strut_lower_fas_strut.T,A(self.P_rbs_strut_lower).T]) + self.R_rbs_strut_upper.T),((-1) * self.R_rbs_strut_lower + multi_dot([A(self.P_rbs_strut_upper),self.ubar_rbs_strut_upper_fas_strut]) + (-1) * multi_dot([A(self.P_rbs_strut_lower),self.ubar_rbs_strut_lower_fas_strut]) + self.R_rbs_strut_upper)]))**(1.0/2.0))[0] * (config.UF_fas_strut_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_rbs_strut_lower.T + multi_dot([self.ubar_rbs_strut_upper_fas_strut.T,A(self.P_rbs_strut_upper).T]) + (-1) * multi_dot([self.ubar_rbs_strut_lower_fas_strut.T,A(self.P_rbs_strut_lower).T]) + self.R_rbs_strut_upper.T),((-1) * self.R_rbs_strut_lower + multi_dot([A(self.P_rbs_strut_upper),self.ubar_rbs_strut_upper_fas_strut]) + (-1) * multi_dot([A(self.P_rbs_strut_lower),self.ubar_rbs_strut_lower_fas_strut]) + self.R_rbs_strut_upper)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_rbs_strut_lower.T + multi_dot([self.ubar_rbs_strut_upper_fas_strut.T,A(self.P_rbs_strut_upper).T]) + (-1) * multi_dot([self.ubar_rbs_strut_lower_fas_strut.T,A(self.P_rbs_strut_lower).T]) + self.R_rbs_strut_upper.T),((-1) * self.Rd_rbs_strut_lower + multi_dot([B(self.P_rbs_strut_upper,self.ubar_rbs_strut_upper_fas_strut),self.Pd_rbs_strut_upper]) + (-1) * multi_dot([B(self.P_rbs_strut_lower,self.ubar_rbs_strut_lower_fas_strut),self.Pd_rbs_strut_lower]) + self.Rd_rbs_strut_upper)])) + config.UF_fas_strut_Fs((config.fas_strut_FL + (-1 * ((multi_dot([((-1) * self.R_rbs_strut_lower.T + multi_dot([self.ubar_rbs_strut_upper_fas_strut.T,A(self.P_rbs_strut_upper).T]) + (-1) * multi_dot([self.ubar_rbs_strut_lower_fas_strut.T,A(self.P_rbs_strut_lower).T]) + self.R_rbs_strut_upper.T),((-1) * self.R_rbs_strut_lower + multi_dot([A(self.P_rbs_strut_upper),self.ubar_rbs_strut_upper_fas_strut]) + (-1) * multi_dot([A(self.P_rbs_strut_lower),self.ubar_rbs_strut_lower_fas_strut]) + self.R_rbs_strut_upper)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_rbs_strut_lower + multi_dot([A(self.P_rbs_strut_upper),self.ubar_rbs_strut_upper_fas_strut]) + (-1) * multi_dot([A(self.P_rbs_strut_lower),self.ubar_rbs_strut_lower_fas_strut]) + self.R_rbs_strut_upper)
        self.T_rbs_strut_upper_fas_strut = Z3x1

        self.reactions = {'F_ground_jcs_rev_1' : self.F_ground_jcs_rev_1,
                        'T_ground_jcs_rev_1' : self.T_ground_jcs_rev_1,
                        'F_ground_mcs_act_1' : self.F_ground_mcs_act_1,
                        'T_ground_mcs_act_1' : self.T_ground_mcs_act_1,
                        'F_ground_jcs_rev_2' : self.F_ground_jcs_rev_2,
                        'T_ground_jcs_rev_2' : self.T_ground_jcs_rev_2,
                        'F_ground_mcs_act_2' : self.F_ground_mcs_act_2,
                        'T_ground_mcs_act_2' : self.T_ground_mcs_act_2,
                        'F_ground_jcs_rev_3' : self.F_ground_jcs_rev_3,
                        'T_ground_jcs_rev_3' : self.T_ground_jcs_rev_3,
                        'F_ground_mcs_act_3' : self.F_ground_mcs_act_3,
                        'T_ground_mcs_act_3' : self.T_ground_mcs_act_3,
                        'F_rbs_rocker_1_jcs_bottom_rev_1' : self.F_rbs_rocker_1_jcs_bottom_rev_1,
                        'T_rbs_rocker_1_jcs_bottom_rev_1' : self.T_rbs_rocker_1_jcs_bottom_rev_1,
                        'F_rbs_rocker_2_jcs_bottom_rev_2' : self.F_rbs_rocker_2_jcs_bottom_rev_2,
                        'T_rbs_rocker_2_jcs_bottom_rev_2' : self.T_rbs_rocker_2_jcs_bottom_rev_2,
                        'F_rbs_rocker_3_jcs_bottom_rev_3' : self.F_rbs_rocker_3_jcs_bottom_rev_3,
                        'T_rbs_rocker_3_jcs_bottom_rev_3' : self.T_rbs_rocker_3_jcs_bottom_rev_3,
                        'F_rbs_link_1_jcs_upper_sph_1' : self.F_rbs_link_1_jcs_upper_sph_1,
                        'T_rbs_link_1_jcs_upper_sph_1' : self.T_rbs_link_1_jcs_upper_sph_1,
                        'F_rbs_link_2_jcs_upper_sph_2' : self.F_rbs_link_2_jcs_upper_sph_2,
                        'T_rbs_link_2_jcs_upper_sph_2' : self.T_rbs_link_2_jcs_upper_sph_2,
                        'F_rbs_link_3_jcs_upper_sph_3' : self.F_rbs_link_3_jcs_upper_sph_3,
                        'T_rbs_link_3_jcs_upper_sph_3' : self.T_rbs_link_3_jcs_upper_sph_3,
                        'F_rbs_strut_lower_jcs_strut_lower' : self.F_rbs_strut_lower_jcs_strut_lower,
                        'T_rbs_strut_lower_jcs_strut_lower' : self.T_rbs_strut_lower_jcs_strut_lower,
                        'F_rbs_strut_lower_jcs_strut_cyl' : self.F_rbs_strut_lower_jcs_strut_cyl,
                        'T_rbs_strut_lower_jcs_strut_cyl' : self.T_rbs_strut_lower_jcs_strut_cyl,
                        'F_rbs_strut_upper_jcs_strut_upper' : self.F_rbs_strut_upper_jcs_strut_upper,
                        'T_rbs_strut_upper_jcs_strut_upper' : self.T_rbs_strut_upper_jcs_strut_upper,
                        'F_rbs_strut_upper_fas_strut' : self.F_rbs_strut_upper_fas_strut,
                        'T_rbs_strut_upper_fas_strut' : self.T_rbs_strut_upper_fas_strut}

