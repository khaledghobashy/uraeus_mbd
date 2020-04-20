
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

        self.indicies_map = {'ground': 0, 'rbs_table': 1, 'rbs_l1': 2, 'rbs_l2': 3, 'rbs_l3': 4, 'rbs_l4': 5, 'rbs_l5': 6, 'rbs_l6': 7, 'rbs_l7': 8, 'rbs_l8': 9}

        self.n  = 70
        self.nc = 70
        self.nrows = 49
        self.ncols = 2*10
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_ground_jcs_a', 'T_ground_jcs_a', 'F_ground_jcs_b', 'T_ground_jcs_b', 'F_ground_jcs_c', 'T_ground_jcs_c', 'F_rbs_table_jcs_d', 'T_rbs_table_jcs_d', 'F_rbs_table_jcs_e', 'T_rbs_table_jcs_e', 'F_rbs_table_jcs_f', 'T_rbs_table_jcs_f', 'F_rbs_l1_jcs_h1', 'T_rbs_l1_jcs_h1', 'F_rbs_l2_jcs_k1', 'T_rbs_l2_jcs_k1', 'F_rbs_l3_jcs_l1', 'T_rbs_l3_jcs_l1', 'F_rbs_l4_jcs_h2', 'T_rbs_l4_jcs_h2', 'F_rbs_l5_jcs_k2', 'T_rbs_l5_jcs_k2', 'F_rbs_l6_jcs_l2', 'T_rbs_l6_jcs_l2', 'F_rbs_l7_jcs_trans', 'T_rbs_l7_jcs_trans', 'F_rbs_l7_mcs_act', 'T_rbs_l7_mcs_act', 'F_rbs_l7_fas_strut', 'T_rbs_l7_fas_strut']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.ground*2, self.ground*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.ground*2, self.ground*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.ground*2, self.ground*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.ground*2, self.ground*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.ground*2, self.ground*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_l6*2, self.rbs_l6*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l6*2, self.rbs_l6*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l6*2, self.rbs_l6*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l6*2, self.rbs_l6*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l6*2, self.rbs_l6*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l8*2, self.rbs_l8*2+1], dtype=np.intc)

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
        self.config.R_rbs_table,
        self.config.P_rbs_table,
        self.config.R_rbs_l1,
        self.config.P_rbs_l1,
        self.config.R_rbs_l2,
        self.config.P_rbs_l2,
        self.config.R_rbs_l3,
        self.config.P_rbs_l3,
        self.config.R_rbs_l4,
        self.config.P_rbs_l4,
        self.config.R_rbs_l5,
        self.config.P_rbs_l5,
        self.config.R_rbs_l6,
        self.config.P_rbs_l6,
        self.config.R_rbs_l7,
        self.config.P_rbs_l7,
        self.config.R_rbs_l8,
        self.config.P_rbs_l8], out=self._q)

        np.concatenate([self.config.Rd_ground,
        self.config.Pd_ground,
        self.config.Rd_rbs_table,
        self.config.Pd_rbs_table,
        self.config.Rd_rbs_l1,
        self.config.Pd_rbs_l1,
        self.config.Rd_rbs_l2,
        self.config.Pd_rbs_l2,
        self.config.Rd_rbs_l3,
        self.config.Pd_rbs_l3,
        self.config.Rd_rbs_l4,
        self.config.Pd_rbs_l4,
        self.config.Rd_rbs_l5,
        self.config.Pd_rbs_l5,
        self.config.Rd_rbs_l6,
        self.config.Pd_rbs_l6,
        self.config.Rd_rbs_l7,
        self.config.Pd_rbs_l7,
        self.config.Rd_rbs_l8,
        self.config.Pd_rbs_l8], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p + 'ground']
        self.rbs_table = indicies_map[p + 'rbs_table']
        self.rbs_l1 = indicies_map[p + 'rbs_l1']
        self.rbs_l2 = indicies_map[p + 'rbs_l2']
        self.rbs_l3 = indicies_map[p + 'rbs_l3']
        self.rbs_l4 = indicies_map[p + 'rbs_l4']
        self.rbs_l5 = indicies_map[p + 'rbs_l5']
        self.rbs_l6 = indicies_map[p + 'rbs_l6']
        self.rbs_l7 = indicies_map[p + 'rbs_l7']
        self.rbs_l8 = indicies_map[p + 'rbs_l8']
    

    
    def eval_constants(self):
        config = self.config

        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.m_ground = 1.0
        self.Jbar_ground = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.F_rbs_table_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_table]], dtype=np.float64)
        self.F_rbs_l1_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l1]], dtype=np.float64)
        self.F_rbs_l2_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l2]], dtype=np.float64)
        self.F_rbs_l3_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l3]], dtype=np.float64)
        self.F_rbs_l4_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l4]], dtype=np.float64)
        self.F_rbs_l5_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l5]], dtype=np.float64)
        self.F_rbs_l6_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l6]], dtype=np.float64)
        self.F_rbs_l7_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l7]], dtype=np.float64)
        self.T_rbs_l7_fas_strut = Z3x1
        self.T_rbs_l8_fas_strut = Z3x1
        self.F_rbs_l8_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l8]], dtype=np.float64)

        self.Mbar_ground_jcs_a = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_l1_jcs_a = multi_dot([A(config.P_rbs_l1).T,triad(config.ax1_jcs_a)])
        self.ubar_ground_jcs_a = (multi_dot([A(self.P_ground).T,config.pt1_jcs_a]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_l1_jcs_a = (multi_dot([A(config.P_rbs_l1).T,config.pt1_jcs_a]) + (-1) * multi_dot([A(config.P_rbs_l1).T,config.R_rbs_l1]))
        self.Mbar_ground_jcs_b = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_b)])
        self.Mbar_rbs_l2_jcs_b = multi_dot([A(config.P_rbs_l2).T,triad(config.ax1_jcs_b)])
        self.ubar_ground_jcs_b = (multi_dot([A(self.P_ground).T,config.pt1_jcs_b]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_l2_jcs_b = (multi_dot([A(config.P_rbs_l2).T,config.pt1_jcs_b]) + (-1) * multi_dot([A(config.P_rbs_l2).T,config.R_rbs_l2]))
        self.Mbar_ground_jcs_c = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_c)])
        self.Mbar_rbs_l3_jcs_c = multi_dot([A(config.P_rbs_l3).T,triad(config.ax1_jcs_c)])
        self.ubar_ground_jcs_c = (multi_dot([A(self.P_ground).T,config.pt1_jcs_c]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_l3_jcs_c = (multi_dot([A(config.P_rbs_l3).T,config.pt1_jcs_c]) + (-1) * multi_dot([A(config.P_rbs_l3).T,config.R_rbs_l3]))
        self.Mbar_rbs_table_jcs_d = multi_dot([A(config.P_rbs_table).T,triad(config.ax1_jcs_d)])
        self.Mbar_rbs_l4_jcs_d = multi_dot([A(config.P_rbs_l4).T,triad(config.ax1_jcs_d)])
        self.ubar_rbs_table_jcs_d = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_d]) + (-1) * multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.ubar_rbs_l4_jcs_d = (multi_dot([A(config.P_rbs_l4).T,config.pt1_jcs_d]) + (-1) * multi_dot([A(config.P_rbs_l4).T,config.R_rbs_l4]))
        self.Mbar_rbs_table_jcs_e = multi_dot([A(config.P_rbs_table).T,triad(config.ax1_jcs_e)])
        self.Mbar_rbs_l5_jcs_e = multi_dot([A(config.P_rbs_l5).T,triad(config.ax1_jcs_e)])
        self.ubar_rbs_table_jcs_e = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_e]) + (-1) * multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.ubar_rbs_l5_jcs_e = (multi_dot([A(config.P_rbs_l5).T,config.pt1_jcs_e]) + (-1) * multi_dot([A(config.P_rbs_l5).T,config.R_rbs_l5]))
        self.Mbar_rbs_table_jcs_f = multi_dot([A(config.P_rbs_table).T,triad(config.ax1_jcs_f)])
        self.Mbar_rbs_l6_jcs_f = multi_dot([A(config.P_rbs_l6).T,triad(config.ax1_jcs_f)])
        self.ubar_rbs_table_jcs_f = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_f]) + (-1) * multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.ubar_rbs_l6_jcs_f = (multi_dot([A(config.P_rbs_l6).T,config.pt1_jcs_f]) + (-1) * multi_dot([A(config.P_rbs_l6).T,config.R_rbs_l6]))
        self.Mbar_rbs_l1_jcs_h1 = multi_dot([A(config.P_rbs_l1).T,triad(config.ax1_jcs_h1)])
        self.Mbar_rbs_l7_jcs_h1 = multi_dot([A(config.P_rbs_l7).T,triad(config.ax1_jcs_h1)])
        self.ubar_rbs_l1_jcs_h1 = (multi_dot([A(config.P_rbs_l1).T,config.pt1_jcs_h1]) + (-1) * multi_dot([A(config.P_rbs_l1).T,config.R_rbs_l1]))
        self.ubar_rbs_l7_jcs_h1 = (multi_dot([A(config.P_rbs_l7).T,config.pt1_jcs_h1]) + (-1) * multi_dot([A(config.P_rbs_l7).T,config.R_rbs_l7]))
        self.Mbar_rbs_l2_jcs_k1 = multi_dot([A(config.P_rbs_l2).T,triad(config.ax1_jcs_k1)])
        self.Mbar_rbs_l7_jcs_k1 = multi_dot([A(config.P_rbs_l7).T,triad(config.ax1_jcs_k1)])
        self.ubar_rbs_l2_jcs_k1 = (multi_dot([A(config.P_rbs_l2).T,config.pt1_jcs_k1]) + (-1) * multi_dot([A(config.P_rbs_l2).T,config.R_rbs_l2]))
        self.ubar_rbs_l7_jcs_k1 = (multi_dot([A(config.P_rbs_l7).T,config.pt1_jcs_k1]) + (-1) * multi_dot([A(config.P_rbs_l7).T,config.R_rbs_l7]))
        self.Mbar_rbs_l3_jcs_l1 = multi_dot([A(config.P_rbs_l3).T,triad(config.ax1_jcs_l1)])
        self.Mbar_rbs_l8_jcs_l1 = multi_dot([A(config.P_rbs_l8).T,triad(config.ax1_jcs_l1)])
        self.ubar_rbs_l3_jcs_l1 = (multi_dot([A(config.P_rbs_l3).T,config.pt1_jcs_l1]) + (-1) * multi_dot([A(config.P_rbs_l3).T,config.R_rbs_l3]))
        self.ubar_rbs_l8_jcs_l1 = (multi_dot([A(config.P_rbs_l8).T,config.pt1_jcs_l1]) + (-1) * multi_dot([A(config.P_rbs_l8).T,config.R_rbs_l8]))
        self.Mbar_rbs_l4_jcs_h2 = multi_dot([A(config.P_rbs_l4).T,triad(config.ax1_jcs_h2)])
        self.Mbar_rbs_l7_jcs_h2 = multi_dot([A(config.P_rbs_l7).T,triad(config.ax1_jcs_h2)])
        self.ubar_rbs_l4_jcs_h2 = (multi_dot([A(config.P_rbs_l4).T,config.pt1_jcs_h2]) + (-1) * multi_dot([A(config.P_rbs_l4).T,config.R_rbs_l4]))
        self.ubar_rbs_l7_jcs_h2 = (multi_dot([A(config.P_rbs_l7).T,config.pt1_jcs_h2]) + (-1) * multi_dot([A(config.P_rbs_l7).T,config.R_rbs_l7]))
        self.Mbar_rbs_l5_jcs_k2 = multi_dot([A(config.P_rbs_l5).T,triad(config.ax1_jcs_k2)])
        self.Mbar_rbs_l7_jcs_k2 = multi_dot([A(config.P_rbs_l7).T,triad(config.ax1_jcs_k2)])
        self.ubar_rbs_l5_jcs_k2 = (multi_dot([A(config.P_rbs_l5).T,config.pt1_jcs_k2]) + (-1) * multi_dot([A(config.P_rbs_l5).T,config.R_rbs_l5]))
        self.ubar_rbs_l7_jcs_k2 = (multi_dot([A(config.P_rbs_l7).T,config.pt1_jcs_k2]) + (-1) * multi_dot([A(config.P_rbs_l7).T,config.R_rbs_l7]))
        self.Mbar_rbs_l6_jcs_l2 = multi_dot([A(config.P_rbs_l6).T,triad(config.ax1_jcs_l2)])
        self.Mbar_rbs_l8_jcs_l2 = multi_dot([A(config.P_rbs_l8).T,triad(config.ax1_jcs_l2)])
        self.ubar_rbs_l6_jcs_l2 = (multi_dot([A(config.P_rbs_l6).T,config.pt1_jcs_l2]) + (-1) * multi_dot([A(config.P_rbs_l6).T,config.R_rbs_l6]))
        self.ubar_rbs_l8_jcs_l2 = (multi_dot([A(config.P_rbs_l8).T,config.pt1_jcs_l2]) + (-1) * multi_dot([A(config.P_rbs_l8).T,config.R_rbs_l8]))
        self.Mbar_rbs_l7_jcs_trans = multi_dot([A(config.P_rbs_l7).T,triad(config.ax1_jcs_trans)])
        self.Mbar_rbs_l8_jcs_trans = multi_dot([A(config.P_rbs_l8).T,triad(config.ax1_jcs_trans)])
        self.ubar_rbs_l7_jcs_trans = (multi_dot([A(config.P_rbs_l7).T,config.pt1_jcs_trans]) + (-1) * multi_dot([A(config.P_rbs_l7).T,config.R_rbs_l7]))
        self.ubar_rbs_l8_jcs_trans = (multi_dot([A(config.P_rbs_l8).T,config.pt1_jcs_trans]) + (-1) * multi_dot([A(config.P_rbs_l8).T,config.R_rbs_l8]))
        self.Mbar_rbs_l7_jcs_trans = multi_dot([A(config.P_rbs_l7).T,triad(config.ax1_jcs_trans)])
        self.Mbar_rbs_l8_jcs_trans = multi_dot([A(config.P_rbs_l8).T,triad(config.ax1_jcs_trans)])
        self.ubar_rbs_l7_jcs_trans = (multi_dot([A(config.P_rbs_l7).T,config.pt1_jcs_trans]) + (-1) * multi_dot([A(config.P_rbs_l7).T,config.R_rbs_l7]))
        self.ubar_rbs_l8_jcs_trans = (multi_dot([A(config.P_rbs_l8).T,config.pt1_jcs_trans]) + (-1) * multi_dot([A(config.P_rbs_l8).T,config.R_rbs_l8]))
        self.ubar_rbs_l7_fas_strut = (multi_dot([A(config.P_rbs_l7).T,config.pt1_fas_strut]) + (-1) * multi_dot([A(config.P_rbs_l7).T,config.R_rbs_l7]))
        self.ubar_rbs_l8_fas_strut = (multi_dot([A(config.P_rbs_l8).T,config.pt2_fas_strut]) + (-1) * multi_dot([A(config.P_rbs_l8).T,config.R_rbs_l8]))

    
    def _map_gen_coordinates(self):
        q = self._q
        self.R_ground = q[0:3]
        self.P_ground = q[3:7]
        self.R_rbs_table = q[7:10]
        self.P_rbs_table = q[10:14]
        self.R_rbs_l1 = q[14:17]
        self.P_rbs_l1 = q[17:21]
        self.R_rbs_l2 = q[21:24]
        self.P_rbs_l2 = q[24:28]
        self.R_rbs_l3 = q[28:31]
        self.P_rbs_l3 = q[31:35]
        self.R_rbs_l4 = q[35:38]
        self.P_rbs_l4 = q[38:42]
        self.R_rbs_l5 = q[42:45]
        self.P_rbs_l5 = q[45:49]
        self.R_rbs_l6 = q[49:52]
        self.P_rbs_l6 = q[52:56]
        self.R_rbs_l7 = q[56:59]
        self.P_rbs_l7 = q[59:63]
        self.R_rbs_l8 = q[63:66]
        self.P_rbs_l8 = q[66:70]

    
    def _map_gen_velocities(self):
        qd = self._qd
        self.Rd_ground = qd[0:3]
        self.Pd_ground = qd[3:7]
        self.Rd_rbs_table = qd[7:10]
        self.Pd_rbs_table = qd[10:14]
        self.Rd_rbs_l1 = qd[14:17]
        self.Pd_rbs_l1 = qd[17:21]
        self.Rd_rbs_l2 = qd[21:24]
        self.Pd_rbs_l2 = qd[24:28]
        self.Rd_rbs_l3 = qd[28:31]
        self.Pd_rbs_l3 = qd[31:35]
        self.Rd_rbs_l4 = qd[35:38]
        self.Pd_rbs_l4 = qd[38:42]
        self.Rd_rbs_l5 = qd[42:45]
        self.Pd_rbs_l5 = qd[45:49]
        self.Rd_rbs_l6 = qd[49:52]
        self.Pd_rbs_l6 = qd[52:56]
        self.Rd_rbs_l7 = qd[56:59]
        self.Pd_rbs_l7 = qd[59:63]
        self.Rd_rbs_l8 = qd[63:66]
        self.Pd_rbs_l8 = qd[66:70]

    
    def _map_gen_accelerations(self):
        qdd = self._qdd
        self.Rdd_ground = qdd[0:3]
        self.Pdd_ground = qdd[3:7]
        self.Rdd_rbs_table = qdd[7:10]
        self.Pdd_rbs_table = qdd[10:14]
        self.Rdd_rbs_l1 = qdd[14:17]
        self.Pdd_rbs_l1 = qdd[17:21]
        self.Rdd_rbs_l2 = qdd[21:24]
        self.Pdd_rbs_l2 = qdd[24:28]
        self.Rdd_rbs_l3 = qdd[28:31]
        self.Pdd_rbs_l3 = qdd[31:35]
        self.Rdd_rbs_l4 = qdd[35:38]
        self.Pdd_rbs_l4 = qdd[38:42]
        self.Rdd_rbs_l5 = qdd[42:45]
        self.Pdd_rbs_l5 = qdd[45:49]
        self.Rdd_rbs_l6 = qdd[49:52]
        self.Pdd_rbs_l6 = qdd[52:56]
        self.Rdd_rbs_l7 = qdd[56:59]
        self.Pdd_rbs_l7 = qdd[59:63]
        self.Rdd_rbs_l8 = qdd[63:66]
        self.Pdd_rbs_l8 = qdd[66:70]

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcs_a = Lambda[0:4]
        self.L_jcs_b = Lambda[4:9]
        self.L_jcs_c = Lambda[9:14]
        self.L_jcs_d = Lambda[14:17]
        self.L_jcs_e = Lambda[17:21]
        self.L_jcs_f = Lambda[21:24]
        self.L_jcs_h1 = Lambda[24:27]
        self.L_jcs_k1 = Lambda[27:31]
        self.L_jcs_l1 = Lambda[31:34]
        self.L_jcs_h2 = Lambda[34:38]
        self.L_jcs_k2 = Lambda[38:43]
        self.L_jcs_l2 = Lambda[43:48]
        self.L_jcs_trans = Lambda[48:53]
        self.L_mcs_act = Lambda[53:54]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.Mbar_ground_jcs_a[:,0:1].T
        x1 = self.P_ground
        x2 = A(x1)
        x3 = x2.T
        x4 = self.P_rbs_l1
        x5 = A(x4)
        x6 = self.Mbar_rbs_l1_jcs_a[:,2:3]
        x7 = self.Mbar_ground_jcs_a[:,1:2].T
        x8 = self.R_ground
        x9 = self.R_rbs_l1
        x10 = (x8 + (-1) * x9 + multi_dot([x2,self.ubar_ground_jcs_a]) + (-1) * multi_dot([x5,self.ubar_rbs_l1_jcs_a]))
        x11 = self.R_rbs_l2
        x12 = self.P_rbs_l2
        x13 = A(x12)
        x14 = self.Mbar_rbs_l2_jcs_b[:,2:3]
        x15 = self.R_rbs_l3
        x16 = self.P_rbs_l3
        x17 = A(x16)
        x18 = self.Mbar_rbs_l3_jcs_c[:,2:3]
        x19 = self.R_rbs_table
        x20 = self.R_rbs_l4
        x21 = self.P_rbs_table
        x22 = A(x21)
        x23 = self.P_rbs_l4
        x24 = A(x23)
        x25 = self.Mbar_rbs_table_jcs_e[:,0:1].T
        x26 = x22.T
        x27 = self.P_rbs_l5
        x28 = A(x27)
        x29 = self.Mbar_rbs_l5_jcs_e[:,2:3]
        x30 = self.Mbar_rbs_table_jcs_e[:,1:2].T
        x31 = self.R_rbs_l5
        x32 = (x19 + (-1) * x31 + multi_dot([x22,self.ubar_rbs_table_jcs_e]) + (-1) * multi_dot([x28,self.ubar_rbs_l5_jcs_e]))
        x33 = self.R_rbs_l6
        x34 = self.P_rbs_l6
        x35 = A(x34)
        x36 = self.R_rbs_l7
        x37 = (-1) * x36
        x38 = self.P_rbs_l7
        x39 = A(x38)
        x40 = self.Mbar_rbs_l2_jcs_k1[:,0:1].T
        x41 = x13.T
        x42 = self.Mbar_rbs_l7_jcs_k1[:,2:3]
        x43 = self.Mbar_rbs_l2_jcs_k1[:,1:2].T
        x44 = (x11 + x37 + multi_dot([x13,self.ubar_rbs_l2_jcs_k1]) + (-1) * multi_dot([x39,self.ubar_rbs_l7_jcs_k1]))
        x45 = (-1) * self.R_rbs_l8
        x46 = self.P_rbs_l8
        x47 = A(x46)
        x48 = self.Mbar_rbs_l4_jcs_h2[:,0:1].T
        x49 = x24.T
        x50 = self.Mbar_rbs_l7_jcs_h2[:,2:3]
        x51 = self.Mbar_rbs_l4_jcs_h2[:,1:2].T
        x52 = (x20 + x37 + multi_dot([x24,self.ubar_rbs_l4_jcs_h2]) + (-1) * multi_dot([x39,self.ubar_rbs_l7_jcs_h2]))
        x53 = x28.T
        x54 = self.Mbar_rbs_l7_jcs_k2[:,2:3]
        x55 = x35.T
        x56 = self.Mbar_rbs_l8_jcs_l2[:,2:3]
        x57 = self.Mbar_rbs_l7_jcs_trans[:,0:1].T
        x58 = x39.T
        x59 = self.Mbar_rbs_l8_jcs_trans[:,2:3]
        x60 = self.Mbar_rbs_l7_jcs_trans[:,1:2].T
        x61 = (x36 + x45 + multi_dot([x39,self.ubar_rbs_l7_jcs_trans]) + (-1) * multi_dot([x47,self.ubar_rbs_l8_jcs_trans]))
        x62 = I1
        x63 = (-1) * x62

        self.pos_eq_blocks = (multi_dot([x0,x3,x5,x6]),
        multi_dot([x7,x3,x5,x6]),
        multi_dot([x0,x3,x10]),
        multi_dot([x7,x3,x10]),
        (x8 + (-1) * x11 + multi_dot([x2,self.ubar_ground_jcs_b]) + (-1) * multi_dot([x13,self.ubar_rbs_l2_jcs_b])),
        multi_dot([self.Mbar_ground_jcs_b[:,0:1].T,x3,x13,x14]),
        multi_dot([self.Mbar_ground_jcs_b[:,1:2].T,x3,x13,x14]),
        (x8 + (-1) * x15 + multi_dot([x2,self.ubar_ground_jcs_c]) + (-1) * multi_dot([x17,self.ubar_rbs_l3_jcs_c])),
        multi_dot([self.Mbar_ground_jcs_c[:,0:1].T,x3,x17,x18]),
        multi_dot([self.Mbar_ground_jcs_c[:,1:2].T,x3,x17,x18]),
        (x19 + (-1) * x20 + multi_dot([x22,self.ubar_rbs_table_jcs_d]) + (-1) * multi_dot([x24,self.ubar_rbs_l4_jcs_d])),
        multi_dot([x25,x26,x28,x29]),
        multi_dot([x30,x26,x28,x29]),
        multi_dot([x25,x26,x32]),
        multi_dot([x30,x26,x32]),
        (x19 + (-1) * x33 + multi_dot([x22,self.ubar_rbs_table_jcs_f]) + (-1) * multi_dot([x35,self.ubar_rbs_l6_jcs_f])),
        (x9 + x37 + multi_dot([x5,self.ubar_rbs_l1_jcs_h1]) + (-1) * multi_dot([x39,self.ubar_rbs_l7_jcs_h1])),
        multi_dot([x40,x41,x39,x42]),
        multi_dot([x43,x41,x39,x42]),
        multi_dot([x40,x41,x44]),
        multi_dot([x43,x41,x44]),
        (x15 + x45 + multi_dot([x17,self.ubar_rbs_l3_jcs_l1]) + (-1) * multi_dot([x47,self.ubar_rbs_l8_jcs_l1])),
        multi_dot([x48,x49,x39,x50]),
        multi_dot([x51,x49,x39,x50]),
        multi_dot([x48,x49,x52]),
        multi_dot([x51,x49,x52]),
        (x31 + x37 + multi_dot([x28,self.ubar_rbs_l5_jcs_k2]) + (-1) * multi_dot([x39,self.ubar_rbs_l7_jcs_k2])),
        multi_dot([self.Mbar_rbs_l5_jcs_k2[:,0:1].T,x53,x39,x54]),
        multi_dot([self.Mbar_rbs_l5_jcs_k2[:,1:2].T,x53,x39,x54]),
        (x33 + x45 + multi_dot([x35,self.ubar_rbs_l6_jcs_l2]) + (-1) * multi_dot([x47,self.ubar_rbs_l8_jcs_l2])),
        multi_dot([self.Mbar_rbs_l6_jcs_l2[:,0:1].T,x55,x47,x56]),
        multi_dot([self.Mbar_rbs_l6_jcs_l2[:,1:2].T,x55,x47,x56]),
        multi_dot([x57,x58,x47,x59]),
        multi_dot([x60,x58,x47,x59]),
        multi_dot([x57,x58,x61]),
        multi_dot([x60,x58,x61]),
        multi_dot([x57,x58,x47,self.Mbar_rbs_l8_jcs_trans[:,1:2]]),
        ((-1 * config.UF_mcs_act(t)) * x62 + multi_dot([self.Mbar_rbs_l7_jcs_trans[:,2:3].T,x58,x61])),
        x8,
        (x1 + (-1) * self.Pg_ground),
        (x63 + multi_dot([x21.T,x21])),
        (x63 + multi_dot([x4.T,x4])),
        (x63 + multi_dot([x12.T,x12])),
        (x63 + multi_dot([x16.T,x16])),
        (x63 + multi_dot([x23.T,x23])),
        (x63 + multi_dot([x27.T,x27])),
        (x63 + multi_dot([x34.T,x34])),
        (x63 + multi_dot([x38.T,x38])),
        (x63 + multi_dot([x46.T,x46])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z1x1
        v1 = Z3x1

        self.vel_eq_blocks = (v0,
        v0,
        v0,
        v0,
        v1,
        v0,
        v0,
        v1,
        v0,
        v0,
        v1,
        v0,
        v0,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v0,
        v0,
        v1,
        v0,
        v0,
        v0,
        v0,
        v1,
        v0,
        v0,
        v1,
        v0,
        v0,
        v0,
        v0,
        v0,
        v0,
        v0,
        (v0 + (-1 * derivative(config.UF_mcs_act, t, 0.1, 1)) * I1),
        v1,
        Z4x1,
        v0,
        v0,
        v0,
        v0,
        v0,
        v0,
        v0,
        v0,
        v0,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Mbar_ground_jcs_a[:,0:1]
        a1 = a0.T
        a2 = self.P_ground
        a3 = A(a2).T
        a4 = self.Pd_rbs_l1
        a5 = self.Mbar_rbs_l1_jcs_a[:,2:3]
        a6 = B(a4,a5)
        a7 = a5.T
        a8 = self.P_rbs_l1
        a9 = A(a8).T
        a10 = self.Pd_ground
        a11 = B(a10,a0)
        a12 = a10.T
        a13 = B(a2,a0).T
        a14 = B(a8,a5)
        a15 = self.Mbar_ground_jcs_a[:,1:2]
        a16 = a15.T
        a17 = B(a10,a15)
        a18 = B(a2,a15).T
        a19 = self.ubar_ground_jcs_a
        a20 = self.ubar_rbs_l1_jcs_a
        a21 = (multi_dot([B(a10,a19),a10]) + (-1) * multi_dot([B(a4,a20),a4]))
        a22 = (self.Rd_ground + (-1) * self.Rd_rbs_l1 + multi_dot([B(a2,a19),a10]) + (-1) * multi_dot([B(a8,a20),a4]))
        a23 = (self.R_ground.T + (-1) * self.R_rbs_l1.T + multi_dot([a19.T,a3]) + (-1) * multi_dot([a20.T,a9]))
        a24 = self.Pd_rbs_l2
        a25 = self.Mbar_ground_jcs_b[:,0:1]
        a26 = self.Mbar_rbs_l2_jcs_b[:,2:3]
        a27 = B(a24,a26)
        a28 = a26.T
        a29 = self.P_rbs_l2
        a30 = A(a29).T
        a31 = B(a29,a26)
        a32 = self.Mbar_ground_jcs_b[:,1:2]
        a33 = self.Pd_rbs_l3
        a34 = self.Mbar_ground_jcs_c[:,0:1]
        a35 = self.Mbar_rbs_l3_jcs_c[:,2:3]
        a36 = B(a33,a35)
        a37 = a35.T
        a38 = self.P_rbs_l3
        a39 = A(a38).T
        a40 = B(a38,a35)
        a41 = self.Mbar_ground_jcs_c[:,1:2]
        a42 = self.Pd_rbs_table
        a43 = self.Pd_rbs_l4
        a44 = self.Mbar_rbs_table_jcs_e[:,0:1]
        a45 = a44.T
        a46 = self.P_rbs_table
        a47 = A(a46).T
        a48 = self.Pd_rbs_l5
        a49 = self.Mbar_rbs_l5_jcs_e[:,2:3]
        a50 = B(a48,a49)
        a51 = a49.T
        a52 = self.P_rbs_l5
        a53 = A(a52).T
        a54 = B(a42,a44)
        a55 = a42.T
        a56 = B(a46,a44).T
        a57 = B(a52,a49)
        a58 = self.Mbar_rbs_table_jcs_e[:,1:2]
        a59 = a58.T
        a60 = B(a42,a58)
        a61 = B(a46,a58).T
        a62 = self.ubar_rbs_table_jcs_e
        a63 = self.ubar_rbs_l5_jcs_e
        a64 = (multi_dot([B(a42,a62),a42]) + (-1) * multi_dot([B(a48,a63),a48]))
        a65 = (self.Rd_rbs_table + (-1) * self.Rd_rbs_l5 + multi_dot([B(a46,a62),a42]) + (-1) * multi_dot([B(a52,a63),a48]))
        a66 = (self.R_rbs_table.T + (-1) * self.R_rbs_l5.T + multi_dot([a62.T,a47]) + (-1) * multi_dot([a63.T,a53]))
        a67 = self.Pd_rbs_l6
        a68 = self.Pd_rbs_l7
        a69 = self.Mbar_rbs_l7_jcs_k1[:,2:3]
        a70 = a69.T
        a71 = self.P_rbs_l7
        a72 = A(a71).T
        a73 = self.Mbar_rbs_l2_jcs_k1[:,0:1]
        a74 = B(a24,a73)
        a75 = a73.T
        a76 = B(a68,a69)
        a77 = a24.T
        a78 = B(a29,a73).T
        a79 = B(a71,a69)
        a80 = self.Mbar_rbs_l2_jcs_k1[:,1:2]
        a81 = B(a24,a80)
        a82 = a80.T
        a83 = B(a29,a80).T
        a84 = self.ubar_rbs_l2_jcs_k1
        a85 = self.ubar_rbs_l7_jcs_k1
        a86 = (multi_dot([B(a24,a84),a24]) + (-1) * multi_dot([B(a68,a85),a68]))
        a87 = self.Rd_rbs_l7
        a88 = (-1) * a87
        a89 = (self.Rd_rbs_l2 + a88 + multi_dot([B(a29,a84),a24]) + (-1) * multi_dot([B(a71,a85),a68]))
        a90 = self.R_rbs_l7.T
        a91 = (-1) * a90
        a92 = (self.R_rbs_l2.T + a91 + multi_dot([a84.T,a30]) + (-1) * multi_dot([a85.T,a72]))
        a93 = self.Pd_rbs_l8
        a94 = self.Mbar_rbs_l7_jcs_h2[:,2:3]
        a95 = a94.T
        a96 = self.Mbar_rbs_l4_jcs_h2[:,0:1]
        a97 = B(a43,a96)
        a98 = a96.T
        a99 = self.P_rbs_l4
        a100 = A(a99).T
        a101 = B(a68,a94)
        a102 = a43.T
        a103 = B(a99,a96).T
        a104 = B(a71,a94)
        a105 = self.Mbar_rbs_l4_jcs_h2[:,1:2]
        a106 = B(a43,a105)
        a107 = a105.T
        a108 = B(a99,a105).T
        a109 = self.ubar_rbs_l4_jcs_h2
        a110 = self.ubar_rbs_l7_jcs_h2
        a111 = (multi_dot([B(a43,a109),a43]) + (-1) * multi_dot([B(a68,a110),a68]))
        a112 = (self.Rd_rbs_l4 + a88 + multi_dot([B(a99,a109),a43]) + (-1) * multi_dot([B(a71,a110),a68]))
        a113 = (self.R_rbs_l4.T + a91 + multi_dot([a109.T,a100]) + (-1) * multi_dot([a110.T,a72]))
        a114 = self.Mbar_rbs_l7_jcs_k2[:,2:3]
        a115 = a114.T
        a116 = self.Mbar_rbs_l5_jcs_k2[:,0:1]
        a117 = B(a68,a114)
        a118 = a48.T
        a119 = B(a71,a114)
        a120 = self.Mbar_rbs_l5_jcs_k2[:,1:2]
        a121 = self.Mbar_rbs_l8_jcs_l2[:,2:3]
        a122 = a121.T
        a123 = self.P_rbs_l8
        a124 = A(a123).T
        a125 = self.Mbar_rbs_l6_jcs_l2[:,0:1]
        a126 = self.P_rbs_l6
        a127 = A(a126).T
        a128 = B(a93,a121)
        a129 = a67.T
        a130 = B(a123,a121)
        a131 = self.Mbar_rbs_l6_jcs_l2[:,1:2]
        a132 = self.Mbar_rbs_l8_jcs_trans[:,2:3]
        a133 = a132.T
        a134 = self.Mbar_rbs_l7_jcs_trans[:,0:1]
        a135 = B(a68,a134)
        a136 = a134.T
        a137 = B(a93,a132)
        a138 = a68.T
        a139 = B(a71,a134).T
        a140 = B(a123,a132)
        a141 = self.Mbar_rbs_l7_jcs_trans[:,1:2]
        a142 = B(a68,a141)
        a143 = a141.T
        a144 = B(a71,a141).T
        a145 = self.ubar_rbs_l7_jcs_trans
        a146 = self.ubar_rbs_l8_jcs_trans
        a147 = (multi_dot([B(a68,a145),a68]) + (-1) * multi_dot([B(a93,a146),a93]))
        a148 = (a87 + (-1) * self.Rd_rbs_l8 + multi_dot([B(a71,a145),a68]) + (-1) * multi_dot([B(a123,a146),a93]))
        a149 = (a90 + (-1) * self.R_rbs_l8.T + multi_dot([a145.T,a72]) + (-1) * multi_dot([a146.T,a124]))
        a150 = self.Mbar_rbs_l8_jcs_trans[:,1:2]
        a151 = self.Mbar_rbs_l7_jcs_trans[:,2:3]

        self.acc_eq_blocks = ((multi_dot([a1,a3,a6,a4]) + multi_dot([a7,a9,a11,a10]) + (2) * multi_dot([a12,a13,a14,a4])),
        (multi_dot([a16,a3,a6,a4]) + multi_dot([a7,a9,a17,a10]) + (2) * multi_dot([a12,a18,a14,a4])),
        (multi_dot([a1,a3,a21]) + (2) * multi_dot([a12,a13,a22]) + multi_dot([a23,a11,a10])),
        (multi_dot([a16,a3,a21]) + (2) * multi_dot([a12,a18,a22]) + multi_dot([a23,a17,a10])),
        (multi_dot([B(a10,self.ubar_ground_jcs_b),a10]) + (-1) * multi_dot([B(a24,self.ubar_rbs_l2_jcs_b),a24])),
        (multi_dot([a25.T,a3,a27,a24]) + multi_dot([a28,a30,B(a10,a25),a10]) + (2) * multi_dot([a12,B(a2,a25).T,a31,a24])),
        (multi_dot([a32.T,a3,a27,a24]) + multi_dot([a28,a30,B(a10,a32),a10]) + (2) * multi_dot([a12,B(a2,a32).T,a31,a24])),
        (multi_dot([B(a10,self.ubar_ground_jcs_c),a10]) + (-1) * multi_dot([B(a33,self.ubar_rbs_l3_jcs_c),a33])),
        (multi_dot([a34.T,a3,a36,a33]) + multi_dot([a37,a39,B(a10,a34),a10]) + (2) * multi_dot([a12,B(a2,a34).T,a40,a33])),
        (multi_dot([a41.T,a3,a36,a33]) + multi_dot([a37,a39,B(a10,a41),a10]) + (2) * multi_dot([a12,B(a2,a41).T,a40,a33])),
        (multi_dot([B(a42,self.ubar_rbs_table_jcs_d),a42]) + (-1) * multi_dot([B(a43,self.ubar_rbs_l4_jcs_d),a43])),
        (multi_dot([a45,a47,a50,a48]) + multi_dot([a51,a53,a54,a42]) + (2) * multi_dot([a55,a56,a57,a48])),
        (multi_dot([a59,a47,a50,a48]) + multi_dot([a51,a53,a60,a42]) + (2) * multi_dot([a55,a61,a57,a48])),
        (multi_dot([a45,a47,a64]) + (2) * multi_dot([a55,a56,a65]) + multi_dot([a66,a54,a42])),
        (multi_dot([a59,a47,a64]) + (2) * multi_dot([a55,a61,a65]) + multi_dot([a66,a60,a42])),
        (multi_dot([B(a42,self.ubar_rbs_table_jcs_f),a42]) + (-1) * multi_dot([B(a67,self.ubar_rbs_l6_jcs_f),a67])),
        (multi_dot([B(a4,self.ubar_rbs_l1_jcs_h1),a4]) + (-1) * multi_dot([B(a68,self.ubar_rbs_l7_jcs_h1),a68])),
        (multi_dot([a70,a72,a74,a24]) + multi_dot([a75,a30,a76,a68]) + (2) * multi_dot([a77,a78,a79,a68])),
        (multi_dot([a70,a72,a81,a24]) + multi_dot([a82,a30,a76,a68]) + (2) * multi_dot([a77,a83,a79,a68])),
        (multi_dot([a75,a30,a86]) + (2) * multi_dot([a77,a78,a89]) + multi_dot([a92,a74,a24])),
        (multi_dot([a82,a30,a86]) + (2) * multi_dot([a77,a83,a89]) + multi_dot([a92,a81,a24])),
        (multi_dot([B(a33,self.ubar_rbs_l3_jcs_l1),a33]) + (-1) * multi_dot([B(a93,self.ubar_rbs_l8_jcs_l1),a93])),
        (multi_dot([a95,a72,a97,a43]) + multi_dot([a98,a100,a101,a68]) + (2) * multi_dot([a102,a103,a104,a68])),
        (multi_dot([a95,a72,a106,a43]) + multi_dot([a107,a100,a101,a68]) + (2) * multi_dot([a102,a108,a104,a68])),
        (multi_dot([a98,a100,a111]) + (2) * multi_dot([a102,a103,a112]) + multi_dot([a113,a97,a43])),
        (multi_dot([a107,a100,a111]) + (2) * multi_dot([a102,a108,a112]) + multi_dot([a113,a106,a43])),
        (multi_dot([B(a48,self.ubar_rbs_l5_jcs_k2),a48]) + (-1) * multi_dot([B(a68,self.ubar_rbs_l7_jcs_k2),a68])),
        (multi_dot([a115,a72,B(a48,a116),a48]) + multi_dot([a116.T,a53,a117,a68]) + (2) * multi_dot([a118,B(a52,a116).T,a119,a68])),
        (multi_dot([a115,a72,B(a48,a120),a48]) + multi_dot([a120.T,a53,a117,a68]) + (2) * multi_dot([a118,B(a52,a120).T,a119,a68])),
        (multi_dot([B(a67,self.ubar_rbs_l6_jcs_l2),a67]) + (-1) * multi_dot([B(a93,self.ubar_rbs_l8_jcs_l2),a93])),
        (multi_dot([a122,a124,B(a67,a125),a67]) + multi_dot([a125.T,a127,a128,a93]) + (2) * multi_dot([a129,B(a126,a125).T,a130,a93])),
        (multi_dot([a122,a124,B(a67,a131),a67]) + multi_dot([a131.T,a127,a128,a93]) + (2) * multi_dot([a129,B(a126,a131).T,a130,a93])),
        (multi_dot([a133,a124,a135,a68]) + multi_dot([a136,a72,a137,a93]) + (2) * multi_dot([a138,a139,a140,a93])),
        (multi_dot([a133,a124,a142,a68]) + multi_dot([a143,a72,a137,a93]) + (2) * multi_dot([a138,a144,a140,a93])),
        (multi_dot([a136,a72,a147]) + (2) * multi_dot([a138,a139,a148]) + multi_dot([a149,a135,a68])),
        (multi_dot([a143,a72,a147]) + (2) * multi_dot([a138,a144,a148]) + multi_dot([a149,a142,a68])),
        (multi_dot([a150.T,a124,a135,a68]) + multi_dot([a136,a72,B(a93,a150),a93]) + (2) * multi_dot([a138,a139,B(a123,a150),a93])),
        ((-1 * derivative(config.UF_mcs_act, t, 0.1, 2)) * I1 + multi_dot([a151.T,a72,a147]) + (2) * multi_dot([a138,B(a71,a151).T,a148]) + multi_dot([a149,B(a68,a151),a68])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a55,a42]),
        (2) * multi_dot([a4.T,a4]),
        (2) * multi_dot([a77,a24]),
        (2) * multi_dot([a33.T,a33]),
        (2) * multi_dot([a102,a43]),
        (2) * multi_dot([a118,a48]),
        (2) * multi_dot([a129,a67]),
        (2) * multi_dot([a138,a68]),
        (2) * multi_dot([a93.T,a93]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = Z1x3
        j1 = self.Mbar_rbs_l1_jcs_a[:,2:3]
        j2 = j1.T
        j3 = self.P_rbs_l1
        j4 = A(j3).T
        j5 = self.P_ground
        j6 = self.Mbar_ground_jcs_a[:,0:1]
        j7 = B(j5,j6)
        j8 = self.Mbar_ground_jcs_a[:,1:2]
        j9 = B(j5,j8)
        j10 = j6.T
        j11 = A(j5).T
        j12 = multi_dot([j10,j11])
        j13 = self.ubar_ground_jcs_a
        j14 = B(j5,j13)
        j15 = self.ubar_rbs_l1_jcs_a
        j16 = (self.R_ground.T + (-1) * self.R_rbs_l1.T + multi_dot([j13.T,j11]) + (-1) * multi_dot([j15.T,j4]))
        j17 = j8.T
        j18 = multi_dot([j17,j11])
        j19 = B(j3,j1)
        j20 = B(j3,j15)
        j21 = I3
        j22 = self.Mbar_rbs_l2_jcs_b[:,2:3]
        j23 = j22.T
        j24 = self.P_rbs_l2
        j25 = A(j24).T
        j26 = self.Mbar_ground_jcs_b[:,0:1]
        j27 = self.Mbar_ground_jcs_b[:,1:2]
        j28 = (-1) * j21
        j29 = B(j24,j22)
        j30 = self.Mbar_rbs_l3_jcs_c[:,2:3]
        j31 = j30.T
        j32 = self.P_rbs_l3
        j33 = A(j32).T
        j34 = self.Mbar_ground_jcs_c[:,0:1]
        j35 = self.Mbar_ground_jcs_c[:,1:2]
        j36 = B(j32,j30)
        j37 = self.P_rbs_table
        j38 = self.P_rbs_l4
        j39 = self.Mbar_rbs_l5_jcs_e[:,2:3]
        j40 = j39.T
        j41 = self.P_rbs_l5
        j42 = A(j41).T
        j43 = self.Mbar_rbs_table_jcs_e[:,0:1]
        j44 = B(j37,j43)
        j45 = self.Mbar_rbs_table_jcs_e[:,1:2]
        j46 = B(j37,j45)
        j47 = j43.T
        j48 = A(j37).T
        j49 = multi_dot([j47,j48])
        j50 = self.ubar_rbs_table_jcs_e
        j51 = B(j37,j50)
        j52 = self.ubar_rbs_l5_jcs_e
        j53 = (self.R_rbs_table.T + (-1) * self.R_rbs_l5.T + multi_dot([j50.T,j48]) + (-1) * multi_dot([j52.T,j42]))
        j54 = j45.T
        j55 = multi_dot([j54,j48])
        j56 = B(j41,j39)
        j57 = B(j41,j52)
        j58 = self.P_rbs_l6
        j59 = self.P_rbs_l7
        j60 = self.Mbar_rbs_l7_jcs_k1[:,2:3]
        j61 = j60.T
        j62 = A(j59).T
        j63 = self.Mbar_rbs_l2_jcs_k1[:,0:1]
        j64 = B(j24,j63)
        j65 = self.Mbar_rbs_l2_jcs_k1[:,1:2]
        j66 = B(j24,j65)
        j67 = j63.T
        j68 = multi_dot([j67,j25])
        j69 = self.ubar_rbs_l2_jcs_k1
        j70 = B(j24,j69)
        j71 = self.R_rbs_l7.T
        j72 = (-1) * j71
        j73 = self.ubar_rbs_l7_jcs_k1
        j74 = (self.R_rbs_l2.T + j72 + multi_dot([j69.T,j25]) + (-1) * multi_dot([j73.T,j62]))
        j75 = j65.T
        j76 = multi_dot([j75,j25])
        j77 = B(j59,j60)
        j78 = B(j59,j73)
        j79 = self.P_rbs_l8
        j80 = self.Mbar_rbs_l7_jcs_h2[:,2:3]
        j81 = j80.T
        j82 = self.Mbar_rbs_l4_jcs_h2[:,0:1]
        j83 = B(j38,j82)
        j84 = self.Mbar_rbs_l4_jcs_h2[:,1:2]
        j85 = B(j38,j84)
        j86 = j82.T
        j87 = A(j38).T
        j88 = multi_dot([j86,j87])
        j89 = self.ubar_rbs_l4_jcs_h2
        j90 = B(j38,j89)
        j91 = self.ubar_rbs_l7_jcs_h2
        j92 = (self.R_rbs_l4.T + j72 + multi_dot([j89.T,j87]) + (-1) * multi_dot([j91.T,j62]))
        j93 = j84.T
        j94 = multi_dot([j93,j87])
        j95 = B(j59,j80)
        j96 = B(j59,j91)
        j97 = self.Mbar_rbs_l7_jcs_k2[:,2:3]
        j98 = j97.T
        j99 = self.Mbar_rbs_l5_jcs_k2[:,0:1]
        j100 = self.Mbar_rbs_l5_jcs_k2[:,1:2]
        j101 = B(j59,j97)
        j102 = self.Mbar_rbs_l8_jcs_l2[:,2:3]
        j103 = j102.T
        j104 = A(j79).T
        j105 = self.Mbar_rbs_l6_jcs_l2[:,0:1]
        j106 = self.Mbar_rbs_l6_jcs_l2[:,1:2]
        j107 = A(j58).T
        j108 = B(j79,j102)
        j109 = self.Mbar_rbs_l8_jcs_trans[:,2:3]
        j110 = j109.T
        j111 = self.Mbar_rbs_l7_jcs_trans[:,0:1]
        j112 = B(j59,j111)
        j113 = self.Mbar_rbs_l7_jcs_trans[:,1:2]
        j114 = B(j59,j113)
        j115 = j111.T
        j116 = multi_dot([j115,j62])
        j117 = self.ubar_rbs_l7_jcs_trans
        j118 = B(j59,j117)
        j119 = self.ubar_rbs_l8_jcs_trans
        j120 = (j71 + (-1) * self.R_rbs_l8.T + multi_dot([j117.T,j62]) + (-1) * multi_dot([j119.T,j104]))
        j121 = j113.T
        j122 = multi_dot([j121,j62])
        j123 = self.Mbar_rbs_l8_jcs_trans[:,1:2]
        j124 = B(j79,j109)
        j125 = B(j79,j119)
        j126 = self.Mbar_rbs_l7_jcs_trans[:,2:3]
        j127 = j126.T
        j128 = multi_dot([j127,j62])

        self.jac_eq_blocks = (j0,
        multi_dot([j2,j4,j7]),
        j0,
        multi_dot([j10,j11,j19]),
        j0,
        multi_dot([j2,j4,j9]),
        j0,
        multi_dot([j17,j11,j19]),
        j12,
        (multi_dot([j10,j11,j14]) + multi_dot([j16,j7])),
        (-1) * j12,
        (-1) * multi_dot([j10,j11,j20]),
        j18,
        (multi_dot([j17,j11,j14]) + multi_dot([j16,j9])),
        (-1) * j18,
        (-1) * multi_dot([j17,j11,j20]),
        j21,
        B(j5,self.ubar_ground_jcs_b),
        j28,
        (-1) * B(j24,self.ubar_rbs_l2_jcs_b),
        j0,
        multi_dot([j23,j25,B(j5,j26)]),
        j0,
        multi_dot([j26.T,j11,j29]),
        j0,
        multi_dot([j23,j25,B(j5,j27)]),
        j0,
        multi_dot([j27.T,j11,j29]),
        j21,
        B(j5,self.ubar_ground_jcs_c),
        j28,
        (-1) * B(j32,self.ubar_rbs_l3_jcs_c),
        j0,
        multi_dot([j31,j33,B(j5,j34)]),
        j0,
        multi_dot([j34.T,j11,j36]),
        j0,
        multi_dot([j31,j33,B(j5,j35)]),
        j0,
        multi_dot([j35.T,j11,j36]),
        j21,
        B(j37,self.ubar_rbs_table_jcs_d),
        j28,
        (-1) * B(j38,self.ubar_rbs_l4_jcs_d),
        j0,
        multi_dot([j40,j42,j44]),
        j0,
        multi_dot([j47,j48,j56]),
        j0,
        multi_dot([j40,j42,j46]),
        j0,
        multi_dot([j54,j48,j56]),
        j49,
        (multi_dot([j47,j48,j51]) + multi_dot([j53,j44])),
        (-1) * j49,
        (-1) * multi_dot([j47,j48,j57]),
        j55,
        (multi_dot([j54,j48,j51]) + multi_dot([j53,j46])),
        (-1) * j55,
        (-1) * multi_dot([j54,j48,j57]),
        j21,
        B(j37,self.ubar_rbs_table_jcs_f),
        j28,
        (-1) * B(j58,self.ubar_rbs_l6_jcs_f),
        j21,
        B(j3,self.ubar_rbs_l1_jcs_h1),
        j28,
        (-1) * B(j59,self.ubar_rbs_l7_jcs_h1),
        j0,
        multi_dot([j61,j62,j64]),
        j0,
        multi_dot([j67,j25,j77]),
        j0,
        multi_dot([j61,j62,j66]),
        j0,
        multi_dot([j75,j25,j77]),
        j68,
        (multi_dot([j67,j25,j70]) + multi_dot([j74,j64])),
        (-1) * j68,
        (-1) * multi_dot([j67,j25,j78]),
        j76,
        (multi_dot([j75,j25,j70]) + multi_dot([j74,j66])),
        (-1) * j76,
        (-1) * multi_dot([j75,j25,j78]),
        j21,
        B(j32,self.ubar_rbs_l3_jcs_l1),
        j28,
        (-1) * B(j79,self.ubar_rbs_l8_jcs_l1),
        j0,
        multi_dot([j81,j62,j83]),
        j0,
        multi_dot([j86,j87,j95]),
        j0,
        multi_dot([j81,j62,j85]),
        j0,
        multi_dot([j93,j87,j95]),
        j88,
        (multi_dot([j86,j87,j90]) + multi_dot([j92,j83])),
        (-1) * j88,
        (-1) * multi_dot([j86,j87,j96]),
        j94,
        (multi_dot([j93,j87,j90]) + multi_dot([j92,j85])),
        (-1) * j94,
        (-1) * multi_dot([j93,j87,j96]),
        j21,
        B(j41,self.ubar_rbs_l5_jcs_k2),
        j28,
        (-1) * B(j59,self.ubar_rbs_l7_jcs_k2),
        j0,
        multi_dot([j98,j62,B(j41,j99)]),
        j0,
        multi_dot([j99.T,j42,j101]),
        j0,
        multi_dot([j98,j62,B(j41,j100)]),
        j0,
        multi_dot([j100.T,j42,j101]),
        j21,
        B(j58,self.ubar_rbs_l6_jcs_l2),
        j28,
        (-1) * B(j79,self.ubar_rbs_l8_jcs_l2),
        j0,
        multi_dot([j103,j104,B(j58,j105)]),
        j0,
        multi_dot([j105.T,j107,j108]),
        j0,
        multi_dot([j103,j104,B(j58,j106)]),
        j0,
        multi_dot([j106.T,j107,j108]),
        j0,
        multi_dot([j110,j104,j112]),
        j0,
        multi_dot([j115,j62,j124]),
        j0,
        multi_dot([j110,j104,j114]),
        j0,
        multi_dot([j121,j62,j124]),
        j116,
        (multi_dot([j115,j62,j118]) + multi_dot([j120,j112])),
        (-1) * j116,
        (-1) * multi_dot([j115,j62,j125]),
        j122,
        (multi_dot([j121,j62,j118]) + multi_dot([j120,j114])),
        (-1) * j122,
        (-1) * multi_dot([j121,j62,j125]),
        j0,
        multi_dot([j123.T,j104,j112]),
        j0,
        multi_dot([j115,j62,B(j79,j123)]),
        j128,
        (multi_dot([j127,j62,j118]) + multi_dot([j120,B(j59,j126)])),
        (-1) * j128,
        (-1) * multi_dot([j127,j62,j125]),
        j21,
        Z3x4,
        Z4x3,
        I4,
        j0,
        (2) * j37.T,
        j0,
        (2) * j3.T,
        j0,
        (2) * j24.T,
        j0,
        (2) * j32.T,
        j0,
        (2) * j38.T,
        j0,
        (2) * j41.T,
        j0,
        (2) * j58.T,
        j0,
        (2) * j59.T,
        j0,
        (2) * j79.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_ground)
        m2 = G(self.P_rbs_table)
        m3 = G(self.P_rbs_l1)
        m4 = G(self.P_rbs_l2)
        m5 = G(self.P_rbs_l3)
        m6 = G(self.P_rbs_l4)
        m7 = G(self.P_rbs_l5)
        m8 = G(self.P_rbs_l6)
        m9 = G(self.P_rbs_l7)
        m10 = G(self.P_rbs_l8)

        self.mass_eq_blocks = (self.m_ground * m0,
        (4) * multi_dot([m1.T,self.Jbar_ground,m1]),
        config.m_rbs_table * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbs_table,m2]),
        config.m_rbs_l1 * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbs_l1,m3]),
        config.m_rbs_l2 * m0,
        (4) * multi_dot([m4.T,config.Jbar_rbs_l2,m4]),
        config.m_rbs_l3 * m0,
        (4) * multi_dot([m5.T,config.Jbar_rbs_l3,m5]),
        config.m_rbs_l4 * m0,
        (4) * multi_dot([m6.T,config.Jbar_rbs_l4,m6]),
        config.m_rbs_l5 * m0,
        (4) * multi_dot([m7.T,config.Jbar_rbs_l5,m7]),
        config.m_rbs_l6 * m0,
        (4) * multi_dot([m8.T,config.Jbar_rbs_l6,m8]),
        config.m_rbs_l7 * m0,
        (4) * multi_dot([m9.T,config.Jbar_rbs_l7,m9]),
        config.m_rbs_l8 * m0,
        (4) * multi_dot([m10.T,config.Jbar_rbs_l8,m10]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = Z3x1
        f1 = Z4x1
        f2 = G(self.Pd_rbs_table)
        f3 = G(self.Pd_rbs_l1)
        f4 = G(self.Pd_rbs_l2)
        f5 = G(self.Pd_rbs_l3)
        f6 = G(self.Pd_rbs_l4)
        f7 = G(self.Pd_rbs_l5)
        f8 = G(self.Pd_rbs_l6)
        f9 = self.R_rbs_l7
        f10 = self.R_rbs_l8
        f11 = self.ubar_rbs_l7_fas_strut
        f12 = self.P_rbs_l7
        f13 = A(f12)
        f14 = self.ubar_rbs_l8_fas_strut
        f15 = self.P_rbs_l8
        f16 = A(f15)
        f17 = (f9.T + (-1) * f10.T + multi_dot([f11.T,f13.T]) + (-1) * multi_dot([f14.T,f16.T]))
        f18 = multi_dot([f13,f11])
        f19 = multi_dot([f16,f14])
        f20 = (f9 + (-1) * f10 + f18 + (-1) * f19)
        f21 = ((multi_dot([f17,f20]))**(1.0/2.0))[0]
        f22 = 1.0/f21
        f23 = config.UF_fas_strut_Fs((config.fas_strut_FL + (-1 * f21)))
        f24 = self.Pd_rbs_l7
        f25 = self.Pd_rbs_l8
        f26 = config.UF_fas_strut_Fd((-1 * 1.0/f21) * multi_dot([f17,(self.Rd_rbs_l7 + (-1) * self.Rd_rbs_l8 + multi_dot([B(f12,f11),f24]) + (-1) * multi_dot([B(f15,f14),f25]))]))
        f27 = (f22 * (f23 + f26)) * f20
        f28 = G(f24)
        f29 = (2 * f23)
        f30 = (2 * f26)
        f31 = G(f25)

        self.frc_eq_blocks = (f0,
        f1,
        self.F_rbs_table_gravity,
        (8) * multi_dot([f2.T,config.Jbar_rbs_table,f2,self.P_rbs_table]),
        self.F_rbs_l1_gravity,
        (8) * multi_dot([f3.T,config.Jbar_rbs_l1,f3,self.P_rbs_l1]),
        self.F_rbs_l2_gravity,
        (8) * multi_dot([f4.T,config.Jbar_rbs_l2,f4,self.P_rbs_l2]),
        self.F_rbs_l3_gravity,
        (8) * multi_dot([f5.T,config.Jbar_rbs_l3,f5,self.P_rbs_l3]),
        self.F_rbs_l4_gravity,
        (8) * multi_dot([f6.T,config.Jbar_rbs_l4,f6,self.P_rbs_l4]),
        self.F_rbs_l5_gravity,
        (8) * multi_dot([f7.T,config.Jbar_rbs_l5,f7,self.P_rbs_l5]),
        self.F_rbs_l6_gravity,
        (8) * multi_dot([f8.T,config.Jbar_rbs_l6,f8,self.P_rbs_l6]),
        (self.F_rbs_l7_gravity + f27),
        ((8) * multi_dot([f28.T,config.Jbar_rbs_l7,f28,f12]) + (f22 * ((-1 * f29) + (-1 * f30))) * multi_dot([E(f12).T,skew(f18).T,f20])),
        (self.F_rbs_l8_gravity + f0 + (-1) * f27),
        (f1 + (8) * multi_dot([f31.T,config.Jbar_rbs_l8,f31,f15]) + (f22 * (f29 + f30)) * multi_dot([E(f15).T,skew(f19).T,f20])),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_ground_jcs_a = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_ground),self.Mbar_ground_jcs_a[:,0:1]]),multi_dot([A(self.P_ground),self.Mbar_ground_jcs_a[:,1:2]])],[multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,0:1]).T,A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_a[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,1:2]).T,A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_a[:,2:3]]),(multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,0:1]).T,((-1) * self.R_rbs_l1 + multi_dot([A(self.P_ground),self.ubar_ground_jcs_a]) + (-1) * multi_dot([A(self.P_rbs_l1),self.ubar_rbs_l1_jcs_a]) + self.R_ground)]) + multi_dot([B(self.P_ground,self.ubar_ground_jcs_a).T,A(self.P_ground),self.Mbar_ground_jcs_a[:,0:1]])),(multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,1:2]).T,((-1) * self.R_rbs_l1 + multi_dot([A(self.P_ground),self.ubar_ground_jcs_a]) + (-1) * multi_dot([A(self.P_rbs_l1),self.ubar_rbs_l1_jcs_a]) + self.R_ground)]) + multi_dot([B(self.P_ground,self.ubar_ground_jcs_a).T,A(self.P_ground),self.Mbar_ground_jcs_a[:,1:2]]))]]),self.L_jcs_a])
        self.F_ground_jcs_a = Q_ground_jcs_a[0:3]
        Te_ground_jcs_a = Q_ground_jcs_a[3:7]
        self.T_ground_jcs_a = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_a])),self.F_ground_jcs_a]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_a]))
        Q_ground_jcs_b = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_ground,self.ubar_ground_jcs_b).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_b[:,0:1]).T,A(self.P_rbs_l2),self.Mbar_rbs_l2_jcs_b[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_b[:,1:2]).T,A(self.P_rbs_l2),self.Mbar_rbs_l2_jcs_b[:,2:3]])]]),self.L_jcs_b])
        self.F_ground_jcs_b = Q_ground_jcs_b[0:3]
        Te_ground_jcs_b = Q_ground_jcs_b[3:7]
        self.T_ground_jcs_b = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_b])),self.F_ground_jcs_b]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_b]))
        Q_ground_jcs_c = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_ground,self.ubar_ground_jcs_c).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_c[:,0:1]).T,A(self.P_rbs_l3),self.Mbar_rbs_l3_jcs_c[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_c[:,1:2]).T,A(self.P_rbs_l3),self.Mbar_rbs_l3_jcs_c[:,2:3]])]]),self.L_jcs_c])
        self.F_ground_jcs_c = Q_ground_jcs_c[0:3]
        Te_ground_jcs_c = Q_ground_jcs_c[3:7]
        self.T_ground_jcs_c = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_c])),self.F_ground_jcs_c]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_c]))
        Q_rbs_table_jcs_d = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_table,self.ubar_rbs_table_jcs_d).T]]),self.L_jcs_d])
        self.F_rbs_table_jcs_d = Q_rbs_table_jcs_d[0:3]
        Te_rbs_table_jcs_d = Q_rbs_table_jcs_d[3:7]
        self.T_rbs_table_jcs_d = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_table),self.ubar_rbs_table_jcs_d])),self.F_rbs_table_jcs_d]) + (0.5) * multi_dot([E(self.P_rbs_table),Te_rbs_table_jcs_d]))
        Q_rbs_table_jcs_e = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_table),self.Mbar_rbs_table_jcs_e[:,0:1]]),multi_dot([A(self.P_rbs_table),self.Mbar_rbs_table_jcs_e[:,1:2]])],[multi_dot([B(self.P_rbs_table,self.Mbar_rbs_table_jcs_e[:,0:1]).T,A(self.P_rbs_l5),self.Mbar_rbs_l5_jcs_e[:,2:3]]),multi_dot([B(self.P_rbs_table,self.Mbar_rbs_table_jcs_e[:,1:2]).T,A(self.P_rbs_l5),self.Mbar_rbs_l5_jcs_e[:,2:3]]),(multi_dot([B(self.P_rbs_table,self.Mbar_rbs_table_jcs_e[:,0:1]).T,((-1) * self.R_rbs_l5 + multi_dot([A(self.P_rbs_table),self.ubar_rbs_table_jcs_e]) + (-1) * multi_dot([A(self.P_rbs_l5),self.ubar_rbs_l5_jcs_e]) + self.R_rbs_table)]) + multi_dot([B(self.P_rbs_table,self.ubar_rbs_table_jcs_e).T,A(self.P_rbs_table),self.Mbar_rbs_table_jcs_e[:,0:1]])),(multi_dot([B(self.P_rbs_table,self.Mbar_rbs_table_jcs_e[:,1:2]).T,((-1) * self.R_rbs_l5 + multi_dot([A(self.P_rbs_table),self.ubar_rbs_table_jcs_e]) + (-1) * multi_dot([A(self.P_rbs_l5),self.ubar_rbs_l5_jcs_e]) + self.R_rbs_table)]) + multi_dot([B(self.P_rbs_table,self.ubar_rbs_table_jcs_e).T,A(self.P_rbs_table),self.Mbar_rbs_table_jcs_e[:,1:2]]))]]),self.L_jcs_e])
        self.F_rbs_table_jcs_e = Q_rbs_table_jcs_e[0:3]
        Te_rbs_table_jcs_e = Q_rbs_table_jcs_e[3:7]
        self.T_rbs_table_jcs_e = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_table),self.ubar_rbs_table_jcs_e])),self.F_rbs_table_jcs_e]) + (0.5) * multi_dot([E(self.P_rbs_table),Te_rbs_table_jcs_e]))
        Q_rbs_table_jcs_f = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_table,self.ubar_rbs_table_jcs_f).T]]),self.L_jcs_f])
        self.F_rbs_table_jcs_f = Q_rbs_table_jcs_f[0:3]
        Te_rbs_table_jcs_f = Q_rbs_table_jcs_f[3:7]
        self.T_rbs_table_jcs_f = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_table),self.ubar_rbs_table_jcs_f])),self.F_rbs_table_jcs_f]) + (0.5) * multi_dot([E(self.P_rbs_table),Te_rbs_table_jcs_f]))
        Q_rbs_l1_jcs_h1 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l1,self.ubar_rbs_l1_jcs_h1).T]]),self.L_jcs_h1])
        self.F_rbs_l1_jcs_h1 = Q_rbs_l1_jcs_h1[0:3]
        Te_rbs_l1_jcs_h1 = Q_rbs_l1_jcs_h1[3:7]
        self.T_rbs_l1_jcs_h1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l1),self.ubar_rbs_l1_jcs_h1])),self.F_rbs_l1_jcs_h1]) + (0.5) * multi_dot([E(self.P_rbs_l1),Te_rbs_l1_jcs_h1]))
        Q_rbs_l2_jcs_k1 = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_l2),self.Mbar_rbs_l2_jcs_k1[:,0:1]]),multi_dot([A(self.P_rbs_l2),self.Mbar_rbs_l2_jcs_k1[:,1:2]])],[multi_dot([B(self.P_rbs_l2,self.Mbar_rbs_l2_jcs_k1[:,0:1]).T,A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_k1[:,2:3]]),multi_dot([B(self.P_rbs_l2,self.Mbar_rbs_l2_jcs_k1[:,1:2]).T,A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_k1[:,2:3]]),(multi_dot([B(self.P_rbs_l2,self.Mbar_rbs_l2_jcs_k1[:,0:1]).T,((-1) * self.R_rbs_l7 + multi_dot([A(self.P_rbs_l2),self.ubar_rbs_l2_jcs_k1]) + (-1) * multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_jcs_k1]) + self.R_rbs_l2)]) + multi_dot([B(self.P_rbs_l2,self.ubar_rbs_l2_jcs_k1).T,A(self.P_rbs_l2),self.Mbar_rbs_l2_jcs_k1[:,0:1]])),(multi_dot([B(self.P_rbs_l2,self.Mbar_rbs_l2_jcs_k1[:,1:2]).T,((-1) * self.R_rbs_l7 + multi_dot([A(self.P_rbs_l2),self.ubar_rbs_l2_jcs_k1]) + (-1) * multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_jcs_k1]) + self.R_rbs_l2)]) + multi_dot([B(self.P_rbs_l2,self.ubar_rbs_l2_jcs_k1).T,A(self.P_rbs_l2),self.Mbar_rbs_l2_jcs_k1[:,1:2]]))]]),self.L_jcs_k1])
        self.F_rbs_l2_jcs_k1 = Q_rbs_l2_jcs_k1[0:3]
        Te_rbs_l2_jcs_k1 = Q_rbs_l2_jcs_k1[3:7]
        self.T_rbs_l2_jcs_k1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l2),self.ubar_rbs_l2_jcs_k1])),self.F_rbs_l2_jcs_k1]) + (0.5) * multi_dot([E(self.P_rbs_l2),Te_rbs_l2_jcs_k1]))
        Q_rbs_l3_jcs_l1 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l3,self.ubar_rbs_l3_jcs_l1).T]]),self.L_jcs_l1])
        self.F_rbs_l3_jcs_l1 = Q_rbs_l3_jcs_l1[0:3]
        Te_rbs_l3_jcs_l1 = Q_rbs_l3_jcs_l1[3:7]
        self.T_rbs_l3_jcs_l1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l3),self.ubar_rbs_l3_jcs_l1])),self.F_rbs_l3_jcs_l1]) + (0.5) * multi_dot([E(self.P_rbs_l3),Te_rbs_l3_jcs_l1]))
        Q_rbs_l4_jcs_h2 = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_l4),self.Mbar_rbs_l4_jcs_h2[:,0:1]]),multi_dot([A(self.P_rbs_l4),self.Mbar_rbs_l4_jcs_h2[:,1:2]])],[multi_dot([B(self.P_rbs_l4,self.Mbar_rbs_l4_jcs_h2[:,0:1]).T,A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_h2[:,2:3]]),multi_dot([B(self.P_rbs_l4,self.Mbar_rbs_l4_jcs_h2[:,1:2]).T,A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_h2[:,2:3]]),(multi_dot([B(self.P_rbs_l4,self.Mbar_rbs_l4_jcs_h2[:,0:1]).T,((-1) * self.R_rbs_l7 + multi_dot([A(self.P_rbs_l4),self.ubar_rbs_l4_jcs_h2]) + (-1) * multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_jcs_h2]) + self.R_rbs_l4)]) + multi_dot([B(self.P_rbs_l4,self.ubar_rbs_l4_jcs_h2).T,A(self.P_rbs_l4),self.Mbar_rbs_l4_jcs_h2[:,0:1]])),(multi_dot([B(self.P_rbs_l4,self.Mbar_rbs_l4_jcs_h2[:,1:2]).T,((-1) * self.R_rbs_l7 + multi_dot([A(self.P_rbs_l4),self.ubar_rbs_l4_jcs_h2]) + (-1) * multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_jcs_h2]) + self.R_rbs_l4)]) + multi_dot([B(self.P_rbs_l4,self.ubar_rbs_l4_jcs_h2).T,A(self.P_rbs_l4),self.Mbar_rbs_l4_jcs_h2[:,1:2]]))]]),self.L_jcs_h2])
        self.F_rbs_l4_jcs_h2 = Q_rbs_l4_jcs_h2[0:3]
        Te_rbs_l4_jcs_h2 = Q_rbs_l4_jcs_h2[3:7]
        self.T_rbs_l4_jcs_h2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l4),self.ubar_rbs_l4_jcs_h2])),self.F_rbs_l4_jcs_h2]) + (0.5) * multi_dot([E(self.P_rbs_l4),Te_rbs_l4_jcs_h2]))
        Q_rbs_l5_jcs_k2 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_l5,self.ubar_rbs_l5_jcs_k2).T,multi_dot([B(self.P_rbs_l5,self.Mbar_rbs_l5_jcs_k2[:,0:1]).T,A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_k2[:,2:3]]),multi_dot([B(self.P_rbs_l5,self.Mbar_rbs_l5_jcs_k2[:,1:2]).T,A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_k2[:,2:3]])]]),self.L_jcs_k2])
        self.F_rbs_l5_jcs_k2 = Q_rbs_l5_jcs_k2[0:3]
        Te_rbs_l5_jcs_k2 = Q_rbs_l5_jcs_k2[3:7]
        self.T_rbs_l5_jcs_k2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l5),self.ubar_rbs_l5_jcs_k2])),self.F_rbs_l5_jcs_k2]) + (0.5) * multi_dot([E(self.P_rbs_l5),Te_rbs_l5_jcs_k2]))
        Q_rbs_l6_jcs_l2 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_l6,self.ubar_rbs_l6_jcs_l2).T,multi_dot([B(self.P_rbs_l6,self.Mbar_rbs_l6_jcs_l2[:,0:1]).T,A(self.P_rbs_l8),self.Mbar_rbs_l8_jcs_l2[:,2:3]]),multi_dot([B(self.P_rbs_l6,self.Mbar_rbs_l6_jcs_l2[:,1:2]).T,A(self.P_rbs_l8),self.Mbar_rbs_l8_jcs_l2[:,2:3]])]]),self.L_jcs_l2])
        self.F_rbs_l6_jcs_l2 = Q_rbs_l6_jcs_l2[0:3]
        Te_rbs_l6_jcs_l2 = Q_rbs_l6_jcs_l2[3:7]
        self.T_rbs_l6_jcs_l2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l6),self.ubar_rbs_l6_jcs_l2])),self.F_rbs_l6_jcs_l2]) + (0.5) * multi_dot([E(self.P_rbs_l6),Te_rbs_l6_jcs_l2]))
        Q_rbs_l7_jcs_trans = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_trans[:,0:1]]),multi_dot([A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_trans[:,1:2]]),Z1x3.T],[multi_dot([B(self.P_rbs_l7,self.Mbar_rbs_l7_jcs_trans[:,0:1]).T,A(self.P_rbs_l8),self.Mbar_rbs_l8_jcs_trans[:,2:3]]),multi_dot([B(self.P_rbs_l7,self.Mbar_rbs_l7_jcs_trans[:,1:2]).T,A(self.P_rbs_l8),self.Mbar_rbs_l8_jcs_trans[:,2:3]]),(multi_dot([B(self.P_rbs_l7,self.Mbar_rbs_l7_jcs_trans[:,0:1]).T,((-1) * self.R_rbs_l8 + multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_jcs_trans]) + (-1) * multi_dot([A(self.P_rbs_l8),self.ubar_rbs_l8_jcs_trans]) + self.R_rbs_l7)]) + multi_dot([B(self.P_rbs_l7,self.ubar_rbs_l7_jcs_trans).T,A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_trans[:,0:1]])),(multi_dot([B(self.P_rbs_l7,self.Mbar_rbs_l7_jcs_trans[:,1:2]).T,((-1) * self.R_rbs_l8 + multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_jcs_trans]) + (-1) * multi_dot([A(self.P_rbs_l8),self.ubar_rbs_l8_jcs_trans]) + self.R_rbs_l7)]) + multi_dot([B(self.P_rbs_l7,self.ubar_rbs_l7_jcs_trans).T,A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_trans[:,1:2]])),multi_dot([B(self.P_rbs_l7,self.Mbar_rbs_l7_jcs_trans[:,0:1]).T,A(self.P_rbs_l8),self.Mbar_rbs_l8_jcs_trans[:,1:2]])]]),self.L_jcs_trans])
        self.F_rbs_l7_jcs_trans = Q_rbs_l7_jcs_trans[0:3]
        Te_rbs_l7_jcs_trans = Q_rbs_l7_jcs_trans[3:7]
        self.T_rbs_l7_jcs_trans = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_jcs_trans])),self.F_rbs_l7_jcs_trans]) + (0.5) * multi_dot([E(self.P_rbs_l7),Te_rbs_l7_jcs_trans]))
        Q_rbs_l7_mcs_act = (-1) * multi_dot([np.bmat([[multi_dot([A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_trans[:,2:3]])],[(multi_dot([B(self.P_rbs_l7,self.Mbar_rbs_l7_jcs_trans[:,2:3]).T,((-1) * self.R_rbs_l8 + multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_jcs_trans]) + (-1) * multi_dot([A(self.P_rbs_l8),self.ubar_rbs_l8_jcs_trans]) + self.R_rbs_l7)]) + multi_dot([B(self.P_rbs_l7,self.ubar_rbs_l7_jcs_trans).T,A(self.P_rbs_l7),self.Mbar_rbs_l7_jcs_trans[:,2:3]]))]]),self.L_mcs_act])
        self.F_rbs_l7_mcs_act = Q_rbs_l7_mcs_act[0:3]
        Te_rbs_l7_mcs_act = Q_rbs_l7_mcs_act[3:7]
        self.T_rbs_l7_mcs_act = (0.5) * multi_dot([E(self.P_rbs_l7),Te_rbs_l7_mcs_act])
        self.F_rbs_l7_fas_strut = (1.0/((multi_dot([((-1) * self.R_rbs_l8.T + multi_dot([self.ubar_rbs_l7_fas_strut.T,A(self.P_rbs_l7).T]) + (-1) * multi_dot([self.ubar_rbs_l8_fas_strut.T,A(self.P_rbs_l8).T]) + self.R_rbs_l7.T),((-1) * self.R_rbs_l8 + multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_fas_strut]) + (-1) * multi_dot([A(self.P_rbs_l8),self.ubar_rbs_l8_fas_strut]) + self.R_rbs_l7)]))**(1.0/2.0))[0] * (config.UF_fas_strut_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_rbs_l8.T + multi_dot([self.ubar_rbs_l7_fas_strut.T,A(self.P_rbs_l7).T]) + (-1) * multi_dot([self.ubar_rbs_l8_fas_strut.T,A(self.P_rbs_l8).T]) + self.R_rbs_l7.T),((-1) * self.R_rbs_l8 + multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_fas_strut]) + (-1) * multi_dot([A(self.P_rbs_l8),self.ubar_rbs_l8_fas_strut]) + self.R_rbs_l7)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_rbs_l8.T + multi_dot([self.ubar_rbs_l7_fas_strut.T,A(self.P_rbs_l7).T]) + (-1) * multi_dot([self.ubar_rbs_l8_fas_strut.T,A(self.P_rbs_l8).T]) + self.R_rbs_l7.T),((-1) * self.Rd_rbs_l8 + multi_dot([B(self.P_rbs_l7,self.ubar_rbs_l7_fas_strut),self.Pd_rbs_l7]) + (-1) * multi_dot([B(self.P_rbs_l8,self.ubar_rbs_l8_fas_strut),self.Pd_rbs_l8]) + self.Rd_rbs_l7)])) + config.UF_fas_strut_Fs((config.fas_strut_FL + (-1 * ((multi_dot([((-1) * self.R_rbs_l8.T + multi_dot([self.ubar_rbs_l7_fas_strut.T,A(self.P_rbs_l7).T]) + (-1) * multi_dot([self.ubar_rbs_l8_fas_strut.T,A(self.P_rbs_l8).T]) + self.R_rbs_l7.T),((-1) * self.R_rbs_l8 + multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_fas_strut]) + (-1) * multi_dot([A(self.P_rbs_l8),self.ubar_rbs_l8_fas_strut]) + self.R_rbs_l7)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_rbs_l8 + multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_fas_strut]) + (-1) * multi_dot([A(self.P_rbs_l8),self.ubar_rbs_l8_fas_strut]) + self.R_rbs_l7)
        self.T_rbs_l7_fas_strut = Z3x1

        self.reactions = {'F_ground_jcs_a' : self.F_ground_jcs_a,
                        'T_ground_jcs_a' : self.T_ground_jcs_a,
                        'F_ground_jcs_b' : self.F_ground_jcs_b,
                        'T_ground_jcs_b' : self.T_ground_jcs_b,
                        'F_ground_jcs_c' : self.F_ground_jcs_c,
                        'T_ground_jcs_c' : self.T_ground_jcs_c,
                        'F_rbs_table_jcs_d' : self.F_rbs_table_jcs_d,
                        'T_rbs_table_jcs_d' : self.T_rbs_table_jcs_d,
                        'F_rbs_table_jcs_e' : self.F_rbs_table_jcs_e,
                        'T_rbs_table_jcs_e' : self.T_rbs_table_jcs_e,
                        'F_rbs_table_jcs_f' : self.F_rbs_table_jcs_f,
                        'T_rbs_table_jcs_f' : self.T_rbs_table_jcs_f,
                        'F_rbs_l1_jcs_h1' : self.F_rbs_l1_jcs_h1,
                        'T_rbs_l1_jcs_h1' : self.T_rbs_l1_jcs_h1,
                        'F_rbs_l2_jcs_k1' : self.F_rbs_l2_jcs_k1,
                        'T_rbs_l2_jcs_k1' : self.T_rbs_l2_jcs_k1,
                        'F_rbs_l3_jcs_l1' : self.F_rbs_l3_jcs_l1,
                        'T_rbs_l3_jcs_l1' : self.T_rbs_l3_jcs_l1,
                        'F_rbs_l4_jcs_h2' : self.F_rbs_l4_jcs_h2,
                        'T_rbs_l4_jcs_h2' : self.T_rbs_l4_jcs_h2,
                        'F_rbs_l5_jcs_k2' : self.F_rbs_l5_jcs_k2,
                        'T_rbs_l5_jcs_k2' : self.T_rbs_l5_jcs_k2,
                        'F_rbs_l6_jcs_l2' : self.F_rbs_l6_jcs_l2,
                        'T_rbs_l6_jcs_l2' : self.T_rbs_l6_jcs_l2,
                        'F_rbs_l7_jcs_trans' : self.F_rbs_l7_jcs_trans,
                        'T_rbs_l7_jcs_trans' : self.T_rbs_l7_jcs_trans,
                        'F_rbs_l7_mcs_act' : self.F_rbs_l7_mcs_act,
                        'T_rbs_l7_mcs_act' : self.T_rbs_l7_mcs_act,
                        'F_rbs_l7_fas_strut' : self.F_rbs_l7_fas_strut,
                        'T_rbs_l7_fas_strut' : self.T_rbs_l7_fas_strut}

