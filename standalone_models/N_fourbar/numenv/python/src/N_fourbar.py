
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

        self.indicies_map = {'ground': 0, 'rbs_l1': 1, 'rbs_l2': 2, 'rbs_l3': 3, 'rbs_l4': 4, 'rbs_l5': 5, 'rbs_l6': 6, 'rbs_l7': 7, 'rbs_l8': 8, 'rbs_l9': 9}

        self.n  = 70
        self.nc = 65
        self.nrows = 34
        self.ncols = 2*10
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_ground_jcs_j1', 'T_ground_jcs_j1', 'F_rbs_l1_jcs_j2', 'T_rbs_l1_jcs_j2', 'F_rbs_l2_jcs_j3', 'T_rbs_l2_jcs_j3', 'F_rbs_l3_jcs_j4', 'T_rbs_l3_jcs_j4', 'F_rbs_l3_jcs_j5', 'T_rbs_l3_jcs_j5', 'F_rbs_l4_jcs_j6', 'T_rbs_l4_jcs_j6', 'F_rbs_l5_jcs_j7', 'T_rbs_l5_jcs_j7', 'F_rbs_l5_jcs_j8', 'T_rbs_l5_jcs_j8', 'F_rbs_l6_jcs_j9', 'T_rbs_l6_jcs_j9', 'F_rbs_l7_jcs_j10', 'T_rbs_l7_jcs_j10', 'F_rbs_l7_jcs_j11', 'T_rbs_l7_jcs_j11', 'F_rbs_l8_jcs_j12', 'T_rbs_l8_jcs_j12', 'F_rbs_l9_jcs_j13', 'T_rbs_l9_jcs_j13']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.ground*2, self.ground*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.ground*2, self.ground*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.ground*2, self.ground*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.ground*2, self.ground*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.ground*2, self.ground*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.ground*2, self.ground*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l6*2, self.rbs_l6*2+1, self.rbs_l6*2, self.rbs_l6*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.ground*2, self.ground*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.ground*2, self.ground*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.ground*2, self.ground*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l9*2, self.rbs_l9*2+1, self.ground*2, self.ground*2+1, self.rbs_l9*2, self.rbs_l9*2+1, self.ground*2, self.ground*2+1, self.rbs_l9*2, self.rbs_l9*2+1, self.ground*2, self.ground*2+1, self.rbs_l9*2, self.rbs_l9*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l6*2, self.rbs_l6*2+1, self.rbs_l7*2, self.rbs_l7*2+1, self.rbs_l8*2, self.rbs_l8*2+1, self.rbs_l9*2, self.rbs_l9*2+1], dtype=np.intc)

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
        self.config.P_rbs_l8,
        self.config.R_rbs_l9,
        self.config.P_rbs_l9], out=self._q)

        np.concatenate([self.config.Rd_ground,
        self.config.Pd_ground,
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
        self.config.Pd_rbs_l8,
        self.config.Rd_rbs_l9,
        self.config.Pd_rbs_l9], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p + 'ground']
        self.rbs_l1 = indicies_map[p + 'rbs_l1']
        self.rbs_l2 = indicies_map[p + 'rbs_l2']
        self.rbs_l3 = indicies_map[p + 'rbs_l3']
        self.rbs_l4 = indicies_map[p + 'rbs_l4']
        self.rbs_l5 = indicies_map[p + 'rbs_l5']
        self.rbs_l6 = indicies_map[p + 'rbs_l6']
        self.rbs_l7 = indicies_map[p + 'rbs_l7']
        self.rbs_l8 = indicies_map[p + 'rbs_l8']
        self.rbs_l9 = indicies_map[p + 'rbs_l9']
    

    
    def eval_constants(self):
        config = self.config

        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.m_ground = 1.0
        self.Jbar_ground = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.F_rbs_l1_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l1]], dtype=np.float64)
        self.F_rbs_l2_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l2]], dtype=np.float64)
        self.F_rbs_l3_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l3]], dtype=np.float64)
        self.F_rbs_l4_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l4]], dtype=np.float64)
        self.F_rbs_l5_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l5]], dtype=np.float64)
        self.F_rbs_l6_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l6]], dtype=np.float64)
        self.F_rbs_l7_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l7]], dtype=np.float64)
        self.F_rbs_l8_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l8]], dtype=np.float64)
        self.F_rbs_l9_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_l9]], dtype=np.float64)

        self.Mbar_ground_jcs_j1 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_j1)])
        self.Mbar_rbs_l1_jcs_j1 = multi_dot([A(config.P_rbs_l1).T,triad(config.ax1_jcs_j1)])
        self.ubar_ground_jcs_j1 = (multi_dot([A(self.P_ground).T,config.pt1_jcs_j1]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_l1_jcs_j1 = (multi_dot([A(config.P_rbs_l1).T,config.pt1_jcs_j1]) + (-1) * multi_dot([A(config.P_rbs_l1).T,config.R_rbs_l1]))
        self.Mbar_rbs_l1_jcs_j2 = multi_dot([A(config.P_rbs_l1).T,triad(config.ax1_jcs_j2)])
        self.Mbar_rbs_l2_jcs_j2 = multi_dot([A(config.P_rbs_l2).T,triad(config.ax1_jcs_j2)])
        self.ubar_rbs_l1_jcs_j2 = (multi_dot([A(config.P_rbs_l1).T,config.pt1_jcs_j2]) + (-1) * multi_dot([A(config.P_rbs_l1).T,config.R_rbs_l1]))
        self.ubar_rbs_l2_jcs_j2 = (multi_dot([A(config.P_rbs_l2).T,config.pt1_jcs_j2]) + (-1) * multi_dot([A(config.P_rbs_l2).T,config.R_rbs_l2]))
        self.Mbar_rbs_l2_jcs_j3 = multi_dot([A(config.P_rbs_l2).T,triad(config.ax1_jcs_j3)])
        self.Mbar_rbs_l3_jcs_j3 = multi_dot([A(config.P_rbs_l3).T,triad(config.ax1_jcs_j3)])
        self.ubar_rbs_l2_jcs_j3 = (multi_dot([A(config.P_rbs_l2).T,config.pt1_jcs_j3]) + (-1) * multi_dot([A(config.P_rbs_l2).T,config.R_rbs_l2]))
        self.ubar_rbs_l3_jcs_j3 = (multi_dot([A(config.P_rbs_l3).T,config.pt1_jcs_j3]) + (-1) * multi_dot([A(config.P_rbs_l3).T,config.R_rbs_l3]))
        self.Mbar_rbs_l3_jcs_j4 = multi_dot([A(config.P_rbs_l3).T,triad(config.ax1_jcs_j4)])
        self.Mbar_ground_jcs_j4 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_j4)])
        self.ubar_rbs_l3_jcs_j4 = (multi_dot([A(config.P_rbs_l3).T,config.pt1_jcs_j4]) + (-1) * multi_dot([A(config.P_rbs_l3).T,config.R_rbs_l3]))
        self.ubar_ground_jcs_j4 = (multi_dot([A(self.P_ground).T,config.pt1_jcs_j4]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbs_l3_jcs_j5 = multi_dot([A(config.P_rbs_l3).T,triad(config.ax1_jcs_j5)])
        self.Mbar_rbs_l4_jcs_j5 = multi_dot([A(config.P_rbs_l4).T,triad(config.ax1_jcs_j5)])
        self.ubar_rbs_l3_jcs_j5 = (multi_dot([A(config.P_rbs_l3).T,config.pt1_jcs_j5]) + (-1) * multi_dot([A(config.P_rbs_l3).T,config.R_rbs_l3]))
        self.ubar_rbs_l4_jcs_j5 = (multi_dot([A(config.P_rbs_l4).T,config.pt1_jcs_j5]) + (-1) * multi_dot([A(config.P_rbs_l4).T,config.R_rbs_l4]))
        self.Mbar_rbs_l4_jcs_j6 = multi_dot([A(config.P_rbs_l4).T,triad(config.ax1_jcs_j6)])
        self.Mbar_rbs_l5_jcs_j6 = multi_dot([A(config.P_rbs_l5).T,triad(config.ax1_jcs_j6)])
        self.ubar_rbs_l4_jcs_j6 = (multi_dot([A(config.P_rbs_l4).T,config.pt1_jcs_j6]) + (-1) * multi_dot([A(config.P_rbs_l4).T,config.R_rbs_l4]))
        self.ubar_rbs_l5_jcs_j6 = (multi_dot([A(config.P_rbs_l5).T,config.pt1_jcs_j6]) + (-1) * multi_dot([A(config.P_rbs_l5).T,config.R_rbs_l5]))
        self.Mbar_rbs_l5_jcs_j7 = multi_dot([A(config.P_rbs_l5).T,triad(config.ax1_jcs_j7)])
        self.Mbar_ground_jcs_j7 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_j7)])
        self.ubar_rbs_l5_jcs_j7 = (multi_dot([A(config.P_rbs_l5).T,config.pt1_jcs_j7]) + (-1) * multi_dot([A(config.P_rbs_l5).T,config.R_rbs_l5]))
        self.ubar_ground_jcs_j7 = (multi_dot([A(self.P_ground).T,config.pt1_jcs_j7]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbs_l5_jcs_j8 = multi_dot([A(config.P_rbs_l5).T,triad(config.ax1_jcs_j8)])
        self.Mbar_rbs_l6_jcs_j8 = multi_dot([A(config.P_rbs_l6).T,triad(config.ax1_jcs_j8)])
        self.ubar_rbs_l5_jcs_j8 = (multi_dot([A(config.P_rbs_l5).T,config.pt1_jcs_j8]) + (-1) * multi_dot([A(config.P_rbs_l5).T,config.R_rbs_l5]))
        self.ubar_rbs_l6_jcs_j8 = (multi_dot([A(config.P_rbs_l6).T,config.pt1_jcs_j8]) + (-1) * multi_dot([A(config.P_rbs_l6).T,config.R_rbs_l6]))
        self.Mbar_rbs_l6_jcs_j9 = multi_dot([A(config.P_rbs_l6).T,triad(config.ax1_jcs_j9)])
        self.Mbar_rbs_l7_jcs_j9 = multi_dot([A(config.P_rbs_l7).T,triad(config.ax1_jcs_j9)])
        self.ubar_rbs_l6_jcs_j9 = (multi_dot([A(config.P_rbs_l6).T,config.pt1_jcs_j9]) + (-1) * multi_dot([A(config.P_rbs_l6).T,config.R_rbs_l6]))
        self.ubar_rbs_l7_jcs_j9 = (multi_dot([A(config.P_rbs_l7).T,config.pt1_jcs_j9]) + (-1) * multi_dot([A(config.P_rbs_l7).T,config.R_rbs_l7]))
        self.Mbar_rbs_l7_jcs_j10 = multi_dot([A(config.P_rbs_l7).T,triad(config.ax1_jcs_j10)])
        self.Mbar_ground_jcs_j10 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_j10)])
        self.ubar_rbs_l7_jcs_j10 = (multi_dot([A(config.P_rbs_l7).T,config.pt1_jcs_j10]) + (-1) * multi_dot([A(config.P_rbs_l7).T,config.R_rbs_l7]))
        self.ubar_ground_jcs_j10 = (multi_dot([A(self.P_ground).T,config.pt1_jcs_j10]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbs_l7_jcs_j11 = multi_dot([A(config.P_rbs_l7).T,triad(config.ax1_jcs_j11)])
        self.Mbar_rbs_l8_jcs_j11 = multi_dot([A(config.P_rbs_l8).T,triad(config.ax1_jcs_j11)])
        self.ubar_rbs_l7_jcs_j11 = (multi_dot([A(config.P_rbs_l7).T,config.pt1_jcs_j11]) + (-1) * multi_dot([A(config.P_rbs_l7).T,config.R_rbs_l7]))
        self.ubar_rbs_l8_jcs_j11 = (multi_dot([A(config.P_rbs_l8).T,config.pt1_jcs_j11]) + (-1) * multi_dot([A(config.P_rbs_l8).T,config.R_rbs_l8]))
        self.Mbar_rbs_l8_jcs_j12 = multi_dot([A(config.P_rbs_l8).T,triad(config.ax1_jcs_j12)])
        self.Mbar_rbs_l9_jcs_j12 = multi_dot([A(config.P_rbs_l9).T,triad(config.ax1_jcs_j12)])
        self.ubar_rbs_l8_jcs_j12 = (multi_dot([A(config.P_rbs_l8).T,config.pt1_jcs_j12]) + (-1) * multi_dot([A(config.P_rbs_l8).T,config.R_rbs_l8]))
        self.ubar_rbs_l9_jcs_j12 = (multi_dot([A(config.P_rbs_l9).T,config.pt1_jcs_j12]) + (-1) * multi_dot([A(config.P_rbs_l9).T,config.R_rbs_l9]))
        self.Mbar_rbs_l9_jcs_j13 = multi_dot([A(config.P_rbs_l9).T,triad(config.ax1_jcs_j13)])
        self.Mbar_ground_jcs_j13 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_j13)])
        self.ubar_rbs_l9_jcs_j13 = (multi_dot([A(config.P_rbs_l9).T,config.pt1_jcs_j13]) + (-1) * multi_dot([A(config.P_rbs_l9).T,config.R_rbs_l9]))
        self.ubar_ground_jcs_j13 = (multi_dot([A(self.P_ground).T,config.pt1_jcs_j13]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))

    
    def _map_gen_coordinates(self):
        q = self._q
        self.R_ground = q[0:3]
        self.P_ground = q[3:7]
        self.R_rbs_l1 = q[7:10]
        self.P_rbs_l1 = q[10:14]
        self.R_rbs_l2 = q[14:17]
        self.P_rbs_l2 = q[17:21]
        self.R_rbs_l3 = q[21:24]
        self.P_rbs_l3 = q[24:28]
        self.R_rbs_l4 = q[28:31]
        self.P_rbs_l4 = q[31:35]
        self.R_rbs_l5 = q[35:38]
        self.P_rbs_l5 = q[38:42]
        self.R_rbs_l6 = q[42:45]
        self.P_rbs_l6 = q[45:49]
        self.R_rbs_l7 = q[49:52]
        self.P_rbs_l7 = q[52:56]
        self.R_rbs_l8 = q[56:59]
        self.P_rbs_l8 = q[59:63]
        self.R_rbs_l9 = q[63:66]
        self.P_rbs_l9 = q[66:70]

    
    def _map_gen_velocities(self):
        qd = self._qd
        self.Rd_ground = qd[0:3]
        self.Pd_ground = qd[3:7]
        self.Rd_rbs_l1 = qd[7:10]
        self.Pd_rbs_l1 = qd[10:14]
        self.Rd_rbs_l2 = qd[14:17]
        self.Pd_rbs_l2 = qd[17:21]
        self.Rd_rbs_l3 = qd[21:24]
        self.Pd_rbs_l3 = qd[24:28]
        self.Rd_rbs_l4 = qd[28:31]
        self.Pd_rbs_l4 = qd[31:35]
        self.Rd_rbs_l5 = qd[35:38]
        self.Pd_rbs_l5 = qd[38:42]
        self.Rd_rbs_l6 = qd[42:45]
        self.Pd_rbs_l6 = qd[45:49]
        self.Rd_rbs_l7 = qd[49:52]
        self.Pd_rbs_l7 = qd[52:56]
        self.Rd_rbs_l8 = qd[56:59]
        self.Pd_rbs_l8 = qd[59:63]
        self.Rd_rbs_l9 = qd[63:66]
        self.Pd_rbs_l9 = qd[66:70]

    
    def _map_gen_accelerations(self):
        qdd = self._qdd
        self.Rdd_ground = qdd[0:3]
        self.Pdd_ground = qdd[3:7]
        self.Rdd_rbs_l1 = qdd[7:10]
        self.Pdd_rbs_l1 = qdd[10:14]
        self.Rdd_rbs_l2 = qdd[14:17]
        self.Pdd_rbs_l2 = qdd[17:21]
        self.Rdd_rbs_l3 = qdd[21:24]
        self.Pdd_rbs_l3 = qdd[24:28]
        self.Rdd_rbs_l4 = qdd[28:31]
        self.Pdd_rbs_l4 = qdd[31:35]
        self.Rdd_rbs_l5 = qdd[35:38]
        self.Pdd_rbs_l5 = qdd[38:42]
        self.Rdd_rbs_l6 = qdd[42:45]
        self.Pdd_rbs_l6 = qdd[45:49]
        self.Rdd_rbs_l7 = qdd[49:52]
        self.Pdd_rbs_l7 = qdd[52:56]
        self.Rdd_rbs_l8 = qdd[56:59]
        self.Pdd_rbs_l8 = qdd[59:63]
        self.Rdd_rbs_l9 = qdd[63:66]
        self.Pdd_rbs_l9 = qdd[66:70]

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcs_j1 = Lambda[0:5]
        self.L_jcs_j2 = Lambda[5:8]
        self.L_jcs_j3 = Lambda[8:11]
        self.L_jcs_j4 = Lambda[11:16]
        self.L_jcs_j5 = Lambda[16:19]
        self.L_jcs_j6 = Lambda[19:22]
        self.L_jcs_j7 = Lambda[22:27]
        self.L_jcs_j8 = Lambda[27:30]
        self.L_jcs_j9 = Lambda[30:33]
        self.L_jcs_j10 = Lambda[33:38]
        self.L_jcs_j11 = Lambda[38:41]
        self.L_jcs_j12 = Lambda[41:44]
        self.L_jcs_j13 = Lambda[44:49]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_ground
        x1 = self.R_rbs_l1
        x2 = self.P_ground
        x3 = A(x2)
        x4 = self.P_rbs_l1
        x5 = A(x4)
        x6 = x3.T
        x7 = self.Mbar_rbs_l1_jcs_j1[:,2:3]
        x8 = self.R_rbs_l2
        x9 = self.P_rbs_l2
        x10 = A(x9)
        x11 = self.R_rbs_l3
        x12 = self.P_rbs_l3
        x13 = A(x12)
        x14 = (-1) * x0
        x15 = x13.T
        x16 = self.Mbar_ground_jcs_j4[:,2:3]
        x17 = self.R_rbs_l4
        x18 = self.P_rbs_l4
        x19 = A(x18)
        x20 = self.R_rbs_l5
        x21 = self.P_rbs_l5
        x22 = A(x21)
        x23 = x22.T
        x24 = self.Mbar_ground_jcs_j7[:,2:3]
        x25 = self.R_rbs_l6
        x26 = self.P_rbs_l6
        x27 = A(x26)
        x28 = self.R_rbs_l7
        x29 = self.P_rbs_l7
        x30 = A(x29)
        x31 = x30.T
        x32 = self.Mbar_ground_jcs_j10[:,2:3]
        x33 = self.R_rbs_l8
        x34 = self.P_rbs_l8
        x35 = A(x34)
        x36 = self.R_rbs_l9
        x37 = self.P_rbs_l9
        x38 = A(x37)
        x39 = x38.T
        x40 = self.Mbar_ground_jcs_j13[:,2:3]
        x41 = (-1) * I1

        self.pos_eq_blocks = ((x0 + (-1) * x1 + multi_dot([x3,self.ubar_ground_jcs_j1]) + (-1) * multi_dot([x5,self.ubar_rbs_l1_jcs_j1])),
        multi_dot([self.Mbar_ground_jcs_j1[:,0:1].T,x6,x5,x7]),
        multi_dot([self.Mbar_ground_jcs_j1[:,1:2].T,x6,x5,x7]),
        (x1 + (-1) * x8 + multi_dot([x5,self.ubar_rbs_l1_jcs_j2]) + (-1) * multi_dot([x10,self.ubar_rbs_l2_jcs_j2])),
        (x8 + (-1) * x11 + multi_dot([x10,self.ubar_rbs_l2_jcs_j3]) + (-1) * multi_dot([x13,self.ubar_rbs_l3_jcs_j3])),
        (x11 + x14 + multi_dot([x13,self.ubar_rbs_l3_jcs_j4]) + (-1) * multi_dot([x3,self.ubar_ground_jcs_j4])),
        multi_dot([self.Mbar_rbs_l3_jcs_j4[:,0:1].T,x15,x3,x16]),
        multi_dot([self.Mbar_rbs_l3_jcs_j4[:,1:2].T,x15,x3,x16]),
        (x11 + (-1) * x17 + multi_dot([x13,self.ubar_rbs_l3_jcs_j5]) + (-1) * multi_dot([x19,self.ubar_rbs_l4_jcs_j5])),
        (x17 + (-1) * x20 + multi_dot([x19,self.ubar_rbs_l4_jcs_j6]) + (-1) * multi_dot([x22,self.ubar_rbs_l5_jcs_j6])),
        (x20 + x14 + multi_dot([x22,self.ubar_rbs_l5_jcs_j7]) + (-1) * multi_dot([x3,self.ubar_ground_jcs_j7])),
        multi_dot([self.Mbar_rbs_l5_jcs_j7[:,0:1].T,x23,x3,x24]),
        multi_dot([self.Mbar_rbs_l5_jcs_j7[:,1:2].T,x23,x3,x24]),
        (x20 + (-1) * x25 + multi_dot([x22,self.ubar_rbs_l5_jcs_j8]) + (-1) * multi_dot([x27,self.ubar_rbs_l6_jcs_j8])),
        (x25 + (-1) * x28 + multi_dot([x27,self.ubar_rbs_l6_jcs_j9]) + (-1) * multi_dot([x30,self.ubar_rbs_l7_jcs_j9])),
        (x28 + x14 + multi_dot([x30,self.ubar_rbs_l7_jcs_j10]) + (-1) * multi_dot([x3,self.ubar_ground_jcs_j10])),
        multi_dot([self.Mbar_rbs_l7_jcs_j10[:,0:1].T,x31,x3,x32]),
        multi_dot([self.Mbar_rbs_l7_jcs_j10[:,1:2].T,x31,x3,x32]),
        (x28 + (-1) * x33 + multi_dot([x30,self.ubar_rbs_l7_jcs_j11]) + (-1) * multi_dot([x35,self.ubar_rbs_l8_jcs_j11])),
        (x33 + (-1) * x36 + multi_dot([x35,self.ubar_rbs_l8_jcs_j12]) + (-1) * multi_dot([x38,self.ubar_rbs_l9_jcs_j12])),
        (x36 + x14 + multi_dot([x38,self.ubar_rbs_l9_jcs_j13]) + (-1) * multi_dot([x3,self.ubar_ground_jcs_j13])),
        multi_dot([self.Mbar_rbs_l9_jcs_j13[:,0:1].T,x39,x3,x40]),
        multi_dot([self.Mbar_rbs_l9_jcs_j13[:,1:2].T,x39,x3,x40]),
        x0,
        (x2 + (-1) * self.Pg_ground),
        (x41 + multi_dot([x4.T,x4])),
        (x41 + multi_dot([x9.T,x9])),
        (x41 + multi_dot([x12.T,x12])),
        (x41 + multi_dot([x18.T,x18])),
        (x41 + multi_dot([x21.T,x21])),
        (x41 + multi_dot([x26.T,x26])),
        (x41 + multi_dot([x29.T,x29])),
        (x41 + multi_dot([x34.T,x34])),
        (x41 + multi_dot([x37.T,x37])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z3x1
        v1 = Z1x1

        self.vel_eq_blocks = (v0,
        v1,
        v1,
        v0,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v0,
        v1,
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
        a1 = self.Pd_rbs_l1
        a2 = self.Mbar_ground_jcs_j1[:,0:1]
        a3 = self.P_ground
        a4 = A(a3).T
        a5 = self.Mbar_rbs_l1_jcs_j1[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_rbs_l1
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_ground_jcs_j1[:,1:2]
        a13 = self.Pd_rbs_l2
        a14 = self.Pd_rbs_l3
        a15 = self.Mbar_ground_jcs_j4[:,2:3]
        a16 = a15.T
        a17 = self.Mbar_rbs_l3_jcs_j4[:,0:1]
        a18 = self.P_rbs_l3
        a19 = A(a18).T
        a20 = B(a0,a15)
        a21 = a14.T
        a22 = B(a3,a15)
        a23 = self.Mbar_rbs_l3_jcs_j4[:,1:2]
        a24 = self.Pd_rbs_l4
        a25 = self.Pd_rbs_l5
        a26 = self.Mbar_ground_jcs_j7[:,2:3]
        a27 = a26.T
        a28 = self.Mbar_rbs_l5_jcs_j7[:,0:1]
        a29 = self.P_rbs_l5
        a30 = A(a29).T
        a31 = B(a0,a26)
        a32 = a25.T
        a33 = B(a3,a26)
        a34 = self.Mbar_rbs_l5_jcs_j7[:,1:2]
        a35 = self.Pd_rbs_l6
        a36 = self.Pd_rbs_l7
        a37 = self.Mbar_ground_jcs_j10[:,2:3]
        a38 = a37.T
        a39 = self.Mbar_rbs_l7_jcs_j10[:,0:1]
        a40 = self.P_rbs_l7
        a41 = A(a40).T
        a42 = B(a0,a37)
        a43 = a36.T
        a44 = B(a3,a37)
        a45 = self.Mbar_rbs_l7_jcs_j10[:,1:2]
        a46 = self.Pd_rbs_l8
        a47 = self.Pd_rbs_l9
        a48 = self.Mbar_ground_jcs_j13[:,2:3]
        a49 = a48.T
        a50 = self.Mbar_rbs_l9_jcs_j13[:,0:1]
        a51 = self.P_rbs_l9
        a52 = A(a51).T
        a53 = B(a0,a48)
        a54 = a47.T
        a55 = B(a3,a48)
        a56 = self.Mbar_rbs_l9_jcs_j13[:,1:2]

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_ground_jcs_j1),a0]) + (-1) * multi_dot([B(a1,self.ubar_rbs_l1_jcs_j1),a1])),
        (multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + (2) * multi_dot([a10,B(a3,a2).T,a11,a1])),
        (multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + (2) * multi_dot([a10,B(a3,a12).T,a11,a1])),
        (multi_dot([B(a1,self.ubar_rbs_l1_jcs_j2),a1]) + (-1) * multi_dot([B(a13,self.ubar_rbs_l2_jcs_j2),a13])),
        (multi_dot([B(a13,self.ubar_rbs_l2_jcs_j3),a13]) + (-1) * multi_dot([B(a14,self.ubar_rbs_l3_jcs_j3),a14])),
        (multi_dot([B(a14,self.ubar_rbs_l3_jcs_j4),a14]) + (-1) * multi_dot([B(a0,self.ubar_ground_jcs_j4),a0])),
        (multi_dot([a16,a4,B(a14,a17),a14]) + multi_dot([a17.T,a19,a20,a0]) + (2) * multi_dot([a21,B(a18,a17).T,a22,a0])),
        (multi_dot([a16,a4,B(a14,a23),a14]) + multi_dot([a23.T,a19,a20,a0]) + (2) * multi_dot([a21,B(a18,a23).T,a22,a0])),
        (multi_dot([B(a14,self.ubar_rbs_l3_jcs_j5),a14]) + (-1) * multi_dot([B(a24,self.ubar_rbs_l4_jcs_j5),a24])),
        (multi_dot([B(a24,self.ubar_rbs_l4_jcs_j6),a24]) + (-1) * multi_dot([B(a25,self.ubar_rbs_l5_jcs_j6),a25])),
        (multi_dot([B(a25,self.ubar_rbs_l5_jcs_j7),a25]) + (-1) * multi_dot([B(a0,self.ubar_ground_jcs_j7),a0])),
        (multi_dot([a27,a4,B(a25,a28),a25]) + multi_dot([a28.T,a30,a31,a0]) + (2) * multi_dot([a32,B(a29,a28).T,a33,a0])),
        (multi_dot([a27,a4,B(a25,a34),a25]) + multi_dot([a34.T,a30,a31,a0]) + (2) * multi_dot([a32,B(a29,a34).T,a33,a0])),
        (multi_dot([B(a25,self.ubar_rbs_l5_jcs_j8),a25]) + (-1) * multi_dot([B(a35,self.ubar_rbs_l6_jcs_j8),a35])),
        (multi_dot([B(a35,self.ubar_rbs_l6_jcs_j9),a35]) + (-1) * multi_dot([B(a36,self.ubar_rbs_l7_jcs_j9),a36])),
        (multi_dot([B(a36,self.ubar_rbs_l7_jcs_j10),a36]) + (-1) * multi_dot([B(a0,self.ubar_ground_jcs_j10),a0])),
        (multi_dot([a38,a4,B(a36,a39),a36]) + multi_dot([a39.T,a41,a42,a0]) + (2) * multi_dot([a43,B(a40,a39).T,a44,a0])),
        (multi_dot([a38,a4,B(a36,a45),a36]) + multi_dot([a45.T,a41,a42,a0]) + (2) * multi_dot([a43,B(a40,a45).T,a44,a0])),
        (multi_dot([B(a36,self.ubar_rbs_l7_jcs_j11),a36]) + (-1) * multi_dot([B(a46,self.ubar_rbs_l8_jcs_j11),a46])),
        (multi_dot([B(a46,self.ubar_rbs_l8_jcs_j12),a46]) + (-1) * multi_dot([B(a47,self.ubar_rbs_l9_jcs_j12),a47])),
        (multi_dot([B(a47,self.ubar_rbs_l9_jcs_j13),a47]) + (-1) * multi_dot([B(a0,self.ubar_ground_jcs_j13),a0])),
        (multi_dot([a49,a4,B(a47,a50),a47]) + multi_dot([a50.T,a52,a53,a0]) + (2) * multi_dot([a54,B(a51,a50).T,a55,a0])),
        (multi_dot([a49,a4,B(a47,a56),a47]) + multi_dot([a56.T,a52,a53,a0]) + (2) * multi_dot([a54,B(a51,a56).T,a55,a0])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a1.T,a1]),
        (2) * multi_dot([a13.T,a13]),
        (2) * multi_dot([a21,a14]),
        (2) * multi_dot([a24.T,a24]),
        (2) * multi_dot([a32,a25]),
        (2) * multi_dot([a35.T,a35]),
        (2) * multi_dot([a43,a36]),
        (2) * multi_dot([a46.T,a46]),
        (2) * multi_dot([a54,a47]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = I3
        j1 = self.P_ground
        j2 = Z1x3
        j3 = self.Mbar_rbs_l1_jcs_j1[:,2:3]
        j4 = j3.T
        j5 = self.P_rbs_l1
        j6 = A(j5).T
        j7 = self.Mbar_ground_jcs_j1[:,0:1]
        j8 = self.Mbar_ground_jcs_j1[:,1:2]
        j9 = (-1) * j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.P_rbs_l2
        j13 = self.P_rbs_l3
        j14 = self.Mbar_ground_jcs_j4[:,2:3]
        j15 = j14.T
        j16 = self.Mbar_rbs_l3_jcs_j4[:,0:1]
        j17 = self.Mbar_rbs_l3_jcs_j4[:,1:2]
        j18 = A(j13).T
        j19 = B(j1,j14)
        j20 = self.P_rbs_l4
        j21 = self.P_rbs_l5
        j22 = self.Mbar_ground_jcs_j7[:,2:3]
        j23 = j22.T
        j24 = self.Mbar_rbs_l5_jcs_j7[:,0:1]
        j25 = self.Mbar_rbs_l5_jcs_j7[:,1:2]
        j26 = A(j21).T
        j27 = B(j1,j22)
        j28 = self.P_rbs_l6
        j29 = self.P_rbs_l7
        j30 = self.Mbar_ground_jcs_j10[:,2:3]
        j31 = j30.T
        j32 = self.Mbar_rbs_l7_jcs_j10[:,0:1]
        j33 = self.Mbar_rbs_l7_jcs_j10[:,1:2]
        j34 = A(j29).T
        j35 = B(j1,j30)
        j36 = self.P_rbs_l8
        j37 = self.P_rbs_l9
        j38 = self.Mbar_ground_jcs_j13[:,2:3]
        j39 = j38.T
        j40 = self.Mbar_rbs_l9_jcs_j13[:,0:1]
        j41 = self.Mbar_rbs_l9_jcs_j13[:,1:2]
        j42 = A(j37).T
        j43 = B(j1,j38)

        self.jac_eq_blocks = (j0,
        B(j1,self.ubar_ground_jcs_j1),
        j9,
        (-1) * B(j5,self.ubar_rbs_l1_jcs_j1),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j7.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j2,
        multi_dot([j8.T,j10,j11]),
        j0,
        B(j5,self.ubar_rbs_l1_jcs_j2),
        j9,
        (-1) * B(j12,self.ubar_rbs_l2_jcs_j2),
        j0,
        B(j12,self.ubar_rbs_l2_jcs_j3),
        j9,
        (-1) * B(j13,self.ubar_rbs_l3_jcs_j3),
        j9,
        (-1) * B(j1,self.ubar_ground_jcs_j4),
        j0,
        B(j13,self.ubar_rbs_l3_jcs_j4),
        j2,
        multi_dot([j16.T,j18,j19]),
        j2,
        multi_dot([j15,j10,B(j13,j16)]),
        j2,
        multi_dot([j17.T,j18,j19]),
        j2,
        multi_dot([j15,j10,B(j13,j17)]),
        j0,
        B(j13,self.ubar_rbs_l3_jcs_j5),
        j9,
        (-1) * B(j20,self.ubar_rbs_l4_jcs_j5),
        j0,
        B(j20,self.ubar_rbs_l4_jcs_j6),
        j9,
        (-1) * B(j21,self.ubar_rbs_l5_jcs_j6),
        j9,
        (-1) * B(j1,self.ubar_ground_jcs_j7),
        j0,
        B(j21,self.ubar_rbs_l5_jcs_j7),
        j2,
        multi_dot([j24.T,j26,j27]),
        j2,
        multi_dot([j23,j10,B(j21,j24)]),
        j2,
        multi_dot([j25.T,j26,j27]),
        j2,
        multi_dot([j23,j10,B(j21,j25)]),
        j0,
        B(j21,self.ubar_rbs_l5_jcs_j8),
        j9,
        (-1) * B(j28,self.ubar_rbs_l6_jcs_j8),
        j0,
        B(j28,self.ubar_rbs_l6_jcs_j9),
        j9,
        (-1) * B(j29,self.ubar_rbs_l7_jcs_j9),
        j9,
        (-1) * B(j1,self.ubar_ground_jcs_j10),
        j0,
        B(j29,self.ubar_rbs_l7_jcs_j10),
        j2,
        multi_dot([j32.T,j34,j35]),
        j2,
        multi_dot([j31,j10,B(j29,j32)]),
        j2,
        multi_dot([j33.T,j34,j35]),
        j2,
        multi_dot([j31,j10,B(j29,j33)]),
        j0,
        B(j29,self.ubar_rbs_l7_jcs_j11),
        j9,
        (-1) * B(j36,self.ubar_rbs_l8_jcs_j11),
        j0,
        B(j36,self.ubar_rbs_l8_jcs_j12),
        j9,
        (-1) * B(j37,self.ubar_rbs_l9_jcs_j12),
        j9,
        (-1) * B(j1,self.ubar_ground_jcs_j13),
        j0,
        B(j37,self.ubar_rbs_l9_jcs_j13),
        j2,
        multi_dot([j40.T,j42,j43]),
        j2,
        multi_dot([j39,j10,B(j37,j40)]),
        j2,
        multi_dot([j41.T,j42,j43]),
        j2,
        multi_dot([j39,j10,B(j37,j41)]),
        j0,
        Z3x4,
        Z4x3,
        I4,
        j2,
        (2) * j5.T,
        j2,
        (2) * j12.T,
        j2,
        (2) * j13.T,
        j2,
        (2) * j20.T,
        j2,
        (2) * j21.T,
        j2,
        (2) * j28.T,
        j2,
        (2) * j29.T,
        j2,
        (2) * j36.T,
        j2,
        (2) * j37.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_ground)
        m2 = G(self.P_rbs_l1)
        m3 = G(self.P_rbs_l2)
        m4 = G(self.P_rbs_l3)
        m5 = G(self.P_rbs_l4)
        m6 = G(self.P_rbs_l5)
        m7 = G(self.P_rbs_l6)
        m8 = G(self.P_rbs_l7)
        m9 = G(self.P_rbs_l8)
        m10 = G(self.P_rbs_l9)

        self.mass_eq_blocks = (self.m_ground * m0,
        (4) * multi_dot([m1.T,self.Jbar_ground,m1]),
        config.m_rbs_l1 * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbs_l1,m2]),
        config.m_rbs_l2 * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbs_l2,m3]),
        config.m_rbs_l3 * m0,
        (4) * multi_dot([m4.T,config.Jbar_rbs_l3,m4]),
        config.m_rbs_l4 * m0,
        (4) * multi_dot([m5.T,config.Jbar_rbs_l4,m5]),
        config.m_rbs_l5 * m0,
        (4) * multi_dot([m6.T,config.Jbar_rbs_l5,m6]),
        config.m_rbs_l6 * m0,
        (4) * multi_dot([m7.T,config.Jbar_rbs_l6,m7]),
        config.m_rbs_l7 * m0,
        (4) * multi_dot([m8.T,config.Jbar_rbs_l7,m8]),
        config.m_rbs_l8 * m0,
        (4) * multi_dot([m9.T,config.Jbar_rbs_l8,m9]),
        config.m_rbs_l9 * m0,
        (4) * multi_dot([m10.T,config.Jbar_rbs_l9,m10]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_l1)
        f1 = G(self.Pd_rbs_l2)
        f2 = G(self.Pd_rbs_l3)
        f3 = G(self.Pd_rbs_l4)
        f4 = G(self.Pd_rbs_l5)
        f5 = G(self.Pd_rbs_l6)
        f6 = G(self.Pd_rbs_l7)
        f7 = G(self.Pd_rbs_l8)
        f8 = G(self.Pd_rbs_l9)

        self.frc_eq_blocks = (Z3x1,
        Z4x1,
        self.F_rbs_l1_gravity,
        (8) * multi_dot([f0.T,config.Jbar_rbs_l1,f0,self.P_rbs_l1]),
        self.F_rbs_l2_gravity,
        (8) * multi_dot([f1.T,config.Jbar_rbs_l2,f1,self.P_rbs_l2]),
        self.F_rbs_l3_gravity,
        (8) * multi_dot([f2.T,config.Jbar_rbs_l3,f2,self.P_rbs_l3]),
        self.F_rbs_l4_gravity,
        (8) * multi_dot([f3.T,config.Jbar_rbs_l4,f3,self.P_rbs_l4]),
        self.F_rbs_l5_gravity,
        (8) * multi_dot([f4.T,config.Jbar_rbs_l5,f4,self.P_rbs_l5]),
        self.F_rbs_l6_gravity,
        (8) * multi_dot([f5.T,config.Jbar_rbs_l6,f5,self.P_rbs_l6]),
        self.F_rbs_l7_gravity,
        (8) * multi_dot([f6.T,config.Jbar_rbs_l7,f6,self.P_rbs_l7]),
        self.F_rbs_l8_gravity,
        (8) * multi_dot([f7.T,config.Jbar_rbs_l8,f7,self.P_rbs_l8]),
        self.F_rbs_l9_gravity,
        (8) * multi_dot([f8.T,config.Jbar_rbs_l9,f8,self.P_rbs_l9]),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_ground_jcs_j1 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_ground,self.ubar_ground_jcs_j1).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_j1[:,0:1]).T,A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_j1[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_j1[:,1:2]).T,A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_j1[:,2:3]])]]),self.L_jcs_j1])
        self.F_ground_jcs_j1 = Q_ground_jcs_j1[0:3]
        Te_ground_jcs_j1 = Q_ground_jcs_j1[3:7]
        self.T_ground_jcs_j1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_j1])),self.F_ground_jcs_j1]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_j1]))
        Q_rbs_l1_jcs_j2 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l1,self.ubar_rbs_l1_jcs_j2).T]]),self.L_jcs_j2])
        self.F_rbs_l1_jcs_j2 = Q_rbs_l1_jcs_j2[0:3]
        Te_rbs_l1_jcs_j2 = Q_rbs_l1_jcs_j2[3:7]
        self.T_rbs_l1_jcs_j2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l1),self.ubar_rbs_l1_jcs_j2])),self.F_rbs_l1_jcs_j2]) + (0.5) * multi_dot([E(self.P_rbs_l1),Te_rbs_l1_jcs_j2]))
        Q_rbs_l2_jcs_j3 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l2,self.ubar_rbs_l2_jcs_j3).T]]),self.L_jcs_j3])
        self.F_rbs_l2_jcs_j3 = Q_rbs_l2_jcs_j3[0:3]
        Te_rbs_l2_jcs_j3 = Q_rbs_l2_jcs_j3[3:7]
        self.T_rbs_l2_jcs_j3 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l2),self.ubar_rbs_l2_jcs_j3])),self.F_rbs_l2_jcs_j3]) + (0.5) * multi_dot([E(self.P_rbs_l2),Te_rbs_l2_jcs_j3]))
        Q_rbs_l3_jcs_j4 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_l3,self.ubar_rbs_l3_jcs_j4).T,multi_dot([B(self.P_rbs_l3,self.Mbar_rbs_l3_jcs_j4[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_j4[:,2:3]]),multi_dot([B(self.P_rbs_l3,self.Mbar_rbs_l3_jcs_j4[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcs_j4[:,2:3]])]]),self.L_jcs_j4])
        self.F_rbs_l3_jcs_j4 = Q_rbs_l3_jcs_j4[0:3]
        Te_rbs_l3_jcs_j4 = Q_rbs_l3_jcs_j4[3:7]
        self.T_rbs_l3_jcs_j4 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l3),self.ubar_rbs_l3_jcs_j4])),self.F_rbs_l3_jcs_j4]) + (0.5) * multi_dot([E(self.P_rbs_l3),Te_rbs_l3_jcs_j4]))
        Q_rbs_l3_jcs_j5 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l3,self.ubar_rbs_l3_jcs_j5).T]]),self.L_jcs_j5])
        self.F_rbs_l3_jcs_j5 = Q_rbs_l3_jcs_j5[0:3]
        Te_rbs_l3_jcs_j5 = Q_rbs_l3_jcs_j5[3:7]
        self.T_rbs_l3_jcs_j5 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l3),self.ubar_rbs_l3_jcs_j5])),self.F_rbs_l3_jcs_j5]) + (0.5) * multi_dot([E(self.P_rbs_l3),Te_rbs_l3_jcs_j5]))
        Q_rbs_l4_jcs_j6 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l4,self.ubar_rbs_l4_jcs_j6).T]]),self.L_jcs_j6])
        self.F_rbs_l4_jcs_j6 = Q_rbs_l4_jcs_j6[0:3]
        Te_rbs_l4_jcs_j6 = Q_rbs_l4_jcs_j6[3:7]
        self.T_rbs_l4_jcs_j6 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l4),self.ubar_rbs_l4_jcs_j6])),self.F_rbs_l4_jcs_j6]) + (0.5) * multi_dot([E(self.P_rbs_l4),Te_rbs_l4_jcs_j6]))
        Q_rbs_l5_jcs_j7 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_l5,self.ubar_rbs_l5_jcs_j7).T,multi_dot([B(self.P_rbs_l5,self.Mbar_rbs_l5_jcs_j7[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_j7[:,2:3]]),multi_dot([B(self.P_rbs_l5,self.Mbar_rbs_l5_jcs_j7[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcs_j7[:,2:3]])]]),self.L_jcs_j7])
        self.F_rbs_l5_jcs_j7 = Q_rbs_l5_jcs_j7[0:3]
        Te_rbs_l5_jcs_j7 = Q_rbs_l5_jcs_j7[3:7]
        self.T_rbs_l5_jcs_j7 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l5),self.ubar_rbs_l5_jcs_j7])),self.F_rbs_l5_jcs_j7]) + (0.5) * multi_dot([E(self.P_rbs_l5),Te_rbs_l5_jcs_j7]))
        Q_rbs_l5_jcs_j8 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l5,self.ubar_rbs_l5_jcs_j8).T]]),self.L_jcs_j8])
        self.F_rbs_l5_jcs_j8 = Q_rbs_l5_jcs_j8[0:3]
        Te_rbs_l5_jcs_j8 = Q_rbs_l5_jcs_j8[3:7]
        self.T_rbs_l5_jcs_j8 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l5),self.ubar_rbs_l5_jcs_j8])),self.F_rbs_l5_jcs_j8]) + (0.5) * multi_dot([E(self.P_rbs_l5),Te_rbs_l5_jcs_j8]))
        Q_rbs_l6_jcs_j9 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l6,self.ubar_rbs_l6_jcs_j9).T]]),self.L_jcs_j9])
        self.F_rbs_l6_jcs_j9 = Q_rbs_l6_jcs_j9[0:3]
        Te_rbs_l6_jcs_j9 = Q_rbs_l6_jcs_j9[3:7]
        self.T_rbs_l6_jcs_j9 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l6),self.ubar_rbs_l6_jcs_j9])),self.F_rbs_l6_jcs_j9]) + (0.5) * multi_dot([E(self.P_rbs_l6),Te_rbs_l6_jcs_j9]))
        Q_rbs_l7_jcs_j10 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_l7,self.ubar_rbs_l7_jcs_j10).T,multi_dot([B(self.P_rbs_l7,self.Mbar_rbs_l7_jcs_j10[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_j10[:,2:3]]),multi_dot([B(self.P_rbs_l7,self.Mbar_rbs_l7_jcs_j10[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcs_j10[:,2:3]])]]),self.L_jcs_j10])
        self.F_rbs_l7_jcs_j10 = Q_rbs_l7_jcs_j10[0:3]
        Te_rbs_l7_jcs_j10 = Q_rbs_l7_jcs_j10[3:7]
        self.T_rbs_l7_jcs_j10 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_jcs_j10])),self.F_rbs_l7_jcs_j10]) + (0.5) * multi_dot([E(self.P_rbs_l7),Te_rbs_l7_jcs_j10]))
        Q_rbs_l7_jcs_j11 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l7,self.ubar_rbs_l7_jcs_j11).T]]),self.L_jcs_j11])
        self.F_rbs_l7_jcs_j11 = Q_rbs_l7_jcs_j11[0:3]
        Te_rbs_l7_jcs_j11 = Q_rbs_l7_jcs_j11[3:7]
        self.T_rbs_l7_jcs_j11 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l7),self.ubar_rbs_l7_jcs_j11])),self.F_rbs_l7_jcs_j11]) + (0.5) * multi_dot([E(self.P_rbs_l7),Te_rbs_l7_jcs_j11]))
        Q_rbs_l8_jcs_j12 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l8,self.ubar_rbs_l8_jcs_j12).T]]),self.L_jcs_j12])
        self.F_rbs_l8_jcs_j12 = Q_rbs_l8_jcs_j12[0:3]
        Te_rbs_l8_jcs_j12 = Q_rbs_l8_jcs_j12[3:7]
        self.T_rbs_l8_jcs_j12 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l8),self.ubar_rbs_l8_jcs_j12])),self.F_rbs_l8_jcs_j12]) + (0.5) * multi_dot([E(self.P_rbs_l8),Te_rbs_l8_jcs_j12]))
        Q_rbs_l9_jcs_j13 = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_l9,self.ubar_rbs_l9_jcs_j13).T,multi_dot([B(self.P_rbs_l9,self.Mbar_rbs_l9_jcs_j13[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_j13[:,2:3]]),multi_dot([B(self.P_rbs_l9,self.Mbar_rbs_l9_jcs_j13[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcs_j13[:,2:3]])]]),self.L_jcs_j13])
        self.F_rbs_l9_jcs_j13 = Q_rbs_l9_jcs_j13[0:3]
        Te_rbs_l9_jcs_j13 = Q_rbs_l9_jcs_j13[3:7]
        self.T_rbs_l9_jcs_j13 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l9),self.ubar_rbs_l9_jcs_j13])),self.F_rbs_l9_jcs_j13]) + (0.5) * multi_dot([E(self.P_rbs_l9),Te_rbs_l9_jcs_j13]))

        self.reactions = {'F_ground_jcs_j1' : self.F_ground_jcs_j1,
                        'T_ground_jcs_j1' : self.T_ground_jcs_j1,
                        'F_rbs_l1_jcs_j2' : self.F_rbs_l1_jcs_j2,
                        'T_rbs_l1_jcs_j2' : self.T_rbs_l1_jcs_j2,
                        'F_rbs_l2_jcs_j3' : self.F_rbs_l2_jcs_j3,
                        'T_rbs_l2_jcs_j3' : self.T_rbs_l2_jcs_j3,
                        'F_rbs_l3_jcs_j4' : self.F_rbs_l3_jcs_j4,
                        'T_rbs_l3_jcs_j4' : self.T_rbs_l3_jcs_j4,
                        'F_rbs_l3_jcs_j5' : self.F_rbs_l3_jcs_j5,
                        'T_rbs_l3_jcs_j5' : self.T_rbs_l3_jcs_j5,
                        'F_rbs_l4_jcs_j6' : self.F_rbs_l4_jcs_j6,
                        'T_rbs_l4_jcs_j6' : self.T_rbs_l4_jcs_j6,
                        'F_rbs_l5_jcs_j7' : self.F_rbs_l5_jcs_j7,
                        'T_rbs_l5_jcs_j7' : self.T_rbs_l5_jcs_j7,
                        'F_rbs_l5_jcs_j8' : self.F_rbs_l5_jcs_j8,
                        'T_rbs_l5_jcs_j8' : self.T_rbs_l5_jcs_j8,
                        'F_rbs_l6_jcs_j9' : self.F_rbs_l6_jcs_j9,
                        'T_rbs_l6_jcs_j9' : self.T_rbs_l6_jcs_j9,
                        'F_rbs_l7_jcs_j10' : self.F_rbs_l7_jcs_j10,
                        'T_rbs_l7_jcs_j10' : self.T_rbs_l7_jcs_j10,
                        'F_rbs_l7_jcs_j11' : self.F_rbs_l7_jcs_j11,
                        'T_rbs_l7_jcs_j11' : self.T_rbs_l7_jcs_j11,
                        'F_rbs_l8_jcs_j12' : self.F_rbs_l8_jcs_j12,
                        'T_rbs_l8_jcs_j12' : self.T_rbs_l8_jcs_j12,
                        'F_rbs_l9_jcs_j13' : self.F_rbs_l9_jcs_j13,
                        'T_rbs_l9_jcs_j13' : self.T_rbs_l9_jcs_j13}

