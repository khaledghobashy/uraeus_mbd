
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

        self.indicies_map = {'ground': 0, 'rbs_l1': 1, 'rbs_l2': 2, 'rbs_l3': 3, 'rbs_l4': 4, 'rbs_l5': 5}

        self.n  = 42
        self.nc = 41
        self.nrows = 28
        self.ncols = 2*6
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_ground_jcs_a', 'T_ground_jcs_a', 'F_ground_mcs_act', 'T_ground_mcs_act', 'F_ground_jcs_b', 'T_ground_jcs_b', 'F_rbs_l1_jcs_c', 'T_rbs_l1_jcs_c', 'F_rbs_l2_jcs_d', 'T_rbs_l2_jcs_d', 'F_rbs_l3_jcs_e', 'T_rbs_l3_jcs_e', 'F_rbs_l4_jcs_f', 'T_rbs_l4_jcs_f', 'F_rbs_l5_jcs_h', 'T_rbs_l5_jcs_h']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.ground*2, self.ground*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.ground*2, self.ground*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l5*2, self.rbs_l5*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l4*2, self.rbs_l4*2+1, self.rbs_l5*2, self.rbs_l5*2+1], dtype=np.intc)

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
        self.config.P_rbs_l5], out=self._q)

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
        self.config.Pd_rbs_l5], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p + 'ground']
        self.rbs_l1 = indicies_map[p + 'rbs_l1']
        self.rbs_l2 = indicies_map[p + 'rbs_l2']
        self.rbs_l3 = indicies_map[p + 'rbs_l3']
        self.rbs_l4 = indicies_map[p + 'rbs_l4']
        self.rbs_l5 = indicies_map[p + 'rbs_l5']
    

    
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

        self.Mbar_ground_jcs_a = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_l1_jcs_a = multi_dot([A(config.P_rbs_l1).T,triad(config.ax1_jcs_a)])
        self.ubar_ground_jcs_a = (multi_dot([A(self.P_ground).T,config.pt1_jcs_a]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_l1_jcs_a = (multi_dot([A(config.P_rbs_l1).T,config.pt1_jcs_a]) + (-1) * multi_dot([A(config.P_rbs_l1).T,config.R_rbs_l1]))
        self.Mbar_ground_jcs_a = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_l1_jcs_a = multi_dot([A(config.P_rbs_l1).T,triad(config.ax1_jcs_a)])
        self.Mbar_ground_jcs_b = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_b)])
        self.Mbar_rbs_l2_jcs_b = multi_dot([A(config.P_rbs_l2).T,triad(config.ax1_jcs_b)])
        self.ubar_ground_jcs_b = (multi_dot([A(self.P_ground).T,config.pt1_jcs_b]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_l2_jcs_b = (multi_dot([A(config.P_rbs_l2).T,config.pt1_jcs_b]) + (-1) * multi_dot([A(config.P_rbs_l2).T,config.R_rbs_l2]))
        self.Mbar_rbs_l1_jcs_c = multi_dot([A(config.P_rbs_l1).T,triad(config.ax1_jcs_c)])
        self.Mbar_rbs_l3_jcs_c = multi_dot([A(config.P_rbs_l3).T,triad(config.ax1_jcs_c)])
        self.ubar_rbs_l1_jcs_c = (multi_dot([A(config.P_rbs_l1).T,config.pt1_jcs_c]) + (-1) * multi_dot([A(config.P_rbs_l1).T,config.R_rbs_l1]))
        self.ubar_rbs_l3_jcs_c = (multi_dot([A(config.P_rbs_l3).T,config.pt1_jcs_c]) + (-1) * multi_dot([A(config.P_rbs_l3).T,config.R_rbs_l3]))
        self.Mbar_rbs_l2_jcs_d = multi_dot([A(config.P_rbs_l2).T,triad(config.ax1_jcs_d)])
        self.Mbar_rbs_l3_jcs_d = multi_dot([A(config.P_rbs_l3).T,triad(config.ax1_jcs_d)])
        self.ubar_rbs_l2_jcs_d = (multi_dot([A(config.P_rbs_l2).T,config.pt1_jcs_d]) + (-1) * multi_dot([A(config.P_rbs_l2).T,config.R_rbs_l2]))
        self.ubar_rbs_l3_jcs_d = (multi_dot([A(config.P_rbs_l3).T,config.pt1_jcs_d]) + (-1) * multi_dot([A(config.P_rbs_l3).T,config.R_rbs_l3]))
        self.Mbar_rbs_l3_jcs_e = multi_dot([A(config.P_rbs_l3).T,triad(config.ax1_jcs_e)])
        self.Mbar_rbs_l4_jcs_e = multi_dot([A(config.P_rbs_l4).T,triad(config.ax1_jcs_e)])
        self.ubar_rbs_l3_jcs_e = (multi_dot([A(config.P_rbs_l3).T,config.pt1_jcs_e]) + (-1) * multi_dot([A(config.P_rbs_l3).T,config.R_rbs_l3]))
        self.ubar_rbs_l4_jcs_e = (multi_dot([A(config.P_rbs_l4).T,config.pt1_jcs_e]) + (-1) * multi_dot([A(config.P_rbs_l4).T,config.R_rbs_l4]))
        self.Mbar_rbs_l4_jcs_f = multi_dot([A(config.P_rbs_l4).T,triad(config.ax1_jcs_f)])
        self.Mbar_rbs_l5_jcs_f = multi_dot([A(config.P_rbs_l5).T,triad(config.ax1_jcs_f)])
        self.ubar_rbs_l4_jcs_f = (multi_dot([A(config.P_rbs_l4).T,config.pt1_jcs_f]) + (-1) * multi_dot([A(config.P_rbs_l4).T,config.R_rbs_l4]))
        self.ubar_rbs_l5_jcs_f = (multi_dot([A(config.P_rbs_l5).T,config.pt1_jcs_f]) + (-1) * multi_dot([A(config.P_rbs_l5).T,config.R_rbs_l5]))
        self.Mbar_rbs_l5_jcs_h = multi_dot([A(config.P_rbs_l5).T,triad(config.ax1_jcs_h)])
        self.Mbar_rbs_l2_jcs_h = multi_dot([A(config.P_rbs_l2).T,triad(config.ax1_jcs_h)])
        self.ubar_rbs_l5_jcs_h = (multi_dot([A(config.P_rbs_l5).T,config.pt1_jcs_h]) + (-1) * multi_dot([A(config.P_rbs_l5).T,config.R_rbs_l5]))
        self.ubar_rbs_l2_jcs_h = (multi_dot([A(config.P_rbs_l2).T,config.pt1_jcs_h]) + (-1) * multi_dot([A(config.P_rbs_l2).T,config.R_rbs_l2]))

    
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

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcs_a = Lambda[0:5]
        self.L_mcs_act = Lambda[5:6]
        self.L_jcs_b = Lambda[6:11]
        self.L_jcs_c = Lambda[11:15]
        self.L_jcs_d = Lambda[15:18]
        self.L_jcs_e = Lambda[18:22]
        self.L_jcs_f = Lambda[22:25]
        self.L_jcs_h = Lambda[25:29]

    
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
        x7 = self.Mbar_rbs_l1_jcs_a[:,2:3]
        x8 = self.Mbar_rbs_l1_jcs_a[:,0:1]
        x9 = self.R_rbs_l2
        x10 = (-1) * x9
        x11 = self.P_rbs_l2
        x12 = A(x11)
        x13 = self.Mbar_rbs_l2_jcs_b[:,2:3]
        x14 = self.Mbar_rbs_l1_jcs_c[:,0:1].T
        x15 = x5.T
        x16 = self.P_rbs_l3
        x17 = A(x16)
        x18 = self.Mbar_rbs_l3_jcs_c[:,2:3]
        x19 = self.Mbar_rbs_l1_jcs_c[:,1:2].T
        x20 = self.R_rbs_l3
        x21 = (-1) * x20
        x22 = (x1 + x21 + multi_dot([x5,self.ubar_rbs_l1_jcs_c]) + (-1) * multi_dot([x17,self.ubar_rbs_l3_jcs_c]))
        x23 = self.Mbar_rbs_l3_jcs_e[:,0:1].T
        x24 = x17.T
        x25 = self.P_rbs_l4
        x26 = A(x25)
        x27 = self.Mbar_rbs_l4_jcs_e[:,2:3]
        x28 = self.Mbar_rbs_l3_jcs_e[:,1:2].T
        x29 = self.R_rbs_l4
        x30 = (x20 + (-1) * x29 + multi_dot([x17,self.ubar_rbs_l3_jcs_e]) + (-1) * multi_dot([x26,self.ubar_rbs_l4_jcs_e]))
        x31 = self.R_rbs_l5
        x32 = self.P_rbs_l5
        x33 = A(x32)
        x34 = self.Mbar_rbs_l5_jcs_h[:,0:1].T
        x35 = x33.T
        x36 = self.Mbar_rbs_l2_jcs_h[:,2:3]
        x37 = self.Mbar_rbs_l5_jcs_h[:,1:2].T
        x38 = (x31 + x10 + multi_dot([x33,self.ubar_rbs_l5_jcs_h]) + (-1) * multi_dot([x12,self.ubar_rbs_l2_jcs_h]))
        x39 = (-1) * I1

        self.pos_eq_blocks = ((x0 + (-1) * x1 + multi_dot([x3,self.ubar_ground_jcs_a]) + (-1) * multi_dot([x5,self.ubar_rbs_l1_jcs_a])),
        multi_dot([self.Mbar_ground_jcs_a[:,0:1].T,x6,x5,x7]),
        multi_dot([self.Mbar_ground_jcs_a[:,1:2].T,x6,x5,x7]),
        (cos(config.UF_mcs_act(t)) * multi_dot([self.Mbar_ground_jcs_a[:,1:2].T,x6,x5,x8]) + (-1 * sin(config.UF_mcs_act(t))) * multi_dot([self.Mbar_ground_jcs_a[:,0:1].T,x6,x5,x8])),
        (x0 + x10 + multi_dot([x3,self.ubar_ground_jcs_b]) + (-1) * multi_dot([x12,self.ubar_rbs_l2_jcs_b])),
        multi_dot([self.Mbar_ground_jcs_b[:,0:1].T,x6,x12,x13]),
        multi_dot([self.Mbar_ground_jcs_b[:,1:2].T,x6,x12,x13]),
        multi_dot([x14,x15,x17,x18]),
        multi_dot([x19,x15,x17,x18]),
        multi_dot([x14,x15,x22]),
        multi_dot([x19,x15,x22]),
        (x9 + x21 + multi_dot([x12,self.ubar_rbs_l2_jcs_d]) + (-1) * multi_dot([x17,self.ubar_rbs_l3_jcs_d])),
        multi_dot([x23,x24,x26,x27]),
        multi_dot([x28,x24,x26,x27]),
        multi_dot([x23,x24,x30]),
        multi_dot([x28,x24,x30]),
        (x29 + (-1) * x31 + multi_dot([x26,self.ubar_rbs_l4_jcs_f]) + (-1) * multi_dot([x33,self.ubar_rbs_l5_jcs_f])),
        multi_dot([x34,x35,x12,x36]),
        multi_dot([x37,x35,x12,x36]),
        multi_dot([x34,x35,x38]),
        multi_dot([x37,x35,x38]),
        x0,
        (x2 + (-1) * self.Pg_ground),
        (x39 + multi_dot([x4.T,x4])),
        (x39 + multi_dot([x11.T,x11])),
        (x39 + multi_dot([x16.T,x16])),
        (x39 + multi_dot([x25.T,x25])),
        (x39 + multi_dot([x32.T,x32])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z3x1
        v1 = Z1x1

        self.vel_eq_blocks = (v0,
        v1,
        v1,
        (v1 + (-1 * derivative(config.UF_mcs_act, t, 0.1, 1)) * I1),
        v0,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v0,
        v1,
        v1,
        v1,
        v1,
        v0,
        v1,
        v1,
        v1,
        v1,
        v0,
        Z4x1,
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
        a2 = self.Mbar_ground_jcs_a[:,0:1]
        a3 = self.P_ground
        a4 = A(a3).T
        a5 = self.Mbar_rbs_l1_jcs_a[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_rbs_l1
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_ground_jcs_a[:,1:2]
        a13 = self.Mbar_rbs_l1_jcs_a[:,0:1]
        a14 = self.Mbar_ground_jcs_a[:,1:2]
        a15 = self.Mbar_ground_jcs_a[:,0:1]
        a16 = self.Pd_rbs_l2
        a17 = self.Mbar_ground_jcs_b[:,0:1]
        a18 = self.Mbar_rbs_l2_jcs_b[:,2:3]
        a19 = B(a16,a18)
        a20 = a18.T
        a21 = self.P_rbs_l2
        a22 = A(a21).T
        a23 = B(a21,a18)
        a24 = self.Mbar_ground_jcs_b[:,1:2]
        a25 = self.Mbar_rbs_l1_jcs_c[:,0:1]
        a26 = a25.T
        a27 = self.Pd_rbs_l3
        a28 = self.Mbar_rbs_l3_jcs_c[:,2:3]
        a29 = B(a27,a28)
        a30 = a28.T
        a31 = self.P_rbs_l3
        a32 = A(a31).T
        a33 = B(a1,a25)
        a34 = a1.T
        a35 = B(a8,a25).T
        a36 = B(a31,a28)
        a37 = self.Mbar_rbs_l1_jcs_c[:,1:2]
        a38 = a37.T
        a39 = B(a1,a37)
        a40 = B(a8,a37).T
        a41 = self.ubar_rbs_l1_jcs_c
        a42 = self.ubar_rbs_l3_jcs_c
        a43 = (multi_dot([B(a1,a41),a1]) + (-1) * multi_dot([B(a27,a42),a27]))
        a44 = self.Rd_rbs_l3
        a45 = (self.Rd_rbs_l1 + (-1) * a44 + multi_dot([B(a8,a41),a1]) + (-1) * multi_dot([B(a31,a42),a27]))
        a46 = self.R_rbs_l3.T
        a47 = (self.R_rbs_l1.T + (-1) * a46 + multi_dot([a41.T,a9]) + (-1) * multi_dot([a42.T,a32]))
        a48 = self.Mbar_rbs_l4_jcs_e[:,2:3]
        a49 = a48.T
        a50 = self.P_rbs_l4
        a51 = A(a50).T
        a52 = self.Mbar_rbs_l3_jcs_e[:,0:1]
        a53 = B(a27,a52)
        a54 = a52.T
        a55 = self.Pd_rbs_l4
        a56 = B(a55,a48)
        a57 = a27.T
        a58 = B(a31,a52).T
        a59 = B(a50,a48)
        a60 = self.Mbar_rbs_l3_jcs_e[:,1:2]
        a61 = B(a27,a60)
        a62 = a60.T
        a63 = B(a31,a60).T
        a64 = self.ubar_rbs_l3_jcs_e
        a65 = self.ubar_rbs_l4_jcs_e
        a66 = (multi_dot([B(a27,a64),a27]) + (-1) * multi_dot([B(a55,a65),a55]))
        a67 = (a44 + (-1) * self.Rd_rbs_l4 + multi_dot([B(a31,a64),a27]) + (-1) * multi_dot([B(a50,a65),a55]))
        a68 = (a46 + (-1) * self.R_rbs_l4.T + multi_dot([a64.T,a32]) + (-1) * multi_dot([a65.T,a51]))
        a69 = self.Pd_rbs_l5
        a70 = self.Mbar_rbs_l2_jcs_h[:,2:3]
        a71 = a70.T
        a72 = self.Mbar_rbs_l5_jcs_h[:,0:1]
        a73 = B(a69,a72)
        a74 = a72.T
        a75 = self.P_rbs_l5
        a76 = A(a75).T
        a77 = B(a16,a70)
        a78 = a69.T
        a79 = B(a75,a72).T
        a80 = B(a21,a70)
        a81 = self.Mbar_rbs_l5_jcs_h[:,1:2]
        a82 = B(a69,a81)
        a83 = a81.T
        a84 = B(a75,a81).T
        a85 = self.ubar_rbs_l5_jcs_h
        a86 = self.ubar_rbs_l2_jcs_h
        a87 = (multi_dot([B(a69,a85),a69]) + (-1) * multi_dot([B(a16,a86),a16]))
        a88 = (self.Rd_rbs_l5 + (-1) * self.Rd_rbs_l2 + multi_dot([B(a75,a85),a69]) + (-1) * multi_dot([B(a21,a86),a16]))
        a89 = (self.R_rbs_l5.T + (-1) * self.R_rbs_l2.T + multi_dot([a85.T,a76]) + (-1) * multi_dot([a86.T,a22]))

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_ground_jcs_a),a0]) + (-1) * multi_dot([B(a1,self.ubar_rbs_l1_jcs_a),a1])),
        (multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + (2) * multi_dot([a10,B(a3,a2).T,a11,a1])),
        (multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + (2) * multi_dot([a10,B(a3,a12).T,a11,a1])),
        ((-1 * derivative(config.UF_mcs_act, t, 0.1, 2)) * I1 + multi_dot([a13.T,a9,(cos(config.UF_mcs_act(t)) * B(a0,a14) + (-1 * sin(config.UF_mcs_act(t))) * B(a0,a15)),a0]) + multi_dot([(cos(config.UF_mcs_act(t)) * multi_dot([a14.T,a4]) + (-1 * sin(config.UF_mcs_act(t))) * multi_dot([a15.T,a4])),B(a1,a13),a1]) + (2) * multi_dot([(cos(config.UF_mcs_act(t)) * multi_dot([a10,B(a3,a14).T]) + (-1 * sin(config.UF_mcs_act(t))) * multi_dot([a10,B(a3,a15).T])),B(a8,a13),a1])),
        (multi_dot([B(a0,self.ubar_ground_jcs_b),a0]) + (-1) * multi_dot([B(a16,self.ubar_rbs_l2_jcs_b),a16])),
        (multi_dot([a17.T,a4,a19,a16]) + multi_dot([a20,a22,B(a0,a17),a0]) + (2) * multi_dot([a10,B(a3,a17).T,a23,a16])),
        (multi_dot([a24.T,a4,a19,a16]) + multi_dot([a20,a22,B(a0,a24),a0]) + (2) * multi_dot([a10,B(a3,a24).T,a23,a16])),
        (multi_dot([a26,a9,a29,a27]) + multi_dot([a30,a32,a33,a1]) + (2) * multi_dot([a34,a35,a36,a27])),
        (multi_dot([a38,a9,a29,a27]) + multi_dot([a30,a32,a39,a1]) + (2) * multi_dot([a34,a40,a36,a27])),
        (multi_dot([a26,a9,a43]) + (2) * multi_dot([a34,a35,a45]) + multi_dot([a47,a33,a1])),
        (multi_dot([a38,a9,a43]) + (2) * multi_dot([a34,a40,a45]) + multi_dot([a47,a39,a1])),
        (multi_dot([B(a16,self.ubar_rbs_l2_jcs_d),a16]) + (-1) * multi_dot([B(a27,self.ubar_rbs_l3_jcs_d),a27])),
        (multi_dot([a49,a51,a53,a27]) + multi_dot([a54,a32,a56,a55]) + (2) * multi_dot([a57,a58,a59,a55])),
        (multi_dot([a49,a51,a61,a27]) + multi_dot([a62,a32,a56,a55]) + (2) * multi_dot([a57,a63,a59,a55])),
        (multi_dot([a54,a32,a66]) + (2) * multi_dot([a57,a58,a67]) + multi_dot([a68,a53,a27])),
        (multi_dot([a62,a32,a66]) + (2) * multi_dot([a57,a63,a67]) + multi_dot([a68,a61,a27])),
        (multi_dot([B(a55,self.ubar_rbs_l4_jcs_f),a55]) + (-1) * multi_dot([B(a69,self.ubar_rbs_l5_jcs_f),a69])),
        (multi_dot([a71,a22,a73,a69]) + multi_dot([a74,a76,a77,a16]) + (2) * multi_dot([a78,a79,a80,a16])),
        (multi_dot([a71,a22,a82,a69]) + multi_dot([a83,a76,a77,a16]) + (2) * multi_dot([a78,a84,a80,a16])),
        (multi_dot([a74,a76,a87]) + (2) * multi_dot([a78,a79,a88]) + multi_dot([a89,a73,a69])),
        (multi_dot([a83,a76,a87]) + (2) * multi_dot([a78,a84,a88]) + multi_dot([a89,a82,a69])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a34,a1]),
        (2) * multi_dot([a16.T,a16]),
        (2) * multi_dot([a57,a27]),
        (2) * multi_dot([a55.T,a55]),
        (2) * multi_dot([a78,a69]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = I3
        j1 = self.P_ground
        j2 = Z1x3
        j3 = self.Mbar_rbs_l1_jcs_a[:,2:3]
        j4 = j3.T
        j5 = self.P_rbs_l1
        j6 = A(j5).T
        j7 = self.Mbar_ground_jcs_a[:,0:1]
        j8 = self.Mbar_ground_jcs_a[:,1:2]
        j9 = (-1) * j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_rbs_l1_jcs_a[:,0:1]
        j13 = self.Mbar_ground_jcs_a[:,1:2]
        j14 = self.Mbar_ground_jcs_a[:,0:1]
        j15 = self.Mbar_rbs_l2_jcs_b[:,2:3]
        j16 = j15.T
        j17 = self.P_rbs_l2
        j18 = A(j17).T
        j19 = self.Mbar_ground_jcs_b[:,0:1]
        j20 = self.Mbar_ground_jcs_b[:,1:2]
        j21 = B(j17,j15)
        j22 = self.Mbar_rbs_l3_jcs_c[:,2:3]
        j23 = j22.T
        j24 = self.P_rbs_l3
        j25 = A(j24).T
        j26 = self.Mbar_rbs_l1_jcs_c[:,0:1]
        j27 = B(j5,j26)
        j28 = self.Mbar_rbs_l1_jcs_c[:,1:2]
        j29 = B(j5,j28)
        j30 = j26.T
        j31 = multi_dot([j30,j6])
        j32 = self.ubar_rbs_l1_jcs_c
        j33 = B(j5,j32)
        j34 = self.R_rbs_l3.T
        j35 = self.ubar_rbs_l3_jcs_c
        j36 = (self.R_rbs_l1.T + (-1) * j34 + multi_dot([j32.T,j6]) + (-1) * multi_dot([j35.T,j25]))
        j37 = j28.T
        j38 = multi_dot([j37,j6])
        j39 = B(j24,j22)
        j40 = B(j24,j35)
        j41 = self.Mbar_rbs_l4_jcs_e[:,2:3]
        j42 = j41.T
        j43 = self.P_rbs_l4
        j44 = A(j43).T
        j45 = self.Mbar_rbs_l3_jcs_e[:,0:1]
        j46 = B(j24,j45)
        j47 = self.Mbar_rbs_l3_jcs_e[:,1:2]
        j48 = B(j24,j47)
        j49 = j45.T
        j50 = multi_dot([j49,j25])
        j51 = self.ubar_rbs_l3_jcs_e
        j52 = B(j24,j51)
        j53 = self.ubar_rbs_l4_jcs_e
        j54 = (j34 + (-1) * self.R_rbs_l4.T + multi_dot([j51.T,j25]) + (-1) * multi_dot([j53.T,j44]))
        j55 = j47.T
        j56 = multi_dot([j55,j25])
        j57 = B(j43,j41)
        j58 = B(j43,j53)
        j59 = self.P_rbs_l5
        j60 = self.Mbar_rbs_l2_jcs_h[:,2:3]
        j61 = j60.T
        j62 = self.Mbar_rbs_l5_jcs_h[:,0:1]
        j63 = B(j59,j62)
        j64 = self.Mbar_rbs_l5_jcs_h[:,1:2]
        j65 = B(j59,j64)
        j66 = j62.T
        j67 = A(j59).T
        j68 = multi_dot([j66,j67])
        j69 = self.ubar_rbs_l5_jcs_h
        j70 = B(j59,j69)
        j71 = self.ubar_rbs_l2_jcs_h
        j72 = (self.R_rbs_l5.T + (-1) * self.R_rbs_l2.T + multi_dot([j69.T,j67]) + (-1) * multi_dot([j71.T,j18]))
        j73 = j64.T
        j74 = multi_dot([j73,j67])
        j75 = B(j17,j60)
        j76 = B(j17,j71)

        self.jac_eq_blocks = (j0,
        B(j1,self.ubar_ground_jcs_a),
        j9,
        (-1) * B(j5,self.ubar_rbs_l1_jcs_a),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j7.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j2,
        multi_dot([j8.T,j10,j11]),
        j2,
        multi_dot([j12.T,j6,(cos(config.UF_mcs_act(t)) * B(j1,j13) + (-1 * sin(config.UF_mcs_act(t))) * B(j1,j14))]),
        j2,
        multi_dot([(cos(config.UF_mcs_act(t)) * multi_dot([j13.T,j10]) + (-1 * sin(config.UF_mcs_act(t))) * multi_dot([j14.T,j10])),B(j5,j12)]),
        j0,
        B(j1,self.ubar_ground_jcs_b),
        j9,
        (-1) * B(j17,self.ubar_rbs_l2_jcs_b),
        j2,
        multi_dot([j16,j18,B(j1,j19)]),
        j2,
        multi_dot([j19.T,j10,j21]),
        j2,
        multi_dot([j16,j18,B(j1,j20)]),
        j2,
        multi_dot([j20.T,j10,j21]),
        j2,
        multi_dot([j23,j25,j27]),
        j2,
        multi_dot([j30,j6,j39]),
        j2,
        multi_dot([j23,j25,j29]),
        j2,
        multi_dot([j37,j6,j39]),
        j31,
        (multi_dot([j30,j6,j33]) + multi_dot([j36,j27])),
        (-1) * j31,
        (-1) * multi_dot([j30,j6,j40]),
        j38,
        (multi_dot([j37,j6,j33]) + multi_dot([j36,j29])),
        (-1) * j38,
        (-1) * multi_dot([j37,j6,j40]),
        j0,
        B(j17,self.ubar_rbs_l2_jcs_d),
        j9,
        (-1) * B(j24,self.ubar_rbs_l3_jcs_d),
        j2,
        multi_dot([j42,j44,j46]),
        j2,
        multi_dot([j49,j25,j57]),
        j2,
        multi_dot([j42,j44,j48]),
        j2,
        multi_dot([j55,j25,j57]),
        j50,
        (multi_dot([j49,j25,j52]) + multi_dot([j54,j46])),
        (-1) * j50,
        (-1) * multi_dot([j49,j25,j58]),
        j56,
        (multi_dot([j55,j25,j52]) + multi_dot([j54,j48])),
        (-1) * j56,
        (-1) * multi_dot([j55,j25,j58]),
        j0,
        B(j43,self.ubar_rbs_l4_jcs_f),
        j9,
        (-1) * B(j59,self.ubar_rbs_l5_jcs_f),
        j2,
        multi_dot([j66,j67,j75]),
        j2,
        multi_dot([j61,j18,j63]),
        j2,
        multi_dot([j73,j67,j75]),
        j2,
        multi_dot([j61,j18,j65]),
        (-1) * j68,
        (-1) * multi_dot([j66,j67,j76]),
        j68,
        (multi_dot([j66,j67,j70]) + multi_dot([j72,j63])),
        (-1) * j74,
        (-1) * multi_dot([j73,j67,j76]),
        j74,
        (multi_dot([j73,j67,j70]) + multi_dot([j72,j65])),
        j0,
        Z3x4,
        Z4x3,
        I4,
        j2,
        (2) * j5.T,
        j2,
        (2) * j17.T,
        j2,
        (2) * j24.T,
        j2,
        (2) * j43.T,
        j2,
        (2) * j59.T,)

    
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
        (4) * multi_dot([m6.T,config.Jbar_rbs_l5,m6]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_l1)
        f1 = G(self.Pd_rbs_l2)
        f2 = G(self.Pd_rbs_l3)
        f3 = G(self.Pd_rbs_l4)
        f4 = G(self.Pd_rbs_l5)

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
        (8) * multi_dot([f4.T,config.Jbar_rbs_l5,f4,self.P_rbs_l5]),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_ground_jcs_a = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_ground,self.ubar_ground_jcs_a).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,0:1]).T,A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_a[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,1:2]).T,A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_a[:,2:3]])]]),self.L_jcs_a])
        self.F_ground_jcs_a = Q_ground_jcs_a[0:3]
        Te_ground_jcs_a = Q_ground_jcs_a[3:7]
        self.T_ground_jcs_a = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_a])),self.F_ground_jcs_a]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_a]))
        Q_ground_mcs_act = (-1) * multi_dot([np.bmat([[Z1x3.T],[multi_dot([((-1 * sin(config.UF_mcs_act(t))) * B(self.P_ground,self.Mbar_ground_jcs_a[:,0:1]).T + cos(config.UF_mcs_act(t)) * B(self.P_ground,self.Mbar_ground_jcs_a[:,1:2]).T),A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_a[:,0:1]])]]),self.L_mcs_act])
        self.F_ground_mcs_act = Q_ground_mcs_act[0:3]
        Te_ground_mcs_act = Q_ground_mcs_act[3:7]
        self.T_ground_mcs_act = (0.5) * multi_dot([E(self.P_ground),Te_ground_mcs_act])
        Q_ground_jcs_b = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_ground,self.ubar_ground_jcs_b).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_b[:,0:1]).T,A(self.P_rbs_l2),self.Mbar_rbs_l2_jcs_b[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_b[:,1:2]).T,A(self.P_rbs_l2),self.Mbar_rbs_l2_jcs_b[:,2:3]])]]),self.L_jcs_b])
        self.F_ground_jcs_b = Q_ground_jcs_b[0:3]
        Te_ground_jcs_b = Q_ground_jcs_b[3:7]
        self.T_ground_jcs_b = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_b])),self.F_ground_jcs_b]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_b]))
        Q_rbs_l1_jcs_c = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_c[:,0:1]]),multi_dot([A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_c[:,1:2]])],[multi_dot([B(self.P_rbs_l1,self.Mbar_rbs_l1_jcs_c[:,0:1]).T,A(self.P_rbs_l3),self.Mbar_rbs_l3_jcs_c[:,2:3]]),multi_dot([B(self.P_rbs_l1,self.Mbar_rbs_l1_jcs_c[:,1:2]).T,A(self.P_rbs_l3),self.Mbar_rbs_l3_jcs_c[:,2:3]]),(multi_dot([B(self.P_rbs_l1,self.Mbar_rbs_l1_jcs_c[:,0:1]).T,((-1) * self.R_rbs_l3 + multi_dot([A(self.P_rbs_l1),self.ubar_rbs_l1_jcs_c]) + (-1) * multi_dot([A(self.P_rbs_l3),self.ubar_rbs_l3_jcs_c]) + self.R_rbs_l1)]) + multi_dot([B(self.P_rbs_l1,self.ubar_rbs_l1_jcs_c).T,A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_c[:,0:1]])),(multi_dot([B(self.P_rbs_l1,self.Mbar_rbs_l1_jcs_c[:,1:2]).T,((-1) * self.R_rbs_l3 + multi_dot([A(self.P_rbs_l1),self.ubar_rbs_l1_jcs_c]) + (-1) * multi_dot([A(self.P_rbs_l3),self.ubar_rbs_l3_jcs_c]) + self.R_rbs_l1)]) + multi_dot([B(self.P_rbs_l1,self.ubar_rbs_l1_jcs_c).T,A(self.P_rbs_l1),self.Mbar_rbs_l1_jcs_c[:,1:2]]))]]),self.L_jcs_c])
        self.F_rbs_l1_jcs_c = Q_rbs_l1_jcs_c[0:3]
        Te_rbs_l1_jcs_c = Q_rbs_l1_jcs_c[3:7]
        self.T_rbs_l1_jcs_c = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l1),self.ubar_rbs_l1_jcs_c])),self.F_rbs_l1_jcs_c]) + (0.5) * multi_dot([E(self.P_rbs_l1),Te_rbs_l1_jcs_c]))
        Q_rbs_l2_jcs_d = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l2,self.ubar_rbs_l2_jcs_d).T]]),self.L_jcs_d])
        self.F_rbs_l2_jcs_d = Q_rbs_l2_jcs_d[0:3]
        Te_rbs_l2_jcs_d = Q_rbs_l2_jcs_d[3:7]
        self.T_rbs_l2_jcs_d = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l2),self.ubar_rbs_l2_jcs_d])),self.F_rbs_l2_jcs_d]) + (0.5) * multi_dot([E(self.P_rbs_l2),Te_rbs_l2_jcs_d]))
        Q_rbs_l3_jcs_e = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_l3),self.Mbar_rbs_l3_jcs_e[:,0:1]]),multi_dot([A(self.P_rbs_l3),self.Mbar_rbs_l3_jcs_e[:,1:2]])],[multi_dot([B(self.P_rbs_l3,self.Mbar_rbs_l3_jcs_e[:,0:1]).T,A(self.P_rbs_l4),self.Mbar_rbs_l4_jcs_e[:,2:3]]),multi_dot([B(self.P_rbs_l3,self.Mbar_rbs_l3_jcs_e[:,1:2]).T,A(self.P_rbs_l4),self.Mbar_rbs_l4_jcs_e[:,2:3]]),(multi_dot([B(self.P_rbs_l3,self.Mbar_rbs_l3_jcs_e[:,0:1]).T,((-1) * self.R_rbs_l4 + multi_dot([A(self.P_rbs_l3),self.ubar_rbs_l3_jcs_e]) + (-1) * multi_dot([A(self.P_rbs_l4),self.ubar_rbs_l4_jcs_e]) + self.R_rbs_l3)]) + multi_dot([B(self.P_rbs_l3,self.ubar_rbs_l3_jcs_e).T,A(self.P_rbs_l3),self.Mbar_rbs_l3_jcs_e[:,0:1]])),(multi_dot([B(self.P_rbs_l3,self.Mbar_rbs_l3_jcs_e[:,1:2]).T,((-1) * self.R_rbs_l4 + multi_dot([A(self.P_rbs_l3),self.ubar_rbs_l3_jcs_e]) + (-1) * multi_dot([A(self.P_rbs_l4),self.ubar_rbs_l4_jcs_e]) + self.R_rbs_l3)]) + multi_dot([B(self.P_rbs_l3,self.ubar_rbs_l3_jcs_e).T,A(self.P_rbs_l3),self.Mbar_rbs_l3_jcs_e[:,1:2]]))]]),self.L_jcs_e])
        self.F_rbs_l3_jcs_e = Q_rbs_l3_jcs_e[0:3]
        Te_rbs_l3_jcs_e = Q_rbs_l3_jcs_e[3:7]
        self.T_rbs_l3_jcs_e = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l3),self.ubar_rbs_l3_jcs_e])),self.F_rbs_l3_jcs_e]) + (0.5) * multi_dot([E(self.P_rbs_l3),Te_rbs_l3_jcs_e]))
        Q_rbs_l4_jcs_f = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l4,self.ubar_rbs_l4_jcs_f).T]]),self.L_jcs_f])
        self.F_rbs_l4_jcs_f = Q_rbs_l4_jcs_f[0:3]
        Te_rbs_l4_jcs_f = Q_rbs_l4_jcs_f[3:7]
        self.T_rbs_l4_jcs_f = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l4),self.ubar_rbs_l4_jcs_f])),self.F_rbs_l4_jcs_f]) + (0.5) * multi_dot([E(self.P_rbs_l4),Te_rbs_l4_jcs_f]))
        Q_rbs_l5_jcs_h = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_l5),self.Mbar_rbs_l5_jcs_h[:,0:1]]),multi_dot([A(self.P_rbs_l5),self.Mbar_rbs_l5_jcs_h[:,1:2]])],[multi_dot([B(self.P_rbs_l5,self.Mbar_rbs_l5_jcs_h[:,0:1]).T,A(self.P_rbs_l2),self.Mbar_rbs_l2_jcs_h[:,2:3]]),multi_dot([B(self.P_rbs_l5,self.Mbar_rbs_l5_jcs_h[:,1:2]).T,A(self.P_rbs_l2),self.Mbar_rbs_l2_jcs_h[:,2:3]]),(multi_dot([B(self.P_rbs_l5,self.Mbar_rbs_l5_jcs_h[:,0:1]).T,((-1) * self.R_rbs_l2 + multi_dot([A(self.P_rbs_l5),self.ubar_rbs_l5_jcs_h]) + (-1) * multi_dot([A(self.P_rbs_l2),self.ubar_rbs_l2_jcs_h]) + self.R_rbs_l5)]) + multi_dot([B(self.P_rbs_l5,self.ubar_rbs_l5_jcs_h).T,A(self.P_rbs_l5),self.Mbar_rbs_l5_jcs_h[:,0:1]])),(multi_dot([B(self.P_rbs_l5,self.Mbar_rbs_l5_jcs_h[:,1:2]).T,((-1) * self.R_rbs_l2 + multi_dot([A(self.P_rbs_l5),self.ubar_rbs_l5_jcs_h]) + (-1) * multi_dot([A(self.P_rbs_l2),self.ubar_rbs_l2_jcs_h]) + self.R_rbs_l5)]) + multi_dot([B(self.P_rbs_l5,self.ubar_rbs_l5_jcs_h).T,A(self.P_rbs_l5),self.Mbar_rbs_l5_jcs_h[:,1:2]]))]]),self.L_jcs_h])
        self.F_rbs_l5_jcs_h = Q_rbs_l5_jcs_h[0:3]
        Te_rbs_l5_jcs_h = Q_rbs_l5_jcs_h[3:7]
        self.T_rbs_l5_jcs_h = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l5),self.ubar_rbs_l5_jcs_h])),self.F_rbs_l5_jcs_h]) + (0.5) * multi_dot([E(self.P_rbs_l5),Te_rbs_l5_jcs_h]))

        self.reactions = {'F_ground_jcs_a' : self.F_ground_jcs_a,
                        'T_ground_jcs_a' : self.T_ground_jcs_a,
                        'F_ground_mcs_act' : self.F_ground_mcs_act,
                        'T_ground_mcs_act' : self.T_ground_mcs_act,
                        'F_ground_jcs_b' : self.F_ground_jcs_b,
                        'T_ground_jcs_b' : self.T_ground_jcs_b,
                        'F_rbs_l1_jcs_c' : self.F_rbs_l1_jcs_c,
                        'T_rbs_l1_jcs_c' : self.T_rbs_l1_jcs_c,
                        'F_rbs_l2_jcs_d' : self.F_rbs_l2_jcs_d,
                        'T_rbs_l2_jcs_d' : self.T_rbs_l2_jcs_d,
                        'F_rbs_l3_jcs_e' : self.F_rbs_l3_jcs_e,
                        'T_rbs_l3_jcs_e' : self.T_rbs_l3_jcs_e,
                        'F_rbs_l4_jcs_f' : self.F_rbs_l4_jcs_f,
                        'T_rbs_l4_jcs_f' : self.T_rbs_l4_jcs_f,
                        'F_rbs_l5_jcs_h' : self.F_rbs_l5_jcs_h,
                        'T_rbs_l5_jcs_h' : self.T_rbs_l5_jcs_h}

