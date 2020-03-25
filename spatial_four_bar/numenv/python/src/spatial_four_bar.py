
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

        self.indicies_map = {'ground': 0, 'rbs_l1': 1, 'rbs_l2': 2, 'rbs_l3': 3}

        self.n  = 28
        self.nc = 28
        self.nrows = 15
        self.ncols = 2*4
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_ground_jcs_a', 'T_ground_jcs_a', 'F_ground_mcs_act', 'T_ground_mcs_act', 'F_rbs_l1_jcs_b', 'T_rbs_l1_jcs_b', 'F_rbs_l2_jcs_c', 'T_rbs_l2_jcs_c', 'F_rbs_l3_jcs_d', 'T_rbs_l3_jcs_d']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.ground*2, self.ground*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.ground*2, self.ground*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.ground*2, self.ground*2+1, self.rbs_l3*2, self.rbs_l3*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_l1*2, self.rbs_l1*2+1, self.rbs_l2*2, self.rbs_l2*2+1, self.rbs_l3*2, self.rbs_l3*2+1], dtype=np.intc)

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
        self.config.P_rbs_l3], out=self._q)

        np.concatenate([self.config.Rd_ground,
        self.config.Pd_ground,
        self.config.Rd_rbs_l1,
        self.config.Pd_rbs_l1,
        self.config.Rd_rbs_l2,
        self.config.Pd_rbs_l2,
        self.config.Rd_rbs_l3,
        self.config.Pd_rbs_l3], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p + 'ground']
        self.rbs_l1 = indicies_map[p + 'rbs_l1']
        self.rbs_l2 = indicies_map[p + 'rbs_l2']
        self.rbs_l3 = indicies_map[p + 'rbs_l3']
    

    
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

        self.Mbar_ground_jcs_a = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_l1_jcs_a = multi_dot([A(config.P_rbs_l1).T,triad(config.ax1_jcs_a)])
        self.ubar_ground_jcs_a = (multi_dot([A(self.P_ground).T,config.pt1_jcs_a]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_l1_jcs_a = (multi_dot([A(config.P_rbs_l1).T,config.pt1_jcs_a]) + (-1) * multi_dot([A(config.P_rbs_l1).T,config.R_rbs_l1]))
        self.Mbar_ground_jcs_a = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_l1_jcs_a = multi_dot([A(config.P_rbs_l1).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_l1_jcs_b = multi_dot([A(config.P_rbs_l1).T,triad(config.ax1_jcs_b)])
        self.Mbar_rbs_l2_jcs_b = multi_dot([A(config.P_rbs_l2).T,triad(config.ax1_jcs_b)])
        self.ubar_rbs_l1_jcs_b = (multi_dot([A(config.P_rbs_l1).T,config.pt1_jcs_b]) + (-1) * multi_dot([A(config.P_rbs_l1).T,config.R_rbs_l1]))
        self.ubar_rbs_l2_jcs_b = (multi_dot([A(config.P_rbs_l2).T,config.pt1_jcs_b]) + (-1) * multi_dot([A(config.P_rbs_l2).T,config.R_rbs_l2]))
        self.Mbar_rbs_l2_jcs_c = multi_dot([A(config.P_rbs_l2).T,triad(config.ax1_jcs_c)])
        self.Mbar_rbs_l3_jcs_c = multi_dot([A(config.P_rbs_l3).T,triad(config.ax2_jcs_c,triad(config.ax1_jcs_c)[0:3,1:2])])
        self.ubar_rbs_l2_jcs_c = (multi_dot([A(config.P_rbs_l2).T,config.pt1_jcs_c]) + (-1) * multi_dot([A(config.P_rbs_l2).T,config.R_rbs_l2]))
        self.ubar_rbs_l3_jcs_c = (multi_dot([A(config.P_rbs_l3).T,config.pt1_jcs_c]) + (-1) * multi_dot([A(config.P_rbs_l3).T,config.R_rbs_l3]))
        self.Mbar_rbs_l3_jcs_d = multi_dot([A(config.P_rbs_l3).T,triad(config.ax1_jcs_d)])
        self.Mbar_ground_jcs_d = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_d)])
        self.ubar_rbs_l3_jcs_d = (multi_dot([A(config.P_rbs_l3).T,config.pt1_jcs_d]) + (-1) * multi_dot([A(config.P_rbs_l3).T,config.R_rbs_l3]))
        self.ubar_ground_jcs_d = (multi_dot([A(self.P_ground).T,config.pt1_jcs_d]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))

    
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

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcs_a = Lambda[0:5]
        self.L_mcs_act = Lambda[5:6]
        self.L_jcs_b = Lambda[6:9]
        self.L_jcs_c = Lambda[9:13]
        self.L_jcs_d = Lambda[13:18]

    
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
        x10 = self.P_rbs_l2
        x11 = A(x10)
        x12 = self.R_rbs_l3
        x13 = self.P_rbs_l3
        x14 = A(x13)
        x15 = x14.T
        x16 = self.Mbar_ground_jcs_d[:,2:3]
        x17 = (-1) * I1

        self.pos_eq_blocks = ((x0 + (-1) * x1 + multi_dot([x3,self.ubar_ground_jcs_a]) + (-1) * multi_dot([x5,self.ubar_rbs_l1_jcs_a])),
        multi_dot([self.Mbar_ground_jcs_a[:,0:1].T,x6,x5,x7]),
        multi_dot([self.Mbar_ground_jcs_a[:,1:2].T,x6,x5,x7]),
        (cos(config.UF_mcs_act(t)) * multi_dot([self.Mbar_ground_jcs_a[:,1:2].T,x6,x5,x8]) + (-1 * sin(config.UF_mcs_act(t))) * multi_dot([self.Mbar_ground_jcs_a[:,0:1].T,x6,x5,x8])),
        (x1 + (-1) * x9 + multi_dot([x5,self.ubar_rbs_l1_jcs_b]) + (-1) * multi_dot([x11,self.ubar_rbs_l2_jcs_b])),
        (x9 + (-1) * x12 + multi_dot([x11,self.ubar_rbs_l2_jcs_c]) + (-1) * multi_dot([x14,self.ubar_rbs_l3_jcs_c])),
        multi_dot([self.Mbar_rbs_l2_jcs_c[:,0:1].T,x11.T,x14,self.Mbar_rbs_l3_jcs_c[:,0:1]]),
        (x12 + (-1) * x0 + multi_dot([x14,self.ubar_rbs_l3_jcs_d]) + (-1) * multi_dot([x3,self.ubar_ground_jcs_d])),
        multi_dot([self.Mbar_rbs_l3_jcs_d[:,0:1].T,x15,x3,x16]),
        multi_dot([self.Mbar_rbs_l3_jcs_d[:,1:2].T,x15,x3,x16]),
        x0,
        (x2 + (-1) * self.Pg_ground),
        (x17 + multi_dot([x4.T,x4])),
        (x17 + multi_dot([x10.T,x10])),
        (x17 + multi_dot([x13.T,x13])),)

    
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
        v0,
        v1,
        v0,
        v1,
        v1,
        v0,
        Z4x1,
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
        a17 = self.Pd_rbs_l3
        a18 = self.Mbar_rbs_l2_jcs_c[:,0:1]
        a19 = self.P_rbs_l2
        a20 = self.Mbar_rbs_l3_jcs_c[:,0:1]
        a21 = self.P_rbs_l3
        a22 = A(a21).T
        a23 = a16.T
        a24 = self.Mbar_ground_jcs_d[:,2:3]
        a25 = a24.T
        a26 = self.Mbar_rbs_l3_jcs_d[:,0:1]
        a27 = B(a0,a24)
        a28 = a17.T
        a29 = B(a3,a24)
        a30 = self.Mbar_rbs_l3_jcs_d[:,1:2]

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_ground_jcs_a),a0]) + (-1) * multi_dot([B(a1,self.ubar_rbs_l1_jcs_a),a1])),
        (multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + (2) * multi_dot([a10,B(a3,a2).T,a11,a1])),
        (multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + (2) * multi_dot([a10,B(a3,a12).T,a11,a1])),
        ((-1 * derivative(config.UF_mcs_act, t, 0.1, 2)) * I1 + multi_dot([a13.T,a9,(cos(config.UF_mcs_act(t)) * B(a0,a14) + (-1 * sin(config.UF_mcs_act(t))) * B(a0,a15)),a0]) + multi_dot([(cos(config.UF_mcs_act(t)) * multi_dot([a14.T,a4]) + (-1 * sin(config.UF_mcs_act(t))) * multi_dot([a15.T,a4])),B(a1,a13),a1]) + (2) * multi_dot([(cos(config.UF_mcs_act(t)) * multi_dot([a10,B(a3,a14).T]) + (-1 * sin(config.UF_mcs_act(t))) * multi_dot([a10,B(a3,a15).T])),B(a8,a13),a1])),
        (multi_dot([B(a1,self.ubar_rbs_l1_jcs_b),a1]) + (-1) * multi_dot([B(a16,self.ubar_rbs_l2_jcs_b),a16])),
        (multi_dot([B(a16,self.ubar_rbs_l2_jcs_c),a16]) + (-1) * multi_dot([B(a17,self.ubar_rbs_l3_jcs_c),a17])),
        (multi_dot([a18.T,A(a19).T,B(a17,a20),a17]) + multi_dot([a20.T,a22,B(a16,a18),a16]) + (2) * multi_dot([a23,B(a19,a18).T,B(a21,a20),a17])),
        (multi_dot([B(a17,self.ubar_rbs_l3_jcs_d),a17]) + (-1) * multi_dot([B(a0,self.ubar_ground_jcs_d),a0])),
        (multi_dot([a25,a4,B(a17,a26),a17]) + multi_dot([a26.T,a22,a27,a0]) + (2) * multi_dot([a28,B(a21,a26).T,a29,a0])),
        (multi_dot([a25,a4,B(a17,a30),a17]) + multi_dot([a30.T,a22,a27,a0]) + (2) * multi_dot([a28,B(a21,a30).T,a29,a0])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a1.T,a1]),
        (2) * multi_dot([a23,a16]),
        (2) * multi_dot([a28,a17]),)

    
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
        j15 = self.P_rbs_l2
        j16 = self.Mbar_rbs_l3_jcs_c[:,0:1]
        j17 = self.P_rbs_l3
        j18 = A(j17).T
        j19 = self.Mbar_rbs_l2_jcs_c[:,0:1]
        j20 = self.Mbar_ground_jcs_d[:,2:3]
        j21 = j20.T
        j22 = self.Mbar_rbs_l3_jcs_d[:,0:1]
        j23 = self.Mbar_rbs_l3_jcs_d[:,1:2]
        j24 = B(j1,j20)

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
        B(j5,self.ubar_rbs_l1_jcs_b),
        j9,
        (-1) * B(j15,self.ubar_rbs_l2_jcs_b),
        j0,
        B(j15,self.ubar_rbs_l2_jcs_c),
        j9,
        (-1) * B(j17,self.ubar_rbs_l3_jcs_c),
        j2,
        multi_dot([j16.T,j18,B(j15,j19)]),
        j2,
        multi_dot([j19.T,A(j15).T,B(j17,j16)]),
        j9,
        (-1) * B(j1,self.ubar_ground_jcs_d),
        j0,
        B(j17,self.ubar_rbs_l3_jcs_d),
        j2,
        multi_dot([j22.T,j18,j24]),
        j2,
        multi_dot([j21,j10,B(j17,j22)]),
        j2,
        multi_dot([j23.T,j18,j24]),
        j2,
        multi_dot([j21,j10,B(j17,j23)]),
        j0,
        Z3x4,
        Z4x3,
        I4,
        j2,
        (2) * j5.T,
        j2,
        (2) * j15.T,
        j2,
        (2) * j17.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_ground)
        m2 = G(self.P_rbs_l1)
        m3 = G(self.P_rbs_l2)
        m4 = G(self.P_rbs_l3)

        self.mass_eq_blocks = (self.m_ground * m0,
        (4) * multi_dot([m1.T,self.Jbar_ground,m1]),
        config.m_rbs_l1 * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbs_l1,m2]),
        config.m_rbs_l2 * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbs_l2,m3]),
        config.m_rbs_l3 * m0,
        (4) * multi_dot([m4.T,config.Jbar_rbs_l3,m4]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_l1)
        f1 = G(self.Pd_rbs_l2)
        f2 = G(self.Pd_rbs_l3)

        self.frc_eq_blocks = (Z3x1,
        Z4x1,
        self.F_rbs_l1_gravity,
        (8) * multi_dot([f0.T,config.Jbar_rbs_l1,f0,self.P_rbs_l1]),
        self.F_rbs_l2_gravity,
        (8) * multi_dot([f1.T,config.Jbar_rbs_l2,f1,self.P_rbs_l2]),
        self.F_rbs_l3_gravity,
        (8) * multi_dot([f2.T,config.Jbar_rbs_l3,f2,self.P_rbs_l3]),)

    
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
        Q_rbs_l1_jcs_b = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_l1,self.ubar_rbs_l1_jcs_b).T]]),self.L_jcs_b])
        self.F_rbs_l1_jcs_b = Q_rbs_l1_jcs_b[0:3]
        Te_rbs_l1_jcs_b = Q_rbs_l1_jcs_b[3:7]
        self.T_rbs_l1_jcs_b = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l1),self.ubar_rbs_l1_jcs_b])),self.F_rbs_l1_jcs_b]) + (0.5) * multi_dot([E(self.P_rbs_l1),Te_rbs_l1_jcs_b]))
        Q_rbs_l2_jcs_c = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbs_l2,self.ubar_rbs_l2_jcs_c).T,multi_dot([B(self.P_rbs_l2,self.Mbar_rbs_l2_jcs_c[:,0:1]).T,A(self.P_rbs_l3),self.Mbar_rbs_l3_jcs_c[:,0:1]])]]),self.L_jcs_c])
        self.F_rbs_l2_jcs_c = Q_rbs_l2_jcs_c[0:3]
        Te_rbs_l2_jcs_c = Q_rbs_l2_jcs_c[3:7]
        self.T_rbs_l2_jcs_c = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l2),self.ubar_rbs_l2_jcs_c])),self.F_rbs_l2_jcs_c]) + (0.5) * multi_dot([E(self.P_rbs_l2),Te_rbs_l2_jcs_c]))
        Q_rbs_l3_jcs_d = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_l3,self.ubar_rbs_l3_jcs_d).T,multi_dot([B(self.P_rbs_l3,self.Mbar_rbs_l3_jcs_d[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_d[:,2:3]]),multi_dot([B(self.P_rbs_l3,self.Mbar_rbs_l3_jcs_d[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcs_d[:,2:3]])]]),self.L_jcs_d])
        self.F_rbs_l3_jcs_d = Q_rbs_l3_jcs_d[0:3]
        Te_rbs_l3_jcs_d = Q_rbs_l3_jcs_d[3:7]
        self.T_rbs_l3_jcs_d = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_l3),self.ubar_rbs_l3_jcs_d])),self.F_rbs_l3_jcs_d]) + (0.5) * multi_dot([E(self.P_rbs_l3),Te_rbs_l3_jcs_d]))

        self.reactions = {'F_ground_jcs_a' : self.F_ground_jcs_a,
                        'T_ground_jcs_a' : self.T_ground_jcs_a,
                        'F_ground_mcs_act' : self.F_ground_mcs_act,
                        'T_ground_mcs_act' : self.T_ground_mcs_act,
                        'F_rbs_l1_jcs_b' : self.F_rbs_l1_jcs_b,
                        'T_rbs_l1_jcs_b' : self.T_rbs_l1_jcs_b,
                        'F_rbs_l2_jcs_c' : self.F_rbs_l2_jcs_c,
                        'T_rbs_l2_jcs_c' : self.T_rbs_l2_jcs_c,
                        'F_rbs_l3_jcs_d' : self.F_rbs_l3_jcs_d,
                        'T_rbs_l3_jcs_d' : self.T_rbs_l3_jcs_d}

