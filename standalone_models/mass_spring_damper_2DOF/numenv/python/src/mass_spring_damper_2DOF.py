
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

        self.indicies_map = {'ground': 0, 'rbs_body_1': 1, 'rbs_body_2': 2}

        self.n  = 21
        self.nc = 19
        self.nrows = 14
        self.ncols = 2*3
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_rbs_body_1_jcs_trans_1', 'T_rbs_body_1_jcs_trans_1', 'F_rbs_body_1_fas_TSDA_1', 'T_rbs_body_1_fas_TSDA_1', 'F_rbs_body_2_jcs_trans_2', 'T_rbs_body_2_jcs_trans_2', 'F_rbs_body_2_fas_TSDA_2', 'T_rbs_body_2_fas_TSDA_2']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.ground*2, self.ground*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.ground*2, self.ground*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.ground*2, self.ground*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.ground*2, self.ground*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.ground*2, self.ground*2+1, self.rbs_body_2*2, self.rbs_body_2*2+1, self.ground*2, self.ground*2+1, self.rbs_body_2*2, self.rbs_body_2*2+1, self.ground*2, self.ground*2+1, self.rbs_body_2*2, self.rbs_body_2*2+1, self.ground*2, self.ground*2+1, self.rbs_body_2*2, self.rbs_body_2*2+1, self.ground*2, self.ground*2+1, self.rbs_body_2*2, self.rbs_body_2*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.rbs_body_2*2, self.rbs_body_2*2+1], dtype=np.intc)

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
        self.config.R_rbs_body_1,
        self.config.P_rbs_body_1,
        self.config.R_rbs_body_2,
        self.config.P_rbs_body_2], out=self._q)

        np.concatenate([self.config.Rd_ground,
        self.config.Pd_ground,
        self.config.Rd_rbs_body_1,
        self.config.Pd_rbs_body_1,
        self.config.Rd_rbs_body_2,
        self.config.Pd_rbs_body_2], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p + 'ground']
        self.rbs_body_1 = indicies_map[p + 'rbs_body_1']
        self.rbs_body_2 = indicies_map[p + 'rbs_body_2']
    

    
    def eval_constants(self):
        config = self.config

        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.m_ground = 1.0
        self.Jbar_ground = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.F_rbs_body_1_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_body_1]], dtype=np.float64)
        self.T_rbs_body_1_fas_TSDA_1 = Z3x1
        self.T_ground_fas_TSDA_1 = Z3x1
        self.F_rbs_body_2_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_body_2]], dtype=np.float64)
        self.T_rbs_body_2_fas_TSDA_2 = Z3x1
        self.T_rbs_body_1_fas_TSDA_2 = Z3x1

        self.Mbar_rbs_body_1_jcs_trans_1 = multi_dot([A(config.P_rbs_body_1).T,triad(config.ax1_jcs_trans_1)])
        self.Mbar_ground_jcs_trans_1 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_trans_1)])
        self.ubar_rbs_body_1_jcs_trans_1 = (multi_dot([A(config.P_rbs_body_1).T,config.pt1_jcs_trans_1]) + (-1) * multi_dot([A(config.P_rbs_body_1).T,config.R_rbs_body_1]))
        self.ubar_ground_jcs_trans_1 = (multi_dot([A(self.P_ground).T,config.pt1_jcs_trans_1]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_body_1_fas_TSDA_1 = (multi_dot([A(config.P_rbs_body_1).T,config.pt1_fas_TSDA_1]) + (-1) * multi_dot([A(config.P_rbs_body_1).T,config.R_rbs_body_1]))
        self.ubar_ground_fas_TSDA_1 = (multi_dot([A(self.P_ground).T,config.pt2_fas_TSDA_1]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbs_body_2_jcs_trans_2 = multi_dot([A(config.P_rbs_body_2).T,triad(config.ax1_jcs_trans_2)])
        self.Mbar_ground_jcs_trans_2 = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_trans_2)])
        self.ubar_rbs_body_2_jcs_trans_2 = (multi_dot([A(config.P_rbs_body_2).T,config.pt1_jcs_trans_2]) + (-1) * multi_dot([A(config.P_rbs_body_2).T,config.R_rbs_body_2]))
        self.ubar_ground_jcs_trans_2 = (multi_dot([A(self.P_ground).T,config.pt1_jcs_trans_2]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_body_2_fas_TSDA_2 = (multi_dot([A(config.P_rbs_body_2).T,config.pt1_fas_TSDA_2]) + (-1) * multi_dot([A(config.P_rbs_body_2).T,config.R_rbs_body_2]))
        self.ubar_rbs_body_1_fas_TSDA_2 = (multi_dot([A(config.P_rbs_body_1).T,config.pt2_fas_TSDA_2]) + (-1) * multi_dot([A(config.P_rbs_body_1).T,config.R_rbs_body_1]))

    
    def _map_gen_coordinates(self):
        q = self._q
        self.R_ground = q[0:3]
        self.P_ground = q[3:7]
        self.R_rbs_body_1 = q[7:10]
        self.P_rbs_body_1 = q[10:14]
        self.R_rbs_body_2 = q[14:17]
        self.P_rbs_body_2 = q[17:21]

    
    def _map_gen_velocities(self):
        qd = self._qd
        self.Rd_ground = qd[0:3]
        self.Pd_ground = qd[3:7]
        self.Rd_rbs_body_1 = qd[7:10]
        self.Pd_rbs_body_1 = qd[10:14]
        self.Rd_rbs_body_2 = qd[14:17]
        self.Pd_rbs_body_2 = qd[17:21]

    
    def _map_gen_accelerations(self):
        qdd = self._qdd
        self.Rdd_ground = qdd[0:3]
        self.Pdd_ground = qdd[3:7]
        self.Rdd_rbs_body_1 = qdd[7:10]
        self.Pdd_rbs_body_1 = qdd[10:14]
        self.Rdd_rbs_body_2 = qdd[14:17]
        self.Pdd_rbs_body_2 = qdd[17:21]

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcs_trans_1 = Lambda[0:5]
        self.L_jcs_trans_2 = Lambda[5:10]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.Mbar_rbs_body_1_jcs_trans_1[:,0:1].T
        x1 = self.P_rbs_body_1
        x2 = A(x1)
        x3 = x2.T
        x4 = self.P_ground
        x5 = A(x4)
        x6 = self.Mbar_ground_jcs_trans_1[:,2:3]
        x7 = self.Mbar_rbs_body_1_jcs_trans_1[:,1:2].T
        x8 = self.R_ground
        x9 = (-1) * x8
        x10 = (self.R_rbs_body_1 + x9 + multi_dot([x2,self.ubar_rbs_body_1_jcs_trans_1]) + (-1) * multi_dot([x5,self.ubar_ground_jcs_trans_1]))
        x11 = self.Mbar_rbs_body_2_jcs_trans_2[:,0:1].T
        x12 = self.P_rbs_body_2
        x13 = A(x12)
        x14 = x13.T
        x15 = self.Mbar_ground_jcs_trans_2[:,2:3]
        x16 = self.Mbar_rbs_body_2_jcs_trans_2[:,1:2].T
        x17 = (self.R_rbs_body_2 + x9 + multi_dot([x13,self.ubar_rbs_body_2_jcs_trans_2]) + (-1) * multi_dot([x5,self.ubar_ground_jcs_trans_2]))
        x18 = (-1) * I1

        self.pos_eq_blocks = (multi_dot([x0,x3,x5,x6]),
        multi_dot([x7,x3,x5,x6]),
        multi_dot([x0,x3,x10]),
        multi_dot([x7,x3,x10]),
        multi_dot([x0,x3,x5,self.Mbar_ground_jcs_trans_1[:,1:2]]),
        multi_dot([x11,x14,x5,x15]),
        multi_dot([x16,x14,x5,x15]),
        multi_dot([x11,x14,x17]),
        multi_dot([x16,x14,x17]),
        multi_dot([x11,x14,x5,self.Mbar_ground_jcs_trans_2[:,1:2]]),
        x8,
        (x4 + (-1) * self.Pg_ground),
        (x18 + multi_dot([x1.T,x1])),
        (x18 + multi_dot([x12.T,x12])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z1x1

        self.vel_eq_blocks = (v0,
        v0,
        v0,
        v0,
        v0,
        v0,
        v0,
        v0,
        v0,
        v0,
        Z3x1,
        Z4x1,
        v0,
        v0,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Mbar_ground_jcs_trans_1[:,2:3]
        a1 = a0.T
        a2 = self.P_ground
        a3 = A(a2).T
        a4 = self.Pd_rbs_body_1
        a5 = self.Mbar_rbs_body_1_jcs_trans_1[:,0:1]
        a6 = B(a4,a5)
        a7 = a5.T
        a8 = self.P_rbs_body_1
        a9 = A(a8).T
        a10 = self.Pd_ground
        a11 = B(a10,a0)
        a12 = a4.T
        a13 = B(a8,a5).T
        a14 = B(a2,a0)
        a15 = self.Mbar_rbs_body_1_jcs_trans_1[:,1:2]
        a16 = B(a4,a15)
        a17 = a15.T
        a18 = B(a8,a15).T
        a19 = self.ubar_rbs_body_1_jcs_trans_1
        a20 = self.ubar_ground_jcs_trans_1
        a21 = (multi_dot([B(a4,a19),a4]) + (-1) * multi_dot([B(a10,a20),a10]))
        a22 = (-1) * self.Rd_ground
        a23 = (self.Rd_rbs_body_1 + a22 + multi_dot([B(a8,a19),a4]) + (-1) * multi_dot([B(a2,a20),a10]))
        a24 = (-1) * self.R_ground.T
        a25 = (self.R_rbs_body_1.T + a24 + multi_dot([a19.T,a9]) + (-1) * multi_dot([a20.T,a3]))
        a26 = self.Mbar_ground_jcs_trans_1[:,1:2]
        a27 = self.Mbar_rbs_body_2_jcs_trans_2[:,0:1]
        a28 = a27.T
        a29 = self.P_rbs_body_2
        a30 = A(a29).T
        a31 = self.Mbar_ground_jcs_trans_2[:,2:3]
        a32 = B(a10,a31)
        a33 = a31.T
        a34 = self.Pd_rbs_body_2
        a35 = B(a34,a27)
        a36 = a34.T
        a37 = B(a29,a27).T
        a38 = B(a2,a31)
        a39 = self.Mbar_rbs_body_2_jcs_trans_2[:,1:2]
        a40 = a39.T
        a41 = B(a34,a39)
        a42 = B(a29,a39).T
        a43 = self.ubar_rbs_body_2_jcs_trans_2
        a44 = self.ubar_ground_jcs_trans_2
        a45 = (multi_dot([B(a34,a43),a34]) + (-1) * multi_dot([B(a10,a44),a10]))
        a46 = (self.Rd_rbs_body_2 + a22 + multi_dot([B(a29,a43),a34]) + (-1) * multi_dot([B(a2,a44),a10]))
        a47 = (self.R_rbs_body_2.T + a24 + multi_dot([a43.T,a30]) + (-1) * multi_dot([a44.T,a3]))
        a48 = self.Mbar_ground_jcs_trans_2[:,1:2]

        self.acc_eq_blocks = ((multi_dot([a1,a3,a6,a4]) + multi_dot([a7,a9,a11,a10]) + (2) * multi_dot([a12,a13,a14,a10])),
        (multi_dot([a1,a3,a16,a4]) + multi_dot([a17,a9,a11,a10]) + (2) * multi_dot([a12,a18,a14,a10])),
        (multi_dot([a7,a9,a21]) + (2) * multi_dot([a12,a13,a23]) + multi_dot([a25,a6,a4])),
        (multi_dot([a17,a9,a21]) + (2) * multi_dot([a12,a18,a23]) + multi_dot([a25,a16,a4])),
        (multi_dot([a26.T,a3,a6,a4]) + multi_dot([a7,a9,B(a10,a26),a10]) + (2) * multi_dot([a12,a13,B(a2,a26),a10])),
        (multi_dot([a28,a30,a32,a10]) + multi_dot([a33,a3,a35,a34]) + (2) * multi_dot([a36,a37,a38,a10])),
        (multi_dot([a40,a30,a32,a10]) + multi_dot([a33,a3,a41,a34]) + (2) * multi_dot([a36,a42,a38,a10])),
        (multi_dot([a28,a30,a45]) + (2) * multi_dot([a36,a37,a46]) + multi_dot([a47,a35,a34])),
        (multi_dot([a40,a30,a45]) + (2) * multi_dot([a36,a42,a46]) + multi_dot([a47,a41,a34])),
        (multi_dot([a28,a30,B(a10,a48),a10]) + multi_dot([a48.T,a3,a35,a34]) + (2) * multi_dot([a36,a37,B(a2,a48),a10])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a12,a4]),
        (2) * multi_dot([a36,a34]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = Z1x3
        j1 = self.Mbar_ground_jcs_trans_1[:,2:3]
        j2 = j1.T
        j3 = self.P_ground
        j4 = A(j3).T
        j5 = self.P_rbs_body_1
        j6 = self.Mbar_rbs_body_1_jcs_trans_1[:,0:1]
        j7 = B(j5,j6)
        j8 = self.Mbar_rbs_body_1_jcs_trans_1[:,1:2]
        j9 = B(j5,j8)
        j10 = j6.T
        j11 = A(j5).T
        j12 = multi_dot([j10,j11])
        j13 = self.ubar_rbs_body_1_jcs_trans_1
        j14 = B(j5,j13)
        j15 = (-1) * self.R_ground.T
        j16 = self.ubar_ground_jcs_trans_1
        j17 = (self.R_rbs_body_1.T + j15 + multi_dot([j13.T,j11]) + (-1) * multi_dot([j16.T,j4]))
        j18 = j8.T
        j19 = multi_dot([j18,j11])
        j20 = self.Mbar_ground_jcs_trans_1[:,1:2]
        j21 = B(j3,j1)
        j22 = B(j3,j16)
        j23 = self.Mbar_ground_jcs_trans_2[:,2:3]
        j24 = j23.T
        j25 = self.P_rbs_body_2
        j26 = self.Mbar_rbs_body_2_jcs_trans_2[:,0:1]
        j27 = B(j25,j26)
        j28 = self.Mbar_rbs_body_2_jcs_trans_2[:,1:2]
        j29 = B(j25,j28)
        j30 = j26.T
        j31 = A(j25).T
        j32 = multi_dot([j30,j31])
        j33 = self.ubar_rbs_body_2_jcs_trans_2
        j34 = B(j25,j33)
        j35 = self.ubar_ground_jcs_trans_2
        j36 = (self.R_rbs_body_2.T + j15 + multi_dot([j33.T,j31]) + (-1) * multi_dot([j35.T,j4]))
        j37 = j28.T
        j38 = multi_dot([j37,j31])
        j39 = self.Mbar_ground_jcs_trans_2[:,1:2]
        j40 = B(j3,j23)
        j41 = B(j3,j35)

        self.jac_eq_blocks = (j0,
        multi_dot([j10,j11,j21]),
        j0,
        multi_dot([j2,j4,j7]),
        j0,
        multi_dot([j18,j11,j21]),
        j0,
        multi_dot([j2,j4,j9]),
        (-1) * j12,
        (-1) * multi_dot([j10,j11,j22]),
        j12,
        (multi_dot([j10,j11,j14]) + multi_dot([j17,j7])),
        (-1) * j19,
        (-1) * multi_dot([j18,j11,j22]),
        j19,
        (multi_dot([j18,j11,j14]) + multi_dot([j17,j9])),
        j0,
        multi_dot([j10,j11,B(j3,j20)]),
        j0,
        multi_dot([j20.T,j4,j7]),
        j0,
        multi_dot([j30,j31,j40]),
        j0,
        multi_dot([j24,j4,j27]),
        j0,
        multi_dot([j37,j31,j40]),
        j0,
        multi_dot([j24,j4,j29]),
        (-1) * j32,
        (-1) * multi_dot([j30,j31,j41]),
        j32,
        (multi_dot([j30,j31,j34]) + multi_dot([j36,j27])),
        (-1) * j38,
        (-1) * multi_dot([j37,j31,j41]),
        j38,
        (multi_dot([j37,j31,j34]) + multi_dot([j36,j29])),
        j0,
        multi_dot([j30,j31,B(j3,j39)]),
        j0,
        multi_dot([j39.T,j4,j27]),
        I3,
        Z3x4,
        Z4x3,
        I4,
        j0,
        (2) * j5.T,
        j0,
        (2) * j25.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_ground)
        m2 = G(self.P_rbs_body_1)
        m3 = G(self.P_rbs_body_2)

        self.mass_eq_blocks = (self.m_ground * m0,
        (4) * multi_dot([m1.T,self.Jbar_ground,m1]),
        config.m_rbs_body_1 * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbs_body_1,m2]),
        config.m_rbs_body_2 * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbs_body_2,m3]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = Z3x1
        f1 = self.R_rbs_body_1
        f2 = f1.T
        f3 = self.R_ground
        f4 = self.ubar_rbs_body_1_fas_TSDA_1
        f5 = self.P_rbs_body_1
        f6 = A(f5)
        f7 = f6.T
        f8 = self.ubar_ground_fas_TSDA_1
        f9 = self.P_ground
        f10 = A(f9)
        f11 = (f2 + (-1) * f3.T + multi_dot([f4.T,f7]) + (-1) * multi_dot([f8.T,f10.T]))
        f12 = multi_dot([f6,f4])
        f13 = multi_dot([f10,f8])
        f14 = (f1 + (-1) * f3 + f12 + (-1) * f13)
        f15 = ((multi_dot([f11,f14]))**(1.0/2.0))[0]
        f16 = 1.0/f15
        f17 = config.UF_fas_TSDA_1_Fs((config.fas_TSDA_1_FL + (-1 * f15)))
        f18 = self.Rd_rbs_body_1
        f19 = self.Pd_rbs_body_1
        f20 = config.UF_fas_TSDA_1_Fd((-1 * 1.0/f15) * multi_dot([f11,(f18 + (-1) * self.Rd_ground + multi_dot([B(f5,f4),f19]) + (-1) * multi_dot([B(f9,f8),self.Pd_ground]))]))
        f21 = (f16 * (f17 + f20)) * f14
        f22 = Z4x1
        f23 = (2 * f17)
        f24 = (2 * f20)
        f25 = self.R_rbs_body_2
        f26 = self.ubar_rbs_body_2_fas_TSDA_2
        f27 = self.P_rbs_body_2
        f28 = A(f27)
        f29 = self.ubar_rbs_body_1_fas_TSDA_2
        f30 = (f25.T + (-1) * f2 + multi_dot([f26.T,f28.T]) + (-1) * multi_dot([f29.T,f7]))
        f31 = multi_dot([f28,f26])
        f32 = multi_dot([f6,f29])
        f33 = (f25 + (-1) * f1 + f31 + (-1) * f32)
        f34 = ((multi_dot([f30,f33]))**(1.0/2.0))[0]
        f35 = 1.0/f34
        f36 = config.UF_fas_TSDA_2_Fs((config.fas_TSDA_2_FL + (-1 * f34)))
        f37 = self.Pd_rbs_body_2
        f38 = config.UF_fas_TSDA_2_Fd((-1 * 1.0/f34) * multi_dot([f30,(self.Rd_rbs_body_2 + (-1) * f18 + multi_dot([B(f27,f26),f37]) + (-1) * multi_dot([B(f5,f29),f19]))]))
        f39 = (f35 * (f36 + f38)) * f33
        f40 = G(f19)
        f41 = E(f5).T
        f42 = (2 * f36)
        f43 = (2 * f38)
        f44 = G(f37)

        self.frc_eq_blocks = ((f0 + (-1) * f21),
        (f22 + (f16 * (f23 + f24)) * multi_dot([E(f9).T,skew(f13).T,f14])),
        (self.F_rbs_body_1_gravity + f0 + f21 + (-1) * f39),
        (f22 + (8) * multi_dot([f40.T,config.Jbar_rbs_body_1,f40,f5]) + (f16 * ((-1 * f23) + (-1 * f24))) * multi_dot([f41,skew(f12).T,f14]) + (f35 * (f42 + f43)) * multi_dot([f41,skew(f32).T,f33])),
        (self.F_rbs_body_2_gravity + f39),
        ((8) * multi_dot([f44.T,config.Jbar_rbs_body_2,f44,f27]) + (f35 * ((-1 * f42) + (-1 * f43))) * multi_dot([E(f27).T,skew(f31).T,f33])),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbs_body_1_jcs_trans_1 = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_body_1),self.Mbar_rbs_body_1_jcs_trans_1[:,0:1]]),multi_dot([A(self.P_rbs_body_1),self.Mbar_rbs_body_1_jcs_trans_1[:,1:2]]),Z1x3.T],[multi_dot([B(self.P_rbs_body_1,self.Mbar_rbs_body_1_jcs_trans_1[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_trans_1[:,2:3]]),multi_dot([B(self.P_rbs_body_1,self.Mbar_rbs_body_1_jcs_trans_1[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcs_trans_1[:,2:3]]),(multi_dot([B(self.P_rbs_body_1,self.Mbar_rbs_body_1_jcs_trans_1[:,0:1]).T,((-1) * self.R_ground + multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_jcs_trans_1]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_jcs_trans_1]) + self.R_rbs_body_1)]) + multi_dot([B(self.P_rbs_body_1,self.ubar_rbs_body_1_jcs_trans_1).T,A(self.P_rbs_body_1),self.Mbar_rbs_body_1_jcs_trans_1[:,0:1]])),(multi_dot([B(self.P_rbs_body_1,self.Mbar_rbs_body_1_jcs_trans_1[:,1:2]).T,((-1) * self.R_ground + multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_jcs_trans_1]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_jcs_trans_1]) + self.R_rbs_body_1)]) + multi_dot([B(self.P_rbs_body_1,self.ubar_rbs_body_1_jcs_trans_1).T,A(self.P_rbs_body_1),self.Mbar_rbs_body_1_jcs_trans_1[:,1:2]])),multi_dot([B(self.P_rbs_body_1,self.Mbar_rbs_body_1_jcs_trans_1[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_trans_1[:,1:2]])]]),self.L_jcs_trans_1])
        self.F_rbs_body_1_jcs_trans_1 = Q_rbs_body_1_jcs_trans_1[0:3]
        Te_rbs_body_1_jcs_trans_1 = Q_rbs_body_1_jcs_trans_1[3:7]
        self.T_rbs_body_1_jcs_trans_1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_jcs_trans_1])),self.F_rbs_body_1_jcs_trans_1]) + (0.5) * multi_dot([E(self.P_rbs_body_1),Te_rbs_body_1_jcs_trans_1]))
        self.F_rbs_body_1_fas_TSDA_1 = (1.0/((multi_dot([((-1) * self.R_ground.T + multi_dot([self.ubar_rbs_body_1_fas_TSDA_1.T,A(self.P_rbs_body_1).T]) + (-1) * multi_dot([self.ubar_ground_fas_TSDA_1.T,A(self.P_ground).T]) + self.R_rbs_body_1.T),((-1) * self.R_ground + multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_fas_TSDA_1]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_fas_TSDA_1]) + self.R_rbs_body_1)]))**(1.0/2.0))[0] * (config.UF_fas_TSDA_1_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_ground.T + multi_dot([self.ubar_rbs_body_1_fas_TSDA_1.T,A(self.P_rbs_body_1).T]) + (-1) * multi_dot([self.ubar_ground_fas_TSDA_1.T,A(self.P_ground).T]) + self.R_rbs_body_1.T),((-1) * self.R_ground + multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_fas_TSDA_1]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_fas_TSDA_1]) + self.R_rbs_body_1)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_ground.T + multi_dot([self.ubar_rbs_body_1_fas_TSDA_1.T,A(self.P_rbs_body_1).T]) + (-1) * multi_dot([self.ubar_ground_fas_TSDA_1.T,A(self.P_ground).T]) + self.R_rbs_body_1.T),((-1) * self.Rd_ground + multi_dot([B(self.P_rbs_body_1,self.ubar_rbs_body_1_fas_TSDA_1),self.Pd_rbs_body_1]) + (-1) * multi_dot([B(self.P_ground,self.ubar_ground_fas_TSDA_1),self.Pd_ground]) + self.Rd_rbs_body_1)])) + config.UF_fas_TSDA_1_Fs((config.fas_TSDA_1_FL + (-1 * ((multi_dot([((-1) * self.R_ground.T + multi_dot([self.ubar_rbs_body_1_fas_TSDA_1.T,A(self.P_rbs_body_1).T]) + (-1) * multi_dot([self.ubar_ground_fas_TSDA_1.T,A(self.P_ground).T]) + self.R_rbs_body_1.T),((-1) * self.R_ground + multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_fas_TSDA_1]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_fas_TSDA_1]) + self.R_rbs_body_1)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_ground + multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_fas_TSDA_1]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_fas_TSDA_1]) + self.R_rbs_body_1)
        self.T_rbs_body_1_fas_TSDA_1 = Z3x1
        Q_rbs_body_2_jcs_trans_2 = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_body_2),self.Mbar_rbs_body_2_jcs_trans_2[:,0:1]]),multi_dot([A(self.P_rbs_body_2),self.Mbar_rbs_body_2_jcs_trans_2[:,1:2]]),Z1x3.T],[multi_dot([B(self.P_rbs_body_2,self.Mbar_rbs_body_2_jcs_trans_2[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_trans_2[:,2:3]]),multi_dot([B(self.P_rbs_body_2,self.Mbar_rbs_body_2_jcs_trans_2[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcs_trans_2[:,2:3]]),(multi_dot([B(self.P_rbs_body_2,self.Mbar_rbs_body_2_jcs_trans_2[:,0:1]).T,((-1) * self.R_ground + multi_dot([A(self.P_rbs_body_2),self.ubar_rbs_body_2_jcs_trans_2]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_jcs_trans_2]) + self.R_rbs_body_2)]) + multi_dot([B(self.P_rbs_body_2,self.ubar_rbs_body_2_jcs_trans_2).T,A(self.P_rbs_body_2),self.Mbar_rbs_body_2_jcs_trans_2[:,0:1]])),(multi_dot([B(self.P_rbs_body_2,self.Mbar_rbs_body_2_jcs_trans_2[:,1:2]).T,((-1) * self.R_ground + multi_dot([A(self.P_rbs_body_2),self.ubar_rbs_body_2_jcs_trans_2]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_jcs_trans_2]) + self.R_rbs_body_2)]) + multi_dot([B(self.P_rbs_body_2,self.ubar_rbs_body_2_jcs_trans_2).T,A(self.P_rbs_body_2),self.Mbar_rbs_body_2_jcs_trans_2[:,1:2]])),multi_dot([B(self.P_rbs_body_2,self.Mbar_rbs_body_2_jcs_trans_2[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_trans_2[:,1:2]])]]),self.L_jcs_trans_2])
        self.F_rbs_body_2_jcs_trans_2 = Q_rbs_body_2_jcs_trans_2[0:3]
        Te_rbs_body_2_jcs_trans_2 = Q_rbs_body_2_jcs_trans_2[3:7]
        self.T_rbs_body_2_jcs_trans_2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_body_2),self.ubar_rbs_body_2_jcs_trans_2])),self.F_rbs_body_2_jcs_trans_2]) + (0.5) * multi_dot([E(self.P_rbs_body_2),Te_rbs_body_2_jcs_trans_2]))
        self.F_rbs_body_2_fas_TSDA_2 = (1.0/((multi_dot([((-1) * self.R_rbs_body_1.T + multi_dot([self.ubar_rbs_body_2_fas_TSDA_2.T,A(self.P_rbs_body_2).T]) + (-1) * multi_dot([self.ubar_rbs_body_1_fas_TSDA_2.T,A(self.P_rbs_body_1).T]) + self.R_rbs_body_2.T),((-1) * self.R_rbs_body_1 + multi_dot([A(self.P_rbs_body_2),self.ubar_rbs_body_2_fas_TSDA_2]) + (-1) * multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_fas_TSDA_2]) + self.R_rbs_body_2)]))**(1.0/2.0))[0] * (config.UF_fas_TSDA_2_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_rbs_body_1.T + multi_dot([self.ubar_rbs_body_2_fas_TSDA_2.T,A(self.P_rbs_body_2).T]) + (-1) * multi_dot([self.ubar_rbs_body_1_fas_TSDA_2.T,A(self.P_rbs_body_1).T]) + self.R_rbs_body_2.T),((-1) * self.R_rbs_body_1 + multi_dot([A(self.P_rbs_body_2),self.ubar_rbs_body_2_fas_TSDA_2]) + (-1) * multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_fas_TSDA_2]) + self.R_rbs_body_2)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_rbs_body_1.T + multi_dot([self.ubar_rbs_body_2_fas_TSDA_2.T,A(self.P_rbs_body_2).T]) + (-1) * multi_dot([self.ubar_rbs_body_1_fas_TSDA_2.T,A(self.P_rbs_body_1).T]) + self.R_rbs_body_2.T),((-1) * self.Rd_rbs_body_1 + multi_dot([B(self.P_rbs_body_2,self.ubar_rbs_body_2_fas_TSDA_2),self.Pd_rbs_body_2]) + (-1) * multi_dot([B(self.P_rbs_body_1,self.ubar_rbs_body_1_fas_TSDA_2),self.Pd_rbs_body_1]) + self.Rd_rbs_body_2)])) + config.UF_fas_TSDA_2_Fs((config.fas_TSDA_2_FL + (-1 * ((multi_dot([((-1) * self.R_rbs_body_1.T + multi_dot([self.ubar_rbs_body_2_fas_TSDA_2.T,A(self.P_rbs_body_2).T]) + (-1) * multi_dot([self.ubar_rbs_body_1_fas_TSDA_2.T,A(self.P_rbs_body_1).T]) + self.R_rbs_body_2.T),((-1) * self.R_rbs_body_1 + multi_dot([A(self.P_rbs_body_2),self.ubar_rbs_body_2_fas_TSDA_2]) + (-1) * multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_fas_TSDA_2]) + self.R_rbs_body_2)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_rbs_body_1 + multi_dot([A(self.P_rbs_body_2),self.ubar_rbs_body_2_fas_TSDA_2]) + (-1) * multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_fas_TSDA_2]) + self.R_rbs_body_2)
        self.T_rbs_body_2_fas_TSDA_2 = Z3x1

        self.reactions = {'F_rbs_body_1_jcs_trans_1' : self.F_rbs_body_1_jcs_trans_1,
                        'T_rbs_body_1_jcs_trans_1' : self.T_rbs_body_1_jcs_trans_1,
                        'F_rbs_body_1_fas_TSDA_1' : self.F_rbs_body_1_fas_TSDA_1,
                        'T_rbs_body_1_fas_TSDA_1' : self.T_rbs_body_1_fas_TSDA_1,
                        'F_rbs_body_2_jcs_trans_2' : self.F_rbs_body_2_jcs_trans_2,
                        'T_rbs_body_2_jcs_trans_2' : self.T_rbs_body_2_jcs_trans_2,
                        'F_rbs_body_2_fas_TSDA_2' : self.F_rbs_body_2_fas_TSDA_2,
                        'T_rbs_body_2_fas_TSDA_2' : self.T_rbs_body_2_fas_TSDA_2}

