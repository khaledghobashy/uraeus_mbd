
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

        self.indicies_map = {'ground': 0, 'rbs_body': 1}

        self.n  = 14
        self.nc = 13
        self.nrows = 8
        self.ncols = 2*2
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_rbs_body_jcs_trans', 'T_rbs_body_jcs_trans', 'F_rbs_body_fas_TSDA', 'T_rbs_body_fas_TSDA']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1], dtype=np.intc)

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
        self.config.R_rbs_body,
        self.config.P_rbs_body], out=self._q)

        np.concatenate([self.config.Rd_ground,
        self.config.Pd_ground,
        self.config.Rd_rbs_body,
        self.config.Pd_rbs_body], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p + 'ground']
        self.rbs_body = indicies_map[p + 'rbs_body']
    

    
    def eval_constants(self):
        config = self.config

        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.m_ground = 1.0
        self.Jbar_ground = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.F_rbs_body_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_body]], dtype=np.float64)
        self.T_rbs_body_fas_TSDA = Z3x1
        self.T_ground_fas_TSDA = Z3x1

        self.Mbar_rbs_body_jcs_trans = multi_dot([A(config.P_rbs_body).T,triad(config.ax1_jcs_trans)])
        self.Mbar_ground_jcs_trans = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_trans)])
        self.ubar_rbs_body_jcs_trans = (multi_dot([A(config.P_rbs_body).T,config.pt1_jcs_trans]) + (-1) * multi_dot([A(config.P_rbs_body).T,config.R_rbs_body]))
        self.ubar_ground_jcs_trans = (multi_dot([A(self.P_ground).T,config.pt1_jcs_trans]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_body_fas_TSDA = (multi_dot([A(config.P_rbs_body).T,config.pt1_fas_TSDA]) + (-1) * multi_dot([A(config.P_rbs_body).T,config.R_rbs_body]))
        self.ubar_ground_fas_TSDA = (multi_dot([A(self.P_ground).T,config.pt2_fas_TSDA]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))

    
    def _map_gen_coordinates(self):
        q = self._q
        self.R_ground = q[0:3]
        self.P_ground = q[3:7]
        self.R_rbs_body = q[7:10]
        self.P_rbs_body = q[10:14]

    
    def _map_gen_velocities(self):
        qd = self._qd
        self.Rd_ground = qd[0:3]
        self.Pd_ground = qd[3:7]
        self.Rd_rbs_body = qd[7:10]
        self.Pd_rbs_body = qd[10:14]

    
    def _map_gen_accelerations(self):
        qdd = self._qdd
        self.Rdd_ground = qdd[0:3]
        self.Pdd_ground = qdd[3:7]
        self.Rdd_rbs_body = qdd[7:10]
        self.Pdd_rbs_body = qdd[10:14]

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcs_trans = Lambda[0:5]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.Mbar_rbs_body_jcs_trans[:,0:1].T
        x1 = self.P_rbs_body
        x2 = A(x1)
        x3 = x2.T
        x4 = self.P_ground
        x5 = A(x4)
        x6 = self.Mbar_ground_jcs_trans[:,2:3]
        x7 = self.Mbar_rbs_body_jcs_trans[:,1:2].T
        x8 = self.R_ground
        x9 = (self.R_rbs_body + (-1) * x8 + multi_dot([x2,self.ubar_rbs_body_jcs_trans]) + (-1) * multi_dot([x5,self.ubar_ground_jcs_trans]))

        self.pos_eq_blocks = (multi_dot([x0,x3,x5,x6]),
        multi_dot([x7,x3,x5,x6]),
        multi_dot([x0,x3,x9]),
        multi_dot([x7,x3,x9]),
        multi_dot([x0,x3,x5,self.Mbar_ground_jcs_trans[:,1:2]]),
        x8,
        (x4 + (-1) * self.Pg_ground),
        ((-1) * I1 + multi_dot([x1.T,x1])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z1x1

        self.vel_eq_blocks = (v0,
        v0,
        v0,
        v0,
        v0,
        Z3x1,
        Z4x1,
        v0,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Mbar_rbs_body_jcs_trans[:,0:1]
        a1 = a0.T
        a2 = self.P_rbs_body
        a3 = A(a2).T
        a4 = self.Pd_ground
        a5 = self.Mbar_ground_jcs_trans[:,2:3]
        a6 = B(a4,a5)
        a7 = a5.T
        a8 = self.P_ground
        a9 = A(a8).T
        a10 = self.Pd_rbs_body
        a11 = B(a10,a0)
        a12 = a10.T
        a13 = B(a2,a0).T
        a14 = B(a8,a5)
        a15 = self.Mbar_rbs_body_jcs_trans[:,1:2]
        a16 = a15.T
        a17 = B(a10,a15)
        a18 = B(a2,a15).T
        a19 = self.ubar_rbs_body_jcs_trans
        a20 = self.ubar_ground_jcs_trans
        a21 = (multi_dot([B(a10,a19),a10]) + (-1) * multi_dot([B(a4,a20),a4]))
        a22 = (self.Rd_rbs_body + (-1) * self.Rd_ground + multi_dot([B(a2,a19),a10]) + (-1) * multi_dot([B(a8,a20),a4]))
        a23 = (self.R_rbs_body.T + (-1) * self.R_ground.T + multi_dot([a19.T,a3]) + (-1) * multi_dot([a20.T,a9]))
        a24 = self.Mbar_ground_jcs_trans[:,1:2]

        self.acc_eq_blocks = ((multi_dot([a1,a3,a6,a4]) + multi_dot([a7,a9,a11,a10]) + (2) * multi_dot([a12,a13,a14,a4])),
        (multi_dot([a16,a3,a6,a4]) + multi_dot([a7,a9,a17,a10]) + (2) * multi_dot([a12,a18,a14,a4])),
        (multi_dot([a1,a3,a21]) + (2) * multi_dot([a12,a13,a22]) + multi_dot([a23,a11,a10])),
        (multi_dot([a16,a3,a21]) + (2) * multi_dot([a12,a18,a22]) + multi_dot([a23,a17,a10])),
        (multi_dot([a1,a3,B(a4,a24),a4]) + multi_dot([a24.T,a9,a11,a10]) + (2) * multi_dot([a12,a13,B(a8,a24),a4])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a12,a10]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = Z1x3
        j1 = self.Mbar_ground_jcs_trans[:,2:3]
        j2 = j1.T
        j3 = self.P_ground
        j4 = A(j3).T
        j5 = self.P_rbs_body
        j6 = self.Mbar_rbs_body_jcs_trans[:,0:1]
        j7 = B(j5,j6)
        j8 = self.Mbar_rbs_body_jcs_trans[:,1:2]
        j9 = B(j5,j8)
        j10 = j6.T
        j11 = A(j5).T
        j12 = multi_dot([j10,j11])
        j13 = self.ubar_rbs_body_jcs_trans
        j14 = B(j5,j13)
        j15 = self.ubar_ground_jcs_trans
        j16 = (self.R_rbs_body.T + (-1) * self.R_ground.T + multi_dot([j13.T,j11]) + (-1) * multi_dot([j15.T,j4]))
        j17 = j8.T
        j18 = multi_dot([j17,j11])
        j19 = self.Mbar_ground_jcs_trans[:,1:2]
        j20 = B(j3,j1)
        j21 = B(j3,j15)

        self.jac_eq_blocks = (j0,
        multi_dot([j10,j11,j20]),
        j0,
        multi_dot([j2,j4,j7]),
        j0,
        multi_dot([j17,j11,j20]),
        j0,
        multi_dot([j2,j4,j9]),
        (-1) * j12,
        (-1) * multi_dot([j10,j11,j21]),
        j12,
        (multi_dot([j10,j11,j14]) + multi_dot([j16,j7])),
        (-1) * j18,
        (-1) * multi_dot([j17,j11,j21]),
        j18,
        (multi_dot([j17,j11,j14]) + multi_dot([j16,j9])),
        j0,
        multi_dot([j10,j11,B(j3,j19)]),
        j0,
        multi_dot([j19.T,j4,j7]),
        I3,
        Z3x4,
        Z4x3,
        I4,
        j0,
        (2) * j5.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_ground)
        m2 = G(self.P_rbs_body)

        self.mass_eq_blocks = (self.m_ground * m0,
        (4) * multi_dot([m1.T,self.Jbar_ground,m1]),
        config.m_rbs_body * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbs_body,m2]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = self.R_rbs_body
        f1 = self.R_ground
        f2 = self.ubar_rbs_body_fas_TSDA
        f3 = self.P_rbs_body
        f4 = A(f3)
        f5 = self.ubar_ground_fas_TSDA
        f6 = self.P_ground
        f7 = A(f6)
        f8 = (f0.T + (-1) * f1.T + multi_dot([f2.T,f4.T]) + (-1) * multi_dot([f5.T,f7.T]))
        f9 = multi_dot([f4,f2])
        f10 = multi_dot([f7,f5])
        f11 = (f0 + (-1) * f1 + f9 + (-1) * f10)
        f12 = ((multi_dot([f8,f11]))**(1.0/2.0))[0]
        f13 = 1.0/f12
        f14 = config.UF_fas_TSDA_Fs((config.fas_TSDA_FL + (-1 * f12)))
        f15 = self.Pd_rbs_body
        f16 = config.UF_fas_TSDA_Fd((-1 * 1.0/f12) * multi_dot([f8,(self.Rd_rbs_body + (-1) * self.Rd_ground + multi_dot([B(f3,f2),f15]) + (-1) * multi_dot([B(f6,f5),self.Pd_ground]))]))
        f17 = (f13 * (f14 + f16)) * f11
        f18 = (2 * f14)
        f19 = (2 * f16)
        f20 = G(f15)

        self.frc_eq_blocks = ((Z3x1 + (-1) * f17),
        (Z4x1 + (f13 * (f18 + f19)) * multi_dot([E(f6).T,skew(f10).T,f11])),
        (self.F_rbs_body_gravity + f17),
        ((8) * multi_dot([f20.T,config.Jbar_rbs_body,f20,f3]) + (f13 * ((-1 * f18) + (-1 * f19))) * multi_dot([E(f3).T,skew(f9).T,f11])),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbs_body_jcs_trans = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_body),self.Mbar_rbs_body_jcs_trans[:,0:1]]),multi_dot([A(self.P_rbs_body),self.Mbar_rbs_body_jcs_trans[:,1:2]]),Z1x3.T],[multi_dot([B(self.P_rbs_body,self.Mbar_rbs_body_jcs_trans[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_trans[:,2:3]]),multi_dot([B(self.P_rbs_body,self.Mbar_rbs_body_jcs_trans[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcs_trans[:,2:3]]),(multi_dot([B(self.P_rbs_body,self.Mbar_rbs_body_jcs_trans[:,0:1]).T,((-1) * self.R_ground + multi_dot([A(self.P_rbs_body),self.ubar_rbs_body_jcs_trans]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_jcs_trans]) + self.R_rbs_body)]) + multi_dot([B(self.P_rbs_body,self.ubar_rbs_body_jcs_trans).T,A(self.P_rbs_body),self.Mbar_rbs_body_jcs_trans[:,0:1]])),(multi_dot([B(self.P_rbs_body,self.Mbar_rbs_body_jcs_trans[:,1:2]).T,((-1) * self.R_ground + multi_dot([A(self.P_rbs_body),self.ubar_rbs_body_jcs_trans]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_jcs_trans]) + self.R_rbs_body)]) + multi_dot([B(self.P_rbs_body,self.ubar_rbs_body_jcs_trans).T,A(self.P_rbs_body),self.Mbar_rbs_body_jcs_trans[:,1:2]])),multi_dot([B(self.P_rbs_body,self.Mbar_rbs_body_jcs_trans[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_trans[:,1:2]])]]),self.L_jcs_trans])
        self.F_rbs_body_jcs_trans = Q_rbs_body_jcs_trans[0:3]
        Te_rbs_body_jcs_trans = Q_rbs_body_jcs_trans[3:7]
        self.T_rbs_body_jcs_trans = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_body),self.ubar_rbs_body_jcs_trans])),self.F_rbs_body_jcs_trans]) + (0.5) * multi_dot([E(self.P_rbs_body),Te_rbs_body_jcs_trans]))
        self.F_rbs_body_fas_TSDA = (1.0/((multi_dot([((-1) * self.R_ground.T + multi_dot([self.ubar_rbs_body_fas_TSDA.T,A(self.P_rbs_body).T]) + (-1) * multi_dot([self.ubar_ground_fas_TSDA.T,A(self.P_ground).T]) + self.R_rbs_body.T),((-1) * self.R_ground + multi_dot([A(self.P_rbs_body),self.ubar_rbs_body_fas_TSDA]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_fas_TSDA]) + self.R_rbs_body)]))**(1.0/2.0))[0] * (config.UF_fas_TSDA_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_ground.T + multi_dot([self.ubar_rbs_body_fas_TSDA.T,A(self.P_rbs_body).T]) + (-1) * multi_dot([self.ubar_ground_fas_TSDA.T,A(self.P_ground).T]) + self.R_rbs_body.T),((-1) * self.R_ground + multi_dot([A(self.P_rbs_body),self.ubar_rbs_body_fas_TSDA]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_fas_TSDA]) + self.R_rbs_body)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_ground.T + multi_dot([self.ubar_rbs_body_fas_TSDA.T,A(self.P_rbs_body).T]) + (-1) * multi_dot([self.ubar_ground_fas_TSDA.T,A(self.P_ground).T]) + self.R_rbs_body.T),((-1) * self.Rd_ground + multi_dot([B(self.P_rbs_body,self.ubar_rbs_body_fas_TSDA),self.Pd_rbs_body]) + (-1) * multi_dot([B(self.P_ground,self.ubar_ground_fas_TSDA),self.Pd_ground]) + self.Rd_rbs_body)])) + config.UF_fas_TSDA_Fs((config.fas_TSDA_FL + (-1 * ((multi_dot([((-1) * self.R_ground.T + multi_dot([self.ubar_rbs_body_fas_TSDA.T,A(self.P_rbs_body).T]) + (-1) * multi_dot([self.ubar_ground_fas_TSDA.T,A(self.P_ground).T]) + self.R_rbs_body.T),((-1) * self.R_ground + multi_dot([A(self.P_rbs_body),self.ubar_rbs_body_fas_TSDA]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_fas_TSDA]) + self.R_rbs_body)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_ground + multi_dot([A(self.P_rbs_body),self.ubar_rbs_body_fas_TSDA]) + (-1) * multi_dot([A(self.P_ground),self.ubar_ground_fas_TSDA]) + self.R_rbs_body)
        self.T_rbs_body_fas_TSDA = Z3x1

        self.reactions = {'F_rbs_body_jcs_trans' : self.F_rbs_body_jcs_trans,
                        'T_rbs_body_jcs_trans' : self.T_rbs_body_jcs_trans,
                        'F_rbs_body_fas_TSDA' : self.F_rbs_body_fas_TSDA,
                        'T_rbs_body_fas_TSDA' : self.T_rbs_body_fas_TSDA}

