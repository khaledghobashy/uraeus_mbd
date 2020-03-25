
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
        self.nrows = 10
        self.ncols = 2*3
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_ground_jcs_a', 'T_ground_jcs_a', 'F_rbs_body_1_jcs_b', 'T_rbs_body_1_jcs_b']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.ground*2, self.ground*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.ground*2, self.ground*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.rbs_body_2*2, self.rbs_body_2*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.rbs_body_2*2, self.rbs_body_2*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.rbs_body_2*2, self.rbs_body_2*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_body_1*2, self.rbs_body_1*2+1, self.rbs_body_2*2, self.rbs_body_2*2+1], dtype=np.intc)

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
        self.F_rbs_body_2_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_body_2]], dtype=np.float64)

        self.Mbar_ground_jcs_a = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_body_1_jcs_a = multi_dot([A(config.P_rbs_body_1).T,triad(config.ax1_jcs_a)])
        self.ubar_ground_jcs_a = (multi_dot([A(self.P_ground).T,config.pt1_jcs_a]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_body_1_jcs_a = (multi_dot([A(config.P_rbs_body_1).T,config.pt1_jcs_a]) + (-1) * multi_dot([A(config.P_rbs_body_1).T,config.R_rbs_body_1]))
        self.Mbar_rbs_body_1_jcs_b = multi_dot([A(config.P_rbs_body_1).T,triad(config.ax1_jcs_b)])
        self.Mbar_rbs_body_2_jcs_b = multi_dot([A(config.P_rbs_body_2).T,triad(config.ax1_jcs_b)])
        self.ubar_rbs_body_1_jcs_b = (multi_dot([A(config.P_rbs_body_1).T,config.pt1_jcs_b]) + (-1) * multi_dot([A(config.P_rbs_body_1).T,config.R_rbs_body_1]))
        self.ubar_rbs_body_2_jcs_b = (multi_dot([A(config.P_rbs_body_2).T,config.pt1_jcs_b]) + (-1) * multi_dot([A(config.P_rbs_body_2).T,config.R_rbs_body_2]))

    
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
        self.L_jcs_a = Lambda[0:5]
        self.L_jcs_b = Lambda[5:10]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_ground
        x1 = self.R_rbs_body_1
        x2 = self.P_ground
        x3 = A(x2)
        x4 = self.P_rbs_body_1
        x5 = A(x4)
        x6 = x3.T
        x7 = self.Mbar_rbs_body_1_jcs_a[:,2:3]
        x8 = self.P_rbs_body_2
        x9 = A(x8)
        x10 = x5.T
        x11 = self.Mbar_rbs_body_2_jcs_b[:,2:3]
        x12 = (-1) * I1

        self.pos_eq_blocks = ((x0 + (-1) * x1 + multi_dot([x3,self.ubar_ground_jcs_a]) + (-1) * multi_dot([x5,self.ubar_rbs_body_1_jcs_a])),
        multi_dot([self.Mbar_ground_jcs_a[:,0:1].T,x6,x5,x7]),
        multi_dot([self.Mbar_ground_jcs_a[:,1:2].T,x6,x5,x7]),
        (x1 + (-1) * self.R_rbs_body_2 + multi_dot([x5,self.ubar_rbs_body_1_jcs_b]) + (-1) * multi_dot([x9,self.ubar_rbs_body_2_jcs_b])),
        multi_dot([self.Mbar_rbs_body_1_jcs_b[:,0:1].T,x10,x9,x11]),
        multi_dot([self.Mbar_rbs_body_1_jcs_b[:,1:2].T,x10,x9,x11]),
        x0,
        (x2 + (-1) * self.Pg_ground),
        (x12 + multi_dot([x4.T,x4])),
        (x12 + multi_dot([x8.T,x8])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z3x1
        v1 = Z1x1

        self.vel_eq_blocks = (v0,
        v1,
        v1,
        v0,
        v1,
        v1,
        v0,
        Z4x1,
        v1,
        v1,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_ground
        a1 = self.Pd_rbs_body_1
        a2 = self.Mbar_ground_jcs_a[:,0:1]
        a3 = self.P_ground
        a4 = A(a3).T
        a5 = self.Mbar_rbs_body_1_jcs_a[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_rbs_body_1
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_ground_jcs_a[:,1:2]
        a13 = self.Pd_rbs_body_2
        a14 = self.Mbar_rbs_body_1_jcs_b[:,0:1]
        a15 = self.Mbar_rbs_body_2_jcs_b[:,2:3]
        a16 = B(a13,a15)
        a17 = a15.T
        a18 = self.P_rbs_body_2
        a19 = A(a18).T
        a20 = a1.T
        a21 = B(a18,a15)
        a22 = self.Mbar_rbs_body_1_jcs_b[:,1:2]

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_ground_jcs_a),a0]) + (-1) * multi_dot([B(a1,self.ubar_rbs_body_1_jcs_a),a1])),
        (multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + (2) * multi_dot([a10,B(a3,a2).T,a11,a1])),
        (multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + (2) * multi_dot([a10,B(a3,a12).T,a11,a1])),
        (multi_dot([B(a1,self.ubar_rbs_body_1_jcs_b),a1]) + (-1) * multi_dot([B(a13,self.ubar_rbs_body_2_jcs_b),a13])),
        (multi_dot([a14.T,a9,a16,a13]) + multi_dot([a17,a19,B(a1,a14),a1]) + (2) * multi_dot([a20,B(a8,a14).T,a21,a13])),
        (multi_dot([a22.T,a9,a16,a13]) + multi_dot([a17,a19,B(a1,a22),a1]) + (2) * multi_dot([a20,B(a8,a22).T,a21,a13])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a20,a1]),
        (2) * multi_dot([a13.T,a13]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = I3
        j1 = self.P_ground
        j2 = Z1x3
        j3 = self.Mbar_rbs_body_1_jcs_a[:,2:3]
        j4 = j3.T
        j5 = self.P_rbs_body_1
        j6 = A(j5).T
        j7 = self.Mbar_ground_jcs_a[:,0:1]
        j8 = self.Mbar_ground_jcs_a[:,1:2]
        j9 = (-1) * j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_rbs_body_2_jcs_b[:,2:3]
        j13 = j12.T
        j14 = self.P_rbs_body_2
        j15 = A(j14).T
        j16 = self.Mbar_rbs_body_1_jcs_b[:,0:1]
        j17 = self.Mbar_rbs_body_1_jcs_b[:,1:2]
        j18 = B(j14,j12)

        self.jac_eq_blocks = (j0,
        B(j1,self.ubar_ground_jcs_a),
        j9,
        (-1) * B(j5,self.ubar_rbs_body_1_jcs_a),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j7.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j2,
        multi_dot([j8.T,j10,j11]),
        j0,
        B(j5,self.ubar_rbs_body_1_jcs_b),
        j9,
        (-1) * B(j14,self.ubar_rbs_body_2_jcs_b),
        j2,
        multi_dot([j13,j15,B(j5,j16)]),
        j2,
        multi_dot([j16.T,j6,j18]),
        j2,
        multi_dot([j13,j15,B(j5,j17)]),
        j2,
        multi_dot([j17.T,j6,j18]),
        j0,
        Z3x4,
        Z4x3,
        I4,
        j2,
        (2) * j5.T,
        j2,
        (2) * j14.T,)

    
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

        f0 = G(self.Pd_rbs_body_1)
        f1 = G(self.Pd_rbs_body_2)

        self.frc_eq_blocks = (Z3x1,
        Z4x1,
        self.F_rbs_body_1_gravity,
        (8) * multi_dot([f0.T,config.Jbar_rbs_body_1,f0,self.P_rbs_body_1]),
        self.F_rbs_body_2_gravity,
        (8) * multi_dot([f1.T,config.Jbar_rbs_body_2,f1,self.P_rbs_body_2]),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_ground_jcs_a = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_ground,self.ubar_ground_jcs_a).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,0:1]).T,A(self.P_rbs_body_1),self.Mbar_rbs_body_1_jcs_a[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,1:2]).T,A(self.P_rbs_body_1),self.Mbar_rbs_body_1_jcs_a[:,2:3]])]]),self.L_jcs_a])
        self.F_ground_jcs_a = Q_ground_jcs_a[0:3]
        Te_ground_jcs_a = Q_ground_jcs_a[3:7]
        self.T_ground_jcs_a = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_a])),self.F_ground_jcs_a]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_a]))
        Q_rbs_body_1_jcs_b = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_body_1,self.ubar_rbs_body_1_jcs_b).T,multi_dot([B(self.P_rbs_body_1,self.Mbar_rbs_body_1_jcs_b[:,0:1]).T,A(self.P_rbs_body_2),self.Mbar_rbs_body_2_jcs_b[:,2:3]]),multi_dot([B(self.P_rbs_body_1,self.Mbar_rbs_body_1_jcs_b[:,1:2]).T,A(self.P_rbs_body_2),self.Mbar_rbs_body_2_jcs_b[:,2:3]])]]),self.L_jcs_b])
        self.F_rbs_body_1_jcs_b = Q_rbs_body_1_jcs_b[0:3]
        Te_rbs_body_1_jcs_b = Q_rbs_body_1_jcs_b[3:7]
        self.T_rbs_body_1_jcs_b = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_body_1),self.ubar_rbs_body_1_jcs_b])),self.F_rbs_body_1_jcs_b]) + (0.5) * multi_dot([E(self.P_rbs_body_1),Te_rbs_body_1_jcs_b]))

        self.reactions = {'F_ground_jcs_a' : self.F_ground_jcs_a,
                        'T_ground_jcs_a' : self.T_ground_jcs_a,
                        'F_rbs_body_1_jcs_b' : self.F_rbs_body_1_jcs_b,
                        'T_rbs_body_1_jcs_b' : self.T_rbs_body_1_jcs_b}

