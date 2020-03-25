
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
        self.nrows = 6
        self.ncols = 2*2
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_ground_jcs_a', 'T_ground_jcs_a']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1], dtype=np.intc)

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

        self.Mbar_ground_jcs_a = multi_dot([A(self.P_ground).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_body_jcs_a = multi_dot([A(config.P_rbs_body).T,triad(config.ax1_jcs_a)])
        self.ubar_ground_jcs_a = (multi_dot([A(self.P_ground).T,config.pt1_jcs_a]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_body_jcs_a = (multi_dot([A(config.P_rbs_body).T,config.pt1_jcs_a]) + (-1) * multi_dot([A(config.P_rbs_body).T,config.R_rbs_body]))

    
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
        self.L_jcs_a = Lambda[0:5]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_ground
        x1 = self.P_ground
        x2 = A(x1)
        x3 = self.P_rbs_body
        x4 = A(x3)
        x5 = x2.T
        x6 = self.Mbar_rbs_body_jcs_a[:,2:3]

        self.pos_eq_blocks = ((x0 + (-1) * self.R_rbs_body + multi_dot([x2,self.ubar_ground_jcs_a]) + (-1) * multi_dot([x4,self.ubar_rbs_body_jcs_a])),
        multi_dot([self.Mbar_ground_jcs_a[:,0:1].T,x5,x4,x6]),
        multi_dot([self.Mbar_ground_jcs_a[:,1:2].T,x5,x4,x6]),
        x0,
        (x1 + (-1) * self.Pg_ground),
        ((-1) * I1 + multi_dot([x3.T,x3])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z3x1
        v1 = Z1x1

        self.vel_eq_blocks = (v0,
        v1,
        v1,
        v0,
        Z4x1,
        v1,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_ground
        a1 = self.Pd_rbs_body
        a2 = self.Mbar_ground_jcs_a[:,0:1]
        a3 = self.P_ground
        a4 = A(a3).T
        a5 = self.Mbar_rbs_body_jcs_a[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_rbs_body
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_ground_jcs_a[:,1:2]

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_ground_jcs_a),a0]) + (-1) * multi_dot([B(a1,self.ubar_rbs_body_jcs_a),a1])),
        (multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + (2) * multi_dot([a10,B(a3,a2).T,a11,a1])),
        (multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + (2) * multi_dot([a10,B(a3,a12).T,a11,a1])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a1.T,a1]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = I3
        j1 = self.P_ground
        j2 = Z1x3
        j3 = self.Mbar_rbs_body_jcs_a[:,2:3]
        j4 = j3.T
        j5 = self.P_rbs_body
        j6 = A(j5).T
        j7 = self.Mbar_ground_jcs_a[:,0:1]
        j8 = self.Mbar_ground_jcs_a[:,1:2]
        j9 = A(j1).T
        j10 = B(j5,j3)

        self.jac_eq_blocks = (j0,
        B(j1,self.ubar_ground_jcs_a),
        (-1) * j0,
        (-1) * B(j5,self.ubar_rbs_body_jcs_a),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j7.T,j9,j10]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j2,
        multi_dot([j8.T,j9,j10]),
        j0,
        Z3x4,
        Z4x3,
        I4,
        j2,
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

        f0 = G(self.Pd_rbs_body)

        self.frc_eq_blocks = (Z3x1,
        Z4x1,
        self.F_rbs_body_gravity,
        (8) * multi_dot([f0.T,config.Jbar_rbs_body,f0,self.P_rbs_body]),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_ground_jcs_a = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_ground,self.ubar_ground_jcs_a).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,0:1]).T,A(self.P_rbs_body),self.Mbar_rbs_body_jcs_a[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,1:2]).T,A(self.P_rbs_body),self.Mbar_rbs_body_jcs_a[:,2:3]])]]),self.L_jcs_a])
        self.F_ground_jcs_a = Q_ground_jcs_a[0:3]
        Te_ground_jcs_a = Q_ground_jcs_a[3:7]
        self.T_ground_jcs_a = ((-1) * multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_a])),self.F_ground_jcs_a]) + (0.5) * multi_dot([E(self.P_ground),Te_ground_jcs_a]))

        self.reactions = {'F_ground_jcs_a' : self.F_ground_jcs_a,
                        'T_ground_jcs_a' : self.T_ground_jcs_a}

