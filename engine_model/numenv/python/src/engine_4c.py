
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

        self.indicies_map = {'ground': 0, 'rbs_piston_1': 1, 'rbs_piston_2': 2, 'rbs_piston_3': 3, 'rbs_piston_4': 4, 'rbs_connect_1': 5, 'rbs_connect_2': 6, 'rbs_connect_3': 7, 'rbs_connect_4': 8, 'rbs_crank_shaft': 9, 'rbs_engine_block': 10}

        self.n  = 77
        self.nc = 62
        self.nrows = 39
        self.ncols = 2*11
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_ground_fas_bush_1', 'T_ground_fas_bush_1', 'F_ground_fas_bush_2', 'T_ground_fas_bush_2', 'F_ground_fas_bush_3', 'T_ground_fas_bush_3', 'F_ground_fas_bush_4', 'T_ground_fas_bush_4', 'F_rbs_piston_1_jcs_cyl_1', 'T_rbs_piston_1_jcs_cyl_1', 'F_rbs_piston_1_jcs_sph_1', 'T_rbs_piston_1_jcs_sph_1', 'F_rbs_piston_2_jcs_cyl_2', 'T_rbs_piston_2_jcs_cyl_2', 'F_rbs_piston_2_jcs_sph_2', 'T_rbs_piston_2_jcs_sph_2', 'F_rbs_piston_3_jcs_cyl_3', 'T_rbs_piston_3_jcs_cyl_3', 'F_rbs_piston_3_jcs_sph_3', 'T_rbs_piston_3_jcs_sph_3', 'F_rbs_piston_4_jcs_cyl_4', 'T_rbs_piston_4_jcs_cyl_4', 'F_rbs_piston_4_jcs_sph_4', 'T_rbs_piston_4_jcs_sph_4', 'F_rbs_crank_shaft_jcs_bsph_1', 'T_rbs_crank_shaft_jcs_bsph_1', 'F_rbs_crank_shaft_jcs_bsph_2', 'T_rbs_crank_shaft_jcs_bsph_2', 'F_rbs_crank_shaft_jcs_bsph_3', 'T_rbs_crank_shaft_jcs_bsph_3', 'F_rbs_crank_shaft_jcs_bsph_4', 'T_rbs_crank_shaft_jcs_bsph_4', 'F_rbs_crank_shaft_jcs_crank_joint', 'T_rbs_crank_shaft_jcs_crank_joint']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbs_piston_1*2, self.rbs_piston_1*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_1*2, self.rbs_piston_1*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_1*2, self.rbs_piston_1*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_1*2, self.rbs_piston_1*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_1*2, self.rbs_piston_1*2+1, self.rbs_connect_1*2, self.rbs_connect_1*2+1, self.rbs_piston_2*2, self.rbs_piston_2*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_2*2, self.rbs_piston_2*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_2*2, self.rbs_piston_2*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_2*2, self.rbs_piston_2*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_2*2, self.rbs_piston_2*2+1, self.rbs_connect_2*2, self.rbs_connect_2*2+1, self.rbs_piston_3*2, self.rbs_piston_3*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_3*2, self.rbs_piston_3*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_3*2, self.rbs_piston_3*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_3*2, self.rbs_piston_3*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_3*2, self.rbs_piston_3*2+1, self.rbs_connect_3*2, self.rbs_connect_3*2+1, self.rbs_piston_4*2, self.rbs_piston_4*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_4*2, self.rbs_piston_4*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_4*2, self.rbs_piston_4*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_4*2, self.rbs_piston_4*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_piston_4*2, self.rbs_piston_4*2+1, self.rbs_connect_4*2, self.rbs_connect_4*2+1, self.rbs_connect_1*2, self.rbs_connect_1*2+1, self.rbs_crank_shaft*2, self.rbs_crank_shaft*2+1, self.rbs_connect_2*2, self.rbs_connect_2*2+1, self.rbs_crank_shaft*2, self.rbs_crank_shaft*2+1, self.rbs_connect_3*2, self.rbs_connect_3*2+1, self.rbs_crank_shaft*2, self.rbs_crank_shaft*2+1, self.rbs_connect_4*2, self.rbs_connect_4*2+1, self.rbs_crank_shaft*2, self.rbs_crank_shaft*2+1, self.rbs_crank_shaft*2, self.rbs_crank_shaft*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_crank_shaft*2, self.rbs_crank_shaft*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.rbs_crank_shaft*2, self.rbs_crank_shaft*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_piston_1*2, self.rbs_piston_1*2+1, self.rbs_piston_2*2, self.rbs_piston_2*2+1, self.rbs_piston_3*2, self.rbs_piston_3*2+1, self.rbs_piston_4*2, self.rbs_piston_4*2+1, self.rbs_connect_1*2, self.rbs_connect_1*2+1, self.rbs_connect_2*2, self.rbs_connect_2*2+1, self.rbs_connect_3*2, self.rbs_connect_3*2+1, self.rbs_connect_4*2, self.rbs_connect_4*2+1, self.rbs_crank_shaft*2, self.rbs_crank_shaft*2+1, self.rbs_engine_block*2, self.rbs_engine_block*2+1], dtype=np.intc)

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
        self.config.R_rbs_piston_1,
        self.config.P_rbs_piston_1,
        self.config.R_rbs_piston_2,
        self.config.P_rbs_piston_2,
        self.config.R_rbs_piston_3,
        self.config.P_rbs_piston_3,
        self.config.R_rbs_piston_4,
        self.config.P_rbs_piston_4,
        self.config.R_rbs_connect_1,
        self.config.P_rbs_connect_1,
        self.config.R_rbs_connect_2,
        self.config.P_rbs_connect_2,
        self.config.R_rbs_connect_3,
        self.config.P_rbs_connect_3,
        self.config.R_rbs_connect_4,
        self.config.P_rbs_connect_4,
        self.config.R_rbs_crank_shaft,
        self.config.P_rbs_crank_shaft,
        self.config.R_rbs_engine_block,
        self.config.P_rbs_engine_block], out=self._q)

        np.concatenate([self.config.Rd_ground,
        self.config.Pd_ground,
        self.config.Rd_rbs_piston_1,
        self.config.Pd_rbs_piston_1,
        self.config.Rd_rbs_piston_2,
        self.config.Pd_rbs_piston_2,
        self.config.Rd_rbs_piston_3,
        self.config.Pd_rbs_piston_3,
        self.config.Rd_rbs_piston_4,
        self.config.Pd_rbs_piston_4,
        self.config.Rd_rbs_connect_1,
        self.config.Pd_rbs_connect_1,
        self.config.Rd_rbs_connect_2,
        self.config.Pd_rbs_connect_2,
        self.config.Rd_rbs_connect_3,
        self.config.Pd_rbs_connect_3,
        self.config.Rd_rbs_connect_4,
        self.config.Pd_rbs_connect_4,
        self.config.Rd_rbs_crank_shaft,
        self.config.Pd_rbs_crank_shaft,
        self.config.Rd_rbs_engine_block,
        self.config.Pd_rbs_engine_block], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p + 'ground']
        self.rbs_piston_1 = indicies_map[p + 'rbs_piston_1']
        self.rbs_piston_2 = indicies_map[p + 'rbs_piston_2']
        self.rbs_piston_3 = indicies_map[p + 'rbs_piston_3']
        self.rbs_piston_4 = indicies_map[p + 'rbs_piston_4']
        self.rbs_connect_1 = indicies_map[p + 'rbs_connect_1']
        self.rbs_connect_2 = indicies_map[p + 'rbs_connect_2']
        self.rbs_connect_3 = indicies_map[p + 'rbs_connect_3']
        self.rbs_connect_4 = indicies_map[p + 'rbs_connect_4']
        self.rbs_crank_shaft = indicies_map[p + 'rbs_crank_shaft']
        self.rbs_engine_block = indicies_map[p + 'rbs_engine_block']
    

    
    def eval_constants(self):
        config = self.config

        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.m_ground = 1.0
        self.Jbar_ground = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.F_rbs_piston_1_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_piston_1]], dtype=np.float64)
        self.F_rbs_piston_2_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_piston_2]], dtype=np.float64)
        self.F_rbs_piston_3_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_piston_3]], dtype=np.float64)
        self.F_rbs_piston_4_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_piston_4]], dtype=np.float64)
        self.F_rbs_connect_1_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_connect_1]], dtype=np.float64)
        self.F_rbs_connect_2_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_connect_2]], dtype=np.float64)
        self.F_rbs_connect_3_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_connect_3]], dtype=np.float64)
        self.F_rbs_connect_4_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_connect_4]], dtype=np.float64)
        self.F_rbs_crank_shaft_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_crank_shaft]], dtype=np.float64)
        self.F_rbs_crank_shaft_fas_drive = Z3x1
        self.F_rbs_engine_block_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_engine_block]], dtype=np.float64)

        self.vbar_ground_fas_bush_1 = multi_dot([A(self.P_ground).T,config.ax1_fas_bush_1,(multi_dot([config.ax1_fas_bush_1.T,A(self.P_ground),A(self.P_ground).T,config.ax1_fas_bush_1]))**(-1.0/2.0)])
        self.Mbar_ground_fas_bush_1 = multi_dot([A(self.P_ground).T,triad(config.ax1_fas_bush_1)])
        self.Mbar_rbs_engine_block_fas_bush_1 = multi_dot([A(config.P_rbs_engine_block).T,triad(config.ax1_fas_bush_1)])
        self.ubar_ground_fas_bush_1 = (multi_dot([A(self.P_ground).T,config.pt1_fas_bush_1]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_engine_block_fas_bush_1 = (multi_dot([A(config.P_rbs_engine_block).T,config.pt1_fas_bush_1]) + (-1) * multi_dot([A(config.P_rbs_engine_block).T,config.R_rbs_engine_block]))
        self.vbar_ground_fas_bush_2 = multi_dot([A(self.P_ground).T,config.ax1_fas_bush_2,(multi_dot([config.ax1_fas_bush_2.T,A(self.P_ground),A(self.P_ground).T,config.ax1_fas_bush_2]))**(-1.0/2.0)])
        self.Mbar_ground_fas_bush_2 = multi_dot([A(self.P_ground).T,triad(config.ax1_fas_bush_2)])
        self.Mbar_rbs_engine_block_fas_bush_2 = multi_dot([A(config.P_rbs_engine_block).T,triad(config.ax1_fas_bush_2)])
        self.ubar_ground_fas_bush_2 = (multi_dot([A(self.P_ground).T,config.pt1_fas_bush_2]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_engine_block_fas_bush_2 = (multi_dot([A(config.P_rbs_engine_block).T,config.pt1_fas_bush_2]) + (-1) * multi_dot([A(config.P_rbs_engine_block).T,config.R_rbs_engine_block]))
        self.vbar_ground_fas_bush_3 = multi_dot([A(self.P_ground).T,config.ax1_fas_bush_3,(multi_dot([config.ax1_fas_bush_3.T,A(self.P_ground),A(self.P_ground).T,config.ax1_fas_bush_3]))**(-1.0/2.0)])
        self.Mbar_ground_fas_bush_3 = multi_dot([A(self.P_ground).T,triad(config.ax1_fas_bush_3)])
        self.Mbar_rbs_engine_block_fas_bush_3 = multi_dot([A(config.P_rbs_engine_block).T,triad(config.ax1_fas_bush_3)])
        self.ubar_ground_fas_bush_3 = (multi_dot([A(self.P_ground).T,config.pt1_fas_bush_3]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_engine_block_fas_bush_3 = (multi_dot([A(config.P_rbs_engine_block).T,config.pt1_fas_bush_3]) + (-1) * multi_dot([A(config.P_rbs_engine_block).T,config.R_rbs_engine_block]))
        self.vbar_ground_fas_bush_4 = multi_dot([A(self.P_ground).T,config.ax1_fas_bush_4,(multi_dot([config.ax1_fas_bush_4.T,A(self.P_ground),A(self.P_ground).T,config.ax1_fas_bush_4]))**(-1.0/2.0)])
        self.Mbar_ground_fas_bush_4 = multi_dot([A(self.P_ground).T,triad(config.ax1_fas_bush_4)])
        self.Mbar_rbs_engine_block_fas_bush_4 = multi_dot([A(config.P_rbs_engine_block).T,triad(config.ax1_fas_bush_4)])
        self.ubar_ground_fas_bush_4 = (multi_dot([A(self.P_ground).T,config.pt1_fas_bush_4]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbs_engine_block_fas_bush_4 = (multi_dot([A(config.P_rbs_engine_block).T,config.pt1_fas_bush_4]) + (-1) * multi_dot([A(config.P_rbs_engine_block).T,config.R_rbs_engine_block]))
        self.Mbar_rbs_piston_1_jcs_cyl_1 = multi_dot([A(config.P_rbs_piston_1).T,triad(config.ax1_jcs_cyl_1)])
        self.Mbar_rbs_engine_block_jcs_cyl_1 = multi_dot([A(config.P_rbs_engine_block).T,triad(config.ax1_jcs_cyl_1)])
        self.ubar_rbs_piston_1_jcs_cyl_1 = (multi_dot([A(config.P_rbs_piston_1).T,config.pt1_jcs_cyl_1]) + (-1) * multi_dot([A(config.P_rbs_piston_1).T,config.R_rbs_piston_1]))
        self.ubar_rbs_engine_block_jcs_cyl_1 = (multi_dot([A(config.P_rbs_engine_block).T,config.pt1_jcs_cyl_1]) + (-1) * multi_dot([A(config.P_rbs_engine_block).T,config.R_rbs_engine_block]))
        self.Mbar_rbs_piston_1_jcs_sph_1 = multi_dot([A(config.P_rbs_piston_1).T,triad(config.ax1_jcs_sph_1)])
        self.Mbar_rbs_connect_1_jcs_sph_1 = multi_dot([A(config.P_rbs_connect_1).T,triad(config.ax1_jcs_sph_1)])
        self.ubar_rbs_piston_1_jcs_sph_1 = (multi_dot([A(config.P_rbs_piston_1).T,config.pt1_jcs_sph_1]) + (-1) * multi_dot([A(config.P_rbs_piston_1).T,config.R_rbs_piston_1]))
        self.ubar_rbs_connect_1_jcs_sph_1 = (multi_dot([A(config.P_rbs_connect_1).T,config.pt1_jcs_sph_1]) + (-1) * multi_dot([A(config.P_rbs_connect_1).T,config.R_rbs_connect_1]))
        self.Mbar_rbs_piston_2_jcs_cyl_2 = multi_dot([A(config.P_rbs_piston_2).T,triad(config.ax1_jcs_cyl_2)])
        self.Mbar_rbs_engine_block_jcs_cyl_2 = multi_dot([A(config.P_rbs_engine_block).T,triad(config.ax1_jcs_cyl_2)])
        self.ubar_rbs_piston_2_jcs_cyl_2 = (multi_dot([A(config.P_rbs_piston_2).T,config.pt1_jcs_cyl_2]) + (-1) * multi_dot([A(config.P_rbs_piston_2).T,config.R_rbs_piston_2]))
        self.ubar_rbs_engine_block_jcs_cyl_2 = (multi_dot([A(config.P_rbs_engine_block).T,config.pt1_jcs_cyl_2]) + (-1) * multi_dot([A(config.P_rbs_engine_block).T,config.R_rbs_engine_block]))
        self.Mbar_rbs_piston_2_jcs_sph_2 = multi_dot([A(config.P_rbs_piston_2).T,triad(config.ax1_jcs_sph_2)])
        self.Mbar_rbs_connect_2_jcs_sph_2 = multi_dot([A(config.P_rbs_connect_2).T,triad(config.ax1_jcs_sph_2)])
        self.ubar_rbs_piston_2_jcs_sph_2 = (multi_dot([A(config.P_rbs_piston_2).T,config.pt1_jcs_sph_2]) + (-1) * multi_dot([A(config.P_rbs_piston_2).T,config.R_rbs_piston_2]))
        self.ubar_rbs_connect_2_jcs_sph_2 = (multi_dot([A(config.P_rbs_connect_2).T,config.pt1_jcs_sph_2]) + (-1) * multi_dot([A(config.P_rbs_connect_2).T,config.R_rbs_connect_2]))
        self.Mbar_rbs_piston_3_jcs_cyl_3 = multi_dot([A(config.P_rbs_piston_3).T,triad(config.ax1_jcs_cyl_3)])
        self.Mbar_rbs_engine_block_jcs_cyl_3 = multi_dot([A(config.P_rbs_engine_block).T,triad(config.ax1_jcs_cyl_3)])
        self.ubar_rbs_piston_3_jcs_cyl_3 = (multi_dot([A(config.P_rbs_piston_3).T,config.pt1_jcs_cyl_3]) + (-1) * multi_dot([A(config.P_rbs_piston_3).T,config.R_rbs_piston_3]))
        self.ubar_rbs_engine_block_jcs_cyl_3 = (multi_dot([A(config.P_rbs_engine_block).T,config.pt1_jcs_cyl_3]) + (-1) * multi_dot([A(config.P_rbs_engine_block).T,config.R_rbs_engine_block]))
        self.Mbar_rbs_piston_3_jcs_sph_3 = multi_dot([A(config.P_rbs_piston_3).T,triad(config.ax1_jcs_sph_3)])
        self.Mbar_rbs_connect_3_jcs_sph_3 = multi_dot([A(config.P_rbs_connect_3).T,triad(config.ax1_jcs_sph_3)])
        self.ubar_rbs_piston_3_jcs_sph_3 = (multi_dot([A(config.P_rbs_piston_3).T,config.pt1_jcs_sph_3]) + (-1) * multi_dot([A(config.P_rbs_piston_3).T,config.R_rbs_piston_3]))
        self.ubar_rbs_connect_3_jcs_sph_3 = (multi_dot([A(config.P_rbs_connect_3).T,config.pt1_jcs_sph_3]) + (-1) * multi_dot([A(config.P_rbs_connect_3).T,config.R_rbs_connect_3]))
        self.Mbar_rbs_piston_4_jcs_cyl_4 = multi_dot([A(config.P_rbs_piston_4).T,triad(config.ax1_jcs_cyl_4)])
        self.Mbar_rbs_engine_block_jcs_cyl_4 = multi_dot([A(config.P_rbs_engine_block).T,triad(config.ax1_jcs_cyl_4)])
        self.ubar_rbs_piston_4_jcs_cyl_4 = (multi_dot([A(config.P_rbs_piston_4).T,config.pt1_jcs_cyl_4]) + (-1) * multi_dot([A(config.P_rbs_piston_4).T,config.R_rbs_piston_4]))
        self.ubar_rbs_engine_block_jcs_cyl_4 = (multi_dot([A(config.P_rbs_engine_block).T,config.pt1_jcs_cyl_4]) + (-1) * multi_dot([A(config.P_rbs_engine_block).T,config.R_rbs_engine_block]))
        self.Mbar_rbs_piston_4_jcs_sph_4 = multi_dot([A(config.P_rbs_piston_4).T,triad(config.ax1_jcs_sph_4)])
        self.Mbar_rbs_connect_4_jcs_sph_4 = multi_dot([A(config.P_rbs_connect_4).T,triad(config.ax1_jcs_sph_4)])
        self.ubar_rbs_piston_4_jcs_sph_4 = (multi_dot([A(config.P_rbs_piston_4).T,config.pt1_jcs_sph_4]) + (-1) * multi_dot([A(config.P_rbs_piston_4).T,config.R_rbs_piston_4]))
        self.ubar_rbs_connect_4_jcs_sph_4 = (multi_dot([A(config.P_rbs_connect_4).T,config.pt1_jcs_sph_4]) + (-1) * multi_dot([A(config.P_rbs_connect_4).T,config.R_rbs_connect_4]))
        self.vbar_rbs_crank_shaft_fas_drive = multi_dot([A(config.P_rbs_crank_shaft).T,config.ax1_fas_drive,(multi_dot([config.ax1_fas_drive.T,A(config.P_rbs_crank_shaft),A(config.P_rbs_crank_shaft).T,config.ax1_fas_drive]))**(-1.0/2.0)])
        self.Mbar_rbs_crank_shaft_fas_drive = multi_dot([A(config.P_rbs_crank_shaft).T,triad(config.ax1_fas_drive)])
        self.Mbar_ground_fas_drive = multi_dot([A(self.P_ground).T,triad(config.ax1_fas_drive)])
        self.Mbar_rbs_crank_shaft_jcs_bsph_1 = multi_dot([A(config.P_rbs_crank_shaft).T,triad(config.ax1_jcs_bsph_1)])
        self.Mbar_rbs_connect_1_jcs_bsph_1 = multi_dot([A(config.P_rbs_connect_1).T,triad(config.ax1_jcs_bsph_1)])
        self.ubar_rbs_crank_shaft_jcs_bsph_1 = (multi_dot([A(config.P_rbs_crank_shaft).T,config.pt1_jcs_bsph_1]) + (-1) * multi_dot([A(config.P_rbs_crank_shaft).T,config.R_rbs_crank_shaft]))
        self.ubar_rbs_connect_1_jcs_bsph_1 = (multi_dot([A(config.P_rbs_connect_1).T,config.pt1_jcs_bsph_1]) + (-1) * multi_dot([A(config.P_rbs_connect_1).T,config.R_rbs_connect_1]))
        self.Mbar_rbs_crank_shaft_jcs_bsph_2 = multi_dot([A(config.P_rbs_crank_shaft).T,triad(config.ax1_jcs_bsph_2)])
        self.Mbar_rbs_connect_2_jcs_bsph_2 = multi_dot([A(config.P_rbs_connect_2).T,triad(config.ax1_jcs_bsph_2)])
        self.ubar_rbs_crank_shaft_jcs_bsph_2 = (multi_dot([A(config.P_rbs_crank_shaft).T,config.pt1_jcs_bsph_2]) + (-1) * multi_dot([A(config.P_rbs_crank_shaft).T,config.R_rbs_crank_shaft]))
        self.ubar_rbs_connect_2_jcs_bsph_2 = (multi_dot([A(config.P_rbs_connect_2).T,config.pt1_jcs_bsph_2]) + (-1) * multi_dot([A(config.P_rbs_connect_2).T,config.R_rbs_connect_2]))
        self.Mbar_rbs_crank_shaft_jcs_bsph_3 = multi_dot([A(config.P_rbs_crank_shaft).T,triad(config.ax1_jcs_bsph_3)])
        self.Mbar_rbs_connect_3_jcs_bsph_3 = multi_dot([A(config.P_rbs_connect_3).T,triad(config.ax1_jcs_bsph_3)])
        self.ubar_rbs_crank_shaft_jcs_bsph_3 = (multi_dot([A(config.P_rbs_crank_shaft).T,config.pt1_jcs_bsph_3]) + (-1) * multi_dot([A(config.P_rbs_crank_shaft).T,config.R_rbs_crank_shaft]))
        self.ubar_rbs_connect_3_jcs_bsph_3 = (multi_dot([A(config.P_rbs_connect_3).T,config.pt1_jcs_bsph_3]) + (-1) * multi_dot([A(config.P_rbs_connect_3).T,config.R_rbs_connect_3]))
        self.Mbar_rbs_crank_shaft_jcs_bsph_4 = multi_dot([A(config.P_rbs_crank_shaft).T,triad(config.ax1_jcs_bsph_4)])
        self.Mbar_rbs_connect_4_jcs_bsph_4 = multi_dot([A(config.P_rbs_connect_4).T,triad(config.ax1_jcs_bsph_4)])
        self.ubar_rbs_crank_shaft_jcs_bsph_4 = (multi_dot([A(config.P_rbs_crank_shaft).T,config.pt1_jcs_bsph_4]) + (-1) * multi_dot([A(config.P_rbs_crank_shaft).T,config.R_rbs_crank_shaft]))
        self.ubar_rbs_connect_4_jcs_bsph_4 = (multi_dot([A(config.P_rbs_connect_4).T,config.pt1_jcs_bsph_4]) + (-1) * multi_dot([A(config.P_rbs_connect_4).T,config.R_rbs_connect_4]))
        self.Mbar_rbs_crank_shaft_jcs_crank_joint = multi_dot([A(config.P_rbs_crank_shaft).T,triad(config.ax1_jcs_crank_joint)])
        self.Mbar_rbs_engine_block_jcs_crank_joint = multi_dot([A(config.P_rbs_engine_block).T,triad(config.ax1_jcs_crank_joint)])
        self.ubar_rbs_crank_shaft_jcs_crank_joint = (multi_dot([A(config.P_rbs_crank_shaft).T,config.pt1_jcs_crank_joint]) + (-1) * multi_dot([A(config.P_rbs_crank_shaft).T,config.R_rbs_crank_shaft]))
        self.ubar_rbs_engine_block_jcs_crank_joint = (multi_dot([A(config.P_rbs_engine_block).T,config.pt1_jcs_crank_joint]) + (-1) * multi_dot([A(config.P_rbs_engine_block).T,config.R_rbs_engine_block]))

    
    def _map_gen_coordinates(self):
        q = self._q
        self.R_ground = q[0:3]
        self.P_ground = q[3:7]
        self.R_rbs_piston_1 = q[7:10]
        self.P_rbs_piston_1 = q[10:14]
        self.R_rbs_piston_2 = q[14:17]
        self.P_rbs_piston_2 = q[17:21]
        self.R_rbs_piston_3 = q[21:24]
        self.P_rbs_piston_3 = q[24:28]
        self.R_rbs_piston_4 = q[28:31]
        self.P_rbs_piston_4 = q[31:35]
        self.R_rbs_connect_1 = q[35:38]
        self.P_rbs_connect_1 = q[38:42]
        self.R_rbs_connect_2 = q[42:45]
        self.P_rbs_connect_2 = q[45:49]
        self.R_rbs_connect_3 = q[49:52]
        self.P_rbs_connect_3 = q[52:56]
        self.R_rbs_connect_4 = q[56:59]
        self.P_rbs_connect_4 = q[59:63]
        self.R_rbs_crank_shaft = q[63:66]
        self.P_rbs_crank_shaft = q[66:70]
        self.R_rbs_engine_block = q[70:73]
        self.P_rbs_engine_block = q[73:77]

    
    def _map_gen_velocities(self):
        qd = self._qd
        self.Rd_ground = qd[0:3]
        self.Pd_ground = qd[3:7]
        self.Rd_rbs_piston_1 = qd[7:10]
        self.Pd_rbs_piston_1 = qd[10:14]
        self.Rd_rbs_piston_2 = qd[14:17]
        self.Pd_rbs_piston_2 = qd[17:21]
        self.Rd_rbs_piston_3 = qd[21:24]
        self.Pd_rbs_piston_3 = qd[24:28]
        self.Rd_rbs_piston_4 = qd[28:31]
        self.Pd_rbs_piston_4 = qd[31:35]
        self.Rd_rbs_connect_1 = qd[35:38]
        self.Pd_rbs_connect_1 = qd[38:42]
        self.Rd_rbs_connect_2 = qd[42:45]
        self.Pd_rbs_connect_2 = qd[45:49]
        self.Rd_rbs_connect_3 = qd[49:52]
        self.Pd_rbs_connect_3 = qd[52:56]
        self.Rd_rbs_connect_4 = qd[56:59]
        self.Pd_rbs_connect_4 = qd[59:63]
        self.Rd_rbs_crank_shaft = qd[63:66]
        self.Pd_rbs_crank_shaft = qd[66:70]
        self.Rd_rbs_engine_block = qd[70:73]
        self.Pd_rbs_engine_block = qd[73:77]

    
    def _map_gen_accelerations(self):
        qdd = self._qdd
        self.Rdd_ground = qdd[0:3]
        self.Pdd_ground = qdd[3:7]
        self.Rdd_rbs_piston_1 = qdd[7:10]
        self.Pdd_rbs_piston_1 = qdd[10:14]
        self.Rdd_rbs_piston_2 = qdd[14:17]
        self.Pdd_rbs_piston_2 = qdd[17:21]
        self.Rdd_rbs_piston_3 = qdd[21:24]
        self.Pdd_rbs_piston_3 = qdd[24:28]
        self.Rdd_rbs_piston_4 = qdd[28:31]
        self.Pdd_rbs_piston_4 = qdd[31:35]
        self.Rdd_rbs_connect_1 = qdd[35:38]
        self.Pdd_rbs_connect_1 = qdd[38:42]
        self.Rdd_rbs_connect_2 = qdd[42:45]
        self.Pdd_rbs_connect_2 = qdd[45:49]
        self.Rdd_rbs_connect_3 = qdd[49:52]
        self.Pdd_rbs_connect_3 = qdd[52:56]
        self.Rdd_rbs_connect_4 = qdd[56:59]
        self.Pdd_rbs_connect_4 = qdd[59:63]
        self.Rdd_rbs_crank_shaft = qdd[63:66]
        self.Pdd_rbs_crank_shaft = qdd[66:70]
        self.Rdd_rbs_engine_block = qdd[70:73]
        self.Pdd_rbs_engine_block = qdd[73:77]

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcs_cyl_1 = Lambda[0:4]
        self.L_jcs_sph_1 = Lambda[4:7]
        self.L_jcs_cyl_2 = Lambda[7:11]
        self.L_jcs_sph_2 = Lambda[11:14]
        self.L_jcs_cyl_3 = Lambda[14:18]
        self.L_jcs_sph_3 = Lambda[18:21]
        self.L_jcs_cyl_4 = Lambda[21:25]
        self.L_jcs_sph_4 = Lambda[25:28]
        self.L_jcs_bsph_1 = Lambda[28:31]
        self.L_jcs_bsph_2 = Lambda[31:34]
        self.L_jcs_bsph_3 = Lambda[34:37]
        self.L_jcs_bsph_4 = Lambda[37:40]
        self.L_jcs_crank_joint = Lambda[40:45]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.Mbar_rbs_piston_1_jcs_cyl_1[:,0:1].T
        x1 = self.P_rbs_piston_1
        x2 = A(x1)
        x3 = x2.T
        x4 = self.P_rbs_engine_block
        x5 = A(x4)
        x6 = self.Mbar_rbs_engine_block_jcs_cyl_1[:,2:3]
        x7 = self.Mbar_rbs_piston_1_jcs_cyl_1[:,1:2].T
        x8 = self.R_rbs_piston_1
        x9 = (-1) * self.R_rbs_engine_block
        x10 = (x8 + x9 + multi_dot([x2,self.ubar_rbs_piston_1_jcs_cyl_1]) + (-1) * multi_dot([x5,self.ubar_rbs_engine_block_jcs_cyl_1]))
        x11 = (-1) * self.R_rbs_connect_1
        x12 = self.P_rbs_connect_1
        x13 = A(x12)
        x14 = self.Mbar_rbs_piston_2_jcs_cyl_2[:,0:1].T
        x15 = self.P_rbs_piston_2
        x16 = A(x15)
        x17 = x16.T
        x18 = self.Mbar_rbs_engine_block_jcs_cyl_2[:,2:3]
        x19 = self.Mbar_rbs_piston_2_jcs_cyl_2[:,1:2].T
        x20 = self.R_rbs_piston_2
        x21 = (x20 + x9 + multi_dot([x16,self.ubar_rbs_piston_2_jcs_cyl_2]) + (-1) * multi_dot([x5,self.ubar_rbs_engine_block_jcs_cyl_2]))
        x22 = (-1) * self.R_rbs_connect_2
        x23 = self.P_rbs_connect_2
        x24 = A(x23)
        x25 = self.Mbar_rbs_piston_3_jcs_cyl_3[:,0:1].T
        x26 = self.P_rbs_piston_3
        x27 = A(x26)
        x28 = x27.T
        x29 = self.Mbar_rbs_engine_block_jcs_cyl_3[:,2:3]
        x30 = self.Mbar_rbs_piston_3_jcs_cyl_3[:,1:2].T
        x31 = self.R_rbs_piston_3
        x32 = (x31 + x9 + multi_dot([x27,self.ubar_rbs_piston_3_jcs_cyl_3]) + (-1) * multi_dot([x5,self.ubar_rbs_engine_block_jcs_cyl_3]))
        x33 = (-1) * self.R_rbs_connect_3
        x34 = self.P_rbs_connect_3
        x35 = A(x34)
        x36 = self.Mbar_rbs_piston_4_jcs_cyl_4[:,0:1].T
        x37 = self.P_rbs_piston_4
        x38 = A(x37)
        x39 = x38.T
        x40 = self.Mbar_rbs_engine_block_jcs_cyl_4[:,2:3]
        x41 = self.Mbar_rbs_piston_4_jcs_cyl_4[:,1:2].T
        x42 = self.R_rbs_piston_4
        x43 = (x42 + x9 + multi_dot([x38,self.ubar_rbs_piston_4_jcs_cyl_4]) + (-1) * multi_dot([x5,self.ubar_rbs_engine_block_jcs_cyl_4]))
        x44 = (-1) * self.R_rbs_connect_4
        x45 = self.P_rbs_connect_4
        x46 = A(x45)
        x47 = self.R_rbs_crank_shaft
        x48 = self.P_rbs_crank_shaft
        x49 = A(x48)
        x50 = x49.T
        x51 = self.Mbar_rbs_engine_block_jcs_crank_joint[:,2:3]
        x52 = (-1) * I1

        self.pos_eq_blocks = (multi_dot([x0,x3,x5,x6]),
        multi_dot([x7,x3,x5,x6]),
        multi_dot([x0,x3,x10]),
        multi_dot([x7,x3,x10]),
        (x8 + x11 + multi_dot([x2,self.ubar_rbs_piston_1_jcs_sph_1]) + (-1) * multi_dot([x13,self.ubar_rbs_connect_1_jcs_sph_1])),
        multi_dot([x14,x17,x5,x18]),
        multi_dot([x19,x17,x5,x18]),
        multi_dot([x14,x17,x21]),
        multi_dot([x19,x17,x21]),
        (x20 + x22 + multi_dot([x16,self.ubar_rbs_piston_2_jcs_sph_2]) + (-1) * multi_dot([x24,self.ubar_rbs_connect_2_jcs_sph_2])),
        multi_dot([x25,x28,x5,x29]),
        multi_dot([x30,x28,x5,x29]),
        multi_dot([x25,x28,x32]),
        multi_dot([x30,x28,x32]),
        (x31 + x33 + multi_dot([x27,self.ubar_rbs_piston_3_jcs_sph_3]) + (-1) * multi_dot([x35,self.ubar_rbs_connect_3_jcs_sph_3])),
        multi_dot([x36,x39,x5,x40]),
        multi_dot([x41,x39,x5,x40]),
        multi_dot([x36,x39,x43]),
        multi_dot([x41,x39,x43]),
        (x42 + x44 + multi_dot([x38,self.ubar_rbs_piston_4_jcs_sph_4]) + (-1) * multi_dot([x46,self.ubar_rbs_connect_4_jcs_sph_4])),
        (x47 + x11 + multi_dot([x49,self.ubar_rbs_crank_shaft_jcs_bsph_1]) + (-1) * multi_dot([x13,self.ubar_rbs_connect_1_jcs_bsph_1])),
        (x47 + x22 + multi_dot([x49,self.ubar_rbs_crank_shaft_jcs_bsph_2]) + (-1) * multi_dot([x24,self.ubar_rbs_connect_2_jcs_bsph_2])),
        (x47 + x33 + multi_dot([x49,self.ubar_rbs_crank_shaft_jcs_bsph_3]) + (-1) * multi_dot([x35,self.ubar_rbs_connect_3_jcs_bsph_3])),
        (x47 + x44 + multi_dot([x49,self.ubar_rbs_crank_shaft_jcs_bsph_4]) + (-1) * multi_dot([x46,self.ubar_rbs_connect_4_jcs_bsph_4])),
        (x47 + x9 + multi_dot([x49,self.ubar_rbs_crank_shaft_jcs_crank_joint]) + (-1) * multi_dot([x5,self.ubar_rbs_engine_block_jcs_crank_joint])),
        multi_dot([self.Mbar_rbs_crank_shaft_jcs_crank_joint[:,0:1].T,x50,x5,x51]),
        multi_dot([self.Mbar_rbs_crank_shaft_jcs_crank_joint[:,1:2].T,x50,x5,x51]),
        self.R_ground,
        ((-1) * self.Pg_ground + self.P_ground),
        (x52 + multi_dot([x1.T,x1])),
        (x52 + multi_dot([x15.T,x15])),
        (x52 + multi_dot([x26.T,x26])),
        (x52 + multi_dot([x37.T,x37])),
        (x52 + multi_dot([x12.T,x12])),
        (x52 + multi_dot([x23.T,x23])),
        (x52 + multi_dot([x34.T,x34])),
        (x52 + multi_dot([x45.T,x45])),
        (x52 + multi_dot([x48.T,x48])),
        (x52 + multi_dot([x4.T,x4])),)

    
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
        v0,
        v0,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v0,
        v0,
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
        v0,
        v0,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Mbar_rbs_engine_block_jcs_cyl_1[:,2:3]
        a1 = a0.T
        a2 = self.P_rbs_engine_block
        a3 = A(a2).T
        a4 = self.Pd_rbs_piston_1
        a5 = self.Mbar_rbs_piston_1_jcs_cyl_1[:,0:1]
        a6 = B(a4,a5)
        a7 = a5.T
        a8 = self.P_rbs_piston_1
        a9 = A(a8).T
        a10 = self.Pd_rbs_engine_block
        a11 = B(a10,a0)
        a12 = a4.T
        a13 = B(a8,a5).T
        a14 = B(a2,a0)
        a15 = self.Mbar_rbs_piston_1_jcs_cyl_1[:,1:2]
        a16 = B(a4,a15)
        a17 = a15.T
        a18 = B(a8,a15).T
        a19 = self.ubar_rbs_piston_1_jcs_cyl_1
        a20 = self.ubar_rbs_engine_block_jcs_cyl_1
        a21 = (multi_dot([B(a4,a19),a4]) + (-1) * multi_dot([B(a10,a20),a10]))
        a22 = (-1) * self.Rd_rbs_engine_block
        a23 = (self.Rd_rbs_piston_1 + a22 + multi_dot([B(a8,a19),a4]) + (-1) * multi_dot([B(a2,a20),a10]))
        a24 = (-1) * self.R_rbs_engine_block.T
        a25 = (self.R_rbs_piston_1.T + a24 + multi_dot([a19.T,a9]) + (-1) * multi_dot([a20.T,a3]))
        a26 = self.Pd_rbs_connect_1
        a27 = self.Mbar_rbs_piston_2_jcs_cyl_2[:,0:1]
        a28 = a27.T
        a29 = self.P_rbs_piston_2
        a30 = A(a29).T
        a31 = self.Mbar_rbs_engine_block_jcs_cyl_2[:,2:3]
        a32 = B(a10,a31)
        a33 = a31.T
        a34 = self.Pd_rbs_piston_2
        a35 = B(a34,a27)
        a36 = a34.T
        a37 = B(a29,a27).T
        a38 = B(a2,a31)
        a39 = self.Mbar_rbs_piston_2_jcs_cyl_2[:,1:2]
        a40 = a39.T
        a41 = B(a34,a39)
        a42 = B(a29,a39).T
        a43 = self.ubar_rbs_piston_2_jcs_cyl_2
        a44 = self.ubar_rbs_engine_block_jcs_cyl_2
        a45 = (multi_dot([B(a34,a43),a34]) + (-1) * multi_dot([B(a10,a44),a10]))
        a46 = (self.Rd_rbs_piston_2 + a22 + multi_dot([B(a29,a43),a34]) + (-1) * multi_dot([B(a2,a44),a10]))
        a47 = (self.R_rbs_piston_2.T + a24 + multi_dot([a43.T,a30]) + (-1) * multi_dot([a44.T,a3]))
        a48 = self.Pd_rbs_connect_2
        a49 = self.Mbar_rbs_piston_3_jcs_cyl_3[:,0:1]
        a50 = a49.T
        a51 = self.P_rbs_piston_3
        a52 = A(a51).T
        a53 = self.Mbar_rbs_engine_block_jcs_cyl_3[:,2:3]
        a54 = B(a10,a53)
        a55 = a53.T
        a56 = self.Pd_rbs_piston_3
        a57 = B(a56,a49)
        a58 = a56.T
        a59 = B(a51,a49).T
        a60 = B(a2,a53)
        a61 = self.Mbar_rbs_piston_3_jcs_cyl_3[:,1:2]
        a62 = a61.T
        a63 = B(a56,a61)
        a64 = B(a51,a61).T
        a65 = self.ubar_rbs_piston_3_jcs_cyl_3
        a66 = self.ubar_rbs_engine_block_jcs_cyl_3
        a67 = (multi_dot([B(a56,a65),a56]) + (-1) * multi_dot([B(a10,a66),a10]))
        a68 = (self.Rd_rbs_piston_3 + a22 + multi_dot([B(a51,a65),a56]) + (-1) * multi_dot([B(a2,a66),a10]))
        a69 = (self.R_rbs_piston_3.T + a24 + multi_dot([a65.T,a52]) + (-1) * multi_dot([a66.T,a3]))
        a70 = self.Pd_rbs_connect_3
        a71 = self.Mbar_rbs_piston_4_jcs_cyl_4[:,0:1]
        a72 = a71.T
        a73 = self.P_rbs_piston_4
        a74 = A(a73).T
        a75 = self.Mbar_rbs_engine_block_jcs_cyl_4[:,2:3]
        a76 = B(a10,a75)
        a77 = a75.T
        a78 = self.Pd_rbs_piston_4
        a79 = B(a78,a71)
        a80 = a78.T
        a81 = B(a73,a71).T
        a82 = B(a2,a75)
        a83 = self.Mbar_rbs_piston_4_jcs_cyl_4[:,1:2]
        a84 = a83.T
        a85 = B(a78,a83)
        a86 = B(a73,a83).T
        a87 = self.ubar_rbs_piston_4_jcs_cyl_4
        a88 = self.ubar_rbs_engine_block_jcs_cyl_4
        a89 = (multi_dot([B(a78,a87),a78]) + (-1) * multi_dot([B(a10,a88),a10]))
        a90 = (self.Rd_rbs_piston_4 + a22 + multi_dot([B(a73,a87),a78]) + (-1) * multi_dot([B(a2,a88),a10]))
        a91 = (self.R_rbs_piston_4.T + a24 + multi_dot([a87.T,a74]) + (-1) * multi_dot([a88.T,a3]))
        a92 = self.Pd_rbs_connect_4
        a93 = self.Pd_rbs_crank_shaft
        a94 = self.Mbar_rbs_engine_block_jcs_crank_joint[:,2:3]
        a95 = a94.T
        a96 = self.Mbar_rbs_crank_shaft_jcs_crank_joint[:,0:1]
        a97 = self.P_rbs_crank_shaft
        a98 = A(a97).T
        a99 = B(a10,a94)
        a100 = a93.T
        a101 = B(a2,a94)
        a102 = self.Mbar_rbs_crank_shaft_jcs_crank_joint[:,1:2]

        self.acc_eq_blocks = ((multi_dot([a1,a3,a6,a4]) + multi_dot([a7,a9,a11,a10]) + (2) * multi_dot([a12,a13,a14,a10])),
        (multi_dot([a1,a3,a16,a4]) + multi_dot([a17,a9,a11,a10]) + (2) * multi_dot([a12,a18,a14,a10])),
        (multi_dot([a7,a9,a21]) + (2) * multi_dot([a12,a13,a23]) + multi_dot([a25,a6,a4])),
        (multi_dot([a17,a9,a21]) + (2) * multi_dot([a12,a18,a23]) + multi_dot([a25,a16,a4])),
        (multi_dot([B(a4,self.ubar_rbs_piston_1_jcs_sph_1),a4]) + (-1) * multi_dot([B(a26,self.ubar_rbs_connect_1_jcs_sph_1),a26])),
        (multi_dot([a28,a30,a32,a10]) + multi_dot([a33,a3,a35,a34]) + (2) * multi_dot([a36,a37,a38,a10])),
        (multi_dot([a40,a30,a32,a10]) + multi_dot([a33,a3,a41,a34]) + (2) * multi_dot([a36,a42,a38,a10])),
        (multi_dot([a28,a30,a45]) + (2) * multi_dot([a36,a37,a46]) + multi_dot([a47,a35,a34])),
        (multi_dot([a40,a30,a45]) + (2) * multi_dot([a36,a42,a46]) + multi_dot([a47,a41,a34])),
        (multi_dot([B(a34,self.ubar_rbs_piston_2_jcs_sph_2),a34]) + (-1) * multi_dot([B(a48,self.ubar_rbs_connect_2_jcs_sph_2),a48])),
        (multi_dot([a50,a52,a54,a10]) + multi_dot([a55,a3,a57,a56]) + (2) * multi_dot([a58,a59,a60,a10])),
        (multi_dot([a62,a52,a54,a10]) + multi_dot([a55,a3,a63,a56]) + (2) * multi_dot([a58,a64,a60,a10])),
        (multi_dot([a50,a52,a67]) + (2) * multi_dot([a58,a59,a68]) + multi_dot([a69,a57,a56])),
        (multi_dot([a62,a52,a67]) + (2) * multi_dot([a58,a64,a68]) + multi_dot([a69,a63,a56])),
        (multi_dot([B(a56,self.ubar_rbs_piston_3_jcs_sph_3),a56]) + (-1) * multi_dot([B(a70,self.ubar_rbs_connect_3_jcs_sph_3),a70])),
        (multi_dot([a72,a74,a76,a10]) + multi_dot([a77,a3,a79,a78]) + (2) * multi_dot([a80,a81,a82,a10])),
        (multi_dot([a84,a74,a76,a10]) + multi_dot([a77,a3,a85,a78]) + (2) * multi_dot([a80,a86,a82,a10])),
        (multi_dot([a72,a74,a89]) + (2) * multi_dot([a80,a81,a90]) + multi_dot([a91,a79,a78])),
        (multi_dot([a84,a74,a89]) + (2) * multi_dot([a80,a86,a90]) + multi_dot([a91,a85,a78])),
        (multi_dot([B(a78,self.ubar_rbs_piston_4_jcs_sph_4),a78]) + (-1) * multi_dot([B(a92,self.ubar_rbs_connect_4_jcs_sph_4),a92])),
        (multi_dot([B(a93,self.ubar_rbs_crank_shaft_jcs_bsph_1),a93]) + (-1) * multi_dot([B(a26,self.ubar_rbs_connect_1_jcs_bsph_1),a26])),
        (multi_dot([B(a93,self.ubar_rbs_crank_shaft_jcs_bsph_2),a93]) + (-1) * multi_dot([B(a48,self.ubar_rbs_connect_2_jcs_bsph_2),a48])),
        (multi_dot([B(a93,self.ubar_rbs_crank_shaft_jcs_bsph_3),a93]) + (-1) * multi_dot([B(a70,self.ubar_rbs_connect_3_jcs_bsph_3),a70])),
        (multi_dot([B(a93,self.ubar_rbs_crank_shaft_jcs_bsph_4),a93]) + (-1) * multi_dot([B(a92,self.ubar_rbs_connect_4_jcs_bsph_4),a92])),
        (multi_dot([B(a93,self.ubar_rbs_crank_shaft_jcs_crank_joint),a93]) + (-1) * multi_dot([B(a10,self.ubar_rbs_engine_block_jcs_crank_joint),a10])),
        (multi_dot([a95,a3,B(a93,a96),a93]) + multi_dot([a96.T,a98,a99,a10]) + (2) * multi_dot([a100,B(a97,a96).T,a101,a10])),
        (multi_dot([a95,a3,B(a93,a102),a93]) + multi_dot([a102.T,a98,a99,a10]) + (2) * multi_dot([a100,B(a97,a102).T,a101,a10])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a12,a4]),
        (2) * multi_dot([a36,a34]),
        (2) * multi_dot([a58,a56]),
        (2) * multi_dot([a80,a78]),
        (2) * multi_dot([a26.T,a26]),
        (2) * multi_dot([a48.T,a48]),
        (2) * multi_dot([a70.T,a70]),
        (2) * multi_dot([a92.T,a92]),
        (2) * multi_dot([a100,a93]),
        (2) * multi_dot([a10.T,a10]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = Z1x3
        j1 = self.Mbar_rbs_engine_block_jcs_cyl_1[:,2:3]
        j2 = j1.T
        j3 = self.P_rbs_engine_block
        j4 = A(j3).T
        j5 = self.P_rbs_piston_1
        j6 = self.Mbar_rbs_piston_1_jcs_cyl_1[:,0:1]
        j7 = B(j5,j6)
        j8 = self.Mbar_rbs_piston_1_jcs_cyl_1[:,1:2]
        j9 = B(j5,j8)
        j10 = j6.T
        j11 = A(j5).T
        j12 = multi_dot([j10,j11])
        j13 = self.ubar_rbs_piston_1_jcs_cyl_1
        j14 = B(j5,j13)
        j15 = (-1) * self.R_rbs_engine_block.T
        j16 = self.ubar_rbs_engine_block_jcs_cyl_1
        j17 = (self.R_rbs_piston_1.T + j15 + multi_dot([j13.T,j11]) + (-1) * multi_dot([j16.T,j4]))
        j18 = j8.T
        j19 = multi_dot([j18,j11])
        j20 = B(j3,j1)
        j21 = B(j3,j16)
        j22 = I3
        j23 = (-1) * j22
        j24 = self.P_rbs_connect_1
        j25 = self.Mbar_rbs_engine_block_jcs_cyl_2[:,2:3]
        j26 = j25.T
        j27 = self.P_rbs_piston_2
        j28 = self.Mbar_rbs_piston_2_jcs_cyl_2[:,0:1]
        j29 = B(j27,j28)
        j30 = self.Mbar_rbs_piston_2_jcs_cyl_2[:,1:2]
        j31 = B(j27,j30)
        j32 = j28.T
        j33 = A(j27).T
        j34 = multi_dot([j32,j33])
        j35 = self.ubar_rbs_piston_2_jcs_cyl_2
        j36 = B(j27,j35)
        j37 = self.ubar_rbs_engine_block_jcs_cyl_2
        j38 = (self.R_rbs_piston_2.T + j15 + multi_dot([j35.T,j33]) + (-1) * multi_dot([j37.T,j4]))
        j39 = j30.T
        j40 = multi_dot([j39,j33])
        j41 = B(j3,j25)
        j42 = B(j3,j37)
        j43 = self.P_rbs_connect_2
        j44 = self.Mbar_rbs_engine_block_jcs_cyl_3[:,2:3]
        j45 = j44.T
        j46 = self.P_rbs_piston_3
        j47 = self.Mbar_rbs_piston_3_jcs_cyl_3[:,0:1]
        j48 = B(j46,j47)
        j49 = self.Mbar_rbs_piston_3_jcs_cyl_3[:,1:2]
        j50 = B(j46,j49)
        j51 = j47.T
        j52 = A(j46).T
        j53 = multi_dot([j51,j52])
        j54 = self.ubar_rbs_piston_3_jcs_cyl_3
        j55 = B(j46,j54)
        j56 = self.ubar_rbs_engine_block_jcs_cyl_3
        j57 = (self.R_rbs_piston_3.T + j15 + multi_dot([j54.T,j52]) + (-1) * multi_dot([j56.T,j4]))
        j58 = j49.T
        j59 = multi_dot([j58,j52])
        j60 = B(j3,j44)
        j61 = B(j3,j56)
        j62 = self.P_rbs_connect_3
        j63 = self.Mbar_rbs_engine_block_jcs_cyl_4[:,2:3]
        j64 = j63.T
        j65 = self.P_rbs_piston_4
        j66 = self.Mbar_rbs_piston_4_jcs_cyl_4[:,0:1]
        j67 = B(j65,j66)
        j68 = self.Mbar_rbs_piston_4_jcs_cyl_4[:,1:2]
        j69 = B(j65,j68)
        j70 = j66.T
        j71 = A(j65).T
        j72 = multi_dot([j70,j71])
        j73 = self.ubar_rbs_piston_4_jcs_cyl_4
        j74 = B(j65,j73)
        j75 = self.ubar_rbs_engine_block_jcs_cyl_4
        j76 = (self.R_rbs_piston_4.T + j15 + multi_dot([j73.T,j71]) + (-1) * multi_dot([j75.T,j4]))
        j77 = j68.T
        j78 = multi_dot([j77,j71])
        j79 = B(j3,j63)
        j80 = B(j3,j75)
        j81 = self.P_rbs_connect_4
        j82 = self.P_rbs_crank_shaft
        j83 = self.Mbar_rbs_engine_block_jcs_crank_joint[:,2:3]
        j84 = j83.T
        j85 = self.Mbar_rbs_crank_shaft_jcs_crank_joint[:,0:1]
        j86 = self.Mbar_rbs_crank_shaft_jcs_crank_joint[:,1:2]
        j87 = A(j82).T
        j88 = B(j3,j83)

        self.jac_eq_blocks = (j0,
        multi_dot([j2,j4,j7]),
        j0,
        multi_dot([j10,j11,j20]),
        j0,
        multi_dot([j2,j4,j9]),
        j0,
        multi_dot([j18,j11,j20]),
        j12,
        (multi_dot([j10,j11,j14]) + multi_dot([j17,j7])),
        (-1) * j12,
        (-1) * multi_dot([j10,j11,j21]),
        j19,
        (multi_dot([j18,j11,j14]) + multi_dot([j17,j9])),
        (-1) * j19,
        (-1) * multi_dot([j18,j11,j21]),
        j22,
        B(j5,self.ubar_rbs_piston_1_jcs_sph_1),
        j23,
        (-1) * B(j24,self.ubar_rbs_connect_1_jcs_sph_1),
        j0,
        multi_dot([j26,j4,j29]),
        j0,
        multi_dot([j32,j33,j41]),
        j0,
        multi_dot([j26,j4,j31]),
        j0,
        multi_dot([j39,j33,j41]),
        j34,
        (multi_dot([j32,j33,j36]) + multi_dot([j38,j29])),
        (-1) * j34,
        (-1) * multi_dot([j32,j33,j42]),
        j40,
        (multi_dot([j39,j33,j36]) + multi_dot([j38,j31])),
        (-1) * j40,
        (-1) * multi_dot([j39,j33,j42]),
        j22,
        B(j27,self.ubar_rbs_piston_2_jcs_sph_2),
        j23,
        (-1) * B(j43,self.ubar_rbs_connect_2_jcs_sph_2),
        j0,
        multi_dot([j45,j4,j48]),
        j0,
        multi_dot([j51,j52,j60]),
        j0,
        multi_dot([j45,j4,j50]),
        j0,
        multi_dot([j58,j52,j60]),
        j53,
        (multi_dot([j51,j52,j55]) + multi_dot([j57,j48])),
        (-1) * j53,
        (-1) * multi_dot([j51,j52,j61]),
        j59,
        (multi_dot([j58,j52,j55]) + multi_dot([j57,j50])),
        (-1) * j59,
        (-1) * multi_dot([j58,j52,j61]),
        j22,
        B(j46,self.ubar_rbs_piston_3_jcs_sph_3),
        j23,
        (-1) * B(j62,self.ubar_rbs_connect_3_jcs_sph_3),
        j0,
        multi_dot([j64,j4,j67]),
        j0,
        multi_dot([j70,j71,j79]),
        j0,
        multi_dot([j64,j4,j69]),
        j0,
        multi_dot([j77,j71,j79]),
        j72,
        (multi_dot([j70,j71,j74]) + multi_dot([j76,j67])),
        (-1) * j72,
        (-1) * multi_dot([j70,j71,j80]),
        j78,
        (multi_dot([j77,j71,j74]) + multi_dot([j76,j69])),
        (-1) * j78,
        (-1) * multi_dot([j77,j71,j80]),
        j22,
        B(j65,self.ubar_rbs_piston_4_jcs_sph_4),
        j23,
        (-1) * B(j81,self.ubar_rbs_connect_4_jcs_sph_4),
        j23,
        (-1) * B(j24,self.ubar_rbs_connect_1_jcs_bsph_1),
        j22,
        B(j82,self.ubar_rbs_crank_shaft_jcs_bsph_1),
        j23,
        (-1) * B(j43,self.ubar_rbs_connect_2_jcs_bsph_2),
        j22,
        B(j82,self.ubar_rbs_crank_shaft_jcs_bsph_2),
        j23,
        (-1) * B(j62,self.ubar_rbs_connect_3_jcs_bsph_3),
        j22,
        B(j82,self.ubar_rbs_crank_shaft_jcs_bsph_3),
        j23,
        (-1) * B(j81,self.ubar_rbs_connect_4_jcs_bsph_4),
        j22,
        B(j82,self.ubar_rbs_crank_shaft_jcs_bsph_4),
        j22,
        B(j82,self.ubar_rbs_crank_shaft_jcs_crank_joint),
        j23,
        (-1) * B(j3,self.ubar_rbs_engine_block_jcs_crank_joint),
        j0,
        multi_dot([j84,j4,B(j82,j85)]),
        j0,
        multi_dot([j85.T,j87,j88]),
        j0,
        multi_dot([j84,j4,B(j82,j86)]),
        j0,
        multi_dot([j86.T,j87,j88]),
        j22,
        Z3x4,
        Z4x3,
        I4,
        j0,
        (2) * j5.T,
        j0,
        (2) * j27.T,
        j0,
        (2) * j46.T,
        j0,
        (2) * j65.T,
        j0,
        (2) * j24.T,
        j0,
        (2) * j43.T,
        j0,
        (2) * j62.T,
        j0,
        (2) * j81.T,
        j0,
        (2) * j82.T,
        j0,
        (2) * j3.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_ground)
        m2 = G(self.P_rbs_piston_1)
        m3 = G(self.P_rbs_piston_2)
        m4 = G(self.P_rbs_piston_3)
        m5 = G(self.P_rbs_piston_4)
        m6 = G(self.P_rbs_connect_1)
        m7 = G(self.P_rbs_connect_2)
        m8 = G(self.P_rbs_connect_3)
        m9 = G(self.P_rbs_connect_4)
        m10 = G(self.P_rbs_crank_shaft)
        m11 = G(self.P_rbs_engine_block)

        self.mass_eq_blocks = (self.m_ground * m0,
        (4) * multi_dot([m1.T,self.Jbar_ground,m1]),
        config.m_rbs_piston_1 * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbs_piston_1,m2]),
        config.m_rbs_piston_2 * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbs_piston_2,m3]),
        config.m_rbs_piston_3 * m0,
        (4) * multi_dot([m4.T,config.Jbar_rbs_piston_3,m4]),
        config.m_rbs_piston_4 * m0,
        (4) * multi_dot([m5.T,config.Jbar_rbs_piston_4,m5]),
        config.m_rbs_connect_1 * m0,
        (4) * multi_dot([m6.T,config.Jbar_rbs_connect_1,m6]),
        config.m_rbs_connect_2 * m0,
        (4) * multi_dot([m7.T,config.Jbar_rbs_connect_2,m7]),
        config.m_rbs_connect_3 * m0,
        (4) * multi_dot([m8.T,config.Jbar_rbs_connect_3,m8]),
        config.m_rbs_connect_4 * m0,
        (4) * multi_dot([m9.T,config.Jbar_rbs_connect_4,m9]),
        config.m_rbs_crank_shaft * m0,
        (4) * multi_dot([m10.T,config.Jbar_rbs_crank_shaft,m10]),
        config.m_rbs_engine_block * m0,
        (4) * multi_dot([m11.T,config.Jbar_rbs_engine_block,m11]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = self.P_ground
        f1 = A(f0)
        f2 = self.Mbar_ground_fas_bush_1
        f3 = f2.T
        f4 = f1.T
        f5 = self.ubar_ground_fas_bush_1
        f6 = multi_dot([f1,f5])
        f7 = self.P_rbs_engine_block
        f8 = A(f7)
        f9 = self.ubar_rbs_engine_block_fas_bush_1
        f10 = multi_dot([f8,f9])
        f11 = ((-1) * self.R_rbs_engine_block + self.R_ground)
        f12 = (f11 + ((-1) * f10 + f6))
        f13 = self.Pd_ground
        f14 = self.Pd_rbs_engine_block
        f15 = ((-1) * self.Rd_rbs_engine_block + self.Rd_ground)
        f16 = (f15 + (multi_dot([B(f0,f5),f13]) + (-1) * multi_dot([B(f7,f9),f14])))
        f17 = (config.Kt_fas_bush_1 * multi_dot([f3,f4,f12]) + config.Ct_fas_bush_1 * multi_dot([f3,f4,f16]))
        f18 = self.Mbar_ground_fas_bush_2
        f19 = f18.T
        f20 = self.ubar_ground_fas_bush_2
        f21 = multi_dot([f1,f20])
        f22 = self.ubar_rbs_engine_block_fas_bush_2
        f23 = multi_dot([f8,f22])
        f24 = (f11 + ((-1) * f23 + f21))
        f25 = (f15 + (multi_dot([B(f0,f20),f13]) + (-1) * multi_dot([B(f7,f22),f14])))
        f26 = (config.Kt_fas_bush_2 * multi_dot([f19,f4,f24]) + config.Ct_fas_bush_2 * multi_dot([f19,f4,f25]))
        f27 = self.Mbar_ground_fas_bush_3
        f28 = f27.T
        f29 = self.ubar_ground_fas_bush_3
        f30 = multi_dot([f1,f29])
        f31 = self.ubar_rbs_engine_block_fas_bush_3
        f32 = multi_dot([f8,f31])
        f33 = (f11 + ((-1) * f32 + f30))
        f34 = (f15 + (multi_dot([B(f0,f29),f13]) + (-1) * multi_dot([B(f7,f31),f14])))
        f35 = (config.Kt_fas_bush_3 * multi_dot([f28,f4,f33]) + config.Ct_fas_bush_3 * multi_dot([f28,f4,f34]))
        f36 = self.Mbar_ground_fas_bush_4
        f37 = f36.T
        f38 = self.ubar_ground_fas_bush_4
        f39 = multi_dot([f1,f38])
        f40 = self.ubar_rbs_engine_block_fas_bush_4
        f41 = multi_dot([f8,f40])
        f42 = (f11 + ((-1) * f41 + f39))
        f43 = (f15 + (multi_dot([B(f0,f38),f13]) + (-1) * multi_dot([B(f7,f40),f14])))
        f44 = (config.Kt_fas_bush_4 * multi_dot([f37,f4,f42]) + config.Ct_fas_bush_4 * multi_dot([f37,f4,f43]))
        f45 = E(f0).T
        f46 = G(self.Pd_rbs_piston_1)
        f47 = G(self.Pd_rbs_piston_2)
        f48 = G(self.Pd_rbs_piston_3)
        f49 = G(self.Pd_rbs_piston_4)
        f50 = G(self.Pd_rbs_connect_1)
        f51 = G(self.Pd_rbs_connect_2)
        f52 = G(self.Pd_rbs_connect_3)
        f53 = G(self.Pd_rbs_connect_4)
        f54 = Z3x1
        f55 = self.P_rbs_crank_shaft
        f56 = G(self.Pd_rbs_crank_shaft)
        f57 = self.Mbar_rbs_engine_block_fas_bush_1
        f58 = f57.T
        f59 = f8.T
        f60 = (config.Kt_fas_bush_1 * multi_dot([f58,f59,f12]) + config.Ct_fas_bush_1 * multi_dot([f58,f59,f16]))
        f61 = self.Mbar_rbs_engine_block_fas_bush_2
        f62 = f61.T
        f63 = (config.Kt_fas_bush_2 * multi_dot([f62,f59,f24]) + config.Ct_fas_bush_2 * multi_dot([f62,f59,f25]))
        f64 = self.Mbar_rbs_engine_block_fas_bush_3
        f65 = f64.T
        f66 = (config.Kt_fas_bush_3 * multi_dot([f65,f59,f33]) + config.Ct_fas_bush_3 * multi_dot([f65,f59,f34]))
        f67 = self.Mbar_rbs_engine_block_fas_bush_4
        f68 = f67.T
        f69 = (config.Kt_fas_bush_4 * multi_dot([f68,f59,f42]) + config.Ct_fas_bush_4 * multi_dot([f68,f59,f43]))
        f70 = G(f14)
        f71 = E(f7).T

        self.frc_eq_blocks = (((-1) * multi_dot([f1,f2,f17]) + (-1) * multi_dot([f1,f18,f26]) + (-1) * multi_dot([f1,f27,f35]) + (-1) * multi_dot([f1,f36,f44])),
        ((2) * multi_dot([f45,skew(f6).T,f1,f2,f17]) + (2) * multi_dot([f45,skew(f21).T,f1,f18,f26]) + (2) * multi_dot([f45,skew(f30).T,f1,f27,f35]) + (2) * multi_dot([f45,skew(f39).T,f1,f36,f44])),
        self.F_rbs_piston_1_gravity,
        (8) * multi_dot([f46.T,config.Jbar_rbs_piston_1,f46,self.P_rbs_piston_1]),
        self.F_rbs_piston_2_gravity,
        (8) * multi_dot([f47.T,config.Jbar_rbs_piston_2,f47,self.P_rbs_piston_2]),
        self.F_rbs_piston_3_gravity,
        (8) * multi_dot([f48.T,config.Jbar_rbs_piston_3,f48,self.P_rbs_piston_3]),
        self.F_rbs_piston_4_gravity,
        (8) * multi_dot([f49.T,config.Jbar_rbs_piston_4,f49,self.P_rbs_piston_4]),
        self.F_rbs_connect_1_gravity,
        (8) * multi_dot([f50.T,config.Jbar_rbs_connect_1,f50,self.P_rbs_connect_1]),
        self.F_rbs_connect_2_gravity,
        (8) * multi_dot([f51.T,config.Jbar_rbs_connect_2,f51,self.P_rbs_connect_2]),
        self.F_rbs_connect_3_gravity,
        (8) * multi_dot([f52.T,config.Jbar_rbs_connect_3,f52,self.P_rbs_connect_3]),
        self.F_rbs_connect_4_gravity,
        (8) * multi_dot([f53.T,config.Jbar_rbs_connect_4,f53,self.P_rbs_connect_4]),
        (self.F_rbs_crank_shaft_gravity + f54),
        ((2 * config.UF_fas_drive(t)) * multi_dot([G(f55).T,self.vbar_rbs_crank_shaft_fas_drive]) + (8) * multi_dot([f56.T,config.Jbar_rbs_crank_shaft,f56,f55])),
        (self.F_rbs_engine_block_gravity + f54 + multi_dot([f8,f57,f60]) + multi_dot([f8,f61,f63]) + multi_dot([f8,f64,f66]) + multi_dot([f8,f67,f69])),
        (Z4x1 + (8) * multi_dot([f70.T,config.Jbar_rbs_engine_block,f70,f7]) + (-2) * multi_dot([f71,skew(f10).T,f8,f57,f60]) + (-2) * multi_dot([f71,skew(f23).T,f8,f61,f63]) + (-2) * multi_dot([f71,skew(f32).T,f8,f64,f66]) + (-2) * multi_dot([f71,skew(f41).T,f8,f67,f69])),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        self.F_ground_fas_bush_1 = (-1) * multi_dot([A(self.P_ground),self.Mbar_ground_fas_bush_1,(config.Ct_fas_bush_1 * multi_dot([self.Mbar_ground_fas_bush_1.T,A(self.P_ground).T,((-1) * self.Rd_rbs_engine_block + multi_dot([B(self.P_ground,self.ubar_ground_fas_bush_1),self.Pd_ground]) + (-1) * multi_dot([B(self.P_rbs_engine_block,self.ubar_rbs_engine_block_fas_bush_1),self.Pd_rbs_engine_block]) + self.Rd_ground)]) + config.Kt_fas_bush_1 * multi_dot([self.Mbar_ground_fas_bush_1.T,A(self.P_ground).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_ground),self.ubar_ground_fas_bush_1]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_fas_bush_1]) + self.R_ground)]))])
        self.T_ground_fas_bush_1 = Z3x1
        self.F_ground_fas_bush_2 = (-1) * multi_dot([A(self.P_ground),self.Mbar_ground_fas_bush_2,(config.Ct_fas_bush_2 * multi_dot([self.Mbar_ground_fas_bush_2.T,A(self.P_ground).T,((-1) * self.Rd_rbs_engine_block + multi_dot([B(self.P_ground,self.ubar_ground_fas_bush_2),self.Pd_ground]) + (-1) * multi_dot([B(self.P_rbs_engine_block,self.ubar_rbs_engine_block_fas_bush_2),self.Pd_rbs_engine_block]) + self.Rd_ground)]) + config.Kt_fas_bush_2 * multi_dot([self.Mbar_ground_fas_bush_2.T,A(self.P_ground).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_ground),self.ubar_ground_fas_bush_2]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_fas_bush_2]) + self.R_ground)]))])
        self.T_ground_fas_bush_2 = Z3x1
        self.F_ground_fas_bush_3 = (-1) * multi_dot([A(self.P_ground),self.Mbar_ground_fas_bush_3,(config.Ct_fas_bush_3 * multi_dot([self.Mbar_ground_fas_bush_3.T,A(self.P_ground).T,((-1) * self.Rd_rbs_engine_block + multi_dot([B(self.P_ground,self.ubar_ground_fas_bush_3),self.Pd_ground]) + (-1) * multi_dot([B(self.P_rbs_engine_block,self.ubar_rbs_engine_block_fas_bush_3),self.Pd_rbs_engine_block]) + self.Rd_ground)]) + config.Kt_fas_bush_3 * multi_dot([self.Mbar_ground_fas_bush_3.T,A(self.P_ground).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_ground),self.ubar_ground_fas_bush_3]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_fas_bush_3]) + self.R_ground)]))])
        self.T_ground_fas_bush_3 = Z3x1
        self.F_ground_fas_bush_4 = (-1) * multi_dot([A(self.P_ground),self.Mbar_ground_fas_bush_4,(config.Ct_fas_bush_4 * multi_dot([self.Mbar_ground_fas_bush_4.T,A(self.P_ground).T,((-1) * self.Rd_rbs_engine_block + multi_dot([B(self.P_ground,self.ubar_ground_fas_bush_4),self.Pd_ground]) + (-1) * multi_dot([B(self.P_rbs_engine_block,self.ubar_rbs_engine_block_fas_bush_4),self.Pd_rbs_engine_block]) + self.Rd_ground)]) + config.Kt_fas_bush_4 * multi_dot([self.Mbar_ground_fas_bush_4.T,A(self.P_ground).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_ground),self.ubar_ground_fas_bush_4]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_fas_bush_4]) + self.R_ground)]))])
        self.T_ground_fas_bush_4 = Z3x1
        Q_rbs_piston_1_jcs_cyl_1 = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_piston_1),self.Mbar_rbs_piston_1_jcs_cyl_1[:,0:1]]),multi_dot([A(self.P_rbs_piston_1),self.Mbar_rbs_piston_1_jcs_cyl_1[:,1:2]])],[multi_dot([B(self.P_rbs_piston_1,self.Mbar_rbs_piston_1_jcs_cyl_1[:,0:1]).T,A(self.P_rbs_engine_block),self.Mbar_rbs_engine_block_jcs_cyl_1[:,2:3]]),multi_dot([B(self.P_rbs_piston_1,self.Mbar_rbs_piston_1_jcs_cyl_1[:,1:2]).T,A(self.P_rbs_engine_block),self.Mbar_rbs_engine_block_jcs_cyl_1[:,2:3]]),(multi_dot([B(self.P_rbs_piston_1,self.Mbar_rbs_piston_1_jcs_cyl_1[:,0:1]).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_rbs_piston_1),self.ubar_rbs_piston_1_jcs_cyl_1]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_jcs_cyl_1]) + self.R_rbs_piston_1)]) + multi_dot([B(self.P_rbs_piston_1,self.ubar_rbs_piston_1_jcs_cyl_1).T,A(self.P_rbs_piston_1),self.Mbar_rbs_piston_1_jcs_cyl_1[:,0:1]])),(multi_dot([B(self.P_rbs_piston_1,self.Mbar_rbs_piston_1_jcs_cyl_1[:,1:2]).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_rbs_piston_1),self.ubar_rbs_piston_1_jcs_cyl_1]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_jcs_cyl_1]) + self.R_rbs_piston_1)]) + multi_dot([B(self.P_rbs_piston_1,self.ubar_rbs_piston_1_jcs_cyl_1).T,A(self.P_rbs_piston_1),self.Mbar_rbs_piston_1_jcs_cyl_1[:,1:2]]))]]),self.L_jcs_cyl_1])
        self.F_rbs_piston_1_jcs_cyl_1 = Q_rbs_piston_1_jcs_cyl_1[0:3]
        Te_rbs_piston_1_jcs_cyl_1 = Q_rbs_piston_1_jcs_cyl_1[3:7]
        self.T_rbs_piston_1_jcs_cyl_1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_piston_1),self.ubar_rbs_piston_1_jcs_cyl_1])),self.F_rbs_piston_1_jcs_cyl_1]) + (0.5) * multi_dot([E(self.P_rbs_piston_1),Te_rbs_piston_1_jcs_cyl_1]))
        Q_rbs_piston_1_jcs_sph_1 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_piston_1,self.ubar_rbs_piston_1_jcs_sph_1).T]]),self.L_jcs_sph_1])
        self.F_rbs_piston_1_jcs_sph_1 = Q_rbs_piston_1_jcs_sph_1[0:3]
        Te_rbs_piston_1_jcs_sph_1 = Q_rbs_piston_1_jcs_sph_1[3:7]
        self.T_rbs_piston_1_jcs_sph_1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_piston_1),self.ubar_rbs_piston_1_jcs_sph_1])),self.F_rbs_piston_1_jcs_sph_1]) + (0.5) * multi_dot([E(self.P_rbs_piston_1),Te_rbs_piston_1_jcs_sph_1]))
        Q_rbs_piston_2_jcs_cyl_2 = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_piston_2),self.Mbar_rbs_piston_2_jcs_cyl_2[:,0:1]]),multi_dot([A(self.P_rbs_piston_2),self.Mbar_rbs_piston_2_jcs_cyl_2[:,1:2]])],[multi_dot([B(self.P_rbs_piston_2,self.Mbar_rbs_piston_2_jcs_cyl_2[:,0:1]).T,A(self.P_rbs_engine_block),self.Mbar_rbs_engine_block_jcs_cyl_2[:,2:3]]),multi_dot([B(self.P_rbs_piston_2,self.Mbar_rbs_piston_2_jcs_cyl_2[:,1:2]).T,A(self.P_rbs_engine_block),self.Mbar_rbs_engine_block_jcs_cyl_2[:,2:3]]),(multi_dot([B(self.P_rbs_piston_2,self.Mbar_rbs_piston_2_jcs_cyl_2[:,0:1]).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_rbs_piston_2),self.ubar_rbs_piston_2_jcs_cyl_2]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_jcs_cyl_2]) + self.R_rbs_piston_2)]) + multi_dot([B(self.P_rbs_piston_2,self.ubar_rbs_piston_2_jcs_cyl_2).T,A(self.P_rbs_piston_2),self.Mbar_rbs_piston_2_jcs_cyl_2[:,0:1]])),(multi_dot([B(self.P_rbs_piston_2,self.Mbar_rbs_piston_2_jcs_cyl_2[:,1:2]).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_rbs_piston_2),self.ubar_rbs_piston_2_jcs_cyl_2]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_jcs_cyl_2]) + self.R_rbs_piston_2)]) + multi_dot([B(self.P_rbs_piston_2,self.ubar_rbs_piston_2_jcs_cyl_2).T,A(self.P_rbs_piston_2),self.Mbar_rbs_piston_2_jcs_cyl_2[:,1:2]]))]]),self.L_jcs_cyl_2])
        self.F_rbs_piston_2_jcs_cyl_2 = Q_rbs_piston_2_jcs_cyl_2[0:3]
        Te_rbs_piston_2_jcs_cyl_2 = Q_rbs_piston_2_jcs_cyl_2[3:7]
        self.T_rbs_piston_2_jcs_cyl_2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_piston_2),self.ubar_rbs_piston_2_jcs_cyl_2])),self.F_rbs_piston_2_jcs_cyl_2]) + (0.5) * multi_dot([E(self.P_rbs_piston_2),Te_rbs_piston_2_jcs_cyl_2]))
        Q_rbs_piston_2_jcs_sph_2 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_piston_2,self.ubar_rbs_piston_2_jcs_sph_2).T]]),self.L_jcs_sph_2])
        self.F_rbs_piston_2_jcs_sph_2 = Q_rbs_piston_2_jcs_sph_2[0:3]
        Te_rbs_piston_2_jcs_sph_2 = Q_rbs_piston_2_jcs_sph_2[3:7]
        self.T_rbs_piston_2_jcs_sph_2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_piston_2),self.ubar_rbs_piston_2_jcs_sph_2])),self.F_rbs_piston_2_jcs_sph_2]) + (0.5) * multi_dot([E(self.P_rbs_piston_2),Te_rbs_piston_2_jcs_sph_2]))
        Q_rbs_piston_3_jcs_cyl_3 = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_piston_3),self.Mbar_rbs_piston_3_jcs_cyl_3[:,0:1]]),multi_dot([A(self.P_rbs_piston_3),self.Mbar_rbs_piston_3_jcs_cyl_3[:,1:2]])],[multi_dot([B(self.P_rbs_piston_3,self.Mbar_rbs_piston_3_jcs_cyl_3[:,0:1]).T,A(self.P_rbs_engine_block),self.Mbar_rbs_engine_block_jcs_cyl_3[:,2:3]]),multi_dot([B(self.P_rbs_piston_3,self.Mbar_rbs_piston_3_jcs_cyl_3[:,1:2]).T,A(self.P_rbs_engine_block),self.Mbar_rbs_engine_block_jcs_cyl_3[:,2:3]]),(multi_dot([B(self.P_rbs_piston_3,self.Mbar_rbs_piston_3_jcs_cyl_3[:,0:1]).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_rbs_piston_3),self.ubar_rbs_piston_3_jcs_cyl_3]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_jcs_cyl_3]) + self.R_rbs_piston_3)]) + multi_dot([B(self.P_rbs_piston_3,self.ubar_rbs_piston_3_jcs_cyl_3).T,A(self.P_rbs_piston_3),self.Mbar_rbs_piston_3_jcs_cyl_3[:,0:1]])),(multi_dot([B(self.P_rbs_piston_3,self.Mbar_rbs_piston_3_jcs_cyl_3[:,1:2]).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_rbs_piston_3),self.ubar_rbs_piston_3_jcs_cyl_3]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_jcs_cyl_3]) + self.R_rbs_piston_3)]) + multi_dot([B(self.P_rbs_piston_3,self.ubar_rbs_piston_3_jcs_cyl_3).T,A(self.P_rbs_piston_3),self.Mbar_rbs_piston_3_jcs_cyl_3[:,1:2]]))]]),self.L_jcs_cyl_3])
        self.F_rbs_piston_3_jcs_cyl_3 = Q_rbs_piston_3_jcs_cyl_3[0:3]
        Te_rbs_piston_3_jcs_cyl_3 = Q_rbs_piston_3_jcs_cyl_3[3:7]
        self.T_rbs_piston_3_jcs_cyl_3 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_piston_3),self.ubar_rbs_piston_3_jcs_cyl_3])),self.F_rbs_piston_3_jcs_cyl_3]) + (0.5) * multi_dot([E(self.P_rbs_piston_3),Te_rbs_piston_3_jcs_cyl_3]))
        Q_rbs_piston_3_jcs_sph_3 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_piston_3,self.ubar_rbs_piston_3_jcs_sph_3).T]]),self.L_jcs_sph_3])
        self.F_rbs_piston_3_jcs_sph_3 = Q_rbs_piston_3_jcs_sph_3[0:3]
        Te_rbs_piston_3_jcs_sph_3 = Q_rbs_piston_3_jcs_sph_3[3:7]
        self.T_rbs_piston_3_jcs_sph_3 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_piston_3),self.ubar_rbs_piston_3_jcs_sph_3])),self.F_rbs_piston_3_jcs_sph_3]) + (0.5) * multi_dot([E(self.P_rbs_piston_3),Te_rbs_piston_3_jcs_sph_3]))
        Q_rbs_piston_4_jcs_cyl_4 = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbs_piston_4),self.Mbar_rbs_piston_4_jcs_cyl_4[:,0:1]]),multi_dot([A(self.P_rbs_piston_4),self.Mbar_rbs_piston_4_jcs_cyl_4[:,1:2]])],[multi_dot([B(self.P_rbs_piston_4,self.Mbar_rbs_piston_4_jcs_cyl_4[:,0:1]).T,A(self.P_rbs_engine_block),self.Mbar_rbs_engine_block_jcs_cyl_4[:,2:3]]),multi_dot([B(self.P_rbs_piston_4,self.Mbar_rbs_piston_4_jcs_cyl_4[:,1:2]).T,A(self.P_rbs_engine_block),self.Mbar_rbs_engine_block_jcs_cyl_4[:,2:3]]),(multi_dot([B(self.P_rbs_piston_4,self.Mbar_rbs_piston_4_jcs_cyl_4[:,0:1]).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_rbs_piston_4),self.ubar_rbs_piston_4_jcs_cyl_4]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_jcs_cyl_4]) + self.R_rbs_piston_4)]) + multi_dot([B(self.P_rbs_piston_4,self.ubar_rbs_piston_4_jcs_cyl_4).T,A(self.P_rbs_piston_4),self.Mbar_rbs_piston_4_jcs_cyl_4[:,0:1]])),(multi_dot([B(self.P_rbs_piston_4,self.Mbar_rbs_piston_4_jcs_cyl_4[:,1:2]).T,((-1) * self.R_rbs_engine_block + multi_dot([A(self.P_rbs_piston_4),self.ubar_rbs_piston_4_jcs_cyl_4]) + (-1) * multi_dot([A(self.P_rbs_engine_block),self.ubar_rbs_engine_block_jcs_cyl_4]) + self.R_rbs_piston_4)]) + multi_dot([B(self.P_rbs_piston_4,self.ubar_rbs_piston_4_jcs_cyl_4).T,A(self.P_rbs_piston_4),self.Mbar_rbs_piston_4_jcs_cyl_4[:,1:2]]))]]),self.L_jcs_cyl_4])
        self.F_rbs_piston_4_jcs_cyl_4 = Q_rbs_piston_4_jcs_cyl_4[0:3]
        Te_rbs_piston_4_jcs_cyl_4 = Q_rbs_piston_4_jcs_cyl_4[3:7]
        self.T_rbs_piston_4_jcs_cyl_4 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_piston_4),self.ubar_rbs_piston_4_jcs_cyl_4])),self.F_rbs_piston_4_jcs_cyl_4]) + (0.5) * multi_dot([E(self.P_rbs_piston_4),Te_rbs_piston_4_jcs_cyl_4]))
        Q_rbs_piston_4_jcs_sph_4 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_piston_4,self.ubar_rbs_piston_4_jcs_sph_4).T]]),self.L_jcs_sph_4])
        self.F_rbs_piston_4_jcs_sph_4 = Q_rbs_piston_4_jcs_sph_4[0:3]
        Te_rbs_piston_4_jcs_sph_4 = Q_rbs_piston_4_jcs_sph_4[3:7]
        self.T_rbs_piston_4_jcs_sph_4 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_piston_4),self.ubar_rbs_piston_4_jcs_sph_4])),self.F_rbs_piston_4_jcs_sph_4]) + (0.5) * multi_dot([E(self.P_rbs_piston_4),Te_rbs_piston_4_jcs_sph_4]))
        Q_rbs_crank_shaft_jcs_bsph_1 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_crank_shaft,self.ubar_rbs_crank_shaft_jcs_bsph_1).T]]),self.L_jcs_bsph_1])
        self.F_rbs_crank_shaft_jcs_bsph_1 = Q_rbs_crank_shaft_jcs_bsph_1[0:3]
        Te_rbs_crank_shaft_jcs_bsph_1 = Q_rbs_crank_shaft_jcs_bsph_1[3:7]
        self.T_rbs_crank_shaft_jcs_bsph_1 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_crank_shaft),self.ubar_rbs_crank_shaft_jcs_bsph_1])),self.F_rbs_crank_shaft_jcs_bsph_1]) + (0.5) * multi_dot([E(self.P_rbs_crank_shaft),Te_rbs_crank_shaft_jcs_bsph_1]))
        Q_rbs_crank_shaft_jcs_bsph_2 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_crank_shaft,self.ubar_rbs_crank_shaft_jcs_bsph_2).T]]),self.L_jcs_bsph_2])
        self.F_rbs_crank_shaft_jcs_bsph_2 = Q_rbs_crank_shaft_jcs_bsph_2[0:3]
        Te_rbs_crank_shaft_jcs_bsph_2 = Q_rbs_crank_shaft_jcs_bsph_2[3:7]
        self.T_rbs_crank_shaft_jcs_bsph_2 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_crank_shaft),self.ubar_rbs_crank_shaft_jcs_bsph_2])),self.F_rbs_crank_shaft_jcs_bsph_2]) + (0.5) * multi_dot([E(self.P_rbs_crank_shaft),Te_rbs_crank_shaft_jcs_bsph_2]))
        Q_rbs_crank_shaft_jcs_bsph_3 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_crank_shaft,self.ubar_rbs_crank_shaft_jcs_bsph_3).T]]),self.L_jcs_bsph_3])
        self.F_rbs_crank_shaft_jcs_bsph_3 = Q_rbs_crank_shaft_jcs_bsph_3[0:3]
        Te_rbs_crank_shaft_jcs_bsph_3 = Q_rbs_crank_shaft_jcs_bsph_3[3:7]
        self.T_rbs_crank_shaft_jcs_bsph_3 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_crank_shaft),self.ubar_rbs_crank_shaft_jcs_bsph_3])),self.F_rbs_crank_shaft_jcs_bsph_3]) + (0.5) * multi_dot([E(self.P_rbs_crank_shaft),Te_rbs_crank_shaft_jcs_bsph_3]))
        Q_rbs_crank_shaft_jcs_bsph_4 = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbs_crank_shaft,self.ubar_rbs_crank_shaft_jcs_bsph_4).T]]),self.L_jcs_bsph_4])
        self.F_rbs_crank_shaft_jcs_bsph_4 = Q_rbs_crank_shaft_jcs_bsph_4[0:3]
        Te_rbs_crank_shaft_jcs_bsph_4 = Q_rbs_crank_shaft_jcs_bsph_4[3:7]
        self.T_rbs_crank_shaft_jcs_bsph_4 = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_crank_shaft),self.ubar_rbs_crank_shaft_jcs_bsph_4])),self.F_rbs_crank_shaft_jcs_bsph_4]) + (0.5) * multi_dot([E(self.P_rbs_crank_shaft),Te_rbs_crank_shaft_jcs_bsph_4]))
        Q_rbs_crank_shaft_jcs_crank_joint = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbs_crank_shaft,self.ubar_rbs_crank_shaft_jcs_crank_joint).T,multi_dot([B(self.P_rbs_crank_shaft,self.Mbar_rbs_crank_shaft_jcs_crank_joint[:,0:1]).T,A(self.P_rbs_engine_block),self.Mbar_rbs_engine_block_jcs_crank_joint[:,2:3]]),multi_dot([B(self.P_rbs_crank_shaft,self.Mbar_rbs_crank_shaft_jcs_crank_joint[:,1:2]).T,A(self.P_rbs_engine_block),self.Mbar_rbs_engine_block_jcs_crank_joint[:,2:3]])]]),self.L_jcs_crank_joint])
        self.F_rbs_crank_shaft_jcs_crank_joint = Q_rbs_crank_shaft_jcs_crank_joint[0:3]
        Te_rbs_crank_shaft_jcs_crank_joint = Q_rbs_crank_shaft_jcs_crank_joint[3:7]
        self.T_rbs_crank_shaft_jcs_crank_joint = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbs_crank_shaft),self.ubar_rbs_crank_shaft_jcs_crank_joint])),self.F_rbs_crank_shaft_jcs_crank_joint]) + (0.5) * multi_dot([E(self.P_rbs_crank_shaft),Te_rbs_crank_shaft_jcs_crank_joint]))

        self.reactions = {'F_ground_fas_bush_1' : self.F_ground_fas_bush_1,
                        'T_ground_fas_bush_1' : self.T_ground_fas_bush_1,
                        'F_ground_fas_bush_2' : self.F_ground_fas_bush_2,
                        'T_ground_fas_bush_2' : self.T_ground_fas_bush_2,
                        'F_ground_fas_bush_3' : self.F_ground_fas_bush_3,
                        'T_ground_fas_bush_3' : self.T_ground_fas_bush_3,
                        'F_ground_fas_bush_4' : self.F_ground_fas_bush_4,
                        'T_ground_fas_bush_4' : self.T_ground_fas_bush_4,
                        'F_rbs_piston_1_jcs_cyl_1' : self.F_rbs_piston_1_jcs_cyl_1,
                        'T_rbs_piston_1_jcs_cyl_1' : self.T_rbs_piston_1_jcs_cyl_1,
                        'F_rbs_piston_1_jcs_sph_1' : self.F_rbs_piston_1_jcs_sph_1,
                        'T_rbs_piston_1_jcs_sph_1' : self.T_rbs_piston_1_jcs_sph_1,
                        'F_rbs_piston_2_jcs_cyl_2' : self.F_rbs_piston_2_jcs_cyl_2,
                        'T_rbs_piston_2_jcs_cyl_2' : self.T_rbs_piston_2_jcs_cyl_2,
                        'F_rbs_piston_2_jcs_sph_2' : self.F_rbs_piston_2_jcs_sph_2,
                        'T_rbs_piston_2_jcs_sph_2' : self.T_rbs_piston_2_jcs_sph_2,
                        'F_rbs_piston_3_jcs_cyl_3' : self.F_rbs_piston_3_jcs_cyl_3,
                        'T_rbs_piston_3_jcs_cyl_3' : self.T_rbs_piston_3_jcs_cyl_3,
                        'F_rbs_piston_3_jcs_sph_3' : self.F_rbs_piston_3_jcs_sph_3,
                        'T_rbs_piston_3_jcs_sph_3' : self.T_rbs_piston_3_jcs_sph_3,
                        'F_rbs_piston_4_jcs_cyl_4' : self.F_rbs_piston_4_jcs_cyl_4,
                        'T_rbs_piston_4_jcs_cyl_4' : self.T_rbs_piston_4_jcs_cyl_4,
                        'F_rbs_piston_4_jcs_sph_4' : self.F_rbs_piston_4_jcs_sph_4,
                        'T_rbs_piston_4_jcs_sph_4' : self.T_rbs_piston_4_jcs_sph_4,
                        'F_rbs_crank_shaft_jcs_bsph_1' : self.F_rbs_crank_shaft_jcs_bsph_1,
                        'T_rbs_crank_shaft_jcs_bsph_1' : self.T_rbs_crank_shaft_jcs_bsph_1,
                        'F_rbs_crank_shaft_jcs_bsph_2' : self.F_rbs_crank_shaft_jcs_bsph_2,
                        'T_rbs_crank_shaft_jcs_bsph_2' : self.T_rbs_crank_shaft_jcs_bsph_2,
                        'F_rbs_crank_shaft_jcs_bsph_3' : self.F_rbs_crank_shaft_jcs_bsph_3,
                        'T_rbs_crank_shaft_jcs_bsph_3' : self.T_rbs_crank_shaft_jcs_bsph_3,
                        'F_rbs_crank_shaft_jcs_bsph_4' : self.F_rbs_crank_shaft_jcs_bsph_4,
                        'T_rbs_crank_shaft_jcs_bsph_4' : self.T_rbs_crank_shaft_jcs_bsph_4,
                        'F_rbs_crank_shaft_jcs_crank_joint' : self.F_rbs_crank_shaft_jcs_crank_joint,
                        'T_rbs_crank_shaft_jcs_crank_joint' : self.T_rbs_crank_shaft_jcs_crank_joint}

