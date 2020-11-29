
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

        self.indicies_map = {'ground': 0, 'rbr_uca': 1, 'rbl_uca': 2, 'rbr_lca': 3, 'rbl_lca': 4, 'rbr_upright': 5, 'rbl_upright': 6, 'rbr_upper_strut': 7, 'rbl_upper_strut': 8, 'rbr_lower_strut': 9, 'rbl_lower_strut': 10, 'rbr_tie_rod': 11, 'rbl_tie_rod': 12, 'rbr_hub': 13, 'rbl_hub': 14}

        self.n  = 105
        self.nc = 105
        self.nrows = 64
        self.ncols = 2*15
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_rbr_uca_jcr_uca_chassis', 'T_rbr_uca_jcr_uca_chassis', 'F_rbr_uca_jcr_uca_upright', 'T_rbr_uca_jcr_uca_upright', 'F_rbl_uca_jcl_uca_chassis', 'T_rbl_uca_jcl_uca_chassis', 'F_rbl_uca_jcl_uca_upright', 'T_rbl_uca_jcl_uca_upright', 'F_rbr_lca_jcr_lca_chassis', 'T_rbr_lca_jcr_lca_chassis', 'F_rbr_lca_jcr_lca_upright', 'T_rbr_lca_jcr_lca_upright', 'F_rbl_lca_jcl_lca_chassis', 'T_rbl_lca_jcl_lca_chassis', 'F_rbl_lca_jcl_lca_upright', 'T_rbl_lca_jcl_lca_upright', 'F_rbr_upright_jcr_hub_bearing', 'T_rbr_upright_jcr_hub_bearing', 'F_rbr_upright_mcr_wheel_lock', 'T_rbr_upright_mcr_wheel_lock', 'F_rbl_upright_jcl_hub_bearing', 'T_rbl_upright_jcl_hub_bearing', 'F_rbl_upright_mcl_wheel_lock', 'T_rbl_upright_mcl_wheel_lock', 'F_rbr_upper_strut_jcr_strut_chassis', 'T_rbr_upper_strut_jcr_strut_chassis', 'F_rbr_upper_strut_jcr_strut', 'T_rbr_upper_strut_jcr_strut', 'F_rbr_upper_strut_far_strut', 'T_rbr_upper_strut_far_strut', 'F_rbl_upper_strut_jcl_strut_chassis', 'T_rbl_upper_strut_jcl_strut_chassis', 'F_rbl_upper_strut_jcl_strut', 'T_rbl_upper_strut_jcl_strut', 'F_rbl_upper_strut_fal_strut', 'T_rbl_upper_strut_fal_strut', 'F_rbr_lower_strut_jcr_strut_lca', 'T_rbr_lower_strut_jcr_strut_lca', 'F_rbl_lower_strut_jcl_strut_lca', 'T_rbl_lower_strut_jcl_strut_lca', 'F_rbr_tie_rod_jcr_tie_steering', 'T_rbr_tie_rod_jcr_tie_steering', 'F_rbr_tie_rod_jcr_tie_upright', 'T_rbr_tie_rod_jcr_tie_upright', 'F_rbl_tie_rod_jcl_tie_steering', 'T_rbl_tie_rod_jcl_tie_steering', 'F_rbl_tie_rod_jcl_tie_upright', 'T_rbl_tie_rod_jcl_tie_upright', 'F_rbr_hub_mcr_wheel_travel', 'T_rbr_hub_mcr_wheel_travel', 'F_rbl_hub_mcl_wheel_travel', 'T_rbl_hub_mcl_wheel_travel']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.ground*2, self.ground*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.ground*2, self.ground*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.ground*2, self.ground*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.ground*2, self.ground*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.ground*2, self.ground*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.ground*2, self.ground*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.ground*2, self.ground*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.ground*2, self.ground*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.ground*2, self.ground*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.ground*2, self.ground*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.ground*2, self.ground*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.ground*2, self.ground*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.ground*2, self.ground*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.ground*2, self.ground*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.ground*2, self.ground*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.ground*2, self.ground*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.ground*2, self.ground*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.ground*2, self.ground*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.ground*2, self.ground*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.ground*2, self.ground*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.ground*2, self.ground*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_hub*2, self.rbl_hub*2+1], dtype=np.intc)

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
        self.config.R_rbr_uca,
        self.config.P_rbr_uca,
        self.config.R_rbl_uca,
        self.config.P_rbl_uca,
        self.config.R_rbr_lca,
        self.config.P_rbr_lca,
        self.config.R_rbl_lca,
        self.config.P_rbl_lca,
        self.config.R_rbr_upright,
        self.config.P_rbr_upright,
        self.config.R_rbl_upright,
        self.config.P_rbl_upright,
        self.config.R_rbr_upper_strut,
        self.config.P_rbr_upper_strut,
        self.config.R_rbl_upper_strut,
        self.config.P_rbl_upper_strut,
        self.config.R_rbr_lower_strut,
        self.config.P_rbr_lower_strut,
        self.config.R_rbl_lower_strut,
        self.config.P_rbl_lower_strut,
        self.config.R_rbr_tie_rod,
        self.config.P_rbr_tie_rod,
        self.config.R_rbl_tie_rod,
        self.config.P_rbl_tie_rod,
        self.config.R_rbr_hub,
        self.config.P_rbr_hub,
        self.config.R_rbl_hub,
        self.config.P_rbl_hub], out=self._q)

        np.concatenate([self.config.Rd_ground,
        self.config.Pd_ground,
        self.config.Rd_rbr_uca,
        self.config.Pd_rbr_uca,
        self.config.Rd_rbl_uca,
        self.config.Pd_rbl_uca,
        self.config.Rd_rbr_lca,
        self.config.Pd_rbr_lca,
        self.config.Rd_rbl_lca,
        self.config.Pd_rbl_lca,
        self.config.Rd_rbr_upright,
        self.config.Pd_rbr_upright,
        self.config.Rd_rbl_upright,
        self.config.Pd_rbl_upright,
        self.config.Rd_rbr_upper_strut,
        self.config.Pd_rbr_upper_strut,
        self.config.Rd_rbl_upper_strut,
        self.config.Pd_rbl_upper_strut,
        self.config.Rd_rbr_lower_strut,
        self.config.Pd_rbr_lower_strut,
        self.config.Rd_rbl_lower_strut,
        self.config.Pd_rbl_lower_strut,
        self.config.Rd_rbr_tie_rod,
        self.config.Pd_rbr_tie_rod,
        self.config.Rd_rbl_tie_rod,
        self.config.Pd_rbl_tie_rod,
        self.config.Rd_rbr_hub,
        self.config.Pd_rbr_hub,
        self.config.Rd_rbl_hub,
        self.config.Pd_rbl_hub], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p + 'ground']
        self.rbr_uca = indicies_map[p + 'rbr_uca']
        self.rbl_uca = indicies_map[p + 'rbl_uca']
        self.rbr_lca = indicies_map[p + 'rbr_lca']
        self.rbl_lca = indicies_map[p + 'rbl_lca']
        self.rbr_upright = indicies_map[p + 'rbr_upright']
        self.rbl_upright = indicies_map[p + 'rbl_upright']
        self.rbr_upper_strut = indicies_map[p + 'rbr_upper_strut']
        self.rbl_upper_strut = indicies_map[p + 'rbl_upper_strut']
        self.rbr_lower_strut = indicies_map[p + 'rbr_lower_strut']
        self.rbl_lower_strut = indicies_map[p + 'rbl_lower_strut']
        self.rbr_tie_rod = indicies_map[p + 'rbr_tie_rod']
        self.rbl_tie_rod = indicies_map[p + 'rbl_tie_rod']
        self.rbr_hub = indicies_map[p + 'rbr_hub']
        self.rbl_hub = indicies_map[p + 'rbl_hub']
    

    
    def eval_constants(self):
        config = self.config

        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.m_ground = 1.0
        self.Jbar_ground = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.F_rbr_uca_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_uca]], dtype=np.float64)
        self.F_rbl_uca_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_uca]], dtype=np.float64)
        self.F_rbr_lca_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_lca]], dtype=np.float64)
        self.F_rbl_lca_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_lca]], dtype=np.float64)
        self.F_rbr_upright_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_upright]], dtype=np.float64)
        self.F_rbl_upright_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_upright]], dtype=np.float64)
        self.F_rbr_upper_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_upper_strut]], dtype=np.float64)
        self.T_rbr_upper_strut_far_strut = Z3x1
        self.T_rbr_lower_strut_far_strut = Z3x1
        self.F_rbl_upper_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_upper_strut]], dtype=np.float64)
        self.T_rbl_upper_strut_fal_strut = Z3x1
        self.T_rbl_lower_strut_fal_strut = Z3x1
        self.F_rbr_lower_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_lower_strut]], dtype=np.float64)
        self.F_rbl_lower_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_lower_strut]], dtype=np.float64)
        self.F_rbr_tie_rod_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_tie_rod]], dtype=np.float64)
        self.F_rbl_tie_rod_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_tie_rod]], dtype=np.float64)
        self.F_rbr_hub_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_hub]], dtype=np.float64)
        self.F_rbl_hub_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_hub]], dtype=np.float64)

        self.M_ground = self.m_ground * I3
        self.M_rbr_uca = config.m_rbr_uca * I3
        self.M_rbl_uca = config.m_rbl_uca * I3
        self.M_rbr_lca = config.m_rbr_lca * I3
        self.M_rbl_lca = config.m_rbl_lca * I3
        self.M_rbr_upright = config.m_rbr_upright * I3
        self.M_rbl_upright = config.m_rbl_upright * I3
        self.M_rbr_upper_strut = config.m_rbr_upper_strut * I3
        self.M_rbl_upper_strut = config.m_rbl_upper_strut * I3
        self.M_rbr_lower_strut = config.m_rbr_lower_strut * I3
        self.M_rbl_lower_strut = config.m_rbl_lower_strut * I3
        self.M_rbr_tie_rod = config.m_rbr_tie_rod * I3
        self.M_rbl_tie_rod = config.m_rbl_tie_rod * I3
        self.M_rbr_hub = config.m_rbr_hub * I3
        self.M_rbl_hub = config.m_rbl_hub * I3
        self.Mbar_rbr_uca_jcr_uca_chassis = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_uca_chassis)])
        self.Mbar_ground_jcr_uca_chassis = multi_dot([A(self.P_ground).T,triad(config.ax1_jcr_uca_chassis)])
        self.ubar_rbr_uca_jcr_uca_chassis = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_uca_chassis]) + (-1) * multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_ground_jcr_uca_chassis = (multi_dot([A(self.P_ground).T,config.pt1_jcr_uca_chassis]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_uca_upright)])
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_uca_upright)])
        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_uca_upright]) + (-1) * multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_uca_upright]) + (-1) * multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbl_uca_jcl_uca_chassis = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_uca_chassis)])
        self.Mbar_ground_jcl_uca_chassis = multi_dot([A(self.P_ground).T,triad(config.ax1_jcl_uca_chassis)])
        self.ubar_rbl_uca_jcl_uca_chassis = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_uca_chassis]) + (-1) * multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_ground_jcl_uca_chassis = (multi_dot([A(self.P_ground).T,config.pt1_jcl_uca_chassis]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_uca_upright)])
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_uca_upright)])
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_uca_upright]) + (-1) * multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_uca_upright]) + (-1) * multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbr_lca_jcr_lca_chassis = multi_dot([A(config.P_rbr_lca).T,triad(config.ax1_jcr_lca_chassis)])
        self.Mbar_ground_jcr_lca_chassis = multi_dot([A(self.P_ground).T,triad(config.ax1_jcr_lca_chassis)])
        self.ubar_rbr_lca_jcr_lca_chassis = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_lca_chassis]) + (-1) * multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.ubar_ground_jcr_lca_chassis = (multi_dot([A(self.P_ground).T,config.pt1_jcr_lca_chassis]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([A(config.P_rbr_lca).T,triad(config.ax1_jcr_lca_upright)])
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_lca_upright)])
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_lca_upright]) + (-1) * multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_lca_upright]) + (-1) * multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbl_lca_jcl_lca_chassis = multi_dot([A(config.P_rbl_lca).T,triad(config.ax1_jcl_lca_chassis)])
        self.Mbar_ground_jcl_lca_chassis = multi_dot([A(self.P_ground).T,triad(config.ax1_jcl_lca_chassis)])
        self.ubar_rbl_lca_jcl_lca_chassis = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_lca_chassis]) + (-1) * multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.ubar_ground_jcl_lca_chassis = (multi_dot([A(self.P_ground).T,config.pt1_jcl_lca_chassis]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([A(config.P_rbl_lca).T,triad(config.ax1_jcl_lca_upright)])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_lca_upright)])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_lca_upright]) + (-1) * multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_lca_upright]) + (-1) * multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_hub_bearing)])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([A(config.P_rbr_hub).T,triad(config.ax1_jcr_hub_bearing)])
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_hub_bearing]) + (-1) * multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([A(config.P_rbr_hub).T,config.pt1_jcr_hub_bearing]) + (-1) * multi_dot([A(config.P_rbr_hub).T,config.R_rbr_hub]))
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_hub_bearing)])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([A(config.P_rbr_hub).T,triad(config.ax1_jcr_hub_bearing)])
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_hub_bearing)])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([A(config.P_rbl_hub).T,triad(config.ax1_jcl_hub_bearing)])
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_hub_bearing]) + (-1) * multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([A(config.P_rbl_hub).T,config.pt1_jcl_hub_bearing]) + (-1) * multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_hub_bearing)])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([A(config.P_rbl_hub).T,triad(config.ax1_jcl_hub_bearing)])
        self.Mbar_rbr_upper_strut_jcr_strut_chassis = multi_dot([A(config.P_rbr_upper_strut).T,triad(config.ax1_jcr_strut_chassis)])
        self.Mbar_ground_jcr_strut_chassis = multi_dot([A(self.P_ground).T,triad(config.ax2_jcr_strut_chassis,triad(config.ax1_jcr_strut_chassis)[0:3,1:2])])
        self.ubar_rbr_upper_strut_jcr_strut_chassis = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_jcr_strut_chassis]) + (-1) * multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_ground_jcr_strut_chassis = (multi_dot([A(self.P_ground).T,config.pt1_jcr_strut_chassis]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([A(config.P_rbr_upper_strut).T,triad(config.ax1_jcr_strut)])
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([A(config.P_rbr_lower_strut).T,triad(config.ax1_jcr_strut)])
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_jcr_strut]) + (-1) * multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt1_jcr_strut]) + (-1) * multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.ubar_rbr_upper_strut_far_strut = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_far_strut]) + (-1) * multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_rbr_lower_strut_far_strut = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt2_far_strut]) + (-1) * multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.Mbar_rbl_upper_strut_jcl_strut_chassis = multi_dot([A(config.P_rbl_upper_strut).T,triad(config.ax1_jcl_strut_chassis)])
        self.Mbar_ground_jcl_strut_chassis = multi_dot([A(self.P_ground).T,triad(config.ax2_jcl_strut_chassis,triad(config.ax1_jcl_strut_chassis)[0:3,1:2])])
        self.ubar_rbl_upper_strut_jcl_strut_chassis = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_jcl_strut_chassis]) + (-1) * multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_ground_jcl_strut_chassis = (multi_dot([A(self.P_ground).T,config.pt1_jcl_strut_chassis]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([A(config.P_rbl_upper_strut).T,triad(config.ax1_jcl_strut)])
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([A(config.P_rbl_lower_strut).T,triad(config.ax1_jcl_strut)])
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_jcl_strut]) + (-1) * multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt1_jcl_strut]) + (-1) * multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.ubar_rbl_upper_strut_fal_strut = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_fal_strut]) + (-1) * multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_rbl_lower_strut_fal_strut = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt2_fal_strut]) + (-1) * multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.Mbar_rbr_lower_strut_jcr_strut_lca = multi_dot([A(config.P_rbr_lower_strut).T,triad(config.ax1_jcr_strut_lca)])
        self.Mbar_rbr_lca_jcr_strut_lca = multi_dot([A(config.P_rbr_lca).T,triad(config.ax2_jcr_strut_lca,triad(config.ax1_jcr_strut_lca)[0:3,1:2])])
        self.ubar_rbr_lower_strut_jcr_strut_lca = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt1_jcr_strut_lca]) + (-1) * multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.ubar_rbr_lca_jcr_strut_lca = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_strut_lca]) + (-1) * multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.Mbar_rbl_lower_strut_jcl_strut_lca = multi_dot([A(config.P_rbl_lower_strut).T,triad(config.ax1_jcl_strut_lca)])
        self.Mbar_rbl_lca_jcl_strut_lca = multi_dot([A(config.P_rbl_lca).T,triad(config.ax2_jcl_strut_lca,triad(config.ax1_jcl_strut_lca)[0:3,1:2])])
        self.ubar_rbl_lower_strut_jcl_strut_lca = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt1_jcl_strut_lca]) + (-1) * multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.ubar_rbl_lca_jcl_strut_lca = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_strut_lca]) + (-1) * multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.Mbar_rbr_tie_rod_jcr_tie_steering = multi_dot([A(config.P_rbr_tie_rod).T,triad(config.ax1_jcr_tie_steering)])
        self.Mbar_ground_jcr_tie_steering = multi_dot([A(self.P_ground).T,triad(config.ax2_jcr_tie_steering,triad(config.ax1_jcr_tie_steering)[0:3,1:2])])
        self.ubar_rbr_tie_rod_jcr_tie_steering = (multi_dot([A(config.P_rbr_tie_rod).T,config.pt1_jcr_tie_steering]) + (-1) * multi_dot([A(config.P_rbr_tie_rod).T,config.R_rbr_tie_rod]))
        self.ubar_ground_jcr_tie_steering = (multi_dot([A(self.P_ground).T,config.pt1_jcr_tie_steering]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([A(config.P_rbr_tie_rod).T,triad(config.ax1_jcr_tie_upright)])
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_tie_upright)])
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([A(config.P_rbr_tie_rod).T,config.pt1_jcr_tie_upright]) + (-1) * multi_dot([A(config.P_rbr_tie_rod).T,config.R_rbr_tie_rod]))
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_tie_upright]) + (-1) * multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbl_tie_rod_jcl_tie_steering = multi_dot([A(config.P_rbl_tie_rod).T,triad(config.ax1_jcl_tie_steering)])
        self.Mbar_ground_jcl_tie_steering = multi_dot([A(self.P_ground).T,triad(config.ax2_jcl_tie_steering,triad(config.ax1_jcl_tie_steering)[0:3,1:2])])
        self.ubar_rbl_tie_rod_jcl_tie_steering = (multi_dot([A(config.P_rbl_tie_rod).T,config.pt1_jcl_tie_steering]) + (-1) * multi_dot([A(config.P_rbl_tie_rod).T,config.R_rbl_tie_rod]))
        self.ubar_ground_jcl_tie_steering = (multi_dot([A(self.P_ground).T,config.pt1_jcl_tie_steering]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([A(config.P_rbl_tie_rod).T,triad(config.ax1_jcl_tie_upright)])
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_tie_upright)])
        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([A(config.P_rbl_tie_rod).T,config.pt1_jcl_tie_upright]) + (-1) * multi_dot([A(config.P_rbl_tie_rod).T,config.R_rbl_tie_rod]))
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_tie_upright]) + (-1) * multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.ubar_rbr_hub_mcr_wheel_travel = (multi_dot([A(config.P_rbr_hub).T,config.pt1_mcr_wheel_travel]) + (-1) * multi_dot([A(config.P_rbr_hub).T,config.R_rbr_hub]))
        self.ubar_ground_mcr_wheel_travel = (multi_dot([A(self.P_ground).T,config.pt2_mcr_wheel_travel]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbr_hub_far_tire = (multi_dot([A(config.P_rbr_hub).T,config.pt1_far_tire]) + (-1) * multi_dot([A(config.P_rbr_hub).T,config.R_rbr_hub]))
        self.ubar_ground_far_tire = (multi_dot([A(self.P_ground).T,config.pt1_far_tire]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbl_hub_mcl_wheel_travel = (multi_dot([A(config.P_rbl_hub).T,config.pt1_mcl_wheel_travel]) + (-1) * multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
        self.ubar_ground_mcl_wheel_travel = (multi_dot([A(self.P_ground).T,config.pt2_mcl_wheel_travel]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
        self.ubar_rbl_hub_fal_tire = (multi_dot([A(config.P_rbl_hub).T,config.pt1_fal_tire]) + (-1) * multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
        self.ubar_ground_fal_tire = (multi_dot([A(self.P_ground).T,config.pt1_fal_tire]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))

    
    def _map_gen_coordinates(self):
        q = self._q
        self.R_ground = q[0:3]
        self.P_ground = q[3:7]
        self.R_rbr_uca = q[7:10]
        self.P_rbr_uca = q[10:14]
        self.R_rbl_uca = q[14:17]
        self.P_rbl_uca = q[17:21]
        self.R_rbr_lca = q[21:24]
        self.P_rbr_lca = q[24:28]
        self.R_rbl_lca = q[28:31]
        self.P_rbl_lca = q[31:35]
        self.R_rbr_upright = q[35:38]
        self.P_rbr_upright = q[38:42]
        self.R_rbl_upright = q[42:45]
        self.P_rbl_upright = q[45:49]
        self.R_rbr_upper_strut = q[49:52]
        self.P_rbr_upper_strut = q[52:56]
        self.R_rbl_upper_strut = q[56:59]
        self.P_rbl_upper_strut = q[59:63]
        self.R_rbr_lower_strut = q[63:66]
        self.P_rbr_lower_strut = q[66:70]
        self.R_rbl_lower_strut = q[70:73]
        self.P_rbl_lower_strut = q[73:77]
        self.R_rbr_tie_rod = q[77:80]
        self.P_rbr_tie_rod = q[80:84]
        self.R_rbl_tie_rod = q[84:87]
        self.P_rbl_tie_rod = q[87:91]
        self.R_rbr_hub = q[91:94]
        self.P_rbr_hub = q[94:98]
        self.R_rbl_hub = q[98:101]
        self.P_rbl_hub = q[101:105]

    
    def _map_gen_velocities(self):
        qd = self._qd
        self.Rd_ground = qd[0:3]
        self.Pd_ground = qd[3:7]
        self.Rd_rbr_uca = qd[7:10]
        self.Pd_rbr_uca = qd[10:14]
        self.Rd_rbl_uca = qd[14:17]
        self.Pd_rbl_uca = qd[17:21]
        self.Rd_rbr_lca = qd[21:24]
        self.Pd_rbr_lca = qd[24:28]
        self.Rd_rbl_lca = qd[28:31]
        self.Pd_rbl_lca = qd[31:35]
        self.Rd_rbr_upright = qd[35:38]
        self.Pd_rbr_upright = qd[38:42]
        self.Rd_rbl_upright = qd[42:45]
        self.Pd_rbl_upright = qd[45:49]
        self.Rd_rbr_upper_strut = qd[49:52]
        self.Pd_rbr_upper_strut = qd[52:56]
        self.Rd_rbl_upper_strut = qd[56:59]
        self.Pd_rbl_upper_strut = qd[59:63]
        self.Rd_rbr_lower_strut = qd[63:66]
        self.Pd_rbr_lower_strut = qd[66:70]
        self.Rd_rbl_lower_strut = qd[70:73]
        self.Pd_rbl_lower_strut = qd[73:77]
        self.Rd_rbr_tie_rod = qd[77:80]
        self.Pd_rbr_tie_rod = qd[80:84]
        self.Rd_rbl_tie_rod = qd[84:87]
        self.Pd_rbl_tie_rod = qd[87:91]
        self.Rd_rbr_hub = qd[91:94]
        self.Pd_rbr_hub = qd[94:98]
        self.Rd_rbl_hub = qd[98:101]
        self.Pd_rbl_hub = qd[101:105]

    
    def _map_gen_accelerations(self):
        qdd = self._qdd
        self.Rdd_ground = qdd[0:3]
        self.Pdd_ground = qdd[3:7]
        self.Rdd_rbr_uca = qdd[7:10]
        self.Pdd_rbr_uca = qdd[10:14]
        self.Rdd_rbl_uca = qdd[14:17]
        self.Pdd_rbl_uca = qdd[17:21]
        self.Rdd_rbr_lca = qdd[21:24]
        self.Pdd_rbr_lca = qdd[24:28]
        self.Rdd_rbl_lca = qdd[28:31]
        self.Pdd_rbl_lca = qdd[31:35]
        self.Rdd_rbr_upright = qdd[35:38]
        self.Pdd_rbr_upright = qdd[38:42]
        self.Rdd_rbl_upright = qdd[42:45]
        self.Pdd_rbl_upright = qdd[45:49]
        self.Rdd_rbr_upper_strut = qdd[49:52]
        self.Pdd_rbr_upper_strut = qdd[52:56]
        self.Rdd_rbl_upper_strut = qdd[56:59]
        self.Pdd_rbl_upper_strut = qdd[59:63]
        self.Rdd_rbr_lower_strut = qdd[63:66]
        self.Pdd_rbr_lower_strut = qdd[66:70]
        self.Rdd_rbl_lower_strut = qdd[70:73]
        self.Pdd_rbl_lower_strut = qdd[73:77]
        self.Rdd_rbr_tie_rod = qdd[77:80]
        self.Pdd_rbr_tie_rod = qdd[80:84]
        self.Rdd_rbl_tie_rod = qdd[84:87]
        self.Pdd_rbl_tie_rod = qdd[87:91]
        self.Rdd_rbr_hub = qdd[91:94]
        self.Pdd_rbr_hub = qdd[94:98]
        self.Rdd_rbl_hub = qdd[98:101]
        self.Pdd_rbl_hub = qdd[101:105]

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcr_uca_chassis = Lambda[0:5]
        self.L_jcr_uca_upright = Lambda[5:8]
        self.L_jcl_uca_chassis = Lambda[8:13]
        self.L_jcl_uca_upright = Lambda[13:16]
        self.L_jcr_lca_chassis = Lambda[16:21]
        self.L_jcr_lca_upright = Lambda[21:24]
        self.L_jcl_lca_chassis = Lambda[24:29]
        self.L_jcl_lca_upright = Lambda[29:32]
        self.L_jcr_hub_bearing = Lambda[32:37]
        self.L_mcr_wheel_lock = Lambda[37:38]
        self.L_jcl_hub_bearing = Lambda[38:43]
        self.L_mcl_wheel_lock = Lambda[43:44]
        self.L_jcr_strut_chassis = Lambda[44:48]
        self.L_jcr_strut = Lambda[48:52]
        self.L_jcl_strut_chassis = Lambda[52:56]
        self.L_jcl_strut = Lambda[56:60]
        self.L_jcr_strut_lca = Lambda[60:64]
        self.L_jcl_strut_lca = Lambda[64:68]
        self.L_jcr_tie_steering = Lambda[68:72]
        self.L_jcr_tie_upright = Lambda[72:75]
        self.L_jcl_tie_steering = Lambda[75:79]
        self.L_jcl_tie_upright = Lambda[79:82]
        self.L_mcr_wheel_travel = Lambda[82:83]
        self.L_mcl_wheel_travel = Lambda[83:84]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbr_uca
        x1 = self.R_ground
        x2 = (-1) * x1
        x3 = self.P_rbr_uca
        x4 = A(x3)
        x5 = self.P_ground
        x6 = A(x5)
        x7 = x4.T
        x8 = self.Mbar_ground_jcr_uca_chassis[:,2:3]
        x9 = self.R_rbr_upright
        x10 = (-1) * x9
        x11 = self.P_rbr_upright
        x12 = A(x11)
        x13 = self.R_rbl_uca
        x14 = self.P_rbl_uca
        x15 = A(x14)
        x16 = x15.T
        x17 = self.Mbar_ground_jcl_uca_chassis[:,2:3]
        x18 = self.R_rbl_upright
        x19 = (-1) * x18
        x20 = self.P_rbl_upright
        x21 = A(x20)
        x22 = self.R_rbr_lca
        x23 = self.P_rbr_lca
        x24 = A(x23)
        x25 = x24.T
        x26 = self.Mbar_ground_jcr_lca_chassis[:,2:3]
        x27 = self.R_rbl_lca
        x28 = self.P_rbl_lca
        x29 = A(x28)
        x30 = x29.T
        x31 = self.Mbar_ground_jcl_lca_chassis[:,2:3]
        x32 = self.R_rbr_hub
        x33 = self.P_rbr_hub
        x34 = A(x33)
        x35 = x12.T
        x36 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x37 = t
        x38 = config.UF_mcr_wheel_lock(x37)
        x39 = self.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        x40 = self.R_rbl_hub
        x41 = self.P_rbl_hub
        x42 = A(x41)
        x43 = x21.T
        x44 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x45 = config.UF_mcl_wheel_lock(x37)
        x46 = self.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        x47 = self.R_rbr_upper_strut
        x48 = self.P_rbr_upper_strut
        x49 = A(x48)
        x50 = x49.T
        x51 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x52 = self.P_rbr_lower_strut
        x53 = A(x52)
        x54 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x55 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x56 = self.R_rbr_lower_strut
        x57 = (x47 + (-1) * x56 + multi_dot([x49,self.ubar_rbr_upper_strut_jcr_strut]) + (-1) * multi_dot([x53,self.ubar_rbr_lower_strut_jcr_strut]))
        x58 = self.R_rbl_upper_strut
        x59 = self.P_rbl_upper_strut
        x60 = A(x59)
        x61 = x60.T
        x62 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x63 = self.P_rbl_lower_strut
        x64 = A(x63)
        x65 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x66 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x67 = self.R_rbl_lower_strut
        x68 = (x58 + (-1) * x67 + multi_dot([x60,self.ubar_rbl_upper_strut_jcl_strut]) + (-1) * multi_dot([x64,self.ubar_rbl_lower_strut_jcl_strut]))
        x69 = self.R_rbr_tie_rod
        x70 = self.P_rbr_tie_rod
        x71 = A(x70)
        x72 = self.R_rbl_tie_rod
        x73 = self.P_rbl_tie_rod
        x74 = A(x73)
        x75 = I1
        x76 = x75
        x77 = (-1) * x75

        self.pos_eq_blocks = ((x0 + x2 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_chassis]) + (-1) * multi_dot([x6,self.ubar_ground_jcr_uca_chassis])),
        multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1].T,x7,x6,x8]),
        multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2].T,x7,x6,x8]),
        (x0 + x10 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_upright]) + (-1) * multi_dot([x12,self.ubar_rbr_upright_jcr_uca_upright])),
        (x13 + x2 + multi_dot([x15,self.ubar_rbl_uca_jcl_uca_chassis]) + (-1) * multi_dot([x6,self.ubar_ground_jcl_uca_chassis])),
        multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1].T,x16,x6,x17]),
        multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2].T,x16,x6,x17]),
        (x13 + x19 + multi_dot([x15,self.ubar_rbl_uca_jcl_uca_upright]) + (-1) * multi_dot([x21,self.ubar_rbl_upright_jcl_uca_upright])),
        (x22 + x2 + multi_dot([x24,self.ubar_rbr_lca_jcr_lca_chassis]) + (-1) * multi_dot([x6,self.ubar_ground_jcr_lca_chassis])),
        multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1].T,x25,x6,x26]),
        multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2].T,x25,x6,x26]),
        (x22 + x10 + multi_dot([x24,self.ubar_rbr_lca_jcr_lca_upright]) + (-1) * multi_dot([x12,self.ubar_rbr_upright_jcr_lca_upright])),
        (x27 + x2 + multi_dot([x29,self.ubar_rbl_lca_jcl_lca_chassis]) + (-1) * multi_dot([x6,self.ubar_ground_jcl_lca_chassis])),
        multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1].T,x30,x6,x31]),
        multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2].T,x30,x6,x31]),
        (x27 + x19 + multi_dot([x29,self.ubar_rbl_lca_jcl_lca_upright]) + (-1) * multi_dot([x21,self.ubar_rbl_upright_jcl_lca_upright])),
        (x9 + (-1) * x32 + multi_dot([x12,self.ubar_rbr_upright_jcr_hub_bearing]) + (-1) * multi_dot([x34,self.ubar_rbr_hub_jcr_hub_bearing])),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x35,x34,x36]),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x35,x34,x36]),
        (cos(x38) * multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x35,x34,x39]) + (-1 * sin(x38)) * multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x35,x34,x39])),
        (x18 + (-1) * x40 + multi_dot([x21,self.ubar_rbl_upright_jcl_hub_bearing]) + (-1) * multi_dot([x42,self.ubar_rbl_hub_jcl_hub_bearing])),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x43,x42,x44]),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x43,x42,x44]),
        (cos(x45) * multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x43,x42,x46]) + (-1 * sin(x45)) * multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x43,x42,x46])),
        (x47 + x2 + multi_dot([x49,self.ubar_rbr_upper_strut_jcr_strut_chassis]) + (-1) * multi_dot([x6,self.ubar_ground_jcr_strut_chassis])),
        multi_dot([self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x50,x6,self.Mbar_ground_jcr_strut_chassis[:,0:1]]),
        multi_dot([x51,x50,x53,x54]),
        multi_dot([x55,x50,x53,x54]),
        multi_dot([x51,x50,x57]),
        multi_dot([x55,x50,x57]),
        (x58 + x2 + multi_dot([x60,self.ubar_rbl_upper_strut_jcl_strut_chassis]) + (-1) * multi_dot([x6,self.ubar_ground_jcl_strut_chassis])),
        multi_dot([self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1].T,x61,x6,self.Mbar_ground_jcl_strut_chassis[:,0:1]]),
        multi_dot([x62,x61,x64,x65]),
        multi_dot([x66,x61,x64,x65]),
        multi_dot([x62,x61,x68]),
        multi_dot([x66,x61,x68]),
        (x56 + (-1) * x22 + multi_dot([x53,self.ubar_rbr_lower_strut_jcr_strut_lca]) + (-1) * multi_dot([x24,self.ubar_rbr_lca_jcr_strut_lca])),
        multi_dot([self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1].T,x53.T,x24,self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]]),
        (x67 + (-1) * x27 + multi_dot([x64,self.ubar_rbl_lower_strut_jcl_strut_lca]) + (-1) * multi_dot([x29,self.ubar_rbl_lca_jcl_strut_lca])),
        multi_dot([self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1].T,x64.T,x29,self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]]),
        (x69 + x2 + multi_dot([x71,self.ubar_rbr_tie_rod_jcr_tie_steering]) + (-1) * multi_dot([x6,self.ubar_ground_jcr_tie_steering])),
        multi_dot([self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x71.T,x6,self.Mbar_ground_jcr_tie_steering[:,0:1]]),
        (x69 + x10 + multi_dot([x71,self.ubar_rbr_tie_rod_jcr_tie_upright]) + (-1) * multi_dot([x12,self.ubar_rbr_upright_jcr_tie_upright])),
        (x72 + x2 + multi_dot([x74,self.ubar_rbl_tie_rod_jcl_tie_steering]) + (-1) * multi_dot([x6,self.ubar_ground_jcl_tie_steering])),
        multi_dot([self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x74.T,x6,self.Mbar_ground_jcl_tie_steering[:,0:1]]),
        (x72 + x19 + multi_dot([x74,self.ubar_rbl_tie_rod_jcl_tie_upright]) + (-1) * multi_dot([x21,self.ubar_rbl_upright_jcl_tie_upright])),
        ((-1 * config.UF_mcr_wheel_travel(t)) * x76 + (x32 + multi_dot([x34,self.ubar_rbr_hub_mcr_wheel_travel]))[2:3] + (-1) * (x1 + multi_dot([x6,self.ubar_ground_mcr_wheel_travel]))[2:3]),
        ((-1 * config.UF_mcl_wheel_travel(t)) * x76 + (x40 + multi_dot([x42,self.ubar_rbl_hub_mcl_wheel_travel]))[2:3] + (-1) * (x1 + multi_dot([x6,self.ubar_ground_mcl_wheel_travel]))[2:3]),
        x1,
        (x5 + (-1) * self.Pg_ground),
        (x77 + multi_dot([x3.T,x3])),
        (x77 + multi_dot([x14.T,x14])),
        (x77 + multi_dot([x23.T,x23])),
        (x77 + multi_dot([x28.T,x28])),
        (x77 + multi_dot([x11.T,x11])),
        (x77 + multi_dot([x20.T,x20])),
        (x77 + multi_dot([x48.T,x48])),
        (x77 + multi_dot([x59.T,x59])),
        (x77 + multi_dot([x52.T,x52])),
        (x77 + multi_dot([x63.T,x63])),
        (x77 + multi_dot([x70.T,x70])),
        (x77 + multi_dot([x73.T,x73])),
        (x77 + multi_dot([x33.T,x33])),
        (x77 + multi_dot([x41.T,x41])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z3x1
        v1 = Z1x1
        v2 = I1

        self.vel_eq_blocks = (v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v1,
        (v1 + (-1 * derivative(config.UF_mcr_wheel_lock, t, 0.1, 1)) * v2),
        v0,
        v1,
        v1,
        (v1 + (-1 * derivative(config.UF_mcl_wheel_lock, t, 0.1, 1)) * v2),
        v0,
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
        v1,
        v0,
        v1,
        v0,
        v1,
        v0,
        v1,
        v0,
        v0,
        v1,
        v0,
        (v1 + (-1 * derivative(config.UF_mcr_wheel_travel, t, 0.1, 1)) * v2),
        (v1 + (-1 * derivative(config.UF_mcl_wheel_travel, t, 0.1, 1)) * v2),
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
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbr_uca
        a1 = self.Pd_ground
        a2 = self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        a3 = self.P_rbr_uca
        a4 = A(a3).T
        a5 = self.Mbar_ground_jcr_uca_chassis[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_ground
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        a13 = self.Pd_rbr_upright
        a14 = self.Pd_rbl_uca
        a15 = self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        a16 = self.P_rbl_uca
        a17 = A(a16).T
        a18 = self.Mbar_ground_jcl_uca_chassis[:,2:3]
        a19 = B(a1,a18)
        a20 = a18.T
        a21 = a14.T
        a22 = B(a8,a18)
        a23 = self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        a24 = self.Pd_rbl_upright
        a25 = self.Pd_rbr_lca
        a26 = self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        a27 = self.P_rbr_lca
        a28 = A(a27).T
        a29 = self.Mbar_ground_jcr_lca_chassis[:,2:3]
        a30 = B(a1,a29)
        a31 = a29.T
        a32 = a25.T
        a33 = B(a8,a29)
        a34 = self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        a35 = self.Pd_rbl_lca
        a36 = self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        a37 = self.P_rbl_lca
        a38 = A(a37).T
        a39 = self.Mbar_ground_jcl_lca_chassis[:,2:3]
        a40 = B(a1,a39)
        a41 = a39.T
        a42 = a35.T
        a43 = B(a8,a39)
        a44 = self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        a45 = self.Pd_rbr_hub
        a46 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a47 = self.P_rbr_upright
        a48 = A(a47).T
        a49 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a50 = B(a45,a49)
        a51 = a49.T
        a52 = self.P_rbr_hub
        a53 = A(a52).T
        a54 = a13.T
        a55 = B(a52,a49)
        a56 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a57 = I1
        a58 = self.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        a59 = t
        a60 = config.UF_mcr_wheel_lock(a59)
        a61 = cos(a60)
        a62 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a63 = sin(a60)
        a64 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a65 = self.Pd_rbl_hub
        a66 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a67 = self.P_rbl_upright
        a68 = A(a67).T
        a69 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a70 = B(a65,a69)
        a71 = a69.T
        a72 = self.P_rbl_hub
        a73 = A(a72).T
        a74 = a24.T
        a75 = B(a72,a69)
        a76 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a77 = self.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        a78 = config.UF_mcl_wheel_lock(a59)
        a79 = cos(a78)
        a80 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a81 = sin(a78)
        a82 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a83 = self.Pd_rbr_upper_strut
        a84 = self.Mbar_ground_jcr_strut_chassis[:,0:1]
        a85 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a86 = self.P_rbr_upper_strut
        a87 = A(a86).T
        a88 = a83.T
        a89 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a90 = a89.T
        a91 = self.Pd_rbr_lower_strut
        a92 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a93 = B(a91,a92)
        a94 = a92.T
        a95 = self.P_rbr_lower_strut
        a96 = A(a95).T
        a97 = B(a83,a89)
        a98 = B(a86,a89).T
        a99 = B(a95,a92)
        a100 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a101 = a100.T
        a102 = B(a83,a100)
        a103 = B(a86,a100).T
        a104 = self.ubar_rbr_upper_strut_jcr_strut
        a105 = self.ubar_rbr_lower_strut_jcr_strut
        a106 = (multi_dot([B(a83,a104),a83]) + (-1) * multi_dot([B(a91,a105),a91]))
        a107 = (self.Rd_rbr_upper_strut + (-1) * self.Rd_rbr_lower_strut + multi_dot([B(a86,a104),a83]) + (-1) * multi_dot([B(a95,a105),a91]))
        a108 = (self.R_rbr_upper_strut.T + (-1) * self.R_rbr_lower_strut.T + multi_dot([a104.T,a87]) + (-1) * multi_dot([a105.T,a96]))
        a109 = self.Pd_rbl_upper_strut
        a110 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a111 = self.P_rbl_upper_strut
        a112 = A(a111).T
        a113 = self.Mbar_ground_jcl_strut_chassis[:,0:1]
        a114 = a109.T
        a115 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a116 = a115.T
        a117 = self.P_rbl_lower_strut
        a118 = A(a117).T
        a119 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a120 = B(a109,a119)
        a121 = a119.T
        a122 = self.Pd_rbl_lower_strut
        a123 = B(a122,a115)
        a124 = B(a111,a119).T
        a125 = B(a117,a115)
        a126 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a127 = B(a109,a126)
        a128 = a126.T
        a129 = B(a111,a126).T
        a130 = self.ubar_rbl_upper_strut_jcl_strut
        a131 = self.ubar_rbl_lower_strut_jcl_strut
        a132 = (multi_dot([B(a109,a130),a109]) + (-1) * multi_dot([B(a122,a131),a122]))
        a133 = (self.Rd_rbl_upper_strut + (-1) * self.Rd_rbl_lower_strut + multi_dot([B(a111,a130),a109]) + (-1) * multi_dot([B(a117,a131),a122]))
        a134 = (self.R_rbl_upper_strut.T + (-1) * self.R_rbl_lower_strut.T + multi_dot([a130.T,a112]) + (-1) * multi_dot([a131.T,a118]))
        a135 = self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a136 = self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a137 = a91.T
        a138 = self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a139 = self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a140 = a122.T
        a141 = self.Pd_rbr_tie_rod
        a142 = self.Mbar_ground_jcr_tie_steering[:,0:1]
        a143 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a144 = self.P_rbr_tie_rod
        a145 = a141.T
        a146 = self.Pd_rbl_tie_rod
        a147 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a148 = self.P_rbl_tie_rod
        a149 = self.Mbar_ground_jcl_tie_steering[:,0:1]
        a150 = a146.T

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_chassis),a0]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcr_uca_chassis),a1])),
        (multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + (2) * multi_dot([a10,B(a3,a2).T,a11,a1])),
        (multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + (2) * multi_dot([a10,B(a3,a12).T,a11,a1])),
        (multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_upright),a0]) + (-1) * multi_dot([B(a13,self.ubar_rbr_upright_jcr_uca_upright),a13])),
        (multi_dot([B(a14,self.ubar_rbl_uca_jcl_uca_chassis),a14]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcl_uca_chassis),a1])),
        (multi_dot([a15.T,a17,a19,a1]) + multi_dot([a20,a9,B(a14,a15),a14]) + (2) * multi_dot([a21,B(a16,a15).T,a22,a1])),
        (multi_dot([a23.T,a17,a19,a1]) + multi_dot([a20,a9,B(a14,a23),a14]) + (2) * multi_dot([a21,B(a16,a23).T,a22,a1])),
        (multi_dot([B(a14,self.ubar_rbl_uca_jcl_uca_upright),a14]) + (-1) * multi_dot([B(a24,self.ubar_rbl_upright_jcl_uca_upright),a24])),
        (multi_dot([B(a25,self.ubar_rbr_lca_jcr_lca_chassis),a25]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcr_lca_chassis),a1])),
        (multi_dot([a26.T,a28,a30,a1]) + multi_dot([a31,a9,B(a25,a26),a25]) + (2) * multi_dot([a32,B(a27,a26).T,a33,a1])),
        (multi_dot([a34.T,a28,a30,a1]) + multi_dot([a31,a9,B(a25,a34),a25]) + (2) * multi_dot([a32,B(a27,a34).T,a33,a1])),
        (multi_dot([B(a25,self.ubar_rbr_lca_jcr_lca_upright),a25]) + (-1) * multi_dot([B(a13,self.ubar_rbr_upright_jcr_lca_upright),a13])),
        (multi_dot([B(a35,self.ubar_rbl_lca_jcl_lca_chassis),a35]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcl_lca_chassis),a1])),
        (multi_dot([a36.T,a38,a40,a1]) + multi_dot([a41,a9,B(a35,a36),a35]) + (2) * multi_dot([a42,B(a37,a36).T,a43,a1])),
        (multi_dot([a44.T,a38,a40,a1]) + multi_dot([a41,a9,B(a35,a44),a35]) + (2) * multi_dot([a42,B(a37,a44).T,a43,a1])),
        (multi_dot([B(a35,self.ubar_rbl_lca_jcl_lca_upright),a35]) + (-1) * multi_dot([B(a24,self.ubar_rbl_upright_jcl_lca_upright),a24])),
        (multi_dot([B(a13,self.ubar_rbr_upright_jcr_hub_bearing),a13]) + (-1) * multi_dot([B(a45,self.ubar_rbr_hub_jcr_hub_bearing),a45])),
        (multi_dot([a46.T,a48,a50,a45]) + multi_dot([a51,a53,B(a13,a46),a13]) + (2) * multi_dot([a54,B(a47,a46).T,a55,a45])),
        (multi_dot([a56.T,a48,a50,a45]) + multi_dot([a51,a53,B(a13,a56),a13]) + (2) * multi_dot([a54,B(a47,a56).T,a55,a45])),
        ((-1 * derivative(config.UF_mcr_wheel_lock, t, 0.1, 2)) * a57 + multi_dot([a58.T,a53,(a61 * B(a13,a62) + (-1 * a63) * B(a13,a64)),a13]) + multi_dot([(a61 * multi_dot([a62.T,a48]) + (-1 * a63) * multi_dot([a64.T,a48])),B(a45,a58),a45]) + (2) * multi_dot([(a61 * multi_dot([a54,B(a47,a62).T]) + (-1 * a63) * multi_dot([a54,B(a47,a64).T])),B(a52,a58),a45])),
        (multi_dot([B(a24,self.ubar_rbl_upright_jcl_hub_bearing),a24]) + (-1) * multi_dot([B(a65,self.ubar_rbl_hub_jcl_hub_bearing),a65])),
        (multi_dot([a66.T,a68,a70,a65]) + multi_dot([a71,a73,B(a24,a66),a24]) + (2) * multi_dot([a74,B(a67,a66).T,a75,a65])),
        (multi_dot([a76.T,a68,a70,a65]) + multi_dot([a71,a73,B(a24,a76),a24]) + (2) * multi_dot([a74,B(a67,a76).T,a75,a65])),
        ((-1 * derivative(config.UF_mcl_wheel_lock, t, 0.1, 2)) * a57 + multi_dot([a77.T,a73,(a79 * B(a24,a80) + (-1 * a81) * B(a24,a82)),a24]) + multi_dot([(a79 * multi_dot([a80.T,a68]) + (-1 * a81) * multi_dot([a82.T,a68])),B(a65,a77),a65]) + (2) * multi_dot([(a79 * multi_dot([a74,B(a67,a80).T]) + (-1 * a81) * multi_dot([a74,B(a67,a82).T])),B(a72,a77),a65])),
        (multi_dot([B(a83,self.ubar_rbr_upper_strut_jcr_strut_chassis),a83]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcr_strut_chassis),a1])),
        (multi_dot([a84.T,a9,B(a83,a85),a83]) + multi_dot([a85.T,a87,B(a1,a84),a1]) + (2) * multi_dot([a88,B(a86,a85).T,B(a8,a84),a1])),
        (multi_dot([a90,a87,a93,a91]) + multi_dot([a94,a96,a97,a83]) + (2) * multi_dot([a88,a98,a99,a91])),
        (multi_dot([a101,a87,a93,a91]) + multi_dot([a94,a96,a102,a83]) + (2) * multi_dot([a88,a103,a99,a91])),
        (multi_dot([a90,a87,a106]) + (2) * multi_dot([a88,a98,a107]) + multi_dot([a108,a97,a83])),
        (multi_dot([a101,a87,a106]) + (2) * multi_dot([a88,a103,a107]) + multi_dot([a108,a102,a83])),
        (multi_dot([B(a109,self.ubar_rbl_upper_strut_jcl_strut_chassis),a109]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcl_strut_chassis),a1])),
        (multi_dot([a110.T,a112,B(a1,a113),a1]) + multi_dot([a113.T,a9,B(a109,a110),a109]) + (2) * multi_dot([a114,B(a111,a110).T,B(a8,a113),a1])),
        (multi_dot([a116,a118,a120,a109]) + multi_dot([a121,a112,a123,a122]) + (2) * multi_dot([a114,a124,a125,a122])),
        (multi_dot([a116,a118,a127,a109]) + multi_dot([a128,a112,a123,a122]) + (2) * multi_dot([a114,a129,a125,a122])),
        (multi_dot([a121,a112,a132]) + (2) * multi_dot([a114,a124,a133]) + multi_dot([a134,a120,a109])),
        (multi_dot([a128,a112,a132]) + (2) * multi_dot([a114,a129,a133]) + multi_dot([a134,a127,a109])),
        (multi_dot([B(a91,self.ubar_rbr_lower_strut_jcr_strut_lca),a91]) + (-1) * multi_dot([B(a25,self.ubar_rbr_lca_jcr_strut_lca),a25])),
        (multi_dot([a135.T,a28,B(a91,a136),a91]) + multi_dot([a136.T,a96,B(a25,a135),a25]) + (2) * multi_dot([a137,B(a95,a136).T,B(a27,a135),a25])),
        (multi_dot([B(a122,self.ubar_rbl_lower_strut_jcl_strut_lca),a122]) + (-1) * multi_dot([B(a35,self.ubar_rbl_lca_jcl_strut_lca),a35])),
        (multi_dot([a138.T,a118,B(a35,a139),a35]) + multi_dot([a139.T,a38,B(a122,a138),a122]) + (2) * multi_dot([a140,B(a117,a138).T,B(a37,a139),a35])),
        (multi_dot([B(a141,self.ubar_rbr_tie_rod_jcr_tie_steering),a141]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcr_tie_steering),a1])),
        (multi_dot([a142.T,a9,B(a141,a143),a141]) + multi_dot([a143.T,A(a144).T,B(a1,a142),a1]) + (2) * multi_dot([a145,B(a144,a143).T,B(a8,a142),a1])),
        (multi_dot([B(a141,self.ubar_rbr_tie_rod_jcr_tie_upright),a141]) + (-1) * multi_dot([B(a13,self.ubar_rbr_upright_jcr_tie_upright),a13])),
        (multi_dot([B(a146,self.ubar_rbl_tie_rod_jcl_tie_steering),a146]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcl_tie_steering),a1])),
        (multi_dot([a147.T,A(a148).T,B(a1,a149),a1]) + multi_dot([a149.T,a9,B(a146,a147),a146]) + (2) * multi_dot([a150,B(a148,a147).T,B(a8,a149),a1])),
        (multi_dot([B(a146,self.ubar_rbl_tie_rod_jcl_tie_upright),a146]) + (-1) * multi_dot([B(a24,self.ubar_rbl_upright_jcl_tie_upright),a24])),
        ((-1 * derivative(config.UF_mcr_wheel_travel, t, 0.1, 2)) * a57 + (multi_dot([B(a45,self.ubar_rbr_hub_mcr_wheel_travel),a45]) + (-1) * multi_dot([B(a1,self.ubar_ground_mcr_wheel_travel),a1]))[2:3]),
        ((-1 * derivative(config.UF_mcl_wheel_travel, t, 0.1, 2)) * a57 + (multi_dot([B(a65,self.ubar_rbl_hub_mcl_wheel_travel),a65]) + (-1) * multi_dot([B(a1,self.ubar_ground_mcl_wheel_travel),a1]))[2:3]),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a10,a0]),
        (2) * multi_dot([a21,a14]),
        (2) * multi_dot([a32,a25]),
        (2) * multi_dot([a42,a35]),
        (2) * multi_dot([a54,a13]),
        (2) * multi_dot([a74,a24]),
        (2) * multi_dot([a88,a83]),
        (2) * multi_dot([a114,a109]),
        (2) * multi_dot([a137,a91]),
        (2) * multi_dot([a140,a122]),
        (2) * multi_dot([a145,a141]),
        (2) * multi_dot([a150,a146]),
        (2) * multi_dot([a45.T,a45]),
        (2) * multi_dot([a65.T,a65]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = I3
        j1 = self.P_rbr_uca
        j2 = Z1x3
        j3 = self.Mbar_ground_jcr_uca_chassis[:,2:3]
        j4 = j3.T
        j5 = self.P_ground
        j6 = A(j5).T
        j7 = self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        j8 = self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        j9 = (-1) * j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.P_rbr_upright
        j13 = self.P_rbl_uca
        j14 = self.Mbar_ground_jcl_uca_chassis[:,2:3]
        j15 = j14.T
        j16 = self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        j17 = self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        j18 = A(j13).T
        j19 = B(j5,j14)
        j20 = self.P_rbl_upright
        j21 = self.P_rbr_lca
        j22 = self.Mbar_ground_jcr_lca_chassis[:,2:3]
        j23 = j22.T
        j24 = self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        j25 = self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        j26 = A(j21).T
        j27 = B(j5,j22)
        j28 = self.P_rbl_lca
        j29 = self.Mbar_ground_jcl_lca_chassis[:,2:3]
        j30 = j29.T
        j31 = self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        j32 = self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        j33 = A(j28).T
        j34 = B(j5,j29)
        j35 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j36 = j35.T
        j37 = self.P_rbr_hub
        j38 = A(j37).T
        j39 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j40 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j41 = A(j12).T
        j42 = B(j37,j35)
        j43 = self.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        j44 = t
        j45 = config.UF_mcr_wheel_lock(j44)
        j46 = cos(j45)
        j47 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j48 = sin(j45)
        j49 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j50 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j51 = j50.T
        j52 = self.P_rbl_hub
        j53 = A(j52).T
        j54 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j55 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j56 = A(j20).T
        j57 = B(j52,j50)
        j58 = self.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        j59 = config.UF_mcl_wheel_lock(j44)
        j60 = cos(j59)
        j61 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j62 = sin(j59)
        j63 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j64 = self.P_rbr_upper_strut
        j65 = self.Mbar_ground_jcr_strut_chassis[:,0:1]
        j66 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j67 = A(j64).T
        j68 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j69 = j68.T
        j70 = self.P_rbr_lower_strut
        j71 = A(j70).T
        j72 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j73 = B(j64,j72)
        j74 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j75 = B(j64,j74)
        j76 = j72.T
        j77 = multi_dot([j76,j67])
        j78 = self.ubar_rbr_upper_strut_jcr_strut
        j79 = B(j64,j78)
        j80 = self.ubar_rbr_lower_strut_jcr_strut
        j81 = (self.R_rbr_upper_strut.T + (-1) * self.R_rbr_lower_strut.T + multi_dot([j78.T,j67]) + (-1) * multi_dot([j80.T,j71]))
        j82 = j74.T
        j83 = multi_dot([j82,j67])
        j84 = B(j70,j68)
        j85 = B(j70,j80)
        j86 = self.P_rbl_upper_strut
        j87 = self.Mbar_ground_jcl_strut_chassis[:,0:1]
        j88 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j89 = A(j86).T
        j90 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j91 = j90.T
        j92 = self.P_rbl_lower_strut
        j93 = A(j92).T
        j94 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j95 = B(j86,j94)
        j96 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j97 = B(j86,j96)
        j98 = j94.T
        j99 = multi_dot([j98,j89])
        j100 = self.ubar_rbl_upper_strut_jcl_strut
        j101 = B(j86,j100)
        j102 = self.ubar_rbl_lower_strut_jcl_strut
        j103 = (self.R_rbl_upper_strut.T + (-1) * self.R_rbl_lower_strut.T + multi_dot([j100.T,j89]) + (-1) * multi_dot([j102.T,j93]))
        j104 = j96.T
        j105 = multi_dot([j104,j89])
        j106 = B(j92,j90)
        j107 = B(j92,j102)
        j108 = self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j109 = self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j110 = self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j111 = self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j112 = self.P_rbr_tie_rod
        j113 = self.Mbar_ground_jcr_tie_steering[:,0:1]
        j114 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j115 = self.P_rbl_tie_rod
        j116 = self.Mbar_ground_jcl_tie_steering[:,0:1]
        j117 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        j118 = j0[2:3,0:3]
        j119 = (-1) * j118

        self.jac_eq_blocks = (j9,
        (-1) * B(j5,self.ubar_ground_jcr_uca_chassis),
        j0,
        B(j1,self.ubar_rbr_uca_jcr_uca_chassis),
        j2,
        multi_dot([j7.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j8.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j0,
        B(j1,self.ubar_rbr_uca_jcr_uca_upright),
        j9,
        (-1) * B(j12,self.ubar_rbr_upright_jcr_uca_upright),
        j9,
        (-1) * B(j5,self.ubar_ground_jcl_uca_chassis),
        j0,
        B(j13,self.ubar_rbl_uca_jcl_uca_chassis),
        j2,
        multi_dot([j16.T,j18,j19]),
        j2,
        multi_dot([j15,j6,B(j13,j16)]),
        j2,
        multi_dot([j17.T,j18,j19]),
        j2,
        multi_dot([j15,j6,B(j13,j17)]),
        j0,
        B(j13,self.ubar_rbl_uca_jcl_uca_upright),
        j9,
        (-1) * B(j20,self.ubar_rbl_upright_jcl_uca_upright),
        j9,
        (-1) * B(j5,self.ubar_ground_jcr_lca_chassis),
        j0,
        B(j21,self.ubar_rbr_lca_jcr_lca_chassis),
        j2,
        multi_dot([j24.T,j26,j27]),
        j2,
        multi_dot([j23,j6,B(j21,j24)]),
        j2,
        multi_dot([j25.T,j26,j27]),
        j2,
        multi_dot([j23,j6,B(j21,j25)]),
        j0,
        B(j21,self.ubar_rbr_lca_jcr_lca_upright),
        j9,
        (-1) * B(j12,self.ubar_rbr_upright_jcr_lca_upright),
        j9,
        (-1) * B(j5,self.ubar_ground_jcl_lca_chassis),
        j0,
        B(j28,self.ubar_rbl_lca_jcl_lca_chassis),
        j2,
        multi_dot([j31.T,j33,j34]),
        j2,
        multi_dot([j30,j6,B(j28,j31)]),
        j2,
        multi_dot([j32.T,j33,j34]),
        j2,
        multi_dot([j30,j6,B(j28,j32)]),
        j0,
        B(j28,self.ubar_rbl_lca_jcl_lca_upright),
        j9,
        (-1) * B(j20,self.ubar_rbl_upright_jcl_lca_upright),
        j0,
        B(j12,self.ubar_rbr_upright_jcr_hub_bearing),
        j9,
        (-1) * B(j37,self.ubar_rbr_hub_jcr_hub_bearing),
        j2,
        multi_dot([j36,j38,B(j12,j39)]),
        j2,
        multi_dot([j39.T,j41,j42]),
        j2,
        multi_dot([j36,j38,B(j12,j40)]),
        j2,
        multi_dot([j40.T,j41,j42]),
        j2,
        multi_dot([j43.T,j38,(j46 * B(j12,j47) + (-1 * j48) * B(j12,j49))]),
        j2,
        multi_dot([(j46 * multi_dot([j47.T,j41]) + (-1 * j48) * multi_dot([j49.T,j41])),B(j37,j43)]),
        j0,
        B(j20,self.ubar_rbl_upright_jcl_hub_bearing),
        j9,
        (-1) * B(j52,self.ubar_rbl_hub_jcl_hub_bearing),
        j2,
        multi_dot([j51,j53,B(j20,j54)]),
        j2,
        multi_dot([j54.T,j56,j57]),
        j2,
        multi_dot([j51,j53,B(j20,j55)]),
        j2,
        multi_dot([j55.T,j56,j57]),
        j2,
        multi_dot([j58.T,j53,(j60 * B(j20,j61) + (-1 * j62) * B(j20,j63))]),
        j2,
        multi_dot([(j60 * multi_dot([j61.T,j56]) + (-1 * j62) * multi_dot([j63.T,j56])),B(j52,j58)]),
        j9,
        (-1) * B(j5,self.ubar_ground_jcr_strut_chassis),
        j0,
        B(j64,self.ubar_rbr_upper_strut_jcr_strut_chassis),
        j2,
        multi_dot([j66.T,j67,B(j5,j65)]),
        j2,
        multi_dot([j65.T,j6,B(j64,j66)]),
        j2,
        multi_dot([j69,j71,j73]),
        j2,
        multi_dot([j76,j67,j84]),
        j2,
        multi_dot([j69,j71,j75]),
        j2,
        multi_dot([j82,j67,j84]),
        j77,
        (multi_dot([j76,j67,j79]) + multi_dot([j81,j73])),
        (-1) * j77,
        (-1) * multi_dot([j76,j67,j85]),
        j83,
        (multi_dot([j82,j67,j79]) + multi_dot([j81,j75])),
        (-1) * j83,
        (-1) * multi_dot([j82,j67,j85]),
        j9,
        (-1) * B(j5,self.ubar_ground_jcl_strut_chassis),
        j0,
        B(j86,self.ubar_rbl_upper_strut_jcl_strut_chassis),
        j2,
        multi_dot([j88.T,j89,B(j5,j87)]),
        j2,
        multi_dot([j87.T,j6,B(j86,j88)]),
        j2,
        multi_dot([j91,j93,j95]),
        j2,
        multi_dot([j98,j89,j106]),
        j2,
        multi_dot([j91,j93,j97]),
        j2,
        multi_dot([j104,j89,j106]),
        j99,
        (multi_dot([j98,j89,j101]) + multi_dot([j103,j95])),
        (-1) * j99,
        (-1) * multi_dot([j98,j89,j107]),
        j105,
        (multi_dot([j104,j89,j101]) + multi_dot([j103,j97])),
        (-1) * j105,
        (-1) * multi_dot([j104,j89,j107]),
        j9,
        (-1) * B(j21,self.ubar_rbr_lca_jcr_strut_lca),
        j0,
        B(j70,self.ubar_rbr_lower_strut_jcr_strut_lca),
        j2,
        multi_dot([j109.T,j71,B(j21,j108)]),
        j2,
        multi_dot([j108.T,j26,B(j70,j109)]),
        j9,
        (-1) * B(j28,self.ubar_rbl_lca_jcl_strut_lca),
        j0,
        B(j92,self.ubar_rbl_lower_strut_jcl_strut_lca),
        j2,
        multi_dot([j111.T,j93,B(j28,j110)]),
        j2,
        multi_dot([j110.T,j33,B(j92,j111)]),
        j9,
        (-1) * B(j5,self.ubar_ground_jcr_tie_steering),
        j0,
        B(j112,self.ubar_rbr_tie_rod_jcr_tie_steering),
        j2,
        multi_dot([j114.T,A(j112).T,B(j5,j113)]),
        j2,
        multi_dot([j113.T,j6,B(j112,j114)]),
        j9,
        (-1) * B(j12,self.ubar_rbr_upright_jcr_tie_upright),
        j0,
        B(j112,self.ubar_rbr_tie_rod_jcr_tie_upright),
        j9,
        (-1) * B(j5,self.ubar_ground_jcl_tie_steering),
        j0,
        B(j115,self.ubar_rbl_tie_rod_jcl_tie_steering),
        j2,
        multi_dot([j117.T,A(j115).T,B(j5,j116)]),
        j2,
        multi_dot([j116.T,j6,B(j115,j117)]),
        j9,
        (-1) * B(j20,self.ubar_rbl_upright_jcl_tie_upright),
        j0,
        B(j115,self.ubar_rbl_tie_rod_jcl_tie_upright),
        j119,
        (-1) * B(j5,self.ubar_ground_mcr_wheel_travel)[2:3,0:4],
        j118,
        B(j37,self.ubar_rbr_hub_mcr_wheel_travel)[2:3,0:4],
        j119,
        (-1) * B(j5,self.ubar_ground_mcl_wheel_travel)[2:3,0:4],
        j118,
        B(j52,self.ubar_rbl_hub_mcl_wheel_travel)[2:3,0:4],
        j0,
        Z3x4,
        Z4x3,
        I4,
        j2,
        (2) * j1.T,
        j2,
        (2) * j13.T,
        j2,
        (2) * j21.T,
        j2,
        (2) * j28.T,
        j2,
        (2) * j12.T,
        j2,
        (2) * j20.T,
        j2,
        (2) * j64.T,
        j2,
        (2) * j86.T,
        j2,
        (2) * j70.T,
        j2,
        (2) * j92.T,
        j2,
        (2) * j112.T,
        j2,
        (2) * j115.T,
        j2,
        (2) * j37.T,
        j2,
        (2) * j52.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = G(self.P_ground)
        m1 = G(self.P_rbr_uca)
        m2 = G(self.P_rbl_uca)
        m3 = G(self.P_rbr_lca)
        m4 = G(self.P_rbl_lca)
        m5 = G(self.P_rbr_upright)
        m6 = G(self.P_rbl_upright)
        m7 = G(self.P_rbr_upper_strut)
        m8 = G(self.P_rbl_upper_strut)
        m9 = G(self.P_rbr_lower_strut)
        m10 = G(self.P_rbl_lower_strut)
        m11 = G(self.P_rbr_tie_rod)
        m12 = G(self.P_rbl_tie_rod)
        m13 = G(self.P_rbr_hub)
        m14 = G(self.P_rbl_hub)

        self.mass_eq_blocks = (self.M_ground,
        (4) * multi_dot([m0.T,self.Jbar_ground,m0]),
        self.M_rbr_uca,
        (4) * multi_dot([m1.T,config.Jbar_rbr_uca,m1]),
        self.M_rbl_uca,
        (4) * multi_dot([m2.T,config.Jbar_rbl_uca,m2]),
        self.M_rbr_lca,
        (4) * multi_dot([m3.T,config.Jbar_rbr_lca,m3]),
        self.M_rbl_lca,
        (4) * multi_dot([m4.T,config.Jbar_rbl_lca,m4]),
        self.M_rbr_upright,
        (4) * multi_dot([m5.T,config.Jbar_rbr_upright,m5]),
        self.M_rbl_upright,
        (4) * multi_dot([m6.T,config.Jbar_rbl_upright,m6]),
        self.M_rbr_upper_strut,
        (4) * multi_dot([m7.T,config.Jbar_rbr_upper_strut,m7]),
        self.M_rbl_upper_strut,
        (4) * multi_dot([m8.T,config.Jbar_rbl_upper_strut,m8]),
        self.M_rbr_lower_strut,
        (4) * multi_dot([m9.T,config.Jbar_rbr_lower_strut,m9]),
        self.M_rbl_lower_strut,
        (4) * multi_dot([m10.T,config.Jbar_rbl_lower_strut,m10]),
        self.M_rbr_tie_rod,
        (4) * multi_dot([m11.T,config.Jbar_rbr_tie_rod,m11]),
        self.M_rbl_tie_rod,
        (4) * multi_dot([m12.T,config.Jbar_rbl_tie_rod,m12]),
        self.M_rbr_hub,
        (4) * multi_dot([m13.T,config.Jbar_rbr_hub,m13]),
        self.M_rbl_hub,
        (4) * multi_dot([m14.T,config.Jbar_rbl_hub,m14]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = Z3x1
        f1 = Z4x1
        f2 = G(self.Pd_rbr_uca)
        f3 = G(self.Pd_rbl_uca)
        f4 = G(self.Pd_rbr_lca)
        f5 = G(self.Pd_rbl_lca)
        f6 = G(self.Pd_rbr_upright)
        f7 = G(self.Pd_rbl_upright)
        f8 = self.R_rbr_upper_strut
        f9 = self.R_rbr_lower_strut
        f10 = self.ubar_rbr_upper_strut_far_strut
        f11 = self.P_rbr_upper_strut
        f12 = A(f11)
        f13 = self.ubar_rbr_lower_strut_far_strut
        f14 = self.P_rbr_lower_strut
        f15 = A(f14)
        f16 = (f8.T + (-1) * f9.T + multi_dot([f10.T,f12.T]) + (-1) * multi_dot([f13.T,f15.T]))
        f17 = multi_dot([f12,f10])
        f18 = multi_dot([f15,f13])
        f19 = (f8 + (-1) * f9 + f17 + (-1) * f18)
        f20 = ((multi_dot([f16,f19]))**(1.0/2.0))[0]
        f21 = 1.0/f20
        f22 = config.UF_far_strut_Fs((config.far_strut_FL + (-1 * f20)))
        f23 = self.Pd_rbr_upper_strut
        f24 = self.Pd_rbr_lower_strut
        f25 = config.UF_far_strut_Fd((-1 * 1.0/f20) * multi_dot([f16,(self.Rd_rbr_upper_strut + (-1) * self.Rd_rbr_lower_strut + multi_dot([B(f11,f10),f23]) + (-1) * multi_dot([B(f14,f13),f24]))]))
        f26 = (f21 * (f22 + f25)) * f19
        f27 = G(f23)
        f28 = (2 * f22)
        f29 = (2 * f25)
        f30 = self.R_rbl_upper_strut
        f31 = self.R_rbl_lower_strut
        f32 = self.ubar_rbl_upper_strut_fal_strut
        f33 = self.P_rbl_upper_strut
        f34 = A(f33)
        f35 = self.ubar_rbl_lower_strut_fal_strut
        f36 = self.P_rbl_lower_strut
        f37 = A(f36)
        f38 = (f30.T + (-1) * f31.T + multi_dot([f32.T,f34.T]) + (-1) * multi_dot([f35.T,f37.T]))
        f39 = multi_dot([f34,f32])
        f40 = multi_dot([f37,f35])
        f41 = (f30 + (-1) * f31 + f39 + (-1) * f40)
        f42 = ((multi_dot([f38,f41]))**(1.0/2.0))[0]
        f43 = 1.0/f42
        f44 = config.UF_fal_strut_Fs((config.fal_strut_FL + (-1 * f42)))
        f45 = self.Pd_rbl_upper_strut
        f46 = self.Pd_rbl_lower_strut
        f47 = config.UF_fal_strut_Fd((-1 * 1.0/f42) * multi_dot([f38,(self.Rd_rbl_upper_strut + (-1) * self.Rd_rbl_lower_strut + multi_dot([B(f33,f32),f45]) + (-1) * multi_dot([B(f36,f35),f46]))]))
        f48 = (f43 * (f44 + f47)) * f41
        f49 = G(f45)
        f50 = (2 * f44)
        f51 = (2 * f47)
        f52 = G(f24)
        f53 = G(f46)
        f54 = G(self.Pd_rbr_tie_rod)
        f55 = G(self.Pd_rbl_tie_rod)
        f56 = t
        f57 = config.UF_far_tire_F(f56)
        f58 = G(self.Pd_rbr_hub)
        f59 = self.P_rbr_hub
        f60 = config.UF_fal_tire_F(f56)
        f61 = G(self.Pd_rbl_hub)
        f62 = self.P_rbl_hub

        self.frc_eq_blocks = (f0,
        f1,
        self.F_rbr_uca_gravity,
        (8) * multi_dot([f2.T,config.Jbar_rbr_uca,f2,self.P_rbr_uca]),
        self.F_rbl_uca_gravity,
        (8) * multi_dot([f3.T,config.Jbar_rbl_uca,f3,self.P_rbl_uca]),
        self.F_rbr_lca_gravity,
        (8) * multi_dot([f4.T,config.Jbar_rbr_lca,f4,self.P_rbr_lca]),
        self.F_rbl_lca_gravity,
        (8) * multi_dot([f5.T,config.Jbar_rbl_lca,f5,self.P_rbl_lca]),
        self.F_rbr_upright_gravity,
        (8) * multi_dot([f6.T,config.Jbar_rbr_upright,f6,self.P_rbr_upright]),
        self.F_rbl_upright_gravity,
        (8) * multi_dot([f7.T,config.Jbar_rbl_upright,f7,self.P_rbl_upright]),
        (self.F_rbr_upper_strut_gravity + f26),
        ((8) * multi_dot([f27.T,config.Jbar_rbr_upper_strut,f27,f11]) + (f21 * ((-1 * f28) + (-1 * f29))) * multi_dot([E(f11).T,skew(f17).T,f19])),
        (self.F_rbl_upper_strut_gravity + f48),
        ((8) * multi_dot([f49.T,config.Jbar_rbl_upper_strut,f49,f33]) + (f43 * ((-1 * f50) + (-1 * f51))) * multi_dot([E(f33).T,skew(f39).T,f41])),
        (self.F_rbr_lower_strut_gravity + f0 + (-1) * f26),
        (f1 + (8) * multi_dot([f52.T,config.Jbar_rbr_lower_strut,f52,f14]) + (f21 * (f28 + f29)) * multi_dot([E(f14).T,skew(f18).T,f19])),
        (self.F_rbl_lower_strut_gravity + f0 + (-1) * f48),
        (f1 + (8) * multi_dot([f53.T,config.Jbar_rbl_lower_strut,f53,f36]) + (f43 * (f50 + f51)) * multi_dot([E(f36).T,skew(f40).T,f41])),
        self.F_rbr_tie_rod_gravity,
        (8) * multi_dot([f54.T,config.Jbar_rbr_tie_rod,f54,self.P_rbr_tie_rod]),
        self.F_rbl_tie_rod_gravity,
        (8) * multi_dot([f55.T,config.Jbar_rbl_tie_rod,f55,self.P_rbl_tie_rod]),
        (self.F_rbr_hub_gravity + f57),
        ((8) * multi_dot([f58.T,config.Jbar_rbr_hub,f58,f59]) + (2) * multi_dot([E(f59).T,(config.UF_far_tire_T(f56) + multi_dot([skew(multi_dot([A(f59),self.ubar_rbr_hub_far_tire])),f57]))])),
        (self.F_rbl_hub_gravity + f60),
        ((8) * multi_dot([f61.T,config.Jbar_rbl_hub,f61,f62]) + (2) * multi_dot([E(f62).T,(config.UF_fal_tire_T(f56) + multi_dot([skew(multi_dot([A(f62),self.ubar_rbl_hub_fal_tire])),f60]))])),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbr_uca_jcr_uca_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_uca_chassis).T,multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcr_uca_chassis[:,2:3]]),multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcr_uca_chassis[:,2:3]])]]),self.L_jcr_uca_chassis])
        self.F_rbr_uca_jcr_uca_chassis = Q_rbr_uca_jcr_uca_chassis[0:3]
        Te_rbr_uca_jcr_uca_chassis = Q_rbr_uca_jcr_uca_chassis[3:7]
        self.T_rbr_uca_jcr_uca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_uca_chassis])),self.F_rbr_uca_jcr_uca_chassis]) + (0.5) * multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_uca_chassis]))
        Q_rbr_uca_jcr_uca_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_uca_upright).T]]),self.L_jcr_uca_upright])
        self.F_rbr_uca_jcr_uca_upright = Q_rbr_uca_jcr_uca_upright[0:3]
        Te_rbr_uca_jcr_uca_upright = Q_rbr_uca_jcr_uca_upright[3:7]
        self.T_rbr_uca_jcr_uca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_uca_upright])),self.F_rbr_uca_jcr_uca_upright]) + (0.5) * multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_uca_upright]))
        Q_rbl_uca_jcl_uca_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_uca_chassis).T,multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcl_uca_chassis[:,2:3]]),multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcl_uca_chassis[:,2:3]])]]),self.L_jcl_uca_chassis])
        self.F_rbl_uca_jcl_uca_chassis = Q_rbl_uca_jcl_uca_chassis[0:3]
        Te_rbl_uca_jcl_uca_chassis = Q_rbl_uca_jcl_uca_chassis[3:7]
        self.T_rbl_uca_jcl_uca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_uca_chassis])),self.F_rbl_uca_jcl_uca_chassis]) + (0.5) * multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_uca_chassis]))
        Q_rbl_uca_jcl_uca_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_uca_upright).T]]),self.L_jcl_uca_upright])
        self.F_rbl_uca_jcl_uca_upright = Q_rbl_uca_jcl_uca_upright[0:3]
        Te_rbl_uca_jcl_uca_upright = Q_rbl_uca_jcl_uca_upright[3:7]
        self.T_rbl_uca_jcl_uca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_uca_upright])),self.F_rbl_uca_jcl_uca_upright]) + (0.5) * multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_uca_upright]))
        Q_rbr_lca_jcr_lca_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbr_lca,self.ubar_rbr_lca_jcr_lca_chassis).T,multi_dot([B(self.P_rbr_lca,self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcr_lca_chassis[:,2:3]]),multi_dot([B(self.P_rbr_lca,self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcr_lca_chassis[:,2:3]])]]),self.L_jcr_lca_chassis])
        self.F_rbr_lca_jcr_lca_chassis = Q_rbr_lca_jcr_lca_chassis[0:3]
        Te_rbr_lca_jcr_lca_chassis = Q_rbr_lca_jcr_lca_chassis[3:7]
        self.T_rbr_lca_jcr_lca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_lca),self.ubar_rbr_lca_jcr_lca_chassis])),self.F_rbr_lca_jcr_lca_chassis]) + (0.5) * multi_dot([E(self.P_rbr_lca),Te_rbr_lca_jcr_lca_chassis]))
        Q_rbr_lca_jcr_lca_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbr_lca,self.ubar_rbr_lca_jcr_lca_upright).T]]),self.L_jcr_lca_upright])
        self.F_rbr_lca_jcr_lca_upright = Q_rbr_lca_jcr_lca_upright[0:3]
        Te_rbr_lca_jcr_lca_upright = Q_rbr_lca_jcr_lca_upright[3:7]
        self.T_rbr_lca_jcr_lca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_lca),self.ubar_rbr_lca_jcr_lca_upright])),self.F_rbr_lca_jcr_lca_upright]) + (0.5) * multi_dot([E(self.P_rbr_lca),Te_rbr_lca_jcr_lca_upright]))
        Q_rbl_lca_jcl_lca_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbl_lca,self.ubar_rbl_lca_jcl_lca_chassis).T,multi_dot([B(self.P_rbl_lca,self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcl_lca_chassis[:,2:3]]),multi_dot([B(self.P_rbl_lca,self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcl_lca_chassis[:,2:3]])]]),self.L_jcl_lca_chassis])
        self.F_rbl_lca_jcl_lca_chassis = Q_rbl_lca_jcl_lca_chassis[0:3]
        Te_rbl_lca_jcl_lca_chassis = Q_rbl_lca_jcl_lca_chassis[3:7]
        self.T_rbl_lca_jcl_lca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_lca),self.ubar_rbl_lca_jcl_lca_chassis])),self.F_rbl_lca_jcl_lca_chassis]) + (0.5) * multi_dot([E(self.P_rbl_lca),Te_rbl_lca_jcl_lca_chassis]))
        Q_rbl_lca_jcl_lca_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbl_lca,self.ubar_rbl_lca_jcl_lca_upright).T]]),self.L_jcl_lca_upright])
        self.F_rbl_lca_jcl_lca_upright = Q_rbl_lca_jcl_lca_upright[0:3]
        Te_rbl_lca_jcl_lca_upright = Q_rbl_lca_jcl_lca_upright[3:7]
        self.T_rbl_lca_jcl_lca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_lca),self.ubar_rbl_lca_jcl_lca_upright])),self.F_rbl_lca_jcl_lca_upright]) + (0.5) * multi_dot([E(self.P_rbl_lca),Te_rbl_lca_jcl_lca_upright]))
        Q_rbr_upright_jcr_hub_bearing = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbr_upright,self.ubar_rbr_upright_jcr_hub_bearing).T,multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]])]]),self.L_jcr_hub_bearing])
        self.F_rbr_upright_jcr_hub_bearing = Q_rbr_upright_jcr_hub_bearing[0:3]
        Te_rbr_upright_jcr_hub_bearing = Q_rbr_upright_jcr_hub_bearing[3:7]
        self.T_rbr_upright_jcr_hub_bearing = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_upright),self.ubar_rbr_upright_jcr_hub_bearing])),self.F_rbr_upright_jcr_hub_bearing]) + (0.5) * multi_dot([E(self.P_rbr_upright),Te_rbr_upright_jcr_hub_bearing]))
        Q_rbr_upright_mcr_wheel_lock = (-1) * multi_dot([np.bmat([[Z1x3.T],[multi_dot([((-1 * sin(config.UF_mcr_wheel_lock(t))) * B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]).T + cos(config.UF_mcr_wheel_lock(t)) * B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]).T),A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]])]]),self.L_mcr_wheel_lock])
        self.F_rbr_upright_mcr_wheel_lock = Q_rbr_upright_mcr_wheel_lock[0:3]
        Te_rbr_upright_mcr_wheel_lock = Q_rbr_upright_mcr_wheel_lock[3:7]
        self.T_rbr_upright_mcr_wheel_lock = (0.5) * multi_dot([E(self.P_rbr_upright),Te_rbr_upright_mcr_wheel_lock])
        Q_rbl_upright_jcl_hub_bearing = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbl_upright,self.ubar_rbl_upright_jcl_hub_bearing).T,multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]])]]),self.L_jcl_hub_bearing])
        self.F_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[0:3]
        Te_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[3:7]
        self.T_rbl_upright_jcl_hub_bearing = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_upright),self.ubar_rbl_upright_jcl_hub_bearing])),self.F_rbl_upright_jcl_hub_bearing]) + (0.5) * multi_dot([E(self.P_rbl_upright),Te_rbl_upright_jcl_hub_bearing]))
        Q_rbl_upright_mcl_wheel_lock = (-1) * multi_dot([np.bmat([[Z1x3.T],[multi_dot([((-1 * sin(config.UF_mcl_wheel_lock(t))) * B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]).T + cos(config.UF_mcl_wheel_lock(t)) * B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]).T),A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]])]]),self.L_mcl_wheel_lock])
        self.F_rbl_upright_mcl_wheel_lock = Q_rbl_upright_mcl_wheel_lock[0:3]
        Te_rbl_upright_mcl_wheel_lock = Q_rbl_upright_mcl_wheel_lock[3:7]
        self.T_rbl_upright_mcl_wheel_lock = (0.5) * multi_dot([E(self.P_rbl_upright),Te_rbl_upright_mcl_wheel_lock])
        Q_rbr_upper_strut_jcr_strut_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut_chassis).T,multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcr_strut_chassis[:,0:1]])]]),self.L_jcr_strut_chassis])
        self.F_rbr_upper_strut_jcr_strut_chassis = Q_rbr_upper_strut_jcr_strut_chassis[0:3]
        Te_rbr_upper_strut_jcr_strut_chassis = Q_rbr_upper_strut_jcr_strut_chassis[3:7]
        self.T_rbr_upper_strut_jcr_strut_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut_chassis])),self.F_rbr_upper_strut_jcr_strut_chassis]) + (0.5) * multi_dot([E(self.P_rbr_upper_strut),Te_rbr_upper_strut_jcr_strut_chassis]))
        Q_rbr_upper_strut_jcr_strut = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]]),multi_dot([A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]])],[multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]).T,A(self.P_rbr_lower_strut),self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]),multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]).T,A(self.P_rbr_lower_strut),self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]),(multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]).T,((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut]) + self.R_rbr_upper_strut)]) + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut).T,A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]])),(multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]).T,((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut]) + self.R_rbr_upper_strut)]) + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut).T,A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]]))]]),self.L_jcr_strut])
        self.F_rbr_upper_strut_jcr_strut = Q_rbr_upper_strut_jcr_strut[0:3]
        Te_rbr_upper_strut_jcr_strut = Q_rbr_upper_strut_jcr_strut[3:7]
        self.T_rbr_upper_strut_jcr_strut = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut])),self.F_rbr_upper_strut_jcr_strut]) + (0.5) * multi_dot([E(self.P_rbr_upper_strut),Te_rbr_upper_strut_jcr_strut]))
        self.F_rbr_upper_strut_far_strut = (1.0/((multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)]))**(1.0/2.0))[0] * (config.UF_far_strut_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.Rd_rbr_lower_strut + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_far_strut),self.Pd_rbr_upper_strut]) + (-1) * multi_dot([B(self.P_rbr_lower_strut,self.ubar_rbr_lower_strut_far_strut),self.Pd_rbr_lower_strut]) + self.Rd_rbr_upper_strut)])) + config.UF_far_strut_Fs((config.far_strut_FL + (-1 * ((multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)
        self.T_rbr_upper_strut_far_strut = Z3x1
        Q_rbl_upper_strut_jcl_strut_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut_chassis).T,multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcl_strut_chassis[:,0:1]])]]),self.L_jcl_strut_chassis])
        self.F_rbl_upper_strut_jcl_strut_chassis = Q_rbl_upper_strut_jcl_strut_chassis[0:3]
        Te_rbl_upper_strut_jcl_strut_chassis = Q_rbl_upper_strut_jcl_strut_chassis[3:7]
        self.T_rbl_upper_strut_jcl_strut_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut_chassis])),self.F_rbl_upper_strut_jcl_strut_chassis]) + (0.5) * multi_dot([E(self.P_rbl_upper_strut),Te_rbl_upper_strut_jcl_strut_chassis]))
        Q_rbl_upper_strut_jcl_strut = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]]),multi_dot([A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]])],[multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]).T,A(self.P_rbl_lower_strut),self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]),multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]).T,A(self.P_rbl_lower_strut),self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]),(multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]).T,((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut]) + self.R_rbl_upper_strut)]) + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut).T,A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]])),(multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]).T,((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut]) + self.R_rbl_upper_strut)]) + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut).T,A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]]))]]),self.L_jcl_strut])
        self.F_rbl_upper_strut_jcl_strut = Q_rbl_upper_strut_jcl_strut[0:3]
        Te_rbl_upper_strut_jcl_strut = Q_rbl_upper_strut_jcl_strut[3:7]
        self.T_rbl_upper_strut_jcl_strut = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut])),self.F_rbl_upper_strut_jcl_strut]) + (0.5) * multi_dot([E(self.P_rbl_upper_strut),Te_rbl_upper_strut_jcl_strut]))
        self.F_rbl_upper_strut_fal_strut = (1.0/((multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)]))**(1.0/2.0))[0] * (config.UF_fal_strut_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.Rd_rbl_lower_strut + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_fal_strut),self.Pd_rbl_upper_strut]) + (-1) * multi_dot([B(self.P_rbl_lower_strut,self.ubar_rbl_lower_strut_fal_strut),self.Pd_rbl_lower_strut]) + self.Rd_rbl_upper_strut)])) + config.UF_fal_strut_Fs((config.fal_strut_FL + (-1 * ((multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)
        self.T_rbl_upper_strut_fal_strut = Z3x1
        Q_rbr_lower_strut_jcr_strut_lca = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbr_lower_strut,self.ubar_rbr_lower_strut_jcr_strut_lca).T,multi_dot([B(self.P_rbr_lower_strut,self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]).T,A(self.P_rbr_lca),self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]])]]),self.L_jcr_strut_lca])
        self.F_rbr_lower_strut_jcr_strut_lca = Q_rbr_lower_strut_jcr_strut_lca[0:3]
        Te_rbr_lower_strut_jcr_strut_lca = Q_rbr_lower_strut_jcr_strut_lca[3:7]
        self.T_rbr_lower_strut_jcr_strut_lca = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut_lca])),self.F_rbr_lower_strut_jcr_strut_lca]) + (0.5) * multi_dot([E(self.P_rbr_lower_strut),Te_rbr_lower_strut_jcr_strut_lca]))
        Q_rbl_lower_strut_jcl_strut_lca = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbl_lower_strut,self.ubar_rbl_lower_strut_jcl_strut_lca).T,multi_dot([B(self.P_rbl_lower_strut,self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]).T,A(self.P_rbl_lca),self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]])]]),self.L_jcl_strut_lca])
        self.F_rbl_lower_strut_jcl_strut_lca = Q_rbl_lower_strut_jcl_strut_lca[0:3]
        Te_rbl_lower_strut_jcl_strut_lca = Q_rbl_lower_strut_jcl_strut_lca[3:7]
        self.T_rbl_lower_strut_jcl_strut_lca = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut_lca])),self.F_rbl_lower_strut_jcl_strut_lca]) + (0.5) * multi_dot([E(self.P_rbl_lower_strut),Te_rbl_lower_strut_jcl_strut_lca]))
        Q_rbr_tie_rod_jcr_tie_steering = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbr_tie_rod,self.ubar_rbr_tie_rod_jcr_tie_steering).T,multi_dot([B(self.P_rbr_tie_rod,self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcr_tie_steering[:,0:1]])]]),self.L_jcr_tie_steering])
        self.F_rbr_tie_rod_jcr_tie_steering = Q_rbr_tie_rod_jcr_tie_steering[0:3]
        Te_rbr_tie_rod_jcr_tie_steering = Q_rbr_tie_rod_jcr_tie_steering[3:7]
        self.T_rbr_tie_rod_jcr_tie_steering = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_tie_rod),self.ubar_rbr_tie_rod_jcr_tie_steering])),self.F_rbr_tie_rod_jcr_tie_steering]) + (0.5) * multi_dot([E(self.P_rbr_tie_rod),Te_rbr_tie_rod_jcr_tie_steering]))
        Q_rbr_tie_rod_jcr_tie_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbr_tie_rod,self.ubar_rbr_tie_rod_jcr_tie_upright).T]]),self.L_jcr_tie_upright])
        self.F_rbr_tie_rod_jcr_tie_upright = Q_rbr_tie_rod_jcr_tie_upright[0:3]
        Te_rbr_tie_rod_jcr_tie_upright = Q_rbr_tie_rod_jcr_tie_upright[3:7]
        self.T_rbr_tie_rod_jcr_tie_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_tie_rod),self.ubar_rbr_tie_rod_jcr_tie_upright])),self.F_rbr_tie_rod_jcr_tie_upright]) + (0.5) * multi_dot([E(self.P_rbr_tie_rod),Te_rbr_tie_rod_jcr_tie_upright]))
        Q_rbl_tie_rod_jcl_tie_steering = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbl_tie_rod,self.ubar_rbl_tie_rod_jcl_tie_steering).T,multi_dot([B(self.P_rbl_tie_rod,self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcl_tie_steering[:,0:1]])]]),self.L_jcl_tie_steering])
        self.F_rbl_tie_rod_jcl_tie_steering = Q_rbl_tie_rod_jcl_tie_steering[0:3]
        Te_rbl_tie_rod_jcl_tie_steering = Q_rbl_tie_rod_jcl_tie_steering[3:7]
        self.T_rbl_tie_rod_jcl_tie_steering = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_tie_rod),self.ubar_rbl_tie_rod_jcl_tie_steering])),self.F_rbl_tie_rod_jcl_tie_steering]) + (0.5) * multi_dot([E(self.P_rbl_tie_rod),Te_rbl_tie_rod_jcl_tie_steering]))
        Q_rbl_tie_rod_jcl_tie_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbl_tie_rod,self.ubar_rbl_tie_rod_jcl_tie_upright).T]]),self.L_jcl_tie_upright])
        self.F_rbl_tie_rod_jcl_tie_upright = Q_rbl_tie_rod_jcl_tie_upright[0:3]
        Te_rbl_tie_rod_jcl_tie_upright = Q_rbl_tie_rod_jcl_tie_upright[3:7]
        self.T_rbl_tie_rod_jcl_tie_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_tie_rod),self.ubar_rbl_tie_rod_jcl_tie_upright])),self.F_rbl_tie_rod_jcl_tie_upright]) + (0.5) * multi_dot([E(self.P_rbl_tie_rod),Te_rbl_tie_rod_jcl_tie_upright]))
        Q_rbr_hub_mcr_wheel_travel = (-1) * multi_dot([np.bmat([[I3[2:3,0:3].T],[B(self.P_rbr_hub,self.ubar_rbr_hub_mcr_wheel_travel)[2:3,0:4].T]]),self.L_mcr_wheel_travel])
        self.F_rbr_hub_mcr_wheel_travel = Q_rbr_hub_mcr_wheel_travel[0:3]
        Te_rbr_hub_mcr_wheel_travel = Q_rbr_hub_mcr_wheel_travel[3:7]
        self.T_rbr_hub_mcr_wheel_travel = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_hub),self.ubar_rbr_hub_mcr_wheel_travel])),self.F_rbr_hub_mcr_wheel_travel]) + (0.5) * multi_dot([E(self.P_rbr_hub),Te_rbr_hub_mcr_wheel_travel]))
        Q_rbl_hub_mcl_wheel_travel = (-1) * multi_dot([np.bmat([[I3[2:3,0:3].T],[B(self.P_rbl_hub,self.ubar_rbl_hub_mcl_wheel_travel)[2:3,0:4].T]]),self.L_mcl_wheel_travel])
        self.F_rbl_hub_mcl_wheel_travel = Q_rbl_hub_mcl_wheel_travel[0:3]
        Te_rbl_hub_mcl_wheel_travel = Q_rbl_hub_mcl_wheel_travel[3:7]
        self.T_rbl_hub_mcl_wheel_travel = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_hub),self.ubar_rbl_hub_mcl_wheel_travel])),self.F_rbl_hub_mcl_wheel_travel]) + (0.5) * multi_dot([E(self.P_rbl_hub),Te_rbl_hub_mcl_wheel_travel]))

        self.reactions = {'F_rbr_uca_jcr_uca_chassis' : self.F_rbr_uca_jcr_uca_chassis,
                        'T_rbr_uca_jcr_uca_chassis' : self.T_rbr_uca_jcr_uca_chassis,
                        'F_rbr_uca_jcr_uca_upright' : self.F_rbr_uca_jcr_uca_upright,
                        'T_rbr_uca_jcr_uca_upright' : self.T_rbr_uca_jcr_uca_upright,
                        'F_rbl_uca_jcl_uca_chassis' : self.F_rbl_uca_jcl_uca_chassis,
                        'T_rbl_uca_jcl_uca_chassis' : self.T_rbl_uca_jcl_uca_chassis,
                        'F_rbl_uca_jcl_uca_upright' : self.F_rbl_uca_jcl_uca_upright,
                        'T_rbl_uca_jcl_uca_upright' : self.T_rbl_uca_jcl_uca_upright,
                        'F_rbr_lca_jcr_lca_chassis' : self.F_rbr_lca_jcr_lca_chassis,
                        'T_rbr_lca_jcr_lca_chassis' : self.T_rbr_lca_jcr_lca_chassis,
                        'F_rbr_lca_jcr_lca_upright' : self.F_rbr_lca_jcr_lca_upright,
                        'T_rbr_lca_jcr_lca_upright' : self.T_rbr_lca_jcr_lca_upright,
                        'F_rbl_lca_jcl_lca_chassis' : self.F_rbl_lca_jcl_lca_chassis,
                        'T_rbl_lca_jcl_lca_chassis' : self.T_rbl_lca_jcl_lca_chassis,
                        'F_rbl_lca_jcl_lca_upright' : self.F_rbl_lca_jcl_lca_upright,
                        'T_rbl_lca_jcl_lca_upright' : self.T_rbl_lca_jcl_lca_upright,
                        'F_rbr_upright_jcr_hub_bearing' : self.F_rbr_upright_jcr_hub_bearing,
                        'T_rbr_upright_jcr_hub_bearing' : self.T_rbr_upright_jcr_hub_bearing,
                        'F_rbr_upright_mcr_wheel_lock' : self.F_rbr_upright_mcr_wheel_lock,
                        'T_rbr_upright_mcr_wheel_lock' : self.T_rbr_upright_mcr_wheel_lock,
                        'F_rbl_upright_jcl_hub_bearing' : self.F_rbl_upright_jcl_hub_bearing,
                        'T_rbl_upright_jcl_hub_bearing' : self.T_rbl_upright_jcl_hub_bearing,
                        'F_rbl_upright_mcl_wheel_lock' : self.F_rbl_upright_mcl_wheel_lock,
                        'T_rbl_upright_mcl_wheel_lock' : self.T_rbl_upright_mcl_wheel_lock,
                        'F_rbr_upper_strut_jcr_strut_chassis' : self.F_rbr_upper_strut_jcr_strut_chassis,
                        'T_rbr_upper_strut_jcr_strut_chassis' : self.T_rbr_upper_strut_jcr_strut_chassis,
                        'F_rbr_upper_strut_jcr_strut' : self.F_rbr_upper_strut_jcr_strut,
                        'T_rbr_upper_strut_jcr_strut' : self.T_rbr_upper_strut_jcr_strut,
                        'F_rbr_upper_strut_far_strut' : self.F_rbr_upper_strut_far_strut,
                        'T_rbr_upper_strut_far_strut' : self.T_rbr_upper_strut_far_strut,
                        'F_rbl_upper_strut_jcl_strut_chassis' : self.F_rbl_upper_strut_jcl_strut_chassis,
                        'T_rbl_upper_strut_jcl_strut_chassis' : self.T_rbl_upper_strut_jcl_strut_chassis,
                        'F_rbl_upper_strut_jcl_strut' : self.F_rbl_upper_strut_jcl_strut,
                        'T_rbl_upper_strut_jcl_strut' : self.T_rbl_upper_strut_jcl_strut,
                        'F_rbl_upper_strut_fal_strut' : self.F_rbl_upper_strut_fal_strut,
                        'T_rbl_upper_strut_fal_strut' : self.T_rbl_upper_strut_fal_strut,
                        'F_rbr_lower_strut_jcr_strut_lca' : self.F_rbr_lower_strut_jcr_strut_lca,
                        'T_rbr_lower_strut_jcr_strut_lca' : self.T_rbr_lower_strut_jcr_strut_lca,
                        'F_rbl_lower_strut_jcl_strut_lca' : self.F_rbl_lower_strut_jcl_strut_lca,
                        'T_rbl_lower_strut_jcl_strut_lca' : self.T_rbl_lower_strut_jcl_strut_lca,
                        'F_rbr_tie_rod_jcr_tie_steering' : self.F_rbr_tie_rod_jcr_tie_steering,
                        'T_rbr_tie_rod_jcr_tie_steering' : self.T_rbr_tie_rod_jcr_tie_steering,
                        'F_rbr_tie_rod_jcr_tie_upright' : self.F_rbr_tie_rod_jcr_tie_upright,
                        'T_rbr_tie_rod_jcr_tie_upright' : self.T_rbr_tie_rod_jcr_tie_upright,
                        'F_rbl_tie_rod_jcl_tie_steering' : self.F_rbl_tie_rod_jcl_tie_steering,
                        'T_rbl_tie_rod_jcl_tie_steering' : self.T_rbl_tie_rod_jcl_tie_steering,
                        'F_rbl_tie_rod_jcl_tie_upright' : self.F_rbl_tie_rod_jcl_tie_upright,
                        'T_rbl_tie_rod_jcl_tie_upright' : self.T_rbl_tie_rod_jcl_tie_upright,
                        'F_rbr_hub_mcr_wheel_travel' : self.F_rbr_hub_mcr_wheel_travel,
                        'T_rbr_hub_mcr_wheel_travel' : self.T_rbr_hub_mcr_wheel_travel,
                        'F_rbl_hub_mcl_wheel_travel' : self.F_rbl_hub_mcl_wheel_travel,
                        'T_rbl_hub_mcl_wheel_travel' : self.T_rbl_hub_mcl_wheel_travel}

