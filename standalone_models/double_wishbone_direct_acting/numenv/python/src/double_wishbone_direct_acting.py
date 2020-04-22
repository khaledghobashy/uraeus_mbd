
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
        self.nc = 101
        self.nrows = 60
        self.ncols = 2*15
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_rbr_uca_jcr_uca_chassis', 'T_rbr_uca_jcr_uca_chassis', 'F_rbr_uca_jcr_uca_upright', 'T_rbr_uca_jcr_uca_upright', 'F_rbl_uca_jcl_uca_chassis', 'T_rbl_uca_jcl_uca_chassis', 'F_rbl_uca_jcl_uca_upright', 'T_rbl_uca_jcl_uca_upright', 'F_rbr_lca_jcr_lca_chassis', 'T_rbr_lca_jcr_lca_chassis', 'F_rbr_lca_jcr_lca_upright', 'T_rbr_lca_jcr_lca_upright', 'F_rbl_lca_jcl_lca_chassis', 'T_rbl_lca_jcl_lca_chassis', 'F_rbl_lca_jcl_lca_upright', 'T_rbl_lca_jcl_lca_upright', 'F_rbr_upright_jcr_hub_bearing', 'T_rbr_upright_jcr_hub_bearing', 'F_rbl_upright_jcl_hub_bearing', 'T_rbl_upright_jcl_hub_bearing', 'F_rbr_upper_strut_jcr_strut_chassis', 'T_rbr_upper_strut_jcr_strut_chassis', 'F_rbr_upper_strut_jcr_strut', 'T_rbr_upper_strut_jcr_strut', 'F_rbr_upper_strut_far_strut', 'T_rbr_upper_strut_far_strut', 'F_rbl_upper_strut_jcl_strut_chassis', 'T_rbl_upper_strut_jcl_strut_chassis', 'F_rbl_upper_strut_jcl_strut', 'T_rbl_upper_strut_jcl_strut', 'F_rbl_upper_strut_fal_strut', 'T_rbl_upper_strut_fal_strut', 'F_rbr_lower_strut_jcr_strut_lca', 'T_rbr_lower_strut_jcr_strut_lca', 'F_rbl_lower_strut_jcl_strut_lca', 'T_rbl_lower_strut_jcl_strut_lca', 'F_rbr_tie_rod_jcr_tie_steering', 'T_rbr_tie_rod_jcr_tie_steering', 'F_rbr_tie_rod_jcr_tie_upright', 'T_rbr_tie_rod_jcr_tie_upright', 'F_rbl_tie_rod_jcl_tie_steering', 'T_rbl_tie_rod_jcl_tie_steering', 'F_rbl_tie_rod_jcl_tie_upright', 'T_rbl_tie_rod_jcl_tie_upright']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.ground*2, self.ground*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.ground*2, self.ground*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.ground*2, self.ground*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.ground*2, self.ground*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.ground*2, self.ground*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.ground*2, self.ground*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.ground*2, self.ground*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.ground*2, self.ground*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.ground*2, self.ground*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.ground*2, self.ground*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.ground*2, self.ground*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.ground*2, self.ground*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.ground*2, self.ground*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.ground*2, self.ground*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.ground*2, self.ground*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.ground*2, self.ground*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.ground*2, self.ground*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.ground*2, self.ground*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.ground*2, self.ground*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.ground*2, self.ground*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_hub*2, self.rbl_hub*2+1], dtype=np.intc)

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
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_hub_bearing)])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([A(config.P_rbl_hub).T,triad(config.ax1_jcl_hub_bearing)])
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_hub_bearing]) + (-1) * multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([A(config.P_rbl_hub).T,config.pt1_jcl_hub_bearing]) + (-1) * multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
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
        self.ubar_rbr_hub_far_tire = (multi_dot([A(config.P_rbr_hub).T,config.pt1_far_tire]) + (-1) * multi_dot([A(config.P_rbr_hub).T,config.R_rbr_hub]))
        self.ubar_ground_far_tire = (multi_dot([A(self.P_ground).T,config.pt1_far_tire]) + (-1) * multi_dot([A(self.P_ground).T,self.R_ground]))
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
        self.L_jcl_hub_bearing = Lambda[37:42]
        self.L_jcr_strut_chassis = Lambda[42:46]
        self.L_jcr_strut = Lambda[46:50]
        self.L_jcl_strut_chassis = Lambda[50:54]
        self.L_jcl_strut = Lambda[54:58]
        self.L_jcr_strut_lca = Lambda[58:62]
        self.L_jcl_strut_lca = Lambda[62:66]
        self.L_jcr_tie_steering = Lambda[66:70]
        self.L_jcr_tie_upright = Lambda[70:73]
        self.L_jcl_tie_steering = Lambda[73:77]
        self.L_jcl_tie_upright = Lambda[77:80]

    
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
        x32 = self.P_rbr_hub
        x33 = A(x32)
        x34 = x12.T
        x35 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x36 = self.P_rbl_hub
        x37 = A(x36)
        x38 = x21.T
        x39 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x40 = self.R_rbr_upper_strut
        x41 = self.P_rbr_upper_strut
        x42 = A(x41)
        x43 = x42.T
        x44 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x45 = self.P_rbr_lower_strut
        x46 = A(x45)
        x47 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x48 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x49 = self.R_rbr_lower_strut
        x50 = (x40 + (-1) * x49 + multi_dot([x42,self.ubar_rbr_upper_strut_jcr_strut]) + (-1) * multi_dot([x46,self.ubar_rbr_lower_strut_jcr_strut]))
        x51 = self.R_rbl_upper_strut
        x52 = self.P_rbl_upper_strut
        x53 = A(x52)
        x54 = x53.T
        x55 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x56 = self.P_rbl_lower_strut
        x57 = A(x56)
        x58 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x59 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x60 = self.R_rbl_lower_strut
        x61 = (x51 + (-1) * x60 + multi_dot([x53,self.ubar_rbl_upper_strut_jcl_strut]) + (-1) * multi_dot([x57,self.ubar_rbl_lower_strut_jcl_strut]))
        x62 = self.R_rbr_tie_rod
        x63 = self.P_rbr_tie_rod
        x64 = A(x63)
        x65 = self.R_rbl_tie_rod
        x66 = self.P_rbl_tie_rod
        x67 = A(x66)
        x68 = (-1) * I1

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
        (x9 + (-1) * self.R_rbr_hub + multi_dot([x12,self.ubar_rbr_upright_jcr_hub_bearing]) + (-1) * multi_dot([x33,self.ubar_rbr_hub_jcr_hub_bearing])),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x34,x33,x35]),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x34,x33,x35]),
        (x18 + (-1) * self.R_rbl_hub + multi_dot([x21,self.ubar_rbl_upright_jcl_hub_bearing]) + (-1) * multi_dot([x37,self.ubar_rbl_hub_jcl_hub_bearing])),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x38,x37,x39]),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x38,x37,x39]),
        (x40 + x2 + multi_dot([x42,self.ubar_rbr_upper_strut_jcr_strut_chassis]) + (-1) * multi_dot([x6,self.ubar_ground_jcr_strut_chassis])),
        multi_dot([self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x43,x6,self.Mbar_ground_jcr_strut_chassis[:,0:1]]),
        multi_dot([x44,x43,x46,x47]),
        multi_dot([x48,x43,x46,x47]),
        multi_dot([x44,x43,x50]),
        multi_dot([x48,x43,x50]),
        (x51 + x2 + multi_dot([x53,self.ubar_rbl_upper_strut_jcl_strut_chassis]) + (-1) * multi_dot([x6,self.ubar_ground_jcl_strut_chassis])),
        multi_dot([self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1].T,x54,x6,self.Mbar_ground_jcl_strut_chassis[:,0:1]]),
        multi_dot([x55,x54,x57,x58]),
        multi_dot([x59,x54,x57,x58]),
        multi_dot([x55,x54,x61]),
        multi_dot([x59,x54,x61]),
        (x49 + (-1) * x22 + multi_dot([x46,self.ubar_rbr_lower_strut_jcr_strut_lca]) + (-1) * multi_dot([x24,self.ubar_rbr_lca_jcr_strut_lca])),
        multi_dot([self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1].T,x46.T,x24,self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]]),
        (x60 + (-1) * x27 + multi_dot([x57,self.ubar_rbl_lower_strut_jcl_strut_lca]) + (-1) * multi_dot([x29,self.ubar_rbl_lca_jcl_strut_lca])),
        multi_dot([self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1].T,x57.T,x29,self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]]),
        (x62 + x2 + multi_dot([x64,self.ubar_rbr_tie_rod_jcr_tie_steering]) + (-1) * multi_dot([x6,self.ubar_ground_jcr_tie_steering])),
        multi_dot([self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x64.T,x6,self.Mbar_ground_jcr_tie_steering[:,0:1]]),
        (x62 + x10 + multi_dot([x64,self.ubar_rbr_tie_rod_jcr_tie_upright]) + (-1) * multi_dot([x12,self.ubar_rbr_upright_jcr_tie_upright])),
        (x65 + x2 + multi_dot([x67,self.ubar_rbl_tie_rod_jcl_tie_steering]) + (-1) * multi_dot([x6,self.ubar_ground_jcl_tie_steering])),
        multi_dot([self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x67.T,x6,self.Mbar_ground_jcl_tie_steering[:,0:1]]),
        (x65 + x19 + multi_dot([x67,self.ubar_rbl_tie_rod_jcl_tie_upright]) + (-1) * multi_dot([x21,self.ubar_rbl_upright_jcl_tie_upright])),
        x1,
        (x5 + (-1) * self.Pg_ground),
        (x68 + multi_dot([x3.T,x3])),
        (x68 + multi_dot([x14.T,x14])),
        (x68 + multi_dot([x23.T,x23])),
        (x68 + multi_dot([x28.T,x28])),
        (x68 + multi_dot([x11.T,x11])),
        (x68 + multi_dot([x20.T,x20])),
        (x68 + multi_dot([x41.T,x41])),
        (x68 + multi_dot([x52.T,x52])),
        (x68 + multi_dot([x45.T,x45])),
        (x68 + multi_dot([x56.T,x56])),
        (x68 + multi_dot([x63.T,x63])),
        (x68 + multi_dot([x66.T,x66])),
        (x68 + multi_dot([x32.T,x32])),
        (x68 + multi_dot([x36.T,x36])),)

    
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
        a57 = self.Pd_rbl_hub
        a58 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a59 = self.P_rbl_upright
        a60 = A(a59).T
        a61 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a62 = B(a57,a61)
        a63 = a61.T
        a64 = self.P_rbl_hub
        a65 = A(a64).T
        a66 = a24.T
        a67 = B(a64,a61)
        a68 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a69 = self.Pd_rbr_upper_strut
        a70 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a71 = self.P_rbr_upper_strut
        a72 = A(a71).T
        a73 = self.Mbar_ground_jcr_strut_chassis[:,0:1]
        a74 = a69.T
        a75 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a76 = a75.T
        a77 = self.Pd_rbr_lower_strut
        a78 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a79 = B(a77,a78)
        a80 = a78.T
        a81 = self.P_rbr_lower_strut
        a82 = A(a81).T
        a83 = B(a69,a75)
        a84 = B(a71,a75).T
        a85 = B(a81,a78)
        a86 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a87 = a86.T
        a88 = B(a69,a86)
        a89 = B(a71,a86).T
        a90 = self.ubar_rbr_upper_strut_jcr_strut
        a91 = self.ubar_rbr_lower_strut_jcr_strut
        a92 = (multi_dot([B(a69,a90),a69]) + (-1) * multi_dot([B(a77,a91),a77]))
        a93 = (self.Rd_rbr_upper_strut + (-1) * self.Rd_rbr_lower_strut + multi_dot([B(a71,a90),a69]) + (-1) * multi_dot([B(a81,a91),a77]))
        a94 = (self.R_rbr_upper_strut.T + (-1) * self.R_rbr_lower_strut.T + multi_dot([a90.T,a72]) + (-1) * multi_dot([a91.T,a82]))
        a95 = self.Pd_rbl_upper_strut
        a96 = self.Mbar_ground_jcl_strut_chassis[:,0:1]
        a97 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a98 = self.P_rbl_upper_strut
        a99 = A(a98).T
        a100 = a95.T
        a101 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a102 = a101.T
        a103 = self.Pd_rbl_lower_strut
        a104 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a105 = B(a103,a104)
        a106 = a104.T
        a107 = self.P_rbl_lower_strut
        a108 = A(a107).T
        a109 = B(a95,a101)
        a110 = B(a98,a101).T
        a111 = B(a107,a104)
        a112 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a113 = a112.T
        a114 = B(a95,a112)
        a115 = B(a98,a112).T
        a116 = self.ubar_rbl_upper_strut_jcl_strut
        a117 = self.ubar_rbl_lower_strut_jcl_strut
        a118 = (multi_dot([B(a95,a116),a95]) + (-1) * multi_dot([B(a103,a117),a103]))
        a119 = (self.Rd_rbl_upper_strut + (-1) * self.Rd_rbl_lower_strut + multi_dot([B(a98,a116),a95]) + (-1) * multi_dot([B(a107,a117),a103]))
        a120 = (self.R_rbl_upper_strut.T + (-1) * self.R_rbl_lower_strut.T + multi_dot([a116.T,a99]) + (-1) * multi_dot([a117.T,a108]))
        a121 = self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a122 = self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a123 = a77.T
        a124 = self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a125 = self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a126 = a103.T
        a127 = self.Pd_rbr_tie_rod
        a128 = self.Mbar_ground_jcr_tie_steering[:,0:1]
        a129 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a130 = self.P_rbr_tie_rod
        a131 = a127.T
        a132 = self.Pd_rbl_tie_rod
        a133 = self.Mbar_ground_jcl_tie_steering[:,0:1]
        a134 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a135 = self.P_rbl_tie_rod
        a136 = a132.T

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
        (multi_dot([B(a24,self.ubar_rbl_upright_jcl_hub_bearing),a24]) + (-1) * multi_dot([B(a57,self.ubar_rbl_hub_jcl_hub_bearing),a57])),
        (multi_dot([a58.T,a60,a62,a57]) + multi_dot([a63,a65,B(a24,a58),a24]) + (2) * multi_dot([a66,B(a59,a58).T,a67,a57])),
        (multi_dot([a68.T,a60,a62,a57]) + multi_dot([a63,a65,B(a24,a68),a24]) + (2) * multi_dot([a66,B(a59,a68).T,a67,a57])),
        (multi_dot([B(a69,self.ubar_rbr_upper_strut_jcr_strut_chassis),a69]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcr_strut_chassis),a1])),
        (multi_dot([a70.T,a72,B(a1,a73),a1]) + multi_dot([a73.T,a9,B(a69,a70),a69]) + (2) * multi_dot([a74,B(a71,a70).T,B(a8,a73),a1])),
        (multi_dot([a76,a72,a79,a77]) + multi_dot([a80,a82,a83,a69]) + (2) * multi_dot([a74,a84,a85,a77])),
        (multi_dot([a87,a72,a79,a77]) + multi_dot([a80,a82,a88,a69]) + (2) * multi_dot([a74,a89,a85,a77])),
        (multi_dot([a76,a72,a92]) + (2) * multi_dot([a74,a84,a93]) + multi_dot([a94,a83,a69])),
        (multi_dot([a87,a72,a92]) + (2) * multi_dot([a74,a89,a93]) + multi_dot([a94,a88,a69])),
        (multi_dot([B(a95,self.ubar_rbl_upper_strut_jcl_strut_chassis),a95]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcl_strut_chassis),a1])),
        (multi_dot([a96.T,a9,B(a95,a97),a95]) + multi_dot([a97.T,a99,B(a1,a96),a1]) + (2) * multi_dot([a100,B(a98,a97).T,B(a8,a96),a1])),
        (multi_dot([a102,a99,a105,a103]) + multi_dot([a106,a108,a109,a95]) + (2) * multi_dot([a100,a110,a111,a103])),
        (multi_dot([a113,a99,a105,a103]) + multi_dot([a106,a108,a114,a95]) + (2) * multi_dot([a100,a115,a111,a103])),
        (multi_dot([a102,a99,a118]) + (2) * multi_dot([a100,a110,a119]) + multi_dot([a120,a109,a95])),
        (multi_dot([a113,a99,a118]) + (2) * multi_dot([a100,a115,a119]) + multi_dot([a120,a114,a95])),
        (multi_dot([B(a77,self.ubar_rbr_lower_strut_jcr_strut_lca),a77]) + (-1) * multi_dot([B(a25,self.ubar_rbr_lca_jcr_strut_lca),a25])),
        (multi_dot([a121.T,a82,B(a25,a122),a25]) + multi_dot([a122.T,a28,B(a77,a121),a77]) + (2) * multi_dot([a123,B(a81,a121).T,B(a27,a122),a25])),
        (multi_dot([B(a103,self.ubar_rbl_lower_strut_jcl_strut_lca),a103]) + (-1) * multi_dot([B(a35,self.ubar_rbl_lca_jcl_strut_lca),a35])),
        (multi_dot([a124.T,a108,B(a35,a125),a35]) + multi_dot([a125.T,a38,B(a103,a124),a103]) + (2) * multi_dot([a126,B(a107,a124).T,B(a37,a125),a35])),
        (multi_dot([B(a127,self.ubar_rbr_tie_rod_jcr_tie_steering),a127]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcr_tie_steering),a1])),
        (multi_dot([a128.T,a9,B(a127,a129),a127]) + multi_dot([a129.T,A(a130).T,B(a1,a128),a1]) + (2) * multi_dot([a131,B(a130,a129).T,B(a8,a128),a1])),
        (multi_dot([B(a127,self.ubar_rbr_tie_rod_jcr_tie_upright),a127]) + (-1) * multi_dot([B(a13,self.ubar_rbr_upright_jcr_tie_upright),a13])),
        (multi_dot([B(a132,self.ubar_rbl_tie_rod_jcl_tie_steering),a132]) + (-1) * multi_dot([B(a1,self.ubar_ground_jcl_tie_steering),a1])),
        (multi_dot([a133.T,a9,B(a132,a134),a132]) + multi_dot([a134.T,A(a135).T,B(a1,a133),a1]) + (2) * multi_dot([a136,B(a135,a134).T,B(a8,a133),a1])),
        (multi_dot([B(a132,self.ubar_rbl_tie_rod_jcl_tie_upright),a132]) + (-1) * multi_dot([B(a24,self.ubar_rbl_upright_jcl_tie_upright),a24])),
        Z3x1,
        Z4x1,
        (2) * multi_dot([a10,a0]),
        (2) * multi_dot([a21,a14]),
        (2) * multi_dot([a32,a25]),
        (2) * multi_dot([a42,a35]),
        (2) * multi_dot([a54,a13]),
        (2) * multi_dot([a66,a24]),
        (2) * multi_dot([a74,a69]),
        (2) * multi_dot([a100,a95]),
        (2) * multi_dot([a123,a77]),
        (2) * multi_dot([a126,a103]),
        (2) * multi_dot([a131,a127]),
        (2) * multi_dot([a136,a132]),
        (2) * multi_dot([a45.T,a45]),
        (2) * multi_dot([a57.T,a57]),)

    
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
        j43 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j44 = j43.T
        j45 = self.P_rbl_hub
        j46 = A(j45).T
        j47 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j48 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j49 = A(j20).T
        j50 = B(j45,j43)
        j51 = self.P_rbr_upper_strut
        j52 = self.Mbar_ground_jcr_strut_chassis[:,0:1]
        j53 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j54 = A(j51).T
        j55 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j56 = j55.T
        j57 = self.P_rbr_lower_strut
        j58 = A(j57).T
        j59 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j60 = B(j51,j59)
        j61 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j62 = B(j51,j61)
        j63 = j59.T
        j64 = multi_dot([j63,j54])
        j65 = self.ubar_rbr_upper_strut_jcr_strut
        j66 = B(j51,j65)
        j67 = self.ubar_rbr_lower_strut_jcr_strut
        j68 = (self.R_rbr_upper_strut.T + (-1) * self.R_rbr_lower_strut.T + multi_dot([j65.T,j54]) + (-1) * multi_dot([j67.T,j58]))
        j69 = j61.T
        j70 = multi_dot([j69,j54])
        j71 = B(j57,j55)
        j72 = B(j57,j67)
        j73 = self.P_rbl_upper_strut
        j74 = self.Mbar_ground_jcl_strut_chassis[:,0:1]
        j75 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j76 = A(j73).T
        j77 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j78 = j77.T
        j79 = self.P_rbl_lower_strut
        j80 = A(j79).T
        j81 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j82 = B(j73,j81)
        j83 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j84 = B(j73,j83)
        j85 = j81.T
        j86 = multi_dot([j85,j76])
        j87 = self.ubar_rbl_upper_strut_jcl_strut
        j88 = B(j73,j87)
        j89 = self.ubar_rbl_lower_strut_jcl_strut
        j90 = (self.R_rbl_upper_strut.T + (-1) * self.R_rbl_lower_strut.T + multi_dot([j87.T,j76]) + (-1) * multi_dot([j89.T,j80]))
        j91 = j83.T
        j92 = multi_dot([j91,j76])
        j93 = B(j79,j77)
        j94 = B(j79,j89)
        j95 = self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j96 = self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j97 = self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j98 = self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j99 = self.P_rbr_tie_rod
        j100 = self.Mbar_ground_jcr_tie_steering[:,0:1]
        j101 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j102 = self.P_rbl_tie_rod
        j103 = self.Mbar_ground_jcl_tie_steering[:,0:1]
        j104 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]

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
        j0,
        B(j20,self.ubar_rbl_upright_jcl_hub_bearing),
        j9,
        (-1) * B(j45,self.ubar_rbl_hub_jcl_hub_bearing),
        j2,
        multi_dot([j44,j46,B(j20,j47)]),
        j2,
        multi_dot([j47.T,j49,j50]),
        j2,
        multi_dot([j44,j46,B(j20,j48)]),
        j2,
        multi_dot([j48.T,j49,j50]),
        j9,
        (-1) * B(j5,self.ubar_ground_jcr_strut_chassis),
        j0,
        B(j51,self.ubar_rbr_upper_strut_jcr_strut_chassis),
        j2,
        multi_dot([j53.T,j54,B(j5,j52)]),
        j2,
        multi_dot([j52.T,j6,B(j51,j53)]),
        j2,
        multi_dot([j56,j58,j60]),
        j2,
        multi_dot([j63,j54,j71]),
        j2,
        multi_dot([j56,j58,j62]),
        j2,
        multi_dot([j69,j54,j71]),
        j64,
        (multi_dot([j63,j54,j66]) + multi_dot([j68,j60])),
        (-1) * j64,
        (-1) * multi_dot([j63,j54,j72]),
        j70,
        (multi_dot([j69,j54,j66]) + multi_dot([j68,j62])),
        (-1) * j70,
        (-1) * multi_dot([j69,j54,j72]),
        j9,
        (-1) * B(j5,self.ubar_ground_jcl_strut_chassis),
        j0,
        B(j73,self.ubar_rbl_upper_strut_jcl_strut_chassis),
        j2,
        multi_dot([j75.T,j76,B(j5,j74)]),
        j2,
        multi_dot([j74.T,j6,B(j73,j75)]),
        j2,
        multi_dot([j78,j80,j82]),
        j2,
        multi_dot([j85,j76,j93]),
        j2,
        multi_dot([j78,j80,j84]),
        j2,
        multi_dot([j91,j76,j93]),
        j86,
        (multi_dot([j85,j76,j88]) + multi_dot([j90,j82])),
        (-1) * j86,
        (-1) * multi_dot([j85,j76,j94]),
        j92,
        (multi_dot([j91,j76,j88]) + multi_dot([j90,j84])),
        (-1) * j92,
        (-1) * multi_dot([j91,j76,j94]),
        j9,
        (-1) * B(j21,self.ubar_rbr_lca_jcr_strut_lca),
        j0,
        B(j57,self.ubar_rbr_lower_strut_jcr_strut_lca),
        j2,
        multi_dot([j96.T,j58,B(j21,j95)]),
        j2,
        multi_dot([j95.T,j26,B(j57,j96)]),
        j9,
        (-1) * B(j28,self.ubar_rbl_lca_jcl_strut_lca),
        j0,
        B(j79,self.ubar_rbl_lower_strut_jcl_strut_lca),
        j2,
        multi_dot([j98.T,j80,B(j28,j97)]),
        j2,
        multi_dot([j97.T,j33,B(j79,j98)]),
        j9,
        (-1) * B(j5,self.ubar_ground_jcr_tie_steering),
        j0,
        B(j99,self.ubar_rbr_tie_rod_jcr_tie_steering),
        j2,
        multi_dot([j101.T,A(j99).T,B(j5,j100)]),
        j2,
        multi_dot([j100.T,j6,B(j99,j101)]),
        j9,
        (-1) * B(j12,self.ubar_rbr_upright_jcr_tie_upright),
        j0,
        B(j99,self.ubar_rbr_tie_rod_jcr_tie_upright),
        j9,
        (-1) * B(j5,self.ubar_ground_jcl_tie_steering),
        j0,
        B(j102,self.ubar_rbl_tie_rod_jcl_tie_steering),
        j2,
        multi_dot([j104.T,A(j102).T,B(j5,j103)]),
        j2,
        multi_dot([j103.T,j6,B(j102,j104)]),
        j9,
        (-1) * B(j20,self.ubar_rbl_upright_jcl_tie_upright),
        j0,
        B(j102,self.ubar_rbl_tie_rod_jcl_tie_upright),
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
        (2) * j51.T,
        j2,
        (2) * j73.T,
        j2,
        (2) * j57.T,
        j2,
        (2) * j79.T,
        j2,
        (2) * j99.T,
        j2,
        (2) * j102.T,
        j2,
        (2) * j37.T,
        j2,
        (2) * j45.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_ground)
        m2 = G(self.P_rbr_uca)
        m3 = G(self.P_rbl_uca)
        m4 = G(self.P_rbr_lca)
        m5 = G(self.P_rbl_lca)
        m6 = G(self.P_rbr_upright)
        m7 = G(self.P_rbl_upright)
        m8 = G(self.P_rbr_upper_strut)
        m9 = G(self.P_rbl_upper_strut)
        m10 = G(self.P_rbr_lower_strut)
        m11 = G(self.P_rbl_lower_strut)
        m12 = G(self.P_rbr_tie_rod)
        m13 = G(self.P_rbl_tie_rod)
        m14 = G(self.P_rbr_hub)
        m15 = G(self.P_rbl_hub)

        self.mass_eq_blocks = (self.m_ground * m0,
        (4) * multi_dot([m1.T,self.Jbar_ground,m1]),
        config.m_rbr_uca * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbr_uca,m2]),
        config.m_rbl_uca * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbl_uca,m3]),
        config.m_rbr_lca * m0,
        (4) * multi_dot([m4.T,config.Jbar_rbr_lca,m4]),
        config.m_rbl_lca * m0,
        (4) * multi_dot([m5.T,config.Jbar_rbl_lca,m5]),
        config.m_rbr_upright * m0,
        (4) * multi_dot([m6.T,config.Jbar_rbr_upright,m6]),
        config.m_rbl_upright * m0,
        (4) * multi_dot([m7.T,config.Jbar_rbl_upright,m7]),
        config.m_rbr_upper_strut * m0,
        (4) * multi_dot([m8.T,config.Jbar_rbr_upper_strut,m8]),
        config.m_rbl_upper_strut * m0,
        (4) * multi_dot([m9.T,config.Jbar_rbl_upper_strut,m9]),
        config.m_rbr_lower_strut * m0,
        (4) * multi_dot([m10.T,config.Jbar_rbr_lower_strut,m10]),
        config.m_rbl_lower_strut * m0,
        (4) * multi_dot([m11.T,config.Jbar_rbl_lower_strut,m11]),
        config.m_rbr_tie_rod * m0,
        (4) * multi_dot([m12.T,config.Jbar_rbr_tie_rod,m12]),
        config.m_rbl_tie_rod * m0,
        (4) * multi_dot([m13.T,config.Jbar_rbl_tie_rod,m13]),
        config.m_rbr_hub * m0,
        (4) * multi_dot([m14.T,config.Jbar_rbr_hub,m14]),
        config.m_rbl_hub * m0,
        (4) * multi_dot([m15.T,config.Jbar_rbl_hub,m15]),)

    
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
        Q_rbl_upright_jcl_hub_bearing = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbl_upright,self.ubar_rbl_upright_jcl_hub_bearing).T,multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]])]]),self.L_jcl_hub_bearing])
        self.F_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[0:3]
        Te_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[3:7]
        self.T_rbl_upright_jcl_hub_bearing = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_upright),self.ubar_rbl_upright_jcl_hub_bearing])),self.F_rbl_upright_jcl_hub_bearing]) + (0.5) * multi_dot([E(self.P_rbl_upright),Te_rbl_upright_jcl_hub_bearing]))
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
                        'F_rbl_upright_jcl_hub_bearing' : self.F_rbl_upright_jcl_hub_bearing,
                        'T_rbl_upright_jcl_hub_bearing' : self.T_rbl_upright_jcl_hub_bearing,
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
                        'T_rbl_tie_rod_jcl_tie_upright' : self.T_rbl_tie_rod_jcl_tie_upright}

