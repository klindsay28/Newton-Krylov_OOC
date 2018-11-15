#!/usr/bin/env python
"""comp_precond_jacobian_fcn_state_prod for test problem for Newton-Krylov solver"""

import argparse
import configparser

import numpy as np
from scipy.linalg import solve_banded, null_space
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from comp_fcn_common import t_del, nz, dz_r, dz_mid_r, mixing_coeff_log_avg
from model import ModelState, ModelStaticVars, tracer_module_names

def _parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="comp_precond_jacobian_fcn_state_prod.sh test problem for Newton-Krylov solver")
    parser.add_argument('in_fname', help='name of file with input')
    parser.add_argument('res_fname', help='name of file for output')
    parser.add_argument('--cfg_fname', help='name of configuration file',
                        default='newton_krylov.cfg')
    return parser.parse_args()

def main(args):
    """comp_precond_jacobian_fcn_state_prod for test problem for Newton-Krylov solver"""

    config = configparser.ConfigParser()
    config.read(args.cfg_fname)
    ModelStaticVars(config['modelinfo'], args.cfg_fname)

    ms_in = ModelState(args.in_fname)
    ms_res = ms_in.copy()

    for tracer_module_name in tracer_module_names():
        if tracer_module_name == 'iage':
            comp_precond_iage(ms_in, ms_res)
        if tracer_module_name == 'phosphorus':
            comp_precond_phosphorus(ms_in, ms_res)

    ms_res.dump(args.res_fname)

def comp_precond_iage(ms_in, ms_res):
    """apply preconditioner for iage"""

    iage_in = ms_in.get_tracer_vals('iage')
    rhs = (1.0 / t_del) * iage_in

    mca = mixing_coeff_log_avg()

    l_and_u = (1, 1)
    matrix_diagonals = np.zeros((3, nz))
    matrix_diagonals[0, 1:] = mca[1:-1] * dz_mid_r * dz_r[:-1]
    matrix_diagonals[1, :-1] -= mca[1:-1] * dz_mid_r * dz_r[:-1]
    matrix_diagonals[1, 1:] -= mca[1:-1] * dz_mid_r * dz_r[1:]
    matrix_diagonals[2, :-1] = mca[1:-1] * dz_mid_r * dz_r[1:]
    matrix_diagonals[1, 0] = -24.0
    matrix_diagonals[0, 1] = 0

    res = solve_banded(l_and_u, matrix_diagonals, rhs)

    ms_res.set_tracer_vals('iage', res - iage_in)

def comp_precond_phosphorus(ms_in, ms_res):
    """apply preconditioner for phosphorus"""

    po4_s = ms_in.get_tracer_vals('po4_s')
    dop_s = ms_in.get_tracer_vals('dop_s')
    pop_s = ms_in.get_tracer_vals('pop_s')
    rhs = (1.0 / t_del) * np.concatenate((po4_s, dop_s, pop_s))

    mca = mixing_coeff_log_avg()

    matrix = diags([diag_0_phosphorus(mca),
                    diag_p_1_phosphorus(mca), diag_p_nz_phosphorus(), diag_p_2nz_phosphorus(),
                    diag_m_1_phosphorus(mca), diag_m_nz_phosphorus(), diag_m_2nz_phosphorus()],
                   [0, 1, nz, 2*nz, -1, -nz, -2*nz], format='csr')

    res = spsolve(matrix, rhs)

    matrix_ns = null_space(matrix.todense())
    res -= res.mean()/matrix_ns[:, 0].mean() * matrix_ns[:, 0]

    ms_res.set_tracer_vals('po4_s', res[0:nz] - po4_s)
    ms_res.set_tracer_vals('dop_s', res[nz:2*nz] - dop_s)
    ms_res.set_tracer_vals('pop_s', res[2*nz:3*nz] - pop_s)

    # ms_res.set_tracer_vals('po4_s', ns[0:nz])
    # ms_res.set_tracer_vals('dop_s', ns[nz:2*nz])
    # ms_res.set_tracer_vals('pop_s', ns[2*nz:3*nz])

def diag_0_phosphorus(mca):
    """return main diagonal of Jacobian preconditioner for phosphorus shadow tracers"""
    diag_0_single_tracer = np.zeros(nz)
    diag_0_single_tracer[:-1] -= mca[1:-1] * dz_mid_r * dz_r[:-1]
    diag_0_single_tracer[1:] -= mca[1:-1] * dz_mid_r * dz_r[1:]
    diag_0_po4_s = diag_0_single_tracer.copy()
    diag_0_po4_s[0] -= 1.0 # po4_s restoring in top layer
    diag_0_dop_s = diag_0_single_tracer.copy()
    diag_0_dop_s -= 0.01 # dop_s remin
    diag_0_pop_s = diag_0_single_tracer.copy()
    diag_0_pop_s -= 0.01 # pop_s remin
    diag_0_pop_s[:-1] -= 1.0 * dz_r[:-1] # pop_s sinking loss to layer below
    return np.concatenate((diag_0_po4_s, diag_0_dop_s, diag_0_pop_s))

def diag_p_1_phosphorus(mca):
    """return +1 upper diagonal of Jacobian preconditioner for phosphorus shadow tracers"""
    diag_p_1_single_tracer = mca[1:-1] * dz_mid_r * dz_r[:-1]
    diag_p_1_po4_s = diag_p_1_single_tracer.copy()
    zero = np.zeros(1)
    diag_p_1_dop_s = diag_p_1_single_tracer.copy()
    diag_p_1_pop_s = diag_p_1_single_tracer.copy()
    return np.concatenate((diag_p_1_po4_s, zero, diag_p_1_dop_s, zero, diag_p_1_pop_s))

def diag_p_nz_phosphorus():
    """return +nz upper diagonal of Jacobian preconditioner for phosphorus shadow tracers"""
    diag_p_1_dop_po4 = 0.01 * np.ones(nz) # dop_s remin
    diag_p_1_pop_dop = np.zeros(nz)
    return np.concatenate((diag_p_1_dop_po4, diag_p_1_pop_dop))

def diag_p_2nz_phosphorus():
    """return +2nz upper diagonal of Jacobian preconditioner for phosphorus shadow tracers"""
    return 0.01 * np.ones(nz) # pop_s remin

def diag_m_1_phosphorus(mca):
    """return +1 upper diagonal of Jacobian preconditioner for phosphorus shadow tracers"""
    diag_m_1_single_tracer = mca[1:-1] * dz_mid_r * dz_r[1:]
    diag_m_1_po4_s = diag_m_1_single_tracer.copy()
    zero = np.zeros(1)
    diag_m_1_dop_s = diag_m_1_single_tracer.copy()
    diag_m_1_pop_s = diag_m_1_single_tracer.copy()
    diag_m_1_pop_s += 1.0 * dz_r[1:] # pop_s sinking gain from layer above
    return np.concatenate((diag_m_1_po4_s, zero, diag_m_1_dop_s, zero, diag_m_1_pop_s))

def diag_m_nz_phosphorus():
    """return -nz lower diagonal of Jacobian preconditioner for phosphorus shadow tracers"""
    diag_p_1_po4_dop = np.zeros(nz)
    diag_p_1_po4_dop[0] = 0.67 # po4_s restoring conservation balance
    diag_p_1_dop_pop = np.zeros(nz)
    return np.concatenate((diag_p_1_po4_dop, diag_p_1_dop_pop))

def diag_m_2nz_phosphorus():
    """return -2nz lower diagonal of Jacobian preconditioner for phosphorus shadow tracers"""
    diag_p_1_po4_pop = np.zeros(nz)
    diag_p_1_po4_pop[0] = 0.33 # po4_s restoring conservation balance
    return diag_p_1_po4_pop

if __name__ == '__main__':
    main(_parse_args())
