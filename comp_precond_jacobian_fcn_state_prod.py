#!/usr/bin/env python
"""comp_precond_jacobian_fcn_state_prod for test problem for Newton-Krylov solver"""

import argparse
import configparser

import numpy as np
from scipy.linalg import solve_banded

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
        # if tracer_module_name == 'phosphorus':
        #     comp_precond_phosphorus(ms_in, ms_res)

    ms_res.dump(args.res_fname)

def comp_precond_iage(ms_in, ms_res):
    """apply preconditioner for iage"""
    rhs = (1.0 / t_del) * ms_in.get_tracer_vals('iage')

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

    ms_res.set_tracer_vals('iage', res)

# def comp_precond_phosphorus(ms_in, ms_res):
#     """apply preconditioner for phosphorus"""
#     # d(d[po4_s,dop_s,pop_s]_dt)/d[po4_s,dop_s,pop_s] x res = in
#
#     # ignore mixing_tend
#     # dpo4_dt = -po4_uptake + pop_remin
#     # dpop_dt = po4_uptake - pop_remin + sinking_tend(pop)
#     dpo4_dt = ms_in.get_tracer_vals('po4')
#     dpop_dt = ms_in.get_tracer_vals('pop')
#     # dpo4_dt + dpop_dt = sinking_tend_pop = -d/dz pop_flux = -d/dz (5.0 m/d * pop)
#     pop = -0.2 * np.cumsum(dz * (dpo4_dt + dpop_dt))
#     po4_uptake = 0.1 * pop - dpo4_dt
#     po4_lim = np.where((z_mid < 100.0), np.exp((1.0 / 25.0) * z_mid) * po4_uptake, 0.0)
#     po4 = po4_lim * (po4_iterate + 0.5)**2 / 0.5
#     ms_res.set_tracer_vals('po4', po4)
#     ms_res.set_tracer_vals('pop', pop)

if __name__ == '__main__':
    main(_parse_args())
