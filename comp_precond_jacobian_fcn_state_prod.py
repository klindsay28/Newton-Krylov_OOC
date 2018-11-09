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
        if tracer_module_name == 'IAGE':
            comp_precond_IAGE(ms_in, ms_res)
        # if tracer_module_name == 'PO4_POP':
        #     comp_precond_PO4_POP(ms_in, ms_res)

    ms_res.dump(args.res_fname)

def comp_precond_IAGE(ms_in, ms_res):
    """apply preconditioner for IAGE"""
    rhs = (1.0 / t_del) * ms_in.get_tracer_vals('IAGE')

    mca = mixing_coeff_log_avg()

    l_and_u = (1, 1)
    ab = np.zeros((3, nz))
    ab[0, 1:] = mca[1:-1] * dz_mid_r * dz_r[:-1]
    ab[1, :-1] -= mca[1:-1] * dz_mid_r * dz_r[:-1]
    ab[1, 1:] -= mca[1:-1] * dz_mid_r * dz_r[1:]
    ab[2, :-1] = mca[1:-1] * dz_mid_r * dz_r[1:]
    ab[1, 0] = -24.0
    ab[0, 1] = 0

    res = solve_banded(l_and_u, ab, rhs)

    ms_res.set_tracer_vals('IAGE', res)

# def comp_precond_PO4_POP(ms_in, ms_res):
#     """apply preconditioner for PO4_POP"""
#     # d(d[PO4,POP]_dt)/d[PO4,POP] x res = in
#
#     # ignore mixing_tend
#     # dPO4_dt = -PO4_uptake + POP_remin
#     # dPOP_dt = PO4_uptake - POP_remin + sinking_tend(POP)
#     dPO4_dt = ms_in.get_tracer_vals('PO4')
#     dPOP_dt = ms_in.get_tracer_vals('POP')
#     # dPO4_dt + dPOP_dt = sinking_tend_POP = -d/dz POP_flux = -d/dz (5.0 m/d * POP)
#     POP = -0.2 * np.cumsum(dz * (dPO4_dt + dPOP_dt))
#     PO4_uptake = 0.1 * POP - dPO4_dt
#     PO4_lim = np.where((z_mid < 100.0), np.exp((1.0 / 25.0) * z_mid) * PO4_uptake, 0.0)
#     PO4 = PO4_lim * (PO4_iterate + 0.5)**2 / 0.5
#     ms_res.set_tracer_vals('PO4', PO4)
#     ms_res.set_tracer_vals('POP', POP)

if __name__ == '__main__':
    main(_parse_args())
