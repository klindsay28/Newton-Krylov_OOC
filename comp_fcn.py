#!/usr/bin/env python
"""test problem for Newton-Krylov solver"""

import argparse
import configparser

import netCDF4 as nc
import numpy as np
from scipy.integrate import solve_ivp

from comp_fcn_common import t_beg, t_end, nz, z_mid, dz_r, dz_mid_r, mixing_coeff
from model import ModelState, ModelStaticVars, tracer_module_names, tracer_names

def _parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="test problem for Newton-Krylov solver")
    parser.add_argument('in_fname', help='name of file with input')
    parser.add_argument('res_fname', help='name of file for output')
    parser.add_argument('--cfg_fname', help='name of configuration file',
                        default='newton_krylov.cfg')
    parser.add_argument('--hist_fname', help='name of history file',
                        default='None')
    return parser.parse_args()

def main(args):
    """test problem for Newton-Krylov solver"""

    config = configparser.ConfigParser()
    config.read(args.cfg_fname)
    ModelStaticVars(config['modelinfo'], args.cfg_fname)

    ms_in = ModelState(args.in_fname)

    tracer_names_loc = tracer_names()
    tracer_cnt = len(tracer_names_loc)
    y0 = np.empty((tracer_cnt, nz))
    for tracer_ind, tracer_name in enumerate(tracer_names_loc):
        y0[tracer_ind, :] = ms_in.get_tracer_vals(tracer_name)

    # get dense output, if requested
    t_eval = np.linspace(t_beg, t_end, 101 if args.hist_fname != 'None' else 2)

    sol = solve_ivp(comp_tend, (t_beg, t_end), y0.reshape(-1), 'Radau', t_eval,
                    atol=1.0e-10, rtol=1.0e-10)

    if not args.hist_fname == 'None':
        write_hist(sol, args.hist_fname)

    ms_res = ms_in.copy()
    res_vals = sol.y[:, -1].reshape((tracer_cnt, nz)) - y0
    for tracer_ind, tracer_name in enumerate(tracer_names_loc):
        ms_res.set_tracer_vals(tracer_name, res_vals[tracer_ind, :])

    ms_res.dump(args.res_fname)

def comp_tend(t, y_flat):
    """compute tendency function"""
    tracer_names_loc = tracer_names()
    tracer_cnt = len(tracer_names_loc)
    y = y_flat.reshape((tracer_cnt, -1))
    dy_dt = np.empty_like(y)
    for tracer_module_name in tracer_module_names():
        if tracer_module_name == 'IAGE':
            tracer_ind = tracer_names_loc.index('IAGE')
            dy_dt[tracer_ind, :] = comp_tend_IAGE(t, y[tracer_ind, :])
        # if tracer_module_name == 'PO4_POP':
        #     comp_precond_PO4_POP(iterate, ms_in, ms_res)
    return dy_dt.reshape(-1)

def comp_tend_IAGE(t, y):
    """compute tendency for IAGE tracers"""
    # age 1/year
    dy_dt = (1.0 / 365.0) * np.ones_like(y) + mixing_tend(t, y)
    # restore in surface to 0 at a rate of 24.0/day
    dy_dt[0] = -24.0 * y[0]
    return dy_dt

# def comp_tend_PO4_POP(ms_in, ms_res):
#     """compute tendency for PO4_POP tracers"""
#     PO4 = ms_in.get_tracer_vals('PO4')
#     POP = ms_in.get_tracer_vals('POP')
#
#     # compute tendencies, units are per-day
#     # light has e-folding decay of 25m, PO4 half-saturation = 0.5
#     PO4_lim = PO4 / (PO4 + 0.5)
#     PO4_uptake = np.where((z_mid < 100.0) & (PO4 > 0.0),
#                           np.exp((-1.0 / 25.0) * z_mid) * PO4_lim, 0.0)
#     POP_remin = np.where(POP > 0.0, 0.1 * POP, 0.0)
#
#     dPO4_dt = -PO4_uptake + POP_remin + mixing_tend(PO4)
#     dPOP_dt = PO4_uptake - POP_remin + mixing_tend(POP) + sinking_tend(POP)
#
#     ms_res.set_tracer_vals('PO4', dPO4_dt)
#     ms_res.set_tracer_vals('POP', dPOP_dt)

def mixing_tend(t, tracer):
    """tracer tendency from mixing"""
    tracer_grad = np.zeros(1+nz)
    tracer_grad[1:-1] = np.ediff1d(tracer) * dz_mid_r
    tracer_flux = -1.0 * mixing_coeff(t) * tracer_grad
    return -1.0 * np.ediff1d(tracer_flux) * dz_r

def sinking_tend(tracer):
    """tracer tendency from sinking"""
    tracer_flux = np.zeros(1+nz)
    tracer_flux[1:-1] = 5.0 * tracer[:-1] # assume velocity is 5 m/day
    return -1.0 * np.ediff1d(tracer_flux) * dz_r

def write_hist(sol, hist_fname):
    """write generated tracer values to hist_fname"""
    tracer_names_loc = tracer_names()
    tracer_cnt = len(tracer_names_loc)
    with nc.Dataset(hist_fname, mode='w') as fptr:
        fptr.createDimension('time', None)
        fptr.createDimension('depth', nz)

        fptr.createVariable('time', 'f8', dimensions=('time',))
        fptr.createVariable('depth', 'f8', dimensions=('depth',))
        for tracer_name in tracer_names_loc:
            fptr.createVariable(tracer_name, 'f8', dimensions=('time', 'depth'))

        fptr.variables['time'][:] = sol.t
        fptr.variables['depth'][:] = z_mid
        for tracer_ind, tracer_name in enumerate(tracer_names_loc):
            tracer_vals = sol.y.reshape((tracer_cnt, nz, -1))[tracer_ind, :, :]
            fptr.variables[tracer_name][:] = tracer_vals.transpose()

if __name__ == '__main__':
    main(_parse_args())
