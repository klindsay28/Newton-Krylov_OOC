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
    tracer_vals_init = np.empty((tracer_cnt, nz))
    for tracer_ind, tracer_name in enumerate(tracer_names_loc):
        tracer_vals_init[tracer_ind, :] = ms_in.get_tracer_vals(tracer_name)

    # solve ODEs, using scipy.integrate
    # get dense output, if requested
    sol = solve_ivp(comp_tend, (t_beg, t_end), tracer_vals_init.reshape(-1), 'Radau',
                    np.linspace(t_beg, t_end, 101 if args.hist_fname != 'None' else 2),
                    atol=1.0e-8, rtol=1.0e-8)

    if not args.hist_fname == 'None':
        write_hist(sol, args.hist_fname)

    ms_res = ms_in.copy()
    res_vals = sol.y[:, -1].reshape((tracer_cnt, nz)) - tracer_vals_init
    for tracer_ind, tracer_name in enumerate(tracer_names_loc):
        ms_res.set_tracer_vals(tracer_name, res_vals[tracer_ind, :])

    ms_res.dump(args.res_fname)

def comp_tend(time, tracer_vals_flat):
    """compute tendency function"""
    tracer_names_loc = tracer_names()
    tracer_cnt = len(tracer_names_loc)
    tracer_vals = tracer_vals_flat.reshape((tracer_cnt, -1))
    dtracer_vals_dt = np.empty_like(tracer_vals)
    kappa = mixing_coeff(time)
    for tracer_module_name in tracer_module_names():
        if tracer_module_name == 'iage':
            tracer_ind = tracer_names_loc.index('iage')
            comp_tend_iage(kappa, tracer_vals[tracer_ind, :], dtracer_vals_dt[tracer_ind, :])
        if tracer_module_name == 'phosphorus':
            tracer_ind0 = tracer_names_loc.index('po4')
            comp_tend_phosphorus(kappa, tracer_vals[tracer_ind0:tracer_ind0+3, :],
                                 dtracer_vals_dt[tracer_ind0:tracer_ind0+3, :])
    return dtracer_vals_dt.reshape(-1)

def comp_tend_iage(kappa, tracer_vals, dtracer_vals_dt):
    """compute tendency for iage"""
    # age 1/year
    dtracer_vals_dt[:] = (1.0 / 365.0) + mixing_tend(kappa, tracer_vals)
    # restore in surface to 0 at a rate of 24.0/day
    dtracer_vals_dt[0] = -24.0 * tracer_vals[0]

def comp_tend_phosphorus(kappa, tracer_vals, dtracer_vals_dt):
    """compute tendency for phosphorus tracers"""

    po4 = tracer_vals[0, :]
    dop = tracer_vals[1, :]
    pop = tracer_vals[2, :]

    # compute tendencies, units are per-day
    # light has e-folding decay of 25m, po4 half-saturation = 0.5
    po4_lim = po4 / (po4 + 0.5)
    po4_uptake = np.where(po4 > 0.0, np.exp((-1.0 / 25.0) * z_mid) * po4_lim, 0.0)
    # dop remin rate is 1% / day
    dop_remin = np.where(dop > 0.0, 0.01 * dop, 0.0)
    # pop remin rate is 0.5% / day
    pop_remin = np.where(pop > 0.0, 0.005 * pop, 0.0)

    sigma = 0.67

    dtracer_vals_dt[0, :] = -po4_uptake + dop_remin + pop_remin + mixing_tend(kappa, po4)
    dtracer_vals_dt[1, :] = sigma * po4_uptake - dop_remin + mixing_tend(kappa, dop)
    dtracer_vals_dt[2, :] = (1.0 - sigma) * po4_uptake - pop_remin + mixing_tend(kappa, pop) \
                  + sinking_tend(pop)

def mixing_tend(kappa, tracer_vals):
    """tracer tendency from mixing"""
    tracer_grad = np.zeros(1+nz)
    tracer_grad[1:-1] = np.ediff1d(tracer_vals) * dz_mid_r
    tracer_flux = -1.0 * kappa * tracer_grad
    return -1.0 * np.ediff1d(tracer_flux) * dz_r

def sinking_tend(tracer_vals):
    """tracer tendency from sinking"""
    tracer_flux = np.zeros(1+nz)
    tracer_flux[1:-1] = 1.0 * tracer_vals[:-1] # assume velocity is 1 m/day
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
