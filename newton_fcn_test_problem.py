#!/usr/bin/env python
"""test problem for Newton-Krylov solver"""

import argparse
import configparser
import logging
import subprocess
import sys

import numpy as np
from scipy.linalg import solve_banded, svd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp

from netCDF4 import Dataset

from model import TracerModuleStateBase, ModelState, ModelStaticVars

def _parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="test problem for Newton-Krylov solver")
    parser.add_argument('cmd', choices=['comp_fcn', 'apply_precond_jacobian'],
                        help='command to run')
    parser.add_argument('in_fname', help='name of file with input')
    parser.add_argument('res_fname', help='name of file for result')
    parser.add_argument('--cfg_fname', help='name of configuration file',
                        default='newton_krylov.cfg')
    parser.add_argument('--hist_fname', help='name of history file', default='None')
    parser.add_argument('--resume_script_fname', help='name of script to resume nk_driver.py',
                        default='None')
    return parser.parse_args()

def main(args):
    """test problem for Newton-Krylov solver"""

    config = configparser.ConfigParser()
    config.read(args.cfg_fname)
    solverinfo = config['solverinfo']

    if args.resume_script_fname == 'None':
        logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s',
                            level=solverinfo['logging_level'])
    else:
        logging.basicConfig(filename=solverinfo['logging_fname'], filemode='a',
                            format='%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s',
                            level=solverinfo['logging_level'])
    logger = logging.getLogger(__name__)

    logger.info('entering, cmd=%s', args.cmd)

    # store cfg_fname in modelinfo, to ease access to its values elsewhere
    config['modelinfo']['cfg_fname'] = args.cfg_fname

    ModelStaticVars(config['modelinfo'])

    newton_fcn = NewtonFcn()

    if args.cmd == 'comp_fcn':
        ms_res = newton_fcn.comp_fcn(ModelState(args.in_fname), args.hist_fname)
    elif args.cmd == 'apply_precond_jacobian':
        ms_res = newton_fcn.apply_precond_jacobian(ModelState(args.in_fname))
    else:
        raise ValueError('unknown cmd=%s' % args.cmd)

    ms_res.dump(args.res_fname)

    if args.resume_script_fname != 'None':
        logger.info('resuming with %s', args.resume_script_fname)
        subprocess.Popen(args.resume_script_fname)
    else:
        logger.info('done')

################################################################################

class TracerModuleState(TracerModuleStateBase):
    """
    Derived class for representing a collection of model tracers.
    It implements _read_vals and dump.
    """

    def _read_vals(self, tracer_module_name, vals_fname):
        """return tracer values and dimension names and lengths, read from vals_fname)"""
        dims = {}
        with Dataset(vals_fname, mode='r') as fptr:
            fptr.set_auto_mask(False)
            # get dims from first variable
            dimnames0 = fptr.variables[self.tracer_names()[0]].dimensions
            for dimname in dimnames0:
                dims[dimname] = fptr.dimensions[dimname].size
            # all tracers are stored in a single array
            # tracer index is the leading index
            vals = np.empty(shape=(self.tracer_cnt(),) + tuple(dims.values()))
            # check that all vars have the same dimensions
            for tracer_name in self.tracer_names():
                if fptr.variables[tracer_name].dimensions != dimnames0:
                    raise ValueError('not all vars have same dimensions',
                                     'tracer_module_name=', tracer_module_name,
                                     'vals_fname=', vals_fname)
            # read values
            if len(dims) > 3:
                raise ValueError('ndim too large (for implementation of dot_prod)',
                                 'tracer_module_name=', tracer_module_name,
                                 'vals_fname=', vals_fname,
                                 'ndim=', len(dims))
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                varid = fptr.variables[tracer_name]
                vals[tracer_ind, :] = varid[:]
        return vals, dims

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleState object
        to an open file
        """
        if action == 'define':
            for dimname, dimlen in self._dims.items():
                try:
                    if fptr.dimensions[dimname].size != dimlen:
                        raise ValueError('dimname already exists and has wrong size',
                                         'tracer_module_name=', self._tracer_module_name,
                                         'dimname=', dimname)
                except KeyError:
                    fptr.createDimension(dimname, dimlen)
            dimnames = tuple(self._dims.keys())
            for tracer_name in self.tracer_names():
                fptr.createVariable(tracer_name, 'f8', dimensions=dimnames)
        elif action == 'write':
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                fptr.variables[tracer_name][:] = self._vals[tracer_ind, :]
        else:
            raise ValueError('unknown action=', action)
        return self

################################################################################

class NewtonFcn():
    """class of methods related to problem being solved with Newton's method"""
    def __init__(self):
        self.time_range = (0.0, 365.0)
        self.depth = Depth('grid_files/depth_axis_test.nc')

        # tracer_module_names and tracer_names will be stored in the following attributes,
        # enabling access to them from inside _comp_tend
        self._tracer_module_names = None
        self._tracer_names = None

    def comp_fcn(self, ms_in, hist_fname='None'):
        """evalute function being solved with Newton's method"""
        self._tracer_module_names = ms_in.tracer_module_names
        self._tracer_names = ms_in.tracer_names()
        tracer_vals_init = np.empty((len(self._tracer_names), self.depth.axis.nlevs))
        for tracer_ind, tracer_name in enumerate(self._tracer_names):
            tracer_vals_init[tracer_ind, :] = ms_in.get_tracer_vals(tracer_name)

        # solve ODEs, using scipy.integrate
        # get dense output, if requested
        sol = solve_ivp(self._comp_tend, self.time_range, tracer_vals_init.reshape(-1), 'Radau',
                        np.linspace(self.time_range[0], self.time_range[1],
                                    101 if hist_fname != 'None' else 2),
                        atol=1.0e-10, rtol=1.0e-10)

        if hist_fname != 'None':
            self._write_hist(sol, hist_fname)

        ms_res = ms_in.copy()
        res_vals = sol.y[:, -1].reshape(tracer_vals_init.shape) - tracer_vals_init
        for tracer_ind, tracer_name in enumerate(self._tracer_names):
            ms_res.set_tracer_vals(tracer_name, res_vals[tracer_ind, :])

        return ms_res

    def _comp_tend(self, time, tracer_vals_flat):
        """compute tendency function"""
        tracer_vals = tracer_vals_flat.reshape((len(self._tracer_names), -1))
        dtracer_vals_dt = np.empty_like(tracer_vals)
        for tracer_module_name in self._tracer_module_names:
            if tracer_module_name == 'iage_test':
                tracer_ind = self._tracer_names.index('iage_test')
                self._comp_tend_iage_test(time, tracer_vals[tracer_ind, :],
                                          dtracer_vals_dt[tracer_ind, :])
            if tracer_module_name == 'phosphorus':
                tracer_ind0 = self._tracer_names.index('po4')
                self._comp_tend_phosphorus(time, tracer_vals[tracer_ind0:tracer_ind0+6, :],
                                           dtracer_vals_dt[tracer_ind0:tracer_ind0+6, :])
        return dtracer_vals_dt.reshape(-1)

    def _comp_tend_iage_test(self, time, tracer_vals, dtracer_vals_dt):
        """
        compute tendency for iage_test
        tendency units are tr_units / day
        """
        # age 1/year
        dtracer_vals_dt[:] = (1.0 / 365.0) + self.depth.mixing_tend(time, tracer_vals)
        # restore in surface to 0 at a rate of 24.0 / day
        dtracer_vals_dt[0] = -24.0 * tracer_vals[0]

    def _comp_tend_phosphorus(self, time, tracer_vals, dtracer_vals_dt):
        """
        compute tendency for phosphorus tracers
        tendency units are tr_units / day
        """

        # light has e-folding decay of 25m, po4 half-saturation = 0.5
        po4 = tracer_vals[0, :]
        po4_lim = np.where(po4 > 0.0, po4 / (po4 + 0.5), 0.0)
        po4_uptake = np.exp((-1.0 / 25.0) * self.depth.axis.mid) * po4_lim

        self._comp_tend_phosphorus_core(time, po4_uptake, tracer_vals[0:3, :],
                                        dtracer_vals_dt[0:3, :])
        self._comp_tend_phosphorus_core(time, po4_uptake, tracer_vals[3:6, :],
                                        dtracer_vals_dt[3:6, :])

        # restore po4_s to po4, at a rate of 1 / day
        # compensate equally from and dop and pop, so that total shadow phosphorus is conserved
        rest_term = 1.0 * (dtracer_vals_dt[0, 0] - dtracer_vals_dt[3, 0])
        dtracer_vals_dt[3, 0] += rest_term
        dtracer_vals_dt[4, 0] -= 0.67 * rest_term
        dtracer_vals_dt[5, 0] -= 0.33 * rest_term

    def _comp_tend_phosphorus_core(self, time, po4_uptake, tracer_vals, dtracer_vals_dt):
        """
        core fuction for computing tendency for phosphorus tracers
        tendency units are tr_units / day
        """

        po4 = tracer_vals[0, :]
        dop = tracer_vals[1, :]
        pop = tracer_vals[2, :]

        # dop remin rate is 1% / day
        dop_remin = np.where(dop > 0.0, 0.01 * dop, 0.0)
        # pop remin rate is 1% / day
        pop_remin = np.where(pop > 0.0, 0.01 * pop, 0.0)

        sigma = 0.67

        dtracer_vals_dt[0, :] = -po4_uptake + dop_remin + pop_remin \
            + self.depth.mixing_tend(time, po4)
        dtracer_vals_dt[1, :] = sigma * po4_uptake - dop_remin \
            + self.depth.mixing_tend(time, dop)
        dtracer_vals_dt[2, :] = (1.0 - sigma) * po4_uptake - pop_remin \
            + self.depth.mixing_tend(time, pop) + self._sinking_tend(pop)

    def _sinking_tend(self, tracer_vals):
        """tracer tendency from sinking"""
        tracer_flux_neg = np.zeros(1+self.depth.axis.nlevs)
        tracer_flux_neg[1:-1] = -tracer_vals[:-1] # assume velocity is 1 m / day
        return np.ediff1d(tracer_flux_neg) * self.depth.axis.delta_r

    def _write_hist(self, sol, hist_fname):
        """write tracer values generated in comp_fcn to hist_fname"""
        with Dataset(hist_fname, mode='w') as fptr:
            fptr.createDimension('time', None)
            fptr.createDimension('depth', self.depth.axis.nlevs)

            fptr.createVariable('time', 'f8', dimensions=('time',))
            fptr.createVariable('depth', 'f8', dimensions=('depth',))
            for tracer_name in self._tracer_names:
                fptr.createVariable(tracer_name, 'f8', dimensions=('time', 'depth'))

            fptr.variables['time'][:] = sol.t
            fptr.variables['depth'][:] = self.depth.axis.mid

            tracer_vals = sol.y.reshape((len(self._tracer_names), self.depth.axis.nlevs, -1))
            for tracer_ind, tracer_name in enumerate(self._tracer_names):
                fptr.variables[tracer_name][:] = tracer_vals[tracer_ind, :, :].transpose()

    def apply_precond_jacobian(self, ms_in):
        """apply preconditioner of jacobian of comp_fcn to model state object, ms_in"""

        ms_res = ms_in.copy()

        mca = self.depth.mixing_coeff_time_op(self.time_range, 'log_avg', 100)

        for tracer_module_name in ms_in.tracer_module_names:
            if tracer_module_name == 'iage_test':
                self._apply_precond_jacobian_iage_test(ms_in, mca, ms_res)
            if tracer_module_name == 'phosphorus':
                self._apply_precond_jacobian_phosphorus(ms_in, mca, ms_res)

        return ms_res

    def _apply_precond_jacobian_iage_test(self, ms_in, mca, ms_res):
        """apply preconditioner of jacobian of iage_test fcn"""

        iage_test_in = ms_in.get_tracer_vals('iage_test')
        rhs = (1.0 / (self.time_range[1] - self.time_range[0])) * iage_test_in

        l_and_u = (1, 1)
        matrix_diagonals = np.zeros((3, self.depth.axis.nlevs))
        matrix_diagonals[0, 1:] = mca[1:-1] * self.depth.axis.delta_mid_r \
            * self.depth.axis.delta_r[:-1]
        matrix_diagonals[1, :-1] -= mca[1:-1] * self.depth.axis.delta_mid_r \
            * self.depth.axis.delta_r[:-1]
        matrix_diagonals[1, 1:] -= mca[1:-1] * self.depth.axis.delta_mid_r \
            * self.depth.axis.delta_r[1:]
        matrix_diagonals[2, :-1] = mca[1:-1] * self.depth.axis.delta_mid_r \
            * self.depth.axis.delta_r[1:]
        matrix_diagonals[1, 0] = -24.0
        matrix_diagonals[0, 1] = 0

        res = solve_banded(l_and_u, matrix_diagonals, rhs)

        ms_res.set_tracer_vals('iage_test', res - iage_test_in)

    def _apply_precond_jacobian_phosphorus(self, ms_in, mca, ms_res):
        """
        apply preconditioner of jacobian of phosphorus fcn
        it is only applied to shadow phosphorus tracers
        """

        po4_s = ms_in.get_tracer_vals('po4_s')
        dop_s = ms_in.get_tracer_vals('dop_s')
        pop_s = ms_in.get_tracer_vals('pop_s')
        rhs = (1.0 / (self.time_range[1] - self.time_range[0])) \
            * np.concatenate((po4_s, dop_s, pop_s))

        nz = self.depth.axis.nlevs # pylint: disable=C0103

        matrix = diags([self._diag_0_phosphorus(mca),
                        self._diag_p_1_phosphorus(mca), self._diag_m_1_phosphorus(mca),
                        self._diag_p_nz_phosphorus(), self._diag_m_nz_phosphorus(),
                        self._diag_p_2nz_phosphorus(), self._diag_m_2nz_phosphorus()],
                       [0, 1, -1, nz, -nz, 2*nz, -2*nz], format='csr')

        res = spsolve(matrix, rhs)

        _, sing_vals, r_sing_vects = svd(matrix.todense())
        min_ind = sing_vals.argmin()
        res -= (res.mean()/r_sing_vects[min_ind, :].mean()) * r_sing_vects[min_ind, :]

        ms_res.set_tracer_vals('po4_s', res[0:nz] - po4_s)
        ms_res.set_tracer_vals('dop_s', res[nz:2*nz] - dop_s)
        ms_res.set_tracer_vals('pop_s', res[2*nz:3*nz] - pop_s)

    def _diag_0_phosphorus(self, mca):
        """return main diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_0_single_tracer = np.zeros(self.depth.axis.nlevs)
        diag_0_single_tracer[:-1] -= mca[1:-1] * self.depth.axis.delta_mid_r \
            * self.depth.axis.delta_r[:-1]
        diag_0_single_tracer[1:] -= mca[1:-1] * self.depth.axis.delta_mid_r \
            * self.depth.axis.delta_r[1:]
        diag_0_po4_s = diag_0_single_tracer.copy()
        diag_0_po4_s[0] -= 1.0 # po4_s restoring in top layer
        diag_0_dop_s = diag_0_single_tracer.copy()
        diag_0_dop_s -= 0.01 # dop_s remin
        diag_0_pop_s = diag_0_single_tracer.copy()
        diag_0_pop_s -= 0.01 # pop_s remin
        diag_0_pop_s[:-1] -= 1.0 * self.depth.axis.delta_r[:-1] # pop_s sinking loss to layer below
        return np.concatenate((diag_0_po4_s, diag_0_dop_s, diag_0_pop_s))

    def _diag_p_1_phosphorus(self, mca):
        """return +1 upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_p_1_single_tracer = mca[1:-1] * self.depth.axis.delta_mid_r \
            * self.depth.axis.delta_r[:-1]
        diag_p_1_po4_s = diag_p_1_single_tracer.copy()
        zero = np.zeros(1)
        diag_p_1_dop_s = diag_p_1_single_tracer.copy()
        diag_p_1_pop_s = diag_p_1_single_tracer.copy()
        return np.concatenate((diag_p_1_po4_s, zero, diag_p_1_dop_s, zero, diag_p_1_pop_s))

    def _diag_m_1_phosphorus(self, mca):
        """return +1 upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_m_1_single_tracer = mca[1:-1] * self.depth.axis.delta_mid_r \
            * self.depth.axis.delta_r[1:]
        diag_m_1_po4_s = diag_m_1_single_tracer.copy()
        zero = np.zeros(1)
        diag_m_1_dop_s = diag_m_1_single_tracer.copy()
        diag_m_1_pop_s = diag_m_1_single_tracer.copy()
        diag_m_1_pop_s += 1.0 * self.depth.axis.delta_r[1:] # pop_s sinking gain from layer above
        return np.concatenate((diag_m_1_po4_s, zero, diag_m_1_dop_s, zero, diag_m_1_pop_s))

    def _diag_p_nz_phosphorus(self):
        """return +nz upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_p_1_dop_po4 = 0.01 * np.ones(self.depth.axis.nlevs) # dop_s remin
        diag_p_1_pop_dop = np.zeros(self.depth.axis.nlevs)
        return np.concatenate((diag_p_1_dop_po4, diag_p_1_pop_dop))

    def _diag_m_nz_phosphorus(self):
        """return -nz lower diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_p_1_po4_dop = np.zeros(self.depth.axis.nlevs)
        diag_p_1_po4_dop[0] = 0.67 # po4_s restoring conservation balance
        diag_p_1_dop_pop = np.zeros(self.depth.axis.nlevs)
        return np.concatenate((diag_p_1_po4_dop, diag_p_1_dop_pop))

    def _diag_p_2nz_phosphorus(self):
        """return +2nz upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        return 0.01 * np.ones(self.depth.axis.nlevs) # pop_s remin

    def _diag_m_2nz_phosphorus(self):
        """return -2nz lower diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_p_1_po4_pop = np.zeros(self.depth.axis.nlevs)
        diag_p_1_po4_pop[0] = 0.33 # po4_s restoring conservation balance
        return diag_p_1_po4_pop

class SpatialAxis():
    """class for spatial axis related quantities"""
    def __init__(self, fname, edges_varname):
        with Dataset(fname) as fptr:
            fptr.set_auto_mask(False)
            self.edges = fptr.variables[edges_varname][:]
            self.mid = 0.5 * (self.edges[:-1] + self.edges[1:])
            self.delta = np.ediff1d(self.edges)
            self.delta_r = 1.0 / self.delta
            self.delta_mid_r = 1.0 / np.ediff1d(self.mid)
            self.nlevs = len(self.mid)

class Depth():
    """class for depth axis vals and methods"""
    def __init__(self, depth_fname):
        self.axis = SpatialAxis(depth_fname, 'depth_edges')

        self._time_val = None
        self._mixing_coeff_vals = np.empty(self.axis.nlevs)

    def mixing_tend(self, time, tracer_vals):
        """tracer tendency from mixing"""
        tracer_grad = np.zeros(1+self.axis.nlevs)
        tracer_grad[1:-1] = np.ediff1d(tracer_vals) * self.axis.delta_mid_r
        tracer_flux_neg = self.mixing_coeff(time) * tracer_grad
        return np.ediff1d(tracer_flux_neg) * self.axis.delta_r

    def mixing_coeff(self, time, log=False):
        """
        vertical mixing coefficient, m2 d-1
        if log==True, then return the natural log of the vertical mixing coefficient
        store computed vals, so their computation can be skipped on subsequent calls
        """

        # if vals have already been computed for this time, skip computation
        if not log and time == self._time_val:
            return self._mixing_coeff_vals

        bldepth_min = 50.0
        bldepth_max = 150.0
        bldepth_del = bldepth_max - bldepth_min
        bldepth = bldepth_min \
            + bldepth_del * (0.5 + 0.5 * np.cos((2 * np.pi) * ((time / 365.0) - 0.25)))
        # z_lin ranges from 0.0 to 1.0 over span of 50.0 m, is 0.5 at bldepth
        z_lin = np.maximum(0.0, np.minimum(1.0, 0.5 + (self.axis.edges - bldepth) * (1.0 / 50.0)))
        res_log10_shallow = 0.0
        res_log10_deep = -5.0
        res_log10_del = res_log10_deep - res_log10_shallow
        res_log10 = res_log10_shallow + res_log10_del * z_lin
        if log:
            # avoid 10.0 ** ... operation if log(mixing_coeff) is the desired quantity
            log_spd = 11.366742954792146 # natural log of 86400
            log_10 = 2.302585092994046 # natural log of 10.0
            return log_spd + log_10 * res_log10
        self._time_val = time
        self._mixing_coeff_vals = 86400.0 * 10.0 ** res_log10
        return self._mixing_coeff_vals

    def mixing_coeff_time_op(self, time_range, time_op, samps):
        """
        mixing_coeff over the interval specified by time_range, reduced with time_op
        valid time_op values: 'log_avg', 'avg', 'max'
        compute using midpoint rule with samps intervals
        """
        res = np.zeros(1+self.axis.nlevs)
        t_del_samp = (time_range[1] - time_range[0]) / samps
        if time_op == 'log_avg':
            for t_ind in range(samps):
                res += self.mixing_coeff(time_range[0] + (t_ind + 0.5) * t_del_samp, log=True)
            res *= (1.0 / samps)
            res[:] = np.exp(res[:])
        elif time_op == 'avg':
            for t_ind in range(samps):
                res += self.mixing_coeff(time_range[0] + (t_ind + 0.5) * t_del_samp)
            res *= (1.0 / samps)
        elif time_op == 'max':
            for t_ind in range(samps):
                res = np.maximum(res, self.mixing_coeff(time_range[0] + (t_ind + 0.5) * t_del_samp))
        else:
            raise ValueError('unknown time_op=%s' % time_op)
        return res

################################################################################

if __name__ == '__main__':
    main(_parse_args())
