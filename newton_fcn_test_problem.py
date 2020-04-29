#!/usr/bin/env python
"""test problem for Newton-Krylov solver"""

import argparse
import configparser
import logging
import os
import subprocess
import sys

import numpy as np
from scipy.linalg import solve_banded, svd
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp

from netCDF4 import Dataset

from model import ModelStateBase, TracerModuleStateBase
from model_config import ModelConfig, get_modelinfo
from newton_fcn_base import NewtonFcnBase
from solver import SolverState

def _parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="test problem for Newton-Krylov solver")
    parser.add_argument(
        'cmd', choices=['comp_fcn', 'gen_precond_jacobian', 'apply_precond_jacobian'],
        help='command to run')
    parser.add_argument(
        '--cfg_fname', help='name of configuration file', default='newton_krylov.cfg')
    parser.add_argument(
        '--workdir', help='directory that filename are relative to', default='.')
    parser.add_argument('--hist_fname', help='name of history file', default=None)
    parser.add_argument('--precond_fname', help='name of precond file', default=None)
    parser.add_argument('--in_fname', help='name of file with input')
    parser.add_argument('--res_fname', help='name of file for result')
    return parser.parse_args()

def main(args):
    """test problem for Newton-Krylov solver"""

    config = configparser.ConfigParser()
    config.read(args.cfg_fname)
    solverinfo = config['solverinfo']

    logging_format = '%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s'
    logging.basicConfig(
        stream=sys.stdout, format=logging_format, level=solverinfo['logging_level'])
    logger = logging.getLogger(__name__)

    logger.info('args.cmd="%s"', args.cmd)

    # store cfg_fname in modelinfo, to ease access to its value elsewhere
    config['modelinfo']['cfg_fname'] = args.cfg_fname

    ModelConfig(config['modelinfo'])

    newton_fcn = NewtonFcn()

    solver_state = SolverState('newton_fcn_test_problem', args.workdir)

    ms_in = ModelState(os.path.join(args.workdir, args.in_fname))
    if args.cmd == 'comp_fcn':
        ms_in.log('state_in')
        newton_fcn.comp_fcn(
            ms_in, os.path.join(args.workdir, args.res_fname), None,
            os.path.join(args.workdir, args.hist_fname))
        ModelState(os.path.join(args.workdir, args.res_fname)).log('fcn')
    elif args.cmd == 'gen_precond_jacobian':
        newton_fcn.gen_precond_jacobian(
            ms_in, os.path.join(args.workdir, args.hist_fname),
            os.path.join(args.workdir, args.precond_fname), solver_state)
    elif args.cmd == 'apply_precond_jacobian':
        ms_in.log('state_in')
        newton_fcn.apply_precond_jacobian(
            ms_in, os.path.join(args.workdir, args.precond_fname),
            os.path.join(args.workdir, args.res_fname), solver_state)
        ModelState(os.path.join(args.workdir, args.res_fname)).log('precond_res')
    else:
        msg = 'unknown cmd=%s' % args.cmd
        raise ValueError(msg)

    logger.info('done')

################################################################################

class ModelState(ModelStateBase):
    """class for representing the state space of a model"""

    # give ModelState operators higher priority than those of numpy
    __array_priority__ = 100

    def __init__(self, vals_fname=None):
        logger = logging.getLogger(__name__)
        logger.debug('ModelState, vals_fname="%s"', vals_fname)
        super().__init__(TracerModuleState, vals_fname)

################################################################################

class TracerModuleState(TracerModuleStateBase):
    """
    Derived class for representing a collection of model tracers.
    It implements _read_vals and dump.
    """

    def _read_vals(self, tracer_module_name, vals_fname):
        """return tracer values and dimension names and lengths, read from vals_fname)"""
        logger = logging.getLogger(__name__)
        logger.debug(
            'tracer_module_name="%s", vals_fname="%s"', tracer_module_name, vals_fname)
        dims = {}
        with Dataset(vals_fname, mode='r') as fptr:
            fptr.set_auto_mask(False)
            # get dims from first variable
            dimnames0 = fptr.variables[self.tracer_names()[0]].dimensions
            for dimname in dimnames0:
                dims[dimname] = fptr.dimensions[dimname].size
            # all tracers are stored in a single array
            # tracer index is the leading index
            vals = np.empty((self.tracer_cnt(),) + tuple(dims.values()))
            # check that all vars have the same dimensions
            for tracer_name in self.tracer_names():
                if fptr.variables[tracer_name].dimensions != dimnames0:
                    msg = 'not all vars have same dimensions' \
                        ', tracer_module_name=%s, vals_fname=%s' \
                        % (tracer_module_name, vals_fname)
                    raise ValueError(msg)
            # read values
            if len(dims) > 3:
                msg = 'ndim too large (for implementation of dot_prod)' \
                    'tracer_module_name=%s, vals_fname=%s, ndim=%s' \
                    % (tracer_module_name, vals_fname, len(dims))
                raise ValueError(msg)
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
                        msg = 'dimname already exists and has wrong size' \
                            'tracer_module_name=%s, dimname=%s' \
                            % (self._tracer_module_name, dimname)
                        raise ValueError(msg)
                except KeyError:
                    fptr.createDimension(dimname, dimlen)
            dimnames = tuple(self._dims.keys())
            # define all tracers
            for tracer_name in self.tracer_names():
                fptr.createVariable(tracer_name, 'f8', dimensions=dimnames)
        elif action == 'write':
            # write all tracers
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                fptr.variables[tracer_name][:] = self._vals[tracer_ind, :]
        else:
            msg = 'unknown action=%s', action
            raise ValueError(msg)
        return self

################################################################################

class NewtonFcn(NewtonFcnBase):
    """class of methods related to problem being solved with Newton's method"""
    def __init__(self):
        self.time_range = (0.0, 365.0)
        self.depth = Depth('grid_files/depth_axis_test.nc')

        # light has e-folding decay of 25m
        self.light_lim = np.exp((-1.0 / 25.0) * self.depth.axis.mid)

        # tracer_module_names and tracer_names will be stored in the following attributes,
        # enabling access to them from inside _comp_tend
        self._tracer_module_names = None
        self._tracer_names = None

    def model_state_obj(self, fname=None):
        """return a ModelState object compatible with this function"""
        return ModelState(fname)

    def comp_fcn(self, ms_in, res_fname, solver_state, hist_fname=None):
        """evalute function being solved with Newton's method"""
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s", hist_fname="%s"', res_fname, hist_fname)

        if solver_state is not None:
            fcn_complete_step = 'comp_fcn complete for %s' % res_fname
            if solver_state.step_logged(fcn_complete_step):
                logger.debug('"%s" logged, returning result', fcn_complete_step)
                return ModelState(res_fname)
            logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        self._tracer_module_names = ms_in.tracer_module_names
        self._tracer_names = ms_in.tracer_names()
        tracer_vals_init = np.empty((len(self._tracer_names), self.depth.axis.nlevs))
        for tracer_ind, tracer_name in enumerate(self._tracer_names):
            tracer_vals_init[tracer_ind, :] = ms_in.get_tracer_vals(tracer_name)

        # solve ODEs, using scipy.integrate
        # get dense output, if requested
        t_eval = np.linspace(
            self.time_range[0], self.time_range[1],
            101 if hist_fname is not None else 2)
        sol = solve_ivp(
            self._comp_tend, self.time_range, tracer_vals_init.reshape(-1), 'Radau',
            t_eval, atol=1.0e-10, rtol=1.0e-10)

        if hist_fname is not None:
            self._write_hist(sol, hist_fname)

        ms_res = ms_in.copy()
        res_vals = sol.y[:, -1].reshape(tracer_vals_init.shape) - tracer_vals_init
        for tracer_ind, tracer_name in enumerate(self._tracer_names):
            ms_res.set_tracer_vals(tracer_name, res_vals[tracer_ind, :])

        self.comp_fcn_postprocess(ms_res, res_fname)

        if solver_state is not None:
            solver_state.log_step(fcn_complete_step)
            logger.debug('invoking resume script and exiting')
            # use Popen instead of run because we don't want to wait
            subprocess.Popen([get_modelinfo('nk_driver_invoker_fname'), '--resume'])
            raise SystemExit

        return ms_res

    def _comp_tend(self, time, tracer_vals_flat):
        """compute tendency function"""
        tracer_vals = tracer_vals_flat.reshape((len(self._tracer_names), -1))
        dtracer_vals_dt = np.empty_like(tracer_vals)
        for tracer_module_name in self._tracer_module_names:
            if tracer_module_name == 'iage_test':
                tracer_ind = self._tracer_names.index('iage_test')
                self._comp_tend_iage_test(
                    time, tracer_vals[tracer_ind, :], dtracer_vals_dt[tracer_ind, :])
            if tracer_module_name == 'phosphorus':
                tracer_ind0 = self._tracer_names.index('po4')
                self._comp_tend_phosphorus(
                    time, tracer_vals[tracer_ind0:tracer_ind0+6, :],
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

        # po4 half-saturation = 0.5
        po4 = tracer_vals[0, :]
        po4_lim = np.where(po4 > 0.0, po4 / (po4 + 0.5), 0.0)
        po4_uptake = self.light_lim * po4_lim

        self._comp_tend_phosphorus_core(
            time, po4_uptake, tracer_vals[0:3, :], dtracer_vals_dt[0:3, :])
        self._comp_tend_phosphorus_core(
            time, po4_uptake, tracer_vals[3:6, :], dtracer_vals_dt[3:6, :])

        # restore po4_s to po4, at a rate of 1 / day
        # compensate equally from and dop and pop,
        # so that total shadow phosphorus is conserved
        rest_term = 1.0 * (tracer_vals[0, 0] - tracer_vals[3, 0])
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

    def _def_hist_dims(self, fptr):
        """define netCDF4 dimensions relevant to test_problem"""
        fptr.createDimension('time', None)
        fptr.createDimension('depth', self.depth.axis.nlevs)
        fptr.createDimension('depth_edges', 1+self.depth.axis.nlevs)

    def _def_hist_coord_vars(self, fptr):
        """define netCDF4 coordinate vars relevant to test_problem"""
        fptr.createVariable('time', 'f8', dimensions=('time',))
        fptr.variables['time'].long_name = 'time'
        fptr.variables['time'].units = 'days since 0001-01-01'

        fptr.createVariable('depth', 'f8', dimensions=('depth',))
        fptr.variables['depth'].long_name = 'depth'
        fptr.variables['depth'].units = 'm'

        fptr.createVariable('depth_edges', 'f8', dimensions=('depth_edges',))
        fptr.variables['depth_edges'].long_name = 'depth_edges'
        fptr.variables['depth_edges'].units = 'm'

    def _write_hist_coord_vars(self, fptr, sol):
        """write netCDF4 coordinate vars relevant to test_problem"""
        fptr.variables['time'][:] = sol.t
        fptr.variables['depth'][:] = self.depth.axis.mid
        fptr.variables['depth_edges'][:] = self.depth.axis.edges

    def _write_hist(self, sol, hist_fname):
        """write tracer values generated in comp_fcn to hist_fname"""
        with Dataset(hist_fname, mode='w') as fptr:
            self._def_hist_dims(fptr)
            self._def_hist_coord_vars(fptr)

            for tracer_name in self._tracer_names:
                fptr.createVariable(tracer_name, 'f8', dimensions=('time', 'depth'))

            fptr.createVariable('bldepth', 'f8', dimensions=('time'))
            fptr.variables['bldepth'].long_name = 'boundary layer depth'
            fptr.variables['bldepth'].units = 'm'

            fptr.createVariable('mixing_coeff', 'f8', dimensions=('time', 'depth_edges'))
            fptr.variables['mixing_coeff'].long_name = 'vertical mixing coefficient'
            fptr.variables['mixing_coeff'].units = 'm2 s-1'

            self._write_hist_coord_vars(fptr, sol)

            tracer_vals = sol.y.reshape(
                (len(self._tracer_names), self.depth.axis.nlevs, -1))
            for tracer_ind, tracer_name in enumerate(self._tracer_names):
                fptr.variables[tracer_name][:] = tracer_vals[tracer_ind, :, :].transpose()

            for time_ind, time in enumerate(sol.t):
                fptr.variables['bldepth'][time_ind] = self.depth.bldepth(time)
                fptr.variables['mixing_coeff'][time_ind, :] = \
                    (1.0 / 86400.0) * self.depth.mixing_coeff(time)

    def apply_precond_jacobian(self, ms_in, precond_fname, res_fname, solver_state):
        """apply preconditioner of jacobian of comp_fcn to model state object, ms_in"""
        logger = logging.getLogger(__name__)
        logger.debug('precond_fname="%s", res_fname="%s"', precond_fname, res_fname)

        fcn_complete_step = 'apply_precond_jacobian complete for %s' % res_fname
        if solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return ModelState(res_fname)
        logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        ms_res = ms_in.copy()

        with Dataset(precond_fname, 'r') as fptr:
            # hist and precond files have mixing_coeff in m2 s-1
            # convert back to model units of m2 d-1
            mca = 86400.0 * fptr.variables['mixing_coeff_log_avg'][:]

        for tracer_module_name in ms_in.tracer_module_names:
            if tracer_module_name == 'iage_test':
                self._apply_precond_jacobian_iage_test(ms_in, mca, ms_res)
            if tracer_module_name == 'phosphorus':
                self._apply_precond_jacobian_phosphorus(ms_in, mca, ms_res)

        solver_state.log_step(fcn_complete_step)

        return ms_res.dump(res_fname)

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

        matrix = diags(
            [self._diag_0_phosphorus(mca), self._diag_p_1_phosphorus(mca),
             self._diag_m_1_phosphorus(mca), self._diag_p_nz_phosphorus(),
             self._diag_m_nz_phosphorus(), self._diag_p_2nz_phosphorus(),
             self._diag_m_2nz_phosphorus()],
            [0, 1, -1, nz, -nz, 2*nz, -2*nz], format='csr')

        matrix_adj = matrix - 1.0e-8 * eye(3*nz)
        res = spsolve(matrix_adj, rhs)

        _, sing_vals, r_sing_vects = svd(matrix.todense())
        min_ind = sing_vals.argmin()
        dz3 = np.concatenate(
            (self.depth.axis.delta, self.depth.axis.delta, self.depth.axis.delta))
        numer = (res * dz3).sum()
        denom = (r_sing_vects[min_ind, :] * dz3).sum()
        res -= numer / denom * r_sing_vects[min_ind, :]

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
        # pop_s sinking loss to layer below
        diag_0_pop_s[:-1] -= 1.0 * self.depth.axis.delta_r[:-1]
        return np.concatenate((diag_0_po4_s, diag_0_dop_s, diag_0_pop_s))

    def _diag_p_1_phosphorus(self, mca):
        """return +1 upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_p_1_single_tracer = mca[1:-1] * self.depth.axis.delta_mid_r \
            * self.depth.axis.delta_r[:-1]
        diag_p_1_po4_s = diag_p_1_single_tracer.copy()
        zero = np.zeros(1)
        diag_p_1_dop_s = diag_p_1_single_tracer.copy()
        diag_p_1_pop_s = diag_p_1_single_tracer.copy()
        return np.concatenate(
            (diag_p_1_po4_s, zero, diag_p_1_dop_s, zero, diag_p_1_pop_s))

    def _diag_m_1_phosphorus(self, mca):
        """return +1 upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_m_1_single_tracer = mca[1:-1] * self.depth.axis.delta_mid_r \
            * self.depth.axis.delta_r[1:]
        diag_m_1_po4_s = diag_m_1_single_tracer.copy()
        zero = np.zeros(1)
        diag_m_1_dop_s = diag_m_1_single_tracer.copy()
        diag_m_1_pop_s = diag_m_1_single_tracer.copy()
        # pop_s sinking gain from layer above
        diag_m_1_pop_s += 1.0 * self.depth.axis.delta_r[1:]
        return np.concatenate(
            (diag_m_1_po4_s, zero, diag_m_1_dop_s, zero, diag_m_1_pop_s))

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

    def mixing_coeff(self, time):
        """
        vertical mixing coefficient, m2 d-1
        store computed vals, so their computation can be skipped on subsequent calls
        """

        # if vals have already been computed for this time, skip computation
        if time == self._time_val:
            return self._mixing_coeff_vals

        # z_lin ranges from 0.0 to 1.0 over span of 50.0 m, is 0.5 at bldepth
        z_lin = 0.5 + (self.axis.edges - self.bldepth(time)) * (1.0 / 50.0)
        z_lin = np.maximum(0.0, np.minimum(1.0, z_lin))
        res_log10_shallow = 0.0
        res_log10_deep = -5.0
        res_log10_del = res_log10_deep - res_log10_shallow
        res_log10 = res_log10_shallow + res_log10_del * z_lin
        self._time_val = time
        self._mixing_coeff_vals = 86400.0 * 10.0 ** res_log10
        return self._mixing_coeff_vals

    def bldepth(self, time):
        """time varying boundary layer depth"""
        bldepth_min = 50.0
        bldepth_max = 150.0
        bldepth_del_frac = 0.5 + 0.5 * np.cos((2 * np.pi) * ((time / 365.0) - 0.25))
        return bldepth_min + (bldepth_max - bldepth_min) * bldepth_del_frac

    def _z_lin_avg(self, k, bldepth):
        """
        average of max(0, min(1, 0.5 + (z - bldepth) * (1.0 / dz_trans)))
        over interval containing self.axis.edges[k]
        this function ranges from 0.0 to 1.0 over span of dz_trans m, is 0.5 at bldepth
        """
        dz_trans = 50.0
        z_trans0 = bldepth - 0.5 * dz_trans
        z_trans1 = bldepth + 0.5 * dz_trans
        z0 = self.axis.mid[k-1] if k > 0 else self.axis.edges[0]
        z1 = self.axis.mid[k] if k < self.axis.nlevs else self.axis.edges[-1]
        if z1 <= z_trans0:
            return 0.0
        if z1 <= z_trans1:
            if z0 <= z_trans0:
                zm = 0.5 * (z1 + z_trans0)
                val = 0.5 + (zm - bldepth) * (1.0 / dz_trans)
                return val * (z1 - z_trans0) / (z1 - z0)
            else:
                zm = 0.5 * (z1 + z0)
                return 0.5 + (zm - bldepth) * (1.0 / dz_trans)
        # z1 > z_trans1
        if z0 <= z_trans0:
            zm = bldepth
            val = 0.5 + (zm - bldepth) * (1.0 / dz_trans)
            return (val * dz_trans + 1.0 * (z1 - z_trans1)) / (z1 - z0)
        if z0 <= z_trans1:
            zm = 0.5 * (z_trans1 + z0)
            val = 0.5 + (zm - bldepth) * (1.0 / dz_trans)
            return (val * (z_trans1 - z0) + 1.0 * (z1 - z_trans1)) / (z1 - z0)
        return 1.0

################################################################################

if __name__ == '__main__':
    main(_parse_args())
