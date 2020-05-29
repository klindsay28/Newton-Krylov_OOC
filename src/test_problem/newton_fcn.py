#!/usr/bin/env python
"""test_problem hooks for Newton-Krylov solver"""

import copy
from distutils.util import strtobool
import logging
import os
import subprocess
import sys

from netCDF4 import Dataset
import numpy as np
from scipy.linalg import solve_banded, svd
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp

from test_problem.src.spatial_axis import SpatialAxis
from test_problem.src.hist import hist_write
from test_problem.src.vert_mix import VertMix

from ..model import ModelStateBase, TracerModuleStateBase
from ..model_config import ModelConfig, get_modelinfo
from ..newton_fcn_base import NewtonFcnBase
from ..share import args_replace, common_args, read_cfg_file


def _parse_args():
    """parse command line arguments"""
    parser = common_args("test problem for Newton-Krylov solver")
    parser.add_argument(
        "cmd",
        choices=["comp_fcn", "gen_precond_jacobian", "apply_precond_jacobian",],
        help="command to run",
    )
    parser.add_argument(
        "--fname_dir",
        help="directory that relative fname arguments are relative to",
        default=".",
    )
    parser.add_argument("--hist_fname", help="name of history file", default=None)
    parser.add_argument("--precond_fname", help="name of precond file", default=None)
    parser.add_argument("--in_fname", help="name of file with input")
    parser.add_argument("--res_fname", help="name of file for result")

    return args_replace(parser.parse_args(), model_name="test_problem")


def _resolve_fname(fname_dir, fname):
    """prepend fname_dir to fname, if fname is a relative path"""
    if fname is None or os.path.isabs(fname):
        return fname
    return os.path.join(fname_dir, fname)


def main(args):
    """test problem for Newton-Krylov solver"""

    config = read_cfg_file(args)
    solverinfo = config["solverinfo"]

    logging_format = "%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s"
    logging.basicConfig(
        stream=sys.stdout, format=logging_format, level=solverinfo["logging_level"]
    )
    logger = logging.getLogger(__name__)

    logger.info('args.cmd="%s"', args.cmd)

    # store cfg_fname in modelinfo, to ease access to its value elsewhere
    config["modelinfo"]["cfg_fname"] = args.cfg_fname

    ModelConfig(config["modelinfo"])

    newton_fcn = NewtonFcn()

    ms_in = ModelState(TracerModuleState, _resolve_fname(args.fname_dir, args.in_fname))
    if args.cmd == "comp_fcn":
        ms_in.log("state_in")
        newton_fcn.comp_fcn(
            ms_in,
            _resolve_fname(args.fname_dir, args.res_fname),
            solver_state=None,
            hist_fname=_resolve_fname(args.fname_dir, args.hist_fname),
        )
        ModelState(
            TracerModuleState, _resolve_fname(args.fname_dir, args.res_fname)
        ).log("fcn")
    elif args.cmd == "gen_precond_jacobian":
        newton_fcn.gen_precond_jacobian(
            ms_in,
            _resolve_fname(args.fname_dir, args.hist_fname),
            _resolve_fname(args.fname_dir, args.precond_fname),
        )
    elif args.cmd == "apply_precond_jacobian":
        ms_in.log("state_in")
        newton_fcn.apply_precond_jacobian(
            ms_in,
            _resolve_fname(args.fname_dir, args.precond_fname),
            _resolve_fname(args.fname_dir, args.res_fname),
            solver_state=None,
        )
        ModelState(
            TracerModuleState, _resolve_fname(args.fname_dir, args.res_fname)
        ).log("precond_res")
    else:
        msg = "unknown cmd=%s" % args.cmd
        raise ValueError(msg)

    logger.info("done")


################################################################################


class ModelState(ModelStateBase):
    """class for representing the state space of a model"""

    # give ModelState operators higher priority than those of numpy
    __array_priority__ = 100

    def __init__(self, tracer_module_state_class, fname):
        logger = logging.getLogger(__name__)
        logger.debug('ModelState, fname="%s"', fname)
        super().__init__(tracer_module_state_class, fname)

    def tracer_dims_keep_in_stats(self):
        """tuple of dimensions to keep for tracers in stats file"""
        return ("depth",)

    def hist_time_mean_weights(self, fptr_hist):
        """return weights for computing time-mean in hist file"""
        # downweight endpoints because test_problem writes t=0 and t=365 to hist
        timelen = len(fptr_hist.dimensions["time"])
        weights = np.full(timelen, 1.0 / (timelen - 1))
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return weights


################################################################################


class TracerModuleState(TracerModuleStateBase):
    """
    Derived class for representing a collection of model tracers.
    It implements _read_vals and dump.
    """

    def _read_vals(self, tracer_module_name, fname):
        """return tracer values and dimension names and lengths, read from fname)"""
        logger = logging.getLogger(__name__)
        logger.debug(
            'tracer_module_name="%s", fname="%s"', tracer_module_name, fname,
        )
        if fname == "gen_init_iterate":
            depth = SpatialAxis("depth", get_modelinfo("depth_fname"))
            vals = np.empty((len(self._tracer_module_def), depth.nlevs))
            for tracer_ind, tracer_metadata in enumerate(
                self._tracer_module_def.values()
            ):
                if "init_iterate_vals" in tracer_metadata:
                    vals[tracer_ind, :] = np.interp(
                        depth.mid,
                        tracer_metadata["init_iterate_val_depths"],
                        tracer_metadata["init_iterate_vals"],
                    )
                elif "shadows" in tracer_metadata:
                    shadowed_tracer = tracer_metadata["shadows"]
                    shadow_tracer_metadata = self._tracer_module_def[shadowed_tracer]
                    vals[tracer_ind, :] = np.interp(
                        depth.mid,
                        shadow_tracer_metadata["init_iterate_val_depths"],
                        shadow_tracer_metadata["init_iterate_vals"],
                    )
                else:
                    msg = (
                        "gen_init_iterate failure for %s"
                        % self.tracer_names()[tracer_ind]
                    )
                    raise ValueError(msg)
            return vals, {"depth": depth.nlevs}
        dims = {}
        with Dataset(fname, mode="r") as fptr:
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
                    msg = (
                        "not all vars have same dimensions"
                        ", tracer_module_name=%s, fname=%s"
                        % (tracer_module_name, fname)
                    )
                    raise ValueError(msg)
            # read values
            if len(dims) > 3:
                msg = (
                    "ndim too large (for implementation of dot_prod)"
                    "tracer_module_name=%s, fname=%s, ndim=%s"
                    % (tracer_module_name, fname, len(dims))
                )
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
        if action == "define":
            for dimname, dimlen in self._dims.items():
                try:
                    if fptr.dimensions[dimname].size != dimlen:
                        msg = (
                            "dimname already exists and has wrong size"
                            "tracer_module_name=%s, dimname=%s"
                            % (self._tracer_module_name, dimname)
                        )
                        raise ValueError(msg)
                except KeyError:
                    fptr.createDimension(dimname, dimlen)
            dimnames = tuple(self._dims.keys())
            # define all tracers
            for tracer_name in self.tracer_names():
                fptr.createVariable(tracer_name, "f8", dimensions=dimnames)
        elif action == "write":
            # write all tracers
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                fptr.variables[tracer_name][:] = self._vals[tracer_ind, :]
        else:
            msg = "unknown action=%s", action
            raise ValueError(msg)
        return self


################################################################################


class NewtonFcn(NewtonFcnBase):
    """class of methods related to problem being solved with Newton's method"""

    def __init__(self):
        self.time_range = (0.0, 365.0)
        self.depth = SpatialAxis("depth", get_modelinfo("depth_fname"))

        self.vert_mix = VertMix(self.depth)

        # light has e-folding decay of 25m
        self.light_lim = np.exp((-1.0 / 25.0) * self.depth.mid)

        self._sinking_tend_work = np.zeros(1 + self.depth.nlevs)

        # integral of surface flux over year is 1 mol m-2
        self._dye_sink_surf_flux_times = 365.0 * np.array([0.1, 0.2, 0.6, 0.7])
        self._dye_sink_surf_flux_vals = np.array([0.0, 2.0, 2.0, 0.0]) / 365.0
        self._dye_sink_surf_flux_time = None
        self._dye_sink_surf_flux_val = 0.0

        # tracer_module_names and tracer_names will be stored in the following attributes,
        # enabling access to them from inside _comp_tend
        self._tracer_module_names = None
        self._tracer_names = None

    def model_state_obj(self, fname):
        """return a ModelState object compatible with this function"""
        return ModelState(TracerModuleState, fname)

    def comp_fcn(self, ms_in, res_fname, solver_state, hist_fname=None):
        """evalute function being solved with Newton's method"""
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s", hist_fname="%s"', res_fname, hist_fname)

        if solver_state is not None:
            fcn_complete_step = "comp_fcn complete for %s" % res_fname
            if solver_state.step_logged(fcn_complete_step):
                logger.debug('"%s" logged, returning result', fcn_complete_step)
                return ModelState(TracerModuleState, res_fname)
            logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        self._tracer_module_names = ms_in.tracer_module_names
        self._tracer_names = ms_in.tracer_names()
        tracer_vals_init = np.empty((len(self._tracer_names), self.depth.nlevs))
        for tracer_ind, tracer_name in enumerate(self._tracer_names):
            tracer_vals_init[tracer_ind, :] = ms_in.get_tracer_vals(tracer_name)

        # solve ODEs, using scipy.integrate
        # get dense output, if requested
        if hist_fname is not None:
            t_eval = np.linspace(self.time_range[0], self.time_range[1], 101)
        else:
            t_eval = np.array(self.time_range)
        sol = solve_ivp(
            self._comp_tend,
            self.time_range,
            tracer_vals_init.reshape(-1),
            "Radau",
            t_eval,
            atol=1.0e-10,
            rtol=1.0e-10,
            args=(self.vert_mix,),
        )

        if hist_fname is not None:
            hist_write(ms_in, sol, hist_fname, self)

        ms_res = copy.deepcopy(ms_in)
        res_vals = sol.y[:, -1].reshape(tracer_vals_init.shape) - tracer_vals_init
        for tracer_ind, tracer_name in enumerate(self._tracer_names):
            ms_res.set_tracer_vals(tracer_name, res_vals[tracer_ind, :])

        caller = __name__ + ".NewtonFcn.comp_fcn"
        self.comp_fcn_postprocess(ms_res, res_fname, caller)

        if solver_state is not None:
            solver_state.log_step(fcn_complete_step)
            if strtobool(get_modelinfo("reinvoke")):
                cmd = [get_modelinfo("invoker_script_fname"), "--resume"]
                logger.info('cmd="%s"', " ".join(cmd))
                # use Popen instead of run because we don't want to wait
                subprocess.Popen(cmd)
                raise SystemExit

        return ms_res

    def _comp_tend(self, time, tracer_vals_flat, vert_mix):
        """compute tendency function"""
        tracer_vals = tracer_vals_flat.reshape((len(self._tracer_names), -1))
        dtracer_vals_dt = np.empty_like(tracer_vals)
        for tracer_module_name in self._tracer_module_names:
            if tracer_module_name == "iage":
                tracer_ind = self._tracer_names.index("iage")
                self._comp_tend_iage(
                    time,
                    tracer_vals[tracer_ind, :],
                    dtracer_vals_dt[tracer_ind, :],
                    vert_mix,
                )
            elif tracer_module_name[:9] == "dye_sink_":
                tracer_ind = self._tracer_names.index(tracer_module_name)
                self._comp_tend_dye_sink(
                    tracer_module_name[9:],
                    time,
                    tracer_vals[tracer_ind, :],
                    dtracer_vals_dt[tracer_ind, :],
                    vert_mix,
                )
            elif tracer_module_name == "phosphorus":
                tracer_ind0 = self._tracer_names.index("po4")
                self._comp_tend_phosphorus(
                    time,
                    tracer_vals[tracer_ind0 : tracer_ind0 + 6, :],
                    dtracer_vals_dt[tracer_ind0 : tracer_ind0 + 6, :],
                    vert_mix,
                )
            else:
                msg = "unknown tracer module %s" % tracer_module_name
                raise ValueError(msg)
        return dtracer_vals_dt.reshape(-1)

    def _comp_tend_iage(self, time, tracer_vals, dtracer_vals_dt, vert_mix):
        """
        compute tendency for iage
        tendency units are tr_units / day
        """
        # surface_flux piston velocity = 240 m / day
        # same as restoring 24 / day over 10 m
        surf_flux = -240.0 * tracer_vals[0]
        dtracer_vals_dt[:] = vert_mix.tend(time, tracer_vals, surf_flux)
        # age 1/year
        dtracer_vals_dt[:] += 1.0 / 365.0

    def _comp_tend_dye_sink(self, suff, time, tracer_vals, dtracer_vals_dt, vert_mix):
        """
        compute tendency for dye_sink tracer
        tendency units are tr_units / day
        """
        surf_flux = self._dye_sink_surf_flux(time)
        dtracer_vals_dt[:] = vert_mix.tend(time, tracer_vals, surf_flux)
        # decay (suff / 1000) / y
        dtracer_vals_dt[:] -= int(suff) * 0.001 / 365.0 * tracer_vals

    def _dye_sink_surf_flux(self, time):
        """return surf flux applied to dye_sink tracers"""
        if time != self._dye_sink_surf_flux_time:
            self._dye_sink_surf_flux_val = np.interp(
                time, self._dye_sink_surf_flux_times, self._dye_sink_surf_flux_vals
            )
            time = self._dye_sink_surf_flux_time
        return self._dye_sink_surf_flux_val

    def _comp_tend_phosphorus(self, time, tracer_vals, dtracer_vals_dt, vert_mix):
        """
        compute tendency for phosphorus tracers
        tendency units are tr_units / day
        """

        po4_uptake = self.po4_uptake(tracer_vals[0, :])

        self._comp_tend_phosphorus_core(
            time, po4_uptake, tracer_vals[0:3, :], dtracer_vals_dt[0:3, :], vert_mix
        )
        self._comp_tend_phosphorus_core(
            time, po4_uptake, tracer_vals[3:6, :], dtracer_vals_dt[3:6, :], vert_mix
        )

        # restore po4_s to po4, at a rate of 1 / day
        # compensate equally from and dop and pop,
        # so that total shadow phosphorus is conserved
        rest_term = 1.0 * (tracer_vals[0, 0] - tracer_vals[3, 0])
        dtracer_vals_dt[3, 0] += rest_term
        dtracer_vals_dt[4, 0] -= 0.67 * rest_term
        dtracer_vals_dt[5, 0] -= 0.33 * rest_term

    def po4_uptake(self, po4):
        """return po4_uptake, [mmol m-3 d-1]"""

        # po4 half-saturation = 0.5
        # maximum uptake rate = 1 d-1
        po4_lim = po4 / (po4 + 0.5)
        return self.light_lim * po4_lim

    def _comp_tend_phosphorus_core(
        self, time, po4_uptake, tracer_vals, dtracer_vals_dt, vert_mix
    ):
        """
        core fuction for computing tendency for phosphorus tracers
        tendency units are tr_units / day
        """

        po4 = tracer_vals[0, :]
        dop = tracer_vals[1, :]
        pop = tracer_vals[2, :]

        # dop remin rate is 1% / day
        dop_remin = 0.01 * dop
        # pop remin rate is 1% / day
        pop_remin = 0.01 * pop

        sigma = 0.67

        dtracer_vals_dt[0, :] = (
            -po4_uptake + dop_remin + pop_remin + vert_mix.tend(time, po4)
        )
        dtracer_vals_dt[1, :] = (
            sigma * po4_uptake - dop_remin + vert_mix.tend(time, dop)
        )
        dtracer_vals_dt[2, :] = (
            (1.0 - sigma) * po4_uptake
            - pop_remin
            + vert_mix.tend(time, pop)
            + self._sinking_tend(pop)
        )

    def _sinking_tend(self, tracer_vals):
        """tracer tendency from sinking"""
        self._sinking_tend_work[1:-1] = -tracer_vals[
            :-1
        ]  # assume velocity is 1 m / day
        return (
            self._sinking_tend_work[1:] - self._sinking_tend_work[:-1]
        ) * self.depth.delta_r

    def apply_precond_jacobian(self, ms_in, precond_fname, res_fname, solver_state):
        """apply preconditioner of jacobian of comp_fcn to model state object, ms_in"""
        logger = logging.getLogger(__name__)
        logger.debug('precond_fname="%s", res_fname="%s"', precond_fname, res_fname)

        if solver_state is not None:
            fcn_complete_step = "apply_precond_jacobian complete for %s" % res_fname
            if solver_state.step_logged(fcn_complete_step):
                logger.debug('"%s" logged, returning result', fcn_complete_step)
                return ModelState(TracerModuleState, res_fname)
            logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        ms_res = copy.deepcopy(ms_in)

        with Dataset(precond_fname, mode="r") as fptr:
            # hist, and thus precond, files have mixing_coeff in m2 s-1
            # convert back to model units of m2 d-1
            mca = 86400.0 * fptr.variables["mixing_coeff_log_avg"][1:-1]

        for tracer_module_name in ms_in.tracer_module_names:
            if tracer_module_name == "iage":
                self._apply_precond_jacobian_iage(ms_in, mca, ms_res)
            elif tracer_module_name[:9] == "dye_sink_":
                self._apply_precond_jacobian_dye_sink(
                    tracer_module_name, ms_in, mca, ms_res
                )
            elif tracer_module_name == "phosphorus":
                self._apply_precond_jacobian_phosphorus(ms_in, mca, ms_res)
            else:
                msg = "unknown tracer module %s" % tracer_module_name
                raise ValueError(msg)

        if solver_state is not None:
            solver_state.log_step(fcn_complete_step)

        caller = __name__ + ".NewtonFcn.apply_precond_jacobian"
        return ms_res.dump(res_fname, caller)

    def _apply_precond_jacobian_iage(self, ms_in, mca, ms_res):
        """apply preconditioner of jacobian of iage fcn"""

        iage_in = ms_in.get_tracer_vals("iage")
        rhs = (1.0 / (self.time_range[1] - self.time_range[0])) * iage_in

        l_and_u = (1, 1)
        matrix_diagonals = np.zeros((3, self.depth.nlevs))
        matrix_diagonals[0, 1:] = mca * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        matrix_diagonals[1, :-1] -= (
            mca * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        matrix_diagonals[1, 1:] -= mca * self.depth.delta_mid_r * self.depth.delta_r[1:]
        matrix_diagonals[2, :-1] = mca * self.depth.delta_mid_r * self.depth.delta_r[1:]
        matrix_diagonals[1, 0] = -240.0 * self.depth.delta_r[0]
        matrix_diagonals[0, 1] = 0

        res = solve_banded(l_and_u, matrix_diagonals, rhs)

        ms_res.set_tracer_vals("iage", res - iage_in)

    def _apply_precond_jacobian_dye_sink(self, name, ms_in, mca, ms_res):
        """apply preconditioner of jacobian of dye_sink fcn"""

        dye_sink_in = ms_in.get_tracer_vals(name)
        rhs = (1.0 / (self.time_range[1] - self.time_range[0])) * dye_sink_in

        l_and_u = (1, 1)
        matrix_diagonals = np.zeros((3, self.depth.nlevs))
        matrix_diagonals[0, 1:] = mca * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        matrix_diagonals[1, :-1] -= (
            mca * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        matrix_diagonals[1, 1:] -= mca * self.depth.delta_mid_r * self.depth.delta_r[1:]
        matrix_diagonals[2, :-1] = mca * self.depth.delta_mid_r * self.depth.delta_r[1:]
        matrix_diagonals[0, 1] = 0

        # decay (suff / 1000) / y
        suff = name[9:]
        matrix_diagonals[1, :] -= int(suff) * 0.001 / 365.0

        res = solve_banded(l_and_u, matrix_diagonals, rhs)

        ms_res.set_tracer_vals(name, res - dye_sink_in)

    def _apply_precond_jacobian_phosphorus(self, ms_in, mca, ms_res):
        """
        apply preconditioner of jacobian of phosphorus fcn
        it is only applied to shadow phosphorus tracers
        """

        po4_s = ms_in.get_tracer_vals("po4_s")
        dop_s = ms_in.get_tracer_vals("dop_s")
        pop_s = ms_in.get_tracer_vals("pop_s")
        rhs = (1.0 / (self.time_range[1] - self.time_range[0])) * np.concatenate(
            (po4_s, dop_s, pop_s)
        )

        nlevs = self.depth.nlevs

        matrix = diags(
            [
                self._diag_0_phosphorus(mca),
                self._diag_p_1_phosphorus(mca),
                self._diag_m_1_phosphorus(mca),
                self._diag_p_nlevs_phosphorus(),
                self._diag_m_nlevs_phosphorus(),
                self._diag_p_2nlevs_phosphorus(),
                self._diag_m_2nlevs_phosphorus(),
            ],
            [0, 1, -1, nlevs, -nlevs, 2 * nlevs, -2 * nlevs],
            format="csr",
        )

        matrix_adj = matrix - 1.0e-8 * eye(3 * nlevs)
        res = spsolve(matrix_adj, rhs)

        _, sing_vals, r_sing_vects = svd(matrix.todense())
        min_ind = sing_vals.argmin()
        dz3 = np.concatenate((self.depth.delta, self.depth.delta, self.depth.delta))
        numer = (res * dz3).sum()
        denom = (r_sing_vects[min_ind, :] * dz3).sum()
        res -= numer / denom * r_sing_vects[min_ind, :]

        ms_res.set_tracer_vals("po4_s", res[0:nlevs] - po4_s)
        ms_res.set_tracer_vals("dop_s", res[nlevs : 2 * nlevs] - dop_s)
        ms_res.set_tracer_vals("pop_s", res[2 * nlevs : 3 * nlevs] - pop_s)

    def _diag_0_phosphorus(self, mca):
        """return main diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_0_single_tracer = np.zeros(self.depth.nlevs)
        diag_0_single_tracer[:-1] -= (
            mca * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        diag_0_single_tracer[1:] -= (
            mca * self.depth.delta_mid_r * self.depth.delta_r[1:]
        )
        diag_0_po4_s = diag_0_single_tracer.copy()
        diag_0_po4_s[0] -= 1.0  # po4_s restoring in top layer
        diag_0_dop_s = diag_0_single_tracer.copy()
        diag_0_dop_s -= 0.01  # dop_s remin
        diag_0_pop_s = diag_0_single_tracer.copy()
        diag_0_pop_s -= 0.01  # pop_s remin
        # pop_s sinking loss to layer below
        diag_0_pop_s[:-1] -= 1.0 * self.depth.delta_r[:-1]
        return np.concatenate((diag_0_po4_s, diag_0_dop_s, diag_0_pop_s))

    def _diag_p_1_phosphorus(self, mca):
        """return +1 upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_p_1_single_tracer = mca * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        diag_p_1_po4_s = diag_p_1_single_tracer.copy()
        zero = np.zeros(1)
        diag_p_1_dop_s = diag_p_1_single_tracer.copy()
        diag_p_1_pop_s = diag_p_1_single_tracer.copy()
        return np.concatenate(
            (diag_p_1_po4_s, zero, diag_p_1_dop_s, zero, diag_p_1_pop_s)
        )

    def _diag_m_1_phosphorus(self, mca):
        """return +1 upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_m_1_single_tracer = mca * self.depth.delta_mid_r * self.depth.delta_r[1:]
        diag_m_1_po4_s = diag_m_1_single_tracer.copy()
        zero = np.zeros(1)
        diag_m_1_dop_s = diag_m_1_single_tracer.copy()
        diag_m_1_pop_s = diag_m_1_single_tracer.copy()
        # pop_s sinking gain from layer above
        diag_m_1_pop_s += 1.0 * self.depth.delta_r[1:]
        return np.concatenate(
            (diag_m_1_po4_s, zero, diag_m_1_dop_s, zero, diag_m_1_pop_s)
        )

    def _diag_p_nlevs_phosphorus(self):
        """
        return +nlevs upper diagonal of preconditioner of jacobian of phosphorus fcn
        """
        diag_p_1_dop_po4 = 0.01 * np.ones(self.depth.nlevs)  # dop_s remin
        diag_p_1_pop_dop = np.zeros(self.depth.nlevs)
        return np.concatenate((diag_p_1_dop_po4, diag_p_1_pop_dop))

    def _diag_m_nlevs_phosphorus(self):
        """
        return -nlevs lower diagonal of preconditioner of jacobian of phosphorus fcn
        """
        diag_p_1_po4_dop = np.zeros(self.depth.nlevs)
        diag_p_1_po4_dop[0] = 0.67  # po4_s restoring conservation balance
        diag_p_1_dop_pop = np.zeros(self.depth.nlevs)
        return np.concatenate((diag_p_1_po4_dop, diag_p_1_dop_pop))

    def _diag_p_2nlevs_phosphorus(self):
        """
        return +2nlevs upper diagonal of preconditioner of jacobian of phosphorus fcn
        """
        return 0.01 * np.ones(self.depth.nlevs)  # pop_s remin

    def _diag_m_2nlevs_phosphorus(self):
        """
        return -2nlevs lower diagonal of preconditioner of jacobian of phosphorus fcn
        """
        diag_p_1_po4_pop = np.zeros(self.depth.nlevs)
        diag_p_1_po4_pop[0] = 0.33  # po4_s restoring conservation balance
        return diag_p_1_po4_pop


################################################################################

if __name__ == "__main__":
    main(_parse_args())
