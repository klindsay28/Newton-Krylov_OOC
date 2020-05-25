#!/usr/bin/env python
"""test_problem hooks for Newton-Krylov solver"""

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

    return args_replace(parser.parse_args())


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

    ms_in = ModelState(_resolve_fname(args.fname_dir, args.in_fname))
    if args.cmd == "comp_fcn":
        ms_in.log("state_in")
        newton_fcn.comp_fcn(
            ms_in,
            _resolve_fname(args.fname_dir, args.res_fname),
            solver_state=None,
            hist_fname=_resolve_fname(args.fname_dir, args.hist_fname),
        )
        ModelState(_resolve_fname(args.fname_dir, args.res_fname)).log("fcn")
    elif args.cmd == "gen_precond_jacobian":
        newton_fcn.gen_precond_jacobian(
            ms_in,
            _resolve_fname(args.fname_dir, args.hist_fname),
            _resolve_fname(args.fname_dir, args.precond_fname),
            solver_state=None,
        )
    elif args.cmd == "apply_precond_jacobian":
        ms_in.log("state_in")
        newton_fcn.apply_precond_jacobian(
            ms_in,
            _resolve_fname(args.fname_dir, args.precond_fname),
            _resolve_fname(args.fname_dir, args.res_fname),
            solver_state=None,
        )
        ModelState(_resolve_fname(args.fname_dir, args.res_fname)).log("precond_res")
    else:
        msg = "unknown cmd=%s" % args.cmd
        raise ValueError(msg)

    logger.info("done")


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
            'tracer_module_name="%s", vals_fname="%s"', tracer_module_name, vals_fname,
        )
        if vals_fname == "gen_ic":
            depth = SpatialAxis("depth", get_modelinfo("depth_fname"))
            vals = np.empty((len(self._tracer_module_def), depth.nlevs))
            for tracer_ind, tracer_metadata in enumerate(
                self._tracer_module_def.values()
            ):
                if "ic_vals" in tracer_metadata:
                    vals[tracer_ind, :] = np.interp(
                        depth.mid,
                        tracer_metadata["ic_val_depths"],
                        tracer_metadata["ic_vals"],
                    )
                elif "shadows" in tracer_metadata:
                    shadowed_tracer = tracer_metadata["shadows"]
                    shadow_tracer_metadata = self._tracer_module_def[shadowed_tracer]
                    vals[tracer_ind, :] = np.interp(
                        depth.mid,
                        shadow_tracer_metadata["ic_val_depths"],
                        shadow_tracer_metadata["ic_vals"],
                    )
                else:
                    msg = "gen_ic failure for %s" % self.tracer_names()[tracer_ind]
                    raise ValueError(msg)
            return vals, {"depth": depth.nlevs}
        dims = {}
        with Dataset(vals_fname, mode="r") as fptr:
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
                        ", tracer_module_name=%s, vals_fname=%s"
                        % (tracer_module_name, vals_fname)
                    )
                    raise ValueError(msg)
            # read values
            if len(dims) > 3:
                msg = (
                    "ndim too large (for implementation of dot_prod)"
                    "tracer_module_name=%s, vals_fname=%s, ndim=%s"
                    % (tracer_module_name, vals_fname, len(dims))
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

        # light has e-folding decay of 25m
        self.light_lim = np.exp((-1.0 / 25.0) * self.depth.mid)

        self._time_val = None
        self._mixing_coeff_vals = np.empty(self.depth.nlevs)

        self._dye_sink_surf_val_times = 365.0 * np.array([0.1, 0.2, 0.6, 0.7])
        self._dye_sink_surf_val_vals = np.array([0.0, 1.0, 1.0, 0.0])
        self._dye_sink_surf_val_time = None
        self._dye_sink_surf_val_val = 0.0

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
            fcn_complete_step = "comp_fcn complete for %s" % res_fname
            if solver_state.step_logged(fcn_complete_step):
                logger.debug('"%s" logged, returning result', fcn_complete_step)
                return ModelState(res_fname)
            logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        self._tracer_module_names = ms_in.tracer_module_names
        self._tracer_names = ms_in.tracer_names()
        tracer_vals_init = np.empty((len(self._tracer_names), self.depth.nlevs))
        for tracer_ind, tracer_name in enumerate(self._tracer_names):
            tracer_vals_init[tracer_ind, :] = ms_in.get_tracer_vals(tracer_name)

        # solve ODEs, using scipy.integrate
        # get dense output, if requested
        t_eval = np.linspace(
            self.time_range[0], self.time_range[1], 101 if hist_fname is not None else 2
        )
        sol = solve_ivp(
            self._comp_tend,
            self.time_range,
            tracer_vals_init.reshape(-1),
            "Radau",
            t_eval,
            atol=1.0e-10,
            rtol=1.0e-10,
        )

        if hist_fname is not None:
            hist_write(ms_in, sol, hist_fname, self)

        ms_res = ms_in.copy()
        res_vals = sol.y[:, -1].reshape(tracer_vals_init.shape) - tracer_vals_init
        for tracer_ind, tracer_name in enumerate(self._tracer_names):
            ms_res.set_tracer_vals(tracer_name, res_vals[tracer_ind, :])

        self.comp_fcn_postprocess(ms_res, res_fname)

        if solver_state is not None:
            solver_state.log_step(fcn_complete_step)
            if strtobool(get_modelinfo("reinvoke")):
                cmd = [get_modelinfo("invoker_script_fname"), "--resume"]
                logger.info('cmd="%s"', " ".join(cmd))
                # use Popen instead of run because we don't want to wait
                subprocess.Popen(cmd)
                raise SystemExit

        return ms_res

    def _comp_tend(self, time, tracer_vals_flat):
        """compute tendency function"""
        tracer_vals = tracer_vals_flat.reshape((len(self._tracer_names), -1))
        dtracer_vals_dt = np.empty_like(tracer_vals)
        for tracer_module_name in self._tracer_module_names:
            if tracer_module_name == "iage":
                tracer_ind = self._tracer_names.index("iage")
                self._comp_tend_iage(
                    time, tracer_vals[tracer_ind, :], dtracer_vals_dt[tracer_ind, :]
                )
            elif tracer_module_name[:9] == "dye_sink_":
                tracer_ind = self._tracer_names.index(tracer_module_name)
                self._comp_tend_dye_sink(
                    tracer_module_name[9:],
                    time,
                    tracer_vals[tracer_ind, :],
                    dtracer_vals_dt[tracer_ind, :],
                )
            elif tracer_module_name == "phosphorus":
                tracer_ind0 = self._tracer_names.index("po4")
                self._comp_tend_phosphorus(
                    time,
                    tracer_vals[tracer_ind0 : tracer_ind0 + 6, :],
                    dtracer_vals_dt[tracer_ind0 : tracer_ind0 + 6, :],
                )
            else:
                msg = "unknown tracer module %s" % tracer_module_name
                raise ValueError(msg)
        return dtracer_vals_dt.reshape(-1)

    def _mixing_tend(self, time, tracer_vals):
        """single tracer tendency from mixing"""
        tracer_grad = self.depth.grad_vals_mid(tracer_vals)
        return self.depth.grad_vals_edges(self.mixing_coeff(time) * tracer_grad)

    def _mixing_tend_sf(self, time, tracer_vals, surf_flux):
        """single tracer tendency from mixing, with surface flux"""
        tracer_grad = self.depth.grad_vals_mid(tracer_vals)
        flux_neg = self.mixing_coeff(time) * tracer_grad
        flux_neg[0] = -surf_flux
        return self.depth.grad_vals_edges(flux_neg)

    def mixing_coeff(self, time):
        """
        vertical mixing coefficient, m2 d-1
        store computed vals, so their computation can be skipped on subsequent calls
        """

        # if vals have already been computed for this time, skip computation
        if time == self._time_val:
            return self._mixing_coeff_vals

        # z_lin ranges from 0.0 to 1.0 over span of 40.0 m, is 0.5 at bldepth
        z_lin = 0.5 + (self.depth.edges - self.bldepth(time)) * (1.0 / 40.0)
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
        average of max(0, min(1, 0.5 + (z - bldepth) * (1.0 / trans_range)))
        over interval containing self.depth.edges[k]
        this function ranges from 0.0 to 1.0 over span of trans_range m, is 0.5 at bldepth
        """
        trans_range = 50.0
        depth_trans_lo = bldepth - 0.5 * trans_range
        depth_trans_hi = bldepth + 0.5 * trans_range
        depth_lo = self.depth.mid[k - 1] if k > 0 else self.depth.edges[0]
        depth_hi = self.depth.mid[k] if k < self.depth.nlevs else self.depth.edges[-1]
        if depth_hi <= depth_trans_lo:
            return 0.0
        if depth_hi <= depth_trans_hi:
            if depth_lo <= depth_trans_lo:
                depth_mid = 0.5 * (depth_hi + depth_trans_lo)
                val = 0.5 + (depth_mid - bldepth) * (1.0 / trans_range)
                return val * (depth_hi - depth_trans_lo) / (depth_hi - depth_lo)
            # depth_lo > depth_trans_lo
            depth_mid = 0.5 * (depth_hi + depth_lo)
            return 0.5 + (depth_mid - bldepth) * (1.0 / trans_range)
        # depth_hi > depth_trans_hi
        if depth_lo <= depth_trans_lo:
            depth_mid = bldepth
            val = 0.5 + (depth_mid - bldepth) * (1.0 / trans_range)
            return (val * trans_range + 1.0 * (depth_hi - depth_trans_hi)) / (
                depth_hi - depth_lo
            )
        if depth_lo <= depth_trans_hi:
            depth_mid = 0.5 * (depth_trans_hi + depth_lo)
            val = 0.5 + (depth_mid - bldepth) * (1.0 / trans_range)
            return (
                val * (depth_trans_hi - depth_lo) + 1.0 * (depth_hi - depth_trans_hi)
            ) / (depth_hi - depth_lo)
        return 1.0

    def _comp_tend_iage(self, time, tracer_vals, dtracer_vals_dt):
        """
        compute tendency for iage
        tendency units are tr_units / day
        """
        # surface_flux piston velocity = 240 m / day
        # same as restoring 24 / day over 10 m
        surf_flux = -240.0 * tracer_vals[0]
        dtracer_vals_dt[:] = self._mixing_tend_sf(time, tracer_vals, surf_flux)
        # age 1/year
        dtracer_vals_dt[:] += 1.0 / 365.0

    def _comp_tend_dye_sink(self, suff, time, tracer_vals, dtracer_vals_dt):
        """
        compute tendency for dye_sink tracer
        tendency units are tr_units / day
        """
        # surface_flux piston velocity = 240 m / day
        # same as restoring 24 / day over 10 m
        target_surf_val = self._dye_sink_surf_val(time)
        surf_flux = 240.0 * (target_surf_val - tracer_vals[0])
        dtracer_vals_dt[:] = self._mixing_tend_sf(time, tracer_vals, surf_flux)
        # decay (suff / 1000) / y
        dtracer_vals_dt[:] -= int(suff) * 0.001 / 365.0 * tracer_vals

    def _dye_sink_surf_val(self, time):
        """return surf value that dye_sink tracers are restored to"""
        if time != self._dye_sink_surf_val_time:
            self._dye_sink_surf_val_val = np.interp(
                time, self._dye_sink_surf_val_times, self._dye_sink_surf_val_vals
            )
            time = self._dye_sink_surf_val_time
        return self._dye_sink_surf_val_val

    def _comp_tend_phosphorus(self, time, tracer_vals, dtracer_vals_dt):
        """
        compute tendency for phosphorus tracers
        tendency units are tr_units / day
        """

        po4_uptake = self.po4_uptake(time, tracer_vals[0, :])

        self._comp_tend_phosphorus_core(
            time, po4_uptake, tracer_vals[0:3, :], dtracer_vals_dt[0:3, :]
        )
        self._comp_tend_phosphorus_core(
            time, po4_uptake, tracer_vals[3:6, :], dtracer_vals_dt[3:6, :]
        )

        # restore po4_s to po4, at a rate of 1 / day
        # compensate equally from and dop and pop,
        # so that total shadow phosphorus is conserved
        rest_term = 1.0 * (tracer_vals[0, 0] - tracer_vals[3, 0])
        dtracer_vals_dt[3, 0] += rest_term
        dtracer_vals_dt[4, 0] -= 0.67 * rest_term
        dtracer_vals_dt[5, 0] -= 0.33 * rest_term

    def po4_uptake(self, time, po4):
        """return po4_uptake, [mmol m-3 d-1]"""

        # po4 half-saturation = 0.5
        # maximum uptake rate = 1 d-1
        po4_lim = np.where(po4 > 0.0, po4 / (po4 + 0.5), 0.0)
        return self.light_lim * po4_lim

    def _comp_tend_phosphorus_core(
        self, time, po4_uptake, tracer_vals, dtracer_vals_dt
    ):
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

        dtracer_vals_dt[0, :] = (
            -po4_uptake + dop_remin + pop_remin + self._mixing_tend(time, po4)
        )
        dtracer_vals_dt[1, :] = (
            sigma * po4_uptake - dop_remin + self._mixing_tend(time, dop)
        )
        dtracer_vals_dt[2, :] = (
            (1.0 - sigma) * po4_uptake
            - pop_remin
            + self._mixing_tend(time, pop)
            + self._sinking_tend(pop)
        )

    def _sinking_tend(self, tracer_vals):
        """tracer tendency from sinking"""
        tracer_flux_neg = np.zeros(1 + self.depth.nlevs)
        tracer_flux_neg[1:-1] = -tracer_vals[:-1]  # assume velocity is 1 m / day
        return self.depth.grad_vals_edges(tracer_flux_neg)

    def apply_precond_jacobian(self, ms_in, precond_fname, res_fname, solver_state):
        """apply preconditioner of jacobian of comp_fcn to model state object, ms_in"""
        logger = logging.getLogger(__name__)
        logger.debug('precond_fname="%s", res_fname="%s"', precond_fname, res_fname)

        if solver_state is not None:
            fcn_complete_step = "apply_precond_jacobian complete for %s" % res_fname
            if solver_state.step_logged(fcn_complete_step):
                logger.debug('"%s" logged, returning result', fcn_complete_step)
                return ModelState(res_fname)
            logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        ms_res = ms_in.copy()

        with Dataset(precond_fname, mode="r") as fptr:
            # hist, and thus precond, files have mixing_coeff in m2 s-1
            # convert back to model units of m2 d-1
            mca = 86400.0 * fptr.variables["mixing_coeff_log_avg"][:]

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

        return ms_res.dump(res_fname)

    def _apply_precond_jacobian_iage(self, ms_in, mca, ms_res):
        """apply preconditioner of jacobian of iage fcn"""

        iage_in = ms_in.get_tracer_vals("iage")
        rhs = (1.0 / (self.time_range[1] - self.time_range[0])) * iage_in

        l_and_u = (1, 1)
        matrix_diagonals = np.zeros((3, self.depth.nlevs))
        matrix_diagonals[0, 1:] = (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        matrix_diagonals[1, :-1] -= (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        matrix_diagonals[1, 1:] -= (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[1:]
        )
        matrix_diagonals[2, :-1] = (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[1:]
        )
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
        matrix_diagonals[0, 1:] = (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        matrix_diagonals[1, :-1] -= (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        matrix_diagonals[1, 1:] -= (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[1:]
        )
        matrix_diagonals[2, :-1] = (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[1:]
        )
        matrix_diagonals[1, 0] = -240.0 * self.depth.delta_r[0]
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

        nz = self.depth.nlevs  # pylint: disable=C0103

        matrix = diags(
            [
                self._diag_0_phosphorus(mca),
                self._diag_p_1_phosphorus(mca),
                self._diag_m_1_phosphorus(mca),
                self._diag_p_nz_phosphorus(),
                self._diag_m_nz_phosphorus(),
                self._diag_p_2nz_phosphorus(),
                self._diag_m_2nz_phosphorus(),
            ],
            [0, 1, -1, nz, -nz, 2 * nz, -2 * nz],
            format="csr",
        )

        matrix_adj = matrix - 1.0e-8 * eye(3 * nz)
        res = spsolve(matrix_adj, rhs)

        _, sing_vals, r_sing_vects = svd(matrix.todense())
        min_ind = sing_vals.argmin()
        dz3 = np.concatenate((self.depth.delta, self.depth.delta, self.depth.delta))
        numer = (res * dz3).sum()
        denom = (r_sing_vects[min_ind, :] * dz3).sum()
        res -= numer / denom * r_sing_vects[min_ind, :]

        ms_res.set_tracer_vals("po4_s", res[0:nz] - po4_s)
        ms_res.set_tracer_vals("dop_s", res[nz : 2 * nz] - dop_s)
        ms_res.set_tracer_vals("pop_s", res[2 * nz : 3 * nz] - pop_s)

    def _diag_0_phosphorus(self, mca):
        """return main diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_0_single_tracer = np.zeros(self.depth.nlevs)
        diag_0_single_tracer[:-1] -= (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        diag_0_single_tracer[1:] -= (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[1:]
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
        diag_p_1_single_tracer = (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        diag_p_1_po4_s = diag_p_1_single_tracer.copy()
        zero = np.zeros(1)
        diag_p_1_dop_s = diag_p_1_single_tracer.copy()
        diag_p_1_pop_s = diag_p_1_single_tracer.copy()
        return np.concatenate(
            (diag_p_1_po4_s, zero, diag_p_1_dop_s, zero, diag_p_1_pop_s)
        )

    def _diag_m_1_phosphorus(self, mca):
        """return +1 upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_m_1_single_tracer = (
            mca[1:-1] * self.depth.delta_mid_r * self.depth.delta_r[1:]
        )
        diag_m_1_po4_s = diag_m_1_single_tracer.copy()
        zero = np.zeros(1)
        diag_m_1_dop_s = diag_m_1_single_tracer.copy()
        diag_m_1_pop_s = diag_m_1_single_tracer.copy()
        # pop_s sinking gain from layer above
        diag_m_1_pop_s += 1.0 * self.depth.delta_r[1:]
        return np.concatenate(
            (diag_m_1_po4_s, zero, diag_m_1_dop_s, zero, diag_m_1_pop_s)
        )

    def _diag_p_nz_phosphorus(self):
        """return +nz upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_p_1_dop_po4 = 0.01 * np.ones(self.depth.nlevs)  # dop_s remin
        diag_p_1_pop_dop = np.zeros(self.depth.nlevs)
        return np.concatenate((diag_p_1_dop_po4, diag_p_1_pop_dop))

    def _diag_m_nz_phosphorus(self):
        """return -nz lower diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_p_1_po4_dop = np.zeros(self.depth.nlevs)
        diag_p_1_po4_dop[0] = 0.67  # po4_s restoring conservation balance
        diag_p_1_dop_pop = np.zeros(self.depth.nlevs)
        return np.concatenate((diag_p_1_po4_dop, diag_p_1_dop_pop))

    def _diag_p_2nz_phosphorus(self):
        """return +2nz upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        return 0.01 * np.ones(self.depth.nlevs)  # pop_s remin

    def _diag_m_2nz_phosphorus(self):
        """return -2nz lower diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_p_1_po4_pop = np.zeros(self.depth.nlevs)
        diag_p_1_po4_pop[0] = 0.33  # po4_s restoring conservation balance
        return diag_p_1_po4_pop


################################################################################

if __name__ == "__main__":
    main(_parse_args())
