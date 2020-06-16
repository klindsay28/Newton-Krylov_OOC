#!/usr/bin/env python
"""test_problem model specifics for ModelStateBase"""

import copy
from datetime import datetime
from distutils.util import strtobool
from inspect import signature
import logging
import os
import subprocess
import sys

from netCDF4 import Dataset
import numpy as np
from scipy.integrate import solve_ivp

from ..model_config import ModelConfig, get_modelinfo
from ..model_state_base import ModelStateBase
from ..share import args_replace, common_args, read_cfg_file, logging_config
from ..utils import class_name, create_dimensions_verify, create_vars

from .spatial_axis import SpatialAxis
from .vert_mix import VertMix


def parse_args(args_list_in=None):
    """parse command line arguments"""

    args_list = [] if args_list_in is None else args_list_in
    parser, args_remaining = common_args(
        "test problem model standalone driver for Newton-Krylov solver",
        "test_problem",
        args_list,
    )
    parser.add_argument(
        "cmd",
        choices=["comp_fcn", "gen_precond_jacobian", "apply_precond_jacobian"],
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

    return args_replace(parser.parse_args(args_remaining))


def _resolve_fname(fname_dir, fname):
    """prepend fname_dir to fname, if fname is a relative path"""
    if fname is None or os.path.isabs(fname):
        return fname
    return os.path.join(fname_dir, fname)


def main(args):
    """test problem for Newton-Krylov solver"""

    config = read_cfg_file(args)
    solverinfo = config["solverinfo"]

    logging_config(args, solverinfo, filemode="a")
    logger = logging.getLogger(__name__)

    logger.info('args.cmd="%s"', args.cmd)

    # store cfg_fname in modelinfo, to ease access to its value elsewhere
    config["modelinfo"]["cfg_fname"] = args.cfg_fname

    ModelConfig(config["modelinfo"])

    ms_in = ModelState(_resolve_fname(args.fname_dir, args.in_fname))
    if args.cmd == "comp_fcn":
        ms_in.log("state_in")
        ms_in.comp_fcn(
            _resolve_fname(args.fname_dir, args.res_fname),
            solver_state=None,
            hist_fname=_resolve_fname(args.fname_dir, args.hist_fname),
        )
        ModelState(_resolve_fname(args.fname_dir, args.res_fname)).log("fcn")
    elif args.cmd == "gen_precond_jacobian":
        ms_in.gen_precond_jacobian(
            _resolve_fname(args.fname_dir, args.hist_fname),
            _resolve_fname(args.fname_dir, args.precond_fname),
            solver_state=None,
        )
    elif args.cmd == "apply_precond_jacobian":
        ms_in.log("state_in")
        ms_in.apply_precond_jacobian(
            _resolve_fname(args.fname_dir, args.precond_fname),
            _resolve_fname(args.fname_dir, args.res_fname),
            solver_state=None,
        )
        ModelState(_resolve_fname(args.fname_dir, args.res_fname)).log("precond_res")
    else:
        msg = "unknown cmd=%s" % args.cmd
        raise ValueError(msg)

    logger.info("done")


class ModelState(ModelStateBase):
    """test_problem model specifics for ModelStateBase"""

    # give ModelState operators higher priority than those of numpy
    __array_priority__ = 100

    def __init__(self, fname):
        self.time_range = (0.0, 365.0)
        self.depth = SpatialAxis(axisname="depth", fname=get_modelinfo("depth_fname"))

        self.vert_mix = VertMix(self.depth)

        super().__init__(fname)

    def hist_time_mean_weights(self, fptr_hist):
        """return weights for computing time-mean in hist file"""
        # downweight endpoints because test_problem writes t=0 and t=365 to hist
        timelen = len(fptr_hist.dimensions["time"])
        weights = np.full(timelen, 1.0 / (timelen - 1))
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return weights

    def get_tracer_vals_all(self):
        """get all tracer values"""
        res_vals = np.empty((self.tracer_cnt, len(self.depth)))
        ind0 = 0
        for tracer_module in self.tracer_modules:
            cnt = tracer_module.tracer_cnt
            res_vals[ind0 : ind0 + cnt, :] = tracer_module.get_tracer_vals_all()
            ind0 = ind0 + cnt
        return res_vals

    def set_tracer_vals_all(self, vals, reseat_vals=False):
        """set all tracer values"""
        ind0 = 0
        if reseat_vals:
            tracer_modules_orig = self.tracer_modules
            self.tracer_modules = np.empty(len(tracer_modules_orig), dtype=np.object)
            for ind, tracer_module_orig in enumerate(tracer_modules_orig):
                self.tracer_modules[ind] = copy.copy(tracer_module_orig)
                cnt = tracer_module_orig.tracer_cnt
                self.tracer_modules[ind].set_tracer_vals_all(
                    vals[ind0 : ind0 + cnt, :], reseat_vals=reseat_vals
                )
                ind0 = ind0 + cnt
        else:
            for tracer_module in self.tracer_modules:
                cnt = tracer_module.tracer_cnt
                tracer_module.set_tracer_vals_all(
                    vals[ind0 : ind0 + cnt, :], reseat_vals=reseat_vals
                )
                ind0 = ind0 + cnt

    def comp_fcn(self, res_fname, solver_state, hist_fname=None):
        """evalute function being solved with Newton's method"""
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s", hist_fname="%s"', res_fname, hist_fname)

        if solver_state is not None:
            fcn_complete_step = "comp_fcn complete for %s" % res_fname
            if solver_state.step_logged(fcn_complete_step):
                logger.debug('"%s" logged, returning result', fcn_complete_step)
                return ModelState(res_fname)
            logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        # memory for result, use it initially for passing initial value to solve_ivp
        res_vals = self.get_tracer_vals_all()

        # solve ODEs, using scipy.integrate
        # get dense output, if requested
        if hist_fname is not None:
            t_eval = np.linspace(self.time_range[0], self.time_range[1], 101)
        else:
            t_eval = np.array(self.time_range)
        sol = solve_ivp(
            self._comp_tend,
            self.time_range,
            res_vals.reshape(-1),
            "Radau",
            t_eval,
            atol=1.0e-10,
            rtol=1.0e-10,
        )

        self._gen_hist(sol, hist_fname)

        res_vals[:] = sol.y[:, -1].reshape(res_vals.shape) - res_vals

        # ModelState instance for result
        res_ms = copy.copy(self)
        res_ms.set_tracer_vals_all(res_vals, reseat_vals=True)

        caller = class_name(self) + ".comp_fcn"
        res_ms.comp_fcn_postprocess(res_fname, caller)

        if solver_state is not None:
            solver_state.log_step(fcn_complete_step)
            if strtobool(get_modelinfo("reinvoke")):
                cmd = [get_modelinfo("invoker_script_fname"), "--resume"]
                logger.info('cmd="%s"', " ".join(cmd))
                # use Popen instead of run because we don't want to wait
                subprocess.Popen(cmd)
                raise SystemExit

        return res_ms

    def _comp_tend(self, time, tracer_vals_flat):
        """compute tendency function"""

        tracer_vals = tracer_vals_flat.reshape((self.tracer_cnt, -1))
        dtracer_vals_dt = np.empty_like(tracer_vals)

        ind0 = 0
        for tracer_module in self.tracer_modules:
            cnt = tracer_module.tracer_cnt
            tracer_module.comp_tend(
                time,
                tracer_vals[ind0 : ind0 + cnt, :],
                dtracer_vals_dt[ind0 : ind0 + cnt, :],
                self.vert_mix,
            )
            ind0 = ind0 + cnt
        return dtracer_vals_dt.reshape(-1)

    def _gen_hist(self, sol, hist_fname):
        """write tracer values generated in comp_fcn to hist_fname"""

        if hist_fname is None:
            return

        # generate hist var metadata, starting with tracer module independent vars
        hist_vars_metadata = {
            "bldepth": {
                "dimensions": ("time"),
                "attrs": {"long_name": "boundary layer depth", "units": "m"},
            },
            "mixing_coeff": {
                "dimensions": ("time", "depth_edges"),
                "attrs": {
                    "long_name": "vertical mixing coefficient",
                    "units": "m^2 / d",
                },
            },
        }
        # add tracer module hist vars
        for tracer_module in self.tracer_modules:
            hist_vars_metadata.update(tracer_module.hist_vars_metadata())

        # create the hist file
        with Dataset(hist_fname, mode="w", format="NETCDF3_64BIT_OFFSET") as fptr:
            datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            name = __name__ + "._gen_hist"
            msg = datestamp + ": created by " + name
            setattr(fptr, "history", msg)

            # define hist dimensions and coordinate vars
            self._def_dims_coord_vars_hist(fptr)

            # set cell_methods attribute and define hist vars
            for metadata in hist_vars_metadata.values():
                if "time" in metadata["dimensions"]:
                    metadata["attrs"]["cell_methods"] = "time: point"
            create_vars(fptr, hist_vars_metadata)

            # write coordinate vars
            self._write_coord_vars_hist(fptr, sol.t)

            # (re-)compute and write tracer module independent vars
            for time_ind, time in enumerate(sol.t):
                fptr.variables["bldepth"][time_ind] = self.vert_mix.bldepth(time)
                fptr.variables["mixing_coeff"][time_ind, 1:-1] = (
                    self.vert_mix.mixing_coeff(time) * self.depth.delta_mid
                )
                # kludge to avoid missing values
                fptr.variables["mixing_coeff"][time_ind, 0] = fptr.variables[
                    "mixing_coeff"
                ][time_ind, 1]
                fptr.variables["mixing_coeff"][time_ind, -1] = fptr.variables[
                    "mixing_coeff"
                ][time_ind, -2]

            # write tracer module hist vars, providing appropriate segment of sol.y
            tracer_vals_all = sol.y.reshape((self.tracer_cnt, len(self.depth), -1))
            ind0 = 0
            for tracer_module in self.tracer_modules:
                cnt = tracer_module.tracer_cnt
                tracer_module.write_hist_vars(
                    fptr, tracer_vals_all[ind0 : ind0 + cnt, :]
                )
                ind0 = ind0 + cnt

    def _def_dims_coord_vars_hist(self, fptr):
        """define netCDF4 dimensions and coordinate vars relevant to test_problem"""
        fptr.createDimension("time", None)

        create_dimensions_verify(fptr, self.depth.dump_dimensions())

        fptr.createVariable("time", "f8", dimensions=("time",))
        fptr.variables["time"].long_name = "time"
        fptr.variables["time"].units = "days since 0001-01-01"
        fptr.variables["time"].calendar = "noleap"

        create_vars(fptr, self.depth.dump_vars_metadata())

    def _write_coord_vars_hist(self, fptr, time):
        """write netCDF4 coordinate vars relevant to test_problem"""
        fptr.variables["time"][:] = time

        self.depth.dump_write(fptr)

    def apply_precond_jacobian(self, precond_fname, res_fname, solver_state):
        """apply preconditioner of jacobian of comp_fcn to model state object, self"""
        logger = logging.getLogger(__name__)
        logger.debug('precond_fname="%s", res_fname="%s"', precond_fname, res_fname)

        if solver_state is not None:
            fcn_complete_step = "apply_precond_jacobian complete for %s" % res_fname
            if solver_state.step_logged(fcn_complete_step):
                logger.debug('"%s" logged, returning result', fcn_complete_step)
                return ModelState(res_fname)
            logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        # ModelState instance for result
        res_ms = copy.deepcopy(self)

        pos_args = ["self", "time_range", "res_tms"]

        arg_to_hist_dict = {
            "mca": "mixing_coeff_log_mean",
            "po4_s_restore_tau_r": "po4_s_restore_tau_r_mean",
        }

        with Dataset(precond_fname, mode="r") as fptr:
            for tracer_module_ind, tracer_module in enumerate(self.tracer_modules):
                kwargs = {}
                for arg in signature(tracer_module.apply_precond_jacobian).parameters:
                    if arg in pos_args:
                        continue
                    hist_varname = arg_to_hist_dict[arg]
                    hist_var = fptr.variables[hist_varname]
                    if "depth_edges" in hist_var.dimensions:
                        kwargs[arg] = hist_var[1:-1]
                    else:
                        kwargs[arg] = hist_var[:]

                tracer_module.apply_precond_jacobian(
                    self.time_range, res_ms.tracer_modules[tracer_module_ind], **kwargs,
                )

        if solver_state is not None:
            solver_state.log_step(fcn_complete_step)

        caller = class_name(self) + ".apply_precond_jacobian"
        return res_ms.dump(res_fname, caller)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
