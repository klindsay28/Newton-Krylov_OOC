#!/usr/bin/env python
"""set up files needed to run NK solver for test_problem"""

import argparse
import errno
import configparser
import importlib
import logging
import os
import sys

import git
import numpy as np

from netCDF4 import Dataset

from ..gen_invoker_script import mkdir_exist_okay
from ..model_config import ModelConfig
from .newton_fcn_test_problem import ModelState, NewtonFcn
from test_problem.src.spatial_axis import SpatialAxis


def _parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="setup test_problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        help="name of model that solver is being applied to",
        default="test_problem",
    )
    parser.add_argument(
        "--cfg_fname",
        help="name of configuration file",
        default="models/{model}/newton_krylov.cfg",
    )
    parser.add_argument(
        "--axisname", help="axis name", default="depth",
    )
    parser.add_argument(
        "--units", help="axis units", default="m",
    )
    parser.add_argument(
        "--nlevs", type=int, help="number of layers", default=40,
    )
    parser.add_argument(
        "--edge_start", type=float, help="start of edges", default=0.0,
    )
    parser.add_argument(
        "--edge_end", type=float, help="end of edges", default=400.0,
    )
    parser.add_argument(
        "--delta_start", type=float, help="thickness of first layer", default=5.0,
    )
    parser.add_argument(
        "--fp_cnt",
        type=int,
        help="number of fixed point iterations to apply to ic",
        default=2,
    )

    parsed_args = parser.parse_args()

    # replace {model} with specified model
    parsed_args.cfg_fname = parsed_args.cfg_fname.replace("{model}", parsed_args.model)

    return parsed_args


def main(args):
    """set up files needed to run NK solver for test_problem"""

    defaults = os.environ
    defaults["repo_root"] = git.Repo(search_parent_directories=True).working_dir
    config = configparser.ConfigParser(defaults)
    config.read_file(open(args.cfg_fname))
    solverinfo = config["solverinfo"]

    logging_format = "%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s"
    logging.basicConfig(
        stream=sys.stdout, format=logging_format, level=solverinfo["logging_level"]
    )
    logger = logging.getLogger(__name__)
    solverinfo = config["solverinfo"]

    logging_format = "%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s"
    logging.basicConfig(
        stream=sys.stdout, format=logging_format, level=solverinfo["logging_level"]
    )
    logger = logging.getLogger(__name__)

    logger.info('args.model="%s"', args.model)
    logger.info('args.cfg_fname="%s"', args.cfg_fname)

    defn_dict = {
        "units": args.units,
        "nlevs": args.nlevs,
        "edge_start": args.edge_start,
        "edge_end": args.edge_end,
        "delta_start": args.delta_start,
    }
    depth = SpatialAxis(axisname=args.axisname, defn_dict=defn_dict)

    modelinfo = config["modelinfo"]

    grid_weight_fname = modelinfo["grid_weight_fname"]
    logger.info('grid_weight_fname="%s"', grid_weight_fname)
    mkdir_exist_okay(os.path.dirname(grid_weight_fname))
    depth.dump(grid_weight_fname)

    workdir = config["solverinfo"]["workdir"]
    gen_ic_workdir = os.path.join(workdir, "gen_ic")
    mkdir_exist_okay(gen_ic_workdir)

    model_config = ModelConfig(modelinfo)

    tracer_module_defs = model_config.tracer_module_defs
    ic_vals = gen_ic_vals(tracer_module_defs, depth)
    ic_fname = os.path.join(gen_ic_workdir, "ic_00.nc")
    dump_vals(tracer_module_defs, depth, ic_vals, ic_fname)

    ic = ModelState(vals_fname="gen_ic")
    ic.dump(os.path.join(gen_ic_workdir, "ic_00.nc"))

    newton_fcn = NewtonFcn()

    for fp_iter in range(args.fp_cnt):
        logger.info('fp_iter="%d"', fp_iter)
        ic_fcn = newton_fcn.comp_fcn(
            ic,
            os.path.join(gen_ic_workdir, "fcn_%02d.nc" % fp_iter),
            None,
            os.path.join(gen_ic_workdir, "hist_%02d.nc" % fp_iter),
        )
        ic += ic_fcn
        ic.copy_shadow_tracers_to_real_tracers()
        ic.dump(os.path.join(gen_ic_workdir, "ic_%02d.nc" % (1 + fp_iter)))
    ic.dump(config["modelinfo"]["init_iterate_fname"])


def gen_ic_vals(tracer_module_defs, depth):
    """return ic values defined by tracer_module_defs"""
    ret = np.empty(len(tracer_module_defs), dtype=np.object)
    for tracer_module_ind, tracer_module_def in enumerate(tracer_module_defs.values()):
        vals = np.empty((len(tracer_module_def), depth.nlevs))
        for tracer_ind, tracer_metadata in enumerate(tracer_module_def.values()):
            if "ic_vals" in tracer_metadata:
                vals[tracer_ind, :] = np.interp(
                    depth.mid,
                    tracer_metadata["ic_val_depths"],
                    tracer_metadata["ic_vals"],
                )
            elif "shadows" in tracer_metadata:
                shadowed_tracer = tracer_metadata["shadows"]
                shadow_tracer_metadata = tracer_module_def[shadowed_tracer]
                vals[tracer_ind, :] = np.interp(
                    depth.mid,
                    shadow_tracer_metadata["ic_val_depths"],
                    shadow_tracer_metadata["ic_vals"],
                )
            else:
                msg = "gen_ic failure for %s" % self.tracer_names()[tracer_ind]
                raise ValueError(msg)
        ret[tracer_module_ind] = vals
    return ret


def dump_vals(tracer_module_defs, depth, vals, fname):
    """return ic values defined by tracer_module_defs"""
    with Dataset(fname, mode="w") as fptr:
        fptr.createDimension("depth", depth.nlevs)
        fptr.createVariable("depth", "f8", dimensions=("depth",))
        for tracer_module_def in tracer_module_defs.values():
            for tracer_name in tracer_module_def:
                fptr.createVariable(tracer_name, "f8", dimensions=("depth",))
        fptr.variables["depth"][:] = depth.mid
        for tracer_module_ind, tracer_module_def in enumerate(
            tracer_module_defs.values()
        ):
            tracer_module_vals = vals[tracer_module_ind]
            for tracer_ind, tracer_name in enumerate(tracer_module_def):
                fptr.variables[tracer_name][:] = tracer_module_vals[tracer_ind, :]


################################################################################

if __name__ == "__main__":
    main(_parse_args())
