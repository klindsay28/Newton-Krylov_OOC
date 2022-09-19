#!/usr/bin/env python
"""set up files needed to run NK solver for py_driver_2d"""

import cProfile
import logging
import os
import pstats
import sys
from datetime import datetime

import numpy as np
from netCDF4 import Dataset

from .. import gen_invoker_script
from ..model_config import ModelConfig
from ..share import (
    args_replace,
    common_args,
    logging_config,
    read_cfg_files,
    repro_fname,
)
from ..spatial_axis import spatial_axis_defn_dict, spatial_axis_from_defn_dict
from ..utils import (
    create_dimensions_verify,
    create_vars,
    extract_dimensions,
    mkdir_exist_okay,
)
from .model_state import ModelState


def parse_args(args_list_in=None):
    """parse command line arguments"""

    args_list = [] if args_list_in is None else args_list_in
    parser, args_remaining = common_args(
        "setup solver for py_driver_2d model", "py_driver_2d", args_list
    )

    parser.add_argument(
        "--init_iterate_opt",
        help="option for specifying initial iterate",
        default="gen_init_iterate",
    )
    parser.add_argument(
        "--fp_cnt",
        type=int,
        help="number of fixed point iterations to apply to init_iterate",
        default=2,
    )
    parser.add_argument(
        "--prof_comp_fcn_fname",
        help="profile comp_fcn call; write output to provided argument",
        default=None,
    )

    return args_replace(parser.parse_args(args_remaining))


def main(args):
    """set up files needed to run NK solver for py_driver_2d"""

    config = read_cfg_files(args)
    solverinfo = config["solverinfo"]

    logging_config(solverinfo, filemode="w")
    logger = logging.getLogger(__name__)

    logger.info('args.cfg_fnames="%s"', repro_fname(solverinfo, args.cfg_fnames))

    # ensure workdir exists
    mkdir_exist_okay(solverinfo["workdir"])

    # generate invoker script
    args.model_name = "py_driver_2d"
    gen_invoker_script.main(args)

    modelinfo = config["modelinfo"]

    caller = "nk_ooc.py_driver_2d.setup_solver.main"

    # generate and write grid_weight
    grid_weight_fname = modelinfo["grid_weight_fname"]
    logger.info('grid_weight_fname="%s"', repro_fname(modelinfo, grid_weight_fname))
    mkdir_exist_okay(os.path.dirname(grid_weight_fname))
    gen_grid_weight_file(args, modelinfo)

    # generate and write region_mask, if specified
    region_mask_fname = modelinfo["region_mask_fname"]
    varname = modelinfo["region_mask_varname"]
    if region_mask_fname is not None and varname is not None:
        logger.info('region_mask_fname="%s"', repro_fname(modelinfo, region_mask_fname))
        mkdir_exist_okay(os.path.dirname(region_mask_fname))
        gen_region_mask_file(modelinfo)

    # confirm that model configuration works with generated file
    # ModelState relies on model being configured
    ModelState.model_config_obj = ModelConfig(modelinfo)

    # generate initial condition
    init_iterate = ModelState(args.init_iterate_opt)

    if args.prof_comp_fcn_fname is not None:
        cProfile.runctx(
            "init_iterate.comp_fcn(res_fname=None, solver_state=None, hist_fname=None)",
            globals=None,
            locals={"init_iterate": init_iterate},
            filename=args.prof_comp_fcn_fname,
        )
        stats_obj = pstats.Stats(args.prof_comp_fcn_fname)
        stats_obj.strip_dirs().sort_stats("time").print_stats(20)
        return

    # perform fixed point iteration(s) on init_iterate
    if args.fp_cnt > 0:
        workdir = solverinfo["workdir"]
        gen_init_iterate_workdir = os.path.join(workdir, "gen_init_iterate")
        mkdir_exist_okay(gen_init_iterate_workdir)

        for fp_iter in range(args.fp_cnt):
            logger.info("fp_iter=%d", fp_iter)
            init_iterate.dump(
                os.path.join(
                    gen_init_iterate_workdir, "init_iterate_%04d.nc" % fp_iter
                ),
                caller,
            )
            init_iterate_fcn = init_iterate.comp_fcn(
                os.path.join(gen_init_iterate_workdir, "fcn_%04d.nc" % fp_iter),
                None,
                os.path.join(gen_init_iterate_workdir, "hist_%04d.nc" % fp_iter),
            )
            init_iterate += init_iterate_fcn
            init_iterate.copy_shadow_tracers_to_real_tracers()

    # write generated init_iterate to where solver expects it to be
    init_iterate_fname = solverinfo["init_iterate_fname"]
    logger.info('init_iterate_fname="%s"', repro_fname(solverinfo, init_iterate_fname))
    mkdir_exist_okay(os.path.dirname(init_iterate_fname))
    init_iterate.dump(init_iterate_fname, caller)


def gen_grid_weight_file(args, modelinfo):
    """generate grid weight file based on args and modelinfo"""

    # generate axes from args and modelinfo
    axisnames = ["depth", "ypos"]
    axes = {axisname: gen_axis(axisname, args, modelinfo) for axisname in axisnames}

    with Dataset(
        modelinfo["grid_weight_fname"], mode="w", format="NETCDF3_64BIT_OFFSET"
    ) as fptr:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "nk_ooc.py_driver_2d.setup_solver.gen_grid_weight_file"
        fptr.history = datestamp + ": created by " + name

        for axis in axes.values():
            create_dimensions_verify(fptr, axis.dump_dimensions())
            create_vars(fptr, axis.dump_vars_metadata())

        vars_metadata = {}
        vars_metadata[modelinfo["grid_weight_varname"]] = {
            "dimensions": tuple(axisnames),
            "attrs": {"long_name": "grid-cell area", "units": "m^2"},
        }
        create_vars(fptr, vars_metadata)

        for axis in axes.values():
            axis.dump_write(fptr)

        weight = np.outer(axes["depth"].delta, axes["ypos"].delta)
        fptr.variables[modelinfo["grid_weight_varname"]][:] = weight


def gen_region_mask_file(modelinfo):
    """generate region_mask file based on modelinfo"""

    with Dataset(modelinfo["grid_weight_fname"], mode="r") as fptr:
        mask_dimensions = extract_dimensions(fptr, modelinfo["grid_weight_varname"])
        mask_shape = fptr.variables[modelinfo["grid_weight_varname"]][:].shape

    max_abs_vvel = float(modelinfo["max_abs_vvel"])
    horiz_mix_coeff = float(modelinfo["horiz_mix_coeff"])
    if max_abs_vvel == 0.0 and horiz_mix_coeff == 0.0:
        mask = np.empty(mask_shape, dtype=np.int32)
        for ypos_i in range(mask_shape[1]):
            mask[:, ypos_i] = ypos_i + 1
    else:
        mask = np.ones(mask_shape, dtype=np.int32)

    mode_out = (
        "a" if modelinfo["region_mask_fname"] == modelinfo["grid_weight_fname"] else "w"
    )

    with Dataset(
        modelinfo["region_mask_fname"], mode=mode_out, format="NETCDF3_64BIT_OFFSET"
    ) as fptr_out:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "nk_ooc.py_driver_2d.setup_solver.gen_region_mask_file"
        msg = datestamp + ": "
        if mode_out == "a":
            history_in = getattr(fptr_out, "history", None)
            msg = msg + modelinfo["region_mask_varname"] + " appended by " + name
        else:
            msg = msg + "created by " + name
        fptr_out.history = msg if history_in is None else "\n".join([msg, history_in])

        # propagate dimension sizes from fptr_in to fptr_out
        create_dimensions_verify(fptr_out, mask_dimensions)

        vars_metadata = {
            modelinfo["region_mask_varname"]: {
                "datatype": mask.dtype,
                "dimensions": tuple(mask_dimensions),
                "attrs": {"long_name": "Region Mask"},
            }
        }
        create_vars(fptr_out, vars_metadata)

        fptr_out.variables[modelinfo["region_mask_varname"]][:] = mask


def gen_axis(axisname, args, modelinfo):
    """
    generate axis object based on args and modelinfo
    """

    defn_dict = {}
    for key, defn in spatial_axis_defn_dict(axisname=axisname).items():
        axis_key = "_".join([axisname, key])
        if axis_key in modelinfo:
            defn_dict[key] = (defn["type"])(modelinfo[axis_key])
        if hasattr(args, axis_key):
            defn_dict[key] = getattr(args, axis_key)

    return spatial_axis_from_defn_dict(defn_dict=spatial_axis_defn_dict(**defn_dict))


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
