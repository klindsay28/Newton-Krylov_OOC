"""functions common to multiple tests"""

import os

from netCDF4 import Dataset

from nk_ooc.model_config import ModelConfig
from nk_ooc.share import common_args, read_cfg_files
from nk_ooc.spatial_axis import spatial_axis_defn_dict, spatial_axis_from_defn_dict
from nk_ooc.utils import create_vars, mkdir_exist_okay


def config_test_problem():
    """read cfg for test_problem"""

    # workdir for tests
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    workdir = os.path.join(repo_root, "tests", "workdir")
    mkdir_exist_okay(workdir)

    # read cfg files
    args_list = ["--workdir", workdir]
    parser, args_remaining = common_args("test_model_config", "test_problem", args_list)
    args = parser.parse_args(args_remaining)
    config = read_cfg_files(args)

    # set up depth axis, using default settings
    # dump to expected location
    depth = spatial_axis_from_defn_dict(defn_dict=spatial_axis_defn_dict())
    grid_vars_fname = config["modelinfo"]["grid_vars_fname"]
    depth.dump(grid_vars_fname, caller="tests.share.config_test_problem")

    # add region_mask to grid_vars_fname
    vars_metadata = {
        "region_mask": {
            "datatype": "i4",
            "dimensions": "depth",
            "attrs": {
                "long_name": "Region Mask",
                "cell_measures": "thickness: depth_delta",
            },
        }
    }
    with Dataset(grid_vars_fname, mode="a") as fptr:
        create_vars(fptr, vars_metadata)
        fptr.variables["region_mask"][:] = 1

    # configure model and return result
    return ModelConfig(config["modelinfo"])
