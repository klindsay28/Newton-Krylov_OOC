"""functions common to multiple tests"""

import os

import git

from src.model_config import ModelConfig
from src.share import common_args, read_cfg_files
from src.spatial_axis import SpatialAxis, spatial_axis_defn_dict
from src.utils import mkdir_exist_okay


def config_test_problem():
    """read cfg for test_problem"""

    # workdir for tests
    repo_root = git.Repo(search_parent_directories=True).working_dir
    workdir = os.path.join(repo_root, "tests", "workdir")
    mkdir_exist_okay(workdir)

    # read cfg files
    args_list = ["--workdir", workdir]
    parser, args_remaining = common_args("test_model_config", "test_problem", args_list)
    args = parser.parse_args(args_remaining)
    config = read_cfg_files(args)

    # set up depth axis, using default settings
    # dump to expected location
    depth = SpatialAxis(defn_dict=spatial_axis_defn_dict())
    depth_fname = config["modelinfo"]["depth_fname"]
    depth.dump(depth_fname, caller="tests.share.config_test_problem")

    # configure model
    ModelConfig(config["modelinfo"])

    return config
