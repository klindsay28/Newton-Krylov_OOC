"""test test_problem features"""

import os

from src.model_config import ModelConfig
from src.share import common_args, read_cfg_files
from src.test_problem.model_state import ModelState


def test_depth_shared():
    """confirm that depth axis is shared"""
    workdir = os.path.join(os.getenv("HOME"), "travis_short_workdir")
    args_list = ["--workdir", workdir]
    parser, args_remaining = common_args("test_model_config", "test_problem", args_list)
    args = parser.parse_args(args_remaining)
    config = read_cfg_files(args)
    ModelConfig(config["modelinfo"])

    model_state_a = ModelState("gen_init_iterate")
    assert model_state_a.tracer_modules[0].depth is model_state_a.depth

    model_state_b = ModelState("gen_init_iterate")
    assert model_state_a.depth is model_state_b.depth
