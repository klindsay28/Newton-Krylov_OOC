"""test functions in share.py"""

import os

import pytest

from src.share import common_args, read_cfg_files


def test_parse_args():
    """run parse_args function and confirm that args.model_name is set"""
    model_name = "test_problem"
    parser, args_remaining = common_args("test_share", model_name, [])
    args = parser.parse_args(args_remaining)

    assert args.model_name == model_name


@pytest.mark.parametrize("args_list", [[], ["--persist"]])
def test_read_cfg_files(args_list):
    """run read_cfg_files function and confirm a setting from each section"""
    model_name = "test_problem"
    workdir = os.path.join(os.getenv("HOME"), "travis_short_workdir")
    args_list.extend(["--workdir", workdir])
    parser, args_remaining = common_args("test_share", model_name, args_list)
    args = parser.parse_args(args_remaining)
    config = read_cfg_files(args)

    assert config["DEFAULT"]["model_name"] == model_name
    assert config["solverinfo"]["newton_max_iter"] == "5"

    expected = "False" if "--persist" in args_list else "True"
    assert config["modelinfo"]["reinvoke"] == expected
