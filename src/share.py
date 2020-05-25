"""functions shared across multiple modules"""

import argparse
import configparser
import os

import git

from .utils import mkdir_exist_okay


def common_args(description, model_name="test_problem"):
    """instantiate and return a parser, using common options"""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        help="name of model that solver is being applied to",
        default=model_name,
    )
    parser.add_argument(
        "--cfg_fname",
        help="name of configuration file",
        default="models/{model_name}/newton_krylov.cfg",
    )
    parser.add_argument(
        "--workdir", help="override workdir from cfg file", default=None,
    )
    parser.add_argument(
        "--tracer_module_names",
        help="override tracer_module_names from cfg file",
        default=None,
    )
    if model_name == "test_problem":
        parser.add_argument(
            "--persist", help="override reinvoke from cfg file", action="store_true",
        )
    return parser


def args_replace(args):
    """apply common args replacements/format on string arguments"""
    # pass "{suff}" through
    str_subs = {"model_name": args.model_name, "suff": "{suff}"}
    for arg, value in vars(args).items():
        if isinstance(value, str):
            setattr(args, arg, value.format(**str_subs))
    return args


def read_cfg_file(args):
    """
    read cfg file
    set defaults common to all occurrances
    """
    cfg_fname = args.cfg_fname

    defaults = {key: os.environ[key] for key in ["HOME", "USER"]}
    defaults["repo_root"] = git.Repo(search_parent_directories=True).working_dir
    config = configparser.ConfigParser(defaults, allow_no_value=True)
    config.read_file(open(cfg_fname))

    # verify that only names in no_value_allowed have no value
    # no_value_allowed is allowed to have no value or not be present
    if "no_value_allowed" in config["DEFAULT"]:
        no_value_allowed = config["DEFAULT"]["no_value_allowed"]
    else:
        no_value_allowed = None
    nva_list = [] if no_value_allowed is None else no_value_allowed.split(",")
    nva_list.append("no_value_allowed")
    for section in config.sections():
        for name in config[section]:
            if config[section][name] is None and name not in nva_list:
                msg = "%s not allowed to be empty in cfg file %s" % (name, cfg_fname)
                raise ValueError(msg)

    if args.workdir is not None:
        config["DEFAULT"]["workdir"] = args.workdir

    if args.tracer_module_names is not None:
        config["modelinfo"]["tracer_module_names"] = args.tracer_module_names

    if "persist" in args and args.persist:
        config["modelinfo"]["reinvoke"] = "False"

    cfg_out_fname = config["solverinfo"]["cfg_out_fname"]
    if cfg_out_fname is not None:
        mkdir_exist_okay(os.path.dirname(cfg_out_fname))
        with open(cfg_out_fname, "w") as fptr:
            config.write(fptr)

    return config
