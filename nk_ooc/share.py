"""functions shared across multiple modules"""

import argparse
import configparser
import logging
import os
import sys

from .utils import mkdir_exist_okay, strtobool

cfg_override_args = {
    "workdir": {"section": "DEFAULT"},
    "logging_fname": {"section": "solverinfo"},
    "logging_reproducible": {
        "section": "DEFAULT",
        "action": "store_true",
        "override_val": "True",
    },
    "logging_level": {"section": "solverinfo"},
    "newton_max_iter": {"section": "solverinfo"},
    "newton_rel_tol": {"section": "solverinfo"},
    "init_iterate_fname": {"section": "solverinfo"},
    "tracer_module_names": {"section": "modelinfo"},
    "persist": {
        "model_names": ["test_problem", "py_driver_2d"],
        "override_var": "reinvoke",
        "action": "store_true",
        "override_val": "False",
        "section": "modelinfo",
    },
}


def common_args(description, model_name, args_list):
    """instantiate and return a parser, using common options"""

    # process --model_name so that it can be passed to common_args
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--model_name",
        help="name of the model that solver is being applied to; "
        "using a non-default value alters subsequent options",
        default=model_name,
    )
    args, args_remaining = parent_parser.parse_known_args(args_list)

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parent_parser],
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    input_dir = os.path.join(repo_root, "input", args.model_name)
    defaults = [os.path.join(input_dir, "newton_krylov.cfg")]
    params_fname = os.path.join(input_dir, "model_params.cfg")
    if os.path.exists(params_fname):
        defaults.append(params_fname)
    parser.add_argument(
        "--cfg_fnames", help="names of configuration files", default=",".join(defaults)
    )

    # add arguments that override cfg file
    for argname, metadata in cfg_override_args.items():
        # skip arguments that are model specific for different model_names
        if args.model_name not in metadata.get("model_names", [args.model_name]):
            continue
        override_var = metadata.get("override_var", argname)
        if "action" not in metadata:
            parser.add_argument(
                f"--{argname}",
                help=f"override {override_var} from cfg file",
                default=None,
            )
        elif metadata["action"] in ["store_true"]:
            parser.add_argument(
                f"--{argname}",
                help=f"override {override_var} from cfg file",
                action=metadata["action"],
            )
        else:
            raise NotImplementedError(f'action={metadata["action"]} not implemented')

    return parser, args_remaining


def args_replace(args):
    """apply common args replacements/format on string arguments"""
    model_name_repl = args.model_name
    # pass "{suff}" through
    str_subs = {"model_name": model_name_repl, "suff": "{suff}"}
    for arg, value in vars(args).items():
        if isinstance(value, str):
            setattr(args, arg, value.format(**str_subs))
    return args


def read_cfg_files(args):
    """
    read cfg files
    set defaults common to all occurrances
    """
    cfg_fnames = args.cfg_fnames

    defaults = {key: os.environ[key] for key in ["HOME", "USER"]}
    defaults["repo_root"] = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config = configparser.ConfigParser(defaults, allow_no_value=True)
    files_read = config.read(cfg_fnames.split(","))
    if len(files_read) == 0:
        msg = f"cfg_fnames not read: {cfg_fnames}"
        raise RuntimeError(msg)

    _check_config_no_values(cfg_fnames, config)

    _apply_cfg_override_args(args, config)

    # write cfg contents to a file, if requested
    cfg_out_fname = config["solverinfo"]["cfg_out_fname"]
    if cfg_out_fname is not None:
        mkdir_exist_okay(os.path.dirname(cfg_out_fname))
        with open(cfg_out_fname, "w") as fptr:
            config.write(fptr)

    return config


def _check_config_no_values(cfg_fnames, config):
    """verify that only names in no_value_allowed have no value"""
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
                msg = f"{name} not allowed to be empty in cfg file {cfg_fnames}"
                raise ValueError(msg)


def _apply_cfg_override_args(args, config):
    """apply cfg_override_args to config"""
    for argname, metadata in cfg_override_args.items():
        # skip conditional overrides that were not added
        if argname not in args:
            continue
        override_var = metadata.get("override_var", argname)
        if override_var not in config[metadata["section"]]:
            msg = f'{override_var} not in cfg section {metadata["section"]}'
            raise ValueError(msg)
        if "action" not in metadata:
            if getattr(args, argname) is not None:
                config[metadata["section"]][override_var] = getattr(args, argname)
        elif metadata["action"] == "store_true":
            if getattr(args, argname):
                config[metadata["section"]][override_var] = metadata["override_val"]


def logging_config(solverinfo, filemode):
    """configure logging"""
    logging_format_list = []
    if not strtobool(solverinfo["logging_reproducible"]):
        logging_format_list.extend(["%(asctime)s", "%(process)s"])
    logging_format_list.extend(["%(filename)s", "%(funcName)s", "%(message)s"])
    logging_format = ":".join(logging_format_list)
    logging.basicConfig(
        format=logging_format,
        level=solverinfo["logging_level"],
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(filename=solverinfo["logging_fname"], mode=filemode),
        ],
    )


def repro_fname(cfg_section, fname):
    """return version of fname appropriate for reproducible logging, if specified"""
    ret = fname
    if strtobool(cfg_section["logging_reproducible"]):
        ret = ret.replace(cfg_section["workdir"], "$workdir")
        ret = ret.replace(cfg_section["repo_root"], "$repo_root")
    return ret
