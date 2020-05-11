"""general purpose utility functions"""

import configparser
import errno
import os

import git


def mkdir_exist_okay(path):
    """
    Create a directory named path.
    It is okay if it already exists.
    """
    try:
        os.mkdir(path)
    except OSError as err:
        if err.errno == errno.EEXIST:
            pass
        else:
            raise


def read_cfg_file(cfg_fname):
    """
    read cfg_fname
    set defaults common to all occurrances
    """
    defaults = os.environ
    defaults["repo_root"] = git.Repo(search_parent_directories=True).working_dir
    config = configparser.ConfigParser(defaults)
    config.read_file(open(cfg_fname))
    return config
