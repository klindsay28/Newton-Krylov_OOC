"""
example of accessing/using newton_fcn_modname outside of nk_driver
"""

import configparser
import importlib
import logging
import sys

from ..model_config import ModelConfig
from ..nk_driver import parse_args
from ..solver import SolverState

args = parse_args()
config = configparser.ConfigParser()
config.read_file(open(args.cfg_fname))

logging_format = "%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s"
logging.basicConfig(format=logging_format, stream=sys.stdout, level="DEBUG")
logger = logging.getLogger(__name__)

ModelConfig(config["modelinfo"])

solver_state = SolverState("cime_pop", ".")

# import module with NewtonFcn class
newton_fcn_mod = importlib.import_module(config["modelinfo"]["newton_fcn_modname"])
newton_fcn_obj = newton_fcn_mod.NewtonFcn()

newton_fcn_obj._gen_precond_matrix_files(solver_state)
