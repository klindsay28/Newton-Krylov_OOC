"""
example of using model.py outside of nk_driver
"""

import configparser
import logging
import sys

from model import ModelStaticVars, get_modelinfo
from nk_driver import parse_args
from solver import SolverState

args = parse_args()
config = configparser.ConfigParser()
config.read(args.cfg_fname)

logging_format = '%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s'
logging.basicConfig(format=logging_format, stream=sys.stdout, level='DEBUG')
logger = logging.getLogger(__name__)

msv = ModelStaticVars(config['modelinfo'])

solver_state = SolverState('cime_pop', '.')

msv.newton_fcn._gen_precond_matrix_files(solver_state)
