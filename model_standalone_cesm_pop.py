"""
example of using model.py outside of nk_driver
"""

import configparser
import logging
import sys

from model import ModelStaticVars, get_modelinfo
from nk_driver import parse_args

args = parse_args()
config = configparser.ConfigParser()
config.read(args.cfg_fname)

logging.basicConfig(format='%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s',
                    stream=sys.stdout, level='DEBUG')
logger = logging.getLogger(__name__)

ModelStaticVars(config['modelinfo'])

print(get_modelinfo('caseroot'))
print(get_modelinfo('batch_cmd_script').replace('\n', ' ').replace('\r', ' '))
print(get_modelinfo('batch_cmd_script').split())
print(get_modelinfo('batch_cmd_precond').replace('\n', ' ').replace('\r', ' '))
print(get_modelinfo('batch_cmd_precond').split())
print(__file__)
