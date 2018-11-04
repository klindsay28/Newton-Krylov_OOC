"""
example of using model.py outside of nk_driver
"""

import configparser
import logging
from model import model_init_static_vars
from model import ModelState
from nk_driver import parse_args

args = parse_args()
config = configparser.ConfigParser()
config.read(args.cfg_fname)

logging.basicConfig(format='%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s',
                    level='INFO')

model_init_static_vars(config['modelinfo'])

x = ModelState(['x'], 'iterate_00.nc')
x.log()
