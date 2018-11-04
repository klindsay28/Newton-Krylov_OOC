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

var1 = ModelState(['x'], 'iterate_00.nc')
var1.log('var1')

var2 = var1
var2.log('var2 = var1')

var3 = var1.copy()
var3.log('var3 = var1.copy()')

var1 *= 2.0
var1.log('var1 doubled')
var2.log('var2')
var3.log('var3')
