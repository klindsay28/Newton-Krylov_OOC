"""
example of using model.py outside of nk_driver
"""

import configparser
import logging

import numpy as np

from model import ModelState, ModelStaticVars, RegionScalars, log_vals, region_cnt
from nk_driver import parse_args

args = parse_args()
config = configparser.ConfigParser()
config.read(args.cfg_fname)

logging.basicConfig(format='%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)

ModelStaticVars(config['modelinfo'])

var1 = ModelState('iterate_00.nc')
logger.info('calling var1.log')
var1.log()
logger.info('calling var1.log with msg')
var1.log('msg')

rhs = np.array([RegionScalars(np.linspace(1.0, 2.0, region_cnt())),
                RegionScalars(np.linspace(2.0, 3.0, region_cnt()))])
logger.info('rhs')
log_vals('rhs', rhs)

logger.info('multiplying var1 by rhs')

var1 *= rhs
var1.log()
