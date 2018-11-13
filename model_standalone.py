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
                    level='DEBUG')
logger = logging.getLogger(__name__)

ModelStaticVars(config['modelinfo'])

ms = ModelState('work/krylov_00/precond_fcn_00.nc')
ms.log('precond_fcn_00')

ms = ModelState('work/krylov_00/basis_00.nc')
ms.log('basis_00')

ms = ModelState('work/krylov_00/ext_in.nc')
ms.log('ext_in')

ms = ModelState('work/krylov_00/fcn_res.nc')
ms.log('fcn_res')
