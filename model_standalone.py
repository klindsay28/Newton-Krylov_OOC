"""
example of using model.py outside of nk_driver
"""

import configparser
import logging
import sys

from model import ModelState, ModelStaticVars
from nk_driver import parse_args

args = parse_args()
config = configparser.ConfigParser()
config.read(args.cfg_fname)

logging.basicConfig(format='%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s',
                    stream=sys.stdout, level='DEBUG')
logger = logging.getLogger(__name__)

ModelStaticVars(config['modelinfo'])

ms = ModelState('iterate_00.nc')
ms.log('iterate_00')

ms = ModelState('fcn_00.nc')
ms.log('fcn_00')

ms = ModelState('w_00.nc')
ms.log('w_00')

ms = ModelState('iterate_00_fp1.nc')
ms.log('iterate_00_fp1')

ms = ModelState('fcn_00_fp1.nc')
ms.log('fcn_00_fp1')

ms = ModelState('w_00_fp1.nc')
ms.log('w_00_fp1')

ms = ModelState('iterate_00_fp2.nc')
ms.log('iterate_00_fp2')

ms = ModelState('fcn_00_fp2.nc')
ms.log('fcn_00_fp2')

ms = ModelState('w_00_fp2.nc')
ms.log('w_00_fp2')
