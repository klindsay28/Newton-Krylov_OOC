"""
example of using model.py outside of nk_driver
"""

import configparser
import logging
import sys

from model import ModelStateBase
from model_config import ModelConfig
from nk_driver import parse_args

args = parse_args()
config = configparser.ConfigParser()
config.read(args.cfg_fname)

logging_format = '%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s'
logging.basicConfig(format=logging_format, stream=sys.stdout, level='DEBUG')
logger = logging.getLogger(__name__)

ModelConfig(config['modelinfo'])

ms = ModelStateBase('iterate_test_00.nc')
ms.log('iterate_test_00')

ms = ModelStateBase('fcn_test_00.nc')
ms.log('fcn_test_00')

ms = ModelStateBase('w_test_00.nc')
ms.log('w_test_00')

ms = ModelStateBase('iterate_test_00_fp1.nc')
ms.log('iterate_test_00_fp1')

ms = ModelStateBase('fcn_test_00_fp1.nc')
ms.log('fcn_test_00_fp1')

ms = ModelStateBase('w_test_00_fp1.nc')
ms.log('w_test_00_fp1')

ms = ModelStateBase('iterate_test_00_fp2.nc')
ms.log('iterate_test_00_fp2')

ms = ModelStateBase('fcn_test_00_fp2.nc')
ms.log('fcn_test_00_fp2')

ms = ModelStateBase('w_test_00_fp2.nc')
ms.log('w_test_00_fp2')
