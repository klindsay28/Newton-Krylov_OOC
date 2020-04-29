"""
example of accessing/using newton_fcn_modname outside of nk_driver
"""

import configparser
import importlib
import logging
import sys

from model_config import ModelConfig
from nk_driver import parse_args

args = parse_args()
config = configparser.ConfigParser()
config.read(args.cfg_fname)

logging_format = "%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s"
logging.basicConfig(format=logging_format, stream=sys.stdout, level="DEBUG")
logger = logging.getLogger(__name__)

ModelConfig(config["modelinfo"])

# import module with NewtonFcn class
newton_fcn_mod = importlib.import_module(config["modelinfo"]["newton_fcn_modname"])
newton_fcn_obj = newton_fcn_mod.NewtonFcn()

ms = newton_fcn_obj.model_state_obj("iterate_test_00.nc")
ms.log("iterate_test_00")

ms = newton_fcn_obj.model_state_obj("fcn_test_00.nc")
ms.log("fcn_test_00")

ms = newton_fcn_obj.model_state_obj("w_test_00.nc")
ms.log("w_test_00")

ms = newton_fcn_obj.model_state_obj("iterate_test_00_fp1.nc")
ms.log("iterate_test_00_fp1")

ms = newton_fcn_obj.model_state_obj("fcn_test_00_fp1.nc")
ms.log("fcn_test_00_fp1")

ms = newton_fcn_obj.model_state_obj("w_test_00_fp1.nc")
ms.log("w_test_00_fp1")

ms = newton_fcn_obj.model_state_obj("iterate_test_00_fp2.nc")
ms.log("iterate_test_00_fp2")

ms = newton_fcn_obj.model_state_obj("fcn_test_00_fp2.nc")
ms.log("fcn_test_00_fp2")

ms = newton_fcn_obj.model_state_obj("w_test_00_fp2.nc")
ms.log("w_test_00_fp2")
