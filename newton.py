#!/usr/bin/env python
"""Newton's method example"""

import logging
import numpy as np

def fcn(arg):
    """function whose root is being found"""
    return np.cos(arg)-0.7*arg

def arg_inc(arg, fcn_val):
    """Newton's method increment"""
    dfcn_darg = -np.sin(arg)-0.7
    return -fcn_val / dfcn_darg

def main():
    """Newton's method example"""

    logging.basicConfig(filename='newton.log', filemode='w',
                        format='%(asctime)s:%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger()

    newton_iter = 0
    arg = 0.0
    fcn_val = fcn(arg)
    logger.info("newton_iter=%d, arg=%e, y=%e", newton_iter, arg, fcn_val)

    while np.abs(fcn_val) > 1.0e-10:
        newton_iter = newton_iter + 1
        darg = arg_inc(arg, fcn_val)
        arg = arg + darg
        fcn_val = fcn(arg)
        logger.info("newton_iter=%d, arg=%e, y=%e", newton_iter, arg, fcn_val)

if __name__ == '__main__':
    main()
