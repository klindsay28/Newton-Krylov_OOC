#!/usr/bin/env python
"""Newton's method example"""

import logging
import os
import numpy as np
import file_wrap

def comp_fcn(fname_arg, fname_fcn):
    """function whose root is being found"""
    arg = file_wrap.read_var(fname_arg, 'arg')
    fcn = np.cos(arg)-0.7*arg
    file_wrap.write_var(fname_fcn, 'fcn', fcn)

def comp_arg_inc(fname_arg, fname_fcn, fname_arg_inc):
    """Newton's method increment"""
    arg = file_wrap.read_var(fname_arg, 'arg')
    fcn = file_wrap.read_var(fname_fcn, 'fcn')
    dfcn_darg = -np.sin(arg)-0.7
    file_wrap.write_var(fname_arg_inc, 'arg_inc', -fcn/dfcn_darg)

def main():
    """Newton's method example"""

    logging.basicConfig(filename='newton.log', filemode='w',
                        format='%(asctime)s:%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger()

    workdir = 'work'
    try:
        os.mkdir(workdir)
    except FileExistsError:
        pass

    newton_iter = 0
    arg = 0.0

    fname_arg = workdir+'/arg_%02d.nc' % newton_iter
    file_wrap.write_var(fname_arg, 'arg', arg)
    fname_fcn = workdir+'/fcn_%02d.nc' % newton_iter
    comp_fcn(fname_arg, fname_fcn)
    fcn_val = file_wrap.read_var(fname_fcn, 'fcn')
    logger.info("newton_iter=%d, arg=%e, y=%e", newton_iter, arg, fcn_val)

    while np.abs(fcn_val) > 1.0e-10:
        newton_iter = newton_iter + 1
        fname_arg_inc = workdir+'/arg_inc_%02d.nc' % newton_iter
        comp_arg_inc(fname_arg, fname_fcn, fname_arg_inc)
        darg = file_wrap.read_var(fname_arg_inc, 'arg_inc')
        arg = arg + darg
        fname_arg = workdir+'/arg_%02d.nc' % newton_iter
        file_wrap.write_var(fname_arg, 'arg', arg)
        fname_fcn = workdir+'/fcn_%02d.nc' % newton_iter
        comp_fcn(fname_arg, fname_fcn)
        fcn_val = file_wrap.read_var(fname_fcn, 'fcn')
        logger.info("newton_iter=%d, arg=%e, y=%e", newton_iter, arg, fcn_val)

if __name__ == '__main__':
    main()
