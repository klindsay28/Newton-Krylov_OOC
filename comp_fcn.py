#!/usr/bin/env python
"""test problem for Newton-Krylov solver"""

import argparse
import numpy as np
from model import ModelState

def _parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="test problem for Newton-Krylov solver")
    parser.add_argument('in_fname', help='name of file with ModelState input')
    parser.add_argument('res_fname', help='name of file where output ModelState gets written')
    return parser.parse_args()

def main(args):
    """test problem for Newton-Krylov solver"""
    ms_in = ModelState(['x', 'y'], args.in_fname)
    ms_res = 1.0 * ms_in

    x1 = ms_in._tracer_modules[0]._vals[0, :]
    sx1 = np.linspace(0.7, 0.8, np.size(x1)).reshape(np.shape(x1))
    ms_res._tracer_modules[0]._vals[0, :] = np.cos(x1) - sx1 * x1

    x2 = ms_in._tracer_modules[0]._vals[1, :]
    sx2 = np.linspace(0.8, 0.9, np.size(x1)).reshape(np.shape(x2))
    ms_res._tracer_modules[0]._vals[1, :] = np.cos(x2) - sx2 * x2

    y = ms_in._tracer_modules[1]._vals[:]
    sy = np.linspace(0.9, 1.0, np.size(x1)).reshape(np.shape(y))
    ms_res._tracer_modules[1]._vals[:] = np.cos(y) - sy * y

    ms_res.dump(args.res_fname)

if __name__ == '__main__':
    main(_parse_args())
