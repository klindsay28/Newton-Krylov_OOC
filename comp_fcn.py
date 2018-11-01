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

    x1 = ms_in.get_tracer_vals('x1')
    x2 = ms_in.get_tracer_vals('x2')
    y = ms_in.get_tracer_vals('y')

    sx1 = np.linspace(0.7, 0.8, np.size(x1)).reshape(np.shape(x1))
    sx2 = np.linspace(0.8, 0.9, np.size(x1)).reshape(np.shape(x2))
    sy = np.linspace(0.9, 1.0, np.size(x1)).reshape(np.shape(y))

    ms_res.set_tracer_vals('x1', np.cos(x1) - sx1 * x2)
    ms_res.set_tracer_vals('x2', np.cos(x2) - sx2 * x1)
    ms_res.set_tracer_vals('y', np.cos(y) - sy * y * np.average(y))

    ms_res.dump(args.res_fname)

if __name__ == '__main__':
    main(_parse_args())
