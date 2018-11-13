#!/usr/bin/env python
"""apply tracer_weight to a ModelState object in a file"""

import argparse
import configparser
from model import ModelState, ModelStaticVars

def _parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="apply tracer_weight to a ModelState object in a file")
    parser.add_argument(
        'fname', help='name of file with the object that tracer_weight is being applied to')
    parser.add_argument(
        '--cfg_fname', help='name of configuration file', default='newton_krylov.cfg')
    return parser.parse_args()

def main(args):
    """apply tracer_weight to a ModelState object in a file"""

    config = configparser.ConfigParser()
    config.read(args.cfg_fname)
    ModelStaticVars(config['modelinfo'], args.cfg_fname)

    ModelState(args.fname).apply_tracer_weight().dump(args.fname)

if __name__ == '__main__':
    main(_parse_args())
