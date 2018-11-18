#!/usr/bin/env python
"""driver for Newton-Krylov solver"""

import argparse
import configparser
import logging
import os

from model import ModelStaticVars
from newton_solver import NewtonSolver

def parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(description="Newton's method example")

    parser.add_argument('--cfg_fname', help='name of configuration file',
                        default='newton_krylov.cfg')

    parser.add_argument('--resume', help="resume Newton's method from solver's saved state",
                        action='store_true', default=False)
    parser.add_argument('--rewind', help="rewind last step to recover from error",
                        action='store_true', default=False)

    return parser.parse_args()

def main(args):
    """driver for Newton-Krylov solver"""

    config = configparser.ConfigParser()
    config.read(args.cfg_fname)
    solverinfo = config['solverinfo']

    logging.basicConfig(filename=solverinfo['logging_fname'],
                        filemode='a' if args.resume else 'w',
                        format='%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s',
                        level=solverinfo['logging_level'])
    logger = logging.getLogger(__name__)

    if os.path.exists('KILL'):
        logger.warning('KILL file detected, exiting')
        raise SystemExit

    ModelStaticVars(config['modelinfo'], args.cfg_fname,
                    logging.DEBUG if args.resume else logging.INFO)

    newton_solver = NewtonSolver(solverinfo=solverinfo,
                                 modelinfo=config['modelinfo'],
                                 resume=args.resume,
                                 rewind=args.rewind)

    while True:
        if newton_solver.converged_flat().all():
            logger.info('convergence criterion satisfied')
            newton_solver.log()
            break
        newton_solver.step()

if __name__ == '__main__':
    main(parse_args())
