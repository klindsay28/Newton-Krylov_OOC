#!/usr/bin/env python
"""driver for Newton-Krylov solver"""

import argparse
import configparser
import logging
import os
import stat

from model import ModelStaticVars, get_modelinfo
from newton_solver import NewtonSolver

def gen_resume_script(solverinfo):
    """generate script that will be called to resume nk_driver.py"""

    # The contents are in a script, instead of a multi-command subprocess.run args argument, so that
    # the script can be passed to a batch job submit command. This is particularly useful when
    # resume is being called from a batch job that is running on many dedicated cores, and you don't
    # want to waste all of those cores running nk_driver, which is a single-core job.

    script_fname = get_modelinfo('resume_script_fname')
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(script_fname, mode='w') as fptr:
        fptr.write('#!/bin/bash\n')
        fptr.write('cd %s\n' % cwd)
        fptr.write('source %s\n' % solverinfo['newton_krylov_env_cmds_fname'])
        fptr.write('./nk_driver.py --cfg_fname %s --resume\n' % get_modelinfo('cfg_fname'))

    # ensure script_fname is executable by the user, while preserving other permissions
    fstat = os.stat(script_fname)
    os.chmod(script_fname, fstat.st_mode | stat.S_IXUSR)

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

    # store cfg_fname and resume_script_fname in modelinfo, to ease access to their values elsewhere
    config['modelinfo']['cfg_fname'] = args.cfg_fname
    cwd = os.path.dirname(os.path.realpath(__file__))
    resume_script_fname = os.path.join(cwd, 'generated_scripts', 'nk_driver_resume.sh')
    config['modelinfo']['resume_script_fname'] = resume_script_fname

    ModelStaticVars(config['modelinfo'], logging.DEBUG if args.resume else logging.INFO)

    gen_resume_script(solverinfo)

    newton_solver = NewtonSolver(solverinfo=solverinfo,
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
