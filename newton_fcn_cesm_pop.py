#!/usr/bin/env python
"""cesm pop hooks for Newton-Krylov solver"""

import argparse
import configparser
import glob
import os
import shutil
import stat
import subprocess

import numpy as np

from netCDF4 import Dataset

from model import TracerModuleStateBase, ModelState, ModelStaticVars, get_modelinfo

def _parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="cesm pop hooks for Newton-Krylov solver")
    parser.add_argument('cmd', choices=['comp_fcn', 'apply_precond_jacobian'],
                        help='command to run')
    parser.add_argument('in_fname', help='name of file with input')
    parser.add_argument('res_fname', help='name of file for result')
    parser.add_argument('--cfg_fname', help='name of configuration file',
                        default='newton_krylov_cesm_pop.cfg')
    parser.add_argument('--hist_fname', help='name of history file', default='None')
    parser.add_argument('--resume_script_fname', help='name of script to resume nk_driver.py',
                        default='None')
    parser.add_argument('--post_modelrun', help='is the script being invoked after a model run',
                        action='store_true', default=False)

    return parser.parse_args()

def main(args):
    """cesm pop hooks for Newton-Krylov solver"""

    config = configparser.ConfigParser()
    config.read(args.cfg_fname)

    # store cfg_fname in modelinfo, to ease access to its values elsewhere
    config['modelinfo']['cfg_fname'] = args.cfg_fname

    ModelStaticVars(config['modelinfo'])

    if args.cmd == 'comp_fcn':
        if not args.post_modelrun:
            # note that the following will not return, as it raises SystemExit
            _comp_fcn_pre_modelrun(ModelState(args.in_fname), args.hist_fname,
                                   args.resume_script_fname, args.in_fname, args.res_fname)
        else:
            ms_res = _comp_fcn_post_modelrun(ModelState(args.in_fname), args.hist_fname)
            ms_res.dump(args.res_fname)
            if args.resume_script_fname != 'None':
                subprocess.Popen(args.resume_script_fname)
    elif args.cmd == 'apply_precond_jacobian':
        ms_res = _apply_precond_jacobian(ModelState(args.in_fname))
        ms_res.dump(args.res_fname)
    else:
        raise ValueError('unknown cmd=%s' % args.cmd)

################################################################################

class TracerModuleState(TracerModuleStateBase):
    """
    Derived class for representing a collection of model tracers.
    It implements _read_vals and dump.
    """

    def _read_vals(self, tracer_module_name, vals_fname):
        """return tracer values and dimension names and lengths, read from vals_fname)"""
        dims = {}
        suffix = '_CUR'
        with Dataset(vals_fname, mode='r') as fptr:
            fptr.set_auto_mask(False)
            # get dims from first variable
            dimnames0 = fptr.variables[self.tracer_names()[0]+suffix].dimensions
            for dimname in dimnames0:
                dims[dimname] = fptr.dimensions[dimname].size
            # all tracers are stored in a single array
            # tracer index is the leading index
            vals = np.empty(shape=(self.tracer_cnt(),) + tuple(dims.values()))
            # check that all vars have the same dimensions
            for tracer_name in self.tracer_names():
                if fptr.variables[tracer_name+suffix].dimensions != dimnames0:
                    raise ValueError('not all vars have same dimensions',
                                     'tracer_module_name=', tracer_module_name,
                                     'vals_fname=', vals_fname)
            # read values
            if len(dims) > 3:
                raise ValueError('ndim too large (for implementation of dot_prod)',
                                 'tracer_module_name=', tracer_module_name,
                                 'vals_fname=', vals_fname,
                                 'ndim=', len(dims))
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                varid = fptr.variables[tracer_name+suffix]
                vals[tracer_ind, :] = varid[:]
        return vals, dims

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleState object
        to an open file
        """
        if action == 'define':
            for dimname, dimlen in self._dims.items():
                try:
                    if fptr.dimensions[dimname].size != dimlen:
                        raise ValueError('dimname already exists and has wrong size',
                                         'tracer_module_name=', self._tracer_module_name,
                                         'dimname=', dimname)
                except KeyError:
                    fptr.createDimension(dimname, dimlen)
            dimnames = tuple(self._dims.keys())
            # define all tracers, with _CUR and _OLD suffixes
            for tracer_name in self.tracer_names():
                for suffix in ['_CUR', '_OLD']:
                    fptr.createVariable(tracer_name+suffix, 'f8', dimensions=dimnames)
        elif action == 'write':
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                for suffix in ['_CUR', '_OLD']:
                    fptr.variables[tracer_name+suffix][:] = self._vals[tracer_ind, :]
        else:
            raise ValueError('unknown action=', action)
        return self

################################################################################

class NewtonFcn():
    """class of methods related to problem being solved with Newton's method"""
    def __init__(self):
        pass

    def comp_fcn(self, ms_in, hist_fname='None'):
        """evalute function being solved with Newton's method"""
        raise NotImplementedError(
            'comp_fcn not implemented for inline execution in ' + __file__)

    def apply_precond_jacobian(self, ms_in):
        """apply preconditioner of jacobian of comp_fcn to model state object, ms_in"""
        raise NotImplementedError(
            'apply_precond_jacobian not implemented for inline execution in ' + __file__)

################################################################################

def _comp_fcn_pre_modelrun(ms_in, hist_fname, resume_script_fname, in_fname, res_fname):
    """pre-modelrun step of evaluting the function being solved with Newton's method"""

    # relative pathname of tracer_ic
    tracer_ic_fname_rel = 'tracer_ic.nc'
    fname = os.path.join(_xmlquery('RUNDIR'), tracer_ic_fname_rel)
    ms_in.dump(fname)

    # ensure certain env xml vars are set properly
    _xmlchange('POP_PASSIVE_TRACER_RESTART_OVERRIDE', tracer_ic_fname_rel)
    _xmlchange('CONTINUE_RUN', 'FALSE')

    # copy rpointer files to rundir
    rundir = _xmlquery('RUNDIR')
    for src in glob.glob(os.path.join(get_modelinfo('rpointer_dir'), 'rpointer.*')):
        shutil.copy(src, rundir)

    # generate post-modelrun script and point POSTRUN_SCRIPT to it
    # this will propagate cfg_fname and hist_fname across model run
    cwd = os.path.dirname(os.path.realpath(__file__))
    post_modelrun_script_fnames = \
        [os.path.join(cwd, 'generated_scripts', 'post_modelrun_direct.sh'),
         os.path.join(cwd, 'generated_scripts', 'post_modelrun_indirect.sh')]
    _gen_post_modelrun_scripts(hist_fname, resume_script_fname, in_fname, res_fname,
                               post_modelrun_script_fnames)
    _xmlchange('POSTRUN_SCRIPT', post_modelrun_script_fnames[0])

    # submit the model run and exit
    _case_submit()

    raise SystemExit

def _gen_post_modelrun_scripts(hist_fname, resume_script_fname, in_fname, res_fname, script_fnames):
    """
    generate scripts that will be called by cime after the model run
    script_fname[0] is called by CIME, and submits script_fnames[1] as a batch job
    script_fnames[1] sets up the environment necessary for the Newton-Krylov scripts to run
    and the re-invokes __file__
    """
    cwd = os.path.dirname(os.path.realpath(__file__))
    batch_cmd_script = get_modelinfo('batch_cmd_script').replace('\n', ' ').replace('\r', ' ')
    with open(script_fnames[0], mode='w') as fptr:
        fptr.write('#!/bin/bash\n')
        fptr.write('cd %s\n' % cwd)
        fptr.write('%s %s\n' % (batch_cmd_script, script_fnames[1]))

    # argument list for the call to __file__ is long, build it up argument by argument
    file_args = " --cfg_fname %s" % get_modelinfo('cfg_fname')
    file_args += " --hist_fname %s" % hist_fname
    file_args += " --resume_script_fname %s" % resume_script_fname
    file_args += " --post_modelrun"
    file_args += " comp_fcn"
    file_args += " %s" % in_fname
    file_args += " %s" % res_fname

    with open(script_fnames[1], mode='w') as fptr:
        fptr.write('#!/bin/bash\n')
        fptr.write('cd %s\n' % cwd)
        fptr.write('source %s\n' % get_modelinfo('newton_krylov_env_cmds_fname'))
        fptr.write('%s%s\n' % (__file__, file_args))

    # ensure script_fnames are executable by the user, while preserving other permissions
    for script_fname in script_fnames:
        fstat = os.stat(script_fname)
        os.chmod(script_fname, fstat.st_mode | stat.S_IXUSR)

def _comp_fcn_post_modelrun(ms_in, hist_fname):
    """post-modelrun step of evaluting the function being solved with Newton's method"""

    if hist_fname != 'None':
        pass

    # determine name of end of run restart file from POP's rpointer file
    rpointer_fname = os.path.join(_xmlquery('RUNDIR'), 'rpointer.ocn.restart')
    with open(rpointer_fname, mode='r') as fptr:
        rest_file_fname_rel = fptr.readline().strip()
    fname = os.path.join(_xmlquery('RUNDIR'), rest_file_fname_rel)

    return ModelState(fname) - ms_in

def _apply_precond_jacobian(ms_in):
    """apply preconditioner of jacobian of comp_fcn to model state object, ms_in"""

    ms_res = ms_in.copy()

    return ms_res

def _xmlquery(varname):
    """run CIME's _xmlquery for varname in the directory caseroot, return the value"""
    caseroot = get_modelinfo('caseroot')
    obj = subprocess.run(['./xmlquery', '--value', varname], stdout=subprocess.PIPE,
                         cwd=caseroot, check=True)
    return obj.stdout.decode()

def _xmlchange(varname, value):
    """run CIME's _xmlchange in the directory caseroot, setting varname to value"""
    # skip change command if it would not change the value
    # this avoids clutter in the file CaseStatus
    if value != _xmlquery(varname):
        caseroot = get_modelinfo('caseroot')
        subprocess.run(['./xmlchange', '%s=%s' % (varname, value)], cwd=caseroot, check=True)

def _case_submit():
    """submit a CIME case, return after submit completes"""

    cwd = os.path.dirname(os.path.realpath(__file__))
    script_fname = os.path.join(cwd, 'generated_scripts', 'case_submit.sh')
    with open(script_fname, mode='w') as fptr:
        fptr.write('#!/bin/bash\n')
        fptr.write('source %s\n' % get_modelinfo('cime_env_cmds_fname'))
        fptr.write('cd %s\n' % get_modelinfo('caseroot'))
        fptr.write('./case.submit\n')

    # ensure script_fname is executable by the user, while preserving other permissions
    fstat = os.stat(script_fname)
    os.chmod(script_fname, fstat.st_mode | stat.S_IXUSR)

    subprocess.run(script_fname, shell=True, check=True)

if __name__ == '__main__':
    main(_parse_args())
