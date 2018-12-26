"""methods specific to CIME"""

import logging
import os
import stat
import subprocess

from model_config import get_modelinfo

def cime_xmlquery(varname):
    """run CIME's xmlquery for varname in the directory caseroot, return the value"""
    caseroot = get_modelinfo('caseroot')
    return subprocess.check_output(
        ['./xmlquery', '--value', varname], cwd=caseroot).decode()

def cime_xmlchange(varname, value):
    """run CIME's xmlchange in the directory caseroot, setting varname to value"""
    # skip change command if it would not change the value
    # this avoids clutter in the file CaseStatus
    if value != cime_xmlquery(varname):
        caseroot = get_modelinfo('caseroot')
        subprocess.run(
            ['./xmlchange', '%s=%s' % (varname, value)], cwd=caseroot, check=True)

def cime_case_submit():
    """submit a CIME case, return after submit completes"""
    logger = logging.getLogger(__name__)

    cwd = os.path.dirname(os.path.realpath(__file__))
    script_fname = os.path.join(cwd, 'generated_scripts', 'case_submit.sh')
    with open(script_fname, mode='w') as fptr:
        fptr.write('#!/bin/bash -l\n')
        fptr.write('source %s\n' % get_modelinfo('cime_env_cmds_fname'))
        fptr.write('cd %s\n' % get_modelinfo('caseroot'))
        fptr.write('./case.submit\n')

    # ensure script_fname is executable by the user, while preserving other permissions
    fstat = os.stat(script_fname)
    os.chmod(script_fname, fstat.st_mode | stat.S_IXUSR)

    logger.info('submitting case=%s', cime_xmlquery('CASE'))
    subprocess.run(script_fname, shell=True, check=True)

def cime_yr_cnt():
    """return how many years are in forward model run"""
    stop_option = cime_xmlquery('STOP_OPTION')
    stop_n = int(cime_xmlquery('STOP_N'))
    if stop_option == 'nyear':
        yr_cnt = stop_n
    elif stop_option == 'nmonth':
        if stop_n % 12 != 0:
            msg = 'number of months=%d not divisible by 12' % stop_n
            raise RuntimeError(msg)
        yr_cnt = int(stop_n) // 12
    else:
        msg = 'stop_option = %s not implemented' % stop_option
        raise NotImplementedError(msg)
    return yr_cnt
