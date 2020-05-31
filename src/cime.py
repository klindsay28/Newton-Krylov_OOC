"""
methods specific to CIME, but independent of models/components that are run with CIME
"""

import logging
import os
import stat
import subprocess

from .model_config import get_modelinfo


def cime_xmlquery(varname, caseroot=None):
    """run CIME's xmlquery for varname in the directory caseroot, return the value"""
    caseroot_loc = get_modelinfo("caseroot") if caseroot is None else caseroot
    return subprocess.check_output(
        ["./xmlquery", "--value", varname], cwd=caseroot_loc
    ).decode()


def cime_xmlchange(varname, value):
    """run CIME's xmlchange in the directory caseroot, setting varname to value"""
    # skip change command if it would not change the value
    # this avoids clutter in the file CaseStatus
    if value != cime_xmlquery(varname):
        caseroot = get_modelinfo("caseroot")
        subprocess.run(
            ["./xmlchange", "%s=%s" % (varname, value)], cwd=caseroot, check=True
        )


def cime_case_submit():
    """submit a CIME case, return after submit completes"""
    logger = logging.getLogger(__name__)

    script_fname = os.path.join(get_modelinfo("workdir"), "case_submit.sh")
    with open(script_fname, mode="w") as fptr:
        fptr.write("#!/bin/bash\n")
        fptr.write("cd %s\n" % get_modelinfo("repo_root"))
        fptr.write("source scripts/cime_env_cmds\n")
        fptr.write("cd %s\n" % get_modelinfo("caseroot"))
        fptr.write("./case.submit\n")

    # ensure script_fname is executable by the user, while preserving other permissions
    fstat = os.stat(script_fname)
    os.chmod(script_fname, fstat.st_mode | stat.S_IXUSR)

    logger.info('submitting case="%s"', cime_xmlquery("CASE"))
    subprocess.run(script_fname, shell=True, check=True)


def cime_yr_cnt(modelinfo=None):
    """
    return how many years are in forward model run
    assumes STOP_OPTION, STOP_N, RESUBMIT are in modelinfo section of cfg file
    """
    if modelinfo is None:
        stop_option = get_modelinfo("STOP_OPTION")
        stop_n = int(get_modelinfo("STOP_N"))
        resubmit = int(get_modelinfo("RESUBMIT"))
    else:
        stop_option = modelinfo["STOP_OPTION"]
        stop_n = int(modelinfo["STOP_N"])
        resubmit = int(modelinfo["RESUBMIT"])

    if stop_option in ["nyear", "nyears"]:
        return (resubmit + 1) * stop_n

    if stop_option in ["nmonth", "nmonths"]:
        nmonths = (resubmit + 1) * stop_n
        if nmonths % 12 != 0:
            msg = "number of months=%d not divisible by 12" % nmonths
            raise RuntimeError(msg)
        return nmonths // 12

    msg = "stop_option = %s not implemented" % stop_option
    raise NotImplementedError(msg)
