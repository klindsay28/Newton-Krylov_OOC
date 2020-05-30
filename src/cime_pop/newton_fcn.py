#!/usr/bin/env python
"""cime_pop hooks for Newton-Krylov solver"""

from __future__ import division

from datetime import datetime
import glob
import logging
import math
import os
import shutil
import stat
import subprocess
import sys

from netCDF4 import Dataset
import numpy as np

from ..cime import cime_xmlquery, cime_xmlchange, cime_case_submit, cime_yr_cnt
from ..model import ModelStateBase, TracerModuleStateBase
from ..model_config import ModelConfig, get_modelinfo, get_precond_matrix_def
from ..newton_fcn_base import NewtonFcnBase
from ..share import args_replace, common_args, read_cfg_file
from ..utils import ann_files_to_mean_file, mon_files_to_mean_file


def _parse_args():
    """parse command line arguments"""
    parser = common_args("cime pop hooks for Newton-Krylov solver", "cime_pop")
    parser.add_argument(
        "cmd",
        choices=["comp_fcn", "gen_precond_jacobian", "apply_precond_jacobian"],
        help="command to run",
    )
    parser.add_argument("--hist_fname", help="name of history file", default=None)
    parser.add_argument("--in_fname", help="name of file with input")
    parser.add_argument("--res_fname", help="name of file for result")

    return args_replace(parser.parse_args(), model_name="cime_pop")


def main(args):
    """cime pop hooks for Newton-Krylov solver"""

    config = read_cfg_file(args)
    solverinfo = config["solverinfo"]

    logging_format = "%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s"
    logging.basicConfig(
        stream=sys.stdout, format=logging_format, level=solverinfo["logging_level"]
    )
    logger = logging.getLogger(__name__)

    logger.info('args.cmd="%s"', args.cmd)

    # store cfg_fname in modelinfo, to ease access to its values elsewhere
    config["modelinfo"]["cfg_fname"] = args.cfg_fname

    ModelConfig(config["modelinfo"])

    msg = "%s not implemented for command line execution in %s " % (args.cmd, __file__)
    if args.cmd == "comp_fcn":
        raise NotImplementedError(msg)
    if args.cmd == "gen_precond_jacobian":
        raise NotImplementedError(msg)
    if args.cmd == "apply_precond_jacobian":
        raise NotImplementedError(msg)
    msg = "unknown cmd=%s" % args.cmd
    raise ValueError(msg)


################################################################################


class ModelState(ModelStateBase):
    """class for representing the state space of a model"""

    # give ModelState operators higher priority than those of numpy
    __array_priority__ = 100

    def __init__(self, tracer_module_state_class, fname):
        logger = logging.getLogger(__name__)
        logger.debug('ModelState, fname="%s"', fname)
        super().__init__(tracer_module_state_class, fname)

    def tracer_dims_keep_in_stats(self):
        """tuple of dimensions to keep for tracers in stats file"""
        return ("z_t", "nlat")

    def gen_precond_jacobian(self, hist_fname, precond_fname):
        """
        Generate file(s) needed for preconditioner of jacobian of comp_fcn
        evaluated at self
        """
        logger = logging.getLogger(__name__)
        logger.debug('hist_fname="%s", precond_fname="%s"', hist_fname, precond_fname)

        super().gen_precond_jacobian(hist_fname, precond_fname)

        self._gen_precond_matrix_files(precond_fname)

    def _gen_precond_matrix_files(self, precond_fname):
        """
        Generate matrix files for preconditioner of jacobian of comp_fcn
        evaluated at self
        """
        logger = logging.getLogger(__name__)
        jacobian_precond_tools_dir = get_modelinfo("jacobian_precond_tools_dir")

        opt_str_subs = {
            "day_cnt": 365 * cime_yr_cnt(),
            "precond_fname": precond_fname,
            "reg_fname": get_modelinfo("region_mask_fname"),
            "irf_fname": get_modelinfo("irf_fname"),
        }

        workdir = os.path.dirname(precond_fname)

        for matrix_name in self.precond_matrix_list():
            matrix_opts = get_precond_matrix_def(matrix_name)["precond_matrices_opts"]
            # apply option string substitutions
            for ind, matrix_opt in enumerate(matrix_opts):
                matrix_opts[ind] = matrix_opt.format(**opt_str_subs)

            matrix_opts_fname = os.path.join(workdir, "matrix_" + matrix_name + ".opts")
            with open(matrix_opts_fname, "w") as fptr:
                for opt in matrix_opts:
                    fptr.write("%s\n" % opt)
            matrix_fname = os.path.join(workdir, "matrix_" + matrix_name + ".nc")
            matrix_gen_exe = os.path.join(jacobian_precond_tools_dir, "bin", "gen_A")
            cmd = [
                matrix_gen_exe,
                "-D1",
                "-o",
                matrix_opts_fname,
                matrix_fname,
            ]
            logger.info('cmd="%s"', " ".join(cmd))
            subprocess.run(cmd, check=True)

            # add creation metadata to file attributes
            with Dataset(matrix_fname, mode="a") as fptr:
                datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg = datestamp + ": created by " + matrix_gen_exe
                fcn_name = __name__ + "NewtonFcn._gen_precond_matrix_files"
                msg = msg + " called from " + fcn_name
                if hasattr(fptr, "history"):
                    msg = msg + "\n" + getattr(fptr, "history")
                setattr(fptr, "history", msg)
                setattr(fptr, "matrix_opts", "\n".join(matrix_opts))


################################################################################


class TracerModuleState(TracerModuleStateBase):
    """
    Derived class for representing a collection of model tracers.
    It implements _read_vals and dump.
    """

    def _read_vals(self, tracer_module_name, fname):
        """return tracer values and dimension names and lengths, read from fname)"""
        logger = logging.getLogger(__name__)
        logger.debug('tracer_module_name="%s", fname="%s"', tracer_module_name, fname)
        dims = {}
        suffix = "_CUR"
        with Dataset(fname, mode="r") as fptr:
            fptr.set_auto_mask(False)
            # get dims from first variable
            dimnames0 = fptr.variables[self.tracer_names()[0] + suffix].dimensions
            for dimname in dimnames0:
                dims[dimname] = fptr.dimensions[dimname].size
            # all tracers are stored in a single array
            # tracer index is the leading index
            vals = np.empty((self.tracer_cnt(),) + tuple(dims.values()))
            # check that all vars have the same dimensions
            for tracer_name in self.tracer_names():
                if fptr.variables[tracer_name + suffix].dimensions != dimnames0:
                    msg = (
                        "not all vars have same dimensions"
                        ", tracer_module_name=%s, fname=%s"
                        % (tracer_module_name, fname)
                    )
                    raise ValueError(msg)
            # read values
            if len(dims) > 3:
                msg = (
                    "ndim too large (for implementation of dot_prod)"
                    "tracer_module_name=%s, fname=%s, ndim=%s"
                    % (tracer_module_name, fname, len(dims))
                )
                raise ValueError(msg)
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                varid = fptr.variables[tracer_name + suffix]
                vals[tracer_ind, :] = varid[:]
        return vals, dims

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleState object
        to an open file
        """
        if action == "define":
            for dimname, dimlen in self._dims.items():
                try:
                    if fptr.dimensions[dimname].size != dimlen:
                        msg = (
                            "dimname already exists and has wrong size"
                            "tracer_module_name=%s, dimname=%s"
                            % (self._tracer_module_name, dimname)
                        )
                        raise ValueError(msg)
                except KeyError:
                    fptr.createDimension(dimname, dimlen)
            dimnames = tuple(self._dims.keys())
            # define all tracers, with _CUR and _OLD suffixes
            for tracer_name in self.tracer_names():
                for suffix in ["_CUR", "_OLD"]:
                    fptr.createVariable(tracer_name + suffix, "f8", dimensions=dimnames)
        elif action == "write":
            # write all tracers, with _CUR and _OLD suffixes
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                for suffix in ["_CUR", "_OLD"]:
                    fptr.variables[tracer_name + suffix][:] = self._vals[tracer_ind, :]
        else:
            msg = "unknown action=", action
            raise ValueError(msg)
        return self


################################################################################


class NewtonFcn(NewtonFcnBase):
    """class of methods related to problem being solved with Newton's method"""

    def __init__(self):
        pass

    def model_state_obj(self, fname):
        """return a ModelState object compatible with this function"""
        return ModelState(TracerModuleState, fname)

    def comp_fcn(self, ms_in, res_fname, solver_state, hist_fname=None):
        """evalute function being solved with Newton's method"""
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s", hist_fname="%s"', res_fname, hist_fname)

        fcn_complete_step = "comp_fcn complete for %s" % res_fname
        if solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return ModelState(TracerModuleState, res_fname)
        logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        _comp_fcn_pre_modelrun(ms_in, res_fname, solver_state)

        _gen_hist(hist_fname)

        ms_res = _comp_fcn_post_modelrun(ms_in)

        caller = __name__ + ".NewtonFcn.comp_fcn"
        ms_res.comp_fcn_postprocess(res_fname, caller)

        solver_state.log_step(fcn_complete_step)

        return ms_res

    def apply_precond_jacobian(self, ms_in, precond_fname, res_fname, solver_state):
        """apply preconditioner of jacobian of comp_fcn to model state object, ms_in"""
        logger = logging.getLogger(__name__)
        logger.debug('precond_fname="%s", res_fname="%s"', precond_fname, res_fname)

        fcn_complete_step = "apply_precond_jacobian complete for %s" % res_fname
        if solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return ModelState(TracerModuleState, res_fname)
        logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        _apply_precond_jacobian_pre_solve_lin_eqns(ms_in, res_fname, solver_state)

        lin_eqns_soln_fname = os.path.join(
            os.path.dirname(res_fname), "lin_eqns_soln_" + os.path.basename(res_fname)
        )
        ms_res = _apply_precond_jacobian_solve_lin_eqns(
            ms_in, precond_fname, lin_eqns_soln_fname, solver_state
        )

        ms_res -= ms_in

        solver_state.log_step(fcn_complete_step)

        caller = __name__ + ".NewtonFcn.apply_precond_jacobian"
        return ms_res.dump(res_fname, caller)


################################################################################


def _comp_fcn_pre_modelrun(ms_in, res_fname, solver_state):
    """pre-modelrun step of evaluting the function being solved with Newton's method"""
    logger = logging.getLogger(__name__)
    logger.debug('res_fname="%s"', res_fname)

    fcn_complete_step = "_comp_fcn_pre_modelrun complete for %s" % res_fname
    if solver_state.step_logged(fcn_complete_step):
        logger.debug('"%s" logged, returning', fcn_complete_step)
        return
    logger.debug('"%s" not logged, proceeding', fcn_complete_step)

    # relative pathname of tracer_ic
    tracer_ic_fname_rel = "tracer_ic.nc"
    fname = os.path.join(cime_xmlquery("RUNDIR"), tracer_ic_fname_rel)
    caller = __name__ + "._comp_fcn_pre_modelrun"
    ms_in.dump(fname, caller)

    # ensure certain env xml vars are set properly
    cime_xmlchange("POP_PASSIVE_TRACER_RESTART_OVERRIDE", tracer_ic_fname_rel)
    cime_xmlchange("CONTINUE_RUN", "FALSE")

    # copy rpointer files to rundir
    rundir = cime_xmlquery("RUNDIR")
    for src in glob.glob(os.path.join(get_modelinfo("rpointer_dir"), "rpointer.*")):
        shutil.copy(src, rundir)

    # generate post-modelrun script and point POSTRUN_SCRIPT to it
    # this will propagate cfg_fname and hist_fname across model run
    post_modelrun_script_fname = os.path.join(
        solver_state.get_workdir(), "post_modelrun.sh"
    )
    _gen_post_modelrun_script(post_modelrun_script_fname)
    cime_xmlchange("POSTRUN_SCRIPT", post_modelrun_script_fname)

    # set model duration parameters
    for xml_varname in ["STOP_OPTION", "STOP_N", "RESUBMIT"]:
        cime_xmlchange(xml_varname, get_modelinfo(xml_varname))

    # submit the model run and exit
    cime_case_submit()

    solver_state.log_step(fcn_complete_step)

    logger.debug("raising SystemExit")
    raise SystemExit


def _gen_post_modelrun_script(script_fname):
    """
    generate script that will be called by cime after the model run
    script_fname is called by CIME, and submits invoker_script_fname
        with the command batch_cmd_script (which can be an empty string)
    """
    batch_cmd_script = get_modelinfo("batch_cmd_script")
    if batch_cmd_script is not None:
        batch_cmd_script = batch_cmd_script.replace("\n", " ").replace("\r", " ")
    invoker_script_fname = get_modelinfo("invoker_script_fname")
    with open(script_fname, mode="w") as fptr:
        fptr.write("#!/bin/bash\n")
        fptr.write("if ./xmlquery --value RESUBMIT | grep -q '^0$'; then\n")
        fptr.write("    # forward run is done, reinvoke solver\n")
        if batch_cmd_script is None:
            fptr.write("    %s --resume\n" % invoker_script_fname)
        else:
            fptr.write(
                "    %s %s --resume\n" % (batch_cmd_script, invoker_script_fname)
            )
        fptr.write("else\n")
        fptr.write("    # set POP_PASSIVE_TRACER_RESTART_OVERRIDE for resubmit\n")
        fptr.write("    ./xmlchange POP_PASSIVE_TRACER_RESTART_OVERRIDE=none\n")
        fptr.write("fi\n")

    # ensure script_fname is executable by the user, while preserving other permissions
    fstat = os.stat(script_fname)
    os.chmod(script_fname, fstat.st_mode | stat.S_IXUSR)


def _gen_hist(hist_fname):
    """generate history file corresponding to just completed model run"""

    if hist_fname is None:
        return

    # initial implementation only works for annual mean output
    # confirm that this is the case
    if _pop_nl_var_exists("tavg_freq_opt(1)"):
        tavg_freq_opt_0 = _get_pop_nl_var("tavg_freq_opt(1)").split()[0].split("'")[1]
    else:
        tavg_freq_opt_0 = _get_pop_nl_var("tavg_freq_opt").split()[0].split("'")[1]
    if tavg_freq_opt_0 not in ["nyear", "nmonth"]:
        msg = "tavg_freq_opt_0 = %s not implemented" % tavg_freq_opt_0
        raise NotImplementedError(msg)

    tavg_freq_0 = _get_pop_nl_var("tavg_freq").split()[0]
    if tavg_freq_0 != "1":
        msg = "tavg_freq_0 = %s not implemented" % tavg_freq_0
        raise NotImplementedError(msg)

    # get starting year and month
    if cime_xmlquery("RUN_TYPE") == "branch":
        date0 = cime_xmlquery("RUN_REFDATE")
    else:
        date0 = cime_xmlquery("RUN_STARTDATE")
    (yr_str, mon_str, day_str) = date0.split("-")

    # basic error checking

    if day_str != "01":
        msg = "initial day = %s not implemented" % day_str
        raise NotImplementedError(msg)

    if tavg_freq_opt_0 == "nyear" and mon_str != "01":
        msg = "initial month = %s not implemented for nyear tavg output" % mon_str
        raise NotImplementedError(msg)

    # location of history files
    if cime_xmlquery("DOUT_S") == "TRUE":
        hist_dir = os.path.join(cime_xmlquery("DOUT_S_ROOT"), "ocn", "hist")
    else:
        hist_dir = cime_xmlquery("RUNDIR")

    caller = "src.cime_pop.newton_fcn._gen_hist"
    if tavg_freq_opt_0 == "nyear":
        fname_fmt = cime_xmlquery("CASE") + ".pop.h.{year:04d}.nc"
        ann_files_to_mean_file(
            hist_dir, fname_fmt, int(yr_str), cime_yr_cnt(), hist_fname, caller
        )

    if tavg_freq_opt_0 == "nmonth":
        fname_fmt = cime_xmlquery("CASE") + ".pop.h.{year:04d}-{month:02d}.nc"
        mon_files_to_mean_file(
            hist_dir,
            fname_fmt,
            int(yr_str),
            int(mon_str),
            12 * cime_yr_cnt(),
            hist_fname,
            caller,
        )


def _comp_fcn_post_modelrun(ms_in):
    """post-modelrun step of evaluting the function being solved with Newton's method"""

    # determine name of end of run restart file from POP's rpointer file
    rpointer_fname = os.path.join(cime_xmlquery("RUNDIR"), "rpointer.ocn.restart")
    with open(rpointer_fname, mode="r") as fptr:
        rest_file_fname_rel = fptr.readline().strip()
    fname = os.path.join(cime_xmlquery("RUNDIR"), rest_file_fname_rel)

    return ModelState(TracerModuleState, fname) - ms_in


def _apply_precond_jacobian_pre_solve_lin_eqns(ms_in, res_fname, solver_state):
    """
    pre-solve_lin_eqns step of apply_precond_jacobian
    produce computing environment for solve_lin_eqns
    If batch_cmd_precond is non-empty, submit a batch job using that command and exit.
    Otherwise, just return.
    """
    logger = logging.getLogger(__name__)
    logger.debug('res_fname="%s"', res_fname)

    fcn_complete_step = (
        "_apply_precond_jacobian_pre_solve_lin_eqns complete for %s" % res_fname
    )
    if solver_state.step_logged(fcn_complete_step):
        logger.debug('"%s" logged, returning', fcn_complete_step)
        return
    logger.debug('"%s" not logged, proceeding', fcn_complete_step)

    if get_modelinfo("batch_cmd_precond"):
        # precond_task_cnt = int(get_modelinfo("precond_task_cnt"))
        # precond_cpus_per_node = int(get_modelinfo("precond_cpus_per_node"))
        # precond_node_cnt = int(math.ceil(precond_task_cnt / precond_cpus_per_node))

        # determine node_cnt and cpus_per_node
        ocn_grid = cime_xmlquery("OCN_GRID")
        gigabyte_per_node = int(get_modelinfo("gigabyte_per_node"))
        cpus_per_node_max = int(get_modelinfo("cpus_per_node_max"))

        cpus_per_node_list = []
        for matrix_name in ms_in.precond_matrix_list():
            matrix_def = get_precond_matrix_def(matrix_name)
            matrix_solve_opts = matrix_def["precond_matrices_solve_opts"][ocn_grid]
            # account for 1 task/node having increased memory usage (25%)
            gigabyte_per_task = matrix_solve_opts["gigabyte_per_task"]
            cpus_per_node = int(gigabyte_per_node / gigabyte_per_task - 0.25)
            cpus_per_node = min(cpus_per_node_max, cpus_per_node)
            cpus_per_node_list.append(cpus_per_node)
        cpus_per_node = min(cpus_per_node_list)

        # round down to nearest power of 2
        # seems to have (unexplained) performance benefit
        cpus_per_node = 2 ** int(math.log2(cpus_per_node))

        node_cnt_list = []
        for matrix_name in ms_in.precond_matrix_list():
            matrix_def = get_precond_matrix_def(matrix_name)
            matrix_solve_opts = matrix_def["precond_matrices_solve_opts"][ocn_grid]
            task_cnt = matrix_solve_opts["task_cnt"]
            node_cnt = int(math.ceil(task_cnt / cpus_per_node))
            node_cnt_list.append(node_cnt)
        node_cnt = max(node_cnt_list)

        opt_str_subs = {"node_cnt": node_cnt, "cpus_per_node": cpus_per_node}
        batch_cmd = (
            get_modelinfo("batch_cmd_precond").replace("\n", " ").replace("\r", " ")
        )
        cmd = "%s %s --resume" % (
            batch_cmd.format(**opt_str_subs),
            get_modelinfo("invoker_script_fname"),
        )
        logger.info('cmd="%s"', cmd)
        subprocess.run(cmd, check=True, shell=True)
        solver_state.log_step(fcn_complete_step)
        logger.debug("raising SystemExit")
        raise SystemExit

    solver_state.log_step(fcn_complete_step)


def _apply_precond_jacobian_solve_lin_eqns(
    ms_in, precond_fname, res_fname, solver_state
):
    """
    solve_lin_eqns step of apply_precond_jacobian
    """
    logger = logging.getLogger(__name__)
    logger.debug('precond_fname="%s", res_fname="%s"', precond_fname, res_fname)

    fcn_complete_step = (
        "_apply_precond_jacobian_solve_lin_eqns complete for %s" % res_fname
    )
    if solver_state.step_logged(fcn_complete_step):
        logger.debug('"%s" logged, returning result', fcn_complete_step)
        return ModelState(TracerModuleState, res_fname)
    logger.debug('"%s" not logged, proceeding', fcn_complete_step)

    caller = __name__ + "._apply_precond_jacobian_solve_lin_eqns"
    ms_in.dump(res_fname, caller)

    jacobian_precond_tools_dir = get_modelinfo("jacobian_precond_tools_dir")

    tracer_names_all = ms_in.tracer_names()

    ocn_grid = cime_xmlquery("OCN_GRID")

    for (
        matrix_name,
        tracer_names_subset,
    ) in ms_in.tracer_names_per_precond_matrix().items():
        matrix_def = get_precond_matrix_def(matrix_name)
        matrix_solve_opts = matrix_def["precond_matrices_solve_opts"][ocn_grid]
        task_cnt = matrix_solve_opts["task_cnt"]
        nprow, npcol = _matrix_block_decomp(task_cnt)

        matrix_fname = os.path.join(
            solver_state.get_workdir(), "matrix_" + matrix_name + ".nc"
        )
        # split mpi_cmd, in case it has spaces because of arguments
        cmd = get_modelinfo("mpi_cmd").split()
        cmd.extend(
            [
                os.path.join(jacobian_precond_tools_dir, "bin", "solve_ABdist"),
                "-D1",
                "-n",
                "%d,%d" % (nprow, npcol),
                "-v",
                tracer_names_list_to_str(tracer_names_subset),
                matrix_fname,
                res_fname,
            ]
        )
        logger.info('cmd="%s"', " ".join(cmd))
        subprocess.run(cmd, check=True)

        _apply_tracers_sflux_term(
            tracer_names_subset, tracer_names_all, precond_fname, res_fname
        )

    ms_res = ModelState(TracerModuleState, res_fname)

    solver_state.log_step(fcn_complete_step)

    return ms_res


def _matrix_block_decomp(precond_task_cnt):
    """determine size of decomposition to be used in matrix factorization"""
    log2_precond_task_cnt = round(math.log2(precond_task_cnt))
    if 2 ** log2_precond_task_cnt != precond_task_cnt:
        msg = "precond_task_cnt must be a power of 2"
        raise ValueError(msg)
    if (log2_precond_task_cnt % 2) == 0:
        nprow = 2 ** (log2_precond_task_cnt // 2)
        npcol = nprow
    else:
        nprow = 2 ** ((log2_precond_task_cnt - 1) // 2)
        npcol = 2 * nprow
    return nprow, npcol


def tracer_names_list_to_str(tracer_names_list):
    """comma separated string of tracers being solved for"""
    return ",".join([tracer_name + "_CUR" for tracer_name in tracer_names_list])


def _apply_tracers_sflux_term(
    tracer_names_subset, tracer_names_all, precond_fname, res_fname
):
    """
    apply surface flux term of tracers in tracer_names_subset to subsequent tracer_names
    """
    logger = logging.getLogger(__name__)
    logger.debug(
        'tracer_names_subset="%s", precond_fname="%s", res_fname="%s"',
        tracer_names_subset,
        precond_fname,
        res_fname,
    )
    model_state = ModelState(TracerModuleState, res_fname)
    term_applied = False
    delta_time = 365.0 * 86400.0 * cime_yr_cnt()
    with Dataset(precond_fname, mode="r") as fptr:
        fptr.set_auto_mask(False)
        for tracer_name_src in tracer_names_subset:
            for tracer_name_dst_ind in range(
                tracer_names_all.index(tracer_name_src) + 1, len(tracer_names_all)
            ):
                tracer_name_dst = tracer_names_all[tracer_name_dst_ind]
                partial_deriv_var_name = (
                    "d_SF_" + tracer_name_dst + "_d_" + tracer_name_src + "_avg"
                )
                if partial_deriv_var_name in fptr.variables:
                    logger.info('applying "%s"', partial_deriv_var_name)
                    partial_deriv = fptr.variables[partial_deriv_var_name]
                    # get values, replacing _FillValue values with 0.0
                    if hasattr(partial_deriv, "_FillValue"):
                        fill_value = getattr(partial_deriv, "_FillValue")
                        partial_deriv_vals = np.where(
                            partial_deriv[:] == fill_value, 0.0, partial_deriv[:]
                        )
                    src = model_state.get_tracer_vals(tracer_name_src)
                    dst = model_state.get_tracer_vals(tracer_name_dst)
                    dst[0, :] -= (
                        delta_time
                        / fptr.variables["dz"][0]
                        * partial_deriv_vals
                        * src[0, :]
                    )
                    model_state.set_tracer_vals(tracer_name_dst, dst)
                    term_applied = True
    if term_applied:
        caller = __name__ + "._apply_tracers_sflux_term"
        model_state.dump(res_fname, caller)


def _pop_nl_var_exists(var_name):
    """
    does var_name exist as a variable in the pop namelist
    """
    nl_fname = os.path.join(get_modelinfo("caseroot"), "CaseDocs", "pop_in")
    cmd = ["grep", "-q", "^ *" + var_name + " *=", nl_fname]
    return subprocess.call(cmd) == 0


def _get_pop_nl_var(var_name):
    """
    extract the value(s) of a pop namelist variable
    return contents to the right of the '=' character,
        after stripping leading and trailing whitespace, and replacing ',' with ' '
    can lead to unexpected results if the rhs has strings with commas
    does not handle multiple matches of var_name in pop_in
    """
    nl_fname = os.path.join(get_modelinfo("caseroot"), "CaseDocs", "pop_in")
    cmd = ["grep", "^ *" + var_name + " *=", nl_fname]
    line = subprocess.check_output(cmd).decode()
    return line.split("=")[1].strip().replace(",", " ")


if __name__ == "__main__":
    main(_parse_args())
