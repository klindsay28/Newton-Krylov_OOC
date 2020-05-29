"""Base class of methods related to problem being solved with Newton's method"""

from datetime import datetime
import logging
import os

from netCDF4 import Dataset
import numpy as np


class NewtonFcnBase:
    """Base class of methods related to problem being solved with Newton's method"""

    def comp_fcn_postprocess(self, res, res_fname, caller):
        """apply postprocessing that is common to all comp_fcn methods"""
        fcn_name = __name__ + ".NewtonFcnBase.comp_fcn_postprocess"
        caller = fcn_name + " called from " + caller
        return res.zero_extra_tracers().apply_region_mask().dump(res_fname, caller)

    def comp_jacobian_fcn_state_prod(
        self, iterate, fcn, direction, res_fname, solver_state
    ):
        """
        compute the product of the Jacobian of fcn at iterate with the model state
        direction

        assumes direction is a unit vector
        """
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s"', res_fname)

        fcn_complete_step = "comp_jacobian_fcn_state_prod complete for %s" % res_fname

        if solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return type(iterate)(iterate.tracer_module_state_class, res_fname)
        logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        sigma = 1.0e-4 * iterate.norm()

        # perturbed ModelStateBase
        perturb_ms = iterate + sigma * direction
        perturb_fcn_fname = os.path.join(
            solver_state.get_workdir(), "perturb_fcn_" + os.path.basename(res_fname)
        )
        perturb_fcn = self.comp_fcn(  # pylint: disable=E1101
            perturb_ms, perturb_fcn_fname, solver_state
        )

        # compute finite difference
        caller = __name__ + ".NewtonFcnBase.comp_jacobian_fcn_state_prod"
        res = ((perturb_fcn - fcn) / sigma).dump(res_fname, caller)

        solver_state.log_step(fcn_complete_step)

        return res

    def gen_precond_jacobian(self, iterate, hist_fname, precond_fname):
        """Generate file(s) needed for preconditioner of jacobian of comp_fcn."""
        logger = logging.getLogger(__name__)
        logger.debug('hist_fname="%s", precond_fname="%s"', hist_fname, precond_fname)

        hist_vars = iterate.hist_vars_for_precond_list()

        with Dataset(hist_fname, mode="r") as fptr_in, Dataset(
            precond_fname, "w", format="NETCDF3_64BIT_OFFSET"
        ) as fptr_out:
            datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fcn_name = __name__ + ".NewtonFcnBase.gen_precond_jacobian"
            msg = datestamp + ": created by " + fcn_name
            if hasattr(fptr_in, "history"):
                msg = msg + "\n" + getattr(fptr_in, "history")
            setattr(fptr_out, "history", msg)

            # define output vars
            self._def_precond_dims_and_coord_vars(hist_vars, fptr_in, fptr_out)

            for hist_var in hist_vars:
                hist_var_name, _, time_op = hist_var.partition(":")
                hist_var = fptr_in.variables[hist_var_name]
                logger.debug('hist_var_name="%s"', hist_var_name)

                fill_value = (
                    getattr(hist_var, "_FillValue")
                    if hasattr(hist_var, "_FillValue")
                    else None
                )

                if time_op == "avg":
                    precond_var_name = hist_var_name + "_avg"
                    if precond_var_name not in fptr_out.variables:
                        precond_var = fptr_out.createVariable(
                            hist_var_name + "_avg",
                            hist_var.datatype,
                            dimensions=hist_var.dimensions[1:],
                            fill_value=fill_value,
                        )
                        precond_var.long_name = (
                            hist_var.long_name + ", avg over time dim"
                        )
                        precond_var[:] = hist_var[:].mean(axis=0)
                elif time_op == "log_avg":
                    precond_var_name = hist_var_name + "_log_avg"
                    if precond_var_name not in fptr_out.variables:
                        precond_var = fptr_out.createVariable(
                            hist_var_name + "_log_avg",
                            hist_var.datatype,
                            dimensions=hist_var.dimensions[1:],
                            fill_value=fill_value,
                        )
                        precond_var.long_name = (
                            hist_var.long_name + ", log avg over time dim"
                        )
                        precond_var[:] = np.exp(np.log(hist_var[:]).mean(axis=0))
                else:
                    precond_var_name = hist_var_name
                    if precond_var_name not in fptr_out.variables:
                        precond_var = fptr_out.createVariable(
                            hist_var_name,
                            hist_var.datatype,
                            dimensions=hist_var.dimensions,
                            fill_value=fill_value,
                        )
                        precond_var.long_name = hist_var.long_name
                        precond_var[:] = hist_var[:]

                for att_name in ["missing_value", "units", "coordinates", "positive"]:
                    if hasattr(hist_var, att_name):
                        setattr(precond_var, att_name, getattr(hist_var, att_name))

    def _def_precond_dims_and_coord_vars(self, hist_vars, fptr_in, fptr_out):
        """define netCDF4 dimensions needed for hist_vars from hist_fname"""
        logger = logging.getLogger(__name__)
        for hist_var in hist_vars:
            hist_var_name, _, time_op = hist_var.partition(":")
            hist_var = fptr_in.variables[hist_var_name]

            if time_op in ("avg", "log_avg"):
                dimnames = hist_var.dimensions[1:]
            else:
                dimnames = hist_var.dimensions

            for dimname in dimnames:
                if dimname not in fptr_out.dimensions:
                    logger.debug('defining dimension="%s"', dimname)
                    fptr_out.createDimension(dimname, fptr_in.dimensions[dimname].size)
                    # if fptr_in has a cooresponding coordinate variable, then
                    # define it, copy attributes from fptr_in, and write it
                    if dimname in fptr_in.variables:
                        logger.debug('defining variable="%s"', dimname)
                        fptr_out.createVariable(
                            dimname,
                            fptr_in.variables[dimname].datatype,
                            dimensions=(dimname,),
                        )
                        for att_name in fptr_in.variables[dimname].ncattrs():
                            setattr(
                                fptr_out.variables[dimname],
                                att_name,
                                getattr(fptr_in.variables[dimname], att_name),
                            )
                        fptr_out.variables[dimname][:] = fptr_in.variables[dimname][:]
