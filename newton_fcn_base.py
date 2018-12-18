"""Base class of methods related to problem being solved with Newton's method"""

import logging

import numpy as np
from netCDF4 import Dataset

class NewtonFcnBase():
    """Base class of methods related to problem being solved with Newton's method"""

    def gen_precond_jacobian(self, iterate, hist_fname, precond_fname, solver_state):
        """Generate file(s) needed for preconditioner of jacobian of comp_fcn."""
        logger = logging.getLogger(__name__)
        logger.debug('precond_fname="%s"', precond_fname)
        hist_vars = iterate.hist_vars_for_precond_list()

        with Dataset(hist_fname, 'r') as fptr_in, Dataset(precond_fname, 'w') as fptr_out:
            # define output vars
            self._def_precond_dims_and_coord_vars(hist_vars, fptr_in, fptr_out)

            for hist_var in hist_vars:
                hist_var_name, _, time_op = hist_var.partition(':')
                hist_var = fptr_in.variables[hist_var_name]
                logger.debug('hist_var_name="%s"', hist_var_name)

                fill_value = getattr(hist_var, '_FillValue') if hasattr(hist_var, '_FillValue') \
                    else None

                if time_op == 'avg':
                    precond_var_name = hist_var_name+'_avg'
                    if precond_var_name not in fptr_out.variables:
                        precond_var = fptr_out.createVariable(hist_var_name+'_avg',
                                                              hist_var.datatype,
                                                              dimensions=hist_var.dimensions[1:],
                                                              fill_value=fill_value)
                        precond_var.long_name = hist_var.long_name+', avg over time dim'
                        precond_var[:] = hist_var[:].mean(axis=0)
                elif time_op == 'log_avg':
                    precond_var_name = hist_var_name+'_log_avg'
                    if precond_var_name not in fptr_out.variables:
                        precond_var = fptr_out.createVariable(hist_var_name+'_log_avg',
                                                              hist_var.datatype,
                                                              dimensions=hist_var.dimensions[1:],
                                                              fill_value=fill_value)
                        precond_var.long_name = hist_var.long_name+', log avg over time dim'
                        precond_var[:] = np.exp(np.log(hist_var[:]).mean(axis=0))
                else:
                    precond_var_name = hist_var_name
                    if precond_var_name not in fptr_out.variables:
                        precond_var = fptr_out.createVariable(hist_var_name,
                                                              hist_var.datatype,
                                                              dimensions=hist_var.dimensions,
                                                              fill_value=fill_value)
                        precond_var.long_name = hist_var.long_name
                        precond_var[:] = hist_var[:]

                for att_name in ['missing_value', 'units', 'coordinates', 'positive']:
                    if hasattr(hist_var, att_name):
                        setattr(precond_var, att_name, getattr(hist_var, att_name))

    def _def_precond_dims_and_coord_vars(self, hist_vars, fptr_in, fptr_out):
        """define netCDF4 dimensions needed for hist_vars from hist_fname"""
        logger = logging.getLogger(__name__)
        for hist_var in hist_vars:
            hist_var_name, _, time_op = hist_var.partition(':')
            hist_var = fptr_in.variables[hist_var_name]

            if time_op in ('avg', 'log_avg'):
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
                        fptr_out.createVariable(dimname, fptr_in.variables[dimname].datatype,
                                                dimensions=(dimname,))
                        for att_name in fptr_in.variables[dimname].ncattrs():
                            setattr(fptr_out.variables[dimname], att_name,
                                    getattr(fptr_in.variables[dimname], att_name))
                        fptr_out.variables[dimname][:] = fptr_in.variables[dimname][:]
