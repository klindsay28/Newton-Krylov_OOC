"""Base class of methods related to problem being solved with Newton's method"""

import numpy as np
from netCDF4 import Dataset

from model import get_modelinfo, get_tracer_module_def, get_precond_matrix_def

class NewtonFcnBase():
    """Base class of methods related to problem being solved with Newton's method"""

    def gen_precond_jacobian(self, hist_fname, precond_fname, solver_state):
        """Generate file(s) needed for preconditioner of jacobian of comp_fcn."""
        hist_vars = self._hist_vars_for_precond_list()

        with Dataset(hist_fname, 'r') as fptr_in, Dataset(precond_fname, 'w') as fptr_out:
            # define output vars
            self._def_precond_dims_and_coord_vars(hist_vars, fptr_in, fptr_out)

            for hist_var in hist_vars:
                hist_var_name, _, time_op = hist_var.partition(':')
                hist_var = fptr_in.variables[hist_var_name]

                if time_op == 'avg':
                    precond_var = fptr_out.createVariable(hist_var_name+'_avg',
                                                          hist_var.datatype,
                                                          dimensions=hist_var.dimensions[1:])
                    precond_var.long_name = hist_var.long_name+', avg over time dim'
                    precond_var[:] = hist_var[:].mean(axis=0)
                elif time_op == 'log_avg':
                    precond_var = fptr_out.createVariable(hist_var_name+'_log_avg',
                                                          hist_var.datatype,
                                                          dimensions=hist_var.dimensions[1:])
                    precond_var.long_name = hist_var.long_name+', log avg over time dim'
                    precond_var[:] = np.exp(np.log(hist_var[:]).mean(axis=0))
                else:
                    precond_var = fptr_out.createVariable(hist_var_name,
                                                          hist_var.datatype,
                                                          dimensions=hist_var.dimensions)
                    precond_var.long_name = hist_var.long_name
                    precond_var[:] = hist_var[:]

                for att_name in ['units', 'coordinates', 'positive']:
                    try:
                        setattr(precond_var, att_name, getattr(hist_var, att_name))
                    except AttributeError:
                        pass

    def _hist_vars_for_precond_list(self):
        """Return list of hist vars needed for preconditioner of jacobian of comp_fcn"""
        res = []
        for matrix_name in self._precond_matrix_list():
            res.extend(get_precond_matrix_def(matrix_name)['hist_to_precond_var_names'])
        return res

    def _precond_matrix_list(self):
        """Return list of precond matrices being used"""
        res = ['base']
        for tracer_module_name in get_modelinfo('tracer_module_names').split(','):
            tracer_module_def = get_tracer_module_def(tracer_module_name)
            res.extend(tracer_module_def['precond_matrices'].values())
        return res

    def _def_precond_dims_and_coord_vars(self, hist_vars, fptr_in, fptr_out):
        """define netCDF4 dimensions needed for hist_vars from hist_fname"""
        for hist_var in hist_vars:
            hist_var_name, _, time_op = hist_var.partition(':')
            hist_var = fptr_in.variables[hist_var_name]

            if time_op in ('avg', 'log_avg'):
                dimnames = hist_var.dimensions[1:]
            else:
                dimnames = hist_var.dimensions

            for dimname in dimnames:
                if dimname not in fptr_out.dimensions:
                    fptr_out.createDimension(dimname, fptr_in.dimensions[dimname].size)
                    # if fptr_in has a cooresponding coordinate variable, then
                    # define it, copy attributes from fptr_in, and write it
                    if dimname in fptr_in.variables:
                        fptr_out.createVariable(dimname, fptr_in.variables[dimname].datatype,
                                                dimensions=(dimname,))
                        for att_name in fptr_in.variables[dimname].ncattrs():
                            setattr(fptr_out.variables[dimname], att_name,
                                    getattr(fptr_in.variables[dimname], att_name))
                        fptr_out.variables[dimname][:] = fptr_in.variables[dimname][:]
