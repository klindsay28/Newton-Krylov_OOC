"""py_driver_2d model specifics for TracerModuleStateBase"""

import logging

import numpy as np
from netCDF4 import Dataset
from scipy import sparse

from ..region_scalars import RegionScalars
from ..tracer_module_state_base import TracerModuleStateBase
from ..utils import (
    create_dimensions_verify,
    create_vars,
    datatype_sname,
    extract_dimensions,
    units_str_format,
)


class TracerModuleState(TracerModuleStateBase):
    """
    Derived class for representing a collection of model tracers.
    It implements _read_vals and dump.
    """

    def __init__(self, tracer_module_name, fname, model_config_obj, depth, ypos):

        if RegionScalars.region_cnt != 1:
            raise NotImplementedError("region_cnt > 1 not implemented")

        self.depth = depth
        self.ypos = ypos

        super().__init__(tracer_module_name, fname, model_config_obj)

    def _read_vals(self, fname):
        """return tracer values and dimension names and lengths, read from fname)"""
        logger = logging.getLogger(__name__)
        logger.debug('tracer_module_name="%s", fname="%s"', self.name, fname)
        if fname == "zeros":
            tracers_metadata = self._tracer_module_def["tracers"]
            vals = np.zeros((len(tracers_metadata), len(self.depth), len(self.ypos)))
            return vals, {"depth": len(self.depth), "ypos": len(self.ypos)}
        if fname == "gen_init_iterate":
            tracers_metadata = self._tracer_module_def["tracers"]
            vals = np.empty((len(tracers_metadata), len(self.depth), len(self.ypos)))
            for tracer_ind, tracer_metadata in enumerate(tracers_metadata.values()):
                if "init_iterate_vals" in tracer_metadata:
                    column_vals = np.interp(
                        self.depth.mid,
                        tracer_metadata["init_iterate_val_depths"],
                        tracer_metadata["init_iterate_vals"],
                    )
                    vals[tracer_ind, :] = column_vals[:, np.newaxis]
                elif "shadows" in tracer_metadata:
                    shadowed_tracer = tracer_metadata["shadows"]
                    shadow_tracer_metadata = tracers_metadata[shadowed_tracer]
                    column_vals = np.interp(
                        self.depth.mid,
                        shadow_tracer_metadata["init_iterate_val_depths"],
                        shadow_tracer_metadata["init_iterate_vals"],
                    )
                    vals[tracer_ind, :] = column_vals[:, np.newaxis]
                else:
                    msg = (
                        "gen_init_iterate failure for %s"
                        % self.tracer_names()[tracer_ind]
                    )
                    raise ValueError(msg)
            return vals, {"depth": len(self.depth), "ypos": len(self.ypos)}
        with Dataset(fname, mode="r") as fptr:
            fptr.set_auto_mask(False)
            # get dimensions from first variable
            varname = self.tracer_names()[0]
            dimensions = extract_dimensions(fptr, varname)
            # all tracers are stored in a single array
            # tracer index is the leading index
            vals = np.empty((self.tracer_cnt,) + tuple(dimensions.values()))
            # check that all vars have the same dimensions
            for tracer_name in self.tracer_names():
                if extract_dimensions(fptr, tracer_name) != dimensions:
                    msg = (
                        "not all vars have same dimensions"
                        ", tracer_module_name=%s, fname=%s" % (self.name, fname)
                    )
                    raise ValueError(msg)
            # read values
            if len(dimensions) > 3:
                msg = (
                    "ndim too large (for implementation of dot_prod)"
                    "tracer_module_name=%s, fname=%s, ndim=%s"
                    % (self.name, fname, len(dimensions))
                )
                raise ValueError(msg)
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                var = fptr.variables[tracer_name]
                vals[tracer_ind, :] = var[:]
        return vals, dimensions

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleState object
        to an open file
        """
        if action == "define":
            create_dimensions_verify(fptr, self._dimensions)
            for axis in [self.depth, self.ypos]:
                create_dimensions_verify(fptr, axis.dump_dimensions())
                if axis.axisname not in fptr.variables:
                    create_vars(fptr, axis.dump_vars_metadata())
            # define all tracers
            vars_metadata = {}
            dimnames = tuple(self._dimensions.keys())
            for tracer_name in self.tracer_names():
                vars_metadata[tracer_name] = {"dimensions": dimnames}
            create_vars(fptr, vars_metadata)
        elif action == "write":
            for axis in [self.depth, self.ypos]:
                axis.dump_write(fptr)
            # write all tracers
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                fptr.variables[tracer_name][:] = self._vals[tracer_ind, :]
        else:
            msg = "unknown action=%s", action
            raise ValueError(msg)
        return self

    def comp_tend(self, time, tracer_vals, processes):
        """
        compute tendency of tracers
        tendency units are tr_units / s
        """
        shape = (self.tracer_cnt, len(self.depth), len(self.ypos))
        tracer_vals_3d = tracer_vals.reshape(shape)
        tracer_tend_vals_3d = np.zeros(shape)
        for process in processes.values():
            tracer_tend_vals_3d += process.comp_tend(time, tracer_vals_3d)
        return tracer_tend_vals_3d.reshape(-1)

    def hist_vars_metadata(self):
        """return dict of metadata for vars to appear in the hist file"""
        res = {}
        for (
            tracer_like_name,
            tracer_metadata,
        ) in self.hist_vars_metadata_tracer_like().items():
            # tracer itself
            varname = tracer_like_name
            res[varname] = {
                "dimensions": ("time", "depth", "ypos"),
                "attrs": tracer_metadata["attrs"].copy(),
            }

            # mean in time
            varname = tracer_like_name + "_time_mean"
            res[varname] = {
                "dimensions": ("depth", "ypos"),
                "attrs": tracer_metadata["attrs"].copy(),
            }
            res[varname]["attrs"]["long_name"] += ", time mean"

            # anomaly in time
            varname = tracer_like_name + "_time_anom"
            res[varname] = {
                "dimensions": ("time", "depth", "ypos"),
                "attrs": tracer_metadata["attrs"].copy(),
            }
            res[varname]["attrs"]["long_name"] += ", time anomaly"

            # std dev in time
            varname = tracer_like_name + "_time_std"
            res[varname] = {
                "dimensions": ("depth", "ypos"),
                "attrs": tracer_metadata["attrs"].copy(),
            }
            res[varname]["attrs"]["long_name"] += ", time std dev"

            # end state minus start state
            varname = tracer_like_name + "_time_delta"
            res[varname] = {
                "dimensions": ("depth", "ypos"),
                "attrs": tracer_metadata["attrs"].copy(),
            }
            res[varname]["attrs"]["long_name"] += ", end state minus start state"

            # depth integral
            varname = tracer_like_name + "_depth_int"
            res[varname] = {
                "dimensions": ("time", "ypos"),
                "attrs": tracer_metadata["attrs"].copy(),
            }
            res[varname]["attrs"]["long_name"] += ", depth integral"
            units_str = " ".join(
                ["(", res[varname]["attrs"]["units"], ")", "(", self.depth.units, ")"]
            )
            res[varname]["attrs"]["units"] = units_str_format(units_str)

            # mean in ypos
            varname = tracer_like_name + "_ypos_mean"
            res[varname] = {
                "dimensions": ("time", "depth"),
                "attrs": tracer_metadata["attrs"].copy(),
            }
            res[varname]["attrs"]["long_name"] += ", ypos mean"

            # depth-ypos integral
            varname = tracer_like_name + "_depth_ypos_int"
            res[varname] = {
                "dimensions": ("time"),
                "attrs": tracer_metadata["attrs"].copy(),
            }
            res[varname]["attrs"]["long_name"] += ", depth-ypos integral"
            units_str = " ".join(
                [
                    "(",
                    res[varname]["attrs"]["units"],
                    ")",
                    "(",
                    self.depth.units,
                    ")",
                    "(",
                    self.ypos.units,
                    ")",
                ]
            )
            res[varname]["attrs"]["units"] = units_str_format(units_str)
        return res

    def hist_vars_metadata_tracer_like(self):
        """return dict of metadata for tracer-like vars to appear in the hist file"""
        res = {}
        for tracer_name, tracer_metadata in self._tracer_module_def["tracers"].items():
            res[tracer_name] = {"attrs": tracer_metadata["attrs"]}
        return res

    @staticmethod
    def hist_time_mean_weights(fptr):
        """return weights for computing time-mean in hist file"""
        # downweight endpoints because py_driver_2d writes t=0 and t=end to hist
        timelen = len(fptr.dimensions["time"])
        weights = np.full(timelen, 1.0 / (timelen - 1))
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return weights

    def write_hist_vars(self, fptr, tracer_vals_all):
        """write hist vars"""

        time_weights = self.hist_time_mean_weights(fptr)

        for ind, tracer_like_name in enumerate(self.hist_vars_metadata_tracer_like()):
            tracer_vals = np.moveaxis(tracer_vals_all[ind, :], -1, 0)

            # tracer itself
            varname = tracer_like_name
            fptr.variables[varname][:] = tracer_vals

            # mean in time
            varname = tracer_like_name + "_time_mean"
            tracer_vals_mean = np.einsum("i,i...", time_weights, tracer_vals)
            fptr.variables[varname][:] = tracer_vals_mean

            # anomaly in time
            varname = tracer_like_name + "_time_anom"
            tracer_vals_anom = tracer_vals - tracer_vals_mean
            fptr.variables[varname][:] = tracer_vals_anom

            # std dev in time
            varname = tracer_like_name + "_time_std"
            tracer_vals_var = np.einsum("i,i...", time_weights, tracer_vals_anom**2)
            fptr.variables[varname][:] = np.sqrt(tracer_vals_var)

            # end state minus start state
            varname = tracer_like_name + "_time_delta"
            fptr.variables[varname][:] = tracer_vals[-1, :] - tracer_vals[0, :]

            # depth integral
            varname = tracer_like_name + "_depth_int"
            fptr.variables[varname][:] = self.depth.int_vals_mid(tracer_vals, axis=-2)

            # ypos mean
            varname = tracer_like_name + "_ypos_mean"
            fptr.variables[varname][:] = self.ypos.int_vals_mid(tracer_vals, axis=-1)
            fptr.variables[varname][:] /= self.ypos.edges.max() - self.ypos.edges.min()

            # depth-ypos integral
            varname = tracer_like_name + "_depth_ypos_int"
            fptr.variables[varname][:] = self.depth.int_vals_mid(
                self.ypos.int_vals_mid(tracer_vals, axis=-1), axis=-1
            )

    def comp_jacobian(self, time, tracer_vals, processes):
        """
        compute jacobian of tracer tendencies from processes
        jacobian units are 1 / s
        """
        jacobian = sparse.csr_matrix((tracer_vals.size, tracer_vals.size))
        for process in processes.values():
            jacobian += process.comp_jacobian(time, self.tracer_cnt)
        return jacobian

    def comp_jacobian_sparsity(self, time, tracer_vals, processes):
        """
        return sparse matrix with sparsity pattern of jacobian from processes
        """
        jacobian = self.comp_jacobian(time, tracer_vals, processes)
        (row_ind, col_ind, _) = sparse.find(jacobian)
        data = np.ones(row_ind.shape)
        return sparse.csr_matrix((data, (row_ind, col_ind)))

    def stats_dimensions(self, fptr):
        """return dimensions to be used in stats file for this tracer module"""
        res = self.depth.dump_dimensions()
        res.update(self.ypos.dump_dimensions())
        return res

    def stats_vars_metadata(self, fptr_hist):
        """
        return dict of metadata for vars to appear in the stats file for this tracer
        module
        """
        res = self.depth.dump_vars_metadata()
        res.update(self.ypos.dump_vars_metadata())

        # add metadata for tracer-like variables

        for tracer_name in self.stats_vars_tracer_like():
            datatype = datatype_sname(fptr_hist.variables[tracer_name])

            attrs = fptr_hist.variables[tracer_name].__dict__
            del attrs["cell_methods"]

            # tracer itself
            res[tracer_name] = {
                "datatype": datatype,
                "dimensions": ("iteration", "region", "depth", "ypos"),
                "attrs": attrs,
            }

            # ypos average
            varname_stats = "_".join([tracer_name, "mean", "ypos"])
            res[varname_stats] = {
                "datatype": datatype,
                "dimensions": ("iteration", "region", "depth"),
                "attrs": attrs,
            }
        return res

    def stats_vars_vals_iteration_invariant(self, fptr_hist):
        """return iteration-invariant tracer module specific stats variables"""
        res = self.depth.dump_vals_dict()
        res.update(self.ypos.dump_vals_dict())
        return res

    def stats_vars_vals(self, fptr_hist):
        """return tracer module specific stats variables for the current iteration"""

        # return values for tracer-like variables
        time_weights = self.hist_time_mean_weights(fptr_hist)
        ypos_weights = self.ypos.delta.copy()
        ypos_weights /= ypos_weights.sum()
        res = {}
        for tracer_name in self.stats_vars_tracer_like():
            tracer_vals = fptr_hist.variables[tracer_name][:]
            # tracer itself
            # assume region dimension has length 1
            res[tracer_name] = np.einsum("i,i...", time_weights, tracer_vals)

            # ypos average
            varname_stats = "_".join([tracer_name, "mean", "ypos"])
            res[varname_stats] = np.einsum("j,...j", ypos_weights, res[tracer_name])
        return res
