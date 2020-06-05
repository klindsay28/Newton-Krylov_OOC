"""utility functions for notebooks"""

import matplotlib.pyplot as plt


def plot_all_vars(ds):
    """plot all variables in an xarray Dataset"""
    for varname in ds.data_vars:
        if "bounds" in varname or varname in ["depth_delta"]:
            continue
        plot_da = (
            ds[varname].isel(time=slice(1, -1))
            if "time" in ds[varname].dims
            else ds[varname]
        )
        # remove singleton region dimension, if present
        # do not squeeze, because singleton iteration dimension is of interest
        if "region" in plot_da.dims:
            plot_da = plot_da.isel(region=0)
        rank = len(plot_da.dims)
        if rank == 2:
            cbar_kwargs = {"orientation": "horizontal"}
            plot_da.plot(cbar_kwargs=cbar_kwargs)
        else:
            plot_da.plot.line("-ok")
            if "fcn_norm" in varname or "increment_norm" in varname:
                plt.yscale("log")
            if "fcn_mean" in varname or "increment_mean" in varname:
                plt.yscale("symlog")
        plt.title(varname)
        plt.show()
