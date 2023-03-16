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
        title = varname
        if rank == 2:
            cbar_kwargs = {"orientation": "horizontal"}
            plot_da.plot(cbar_kwargs=cbar_kwargs)
        else:
            if "fcn_mean" in varname or "increment_mean" in varname:
                plot_da = abs(plot_da)
                title = f"abs({title})"
            plot_da.plot.line("-ok")
            log_substrs = ["fcn_norm", "increment_norm", "fcn_mean", "increment_mean"]
            if any([substr in varname for substr in log_substrs]):
                plt.yscale("log")
        plt.title(title)
        plt.show()
