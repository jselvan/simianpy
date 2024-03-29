import numpy as np
import pandas as pd

from simianpy.plotting.imshow import Image

try:
    import holoviews as hv
except ImportError:
    hv = None
else:
    from holoviews import dim, opts


def _heatmap_holoviews(heatmap_data, x, y, z):
    if hv is None:
        raise ImportError(
            "holoviews module could not be imported. use a different engine"
        )

    heatmap = hv.HeatMap(heatmap_data.reset_index(), [f"{x}_bin", f"{y}_bin"], z)
    return heatmap


def HeatMap(
    x,
    y,
    z,
    data=None,
    bins=10,
    engine="holoviews",
    ax=None,
    im_kwargs={},
    colorbar="vertical",
    method="mean",
):
    """ A heatmap plotting utility for irregularly spaced data
    
    """  # TODO: complete docstring
    if data is None:
        data = pd.DataFrame({"x": x, "y": y, "z": z})
        x, y, z = "x", "y", "z"
    for component in [x, y]:
        var = data[component]
        if isinstance(bins, int):
            nbins = bins
            bins = np.linspace(np.floor(var.min()), np.ceil(var.max()), nbins)
        else:
            nbins = len(bins)
        data.loc[:, f"{component}_bin"] = pd.cut(var, nbins, labels=bins)
    if method == "mean":
        heatmap_data = data.groupby([f"{x}_bin", f"{y}_bin"])[z].mean()
    elif method == "sum":
        heatmap_data = data.groupby([f"{x}_bin", f"{y}_bin"])[z].sum()
    else:
        raise NotImplementedError(
            'Only "mean" and "sum" methods are currently supported'
        )

    if engine == "holoviews":
        heatmap = _heatmap_holoviews(heatmap_data, x, y, z)
    elif engine == "matplotlib":
        # TODO: test
        heatmap = Image(
            f"{x}_bin",
            f"{y}_bin",
            z,
            data=heatmap_data,
            ax=ax,
            im_kwargs=im_kwargs,
            colorbar=colorbar,
        )
    else:
        raise ValueError(
            f"Engine not implemented: {engine}. Choose one of: ['holoviews']"
        )

    return heatmap

