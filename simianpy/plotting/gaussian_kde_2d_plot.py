import numpy as np
import pandas as pd

from simianpy.analysis.gaussian_kde_2d import GaussianKDE2D
from simianpy.plotting.imshow import Image
from simianpy.plotting.util import get_ax


def GaussianKDE2DPlot(
    x,
    y,
    weights,
    data=None,
    range=None,
    xticks=None,
    yticks=None,
    resolution=100,
    bw_method=None,
    norm=True,
    ax=None,
    im_kwargs=dict(),
    return_density=False,
):
    if data is None:
        if weights is None:
            data = pd.DataFrame({"x": x, "y": y})
        else:
            data = pd.DataFrame({"x": x, "y": y, "weights": weights})
            weights = "weights"
        x, y = "x", "y"
    pdf = GaussianKDE2D.from_dataframe(data=data, x=x, y=y, weights=weights, norm=norm)
    density = pdf.evaluate(
        range=range,
        xticks=xticks,
        yticks=yticks,
        resolution=resolution,
        return_series=True,
    )
    im = Image("x", "y", "density", density, ax=ax, im_kwargs=im_kwargs)
    if return_density:
        return im, density
    else:
        return im
