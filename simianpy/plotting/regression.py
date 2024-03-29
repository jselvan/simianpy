import matplotlib.pyplot as plt
import numpy as np

from simianpy.analysis import LinearRegression
from simianpy.plotting.util import get_ax

try:
    import holoviews as hv
except ImportError:
    hv = None
else:
    from holoviews import dim, opts


default_params = {"matplotlib": {}, "holoviews": {}}


def _regression_holoviews(regression_output, fitline, scatter):
    if hv is None:
        raise ImportError(
            "holoviews module could not be imported. use a different engine"
        )

    plots = []

    if fitline:
        fitline_plot = hv.Curve(
            (regression_output.x_pred, regression_output.y_pred),
            regression_output.x_label,
            regression_output.y_label,
        )
        plots.append(fitline_plot)

    if scatter:
        scatter_plot = hv.Scatter(
            regression_output.data, regression_output.x_label, regression_output.y_label
        )
        plots.append(scatter_plot)

    return hv.Overlay(plots) if plots else None


def Regression(
    x,
    y,
    data=None,
    x_pred=None,
    drop_na=True,
    fitline=True,
    scatter=True,
    text=False,
    engine="matplotlib",
    ax=None,
    fitline_kwargs={},
    scatter_kwargs={},
):
    """ A convenience function for generating a regression plot

    Parameters
    ----------
    x, y, data, drop_na
        see simi.analysis.linear_regression.LinearRegression
    fitline: bool; default=True
        determines if a fitline is plotted
    scatter: bool; default=True
        determines if the original data is plotted
    engine: str; default='matplotlib'
    ax: Axes; default=None
    fitline_kwargs: dict-like; default={}
    scatter_kwargs: dict-like; default={}
    """
    regression_output = LinearRegression(x, y, data, drop_na, x_pred=x_pred)

    if engine == "holoviews":
        plot = _regression_holoviews(regression_output, fitline, scatter)
        return plot
    elif engine == "matplotlib":
        ax = get_ax(ax)
        if fitline:
            ax.plot(
                regression_output.x_pred, regression_output.y_pred, **fitline_kwargs
            )
        if scatter:
            ax.scatter(
                regression_output.data[regression_output.x_label],
                regression_output.data[regression_output.y_label],
                **scatter_kwargs,
            )
        if text:
            ax.text(
                0.05,
                0.95,
                str(regression_output).replace('; ', '\n'),
                transform=ax.transAxes,
                verticalalignment="top",
            )
        return ax
    else:
        raise ValueError(
            f"Engine not implemented: {engine}. Choose one of: ['holoviews','matplotlib']"
        )
