import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_ax(ax):
    if ax is None:
        _, ax = plt.subplots()
    elif ax == "hold":
        ax = plt.gca()
    return ax


def ax_formatting(ax, **kwargs):
    xlabel = kwargs.get("xlabel")
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ylabel = kwargs.get("ylabel")
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_xlim(kwargs.get("xlim"))
    ax.set_ylim(kwargs.get("ylim"))


def add_colorbar(im, orientation="vertical", ax=None):
    ax = ax or im.axes
    fig = ax.figure
    if orientation == "vertical":
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    elif orientation == "horizontal":
        cax = make_axes_locatable(ax).append_axes("bottom", size="5%", pad=0.05)
    elif isinstance(orientation, dict):
        cax = make_axes_locatable(ax).append_axes(**orientation)
    else:
        raise ValueError("orientation must be one of ['vertical', 'horizontal']")

    cbar = fig.colorbar(im, cax=cax, orientation=orientation)
    return cbar


def get_scalar_mappable(vmin, vmax, cmap):
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    return sm


# given a set of x or y coordinates, draw a line from that point on the axis to
# the corresponding point on the provided curve, and a line from that point to the
# corresponding point on the other axis
def draw_lines(x, y, vlines=None, hlines=None, ax=None, **kwargs):
    import numpy as np

    ax = get_ax(ax)
    coords_x = []
    coords_y = []
    line_params = kwargs.pop("line_params", {})
    if vlines is not None:
        endpoints = np.interp(vlines, x, y)
        ax.vlines(vlines, 0, endpoints, **line_params)
        ax.hlines(endpoints, 0, vlines, **line_params)
        coords_x.extend(vlines)
        coords_y.extend(endpoints)
    if hlines is not None:
        endpoints = np.interp(hlines, y, x)
        ax.hlines(hlines, 0, endpoints, **line_params)
        ax.vlines(endpoints, 0, hlines, **line_params)
        coords_x.extend(endpoints)
        coords_y.extend(hlines)
    label_template = kwargs.pop("label_template", "({x}, {y})")
    label_params = kwargs.pop("label_params", {})
    if coords_x and coords_y:
        for x, y in zip(coords_x, coords_y):
            ax.text(x, y, label_template.format(x=x, y=y), **label_params)
    return ax
