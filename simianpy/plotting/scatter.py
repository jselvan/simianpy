import matplotlib.pyplot as plt
import numpy as np

from simianpy.plotting import Histogram


class Scatter:
    default_rect_scatter = 0.1, 0.1, 0.7, 0.7
    default_spacing = 0.005
    default_figsize = (8, 8)
    default_nbins = 20
    axes_names = "ax_scatter", "ax_histx", "ax_histy"

    def __init__(self, x, y, hist="xy", **kwargs):
        if hist != "xy":
            raise NotImplementedError('Only support for hist=="xy"')
        histx = "x" in hist
        histy = "y" in hist
        if not all(ax in kwargs for ax in self.axes_names):
            kwargs.update(self.get_axes())
        
        self.ax_scatter, self.ax_histx, self.ax_histy = (
            kwargs["ax_scatter"],
            kwargs["ax_histx"],
            kwargs["ax_histy"],
        )
        self.ax_scatter.sharex(self.ax_histx)
        self.ax_scatter.sharey(self.ax_histy)
        self.ax_scatter.tick_params(direction="in", top=True, right=True)
        self.ax_histx.tick_params(direction="in", labelbottom=False)
        self.ax_histy.tick_params(direction="in", labelleft=False)
        self.ax_scatter.scatter(
            x, y, label=kwargs.get("label"), **kwargs.get("scatter_kwargs", {})
        )

        xbins = kwargs.get("xbins", np.linspace(x.min(), x.max(), self.default_nbins))
        ybins = kwargs.get("ybins", np.linspace(y.min(), y.max(), self.default_nbins))
        self.ax_scatter.set_xlim(xbins.min(), xbins.max())
        self.ax_scatter.set_ylim(ybins.min(), ybins.max())

        hist_kwargs = kwargs.get("hist_kwargs", {})
        hist_params = {
            k: hist_kwargs.pop(k)
            for k in ("density", "proportion", "multiplier", "invert")
            if k in hist_kwargs
        }
        hist_params["params"] = hist_kwargs.copy()
        Histogram(x, bins=xbins, ax=self.ax_histx, **hist_params)
        hist_kwargs["orientation"] = "horizontal"
        hist_params["params"] = hist_kwargs
        Histogram(y, bins=ybins, ax=self.ax_histy, **hist_params)

        if "xlabel" in kwargs:
            self.ax_scatter.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            self.ax_scatter.set_ylabel(kwargs["ylabel"])

    @classmethod
    def get_axes(cls):
        left, bottom, width, height = cls.default_rect_scatter
        spacing = cls.default_spacing
        figsize = cls.default_figsize
        rect_scatter = [left, bottom, width, height]
        rect_histx = [
            left,
            bottom + height + spacing,
            width,
            1 - (bottom + height + 2 * spacing),
        ]
        rect_histy = [
            left + width + spacing,
            bottom,
            1 - (left + width + 2 * spacing),
            height,
        ]
        plt.figure(figsize=figsize)
        ax_scatter = plt.axes(rect_scatter)
        ax_histx = plt.axes(rect_histx)
        ax_histy = plt.axes(rect_histy)
        return dict(ax_scatter=ax_scatter, ax_histx=ax_histx, ax_histy=ax_histy)

    @classmethod
    def from_dataframe(cls, x, y, data, hist="xy", **kwargs):
        x, y = data[x], data[y]
        return cls(x, y, hist, **kwargs)


# # definitions for the axes
# left, width = 0.1, 0.65
# bottom, height = 0.1, 0.65
# spacing = 0.005

# rect_scatter = [left, bottom, width, height]
# rect_histx = [left, bottom + height + spacing, width, 0.2]
# rect_histy = [left + width + spacing, bottom, 0.2, height]

# # start with a rectangular Figure
# plt.figure(figsize=(8, 8))

# ax_scatter = plt.axes(rect_scatter)
# ax_scatter.tick_params(direction='in', top=True, right=True)
# ax_histx = plt.axes(rect_histx)
# ax_histx.tick_params(direction='in', labelbottom=False)
# ax_histy = plt.axes(rect_histy)
# ax_histy.tick_params(direction='in', labelleft=False)

# # the scatter plot:
# ax_scatter.scatter(x, y)

# # now determine nice limits by hand:
# binwidth = 0.25
# lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
# ax_scatter.set_xlim((-lim, lim))
# ax_scatter.set_ylim((-lim, lim))

# bins = np.arange(-lim, lim + binwidth, binwidth)
# ax_histx.hist(x, bins=xbins)
# ax_histy.hist(y, bins=ybins, orientation='horizontal')

# ax_histx.set_xlim(ax_scatter.get_xlim())
# ax_histy.set_ylim(ax_scatter.get_ylim())

# plt.show()
