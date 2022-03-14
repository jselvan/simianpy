"""Plotting tools

Contains
--------
    Histogram
    Regression

Modules
-------
    histogram
    regression
"""

from .catplot import Bar, CatPlot, Line, ViolinPlot
from .gaussian_kde_2d_plot import GaussianKDE2DPlot
from .histogram import Histogram
from .imshow import Image
from .regression import Regression
from .scatter import Scatter
from .spike import Raster, plot_PSTH
