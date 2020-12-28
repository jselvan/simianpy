from simianpy.plotting.util import get_ax
from simianpy.analysis.gaussian_kde_2d import GaussianKDE2D

import numpy as np

def GaussianKDE2DPlot(x, y, weights, data=None, range=None, xticks=None, yticks=None, resolution=100, bw_method=None, norm=True, ax=None, im_kwargs=dict()):
    if data is not None:
        x, y = data[x], data[y]
        if weights is not None:
            weights = data[weights].values
    
    if range is not None:
        (xmin, xmax), (ymin, ymax) = range
        xticks = np.linspace(xmin, xmax, resolution)
        yticks = np.linspace(ymin, ymax, resolution)

    if xticks is None:
        xticks = np.linspace(x.min(), x.max(), resolution)
    
    if yticks is None:
        yticks = np.linspace(y.min(), y.max(), resolution)
    
    pdf = GaussianKDE2D.from_arrays(x, y,weights=weights,bw_method=bw_method,norm=norm)
    xx, yy = np.meshgrid(xticks, yticks)
    zz = np.rot90(np.reshape(pdf((np.ravel(xx), np.ravel(yy))), xx.shape).T)
    bounds = xticks.min(), xticks.max(), yticks.min(), yticks.max()

    ax = get_ax(ax)
    im = ax.imshow(zz, extent=bounds, **im_kwargs)
    return im
