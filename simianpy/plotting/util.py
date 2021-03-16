import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_ax(ax):
    if ax is None:
        _, ax = plt.subplots()
    elif ax=='hold':
        ax = plt.gca()
    return ax

def ax_formatting(ax, **kwargs):
    xlabel = kwargs.get('xlabel')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    ylabel = kwargs.get('ylabel')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    ax.set_xlim(kwargs.get('xlim'))
    ax.set_ylim(kwargs.get('ylim'))

def add_colorbar(im, orientation='vertical'):
    fig, ax = im.get_figure(), im.axes
    if orientation=='vertical':
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    elif orientation=='horizontal':
        cax = make_axes_locatable(ax).append_axes('bottom', size='5%', pad=0.05)
    else:
        raise ValueError("orientation must be one of ['vertical', 'horizontal']")

    cbar = fig.colorbar(im, cax=cax, orientation=orientation)
    return cbar