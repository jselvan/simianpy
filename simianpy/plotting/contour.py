from simianpy.plotting.util import get_ax, add_colorbar, ax_formatting

default_colorbar_orientation = 'vertical'
def Contour(x, y, z, cmap='bwr', filename=None, ax=None, colorbar='vertical', ax_kwargs={}):
    ax = get_ax(ax)
    ax.set_aspect('auto')

    im = ax.contourf(x, y, z, cmap=cmap)

    if colorbar:
        if colorbar not in ['vertical', 'horizontal']:
            colorbar = default_colorbar_orientation
        add_colorbar(im, colorbar)
    
    ax_formatting(ax, **ax_kwargs)
    return im