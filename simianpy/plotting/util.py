import matplotlib.pyplot as plt

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