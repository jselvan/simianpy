"""
Plotting tools for spiking data
===============================

Currently uses only matplotlib due to difficulty in using Holoviews' eventplot tool

Contains
--------
Raster
SpikeDensity

Examples
--------
# make random data
np.random.seed(1997)
data=np.random.randint(0,300,(50,100))
data=np.random.gamma(4,5,(50,100))
evt=np.random.randint(11,24,(50,1))
idx = (-evt).argsort(axis=None)

# Plot in overlay with twin axes
fig,ax=plt.subplots()
Raster(data[idx], ax=ax, eventplot_params=dict(color='k'))
Raster(evt[idx], ax=ax, eventplot_params=dict(color='g',linewidths=5))
SpikeDensity(data, ax=ax.twinx(), line_params=dict(color='r'), hist_params=dict(bins=20))
fig.savefig('fig.png')

## Plot in layout
fig,(ax1, ax2)=plt.subplots(2,sharex=True)
Raster(data[idx], ax=ax1, eventplot_params=dict(color='k'))
Raster(evt[idx], ax=ax1, eventplot_params=dict(color='g',linewidths=5))
SpikeDensity(data, ax=ax2, line_params=dict(color='r'), hist_params=dict(bins=20))
for spine in ['top', 'right']: ax2.spines[spine].set_visible(False)
ax1.axis('off')
fig.tight_layout()
fig.savefig('fig_.png')
"""
from .util import get_ax, ax_formatting

from collections import ChainMap

import numpy as np
import matplotlib.pyplot as plt 

# try:
#     import holoviews as hv
# except ImportError:
#     hv = None
# else:
#     from holoviews import opts, dim

# TODO: Implement holoviews
# def _holoviews():
#     overlay = hv.NdOverlay({i: hv.Spikes(np.random.randint(0, 100, 10), kdims='Time').opts(position=i)
#                         for i in range(10)}) #.opts(yticks=[((i+1)*0.1-0.05, i) for i in range(10)])
#     overlay.opts(
#         opts.Spikes(spike_length=0.01),
#         opts.NdOverlay(show_legend=False))
#     return overlay

# def _matplotlib(ax):
#     pass

default_raster_eventplot_params = dict(color='k')
default_PSTH_line_params = dict(color='r')
default_PSTH_hist_params = dict(bins=50)

def Raster(data, ax=None, eventplot_params={}, **kwargs):
    ax = get_ax(ax)
    eventplot_params = ChainMap(eventplot_params, default_raster_eventplot_params)
    ax.eventplot(data,**eventplot_params)
    return ax

def PSTH(data, ax=None, line_params={}, hist_params={}, sampling_rate=1e3, **kwargs):
    ax = get_ax(ax)
    line_params = ChainMap(line_params, default_PSTH_line_params)
    counts, bins = np.histogram(np.concatenate(data), **hist_params)
    counts = counts / (np.diff(bins)/sampling_rate) / len(data)
    xticks = np.mean([bins[:-1], bins[1:]],axis=0)
    ax.plot(xticks, counts, **line_params)
    return ax