import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

def create_jpsth_axes(jpsth_data):
    tmin, tmax = jpsth_data["t"].min(), jpsth_data["t"].max()
    xcorr_lagmin, xcorr_lagmax = jpsth_data["lags"].min(), jpsth_data["lags"].max()
    xcorr_min, xcorr_max = jpsth_data["xcorr_hist"].min()*1e3, jpsth_data["xcorr_hist"].max()*1e3

    fig = plt.figure(figsize=(12,8))
    axes = dict()
    subplot_pad = 0
    unit_x, unit_y = 1/12, 1/8
    square_size = 4
    ax_scale_ratio = 5
    square_x = unit_x*square_size
    square_y = unit_y*square_size
    expansion_x = square_x / ((1+ax_scale_ratio)*(2**0.5))
    expansion_y = square_y / ((1+ax_scale_ratio)*(2**0.5))

    rects = {
        'psth_a': [
            subplot_pad,
            unit_y,
            unit_x,
            square_y
        ],
        'psth_b': [
            unit_x,
            subplot_pad,
            square_x,
            unit_y
        ],
        'jpsth': [
            unit_x,
            unit_y,
            square_x,
            square_y
        ],
        'coincidence_hist': [
            unit_x*(square_size+1.1)-expansion_x,
            unit_y-expansion_y,
            square_x+expansion_x*2,
            square_y+expansion_y*2
        ],
        'xcorr_hist': [
            unit_x*(1.5*square_size+1.1)+2*subplot_pad-expansion_x,
            unit_y*(0.5*square_size+1.1)+2*subplot_pad-expansion_y,
            square_x+expansion_x*2,
            square_y+expansion_y*2
        ],
    }
    axes['psth_a'] = fig.add_axes(rects['psth_a'])
    axes['psth_b'] = fig.add_axes(rects['psth_b'])
    axes['jpsth'] = fig.add_axes(rects['jpsth'])
    affine = Affine2D().scale(ax_scale_ratio,1).rotate_deg(45)
    axes['coincidence_hist_'] = floating_axes.FloatingAxes(
        fig,
        rect=rects['coincidence_hist'],
        grid_helper=floating_axes.GridHelperCurveLinear(
            affine, extremes=(tmin, tmax, -1, 1)
        )
    )
    axes['coincidence_hist'] = axes['coincidence_hist_'].get_aux_axes(affine)
    affine = Affine2D().scale(ax_scale_ratio,1).rotate_deg(-45)
    axes['xcorr_hist_'] = floating_axes.FloatingAxes(
        fig,
        rect=rects['xcorr_hist'],
        grid_helper=floating_axes.GridHelperCurveLinear(
            affine, extremes=(xcorr_lagmin,xcorr_lagmax,xcorr_min,xcorr_max)
        )
    )
    # axes['xcorr_hist_'].set_aspect(100)
    axes['xcorr_hist'] = axes['xcorr_hist_'].get_aux_axes(affine)
    
    fig.add_axes(axes['coincidence_hist_'])
    # fig.add_axes(axes['coincidence_hist'])
    # fig.add_axes(axes['xcorr_hist'])
    fig.add_axes(axes['xcorr_hist_'])
    # fig.add_axes(axes['xcorr_hist'])
    ## FORMATTING
    axes['coincidence_hist_'].set_xticklabels([])
    axes['coincidence_hist_'].yaxis.set_label_position("right")
    axes['coincidence_hist_'].yaxis.tick_right()
    for side in ['top','bottom','left']:
        axes['coincidence_hist_'].axis[side].set_visible(False)
    axes['coincidence_hist_'].axis["y=0"] = axis = axes['coincidence_hist_'].new_floating_axis(1, 0)
    axis.set_visible(True)
    axes['coincidence_hist_'].set_aspect('auto')
    for side in ['top','right','left']:
        axes['xcorr_hist_'].axis[side].set_visible(False)
    
    axes['jpsth'].get_shared_x_axes().join(axes['jpsth'], axes['psth_b'])
    axes['jpsth'].get_shared_y_axes().join(axes['jpsth'], axes['psth_a'])
    
    for ax in ['jpsth','psth_a','psth_b']: axes[ax].axis('off')

    axes['psth_a'].invert_xaxis()
    axes['psth_b'].invert_yaxis()
    return fig, axes

import simianpy as simi
from simianpy.analysis.spike_train.jpsth import jpsth
import numpy as np
import pandas as pd
paths = {
    'a': r"C:\Users\selja\OneDrive\PhD Neuroscience - Western University\Comprehensive exam\grant\simulate\data\a_fix_neg.npy",
    'b': r"C:\Users\selja\OneDrive\PhD Neuroscience - Western University\Comprehensive exam\grant\simulate\data\b_fix_neg_eye.npy",
}
a = np.load(paths['a'])
b = np.load(paths['b'])
t = np.arange(-.25, .75, 1e-3)[:-1]
jpsth_data = jpsth(t,a,b)
fig,axes=create_jpsth_axes(jpsth_data)
data = pd.DataFrame(jpsth_data['normalized_jpsth'], index=t, columns=t).stack()
data.index.names = 'a', 'b'
data.name = 'jpsth'
simi.plotting.Image(
    'a', 'b', 'jpsth', data,
    ax=axes['jpsth'],
    colorbar = False,
    # colorbar=dict(direction='top', size='5%', pad=0.05),
    # im_kwargs=dict(cmap='jet', vmin=-1, vmax=1)
)
axes['psth_a'].barh(jpsth_data['t'],jpsth_data['unita']['mean'], 1e-3)
axes['psth_b'].bar(jpsth_data['t'], jpsth_data['unitb']['mean'], 1e-3)
axes['coincidence_hist'].bar(jpsth_data['t'], jpsth_data['coincidence_histogram'], 1e-3, color='k')
axes['xcorr_hist'].bar(jpsth_data['lags'], jpsth_data['xcorr_hist']*1e3, 1, color='k')