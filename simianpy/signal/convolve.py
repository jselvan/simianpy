import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

import simianpy.misc
from simianpy.plotting.util import get_ax


class Convolve():
    """Class to design a function for a linear convolution

    Parameters
    ----------
    size: int, optional, default: 10
    window: str, optional, default: 'boxcar'

    Attributes
    ----------
    size
    window

    Methods
    -------
    plot_window(ax=None)
        plot the window being used

    Examples
    --------
    convolve = Convolve(size=11, window='boxcar')

    # to visualize the window:
    convolve.plot_window()
    plt.show()

    # to convolve the data
    convolved_data = convolve(raw_data, pad=False)

    Notes
    -----
    The size of the window should usually be an odd number
    """
    def __str__(self):
        return f"Convolve - window type: '{self.window.name}'; window size {self.size}."

    def __init__(self, size=11, window='boxcar',use_gpu=False):
        self.size = int(size) #TODO: check first or always cast to int?
        self.window = window
        self.xp, self.use_gpu = simianpy.misc.get_xp(use_gpu)
        #TODO: implement being able to use the scipy versions instead?
        if self.use_gpu:
            try:
                import cupyx.scipy.signal
            except ImportError:
                self.conv = scipy.signal.convolve
            else:
                self.conv = cupyx.scipy.signal.convolve
        else:
            self.conv = scipy.signal.convolve

    
    @property
    def window(self):
        """ can be set to any valid input to scipy.signal.get_window, 'gaussian' or custom kernel - returns kernel as pd.Series """
        return self._window
    @window.setter
    def window(self, window):
        size_half = self.size//2
        x = np.arange(-size_half, size_half+1 if self.size%2 else size_half)
        if np.isscalar(window) and window == 'gaussian':
            kernel = np.exp( -( 2*x / size_half )**2 )
        elif isinstance(window, str):
            kernel = scipy.signal.get_window(window, self.size-1)
            kernel = np.concatenate([kernel, kernel[:1]])
        else:
            window = np.asarray(window)
            assert window.size == self.size, f"Provided window: {window} does not match provided size: {self.size}"
            self._window = pd.Series(window, index=x, name='custom')
            return
        kernel /= kernel.sum()
        self._window = pd.Series(kernel, index=x, name=str(window))
    
    def _smooth_edge(self, edge_data):
        return self.xp.array([self.xp.mean(edge_data[:(idx+1)]) for idx in range(edge_data.size)])

    def __call__(self, data, axis=0):
        if self.use_gpu:
            data = self.xp.array(data)
            window = self.xp.array(self.window.values)
        else:
            window = self.window.values
        
        if data.ndim == 1:
            pass
        elif data.ndim == 2:
            window = self.xp.expand_dims(window, axis)
        return self.conv(data, window, 'same')
    
    def plot_window(self, ax=None):
        ax = get_ax(ax)
        self.window.plot(ax=ax)
