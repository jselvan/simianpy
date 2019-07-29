import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import pandas as pd

class Smooth():
    """Class to design smoothing functions

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
    smooth = Smooth(size = 10, window = 'boxcar')

    # to visualize the window:
    smooth.plot_window()
    plt.show()

    # to smooth the data
    smoothed_data = smooth(raw_data, pad = False)
    """
    def __str__(self):
        return f"Smooth - window type: '{self.window.name}'; window size {self.size}."

    def __init__(self, size = 10, window = 'boxcar'):
        self.size = size
        self.window = window
    
    @property
    def size(self):
        """ width of the smooth window will be (2 * self.size + 1)"""
        return self._size
    @size.setter
    def size(self, size):
        assert isinstance(size, int), TypeError(f'size must be int not {type(size)}')
        self._size = size
    
    @property
    def window(self):
        """ can be set to any valid input to scipy.signal.get_window - returns kernel as pd.Series """
        return self._window
    @window.setter
    def window(self, window):
        x = np.arange(-self.size, self.size+1)
        if window == 'gaussian':
            kernel = np.exp( -( 2*x / self.size )**2 )
        else:
            kernel = scipy.signal.get_window(window, 2*self.size)
            kernel = np.concatenate([kernel, kernel[:1]])
        kernel /= kernel.sum()
        self._window = pd.Series(kernel, index = x, name = str(window))
    
    @staticmethod
    def _smooth_edge(edge_data):
        return np.array([np.mean(edge_data[:(idx+1)]) for idx in range(edge_data.size)])

    def __call__(self, data, pad = True):
        if pad:
            left = self._smooth_edge( data[:self.size] )
            right = self._smooth_edge(data[:-(self.size+1):-1])[::-1]
            return np.concatenate( [left, np.convolve(data, self.window, 'valid'), right] )
        else:
            return np.convolve(data, self.window, 'same')
    
    def plot_window(self, ax=None):
        if ax is None:
            ax = plt.gca()
        self.window.plot(ax=ax)