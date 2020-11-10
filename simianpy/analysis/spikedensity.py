import warnings
import simianpy.signal

import operator

import numpy as np
import pandas as pd

#TODO: Implement optimal window width using: https://github.com/cooperlab/AdaptiveKDE/tree/master/adaptivekde

class SDF:
    supported_output_units = 'rate',

    def __init__(self, sampling_rate, convolve=None, window=None, window_size=None, output_units='rate', timestamps=None, time_range=None):
        self.sampling_rate = sampling_rate

        if convolve is not None:
            self.convolve = convolve
        elif window is not None:
            if window_size is None:
                window_size = len(window)
            self.convolve = simianpy.signal.Convolve(size=window_size,window=window)
        else:
            raise TypeError("Missing required argument: Must provide convolve or window")
        
        if output_units in self.supported_output_units:
            self.output_units = output_units
        else:
            raise ValueError(f"Provided output units ({output_units}) is not one of supported options: {self.supported_output_units}")

        if timestamps is not None:
            self.timestamps = timestamps
        elif time_range is not None:
            self.timestamps = np.arange(*time_range, 1e3/sampling_rate)
    
    def _estimate_globally_optimized_bandwidth(self, data, bandwidths=None):
        dt = 1/self.sampling_rate
        # if bandwidths is not None:
        #     for bandwidth in bandwidths:



    def _binarize(self, data):
        binarized = np.zeros_like(self.timestamps)
        idx = ((data - self.timestamps.min()) * self.sampling_rate / 1000).astype(int)
        above_bounds = (idx >= binarized.size)
        if above_bounds.any():
            warnings.warn('Some timestamps were not in range')
            idx[above_bounds] = binarized.size-1
        below_bounds = (idx < 0)
        if below_bounds.any():
            warnings.warn('Some timestamps were not in range')
            idx[below_bounds] = 0
        binarized[idx] = 1
        return binarized
    
    def parse_single_trial_binary(self, data):
        if data.any():
            sdf = self.convolve(data)
        else:
            sdf = np.zeros_like(data)
        
        if self.output_units == 'rate':
            sdf *= self.sampling_rate

        return sdf

    def parse_single_trial_timestamps(self, data):
        data = self._binarize(data)
        sdf = self.parse_single_trial_binary(data)
        return sdf

    def compute(self, data, on=None, variance=True, input='timestamps'):
        if on is not None:
            data = map(operator.itemgetter(1), data.groupby(on))

        n = 0        
        x_sum = np.zeros_like(self.timestamps)
        if variance:
            x_squared_sum = np.zeros_like(self.timestamps)

        for trial in data:
            if input=='timestamps':
                sdf = self.parse_single_trial_timestamps(trial)
            elif input=='binary':
                sdf = self.parse_single_trial_binary(trial)
            x_sum += sdf
            if variance:
                x_squared_sum += (sdf**2)
            n += 1
        
        sdf = pd.DataFrame(index=self.timestamps)
        sdf['mean'] = x_sum / n
        if variance:
            sdf['variance'] = (x_squared_sum + (x_sum**2)/n)/n
        
        return sdf
