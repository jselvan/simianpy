import warnings
import simianpy.signal
import simianpy.misc

import operator

import numpy as np
import pandas as pd
import scipy.interpolate

#TODO: Implement optimal window width using: https://github.com/cooperlab/AdaptiveKDE/tree/master/adaptivekde
#TODO: support cupy use even when not using globally optimal bandwidth
#TODO: add shinomoto refs to docstring
class SDF:
    """ A helper class for computing spike density functions

    Parameters
    ----------
    sampling_rate: numeric
        The sampling rate of desired spike density function in Hz
    convolve: str, callable or None, default=None
        If 'optimal', a globally optimal bandwidth is used to convolve a gaussian
        NOTE: 'optimal'  has some weird behaviour, further testing required
        If callable, the provided object is called on the binarized data
        If None, `window` and\or `window_size` must be provided
    window, window_size: default=None
        ignored if `convolve` is not None, else required
        `window_size` is inferred from `window` if not provided
        if 'psp', use a psp window [1]
        see simi.signal.Convolve for more info
    output_units: str, default='rate'
        Only 'rate' is currently supported
    timestamps: array-like or None, default=None
        if None, inferred from `time_range` and `sampling_rate`
    time_range: array-like (len==2) or None, default=None
        ignored if `timestamps` is None, else required
        must be a two-tuple of start and end of timestamps
        self.timestamps is then defined as: arange(*time_range, 1e3/sampling_rate)
    use_gpu: bool, default=False
        Uses GPU acceleration when called. Requires cupy and CUDA support
        Currently only supported when `convolve` is 'optimal'
    
    Examples
    --------
    The easiest way to use this class:

    initialize the class with desired properties. Here the spike density will be 
    computed at 100Hz using a 1s wide gaussian window over a 60 minute range
    >>> import simianpy as simi
    >>> SDF=simi.analysis.SDF(100, window='gaussian', window_size=1e3, time_range=(0,60*60*1e3))
    
    Collect all your spikes in a nested list. Here each sublist has the spike times
    for a single unit in milliseconds.
    >>> units, spike_list = [], []
    >>> for unit, spikes in spk.groupby('unitid'):
    >>>     units.append(unit)
    >>>     spike_list.append(spikes.values)

    Using the compute_all method, compute the spike density function and store as
    a pandas dataframe.
    >>> sdf=SDF.compute_all(spike_list)
    >>> sdf=pd.DataFrame(sdf, index=pd.Index(units,name='unitid'), columns=SDF.timestamps)

    Methods
    -------
    SDF.parse_single_trial_binary(data)
        Input data must be counts of spikes in bins corresponding to SDF.hist_bins
    SDF.parse_single_trial_timestamps(data)
        Computes bin counts and evaluates using SDF.parse_single_trial_binary
    SDF.compute(data, on=None, variance=True, input='timestamps')
        Computes mean and error terms for a spike density function across multiple trials
        Loops through using the appropriate parse_single_trial_* function as specified by 'input'
        Returns pd.DataFrame(index=SDF.timestamps,columns=['mean','variance','se'] if variance else ['mean'])
    SDF.compute_all(data, input='timestamps')
        

    References
    ----------
    [1] http://www.psy.vanderbilt.edu/faculty/schall/scientific-tools/
    """
    supported_output_units = 'rate',

    def __init__(self, sampling_rate, convolve=None, window=None, window_size=None, output_units='rate', timestamps=None, time_range=None, use_gpu=False):
        self.xp, self.use_gpu = simianpy.misc.get_xp(use_gpu)

        self.sampling_rate = sampling_rate

        if convolve is not None:
            self.convolve = convolve
        elif window is not None:
            if window == 'psp':
                growth, decay = 1, 20 # in ms
                length = 4*decay # should be close to 0 at this point
                window = np.array([(1-np.exp(-(i+1)/growth))*np.exp(-(i+1)/decay) for i in range(length)])
                if sampling_rate != 1e3:
                    window = scipy.interpolate.interp1d(np.arange(0,length),window)(np.arange(0,length-1,1e3/sampling_rate)) #upsample to the sampling rate
                window = np.concatenate((np.zeros(window.size-1),window))
                window /= window.sum()
            if window_size is None:
                window_size = len(window)
            else:
                window_size = window_size * (1e3/self.sampling_rate)
            self.convolve = simianpy.signal.Convolve(size=window_size,window=window,use_gpu=self.use_gpu)
        else:
            raise TypeError("Missing required argument: Must provide convolve or window")
        
        if output_units in self.supported_output_units:
            self.output_units = output_units
        else:
            raise ValueError(f"Provided output units ({output_units}) is not one of supported options: {self.supported_output_units}")

        if timestamps is not None:
            self.timestamps = timestamps
        elif time_range is not None:
            self.timestamps = self.xp.arange(*time_range, 1e3/sampling_rate)
        half_dt = 1e3/2/self.sampling_rate
        self.hist_bins = (self.timestamps - half_dt).tolist() + [self.timestamps[-1]+half_dt]
    
    # Global optimization functions from: https://github.com/cooperlab/AdaptiveKDE/tree/master/adaptivekde
    def _logexp(self, x):
        if x<1e2:
            return self.xp.log(1+self.xp.exp(x))
        else:
            return x
    
    def _ilogexp(self, x):
        if x<1e2:
            return self.xp.log(self.xp.exp(x)-1)
        else:
            return x
    
    def _global_optimization_cost_function(self, y_hist,N,w,dt):
        yh = self._fftkernel(y_hist, w / dt)
        C = self.xp.sum(yh**2) * dt - 2 * self.xp.sum(yh * y_hist) * dt + 2 / (2 * self.xp.pi)**0.5 / w / N
        C = C * N**2
        return C, yh
    
    def _fftkernel(self, x, w):
        L = x.size
        Lmax = L + 3 * w
        n = int(2 ** self.xp.ceil(self.xp.log2(Lmax)))
        X = self.xp.fft.fft(x, n)
        f = self.xp.linspace(0, n-1, n) / n
        f = self.xp.concatenate((-f[0: self.xp.int(n / 2 + 1)],
                            f[1: self.xp.int(n / 2 - 1 + 1)][::-1]))
        K = self.xp.exp(-0.5 * (w * 2 * self.xp.pi * f) ** 2)
        y = self.xp.real(self.xp.fft.ifft(X * K, n))
        y = y[0:L]
        return y

    def _globally_optimized_bandwidth(self, data):
        dt = 1/self.sampling_rate
        N = data.sum()
        y_hist = data / N / dt
        
        k = 0

        Wmin = 2*dt
        Wmax = (self.xp.max(data) - self.xp.min(data))
        tol = 10e-5
        phi = (5**0.5 + 1) / 2
        a = self._ilogexp(Wmin)
        b = self._ilogexp(Wmax)
        c1 = (phi - 1) * a + (2 - phi) * b
        c2 = (2 - phi) * a + (phi - 1) * b
        f1, _ = self._global_optimization_cost_function(y_hist, N, self._logexp(c1), dt)
        f2, _ = self._global_optimization_cost_function(y_hist, N, self._logexp(c2), dt)
        while (self.xp.abs(b-a) > tol * (self.xp.abs(c1) + self.xp.abs(c2))) & (k < 20):
            if f1 < f2:
                b = c2
                c2 = c1
                c1 = (phi - 1) * a + (2 - phi) * b
                f2 = f1
                f1, yh1 = self._global_optimization_cost_function(y_hist, N, self._logexp(c1), dt)
                y = yh1 / self.xp.sum(yh1 * dt)
            else:
                a = c1
                c1 = c2
                c2 = (2 - phi) * a + (phi - 1) * b
                f1 = f2
                f2, yh2 = self._global_optimization_cost_function(y_hist, N, self._logexp(c2), dt)
                y = yh2 / self.xp.sum(yh2 * dt)
            k += 1
        y /= self.sampling_rate
        return y

    def _binarize(self, data):
        binarized, _ = np.histogram(data, self.hist_bins)
        return binarized
    
    def parse_single_trial_binary(self, data):
        if self.use_gpu:
            data = self.xp.array(data)
        if data.any():
            if self.convolve == 'optimal':
                sdf = self._globally_optimized_bandwidth(data)
            else:
                sdf = self.convolve(data)
        else:
            sdf = self.xp.zeros_like(data)
        
        if self.output_units == 'rate':
            sdf *= self.sampling_rate

        return sdf

    def parse_single_trial_timestamps(self, data):
        if self.use_gpu:
            data = self.xp.array(data)
        data = self._binarize(data)
        sdf = self.parse_single_trial_binary(data)
        return sdf

    def compute(self, data, on=None, variance=True, input='timestamps'):
        if on is not None:
            data = map(operator.itemgetter(1), data.groupby(on))
        n = 0        
        x_sum = self.xp.zeros_like(self.timestamps)
        if variance:
            x_squared_sum = self.xp.zeros_like(self.timestamps)

        for trial in data:
            if input=='timestamps':
                sdf = self.parse_single_trial_timestamps(trial)
            elif input=='binary':
                sdf = self.parse_single_trial_binary(trial)
            x_sum += sdf
            if variance:
                x_squared_sum += (sdf**2)
            n += 1

        if self.use_gpu:
            timestamps, x_sum = self.timestamps.get(), x_sum.get()
            if variance:
                x_squared_sum = x_squared_sum.get()
        else:
            timestamps = self.timestamps
        sdf = pd.DataFrame(index=timestamps)
        sdf['mean'] = x_sum / n
        if variance:
            sdf['variance'] = (x_squared_sum + (x_sum**2)/n)/n #TODO: check if variance formula is correct
            sdf['se'] = (sdf['variance']/n)**.5
        return sdf

    def compute_all(self, data, input='timestamps'):
        if input=='timestamps':
            data = self.xp.array([self._binarize(trial) for trial in data])
        elif input=='binary':
            data = self.xp.array(data)
        if self.convolve == 'optimal':
            sdf = [self.parse_single_trial_binary(trial) for trial in data]
        else:
            sdf = self.convolve(data, axis=0, pad=False)
        if self.output_units == 'rate':
            sdf *= self.sampling_rate
        return sdf