import warnings
import numpy as np

def bin_estimation(data, nbins_bounds=(4,50), range=None):
    """ Estimate the optimal number of bins for PSTH of neural data

    Parameters
    ----------
    data
    nbins_bounds
    range

    Returns
    -------
    bin_edges

    References
    ----------
    [1] Shimazaki, H., & Shinomoto, S. (2007). A method for selecting the bin size of a time histogram. Neural computation, 19(6), 1503-1527.
    """#TODO: complete docstring
    if len(data) == 0:
        return []
    nbins = np.arange(*nbins_bounds)
    if range is None:
        range = min(data), max(data)
    else:
        warnings.warn("Bin estimation is not likely to be accurate if a range is forced")
    cost = np.zeros(nbins.size)

    for idx, n in enumerate(nbins):
        counts, _ = np.histogram(data, bins=n, range=range)
        cost[idx] = (2*counts.mean()-counts.var())/(((range[1]-range[0])/n)**2)
    
    optimal_nbins = nbins[cost.argmin()]
    return np.linspace(*range,optimal_nbins+1)

class PSTH:
    supported_output_units = 'rate',
    def __init__(self, sampling_rate=3e4, bins='optimal', range=None, output_units='rate', nbins_bounds=(4,50)):
        self._nbins_bounds = nbins_bounds #used for bin estimation
        self.sampling_rate = sampling_rate

        if output_units in self.supported_output_units:
            self.output_units = output_units
        else:
            raise ValueError(f"Provided output units ({output_units}) is not one of supported options: {self.supported_output_units}")
        
        self.range = range
        if bins == 'optimal':
            self._bins = 'optimal'
        elif np.isscalar(bins):
            if range is None:
                raise TypeError("Must provide a range if bins is scalar")
            self._bins = np.linspace(*range, bins+1)
        else:
            self._bins = np.asarray(bins)
    
    def _get_x(self, bins):
        return np.mean([bins[:-1], bins[1:]], axis=0)
    
    def _get_bins(self, data):
        if self._bins == 'optimal':
            return bin_estimation(data, self._nbins_bounds, self.range)
        else:
            return self._bins
    
    def compute(self, data, nrows=None):
        if not all(np.isscalar(val) for val in data):
            data = np.concatenate(data)
        else:
            data = np.asarray(data)
        if nrows is None:
            nrows = len(data)
        bins = self._get_bins(data)
        timepoints = self._get_x(bins)
        counts, bins = np.histogram(data, bins=bins)
        if self.output_units == 'rate':
            counts = counts / (1e3*np.diff(bins)/self.sampling_rate) / nrows
        return timepoints, counts