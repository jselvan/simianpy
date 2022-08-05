"""Signal processing tools

Contains
--------
    Filter - helper class for designing frequency filter
    FFT - helper class for performing a FFT on time series data 
    Smooth - helper class for designing a smoothing function

Modules
-------
    filter
    fft
    smooth
"""
from .convolve import Convolve
from .fft import FFT
from .filter import Filter
from .sosfilter import sosFilter
