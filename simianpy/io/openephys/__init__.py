"""OpenEphys ('.continuous', '.spikes', '.events') interface

Recommended use:
>>> OE = simi.io.OpenEphys(...)
>>> OE.open()

You may use the lower level interface:
>>> data = simi.io.openephys.load(...)

OR

>>> spkdata = simi.io.openephys.openephys.loadSpikes(...)
"""
from .openephys import load
from .io import OpenEphys