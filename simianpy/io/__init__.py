"""Input/Output tools

Tools
-----
    Nex -- class for working with Neuroexplorer files
    OpenEphys -- class for working with OpenEphys files
    RHS -- class for working with Intan RHS files
    ephys2nex -- convert OpenEphys files to Neuroexplorer (.nex) files


Modules
-------
    File -- baseclass for file io
    nex -- io for Neuroexplorer file format ('.nex', '.nex5')
    openephys -- io for OpenEphys file format ('.continuous', '.spikes', '.events')
    intan -- io for intan file formats ('.rhs')
    convert -- functions for converting between file formats
"""
from .convert import ephys2nex
from .intan import RHS
from .nex import Nex
from .openephys import OpenEphys
from .raw import load_raw
from .trodes import Trodes
