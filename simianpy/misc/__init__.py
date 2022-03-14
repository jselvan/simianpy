"""Miscellaneous tools

Contains
--------
    getLogger -- function that returns a logger object

Modules
-------
    logging
"""
from .binary_digitize import binary_digitize
from .cupy import get_xp
from .cut import cut
from .logging import add_logging, getLogger
