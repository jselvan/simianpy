"""Analysis tools

Contains
--------
    BehaviouralData -- class with basic functionality for working with the behavioural data in a file
    DetectSaccades -- function to detect saccades

Modules
-------
    behaviouraldata
    detectsaccades
"""
from .behaviouraldata import BehaviouralData
from .stats.linear_regression import LinearRegression
from .spikedensity import SDF
from .blink_mask import get_blink_mask
from .csd import CSD