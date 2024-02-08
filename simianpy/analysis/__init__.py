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
from .detectsaccades import DetectSaccades
from .detectfixations import DetectFixations
from .stats.linear_regression import LinearRegression
from .spikedensity import SDF
from .blink_mask import get_blink_mask
from .gaussian_kde_2d import GaussianKDE2D
from .csd import CSD