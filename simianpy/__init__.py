"""
SimianPy
========
  
Metadata
--------
Author: Janahan Selvanayagam  
Email: seljanahan@hotmail.com  
  
How to use
----------
The documentation assumes simianpy as simi  

>>> import simianpy as simi  


There are various modules within simianpy to be used:  

    io - input/output tools  
    signal - signal processsing tools  
    analysis - analysis tools  
    misc - miscellaneous tools  
  
Read the docs (I have not generated this yet!) or use an interactive shell & docstrings to get more info  
"""
from . import io, misc, signal, analysis
from ._version import version

__version__ = version
