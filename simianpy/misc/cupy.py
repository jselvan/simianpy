import warnings

import numpy as np 

def get_xp(use_gpu):
    if use_gpu:
        try:
            import cupy as cp
        except ImportError:
            warnings.warn('Failed to import cupy, using CPU')
            xp = np
            use_gpu = False
        else:
            xp = cp
    else:
        xp = np
    return xp, use_gpu