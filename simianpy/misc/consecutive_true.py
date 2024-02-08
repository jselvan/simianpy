from skimage.util.shape import view_as_windows
import numpy as np

def consecutive_true(boolarray, n, axis=-1):
    window = np.ones(boolarray.ndim)
    window[axis] = n
    return ((view_as_windows(boolarray, window))
            .squeeze().all(axis=-1).argmax(axis=axis))