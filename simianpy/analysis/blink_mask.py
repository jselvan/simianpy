import numpy as np

from ..misc import binary_digitize


def get_blink_mask(eyedata, pad=50, blink_threshold=None):
    """ Get a binary mask of blinks from eye data.

    Parameters
    ----------
    eyedata : ndarray
        Eye data.
    pad : int
        Number of samples to pad the beginning and end of blink events.
    blink_threshold : float
        Threshold for detecting blinks.
        If None, the blink threshold is set eyedata.max()-1.
    
    Returns
    -------
    blink_mask : ndarray
        Binary mask of blinks.
    """
    if blink_threshold is None:
        blink_threshold = eyedata.max() - 1

    blink_start, blink_end = binary_digitize((eyedata >= blink_threshold).any(axis=1))

    blink_mask = np.ones(eyedata.shape, dtype=bool)
    blink_mask_length, _ = blink_mask.shape
    for start, end in zip(blink_start, blink_end):
        blink_mask[
            start - pad
            if start > pad
            else 0 : end + pad
            if end < (blink_mask_length - pad)
            else blink_mask_length,
            :,
        ] = False
    return blink_mask
