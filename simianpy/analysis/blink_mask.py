from ..misc import binary_digitize

import numpy as np

def get_blink_mask(eyedata, pad=50, blink_threshold=None):
    """ get a blink mask from eyedata
    pad is in samples in both directions
    """
    #TODO: complete docstring
    if blink_threshold is None:
        blink_threshold = eyedata.max() - 1
    
    blink_start, blink_end = binary_digitize((eyedata>=blink_threshold).any(axis=1))
    
    blink_mask = np.ones(eyedata.shape, dtype=bool)
    blink_mask_length, _ = blink_mask.shape
    for start, end in zip(blink_start, blink_end):
        blink_mask[start-pad if start > pad else 0:end+pad if end < (blink_mask_length-pad) else blink_mask_length, :] = False
    return blink_mask