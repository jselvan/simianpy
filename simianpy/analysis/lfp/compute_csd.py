import numpy as np


def compute_csd(lfp, depths, five_point=True):
    if five_point:
        numerator = (
            2 * lfp[:-4, :]
            - lfp[1:-3, :]
            - 2 * lfp[2:-2, :]
            - lfp[3:-1, :]
            + 2 * lfp[4:, :]
        )
    else:
        numerator = (
            lfp[-2,:] 
            + lfp[2,:] 
            - 2 * lfp[1:-1]
        )
    h = np.unique(np.diff(depths))
    if h.size > 1:
        raise ValueError("depths must be equally spaced")
    else:
        h = h[0]
    denominator = 7 * h ** 2
    csd = numerator / denominator
    return csd, depths[2:-2]
