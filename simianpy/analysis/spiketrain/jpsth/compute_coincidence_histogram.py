import numpy as np


def compute_coincidence_histogram(jpsth_data, coincidence_widths):
    pstch = np.zeros(jpsth_data.shape[0])
    for width in coincidence_widths:
        pstch[np.abs(width) :] += np.diag(jpsth_data, width)
    return pstch
