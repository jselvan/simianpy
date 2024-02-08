import numpy as np


def compute_crosscorr(jpsth_data, lags):
    numerator = np.array([np.sum(np.diag(jpsth_data, lag)) for lag in lags])
    denominator = jpsth_data.shape[0] - np.abs(lags)
    return numerator / denominator
