import numpy as np


def compute_psth_stats(psth):
    return {
        "mean": np.mean(psth, axis=0),
        "std": np.std(psth, axis=0, ddof=1),
        "var": np.var(psth, axis=0, ddof=1),
    }
