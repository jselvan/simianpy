import warnings

import numpy as np


def normalize_jpsth(raw_jpsth, unita_stats, unitb_stats):
    psth_outer_product = np.outer(unita_stats["mean"], unitb_stats["mean"])
    normalizer = np.outer(unita_stats["std"], unitb_stats["std"])
    unnormalized_jpsth = raw_jpsth - psth_outer_product

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normalized_jpsth = unnormalized_jpsth / normalizer
    normalized_jpsth[np.isnan(normalized_jpsth)] = 0
    return normalized_jpsth
