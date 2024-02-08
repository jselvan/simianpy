import numpy as np
import pandas as pd


def psth_factory(bins):
    bin_left = pd.Index(bins[:-1], name='time')

    def psth(spikes):
        counts, _ = np.histogram(spikes, bins=bins)
        return pd.Series(counts, index=bin_left, name="spikes")

    return psth