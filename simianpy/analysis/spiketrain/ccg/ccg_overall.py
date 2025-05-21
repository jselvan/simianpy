from itertools import combinations

import cupy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm

from simianpy.misc.cupy.view_as_windows import view_as_windows


def compute_ccg_overall(
    spike_times, spike_units, normalize="geom_mean", include_acg=True, lag=50
):
    # determine bins, lags and units
    lags = np.arange(-lag, lag + 1)
    bins = cp.arange(spike_times.min(), spike_times.max())
    units = np.unique(spike_units)
    n_units, n_bins = units.size, bins.size - 1
    unitidx = np.arange(n_units)

    unita, unitb = np.array(list(combinations(unitidx, 2))).T
    if include_acg:
        unita = np.concatenate([unita, unitidx])
        unitb = np.concatenate([unitb, unitidx])
    # compute rasters
    spike_times = cp.asarray(spike_times)
    raster = cp.zeros((n_units, n_bins), dtype=bool)
    for idx, unit in enumerate(units):
        unit_spikes = spike_times[spike_units == unit]
        raster[idx, :] = cp.histogram(unit_spikes, bins)[0].astype(bool)

    window_size = n_bins - 2 * lag
    ccg_all = cp.zeros((unita.size, lags.size))
    for idx, (unita_, unitb_) in tqdm(enumerate(zip(unita, unitb)), total=unita.size):
        raster_a = cp.expand_dims(raster[unita_, lag:-lag], 0)
        raster_b = view_as_windows(raster[unitb_, :], window_size)
        ccg_all[idx, :] = ccg = (raster_a * raster_b).sum(1)
        if normalize == "geom_mean":
            fr_geom_mean = cp.sqrt(cp.sum(raster_a, 1) * cp.sum(raster_b, 1))
            ccg = ccg / fr_geom_mean

    # construct dataframe and return
    mi = pd.MultiIndex.from_arrays(
        [units[unita], units[unitb]], names=["unita", "unitb"]
    )
    lags = pd.Index(lags, name="lag")
    ccg = pd.DataFrame(ccg_all.get(), index=mi, columns=lags)
    return ccg
