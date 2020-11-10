from simianpy.signal import Filter

import numpy as np
import scipy.signal

def DetectSpikes(data, threshold = -20, negative = True):
    """Detect spikes in a vector

    Parameters
    ----------
    data: pd.Series
        continuous neural data
    threshold: float
        minimum height of spike peaks
    negative: bool; default = True
        if True, find negative peaks deflecting below the threshold values

    Returns
    -------
    spk: pd.DataFrame

    Examples
    --------
    lfp_raw = simi.io.openephys.load('data.continuous')
    Filter = simi.signal.Filter('bandpass',6,[300,6000],3e4)
    filt_lfp = Filter(lfp_raw)
    spk = DetectSpikes(filt_lfp, -20, True)

    ax.plot(lfp_raw)
    ax.plot(spk.index, lfp_raw.loc[spk.index], 'x')
    """
    #TODO implement spike detection
    scipy.signal.find_peaks( -data if negative else data, height = -threshold if negative else threshold)