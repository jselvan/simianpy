import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

import simianpy as simi

power_bands = {
    "alpha": (15, 22),
    "gamma": (80, 150)
}
def get_multitaper_lfp_power(lfp, depths, sampling_rate, NFFT=None, NW=2.5, k=None, normalize=True):
    """get multitaper lfp power

    Parameters
    ----------
    lfp : array_like (n_trials, n_samples, n_channels)
        lfp data
    depths : array_like (n_channels,)
        channel depths
    NW : float, default 2.5
        time-bandwidth product
    k : int, default None
        number of tapers, if None, k = NW*2
    """
    try:
        use_scipy=False
        from spectrum import dpss
    except ImportError:
        use_scipy=True
        from scipy.signal.windows import dpss

    n_trials, n_samples, n_channels = lfp.shape
    if NFFT=='auto':
        NFFT = max(256, int(2**np.ceil(np.log2(n_samples))))
    elif NFFT==None:
        NFFT = n_samples
    
    if use_scipy:
        if k is None:
            k = int(NW*2)
        tapers = dpss(n_samples, NW, k).T
    else:
        tapers, eigenvalues = dpss(n_samples, NW, k)

    # get multitaper lfp power
    # tapers dimensions: (n_tapers, n_samples)
    # lfp dimensions: (n_trials, n_samples, n_channels)
    # Sk_complex dimensions: (n_tapers, n_trials, freq, n_channels)
    Sk_complex = np.fft.fft(np.expand_dims(tapers.T, (1,3)) * np.expand_dims(lfp, 0), NFFT, axis=2)

    # grab the real part
    Sk = np.abs(Sk_complex)**2
    NFFT_half = int((NFFT-1)/2) if NFFT % 2 else int(NFFT/2)
    psd = Sk[:, :, :NFFT_half, :] * 2
    freq = np.fft.fftfreq(NFFT, 1/sampling_rate)[:NFFT_half]

    # compute the average across tapers and trials
    psd = psd.mean(axis=(0,1))

    # normalize to max power across channels
    if normalize:
        psd /= psd.max(axis=-1, keepdims=True)
    psd_df = pd.DataFrame(psd, index=pd.Index(freq, name='freq'), columns=pd.Index(depths, name='depth')).loc[0:150]
    return psd_df

def plot_multitaper_lfp_power(spectrolaminar, psd_df):
    fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]})

    for band, (left, right) in power_bands.items():
        axes[0].axvline(left, color='k', linestyle='--')
        axes[0].axvline(right, color='k', linestyle=':')
    simi.plotting.Image('freq', 'depth', 0, psd_df.stack(), im_kwargs=dict(aspect='auto', cmap='viridis'), ax=axes[0])

    cmap = {'alpha':'#e41a1c', 'gamma': '#377eb8'}
    for band, data in spectrolaminar.items():
        axes[1].plot(data, spectrolaminar.index, label=band, color=cmap[band])
    for spine in ['top', 'right']:
        axes[1].spines[spine].set_visible(False)
    axes[1].legend()

# FLIP method
def _FLIP(inputdata):
    ssxm, ssxym, _, ssym = np.cov(inputdata.index.values, inputdata.values, bias=1).flat
    return ssxym / ssxm
def FLIP(spectrolaminar):
    """FLIP method

    Parameters
    ----------
    spectrolaminar : pandas.DataFrame
        spectrolaminar data
    """
    warnings.warn("FLIP method has not been tested")
    max_G, depth, window = 0, None, (None, None)
    for window_size in range(5, 20):
        flip = spectrolaminar.rolling(window_size, center=True).aggregate(_FLIP)
        G = - flip.alpha * flip.gamma
        if G.max() > max_G:
            max_G = G.max()
            depth = G.idxmax()
            idx = G.argmax()
            window = (G.index[idx - window_size//2], G.index[idx + window_size//2])
    return depth, max_G, window