import numpy as np
import pandas as pd

from simianpy.analysis.spike_train.psth_factory import psth_factory

class SpikeTrainSet:
    def __init__(self, trialids, unitids, spike_times):
        self.trialids = trialids
        self.unitids = unitids
        self.spike_times = spike_times
        # ensure that they are all the same length
        if not (len(trialids) == len(unitids) == len(spike_times)):
            raise ValueError("All inputs must be the same length")

    @classmethod
    def from_arrays(cls, spike_timestamps, event_timestamps, window, spike_labels=None, event_labels=None):
        if spike_labels is None:
            spike_labels = np.zeros(len(spike_timestamps), dtype=np.int32)
        if event_labels is None:
            event_labels = np.arange(len(event_timestamps), dtype=np.int32)
        spike_timestamps = np.array(spike_timestamps)
        event_timestamps = np.array(event_timestamps)
        spike_labels = np.array(spike_labels)
        event_labels = np.array(event_labels)

        spk_sort_idx = np.argsort(spike_timestamps)
        evt_sort_idx = np.argsort(event_timestamps)
        spike_timestamps = spike_timestamps[spk_sort_idx]
        spike_labels = spike_labels[spk_sort_idx]
        event_timestamps = event_timestamps[evt_sort_idx]
        event_labels = event_labels[evt_sort_idx]

        left, right = event_timestamps+window[0], event_timestamps+window[1]
        lidx = np.searchsorted(spike_timestamps, left, side='left')
        ridx = np.searchsorted(spike_timestamps, right, side='right')
        lengths = ridx - lidx
        offset = np.repeat(event_timestamps, lengths)
        event_labels = np.repeat(event_labels, lengths)
        spk_idx = np.concatenate([np.arange(l, r) for l, r in zip(lidx, ridx)])
        spike_times_aligned = spike_timestamps[spk_idx] - offset
        spike_labels_aligned = spike_labels[spk_idx]

        return cls(event_labels, spike_labels_aligned, spike_times_aligned)

    @classmethod
    def from_series(cls, spike_timestamps, event_timestamps, window):
        spike_labels = spike_timestamps.index
        spike_timestamps = spike_timestamps.values
        event_labels = event_timestamps.index
        event_timestamps = event_timestamps.values
        return cls.from_arrays(spike_timestamps, event_timestamps, window, spike_labels, event_labels)
    
    def to_dataframe(self):
        return pd.DataFrame({
            'trialid': self.trialids,
            'unitid': self.unitids,
            'spike_time': self.spike_times
        })
    
    def to_psth(self, bins):
        psth_fun = psth_factory(bins)
        psth = self.to_dataframe().groupby(['trialid', 'unitid'])['spike_time'].apply(psth_fun)
        return psth.to_xarray().fillna(0)