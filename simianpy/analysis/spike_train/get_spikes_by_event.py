import numpy as np
import pandas as pd

def get_spikes_by_event(spike_timestamps, event_timestamp, pad):
    return (
        spike_timestamps[
            ((event_timestamp + pad[0]) < spike_timestamps)
            & ((event_timestamp + pad[1]) > spike_timestamps)
        ]
        - event_timestamp
    )

def get_spikes_by_event_array(spike_timestamps, event_timestamps, window, spike_labels=None, event_labels=None):
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

    return event_labels, spike_labels_aligned, spike_times_aligned

def get_spike_by_event_series(spike_timestamps, event_timestamps, window):
    spike_timestamps = spike_timestamps.sort_values()
    
    left, right = event_timestamps+window[0], event_timestamps+window[1]
    lidx = np.searchsorted(spike_timestamps, left, side='left')
    ridx = np.searchsorted(spike_timestamps, right, side='right')
    lengths = ridx - lidx
    offset = np.repeat(event_timestamps, lengths)
    event_labels = np.repeat(event_timestamps.index, lengths)
    spk_idx = np.concatenate([np.arange(l, r) for l, r in zip(lidx, ridx)])
    spike_times_aligned = spike_timestamps.iloc[spk_idx] - offset
    spike_times_aligned.index = pd.MultiIndex.from_arrays([event_labels, spike_times_aligned.index])
    return spike_times_aligned

